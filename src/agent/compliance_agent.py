"""
Core compliance assessment agent.

This is the central orchestrator of the system. It implements a transparent,
three-phase pipeline that is fully logged and auditable:

    Phase 1: RETRIEVE — Find relevant Sharia standard chunks
    Phase 2: REASON  — Present context to LLM with structured prompt
    Phase 3: EXTRACT — Parse LLM response into a ComplianceAssessment

Design decision — explicit pipeline vs. agent framework:

We intentionally avoid LangChain's AgentExecutor, LlamaIndex's QueryEngine,
or similar abstractions. While these save boilerplate, they:
1. Hide the prompt construction logic (critical for compliance auditing)
2. Make debugging harder (can't inspect the exact prompt sent to the LLM)
3. Add unnecessary abstraction layers for a single-tool pipeline
4. Make it harder to add domain-specific error handling

Our pipeline is ~120 lines of explicit Python. Every step is logged with
the request's trace ID. An auditor can reconstruct exactly what happened
for any given assessment by reading the logs.
"""

import json
import time
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.observability.logger import get_logger
from src.observability.tracer import get_current_trace_id
from src.rag.retriever import Retriever, RetrievedChunk
from src.agent.prompts import SYSTEM_PROMPT, build_assessment_prompt
from src.agent.verdict import (
    ComplianceAssessment,
    SourceReference,
    Verdict,
)

logger = get_logger(__name__)


class ComplianceAgent:
    """
    Orchestrates Sharia compliance assessment using RAG + LLM.

    The agent follows an explicit, auditable three-phase pipeline:
    1. RETRIEVE — Find relevant Sharia standards from the vector store
    2. REASON  — Present retrieved context to the LLM with a structured prompt
    3. EXTRACT — Parse the LLM's response into a structured verdict

    Each phase is logged with the request's trace ID for full observability.
    """

    def __init__(
        self,
        retriever: Retriever,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._retriever = retriever
        self._model = model or settings.llm_model
        resolved_base_url = base_url or settings.effective_llm_base_url
        resolved_api_key = api_key or settings.effective_llm_api_key or "none"
        self._client = OpenAI(
            base_url=resolved_base_url if resolved_base_url else None,
            api_key=resolved_api_key,
        )

    async def assess(self, query: str) -> ComplianceAssessment:
        """
        Run a full compliance assessment for the given query.

        This is the main entry point called by the API layer.

        Args:
            query: Natural language compliance question.

        Returns:
            ComplianceAssessment with verdict, reasoning, and citations.
        """
        trace_id = get_current_trace_id()
        start_time = time.monotonic()

        logger.info(
            "assessment_started",
            query=query[:200],
            model=self._model,
        )

        # ── Phase 1: RETRIEVE ────────────────────────────────
        retrieved_chunks = self._retrieve(query)

        # ── Phase 2: REASON ──────────────────────────────────
        llm_response = self._reason(query, retrieved_chunks)

        # ── Phase 3: EXTRACT ─────────────────────────────────
        assessment = self._extract_verdict(
            query=query,
            llm_response=llm_response,
            retrieved_chunks=retrieved_chunks,
            trace_id=trace_id,
        )

        elapsed_ms = round((time.monotonic() - start_time) * 1000)

        logger.info(
            "assessment_completed",
            verdict=assessment.verdict.value,
            confidence=assessment.confidence,
            sources_cited=len(assessment.sources),
            latency_ms=elapsed_ms,
        )

        return assessment

    def _retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Phase 1: Retrieve relevant document chunks.

        Logs the number of chunks found and their sources for
        traceability.
        """
        chunks = self._retriever.retrieve(query)

        logger.info(
            "retrieval_phase_complete",
            chunks_retrieved=len(chunks),
            sources=[
                {"source": c.source, "section": c.section, "score": c.similarity_score}
                for c in chunks
            ],
        )

        return chunks

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        reraise=True,
    )
    def _reason(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> str:
        """
        Phase 2: Send context + query to the LLM for reasoning.

        The full prompt sent to the LLM is logged for auditability.
        Retries up to 3 times with exponential backoff on transient errors.
        """
        # Build the context strings from retrieved chunks
        context_strings = [c.to_context_string() for c in chunks]

        # Construct the user message
        user_message = build_assessment_prompt(query, context_strings)

        # Log the full prompt for observability and debugging
        logger.info(
            "llm_prompt_constructed",
            system_prompt_length=len(SYSTEM_PROMPT),
            user_message_length=len(user_message),
            context_chunks_count=len(context_strings),
        )
        logger.info(
            "llm_prompt_full",
            trace_id=get_current_trace_id(),
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
        )

        # Call chat completions API
        # Note: NVIDIA NIM / integrate.api.nvidia.com rejects response_format={"type": "json_object"}
        # with 422 Unprocessable Entity. We omit response_format when targeting NVIDIA NIM,
        # relying on explicit system instructions and robust JSON cleanup.
        client_base = str(self._client.base_url) if self._client.base_url else ""
        is_nvidia = "nvidia.com" in client_base or "nvidia" in self._model.lower()

        create_kwargs = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.1,  # Low temperature for consistent, deterministic output
            "max_tokens": 1500,
        }
        if not is_nvidia:
            create_kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**create_kwargs)

        content = response.choices[0].message.content or ""

        logger.info(
            "llm_response_received",
            model=self._model,
            usage_prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            usage_completion_tokens=response.usage.completion_tokens if response.usage else 0,
            response_length=len(content),
        )

        return content

    @staticmethod
    def _clean_json_response(text: str) -> str:
        """
        Extract clean JSON string from LLM response.

        Handles markdown code blocks (e.g. ```json ... ```) and
        conversational prefixes/suffixes produced by open models like Llama 3.1.
        """
        stripped = text.strip()
        # Handle markdown code blocks
        if "```json" in stripped:
            parts = stripped.split("```json", 1)[1].split("```", 1)
            return parts[0].strip()
        if "```" in stripped:
            parts = stripped.split("```", 1)[1].split("```", 1)
            return parts[0].strip()

        # If there is conversational text around raw JSON, extract { ... }
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            return stripped[start : end + 1].strip()

        return stripped

    def _extract_verdict(
        self,
        query: str,
        llm_response: str,
        retrieved_chunks: list[RetrievedChunk],
        trace_id: str,
    ) -> ComplianceAssessment:
        """
        Phase 3: Parse the LLM's JSON response into a structured verdict.

        Handles malformed responses gracefully — if the LLM returns invalid
        JSON, we return a NEEDS_REVIEW verdict with an explanation rather
        than crashing the request.
        """
        cleaned_json = self._clean_json_response(llm_response)
        try:
            parsed = json.loads(cleaned_json)
        except json.JSONDecodeError:
            logger.error(
                "llm_response_not_json",
                response_preview=llm_response[:500],
            )
            return self._fallback_assessment(
                query=query,
                reason="LLM response was not valid JSON. Manual review required.",
                trace_id=trace_id,
                chunks=retrieved_chunks,
            )

        # Extract and validate the verdict
        raw_verdict = parsed.get("verdict", "NEEDS_REVIEW").upper().strip()
        try:
            verdict = Verdict(raw_verdict)
        except ValueError:
            logger.warning(
                "unexpected_verdict_value",
                raw_verdict=raw_verdict,
            )
            verdict = Verdict.NEEDS_REVIEW

        # Build source references from the retrieved chunks
        sources = [
            SourceReference(
                document=chunk.source,
                section=chunk.section,
                relevance_score=chunk.similarity_score,
            )
            for chunk in retrieved_chunks
        ]

        # Extract confidence, clamped to [0, 1]
        raw_confidence = parsed.get("confidence", 0.5)
        confidence = max(0.0, min(1.0, float(raw_confidence)))

        # Extract conditions list
        conditions = parsed.get("conditions", [])
        if isinstance(conditions, str):
            conditions = [conditions]

        assessment = ComplianceAssessment(
            query=query,
            verdict=verdict,
            reasoning=parsed.get("reasoning", "No reasoning provided."),
            conditions=conditions,
            sources=sources,
            confidence=confidence,
            trace_id=trace_id,
        )

        logger.info(
            "verdict_extracted",
            verdict=verdict.value,
            confidence=confidence,
            conditions_count=len(conditions),
        )

        return assessment

    def _fallback_assessment(
        self,
        query: str,
        reason: str,
        trace_id: str,
        chunks: list[RetrievedChunk],
    ) -> ComplianceAssessment:
        """
        Generate a safe fallback when the LLM fails or returns garbage.

        Always returns NEEDS_REVIEW — we never auto-approve or auto-reject
        when the system is uncertain. This is a deliberate safety design:
        false confidence is worse than admitting uncertainty.
        """
        sources = [
            SourceReference(
                document=c.source,
                section=c.section,
                relevance_score=c.similarity_score,
            )
            for c in chunks
        ]

        return ComplianceAssessment(
            query=query,
            verdict=Verdict.NEEDS_REVIEW,
            reasoning=reason,
            conditions=["Manual review by Sharia Supervisory Board required"],
            sources=sources,
            confidence=0.0,
            trace_id=trace_id,
        )
