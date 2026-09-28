"""Tests for the compliance agent's verdict extraction logic."""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.agent.compliance_agent import ComplianceAgent
from src.agent.verdict import Verdict
from src.rag.retriever import RetrievedChunk


class TestVerdictExtraction:
    """Test the agent's ability to parse LLM responses into structured verdicts."""

    @pytest.fixture
    def agent(self) -> ComplianceAgent:
        """Agent with a mocked retriever (no real vector store needed)."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        return ComplianceAgent(retriever=mock_retriever)

    @pytest.fixture
    def sample_chunks(self) -> list[RetrievedChunk]:
        """Sample retrieved chunks for testing."""
        return [
            RetrievedChunk(
                text="Fixed interest is Riba and is prohibited.",
                source="aaoifi_riba_prohibition",
                section="Riba al-Nasiah",
                similarity_score=0.9,
            ),
        ]

    def test_valid_json_response_parsed(
        self, agent: ComplianceAgent, sample_chunks: list[RetrievedChunk]
    ) -> None:
        """Valid JSON from LLM should be parsed into ComplianceAssessment."""
        llm_response = json.dumps({
            "verdict": "NON_COMPLIANT",
            "reasoning": "Fixed interest constitutes Riba.",
            "conditions": [],
            "confidence": 0.95,
        })

        result = agent._extract_verdict(
            query="test query",
            llm_response=llm_response,
            retrieved_chunks=sample_chunks,
            trace_id="test-trace",
        )

        assert result.verdict == Verdict.NON_COMPLIANT
        assert result.confidence == 0.95
        assert "Riba" in result.reasoning

    def test_invalid_json_returns_needs_review(
        self, agent: ComplianceAgent, sample_chunks: list[RetrievedChunk]
    ) -> None:
        """Invalid JSON should trigger fallback to NEEDS_REVIEW."""
        result = agent._extract_verdict(
            query="test query",
            llm_response="This is not JSON at all",
            retrieved_chunks=sample_chunks,
            trace_id="test-trace",
        )

        assert result.verdict == Verdict.NEEDS_REVIEW
        assert result.confidence == 0.0

    def test_unknown_verdict_defaults_to_needs_review(
        self, agent: ComplianceAgent, sample_chunks: list[RetrievedChunk]
    ) -> None:
        """Unexpected verdict value from LLM should default to NEEDS_REVIEW."""
        llm_response = json.dumps({
            "verdict": "MAYBE_COMPLIANT",
            "reasoning": "Uncertain.",
            "conditions": [],
            "confidence": 0.5,
        })

        result = agent._extract_verdict(
            query="test query",
            llm_response=llm_response,
            retrieved_chunks=sample_chunks,
            trace_id="test-trace",
        )

        assert result.verdict == Verdict.NEEDS_REVIEW

    def test_confidence_is_clamped(
        self, agent: ComplianceAgent, sample_chunks: list[RetrievedChunk]
    ) -> None:
        """Confidence values outside [0,1] should be clamped."""
        llm_response = json.dumps({
            "verdict": "COMPLIANT",
            "reasoning": "All good.",
            "conditions": [],
            "confidence": 1.5,  # Out of range
        })

        result = agent._extract_verdict(
            query="test query",
            llm_response=llm_response,
            retrieved_chunks=sample_chunks,
            trace_id="test-trace",
        )

        assert result.confidence == 1.0

    def test_sources_from_chunks(
        self, agent: ComplianceAgent, sample_chunks: list[RetrievedChunk]
    ) -> None:
        """Source references should be built from retrieved chunks."""
        llm_response = json.dumps({
            "verdict": "COMPLIANT",
            "reasoning": "OK.",
            "conditions": [],
            "confidence": 0.9,
        })

        result = agent._extract_verdict(
            query="test query",
            llm_response=llm_response,
            retrieved_chunks=sample_chunks,
            trace_id="test-trace",
        )

        assert len(result.sources) == 1
        assert result.sources[0].document == "aaoifi_riba_prohibition"
        assert result.sources[0].section == "Riba al-Nasiah"

    def test_conditions_string_converted_to_list(
        self, agent: ComplianceAgent, sample_chunks: list[RetrievedChunk]
    ) -> None:
        """If LLM returns conditions as a string, it should be wrapped in a list."""
        llm_response = json.dumps({
            "verdict": "NEEDS_REVIEW",
            "reasoning": "Depends on structure.",
            "conditions": "SSB approval required",
            "confidence": 0.6,
        })

        result = agent._extract_verdict(
            query="test query",
            llm_response=llm_response,
            retrieved_chunks=sample_chunks,
            trace_id="test-trace",
        )

        assert isinstance(result.conditions, list)
        assert len(result.conditions) == 1

    def test_markdown_fenced_json_parsed(
        self, agent: ComplianceAgent, sample_chunks: list[RetrievedChunk]
    ) -> None:
        """Markdown code fence around JSON (common in Llama models) should be stripped."""
        llm_response = """```json
{
    "verdict": "COMPLIANT",
    "reasoning": "Product complies with AAOIFI standards.",
    "conditions": ["Annual review"],
    "confidence": 0.9
}
```"""
        result = agent._extract_verdict(
            query="test query",
            llm_response=llm_response,
            retrieved_chunks=sample_chunks,
            trace_id="test-trace",
        )

        assert result.verdict == Verdict.COMPLIANT
        assert result.confidence == 0.9
        assert result.conditions == ["Annual review"]

    def test_conversational_wrapped_json_parsed(
        self, agent: ComplianceAgent, sample_chunks: list[RetrievedChunk]
    ) -> None:
        """Conversational text surrounding raw JSON should still parse correctly."""
        llm_response = (
            "Here is the compliance assessment based on AAOIFI standards:\n"
            "{\"verdict\": \"NON_COMPLIANT\", \"reasoning\": \"Violates Riba rules\", "
            "\"conditions\": [], \"confidence\": 0.92}\n"
            "Let me know if you need more details."
        )
        result = agent._extract_verdict(
            query="test query",
            llm_response=llm_response,
            retrieved_chunks=sample_chunks,
            trace_id="test-trace",
        )

        assert result.verdict == Verdict.NON_COMPLIANT
        assert result.confidence == 0.92
        assert "Violates Riba" in result.reasoning

    def test_nvidia_nim_omits_response_format(self, sample_chunks: list[RetrievedChunk]) -> None:
        """Calling NVIDIA NIM should not pass response_format (prevents 422 error)."""
        mock_retriever = MagicMock()
        agent = ComplianceAgent(
            retriever=mock_retriever,
            model="meta/llama-3.1-70b-instruct",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="nvapi-test",
        )
        with patch.object(agent._client.chat.completions, "create") as mock_create:
            mock_create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content='{"verdict": "COMPLIANT"}'))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=20),
            )
            agent._reason("test query", sample_chunks)
            called_kwargs = mock_create.call_args.kwargs
            assert "response_format" not in called_kwargs
            assert called_kwargs["model"] == "meta/llama-3.1-70b-instruct"
