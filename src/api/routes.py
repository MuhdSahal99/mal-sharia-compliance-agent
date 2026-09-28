"""
API route definitions.

Two endpoints as required:
- POST /assess  → Submit a compliance query, get a structured verdict
- GET  /health  → Service health check with vector store status

Additional:
- GET  /        → Root redirect to docs (convenience)

Routes are thin — they validate input, delegate to the agent, and format
the response. No business logic lives here.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

from src import __version__
from src.api.schemas import (
    AssessmentRequest,
    AssessmentResponse,
    ErrorResponse,
    HealthResponse,
    SourceReferenceResponse,
)
from src.observability.logger import get_logger
from src.observability.tracer import get_current_trace_id

logger = get_logger(__name__)

# Router instance — mounted by main.py
router = APIRouter()

# These will be injected at startup by main.py
_agent = None
_store = None


def configure_routes(agent, store) -> None:  # type: ignore[no-untyped-def]
    """
    Inject dependencies into routes.

    Called once at application startup. We use module-level state rather
    than FastAPI's dependency injection for the agent and store because:
    1. They're singletons with expensive initialization (embedding model, DB)
    2. They should be initialized once at startup, not per-request
    3. This avoids circular import issues with the app factory
    """
    global _agent, _store
    _agent = agent
    _store = store


@router.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns service status, version, and vector store statistics.",
)
async def health_check() -> HealthResponse:
    """
    Health endpoint for monitoring and deployment readiness probes.

    Reports the number of documents in the vector store so operators
    can verify that ingestion completed successfully.
    """
    from src.config import settings

    doc_count = _store.count() if _store else 0

    return HealthResponse(
        status="healthy",
        version=__version__,
        vector_store_documents=doc_count,
        model=settings.llm_model,
    )


@router.post(
    "/assess",
    response_model=AssessmentResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Assess Sharia compliance",
    description=(
        "Submit a plain-English query about a financial product or transaction. "
        "Returns a structured compliance verdict (COMPLIANT, NON_COMPLIANT, or "
        "NEEDS_REVIEW) with cited reasoning from AAOIFI standards."
    ),
)
async def assess_compliance(request: AssessmentRequest) -> AssessmentResponse:
    """
    Main assessment endpoint.

    Flow:
    1. Validate the query (Pydantic does this automatically)
    2. Delegate to the ComplianceAgent
    3. Transform the internal ComplianceAssessment into the API response schema
    """
    trace_id = get_current_trace_id()

    if _agent is None:
        logger.error("agent_not_initialized")
        raise HTTPException(
            status_code=500,
            detail="Compliance agent not initialized. Check server logs.",
        )

    try:
        assessment = await _agent.assess(request.query)

        return AssessmentResponse(
            trace_id=trace_id,
            query=assessment.query,
            verdict=assessment.verdict,
            reasoning=assessment.reasoning,
            conditions=assessment.conditions,
            sources=[
                SourceReferenceResponse(
                    document=s.document,
                    section=s.section,
                    relevance_score=s.relevance_score,
                )
                for s in assessment.sources
            ],
            confidence=assessment.confidence,
        )

    except Exception as e:
        logger.exception(
            "assessment_failed",
            error=str(e),
            query=request.query[:100],
        )
        raise HTTPException(
            status_code=500,
            detail=f"Assessment failed: {str(e)}. Trace ID: {trace_id}",
        )
