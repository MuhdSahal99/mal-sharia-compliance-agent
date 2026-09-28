"""
API request/response schemas.

These Pydantic models define the API contract. They are separate from the
internal domain models (verdict.py) to allow the API schema to evolve
independently of the internal representation.

In practice, for a small system like this, the schemas closely mirror the
domain models. In a larger system, the API schema might aggregate data
from multiple services or transform internal fields for external consumers.
"""

from pydantic import BaseModel, Field

from src.agent.verdict import Verdict


class AssessmentRequest(BaseModel):
    """POST /assess request body."""

    query: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="The compliance question in plain English",
        examples=["Can Mal offer a fixed-return savings account?"],
    )


class SourceReferenceResponse(BaseModel):
    """A cited source in the assessment response."""

    document: str
    section: str
    relevance_score: float


class AssessmentResponse(BaseModel):
    """POST /assess response body."""

    trace_id: str = Field(description="Unique request identifier for debugging")
    query: str = Field(description="The original query")
    verdict: Verdict = Field(description="COMPLIANT, NON_COMPLIANT, or NEEDS_REVIEW")
    reasoning: str = Field(description="Detailed explanation with citations")
    conditions: list[str] = Field(
        default_factory=list,
        description="Conditions required for compliance",
    )
    sources: list[SourceReferenceResponse] = Field(
        default_factory=list,
        description="Sharia standards consulted",
    )
    confidence: float = Field(
        description="Assessment confidence score (0-1)"
    )


class HealthResponse(BaseModel):
    """GET /health response body."""

    status: str = Field(default="healthy")
    version: str
    vector_store_documents: int = Field(
        description="Number of document chunks in the vector store"
    )
    model: str = Field(description="LLM model in use")


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str
    detail: str
    trace_id: str = ""
