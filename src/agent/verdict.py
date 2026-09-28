"""
Verdict model and compliance assessment data structures.

These Pydantic models define the structured output format for compliance
assessments. Using an enum for the verdict ensures type safety and prevents
the LLM from inventing non-standard verdict categories.
"""

from enum import Enum
from pydantic import BaseModel, Field


class Verdict(str, Enum):
    """
    Three-tier compliance verdict.

    Design decision — why three tiers, not two:
    Binary COMPLIANT/NON_COMPLIANT forces the agent to make a definitive
    call even when the answer genuinely depends on implementation details.
    NEEDS_REVIEW captures cases where compliance depends on conditions
    (e.g., "compliant IF the profit-sharing ratio is agreed upfront").
    This maps to real-world compliance workflows where ambiguous cases
    are escalated to the Sharia Supervisory Board.
    """

    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class SourceReference(BaseModel):
    """A reference to a specific section of a Sharia standard."""

    document: str = Field(
        description="Source document name (e.g., 'aaoifi_riba_prohibition')"
    )
    section: str = Field(
        description="Section title within the document"
    )
    relevance_score: float = Field(
        description="Cosine similarity score (0-1) indicating retrieval confidence"
    )


class ComplianceAssessment(BaseModel):
    """
    Structured output from the compliance agent.

    This is the primary data contract — all consumers (API, frontend,
    audit logs) work with this model.
    """

    query: str = Field(
        description="The original compliance question"
    )
    verdict: Verdict = Field(
        description="The compliance determination"
    )
    reasoning: str = Field(
        description="Detailed explanation citing specific Sharia standards"
    )
    conditions: list[str] = Field(
        default_factory=list,
        description=(
            "Conditions that must be met for compliance (populated when "
            "verdict is NEEDS_REVIEW or COMPLIANT with caveats)"
        ),
    )
    sources: list[SourceReference] = Field(
        default_factory=list,
        description="Sharia standard sections consulted for this assessment"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Agent's confidence in the verdict (0-1). Based on retrieval "
            "quality and clarity of applicable standards."
        ),
    )
    trace_id: str = Field(
        default="",
        description="Request trace ID for observability and audit"
    )
