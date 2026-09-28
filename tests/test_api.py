"""Tests for the API endpoints."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport

from src.main import create_app
from src.agent.verdict import ComplianceAssessment, Verdict, SourceReference


@pytest.fixture
def mock_agent() -> AsyncMock:
    """Mock compliance agent that returns a fixed assessment."""
    agent = AsyncMock()
    agent.assess.return_value = ComplianceAssessment(
        query="Can Mal offer a fixed-return savings account?",
        verdict=Verdict.NON_COMPLIANT,
        reasoning=(
            "A fixed-return savings account guarantees a predetermined return "
            "on the deposited principal, which constitutes Riba al-Nasiah "
            "under AAOIFI Sharia Standard No. 3 (Ruling R-3.01)."
        ),
        conditions=[],
        sources=[
            SourceReference(
                document="aaoifi_riba_prohibition",
                section="Fixed-Return Savings Accounts",
                relevance_score=0.92,
            )
        ],
        confidence=0.95,
        trace_id="test-trace-123",
    )
    return agent


@pytest.fixture
def mock_store() -> MagicMock:
    """Mock vector store that reports a document count."""
    store = MagicMock()
    store.count.return_value = 42
    return store


class TestHealthEndpoint:
    """Tests for GET /health."""

    @pytest.mark.asyncio
    async def test_health_returns_200(
        self, mock_agent: AsyncMock, mock_store: MagicMock
    ) -> None:
        """Health endpoint should return 200 with status info."""
        app = create_app()

        # Patch the route dependencies
        with patch("src.api.routes._agent", mock_agent), \
             patch("src.api.routes._store", mock_store):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "vector_store_documents" in data

    @pytest.mark.asyncio
    async def test_health_includes_trace_id_header(
        self, mock_agent: AsyncMock, mock_store: MagicMock
    ) -> None:
        """Every response should include X-Trace-ID header."""
        app = create_app()

        with patch("src.api.routes._agent", mock_agent), \
             patch("src.api.routes._store", mock_store):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/health")

        assert "x-trace-id" in response.headers


class TestAssessEndpoint:
    """Tests for POST /assess."""

    @pytest.mark.asyncio
    async def test_assess_returns_structured_verdict(
        self, mock_agent: AsyncMock, mock_store: MagicMock
    ) -> None:
        """Assessment should return a valid structured response."""
        app = create_app()

        with patch("src.api.routes._agent", mock_agent), \
             patch("src.api.routes._store", mock_store):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/assess",
                    json={"query": "Can Mal offer a fixed-return savings account?"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] == "NON_COMPLIANT"
        assert "reasoning" in data
        assert "trace_id" in data
        assert len(data["sources"]) > 0

    @pytest.mark.asyncio
    async def test_assess_rejects_short_query(
        self, mock_agent: AsyncMock, mock_store: MagicMock
    ) -> None:
        """Queries shorter than 10 characters should be rejected."""
        app = create_app()

        with patch("src.api.routes._agent", mock_agent), \
             patch("src.api.routes._store", mock_store):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/assess",
                    json={"query": "short"},
                )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_assess_rejects_missing_query(
        self, mock_agent: AsyncMock, mock_store: MagicMock
    ) -> None:
        """Missing query field should return 422."""
        app = create_app()

        with patch("src.api.routes._agent", mock_agent), \
             patch("src.api.routes._store", mock_store):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/assess", json={})

        assert response.status_code == 422
