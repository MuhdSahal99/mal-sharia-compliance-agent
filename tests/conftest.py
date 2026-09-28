"""Test configuration and shared fixtures."""

import pytest
from src.rag.chunker import DocumentChunker, DocumentChunk


@pytest.fixture
def chunker() -> DocumentChunker:
    """Provide a chunker with small chunk size for testing."""
    return DocumentChunker(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def sample_document() -> str:
    """A small synthetic document for testing chunking."""
    return """# Test Standard

## Section One — Riba Rules

Riba is categorically prohibited in Islamic finance. Any predetermined
guaranteed return on a loan constitutes Riba al-Nasiah. Fixed interest
rates on savings accounts are non-compliant with Sharia law.

Permissible alternatives include Mudaraba-based savings accounts where
returns are based on actual profit-sharing ratios.

## Section Two — Murabaha Rules

Murabaha is a cost-plus sale where the seller discloses the original
cost and adds a known profit margin. The bank must genuinely own the
asset before selling it to the client.

## Section Three — Key Rulings

| Ruling | Subject | Verdict |
|--------|---------|---------|
| R-1 | Fixed interest | NON_COMPLIANT |
| R-2 | Profit sharing | COMPLIANT |
"""


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Pre-built chunks for retriever testing."""
    return [
        DocumentChunk(
            text="Riba is prohibited. Fixed interest is non-compliant.",
            metadata={"source": "test_doc", "section": "Riba Rules"},
        ),
        DocumentChunk(
            text="Murabaha requires genuine asset ownership by the bank.",
            metadata={"source": "test_doc", "section": "Murabaha Rules"},
        ),
    ]
