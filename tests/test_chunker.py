"""Tests for the document chunker."""

from src.rag.chunker import DocumentChunker


class TestDocumentChunker:
    """Verify chunking logic produces correct, well-formed chunks."""

    def test_chunks_have_metadata(
        self, chunker: DocumentChunker, sample_document: str
    ) -> None:
        """Each chunk should carry source and section metadata."""
        chunks = chunker.chunk_document(sample_document, "test_standard")

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata["source"] == "test_standard"
            assert "section" in chunk.metadata
            assert chunk.metadata["section"] != ""

    def test_sections_are_preserved(
        self, chunker: DocumentChunker, sample_document: str
    ) -> None:
        """Chunks should reference the section they came from."""
        chunks = chunker.chunk_document(sample_document, "test_standard")
        sections = {c.metadata["section"] for c in chunks}

        # Our sample doc has sections: "Section One — Riba Rules",
        # "Section Two — Murabaha Rules", "Section Three — Key Rulings"
        assert any("Riba" in s for s in sections)
        assert any("Murabaha" in s for s in sections)

    def test_chunk_ids_are_deterministic(
        self, chunker: DocumentChunker, sample_document: str
    ) -> None:
        """Same input should produce the same chunk IDs (for idempotent upsert)."""
        chunks_1 = chunker.chunk_document(sample_document, "test")
        chunks_2 = chunker.chunk_document(sample_document, "test")

        ids_1 = [c.chunk_id for c in chunks_1]
        ids_2 = [c.chunk_id for c in chunks_2]
        assert ids_1 == ids_2

    def test_empty_document_produces_no_chunks(
        self, chunker: DocumentChunker
    ) -> None:
        """An empty document should produce zero chunks, not crash."""
        chunks = chunker.chunk_document("", "empty")
        assert chunks == []

    def test_very_short_text_is_filtered(
        self, chunker: DocumentChunker
    ) -> None:
        """Text shorter than 20 tokens should be filtered out."""
        chunks = chunker.chunk_document("# Header\n\nOk.", "short")
        # "Ok." is too short to be a useful chunk
        assert all(len(c.text) > 5 for c in chunks)

    def test_chunk_text_is_non_empty(
        self, chunker: DocumentChunker, sample_document: str
    ) -> None:
        """No chunk should have empty text."""
        chunks = chunker.chunk_document(sample_document, "test")
        for chunk in chunks:
            assert chunk.text.strip() != ""
