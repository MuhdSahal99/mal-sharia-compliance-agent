"""
Document ingestion pipeline.

Orchestrates the full ingestion flow:
    Read markdown files -> Chunk -> Embed -> Store in Pinecone

This is designed to be run as a one-time setup step (via scripts/ingest.py)
or called at application startup to ensure the vector store is populated.

Idempotency: Because PineconeStore uses upsert (not insert), running this
multiple times with the same documents is safe — it overwrites existing
chunks rather than duplicating them.
"""

from pathlib import Path

from src.config import settings
from src.observability.logger import get_logger
from src.rag.chunker import DocumentChunker
from src.rag.embedder import Embedder
from src.vectorstore.pinecone_store import PineconeStore

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Reads, chunks, embeds, and stores Sharia standard documents.

    Usage:
        pipeline = IngestionPipeline(embedder, store)
        pipeline.ingest_directory(settings.documents_dir)
    """

    def __init__(
        self,
        embedder: Embedder,
        store: PineconeStore,
        chunker: DocumentChunker | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._chunker = chunker or DocumentChunker()

    def ingest_directory(self, directory: Path | None = None) -> int:
        """
        Ingest all markdown files from the specified directory.

        Args:
            directory: Path to the documents directory.
                       Defaults to src/documents/.

        Returns:
            Total number of chunks ingested.
        """
        doc_dir = directory or settings.documents_dir

        if not doc_dir.exists():
            logger.error("documents_dir_not_found", path=str(doc_dir))
            raise FileNotFoundError(f"Documents directory not found: {doc_dir}")

        md_files = sorted(doc_dir.glob("*.md"))
        if not md_files:
            logger.warning("no_documents_found", path=str(doc_dir))
            return 0

        logger.info(
            "ingestion_started",
            directory=str(doc_dir),
            file_count=len(md_files),
        )

        total_chunks = 0

        for filepath in md_files:
            count = self._ingest_file(filepath)
            total_chunks += count

        logger.info(
            "ingestion_completed",
            total_chunks=total_chunks,
            total_files=len(md_files),
            store_count=self._store.count(),
        )

        return total_chunks

    def _ingest_file(self, filepath: Path) -> int:
        """
        Ingest a single markdown file.

        Pipeline: Read → Chunk → Embed (batched) → Store

        Args:
            filepath: Path to the markdown file.

        Returns:
            Number of chunks created from this file.
        """
        text = filepath.read_text(encoding="utf-8")
        source_name = filepath.stem  # e.g., "aaoifi_riba_prohibition"

        # Step 1: Chunk the document
        chunks = self._chunker.chunk_document(text, source_name)

        if not chunks:
            logger.warning("no_chunks_produced", source=source_name)
            return 0

        # Step 2: Embed all chunks in a single batch (efficient API usage)
        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_texts(texts)

        # Step 3: Store with metadata
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        self._store.add_documents(
            ids=ids,
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas,
        )

        logger.info(
            "file_ingested",
            source=source_name,
            chunks=len(chunks),
        )

        return len(chunks)
