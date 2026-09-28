"""
Standalone ingestion script.

Run this to (re-)ingest all Sharia standard documents into the vector store.
Can be run independently of the API server.

Usage:
    python -m scripts.ingest
    # or
    python scripts/ingest.py
"""

import sys
from pathlib import Path

# Add project root to path so we can import src.*
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.observability.logger import get_logger
from src.rag.embedder import Embedder
from src.rag.ingestion import IngestionPipeline
from src.vectorstore.pinecone_store import PineconeStore

logger = get_logger(__name__)


def main() -> None:
    """Run the ingestion pipeline."""
    logger.info("ingestion_script_started")

    embedder = Embedder()
    store = PineconeStore()

    # Reset the store for clean re-ingestion
    existing_count = store.count()
    if existing_count > 0:
        logger.info(
            "clearing_existing_documents",
            count=existing_count,
        )
        store.reset()

    pipeline = IngestionPipeline(embedder=embedder, store=store)
    total = pipeline.ingest_directory(settings.documents_dir)

    logger.info(
        "ingestion_script_completed",
        total_chunks=total,
        store_count=store.count(),
    )
    print(f"\n✓ Ingested {total} chunks from {settings.documents_dir}")
    print(f"  Vector store: Pinecone/{settings.pinecone_index_name} ({store.count()} documents)")


if __name__ == "__main__":
    main()
