"""Pinecone-backed vector store for cloud deployments."""

from __future__ import annotations

import time

from pinecone import Pinecone, ServerlessSpec

from src.config import settings
from src.observability.logger import get_logger

logger = get_logger(__name__)


class PineconeStore:
    """Store and query document vectors in a Pinecone serverless index."""

    def __init__(
        self,
        index_name: str = settings.pinecone_index_name,
        namespace: str = settings.pinecone_namespace,
    ) -> None:
        if not settings.pinecone_api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is required when using the Pinecone vector store."
            )

        self._index_name = index_name
        self._namespace = namespace
        self._client = Pinecone(api_key=settings.pinecone_api_key)

        if not self._index_exists(index_name):
            logger.info(
                "pinecone_index_creating",
                index=index_name,
                dimension=settings.pinecone_dimension,
            )
            self._client.create_index(
                name=index_name,
                dimension=settings.pinecone_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region,
                ),
            )
            self._wait_until_ready(index_name)

        index_description = self._client.describe_index(index_name)
        self._index = self._client.Index(host=index_description.host)

        logger.info(
            "pinecone_initialized",
            index=index_name,
            namespace=namespace,
            existing_count=self.count(),
        )

    def _index_exists(self, index_name: str) -> bool:
        return index_name in self._client.list_indexes().names()

    def _wait_until_ready(self, index_name: str) -> None:
        for _ in range(60):
            description = self._client.describe_index(index_name)
            if description.status.get("ready"):
                return
            time.sleep(2)
        raise TimeoutError(f"Pinecone index did not become ready: {index_name}")

    def add_documents(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, str]],
    ) -> None:
        """Upsert document vectors and searchable metadata."""
        vectors = [
            {
                "id": doc_id,
                "values": embedding,
                "metadata": {**metadata, "text": text},
            }
            for doc_id, embedding, text, metadata in zip(
                ids, embeddings, texts, metadatas, strict=True
            )
        ]
        self._index.upsert(vectors=vectors, namespace=self._namespace)
        logger.info("documents_upserted", count=len(vectors))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = settings.retrieval_top_k,
    ) -> dict:
        """Query Pinecone and return a Chroma-compatible result shape."""
        if self.count() == 0:
            return {"ids": [[]], "scores": [[]], "metadatas": [[]]}

        response = self._index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self._namespace,
            include_metadata=True,
        )
        matches = response.matches

        return {
            "ids": [[match.id for match in matches]],
            "scores": [[match.score for match in matches]],
            "metadatas": [[match.metadata or {} for match in matches]],
        }

    def count(self) -> int:
        """Return the number of vectors in this namespace."""
        stats = self._index.describe_index_stats(namespace=self._namespace)
        namespace_stats = stats.namespaces.get(self._namespace)
        if namespace_stats is None:
            return 0
        if isinstance(namespace_stats, dict):
            return int(namespace_stats.get("vector_count", 0))
        return int(getattr(namespace_stats, "vector_count", 0))

    def reset(self) -> None:
        """Delete all vectors from this namespace for clean re-ingestion."""
        self._index.delete(delete_all=True, namespace=self._namespace)
        logger.warning("namespace_reset", index=self._index_name, namespace=self._namespace)