"""
Document retriever — queries the vector store and filters results.

The retriever sits between the agent and the raw vector store, adding:
1. Similarity threshold filtering (drop irrelevant chunks)
2. Structured result formatting (consistent interface for the agent)
3. Logging of which chunks were retrieved (observability requirement)
"""

from dataclasses import dataclass

from src.config import settings
from src.observability.logger import get_logger
from src.rag.embedder import Embedder
from src.vectorstore.pinecone_store import PineconeStore

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with similarity score."""

    text: str
    source: str
    section: str
    similarity_score: float  # 1.0 = identical, 0.0 = unrelated

    def to_context_string(self) -> str:
        """Format for inclusion in the LLM prompt."""
        return (
            f"[Source: {self.source} | Section: {self.section} | "
            f"Relevance: {self.similarity_score:.2f}]\n{self.text}"
        )


class Retriever:
    """
    Retrieves relevant Sharia standard chunks for a compliance query.

    Encapsulates embedding + vector search + threshold filtering.
    The agent calls retriever.retrieve(query) and gets back a clean
    list of RetrievedChunk objects — no awareness of Pinecone or
    embedding mechanics needed.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: PineconeStore,
        top_k: int = settings.retrieval_top_k,
        threshold: float = settings.retrieval_similarity_threshold,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._threshold = threshold

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Retrieve relevant document chunks for the given query.

        Pipeline:
        1. Embed the query using the same model used for documents
        2. Query Pinecone for the top_k most similar chunks
        3. Use Pinecone cosine similarity scores
        4. Filter out chunks below the similarity threshold
        5. Return structured RetrievedChunk objects

        Args:
            query: Natural language compliance question.

        Returns:
            List of RetrievedChunk objects, sorted by relevance (highest first).
        """
        # Step 1: Embed the query
        query_embedding = self._embedder.embed_query(query)

        # Step 2: Vector search
        raw_results = self._store.query(
            query_embedding=query_embedding,
            top_k=self._top_k,
        )

        # Step 3 & 4: Parse results and apply threshold
        chunks: list[RetrievedChunk] = []

        if not raw_results["ids"] or not raw_results["ids"][0]:
            logger.warning("no_chunks_retrieved", query=query[:100])
            return chunks

        for i, doc_id in enumerate(raw_results["ids"][0]):
            similarity = raw_results["scores"][0][i]

            if similarity < self._threshold:
                continue

            metadata = raw_results["metadatas"][0][i]
            text = metadata.get("text", "")

            chunk = RetrievedChunk(
                text=text,
                source=metadata.get("source", "unknown"),
                section=metadata.get("section", "unknown"),
                similarity_score=round(similarity, 4),
            )
            chunks.append(chunk)

        # Sort by similarity (highest first) for consistent ordering
        chunks.sort(key=lambda c: c.similarity_score, reverse=True)

        logger.info(
            "chunks_retrieved",
            query=query[:100],
            total_candidates=len(raw_results["ids"][0]),
            after_threshold=len(chunks),
            top_score=chunks[0].similarity_score if chunks else 0.0,
        )

        return chunks
