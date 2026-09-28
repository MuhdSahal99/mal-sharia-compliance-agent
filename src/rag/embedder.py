"""
Text embedding using OpenAI's embedding API.

Design decision — OpenAI embeddings vs. local sentence-transformers:

Option A: sentence-transformers/all-MiniLM-L6-v2 (local)
  + Free, no API dependency
  + Full control, works offline
  - Requires PyTorch (~2GB Docker image)
  - Lower quality on domain-specific financial text
  - 384 dimensions (less expressive)

Option B: OpenAI text-embedding-3-small (chosen)
  + 1536 dimensions, strong semantic quality
  + Lightweight deployment (no PyTorch)
  + Consistent with using OpenAI for LLM
  - API cost (~$0.02 per 1M tokens — negligible for our corpus)
  - External dependency

We chose Option B because deployment size matters for free-tier hosting,
and embedding quality directly impacts retrieval accuracy. At scale, we'd
evaluate domain-specific models fine-tuned on Islamic finance text.

Retry logic: Embedding calls are wrapped with tenacity for resilience
against transient API errors (rate limits, timeouts).
"""

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.observability.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """Generates text embeddings via OpenAI or NVIDIA NIM API with retry logic."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model or settings.effective_embedding_model
        resolved_base_url = base_url or settings.effective_embedding_base_url
        resolved_api_key = api_key or settings.effective_embedding_api_key or "none"
        self._client = OpenAI(
            base_url=resolved_base_url if resolved_base_url else None,
            api_key=resolved_api_key,
        )
        client_base = str(self._client.base_url) if self._client.base_url else ""
        self._is_nvidia = "nvidia.com" in client_base or "nvidia" in self._model.lower()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).

        Raises:
            openai.APIError: After 3 retries with exponential backoff.
        """
        if not texts:
            return []

        # Accept batches natively
        kwargs = {
            "input": texts,
            "model": self._model,
            "dimensions": settings.embedding_dimensions,
        }
        if self._is_nvidia:
            kwargs["extra_body"] = {"input_type": "passage"}

        response = self._client.embeddings.create(**kwargs)

        embeddings = [item.embedding for item in response.data]

        logger.debug(
            "generated_embeddings",
            count=len(texts),
            model=self._model,
            dimensions=len(embeddings[0]) if embeddings else 0,
        )
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.

        Uses input_type="query" for NVIDIA asymmetric embedding models
        to maximize retrieval accuracy.
        """
        if not query.strip():
            return []

        kwargs = {
            "input": [query],
            "model": self._model,
            "dimensions": settings.embedding_dimensions,
        }
        if self._is_nvidia:
            kwargs["extra_body"] = {"input_type": "query"}

        response = self._client.embeddings.create(**kwargs)
        return response.data[0].embedding
