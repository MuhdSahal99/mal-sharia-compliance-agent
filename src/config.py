"""
Application configuration loaded from environment variables.

Uses pydantic-settings for type-safe configuration with validation.
Every setting has a sensible default so the app can start in development
with just an OPENAI_API_KEY set.

Design decision: Single Settings object instantiated once at import time.
All modules import `settings` from here rather than reading os.environ
directly — this gives us one place to validate, document, and test
configuration.
"""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Don't fail on unexpected env vars
    )

    # ── LLM Configuration ─────────────────────────────────────
    # API keys
    nvidia_api_key: str = ""
    gemini_api_key: str = ""
    openai_api_key: str = ""
    llm_api_key: str = ""

    # Endpoints & Models
    # Default to NVIDIA NIM with free Llama 3.1 70B Instruct
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-2.5-flash"

    # Embedding Configuration
    embedding_api_key: str = ""
    embedding_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 2048

    @property
    def effective_llm_api_key(self) -> str:
        """Resolve LLM API key with fallback order: llm_api_key -> nvidia_api_key -> openai_api_key."""
        if self.llm_api_key:
            return self.llm_api_key
        if self.gemini_api_key:
            return self.gemini_api_key
        if self.nvidia_api_key:
            return self.nvidia_api_key
        return self.openai_api_key

    @property
    def effective_llm_base_url(self) -> str | None:
        """Resolve LLM base URL. Empty string resolves to None (standard OpenAI)."""
        if self.llm_base_url:
            return self.llm_base_url
        if not self.gemini_api_key and not self.nvidia_api_key and self.openai_api_key:
            return None
        return "https://integrate.api.nvidia.com/v1"

    @property
    def effective_embedding_api_key(self) -> str:
        """Resolve embedding API key with fallback order."""
        if self.embedding_api_key:
            return self.embedding_api_key
        if self.gemini_api_key:
            return self.gemini_api_key
        if self.nvidia_api_key:
            return self.nvidia_api_key
        return self.openai_api_key

    @property
    def effective_embedding_base_url(self) -> str | None:
        """Resolve embedding base URL."""
        if self.embedding_base_url:
            return self.embedding_base_url
        if not self.gemini_api_key and not self.nvidia_api_key and self.openai_api_key:
            return None
        return "https://integrate.api.nvidia.com/v1"

    @property
    def effective_embedding_model(self) -> str:
        """Resolve embedding model name."""
        if self.embedding_model:
            return self.embedding_model
        if self.gemini_api_key:
            return "gemini-embedding-001"
        if self.openai_api_key and not self.nvidia_api_key:
            return "text-embedding-3-small"
        return "gemini-embedding-001"

    # ── Retrieval ─────────────────────────────────────────────
    retrieval_top_k: int = 5
    retrieval_similarity_threshold: float = 0.25

    # ── Chunking ──────────────────────────────────────────────
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 80

    # ── Server ────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # ── Pinecone ──────────────────────────────────────────────
    pinecone_api_key: str = ""
    pinecone_index_name: str = "mal-sharia-standards"
    pinecone_namespace: str = "sharia-standards"
    pinecone_dimension: int = 2048
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # ── Derived paths ─────────────────────────────────────────
    @property
    def documents_dir(self) -> Path:
        """Path to the directory containing Sharia standard documents."""
        return Path(__file__).parent / "documents"

    @property
    def project_root(self) -> Path:
        """Project root directory (one level up from src/)."""
        return Path(__file__).parent.parent


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Using lru_cache ensures we parse .env exactly once, even if multiple
    modules call get_settings(). This avoids subtle bugs where settings
    differ across modules if the .env file changes mid-process.
    """
    return Settings()


# Convenience alias — most modules just do: from src.config import settings
settings = get_settings()
