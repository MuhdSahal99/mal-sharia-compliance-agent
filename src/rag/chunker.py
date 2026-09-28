"""
Document chunking with section-aware splitting.

Design decision — section-aware vs. naive fixed-size chunking:

Naive fixed-size chunking (split every N tokens) is simple but breaks
regulatory text mid-sentence and mid-section, destroying semantic coherence.
For compliance use cases, this is especially damaging because a ruling often
depends on context from the section header and preceding paragraphs.

Our strategy:
1. Parse markdown section boundaries (## headers)
2. Within each section, split on paragraph boundaries
3. Merge small paragraphs into chunks up to CHUNK_SIZE_TOKENS
4. Apply CHUNK_OVERLAP_TOKENS overlap between chunks
5. Attach metadata (source document, section title) to each chunk

This preserves semantic coherence while keeping chunks small enough
for accurate retrieval. The section title in metadata lets the agent
cite specific standards in its reasoning.
"""

from dataclasses import dataclass, field
import re
import tiktoken

from src.config import settings
from src.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentChunk:
    """A single chunk of text with provenance metadata."""

    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Deterministic ID for deduplication during re-ingestion."""
        source = self.metadata.get("source", "unknown")
        section = self.metadata.get("section", "unknown")
        # Use a hash of the text content for uniqueness within a section
        text_hash = hash(self.text) % (10**8)
        return f"{source}::{section}::{text_hash}"


class DocumentChunker:
    """
    Splits markdown documents into semantically coherent chunks.

    The chunker respects document structure by splitting on section headers
    first, then on paragraph boundaries within sections. This ensures each
    chunk contains text from a single logical section, making retrieval
    results more relevant and citation more accurate.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size_tokens,
        chunk_overlap: int = settings.chunk_overlap_tokens,
        model_name: str = "cl100k_base",  # Tokenizer used by GPT-4 / text-embedding-3
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._tokenizer = tiktoken.get_encoding(model_name)

    def chunk_document(
        self, text: str, source_name: str
    ) -> list[DocumentChunk]:
        """
        Split a markdown document into chunks with metadata.

        Args:
            text: The full document text in markdown format.
            source_name: Filename or identifier for provenance tracking.

        Returns:
            List of DocumentChunk objects with metadata.
        """
        sections = self._split_into_sections(text)
        all_chunks: list[DocumentChunk] = []

        for section_title, section_text in sections:
            paragraphs = self._split_into_paragraphs(section_text)
            merged = self._merge_paragraphs(paragraphs)

            for chunk_text in merged:
                # Skip chunks that are too small to be useful
                token_count = len(self._tokenizer.encode(chunk_text))
                if token_count < 20:
                    continue

                chunk = DocumentChunk(
                    text=chunk_text.strip(),
                    metadata={
                        "source": source_name,
                        "section": section_title,
                        "token_count": str(token_count),
                    },
                )
                all_chunks.append(chunk)

        logger.info(
            "chunked_document",
            source=source_name,
            total_chunks=len(all_chunks),
            sections_found=len(sections),
        )
        return all_chunks

    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """
        Split markdown text by ## headers.

        Returns a list of (section_title, section_body) tuples.
        Text before the first header gets the title "Preamble".
        """
        # Match markdown headers at any level (##, ###, etc.)
        pattern = r"^(#{1,4})\s+(.+)$"
        lines = text.split("\n")

        sections: list[tuple[str, str]] = []
        current_title = "Preamble"
        current_lines: list[str] = []

        for line in lines:
            match = re.match(pattern, line)
            if match:
                # Save the previous section
                if current_lines:
                    body = "\n".join(current_lines).strip()
                    if body:
                        sections.append((current_title, body))

                current_title = match.group(2).strip()
                current_lines = []
            else:
                current_lines.append(line)

        # Don't forget the last section
        if current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_title, body))

        return sections

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text on double newlines to get paragraph-level segments."""
        paragraphs = re.split(r"\n{2,}", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _merge_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """
        Merge consecutive paragraphs into chunks of ~CHUNK_SIZE tokens.

        Applies overlap by re-including the last CHUNK_OVERLAP tokens
        worth of text from the previous chunk at the start of the next.
        """
        if not paragraphs:
            return []

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = len(self._tokenizer.encode(para))

            # If a single paragraph exceeds chunk_size, it becomes its own chunk
            if para_tokens > self._chunk_size:
                # Flush what we have
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split the oversized paragraph by sentences
                chunks.extend(self._split_oversized_paragraph(para))
                continue

            # If adding this paragraph would exceed the limit, flush
            if current_tokens + para_tokens > self._chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))

                # Apply overlap: carry the last paragraph(s) forward
                overlap_chunk, overlap_tokens = self._get_overlap(current_chunk)
                current_chunk = overlap_chunk
                current_tokens = overlap_tokens

            current_chunk.append(para)
            current_tokens += para_tokens

        # Flush remaining
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _split_oversized_paragraph(self, text: str) -> list[str]:
        """
        Handle paragraphs that exceed chunk_size by splitting on sentence
        boundaries (period followed by space).
        """
        sentences = re.split(r"(?<=\.)\s+", text)
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = len(self._tokenizer.encode(sentence))
            if current_tokens + sent_tokens > self._chunk_size and current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(sentence)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _get_overlap(
        self, paragraphs: list[str]
    ) -> tuple[list[str], int]:
        """
        Select paragraphs from the end of the list that fit within
        CHUNK_OVERLAP tokens, to carry forward as overlap.
        """
        overlap: list[str] = []
        overlap_tokens = 0

        for para in reversed(paragraphs):
            para_tokens = len(self._tokenizer.encode(para))
            if overlap_tokens + para_tokens > self._chunk_overlap:
                break
            overlap.insert(0, para)
            overlap_tokens += para_tokens

        return overlap, overlap_tokens
