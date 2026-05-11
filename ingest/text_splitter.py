"""
Text splitter module for GenAI RAG System.
Structure-aware splitting that preserves heading context in every chunk.
"""

import hashlib
import re
from typing import Optional

from config.settings import get_settings
from schemas.documents import Document, DocumentChunk, DocumentMetadata
from utils.logger import get_logger

logger = get_logger(__name__)

# Regex to detect markdown-style headings
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


class TextSplitter:
    """Structure-aware text splitter that keeps heading context in every chunk."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )

    def split_document(self, document: Document) -> list[DocumentChunk]:
        """
        Split a document into chunks.

        Args:
            document: Document to split.

        Returns:
            List of DocumentChunk objects.
        """
        text = document.content
        if not text.strip():
            logger.warning(f"Empty document: {document.document_id}")
            return []

        chunks_text = self._split_text(text)
        total_chunks = len(chunks_text)
        chunks = []

        for idx, chunk_text in enumerate(chunks_text):
            chunk_id = self._generate_chunk_id(document.document_id, idx)
            metadata = DocumentMetadata(
                source=document.metadata.source,
                file_type=document.metadata.file_type,
                file_size_bytes=document.metadata.file_size_bytes,
                created_at=document.metadata.created_at,
                chunk_index=idx,
                total_chunks=total_chunks,
                custom_metadata=document.metadata.custom_metadata.copy(),
            )

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                content=chunk_text,
                metadata=metadata,
            )
            chunks.append(chunk)

        logger.info(
            f"Split document {document.document_id} into {total_chunks} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def split_documents(self, documents: list[Document]) -> list[DocumentChunk]:
        """Split multiple documents into chunks."""
        all_chunks = []
        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"Split {len(documents)} documents into {len(all_chunks)} total chunks")
        return all_chunks

    def _split_text(self, text: str) -> list[str]:
        """
        Split text into chunks with section heading context.
        Each chunk is prefixed with its parent heading so vector search
        can match on section topic even for body-text chunks.
        """
        sections = self._split_into_sections(text)
        chunks = []

        for heading, body in sections:
            heading_prefix = f"{heading}\n" if heading else ""
            prefix_len = len(heading_prefix)
            effective_size = self.chunk_size - prefix_len

            if effective_size < 50:
                effective_size = self.chunk_size

            sub_chunks = self._sentence_split(body, effective_size)

            for sc in sub_chunks:
                chunk_text = f"{heading_prefix}{sc}".strip()
                if chunk_text:
                    chunks.append(chunk_text)

        # Apply overlap between adjacent chunks
        return self._apply_overlap(chunks)

    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """
        Split text by markdown headings into (heading, body) pairs.
        If no headings found, returns one section with empty heading.
        """
        heading_positions = [(m.start(), m.group(0)) for m in _HEADING_RE.finditer(text)]

        if not heading_positions:
            return [("", text)]

        sections = []
        # Content before first heading
        if heading_positions[0][0] > 0:
            pre_text = text[: heading_positions[0][0]].strip()
            if pre_text:
                sections.append(("", pre_text))

        for i, (pos, heading) in enumerate(heading_positions):
            end_pos = heading_positions[i + 1][0] if i + 1 < len(heading_positions) else len(text)
            body = text[pos + len(heading) : end_pos].strip()
            if body:
                sections.append((heading.strip(), body))

        return sections if sections else [("", text)]

    def _sentence_split(self, text: str, max_size: int) -> list[str]:
        """
        Split text respecting sentence boundaries.
        Tries paragraph → sentence → word → character boundaries in order.
        """
        if len(text) <= max_size:
            return [text.strip()] if text.strip() else []

        # Try splitting by paragraphs first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            return self._merge_splits(paragraphs, max_size)

        # Try splitting by sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            return self._merge_splits(sentences, max_size)

        # Fall back to word boundaries
        words = text.split()
        if len(words) > 1:
            return self._merge_splits(words, max_size, separator=" ")

        # Last resort: character split
        return self._split_by_characters(text, max_size)

    def _merge_splits(
        self, parts: list[str], max_size: int, separator: str = "\n\n"
    ) -> list[str]:
        """Merge small parts into chunks up to max_size."""
        chunks = []
        current = ""

        for part in parts:
            candidate = f"{current}{separator}{part}" if current else part
            if len(candidate) <= max_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(part) > max_size:
                    # Recursively split oversized part
                    chunks.extend(self._sentence_split(part, max_size))
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _split_by_characters(self, text: str, max_size: Optional[int] = None) -> list[str]:
        """Split text by character count with overlap."""
        max_size = max_size or self.chunk_size
        chunks = []
        start = 0

        while start < len(text):
            end = start + max_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.chunk_overlap

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply overlap between adjacent chunks."""
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            combined = f"{overlap_text} {chunks[i]}"

            if len(combined) <= self.chunk_size:
                overlapped.append(combined)
            else:
                overlapped.append(chunks[i])

        return overlapped

    def _generate_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """Generate a deterministic chunk ID."""
        key = f"{document_id}:chunk:{chunk_index}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]