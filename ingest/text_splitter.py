"""
Text splitter module for GenAI RAG System.
Handles splitting documents into chunks with configurable size and overlap.
"""

import hashlib
from typing import Optional

from config.settings import get_settings
from schemas.documents import Document, DocumentChunk, DocumentMetadata
from utils.logger import get_logger

logger = get_logger(__name__)


class TextSplitter:
    """Splits documents into smaller chunks for vector storage."""

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
        Split text into chunks using recursive character splitting.
        Tries to split on paragraph boundaries first, then sentences, then characters.
        """
        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._recursive_split(text, separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using a hierarchy of separators."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        separator = separators[0]
        remaining_separators = separators[1:] if len(separators) > 1 else [""]

        if separator == "":
            # Last resort: split by character count
            return self._split_by_characters(text)

        splits = text.split(separator)
        chunks = []
        current_chunk = ""

        for split in splits:
            candidate = f"{current_chunk}{separator}{split}" if current_chunk else split

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                if len(split) > self.chunk_size:
                    # Recursively split oversized pieces
                    sub_chunks = self._recursive_split(split, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Apply overlap
        return self._apply_overlap(chunks)

    def _split_by_characters(self, text: str) -> list[str]:
        """Split text by character count with overlap."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
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