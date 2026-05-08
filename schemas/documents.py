"""
Document schemas for the RAG system.
Defines the data models for documents, chunks, and metadata.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata associated with a document."""

    source: str = Field(description="Source file path or URL")
    file_type: str = Field(description="File extension/type")
    file_size_bytes: int = Field(default=0, description="File size in bytes")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Ingestion timestamp")
    page_number: Optional[int] = Field(default=None, description="Page number (for PDFs)")
    chunk_index: Optional[int] = Field(default=None, description="Chunk index within document")
    total_chunks: Optional[int] = Field(default=None, description="Total chunks in document")
    custom_metadata: dict = Field(default_factory=dict, description="Additional metadata")


class Document(BaseModel):
    """Represents a full document before chunking."""

    document_id: str = Field(description="Unique document identifier")
    content: str = Field(description="Full document text content")
    metadata: DocumentMetadata = Field(description="Document metadata")


class DocumentChunk(BaseModel):
    """Represents a chunk of a document after splitting."""

    chunk_id: str = Field(description="Unique chunk identifier")
    document_id: str = Field(description="Parent document identifier")
    content: str = Field(description="Chunk text content")
    metadata: DocumentMetadata = Field(description="Chunk metadata")
    embedding: Optional[list[float]] = Field(default=None, description="Vector embedding")