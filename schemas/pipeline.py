"""
Pipeline schemas for the RAG system.
Defines data models for queries, responses, and retrieval results.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming query request to the RAG pipeline."""

    query: str = Field(description="User's question/query")
    top_k: Optional[int] = Field(default=None, description="Override default top-k retrieval")
    filters: Optional[dict] = Field(default=None, description="Metadata filters for retrieval")
    include_sources: bool = Field(default=True, description="Include source references in response")


class RetrievalResult(BaseModel):
    """A single retrieval result from the vector store."""

    chunk_id: str = Field(description="Chunk identifier")
    content: str = Field(description="Retrieved text content")
    score: float = Field(description="Similarity score")
    source: str = Field(description="Source document path")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class RAGContext(BaseModel):
    """Context assembled for LLM generation."""

    query: str = Field(description="Original user query")
    retrieved_chunks: list[RetrievalResult] = Field(description="Retrieved document chunks")
    combined_context: str = Field(description="Combined text context for LLM")
    retrieval_time_ms: float = Field(default=0.0, description="Retrieval latency in milliseconds")


class QueryResponse(BaseModel):
    """Response from the RAG pipeline."""

    query: str = Field(description="Original user query")
    answer: str = Field(description="Generated answer")
    sources: list[RetrievalResult] = Field(default_factory=list, description="Source documents")
    context_used: str = Field(default="", description="Context provided to LLM")
    confidence_score: Optional[float] = Field(default=None, description="Response confidence")
    total_time_ms: float = Field(default=0.0, description="Total processing time in ms")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    metadata: dict = Field(default_factory=dict, description="Additional response metadata")
    followup_questions: list[str] = Field(default_factory=list, description="Suggested follow-up questions")
    token_usage: dict = Field(default_factory=dict, description="Token usage for this request")