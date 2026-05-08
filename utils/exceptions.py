"""
Custom exception hierarchy for GenAI RAG System.
Provides specific exceptions for different system components.
"""


class RAGSystemError(Exception):
    """Base exception for all RAG system errors."""

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(RAGSystemError):
    """Raised when configuration is invalid or missing."""
    pass


class DocumentIngestionError(RAGSystemError):
    """Raised when document ingestion fails."""
    pass


class VectorStoreError(RAGSystemError):
    """Raised when vector store operations fail."""
    pass


class LLMError(RAGSystemError):
    """Raised when LLM API calls fail."""
    pass


class EvaluationError(RAGSystemError):
    """Raised when evaluation pipeline fails."""
    pass


class PipelineError(RAGSystemError):
    """Raised when RAG pipeline execution fails."""
    pass


class RetrievalError(RAGSystemError):
    """Raised when document retrieval fails."""
    pass

