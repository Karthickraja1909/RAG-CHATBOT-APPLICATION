"""
Unit tests for the RAG pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch

from schemas.pipeline import QueryRequest, QueryResponse, RetrievalResult
from utils.exceptions import PipelineError


class TestRAGPipeline:
    """Tests for RAGPipeline class."""

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store for testing."""
        mock = MagicMock()
        mock.search.return_value = [
            RetrievalResult(
                chunk_id="chunk_001",
                content="RAG combines retrieval and generation for accurate answers.",
                score=0.92,
                source="doc1.txt",
                metadata={"document_id": "doc_001"},
            ),
            RetrievalResult(
                chunk_id="chunk_002",
                content="The retrieval step finds relevant documents from a knowledge base.",
                score=0.85,
                source="doc2.txt",
                metadata={"document_id": "doc_002"},
            ),
        ]
        return mock

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for testing."""
        mock = MagicMock()
        mock.generate_with_context.return_value = (
            "RAG is a technique that retrieves relevant documents and uses them "
            "as context for generating accurate responses."
        )
        mock.generate.return_value = "Follow-up question 1\nFollow-up question 2\nFollow-up question 3"
        mock.model = "gpt-4o-mini"
        mock.token_usage.summary = {
            "prompt_tokens": 60,
            "completion_tokens": 40,
            "total_tokens": 100,
            "request_count": 1,
        }
        return mock

    def test_query_returns_response(self, mock_vector_store, mock_llm_service):
        """Test that query returns a proper response."""
        from pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            llm_service=mock_llm_service,
            system_prompt="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        )

        request = QueryRequest(query="What is RAG?")
        response = pipeline.query(request)

        assert isinstance(response, QueryResponse)
        assert response.query == "What is RAG?"
        assert len(response.answer) > 0
        assert len(response.sources) == 2
        assert response.total_time_ms > 0

    def test_query_simple(self, mock_vector_store, mock_llm_service):
        """Test the simplified query interface."""
        from pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            llm_service=mock_llm_service,
            system_prompt="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        )

        answer = pipeline.query_simple("What is RAG?")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_query_with_empty_results(self, mock_llm_service):
        """Test query when vector store returns no results."""
        from pipeline.rag_pipeline import RAGPipeline

        empty_store = MagicMock()
        empty_store.search.return_value = []

        pipeline = RAGPipeline(
            vector_store=empty_store,
            llm_service=mock_llm_service,
            system_prompt="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        )

        request = QueryRequest(query="Unknown topic")
        response = pipeline.query(request)

        assert "don't have enough information" in response.answer

    def test_query_includes_metadata(self, mock_vector_store, mock_llm_service):
        """Test that response includes proper metadata."""
        from pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            llm_service=mock_llm_service,
            system_prompt="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        )

        request = QueryRequest(query="Tell me about RAG")
        response = pipeline.query(request)

        assert "retrieval_time_ms" in response.metadata
        assert "generation_time_ms" in response.metadata
        assert "chunks_retrieved" in response.metadata

    def test_query_respects_top_k(self, mock_vector_store, mock_llm_service):
        """Test that custom top_k is passed to vector store."""
        from pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            llm_service=mock_llm_service,
            system_prompt="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        )

        request = QueryRequest(query="Test query", top_k=3)
        pipeline.query(request)

        mock_vector_store.search.assert_called_once_with(query="Test query", top_k=3)