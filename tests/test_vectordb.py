"""
Unit tests for vector store (FAISS).
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from schemas.documents import DocumentChunk, DocumentMetadata
from utils.exceptions import VectorStoreError


class TestFAISSVectorStore:
    """Tests for FAISSVectorStore class."""

    @pytest.fixture(autouse=True)
    def patch_settings(self):
        """Patch settings to use 1536 dimension for tests."""
        with patch("vectordb.faiss_store.get_settings") as mock_settings:
            settings = MagicMock()
            settings.embedding_dimension = 1536
            settings.faiss_index_path = "test_index"
            mock_settings.return_value = settings
            yield

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service."""
        mock = MagicMock()
        mock.embed_text.return_value = [0.1] * 1536
        mock.embed_texts.return_value = [[0.1] * 1536, [0.2] * 1536]
        return mock

    @pytest.fixture
    def sample_chunks(self):
        """Create sample document chunks."""
        return [
            DocumentChunk(
                chunk_id="chunk_001",
                document_id="doc_001",
                content="RAG combines retrieval with generation for accurate responses.",
                metadata=DocumentMetadata(
                    source="doc1.txt", file_type=".txt", chunk_index=0, total_chunks=1
                ),
            ),
            DocumentChunk(
                chunk_id="chunk_002",
                document_id="doc_001",
                content="FAISS enables efficient vector similarity search at scale.",
                metadata=DocumentMetadata(
                    source="doc1.txt", file_type=".txt", chunk_index=1, total_chunks=2
                ),
            ),
        ]

    @patch("vectordb.faiss_store.EmbeddingService")
    def test_add_chunks(self, mock_emb_class, mock_embedding_service, sample_chunks, tmp_path):
        """Test adding chunks to vector store."""
        mock_emb_class.return_value = mock_embedding_service

        from vectordb.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(
            embedding_service=mock_embedding_service,
            index_path=str(tmp_path / "test_index"),
        )

        count = store.add_chunks(sample_chunks)

        assert count == 2
        assert store._index.ntotal == 2
        mock_embedding_service.embed_texts.assert_called_once()

    @patch("vectordb.faiss_store.EmbeddingService")
    def test_search(self, mock_emb_class, mock_embedding_service, sample_chunks, tmp_path):
        """Test searching the vector store."""
        mock_emb_class.return_value = mock_embedding_service

        from vectordb.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(
            embedding_service=mock_embedding_service,
            index_path=str(tmp_path / "test_index"),
        )
        store.add_chunks(sample_chunks)

        results = store.search("What is RAG?", top_k=2, threshold=0.0)

        assert len(results) > 0
        assert results[0].content in [c.content for c in sample_chunks]

    @patch("vectordb.faiss_store.EmbeddingService")
    def test_search_empty_store(self, mock_emb_class, mock_embedding_service, tmp_path):
        """Test searching an empty store returns empty results."""
        mock_emb_class.return_value = mock_embedding_service

        from vectordb.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(
            embedding_service=mock_embedding_service,
            index_path=str(tmp_path / "test_index"),
        )

        results = store.search("query")
        assert results == []

    @patch("vectordb.faiss_store.EmbeddingService")
    def test_get_stats(self, mock_emb_class, mock_embedding_service, sample_chunks, tmp_path):
        """Test getting vector store statistics."""
        mock_emb_class.return_value = mock_embedding_service

        from vectordb.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(
            embedding_service=mock_embedding_service,
            index_path=str(tmp_path / "test_index"),
        )
        store.add_chunks(sample_chunks)

        stats = store.get_stats()

        assert stats["total_vectors"] == 2
        assert stats["dimension"] == 1536
        assert stats["total_documents"] == 1

    @patch("vectordb.faiss_store.EmbeddingService")
    def test_clear_store(self, mock_emb_class, mock_embedding_service, sample_chunks, tmp_path):
        """Test clearing the vector store."""
        mock_emb_class.return_value = mock_embedding_service

        from vectordb.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(
            embedding_service=mock_embedding_service,
            index_path=str(tmp_path / "test_index"),
        )
        store.add_chunks(sample_chunks)
        store.clear()

        assert store._index.ntotal == 0
        assert store._metadata_store == []

    @patch("vectordb.faiss_store.EmbeddingService")
    def test_delete_by_document_id(self, mock_emb_class, mock_embedding_service, sample_chunks, tmp_path):
        """Test deleting chunks by document ID."""
        mock_emb_class.return_value = mock_embedding_service

        from vectordb.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(
            embedding_service=mock_embedding_service,
            index_path=str(tmp_path / "test_index"),
        )
        store.add_chunks(sample_chunks)

        removed = store.delete_by_document_id("doc_001")

        assert removed == 2
        assert store._index.ntotal == 0