"""
Unit tests for document ingestion pipeline.
"""

import pytest
from pathlib import Path

from ingest.document_loader import DocumentLoader
from ingest.text_splitter import TextSplitter
from ingest.ingestion_pipeline import IngestionPipeline
from schemas.documents import Document, DocumentMetadata
from utils.exceptions import DocumentIngestionError


class TestDocumentLoader:
    """Tests for DocumentLoader class."""

    def test_load_text_file(self, temp_documents_dir):
        """Test loading a plain text file."""
        loader = DocumentLoader()
        doc = loader.load_document(temp_documents_dir / "test_doc.txt")

        assert doc.document_id is not None
        assert "artificial intelligence" in doc.content
        assert doc.metadata.file_type == ".txt"
        assert doc.metadata.source == str(temp_documents_dir / "test_doc.txt")

    def test_load_markdown_file(self, temp_documents_dir):
        """Test loading a markdown file."""
        loader = DocumentLoader()
        doc = loader.load_document(temp_documents_dir / "test_doc.md")

        assert doc.document_id is not None
        assert "Machine Learning" in doc.content
        assert doc.metadata.file_type == ".md"

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading a non-existent file raises error."""
        loader = DocumentLoader()

        with pytest.raises(DocumentIngestionError):
            loader.load_document("/nonexistent/path/file.txt")

    def test_load_unsupported_extension_raises_error(self, tmp_path):
        """Test that unsupported file extensions raise error."""
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("content")

        loader = DocumentLoader()
        with pytest.raises(DocumentIngestionError):
            loader.load_document(unsupported_file)

    def test_load_directory(self, temp_documents_dir):
        """Test loading all documents from a directory."""
        loader = DocumentLoader()
        docs = loader.load_directory(temp_documents_dir)

        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)

    def test_load_empty_directory(self, tmp_path):
        """Test loading from empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        loader = DocumentLoader()
        docs = loader.load_directory(empty_dir)
        assert docs == []


class TestTextSplitter:
    """Tests for TextSplitter class."""

    def test_split_short_document(self, sample_documents):
        """Test that short documents remain as single chunks."""
        splitter = TextSplitter(chunk_size=5000, chunk_overlap=200)
        chunks = splitter.split_document(sample_documents[0])

        assert len(chunks) >= 1
        assert chunks[0].document_id == "doc_001"

    def test_split_produces_chunks_with_metadata(self, sample_documents):
        """Test that chunks preserve document metadata."""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_document(sample_documents[1])

        for chunk in chunks:
            assert chunk.metadata.source == "test_docs/faiss_guide.txt"
            assert chunk.metadata.file_type == ".txt"
            assert chunk.chunk_id is not None

    def test_chunk_overlap_less_than_size(self):
        """Test that overlap must be less than chunk size."""
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=100, chunk_overlap=150)

    def test_split_empty_document(self):
        """Test splitting an empty document."""
        doc = Document(
            document_id="empty",
            content="",
            metadata=DocumentMetadata(source="empty.txt", file_type=".txt"),
        )
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_document(doc)
        assert chunks == []

    def test_split_multiple_documents(self, sample_documents):
        """Test splitting multiple documents."""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_documents(sample_documents)

        assert len(chunks) > 0
        doc_ids = set(c.document_id for c in chunks)
        assert "doc_001" in doc_ids
        assert "doc_002" in doc_ids


class TestIngestionPipeline:
    """Tests for IngestionPipeline class."""

    def test_ingest_file(self, temp_documents_dir):
        """Test ingesting a single file."""
        pipeline = IngestionPipeline()
        chunks = pipeline.ingest_file(temp_documents_dir / "test_doc.txt")

        assert len(chunks) > 0
        assert all(c.content for c in chunks)

    def test_ingest_directory(self, temp_documents_dir):
        """Test ingesting all files in a directory."""
        pipeline = IngestionPipeline()
        chunks = pipeline.ingest_directory(temp_documents_dir)

        assert len(chunks) > 0

    def test_ingest_with_custom_splitter(self, temp_documents_dir):
        """Test ingestion with custom text splitter parameters."""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        pipeline = IngestionPipeline(text_splitter=splitter)
        chunks = pipeline.ingest_file(temp_documents_dir / "test_doc.txt")

        assert len(chunks) > 1  # Small chunk size should produce multiple chunks