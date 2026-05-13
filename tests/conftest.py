"""
Pytest configuration and fixtures for GenAI RAG System tests.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Set test environment variables to avoid using real API keys."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("FAISS_INDEX_PATH", "data/test_vector_store/faiss_index")
    monkeypatch.setenv("DOCUMENTS_DIRECTORY", "data/test_documents")
    monkeypatch.setenv("EVAL_DATASET_PATH", "data/evaluation/eval_dataset.json")
    monkeypatch.setenv("EVAL_RESULTS_PATH", "data/evaluation/test_results")


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    from schemas.documents import Document, DocumentMetadata

    return [
        Document(
            document_id="doc_001",
            content="RAG (Retrieval Augmented Generation) is an AI technique that combines retrieval and generation. "
                    "It retrieves relevant documents and uses them as context for generating accurate responses.",
            metadata=DocumentMetadata(
                source="test_docs/rag_intro.txt",
                file_type=".txt",
                file_size_bytes=200,
            ),
        ),
        Document(
            document_id="doc_002",
            content="FAISS is a library for efficient similarity search. It supports various index types "
                    "including flat indexes for exact search and IVF indexes for approximate search. "
                    "It can handle billions of vectors with GPU acceleration.",
            metadata=DocumentMetadata(
                source="test_docs/faiss_guide.txt",
                file_type=".txt",
                file_size_bytes=250,
            ),
        ),
    ]


@pytest.fixture
def sample_eval_dataset():
    """Provide a sample evaluation dataset for testing."""
    from schemas.evaluation import EvaluationDataset, EvaluationSample

    return EvaluationDataset(
        dataset_id="test_dataset_001",
        name="Test Dataset",
        description="Dataset for unit tests",
        samples=[
            EvaluationSample(
                sample_id="test_sample_001",
                user_input="What is RAG?",
                actual_output="RAG is Retrieval Augmented Generation, a technique combining retrieval with generation.",
                expected_output="RAG combines document retrieval with text generation for accurate responses.",
                context=["RAG is an AI technique that combines retrieval and generation."],
                retrieval_context=["RAG retrieves relevant documents and uses them as context for generation."],
            ),
            EvaluationSample(
                sample_id="test_sample_002",
                user_input="How does FAISS work?",
                actual_output="FAISS performs efficient similarity search using vector indexes.",
                expected_output="FAISS is a library for efficient similarity search of dense vectors.",
                context=["FAISS supports various index types for similarity search."],
                retrieval_context=["FAISS handles similarity search with flat and IVF indexes."],
            ),
        ],
    )


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing without API calls."""
    mock_client = MagicMock()

    # Mock embeddings
    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1] * 1536
    mock_embeddings_response = MagicMock()
    mock_embeddings_response.data = [mock_embedding]
    mock_client.embeddings.create.return_value = mock_embeddings_response

    # Mock chat completions
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a test response from the LLM."
    mock_usage = MagicMock()
    mock_usage.total_tokens = 100
    mock_usage.prompt_tokens = 60
    mock_usage.completion_tokens = 40
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = mock_usage
    mock_client.chat.completions.create.return_value = mock_completion

    return mock_client


@pytest.fixture
def temp_documents_dir(tmp_path):
    """Create a temporary documents directory with test files."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    # Create test text file
    (docs_dir / "test_doc.txt").write_text(
        "This is a test document about artificial intelligence. "
        "AI is transforming many industries including healthcare and finance.",
        encoding="utf-8",
    )

    # Create test markdown file
    (docs_dir / "test_doc.md").write_text(
        "# Machine Learning\n\n"
        "Machine learning is a subset of artificial intelligence. "
        "It enables systems to learn from data without explicit programming.",
        encoding="utf-8",
    )

    return docs_dir


@pytest.fixture
def eval_dataset_file(tmp_path):
    """Create a temporary evaluation dataset file."""
    dataset = {
        "dataset_id": "test_ds_001",
        "name": "Test Evaluation Dataset",
        "description": "For testing",
        "samples": [
            {
                "sample_id": "s1",
                "user_input": "What is Python?",
                "actual_output": "Python is a programming language.",
                "expected_output": "Python is a high-level programming language.",
                "context": ["Python is a versatile programming language."],
                "retrieval_context": ["Python is used for web development and data science."],
            }
        ],
    }

    file_path = tmp_path / "eval_dataset.json"
    file_path.write_text(json.dumps(dataset), encoding="utf-8")
    return file_path