"""
Centralized configuration management for GenAI RAG System.
All settings are loaded from environment variables or .env file.
No hardcoded values - everything is configurable.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "case_sensitive": False, "extra": "ignore"}

    # ─── Application ───────────────────────────────────────────────
    app_name: str = Field(default="GenAI RAG System", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Runtime environment")
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path for JSON structured logs (e.g. logs/app.log)")
    debug: bool = Field(default=False, description="Debug mode flag")

    # ─── LLM Configuration ────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model for generation")
    openai_embedding_model: str = Field(default="openai/text-embedding-3-small", description="Embedding model")
    openai_temperature: float = Field(default=0.0, description="LLM temperature")
    openai_max_tokens: int = Field(default=300, description="Max tokens for LLM response")
    openai_api_base: Optional[str] = Field(default=None, description="Custom OpenAI API base URL")
    openai_api_version: Optional[str] = Field(default=None, description="OpenAI API version (Azure)")

    # ─── OpenRouter (optional - alternative LLM provider) ─────────
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter API base URL")
    openrouter_model: str = Field(default="openai/gpt-oss-120b:free", description="OpenRouter model identifier")
    openrouter_embedding_model: str = Field(default="openai/text-embedding-3-small", description="OpenRouter embedding model")
    use_openrouter: bool = Field(default=False, description="Use OpenRouter instead of direct OpenAI")

    # ─── Azure OpenAI (optional) ──────────────────────────────────
    azure_openai_endpoint: Optional[str] = Field(default=None, description="Azure OpenAI endpoint")
    azure_openai_api_key: Optional[str] = Field(default=None, description="Azure OpenAI API key")
    azure_openai_deployment: Optional[str] = Field(default=None, description="Azure deployment name")
    azure_openai_api_version: str = Field(default="2024-02-15-preview", description="Azure API version")

    # ─── Vector Store ─────────────────────────────────────────────
    vector_store_type: str = Field(default="faiss", description="Vector store backend")
    faiss_index_path: str = Field(default="data/vector_store/faiss_index", description="FAISS index path")
    embedding_dimension: int = Field(default=1536, description="Embedding vector dimension")
    similarity_top_k: int = Field(default=8, description="Number of similar documents to retrieve")
    similarity_threshold: float = Field(default=0.25, description="Minimum similarity score")

    # ─── Document Ingestion ───────────────────────────────────────
    chunk_size: int = Field(default=500, description="Document chunk size in characters")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks")
    supported_file_types: str = Field(
        default=".pdf,.txt,.md,.docx,.csv",
        description="Comma-separated supported file extensions",
    )
    documents_directory: str = Field(default="data/documents", description="Source documents directory")
    max_file_size_mb: int = Field(default=50, description="Max file size in MB")

    # ─── RAG Pipeline ─────────────────────────────────────────────
    retrieval_strategy: str = Field(default="similarity", description="Retrieval strategy")
    reranking_enabled: bool = Field(default=False, description="Enable LLM-based reranking to reduce retrieval noise")
    reranker_model: str = Field(default="gpt-4o-mini", description="Model used for reranking retrieved chunks")
    reranker_top_n: int = Field(default=4, description="Number of chunks to keep after reranking (must be <= similarity_top_k)")
    reranker_relevance_threshold: float = Field(default=0.5, description="Minimum reranker relevance score (0.0-1.0). Chunks below this are dropped.")
    context_window_size: int = Field(default=4096, description="Context window for LLM")
    system_prompt_path: str = Field(default="config/prompts/system_prompt.txt", description="System prompt file")

    # ─── Evaluation Configuration ─────────────────────────────────
    eval_model: str = Field(default="gpt-4o-mini", description="Model for LLM-as-a-Judge evaluation")
    eval_embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model for eval")
    eval_threshold: float = Field(default=0.7, description="Minimum pass threshold for metrics")
    eval_dataset_path: str = Field(default="data/evaluation/eval_dataset.json", description="Evaluation dataset")
    eval_results_path: str = Field(default="data/evaluation/results", description="Evaluation results directory")
    eval_batch_size: int = Field(default=10, description="Batch size for evaluation runs")

    # ─── DeepEval Specific ────────────────────────────────────────
    deepeval_metrics: str = Field(
        default="answer_relevancy,faithfulness,contextual_precision,contextual_recall",
        description="Comma-separated DeepEval metrics to run",
    )
    deepeval_threshold: float = Field(default=0.7, description="DeepEval pass threshold")

    # ─── RAGAS Specific ───────────────────────────────────────────
    ragas_metrics: str = Field(
        default="faithfulness,answer_relevancy,context_precision,context_recall",
        description="Comma-separated RAGAS metrics to run",
    )
    ragas_threshold: float = Field(default=0.7, description="RAGAS pass threshold")

    # ─── CI/CD & Reporting ────────────────────────────────────────
    report_format: str = Field(default="json", description="Report output format (json, html, csv)")
    artifact_directory: str = Field(default="artifacts", description="CI/CD artifacts directory")
    fail_on_threshold_breach: bool = Field(default=True, description="Fail CI if metrics below threshold")

    # ─── Email Notification ───────────────────────────────────────
    email_enabled: bool = Field(default=True, description="Enable email report delivery")
    email_smtp_host: str = Field(default="smtp.gmail.com", description="SMTP server host")
    email_smtp_port: int = Field(default=587, description="SMTP server port")
    email_sender: str = Field(default="", description="Sender email address")
    email_sender_password: str = Field(default="", description="Sender email app password")
    email_recipients: str = Field(default="", description="Comma-separated recipient email addresses")
    email_subject_prefix: str = Field(default="[GenAI RAG Eval]", description="Email subject prefix")

    # ─── Evaluation Automation ────────────────────────────────────
    eval_frameworks: str = Field(
        default="deepeval,ragas",
        description="Comma-separated frameworks to run (deepeval, ragas)",
    )
    eval_run_rag_pipeline: bool = Field(default=True, description="Run RAG pipeline to generate outputs before evaluation")
    eval_custom_metrics_enabled: bool = Field(default=False, description="Enable custom production metrics (adds bias, toxicity, coherence, completeness, conciseness — increases API cost)")
    eval_auto_generate_dataset: bool = Field(default=False, description="Auto-generate eval dataset from ingested documents before evaluation")

    # ─── Testset Generation ───────────────────────────────────────
    testset_strategy: str = Field(
        default="ragas",
        description="Testset generation strategy: 'ragas' (RAGAS TestsetGenerator), 'document' (chunk-based LLM), 'hybrid' (both merged)",
    )
    testset_size: int = Field(default=20, description="Number of test cases to generate")

    # ─── Hybrid Search (BM25 + Dense) ─────────────────────────────
    hybrid_search_enabled: bool = Field(default=False, description="Enable BM25 + dense vector hybrid search")
    hybrid_search_alpha: float = Field(default=0.5, description="Weight for dense vs BM25 (1.0=pure dense, 0.0=pure BM25)")

    # ─── Response Cache ───────────────────────────────────────────
    response_cache_enabled: bool = Field(default=False, description="Enable in-memory response caching")
    response_cache_max_size: int = Field(default=256, description="Maximum number of cached responses")
    response_cache_ttl_seconds: int = Field(default=3600, description="Cache entry TTL in seconds")

    # ─── Embedding Cache ──────────────────────────────────────────
    embedding_cache_enabled: bool = Field(default=True, description="Cache embeddings to avoid re-embedding identical text")
    embedding_cache_max_size: int = Field(default=2048, description="Maximum number of cached embeddings")

    # ─── Follow-up Questions ──────────────────────────────────────
    followup_enabled: bool = Field(default=True, description="Generate suggested follow-up questions after each answer")
    followup_count: int = Field(default=3, description="Number of follow-up questions to generate")

    # ─── Metadata Enrichment ──────────────────────────────────────
    metadata_enrichment_enabled: bool = Field(default=False, description="Auto-tag documents with metadata (department, topic) using LLM during ingestion")

    # ─── FastAPI Server ───────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="FastAPI server host")
    api_port: int = Field(default=8000, description="FastAPI server port")

    # ─── Connection Pooling ───────────────────────────────────────
    llm_connection_pool_size: int = Field(default=10, description="Max HTTP connections for LLM client pool")
    llm_request_timeout: int = Field(default=60, description="LLM request timeout in seconds")

    @property
    def supported_extensions(self) -> list[str]:
        """Get list of supported file extensions."""
        return [ext.strip() for ext in self.supported_file_types.split(",")]

    @property
    def deepeval_metrics_list(self) -> list[str]:
        """Get list of DeepEval metrics."""
        return [m.strip() for m in self.deepeval_metrics.split(",")]

    @property
    def ragas_metrics_list(self) -> list[str]:
        """Get list of RAGAS metrics."""
        return [m.strip() for m in self.ragas_metrics.split(",")]

    @property
    def eval_frameworks_list(self) -> list[str]:
        """Get list of evaluation frameworks to run."""
        return [f.strip() for f in self.eval_frameworks.split(",")]

    @property
    def email_recipients_list(self) -> list[str]:
        """Get list of recipient email addresses."""
        return [e.strip() for e in self.email_recipients.split(",") if e.strip()]

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent

    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert relative path to absolute path from project root."""
        return self.project_root / relative_path


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()
