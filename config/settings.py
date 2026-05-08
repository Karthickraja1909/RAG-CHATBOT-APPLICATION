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
    debug: bool = Field(default=False, description="Debug mode flag")

    # ─── LLM Configuration ────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model for generation")
    openai_embedding_model: str = Field(default="text-embedding-3-large", description="Embedding model")
    openai_temperature: float = Field(default=0.0, description="LLM temperature")
    openai_max_tokens: int = Field(default=256, description="Max tokens for LLM response")
    openai_api_base: Optional[str] = Field(default=None, description="Custom OpenAI API base URL")
    openai_api_version: Optional[str] = Field(default=None, description="OpenAI API version (Azure)")

    # ─── OpenRouter (optional - alternative LLM provider) ─────────
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter API base URL")
    openrouter_model: str = Field(default="openai/gpt-4o-mini", description="OpenRouter model identifier")
    openrouter_embedding_model: str = Field(default="openai/text-embedding-3-large", description="OpenRouter embedding model")
    use_openrouter: bool = Field(default=False, description="Use OpenRouter instead of direct OpenAI")

    # ─── Azure OpenAI (optional) ──────────────────────────────────
    azure_openai_endpoint: Optional[str] = Field(default=None, description="Azure OpenAI endpoint")
    azure_openai_api_key: Optional[str] = Field(default=None, description="Azure OpenAI API key")
    azure_openai_deployment: Optional[str] = Field(default=None, description="Azure deployment name")
    azure_openai_api_version: str = Field(default="2024-02-15-preview", description="Azure API version")

    # ─── Vector Store ─────────────────────────────────────────────
    vector_store_type: str = Field(default="faiss", description="Vector store backend")
    faiss_index_path: str = Field(default="data/vector_store/faiss_index", description="FAISS index path")
    embedding_dimension: int = Field(default=3072, description="Embedding vector dimension")
    similarity_top_k: int = Field(default=5, description="Number of similar documents to retrieve")
    similarity_threshold: float = Field(default=0.3, description="Minimum similarity score")

    # ─── Document Ingestion ───────────────────────────────────────
    chunk_size: int = Field(default=1000, description="Document chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    supported_file_types: str = Field(
        default=".pdf,.txt,.md,.docx,.csv",
        description="Comma-separated supported file extensions",
    )
    documents_directory: str = Field(default="data/documents", description="Source documents directory")
    max_file_size_mb: int = Field(default=50, description="Max file size in MB")

    # ─── RAG Pipeline ─────────────────────────────────────────────
    retrieval_strategy: str = Field(default="similarity", description="Retrieval strategy")
    reranking_enabled: bool = Field(default=False, description="Enable reranking")
    context_window_size: int = Field(default=4096, description="Context window for LLM")
    system_prompt_path: str = Field(default="config/prompts/system_prompt.txt", description="System prompt file")

    # ─── Evaluation Configuration ─────────────────────────────────
    eval_model: str = Field(default="gpt-4o-mini", description="Model for LLM-as-a-Judge evaluation")
    eval_embedding_model: str = Field(default="text-embedding-3-large", description="Embedding model for eval")
    eval_threshold: float = Field(default=0.7, description="Minimum pass threshold for metrics")
    eval_dataset_path: str = Field(default="data/evaluation/eval_dataset.json", description="Evaluation dataset")
    eval_results_path: str = Field(default="data/evaluation/results", description="Evaluation results directory")
    eval_batch_size: int = Field(default=10, description="Batch size for evaluation runs")

    # ─── DeepEval Specific ────────────────────────────────────────
    deepeval_metrics: str = Field(
        default="answer_relevancy,faithfulness,contextual_precision,contextual_recall,hallucination",
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
    eval_custom_metrics_enabled: bool = Field(default=True, description="Enable custom production metrics")
    eval_auto_generate_dataset: bool = Field(default=False, description="Auto-generate eval dataset from ingested documents before evaluation")

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