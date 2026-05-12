"""
Evaluation schemas for the RAG system.
Defines data models for evaluation samples, results, and reports.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class EvaluationSample(BaseModel):
    """A single evaluation test case."""

    sample_id: str = Field(description="Unique sample identifier")
    user_input: str = Field(description="User query/question")
    actual_output: Optional[str] = Field(default=None, description="LLM generated response")
    expected_output: Optional[str] = Field(default=None, description="Ground truth/reference answer")
    context: Optional[list[str]] = Field(default=None, description="Ground truth context")
    retrieval_context: Optional[list[str]] = Field(default=None, description="Retrieved context from RAG")

    # ─── Enterprise Metadata Fields ───────────────────────────────
    difficulty: Optional[str] = Field(default=None, description="Question difficulty: easy, medium, hard")
    question_type: Optional[str] = Field(default=None, description="Question type: simple, reasoning, multi_context")
    source_strategy: Optional[str] = Field(default=None, description="Generation strategy: manual, document_grounded, ragas_synthetic, production")
    auto_generated: bool = Field(default=False, description="Whether this sample was auto-generated")
    human_reviewed: bool = Field(default=False, description="Whether this sample has been human-reviewed")


class EvaluationDataset(BaseModel):
    """Collection of evaluation samples."""

    dataset_id: str = Field(description="Unique dataset identifier")
    name: str = Field(description="Dataset name")
    description: str = Field(default="", description="Dataset description")
    version: Optional[str] = Field(default=None, description="Dataset version")
    samples: list[EvaluationSample] = Field(default_factory=list, description="Evaluation samples")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")

    @property
    def size(self) -> int:
        """Number of samples in the dataset."""
        return len(self.samples)


class MetricResult(BaseModel):
    """Result of a single metric evaluation."""

    metric_name: str = Field(description="Name of the metric")
    score: float = Field(description="Metric score (0.0 to 1.0)")
    threshold: float = Field(description="Pass/fail threshold")
    passed: bool = Field(description="Whether the metric passed")
    reason: Optional[str] = Field(default=None, description="Explanation for the score")


class EvaluationResult(BaseModel):
    """Result for a single evaluation sample across all metrics."""

    sample_id: str = Field(description="Reference to evaluation sample")
    metrics: list[MetricResult] = Field(default_factory=list, description="Metric results")
    overall_passed: bool = Field(default=False, description="Whether all metrics passed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Evaluation timestamp")

    @property
    def average_score(self) -> float:
        """Average score across all metrics."""
        if not self.metrics:
            return 0.0
        return sum(m.score for m in self.metrics) / len(self.metrics)


class EvaluationReport(BaseModel):
    """Complete evaluation report across all samples."""

    report_id: str = Field(description="Unique report identifier")
    framework: str = Field(description="Evaluation framework (deepeval/ragas)")
    dataset_id: str = Field(description="Reference to evaluation dataset")
    results: list[EvaluationResult] = Field(default_factory=list, description="Per-sample results")
    summary: dict = Field(default_factory=dict, description="Aggregate metrics summary")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Report creation timestamp")
    duration_seconds: float = Field(default=0.0, description="Total evaluation duration")
    config: dict = Field(default_factory=dict, description="Configuration used for evaluation")

    @property
    def total_samples(self) -> int:
        """Total samples evaluated."""
        return len(self.results)

    @property
    def passed_samples(self) -> int:
        """Number of samples that passed all metrics."""
        return sum(1 for r in self.results if r.overall_passed)

    @property
    def pass_rate(self) -> float:
        """Percentage of samples that passed."""
        if not self.results:
            return 0.0
        return self.passed_samples / self.total_samples


