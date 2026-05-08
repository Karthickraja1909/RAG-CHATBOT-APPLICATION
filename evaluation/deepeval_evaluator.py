"""
DeepEval evaluator module for GenAI RAG System.
Production-level LLM evaluation using DeepEval framework.
Supports: AnswerRelevancy, Faithfulness, ContextualPrecision, ContextualRecall, Hallucination, GEval.
"""

import time
import uuid
from typing import Optional

from config.settings import get_settings
from schemas.evaluation import EvaluationDataset, EvaluationReport, EvaluationResult, EvaluationSample, MetricResult
from utils.exceptions import EvaluationError
from utils.logger import get_logger

logger = get_logger(__name__)


class DeepEvalEvaluator:
    """
    Production-level evaluator using DeepEval framework.
    Evaluates RAG pipeline outputs with configurable metrics.
    """

    def __init__(
        self,
        metrics: Optional[list[str]] = None,
        threshold: Optional[float] = None,
        model: Optional[str] = None,
    ):
        self.settings = get_settings()
        self.metrics_config = metrics or self.settings.deepeval_metrics_list
        self.threshold = threshold or self.settings.deepeval_threshold
        self.model = model or self.settings.eval_model

        self._metrics = None
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize DeepEval metric instances based on configuration."""
        try:
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric,
                ContextualRelevancyMetric,
                FaithfulnessMetric,
                GEval,
                HallucinationMetric,
            )
            from deepeval.metrics import BiasMetric, ToxicityMetric
            from deepeval.test_case import LLMTestCaseParams

            # Determine eval model for DeepEval
            # DeepEval uses OPENAI_API_KEY env var by default.
            # For OpenRouter: set OPENAI_API_KEY and OPENAI_API_BASE_URL env vars
            eval_model = self.model
            if self.settings.use_openrouter and self.settings.openrouter_api_key:
                import os
                os.environ["OPENAI_API_KEY"] = self.settings.openrouter_api_key
                os.environ["OPENAI_BASE_URL"] = self.settings.openrouter_base_url
                eval_model = self.settings.openrouter_model
                logger.info(f"DeepEval configured for OpenRouter: {eval_model}")

            metric_map = {
                "answer_relevancy": lambda: AnswerRelevancyMetric(
                    threshold=self.threshold,
                    model=eval_model,
                ),
                "faithfulness": lambda: FaithfulnessMetric(
                    threshold=self.threshold,
                    model=eval_model,
                ),
                "contextual_precision": lambda: ContextualPrecisionMetric(
                    threshold=self.threshold,
                    model=eval_model,
                ),
                "contextual_recall": lambda: ContextualRecallMetric(
                    threshold=self.threshold,
                    model=eval_model,
                ),
                "contextual_relevancy": lambda: ContextualRelevancyMetric(
                    threshold=self.threshold,
                    model=eval_model,
                ),
                "hallucination": lambda: HallucinationMetric(
                    threshold=self.threshold,
                    model=eval_model,
                ),
                # ─── Custom Production Metrics ────────────────────────
                "bias": lambda: BiasMetric(
                    threshold=self.threshold,
                    model=eval_model,
                ),
                "toxicity": lambda: ToxicityMetric(
                    threshold=self.threshold,
                    model=eval_model,
                ),
                "coherence": lambda: GEval(
                    name="Coherence",
                    criteria="Evaluate if the response is logically coherent, well-structured, and easy to follow. Check that sentences connect logically and the overall response makes sense as a unified answer.",
                    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                    threshold=self.threshold,
                    model=eval_model,
                ),
                "completeness": lambda: GEval(
                    name="Completeness",
                    criteria="Evaluate if the response thoroughly answers the user's question covering all key aspects. A complete response addresses all parts of the question without missing critical information.",
                    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                    threshold=self.threshold,
                    model=eval_model,
                ),
                "conciseness": lambda: GEval(
                    name="Conciseness",
                    criteria="Evaluate if the response is concise and avoids unnecessary repetition or filler content. A concise response delivers information efficiently without being overly verbose.",
                    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                    threshold=self.threshold,
                    model=eval_model,
                ),
            }

            self._metrics = {}
            for metric_name in self.metrics_config:
                name = metric_name.strip().lower()
                if name in metric_map:
                    self._metrics[name] = metric_map[name]()
                    logger.debug(f"Initialized DeepEval metric: {name}")
                else:
                    logger.warning(
                        f"Unknown metric '{name}'. Available: {list(metric_map.keys())}"
                    )

            # Add custom production metrics if enabled
            if self.settings.eval_custom_metrics_enabled:
                custom_metrics = ["bias", "toxicity", "coherence", "completeness", "conciseness"]
                for name in custom_metrics:
                    if name not in self._metrics and name in metric_map:
                        self._metrics[name] = metric_map[name]()
                        logger.debug(f"Initialized custom metric: {name}")

            if not self._metrics:
                raise EvaluationError("No valid DeepEval metrics configured")

            logger.info(f"DeepEval evaluator initialized with {len(self._metrics)} metrics")

        except ImportError as e:
            raise EvaluationError(
                f"DeepEval is not installed: {e}. Install with: pip install deepeval",
                details={"error": str(e)},
            )

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """
        Evaluate a single sample against all configured metrics.

        Args:
            sample: EvaluationSample with user_input, actual_output, etc.

        Returns:
            EvaluationResult with metric scores.
        """
        from deepeval.test_case import LLMTestCase

        if not sample.actual_output:
            raise EvaluationError(
                f"Sample {sample.sample_id} has no actual_output for evaluation"
            )

        test_case = LLMTestCase(
            input=sample.user_input,
            actual_output=sample.actual_output,
            expected_output=sample.expected_output,
            context=sample.context,
            retrieval_context=sample.retrieval_context,
        )

        metric_results = []
        for metric_name, metric in self._metrics.items():
            try:
                metric.measure(test_case)
                result = MetricResult(
                    metric_name=metric_name,
                    score=metric.score,
                    threshold=self.threshold,
                    passed=metric.score >= self.threshold,
                    reason=metric.reason if hasattr(metric, "reason") else None,
                )
                metric_results.append(result)
                logger.debug(
                    f"  {metric_name}: {metric.score:.3f} "
                    f"({'PASS' if result.passed else 'FAIL'})"
                )

            except Exception as e:
                logger.error(f"Metric '{metric_name}' failed for sample {sample.sample_id}: {e}")
                metric_results.append(MetricResult(
                    metric_name=metric_name,
                    score=0.0,
                    threshold=self.threshold,
                    passed=False,
                    reason=f"Evaluation error: {str(e)}",
                ))

        overall_passed = all(m.passed for m in metric_results)

        return EvaluationResult(
            sample_id=sample.sample_id,
            metrics=metric_results,
            overall_passed=overall_passed,
        )

    def evaluate_dataset(self, dataset: EvaluationDataset) -> EvaluationReport:
        """
        Evaluate an entire dataset against all metrics.

        Args:
            dataset: EvaluationDataset with samples.

        Returns:
            EvaluationReport with all results and summary.
        """
        start_time = time.time()
        logger.info(
            f"Starting DeepEval evaluation: {dataset.size} samples × {len(self._metrics)} metrics"
        )

        results = []
        for i, sample in enumerate(dataset.samples, 1):
            logger.info(f"Evaluating sample {i}/{dataset.size}: {sample.sample_id}")
            try:
                result = self.evaluate_sample(sample)
                results.append(result)
            except EvaluationError as e:
                logger.error(f"Sample {sample.sample_id} evaluation failed: {e.message}")

        duration = time.time() - start_time

        # Generate summary
        summary = self._generate_summary(results)

        report = EvaluationReport(
            report_id=f"deepeval_{uuid.uuid4().hex[:8]}",
            framework="deepeval",
            dataset_id=dataset.dataset_id,
            results=results,
            summary=summary,
            duration_seconds=duration,
            config={
                "metrics": self.metrics_config,
                "threshold": self.threshold,
                "model": self.model,
            },
        )

        logger.info(
            f"DeepEval evaluation complete: {report.passed_samples}/{report.total_samples} passed "
            f"({report.pass_rate:.1%}) in {duration:.1f}s"
        )

        return report

    def evaluate_single(
        self,
        user_input: str,
        actual_output: str,
        expected_output: Optional[str] = None,
        context: Optional[list[str]] = None,
        retrieval_context: Optional[list[str]] = None,
    ) -> EvaluationResult:
        """
        Convenience method to evaluate a single query-response pair.

        Args:
            user_input: The user's question.
            actual_output: The LLM-generated response.
            expected_output: Optional ground truth answer.
            context: Optional ground truth context.
            retrieval_context: Optional retrieved context from RAG.

        Returns:
            EvaluationResult with metric scores.
        """
        sample = EvaluationSample(
            sample_id=f"single_{uuid.uuid4().hex[:8]}",
            user_input=user_input,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
        )
        return self.evaluate_sample(sample)

    def _generate_summary(self, results: list[EvaluationResult]) -> dict:
        """Generate aggregate summary statistics."""
        if not results:
            return {"total_samples": 0, "passed": 0, "failed": 0}

        # Per-metric aggregation
        metric_scores: dict[str, list[float]] = {}
        for result in results:
            for metric in result.metrics:
                if metric.metric_name not in metric_scores:
                    metric_scores[metric.metric_name] = []
                metric_scores[metric.metric_name].append(metric.score)

        metric_summary = {}
        for name, scores in metric_scores.items():
            metric_summary[name] = {
                "average_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "pass_rate": sum(1 for s in scores if s >= self.threshold) / len(scores),
                "total_evaluated": len(scores),
            }

        return {
            "total_samples": len(results),
            "passed": sum(1 for r in results if r.overall_passed),
            "failed": sum(1 for r in results if not r.overall_passed),
            "overall_pass_rate": sum(1 for r in results if r.overall_passed) / len(results),
            "metrics": metric_summary,
        }