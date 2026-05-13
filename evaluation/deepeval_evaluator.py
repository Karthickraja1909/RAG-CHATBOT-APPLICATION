"""
DeepEval evaluator module for GenAI RAG System.
Production-level LLM evaluation using DeepEval framework.
Supports: AnswerRelevancy, Faithfulness, ContextualPrecision, ContextualRecall, Hallucination, GEval.
"""

import asyncio
import time
import uuid
from typing import Optional

import openai

from config.settings import get_settings
from schemas.evaluation import EvaluationDataset, EvaluationReport, EvaluationResult, EvaluationSample, MetricResult
from utils.exceptions import EvaluationError
from utils.logger import get_logger

logger = get_logger(__name__)


class _CreditExhausted(Exception):
    """Raised when OpenRouter returns 402 (insufficient credits)."""
    pass


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
            from deepeval.models import DeepEvalBaseLLM
            from deepeval.test_case import LLMTestCaseParams

            # ── Build a custom model wrapper that caps max_tokens ──
            # DeepEval's default GPTModel requests max_tokens=16384 for gpt-4o-mini,
            # which exceeds OpenRouter free-tier credit limits.
            eval_model_name = self.model
            eval_model = None  # Will be a DeepEvalBaseLLM instance or a string

            if self.settings.use_openrouter and self.settings.openrouter_api_key:
                import os
                os.environ["OPENAI_API_KEY"] = self.settings.openrouter_api_key
                os.environ["OPENAI_BASE_URL"] = self.settings.openrouter_base_url
                eval_model_name = self.settings.openrouter_model

                # Custom model wrapper to control max_tokens
                class _TokenCappedModel(DeepEvalBaseLLM):
                    def __init__(self, model_name, api_key, base_url, max_tokens=256):
                        self._model_name = model_name
                        self._max_tokens = max_tokens
                        self._client = openai.OpenAI(
                            api_key=api_key,
                            base_url=base_url,
                            timeout=60.0,
                        )
                        super().__init__(model_name)

                    def load_model(self):
                        return self._model_name

                    def generate(self, prompt: str, schema=None) -> str:
                        try:
                            resp = self._client.chat.completions.create(
                                model=self._model_name,
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=self._max_tokens,
                                temperature=0.0,
                            )
                            return resp.choices[0].message.content
                        except openai.APIStatusError as e:
                            if e.status_code == 402:
                                raise _CreditExhausted(str(e))
                            raise

                    async def a_generate(self, prompt: str, schema=None) -> str:
                        try:
                            client = openai.AsyncOpenAI(
                                api_key=self._client.api_key,
                                base_url=str(self._client.base_url),
                                timeout=60.0,
                            )
                            resp = await client.chat.completions.create(
                                model=self._model_name,
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=self._max_tokens,
                                temperature=0.0,
                            )
                            return resp.choices[0].message.content
                        except openai.APIStatusError as e:
                            if e.status_code == 402:
                                raise _CreditExhausted(str(e))
                            raise

                    def get_model_name(self) -> str:
                        return self._model_name

                eval_model = _TokenCappedModel(
                    model_name=eval_model_name,
                    api_key=self.settings.openrouter_api_key,
                    base_url=self.settings.openrouter_base_url,
                    max_tokens=100,
                )
                logger.info(f"DeepEval configured for OpenRouter: {eval_model_name} (max_tokens=100)")
            else:
                eval_model = eval_model_name

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

    async def aevaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """
        Evaluate a single sample asynchronously — runs all metrics concurrently.
        """
        from deepeval.test_case import LLMTestCase

        if not sample.actual_output:
            raise EvaluationError(
                f"Sample {sample.sample_id} has no actual_output for evaluation"
            )

        max_chars = getattr(self, "_max_context_chars", 1500)
        retrieval_context = [c[:max_chars] for c in (sample.retrieval_context or [])][:3]
        context = [c[:max_chars] for c in (sample.context or [])][:3] if sample.context else None

        test_case = LLMTestCase(
            input=sample.user_input,
            actual_output=sample.actual_output[:2000],
            expected_output=sample.expected_output[:2000] if sample.expected_output else None,
            context=context,
            retrieval_context=retrieval_context,
        )

        async def _measure_one(metric_name: str, metric):
            try:
                await metric.a_measure(test_case)
                result = MetricResult(
                    metric_name=metric_name,
                    score=metric.score,
                    threshold=self.threshold,
                    passed=metric.score >= self.threshold,
                    reason=metric.reason if hasattr(metric, "reason") else None,
                )
                logger.debug(
                    f"  {metric_name}: {metric.score:.3f} "
                    f"({'PASS' if result.passed else 'FAIL'})"
                )
                return result
            except Exception as e:
                error_str = str(e)
                if "402" in error_str and "credits" in error_str.lower():
                    logger.error(f"OpenRouter credits exhausted at metric '{metric_name}'.")
                    raise _CreditExhausted(error_str)
                logger.error(f"Metric '{metric_name}' failed for sample {sample.sample_id}: {e}")
                return MetricResult(
                    metric_name=metric_name,
                    score=0.0,
                    threshold=self.threshold,
                    passed=False,
                    reason=f"Evaluation error: {str(e)}",
                )

        tasks = [_measure_one(name, metric) for name, metric in self._metrics.items()]
        metric_results = await asyncio.gather(*tasks)

        return EvaluationResult(
            sample_id=sample.sample_id,
            metrics=list(metric_results),
            overall_passed=all(m.passed for m in metric_results),
        )

    async def aevaluate_dataset(self, dataset: EvaluationDataset) -> EvaluationReport:
        """
        Evaluate an entire dataset asynchronously with batched concurrency.
        Uses eval_batch_size to control how many samples run in parallel.
        """
        start_time = time.time()
        batch_size = self.settings.eval_batch_size
        logger.info(
            f"Starting async DeepEval evaluation: {dataset.size} samples × "
            f"{len(self._metrics)} metrics (batch_size={batch_size})"
        )

        results: list[EvaluationResult] = []
        semaphore = asyncio.Semaphore(batch_size)
        credit_exhausted = False

        async def _eval_with_semaphore(idx: int, sample: EvaluationSample):
            nonlocal credit_exhausted
            if credit_exhausted:
                return None
            async with semaphore:
                logger.info(f"Evaluating sample {idx}/{dataset.size}: {sample.sample_id}")
                try:
                    return await self.aevaluate_sample(sample)
                except _CreditExhausted:
                    credit_exhausted = True
                    logger.warning("Credits exhausted — cancelling remaining samples")
                    return None
                except EvaluationError as e:
                    logger.error(f"Sample {sample.sample_id} evaluation failed: {e.message}")
                    return None

        tasks = [
            _eval_with_semaphore(i, sample)
            for i, sample in enumerate(dataset.samples, 1)
        ]
        raw_results = await asyncio.gather(*tasks)
        results = [r for r in raw_results if r is not None]

        duration = time.time() - start_time
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
                "batch_size": batch_size,
                "async": True,
            },
        )

        logger.info(
            f"Async DeepEval evaluation complete: {report.passed_samples}/{report.total_samples} passed "
            f"({report.pass_rate:.1%}) in {duration:.1f}s"
        )

        return report

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
