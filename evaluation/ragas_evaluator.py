"""
RAGAS evaluator module for GenAI RAG System.
Production-level LLM evaluation using RAGAS framework.
Supports: Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, and more.
"""

import time
import uuid
from typing import Optional

from config.settings import get_settings
from schemas.evaluation import (
    EvaluationDataset,
    EvaluationReport,
    EvaluationResult,
    EvaluationSample,
    MetricResult,
)
from utils.exceptions import EvaluationError
from utils.logger import get_logger

logger = get_logger(__name__)


class _CreditExhausted(Exception):
    """Raised when OpenRouter returns 402 (insufficient credits)."""
    pass


class RagasEvaluator:
    """
    Production-level evaluator using RAGAS framework.
    Evaluates RAG pipeline outputs with RAG-specific metrics.
    """

    def __init__(
        self,
        metrics: Optional[list[str]] = None,
        threshold: Optional[float] = None,
        model: Optional[str] = None,
    ):
        self.settings = get_settings()
        self.metrics_config = metrics or self.settings.ragas_metrics_list
        self.threshold = threshold or self.settings.ragas_threshold
        self.model = model or self.settings.eval_model

        self._metrics = None
        self._llm = None
        self._embeddings = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize RAGAS metrics and LLM wrapper."""
        try:
            from ragas.metrics import (
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
            )
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            # Initialize LLM for evaluation - Provider: OpenRouter > Azure > OpenAI
            if self.settings.use_openrouter and self.settings.openrouter_api_key:
                llm = ChatOpenAI(
                    model=self.settings.openrouter_model,
                    api_key=self.settings.openrouter_api_key,
                    base_url=self.settings.openrouter_base_url,
                    temperature=0.0,
                    max_tokens=100,
                    request_timeout=60.0,
                )
                embeddings = OpenAIEmbeddings(
                    model=self.settings.eval_embedding_model,
                    api_key=self.settings.openrouter_api_key,
                    base_url=self.settings.openrouter_base_url,
                    timeout=60.0,
                )
                logger.info(f"RAGAS using OpenRouter: {self.settings.openrouter_model}")
            elif self.settings.azure_openai_endpoint:
                from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

                llm = AzureChatOpenAI(
                    azure_endpoint=self.settings.azure_openai_endpoint,
                    api_key=self.settings.azure_openai_api_key,
                    api_version=self.settings.azure_openai_api_version,
                    deployment_name=self.settings.azure_openai_deployment,
                    temperature=0.0,
                )
                embeddings = AzureOpenAIEmbeddings(
                    azure_endpoint=self.settings.azure_openai_endpoint,
                    api_key=self.settings.azure_openai_api_key,
                    api_version=self.settings.azure_openai_api_version,
                )
                logger.info("RAGAS using Azure OpenAI")
            else:
                llm = ChatOpenAI(
                    model=self.model,
                    api_key=self.settings.openai_api_key,
                    temperature=0.0,
                    max_tokens=100,
                    request_timeout=60.0,
                )
                embeddings = OpenAIEmbeddings(
                    model=self.settings.eval_embedding_model,
                    api_key=self.settings.openai_api_key,
                )
                logger.info(f"RAGAS using OpenAI: {self.model}")

            self._llm = LangchainLLMWrapper(llm)
            self._embeddings = LangchainEmbeddingsWrapper(embeddings)

            # Initialize metrics
            metric_map = {
                "faithfulness": Faithfulness,
                "answer_relevancy": AnswerRelevancy,
                "context_precision": ContextPrecision,
                "context_recall": ContextRecall,
            }

            self._metrics = {}
            for metric_name in self.metrics_config:
                name = metric_name.strip().lower()
                if name in metric_map:
                    metric_instance = metric_map[name](llm=self._llm)
                    if name == "answer_relevancy":
                        metric_instance = metric_map[name](llm=self._llm, embeddings=self._embeddings)
                    self._metrics[name] = metric_instance
                    logger.debug(f"Initialized RAGAS metric: {name}")
                else:
                    logger.warning(f"Unknown RAGAS metric '{name}'. Available: {list(metric_map.keys())}")

            if not self._metrics:
                raise EvaluationError("No valid RAGAS metrics configured")

            logger.info(f"RAGAS evaluator initialized with {len(self._metrics)} metrics")

        except ImportError as e:
            raise EvaluationError(
                f"RAGAS dependencies not installed: {e}. "
                "Install with: pip install ragas langchain-openai",
                details={"error": str(e)},
            )

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """
        Evaluate a single sample against all configured RAGAS metrics.

        Args:
            sample: EvaluationSample with required fields.

        Returns:
            EvaluationResult with metric scores.
        """
        from ragas import SingleTurnSample, evaluate

        if not sample.actual_output:
            raise EvaluationError(
                f"Sample {sample.sample_id} has no actual_output for evaluation"
            )

        # Truncate contexts to reduce token usage (avoids 402 credit errors on OpenRouter)
        max_chars = 1500
        retrieved_contexts = [c[:max_chars] for c in (sample.retrieval_context or [])][:3]
        reference_contexts = [c[:max_chars] for c in (sample.context or [])][:3] if sample.context else []

        ragas_sample = SingleTurnSample(
            user_input=sample.user_input,
            response=sample.actual_output[:2000],
            reference=sample.expected_output[:2000] if sample.expected_output else "",
            retrieved_contexts=retrieved_contexts,
            reference_contexts=reference_contexts,
        )

        metric_results = []
        for metric_name, metric in self._metrics.items():
            try:
                score = metric.single_turn_score(ragas_sample)

                result = MetricResult(
                    metric_name=metric_name,
                    score=float(score),
                    threshold=self.threshold,
                    passed=float(score) >= self.threshold,
                    reason=None,
                )
                metric_results.append(result)
                logger.debug(
                    f"  {metric_name}: {score:.3f} ({'PASS' if result.passed else 'FAIL'})"
                )

            except Exception as e:
                error_str = str(e)
                if "402" in error_str and "credits" in error_str.lower():
                    logger.error(f"OpenRouter credits exhausted at metric '{metric_name}'. Aborting.")
                    metric_results.append(MetricResult(
                        metric_name=metric_name, score=0.0, threshold=self.threshold,
                        passed=False, reason="Credits exhausted",
                    ))
                    raise _CreditExhausted(error_str)
                logger.error(f"RAGAS metric '{metric_name}' failed for sample {sample.sample_id}: {e}")
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
        Evaluate an entire dataset using RAGAS batch evaluation.

        Args:
            dataset: EvaluationDataset with samples.

        Returns:
            EvaluationReport with all results and summary.
        """
        start_time = time.time()
        logger.info(
            f"Starting RAGAS evaluation: {dataset.size} samples × {len(self._metrics)} metrics"
        )

        results = []
        for i, sample in enumerate(dataset.samples, 1):
            logger.info(f"Evaluating sample {i}/{dataset.size}: {sample.sample_id}")
            try:
                result = self.evaluate_sample(sample)
                results.append(result)
            except _CreditExhausted:
                logger.warning(f"Stopping RAGAS evaluation early — credits exhausted after {len(results)} samples")
                break
            except EvaluationError as e:
                logger.error(f"Sample {sample.sample_id} evaluation failed: {e.message}")

        duration = time.time() - start_time

        summary = self._generate_summary(results)

        report = EvaluationReport(
            report_id=f"ragas_{uuid.uuid4().hex[:8]}",
            framework="ragas",
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
            f"RAGAS evaluation complete: {report.passed_samples}/{report.total_samples} passed "
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
