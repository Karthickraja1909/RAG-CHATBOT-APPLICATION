"""
Evaluation pipeline module for GenAI RAG System.
Orchestrates the full evaluation workflow: load data → run RAG → evaluate → report.
"""

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Optional

from config.settings import get_settings
from evaluation.dataset_manager import DatasetManager
from evaluation.deepeval_evaluator import DeepEvalEvaluator
from evaluation.ragas_evaluator import RagasEvaluator
from pipeline.rag_pipeline import RAGPipeline
from schemas.evaluation import EvaluationDataset, EvaluationReport, EvaluationSample
from schemas.pipeline import QueryRequest
from utils.exceptions import EvaluationError
from utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationPipeline:
    """
    End-to-end evaluation pipeline for the RAG system.
    Supports both DeepEval and RAGAS frameworks.
    Workflow: Load Dataset → Run RAG Pipeline → Evaluate with Frameworks → Generate Reports.
    """

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        deepeval_evaluator: Optional[DeepEvalEvaluator] = None,
        ragas_evaluator: Optional[RagasEvaluator] = None,
        dataset_manager: Optional[DatasetManager] = None,
    ):
        self.settings = get_settings()
        self.rag_pipeline = rag_pipeline
        self.deepeval_evaluator = deepeval_evaluator
        self.ragas_evaluator = ragas_evaluator
        self.dataset_manager = dataset_manager or DatasetManager()

    async def arun_full_evaluation(
        self,
        dataset_path: Optional[str] = None,
        frameworks: Optional[list[str]] = None,
        run_rag: bool = True,
    ) -> dict[str, EvaluationReport]:
        """
        Run the complete evaluation pipeline asynchronously.
        Uses aevaluate_dataset() on each framework for concurrent sample evaluation.

        Args:
            dataset_path: Path to evaluation dataset JSON.
            frameworks: List of frameworks to use ("deepeval", "ragas", or both).
            run_rag: Whether to run RAG pipeline to generate actual_outputs.

        Returns:
            Dictionary mapping framework name to EvaluationReport.
        """
        frameworks = frameworks or ["deepeval", "ragas"]
        start_time = time.time()

        logger.info(f"Starting async evaluation pipeline with frameworks: {frameworks}")

        # Step 1: Load dataset
        dataset = self.dataset_manager.load_dataset(dataset_path)
        logger.info(f"Loaded dataset: {dataset.name} ({dataset.size} samples)")

        # Step 2: Run RAG pipeline (sync — needs sequential API calls)
        if run_rag and self.rag_pipeline:
            dataset = self._run_rag_on_dataset(dataset)

        # Step 3: Run evaluation frameworks concurrently
        reports: dict[str, EvaluationReport] = {}

        async def _run_deepeval():
            try:
                evaluator = self.deepeval_evaluator or DeepEvalEvaluator()
                return await evaluator.aevaluate_dataset(dataset)
            except EvaluationError as e:
                logger.error(f"DeepEval async evaluation failed: {e.message}")
                return None

        async def _run_ragas():
            try:
                evaluator = self.ragas_evaluator or RagasEvaluator()
                return await evaluator.aevaluate_dataset(dataset)
            except EvaluationError as e:
                logger.error(f"RAGAS async evaluation failed: {e.message}")
                return None

        tasks = {}
        if "deepeval" in frameworks:
            tasks["deepeval"] = _run_deepeval()
        if "ragas" in frameworks:
            tasks["ragas"] = _run_ragas()

        if tasks:
            results = await asyncio.gather(*tasks.values())
            for name, result in zip(tasks.keys(), results):
                if result is not None:
                    reports[name] = result

        # Step 4: Save reports
        for framework, report in reports.items():
            self._save_report(report, framework)

        total_time = time.time() - start_time
        logger.info(f"Async evaluation pipeline completed in {total_time:.1f}s")

        return reports

    async def arun_deepeval_evaluation(
        self,
        dataset_path: Optional[str] = None,
        run_rag: bool = True,
    ) -> EvaluationReport:
        """Run evaluation using only DeepEval framework asynchronously."""
        reports = await self.arun_full_evaluation(
            dataset_path=dataset_path,
            frameworks=["deepeval"],
            run_rag=run_rag,
        )
        return reports.get("deepeval")

    async def arun_ragas_evaluation(
        self,
        dataset_path: Optional[str] = None,
        run_rag: bool = True,
    ) -> EvaluationReport:
        """Run evaluation using only RAGAS framework asynchronously."""
        reports = await self.arun_full_evaluation(
            dataset_path=dataset_path,
            frameworks=["ragas"],
            run_rag=run_rag,
        )
        return reports.get("ragas")

    async def aevaluate_single_query(
        self,
        user_input: str,
        expected_output: Optional[str] = None,
        context: Optional[list[str]] = None,
        frameworks: Optional[list[str]] = None,
    ) -> dict:
        """
        Evaluate a single query end-to-end asynchronously.

        Args:
            user_input: User question.
            expected_output: Expected answer (ground truth).
            context: Ground truth context.
            frameworks: Evaluation frameworks to use.

        Returns:
            Dictionary with RAG response and evaluation results.
        """
        frameworks = frameworks or ["deepeval", "ragas"]

        # Run RAG
        actual_output = ""
        retrieval_context = []

        if self.rag_pipeline:
            request = QueryRequest(query=user_input)
            response = self.rag_pipeline.query(request)
            actual_output = response.answer
            retrieval_context = [s.content for s in response.sources]

        results = {"query": user_input, "answer": actual_output, "evaluations": {}}

        sample = EvaluationSample(
            sample_id=f"single_{uuid.uuid4().hex[:8]}",
            user_input=user_input,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
        )

        # DeepEval
        if "deepeval" in frameworks:
            try:
                evaluator = self.deepeval_evaluator or DeepEvalEvaluator()
                result = await evaluator.aevaluate_sample(sample)
                results["evaluations"]["deepeval"] = {
                    "passed": result.overall_passed,
                    "average_score": result.average_score,
                    "metrics": {m.metric_name: m.score for m in result.metrics},
                }
            except Exception as e:
                logger.error(f"DeepEval single evaluation failed: {e}")

        # RAGAS
        if "ragas" in frameworks:
            try:
                evaluator = self.ragas_evaluator or RagasEvaluator()
                result = await evaluator.aevaluate_sample(sample)
                results["evaluations"]["ragas"] = {
                    "passed": result.overall_passed,
                    "average_score": result.average_score,
                    "metrics": {m.metric_name: m.score for m in result.metrics},
                }
            except Exception as e:
                logger.error(f"RAGAS single evaluation failed: {e}")

        return results

    def check_threshold_compliance(self, report: EvaluationReport) -> bool:
        """
        Check if evaluation report meets threshold requirements.
        Used in CI/CD to determine pass/fail.

        Args:
            report: EvaluationReport to check.

        Returns:
            True if all metrics pass threshold, False otherwise.
        """
        if not report.results:
            return False

        threshold = self.settings.eval_threshold
        all_passed = report.pass_rate >= threshold

        if not all_passed and self.settings.fail_on_threshold_breach:
            logger.error(
                f"Threshold compliance FAILED: pass_rate={report.pass_rate:.1%} "
                f"(required: {threshold:.1%})"
            )
        else:
            logger.info(f"Threshold compliance PASSED: pass_rate={report.pass_rate:.1%}")

        return all_passed

    def _run_rag_on_dataset(self, dataset: EvaluationDataset) -> EvaluationDataset:
        """Run RAG pipeline on each sample to generate actual_output and retrieval_context."""
        logger.info(f"Running RAG pipeline on {dataset.size} samples...")

        updated_samples = []
        success_count = 0
        for i, sample in enumerate(dataset.samples, 1):
            try:
                request = QueryRequest(query=sample.user_input)
                response = self.rag_pipeline.query(request)

                updated_sample = EvaluationSample(
                    sample_id=sample.sample_id,
                    user_input=sample.user_input,
                    actual_output=response.answer,
                    expected_output=sample.expected_output,
                    context=sample.context,
                    retrieval_context=[s.content for s in response.sources],
                )
                updated_samples.append(updated_sample)
                success_count += 1
                preview = response.answer[:120].replace("\n", " ") if response.answer else "(empty)"
                logger.info(f"  Sample {i}/{dataset.size} [{sample.sample_id}] OK — {len(response.sources)} sources, output: {preview}...")

            except Exception as e:
                logger.error(f"RAG failed for sample {sample.sample_id}: {type(e).__name__}: {e}")
                # Provide fallback so evaluators don't get None actual_output
                fallback = EvaluationSample(
                    sample_id=sample.sample_id,
                    user_input=sample.user_input,
                    actual_output=f"[RAG_ERROR] {type(e).__name__}: {e}",
                    expected_output=sample.expected_output,
                    context=sample.context,
                    retrieval_context=[],
                )
                updated_samples.append(fallback)

        logger.info(f"RAG pipeline completed: {success_count}/{dataset.size} samples succeeded")

        dataset.samples = updated_samples
        return dataset

    def _save_report(self, report: EvaluationReport, framework: str) -> Path:
        """Save evaluation report to configured results directory."""
        results_dir = Path(self.settings.eval_results_path)
        results_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{framework}_report_{report.report_id}.json"
        report_path = results_dir / filename

        report_data = report.model_dump(mode="json")

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Saved {framework} report to: {report_path}")
        return report_path