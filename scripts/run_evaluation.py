"""
Automated LLM Evaluation Pipeline Runner.
Evaluates the complete RAG system using DeepEval and RAGAS frameworks.
All configuration from .env — no CLI arguments required.

Usage:
    python -m scripts.run_evaluation
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from evaluation.evaluation_pipeline import EvaluationPipeline
from evaluation.dataset_manager import DatasetManager
from pipeline.rag_pipeline import RAGPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """
    Fully automated evaluation pipeline.
    Reads all config from .env — runs both DeepEval and RAGAS,
    checks thresholds, generates report, sends email notification.
    """
    settings = get_settings()
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("  GenAI RAG System — Automated LLM Evaluation Pipeline")
    logger.info("=" * 70)
    logger.info(f"  Timestamp       : {timestamp}")
    logger.info(f"  Frameworks      : {settings.eval_frameworks_list}")
    logger.info(f"  Threshold       : {settings.eval_threshold}")
    logger.info(f"  Dataset         : {settings.eval_dataset_path}")
    logger.info(f"  Eval Model      : {settings.eval_model}")
    logger.info(f"  Run RAG Pipeline: {settings.eval_run_rag_pipeline}")
    logger.info(f"  Custom Metrics  : {settings.eval_custom_metrics_enabled}")
    logger.info(f"  Email Enabled   : {settings.email_enabled}")
    if settings.email_enabled and settings.email_recipients_list:
        logger.info(f"  Recipients      : {settings.email_recipients_list}")
    logger.info("=" * 70)

    # ─── Step 1: Initialize RAG Pipeline ──────────────────────────────
    rag_pipeline = None
    if settings.eval_run_rag_pipeline:
        try:
            rag_pipeline = RAGPipeline()
            logger.info("✓ RAG Pipeline initialized")
        except Exception as e:
            logger.warning(f"RAG Pipeline init failed: {e}. Will use dataset pre-computed outputs.")

    # ─── Step 1.5: Auto-generate eval dataset if enabled ──────────────
    if settings.eval_auto_generate_dataset:
        dataset_path = Path(settings.eval_dataset_path)
        logger.info("Auto-generating evaluation dataset from ingested documents...")
        try:
            from scripts.generate_eval_dataset import main as generate_dataset
            generate_dataset()
            logger.info("✓ Evaluation dataset generated dynamically")
        except Exception as e:
            logger.warning(f"Dataset generation failed: {e}. Using existing dataset.")

    # ─── Step 2: Initialize Evaluation Pipeline ───────────────────────
    eval_pipeline = EvaluationPipeline(rag_pipeline=rag_pipeline)

    # ─── Step 3: Run Full Evaluation ──────────────────────────────────
    try:
        reports = asyncio.run(eval_pipeline.arun_full_evaluation(
            dataset_path=settings.eval_dataset_path,
            frameworks=settings.eval_frameworks_list,
            run_rag=settings.eval_run_rag_pipeline and rag_pipeline is not None,
        ))
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}")
        sys.exit(1)

    if not reports:
        logger.error("No evaluation reports generated. Check framework configuration.")
        sys.exit(1)

    # ─── Step 4: Generate Combined Report ─────────────────────────────
    output_dir = Path(settings.eval_results_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_report = _build_combined_report(reports, settings, timestamp, start_time)

    report_filename = f"evaluation_report_{timestamp}.json"
    report_path = output_dir / report_filename
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"✓ Combined report saved: {report_path}")

    # Also save as latest.json for easy access
    latest_path = output_dir / "latest_report.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2, ensure_ascii=False, default=str)

    # ─── Step 5: Threshold Compliance Check ───────────────────────────
    threshold_passed = _check_thresholds(combined_report, settings)

    # ─── Step 6: Print Summary ────────────────────────────────────────
    _print_summary(combined_report, threshold_passed)

    # ─── Step 7: Send Email Report ────────────────────────────────────
    if settings.email_enabled and settings.email_recipients_list:
        try:
            from utils.email_sender import send_evaluation_report
            email_sent = send_evaluation_report(combined_report, report_path)
            if email_sent:
                logger.info("✓ Evaluation report sent via email")
            else:
                logger.warning("✗ Email report was NOT sent — check email configuration and logs above")
        except Exception as e:
            logger.error(f"Email sending failed with exception: {type(e).__name__}: {e}")
    else:
        if not settings.email_enabled:
            logger.info("Email notifications disabled (EMAIL_ENABLED=false)")
        elif not settings.email_recipients_list:
            logger.warning("No email recipients configured (EMAIL_RECIPIENTS is empty)")

    # ─── Step 8: Exit with appropriate code ───────────────────────────
    total_duration = time.time() - start_time
    logger.info(f"\nTotal pipeline duration: {total_duration:.1f}s")

    if not threshold_passed and settings.fail_on_threshold_breach:
        logger.error("Pipeline FAILED — metrics below threshold")
        sys.exit(1)

    logger.info("Pipeline PASSED — all metrics within threshold")
    sys.exit(0)


def _build_combined_report(
    reports: dict, settings, timestamp: str, start_time: float
) -> dict:
    """Build a comprehensive combined evaluation report."""
    combined = {
        "report_id": f"eval_{timestamp}",
        "timestamp": timestamp,
        "configuration": {
            "eval_model": settings.eval_model,
            "frameworks": settings.eval_frameworks_list,
            "threshold": settings.eval_threshold,
            "deepeval_threshold": settings.deepeval_threshold,
            "ragas_threshold": settings.ragas_threshold,
            "deepeval_metrics": settings.deepeval_metrics_list,
            "ragas_metrics": settings.ragas_metrics_list,
            "dataset_path": settings.eval_dataset_path,
            "run_rag_pipeline": settings.eval_run_rag_pipeline,
            "custom_metrics_enabled": settings.eval_custom_metrics_enabled,
        },
        "results": {},
        "summary": {
            "overall_pass_rate": 0.0,
            "total_samples": 0,
            "total_passed": 0,
            "total_failed": 0,
            "threshold_compliance": False,
            "duration_seconds": 0.0,
        },
    }

    total_pass_rates = []

    for framework, report in reports.items():
        framework_result = {
            "total_samples": report.total_samples,
            "passed_samples": report.passed_samples,
            "pass_rate": report.pass_rate,
            "duration_seconds": report.duration_seconds,
            "metrics_summary": report.summary.get("metrics", {}),
            "per_sample_results": [],
        }

        # Include per-sample detail
        for result in report.results:
            sample_detail = {
                "sample_id": result.sample_id,
                "overall_passed": result.overall_passed,
                "average_score": result.average_score,
                "metrics": {m.metric_name: {"score": m.score, "passed": m.passed, "reason": m.reason} for m in result.metrics},
            }
            framework_result["per_sample_results"].append(sample_detail)

        combined["results"][framework] = framework_result
        total_pass_rates.append(report.pass_rate)
        combined["summary"]["total_samples"] += report.total_samples
        combined["summary"]["total_passed"] += report.passed_samples
        combined["summary"]["total_failed"] += (report.total_samples - report.passed_samples)

    if total_pass_rates:
        combined["summary"]["overall_pass_rate"] = sum(total_pass_rates) / len(total_pass_rates)

    combined["summary"]["duration_seconds"] = time.time() - start_time
    combined["summary"]["threshold_compliance"] = combined["summary"]["overall_pass_rate"] >= settings.eval_threshold

    return combined


def _check_thresholds(report: dict, settings) -> bool:
    """Check if all metrics meet configured thresholds."""
    overall_pass_rate = report["summary"]["overall_pass_rate"]
    threshold = settings.eval_threshold

    logger.info("\n" + "─" * 50)
    logger.info("THRESHOLD COMPLIANCE CHECK")
    logger.info(f"Required: {threshold:.1%}")
    logger.info(f"Achieved: {overall_pass_rate:.1%}")
    logger.info("─" * 50)

    all_passed = True

    for framework, data in report["results"].items():
        pass_rate = data["pass_rate"]
        status = "✓ PASS" if pass_rate >= threshold else "✗ FAIL"
        logger.info(f"  {framework.upper()}: {pass_rate:.1%} [{status}]")

        if pass_rate < threshold:
            all_passed = False

        # Check individual metrics
        for metric_name, metric_data in data.get("metrics_summary", {}).items():
            avg = metric_data.get("average_score", 0.0)
            m_status = "✓" if avg >= threshold else "✗"
            logger.info(f"    {m_status} {metric_name}: {avg:.3f}")

    logger.info("─" * 50)
    return all_passed


def _print_summary(report: dict, threshold_passed: bool):
    """Print final evaluation summary."""
    summary = report["summary"]

    logger.info("\n" + "═" * 70)
    logger.info("  EVALUATION RESULTS SUMMARY")
    logger.info("═" * 70)
    logger.info(f"  Overall Pass Rate  : {summary['overall_pass_rate']:.1%}")
    logger.info(f"  Total Samples      : {summary['total_samples']}")
    logger.info(f"  Passed             : {summary['total_passed']}")
    logger.info(f"  Failed             : {summary['total_failed']}")
    logger.info(f"  Duration           : {summary['duration_seconds']:.1f}s")
    logger.info(f"  Threshold Met      : {'YES' if threshold_passed else 'NO'}")
    logger.info("═" * 70)


if __name__ == "__main__":
    main()