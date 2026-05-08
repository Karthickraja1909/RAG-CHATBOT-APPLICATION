"""
Threshold compliance checker for CI/CD pipeline.
Reads the latest evaluation report and validates against configured thresholds.
All configuration from .env — no CLI arguments.

Usage:
    python -m scripts.check_thresholds
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Check evaluation results against configured thresholds."""
    settings = get_settings()
    results_dir = Path(settings.eval_results_path)
    threshold = settings.eval_threshold

    logger.info("=" * 50)
    logger.info("THRESHOLD COMPLIANCE CHECK")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Required threshold: {threshold:.1%}")
    logger.info("=" * 50)

    # Load latest report
    latest_path = results_dir / "latest_report.json"
    if not latest_path.exists():
        logger.error(f"No evaluation report found at: {latest_path}")
        logger.error("Run the evaluation pipeline first: python -m scripts.run_evaluation")
        sys.exit(1)

    with open(latest_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    summary = report.get("summary", {})
    overall_pass_rate = summary.get("overall_pass_rate", 0.0)
    results = report.get("results", {})

    all_passed = True
    failures = []

    for framework, data in results.items():
        pass_rate = data.get("pass_rate", 0.0)
        status = "✓ PASS" if pass_rate >= threshold else "✗ FAIL"
        logger.info(f"\n{framework.upper()}: {pass_rate:.1%} [{status}]")

        if pass_rate < threshold:
            all_passed = False
            failures.append(f"{framework}: {pass_rate:.1%} < {threshold:.1%}")

        # Check individual metrics
        for metric_name, metric_data in data.get("metrics_summary", {}).items():
            avg_score = metric_data.get("average_score", 0.0)
            m_status = "✓" if avg_score >= threshold else "✗"
            logger.info(f"  {m_status} {metric_name}: {avg_score:.3f}")

            if avg_score < threshold:
                all_passed = False
                failures.append(f"{framework}/{metric_name}: {avg_score:.3f} < {threshold}")

    logger.info("\n" + "=" * 50)

    if all_passed:
        logger.info("✅ ALL THRESHOLDS PASSED")
        logger.info(f"Overall pass rate: {overall_pass_rate:.1%} (required: {threshold:.1%})")
        sys.exit(0)
    else:
        logger.error("❌ THRESHOLD COMPLIANCE FAILED")
        for failure in failures:
            logger.error(f"  - {failure}")

        if settings.fail_on_threshold_breach:
            sys.exit(1)
        else:
            logger.warning("fail_on_threshold_breach=false, exiting with 0")
            sys.exit(0)


if __name__ == "__main__":
    main()