"""
Dynamic Evaluation Dataset Generator.

Generates eval Q&A pairs from ingested documents using configurable strategies:
  - ragas:    RAGAS TestsetGenerator (enterprise recommended — diverse multi-complexity)
  - document: Document-grounded LLM generation (chunk-based, fast)
  - hybrid:   Both strategies merged with deduplication (maximum coverage)

All configuration via .env / environment variables:
  TESTSET_STRATEGY=ragas       # ragas | document | hybrid
  TESTSET_SIZE=20              # Number of test cases to generate

Usage:
    python -m scripts.generate_eval_dataset
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from evaluation.dataset_manager import DatasetManager
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """
    Generate evaluation dataset from ingested documents.
    All configuration is read from settings (.env / environment variables).
    """
    settings = get_settings()
    start_time = time.time()

    strategy = settings.testset_strategy
    testset_size = settings.testset_size

    logger.info("=" * 60)
    logger.info("  Dynamic Evaluation Dataset Generator")
    logger.info("=" * 60)
    logger.info(f"  Strategy  : {strategy}")
    logger.info(f"  Size      : {testset_size}")
    logger.info(f"  Model     : {settings.eval_model}")
    logger.info(f"  Output    : {settings.eval_dataset_path}")
    logger.info("=" * 60)

    # ─── Generate dataset using DatasetManager ────────────────────
    manager = DatasetManager()

    try:
        dataset = manager.generate_dataset(
            strategy=strategy,
            testset_size=testset_size,
        )
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        sys.exit(1)

    if dataset.size == 0:
        logger.error("No samples generated. Check LLM connectivity and document ingestion.")
        sys.exit(1)

    # ─── Quality validation ───────────────────────────────────────
    logger.info("Running quality validation...")
    dataset, quality_report = manager.validate_quality(dataset)
    logger.info(
        f"Quality gate: {quality_report['total_valid']}/{quality_report['total_input']} samples passed "
        f"(score={quality_report['quality_score']:.1%})"
    )

    if quality_report["total_rejected"] > 0:
        for r in quality_report["rejected_samples"][:5]:
            logger.warning(f"  Rejected [{r['sample_id']}]: {r['issues']}")

    # ─── Save dataset ─────────────────────────────────────────────
    output_path = manager.save_dataset(dataset)

    # ─── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - start_time

    # Count by question type
    type_counts = {}
    for sample in dataset.samples:
        qt = sample.question_type or "unclassified"
        type_counts[qt] = type_counts.get(qt, 0) + 1

    difficulty_counts = {}
    for sample in dataset.samples:
        d = sample.difficulty or "unclassified"
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1

    logger.info("=" * 60)
    logger.info(f"  Strategy       : {strategy}")
    logger.info(f"  Total samples  : {dataset.size}")
    logger.info(f"  Question types : {type_counts}")
    logger.info(f"  Difficulty     : {difficulty_counts}")
    logger.info(f"  Saved to       : {output_path}")
    logger.info(f"  Duration       : {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()