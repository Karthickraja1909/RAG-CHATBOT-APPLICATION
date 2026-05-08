"""
Dataset manager for evaluation data.
Handles loading, saving, and managing evaluation datasets.
"""

import json
import uuid
from pathlib import Path
from typing import Optional, Union

from config.settings import get_settings
from schemas.evaluation import EvaluationDataset, EvaluationSample
from utils.exceptions import EvaluationError
from utils.logger import get_logger

logger = get_logger(__name__)


class DatasetManager:
    """Manages evaluation datasets - load, save, validate, and transform."""

    def __init__(self):
        self.settings = get_settings()

    def load_dataset(self, file_path: Union[str, Path] = None) -> EvaluationDataset:
        """
        Load an evaluation dataset from JSON file.

        Expected JSON structure:
        {
            "dataset_id": "...",
            "name": "...",
            "description": "...",
            "samples": [
                {
                    "sample_id": "...",
                    "user_input": "...",
                    "expected_output": "...",
                    "context": ["..."],
                    "retrieval_context": ["..."]
                }
            ]
        }
        """
        path = Path(file_path or self.settings.eval_dataset_path)

        if not path.exists():
            raise EvaluationError(
                f"Evaluation dataset not found: {path}",
                details={"file_path": str(path)},
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            dataset = EvaluationDataset(**data)
            logger.info(f"Loaded evaluation dataset '{dataset.name}' with {dataset.size} samples")
            return dataset

        except json.JSONDecodeError as e:
            raise EvaluationError(
                f"Invalid JSON in dataset file: {path}",
                details={"error": str(e)},
            )
        except Exception as e:
            raise EvaluationError(
                f"Failed to load dataset: {e}",
                details={"file_path": str(path)},
            )

    def save_dataset(self, dataset: EvaluationDataset, file_path: Union[str, Path] = None) -> Path:
        """Save an evaluation dataset to JSON file."""
        path = Path(file_path or self.settings.eval_dataset_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dataset.model_dump(mode="json"), f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Saved dataset '{dataset.name}' ({dataset.size} samples) to {path}")
            return path

        except Exception as e:
            raise EvaluationError(f"Failed to save dataset: {e}")

    def create_dataset_from_rag_pipeline(
        self,
        queries: list[str],
        expected_outputs: list[str],
        contexts: Optional[list[list[str]]] = None,
        name: str = "RAG Evaluation Dataset",
        description: str = "",
    ) -> EvaluationDataset:
        """
        Create an evaluation dataset from query-answer pairs.

        Args:
            queries: List of user queries.
            expected_outputs: List of expected/reference answers.
            contexts: Optional list of context lists per query.
            name: Dataset name.
            description: Dataset description.

        Returns:
            EvaluationDataset object.
        """
        if len(queries) != len(expected_outputs):
            raise EvaluationError(
                "queries and expected_outputs must have the same length",
                details={"queries_count": len(queries), "expected_count": len(expected_outputs)},
            )

        samples = []
        for i, (query, expected) in enumerate(zip(queries, expected_outputs)):
            sample = EvaluationSample(
                sample_id=f"sample_{uuid.uuid4().hex[:8]}",
                user_input=query,
                expected_output=expected,
                context=contexts[i] if contexts and i < len(contexts) else None,
            )
            samples.append(sample)

        dataset = EvaluationDataset(
            dataset_id=f"dataset_{uuid.uuid4().hex[:8]}",
            name=name,
            description=description,
            samples=samples,
        )

        logger.info(f"Created dataset '{name}' with {len(samples)} samples")
        return dataset

    def validate_dataset(self, dataset: EvaluationDataset) -> dict:
        """
        Validate dataset quality and completeness.

        Returns:
            Dictionary with validation results and warnings.
        """
        issues = []
        warnings = []

        for i, sample in enumerate(dataset.samples):
            if not sample.user_input.strip():
                issues.append(f"Sample {i}: empty user_input")
            if sample.expected_output and not sample.expected_output.strip():
                warnings.append(f"Sample {i}: empty expected_output")
            if not sample.sample_id:
                issues.append(f"Sample {i}: missing sample_id")

        result = {
            "valid": len(issues) == 0,
            "total_samples": dataset.size,
            "issues": issues,
            "warnings": warnings,
            "has_expected_outputs": any(s.expected_output for s in dataset.samples),
            "has_contexts": any(s.context for s in dataset.samples),
        }

        if issues:
            logger.warning(f"Dataset validation found {len(issues)} issues")
        else:
            logger.info(f"Dataset validation passed ({dataset.size} samples)")

        return result