"""
Unit tests for evaluation modules (DeepEval and RAGAS).
Tests evaluation logic without making real API calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from schemas.evaluation import (
    EvaluationDataset,
    EvaluationReport,
    EvaluationResult,
    EvaluationSample,
    MetricResult,
)
from utils.exceptions import EvaluationError


class TestDatasetManager:
    """Tests for DatasetManager class."""

    def test_load_dataset(self, eval_dataset_file, monkeypatch):
        """Test loading evaluation dataset from file."""
        monkeypatch.setenv("EVAL_DATASET_PATH", str(eval_dataset_file))

        from evaluation.dataset_manager import DatasetManager

        manager = DatasetManager()
        dataset = manager.load_dataset(eval_dataset_file)

        assert dataset.dataset_id == "test_ds_001"
        assert dataset.size == 1
        assert dataset.samples[0].user_input == "What is Python?"

    def test_load_nonexistent_dataset_raises_error(self):
        """Test that loading non-existent file raises error."""
        from evaluation.dataset_manager import DatasetManager

        manager = DatasetManager()
        with pytest.raises(EvaluationError):
            manager.load_dataset("/nonexistent/path.json")

    def test_create_dataset_from_pairs(self):
        """Test creating dataset from query-answer pairs."""
        from evaluation.dataset_manager import DatasetManager

        manager = DatasetManager()
        dataset = manager.create_dataset_from_rag_pipeline(
            queries=["What is AI?", "What is ML?"],
            expected_outputs=["AI is artificial intelligence.", "ML is machine learning."],
            name="Test Dataset",
        )

        assert dataset.size == 2
        assert dataset.name == "Test Dataset"
        assert dataset.samples[0].user_input == "What is AI?"

    def test_create_dataset_mismatched_lengths_raises_error(self):
        """Test that mismatched query/answer lengths raise error."""
        from evaluation.dataset_manager import DatasetManager

        manager = DatasetManager()
        with pytest.raises(EvaluationError):
            manager.create_dataset_from_rag_pipeline(
                queries=["Q1", "Q2"],
                expected_outputs=["A1"],
            )

    def test_validate_dataset(self, sample_eval_dataset):
        """Test dataset validation."""
        from evaluation.dataset_manager import DatasetManager

        manager = DatasetManager()
        result = manager.validate_dataset(sample_eval_dataset)

        assert result["valid"] is True
        assert result["total_samples"] == 2
        assert result["has_expected_outputs"] is True

    def test_save_and_load_dataset(self, tmp_path):
        """Test saving and loading a dataset."""
        from evaluation.dataset_manager import DatasetManager

        manager = DatasetManager()
        dataset = manager.create_dataset_from_rag_pipeline(
            queries=["Test Q"],
            expected_outputs=["Test A"],
            name="Save Test",
        )

        save_path = tmp_path / "saved_dataset.json"
        manager.save_dataset(dataset, save_path)

        loaded = manager.load_dataset(save_path)
        assert loaded.name == "Save Test"
        assert loaded.size == 1


class TestDeepEvalEvaluator:
    """Tests for DeepEvalEvaluator (mocked - no real API calls)."""

    @pytest.fixture
    def mock_deepeval_metrics(self):
        """Mock DeepEval metrics."""
        with patch("evaluation.deepeval_evaluator.DeepEvalEvaluator._initialize_metrics") as mock_init:
            mock_init.return_value = None
            from evaluation.deepeval_evaluator import DeepEvalEvaluator
            evaluator = DeepEvalEvaluator.__new__(DeepEvalEvaluator)
            evaluator.settings = MagicMock()
            evaluator.settings.deepeval_metrics_list = ["answer_relevancy", "faithfulness"]
            evaluator.settings.deepeval_threshold = 0.7
            evaluator.settings.eval_model = "gpt-4o-mini"
            evaluator.settings.eval_batch_size = 5
            evaluator.metrics_config = ["answer_relevancy", "faithfulness"]
            evaluator.threshold = 0.7
            evaluator.model = "gpt-4o-mini"

            # Create mock metrics with async a_measure
            mock_metric_1 = MagicMock()
            mock_metric_1.score = 0.85
            mock_metric_1.reason = "Good relevancy"
            mock_metric_1.a_measure = AsyncMock()

            mock_metric_2 = MagicMock()
            mock_metric_2.score = 0.90
            mock_metric_2.reason = "Faithful to context"
            mock_metric_2.a_measure = AsyncMock()

            evaluator._metrics = {
                "answer_relevancy": mock_metric_1,
                "faithfulness": mock_metric_2,
            }

            return evaluator

    @pytest.mark.asyncio
    async def test_evaluate_sample(self, mock_deepeval_metrics, sample_eval_dataset):
        """Test evaluating a single sample."""
        with patch("evaluation.deepeval_evaluator.LLMTestCase") as MockTestCase:
            MockTestCase.return_value = MagicMock()

            result = await mock_deepeval_metrics.aevaluate_sample(sample_eval_dataset.samples[0])

            assert isinstance(result, EvaluationResult)
            assert len(result.metrics) == 2
            assert result.overall_passed is True
            assert result.average_score > 0.7

    @pytest.mark.asyncio
    async def test_evaluate_dataset(self, mock_deepeval_metrics, sample_eval_dataset):
        """Test evaluating a full dataset."""
        with patch("evaluation.deepeval_evaluator.LLMTestCase") as MockTestCase:
            MockTestCase.return_value = MagicMock()

            report = await mock_deepeval_metrics.aevaluate_dataset(sample_eval_dataset)

            assert isinstance(report, EvaluationReport)
            assert report.framework == "deepeval"
            assert report.total_samples == 2
            assert report.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_evaluate_sample_without_output_raises_error(self, mock_deepeval_metrics):
        """Test that sample without actual_output raises error."""
        sample = EvaluationSample(
            sample_id="no_output",
            user_input="Test question",
            actual_output=None,
        )

        with pytest.raises(EvaluationError):
            await mock_deepeval_metrics.aevaluate_sample(sample)


class TestRagasEvaluator:
    """Tests for RagasEvaluator (mocked - no real API calls)."""

    @pytest.fixture
    def mock_ragas_evaluator(self):
        """Mock RAGAS evaluator."""
        with patch("evaluation.ragas_evaluator.RagasEvaluator._initialize") as mock_init:
            mock_init.return_value = None
            from evaluation.ragas_evaluator import RagasEvaluator
            evaluator = RagasEvaluator.__new__(RagasEvaluator)
            evaluator.settings = MagicMock()
            evaluator.settings.ragas_metrics_list = ["faithfulness", "answer_relevancy"]
            evaluator.settings.ragas_threshold = 0.7
            evaluator.settings.eval_model = "gpt-4o-mini"
            evaluator.settings.eval_batch_size = 5
            evaluator.metrics_config = ["faithfulness", "answer_relevancy"]
            evaluator.threshold = 0.7
            evaluator.model = "gpt-4o-mini"

            # Mock metrics with async single_turn_ascore
            mock_metric_1 = MagicMock()
            mock_metric_1.single_turn_ascore = AsyncMock(return_value=0.88)

            mock_metric_2 = MagicMock()
            mock_metric_2.single_turn_ascore = AsyncMock(return_value=0.92)

            evaluator._metrics = {
                "faithfulness": mock_metric_1,
                "answer_relevancy": mock_metric_2,
            }

            return evaluator

    @pytest.mark.asyncio
    async def test_evaluate_sample(self, mock_ragas_evaluator, sample_eval_dataset):
        """Test evaluating a single sample with RAGAS."""
        with patch("evaluation.ragas_evaluator.SingleTurnSample") as MockSample:
            MockSample.return_value = MagicMock()

            result = await mock_ragas_evaluator.aevaluate_sample(sample_eval_dataset.samples[0])

            assert isinstance(result, EvaluationResult)
            assert len(result.metrics) == 2
            assert result.overall_passed is True

    @pytest.mark.asyncio
    async def test_evaluate_dataset(self, mock_ragas_evaluator, sample_eval_dataset):
        """Test evaluating a full dataset with RAGAS."""
        with patch("evaluation.ragas_evaluator.SingleTurnSample") as MockSample:
            MockSample.return_value = MagicMock()

            report = await mock_ragas_evaluator.aevaluate_dataset(sample_eval_dataset)

            assert isinstance(report, EvaluationReport)
            assert report.framework == "ragas"
            assert report.total_samples == 2

    def test_generate_summary(self, mock_ragas_evaluator):
        """Test summary generation from results."""
        results = [
            EvaluationResult(
                sample_id="s1",
                metrics=[
                    MetricResult(metric_name="faithfulness", score=0.9, threshold=0.7, passed=True),
                    MetricResult(metric_name="answer_relevancy", score=0.8, threshold=0.7, passed=True),
                ],
                overall_passed=True,
            ),
            EvaluationResult(
                sample_id="s2",
                metrics=[
                    MetricResult(metric_name="faithfulness", score=0.5, threshold=0.7, passed=False),
                    MetricResult(metric_name="answer_relevancy", score=0.6, threshold=0.7, passed=False),
                ],
                overall_passed=False,
            ),
        ]

        summary = mock_ragas_evaluator._generate_summary(results)

        assert summary["total_samples"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["overall_pass_rate"] == 0.5
        assert "faithfulness" in summary["metrics"]
        assert summary["metrics"]["faithfulness"]["average_score"] == 0.7