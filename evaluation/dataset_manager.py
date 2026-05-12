"""
Dataset manager for evaluation data.
Handles loading, saving, managing, generating, and merging evaluation datasets.
Supports enterprise workflows: RAGAS generation, deduplication, quality validation,
and dataset versioning.
"""

import json
import uuid
from difflib import SequenceMatcher
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

    # ─── Enterprise Dataset Generation ────────────────────────────────

    def generate_dataset(
        self,
        strategy: Optional[str] = None,
        testset_size: Optional[int] = None,
        documents_dir: Optional[str] = None,
    ) -> EvaluationDataset:
        """
        Generate an evaluation dataset using the specified strategy.

        Supports three strategies:
          - "ragas": RAGAS TestsetGenerator (recommended for enterprise)
          - "document": Document-grounded LLM generation (existing approach)
          - "hybrid": Both strategies merged with deduplication

        Args:
            strategy: Generation strategy. Defaults to settings.testset_strategy.
            testset_size: Number of test cases. Defaults to settings.testset_size.
            documents_dir: Source documents directory.

        Returns:
            EvaluationDataset with generated test cases.
        """
        strategy = strategy or self.settings.testset_strategy
        testset_size = testset_size or self.settings.testset_size

        logger.info(f"Generating evaluation dataset: strategy={strategy}, size={testset_size}")

        if strategy == "ragas":
            return self._generate_ragas(testset_size, documents_dir)
        elif strategy == "document":
            return self._generate_document_grounded(testset_size, documents_dir)
        elif strategy == "hybrid":
            return self._generate_hybrid(testset_size, documents_dir)
        else:
            raise EvaluationError(
                f"Unknown testset generation strategy: '{strategy}'. "
                "Supported: 'ragas', 'document', 'hybrid'",
            )

    def _generate_ragas(
        self,
        testset_size: int,
        documents_dir: Optional[str] = None,
    ) -> EvaluationDataset:
        """Generate using RAGAS TestsetGenerator."""
        from evaluation.testset_generator import RAGASTestsetGenerator

        generator = RAGASTestsetGenerator()
        return generator.generate(testset_size=testset_size, documents_dir=documents_dir)

    def _generate_document_grounded(
        self,
        testset_size: int,
        documents_dir: Optional[str] = None,
    ) -> EvaluationDataset:
        """Generate using document chunks + LLM (existing approach)."""
        from pipeline.llm_service import LLMService
        from vectordb.faiss_store import FAISSVectorStore

        vector_store = FAISSVectorStore()
        stats = vector_store.get_stats()

        if stats["total_vectors"] == 0:
            raise EvaluationError(
                "Vector store is empty. Run ingestion first: python -m scripts.ingest_documents"
            )

        all_chunks = vector_store._chunk_contents
        all_metadata = vector_store._metadata_store

        # Group chunks by source document
        doc_chunks: dict[str, list[dict]] = {}
        for i, (content, meta) in enumerate(zip(all_chunks, all_metadata)):
            source = meta.get("source", "unknown")
            if source not in doc_chunks:
                doc_chunks[source] = []
            doc_chunks[source].append({"content": content, "index": i, "meta": meta})

        # Create chunk batches (3-6 related chunks per batch)
        chunk_batches = []
        for source, chunks in doc_chunks.items():
            batch_size = min(6, len(chunks))
            step = max(1, batch_size - 1)
            for start in range(0, len(chunks), step):
                batch = chunks[start:start + batch_size]
                if len(batch) >= 2:
                    chunk_batches.append({"source": source, "chunks": batch})

        if not chunk_batches:
            raise EvaluationError("Not enough chunks to generate evaluation dataset")

        import random
        questions_per_batch = min(5, testset_size)
        max_batches = max(1, testset_size // questions_per_batch)
        if len(chunk_batches) > max_batches:
            chunk_batches = random.sample(chunk_batches, max_batches)

        llm = LLMService(temperature=0.3, max_tokens=2048)
        all_samples = []
        sample_counter = 1

        generation_prompt = (
            "You are an expert evaluation dataset creator for RAG systems.\n\n"
            "Given the following document chunks, generate exactly {num_questions} "
            "diverse evaluation question-answer pairs.\n\n"
            "DOCUMENT CHUNKS:\n{chunks_text}\n\n"
            "REQUIREMENTS:\n"
            "1. Each question must be answerable using ONLY the provided chunks\n"
            "2. Questions should vary in complexity: simple factual, multi-step reasoning, comparison\n"
            "3. Expected answers must be comprehensive and directly grounded in the chunks\n"
            "4. Include 2-4 relevant context passages (exact quotes) for each question\n\n"
            'OUTPUT FORMAT — respond with ONLY valid JSON, no markdown fences:\n'
            '[{{"question": "...", "expected_output": "...", "context": ["..."]}}]\n\n'
            "Generate exactly {num_questions} diverse, high-quality Q&A pairs. Output ONLY the JSON array."
        )

        import json as json_mod

        for batch_idx, batch in enumerate(chunk_batches, 1):
            source = batch["source"]
            chunks = batch["chunks"]

            chunks_text = "\n\n".join(
                f"--- Chunk {i+1} (from {source}) ---\n{c['content']}"
                for i, c in enumerate(chunks)
            )

            prompt = generation_prompt.format(
                num_questions=questions_per_batch,
                chunks_text=chunks_text,
            )

            try:
                response = llm.generate(
                    prompt=prompt,
                    system_message="You are an expert at creating evaluation datasets for RAG systems. Output ONLY valid JSON.",
                )

                response_text = response.strip()
                if response_text.startswith("```"):
                    response_text = response_text.split("\n", 1)[1]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()

                qa_pairs = json_mod.loads(response_text)

                for qa in qa_pairs:
                    sample_id = f"eval_doc_{sample_counter:03d}"
                    chunk_contents = [c["content"] for c in chunks]

                    sample = EvaluationSample(
                        sample_id=sample_id,
                        user_input=qa["question"],
                        expected_output=qa["expected_output"],
                        context=qa.get("context", chunk_contents[:3]),
                        retrieval_context=None,
                        difficulty=None,
                        question_type=None,
                        source_strategy="document_grounded",
                        auto_generated=True,
                        human_reviewed=False,
                    )
                    all_samples.append(sample)
                    sample_counter += 1

                logger.info(f"  Batch {batch_idx}: generated {len(qa_pairs)} Q&A pairs")

            except json_mod.JSONDecodeError as e:
                logger.warning(f"  Batch {batch_idx}: failed to parse LLM response: {e}")
                continue
            except Exception as e:
                logger.warning(f"  Batch {batch_idx}: generation failed: {e}")
                continue

        if not all_samples:
            raise EvaluationError("No samples generated. Check LLM connectivity.")

        import time

        return EvaluationDataset(
            dataset_id=f"dataset_doc_{uuid.uuid4().hex[:12]}",
            name="Document-Grounded Evaluation Dataset",
            description=(
                f"Generated from {stats['total_documents']} ingested documents "
                f"({stats['total_vectors']} chunks) using LLM-based question generation."
            ),
            version="auto",
            samples=all_samples,
        )

    def _generate_hybrid(
        self,
        testset_size: int,
        documents_dir: Optional[str] = None,
    ) -> EvaluationDataset:
        """Generate using both RAGAS and document-grounded, then merge."""
        ragas_size = testset_size // 2
        doc_size = testset_size - ragas_size

        datasets = []

        # Generate RAGAS testset
        try:
            logger.info(f"Hybrid: generating {ragas_size} RAGAS samples...")
            ragas_dataset = self._generate_ragas(ragas_size, documents_dir)
            datasets.append(ragas_dataset)
            logger.info(f"Hybrid: RAGAS generated {ragas_dataset.size} samples")
        except EvaluationError as e:
            logger.warning(f"Hybrid: RAGAS generation failed: {e.message}. Falling back to document-only.")
            doc_size = testset_size

        # Generate document-grounded testset
        try:
            logger.info(f"Hybrid: generating {doc_size} document-grounded samples...")
            doc_dataset = self._generate_document_grounded(doc_size, documents_dir)
            datasets.append(doc_dataset)
            logger.info(f"Hybrid: document-grounded generated {doc_dataset.size} samples")
        except EvaluationError as e:
            logger.warning(f"Hybrid: document-grounded generation failed: {e.message}")

        if not datasets:
            raise EvaluationError("Hybrid generation failed: both strategies produced no results.")

        if len(datasets) == 1:
            return datasets[0]

        return self.merge_datasets(datasets, name="Hybrid Evaluation Dataset", deduplicate=True)

    # ─── Enterprise Dataset Operations ────────────────────────────────

    def deduplicate_samples(
        self,
        dataset: EvaluationDataset,
        threshold: float = 0.85,
    ) -> EvaluationDataset:
        """
        Remove near-duplicate questions using string similarity.

        Uses SequenceMatcher for fast, cost-free deduplication.
        Questions with similarity >= threshold are considered duplicates;
        the first occurrence is kept.

        Args:
            dataset: Dataset to deduplicate.
            threshold: Similarity threshold (0.0 to 1.0). Default 0.85.

        Returns:
            New EvaluationDataset with duplicates removed.
        """
        if dataset.size <= 1:
            return dataset

        unique_samples = [dataset.samples[0]]
        removed_count = 0

        for candidate in dataset.samples[1:]:
            candidate_norm = candidate.user_input.strip().lower()
            is_duplicate = False

            for existing in unique_samples:
                existing_norm = existing.user_input.strip().lower()
                similarity = SequenceMatcher(None, candidate_norm, existing_norm).ratio()
                if similarity >= threshold:
                    is_duplicate = True
                    removed_count += 1
                    break

            if not is_duplicate:
                unique_samples.append(candidate)

        if removed_count > 0:
            logger.info(f"Deduplication: removed {removed_count} duplicates ({dataset.size} → {len(unique_samples)})")

        return EvaluationDataset(
            dataset_id=dataset.dataset_id,
            name=dataset.name,
            description=dataset.description,
            version=dataset.version,
            samples=unique_samples,
            created_at=dataset.created_at,
        )

    def merge_datasets(
        self,
        datasets: list[EvaluationDataset],
        name: Optional[str] = None,
        description: Optional[str] = None,
        deduplicate: bool = True,
    ) -> EvaluationDataset:
        """
        Merge multiple evaluation datasets into one.

        Combines samples from all datasets, optionally deduplicating.
        Re-indexes sample IDs to ensure uniqueness.

        Args:
            datasets: List of EvaluationDataset objects to merge.
            name: Name for the merged dataset. Auto-generated if not provided.
            description: Description for merged dataset.
            deduplicate: Whether to deduplicate after merging. Default True.

        Returns:
            Merged EvaluationDataset.
        """
        if not datasets:
            raise EvaluationError("No datasets provided for merging")

        if len(datasets) == 1:
            return datasets[0]

        all_samples = []
        source_names = []

        for ds in datasets:
            all_samples.extend(ds.samples)
            source_names.append(f"{ds.name} ({ds.size})")

        # Re-index sample IDs to ensure uniqueness
        for idx, sample in enumerate(all_samples, 1):
            sample.sample_id = f"eval_merged_{idx:03d}"

        merged = EvaluationDataset(
            dataset_id=f"dataset_merged_{uuid.uuid4().hex[:12]}",
            name=name or f"Merged Dataset ({len(datasets)} sources)",
            description=description or f"Merged from: {', '.join(source_names)}",
            version="auto",
            samples=all_samples,
        )

        if deduplicate:
            merged = self.deduplicate_samples(merged)

        logger.info(f"Merged {len(datasets)} datasets → {merged.size} samples")
        return merged

    def validate_quality(self, dataset: EvaluationDataset) -> tuple[EvaluationDataset, dict]:
        """
        Validate sample quality using heuristic checks.

        Performs structural and content quality checks:
          - Non-empty user_input and expected_output
          - Minimum length thresholds
          - Context relevance (keyword overlap)
          - Question diversity

        Args:
            dataset: Dataset to validate.

        Returns:
            Tuple of (filtered dataset with only valid samples, validation report dict).
        """
        valid_samples = []
        rejected = []
        warnings = []

        for sample in dataset.samples:
            issues = []

            # Check user_input
            if not sample.user_input or len(sample.user_input.strip()) < 10:
                issues.append("user_input too short (< 10 chars)")

            # Check expected_output
            if sample.expected_output and len(sample.expected_output.strip()) < 20:
                issues.append("expected_output too short (< 20 chars)")

            # Check for question mark (basic question format)
            if sample.user_input and "?" not in sample.user_input:
                warnings.append(f"Sample {sample.sample_id}: no question mark in user_input")

            # Check context-answer overlap (basic relevance check)
            if sample.expected_output and sample.context:
                context_text = " ".join(sample.context).lower()
                answer_words = set(sample.expected_output.lower().split())
                # At least 20% of answer words should appear in context
                overlap = sum(1 for w in answer_words if w in context_text and len(w) > 3)
                if len(answer_words) > 0 and overlap / len(answer_words) < 0.15:
                    issues.append("low context-answer relevance")

            if issues:
                rejected.append({
                    "sample_id": sample.sample_id,
                    "question": sample.user_input[:80],
                    "issues": issues,
                })
            else:
                valid_samples.append(sample)

        report = {
            "total_input": dataset.size,
            "total_valid": len(valid_samples),
            "total_rejected": len(rejected),
            "rejected_samples": rejected,
            "warnings": warnings,
            "quality_score": len(valid_samples) / max(dataset.size, 1),
        }

        if rejected:
            logger.info(
                f"Quality validation: {len(valid_samples)}/{dataset.size} passed, "
                f"{len(rejected)} rejected"
            )

        validated_dataset = EvaluationDataset(
            dataset_id=dataset.dataset_id,
            name=dataset.name,
            description=dataset.description,
            version=dataset.version,
            samples=valid_samples,
            created_at=dataset.created_at,
        )

        return validated_dataset, report