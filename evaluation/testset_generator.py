"""
RAGAS Testset Generator for enterprise-grade dynamic evaluation dataset creation.

Uses RAGAS TestsetGenerator to synthesize diverse, multi-complexity test cases
from ingested documents. Generates simple, reasoning, and multi-context questions
automatically — no hardcoded test cases needed.

Supports: OpenAI, Azure OpenAI, OpenRouter providers.

Usage:
    from evaluation.testset_generator import RAGASTestsetGenerator

    generator = RAGASTestsetGenerator()
    dataset = generator.generate(testset_size=30)
"""

import time
import uuid
from difflib import SequenceMatcher
from typing import Optional

from config.settings import get_settings
from ingest.document_loader import DocumentLoader
from schemas.evaluation import EvaluationDataset, EvaluationSample
from utils.exceptions import EvaluationError
from utils.logger import get_logger

logger = get_logger(__name__)

# ── RAGAS synthesizer name → (question_type, difficulty) mapping ──────
# RAGAS TestsetGenerator labels each generated question with a synthesizer
# name indicating the complexity type. We map these to our enterprise schema.
_SYNTHESIZER_MAP = {
    "single_hop_specific": ("simple", "easy"),
    "single_hop": ("simple", "easy"),
    "multi_hop_abstract": ("reasoning", "hard"),
    "multi_hop_specific": ("multi_context", "medium"),
    "multi_hop": ("multi_context", "medium"),
    "abstract": ("reasoning", "hard"),
    "specific": ("simple", "easy"),
}


class RAGASTestsetGenerator:
    """
    Enterprise-grade evaluation testset generator using RAGAS framework.

    Reads documents from the configured directory, builds a knowledge graph,
    and synthesizes diverse Q&A pairs with varying complexity levels:
      - Simple (single-hop factual recall)
      - Reasoning (requires inference)
      - Multi-context (needs information from multiple passages)

    Each generated sample is tagged with difficulty, question_type, and
    source_strategy metadata for enterprise audit trails.
    """

    def __init__(self, model: Optional[str] = None):
        self.settings = get_settings()
        self.model = model or self.settings.eval_model
        self._generator = None
        self._llm = None
        self._embeddings = None
        self._document_loader = DocumentLoader()
        self._initialize()

    def _initialize(self) -> None:
        """Initialize RAGAS TestsetGenerator with LLM and embeddings."""
        try:
            from ragas.testset import TestsetGenerator
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        except ImportError as e:
            raise EvaluationError(
                f"RAGAS testset generation dependencies not installed: {e}. "
                "Install with: pip install ragas langchain-openai langchain-core",
                details={"error": str(e)},
            )

        try:
            llm, embeddings = self._create_llm_and_embeddings(ChatOpenAI, OpenAIEmbeddings)

            self._llm = LangchainLLMWrapper(llm)
            self._embeddings = LangchainEmbeddingsWrapper(embeddings)

            # Initialize RAGAS TestsetGenerator — handle API variations across versions
            try:
                self._generator = TestsetGenerator(
                    llm=self._llm,
                    embedding_model=self._embeddings,
                )
            except TypeError:
                self._generator = TestsetGenerator(
                    generator_llm=self._llm,
                    generator_embeddings=self._embeddings,
                )

            logger.info("RAGAS TestsetGenerator initialized successfully")

        except EvaluationError:
            raise
        except Exception as e:
            raise EvaluationError(
                f"Failed to initialize RAGAS TestsetGenerator: {e}",
                details={"model": self.model, "error": str(e)},
            )

    def _create_llm_and_embeddings(self, ChatOpenAI, OpenAIEmbeddings):
        """Create LLM and embeddings instances based on configured provider."""
        if self.settings.use_openrouter and self.settings.openrouter_api_key:
            llm = ChatOpenAI(
                model=self.settings.openrouter_model,
                api_key=self.settings.openrouter_api_key,
                base_url=self.settings.openrouter_base_url,
                temperature=0.3,
                max_tokens=2048,
                timeout=120.0,
            )
            embeddings = OpenAIEmbeddings(
                model=self.settings.eval_embedding_model,
                api_key=self.settings.openrouter_api_key,
                base_url=self.settings.openrouter_base_url,
                timeout=60.0,
            )
            logger.info(f"RAGAS TestsetGenerator using OpenRouter: {self.settings.openrouter_model}")

        elif self.settings.azure_openai_endpoint:
            from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

            llm = AzureChatOpenAI(
                azure_endpoint=self.settings.azure_openai_endpoint,
                api_key=self.settings.azure_openai_api_key,
                api_version=self.settings.azure_openai_api_version,
                deployment_name=self.settings.azure_openai_deployment,
                temperature=0.3,
                max_tokens=2048,
            )
            embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=self.settings.azure_openai_endpoint,
                api_key=self.settings.azure_openai_api_key,
                api_version=self.settings.azure_openai_api_version,
            )
            logger.info("RAGAS TestsetGenerator using Azure OpenAI")

        else:
            llm = ChatOpenAI(
                model=self.model,
                api_key=self.settings.openai_api_key,
                temperature=0.3,
                max_tokens=2048,
                timeout=120.0,
            )
            embeddings = OpenAIEmbeddings(
                model=self.settings.eval_embedding_model,
                api_key=self.settings.openai_api_key,
            )
            logger.info(f"RAGAS TestsetGenerator using OpenAI: {self.model}")

        return llm, embeddings

    # ─── Public Generation Methods ────────────────────────────────────

    def generate(
        self,
        testset_size: Optional[int] = None,
        documents_dir: Optional[str] = None,
    ) -> EvaluationDataset:
        """
        Generate evaluation dataset from documents in the configured directory.

        Loads documents → converts to LangChain format → runs RAGAS TestsetGenerator
        → converts output to EvaluationDataset with enterprise metadata.

        Args:
            testset_size: Number of test cases to generate. Defaults to settings.testset_size.
            documents_dir: Documents directory path. Defaults to settings.documents_directory.

        Returns:
            EvaluationDataset with generated test cases tagged with difficulty,
            question_type, and source_strategy.

        Raises:
            EvaluationError: If no documents found or generation fails.
        """
        testset_size = testset_size or self.settings.testset_size
        docs_dir = documents_dir or self.settings.documents_directory

        logger.info(f"Loading documents from: {docs_dir}")
        documents = self._document_loader.load_directory(docs_dir)

        if not documents:
            raise EvaluationError(
                f"No documents found in {docs_dir}. "
                "Run ingestion first: python -m scripts.ingest_documents",
                details={"directory": str(docs_dir)},
            )

        logger.info(f"Loaded {len(documents)} documents ({sum(len(d.content) for d in documents)} chars total)")
        return self.generate_from_documents(documents, testset_size)

    def generate_from_documents(
        self,
        documents: list,
        testset_size: Optional[int] = None,
    ) -> EvaluationDataset:
        """
        Generate evaluation dataset from pre-loaded Document objects.

        Args:
            documents: List of Document objects (our schema).
            testset_size: Number of test cases to generate.

        Returns:
            EvaluationDataset with generated test cases.
        """
        testset_size = testset_size or self.settings.testset_size
        start_time = time.time()

        # Convert to LangChain Document format
        logger.info(f"Converting {len(documents)} documents to LangChain format...")
        lc_docs = self._convert_to_langchain_docs(documents)

        # Generate testset using RAGAS
        logger.info(f"Generating {testset_size} test cases using RAGAS TestsetGenerator...")
        logger.info("This may take several minutes depending on document size and LLM speed.")

        try:
            testset = self._run_generation(lc_docs, testset_size)
        except Exception as e:
            raise EvaluationError(
                f"RAGAS testset generation failed: {e}",
                details={
                    "testset_size": testset_size,
                    "num_docs": len(documents),
                    "error": str(e),
                },
            )

        elapsed = time.time() - start_time
        logger.info(f"RAGAS generation completed in {elapsed:.1f}s")

        # Convert RAGAS output to our schema
        dataset = self._convert_testset_to_dataset(testset)

        # Post-processing: deduplicate
        original_count = dataset.size
        dataset = self._deduplicate_samples(dataset)
        if dataset.size < original_count:
            logger.info(f"Deduplication removed {original_count - dataset.size} near-duplicate samples")

        logger.info(
            f"Generated {dataset.size} evaluation samples "
            f"(simple={sum(1 for s in dataset.samples if s.question_type == 'simple')}, "
            f"reasoning={sum(1 for s in dataset.samples if s.question_type == 'reasoning')}, "
            f"multi_context={sum(1 for s in dataset.samples if s.question_type == 'multi_context')})"
        )
        return dataset

    # ─── Internal Helpers ─────────────────────────────────────────────

    def _run_generation(self, lc_docs: list, testset_size: int):
        """Run RAGAS TestsetGenerator with fallback for API version differences."""
        # Try standard RAGAS 0.2.x API
        try:
            return self._generator.generate_with_langchain_docs(
                documents=lc_docs,
                testset_size=testset_size,
            )
        except AttributeError:
            pass

        # Fallback: try alternate method signatures
        try:
            return self._generator.generate(
                documents=lc_docs,
                test_size=testset_size,
            )
        except (AttributeError, TypeError):
            pass

        # Last fallback
        return self._generator.generate(
            test_size=testset_size,
            documents=lc_docs,
        )

    def _convert_to_langchain_docs(self, documents: list) -> list:
        """Convert our Document objects to LangChain Document format."""
        from langchain_core.documents import Document as LCDocument

        lc_docs = []
        for doc in documents:
            lc_doc = LCDocument(
                page_content=doc.content,
                metadata={
                    "source": doc.metadata.source,
                    "file_type": doc.metadata.file_type,
                    "document_id": doc.document_id,
                },
            )
            lc_docs.append(lc_doc)

        return lc_docs

    def _convert_testset_to_dataset(self, testset) -> EvaluationDataset:
        """Convert RAGAS Testset output to our EvaluationDataset schema."""
        try:
            df = testset.to_pandas()
        except Exception:
            try:
                rows = testset.to_list()
                import pandas as pd
                df = pd.DataFrame(rows)
            except Exception as e:
                raise EvaluationError(
                    f"Failed to convert RAGAS testset output: {e}",
                    details={"error": str(e)},
                )

        if df.empty:
            raise EvaluationError(
                "RAGAS TestsetGenerator returned empty testset. "
                "Ensure documents have sufficient content for question generation.",
            )

        samples = []
        for idx, row in df.iterrows():
            sample_id = f"eval_ragas_{idx + 1:03d}"

            # Extract synthesizer info for metadata
            synthesizer_name = str(row.get("synthesizer_name", "unknown"))
            question_type, difficulty = self._map_synthesizer_to_metadata(synthesizer_name)

            # Extract contexts — handle various RAGAS output formats
            reference_contexts = self._extract_contexts(row)

            # Build evaluation sample
            user_input = str(row.get("user_input", ""))
            expected_output = str(row.get("reference", ""))

            if not user_input.strip():
                logger.warning(f"Skipping sample {sample_id}: empty user_input")
                continue

            sample = EvaluationSample(
                sample_id=sample_id,
                user_input=user_input,
                expected_output=expected_output if expected_output.strip() else None,
                context=reference_contexts if reference_contexts else None,
                retrieval_context=None,  # Populated by RAG pipeline during evaluation
                difficulty=difficulty,
                question_type=question_type,
                source_strategy="ragas_synthetic",
                auto_generated=True,
                human_reviewed=False,
            )
            samples.append(sample)

        dataset = EvaluationDataset(
            dataset_id=f"dataset_ragas_{uuid.uuid4().hex[:12]}",
            name="RAGAS Auto-Generated Evaluation Dataset",
            description=(
                f"Dynamically generated using RAGAS TestsetGenerator. "
                f"Contains {len(samples)} diverse test cases with varying complexity "
                f"(simple, reasoning, multi-context)."
            ),
            version="auto",
            samples=samples,
        )

        return dataset

    def _extract_contexts(self, row) -> list[str]:
        """Extract context list from a RAGAS testset row, handling format variations."""
        reference_contexts = row.get("reference_contexts", [])

        if reference_contexts is None:
            return []
        if isinstance(reference_contexts, str):
            return [reference_contexts] if reference_contexts.strip() else []
        if hasattr(reference_contexts, "tolist"):
            reference_contexts = reference_contexts.tolist()
        if isinstance(reference_contexts, list):
            return [str(c) for c in reference_contexts if c]
        return []

    def _map_synthesizer_to_metadata(self, synthesizer_name: str) -> tuple[str, str]:
        """Map RAGAS synthesizer name to (question_type, difficulty)."""
        name_lower = synthesizer_name.lower().replace(" ", "_")

        for key, (q_type, diff) in _SYNTHESIZER_MAP.items():
            if key in name_lower:
                return q_type, diff

        return "unknown", "medium"

    def _deduplicate_samples(
        self,
        dataset: EvaluationDataset,
        threshold: float = 0.85,
    ) -> EvaluationDataset:
        """
        Remove near-duplicate questions using string similarity.
        Uses SequenceMatcher for fast, cost-free deduplication.
        """
        if not dataset.samples:
            return dataset

        unique_samples = [dataset.samples[0]]

        for candidate in dataset.samples[1:]:
            candidate_norm = candidate.user_input.strip().lower()
            is_duplicate = False

            for existing in unique_samples:
                existing_norm = existing.user_input.strip().lower()
                similarity = SequenceMatcher(None, candidate_norm, existing_norm).ratio()
                if similarity >= threshold:
                    is_duplicate = True
                    logger.debug(
                        f"Dedup: removed '{candidate.user_input[:60]}...' "
                        f"(similarity={similarity:.2f} with '{existing.user_input[:60]}...')"
                    )
                    break

            if not is_duplicate:
                unique_samples.append(candidate)

        return EvaluationDataset(
            dataset_id=dataset.dataset_id,
            name=dataset.name,
            description=dataset.description,
            version=dataset.version,
            samples=unique_samples,
            created_at=dataset.created_at,
        )
