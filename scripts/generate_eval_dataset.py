"""
Dynamic Evaluation Dataset Generator.
Generates eval Q&A pairs from ingested documents using LLM.
Produces eval_dataset.json matching the DeepEval/RAGAS format.

Usage:
    python -m scripts.generate_eval_dataset
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from pipeline.llm_service import LLMService
from vectordb.faiss_store import FAISSVectorStore
from utils.logger import get_logger

logger = get_logger(__name__)

GENERATION_PROMPT = """You are an expert evaluation dataset creator for RAG (Retrieval Augmented Generation) systems.

Given the following document chunks from a knowledge base, generate exactly {num_questions} diverse evaluation question-answer pairs.

DOCUMENT CHUNKS:
{chunks_text}

REQUIREMENTS:
1. Each question must be answerable using ONLY the provided chunks
2. Questions should vary in complexity: simple factual, multi-step reasoning, comparison, and explanation
3. Expected answers must be comprehensive and directly grounded in the chunks
4. Include 2-4 relevant context passages (exact quotes from the chunks) for each question

OUTPUT FORMAT — respond with ONLY valid JSON, no markdown fences:
[
  {{
    "question": "The evaluation question",
    "expected_output": "Comprehensive expected answer grounded in the chunks",
    "context": ["Exact relevant passage 1 from chunks", "Exact relevant passage 2 from chunks"]
  }}
]

Generate exactly {num_questions} diverse, high-quality Q&A pairs. Output ONLY the JSON array."""


def main():
    """Generate evaluation dataset from ingested documents."""
    settings = get_settings()
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("  Dynamic Evaluation Dataset Generator")
    logger.info("=" * 60)

    # ─── Step 1: Load chunks from vector store ────────────────────
    logger.info("Loading chunks from vector store...")
    vector_store = FAISSVectorStore()
    stats = vector_store.get_stats()
    total_chunks = stats["total_vectors"]

    if total_chunks == 0:
        logger.error("Vector store is empty. Ingest documents first: python -m scripts.ingest_documents")
        sys.exit(1)

    logger.info(f"Found {total_chunks} chunks from {stats['total_documents']} documents")

    all_chunks = vector_store._chunk_contents
    all_metadata = vector_store._metadata_store

    # ─── Step 2: Sample chunk groups for question generation ──────
    # Group chunks by source document
    doc_chunks: dict[str, list[dict]] = {}
    for i, (content, meta) in enumerate(zip(all_chunks, all_metadata)):
        source = meta.get("source", "unknown")
        if source not in doc_chunks:
            doc_chunks[source] = []
        doc_chunks[source].append({"content": content, "index": i, "meta": meta})

    logger.info(f"Documents found: {list(doc_chunks.keys())}")

    # Decide how many questions per document group
    num_samples = min(int(settings.eval_batch_size) * 2, 20)  # Default: ~20 samples
    questions_per_batch = min(5, num_samples)

    # Create chunk batches for generation (3-6 related chunks per batch)
    chunk_batches = []
    for source, chunks in doc_chunks.items():
        # Slide window of 3-6 chunks
        batch_size = min(6, len(chunks))
        step = max(1, batch_size - 1)
        for start in range(0, len(chunks), step):
            batch = chunks[start:start + batch_size]
            if len(batch) >= 2:
                chunk_batches.append({"source": source, "chunks": batch})

    if not chunk_batches:
        logger.error("Not enough chunks to generate evaluation dataset")
        sys.exit(1)

    # Limit batches based on desired sample count
    max_batches = max(1, num_samples // questions_per_batch)
    if len(chunk_batches) > max_batches:
        chunk_batches = random.sample(chunk_batches, max_batches)

    logger.info(f"Will generate from {len(chunk_batches)} chunk batches ({questions_per_batch} Qs each)")

    # ─── Step 3: Generate Q&A pairs using LLM ────────────────────
    llm = LLMService(temperature=0.3, max_tokens=2048)
    all_samples = []
    sample_counter = 1

    for batch_idx, batch in enumerate(chunk_batches, 1):
        source = batch["source"]
        chunks = batch["chunks"]

        logger.info(f"Generating batch {batch_idx}/{len(chunk_batches)} from: {source}")

        # Build chunks text
        chunks_text = "\n\n".join(
            f"--- Chunk {i+1} (from {source}) ---\n{c['content']}"
            for i, c in enumerate(chunks)
        )

        prompt = GENERATION_PROMPT.format(
            num_questions=questions_per_batch,
            chunks_text=chunks_text,
        )

        try:
            response = llm.generate(
                prompt=prompt,
                system_message="You are an expert at creating evaluation datasets for RAG systems. Output ONLY valid JSON.",
            )

            # Parse response — handle possible markdown fences
            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

            qa_pairs = json.loads(response_text)

            for qa in qa_pairs:
                sample_id = f"eval_auto_{sample_counter:03d}"
                chunk_contents = [c["content"] for c in chunks]

                sample = {
                    "sample_id": sample_id,
                    "user_input": qa["question"],
                    "expected_output": qa["expected_output"],
                    "context": qa.get("context", chunk_contents[:3]),
                    "retrieval_context": None,  # Will be populated by RAG pipeline during evaluation
                }
                all_samples.append(sample)
                sample_counter += 1

            logger.info(f"  Generated {len(qa_pairs)} Q&A pairs")

        except json.JSONDecodeError as e:
            logger.warning(f"  Failed to parse LLM response for batch {batch_idx}: {e}")
            continue
        except Exception as e:
            logger.warning(f"  Generation failed for batch {batch_idx}: {e}")
            continue

    if not all_samples:
        logger.error("No samples generated. Check LLM connectivity and credits.")
        sys.exit(1)

    # ─── Step 4: Save dataset ─────────────────────────────────────
    dataset = {
        "dataset_id": f"dataset_auto_{int(time.time())}",
        "name": "Auto-Generated Evaluation Dataset",
        "description": f"Dynamically generated from {stats['total_documents']} ingested documents ({total_chunks} chunks). Questions are grounded in actual knowledge base content.",
        "version": "auto",
        "samples": all_samples,
    }

    output_path = Path(settings.eval_dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"  Generated {len(all_samples)} evaluation samples")
    logger.info(f"  Saved to: {output_path}")
    logger.info(f"  Duration: {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()