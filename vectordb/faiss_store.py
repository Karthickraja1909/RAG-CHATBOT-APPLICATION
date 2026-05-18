"""
FAISS vector store module for GenAI RAG System.
Handles storing, indexing, and retrieving document embeddings using FAISS.
Supports hybrid search (BM25 sparse + dense vector fusion).
"""

import math
import pickle
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Union

import numpy as np

from config.settings import get_settings
from schemas.documents import DocumentChunk
from schemas.pipeline import RetrievalResult
from utils.exceptions import VectorStoreError
from utils.logger import get_logger
from vectordb.embedding_service import EmbeddingService

logger = get_logger(__name__)


# ─── Lightweight BM25 implementation (no external dependency) ─────
class _BM25:
    """Okapi BM25 scoring for sparse keyword search."""

    _STOP_WORDS = frozenset(
        "a an and are as at be by for from has have in is it of on or that the to was with".split()
    )

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._corpus_size = 0
        self._avg_dl = 0.0
        self._doc_freqs: Counter = Counter()
        self._doc_lens: list[int] = []
        self._term_freqs: list[Counter] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in tokens if t not in _BM25._STOP_WORDS and len(t) > 1]

    def index(self, documents: list[str]) -> None:
        self._corpus_size = len(documents)
        self._term_freqs = []
        self._doc_lens = []
        self._doc_freqs = Counter()

        for doc in documents:
            tokens = self._tokenize(doc)
            tf = Counter(tokens)
            self._term_freqs.append(tf)
            self._doc_lens.append(len(tokens))
            self._doc_freqs.update(tf.keys())

        self._avg_dl = sum(self._doc_lens) / self._corpus_size if self._corpus_size else 1.0

    def score(self, query: str) -> list[float]:
        query_terms = self._tokenize(query)
        scores = [0.0] * self._corpus_size

        for term in query_terms:
            if term not in self._doc_freqs:
                continue
            df = self._doc_freqs[term]
            idf = math.log((self._corpus_size - df + 0.5) / (df + 0.5) + 1.0)

            for i in range(self._corpus_size):
                tf = self._term_freqs[i].get(term, 0)
                dl = self._doc_lens[i]
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (1 - self._b + self._b * dl / self._avg_dl)
                scores[i] += idf * numerator / denominator

        return scores


class FAISSVectorStore:
    """FAISS-based vector store for document retrieval."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        index_path: Optional[str] = None,
    ):
        self.settings = get_settings()
        self.embedding_service = embedding_service or EmbeddingService()
        self.index_path = Path(index_path or self.settings.faiss_index_path)
        self.dimension = self.settings.embedding_dimension

        self._index = None
        self._metadata_store: list[dict] = []
        self._chunk_contents: list[str] = []

        # BM25 sparse index (built lazily after loading or adding chunks)
        self._bm25: Optional[_BM25] = None

        self._initialize_index()

    def _initialize_index(self) -> None:
        """Initialize or load FAISS index."""
        try:
            import faiss

            if self._index_exists():
                self._load_index()
                self._rebuild_bm25()
            else:
                self._index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine with normalized vectors)
                logger.info(f"Created new FAISS index (dimension={self.dimension})")

        except ImportError:
            raise VectorStoreError(
                "FAISS is required. Install with: pip install faiss-cpu"
            )

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of DocumentChunk objects to store.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        start_time = time.time()

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        # Normalize vectors for cosine similarity
        vectors = np.array(embeddings, dtype=np.float32)
        faiss = self._get_faiss()
        faiss.normalize_L2(vectors)

        # Add to index
        self._index.add(vectors)

        # Store metadata and contents
        for chunk in chunks:
            self._metadata_store.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source": chunk.metadata.source,
                "file_type": chunk.metadata.file_type,
                "chunk_index": chunk.metadata.chunk_index,
            })
            self._chunk_contents.append(chunk.content)

        elapsed = time.time() - start_time
        logger.info(f"Added {len(chunks)} chunks to FAISS index in {elapsed:.2f}s (total: {self._index.ntotal})")

        # Auto-save
        self._save_index()
        self._rebuild_bm25()

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[RetrievalResult]:
        """
        Search the vector store for similar documents.
        Supports hybrid search (BM25 + dense) when enabled in settings.
        Returns deduplicated results.
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("Vector store is empty. No results to return.")
            return []

        top_k = top_k or self.settings.similarity_top_k
        threshold = threshold or self.settings.similarity_threshold

        # Fetch extra candidates for deduplication, then trim to top_k
        fetch_k = min(top_k * 2, self._index.ntotal)

        start_time = time.time()

        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        query_vector = np.array([query_embedding], dtype=np.float32)

        faiss = self._get_faiss()
        faiss.normalize_L2(query_vector)

        # Dense search
        scores, indices = self._index.search(query_vector, fetch_k)

        # Hybrid fusion (if enabled and BM25 index exists)
        use_hybrid = self.settings.hybrid_search_enabled and self._bm25 is not None
        bm25_scores: Optional[list[float]] = None
        if use_hybrid:
            bm25_scores = self._bm25.score(query)
            # Normalize BM25 scores to [0, 1]
            bm25_max = max(bm25_scores) if bm25_scores else 1.0
            if bm25_max > 0:
                bm25_scores = [s / bm25_max for s in bm25_scores]

        alpha = self.settings.hybrid_search_alpha  # 1.0 = pure dense, 0.0 = pure BM25

        # Build results with deduplication
        results = []
        seen_content_hashes = set()

        # Combine scores for candidates returned by dense search
        candidate_scores: list[tuple[int, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            final_score = float(score)
            if use_hybrid and bm25_scores:
                final_score = alpha * float(score) + (1 - alpha) * bm25_scores[idx]
            candidate_scores.append((int(idx), final_score))

        # If hybrid, also consider top BM25-only candidates not in dense results
        if use_hybrid and bm25_scores:
            dense_idxs = {int(idx) for idx in indices[0] if idx != -1}
            bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
            for bm25_idx, bm25_s in bm25_ranked[:fetch_k]:
                if bm25_idx not in dense_idxs:
                    # Reconstruct dense score for this idx
                    vec = np.array([self._index.reconstruct(bm25_idx)], dtype=np.float32)
                    dense_s = float(np.dot(query_vector[0], vec[0]))
                    final = alpha * dense_s + (1 - alpha) * bm25_s
                    candidate_scores.append((bm25_idx, final))

        # Sort all candidates by final score descending
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        for idx, final_score in candidate_scores:
            if final_score < threshold:
                continue

            content = self._chunk_contents[idx]
            content_key = content[:200].strip().lower()
            if content_key in seen_content_hashes:
                continue
            seen_content_hashes.add(content_key)

            metadata = self._metadata_store[idx]
            result = RetrievalResult(
                chunk_id=metadata["chunk_id"],
                content=content,
                score=float(final_score),
                source=metadata["source"],
                metadata=metadata,
            )
            results.append(result)

            if len(results) >= top_k:
                break

        elapsed = time.time() - start_time
        mode = "hybrid" if use_hybrid else "dense"
        logger.info(f"Search ({mode}) returned {len(results)} results in {elapsed * 1000:.1f}ms")

        return results

    def delete_by_document_id(self, document_id: str) -> int:
        """
        Remove all chunks belonging to a document (rebuild index without those chunks).

        Args:
            document_id: Document ID to remove.

        Returns:
            Number of chunks removed.
        """
        indices_to_keep = [
            i for i, m in enumerate(self._metadata_store)
            if m["document_id"] != document_id
        ]
        removed_count = len(self._metadata_store) - len(indices_to_keep)

        if removed_count == 0:
            return 0

        # Rebuild index
        faiss = self._get_faiss()
        new_index = faiss.IndexFlatIP(self.dimension)

        if indices_to_keep:
            vectors = np.array([
                self._index.reconstruct(i) for i in indices_to_keep
            ], dtype=np.float32)
            new_index.add(vectors)

        self._index = new_index
        self._metadata_store = [self._metadata_store[i] for i in indices_to_keep]
        self._chunk_contents = [self._chunk_contents[i] for i in indices_to_keep]

        self._save_index()
        logger.info(f"Removed {removed_count} chunks for document {document_id}")
        return removed_count

    def get_stats(self) -> dict:
        """Get vector store statistics."""
        return {
            "total_vectors": self._index.ntotal if self._index else 0,
            "dimension": self.dimension,
            "index_path": str(self.index_path),
            "total_documents": len(set(m["document_id"] for m in self._metadata_store)),
        }

    def clear(self) -> None:
        """Clear the entire vector store."""
        faiss = self._get_faiss()
        self._index = faiss.IndexFlatIP(self.dimension)
        self._metadata_store = []
        self._chunk_contents = []
        self._save_index()
        logger.info("Vector store cleared")

    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            self.index_path.mkdir(parents=True, exist_ok=True)

            faiss = self._get_faiss()
            faiss.write_index(self._index, str(self.index_path / "index.faiss"))

            metadata_path = self.index_path / "metadata.pkl"
            with open(metadata_path, "wb") as f:
                pickle.dump({
                    "metadata_store": self._metadata_store,
                    "chunk_contents": self._chunk_contents,
                }, f)

            logger.debug(f"Saved FAISS index to {self.index_path}")

        except Exception as e:
            raise VectorStoreError(
                f"Failed to save FAISS index: {e}",
                details={"index_path": str(self.index_path)},
            )

    def _load_index(self) -> None:
        """Load FAISS index and metadata from disk."""
        try:
            faiss = self._get_faiss()
            self._index = faiss.read_index(str(self.index_path / "index.faiss"))

            metadata_path = self.index_path / "metadata.pkl"
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                self._metadata_store = data["metadata_store"]
                self._chunk_contents = data["chunk_contents"]

            logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors")

        except Exception as e:
            raise VectorStoreError(
                f"Failed to load FAISS index: {e}",
                details={"index_path": str(self.index_path)},
            )

    def _index_exists(self) -> bool:
        """Check if a saved index exists."""
        return (self.index_path / "index.faiss").exists()

    def _rebuild_bm25(self) -> None:
        """Rebuild the BM25 index from current chunk contents."""
        if not self.settings.hybrid_search_enabled:
            return
        if not self._chunk_contents:
            self._bm25 = None
            return
        self._bm25 = _BM25()
        self._bm25.index(self._chunk_contents)
        logger.debug(f"BM25 index built over {len(self._chunk_contents)} chunks")

    def _get_faiss(self):
        """Import and return faiss module."""
        try:
            import faiss
            return faiss
        except ImportError:
            raise VectorStoreError("FAISS is required. Install with: pip install faiss-cpu")
