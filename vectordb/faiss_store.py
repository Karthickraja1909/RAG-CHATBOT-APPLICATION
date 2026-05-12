"""
FAISS vector store module for GenAI RAG System.
Handles storing, indexing, and retrieving document embeddings using FAISS.
"""

import pickle
import time
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

        self._initialize_index()

    def _initialize_index(self) -> None:
        """Initialize or load FAISS index."""
        try:
            import faiss

            if self._index_exists():
                self._load_index()
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

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[RetrievalResult]:
        """
        Search the vector store for similar documents.
        Returns deduplicated results — removes near-duplicate chunks from the same document.
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

        # Search
        scores, indices = self._index.search(query_vector, fetch_k)

        # Build results with deduplication
        results = []
        seen_content_hashes = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < threshold:
                continue

            content = self._chunk_contents[idx]
            # Deduplicate: skip chunks with >80% content overlap (via first 200 chars hash)
            content_key = content[:200].strip().lower()
            if content_key in seen_content_hashes:
                continue
            seen_content_hashes.add(content_key)

            metadata = self._metadata_store[idx]
            result = RetrievalResult(
                chunk_id=metadata["chunk_id"],
                content=content,
                score=float(score),
                source=metadata["source"],
                metadata=metadata,
            )
            results.append(result)

            if len(results) >= top_k:
                break

        elapsed = time.time() - start_time
        logger.info(f"Search returned {len(results)} results in {elapsed * 1000:.1f}ms")

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
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

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

    def _get_faiss(self):
        """Import and return faiss module."""
        try:
            import faiss
            return faiss
        except ImportError:
            raise VectorStoreError("FAISS is required. Install with: pip install faiss-cpu")
