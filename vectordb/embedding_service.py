"""
Embedding service module for GenAI RAG System.
Handles generating vector embeddings using OpenAI or Azure OpenAI.
Supports in-memory embedding cache to avoid re-embedding identical text.
"""

import hashlib
import time
from collections import OrderedDict
from typing import Optional

import openai

from config.settings import get_settings
from utils.exceptions import LLMError
from utils.logger import get_logger

logger = get_logger(__name__)


class _EmbeddingCache:
    """LRU in-memory cache for embeddings keyed by text hash."""

    def __init__(self, max_size: int):
        self._max_size = max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[list[float]]:
        key = self._key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, text: str, embedding: list[float]) -> None:
        key = self._key(text)
        self._cache[key] = embedding
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }


class EmbeddingService:
    """Service for generating text embeddings with optional caching."""

    def __init__(self, model: Optional[str] = None):
        self.settings = get_settings()
        self.model = model or self.settings.openai_embedding_model

        # Embedding cache
        if self.settings.embedding_cache_enabled:
            self._cache = _EmbeddingCache(max_size=self.settings.embedding_cache_max_size)
        else:
            self._cache = None

        # Provider selection: OpenRouter > Azure > OpenAI
        if self.settings.use_openrouter and self.settings.openrouter_api_key:
            self.client = openai.OpenAI(
                api_key=self.settings.openrouter_api_key,
                base_url=self.settings.openrouter_base_url,
                timeout=60.0,
            )
            # Use OpenRouter-specific embedding model if no explicit model passed
            if not model:
                self.model = self.settings.openrouter_embedding_model
            # Ensure model has provider prefix for OpenRouter (e.g. openai/text-embedding-3-small)
            if "/" not in self.model:
                self.model = f"openai/{self.model}"
            logger.info(f"Embedding Service using OpenRouter: {self.model}")
        elif self.settings.azure_openai_endpoint:
            self.client = openai.AzureOpenAI(
                api_key=self.settings.azure_openai_api_key,
                api_version=self.settings.azure_openai_api_version,
                azure_endpoint=self.settings.azure_openai_endpoint,
                timeout=60.0,
            )
            logger.info(f"Embedding Service using Azure OpenAI: {self.model}")
        else:
            self.client = openai.OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_api_base,
                timeout=60.0,
            )
            logger.info(f"Embedding Service using OpenAI: {self.model}")

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text (cache-aware).

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        if not text.strip():
            raise LLMError("Cannot embed empty text")

        # Check cache first
        if self._cache:
            cached = self._cache.get(text)
            if cached is not None:
                return cached

        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model,
            )
            embedding = response.data[0].embedding

            if self._cache:
                self._cache.put(text, embedding)

            return embedding

        except openai.APIError as e:
            raise LLMError(
                f"OpenAI API error during embedding: {e}",
                details={"model": self.model, "text_length": len(text)},
            )

    def embed_texts(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """
        Generate embeddings for multiple texts with batching and per-text caching.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per API call.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        # Resolve cached embeddings first, collect indices that need API calls
        results: list[Optional[list[float]]] = [None] * len(texts)
        to_embed: list[tuple[int, str]] = []

        for idx, text in enumerate(texts):
            if self._cache:
                cached = self._cache.get(text)
                if cached is not None:
                    results[idx] = cached
                    continue
            to_embed.append((idx, text))

        if self._cache and (len(texts) - len(to_embed)) > 0:
            logger.debug(f"Embedding cache: {len(texts) - len(to_embed)} hits, {len(to_embed)} misses")

        # Batch-embed uncached texts
        if to_embed:
            uncached_texts = [t for _, t in to_embed]
            total_batches = (len(uncached_texts) + batch_size - 1) // batch_size
            start_time = time.time()
            flat_embeddings: list[list[float]] = []

            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i : i + batch_size]
                batch_num = (i // batch_size) + 1

                try:
                    valid_texts = [t if t.strip() else " " for t in batch]
                    response = self.client.embeddings.create(
                        input=valid_texts,
                        model=self.model,
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    flat_embeddings.extend(batch_embeddings)
                    logger.debug(f"Embedded batch {batch_num}/{total_batches} ({len(batch)} texts)")

                except openai.APIError as e:
                    raise LLMError(
                        f"OpenAI API error in batch {batch_num}: {e}",
                        details={"batch_num": batch_num, "batch_size": len(batch)},
                    )

            elapsed = time.time() - start_time
            logger.info(f"Embedded {len(uncached_texts)} texts in {elapsed:.2f}s ({total_batches} batches)")

            # Map back and cache
            for embed_idx, (orig_idx, orig_text) in enumerate(to_embed):
                emb = flat_embeddings[embed_idx]
                results[orig_idx] = emb
                if self._cache:
                    self._cache.put(orig_text, emb)

        return results  # type: ignore[return-value]

    def get_cache_stats(self) -> Optional[dict]:
        """Return embedding cache statistics or None if caching is disabled."""
        return self._cache.stats if self._cache else None
