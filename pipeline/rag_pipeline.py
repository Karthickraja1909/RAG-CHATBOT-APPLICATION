"""
RAG Pipeline module for GenAI RAG System.
Orchestrates the full retrieval-augmented generation flow.
Features: response caching, follow-up question generation, async execution,
correlation ID tracing, optional LLM-based reranking.
"""

import asyncio
import hashlib
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from config.settings import get_settings
from pipeline.llm_service import LLMService
from schemas.pipeline import QueryRequest, QueryResponse, RAGContext, RetrievalResult
from utils.exceptions import PipelineError
from utils.logger import get_logger, set_correlation_id, get_correlation_id

logger = get_logger(__name__)


# ---- In-memory LRU response cache with TTL ----
class _ResponseCache:
    """Thread-safe LRU response cache with TTL expiry."""

    def __init__(self, max_size: int, ttl_seconds: int):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, QueryResponse]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(query: str, top_k: Optional[int]) -> str:
        raw = f"{query.strip().lower()}|{top_k or 'default'}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query: str, top_k: Optional[int]) -> Optional[QueryResponse]:
        key = self._key(query, top_k)
        with self._lock:
            if key in self._cache:
                ts, resp = self._cache[key]
                if time.time() - ts < self._ttl:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return resp
                else:
                    del self._cache[key]
            self._misses += 1
            return None

    def put(self, query: str, top_k: Optional[int], response: QueryResponse) -> None:
        key = self._key(query, top_k)
        with self._lock:
            self._cache[key] = (time.time(), response)
            self._cache.move_to_end(key)
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


class RAGPipeline:
    """Production RAG pipeline: Query -> Retrieve -> Rerank (optional) -> Generate -> Respond."""

    def __init__(
        self,
        vector_store=None,
        llm_service: Optional[LLMService] = None,
        system_prompt: Optional[str] = None,
        reranker=None,
    ):
        from vectordb.faiss_store import FAISSVectorStore

        self.settings = get_settings()
        self.vector_store = vector_store or FAISSVectorStore()
        self.llm_service = llm_service or LLMService()
        self.system_prompt = system_prompt or self._load_system_prompt()

        # Initialize reranker if enabled
        self.reranker = reranker
        if self.reranker is None and self.settings.reranking_enabled:
            from pipeline.reranker import Reranker
            self.reranker = Reranker()
            logger.info(
                f"Reranker enabled (model={self.settings.reranker_model}, "
                f"top_n={self.settings.reranker_top_n}, "
                f"threshold={self.settings.reranker_relevance_threshold})"
            )

        # Response cache
        self._cache: Optional[_ResponseCache] = None
        if self.settings.response_cache_enabled:
            self._cache = _ResponseCache(
                max_size=self.settings.response_cache_max_size,
                ttl_seconds=self.settings.response_cache_ttl_seconds,
            )
            logger.info(
                f"Response cache enabled (max_size={self.settings.response_cache_max_size}, "
                f"ttl={self.settings.response_cache_ttl_seconds}s)"
            )

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def query(self, request: QueryRequest) -> QueryResponse:
        """
        Execute the full RAG pipeline for a query (synchronous).
        Sets a correlation ID for the entire request.
        """
        cid = set_correlation_id()
        start_time = time.time()

        try:
            # Check cache first
            if self._cache:
                cached = self._cache.get(request.query, request.top_k)
                if cached is not None:
                    logger.info(f"[{cid}] Cache hit for query")
                    return cached

            # Step 1: Retrieve relevant documents
            rag_context = self._retrieve(request)

            # Step 2: Generate answer using LLM
            answer = self._generate(request.query, rag_context)

            # Step 3: Generate follow-up questions
            followups: list[str] = []
            if self.settings.followup_enabled:
                followups = self._generate_followups(request.query, answer)

            # Step 4: Build response
            total_time_ms = (time.time() - start_time) * 1000
            token_summary = self.llm_service.token_usage.summary

            response = QueryResponse(
                query=request.query,
                answer=answer,
                sources=rag_context.retrieved_chunks if request.include_sources else [],
                context_used=rag_context.combined_context,
                total_time_ms=total_time_ms,
                followup_questions=followups,
                token_usage=token_summary,
                metadata={
                    "correlation_id": cid,
                    "retrieval_time_ms": rag_context.retrieval_time_ms,
                    "generation_time_ms": total_time_ms - rag_context.retrieval_time_ms,
                    "chunks_retrieved": len(rag_context.retrieved_chunks),
                    "model": self.llm_service.model,
                    "reranking_enabled": self.reranker is not None,
                    "cache_hit": False,
                },
            )

            # Store in cache
            if self._cache:
                self._cache.put(request.query, request.top_k, response)

            logger.info(
                f"[{cid}] RAG query completed in {total_time_ms:.1f}ms "
                f"(retrieval={rag_context.retrieval_time_ms:.1f}ms, "
                f"chunks={len(rag_context.retrieved_chunks)})"
            )
            return response

        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(
                f"RAG pipeline failed: {type(e).__name__}: {e}",
                details={"query": request.query, "correlation_id": cid},
            )

    async def aquery(self, request: QueryRequest) -> QueryResponse:
        """
        Async version of query — runs the blocking pipeline in an executor
        so it can be used in async frameworks (FastAPI, etc.) without blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.query, request)

    def query_simple(self, question: str) -> str:
        """Simplified query interface - takes a string, returns a string."""
        request = QueryRequest(query=question)
        response = self.query(request)
        return response.answer

    def get_cache_stats(self) -> Optional[dict]:
        """Return response cache stats or None if caching is disabled."""
        return self._cache.stats if self._cache else None

    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self._cache:
            self._cache.clear()
            logger.info("Response cache cleared")

    # ----------------------------------------------------------------
    # Internal pipeline steps
    # ----------------------------------------------------------------

    def _retrieve(self, request: QueryRequest) -> RAGContext:
        """Retrieve relevant documents from vector store, with optional reranking."""
        start_time = time.time()

        results = self.vector_store.search(
            query=request.query,
            top_k=request.top_k,
        )

        # Apply reranking if enabled
        if self.reranker and results:
            results = self.reranker.rerank(query=request.query, results=results)

        combined_context = self._build_context(results)
        retrieval_time_ms = (time.time() - start_time) * 1000

        return RAGContext(
            query=request.query,
            retrieved_chunks=results,
            combined_context=combined_context,
            retrieval_time_ms=retrieval_time_ms,
        )

    def _generate(self, query: str, context: RAGContext) -> str:
        """Generate answer using LLM with retrieved context."""
        if not context.combined_context:
            return "I don't have enough information to answer this question based on the available documents."

        return self.llm_service.generate_with_context(
            query=query,
            context=context.combined_context,
            system_prompt_template=self.system_prompt,
        )

    def _generate_followups(self, query: str, answer: str) -> list[str]:
        """Generate suggested follow-up questions using the LLM."""
        count = self.settings.followup_count
        prompt = (
            f"Based on this question and answer, suggest {count} concise follow-up questions "
            f"the user might ask next. Return ONLY the questions, one per line, no numbering.\n\n"
            f"Question: {query}\nAnswer: {answer[:500]}"
        )
        try:
            raw = self.llm_service.generate(prompt, system_message="You generate follow-up questions.")
            lines = [line.strip().lstrip("0123456789.-) ") for line in raw.strip().splitlines() if line.strip()]
            return lines[:count]
        except Exception:
            logger.debug("Follow-up generation failed, skipping")
            return []

    def _build_context(self, results: list[RetrievalResult]) -> str:
        """Build combined context string from retrieval results."""
        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[{i}] {result.content}")

        return "\n\n---\n\n".join(context_parts)

    def _load_system_prompt(self) -> str:
        """Load system prompt from configured file."""
        prompt_path = self.settings.get_absolute_path(self.settings.system_prompt_path)

        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")

        # Fallback default prompt
        return (
            "You are a helpful AI assistant. Answer the question based on the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
