"""
RAG Pipeline module for GenAI RAG System.
Orchestrates the full retrieval-augmented generation flow.
"""

import time
from pathlib import Path
from typing import Optional

from config.settings import get_settings
from pipeline.llm_service import LLMService
from schemas.pipeline import QueryRequest, QueryResponse, RAGContext, RetrievalResult
from utils.exceptions import PipelineError
from utils.logger import get_logger
from vectordb.faiss_store import FAISSVectorStore

logger = get_logger(__name__)


class RAGPipeline:
    """Production RAG pipeline: Query → Retrieve → Generate → Respond."""

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        llm_service: Optional[LLMService] = None,
        system_prompt: Optional[str] = None,
    ):
        self.settings = get_settings()
        self.vector_store = vector_store or FAISSVectorStore()
        self.llm_service = llm_service or LLMService()
        self.system_prompt = system_prompt or self._load_system_prompt()

    def query(self, request: QueryRequest) -> QueryResponse:
        """
        Execute the full RAG pipeline for a query.

        Args:
            request: QueryRequest with user's question and options.

        Returns:
            QueryResponse with answer, sources, and metadata.
        """
        start_time = time.time()

        try:
            # Step 1: Retrieve relevant documents
            rag_context = self._retrieve(request)

            # Step 2: Generate answer using LLM
            answer = self._generate(request.query, rag_context)

            # Step 3: Build response
            total_time_ms = (time.time() - start_time) * 1000
            response = QueryResponse(
                query=request.query,
                answer=answer,
                sources=rag_context.retrieved_chunks if request.include_sources else [],
                context_used=rag_context.combined_context,
                total_time_ms=total_time_ms,
                metadata={
                    "retrieval_time_ms": rag_context.retrieval_time_ms,
                    "generation_time_ms": total_time_ms - rag_context.retrieval_time_ms,
                    "chunks_retrieved": len(rag_context.retrieved_chunks),
                    "model": self.settings.openai_model,
                },
            )

            logger.info(
                f"RAG query completed in {total_time_ms:.1f}ms "
                f"(retrieval={rag_context.retrieval_time_ms:.1f}ms, "
                f"chunks={len(rag_context.retrieved_chunks)})"
            )
            return response

        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(
                f"RAG pipeline failed: {type(e).__name__}: {e}",
                details={"query": request.query},
            )

    def query_simple(self, question: str) -> str:
        """
        Simplified query interface - takes a string, returns a string.

        Args:
            question: User's question.

        Returns:
            Generated answer string.
        """
        request = QueryRequest(query=question)
        response = self.query(request)
        return response.answer

    def _retrieve(self, request: QueryRequest) -> RAGContext:
        """Retrieve relevant documents from vector store."""
        start_time = time.time()

        results = self.vector_store.search(
            query=request.query,
            top_k=request.top_k,
        )

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

    def _build_context(self, results: list[RetrievalResult]) -> str:
        """Build combined context string from retrieval results."""
        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Source {i}] (score: {result.score:.3f})\n{result.content}")

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