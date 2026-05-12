"""
FastAPI REST API for GenAI RAG System.
Production API server with health checks, query endpoint, metrics,
settings, vector-store stats, and chat export.

Run: uvicorn api.app:app --host 0.0.0.0 --port 8000
Or:  python -m api.app
"""

import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from pipeline.rag_pipeline import RAGPipeline
from schemas.pipeline import QueryRequest
from utils.logger import get_logger, set_correlation_id

logger = get_logger(__name__)

# ─── Global pipeline instance ────────────────────────────────────
_pipeline: Optional[RAGPipeline] = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle events."""
    logger.info("Starting FastAPI server — initializing RAG pipeline")
    _get_pipeline()
    yield
    logger.info("Shutting down FastAPI server")


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production RAG Pipeline REST API",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Correlation-ID middleware ────────────────────────────────────
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    cid = request.headers.get("X-Correlation-ID") or set_correlation_id()
    if not request.headers.get("X-Correlation-ID"):
        set_correlation_id(cid)
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = cid
    return response


# ─── Request / Response models ───────────────────────────────────
class QueryIn(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    include_sources: bool = True


class SourceOut(BaseModel):
    chunk_id: str
    content: str
    score: float
    source: str


class QueryOut(BaseModel):
    query: str
    answer: str
    sources: list[SourceOut]
    followup_questions: list[str]
    total_time_ms: float
    token_usage: dict
    correlation_id: str
    timestamp: str


class HealthOut(BaseModel):
    status: str
    version: str
    environment: str
    uptime_seconds: float
    vector_store: dict
    cache: Optional[dict]
    token_usage: dict
    embedding_cache: Optional[dict]


class SettingsOut(BaseModel):
    """Read-only system configuration — mirrors Streamlit Settings page."""

    provider: str
    model: str
    temperature: float
    max_tokens: int
    connection_pool: int
    request_timeout: int
    top_k: int
    similarity_threshold: float
    chunk_size: int
    chunk_overlap: int
    hybrid_search_enabled: bool
    hybrid_search_alpha: float
    reranking_enabled: bool
    response_cache_enabled: bool
    response_cache_max_size: int
    response_cache_ttl_seconds: int
    embedding_cache_enabled: bool
    embedding_cache_max_size: int
    followup_enabled: bool
    followup_count: int
    metadata_enrichment_enabled: bool


# ─── Start time ──────────────────────────────────────────────────
_start_time = time.time()


# ─── Endpoints ────────────────────────────────────────────────────
@app.get("/health", response_model=HealthOut, tags=["System"])
async def health_check():
    """Health check endpoint with system metrics."""
    pipeline = _get_pipeline()
    vs_stats = pipeline.vector_store.get_stats()
    cache_stats = pipeline.get_cache_stats()
    token_stats = pipeline.llm_service.token_usage.summary
    embed_cache = pipeline.vector_store.embedding_service.get_cache_stats()

    return HealthOut(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(time.time() - _start_time, 1),
        vector_store=vs_stats,
        cache=cache_stats,
        token_usage=token_stats,
        embedding_cache=embed_cache,
    )


@app.post("/query", response_model=QueryOut, tags=["RAG"])
async def query_endpoint(body: QueryIn):
    """Execute a RAG query and return the answer with sources."""
    pipeline = _get_pipeline()
    cid = set_correlation_id()

    try:
        request = QueryRequest(
            query=body.query,
            top_k=body.top_k,
            include_sources=body.include_sources,
        )
        response = await pipeline.aquery(request)

        sources = [
            SourceOut(
                chunk_id=s.chunk_id,
                content=s.content,
                score=s.score,
                source=s.source,
            )
            for s in response.sources
        ]

        return QueryOut(
            query=response.query,
            answer=response.answer,
            sources=sources,
            followup_questions=response.followup_questions,
            total_time_ms=response.total_time_ms,
            token_usage=response.token_usage,
            correlation_id=cid,
            timestamp=response.timestamp.isoformat(),
        )

    except Exception as e:
        logger.error(f"[{cid}] Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["System"])
async def metrics():
    """Return token usage and cache metrics."""
    pipeline = _get_pipeline()
    return {
        "token_usage": pipeline.llm_service.token_usage.summary,
        "response_cache": pipeline.get_cache_stats(),
        "embedding_cache": pipeline.vector_store.embedding_service.get_cache_stats(),
        "vector_store": pipeline.vector_store.get_stats(),
    }


@app.get("/settings", response_model=SettingsOut, tags=["System"])
async def get_system_settings():
    """Read-only system configuration — mirrors Streamlit Settings page."""
    s = get_settings()

    if s.use_openrouter and s.openrouter_api_key:
        provider, model = "OpenRouter", s.openrouter_model
    elif s.azure_openai_endpoint:
        provider, model = "Azure OpenAI", s.azure_openai_deployment or "N/A"
    else:
        provider, model = "OpenAI", s.openai_model

    return SettingsOut(
        provider=provider,
        model=model,
        temperature=s.openai_temperature,
        max_tokens=s.openai_max_tokens,
        connection_pool=s.llm_connection_pool_size,
        request_timeout=s.llm_request_timeout,
        top_k=s.similarity_top_k,
        similarity_threshold=s.similarity_threshold,
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        hybrid_search_enabled=s.hybrid_search_enabled,
        hybrid_search_alpha=s.hybrid_search_alpha,
        reranking_enabled=s.reranking_enabled,
        response_cache_enabled=s.response_cache_enabled,
        response_cache_max_size=s.response_cache_max_size,
        response_cache_ttl_seconds=s.response_cache_ttl_seconds,
        embedding_cache_enabled=s.embedding_cache_enabled,
        embedding_cache_max_size=s.embedding_cache_max_size,
        followup_enabled=s.followup_enabled,
        followup_count=s.followup_count,
        metadata_enrichment_enabled=s.metadata_enrichment_enabled,
    )


@app.get("/vector-store/stats", tags=["System"])
async def vector_store_stats():
    """Dedicated vector-store statistics — mirrors Streamlit Dashboard."""
    pipeline = _get_pipeline()
    return pipeline.vector_store.get_stats()


@app.post("/cache/clear", tags=["System"])
async def clear_cache():
    """Clear the response cache."""
    pipeline = _get_pipeline()
    pipeline.clear_cache()
    return {"status": "cache cleared"}


# ─── Direct execution ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
