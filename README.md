# GenAI RAG System — Enterprise RAG Application with LLM Evaluation

> A complete, production-grade Retrieval-Augmented Generation (RAG) system with hybrid search, LLM-based reranking, async evaluation pipelines (DeepEval + RAGAS), FastAPI REST API, enterprise Streamlit UI, CI/CD automation, and zero hardcoded values.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Key Features](#key-features)
5. [Setup & Installation](#setup--installation)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [REST API](#rest-api)
9. [Streamlit UI](#streamlit-ui)
10. [LLM Evaluation Pipeline](#llm-evaluation-pipeline)
11. [CI/CD Pipeline](#cicd-pipeline)
12. [Testing](#testing)
13. [Production Deployment Checklist](#production-deployment-checklist)

---

## Overview

This system implements a **full enterprise-grade RAG pipeline** with:

- **Document Ingestion** → Load, enrich metadata, chunk, and embed documents (PDF, TXT, MD, DOCX, CSV)
- **Vector Storage** → FAISS with cosine similarity, BM25 hybrid search, embedding cache
- **RAG Pipeline** → Retrieve → Rerank → Generate → Follow-ups, with response caching and async support
- **REST API** → FastAPI with health checks, query, metrics, settings, and cache management
- **Enterprise UI** → Streamlit with chat, observability dashboard, and tabbed configuration
- **LLM Evaluation** → Async evaluation using DeepEval and RAGAS with batched concurrency
- **Dynamic Testset Generation** → RAGAS TestsetGenerator, chunk-based LLM, or hybrid strategies
- **CI/CD Integration** → GitHub Actions for linting, testing, evaluation, and threshold gating
- **Observability** → Structured logging with correlation IDs, token usage tracking, connection pooling
- **Zero Hardcoding** → All values from environment variables via Pydantic Settings

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        GenAI RAG System                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   UI Layer                                                               │
│   ┌──────────────────┐    ┌──────────────────┐    ┌────────────────┐    │
│   │  Streamlit UI    │    │  FastAPI REST API │    │  CLI REPL      │    │
│   │  (app_ui.py)     │    │  (api/app.py)    │    │  (main.py)     │    │
│   │  Chat/Dashboard/ │    │  /health /query   │    │  Interactive   │    │
│   │  Settings        │    │  /metrics /settings│   │  Query Mode    │    │
│   └────────┬─────────┘    └────────┬──────────┘   └───────┬────────┘    │
│            └──────────────────┬────┘──────────────────────┘             │
│                               ▼                                          │
│   Pipeline Layer                                                         │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │                    RAG Pipeline (rag_pipeline.py)                 │   │
│   │  Query → [Cache Check] → Retrieve → Rerank → Generate           │   │
│   │                                            → Follow-ups          │   │
│   │                                                                  │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │   │
│   │  │ LLM Service  │  │   Reranker   │  │  Response Cache      │   │   │
│   │  │ (httpx pool, │  │  (LLM-based  │  │  (LRU + TTL,         │   │   │
│   │  │  token track) │  │   scoring)   │  │   thread-safe)       │   │   │
│   │  └──────────────┘  └──────────────┘  └──────────────────────┘   │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                               ▼                                          │
│   Storage Layer                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │             FAISS Vector Store (faiss_store.py)                   │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │   │
│   │  │ Dense Vector  │  │  BM25 Sparse │  │  Embedding Cache     │   │   │
│   │  │ (IndexFlatIP) │  │  (built-in)  │  │  (SHA-256 keyed LRU) │   │   │
│   │  └──────────────┘  └──────────────┘  └──────────────────────┘   │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   Ingestion Layer                                                        │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐      │
│   │  Doc Loader  │→ │  Metadata    │→ │  Text Splitter           │      │
│   │  (PDF/MD/    │  │  Enricher    │  │  (section-aware,         │      │
│   │   DOCX/CSV)  │  │  (LLM-based) │  │   heading context)      │      │
│   └──────────────┘  └──────────────┘  └──────────────────────────┘      │
│                                                                          │
│   Evaluation Layer                                                       │
│   ┌───────────────────────────────────────────────────────────────────┐  │
│   │              Async Evaluation Pipeline                             │  │
│   │  ┌─────────────────┐         ┌─────────────────┐                 │  │
│   │  │    DeepEval      │         │      RAGAS       │   (concurrent) │  │
│   │  │  aevaluate_*()   │         │  aevaluate_*()   │                │  │
│   │  │  • AnswerRelevancy│        │  • Faithfulness   │                │  │
│   │  │  • Faithfulness   │        │  • AnswerRelevancy│                │  │
│   │  │  • ContextPrec.   │        │  • ContextPrec.   │                │  │
│   │  │  • Hallucination  │        │  • ContextRecall  │                │  │
│   │  │  • Bias/Toxicity  │        │                   │                │  │
│   │  │  • GEval custom   │        │                   │                │  │
│   │  └─────────────────┘         └─────────────────┘                 │  │
│   │                                                                   │  │
│   │  ┌─────────────────┐         ┌─────────────────┐                 │  │
│   │  │  Testset Gen.    │         │  Dataset Manager │                 │  │
│   │  │  (RAGAS/Doc/     │         │  (CRUD, merge,   │                 │  │
│   │  │   Hybrid)        │         │   quality check)  │                │  │
│   │  └─────────────────┘         └─────────────────┘                 │  │
│   └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│   ┌───────────────────────────────────────────────────────────────────┐  │
│   │              CI/CD Pipeline (GitHub Actions)                       │  │
│   │  Lint → Unit Tests → Integration Tests → LLM Evaluation → Report │  │
│   └───────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
RAG-CHATBOT-APPLICATION/
├── .env.example                    # All environment variables with comments
├── .streamlit/config.toml          # Streamlit theme (Indigo-Slate palette)
├── main.py                         # Interactive CLI REPL
├── app_ui.py                       # Enterprise Streamlit UI (chat/dashboard/settings)
├── pyproject.toml                  # Build config, pytest, ruff, mypy, coverage
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation
│
├── api/                            # ─── REST API ───
│   ├── __init__.py                 # Package init, re-exports app
│   └── app.py                      # FastAPI server (6 endpoints + middleware)
│
├── config/                         # ─── Configuration ───
│   ├── __init__.py
│   ├── settings.py                 # Pydantic Settings (~60 env-driven fields)
│   └── prompts/
│       └── system_prompt.txt       # RAG system prompt template
│
├── schemas/                        # ─── Data Models (Pydantic v2) ───
│   ├── __init__.py
│   ├── documents.py                # DocumentMetadata, Document, DocumentChunk
│   ├── evaluation.py               # EvaluationSample/Dataset/Result/Report
│   └── pipeline.py                 # QueryRequest/Response, RetrievalResult
│
├── utils/                          # ─── Shared Utilities ───
│   ├── __init__.py
│   ├── logger.py                   # Structured logging + correlation IDs
│   ├── exceptions.py               # RAGSystemError hierarchy (7 types)
│   ├── email_sender.py             # SMTP HTML evaluation reports
│   └── file_handler.py             # File I/O utilities
│
├── ingest/                         # ─── Document Ingestion ───
│   ├── __init__.py
│   ├── document_loader.py          # Multi-format loader (PDF/TXT/MD/DOCX/CSV)
│   ├── text_splitter.py            # Section-aware splitting with heading context
│   ├── metadata_enricher.py        # LLM-based auto-tagging (dept/topic/author)
│   └── ingestion_pipeline.py       # Orchestrator: load → enrich → split → store
│
├── vectordb/                       # ─── Vector Store ───
│   ├── __init__.py
│   ├── embedding_service.py        # Embedding generation with LRU cache
│   └── faiss_store.py              # FAISS + built-in BM25 hybrid search
│
├── pipeline/                       # ─── RAG Pipeline ───
│   ├── __init__.py
│   ├── llm_service.py              # LLM client (httpx pool, token tracking)
│   ├── rag_pipeline.py             # RAG orchestrator (cache, rerank, follow-ups)
│   └── reranker.py                 # LLM-based chunk reranker
│
├── evaluation/                     # ─── LLM Evaluation ───
│   ├── __init__.py
│   ├── dataset_manager.py          # Dataset CRUD + generation (3 strategies)
│   ├── deepeval_evaluator.py       # DeepEval: 11 metrics, sync + async
│   ├── ragas_evaluator.py          # RAGAS: 4 metrics, sync + async
│   ├── evaluation_pipeline.py      # Orchestrator: sync + async + concurrent
│   └── testset_generator.py        # RAGAS TestsetGenerator wrapper
│
├── scripts/                        # ─── Entry Points ───
│   ├── __init__.py
│   ├── ingest_documents.py         # Document ingestion CLI
│   ├── generate_eval_dataset.py    # Dynamic dataset generation
│   ├── run_evaluation.py           # Full evaluation pipeline runner
│   └── check_thresholds.py         # CI threshold gate
│
├── tests/                          # ─── Test Suite (37 tests) ───
│   ├── __init__.py
│   ├── conftest.py                 # Fixtures (env, mocks, sample data)
│   ├── test_ingestion.py           # Document loading + splitting tests
│   ├── test_vectordb.py            # FAISS store tests
│   ├── test_pipeline.py            # RAG pipeline tests
│   └── test_evaluation.py          # Evaluation framework tests
│
├── data/
│   ├── documents/                  # Source documents for RAG
│   ├── vector_store/faiss_index/   # Persisted FAISS index
│   └── evaluation/
│       └── eval_dataset.json       # Evaluation test cases
│
└── .github/workflows/
    └── checkandevaluation.yml      # CI/CD pipeline (4 jobs)
```

---

## Key Features

### Core RAG Pipeline

| Feature | Description |
|---------|-------------|
| **Hybrid Search** | BM25 keyword + FAISS dense vector fusion with configurable alpha weight |
| **LLM-Based Reranking** | Scores each retrieved chunk for query relevance, drops noise, re-orders by score |
| **Response Caching** | In-memory LRU cache with TTL — avoids duplicate LLM calls for repeated queries |
| **Embedding Cache** | SHA-256 keyed LRU cache — avoids re-embedding identical text across ingestion runs |
| **Follow-up Questions** | Auto-generates suggested follow-up questions after each answer |
| **Async Execution** | `aquery()` async wrapper for non-blocking pipeline execution |
| **Correlation IDs** | End-to-end request tracing through all pipeline stages via `contextvars` |
| **Token Tracking** | Thread-safe cumulative token usage tracking (prompt/completion/total) |
| **Connection Pooling** | `httpx` connection pool for LLM HTTP clients (configurable size + timeout) |

### Document Ingestion

| Feature | Description |
|---------|-------------|
| **Multi-Format** | PDF (PyPDF2), TXT, Markdown, DOCX (python-docx), CSV |
| **Section-Aware Splitting** | Preserves markdown heading hierarchy as chunk context |
| **Metadata Enrichment** | LLM-based auto-tagging: department, topic, author, date, language, document_type |

### LLM Evaluation

| Feature | Description |
|---------|-------------|
| **Dual Framework** | Both DeepEval (11 metrics) and RAGAS (4 metrics) |
| **Async Evaluation** | `aevaluate_sample()` runs metrics concurrently per sample; `aevaluate_dataset()` runs samples in parallel batches |
| **Batched Concurrency** | `EVAL_BATCH_SIZE` controls max concurrent samples via `asyncio.Semaphore` |
| **Dynamic Testset Generation** | 3 strategies: RAGAS TestsetGenerator, chunk-based LLM, hybrid (merged + deduplicated) |
| **Custom GEval Metrics** | Coherence, Completeness, Conciseness via DeepEval GEval with custom criteria |
| **Credit-Safe** | Detects OpenRouter 402 errors and aborts gracefully, preserving partial results |

### API & UI

| Feature | Description |
|---------|-------------|
| **FastAPI REST API** | 6 endpoints with OpenAPI docs, correlation-ID middleware, CORS |
| **Enterprise Streamlit UI** | Chat (source cards, follow-up pills, export), Dashboard (KPI cards, token/latency charts), Settings (tabbed config) |
| **CLI REPL** | Interactive query mode with `stats` command |
| **Email Reports** | HTML evaluation reports with per-metric tables, sent via SMTP |

### Infrastructure

| Feature | Description |
|---------|-------------|
| **Zero Hardcoding** | All config via `.env` + Pydantic Settings (~60 fields) |
| **Structured Logging** | JSON file handler + correlation IDs for distributed tracing |
| **Custom Exceptions** | Typed hierarchy: `RAGSystemError` → 7 specific error types |
| **CI/CD** | GitHub Actions: lint → test → integration → LLM evaluation → threshold gate |

---

## Setup & Installation

### 1. Clone & Create Virtual Environment

```bash
cd RAG-CHATBOT-APPLICATION
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

**Option A: OpenAI Direct**

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

**Option B: OpenRouter (recommended — 200+ models, single API key)**

```env
USE_OPENROUTER=true
OPENROUTER_API_KEY=sk-or-your-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

**Option C: Azure OpenAI (enterprise)**

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

> Provider priority: **OpenRouter** → **Azure OpenAI** → **OpenAI**

### 4. Ingest Documents

```bash
# Place documents in data/documents/, then:
python -m scripts.ingest_documents --directory data/documents
```

### 5. Run

```bash
# Interactive CLI
python main.py

# Streamlit UI
streamlit run app_ui.py

# FastAPI server
uvicorn api.app:app --host 0.0.0.0 --port 8000
# or: python -m api.app
```

---

## Configuration

All configuration is centralized in `config/settings.py` via Pydantic Settings. See `.env.example` for the complete list with comments.

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model for generation |
| `OPENAI_TEMPERATURE` | `0.0` | LLM temperature |
| `OPENAI_MAX_TOKENS` | `256` | Max tokens per response |
| `SIMILARITY_TOP_K` | `8` | Documents to retrieve |
| `SIMILARITY_THRESHOLD` | `0.25` | Minimum similarity score |
| `CHUNK_SIZE` | `500` | Document chunk size (chars) |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |

### Enterprise Features

| Variable | Default | Description |
|----------|---------|-------------|
| `HYBRID_SEARCH_ENABLED` | `false` | BM25 + dense vector hybrid search |
| `HYBRID_SEARCH_ALPHA` | `0.5` | Weight: 1.0=dense, 0.0=BM25 |
| `RERANKING_ENABLED` | `true` | LLM-based reranking post-retrieval |
| `RERANKER_TOP_N` | `4` | Chunks to keep after reranking |
| `RERANKER_RELEVANCE_THRESHOLD` | `0.5` | Min reranker score |
| `RESPONSE_CACHE_ENABLED` | `false` | In-memory response caching |
| `RESPONSE_CACHE_TTL_SECONDS` | `3600` | Cache TTL |
| `EMBEDDING_CACHE_ENABLED` | `true` | Embedding dedup cache |
| `FOLLOWUP_ENABLED` | `true` | Follow-up question generation |
| `FOLLOWUP_COUNT` | `3` | Number of follow-ups |
| `METADATA_ENRICHMENT_ENABLED` | `false` | LLM-based metadata auto-tagging |

### Evaluation

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_FRAMEWORKS` | `deepeval,ragas` | Frameworks to run |
| `EVAL_THRESHOLD` | `0.7` | Minimum pass score |
| `EVAL_BATCH_SIZE` | `10` | Concurrent samples in async eval |
| `EVAL_CUSTOM_METRICS_ENABLED` | `false` | Bias, toxicity, coherence, completeness, conciseness |
| `EVAL_AUTO_GENERATE_DATASET` | `false` | Auto-generate eval dataset before evaluation |
| `TESTSET_STRATEGY` | `ragas` | Generation strategy: ragas, document, hybrid |
| `TESTSET_SIZE` | `20` | Number of test cases to generate |

---

## Usage

### Three UI Surfaces

| Surface | Command | URL |
|---------|---------|-----|
| **Streamlit UI** | `streamlit run app_ui.py` | `http://localhost:8501` |
| **FastAPI REST API** | `uvicorn api.app:app --host 0.0.0.0 --port 8000` | `http://localhost:8000/docs` |
| **CLI REPL** | `python main.py` | Terminal |

### Run Evaluation

```bash
# Full evaluation pipeline (both DeepEval + RAGAS)
python -m scripts.run_evaluation

# Generate dynamic test dataset
python -m scripts.generate_eval_dataset

# Check thresholds (CI gate)
python -m scripts.check_thresholds
```

---

## REST API

FastAPI server at `api/app.py` with Swagger docs at `/docs`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | System health: uptime, vector store stats, token usage, cache stats |
| `POST` | `/query` | Execute RAG query — returns answer, sources, follow-ups, token usage |
| `GET` | `/metrics` | Token usage, response cache, embedding cache, vector store metrics |
| `GET` | `/settings` | Read-only system configuration (mirrors Streamlit Settings page) |
| `GET` | `/vector-store/stats` | Vector store statistics (total vectors, documents, dimension) |
| `POST` | `/cache/clear` | Clear the response cache |

### Middleware

- **Correlation-ID** — reads `X-Correlation-ID` header or generates one; returns it in response
- **CORS** — configurable allowed origins

### Running

```bash
# Development
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Or directly
python -m api.app
```

### Example Requests

```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "top_k": 5}'

# System settings
curl http://localhost:8000/settings

# Vector store stats
curl http://localhost:8000/vector-store/stats

# Clear cache
curl -X POST http://localhost:8000/cache/clear
```

---

## Streamlit UI

Enterprise-grade multi-page application at `app_ui.py`.

### Pages

| Page | Features |
|------|----------|
| **💬 Chat** | Chat interface with follow-up pills, source cards (color-coded score badges), expandable content preview, JSON/Markdown export |
| **📊 Dashboard** | KPI cards (tokens, requests), token-per-query bar chart, latency area chart, vector store + cache stats |
| **⚙️ Settings** | Tabbed view: LLM, Retrieval, Features, Evaluation — all read-only from `.env` |

### Design System

- Indigo-Slate color palette with hover transitions
- Responsive CSS grid KPI cards (collapses on mobile)
- Source cards with color-coded score badges (green ≥0.7, amber ≥0.4, red <0.4)
- Follow-up question pills
- XSS protection via `html.escape()`
- Live connection status indicator in sidebar
- Session metrics (query count, avg latency)

```bash
streamlit run app_ui.py
# Opens at http://localhost:8501
```

---

## LLM Evaluation Pipeline

### Sync Execution

```bash
# Full evaluation (both frameworks)
python -m scripts.run_evaluation

# Generate dynamic test dataset
python -m scripts.generate_eval_dataset

# Check thresholds (CI gate)
python -m scripts.check_thresholds
```

### Async Execution (Python)

```python
import asyncio
from evaluation.evaluation_pipeline import EvaluationPipeline
from pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
eval_pipeline = EvaluationPipeline(rag_pipeline=pipeline)

# Run both frameworks concurrently with batched sample evaluation
reports = asyncio.run(eval_pipeline.arun_full_evaluation())

# Or single framework
report = asyncio.run(eval_pipeline.arun_deepeval_evaluation())
report = asyncio.run(eval_pipeline.arun_ragas_evaluation())
```

### DeepEval Metrics (11 total)

| Metric | Type | What It Measures |
|--------|------|-----------------|
| `answer_relevancy` | Built-in | Answer relevance to question |
| `faithfulness` | Built-in | Grounded in retrieved context |
| `contextual_precision` | Built-in | Signal-to-noise in retrieved docs |
| `contextual_recall` | Built-in | Coverage of ground truth |
| `contextual_relevancy` | Built-in | Relevance of retrieved context |
| `hallucination` | Built-in | Fabricated information detection |
| `bias` | Built-in | Prejudiced or biased content |
| `toxicity` | Built-in | Harmful or offensive content |
| `coherence` | GEval | Logical structure and flow |
| `completeness` | GEval | Thoroughness of answer |
| `conciseness` | GEval | Avoids unnecessary verbosity |

> Custom metrics (bias, toxicity, coherence, completeness, conciseness) enabled via `EVAL_CUSTOM_METRICS_ENABLED=true`.

### RAGAS Metrics (4 total)

| Metric | What It Measures |
|--------|-----------------|
| `faithfulness` | Factual consistency with context |
| `answer_relevancy` | Relevance of answer to question |
| `context_precision` | Precision of retrieved context |
| `context_recall` | Recall of ground truth by context |

### Dynamic Testset Generation

| Strategy | How It Works |
|----------|-------------|
| `ragas` | RAGAS TestsetGenerator — multi-complexity (simple/reasoning/multi-context) |
| `document` | Chunk-based LLM generation from ingested documents |
| `hybrid` | Both merged + deduplicated via SequenceMatcher |

### Evaluation Dataset Format

```json
{
  "dataset_id": "unique_id",
  "name": "My Evaluation Set",
  "samples": [
    {
      "sample_id": "s1",
      "user_input": "What is RAG?",
      "expected_output": "RAG combines retrieval with generation...",
      "context": ["Ground truth context..."],
      "retrieval_context": null
    }
  ]
}
```

---

## CI/CD Pipeline

### Workflow: `checkandevaluation.yml`

```
Push/PR → Code Quality → Unit Tests → Integration Tests → LLM Evaluation → Threshold Check
```

| Job | Trigger | What It Does |
|-----|---------|-------------|
| **Code Quality** | Every push/PR | Ruff linting + formatting + mypy type checking |
| **Unit Tests** | Every push/PR | pytest with coverage (70% minimum) |
| **Integration Tests** | main branch + manual | Real API tests with `@pytest.mark.integration` |
| **LLM Evaluation** | main branch + manual | Full DeepEval + RAGAS eval → threshold gate → email report |

### Required GitHub Secrets

| Secret | Required | Purpose |
|--------|----------|---------|
| `OPENROUTER_API_KEY` | Yes (if OpenRouter) | OpenRouter API authentication |
| `OPENAI_API_KEY` | Yes (if direct OpenAI) | OpenAI API authentication |
| `EMAIL_SENDER` | If email enabled | SMTP sender email |
| `EMAIL_SENDER_PASSWORD` | If email enabled | SMTP app password |
| `EMAIL_RECIPIENTS` | If email enabled | Comma-separated recipients |

### Optional GitHub Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `USE_OPENROUTER` | `false` | Route calls through OpenRouter |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | Model for RAG + eval |
| `EVAL_MODEL` | `gpt-4o-mini` | LLM-as-a-Judge model |
| `EVAL_THRESHOLD` | `0.7` | Minimum pass score |
| `EVAL_CUSTOM_METRICS_ENABLED` | `false` | Enable custom production metrics |
| `DEEPEVAL_METRICS` | `answer_relevancy,...` | DeepEval metrics to run |
| `RAGAS_METRICS` | `faithfulness,...` | RAGAS metrics to run |
| `EMAIL_ENABLED` | `true` | Send email after evaluation |
| `EMAIL_SMTP_HOST` | `smtp.gmail.com` | SMTP server |
| `EMAIL_SMTP_PORT` | `587` | SMTP port |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=. --cov-report=term-missing

# Specific file
pytest tests/test_pipeline.py -v

# Exclude integration tests
pytest tests/ -v -k "not integration"
```

### Test Coverage

| Module | Tests |
|--------|-------|
| `test_ingestion.py` | Document loading (txt/md/pdf/unsupported/empty), text splitting (metadata/overlap/multi-section), ingestion pipeline |
| `test_vectordb.py` | FAISS add/search/empty/stats/clear/delete_by_document_id |
| `test_pipeline.py` | RAG response structure, query_simple, empty results, metadata, top_k passthrough |
| `test_evaluation.py` | Dataset load/save/create/validate, DeepEval sample/dataset/no-output, RAGAS sample/dataset/summary |

---

## LLM Provider Configuration

### Supported Providers

| Provider | Key Variable | Notes |
|----------|-------------|-------|
| **OpenAI** | `OPENAI_API_KEY` | Default provider |
| **OpenRouter** | `OPENROUTER_API_KEY` + `USE_OPENROUTER=true` | 200+ models via single API |
| **Azure OpenAI** | `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY` | Enterprise managed |

### OpenRouter Embedding Models

| Model | Dimension | Notes |
|-------|-----------|-------|
| `openai/text-embedding-3-small` | 1536 | Default, fast |
| `openai/text-embedding-3-large` | 3072 | Higher quality — set `EMBEDDING_DIMENSION=3072` |

---

## Email Notifications

Auto-sends HTML evaluation reports after each pipeline run.

```env
EMAIL_ENABLED=true
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_SENDER=your-email@gmail.com
EMAIL_SENDER_PASSWORD=your-app-password
EMAIL_RECIPIENTS=recipient1@example.com,recipient2@example.com
```

**Report includes:** color-coded pass/fail tables, per-framework metric summaries, per-sample breakdowns, JSON attachment.

> **Gmail users**: Use an [App Password](https://support.google.com/accounts/answer/185833) instead of your regular password.

---

## Production Deployment Checklist

- [ ] Set `ENVIRONMENT=production` in `.env`
- [ ] Configure LLM API key (`OPENAI_API_KEY` or `OPENROUTER_API_KEY`)
- [ ] Place source documents in `data/documents/`
- [ ] Run ingestion: `python -m scripts.ingest_documents --directory data/documents`
- [ ] Verify vector store: `data/vector_store/faiss_index/` populated
- [ ] Enable enterprise features as needed:
  - [ ] `RERANKING_ENABLED=true` (reduces retrieval noise)
  - [ ] `HYBRID_SEARCH_ENABLED=true` (better recall)
  - [ ] `RESPONSE_CACHE_ENABLED=true` (avoids duplicate LLM calls)
  - [ ] `METADATA_ENRICHMENT_ENABLED=true` (auto-tag documents)
- [ ] Configure evaluation: set `EVAL_THRESHOLD`, `EVAL_FRAMEWORKS`, `EVAL_CUSTOM_METRICS_ENABLED`
- [ ] Configure email: `EMAIL_ENABLED=true`, `EMAIL_SENDER`, `EMAIL_SENDER_PASSWORD`, `EMAIL_RECIPIENTS`
- [ ] Run evaluation: `python -m scripts.run_evaluation`
- [ ] Set up GitHub Secrets and Variables for CI/CD
- [ ] Enable GitHub Actions workflow
- [ ] Start API: `uvicorn api.app:app --host 0.0.0.0 --port 8000`
- [ ] Start UI: `streamlit run app_ui.py`

---

## Design Principles

1. **Zero Hardcoded Values** — Every configurable value from environment variables via Pydantic Settings
2. **Separation of Concerns** — Each module has a single responsibility
3. **Async-First Evaluation** — Concurrent metric + sample evaluation for faster pipeline runs
4. **Observable** — Correlation IDs, structured logging, token tracking at every stage
5. **Resilient** — Credit exhaustion detection, graceful degradation, partial result preservation
6. **Testable** — Every component testable in isolation with mocks
7. **CI/CD Native** — Threshold gates, artifact generation, email reporting built-in
