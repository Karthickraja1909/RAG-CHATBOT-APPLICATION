# GenAI RAG System — Production-Level RAG Application with LLM Evaluation

> A complete, production-ready Retrieval-Augmented Generation (RAG) system with integrated LLM evaluation pipelines using **DeepEval** and **RAGAS** frameworks, CI/CD automation, and zero hardcoded values.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Key Features](#key-features)
5. [Setup & Installation](#setup--installation)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [LLM Evaluation Pipeline](#llm-evaluation-pipeline)
9. [CI/CD Pipeline](#cicd-pipeline)
10. [Testing](#testing)
11. [Production Deployment Checklist](#production-deployment-checklist)

---

## Overview

This system implements a **full production-grade RAG pipeline** with:

- **Document Ingestion** → Load, chunk, and embed documents (PDF, TXT, MD, DOCX, CSV)
- **Vector Storage** → FAISS-based similarity search with cosine similarity
- **RAG Pipeline** → Retrieve relevant context → Generate LLM answers
- **LLM Evaluation** → Automated quality assessment using DeepEval and RAGAS
- **CI/CD Integration** → GitHub Actions for automated testing and evaluation
- **Zero Hardcoding** → All values from environment variables via `.env` config

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GenAI RAG System                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│   │   Document   │─── │    Vector    │─── │   RAG Pipeline   │  │
│   │  Ingestion   │    │  Store(FAISS)│    │  (Query→Answer)  │  │
│   └──────────────┘    └──────────────┘    └──────────────────┘  │
│         │                    │                       │            │
│         ▼                    ▼                       ▼            │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│   │   Chunking   │    │  Embeddings  │    │   LLM Service    │  │
│   │  (Recursive) │    │  (OpenAI)    │    │  (GPT-4o-mini)   │  │
│   └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                   │
│   ┌───────────────────────────────────────────────────────────┐  │
│   │              LLM Evaluation Pipeline                       │  │
│   │                                                           │  │
│   │   ┌─────────────┐         ┌─────────────┐               │  │
│   │   │  DeepEval   │         │    RAGAS    │               │  │
│   │   │             │         │             │               │  │
│   │   │ • Answer    │         │ • Faith-    │               │  │
│   │   │   Relevancy │         │   fulness   │               │  │
│   │   │ • Faith-    │         │ • Context   │               │  │
│   │   │   fulness   │         │   Precision │               │  │
│   │   │ • Context   │         │ • Context   │               │  │
│   │   │   Precision │         │   Recall    │               │  │
│   │   │ • Halluc.   │         │ • Answer    │               │  │
│   │   │   Detection │         │   Relevancy │               │  │
│   │   └─────────────┘         └─────────────┘               │  │
│   └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│   ┌───────────────────────────────────────────────────────────┐  │
│   │              CI/CD Pipeline (GitHub Actions)               │  │
│   │  Lint → Test → Evaluate → Threshold Check → Deploy        │  │
│   └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
genai_rag_system/
├── .env.example                    # Environment variable template
├── .gitignore                      # Git ignore rules
├── main.py                         # Application entry point
├── pyproject.toml                  # Python project configuration
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation
│
├── config/                         # ─── Configuration Layer ───
│   ├── __init__.py
│   ├── settings.py                 # Centralized settings (Pydantic)
│   └── prompts/
│       └── system_prompt.txt       # RAG system prompt template
│
├── schemas/                        # ─── Data Models ───
│   ├── __init__.py
│   ├── documents.py                # Document & chunk schemas
│   ├── evaluation.py               # Evaluation result schemas
│   └── pipeline.py                 # Query/response schemas
│
├── utils/                          # ─── Shared Utilities ───
│   ├── __init__.py
│   ├── logger.py                   # Structured logging
│   ├── file_handler.py             # File I/O utilities
│   ├── email_sender.py             # Email notification (SMTP)
│   └── exceptions.py               # Custom exception hierarchy
│
├── ingest/                         # ─── Document Ingestion ───
│   ├── __init__.py
│   ├── document_loader.py          # Multi-format document loading
│   ├── text_splitter.py            # Recursive text chunking
│   └── ingestion_pipeline.py       # Ingestion orchestrator
│
├── vectordb/                       # ─── Vector Store ───
│   ├── __init__.py
│   ├── embedding_service.py        # OpenAI embedding generation
│   └── faiss_store.py              # FAISS index management
│
├── pipeline/                       # ─── RAG Pipeline ───
│   ├── __init__.py
│   ├── llm_service.py              # LLM API interaction
│   └── rag_pipeline.py             # Full RAG orchestration
│
├── evaluation/                     # ─── LLM Evaluation ───
│   ├── __init__.py
│   ├── dataset_manager.py          # Evaluation dataset management
│   ├── deepeval_evaluator.py       # DeepEval framework integration
│   ├── ragas_evaluator.py          # RAGAS framework integration
│   └── evaluation_pipeline.py      # End-to-end evaluation orchestrator
│
├── scripts/                        # ─── CLI Scripts ───
│   ├── __init__.py
│   ├── ingest_documents.py         # Document ingestion CLI
│   ├── run_evaluation.py           # Evaluation runner CLI
│   └── check_thresholds.py         # CI/CD threshold checker
│
├── tests/                          # ─── Test Suite ───
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures & config
│   ├── test_ingestion.py           # Ingestion unit tests
│   ├── test_vectordb.py            # Vector store unit tests
│   ├── test_pipeline.py            # Pipeline unit tests
│   └── test_evaluation.py          # Evaluation unit tests
│
├── data/                           # ─── Data Directory ───
│   ├── documents/                  # Source documents for RAG
│   ├── vector_store/               # FAISS index (generated)
│   └── evaluation/
│       └── eval_dataset.json       # Evaluation test cases
│
└── .github/
    └── workflows/
        ├── ci-cd.yml               # Main CI/CD pipeline
        └── evaluation-pipeline.yml  # Scheduled evaluation
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Zero Hardcoding** | All config via `.env` / environment variables using Pydantic Settings |
| **Production Schemas** | Typed data models (Pydantic) for all inputs/outputs |
| **Multi-Format Ingestion** | PDF, TXT, Markdown, DOCX, CSV support |
| **FAISS Vector Store** | Efficient similarity search with persistence |
| **Dual Evaluation** | Both DeepEval and RAGAS for comprehensive LLM assessment |
| **Custom Production Metrics** | Bias, Toxicity, Coherence, Completeness, Conciseness (GEval) |
| **Fully Automated** | Zero CLI arguments — all config from `.env`, just run one command |
| **Email Notifications** | Auto-send HTML evaluation reports to configured recipients via SMTP |
| **CI/CD Pipeline** | GitHub Actions: lint → test → evaluate → threshold check |
| **Modular Design** | Each component is independently testable and replaceable |
| **Custom Exceptions** | Typed error hierarchy for proper error handling |
| **Structured Logging** | Consistent log format across all modules |

---

## Setup & Installation

### 1. Clone & Create Virtual Environment

```bash
cd genai_rag_system
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
# Copy the example env file
cp .env.example .env
```

**Option A: Using OpenAI directly**

```env
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_MODEL=gpt-4o-mini
```

**Option B: Using OpenRouter (recommended — access 200+ models via single API)**

```env
USE_OPENROUTER=true
OPENROUTER_API_KEY=sk-or-your-actual-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

> When `USE_OPENROUTER=true`, **all LLM and embedding calls** (RAG pipeline + vector embeddings + evaluation) route through OpenRouter automatically. No other changes needed.

**Available OpenRouter embedding models:**

| Model | Dimension | Notes |
|-------|-----------|-------|
| `openai/text-embedding-3-small` | 1536 | Default, fast & cheap |
| `openai/text-embedding-3-large` | 3072 | Higher quality, set `EMBEDDING_DIMENSION=3072` |
| `openai/text-embedding-ada-002` | 1536 | Legacy |

**Email setup (optional):**

```env
EMAIL_ENABLED=true
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_SENDER=your-email@gmail.com
EMAIL_SENDER_PASSWORD=your-app-password
EMAIL_RECIPIENTS=recipient1@example.com,recipient2@example.com
```

### 4. Ingest Documents

```bash
# Place documents in data/documents/ directory, then:
python -m scripts.ingest_documents --directory data/documents
```

### 5. Run the Application

```bash
python main.py
```

---

## Configuration

All configuration is centralized in `config/settings.py` and loaded from environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model for generation |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `1000` | Document chunk size |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `SIMILARITY_TOP_K` | `5` | Docs to retrieve |
| `EVAL_THRESHOLD` | `0.7` | Minimum pass score (only in `.env`) |
| `EVAL_FRAMEWORKS` | `deepeval,ragas` | Frameworks to run |
| `EVAL_CUSTOM_METRICS_ENABLED` | `true` | Enable custom production metrics |
| `EVAL_RUN_RAG_PIPELINE` | `true` | Run RAG before evaluation |
| `EMAIL_ENABLED` | `true` | Auto-send email reports |
| `EMAIL_RECIPIENTS` | (required if email) | Comma-separated recipient emails |
| `DEEPEVAL_METRICS` | See .env.example | DeepEval metrics to evaluate |
| `RAGAS_METRICS` | See .env.example | RAGAS metrics to run |

See `.env.example` for the complete list.

---

## Usage

### Interactive Query Mode

```bash
python main.py
```

### Ingest Documents

```bash
# Ingest entire directory
python -m scripts.ingest_documents --directory data/documents

# Ingest single file
python -m scripts.ingest_documents --file path/to/document.pdf

# Clear and re-ingest
python -m scripts.ingest_documents --clear --directory data/documents
```

### Run LLM Evaluation (Fully Automated)

All configuration is read from `.env` — no CLI arguments needed:

```bash
# Run full evaluation pipeline (both DeepEval + RAGAS)
python -m scripts.run_evaluation
```

This single command will:
1. Initialize the RAG pipeline
2. Run evaluation using configured frameworks (`EVAL_FRAMEWORKS`)
3. Apply all standard + custom production metrics
4. Generate combined JSON report
5. Check thresholds
6. Send HTML email report to configured recipients
7. Exit with appropriate code (0=pass, 1=fail)

Configure which frameworks, metrics, and thresholds to use in `.env`:

```env
EVAL_FRAMEWORKS=deepeval,ragas
EVAL_THRESHOLD=0.7
EVAL_CUSTOM_METRICS_ENABLED=true
EVAL_RUN_RAG_PIPELINE=true
```

### Check Thresholds (CI/CD)

```bash
# Also reads everything from .env — no CLI args
python -m scripts.check_thresholds
```

---

## LLM Evaluation Pipeline

### DeepEval Metrics

| Metric | What It Measures | Required Fields |
|--------|-----------------|-----------------|
| `answer_relevancy` | Is the answer relevant to the question? | input, actual_output |
| `faithfulness` | Is the answer grounded in retrieved context? | input, actual_output, retrieval_context |
| `contextual_precision` | Are retrieved docs relevant? | input, expected_output, retrieval_context |
| `contextual_recall` | Did we retrieve all needed info? | input, expected_output, retrieval_context |
| `hallucination` | Does the answer contain fabricated info? | input, actual_output, context |

### Custom Production Metrics (enabled via `EVAL_CUSTOM_METRICS_ENABLED=true`)

| Metric | What It Measures | Type |
|--------|-----------------|------|
| `bias` | Detects biased or prejudiced content in responses | Built-in |
| `toxicity` | Detects harmful, offensive, or toxic content | Built-in |
| `coherence` | Logical structure, readability, and flow of the response | GEval (custom criteria) |
| `completeness` | Thoroughness — covers all aspects of the question | GEval (custom criteria) |
| `conciseness` | Avoids unnecessary repetition and verbosity | GEval (custom criteria) |

### RAGAS Metrics

| Metric | What It Measures |
|--------|-----------------|
| `faithfulness` | Factual consistency with context |
| `answer_relevancy` | Relevance of answer to question |
| `context_precision` | Signal-to-noise in retrieved context |
| `context_recall` | Coverage of ground truth by context |

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

### Workflow: `ci-cd.yml`

```
Push/PR → Code Quality → Unit Tests → Integration Tests → LLM Evaluation → Build
```

1. **Code Quality**: Ruff linting + formatting + mypy type checking
2. **Unit Tests**: All tests with coverage report (70% minimum)
3. **Integration Tests**: Real API tests (main branch only)
4. **LLM Evaluation**: Full DeepEval + RAGAS evaluation with threshold gate
5. **Build**: Package the application

### Workflow: `evaluation-pipeline.yml`

- Scheduled weekly evaluation (Monday 6 AM UTC)
- Manual trigger with configurable framework and threshold
- Results saved as artifacts with 30-day retention

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for evaluation |

### Optional GitHub Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_MODEL` | `gpt-4o-mini` | Model for evaluation |
| `EVAL_THRESHOLD` | `0.7` | Pass threshold |
| `DEEPEVAL_METRICS` | See config | Metrics to run |
| `RAGAS_METRICS` | See config | RAGAS metrics |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=term-missing

# Run specific test file
pytest tests/test_evaluation.py -v

# Run only unit tests (exclude integration)
pytest tests/ -v -k "not integration"

# Run with detailed output
pytest tests/ -v --tb=long
```

---

## Production Deployment Checklist

- [ ] Set `ENVIRONMENT=production` in env
- [ ] Configure real `OPENAI_API_KEY` (or `OPENROUTER_API_KEY` if using OpenRouter)
- [ ] Set appropriate `EVAL_THRESHOLD` (recommend 0.7-0.8)
- [ ] Place source documents in `data/documents/`
- [ ] Run ingestion: `python -m scripts.ingest_documents`
- [ ] Verify vector store: check `data/vector_store/` is populated
- [ ] Configure email: set `EMAIL_ENABLED=true`, `EMAIL_RECIPIENTS`, SMTP credentials in `.env`
- [ ] Enable custom metrics: `EVAL_CUSTOM_METRICS_ENABLED=true`
- [ ] Run evaluation: `python -m scripts.run_evaluation` (no arguments needed)
- [ ] Verify all metrics pass threshold + email report received
- [ ] Set up GitHub Secrets for CI/CD
- [ ] Enable GitHub Actions workflows

---

## LLM Provider Configuration

This system supports **three LLM providers** — configure via `.env`:

### OpenAI (Default)

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

### OpenRouter (Alternative — access 200+ models via single API)

```env
USE_OPENROUTER=true
OPENROUTER_API_KEY=sk-or-your-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

OpenRouter uses the OpenAI-compatible API format. Both LLM and embedding calls route through a single API key.

**LLM models**: `openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`, `google/gemini-pro`, `meta-llama/llama-3-70b-instruct`, etc.

**Embedding models**: `openai/text-embedding-3-small` (1536d), `openai/text-embedding-3-large` (3072d)

> If you change embedding model dimension (e.g., to `text-embedding-3-large`), also set `EMBEDDING_DIMENSION=3072` in `.env`.

### Azure OpenAI (Enterprise)

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

Provider priority: **OpenRouter** (if `USE_OPENROUTER=true`) → **Azure OpenAI** (if endpoint set) → **OpenAI** (default).

---

## Email Notifications

The pipeline automatically sends HTML evaluation reports to configured recipients after each run.

### Configuration (in `.env`)

```env
EMAIL_ENABLED=true
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_SENDER=your-email@gmail.com
EMAIL_SENDER_PASSWORD=your-app-password-here
EMAIL_RECIPIENTS=recipient1@example.com,recipient2@example.com
EMAIL_SUBJECT_PREFIX=[GenAI RAG Eval]
```

### Email Content

- Color-coded HTML report with pass/fail status
- Per-framework metric summary tables
- Per-sample results breakdown
- Full JSON report attached for detailed analysis
- Subject line includes: status (PASSED/FAILED), pass rate, timestamp

> **Gmail users**: Use an [App Password](https://support.google.com/accounts/answer/185833) instead of your regular password.

---

## File Usage Reference

Every file in the project and its specific role:

### Root Files

| File | Usage |
|------|-------|
| `main.py` | Application entry point — starts interactive RAG query mode |
| `requirements.txt` | Python package dependencies for pip install |
| `pyproject.toml` | Project metadata, tool configs (ruff, pytest, mypy) |
| `.env.example` | Template for all environment variables (copy to `.env`) |
| `.gitignore` | Files/dirs excluded from version control |
| `README.md` | This documentation |

### `config/` — Configuration Layer

| File | Usage |
|------|-------|
| `config/__init__.py` | Package init, re-exports `get_settings` |
| `config/settings.py` | **Pydantic Settings class** — loads ALL config from `.env`, zero hardcoding. Uses `@lru_cache` singleton pattern. Supports OpenAI, OpenRouter, Azure OpenAI providers |
| `config/prompts/system_prompt.txt` | RAG system prompt template with `{context}` and `{question}` placeholders — instructs LLM to use only retrieved context |

### `schemas/` — Data Models (Pydantic v2)

| File | Usage |
|------|-------|
| `schemas/__init__.py` | Package init |
| `schemas/documents.py` | `Document`, `DocumentChunk`, `DocumentMetadata` — models for ingested documents |
| `schemas/evaluation.py` | `EvaluationSample`, `EvaluationDataset`, `MetricResult`, `EvaluationResult`, `EvaluationReport` — evaluation data models |
| `schemas/pipeline.py` | `QueryRequest`, `QueryResponse`, `RetrievalResult`, `RAGContext` — RAG pipeline I/O models |

### `utils/` — Shared Utilities

| File | Usage |
|------|-------|
| `utils/__init__.py` | Package init |
| `utils/logger.py` | Structured logging with configurable levels, used by all modules |
| `utils/file_handler.py` | File I/O utilities — safe read/write with validation |
| `utils/email_sender.py` | **Email notifications** — sends HTML evaluation reports via SMTP with color-coded pass/fail tables and JSON attachment. All config from `.env` |
| `utils/exceptions.py` | Custom exception hierarchy: `RAGSystemError` → `DocumentIngestionError`, `VectorStoreError`, `LLMError`, `EvaluationError`, `PipelineError` |

### `ingest/` — Document Ingestion Pipeline

| File | Usage |
|------|-------|
| `ingest/__init__.py` | Package init |
| `ingest/document_loader.py` | Multi-format document loading (PDF, TXT, MD, DOCX, CSV) with file size validation |
| `ingest/text_splitter.py` | Recursive character text splitter — configurable `chunk_size` and `chunk_overlap`, respects sentence/paragraph boundaries |
| `ingest/ingestion_pipeline.py` | Orchestrates full ingestion: load documents → split into chunks → generate embeddings → store in FAISS |

### `vectordb/` — Vector Store Layer

| File | Usage |
|------|-------|
| `vectordb/__init__.py` | Package init |
| `vectordb/embedding_service.py` | OpenAI/OpenRouter embedding generation with batching support — converts text to vectors |
| `vectordb/faiss_store.py` | FAISS index management — add, search, delete, save/load, stats. Uses `IndexFlatIP` with L2-normalized vectors for cosine similarity |

### `pipeline/` — RAG Pipeline

| File | Usage |
|------|-------|
| `pipeline/__init__.py` | Package init |
| `pipeline/llm_service.py` | LLM interaction layer — `generate()` and `generate_with_context()` methods. Supports OpenAI, OpenRouter, Azure. Handles retries and error handling |
| `pipeline/rag_pipeline.py` | Full RAG orchestration — receives query → retrieves relevant chunks from FAISS → constructs prompt with context → calls LLM → returns response with sources |

### `evaluation/` — LLM Evaluation Framework

| File | Usage |
|------|-------|
| `evaluation/__init__.py` | Package init |
| `evaluation/dataset_manager.py` | Load/save/validate evaluation datasets from JSON — handles the `eval_dataset.json` format |
| `evaluation/deepeval_evaluator.py` | **DeepEval integration** — wraps `AnswerRelevancyMetric`, `FaithfulnessMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric`, `HallucinationMetric` + custom production metrics (`BiasMetric`, `ToxicityMetric`, `GEval` for Coherence/Completeness/Conciseness). Supports OpenRouter via env vars |
| `evaluation/ragas_evaluator.py` | **RAGAS integration** — wraps `Faithfulness`, `AnswerRelevancy`, `ContextPrecision`, `ContextRecall` with `LangchainLLMWrapper`. Uses `SingleTurnSample` API |
| `evaluation/evaluation_pipeline.py` | End-to-end evaluation orchestrator — loads dataset → runs RAG pipeline to generate outputs → evaluates with both frameworks → generates reports → saves to disk |

### `scripts/` — CLI Tools

| File | Usage |
|------|-------|
| `scripts/__init__.py` | Package init |
| `scripts/ingest_documents.py` | CLI for document ingestion — `--directory`, `--file`, `--clear` options |
| `scripts/run_evaluation.py` | **Fully automated** evaluation runner — no CLI args, reads all config from `.env`. Runs both frameworks, generates report, checks thresholds, sends email |
| `scripts/check_thresholds.py` | CI/CD threshold gate — reads latest report and threshold from `.env`, no CLI args. Exits 0/1 based on compliance |

### `tests/` — Test Suite

| File | Usage |
|------|-------|
| `tests/__init__.py` | Package init |
| `tests/conftest.py` | Pytest fixtures: `set_test_env`, `sample_documents`, `sample_eval_dataset`, `mock_openai_client`, `temp_documents_dir`, `eval_dataset_file` |
| `tests/test_ingestion.py` | Unit tests for document loading and text splitting |
| `tests/test_vectordb.py` | Unit tests for embedding service and FAISS store |
| `tests/test_pipeline.py` | Unit tests for LLM service and RAG pipeline (mocked API) |
| `tests/test_evaluation.py` | Unit tests for DeepEval evaluator, RAGAS evaluator, and evaluation pipeline |

### `data/` — Data Directory

| Path | Usage |
|------|-------|
| `data/documents/` | Place source documents here for RAG ingestion |
| `data/vector_store/` | Generated FAISS index files (auto-created by ingestion) |
| `data/evaluation/eval_dataset.json` | **Evaluation test cases** — 20 samples covering all metrics. Contains `user_input`, `expected_output`, `context` (ground truth for Hallucination), `retrieval_context` (for Faithfulness/Precision/Recall) |

### `.github/workflows/` — CI/CD Pipelines

| File | Usage |
|------|-------|
| `.github/workflows/ci-cd.yml` | Main pipeline: triggered on push/PR → lint (ruff) → test (pytest+coverage) → integration tests → LLM evaluation → build |
| `.github/workflows/evaluation-pipeline.yml` | Scheduled weekly evaluation (Monday 6AM UTC) + manual trigger. Runs full DeepEval+RAGAS evaluation, checks thresholds, stores reports as artifacts |
- [ ] Configure monitoring/alerting for evaluation failures

---

## Design Principles

1. **No Hardcoded Values** — Every configurable value comes from environment variables
2. **Separation of Concerns** — Each module has a single responsibility
3. **Dependency Injection** — Components accept interfaces, not concrete implementations
4. **Fail Fast** — Validate at boundaries, raise typed exceptions
5. **Observable** — Structured logging at every significant operation
6. **Testable** — Every component can be tested in isolation with mocks
7. **CI/CD First** — Pipeline ensures quality gates before any deployment