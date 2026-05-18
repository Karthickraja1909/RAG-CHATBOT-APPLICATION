# RAGAs (Ragas) - Complete Learning Guide
## Retrieval Augmented Generation Assessment Framework

> **Version:** Ragas 0.4.x (Stable - July 2025)
> **Source:** [docs.ragas.io](https://docs.ragas.io/en/stable/)
> **GitHub:** [vibrantlabsai/ragas](https://github.com/vibrantlabsai/ragas) (13.8k+ stars)

---

## Table of Contents

1. [What is RAGAs & Why Use It](#1-what-is-ragas--why-use-it)
2. [Installation & Setup](#2-installation--setup)
3. [Core Concepts & Architecture](#3-core-concepts--architecture)
4. [RAG Evaluation Metrics - Deep Dive](#4-rag-evaluation-metrics---deep-dive)
5. [Context Precision](#5-context-precision)
6. [Context Recall](#6-context-recall)
7. [Faithfulness](#7-faithfulness)
8. [Answer Relevancy](#8-answer-relevancy)
9. [Additional Metrics](#9-additional-metrics)
10. [Datasets & Evaluation Samples](#10-datasets--evaluation-samples)
11. [Running Evaluations](#11-running-evaluations)
12. [Test Data Generation (Synthetic)](#12-test-data-generation-synthetic)
13. [Agent & Workflow Evaluation](#13-agent--workflow-evaluation)
14. [General Purpose Metrics](#14-general-purpose-metrics)
15. [Integrations (LangChain, LlamaIndex, etc.)](#15-integrations-langchain-llamaindex-etc)
16. [Customization & Advanced Usage](#16-customization--advanced-usage)
17. [Production Best Practices & CI/CD](#17-production-best-practices--cicd)
18. [Syntax Cheat Sheet](#18-syntax-cheat-sheet)
19. [Complete Code Examples (5 Full Scripts)](#19-complete-code-examples-5-full-scripts)
20. [Interview Q&A (30 Questions)](#20-interview-qa-30-questions)
21. [Troubleshooting & Common Errors](#21-troubleshooting--common-errors)
22. [RAGAs vs DeepEval Comparison](#22-ragas-vs-deepeval-comparison)

---

## 1. What is RAGAs & Why Use It

### What is RAGAs?

**RAGAs** (Retrieval Augmented Generation Assessment) is an open-source Python framework for evaluating LLM applications, with a primary focus on **RAG (Retrieval-Augmented Generation)** pipelines. It provides metrics to objectively measure the quality of retrieval, generation, and end-to-end performance.

### Why RAGAs?

| Feature | Description |
|---------|-------------|
| **RAG-Specific Metrics** | Purpose-built metrics for evaluating retrieval + generation quality |
| **LLM-as-a-Judge** | Uses LLMs to evaluate LLM outputs with structured prompts |
| **No Human Labels Needed** | Many metrics work without reference answers |
| **Framework Agnostic** | Works with LangChain, LlamaIndex, Haystack, or custom pipelines |
| **Async-First** | Built on async Python for high throughput |
| **Test Data Generation** | Automatically generate synthetic evaluation datasets |
| **Production Ready** | Caching, cost analysis, CI/CD integration |
| **Extensible** | Create custom metrics, modify prompts, train your own evaluators |

### When to Use RAGAs

- Evaluating RAG pipeline quality (retrieval + generation)
- Comparing different retriever configurations (chunking, embedding models)
- Comparing different LLMs for generation quality
- Regression testing RAG pipelines in CI/CD
- Benchmarking LLM applications before production
- Evaluating AI agents and tool-use workflows

### RAGAs vs Other Frameworks

| Aspect | RAGAs | DeepEval | Traditional Metrics |
|--------|-------|----------|-------------------|
| **Focus** | RAG-specific evaluation | General LLM evaluation | Text similarity |
| **Metrics** | Context Precision/Recall, Faithfulness | G-Eval, Hallucination, Bias | BLEU, ROUGE, F1 |
| **Test Data** | Synthetic testset generation with KG | Dataset management | Manual creation |
| **Integrations** | LangChain, LlamaIndex, Haystack | PyTest native | N/A |
| **API Style** | Async-first, collections-based | PyTest assertions | Function calls |

---

## 2. Installation & Setup

### Basic Installation

```bash
# Install ragas
pip install ragas

# Install with OpenAI support (most common)
pip install ragas openai

# Install from latest main branch
pip install git+https://github.com/vibrantlabsai/ragas.git

# Install for development
git clone https://github.com/vibrantlabsai/ragas.git
pip install -e .
```

### LangChain Compatibility Note

```bash
# If using LangChain integration, pin versions to avoid conflicts
pip install -U "langchain-core>=0.2,<0.3" "langchain-openai>=0.1,<0.2" openai
```

### Environment Setup

```bash
# Set your OpenAI API key (required for LLM-based metrics)
# Windows PowerShell
$env:OPENAI_API_KEY = "your-openai-key"

# Linux/Mac
export OPENAI_API_KEY="your-openai-key"

# Or use .env file with python-dotenv
pip install python-dotenv
```

```python
# .env file approach
import os
from dotenv import load_dotenv
load_dotenv()

# Verify setup
import ragas
print(f"Ragas version: {ragas.__version__}")
```

### Quick Start with CLI

```bash
# Create a new evaluation project (recommended for beginners)
uvx ragas quickstart rag_eval
cd rag_eval

# Install dependencies
uv sync  # or: pip install -e .

# Run evaluation
python evals.py
```

### Project Structure (CLI-generated)

```
rag_eval/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── rag.py                 # Your RAG application
├── evals.py               # Evaluation workflow
├── __init__.py
└── evals/
    ├── datasets/          # Test data files (CSV)
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

---

## 3. Core Concepts & Architecture

### Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    RAGAs Framework                     │
├──────────────────────────────────────────────────────┤
│                                                       │
│   ┌─────────────┐   ┌──────────────┐   ┌──────────┐ │
│   │  Evaluation  │   │   Metrics    │   │ Test Data│ │
│   │   Samples    │   │ Collections  │   │  Synth   │ │
│   │             │   │              │   │          │ │
│   │ SingleTurn  │   │ Faithfulness │   │   KG     │ │
│   │ MultiTurn   │   │ Precision    │   │ Builder  │ │
│   │             │   │ Recall       │   │          │ │
│   │ Dataset     │   │ Relevancy    │   │ Scenario │ │
│   │ Experiment  │   │ AspectCritic │   │ Generator│ │
│   └─────────────┘   └──────────────┘   └──────────┘ │
│                                                       │
│   ┌─────────────┐   ┌──────────────┐   ┌──────────┐ │
│   │    LLMs     │   │  Embeddings  │   │ Prompts  │ │
│   │  (Judges)   │   │              │   │          │ │
│   │             │   │ OpenAI       │   │ Custom   │ │
│   │ OpenAI      │   │ HuggingFace  │   │ Template │ │
│   │ Anthropic   │   │ Azure        │   │ Modify   │ │
│   │ Gemini      │   │              │   │          │ │
│   │ Ollama      │   │              │   │          │ │
│   └─────────────┘   └──────────────┘   └──────────┘ │
└──────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Evaluation Samples

```python
from ragas import SingleTurnSample

# A single evaluation sample = one test case
sample = SingleTurnSample(
    user_input="What is Python?",                    # The user's question
    response="Python is a programming language.",     # LLM's answer
    reference="Python is a high-level programming language.", # Ground truth
    retrieved_contexts=[                              # Retrieved documents
        "Python is a high-level, interpreted programming language.",
        "Python was created by Guido van Rossum."
    ]
)
```

#### 2. LLM Factory (Judge Model)

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory

# Create an LLM instance for evaluation
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
```

#### 3. Embedding Factory

```python
from ragas.embeddings.base import embedding_factory

# Create embeddings for similarity-based metrics
embeddings = embedding_factory(
    "openai",
    model="text-embedding-3-small",
    client=client
)
```

#### 4. Metrics (Collections-based API - Recommended)

```python
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
    ContextUtilization,
)

# Create a metric instance
scorer = Faithfulness(llm=llm)

# Score a single sample
result = await scorer.ascore(
    user_input="...",
    response="...",
    retrieved_contexts=["..."]
)
print(result.value)  # 0.0 to 1.0
```

#### 5. Datasets & Experiments

```python
from ragas import Dataset

# Create a dataset for evaluation
dataset = Dataset(
    name="my_test_dataset",
    backend="local/csv",
    root_dir=".",
)

# Add samples
dataset.append({
    "question": "What is Ragas?",
    "grading_notes": "Ragas is an evaluation framework for LLM applications"
})
dataset.save()
```

### New API vs Legacy API

> **Important:** Ragas 0.4+ introduces a **collections-based API**. The legacy `SingleTurnSample` + `single_turn_ascore()` pattern is deprecated and will be removed in v1.0.

| Feature | New API (Recommended) | Legacy API (Deprecated) |
|---------|----------------------|------------------------|
| **Import** | `from ragas.metrics.collections import Faithfulness` | `from ragas.metrics import Faithfulness` |
| **Score** | `scorer.ascore(user_input=..., response=...)` | `scorer.single_turn_ascore(sample)` |
| **Sync** | `scorer.score(...)` | N/A |
| **Input** | Keyword arguments | `SingleTurnSample` object |

---

## 4. RAG Evaluation Metrics - Deep Dive

### The RAG Evaluation Framework

RAGAs evaluates RAG pipelines across two critical dimensions:

```
User Query ──► Retriever ──► Retrieved Contexts ──► Generator ──► Response
                  │                                      │
                  ▼                                      ▼
         ┌───────────────┐                    ┌──────────────────┐
         │   RETRIEVAL   │                    │    GENERATION    │
         │   METRICS     │                    │    METRICS       │
         │               │                    │                  │
         │ • Context     │                    │ • Faithfulness   │
         │   Precision   │                    │ • Answer         │
         │ • Context     │                    │   Relevancy      │
         │   Recall      │                    │ • Factual        │
         │ • Noise       │                    │   Correctness    │
         │   Sensitivity │                    │                  │
         └───────────────┘                    └──────────────────┘
```

### Metric Categories

| Category | Metrics | What It Measures |
|----------|---------|-----------------|
| **Retrieval** | Context Precision, Context Recall, Context Entities Recall, Noise Sensitivity | Quality of retrieved documents |
| **Generation** | Faithfulness, Answer Relevancy | Quality of LLM's response |
| **End-to-End** | Factual Correctness, Semantic Similarity | Overall answer quality |
| **Agent** | Topic Adherence, Tool Call Accuracy, Tool Call F1, Agent Goal Accuracy | Agent behavior quality |
| **General** | Aspect Critic, Simple Criteria Scoring, Rubrics-based Scoring | Custom evaluation criteria |
| **Traditional NLP** | BLEU, ROUGE, CHRF, Exact Match, String Presence | Text-level comparison |

---

## 5. Context Precision

### What is Context Precision?

Context Precision evaluates whether the **retriever ranks relevant documents higher** than irrelevant ones. It measures how well relevant chunks are placed at the top of the retrieval results.

### Formula

$$\text{Context Precision@K} = \frac{\sum_{k=1}^{K} (\text{Precision@k} \times v_k)}{\text{Total relevant items in top K}}$$

Where:
- $K$ = total number of chunks in retrieved_contexts
- $v_k \in \{0, 1\}$ = relevance indicator at rank $k$
- $\text{Precision@k} = \frac{\text{true positives@k}}{\text{true positives@k + false positives@k}}$

### Two Variants

| Variant | When to Use | Compares Against |
|---------|------------|-----------------|
| **ContextPrecision** | When you have a reference answer | `reference` |
| **ContextUtilization** | When you only have the response (no reference) | `response` |

### Code Example - Context Precision

```python
import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision

async def evaluate_context_precision():
    # Setup
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)
    scorer = ContextPrecision(llm=llm)

    # Evaluate - relevant doc ranked FIRST (good!)
    result = await scorer.ascore(
        user_input="Where is the Eiffel Tower located?",
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=[
            "The Eiffel Tower is located in Paris.",          # Relevant - rank 1
            "The Brandenburg Gate is located in Berlin."       # Irrelevant - rank 2
        ]
    )
    print(f"Context Precision (relevant first): {result.value}")
    # Output: 0.9999999999 (near 1.0 - excellent!)

asyncio.run(evaluate_context_precision())
```

### Code Example - Context Utilization (No Reference)

```python
from ragas.metrics.collections import ContextUtilization

async def evaluate_context_utilization():
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)
    scorer = ContextUtilization(llm=llm)

    # Irrelevant doc ranked FIRST (bad!)
    result = await scorer.ascore(
        user_input="Where is the Eiffel Tower located?",
        response="The Eiffel Tower is located in Paris.",
        retrieved_contexts=[
            "The Brandenburg Gate is located in Berlin.",  # Irrelevant - rank 1
            "The Eiffel Tower is located in Paris."        # Relevant - rank 2
        ]
    )
    print(f"Context Utilization (irrelevant first): {result.value}")
    # Output: 0.49999999995 (penalized for bad ranking!)
```

### Sync Usage

```python
# Use .score() instead of .ascore() for synchronous code
result = scorer.score(
    user_input="Where is the Eiffel Tower located?",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=[...]
)
```

### Non-LLM Context Precision

```python
from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference

# No LLM needed - uses string similarity (Levenshtein distance)
# Requires: pip install rapidfuzz
context_precision = NonLLMContextPrecisionWithReference()

sample = SingleTurnSample(
    retrieved_contexts=["The Eiffel Tower is located in Paris."],
    reference_contexts=[
        "Paris is the capital of France.",
        "The Eiffel Tower is one of the most famous landmarks in Paris."
    ]
)
score = await context_precision.single_turn_ascore(sample)
```

### ID-Based Context Precision

```python
from ragas import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision

sample = SingleTurnSample(
    retrieved_context_ids=["doc_1", "doc_2", "doc_3", "doc_4"],
    reference_context_ids=["doc_1", "doc_4", "doc_5", "doc_6"]
)

id_precision = IDBasedContextPrecision()
score = await id_precision.single_turn_ascore(sample)
# Output: 0.5 (2 out of 4 retrieved docs are relevant)
```

---

## 6. Context Recall

### What is Context Recall?

Context Recall measures **how many relevant documents were successfully retrieved**. It focuses on not missing important information. Higher recall = fewer relevant documents left out.

### Formula

$$\text{Context Recall} = \frac{\text{Number of claims in reference supported by retrieved context}}{\text{Total number of claims in reference}}$$

### How It Works

1. Break the `reference` answer into individual claims
2. Check each claim against the `retrieved_contexts` to see if it can be attributed
3. Calculate the ratio of supported claims

### Code Example

```python
import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall

async def evaluate_context_recall():
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)
    scorer = ContextRecall(llm=llm)

    result = await scorer.ascore(
        user_input="Where is the Eiffel Tower located?",
        retrieved_contexts=["Paris is the capital of France."],
        reference="The Eiffel Tower is located in Paris."
    )
    print(f"Context Recall: {result.value}")
    # Output: 1.0 (the reference claim is supported by the context)

asyncio.run(evaluate_context_recall())
```

### Non-LLM Context Recall

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextRecall

sample = SingleTurnSample(
    retrieved_contexts=["Paris is the capital of France."],
    reference_contexts=[
        "Paris is the capital of France.",
        "The Eiffel Tower is one of the most famous landmarks in Paris."
    ]
)

context_recall = NonLLMContextRecall()
score = await context_recall.single_turn_ascore(sample)
# Output: 0.5 (1 out of 2 reference contexts found)
```

### ID-Based Context Recall

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import IDBasedContextRecall

sample = SingleTurnSample(
    retrieved_context_ids=["doc_1", "doc_2", "doc_3"],
    reference_context_ids=["doc_1", "doc_4", "doc_5", "doc_6"]
)

id_recall = IDBasedContextRecall()
score = await id_recall.single_turn_ascore(sample)
# Output: 0.25 (1 out of 4 reference docs retrieved)
```

---

## 7. Faithfulness

### What is Faithfulness?

Faithfulness measures **how factually consistent the response is with the retrieved context**. A response is faithful if all its claims can be supported by the retrieved documents.

### Formula

$$\text{Faithfulness Score} = \frac{\text{Number of claims supported by retrieved context}}{\text{Total number of claims in the response}}$$

### How It's Calculated (Step by Step)

**Example:**
- **Question:** "Where and when was Einstein born?"
- **Context:** "Albert Einstein (born 14 March 1879) was a German-born theoretical physicist."
- **High faithfulness answer:** "Einstein was born in Germany on 14th March 1879."
- **Low faithfulness answer:** "Einstein was born in Germany on 20th March 1879."

**Steps for low faithfulness answer:**

1. **Break into statements:**
   - Statement 1: "Einstein was born in Germany." ✅
   - Statement 2: "Einstein was born on 20th March 1879." ❌

2. **Verify against context:**
   - Statement 1: Supported by context → Yes
   - Statement 2: Not supported (context says 14 March) → No

3. **Calculate:** Faithfulness = 1/2 = **0.5**

### Code Example

```python
import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

async def evaluate_faithfulness():
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)
    scorer = Faithfulness(llm=llm)

    # High faithfulness - all claims supported by context
    result = await scorer.ascore(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL-NFL World Championship Game was an American football "
            "game played on January 15, 1967, at the Los Angeles Memorial "
            "Coliseum in Los Angeles."
        ]
    )
    print(f"Faithfulness Score: {result.value}")
    # Output: 1.0 (fully faithful!)

asyncio.run(evaluate_faithfulness())
```

### Faithfulness with HHEM (Hallucination Detection Model)

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import FaithfulnesswithHHEM

# Uses Vectara's HHEM-2.1-Open model (free, small, open-source)
# Great for production - no LLM API calls needed for verification step
sample = SingleTurnSample(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=[
        "The First AFL-NFL World Championship Game was played on January 15, 1967."
    ]
)

scorer = FaithfulnesswithHHEM(llm=evaluator_llm)
score = await scorer.single_turn_ascore(sample)

# Configure device and batch size
scorer = FaithfulnesswithHHEM(
    device="cuda:0",      # GPU acceleration
    batch_size=10          # Batch processing
)
```

---

## 8. Answer Relevancy

### What is Answer Relevancy?

Answer Relevancy measures **how relevant the response is to the user's question**. It focuses on whether the answer addresses the question's intent, without evaluating factual accuracy.

### Formula

$$\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \cos(\mathbf{E}_{g_i}, \mathbf{E}_o)$$

Where:
- $\mathbf{E}_{g_i}$ = embedding of the $i$-th generated question
- $\mathbf{E}_o$ = embedding of the original user input
- $N$ = number of generated questions (default: 3)

### How It Works

1. **Reverse-engineer questions** from the response using an LLM
2. **Compute cosine similarity** between generated questions and the original question
3. **Average** the similarity scores

**Intuition:** If the answer is relevant, you should be able to reconstruct the original question from it.

### Code Example

```python
import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerRelevancy

async def evaluate_answer_relevancy():
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)
    embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=client
    )

    scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)

    result = await scorer.ascore(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967"
    )
    print(f"Answer Relevancy Score: {result.value}")
    # Output: ~0.9165 (highly relevant!)

asyncio.run(evaluate_answer_relevancy())
```

### Relevancy Scoring Explained

| Scenario | Score | Why |
|----------|-------|-----|
| Direct answer to question | ~0.9+ | Generated questions closely match original |
| Partial answer | ~0.5-0.8 | Some generated questions match |
| Irrelevant answer | ~0.0-0.3 | Generated questions don't match original |
| Answer with extra unnecessary info | Lower | Penalized for including unrelated details |

---

## 9. Additional Metrics

### Context Entities Recall

Measures how many named entities from the reference are present in the retrieved contexts.

```python
from ragas.metrics import ContextEntityRecall

scorer = ContextEntityRecall(llm=evaluator_llm)
# Uses NER to compare entities between reference and retrieved contexts
```

### Noise Sensitivity

Measures how much irrelevant information (noise) in retrieved contexts affects the response quality.

```python
from ragas.metrics import NoiseSensitivity

scorer = NoiseSensitivity(llm=evaluator_llm)
# Evaluates robustness to noisy/irrelevant retrieved documents
```

### Factual Correctness

Evaluates factual accuracy of the response by comparing against a reference answer.

```python
from ragas.metrics import FactualCorrectness

scorer = FactualCorrectness(llm=evaluator_llm)
# Compares claims in response against claims in reference
```

### Semantic Similarity

Measures semantic similarity between response and reference using embeddings.

```python
from ragas.metrics import SemanticSimilarity

scorer = SemanticSimilarity(embeddings=evaluator_embeddings)
# Uses cosine similarity between response and reference embeddings
```

### Summarization Score

Evaluates quality of text summarization.

```python
from ragas.metrics import Summarization

scorer = Summarization(llm=evaluator_llm)
```

### Traditional NLP Metrics (No LLM Required)

```python
from ragas.metrics import (
    BleuScore,          # BLEU score
    RougeScore,         # ROUGE score
    StringPresence,     # Check if string is present
    ExactMatch,         # Exact string match
    NonLLMStringSimilarity,  # Levenshtein distance
)

# These are fast, cheap, and don't require LLM API calls
bleu = BleuScore()
rouge = RougeScore()
```

---

## 10. Datasets & Evaluation Samples

### SingleTurnSample

```python
from ragas import SingleTurnSample

# Complete sample with all possible fields
sample = SingleTurnSample(
    user_input="What is machine learning?",          # Required: user query
    response="ML is a subset of AI...",              # LLM response
    reference="Machine learning is a field of AI...",# Ground truth answer
    retrieved_contexts=[                              # Retrieved documents
        "Machine learning (ML) is a field of study...",
        "Deep learning is a subset of ML..."
    ],
    reference_contexts=[                              # Ideal contexts (for recall)
        "Machine learning is defined as..."
    ],
    retrieved_context_ids=["doc_1", "doc_2"],         # Document IDs
    reference_context_ids=["doc_1", "doc_3"],         # Reference doc IDs
)
```

### MultiTurnSample

```python
from ragas import MultiTurnSample

# For evaluating multi-turn conversations
sample = MultiTurnSample(
    user_input=[
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "What are its main features?"},
        {"role": "assistant", "content": "Python features include..."}
    ]
)
```

### Creating Evaluation Datasets

```python
from ragas import Dataset
import pandas as pd

# Method 1: Using Dataset class
dataset = Dataset(
    name="rag_test_suite",
    backend="local/csv",
    root_dir="./evals/datasets",
)

samples = [
    {
        "question": "What is Ragas?",
        "grading_notes": "Ragas is an evaluation framework for LLM applications"
    },
    {
        "question": "How to install Ragas?",
        "grading_notes": "Install from pip using: pip install ragas"
    },
    {
        "question": "What are the main features?",
        "grading_notes": "Organized around experiments, datasets, and metrics"
    },
]

for sample in samples:
    dataset.append(sample)
dataset.save()

# Method 2: From pandas DataFrame
df = pd.DataFrame(samples)
df.to_csv("datasets/test_dataset.csv", index=False)

# Method 3: From HuggingFace datasets
from datasets import Dataset as HFDataset

hf_dataset = HFDataset.from_dict({
    "question": ["What is RAG?", "How does retrieval work?"],
    "answer": ["RAG combines retrieval with generation.", "..."],
    "contexts": [["doc1 text", "doc2 text"], ["doc3 text"]],
    "ground_truth": ["RAG is...", "Retrieval works by..."]
})
```

---

## 11. Running Evaluations

### Method 1: Experiment Decorator (Recommended for Projects)

```python
from ragas import Dataset
from ragas.metrics import DiscreteMetric
from ragas.llms import llm_factory
from ragas.experiment import experiment
from openai import AsyncOpenAI

# Setup
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Define metric
my_metric = DiscreteMetric(
    name="correctness",
    prompt=(
        "Check if the response contains points from the grading notes "
        "and return 'pass' or 'fail'.\n"
        "Response: {response}\n"
        "Grading Notes: {grading_notes}"
    ),
    allowed_values=["pass", "fail"],
)

# Define experiment
@experiment()
async def run_experiment(row):
    # Call your RAG pipeline
    response = your_rag_pipeline(row["question"])

    # Score with metric
    score = my_metric.score(
        llm=llm,
        response=response,
        grading_notes=row["grading_notes"]
    )

    return {
        **row,
        "response": response,
        "score": score.value,
    }

# Run
# Results saved to evals/experiments/experiment_name.csv
```

### Method 2: Individual Metric Scoring

```python
import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, ContextPrecision

async def run_eval():
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    # Create metrics
    faithfulness = Faithfulness(llm=llm)
    precision = ContextPrecision(llm=llm)

    # Test data
    user_input = "What is Python?"
    response = "Python is a high-level programming language created by Guido van Rossum."
    reference = "Python is a high-level, interpreted programming language."
    contexts = [
        "Python is a high-level, interpreted programming language.",
        "Java is a compiled programming language."
    ]

    # Score each metric
    faith_result = await faithfulness.ascore(
        user_input=user_input,
        response=response,
        retrieved_contexts=contexts
    )

    prec_result = await precision.ascore(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=contexts
    )

    print(f"Faithfulness: {faith_result.value}")
    print(f"Context Precision: {prec_result.value}")

asyncio.run(run_eval())
```

### Method 3: Batch Evaluation with evaluate()

```python
from ragas import evaluate, SingleTurnSample
from ragas.metrics import Faithfulness, ContextPrecision

# Create samples
samples = [
    SingleTurnSample(
        user_input="What is Python?",
        response="Python is a programming language.",
        retrieved_contexts=["Python is a high-level programming language."],
        reference="Python is a high-level interpreted language."
    ),
    SingleTurnSample(
        user_input="What is Java?",
        response="Java is a compiled language.",
        retrieved_contexts=["Java is an object-oriented programming language."],
        reference="Java is a compiled, object-oriented language."
    ),
]

# Run evaluation
results = evaluate(
    samples=samples,
    metrics=[
        Faithfulness(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
    ]
)

# Access results
print(results)  # DataFrame with scores per sample
print(results.to_pandas())
```

### Method 4: Async Evaluation with aevaluate()

```python
from ragas import aevaluate

# Async version for better performance
results = await aevaluate(
    samples=samples,
    metrics=[faithfulness, context_precision],
)
```

---

## 12. Test Data Generation (Synthetic)

### Why Generate Test Data?

- Creating manual test datasets is time-consuming and expensive
- Synthetic data covers edge cases you might miss
- Ragas uses Knowledge Graphs to generate diverse, realistic test scenarios

### How Test Generation Works

```
Documents ──► Knowledge Graph ──► Scenario Generation ──► Test Samples
                  │                      │
                  ▼                      ▼
           Entity/Relation        Single-hop queries
           Extraction             Multi-hop queries
                                  Reasoning queries
```

### Generate Test Data for RAG

```python
from ragas.testset import TestsetGenerator
from ragas.llms import llm_factory
from langchain_community.document_loaders import DirectoryLoader

# Load your documents
loader = DirectoryLoader("./docs/", glob="**/*.md")
documents = loader.load()

# Setup generator
generator = TestsetGenerator(llm=llm_factory("gpt-4o-mini"))

# Generate testset
testset = generator.generate_with_langchain_docs(
    documents=documents,
    test_size=20,           # Number of test cases
)

# Convert to pandas DataFrame
test_df = testset.to_pandas()
print(test_df.columns)
# ['question', 'contexts', 'ground_truth', 'evolution_type', ...]

# Save for later use
test_df.to_csv("testset.csv", index=False)
```

### Test Generation for Agents

```python
from ragas.testset.agents import AgentTestsetGenerator

# Generate test scenarios for agent evaluation
agent_generator = AgentTestsetGenerator(llm=llm)
agent_testset = agent_generator.generate(
    tools=your_tools,       # Agent's available tools
    test_size=15,
)
```

### Custom Query Types

```python
# Single-hop queries (simple retrieval)
# Multi-hop queries (require combining info from multiple docs)
# Reasoning queries (require logical inference)

# Customize with pre-chunked data
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=llm)
testset = generator.generate(
    documents=pre_chunked_docs,
    test_size=20,
    query_distribution={
        "single_hop": 0.5,
        "multi_hop": 0.3,
        "reasoning": 0.2,
    }
)
```

---

## 13. Agent & Workflow Evaluation

### Agent Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Topic Adherence** | Does the agent stay on-topic? |
| **Tool Call Accuracy** | Are tool calls correct and appropriate? |
| **Tool Call F1** | Precision & recall of tool calls |
| **Agent Goal Accuracy** | Does the agent achieve the user's goal? |

### Evaluate an AI Agent

```python
from ragas.metrics import (
    TopicAdherenceScore,
    ToolCallAccuracy,
    AgentGoalAccuracyWithReference,
)

# Topic Adherence
topic_scorer = TopicAdherenceScore(llm=evaluator_llm)

# Tool Call Accuracy
tool_scorer = ToolCallAccuracy()

# Agent Goal Accuracy
goal_scorer = AgentGoalAccuracyWithReference(llm=evaluator_llm)
```

### Evaluate an AI Workflow

```python
# Ragas supports evaluating multi-step workflows
# Each step can be evaluated independently or as a whole

from ragas import evaluate

# Define workflow steps as samples
workflow_samples = [
    # Step 1: Query understanding
    SingleTurnSample(
        user_input="Find the top 3 Python frameworks",
        response="Searching for Python frameworks...",
        # ...
    ),
    # Step 2: Retrieval
    # Step 3: Generation
]

results = evaluate(
    samples=workflow_samples,
    metrics=[faithfulness, answer_relevancy],
)
```

---

## 14. General Purpose Metrics

### Aspect Critic

Binary evaluation (pass/fail) based on custom criteria.

```python
from ragas.metrics import AspectCritic

# Define custom evaluation criteria
politeness_critic = AspectCritic(
    name="politeness",
    definition="Is the response polite and professional?",
    llm=evaluator_llm,
)

score = await politeness_critic.single_turn_ascore(sample)
# Returns: 0 (fail) or 1 (pass)
```

### Simple Criteria Scoring

Numeric scoring (1-5) based on custom criteria.

```python
from ragas.metrics import SimpleCriteriaScore

clarity_scorer = SimpleCriteriaScore(
    name="clarity",
    definition="How clear and understandable is the response?",
    llm=evaluator_llm,
)

score = await clarity_scorer.single_turn_ascore(sample)
# Returns: 1-5 score
```

### Rubrics-Based Scoring

Detailed scoring with predefined rubrics.

```python
from ragas.metrics import RubricsCriteriaScore

rubric_scorer = RubricsCriteriaScore(
    name="completeness",
    rubrics={
        1: "Response is completely incomplete, missing all key points",
        2: "Response covers less than half of the key points",
        3: "Response covers about half of the key points",
        4: "Response covers most key points with minor gaps",
        5: "Response is comprehensive, covering all key points",
    },
    llm=evaluator_llm,
)

score = await rubric_scorer.single_turn_ascore(sample)
```

### DiscreteMetric (New API)

```python
from ragas.metrics import DiscreteMetric

# Custom discrete metric with any allowed values
sentiment_metric = DiscreteMetric(
    name="sentiment",
    prompt=(
        "Analyze the sentiment of this response: {response}\n"
        "Return 'positive', 'neutral', or 'negative'."
    ),
    allowed_values=["positive", "neutral", "negative"],
)

result = sentiment_metric.score(
    llm=llm,
    response="I love this product! It's amazing!"
)
print(result.value)  # "positive"
```

---

## 15. Integrations (LangChain, LlamaIndex, etc.)

### Supported Integrations

| Integration | Description |
|-------------|-------------|
| **LangChain** | Evaluate LangChain RAG pipelines directly |
| **LlamaIndex** | Evaluate LlamaIndex query engines |
| **Haystack** | Evaluate Haystack pipelines |
| **LangSmith** | Log evaluations to LangSmith |
| **Arize** | Monitor evaluations in Arize |
| **Amazon Bedrock** | Use Bedrock models as evaluators |
| **Google Gemini** | Use Gemini models as evaluators |
| **Ollama** | Use local models as evaluators |
| **LangGraph** | Evaluate LangGraph agent workflows |
| **LlamaStack** | Evaluate LlamaStack applications |

### LangChain Integration

```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

# Wrap LangChain LLM for use with Ragas
langchain_llm = ChatOpenAI(model="gpt-4o-mini")
ragas_llm = LangchainLLMWrapper(langchain_llm)

# Use in metrics
scorer = Faithfulness(llm=ragas_llm)
```

### Using Different LLM Providers

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory

# OpenAI (default)
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Anthropic Claude
client = AsyncOpenAI(
    api_key="your-anthropic-key",
    base_url="https://api.anthropic.com/v1"
)
llm = llm_factory("claude-sonnet-4-20250514", client=client)

# Google Gemini
client = AsyncOpenAI(
    api_key="your-gemini-key",
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)
llm = llm_factory("gemini-2.0-flash", client=client)

# Ollama (Local)
client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)
llm = llm_factory("llama3", client=client)
```

---

## 16. Customization & Advanced Usage

### Customize Evaluation Models

```python
from ragas.llms import llm_factory
from openai import AsyncOpenAI

# Use different models for different metrics
fast_llm = llm_factory("gpt-4o-mini", client=AsyncOpenAI())
strong_llm = llm_factory("gpt-4o", client=AsyncOpenAI())

# Fast model for simple metrics
faithfulness = Faithfulness(llm=fast_llm)

# Strong model for nuanced metrics
aspect_critic = AspectCritic(
    name="technical_accuracy",
    definition="Is the response technically accurate?",
    llm=strong_llm,
)
```

### Modify Metric Prompts

```python
# Customize the prompt used by a metric
from ragas.metrics.collections import Faithfulness

scorer = Faithfulness(llm=llm)

# Access and modify the internal prompt
# (useful for domain-specific evaluation)
scorer.prompt = """
Custom prompt for faithfulness evaluation in medical domain:
Given the context: {context}
And the response: {response}
Evaluate if all medical claims are supported by the context.
"""
```

### Adapt Metrics to Non-English Languages

```python
# Ragas supports language adaptation for metrics
# Modify prompts to work with your target language
from ragas.metrics.collections import Faithfulness

scorer = Faithfulness(llm=llm)
# The LLM-based approach naturally handles multilingual content
# For best results, use a multilingual LLM as the judge
```

### RunConfig for Performance Tuning

```python
from ragas import RunConfig

config = RunConfig(
    max_retries=3,          # Retry failed LLM calls
    max_wait=60,            # Max wait time for retries
    max_workers=8,          # Parallel workers
    timeout=120,            # Timeout per evaluation
)
```

### Caching for Cost Reduction

```python
# Enable caching to avoid redundant LLM calls
from ragas.cache import DiskCache

cache = DiskCache(cache_dir=".ragas_cache")

# Results are cached and reused for identical inputs
# Dramatically reduces API costs during iterative development
```

### Train & Align Custom Metrics

```python
# Train a metric to align with human judgments
# Useful when you have human-annotated data

from ragas.metrics import train_metric

# Provide human-scored examples
training_data = [
    {"input": "...", "output": "...", "human_score": 4},
    {"input": "...", "output": "...", "human_score": 2},
    # ...
]

# Train the metric
trained_metric = train_metric(
    base_metric=faithfulness,
    training_data=training_data,
)
```

---

## 17. Production Best Practices & CI/CD

### Best Practices Checklist

| Practice | Description |
|----------|-------------|
| **Version your datasets** | Track test data changes alongside code |
| **Use consistent LLM judges** | Same model version for comparable results |
| **Set thresholds** | Define pass/fail thresholds for each metric |
| **Cache results** | Avoid redundant API calls |
| **Monitor costs** | Track evaluation API costs |
| **Run async** | Use async evaluation for throughput |
| **Separate concerns** | Evaluate retrieval and generation independently |
| **Use CI/CD** | Automate evaluations on every PR |

### CI/CD Integration Example

```python
# eval_pipeline.py - Run in CI/CD
import asyncio
import sys
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, ContextPrecision, ContextRecall

# Thresholds
THRESHOLDS = {
    "faithfulness": 0.8,
    "context_precision": 0.7,
    "context_recall": 0.7,
}

async def run_ci_eval():
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    # Load test dataset
    test_cases = load_test_cases("tests/eval_dataset.csv")

    # Run evaluations
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }

    results = {}
    for name, metric in metrics.items():
        scores = []
        for case in test_cases:
            result = await metric.ascore(**case)
            scores.append(result.value)
        results[name] = sum(scores) / len(scores)

    # Check thresholds
    all_passed = True
    for metric_name, avg_score in results.items():
        threshold = THRESHOLDS[metric_name]
        status = "PASS" if avg_score >= threshold else "FAIL"
        print(f"{metric_name}: {avg_score:.3f} (threshold: {threshold}) [{status}]")
        if avg_score < threshold:
            all_passed = False

    return 0 if all_passed else 1

exit_code = asyncio.run(run_ci_eval())
sys.exit(exit_code)
```

### GitHub Actions Integration

```yaml
# .github/workflows/rag-eval.yml
name: RAG Evaluation
on:
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install ragas openai
      - run: python eval_pipeline.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

### Cost Analysis

```python
# Track evaluation costs
from ragas.cost import TokenCounter

counter = TokenCounter()

# Wrap your LLM with cost tracking
tracked_llm = counter.wrap(llm)

# Run evaluation
result = await scorer.ascore(llm=tracked_llm, ...)

# Check costs
print(f"Total tokens: {counter.total_tokens}")
print(f"Estimated cost: ${counter.estimated_cost:.4f}")
```

### Production Monitoring

```python
# Continuous evaluation in production
import schedule
import asyncio

async def daily_eval():
    """Run daily evaluation on production logs."""
    # Sample recent production queries
    production_samples = sample_production_logs(n=50)

    # Evaluate
    results = await evaluate_samples(production_samples)

    # Alert if below thresholds
    for metric, score in results.items():
        if score < THRESHOLDS[metric]:
            send_alert(f"RAG quality degradation: {metric} = {score:.3f}")

    # Log results
    log_evaluation_results(results)

schedule.every().day.at("02:00").do(
    lambda: asyncio.run(daily_eval())
)
```

---

## 18. Syntax Cheat Sheet

### Quick Reference

```python
# ============================================================
# SETUP
# ============================================================
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

# ============================================================
# METRICS (New Collections API)
# ============================================================
from ragas.metrics.collections import (
    Faithfulness,           # Response grounded in context?
    ContextPrecision,       # Relevant docs ranked higher?
    ContextRecall,          # All relevant docs retrieved?
    ContextUtilization,     # Like precision but no reference needed
    AnswerRelevancy,        # Response addresses the question?
)

scorer = Faithfulness(llm=llm)

# Async scoring
result = await scorer.ascore(
    user_input="question",
    response="answer",
    retrieved_contexts=["doc1", "doc2"]
)
print(result.value)  # 0.0 - 1.0

# Sync scoring
result = scorer.score(
    user_input="question",
    response="answer",
    retrieved_contexts=["doc1", "doc2"]
)

# ============================================================
# METRICS (Legacy API - Deprecated, use above instead)
# ============================================================
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness as LegacyFaithfulness

sample = SingleTurnSample(
    user_input="question",
    response="answer",
    retrieved_contexts=["doc1"]
)
scorer = LegacyFaithfulness(llm=llm)
score = await scorer.single_turn_ascore(sample)

# ============================================================
# CUSTOM METRICS
# ============================================================
from ragas.metrics import DiscreteMetric, AspectCritic, SimpleCriteriaScore

# Discrete (categorical output)
metric = DiscreteMetric(
    name="quality",
    prompt="Rate: {response}. Return 'good' or 'bad'.",
    allowed_values=["good", "bad"],
)

# Binary (pass/fail)
critic = AspectCritic(
    name="safety",
    definition="Is the response safe and appropriate?",
    llm=llm,
)

# Numeric (1-5)
scorer = SimpleCriteriaScore(
    name="detail",
    definition="How detailed is the response?",
    llm=llm,
)

# ============================================================
# DATASETS
# ============================================================
from ragas import Dataset

dataset = Dataset(name="my_data", backend="local/csv", root_dir=".")
dataset.append({"question": "...", "grading_notes": "..."})
dataset.save()

# ============================================================
# EVALUATION
# ============================================================
from ragas import evaluate, aevaluate

# Sync
results = evaluate(samples=samples, metrics=[metric1, metric2])

# Async
results = await aevaluate(samples=samples, metrics=[metric1, metric2])

# ============================================================
# NON-LLM METRICS (No API key needed)
# ============================================================
from ragas.metrics import (
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
    IDBasedContextPrecision,
    IDBasedContextRecall,
    BleuScore,
    RougeScore,
    ExactMatch,
    StringPresence,
)
```

---

## 19. Complete Code Examples (5 Full Scripts)

### Example 1: Basic RAG Evaluation Pipeline

```python
"""
Example 1: Evaluate a RAG pipeline with core metrics.
Evaluates faithfulness, context precision, context recall, and answer relevancy.
"""
import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
)

async def evaluate_rag_pipeline():
    # Setup
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

    # Create scorers
    scorers = {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
    }

    # Test data (simulating RAG pipeline output)
    test_cases = [
        {
            "user_input": "What is Python used for?",
            "response": "Python is used for web development, data science, AI, and automation.",
            "reference": "Python is commonly used for web development, data analysis, artificial intelligence, and task automation.",
            "retrieved_contexts": [
                "Python is a versatile language used in web development with frameworks like Django and Flask.",
                "Python is widely used in data science and machine learning with libraries like pandas, scikit-learn, and TensorFlow.",
                "Python can automate repetitive tasks using scripts."
            ],
        },
        {
            "user_input": "Who created Python?",
            "response": "Python was created by Guido van Rossum in 1991.",
            "reference": "Python was created by Guido van Rossum and first released in 1991.",
            "retrieved_contexts": [
                "Python was conceived in the late 1980s by Guido van Rossum at CWI in the Netherlands.",
                "The first version of Python (0.9.0) was released in February 1991."
            ],
        },
    ]

    # Evaluate each test case
    print("=" * 70)
    print("RAG PIPELINE EVALUATION RESULTS")
    print("=" * 70)

    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {case['user_input']} ---")

        for metric_name, scorer in scorers.items():
            kwargs = {
                "user_input": case["user_input"],
                "response": case["response"],
                "retrieved_contexts": case["retrieved_contexts"],
            }
            # Add reference for metrics that need it
            if metric_name in ("context_precision", "context_recall"):
                kwargs["reference"] = case["reference"]

            result = await scorer.ascore(**kwargs)
            status = "PASS" if result.value >= 0.7 else "FAIL"
            print(f"  {metric_name}: {result.value:.4f} [{status}]")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    asyncio.run(evaluate_rag_pipeline())
```

### Example 2: Context Quality Comparison

```python
"""
Example 2: Compare different retriever configurations.
Tests how different chunking strategies affect context quality.
"""
import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, ContextUtilization

async def compare_retrievers():
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    precision = ContextPrecision(llm=llm)
    utilization = ContextUtilization(llm=llm)

    question = "What are the benefits of exercise?"
    reference = "Regular exercise improves cardiovascular health, mental well-being, and helps maintain a healthy weight."

    # Retriever A: Good ranking (relevant docs first)
    contexts_a = [
        "Regular exercise strengthens the heart and improves cardiovascular health.",
        "Exercise releases endorphins which improve mental well-being and reduce stress.",
        "Physical activity helps maintain a healthy weight through calorie burning.",
        "The weather forecast for tomorrow shows partly cloudy skies.",  # Noise
    ]

    # Retriever B: Poor ranking (noise first)
    contexts_b = [
        "The weather forecast for tomorrow shows partly cloudy skies.",  # Noise
        "Coffee is one of the most popular beverages worldwide.",        # Noise
        "Regular exercise strengthens the heart and improves cardiovascular health.",
        "Exercise releases endorphins which improve mental well-being.",
    ]

    # Evaluate both
    print("RETRIEVER COMPARISON")
    print("=" * 50)

    # Context Precision (with reference)
    score_a = await precision.ascore(
        user_input=question, reference=reference, retrieved_contexts=contexts_a
    )
    score_b = await precision.ascore(
        user_input=question, reference=reference, retrieved_contexts=contexts_b
    )
    print(f"\nContext Precision:")
    print(f"  Retriever A (good ranking): {score_a.value:.4f}")
    print(f"  Retriever B (poor ranking): {score_b.value:.4f}")

    # Context Utilization (without reference)
    response = "Exercise improves heart health and mental well-being."
    util_a = await utilization.ascore(
        user_input=question, response=response, retrieved_contexts=contexts_a
    )
    util_b = await utilization.ascore(
        user_input=question, response=response, retrieved_contexts=contexts_b
    )
    print(f"\nContext Utilization:")
    print(f"  Retriever A (good ranking): {util_a.value:.4f}")
    print(f"  Retriever B (poor ranking): {util_b.value:.4f}")

    winner = "Retriever A" if score_a.value > score_b.value else "Retriever B"
    print(f"\nWinner: {winner}")

if __name__ == "__main__":
    asyncio.run(compare_retrievers())
```

### Example 3: Hallucination Detection with Faithfulness

```python
"""
Example 3: Detect hallucinations in LLM responses using Faithfulness metric.
Demonstrates how to identify unfaithful claims not supported by context.
"""
import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

async def detect_hallucinations():
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)
    scorer = Faithfulness(llm=llm)

    # Test scenarios
    scenarios = [
        {
            "name": "Fully Faithful Response",
            "user_input": "When was the Eiffel Tower built?",
            "response": "The Eiffel Tower was built in 1889 for the World's Fair in Paris.",
            "retrieved_contexts": [
                "The Eiffel Tower was constructed from 1887 to 1889 as the centerpiece "
                "of the 1889 World's Fair in Paris, France."
            ],
            "expected": "high"
        },
        {
            "name": "Partial Hallucination",
            "user_input": "When was the Eiffel Tower built?",
            "response": "The Eiffel Tower was built in 1889 and is 500 meters tall.",
            "retrieved_contexts": [
                "The Eiffel Tower was constructed from 1887 to 1889. "
                "It stands at 330 meters tall."
            ],
            "expected": "medium (500m is wrong)"
        },
        {
            "name": "Full Hallucination",
            "user_input": "When was the Eiffel Tower built?",
            "response": "The Eiffel Tower was built in 1920 by Napoleon Bonaparte as a military watchtower.",
            "retrieved_contexts": [
                "The Eiffel Tower was constructed from 1887 to 1889 by Gustave Eiffel "
                "as the entrance arch for the World's Fair."
            ],
            "expected": "low (multiple false claims)"
        },
    ]

    print("HALLUCINATION DETECTION RESULTS")
    print("=" * 60)

    for scenario in scenarios:
        result = await scorer.ascore(
            user_input=scenario["user_input"],
            response=scenario["response"],
            retrieved_contexts=scenario["retrieved_contexts"]
        )

        # Classify
        if result.value >= 0.8:
            classification = "FAITHFUL"
        elif result.value >= 0.5:
            classification = "PARTIAL HALLUCINATION"
        else:
            classification = "HALLUCINATION DETECTED"

        print(f"\n{scenario['name']}:")
        print(f"  Response: {scenario['response']}")
        print(f"  Faithfulness: {result.value:.4f}")
        print(f"  Classification: {classification}")
        print(f"  Expected: {scenario['expected']}")

if __name__ == "__main__":
    asyncio.run(detect_hallucinations())
```

### Example 4: Custom Metric with Aspect Critic

```python
"""
Example 4: Create custom evaluation metrics using Aspect Critic and
Simple Criteria Scoring for domain-specific evaluation.
"""
import asyncio
from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import AspectCritic, SimpleCriteriaScore
from ragas.metrics import DiscreteMetric

async def custom_metrics_evaluation():
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    # Define custom metrics for a customer support chatbot

    # 1. Politeness Check (Binary: pass/fail)
    politeness = AspectCritic(
        name="politeness",
        definition=(
            "The response should be polite, professional, and empathetic. "
            "It should not contain rude, dismissive, or condescending language."
        ),
        llm=llm,
    )

    # 2. Completeness Score (Numeric: 1-5)
    completeness = SimpleCriteriaScore(
        name="completeness",
        definition=(
            "How completely does the response address all aspects of the "
            "customer's question? Consider whether all sub-questions are "
            "answered and whether actionable next steps are provided."
        ),
        llm=llm,
    )

    # 3. Tone Classification (Discrete)
    tone_metric = DiscreteMetric(
        name="tone",
        prompt=(
            "Classify the tone of this customer support response:\n"
            "Response: {response}\n"
            "Return one of: 'professional', 'casual', 'empathetic', 'robotic'"
        ),
        allowed_values=["professional", "casual", "empathetic", "robotic"],
    )

    # Test samples
    samples = [
        SingleTurnSample(
            user_input="I've been waiting 3 weeks for my order and I'm very frustrated!",
            response=(
                "I completely understand your frustration, and I sincerely apologize "
                "for the delay. Let me look into your order right away. I'll check "
                "the tracking information and ensure we expedite the delivery. "
                "You should receive an update within 24 hours."
            ),
        ),
        SingleTurnSample(
            user_input="Can you help me reset my password?",
            response="Go to settings. Click reset. Done.",
        ),
    ]

    print("CUSTOM METRICS EVALUATION - Customer Support")
    print("=" * 60)

    for i, sample in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Customer: {sample.user_input}")
        print(f"Response: {sample.response[:80]}...")

        # Evaluate politeness
        polite_score = await politeness.single_turn_ascore(sample)
        print(f"  Politeness: {'PASS' if polite_score == 1 else 'FAIL'}")

        # Evaluate completeness
        complete_score = await completeness.single_turn_ascore(sample)
        print(f"  Completeness: {complete_score}/5")

        # Evaluate tone
        tone_result = tone_metric.score(
            llm=llm,
            response=sample.response,
        )
        print(f"  Tone: {tone_result.value}")

if __name__ == "__main__":
    asyncio.run(custom_metrics_evaluation())
```

### Example 5: End-to-End Evaluation with Report Generation

```python
"""
Example 5: Complete end-to-end RAG evaluation with CSV report generation.
Evaluates multiple samples across all core metrics and generates a report.
"""
import asyncio
import csv
from datetime import datetime
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
)

# Thresholds for pass/fail
THRESHOLDS = {
    "faithfulness": 0.8,
    "context_precision": 0.7,
    "context_recall": 0.7,
    "answer_relevancy": 0.7,
}

async def full_evaluation():
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

    # Metrics
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
    }

    # Test dataset
    test_data = [
        {
            "id": 1,
            "user_input": "What is machine learning?",
            "response": "Machine learning is a subset of AI that enables systems to learn from data.",
            "reference": "Machine learning is a branch of artificial intelligence that allows systems to learn and improve from experience without being explicitly programmed.",
            "retrieved_contexts": [
                "Machine learning (ML) is a branch of artificial intelligence (AI) that focuses on building systems that learn from data.",
                "ML algorithms can identify patterns in large datasets and make predictions."
            ]
        },
        {
            "id": 2,
            "user_input": "What is deep learning?",
            "response": "Deep learning uses neural networks with many layers to learn complex patterns.",
            "reference": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers (deep neural networks) to model complex patterns in data.",
            "retrieved_contexts": [
                "Deep learning is a type of machine learning based on artificial neural networks.",
                "Deep neural networks have multiple hidden layers between input and output."
            ]
        },
        {
            "id": 3,
            "user_input": "What is natural language processing?",
            "response": "NLP is a field that helps computers understand human language. It is used in chatbots and translation.",
            "reference": "Natural Language Processing (NLP) is a field of AI that enables computers to understand, interpret, and generate human language.",
            "retrieved_contexts": [
                "NLP combines computational linguistics with statistical and deep learning models.",
                "Applications of NLP include machine translation, chatbots, and sentiment analysis.",
                "The weather is sunny today."  # Noise
            ]
        },
    ]

    # Run evaluation
    results = []
    for case in test_data:
        row = {"id": case["id"], "question": case["user_input"]}

        for metric_name, scorer in metrics.items():
            kwargs = {
                "user_input": case["user_input"],
                "response": case["response"],
                "retrieved_contexts": case["retrieved_contexts"],
            }
            if metric_name in ("context_precision", "context_recall"):
                kwargs["reference"] = case["reference"]

            result = await scorer.ascore(**kwargs)
            row[metric_name] = round(result.value, 4)
            row[f"{metric_name}_status"] = (
                "PASS" if result.value >= THRESHOLDS[metric_name] else "FAIL"
            )

        results.append(row)

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"ragas_eval_report_{timestamp}.csv"

    with open(report_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    print("\n" + "=" * 70)
    print(f"RAGAS EVALUATION REPORT - {timestamp}")
    print("=" * 70)

    for row in results:
        print(f"\nQ{row['id']}: {row['question']}")
        for metric_name in metrics:
            print(f"  {metric_name}: {row[metric_name]:.4f} [{row[f'{metric_name}_status']}]")

    # Overall averages
    print("\n" + "-" * 70)
    print("OVERALL AVERAGES:")
    for metric_name in metrics:
        avg = sum(r[metric_name] for r in results) / len(results)
        status = "PASS" if avg >= THRESHOLDS[metric_name] else "FAIL"
        print(f"  {metric_name}: {avg:.4f} (threshold: {THRESHOLDS[metric_name]}) [{status}]")

    total_pass = sum(
        1 for r in results
        for m in metrics
        if r[f"{m}_status"] == "PASS"
    )
    total_tests = len(results) * len(metrics)
    print(f"\nTotal: {total_pass}/{total_tests} passed")
    print(f"Report saved to: {report_file}")

if __name__ == "__main__":
    asyncio.run(full_evaluation())
```

---

## 20. Interview Q&A (30 Questions)

### Fundamentals (Q1-Q10)

**Q1: What is RAGAs and what problem does it solve?**

**A:** RAGAs (Retrieval Augmented Generation Assessment) is an open-source Python framework for evaluating LLM applications, specifically RAG pipelines. It solves the problem of objectively measuring RAG quality by providing metrics for both retrieval (Context Precision, Context Recall) and generation (Faithfulness, Answer Relevancy) components. Without RAGAs, teams rely on subjective human evaluation, which is expensive, slow, and inconsistent.

---

**Q2: What are the four core RAG metrics in RAGAs?**

**A:** The four core metrics are:
1. **Context Precision** - Are relevant documents ranked higher than irrelevant ones?
2. **Context Recall** - Were all relevant documents retrieved?
3. **Faithfulness** - Is the response factually grounded in the retrieved context?
4. **Answer Relevancy** - Does the response actually address the user's question?

---

**Q3: How does Faithfulness differ from Answer Relevancy?**

**A:** **Faithfulness** checks whether the response's claims are supported by the retrieved context (grounding). **Answer Relevancy** checks whether the response addresses the user's question (relevance). A response can be faithful (all claims are grounded) but irrelevant (doesn't answer the question), or relevant (addresses the question) but unfaithful (contains hallucinated facts).

---

**Q4: Explain how Context Precision is calculated.**

**A:** Context Precision uses a weighted precision formula: $\text{Precision@K} = \frac{\sum_{k=1}^{K} (\text{Precision@k} \times v_k)}{\text{Total relevant items}}$. It evaluates each retrieved chunk's relevance (using an LLM or reference comparison), then weights by position. Relevant documents at the top of the ranking get higher weight. This measures the retriever's ability to rank relevant documents first.

---

**Q5: What is the difference between Context Precision and Context Utilization?**

**A:** Both measure retrieval ranking quality, but differ in what they compare against:
- **Context Precision** compares retrieved contexts against a **reference answer** (requires ground truth)
- **Context Utilization** compares retrieved contexts against the **generated response** (no reference needed)

Use Context Utilization when you don't have ground truth answers, which is common in production.

---

**Q6: How does RAGAs use LLMs for evaluation (LLM-as-a-Judge)?**

**A:** RAGAs uses a judge LLM (e.g., GPT-4o-mini) to evaluate other LLM outputs. The judge LLM is given structured prompts to: (1) break responses into claims, (2) verify claims against context, (3) generate questions from answers, or (4) assess relevance. This eliminates the need for human annotators while providing consistent, scalable evaluation.

---

**Q7: What is the collections-based API and why was it introduced?**

**A:** The collections-based API (e.g., `from ragas.metrics.collections import Faithfulness`) is the new recommended API in Ragas 0.4+. It replaces the legacy `SingleTurnSample` + `single_turn_ascore()` pattern with keyword arguments: `scorer.ascore(user_input=..., response=...)`. Benefits: simpler syntax, supports both async and sync (`.score()`), better error messages, cleaner imports.

---

**Q8: What non-LLM metrics does RAGAs provide?**

**A:** RAGAs provides several metrics that don't require LLM API calls:
- **NonLLMContextPrecisionWithReference** - Uses Levenshtein distance
- **NonLLMContextRecall** - String comparison-based recall
- **IDBasedContextPrecision/Recall** - Compare document IDs
- **BLEU, ROUGE, CHRF scores** - Traditional NLP metrics
- **ExactMatch, StringPresence** - Direct text matching
- **SemanticSimilarity** - Embedding-based (needs embeddings, not LLM)

---

**Q9: How does RAGAs handle async evaluation?**

**A:** RAGAs is async-first. All LLM-based metrics support `await scorer.ascore()` for async evaluation and `scorer.score()` for sync. Batch evaluation uses `await aevaluate()` (async) or `evaluate()` (sync). The `RunConfig` class controls parallelism with `max_workers`, `timeout`, and `max_retries` parameters.

---

**Q10: What is a SingleTurnSample and what fields does it contain?**

**A:** A `SingleTurnSample` represents one evaluation test case with these fields:
- `user_input` (required) - The user's question
- `response` - The LLM's generated answer
- `reference` - The ground truth answer
- `retrieved_contexts` - List of retrieved document texts
- `reference_contexts` - Ideal contexts for comparison
- `retrieved_context_ids` / `reference_context_ids` - Document IDs

---

### Intermediate (Q11-Q20)

**Q11: How do you detect hallucinations using RAGAs?**

**A:** Use the **Faithfulness** metric. It works in three steps:
1. Break the LLM response into individual claims/statements
2. Check each claim against the retrieved contexts to see if it's supported
3. Calculate the ratio of supported claims to total claims

A score below 1.0 indicates hallucination. The **FaithfulnesswithHHEM** variant uses Vectara's HHEM-2.1-Open model (a T5 classifier) for the verification step, which is faster and cheaper than LLM-based verification.

---

**Q12: How does test data generation work in RAGAs?**

**A:** RAGAs generates synthetic test data through a Knowledge Graph pipeline:
1. **Document Loading** - Load your source documents
2. **KG Building** - Extract entities and relationships into a knowledge graph
3. **Scenario Generation** - Create diverse query scenarios (single-hop, multi-hop, reasoning)
4. **Test Sample Creation** - Generate questions, contexts, and reference answers

This produces realistic, diverse test datasets without manual annotation.

---

**Q13: What is the difference between evaluate() and aevaluate()?**

**A:** Both run batch evaluations across multiple samples:
- `evaluate()` - Synchronous, blocks until all evaluations complete
- `aevaluate()` - Asynchronous, returns a coroutine that can be awaited

Use `aevaluate()` when running in async contexts (like FastAPI endpoints) or when you want to run evaluations concurrently with other tasks.

---

**Q14: How do you evaluate an AI agent with RAGAs?**

**A:** RAGAs provides agent-specific metrics:
- **TopicAdherenceScore** - Does the agent stay on-topic?
- **ToolCallAccuracy** - Are tool/function calls correct?
- **ToolCallF1** - Precision + recall of tool calls
- **AgentGoalAccuracy** - Does the agent achieve the user's goal?

RAGAs also supports generating agent test data with `AgentTestsetGenerator` that creates scenarios based on available tools.

---

**Q15: How do you create custom metrics in RAGAs?**

**A:** Three approaches:
1. **AspectCritic** - Binary pass/fail with natural language definition
2. **SimpleCriteriaScore** - Numeric 1-5 with natural language criteria
3. **RubricsCriteriaScore** - Numeric with detailed rubrics per score level
4. **DiscreteMetric** - Categorical output with custom prompt and allowed values

All accept a `definition` or `prompt` string and use an LLM to evaluate.

---

**Q16: How does Answer Relevancy use embeddings?**

**A:** Answer Relevancy works by:
1. Using an LLM to generate N questions (default 3) that the response would answer
2. Computing cosine similarity between the embedding of each generated question and the original user input
3. Averaging the similarities

If the response is relevant, the reverse-engineered questions should be semantically similar to the original question.

---

**Q17: What is Context Recall and why is a reference always needed?**

**A:** Context Recall measures how many relevant pieces of information were successfully retrieved. Since it measures "what was missed," it needs a reference to compare against. The reference is broken into claims, and each claim is checked against retrieved contexts. Formula: $\frac{\text{claims in reference supported by context}}{\text{total claims in reference}}$.

---

**Q18: How do you integrate RAGAs with LangChain?**

**A:** Two approaches:
1. **LLM Wrapper**: `from ragas.llms import LangchainLLMWrapper` - wraps a LangChain LLM for use in RAGAs metrics
2. **Document Loading**: Use LangChain document loaders with RAGAs test generation: `generator.generate_with_langchain_docs(documents)`
3. **Pipeline Evaluation**: Extract contexts and responses from your LangChain chain, then evaluate with RAGAs metrics

---

**Q19: How do you reduce evaluation costs in production?**

**A:** Several strategies:
1. **Caching** - Use `DiskCache` to avoid redundant LLM calls for identical inputs
2. **Cheaper models** - Use GPT-4o-mini instead of GPT-4o for evaluation
3. **Non-LLM metrics** - Use BLEU, ROUGE, ID-based metrics where possible
4. **HHEM model** - Use FaithfulnesswithHHEM for local faithfulness evaluation
5. **Sampling** - Evaluate a representative sample, not every query
6. **Batch processing** - Use `evaluate()` for efficient batch processing

---

**Q20: What is the DiscreteMetric and how does it differ from other metrics?**

**A:** `DiscreteMetric` is a flexible metric type that returns categorical (non-numeric) values. You define a custom prompt template and a set of `allowed_values`. The LLM evaluates the input and returns one of the allowed values. Unlike `AspectCritic` (binary only) or `SimpleCriteriaScore` (numeric only), DiscreteMetric supports any categorical output (e.g., "positive/neutral/negative", "pass/fail/partial").

---

### Advanced (Q21-Q30)

**Q21: How would you design a RAG evaluation strategy for a production system?**

**A:** A comprehensive strategy includes:
1. **Offline Evaluation** - Run full metric suite on a curated test dataset before deployment
2. **CI/CD Integration** - Automated evaluation on every PR with threshold gates
3. **A/B Testing** - Compare retriever/generator variants with RAGAs metrics
4. **Production Monitoring** - Sample and evaluate production queries daily
5. **Regression Detection** - Track metric trends, alert on degradation
6. **Cost Tracking** - Monitor evaluation API costs
7. **Human Alignment** - Periodically validate LLM-judge accuracy against human ratings

---

**Q22: How does RAGAs handle multi-turn conversations?**

**A:** RAGAs supports multi-turn evaluation through `MultiTurnSample`, which takes a conversation history as input. The conversation is represented as a list of role/content pairs. Metrics can evaluate the entire conversation or the last response in context. RAGAs also supports evaluating AI workflows where multiple steps/agents contribute to the final response.

---

**Q23: What is the relationship between Faithfulness and Factual Correctness?**

**A:** Both measure response accuracy but from different angles:
- **Faithfulness** measures if the response is grounded in the **retrieved context** (are claims supported by what was retrieved?)
- **Factual Correctness** measures if the response matches the **reference answer** (are the facts correct compared to ground truth?)

A response can be faithful to context but factually incorrect if the context itself contains errors. Factual Correctness catches this by comparing against verified ground truth.

---

**Q24: How do you adapt RAGAs metrics for non-English languages?**

**A:** RAGAs supports multilingual evaluation through:
1. **Multilingual LLM judges** - Use models like GPT-4o that support many languages
2. **Prompt adaptation** - Modify metric prompts to include language-specific instructions
3. **Language-adapted test generation** - Generate test data in the target language
4. **Non-LLM metrics** - BLEU, ROUGE work across languages (with appropriate tokenization)

The `adapt_to_language()` function helps modify prompts for specific languages.

---

**Q25: How do you optimize RAGAs evaluation for large-scale datasets?**

**A:** Optimization techniques:
1. **RunConfig** - Set `max_workers=16` for parallel evaluation
2. **Async evaluation** - Use `aevaluate()` for concurrent processing
3. **Caching** - Enable `DiskCache` to skip repeated evaluations
4. **Tiered metrics** - Run cheap metrics (BLEU, ID-based) first, expensive metrics (LLM-based) only on failures
5. **Sampling** - Evaluate random subsets with confidence intervals
6. **Batching** - Process multiple samples per LLM call where possible

---

**Q26: What is Noise Sensitivity and when should you use it?**

**A:** Noise Sensitivity measures how much irrelevant information in retrieved contexts degrades the LLM's response quality. Use it when:
- Your retriever sometimes returns irrelevant documents
- You want to assess the generator's robustness to noisy input
- You're comparing generators on the same retrieval results
- You need to decide between strict vs. permissive retrieval thresholds

High noise sensitivity = the model is easily confused by irrelevant context.

---

**Q27: How do you use RAGAs to compare different LLMs for your RAG pipeline?**

**A:** Benchmark approach:
1. Fix the retriever and test dataset
2. Run each LLM through the same RAG pipeline
3. Evaluate all responses with the same RAGAs metrics and judge model
4. Compare scores across: Faithfulness (grounding), Answer Relevancy (usefulness), Factual Correctness (accuracy)
5. Consider cost per query and latency alongside quality metrics
6. Use the `@experiment()` decorator to track results across runs

---

**Q28: What are ID-based metrics and when are they useful?**

**A:** ID-based metrics (`IDBasedContextPrecision`, `IDBasedContextRecall`) compare document IDs rather than content. Useful when:
- You have a document ID system (database primary keys, file IDs)
- You want fast evaluation without content comparison
- You're evaluating retriever recall/precision at the document level
- You have reference document sets for each query

Formula: $\text{ID Precision} = \frac{|\text{retrieved IDs} \cap \text{reference IDs}|}{|\text{retrieved IDs}|}$

---

**Q29: How do you align RAGAs metrics with human judgment?**

**A:** Use RAGAs' train and align features:
1. Collect human-scored evaluation samples
2. Use `train_metric()` to fine-tune a metric against human scores
3. Measure correlation (Spearman/Kendall) between metric scores and human ratings
4. Adjust prompt wording to improve alignment
5. Use `RubricsCriteriaScore` with rubrics matching your human annotation guidelines
6. Consider using `align-llm-as-judge` workflow for systematic calibration

---

**Q30: Compare RAGAs and DeepEval - when would you choose each?**

**A:**

| Aspect | RAGAs | DeepEval |
|--------|-------|----------|
| **Best for** | RAG-specific evaluation | General LLM evaluation |
| **Test Data Generation** | Built-in KG-based synthetic generation | Dataset management, no generation |
| **Metrics Focus** | Retrieval + Generation quality | Hallucination, Bias, Toxicity, G-Eval |
| **API Style** | Async-first, collections-based | PyTest assertions (assert_test) |
| **Framework Integration** | LangChain, LlamaIndex, Haystack, etc. | Framework agnostic |
| **CI/CD** | Custom scripts, GitHub Actions | Native pytest integration |
| **Dashboard** | CSV exports, LangSmith integration | Confident AI platform |
| **Cost Efficiency** | HHEM model, caching, non-LLM metrics | Mainly LLM-based |

**Choose RAGAs when:** Evaluating RAG pipelines, need synthetic test generation, need retrieval-specific metrics, integrating with LangChain/LlamaIndex.

**Choose DeepEval when:** General LLM evaluation, need pytest integration, evaluating safety/bias/toxicity, want a managed dashboard.

**Use both when:** You need comprehensive evaluation - RAGAs for retrieval quality, DeepEval for response quality and safety.

---

## 21. Troubleshooting & Common Errors

### Common Issues

| Issue | Solution |
|-------|----------|
| `OpenAI API key not found` | Set `OPENAI_API_KEY` environment variable |
| `AsyncIO event loop error` | Use `asyncio.run()` or run in Jupyter with `await` |
| `Module not found: ragas` | `pip install ragas` in your virtual environment |
| `LangChain version conflict` | Pin versions: `pip install "langchain-core>=0.2,<0.3"` |
| `Metric returns None` | Ensure all required fields are provided in the sample |
| `Rate limit exceeded` | Use `RunConfig(max_retries=3, max_wait=60)` |
| `Out of memory (HHEM)` | Reduce `batch_size` or use CPU: `device="cpu"` |
| `Score is NaN` | Check for empty contexts or responses |
| `rapidfuzz not found` | `pip install rapidfuzz` (needed for NonLLM metrics) |

### AsyncIO in Different Environments

```python
# Script
import asyncio
asyncio.run(main())

# Jupyter Notebook
await main()

# FastAPI endpoint
@app.post("/evaluate")
async def evaluate_endpoint():
    result = await scorer.ascore(...)
    return {"score": result.value}

# Sync fallback (any environment)
result = scorer.score(...)  # No async needed
```

### Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific ragas module
logging.getLogger("ragas").setLevel(logging.DEBUG)
```

### Version Migration

```python
# If migrating from v0.1 to v0.2+, key changes:
# - evaluate() API changed
# - Metric names updated
# - SingleTurnSample introduced

# If migrating from v0.3 to v0.4+:
# - Collections-based API introduced
# - Legacy SingleTurnSample pattern deprecated
# - DiscreteMetric added
# - llm_factory() for LLM creation
```

---

## 22. RAGAs vs DeepEval Comparison

### Side-by-Side Feature Comparison

| Feature | RAGAs | DeepEval |
|---------|-------|----------|
| **Primary Focus** | RAG pipeline evaluation | General LLM evaluation |
| **Installation** | `pip install ragas` | `pip install deepeval` |
| **Retrieval Metrics** | Context Precision, Recall, Entities Recall, Noise Sensitivity | Contextual Precision, Recall (fewer variants) |
| **Generation Metrics** | Faithfulness, Answer Relevancy | Hallucination, Answer Relevancy, Faithfulness |
| **Safety Metrics** | Via Aspect Critic (custom) | Built-in Bias, Toxicity metrics |
| **Custom Metrics** | AspectCritic, DiscreteMetric, RubricsScore | G-Eval, custom metrics |
| **Test Data Generation** | Built-in KG-based synthetic generation | Synthesizer (simpler) |
| **API Style** | Async-first, keyword arguments | PyTest assertions |
| **Test Runner** | Custom scripts, experiment decorator | Native pytest integration |
| **LLM Support** | OpenAI, Anthropic, Gemini, Ollama, Bedrock | OpenAI, Anthropic, Azure, custom |
| **Non-LLM Metrics** | BLEU, ROUGE, ID-based, String metrics | Limited |
| **Cost Features** | Caching, HHEM model, cost analysis | Token tracking |
| **Dashboard** | CSV exports, LangSmith/Arize integration | Confident AI platform |
| **Multi-turn** | MultiTurnSample support | ConversationalTestCase |
| **Agent Eval** | Topic Adherence, Tool Call metrics | Tool correctness |
| **Community** | 13.8k+ GitHub stars | 5k+ GitHub stars |

### When to Use Which

```
RAGAs is ideal when:
├── You need to evaluate RETRIEVAL quality specifically
├── You want synthetic test data generation from documents
├── You're using LangChain, LlamaIndex, or Haystack
├── You need non-LLM metrics for fast/cheap evaluation
├── You want async-first evaluation for high throughput
└── You need to evaluate AI agents and workflows

DeepEval is ideal when:
├── You want native pytest integration
├── You need safety metrics (bias, toxicity) out of the box
├── You want a managed dashboard (Confident AI)
├── You prefer assertion-based testing style
├── You need G-Eval for flexible LLM-judged evaluation
└── You want simpler API for general LLM evaluation
```

---

## Quick Links & Resources

| Resource | URL |
|----------|-----|
| **Official Docs** | https://docs.ragas.io/en/stable/ |
| **GitHub** | https://github.com/vibrantlabsai/ragas |
| **PyPI** | https://pypi.org/project/ragas/ |
| **Discord** | Community `#questions` channel |
| **Office Hours** | cal.com/team/vibrantlabs/office-hours |

---


