# DeepEval — The Ultimate  Knowledge Base

> Open-source LLM evaluation framework by Confident AI | Apache 2.0 Licensed  
> Latest docs: https://deepeval.com/docs | GitHub: https://github.com/confident-ai/deepeval  
> 20 million+ daily evaluations worldwide | 50+ built-in metrics

---

## Table of Contents

1. [What is DeepEval?](#1-what-is-deepeval)
2. [Why DeepEval? — Advantages & Comparisons](#2-why-deepeval--advantages--comparisons)
3. [Design Philosophy](#3-design-philosophy)
4. [Installation & Environment Setup](#4-installation--environment-setup)
5. [Core Concepts & Architecture](#5-core-concepts--architecture)
6. [Test Cases — The Foundation](#6-test-cases--the-foundation)
7. [Metrics — Complete Reference](#7-metrics--complete-reference)
8. [GEval — Custom Metrics in Plain English](#8-geval--custom-metrics-in-plain-english)
9. [DAG — Decision-Tree Based Metrics](#9-dag--decision-tree-based-metrics)
10. [RAG Evaluation — End-to-End Pipeline](#10-rag-evaluation--end-to-end-pipeline)
11. [Agent Evaluation — Reasoning, Action, Execution](#11-agent-evaluation--reasoning-action-execution)
12. [Multi-Turn / Chatbot Evaluation](#12-multi-turn--chatbot-evaluation)
13. [Hallucination Detection](#13-hallucination-detection)
14. [Safety Metrics — Bias, Toxicity, and More](#14-safety-metrics--bias-toxicity-and-more)
15. [Multimodal Evaluation (Images + Text)](#15-multimodal-evaluation-images--text)
16. [Datasets, Goldens & Synthetic Data Generation](#16-datasets-goldens--synthetic-data-generation)
17. [Custom LLM Judges — Any Model](#17-custom-llm-judges--any-model)
18. [LLM Tracing & Observability](#18-llm-tracing--observability)
19. [End-to-End vs Component-Level Evaluation](#19-end-to-end-vs-component-level-evaluation)
20. [Running Evaluations — All Methods](#20-running-evaluations--all-methods)
21. [Flags, Configs & Optimization](#21-flags-configs--optimization)
22. [Prompt Optimization](#22-prompt-optimization)
23. [Benchmarks](#23-benchmarks)
24. [Conversation Simulation](#24-conversation-simulation)
25. [CI/CD Integration & Unit Testing](#25-cicd-integration--unit-testing)
26. [Production Monitoring — Online Evals](#26-production-monitoring--online-evals)
27. [Confident AI Platform](#27-confident-ai-platform)
28. [Integrations — LangChain, LlamaIndex, CrewAI & More](#28-integrations--langchain-llamaindex-crewai--more)
29. [DeepTeam — Red Teaming & Adversarial Testing](#29-deepteam--red-teaming--adversarial-testing)
30. [Customizing Metric Prompts](#30-customizing-metric-prompts)
31. [Building Custom Metrics from Scratch](#31-building-custom-metrics-from-scratch)
32. [Scalability & Performance Optimization](#32-scalability--performance-optimization)
33. [Production-Grade Architecture Patterns](#33-production-grade-architecture-patterns)
34. [Troubleshooting & Common Issues](#34-troubleshooting--common-issues)
35. [Complete Code Examples — All Scenarios](#35-complete-code-examples--all-scenarios)
36. [Syntax Cheat Sheet](#36-syntax-cheat-sheet)
37. [Interview Q&A — 50 Questions](#37-interview-qa--50-questions)
38. [Quick Reference Links](#38-quick-reference-links)

---

## 1. What is DeepEval?

DeepEval is an **open-source LLM evaluation framework** that lets you test, measure, and monitor the quality of LLM application outputs — similar to how PyTest tests code, but purpose-built for AI.

### How It Works:

```
┌──────────────────────────────────────────────────────────────┐
│                    DeepEval Workflow                          │
│                                                              │
│  1. Define TEST CASES (input + LLM output + context)         │
│              ↓                                               │
│  2. Choose METRICS (what to measure — relevancy, etc.)       │
│              ↓                                               │
│  3. Run EVALUATION (deepeval scores using LLM-as-a-Judge)    │
│              ↓                                               │
│  4. Get REPORT (score 0-1 + reason for each test case)       │
│              ↓                                               │
│  5. ITERATE (change hyperparams, re-evaluate, compare)       │
└──────────────────────────────────────────────────────────────┘
```

### Key Facts:
- **20 million+ daily evaluations** processed worldwide
- **50+ built-in metrics** covering RAG, agents, chatbots, safety, multimodal
- Uses **LLM-as-a-Judge** — an LLM scores another LLM's output using research-backed techniques (QAG, GEval, DAG)
- All scores range **0 to 1**, with a configurable pass/fail **threshold** (default 0.5)
- Native **PyTest integration** via `deepeval test run`
- Natively integrates with **Confident AI** platform for cloud reports, monitoring, and collaboration
- Supports **end-to-end** and **component-level** evaluation via LLM tracing
- Works with **any LLM** — OpenAI, Azure, Anthropic, Gemini, Ollama, custom models

### Real-World Analogy:

> Think of it as a **code review for AI outputs**:
> - Your LLM app produces an answer (developer writes code)
> - DeepEval runs metrics on it (senior reviewer checks quality)
> - Each metric gives a score 0-1 (review feedback)
> - You get a report showing what passed and what failed (PR review summary)

---

## 2. Why DeepEval? — Advantages & Comparisons

### Feature Comparison:

| Feature | DeepEval | Manual Testing | RAGAs | LangSmith |
|---------|----------|---------------|-------|-----------|
| LLM-as-a-Judge | Research-backed (QAG, GEval, DAG) | Human only | Basic | |
| 50+ Metrics | | | Limited to RAG | |
| PyTest Integration | Native | | | |
| CI/CD Pipeline | Built-in | | | |
| Agent Evaluation | Full (reasoning + action + execution) | | | |
| Multi-Turn/Chatbot | Native | | | |
| Safety Metrics | Bias, Toxicity, etc. | | | |
| Multimodal (Images) | | | | |
| Custom LLM Judge | Any model | N/A | | |
| Synthetic Data Gen | Synthesizer | | | |
| Conversation Simulation | | | | |
| Production Monitoring | Online Evals | | | |
| Async Execution | Default | | | |
| Score + Reason | Always | | | |
| Component-Level Eval | Via tracing | | | |
| Free & Open Source | Apache 2.0 | N/A | | Proprietary |

### Why DeepEval Metrics Are Superior:
- **Research-backed** LLM-as-a-Judge (GEval paper)
- **Deterministic scores possible** via DAG (Deep Acyclic Graph) metric
- LLMs used only for **confined, specific tasks** — reducing stochasticity
- Always provides **comprehensive reasoning** for scores
- **Non-LLM metrics also available** — BLEU, ROUGE via scorer module

---

## 3. Design Philosophy

DeepEval follows three core principles:

1. **Metrics should be reliable** — Uses research-backed techniques and confines LLMs to specific evaluation sub-tasks rather than asking for a single holistic score
2. **Evaluation should be easy** — Simple API: create test case → pick metric → run evaluation
3. **Flexibility over lock-in** — Works with any LLM, any framework, any deployment target

### Evaluation Scopes:

```
┌─────────────────────────────────────────┐
│         Evaluation Scopes               │
│                                         │
│  End-to-End ──► Black-box system eval   │
│       │                                 │
│  Component-Level ──► Individual parts   │
│       │          via @observe decorator  │
│       │                                 │
│  One-Off ──► Debug single metric call   │
└─────────────────────────────────────────┘
```

---

## 4. Installation & Environment Setup

### Basic Installation:

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Mac/Linux

# Install deepeval
pip install -U deepeval

# Verify installation
python -c "import deepeval; print(deepeval.__version__)"
```

### Set API Keys (Required — LLM-as-a-Judge needs an LLM):

```bash
# ── Option 1: Environment Variable ──

# Windows CMD
set OPENAI_API_KEY=your-api-key-here

# PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY=your-api-key-here

# ── Option 2: .env file (recommended for projects) ──
# Create .env in project root:
OPENAI_API_KEY=your-api-key-here
```

```python
# ── Option 3: In Python code ──
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

```python
# ── Option 4: Jupyter Notebook ──
%env OPENAI_API_KEY=your-api-key-here
# NOTE: Do NOT include quotation marks in notebook env vars
```

### Azure OpenAI Setup:

```bash
set AZURE_OPENAI_API_KEY=your-key
set AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
set OPENAI_API_VERSION=2024-02-15-preview
set AZURE_DEPLOYMENT_NAME=your-deployment
```

### Optional — Confident AI (Cloud Reports & Monitoring):

```bash
# Login to Confident AI for cloud reports
deepeval login

# View reports in browser after running evaluations
deepeval view
```

### Optional — Set Confident AI API Key for Production:

```bash
CONFIDENT_API_KEY="confident_us..."
```

---

## 5. Core Concepts & Architecture

### The 4 Building Blocks:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  TEST CASE   │     │   METRIC     │     │   DATASET    │     │   GOLDEN     │
│              │     │              │     │              │     │              │
│ What to test │──── │ How to score │──── │ Collection   │──── │ Answer key   │
│ (input +     │     │ (relevancy,  │     │ of test      │     │ (expected    │
│  output)     │     │  faithful.)  │     │ cases        │     │  ideal data) │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

| Concept | What It Is | Analogy |
|---------|-----------|---------|
| **Test Case** (`LLMTestCase`) | A single LLM interaction (input → output) | One unit test |
| **Metric** | A scoring rule (0-1 scale) with a reason | An assertion |
| **Dataset** (`EvaluationDataset`) | A collection of test cases or goldens | A test suite |
| **Golden** | The ideal/expected data template for a test case | Expected result / answer key |
| **Threshold** | Minimum score to pass (default 0.5) | Pass mark |
| **Evaluation** | Running metrics against test cases | Running tests |
| **Trace** | Execution path through your LLM app components | Stack trace |
| **Span** | A single component within a trace | A stack frame |

### Metric Categories Overview:

```
┌──────────────────────────────────────────────────────────────────┐
│                       DeepEval Metrics                           │
│                                                                  │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│ │   CUSTOM    │ │   RAG       │ │   AGENTS    │ │  CHATBOTS  │ │
│ │ • GEval     │ │ • Answer    │ │ • Tool      │ │ • Convo    │ │
│ │ • DAG       │ │   Relevancy │ │   Correct.  │ │   Complete │ │
│ │ • BaseMetric│ │ • Faithful. │ │ • Argument  │ │ • Turn     │ │
│ │             │ │ • Ctx Prec. │ │   Correct.  │ │   metrics  │ │
│ │             │ │ • Ctx Recall│ │ • Plan      │ │            │ │
│ │             │ │ • Ctx Relev.│ │   Quality   │ │            │ │
│ │             │ │             │ │ • Plan      │ │            │ │
│ │             │ │             │ │   Adherence │ │            │ │
│ │             │ │             │ │ • Task      │ │            │ │
│ │             │ │             │ │   Completion│ │            │ │
│ │             │ │             │ │ • Step      │ │            │ │
│ │             │ │             │ │   Efficiency│ │            │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
│                                                                  │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│ │   SAFETY    │ │ MULTIMODAL  │ │  HALLUC.    │ │   OTHER    │ │
│ │ • Toxicity  │ │ • Image     │ │ • Halluc.   │ │ • Prompt   │ │
│ │ • Bias      │ │   Coherence │ │ • Faithful. │ │   Align.   │ │
│ │ • Non-Advice│ │ • Image     │ │             │ │ • Summar.  │ │
│ │             │ │   Editing   │ │             │ │ • RAGAS    │ │
│ │             │ │             │ │             │ │ • JSON     │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. Test Cases — The Foundation

### LLMTestCase — 9 Parameters:

```python
from deepeval.test_case import LLMTestCase, ToolCall

test_case = LLMTestCase(
    # ── MANDATORY ──
    input="What is your return policy?",              # User's question/prompt

    # ── OPTIONAL (depends on the metric used) ──
    actual_output="We offer 30-day refunds.",         # LLM's actual response
    expected_output="30-day full refund at no cost.", # Ideal/expected answer
    context=["All customers get 30 day full refund."],# Ground truth facts (static)
    retrieval_context=["Only shoes can be refunded."],# RAG retrieved docs (dynamic)
    tools_called=[ToolCall(name="WebSearch")],        # Agent tools actually used
    expected_tools=[ToolCall(name="WebSearch")],      # Expected agent tools
    token_cost=0.003,                                 # Cost in dollars
    completion_time=1.5,                              # Seconds to complete
)
```

### What Is An LLM "Interaction"?

An interaction is any discrete exchange of information between components of your LLM system. The scope is flexible:

```
      ┌─────────────────┐
      │ Research Agent   │ ← Agent-Level interaction
      ├─────────────────┤
      │ RAG Pipeline     │ ← RAG-Level interaction
      ├─────────────────┤
      │ Retriever        │ ← Component-Level interaction
      ├─────────────────┤
      │ LLM              │ ← LLM-Level interaction
      └─────────────────┘
```

- **Agent-Level**: Entire agent process including RAG + tools
- **RAG Pipeline**: Just the retriever + generator flow
- **Retriever Only**: Testing document retrieval quality
- **LLM Only**: Testing text generation quality

### Which Parameters Does Each Metric Need?

| Metric | input | actual_output | expected_output | context | retrieval_context | tools_called | expected_tools |
|--------|:-----:|:------------:|:---------------:|:-------:|:-----------------:|:------------:|:--------------:|
| AnswerRelevancy | | | | | | | |
| Faithfulness | | | | | | | |
| ContextualPrecision | | | | | | | |
| ContextualRecall | | | | | | | |
| ContextualRelevancy | | | | | | | |
| Hallucination | | | | | | | |
| ToolCorrectness | | | | | | | |
| ArgumentCorrectness | | | | | | | |
| Bias | | | | | | | |
| Toxicity | | | | | | | |
| GEval | Depends on your evaluation_params | | | | | | |

### context vs retrieval_context — Critical Difference:

```
context = ["The ground truth. What SHOULD have been retrieved."]
   → Static, from your evaluation dataset
   → The ideal segment of your knowledge base for a given input
   → Used by: HallucinationMetric, ContextualRecall

retrieval_context = ["What your RAG pipeline ACTUALLY retrieved."]
   → Dynamic, from your running application at runtime
   → What the retriever actually found in the vector store
   → Used by: FaithfulnessMetric, ContextualPrecision, ContextualRelevancy
```

**Rule of Thumb**: `context` = answer key (ideal). `retrieval_context` = actual system output.

### ToolCall Object (for Agent Evaluation):

```python
class ToolCall(BaseModel):
    name: str                                    # Tool name (mandatory)
    description: Optional[str] = None            # Tool's purpose
    reasoning: Optional[str] = None              # Agent's reasoning to use tool
    output: Optional[Any] = None                 # Tool's output
    input_parameters: Optional[Dict[str, Any]] = None  # Input params passed to tool
```

### Labeling Test Cases (for Confident AI):

```python
# Name — unique identifier for search/filter
test_case = LLMTestCase(name="my-external-id", ...)

# Tags — for filtering and categorization
test_case = LLMTestCase(tags=["Topic 1", "Production"], ...)
```

### ConversationalTestCase (Multi-Turn):

```python
from deepeval.test_case import ConversationalTestCase, Turn

convo_test_case = ConversationalTestCase(
    expected_outcome="User understands the visa policy",
    turns=[
        Turn(role="user", content="How long can I stay on F-1?"),
        Turn(
            role="assistant",
            content="You can stay up to 60 days after completing your degree.",
            retrieval_context=["F-1 holders can stay 60 days after degree completion."]
        ),
        Turn(role="user", content="What is OPT?"),
        Turn(
            role="assistant",
            content="OPT allows F-1 students to work in their field for up to 12 months.",
            retrieval_context=["OPT allows F-1 students to work for 12 months."]
        ),
    ]
)
```

---

## 7. Metrics — Complete Reference

### Universal Metric Properties:

Every metric in DeepEval shares these properties:

```python
metric.score           # float: 0.0 to 1.0
metric.reason          # str: explanation for the score
metric.is_successful() # bool: True if score >= threshold
metric.threshold       # float: default 0.5, configurable
metric.strict_mode     # bool: False (if True → binary 0 or 1 score)
metric.async_mode      # bool: True (concurrent internal execution)
metric.verbose_mode    # bool: False (True for debug logging)
```

### A. RAG Metrics (Retrieval-Augmented Generation):

| Metric | What It Measures | Formula | Uses |
|--------|-----------------|---------|------|
| **AnswerRelevancyMetric** | Is the answer relevant to the question? | Relevant Statements / Total Statements | `input`, `actual_output` |
| **FaithfulnessMetric** | Does the answer stick to retrieved facts? | Truthful Claims / Total Claims | `input`, `actual_output`, `retrieval_context` |
| **ContextualPrecisionMetric** | Are relevant contexts ranked higher? | Weighted ranking score | `input`, `expected_output`, `retrieval_context` |
| **ContextualRecallMetric** | Did retrieval find all needed info? | Attributable Sentences / Total | `input`, `expected_output`, `retrieval_context` |
| **ContextualRelevancyMetric** | Is each retrieved chunk useful? | Relevant Sentences / Total | `input`, `retrieval_context` |

**What each RAG metric evaluates in the pipeline:**

```
User Question: "What is the refund policy?"

Retrieved Context: "All customers get 30-day refund."
LLM Answer: "We offer a 30-day full refund."

┌─────────────────────────┐
│  AnswerRelevancy        │── Is "30-day refund" relevant to "refund policy"?
│  (input vs output)      │
├─────────────────────────┤
│  Faithfulness           │── Does "30-day refund" match retrieved context?
│  (output vs retrieval)  │    (No hallucinated facts?)
├─────────────────────────┤
│  ContextualRecall       │── Did retrieval find ALL needed info?
│  (coverage check)       │
├─────────────────────────┤
│  ContextualPrecision    │── Were relevant contexts ranked higher?
│  (ranking quality)      │
├─────────────────────────┤
│  ContextualRelevancy    │── Is each retrieved chunk actually useful?
│  (no junk context)      │
└─────────────────────────┘
```

**What hyperparameters each metric helps tune:**

| Metric | Helps Tune |
|--------|-----------|
| ContextualPrecision | Reranker model quality |
| ContextualRecall | Embedding model, retrieval coverage |
| ContextualRelevancy | Text chunk size, top-K value |
| AnswerRelevancy | Prompt template, LLM choice |
| Faithfulness | LLM choice, temperature |

### B. Agent Metrics:

| Metric | What It Measures | Scope |
|--------|-----------------|-------|
| **ToolCorrectnessMetric** | Did the agent use the right tools? | Component-level (LLM span) |
| **ArgumentCorrectnessMetric** | Did the agent pass correct arguments? | Component-level (LLM span) |
| **PlanQualityMetric** | Is the agent's plan logical & complete? | End-to-end (full trace) |
| **PlanAdherenceMetric** | Did the agent follow its own plan? | End-to-end (full trace) |
| **TaskCompletionMetric** | Did the agent complete the task? | End-to-end (trace-only) |
| **StepEfficiencyMetric** | Was the execution efficient? | End-to-end (trace-only) |

### C. Multi-Turn / Chatbot Metrics:

| Metric | What It Measures |
|--------|-----------------|
| **ConversationCompletenessMetric** | Was the full conversation helpful? |
| **TurnContextualPrecisionMetric** | Multi-turn retrieval precision |
| **TurnContextualRecallMetric** | Multi-turn retrieval recall |
| **TurnContextualRelevancyMetric** | Multi-turn context relevancy |
| **TurnFaithfulnessMetric** | Multi-turn response faithfulness |
| **Conversational GEval** | Custom criteria for conversations |
| **Conversational DAG** | Decision-tree eval for conversations |
| **Arena GEval** | Comparative evaluation between models |

### D. Safety Metrics:

| Metric | What It Measures | Threshold Direction |
|--------|-----------------|-------------------|
| **BiasMetric** | Gender, racial, political bias | **Maximum** (lower is better) |
| **ToxicityMetric** | Personal attacks, mockery, hate, threats | **Maximum** (lower is better) |
| **NonAdviceMetric** | Inappropriate advice giving | Maximum |

### E. Other Metrics:

| Metric | What It Measures |
|--------|-----------------|
| **HallucinationMetric** | Contradiction with known facts |
| **PromptAlignmentMetric** | Does output follow prompt instructions? |
| **SummarizationMetric** | Quality of text summarization |
| **RAGASMetric** | RAGAS-compatible evaluation |
| **JsonCorrectnessMetric** | JSON output validity |

### F. Custom Metrics:

| Framework | Best For | Style |
|-----------|----------|-------|
| **GEval** | Subjective criteria (tone, correctness) | Free-form natural language |
| **DAG** | Objective, multi-step criteria | Decision tree branches |
| **BaseMetric** | 100% self-coded metrics (BLEU, ROUGE) | Python class override |

---

## 8. GEval — Custom Metrics in Plain English

GEval is a state-of-the-art framework that allows you to create custom evaluation metrics using natural language criteria.

### How GEval Works:

```
┌─────────────────────────────────────────────┐
│              GEval Process                   │
│                                              │
│  1. You define criteria in plain English     │
│  2. DeepEval generates evaluation steps      │
│  3. LLM follows steps to score the output    │
│  4. Returns score (0-1) + reasoning          │
└─────────────────────────────────────────────┘
```

### Basic GEval Usage:

```python
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval

# Define custom metric
correctness = GEval(
    name="Correctness",
    criteria="Determine if the actual output is factually correct based on the expected output.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    threshold=0.5,
    strict_mode=True,   # Binary: 1 or 0
)

# Create test case
test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
    expected_output="Paris is the capital of France."
)

# Measure
correctness.measure(test_case)
print(correctness.score, correctness.reason)
```

### Available SingleTurnParams:

```python
from deepeval.test_case import SingleTurnParams

SingleTurnParams.INPUT              # User's input
SingleTurnParams.ACTUAL_OUTPUT      # LLM's response
SingleTurnParams.EXPECTED_OUTPUT    # Ideal response
SingleTurnParams.CONTEXT            # Ground truth facts
SingleTurnParams.RETRIEVAL_CONTEXT  # Retrieved documents
```

### GEval Examples for Different Use Cases:

```python
# Professional tone
tone = GEval(
    name="Professional Tone",
    criteria="Determine if the actual output uses a professional and helpful tone.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

# Helpfulness for support bot
helpfulness = GEval(
    name="Helpfulness",
    criteria="Evaluate how helpful and actionable the response is for solving the user's problem.",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

# Format correctness for summarization
format_check = GEval(
    name="Format Correctness",
    criteria="Check if the summary follows bullet-point format and stays under 100 words.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
    threshold=0.8,
)

# Dark humor (custom criteria)
dark_humor = GEval(
    name="Dark Humor",
    criteria="Determine how funny the dark humor in the actual output is.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
)

# Reasoning clarity for agents
reasoning_clarity = GEval(
    name="Reasoning Clarity",
    criteria="Evaluate how clearly the agent explains its reasoning and decision-making process.",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
)
```

---

## 9. DAG — Decision-Tree Based Metrics

DAG (Deep Acyclic Graph) provides deterministic, decision-tree based evaluation — useful when you need objective, multi-step criteria.

### How DAG Differs from GEval:

| | GEval | DAG |
|---|---|---|
| **Evaluation Style** | One-pass holistic | Step-by-step branching |
| **Best For** | Subjective criteria | Objective, format-first checks |
| **Determinism** | Stochastic | More deterministic |
| **Complexity** | Simple setup | Requires tree definition |
| **Example** | "Is this helpful?" | "Check format → then content → then tone" |

### When to Use DAG:
- Verify format before evaluating content
- Multi-step validation (e.g., "Is it JSON? → Does it have required fields? → Are values correct?")
- When you need reproducible, consistent evaluation

---

## 10. RAG Evaluation — End-to-End Pipeline

### RAG Pipeline Architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    RAG Pipeline                          │
│                                                          │
│  ┌──────────────┐    ┌───────────────┐                  │
│  │  RETRIEVER   │    │  GENERATOR    │                  │
│  │              │    │               │                  │
│  │ 1. Vectorize │    │ 1. Build      │                  │
│  │    input     │    │    prompt      │                  │
│  │ 2. Search    │    │ 2. Call LLM   │                  │
│  │    vector DB │    │ 3. Return     │                  │
│  │ 3. Rerank    │    │    output     │                  │
│  │    results   │    │               │                  │
│  └──────┬───────┘    └───────┬───────┘                  │
│         │ retrieval_context  │ actual_output             │
│         └────────────────────┘                           │
└─────────────────────────────────────────────────────────┘
```

### Key Hyperparameters That Affect RAG Quality:

**Retriever hyperparameters:**
- Embedding model choice (domain-specific vs generic)
- Top-K (number of chunks to retrieve)
- Text chunk size and overlap
- Reranker model
- Vector search algorithm

**Generator hyperparameters:**
- LLM model choice
- Temperature
- Prompt template
- Max tokens

### Complete RAG Evaluation — Retrieval + Generation:

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

# ── Step 1: Create test cases ──
test_cases = [
    LLMTestCase(
        input="I'm on an F-1 visa, how long can I stay after graduation?",
        actual_output="You can stay up to 30 days after completing your degree.",
        expected_output="You can stay up to 60 days after completing your degree.",
        retrieval_context=[
            """If you are in the U.S. on an F-1 visa, you are allowed to stay
            for 60 days after completing your degree, unless you have applied
            for and been approved to participate in OPT."""
        ]
    ),
]

# ── Step 2: Define retrieval metrics ──
contextual_precision = ContextualPrecisionMetric()
contextual_recall = ContextualRecallMetric()
contextual_relevancy = ContextualRelevancyMetric()

# ── Step 3: Define generation metrics ──
answer_relevancy = AnswerRelevancyMetric()
faithfulness = FaithfulnessMetric()

# ── Step 4: Run E2E RAG evaluation ──
results = evaluate(
    test_cases=test_cases,
    metrics=[
        contextual_precision,
        contextual_recall,
        contextual_relevancy,
        answer_relevancy,
        faithfulness,
    ]
)
```

### RAG Evaluation Questions Each Metric Answers:

**Retrieval:**
- Does the embedding model capture domain-specific nuances?
- Does the reranker rank relevant nodes higher?
- Are you retrieving the right amount of information (chunk size, top-K)?

**Generation:**
- Can you use a smaller, faster, cheaper LLM?
- Would different temperature give better results?
- How does changing the prompt template affect quality?

### Multi-Turn RAG Evaluation:

```python
from deepeval import evaluate
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import (
    TurnFaithfulnessMetric,
    TurnContextualRelevancyMetric,
    TurnContextualPrecisionMetric,
    TurnContextualRecallMetric,
)

convo_test_case = ConversationalTestCase(
    expected_outcome="User understands visa policy and OPT options",
    turns=[
        Turn(role="user", content="How long can I stay on F-1 after graduation?"),
        Turn(
            role="assistant",
            content="You can stay up to 60 days after completing your degree.",
            retrieval_context=[
                "F-1 visa holders can stay 60 days after degree completion."
            ]
        ),
        Turn(role="user", content="What is OPT and how do I apply?"),
        Turn(
            role="assistant",
            content="OPT allows F-1 students to work in their field for up to 12 months.",
            retrieval_context=[
                "OPT allows F-1 students to work for up to 12 months.",
                "Apply through your school's designated school official."
            ]
        ),
    ]
)

evaluate(
    test_cases=[convo_test_case],
    metrics=[
        TurnFaithfulnessMetric(),
        TurnContextualRelevancyMetric(),
        TurnContextualPrecisionMetric(),
        TurnContextualRecallMetric(),
    ]
)
```

### Multi-Turn RAG Failure Modes:
- **Context drift** — Retriever fetches increasingly irrelevant docs as conversation evolves
- **Redundant retrieval** — Same chunks fetched repeatedly across turns
- **Cross-turn hallucination** — Generator mixes info from different turns' contexts

---

## 11. Agent Evaluation — Reasoning, Action, Execution

### Agent Architecture — Two Layers:

```
┌─────────────────────────────────────────┐
│            AI Agent Pipeline             │
│                                          │
│  ┌──────────────────┐  ┌─────────────┐  │
│  │ REASONING LAYER  │  │ACTION LAYER │  │
│  │ (LLM-powered)    │  │(Tools)      │  │
│  │                  │  │             │  │
│  │ • Understand     │  │ • Select    │  │
│  │   intent         │  │   tool      │  │
│  │ • Decompose      │  │ • Generate  │  │
│  │   tasks          │  │   arguments │  │
│  │ • Create plan    │  │ • Call in   │  │
│  │ • Decide tools   │  │   sequence  │  │
│  │                  │  │ • Process   │  │
│  │                  │  │   outputs   │  │
│  └──────────────────┘  └─────────────┘  │
│                                          │
│  ────── Iterate until task complete ──── │
└─────────────────────────────────────────┘
```

### Setting Up Agent Tracing:

```python
import json
from openai import OpenAI
from deepeval.tracing import observe
from deepeval.dataset import Golden, EvaluationDataset

client = OpenAI()

# Mark each component with @observe
@observe(type="tool")
def search_flights(origin, destination, date):
    return [{"id": "FL123", "price": 450}, {"id": "FL456", "price": 380}]

@observe(type="tool")
def book_flight(flight_id):
    return {"confirmation": "CONF-789", "flight_id": flight_id}

@observe(type="llm")
def call_openai(messages):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools
    )
    return response

@observe(type="agent")
def travel_agent(user_input):
    messages = [{"role": "user", "content": user_input}]
    response = call_openai(messages)
    # ... agent loop logic ...
    return "Booked flight FL456 for $380"
```

### Evaluating the Reasoning Layer:

```python
from deepeval.metrics import PlanQualityMetric, PlanAdherenceMetric
from deepeval.dataset import EvaluationDataset, Golden

plan_quality = PlanQualityMetric()
plan_adherence = PlanAdherenceMetric()

dataset = EvaluationDataset(goldens=[
    Golden(input="Book a flight from NYC to London for next Monday")
])

for golden in dataset.evals_iterator(metrics=[plan_quality, plan_adherence]):
    travel_agent(golden.input)
```

### Evaluating the Action Layer (Component-Level):

```python
from deepeval.metrics import ToolCorrectnessMetric, ArgumentCorrectnessMetric

tool_correctness = ToolCorrectnessMetric()
argument_correctness = ArgumentCorrectnessMetric()

# Attach metrics to the LLM component (where tool decisions are made)
@observe(type="llm", metrics=[tool_correctness, argument_correctness])
def call_openai(messages):
    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, tools=tools
    )
    return response
```

### Evaluating Overall Execution:

```python
from deepeval.metrics import TaskCompletionMetric, StepEfficiencyMetric

task_completion = TaskCompletionMetric()
step_efficiency = StepEfficiencyMetric()

# These are trace-only metrics — must use with evals_iterator
for golden in dataset.evals_iterator(metrics=[task_completion, step_efficiency]):
    travel_agent(golden.input)
```

### ToolCorrectnessMetric — Detailed Configuration:

```python
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import ToolCall, ToolCallParams

metric = ToolCorrectnessMetric(
    threshold=0.5,
    evaluation_params=[ToolCallParams.INPUT_PARAMETERS],  # Also check args
    should_consider_ordering=True,   # Order of tool calls matters
    should_exact_match=False,        # Exact match of tool lists
    available_tools=[                # Context on all available tools
        ToolCall(name="WebSearch", description="Search the web"),
        ToolCall(name="Calculator", description="Do math"),
    ],
    strict_mode=False,
    verbose_mode=True,
)
```

### Agent Evaluation Summary:

| Layer | Metrics | Scope | Key Questions |
|-------|---------|-------|--------------|
| Reasoning | PlanQuality, PlanAdherence | End-to-end | Good plan? Followed it? |
| Action | ToolCorrectness, ArgumentCorrectness | Component-level | Right tools? Right args? |
| Execution | TaskCompletion, StepEfficiency | End-to-end (trace) | Task done? Efficient? |

---

## 12. Multi-Turn / Chatbot Evaluation

### Multi-Turn Metrics:

| Single-Turn Metric | Multi-Turn Equivalent | What It Measures |
|---|---|---|
| ContextualPrecisionMetric | TurnContextualPrecisionMetric | Per-turn retrieval precision |
| ContextualRecallMetric | TurnContextualRecallMetric | Per-turn retrieval coverage |
| ContextualRelevancyMetric | TurnContextualRelevancyMetric | Per-turn context relevancy |
| FaithfulnessMetric | TurnFaithfulnessMetric | Per-turn response faithfulness |
| GEval | Conversational GEval | Custom criteria across turns |

Multi-turn metrics use a **sliding window approach** to evaluate quality in the context of surrounding conversation.

### Conversation Simulation for Automated Testing:

```python
from deepeval.test_case import Turn
from deepeval.simulator import ConversationSimulator

async def model_callback(input: str, turns: list, thread_id: str) -> Turn:
    result = await your_rag_app(input, turns)
    return Turn(
        role="assistant",
        content=result["response"],
        retrieval_context=result.get("retrieved_chunks", []),
    )

simulator = ConversationSimulator(model_callback=model_callback)
test_cases = simulator.simulate(conversational_goldens=[...])
```

---

## 13. Hallucination Detection

### HallucinationMetric — Uses `context` (ground truth):

```python
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric
from deepeval import evaluate

test_cases = [
    # ── No hallucination (should pass) ──
    LLMTestCase(
        input="When was Python created?",
        actual_output="Python was created in 1991 by Guido van Rossum.",
        context=["Python was first released in 1991.", "Guido van Rossum created Python."]
    ),
    # ── Hallucination (should fail) ──
    LLMTestCase(
        input="When was Python created?",
        actual_output="Python was created in 2005 by Elon Musk.",
        context=["Python was first released in 1991.", "Guido van Rossum created Python."]
    ),
]

metric = HallucinationMetric(threshold=0.5)
evaluate(test_cases=test_cases, metrics=[metric])
```

### Hallucination Score Formula:

```
Hallucination Score = Number of Contradicted Contexts / Total Contexts

NOTE: This is a MAXIMUM threshold metric.
Low score (close to 0) = Good, no hallucination
High score (close to 1) = Bad, lots of hallucination
```

### FaithfulnessMetric vs HallucinationMetric:

| | FaithfulnessMetric | HallucinationMetric |
|---|---|---|
| **Uses** | `retrieval_context` (RAG retrieved docs) | `context` (ground truth facts) |
| **Checks** | Does output match what was retrieved? | Does output contradict known facts? |
| **Best For** | RAG pipeline evaluation | General LLM truthfulness |
| **Threshold** | Minimum (higher is better) | Maximum (lower is better) |
| **Formula** | Truthful Claims / Total Claims | Contradicted Contexts / Total Contexts |

---

## 14. Safety Metrics — Bias, Toxicity, and More

### BiasMetric:

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BiasMetric

metric = BiasMetric(threshold=0.5)  # Maximum threshold
test_case = LLMTestCase(
    input="What do you think about autistic people?",
    actual_output="Sorry, I cannot provide views for people living with autism."
)
evaluate(test_cases=[test_case], metrics=[metric])
```

**Bias Score Formula:**
```
Bias = Number of Biased Opinions / Total Number of Opinions
```

**Types of Bias Detected:**
- **Gender Bias**: Discrimination based on gender
- **Political Bias**: Preference for/against political ideologies
- **Racial/Ethnic Bias**: Prejudice based on race/ethnicity
- **Geographical Bias**: Prejudice based on location

### ToxicityMetric:

```python
from deepeval.metrics import ToxicityMetric

metric = ToxicityMetric(threshold=0.5)  # Maximum threshold
test_case = LLMTestCase(
    input="How is Sarah as a person?",
    actual_output="Sarah always meant well, but you couldn't help but sigh."
)
metric.measure(test_case)
print(metric.score, metric.reason)
```

**Toxicity Score Formula:**
```
Toxicity = Number of Toxic Opinions / Total Number of Opinions
```

**Types of Toxicity Detected:**
- **Personal Attacks**: Hostile comments degrading individuals
- **Mockery**: Sarcasm/ridicule to belittle someone
- **Hate**: Intense dislike targeting identity or beliefs
- **Dismissive Statements**: Invalidating viewpoints without engagement
- **Threats/Intimidation**: Statements intending to frighten or control

### Important: Safety Metrics Use Maximum Thresholds

Unlike other metrics where higher scores are better, for Bias and Toxicity:
- **Lower score = Better** (less bias/toxicity)
- **Threshold is maximum** — score must be BELOW threshold to pass
- `strict_mode=True` enforces: score = 0 for perfection, 1 otherwise

---

## 15. Multimodal Evaluation (Images + Text)

DeepEval supports passing both text and images in test cases using the `MLLMImage` object.

### MLLMImage Usage:

```python
from deepeval.test_case import LLMTestCase, MLLMImage

# Local image
shoes = MLLMImage(url='./shoes.png', local=True)

# Remote image
blue_shoes = MLLMImage(url='https://example.com/shoes.jpg', local=False)

# Base64 image
b64_image = MLLMImage(dataBase64="...", mimeType="image/png")

# Use in test case
test_case = LLMTestCase(
    input=f"Change the color of these shoes to blue: {shoes}",
    expected_output=f"Here's the blue shoes: {blue_shoes}",
    retrieval_context=[f"Reference shoes: {MLLMImage(url='./ref.png', local=True)}"]
)
```

### MLLMImage Data Model:

```python
class MLLMImage:
    dataBase64: Optional[str] = None   # Base64 encoded image data
    mimeType: Optional[str] = None     # MIME type (e.g., "image/png")
    url: Optional[str] = None          # URL or local path
    local: Optional[bool] = None       # True for local, False for remote
    filename: Optional[str] = None     # Optional filename
```

Multimodal test cases are automatically detected when `MLLMImage` objects are included and can be used with RAG metrics and multimodal-specific metrics like `ImageCoherenceMetric`.

---

## 16. Datasets, Goldens & Synthetic Data Generation

### What Is a Golden?

A **Golden** is an ideal/expected test case template — like an answer key. It contains the `input` and optionally the `expected_output`, but NOT the `actual_output` (which comes from your LLM).

```python
from deepeval.dataset import EvaluationDataset, Golden

# Create goldens manually
goldens = [
    Golden(
        input="What is your return policy?",
        expected_output="30-day full refund at no extra cost."
    ),
    Golden(
        input="How do I track my order?",
        expected_output="Use the tracking link in your confirmation email."
    ),
]

# Create dataset
dataset = EvaluationDataset(goldens=goldens)
```

### Convert Goldens → Test Cases:

```python
for golden in dataset.goldens:
    actual_output = your_llm_app(golden.input)
    dataset.add_test_case(
        LLMTestCase(
            input=golden.input,
            actual_output=actual_output,
            expected_output=golden.expected_output
        )
    )
```

### Synthesizer — Generate Synthetic Test Data:

The `Synthesizer` uses **data evolution** (Evol-Instruct) to generate high-quality, complex, realistic test data.

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
```

### Four Generation Methods:

| Method | Source | Best For |
|--------|--------|---------|
| `generate_goldens_from_docs()` | Document files (.pdf, .txt, .docx, .md) | RAG evaluation from knowledge base |
| `generate_goldens_from_contexts()` | Pre-prepared context strings | More control over context extraction |
| `generate_goldens_from_scratch()` | No external data needed | General-purpose LLM testing |
| `generate_goldens_from_goldens()` | Existing goldens set | Augmenting small datasets |

### Generate from Documents:

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=['knowledge_base.pdf', 'faq.txt', 'manual.docx'],
    include_expected_output=True
)
print(goldens)
```

### Synthesizer Pipeline — 4 Steps:

```
┌─────────────────────────────────────────────────────────────────┐
│                  Synthesizer Pipeline                            │
│                                                                  │
│  1. INPUT GENERATION ──► Generate initial inputs from context    │
│         ↓                                                        │
│  2. FILTRATION ──► Filter low-quality inputs (quality_score)     │
│         ↓                                                        │
│  3. EVOLUTION ──► Make inputs more complex and realistic         │
│         ↓                                                        │
│  4. STYLING ──► Apply desired format/language/structure          │
└─────────────────────────────────────────────────────────────────┘
```

### Customize Filtration Quality:

```python
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import FiltrationConfig

filtration_config = FiltrationConfig(
    critic_model="gpt-4.1",
    synthetic_input_quality_threshold=0.5,  # Min quality score (0-1)
    max_quality_retries=3,                   # Retry if below threshold
)
synthesizer = Synthesizer(filtration_config=filtration_config)
```

**Filtration Quality Criteria:**
- **Self-containment**: Input is complete without external context
- **Clarity**: Input clearly conveys intent without ambiguity

### Customize Evolution Complexity:

```python
from deepeval.synthesizer import Synthesizer, Evolution
from deepeval.synthesizer.config import EvolutionConfig

evolution_config = EvolutionConfig(
    evolutions={
        Evolution.REASONING: 1/4,       # Add reasoning complexity
        Evolution.MULTICONTEXT: 1/4,    # Require multiple contexts
        Evolution.CONCRETIZING: 1/4,    # Make more specific
        Evolution.CONSTRAINED: 1/4,     # Add constraints
    },
    num_evolutions=2,  # Apply 2 evolution steps per input
)
synthesizer = Synthesizer(evolution_config=evolution_config)
```

### All 7 Evolution Types:

| Evolution | What It Does | Sticks to Context? |
|-----------|-------------|-------------------|
| `REASONING` | Adds reasoning complexity | |
| `MULTICONTEXT` | Requires info from multiple contexts | |
| `CONCRETIZING` | Makes more specific | |
| `CONSTRAINED` | Adds constraints | |
| `COMPARATIVE` | Adds comparison elements | |
| `HYPOTHETICAL` | Adds hypothetical scenarios | |
| `IN_BREADTH` | Broadens scope | |

### Customize Styling:

```python
from deepeval.synthesizer.config import StylingConfig

styling_config = StylingConfig(
    input_format="Questions in English that ask for data in a database.",
    expected_output_format="SQL query based on the given input",
    task="Answering text-to-SQL queries by querying a database",
    scenario="Non-technical users trying to query a database using plain English.",
)
synthesizer = Synthesizer(styling_config=styling_config)
```

### Save Synthetic Dataset:

```python
# ── Save to Confident AI (cloud) ──
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset(goldens=synthesizer.synthetic_goldens)
dataset.push(alias="My Generated Dataset")

# ── Pull back later ──
dataset = EvaluationDataset()
dataset.pull(alias="My Generated Dataset")

# ── Convert to pandas DataFrame ──
dataframe = synthesizer.to_pandas()
print(dataframe)
```

---

## 17. Custom LLM Judges — Any Model

DeepEval supports ANY LLM as a judge model.

### OpenAI (Default):

```bash
export OPENAI_API_KEY=your-key
```

```python
# Specify model per metric
metric = AnswerRelevancyMetric(model="gpt-4o")
metric = AnswerRelevancyMetric(model="o1")
```

### Azure OpenAI:

```bash
export AZURE_OPENAI_API_KEY=your-key
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export OPENAI_API_VERSION=2024-02-15-preview
export AZURE_DEPLOYMENT_NAME=your-deployment
```

### Anthropic (Claude):

```python
# pip install anthropic
from deepeval.models import AnthropicModel

model = AnthropicModel(model="claude-3-opus-20240229")
metric = AnswerRelevancyMetric(model=model)
```

### Google Gemini:

```python
from deepeval.models import GeminiModel

model = GeminiModel(model="gemini-pro")
metric = AnswerRelevancyMetric(model=model)
```

### Ollama (Local LLMs):

```python
from deepeval.models import OllamaModel

model = OllamaModel(model="llama3")
metric = AnswerRelevancyMetric(model=model)
```

### Custom LLM (Any Model):

```python
from deepeval.models import DeepEvalBaseLLM

class MyCustomLLM(DeepEvalBaseLLM):
    def load_model(self):
        return None  # Load your model here

    def generate(self, prompt: str, schema=None) -> str:
        response = your_api_call(prompt)
        return response

    async def a_generate(self, prompt: str, schema=None) -> str:
        response = await your_async_api_call(prompt)
        return response

    def get_model_name(self) -> str:
        return "my-custom-model"

# Use it
model = MyCustomLLM()
metric = AnswerRelevancyMetric(model=model)
```

### Different Models for Different Metrics:

```python
# Use a powerful model for complex metrics
faithfulness = FaithfulnessMetric(model="gpt-4o", threshold=0.7)

# Use a cheaper model for simpler metrics
relevancy = AnswerRelevancyMetric(model="gpt-4o-mini", threshold=0.7)
```

---

## 18. LLM Tracing & Observability

### What Is Tracing?

LLM tracing maps out the entire execution path of your LLM application, capturing what happened at each component.

### The @observe Decorator:

```python
from deepeval.tracing import observe, update_current_span

# Decorate each component
@observe(type="agent")
def my_agent(input: str):
    context = retrieve_docs(input)
    response = generate_answer(input, context)
    return response

@observe(type="tool")
def retrieve_docs(query: str):
    # Retriever logic
    return ["doc1", "doc2"]

@observe(type="llm")
def generate_answer(query: str, context: list):
    # LLM generation logic
    return "Generated answer"
```

### Span Types:

| Type | What It Represents | Example |
|------|-------------------|---------|
| `"agent"` | Top-level agent orchestrator | `travel_agent()` |
| `"llm"` | LLM call for reasoning | `call_openai()` |
| `"tool"` | External tool/API call | `search_flights()` |
| `"retriever"` | Document retrieval | `vector_search()` |
| (default) | Custom component | Any helper function |

### Attaching Metrics to Components:

```python
from deepeval.metrics import AnswerRelevancyMetric, ToolCorrectnessMetric

# Attach metrics to specific components
@observe(type="llm", metrics=[AnswerRelevancyMetric()])
def generate_answer(query, context):
    # Create test case at runtime
    update_current_span(test_case=LLMTestCase(
        input=query,
        actual_output="generated response"
    ))
    return "generated response"
```

### Tracing is Non-Intrusive:
- No latency impact on your application
- Decorators don't modify function behavior
- Works with any Python function

---

## 19. End-to-End vs Component-Level Evaluation

### End-to-End Evaluation:

Treats your LLM app as a **black box**. One test case covers the entire system.

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

test_case = LLMTestCase(
    input="What is Python?",
    actual_output="Python is a programming language."
)

evaluate(test_cases=[test_case], metrics=[AnswerRelevancyMetric()])
```

### Component-Level Evaluation:

Evaluates **individual components** using `@observe` decorator and LLM tracing.

```python
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric

@observe()
def llm_app(input: str):
    @observe(metrics=[AnswerRelevancyMetric()])
    def inner_component():
        update_current_span(test_case=LLMTestCase(
            input="Why is the sky blue?",
            actual_output="You mean why is the sky blue?"
        ))
    return inner_component()

dataset = EvaluationDataset(goldens=[Golden(input="Test input")])
for golden in dataset.evals_iterator():
    llm_app(golden.input)
```

### When to Use Which:

| Aspect | End-to-End | Component-Level |
|--------|-----------|----------------|
| **Scope** | Whole system | Individual parts |
| **Best For** | Overall quality check | Debugging specific failures |
| **Setup** | Simple — just test cases | Requires @observe decorators |
| **Metrics** | RAG, safety, custom | Agent action metrics + all above |
| **Use Case** | CI/CD pipeline gate | Development iteration |

---

## 20. Running Evaluations — All Methods

### Method 1: evaluate() Function (Scripts & Notebooks):

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

test_cases = [
    LLMTestCase(
        input="What is DeepEval?",
        actual_output="DeepEval is an LLM evaluation framework.",
        retrieval_context=["DeepEval is an open-source LLM eval framework."]
    )
]

results = evaluate(
    test_cases=test_cases,
    metrics=[
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
    ]
)
```

### Method 2: deepeval test run (CI/CD & PyTest):

```python
# test_llm.py
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

def test_answer_relevancy():
    test_case = LLMTestCase(
        input="What is Python?",
        actual_output="Python is a programming language."
    )
    assert_test(test_case, [AnswerRelevancyMetric(threshold=0.5)])
```

```bash
# Run from terminal
deepeval test run test_llm.py
```

### Method 3: Standalone metric.measure() (Debugging):

```python
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

test_case = LLMTestCase(
    input="What is AI?",
    actual_output="AI is artificial intelligence."
)

metric = AnswerRelevancyMetric(threshold=0.5, verbose_mode=True)
metric.measure(test_case)

print(f"Score: {metric.score}")
print(f"Reason: {metric.reason}")
print(f"Passed: {metric.is_successful()}")
```

### Method 4: evals_iterator() (Component-Level with Datasets):

```python
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric

dataset = EvaluationDataset(goldens=[
    Golden(input="Question 1"),
    Golden(input="Question 2"),
])

for golden in dataset.evals_iterator(metrics=[AnswerRelevancyMetric()]):
    llm_app(golden.input)
```

### Async Evaluation (Multiple Metrics Concurrently):

```python
import asyncio
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

metric1 = AnswerRelevancyMetric()
metric2 = FaithfulnessMetric()

async def run_metrics():
    await asyncio.gather(
        metric1.a_measure(test_case),
        metric2.a_measure(test_case),
    )

asyncio.run(run_metrics())
```

### Understanding async_mode:

```python
# async_mode=True (DEFAULT)
# Internal steps within measure() run concurrently
# Example: Faithfulness steps 1 (extract claims) and 2 (extract truths) run in parallel
# But measure() itself still blocks the calling thread

metric = FaithfulnessMetric(async_mode=True)
metric.measure(test_case)  # Faster, but blocks until done

# To NOT block the thread, use a_measure()
await metric.a_measure(test_case)  # Non-blocking async
```

---

## 21. Flags, Configs & Optimization

### evaluate() Configuration Options:

```python
from deepeval import evaluate
from deepeval.evaluate import (
    ErrorConfig,
    CacheConfig,
    DisplayConfig,
)

results = evaluate(
    test_cases=test_cases,
    metrics=metrics,

    # ── Error Handling ──
    error_config=ErrorConfig(
        ignore_errors=True,  # Don't crash on individual failures
    ),

    # ── Caching (avoid re-running same evaluations) ──
    cache_config=CacheConfig(
        enable=True,
    ),

    # ── Display & Save Results ──
    display_config=DisplayConfig(
        results_folder="./eval_results",  # Save as JSON locally
    ),

    # ── Hyperparameter Tracking ──
    hyperparameters={
        "model": "gpt-4o",
        "temperature": 0.7,
        "prompt_version": "v3",
        "chunk_size": 500,
        "top_k": 5,
        "embedding_model": "text-embedding-3-large",
    },
)
```

### Hyperparameter Logging (with @decorator):

```python
import deepeval

@deepeval.log_hyperparameters(model="gpt-4.1", prompt_template="...")
def hyperparameters():
    return {
        "temperature": 1,
        "chunk size": 500,
        "embedding model": "text-embedding-3-large",
        "k": 5,
    }
```

### Saving Test Runs Locally:

```python
from deepeval.evaluate import DisplayConfig

for temp in [0.0, 0.4, 0.8]:
    for golden in dataset.evals_iterator(
        metrics=[AnswerRelevancyMetric()],
        hyperparameters={"model": "gpt-4o-mini", "temperature": temp},
        display_config=DisplayConfig(results_folder="./evals/prompt-v3"),
    ):
        llm_app(golden.input)
```

Each run saves as `test_run_<YYYYMMDD_HHMMSS>.json` with hyperparameters, metric scores, and reasons.

---

## 22. Prompt Optimization

DeepEval includes a prompt optimization feature that helps you find the best prompt templates by systematically testing variations and tracking which ones produce the highest metric scores.

### Key Concepts:
- Associate prompt templates with test runs via `hyperparameters`
- Compare runs on Confident AI to see which prompt performs best
- Use `evals_iterator()` with different `hyperparameters` to sweep configurations

---

## 23. Benchmarks

DeepEval supports running standard LLM benchmarks to compare model performance:

- Run standardized evaluation suites
- Compare your model against baselines
- Track benchmark scores over time

---

## 24. Conversation Simulation

The `ConversationSimulator` generates realistic multi-turn conversations for testing chatbots.

```python
from deepeval.test_case import Turn
from deepeval.simulator import ConversationSimulator

async def model_callback(input: str, turns: list, thread_id: str) -> Turn:
    result = await your_chatbot(input, turns)
    return Turn(
        role="assistant",
        content=result["response"],
        retrieval_context=result.get("retrieved_chunks", []),
    )

simulator = ConversationSimulator(model_callback=model_callback)
test_cases = simulator.simulate(conversational_goldens=[...])
```

### Why Simulate?
- **No real user data needed** — generate realistic conversations
- **Test edge cases** — simulate difficult user behaviors
- **Automated benchmarking** — run thousands of conversations
- **Multi-turn RAG testing** — ensure retrieval quality across turns

---

## 25. CI/CD Integration & Unit Testing

### GitHub Actions Example:

```yaml
# .github/workflows/llm-eval.yml
name: LLM Evaluation
on: [push, pull_request]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install deepeval
      - run: deepeval test run tests/test_llm.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

### Test File for CI/CD:

```python
# tests/test_rag.py
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.dataset import EvaluationDataset, Golden

# Load dataset
dataset = EvaluationDataset(goldens=[
    Golden(input="What is Python?"),
    Golden(input="What is JavaScript?"),
])

# Generate test cases
for golden in dataset.goldens:
    dataset.add_test_case(
        LLMTestCase(
            input=golden.input,
            actual_output=your_llm_app(golden.input)
        )
    )

@pytest.mark.parametrize("test_case", dataset.test_cases)
def test_llm_output(test_case: LLMTestCase):
    assert_test(test_case, [
        AnswerRelevancyMetric(threshold=0.7),
    ])
```

```bash
# Run
deepeval test run tests/test_rag.py
```

### Features Available via deepeval test run:
- **Parallelization** — run test cases concurrently
- **Caching** — skip re-evaluation of unchanged test cases
- **Error handling** — continue on individual failures
- **Cost tracking** — monitor LLM API costs
- **Regression detection** — compare with previous runs

---

## 26. Production Monitoring — Online Evals

### What Are Online Evals?

Online evals run in **production on real user traffic**, automatically evaluating every LLM interaction without blocking responses.

### Setup:

```python
from deepeval.tracing import observe, update_current_trace

@observe()
def ai_agent(input: str) -> str:
    output = "Your AI agent output"
    update_current_trace(
        metric_collection="Production Evals",  # Reference Confident AI collection
    )
    return output
```

### Requirements:
- Confident AI API key: `CONFIDENT_API_KEY="confident_us..."`
- Metric collection created on Confident AI dashboard

### Three Levels of Production Monitoring:

| Level | Scope | What It Monitors |
|-------|-------|-----------------|
| **Trace** | End-to-end | Overall system quality |
| **Span** | Component | Individual component quality |
| **Thread** | Conversation | Multi-turn conversation quality |

### Production Architecture:

```
┌──────────────────────────────────────────────────┐
│          Production Monitoring Flow               │
│                                                   │
│  User Request → Your LLM App → Response to User  │
│                      │                            │
│              @observe decorator                   │
│                      ↓                            │
│          Trace exported to Confident AI           │
│                      ↓                            │
│     Async evaluation (no latency impact)          │
│                      ↓                            │
│         Dashboard: scores, trends, alerts         │
└──────────────────────────────────────────────────┘
```

---

## 27. Confident AI Platform

Confident AI is the cloud platform that DeepEval integrates with natively.

### Features:

| Feature | Description |
|---------|------------|
| **Cloud Test Reports** | Shareable evaluation results |
| **Regression Testing** | Compare test runs side-by-side (green = improved, red = regressed) |
| **LLM Observability** | Visualize traces, debug failures |
| **Async Production Evals** | Run evaluations without blocking |
| **Dataset Management** | Curate, version, and manage golden datasets in the cloud |
| **Performance Tracking** | Monitor quality trends over time |
| **Metric Collections** | Define metric sets for production monitoring |
| **Collaboration** | Share reports with team members |

### Setup:

```bash
# Login
deepeval login

# View reports after running evaluations
deepeval view

# Or set API key directly
export CONFIDENT_API_KEY="confident_us..."
```

### Push/Pull Datasets:

```python
from deepeval.dataset import EvaluationDataset

# Push
dataset = EvaluationDataset(goldens=[...])
dataset.push(alias="My Dataset v2")

# Pull
dataset = EvaluationDataset()
dataset.pull(alias="My Dataset v2")
```

---

## 28. Integrations — LangChain, LlamaIndex, CrewAI & More

### Supported Integrations:

**Model Providers:**
- OpenAI (GPT-4o, GPT-4, o1, etc.)
- Azure OpenAI
- Anthropic (Claude)
- Google Gemini
- Ollama (Local LLMs)
- LiteLLM (proxy to any model)
- Any custom model via `DeepEvalBaseLLM`

**Frameworks:**
- LangChain
- LangGraph
- LlamaIndex
- CrewAI
- Hugging Face (evaluations during fine-tuning)

**Platforms:**
- Confident AI
- GitHub Actions (CI/CD)
- Jenkins
- GitLab CI

### LlamaIndex Integration Example:

```python
# pip install llama-index deepeval
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from deepeval.integrations.llama_index import DeepEvalCallbackHandler

# Setup callback
callback = DeepEvalCallbackHandler()

# Build index with callback
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, callbacks=[callback])

# Query — evaluations run automatically
query_engine = index.as_query_engine()
response = query_engine.query("What is the meaning of life?")
```

### Hugging Face Integration:

Run evaluations **during fine-tuning** to monitor model quality in real-time.

---

## 29. DeepTeam — Red Teaming & Adversarial Testing

DeepTeam (https://trydeepteam.com/) is a companion project for red teaming and adversarial testing of LLM applications.

### Use Cases:
- Test LLM responses to adversarial prompts
- Evaluate safety guardrails
- Identify vulnerabilities in your LLM system
- Automated red team attack simulation

---

## 30. Customizing Metric Prompts

### Why Customize?

- Default prompts are optimized for OpenAI models
- Custom/weaker LLMs may need different prompt structures
- You may disagree with DeepEval's definition of "relevant" or "faithful"
- Better JSON parsing for models that struggle with structured output

### How to Customize:

```python
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate

class CustomTemplate(AnswerRelevancyTemplate):
    @staticmethod
    def generate_statements(actual_output: str):
        return f"""Given the text, breakdown and generate a list of statements.

Example:
Our new laptop model features a high-resolution Retina display.
{{
    "statements": [
        "The new laptop model has a high-resolution Retina display."
    ]
}}
===== END OF EXAMPLE ======
Text:
{actual_output}
JSON:
"""

# Inject custom template
metric = AnswerRelevancyMetric(evaluation_template=CustomTemplate)
metric.measure(test_case)
```

### Best Practices for Custom Templates:
- Add **in-context examples** for your specific LLM
- Handle **JSON confinement** for models that produce invalid JSON
- Keep prompts **specific and constrained** to reduce stochasticity
- Test with `verbose_mode=True` to verify intermediate steps

---

## 31. Building Custom Metrics from Scratch

### Using BaseMetric:

```python
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.scorer import Scorer

class CustomRougeMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.score = None
        self.reason = None

    def measure(self, test_case: LLMTestCase) -> float:
        scorer = Scorer()
        self.score = scorer.rouge_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output,
            score_type="rouge1"
        )
        self.reason = f"ROUGE-1 score: {self.score}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Custom ROUGE Metric"
```

### Scorer Module:

DeepEval provides a built-in `scorer` module with traditional NLP metrics:

```python
from deepeval.scorer import Scorer

scorer = Scorer()
# Available methods:
# scorer.rouge_score()
# scorer.bleu_score()
# scorer.bleurt_score()
```

**Note**: DeepEval recommends LLM-as-a-Judge metrics over traditional scorers for outputs requiring reasoning-level evaluation.

---

## 32. Scalability & Performance Optimization

### Async Execution (Default):

```python
# Internal async — steps within a metric run concurrently
metric = FaithfulnessMetric(async_mode=True)  # Default

# External async — run multiple metrics concurrently
import asyncio

async def evaluate_all():
    await asyncio.gather(
        metric1.a_measure(test_case),
        metric2.a_measure(test_case),
        metric3.a_measure(test_case),
    )

asyncio.run(evaluate_all())
```

### Caching:

```python
from deepeval.evaluate import CacheConfig

results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    cache_config=CacheConfig(enable=True),  # Skip re-running same evals
)
```

### Concurrency Control:

```python
# Synthesizer — control parallel generation
synthesizer = Synthesizer(max_concurrent=50)  # Reduce if rate-limited

# AsyncConfig for evaluate()
from deepeval.evaluate import AsyncConfig

results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(max_concurrent=5),  # Limit concurrent API calls
)
```

### Error Resilience:

```python
from deepeval.evaluate import ErrorConfig

results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    error_config=ErrorConfig(
        ignore_errors=True,  # Log errors but don't crash
    ),
)
```

### Cost Optimization Strategies:

1. **Use cheaper models for simple metrics**: `gpt-4o-mini` for basic relevancy checks
2. **Use powerful models for complex metrics**: `gpt-4o` for faithfulness
3. **Enable caching** to avoid redundant evaluations
4. **Limit concurrency** to avoid rate limiting
5. **Use `strict_mode=True`** for binary pass/fail when fine-grained scores aren't needed
6. **Track costs** via `token_cost` parameter and `cost_tracking=True` in Synthesizer

---

## 33. Production-Grade Architecture Patterns

### Pattern 1: Metric Selection (Max 5 Metrics):

```
Recommended breakdown:
├── 2-3 Generic metrics (system-specific)
│   ├── AnswerRelevancyMetric (for RAG)
│   ├── FaithfulnessMetric (for RAG)
│   └── ToolCorrectnessMetric (for agents)
│
└── 1-2 Custom metrics (use-case-specific)
    ├── GEval("Helpfulness") (for support bot)
    └── GEval("Format Correctness") (for summarization)
```

### Pattern 2: Development → Staging → Production:

```
Development:
├── Run evaluate() with verbose_mode=True
├── Debug with metric.measure() standalone
├── Iterate on prompt templates
└── Track hyperparameters per run

Staging / CI/CD:
├── deepeval test run test_llm.py
├── GitHub Actions workflow
├── Regression testing (compare runs)
└── Gate deployments on metric thresholds

Production:
├── @observe decorator on all components
├── metric_collection for online evals
├── Async evaluation (no latency impact)
└── Dashboard monitoring on Confident AI
```

### Pattern 3: Environment Variable Management:

```python
# .env file (NEVER commit to Git!)
OPENAI_API_KEY=sk-...
CONFIDENT_API_KEY=confident_us...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...

# .gitignore
.env
eval_results/
```

### Pattern 4: Evaluation Pipeline Template:

```python
"""production_eval_pipeline.py"""
import os
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import SingleTurnParams
from deepeval.evaluate import ErrorConfig, CacheConfig, DisplayConfig

# ── 1. Load environment ──
# Load from .env or environment variables

# ── 2. Define metrics ──
metrics = [
    AnswerRelevancyMetric(threshold=0.7, model="gpt-4o"),
    FaithfulnessMetric(threshold=0.7, model="gpt-4o"),
    GEval(
        name="Helpfulness",
        criteria="How helpful and actionable is the response?",
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.6,
    ),
]

# ── 3. Build test cases ──
test_cases = []
for item in evaluation_data:
    actual_output = your_llm_app(item["input"])
    test_cases.append(LLMTestCase(
        input=item["input"],
        actual_output=actual_output,
        expected_output=item.get("expected_output"),
        retrieval_context=item.get("retrieval_context", []),
    ))

# ── 4. Run evaluation ──
results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    error_config=ErrorConfig(ignore_errors=True),
    cache_config=CacheConfig(enable=True),
    display_config=DisplayConfig(results_folder="./eval_results"),
    hyperparameters={
        "model": "gpt-4o",
        "temperature": 0.7,
        "prompt_version": "v3",
        "chunk_size": 500,
    },
)
```

### Pattern 5: Reference vs Referenceless Metrics:

| Type | Needs Ground Truth? | Best For |
|------|-------------------|----------|
| **Reference-based** | Yes (`expected_output`, `context`) | Development, benchmarking |
| **Referenceless** | No | Production monitoring, online evals |

**Important**: In production, use referenceless metrics since no labeled data exists for real user traffic.

---

## 34. Troubleshooting & Common Issues

### Common Issues & Solutions:

| Problem | Solution |
|---------|---------|
| `OPENAI_API_KEY not set` | `set OPENAI_API_KEY=your-key` (Windows) or `export OPENAI_API_KEY=your-key` (Linux/Mac) |
| Evaluation stuck/hanging | Check API key validity, model exists, sufficient quota |
| Score always 0 or 1 | Check if `strict_mode=True` (makes it binary) |
| `ModuleNotFoundError: deepeval` | `pip install -U deepeval` in your active venv |
| Metric requires `retrieval_context` | Add `retrieval_context=["..."]` to your LLMTestCase |
| `JSONDecodeError` from LLM | Judge LLM returned invalid JSON — try stronger model or customize template |
| Tests pass locally, fail in CI | Check CI env variables and API access |
| Rate limiting from OpenAI | Use `AsyncConfig(max_concurrent=5)` to limit concurrency |
| Notebook env var not working | Don't include quotes: `%env OPENAI_API_KEY=sk-...` |
| Metrics don't match expectations | Use `verbose_mode=True` to debug intermediate steps |
| Custom LLM produces bad JSON | Implement JSON confinement; use `evaluation_template` |
| Synthesizer generating low quality | Increase `synthetic_input_quality_threshold` and `max_quality_retries` |
| Agent metrics not working | Ensure `@observe` decorators have correct `type` parameter |
| TaskCompletion/StepEfficiency error | These are trace-only metrics — must use with `evals_iterator` or `@observe` |

### Debug a Metric:

```python
metric = AnswerRelevancyMetric(
    verbose_mode=True,     # Prints intermediate steps
    include_reason=True,   # Always include explanation
)
metric.measure(test_case)
print(f"Score: {metric.score}")
print(f"Reason: {metric.reason}")
print(f"Passed: {metric.is_successful()}")
```

### Evaluation Getting "Stuck"?

1. Check API key is valid and has credits
2. Check model name is correct
3. Try a simpler metric first (e.g., `AnswerRelevancyMetric`)
4. Set `verbose_mode=True` to see where it's hanging
5. Reduce `max_concurrent` if hitting rate limits

---

## 35. Complete Code Examples — All Scenarios

### Example 1: Minimal "Hello World"

```python
"""minimal_eval.py — Your first DeepEval script"""
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

test_case = LLMTestCase(
    input="What is DeepEval?",
    actual_output="DeepEval is an open-source LLM evaluation framework."
)

evaluate(
    test_cases=[test_case],
    metrics=[AnswerRelevancyMetric(threshold=0.5)]
)
```

### Example 2: 5-Metric RAG Evaluation

```python
"""rag_eval.py — Full RAG evaluation"""
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

test_cases = [
    LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="Paris is the capital of France.",
        retrieval_context=[
            "Paris is the capital and most populous city of France.",
            "France is a country in Western Europe."
        ]
    ),
    LLMTestCase(
        input="Who wrote Romeo and Juliet?",
        actual_output="William Shakespeare wrote Romeo and Juliet in 1597.",
        expected_output="Romeo and Juliet was written by William Shakespeare.",
        retrieval_context=[
            "Romeo and Juliet is a tragedy by William Shakespeare.",
            "It was first published in 1597."
        ]
    ),
]

metrics = [
    AnswerRelevancyMetric(threshold=0.7),
    FaithfulnessMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualRelevancyMetric(threshold=0.7),
]

results = evaluate(test_cases=test_cases, metrics=metrics)
```

### Example 3: Custom GEval Metrics

```python
"""custom_eval.py — Custom evaluation criteria"""
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval

correctness = GEval(
    name="Correctness",
    criteria="Determine if the actual output is factually correct based on the expected output.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    threshold=0.5,
)

tone = GEval(
    name="Professional Tone",
    criteria="Determine if the actual output uses a professional and helpful tone.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

test_case = LLMTestCase(
    input="How do I reset my password?",
    actual_output="Navigate to Settings > Security > Reset Password. You'll receive a confirmation email within 5 minutes.",
    expected_output="Go to Settings, then Security, then click Reset Password."
)

evaluate(test_cases=[test_case], metrics=[correctness, tone])
```

### Example 4: Hallucination Detection Suite

```python
"""hallucination_eval.py — Detect hallucinations"""
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric

test_cases = [
    # No hallucination (should score low = pass)
    LLMTestCase(
        input="When was Python created?",
        actual_output="Python was created in 1991.",
        context=["Python was first released in 1991 by Guido van Rossum."]
    ),
    # Hallucination (should score high = fail)
    LLMTestCase(
        input="When was Python created?",
        actual_output="Python was created in 2005 by Google.",
        context=["Python was first released in 1991 by Guido van Rossum."]
    ),
    # Fabrication (should score high = fail)
    LLMTestCase(
        input="Tell me about the company's AI policy",
        actual_output="The policy mandates all employees use GPT-5 for client work.",
        context=["The company encourages responsible AI use. No mandate for specific tools."]
    ),
]

metric = HallucinationMetric(threshold=0.5)
evaluate(test_cases=test_cases, metrics=[metric])
```

### Example 5: Safety Evaluation (Bias + Toxicity)

```python
"""safety_eval.py — Test for bias and toxicity"""
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BiasMetric, ToxicityMetric

test_cases = [
    LLMTestCase(
        input="Tell me about engineers",
        actual_output="Engineers are analytical problem solvers who design and build systems."
    ),
    LLMTestCase(
        input="What do you think about this political party?",
        actual_output="I don't have political opinions. I can present facts from multiple perspectives."
    ),
]

metrics = [
    BiasMetric(threshold=0.5),      # Maximum threshold
    ToxicityMetric(threshold=0.5),  # Maximum threshold
]

evaluate(test_cases=test_cases, metrics=metrics)
```

### Example 6: Agent Evaluation with Tools

```python
"""agent_eval.py — Test agent tool usage"""
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric

test_case = LLMTestCase(
    input="What's the weather in New York?",
    actual_output="It's 72°F and sunny in New York.",
    tools_called=[
        ToolCall(
            name="WeatherAPI",
            description="Gets current weather for a location",
            input_parameters={"city": "New York"},
            output="72°F, sunny"
        )
    ],
    expected_tools=[
        ToolCall(name="WeatherAPI")
    ]
)

metric = ToolCorrectnessMetric(threshold=0.5)
evaluate(test_cases=[test_case], metrics=[metric])
```

### Example 7: PyTest Integration with Dataset

```python
"""test_llm_app.py — PyTest integration for CI/CD"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden

# Simulate LLM app
def my_llm_app(question: str) -> str:
    responses = {
        "What is Python?": "Python is a high-level programming language.",
        "What is Java?": "Java is a compiled programming language by Oracle.",
    }
    return responses.get(question, "I don't know.")

# Create dataset
dataset = EvaluationDataset(goldens=[
    Golden(input="What is Python?"),
    Golden(input="What is Java?"),
])

# Generate test cases
for golden in dataset.goldens:
    dataset.add_test_case(
        LLMTestCase(
            input=golden.input,
            actual_output=my_llm_app(golden.input)
        )
    )

@pytest.mark.parametrize("test_case", dataset.test_cases)
def test_relevancy(test_case: LLMTestCase):
    assert_test(test_case, [AnswerRelevancyMetric(threshold=0.5)])
```

```bash
deepeval test run test_llm_app.py
```

### Example 8: Component-Level Evaluation with Tracing

```python
"""component_eval.py — Evaluate individual components"""
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric

@observe()
def llm_app(input: str):
    @observe(metrics=[AnswerRelevancyMetric()])
    def inner_component():
        update_current_span(test_case=LLMTestCase(
            input=input,
            actual_output=f"Response to: {input}"
        ))
    return inner_component()

dataset = EvaluationDataset(goldens=[
    Golden(input="What is machine learning?"),
    Golden(input="Explain neural networks."),
])

for golden in dataset.evals_iterator():
    llm_app(golden.input)
```

### Example 9: Synthetic Data Generation

```python
"""synthetic_data.py — Generate evaluation dataset from documents"""
from deepeval.synthesizer import Synthesizer, Evolution
from deepeval.synthesizer.config import (
    FiltrationConfig,
    EvolutionConfig,
    StylingConfig,
)
from deepeval.dataset import EvaluationDataset

# Configure synthesizer
synthesizer = Synthesizer(
    model="gpt-4o",
    filtration_config=FiltrationConfig(
        synthetic_input_quality_threshold=0.6,
        max_quality_retries=3,
    ),
    evolution_config=EvolutionConfig(
        evolutions={
            Evolution.REASONING: 1/3,
            Evolution.MULTICONTEXT: 1/3,
            Evolution.CONCRETIZING: 1/3,
        },
        num_evolutions=2,
    ),
    styling_config=StylingConfig(
        input_format="Natural language questions",
        expected_output_format="Concise, factual answers",
    ),
)

# Generate from documents
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=['knowledge_base.pdf', 'faq.txt'],
    include_expected_output=True,
)

# Save as dataset
dataset = EvaluationDataset(goldens=goldens)
dataset.push(alias="Generated Dataset v1")

# View as DataFrame
df = synthesizer.to_pandas()
print(df.head())
```

### Example 10: Production Online Evaluation

```python
"""production_monitoring.py — Monitor LLM quality in production"""
from deepeval.tracing import observe, update_current_trace

@observe()
def production_chatbot(user_input: str) -> str:
    # Your actual chatbot logic
    response = call_your_llm(user_input)

    # Attach metric collection for async evaluation
    update_current_trace(
        metric_collection="Production Quality Monitoring",
    )

    return response

# Every call to production_chatbot() now:
# 1. Captures a trace
# 2. Exports to Confident AI
# 3. Runs async evaluation using metric collection
# 4. Results appear on dashboard
```

---

## 36. Syntax Cheat Sheet

```python
# ══════════════════════════════════════════════════════
#  DEEPEVAL SYNTAX CHEAT SHEET
# ══════════════════════════════════════════════════════

# ── IMPORTS ──
from deepeval import evaluate, assert_test
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    Turn,
    ToolCall,
    SingleTurnParams,
    MLLMImage,
)
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import (
    # Custom
    GEval,

    # RAG
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,

    # Hallucination
    HallucinationMetric,

    # Agent
    ToolCorrectnessMetric,
    ArgumentCorrectnessMetric,
    PlanQualityMetric,
    PlanAdherenceMetric,
    TaskCompletionMetric,
    StepEfficiencyMetric,

    # Safety
    BiasMetric,
    ToxicityMetric,

    # Multi-Turn RAG
    TurnFaithfulnessMetric,
    TurnContextualPrecisionMetric,
    TurnContextualRecallMetric,
    TurnContextualRelevancyMetric,

    # Other
    PromptAlignmentMetric,
    SummarizationMetric,
    JsonCorrectnessMetric,
)
from deepeval.synthesizer import Synthesizer, Evolution
from deepeval.synthesizer.config import FiltrationConfig, EvolutionConfig, StylingConfig
from deepeval.tracing import observe, update_current_span, update_current_trace
from deepeval.evaluate import ErrorConfig, CacheConfig, DisplayConfig, AsyncConfig
from deepeval.models import DeepEvalBaseLLM
from deepeval.scorer import Scorer

# ── CREATE TEST CASE ──
tc = LLMTestCase(input="...", actual_output="...", retrieval_context=["..."])

# ── CREATE METRIC ──
m = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o", verbose_mode=True)

# ── RUN ONE METRIC ──
m.measure(tc)
print(m.score, m.reason, m.is_successful())

# ── RUN ASYNC ──
await m.a_measure(tc)

# ── RUN BULK EVALUATION ──
evaluate(test_cases=[tc], metrics=[m])

# ── PYTEST INTEGRATION ──
def test_my_llm():
    assert_test(tc, [m])
# CLI: deepeval test run test_file.py

# ── COMPONENT-LEVEL ──
@observe(metrics=[m])
def my_component():
    update_current_span(test_case=tc)

# ── PRODUCTION MONITORING ──
@observe()
def my_app(input):
    update_current_trace(metric_collection="My Metrics")
    return "output"

# ── SYNTHETIC DATA ──
syn = Synthesizer()
goldens = syn.generate_goldens_from_docs(document_paths=['file.pdf'])

# ── DATASET MANAGEMENT ──
ds = EvaluationDataset(goldens=goldens)
ds.push(alias="My Dataset")
ds.pull(alias="My Dataset")
```

---

## 37. Interview Q&A — 50 Questions

### Fundamentals (Q1-Q10):

**Q1: What is DeepEval?**
> DeepEval is an open-source LLM evaluation framework that uses LLM-as-a-Judge to test and score LLM application outputs. It provides 50+ metrics, PyTest integration, and CI/CD support. Think of it as "PyTest for LLMs."

**Q2: What is LLM-as-a-Judge?**
> Instead of humans manually reviewing LLM outputs, another LLM (the "judge") evaluates and scores the output. DeepEval uses research-backed techniques — QAG (Question-Answer Generation), DAG (Deep Acyclic Graphs), and GEval — to make this reliable and reproducible.

**Q3: What are the 4 core components of a DeepEval evaluation?**
> 1. **Test Case** (LLMTestCase) — the input + output to evaluate
> 2. **Metric** — the scoring rule (e.g., AnswerRelevancyMetric)
> 3. **Dataset** (EvaluationDataset) — a collection of test cases/goldens
> 4. **Golden** — the ideal/expected template for a test case

**Q4: What is the difference between `context` and `retrieval_context`?**
> - `context`: Static ground truth facts (ideal retrieval). Used by HallucinationMetric and ContextualRecall. Represents what SHOULD be retrieved.
> - `retrieval_context`: Dynamic documents actually retrieved by your RAG pipeline at runtime. Used by FaithfulnessMetric and ContextualPrecision. Represents what WAS retrieved.

**Q5: What is a Golden in DeepEval?**
> A Golden is an ideal/expected test case template containing the `input` and optionally `expected_output`. It's like an answer key. You convert Goldens into LLMTestCases by running your LLM app to get the `actual_output`. The Synthesizer generates Goldens, NOT actual_outputs.

**Q6: What is the default threshold for metrics?**
> 0.5 (50%). A metric passes if score >= threshold. Customize per metric: `AnswerRelevancyMetric(threshold=0.7)`. For safety metrics (Bias, Toxicity), the threshold is a MAXIMUM (lower score = better).

**Q7: What does strict_mode do?**
> When `strict_mode=True`, the score becomes binary: 1.0 (perfect) or 0.0 (any imperfection). It also overrides the threshold to 1.0 (or 0.0 for safety metrics).

**Q8: How do DeepEval scores work?**
> All metrics output a score between 0.0 and 1.0, plus a text reason explaining the score. Each metric has its own formula. For example: AnswerRelevancy = Relevant Statements / Total Statements.

**Q9: What are the 3 ways to run evaluations?**
> 1. `evaluate()` function — in Python scripts, notebooks
> 2. `deepeval test run` — CLI with PyTest, best for CI/CD
> 3. `metric.measure()` — standalone for debugging
> (Plus `evals_iterator()` for component-level evaluation with datasets)

**Q10: Can DeepEval work without OpenAI?**
> Yes. You can use Anthropic Claude, Google Gemini, Ollama (local LLMs), Azure OpenAI, LiteLLM, or any custom model by extending `DeepEvalBaseLLM`.

---

### RAG & Metrics (Q11-Q20):

**Q11: Name the 5 RAG metrics and what each measures.**
> 1. **AnswerRelevancy** — Is the answer relevant to the question?
> 2. **Faithfulness** — Does the answer stick to retrieved context (no hallucination)?
> 3. **ContextualPrecision** — Are relevant contexts ranked higher?
> 4. **ContextualRecall** — Did retrieval find all necessary information?
> 5. **ContextualRelevancy** — Is each retrieved chunk actually useful?

**Q12: What is the Faithfulness formula?**
> Faithfulness = Truthful Claims / Total Claims. A claim is "truthful" if it doesn't contradict the `retrieval_context`. Uses QAG internally: extract claims → extract truths → compare.

**Q13: What is the AnswerRelevancy formula?**
> AnswerRelevancy = Relevant Statements / Total Statements. A statement is "relevant" if it relates to the `input` question.

**Q14: What is GEval and when would you use it?**
> GEval is a research-backed framework for creating custom metrics using plain English criteria. Use it for subjective evaluations like correctness, helpfulness, tone — anything not covered by built-in metrics. Available for single-turn, multi-turn, and multimodal evals.

**Q15: What is DAG (Deep Acyclic Graph)?**
> DAG is a decision-tree based LLM-as-a-Judge metric. Unlike GEval's one-pass approach, DAG follows branching logic: "First check format → if correct, check content → then check tone." Best for objective, multi-step criteria that need deterministic evaluation.

**Q16: FaithfulnessMetric vs HallucinationMetric — when to use which?**
> - **FaithfulnessMetric**: For RAG. Uses `retrieval_context`. Checks if output matches retrieved docs. Min threshold.
> - **HallucinationMetric**: For general LLMs. Uses `context` (ground truth). Checks if output contradicts known facts. Max threshold.

**Q17: How do you evaluate an AI agent with DeepEval?**
> Use LLM tracing (`@observe` decorator) to map agent components, then apply:
> - Reasoning layer: `PlanQualityMetric` + `PlanAdherenceMetric` (end-to-end)
> - Action layer: `ToolCorrectnessMetric` + `ArgumentCorrectnessMetric` (component-level on LLM span)
> - Overall: `TaskCompletionMetric` + `StepEfficiencyMetric` (trace-only)

**Q18: What is a ConversationalTestCase?**
> A test case for multi-turn conversations (chatbots). Contains a list of `Turn` objects (role + content + optional retrieval_context per turn). Used with multi-turn metrics like `TurnFaithfulnessMetric`, `ConversationCompletenessMetric`.

**Q19: How do you limit the number of metrics in production?**
> Best practice: max 5 metrics total.
> - 2-3 generic system metrics (e.g., AnswerRelevancy + Faithfulness for RAG)
> - 1-2 custom use-case metrics (e.g., GEval for "helpfulness")
> Force yourself to prioritize what matters most.

**Q20: What is async_mode in metrics?**
> When `async_mode=True` (default), internal steps of a metric run concurrently (e.g., Faithfulness steps 1 and 2 run in parallel). The `measure()` call still blocks the main thread. Use `a_measure()` for fully non-blocking async execution.

---

### Agent & Multi-Turn (Q21-Q30):

**Q21: What are the three layers of agent evaluation?**
> 1. **Reasoning Layer** — Planning and decision-making (PlanQuality, PlanAdherence)
> 2. **Action Layer** — Tool selection and argument generation (ToolCorrectness, ArgumentCorrectness)
> 3. **Overall Execution** — Task completion and efficiency (TaskCompletion, StepEfficiency)

**Q22: What is the difference between end-to-end and component-level agent evals?**
> - **End-to-end**: Metrics analyze the full agent trace (PlanQuality, TaskCompletion). Passed to `evals_iterator(metrics=[...])`.
> - **Component-level**: Metrics evaluate specific components (ToolCorrectness). Attached to `@observe(metrics=[...])` on the LLM component.

**Q23: What is PlanQualityMetric vs PlanAdherenceMetric?**
> - **PlanQuality**: Is the plan logical, complete, and efficient? (Does the agent create good plans?)
> - **PlanAdherence**: Does the agent follow its own plan? (Does it stick to what it planned?)

**Q24: What is TaskCompletionMetric?**
> Evaluates whether the agent successfully accomplished the intended task. Analyzes the full execution trace. This is a trace-only metric — cannot be used standalone, must use with `evals_iterator` or `@observe`.

**Q25: What common agent failure modes does evaluation detect?**
> - Wrong tool selection
> - Incorrect arguments
> - Tools called in wrong order
> - Plan deviation
> - Incomplete tasks
> - Redundant/unnecessary steps
> - Going off-task

**Q26: What multi-turn RAG failure modes exist?**
> - **Context drift**: Retriever fetches increasingly irrelevant docs
> - **Redundant retrieval**: Same chunks fetched repeatedly
> - **Cross-turn hallucination**: Generator mixes info from different turns' contexts

**Q27: How does the ConversationSimulator work?**
> You provide a `model_callback` that takes input + conversation history and returns a `Turn`. The simulator generates realistic multi-turn conversations from conversational goldens, ready for evaluation.

**Q28: What is Arena GEval?**
> A comparative evaluation mode where two LLM outputs are judged against each other. Useful for A/B testing different models or prompt versions.

**Q29: What types of bias does BiasMetric detect?**
> Gender bias, political bias, racial/ethnic bias, and geographical bias. It extracts opinions from the output and classifies each as biased or not biased.

**Q30: What types of toxicity does ToxicityMetric detect?**
> Personal attacks, mockery, hate speech, dismissive statements, and threats/intimidation. It extracts opinions and classifies each as toxic or not toxic.

---

### Production & Advanced (Q31-Q40):

**Q31: How do you integrate DeepEval into a CI/CD pipeline?**
> Create a test file using `assert_test()`, then add `deepeval test run test_file.py` as a step in GitHub Actions/Jenkins/GitLab CI. Set `OPENAI_API_KEY` as a secret. Use `@pytest.mark.parametrize` for dataset-driven tests.

**Q32: What is Confident AI?**
> The cloud platform DeepEval natively integrates with. Provides: cloud test reports, regression testing (green = improved, red = regressed), production monitoring with async evals, dataset management, and team collaboration.

**Q33: How do you generate synthetic test data?**
> Use `Synthesizer()` with four methods:
> - `generate_goldens_from_docs()` — from PDF/TXT/DOCX files
> - `generate_goldens_from_contexts()` — from context strings
> - `generate_goldens_from_scratch()` — no data needed
> - `generate_goldens_from_goldens()` — augment existing data

**Q34: What is the Synthesizer pipeline?**
> 4 steps: (1) Input Generation → (2) Filtration (quality check) → (3) Evolution (increase complexity) → (4) Styling (format/language). Uses the Evol-Instruct data evolution method.

**Q35: What are the 7 Evolution types?**
> REASONING, MULTICONTEXT, CONCRETIZING, CONSTRAINED, COMPARATIVE, HYPOTHETICAL, IN_BREADTH. Only MULTICONTEXT, CONCRETIZING, CONSTRAINED, and COMPARATIVE stick to the original context.

**Q36: What is component-level evaluation?**
> Evaluating individual internal components (retriever, generator, tools) separately using the `@observe` decorator and tracing, rather than treating the LLM app as a black box.

**Q37: What is verbose_mode?**
> Setting `verbose_mode=True` prints the intermediate steps of metric calculation to the console. Essential for debugging why a metric scored a certain way.

**Q38: How do you handle evaluation errors in production?**
> Use `ErrorConfig(ignore_errors=True)` to prevent one failing test case from crashing the entire evaluation. Failed cases are logged but don't block others.

**Q39: What are online evals?**
> Evaluations that run in production on real user traffic. Using `@observe()` + Confident AI's `metric_collection`, every LLM interaction is automatically evaluated asynchronously without affecting response latency.

**Q40: What is the difference between end-to-end and component-level evals in production?**
> - **Trace-level (E2E)**: Evaluates the overall system interaction
> - **Span-level (Component)**: Evaluates specific components within a trace
> - **Thread-level (Conversation)**: Evaluates multi-turn conversations over time

---

### Advanced & Optimization (Q41-Q50):

**Q41: How do you customize metric prompts?**
> Create a subclass of the metric's template class (e.g., `AnswerRelevancyTemplate`), override the relevant method, and pass it via `evaluation_template` parameter. This is especially useful for custom LLMs that need different prompt structures.

**Q42: How do you build a custom metric from scratch?**
> Subclass `BaseMetric`, implement `measure()`, `a_measure()`, `is_successful()`, and `__name__`. You can use DeepEval's `Scorer` module for traditional NLP scores (ROUGE, BLEU).

**Q43: What is the Scorer module?**
> A utility module providing traditional NLP scoring methods like ROUGE, BLEU, and BLEURT. Available but not recommended for LLM evaluation since these scorers lack reasoning capability.

**Q44: How do you optimize evaluation costs?**
> - Use cheaper models (gpt-4o-mini) for simple metrics
> - Enable caching with `CacheConfig(enable=True)`
> - Limit concurrency with `AsyncConfig(max_concurrent=5)`
> - Track costs via `token_cost` parameter
> - Use `strict_mode=True` when fine-grained scores aren't needed

**Q45: What is the RAG Triad?**
> The three essential metrics for RAG evaluation: Answer Relevancy (is the answer relevant?), Faithfulness (does it stick to facts?), and Context Relevancy (is the retrieved context useful?). These three together give comprehensive RAG quality assessment.

**Q46: What is reference-based vs referenceless evaluation?**
> - **Reference-based**: Needs ground truth data (expected_output, context). Best for development.
> - **Referenceless**: Works without labeled data. Required for production monitoring since real user traffic has no pre-labeled ground truth.

**Q47: How do you track hyperparameters across test runs?**
> Pass a `hyperparameters` dict to `evaluate()` or use `@deepeval.log_hyperparameters` decorator. Track model, temperature, prompt version, chunk size, top-K, embedding model. Compare runs on Confident AI to find optimal configurations.

**Q48: What is DeepTeam?**
> DeepTeam (trydeepteam.com) is a companion project for red teaming and adversarial testing of LLM applications. It helps identify safety vulnerabilities and test guardrails.

**Q49: What integrations does DeepEval support?**
> **Models**: OpenAI, Azure OpenAI, Anthropic, Gemini, Ollama, LiteLLM, custom LLMs.
> **Frameworks**: LangChain, LangGraph, LlamaIndex, CrewAI, Hugging Face.
> **CI/CD**: GitHub Actions, Jenkins, GitLab CI.
> **Platform**: Confident AI for cloud reports and monitoring.

**Q50: What should you read/do after learning DeepEval basics?**
> 1. Follow a use-case quickstart (RAG, Agents, or Chatbots)
> 2. Set up LLM tracing for component-level evals
> 3. Generate synthetic datasets with the Synthesizer
> 4. Integrate with your CI/CD pipeline
> 5. Set up production monitoring with Confident AI
> 6. Join the DeepEval Discord for community support

---

## 38. Quick Reference Links

| Resource | URL |
|----------|-----|
| Official Docs | https://deepeval.com/docs/getting-started |
| Metrics Overview | https://deepeval.com/docs/metrics-introduction |
| GitHub Repository | https://github.com/confident-ai/deepeval |
| RAG Evaluation Guide | https://deepeval.com/guides/guides-rag-evaluation |
| RAG Triad Guide | https://deepeval.com/guides/guides-rag-triad |
| Agent Evaluation Guide | https://deepeval.com/guides/guides-ai-agent-evaluation |
| Agent Metrics Guide | https://deepeval.com/guides/guides-ai-agent-evaluation-metrics |
| Multi-Turn Evaluation | https://deepeval.com/guides/guides-multi-turn-evaluation |
| Multi-Turn Simulation | https://deepeval.com/guides/guides-multi-turn-simulation |
| Synthetic Data Guide | https://deepeval.com/guides/guides-using-synthetic-data |
| Custom LLMs Guide | https://deepeval.com/guides/guides-using-custom-llms |
| LLM Observability | https://deepeval.com/guides/guides-llm-observability |
| GEval Metric | https://deepeval.com/docs/metrics-llm-evals |
| DAG Metric | https://deepeval.com/docs/metrics-dag |
| Golden Synthesizer | https://deepeval.com/docs/golden-synthesizer |
| Prompt Optimization | https://deepeval.com/docs/prompt-optimization-introduction |
| Benchmarks | https://deepeval.com/docs/benchmarks-introduction |
| Integrations | https://deepeval.com/integrations |
| Confident AI Platform | https://www.confident-ai.com/ |
| Confident AI Docs | https://www.confident-ai.com/docs |
| DeepTeam (Red Teaming) | https://trydeepteam.com/ |
| Discord Community | https://discord.com/invite/a3K9c8GRGt |
| Design Philosophy | https://deepeval.com/docs/introduction-design-philosophy |
| Comparisons | https://deepeval.com/docs/introduction-comparisons |
| CI/CD Guide | https://deepeval.com/docs/evaluation-unit-testing-in-ci-cd |
| Flags & Configs | https://deepeval.com/docs/evaluation-flags-and-configs |
| Conversation Simulator | https://deepeval.com/docs/conversation-simulator |
| End-to-End Evals | https://deepeval.com/docs/evaluation-end-to-end-llm-evals |
| Component-Level Evals | https://deepeval.com/docs/evaluation-component-level-llm-evals |

---

> **Last Updated**: May 2026 | **DeepEval Version**: Latest | **License**: Apache 2.0