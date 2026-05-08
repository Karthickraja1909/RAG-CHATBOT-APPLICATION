"""
GenAI RAG System — Streamlit Chat Interface
Production-level UI with chat history, source citations, and system metrics.

Run: streamlit run app_ui.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from pipeline.rag_pipeline import RAGPipeline
from schemas.pipeline import QueryRequest


# ─── Page Configuration ───────────────────────────────────────────
st.set_page_config(
    page_title="GenAI RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    /* Source cards */
    .source-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        color: white;
        font-size: 0.85rem;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    .source-card .source-title {
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 4px;
    }
    .source-card .source-score {
        opacity: 0.85;
        font-size: 0.75rem;
    }
    .source-card .source-content {
        margin-top: 6px;
        font-size: 0.8rem;
        opacity: 0.9;
        border-top: 1px solid rgba(255,255,255,0.3);
        padding-top: 6px;
    }

    /* Metrics bar */
    .metric-box {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .metric-box .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .metric-box .metric-label {
        font-size: 0.75rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .app-header p {
        margin: 5px 0 0 0;
        opacity: 0.8;
        font-size: 0.9rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #f8f9fa;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #1a1a2e;
    }

    /* Status indicator */
    .status-online {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #28a745;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Session State ─────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "avg_response_time" not in st.session_state:
    st.session_state.avg_response_time = 0.0
if "response_times" not in st.session_state:
    st.session_state.response_times = []


# ─── Initialize Pipeline ──────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    """Load RAG pipeline (cached across reruns)."""
    try:
        pipeline = RAGPipeline()
        return pipeline
    except Exception as e:
        return None


def get_pipeline():
    """Get or initialize the RAG pipeline."""
    if st.session_state.pipeline is None:
        st.session_state.pipeline = load_pipeline()
    return st.session_state.pipeline


# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    settings = get_settings()

    st.markdown("### ⚙️ System Configuration")
    st.divider()

    # Provider info
    if settings.use_openrouter and settings.openrouter_api_key:
        provider = "OpenRouter"
        model = settings.openrouter_model
        embed_model = settings.openrouter_embedding_model
    elif settings.azure_openai_endpoint:
        provider = "Azure OpenAI"
        model = settings.azure_openai_deployment or "N/A"
        embed_model = settings.openai_embedding_model
    else:
        provider = "OpenAI"
        model = settings.openai_model
        embed_model = settings.openai_embedding_model

    st.markdown(f"""
    | Parameter | Value |
    |-----------|-------|
    | **Provider** | {provider} |
    | **LLM Model** | `{model}` |
    | **Embedding** | `{embed_model}` |
    | **Temperature** | {settings.openai_temperature} |
    | **Max Tokens** | {settings.openai_max_tokens} |
    | **Top-K** | {settings.similarity_top_k} |
    | **Threshold** | {settings.similarity_threshold} |
    | **Chunk Size** | {settings.chunk_size} |
    """)

    st.divider()

    # Vector store stats
    st.markdown("### 📊 Vector Store")
    pipeline = get_pipeline()
    if pipeline:
        try:
            stats = pipeline.vector_store.get_stats()
            col1, col2 = st.columns(2)
            col1.metric("Vectors", stats.get("total_vectors", 0))
            col2.metric("Documents", stats.get("total_documents", 0))
        except Exception:
            st.info("No vector store loaded")
    else:
        st.error("Pipeline not initialized")

    st.divider()

    # Session stats
    st.markdown("### 📈 Session Stats")
    col1, col2 = st.columns(2)
    col1.metric("Queries", st.session_state.total_queries)
    if st.session_state.response_times:
        avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
        col2.metric("Avg Time", f"{avg_time:.0f}ms")
    else:
        col2.metric("Avg Time", "—")

    st.divider()

    # Actions
    st.markdown("### 🔧 Actions")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.response_times = []
        st.rerun()

    if st.button("🔄 Reload Pipeline", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.pipeline = None
        st.rerun()


# ─── Main Content ─────────────────────────────────────────────────
# Header
st.markdown("""
<div class="app-header">
    <h1>🧠 GenAI RAG System</h1>
    <p><span class="status-online"></span>Production RAG Pipeline — Ask questions about your knowledge base</p>
</div>
""", unsafe_allow_html=True)

# Check pipeline status
pipeline = get_pipeline()
if not pipeline:
    st.error("""
    ⚠️ **RAG Pipeline failed to initialize.**

    Please check:
    1. `.env` file exists with valid API keys
    2. Vector store is populated (`python -m scripts.ingest_documents`)
    3. Check logs for detailed error
    """)
    st.stop()


# ─── Chat Interface ───────────────────────────────────────────────
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🧠"):
        st.markdown(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            sources = message["sources"]
            if sources:
                with st.expander(f"📚 Sources ({len(sources)} chunks) — {message.get('time_ms', 0):.0f}ms", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">📄 Source {i}: {src['source']}</div>
                            <div class="source-score">Similarity: {src['score']:.3f}</div>
                            <div class="source-content">{src['content'][:300]}{'...' if len(src['content']) > 300 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)


# ─── Chat Input ───────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your knowledge base..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Searching knowledge base & generating response..."):
            try:
                request = QueryRequest(query=prompt)
                response = pipeline.query(request)

                # Display answer
                st.markdown(response.answer)

                # Track metrics
                st.session_state.total_queries += 1
                st.session_state.response_times.append(response.total_time_ms)

                # Source details
                sources_data = []
                if response.sources:
                    sources_data = [
                        {
                            "source": src.source,
                            "score": src.score,
                            "content": src.content,
                            "chunk_id": src.chunk_id,
                        }
                        for src in response.sources
                    ]

                    with st.expander(f"📚 Sources ({len(response.sources)} chunks) — {response.total_time_ms:.0f}ms", expanded=False):
                        for i, src in enumerate(response.sources, 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-title">📄 Source {i}: {src.source}</div>
                                <div class="source-score">Similarity: {src.score:.3f}</div>
                                <div class="source-content">{src.content[:300]}{'...' if len(src.content) > 300 else ''}</div>
                            </div>
                            """, unsafe_allow_html=True)

                # Save to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": sources_data,
                    "time_ms": response.total_time_ms,
                })

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "time_ms": 0,
                })


# ─── Footer ───────────────────────────────────────────────────────
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"🏷️ {settings.app_name} v{settings.app_version}")
with col2:
    st.caption(f"🌐 Provider: {provider} | Model: {model}")
with col3:
    st.caption(f"📦 Environment: {settings.environment}")