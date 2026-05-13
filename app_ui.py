"""
GenAI RAG System — Enterprise Streamlit Application.
Production-grade UI with chat, observability dashboard, and system configuration.

Run: streamlit run app_ui.py
"""

import html as html_lib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from pipeline.rag_pipeline import RAGPipeline
from schemas.pipeline import QueryRequest

# ─── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="GenAI RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design System (CSS) ─────────────────────────────────────────
_CSS = """
<style>
/* ── Layout ────────────────────────────────────────────────────── */
.main .block-container {
    padding: 1.5rem 2rem 2rem; max-width: 1280px;
}

/* ── Page header ───────────────────────────────────────────────── */
.page-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #334155 100%);
    border-radius: 14px; padding: 28px 34px; margin-bottom: 1.8rem;
    color: #f8fafc; position: relative; overflow: hidden;
}
.page-header::after {
    content: ''; position: absolute; top: -40%; right: -8%;
    width: 260px; height: 260px; border-radius: 50%;
    background: rgba(99,102,241,0.12);
}
.page-header h1 {
    margin: 0; font-size: 1.75rem; font-weight: 700;
    letter-spacing: -0.02em;
}
.page-header .subtitle {
    margin: 6px 0 0; font-size: 0.88rem; opacity: 0.7;
    font-weight: 400;
}
.status-dot {
    display: inline-block; width: 9px; height: 9px;
    border-radius: 50%; margin-right: 7px; vertical-align: middle;
    animation: pulse 2.4s ease-in-out infinite;
}
.status-dot.online  { background: #22c55e; box-shadow: 0 0 6px #22c55e88; }
.status-dot.offline { background: #ef4444; box-shadow: 0 0 6px #ef444488; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.45} }

/* ── KPI cards ─────────────────────────────────────────────────── */
.kpi-grid { display: grid; gap: 14px; }
.kpi-grid.cols-4 { grid-template-columns: repeat(4, 1fr); }
.kpi-grid.cols-3 { grid-template-columns: repeat(3, 1fr); }
.kpi-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 18px 20px;
    transition: box-shadow .2s, transform .15s;
}
.kpi-card:hover {
    box-shadow: 0 4px 16px rgba(15,23,42,0.08);
    transform: translateY(-2px);
}
.kpi-value {
    font-size: 1.65rem; font-weight: 700; color: #0f172a;
    line-height: 1.15;
}
.kpi-label {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: #64748b; margin-bottom: 4px;
}
.kpi-accent { border-left: 4px solid #6366f1; }
.kpi-success { border-left: 4px solid #22c55e; }
.kpi-warn    { border-left: 4px solid #f59e0b; }
.kpi-info    { border-left: 4px solid #3b82f6; }

/* ── Source cards ──────────────────────────────────────────────── */
.src-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 14px 18px; margin: 8px 0;
    transition: border-color .2s;
}
.src-card:hover { border-color: #6366f1; }
.src-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 6px;
}
.src-title { font-weight: 600; font-size: 0.88rem; color: #1e293b; }
.src-badge {
    font-size: 0.7rem; font-weight: 600; padding: 2px 8px;
    border-radius: 999px; background: #6366f1; color: white;
}
.src-preview {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 12px 14px; margin-top: 8px;
    font-family: 'Menlo','Monaco','Consolas',monospace;
    font-size: 0.8rem; line-height: 1.55; color: #334155;
    white-space: pre-wrap; word-wrap: break-word;
    max-height: 240px; overflow-y: auto;
}

/* ── Follow-up pills ──────────────────────────────────────────── */
.followup-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.followup-pill {
    font-size: 0.82rem; padding: 6px 14px;
    background: #eef2ff; border: 1px solid #c7d2fe;
    border-radius: 999px; color: #4338ca;
    transition: background .15s;
}
.followup-pill:hover { background: #c7d2fe; }

/* ── Config table ─────────────────────────────────────────────── */
.cfg-table { width: 100%; border-collapse: separate; border-spacing: 0; }
.cfg-table th {
    text-align: left; font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.06em; color: #64748b; padding: 10px 14px;
    border-bottom: 2px solid #e2e8f0;
}
.cfg-table td {
    padding: 10px 14px; font-size: 0.88rem; color: #1e293b;
    border-bottom: 1px solid #f1f5f9;
}
.cfg-table tr:hover td { background: #f8fafc; }
.cfg-badge-on {
    font-size: 0.72rem; padding: 2px 10px; border-radius: 999px;
    background: #dcfce7; color: #166534; font-weight: 600;
}
.cfg-badge-off {
    font-size: 0.72rem; padding: 2px 10px; border-radius: 999px;
    background: #fee2e2; color: #991b1b; font-weight: 600;
}
.cfg-mono {
    font-family: 'Menlo','Monaco','Consolas',monospace;
    font-size: 0.82rem; background: #f1f5f9; padding: 2px 6px;
    border-radius: 4px;
}

/* ── Sidebar polish ───────────────────────────────────────────── */
[data-testid="stSidebar"] { background: #f8fafc; }
[data-testid="stSidebar"] .stRadio > div { gap: 2px; }
[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px; border-radius: 8px; font-weight: 500;
    transition: background .15s;
}
[data-testid="stSidebar"] .stRadio label:hover { background: #e2e8f0; }

/* ── Chat tweaks ──────────────────────────────────────────────── */
.stChatMessage { border-radius: 14px; margin-bottom: 0.8rem; }

/* ── Responsive ───────────────────────────────────────────────── */
@media (max-width: 768px) {
    .kpi-grid.cols-4, .kpi-grid.cols-3 { grid-template-columns: 1fr 1fr; }
    .page-header { padding: 20px 22px; }
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────
_DEFAULTS = {
    "messages": [],
    "pipeline": None,
    "total_queries": 0,
    "response_times": [],
    "token_history": [],
    "page": "chat",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─── Pipeline ────────────────────────────────────────────────────
@st.cache_resource
def _load_pipeline():
    try:
        return RAGPipeline()
    except Exception:
        return None


def _pipeline():
    if st.session_state.pipeline is None:
        st.session_state.pipeline = _load_pipeline()
    return st.session_state.pipeline


# ─── Helpers ──────────────────────────────────────────────────────
def _provider_info():
    s = get_settings()
    if s.use_openrouter and s.openrouter_api_key:
        return "OpenRouter", s.openrouter_model
    if s.azure_openai_endpoint:
        return "Azure OpenAI", s.azure_openai_deployment or "N/A"
    return "OpenAI", s.openai_model


def _badge(enabled: bool) -> str:
    if enabled:
        return '<span class="cfg-badge-on">Enabled</span>'
    return '<span class="cfg-badge-off">Disabled</span>'


def _safe(text: str) -> str:
    return html_lib.escape(text)


def _render_kpi(label: str, value, accent: str = "accent") -> str:
    return (
        f'<div class="kpi-card kpi-{accent}">'
        f'<div class="kpi-label">{_safe(label)}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    settings = get_settings()
    provider, model_name = _provider_info()

    # ── Brand ─────────────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;padding:8px 0 4px">'
        '<span style="font-size:1.6rem">🧠</span><br>'
        f'<span style="font-weight:700;font-size:1rem;color:#0f172a">{settings.app_name}</span><br>'
        f'<span style="font-size:0.7rem;color:#94a3b8">v{settings.app_version} · {settings.environment}</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Navigation ────────────────────────────────────────────────
    page = st.radio(
        "Navigation",
        ["💬  Chat", "📊  Dashboard", "⚙️  Settings"],
        label_visibility="collapsed",
    )
    if "Chat" in page:
        st.session_state.page = "chat"
    elif "Dashboard" in page:
        st.session_state.page = "dashboard"
    else:
        st.session_state.page = "settings"

    st.divider()

    # ── Connection info ───────────────────────────────────────────
    pipe = _pipeline()
    status_cls = "online" if pipe else "offline"
    status_txt = "Connected" if pipe else "Disconnected"
    st.markdown(
        f'<span class="status-dot {status_cls}"></span>'
        f'<span style="font-size:0.82rem;font-weight:500;color:#334155">{status_txt}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"**Provider:** {provider}  \n**Model:** `{model_name}`")

    st.divider()

    # ── Session metrics ───────────────────────────────────────────
    st.markdown(
        '<span style="font-size:0.72rem;font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.06em;color:#64748b">Session Metrics</span>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("Queries", st.session_state.total_queries)
    if st.session_state.response_times:
        avg = sum(st.session_state.response_times) / len(st.session_state.response_times)
        c2.metric("Avg Latency", f"{avg:.0f}ms")
    else:
        c2.metric("Avg Latency", "—")

    st.divider()

    # ── Actions ───────────────────────────────────────────────────
    ac1, ac2 = st.columns(2)
    with ac1:
        if st.button("🗑️ Clear", use_container_width=True, help="Clear chat history"):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            st.session_state.response_times = []
            st.session_state.token_history = []
            st.rerun()
    with ac2:
        if st.button("🔄 Reload", use_container_width=True, help="Reload RAG pipeline"):
            st.cache_resource.clear()
            st.session_state.pipeline = None
            st.rerun()


# ═══════════════════════════════════════════════════════════════════
#  PAGE: CHAT
# ═══════════════════════════════════════════════════════════════════
def _render_sources(sources: list[dict], time_ms: float = 0):
    """Render source cards with expandable content preview."""
    label = f"📚  {len(sources)} sources retrieved · {time_ms:.0f} ms"
    with st.expander(label, expanded=False):
        for i, src in enumerate(sources, 1):
            score = src.get("score", 0)
            badge_bg = "#22c55e" if score >= 0.7 else ("#f59e0b" if score >= 0.4 else "#ef4444")
            st.markdown(
                f'<div class="src-card">'
                f'<div class="src-header">'
                f'<span class="src-title">📄 {_safe(src.get("source", "unknown"))}</span>'
                f'<span class="src-badge" style="background:{badge_bg}">{score:.3f}</span>'
                f'</div>'
                f'<div class="src-preview">{_safe(src.get("content", ""))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_followups(followups: list[str]):
    """Render follow-up suggestions as styled pills."""
    if not followups:
        return
    pills = "".join(f'<span class="followup-pill">💡 {_safe(fq)}</span>' for fq in followups)
    st.markdown(f'<div class="followup-row">{pills}</div>', unsafe_allow_html=True)


def render_chat_page():
    st.markdown(
        '<div class="page-header">'
        '<h1>🧠 Knowledge Assistant</h1>'
        '<p class="subtitle">'
        '<span class="status-dot online"></span>'
        'Ask questions about your knowledge base — powered by RAG pipeline'
        '</p></div>',
        unsafe_allow_html=True,
    )

    pipeline = _pipeline()
    if not pipeline:
        st.error("**RAG Pipeline failed to initialize.** Verify `.env` configuration and vector store.")
        st.stop()

    # ── Chat history ──────────────────────────────────────────────
    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🧠"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_followups(msg.get("followups", []))
                sources = msg.get("sources", [])
                if sources:
                    _render_sources(sources, msg.get("time_ms", 0))

    # ── Chat input ────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about your knowledge base…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Retrieving & generating…"):
                try:
                    response = pipeline.query(QueryRequest(query=prompt))

                    st.markdown(response.answer)

                    # Track
                    st.session_state.total_queries += 1
                    st.session_state.response_times.append(response.total_time_ms)
                    st.session_state.token_history.append(response.token_usage)

                    followups = response.followup_questions
                    _render_followups(followups)

                    sources_data = []
                    if response.sources:
                        sources_data = [
                            {"source": s.source, "score": s.score, "content": s.content, "chunk_id": s.chunk_id}
                            for s in response.sources
                        ]
                        _render_sources(sources_data, response.total_time_ms)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": sources_data,
                        "followups": followups,
                        "time_ms": response.total_time_ms,
                        "token_usage": response.token_usage,
                    })

                except Exception as e:
                    err = f"An error occurred: {e}"
                    st.error(err)
                    st.session_state.messages.append({
                        "role": "assistant", "content": err,
                        "sources": [], "followups": [], "time_ms": 0, "token_usage": {},
                    })

    # ── Export bar ─────────────────────────────────────────────────
    if st.session_state.messages:
        st.divider()
        c1, c2, c3 = st.columns([1, 1, 2])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        with c1:
            st.download_button(
                "⬇ Export JSON",
                data=json.dumps(st.session_state.messages, indent=2, default=str),
                file_name=f"chat_{ts}.json",
                mime="application/json",
                use_container_width=True,
            )
        with c2:
            md = "\n\n---\n\n".join(
                f"{'**User**' if m['role'] == 'user' else '**Assistant**'}: {m['content']}"
                for m in st.session_state.messages
            )
            st.download_button(
                "⬇ Export Markdown",
                data=md,
                file_name=f"chat_{ts}.md",
                mime="text/markdown",
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════
def render_dashboard_page():
    st.markdown(
        '<div class="page-header">'
        '<h1>📊 Observability Dashboard</h1>'
        '<p class="subtitle">Token usage, latency, vector store, and cache metrics</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    pipeline = _pipeline()
    if not pipeline:
        st.warning("Pipeline not initialized — start a chat first.")
        return

    tok = pipeline.llm_service.token_usage.summary

    # ── Top KPI strip ─────────────────────────────────────────────
    st.markdown(
        '<div class="kpi-grid cols-4">'
        + _render_kpi("Prompt Tokens", f"{tok['prompt_tokens']:,}", "accent")
        + _render_kpi("Completion Tokens", f"{tok['completion_tokens']:,}", "info")
        + _render_kpi("Total Tokens", f"{tok['total_tokens']:,}", "success")
        + _render_kpi("API Requests", tok["request_count"], "warn")
        + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")  # spacer

    # ── Charts ────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("##### Tokens Per Query")
        if st.session_state.token_history:
            df = pd.DataFrame([
                {"Query": i, "Prompt": u.get("prompt_tokens", 0), "Completion": u.get("completion_tokens", 0)}
                for i, u in enumerate(st.session_state.token_history, 1)
            ]).set_index("Query")
            st.bar_chart(df, color=["#6366f1", "#22c55e"])
        else:
            st.info("No queries yet — data appears after your first question.", icon="📈")

    with chart_col2:
        st.markdown("##### Response Latency (ms)")
        if st.session_state.response_times:
            df_rt = pd.DataFrame({"Latency": st.session_state.response_times})
            st.area_chart(df_rt, color="#6366f1")
        else:
            st.info("No queries yet — data appears after your first question.", icon="⏱️")

    st.divider()

    # ── Vector Store & Caches ─────────────────────────────────────
    st.markdown("##### Infrastructure")
    infra1, infra2, infra3 = st.columns(3)

    with infra1:
        st.markdown("**Vector Store**")
        try:
            vs = pipeline.vector_store.get_stats()
            st.markdown(
                '<div class="kpi-grid cols-3">'
                + _render_kpi("Vectors", vs.get("total_vectors", 0), "accent")
                + _render_kpi("Documents", vs.get("total_documents", 0), "info")
                + _render_kpi("Dimension", vs.get("dimension", 0), "success")
                + '</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            st.caption("No vector store data available.")

    with infra2:
        st.markdown("**Response Cache**")
        cache = pipeline.get_cache_stats()
        if cache:
            for k, v in cache.items():
                st.caption(f"`{k}`: **{v}**")
        else:
            st.caption("Caching is disabled.")

    with infra3:
        st.markdown("**Embedding Cache**")
        ecache = pipeline.vector_store.embedding_service.get_cache_stats()
        if ecache:
            for k, v in ecache.items():
                st.caption(f"`{k}`: **{v}**")
        else:
            st.caption("Caching is disabled.")


# ═══════════════════════════════════════════════════════════════════
#  PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════════
def render_settings_page():
    st.markdown(
        '<div class="page-header">'
        '<h1>⚙️ System Configuration</h1>'
        '<p class="subtitle">Read-only view of active environment settings</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    s = get_settings()
    provider, model_name = _provider_info()

    tab_llm, tab_retrieval, tab_features, tab_eval = st.tabs([
        "🤖 LLM", "🔍 Retrieval", "🚀 Features", "📋 Evaluation",
    ])

    # ── LLM Tab ───────────────────────────────────────────────────
    with tab_llm:
        rows = [
            ("Provider", f'<span class="cfg-mono">{_safe(provider)}</span>'),
            ("Model", f'<span class="cfg-mono">{_safe(model_name)}</span>'),
            ("Temperature", str(s.openai_temperature)),
            ("Max Tokens", f"{s.openai_max_tokens:,}"),
            ("Connection Pool", str(s.llm_connection_pool_size)),
            ("Request Timeout", f"{s.llm_request_timeout}s"),
        ]
        _config_table(rows)

    # ── Retrieval Tab ─────────────────────────────────────────────
    with tab_retrieval:
        rows = [
            ("Top-K", str(s.similarity_top_k)),
            ("Similarity Threshold", str(s.similarity_threshold)),
            ("Chunk Size", f"{s.chunk_size:,} chars"),
            ("Chunk Overlap", f"{s.chunk_overlap:,} chars"),
            ("Hybrid Search", _badge(s.hybrid_search_enabled)),
            ("Hybrid Alpha", f"{s.hybrid_search_alpha} <span style='color:#94a3b8;font-size:0.75rem'>(1.0 = dense, 0.0 = BM25)</span>"),
            ("Reranking", _badge(s.reranking_enabled)),
            ("Reranker Top-N", str(s.reranker_top_n)),
            ("Reranker Threshold", str(s.reranker_relevance_threshold)),
        ]
        _config_table(rows)

    # ── Features Tab ──────────────────────────────────────────────
    with tab_features:
        rows = [
            ("Response Cache", f'{_badge(s.response_cache_enabled)} <span style="color:#94a3b8;font-size:0.75rem">max={s.response_cache_max_size} · ttl={s.response_cache_ttl_seconds}s</span>'),
            ("Embedding Cache", f'{_badge(s.embedding_cache_enabled)} <span style="color:#94a3b8;font-size:0.75rem">max={s.embedding_cache_max_size}</span>'),
            ("Follow-up Questions", f'{_badge(s.followup_enabled)} <span style="color:#94a3b8;font-size:0.75rem">count={s.followup_count}</span>'),
            ("Metadata Enrichment", _badge(s.metadata_enrichment_enabled)),
        ]
        _config_table(rows)

    # ── Evaluation Tab ────────────────────────────────────────────
    with tab_eval:
        rows = [
            ("Eval Model", f'<span class="cfg-mono">{_safe(s.eval_model)}</span>'),
            ("Eval Threshold", str(s.eval_threshold)),
            ("Frameworks", f'<span class="cfg-mono">{_safe(s.eval_frameworks)}</span>'),
            ("DeepEval Metrics", f'<span class="cfg-mono">{_safe(s.deepeval_metrics)}</span>'),
            ("RAGAS Metrics", f'<span class="cfg-mono">{_safe(s.ragas_metrics)}</span>'),
            ("Custom Metrics", _badge(s.eval_custom_metrics_enabled)),
            ("Batch Size", str(s.eval_batch_size)),
            ("Auto Generate Dataset", _badge(s.eval_auto_generate_dataset)),
        ]
        _config_table(rows)


def _config_table(rows: list[tuple[str, str]]):
    """Render a styled two-column configuration table."""
    body = "".join(
        f'<tr><td style="font-weight:500">{label}</td><td>{value}</td></tr>'
        for label, value in rows
    )
    st.markdown(
        f'<table class="cfg-table"><thead><tr><th>Parameter</th><th>Value</th></tr></thead>'
        f'<tbody>{body}</tbody></table>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
#  PAGE ROUTER
# ═══════════════════════════════════════════════════════════════════
_PAGES = {"chat": render_chat_page, "dashboard": render_dashboard_page, "settings": render_settings_page}
_PAGES[st.session_state.page]()
