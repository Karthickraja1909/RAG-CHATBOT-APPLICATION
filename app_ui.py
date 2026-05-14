"""
GenAI RAG System — Enterprise Streamlit Application.
Run: streamlit run app_ui.py
"""

import html as html_lib
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from pipeline.rag_pipeline import RAGPipeline
from schemas.pipeline import QueryRequest

# --- Page Configuration ---
st.set_page_config(
    page_title="GenAI RAG System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Design System ---
_CSS = """
<style>
/* ---- Layout ---- */
.main .block-container { padding: 0.5rem 1rem 1rem; max-width: 100%; }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #171717; min-width: 280px; max-width: 300px;
}
[data-testid="stSidebar"] * { color: #d1d5db !important; }
[data-testid="stSidebar"] .stButton > button {
    background: transparent; border: 1px solid #333;
    color: #d1d5db !important; border-radius: 8px;
    font-size: 0.82rem; font-weight: 500;
    padding: 8px 14px; width: 100%;
    transition: background .15s;
    text-align: left; justify-content: flex-start;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #2a2a2a; border-color: #444;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #2a2a2a; border-color: #555; color: #fff !important;
}
[data-testid="stSidebar"] hr {
    border-color: #2a2a2a !important; margin: 6px 0;
}

/* Tighten column gap inside sidebar for conv rows */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    gap: 0.25rem !important;
    align-items: center;
}

/* Delete button — small, inline */
.del-btn button {
    background: transparent !important; border: 1px solid #333 !important;
    color: #6b7280 !important; padding: 6px 10px !important;
    font-size: 0.75rem !important; min-height: 0 !important;
    border-radius: 6px !important;
    transition: all .15s;
}
.del-btn button:hover {
    color: #ef4444 !important;
    background: rgba(239,68,68,0.12) !important;
    border-color: #ef4444 !important;
}

/* Nav row at bottom */
.nav-row { display: flex; gap: 6px; padding: 4px 0; }
.nav-row button {
    flex: 1; font-size: 0.78rem !important;
    padding: 8px 6px !important; text-align: center !important;
    justify-content: center !important;
}

/* Chat history date headers */
.chat-history-date {
    font-size: 0.65rem; color: #6b7280; font-weight: 600;
    padding: 12px 4px 4px; text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ---- Welcome ---- */
.welcome-wrap {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 6rem 2rem 3rem; text-align: center;
}
.welcome-title {
    font-size: 1.5rem; font-weight: 700; color: #1a1a1a;
    margin: 0 0 8px;
}
.welcome-sub {
    font-size: 0.82rem; color: #6b7280; margin: 0 0 2.5rem;
    max-width: 520px; line-height: 1.55;
}
.typewriter-line {
    display: inline-block; overflow: hidden; white-space: nowrap;
    border-right: 2px solid #6366f1;
    font-family: 'Menlo','Monaco','Consolas',monospace;
    font-size: 0.85rem; color: #6b7280;
    width: 0;
    animation: tw-type 2.8s steps(42, end) forwards,
               tw-blink 0.7s step-end infinite;
}
@keyframes tw-type { from { width: 0; } to { width: 42ch; } }
@keyframes tw-blink {
    from, to { border-color: transparent; }
    50% { border-color: #6366f1; }
}
.suggest-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 10px; max-width: 560px; width: 100%;
}
.suggest-card {
    background: #fff; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 14px 16px;
    text-align: left; cursor: pointer;
    transition: border-color .15s, box-shadow .15s;
}
.suggest-card:hover {
    border-color: #6366f1; box-shadow: 0 2px 8px rgba(99,102,241,0.1);
}
.suggest-label {
    font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.04em; color: #6366f1; margin-bottom: 4px;
}
.suggest-text { font-size: 0.82rem; color: #374151; line-height: 1.4; }

/* ---- Chat messages ---- */
.stChatMessage {
    border-radius: 0; border: none; box-shadow: none;
    max-width: 780px; margin: 0 auto; font-size: 0.88rem;
}
.msg-time {
    font-size: 0.62rem; color: #9ca3af; margin-top: 4px;
}

/* ---- Sources ---- */
.src-card {
    background: #f9fafb; border: 1px solid #e5e7eb;
    border-radius: 8px; padding: 10px 14px; margin: 6px 0;
    transition: border-color .15s;
}
.src-card:hover { border-color: #6366f1; }
.src-header {
    display: flex; justify-content: space-between;
    align-items: center; margin-bottom: 4px;
}
.src-title { font-weight: 600; font-size: 0.78rem; color: #1a1a1a; }
.src-badge {
    font-size: 0.65rem; font-weight: 600; padding: 2px 8px;
    border-radius: 999px; background: #6366f1; color: #fff;
}
.src-preview {
    background: #fff; border: 1px solid #e5e7eb;
    border-radius: 6px; padding: 8px 10px; margin-top: 6px;
    font-family: 'Menlo','Monaco','Consolas',monospace;
    font-size: 0.72rem; line-height: 1.5; color: #374151;
    white-space: pre-wrap; word-wrap: break-word;
    max-height: 200px; overflow-y: auto;
}

/* ---- Follow-up pills ---- */
.followup-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.followup-pill {
    font-size: 0.75rem; padding: 6px 14px;
    background: #f3f4f6; border: 1px solid #e5e7eb;
    border-radius: 999px; color: #374151;
    transition: background .12s;
}
.followup-pill:hover { background: #e5e7eb; }

/* ---- KPI cards ---- */
.kpi-grid { display: grid; gap: 12px; }
.kpi-grid.cols-4 { grid-template-columns: repeat(4, 1fr); }
.kpi-grid.cols-3 { grid-template-columns: repeat(3, 1fr); }
.kpi-card {
    background: #fff; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 16px 18px;
    transition: box-shadow .15s;
}
.kpi-card:hover { box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
.kpi-value { font-size: 1.4rem; font-weight: 700; color: #1a1a1a; line-height: 1.15; }
.kpi-label {
    font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: #6b7280; margin-bottom: 3px;
}
.kpi-accent  { border-left: 3px solid #6366f1; }
.kpi-success { border-left: 3px solid #22c55e; }
.kpi-warn    { border-left: 3px solid #f59e0b; }
.kpi-info    { border-left: 3px solid #3b82f6; }

/* ---- Config table ---- */
.cfg-table { width: 100%; border-collapse: separate; border-spacing: 0; }
.cfg-table th {
    text-align: left; font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.05em; color: #6b7280; padding: 10px 14px;
    border-bottom: 2px solid #e5e7eb;
}
.cfg-table td {
    padding: 10px 14px; font-size: 0.82rem; color: #1a1a1a;
    border-bottom: 1px solid #f3f4f6;
}
.cfg-table tr:hover td { background: #f9fafb; }
.cfg-badge-on {
    font-size: 0.65rem; padding: 2px 8px; border-radius: 999px;
    background: #dcfce7; color: #166534; font-weight: 600;
}
.cfg-badge-off {
    font-size: 0.65rem; padding: 2px 8px; border-radius: 999px;
    background: #fee2e2; color: #991b1b; font-weight: 600;
}
.cfg-mono {
    font-family: 'Menlo','Monaco','Consolas',monospace;
    font-size: 0.75rem; background: #f3f4f6; padding: 2px 6px;
    border-radius: 4px;
}

/* ---- Status dot ---- */
.status-dot {
    display: inline-block; width: 7px; height: 7px;
    border-radius: 50%; margin-right: 5px; vertical-align: middle;
}
.status-dot.online  { background: #22c55e; }
.status-dot.offline { background: #ef4444; }

/* ---- Page header ---- */
.page-header {
    background: #fafafa; border-bottom: 1px solid #e5e7eb;
    padding: 20px 28px; margin-bottom: 1.2rem;
}
.page-header h1 {
    margin: 0; font-size: 1.1rem; font-weight: 600; color: #1a1a1a;
}
.page-header .subtitle {
    margin: 3px 0 0; font-size: 0.75rem; color: #6b7280;
}

/* ---- Confirm dialog ---- */
.confirm-box {
    background: #1f1f1f; border: 1px solid #333; border-radius: 10px;
    padding: 16px; margin: 8px 0;
}
.confirm-box p {
    font-size: 0.78rem; margin: 0 0 10px; color: #d1d5db !important;
}

/* ---- Responsive ---- */
@media (max-width: 768px) {
    .kpi-grid.cols-4, .kpi-grid.cols-3 { grid-template-columns: 1fr 1fr; }
    .suggest-grid { grid-template-columns: 1fr; }
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ===================================================================
#  SESSION STATE
# ===================================================================
_DEFAULTS = {
    "conversations": {},
    "active_conv_id": None,
    "pipeline": None,
    "total_queries": 0,
    "response_times": [],
    "token_history": [],
    "page": "chat",
    "confirm_delete": None,
    "confirm_clear": False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ===================================================================
#  CONVERSATION MANAGEMENT
# ===================================================================
def _new_conversation() -> str:
    conv_id = uuid.uuid4().hex[:12]
    st.session_state.conversations[conv_id] = {
        "title": "New chat",
        "messages": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    st.session_state.active_conv_id = conv_id
    return conv_id


def _delete_conversation(conv_id: str):
    st.session_state.conversations.pop(conv_id, None)
    if st.session_state.active_conv_id == conv_id:
        remaining = list(st.session_state.conversations.keys())
        st.session_state.active_conv_id = remaining[0] if remaining else None
    st.session_state.confirm_delete = None


def _clear_all_conversations():
    st.session_state.conversations.clear()
    st.session_state.active_conv_id = None
    st.session_state.confirm_clear = False


def _active_messages() -> list[dict]:
    cid = st.session_state.active_conv_id
    if cid and cid in st.session_state.conversations:
        return st.session_state.conversations[cid]["messages"]
    return []


def _append_message(msg: dict):
    cid = st.session_state.active_conv_id
    if not cid or cid not in st.session_state.conversations:
        cid = _new_conversation()
    conv = st.session_state.conversations[cid]
    msg["timestamp"] = datetime.now().strftime("%I:%M %p")
    conv["messages"].append(msg)
    conv["updated_at"] = datetime.now().isoformat()
    if msg["role"] == "user" and conv["title"] == "New chat":
        conv["title"] = msg["content"][:45].strip()


def _group_conversations() -> dict[str, list[tuple[str, dict]]]:
    now = datetime.now()
    groups: dict[str, list] = {"Today": [], "Yesterday": [], "Previous": []}
    for cid, conv in sorted(
        st.session_state.conversations.items(),
        key=lambda x: x[1]["updated_at"],
        reverse=True,
    ):
        try:
            updated = datetime.fromisoformat(conv["updated_at"])
        except (ValueError, TypeError):
            updated = now
        delta = (now - updated).days
        if delta == 0:
            groups["Today"].append((cid, conv))
        elif delta == 1:
            groups["Yesterday"].append((cid, conv))
        else:
            groups["Previous"].append((cid, conv))
    return groups


def _conversation_count() -> int:
    return len(st.session_state.conversations)


# ===================================================================
#  PIPELINE
# ===================================================================
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


# ===================================================================
#  HELPERS
# ===================================================================
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


# ===================================================================
#  SIDEBAR
# ===================================================================
with st.sidebar:
    # -- New chat button --
    if st.button("+ New chat", use_container_width=True, key="new_chat_btn"):
        _new_conversation()
        st.session_state.page = "chat"
        st.rerun()

    st.divider()

    # -- Chat history (only on chat page) --
    if st.session_state.page == "chat" and st.session_state.conversations:
        grouped = _group_conversations()
        for group_name, convs in grouped.items():
            if not convs:
                continue
            st.markdown(
                f'<div class="chat-history-date">{group_name}</div>',
                unsafe_allow_html=True,
            )
            for cid, conv in convs:
                is_active = cid == st.session_state.active_conv_id
                label = conv["title"] or "New chat"
                msg_count = len(conv["messages"])
                btn_type = "primary" if is_active else "secondary"

                col_title, col_del = st.columns([5, 1], gap="small")
                with col_title:
                    if st.button(
                        label,
                        key=f"conv_{cid}",
                        use_container_width=True,
                        type=btn_type,
                        help=f"{msg_count} messages",
                    ):
                        if not is_active:
                            st.session_state.active_conv_id = cid
                            st.session_state.page = "chat"
                            st.session_state.confirm_delete = None
                            st.rerun()
                with col_del:
                    st.markdown('<div class="del-btn">', unsafe_allow_html=True)
                    if st.button("X", key=f"del_{cid}", use_container_width=True):
                        if st.session_state.confirm_delete == cid:
                            _delete_conversation(cid)
                            st.rerun()
                        else:
                            st.session_state.confirm_delete = cid
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

                # Confirm delete inline
                if st.session_state.confirm_delete == cid:
                    st.markdown(
                        '<div class="confirm-box">'
                        '<p>Delete this chat?</p>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    cf1, cf2 = st.columns(2)
                    with cf1:
                        if st.button("Yes, delete", key=f"confirm_del_{cid}", use_container_width=True):
                            _delete_conversation(cid)
                            st.rerun()
                    with cf2:
                        if st.button("Cancel", key=f"cancel_del_{cid}", use_container_width=True):
                            st.session_state.confirm_delete = None
                            st.rerun()

        # -- Clear all chats --
        st.divider()
        if _conversation_count() > 1:
            if st.session_state.confirm_clear:
                st.markdown(
                    '<div class="confirm-box">'
                    '<p>Clear all conversations?</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                cl1, cl2 = st.columns(2)
                with cl1:
                    if st.button("Yes, clear all", key="confirm_clear_all", use_container_width=True):
                        _clear_all_conversations()
                        st.rerun()
                with cl2:
                    if st.button("Cancel", key="cancel_clear_all", use_container_width=True):
                        st.session_state.confirm_clear = False
                        st.rerun()
            else:
                if st.button("Clear all chats", key="clear_all_btn", use_container_width=True):
                    st.session_state.confirm_clear = True
                    st.rerun()

    # -- Spacer --
    st.markdown('<div style="min-height:20px"></div>', unsafe_allow_html=True)
    st.divider()

    # -- Navigation (stacked vertical — no wrapping) --
    for nav_page, nav_label in [("chat", "Chat"), ("dashboard", "Dashboard"), ("settings", "Settings")]:
        btn_type = "primary" if st.session_state.page == nav_page else "secondary"
        if st.button(nav_label, key=f"nav_{nav_page}", use_container_width=True, type=btn_type):
            st.session_state.page = nav_page
            st.rerun()

    st.divider()

    # -- Connection status --
    pipe = _pipeline()
    provider, model_name = _provider_info()
    status_cls = "online" if pipe else "offline"
    status_txt = "Connected" if pipe else "Disconnected"
    st.markdown(
        f'<div style="font-size:0.75rem;padding:2px 4px">'
        f'<span class="status-dot {status_cls}"></span> {status_txt}'
        f'<br><span style="color:#6b7280;font-size:0.7rem">{provider} / {model_name}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ===================================================================
#  PAGE: CHAT
# ===================================================================
def _render_sources(sources: list[dict], time_ms: float = 0):
    label = f"{len(sources)} sources retrieved ({time_ms:.0f} ms)"
    with st.expander(label, expanded=False):
        for src in sources:
            score = src.get("score", 0)
            badge_bg = "#22c55e" if score >= 0.7 else ("#f59e0b" if score >= 0.4 else "#ef4444")
            st.markdown(
                f'<div class="src-card">'
                f'<div class="src-header">'
                f'<span class="src-title">{_safe(src.get("source", "unknown"))}</span>'
                f'<span class="src-badge" style="background:{badge_bg}">{score:.3f}</span>'
                f'</div>'
                f'<div class="src-preview">{_safe(src.get("content", ""))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_followups(followups: list[str]):
    if not followups:
        return
    pills = "".join(f'<span class="followup-pill">{_safe(fq)}</span>' for fq in followups)
    st.markdown(f'<div class="followup-row">{pills}</div>', unsafe_allow_html=True)


def render_chat_page():
    pipeline = _pipeline()
    messages = _active_messages()

    # -- Welcome screen (no active messages) --
    if not messages:
        st.markdown(
            '<div class="welcome-wrap">'
            '<div class="welcome-title">RAG Chatbot</div>'
            '<div class="welcome-sub">'
            '<span class="typewriter-line">Ask questions about Deepeval and Ragas....</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        suggestions = [
            ("Explore", "What topics are covered in the knowledge base?"),
            ("Explain", "How does the RAG pipeline retrieve context?"),
            ("Evaluate", "What evaluation metrics are used?"),
            ("Compare", "How do DeepEval and RAGAS differ?"),
        ]
        cols = st.columns(2)
        for i, (lbl, question) in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(question, key=f"suggest_{i}", use_container_width=True):
                    if not st.session_state.active_conv_id:
                        _new_conversation()
                    _append_message({"role": "user", "content": question})
                    st.rerun()

        if not pipeline:
            st.error("RAG Pipeline failed to initialize. Verify .env configuration and vector store.")
        return

    if not pipeline:
        st.error("RAG Pipeline failed to initialize. Verify .env configuration and vector store.")
        st.stop()

    # -- Render existing messages --
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            ts = msg.get("timestamp", "")
            if ts:
                st.markdown(f'<div class="msg-time">{ts}</div>', unsafe_allow_html=True)
            if msg["role"] == "assistant":
                _render_followups(msg.get("followups", []))
                sources = msg.get("sources", [])
                if sources:
                    _render_sources(sources, msg.get("time_ms", 0))

    # -- Chat input --
    if prompt := st.chat_input("Type your message..."):
        if not st.session_state.active_conv_id:
            _new_conversation()

        _append_message({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = pipeline.query(QueryRequest(query=prompt))
                    st.markdown(response.answer)

                    st.session_state.total_queries += 1
                    st.session_state.response_times.append(response.total_time_ms)
                    st.session_state.token_history.append(response.token_usage)

                    followups = response.followup_questions
                    _render_followups(followups)

                    sources_data = []
                    if response.sources:
                        sources_data = [
                            {
                                "source": s.source,
                                "score": s.score,
                                "content": s.content,
                                "chunk_id": s.chunk_id,
                            }
                            for s in response.sources
                        ]
                        _render_sources(sources_data, response.total_time_ms)

                    _append_message({
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
                    _append_message({
                        "role": "assistant",
                        "content": err,
                        "sources": [],
                        "followups": [],
                        "time_ms": 0,
                        "token_usage": {},
                    })

    # -- Export buttons --
    if messages:
        st.divider()
        c1, c2, c3, _ = st.columns([1, 1, 1, 3])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with c1:
            st.download_button(
                "Export JSON",
                data=json.dumps(messages, indent=2, default=str),
                file_name=f"chat_{ts}.json",
                mime="application/json",
                use_container_width=True,
            )
        with c2:
            md_lines = []
            for m in messages:
                role = "User" if m["role"] == "user" else "Assistant"
                md_lines.append(f"**{role}**: {m['content']}")
            md_text = "\n\n---\n\n".join(md_lines)
            st.download_button(
                "Export Markdown",
                data=md_text,
                file_name=f"chat_{ts}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with c3:
            csv_rows = "role,timestamp,content\n"
            for m in messages:
                safe_content = m["content"].replace('"', '""')
                csv_rows += f'{m["role"]},{m.get("timestamp","")},"{safe_content}"\n'
            st.download_button(
                "Export CSV",
                data=csv_rows,
                file_name=f"chat_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ===================================================================
#  PAGE: DASHBOARD
# ===================================================================
def render_dashboard_page():
    st.markdown(
        '<div class="page-header">'
        '<h1>Observability Dashboard</h1>'
        '<p class="subtitle">Token usage, latency, vector store, and cache metrics</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    pipeline = _pipeline()
    if not pipeline:
        st.info("Pipeline not initialized. Start a chat first.")
        return

    tok = pipeline.llm_service.token_usage.summary

    st.markdown(
        '<div class="kpi-grid cols-4">'
        + _render_kpi("Prompt Tokens", f"{tok['prompt_tokens']:,}", "accent")
        + _render_kpi("Completion Tokens", f"{tok['completion_tokens']:,}", "info")
        + _render_kpi("Total Tokens", f"{tok['total_tokens']:,}", "success")
        + _render_kpi("API Requests", tok["request_count"], "warn")
        + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("##### Tokens Per Query")
        if st.session_state.token_history:
            df = pd.DataFrame([
                {
                    "Query": i,
                    "Prompt": u.get("prompt_tokens", 0),
                    "Completion": u.get("completion_tokens", 0),
                }
                for i, u in enumerate(st.session_state.token_history, 1)
            ]).set_index("Query")
            st.bar_chart(df, color=["#6366f1", "#22c55e"])
        else:
            st.caption("No queries yet. Data appears after your first question.")

    with chart_col2:
        st.markdown("##### Response Latency (ms)")
        if st.session_state.response_times:
            df_rt = pd.DataFrame({"Latency": st.session_state.response_times})
            st.area_chart(df_rt, color="#6366f1")
        else:
            st.caption("No queries yet. Data appears after your first question.")

    st.divider()

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


# ===================================================================
#  PAGE: SETTINGS
# ===================================================================
def render_settings_page():
    st.markdown(
        '<div class="page-header">'
        '<h1>System Configuration</h1>'
        '<p class="subtitle">Read-only view of active environment settings</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    s = get_settings()
    provider, model_name = _provider_info()

    tab_llm, tab_retrieval, tab_features, tab_eval = st.tabs([
        "LLM", "Retrieval", "Features", "Evaluation",
    ])

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

    with tab_retrieval:
        rows = [
            ("Top-K", str(s.similarity_top_k)),
            ("Similarity Threshold", str(s.similarity_threshold)),
            ("Chunk Size", f"{s.chunk_size:,} chars"),
            ("Chunk Overlap", f"{s.chunk_overlap:,} chars"),
            ("Hybrid Search", _badge(s.hybrid_search_enabled)),
            ("Hybrid Alpha", f"{s.hybrid_search_alpha} <span style='color:#9ca3af;font-size:0.72rem'>(1.0 = dense, 0.0 = BM25)</span>"),
            ("Reranking", _badge(s.reranking_enabled)),
            ("Reranker Top-N", str(s.reranker_top_n)),
            ("Reranker Threshold", str(s.reranker_relevance_threshold)),
        ]
        _config_table(rows)

    with tab_features:
        rows = [
            ("Response Cache", f'{_badge(s.response_cache_enabled)} <span style="color:#9ca3af;font-size:0.72rem">max={s.response_cache_max_size}, ttl={s.response_cache_ttl_seconds}s</span>'),
            ("Embedding Cache", f'{_badge(s.embedding_cache_enabled)} <span style="color:#9ca3af;font-size:0.72rem">max={s.embedding_cache_max_size}</span>'),
            ("Follow-up Questions", f'{_badge(s.followup_enabled)} <span style="color:#9ca3af;font-size:0.72rem">count={s.followup_count}</span>'),
            ("Metadata Enrichment", _badge(s.metadata_enrichment_enabled)),
        ]
        _config_table(rows)

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
    body = "".join(
        f'<tr><td style="font-weight:500">{label}</td><td>{value}</td></tr>'
        for label, value in rows
    )
    st.markdown(
        f'<table class="cfg-table"><thead><tr><th>Parameter</th><th>Value</th></tr></thead>'
        f'<tbody>{body}</tbody></table>',
        unsafe_allow_html=True,
    )


# ===================================================================
#  PAGE ROUTER
# ===================================================================
_PAGES = {
    "chat": render_chat_page,
    "dashboard": render_dashboard_page,
    "settings": render_settings_page,
}
_PAGES[st.session_state.page]()
