"""
app.py — KubeBot: Kubernetes RAG Chatbot

Streamlit UI wrapping the LlamaIndex + ChromaDB + Claude RAG pipeline.
"""

import sys
import os
from pathlib import Path

# Ensure the project root is on sys.path so local modules are always found,
# regardless of where Streamlit is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="KubeBot — Kubernetes RAG Assistant",
    page_icon="⎈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* Global */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* Dark sidebar */
  section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
  }
  section[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
  }

  /* Main background */
  .stApp {
    background: #0d1117;
    color: #e6edf3;
  }

  /* Header strip */
  .kubebot-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 18px 0 10px;
    border-bottom: 1px solid #21262d;
    margin-bottom: 24px;
  }
  .kubebot-header .logo {
    font-size: 2rem;
  }
  .kubebot-header h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #58a6ff;
    margin: 0;
  }
  .kubebot-header .subtitle {
    font-size: 0.8rem;
    color: #7d8590;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2px;
  }
  .k8s-badge {
    background: #1f6feb22;
    border: 1px solid #1f6feb;
    color: #58a6ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: auto;
  }

  /* Chat messages */
  .chat-bubble {
    padding: 14px 18px;
    border-radius: 10px;
    margin-bottom: 12px;
    line-height: 1.7;
    font-size: 0.95rem;
  }
  .chat-bubble.user {
    background: #1c2128;
    border: 1px solid #30363d;
    border-left: 3px solid #58a6ff;
    margin-left: 40px;
  }
  .chat-bubble.assistant {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #3fb950;
    margin-right: 40px;
  }
  .chat-bubble .role-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
    opacity: 0.7;
  }
  .chat-bubble.user .role-label { color: #58a6ff; }
  .chat-bubble.assistant .role-label { color: #3fb950; }

  /* Sources panel */
  .sources-container {
    margin-top: 12px;
    padding: 12px 16px;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
  }
  .sources-container .sources-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #7d8590;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
  }
  .source-item {
    font-size: 0.8rem;
    padding: 6px 0;
    border-bottom: 1px solid #21262d;
    color: #8b949e;
  }
  .source-item:last-child { border-bottom: none; }
  .source-item .topic { color: #58a6ff; font-weight: 500; }
  .source-score {
    float: right;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #3fb950;
    background: #3fb95011;
    padding: 1px 6px;
    border-radius: 3px;
  }

  /* Sidebar stats */
  .stat-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
  }
  .stat-box .stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #7d8590;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .stat-box .stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #58a6ff;
    margin-top: 4px;
  }

  /* Chat input — dark background on every wrapper layer */
  .stChatInput,
  .stChatInput > div,
  .stChatInput > div > div,
  .stChatInput > div > div > div,
  [data-testid="stChatInput"],
  [data-testid="stChatInput"] > div,
  [data-testid="stChatInputContainer"],
  [data-testid="stChatInputContainer"] > div,
  .stBottomBlockContainer,
  [data-testid="stBottomBlockContainer"],
  .stBottom,
  [data-testid="stBottom"],
  .stBottom > div,
  [data-testid="stBottom"] > div {
    background: #0d1117 !important;
    box-shadow: none !important;
    border-color: transparent !important;
  }

  /* The textarea itself */
  .stChatInput textarea,
  [data-testid="stChatInputContainer"] textarea {
    background: #161b22 !important;
    color: #e6edf3 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    caret-color: #58a6ff !important;
    border: 1px solid #30363d !important;
  }

  /* Placeholder text */
  .stChatInput textarea::placeholder,
  [data-testid="stChatInputContainer"] textarea::placeholder {
    color: #8b949e !important;
    opacity: 1 !important;
  }

  /* Focus state */
  .stChatInput textarea:focus,
  [data-testid="stChatInputContainer"] textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px #58a6ff22 !important;
    outline: none !important;
  }

  /* Send button inside chat input */
  [data-testid="stChatInputContainer"] button {
    background: transparent !important;
    color: #58a6ff !important;
  }

  /* Text input (sidebar API key field) */
  .stTextInput input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 8px !important;
  }

  /* Buttons */
  .stButton > button {
    background: #1f6feb !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
  }
  .stButton > button:hover {
    background: #388bfd !important;
  }

  /* Progress bar */
  .stProgress > div > div {
    background: #1f6feb !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #7d8590 !important;
  }

  /* Code blocks */
  code {
    background: #1c2128 !important;
    color: #79c0ff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85em;
    padding: 1px 5px;
    border-radius: 3px;
  }
  pre code {
    color: #e6edf3 !important;
    padding: 0;
  }

  /* Hide Streamlit branding — removing 'header' keeps the sidebar toggle visible */
  #MainMenu, footer { visibility: hidden; }

  /* Style the top header bar to match the dark theme */
  header[data-testid="stHeader"] {
    background: #0d1117 !important;
    border-bottom: 1px solid #21262d !important;
    box-shadow: none !important;
  }

  /* Sidebar open AND close button — always white, same rules for both */
  button[data-testid="baseButton-headerNoPadding"],
  [data-testid="stSidebarCollapseButton"] button {
    color: #ffffff !important;
    opacity: 1 !important;
    visibility: visible !important;
  }
  button[data-testid="baseButton-headerNoPadding"] svg path,
  [data-testid="stSidebarCollapseButton"] button svg path {
    fill: #ffffff !important;
    stroke: #ffffff !important;
  }

  /* Deploy button */
  header [data-testid="stToolbar"],
  header [data-testid="stDecoration"],
  .stDeployButton,
  header button[kind="header"] {
    background: transparent !important;
    color: #8b949e !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
  }
  .stDeployButton:hover,
  header button[kind="header"]:hover {
    background: #21262d !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "top_k" not in st.session_state:
    st.session_state.top_k = 5


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="kubebot-header">
  <div class="logo">⎈</div>
  <div>
    <h1>KubeBot</h1>
    <div class="subtitle">Kubernetes Internals · RAG Assistant</div>
  </div>
  <div class="k8s-badge">Claude + ChromaDB + LlamaIndex</div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⎈ KubeBot Settings")
    st.markdown("---")

    # API key input
    api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        placeholder="sk-ant-...",
        help="Get your key at console.anthropic.com",
    )
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input

    st.markdown("---")

    # Retrieval settings
    st.markdown("**Retrieval Settings**")
    top_k = st.slider(
        "Chunks to retrieve (top-k)",
        min_value=2,
        max_value=10,
        value=st.session_state.top_k,
        help="More chunks = more context but slower responses",
    )
    if top_k != st.session_state.top_k:
        st.session_state.top_k = top_k
        # Rebuild query engine with new top_k if index is ready
        if st.session_state.index:
            from rag import build_query_engine
            st.session_state.query_engine = build_query_engine(
                st.session_state.index, similarity_top_k=top_k
            )

    st.markdown("---")

    # Index controls
    st.markdown("**Vector Index**")

    col1, col2 = st.columns(2)
    with col1:
        build_btn = st.button("🔨 Build Index", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear Chat", use_container_width=True)

    if clear_btn:
        st.session_state.messages = []
        st.rerun()

    # Stats
    if st.session_state.index_ready:
        from rag import get_collection_stats
        stats = get_collection_stats()
        st.markdown(f"""
        <div class="stat-box">
          <div class="stat-label">Chunks Indexed</div>
          <div class="stat-value">{stats['chunk_count']:,}</div>
        </div>
        """, unsafe_allow_html=True)
        st.success("✓ Index ready")
    else:
        st.info("Build the index to start chatting")

    st.markdown("---")

    # Suggested questions
    st.markdown("**💡 Try asking:**")
    example_questions = [
        "How does the kube-scheduler work?",
        "What is the role of etcd in Kubernetes?",
        "Explain the difference between a Deployment and StatefulSet",
        "How do Services route traffic to Pods?",
        "What happens during Pod scheduling?",
        "How does RBAC work in Kubernetes?",
        "Explain PersistentVolumes and PersistentVolumeClaims",
        "What are Init Containers used for?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True, key=f"eq_{q[:20]}"):
            st.session_state._prefill_question = q
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem; color:#7d8590; font-family:JetBrains Mono'>"
        "KubeBot · Built with LlamaIndex,<br>ChromaDB & Claude</div>",
        unsafe_allow_html=True,
    )


# ── Index building ────────────────────────────────────────────────────────────
if build_btn:
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("⚠ Please enter your Anthropic API key in the sidebar first.")
    else:
        with st.spinner(""):
            progress_bar = st.progress(0, text="Initializing...")
            status_text = st.empty()

            def update_progress(fraction: float, message: str):
                progress_bar.progress(fraction, text=message)
                status_text.markdown(
                    f"<div style='font-family:JetBrains Mono;font-size:0.8rem;"
                    f"color:#7d8590'>{message}</div>",
                    unsafe_allow_html=True,
                )

            try:
                from rag import build_index, build_query_engine
                index = build_index(progress_callback=update_progress)
                query_engine = build_query_engine(index, similarity_top_k=st.session_state.top_k)

                st.session_state.index = index
                st.session_state.query_engine = query_engine
                st.session_state.index_ready = True

                progress_bar.empty()
                status_text.empty()
                st.success("✅ Index built and ready!")
                time.sleep(1)
                st.rerun()

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Failed to build index: {e}")


# ── Auto-load index if it already exists ─────────────────────────────────────
if not st.session_state.index_ready and os.getenv("ANTHROPIC_API_KEY"):
    try:
        from rag import get_collection_stats
        stats = get_collection_stats()
        if stats["chunk_count"] > 0:
            from rag import build_index, build_query_engine
            with st.spinner("Loading existing index..."):
                index = build_index()
                query_engine = build_query_engine(index, similarity_top_k=st.session_state.top_k)
                st.session_state.index = index
                st.session_state.query_engine = query_engine
                st.session_state.index_ready = True
            st.rerun()
    except Exception:
        pass


# ── Chat area ─────────────────────────────────────────────────────────────────
chat_col, _ = st.columns([1, 0.001])
with chat_col:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color: #7d8590;">
          <div style="font-size: 3rem; margin-bottom: 16px;">⎈</div>
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: #58a6ff; margin-bottom: 8px;">
            Ready to explore Kubernetes internals
          </div>
          <div style="font-size: 0.85rem; max-width: 480px; margin: 0 auto; line-height: 1.6;">
            Build the index, then ask questions about Pods, Controllers,
            the Control Plane, Networking, Storage, Scheduling, and more.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Render conversation history
    for msg in st.session_state.messages:
        role = msg["role"]
        bubble_class = "user" if role == "user" else "assistant"
        label = "You" if role == "user" else "KubeBot ⎈"

        st.markdown(f"""
        <div class="chat-bubble {bubble_class}">
          <div class="role-label">{label}</div>
          {msg["content"]}
        </div>
        """, unsafe_allow_html=True)

        if role == "assistant" and msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s) retrieved", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-item">
                      <span class="source-score">{src['score']:.3f}</span>
                      <span class="topic">{src['topic'].replace('_', ' ').title()}</span><br>
                      <span style="font-size:0.75rem; color:#7d8590;">{src['snippet']}</span>
                    </div>
                    """, unsafe_allow_html=True)


# ── Chat input ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("_prefill_question", None)

user_input = st.chat_input(
    "Ask anything about Kubernetes internals...",
    disabled=not st.session_state.index_ready,
)

# Use prefilled question if set (from sidebar buttons)
question = prefill or user_input

if question:
    if not st.session_state.index_ready:
        st.warning("⚠ Please build the index first using the sidebar button.")
    elif not os.getenv("ANTHROPIC_API_KEY"):
        st.error("⚠ Please enter your Anthropic API key.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("⎈ Retrieving and reasoning..."):
            try:
                from rag import query_with_sources
                result = query_with_sources(st.session_state.query_engine, question)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠ Error: {e}",
                    "sources": [],
                })

        st.rerun()
