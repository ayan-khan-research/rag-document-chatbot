import streamlit as st
import os
import time
from datetime import datetime
from src.document_processor import DocumentProcessor
from src.rag_engine import RAGEngine
from src.analytics import AnalyticsTracker

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat — RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .chat-message-user {
        background: #e3f2fd;
        color: #000000;   /* add this */
        border-radius: 12px 12px 2px 12px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1976D2;
    }
    .chat-message-bot {
    background: #ffffff;   /* white background */
    color: #000000;        /* black text */
    border-radius: 12px 12px 12px 2px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #7B1FA2;
}
    .source-badge {
        background: #e8f5e9;
        border: 1px solid #4CAF50;
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.75rem;
        color: #2E7D32;
        margin-right: 0.3rem;
        display: inline-block;
    }
    .stAlert { border-radius: 8px; }
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─── Initialize Session State ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "doc_processor" not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor()
if "analytics" not in st.session_state:
    st.session_state.analytics = AnalyticsTracker()
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:1.8rem;">📚 DocChat — AI Document Assistant</h1>
    <p style="margin:0.3rem 0 0; opacity:0.85;">Upload any document and get accurate, cited answers using RAG</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Model Selection
    st.markdown("### 🤖 AI Model")
    model_choice = st.selectbox(
        "Select LLM",
        ["OpenAI GPT-4o", "OpenAI GPT-3.5 Turbo", "Ollama (Local) — Llama3", "Ollama (Local) — Mistral"],
        help="GPT-4o gives best accuracy. Ollama is free but needs local install."
    )

    # API Key input (only for OpenAI)
    if "OpenAI" in model_choice:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Get your key from platform.openai.com"
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    # Embedding Model
    st.markdown("### 🔢 Embedding Model")
    embed_choice = st.selectbox(
        "Select Embeddings",
        ["OpenAI text-embedding-3-small", "HuggingFace all-MiniLM-L6-v2", "HuggingFace BAAI/bge-small-en"],
        help="OpenAI embeddings need API key. HuggingFace is free."
    )

    # Vector Store
    st.markdown("### 🗄️ Vector Store")
    vector_store = st.selectbox("Select Vector DB", ["FAISS (Local)", "ChromaDB (Local)"])

    # RAG Settings
    st.markdown("### 🎯 RAG Settings")
    chunk_size = st.slider("Chunk Size (tokens)", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10)
    top_k = st.slider("Top-K Chunks to Retrieve", 1, 10, 5)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.1, 0.05,
                            help="Lower = more accurate and deterministic")

    st.divider()

    # Document Upload
    st.markdown("## 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drag & drop files here",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "csv", "xlsx"],
        help="Supports PDF, Word, TXT, CSV, Excel"
    )

    if uploaded_files:
        if st.button("🔄 Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing documents..."):
                progress = st.progress(0)
                all_texts = []
                doc_names = []

                for i, file in enumerate(uploaded_files):
                    progress.progress((i + 1) / len(uploaded_files))
                    texts, name = st.session_state.doc_processor.process(file)
                    all_texts.extend(texts)
                    doc_names.append(name)
                    st.session_state.analytics.add_document(name, len(texts))

                # Build RAG Engine
                st.session_state.rag_engine = RAGEngine(
                    texts=all_texts,
                    model_choice=model_choice,
                    embed_choice=embed_choice,
                    vector_store=vector_store,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                    temperature=temperature
                )
                st.session_state.documents_loaded = doc_names
                progress.progress(1.0)
                st.success(f"✅ {len(uploaded_files)} document(s) processed! {len(all_texts)} chunks indexed.")

    # Loaded documents
    if st.session_state.documents_loaded:
        st.markdown("### 📄 Loaded Documents")
        for doc in st.session_state.documents_loaded:
            st.markdown(f"✅ `{doc}`")

        if st.button("🗑️ Clear All Documents", use_container_width=True):
            st.session_state.rag_engine = None
            st.session_state.documents_loaded = []
            st.session_state.messages = []
            st.session_state.analytics.reset()
            st.rerun()

# ─── Main Layout: Chat + Analytics ─────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Chat", "📊 Analytics Dashboard"])

# ════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ════════════════════════════════════════════════════════════
with tab1:
    if not st.session_state.rag_engine:
        st.info("👆 Upload documents from the sidebar and click **Process Documents** to start chatting.")

        # Example queries
        st.markdown("### 💡 What you can do:")
        cols = st.columns(3)
        examples = [
            ("📋 Syllabus", "What topics are covered in Unit 3?"),
            ("📜 Regulations", "What is the attendance policy?"),
            ("🔬 Research", "Summarize the methodology section"),
        ]
        for col, (icon_title, example) in zip(cols, examples):
            with col:
                st.markdown(f"**{icon_title}**")
                st.code(example)
    else:
        # Chat history display
        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                <div style="text-align:center; padding:2rem; color:#666;">
                    <h3>👋 Ready to answer questions!</h3>
                    <p>Ask anything about your uploaded documents.</p>
                </div>
                """, unsafe_allow_html=True)

            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message-user">
                        <strong>🧑 You:</strong><br>{msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    sources_html = "".join([
                        f'<span class="source-badge">📄 {s}</span>'
                        for s in msg.get("sources", [])
                    ])
                    st.markdown(f"""
                    <div class="chat-message-bot">
                        <strong>🤖 DocChat:</strong><br>{msg["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.divider()

        # Query input
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            query = st.text_input(
                "Ask a question",
                placeholder="e.g. What is the grading policy? / Explain the algorithm in chapter 3",
                label_visibility="collapsed"
            )
        with col_btn:
            ask_btn = st.button("Ask ➤", type="primary", use_container_width=True)

        # Suggested questions
        st.markdown("**Suggested:**")
        sugg_cols = st.columns(4)
        suggestions = [
            "Summarize the document",
            "What are the key topics?",
            "List all important dates",
            "What are the main conclusions?"
        ]
        for i, (col, suggestion) in enumerate(zip(sugg_cols, suggestions)):
            with col:
                if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                    query = suggestion
                    ask_btn = True

        # Process Query
        if ask_btn and query:
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.total_queries += 1

            with st.spinner("🔍 Searching documents and generating answer..."):
                start_time = time.time()
                answer, sources, confidence = st.session_state.rag_engine.query(query)
                elapsed = round(time.time() - start_time, 2)

                st.session_state.analytics.add_query(query, elapsed, confidence)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "confidence": confidence,
                    "time": elapsed
                })

            st.success(f"✅ Answered in {elapsed}s | Confidence: {confidence:.0%}")
            st.rerun()

        # Clear chat
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()


# ════════════════════════════════════════════════════════════
# TAB 2: ANALYTICS DASHBOARD
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 Analytics Dashboard")

    analytics = st.session_state.analytics

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📁 Documents Loaded", len(st.session_state.documents_loaded))
    with col2:
        st.metric("💬 Total Queries", analytics.total_queries)
    with col3:
        avg_time = analytics.avg_response_time()
        st.metric("⚡ Avg Response Time", f"{avg_time:.2f}s")
    with col4:
        avg_conf = analytics.avg_confidence()
        st.metric("🎯 Avg Confidence", f"{avg_conf:.0%}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📄 Documents & Chunks")
        if analytics.documents:
            import pandas as pd
            df_docs = pd.DataFrame(analytics.documents, columns=["Document", "Chunks"])
            st.dataframe(df_docs, use_container_width=True, hide_index=True)
        else:
            st.info("No documents loaded yet.")

        st.markdown("### 🕐 Recent Queries")
        if analytics.query_log:
            df_q = pd.DataFrame(analytics.query_log[-10:][::-1],
                                columns=["Query", "Time (s)", "Confidence"])
            df_q["Confidence"] = df_q["Confidence"].apply(lambda x: f"{x:.0%}")
            st.dataframe(df_q, use_container_width=True, hide_index=True)
        else:
            st.info("No queries yet.")

    with col_right:
        st.markdown("### 📈 Response Time Trend")
        if analytics.response_times:
            import pandas as pd
            df_times = pd.DataFrame({
                "Query #": list(range(1, len(analytics.response_times) + 1)),
                "Response Time (s)": analytics.response_times
            })
            st.line_chart(df_times.set_index("Query #"))
        else:
            st.info("Run some queries to see the trend.")

        st.markdown("### 🎯 Confidence Distribution")
        if analytics.confidences:
            import pandas as pd
            df_conf = pd.DataFrame({
                "Query #": list(range(1, len(analytics.confidences) + 1)),
                "Confidence": analytics.confidences
            })
            st.bar_chart(df_conf.set_index("Query #"))
        else:
            st.info("Run some queries to see confidence scores.")
