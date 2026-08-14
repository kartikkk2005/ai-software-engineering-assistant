import os
import streamlit as st
from pathlib import Path

from config import validate_config, GEMINI_API_KEY, OPENAI_API_KEY, EMBEDDING_PROVIDER, LLM_PROVIDER
from github_loader import clone_or_download_repo, get_codebase_files
from file_loader import load_codebase_documents
from chunker import split_documents
from embeddings import index_documents, get_vector_store
from rag import query_codebase, explain_codebase_architecture, analyze_single_file

# Streamlit Page Config
st.set_page_config(
    page_title="AI Software Engineering Assistant (MCP + RAG)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism Dark CSS Aesthetics
st.markdown("""
<style>
    /* Dark Theme Custom Colors */
    :root {
        --bg-color: #0d1117;
        --card-bg: rgba(22, 27, 34, 0.75);
        --accent-purple: #8a2be2;
        --accent-cyan: #00f2fe;
        --accent-gradient: linear-gradient(135deg, #7928CA 0%, #FF0080 100%);
        --text-primary: #f0f6fc;
        --border-color: rgba(255, 255, 255, 0.1);
    }
    
    .stApp {
        background-color: #0d1117;
        color: #f0f6fc;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, rgba(121, 40, 202, 0.2) 0%, rgba(0, 242, 254, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 50%, #00c6ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .header-desc {
        color: #8b949e;
        font-size: 1.05rem;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00f2fe;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #8b949e;
    }
    
    /* Source citation tag */
    .citation-badge {
        display: inline-block;
        background: rgba(0, 242, 254, 0.15);
        color: #00f2fe;
        border: 1px solid rgba(0, 242, 254, 0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.8rem;
        margin: 2px 4px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Render Header
st.markdown("""
<div class="header-box">
    <div class="header-title">⚡ AI Software Engineering Assistant</div>
    <div class="header-desc">
        Context-aware GitHub codebase comprehension powered by <b>RAG (ChromaDB + LangChain)</b> & <b>Model Context Protocol (FastMCP)</b>.
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.shields.io/badge/MCP-FastMCP-purple?style=for-the-badge", use_container_width=True)
    st.markdown("### ⚙️ System Status")
    
    config_status = validate_config()
    if config_status["gemini"]:
        st.success("✅ Gemini API Key Configured")
    elif config_status["openai"]:
        st.success("✅ OpenAI API Key Configured")
    else:
        st.warning("⚠️ No API Key set. Please add GEMINI_API_KEY or OPENAI_API_KEY to `.env`.")

    st.markdown("---")
    st.markdown("### 🛠️ Configuration")
    st.info(f"**Embedding Provider:** `{EMBEDDING_PROVIDER.upper()}`")
    st.info(f"**LLM Provider:** `{LLM_PROVIDER.upper()}`")
    
    st.markdown("---")
    st.markdown("**Technologies Used:**")
    st.caption("• FastMCP (Model Context Protocol)")
    st.caption("• LangChain & ChromaDB")
    st.caption("• Gemini API & OpenAI API")
    st.caption("• Streamlit Interactive UI")

# Initialize Chat History State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_repo" not in st.session_state:
    st.session_state.indexed_repo = None
if "stats" not in st.session_state:
    st.session_state.stats = {"docs": 0, "chunks": 0}

# Main Application Tabs
tab_ingest, tab_chat, tab_search, tab_arch, tab_mcp = st.tabs([
    "📥 Ingest Repository",
    "💬 RAG Code QA",
    "🔍 Vector Search",
    "🏗️ Architecture",
    "⚡ MCP Config"
])

# ---------------------------------------------------------
# TAB 1: INGEST REPOSITORY
# ---------------------------------------------------------
with tab_ingest:
    st.subheader("📥 Index GitHub Repository or Local Codebase")
    st.markdown("Enter a public GitHub URL (e.g. `https://github.com/fastapi/fastapi`) or local directory path to chunk & index into ChromaDB.")
    
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        repo_input = st.text_input(
            "GitHub Repository URL or Path", 
            placeholder="https://github.com/owner/repository",
            key="repo_input_field"
        )
    with col_btn:
        st.write("") # spacing
        st.write("")
        start_ingest = st.button("🚀 Start Indexing", use_container_width=True, type="primary")

    if start_ingest and repo_input:
        with st.status("Processing Repository...", expanded=True) as status:
            try:
                st.write("📁 Step 1: Accessing repository files...")
                repo_path = Path(repo_input) if Path(repo_input).exists() else clone_or_download_repo(repo_input)
                
                st.write("🔍 Step 2: Scanning code files...")
                code_files = get_codebase_files(repo_path)
                st.write(f"Found {len(code_files)} source files.")
                
                st.write("📄 Step 3: Extracting document content & metadata...")
                docs = load_codebase_documents(code_files, repo_path)
                
                st.write("✂️ Step 4: Chunking code with language-aware splitting...")
                chunks = split_documents(docs)
                
                st.write("⚡ Step 5: Generating embeddings and indexing into ChromaDB...")
                index_documents(chunks, reset_db=True)
                
                st.session_state.indexed_repo = repo_path.name
                st.session_state.stats = {"docs": len(docs), "chunks": len(chunks)}
                
                status.update(label=f"✅ Successfully indexed '{repo_path.name}'!", state="complete")
                st.balloons()
            except Exception as e:
                status.update(label=f"❌ Ingestion Failed: {e}", state="error")

    if st.session_state.indexed_repo:
        st.markdown("---")
        st.markdown("#### 📊 Current Indexed Repository Stats")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{st.session_state.indexed_repo}</div><div class="metric-lbl">Active Repository</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{st.session_state.stats["docs"]}</div><div class="metric-lbl">Source Files</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{st.session_state.stats["chunks"]}</div><div class="metric-lbl">Vector Chunks</div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: RAG CODE QA
# ---------------------------------------------------------
with tab_chat:
    st.subheader("💬 Context-Aware Code Q&A Assistant")
    st.caption("Ask questions about code functionality, bug diagnosis, API endpoints, or implementation logic.")

    # Display past chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg and msg["sources"]:
                st.markdown("**Referenced Sources:**")
                for s in msg["sources"]:
                    st.markdown(f'<span class="citation-badge">📄 {s["source"]} (L{s["start_line"]}-L{s["end_line"]})</span>', unsafe_allow_html=True)

    user_query = st.chat_input("Ask a question about the indexed codebase...")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving code context and generating answer..."):
                res = query_codebase(user_query)
                answer = res.get("answer", "No answer available.")
                sources = res.get("sources", [])

                st.write(answer)
                if sources:
                    st.markdown("**Referenced Sources:**")
                    for s in sources:
                        st.markdown(f'<span class="citation-badge">📄 {s["source"]} (L{s["start_line"]}-L{s["end_line"]})</span>', unsafe_allow_html=True)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

# ---------------------------------------------------------
# TAB 3: VECTOR SEARCH
# ---------------------------------------------------------
with tab_search:
    st.subheader("🔍 Semantic Vector Search Explorer")
    st.markdown("Perform raw similarity search across code chunks in ChromaDB.")

    s_col1, s_col2 = st.columns([4, 1])
    with s_col1:
        search_query = st.text_input("Semantic Search Query", placeholder="e.g. rate limit error handling or database connection pool")
    with s_col2:
        top_k_val = st.number_input("Top K Snippets", min_value=1, max_value=20, value=5)

    if st.button("🔎 Run Vector Search", type="primary"):
        if search_query:
            try:
                v_store = get_vector_store()
                results = v_store.similarity_search(search_query, k=top_k_val)
                
                if not results:
                    st.warning("No matching code chunks found.")
                else:
                    st.success(f"Found {len(results)} matching code snippets:")
                    for idx, doc in enumerate(results, 1):
                        src = doc.metadata.get("source", "unknown")
                        s_line = doc.metadata.get("start_line", 1)
                        e_line = doc.metadata.get("end_line", 1)
                        lang = doc.metadata.get("language", "python")
                        
                        with st.expander(f"Result {idx}: {src} (Lines {s_line}-{e_line})"):
                            st.code(doc.page_content, language=lang)
            except Exception as e:
                st.error(f"Search failed: {e}")

# ---------------------------------------------------------
# TAB 4: ARCHITECTURE & CODE EXPLORER
# ---------------------------------------------------------
with tab_arch:
    st.subheader("🏗️ Architectural Breakdown & File Inspector")
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        if st.button("✨ Generate Full Architecture Overview", type="primary", use_container_width=True):
            with st.spinner("Analyzing high-level code structure and dependencies..."):
                arch_res = explain_codebase_architecture()
                st.markdown(arch_res.get("answer", "Analysis failed."))

    with col_a2:
        file_to_analyze = st.text_input("Analyze Specific File", placeholder="e.g. rag.py or embeddings.py")
        if st.button("🔬 Analyze File Details", use_container_width=True):
            if file_to_analyze:
                with st.spinner(f"Analyzing '{file_to_analyze}'..."):
                    file_res = analyze_single_file(file_to_analyze)
                    st.markdown(file_res.get("answer", "Analysis failed."))

# ---------------------------------------------------------
# TAB 5: MCP SERVER CONFIG
# ---------------------------------------------------------
with tab_mcp:
    st.subheader("⚡ Model Context Protocol (MCP) Server Integration")
    st.markdown("""
    This project includes a **FastMCP** server (`mcp_server.py`) that exposes automated codebase retrieval and analysis tools to external AI environments like **Claude Desktop** and **Cursor**.
    """)
    
    st.markdown("#### 🛠️ Available FastMCP Tools:")
    st.markdown("""
    - `index_repository(repo_url)`: Clones & indexes codebase into ChromaDB.
    - `search_codebase(query, top_k)`: Performs semantic similarity search.
    - `ask_codebase_question(question)`: Generates context-grounded RAG answers.
    - `analyze_file(file_path)`: Analyzes a specific file.
    - `explain_architecture()`: Generates system architecture overview.
    """)

    st.markdown("#### 📋 Claude Desktop Configuration (`claude_desktop_config.json`):")
    
    mcp_script_path = str((Path(__file__).parent / "mcp_server.py").resolve()).replace("\\", "\\\\")
    
    config_json = f"""{{
  "mcpServers": {{
    "ai-codebase-assistant": {{
      "command": "python",
      "args": [
        "{mcp_script_path}"
      ]
    }}
  }}
}}"""
    st.code(config_json, language="json")
