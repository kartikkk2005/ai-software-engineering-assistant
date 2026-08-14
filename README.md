# ⚡ AI Software Engineering Assistant (MCP + RAG)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastMCP](https://img.shields.io/badge/MCP-FastMCP-purple?style=for-the-badge)](https://github.com/jlowin/fastmcp)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-green?style=for-the-badge)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-red?style=for-the-badge)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)

An enterprise-grade **AI Software Engineering Assistant** that ingests public or private GitHub repositories, indexes code chunks with language-aware splitting into **ChromaDB**, and provides context-grounded codebase understanding using **Retrieval-Augmented Generation (RAG)** via **Gemini API** / **OpenAI API**.

The system seamlessly integrates **Model Context Protocol (MCP)** via **FastMCP**, enabling external AI clients (e.g. Claude Desktop, Cursor) to directly invoke codebase analysis tools.

---

## 🌟 Key Features

- **🌐 GitHub & Local Ingestion**: Clones GitHub repositories or loads local codebases while automatically filtering out binaries, `.git`, `node_modules`, lockfiles, and vendor directories.
- **✂️ Language-Aware Code Chunking**: Uses language-specific splitters (Python, JavaScript/TypeScript, Go, Java, Markdown) while tracking line numbers (`start_line`, `end_line`) and file metadata.
- **⚡ Vector Search & ChromaDB**: Fast semantic similarity search backed by ChromaDB, supporting Google Generative AI embeddings (`text-embedding-004`), OpenAI embeddings (`text-embedding-3-small`), or local HuggingFace embeddings (`all-MiniLM-L6-v2`).
- **💬 Context-Grounded RAG Engine**: Answers code comprehension, bug diagnosis, and architecture questions with exact file path and line number citations.
- **🔌 Model Context Protocol (FastMCP) Tools**: Exposes standard MCP tools (`index_repository`, `search_codebase`, `ask_codebase_question`, `analyze_file`, `explain_architecture`).
- **🎨 Glassmorphism Interactive Dashboard**: Modern dark-mode Streamlit app for real-time repository ingestion, Q&A chat, vector exploration, and architecture breakdown.

---

## 📁 Repository Structure

```
.
├── app.py              # Streamlit Web UI Application
├── mcp_server.py       # FastMCP Server exposing MCP codebase tools
├── rag.py              # RAG Engine & context-grounded LLM chain
├── embeddings.py       # ChromaDB vector store manager & embedding providers
├── chunker.py          # Language-aware code splitter with line range tracking
├── file_loader.py      # Source code file reader with encoding fallback & metadata
├── github_loader.py    # GitHub repository cloner & tree filter
├── config.py           # Environment variables & configuration loader
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable configuration template
└── .gitignore          # Version control ignore rules
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Installation

Ensure Python 3.10+ is installed. Clone this repository and install dependencies:

```bash
git clone https://github.com/your-username/ai-software-engineering-assistant.git
cd ai-software-engineering-assistant

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` and provide your API keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

EMBEDDING_PROVIDER=gemini
LLM_PROVIDER=gemini
```

---

## 💻 Usage Options

### Option A: Launch Streamlit Web UI

Run the interactive dashboard:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to:
1. Ingest a GitHub URL (e.g., `https://github.com/fastapi/fastapi`).
2. Ask questions about the codebase with source file references.
3. Perform semantic vector search.
4. Generate high-level system architecture breakdowns.

---

### Option B: Run FastMCP Server

Start the FastMCP server:

```bash
python mcp_server.py
```

#### Connect with Claude Desktop

Add the server to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ai-codebase-assistant": {
      "command": "python",
      "args": [
        "C:\\path\\to\\mcp_server.py"
      ]
    }
  }
}
```

---

## 🛠️ FastMCP Tool API Reference

| Tool Name | Description | Parameters |
| :--- | :--- | :--- |
| `index_repository` | Clones & indexes repository into ChromaDB | `repo_url` (str) |
| `search_codebase` | Performs semantic similarity search in vector store | `query` (str), `top_k` (int) |
| `ask_codebase_question` | RAG query returning context-grounded answers | `question` (str) |
| `analyze_file` | Architectural & functional analysis of a specific file | `file_path` (str) |
| `explain_architecture` | Generates system architecture overview | None |

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## 📜 License

MIT License © 2026
