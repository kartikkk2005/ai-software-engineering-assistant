import json
from pathlib import Path
from fastmcp import FastMCP

from github_loader import clone_or_download_repo, get_codebase_files
from file_loader import load_codebase_documents
from chunker import split_documents
from embeddings import index_documents, get_vector_store
from rag import query_codebase, explain_codebase_architecture, analyze_single_file

# Initialize FastMCP Server
mcp = FastMCP("AI Software Engineering Assistant")

@mcp.tool()
def index_repository(repo_url: str) -> str:
    """
    Clones a GitHub repository (or reads local directory), chunks the code files, 
    and indexes them into the ChromaDB vector database.
    
    Args:
        repo_url: GitHub repository URL (e.g. 'https://github.com/owner/repo') or local folder path.
    """
    try:
        print(f"[MCP Server] Received request to index repo: {repo_url}")
        repo_path = Path(repo_url) if Path(repo_url).exists() else clone_or_download_repo(repo_url)
        files = get_codebase_files(repo_path)
        
        if not files:
            return f"Error: No valid code files found in '{repo_path}'."
            
        docs = load_codebase_documents(files, repo_path)
        chunks = split_documents(docs)
        index_documents(chunks, reset_db=True)
        
        return f"Successfully indexed {len(docs)} documents ({len(chunks)} chunks) from '{repo_path.name}' into ChromaDB."
    except Exception as e:
        return f"Failed to index repository: {str(e)}"

@mcp.tool()
def search_codebase(query: str, top_k: int = 5) -> str:
    """
    Performs semantic vector search in the indexed codebase and returns top matching snippets.
    
    Args:
        query: Search term or description (e.g., 'authentication logic', 'database connection').
        top_k: Number of relevant code snippets to retrieve (default: 5).
    """
    try:
        vector_store = get_vector_store()
        results = vector_store.similarity_search(query, k=top_k)
        
        if not results:
            return "No matching code snippets found."
            
        snippets = []
        for i, doc in enumerate(results, 1):
            src = doc.metadata.get("source", "unknown")
            start = doc.metadata.get("start_line", 1)
            end = doc.metadata.get("end_line", 1)
            lang = doc.metadata.get("language", "")
            snippets.append(f"Snippet {i} | File: {src} (Lines {start}-{end})\n```{lang}\n{doc.page_content}\n```")
            
        return "\n\n".join(snippets)
    except Exception as e:
        return f"Search error: {str(e)}"

@mcp.tool()
def ask_codebase_question(question: str) -> str:
    """
    Answers complex questions about the indexed codebase using Retrieval-Augmented Generation (RAG).
    
    Args:
        question: Question about code logic, architecture, bug investigation, or feature explanation.
    """
    try:
        res = query_codebase(question)
        answer = res.get("answer", "No answer generated.")
        sources = res.get("sources", [])
        
        source_str = ""
        if sources:
            source_lines = [f"- {s['source']} (Lines {s['start_line']}-{s['end_line']})" for s in sources]
            source_str = "\n\n### Referenced Sources:\n" + "\n".join(source_lines)
            
        return answer + source_str
    except Exception as e:
        return f"Error executing RAG query: {str(e)}"

@mcp.tool()
def analyze_file(file_path: str) -> str:
    """
    Provides a detailed architectural and functional breakdown of a specific file in the repository.
    
    Args:
        file_path: Relative path of the file to analyze (e.g., 'src/auth/login.py').
    """
    try:
        res = analyze_single_file(file_path)
        return res.get("answer", f"Could not analyze file '{file_path}'.")
    except Exception as e:
        return f"File analysis error: {str(e)}"

@mcp.tool()
def explain_architecture() -> str:
    """
    Analyzes the indexed codebase and returns a comprehensive architectural overview.
    """
    try:
        res = explain_codebase_architecture()
        return res.get("answer", "Could not generate architecture overview.")
    except Exception as e:
        return f"Architecture analysis error: {str(e)}"

if __name__ == "__main__":
    print("[FastMCP] Starting AI Software Engineering Assistant MCP Server...")
    mcp.run()
