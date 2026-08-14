from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from config import GEMINI_API_KEY, OPENAI_API_KEY, LLM_PROVIDER, GEMINI_MODEL, OPENAI_MODEL
from embeddings import get_vector_store

def get_llm():
    """Initializes and returns the configured Chat LLM."""
    provider = LLM_PROVIDER.lower()

    if provider == "gemini" and GEMINI_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print(f"[rag] Initializing ChatGoogleGenerativeAI ({GEMINI_MODEL})...")
            return ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GEMINI_API_KEY,
                temperature=0.2
            )
        except Exception as e:
            print(f"[rag] Failed to init Gemini LLM ({e}). Trying OpenAI...")

    if OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            print(f"[rag] Initializing ChatOpenAI ({OPENAI_MODEL})...")
            return ChatOpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=0.2
            )
        except Exception as e:
            print(f"[rag] Failed to init OpenAI LLM: {e}")

    raise RuntimeError("No valid LLM provider configured. Please provide GEMINI_API_KEY or OPENAI_API_KEY in .env.")

def format_context_snippets(docs: List[Document]) -> str:
    """Formats retrieved code chunks into clear context blocks with file paths and line ranges."""
    formatted = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown file")
        start_line = doc.metadata.get("start_line", 1)
        end_line = doc.metadata.get("end_line", 1)
        lang = doc.metadata.get("language", "")
        
        header = f"--- Snippet {idx}: {source} (Lines {start_line}-{end_line}) ---"
        code_block = f"```{lang}\n{doc.page_content}\n```"
        formatted.append(f"{header}\n{code_block}")
        
    return "\n\n".join(formatted)

def query_codebase(question: str, top_k: int = 5) -> Dict[str, Any]:
    """Queries the indexed vector store for relevant snippets and generates a context-grounded response."""
    try:
        vector_store = get_vector_store()
        retrieved_docs = vector_store.similarity_search(question, k=top_k)

        if not retrieved_docs:
            return {
                "answer": "No relevant codebase information found in the vector index. Please ensure a repository is indexed.",
                "sources": []
            }

        context = format_context_snippets(retrieved_docs)
        llm = get_llm()

        system_prompt = (
            "You are an expert AI Software Engineering Assistant specializing in code comprehension, "
            "architecture analysis, and developer support.\n"
            "Answer the user's question accurately using ONLY the provided code snippets below.\n"
            "Always cite the exact file path and line numbers when referring to specific functions, classes, or code logic.\n"
            "If the provided context does not contain enough information, state clearly what is missing.\n\n"
            f"=== CODEBASE CONTEXT ===\n{context}\n\n"
            f"=== USER QUESTION ===\n{question}"
        )

        response = llm.invoke(system_prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)

        sources = [
            {
                "source": d.metadata.get("source", ""),
                "start_line": d.metadata.get("start_line", 1),
                "end_line": d.metadata.get("end_line", 1),
                "snippet": d.page_content[:200] + "..."
            }
            for d in retrieved_docs
        ]

        return {
            "answer": answer_text,
            "sources": sources,
            "context_used": context
        }

    except Exception as e:
        return {
            "answer": f"Error processing query: {str(e)}",
            "sources": []
        }

def explain_codebase_architecture() -> Dict[str, Any]:
    """Generates a high-level system architecture analysis of the indexed codebase."""
    prompt = (
        "Analyze the indexed codebase architecture. Identify main entry points, core design patterns, "
        "data flow, external dependencies, and key modules."
    )
    return query_codebase(prompt, top_k=8)

def analyze_single_file(file_path: str) -> Dict[str, Any]:
    """Retrieves snippets related to a specific file and provides an analysis of its implementation."""
    prompt = f"Provide a complete analysis of the file '{file_path}'. Explain its primary purpose, classes, functions, and key methods."
    return query_codebase(prompt, top_k=5)
