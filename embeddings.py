import time
import shutil
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from config import GEMINI_API_KEY, OPENAI_API_KEY, EMBEDDING_PROVIDER, CHROMA_DB_DIR

def get_embedding_function():
    """Initializes and returns the configured embedding function with fallbacks."""
    provider = EMBEDDING_PROVIDER.lower()

    if provider == "gemini" and GEMINI_API_KEY:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            print("[embeddings] Initializing Gemini GoogleGenerativeAIEmbeddings (text-embedding-004)...")
            return GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=GEMINI_API_KEY
            )
        except Exception as e:
            print(f"[embeddings] Gemini embeddings error ({e}). Falling back...")

    if (provider == "openai" or OPENAI_API_KEY) and OPENAI_API_KEY:
        try:
            from langchain_openai import OpenAIEmbeddings
            print("[embeddings] Initializing OpenAIEmbeddings (text-embedding-3-small)...")
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=OPENAI_API_KEY
            )
        except Exception as e:
            print(f"[embeddings] OpenAI embeddings error ({e}). Falling back...")

    # Default / Local Fallback: HuggingFace sentence-transformers
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("[embeddings] Initializing local HuggingFaceEmbeddings (all-MiniLM-L6-v2)...")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize any embedding provider: {e}")

def get_vector_store(persist_directory: Optional[str] = None) -> Chroma:
    """Returns an existing Chroma vector store instance."""
    target_dir = persist_directory or str(CHROMA_DB_DIR)
    embeddings = get_embedding_function()
    return Chroma(
        persist_directory=target_dir,
        embedding_function=embeddings
    )

def index_documents(
    documents: List[Document], 
    persist_directory: Optional[str] = None,
    reset_db: bool = True
) -> Optional[Chroma]:
    """Indexes document chunks into ChromaDB in batches with retry handling."""
    if not documents:
        print("[embeddings] No document chunks provided for indexing.")
        return None

    target_dir = persist_directory or str(CHROMA_DB_DIR)
    db_path = Path(target_dir)

    if reset_db and db_path.exists():
        try:
            shutil.rmtree(db_path)
            print(f"[embeddings] Reset existing vector store at '{target_dir}'.")
        except Exception as e:
            print(f"[embeddings] Warning: Failed to delete previous db folder: {e}")

    embeddings = get_embedding_function()

    vector_store = Chroma(
        persist_directory=target_dir,
        embedding_function=embeddings
    )

    batch_size = 25
    total_docs = len(documents)
    print(f"[embeddings] Indexing {total_docs} chunks in batches of {batch_size}...")

    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        max_retries = 5
        retry_delay = 3.0

        for attempt in range(max_retries):
            try:
                vector_store.add_documents(batch)
                break
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "resource_exhausted" in err_msg or "rate limit" in err_msg:
                    print(f"[embeddings] Rate limit (attempt {attempt + 1}/{max_retries}). Waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2.0
                else:
                    print(f"[embeddings] Error indexing batch at index {i}: {e}")
                    if attempt == max_retries - 1:
                        raise e

    print(f"[embeddings] Successfully indexed {total_docs} chunks into ChromaDB at '{target_dir}'.")
    return vector_store
