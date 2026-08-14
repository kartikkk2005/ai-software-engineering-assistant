import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Base Workspace Directory
BASE_DIR = Path(__file__).parent.resolve()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Providers & Models
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Storage Paths
CHROMA_DB_DIR = BASE_DIR / os.getenv("CHROMA_DB_DIR", "vector_db")
REPOS_DIR = BASE_DIR / os.getenv("REPOS_DIR", "repos")

# Ensure required storage directories exist
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
REPOS_DIR.mkdir(parents=True, exist_ok=True)

def validate_config():
    """Validates that at least one LLM / Embedding API key or HuggingFace fallback is configured."""
    has_gemini = bool(GEMINI_API_KEY)
    has_openai = bool(OPENAI_API_KEY)
    return {
        "gemini": has_gemini,
        "openai": has_openai,
        "embedding_provider": EMBEDDING_PROVIDER,
        "llm_provider": LLM_PROVIDER
    }
