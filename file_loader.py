from pathlib import Path
from typing import List, Dict, Optional
from langchain_core.documents import Document

# Programming language map by file extension
EXTENSION_LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".html": "html",
    ".css": "css",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
    ".sh": "bash",
    ".bash": "bash",
    ".ps1": "powershell",
    ".sql": "sql",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".rst": "rst",
    ".toml": "toml",
    ".txt": "text"
}

def detect_language(file_path: Path) -> str:
    """Detects programming language identifier from file extension."""
    ext = file_path.suffix.lower()
    return EXTENSION_LANGUAGE_MAP.get(ext, "text")

def load_file_as_document(file_path: Path, base_repo_path: Path) -> Optional[Document]:
    """Reads a single source file and returns a LangChain Document with metadata."""
    try:
        # Calculate relative path from repository root
        try:
            rel_path = file_path.relative_to(base_repo_path).as_posix()
        except ValueError:
            rel_path = file_path.name

        # Read file with encoding fallback
        content = None
        for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, Exception):
                continue
                
        if content is None:
            return None
            
        # Skip empty files
        if not content.strip():
            return None

        lines = content.splitlines()
        line_count = len(lines)
        language = detect_language(file_path)

        metadata = {
            "source": rel_path,
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_extension": file_path.suffix.lower(),
            "language": language,
            "total_lines": line_count,
            "start_line": 1,
            "end_line": line_count
        }

        return Document(page_content=content, metadata=metadata)
        
    except Exception as e:
        print(f"[file_loader] Error reading '{file_path}': {e}")
        return None

def load_codebase_documents(file_paths: List[Path], base_repo_path: Path) -> List[Document]:
    """Loads multiple codebase files into a list of LangChain Document objects."""
    documents = []
    for file_path in file_paths:
        doc = load_file_as_document(file_path, base_repo_path)
        if doc:
            documents.append(doc)
    print(f"[file_loader] Loaded {len(documents)} valid documents from {len(file_paths)} scanned files.")
    return documents
