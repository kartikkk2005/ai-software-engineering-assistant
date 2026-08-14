from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Mapping detected language strings to LangChain Language enums
LANGCHAIN_LANGUAGE_MAP: Dict[str, Language] = {
    "python": Language.PYTHON,
    "javascript": Language.JS,
    "typescript": Language.TS,
    "java": Language.JAVA,
    "cpp": Language.CPP,
    "c": Language.C,
    "go": Language.GO,
    "rust": Language.RUST,
    "ruby": Language.RUBY,
    "php": Language.PHP,
    "html": Language.HTML,
    "markdown": Language.MARKDOWN,
}

def compute_chunk_line_numbers(parent_content: str, chunk_content: str) -> tuple[int, int]:
    """Estimates start and end line numbers of a chunk within its parent document."""
    try:
        # Find start offset of chunk in full text
        start_idx = parent_content.find(chunk_content)
        if start_idx == -1:
            # Fallback: fuzzy search based on first non-empty line
            first_line = next((line for line in chunk_content.splitlines() if line.strip()), "")
            if first_line:
                start_idx = parent_content.find(first_line)

        if start_idx != -1:
            start_line = parent_content[:start_idx].count("\n") + 1
            chunk_line_count = chunk_content.count("\n")
            end_line = start_line + chunk_line_count
            return start_line, max(start_line, end_line)
    except Exception:
        pass
    return 1, 1

def split_documents(
    documents: List[Document], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[Document]:
    """Splits codebase documents into code-aware text chunks with line number metadata."""
    chunked_docs: List[Document] = []

    for doc in documents:
        lang_str = doc.metadata.get("language", "text")
        parent_content = doc.page_content

        # Select appropriate text splitter
        if lang_str in LANGCHAIN_LANGUAGE_MAP:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=LANGCHAIN_LANGUAGE_MAP[lang_str],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )

        raw_chunks = splitter.split_documents([doc])

        # Enhance chunk metadata with start_line & end_line
        for idx, chunk in enumerate(raw_chunks):
            start_line, end_line = compute_chunk_line_numbers(parent_content, chunk.page_content)
            
            chunk.metadata.update({
                "chunk_id": f"{doc.metadata.get('source')}_chunk_{idx}",
                "start_line": start_line,
                "end_line": end_line,
                "chunk_index": idx,
                "total_chunks": len(raw_chunks)
            })
            chunked_docs.append(chunk)

    print(f"[chunker] Created {len(chunked_docs)} code chunks from {len(documents)} parent documents.")
    return chunked_docs
