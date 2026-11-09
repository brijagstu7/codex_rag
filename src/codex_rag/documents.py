from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import docx2txt
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".html",
    ".htm",
}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".docx"}


def discover_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Knowledge directory does not exist: {root}")
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_EXTENSIONS else []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        content = _read_text(path)
        documents = [Document(page_content=content)]
    elif suffix == ".pdf":
        reader = PdfReader(str(path))
        documents = [
            Document(page_content=page.extract_text() or "", metadata={"page": page_number})
            for page_number, page in enumerate(reader.pages)
        ]
    elif suffix == ".docx":
        documents = [Document(page_content=docx2txt.process(str(path)) or "")]
    else:
        return []

    for document in documents:
        document.metadata["source"] = str(path.resolve())
        document.metadata["filename"] = path.name
    return documents


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def load_documents(paths: Iterable[Path]) -> list[Document]:
    documents: list[Document] = []
    errors: list[str] = []
    for path in paths:
        try:
            documents.extend(load_file(path))
        except Exception as exc:  # Keep useful files while reporting all failures together.
            errors.append(f"{path}: {exc}")
    if errors:
        raise RuntimeError("Failed to load one or more documents:\n" + "\n".join(errors))
    return documents


def split_documents(
    documents: list[Document], *, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks
