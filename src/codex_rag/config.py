from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings loaded from environment variables and an optional .env file."""

    docs_dir: Path
    index_dir: Path
    embedding_model: str
    embedding_cache_dir: Path
    top_k: int
    chunk_size: int
    chunk_overlap: int
    codex_binary: str
    codex_model: str | None
    codex_home: Path
    codex_timeout_seconds: int
    codex_reasoning_effort: str

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> Settings:
        if env_file:
            load_dotenv(Path(env_file), override=False)

        cwd = Path.cwd()

        def project_path(name: str, default: str) -> Path:
            value = Path(os.getenv(name, default)).expanduser()
            return value if value.is_absolute() else cwd / value

        settings = cls(
            docs_dir=project_path("RAG_DOCS_DIR", "knowledge"),
            index_dir=project_path("RAG_INDEX_DIR", ".rag_index"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "local-hash-v1"),
            embedding_cache_dir=project_path("RAG_EMBEDDING_CACHE_DIR", ".cache/fastembed"),
            top_k=int(os.getenv("RAG_TOP_K", "4")),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "120")),
            codex_binary=os.getenv("CODEX_BINARY", "codex"),
            codex_model=_optional_env("CODEX_MODEL"),
            codex_home=Path(os.getenv("CODEX_HOME", "~/.codex")).expanduser(),
            codex_timeout_seconds=int(os.getenv("CODEX_TIMEOUT_SECONDS", "180")),
            codex_reasoning_effort=os.getenv("CODEX_REASONING_EFFORT", "medium"),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.top_k < 1:
            raise ValueError("RAG_TOP_K must be at least 1")
        if self.chunk_size < 100:
            raise ValueError("RAG_CHUNK_SIZE must be at least 100")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be non-negative and smaller than chunk size")
        if self.codex_timeout_seconds < 1:
            raise ValueError("CODEX_TIMEOUT_SECONDS must be positive")
