from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from codex_rag.config import Settings

MANIFEST_FILE = "manifest.json"
FAISS_FILE = "index.faiss"
DOCUMENTS_FILE = "documents.json"


class LocalHashEmbeddings(Embeddings):
    """Dependency-free multilingual hashing embeddings for offline retrieval.

    Character n-grams make the representation useful for Chinese and mixed-language text.
    It is lexical rather than neural-semantic, but deterministic and needs no model download.
    """

    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions

    @staticmethod
    def _features(text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        compact = normalized.replace(" ", "")
        features = re.findall(r"[a-z0-9_]+", normalized)
        for size in (1, 2, 3):
            features.extend(
                compact[index : index + size]
                for index in range(len(compact) - size + 1)
            )
        return features

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dimensions
            vector[index] += 1.0 if value & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FastEmbedLocalEmbeddings(Embeddings):
    """Small LangChain adapter around FastEmbed's CPU/ONNX text embeddings."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "FastEmbed is optional. Install it with: pip install -e '.[semantic]'"
            ) from exc
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self._model.embed([text])).tolist()


class LocalFAISSVectorStore:
    """Minimal FAISS store using LangChain Documents and Embeddings.

    Documents are persisted as JSON instead of pickle, so a local index can be loaded without
    enabling unsafe deserialization.
    """

    def __init__(
        self, index: faiss.Index, documents: list[Document], embeddings: Embeddings
    ) -> None:
        self.index = index
        self.documents = documents
        self.embeddings = embeddings

    @classmethod
    def from_documents(
        cls, documents: list[Document], embeddings: Embeddings
    ) -> LocalFAISSVectorStore:
        vectors = np.asarray(
            embeddings.embed_documents([document.page_content for document in documents]),
            dtype="float32",
        )
        if vectors.ndim != 2 or not len(vectors):
            raise ValueError("Embedding model returned no document vectors")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index=index, documents=documents, embeddings=embeddings)

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        if not self.documents:
            return []
        query_vector = np.asarray([self.embeddings.embed_query(query)], dtype="float32")
        faiss.normalize_L2(query_vector)
        limit = min(k, len(self.documents))
        _, indices = self.index.search(query_vector, limit)
        return [self.documents[index] for index in indices[0] if index >= 0]

    def save_local(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / FAISS_FILE))
        payload = [
            {"page_content": document.page_content, "metadata": document.metadata}
            for document in self.documents
        ]
        (directory / DOCUMENTS_FILE).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load_local(
        cls, directory: Path, embeddings: Embeddings
    ) -> LocalFAISSVectorStore:
        index = faiss.read_index(str(directory / FAISS_FILE))
        payload = json.loads((directory / DOCUMENTS_FILE).read_text(encoding="utf-8"))
        documents = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in payload
        ]
        if index.ntotal != len(documents):
            raise ValueError("FAISS index and document metadata have different lengths")
        return cls(index=index, documents=documents, embeddings=embeddings)


def create_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_model == "local-hash-v1":
        return LocalHashEmbeddings()
    return FastEmbedLocalEmbeddings(
        model_name=settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
    )


def save_index(
    chunks: list[Document], embeddings: Embeddings, index_dir: Path, *, model_name: str
) -> None:
    if not chunks:
        raise ValueError("Cannot build an index from zero document chunks")

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rag-index-", dir=index_dir.parent) as temp_dir:
        temp_path = Path(temp_dir)
        store = LocalFAISSVectorStore.from_documents(chunks, embeddings)
        store.save_local(temp_path)
        manifest = {
            "embedding_model": model_name,
            "chunk_count": len(chunks),
            "source_count": len({chunk.metadata.get("source") for chunk in chunks}),
        }
        (temp_path / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if index_dir.exists():
            shutil.rmtree(index_dir)
        shutil.copytree(temp_path, index_dir)


def load_index(index_dir: Path, embeddings: Embeddings) -> LocalFAISSVectorStore:
    required = [index_dir / MANIFEST_FILE, index_dir / FAISS_FILE, index_dir / DOCUMENTS_FILE]
    if not all(path.exists() for path in required):
        raise FileNotFoundError(
            f"Vector index not found at {index_dir}. Run 'codex-rag ingest' first."
        )
    return LocalFAISSVectorStore.load_local(index_dir, embeddings)


def read_manifest(index_dir: Path) -> dict[str, object]:
    path = index_dir / MANIFEST_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
