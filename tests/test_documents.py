from pathlib import Path

from codex_rag.documents import discover_files, load_documents, split_documents


def test_load_and_split_text_document(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "note.md").write_text("第一段。\n\n第二段。" * 20, encoding="utf-8")
    (knowledge / "ignored.bin").write_bytes(b"ignored")

    files = discover_files(knowledge)
    documents = load_documents(files)
    chunks = split_documents(documents, chunk_size=100, chunk_overlap=10)

    assert [path.name for path in files] == ["note.md"]
    assert chunks
    assert all(chunk.metadata["filename"] == "note.md" for chunk in chunks)
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == list(range(len(chunks)))

