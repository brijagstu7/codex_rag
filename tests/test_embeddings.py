from codex_rag.index import LocalHashEmbeddings


def test_local_hash_embeddings_are_normalized_and_deterministic() -> None:
    embeddings = LocalHashEmbeddings(dimensions=128)
    first = embeddings.embed_query("Codex 使用本地配置")
    second = embeddings.embed_query("Codex 使用本地配置")

    assert first == second
    assert len(first) == 128
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9
