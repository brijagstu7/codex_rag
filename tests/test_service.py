from langchain_core.documents import Document

from codex_rag.service import format_context


def test_format_context_adds_source_numbers_and_page() -> None:
    context = format_context(
        [
            Document(
                page_content="A useful fact",
                metadata={"source": "/docs/a.pdf", "page": 1, "start_index": 20},
            )
        ]
    )
    assert "[1]" in context
    assert "/docs/a.pdf" in context
    assert "page=2" in context
    assert "A useful fact" in context

