from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage

from codex_rag.codex_chat_model import CodexCLIChatModel, messages_to_prompt


def test_messages_to_prompt_preserves_roles() -> None:
    prompt = messages_to_prompt(
        [SystemMessage(content="Use context"), HumanMessage(content="Question")]
    )
    assert "[SYSTEM]\nUse context" in prompt
    assert "[USER]\nQuestion" in prompt
    assert "Do not use tools" in prompt


def test_codex_model_reads_output_file(monkeypatch, tmp_path: Path) -> None:
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["input"] = kwargs["input"]
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("grounded answer", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    model = CodexCLIChatModel(codex_home=tmp_path, model="test-model")
    response = model.invoke("hello")

    assert response.content == "grounded answer"
    assert "--ephemeral" in observed["command"]
    assert "read-only" in observed["command"]
    assert "test-model" in observed["command"]
    assert "hello" in observed["input"]

