from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class CodexCLIError(RuntimeError):
    """Raised when the Codex CLI cannot produce a model response."""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def messages_to_prompt(messages: list[BaseMessage]) -> str:
    """Serialize LangChain messages into a deterministic plain-text transcript."""

    role_names = {
        "system": "SYSTEM",
        "human": "USER",
        "ai": "ASSISTANT",
        "tool": "TOOL",
    }
    sections = []
    for message in messages:
        role = role_names.get(message.type, message.type.upper())
        sections.append(f"[{role}]\n{_content_to_text(message.content)}")
    sections.append("[ASSISTANT]\nReturn only the final answer. Do not use tools or inspect files.")
    return "\n\n".join(sections)


class CodexCLIChatModel(BaseChatModel):
    """LangChain chat model adapter for the locally authenticated Codex CLI.

    Authentication and provider settings are inherited from CODEX_HOME (normally ~/.codex).
    Each invocation is ephemeral, read-only, and receives the RAG context over stdin.
    """

    codex_binary: str = "codex"
    model: str | None = None
    codex_home: Path = Field(default_factory=lambda: Path("~/.codex").expanduser())
    timeout_seconds: int = 180
    reasoning_effort: str = "medium"

    @property
    def _llm_type(self) -> str:
        return "codex-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "codex_binary": self.codex_binary,
            "model": self.model or "from-codex-config",
            "reasoning_effort": self.reasoning_effort,
        }

    def _command(self, answer_file: Path, working_dir: Path) -> list[str]:
        command = [
            self.codex_binary,
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--color",
            "never",
            "--config",
            'network_access="disabled"',
            "--config",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "--cd",
            str(working_dir),
            "--output-last-message",
            str(answer_file),
        ]
        if self.model:
            command.extend(["--model", self.model])
        command.append("-")
        return command

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        prompt = messages_to_prompt(messages)
        env = os.environ.copy()
        env["CODEX_HOME"] = str(self.codex_home)

        try:
            with tempfile.TemporaryDirectory(prefix="codex-rag-") as temp_dir:
                working_dir = Path(temp_dir)
                answer_file = working_dir / "answer.txt"
                result = subprocess.run(
                    self._command(answer_file, working_dir),
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                    env=env,
                )
                if result.returncode != 0:
                    details = (result.stderr or result.stdout).strip()
                    raise CodexCLIError(
                        f"Codex CLI exited with status {result.returncode}: {details[-2000:]}"
                    )
                if not answer_file.exists():
                    raise CodexCLIError("Codex CLI completed without writing its final answer")
                answer = answer_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise CodexCLIError(f"Codex executable not found: {self.codex_binary}") from exc
        except subprocess.TimeoutExpired as exc:
            raise CodexCLIError(
                f"Codex CLI timed out after {self.timeout_seconds} seconds"
            ) from exc

        if not answer:
            raise CodexCLIError("Codex CLI returned an empty answer")

        from langchain_core.messages import AIMessage

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=answer))],
            llm_output={"provider": "codex-cli"},
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        return await asyncio.to_thread(self._generate, messages, stop, None, **kwargs)
