from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from codex_rag.config import Settings
from codex_rag.index import read_manifest
from codex_rag.service import RAGService

app = typer.Typer(
    no_args_is_help=True,
    help="Local LangChain RAG backed by your authenticated Codex CLI.",
)
console = Console()


def _settings() -> Settings:
    return Settings.from_env()


def _print_sources(sources: list[dict[str, object]]) -> None:
    if not sources:
        return
    table = Table(title="检索来源", show_lines=True)
    table.add_column("#", style="cyan", width=4)
    table.add_column("文件")
    table.add_column("页", width=6)
    table.add_column("片段")
    for source in sources:
        table.add_row(
            str(source["number"]),
            str(source["source"]),
            str(source["page"] or "-"),
            str(source["preview"]),
        )
    console.print(table)


@app.command()
def doctor() -> None:
    """Check Codex authentication/configuration and the local index."""

    settings = _settings()
    binary = shutil.which(settings.codex_binary)
    table = Table(title="Codex RAG 环境检查")
    table.add_column("项目")
    table.add_column("状态")
    table.add_column("详情")

    if binary:
        result = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, check=False
        )
        table.add_row("Codex CLI", "OK", result.stdout.strip() or binary)
    else:
        table.add_row("Codex CLI", "FAIL", f"未找到 {settings.codex_binary}")

    config_file = settings.codex_home / "config.toml"
    auth_file = settings.codex_home / "auth.json"
    table.add_row("Codex 配置", "OK" if config_file.exists() else "FAIL", str(config_file))
    table.add_row("Codex 认证", "OK" if auth_file.exists() else "FAIL", str(auth_file))

    manifest = read_manifest(settings.index_dir)
    table.add_row(
        "向量索引",
        "OK" if manifest else "MISSING",
        json.dumps(manifest, ensure_ascii=False) if manifest else str(settings.index_dir),
    )
    console.print(table)


@app.command()
def ingest(
    path: Annotated[
        Path | None, typer.Argument(help="文档目录；默认读取 RAG_DOCS_DIR")
    ] = None,
) -> None:
    """Load documents, split them, and rebuild the local FAISS index."""

    service = RAGService(_settings())
    with console.status("正在加载文档并构建索引..."):
        result = service.ingest(path)
    console.print(
        f"[green]索引完成[/green]：{result['files']} 个文件，"
        f"{result['documents']} 个文档，{result['chunks']} 个片段。"
    )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="要询问知识库的问题")],
    top_k: Annotated[int | None, typer.Option("--top-k", "-k", min=1)] = None,
    show_sources: Annotated[bool, typer.Option("--sources/--no-sources")] = True,
) -> None:
    """Retrieve relevant chunks and ask Codex to answer from them."""

    service = RAGService(_settings())
    with console.status("正在检索并调用 Codex..."):
        answer = service.answer(question, top_k=top_k)
    console.print(Panel(answer.text, title="回答", border_style="green"))
    if show_sources:
        _print_sources(answer.sources)


@app.command()
def chat(
    top_k: Annotated[int | None, typer.Option("--top-k", "-k", min=1)] = None,
) -> None:
    """Start an interactive question-answer loop."""

    service = RAGService(_settings())
    console.print("输入问题开始检索；输入 [bold]exit[/bold] 或 [bold]quit[/bold] 退出。")
    while True:
        try:
            question = console.input("\n[bold cyan]你：[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        try:
            with console.status("正在检索并调用 Codex..."):
                answer = service.answer(question, top_k=top_k)
            console.print(Panel(answer.text, title="Codex", border_style="green"))
            _print_sources(answer.sources)
        except Exception as exc:
            console.print(f"[red]错误：[/red]{exc}")
