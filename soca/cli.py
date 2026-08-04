from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import click
from rich.console import Console
from rich.table import Table

from soca.app import run_voice_loop
from soca.app.profiles import render_profiles, render_status
from soca.app.style.palette import ALT, st
from soca.app.text_chat import run_text_chat
from soca.app.text_runtime import TextRuntimeConfig, resolve_text_runtime_config, run_text_ask
from soca.asr.registry import ASR_MODEL_REGISTRY, DEFAULT_ASR_MODEL_KEY
from soca.config import DEFAULT_MAX_TOKENS
from soca.core import (
    DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    VOICE_RUNTIME_PROFILES,
    resolve_voice_runtime_config,
)
from soca.knowledge.index.persistence import default_index_home
from soca.knowledge.indexing.coordinator import IndexCoordinator
from soca.knowledge.indexing.identity import CorpusSpec
from soca.knowledge.indexing.models import (
    MODEL_REGISTRY,
    install_model,
    load_model,
    model_spec,
    model_status,
)
from soca.knowledge.indexing.watcher import IndexWatcher
from soca.knowledge.markdown_vault import MarkdownVaultKnowledgeSource
from soca.knowledge.retrievers.dense import default_model_home
from soca.knowledge.vault import default_vault_root
from soca.llm.registry import DEFAULT_LLM_MODEL_KEY, LLM_MODEL_REGISTRY
from soca.memory import SessionPersistence

console = Console()
REPO_ROOT = Path(__file__).resolve().parents[1]
UI_VAULT_ENV = "SOCA_VAULT"


def resolve_ui_vault(vault: Path | None) -> Path:
    """Resolve the UI vault without baking a repository fixture into production."""

    if vault is not None:
        return vault.expanduser().resolve()
    configured = os.environ.get(UI_VAULT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return default_vault_root()


def run_module(module: str, *args: str) -> None:
    """Run a Python module using the current interpreter."""
    cmd = [sys.executable, "-m", module, *args]
    raise SystemExit(subprocess.call(cmd, cwd=REPO_ROOT))


def run_script(script: str, *args: str) -> None:
    """Run a script file using the current interpreter."""
    cmd = [sys.executable, script, *args]
    raise SystemExit(subprocess.call(cmd, cwd=REPO_ROOT))


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="soca")
def main() -> None:
    """SoCa local Vietnamese voice assistant toolkit."""


def _index_context(
    vault: Path,
    corpus: str,
    *,
    index_home: Path | None,
    model_key: str | None = None,
) -> IndexCoordinator:
    include_globs = ("wiki/**/*.md",) if corpus == "knowledge" else ("memory/**/*.md",)
    spec = CorpusSpec(vault_path=vault, kind=corpus, include_globs=include_globs)  # type: ignore[arg-type]
    reader = MarkdownVaultKnowledgeSource(vault, include_globs=include_globs)
    model = None
    if model_key is not None:
        try:
            model = load_model(model_key, allow_download=False)
        except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise click.ClickException(
                f"Embedding model chưa được provision: {model_key}. "
                f"Chạy `soca knowledge model install {model_key}`. Chi tiết: {exc}"
            ) from exc
    return IndexCoordinator(
        reader,
        spec=spec,
        index_home=index_home or default_index_home(vault),
        model=model,
    )


@main.group("knowledge")
def knowledge_group() -> None:
    """Manage local knowledge/memory indexes and models."""


@knowledge_group.group("index")
def knowledge_index_group() -> None:
    """Inspect and build the transactional index."""


@knowledge_index_group.command("status")
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=default_vault_root(),
    show_default=True,
)
@click.option(
    "--corpus", type=click.Choice(["knowledge", "memory"]), default="knowledge", show_default=True
)
@click.option("--index-home", type=click.Path(path_type=Path), default=None)
@click.option(
    "--model",
    "model_key",
    type=click.Choice([item.key for item in MODEL_REGISTRY]),
    default="aiteamvn-v2",
    show_default=True,
)
@click.option("--json", "as_json", is_flag=True, help="Print machine-readable status.")
def knowledge_index_status(
    vault: Path, corpus: str, index_home: Path | None, model_key: str, as_json: bool
) -> None:
    """Show sparse/dense/model state without downloading or embedding."""
    try:
        coordinator = _index_context(vault, corpus, index_home=index_home, model_key=model_key)
    except click.ClickException:
        coordinator = _index_context(vault, corpus, index_home=index_home)
    payload = coordinator.status().as_dict()
    payload["model"] = model_status(model_key)
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    table = Table(title=f"SoCa index · {corpus}")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in payload.items():
        table.add_row(key, "" if value is None else str(value))
    console.print(table)


@knowledge_index_group.command("build")
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=default_vault_root(),
    show_default=True,
)
@click.option(
    "--corpus", type=click.Choice(["knowledge", "memory"]), default="knowledge", show_default=True
)
@click.option("--index-home", type=click.Path(path_type=Path), default=None)
@click.option("--dense/--sparse-only", default=True, show_default=True)
@click.option(
    "--verify-content", is_flag=True, help="Read all files even when stat metadata is unchanged."
)
def knowledge_index_build(
    vault: Path, corpus: str, index_home: Path | None, dense: bool, verify_content: bool
) -> None:
    """Synchronize sparse and optionally build a dense generation."""
    _run_index_build(
        vault,
        corpus,
        index_home=index_home,
        dense=dense,
        verify_content=verify_content,
        force_dense=False,
    )


def _run_index_build(
    vault: Path,
    corpus: str,
    *,
    index_home: Path | None,
    dense: bool,
    verify_content: bool,
    force_dense: bool,
) -> None:
    coordinator = _index_context(
        vault,
        corpus,
        index_home=index_home,
        model_key="aiteamvn-v2" if dense else None,
    )
    try:
        report = coordinator.build_blocking(
            dense=dense,
            verify_content=verify_content,
            force_dense=force_dense,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    payload = coordinator.status().as_dict()
    payload.update(
        {
            "sparse_changed": report.sparse.changed,
            "documents_added": report.sparse.added,
            "documents_removed": report.sparse.removed,
            "metadata_only": report.sparse.metadata_only,
            "dense_built": report.dense is not None,
        }
    )
    click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))


@knowledge_index_group.command("rebuild")
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=default_vault_root(),
    show_default=True,
)
@click.option(
    "--corpus", type=click.Choice(["knowledge", "memory"]), default="knowledge", show_default=True
)
@click.option("--index-home", type=click.Path(path_type=Path), default=None)
def knowledge_index_rebuild(vault: Path, corpus: str, index_home: Path | None) -> None:
    """Reconcile the vault and build the current dense generation."""
    _run_index_build(
        vault,
        corpus,
        index_home=index_home,
        dense=True,
        verify_content=True,
        force_dense=True,
    )


@knowledge_index_group.command("verify")
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=default_vault_root(),
    show_default=True,
)
@click.option(
    "--corpus", type=click.Choice(["knowledge", "memory"]), default="knowledge", show_default=True
)
@click.option("--index-home", type=click.Path(path_type=Path), default=None)
@click.option("--json", "as_json", is_flag=True)
def knowledge_index_verify(
    vault: Path, corpus: str, index_home: Path | None, as_json: bool
) -> None:
    """Verify SQLite integrity and generation file ownership."""
    coordinator = _index_context(vault, corpus, index_home=index_home)
    errors = coordinator.verify()
    if as_json:
        click.echo(json.dumps({"ok": not errors, "errors": list(errors)}, ensure_ascii=False))
    elif errors:
        raise click.ClickException("; ".join(errors))
    else:
        click.echo("index ok")
    if errors:
        raise click.exceptions.Exit(1)


@knowledge_index_group.command("gc")
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=default_vault_root(),
    show_default=True,
)
@click.option(
    "--corpus", type=click.Choice(["knowledge", "memory"]), default="knowledge", show_default=True
)
@click.option("--index-home", type=click.Path(path_type=Path), default=None)
@click.option("--apply", is_flag=True, help="Actually delete candidates; default is dry-run.")
def knowledge_index_gc(vault: Path, corpus: str, index_home: Path | None, apply: bool) -> None:
    """List or remove failed/superseded derived generations."""
    coordinator = _index_context(vault, corpus, index_home=index_home)
    candidates = coordinator.gc(apply=apply)
    for candidate in candidates:
        click.echo(("deleted " if apply else "candidate ") + candidate)


@knowledge_index_group.command("inspect")
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=default_vault_root(),
    show_default=True,
)
@click.option(
    "--corpus", type=click.Choice(["knowledge", "memory"]), default="knowledge", show_default=True
)
@click.option("--index-home", type=click.Path(path_type=Path), default=None)
def knowledge_index_inspect(vault: Path, corpus: str, index_home: Path | None) -> None:
    """Print generations, pointers and jobs for operator inspection."""
    coordinator = _index_context(vault, corpus, index_home=index_home)
    click.echo(json.dumps(coordinator.inspect(), ensure_ascii=False, sort_keys=True))


@knowledge_index_group.command("migrate")
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=default_vault_root(),
    show_default=True,
)
@click.option(
    "--corpus", type=click.Choice(["knowledge", "memory"]), default="knowledge", show_default=True
)
@click.option("--index-home", type=click.Path(path_type=Path), default=None)
def knowledge_index_migrate(vault: Path, corpus: str, index_home: Path | None) -> None:
    """Import a valid v1 sparse snapshot and reconcile it with the vault."""
    coordinator = _index_context(vault, corpus, index_home=index_home)
    result = coordinator.migrate_legacy(verify_content=True)
    click.echo(
        json.dumps(
            {
                "revision": result.revision,
                "changed": result.changed,
                "documents": len(result.index.records),
                "chunks": len(result.index.chunks),
            },
            sort_keys=True,
        )
    )


@knowledge_index_group.command("rollback")
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=default_vault_root(),
    show_default=True,
)
@click.option(
    "--corpus", type=click.Choice(["knowledge", "memory"]), default="knowledge", show_default=True
)
@click.option("--index-home", type=click.Path(path_type=Path), default=None)
def knowledge_index_rollback(vault: Path, corpus: str, index_home: Path | None) -> None:
    """Swap the active generation with the compatible previous generation."""
    coordinator = _index_context(
        vault,
        corpus,
        index_home=index_home,
        model_key="aiteamvn-v2",
    )
    try:
        generation = coordinator.rollback()
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps({"active_generation": generation}, sort_keys=True))


@knowledge_index_group.command("watch")
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=default_vault_root(),
    show_default=True,
)
@click.option(
    "--corpus", type=click.Choice(["knowledge", "memory"]), default="knowledge", show_default=True
)
@click.option("--index-home", type=click.Path(path_type=Path), default=None)
@click.option("--interval", type=click.FloatRange(min=0.25), default=2.0, show_default=True)
def knowledge_index_watch(
    vault: Path,
    corpus: str,
    index_home: Path | None,
    interval: float,
) -> None:
    """Continuously reconcile sparse and dense generations."""
    import time

    coordinator = _index_context(
        vault,
        corpus,
        index_home=index_home,
        model_key="aiteamvn-v2",
    )
    watcher = IndexWatcher(
        coordinator,
        interval_seconds=interval,
        on_status=lambda status: click.echo(
            json.dumps(status.as_dict(), ensure_ascii=False, sort_keys=True)
        ),
    )
    watcher.reconcile()
    watcher.start()
    try:
        while True:
            if watcher.last_error is not None:
                raise click.ClickException(str(watcher.last_error))
            time.sleep(min(interval, 1.0))
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()


@knowledge_group.group("model")
def knowledge_model_group() -> None:
    """Provision and verify embedding model artifacts explicitly."""


@knowledge_model_group.command("list")
@click.option("--json", "as_json", is_flag=True)
def knowledge_model_list(as_json: bool) -> None:
    values = [
        {
            "key": item.key,
            "adapter": item.adapter,
            "model_id": item.model_id,
            "dimension": item.dimension,
            "source": item.source,
        }
        for item in MODEL_REGISTRY
    ]
    if as_json:
        click.echo(json.dumps(values, ensure_ascii=False, sort_keys=True))
        return
    for value in values:
        click.echo(f"{value['key']} · {value['model_id']} · {value['dimension']}D")


@knowledge_model_group.command("status")
@click.argument("key", required=False, default="aiteamvn-v2")
@click.option("--json", "as_json", is_flag=True)
def knowledge_model_status(key: str, as_json: bool) -> None:
    try:
        payload = model_status(key)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        for field, value in payload.items():
            click.echo(f"{field}: {value}")


@knowledge_model_group.command("install")
@click.argument("key", required=False, default="aiteamvn-v2")
def knowledge_model_install(key: str) -> None:
    """Download a declared model; this is the only command with network intent."""
    try:
        install_model(key)
    except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"installed {key}")


@knowledge_model_group.command("verify")
@click.argument("key", required=False, default="aiteamvn-v2")
def knowledge_model_verify(key: str) -> None:
    payload = model_status(key)
    if payload.get("state") != "installed":
        raise click.ClickException(
            f"model is not ready: {payload.get('error', payload.get('state'))}"
        )
    click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))


@knowledge_model_group.command("remove")
@click.argument("key", required=False, default="aiteamvn-v2")
@click.option(
    "--apply", is_flag=True, help="Actually remove only the exact model cache candidates."
)
def knowledge_model_remove(key: str, apply: bool) -> None:
    try:
        spec = model_spec(key)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    candidates = ((default_model_home() / spec.cache_subdirectory).resolve(),)
    for path in candidates:
        if apply:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        click.echo(("deleted " if apply else "candidate ") + str(path))


@main.command()
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=default_vault_root(),
    show_default=True,
    help="Knowledge vault root to check.",
)
def status(vault: Path) -> None:
    """Show quick SoCa CLI/runtime readiness without loading models."""
    render_status(console, vault=vault)


@main.command("profiles")
@click.option(
    "--show-paths",
    is_flag=True,
    help="Also show registry artifact paths for each profile.",
)
def profiles_command(show_paths: bool) -> None:
    """Show configured voice runtime profiles without loading models."""
    render_profiles(console, show_paths=show_paths)


@main.command("ask")
@click.argument("text", nargs=-1, required=True)
@click.option(
    "--profile",
    default=DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    type=click.Choice(sorted(VOICE_RUNTIME_PROFILES)),
    show_default=True,
    help="Runtime profile whose LLM defaults are used.",
)
@click.option(
    "--llm-model",
    default=None,
    type=click.Choice(sorted(LLM_MODEL_REGISTRY)),
    help="Override the selected profile LLM for free-chat / knowledge-LLM routes.",
)
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=default_vault_root(),
    show_default=True,
    help="Knowledge vault root containing wiki/ and memory/core.json or archive notes.",
)
@click.option("--no-memory", is_flag=True, help="Disable core/archive/session memory.")
@click.option("--no-llm", is_flag=True, help="Run tool/guardrail-only without loading LLM.")
@click.option("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, show_default=True)
@click.option("--temperature", type=float, default=0.2, show_default=True)
@click.option("--top-p", type=float, default=0.95, show_default=True)
@click.option("--knowledge-limit", type=int, default=3, show_default=True)
@click.option("--memory-context-chars", type=int, default=64_000, show_default=True)
@click.option("--memory-item-chars", type=int, default=900, show_default=True)
@click.option("--session-chars", type=int, default=60_000, show_default=True)
@click.option("--session-turns", type=int, default=6, show_default=True)
@click.option("--turn-chars", type=int, default=500, show_default=True)
@click.option(
    "--session-persistence",
    type=click.Choice(["ram_only", "local_resumable"]),
    default="ram_only",
    show_default=True,
    help="Working-memory persistence; local_resumable is opt-in.",
)
@click.option("--session-id", default="default", show_default=True)
@click.option(
    "--resume-session", is_flag=True, help="Resume the selected local session checkpoint."
)
@click.option(
    "--tool-router",
    type=click.Choice(["deterministic", "llm", "cascade"]),
    default="cascade",
    show_default=True,
)
@click.option(
    "--router-response",
    type=click.Choice(["prompt_json", "json_schema"]),
    default="json_schema",
    show_default=True,
)
@click.option("--semantic-router/--no-semantic-router", default=False, show_default=True)
@click.option("--semantic-router-threshold", type=float, default=0.58, show_default=True)
@click.option("--semantic-router-margin", type=float, default=0.0, show_default=True)
@click.option("--semantic-router-examples", type=click.Path(path_type=Path), default=None)
@click.option("--memory-limit", type=int, default=3, show_default=True)
@click.option(
    "--memory-retrieval",
    type=click.Choice(["chunk_sparse", "hybrid"]),
    default="chunk_sparse",
    show_default=True,
)
@click.option(
    "--memory-dense-backend",
    type=click.Choice(["aiteamvn_v2"]),
    default="aiteamvn_v2",
    show_default=True,
)
@click.option("--trace/--no-trace", default=False, show_default=True)
@click.option("--usage", is_flag=True, help="Show LLM token/latency usage after the turn.")
def ask(
    text: tuple[str, ...],
    profile: str,
    llm_model: str | None,
    vault: Path,
    no_memory: bool,
    no_llm: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
    knowledge_limit: int,
    memory_context_chars: int,
    memory_item_chars: int,
    session_chars: int,
    session_turns: int,
    turn_chars: int,
    session_persistence: str,
    session_id: str,
    resume_session: bool,
    tool_router: str,
    router_response: str,
    semantic_router: bool,
    semantic_router_threshold: float,
    semantic_router_margin: float,
    semantic_router_examples: Path | None,
    memory_limit: int,
    memory_retrieval: str,
    memory_dense_backend: str,
    trace: bool,
    usage: bool,
) -> None:
    """Run one text-only SoCa turn without ASR/TTS."""
    _validate_text_router_options(no_llm=no_llm, tool_router=tool_router)
    config = build_text_runtime_config(
        profile=profile,
        llm_model=llm_model,
        vault=vault,
        no_memory=no_memory,
        no_llm=no_llm,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        knowledge_limit=knowledge_limit,
        memory_context_chars=memory_context_chars,
        memory_item_chars=memory_item_chars,
        session_chars=session_chars,
        session_turns=session_turns,
        turn_chars=turn_chars,
        session_persistence=session_persistence,
        session_id=session_id,
        session_resume=resume_session,
        tool_router_mode=tool_router,
        tool_router_response_mode=router_response,
        semantic_router_enabled=semantic_router,
        semantic_router_threshold=semantic_router_threshold,
        semantic_router_margin=semantic_router_margin,
        semantic_router_examples=semantic_router_examples,
        memory_limit=memory_limit,
        memory_retrieval_mode=memory_retrieval,
        memory_dense_backend=memory_dense_backend,
    )
    run_text_ask(" ".join(text), config, console=console, show_trace=trace, show_usage=usage)


@main.command("chat")
@click.option(
    "--profile",
    default=DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    type=click.Choice(sorted(VOICE_RUNTIME_PROFILES)),
    show_default=True,
    help="Runtime profile whose LLM defaults are used.",
)
@click.option(
    "--llm-model",
    default=None,
    type=click.Choice(sorted(LLM_MODEL_REGISTRY)),
    help="Override the selected profile LLM for free-chat / knowledge-LLM routes.",
)
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=default_vault_root(),
    show_default=True,
    help="Knowledge vault root containing wiki/ and memory/core.json or archive notes.",
)
@click.option("--no-memory", is_flag=True, help="Disable core/archive/session memory.")
@click.option("--no-llm", is_flag=True, help="Run tool/guardrail-only without loading LLM.")
@click.option("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, show_default=True)
@click.option("--temperature", type=float, default=0.2, show_default=True)
@click.option("--top-p", type=float, default=0.95, show_default=True)
@click.option("--knowledge-limit", type=int, default=3, show_default=True)
@click.option("--memory-context-chars", type=int, default=64_000, show_default=True)
@click.option("--memory-item-chars", type=int, default=900, show_default=True)
@click.option("--session-chars", type=int, default=60_000, show_default=True)
@click.option("--session-turns", type=int, default=6, show_default=True)
@click.option("--turn-chars", type=int, default=500, show_default=True)
@click.option(
    "--session-persistence",
    type=click.Choice(["ram_only", "local_resumable"]),
    default="ram_only",
    show_default=True,
    help="Working-memory persistence; local_resumable is opt-in.",
)
@click.option("--session-id", default="default", show_default=True)
@click.option(
    "--resume-session", is_flag=True, help="Resume the selected local session checkpoint."
)
@click.option(
    "--tool-router",
    type=click.Choice(["deterministic", "llm", "cascade"]),
    default="cascade",
    show_default=True,
)
@click.option(
    "--router-response",
    type=click.Choice(["prompt_json", "json_schema"]),
    default="json_schema",
    show_default=True,
)
@click.option("--semantic-router/--no-semantic-router", default=False, show_default=True)
@click.option("--semantic-router-threshold", type=float, default=0.58, show_default=True)
@click.option("--semantic-router-margin", type=float, default=0.0, show_default=True)
@click.option("--semantic-router-examples", type=click.Path(path_type=Path), default=None)
@click.option("--memory-limit", type=int, default=3, show_default=True)
@click.option(
    "--memory-retrieval",
    type=click.Choice(["chunk_sparse", "hybrid"]),
    default="chunk_sparse",
    show_default=True,
)
@click.option(
    "--memory-dense-backend",
    type=click.Choice(["aiteamvn_v2"]),
    default="aiteamvn_v2",
    show_default=True,
)
@click.option("--trace/--no-trace", default=False, show_default=True)
@click.option(
    "--usage", is_flag=True, help="Show per-turn usage line; /usage shows session totals."
)
@click.pass_context
def chat(
    ctx: click.Context,
    profile: str,
    llm_model: str | None,
    vault: Path,
    no_memory: bool,
    no_llm: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
    knowledge_limit: int,
    memory_context_chars: int,
    memory_item_chars: int,
    session_chars: int,
    session_turns: int,
    turn_chars: int,
    session_persistence: str,
    session_id: str,
    resume_session: bool,
    tool_router: str,
    router_response: str,
    semantic_router: bool,
    semantic_router_threshold: float,
    semantic_router_margin: float,
    semantic_router_examples: Path | None,
    memory_limit: int,
    memory_retrieval: str,
    memory_dense_backend: str,
    trace: bool,
    usage: bool,
) -> None:
    """Run an interactive text chat session without ASR/TTS."""
    _validate_text_router_options(no_llm=no_llm, tool_router=tool_router)
    config = build_text_runtime_config(
        profile=profile,
        llm_model=llm_model,
        vault=vault,
        no_memory=no_memory,
        no_llm=no_llm,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        knowledge_limit=knowledge_limit,
        memory_context_chars=memory_context_chars,
        memory_item_chars=memory_item_chars,
        session_chars=session_chars,
        session_turns=session_turns,
        turn_chars=turn_chars,
        session_persistence=session_persistence,
        session_id=session_id,
        session_resume=resume_session,
        tool_router_mode=tool_router,
        tool_router_response_mode=router_response,
        semantic_router_enabled=semantic_router,
        semantic_router_threshold=semantic_router_threshold,
        semantic_router_margin=semantic_router_margin,
        semantic_router_examples=semantic_router_examples,
        memory_limit=memory_limit,
        memory_retrieval_mode=memory_retrieval,
        memory_dense_backend=memory_dense_backend,
    )
    ctx.exit(run_text_chat(config, console=console, show_trace=trace, show_usage=usage))


@main.command(
    "ui",
    epilog=(
        "\b\nQuick examples:\n"
        "  uv run soca ui              # mở main UI\n"
        "  uv run soca ui chat\n"
        "  uv run soca ui voice baseline"
    ),
)
@click.argument(
    "quick_mode",
    required=False,
    type=click.Choice(["status", "chat", "voice", "settings"]),
)
@click.argument(
    "quick_profile",
    required=False,
    type=click.Choice(sorted(VOICE_RUNTIME_PROFILES)),
)
@click.option(
    "--no-model",
    is_flag=True,
    help="Do not load model runtimes. Useful for UI smoke tests.",
)
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Knowledge vault root for the UI session. Overrides SOCA_VAULT; "
        "unset uses the repository Knowledge vault."
    ),
)
@click.option(
    "--session-persistence",
    type=click.Choice(["ram_only", "local_resumable"]),
    default="ram_only",
    show_default=True,
    help="Working-memory persistence; local_resumable is opt-in.",
)
@click.option("--session-id", default="default", show_default=True)
@click.option(
    "--resume-session", is_flag=True, help="Resume the selected local session checkpoint."
)
@click.pass_context
def ui(
    ctx: click.Context,
    quick_mode: str | None,
    quick_profile: str | None,
    no_model: bool,
    vault: Path | None,
    session_persistence: str,
    session_id: str,
    resume_session: bool,
) -> None:
    """Open the SoCa terminal UI (Ink) on top of `soca engine`.

    Quick form: soca ui [status|chat|voice] [profile]. Without a mode the UI
    opens on the main UI; use /settings to configure the runtime.
    """
    selected_vault = resolve_ui_vault(vault)
    if session_persistence != "ram_only" or session_id != "default" or resume_session:
        ctx.exit(
            _launch_ink_ui(
                mode=quick_mode,
                profile=quick_profile,
                no_model=no_model,
                vault=selected_vault,
                session_persistence=session_persistence,
                session_id=session_id,
                resume_session=resume_session,
            )
        )
    ctx.exit(
        _launch_ink_ui(
            mode=quick_mode,
            profile=quick_profile,
            no_model=no_model,
            vault=selected_vault,
        )
    )


def build_text_runtime_config(
    *,
    profile: str = DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    llm_model: str | None,
    vault: Path,
    no_memory: bool,
    no_llm: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
    knowledge_limit: int,
    memory_context_chars: int,
    memory_item_chars: int,
    session_chars: int,
    session_turns: int,
    turn_chars: int,
    session_persistence: str = "ram_only",
    session_id: str = "default",
    session_resume: bool = False,
    tool_router_mode: str = "cascade",
    tool_router_response_mode: str = "json_schema",
    semantic_router_enabled: bool = False,
    semantic_router_threshold: float = 0.58,
    semantic_router_margin: float = 0.0,
    semantic_router_examples: Path | None = None,
    memory_limit: int = 3,
    memory_retrieval_mode: str = "chunk_sparse",
    memory_dense_backend: str = "aiteamvn_v2",
) -> TextRuntimeConfig:
    try:
        return resolve_text_runtime_config(
            profile_key=profile,
            llm_model=llm_model,
            vault=vault,
            no_memory=no_memory,
            no_llm=no_llm,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            knowledge_limit=knowledge_limit,
            memory_context_chars=memory_context_chars,
            memory_item_chars=memory_item_chars,
            session_chars=session_chars,
            session_turns=session_turns,
            turn_chars=turn_chars,
            session_persistence=cast(SessionPersistence, session_persistence),
            session_id=session_id,
            session_resume=session_resume,
            tool_router_mode=tool_router_mode,
            tool_router_response_mode=tool_router_response_mode,
            semantic_router_enabled=semantic_router_enabled,
            semantic_router_threshold=semantic_router_threshold,
            semantic_router_margin=semantic_router_margin,
            semantic_router_examples=semantic_router_examples,
            memory_limit=memory_limit,
            memory_retrieval_mode=memory_retrieval_mode,
            memory_dense_backend=memory_dense_backend,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _validate_text_router_options(*, no_llm: bool, tool_router: str) -> None:
    if no_llm and tool_router == "llm":
        raise click.UsageError(
            "--tool-router llm requires an LLM; remove --no-llm or choose "
            "--tool-router deterministic/cascade."
        )


def _launch_ink_ui(
    *,
    mode: str | None,
    profile: str | None,
    no_model: bool,
    vault: Path,
    session_persistence: str = "ram_only",
    session_id: str = "default",
    resume_session: bool = False,
) -> int:
    """Spawn the Ink UI (ui/dist), which owns the terminal and spawns `soca engine`."""
    import shutil

    node = shutil.which("node")
    repo_root = Path(__file__).resolve().parents[1]
    bundle = repo_root / "ui" / "dist" / "index.js"
    if node is None or not bundle.exists():
        reason = "Node.js chưa cài" if node is None else "ui/dist chưa build"
        raise click.ClickException(
            f"Ink UI chưa sẵn sàng ({reason}). Build bằng: cd ui && npm install && "
            "npm run build"
        )
    args = [node, str(bundle)]
    if mode:
        args.append(mode)
    if profile:
        args.append(profile)
    if no_model:
        args.append("--no-model")
    args.extend(["--session-persistence", session_persistence, "--session-id", session_id])
    if resume_session:
        args.append("--resume-session")
    args.extend(["--vault", str(vault.expanduser().resolve())])
    return subprocess.run(args, cwd=repo_root, check=False).returncode


@main.command()
@click.argument(
    "quick_profile",
    required=False,
    type=click.Choice(sorted(VOICE_RUNTIME_PROFILES)),
)
@click.option(
    "--llm-model",
    default=None,
    type=click.Choice(sorted(LLM_MODEL_REGISTRY)),
    help="Override the selected profile LLM.",
)
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=default_vault_root(),
    show_default=True,
    help="Knowledge vault root containing wiki/ and memory/core.json or archive notes.",
)
@click.option("--no-memory", is_flag=True, help="Disable core/archive/session memory.")
@click.option(
    "--no-model",
    is_flag=True,
    help="Do not load model runtimes (protocol smoke tests).",
)
@click.option(
    "--session-persistence",
    type=click.Choice(["ram_only", "local_resumable"]),
    default="ram_only",
    show_default=True,
    help="Working-memory persistence; local_resumable is opt-in.",
)
@click.option("--session-id", default="default", show_default=True)
@click.option(
    "--resume-session", is_flag=True, help="Resume the selected local session checkpoint."
)
@click.pass_context
def engine(
    ctx: click.Context,
    quick_profile: str | None,
    llm_model: str | None,
    vault: Path,
    no_memory: bool,
    no_model: bool,
    session_persistence: str,
    session_id: str,
    resume_session: bool,
) -> None:
    """Run the headless NDJSON engine (stdio protocol for external UIs).

    The Ink TUI spawns this process: commands in on stdin, events out on
    stdout, audio stays in-process.
    """
    from soca.app.engine import run_engine

    profile = quick_profile or DEFAULT_VOICE_RUNTIME_PROFILE_KEY
    try:
        voice_config = resolve_voice_runtime_config(
            profile_key=profile,
            llm_model=llm_model,
            vault=vault,
            no_memory=no_memory,
            session_persistence=cast(SessionPersistence, session_persistence),
            session_id=session_id,
            session_resume=resume_session,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    text_config = build_text_runtime_config(
        profile=profile,
        llm_model=llm_model,
        vault=vault,
        no_memory=no_memory,
        no_llm=no_model,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=0.2,
        top_p=0.95,
        knowledge_limit=3,
        memory_context_chars=64_000,
        memory_item_chars=900,
        session_chars=60_000,
        session_turns=6,
        turn_chars=500,
        session_persistence=session_persistence,
        session_id=session_id,
        session_resume=resume_session,
    )
    ctx.exit(
        run_engine(
            voice_config=voice_config,
            text_config=text_config,
            profile=profile,
            no_model=no_model,
        )
    )


@main.command(epilog=("\b\nQuick examples:\n  uv run soca voice\n  uv run soca voice baseline"))
@click.argument(
    "quick_profile",
    required=False,
    type=click.Choice(sorted(VOICE_RUNTIME_PROFILES)),
)
@click.option(
    "--profile",
    "profile_option",
    default=None,
    type=click.Choice(sorted(VOICE_RUNTIME_PROFILES)),
    hidden=True,
    help="Voice runtime profile to use before ASR/LLM/voice overrides.",
)
@click.option(
    "--asr-model",
    default=None,
    type=click.Choice(sorted(ASR_MODEL_REGISTRY)),
    hidden=True,
    help="Override the ASR registry key from the selected profile.",
)
@click.option(
    "--llm-model",
    default=None,
    type=click.Choice(sorted(LLM_MODEL_REGISTRY)),
    hidden=True,
    help="Override the LLM registry key from the selected profile.",
)
@click.option(
    "--voice",
    default=None,
    hidden=True,
    help="Override the Valtec voice/speaker id.",
)
@click.option("--endpoint-silence-ms", type=int, default=None, hidden=True)
@click.option("--max-record-ms", type=int, default=None, hidden=True)
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=default_vault_root(),
    show_default=True,
    help="Knowledge vault root containing wiki/ and memory/core.json or archive notes.",
)
@click.option("--no-memory", is_flag=True, help="Disable core/archive/session memory.")
@click.option("--memory-context-chars", type=int, default=64_000, hidden=True)
@click.option("--memory-item-chars", type=int, default=900, hidden=True)
@click.option("--session-chars", type=int, default=60_000, hidden=True)
@click.option("--session-turns", type=int, default=6, hidden=True)
@click.option("--turn-chars", type=int, default=500, hidden=True)
@click.option(
    "--session-persistence",
    type=click.Choice(["ram_only", "local_resumable"]),
    default="ram_only",
    hidden=True,
)
@click.option("--session-id", default="default", hidden=True)
@click.option("--resume-session", is_flag=True, hidden=True)
@click.option(
    "--tool-router",
    type=click.Choice(["deterministic", "llm", "cascade"]),
    default="cascade",
    hidden=True,
)
@click.option(
    "--router-response",
    type=click.Choice(["prompt_json", "json_schema"]),
    default="json_schema",
    hidden=True,
)
@click.option("--semantic-router/--no-semantic-router", default=False, hidden=True)
@click.option("--semantic-router-threshold", type=float, default=0.58, hidden=True)
@click.option("--semantic-router-margin", type=float, default=0.0, hidden=True)
@click.option(
    "--semantic-router-examples", type=click.Path(path_type=Path), default=None, hidden=True
)
@click.option("--memory-limit", type=int, default=3, hidden=True)
@click.option(
    "--memory-retrieval",
    type=click.Choice(["chunk_sparse", "hybrid"]),
    default="chunk_sparse",
    hidden=True,
)
@click.option(
    "--memory-dense-backend", type=click.Choice(["aiteamvn_v2"]), default="aiteamvn_v2", hidden=True
)
@click.option("--max-tokens", type=int, default=None, hidden=True)
@click.option("--temperature", type=float, default=None, hidden=True)
@click.option("--top-p", type=float, default=None, hidden=True)
@click.option(
    "--no-speak-repairs",
    is_flag=True,
    help="Do not speak the conversation-repair follow-up when ASR rejects a turn.",
)
@click.option(
    "--no-speak-rejections",
    is_flag=True,
    hidden=True,
    help="Deprecated alias of --no-speak-repairs.",
)
@click.option(
    "--press-enter-to-record",
    is_flag=True,
    help="Wait for ENTER before each recorded turn. Useful for debugging.",
)
@click.option(
    "--no-warmup",
    is_flag=True,
    help="Skip ASR/LLM/TTS first-call warmup before listening.",
)
@click.option(
    "--usage",
    is_flag=True,
    help="Show ASR/LLM/TTS latency + token usage after each turn.",
)
@click.option(
    "--barge-in/--no-barge-in",
    "barge_in",
    default=True,
    show_default=True,
    help=(
        "Interrupt playback when you start speaking. Use --no-barge-in on "
        "speakers without echo cancellation, otherwise SoCa hears itself."
    ),
)
@click.pass_context
def voice(
    ctx: click.Context,
    quick_profile: str | None,
    profile_option: str | None,
    asr_model: str | None,
    llm_model: str | None,
    voice: str | None,
    endpoint_silence_ms: int | None,
    max_record_ms: int | None,
    vault: Path,
    no_memory: bool,
    memory_context_chars: int,
    memory_item_chars: int,
    session_chars: int,
    session_turns: int,
    turn_chars: int,
    session_persistence: str,
    session_id: str,
    resume_session: bool,
    tool_router: str,
    router_response: str,
    semantic_router: bool,
    semantic_router_threshold: float,
    semantic_router_margin: float,
    semantic_router_examples: Path | None,
    memory_limit: int,
    memory_retrieval: str,
    memory_dense_backend: str,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    no_speak_repairs: bool,
    no_speak_rejections: bool,
    press_enter_to_record: bool,
    no_warmup: bool,
    usage: bool,
    barge_in: bool,
) -> None:
    """Run the interactive SoCa microphone voice loop.

    Quick form: soca voice [profile]
    """
    profile = quick_profile or profile_option or DEFAULT_VOICE_RUNTIME_PROFILE_KEY

    try:
        config = resolve_voice_runtime_config(
            profile_key=profile,
            asr_model=asr_model,
            llm_model=llm_model,
            tts_voice=voice,
            endpoint_silence_ms=endpoint_silence_ms,
            max_record_ms=max_record_ms,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            vault=vault,
            no_memory=no_memory,
            memory_context_chars=memory_context_chars,
            memory_item_chars=memory_item_chars,
            session_chars=session_chars,
            session_turns=session_turns,
            turn_chars=turn_chars,
            session_persistence=cast(SessionPersistence, session_persistence),
            session_id=session_id,
            session_resume=resume_session,
            tool_router_mode=tool_router,
            tool_router_response_mode=router_response,
            semantic_router_enabled=semantic_router,
            semantic_router_threshold=semantic_router_threshold,
            semantic_router_margin=semantic_router_margin,
            semantic_router_examples=semantic_router_examples,
            memory_limit=memory_limit,
            memory_retrieval_mode=memory_retrieval,
            memory_dense_backend=memory_dense_backend,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    player = None
    if barge_in:
        from soca.core.duplex_aec_sink import DuplexAecSink

        player = DuplexAecSink()
    ctx.exit(
        run_voice_loop(
            config,
            no_speak_repairs=no_speak_repairs,
            no_speak_rejections=no_speak_rejections,
            press_enter_to_record=press_enter_to_record,
            warmup=not no_warmup,
            show_usage=usage,
            player=player,
        )
    )


@main.command("asr-smoke")
@click.option(
    "--model",
    default=DEFAULT_ASR_MODEL_KEY,
    type=click.Choice(sorted(ASR_MODEL_REGISTRY)),
    show_default=True,
)
def asr_smoke(model: str) -> None:
    """Run the local ASR smoke test on recorded sample audio."""
    run_script(str(REPO_ROOT / "scripts" / "smoke_test_asr.py"), "--model", model)


@main.command("asr-models")
def asr_models() -> None:
    """List registered PhoWhisper ONNX candidates."""
    table = Table(title="SoCa ASR Registry")
    table.add_column("Key", style=st(ALT) or "none")
    table.add_column("Role")
    table.add_column("Params", justify="right")
    table.add_column("Exists", justify="center")
    table.add_column("Path")

    for config in ASR_MODEL_REGISTRY.values():
        table.add_row(
            config.model_key,
            config.role,
            f"{config.params_m}M",
            "yes" if config.local_dir.exists() else "no",
            str(config.local_dir),
        )

    console.print(table)


@main.command("llm-smoke")
@click.option(
    "--model",
    default=DEFAULT_LLM_MODEL_KEY,
    type=click.Choice(sorted(LLM_MODEL_REGISTRY)),
    show_default=True,
)
def llm_smoke(model: str) -> None:
    """Run the local llama.cpp LLM smoke test."""
    run_script(str(REPO_ROOT / "scripts" / "smoke_test_llm.py"), "--model", model)


@main.command("llm-models")
def llm_models() -> None:
    """List registered local LLM candidates."""
    table = Table(title="SoCa LLM Registry")
    table.add_column("Key", style=st(ALT) or "none")
    table.add_column("Role")
    table.add_column("Prompt")
    table.add_column("Exists", justify="center")
    table.add_column("Path")

    for config in LLM_MODEL_REGISTRY.values():
        table.add_row(
            config.model_key,
            config.role,
            config.prompt_style,
            "yes" if config.local_path.exists() else "no",
            str(config.local_path),
        )

    console.print(table)


@main.command("benchmark-asr")
@click.option("--n-speech", default=50, type=int, show_default=True)
@click.option("--n-noise", default=20, type=int, show_default=True)
@click.option(
    "--providers",
    default="auto",
    type=click.Choice(["auto", "cpu"]),
    show_default=True,
)
def benchmark_asr(n_speech: int, n_noise: int, providers: str) -> None:
    """Run the Table VII-style ASR robustness benchmark."""
    run_module(
        "local.eval_table7",
        "--n-speech",
        str(n_speech),
        "--n-noise",
        str(n_noise),
        "--providers",
        providers,
    )


@main.command("calibrate-asr")
@click.option("--n-speech", default=200, type=int, show_default=True)
@click.option("--n-noise", default=50, type=int, show_default=True)
@click.option(
    "--providers",
    default="auto",
    type=click.Choice(["auto", "cpu"]),
    show_default=True,
)
def calibrate_asr(n_speech: int, n_noise: int, providers: str) -> None:
    """Calibrate ASR confidence thresholds."""
    run_module(
        "local.calibrate_asr_confidence",
        "--n-speech",
        str(n_speech),
        "--n-noise",
        str(n_noise),
        "--providers",
        providers,
    )
