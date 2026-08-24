"""Frozen entry point for the self-contained SoCa desktop engine.

The native shell needs protocol compatibility before Python imports the ASR,
TTS and indexing runtime. This module therefore emits one truthful staging
hello synchronously, then loads the normal engine and supplies its authoritative
context/settings frames. It intentionally implements only the ``engine``
subcommand used by Tauri; the full ``soca`` CLI remains the product CLI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROTOCOL_VERSION = 3
KEYRING_TIMEOUT_SECONDS = 0.75


class SidecarKeyringBackend:
    """Use a bounded helper process for OS Keychain access in the desktop app.

    The macOS Security bridge may keep Python's GIL while it waits for a stale
    Keychain prompt. A Python thread cannot enforce a timeout in that case, so
    the frozen executable invokes its small internal helper in a child process.
    The parent can then terminate and reap that helper at the deadline.
    """

    def __init__(self, executable: Path, *, timeout_seconds: float) -> None:
        self._executable = executable
        self._timeout_seconds = timeout_seconds
        self._unavailable = False

    def get_password(self, service: str, username: str) -> str | None:
        payload = self._request("get", service=service, username=username)
        value = payload.get("value")
        if value is None or isinstance(value, str):
            return value
        raise RuntimeError("Desktop keyring helper returned an invalid value")

    def set_password(self, service: str, username: str, value: str) -> None:
        self._request("set", service=service, username=username, value=value)

    def _request(
        self,
        operation: str,
        *,
        service: str,
        username: str,
        value: str | None = None,
    ) -> dict[str, object]:
        if self._unavailable:
            raise RuntimeError("Desktop keyring helper is unavailable")
        request = {"service": service, "username": username}
        if value is not None:
            request["value"] = value
        try:
            result = subprocess.run(
                [
                    str(self._executable),
                    "keyring",
                    "--keyring-operation",
                    operation,
                ],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self._unavailable = True
            raise RuntimeError("Desktop keyring helper timed out") from exc
        except OSError as exc:
            raise RuntimeError(f"Desktop keyring helper could not start: {exc}") from exc
        if result.returncode != 0:
            raise RuntimeError("Desktop keyring helper failed")
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Desktop keyring helper returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise RuntimeError("Desktop keyring helper returned an invalid payload")
        return response


def emit_staging_hello(*, profile: str | None, no_model: bool) -> None:
    """Publish protocol compatibility before heavyweight runtime imports."""

    payload = {
        "event": "hello",
        "version": PROTOCOL_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "supported_versions": [PROTOCOL_VERSION],
        "profile": profile or "initializing",
        "no_model": no_model,
        "stack": {"llm": "initializing"},
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["engine", "keyring"])
    parser.add_argument("quick_profile", nargs="?")
    parser.add_argument("--llm-model")
    parser.add_argument("--vault", type=Path)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument(
        "--session-persistence",
        choices=["ram_only", "local_resumable"],
        default="ram_only",
    )
    parser.add_argument("--session-id", default="default")
    parser.add_argument("--resume-session", action="store_true")
    parser.add_argument("--keyring-operation", choices=["get", "set"])
    return parser.parse_args(argv)


def run_keyring_helper(operation: str | None) -> int:
    """Run one internal keyring operation without emitting engine protocol."""

    if operation not in {"get", "set"}:
        raise ValueError("keyring helper operation is required")
    try:
        request = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        raise ValueError("keyring helper request must be JSON") from exc
    if not isinstance(request, dict):
        raise ValueError("keyring helper request must be an object")

    from soca.config.secret_store import KEYRING_SERVICE, _load_keyring

    service = request.get("service")
    username = request.get("username")
    if service != KEYRING_SERVICE or not isinstance(username, str) or not username:
        raise ValueError("keyring helper request is invalid")
    keyring = _load_keyring()
    if keyring is None:
        raise RuntimeError("keyring is unavailable")
    if operation == "get":
        value = keyring.get_password(service, username)
        sys.stdout.write(json.dumps({"value": value}) + "\n")
        return 0

    value = request.get("value")
    if not isinstance(value, str) or not value:
        raise ValueError("keyring helper value is invalid")
    keyring.set_password(service, username, value)
    sys.stdout.write("{}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "keyring":
        return run_keyring_helper(args.keyring_operation)
    emit_staging_hello(profile=args.quick_profile, no_model=args.no_model)

    # All imports below this line are allowed to take time. The desktop has
    # already verified protocol compatibility and can render truthful loading
    # readiness instead of an unbounded blank startup screen.
    from soca.app.engine import run_engine
    from soca.app.text_runtime import resolve_text_runtime_config
    from soca.config import DEFAULT_MAX_TOKENS, SecretStore, load_voice_profile
    from soca.core import resolve_voice_runtime_config
    from soca.knowledge.vault import default_vault_root

    profile = args.quick_profile or load_voice_profile()
    vault = (args.vault or default_vault_root()).expanduser().resolve()
    voice_config = resolve_voice_runtime_config(
        profile_key=profile,
        llm_model=args.llm_model,
        vault=vault,
        no_memory=args.no_memory,
        session_persistence=args.session_persistence,
        session_id=args.session_id,
        session_resume=args.resume_session,
    )
    text_config = resolve_text_runtime_config(
        profile_key=profile,
        llm_model=args.llm_model,
        vault=vault,
        no_memory=args.no_memory,
        no_llm=args.no_model,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=0.2,
        top_p=0.95,
        knowledge_limit=3,
        memory_context_chars=64_000,
        memory_item_chars=900,
        session_chars=60_000,
        session_turns=6,
        turn_chars=500,
        session_persistence=args.session_persistence,
        session_id=args.session_id,
        session_resume=args.resume_session,
    )
    # Keychain is useful when available, but its security-service prompt can
    # block indefinitely while holding Python's GIL. Use a bounded child
    # process so the engine command loop always reaches a terminal readiness
    # state. The fallback remains the existing explicit env/0600 JSON path;
    # provider/model selection is never changed.
    secret_store = SecretStore(
        keyring_module=SidecarKeyringBackend(
            Path(sys.executable), timeout_seconds=KEYRING_TIMEOUT_SECONDS
        )
    )
    return run_engine(
        voice_config=voice_config,
        text_config=text_config,
        profile=profile,
        no_model=args.no_model,
        secret_store=secret_store,
        hello_emitted=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
