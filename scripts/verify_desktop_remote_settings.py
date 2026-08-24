"""Verify remote LLM settings readiness on a frozen desktop sidecar.

This smoke proves the packaged remote contract without real credentials:

- remote readiness never consults local GGUF files (backend independence);
- a missing key yields the typed "no API key" readiness reason;
- an invalid key triggers a real provider catalog request that terminates in a
  typed provider failure instead of a silent fallback or a fake catalog;
- switching remote -> local resets the model id to the local default;
- local readiness reports the exact missing file under ``SOCA_MODEL_ROOT``.

A provider is chosen only when no key is resolvable through the real
SecretStore chain (keyring, env, JSON fallback), so the smoke can never spend
a stored credential. Successful catalog/chat traffic still requires the
release-owner real-flow matrix. The live rejection here proves route and typed
failure handling only: OpenAI-compatible endpoints answer an invalid key with
401/403 (typed auth error), while Google's compatibility shim answers with
HTTP 400 (typed catalog error), and both outcomes are recorded as evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from soca.config.llm_settings import DEFAULT_LOCAL_MODEL_ID
from soca.config.secret_store import SecretStore
from soca.llm.providers import PROVIDER_REGISTRY, get_provider

REMOTE_SMOKE_MODEL = "gpt-4o-mini"
COLD_START_TIMEOUT_S = 240.0
FRAME_TIMEOUT_S = 60.0
CATALOG_TIMEOUT_S = 120.0


class VerificationError(RuntimeError):
    pass


def _pick_unused_provider() -> str:
    """Choose a provider with no key anywhere in the real resolution chain."""
    store = SecretStore(env={})
    for key in sorted(PROVIDER_REGISTRY):
        env_name = PROVIDER_REGISTRY[key].api_key_env
        if store.get_key(key) is None and not os.environ.get(env_name, ""):
            return key
    raise VerificationError(
        "every provider has a resolvable key on this host; "
        "refusing to run a smoke that could spend a real credential"
    )


class SidecarSession:
    """Interactive NDJSON session against the frozen sidecar.

    stderr is discarded: an unread stderr pipe can fill and block the child,
    which would keep the process alive past every graceful timeout.
    """

    def __init__(self, sidecar: Path, *, environment: dict[str, str], cwd: Path) -> None:
        self._process = subprocess.Popen(
            [str(sidecar), "engine", "--session-persistence", "ram_only"],
            cwd=cwd,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.frames: list[dict[str, Any]] = []

    def send(self, command: dict[str, Any]) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
        self._process.stdin.flush()

    def _readline_deadline(self, timeout_s: float) -> str | None:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([self._process.stdout], [], [], remaining)
            if not ready:
                return None
            assert self._process.stdout is not None
            line = self._process.stdout.readline()
            if not line:  # EOF
                return None
            stripped = line.strip()
            if stripped:
                return stripped

    def wait_for_hello(self, *, timeout_s: float = COLD_START_TIMEOUT_S) -> None:
        """Block until the frozen runtime finishes its cold start.

        Startup frames emitted before any command (including the default local
        llm_config) are collected here so later assertions cannot match them.
        """
        while True:
            line = self._readline_deadline(timeout_s)
            if line is None:
                raise VerificationError(
                    f"no protocol hello within {timeout_s}s; frames so far: {self.frames}"
                )
            frame = json.loads(line)
            self.frames.append(frame)
            if frame.get("event") == "hello":
                return

    def drain_startup_frames(self, *, quiet_s: float = 1.5) -> None:
        """Collect remaining startup frames until the pipe stays quiet."""
        while True:
            line = self._readline_deadline(quiet_s)
            if line is None:
                return
            self.frames.append(json.loads(line))

    def read_until(
        self, predicate: Callable[[dict[str, Any]], bool], *, timeout_s: float
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            line = self._readline_deadline(remaining)
            if line is None:
                return None
            frame = json.loads(line)
            self.frames.append(frame)
            if predicate(frame):
                return frame

    def close(self) -> int:
        self.send({"cmd": "quit"})
        assert self._process.stdin is not None
        self._process.stdin.close()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            line = self._readline_deadline(5.0)
            if line is None:
                break
            self.frames.append(json.loads(line))
        try:
            return self._process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                return self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)
                raise VerificationError(
                    f"sidecar ignored SIGTERM after quit; frames: {self.frames}"
                ) from None
        finally:
            self._process.stdout.close()


def _isolated_environment(root: Path, *, extra: dict[str, str] | None = None) -> dict[str, str]:
    app_data = root / "app-data"
    environment = {
        **os.environ,
        "XDG_CONFIG_HOME": str(app_data / "config"),
        "XDG_DATA_HOME": str(app_data / "data"),
        "XDG_STATE_HOME": str(app_data / "state"),
        "SOCA_VAULT": str(app_data / "vault"),
        "SOCA_MODEL_ROOT": str(app_data / "models"),
    }
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.pop("VIRTUAL_ENV", None)
    if extra:
        environment.update(extra)
    return environment


def _llm_config(
    session: SidecarSession,
    command: dict[str, Any],
    *,
    expect: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    session.send(command)
    frame = session.read_until(
        lambda item: item.get("event") == "llm_config" and expect(item),
        timeout_s=FRAME_TIMEOUT_S,
    )
    if frame is None:
        raise VerificationError(f"no matching llm_config frame after {command}: {session.frames}")
    return frame


def _start_session(sidecar: Path, root: Path) -> SidecarSession:
    outside = root / "outside-checkout"
    outside.mkdir(parents=True, exist_ok=True)
    session = SidecarSession(sidecar, environment=_isolated_environment(root), cwd=outside)
    session.wait_for_hello()
    session.drain_startup_frames()
    return session


def verify_missing_key_flow(sidecar: Path, provider_key: str, root: Path) -> None:
    provider = get_provider(provider_key)
    session = _start_session(sidecar, root / "missing-key")
    try:
        # Selecting the remote backend without any stored key must be rejected
        # with a typed error instead of committing hosted settings.
        session.send(
            {
                "cmd": "llm_select",
                "backend": "remote",
                "provider": provider_key,
                "model": REMOTE_SMOKE_MODEL,
            }
        )
        rejection = session.read_until(
            lambda item: item.get("event") == "engine_error", timeout_s=FRAME_TIMEOUT_S
        )
        if (
            rejection is None
            or rejection.get("message") != f"Chưa có API key cho {provider.label}."
        ):
            raise VerificationError(f"expected typed no-key rejection, got: {session.frames}")

        # The rejected selection must leave the effective config on the local
        # backend, reporting the exact missing GGUF under SOCA_MODEL_ROOT.
        local = _llm_config(
            session, {"cmd": "llm_config"}, expect=lambda item: item["backend"] == "local"
        )
        if local["model"] != DEFAULT_LOCAL_MODEL_ID:
            raise VerificationError(f"rejected remote select leaked the model id: {local}")
        if local["runtime_ready"] is not False or "Chưa tìm thấy model local tại" not in (
            local["runtime_reason"] or ""
        ):
            raise VerificationError(f"missing local model must be reported truthfully: {local}")
        expected_root = str((root / "missing-key" / "app-data" / "models").resolve())
        if not str(Path(local.get("local_model_path") or "").resolve()).startswith(expected_root):
            raise VerificationError(f"local readiness must point inside SOCA_MODEL_ROOT: {local}")
    finally:
        code = session.close()
    if code != 0:
        raise VerificationError(f"missing-key session exited {code}")


def verify_invalid_key_flow(sidecar: Path, provider_key: str, root: Path) -> str:
    provider = get_provider(provider_key)
    environment = _isolated_environment(
        root / "invalid-key", extra={provider.api_key_env: "sk-soca-smoke-invalid-key"}
    )
    outside = root / "outside-checkout"
    outside.mkdir(parents=True, exist_ok=True)
    session = SidecarSession(sidecar, environment=environment, cwd=outside)
    session.wait_for_hello()
    session.drain_startup_frames()
    try:
        pending = _llm_config(
            session,
            {
                "cmd": "llm_select",
                "backend": "remote",
                "provider": provider_key,
                "model": REMOTE_SMOKE_MODEL,
            },
            expect=lambda item: item["backend"] == "remote",
        )
        if pending["model"] != REMOTE_SMOKE_MODEL or pending["provider"] != provider_key:
            raise VerificationError(f"remote selection not reflected: {pending}")
        if pending["runtime_ready"] is not False:
            raise VerificationError(f"catalog fetch must gate readiness: {pending}")
        if pending["runtime_reason"] != f"Đang tải danh mục model của {provider.label}…":
            raise VerificationError(f"unexpected catalog-pending reason: {pending}")
        if pending.get("local_model_path") is not None:
            raise VerificationError(f"remote readiness consulted a local model path: {pending}")

        failure = session.read_until(
            lambda item: (
                item.get("event") == "engine_error" and item.get("provider") == provider_key
            ),
            timeout_s=CATALOG_TIMEOUT_S,
        )
        if failure is None:
            raise VerificationError(f"no typed provider error for {provider_key}: {session.frames}")
        # The typed terminal failure depends on the live provider: 401/403 maps
        # to the auth message, while e.g. Google's shim answers an invalid key
        # with HTTP 400, which maps to the typed catalog error. Any other
        # message would be an untyped leak.
        message = failure.get("message")
        if message == "API key sai hoặc hết hạn.":
            outcome = "HTTP 401/403 -> typed auth error"
        elif (
            isinstance(message, str)
            and message.startswith("Lỗi khi lấy danh sách model (HTTP ")
            and message.endswith(").")
        ):
            status = message.removeprefix("Lỗi khi lấy danh sách model (HTTP ").removesuffix(").")
            outcome = f"HTTP {status} -> typed catalog error"
        else:
            raise VerificationError(f"expected a typed provider failure, got: {failure}")
        if any(frame.get("event") == "llm_catalog" for frame in session.frames):
            raise VerificationError("an invalid key must not produce a catalog frame")

        # Switching back to local must reset the hosted model id to the local
        # default instead of carrying it across backends.
        local = _llm_config(
            session,
            {"cmd": "llm_select", "backend": "local"},
            expect=lambda item: item["backend"] == "local",
        )
        if local["model"] != DEFAULT_LOCAL_MODEL_ID:
            raise VerificationError(f"remote model id leaked into local backend: {local['model']}")
    finally:
        code = session.close()
    if code != 0:
        raise VerificationError(f"invalid-key session exited {code}")
    return f"{provider.base_url}/models -> {outcome}"


def verify(sidecar: Path) -> None:
    if not sidecar.is_file():
        raise VerificationError(f"sidecar does not exist: {sidecar}")
    provider_key = _pick_unused_provider()
    with tempfile.TemporaryDirectory(prefix="soca-frozen-remote-") as temporary:
        root = Path(temporary)
        verify_missing_key_flow(sidecar, provider_key, root)
        route = verify_invalid_key_flow(sidecar, provider_key, root)
    print("frozen remote settings flow passed")
    print(f"provider: {provider_key}")
    print(f"live catalog route evidence: {route}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", required=True, type=Path)
    arguments = parser.parse_args()
    verify(arguments.sidecar.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
