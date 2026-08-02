from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
import socket
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from soca.asr.qwen_artifacts import get_qwen_artifact
from soca.asr.qwen_backend import QwenASRBackend
from soca.asr.qwen_ipc_protocol import (
    SAMPLE_RATE,
    QwenIPCProtocolError,
    recv_audio_payload,
    recv_header,
    send_frame,
)
from soca.asr.qwen_service_identity import (
    QWEN_SERVICE_PROTOCOL_VERSION,
    QwenLaunchMode,
    QwenServiceIdentity,
    QwenServiceLaunch,
    QwenServiceState,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_CONNECTION_TIMEOUT_S = 60.0
SERVER_BACKLOG = 8
PACKAGE_DISTRIBUTIONS = ("qwen-asr", "soca", "torch", "transformers")


class QwenBackend(Protocol):
    supports_avg_logprob: bool

    def runtime_metadata(self, max_new_tokens: int = 128) -> dict[str, Any]: ...

    def transcribe(
        self,
        audio: Any,
        max_new_tokens: int = 128,
        *,
        context: str | None = None,
    ) -> Any: ...


class ServiceLifecycle:
    def __init__(
        self,
        launch: QwenServiceLaunch,
        *,
        started_at: float,
        packages: Mapping[str, str] | None = None,
    ) -> None:
        self._launch = launch
        self._started_at = started_at
        self._packages = dict(packages or _package_versions())
        self._lock = threading.Lock()
        self._in_flight = 0
        self._state = QwenServiceState.READY
        self._last_failure_type: str | None = None

    def begin_inference(self) -> None:
        with self._lock:
            self._in_flight += 1

    def finish_inference(self, failure: Exception | None = None) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            if failure is not None:
                self._last_failure_type = type(failure).__name__
                self._state = QwenServiceState.FAILED

    def begin_shutdown(self) -> None:
        with self._lock:
            self._state = QwenServiceState.STOPPING

    def record_failure(self, failure: Exception) -> None:
        with self._lock:
            self._last_failure_type = type(failure).__name__
            self._state = QwenServiceState.FAILED

    def identity(self, *, supports_avg_logprob: bool) -> QwenServiceIdentity:
        with self._lock:
            state = self._state
            if state is QwenServiceState.READY and self._in_flight:
                state = QwenServiceState.BUSY
            spec = self._launch.spec
            return QwenServiceIdentity(
                protocol_version=QWEN_SERVICE_PROTOCOL_VERSION,
                state=state,
                launch_mode=self._launch.mode,
                artifact_key=spec.key,
                artifact_role=spec.role.value,
                upstream_revision=spec.upstream.revision,
                mirror_revision=(spec.mirror.revision if spec.mirror is not None else None),
                artifact_digest=spec.digest,
                runtime_lock_digest=spec.runtime_lock_digest or "",
                context_policy_digest=spec.context_policy_digest,
                backend="qwen3_asr",
                device=spec.device,
                dtype=spec.dtype,
                package_versions=self._packages,
                pid=os.getpid(),
                uptime_ms=(time.monotonic() - self._started_at) * 1_000,
                in_flight=self._in_flight,
                supports_avg_logprob=supports_avg_logprob,
                last_failure_type=self._last_failure_type,
                no_fallback_attempted=True,
            )


def _package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PACKAGE_DISTRIBUTIONS}


def _error_response(
    *,
    error_type: str,
    error_message: str,
    request_id: object = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "ok": False,
        "error_type": error_type,
        "error_message": error_message,
    }
    if isinstance(request_id, str):
        response["request_id"] = request_id
    return response


def _validate_transcribe_header(header: Mapping[str, Any]) -> tuple[str, int, str | None]:
    request_id = header.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise QwenIPCProtocolError("request_id must be a non-empty string")
    sample_rate = header.get("sample_rate")
    if sample_rate != SAMPLE_RATE:
        raise QwenIPCProtocolError(f"sample_rate must be {SAMPLE_RATE}")
    max_new_tokens = _validate_max_new_tokens(header.get("max_new_tokens", 128))
    context = header.get("context")
    if context is not None and not isinstance(context, str):
        raise QwenIPCProtocolError("context must be a string or null")
    return request_id, max_new_tokens, context


def _validate_max_new_tokens(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QwenIPCProtocolError("max_new_tokens must be an integer")
    if not 1 <= value <= 4_096:
        raise QwenIPCProtocolError("max_new_tokens must be from 1 to 4096")
    return value


def _handle_connection(
    conn: socket.socket,
    backend: QwenBackend,
    transcribe_lock: threading.Lock,
    lifecycle: ServiceLifecycle,
    shutdown_event: threading.Event | None = None,
) -> None:
    while True:
        try:
            header = recv_header(conn)
        except (ConnectionError, OSError, QwenIPCProtocolError) as exc:
            LOGGER.debug("Qwen client disconnected or sent an invalid frame: %s", exc)
            return

        op = header.get("op")
        if op == "ping":
            send_frame(
                conn,
                {
                    "ok": True,
                    "identity": lifecycle.identity(
                        supports_avg_logprob=backend.supports_avg_logprob
                    ).to_wire(),
                },
            )
            continue

        if op == "runtime_metadata":
            try:
                max_new_tokens = _validate_max_new_tokens(header.get("max_new_tokens", 128))
                metadata = backend.runtime_metadata(max_new_tokens=max_new_tokens)
                send_frame(conn, {"ok": True, "metadata": metadata})
            except QwenIPCProtocolError as exc:
                send_frame(
                    conn,
                    _error_response(
                        error_type="ProtocolError",
                        error_message=str(exc),
                    ),
                )
            except Exception as exc:
                lifecycle.record_failure(exc)
                LOGGER.exception("runtime metadata failed")
                send_frame(
                    conn,
                    _error_response(
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    ),
                )
            continue

        if op == "shutdown" and shutdown_event is not None:
            with transcribe_lock:
                lifecycle.begin_shutdown()
                send_frame(conn, {"ok": True})
                shutdown_event.set()
            return

        if op != "transcribe":
            send_frame(
                conn,
                _error_response(
                    error_type="UnknownOp",
                    error_message=f"Unsupported operation: {op!r}",
                ),
            )
            continue

        request_id = header.get("request_id")
        try:
            audio = recv_audio_payload(conn, header)
        except QwenIPCProtocolError as exc:
            send_frame(
                conn,
                _error_response(
                    error_type="ProtocolError",
                    error_message=str(exc),
                    request_id=request_id,
                ),
            )
            return

        try:
            request_id, max_new_tokens, context = _validate_transcribe_header(header)
            lifecycle.begin_inference()
            with transcribe_lock:
                result = backend.transcribe(
                    audio,
                    max_new_tokens,
                    context=context,
                )
            send_frame(
                conn,
                {
                    "request_id": request_id,
                    "ok": True,
                    "text": result.text,
                    "latency_ms": result.latency_ms,
                    "audio_duration_ms": result.audio_duration_ms,
                    "rtf": result.rtf,
                    "avg_logprob": result.avg_logprob,
                    "avg_logprob_reliable": result.avg_logprob_reliable,
                    "alternatives": list(result.alternatives),
                    "generated_token_count": result.generated_token_count,
                    "hit_max_new_tokens": result.hit_max_new_tokens,
                },
            )
            lifecycle.finish_inference()
        except QwenIPCProtocolError as exc:
            send_frame(
                conn,
                _error_response(
                    error_type="ProtocolError",
                    error_message=str(exc),
                    request_id=request_id,
                ),
            )
        except Exception as exc:
            lifecycle.finish_inference(exc)
            LOGGER.exception("transcribe failed")
            send_frame(
                conn,
                _error_response(
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    request_id=request_id,
                ),
            )


def main() -> None:
    started_at = time.monotonic()
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--artifact-key", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--launch-mode",
        choices=tuple(mode.value for mode in QwenLaunchMode),
        required=True,
    )
    parser.add_argument("--language", default="Vietnamese")
    parser.add_argument(
        "--connection-timeout",
        type=float,
        default=DEFAULT_CONNECTION_TIMEOUT_S,
    )
    args = parser.parse_args()
    if args.connection_timeout <= 0:
        parser.error("--connection-timeout must be positive")

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    spec = get_qwen_artifact(args.artifact_key)
    launch = QwenServiceLaunch(
        spec=spec,
        model_path=args.model_path,
        mode=QwenLaunchMode(args.launch_mode),
    )
    backend = QwenASRBackend(
        model_path=launch.model_path,
        context="",
        language=args.language,
        device=spec.device,
        dtype=spec.dtype,
        require_logprob=True,
    )

    socket_path = Path(args.socket_path)
    ready_path = socket_path.with_suffix(".ready")
    socket_path.unlink(missing_ok=True)
    ready_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    transcribe_lock = threading.Lock()
    shutdown_event = threading.Event()
    lifecycle = ServiceLifecycle(launch, started_at=started_at)
    try:
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        server.listen(SERVER_BACKLOG)
        server.settimeout(0.25)
        ready_path.touch(mode=0o600)
        while not shutdown_event.is_set():
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            conn.settimeout(args.connection_timeout)
            threading.Thread(
                target=_run_connection,
                args=(conn, backend, transcribe_lock, lifecycle, shutdown_event),
                daemon=True,
                name="soca-qwen-asr-client",
            ).start()
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)
        ready_path.unlink(missing_ok=True)


def _run_connection(
    conn: socket.socket,
    backend: QwenBackend,
    transcribe_lock: threading.Lock,
    lifecycle: ServiceLifecycle,
    shutdown_event: threading.Event,
) -> None:
    with conn:
        _handle_connection(conn, backend, transcribe_lock, lifecycle, shutdown_event)


if __name__ == "__main__":
    main()
