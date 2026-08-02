from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from soca.asr.qwen_backend import QwenASRBackend
from soca.asr.qwen_ipc_protocol import (
    SAMPLE_RATE,
    QwenIPCProtocolError,
    recv_audio_payload,
    recv_header,
    send_frame,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_CONNECTION_TIMEOUT_S = 60.0
SERVER_BACKLOG = 8


class QwenBackend(Protocol):
    model_key: str
    supports_avg_logprob: bool
    context: str
    language: str

    def runtime_metadata(self, max_new_tokens: int = 128) -> dict[str, Any]: ...

    def transcribe(
        self,
        audio: Any,
        max_new_tokens: int = 128,
        *,
        context: str | None = None,
    ) -> Any: ...


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
    max_new_tokens = header.get("max_new_tokens", 128)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise QwenIPCProtocolError("max_new_tokens must be an integer")
    if not 1 <= max_new_tokens <= 4_096:
        raise QwenIPCProtocolError("max_new_tokens must be from 1 to 4096")
    context = header.get("context")
    if context is not None and not isinstance(context, str):
        raise QwenIPCProtocolError("context must be a string or null")
    return request_id, max_new_tokens, context


def _handle_connection(
    conn: socket.socket,
    backend: QwenBackend,
    transcribe_lock: threading.Lock,
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
                    "model_key": backend.model_key,
                    "supports_avg_logprob": backend.supports_avg_logprob,
                    "context": backend.context,
                    "language": backend.language,
                },
            )
            continue

        if op == "runtime_metadata":
            try:
                max_new_tokens = int(header.get("max_new_tokens", 128))
                metadata = backend.runtime_metadata(max_new_tokens=max_new_tokens)
                send_frame(conn, {"ok": True, "metadata": metadata})
            except Exception as exc:
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
                },
            )
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--context", default="")
    parser.add_argument("--language", default="Vietnamese")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument(
        "--connection-timeout",
        type=float,
        default=DEFAULT_CONNECTION_TIMEOUT_S,
    )
    args = parser.parse_args()
    if args.connection_timeout <= 0:
        parser.error("--connection-timeout must be positive")

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    backend = QwenASRBackend(
        model_id=args.model_id,
        context=args.context,
        language=args.language,
        device=args.device,
        dtype=args.dtype,
        require_logprob=True,
    )

    socket_path = Path(args.socket_path)
    ready_path = socket_path.with_suffix(".ready")
    socket_path.unlink(missing_ok=True)
    ready_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    transcribe_lock = threading.Lock()
    shutdown_event = threading.Event()
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
                args=(conn, backend, transcribe_lock, shutdown_event),
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
    shutdown_event: threading.Event,
) -> None:
    with conn:
        _handle_connection(conn, backend, transcribe_lock, shutdown_event)


if __name__ == "__main__":
    main()
