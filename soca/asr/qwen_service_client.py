from __future__ import annotations

import logging
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .qwen_ipc_protocol import (
    MAX_AUDIO_SAMPLES,
    SAMPLE_RATE,
    QwenIPCProtocolError,
    recv_header,
    send_frame,
)
from .whisper_onnx import ASRResult

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN_VENV_PYTHON = REPO_ROOT / ".venv-qwen" / "bin" / "python"
DEFAULT_STARTUP_TIMEOUT_S = 60.0
DEFAULT_REQUEST_TIMEOUT_S = 30.0
DEFAULT_SHUTDOWN_TIMEOUT_S = 5.0
MAX_UNIX_SOCKET_PATH_BYTES = 103


class QwenServiceUnavailable(RuntimeError):
    """The Qwen ASR subprocess is not running or not responding."""


class QwenServiceCrashed(RuntimeError):
    """The Qwen ASR subprocess or its IPC connection terminated unexpectedly."""


class QwenServiceTimeout(QwenServiceUnavailable):
    """The Qwen ASR service exceeded a bounded operation deadline."""


class QwenServiceProtocolError(RuntimeError):
    """The service returned a malformed or mismatched response."""


class QwenTranscribeError(RuntimeError):
    def __init__(self, remote_type: str, message: str, request_id: str) -> None:
        super().__init__(f"{remote_type}: {message}")
        self.remote_type = remote_type
        self.request_id = request_id


class QwenASRServiceClient:
    BACKEND = "qwen3_asr_service"
    DECODE_STRATEGY = "llm_decoder"

    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
        context: str = "",
        socket_dir: Path | None = None,
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        shutdown_timeout_s: float = DEFAULT_SHUTDOWN_TIMEOUT_S,
        python_executable: Path | None = None,
    ) -> None:
        for name, value in (
            ("startup_timeout_s", startup_timeout_s),
            ("request_timeout_s", request_timeout_s),
            ("shutdown_timeout_s", shutdown_timeout_s),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        executable = python_executable or QWEN_VENV_PYTHON
        if not executable.is_file():
            raise QwenServiceUnavailable(
                f"{executable} not found. Create a separate venv at .venv-qwen "
                "and install qwen-asr into it."
            )

        root = socket_dir or Path(tempfile.gettempdir())
        if not root.is_dir():
            raise QwenServiceUnavailable(f"Socket directory does not exist: {root}")
        self._socket_path = root / f"soca-qwen-asr-{uuid.uuid4().hex}.sock"
        if len(str(self._socket_path).encode()) > MAX_UNIX_SOCKET_PATH_BYTES:
            raise QwenServiceUnavailable(
                f"Unix socket path exceeds {MAX_UNIX_SOCKET_PATH_BYTES} bytes: "
                f"{self._socket_path}"
            )
        self._ready_path = self._socket_path.with_suffix(".ready")
        self._request_timeout_s = request_timeout_s
        self._shutdown_timeout_s = shutdown_timeout_s
        self._state_lock = threading.Lock()
        self._active_sockets: set[socket.socket] = set()
        self._closed = False
        self._process: subprocess.Popen[bytes] | None = None

        try:
            self._process = subprocess.Popen(
                [
                    str(executable),
                    "-m",
                    "soca.asr.qwen_service_server",
                    "--socket-path",
                    str(self._socket_path),
                    "--model-id",
                    model_id,
                    "--context",
                    context,
                    "--connection-timeout",
                    str(request_timeout_s),
                ],
                cwd=str(REPO_ROOT),
                stdin=subprocess.DEVNULL,
            )
            self._wait_for_ready(startup_timeout_s)
            handshake = self._request({"op": "ping"})
            self._require_success(handshake, operation="ping")
            self.model_key = self._required_string(handshake, "model_key")
            self.supports_avg_logprob = self._required_bool(
                handshake, "supports_avg_logprob"
            )
            self.context = self._required_string(handshake, "context")
            self.language = self._required_string(handshake, "language")
        except Exception:
            self.close()
            raise

    @property
    def process(self) -> subprocess.Popen[bytes] | None:
        return self._process

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    @property
    def ready_path(self) -> Path:
        return self._ready_path

    def _wait_for_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            process = self._require_process()
            if process.poll() is not None:
                raise QwenServiceCrashed(
                    f"qwen_service_server exited with code {process.returncode} "
                    "before becoming ready"
                )
            if self._ready_path.exists():
                return
            time.sleep(0.1)
        raise QwenServiceTimeout(f"Server not ready after {timeout_s}s")

    def _require_process(self) -> subprocess.Popen[bytes]:
        process = self._process
        if process is None:
            raise QwenServiceUnavailable("Qwen ASR service was not started")
        return process

    def _request(self, header: Mapping[str, Any], payload: bytes = b"") -> dict[str, Any]:
        process = self._require_process()
        if process.poll() is not None:
            raise QwenServiceCrashed(
                f"qwen_service_server exited with code {process.returncode}"
            )
        channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        channel.settimeout(self._request_timeout_s)
        with self._state_lock:
            if self._closed:
                channel.close()
                raise QwenServiceUnavailable("Qwen ASR service client is closed")
            self._active_sockets.add(channel)
        try:
            channel.connect(str(self._socket_path))
            send_frame(channel, header, payload)
            return recv_header(channel)
        except TimeoutError as exc:
            raise QwenServiceTimeout(
                f"Qwen ASR request exceeded {self._request_timeout_s}s"
            ) from exc
        except QwenIPCProtocolError as exc:
            raise QwenServiceProtocolError(str(exc)) from exc
        except (ConnectionError, OSError) as exc:
            raise QwenServiceCrashed(str(exc)) from exc
        finally:
            with self._state_lock:
                self._active_sockets.discard(channel)
            channel.close()

    def transcribe(
        self,
        audio: np.ndarray,
        max_new_tokens: int = 128,
        *,
        context: str | None = None,
    ) -> ASRResult:
        if audio.ndim != 1:
            raise ValueError("audio must be a mono one-dimensional array")
        if len(audio) > MAX_AUDIO_SAMPLES:
            raise ValueError(f"audio exceeds the {MAX_AUDIO_SAMPLES}-sample limit")
        normalized = np.asarray(audio, dtype="<f4", order="C")
        if not np.isfinite(normalized).all():
            raise ValueError("audio contains non-finite samples")
        request_id = uuid.uuid4().hex
        response = self._request(
            {
                "op": "transcribe",
                "request_id": request_id,
                "max_new_tokens": max_new_tokens,
                "context": context,
                "n_samples": len(normalized),
                "sample_rate": SAMPLE_RATE,
            },
            payload=normalized.tobytes(),
        )
        response_request_id = response.get("request_id")
        if response_request_id != request_id:
            raise QwenServiceProtocolError(
                f"Response request_id {response_request_id!r} does not match {request_id!r}"
            )
        if response.get("ok") is not True:
            raise QwenTranscribeError(
                self._required_string(response, "error_type"),
                self._required_string(response, "error_message"),
                request_id,
            )
        try:
            return ASRResult(
                text=self._required_string(response, "text"),
                latency_ms=float(response["latency_ms"]),
                audio_duration_ms=float(response["audio_duration_ms"]),
                rtf=float(response["rtf"]),
                avg_logprob=float(response["avg_logprob"]),
                avg_logprob_reliable=self._required_bool(
                    response, "avg_logprob_reliable"
                ),
                alternatives=tuple(response["alternatives"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise QwenServiceProtocolError(f"Malformed transcribe response: {exc}") from exc

    def runtime_metadata(self, max_new_tokens: int = 128) -> dict[str, Any]:
        response = self._request(
            {"op": "runtime_metadata", "max_new_tokens": max_new_tokens}
        )
        self._require_success(response, operation="runtime_metadata")
        metadata = response.get("metadata")
        if not isinstance(metadata, dict):
            raise QwenServiceProtocolError("runtime_metadata response is not an object")
        return metadata

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            active_sockets = tuple(self._active_sockets)
            self._active_sockets.clear()
        for channel in active_sockets:
            channel.close()

        process = self._process
        if process is not None and process.poll() is None:
            if not self._request_graceful_shutdown():
                LOGGER.warning("Qwen ASR graceful shutdown failed; terminating process")
                process.terminate()
            try:
                process.wait(timeout=self._shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                LOGGER.warning("Qwen ASR process did not exit; killing process")
                process.kill()
                process.wait(timeout=self._shutdown_timeout_s)
        self._socket_path.unlink(missing_ok=True)
        self._ready_path.unlink(missing_ok=True)

    def _request_graceful_shutdown(self) -> bool:
        channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        channel.settimeout(self._shutdown_timeout_s)
        try:
            channel.connect(str(self._socket_path))
            send_frame(channel, {"op": "shutdown"})
            response = recv_header(channel)
            return response.get("ok") is True
        except (ConnectionError, OSError, QwenIPCProtocolError):
            return False
        finally:
            channel.close()

    @staticmethod
    def _required_string(payload: Mapping[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str):
            raise QwenServiceProtocolError(f"{key} must be a string")
        return value

    @staticmethod
    def _required_bool(payload: Mapping[str, Any], key: str) -> bool:
        value = payload.get(key)
        if not isinstance(value, bool):
            raise QwenServiceProtocolError(f"{key} must be a boolean")
        return value

    @classmethod
    def _require_success(cls, payload: Mapping[str, Any], *, operation: str) -> None:
        if payload.get("ok") is True:
            return
        error_type = cls._required_string(payload, "error_type")
        error_message = cls._required_string(payload, "error_message")
        raise QwenServiceProtocolError(
            f"{operation} failed with {error_type}: {error_message}"
        )


__all__ = [
    "QWEN_VENV_PYTHON",
    "QwenASRServiceClient",
    "QwenServiceCrashed",
    "QwenServiceProtocolError",
    "QwenServiceTimeout",
    "QwenServiceUnavailable",
    "QwenTranscribeError",
]
