from __future__ import annotations

import logging
import os
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
from .qwen_runtime import default_qwen_runtime_root, default_qwen_venv_python
from .qwen_service_identity import (
    QwenServiceIdentity,
    QwenServiceIdentityError,
    QwenServiceLaunch,
    QwenServiceState,
)
from .result import ASRResult

LOGGER = logging.getLogger(__name__)
# Kept as a source-runtime compatibility export. Runtime resolution in the
# client remains dynamic so a desktop-selected worker is not frozen at import.
QWEN_VENV_PYTHON = default_qwen_venv_python()
DEFAULT_STARTUP_TIMEOUT_S = 60.0
DEFAULT_REQUEST_TIMEOUT_S = 30.0
DEFAULT_SHUTDOWN_TIMEOUT_S = 5.0
MAX_UNIX_SOCKET_PATH_BYTES = 103
SENSITIVE_MODEL_ENVIRONMENT = frozenset({"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"})
# The Qwen service owns a separately locked virtual environment.  A frozen
# desktop parent can expose its bundled packages through Python startup
# variables; inheriting those would mix incompatible transformer versions into
# the worker.  The worker must resolve imports only from its selected runtime.
ISOLATED_PYTHON_ENVIRONMENT = frozenset(
    {
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONUSERBASE",
        "PYTHONEXECUTABLE",
        "VIRTUAL_ENV",
        "__PYVENV_LAUNCHER__",
    }
)


class QwenServiceUnavailable(RuntimeError):
    """The Qwen ASR subprocess is not running or not responding."""


class QwenServiceCrashed(RuntimeError):
    """The Qwen ASR subprocess or its IPC connection terminated unexpectedly."""


class QwenServiceTimeout(QwenServiceUnavailable):
    """The Qwen ASR service exceeded a bounded operation deadline."""


class QwenServiceProtocolError(RuntimeError):
    """The service returned a malformed or mismatched response."""


class QwenServiceIdentityMismatch(QwenServiceProtocolError):
    """The live worker identity differs from the verified launch contract."""


class QwenServiceNotReady(QwenServiceUnavailable):
    """The worker answered but its lifecycle state cannot admit requests."""


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
        launch: QwenServiceLaunch,
        socket_dir: Path | None = None,
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        shutdown_timeout_s: float = DEFAULT_SHUTDOWN_TIMEOUT_S,
        python_executable: Path | None = None,
        process_environment: Mapping[str, str] | None = None,
    ) -> None:
        for name, value in (
            ("startup_timeout_s", startup_timeout_s),
            ("request_timeout_s", request_timeout_s),
            ("shutdown_timeout_s", shutdown_timeout_s),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        executable = python_executable or default_qwen_venv_python()
        if not executable.is_file():
            raise QwenServiceUnavailable(
                f"{executable} not found. Run scripts/provision_qwen_runtime.py "
                "to create the locked qwen-asr environment."
            )

        root = socket_dir or Path(tempfile.gettempdir())
        if not root.is_dir():
            raise QwenServiceUnavailable(f"Socket directory does not exist: {root}")
        self._socket_path = root / f"soca-qwen-asr-{uuid.uuid4().hex}.sock"
        if len(str(self._socket_path).encode()) > MAX_UNIX_SOCKET_PATH_BYTES:
            raise QwenServiceUnavailable(
                f"Unix socket path exceeds {MAX_UNIX_SOCKET_PATH_BYTES} bytes: {self._socket_path}"
            )
        self._ready_path = self._socket_path.with_suffix(".ready")
        self._request_timeout_s = request_timeout_s
        self._shutdown_timeout_s = shutdown_timeout_s
        self._state_lock = threading.Lock()
        self._active_sockets: set[socket.socket] = set()
        self._closed = False
        self._process: subprocess.Popen[bytes] | None = None
        self._launch = launch
        self._lifecycle_state = QwenServiceState.STARTING
        self._last_failure_type: str | None = None
        self.identity: QwenServiceIdentity | None = None

        try:
            child_environment = os.environ.copy()
            if process_environment is not None:
                child_environment.update(process_environment)
            for name in SENSITIVE_MODEL_ENVIRONMENT:
                child_environment.pop(name, None)
            for name in ISOLATED_PYTHON_ENVIRONMENT:
                child_environment.pop(name, None)
            child_environment["PYTHONNOUSERSITE"] = "1"
            child_environment["HF_HUB_OFFLINE"] = "1"
            child_environment["TRANSFORMERS_OFFLINE"] = "1"
            if launch.spec.device == "mps":
                # A production MPS worker must fail on an unsupported op rather
                # than silently execute part of the model on the CPU.
                child_environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
            self._process = subprocess.Popen(
                [
                    str(executable),
                    "-m",
                    "soca.asr.qwen_service_server",
                    "--socket-path",
                    str(self._socket_path),
                    "--artifact-key",
                    launch.spec.key,
                    "--model-path",
                    str(launch.model_path),
                    "--launch-mode",
                    launch.mode.value,
                    "--connection-timeout",
                    str(request_timeout_s),
                ],
                # The provisioned wheel, rather than source checkout or the
                # parent sidecar's _internal directory, is the only SoCa code
                # visible to the isolated Qwen worker.
                cwd=str(default_qwen_runtime_root()),
                stdin=subprocess.DEVNULL,
                env=child_environment,
            )
            self._wait_for_ready(startup_timeout_s)
            identity = self.live_identity()
            if identity.state is not QwenServiceState.READY:
                detail = identity.last_failure_type or identity.state.value
                raise QwenServiceNotReady(
                    f"Qwen ASR worker started in {identity.state.value}: {detail}"
                )
            self.model_key = identity.artifact_key
            self.supports_avg_logprob = identity.supports_avg_logprob
            self._lifecycle_state = identity.state
        except Exception as exc:
            self._record_failure(exc)
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

    @property
    def lifecycle_state(self) -> QwenServiceState:
        return self._lifecycle_state

    @property
    def last_failure_type(self) -> str | None:
        return self._last_failure_type

    def live_identity(self) -> QwenServiceIdentity:
        response = self._request({"op": "ping"})
        self._require_success(response, operation="ping")
        payload = response.get("identity")
        if not isinstance(payload, Mapping):
            error = QwenServiceIdentityMismatch("ping identity must be an object")
            self._record_failure(error)
            raise error
        try:
            identity = QwenServiceIdentity.from_wire(payload)
            identity.assert_matches(self._launch)
        except QwenServiceIdentityError as exc:
            error = QwenServiceIdentityMismatch(str(exc))
            self._record_failure(error)
            raise error from exc
        self.identity = identity
        if identity.state is QwenServiceState.FAILED:
            self._last_failure_type = identity.last_failure_type
        self._lifecycle_state = identity.state
        return identity

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
            error = QwenServiceCrashed(f"qwen_service_server exited with code {process.returncode}")
            self._record_failure(error)
            raise error
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
            error = QwenServiceTimeout(f"Qwen ASR request exceeded {self._request_timeout_s}s")
            self._record_failure(error)
            raise error from exc
        except QwenIPCProtocolError as exc:
            error = QwenServiceProtocolError(str(exc))
            self._record_failure(error)
            raise error from exc
        except (ConnectionError, OSError) as exc:
            error = QwenServiceCrashed(str(exc))
            self._record_failure(error)
            raise error from exc
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
            error = QwenServiceProtocolError(
                f"Response request_id {response_request_id!r} does not match {request_id!r}"
            )
            self._record_failure(error)
            raise error
        if response.get("ok") is not True:
            error = QwenTranscribeError(
                self._required_string(response, "error_type"),
                self._required_string(response, "error_message"),
                request_id,
            )
            self._record_failure(error)
            raise error
        try:
            return ASRResult(
                text=self._required_string(response, "text"),
                latency_ms=float(response["latency_ms"]),
                audio_duration_ms=float(response["audio_duration_ms"]),
                rtf=float(response["rtf"]),
                avg_logprob=float(response["avg_logprob"]),
                avg_logprob_reliable=self._required_bool(response, "avg_logprob_reliable"),
                alternatives=tuple(response["alternatives"]),
                generated_token_count=self._optional_int(response, "generated_token_count"),
                hit_max_new_tokens=self._optional_bool(response, "hit_max_new_tokens"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            error = QwenServiceProtocolError(f"Malformed transcribe response: {exc}")
            self._record_failure(error)
            raise error from exc

    def runtime_metadata(self, max_new_tokens: int = 128) -> dict[str, Any]:
        response = self._request({"op": "runtime_metadata", "max_new_tokens": max_new_tokens})
        self._require_success(response, operation="runtime_metadata")
        metadata = response.get("metadata")
        if not isinstance(metadata, dict):
            error = QwenServiceProtocolError("runtime_metadata response is not an object")
            self._record_failure(error)
            raise error
        return metadata

    @staticmethod
    def _optional_int(response: Mapping[str, Any], name: str) -> int | None:
        value = response.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer or null")
        return value

    @staticmethod
    def _optional_bool(response: Mapping[str, Any], name: str) -> bool | None:
        value = response.get(name)
        if value is None:
            return None
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean or null")
        return value

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            if self._last_failure_type is None:
                self._lifecycle_state = QwenServiceState.STOPPING
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
        if self._last_failure_type is None:
            self._lifecycle_state = QwenServiceState.STOPPED

    def _record_failure(self, error: Exception) -> None:
        self._last_failure_type = type(error).__name__
        self._lifecycle_state = QwenServiceState.FAILED

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

    def _require_success(self, payload: Mapping[str, Any], *, operation: str) -> None:
        if payload.get("ok") is True:
            return
        error_type = self._required_string(payload, "error_type")
        error_message = self._required_string(payload, "error_message")
        error = QwenServiceProtocolError(f"{operation} failed with {error_type}: {error_message}")
        self._record_failure(error)
        raise error


__all__ = [
    "QWEN_VENV_PYTHON",
    "QwenASRServiceClient",
    "QwenServiceCrashed",
    "QwenServiceIdentityMismatch",
    "QwenServiceNotReady",
    "QwenServiceProtocolError",
    "QwenServiceTimeout",
    "QwenServiceUnavailable",
    "QwenTranscribeError",
]
