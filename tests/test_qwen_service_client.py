from __future__ import annotations

import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from local.qwen_contexts import CONTEXTS
from soca.asr.qwen_artifacts import QWEN_REFERENCE_ARTIFACT
from soca.asr.qwen_ipc_protocol import recv_audio_payload, recv_header, send_frame
from soca.asr.qwen_service_client import (
    QWEN_VENV_PYTHON,
    QwenASRServiceClient,
    QwenServiceCrashed,
    QwenServiceProtocolError,
    QwenServiceTimeout,
    QwenServiceUnavailable,
    QwenTranscribeError,
)


class FakeQwenService:
    def __init__(self, socket_path: Path, *, valid_handshake: bool = True) -> None:
        self.socket_path = socket_path
        self.ready_path = socket_path.with_suffix(".ready")
        self.valid_handshake = valid_handshake
        self.received_audio: list[np.ndarray] = []
        self._stop = threading.Event()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(socket_path))
        self._server.listen(8)
        self._server.settimeout(0.05)
        self.ready_path.touch()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except TimeoutError:
                continue
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            header = recv_header(conn)
            op = header.get("op")
            if op == "shutdown":
                send_frame(conn, {"ok": True})
                self._stop.set()
                return
            if op == "ping":
                response = (
                    {
                        "ok": True,
                        "model_key": "fake/model",
                        "supports_avg_logprob": True,
                        "context": "fake context",
                        "language": "Vietnamese",
                    }
                    if self.valid_handshake
                    else {"ok": True, "model_key": 123}
                )
                send_frame(conn, response)
                return
            if op == "runtime_metadata":
                send_frame(
                    conn,
                    {
                        "ok": True,
                        "metadata": {"max_new_tokens": header["max_new_tokens"]},
                    },
                )
                return
            if op != "transcribe":
                send_frame(
                    conn,
                    {
                        "ok": False,
                        "error_type": "UnknownOp",
                        "error_message": "unsupported",
                    },
                )
                return

            audio = recv_audio_payload(conn, header)
            self.received_audio.append(audio.copy())
            context = header.get("context")
            if context == "CRASH":
                return
            if context == "HANG":
                time.sleep(0.2)
            if context == "ERROR":
                send_frame(
                    conn,
                    {
                        "ok": False,
                        "request_id": header["request_id"],
                        "error_type": "ValueError",
                        "error_message": "bad audio",
                    },
                )
                return
            request_id = (
                "wrong-id" if context == "MISMATCH" else header["request_id"]
            )
            try:
                send_frame(
                    conn,
                    {
                        "ok": True,
                        "request_id": request_id,
                        "text": "Bản ghi thử nghiệm.",
                        "latency_ms": 150.0,
                        "audio_duration_ms": 1_000.0,
                        "rtf": 0.15,
                        "avg_logprob": -0.05,
                        "avg_logprob_reliable": True,
                        "alternatives": [],
                    },
                )
            except BrokenPipeError:
                pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._server.close()
        self.socket_path.unlink(missing_ok=True)
        self.ready_path.unlink(missing_ok=True)


@pytest.fixture
def fake_subprocess(monkeypatch, tmp_path):
    executable = tmp_path / "python"
    executable.touch()
    process = MagicMock()
    process.poll.return_value = None
    process.returncode = None
    process.wait.return_value = 0
    services: list[FakeQwenService] = []
    socket_dir = Path(tempfile.mkdtemp(prefix="soca-qwen-test-", dir="/tmp"))
    def popen(args, **kwargs):
        del kwargs
        socket_path = Path(args[args.index("--socket-path") + 1])
        services.append(
            FakeQwenService(socket_path, valid_handshake=True)
        )
        return process

    monkeypatch.setattr("soca.asr.qwen_service_client.subprocess.Popen", popen)

    def build_client(**kwargs) -> QwenASRServiceClient:
        return QwenASRServiceClient(
            socket_dir=socket_dir,
            python_executable=executable,
            **kwargs,
        )

    yield build_client, process, services

    for service in services:
        service.close()
    shutil.rmtree(socket_dir, ignore_errors=True)


def test_client_rejects_missing_external_python(tmp_path) -> None:
    with pytest.raises(QwenServiceUnavailable, match="not found"):
        QwenASRServiceClient(python_executable=tmp_path / "missing")


def test_client_handshake_transcribe_metadata_and_idempotent_close(
    fake_subprocess,
) -> None:
    build_client, process, services = fake_subprocess
    client = build_client()
    audio = np.linspace(-0.5, 0.5, 1_600, dtype=np.float32)

    result = client.transcribe(audio)

    assert client.model_key == "fake/model"
    assert client.context == "fake context"
    assert result.text == "Bản ghi thử nghiệm."
    assert result.avg_logprob == -0.05
    np.testing.assert_array_equal(services[0].received_audio[0], audio)
    assert client.runtime_metadata(256) == {"max_new_tokens": 256}

    client.close()
    client.close()
    process.wait.assert_called_once()
    assert not client.socket_path.exists()
    assert not client.ready_path.exists()


def test_worker_environment_is_offline_and_does_not_inherit_tokens(
    monkeypatch, tmp_path
) -> None:
    executable = tmp_path / "python"
    executable.touch()
    process = MagicMock()
    process.poll.return_value = None
    process.wait.return_value = 0
    services: list[FakeQwenService] = []
    captured_environments: list[dict[str, str]] = []
    socket_dir = Path(tempfile.mkdtemp(prefix="soca-qwen-test-", dir="/tmp"))
    monkeypatch.setenv("HF_TOKEN", "must-not-reach-worker")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "also-private")

    def popen(args, **kwargs):
        captured_environments.append(kwargs["env"])
        socket_path = Path(args[args.index("--socket-path") + 1])
        services.append(FakeQwenService(socket_path))
        return process

    monkeypatch.setattr("soca.asr.qwen_service_client.subprocess.Popen", popen)
    client = QwenASRServiceClient(
        socket_dir=socket_dir,
        python_executable=executable,
        process_environment={
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    client.close()
    default_client = QwenASRServiceClient(
        socket_dir=socket_dir,
        python_executable=executable,
    )
    default_client.close()

    assert captured_environments[0]["HF_HUB_OFFLINE"] == "1"
    assert captured_environments[0]["TRANSFORMERS_OFFLINE"] == "1"
    for environment in captured_environments:
        assert "PATH" in environment
        assert "HF_TOKEN" not in environment
        assert "HUGGING_FACE_HUB_TOKEN" not in environment
    for service in services:
        service.close()
    shutil.rmtree(socket_dir, ignore_errors=True)


def test_transport_drop_is_not_reported_as_transcription_error(fake_subprocess) -> None:
    build_client, _, _ = fake_subprocess
    client = build_client()
    try:
        with pytest.raises(QwenServiceCrashed, match="closed while reading frame"):
            client.transcribe(np.zeros(1_600, dtype=np.float32), context="CRASH")
    finally:
        client.close()


def test_backend_error_is_typed_and_service_remains_usable(fake_subprocess) -> None:
    build_client, _, _ = fake_subprocess
    client = build_client()
    audio = np.zeros(1_600, dtype=np.float32)
    try:
        with pytest.raises(QwenTranscribeError) as exc_info:
            client.transcribe(audio, context="ERROR")
        assert exc_info.value.remote_type == "ValueError"
        assert client.transcribe(audio).text == "Bản ghi thử nghiệm."
    finally:
        client.close()


def test_request_timeout_is_bounded(fake_subprocess) -> None:
    build_client, _, _ = fake_subprocess
    client = build_client(request_timeout_s=0.05)
    try:
        started = time.monotonic()
        with pytest.raises(QwenServiceTimeout):
            client.transcribe(np.zeros(1_600, dtype=np.float32), context="HANG")
        assert time.monotonic() - started < 0.5
    finally:
        client.close()


def test_mismatched_request_id_is_protocol_failure(fake_subprocess) -> None:
    build_client, _, _ = fake_subprocess
    client = build_client()
    try:
        with pytest.raises(QwenServiceProtocolError, match="does not match"):
            client.transcribe(np.zeros(1_600, dtype=np.float32), context="MISMATCH")
    finally:
        client.close()


def test_constructor_failure_reaps_process_and_files(monkeypatch, tmp_path) -> None:
    executable = tmp_path / "python"
    executable.touch()
    process = MagicMock()
    process.poll.return_value = None
    process.wait.return_value = 0
    services: list[FakeQwenService] = []
    socket_dir = Path(tempfile.mkdtemp(prefix="soca-qwen-test-", dir="/tmp"))

    def popen(args, **kwargs):
        del kwargs
        socket_path = Path(args[args.index("--socket-path") + 1])
        services.append(FakeQwenService(socket_path, valid_handshake=False))
        return process

    monkeypatch.setattr("soca.asr.qwen_service_client.subprocess.Popen", popen)
    with pytest.raises(QwenServiceProtocolError, match="model_key"):
        QwenASRServiceClient(
            socket_dir=socket_dir,
            python_executable=executable,
        )

    process.wait.assert_called_once()
    assert not list(socket_dir.glob("soca-qwen-asr-*"))
    for service in services:
        service.close()
    shutil.rmtree(socket_dir, ignore_errors=True)


@pytest.mark.real_model
def test_real_qwen_service_transcribes_recorded_voice_and_cleans_up() -> None:
    if not QWEN_VENV_PYTHON.exists():
        pytest.skip(f"Qwen environment not found: {QWEN_VENV_PYTHON}")
    audio, sample_rate = sf.read(
        "data/asr_codeswitch/wav/cs_033.wav",
        dtype="float32",
    )
    assert sample_rate == 16_000
    client = QwenASRServiceClient(
        model_id=QWEN_REFERENCE_ARTIFACT.upstream.repo_id,
        context=CONTEXTS["tech"],
        startup_timeout_s=120.0,
    )
    process = client.process
    try:
        partial = client.transcribe(audio, context="")
        final = client.transcribe(audio)
        for result in (partial, final):
            normalized = result.text.casefold()
            assert "log level" in normalized
            assert "debug" in normalized
            assert "info" in normalized
            assert result.avg_logprob < 0.0
    finally:
        client.close()

    assert process is not None
    assert process.poll() == 0
    assert not client.socket_path.exists()
    assert not client.ready_path.exists()
