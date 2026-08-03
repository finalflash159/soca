from __future__ import annotations

import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from local.qwen_contexts import CONTEXTS
from soca.asr.qwen_artifacts import (
    QWEN_REFERENCE_ARTIFACT,
    QWEN_RELEASE_ARTIFACT,
    QwenASRArtifactSpec,
)
from soca.asr.qwen_ipc_protocol import recv_audio_payload, recv_header, send_frame
from soca.asr.qwen_service_client import (
    QWEN_VENV_PYTHON,
    QwenASRServiceClient,
    QwenServiceCrashed,
    QwenServiceIdentityMismatch,
    QwenServiceNotReady,
    QwenServiceProtocolError,
    QwenServiceTimeout,
    QwenServiceUnavailable,
    QwenTranscribeError,
)
from soca.asr.qwen_service_identity import (
    QWEN_SERVICE_PROTOCOL_VERSION,
    QwenLaunchMode,
    QwenServiceIdentity,
    QwenServiceLaunch,
    QwenServiceState,
)
from soca.asr.qwen_store import QwenArtifactStore


class FakeQwenService:
    def __init__(
        self,
        socket_path: Path,
        launch: QwenServiceLaunch,
        *,
        valid_handshake: bool = True,
    ) -> None:
        self.socket_path = socket_path
        self.ready_path = socket_path.with_suffix(".ready")
        self.valid_handshake = valid_handshake
        self.launch = launch
        self.failure_type: str | None = None
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
                spec = self.launch.spec
                identity = QwenServiceIdentity(
                    protocol_version=QWEN_SERVICE_PROTOCOL_VERSION,
                    state=(
                        QwenServiceState.FAILED
                        if self.failure_type is not None
                        else QwenServiceState.READY
                    ),
                    launch_mode=self.launch.mode,
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
                    package_versions={
                        "qwen-asr": "test",
                        "soca": "test",
                        "torch": "test",
                        "transformers": "test",
                    },
                    pid=123,
                    uptime_ms=10.0,
                    in_flight=0,
                    supports_avg_logprob=True,
                    last_failure_type=self.failure_type,
                    no_fallback_attempted=True,
                ).to_wire()
                if not self.valid_handshake:
                    identity["artifact_digest"] = "wrong"
                response = {"ok": True, "identity": identity}
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
                self.failure_type = "ValueError"
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
            request_id = "wrong-id" if context == "MISMATCH" else header["request_id"]
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
                        "generated_token_count": 17,
                        "hit_max_new_tokens": False,
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
    model_path = tmp_path / "model"
    model_path.mkdir()
    launch = QwenServiceLaunch.for_provisioning(QWEN_RELEASE_ARTIFACT, model_path)
    socket_dir = Path(tempfile.mkdtemp(prefix="soca-qwen-test-", dir="/tmp"))

    def popen(args, **kwargs):
        del kwargs
        socket_path = Path(args[args.index("--socket-path") + 1])
        services.append(FakeQwenService(socket_path, launch, valid_handshake=True))
        return process

    monkeypatch.setattr("soca.asr.qwen_service_client.subprocess.Popen", popen)

    def build_client(**kwargs) -> QwenASRServiceClient:
        return QwenASRServiceClient(
            launch=launch,
            socket_dir=socket_dir,
            python_executable=executable,
            **kwargs,
        )

    yield build_client, process, services

    for service in services:
        service.close()
    shutil.rmtree(socket_dir, ignore_errors=True)


def test_client_rejects_missing_external_python(tmp_path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    launch = QwenServiceLaunch.for_provisioning(QWEN_RELEASE_ARTIFACT, model_path)
    with pytest.raises(QwenServiceUnavailable, match="not found"):
        QwenASRServiceClient(
            launch=launch,
            python_executable=tmp_path / "missing",
        )


def test_client_handshake_transcribe_metadata_and_idempotent_close(
    fake_subprocess,
) -> None:
    build_client, process, services = fake_subprocess
    client = build_client()
    audio = np.linspace(-0.5, 0.5, 1_600, dtype=np.float32)

    result = client.transcribe(audio)

    assert client.model_key == QWEN_RELEASE_ARTIFACT.key
    assert client.identity is not None
    assert client.identity.state is QwenServiceState.READY
    assert result.text == "Bản ghi thử nghiệm."
    assert result.avg_logprob == -0.05
    assert result.generated_token_count == 17
    assert result.hit_max_new_tokens is False
    np.testing.assert_array_equal(services[0].received_audio[0], audio)
    assert client.runtime_metadata(256) == {"max_new_tokens": 256}

    client.close()
    client.close()
    process.wait.assert_called_once()
    assert not client.socket_path.exists()
    assert not client.ready_path.exists()
    assert client.lifecycle_state is QwenServiceState.STOPPED


def test_worker_environment_is_offline_and_does_not_inherit_tokens(monkeypatch, tmp_path) -> None:
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
    model_path = tmp_path / "model"
    model_path.mkdir()
    launch = QwenServiceLaunch.for_provisioning(QWEN_RELEASE_ARTIFACT, model_path)

    def popen(args, **kwargs):
        captured_environments.append(kwargs["env"])
        socket_path = Path(args[args.index("--socket-path") + 1])
        services.append(FakeQwenService(socket_path, launch))
        return process

    monkeypatch.setattr("soca.asr.qwen_service_client.subprocess.Popen", popen)
    client = QwenASRServiceClient(
        launch=launch,
        socket_dir=socket_dir,
        python_executable=executable,
        process_environment={
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    client.close()
    default_client = QwenASRServiceClient(
        launch=launch,
        socket_dir=socket_dir,
        python_executable=executable,
    )
    default_client.close()

    assert captured_environments[0]["HF_HUB_OFFLINE"] == "1"
    assert captured_environments[0]["TRANSFORMERS_OFFLINE"] == "1"
    assert captured_environments[0]["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    for environment in captured_environments:
        assert "PATH" in environment
        assert "HF_TOKEN" not in environment
        assert "HUGGING_FACE_HUB_TOKEN" not in environment
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert environment["TRANSFORMERS_OFFLINE"] == "1"
        assert environment["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
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
        assert client.live_identity().state is QwenServiceState.FAILED
        assert client.last_failure_type == "ValueError"
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
        assert client.lifecycle_state is QwenServiceState.FAILED
        assert client.last_failure_type == "QwenServiceProtocolError"
    finally:
        client.close()


def test_constructor_failure_reaps_process_and_files(monkeypatch, tmp_path) -> None:
    executable = tmp_path / "python"
    executable.touch()
    process = MagicMock()
    process.poll.return_value = None
    process.wait.return_value = 0
    services: list[FakeQwenService] = []
    model_path = tmp_path / "model"
    model_path.mkdir()
    launch = QwenServiceLaunch.for_provisioning(QWEN_RELEASE_ARTIFACT, model_path)
    socket_dir = Path(tempfile.mkdtemp(prefix="soca-qwen-test-", dir="/tmp"))

    def popen(args, **kwargs):
        del kwargs
        socket_path = Path(args[args.index("--socket-path") + 1])
        services.append(FakeQwenService(socket_path, launch, valid_handshake=False))
        return process

    monkeypatch.setattr("soca.asr.qwen_service_client.subprocess.Popen", popen)
    with pytest.raises(QwenServiceIdentityMismatch, match="does not match"):
        QwenASRServiceClient(
            launch=launch,
            socket_dir=socket_dir,
            python_executable=executable,
        )

    process.wait.assert_called_once()
    assert not list(socket_dir.glob("soca-qwen-asr-*"))
    for service in services:
        service.close()
    shutil.rmtree(socket_dir, ignore_errors=True)


def test_constructor_rejects_failed_worker_state(monkeypatch, tmp_path) -> None:
    executable = tmp_path / "python"
    executable.touch()
    process = MagicMock()
    process.poll.return_value = None
    process.wait.return_value = 0
    services: list[FakeQwenService] = []
    model_path = tmp_path / "model"
    model_path.mkdir()
    launch = QwenServiceLaunch.for_provisioning(QWEN_RELEASE_ARTIFACT, model_path)
    socket_dir = Path(tempfile.mkdtemp(prefix="soca-qwen-test-", dir="/tmp"))

    def popen(args, **kwargs):
        del kwargs
        socket_path = Path(args[args.index("--socket-path") + 1])
        service = FakeQwenService(socket_path, launch)
        service.failure_type = "BackendFailure"
        services.append(service)
        return process

    monkeypatch.setattr("soca.asr.qwen_service_client.subprocess.Popen", popen)
    try:
        with pytest.raises(QwenServiceNotReady, match="failed.*BackendFailure"):
            QwenASRServiceClient(
                launch=launch,
                socket_dir=socket_dir,
                python_executable=executable,
            )
        process.wait.assert_called_once()
        assert not list(socket_dir.glob("soca-qwen-asr-*"))
    finally:
        for service in services:
            service.close()
        shutil.rmtree(socket_dir, ignore_errors=True)


def test_stale_ready_marker_cannot_become_ready_and_reaps_process(monkeypatch, tmp_path) -> None:
    executable = tmp_path / "python"
    executable.touch()
    process = MagicMock()
    process.poll.return_value = None
    process.wait.return_value = 0
    model_path = tmp_path / "model"
    model_path.mkdir()
    launch = QwenServiceLaunch.for_provisioning(QWEN_RELEASE_ARTIFACT, model_path)
    socket_dir = Path(tempfile.mkdtemp(prefix="soca-qwen-stale-", dir="/tmp"))

    def popen(args, **kwargs):
        del kwargs
        socket_path = Path(args[args.index("--socket-path") + 1])
        socket_path.with_suffix(".ready").touch()
        return process

    monkeypatch.setattr("soca.asr.qwen_service_client.subprocess.Popen", popen)
    try:
        with pytest.raises(QwenServiceCrashed):
            QwenASRServiceClient(
                launch=launch,
                socket_dir=socket_dir,
                python_executable=executable,
                request_timeout_s=0.05,
            )

        process.terminate.assert_called_once()
        process.wait.assert_called_once()
        assert not list(socket_dir.iterdir())
    finally:
        shutil.rmtree(socket_dir, ignore_errors=True)


@pytest.mark.parametrize(
    ("process_exit", "expected_error"),
    [
        (7, QwenServiceCrashed),
        (None, QwenServiceTimeout),
    ],
)
def test_startup_failure_is_typed_bounded_and_cleans_files(
    monkeypatch,
    tmp_path,
    process_exit,
    expected_error,
) -> None:
    executable = tmp_path / "python"
    executable.touch()
    process = MagicMock()
    process.poll.return_value = process_exit
    process.returncode = process_exit
    process.wait.return_value = 0
    model_path = tmp_path / "model"
    model_path.mkdir()
    launch = QwenServiceLaunch.for_provisioning(QWEN_RELEASE_ARTIFACT, model_path)
    socket_dir = Path(tempfile.mkdtemp(prefix="soca-qwen-startup-", dir="/tmp"))
    monkeypatch.setattr(
        "soca.asr.qwen_service_client.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    started = time.monotonic()
    try:
        with pytest.raises(expected_error):
            QwenASRServiceClient(
                launch=launch,
                socket_dir=socket_dir,
                python_executable=executable,
                startup_timeout_s=0.02,
                shutdown_timeout_s=0.02,
            )
        assert time.monotonic() - started < 0.3
        assert not list(socket_dir.iterdir())
        if process_exit is None:
            process.terminate.assert_called_once()
            process.wait.assert_called_once()
        else:
            process.terminate.assert_not_called()
    finally:
        shutil.rmtree(socket_dir, ignore_errors=True)


def test_shutdown_escalates_from_terminate_to_kill(fake_subprocess, monkeypatch) -> None:
    build_client, process, _ = fake_subprocess
    client = build_client(shutdown_timeout_s=0.05)
    monkeypatch.setattr(client, "_request_graceful_shutdown", lambda: False)
    process.wait.side_effect = [subprocess.TimeoutExpired("qwen", 0.05), 0]

    client.close()

    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2
    assert client.lifecycle_state is QwenServiceState.STOPPED


@pytest.mark.real_model
@pytest.mark.parametrize(
    "artifact",
    [QWEN_RELEASE_ARTIFACT, QWEN_REFERENCE_ARTIFACT],
    ids=["release-0.6b", "reference-1.7b"],
)
def test_real_qwen_service_transcribes_recorded_voice_and_cleans_up(
    artifact: QwenASRArtifactSpec,
) -> None:
    if not QWEN_VENV_PYTHON.exists():
        pytest.skip(f"Qwen environment not found: {QWEN_VENV_PYTHON}")
    audio, sample_rate = sf.read(
        "data/asr_codeswitch/wav/cs_033.wav",
        dtype="float32",
    )
    assert sample_rate == 16_000
    if not artifact.model_path().is_dir():
        pytest.skip(f"Qwen artifact is not provisioned: {artifact.key}")
    client = QwenASRServiceClient(
        launch=QwenServiceLaunch.for_active(
            artifact,
            QwenArtifactStore(artifact.model_path().parents[1]).verify(artifact, deep=False),
        ),
        startup_timeout_s=120.0,
    )
    process = client.process
    try:
        assert client.identity is not None
        assert client.identity.artifact_key == artifact.key
        assert client.identity.artifact_digest == artifact.digest
        assert client.identity.launch_mode is QwenLaunchMode.ACTIVE
        assert client.identity.no_fallback_attempted is True
        partial = client.transcribe(audio, context="")
        final = client.transcribe(audio, context=CONTEXTS["tech"])
        for result in (partial, final):
            normalized = result.text.casefold()
            assert "level" in normalized
            assert "debug" in normalized
            assert "info" in normalized
            assert result.avg_logprob < 0.0
        if artifact is QWEN_REFERENCE_ARTIFACT:
            assert "log level" in partial.text.casefold()
            assert "log level" in final.text.casefold()
    finally:
        client.close()

    assert process is not None
    assert process.poll() == 0
    assert not client.socket_path.exists()
    assert not client.ready_path.exists()
