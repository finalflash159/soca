from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace

import numpy as np

from soca.asr.qwen_artifacts import QWEN_RELEASE_ARTIFACT
from soca.asr.qwen_ipc_protocol import SAMPLE_RATE, recv_header, send_frame
from soca.asr.qwen_service_identity import (
    QwenServiceIdentity,
    QwenServiceLaunch,
    QwenServiceState,
)
from soca.asr.qwen_service_server import ServiceLifecycle, _handle_connection


class FakeBackend:
    model_key = "fake_qwen_1.7b"
    supports_avg_logprob = True
    context = "test context"
    language = "Vietnamese"

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, int, str | None]] = []
        self.active = 0
        self.max_active = 0
        self._state_lock = threading.Lock()

    def runtime_metadata(self, max_new_tokens: int = 128) -> dict[str, object]:
        if max_new_tokens == 13:
            raise RuntimeError("metadata unavailable")
        return {"backend": "fake", "max_new_tokens": max_new_tokens}

    def transcribe(
        self,
        audio: np.ndarray,
        max_new_tokens: int = 128,
        *,
        context: str | None = None,
    ) -> SimpleNamespace:
        if len(audio) == 0:
            raise ValueError("empty audio")
        with self._state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.02)
            self.calls.append((audio.copy(), max_new_tokens, context))
            return SimpleNamespace(
                text="Xin chào bạn",
                latency_ms=45.0,
                audio_duration_ms=1_000.0,
                rtf=0.045,
                avg_logprob=-0.1,
                avg_logprob_reliable=True,
                alternatives=(),
                generated_token_count=12,
                hit_max_new_tokens=False,
            )
        finally:
            with self._state_lock:
                self.active -= 1


def _start_handler(
    backend: FakeBackend,
    transcribe_lock: threading.Lock,
    lifecycle: ServiceLifecycle,
) -> tuple[socket.socket, socket.socket, threading.Thread]:
    client, server = socket.socketpair()
    thread = threading.Thread(
        target=_handle_connection,
        args=(server, backend, transcribe_lock, lifecycle),
        daemon=True,
    )
    thread.start()
    return client, server, thread


def _lifecycle(tmp_path) -> ServiceLifecycle:
    model_path = tmp_path / "model"
    model_path.mkdir(exist_ok=True)
    launch = QwenServiceLaunch.for_provisioning(QWEN_RELEASE_ARTIFACT, model_path)
    return ServiceLifecycle(
        launch,
        started_at=time.monotonic(),
        packages={
            "qwen-asr": "test",
            "soca": "test",
            "torch": "test",
            "transformers": "test",
        },
    )


def _transcribe_header(request_id: str, n_samples: int) -> dict[str, object]:
    return {
        "op": "transcribe",
        "request_id": request_id,
        "max_new_tokens": 192,
        "context": "tech",
        "n_samples": n_samples,
        "sample_rate": SAMPLE_RATE,
    }


def test_server_contract_and_connection_survives_backend_error(tmp_path) -> None:
    backend = FakeBackend()
    lifecycle = _lifecycle(tmp_path)
    client, server, thread = _start_handler(backend, threading.Lock(), lifecycle)
    try:
        send_frame(client, {"op": "ping"})
        ping = recv_header(client)
        identity = QwenServiceIdentity.from_wire(ping["identity"])
        assert ping["ok"] is True
        assert identity.artifact_key == QWEN_RELEASE_ARTIFACT.key
        assert identity.state is QwenServiceState.READY
        assert identity.supports_avg_logprob is True

        send_frame(client, {"op": "runtime_metadata", "max_new_tokens": 256})
        assert recv_header(client) == {
            "ok": True,
            "metadata": {"backend": "fake", "max_new_tokens": 256},
        }

        send_frame(client, {"op": "runtime_metadata", "max_new_tokens": 13})
        assert recv_header(client)["error_type"] == "RuntimeError"
        send_frame(client, {"op": "ping"})
        metadata_failure = QwenServiceIdentity.from_wire(recv_header(client)["identity"])
        assert metadata_failure.state is QwenServiceState.FAILED
        assert metadata_failure.last_failure_type == "RuntimeError"

        send_frame(client, {"op": "runtime_metadata", "max_new_tokens": True})
        assert recv_header(client)["error_type"] == "ProtocolError"

        audio = np.array([0.1, 0.2, -0.1], dtype="<f4")
        send_frame(client, _transcribe_header("req-1", len(audio)), audio.tobytes())
        response = recv_header(client)
        assert response["ok"] is True
        assert response["request_id"] == "req-1"
        assert response["text"] == "Xin chào bạn"
        np.testing.assert_array_equal(backend.calls[0][0], audio)
        assert backend.calls[0][1:] == (192, "tech")

        send_frame(client, _transcribe_header("req-2", 0))
        response = recv_header(client)
        assert response["ok"] is False
        assert response["request_id"] == "req-2"
        assert response["error_type"] == "ValueError"

        send_frame(client, {"op": "ping"})
        failed_identity = QwenServiceIdentity.from_wire(recv_header(client)["identity"])
        assert failed_identity.state is QwenServiceState.FAILED
        assert failed_identity.last_failure_type == "ValueError"

        send_frame(client, {"op": "not-real"})
        response = recv_header(client)
        assert response["ok"] is False
        assert response["error_type"] == "UnknownOp"
    finally:
        client.close()
        thread.join(timeout=1.0)
        server.close()
    assert not thread.is_alive()


def test_two_connections_never_enter_model_concurrently(tmp_path) -> None:
    backend = FakeBackend()
    transcribe_lock = threading.Lock()
    lifecycle = _lifecycle(tmp_path)
    first = _start_handler(backend, transcribe_lock, lifecycle)
    second = _start_handler(backend, transcribe_lock, lifecycle)
    audio = np.ones(16, dtype="<f4")
    try:
        send_frame(first[0], _transcribe_header("first", len(audio)), audio.tobytes())
        send_frame(second[0], _transcribe_header("second", len(audio)), audio.tobytes())
        assert recv_header(first[0])["ok"] is True
        assert recv_header(second[0])["ok"] is True
        assert backend.max_active == 1
    finally:
        for client, server, thread in (first, second):
            client.close()
            thread.join(timeout=1.0)
            server.close()


def test_invalid_metadata_request_does_not_poison_service_health(tmp_path) -> None:
    backend = FakeBackend()
    lifecycle = _lifecycle(tmp_path)
    client, server, thread = _start_handler(backend, threading.Lock(), lifecycle)
    try:
        send_frame(client, {"op": "runtime_metadata", "max_new_tokens": True})
        assert recv_header(client)["error_type"] == "ProtocolError"
        send_frame(client, {"op": "ping"})
        identity = QwenServiceIdentity.from_wire(recv_header(client)["identity"])
        assert identity.state is QwenServiceState.READY
        assert identity.last_failure_type is None
    finally:
        client.close()
        thread.join(timeout=1.0)
        server.close()


def test_ping_reports_busy_while_inference_is_in_flight(tmp_path) -> None:
    backend = FakeBackend()
    lifecycle = _lifecycle(tmp_path)
    transcribe = _start_handler(backend, threading.Lock(), lifecycle)
    status = _start_handler(backend, threading.Lock(), lifecycle)
    audio = np.ones(16, dtype="<f4")
    try:
        send_frame(
            transcribe[0],
            _transcribe_header("busy", len(audio)),
            audio.tobytes(),
        )
        deadline = time.monotonic() + 1.0
        observed = QwenServiceState.READY
        while time.monotonic() < deadline and observed is not QwenServiceState.BUSY:
            send_frame(status[0], {"op": "ping"})
            observed = QwenServiceIdentity.from_wire(recv_header(status[0])["identity"]).state
        assert observed is QwenServiceState.BUSY
        assert recv_header(transcribe[0])["ok"] is True
    finally:
        for client, server, thread in (transcribe, status):
            client.close()
            thread.join(timeout=1.0)
            server.close()
