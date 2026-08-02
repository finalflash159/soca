from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace

import numpy as np

from soca.asr.qwen_ipc_protocol import SAMPLE_RATE, recv_header, send_frame
from soca.asr.qwen_service_server import _handle_connection


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
            )
        finally:
            with self._state_lock:
                self.active -= 1


def _start_handler(
    backend: FakeBackend,
    transcribe_lock: threading.Lock,
) -> tuple[socket.socket, socket.socket, threading.Thread]:
    client, server = socket.socketpair()
    thread = threading.Thread(
        target=_handle_connection,
        args=(server, backend, transcribe_lock),
        daemon=True,
    )
    thread.start()
    return client, server, thread


def _transcribe_header(request_id: str, n_samples: int) -> dict[str, object]:
    return {
        "op": "transcribe",
        "request_id": request_id,
        "max_new_tokens": 192,
        "context": "tech",
        "n_samples": n_samples,
        "sample_rate": SAMPLE_RATE,
    }


def test_server_contract_and_connection_survives_backend_error() -> None:
    backend = FakeBackend()
    client, server, thread = _start_handler(backend, threading.Lock())
    try:
        send_frame(client, {"op": "ping"})
        assert recv_header(client) == {
            "ok": True,
            "model_key": backend.model_key,
            "supports_avg_logprob": True,
            "context": backend.context,
            "language": backend.language,
        }

        send_frame(client, {"op": "runtime_metadata", "max_new_tokens": 256})
        assert recv_header(client) == {
            "ok": True,
            "metadata": {"backend": "fake", "max_new_tokens": 256},
        }

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
        assert recv_header(client)["ok"] is True

        send_frame(client, {"op": "not-real"})
        response = recv_header(client)
        assert response["ok"] is False
        assert response["error_type"] == "UnknownOp"
    finally:
        client.close()
        thread.join(timeout=1.0)
        server.close()
    assert not thread.is_alive()


def test_two_connections_never_enter_model_concurrently() -> None:
    backend = FakeBackend()
    transcribe_lock = threading.Lock()
    first = _start_handler(backend, transcribe_lock)
    second = _start_handler(backend, transcribe_lock)
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
