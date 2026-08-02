from __future__ import annotations

import socket
import struct

import numpy as np
import pytest

from soca.asr.qwen_ipc_protocol import (
    HEADER_LENGTH_FORMAT,
    MAX_AUDIO_SAMPLES,
    MAX_HEADER_BYTES,
    QwenIPCProtocolError,
    recv_audio_payload,
    recv_exact,
    recv_header,
    send_frame,
)


def test_frame_and_audio_round_trip_over_socketpair() -> None:
    sender, receiver = socket.socketpair()
    audio = np.array([0.5, -0.5, 0.25], dtype="<f4")
    try:
        send_frame(
            sender,
            {"op": "transcribe", "n_samples": len(audio)},
            audio.tobytes(),
        )

        header = recv_header(receiver)
        received = recv_audio_payload(receiver, header)

        assert header == {"op": "transcribe", "n_samples": 3}
        np.testing.assert_array_equal(received, audio)
    finally:
        sender.close()
        receiver.close()


def test_recv_exact_handles_fragmented_frame() -> None:
    sender, receiver = socket.socketpair()
    try:
        sender.sendall(b"abc")
        sender.sendall(b"def")
        assert recv_exact(receiver, 6) == b"abcdef"
    finally:
        sender.close()
        receiver.close()


def test_recv_exact_raises_when_peer_closes_mid_frame() -> None:
    sender, receiver = socket.socketpair()
    try:
        sender.sendall(b"abc")
        sender.shutdown(socket.SHUT_WR)
        with pytest.raises(ConnectionError, match="closed while reading frame"):
            recv_exact(receiver, 6)
    finally:
        sender.close()
        receiver.close()


@pytest.mark.parametrize("header_length", [0, MAX_HEADER_BYTES + 1])
def test_recv_header_rejects_invalid_length_before_allocating(header_length: int) -> None:
    sender, receiver = socket.socketpair()
    try:
        sender.sendall(struct.pack(HEADER_LENGTH_FORMAT, header_length))
        with pytest.raises(QwenIPCProtocolError, match="Header size"):
            recv_header(receiver)
    finally:
        sender.close()
        receiver.close()


def test_recv_header_rejects_non_object_json() -> None:
    sender, receiver = socket.socketpair()
    try:
        encoded = b"[]"
        sender.sendall(struct.pack(HEADER_LENGTH_FORMAT, len(encoded)) + encoded)
        with pytest.raises(QwenIPCProtocolError, match="JSON object"):
            recv_header(receiver)
    finally:
        sender.close()
        receiver.close()


@pytest.mark.parametrize("n_samples", [-1, MAX_AUDIO_SAMPLES + 1, True, "1"])
def test_audio_payload_rejects_invalid_sample_count(n_samples: object) -> None:
    sender, receiver = socket.socketpair()
    try:
        with pytest.raises(QwenIPCProtocolError, match="n_samples"):
            recv_audio_payload(receiver, {"n_samples": n_samples})
    finally:
        sender.close()
        receiver.close()
