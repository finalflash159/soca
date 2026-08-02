from __future__ import annotations

import json
import socket
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np

HEADER_LENGTH_FORMAT = ">I"
HEADER_LENGTH_SIZE = struct.calcsize(HEADER_LENGTH_FORMAT)
MAX_HEADER_BYTES = 64 * 1024
MAX_AUDIO_SECONDS = 300
SAMPLE_RATE = 16_000
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_SECONDS
MAX_AUDIO_BYTES = MAX_AUDIO_SAMPLES * np.dtype(np.float32).itemsize


class QwenIPCProtocolError(ValueError):
    """A malformed or oversized frame was received."""


def send_frame(
    sock: socket.socket,
    header: Mapping[str, Any],
    payload: bytes = b"",
) -> None:
    try:
        header_bytes = json.dumps(
            dict(header),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise QwenIPCProtocolError(f"Header is not JSON serializable: {exc}") from exc
    if not header_bytes or len(header_bytes) > MAX_HEADER_BYTES:
        raise QwenIPCProtocolError(
            f"Header size must be from 1 to {MAX_HEADER_BYTES} bytes"
        )
    if len(payload) > MAX_AUDIO_BYTES:
        raise QwenIPCProtocolError(
            f"Payload exceeds the {MAX_AUDIO_BYTES}-byte transport limit"
        )
    sock.sendall(struct.pack(HEADER_LENGTH_FORMAT, len(header_bytes)))
    sock.sendall(header_bytes)
    if payload:
        sock.sendall(payload)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    if size < 0:
        raise QwenIPCProtocolError("Frame size cannot be negative")
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while reading frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_header(sock: socket.socket) -> dict[str, Any]:
    raw_length = recv_exact(sock, HEADER_LENGTH_SIZE)
    header_length = struct.unpack(HEADER_LENGTH_FORMAT, raw_length)[0]
    if not 0 < header_length <= MAX_HEADER_BYTES:
        raise QwenIPCProtocolError(
            f"Header size must be from 1 to {MAX_HEADER_BYTES} bytes"
        )
    try:
        decoded = json.loads(recv_exact(sock, header_length).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QwenIPCProtocolError(f"Header is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise QwenIPCProtocolError("Header must be a JSON object")
    return decoded


def recv_audio_payload(
    sock: socket.socket,
    header: Mapping[str, Any],
) -> np.ndarray:
    n_samples = header.get("n_samples")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise QwenIPCProtocolError("n_samples must be an integer")
    if not 0 <= n_samples <= MAX_AUDIO_SAMPLES:
        raise QwenIPCProtocolError(
            f"n_samples must be from 0 to {MAX_AUDIO_SAMPLES}"
        )
    raw = recv_exact(sock, n_samples * np.dtype(np.float32).itemsize)
    return np.frombuffer(raw, dtype="<f4")


__all__ = [
    "HEADER_LENGTH_FORMAT",
    "HEADER_LENGTH_SIZE",
    "MAX_AUDIO_BYTES",
    "MAX_AUDIO_SAMPLES",
    "MAX_HEADER_BYTES",
    "QwenIPCProtocolError",
    "SAMPLE_RATE",
    "recv_audio_payload",
    "recv_exact",
    "recv_header",
    "send_frame",
]
