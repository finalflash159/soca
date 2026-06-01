from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class MetricsLogger:
    def __init__(self) -> None:
        self._latencies_ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._latencies_ms[name] = (time.perf_counter() - t0) * 1000

    def snapshot(self) -> dict[str, float]:
        return dict(self._latencies_ms)

    def reset(self) -> None:
        self._latencies_ms.clear()
