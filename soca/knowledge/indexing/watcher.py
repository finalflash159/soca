from __future__ import annotations

from collections.abc import Callable
from threading import Event, Thread

from soca.knowledge.indexing.coordinator import IndexCoordinator
from soca.knowledge.indexing.status import IndexStatus


class IndexWatcher:
    """Portable polling watcher; filesystem events remain an optimization."""

    def __init__(
        self,
        coordinator: IndexCoordinator,
        *,
        interval_seconds: float = 2.0,
        on_status: Callable[[IndexStatus], None] | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("watch interval must be positive")
        self.coordinator = coordinator
        self.interval_seconds = interval_seconds
        self.on_status = on_status
        self._stop = Event()
        self._thread: Thread | None = None
        self.last_error: Exception | None = None

    def reconcile(self) -> IndexStatus:
        self.coordinator.request_sync("watcher")
        status = self.coordinator.status()
        self.last_error = None
        if self.on_status is not None:
            self.on_status(status)
        return status

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="soca-index-watcher", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.reconcile()
            except (OSError, RuntimeError, ValueError) as exc:
                self.last_error = exc
