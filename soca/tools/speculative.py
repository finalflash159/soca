from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from hashlib import sha256
from threading import RLock
from typing import Literal

from .base import SideEffectLevel, Tool, ToolCall, ToolResult, ToolRuntime, ToolSpec

SpeculativeStatus = Literal[
    "pending",
    "ready",
    "failed",
    "consumed",
    "invalidated",
    "rejected",
]


@dataclass(frozen=True)
class SpeculativeSnapshot:
    slot_id: str
    status: SpeculativeStatus
    reason: str = ""


@dataclass
class _SpeculativeEntry:
    slot_id: str
    call: ToolCall
    call_key: str
    identity: str
    status: SpeculativeStatus = "pending"
    reason: str = ""
    result: ToolResult | None = None
    future: Future[ToolResult] | None = None
    lock: RLock = field(default_factory=RLock)

    def snapshot(self) -> SpeculativeSnapshot:
        with self.lock:
            return SpeculativeSnapshot(self.slot_id, self.status, self.reason)


class SpeculativeReceipt:
    def __init__(self, entry: _SpeculativeEntry) -> None:
        self._entry = entry

    @property
    def status(self) -> SpeculativeStatus:
        return self._entry.snapshot().status

    @property
    def reason(self) -> str:
        return self._entry.snapshot().reason

    def wait(self, timeout: float | None = None) -> SpeculativeSnapshot:
        future = self._entry.future
        if future is not None:
            try:
                future.result(timeout=timeout)
            except Exception:  # noqa: BLE001 - state callback exposes the typed reason
                pass
        return self._entry.snapshot()


class SpeculativeToolRuntime(ToolRuntime):
    """Read-only prefetch layer that cannot bypass canonical tool verification.

    A prefetched result is reusable only for the exact serialized call and the
    exact source identity observed when it was computed. Consumers must opt in
    with a one-turn slot. The controller still receives a normal ``ToolResult``
    and retains ownership of observation, verification, synthesis, and the
    terminal outcome.
    """

    def __init__(
        self,
        delegate: ToolRuntime,
        *,
        identity_provider: Callable[[str], str],
        workers: int = 1,
    ) -> None:
        if workers < 1:
            raise ValueError("speculative workers must be positive")
        self._delegate = delegate
        self._identity_provider = identity_provider
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="soca-speculative-retrieval",
        )
        self._entries: dict[str, _SpeculativeEntry] = {}
        self._lock = RLock()
        self._active_slot: ContextVar[str] = ContextVar("soca_speculative_slot", default="")
        self._closed = False

    @property
    def max_side_effect(self) -> SideEffectLevel:
        return self._delegate.max_side_effect

    def register(self, tool: Tool) -> None:
        self._delegate.register(tool)

    def list_specs(self, include_disabled: bool = True) -> tuple[ToolSpec, ...]:
        return self._delegate.list_specs(include_disabled=include_disabled)

    def get(self, name: str) -> Tool | None:
        return self._delegate.get(name)

    def prefetch(self, slot_id: str, call: ToolCall) -> SpeculativeReceipt:
        slot = slot_id.strip()
        if not slot:
            return self._rejected_receipt(slot_id, call, "empty_slot_id")
        if self._closed:
            return self._rejected_receipt(slot, call, "runtime_closed")
        tool = self.get(call.name)
        if tool is None:
            return self._rejected_receipt(slot, call, "unknown_tool")
        if not tool.spec.enabled:
            return self._rejected_receipt(slot, call, "tool_disabled")
        if tool.spec.side_effect is not SideEffectLevel.READ_ONLY:
            return self._rejected_receipt(slot, call, "tool_not_read_only")
        try:
            identity = self._identity_provider(call.name).strip()
        except Exception:  # noqa: BLE001 - provider failure is a typed rejection
            return self._rejected_receipt(slot, call, "identity_unavailable")
        if not identity:
            return self._rejected_receipt(slot, call, "identity_unavailable")

        entry = _SpeculativeEntry(
            slot_id=slot,
            call=call,
            call_key=_call_key(call),
            identity=identity,
        )
        with self._lock:
            previous = self._entries.pop(slot, None)
            if previous is not None:
                self._invalidate(previous, "slot_replaced")
            self._entries[slot] = entry
            future = self._executor.submit(self._delegate.call, call)
            entry.future = future
            future.add_done_callback(lambda completed: self._complete(entry, completed))
        return SpeculativeReceipt(entry)

    @contextmanager
    def using_slot(self, slot_id: str) -> Iterator[None]:
        token = self._active_slot.set(slot_id.strip())
        try:
            yield
        finally:
            self._active_slot.reset(token)

    def call(self, call: ToolCall) -> ToolResult:
        slot = self._active_slot.get()
        if not slot:
            return self._delegate.call(call)

        with self._lock:
            entry = self._entries.pop(slot, None)
        if entry is None:
            return self._canonical_with_marker(call, "slot_not_found")
        if entry.call_key != _call_key(call):
            self._invalidate(entry, "call_mismatch")
            return self._canonical_with_marker(call, "call_mismatch")
        try:
            current_identity = self._identity_provider(call.name).strip()
        except Exception:  # noqa: BLE001 - run canonical and expose why cache was refused
            current_identity = ""
        if not current_identity:
            self._invalidate(entry, "identity_unavailable")
            return self._canonical_with_marker(call, "identity_unavailable")
        if current_identity != entry.identity:
            self._invalidate(entry, "identity_changed")
            return self._canonical_with_marker(call, "identity_changed")

        with entry.lock:
            if entry.status == "ready" and entry.result is not None:
                entry.status = "consumed"
                result = entry.result
                return _with_marker(result, status="hit", reason="exact_identity_match")
            reason = "prefetch_failed" if entry.status == "failed" else "prefetch_pending"
            entry.status = "invalidated"
            entry.reason = reason
        return self._canonical_with_marker(call, reason)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            entries = tuple(self._entries.values())
            self._entries.clear()
        for entry in entries:
            self._invalidate(entry, "runtime_closed")
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _canonical_with_marker(self, call: ToolCall, reason: str) -> ToolResult:
        return _with_marker(self._delegate.call(call), status="miss", reason=reason)

    def _complete(self, entry: _SpeculativeEntry, future: Future[ToolResult]) -> None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - preserve a bounded typed failure
            with entry.lock:
                if entry.status == "pending":
                    entry.status = "failed"
                    entry.reason = f"executor_error:{type(exc).__name__}"
            return
        with entry.lock:
            if entry.status != "pending":
                return
            entry.result = result
            if result.ok:
                entry.status = "ready"
            else:
                entry.status = "failed"
                entry.reason = result.error or str(result.status)

    @staticmethod
    def _invalidate(entry: _SpeculativeEntry, reason: str) -> None:
        future = entry.future
        if future is not None and not future.done():
            future.cancel()
        with entry.lock:
            if entry.status not in {"consumed", "rejected"}:
                entry.status = "invalidated"
                entry.reason = reason

    @staticmethod
    def _rejected_receipt(
        slot_id: str,
        call: ToolCall,
        reason: str,
    ) -> SpeculativeReceipt:
        entry = _SpeculativeEntry(
            slot_id=slot_id,
            call=call,
            call_key=_call_key(call),
            identity="",
            status="rejected",
            reason=reason,
        )
        return SpeculativeReceipt(entry)


def _call_key(call: ToolCall) -> str:
    return json.dumps(
        {"name": call.name, "arguments": call.arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def knowledge_source_identity(source: object) -> str:
    """Return a stable identity for the currently active knowledge generation."""
    status = getattr(source, "index_status", None)
    if status is not None:
        as_dict = getattr(status, "as_dict", None)
        if callable(as_dict):
            payload = as_dict()
        else:
            payload = {
                name: getattr(status, name, None)
                for name in (
                    "corpus_id",
                    "revision",
                    "content_digest",
                    "dense_generation",
                    "embedding_fingerprint",
                )
            }
    else:
        snapshot_provider = getattr(source, "catalog_index_snapshot", None)
        if not callable(snapshot_provider):
            raise RuntimeError("knowledge source does not expose an index identity")
        snapshot = snapshot_provider()
        index = getattr(snapshot, "index", None)
        payload = {
            "revision": getattr(snapshot, "revision", None),
            "content_digest": getattr(index, "content_digest", None),
        }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _with_marker(
    result: ToolResult,
    *,
    status: Literal["hit", "miss"],
    reason: str,
) -> ToolResult:
    return replace(
        result,
        data={
            **result.data,
            "speculative_retrieval": {"status": status, "reason": reason},
        },
    )
