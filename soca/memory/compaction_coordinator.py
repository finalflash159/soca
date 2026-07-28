"""One coordinator for automatic/manual working-memory compaction."""

from __future__ import annotations

import time
from dataclasses import dataclass

from soca.memory.summary import LocalSummaryWorkerProcess
from soca.memory.working import CompactionJob, WorkingMemory, WorkingSummaryArtifact


@dataclass(frozen=True)
class CompactionResult:
    status: str
    generation: int | None = None
    detail: str = ""
    before_tokens: int | None = None
    after_tokens: int | None = None
    compacted_turns: int = 0
    complete_turns: int = 0
    minimum_complete_turns: int | None = None
    elapsed_ms: float | None = None


class WorkingMemoryCompactionCoordinator:
    def __init__(
        self, memory: WorkingMemory, worker: LocalSummaryWorkerProcess | None = None
    ) -> None:
        self.memory = memory
        self.worker = worker
        self._job: CompactionJob | None = None
        self._before_tokens: int | None = None
        self._complete_turns = 0
        self._minimum_complete_turns: int | None = None
        self._started_at: float | None = None
        self._last_async_result: CompactionResult | None = None
        self.last_telemetry: dict[str, object] | None = None

    def request(self, *, manual: bool = False) -> CompactionResult:
        self._last_async_result = None
        snapshot = self.memory.snapshot
        complete_turns = sum(turn.status == "complete" for turn in snapshot.turns)
        minimum_complete_turns = (
            self.memory.policy.manual_compaction_minimum_complete_turns if manual else None
        )
        job = self.memory.prepare_compaction(force=manual)
        if job is None:
            detail = (
                "compaction_already_running"
                if snapshot.pending_compaction
                else ("not_enough_complete_turns" if manual else "below_compaction_boundary")
            )
            return CompactionResult(
                "noop",
                detail=detail,
                before_tokens=snapshot.token_count,
                complete_turns=complete_turns,
                minimum_complete_turns=minimum_complete_turns,
            )
        self._job = job
        self._before_tokens = snapshot.token_count
        self._complete_turns = complete_turns
        self._minimum_complete_turns = minimum_complete_turns
        self._started_at = time.monotonic()
        if self.worker is None:
            self.memory.cancel_compaction(job.generation)
            result = self._result(
                "trim_only",
                detail="summary_model_not_configured",
                after_tokens=self.memory.snapshot.token_count,
            )
            self._reset_job()
            return result
        if not self.worker.start(job):
            self.memory.cancel_compaction(job.generation)
            result = self._result(
                "unavailable",
                detail="summary_model_not_provisioned",
                after_tokens=self.memory.snapshot.token_count,
            )
            self._reset_job()
            return result
        return self._result("accepted")

    def poll(self) -> CompactionResult:
        if self.worker is None or self._job is None:
            return CompactionResult("idle")
        payload = self.worker.poll()
        if payload is None:
            return self._result("running")
        self.last_telemetry = {
            key: payload[key]
            for key in (
                "latency_ms",
                "load_latency_ms",
                "generation_latency_ms",
                "peak_rss_mb",
                "n_ctx",
                "exit_code",
                "worker_stopped",
            )
            if key in payload
        }
        job = self._job
        if not payload.get("ok"):
            self.memory.cancel_compaction(job.generation)
            result = self._result(
                "failed",
                detail=str(payload.get("error", "worker_failed")),
                after_tokens=self.memory.snapshot.token_count,
            )
            self._last_async_result = result
            self._reset_job()
            return result
        raw = payload.get("artifact")
        if not isinstance(raw, dict):
            self.memory.cancel_compaction(job.generation)
            result = self._result(
                "failed",
                detail="invalid_worker_payload",
                after_tokens=self.memory.snapshot.token_count,
            )
            self._last_async_result = result
            self._reset_job()
            return result
        try:
            artifact = WorkingSummaryArtifact(
                version=int(raw["version"]),
                generation=int(raw["generation"]),
                source_through_sequence=int(raw["source_through_sequence"]),
                summary=str(raw["summary"]),
                user_constraints=tuple(raw.get("user_constraints", ())),
                decisions=tuple(raw.get("decisions", ())),
                corrections=tuple(raw.get("corrections", ())),
                open_items=tuple(raw.get("open_items", ())),
                continuity_refs=tuple(raw.get("continuity_refs", ())),
                prompt_fingerprint=str(raw.get("prompt_fingerprint", "")),
            )
        except (KeyError, TypeError, ValueError):
            self.memory.cancel_compaction(job.generation)
            result = self._result(
                "failed",
                detail="invalid_summary_artifact",
                after_tokens=self.memory.snapshot.token_count,
            )
            self._last_async_result = result
            self._reset_job()
            return result
        if not artifact.render().strip():
            self.memory.cancel_compaction(job.generation)
            result = self._result(
                "failed",
                detail="empty_summary_artifact",
                after_tokens=self.memory.snapshot.token_count,
            )
            self._last_async_result = result
            self._reset_job()
            return result
        published = self.memory.publish_summary(job, artifact)
        result = self._result(
            "published" if published else "stale",
            after_tokens=self.memory.snapshot.token_count,
        )
        self._last_async_result = result
        self._reset_job()
        return result

    def cancel(self) -> CompactionResult:
        if self._job is None:
            return CompactionResult("noop")
        if self.worker is not None:
            self.worker.cancel()
        self.memory.cancel_compaction(self._job.generation)
        result = self._result(
            "cancelled",
            after_tokens=self.memory.snapshot.token_count,
        )
        self._reset_job()
        return result

    def status(self) -> CompactionResult:
        if self._job is None:
            return self._last_async_result or CompactionResult("idle")
        return self.poll()

    def _result(
        self,
        status: str,
        *,
        detail: str = "",
        after_tokens: int | None = None,
    ) -> CompactionResult:
        job = self._job
        elapsed_ms = (
            (time.monotonic() - self._started_at) * 1000 if self._started_at is not None else None
        )
        return CompactionResult(
            status=status,
            generation=job.generation if job is not None else None,
            detail=detail,
            before_tokens=self._before_tokens,
            after_tokens=after_tokens,
            compacted_turns=len(job.frozen_turns) if job is not None else 0,
            complete_turns=self._complete_turns,
            minimum_complete_turns=self._minimum_complete_turns,
            elapsed_ms=elapsed_ms,
        )

    def _reset_job(self) -> None:
        self._job = None
        self._before_tokens = None
        self._complete_turns = 0
        self._minimum_complete_turns = None
        self._started_at = None


__all__ = ["CompactionResult", "WorkingMemoryCompactionCoordinator"]
