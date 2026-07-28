"""One coordinator for automatic/manual working-memory compaction."""

from __future__ import annotations

from dataclasses import dataclass

from soca.memory.summary import LocalSummaryWorkerProcess
from soca.memory.working import CompactionJob, WorkingMemory, WorkingSummaryArtifact


@dataclass(frozen=True)
class CompactionResult:
    status: str
    generation: int | None = None
    detail: str = ""


class WorkingMemoryCompactionCoordinator:
    def __init__(self, memory: WorkingMemory, worker: LocalSummaryWorkerProcess | None = None) -> None:
        self.memory = memory
        self.worker = worker
        self._job: CompactionJob | None = None
        self.last_telemetry: dict[str, object] | None = None

    def request(self, *, manual: bool = False) -> CompactionResult:
        job = self.memory.prepare_compaction(force=manual)
        if job is None:
            return CompactionResult("noop", detail="working memory is below its compaction boundary")
        self._job = job
        if self.worker is None:
            self.memory.cancel_compaction(job.generation)
            self._job = None
            return CompactionResult("trim_only", detail="no approved local summary model is configured")
        if not self.worker.start(job):
            self.memory.cancel_compaction(job.generation)
            self._job = None
            return CompactionResult("unavailable", detail="summary model is not provisioned locally")
        return CompactionResult("accepted", generation=job.generation)

    def poll(self) -> CompactionResult:
        if self.worker is None or self._job is None:
            return CompactionResult("idle")
        payload = self.worker.poll()
        if payload is None:
            return CompactionResult("running", generation=self._job.generation)
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
        self._job = None
        if not payload.get("ok"):
            self.memory.cancel_compaction(job.generation)
            return CompactionResult("failed", generation=job.generation, detail=str(payload.get("error", "worker_failed")))
        raw = payload.get("artifact")
        if not isinstance(raw, dict):
            self.memory.cancel_compaction(job.generation)
            return CompactionResult("failed", generation=job.generation, detail="invalid_worker_payload")
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
            return CompactionResult("failed", generation=job.generation, detail="invalid_summary_artifact")
        return CompactionResult(
            "published" if self.memory.publish_summary(job, artifact) else "stale",
            generation=job.generation,
        )

    def cancel(self) -> CompactionResult:
        if self._job is None:
            return CompactionResult("noop")
        generation = self._job.generation
        if self.worker is not None:
            self.worker.cancel()
        self.memory.cancel_compaction(generation)
        self._job = None
        return CompactionResult("cancelled", generation=generation)

    def status(self) -> CompactionResult:
        if self._job is None:
            return CompactionResult("idle")
        return self.poll()


__all__ = ["CompactionResult", "WorkingMemoryCompactionCoordinator"]
