import { describe, expect, it } from "vitest";

import {
  evidenceSummary,
  indexJobRunning,
  initialKnowledge,
  memoryModeSummary,
  reduceKnowledge,
} from "./knowledge";
import type { EngineFrame } from "./protocol";

function fold(frames: EngineFrame[]) {
  return frames.reduce(reduceKnowledge, initialKnowledge);
}

describe("retrieval trace", () => {
  it("keeps columns, fusion and the gate verdict as the engine sent them", () => {
    const state = fold([
      {
        event: "retrieval_trace",
        query: "chuyến Đà Lạt hết bao nhiêu",
        tier: "llm",
        latency_ms: 71.4,
        columns: [{ source: "sparse", hits: [{ path: "a.md", score: 0.6, sparse_score: 0.6 }] }],
        fused: [{ path: "a.md", picked: true, backend: "sparse", score: 0.6 }],
        rejected_count: 2,
        evidence: { source: "knowledge", status: "supported", hit_count: 1, top_score: 0.61 },
      } as EngineFrame,
    ]);
    expect(state.retrieval?.tier).toBe("llm");
    expect(state.retrieval?.columns[0].hits[0].sparse_score).toBe(0.6);
    expect(state.retrieval?.rejectedCount).toBe(2);
    expect(state.retrieval?.evidence?.status).toBe("supported");
  });

  it("tolerates a turn where the gate did not run", () => {
    const state = fold([
      { event: "retrieval_trace", query: "q", tier: "none", evidence: null } as EngineFrame,
    ]);
    expect(state.retrieval?.evidence).toBeNull();
    expect(state.retrieval?.columns).toEqual([]);
  });
});

describe("evidence summary", () => {
  it("distinguishes an unreachable source from an empty vault", () => {
    // docs/18 §4: conflating these is the mistake the wording guards against.
    expect(evidenceSummary({ status: "unavailable" })).toContain("could not be reached");
    expect(evidenceSummary({ status: "insufficient" })).toContain("evidence floor");
  });

  it("says so when the gate never ran", () => {
    expect(evidenceSummary(null)).toContain("did not run");
  });

  it("passes an unknown status through instead of guessing", () => {
    expect(evidenceSummary({ status: "some_new_status" })).toContain("some_new_status");
  });
});

describe("memory", () => {
  it("reads a trace", () => {
    const state = fold([
      {
        event: "memory_trace",
        mode: "archive",
        degraded_reason: "",
        hits: [{ id: "m1", corpus: "episode", relevance: 0.4, recency: 0.2, importance: 0.1, total: 0.7 }],
        hit_count: 1,
        background_status: "running",
        summary_worker_state: "ready",
        recent_turn_count: 6,
        pending_compaction: true,
      } as EngineFrame,
    ]);
    expect(state.memoryTrace?.hits).toHaveLength(1);
    expect(state.memoryTrace?.pendingCompaction).toBe(true);
    expect(memoryModeSummary(state.memoryTrace)).toContain("1 hit.");
  });

  it("reports degradation ahead of the mode", () => {
    const state = fold([
      { event: "memory_trace", mode: "archive", degraded_reason: "index_missing" } as EngineFrame,
    ]);
    expect(memoryModeSummary(state.memoryTrace)).toContain("index_missing");
  });

  it("records a disabled memory snapshot without inventing stats", () => {
    const state = fold([
      { event: "memory", enabled: false, summary: "", recent: "", stats: null } as EngineFrame,
    ]);
    expect(state.memory).toEqual({ enabled: false, summary: "", recent: "", stats: null });
  });
});

describe("proposals", () => {
  const proposal = {
    id: "p1",
    kind: "fact",
    statement: "thích cà phê đen",
    confidence: 0.8,
    createdAt: "2026-08-15T00:00:00Z",
  };

  it("lists what the engine sent", () => {
    const state = fold([{ event: "memory_proposals", proposals: [proposal] } as EngineFrame]);
    expect(state.proposals).toHaveLength(1);
  });

  it("treats an empty inbox as normal", () => {
    // The inbox is always empty in production: MemoryProposal is only
    // constructed in tests and eval. An empty list is not an error state.
    const state = fold([{ event: "memory_proposals", proposals: [] } as EngineFrame]);
    expect(state.proposals).toEqual([]);
  });

  it("removes a row only when the store accepted the action", () => {
    const state = fold([
      { event: "memory_proposals", proposals: [proposal] } as EngineFrame,
      { event: "memory_action", proposal_id: "p1", action: "approved", ok: true, error_code: null } as EngineFrame,
    ]);
    expect(state.proposals).toEqual([]);
    expect(state.lastAction?.ok).toBe(true);
  });

  it("keeps the row and the reason when the action failed", () => {
    const state = fold([
      { event: "memory_proposals", proposals: [proposal] } as EngineFrame,
      {
        event: "memory_action",
        proposal_id: "p1",
        action: "approved",
        ok: false,
        error_code: "memory_unavailable",
      } as EngineFrame,
    ]);
    expect(state.proposals).toHaveLength(1);
    expect(state.lastAction?.errorCode).toBe("memory_unavailable");
  });
});

describe("index job", () => {
  it("is running until it reaches a terminal status", () => {
    const running = fold([
      { event: "knowledge_setup", action: "index", status: "embedding", detail: "…", vault: "/v" } as EngineFrame,
    ]);
    expect(indexJobRunning(running.indexJob)).toBe(true);

    const done = fold([
      { event: "knowledge_setup", action: "index", status: "ok", detail: "done", vault: "/v" } as EngineFrame,
    ]);
    expect(indexJobRunning(done.indexJob)).toBe(false);
  });

  it("does not treat vault init as an index build", () => {
    const state = fold([
      { event: "knowledge_setup", action: "init", status: "running", detail: "…", vault: "/v" } as EngineFrame,
    ]);
    expect(indexJobRunning(state.indexJob)).toBe(false);
  });

  it("carries the error code through on failure", () => {
    const state = fold([
      {
        event: "knowledge_setup",
        action: "init",
        status: "failed",
        detail: "permission denied",
        vault: "/v",
        error_code: "knowledge_init_failed",
      } as EngineFrame,
    ]);
    expect(state.indexJob?.errorCode).toBe("knowledge_init_failed");
  });
});

describe("status frame", () => {
  it("reports index presence from null-ness, not from a flag", () => {
    expect(fold([{ event: "status", knowledge_index: null } as EngineFrame]).indexPresent).toBe(false);
    expect(
      fold([{ event: "status", knowledge_index: { generation: 3 } } as EngineFrame]).indexPresent,
    ).toBe(true);
  });
});

describe("unknown frames", () => {
  it("are ignored", () => {
    expect(fold([{ event: "usage" } as EngineFrame])).toEqual(initialKnowledge);
  });
});
