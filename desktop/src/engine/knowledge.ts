/**
 * Retrieval, memory and vault state.
 *
 * The retrieval inspector exists for a specific reason recorded in the plan:
 * groundedness is the repository's top open blocker, and a UI that makes the
 * evidence behind an answer cheap to check is the second line of defence. That
 * only works if the evidence is shown as the engine decided it — this module
 * never re-scores, re-ranks or re-judges anything (`docs/18` §7 obligation 6).
 */

import type { EngineFrame } from "./protocol";

/** One hit inside a per-backend column, before fusion. */
export interface RetrievalHit {
  path: string;
  score: number;
  sparse_score?: number;
  dense_score?: number;
  fusion_score?: number;
}

export interface RetrievalColumn {
  source: string;
  hits: RetrievalHit[];
}

export interface FusedHit {
  path: string;
  picked: boolean;
  backend: string;
  score: number;
}

/** `EvidenceDecision.as_dict()` — the gate's own verdict, not a client opinion. */
export interface EvidenceDecision {
  source?: string;
  status?: string;
  hit_count?: number;
  top_score?: number | null;
  margin?: number | null;
  rejected_count?: number;
  reason?: string;
  source_state?: string;
  query_coverage?: number | null;
  score_separation?: number | null;
  sparse_top_score?: number | null;
  dense_top_score?: number | null;
  [key: string]: unknown;
}

export interface RetrievalTrace {
  query: string;
  tier: string;
  latencyMs: number;
  columns: RetrievalColumn[];
  fused: FusedHit[];
  rejectedCount: number;
  evidence: EvidenceDecision | null;
}

export interface MemoryHit {
  id: string;
  corpus: string;
  relevance: number;
  recency: number;
  importance: number;
  total: number;
}

export interface MemoryTrace {
  mode: string;
  degradedReason: string;
  hits: MemoryHit[];
  hitCount: number;
  backgroundStatus: string;
  summaryWorkerState: string;
  recentTurnCount: number | null;
  compactedTurnCount: number | null;
  pendingCompaction: boolean;
}

export interface MemorySnapshot {
  enabled: boolean;
  summary: string;
  recent: string;
  stats: Record<string, unknown> | null;
}

export interface MemoryProposal {
  id: string;
  kind: string;
  statement: string;
  confidence: number;
  createdAt: string;
}

export interface IndexJob {
  action: string;
  status: string;
  detail: string;
  vault: string;
  errorCode: string | null;
  /**
   * Build progress, which the engine reports on every step and this used to
   * discard.
   *
   * `_cmd_knowledge_index` emits a `knowledge_setup` frame per phase carrying
   * `phase`, `completed_chunks`, `total_chunks`, `reused_chunks`,
   * `embedded_chunks` and `documents`. Keeping only `detail` left the panel
   * showing one unchanging line for the whole build, which is indistinguishable
   * from a hang.
   */
  phase: string | null;
  completedChunks: number | null;
  totalChunks: number | null;
  reusedChunks: number | null;
  embeddedChunks: number | null;
}

/** The vault, as `status` reports it. */
export interface VaultInfo {
  path: string;
  /** False before `knowledge_init` creates the `wiki/` scaffold. */
  initialized: boolean;
}

/** A built index, as `status` reports it. */
export interface IndexInfo {
  documents: number;
  chunks: number;
  sparseState: string;
  denseState: string;
  revision: number | null;
}

export interface KnowledgeState {
  /** Newest trace only. A per-turn history is a phase-4+ concern. */
  retrieval: RetrievalTrace | null;
  memoryTrace: MemoryTrace | null;
  memory: MemorySnapshot | null;
  proposals: MemoryProposal[];
  /** Result of the last approve/reject, so the UI can confirm or explain. */
  lastAction: {
    proposalId: string;
    action: string;
    ok: boolean;
    errorCode: string | null;
  } | null;
  indexJob: IndexJob | null;
  compaction: { status: string; detail: string | null } | null;
  vault: VaultInfo | null;
  index: IndexInfo | null;
  /** Null until a `status` frame arrives; false means no index exists yet. */
  indexPresent: boolean | null;
}

export const initialKnowledge: KnowledgeState = {
  retrieval: null,
  memoryTrace: null,
  memory: null,
  proposals: [],
  lastAction: null,
  indexJob: null,
  compaction: null,
  vault: null,
  index: null,
  indexPresent: null,
};

function num(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function str(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

const TERMINAL_SETUP_STATUS = new Set(["ok", "failed", "ready"]);

export function indexJobRunning(job: IndexJob | null): boolean {
  return job !== null && job.action === "index" && !TERMINAL_SETUP_STATUS.has(job.status);
}

/** A model download and an index build share the same exclusive engine worker. */
export function knowledgeSetupRunning(job: IndexJob | null): boolean {
  return (
    job !== null &&
    (job.action === "model" || job.action === "index") &&
    !TERMINAL_SETUP_STATUS.has(job.status)
  );
}

export function reduceKnowledge(state: KnowledgeState, frame: EngineFrame): KnowledgeState {
  switch (frame.event) {
    case "retrieval_trace":
      return {
        ...state,
        retrieval: {
          query: str(frame.query),
          tier: str(frame.tier, "none"),
          latencyMs: num(frame.latency_ms),
          columns: Array.isArray(frame.columns) ? (frame.columns as RetrievalColumn[]) : [],
          fused: Array.isArray(frame.fused) ? (frame.fused as FusedHit[]) : [],
          rejectedCount: num(frame.rejected_count),
          evidence:
            frame.evidence !== null && typeof frame.evidence === "object"
              ? (frame.evidence as EvidenceDecision)
              : null,
        },
      };

    case "memory_trace":
      return {
        ...state,
        memoryTrace: {
          mode: str(frame.mode, "none"),
          degradedReason: str(frame.degraded_reason),
          hits: Array.isArray(frame.hits) ? (frame.hits as MemoryHit[]) : [],
          hitCount: num(frame.hit_count),
          backgroundStatus: str(frame.background_status, "idle"),
          summaryWorkerState: str(frame.summary_worker_state, "disabled"),
          recentTurnCount:
            typeof frame.recent_turn_count === "number" ? frame.recent_turn_count : null,
          compactedTurnCount:
            typeof frame.compacted_turn_count === "number" ? frame.compacted_turn_count : null,
          pendingCompaction: frame.pending_compaction === true,
        },
      };

    case "memory":
      return {
        ...state,
        memory: {
          enabled: frame.enabled === true,
          summary: str(frame.summary),
          recent: str(frame.recent),
          stats:
            frame.stats !== null && typeof frame.stats === "object"
              ? (frame.stats as Record<string, unknown>)
              : null,
        },
      };

    case "memory_proposals":
      return {
        ...state,
        proposals: Array.isArray(frame.proposals) ? (frame.proposals as MemoryProposal[]) : [],
      };

    case "memory_action": {
      const proposalId = str(frame.proposal_id);
      const ok = frame.ok === true;
      return {
        ...state,
        lastAction: {
          proposalId,
          action: str(frame.action),
          ok,
          errorCode: typeof frame.error_code === "string" ? frame.error_code : null,
        },
        // Only drop the row when the store actually accepted it.
        proposals: ok
          ? state.proposals.filter((proposal) => proposal.id !== proposalId)
          : state.proposals,
      };
    }

    case "memory_compaction":
      return {
        ...state,
        compaction: {
          status: str(frame.status, "idle"),
          detail: typeof frame.detail === "string" ? frame.detail : null,
        },
      };

    case "knowledge_setup": {
      const optionalNum = (value: unknown): number | null =>
        typeof value === "number" && Number.isFinite(value) ? value : null;
      return {
        ...state,
        indexJob: {
          action: str(frame.action),
          status: str(frame.status),
          detail: str(frame.detail),
          vault: str(frame.vault),
          errorCode: typeof frame.error_code === "string" ? frame.error_code : null,
          phase: typeof frame.phase === "string" ? frame.phase : null,
          completedChunks: optionalNum(frame.completed_chunks),
          totalChunks: optionalNum(frame.total_chunks),
          reusedChunks: optionalNum(frame.reused_chunks),
          embeddedChunks: optionalNum(frame.embedded_chunks),
        },
        // `knowledge_setup.vault` is a plain path string, unlike `status`.
        vault:
          str(frame.vault) !== ""
            ? {
                path: str(frame.vault),
                initialized: state.vault?.initialized ?? false,
              }
            : state.vault,
      };
    }

    case "status": {
      // `knowledge_vault` is an object — `{path, initialized, index_home}` —
      // not a string. Testing `typeof … === "string"` never matched, so the
      // vault path was dropped on every status frame and the panel claimed it
      // had none while the engine was reporting one.
      const vaultFrame = frame.knowledge_vault;
      const vault =
        vaultFrame !== null && typeof vaultFrame === "object"
          ? {
              path: str((vaultFrame as Record<string, unknown>).path),
              initialized: (vaultFrame as Record<string, unknown>).initialized === true,
            }
          : state.vault;

      const indexFrame = frame.knowledge_index;
      const hasIndex = indexFrame !== null && typeof indexFrame === "object";
      const record = indexFrame as Record<string, unknown> | null;
      return {
        ...state,
        vault,
        indexPresent: hasIndex,
        index: hasIndex
          ? {
              documents: num(record?.documents),
              chunks: num(record?.chunks),
              sparseState: str(record?.sparse_state, "unknown"),
              denseState: str(record?.dense_state, "unknown"),
              revision: typeof record?.revision === "number" ? (record.revision as number) : null,
            }
          : null,
      };
    }

    default:
      return state;
  }
}

/**
 * Plain-language reading of an evidence decision.
 *
 * The status alone is a term of art; this is what it means for the answer the
 * user is looking at. Kept as a pure mapping so it cannot drift into judging
 * evidence itself.
 */
export function evidenceSummary(decision: EvidenceDecision | null): string {
  if (decision === null) {
    return "The evidence gate did not run for this turn.";
  }
  switch (decision.status) {
    case "supported":
      return "Retrieved passages cleared the evidence floor.";
    case "weak":
      return "Passages were retrieved but fell below the evidence floor.";
    case "insufficient":
      return "Nothing retrieved met the evidence floor.";
    case "unavailable":
      return "The knowledge source could not be reached — this is not the same as having no notes.";
    case "not_requested":
      return "This turn did not ask for knowledge.";
    default:
      return `Evidence status: ${decision.status ?? "unknown"}.`;
  }
}

/** Memory mode, in the same register. */
export function memoryModeSummary(trace: MemoryTrace | null): string {
  if (trace === null) {
    return "No memory activity yet.";
  }
  if (trace.degradedReason !== "") {
    return `Memory degraded: ${trace.degradedReason}`;
  }
  switch (trace.mode) {
    case "none":
      return "Memory was not consulted for this turn.";
    case "working":
      return "Working memory only.";
    case "archive":
      return `Archive searched · ${trace.hitCount} hit${trace.hitCount === 1 ? "" : "s"}.`;
    default:
      return `Memory mode: ${trace.mode}.`;
  }
}
