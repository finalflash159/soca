/** Typed, engine-owned source verification state for citation dialogs. */

import type { Citation } from "./conversation";
import type { CitationPreviewFrame, EngineFrame } from "./protocol";

export type CitationPreviewStatus =
  | "idle"
  | "loading"
  | "current"
  | "changed"
  | "unverified"
  | "missing"
  | "unavailable";

export interface CitationPreviewState {
  requestId: string | null;
  status: CitationPreviewStatus;
  title: string | null;
  lineStart: number | null;
  lineEnd: number | null;
  passage: string | null;
  errorCode: string | null;
}

export type CitationPreviewIndex = Record<string, CitationPreviewState>;

export const initialCitationPreviews: CitationPreviewIndex = {};

function string(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function line(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : null;
}

export function citationSource(citation: Citation): string {
  return string(citation.source) ?? "knowledge";
}

/** Stable for one source location; a request always refreshes the current view. */
export function citationKey(citation: Citation): string {
  return [
    string(citation.path) ?? "",
    line(citation.line_start) ?? "",
    line(citation.line_end) ?? "",
    citationSource(citation),
  ].join("\u0000");
}

function keyForFrame(frame: CitationPreviewFrame): string {
  return [frame.path, line(frame.line_start) ?? "", line(frame.line_end) ?? "", frame.source].join("\u0000");
}

export type CitationPreviewAction =
  | EngineFrame
  | { type: "citation_preview_requested"; citation: Citation; requestId: string }
  | { type: "citation_preview_failed"; citation: Citation; requestId: string; message: string };

type CitationPreviewRequested = Extract<CitationPreviewAction, { type: "citation_preview_requested" }>;
type CitationPreviewFailed = Extract<CitationPreviewAction, { type: "citation_preview_failed" }>;

function isPreviewRequested(action: CitationPreviewAction): action is CitationPreviewRequested {
  return "type" in action && action.type === "citation_preview_requested";
}

function isPreviewFailed(action: CitationPreviewAction): action is CitationPreviewFailed {
  return "type" in action && action.type === "citation_preview_failed";
}

export function reduceCitationPreviews(
  state: CitationPreviewIndex,
  action: CitationPreviewAction,
): CitationPreviewIndex {
  if (isPreviewRequested(action)) {
    return {
      ...state,
      [citationKey(action.citation)]: {
        requestId: action.requestId,
        status: "loading",
        title: string(action.citation.title),
        lineStart: line(action.citation.line_start),
        lineEnd: line(action.citation.line_end),
        passage: null,
        errorCode: null,
      },
    };
  }
  if (isPreviewFailed(action)) {
    const previous = state[citationKey(action.citation)];
    if (previous?.requestId !== action.requestId) return state;
    return {
      ...state,
      [citationKey(action.citation)]: {
        requestId: action.requestId,
        status: "unavailable",
        title: string(action.citation.title),
        lineStart: line(action.citation.line_start),
        lineEnd: line(action.citation.line_end),
        passage: null,
        errorCode: action.message,
      },
    };
  }
  if ("event" in action && action.event === "citation_preview") {
    const frame = action as CitationPreviewFrame;
    const key = keyForFrame(frame);
    const previous = state[key];
    if (previous?.requestId !== frame.request_id) return state;
    return {
      ...state,
      [key]: {
        requestId: frame.request_id,
        status: frame.status,
        title: string(frame.title),
        lineStart: line(frame.line_start),
        lineEnd: line(frame.line_end),
        passage: string(frame.passage),
        errorCode: string(frame.error_code),
      },
    };
  }
  return state;
}
