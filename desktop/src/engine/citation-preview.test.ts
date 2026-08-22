import { describe, expect, it } from "vitest";

import {
  citationKey,
  initialCitationPreviews,
  reduceCitationPreviews,
} from "./citation-preview";
import type { Citation } from "./conversation";
import type { EngineFrame } from "./protocol";

const citation: Citation = {
  label: "K1",
  path: "wiki/plan.md",
  title: "Kế hoạch",
  line_start: 2,
  line_end: 3,
  source: "knowledge",
};

describe("citation preview state", () => {
  it("shows a requested source only after its matching engine receipt", () => {
    const requested = reduceCitationPreviews(initialCitationPreviews, {
      type: "citation_preview_requested",
      citation,
      requestId: "new",
    });
    const stale = reduceCitationPreviews(requested, {
      event: "citation_preview",
      request_id: "old",
      path: "wiki/plan.md",
      source: "knowledge",
      status: "changed",
      title: "Kế hoạch",
      line_start: 2,
      line_end: 3,
      passage: "Bản cũ",
      fingerprint: "old",
      error_code: null,
    } as EngineFrame);
    const settled = reduceCitationPreviews(stale, {
      event: "citation_preview",
      request_id: "new",
      path: "wiki/plan.md",
      source: "knowledge",
      status: "current",
      title: "Kế hoạch",
      line_start: 2,
      line_end: 3,
      passage: "Bản hiện tại",
      fingerprint: "current",
      error_code: null,
    } as EngineFrame);

    expect(stale).toBe(requested);
    expect(settled[citationKey(citation)]).toMatchObject({
      requestId: "new",
      status: "current",
      passage: "Bản hiện tại",
    });
  });

  it("settles a non-knowledge citation under its own source key", () => {
    const memoryCitation: Citation = {
      path: "memory/core.md",
      line_start: 1,
      line_end: 1,
      source: "memory",
    };
    const requested = reduceCitationPreviews(initialCitationPreviews, {
      type: "citation_preview_requested",
      citation: memoryCitation,
      requestId: "memory",
    });
    const settled = reduceCitationPreviews(requested, {
      event: "citation_preview",
      request_id: "memory",
      path: "memory/core.md",
      source: "memory",
      status: "unavailable",
      title: null,
      line_start: 1,
      line_end: 1,
      passage: null,
      fingerprint: null,
      error_code: "citation_source_unavailable",
    } as EngineFrame);

    expect(settled[citationKey(memoryCitation)]).toMatchObject({
      status: "unavailable",
      errorCode: "citation_source_unavailable",
    });
  });
});
