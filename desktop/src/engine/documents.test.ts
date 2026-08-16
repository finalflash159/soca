import { describe, expect, it } from "vitest";

import type { Turn } from "./conversation";
import { applyMention, documentIndex, filterCommands, mentionQuery, slashQuery } from "./documents";
import { initialKnowledge, reduceKnowledge } from "./knowledge";
import type { EngineFrame } from "./protocol";

const knowledge = reduceKnowledge(initialKnowledge, {
  event: "retrieval_trace",
  query: "q",
  tier: "llm",
  latency_ms: 10,
  columns: [
    { source: "sparse", hits: [{ path: "wiki/a.md", score: 0.4 }] },
    {
      source: "dense",
      hits: [
        { path: "wiki/a.md", score: 0.9 },
        { path: "wiki/b.md", score: 0.5 },
      ],
    },
  ],
  fused: [],
  rejected_count: 0,
  evidence: null,
} as EngineFrame);

const turn = {
  citations: [{ label: "K1", path: "wiki/a.md", title: "Đà Lạt", line_start: 10, line_end: 20 }],
} as unknown as Turn;

describe("document index", () => {
  it("merges backends and keeps the best score per path", () => {
    const documents = documentIndex(knowledge, []);
    const a = documents.find((doc) => doc.path === "wiki/a.md");
    expect(a?.score).toBe(0.9);
    expect(a?.backends.sort()).toEqual(["dense", "sparse"]);
  });

  it("orders by score", () => {
    expect(documentIndex(knowledge, []).map((doc) => doc.path)).toEqual(["wiki/a.md", "wiki/b.md"]);
  });

  it("takes title and line range from citations", () => {
    const a = documentIndex(knowledge, [turn]).find((doc) => doc.path === "wiki/a.md");
    expect(a?.title).toBe("Đà Lạt");
    expect(a?.lines).toEqual([10, 20]);
  });

  it("is empty before anything has been retrieved", () => {
    expect(documentIndex(initialKnowledge, [])).toEqual([]);
  });
});

describe("@ mentions", () => {
  it("opens at a word boundary", () => {
    expect(mentionQuery("hỏi về @dal", 11)).toEqual({ start: 7, query: "dal" });
  });

  it("opens at the start of the draft", () => {
    expect(mentionQuery("@wi", 3)).toEqual({ start: 0, query: "wi" });
  });

  it("does not open inside a word", () => {
    // An email address must not turn into a document picker.
    expect(mentionQuery("mail me@example.com", 19)).toBeNull();
  });

  it("closes once the token contains whitespace", () => {
    expect(mentionQuery("@wiki done", 10)).toBeNull();
  });

  it("replaces the token and leaves a trailing space", () => {
    const result = applyMention("hỏi về @dal", 11, "wiki/a.md");
    expect(result.text).toBe("hỏi về @wiki/a.md ");
    expect(result.caret).toBe(result.text.length);
  });

  it("preserves text after the caret", () => {
    const result = applyMention("@dal rồi sao", 4, "wiki/a.md");
    expect(result.text).toBe("@wiki/a.md  rồi sao");
  });
});

describe("/ commands", () => {
  it("only opens at the start of the draft", () => {
    expect(slashQuery("/mem")).toBe("mem");
    expect(slashQuery("hỏi /mem")).toBeNull();
  });

  it("closes once a space is typed", () => {
    expect(slashQuery("/memory now")).toBeNull();
  });

  it("filters on label and hint", () => {
    expect(filterCommands("mem").map((command) => command.id)).toContain("memory");
    expect(filterCommands("budget").map((command) => command.id)).toContain("context");
  });

  it("returns everything for an empty query", () => {
    expect(filterCommands("").length).toBeGreaterThan(4);
  });
});
