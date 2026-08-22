import { describe, expect, it } from "vitest";

import type { EngineFrame } from "./protocol";
import { initialSessionHistory, reduceSessionHistory } from "./session-history";

const first = {
  session_id: "11111111-1111-4111-8111-111111111111",
  title: "Phiên mới nhất",
  preview: "Tóm tắt",
  updated_at: "2026-08-22T10:00:00Z",
  revision: 4,
  turn_count: 2,
  checkpoint_only: false,
};

const second = {
  session_id: "22222222-2222-4222-8222-222222222222",
  title: "Phiên cũ hơn",
  preview: "",
  updated_at: "2026-08-21T10:00:00Z",
  revision: 1,
  turn_count: 0,
  checkpoint_only: true,
};

describe("session history reducer", () => {
  it("hydrates the bounded saved-session page and finds the active item even when status arrived first", () => {
    let state = reduceSessionHistory(initialSessionHistory, {
      event: "session_status",
      active_session_id: first.session_id,
      persistence: "local_resumable",
      revision: 4,
      busy: false,
    } as EngineFrame);
    state = reduceSessionHistory(state, {
      event: "sessions_page",
      sessions: [first, second],
      next_cursor: "cursor-2",
      persistence: "local_resumable",
    } as EngineFrame);

    expect(state.sessions.map((item) => item.title)).toEqual(["Phiên mới nhất", "Phiên cũ hơn"]);
    expect(state.active?.sessionId).toBe(first.session_id);
    expect(state.active?.checkpointOnly).toBe(false);
    expect(state.nextCursor).toBe("cursor-2");
  });

  it("appends an older page without duplicating the boundary item", () => {
    let state = reduceSessionHistory(initialSessionHistory, {
      type: "sessions_list_requested",
      append: false,
    });
    state = reduceSessionHistory(state, {
      event: "sessions_page",
      sessions: [first],
      next_cursor: "cursor-2",
      persistence: "local_resumable",
    } as EngineFrame);
    state = reduceSessionHistory(state, { type: "sessions_list_requested", append: true });
    state = reduceSessionHistory(state, {
      event: "sessions_page",
      sessions: [first, second],
      next_cursor: null,
      persistence: "local_resumable",
    } as EngineFrame);

    expect(state.sessions.map((item) => item.sessionId)).toEqual([first.session_id, second.session_id]);
  });

  it("keeps the old list visible and exposes a typed storage error", () => {
    let state = reduceSessionHistory(initialSessionHistory, {
      event: "sessions_page",
      sessions: [first],
      next_cursor: null,
      persistence: "local_resumable",
    } as EngineFrame);
    state = reduceSessionHistory(state, {
      event: "engine_error",
      message: "cannot list saved sessions",
      code: "session_list_failed",
    } as EngineFrame);

    expect(state.listState).toBe("error");
    expect(state.listError).toContain("cannot list");
    expect(state.sessions).toHaveLength(1);
  });

  it("keeps the active session and exposes a visible error for a corrupt snapshot", () => {
    let state = reduceSessionHistory(initialSessionHistory, {
      event: "session_snapshot",
      session: first,
      turns: [],
      next_turn_cursor: null,
    } as EngineFrame);
    state = reduceSessionHistory(state, {
      event: "session_snapshot",
      session: second,
      turns: [{ turn_id: "missing-sequence" }],
      next_turn_cursor: null,
    } as EngineFrame);

    expect(state.activeSessionId).toBe(first.session_id);
    expect(state.snapshotError).toContain("không bị thay đổi");
  });
});
