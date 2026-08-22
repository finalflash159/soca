import type {
  EngineErrorFrame,
  EngineFrame,
  SessionOperationFrame,
  SessionPreferencesFrame,
  SessionSnapshotFrame,
  SessionStatusFrame,
  SessionsPageFrame,
} from "./protocol";

export type SessionPersistence = "ram_only" | "local_resumable";
export type SessionLoadState = "idle" | "loading" | "ready" | "error";

export interface SessionSummary {
  sessionId: string;
  title: string;
  preview: string;
  updatedAt: string;
  revision: number;
  turnCount: number;
  checkpointOnly: boolean;
}

export interface SessionOperation {
  requestId: string;
  action: SessionOperationFrame["action"];
  status: SessionOperationFrame["status"];
  sessionId: string | null;
  revision: number | null;
  errorCode: string | null;
}

export interface SessionHistoryState {
  sessions: SessionSummary[];
  nextCursor: string | null;
  listState: SessionLoadState;
  listError: string | null;
  snapshotError: string | null;
  activeSessionId: string | null;
  active: SessionSummary | null;
  persistence: SessionPersistence | null;
  autoOpenLast: boolean;
  busy: boolean;
  operation: SessionOperation | null;
}

export const initialSessionHistory: SessionHistoryState = {
  sessions: [],
  nextCursor: null,
  listState: "idle",
  listError: null,
  snapshotError: null,
  activeSessionId: null,
  active: null,
  persistence: null,
  autoOpenLast: false,
  busy: false,
  operation: null,
};

export type SessionHistoryAction =
  | EngineFrame
  | { type: "sessions_list_requested"; append: boolean }
  | { type: "sessions_list_failed"; message: string }
  | { type: "reset" };

function string(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function number(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function summary(value: unknown): SessionSummary | null {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const record = value as Record<string, unknown>;
  const sessionId = string(record.session_id);
  const title = string(record.title);
  const revision = number(record.revision);
  if (sessionId === null || title === null || revision === null) {
    return null;
  }
  return {
    sessionId,
    title,
    preview: string(record.preview) ?? "",
    updatedAt: string(record.updated_at) ?? "",
    revision,
    turnCount: number(record.turn_count) ?? 0,
    checkpointOnly: record.checkpoint_only === true,
  };
}

function appendWithoutDuplicates(
  current: SessionSummary[],
  incoming: SessionSummary[],
): SessionSummary[] {
  const known = new Set(current.map((item) => item.sessionId));
  return [...current, ...incoming.filter((item) => !known.has(item.sessionId))];
}

function readSessions(frame: SessionsPageFrame): SessionSummary[] {
  return frame.sessions.flatMap((item) => {
    const parsed = summary(item);
    return parsed === null ? [] : [parsed];
  });
}

function readSnapshot(frame: SessionSnapshotFrame): SessionSummary | null {
  return summary(frame.session);
}

function snapshotIsUsable(frame: SessionSnapshotFrame): boolean {
  if (readSnapshot(frame) === null || !Array.isArray(frame.turns)) return false;
  if (
    frame.next_turn_cursor !== null &&
    (!Number.isInteger(frame.next_turn_cursor) || frame.next_turn_cursor < 0)
  ) {
    return false;
  }
  return frame.turns.every((turn) => {
    if (turn === null || typeof turn !== "object" || Array.isArray(turn)) return false;
    const record = turn as Record<string, unknown>;
    return string(record.turn_id) !== null && number(record.sequence) !== null;
  });
}

function sessionFailure(frame: EngineErrorFrame): string | null {
  if (typeof frame.code !== "string" || !frame.code.startsWith("session_")) {
    return null;
  }
  return frame.message;
}

export function sessionOperationMessage(operation: SessionOperation): string {
  const action = {
    create: "tạo phiên",
    open: "mở phiên",
    rename: "đổi tên phiên",
    delete: "xóa phiên",
    preferences_set: "lưu cài đặt phiên",
  }[operation.action];
  if (operation.status === "rejected") {
    return `Không thể ${action} khi có lượt hoặc mic đang hoạt động.`;
  }
  return `Không thể ${action}${operation.errorCode === null ? "." : ` (${operation.errorCode}).`}`;
}

export function reduceSessionHistory(
  state: SessionHistoryState,
  action: SessionHistoryAction,
): SessionHistoryState {
  if ("type" in action && action.type === "reset") {
    return initialSessionHistory;
  }
  if ("type" in action && action.type === "sessions_list_requested") {
    return {
      ...state,
      ...(action.append ? {} : { sessions: [], nextCursor: null }),
      listState: "loading",
      listError: null,
    };
  }
  if ("type" in action && action.type === "sessions_list_failed") {
    return {
      ...state,
      listState: "error",
      listError: (action as { type: "sessions_list_failed"; message: string }).message,
    };
  }

  const frame = action as EngineFrame;
  if (frame.event === "sessions_page") {
    const page = frame as SessionsPageFrame;
    const sessions = readSessions(page);
    const nextSessions =
      state.listState === "loading" && state.sessions.length > 0
        ? appendWithoutDuplicates(state.sessions, sessions)
        : sessions;
    return {
      ...state,
      sessions: nextSessions,
      active:
        state.activeSessionId === null
          ? state.active
          : nextSessions.find((item) => item.sessionId === state.activeSessionId) ?? state.active,
      nextCursor: page.next_cursor,
      persistence: page.persistence,
      listState: "ready",
      listError: null,
    };
  }

  if (frame.event === "session_snapshot" || frame.event === "session_turns_page") {
    const snapshot = frame as SessionSnapshotFrame;
    if (!snapshotIsUsable(snapshot)) {
      return {
        ...state,
        snapshotError: "Không thể đọc dữ liệu phiên đã lưu; nội dung đang hiển thị không bị thay đổi.",
      };
    }
    const active = readSnapshot(snapshot);
    if (active === null) return state;
    return {
      ...state,
      snapshotError: null,
      activeSessionId: active.sessionId,
      active,
      sessions: [active, ...state.sessions.filter((item) => item.sessionId !== active.sessionId)],
    };
  }

  if (frame.event === "session_status") {
    const status = frame as SessionStatusFrame;
    const existing = state.sessions.find((item) => item.sessionId === status.active_session_id) ?? null;
    return {
      ...state,
      activeSessionId: status.active_session_id,
      active:
        existing === null
          ? state.active?.sessionId === status.active_session_id
            ? { ...state.active, revision: status.revision ?? state.active.revision }
            : state.active
          : { ...existing, revision: status.revision ?? existing.revision },
      persistence: status.persistence,
      busy: status.busy,
    };
  }

  if (frame.event === "session_preferences") {
    const preferences = frame as SessionPreferencesFrame;
    return {
      ...state,
      persistence: preferences.persistence,
      autoOpenLast: preferences.auto_open_last,
    };
  }

  if (frame.event === "session_operation") {
    const operation = frame as SessionOperationFrame;
    return {
      ...state,
      operation: {
        requestId: operation.request_id,
        action: operation.action,
        status: operation.status,
        sessionId: operation.session_id,
        revision: operation.revision,
        errorCode: operation.error_code,
      },
      busy:
        operation.status === "rejected" && operation.error_code === "SessionBusyError" ? true : state.busy,
    };
  }

  if (frame.event === "engine_error") {
    const failure = sessionFailure(frame as EngineErrorFrame);
    return failure === null
      ? state
      : { ...state, listState: "error", listError: failure };
  }

  return state;
}
