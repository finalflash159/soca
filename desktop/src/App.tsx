import { BookOpen } from "lucide-react";
import { nanoid } from "nanoid";
import { useEffect, useRef, useState } from "react";

import { ChatPage } from "@/components/ChatPage";
import { KnowledgePanel } from "@/components/KnowledgePanel";
import { EmptyState, PageBody, PageHeader } from "@/components/Page";
import { SessionPanel } from "@/components/SessionPanel";
import { SettingsPanel } from "@/components/SettingsPanel";
import type { PageId } from "@/components/Sidebar";
import { Sidebar } from "@/components/Sidebar";
import { StartupView } from "@/components/StartupView";
import { TopBar } from "@/components/TopBar";
import { VoiceMode } from "@/components/VoiceMode";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { documentIndex } from "@/engine/documents";
import {
  launchOptions,
  saveSessionPersistence,
  savedSessionPersistence,
  type LaunchSessionPersistence,
} from "@/engine/launch";
import { sessionOperationMessage, type SessionSummary } from "@/engine/session-history";
import { useEngine } from "@/engine/useEngine";
import { useTheme } from "@/theme";

type PersistenceChange = "enable" | "disable";

function actionLabel(action: string): string {
  return {
    create: "Đã tạo cuộc trò chuyện mới.",
    open: "Đã mở phiên đã lưu.",
    rename: "Đã đổi tên phiên.",
    delete: "Đã xóa vĩnh viễn phiên.",
    preferences_set: "Đã lưu cài đặt phiên.",
  }[action] ?? "Đã cập nhật phiên.";
}

function focusComposer(): void {
  requestAnimationFrame(() => document.getElementById("chat-composer")?.focus());
}

export default function App() {
  const engine = useEngine();
  const theme = useTheme();
  const [page, setPage] = useState<PageId>("chat");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [transcriptOpen, setTranscriptOpen] = useState(true);
  const [launchPersistence, setLaunchPersistence] = useState<LaunchSessionPersistence>(
    savedSessionPersistence,
  );
  const [persistenceChange, setPersistenceChange] = useState<PersistenceChange | null>(null);
  const [persistenceChangePending, setPersistenceChangePending] = useState(false);
  const [pendingNewAfterVoiceStop, setPendingNewAfterVoiceStop] = useState(false);
  const [sessionAlert, setSessionAlert] = useState<string | null>(null);
  const [sessionNotice, setSessionNotice] = useState<string | null>(null);
  const [sessionTransition, setSessionTransition] = useState<string | null>(null);
  const autoStarted = useRef(false);
  const loadedHello = useRef<object | null>(null);
  const handledOperation = useRef<string | null>(null);
  const snapshotOperationRequests = useRef(new Set<string>());

  const protocolReady = engine.hello !== null && engine.versionMismatch === null;
  const connected = engine.status.state === "running" && protocolReady;
  const starting = engine.status.state === "starting" || (engine.status.state === "running" && !protocolReady);
  const documents = documentIndex(engine.knowledge, engine.conversation.turns);
  const voiceRunning = engine.voice.phase !== "off";
  const operation = engine.sessionHistory.operation;
  const operationInFlight = operation?.status === "started";
  const sessionChanging = operationInFlight || sessionTransition !== null;
  const activeTurn = engine.conversation.turns.some(
    (turn) => turn.finalText === null && turn.error === null,
  );
  const sessionBusy = activeTurn || voiceRunning || engine.sessionHistory.busy || sessionChanging;
  const displayedSessionAlert = sessionAlert ?? engine.sessionHistory.snapshotError;
  const canLoadOlderTurns = connected && !sessionChanging && !engine.sessionHistory.busy;

  const startWithPersistence = async (
    persistence: LaunchSessionPersistence,
    program?: string,
  ): Promise<boolean> =>
    engine.start({
      ...launchOptions(persistence),
      ...(program === undefined ? {} : { program }),
    });

  const restartEngine = async (program?: string) => {
    if (engine.status.state === "running" || engine.status.state === "starting") {
      const stopped = await engine.stop();
      if (!stopped) return;
    }
    await startWithPersistence(launchPersistence, program);
  };

  // List, status and preferences are read only after hello establishes the v3
  // contract. The WebView therefore never sends a normal command to a client it
  // has not verified as compatible.
  useEffect(() => {
    if (!connected || engine.hello === null || loadedHello.current === engine.hello) return;
    loadedHello.current = engine.hello;
    void engine.requestSessions();
    void engine.send({ cmd: "session_status" });
    void engine.send({ cmd: "session_preferences_get" });
  }, [connected, engine]);

  useEffect(() => {
    if (!engine.ready || autoStarted.current) return;
    autoStarted.current = true;
    void startWithPersistence(launchPersistence);
  }, [engine.ready, launchPersistence]);

  useEffect(() => {
    const compact = window.matchMedia("(max-width: 760px)");
    const closeWhenCompact = () => {
      if (compact.matches) setSidebarOpen(false);
    };
    closeWhenCompact();
    compact.addEventListener("change", closeWhenCompact);
    return () => compact.removeEventListener("change", closeWhenCompact);
  }, []);

  useEffect(() => {
    if (!pendingNewAfterVoiceStop || engine.voice.phase !== "off" || !connected) return;
    setPendingNewAfterVoiceStop(false);
    const requestId = nanoid();
    snapshotOperationRequests.current.add(requestId);
    void engine.send({ cmd: "session_create", request_id: requestId });
  }, [connected, engine.voice.phase, pendingNewAfterVoiceStop, engine]);

  useEffect(() => {
    if (operation === null) return;
    if (
      operation.status === "started" &&
      snapshotOperationRequests.current.has(operation.requestId)
    ) {
      setSessionTransition(operation.requestId);
      return;
    }
    if (operation.status === "failed" || operation.status === "rejected") {
      snapshotOperationRequests.current.delete(operation.requestId);
      setSessionTransition(null);
    }
  }, [operation]);

  useEffect(() => {
    if (operation === null || operation.status !== "completed") return;
    // RAM-only creates have no snapshot by contract; the conversation reducer
    // clears only after this completed receipt.
    if (operation.action === "create" && engine.sessionHistory.persistence === "ram_only") {
      snapshotOperationRequests.current.delete(operation.requestId);
      setSessionTransition(null);
      focusComposer();
      return;
    }
    if (
      sessionTransition === operation.requestId &&
      operation.sessionId !== null &&
      engine.conversation.activeSessionId === operation.sessionId
    ) {
      snapshotOperationRequests.current.delete(operation.requestId);
      setSessionTransition(null);
      focusComposer();
    }
  }, [engine.conversation.activeSessionId, engine.sessionHistory.persistence, operation, sessionTransition]);

  useEffect(() => {
    if (engine.sessionHistory.snapshotError === null) return;
    if (sessionTransition !== null) snapshotOperationRequests.current.delete(sessionTransition);
    setSessionTransition(null);
  }, [engine.sessionHistory.snapshotError, sessionTransition]);

  useEffect(() => {
    if (operation === null || operation.status === "started") return;
    const key = `${operation.requestId}:${operation.status}`;
    if (handledOperation.current === key) return;
    handledOperation.current = key;
    if (operation.status === "completed") {
      setSessionAlert(null);
      setSessionNotice(actionLabel(operation.action));
      if (["create", "open", "delete"].includes(operation.action)) {
        setPage("chat");
        if (window.matchMedia("(max-width: 760px)").matches) setSidebarOpen(false);
      }
      if (engine.sessionHistory.persistence === "local_resumable") void engine.requestSessions();
      return;
    }
    setSessionNotice(null);
    setSessionAlert(sessionOperationMessage(operation));
  }, [engine, operation]);

  const startupProblem =
    engine.versionMismatch ??
    (engine.status.state === "failed"
      ? engine.status.message
      : engine.status.state === "stopped" && !engine.status.graceful
        ? "Engine đã thoát mà không gửi `bye`. Kiểm tra log ở terminal."
        : null);

  if (engine.versionMismatch !== null || engine.status.state === "failed" || engine.status.state === "stopped") {
    return (
      <main className="h-screen">
        <StartupView
          starting={false}
          problem={startupProblem}
          onStart={(program) => void restartEngine(program)}
        />
      </main>
    );
  }

  const openPage = (next: PageId) => {
    setPage(next);
    if (!connected) return;
    if (next === "session") {
      for (const cmd of ["status", "context", "usage"] as const) void engine.send({ cmd });
      void engine.send({ cmd: "session_status" });
    }
    if (next === "knowledge") {
      for (const cmd of ["memory", "memory_proposals", "status"] as const) void engine.send({ cmd });
    }
    if (next === "settings") {
      for (const cmd of ["llm_providers", "llm_config", "status", "session_preferences_get"] as const) {
        void engine.send({ cmd });
      }
    }
    if (next === "voice" && !sessionChanging && engine.voice.phase === "off") {
      void engine.send({ cmd: "voice_start" });
    }
  };

  const closeCompactSidebar = () => {
    if (window.matchMedia("(max-width: 760px)").matches) setSidebarOpen(false);
  };

  const leaveVoice = (next: PageId) => {
    if (page === "voice" && next !== "voice" && engine.voice.phase !== "off") {
      void engine.send({ cmd: "voice_stop" });
    }
    openPage(next);
  };

  const createSession = () => {
    if (!connected) return;
    setSessionAlert(null);
    setSessionNotice(null);
    if (activeTurn || engine.sessionHistory.busy || sessionChanging) {
      setSessionAlert("Hãy chờ lượt đang chạy hoàn tất trước khi tạo cuộc trò chuyện mới.");
      return;
    }
    if (engine.voice.phase !== "off") {
      setPendingNewAfterVoiceStop(true);
      setSessionNotice("Đang dừng mic trước khi tạo cuộc trò chuyện mới…");
      void engine.send({ cmd: "voice_stop" });
      return;
    }
    const requestId = nanoid();
    snapshotOperationRequests.current.add(requestId);
    void engine.send({ cmd: "session_create", request_id: requestId });
  };

  const openSession = (session: SessionSummary) => {
    if (!connected || session.sessionId === engine.sessionHistory.activeSessionId) return;
    setSessionAlert(null);
    setSessionNotice(null);
    const requestId = nanoid();
    snapshotOperationRequests.current.add(requestId);
    void engine.send({ cmd: "session_open", request_id: requestId, session_id: session.sessionId });
  };

  const renameSession = (session: SessionSummary, title: string) => {
    setSessionAlert(null);
    void engine.send({
      cmd: "session_rename",
      request_id: nanoid(),
      session_id: session.sessionId,
      title,
      expected_revision: session.revision,
    });
  };

  const deleteSession = (session: SessionSummary) => {
    setSessionAlert(null);
    const requestId = nanoid();
    if (session.sessionId === engine.sessionHistory.activeSessionId) {
      snapshotOperationRequests.current.add(requestId);
    }
    void engine.send({
      cmd: "session_delete",
      request_id: requestId,
      session_id: session.sessionId,
      expected_revision: session.revision,
    });
  };

  const requestPersistenceChange = (enabled: boolean) => {
    setPersistenceChange(enabled ? "enable" : "disable");
  };

  const applyPersistenceChange = async () => {
    if (persistenceChange === null) return;
    const next: LaunchSessionPersistence = persistenceChange === "enable" ? "local_resumable" : "ram_only";
    const previous = launchPersistence;
    setPersistenceChangePending(true);
    setPersistenceChange(null);
    const stopped = await engine.stop();
    if (!stopped) {
      setSessionAlert("Không thể khởi động lại để đổi chế độ lưu phiên. Chế độ hiện tại không thay đổi.");
      setPersistenceChangePending(false);
      return;
    }
    if (!saveSessionPersistence(next)) {
      setSessionAlert("Không thể lưu lựa chọn riêng tư của bạn trên máy. Sơn Ca đã khởi động lại với chế độ trước đó.");
      await startWithPersistence(previous);
      setPersistenceChangePending(false);
      return;
    }
    setLaunchPersistence(next);
    const started = await startWithPersistence(next);
    if (!started) {
      setSessionAlert("Không thể khởi động lại engine với chế độ lưu phiên mới.");
    }
    setPersistenceChangePending(false);
  };

  const toggleMic = () => {
    if (!sessionChanging) {
      void engine.send(engine.voice.phase === "off" ? { cmd: "voice_start" } : { cmd: "voice_stop" });
    }
  };

  return (
    <main className="bg-background flex h-screen w-screen overflow-hidden">
      <a
        href="#main-content"
        className="bg-background text-foreground fixed top-2 left-2 z-[60] -translate-y-16 rounded-md border px-3 py-2 text-sm shadow-sm transition-transform focus:translate-y-0"
      >
        Bỏ qua thanh điều hướng
      </a>
      {sidebarOpen && (
        <>
          <button
            type="button"
            aria-label="Đóng thanh điều hướng"
            className="fixed inset-0 z-40 hidden bg-black/20 max-[760px]:block"
            onClick={() => setSidebarOpen(false)}
          />
          <Sidebar
            page={page}
            onNavigate={(next) => {
              closeCompactSidebar();
              leaveVoice(next);
            }}
            onNewConversation={createSession}
            sessions={engine.sessionHistory}
            connected={connected}
            starting={starting}
            voiceRunning={voiceRunning}
            sessionBusy={sessionBusy}
            newConversationDisabled={!connected || activeTurn || sessionChanging}
            onRestartEngine={() => void restartEngine()}
            onCollapse={() => {
              setSidebarOpen(false);
              requestAnimationFrame(() => document.getElementById("open-sidebar")?.focus());
            }}
            onOpenSession={(session) => {
              closeCompactSidebar();
              openSession(session);
            }}
            onRenameSession={renameSession}
            onDeleteSession={deleteSession}
            onLoadMoreSessions={() => {
              const cursor = engine.sessionHistory.nextCursor;
              if (cursor === null) void engine.requestSessions();
              else void engine.requestSessions(cursor);
            }}
            onOpenSessionSettings={() => openPage("settings")}
          />
        </>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar
          orbState={engine.orbState}
          sidebarOpen={sidebarOpen}
          onOpenSidebar={() => setSidebarOpen(true)}
          onToggleTheme={theme.toggle}
        />

        <div id="main-content" tabIndex={-1} className="flex min-h-0 flex-1 flex-col overflow-hidden">
          {sessionNotice !== null && <p className="sr-only" role="status">{sessionNotice}</p>}
          {displayedSessionAlert !== null && (
            <Alert variant="destructive" className="mx-6 mt-4 w-auto" role="alert">
              <AlertTitle>Không thể cập nhật phiên</AlertTitle>
              <AlertDescription>{displayedSessionAlert}</AlertDescription>
            </Alert>
          )}
          {engine.errors.length > 0 && (
            <Alert variant="destructive" className="mx-6 mt-4 w-auto">
              <AlertTitle>Engine báo lỗi</AlertTitle>
              <AlertDescription>
                <ul className="list-disc pl-4">
                  {engine.errors.slice(-3).map((error, index) => <li key={index}>{error}</li>)}
                </ul>
              </AlertDescription>
            </Alert>
          )}

          {page === "chat" && (
            <ChatPage
              orbState={engine.orbState}
              conversation={engine.conversation}
              documents={documents}
              model={engine.settings.config?.model ?? null}
              connected={connected && !sessionChanging}
              starting={starting}
              onSend={(text) => void engine.send({ cmd: "chat", text })}
              onCommand={(command) => {
                if (command.id === "memory_compact") {
                  void engine.send({ cmd: "memory_compact", action: "request" });
                  return;
                }
                void engine.send({ cmd: command.id } as never);
              }}
              onEnterVoiceMode={() => openPage("voice")}
              onOpenSettings={() => openPage("settings")}
              onLoadOlder={() => void engine.requestOlderTurns()}
              canLoadOlder={canLoadOlderTurns}
            />
          )}

          {page === "voice" && (
            <VoiceMode
              orbState={engine.orbState}
              voice={engine.voice}
              conversation={engine.conversation}
              documents={documents}
              connected={connected && !sessionChanging}
              transcriptOpen={transcriptOpen}
              onToggleTranscript={() => setTranscriptOpen((open) => !open)}
              onToggleMic={toggleMic}
              onLeave={() => leaveVoice("chat")}
              onLoadOlder={() => void engine.requestOlderTurns()}
              canLoadOlder={canLoadOlderTurns}
            />
          )}

          {page === "knowledge" && (
            <div className="min-h-0 flex-1 overflow-auto">
              <PageBody wide>
                <PageHeader title="Kiến thức" description="Vault, chỉ mục truy xuất và bộ nhớ phiên." />
                <KnowledgePanel
                  knowledge={engine.knowledge}
                  connected={connected && !sessionChanging}
                  onInit={() => void engine.send({ cmd: "knowledge_init" })}
                  onIndex={() => void engine.send({ cmd: "knowledge_index" })}
                  onRefreshMemory={() => {
                    void engine.send({ cmd: "memory" });
                    void engine.send({ cmd: "memory_proposals" });
                  }}
                  onCompact={() => void engine.send({ cmd: "memory_compact", action: "request" })}
                  onApprove={(id) => void engine.send({ cmd: "memory_approve", proposal_id: id })}
                  onReject={(id) => void engine.send({ cmd: "memory_reject", proposal_id: id })}
                />
              </PageBody>
            </div>
          )}

          {page === "session" && (
            <div className="min-h-0 flex-1 overflow-auto">
              <PageBody>
                <PageHeader title="Phiên" description="Ngân sách prompt và mức dùng của phiên đang chạy." />
                {engine.session.context === null && engine.session.usage === null ? (
                  <EmptyState
                    icon={BookOpen}
                    title="Chưa có số liệu"
                    description="Engine gửi manifest ngân sách sau mỗi lượt, và bảng mức dùng khi được hỏi."
                    hint={connected ? "Mở lại trang này sau một lượt." : "Engine chưa chạy."}
                  />
                ) : (
                  <SessionPanel
                    session={engine.session}
                    connected={connected && !sessionChanging}
                    onRefresh={() => {
                      void engine.send({ cmd: "context" });
                      void engine.send({ cmd: "usage" });
                    }}
                  />
                )}
              </PageBody>
            </div>
          )}

          {page === "settings" && (
            <div className="min-h-0 flex-1 overflow-auto">
              <PageBody>
                <SettingsPanel
                  settings={engine.settings}
                  connected={connected && !sessionChanging}
                  themeChoice={theme.choice}
                  onSetTheme={theme.setChoice}
                  sessionHistory={engine.sessionHistory}
                  persistenceChangePending={persistenceChangePending}
                  onRequestSessionPersistence={requestPersistenceChange}
                  onSetAutoOpenLast={(auto_open_last) =>
                    void engine.send({ cmd: "session_preferences_set", request_id: nanoid(), auto_open_last })
                  }
                  onLoadProviders={() => {
                    void engine.send({ cmd: "llm_providers" });
                    void engine.send({ cmd: "llm_config" });
                    void engine.send({ cmd: "status" });
                  }}
                  onSetKey={(provider, key) => void engine.send({ cmd: "llm_set_key", provider, key })}
                  onLoadModels={(provider, query) => void engine.send({ cmd: "llm_models", provider, query })}
                  onSelectModel={(provider, modelId) =>
                    void engine.send({
                      cmd: "llm_select",
                      backend: "remote",
                      provider,
                      model: modelId,
                      max_tokens: engine.settings.config?.maxTokens ?? 4096,
                      reasoning_enabled: engine.settings.config?.reasoningEnabled ?? false,
                    })
                  }
                  onSelectProfile={(profileKey) => void engine.send({ cmd: "voice_profile_select", profile: profileKey })}
                  onApplyGeneration={(change) => {
                    const config = engine.settings.config;
                    void engine.send({
                      cmd: "llm_select",
                      backend: change.backend ?? config?.backend ?? "remote",
                      provider: config?.provider ?? "openrouter",
                      model: config?.model ?? "",
                      max_tokens: change.maxTokens ?? config?.maxTokens ?? 4096,
                      reasoning_enabled: change.reasoningEnabled ?? config?.reasoningEnabled ?? false,
                    });
                  }}
                />
              </PageBody>
            </div>
          )}
        </div>
      </div>

      <Dialog open={persistenceChange !== null} onOpenChange={(open) => !open && setPersistenceChange(null)}>
        <DialogContent showCloseButton={false}>
          <DialogHeader>
            <DialogTitle>
              {persistenceChange === "enable" ? "Bật lưu phiên trên máy?" : "Tắt lưu phiên trên máy?"}
            </DialogTitle>
            <DialogDescription>
              {persistenceChange === "enable"
                ? "Sơn Ca sẽ khởi động lại để lưu các phiên tiếp theo dưới dạng văn bản, context làm việc và trạng thái mục tiêu. Phiên hiện tại đang ở RAM sẽ không được ghi ngược lại. Audio và ASR partial không được lưu."
                : "Sơn Ca sẽ khởi động lại ở chế độ chỉ dùng RAM. Các phiên đã lưu vẫn ở trên máy và không bị xóa."}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <DialogClose render={<button type="button" className="border-border h-8 rounded-lg border px-3 text-sm" />}>
              Hủy
            </DialogClose>
            <button
              type="button"
              className="bg-primary text-primary-foreground h-8 rounded-lg px-3 text-sm font-medium disabled:opacity-50"
              disabled={persistenceChangePending}
              onClick={() => void applyPersistenceChange()}
            >
              {persistenceChange === "enable" ? "Đồng ý và khởi động lại" : "Tắt lưu phiên"}
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </main>
  );
}
