/**
 * Shell.
 *
 * Sidebar, top bar, one page. The arrangement is the reference app's and it
 * replaced an icon rail plus a tabbed sheet — see
 * the design notes for the image-by-image reading behind it.
 *
 * Three decisions carried over from the previous shell, each still true:
 *
 * 1. **The engine starts itself.** A first-run screen whose only content is a
 *    button to make the app work is a step, not a feature. `StartupView` is
 *    kept for the case that actually needs a human — a launch that failed.
 * 2. **Voice is a page, not an overlay.** It used to cover the window, which
 *    took the settings and the engine restart away exactly while running the
 *    part most likely to hang.
 * 3. **Entering voice starts the loop, leaving it stops the loop.** There is no
 *    state where the microphone is open with nothing on screen saying so.
 */

import { BookOpen } from "lucide-react";
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
import { documentIndex } from "@/engine/documents";
import { launchOptions } from "@/engine/launch";
import { useEngine } from "@/engine/useEngine";
import { useTheme } from "@/theme";

/** First words of the opening turn — the only session label that is real. */
function sessionTitle(text: string | undefined): string | null {
  if (text === undefined || text.trim() === "") {
    return null;
  }
  const trimmed = text.trim();
  return trimmed.length > 48 ? `${trimmed.slice(0, 48)}…` : trimmed;
}

export default function App() {
  const engine = useEngine();
  const theme = useTheme();
  const [page, setPage] = useState<PageId>("chat");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [transcriptOpen, setTranscriptOpen] = useState(true);
  const autoStarted = useRef(false);

  const { start, ready } = engine;
  // No button whose only job is to make the app usable — but wait for the
  // listeners first, or the engine's opening frames are emitted into the void.
  useEffect(() => {
    if (!ready || autoStarted.current) {
      return;
    }
    autoStarted.current = true;
    void start(launchOptions());
  }, [ready, start]);

  const connected = engine.status.state === "running";
  const starting = engine.status.state === "starting";
  const documents = documentIndex(engine.knowledge, engine.conversation.turns);
  const voiceRunning = engine.voice.phase !== "off";

  // One string for the startup view: a launch failure, or an engine that exited
  // without saying `bye`.
  const startupProblem =
    engine.status.state === "failed"
      ? engine.status.message
      : engine.status.state === "stopped" && !engine.status.graceful
        ? "Engine đã thoát mà không gửi `bye`. Kiểm tra log ở terminal."
        : engine.versionMismatch;

  // The startup view is only for a launch that actually failed. A normal cold
  // start shows the session with a composer that says it is coming up.
  if (engine.status.state === "failed" || engine.status.state === "stopped") {
    return (
      <main className="h-screen">
        <StartupView
          starting={false}
          problem={startupProblem}
          onStart={(program) => void engine.start({ ...launchOptions(), program })}
        />
      </main>
    );
  }

  /** Panels fetch on open rather than on a timer: polling would compete with a
   *  running turn for the engine's attention. */
  const openPage = (next: PageId) => {
    setPage(next);
    if (!connected) {
      return;
    }
    if (next === "session") {
      for (const cmd of ["status", "context", "usage"] as const) {
        void engine.send({ cmd });
      }
    }
    if (next === "knowledge") {
      for (const cmd of ["memory", "memory_proposals", "status"] as const) {
        void engine.send({ cmd });
      }
    }
    if (next === "settings") {
      for (const cmd of ["llm_providers", "llm_config", "status"] as const) {
        void engine.send({ cmd });
      }
    }
    if (next === "voice" && engine.voice.phase === "off") {
      void engine.send({ cmd: "voice_start" });
    }
  };

  /**
   * Leaving the voice page stops the loop.
   *
   * Navigating away is leaving — there is no meaning to a microphone that stays
   * open on the settings page.
   */
  const leaveVoice = (next: PageId) => {
    if (page === "voice" && next !== "voice" && engine.voice.phase !== "off") {
      void engine.send({ cmd: "voice_stop" });
    }
    openPage(next);
  };

  /** Pause and resume capture without leaving the page. */
  const toggleMic = () =>
    void engine.send(engine.voice.phase === "off" ? { cmd: "voice_start" } : { cmd: "voice_stop" });

  const restartEngine = async () => {
    await engine.stop();
    await engine.start(launchOptions());
  };

  return (
    <main className="bg-background flex h-screen w-screen overflow-hidden">
      {sidebarOpen && (
        <Sidebar
          page={page}
          onNavigate={leaveVoice}
          onNewConversation={() => leaveVoice("chat")}
          sessionTitle={sessionTitle(engine.conversation.turns[0]?.userText)}
          connected={connected}
          starting={starting}
          voiceRunning={voiceRunning}
          onRestartEngine={() => void restartEngine()}
          onCollapse={() => setSidebarOpen(false)}
        />
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar
          orbState={engine.orbState}
          sidebarOpen={sidebarOpen}
          onOpenSidebar={() => setSidebarOpen(true)}
          onToggleTheme={theme.toggle}
        />

        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
          {engine.errors.length > 0 && (
            <Alert variant="destructive" className="mx-6 mt-4 w-auto">
              <AlertTitle>Engine báo lỗi</AlertTitle>
              <AlertDescription>
                <ul className="list-disc pl-4">
                  {engine.errors.slice(-3).map((error, index) => (
                    <li key={index}>{error}</li>
                  ))}
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
              connected={connected}
              starting={starting}
              onSend={(text) => void engine.send({ cmd: "chat", text })}
              onCommand={(command) => {
                if (command.id === "memory_compact") {
                  void engine.send({
                    cmd: "memory_compact",
                    action: "request",
                  });
                  return;
                }
                void engine.send({ cmd: command.id } as never);
              }}
              onEnterVoiceMode={() => openPage("voice")}
              onOpenSettings={() => openPage("settings")}
            />
          )}

          {page === "voice" && (
            <VoiceMode
              orbState={engine.orbState}
              voice={engine.voice}
              conversation={engine.conversation}
              documents={documents}
              connected={connected}
              transcriptOpen={transcriptOpen}
              onToggleTranscript={() => setTranscriptOpen((open) => !open)}
              onToggleMic={toggleMic}
              onLeave={() => leaveVoice("chat")}
            />
          )}

          {page === "knowledge" && (
            <div className="min-h-0 flex-1 overflow-auto">
              <PageBody wide>
                <PageHeader
                  title="Kiến thức"
                  description="Vault, chỉ mục truy xuất và bộ nhớ phiên."
                />
                <KnowledgePanel
                  knowledge={engine.knowledge}
                  connected={connected}
                  onInit={() => void engine.send({ cmd: "knowledge_init" })}
                  onIndex={() => void engine.send({ cmd: "knowledge_index" })}
                  onRefreshMemory={() => {
                    void engine.send({ cmd: "memory" });
                    void engine.send({ cmd: "memory_proposals" });
                  }}
                  onCompact={() =>
                    void engine.send({
                      cmd: "memory_compact",
                      action: "request",
                    })
                  }
                  onApprove={(id) => void engine.send({ cmd: "memory_approve", proposal_id: id })}
                  onReject={(id) => void engine.send({ cmd: "memory_reject", proposal_id: id })}
                />
              </PageBody>
            </div>
          )}

          {page === "session" && (
            <div className="min-h-0 flex-1 overflow-auto">
              <PageBody>
                <PageHeader
                  title="Phiên"
                  description="Ngân sách prompt và mức dùng của phiên đang chạy."
                />
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
                    connected={connected}
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
                  connected={connected}
                  themeChoice={theme.choice}
                  onSetTheme={theme.setChoice}
                  onLoadProviders={() => {
                    void engine.send({ cmd: "llm_providers" });
                    void engine.send({ cmd: "llm_config" });
                    void engine.send({ cmd: "status" });
                  }}
                  onSetKey={(provider, key) =>
                    void engine.send({ cmd: "llm_set_key", provider, key })
                  }
                  onLoadModels={(provider, query) =>
                    void engine.send({ cmd: "llm_models", provider, query })
                  }
                  onSelectModel={(provider, modelId) =>
                    // `backend` is mandatory: _cmd_llm_select rejects the command
                    // outright without it. max_tokens and reasoning_enabled are
                    // resent so selecting a model does not silently reset them.
                    void engine.send({
                      cmd: "llm_select",
                      backend: "remote",
                      provider,
                      model: modelId,
                      max_tokens: engine.settings.config?.maxTokens ?? 4096,
                      reasoning_enabled: engine.settings.config?.reasoningEnabled ?? false,
                    })
                  }
                  onSelectProfile={(profileKey) =>
                    void engine.send({
                      cmd: "voice_profile_select",
                      profile: profileKey,
                    })
                  }
                  onApplyGeneration={(change) => {
                    const config = engine.settings.config;
                    // llm_select is the whole-settings command: anything omitted
                    // is reset, so every field is resent from current state.
                    void engine.send({
                      cmd: "llm_select",
                      backend: change.backend ?? config?.backend ?? "remote",
                      provider: config?.provider ?? "openrouter",
                      model: config?.model ?? "",
                      max_tokens: change.maxTokens ?? config?.maxTokens ?? 4096,
                      reasoning_enabled:
                        change.reasoningEnabled ?? config?.reasoningEnabled ?? false,
                    });
                  }}
                />
              </PageBody>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
