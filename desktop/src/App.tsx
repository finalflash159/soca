/**
 * Shell.
 *
 * Three decisions, from reading reference implementations rather than guessing
 * (see `zplan/desktop_ui_research_round2.vi.md`):
 *
 * 1. **The engine starts itself.** A first-run screen whose only content is a
 *    button to make the app work is a step, not a feature. The app launches the
 *    engine on mount and the composer reports progress; `StartupView` is kept
 *    for the case that actually needs a human — a launch that failed.
 * 2. **No tab bar.** The conversation owns the window. Everything else is an
 *    overlay reached from the icon rail.
 * 3. **The inspector overlays, it does not replace.** Retrieval, memory and
 *    settings open as a right-hand sheet on top of the conversation. The whole
 *    argument for a visible evidence trail (plan §5.6.4) collapses if checking
 *    the evidence means leaving the answer.
 */

import { useEffect, useRef, useState } from "react";

import { KnowledgePanel } from "@/components/KnowledgePanel";
import type { InspectorTab } from "@/components/SessionView";
import { SessionView } from "@/components/SessionView";
import { SessionPanel } from "@/components/SessionPanel";
import { SettingsPanel } from "@/components/SettingsPanel";
import { StartupView } from "@/components/StartupView";
import { VoiceHud } from "@/components/VoiceHud";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { documentIndex } from "@/engine/documents";
import { useEngine } from "@/engine/useEngine";

export default function App() {
  const engine = useEngine();
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("session");
  const autoStarted = useRef(false);

  const { start, ready } = engine;
  // No button whose only job is to make the app usable — but wait for the
  // listeners first, or the engine's opening frames are emitted into the void.
  useEffect(() => {
    if (!ready || autoStarted.current) {
      return;
    }
    autoStarted.current = true;
    void start({ program: "soca" });
  }, [ready, start]);

  const connected = engine.status.state === "running";
  const starting = engine.status.state === "starting";
  const documents = documentIndex(engine.knowledge, engine.conversation.turns);

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
          onStart={(program) => void engine.start({ program })}
        />
      </main>
    );
  }

  const openInspector = (tab: InspectorTab) => {
    setInspectorTab(tab);
    setInspectorOpen(true);
    // Fetch on open rather than on a timer: these are cheap, but polling them
    // would compete with a running turn for the engine's attention.
    for (const cmd of ["status", "context", "usage", "memory", "llm_config"] as const) {
      void engine.send({ cmd } as never);
    }
  };

  const restartEngine = async () => {
    await engine.stop();
    await engine.start({ program: "soca" });
  };

  return (
    <main className="h-screen">
      <SessionView
        orbState={engine.orbState}
        conversation={engine.conversation}
        voice={engine.voice}
        documents={documents}
        model={engine.settings.config?.model ?? null}
        connected={connected}
        starting={starting}
        onSend={(text) => void engine.send({ cmd: "chat", text })}
        onCommand={(command) => {
          if (command.id === "memory_compact") {
            void engine.send({ cmd: "memory_compact", action: "request" });
            return;
          }
          void engine.send({ cmd: command.id } as never);
        }}
        onToggleVoice={() =>
          void engine.send(
            engine.voice.phase === "off" ? { cmd: "voice_start" } : { cmd: "voice_stop" },
          )
        }
        onOpenInspector={openInspector}
        onRestartEngine={() => void restartEngine()}
      />

      <Sheet open={inspectorOpen} onOpenChange={setInspectorOpen}>
        <SheetContent className="flex w-[38rem] flex-col gap-0 sm:max-w-none">
          <SheetHeader>
            <SheetTitle>Inspector</SheetTitle>
          </SheetHeader>

          <Tabs value={inspectorTab} onValueChange={(value) => setInspectorTab(value as InspectorTab)} className="flex min-h-0 flex-1 flex-col px-4 pb-4">
            <TabsList>
              <TabsTrigger value="session">Session</TabsTrigger>
              <TabsTrigger value="knowledge">Knowledge</TabsTrigger>
              <TabsTrigger value="voice">Voice</TabsTrigger>
              <TabsTrigger value="settings">Settings</TabsTrigger>
              <TabsTrigger value="frames">Frames</TabsTrigger>
            </TabsList>

            <TabsContent value="session" className="min-h-0 flex-1 overflow-auto">
              <SessionPanel
                session={engine.session}
                connected={connected}
                onRefresh={() => {
                  void engine.send({ cmd: "context" });
                  void engine.send({ cmd: "usage" });
                }}
              />
            </TabsContent>

            <TabsContent value="knowledge" className="min-h-0 flex-1 overflow-auto">
              <KnowledgePanel
                knowledge={engine.knowledge}
                connected={connected}
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
            </TabsContent>

            <TabsContent value="voice" className="min-h-0 flex-1 overflow-auto">
              <VoiceHud
                voice={engine.voice}
                connected={connected}
                onStart={() => void engine.send({ cmd: "voice_start" })}
                onStop={() => void engine.send({ cmd: "voice_stop" })}
              />
            </TabsContent>

            <TabsContent value="settings" className="min-h-0 flex-1 overflow-auto">
              <SettingsPanel
                settings={engine.settings}
                connected={connected}
                onLoadProviders={() => {
                  void engine.send({ cmd: "llm_providers" });
                  void engine.send({ cmd: "llm_config" });
                  void engine.send({ cmd: "status" });
                }}
                onSetKey={(provider, key) => void engine.send({ cmd: "llm_set_key", provider, key })}
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
                  void engine.send({ cmd: "voice_profile_select", profile: profileKey })
                }
              />
            </TabsContent>

            <TabsContent value="frames" className="min-h-0 flex-1">
              <ScrollArea className="h-full">
                <ul className="flex flex-col gap-1 font-mono text-[10px]">
                  {engine.log.map((frame, index) => (
                    <li key={index} className="text-muted-foreground">
                      <span className="text-foreground">{frame.event}</span>
                      {"type" in frame && typeof frame.type === "string" ? `:${frame.type}` : ""}
                    </li>
                  ))}
                </ul>
              </ScrollArea>
            </TabsContent>
          </Tabs>

          {engine.errors.length > 0 && (
            <Alert variant="destructive" className="mx-4 mb-4 w-auto">
              <AlertTitle>Engine errors</AlertTitle>
              <AlertDescription>
                <ul className="list-disc pl-4">
                  {engine.errors.slice(-3).map((error, index) => (
                    <li key={index}>{error}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}
        </SheetContent>
      </Sheet>
    </main>
  );
}
