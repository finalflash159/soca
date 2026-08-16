/**
 * Shell.
 *
 * Two decisions, both from reading LiveKit's reference app rather than guessing
 * (see `zplan/desktop_ui_research_round2.vi.md`):
 *
 * 1. **State-driven views, not tabs.** Before the engine runs the window is a
 *    startup view; after, it is the session. Disabled chrome for a thing you
 *    cannot yet do is worse than not showing it.
 * 2. **The inspector overlays, it does not replace.** Retrieval, memory and
 *    settings open as a right-hand sheet on top of the conversation. The whole
 *    argument for a visible evidence trail (plan §5.6.4) collapses if checking
 *    the evidence means leaving the answer.
 */

import { useState } from "react";

import { KnowledgePanel } from "@/components/KnowledgePanel";
import { SessionView } from "@/components/SessionView";
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
  const [transcriptOpen, setTranscriptOpen] = useState(true);
  const [inspectorOpen, setInspectorOpen] = useState(false);

  const connected = engine.status.state === "running";
  const starting = engine.status.state === "starting";
  const documents = documentIndex(engine.knowledge, engine.conversation.turns);

  // §5.6.3: the settings that change how a turn runs belong beside the input.
  const contextChips = [
    engine.settings.config !== null
      ? { label: "model", value: engine.settings.config.model }
      : null,
    engine.voice.profile !== null ? { label: "profile", value: engine.voice.profile } : null,
  ].filter((chip): chip is { label: string; value: string } => chip !== null);

  // One string for the startup view: a launch failure, or an engine that exited
  // without saying `bye`.
  const startupProblem =
    engine.status.state === "failed"
      ? engine.status.message
      : engine.status.state === "stopped" && !engine.status.graceful
        ? "Engine đã thoát mà không gửi `bye`. Kiểm tra log ở terminal."
        : engine.versionMismatch;

  if (!connected) {
    return (
      <main className="h-screen">
        <StartupView
          starting={starting}
          problem={startupProblem}
          onStart={(program) => void engine.start({ program })}
        />
      </main>
    );
  }

  return (
    <main className="h-screen">
      <SessionView
        orbState={engine.orbState}
        conversation={engine.conversation}
        voice={engine.voice}
        documents={documents}
        contextChips={contextChips}
        transcriptOpen={transcriptOpen}
        inspectorOpen={inspectorOpen}
        onToggleTranscript={() => setTranscriptOpen((open) => !open)}
        onToggleInspector={() => {
          const opening = !inspectorOpen;
          setInspectorOpen(opening);
          // Fetch on open rather than on a timer: these are cheap commands, but
          // polling them would compete with a turn for the engine's attention.
          if (opening) {
            void engine.send({ cmd: "status" });
            void engine.send({ cmd: "memory" });
            void engine.send({ cmd: "llm_config" });
          }
        }}
        onSend={(text) => void engine.send({ cmd: "chat", text })}
        onCommand={(command) => {
          if (command.id === "memory_compact") {
            void engine.send({ cmd: "memory_compact", action: "request" });
            return;
          }
          void engine.send({ cmd: command.id } as never);
        }}
        onVoiceStart={() => void engine.send({ cmd: "voice_start" })}
        onVoiceStop={() => void engine.send({ cmd: "voice_stop" })}
      />

      <Sheet open={inspectorOpen} onOpenChange={setInspectorOpen}>
        <SheetContent className="flex w-[38rem] flex-col gap-0 sm:max-w-none">
          <SheetHeader>
            <SheetTitle>Inspector</SheetTitle>
          </SheetHeader>

          <Tabs defaultValue="knowledge" className="flex min-h-0 flex-1 flex-col px-4 pb-4">
            <TabsList>
              <TabsTrigger value="knowledge">Knowledge</TabsTrigger>
              <TabsTrigger value="voice">Voice</TabsTrigger>
              <TabsTrigger value="settings">Settings</TabsTrigger>
              <TabsTrigger value="frames">Frames</TabsTrigger>
            </TabsList>

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
                  void engine.send({ cmd: "llm_select", provider, model: modelId })
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
