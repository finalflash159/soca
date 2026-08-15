/**
 * Shell and navigation.
 *
 * Four surfaces, one section idiom. The plan's §5.6.1 lesson is that a sidebar
 * built from one section component means the user learns one behaviour rather
 * than four — the same argument applies to tabs, so every panel is a stack of
 * `Card`s with the same header shape.
 *
 * The engine controls stay outside the tabs: they are the precondition for
 * everything else, and hiding them behind a tab would let the app look idle
 * when it is simply not connected.
 */

import { useState } from "react";
import { ThinkingOrb } from "thinking-orbs";

import { ChatView } from "@/components/ChatView";
import { KnowledgePanel } from "@/components/KnowledgePanel";
import { SettingsPanel } from "@/components/SettingsPanel";
import { VoiceHud } from "@/components/VoiceHud";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { orbLabel } from "@/engine/orb";
import { useEngine } from "@/engine/useEngine";

function StatusBadge({ status }: { status: ReturnType<typeof useEngine>["status"] }) {
  switch (status.state) {
    case "running":
      return <Badge>running</Badge>;
    case "starting":
      return <Badge variant="secondary">starting</Badge>;
    case "stopped":
      return (
        <Badge variant={status.graceful ? "secondary" : "destructive"}>
          {status.graceful ? "stopped cleanly" : "stopped without bye"}
        </Badge>
      );
    case "failed":
      return <Badge variant="destructive">failed</Badge>;
    default:
      return <Badge variant="outline">idle</Badge>;
  }
}

export default function App() {
  const engine = useEngine();
  const [program, setProgram] = useState("soca");
  const running = engine.status.state === "running" || engine.status.state === "starting";
  const connected = engine.status.state === "running";

  return (
    <main className="mx-auto flex h-screen max-w-4xl flex-col gap-4 p-6">
      <header className="flex items-center gap-4">
        <ThinkingOrb state={engine.orbState} size={64} />
        <div className="flex flex-col gap-1">
          <h1 className="text-xl font-semibold tracking-tight">SoCa</h1>
          <p className="text-muted-foreground text-sm">{orbLabel(engine.orbState)}</p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <StatusBadge status={engine.status} />
          <input
            className="border-input bg-background h-8 w-28 rounded-md border px-2 text-xs"
            value={program}
            onChange={(event) => setProgram(event.target.value)}
            disabled={running}
            aria-label="Engine executable"
          />
          <Button size="sm" onClick={() => void engine.start({ program })} disabled={running}>
            Start
          </Button>
          <Button size="sm" variant="outline" onClick={() => void engine.stop()} disabled={!running}>
            Stop
          </Button>
        </div>
      </header>

      {engine.versionMismatch !== null && (
        <Alert variant="destructive">
          <AlertTitle>Protocol mismatch</AlertTitle>
          <AlertDescription>{engine.versionMismatch}</AlertDescription>
        </Alert>
      )}

      {engine.status.state === "failed" && (
        <Alert variant="destructive">
          <AlertTitle>Sidecar failed</AlertTitle>
          <AlertDescription>{engine.status.message}</AlertDescription>
        </Alert>
      )}

      {engine.errors.length > 0 && (
        <Alert variant="destructive">
          <AlertTitle>Engine errors</AlertTitle>
          <AlertDescription>
            <ul className="list-disc pl-4">
              {engine.errors.map((error, index) => (
                <li key={index}>{error}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      <Separator />

      <Tabs defaultValue="chat" className="flex min-h-0 flex-1 flex-col">
        <TabsList>
          <TabsTrigger value="chat">Chat</TabsTrigger>
          <TabsTrigger value="voice">Voice</TabsTrigger>
          <TabsTrigger value="knowledge">Knowledge</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="chat" className="flex min-h-0 flex-1 flex-col">
          <ChatView
            conversation={engine.conversation}
            connected={connected}
            onSend={(text) => void engine.send({ cmd: "chat", text })}
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
      </Tabs>

      <details className="text-muted-foreground text-xs">
        <summary className="cursor-pointer select-none">
          Protocol frames ({engine.log.length})
        </summary>
        <ScrollArea className="mt-2 h-32">
          <ul className="flex flex-col gap-1 font-mono text-[10px]">
            {engine.log.map((frame, index) => (
              <li key={index}>
                <span className="text-foreground">{frame.event}</span>
                {"type" in frame && typeof frame.type === "string" ? `:${frame.type}` : ""}
              </li>
            ))}
          </ul>
        </ScrollArea>
      </details>
    </main>
  );
}
