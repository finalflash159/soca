/**
 * Phase 1 surface: prove the sidecar boundary works.
 *
 * The plan's phase-1 milestone is narrow on purpose — start the engine, show
 * `soca status`, exit without orphaning a process. Conversation, voice and
 * knowledge are phases 2–4 and are intentionally absent here.
 */

import { useState } from "react";
import { ThinkingOrb } from "thinking-orbs";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { ChatView } from "@/components/ChatView";
import { VoiceHud } from "@/components/VoiceHud";
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

  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col gap-6 p-8">
      <header className="flex items-center gap-4">
        <ThinkingOrb state={engine.orbState} size={64} />
        <div className="flex flex-col gap-1">
          <h1 className="text-xl font-semibold tracking-tight">SoCa</h1>
          <p className="text-muted-foreground text-sm">{orbLabel(engine.orbState)}</p>
        </div>
        <div className="ml-auto">
          <StatusBadge status={engine.status} />
        </div>
      </header>

      <Separator />

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Engine</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <input
              className="border-input bg-background flex h-9 w-full rounded-md border px-3 py-1 text-sm shadow-xs"
              value={program}
              onChange={(event) => setProgram(event.target.value)}
              disabled={running}
              aria-label="Engine executable"
            />
            <Button
              onClick={() => void engine.start({ program })}
              disabled={running}
            >
              Start
            </Button>
            <Button variant="outline" onClick={() => void engine.stop()} disabled={!running}>
              Stop
            </Button>
          </div>
          <p className="text-muted-foreground text-xs">
            Runs <code>{program} engine</code>. Packaging a Python sidecar is a phase-5
            problem; for now the app expects <code>soca</code> on PATH — use{" "}
            <code>uv</code> with args if it is not.
          </p>

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

          {engine.hello !== null && (
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
              <dt className="text-muted-foreground">profile</dt>
              <dd>{engine.hello.profile}</dd>
              <dt className="text-muted-foreground">protocol</dt>
              <dd>{engine.hello.protocol_version}</dd>
              {Object.entries(engine.hello.stack).map(([key, value]) => (
                <div key={key} className="contents">
                  <dt className="text-muted-foreground">{key}</dt>
                  <dd className="font-mono text-xs">{value}</dd>
                </div>
              ))}
            </dl>
          )}
        </CardContent>
      </Card>

      {engine.engineStatus?.runtime_components && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Runtime components</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-2">
            {engine.engineStatus.runtime_components.map((component, index) => (
              <div key={index} className="flex items-center justify-between text-sm">
                <span>{String(component.name ?? "component")}</span>
                <Badge variant={component.status === "ok" ? "secondary" : "outline"}>
                  {String(component.status ?? "unknown")}
                </Badge>
              </div>
            ))}
          </CardContent>
        </Card>
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

      <VoiceHud
        voice={engine.voice}
        connected={engine.status.state === "running"}
        onStart={() => void engine.send({ cmd: "voice_start" })}
        onStop={() => void engine.send({ cmd: "voice_stop" })}
      />

      <ChatView
        conversation={engine.conversation}
        connected={engine.status.state === "running"}
        onSend={(text) => void engine.send({ cmd: "chat", text })}
      />

      <details className="text-muted-foreground text-xs">
        <summary className="cursor-pointer select-none">
          Protocol frames ({engine.log.length})
        </summary>
        <ScrollArea className="mt-2 h-40">
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
