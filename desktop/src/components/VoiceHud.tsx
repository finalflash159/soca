/**
 * Phase 3 surface: the voice loop.
 *
 * This is where a graphical app earns its keep over the TUI — a level meter and
 * a live partial transcript are genuinely hard to render in a terminal.
 *
 * What it deliberately does **not** do:
 *
 * * It never calls `getUserMedia`. Levels come from the engine's
 *   `voice_level.rms`, measured on the same buffer the recogniser sees. A second
 *   browser capture would be the two-stream arrangement that failed on clock
 *   drift, and it would fight AEC3 for the device.
 * * It shows no endpoint countdown. The engine publishes the silence floor and
 *   ceiling once and never a remaining-time figure, so a countdown would be a
 *   client-invented decision (`docs/18` §7 obligation 6).
 */

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { VoiceState } from "@/engine/voice";
import { LEVEL_HISTORY, partialText, peakLevel, voicePhaseLabel } from "@/engine/voice";

interface VoiceHudProps {
  voice: VoiceState;
  connected: boolean;
  onStart: () => void;
  onStop: () => void;
}

/**
 * Level meter over the retained rms window.
 *
 * Rendered as fixed-width bars rather than a smoothed waveform: rms is one
 * magnitude per frame, not a sample buffer, and drawing a smooth curve through
 * it would imply a resolution the data does not have.
 */
function LevelMeter({ levels, active }: { levels: number[]; active: boolean }) {
  const padded = [...Array(Math.max(0, LEVEL_HISTORY - levels.length)).fill(0), ...levels];
  return (
    <div
      className="flex h-12 items-end gap-[2px]"
      role="img"
      aria-label={`Microphone level, peak ${Math.round(peakLevel(levels) * 100)} percent`}
    >
      {padded.map((level, index) => (
        <div
          key={index}
          className={
            active ? "bg-foreground/70 flex-1 rounded-sm" : "bg-muted-foreground/25 flex-1 rounded-sm"
          }
          // Floor at 2% so an idle mic reads as a baseline rather than a gap.
          style={{ height: `${Math.max(2, level * 100)}%` }}
        />
      ))}
    </div>
  );
}

function EndpointRow({ voice }: { voice: VoiceState }) {
  if (voice.endpoint === null) {
    return null;
  }
  const { adaptive, floorMs, ceilMs, smartTurn } = voice.endpoint;
  return (
    <div className="text-muted-foreground flex flex-wrap items-center gap-2 text-xs">
      <span>
        endpoint {floorMs ?? "—"}–{ceilMs ?? "—"} ms
      </span>
      {adaptive && <Badge variant="outline">adaptive</Badge>}
      {smartTurn && <Badge variant="outline">smart-turn</Badge>}
      {voice.asrModel !== null && (
        <span className="font-mono">{voice.asrModel}</span>
      )}
    </div>
  );
}

export function VoiceHud({ voice, connected, onStart, onStop }: VoiceHudProps) {
  const running = voice.phase !== "off";
  const capturing = voice.phase === "listening";
  const partial = partialText(voice.partial);

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle className="text-base">Voice</CardTitle>
        <div className="flex items-center gap-2">
          {voice.bargeIn !== "idle" && (
            <Badge variant={voice.bargeIn === "fired" ? "default" : "outline"}>
              barge-in {voice.bargeIn}
            </Badge>
          )}
          <Badge variant={running ? "secondary" : "outline"}>{voicePhaseLabel(voice.phase)}</Badge>
          <Button
            size="sm"
            variant={running ? "outline" : "default"}
            disabled={!connected}
            onClick={running ? onStop : onStart}
          >
            {running ? "Stop" : "Start"}
          </Button>
        </div>
      </CardHeader>

      <CardContent className="flex flex-col gap-3">
        <LevelMeter levels={voice.levels} active={capturing} />
        <EndpointRow voice={voice} />

        {partial !== "" && (
          <p className="text-sm">
            <span>{voice.partial?.committed}</span>{" "}
            <span className="text-muted-foreground italic">{voice.partial?.tentative}</span>
          </p>
        )}

        {voice.repairPrompt !== null && (
          // docs/18 §5: rejected speech becomes a repair prompt. It is a turn,
          // not a failure, and must not be styled as an error.
          <div className="border-muted-foreground/30 rounded-md border border-dashed px-3 py-2 text-sm">
            <span className="text-muted-foreground text-xs">Repair prompt · </span>
            {voice.repairPrompt}
          </div>
        )}

        {voice.error !== null && (
          <div className="border-destructive/40 text-destructive rounded-md border px-3 py-2 text-xs">
            {voice.error}
          </div>
        )}

        <div className="text-muted-foreground flex items-center gap-3 text-xs">
          <span>{voice.turnCount} turn{voice.turnCount === 1 ? "" : "s"}</span>
          {voice.profile !== null && <span className="font-mono">{voice.profile}</span>}
          {voice.lastTurn?.rejected === true && <Badge variant="outline">last turn rejected</Badge>}
          {voice.lastTurn?.terminalStatus != null && (
            <span className="font-mono">{voice.lastTurn.terminalStatus}</span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
