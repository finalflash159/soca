/**
 * Voice diagnostics.
 *
 * Read-only on purpose. This panel used to carry its own Start/Stop, which made
 * it the fourth place in the app that could turn the microphone on, each with
 * slightly different consequences. Capture is now started and stopped in exactly
 * one place — entering and leaving voice mode — and this reports what that loop
 * is doing: levels, endpointing, and the last turn's outcome.
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

import { PanelEmpty, PanelRow, PanelSection } from "@/components/PanelSection";
import { Badge } from "@/components/ui/badge";
import type { VoiceState } from "@/engine/voice";
import { LEVEL_HISTORY, partialText, peakLevel, voicePhaseLabel } from "@/engine/voice";

interface VoiceHudProps {
  voice: VoiceState;
}

/**
 * Level meter over the retained rms window.
 *
 * Fixed-width bars, not a smoothed waveform: rms is one magnitude per frame,
 * not a sample buffer, and a smooth curve would imply a resolution the data
 * does not have.
 */
function LevelMeter({ levels, active }: { levels: number[]; active: boolean }) {
  const padded = [...Array(Math.max(0, LEVEL_HISTORY - levels.length)).fill(0), ...levels];
  return (
    <div
      className="flex h-14 items-end gap-[2px]"
      role="img"
      aria-label={`Microphone level, peak ${Math.round(peakLevel(levels) * 100)} percent`}
    >
      {padded.map((level, index) => (
        <div
          key={index}
          className={`flex-1 rounded-[1px] transition-[height] duration-75 ${
            active ? "bg-primary/80" : "bg-muted-foreground/20"
          }`}
          // Floor at 2% so an idle mic reads as a baseline rather than a gap.
          style={{ height: `${Math.max(2, level * 100)}%` }}
        />
      ))}
    </div>
  );
}

export function VoiceHud({ voice }: VoiceHudProps) {
  const running = voice.phase !== "off";
  const capturing = voice.phase === "listening";
  const partial = partialText(voice.partial);

  return (
    <div className="mx-auto flex w-full max-w-[46rem] flex-col gap-3">
      <PanelSection
        title="Voice loop"
        description={voice.profile ?? "no profile yet"}
        status={
          <>
            {voice.bargeIn !== "idle" && (
              <Badge variant={voice.bargeIn === "fired" ? "default" : "outline"}>
                barge-in {voice.bargeIn}
              </Badge>
            )}
            <Badge variant={running ? "secondary" : "outline"}>
              {voicePhaseLabel(voice.phase)}
            </Badge>
          </>
        }
      >
        <div className="flex flex-col gap-3">
          <LevelMeter levels={voice.levels} active={capturing} />

          {partial !== "" ? (
            <p className="text-[15px] leading-7">
              <span>{voice.partial?.committed}</span>{" "}
              <span className="text-muted-foreground">{voice.partial?.tentative}</span>
            </p>
          ) : (
            <PanelEmpty>
              {running
                ? "Chưa bắt được tiếng nào."
                : "Vòng thoại đang tắt. Mở chế độ thoại để bắt đầu."}
            </PanelEmpty>
          )}

          {voice.repairPrompt !== null && (
            // docs/18 §5: rejected speech becomes a repair prompt. It is a turn,
            // not a failure, and must not be styled as an error.
            <div className="border-muted-foreground/30 rounded-md border border-dashed px-3 py-2 text-sm">
              <span className="text-muted-foreground text-[10px] uppercase tracking-wide">
                repair
              </span>
              <p className="mt-1">{voice.repairPrompt}</p>
            </div>
          )}

          {voice.error !== null && (
            <p className="text-destructive text-xs">{voice.error}</p>
          )}
        </div>
      </PanelSection>

      <PanelSection title="Endpointing" description="Published once when capture starts">
        {voice.endpoint === null ? (
          <PanelEmpty>
            No capture yet. The engine sends the silence floor and ceiling at the
            start of a turn, and never a remaining-time figure — so this panel
            shows configuration, not a countdown.
          </PanelEmpty>
        ) : (
          <>
            <PanelRow label="silence window">
              <span className="font-mono">
                {voice.endpoint.floorMs ?? "—"}–{voice.endpoint.ceilMs ?? "—"} ms
              </span>
            </PanelRow>
            <PanelRow label="detector">
              <span className="flex flex-wrap gap-1">
                {voice.endpoint.adaptive && <Badge variant="outline">adaptive</Badge>}
                {voice.endpoint.smartTurn && <Badge variant="outline">smart-turn</Badge>}
                {!voice.endpoint.adaptive && !voice.endpoint.smartTurn && (
                  <span className="text-muted-foreground">fixed</span>
                )}
              </span>
            </PanelRow>
            <PanelRow label="asr">
              <span className="font-mono">{voice.asrModel ?? "—"}</span>
            </PanelRow>
          </>
        )}
      </PanelSection>

      <PanelSection title="Turns" description={`${voice.turnCount} completed`}>
        {voice.lastTurn === null ? (
          <PanelEmpty>No turn has finished yet.</PanelEmpty>
        ) : (
          <>
            <PanelRow label="last outcome">
              {voice.lastTurn.rejected ? (
                <Badge variant="outline">rejected</Badge>
              ) : (
                <span className="font-mono">{voice.lastTurn.terminalStatus ?? "achieved"}</span>
              )}
            </PanelRow>
            {voice.lastTurn.durationS !== null && (
              <PanelRow label="duration">
                <span className="font-mono">{voice.lastTurn.durationS.toFixed(1)} s</span>
              </PanelRow>
            )}
          </>
        )}
      </PanelSection>
    </div>
  );
}
