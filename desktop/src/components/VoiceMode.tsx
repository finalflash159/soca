/**
 * Voice mode.
 *
 * Shape taken from LiveKit's reference session view and Pipecat's voice kit
 * (see `zplan/desktop_ui_research_round3.vi.md`), which agree on three things
 * this screen previously got wrong:
 *
 * 1. **Voice is a view of the same conversation, not a separate place.** The
 *    transcript here is `conversation.turns` — the very list the chat surface
 *    renders. An earlier revision showed no history at all: a spoken turn
 *    scrolled past and was gone, because voice reduced only into live signals.
 * 2. **Three layers of text, not one.** The caption is what is being said *now*
 *    and disappears when the turn ends; the transcript is the record and is
 *    toggled; the phase label is status. Conflating them is why the old screen
 *    felt empty during a turn and blank after it.
 * 3. **Leaving ends the loop.** There is no state where this screen is closed
 *    and the microphone is still hot.
 *
 * The orb stays `thinking-orbs` at its tuned 64px per plan §0.2 — the library
 * ships two fixed designs rather than one scalable one, so CSS-scaling the
 * canvas to fill a screen would only blur it. Size is conveyed by the halo
 * around it, which is amplitude, not agent state.
 */

import { Mic, MicOff, MessageSquareText, X } from "lucide-react";
import type { OrbState } from "thinking-orbs";
import { ThinkingOrb } from "thinking-orbs";

import { Button } from "@/components/ui/button";
import { VoiceTranscript } from "@/components/VoiceTranscript";
import type { ConversationState } from "@/engine/conversation";
import type { VaultDocument } from "@/engine/documents";
import { orbLabel } from "@/engine/orb";
import type { VoiceState } from "@/engine/voice";
import { partialText, peakLevel } from "@/engine/voice";
import { cn } from "@/lib/utils";

interface VoiceModeProps {
  orbState: OrbState;
  voice: VoiceState;
  conversation: ConversationState;
  documents: VaultDocument[];
  connected: boolean;
  transcriptOpen: boolean;
  onToggleTranscript: () => void;
  onToggleMic: () => void;
  onLeave: () => void;
}

/**
 * Recent amplitude, smoothed.
 *
 * The raw per-frame rms makes the halo jitter; the peak of the last handful of
 * frames tracks speech without flickering on the gaps between syllables.
 */
function recentLevel(levels: number[]): number {
  return peakLevel(levels.slice(-6));
}

/**
 * What is being said right now.
 *
 * Ephemeral by design, following Pipecat's `TranscriptOverlay`: it carries the
 * live partial while the user speaks and the repair prompt when speech was
 * rejected, and it is empty the rest of the time. The record lives in the
 * transcript; duplicating it here would give two places to read the same thing.
 */
function Caption({ voice }: { voice: VoiceState }) {
  const partial = partialText(voice.partial);

  if (voice.repairPrompt !== null) {
    // docs/18 §5: rejected speech becomes a question, never an invented
    // transcript. It must not be styled as an error.
    return (
      <p className="text-muted-foreground max-w-xl text-center text-[15px] leading-7 italic">
        {voice.repairPrompt}
      </p>
    );
  }

  if (partial === "") {
    return null;
  }

  return (
    <p className="max-w-xl text-center text-[17px] leading-8">
      <span>{voice.partial?.committed}</span>{" "}
      <span className="text-muted-foreground">{voice.partial?.tentative}</span>
    </p>
  );
}

export function VoiceMode({
  orbState,
  voice,
  conversation,
  documents,
  connected,
  transcriptOpen,
  onToggleTranscript,
  onToggleMic,
  onLeave,
}: VoiceModeProps) {
  const running = voice.phase !== "off";
  const level = running ? recentLevel(voice.levels) : 0;

  return (
    <div
      className="bg-background fixed inset-0 z-50 flex flex-col"
      role="dialog"
      aria-label="Chế độ thoại"
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          onLeave();
        }
      }}
      tabIndex={-1}
    >
      <div
        className={cn(
          "relative flex min-h-0 flex-col items-center justify-center gap-6",
          // The orb yields the screen to the transcript rather than overlapping
          // it — the same trade LiveKit's tile layout makes.
          transcriptOpen ? "shrink-0 pt-10 pb-4" : "flex-1",
        )}
      >
        <div
          className={cn(
            "relative flex items-center justify-center transition-[width,height] duration-300",
            transcriptOpen ? "size-32" : "size-[22rem]",
          )}
        >
          {/* Amplitude halo. Only scale and opacity are driven by rms, so a
              silent room reads as a still sphere rather than as noise. */}
          <div
            className="from-primary/70 via-primary/25 absolute inset-0 rounded-full bg-gradient-to-b to-transparent blur-2xl transition-transform duration-100"
            style={{
              transform: `scale(${0.72 + level * 0.28})`,
              opacity: running ? 0.35 + level * 0.5 : 0.18,
            }}
            aria-hidden
          />
          <div
            className="from-primary/40 to-primary/5 absolute inset-[12%] rounded-full bg-gradient-to-b transition-transform duration-100"
            style={{ transform: `scale(${0.9 + level * 0.12})` }}
            aria-hidden
          />
          <ThinkingOrb state={orbState} size={64} />
        </div>

        <div className="flex min-h-14 flex-col items-center gap-3 px-8">
          <p className="text-muted-foreground text-sm" role="status">
            {running ? orbLabel(orbState) : "Đang tắt mic"}
          </p>
          {!transcriptOpen && <Caption voice={voice} />}
          {voice.error !== null && <p className="text-destructive text-sm">{voice.error}</p>}
        </div>
      </div>

      {transcriptOpen && (
        <VoiceTranscript
          conversation={conversation}
          documents={documents}
          orbState={orbState}
          voice={voice}
        />
      )}

      <div className="shrink-0 px-6 pb-8">
        <div className="border-border/70 bg-card mx-auto flex w-fit items-center gap-2 rounded-full border p-2">
          <Button
            size="sm"
            variant="ghost"
            className={cn(
              "size-11 rounded-full p-0",
              running ? "text-primary" : "text-muted-foreground",
            )}
            title={running ? "Tắt mic" : "Bật mic"}
            aria-label={running ? "Tắt mic" : "Bật mic"}
            aria-pressed={running}
            disabled={!connected}
            onClick={onToggleMic}
          >
            {running ? <Mic className="size-5" /> : <MicOff className="size-5" />}
          </Button>

          <Button
            size="sm"
            variant="ghost"
            className={cn(
              "size-11 rounded-full p-0",
              transcriptOpen ? "text-primary" : "text-muted-foreground",
            )}
            title={transcriptOpen ? "Ẩn hội thoại" : "Hiện hội thoại"}
            aria-label={transcriptOpen ? "Ẩn hội thoại" : "Hiện hội thoại"}
            aria-pressed={transcriptOpen}
            onClick={onToggleTranscript}
          >
            <MessageSquareText className="size-5" />
          </Button>

          <div className="bg-border mx-1 h-6 w-px" aria-hidden />

          {/* Leaving stops the loop. There is no "closed but still listening". */}
          <Button
            size="sm"
            variant="ghost"
            className="text-muted-foreground hover:text-destructive size-11 rounded-full p-0"
            title="Thoát chế độ thoại (Esc)"
            aria-label="Thoát chế độ thoại"
            onClick={onLeave}
          >
            <X className="size-5" />
          </Button>
        </div>
      </div>
    </div>
  );
}
