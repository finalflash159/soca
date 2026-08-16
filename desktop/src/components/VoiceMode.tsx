/**
 * The voice page.
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
 * 3. **Leaving ends the loop.** Navigating away from this page stops capture,
 *    so there is no state where the microphone is open with nothing on screen
 *    saying so.
 *
 * The orb stays `thinking-orbs` at its tuned 64px per plan §0.2 — the library
 * ships two fixed designs rather than one scalable one, so CSS-scaling the
 * canvas to fill a screen would only blur it. Size is conveyed by the halo
 * around it, which is amplitude, not agent state.
 */

import { Activity, Mic, MicOff, MessageSquareText, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { OrbState } from "thinking-orbs";
import { ThinkingOrb } from "thinking-orbs";

import { Button } from "@/components/ui/button";
import { VoiceHud } from "@/components/VoiceHud";
import { VoiceTranscript } from "@/components/VoiceTranscript";
import type { ConversationState, Turn } from "@/engine/conversation";
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
 * The live turn, centre stage.
 *
 * This is where a spoken turn happens, and it stays here for the whole of it —
 * the partial transcript growing word by word as you speak, then the answer
 * arriving sentence by sentence as it is spoken. Only when the turn closes does
 * it leave and become a pair of bubbles in the transcript below.
 *
 * The layer split is Pipecat's `TranscriptOverlay`: what is happening now is
 * large, centred and ephemeral; the record is small, aligned and permanent.
 * Rendering the live turn in both places at once was the mistake — it put the
 * same words on screen twice and made the centre look empty by comparison.
 */
function LiveTurn({ voice, turn }: { voice: VoiceState; turn: Turn | null }) {
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

  // The answer, while it is being spoken. `voice/sentence` lands one guardrail
  // -passed sentence at a time, so this fills in at the pace of the speech.
  const answer = turn === null ? "" : turn.streamedText;
  if (answer !== "") {
    return (
      <p className="max-w-2xl text-center text-[17px] leading-8">
        {answer}
        <span className="bg-primary ml-1 inline-block h-4 w-[2px] animate-pulse align-text-bottom" />
      </p>
    );
  }

  if (partial === "") {
    return null;
  }

  // Recognised-so-far in full contrast, the tentative tail dimmed: the engine
  // may still revise the tail, and showing both at one weight would present a
  // guess as a decision.
  return (
    <p className="max-w-2xl text-center text-[17px] leading-8">
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
  const [detailsOpen, setDetailsOpen] = useState(false);

  // Escape leaves. It was advertised on the close button but wired to a
  // `keydown` on a div nothing ever focused, so it had never once fired.
  const leaveRef = useRef(onLeave);
  leaveRef.current = onLeave;
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") {
        return;
      }
      // Whoever is on top gets the key. Nothing in the app opens a dialog over
      // this page today, but a command palette would, and ending the call
      // because someone dismissed a palette is the kind of surprise that is
      // cheaper to prevent than to notice later.
      if (document.querySelector('[role="dialog"][data-state="open"]') !== null) {
        return;
      }
      leaveRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const running = voice.phase !== "off";
  const level = running ? recentLevel(voice.levels) : 0;

  // The turn in progress stays centre stage; the transcript below shows only
  // what has finished. One turn, one place on screen at a time.
  const newest = conversation.turns[conversation.turns.length - 1];
  const liveTurn =
    newest !== undefined && newest.surface === "voice" && newest.finalText === null ? newest : null;
  const settled = liveTurn === null ? conversation.turns : conversation.turns.slice(0, -1);
  // The orb gives up the screen for history, not for an empty panel. On the
  // first turn there is nothing settled yet, so it stays full size.
  const compact = (transcriptOpen && settled.length > 0) || detailsOpen;

  const status = !running
    ? "Đang tắt mic"
    : voice.phase === "starting"
      ? // Measured at 9.2 s on this machine. Saying so beats a resting orb
        // that reads as "ready" while the microphone is not open yet.
        "Đang nạp mô hình giọng nói…"
      : orbLabel(orbState);

  return (
    <div
      // Fills the page area. The sidebar and top bar are outside it, so
      // settings, the engine health dot and the restart button stay one click
      // away mid-call — this screen used to cover the window and hide them.
      className="bg-background flex h-full w-full flex-col"
      role="region"
      aria-label="Chế độ thoại"
    >
      <div
        className={cn(
          "relative flex min-h-0 flex-col items-center justify-center gap-6",
          // The orb yields the screen to the transcript rather than overlapping
          // it — the same trade LiveKit's tile layout makes.
          compact ? "shrink-0 pt-10 pb-4" : "flex-1",
        )}
      >
        <div
          className={cn(
            "relative flex items-center justify-center transition-[width,height] duration-300",
            compact ? "size-32" : "size-[22rem]",
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

        {/* Always rendered, transcript open or not. This is the live turn, and
            hiding it behind the transcript toggle removed the one thing a voice
            screen exists to show. */}
        <div className="flex min-h-24 flex-col items-center justify-start gap-3 px-8">
          <p className="text-muted-foreground text-sm" role="status">
            {status}
          </p>
          <LiveTurn voice={voice} turn={liveTurn} />
          {voice.error !== null && <p className="text-destructive text-sm">{voice.error}</p>}
        </div>
      </div>

      {transcriptOpen && !detailsOpen && (
        <VoiceTranscript
          conversation={{ ...conversation, turns: settled }}
          documents={documents}
          orbState={orbState}
        />
      )}

      {/* Diagnostics live here rather than in settings, because every reading
          on them is live: navigating away from this page stops the loop, so a
          level meter on another page would only ever show a dead mic. */}
      {detailsOpen && (
        <div className="min-h-0 flex-1 overflow-auto px-6 pb-2">
          <VoiceHud voice={voice} />
        </div>
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

          <Button
            size="sm"
            variant="ghost"
            className={cn(
              "size-11 rounded-full p-0",
              detailsOpen ? "text-primary" : "text-muted-foreground",
            )}
            title={detailsOpen ? "Ẩn chi tiết" : "Chi tiết kỹ thuật"}
            aria-label={detailsOpen ? "Ẩn chi tiết" : "Chi tiết kỹ thuật"}
            aria-pressed={detailsOpen}
            onClick={() => setDetailsOpen((open) => !open)}
          >
            <Activity className="size-5" />
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
