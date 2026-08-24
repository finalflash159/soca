/** Voice capture with the same durable transcript as chat. */

import { Activity, Mic, MicOff, MessageSquareText, X } from "lucide-react";
import { useEffect, useRef, useState, type CSSProperties } from "react";

import { Button } from "@/components/ui/button";
import { VoiceHud } from "@/components/VoiceHud";
import { VoiceTranscript } from "@/components/VoiceTranscript";
import type { Citation, ConversationState, Turn } from "@/engine/conversation";
import type { CitationPreviewIndex } from "@/engine/citation-preview";
import { orbLabel, type OrbState } from "@/engine/orb";
import type { VoiceState } from "@/engine/voice";
import { partialText, peakLevel } from "@/engine/voice";
import { cn } from "@/lib/utils";
import { ThinkingOrb } from "thinking-orbs";

interface VoiceModeProps {
  orbState: OrbState;
  voice: VoiceState;
  conversation: ConversationState;
  citationPreviews: CitationPreviewIndex;
  onRequestCitationPreview: (citation: Citation) => Promise<boolean>;
  connected: boolean;
  transcriptOpen: boolean;
  onToggleTranscript: () => void;
  onToggleMic: () => void;
  onLeave: () => void;
  onLoadOlder: () => void;
  canLoadOlder: boolean;
}

/** Smooth the engine's real microphone RMS frames without fabricating progress. */
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
      <p className="max-w-2xl text-pretty text-center text-[17px] leading-8">
        {answer}
        <span className="bg-primary ml-1 inline-block h-4 w-[2px] animate-pulse align-text-bottom" aria-hidden="true" />
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
    <p className="max-w-2xl text-pretty text-center text-[17px] leading-8">
      <span>{voice.partial?.committed}</span>{" "}
      <span className="text-muted-foreground">{voice.partial?.tentative}</span>
    </p>
  );
}

export function VoiceMode({
  orbState,
  voice,
  conversation,
  citationPreviews,
  onRequestCitationPreview,
  connected,
  transcriptOpen,
  onToggleTranscript,
  onToggleMic,
  onLeave,
  onLoadOlder,
  canLoadOlder,
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
  const historyVisible = transcriptOpen && !detailsOpen;

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
          "relative flex min-h-0 flex-col items-center justify-center gap-6 px-6",
          // Keep the visual anchor fixed. Opening history changes only the
          // surrounding layout, never the sphere cluster's size or shape.
          historyVisible || detailsOpen ? "shrink-0 py-7" : "flex-1",
        )}
      >
        <div
          className={cn(
            "voice-orb-response relative flex shrink-0 items-center justify-center",
            historyVisible || detailsOpen ? "size-32" : "size-48",
          )}
          style={{ "--voice-level": level } as CSSProperties}
          data-testid="voice-activity"
        >
          <ThinkingOrb state={orbState} size={64} aria-hidden />
        </div>

        {/* Always rendered, transcript open or not. This is the live turn, and
            hiding it behind the transcript toggle removed the one thing a voice
            screen exists to show. */}
        <div className="flex min-h-24 max-w-2xl flex-col items-center justify-start gap-3">
          <p className="text-muted-foreground text-pretty text-center text-sm" role="status" aria-atomic="true">
            {status}
          </p>
          <LiveTurn voice={voice} turn={liveTurn} />
          {voice.error !== null && <p className="text-destructive text-sm">{voice.error}</p>}
        </div>
      </div>

      {historyVisible && (
        <VoiceTranscript
          conversation={{ ...conversation, turns: settled }}
          citationPreviews={citationPreviews}
          onRequestCitationPreview={onRequestCitationPreview}
          orbState={orbState}
          onLoadOlder={onLoadOlder}
          canLoadOlder={canLoadOlder}
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
