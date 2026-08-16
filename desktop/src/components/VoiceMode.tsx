/**
 * Full-screen voice mode.
 *
 * One object on an empty field, and a control bar. Everything else is gone —
 * no rail, no transcript, no inspector — because during a spoken turn there is
 * nothing to read and nothing to click.
 *
 * Two layers, and the split matters:
 *
 * * The **orb** is agent state, and stays `thinking-orbs` at its tuned 64 px.
 *   Plan §0.2 makes it the single source for that, and the library ships two
 *   separate designs rather than one scalable one — CSS-scaling the canvas to
 *   fill a screen would just blur it.
 * * The **halo** around it is microphone amplitude, which is not agent state.
 *   It is a plain gradient whose size and opacity track `voice_level.rms`, the
 *   same reading the recogniser sees. The WebView still never opens a mic.
 *
 * So the sphere reacts to your voice while the orb reports what the system is
 * doing, and neither pretends to be the other.
 */

import { Mic, MicOff, Plus, X } from "lucide-react";
import { useState } from "react";
import type { OrbState } from "thinking-orbs";
import { ThinkingOrb } from "thinking-orbs";

import { Button } from "@/components/ui/button";
import { orbLabel } from "@/engine/orb";
import type { VoiceState } from "@/engine/voice";
import { partialText, peakLevel } from "@/engine/voice";
import { cn } from "@/lib/utils";

interface VoiceModeProps {
  orbState: OrbState;
  voice: VoiceState;
  connected: boolean;
  onSend: (text: string) => void;
  onToggleMic: () => void;
  onClose: () => void;
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

export function VoiceMode({
  orbState,
  voice,
  connected,
  onSend,
  onToggleMic,
  onClose,
}: VoiceModeProps) {
  const [draft, setDraft] = useState("");
  const running = voice.phase !== "off";
  const level = running ? recentLevel(voice.levels) : 0;
  const partial = partialText(voice.partial);

  const submit = () => {
    const text = draft.trim();
    if (text === "" || !connected) {
      return;
    }
    onSend(text);
    setDraft("");
  };

  return (
    <div className="bg-background fixed inset-0 z-50 flex flex-col">
      <div className="relative flex min-h-0 flex-1 flex-col items-center justify-center gap-10">
        <div className="relative flex size-[22rem] items-center justify-center">
          {/* Amplitude halo. Scale and opacity are the only things driven by
              rms, so a silent room reads as a still sphere rather than noise. */}
          <div
            className="from-primary/70 via-primary/25 absolute inset-0 rounded-full bg-gradient-to-b to-transparent blur-2xl transition-transform duration-100"
            style={{
              transform: `scale(${0.72 + level * 0.28})`,
              opacity: running ? 0.35 + level * 0.5 : 0.18,
            }}
            aria-hidden
          />
          <div
            className="from-primary/40 to-primary/5 absolute inset-8 rounded-full bg-gradient-to-b transition-transform duration-100"
            style={{ transform: `scale(${0.9 + level * 0.12})` }}
            aria-hidden
          />
          <div className="relative">
            <ThinkingOrb state={orbState} size={64} />
          </div>
        </div>

        <div className="flex min-h-16 max-w-xl flex-col items-center gap-3 px-8 text-center">
          <p className="text-muted-foreground text-sm">
            {running ? orbLabel(orbState) : "Voice đang tắt"}
          </p>
          {partial !== "" && (
            <p className="text-[17px] leading-8">
              <span>{voice.partial?.committed}</span>{" "}
              <span className="text-muted-foreground">{voice.partial?.tentative}</span>
            </p>
          )}
          {voice.repairPrompt !== null && (
            // docs/18 §5: a rejected transcript becomes a repair prompt. It is a
            // turn, not an error.
            <p className="text-muted-foreground text-sm italic">{voice.repairPrompt}</p>
          )}
          {voice.error !== null && <p className="text-destructive text-sm">{voice.error}</p>}
        </div>
      </div>

      <div className="px-6 pb-6">
        <div className="border-border/70 bg-card mx-auto flex max-w-2xl items-center gap-2 rounded-full border px-3 py-2">
          <Button
            size="sm"
            variant="ghost"
            className="text-muted-foreground size-9 shrink-0 rounded-full p-0"
            title="Tài liệu — gõ @ trong khung chat"
            disabled
          >
            <Plus className="size-4" />
          </Button>
          <input
            className="flex-1 bg-transparent text-sm outline-none placeholder:opacity-60"
            value={draft}
            placeholder={connected ? "Hoặc gõ…" : "Engine chưa chạy"}
            disabled={!connected}
            aria-label="Message"
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
              if (event.key === "Escape") {
                onClose();
              }
            }}
          />
          <Button
            size="sm"
            variant="ghost"
            className={cn(
              "size-9 shrink-0 rounded-full p-0",
              running ? "text-primary" : "text-muted-foreground",
            )}
            title={running ? "Tắt mic" : "Bật mic"}
            disabled={!connected}
            onClick={onToggleMic}
          >
            {running ? <Mic className="size-4" /> : <MicOff className="size-4" />}
          </Button>
          <Button
            size="sm"
            className="size-9 shrink-0 rounded-full p-0"
            title="Đóng voice mode (Esc)"
            onClick={onClose}
          >
            <X className="size-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
