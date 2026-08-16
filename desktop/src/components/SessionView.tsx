/**
 * The whole window.
 *
 * Shape taken from the claude.ai new-conversation screen: a narrow icon rail on
 * the left, then one canvas whose only content is a greeting and the composer,
 * sitting a third of the way down rather than centred or pinned. Nothing else
 * competes — no tab bar, no status strip, no cards.
 *
 * There is no separate "connect" step in the interface. The engine starts when
 * the app opens; while it comes up the composer says so. A button whose only job
 * is to make the app usable should not exist.
 *
 * Once a turn exists the greeting gives way to the transcript and the composer
 * docks to the bottom.
 *
 * The orb appears in three places, and the split is deliberate. Plan §0.2 makes
 * it the only agent-state animation, so it has to be where the state matters:
 *
 * * **Inside the running turn**, where the answer will land. For a tool or
 *   retrieval turn nothing is published until verification finishes, so the orb
 *   is the only thing distinguishing planning from retrieving from a stalled
 *   provider call.
 * * **Above the composer** for work with no turn to sit in — an index build, a
 *   memory compaction.
 * * **In the rail**, small and always present, as ambient state.
 *
 * An earlier revision had only the rail copy. That reduced a nine-state
 * indicator to a 20px decoration and hid exactly the information it exists to
 * show.
 */

import { BookOpen, Mic, PanelRight, Settings2, SlidersHorizontal } from "lucide-react";
import { ThinkingOrb } from "thinking-orbs";
import type { OrbState } from "thinking-orbs";

import { ChatView } from "@/components/ChatView";
import { Composer } from "@/components/Composer";
import { Button } from "@/components/ui/button";
import type { ConversationState } from "@/engine/conversation";
import type { SlashCommand, VaultDocument } from "@/engine/documents";
import { orbLabel } from "@/engine/orb";
import type { VoiceState } from "@/engine/voice";
import { LEVEL_HISTORY, partialText } from "@/engine/voice";
import { cn } from "@/lib/utils";

export type InspectorTab = "knowledge" | "voice" | "settings" | "frames";

interface SessionViewProps {
  orbState: OrbState;
  conversation: ConversationState;
  voice: VoiceState;
  documents: VaultDocument[];
  model: string | null;
  connected: boolean;
  starting: boolean;
  onSend: (text: string) => void;
  onCommand: (command: SlashCommand) => void;
  onToggleVoice: () => void;
  onOpenInspector: (tab: InspectorTab) => void;
}

/** Time-of-day greeting. No name — the app does not reliably know one. */
function greeting(): string {
  const hour = new Date().getHours();
  if (hour < 11) {
    return "Chào buổi sáng";
  }
  if (hour < 14) {
    return "Chào buổi trưa";
  }
  if (hour < 18) {
    return "Chào buổi chiều";
  }
  return "Chào buổi tối";
}

/** Compact level strip. Readings are the engine's rms, never a browser mic. */
function LevelStrip({ levels }: { levels: number[] }) {
  const window = levels.slice(-Math.floor(LEVEL_HISTORY / 2));
  const padded = [...Array(Math.max(0, 40 - window.length)).fill(0), ...window];
  return (
    <div className="flex h-5 items-center gap-[2px]" aria-hidden>
      {padded.map((level, index) => (
        <div
          key={index}
          className="bg-primary/70 w-[2px] rounded-full transition-[height] duration-75"
          style={{ height: `${Math.max(10, level * 100)}%` }}
        />
      ))}
    </div>
  );
}

function Rail({
  orbState,
  voiceRunning,
  onOpenInspector,
}: Pick<SessionViewProps, "orbState" | "onOpenInspector"> & { voiceRunning: boolean }) {
  const items: Array<{ tab: InspectorTab; icon: typeof BookOpen; label: string }> = [
    { tab: "knowledge", icon: BookOpen, label: "Knowledge" },
    { tab: "voice", icon: Mic, label: "Voice" },
    { tab: "settings", icon: Settings2, label: "Settings" },
    { tab: "frames", icon: SlidersHorizontal, label: "Protocol frames" },
  ];

  return (
    <nav className="border-border/40 flex w-14 shrink-0 flex-col items-center gap-1 border-r py-3">
      <div className="pb-3" title={orbLabel(orbState)}>
        <ThinkingOrb state={orbState} size={20} />
      </div>
      {items.map(({ tab, icon: Icon, label }) => (
        <Button
          key={tab}
          size="sm"
          variant="ghost"
          className={cn(
            "text-muted-foreground hover:text-foreground size-9 rounded-lg p-0",
            tab === "voice" && voiceRunning && "text-primary",
          )}
          title={label}
          aria-label={label}
          onClick={() => onOpenInspector(tab)}
        >
          <Icon className="size-4" />
        </Button>
      ))}
      <Button
        size="sm"
        variant="ghost"
        className="text-muted-foreground hover:text-foreground mt-auto size-9 rounded-lg p-0"
        title="Inspector"
        aria-label="Inspector"
        onClick={() => onOpenInspector("knowledge")}
      >
        <PanelRight className="size-4" />
      </Button>
    </nav>
  );
}

export function SessionView({
  orbState,
  conversation,
  voice,
  documents,
  model,
  connected,
  starting,
  onSend,
  onCommand,
  onToggleVoice,
  onOpenInspector,
}: SessionViewProps) {
  const voiceRunning = voice.phase !== "off";
  const partial = partialText(voice.partial);
  const hasTurns = conversation.turns.length > 0;
  const lastTurn = conversation.turns[conversation.turns.length - 1];
  // A turn that has not produced text yet renders its own orb inline; anything
  // else that keeps the orb off `breathing` is background work.
  const liveTurnShowsOrb =
    lastTurn !== undefined && lastTurn.finalText === null && lastTurn.streamedText === "";
  const busyOutsideTurn = orbState !== "breathing" && !liveTurnShowsOrb;

  const composer = (
    <Composer
      connected={connected}
      starting={starting}
      documents={documents}
      model={model}
      voiceRunning={voiceRunning}
      variant={hasTurns ? "docked" : "hero"}
      onSend={onSend}
      onCommand={onCommand}
      onToggleVoice={onToggleVoice}
      onOpenSettings={() => onOpenInspector("settings")}
    />
  );

  return (
    <div className="flex h-full w-full">
      <Rail orbState={orbState} voiceRunning={voiceRunning} onOpenInspector={onOpenInspector} />

      <section className="relative flex min-w-0 flex-1 flex-col">
        {hasTurns ? (
          <>
            <div className="relative flex min-h-0 flex-1 flex-col">
              {/* Scrolled text dissolves at the top edge instead of clipping. */}
              <div className="from-background pointer-events-none absolute inset-x-0 top-0 z-10 h-12 bg-gradient-to-b to-transparent" />
              <ChatView
                conversation={conversation}
                documents={documents}
                orbState={orbState}
                orbLabel={orbLabel(orbState)}
              />
            </div>
            <div className="bg-background z-20 flex shrink-0 flex-col gap-2 px-6 pb-5">
              {/* Work that happens outside a turn — an index build, a memory
                  compaction — has no message to sit inside, so the orb reports
                  it here. A live turn shows its own orb in the transcript. */}
              {busyOutsideTurn && (
                <div className="text-muted-foreground mx-auto flex w-full max-w-2xl items-center gap-2 text-xs">
                  <ThinkingOrb state={orbState} size={20} />
                  {orbLabel(orbState)}
                </div>
              )}
              <div className="mx-auto w-full max-w-2xl">{composer}</div>
            </div>
          </>
        ) : (
          // New conversation: greeting and composer are the entire screen, set a
          // third of the way down so the canvas below reads as room, not as a gap.
          <div className="flex min-h-0 flex-1 flex-col items-center px-6 pt-[18vh]">
            <div className="flex w-full max-w-2xl flex-col gap-7">
              <h1 className="flex items-center justify-center gap-3 text-center text-3xl font-normal tracking-tight">
                <ThinkingOrb state={orbState} size={20} />
                {greeting()}
              </h1>
              {composer}
              {busyOutsideTurn && (
                <p className="text-muted-foreground text-center text-xs">{orbLabel(orbState)}</p>
              )}
              {voiceRunning && (
                <div className="flex flex-col items-center gap-2">
                  {voice.phase === "listening" && <LevelStrip levels={voice.levels} />}
                  <p className="text-muted-foreground text-xs">{orbLabel(orbState)}</p>
                  {partial !== "" && (
                    <p className="max-w-md text-center text-[15px] leading-7">
                      <span>{voice.partial?.committed}</span>{" "}
                      <span className="text-muted-foreground">{voice.partial?.tentative}</span>
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
