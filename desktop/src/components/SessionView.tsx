/**
 * The window once the engine is running.
 *
 * Structure taken from LiveKit's `agent-session-block.tsx`:
 *
 * * one full-bleed section, everything inside positioned against it;
 * * the agent visualiser is the centre of the screen, not a corner badge;
 * * the transcript overlays it in a narrow column and can be closed, because a
 *   voice assistant's default view is the agent, not a wall of text;
 * * a gradient at the top edge so scrolled content dissolves under the header
 *   instead of being cut off;
 * * one bottom block holding the input and its controls together, rather than a
 *   bare textarea stranded at the window edge.
 *
 * SoCa's deviations: no camera or screen-share (audio only), and the visualiser
 * is `thinking-orbs` rather than a shader, because plan §0.2 makes it the single
 * source for agent-state animation.
 */

import { Mic, MicOff, PanelRight, ScrollText } from "lucide-react";

import { ChatView } from "@/components/ChatView";
import { Composer } from "@/components/Composer";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import type { ConversationState } from "@/engine/conversation";
import type { SlashCommand, VaultDocument } from "@/engine/documents";
import { orbLabel } from "@/engine/orb";
import type { VoiceState } from "@/engine/voice";
import { LEVEL_HISTORY, partialText } from "@/engine/voice";
import type { OrbState } from "thinking-orbs";
import { ThinkingOrb } from "thinking-orbs";

interface SessionViewProps {
  orbState: OrbState;
  conversation: ConversationState;
  voice: VoiceState;
  documents: VaultDocument[];
  contextChips: Array<{ label: string; value: string }>;
  transcriptOpen: boolean;
  inspectorOpen: boolean;
  onToggleTranscript: () => void;
  onToggleInspector: () => void;
  onSend: (text: string) => void;
  onCommand: (command: SlashCommand) => void;
  onVoiceStart: () => void;
  onVoiceStop: () => void;
}

/**
 * Compact level strip shown under the orb while capturing.
 *
 * Same rule as the Voice panel: the readings are the engine's `voice_level.rms`,
 * never a second browser microphone.
 */
function LevelStrip({ levels }: { levels: number[] }) {
  const window = levels.slice(-Math.floor(LEVEL_HISTORY / 2));
  const padded = [...Array(Math.max(0, 48 - window.length)).fill(0), ...window];
  return (
    <div className="flex h-6 w-48 items-center justify-center gap-[2px]" aria-hidden>
      {padded.map((level, index) => (
        <div
          key={index}
          className="bg-primary/70 w-[2px] rounded-full transition-[height] duration-75"
          style={{ height: `${Math.max(8, level * 100)}%` }}
        />
      ))}
    </div>
  );
}

export function SessionView({
  orbState,
  conversation,
  voice,
  documents,
  contextChips,
  transcriptOpen,
  inspectorOpen,
  onToggleTranscript,
  onToggleInspector,
  onSend,
  onCommand,
  onVoiceStart,
  onVoiceStop,
}: SessionViewProps) {
  const voiceRunning = voice.phase !== "off";
  const partial = partialText(voice.partial);
  const hasTurns = conversation.turns.length > 0;
  // The orb takes the centre until there is something to read; after that it
  // steps back to the header so the transcript owns the space.
  const orbCentred = !transcriptOpen || !hasTurns;

  return (
    <section className="relative flex h-full w-full flex-col overflow-hidden">
      <header className="z-20 flex items-center gap-3 px-5 py-3">
        {!orbCentred && <ThinkingOrb state={orbState} size={20} />}
        <span className="text-sm font-medium tracking-tight">SoCa</span>
        <span className="text-muted-foreground text-xs">{orbLabel(orbState)}</span>
        <div className="ml-auto flex items-center gap-1">
          <Button
            size="sm"
            variant={transcriptOpen ? "secondary" : "ghost"}
            onClick={onToggleTranscript}
            aria-pressed={transcriptOpen}
            title="Transcript"
          >
            <ScrollText className="size-4" />
          </Button>
          <Button
            size="sm"
            variant={inspectorOpen ? "secondary" : "ghost"}
            onClick={onToggleInspector}
            aria-pressed={inspectorOpen}
            title="Inspector"
          >
            <PanelRight className="size-4" />
          </Button>
        </div>
      </header>

      <div className="relative min-h-0 flex-1">
        {/* Agent stage. Stays mounted so the orb animation never restarts. */}
        <div
          className={`pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-4 transition-opacity duration-300 ${
            orbCentred ? "opacity-100" : "opacity-0"
          }`}
        >
          <ThinkingOrb state={orbState} size={64} />
          <p className="text-muted-foreground text-sm">{orbLabel(orbState)}</p>
          {voice.phase === "listening" && <LevelStrip levels={voice.levels} />}
          {partial !== "" && (
            <p className="max-w-md px-6 text-center text-[15px] leading-7">
              <span>{voice.partial?.committed}</span>{" "}
              <span className="text-muted-foreground">{voice.partial?.tentative}</span>
            </p>
          )}
        </div>

        {/* Transcript overlays the stage. */}
        {transcriptOpen && hasTurns && (
          <div className="absolute inset-0">
            {/* Scrolled content dissolves under the header rather than clipping. */}
            <div className="from-background pointer-events-none absolute inset-x-0 top-0 z-10 h-16 bg-gradient-to-b to-transparent" />
            <ChatView conversation={conversation} documents={documents} />
          </div>
        )}
      </div>

      <div className="z-20 px-5 pb-4">
        <div className="border-input bg-card/60 mx-auto flex max-w-2xl flex-col rounded-xl border">
          <Composer
            connected
            documents={documents}
            contextChips={[]}
            onSend={onSend}
            onCommand={onCommand}
          />
          <Separator />
          <div className="flex items-center gap-2 px-3 py-2">
            <Button
              size="sm"
              variant={voiceRunning ? "default" : "ghost"}
              onClick={voiceRunning ? onVoiceStop : onVoiceStart}
              title={voiceRunning ? "Stop the voice loop" : "Start the voice loop"}
            >
              {voiceRunning ? <Mic className="size-4" /> : <MicOff className="size-4" />}
              <span className="ml-1 text-xs">{voiceRunning ? "Voice on" : "Voice"}</span>
            </Button>
            <div className="text-muted-foreground ml-auto flex flex-wrap items-center gap-2 text-[10px]">
              {contextChips.map((chip) => (
                <span key={chip.label} className="font-mono">
                  {chip.value}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
