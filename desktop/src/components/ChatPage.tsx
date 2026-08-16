/**
 * The conversation.
 *
 * Two states, as in the reference: an empty session is a greeting with the
 * composer under it, sitting a third of the way down; a session with turns is
 * a scrolling transcript with the composer docked. The composer is the same
 * component in both, so nothing about it can drift between the two.
 */

import { ThinkingOrb } from "thinking-orbs";
import type { OrbState } from "thinking-orbs";

import { ChatView } from "@/components/ChatView";
import { Composer } from "@/components/Composer";
import type { ConversationState } from "@/engine/conversation";
import type { SlashCommand, VaultDocument } from "@/engine/documents";
import { orbLabel } from "@/engine/orb";

interface ChatPageProps {
  orbState: OrbState;
  conversation: ConversationState;
  documents: VaultDocument[];
  model: string | null;
  connected: boolean;
  starting: boolean;
  onSend: (text: string) => void;
  onCommand: (command: SlashCommand) => void;
  onEnterVoiceMode: () => void;
  onOpenSettings: () => void;
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

export function ChatPage({
  orbState,
  conversation,
  documents,
  model,
  connected,
  starting,
  onSend,
  onCommand,
  onEnterVoiceMode,
  onOpenSettings,
}: ChatPageProps) {
  const hasTurns = conversation.turns.length > 0;
  const lastTurn = conversation.turns[conversation.turns.length - 1];
  // A turn that has not produced text yet renders its own orb inline; anything
  // else that keeps the orb off `breathing` is background work.
  const liveTurnShowsOrb =
    lastTurn !== undefined &&
    lastTurn.finalText === null &&
    lastTurn.streamedText === "";
  const busyOutsideTurn = orbState !== "breathing" && !liveTurnShowsOrb;

  const composer = (
    <Composer
      connected={connected}
      starting={starting}
      documents={documents}
      model={model}
      variant={hasTurns ? "docked" : "hero"}
      onSend={onSend}
      onCommand={onCommand}
      onEnterVoiceMode={onEnterVoiceMode}
      onOpenSettings={onOpenSettings}
    />
  );

  if (!hasTurns) {
    return (
      <div className="flex min-h-0 flex-1 flex-col items-center px-6 pt-[16vh]">
        <div className="flex w-full max-w-2xl flex-col gap-7">
          {/* Pinned to `solving`, like the mark in the top bar. On an empty
              session there is by definition nothing running, so a live orb here
              could only ever show `breathing` — a faint dotted ring that reads
              as a failed load. Real state still has two honest homes: the label
              below, and the orb inside a running turn. */}
          <h1 className="flex items-center justify-center gap-3 text-center text-3xl font-normal tracking-tight">
            <ThinkingOrb state="solving" size={20} aria-hidden />
            {greeting()}
          </h1>
          {composer}
          {busyOutsideTurn && (
            <p className="text-muted-foreground text-center text-xs">
              {orbLabel(orbState)}
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
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
            compaction — has no message to sit inside, so the orb reports it
            here. A live turn shows its own orb in the transcript. */}
        {busyOutsideTurn && (
          <div className="text-muted-foreground mx-auto flex w-full max-w-2xl items-center gap-2 text-xs">
            <ThinkingOrb state={orbState} size={20} />
            {orbLabel(orbState)}
          </div>
        )}
        <div className="mx-auto w-full max-w-2xl">{composer}</div>
      </div>
    </>
  );
}
