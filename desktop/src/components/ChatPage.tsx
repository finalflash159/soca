/**
 * The conversation.
 *
 * Two states, as in the reference: an empty session is a greeting with the
 * composer under it, sitting a third of the way down; a session with turns is
 * a scrolling transcript with the composer docked. The composer is the same
 * component in both, so nothing about it can drift between the two.
 */

import { ChatView } from "@/components/ChatView";
import { BrandMark } from "@/components/BrandMark";
import { Composer } from "@/components/Composer";
import { SessionContext } from "@/components/SessionContext";
import type { ConversationState } from "@/engine/conversation";
import type { Citation } from "@/engine/conversation";
import type { CitationPreviewIndex } from "@/engine/citation-preview";
import type { SlashCommand, VaultDocument } from "@/engine/documents";
import type { SessionState } from "@/engine/session";
import type {
  SessionHistoryState,
  SessionSummary,
} from "@/engine/session-history";
import { orbLabel, type OrbState } from "@/engine/orb";
import { ThinkingOrb } from "thinking-orbs";

interface ChatPageProps {
  orbState: OrbState;
  conversation: ConversationState;
  documents: VaultDocument[];
  model: string | null;
  connected: boolean;
  runtimeReady: boolean;
  voiceReady: boolean;
  voiceReason: string | null;
  starting: boolean;
  onSend: (text: string) => void;
  onCommand: (command: SlashCommand) => void;
  onEnterVoiceMode: () => void;
  onOpenSettings: () => void;
  onLoadOlder: () => void;
  canLoadOlder: boolean;
  citationPreviews: CitationPreviewIndex;
  onRequestCitationPreview: (citation: Citation) => Promise<boolean>;
  session: SessionState;
  sessionHistory: SessionHistoryState;
  sessionBusy: boolean;
  onRefreshSession: () => void;
  onOpenSession: (session: SessionSummary) => void;
  onRenameSession: (session: SessionSummary, title: string) => void;
  onDeleteSession: (session: SessionSummary) => void;
  onLoadMoreSessions: () => void;
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
  runtimeReady,
  voiceReady,
  voiceReason,
  starting,
  onSend,
  onCommand,
  onEnterVoiceMode,
  onOpenSettings,
  onLoadOlder,
  canLoadOlder,
  citationPreviews,
  onRequestCitationPreview,
  session,
  sessionHistory,
  sessionBusy,
  onRefreshSession,
  onOpenSession,
  onRenameSession,
  onDeleteSession,
  onLoadMoreSessions,
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
      runtimeReady={runtimeReady}
      voiceReady={voiceReady}
      voiceReason={voiceReason}
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
          <h1 className="flex items-center justify-center gap-3 text-center text-3xl font-normal tracking-tight">
            <BrandMark iconClassName="text-2xl" nameClassName="sr-only" />
            {greeting()}
          </h1>
          {composer}
          <SessionContext
            session={session}
            history={sessionHistory}
            connected={connected}
            busy={sessionBusy}
            onRefresh={onRefreshSession}
            onOpenSession={onOpenSession}
            onRenameSession={onRenameSession}
            onDeleteSession={onDeleteSession}
            onLoadMoreSessions={onLoadMoreSessions}
            onOpenSessionSettings={onOpenSettings}
          />
          {busyOutsideTurn && (
            <p
              className="text-muted-foreground flex items-center justify-center gap-2 text-center text-xs"
              role="status"
            >
              <ThinkingOrb state={orbState} size={20} aria-hidden />
              {orbLabel(orbState)}
            </p>
          )}
          {!runtimeReady && (
            <p
              className="text-muted-foreground text-center text-sm"
              role="status"
            >
              Chat needs setup.{" "}
              <button
                type="button"
                className="text-foreground font-medium underline-offset-4 hover:underline"
                onClick={onOpenSettings}
              >
                Mở Cài đặt
              </button>
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
          citationPreviews={citationPreviews}
          onRequestCitationPreview={onRequestCitationPreview}
          orbState={orbState}
          orbLabel={orbLabel(orbState)}
          onLoadOlder={onLoadOlder}
          canLoadOlder={canLoadOlder}
        />
      </div>
      <div className="bg-background z-20 flex shrink-0 flex-col gap-2 px-6 pb-5">
        {/* Work that happens outside a turn — an index build, a memory
            compaction — has no message to sit inside, so the orb reports it
            here. A live turn shows its own orb in the transcript. */}
        {busyOutsideTurn && (
          <div className="text-muted-foreground mx-auto flex w-full max-w-2xl items-center gap-2 text-xs">
            <ThinkingOrb state={orbState} size={20} aria-hidden />
            {orbLabel(orbState)}
          </div>
        )}
        <div className="mx-auto w-full max-w-2xl">{composer}</div>
        <SessionContext
          session={session}
          history={sessionHistory}
          connected={connected}
          busy={sessionBusy}
          onRefresh={onRefreshSession}
          onOpenSession={onOpenSession}
          onRenameSession={onRenameSession}
          onDeleteSession={onDeleteSession}
          onLoadMoreSessions={onLoadMoreSessions}
          onOpenSessionSettings={onOpenSettings}
        />
      </div>
    </>
  );
}
