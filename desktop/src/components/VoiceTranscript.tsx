/**
 * The record, inside voice mode.
 *
 * A thin wrapper over `ChatView` rather than a second renderer. A spoken turn
 * and a typed turn are the same `Turn` (see `engine/conversation.ts`), so they
 * must look the same wherever they are read — two renderers would drift, and
 * the whole point of unifying the transcript was that the history is one thing.
 *
 * It shows **settled turns only**. The turn in progress belongs centre stage
 * with the orb, where the partial transcript and the arriving answer are large
 * enough to follow without reading. `VoiceMode` makes that split, so this
 * component receives the list already trimmed and never handles live state.
 */

import type { OrbState } from "thinking-orbs";

import { ChatView } from "@/components/ChatView";
import type { ConversationState } from "@/engine/conversation";
import type { VaultDocument } from "@/engine/documents";
import { orbLabel } from "@/engine/orb";

interface VoiceTranscriptProps {
  /** Settled turns only — `VoiceMode` removes the one still running. */
  conversation: ConversationState;
  documents: VaultDocument[];
  orbState: OrbState;
}

export function VoiceTranscript({ conversation, documents, orbState }: VoiceTranscriptProps) {
  if (conversation.turns.length === 0) {
    return (
      <div className="flex shrink-0 items-start justify-center px-8 pb-4">
        <p className="text-muted-foreground text-sm">Chưa có lượt nào xong.</p>
      </div>
    );
  }

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      {/* Text dissolves at the top edge instead of clipping under the orb. */}
      <div className="from-background pointer-events-none absolute inset-x-0 top-0 z-10 h-10 bg-gradient-to-b to-transparent" />
      <ChatView
        conversation={conversation}
        documents={documents}
        orbState={orbState}
        orbLabel={orbLabel(orbState)}
      />
    </div>
  );
}
