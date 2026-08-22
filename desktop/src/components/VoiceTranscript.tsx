/** Settled voice-mode turns rendered through the shared transcript. */

import type { OrbState } from "thinking-orbs";

import { ChatView } from "@/components/ChatView";
import type { Citation, ConversationState } from "@/engine/conversation";
import type { CitationPreviewIndex } from "@/engine/citation-preview";
import { orbLabel } from "@/engine/orb";

interface VoiceTranscriptProps {
  /** Settled turns only — `VoiceMode` removes the one still running. */
  conversation: ConversationState;
  citationPreviews: CitationPreviewIndex;
  onRequestCitationPreview: (citation: Citation) => Promise<boolean>;
  orbState: OrbState;
  onLoadOlder: () => void;
  canLoadOlder: boolean;
}

export function VoiceTranscript({
  conversation,
  citationPreviews,
  onRequestCitationPreview,
  orbState,
  onLoadOlder,
  canLoadOlder,
}: VoiceTranscriptProps) {
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
        citationPreviews={citationPreviews}
        onRequestCitationPreview={onRequestCitationPreview}
        orbState={orbState}
        orbLabel={orbLabel(orbState)}
        onLoadOlder={onLoadOlder}
        canLoadOlder={canLoadOlder}
      />
    </div>
  );
}
