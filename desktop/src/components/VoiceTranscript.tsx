/**
 * The conversation, inside voice mode.
 *
 * A thin wrapper over `ChatView` rather than a second renderer. A spoken turn
 * and a typed turn are the same `Turn` (see `engine/conversation.ts`), so they
 * must look the same wherever they are read — two renderers would drift, and
 * the whole point of unifying the transcript was that the history is one thing.
 *
 * What this adds is the one thing the chat surface has no equivalent for: the
 * utterance currently being recognised, shown as a pending user bubble. It is
 * not a turn yet — `voice/asr` has not landed — so it cannot live in the turn
 * list, and dropping it would make the screen look frozen while someone talks.
 */

import type { OrbState } from "thinking-orbs";

import { ChatView } from "@/components/ChatView";
import type { ConversationState } from "@/engine/conversation";
import type { VaultDocument } from "@/engine/documents";
import { orbLabel } from "@/engine/orb";
import type { VoiceState } from "@/engine/voice";
import { partialText } from "@/engine/voice";

interface VoiceTranscriptProps {
  conversation: ConversationState;
  documents: VaultDocument[];
  orbState: OrbState;
  voice: VoiceState;
}

export function VoiceTranscript({
  conversation,
  documents,
  orbState,
  voice,
}: VoiceTranscriptProps) {
  const partial = partialText(voice.partial);

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

      {partial !== "" && (
        <div className="mx-auto w-full max-w-2xl shrink-0 px-1 pb-2">
          <div className="flex justify-end">
            <p className="border-border/60 text-muted-foreground ml-10 max-w-[80%] rounded-2xl border border-dashed px-4 py-2.5 text-[15px] leading-7">
              <span className="text-foreground">{voice.partial?.committed}</span>{" "}
              <span>{voice.partial?.tentative}</span>
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
