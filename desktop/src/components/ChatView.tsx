/**
 * The conversation.
 *
 * Standard chat geometry: the user's turn sits right, the assistant's left. The
 * user's is a bubble because it is a short, bounded utterance; the assistant's
 * is bare text on the page because it is prose meant to be read, and boxing a
 * paragraph only narrows it.
 *
 * §5.6.5 — a narrow reading column, metadata at 10px in muted, and no chrome
 * around an answer at all. An earlier revision drew a gold rule down the left of
 * every assistant turn; it decorated the ordinary case and drew the eye to
 * nothing. Only the two exceptional outcomes are marked now — a refusal and a
 * failure — because those must not be mistaken for an answer.
 *
 * Rendering rules from `docs/18-engine-protocol.md`: answers are plain speech
 * text, never markdown; provenance is the structured `citations` list; and a
 * `blocked` turn is a terminal outcome, not an error.
 */

import type { OrbState } from "thinking-orbs";
import { ThinkingOrb } from "thinking-orbs";

import { Conversation, ConversationContent } from "@/components/ai-elements/conversation";
import { CitationChip } from "@/components/CitationChip";
import type { ConversationState, Turn } from "@/engine/conversation";
import { blockedReason, phaseLabel, turnStatus, turnText } from "@/engine/conversation";
import type { VaultDocument } from "@/engine/documents";

interface ChatViewProps {
  conversation: ConversationState;
  documents: VaultDocument[];
  /** Drives the orb on the turn that is still running. */
  orbState: OrbState;
  orbLabel: string;
}

function AssistantTurn({
  turn,
  documents,
  orbState,
  orbLabel,
  live,
}: {
  turn: Turn;
  documents: VaultDocument[];
  orbState: OrbState;
  orbLabel: string;
  /** Only the newest open turn shows the orb; older ones are settled. */
  live: boolean;
}) {
  const status = turnStatus(turn);
  const text = turnText(turn);

  const rule =
    status === "failed"
      ? "border-destructive/70 border-l-2 pl-4"
      : status === "blocked"
        ? "border-muted-foreground/40 border-l-2 border-dashed pl-4"
        : "";

  return (
    <div className={`mr-10 flex flex-col gap-2 ${rule}`}>
      {status === "failed" && <p className="text-destructive text-sm">{turn.error}</p>}

      {status === "blocked" && (
        <>
          <p className="text-sm">{blockedReason(turn)}</p>
          {text !== "" && <p className="text-muted-foreground text-sm">{text}</p>}
        </>
      )}

      {status === "streaming" && text === "" && (
        // A tool or retrieval turn publishes nothing until synthesis and
        // verification finish (docs/18 §6). This is where the nine orb states
        // earn their place: the orb is the only thing telling the user whether
        // the system is planning, retrieving, running a tool or calling out.
        <div className="flex items-center gap-2.5 py-0.5">
          {live && <ThinkingOrb state={orbState} size={20} />}
          <span className="text-muted-foreground text-sm">
            {live ? orbLabel : phaseLabel(turn.phase)}
          </span>
        </div>
      )}

      {(status === "achieved" || (status === "streaming" && text !== "")) && (
        <p className="text-[15px] leading-7 whitespace-pre-wrap">
          {text}
          {status === "streaming" && (
            <span className="bg-primary ml-0.5 inline-block h-4 w-[2px] animate-pulse align-text-bottom" />
          )}
        </p>
      )}

      {(turn.citations.length > 0 || turn.route !== null) && (
        <div className="flex flex-wrap items-center gap-2">
          {turn.citations.map((citation, index) => (
            <CitationChip key={index} citation={citation} documents={documents} />
          ))}
          {turn.route !== null && (
            <span className="text-muted-foreground text-[10px]">
              {turn.route}
              {turn.terminal !== null && ` · ${turn.terminal}`}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export function ChatView({ conversation, documents, orbState, orbLabel }: ChatViewProps) {
  const lastIndex = conversation.turns.length - 1;

  return (
    <Conversation className="min-h-0 flex-1">
      <ConversationContent className="mx-auto w-full max-w-2xl px-1 pb-4">
        {conversation.reassemblyMismatch && (
          <div className="border-destructive/40 text-destructive rounded-md border px-3 py-2 text-xs">
            A streamed answer did not reassemble into the final text once
            whitespace is collapsed — a dropped frame, or a source footer the
            whole-answer cleaner removed. Worth reporting.
          </div>
        )}

        {conversation.turns.map((turn, index) => (
          <div key={`${turn.runId}-${index}`} className="flex flex-col gap-4">
            <div className="flex justify-end">
              <p className="bg-card border-border/60 ml-10 max-w-[80%] rounded-2xl border px-4 py-2.5 text-[15px] leading-7 whitespace-pre-wrap">
                {turn.userText}
              </p>
            </div>
            <AssistantTurn
              turn={turn}
              documents={documents}
              orbState={orbState}
              orbLabel={orbLabel}
              live={index === lastIndex}
            />
          </div>
        ))}
      </ConversationContent>
    </Conversation>
  );
}
