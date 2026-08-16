/**
 * The conversation, deliberately quiet.
 *
 * §5.6.5 — Open WebUI's v0.11.0 spent a whole release on "where things live"
 * rather than new features, and narrowed the conversation column, lightened the
 * type and tightened spacing. The same applies here: a full-width column of
 * 14px text across a 1100px window is a wall, and SoCa's answers are spoken
 * sentences, not documents.
 *
 * So: a 46rem reading column, metadata at 10px in muted, and no chrome around a
 * message unless it carries meaning. The one place colour is spent is the
 * assistant's left rule — gold for an answer, dashed for a refusal, red for a
 * failure — because that distinction is the product's whole point.
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

function Meta({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-muted-foreground flex flex-wrap items-center gap-2 text-[10px]">
      {children}
    </div>
  );
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
      ? "border-destructive/70"
      : status === "blocked"
        ? "border-muted-foreground/40 border-dashed"
        : "border-primary/70";

  return (
    <div className={`border-l-2 pl-4 ${rule}`}>
      {status === "failed" && <p className="text-destructive text-sm">{turn.error}</p>}

      {status === "blocked" && (
        <div className="flex flex-col gap-1">
          <p className="text-sm">{blockedReason(turn)}</p>
          {text !== "" && <p className="text-muted-foreground text-sm">{text}</p>}
        </div>
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
        <div className="mt-2 flex flex-wrap items-center gap-2">
          {turn.citations.map((citation, index) => (
            <CitationChip key={index} citation={citation} documents={documents} />
          ))}
          {turn.route !== null && (
            <Meta>
              <span>{turn.route}</span>
              {turn.terminal !== null && <span>· {turn.terminal}</span>}
              {turn.deltaCount > 1 && <span>· {turn.deltaCount} chunks</span>}
            </Meta>
          )}
        </div>
      )}
    </div>
  );
}

export function ChatView({ conversation, documents, orbState, orbLabel }: ChatViewProps) {
  const lastIndex = conversation.turns.length - 1;
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {conversation.reassemblyMismatch && (
        <div className="border-destructive/40 text-destructive mx-auto mb-3 w-full max-w-2xl rounded-md border px-3 py-2 text-xs">
          A streamed answer did not reassemble into the final text. The engine
          guarantees they match, so this is an engine regression or a dropped
          frame — worth reporting rather than ignoring.
        </div>
      )}

      <Conversation className="min-h-0 flex-1">
        <ConversationContent className="mx-auto w-full max-w-2xl px-1">
          {conversation.turns.length === 0 ? (
            // An empty state that teaches the two keystrokes is worth more than
            // a centred sentence in an otherwise blank column (§5.6.7).
            <div className="flex h-full flex-col justify-center gap-6 py-16">
              <p className="text-muted-foreground text-sm">Hỏi gì đó bằng tiếng Việt.</p>
              <dl className="text-muted-foreground flex flex-col gap-2 text-xs">
                <div className="flex gap-3">
                  <dt className="text-foreground w-6 font-mono">/</dt>
                  <dd>Chạy lệnh engine — status, context, memory, index.</dd>
                </div>
                <div className="flex gap-3">
                  <dt className="text-foreground w-6 font-mono">@</dt>
                  <dd>
                    Trỏ tới tài liệu trong vault. Chỉ gợi ý tài liệu phiên này đã
                    thấy — engine không có lệnh liệt kê vault.
                  </dd>
                </div>
                <div className="flex gap-3">
                  <dt className="text-foreground w-6 font-mono">↵</dt>
                  <dd>Gửi. Shift+↵ xuống dòng.</dd>
                </div>
              </dl>
            </div>
          ) : (
            <div className="flex flex-col gap-8 py-6">
              {conversation.turns.map((turn, index) => (
                <div key={`${turn.runId}-${index}`} className="flex flex-col gap-3">
                  <p className="text-muted-foreground text-[15px] leading-7 whitespace-pre-wrap">
                    {turn.userText}
                  </p>
                  <AssistantTurn
                    turn={turn}
                    documents={documents}
                    orbState={orbState}
                    orbLabel={orbLabel}
                    live={index === lastIndex}
                  />
                </div>
              ))}
            </div>
          )}
        </ConversationContent>
      </Conversation>
    </div>
  );
}
