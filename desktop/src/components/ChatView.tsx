/** Shared, paged rendering of the engine-owned chat and voice transcript. */

import { Mic } from "lucide-react";

import { ActivityOrb } from "@/components/ActivityOrb";
import { Conversation, ConversationContent } from "@/components/ai-elements/conversation";
import { AnswerBody } from "@/components/AnswerBody";
import { CitationChip } from "@/components/CitationChip";
import { TurnSteps } from "@/components/TurnSteps";
import { Button } from "@/components/ui/button";
import type { Citation, ConversationState, Turn } from "@/engine/conversation";
import { blockedReason, phaseLabel, turnStatus, turnText } from "@/engine/conversation";
import type { CitationPreviewIndex } from "@/engine/citation-preview";
import type { OrbState } from "@/engine/orb";

interface ChatViewProps {
  conversation: ConversationState;
  citationPreviews: CitationPreviewIndex;
  onRequestCitationPreview: (citation: Citation) => Promise<boolean>;
  /** Drives the orb on the turn that is still running. */
  orbState: OrbState;
  orbLabel: string;
  onLoadOlder: () => void;
  canLoadOlder: boolean;
}

function AssistantTurn({
  turn,
  citationPreviews,
  onRequestCitationPreview,
  orbState,
  orbLabel,
  live,
}: {
  turn: Turn;
  citationPreviews: CitationPreviewIndex;
  onRequestCitationPreview: (citation: Citation) => Promise<boolean>;
  orbState: OrbState;
  orbLabel: string;
  /** Only the newest open turn shows the orb; older ones are settled. */
  live: boolean;
}) {
  const status = turnStatus(turn);
  const text = turnText(turn);

  // A repair is the engine asking again after rejecting an utterance (docs/18
  // §5). It replaces the answer and is a turn outcome, so it renders as speech
  // rather than as a blocked or failed state.
  if (turn.repair !== null) {
    return (
      <div className="mr-10 flex flex-col gap-1.5">
        <p className="text-[15px] leading-7">{turn.repair}</p>
        <span className="text-muted-foreground text-[10px]">chưa nghe rõ</span>
      </div>
    );
  }

  const rule =
    status === "failed"
      ? "border-destructive/70 border-l-2 pl-4"
      : status === "blocked"
        ? "border-muted-foreground/40 border-l-2 border-dashed pl-4"
        : "";

  return (
    <div className={`mr-10 flex flex-col gap-2 ${rule}`}>
      <TurnSteps steps={turn.steps} running={status === "streaming"} />

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
          {live && <ActivityOrb state={orbState} size={22} />}
          <span className="text-muted-foreground text-sm">
            {live ? orbLabel : phaseLabel(turn.phase)}
          </span>
        </div>
      )}

      {(status === "achieved" || (status === "streaming" && text !== "")) && (
        <div className="relative">
          <AnswerBody text={text} />
          {status === "streaming" && (
            <span
              className="bg-primary ml-0.5 inline-block h-4 w-[2px] align-text-bottom"
              aria-hidden
            />
          )}
        </div>
      )}

      {turn.interrupted && (
        // Barge-in cut the answer short. What was said stands; saying so is the
        // difference between an incomplete answer and a wrong one.
        <span className="text-muted-foreground text-[10px]">bị ngắt giữa chừng</span>
      )}

      {(turn.citations.length > 0 || turn.route !== null) && (
        <div className="flex flex-wrap items-center gap-2">
          {turn.citations.map((citation, index) => (
            <CitationChip
              key={index}
              citation={citation}
              previews={citationPreviews}
              onRequestPreview={onRequestCitationPreview}
            />
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

export function ChatView({
  conversation,
  citationPreviews,
  onRequestCitationPreview,
  orbState,
  orbLabel,
  onLoadOlder,
  canLoadOlder,
}: ChatViewProps) {
  const lastIndex = conversation.turns.length - 1;

  return (
    <Conversation className="min-h-0 flex-1">
      <ConversationContent className="mx-auto w-full max-w-2xl px-1 pb-4">
        {conversation.nextTurnCursor !== null && (
          <div className="flex flex-col items-center gap-2" aria-live="polite">
            <Button
              type="button"
              size="sm"
              variant="outline"
              disabled={!canLoadOlder || conversation.turnPageLoadState === "loading"}
              onClick={onLoadOlder}
            >
              {conversation.turnPageLoadState === "loading" ? "Đang tải lượt cũ hơn…" : "Tải lượt cũ hơn"}
            </Button>
            {conversation.turnPageError !== null && (
              <p className="text-destructive text-xs" role="alert">
                {conversation.turnPageError}
              </p>
            )}
          </div>
        )}
        {conversation.reassemblyMismatch && (
          <div className="border-destructive/40 text-destructive rounded-md border px-3 py-2 text-xs">
            Các mảnh stream không ghép lại đúng câu trả lời cuối (đã bỏ qua khác biệt khoảng trắng)
            — có thể mất frame, hoặc phần “Nguồn:” chỉ bị cắt ở bản toàn văn. Đáng báo lại.
          </div>
        )}

        {conversation.turns.map((turn, index) => (
          <div key={`${turn.runId}-${index}`} className="flex flex-col gap-4">
            {/* A rejected utterance has no transcript to show — the engine
                declined to invent one — so there is no bubble, only the repair
                question below. An empty bubble would read as a lost message. */}
            {turn.userText !== "" && (
              <div className="flex justify-end">
                <p className="bg-card border-border/60 ml-10 flex max-w-[80%] items-start gap-2 rounded-2xl border px-4 py-2.5 text-[15px] leading-7 whitespace-pre-wrap">
                  {turn.surface === "voice" && (
                    <Mic
                      className="text-muted-foreground mt-1.5 size-3.5 shrink-0"
                      aria-label="Nói"
                    />
                  )}
                  {turn.userText}
                </p>
              </div>
            )}
            <AssistantTurn
              turn={turn}
              citationPreviews={citationPreviews}
              onRequestCitationPreview={onRequestCitationPreview}
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
