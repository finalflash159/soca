/**
 * Phase 2 surface: text conversation over the engine's `chat` stream.
 *
 * Rendering rules come from `docs/18-engine-protocol.md`:
 *
 * * Answers are plain speech-style text, not markdown — `SOCA_RUNTIME_SYSTEM_PROMPT`
 *   forbids markdown because this is spoken conversation. The registry's
 *   `MessageResponse` renders markdown, so it is deliberately unused here;
 *   `whitespace-pre-wrap` is the correct renderer for this content.
 * * Provenance comes from the structured `citations` list, never from parsing
 *   `[K1]` out of prose (§4).
 * * A `blocked` turn is a terminal outcome and gets its own presentation, not
 *   the error styling (§7 obligation 4).
 */

import { useState } from "react";

import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Message, MessageContent } from "@/components/ai-elements/message";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { Turn } from "@/engine/conversation";
import { blockedReason, phaseLabel, turnStatus, turnText } from "@/engine/conversation";
import type { ConversationState } from "@/engine/conversation";

interface ChatViewProps {
  conversation: ConversationState;
  connected: boolean;
  onSend: (text: string) => void;
}

function Citations({ turn }: { turn: Turn }) {
  if (turn.citations.length === 0) {
    return null;
  }
  return (
    <div className="mt-2 flex flex-wrap gap-1">
      {turn.citations.map((citation, index) => (
        <Badge key={index} variant="outline" className="font-mono text-[10px]">
          {String(citation.label ?? citation.path ?? index + 1)}
        </Badge>
      ))}
    </div>
  );
}

function AssistantTurn({ turn }: { turn: Turn }) {
  const status = turnStatus(turn);
  const text = turnText(turn);

  if (status === "failed") {
    return (
      <Message from="assistant">
        <MessageContent className="border-destructive/40 text-destructive border">
          {turn.error}
        </MessageContent>
      </Message>
    );
  }

  if (status === "blocked") {
    return (
      <Message from="assistant">
        <MessageContent className="border-muted-foreground/30 border border-dashed">
          <span className="text-muted-foreground text-sm">{blockedReason(turn)}</span>
          {text !== "" && <p className="mt-2 whitespace-pre-wrap">{text}</p>}
          <Citations turn={turn} />
        </MessageContent>
      </Message>
    );
  }

  if (status === "streaming" && text === "") {
    // A tool or retrieval turn publishes nothing until synthesis and
    // verification finish (§6), so the phase is the only honest signal here.
    return (
      <Message from="assistant">
        <MessageContent className="text-muted-foreground text-sm italic">
          {phaseLabel(turn.phase)}…
        </MessageContent>
      </Message>
    );
  }

  return (
    <Message from="assistant">
      <MessageContent>
        <p className="whitespace-pre-wrap">{text}</p>
        {status === "streaming" && (
          <span className="bg-foreground ml-0.5 inline-block h-4 w-[2px] animate-pulse align-text-bottom" />
        )}
        <Citations turn={turn} />
        {turn.route !== null && (
          <div className="text-muted-foreground mt-2 font-mono text-[10px]">
            route {turn.route}
            {turn.terminal !== null && ` · ${turn.terminal}`}
            {turn.deltaCount > 0 && ` · ${turn.deltaCount} chunk${turn.deltaCount === 1 ? "" : "s"}`}
          </div>
        )}
      </MessageContent>
    </Message>
  );
}

export function ChatView({ conversation, connected, onSend }: ChatViewProps) {
  const [draft, setDraft] = useState("");

  const submit = () => {
    const text = draft.trim();
    if (text === "" || !connected) {
      return;
    }
    onSend(text);
    setDraft("");
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      {conversation.reassemblyMismatch && (
        <div className="border-destructive/40 text-destructive rounded-md border px-3 py-2 text-xs">
          A streamed answer did not reassemble into the final text. The engine
          guarantees they match, so this is either an engine regression or a
          dropped frame — worth reporting rather than ignoring.
        </div>
      )}

      <Conversation className="min-h-0 flex-1">
        <ConversationContent>
          {conversation.turns.length === 0 ? (
            <ConversationEmptyState
              title="No turns yet"
              description={
                connected
                  ? "Ask something in Vietnamese."
                  : "Start the engine first."
              }
            />
          ) : (
            conversation.turns.map((turn, index) => (
              <div key={`${turn.runId}-${index}`} className="flex flex-col gap-2">
                <Message from="user">
                  <MessageContent>
                    <p className="whitespace-pre-wrap">{turn.userText}</p>
                  </MessageContent>
                </Message>
                <AssistantTurn turn={turn} />
              </div>
            ))
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      <div className="flex items-end gap-2">
        <textarea
          className="border-input bg-background focus-visible:ring-ring min-h-[44px] flex-1 resize-none rounded-md border px-3 py-2 text-sm shadow-xs focus-visible:ring-1 focus-visible:outline-none disabled:opacity-50"
          value={draft}
          rows={1}
          placeholder={connected ? "Hỏi gì đó…" : "Engine is not running"}
          disabled={!connected}
          aria-label="Message"
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
        />
        <Button onClick={submit} disabled={!connected || draft.trim() === ""}>
          Send
        </Button>
      </div>
    </div>
  );
}
