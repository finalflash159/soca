/**
 * The composer, and the two keyboard affordances the plan singles out.
 *
 * §5.6.7 — `/` runs a command, `@` references a vault document. For a wiki
 * assistant the second one matters most: it turns "choose a source" from a
 * settings decision into a keystroke mid-sentence.
 *
 * §5.6.3 — the model picker moved out of the header and down beside the input.
 * SoCa's equivalent is the active provider/model and the ASR profile, so those
 * sit under the box rather than three tabs away.
 *
 * The `@` list only covers documents seen this session, because the protocol
 * has no vault listing (see `engine/documents.ts`). The footer says so rather
 * than implying a browse that does not exist.
 */

import { useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import type { SlashCommand, VaultDocument } from "@/engine/documents";
import { applyMention, filterCommands, mentionQuery, slashQuery } from "@/engine/documents";

interface ComposerProps {
  connected: boolean;
  documents: VaultDocument[];
  /** Rendered under the input — active model and profile (§5.6.3). */
  contextChips: Array<{ label: string; value: string }>;
  onSend: (text: string) => void;
  onCommand: (command: SlashCommand) => void;
}

export function Composer({
  connected,
  documents,
  contextChips,
  onSend,
  onCommand,
}: ComposerProps) {
  const [draft, setDraft] = useState("");
  const [caret, setCaret] = useState(0);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const slash = slashQuery(draft);
  const mention = mentionQuery(draft, caret);
  const commands = slash === null ? [] : filterCommands(slash);
  const matches =
    mention === null
      ? []
      : documents
          .filter((document) =>
            document.path.toLowerCase().includes(mention.query.toLowerCase()),
          )
          .slice(0, 8);
  const paletteOpen = commands.length > 0 || mention !== null;

  const submit = () => {
    const text = draft.trim();
    if (text === "" || !connected) {
      return;
    }
    onSend(text);
    setDraft("");
    setCaret(0);
  };

  const pickDocument = (path: string) => {
    const next = applyMention(draft, caret, path);
    setDraft(next.text);
    setCaret(next.caret);
    inputRef.current?.focus();
  };

  const runCommand = (command: SlashCommand) => {
    onCommand(command);
    setDraft("");
    setCaret(0);
  };

  return (
    <div className="relative flex flex-col gap-2">
      {paletteOpen && (
        <div className="bg-popover border-border absolute bottom-full mb-2 w-full overflow-hidden rounded-lg border shadow-lg">
          <Command shouldFilter={false}>
            <CommandList className="max-h-56">
              {commands.length > 0 && (
                <CommandGroup heading="Commands">
                  {commands.map((command) => (
                    <CommandItem
                      key={command.id}
                      value={command.id}
                      onSelect={() => runCommand(command)}
                    >
                      <span className="font-mono text-xs">{command.label}</span>
                      <span className="text-muted-foreground ml-2 text-xs">{command.hint}</span>
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}
              {mention !== null && (
                <CommandGroup heading="Documents seen this session">
                  {matches.length === 0 ? (
                    <CommandEmpty>
                      Nothing retrieved yet. `@` completes over documents this
                      session has already seen — the engine exposes no vault
                      listing.
                    </CommandEmpty>
                  ) : (
                    matches.map((document) => (
                      <CommandItem
                        key={document.path}
                        value={document.path}
                        onSelect={() => pickDocument(document.path)}
                      >
                        <span className="truncate font-mono text-xs">{document.path}</span>
                        {document.title !== null && (
                          <span className="text-muted-foreground ml-2 truncate text-xs">
                            {document.title}
                          </span>
                        )}
                      </CommandItem>
                    ))
                  )}
                </CommandGroup>
              )}
            </CommandList>
          </Command>
        </div>
      )}

      <div className="flex items-end gap-2 px-3 py-2">
        <textarea
          ref={inputRef}
          className="max-h-40 min-h-[24px] flex-1 resize-none bg-transparent text-sm leading-relaxed outline-none disabled:opacity-50"
          value={draft}
          rows={1}
          placeholder={connected ? "Hỏi gì đó…   /  lệnh   ·   @  tài liệu" : "Engine is not running"}
          disabled={!connected}
          aria-label="Message"
          onChange={(event) => {
            setDraft(event.target.value);
            setCaret(event.target.selectionStart ?? event.target.value.length);
          }}
          onKeyUp={(event) => setCaret(event.currentTarget.selectionStart ?? 0)}
          onClick={(event) => setCaret(event.currentTarget.selectionStart ?? 0)}
          onKeyDown={(event) => {
            // The palette owns Enter while it is open, so picking an item does
            // not also send the half-typed message.
            if (event.key === "Enter" && !event.shiftKey && !paletteOpen) {
              event.preventDefault();
              submit();
            }
            if (event.key === "Escape") {
              setDraft("");
            }
          }}
        />
        <Button size="sm" onClick={submit} disabled={!connected || draft.trim() === ""}>
          Send
        </Button>
      </div>

      {contextChips.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 px-1">
          {contextChips.map((chip) => (
            <Badge
              key={chip.label}
              variant="outline"
              className="text-muted-foreground gap-1 border-dashed font-normal"
            >
              <span className="text-[10px] uppercase tracking-wide">{chip.label}</span>
              <span className="font-mono text-[10px]">{chip.value}</span>
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}
