/**
 * The composer — and on a new conversation, the whole screen.
 *
 * The reference is the claude.ai new-chat screen: the input is not a strip at
 * the bottom of an empty page, it *is* the page. It is tall, wide, centred, and
 * everything that changes how a turn runs — model, microphone, voice mode —
 * lives on a row inside it. That is §5.6.3 taken further than a caption: the
 * controls are controls, not labels.
 *
 * The two keyboard affordances from §5.6.7 stay: `/` runs an engine command,
 * `@` references a vault document. `@` completes over documents seen this
 * session only, because the protocol has no vault listing
 * (`engine/documents.ts`), and the empty state says so.
 */

import { ArrowUp, ChevronDown, Mic, Paperclip } from "lucide-react";
import { useRef, useState } from "react";

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
import { cn } from "@/lib/utils";

interface ComposerProps {
  connected: boolean;
  /** Shown in the placeholder while the engine is still coming up. */
  starting?: boolean;
  documents: VaultDocument[];
  /** Active model, rendered as a control on the composer's own row. */
  model: string | null;
  /** `hero` is the tall new-conversation form; `docked` sits under a transcript. */
  variant?: "hero" | "docked";
  onSend: (text: string) => void;
  onCommand: (command: SlashCommand) => void;
  onEnterVoiceMode: () => void;
  onOpenSettings: () => void;
}

export function Composer({
  connected,
  starting = false,
  documents,
  model,
  variant = "docked",
  onSend,
  onCommand,
  onEnterVoiceMode,
  onOpenSettings,
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
          .filter((document) => document.path.toLowerCase().includes(mention.query.toLowerCase()))
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

  const placeholder = starting
    ? "Đang khởi động engine…"
    : connected
      ? "Mình giúp được gì?"
      : "Engine chưa chạy";

  return (
    <div className="relative">
      {paletteOpen && (
        <div className="bg-popover border-border absolute bottom-full mb-2 w-full overflow-hidden rounded-xl border shadow-xl">
          <Command shouldFilter={false}>
            <CommandList className="max-h-64">
              {commands.length > 0 && (
                <CommandGroup heading="Lệnh">
                  {commands.map((command) => (
                    <CommandItem
                      key={command.id}
                      value={command.id}
                      onSelect={() => {
                        onCommand(command);
                        setDraft("");
                        setCaret(0);
                      }}
                    >
                      <span className="font-mono text-xs">{command.label}</span>
                      <span className="text-muted-foreground ml-2 text-xs">{command.hint}</span>
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}
              {mention !== null && (
                <CommandGroup heading="Tài liệu phiên này đã thấy">
                  {matches.length === 0 ? (
                    <CommandEmpty>
                      Chưa truy xuất tài liệu nào. `@` chỉ gợi ý những gì phiên này đã thấy — engine
                      không có lệnh liệt kê vault.
                    </CommandEmpty>
                  ) : (
                    matches.map((document) => (
                      <CommandItem
                        key={document.path}
                        value={document.path}
                        onSelect={() => {
                          const next = applyMention(draft, caret, document.path);
                          setDraft(next.text);
                          setCaret(next.caret);
                          inputRef.current?.focus();
                        }}
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

      <div
        className={cn(
          "border-border/70 bg-card focus-within:border-primary/40 flex flex-col rounded-2xl border transition-colors",
          variant === "hero" ? "px-4 pt-4 pb-2" : "px-3 pt-3 pb-1.5",
        )}
      >
        <textarea
          ref={inputRef}
          className={cn(
            "resize-none bg-transparent leading-relaxed outline-none placeholder:opacity-60 disabled:opacity-50",
            variant === "hero" ? "min-h-[64px] text-base" : "max-h-40 min-h-[28px] text-sm",
          )}
          value={draft}
          rows={variant === "hero" ? 2 : 1}
          placeholder={placeholder}
          disabled={!connected}
          aria-label="Message"
          onChange={(event) => {
            setDraft(event.target.value);
            setCaret(event.target.selectionStart ?? event.target.value.length);
          }}
          onKeyUp={(event) => setCaret(event.currentTarget.selectionStart ?? 0)}
          onClick={(event) => setCaret(event.currentTarget.selectionStart ?? 0)}
          onKeyDown={(event) => {
            // The palette owns Enter while open, so choosing an item does not
            // also send the half-typed message.
            if (event.key === "Enter" && !event.shiftKey && !paletteOpen) {
              event.preventDefault();
              submit();
            }
            if (event.key === "Escape") {
              setDraft("");
            }
          }}
        />

        <div className="flex items-center gap-1 pt-1.5">
          {/* Model picker, bottom-left, exactly where the reference puts it: a
              live dot, the model name, and a chevron saying it is changeable.
              The old version was grey text in the corner that looked like a
              caption, so nobody discovered it opened settings. */}
          <button
            type="button"
            onClick={onOpenSettings}
            disabled={!connected}
            className="text-muted-foreground hover:bg-accent hover:text-foreground flex h-8 max-w-[18rem] items-center gap-2 rounded-lg px-2 text-xs transition-colors disabled:opacity-50"
            title="Đổi model"
          >
            <span
              className={cn(
                "size-1.5 shrink-0 rounded-full",
                connected ? "bg-chart-3" : "bg-muted-foreground/50",
              )}
              aria-hidden
            />
            <span className="truncate">{model ?? "Chưa nạp model"}</span>
            <ChevronDown className="size-3 shrink-0 opacity-60" />
          </button>

          <div className="ml-auto flex items-center gap-1">
            <Button
              size="sm"
              variant="ghost"
              className="text-muted-foreground hover:text-foreground size-8 rounded-lg p-0"
              title="Tài liệu — hoặc gõ @"
              aria-label="Chèn tài liệu"
              disabled={!connected}
              onClick={() => {
                setDraft((current) => `${current}@`);
                inputRef.current?.focus();
              }}
            >
              <Paperclip className="size-4" />
            </Button>
            {/* Goes to voice mode rather than toggling capture in place. This
                app has no dictation-into-the-box: a hot microphone with no
                screen to show for it was the confusing state. */}
            <Button
              size="sm"
              variant="ghost"
              className="text-muted-foreground hover:text-foreground size-8 rounded-lg p-0"
              title="Chế độ thoại"
              aria-label="Chế độ thoại"
              disabled={!connected}
              onClick={onEnterVoiceMode}
            >
              <Mic className="size-4" />
            </Button>
            <Button
              size="sm"
              className="size-8 rounded-full p-0"
              disabled={!connected || draft.trim() === ""}
              onClick={submit}
              title="Gửi"
              aria-label="Gửi"
            >
              <ArrowUp className="size-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
