import { useEffect, useRef, useState } from "react";
import { Text, useInput } from "ink";
import { COLOR } from "./theme.js";

/** The subset of Ink's key flags needed by the text editor. */
export interface InputKey {
  backspace?: boolean;
  delete?: boolean;
  leftArrow?: boolean;
  rightArrow?: boolean;
  return?: boolean;
  tab?: boolean;
  upArrow?: boolean;
  downArrow?: boolean;
  escape?: boolean;
  ctrl?: boolean;
  meta?: boolean;
}

export interface InputState {
  value: string;
  /** Cursor position in grapheme clusters, not UTF-16 code units. */
  cursor: number;
}

export interface InputResult {
  state: InputState;
  changed: boolean;
  submit: boolean;
}

function debugInputEvent(input: string, key: InputKey): void {
  if (process.env["SOCA_INPUT_DEBUG"] !== "1") return;
  const activeKeys = Object.entries(key)
    .filter(([, active]) => active)
    .map(([name]) => name)
    .join(",");
  process.stderr.write(
    `[soca-input] data=${JSON.stringify(input)} keys=${activeKeys || "-"}\n`,
  );
}

function segmenter(): Intl.Segmenter | null {
  if (typeof Intl.Segmenter !== "function") return null;
  return new Intl.Segmenter("vi", { granularity: "grapheme" });
}

const VI_SEGMENTER = segmenter();

/** Split text in the same units a user sees, including combining marks. */
export function graphemes(value: string): string[] {
  if (VI_SEGMENTER) return [...VI_SEGMENTER.segment(value)].map((item) => item.segment);
  return [...value];
}

function clampCursor(value: string, cursor: number): number {
  return Math.max(0, Math.min(graphemes(value).length, cursor));
}

function offsetForCursor(value: string, cursor: number): number {
  return graphemes(value).slice(0, clampCursor(value, cursor)).join("").length;
}

function normalizeValue(value: string): string {
  return value.normalize("NFC");
}

function deleteBackward(state: InputState): InputState {
  const parts = graphemes(state.value);
  const cursor = clampCursor(state.value, state.cursor);
  if (cursor === 0) return { ...state, cursor: 0 };
  parts.splice(cursor - 1, 1);
  return { value: parts.join(""), cursor: cursor - 1 };
}

function insertText(state: InputState, text: string): InputState {
  if (!text) return state;
  const offset = offsetForCursor(state.value, state.cursor);
  const value = normalizeValue(
    state.value.slice(0, offset) + text + state.value.slice(offset),
  );
  return {
    value,
    cursor: graphemes(value.slice(0, offset + text.length)).length,
  };
}

/**
 * Remove an ANSI key sequence that arrived as part of a pasted/data chunk.
 * Ink normally parses a standalone sequence, but IMEs and terminals can
 * coalesce it with replacement text. It must never become prompt content.
 */
function stripEscapeSequences(input: string): string {
  return input
    .replace(/\u001b(?:\[[0-?]*[ -/]*[@-~]|O[ -~])/g, "")
    .replace(/\u001b/g, "");
}

function applyTextChunk(state: InputState, input: string): InputState {
  let next = state;
  const clean = stripEscapeSequences(input);
  for (const char of clean) {
    if (char === "\b" || char === "\u007f") {
      next = deleteBackward(next);
    } else if (char === "\r" || char === "\n" || char === "\t") {
      // This is a single-line prompt. A newline in a paste/replacement chunk
      // becomes a space instead of leaking a control byte into the renderer.
      next = insertText(next, " ");
    } else if (char < " ") {
      // Ignore remaining C0 controls. Ctrl+C is handled by Ink itself.
      continue;
    } else {
      // Segmenting the whole chunk preserves a + combining-mark pair as one
      // grapheme while still allowing embedded backspace replacement events.
      const cluster = graphemes(char)[0] ?? "";
      next = insertText(next, cluster);
    }
  }
  return next;
}

export function applyTerminalInput(
  state: InputState,
  input: string,
  key: InputKey,
): InputResult {
  const current: InputState = {
    value: normalizeValue(state.value),
    cursor: clampCursor(state.value, state.cursor),
  };

  if (key.return) return { state: current, changed: false, submit: true };
  // Ink 5 reports the terminal's usual backspace byte (\x7f) as
  // `key.delete`. The old ink-text-input component intentionally treated
  // both flags as backward-delete. Keep that behavior so IME replacement
  // sequences (delete old composition + insert new text) do not duplicate
  // characters. Ink 5 does not expose enough information here to distinguish
  // that byte from a physical forward-delete key.
  if (key.backspace || key.delete) {
    const next = deleteBackward(current);
    return { state: next, changed: next.value !== current.value, submit: false };
  }
  if (key.leftArrow) {
    const next = { ...current, cursor: Math.max(0, current.cursor - 1) };
    return { state: next, changed: false, submit: false };
  }
  if (key.rightArrow) {
    const next = {
      ...current,
      cursor: Math.min(graphemes(current.value).length, current.cursor + 1),
    };
    return { state: next, changed: false, submit: false };
  }
  if (
    key.tab ||
    key.upArrow ||
    key.downArrow ||
    key.escape ||
    key.ctrl ||
    key.meta
  ) {
    return { state: current, changed: false, submit: false };
  }

  const next = applyTextChunk(current, input);
  return { state: next, changed: next.value !== current.value, submit: false };
}

export interface ImeTextInputProps {
  value: string;
  placeholder?: string;
  focus?: boolean;
  mask?: string;
  showCursor?: boolean;
  onChange: (value: string) => void;
  onSubmit?: (value: string) => void;
}

/**
 * A small terminal editor that is deliberately independent of
 * `ink-text-input`. That package's raw-mode editor cannot represent IME
 * replacement/control chunks and is known to lose CJK composition. This
 * component keeps the cursor in grapheme units and treats the input stream as
 * an edit stream, which is also what Vietnamese Telex implementations need.
 */
export function ImeTextInput({
  value,
  placeholder = "",
  focus = true,
  mask,
  showCursor = true,
  onChange,
  onSubmit,
}: ImeTextInputProps) {
  const stateRef = useRef<InputState>({
    value,
    cursor: graphemes(value).length,
  });
  const [cursor, setCursor] = useState(stateRef.current.cursor);

  useEffect(() => {
    const current = stateRef.current;
    if (value !== current.value) {
      const next = { value, cursor: graphemes(value).length };
      stateRef.current = next;
      setCursor(next.cursor);
    }
  }, [value]);

  useInput(
    (input, key) => {
      debugInputEvent(input, key);
      const result = applyTerminalInput(stateRef.current, input, key);
      if (
        result.state.value === stateRef.current.value &&
        result.state.cursor === stateRef.current.cursor &&
        !result.submit
      ) {
        return;
      }
      stateRef.current = result.state;
      setCursor(result.state.cursor);
      if (result.submit) {
        onSubmit?.(result.state.value);
      } else if (result.changed) {
        onChange(result.state.value);
      }
    },
    { isActive: focus },
  );

  const parts = graphemes(value);
  const display = mask ? parts.map(() => mask) : parts;
  const visibleCursor = Math.max(0, Math.min(display.length, cursor));

  if (display.length === 0 && placeholder) {
    if (!showCursor || !focus) return <Text color={COLOR.muted}>{placeholder}</Text>;
    return (
      <Text color={COLOR.muted}>
        <Text inverse color={COLOR.muted}>{placeholder[0] ?? " "}</Text>
        {placeholder.slice(1)}
      </Text>
    );
  }

  return (
    <Text>
      {display.slice(0, visibleCursor).join("")}
      {showCursor && focus ? (
        <Text inverse>{display[visibleCursor] ?? " "}</Text>
      ) : null}
      {display.slice(visibleCursor + (showCursor && focus ? 1 : 0)).join("")}
    </Text>
  );
}
