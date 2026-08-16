/**
 * Light / dark, and the machinery to follow the OS.
 *
 * The class goes on `<html>` because three separate consumers look for it and
 * none of them can be told about a React context: Tailwind's `dark:` variant,
 * shadcn's tokens, and `thinking-orbs`, whose `auto` theme watches ancestors
 * with a `MutationObserver`. One class keeps all three in step.
 */

import { useCallback, useEffect, useState } from "react";

export type ThemeChoice = "system" | "light" | "dark";

const STORAGE_KEY = "soca.theme";

function isDarkChoice(choice: ThemeChoice): boolean {
  if (choice === "system") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  return choice === "dark";
}

function apply(choice: ThemeChoice): void {
  document.documentElement.classList.toggle("dark", isDarkChoice(choice));
}

function stored(): ThemeChoice {
  const value = localStorage.getItem(STORAGE_KEY);
  return value === "light" || value === "dark" || value === "system" ? value : "system";
}

/**
 * Apply the saved theme before React mounts.
 *
 * Called from `main.tsx`. Without it the first paint uses the CSS default and
 * a dark-mode user sees a white flash on every launch.
 */
export function initTheme(): void {
  apply(stored());
}

export function useTheme() {
  const [choice, setChoice] = useState<ThemeChoice>(stored);

  useEffect(() => {
    apply(choice);
    localStorage.setItem(STORAGE_KEY, choice);
  }, [choice]);

  // Only `system` follows the OS, and only while it is selected — an explicit
  // choice must survive the user changing their OS appearance.
  useEffect(() => {
    if (choice !== "system") {
      return;
    }
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => apply("system");
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, [choice]);

  /** Flip to the opposite of what is on screen, whatever produced it. */
  const toggle = useCallback(() => {
    setChoice(document.documentElement.classList.contains("dark") ? "light" : "dark");
  }, []);

  return { choice, setChoice, toggle };
}
