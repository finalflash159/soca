/**
 * The assistant's answer, rendered as what it actually is.
 *
 * It used to render as `whitespace-pre-wrap` plain text, on the strength of a
 * claim in this file's neighbour that answers are "plain speech text, never
 * markdown". That claim was wrong — it is nowhere in `docs/18`, and a captured
 * turn settles it. Measured 2026-08-16 against `openai/gpt-5.6-luna`, one
 * answer contained an ATX heading, a bullet list, a GFM table, a fenced
 * `python` block, inline `$…$` and display `$$…$$` math.
 *
 * So the old rendering showed `#` and `|` as literal characters, LaTeX as
 * source, and — because `pre-wrap` turns the `\n\n` between paragraphs into two
 * real blank lines *on top of* the line height — separated paragraphs by a
 * horizon.
 *
 * Three deliberate choices:
 *
 * * **No raw HTML.** `react-markdown` escapes HTML unless `rehype-raw` is
 *   added, and it is not added. Answer text is model output mixed with
 *   retrieved document text; neither is trusted enough to run as markup.
 * * **No `@tailwindcss/typography`.** Every element is styled here instead, so
 *   the answer inherits the app's own type scale and both palettes rather than
 *   a second, parallel set of colour decisions.
 * * **Streaming is safe as-is.** Chat chunks are whole guardrail-passed
 *   sentences (`docs/18` §6), not tokens, so the parser never sees a half word.
 *   A table that has only emitted its first row renders as text for one frame
 *   and then becomes a table; that flicker is cheaper than withholding the
 *   answer until the turn closes.
 *
 * What it costs, measured on this bundle (vite's own kB figures):
 *
 * ```text
 * baseline, plain text          480.55 kB
 * + markdown + KaTeX            919.19 kB   (+439)
 * + syntax highlighting        1094.61 kB   (+175)
 * ```
 *
 * Restricting `rehype-highlight` to a dozen named grammars was tried and saved
 * nothing at all — 1094.76 kB, a difference of 0.15 kB. The plugin imports
 * `lowlight`'s common set at module scope, so the default cannot be tree-shaken
 * and an explicit list is only ever added on top of it. The default is
 * therefore kept. This is a desktop app loading from disk, so 1 MB is a size,
 * not a latency.
 */

import "katex/dist/katex.min.css";

import type { ComponentPropsWithoutRef, ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import { cn } from "@/lib/utils";

/** Wide content scrolls inside itself; the reading column never scrolls sideways. */
function Scroller({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("-mx-1 overflow-x-auto px-1", className)}>{children}</div>;
}

/** A line that is nothing but `$$…$$`, which is how models write display math. */
const ONE_LINE_DISPLAY_MATH = /^[ \t]*\$\$(.+?)\$\$[ \t]*$/;
/** Opening or closing of a fenced code block, ``` or ~~~. */
const CODE_FENCE = /^[ \t]*(`{3,}|~{3,})/;

/**
 * Give one-line display math the block form `remark-math` requires.
 *
 * Measured, after a first guess that was wrong. `remark-math` v6 renders `$$…$$`
 * as **display** only when the fences sit on their own lines:
 *
 * ```text
 * $$x=1$$          → inline   (cramped, mid-sentence)
 * $$x=1$$ + space  → inline   (the trailing space was a red herring)
 * $$\nx=1\n$$      → display  ✓
 * ```
 *
 * Models write the first form, so a centred equation arrived squeezed into the
 * run of a sentence. Rewriting it to the third form is a pure notation change —
 * the maths is identical, only the block/inline classification differs.
 *
 * Fenced code is skipped. A `$$` inside a shell or LaTeX sample is content, not
 * markup, and rewriting it would corrupt the snippet.
 */
function expandDisplayMath(text: string): string {
  let inFence = false;
  return text
    .split("\n")
    .map((line) => {
      if (CODE_FENCE.test(line)) {
        inFence = !inFence;
        return line;
      }
      if (inFence) {
        return line;
      }
      const match = ONE_LINE_DISPLAY_MATH.exec(line);
      return match === null ? line : `$$\n${match[1].trim()}\n$$`;
    })
    .join("\n");
}

export function AnswerBody({ text, className }: { text: string; className?: string }) {
  return (
    <div
      className={cn(
        "text-[15px] leading-[1.7]",
        // Vertical rhythm lives here, once. Every block gets the same top
        // margin except the first, which is why the answer starts flush with
        // whatever sits above it.
        "[&>*]:mt-4 [&>*:first-child]:mt-0",
        className,
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, [rehypeHighlight, { detect: true, ignoreMissing: true }]]}
        components={{
          h1: ({ children }) => (
            <h1 className="mt-6 text-[19px] font-semibold tracking-tight">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="mt-6 text-[17px] font-semibold tracking-tight">{children}</h2>
          ),
          h3: ({ children }) => <h3 className="mt-5 text-[15px] font-semibold">{children}</h3>,
          h4: ({ children }) => <h4 className="mt-4 text-[15px] font-medium">{children}</h4>,

          p: ({ children }) => <p>{children}</p>,

          ul: ({ children }) => <ul className="list-disc space-y-1.5 pl-5">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal space-y-1.5 pl-5">{children}</ol>,
          // Nested lists must not inherit the block margin, or each level
          // pushes itself a line away from its own bullet.
          li: ({ children }) => (
            <li className="[&>*]:mt-2 [&>ol]:mt-1.5 [&>ul]:mt-1.5">{children}</li>
          ),

          blockquote: ({ children }) => (
            <blockquote className="border-border text-muted-foreground border-l-2 pl-4">
              {children}
            </blockquote>
          ),

          hr: () => <hr className="border-border" />,

          a: ({ children, href }) => (
            <a
              href={href}
              // Answers may cite the open web; a click must not navigate the
              // app shell out of existence inside a WebView.
              target="_blank"
              rel="noreferrer noopener"
              className="text-primary underline underline-offset-2"
            >
              {children}
            </a>
          ),

          strong: ({ children }) => <strong className="font-semibold">{children}</strong>,

          code: ({ className: language, children, ...props }: ComponentPropsWithoutRef<"code">) => {
            // `react-markdown` v10 gives inline code no `language-*` class, which
            // is the only reliable way to tell it from a fenced block here.
            const fenced = typeof language === "string" && language.includes("language-");
            if (!fenced) {
              return (
                <code className="bg-secondary rounded-[4px] px-1.5 py-0.5 font-mono text-[13px]">
                  {children}
                </code>
              );
            }
            return (
              <code className={cn(language, "font-mono text-[13px] leading-6")} {...props}>
                {children}
              </code>
            );
          },

          pre: ({ children }) => (
            <Scroller className="border-border bg-secondary/50 rounded-lg border">
              <pre className="p-3.5">{children}</pre>
            </Scroller>
          ),

          table: ({ children }) => (
            <Scroller>
              <table className="border-border w-full border-collapse overflow-hidden rounded-lg border text-sm">
                {children}
              </table>
            </Scroller>
          ),
          thead: ({ children }) => <thead className="bg-secondary/60">{children}</thead>,
          th: ({ children }) => (
            <th className="border-border border px-3 py-2 text-left font-medium">{children}</th>
          ),
          td: ({ children }) => (
            <td className="border-border border px-3 py-2 align-top">{children}</td>
          ),
        }}
      >
        {expandDisplayMath(text)}
      </ReactMarkdown>
    </div>
  );
}
