/**
 * Rendered against a real answer.
 *
 * `__fixtures__/answer.md` is not written by hand: it is `chat/done.text` from
 * a live engine turn on 2026-08-16, captured through `soca engine` over stdin.
 * It exists because the previous renderer was built on a claim — "answers are
 * plain speech text, never markdown" — that nobody had checked. One captured
 * turn disproved it.
 */

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { AnswerBody } from "./AnswerBody";
import answer from "../engine/__fixtures__/answer.md?raw";

const html = renderToStaticMarkup(<AnswerBody text={answer} />);

describe("a real answer", () => {
  it("uses markdown, which is the whole reason this component exists", () => {
    expect(answer).toMatch(/^# /m);
    expect(answer).toMatch(/^\| /m);
    expect(answer).toMatch(/^```python/m);
    expect(answer).toMatch(/\$\$/);
  });

  it("renders the heading as a heading, not as a literal #", () => {
    expect(html).toContain("<h1");
    expect(html).not.toContain("# Attention");
  });

  it("renders the table as a table, not as pipes", () => {
    expect(html).toContain("<table");
    expect(html).toContain("<thead");
    expect(html).toContain("<td");
  });

  it("renders the fenced block as highlighted code", () => {
    expect(html).toContain("<pre");
    expect(html).toContain("language-python");
    // rehype-highlight ran: without it there are no token spans at all.
    expect(html).toContain("hljs-");
  });

  it("renders both display and inline math through KaTeX", () => {
    // The fixture writes display math as a single line, `$$…$$`, which
    // remark-math classifies as *inline* — `expandDisplayMath` is what turns it
    // back into a block. Without that step this assertion fails.
    expect(html).toContain("katex-display");
    // KaTeX deliberately keeps the TeX source in an <annotation> for MathML and
    // for copy-paste, so finding `\sqrt{d}` in the output is not evidence that
    // rendering failed. What proves it rendered is the typeset layer.
    expect(html).toContain("katex-html");
    expect(html).toContain("<math");
  });

  it("renders the bullet list as a list", () => {
    expect(html).toContain("<ul");
    expect(html).toContain("<li");
  });
});

describe("untrusted input", () => {
  it("escapes HTML rather than running it", () => {
    // Answer text is model output mixed with retrieved document text. Adding
    // `rehype-raw` would make either of them able to inject markup.
    const raw = renderToStaticMarkup(
      <AnswerBody text={'<img src=x onerror="alert(1)"> <script>alert(2)</script>'} />,
    );
    // The payload survives as *visible text*, which is the correct outcome:
    // escaped, not executed. Asserting the absence of the substring "onerror"
    // would be asserting the wrong thing — it is there, harmlessly, as prose.
    expect(raw).not.toContain("<script");
    expect(raw).not.toContain("<img");
    expect(raw).toContain("&lt;script&gt;");
    expect(raw).toContain("&lt;img");
  });

  it("opens links in a new context with no opener", () => {
    const link = renderToStaticMarkup(<AnswerBody text="[x](https://example.com)" />);
    expect(link).toContain('rel="noreferrer noopener"');
    expect(link).toContain('target="_blank"');
  });
});

describe("streaming", () => {
  it("renders a half-finished code fence without throwing", () => {
    // A chunk boundary can land inside a block. Chunks are whole sentences
    // (docs/18 §6), so this is rare, but it must degrade rather than crash.
    expect(() =>
      renderToStaticMarkup(<AnswerBody text={"Đang giải thích:\n\n```python\nx = 1"} />),
    ).not.toThrow();
  });

  it("renders an empty answer without throwing", () => {
    expect(() => renderToStaticMarkup(<AnswerBody text="" />)).not.toThrow();
  });
});

describe("display math normalisation", () => {
  const render = (text: string) => renderToStaticMarkup(<AnswerBody text={text} />);

  it("promotes a one-line $$…$$ to a block", () => {
    expect(render("Trước:\n\n$$x=1$$\n\nSau.")).toContain("katex-display");
  });

  it("leaves an already-fenced formula alone", () => {
    expect(render("Trước:\n\n$$\nx=1\n$$\n\nSau.")).toContain("katex-display");
  });

  it("keeps inline $…$ inline", () => {
    const html = render("Chia cho $\\sqrt{d}$ là được.");
    expect(html).toContain("katex");
    expect(html).not.toContain("katex-display");
  });

  it("does not touch $$ inside a code fence", () => {
    // A shell or LaTeX sample owns its own dollars; rewriting them would
    // silently corrupt the snippet the user is meant to copy.
    const html = render("```bash\n$$foo$$\n```");
    expect(html).toContain("language-bash");
    expect(html).not.toContain("katex");
  });
});
