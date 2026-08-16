/**
 * A citation you can check without leaving the answer.
 *
 * §5.6.4 — Open WebUI's hover preview, applied where it matters most here:
 * groundedness is the top open blocker, and people only verify a citation when
 * verifying costs one hover instead of one navigation.
 *
 * The preview shows what the engine actually sent: path, title, line range, and
 * the retrieval backends and score that produced it. **It does not show the
 * passage text** — `citation_records` carries no snippet, and inventing one
 * would defeat the point of the affordance.
 *
 * Note the API: shadcn's `hover-card` is Base UI's `PreviewCard`, not Radix. It
 * has no `asChild`; the trigger takes a `render` element instead, and the open
 * delays are the primitive's defaults.
 */

import { Badge } from "@/components/ui/badge";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import type { Citation } from "@/engine/conversation";
import type { VaultDocument } from "@/engine/documents";
import { documentFor } from "@/engine/documents";

interface CitationChipProps {
  citation: Citation;
  documents: VaultDocument[];
}

export function CitationChip({ citation, documents }: CitationChipProps) {
  const label = String(citation.label ?? "?");
  const path = typeof citation.path === "string" ? citation.path : "";
  const title = typeof citation.title === "string" ? citation.title : null;
  const start = typeof citation.line_start === "number" ? citation.line_start : null;
  const end = typeof citation.line_end === "number" ? citation.line_end : null;
  const document = documentFor(documents, citation);

  return (
    <HoverCard>
      <HoverCardTrigger
        render={
          <Badge
            variant="outline"
            className="hover:border-primary/60 hover:text-primary cursor-default font-mono text-[10px] transition-colors"
          >
            {label}
          </Badge>
        }
      />
      <HoverCardContent align="start" className="w-80">
        <div className="flex flex-col gap-2">
          <div className="flex flex-col gap-0.5">
            <span className="text-sm font-medium">{title ?? "Untitled"}</span>
            <span className="text-muted-foreground font-mono text-[10px] break-all">{path}</span>
          </div>

          <div className="text-muted-foreground flex flex-wrap gap-2 text-[10px]">
            {start !== null && end !== null && (
              <span>
                lines {start}–{end}
              </span>
            )}
            {typeof citation.source === "string" && <span>{citation.source}</span>}
            {document !== null && document.score > 0 && (
              <span>score {document.score.toFixed(3)}</span>
            )}
          </div>

          {document !== null && document.backends.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {document.backends.map((backend) => (
                <Badge key={backend} variant="secondary" className="text-[10px]">
                  {backend}
                </Badge>
              ))}
            </div>
          )}

          <p className="text-muted-foreground border-border/60 border-t pt-2 text-[10px] leading-relaxed">
            The engine sends the location of the evidence, not its text. Open the
            file to read the passage.
          </p>
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}
