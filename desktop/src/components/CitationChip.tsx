/** A keyboard-accessible, engine-verified view of one grounded source. */

import { FileWarning, LoaderCircle, ShieldCheck, TriangleAlert } from "lucide-react";

import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { Citation } from "@/engine/conversation";
import { citationKey, type CitationPreviewState } from "@/engine/citation-preview";

interface CitationChipProps {
  citation: Citation;
  previews: Record<string, CitationPreviewState>;
  onRequestPreview: (citation: Citation) => Promise<boolean>;
}

function string(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function lines(start: number | null, end: number | null): string | null {
  return start !== null && end !== null ? `dòng ${start}–${end}` : null;
}

function PreviewStatus({ preview }: { preview: CitationPreviewState }) {
  if (preview.status === "loading") {
    return (
      <p className="text-muted-foreground flex items-center gap-2 text-sm" role="status">
        <LoaderCircle className="size-4 animate-spin" aria-hidden /> Đang kiểm tra nguồn…
      </p>
    );
  }
  if (preview.status === "current") {
    return (
      <p className="text-muted-foreground flex items-center gap-2 text-sm" role="status">
        <ShieldCheck className="text-chart-3 size-4" aria-hidden /> Nguồn hiện tại khớp với phiên bản đã dùng.
      </p>
    );
  }
  if (preview.status === "changed") {
    return (
      <p className="border-chart-4/40 bg-chart-4/10 flex items-start gap-2 rounded-lg border p-2.5 text-sm" role="alert">
        <TriangleAlert className="text-chart-4 mt-0.5 size-4 shrink-0" aria-hidden />
        Nguồn đã thay đổi kể từ lượt trả lời. Đoạn bên dưới là bản hiện tại, không phải bằng chứng nguyên gốc.
      </p>
    );
  }
  if (preview.status === "unverified") {
    return (
      <p className="border-border bg-muted/50 rounded-lg border p-2.5 text-sm" role="status">
        Phiên cũ không lưu fingerprint nguồn; chỉ có thể xem bản hiện tại.
      </p>
    );
  }
  return (
    <p className="border-destructive/40 bg-destructive/5 flex items-start gap-2 rounded-lg border p-2.5 text-sm" role="alert">
      <FileWarning className="text-destructive mt-0.5 size-4 shrink-0" aria-hidden />
      {preview.status === "missing"
        ? "Không còn tìm thấy tệp nguồn trong knowledge vault."
        : "Không thể đọc nguồn này trong knowledge vault hiện tại."}
      {preview.errorCode !== null && <span className="sr-only"> Mã lỗi: {preview.errorCode}.</span>}
    </p>
  );
}

export function CitationChip({ citation, previews, onRequestPreview }: CitationChipProps) {
  const label = string(citation.label) ?? "?";
  const path = string(citation.path) ?? "Không rõ đường dẫn";
  const title = string(citation.title) ?? path;
  const fallbackStart = typeof citation.line_start === "number" ? citation.line_start : null;
  const fallbackEnd = typeof citation.line_end === "number" ? citation.line_end : null;
  const preview = previews[citationKey(citation)] ?? {
    requestId: null,
    status: "idle" as const,
    title: null,
    lineStart: fallbackStart,
    lineEnd: fallbackEnd,
    passage: null,
    errorCode: null,
  };
  const range = lines(preview.lineStart ?? fallbackStart, preview.lineEnd ?? fallbackEnd);

  return (
    <Dialog onOpenChange={(open) => open && void onRequestPreview(citation)}>
      <DialogTrigger
        render={
          <Button
            type="button"
            size="xs"
            variant="outline"
            className="h-6 rounded-md px-1.5 font-mono text-[10px]"
            aria-label={`Kiểm tra nguồn ${label}: ${title}`}
          >
            {label}
          </Button>
        }
      />
      <DialogContent showCloseButton={false} className="max-h-[min(42rem,calc(100vh-2rem))] max-w-xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{preview.title ?? title}</DialogTitle>
          <DialogDescription className="font-mono text-xs break-all">{path}</DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3">
          {range !== null && <Badge variant="secondary" className="w-fit text-xs">{range}</Badge>}
          {preview.status !== "idle" && <PreviewStatus preview={preview} />}
          {preview.status === "idle" && (
            <p className="text-muted-foreground text-sm" role="status">Đang chuẩn bị kiểm tra nguồn…</p>
          )}
          {preview.passage !== null && (
            <pre className="border-border bg-muted/40 max-h-72 overflow-auto rounded-lg border p-3 font-sans text-sm leading-6 whitespace-pre-wrap">
              {preview.passage}
            </pre>
          )}
        </div>

        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => void onRequestPreview(citation)}>
            Kiểm tra lại
          </Button>
          <DialogClose render={<Button type="button" />}>Đóng</DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
