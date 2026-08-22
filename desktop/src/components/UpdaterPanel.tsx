import { Download, LoaderCircle, RefreshCw } from "lucide-react";
import { useState } from "react";

import { Field, Section } from "@/components/Page";
import { Button } from "@/components/ui/button";
import { relaunch } from "@tauri-apps/plugin-process";
import { check, type Update } from "@tauri-apps/plugin-updater";

type UpdateState =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "current" }
  | { kind: "available"; update: Update }
  | { kind: "installing"; received: number; total: number | null }
  | { kind: "error"; message: string };

function downloadLabel(received: number, total: number | null): string {
  if (total === null || total <= 0) return "Đang tải bản cập nhật…";
  return `Đang tải bản cập nhật… ${Math.min(100, Math.round((received / total) * 100))}%`;
}

/** Release-only updater surface; developer builds expose their missing config truthfully. */
export function UpdaterPanel() {
  const [state, setState] = useState<UpdateState>({ kind: "idle" });

  const checkForUpdate = async () => {
    if (state.kind === "available") await state.update.close();
    setState({ kind: "checking" });
    try {
      const update = await check();
      setState(update === null ? { kind: "current" } : { kind: "available", update });
    } catch (error) {
      setState({
        kind: "error",
        message:
          error instanceof Error
            ? error.message
            : "Không thể kiểm tra cập nhật cho bản cài đặt này.",
      });
    }
  };

  const installUpdate = async (update: Update) => {
    let received = 0;
    let total: number | null = null;
    setState({ kind: "installing", received, total });
    try {
      await update.downloadAndInstall((event) => {
        if (event.event === "Started") total = event.data.contentLength ?? null;
        if (event.event === "Progress") received += event.data.chunkLength;
        setState({ kind: "installing", received, total });
      });
      await relaunch();
    } catch (error) {
      setState({
        kind: "error",
        message:
          error instanceof Error
            ? error.message
            : "Không thể cài bản cập nhật; phiên và dữ liệu trên máy vẫn được giữ nguyên.",
      });
    }
  };

  const pending = state.kind === "checking" || state.kind === "installing";
  const notice =
    state.kind === "current"
      ? "Bạn đang dùng bản mới nhất."
      : state.kind === "available"
        ? `Có bản ${state.update.version} (đang dùng ${state.update.currentVersion}).`
        : state.kind === "installing"
          ? downloadLabel(state.received, state.total)
          : state.kind === "error"
            ? `Chưa có update khả dụng: ${state.message}`
            : "Chỉ kiểm tra bản phát hành đã ký qua HTTPS; không gửi nội dung phiên hay API key.";

  return (
    <Section
      icon={Download}
      title="Cập nhật ứng dụng"
      description="Bản cập nhật không xoá phiên, vault hay cấu hình trên máy."
      actions={
        <Button size="sm" variant="outline" disabled={pending} onClick={() => void checkForUpdate()}>
          {state.kind === "checking" ? (
            <LoaderCircle className="animate-spin" aria-hidden="true" />
          ) : (
            <RefreshCw aria-hidden="true" />
          )}
          Kiểm tra
        </Button>
      }
    >
      <Field label="Trạng thái">
        {state.kind === "available" ? (
          <div className="border-border flex items-center justify-between gap-3 rounded-lg border p-3">
            <span className="text-sm font-medium">Sẵn sàng cài {state.update.version}</span>
            <Button size="sm" disabled={pending} onClick={() => void installUpdate(state.update)}>
              Cài và mở lại
            </Button>
          </div>
        ) : (
          <p
            className={
              state.kind === "error" ? "text-destructive text-sm" : "text-muted-foreground text-sm"
            }
            role={state.kind === "error" ? "alert" : "status"}
          >
            {notice}
          </p>
        )}
      </Field>
    </Section>
  );
}
