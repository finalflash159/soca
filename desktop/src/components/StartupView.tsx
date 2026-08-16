/**
 * What the window is before the engine is running.
 *
 * LiveKit's reference app switches between two whole views on connection state
 * rather than showing disabled chrome (`view-controller.tsx`: welcome ↔
 * session). Their welcome view is an icon, one line and one button — nothing
 * else competes with the single decision to be made.
 *
 * SoCa needs one extra field, because the app shells out to `soca` rather than
 * bundling it. That field belongs here, at the moment it matters, not in the
 * permanent header where it made the product look like a debug console.
 */

import { useState } from "react";
import { ThinkingOrb } from "thinking-orbs";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";

interface StartupViewProps {
  /** Set while a launch is in flight. */
  starting: boolean;
  /** Populated after a failed launch or an unclean exit. */
  problem: string | null;
  onStart: (program: string) => void;
}

export function StartupView({ starting, problem, onStart }: StartupViewProps) {
  const [program, setProgram] = useState("soca");
  const [showAdvanced, setShowAdvanced] = useState(false);

  return (
    <div className="flex h-full flex-col items-center justify-center gap-6 px-6">
      <ThinkingOrb state={starting ? "connecting" : "breathing"} size={64} />

      <div className="flex flex-col items-center gap-1 text-center">
        <h1 className="text-lg font-medium tracking-tight">SoCa</h1>
        <p className="text-muted-foreground max-w-xs text-sm leading-relaxed">
          Trợ lý tiếng Việt chạy trên máy bạn. Nhấn để khởi động engine.
        </p>
      </div>

      <Button size="lg" disabled={starting} onClick={() => onStart(program)}>
        {starting ? "Đang khởi động…" : "Khởi động"}
      </Button>

      {problem !== null && (
        <p className="text-destructive max-w-sm text-center text-xs leading-relaxed">{problem}</p>
      )}

      <div className="flex flex-col items-center gap-2">
        <button
          type="button"
          className="text-muted-foreground hover:text-foreground text-xs underline-offset-4 hover:underline"
          onClick={() => setShowAdvanced((open) => !open)}
        >
          {showAdvanced ? "Ẩn tuỳ chọn" : "Engine không chạy được?"}
        </button>

        {showAdvanced && (
          <div className="flex flex-col items-center gap-1">
            <Label htmlFor="engine-program" className="text-muted-foreground text-xs">
              Lệnh chạy engine
            </Label>
            <input
              id="engine-program"
              className="border-input bg-card/60 h-8 w-64 rounded-md border px-3 text-center font-mono text-xs"
              value={program}
              onChange={(event) => setProgram(event.target.value)}
              disabled={starting}
            />
            <p className="text-muted-foreground max-w-xs text-center text-[10px] leading-relaxed">
              App gọi <code>{program} engine</code>. Bản đóng gói chưa kèm Python nên{" "}
              <code>soca</code> phải có sẵn trên PATH.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
