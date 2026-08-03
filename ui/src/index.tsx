import { useState } from "react";
import { render } from "ink";
import { App } from "./App.js";
import type { Mode } from "./store.js";

interface CliArgs {
  mode: Mode | null;
  profile?: string;
  noModel: boolean;
  vault?: string;
  sessionPersistence: "ram_only" | "local_resumable";
  sessionId: string;
  resumeSession: boolean;
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = {
    mode: null,
    noModel: false,
    sessionPersistence: "ram_only",
    sessionId: "default",
    resumeSession: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg) continue;
    if (
      arg === "chat" ||
      arg === "voice" ||
      arg === "status" ||
      arg === "settings"
    )
      args.mode = arg;
    else if (arg === "--no-model") args.noModel = true;
    else if (arg === "--vault" && argv[index + 1]) {
      args.vault = argv[index + 1];
      index += 1;
    }
    else if (
      arg === "--session-persistence" &&
      (argv[index + 1] === "ram_only" || argv[index + 1] === "local_resumable")
    ) {
      const persistence = argv[index + 1];
      if (persistence !== "ram_only" && persistence !== "local_resumable")
        continue;
      args.sessionPersistence = persistence;
      index += 1;
    }
    else if (arg === "--session-id" && argv[index + 1]) {
      const sessionId = argv[index + 1];
      if (!sessionId) continue;
      args.sessionId = sessionId;
      index += 1;
    }
    else if (arg === "--resume-session") args.resumeSession = true;
    else if (!arg.startsWith("-") && !args.profile && args.mode)
      args.profile = arg;
    else if (!arg.startsWith("-") && !args.mode) args.mode = null;
  }
  return args;
}

function Root({ args }: { args: CliArgs }) {
  // A bare launch opens the setup surface first. It reports whether the
  // repository Knowledge vault is initialized and keeps chat/voice behind the
  // same explicit settings gate as the quick chat/voice forms.
  const [target] = useState<Mode>(args.mode ?? "settings");
  return (
    <App
      target={target}
      profile={args.profile}
      noModel={args.noModel}
      vault={args.vault}
      sessionPersistence={args.sessionPersistence}
      sessionId={args.sessionId}
      resumeSession={args.resumeSession}
    />
  );
}

const args = parseArgs(process.argv.slice(2));
render(<Root args={args} />, { exitOnCtrlC: true });
