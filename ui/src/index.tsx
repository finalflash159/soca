import { useState } from "react";
import { render } from "ink";
import { App, Splash } from "./App.js";
import type { Mode } from "./store.js";

interface CliArgs {
  mode: Mode | null;
  profile?: string;
  noModel: boolean;
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = { mode: null, noModel: false };
  for (const arg of argv) {
    if (arg === "chat" || arg === "voice" || arg === "status") args.mode = arg;
    else if (arg === "--no-model") args.noModel = true;
    else if (!arg.startsWith("-") && !args.profile && args.mode)
      args.profile = arg;
    else if (!arg.startsWith("-") && !args.mode) args.mode = null;
  }
  return args;
}

function Root({ args }: { args: CliArgs }) {
  const [mode, setMode] = useState<Mode | null>(args.mode);
  if (mode === null) return <Splash onDone={setMode} />;
  return <App mode={mode} profile={args.profile} noModel={args.noModel} />;
}

const args = parseArgs(process.argv.slice(2));
render(<Root args={args} />, { exitOnCtrlC: true });
