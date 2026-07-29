import { spawn, type ChildProcessByStdio } from "node:child_process";
import { createInterface } from "node:readline";
import { EventEmitter } from "node:events";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Readable, Writable } from "node:stream";
import {
  parseEngineEvent,
  type EngineCommand,
  type EngineEvent,
} from "./protocol.js";

export interface EngineOptions {
  profile?: string;
  noModel?: boolean;
  vault?: string;
  sessionPersistence?: "ram_only" | "local_resumable";
  sessionId?: string;
  resumeSession?: boolean;
}

/** Locate the repo's soca executable relative to this file (ui/src or ui/dist). */
function defaultEngineCommand(options: EngineOptions): string[] {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const repoRoot = path.resolve(here, "..", "..");
  const venvSoca = path.join(repoRoot, ".venv", "bin", "soca");
  const base = existsSync(venvSoca) ? [venvSoca] : ["uv", "run", "soca"];
  const args = ["engine"];
  if (options.profile) args.push(options.profile);
  if (options.noModel) args.push("--no-model");
  if (options.vault) args.push("--vault", options.vault);
  if (options.sessionPersistence)
    args.push("--session-persistence", options.sessionPersistence);
  if (options.sessionId) args.push("--session-id", options.sessionId);
  if (options.resumeSession) args.push("--resume-session");
  return [...base, ...args];
}

/**
 * Child-process client for `soca engine`: commands go in as NDJSON on stdin,
 * events come back on stdout. Emits: "event" (EngineEvent), "stderr" (string),
 * "exit" (code).
 */
export class EngineClient extends EventEmitter {
  private child: ChildProcessByStdio<Writable, Readable, Readable> | null =
    null;

  start(options: EngineOptions = {}): void {
    const custom = process.env["SOCA_ENGINE_CMD"];
    const command = custom ? custom.split(" ") : defaultEngineCommand(options);
    const [executable, ...args] = command;
    if (!executable) throw new Error("empty engine command");

    const here = path.dirname(fileURLToPath(import.meta.url));
    const child = spawn(executable, args, {
      cwd: path.resolve(here, "..", ".."),
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child = child;

    createInterface({ input: child.stdout }).on("line", (line) => {
      const event = parseEngineEvent(line);
      if (event) this.emit("event", event);
    });
    createInterface({ input: child.stderr }).on("line", (line) => {
      this.emit("stderr", line);
    });
    child.on("exit", (code) => {
      this.child = null;
      this.emit("exit", code ?? 0);
    });
    child.on("error", (error) => {
      this.child = null;
      this.emit("event", {
        event: "engine_error",
        message: `không chạy được engine: ${error.message}`,
      } satisfies EngineEvent);
    });
  }

  send(command: EngineCommand): void {
    if (!this.child) return;
    this.child.stdin.write(JSON.stringify(command) + "\n");
  }

  stop(): void {
    if (!this.child) return;
    this.send({ cmd: "quit" });
    const child = this.child;
    setTimeout(() => child.kill("SIGTERM"), 3000).unref();
  }
}
