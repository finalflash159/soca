//! Sidecar manager for the `soca engine` NDJSON process.
//!
//! The contract implemented here is `docs/18-engine-protocol.md`. Two rules from
//! §7 drive the whole design:
//!
//! * Commands go in on stdin, events come out on stdout, and stderr is
//!   diagnostics only — it must never be parsed as protocol.
//! * Shutdown closes stdin and waits for `bye`. `bye` is the only proof that
//!   audio devices and provider clients were released, so SIGKILL is a last
//!   resort, not the normal path.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};

/// Frontend event channel for one decoded protocol frame.
pub const EVENT_CHANNEL: &str = "soca://engine-event";
/// Frontend event channel for sidecar lifecycle changes the protocol cannot express.
pub const STATUS_CHANNEL: &str = "soca://engine-status";

/// How long to wait for `bye` after closing stdin before escalating.
const GRACEFUL_BYE_TIMEOUT: Duration = Duration::from_secs(35);
/// How long to wait for the process to exit after `bye` before killing it.
const EXIT_TIMEOUT: Duration = Duration::from_secs(5);
/// Poll interval while waiting for process exit.
const EXIT_POLL: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase", tag = "state")]
pub enum EngineStatus {
    Starting {
        program: String,
    },
    Running,
    /// The engine emitted `bye` and exited on its own terms.
    Stopped {
        code: Option<i32>,
        graceful: bool,
    },
    Failed {
        message: String,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LaunchOptions {
    /// Executable to run. Defaults to `soca` resolved on PATH.
    #[serde(default)]
    pub program: Option<String>,
    /// Arguments before `engine`. Lets a dev run `uv run soca engine`.
    #[serde(default)]
    pub args: Vec<String>,
    /// Working directory for the child process.
    #[serde(default)]
    pub cwd: Option<String>,
}

impl LaunchOptions {
    fn resolve(&self) -> (String, Vec<String>) {
        let program = self
            .program
            .clone()
            .unwrap_or_else(|| "soca".to_string());
        let mut args = self.args.clone();
        args.push("engine".to_string());
        (program, args)
    }
}

struct Running {
    child: Child,
    stdin: Option<ChildStdin>,
    /// Signalled by the reader thread when the `bye` frame is decoded.
    bye_rx: mpsc::Receiver<()>,
    reader: Option<JoinHandle<()>>,
    stderr: Option<JoinHandle<()>>,
    /// Tells the reader thread that a stop is intentional, so an EOF on stdout
    /// is not reported to the UI as a crash.
    stopping: Arc<AtomicBool>,
}

#[derive(Default)]
pub struct EngineState {
    running: Mutex<Option<Running>>,
}

fn emit_status(app: &AppHandle, status: EngineStatus) {
    // A failed emit means the window is gone; the process teardown below still
    // has to run, so this is logged rather than propagated.
    if let Err(error) = app.emit(STATUS_CHANNEL, status) {
        eprintln!("[soca-desktop] status emit failed: {error}");
    }
}

/// Read stdout frames until EOF, forwarding each decoded object to the frontend.
fn spawn_reader(
    app: AppHandle,
    stdout: std::process::ChildStdout,
    bye_tx: mpsc::Sender<()>,
    stopping: Arc<AtomicBool>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let line = match line {
                Ok(line) => line,
                Err(error) => {
                    eprintln!("[soca-desktop] stdout read failed: {error}");
                    break;
                }
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<serde_json::Value>(trimmed) {
                Ok(frame) => {
                    let is_bye = frame
                        .get("event")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|name| name == "bye");
                    if let Err(error) = app.emit(EVENT_CHANNEL, &frame) {
                        eprintln!("[soca-desktop] event emit failed: {error}");
                    }
                    if is_bye {
                        // Ignored on purpose: the receiver is dropped when a stop
                        // times out, and the thread must still drain to EOF.
                        let _ = bye_tx.send(());
                    }
                }
                Err(error) => {
                    // docs/18 §1: a client tolerates frames it cannot parse rather
                    // than tearing down the session.
                    eprintln!("[soca-desktop] undecodable frame ({error}): {trimmed}");
                }
            }
        }
        if !stopping.load(Ordering::SeqCst) {
            emit_status(
                &app,
                EngineStatus::Failed {
                    message: "engine stdout closed unexpectedly".to_string(),
                },
            );
        }
    })
}

/// Drain stderr so a chatty model loader cannot fill the pipe buffer and block
/// the child. Never parsed as protocol.
fn spawn_stderr_drain(stderr: std::process::ChildStderr) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            eprintln!("[soca engine] {line}");
        }
    })
}

#[tauri::command]
pub fn engine_start(
    app: AppHandle,
    state: State<'_, EngineState>,
    options: Option<LaunchOptions>,
) -> Result<(), String> {
    let mut guard = state.running.lock().map_err(|_| "engine state poisoned")?;
    if guard.is_some() {
        return Err("engine already running".to_string());
    }

    let options = options.unwrap_or(LaunchOptions {
        program: None,
        args: Vec::new(),
        cwd: None,
    });
    let (program, args) = options.resolve();
    emit_status(
        &app,
        EngineStatus::Starting {
            program: format!("{program} {}", args.join(" ")),
        },
    );

    let mut command = Command::new(&program);
    command
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(cwd) = &options.cwd {
        command.current_dir(cwd);
    }

    let mut child = command.spawn().map_err(|error| {
        format!("could not start `{program}`: {error}. Is soca on PATH, or set a program override?")
    })?;

    let stdin = child.stdin.take().ok_or("child stdin unavailable")?;
    let stdout = child.stdout.take().ok_or("child stdout unavailable")?;
    let stderr = child.stderr.take().ok_or("child stderr unavailable")?;

    let stopping = Arc::new(AtomicBool::new(false));
    let (bye_tx, bye_rx) = mpsc::channel();
    let reader = spawn_reader(app.clone(), stdout, bye_tx, Arc::clone(&stopping));
    let stderr_drain = spawn_stderr_drain(stderr);

    *guard = Some(Running {
        child,
        stdin: Some(stdin),
        bye_rx,
        reader: Some(reader),
        stderr: Some(stderr_drain),
        stopping,
    });
    emit_status(&app, EngineStatus::Running);
    Ok(())
}

#[tauri::command]
pub fn engine_send(state: State<'_, EngineState>, command: serde_json::Value) -> Result<(), String> {
    if !command.is_object() {
        return Err("command must be a JSON object".to_string());
    }
    let mut guard = state.running.lock().map_err(|_| "engine state poisoned")?;
    let running = guard.as_mut().ok_or("engine is not running")?;
    let stdin = running.stdin.as_mut().ok_or("engine stdin is closed")?;

    let mut line = serde_json::to_string(&command).map_err(|error| error.to_string())?;
    line.push('\n');
    stdin
        .write_all(line.as_bytes())
        .and_then(|()| stdin.flush())
        .map_err(|error| format!("write to engine failed: {error}"))
}

#[tauri::command]
pub fn engine_is_running(state: State<'_, EngineState>) -> Result<bool, String> {
    let guard = state.running.lock().map_err(|_| "engine state poisoned")?;
    Ok(guard.is_some())
}

#[tauri::command]
pub fn engine_stop(app: AppHandle, state: State<'_, EngineState>) -> Result<(), String> {
    let running = {
        let mut guard = state.running.lock().map_err(|_| "engine state poisoned")?;
        guard.take()
    };
    let Some(running) = running else {
        return Ok(());
    };
    let outcome = shutdown(running);
    emit_status(&app, outcome);
    Ok(())
}

/// The docs/18 §7 shutdown sequence, with escalation at every step.
fn shutdown(mut running: Running) -> EngineStatus {
    running.stopping.store(true, Ordering::SeqCst);

    // 1. Ask politely, then close stdin. Either alone ends the loop; both
    //    together also cover an engine that is mid-command when we ask.
    if let Some(stdin) = running.stdin.as_mut() {
        let _ = stdin.write_all(b"{\"cmd\": \"quit\"}\n");
        let _ = stdin.flush();
    }
    drop(running.stdin.take());

    // 2. Wait for `bye`.
    let graceful = match running.bye_rx.recv_timeout(GRACEFUL_BYE_TIMEOUT) {
        Ok(()) => true,
        Err(RecvTimeoutError::Timeout) => {
            eprintln!("[soca-desktop] engine did not emit bye within {GRACEFUL_BYE_TIMEOUT:?}");
            false
        }
        // Sender dropped: the reader thread hit EOF without a bye frame.
        Err(RecvTimeoutError::Disconnected) => false,
    };

    // 3. Give the process a moment to exit on its own after bye.
    let deadline = std::time::Instant::now() + EXIT_TIMEOUT;
    let mut code = None;
    loop {
        match running.child.try_wait() {
            Ok(Some(status)) => {
                code = status.code();
                break;
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    // 4. Last resort. Only reached when the engine ignored both
                    //    quit and EOF, which is itself worth reporting.
                    eprintln!("[soca-desktop] engine still alive after bye; killing");
                    let _ = running.child.kill();
                    let _ = running.child.wait();
                    break;
                }
                std::thread::sleep(EXIT_POLL);
            }
            Err(error) => {
                eprintln!("[soca-desktop] try_wait failed: {error}");
                let _ = running.child.kill();
                break;
            }
        }
    }

    // 5. Join the pipe threads so no orphan thread outlives the process.
    if let Some(handle) = running.reader.take() {
        let _ = handle.join();
    }
    if let Some(handle) = running.stderr.take() {
        let _ = handle.join();
    }

    EngineStatus::Stopped { code, graceful }
}

/// Stop the engine on window close so quitting the app never orphans a child.
pub fn shutdown_on_exit(app: &AppHandle) {
    let Some(state) = app.try_state::<EngineState>() else {
        return;
    };
    let running = match state.running.lock() {
        Ok(mut guard) => guard.take(),
        Err(_) => return,
    };
    if let Some(running) = running {
        let _ = shutdown(running);
    }
}
