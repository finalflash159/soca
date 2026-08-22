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

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
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
/// Basename registered in `bundle.externalBin`.
const SIDECAR_BASENAME: &str = "soca-engine";

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

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LaunchOptions {
    /// Explicit recovery executable. When absent, a packaged app resolves only
    /// the bundled sidecar; a checkout-backed development build supplies `soca`.
    #[serde(default)]
    pub program: Option<String>,
    /// Arguments before `engine`. Lets a dev run `uv run soca engine`.
    #[serde(default)]
    pub args: Vec<String>,
    /// Working directory for the child process.
    #[serde(default)]
    pub cwd: Option<String>,
    /// Extra environment variables for the child, merged over the inherited set.
    ///
    /// This exists for `PYTHONPATH`. `soca` on PATH is an editable install that
    /// pins one checkout, so a dev running this app from a git worktree would
    /// otherwise launch the *other* checkout's Python source and never see
    /// their own changes.
    #[serde(default)]
    pub env: HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct ResolvedLaunch {
    program: String,
    args: Vec<String>,
}

fn sidecar_filename() -> String {
    format!("{SIDECAR_BASENAME}{}", std::env::consts::EXE_SUFFIX)
}

/// Return only binaries owned by this bundle. A shipped app must not silently
/// start a random venv or PATH installation after its selected runtime fails.
fn engine_candidates(app: &AppHandle) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let filename = sidecar_filename();

    // Tauri places externalBin files next to the executable. resource_dir is
    // retained for bundle layouts used by older Tauri versions and test hosts.
    if let Ok(dir) = app.path().resource_dir() {
        candidates.push(dir.join(&filename));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(filename));
        }
    }
    candidates
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

impl LaunchOptions {
    fn resolve(&self, app: &AppHandle) -> Result<ResolvedLaunch, String> {
        let mut args = self.args.clone();
        args.push("engine".to_string());

        // An explicit UI choice is allowed to use PATH. It is the user's
        // recovery override, not an automatic fallback selected by the app.
        if let Some(program) = self.program.clone().filter(|value| !value.is_empty()) {
            return Ok(ResolvedLaunch { program, args });
        }
        if let Ok(value) = std::env::var("SOCA_ENGINE") {
            let program = value.trim();
            if !program.is_empty() {
                if !is_executable(Path::new(program)) {
                    return Err(format!(
                        "SOCA_ENGINE không trỏ tới file chạy được: {program}"
                    ));
                }
                return Ok(ResolvedLaunch {
                    program: program.to_string(),
                    args,
                });
            }
        }

        let candidates = engine_candidates(app);
        for candidate in &candidates {
            if is_executable(candidate) {
                return Ok(ResolvedLaunch {
                    program: candidate.display().to_string(),
                    args,
                });
            }
        }

        // A release failure must identify the missing artifact, not fall back
        // to a potentially incompatible interpreter found on PATH.
        let tried = candidates
            .iter()
            .map(|path| format!("  {}", path.display()))
            .collect::<Vec<_>>()
            .join("\n");
        Err(format!(
            "Không tìm thấy bundled engine `{SIDECAR_BASENAME}`. Đã thử:\n{tried}\n\n\
             Bản cài đặt bị thiếu hoặc hỏng runtime đi kèm. Cài lại đúng bản phát \
             hành, hoặc chủ động chọn một executable bằng tuỳ chọn khôi phục."
        ))
    }
}

/// Map Python's portable XDG configuration onto Tauri's platform-native
/// application-data root. The app owns this directory across upgrades; Python
/// continues to own private file modes and SQLite schema validation beneath it.
fn bundled_runtime_env(app_data_dir: &Path) -> Result<HashMap<String, String>, String> {
    let config = app_data_dir.join("config");
    let data = app_data_dir.join("data");
    let state = app_data_dir.join("state");
    let vault = app_data_dir.join("vault");
    for directory in [&config, &data, &state, &vault] {
        fs::create_dir_all(directory).map_err(|error| {
            format!(
                "Không thể tạo thư mục dữ liệu desktop {}: {error}",
                directory.display()
            )
        })?;
    }

    let mut env = HashMap::from([
        ("XDG_CONFIG_HOME".to_string(), config.display().to_string()),
        ("XDG_DATA_HOME".to_string(), data.display().to_string()),
        ("XDG_STATE_HOME".to_string(), state.display().to_string()),
        ("SOCA_VAULT".to_string(), vault.display().to_string()),
    ]);

    // Previous desktop builds delegated to a normal Python process, whose
    // historical default was this root on every OS. The engine validates and
    // migrates it only when legacy checkpoint files exist.
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        let legacy = PathBuf::from(home)
            .join(".local")
            .join("state")
            .join("soca")
            .join("sessions");
        env.insert(
            "SOCA_LEGACY_SESSION_ROOT".to_string(),
            legacy.display().to_string(),
        );
    }
    Ok(env)
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

    let options = options.unwrap_or_default();
    let resolved = options.resolve(&app)?;
    let program = resolved.program;
    let args = resolved.args;
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
    command.envs(&options.env);
    if !cfg!(debug_assertions) {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| format!("Không xác định được app-data directory: {error}"))?;
        command.envs(bundled_runtime_env(&app_data_dir)?);
    }

    // Which interpreter and which source tree actually ran is the first thing
    // anyone needs when the engine misbehaves, and it is invisible otherwise.
    eprintln!(
        "[soca-desktop] launching `{program} {}`{}",
        args.join(" "),
        options
            .env
            .iter()
            .map(|(key, value)| format!(" {key}={value}"))
            .collect::<String>()
    );

    let mut child = command.spawn().map_err(|error| {
        format!("Không thể khởi động `{program}`: {error}. Chọn executable khôi phục hoặc cài lại runtime đi kèm.")
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
pub fn engine_send(
    state: State<'_, EngineState>,
    command: serde_json::Value,
) -> Result<(), String> {
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

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn bundled_runtime_env_keeps_python_data_under_the_app_data_root() {
        let root =
            std::env::temp_dir().join(format!("soca-desktop-runtime-env-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);

        let env = bundled_runtime_env(&root).expect("runtime data env");

        assert_eq!(
            env["XDG_CONFIG_HOME"],
            root.join("config").display().to_string()
        );
        assert_eq!(
            env["XDG_DATA_HOME"],
            root.join("data").display().to_string()
        );
        assert_eq!(
            env["XDG_STATE_HOME"],
            root.join("state").display().to_string()
        );
        assert_eq!(env["SOCA_VAULT"], root.join("vault").display().to_string());
        assert!(root.join("config").is_dir());
        assert!(root.join("data").is_dir());
        assert!(root.join("state").is_dir());
        assert!(root.join("vault").is_dir());

        fs::remove_dir_all(root).expect("remove test runtime data");
    }

    #[test]
    fn shutdown_sends_quit_waits_for_bye_and_reaps_the_child() {
        let mut child = Command::new("sh")
            .args(["-c", "IFS= read -r _; printf '{\"event\":\"bye\"}\\n'"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn fixture child");
        let stdin = child.stdin.take().expect("fixture stdin");
        let stdout = child.stdout.take().expect("fixture stdout");
        let (bye_tx, bye_rx) = mpsc::channel();
        let stopping = Arc::new(AtomicBool::new(false));
        let reader = std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if line.contains("\"event\":\"bye\"") {
                    let _ = bye_tx.send(());
                    break;
                }
            }
        });

        let result = shutdown(Running {
            child,
            stdin: Some(stdin),
            bye_rx,
            reader: Some(reader),
            stderr: None,
            stopping,
        });

        assert!(matches!(
            result,
            EngineStatus::Stopped {
                code: Some(0),
                graceful: true
            }
        ));
    }
}
