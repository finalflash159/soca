# 19 — Desktop packaging

Updated 2026-08-17.

## Status

`npm run tauri build` in [`desktop/`](../desktop/) works and produces real
installers:

```text
SoCa.app                    9.6 MB
SoCa_0.1.0_aarch64.dmg      4.2 MB
```

Those numbers are small because **the bundle contains only the interface**. The
Python engine and the models sit outside it, deliberately — see §2.

## 1. The bug that made a bundled app unusable

This is what broke the packaged app completely while `npm run tauri dev` kept
working.

macOS launches an app from Finder through `launchd`, not through a shell.
Measured on this machine:

```console
$ launchctl getenv PATH
                      # empty ⇒ defaults to /usr/bin:/bin:/usr/sbin:/sbin
$ which soca
<repo>/.venv/bin/soca
```

`soca` lives in a virtualenv, which is on none of those four directories.
Development worked **only because** a terminal had exported `PATH` first.
Double-clicking `SoCa.app` made `Command::new("soca")` fail at spawn.

`engine_candidates()` in [`engine.rs`](../desktop/src-tauri/src/engine.rs)
replaces that assumption with a real search order:

1. `SOCA_ENGINE` — an environment override, for pointing at a working checkout
2. a sidecar shipped inside the bundle (`resource_dir`, then next to the executable)
3. `~/.local/bin/soca`, `/opt/homebrew/bin/soca`, `/usr/local/bin/soca`
4. `PATH` — kept last so a developer's shell still wins when there is one

When nothing matches, the error **names every path it tried** and explains that
an app opened from Finder does not use the terminal's `PATH`. A bare "command
not found" sends people to inspect a variable that was never going to be read.

## 2. Why the engine is not inside the bundle

Measured:

| Component                                        |   Size |
| ------------------------------------------------ | -----: |
| `.venv`                                          | 1.7 GB |
| ` └ torch`                                       | 415 MB |
| ` └ pyarrow`                                     | 123 MB |
| ` └ llvmlite`                                    | 113 MB |
| Downloaded models (`~/.local/share/soca/models`) |  20 GB |
| HuggingFace cache                                |  44 GB |

Shipping all of it would mean a 20 GB installer, and every interface fix would
re-download the lot. This is the same reason Ollama and LM Studio do not do it:
a thin app, with models fetched on demand. SoCa already provisions artifacts
that way ([`soca/asr/qwen_store.py`](../soca/asr/qwen_store.py)), so the model
half of the problem is already solved.

Caveat on the figure: 1.7 GB is the **development** venv, which includes
`notebook`, `pandas`, `pyarrow` and `llvmlite` from the `dev` and `eval` extras.
A runtime-only install is considerably smaller but still carries torch. That
number has not been measured.

## 3. What is missing, and why it is blocked

### 3.1 Code signing — needs your certificate

`bundle.macOS.signingIdentity` is empty, so the `.app` is **unsigned**. Anyone
downloading it is stopped by Gatekeeper and has to right-click → Open, or run:

```console
xattr -dr com.apple.quarantine /Applications/SoCa.app
```

Signing requires an **Apple Developer ID** ($99/year) tied to your account;
nobody can do that step for you. Once you have one, add it to
`tauri.conf.json`:

```json
"macOS": { "signingIdentity": "Developer ID Application: YOUR NAME (TEAMID)" }
```

and notarize with `xcrun notarytool`.

### 3.2 Auto-update — needs a keypair and a host

`tauri-plugin-updater` is not installed. It needs an update-signing keypair
(`tauri signer generate`) and a stable URL serving the manifest. Both are
infrastructure decisions, not code.

### 3.3 The Python sidecar — not built

The search order in §1 already reserves a slot for a bundled binary
(`externalBin`), but no build produces one yet. Two approaches, neither tried:

- **PyInstaller** — not installed; this stack carries torch, onnxruntime and
  `llama-cpp-python`, all with native libraries, so the risk of a broken freeze
  is high.
- **A standalone Python plus a relocatable venv** — heavier, more predictable.

### 3.4 Windows and Linux — never built

`bundle.targets` is `"all"`, but only macOS arm64 has been built. `.msi` and
`.AppImage` have never been exercised.

## 4. Installing today

Until §3.3 lands, the engine is an **external dependency** and is installed
separately:

```console
git clone <repo-url> && cd shrike-7
uv sync --extra llm-remote
```

Then open `SoCa.app`. If it cannot find the engine, set the variable that a GUI
app actually reads — launchd's, not your shell's:

```console
launchctl setenv SOCA_ENGINE <repo>/.venv/bin/soca
```

or type the path into the app's startup screen.
