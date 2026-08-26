# 19 — Desktop packaging and release evidence

Updated 2026-08-23.

## Release status

The desktop build now packages a native Python engine sidecar with Tauri. It is
not yet a public release claim: Apple and Windows signing credentials, updater
keys, notarization evidence, and native CI artifacts must exist before a signed
release can be published. The release workflow fails before producing an
installer when any of those inputs is absent.

| Capability | State | Evidence / remaining gate |
| --- | --- | --- |
| Bundled engine | implemented | PyInstaller native one-directory runtime, attached through Tauri `resources` |
| Persistence outside a checkout | implemented | frozen-sidecar migration/persistence verifier runs in packaging CI |
| macOS arm64 installer | locally reproducible | signed/notarized release is blocked on Apple credentials and CI evidence |
| Windows x64 installer | defined in CI | native package and signing evidence are pending the Windows workflow |
| Linux x64 package | CI artifact only | AppImage/DEB proof is defined; Linux is not a signed public release target |
| In-app updater | implemented, release-configured | requires the production signing key and `latest.json` endpoint |

## What is in a bundle

`npm run package:sidecar` invokes PyInstaller using the active native Python
interpreter. It produces `src-tauri/resources/soca-engine/`: the executable and
its native dependency closure. Tauri copies every file from that directory into
`$RESOURCE/resources/soca-engine/`. The generated runtime is ignored by Git and
recreated on every package build. Unlike a PyInstaller one-file executable, it
does not unpack hundreds of megabytes into a temporary directory at every cold
launch.

`npm run tauri:build` first builds that sidecar and then invokes Tauri with
[`tauri.package.conf.json`](../desktop/src-tauri/tauri.package.conf.json). A
packaged app starts only its own bundled engine. It does not silently fall back
to `PATH`, Homebrew, a virtual environment, or a checkout. The recovery field
in the startup UI is the sole explicit override; a missing bundled engine is a
visible startup error.

On packaged desktop builds, macOS Keychain access runs in a bounded helper
process (750 ms). A stale security-service prompt can hold Python's GIL, so a
thread timeout would not keep the engine responsive. The parent terminates the
helper at the deadline and continues through the existing explicit environment
and owner-only `0600` JSON credential paths; it reports the normal no-key
readiness state when none exists. This never substitutes a provider, model, or
local runtime. CLI behavior remains keyring-first and synchronous.

The runtime is self-contained, but model weights are intentionally not embedded.
They can be large and are provisioned by the engine. A package can therefore
start without a checkout, but a local-model conversation still needs its chosen
model artifact; remote operation needs its configured provider/key.

The frozen launcher emits its single protocol-v3 `hello` before importing the
heavy ASR/TTS/indexing stack. That frame establishes compatibility only: the UI
continues to show the actual model/backend readiness from the initialized
engine's `llm_config` and runtime-component frames, and keeps chat disabled
until that readiness is true. It never substitutes a model or treats the
preflight as model-loaded.

The engine reports local and remote readiness as distinct states: missing local
weights never block a configured remote provider, and remote readiness is only
reported after its key, provider catalog and selected model are all valid. A
missing remote key never pretends that the local model is unavailable.

For local use, SoCa first looks in its managed model root under app data. The
**Thư mục model local** control can instead select an existing absolute
`models/` directory; it persists only that path, restarts the sidecar, and does
not copy or silently delete multi-gigabyte artifacts. This makes an existing
developer/downloaded model usable by the package while keeping model placement
an explicit user choice. If neither location has the selected GGUF, the UI
shows the exact expected file instead of claiming the runtime is ready.

## Data, sessions, migration, and uninstall

At packaged-runtime launch, the Rust host maps the engine's XDG directories and
`SOCA_VAULT` beneath Tauri's per-user app-data root:

```text
<Tauri app-data>/config  -> XDG_CONFIG_HOME
<Tauri app-data>/data    -> XDG_DATA_HOME
<Tauri app-data>/state   -> XDG_STATE_HOME
<Tauri app-data>/vault   -> SOCA_VAULT
<Tauri app-data>/data/soca/models -> SOCA_MODEL_ROOT (managed default)
```

`SOCA_MODEL_ROOT` points at the managed model root unless the user explicitly
selects an existing external model directory through the settings UI; only the
selected path is persisted.

This root is selected by the operating system and Tauri for the application
identifier `com.finalflash159.soca`; it is deliberately not the installation
directory. Updating or replacing the application leaves it in place.

Session retention remains opt-in. In-memory mode leaves no resumable transcript.
When local resumable sessions are enabled, the SQLite session repository is
created under `<Tauri app-data>/state/soca/sessions`; POSIX builds verify private
directory/file permissions. On first packaged launch it imports any legacy
checkpoint directory once, preserves a manifest-backed backup, and records the
migration so a restart cannot import duplicates. Windows uses byte-range locking
for the same session lease contract; ACL privacy verification remains a native
Windows release-evidence requirement, not an unproven claim.

Uninstalling the app must not be presented as deletion of personal data. The
app does not remove its app-data root. To remove local sessions, vault data and
settings, first export anything wanted, close SoCa, then delete the OS-managed
app-data root for `com.finalflash159.soca` through that platform's file manager
or settings UI. This is irreversible.

## Updates, signing, and rollback

The Settings page can check, download, install, and relaunch a signed release
through the Tauri updater. The updater plugin is compiled only by the signed
release workflow, after it has generated key-bearing configuration. Package
proofs and development builds do not register a dummy updater, so they can open
normally and the panel reports that update checking is unavailable instead of
panicking before a window appears. Release-only configuration is generated by
[`write_tauri_release_config.py`](../scripts/write_tauri_release_config.py), not
committed with a placeholder public key. It requires:

- a Tauri updater signing keypair and the production public key;
- HTTPS release metadata at the configured `latest.json` endpoint;
- Apple Developer ID certificate, signing identity, and notarization credentials
  for macOS; and
- a Windows code-signing certificate, thumbprint, and timestamp service for
  Windows.

The tag workflow refuses to publish if a required secret is missing. Its updater
artifacts are signed by Tauri; platform packages are signed separately. A
successful build without those credentials is only a package proof, never a
trusted public release.

The workflow creates or updates an `app-v<version>` **draft** release. A release
owner must inspect the signed installer artifacts, updater manifest and native
CI evidence, then publish that draft deliberately. While it remains a draft,
the updater endpoint continues to serve the last public release; a green build
alone does not make a new version available to users.

Updates preserve the app-data root. Rollback is an explicit operator action:
install a verified, signed prior release that is compatible with the existing
session schema, then keep the data root unchanged. There is no automatic
fallback to a prior engine, data schema, or package. Before a future schema
change, the release must document its compatibility and backup/restore path.

## Build and verification commands

From [`desktop/`](../desktop/):

```bash
npm ci
npm test
npm run package:sidecar
npm run tauri:build -- --bundles app,dmg
codesign --verify --deep --strict --verbose=4 \
  src-tauri/target/release/bundle/macos/SoCa.app
```

The tracked package config uses Tauri's macOS pseudo-identity `-` for a
reproducible local ad-hoc signature. It is a Finder-install smoke gate only;
the release workflow overlays its Apple Developer identity and must still
notarize the public artifact.

The independent runtime proof deliberately runs outside the source checkout:

```bash
uv run python scripts/verify_desktop_sidecar.py \
  --sidecar desktop/src-tauri/resources/soca-engine/soca-engine
uv run python scripts/verify_desktop_remote_settings.py \
  --sidecar desktop/src-tauri/target/release/bundle/macos/SoCa.app/Contents/Resources/resources/soca-engine/soca-engine
```

The storage verifier exercises the frozen session protocol, legacy migration,
and persistence path. The remote-settings verifier proves the packaged remote
contract without spending a real credential: it only picks a provider with no
key resolvable through the real chain (keyring, env, JSON fallback), then
injects an invalid key and requires the live provider catalog request to end
in a typed provider failure — OpenAI-compatible endpoints answer 401/403 with
the typed auth error, while Google's compatibility shim answers HTTP 400 with
the typed catalog error — instead of a fake catalog or a silent fallback. This
is failure-route evidence only; a successful catalog load and chat still need
the release-owner real-flow matrix.

On pull requests, [`desktop-package.yml`](../.github/workflows/desktop-package.yml)
builds native macOS arm64, Windows x64, and Linux x64 artifacts, then runs this
frozen migration/persistence proof. Tags run
[`desktop-release.yml`](../.github/workflows/desktop-release.yml), which adds
the signing and updater gates above. PyInstaller builds per native target; it
does not cross-compile the Python runtime.

The initial freeze explicitly collects `llama_cpp` native libraries and carries
the `torchcodec` distribution metadata required by `transformers` at import
time. This is tested because a missing `libllama` or metadata can otherwise let
an installer build while the engine crashes before its protocol `hello` frame.

## Privacy and support boundary

See [`RELEASE_NOTES.md`](../RELEASE_NOTES.md) for end-user retention, update,
uninstall, and rollback guidance. Do not advise users to bypass Gatekeeper,
SmartScreen, quarantine, or signature warnings: those warnings mean the signed
release gate above has not been satisfied.
