# Qwen ASR worker runtime

This project locks the dependency environment used only by the Qwen ASR
subprocess. It deliberately excludes upstream web/demo packages that are not
imported by model startup, transcription, IPC, or teardown.

Provision it from the repository root:

```bash
uv run python scripts/provision_qwen_runtime.py
```

The provisioner performs an exact frozen sync, builds the current SoCa wheel,
installs that wheel without its main-runtime dependencies, verifies the import
boundary, and writes a private local receipt. It never installs the repository
editable and never updates packages in place.

The supported lock currently targets macOS arm64 with CPython 3.11.14. Other
platforms must get an independently generated and tested lock before being
advertised as supported.
