# SoCa edge daemon

`soca-edge` is the Linux ARM capture boundary for SoCa. It captures one CPAL
input stream, moves samples through a bounded SPSC ring, resamples outside the
audio callback, runs the pinned Silero VAD and Smart Turn ONNX models, applies
the production adaptive endpoint policy, and emits typed NDJSON. It does not
generate an answer and cannot bypass the host controller or its verification.

## Build contract

- Rust 1.88 or newer, edition 2024.
- `cpal` 0.18.1 and `rtrb` 0.3.4 are exact pins in `Cargo.lock`.
- `ort` 2.0.0-rc.12 uses the ONNX Runtime 1.24 API and loads the runtime
  dynamically. Set `ORT_DYLIB_PATH` to the exact target ONNX Runtime shared
  library before starting the daemon.
- Linux builds require the ALSA development package (`libasound2-dev` on
  Debian/Raspberry Pi OS).

```bash
cd edge/soca-edge
cargo build --release --locked
cargo test --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
```

The research plan originally called this a “static binary”. That is not the
current packaging claim: ONNX Runtime remains a native shared-library
dependency, so the executable is a single Rust daemon plus a pinned ORT
runtime and model files. A fully static ORT build is deferred until it has a
reproducible aarch64 packaging gate.

## Model provisioning and run

Provision the exact production Smart Turn model with
`uv run python scripts/download_smart_turn.py`. Provision the Silero VAD ONNX
from the installed `silero-vad` 6.2.1 package. The daemon hashes both files at
startup and includes the hashes in its ready event and device receipt.

```bash
ORT_DYLIB_PATH=/opt/soca/lib/libonnxruntime.so \
  target/release/soca-edge \
  --silero-model /opt/soca/models/silero_vad.onnx \
  --smart-turn-model /opt/soca/models/smart-turn-v3.2-cpu.onnx
```

Every stdout line is one `soca-edge-ndjson-v1` object: `ready`, `turn`, or
`failure`. A `turn` carries base64 PCM signed 16-bit little-endian audio at
16 kHz and endpoint telemetry. A full ring increments
`dropped_capture_samples`; a CPAL error emits `failure` and terminates instead
of switching devices or models.

## ARM device release gate

Run at least five minutes on a real Linux aarch64 SBC and complete at least
five turns:

```bash
target/release/soca-edge \
  --silero-model /opt/soca/models/silero_vad.onnx \
  --smart-turn-model /opt/soca/models/smart-turn-v3.2-cpu.onnx \
  --measurement-seconds 300 \
  --receipt artifacts/local/soca-edge-sbc-receipt.json

uv run python -m eval.eval_edge_daemon \
  --receipt artifacts/local/soca-edge-sbc-receipt.json \
  --output artifacts/local/soca-edge-sbc-gate.json
```

Only Linux/aarch64, zero dropped samples, a healthy stream, processing p95 at
or below one 32 ms frame, peak RSS at or below 512 MiB, and matching model
identities can pass. macOS arm64 compilation or unit tests are not an ARM SBC
measurement.
