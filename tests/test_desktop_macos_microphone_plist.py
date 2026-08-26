from __future__ import annotations

import json
import plistlib
from pathlib import Path


def test_macos_bundle_declares_its_microphone_purpose() -> None:
    """macOS must be able to grant the bundled Voice capture process access."""

    path = Path(__file__).resolve().parents[1] / "desktop" / "src-tauri" / "Info.plist"
    with path.open("rb") as source:
        payload = plistlib.load(source)

    purpose = payload.get("NSMicrophoneUsageDescription")
    assert isinstance(purpose, str)
    assert purpose.strip()


def test_macos_bundle_signs_the_audio_input_entitlement() -> None:
    """The usage description alone cannot obtain microphone access on macOS."""

    root = Path(__file__).resolve().parents[1]
    config_path = root / "desktop" / "src-tauri" / "tauri.package.conf.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    entitlements_path = root / "desktop" / "src-tauri" / config["bundle"]["macOS"]["entitlements"]

    with entitlements_path.open("rb") as source:
        entitlements = plistlib.load(source)

    assert entitlements["com.apple.security.device.audio-input"] is True
