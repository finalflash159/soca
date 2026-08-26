from __future__ import annotations

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
