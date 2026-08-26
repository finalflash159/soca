"""Write the untracked, key-bearing Tauri release config used only in CI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

UPDATE_ENDPOINT = "https://github.com/finalflash159/soca/releases/latest/download/latest.json"
MACOS_ENTITLEMENTS = "src-tauri/SoCa.entitlements"


def release_config(environment: dict[str, str], *, platform: str) -> dict[str, object]:
    pubkey = environment.get("TAURI_UPDATER_PUBKEY", "").strip()
    if not pubkey:
        raise ValueError("TAURI_UPDATER_PUBKEY is required for a releasable updater artifact")

    bundle: dict[str, object] = {
        "createUpdaterArtifacts": True,
        "resources": ["resources/soca-engine/**/*"],
    }
    if platform == "darwin":
        identity = environment.get("APPLE_SIGNING_IDENTITY", "").strip()
        if not identity:
            raise ValueError("APPLE_SIGNING_IDENTITY is required for a macOS release")
        bundle["macOS"] = {
            "signingIdentity": identity,
            "entitlements": MACOS_ENTITLEMENTS,
        }
    elif platform == "win32":
        thumbprint = environment.get("WINDOWS_CERTIFICATE_THUMBPRINT", "").strip()
        timestamp = environment.get("WINDOWS_TIMESTAMP_URL", "").strip()
        if not thumbprint or not timestamp:
            raise ValueError(
                "WINDOWS_CERTIFICATE_THUMBPRINT and WINDOWS_TIMESTAMP_URL are required for a Windows release"
            )
        bundle["windows"] = {
            "certificateThumbprint": thumbprint,
            "digestAlgorithm": "sha256",
            "timestampUrl": timestamp,
        }

    return {
        "bundle": bundle,
        "plugins": {
            "updater": {
                "pubkey": pubkey,
                "endpoints": [UPDATE_ENDPOINT],
                "windows": {"installMode": "passive"},
            }
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", default=sys.platform)
    args = parser.parse_args()
    config = release_config(dict(os.environ), platform=args.platform)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
