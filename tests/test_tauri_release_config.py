from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "write_tauri_release_config.py"
    spec = importlib.util.spec_from_file_location("write_tauri_release_config", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_config_requires_a_public_updater_key() -> None:
    module = _module()

    with pytest.raises(ValueError, match="TAURI_UPDATER_PUBKEY"):
        module.release_config({}, platform="linux")


def test_release_config_uses_signed_github_updates_and_platform_signing() -> None:
    module = _module()
    macos = module.release_config(
        {"TAURI_UPDATER_PUBKEY": "PUBLIC KEY", "APPLE_SIGNING_IDENTITY": "Developer ID"},
        platform="darwin",
    )
    windows = module.release_config(
        {
            "TAURI_UPDATER_PUBKEY": "PUBLIC KEY",
            "WINDOWS_CERTIFICATE_THUMBPRINT": "ABCD",
            "WINDOWS_TIMESTAMP_URL": "https://timestamp.example",
        },
        platform="win32",
    )

    assert macos["bundle"]["createUpdaterArtifacts"] is True
    assert macos["plugins"]["updater"]["endpoints"] == [module.UPDATE_ENDPOINT]
    assert windows["bundle"]["windows"]["certificateThumbprint"] == "ABCD"
    assert windows["plugins"]["updater"]["windows"] == {"installMode": "passive"}
