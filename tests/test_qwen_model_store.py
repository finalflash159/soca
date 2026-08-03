from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from soca.asr.qwen_artifacts import (
    ArtifactFile,
    ArtifactRole,
    ArtifactSource,
    QwenASRArtifactSpec,
)
from soca.asr.qwen_store import (
    ArtifactInvalid,
    ArtifactSourceKind,
    ArtifactState,
    InsufficientArtifactDisk,
    MirrorNotPinned,
    ProvisionLockBusy,
    QwenArtifactStore,
    QwenSnapshotResolver,
    UnsupportedArtifactPlatform,
    WorkerRuntimeInvalid,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[QwenASRArtifactSpec, Path, Path]:
    runtime_lock = tmp_path / "uv.lock"
    runtime_lock.write_text("locked runtime", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen3_asr", "architectures": ["Qwen3ASR"]}),
        encoding="utf-8",
    )
    (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (source / "preprocessor_config.json").write_text(
        json.dumps({"feature_extractor_type": "WhisperFeatureExtractor"}),
        encoding="utf-8",
    )
    save_file({"weight": np.ones((2, 2), dtype=np.float32)}, source / "model.safetensors")
    files = tuple(
        ArtifactFile(path=path.name, size=path.stat().st_size, sha256=_sha256(path))
        for path in sorted(source.iterdir())
    )
    spec = QwenASRArtifactSpec(
        key="qwen_test",
        role=ArtifactRole.RELEASE,
        upstream=ArtifactSource("Qwen/test", "a" * 40),
        mirror=None,
        files=files,
        license="apache-2.0",
        device="cpu",
        dtype="float32",
        runtime_lock_digest=_sha256(runtime_lock),
        context_policy_digest=None,
        minimum_protocol_version=1,
    )
    return spec, source, tmp_path / "store"


def _install(
    store: QwenArtifactStore,
    spec: QwenASRArtifactSpec,
    source: Path,
    **kwargs,
):
    return store.install_from_snapshot(
        spec,
        source,
        runtime_lock=source.parent / "uv.lock",
        system="Darwin",
        machine="arm64",
        **kwargs,
    )


def test_install_activates_verified_private_read_only_generation(tmp_path: Path) -> None:
    spec, source, root = _fixture(tmp_path)
    probes: list[Path] = []
    store = QwenArtifactStore(root, disk_free=lambda _: spec.total_bytes * 3)

    receipt = _install(
        store,
        spec,
        source,
        source_kind=ArtifactSourceKind.UPSTREAM,
        health_probe=lambda path: probes.append(path) or {"transcript": "xin chào"},
    )

    assert len(probes) == 1
    assert probes[0].name.startswith(".staging-qwen_test-")
    assert receipt.artifact_digest == spec.digest
    assert receipt.source.repo_id == spec.upstream.repo_id
    assert receipt.health["transcript_nonempty"] is True
    assert "transcript" not in receipt.health
    assert spec.model_path(root).is_dir()
    assert stat.S_IMODE(spec.receipt_path(root).stat().st_mode) == 0o600
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o400
        for path in spec.model_path(root).iterdir()
        if path.is_file()
    )
    assert stat.S_IMODE((source / "config.json").stat().st_mode) & stat.S_IWUSR


def test_install_is_idempotent_and_does_not_probe_twice(tmp_path: Path) -> None:
    spec, source, root = _fixture(tmp_path)
    calls = 0

    def probe(_: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"transcript": "ok"}

    store = QwenArtifactStore(root, disk_free=lambda _: spec.total_bytes * 3)
    first = _install(
        store, spec, source, source_kind=ArtifactSourceKind.UPSTREAM, health_probe=probe
    )
    second = _install(
        store, spec, source, source_kind=ArtifactSourceKind.UPSTREAM, health_probe=probe
    )

    assert first == second
    assert calls == 1


def test_refresh_receipt_revalidates_existing_bytes_after_device_change(tmp_path: Path) -> None:
    spec, source, root = _fixture(tmp_path)
    store = QwenArtifactStore(root, disk_free=lambda _: spec.total_bytes * 3)
    first = _install(
        store,
        spec,
        source,
        source_kind=ArtifactSourceKind.UPSTREAM,
        health_probe=lambda _: {"transcript": "ok"},
    )
    mps_spec = replace(spec, device="mps", dtype="float16")

    refreshed = store.refresh_receipt(
        mps_spec,
        source_kind=ArtifactSourceKind.UPSTREAM,
        runtime_lock=source.parent / "uv.lock",
        health_probe=lambda _: {"transcript": "ok on mps"},
    )

    assert refreshed.artifact_digest == mps_spec.digest
    assert refreshed.artifact_digest != first.artifact_digest
    assert mps_spec.model_path(root).is_dir()
    assert stat.S_IMODE(mps_spec.receipt_path(root).stat().st_mode) == 0o600
    assert store.verify(mps_spec, deep=False) == refreshed


def test_wrong_hash_and_health_failure_never_activate(tmp_path: Path) -> None:
    spec, source, root = _fixture(tmp_path)
    config = source / "config.json"
    original = config.read_bytes()
    config.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    store = QwenArtifactStore(root, disk_free=lambda _: spec.total_bytes * 3)

    with pytest.raises(ArtifactInvalid, match="sha256"):
        _install(
            store,
            spec,
            source,
            source_kind=ArtifactSourceKind.UPSTREAM,
            health_probe=lambda _: {"transcript": "ok"},
        )

    assert not spec.model_path(root).exists()
    assert not spec.receipt_path(root).exists()


def test_mirror_source_requires_a_pinned_mirror(tmp_path: Path) -> None:
    spec, source, root = _fixture(tmp_path)

    with pytest.raises(MirrorNotPinned):
        _install(
            QwenArtifactStore(root),
            spec,
            source,
            source_kind=ArtifactSourceKind.MIRROR,
            health_probe=lambda _: {"transcript": "ok"},
        )


def test_preflight_rejects_insufficient_disk_before_copy(tmp_path: Path) -> None:
    spec, source, root = _fixture(tmp_path)
    store = QwenArtifactStore(root, disk_free=lambda _: spec.total_bytes - 1)

    with pytest.raises(InsufficientArtifactDisk):
        _install(
            store,
            spec,
            source,
            source_kind=ArtifactSourceKind.UPSTREAM,
            health_probe=lambda _: {"transcript": "ok"},
        )

    assert not spec.model_path(root).exists()


def test_preflight_reports_manifest_bytes_and_exact_runtime_lock(tmp_path: Path) -> None:
    spec, source, root = _fixture(tmp_path)
    lock = tmp_path / "uv.lock"
    lock.write_text("locked", encoding="utf-8")
    spec = replace(spec, runtime_lock_digest=_sha256(lock))
    store = QwenArtifactStore(root, disk_free=lambda _: spec.total_bytes * 4)

    report = store.preflight(
        spec,
        source,
        runtime_lock=lock,
        system="Darwin",
        machine="arm64",
    )

    assert report.platform == "Darwin"
    assert report.architecture == "arm64"
    assert report.final_bytes == spec.total_bytes
    assert report.staging_bytes == spec.total_bytes
    assert report.reusable_bytes == 0
    assert report.required_free_bytes == spec.total_bytes
    assert report.free_bytes == spec.total_bytes * 4


def test_preflight_rejects_unsupported_platform_and_runtime_drift(tmp_path: Path) -> None:
    spec, source, root = _fixture(tmp_path)
    lock = tmp_path / "uv.lock"
    lock.write_text("wrong", encoding="utf-8")
    store = QwenArtifactStore(root, disk_free=lambda _: spec.total_bytes * 4)

    with pytest.raises(UnsupportedArtifactPlatform, match="Linux/x86_64"):
        store.preflight(
            spec,
            source,
            runtime_lock=lock,
            system="Linux",
            machine="x86_64",
        )
    with pytest.raises(WorkerRuntimeInvalid, match="digest"):
        store.preflight(
            spec,
            source,
            runtime_lock=lock,
            system="Darwin",
            machine="arm64",
        )


def test_install_cannot_bypass_platform_preflight(tmp_path: Path) -> None:
    spec, source, root = _fixture(tmp_path)
    store = QwenArtifactStore(root, disk_free=lambda _: spec.total_bytes * 4)

    with pytest.raises(UnsupportedArtifactPlatform, match="Linux/x86_64"):
        store.install_from_snapshot(
            spec,
            source,
            source_kind=ArtifactSourceKind.UPSTREAM,
            health_probe=lambda _: {"transcript": "ok"},
            runtime_lock=tmp_path / "uv.lock",
            system="Linux",
            machine="x86_64",
        )

    assert not root.exists()


def test_store_rejects_symlinked_parent_before_creating_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    redirect = tmp_path / "redirect"
    redirect.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactInvalid, match="contains a symlink"):
        QwenArtifactStore(
            redirect / "store",
        )

    assert not (outside / "store").exists()


def test_verification_and_chmod_reject_symlinks_inside_generation(
    tmp_path: Path,
) -> None:
    spec, source, root = _fixture(tmp_path)
    external = tmp_path / "external"
    external.write_text("outside", encoding="utf-8")
    (source / "escape").symlink_to(external)
    store = QwenArtifactStore(root)

    with pytest.raises(ArtifactInvalid, match="tree contains a symlink"):
        store.verify_directory(spec, source, deep=False)
    with pytest.raises(ArtifactInvalid, match="tree contains a symlink"):
        store._make_read_only(source)

    assert stat.S_IMODE(external.stat().st_mode) & stat.S_IWUSR


def test_concurrent_provision_lock_is_typed(tmp_path: Path) -> None:
    spec, source, root = _fixture(tmp_path)
    store = QwenArtifactStore(root, disk_free=lambda _: spec.total_bytes * 3)

    with store.provision_lock():
        with pytest.raises(ProvisionLockBusy):
            _install(
                store,
                spec,
                source,
                source_kind=ArtifactSourceKind.UPSTREAM,
                health_probe=lambda _: {"transcript": "ok"},
            )


def test_quick_verify_detects_post_activation_mutation(tmp_path: Path) -> None:
    spec, source, root = _fixture(tmp_path)
    store = QwenArtifactStore(root, disk_free=lambda _: spec.total_bytes * 3)
    _install(
        store,
        spec,
        source,
        source_kind=ArtifactSourceKind.UPSTREAM,
        health_probe=lambda _: {"transcript": "ok"},
    )
    config = spec.model_path(root) / "config.json"
    os_mode = stat.S_IMODE(config.stat().st_mode)
    config.chmod(0o600)
    content = config.read_bytes()
    config.write_bytes(content)
    config.chmod(os_mode)

    with pytest.raises(ArtifactInvalid, match="changed after activation"):
        store.verify(spec, deep=False)
    assert store.inspect(spec).state is ArtifactState.INVALID


def test_deep_verify_runs_health_probe_without_storing_transcript(tmp_path: Path) -> None:
    spec, source, root = _fixture(tmp_path)
    store = QwenArtifactStore(root, disk_free=lambda _: spec.total_bytes * 3)
    _install(
        store,
        spec,
        source,
        source_kind=ArtifactSourceKind.UPSTREAM,
        health_probe=lambda _: {"transcript": "first"},
    )

    receipt = store.verify(
        spec,
        deep=True,
        health_probe=lambda path: {"transcript": "second", "path": path.name},
    )

    assert receipt.health["transcript_nonempty"] is True
    assert "second" not in spec.receipt_path(root).read_text(encoding="utf-8")


def test_health_failure_leaves_only_private_staging_cleanup(tmp_path: Path) -> None:
    spec, source, root = _fixture(tmp_path)
    store = QwenArtifactStore(root, disk_free=lambda _: spec.total_bytes * 3)

    with pytest.raises(ArtifactInvalid, match="empty transcript"):
        _install(
            store,
            spec,
            source,
            source_kind=ArtifactSourceKind.UPSTREAM,
            health_probe=lambda _: {"transcript": ""},
        )

    assert not spec.model_path(root).exists()
    assert not tuple(root.glob(".staging-*"))


def test_health_receipt_rejects_sensitive_fields(tmp_path: Path) -> None:
    spec, source, root = _fixture(tmp_path)
    store = QwenArtifactStore(root, disk_free=lambda _: spec.total_bytes * 3)

    with pytest.raises(ArtifactInvalid, match="sensitive"):
        _install(
            store,
            spec,
            source,
            source_kind=ArtifactSourceKind.UPSTREAM,
            health_probe=lambda _: {"transcript": "ok", "api_token": "private"},
        )

    assert not spec.model_path(root).exists()
    assert not spec.receipt_path(root).exists()


def test_activation_permission_failure_rolls_back_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec, source, root = _fixture(tmp_path)
    store = QwenArtifactStore(root, disk_free=lambda _: spec.total_bytes * 3)

    def fail_read_only(_: Path) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(store, "_make_read_only", fail_read_only)
    with pytest.raises(ArtifactInvalid, match="activation"):
        _install(
            store,
            spec,
            source,
            source_kind=ArtifactSourceKind.UPSTREAM,
            health_probe=lambda _: {"transcript": "ok"},
        )

    assert not spec.model_path(root).exists()
    assert not spec.receipt_path(root).exists()


def test_gc_never_removes_active_generation(tmp_path: Path) -> None:
    spec, source, root = _fixture(tmp_path)
    store = QwenArtifactStore(root, disk_free=lambda _: spec.total_bytes * 3)
    _install(
        store,
        spec,
        source,
        source_kind=ArtifactSourceKind.UPSTREAM,
        health_probe=lambda _: {"transcript": "ok"},
    )
    stale = root / spec.key / ("c" * 40)
    stale.mkdir()
    (stale / "orphan").write_text("stale", encoding="utf-8")

    assert stale in store.gc((spec,), dry_run=True)
    with pytest.raises(ArtifactInvalid, match="active generation"):
        store.gc((spec,), generation=spec.upstream.revision)
    removed = store.gc((spec,), generation="c" * 40)

    assert removed == (stale,)
    assert not stale.exists()
    assert spec.model_path(root).exists()


def test_receipt_rejects_unknown_fields_and_never_contains_token(tmp_path: Path) -> None:
    spec, source, root = _fixture(tmp_path)
    store = QwenArtifactStore(root, disk_free=lambda _: spec.total_bytes * 3)
    _install(
        store,
        spec,
        source,
        source_kind=ArtifactSourceKind.UPSTREAM,
        health_probe=lambda _: {"transcript": "ok"},
    )
    receipt_path = spec.receipt_path(root)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert "token" not in payload
    assert "hf_token" not in payload
    assert all("token" not in key.casefold() for key in payload["health"])
    payload["unexpected"] = True
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path.chmod(0o600)

    with pytest.raises(ArtifactInvalid, match="fields"):
        store.load_receipt(spec)


def test_snapshot_resolver_uses_exact_selected_source_without_fallback(tmp_path: Path) -> None:
    spec, source, _ = _fixture(tmp_path)
    observed: dict[str, object] = {}

    def fetch(**kwargs) -> str:
        observed.update(kwargs)
        return str(source)

    resolved = QwenSnapshotResolver(fetch).resolve(
        spec,
        source_kind=ArtifactSourceKind.UPSTREAM,
        cache_only=True,
        token="secret-value",
    )

    assert resolved == source
    assert observed["repo_id"] == spec.upstream.repo_id
    assert observed["revision"] == spec.upstream.revision
    assert observed["local_files_only"] is True
    assert observed["token"] == "secret-value"
    assert set(observed["allow_patterns"]) == {entry.path for entry in spec.files}


def test_snapshot_resolver_wraps_source_failure_without_leaking_token(tmp_path: Path) -> None:
    spec, _, _ = _fixture(tmp_path)

    def fail(**kwargs) -> str:
        raise RuntimeError(f"provider failed with {kwargs['token']}")

    with pytest.raises(ArtifactInvalid) as caught:
        QwenSnapshotResolver(fail).resolve(
            spec,
            source_kind=ArtifactSourceKind.UPSTREAM,
            cache_only=False,
            token="do-not-log-this",
        )

    assert "do-not-log-this" not in str(caught.value)
