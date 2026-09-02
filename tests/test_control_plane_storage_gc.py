from __future__ import annotations

import hashlib
import os

import pytest

from blueprint_pipeline.control_plane_storage_gc import (
    ControlPlaneStorageGCError,
    EXECUTE_ACK,
    apply_gc_manifest,
    build_gc_manifest,
)


def _blob(root, payload: bytes):
    digest = hashlib.sha256(payload).hexdigest()
    path = root / digest
    path.write_bytes(payload)
    os.utime(path, (10, 10))
    return path


def test_gc_only_selects_old_unreferenced_verified_blobs(tmp_path) -> None:
    root = tmp_path / "sha256"
    root.mkdir()
    unreferenced = _blob(root, b"unreferenced")
    linked = _blob(root, b"linked")
    os.link(linked, tmp_path / "projection")
    young = _blob(root, b"young")
    os.utime(young, (95, 95))
    corrupt = root / ("f" * 64)
    corrupt.write_bytes(b"wrong digest")
    os.utime(corrupt, (10, 10))

    manifest = build_gc_manifest(
        content_store_roots=[root],
        minimum_age_seconds=20,
        now=lambda: 100,
    )

    assert manifest["candidate_count"] == 1
    assert manifest["candidate_bytes"] == len(b"unreferenced")
    assert manifest["candidates"][0]["digest"].endswith(unreferenced.name)
    assert manifest["retained_counts"] == {
        "linked": 1,
        "young": 1,
        "unsafe_or_unverified": 1,
    }
    assert manifest["evidence_roots_scanned"] is False


def test_gc_apply_requires_ack_and_rechecks_link_count(tmp_path) -> None:
    root = tmp_path / "sha256"
    root.mkdir()
    candidate = _blob(root, b"candidate")
    manifest = build_gc_manifest(
        content_store_roots=[root], minimum_age_seconds=0, now=lambda: 100
    )
    with pytest.raises(
        ControlPlaneStorageGCError,
        match="control_plane_storage_gc_apply_not_authorized",
    ):
        apply_gc_manifest(manifest, ack="wrong")

    os.link(candidate, tmp_path / "late-projection")
    changed = apply_gc_manifest(manifest, ack=EXECUTE_ACK)
    assert changed["removed_count"] == 0
    assert changed["skipped"] == [
        {"digest": "sha256:" + candidate.name, "reason": "candidate_changed"}
    ]
    assert candidate.exists()


def test_gc_apply_removes_only_manifest_candidates(tmp_path) -> None:
    root = tmp_path / "sha256"
    root.mkdir()
    candidate = _blob(root, b"candidate")
    manifest = build_gc_manifest(
        content_store_roots=[root], minimum_age_seconds=0, now=lambda: 100
    )

    result = apply_gc_manifest(manifest, ack=EXECUTE_ACK)

    assert result["removed_count"] == 1
    assert result["removed_bytes"] == len(b"candidate")
    assert result["evidence_removed"] is False
    assert not candidate.exists()


def test_gc_rejects_non_sha256_or_symlink_roots(tmp_path) -> None:
    unsafe = tmp_path / "evidence"
    unsafe.mkdir()
    with pytest.raises(
        ControlPlaneStorageGCError,
        match="control_plane_storage_gc_root_unsafe",
    ):
        build_gc_manifest(content_store_roots=[unsafe])

    safe = tmp_path / "safe" / "sha256"
    safe.mkdir(parents=True)
    alias = tmp_path / "sha256"
    alias.symlink_to(safe, target_is_directory=True)
    with pytest.raises(
        ControlPlaneStorageGCError,
        match="control_plane_storage_gc_root_unsafe",
    ):
        build_gc_manifest(content_store_roots=[alias])
