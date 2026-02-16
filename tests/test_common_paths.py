"""Tests for gs:// path resolution helpers."""

from pathlib import Path

from blueprint_pipeline.common import resolve_gs_uri_to_path


def test_resolve_gs_uri_prefers_flat_layout_when_bucket_root(tmp_path: Path) -> None:
    bucket_root = tmp_path / "bucket"
    bucket_root.mkdir(parents=True, exist_ok=True)

    resolved = resolve_gs_uri_to_path(
        "gs://bucket/scenes/scene_a/captures/cap_a/pipeline/.nurec_complete",
        bucket_root,
    )
    assert resolved == bucket_root / "scenes/scene_a/captures/cap_a/pipeline/.nurec_complete"


def test_resolve_gs_uri_ignores_stale_nested_bucket_dir(tmp_path: Path) -> None:
    bucket_root = tmp_path / "bucket"
    bucket_root.mkdir(parents=True, exist_ok=True)
    # Simulate bad residue from prior runs that created /<bucket>/<bucket>/...
    (bucket_root / "bucket").mkdir(parents=True, exist_ok=True)

    resolved = resolve_gs_uri_to_path(
        "gs://bucket/scenes/scene_a/captures/cap_a/pipeline/.nurec_complete",
        bucket_root,
    )
    assert resolved == bucket_root / "scenes/scene_a/captures/cap_a/pipeline/.nurec_complete"


def test_resolve_gs_uri_prefers_bucket_layout_when_bucket_dir_exists(tmp_path: Path) -> None:
    mount_root = tmp_path / "mnt_gcs"
    (mount_root / "bucket").mkdir(parents=True, exist_ok=True)

    resolved = resolve_gs_uri_to_path(
        "gs://bucket/scenes/scene_a/captures/cap_a/pipeline/.nurec_complete",
        mount_root,
    )
    assert resolved == mount_root / "bucket/scenes/scene_a/captures/cap_a/pipeline/.nurec_complete"
