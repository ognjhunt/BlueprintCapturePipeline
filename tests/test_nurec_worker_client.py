"""Tests for NuRec worker client dispatch behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline.capture_bridge import CaptureDescriptor
from blueprint_pipeline.nurec_worker_client import NurecWorkerClient, NurecWorkerConfig


def _descriptor() -> CaptureDescriptor:
    return CaptureDescriptor.from_dict(
        {
            "schema_version": "v1",
            "scene_id": "scene_1",
            "capture_id": "cap_1",
            "capture_source": "iphone",
            "capture_tier": "tier1_iphone",
            "raw_prefix_uri": "gs://bucket/scenes/scene_1/iphone/cap_1/raw",
            "frames_index_uri": "gs://bucket/scenes/scene_1/captures/cap_1/frames/index.jsonl",
            "nurec_mode": "mono_pose_assisted",
        }
    )


def test_local_worker_dispatch_includes_repo_pythonpath(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(command, **kwargs):  # noqa: ANN001
        captured["command"] = command
        captured["env"] = kwargs.get("env", {})
        return _Proc()

    monkeypatch.setattr("blueprint_pipeline.nurec_worker_client.subprocess.run", _fake_run)

    client = NurecWorkerClient(
        storage_root=tmp_path,
        bucket="bucket",
        pipeline_prefix="scenes/scene_1/captures/cap_1/pipeline",
        config=NurecWorkerConfig(worker_mode="local_worker"),
    )
    spec_path = client.pipeline_dir / "nurec_job_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("{}", encoding="utf-8")

    client.dispatch(spec_path=spec_path)

    env = captured["env"]
    assert "PYTHONPATH" in env
    assert str(client._repo_src) in env["PYTHONPATH"]


def test_run_clears_stale_markers_before_dispatch(monkeypatch, tmp_path: Path) -> None:
    client = NurecWorkerClient(
        storage_root=tmp_path,
        bucket="bucket",
        pipeline_prefix="scenes/scene_1/captures/cap_1/pipeline",
        config=NurecWorkerConfig(worker_mode="external_markers"),
    )

    complete_marker = client.pipeline_dir / ".nurec_complete"
    failed_marker = client.pipeline_dir / ".nurec_failed"
    complete_marker.parent.mkdir(parents=True, exist_ok=True)
    complete_marker.write_text("stale", encoding="utf-8")
    failed_marker.write_text("stale", encoding="utf-8")

    def _fake_wait() -> None:
        (client.pipeline_dir / ".nurec_complete").write_text("fresh", encoding="utf-8")

    monkeypatch.setattr(client, "wait_for_completion", _fake_wait)
    monkeypatch.setattr(client, "collect_outputs", lambda: {"status": "completed"})

    out = client.run(
        descriptor=_descriptor(),
        descriptor_uri="gs://bucket/scenes/scene_1/captures/cap_1/capture_descriptor.json",
        object_index_uri="gs://bucket/scenes/scene_1/iphone/cap_1/raw/arkit/objects/index.json",
    )

    assert out["status"] == "completed"
    assert complete_marker.read_text(encoding="utf-8") == "fresh"
    assert not failed_marker.exists()


def _write_collision_mesh(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "3 0 1 2",
            ]
        ),
        encoding="utf-8",
    )


def test_collect_outputs_includes_new_visual_artifacts(tmp_path: Path) -> None:
    client = NurecWorkerClient(
        storage_root=tmp_path,
        bucket="bucket",
        pipeline_prefix="scenes/scene_1/captures/cap_1/pipeline",
        config=NurecWorkerConfig(worker_mode="external_markers"),
    )
    nurec_dir = client.nurec_dir
    nurec_dir.mkdir(parents=True, exist_ok=True)
    (nurec_dir / "export_last.usdz").write_bytes(b"usdz")
    _write_collision_mesh(nurec_dir / "nvblox_mesh.ply")
    (nurec_dir / "occupancy.bin").write_bytes(b"occ")
    (nurec_dir / "visual_mesh.glb").write_bytes(b"glb")
    (nurec_dir / "visual_pointcloud.ply").write_bytes(b"ply")
    (nurec_dir / "mesh_manifest.json").write_text("{}", encoding="utf-8")

    payload = client.collect_outputs()
    artifacts = payload["artifacts"]
    assert "visual_usdz" in artifacts
    assert "collision_mesh_ply" in artifacts
    assert "visual_mesh_glb" in artifacts
    assert "visual_pointcloud_ply" in artifacts
    assert "mesh_manifest_json" in artifacts


def test_collect_outputs_respects_visual_mesh_disable_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VISUAL_MESH_ENABLED", "false")
    client = NurecWorkerClient(
        storage_root=tmp_path,
        bucket="bucket",
        pipeline_prefix="scenes/scene_1/captures/cap_1/pipeline",
        config=NurecWorkerConfig(worker_mode="external_markers"),
    )
    nurec_dir = client.nurec_dir
    nurec_dir.mkdir(parents=True, exist_ok=True)
    (nurec_dir / "export_last.usdz").write_bytes(b"usdz")
    _write_collision_mesh(nurec_dir / "nvblox_mesh.ply")
    (nurec_dir / "occupancy.bin").write_bytes(b"occ")

    payload = client.collect_outputs()
    artifacts = payload["artifacts"]
    assert "visual_usdz" in artifacts
    assert "visual_mesh_glb" not in artifacts


def test_collect_outputs_includes_optional_scene_cleaning_artifacts(tmp_path: Path) -> None:
    client = NurecWorkerClient(
        storage_root=tmp_path,
        bucket="bucket",
        pipeline_prefix="scenes/scene_1/captures/cap_1/pipeline",
        config=NurecWorkerConfig(worker_mode="external_markers"),
    )
    nurec_dir = client.nurec_dir
    nurec_dir.mkdir(parents=True, exist_ok=True)
    (nurec_dir / "export_last.usdz").write_bytes(b"usdz")
    _write_collision_mesh(nurec_dir / "nvblox_mesh.ply")
    (nurec_dir / "occupancy.bin").write_bytes(b"occ")
    (nurec_dir / "visual_mesh.glb").write_bytes(b"glb")
    (nurec_dir / "inpainted_visual_mesh.glb").write_bytes(b"cleaned")
    (nurec_dir / "instance_masks").mkdir(parents=True, exist_ok=True)
    (nurec_dir / "instance_masks" / "frame_00001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (nurec_dir / "colmap_undistorted" / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    (nurec_dir / "colmap_undistorted" / "sparse" / "0" / "cameras.bin").write_bytes(b"\x00")
    (nurec_dir / "colmap_undistorted" / "images").mkdir(parents=True, exist_ok=True)
    (nurec_dir / "colmap_undistorted" / "images" / "frame_00001.jpg").write_bytes(b"jpg")

    payload = client.collect_outputs()
    artifacts = payload["artifacts"]
    assert "inpainted_visual_mesh_glb" in artifacts
    assert "sam3_instance_masks_dir" in artifacts
    assert "colmap_undistorted_sparse_dir" in artifacts
    assert "colmap_undistorted_images_dir" in artifacts


def test_collect_outputs_ignores_traversal_in_manifest_primary_visual(tmp_path: Path) -> None:
    client = NurecWorkerClient(
        storage_root=tmp_path,
        bucket="bucket",
        pipeline_prefix="scenes/scene_1/captures/cap_1/pipeline",
        config=NurecWorkerConfig(worker_mode="external_markers"),
    )
    nurec_dir = client.nurec_dir
    nurec_dir.mkdir(parents=True, exist_ok=True)
    export_usdz = nurec_dir / "export_last.usdz"
    export_usdz.write_bytes(b"safe")
    _write_collision_mesh(nurec_dir / "nvblox_mesh.ply")
    (nurec_dir / "occupancy.bin").write_bytes(b"occ")
    (nurec_dir / "visual_mesh.glb").write_bytes(b"glb")

    outside = tmp_path / "secret.usdz"
    outside.write_bytes(b"secret")
    (nurec_dir / "mesh_manifest.json").write_text(
        '{"primary_visual_asset": "../../../../../secret.usdz"}',
        encoding="utf-8",
    )

    payload = client.collect_outputs()
    assert payload["artifacts"]["visual_usdz"].endswith("/export_last.usdz")


def test_collect_outputs_allows_scoped_manifest_primary_visual(tmp_path: Path) -> None:
    client = NurecWorkerClient(
        storage_root=tmp_path,
        bucket="bucket",
        pipeline_prefix="scenes/scene_1/captures/cap_1/pipeline",
        config=NurecWorkerConfig(worker_mode="external_markers"),
    )
    nurec_dir = client.nurec_dir
    nurec_dir.mkdir(parents=True, exist_ok=True)
    (nurec_dir / "export_last.usdz").write_bytes(b"safe")
    _write_collision_mesh(nurec_dir / "nvblox_mesh.ply")
    (nurec_dir / "occupancy.bin").write_bytes(b"occ")
    (nurec_dir / "visual_mesh.glb").write_bytes(b"glb")

    alt = nurec_dir / "variants" / "alt.usdz"
    alt.parent.mkdir(parents=True, exist_ok=True)
    alt.write_bytes(b"alt")
    (nurec_dir / "mesh_manifest.json").write_text(
        '{"primary_visual_asset": "variants/alt.usdz"}',
        encoding="utf-8",
    )

    payload = client.collect_outputs()
    assert payload["artifacts"]["visual_usdz"].endswith("/variants/alt.usdz")
