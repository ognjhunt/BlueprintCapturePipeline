"""Tests for standalone NuRec worker marker contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.nurec_worker import run_job_spec


def _write_job_spec(path: Path, *, scene_id: str = "scene_1", capture_id: str = "cap_1") -> None:
    payload = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "capture": {
            "raw_prefix_uri": f"gs://bucket/scenes/{scene_id}/iphone/{capture_id}/raw",
            "frames_index_uri": f"gs://bucket/scenes/{scene_id}/captures/{capture_id}/frames/index.jsonl",
        },
        "outputs": {
            "nurec_prefix": f"gs://bucket/scenes/{scene_id}/captures/{capture_id}/pipeline/nurec",
            "complete_marker": f"gs://bucket/scenes/{scene_id}/captures/{capture_id}/pipeline/.nurec_complete",
            "failed_marker": f"gs://bucket/scenes/{scene_id}/captures/{capture_id}/pipeline/.nurec_failed",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_nurec_worker_writes_complete_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage_root = tmp_path
    job_spec = tmp_path / "bucket/specs/nurec_job_spec.json"
    _write_job_spec(job_spec)

    nurec_dir = tmp_path / "bucket/scenes/scene_1/captures/cap_1/pipeline/nurec"
    nurec_dir.mkdir(parents=True, exist_ok=True)
    (nurec_dir / "export_last.usdz").write_bytes(b"usdz")
    (nurec_dir / "nvblox_mesh.ply").write_text(
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
    (nurec_dir / "occupancy.bin").write_bytes(b"occ")

    monkeypatch.setenv("NUREC_SKIP_PIPELINE_COMMAND", "true")
    code = run_job_spec(job_spec, storage_root=storage_root)

    complete_marker = tmp_path / "bucket/scenes/scene_1/captures/cap_1/pipeline/.nurec_complete"
    assert code == 0
    assert complete_marker.is_file()


def test_nurec_worker_writes_failed_marker_on_missing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage_root = tmp_path
    job_spec = tmp_path / "bucket/specs/nurec_job_spec.json"
    _write_job_spec(job_spec)
    monkeypatch.setenv("NUREC_SKIP_PIPELINE_COMMAND", "true")

    code = run_job_spec(job_spec, storage_root=storage_root)

    failed_marker = tmp_path / "bucket/scenes/scene_1/captures/cap_1/pipeline/.nurec_failed"
    assert code == 1
    assert failed_marker.is_file()
