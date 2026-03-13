from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.local_bundle_workflow import detect_bundle_identity, stage_local_bundle


def test_detect_bundle_identity_reads_raw_bundle_metadata(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    payloads = {
        "manifest.json": {"scene_id": "scene-1", "capture_id": "capture-1"},
        "capture_context.json": {"sceneId": "scene-1"},
        "capture_upload_complete.json": {"captureId": "capture-1"},
    }
    for name, payload in payloads.items():
        (raw_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    identity = detect_bundle_identity(tmp_path)
    assert identity.scene_id == "scene-1"
    assert identity.capture_id == "capture-1"


def test_stage_local_bundle_copy_strips_stale_object_index_derivatives(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    raw_dir = source_root / "raw"
    raw_dir.mkdir(parents=True)
    payloads = {
        "manifest.json": {"scene_id": "scene-1", "capture_id": "capture-1"},
        "capture_context.json": {"sceneId": "scene-1"},
        "capture_upload_complete.json": {"captureId": "capture-1"},
        "object_index.json": {"objects": [{"id": "old"}]},
        "object_index_build_report.json": {"status": "built"},
        "object_index_keyframes.json": {"keyframes": []},
        "object_grounding_hints.json": {"hints": []},
    }
    for name, payload in payloads.items():
        (raw_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    (raw_dir / "object_index_artifacts").mkdir()
    (raw_dir / "object_index_artifacts" / "stale.txt").write_text("stale", encoding="utf-8")

    capture_root = stage_local_bundle(
        source_bundle=source_root,
        storage_root=tmp_path / "storage",
        bucket="local-blueprint",
        mode="copy",
    )

    staged_raw = capture_root / "raw"
    assert (staged_raw / "manifest.json").is_file()
    assert (staged_raw / "capture_context.json").is_file()
    assert (staged_raw / "capture_upload_complete.json").is_file()
    assert not (staged_raw / "object_index.json").exists()
    assert not (staged_raw / "object_index_build_report.json").exists()
    assert not (staged_raw / "object_index_keyframes.json").exists()
    assert not (staged_raw / "object_grounding_hints.json").exists()
    assert not (staged_raw / "object_index_artifacts").exists()
