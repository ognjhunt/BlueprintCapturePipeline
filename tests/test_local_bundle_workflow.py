from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.local_bundle_workflow import (
    detect_bundle_identity,
    run_local_bundle_workflow,
    stage_local_bundle,
)
from blueprint_pipeline.local_capture import resolve_local_capture_context


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_source_bundle(tmp_path: Path, *, scene_id: str = "scene_vm", capture_id: str = "cap_vm") -> Path:
    bundle_root = tmp_path / "download" / capture_id
    raw_root = bundle_root / "raw"
    _write_json(
        raw_root / "manifest.json",
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "capture_source": "iphone",
            "capture_modality": "iphone_arkit_lidar",
            "has_lidar": True,
            "video_uri": "raw/walkthrough.mov",
        },
    )
    _write_json(
        raw_root / "capture_context.json",
        {
            "sceneId": scene_id,
            "captureId": capture_id,
            "captureSource": "iphoneVideo",
            "captureModality": "iphone_arkit_lidar",
        },
    )
    _write_json(
        raw_root / "intake_packet.json",
        {
            "workflowName": "Desk reset",
            "taskSteps": ["approach", "clear"],
            "zone": "desk",
            "owner": "operator",
        },
    )
    _write_json(
        raw_root / "capture_upload_complete.json",
        {
            "sceneId": scene_id,
            "captureId": capture_id,
        },
    )
    (raw_root / "walkthrough.mov").write_bytes(b"mov")
    (raw_root / "arkit").mkdir(parents=True, exist_ok=True)
    (raw_root / "arkit" / "poses.jsonl").write_text("{}\n", encoding="utf-8")
    (raw_root / "arkit" / "intrinsics.json").write_text("{}", encoding="utf-8")
    return bundle_root


def test_detect_bundle_identity_rejects_mismatched_ids(tmp_path: Path) -> None:
    source_bundle = _build_source_bundle(tmp_path)
    _write_json(
        source_bundle / "raw" / "capture_context.json",
        {
            "sceneId": "different-scene",
            "captureId": "cap_vm",
        },
    )

    with pytest.raises(Exception, match="Conflicting scene IDs"):
        detect_bundle_identity(source_bundle)


def test_stage_local_bundle_links_raw_tree_and_refuses_overwrite(tmp_path: Path) -> None:
    source_bundle = _build_source_bundle(tmp_path)
    storage_root = tmp_path / "vm-storage"

    capture_root = stage_local_bundle(source_bundle=source_bundle, storage_root=storage_root)
    raw_root = capture_root / "raw"

    assert raw_root.is_symlink()
    assert raw_root.resolve() == (source_bundle / "raw").resolve()

    with pytest.raises(Exception, match="already exists"):
        stage_local_bundle(source_bundle=source_bundle, storage_root=storage_root)


def test_resolve_local_capture_context_points_to_staging_helper() -> None:
    with pytest.raises(Exception, match="stage_capture_bundle.py"):
        resolve_local_capture_context("/tmp/not-a-staged-bundle")


def test_run_local_bundle_workflow_writes_qualification_and_evaluation_prep_without_object_index(
    tmp_path: Path,
) -> None:
    source_bundle = _build_source_bundle(tmp_path, scene_id="scene_eval", capture_id="cap_eval")
    storage_root = tmp_path / "vm-storage"

    result = run_local_bundle_workflow(
        source_bundle=source_bundle,
        storage_root=storage_root,
        run_qualification=True,
        run_evaluation_prep=True,
    )

    capture_root = Path(str(result["capture_root"]))
    eval_dir = capture_root / "pipeline" / "evaluation_prep"
    object_geometry = json.loads((eval_dir / "object_geometry_manifest.json").read_text(encoding="utf-8"))
    eval_manifest = json.loads((eval_dir / "evaluation_prep_manifest.json").read_text(encoding="utf-8"))

    assert (capture_root / "capture_descriptor.json").is_file()
    assert (capture_root / "pipeline" / "scene_memory" / "scene_memory_manifest.json").is_file()
    assert object_geometry["status"] == "empty_object_index"
    assert object_geometry["object_index_present"] is True
    assert object_geometry["objects"] == []
    assert eval_manifest["artifacts"]["task_anchor_manifest"] == "task_anchor_manifest.json"
    assert eval_manifest["artifacts"]["site_world_spec"] == "site_world_spec.json"
    assert eval_manifest["artifacts"]["site_world_registration"] == "site_world_registration.json"
    assert eval_manifest["artifacts"]["site_world_health"] == "site_world_health.json"
