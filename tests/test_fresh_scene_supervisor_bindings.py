from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.fresh_scene_supervisor_bindings import (
    FreshSceneSupervisorBindingError,
    compile_fresh_scene_supervisor_bindings,
    materialize_fresh_scene_supervisor_bindings,
)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _status(path: Path) -> Path:
    value = {
        "schema_version": "fresh_scene_paired_target_preparation.v1",
        "status": "blocked",
        "task_count": 1,
        "task_ids": ["task_a"],
        "first_blocker": "fresh_scene_sam31_task_inputs_missing",
        "next_required_stage": "sam31_task_inputs",
        "status_digest": "",
    }
    value["status_digest"] = canonical_digest(value, digest_field="status_digest")
    return _write(path, value)


def _sam_request(root: Path) -> Path:
    for name in ("calibrated.json", "task.json", "profile.json", "prompts.json"):
        _write(root / name, {})
    ffmpeg = root / "ffmpeg"
    ffmpeg.write_bytes(b"fixture")
    value = {
        "schema_version": "fresh_scene_sam31_task_input_tool_request.v1",
        "calibrated_view_receipt_path": str(root / "calibrated.json"),
        "task_freeze_path": str(root / "task.json"),
        "provider_profile_path": str(root / "profile.json"),
        "prompts_path": str(root / "prompts.json"),
        "ffmpeg_executable": str(ffmpeg),
        "frame_rate_hz": 1,
        "request_digest": "",
    }
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return _write(root / "sam-request.json", value)


def _mask_request(root: Path) -> Path:
    for name in ("task.json", "tracks.json", "cameras.json", "review.json"):
        _write(root / name, {})
    images = root / "images"
    images.mkdir(parents=True)
    (images / "camera_0.png").write_bytes(b"fixture-png")
    value = {
        "schema_version": "fresh_scene_calibrated_mask_tool_request.v1",
        "task_freeze_paths": [str(root / "task.json")],
        "task_inputs": {
            "task_a": {
                "source_track_result_path": str(root / "tracks.json"),
                "camera_contract_path": str(root / "cameras.json"),
                "source_image_root": str(images),
                "camera_frame_map": {"camera_0": "task_a:camera_0"},
            }
        },
        "selected_track_ids_by_task": {"task_a": ["track-a"]},
        "reviewed_track_selection_receipt_path": str(root / "review.json"),
        "request_digest": "",
    }
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return _write(root / "mask-request.json", value)


def _removal_request(root: Path) -> Path:
    for name in (
        "scene.ply",
        "collision.usda",
        "registered.json",
        "task.json",
        "tracks.json",
        "cameras.json",
        "image.png",
        "mask.png",
    ):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"fixture")
    def record(name: str) -> dict:
        path = root / name
        import hashlib

        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    def relative_record(name: str) -> dict:
        absolute = record(name)
        absolute.pop("path")
        return {"relative_path": name, **absolute}
    mask_receipt = {
        "tasks": [
            {
                "task_id": "task_a",
                "task_freeze": record("task.json"),
                "source_track_result": record("tracks.json"),
                "camera_contract": record("cameras.json"),
                "source_images": [{"image": relative_record("image.png")}],
                "masks": [{"mask": relative_record("mask.png")}],
            }
        ]
    }
    masks_path = _write(root / "masks.json", mask_receipt)
    value = {
        "schema_version": "fresh_scene_removal_freeze_tool_request.v1",
        "source_standard_splat_path": str(root / "scene.ply"),
        "source_collision_path": str(root / "collision.usda"),
        "registered_frame_receipt_path": str(root / "registered.json"),
        "calibrated_mask_set_receipt_path": str(masks_path),
        "tasks": {
            "task_a": {
                "target_collision_prim_path": "/Root/Target",
                "scene": {"task_id": "task_a"},
                "policy": {"minimum_core_camera_count": 2},
                "historical_baseline": {"method": "fixture"},
            }
        },
        "request_digest": "",
    }
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return _write(root / "removal-request.json", value)


def _cutout_request(root: Path) -> Path:
    for name in ("scene.ply", "task.json", "sweep.json"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"fixture")
    arrays = []
    for index in range(2):
        path = root / f"contribution-{index}.npz"
        path.write_bytes(f"fixture-{index}".encode())
        import hashlib

        arrays.append(
            {
                "relative_path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = _write(root / "manifest.json", {"repetitions": arrays})
    value = {
        "schema_version": "fresh_scene_segment_cutout_tool_request.v1",
        "source_standard_splat_path": str(root / "scene.ply"),
        "task_freeze_paths": [str(root / "task.json")],
        "sweep_freeze_paths_by_task": {"task_a": str(root / "sweep.json")},
        "contribution_manifest_paths_by_task": {"task_a": str(manifest)},
        "request_digest": "",
    }
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return _write(root / "cutout-request.json", value)


def test_host_resident_manifest_compiles_exact_agents_sdk_bindings(tmp_path: Path) -> None:
    status = _status(tmp_path / "status.json")
    request = _sam_request(tmp_path / "inputs")
    manifest_path = tmp_path / "binding.json"

    manifest = materialize_fresh_scene_supervisor_bindings(
        preparation_status_path=status,
        sam31_task_input_request_path=request,
        output_path=manifest_path,
        roots=[tmp_path],
    )
    compiled = compile_fresh_scene_supervisor_bindings(
        manifest_path, roots=[tmp_path]
    )

    assert manifest["agent_receives_paths"] is False
    assert manifest["paid_execution_authorized"] is False
    assert manifest["provider_mutations_performed"] == 0
    assert compiled["requested_tool_ids"] == [
        "inspect_fresh_scene_preparation",
        "materialize_sam31_task_inputs",
    ]
    assert set(compiled["context_bindings"]) == {
        "fresh_scene_preparation_status",
        "fresh_scene_sam31_task_input_request",
    }
    assert (
        compiled["context_bindings"]["fresh_scene_sam31_task_input_request"][
            "request_digest"
        ]
        == json.loads(request.read_text())["request_digest"]
    )


def test_binding_rejects_outside_root_and_changed_request(tmp_path: Path) -> None:
    resident = tmp_path / "resident"
    resident.mkdir()
    status = _status(resident / "status.json")
    outside = _sam_request(tmp_path / "outside")
    with pytest.raises(
        FreshSceneSupervisorBindingError,
        match="fresh_scene_tool_request_not_host_resident",
    ):
        materialize_fresh_scene_supervisor_bindings(
            preparation_status_path=status,
            sam31_task_input_request_path=outside,
            output_path=resident / "bad-binding.json",
            roots=[resident],
        )

    request = _sam_request(resident / "inputs")
    manifest_path = resident / "binding.json"
    materialize_fresh_scene_supervisor_bindings(
        preparation_status_path=status,
        sam31_task_input_request_path=request,
        output_path=manifest_path,
        roots=[resident],
    )
    request.write_text("{}", encoding="utf-8")
    with pytest.raises(
        FreshSceneSupervisorBindingError,
        match="fresh_scene_tool_request_bytes_changed",
    ):
        compile_fresh_scene_supervisor_bindings(manifest_path, roots=[resident])


def test_binding_rejects_changed_request_input_bytes(tmp_path: Path) -> None:
    status = _status(tmp_path / "status.json")
    request = _sam_request(tmp_path / "inputs")
    manifest_path = tmp_path / "binding.json"
    materialize_fresh_scene_supervisor_bindings(
        preparation_status_path=status,
        sam31_task_input_request_path=request,
        output_path=manifest_path,
        roots=[tmp_path],
    )
    (tmp_path / "inputs/prompts.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(
        FreshSceneSupervisorBindingError,
        match="fresh_scene_tool_request_input_bytes_changed",
    ):
        compile_fresh_scene_supervisor_bindings(manifest_path, roots=[tmp_path])


def test_mask_binding_rehashes_human_track_review_receipt(tmp_path: Path) -> None:
    status = _status(tmp_path / "status.json")
    request = _mask_request(tmp_path / "mask-inputs")
    manifest_path = tmp_path / "binding.json"
    materialize_fresh_scene_supervisor_bindings(
        preparation_status_path=status,
        calibrated_mask_request_path=request,
        output_path=manifest_path,
        roots=[tmp_path],
    )
    (tmp_path / "mask-inputs/review.json").write_text(
        '{"changed":true}', encoding="utf-8"
    )
    with pytest.raises(
        FreshSceneSupervisorBindingError,
        match="fresh_scene_tool_request_input_bytes_changed",
    ):
        compile_fresh_scene_supervisor_bindings(manifest_path, roots=[tmp_path])


def test_removal_binding_rehashes_all_scientific_inputs(tmp_path: Path) -> None:
    status = _status(tmp_path / "status.json")
    request = _removal_request(tmp_path / "removal-inputs")
    manifest_path = tmp_path / "binding.json"
    manifest = materialize_fresh_scene_supervisor_bindings(
        preparation_status_path=status,
        removal_freeze_request_path=request,
        output_path=manifest_path,
        roots=[tmp_path],
    )
    compiled = compile_fresh_scene_supervisor_bindings(manifest_path, roots=[tmp_path])
    assert compiled["requested_tool_ids"] == [
        "inspect_fresh_scene_preparation",
        "materialize_fresh_scene_removal_freezes",
    ]
    assert len(
        manifest["tool_requests"]["fresh_scene_removal_freeze_request"][
            "input_inventory"
        ]
    ) == 9
    (tmp_path / "removal-inputs/scene.ply").write_bytes(b"changed")
    with pytest.raises(
        FreshSceneSupervisorBindingError,
        match="fresh_scene_tool_request_input_bytes_changed",
    ):
        compile_fresh_scene_supervisor_bindings(manifest_path, roots=[tmp_path])


def test_cutout_binding_rehashes_contribution_arrays(tmp_path: Path) -> None:
    status = _status(tmp_path / "status.json")
    request = _cutout_request(tmp_path / "cutout-inputs")
    manifest_path = tmp_path / "binding.json"
    manifest = materialize_fresh_scene_supervisor_bindings(
        preparation_status_path=status,
        segment_cutout_request_path=request,
        output_path=manifest_path,
        roots=[tmp_path],
    )
    compiled = compile_fresh_scene_supervisor_bindings(manifest_path, roots=[tmp_path])
    assert compiled["requested_tool_ids"] == [
        "inspect_fresh_scene_preparation",
        "materialize_fresh_scene_segment_cutout",
    ]
    assert len(
        manifest["tool_requests"]["fresh_scene_segment_cutout_request"][
            "input_inventory"
        ]
    ) == 6
    (tmp_path / "cutout-inputs/contribution-0.npz").write_bytes(b"changed")
    with pytest.raises(
        FreshSceneSupervisorBindingError,
        match="fresh_scene_tool_request_input_bytes_changed",
    ):
        compile_fresh_scene_supervisor_bindings(manifest_path, roots=[tmp_path])
