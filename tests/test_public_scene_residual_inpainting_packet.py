from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.gaussian_splat_decode import (
    SplatData,
    write_standard_3dgs_ply,
    write_standard_3dgs_ply_subset_exact,
)
from blueprint_pipeline.public_scene_residual_inpainting_packet import (
    BACKEND_ADMISSION_SCHEMA,
    PACKET_SCHEMA,
    REQUEST_SCHEMA,
    ResidualInpaintingInputPacketError,
    build_residual_inpainting_input_request,
    materialize_residual_inpainting_input_packet,
)
from blueprint_pipeline.public_scene_replacement_depth_composition import (
    REQUEST_SCHEMA as DEPTH_COMPOSITION_REQUEST_SCHEMA,
    materialize_replacement_depth_composition,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _absolute_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _relative_record(root: Path, path: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _task_freeze(task_id: str, slot: int) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "dual_task_task_freeze.v1",
        "task_id": task_id,
        "prompt": f"relocate independently observed object {slot}",
        "task_kind": "rigid_object_manipulation",
        "scene_freeze_digest": _digest("a"),
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "frozen_before_learned_policy_execution": True,
        "learned_policy_outcomes_accessed": False,
        "source_object": {
            "instance_id": f"source_{slot}",
            "semantic_label": "fixture_object",
            "observed_bounds_world_m": {
                "minimum": [0.0, 0.0, 0.0],
                "maximum": [0.1, 0.1, 0.1],
            },
            "observed_pose_world": {
                "position_world_m": [0.05, 0.05, 0.05],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "support_or_attachment_id": f"support_{slot}",
            "collision_identity_receipt_digest": _digest("b"),
            "support_receipt_digest": _digest("c"),
            "franka_placement_packet_digest": _digest("d"),
            "visibility_receipt_digest": _digest("e"),
        },
        "removal_plan": {
            "removal_id": f"removal_{slot}",
            "mask_set_id": f"mask_set_{slot}",
            "source_collider_prim_path": f"/Root/source_{slot}",
            "collider_deletion_id": f"collider_{slot}",
            "replacement_asset_id": f"replacement_{slot}",
            "replacement_qualification_id": f"qualification_{slot}",
        },
        "cameras": {
            "external": f"external_{slot}",
            "wrist": f"wrist_{slot}",
            "overview": f"overview_{slot}",
        },
        "overview_camera_policy_input": False,
        "overview_camera_deterministic_scoring_input": False,
        "execution_contract": {
            "control_frequency_hz": 20,
            "maximum_steps": 200,
            "settle_window_steps": 10,
            "seeds": [slot + 1],
            "canonical_scenario_cell_id": f"canonical_{slot}",
            "reset_state": {"robot": "home", "object": "source_start"},
        },
        "deterministic_success_predicates": ["released", "settled"],
        "failure_rungs": ["never_moved", "collision_failure"],
        "target_configuration": {
            "kind": "pose_volume",
            "position_bounds_world_m": {
                "minimum": [0.2, 0.2, 0.0],
                "maximum": [0.3, 0.3, 0.1],
            },
            "orientation_reference_xyzw": [0.0, 0.0, 0.0, 1.0],
            "maximum_orientation_error_rad": 0.1,
            "support_id": f"destination_{slot}",
            "release_required": True,
        },
        "articulation_graph": None,
        "task_freeze_digest": "",
    }
    value["task_freeze_digest"] = canonical_digest(
        value, digest_field="task_freeze_digest"
    )
    return value


def _shared_splat(path: Path, *, count: int) -> Path:
    values = np.arange(count, dtype=np.float32)
    return write_standard_3dgs_ply(
        SplatData(
            count=count,
            xyz=np.stack((values, values + 10, values + 20), axis=1),
            opacity=values + 30,
            f_dc=np.stack((values + 40, values + 50, values + 60), axis=1),
            scales=np.stack((values + 70, values + 80, values + 90), axis=1),
            quats=np.stack((values + 100, values + 110, values + 120, values + 130), axis=1),
            properties=(),
            sh_rest=None,
        ),
        path,
    )


def _backend(path: Path, *, retention_days: int = 7) -> Path:
    value: dict[str, object] = {
        "schema_version": BACKEND_ADMISSION_SCHEMA,
        "status": "rights_admitted_for_private_derived_inpainting",
        "backend_id": "released_multiview_inpainting",
        "source_repository": "https://example.test/released-code",
        "source_revision": "deadbeef",
        "source_archive_sha256": _digest("1"),
        "environment_lock_sha256": _digest("2"),
        "model_identity": "released-model-v1",
        "private_derived_upload_policy": {
            "raw_dataset_bytes_upload": False,
            "private_derived_upload": True,
            "maximum_retention_days": retention_days,
            "provider_training": False,
            "publication": False,
        },
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    _write_json(path, value)
    return path


def _make_coverage(
    root: Path,
    *,
    task: dict[str, object],
    shared_digest: str,
    expected_asset_ids: list[str],
    camera_id: str,
    composition_path: Path,
) -> Path:
    mask = root / "uncovered_source_support_masks" / f"{camera_id}.png"
    mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.array([[0, 255], [0, 0]], dtype=np.uint8), mode="L").save(mask)
    removal = task["removal_plan"]
    assert isinstance(removal, dict)
    value: dict[str, object] = {
        "schema_version": "adp009b_source_layer_replacement_coverage_audit.v1",
        "status": "source_layer_coverage_measured",
        "task_id": task["task_id"],
        "task_freeze_digest": task["task_freeze_digest"],
        "removal_id": removal["removal_id"],
        "mask_set_id": removal["mask_set_id"],
        "replacement_asset_id": removal["replacement_asset_id"],
        "source_layer_splat_digest": shared_digest,
        "camera_ids": [camera_id],
        "uncovered_source_support_masks_are_inpainting_authority": True,
        "inpainting_mask_eligibility": {
            "full_resolution_source_frames": True,
            "full_resolution_replacement_depth": True,
            "calibrated_method_input_pair": True,
            "authorizes_only": "future_exact_mask_contained_multi_view_edit_input",
            "inpainting_result_qualified": False,
        },
        "replacement_depth_composition": _absolute_record(composition_path),
        "uncovered_source_support_masks": [
            {
                **_relative_record(root, mask),
                "camera_id": camera_id,
                "pixel_count": 1,
                "derived_from_all_state_cells": 3,
            }
        ],
        "manifest_digest": "",
    }
    value["manifest_digest"] = canonical_digest(value, digest_field="manifest_digest")
    path = root / "coverage.json"
    _write_json(path, value)
    return path


def _make_composition(
    path: Path,
    *,
    task: dict[str, object],
    expected_asset_ids: list[str],
) -> Path:
    root = path.parent / "composition_inputs"
    input_paths: list[Path] = []
    for index, asset_id in enumerate(expected_asset_ids):
        sweep_root = root / asset_id
        sweep_root.mkdir(parents=True, exist_ok=True)
        array = sweep_root / "depth.npy"
        depth = np.full((1, 2, 2), np.inf, dtype=np.float32)
        depth[0, index % 2, (index // 2) % 2] = float(index + 1)
        np.save(array, depth, allow_pickle=False)
        sweep: dict[str, object] = {
            "schema_version": "replacement_usd_depth_sweep.v2",
            "status": "actual_usd_geometry_depth_rasterized",
            "asset_id": asset_id,
            "task_freeze_digest": (
                task["task_freeze_digest"]
                if asset_id == task["removal_plan"]["replacement_asset_id"]
                else _digest(f"{index + 3:x}"[-1])
            ),
            "camera_contract_digest": _digest("a"),
            "camera_rows_digest": _digest("b"),
            "actual_usd_geometry_depth_rasterized": True,
            "caller_supplied_coverage_mask": False,
            "resolution_scale": 1.0,
            "cells": [{"camera_id": f"camera_{task['task_id'].split('_')[-1]}", "cell_id": "reset"}],
            "depth_dimensions": [2, 2],
            "arrays": _relative_record(sweep_root, array),
            "manifest_digest": "",
        }
        sweep["manifest_digest"] = canonical_digest(sweep, digest_field="manifest_digest")
        sweep_path = sweep_root / "sweep.json"
        _write_json(sweep_path, sweep)
        input_paths.append(sweep_path)
    request: dict[str, object] = {
        "schema_version": DEPTH_COMPOSITION_REQUEST_SCHEMA,
        "task_id": task["task_id"],
        "task_freeze_digest": task["task_freeze_digest"],
        "scored_task_asset_id": task["removal_plan"]["replacement_asset_id"],
        "frozen_before_removal_execution": True,
        "learned_policy_outcomes_accessed": False,
        "input_sweep_manifest_paths": [str(item) for item in input_paths],
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path = path.parent / "composition_request.json"
    _write_json(request_path, request)
    materialize_replacement_depth_composition(
        request_path=request_path, output_root=path.parent / "composition_receipt"
    )
    return path.parent / "composition_receipt" / "public_scene_replacement_depth_composition.v1.json"


def _make_render(root: Path, *, shared_digest: str, camera_id: str) -> Path:
    frame = root / "frames" / f"{camera_id}.png"
    frame.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((2, 2, 3), 127, dtype=np.uint8), mode="RGB").save(frame)
    value: dict[str, object] = {
        "schema_version": "sealed_camera_render_manifest.v1",
        "status": "rendered_exact_cameras",
        "authorization_class": "method_input",
        "splat_digest": shared_digest,
        "source_splat": {"retained_gaussian_count": 8},
        "calibrated_camera_file": {"binding": "caller_file_exact_match"},
        "calibrated_cameras": [{"id": camera_id}],
        "render_settings": {"dimensions": {"width": 2, "height": 2}},
        "renders": [{**_relative_record(root, frame), "camera_id": camera_id}],
        "sealed_camera_render_manifest_digest": "",
    }
    value["sealed_camera_render_manifest_digest"] = canonical_digest(
        value, digest_field="sealed_camera_render_manifest_digest"
    )
    path = root / "render.json"
    _write_json(path, value)
    return path


def _packet_inputs(tmp_path: Path, *, count: int = 2) -> tuple[Path, Path]:
    source_root = tmp_path / "candidate_set"
    source = _shared_splat(source_root / "source.ply", count=10)
    deleted_indices = np.array([1, 6], dtype=np.int64)
    retained_indices = np.array([0, 2, 3, 4, 5, 7, 8, 9], dtype=np.int64)
    shared_root = source_root / "shared_scene_union"
    shared_root.mkdir(parents=True, exist_ok=True)
    np.save(shared_root / "deleted_source_indices.npy", deleted_indices, allow_pickle=False)
    np.save(shared_root / "retained_source_indices.npy", retained_indices, allow_pickle=False)
    retained = write_standard_3dgs_ply_subset_exact(
        source, shared_root / "retained_scene_gaussians.ply", retained_indices
    )
    tasks: list[dict[str, object]] = []
    task_records: list[dict[str, object]] = []
    task_paths: list[Path] = []
    for slot in range(1, count + 1):
        task = _task_freeze(f"task_{slot}", slot)
        task_path = tmp_path / "task_freezes" / f"task_{slot}.json"
        _write_json(task_path, task)
        tasks.append(task)
        task_paths.append(task_path)
        removal = task["removal_plan"]
        assert isinstance(removal, dict)
        task_records.append(
            {
                "task_id": task["task_id"],
                "task_freeze_digest": task["task_freeze_digest"],
                "removal_id": removal["removal_id"],
                "mask_set_id": removal["mask_set_id"],
                "task_freeze": _absolute_record(task_path),
            }
        )
    candidate_set: dict[str, object] = {
        "schema_version": "adp009b_direct_evidence_expansion_set.v1",
        "task_candidates": task_records,
        "shared_scene_union": {
            "counts": {"source": 10, "deleted_total": 2, "retained_total": 8},
            "outputs": {
                "deleted_source_indices": _relative_record(
                    source_root, shared_root / "deleted_source_indices.npy"
                ),
                "retained_source_indices": _relative_record(
                    source_root, shared_root / "retained_source_indices.npy"
                ),
                "retained_scene_gaussians": _relative_record(source_root, retained),
            },
        },
        "source_standard_splat": _absolute_record(source),
        "claim_boundary": {"candidate_derived_layers_only": True},
        "receipt_digest": "",
    }
    candidate_set["receipt_digest"] = canonical_digest(
        candidate_set, digest_field="receipt_digest"
    )
    candidate_path = source_root / "candidate_set.json"
    _write_json(candidate_path, candidate_set)
    assets = [f"replacement_{slot}" for slot in range(1, count + 1)]
    lanes: list[dict[str, object]] = []
    for slot, task in enumerate(tasks, start=1):
        lane_root = tmp_path / "lanes" / f"task_{slot}"
        composition = _make_composition(
            lane_root / "composition.json", task=task, expected_asset_ids=assets
        )
        coverage = _make_coverage(
            lane_root / "coverage",
            task=task,
            shared_digest=_sha256(retained),
            expected_asset_ids=assets,
            camera_id=f"camera_{slot}",
            composition_path=composition,
        )
        render = _make_render(
            lane_root / "render", shared_digest=_sha256(retained), camera_id=f"camera_{slot}"
        )
        lanes.append(
            {
                "task_id": task["task_id"],
                "coverage_audit_path": str(coverage),
                "retained_render_manifest_path": str(render),
                "co_present_replacements_required": True,
            }
        )
    backend = _backend(tmp_path / "backend.json")
    request: dict[str, object] = {
        "schema_version": REQUEST_SCHEMA,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009D",
        "frozen_before_inpainting_execution": True,
        "learned_policy_outcomes_accessed": False,
        "candidate_set_path": str(candidate_path),
        "backend_admission_path": str(backend),
        "private_upload_policy": {
            "raw_dataset_bytes_upload": False,
            "private_derived_upload": True,
            "maximum_retention_days": 7,
            "provider_training": False,
            "publication": False,
        },
        "task_lanes": lanes,
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path = tmp_path / "request.json"
    _write_json(request_path, request)
    return request_path, candidate_path


def test_materializes_exact_mask_packet_for_five_independent_replacements(tmp_path: Path) -> None:
    request_path, _candidate = _packet_inputs(tmp_path, count=5)

    packet = materialize_residual_inpainting_input_packet(
        request_path=request_path, output_root=tmp_path / "packet"
    )

    assert packet["schema_version"] == PACKET_SCHEMA
    assert packet["replacement_object_count"] == 5
    assert packet["maximum_replacement_objects"] == 5
    assert packet["claim_boundary"]["released_code_inpainting_executed"] is False
    assert packet["claim_boundary"]["inpainting_result_qualified"] is False
    assert len(packet["lanes"]) == 5


def test_blocks_coverage_from_a_different_retained_scene(tmp_path: Path) -> None:
    request_path, _candidate = _packet_inputs(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    coverage_path = Path(request["task_lanes"][0]["coverage_audit_path"])
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["source_layer_splat_digest"] = _digest("f")
    coverage["manifest_digest"] = canonical_digest(coverage, digest_field="manifest_digest")
    _write_json(coverage_path, coverage)

    with pytest.raises(ResidualInpaintingInputPacketError, match="coverage_audit_invalid"):
        materialize_residual_inpainting_input_packet(
            request_path=request_path, output_root=tmp_path / "packet"
        )


def test_blocks_single_object_depth_coverage_in_a_co_present_scene(tmp_path: Path) -> None:
    request_path, _candidate = _packet_inputs(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    coverage_path = Path(request["task_lanes"][0]["coverage_audit_path"])
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    composition_path = Path(coverage["replacement_depth_composition"]["path"])
    composition = json.loads(composition_path.read_text(encoding="utf-8"))
    composition["replacement_asset_ids"] = ["replacement_1"]
    composition["receipt_digest"] = canonical_digest(
        composition, digest_field="receipt_digest"
    )
    _write_json(composition_path, composition)
    coverage["replacement_depth_composition"] = _absolute_record(composition_path)
    coverage["manifest_digest"] = canonical_digest(coverage, digest_field="manifest_digest")
    _write_json(coverage_path, coverage)

    with pytest.raises(
        ResidualInpaintingInputPacketError, match="co_present_depth_coverage_missing"
    ):
        materialize_residual_inpainting_input_packet(
            request_path=request_path, output_root=tmp_path / "packet"
        )


def test_blocks_reconnaissance_retained_render(tmp_path: Path) -> None:
    request_path, _candidate = _packet_inputs(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    render_path = Path(request["task_lanes"][0]["retained_render_manifest_path"])
    render = json.loads(render_path.read_text(encoding="utf-8"))
    render["authorization_class"] = "reconnaissance_preview"
    render["sealed_camera_render_manifest_digest"] = canonical_digest(
        render, digest_field="sealed_camera_render_manifest_digest"
    )
    _write_json(render_path, render)

    with pytest.raises(ResidualInpaintingInputPacketError, match="retained_render_invalid"):
        materialize_residual_inpainting_input_packet(
            request_path=request_path, output_root=tmp_path / "packet"
        )


def test_blocks_raw_upload_backend_admission(tmp_path: Path) -> None:
    request_path, _candidate = _packet_inputs(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    backend_path = Path(request["backend_admission_path"])
    backend = json.loads(backend_path.read_text(encoding="utf-8"))
    backend["private_derived_upload_policy"]["raw_dataset_bytes_upload"] = True
    backend["receipt_digest"] = canonical_digest(backend, digest_field="receipt_digest")
    _write_json(backend_path, backend)

    with pytest.raises(
        ResidualInpaintingInputPacketError, match="backend_private_upload_policy_invalid"
    ):
        materialize_residual_inpainting_input_packet(
            request_path=request_path, output_root=tmp_path / "packet"
        )


def test_request_requires_one_through_five_lanes_and_no_raw_upload() -> None:
    request = {
        "schema_version": REQUEST_SCHEMA,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009D",
        "frozen_before_inpainting_execution": True,
        "learned_policy_outcomes_accessed": False,
        "candidate_set_path": "/candidate.json",
        "backend_admission_path": "/backend.json",
        "private_upload_policy": {
            "raw_dataset_bytes_upload": True,
            "private_derived_upload": True,
            "maximum_retention_days": 7,
            "provider_training": False,
            "publication": False,
        },
        "task_lanes": [],
    }

    with pytest.raises(ResidualInpaintingInputPacketError) as error:
        build_residual_inpainting_input_request(request)

    assert "private_upload_policy_invalid" in str(error.value)
    assert "task_lane_count_invalid" in str(error.value)
