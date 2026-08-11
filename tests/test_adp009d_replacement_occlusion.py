from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.gaussian_splat_decode import (
    SplatData,
    read_standard_3dgs_ply,
    verify_standard_3dgs_ply_subset_exact,
    write_standard_3dgs_ply,
    write_standard_3dgs_ply_subset_exact,
)
from blueprint_pipeline.public_scene_replacement_occlusion import (
    CONTRIBUTION_SCHEMA,
    COVERAGE_SCHEMA,
    DIRECT_EVIDENCE_EXPANSION_SET_SCHEMA,
    OWNERSHIP_COVERAGE_CUTOUT_CANDIDATE_SCHEMA,
    OWNERSHIP_COVERAGE_CUTOUT_SET_SCHEMA,
    REQUEST_SCHEMA,
    ReplacementOcclusionError,
    build_replacement_occlusion_request,
    classify_gaussian_contributions,
    coverage_safe_ambiguous,
    materialize_bound_index_union_candidate,
    materialize_direct_evidence_expansion_candidate,
    materialize_direct_evidence_expansion_set,
    materialize_ownership_coverage_cutout_candidate,
    materialize_ownership_coverage_cutout_set,
    materialize_replacement_occlusion_cutout,
    select_direct_calibration_evidence_expansion,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, root: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _source_splat(path: Path, *, count: int = 6) -> Path:
    values = np.arange(count, dtype=np.float32)
    splat = SplatData(
        count=count,
        xyz=np.stack((values, values + 10, values + 20), axis=1),
        opacity=values + 30,
        f_dc=np.stack((values + 40, values + 50, values + 60), axis=1),
        scales=np.stack((values + 70, values + 80, values + 90), axis=1),
        quats=np.stack((values + 100, values + 110, values + 120, values + 130), axis=1),
        properties=(),
        sh_rest=None,
    )
    return write_standard_3dgs_ply(splat, path)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _coverage_task_freeze(task_id: str, slot: int) -> dict[str, object]:
    """Build a valid, task-neutral rigid fixture for the 1--5 object seam."""

    payload: dict[str, object] = {
        "schema_version": "dual_task_task_freeze.v1",
        "task_id": task_id,
        "prompt": f"relocate independently observed fixture object {slot}",
        "task_kind": "rigid_object_manipulation",
        "scene_freeze_digest": _digest("a"),
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "frozen_before_learned_policy_execution": True,
        "learned_policy_outcomes_accessed": False,
        "source_object": {
            "instance_id": f"fixture_source_{slot}",
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
            "source_collider_prim_path": f"/Root/fixture_source_{slot}",
            "collider_deletion_id": f"collider_deletion_{slot}",
            "replacement_asset_id": f"replacement_asset_{slot}",
            "replacement_qualification_id": f"replacement_qualification_{slot}",
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
            "support_id": f"destination_support_{slot}",
            "release_required": True,
        },
        "articulation_graph": None,
        "task_freeze_digest": "",
    }
    payload["task_freeze_digest"] = canonical_digest(
        payload, digest_field="task_freeze_digest"
    )
    return payload


def _coverage_cutout_inputs(
    tmp_path: Path,
    *,
    source: Path,
    task_id: str,
    slot: int,
    owned: list[int],
    ambiguous: list[int],
) -> tuple[Path, Path, Path]:
    """Write one frozen calibration-only ownership lane for a fixture task."""

    task = _coverage_task_freeze(task_id, slot)
    task_path = tmp_path / "tasks" / f"{task_id}.json"
    _write_json(task_path, task)
    excision: dict[str, object] = {
        "schema_version": "adp009b_gaussian_excision_audit_freeze.v1",
        "status": "frozen_before_excision_execution",
        "learned_policy_outcomes_observed": False,
        "replacement_usd_inserted": False,
        "source_standard_splat": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
        },
        "scene": {
            "task_id": task_id,
            "target_instance_id": task["source_object"]["instance_id"],
            "removal_id": task["removal_plan"]["removal_id"],
            "mask_set_id": task["removal_plan"]["mask_set_id"],
        },
        "freeze_digest": "",
    }
    excision["freeze_digest"] = canonical_digest(excision, digest_field="freeze_digest")
    excision_path = tmp_path / "excision" / f"{task_id}.json"
    _write_json(excision_path, excision)

    ownership_root = tmp_path / "ownership" / task_id
    ownership_root.mkdir(parents=True)
    selected = set(owned) | set(ambiguous)
    source_count = read_standard_3dgs_ply(source).count
    retained = [index for index in range(source_count) if index not in selected]
    array_paths = {
        "owned_indices": ownership_root / "owned.npy",
        "ambiguous_indices": ownership_root / "ambiguous.npy",
        "retained_indices": ownership_root / "retained.npy",
        "historical_obb_source_indices": ownership_root / "historical_obb.npy",
        "protected_camera_count": ownership_root / "protected_camera_count.npy",
        "core_camera_count": ownership_root / "core_camera_count.npy",
        "core_fraction": ownership_root / "core_fraction.npy",
        "geometry_score": ownership_root / "geometry_score.npy",
    }
    np.save(
        array_paths["owned_indices"],
        np.asarray(sorted(owned), dtype=np.int64),
        allow_pickle=False,
    )
    np.save(
        array_paths["ambiguous_indices"],
        np.asarray(sorted(ambiguous), dtype=np.int64),
        allow_pickle=False,
    )
    np.save(
        array_paths["retained_indices"],
        np.asarray(retained, dtype=np.int64),
        allow_pickle=False,
    )
    np.save(
        array_paths["historical_obb_source_indices"],
        np.asarray(sorted(selected), dtype=np.int64),
        allow_pickle=False,
    )
    protected = np.zeros(source_count, dtype=np.int16)
    core_count = np.zeros(source_count, dtype=np.int16)
    core_fraction = np.zeros(source_count, dtype=np.float64)
    geometry = np.zeros(source_count, dtype=np.float64)
    for index in ambiguous:
        core_count[index] = 2
        core_fraction[index] = 0.95
        geometry[index] = 1.0
    np.save(array_paths["protected_camera_count"], protected, allow_pickle=False)
    np.save(array_paths["core_camera_count"], core_count, allow_pickle=False)
    np.save(array_paths["core_fraction"], core_fraction, allow_pickle=False)
    np.save(array_paths["geometry_score"], geometry, allow_pickle=False)
    ownership: dict[str, object] = {
        "schema_version": "adp009b_gaussian_excision_ownership_receipt.v1",
        "status": "three_way_ownership_materialized_heldout_not_evaluated",
        "freeze_digest": excision["freeze_digest"],
        "source_standard_splat": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
        },
        "ownership": {
            "source_gaussian_count": source_count,
            "owned_count": len(owned),
            "retained_count": len(retained),
            "ambiguous_count": len(ambiguous),
            "exhaustive": True,
            "pairwise_disjoint": True,
        },
        "heldout_cameras_accessed_for_classification": False,
        "replacement_usd_inserted": False,
        "outputs": {
            name: _record(path, ownership_root) for name, path in array_paths.items()
        },
        "receipt_digest": "",
    }
    ownership["receipt_digest"] = canonical_digest(
        ownership, digest_field="receipt_digest"
    )
    ownership_path = ownership_root / "ownership.json"
    _write_json(ownership_path, ownership)
    return task_path, excision_path, ownership_path


def test_exact_subset_writer_preserves_source_vertex_rows(tmp_path: Path) -> None:
    source = _source_splat(tmp_path / "source.ply")
    output = write_standard_3dgs_ply_subset_exact(
        source, tmp_path / "retained.ply", np.array([0, 2, 5], dtype=np.int64)
    )

    proof = verify_standard_3dgs_ply_subset_exact(source, output, [0, 2, 5])

    assert proof == {
        "source_vertex_count": 6,
        "retained_vertex_count": 3,
        "row_size_bytes": 56,
        "retained_rows_byte_exact": True,
        "retained_order_matches_source": True,
    }
    assert read_standard_3dgs_ply(output).xyz[:, 0].tolist() == [0.0, 2.0, 5.0]


def test_three_way_contribution_classification_and_coverage_conditioning() -> None:
    result = classify_gaussian_contributions(
        np.array([9, 1, 5, 6, 0, 8], dtype=float),
        np.array([1, 9, 5, 4, 0, 2], dtype=float),
        retained_max_foreground_fraction=0.2,
        owned_min_foreground_fraction=0.8,
        minimum_total_contribution=0.1,
    )
    ambiguous = np.flatnonzero(result["ambiguous"])
    safe = coverage_safe_ambiguous(
        ambiguous,
        np.array([2, 3], dtype=np.int64),
        np.ones((2, 2), dtype=float),
        np.array([[0.0, 0.2], [0.0, 0.0]], dtype=float),
        minimum_cell_visible_contribution=0.1,
        maximum_uncovered_fraction=0.0,
        maximum_uncovered_contribution=0.0,
    )

    assert np.flatnonzero(result["owned"]).tolist() == [0, 5]
    assert np.flatnonzero(result["retained"]).tolist() == [1, 4]
    assert ambiguous.tolist() == [2, 3]
    assert safe.tolist() == [True, False]


def _packet(
    tmp_path: Path,
    *,
    scene_id: str,
    instance_id: str,
    semantic_label: str,
    leave_residual: bool,
) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    source = _source_splat(data / "inputs" / "scene_standard.ply")

    contribution_arrays = data / "evidence" / "contribution.npz"
    contribution_arrays.parent.mkdir(parents=True)
    np.savez(
        contribution_arrays,
        foreground_contribution=np.array([9, 1, 5, 6, 0, 8], dtype=np.float64),
        background_contribution=np.array([1, 9, 5, 4, 0, 2], dtype=np.float64),
    )
    scene = {
        "publisher_scene_id": scene_id,
        "target_instance_id": instance_id,
        "target_semantic_label": semantic_label,
    }
    contribution_manifest: dict[str, object] = {
        "schema_version": CONTRIBUTION_SCHEMA,
        "scene": scene,
        "source_standard_splat_sha256": _sha256(source),
        "method": {
            "name": "FlashSplat",
            "repository": "https://github.com/florinshen/FlashSplat",
            "commit": "a" * 40,
            "released_code_executed": True,
            "source_modified": False,
        },
        "arrays": _record(contribution_arrays, contribution_arrays.parent),
    }
    contribution_manifest["manifest_digest"] = canonical_digest(
        contribution_manifest, digest_field="manifest_digest"
    )
    contribution_manifest_path = data / "evidence" / "contribution_manifest.json"
    _write_json(contribution_manifest_path, contribution_manifest)

    camera_ids = ["external", "wrist"]
    angles = [0.0, 45.0]
    cells = [
        {
            "camera_id": camera,
            "commanded_door_angle_deg": angle,
            "readback_door_angle_deg": angle,
        }
        for camera in camera_ids
        for angle in angles
    ]
    removal_alpha = np.zeros((len(cells), 2, 2), dtype=np.float32)
    removal_alpha[:, 0, 0] = 1.0
    replacement_depth = np.full_like(removal_alpha, np.inf)
    replacement_depth[:, 0, 0] = 1.0
    if leave_residual:
        replacement_depth[-1, 0, 0] = np.inf
    coverage_arrays = data / "evidence" / "coverage.npz"
    np.savez(
        coverage_arrays,
        removal_alpha=removal_alpha,
        replacement_depth_m=replacement_depth,
        gaussian_indices=np.array([2, 3], dtype=np.int64),
        gaussian_visible_contribution=np.ones((len(cells), 2), dtype=np.float64),
        gaussian_uncovered_contribution=np.column_stack(
            (np.zeros(len(cells)), np.array([0.2, 0.0, 0.0, 0.0]))
        ),
    )
    coverage_manifest: dict[str, object] = {
        "schema_version": COVERAGE_SCHEMA,
        "scene": scene,
        "actual_mesh_depth_rasterized": True,
        "caller_supplied_coverage_mask": False,
        "renderer": {"name": "fixture_mesh_rasterizer", "version": "1"},
        "replacement_usd": {"sha256": "sha256:" + "b" * 64},
        "cells": cells,
        "arrays": _record(coverage_arrays, coverage_arrays.parent),
    }
    coverage_manifest["manifest_digest"] = canonical_digest(
        coverage_manifest, digest_field="manifest_digest"
    )
    coverage_manifest_path = data / "evidence" / "coverage_manifest.json"
    _write_json(coverage_manifest_path, coverage_manifest)

    request: dict[str, object] = {
        "schema_version": REQUEST_SCHEMA,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009B",
        "frozen_before_cutout": True,
        "learned_policy_outcomes_observed": False,
        "scene": scene,
        "inputs": {
            "source_standard_splat": _record(source, data),
            "contribution_manifest": _record(contribution_manifest_path, data),
            "coverage_manifest": _record(coverage_manifest_path, data),
        },
        "policy": {
            "retained_max_foreground_fraction": 0.2,
            "owned_min_foreground_fraction": 0.8,
            "minimum_total_contribution": 0.1,
            "minimum_cell_visible_contribution": 0.1,
            "maximum_ambiguous_uncovered_fraction": 0.0,
            "maximum_ambiguous_uncovered_contribution": 0.0,
            "confident_removal_alpha_threshold": 0.5,
            "maximum_confident_uncovered_pixels_per_cell": 0,
            "maximum_residual_alpha_fraction_per_cell": 0.0,
            "door_angle_readback_tolerance_deg": 0.01,
            "required_camera_ids": camera_ids,
            "required_door_angles_deg": angles,
        },
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path = repo / "requests" / "cutout.json"
    _write_json(request_path, request)
    return repo, data, request_path


@pytest.mark.parametrize(
    ("scene_id", "instance_id", "semantic_label", "leave_residual", "expected_status"),
    [
        ("840313", "160", "canned_beverage", False, "cutout_admitted_inpainting_not_required"),
        ("840796", "123", "refrigerator", True, "cutout_candidate_residual_measured"),
    ],
)
def test_materialize_scene_neutral_cutout_for_original_and_second_fixture(
    tmp_path: Path,
    scene_id: str,
    instance_id: str,
    semantic_label: str,
    leave_residual: bool,
    expected_status: str,
) -> None:
    repo, data, request_path = _packet(
        tmp_path,
        scene_id=scene_id,
        instance_id=instance_id,
        semantic_label=semantic_label,
        leave_residual=leave_residual,
    )

    receipt = materialize_replacement_occlusion_cutout(
        request_path=request_path,
        repo_root=repo,
        data_root=data,
        output_root=data / "output",
        receipt_output=repo / "receipts" / "cutout.json",
    )

    assert receipt["status"] == expected_status
    assert receipt["ownership"] == {
        **receipt["ownership"],
        "owned_count": 2,
        "retained_owned_evidence_count": 2,
        "ambiguous_count": 2,
        "coverage_safe_ambiguous_deleted_count": 1,
        "ambiguous_retained_count": 1,
        "total_deleted_count": 3,
        "total_retained_count": 3,
    }
    assert receipt["preservation"]["retained_rows_byte_exact"] is True
    assert np.load(data / "output" / "retained_source_indices.npy").tolist() == [1, 3, 4]
    assert read_standard_3dgs_ply(
        data / "output" / "scene_without_target_gaussians.ply"
    ).xyz[:, 0].tolist() == [1.0, 3.0, 4.0]
    assert receipt["source_collider_removed"] is False
    assert receipt["hybrid_render_qualified"] is False
    if leave_residual:
        assert receipt["inpainting_disposition"] == (
            "conditional_seam_ladder_required_for_measured_residual"
        )
        assert receipt["coverage"]["worst_confident_uncovered_pixel_count"] == 1
    else:
        assert receipt["inpainting_disposition"] == (
            "inpainting_not_required_by_replacement_occlusion"
        )
        assert receipt["coverage"]["worst_confident_uncovered_pixel_count"] == 0


def test_request_rejects_caller_asserted_inpainting_outcome() -> None:
    with pytest.raises(ReplacementOcclusionError) as exc:
        build_replacement_occlusion_request(
            {
                "schema_version": REQUEST_SCHEMA,
                "program_id": "arm-decision-proof-v1",
                "adp_item": "ADP-009B",
                "frozen_before_cutout": True,
                "learned_policy_outcomes_observed": False,
                "inpainting_not_required": True,
            }
        )
    assert "replacement_occlusion_caller_outcome_forbidden" in exc.value.codes


def test_direct_evidence_expansion_ignores_neighbor_score_and_outcomes() -> None:
    selected = select_direct_calibration_evidence_expansion(
        np.array([1, 2, 3, 4]),
        np.array([False, False, True, False, False]),
        np.array([0, 0, 0, 1, 0]),
        np.array([0, 2, 3, 4, 1]),
        np.array([0.0, 0.99, 1.0, 1.0, 1.0]),
        np.array([0.0, 0.9, 1.0, 1.0, 1.0]),
        minimum_core_camera_count=2,
        minimum_core_fraction=0.9,
        minimum_geometry_score=0.5,
    )
    assert selected.tolist() == [1]


def test_direct_evidence_expansion_materializes_byte_exact_candidate(
    tmp_path: Path,
) -> None:
    source = _source_splat(tmp_path / "source.ply")
    arrays = {
        "owned": np.array([0], dtype=np.int64),
        "candidate": np.array([1, 2, 3, 4], dtype=np.int64),
        "protected": np.array([0, 0, 0, 1, 0, 0], dtype=np.int16),
        "core_count": np.array([0, 2, 1, 3, 2, 0], dtype=np.int16),
        "core_fraction": np.array([0.0, 0.99, 1.0, 1.0, 0.5, 0.0]),
        "geometry": np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
    }
    paths = {}
    for name, values in arrays.items():
        path = tmp_path / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        paths[name] = path
    receipt = materialize_direct_evidence_expansion_candidate(
        source_standard_splat_path=source,
        owned_indices_path=paths["owned"],
        candidate_indices_path=paths["candidate"],
        protected_camera_count_path=paths["protected"],
        core_camera_count_path=paths["core_count"],
        core_fraction_path=paths["core_fraction"],
        geometry_score_path=paths["geometry"],
        output_root=tmp_path / "output",
        minimum_core_camera_count=2,
        minimum_core_fraction=0.9,
        minimum_geometry_score=0.5,
    )
    assert receipt["counts"] == {
        "source": 6,
        "owned": 1,
        "direct_evidence_expansion": 1,
        "deleted_total": 2,
        "retained_total": 4,
    }
    assert np.load(tmp_path / "output/deleted_source_indices.npy").tolist() == [0, 1]
    assert receipt["preservation"]["retained_rows_byte_exact"] is True


def test_bound_index_union_preserves_all_unselected_rows(tmp_path: Path) -> None:
    source = _source_splat(tmp_path / "source.ply")
    required = tmp_path / "required.npy"
    registered = tmp_path / "registered.npy"
    np.save(required, np.array([0, 4], dtype=np.int64), allow_pickle=False)
    np.save(registered, np.array([1, 4, 5], dtype=np.int64), allow_pickle=False)

    receipt = materialize_bound_index_union_candidate(
        source_standard_splat_path=source,
        required_deletion_indices_path=required,
        registered_volume_indices_path=registered,
        output_root=tmp_path / "output",
    )

    assert receipt["counts"] == {
        "source": 6,
        "required_deletion": 2,
        "registered_volume": 3,
        "registered_volume_only": 2,
        "deleted_total": 4,
        "retained_total": 2,
    }
    assert np.load(tmp_path / "output/deleted_source_indices.npy").tolist() == [0, 1, 4, 5]
    assert np.load(tmp_path / "output/retained_source_indices.npy").tolist() == [2, 3]
    assert receipt["preservation"]["retained_rows_byte_exact"] is True


def test_ownership_coverage_candidate_keeps_ambiguous_claims_conditional(
    tmp_path: Path,
) -> None:
    source = _source_splat(tmp_path / "source.ply")
    ownership_root = tmp_path / "ownership"
    ownership_root.mkdir()
    owned_path = ownership_root / "owned_source_indices.npy"
    ambiguous_path = ownership_root / "ambiguous_source_indices.npy"
    retained_path = ownership_root / "retained_source_indices.npy"
    np.save(owned_path, np.array([0, 1], dtype=np.int64), allow_pickle=False)
    np.save(ambiguous_path, np.array([2, 3], dtype=np.int64), allow_pickle=False)
    np.save(retained_path, np.array([4, 5], dtype=np.int64), allow_pickle=False)
    ownership = {
        "schema_version": "adp009b_gaussian_excision_ownership_receipt.v1",
        "status": "three_way_ownership_materialized_heldout_not_evaluated",
        "freeze_digest": "sha256:" + "1" * 64,
        "source_standard_splat": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
        },
        "ownership": {
            "source_gaussian_count": 6,
            "owned_count": 2,
            "retained_count": 2,
            "ambiguous_count": 2,
            "exhaustive": True,
            "pairwise_disjoint": True,
        },
        "heldout_cameras_accessed_for_classification": False,
        "replacement_usd_inserted": False,
        "outputs": {
            "owned_indices": _record(owned_path, ownership_root),
            "ambiguous_indices": _record(ambiguous_path, ownership_root),
            "retained_indices": _record(retained_path, ownership_root),
        },
        "receipt_digest": "",
    }
    ownership["receipt_digest"] = canonical_digest(
        ownership, digest_field="receipt_digest"
    )
    ownership_path = ownership_root / "ownership.json"
    _write_json(ownership_path, ownership)

    receipt = materialize_ownership_coverage_cutout_candidate(
        source_standard_splat_path=source,
        ownership_receipt_path=ownership_path,
        output_root=tmp_path / "cutout",
    )

    assert receipt["schema_version"] == OWNERSHIP_COVERAGE_CUTOUT_CANDIDATE_SCHEMA
    assert receipt["counts"] == {
        "source": 6,
        "owned": 2,
        "ambiguous_pending_coverage": 2,
        "deleted_total": 4,
        "retained_total": 2,
    }
    assert receipt["selection"]["heldout_pixels_used_to_select_indices"] is False
    assert receipt["claim_boundary"]["factual_gaussian_ownership_established"] is False
    assert np.load(tmp_path / "cutout/deleted_source_indices.npy").tolist() == [0, 1, 2, 3]
    assert receipt["preservation"]["retained_rows_byte_exact"] is True

    ownership["heldout_cameras_accessed_for_classification"] = True
    ownership["receipt_digest"] = canonical_digest(
        ownership, digest_field="receipt_digest"
    )
    _write_json(ownership_path, ownership)
    with pytest.raises(
        ReplacementOcclusionError,
        match="ownership_coverage_cutout_ownership_receipt_invalid",
    ):
        materialize_ownership_coverage_cutout_candidate(
            source_standard_splat_path=source,
            ownership_receipt_path=ownership_path,
            output_root=tmp_path / "blocked-cutout",
        )


def test_coverage_conditioned_successor_set_supports_five_independent_objects(
    tmp_path: Path,
) -> None:
    """One shared source splat can carry five independently frozen replacements."""

    source = _source_splat(tmp_path / "source.ply", count=15)
    task_paths: list[Path] = []
    excision_paths: dict[str, Path] = {}
    ownership_paths: dict[str, Path] = {}
    for slot in range(5):
        task_id = f"task_{slot}"
        task_path, excision_path, ownership_path = _coverage_cutout_inputs(
            tmp_path,
            source=source,
            task_id=task_id,
            slot=slot,
            owned=[slot * 2],
            ambiguous=[slot * 2 + 1],
        )
        task_paths.append(task_path)
        excision_paths[task_id] = excision_path
        ownership_paths[task_id] = ownership_path

    receipt = materialize_ownership_coverage_cutout_set(
        source_standard_splat_path=source,
        task_freeze_paths=task_paths,
        excision_freeze_paths_by_task=excision_paths,
        ownership_receipt_paths_by_task=ownership_paths,
        output_root=tmp_path / "successor-set",
    )

    assert receipt["schema_version"] == OWNERSHIP_COVERAGE_CUTOUT_SET_SCHEMA
    assert receipt["task_set"]["task_count"] == 5
    assert receipt["task_set"]["maximum_task_count"] == 5
    assert len(receipt["task_candidates"]) == 5
    assert receipt["shared_scene_union"]["counts"] == {
        "source": 15,
        "deleted_total": 10,
        "retained_total": 5,
    }
    assert receipt["selection"]["factual_gaussian_ownership_established_for_ambiguous_records"] is False
    assert receipt["claim_boundary"]["candidate_derived_layers_only"] is True
    assert receipt["shared_scene_union"]["preservation"]["retained_rows_byte_exact"] is True
    assert np.load(
        tmp_path / "successor-set/shared_scene_union/deleted_source_indices.npy"
    ).tolist() == list(range(10))
    assert np.load(
        tmp_path / "successor-set/shared_scene_union/retained_source_indices.npy"
    ).tolist() == list(range(10, 15))


def test_coverage_conditioned_successor_set_rejects_shared_deletion_before_write(
    tmp_path: Path,
) -> None:
    """Independent object lanes may not silently share one selected splat."""

    source = _source_splat(tmp_path / "source.ply", count=6)
    task_a, excision_a, ownership_a = _coverage_cutout_inputs(
        tmp_path,
        source=source,
        task_id="task_a",
        slot=0,
        owned=[0],
        ambiguous=[1],
    )
    task_b, excision_b, ownership_b = _coverage_cutout_inputs(
        tmp_path,
        source=source,
        task_id="task_b",
        slot=1,
        owned=[2],
        ambiguous=[1],
    )
    output = tmp_path / "blocked-successor-set"

    with pytest.raises(
        ReplacementOcclusionError,
        match="coverage_cutout_set_independent_candidate_overlap:task_a:task_b",
    ):
        materialize_ownership_coverage_cutout_set(
            source_standard_splat_path=source,
            task_freeze_paths=[task_a, task_b],
            excision_freeze_paths_by_task={"task_a": excision_a, "task_b": excision_b},
            ownership_receipt_paths_by_task={"task_a": ownership_a, "task_b": ownership_b},
            output_root=output,
        )

    assert not output.exists()


def test_direct_evidence_successor_set_supports_five_independent_objects(
    tmp_path: Path,
) -> None:
    """The selective calibration-only successor is also a 1--5 object seam."""

    source = _source_splat(tmp_path / "source.ply", count=15)
    task_paths: list[Path] = []
    excision_paths: dict[str, Path] = {}
    ownership_paths: dict[str, Path] = {}
    policies: dict[str, dict[str, object]] = {}
    for slot in range(5):
        task_id = f"task_{slot}"
        task_path, excision_path, ownership_path = _coverage_cutout_inputs(
            tmp_path,
            source=source,
            task_id=task_id,
            slot=slot,
            owned=[slot * 2],
            ambiguous=[slot * 2 + 1],
        )
        task_paths.append(task_path)
        excision_paths[task_id] = excision_path
        ownership_paths[task_id] = ownership_path
        policies[task_id] = {
            "minimum_core_camera_count": 2,
            "minimum_core_fraction": 0.95,
            "minimum_geometry_score": 0.0,
        }

    receipt = materialize_direct_evidence_expansion_set(
        source_standard_splat_path=source,
        task_freeze_paths=task_paths,
        excision_freeze_paths_by_task=excision_paths,
        ownership_receipt_paths_by_task=ownership_paths,
        selection_policy_by_task=policies,
        output_root=tmp_path / "direct-successor-set",
    )

    assert receipt["schema_version"] == DIRECT_EVIDENCE_EXPANSION_SET_SCHEMA
    assert receipt["task_set"]["task_count"] == 5
    assert len(receipt["task_candidates"]) == 5
    assert receipt["shared_scene_union"]["counts"] == {
        "source": 15,
        "deleted_total": 10,
        "retained_total": 5,
    }
    assert receipt["selection"]["heldout_pixels_used_to_select_indices"] is False
    assert (
        receipt["selection"]["factual_gaussian_ownership_established_for_direct_expansion"]
        is False
    )
    assert receipt["shared_scene_union"]["preservation"]["retained_rows_byte_exact"] is True
    assert np.load(
        tmp_path / "direct-successor-set/shared_scene_union/deleted_source_indices.npy"
    ).tolist() == list(range(10))


def test_direct_evidence_successor_set_rejects_shared_deletion_before_write(
    tmp_path: Path,
) -> None:
    """A direct-evidence expansion cannot silently consume another task's row."""

    source = _source_splat(tmp_path / "source.ply", count=6)
    task_a, excision_a, ownership_a = _coverage_cutout_inputs(
        tmp_path,
        source=source,
        task_id="task_a",
        slot=0,
        owned=[0],
        ambiguous=[1],
    )
    task_b, excision_b, ownership_b = _coverage_cutout_inputs(
        tmp_path,
        source=source,
        task_id="task_b",
        slot=1,
        owned=[2],
        ambiguous=[1],
    )
    output = tmp_path / "blocked-direct-successor-set"
    policy = {
        "minimum_core_camera_count": 2,
        "minimum_core_fraction": 0.95,
        "minimum_geometry_score": 0.0,
    }

    with pytest.raises(
        ReplacementOcclusionError,
        match="direct_expansion_set_independent_candidate_overlap:task_a:task_b",
    ):
        materialize_direct_evidence_expansion_set(
            source_standard_splat_path=source,
            task_freeze_paths=[task_a, task_b],
            excision_freeze_paths_by_task={"task_a": excision_a, "task_b": excision_b},
            ownership_receipt_paths_by_task={"task_a": ownership_a, "task_b": ownership_b},
            selection_policy_by_task={"task_a": policy, "task_b": policy},
            output_root=output,
        )

    assert not output.exists()
