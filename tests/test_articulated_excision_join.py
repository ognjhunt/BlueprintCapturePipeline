from __future__ import annotations

import math
import json
from pathlib import Path

import pytest

from blueprint_pipeline.articulated_excision_join import (
    ArticulatedExcisionJoinError,
    COVERAGE_SCHEMA_VERSION,
    JOIN_SCHEMA_VERSION,
    compile_articulated_excision_join,
    compile_coverage_conditioned_cutout_receipt,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


CAMERA_IDS = [
    "far_left",
    "far_right",
    "front_medium",
    "front_working",
    "left_translate",
    "low_right",
    "raised_left",
    "right_translate",
]
DOOR_STATES = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]
T_WORLD_ASSET = [
    [1.0, 0.0, 0.0, 1.9742142],
    [0.0, 1.0, 0.0, 1.4792181],
    [0.0, 0.0, 1.0, 2e-09],
    [0.0, 0.0, 0.0, 1.0],
]
REPO_ROOT = Path(__file__).resolve().parents[1]


def _digest(payload: dict, field: str) -> dict:
    payload = dict(payload)
    payload[field] = ""
    payload[field] = canonical_digest(payload, digest_field=field)
    return payload


def _ownership() -> dict:
    return _digest(
        {
            "schema_version": "adp009b_gaussian_excision_ownership.v1",
            "publisher_scene_id": "840796",
            "target_instance_id": "123",
            "source_gaussian_count": 593665,
            "owned_index_set_sha256": "sha256:" + "a" * 64,
            "ambiguous_index_set_sha256": "sha256:" + "b" * 64,
            "retained_scene_ply_sha256": "sha256:" + "c" * 64,
            "retained_scene_gaussian_count": 589874,
            "heldout_audit_passed": True,
            "heldout_maximum_residual_connected_component_pixels": 0,
            "heldout_maximum_protected_significant_pixels": 0,
            "receipt_digest": "",
        },
        "receipt_digest",
    )


def _collider_removal() -> dict:
    return _digest(
        {
            "schema_version": "source_collider_subtree_removal.v1",
            "publisher_scene_id": "840796",
            "removed_prim_path": "/Root/ZC2DFJJVAIJFUPTUJQ888888",
            "sage_collision_usd_sha256": "sha256:" + "d" * 64,
            "removed_scene_usd_sha256": "sha256:" + "e" * 64,
            "removed_prim_count": 1,
            "remaining_target_collision_prim_count": 0,
            "receipt_digest": "",
        },
        "receipt_digest",
    )


def _replacement() -> dict:
    return {
        "replacement_usd_sha256": "sha256:" + "f" * 64,
        "topology_receipt_digest": "sha256:" + "1" * 64,
        "physics_receipt_digest": "sha256:" + "2" * 64,
        "T_world_asset": [list(row) for row in T_WORLD_ASSET],
    }


def _door_matrix(*, classes: list[str], blocked: bool = False) -> dict:
    return _digest(
        {
            "schema_version": "articulated_door_state_clearance.v1",
            "status": (
                "blocked_by_door_state_contact"
                if blocked
                else "door_state_matrix_clearance_candidate_only"
            ),
            "door_state_rows": [
                {"angle_degrees": angle, "clear": not blocked} for angle in DOOR_STATES
            ],
            "static_obstacle_classes_bound": sorted(classes),
            "receipt_digest": "",
        },
        "receipt_digest",
    )


def _coverage(
    *,
    residual: int = 0,
    protected_changed: int = 0,
    contained: bool = True,
) -> dict:
    cells = []
    for camera_id in CAMERA_IDS:
        for angle in DOOR_STATES:
            cells.append(
                {
                    "camera_id": camera_id,
                    "door_state_angle_degrees": angle,
                    "residual_significant_pixels": residual,
                    "residual_max_connected_component_pixels": residual,
                    "residual_inside_target_core_mask": contained,
                    "outside_mask_changed_pixels": protected_changed,
                }
            )
    return _digest(
        {
            "schema_version": COVERAGE_SCHEMA_VERSION,
            "camera_ids": list(CAMERA_IDS),
            "door_state_angles_degrees": list(DOOR_STATES),
            "cells": cells,
            "maximum_residual_connected_component_pixels": 4,
            "maximum_protected_changed_pixels": 0,
            "coverage_qualified": True,
            "caller_asserted_coverage_accepted": False,
            "rendered_pixels_changed_by_audit": False,
            "receipt_digest": "",
        },
        "receipt_digest",
    )


def _bound_cutout() -> dict:
    return _digest(
        {
            "schema_version": "adp009b_bound_index_union_candidate.v1",
            "status": "bound_cutout_materialized_pending_coverage_and_seam_gates",
            "counts": {
                "source": 100,
                "deleted_total": 10,
                "retained_total": 90,
            },
            "outputs": {
                "deleted_source_indices": {"sha256": "sha256:" + "3" * 64},
                "retained_source_indices": {"sha256": "sha256:" + "4" * 64},
                "retained_scene_gaussians": {"sha256": "sha256:" + "5" * 64},
            },
            "preservation": {
                "retained_rows_byte_exact": True,
                "retained_order_matches_source": True,
                "retained_vertex_count": 90,
                "source_vertex_count": 100,
            },
            "selection": {
                "caller_asserted_coverage": False,
                "heldout_pixels_used_to_select_indices": False,
                "learned_policy_outcomes_used": False,
            },
            "receipt_digest": "",
        },
        "receipt_digest",
    )


def _join(**overrides):
    arguments = {
        "ownership_receipt": _ownership(),
        "collider_removal_receipt": _collider_removal(),
        "replacement_binding": _replacement(),
        "door_state_receipt": _door_matrix(
            classes=["replacement_body", "replacement_lower_door", "franka_base"]
        ),
        "coverage_receipt": _coverage(),
        "expected_T_world_asset": [list(row) for row in T_WORLD_ASSET],
        "expected_camera_ids": list(CAMERA_IDS),
        "expected_door_state_angles_degrees": list(DOOR_STATES),
    }
    arguments.update(overrides)
    return compile_articulated_excision_join(**arguments)


def test_join_resolves_inpainting_not_required_when_no_residue() -> None:
    decision = _join()

    assert decision["schema_version"] == JOIN_SCHEMA_VERSION
    assert decision["status"] == "join_admitted"
    assert decision["inpainting_policy"] == "inpainting_not_required"
    assert decision["claim_boundary"]["hidden_interior_is_observed_truth"] is False
    assert decision["claim_boundary"]["native_simulator_qualified"] is False
    assert decision["receipt_digest"].startswith("sha256:")


def test_join_permits_only_narrow_contained_seam_repair() -> None:
    decision = _join(coverage_receipt=_coverage(residual=3, contained=True))

    assert decision["status"] == "join_admitted"
    assert decision["inpainting_policy"] == "narrow_mask_contained_seam_repair_only"


def test_join_accepts_coverage_conditioned_cutout_without_claiming_ownership() -> None:
    coverage = _coverage(residual=3, contained=True)
    cutout = compile_coverage_conditioned_cutout_receipt(
        bound_cutout_candidate=_bound_cutout(),
        coverage_receipt=coverage,
    )

    decision = _join(
        ownership_receipt=cutout,
        coverage_receipt=coverage,
    )

    assert cutout["factual_gaussian_ownership_claimed"] is False
    assert cutout["broad_inpainting_authorized"] is False
    assert decision["status"] == "join_admitted"
    assert decision["inpainting_policy"] == "narrow_mask_contained_seam_repair_only"
    assert decision["bindings"]["cutout_method"] == (
        "byte_exact_deletion_plus_actual_usd_coverage"
    )


def test_coverage_conditioned_cutout_rejects_unqualified_coverage() -> None:
    coverage = _coverage()
    coverage["coverage_qualified"] = False
    coverage["receipt_digest"] = canonical_digest(
        coverage, digest_field="receipt_digest"
    )

    with pytest.raises(
        ArticulatedExcisionJoinError,
        match="coverage_conditioned_cutout_coverage_not_qualified",
    ):
        compile_coverage_conditioned_cutout_receipt(
            bound_cutout_candidate=_bound_cutout(),
            coverage_receipt=coverage,
        )


def test_join_blocks_uncontained_or_oversized_residue() -> None:
    with pytest.raises(ArticulatedExcisionJoinError) as excinfo:
        _join(coverage_receipt=_coverage(residual=3, contained=False))
    assert any(
        "residual_outside_target_core_mask" in error for error in excinfo.value.errors
    )

    with pytest.raises(ArticulatedExcisionJoinError) as excinfo:
        _join(coverage_receipt=_coverage(residual=9, contained=True))
    assert any(
        "residual_component_above_threshold" in error for error in excinfo.value.errors
    )


def test_join_blocks_changed_protected_pixels() -> None:
    with pytest.raises(ArticulatedExcisionJoinError) as excinfo:
        _join(coverage_receipt=_coverage(protected_changed=2))

    assert any(
        "untouched_scene_pixels_changed" in error for error in excinfo.value.errors
    )


def test_join_blocks_unbound_door_state_classes() -> None:
    with pytest.raises(ArticulatedExcisionJoinError) as excinfo:
        _join(door_state_receipt=_door_matrix(classes=["replacement_body"]))

    assert any(
        "door_state_obstacle_classes_incomplete" in error
        for error in excinfo.value.errors
    )


def test_join_blocks_replacement_transform_mismatch() -> None:
    replacement = _replacement()
    replacement["T_world_asset"][0][3] = 2.1

    with pytest.raises(ArticulatedExcisionJoinError) as excinfo:
        _join(replacement_binding=replacement)

    assert any(
        "replacement_world_transform_mismatch" in error for error in excinfo.value.errors
    )


def test_join_blocks_failed_heldout_audit_and_tampered_digest() -> None:
    ownership = _ownership()
    ownership["heldout_audit_passed"] = False
    ownership["receipt_digest"] = ""
    ownership["receipt_digest"] = canonical_digest(
        ownership, digest_field="receipt_digest"
    )
    with pytest.raises(ArticulatedExcisionJoinError) as excinfo:
        _join(ownership_receipt=ownership)
    assert any("heldout_audit_not_passed" in error for error in excinfo.value.errors)

    tampered = _ownership()
    tampered["source_gaussian_count"] = 1
    with pytest.raises(ArticulatedExcisionJoinError) as excinfo:
        _join(ownership_receipt=tampered)
    assert any(
        "ownership_receipt_digest_invalid" in error for error in excinfo.value.errors
    )


def test_join_blocks_incomplete_coverage_grid() -> None:
    coverage = _coverage()
    coverage["cells"] = coverage["cells"][:-1]
    coverage["receipt_digest"] = ""
    coverage["receipt_digest"] = canonical_digest(
        coverage, digest_field="receipt_digest"
    )

    with pytest.raises(ArticulatedExcisionJoinError) as excinfo:
        _join(coverage_receipt=coverage)

    assert any("coverage_grid_incomplete" in error for error in excinfo.value.errors)


def test_join_requires_finite_transform() -> None:
    with pytest.raises(ArticulatedExcisionJoinError) as excinfo:
        _join(expected_T_world_asset=[[math.nan] * 4] * 4)

    assert any(
        "expected_transform_invalid" in error for error in excinfo.value.errors
    )


def test_checked_in_840796_construction_manifest_is_digest_bound() -> None:
    path = (
        REPO_ROOT
        / "docs/arm_decision_proof_v1/manifests"
        / "second_scene_840796_coverage_conditioned_construction.v1.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["manifest_digest"] == canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    assert manifest["status"] == (
        "construction_join_admitted_native_qualification_unobserved"
    )
    assert {
        binding["role"] for binding in manifest["construction_bindings"]
    } == {
        "byte_exact_gaussian_cutout_union",
        "actual_usd_target_core_coverage",
        "eight_camera_twelve_state_hybrid_review",
        "source_collider_subtree_removal",
        "articulated_replacement_usd",
        "construction_join",
    }
    assert manifest["claim_boundary"] == {
        "all_deleted_gaussians_factually_owned_by_source_object": False,
        "hidden_background_recovered": False,
        "native_simulator_qualified": False,
        "physical_equivalence_proven": False,
        "construction_join_admitted": True,
    }
