from __future__ import annotations

from copy import deepcopy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.deformable_scene_visual_review import (
    DeformableSceneVisualReviewError,
    RECEIPT_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    materialize_deformable_scene_visual_review,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _render() -> dict:
    result = {
        "schema_version": "public_scene_splat_render.v1",
        "cameras": [
            {
                "id": "cloth_a",
                "bytes": 100,
                "digest": _sha("a"),
                "nonblank": True,
            },
            {
                "id": "basket_a",
                "bytes": 200,
                "digest": _sha("b"),
                "nonblank": True,
            },
            {
                "id": "basket_b",
                "bytes": 300,
                "digest": _sha("c"),
                "nonblank": True,
            },
        ],
        "render_manifest_digest": "",
    }
    result["render_manifest_digest"] = canonical_digest(
        result, digest_field="render_manifest_digest"
    )
    return result


def _topology() -> dict:
    result = {
        "schema_version": "interiorgs_sage_collision_component_topology.v1",
        "targets": [
            {
                "interiorgs_instance_id": "79",
                "component_collision_identity_passed": True,
                "opening_probe": None,
            },
            {
                "interiorgs_instance_id": "87",
                "component_collision_identity_passed": True,
                "opening_probe": {"open_collision_cavity_passed": True},
            },
            {
                "interiorgs_instance_id": "108",
                "component_collision_identity_passed": False,
                "opening_probe": None,
            },
        ],
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    return result


def _request(render: dict, topology: dict) -> dict:
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "scene_id": "fixture",
        "reviewer_id": "fixture-reviewer",
        "reviewed_at": "2026-08-10T00:00:00Z",
        "learned_policy_outcomes_inspected": False,
        "reconnaissance_only": True,
        "render_manifest_digest": render["render_manifest_digest"],
        "collision_topology_receipt_digest": topology["receipt_digest"],
        "targets": [
            {
                "target_id": "cloth_79",
                "publisher_instance_id": "79",
                "target_kind": "movable_deformable",
                "material_class": "towel_or_cloth",
                "material_class_supported_by_observation": True,
                "rest_state": "rolled",
                "support_relation": "observed_deformable_stack",
                "rigid_exterior_observed": False,
                "open_rim_observed": False,
                "interior_occupied": False,
                "complete_interior_appearance_observed": False,
                "cited_frames": [{"camera_id": "cloth_a", "size_bytes": 100, "sha256": _sha("a")}],
                "review_notes": "Observed rolled towel in a stack.",
            },
            {
                "target_id": "basket_87",
                "publisher_instance_id": "87",
                "target_kind": "destination_receptacle",
                "material_class": "not_applicable",
                "material_class_supported_by_observation": False,
                "rest_state": "not_applicable",
                "support_relation": "observed_container_contents",
                "rigid_exterior_observed": True,
                "open_rim_observed": True,
                "interior_occupied": True,
                "complete_interior_appearance_observed": False,
                "cited_frames": [
                    {"camera_id": "basket_a", "size_bytes": 200, "sha256": _sha("b")},
                    {"camera_id": "basket_b", "size_bytes": 300, "sha256": _sha("c")},
                ],
                "review_notes": "Open rim and exterior observed; contents hide the floor.",
            },
            {
                "target_id": "basket_108",
                "publisher_instance_id": "108",
                "target_kind": "destination_receptacle",
                "material_class": "not_applicable",
                "material_class_supported_by_observation": False,
                "rest_state": "not_applicable",
                "support_relation": "observed_container_contents",
                "rigid_exterior_observed": True,
                "open_rim_observed": True,
                "interior_occupied": True,
                "complete_interior_appearance_observed": False,
                "cited_frames": [{"camera_id": "basket_b", "size_bytes": 300, "sha256": _sha("c")}],
                "review_notes": "Filled alternative without qualified collision identity.",
            },
        ],
    }


def test_review_derives_engineered_twin_basis_without_promoting_hidden_truth() -> None:
    render = _render()
    topology = _topology()

    result = materialize_deformable_scene_visual_review(
        _request(render, topology),
        render_manifest=render,
        collision_topology=topology,
    )

    by_id = {row["target_id"]: row for row in result["targets"]}
    assert by_id["cloth_79"]["selection_role"] == "selected_movable_design_basis"
    assert by_id["basket_87"]["selection_role"] == "engineered_twin_design_basis"
    assert by_id["basket_87"]["source_destination_admitted"] is False
    assert by_id["basket_108"]["selection_role"] == "rejected_destination_candidate"
    assert result["composition_required"] is True
    assert result["schema_version"] == RECEIPT_SCHEMA_VERSION
    assert result["claim_boundary"]["engineered_twin_hidden_geometry_is_source_truth"] is False
    assert result["review_digest"].startswith("sha256:")


def test_exact_empty_observed_receptacle_can_be_source_destination() -> None:
    render = _render()
    topology = _topology()
    request = _request(render, topology)
    basket = request["targets"][1]
    basket["interior_occupied"] = False
    basket["complete_interior_appearance_observed"] = True

    result = materialize_deformable_scene_visual_review(
        request, render_manifest=render, collision_topology=topology
    )

    basket_result = next(row for row in result["targets"] if row["target_id"] == "basket_87")
    assert basket_result["selection_role"] == "source_destination"
    assert result["composition_required"] is False


def test_tampered_frame_identity_fails_closed() -> None:
    render = _render()
    topology = _topology()
    request = _request(render, topology)
    request["targets"][0]["cited_frames"][0]["sha256"] = _sha("f")

    with pytest.raises(DeformableSceneVisualReviewError) as caught:
        materialize_deformable_scene_visual_review(
            request, render_manifest=render, collision_topology=topology
        )

    assert "visual_review_frame_identity_invalid:cloth_a" in caught.value.errors


def test_caller_cannot_assert_admission_or_policy_outcomes() -> None:
    render = _render()
    topology = _topology()
    request = _request(render, topology)
    request["admitted"] = True
    request["learned_policy_outcomes_inspected"] = True

    with pytest.raises(DeformableSceneVisualReviewError) as caught:
        materialize_deformable_scene_visual_review(
            request, render_manifest=render, collision_topology=topology
        )

    assert "visual_review_caller_asserted_outcome_forbidden" in caught.value.errors
    assert "visual_review_policy_outcome_leakage" in caught.value.errors


def test_engineered_twin_basis_does_not_truthify_source_cavity() -> None:
    render = _render()
    topology = _topology()
    tampered = deepcopy(topology)
    tampered["targets"][1]["opening_probe"]["open_collision_cavity_passed"] = False
    tampered["receipt_digest"] = canonical_digest(tampered, digest_field="receipt_digest")
    request = _request(render, tampered)

    result = materialize_deformable_scene_visual_review(
        request, render_manifest=render, collision_topology=tampered
    )

    basket = next(row for row in result["targets"] if row["target_id"] == "basket_87")
    assert basket["open_collision_cavity_passed"] is False
    assert basket["source_destination_admitted"] is False
    assert basket["engineered_twin_design_basis_admitted"] is True
    assert basket["selection_role"] == "engineered_twin_design_basis"


def test_engineered_twin_basis_still_requires_registered_outer_identity() -> None:
    render = _render()
    topology = _topology()
    tampered = deepcopy(topology)
    tampered["targets"][1]["component_collision_identity_passed"] = False
    tampered["receipt_digest"] = canonical_digest(tampered, digest_field="receipt_digest")

    with pytest.raises(DeformableSceneVisualReviewError) as caught:
        materialize_deformable_scene_visual_review(
            _request(render, tampered),
            render_manifest=render,
            collision_topology=tampered,
        )

    assert "visual_review_destination_basis_not_exactly_one" in caught.value.errors
