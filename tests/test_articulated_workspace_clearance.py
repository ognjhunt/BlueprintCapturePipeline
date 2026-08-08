from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.articulated_workspace_clearance import (
    ArticulatedWorkspaceClearanceError,
    evaluate_revolute_member_sweep,
    evaluate_revolute_member_sweep_against_sage_meshes,
    inventory_sage_sweep_obstacles,
    obstacles_from_sage_sweep_inventory,
    validate_sage_mesh_sweep,
    validate_articulated_workspace_clearance,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


ROOT = Path(__file__).resolve().parents[1]
REJECTION = (
    ROOT
    / "docs/arm_decision_proof_v1/manifests"
    / "second_scene_candidate_840411_clearance_rejection.v1.json"
)


def _obstacle(*, minimum: list[float], maximum: list[float], obstacle_id: str = "chair") -> dict:
    return {
        "obstacle_id": obstacle_id,
        "world_aabb_min_m": minimum,
        "world_aabb_max_m": maximum,
        "source_receipt_digest": "sha256:fixture",
    }


def test_original_rigid_fixture_has_no_articulated_sweep_contract() -> None:
    with pytest.raises(ArticulatedWorkspaceClearanceError) as caught:
        evaluate_revolute_member_sweep(
            hinge_origin_world_m=[0.0, 0.0, 0.0],
            closed_endpoint_world_m=[0.0, 0.0, 0.0],
            member_vertical_interval_m=[0.0, 1.0],
            start_angle_degrees=0.0,
            end_angle_degrees=45.0,
            obstacles=[],
        )

    assert caught.value.errors == ("sweep_member_radius_invalid",)


def test_840411_right_door_centerline_hits_chair_before_45_degrees() -> None:
    result = evaluate_revolute_member_sweep(
        hinge_origin_world_m=[4.475898768, 1.452210456, 0.0],
        closed_endpoint_world_m=[4.475898768, 1.9413037, 0.0],
        member_vertical_interval_m=[0.0, 1.826109993],
        start_angle_degrees=0.0,
        end_angle_degrees=45.0,
        obstacles=[
            _obstacle(
                obstacle_id="chair:227",
                minimum=[3.697225536, 1.345660441, 0.001594507],
                maximum=[4.274605183, 1.871812323, 0.856457869],
            )
        ],
        angular_resolution_degrees=0.25,
        member_half_thickness_m=0.0,
    )

    assert result["status"] == "blocked_by_observed_obstacle"
    assert result["first_collision"]["obstacle_id"] == "chair:227"
    assert 25.0 < result["first_collision"]["angle_degrees"] < 27.0
    assert result["first_collision"]["angle_degrees"] < 45.0
    assert result["claim_boundary"][
        "zero_thickness_centerline_collision_is_strong_rejection"
    ]
    assert validate_articulated_workspace_clearance(result) == result


def test_clear_sweep_remains_candidate_only() -> None:
    result = evaluate_revolute_member_sweep(
        hinge_origin_world_m=[0.0, 0.0, 0.0],
        closed_endpoint_world_m=[0.0, 0.5, 0.0],
        member_vertical_interval_m=[0.0, 1.0],
        start_angle_degrees=0.0,
        end_angle_degrees=45.0,
        obstacles=[
            _obstacle(minimum=[2.0, 2.0, 0.0], maximum=[3.0, 3.0, 1.0])
        ],
    )

    assert result["status"] == "clearance_candidate_only"
    assert result["first_collision"] is None
    assert result["claim_boundary"]["ik_or_contact_qualified"] is False


def test_840411_left_door_negative_sweep_hits_other_chair() -> None:
    result = evaluate_revolute_member_sweep(
        hinge_origin_world_m=[4.475898768, 2.430396944, 0.0],
        closed_endpoint_world_m=[4.475898768, 1.9413037, 0.0],
        member_vertical_interval_m=[0.0, 1.826109993],
        start_angle_degrees=0.0,
        end_angle_degrees=-45.0,
        obstacles=[
            _obstacle(
                obstacle_id="chair:226",
                minimum=[3.697225536, 1.980155103, 0.001594507],
                maximum=[4.274605183, 2.506306986, 0.856457869],
            )
        ],
    )

    assert result["status"] == "blocked_by_observed_obstacle"
    assert -25.0 < result["first_collision"]["angle_degrees"] < -24.0


def test_vertical_separation_does_not_false_positive() -> None:
    result = evaluate_revolute_member_sweep(
        hinge_origin_world_m=[0.0, 0.0, 2.0],
        closed_endpoint_world_m=[0.0, 0.5, 2.0],
        member_vertical_interval_m=[2.0, 3.0],
        start_angle_degrees=0.0,
        end_angle_degrees=90.0,
        obstacles=[
            _obstacle(minimum=[-1.0, -1.0, 0.0], maximum=[1.0, 1.0, 1.0])
        ],
    )

    assert result["status"] == "clearance_candidate_only"


def test_checked_rejection_binds_both_chairs_and_resumes_candidate_order() -> None:
    rejection = json.loads(REJECTION.read_text(encoding="utf-8"))

    validate_articulated_workspace_clearance(rejection["right_door_sweep"])
    validate_articulated_workspace_clearance(rejection["left_door_sweep"])
    assert rejection["status"] == "rejected_before_task_freeze"
    assert rejection["selection_effect"] == "resume_frozen_candidate_order_at_840796"
    assert rejection["learned_policy_outcomes_accessed"] is False
    exact = rejection["right_door_exact_sage_mesh_sweep"]
    assert exact["bound_interiorgs_obstacle_instance_id"] == "227"
    assert exact["first_collision_angle_degrees"] == 26.75
    assert exact["first_collision_angle_degrees"] < exact[
        "required_open_angle_degrees"
    ]
    assert exact["triangle_prism_intersection_tested"] is True
    assert rejection["record_digest"] == canonical_digest(
        rejection, digest_field="record_digest"
    )


def _mesh(stage, path: str, minimum, maximum, *, collision: bool = True) -> None:
    from pxr import UsdGeom, UsdPhysics

    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
    mesh.CreateFaceVertexIndicesAttr(
        [0, 1, 2, 3, 4, 7, 6, 5, 0, 4, 5, 1, 1, 5, 6, 2, 2, 6, 7, 3, 4, 0, 3, 7]
    )
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())


def test_full_sage_stage_inventory_excludes_only_target_and_binds_sweep(
    tmp_path: Path,
) -> None:
    import hashlib

    from pxr import Usd, UsdGeom

    collision = tmp_path / "collision.usda"
    stage = Usd.Stage.CreateNew(str(collision))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    _mesh(stage, "/Root/Fridge", (-0.1, -0.1, 0.0), (0.1, 0.1, 1.5))
    _mesh(stage, "/Root/Cabinet", (0.6, 0.4, 0.0), (0.9, 0.7, 1.5))
    _mesh(stage, "/Root/Far", (4.0, 4.0, 0.0), (5.0, 5.0, 1.5))
    _mesh(stage, "/Root/VisualOnly", (0.2, 0.2, 0.0), (0.3, 0.3, 1.0), collision=False)
    stage.GetRootLayer().Save()
    source_sha = "sha256:" + hashlib.sha256(collision.read_bytes()).hexdigest()
    identity = {
        "schema_version": "interiorgs_sage_collision_identity.v1",
        "source_files": {"sage_collision_usd": {"sha256": source_sha}},
        "target": {"interiorgs_instance_id": "123", "semantic_label": "refrigerator"},
        "whole_object_matches": [{"prim_path": "/Root/Fridge"}],
        "receipt_digest": "",
    }
    identity["receipt_digest"] = canonical_digest(identity, digest_field="receipt_digest")
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    inventory = inventory_sage_sweep_obstacles(
        sage_collision_usd_path=collision,
        target_collision_identity_receipt_path=identity_path,
        hinge_origin_world_m=[0.0, 0.0, 0.5],
        closed_endpoint_world_m=[1.0, 0.0, 0.5],
        member_vertical_interval_m=[0.0, 1.5],
    )

    assert inventory["traversed_mesh_count"] == 4
    assert inventory["collision_mesh_count"] == 3
    assert inventory["excluded_target_prim_paths"] == ["/Root/Fridge"]
    assert [row["source_prim_path"] for row in inventory["obstacles"]] == [
        "/Root/Cabinet"
    ]
    obstacles = obstacles_from_sage_sweep_inventory(inventory)
    assert obstacles[0]["source_receipt_digest"] == inventory["receipt_digest"]

    sweep = evaluate_revolute_member_sweep(
        hinge_origin_world_m=[0.0, 0.0, 0.5],
        closed_endpoint_world_m=[1.0, 0.0, 0.5],
        member_vertical_interval_m=[0.0, 1.5],
        start_angle_degrees=0.0,
        end_angle_degrees=45.0,
        obstacles=obstacles,
    )
    assert sweep["status"] == "blocked_by_observed_obstacle"

    exact = evaluate_revolute_member_sweep_against_sage_meshes(
        sage_collision_usd_path=collision,
        obstacle_inventory=inventory,
        hinge_origin_world_m=[0.0, 0.0, 0.5],
        closed_endpoint_world_m=[1.0, 0.0, 0.5],
        member_vertical_interval_m=[0.0, 1.5],
        start_angle_degrees=0.0,
        end_angle_degrees=45.0,
        member_half_thickness_m=0.05,
    )
    assert exact["status"] == "blocked_by_exact_sage_mesh_contact"
    assert exact["first_collision"]["source_prim_path"] == "/Root/Cabinet"
    assert exact["mesh_geometry"][0]["triangle_count"] == 12
    assert exact["claim_boundary"]["triangle_prism_intersection_tested"]
    assert validate_sage_mesh_sweep(exact) == exact


def test_sage_inventory_rejects_collision_source_substitution(tmp_path: Path) -> None:
    identity = {
        "schema_version": "interiorgs_sage_collision_identity.v1",
        "source_files": {"sage_collision_usd": {"sha256": "sha256:not-the-source"}},
        "whole_object_matches": [{"prim_path": "/Root/Fridge"}],
        "receipt_digest": "",
    }
    identity["receipt_digest"] = canonical_digest(identity, digest_field="receipt_digest")
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    collision = tmp_path / "collision.usda"
    collision.write_text("#usda 1.0\n", encoding="utf-8")

    with pytest.raises(
        ArticulatedWorkspaceClearanceError,
        match="sage_sweep_collision_source_digest_mismatch",
    ):
        inventory_sage_sweep_obstacles(
            sage_collision_usd_path=collision,
            target_collision_identity_receipt_path=identity_path,
            hinge_origin_world_m=[0.0, 0.0, 0.5],
            closed_endpoint_world_m=[1.0, 0.0, 0.5],
            member_vertical_interval_m=[0.0, 1.5],
        )
