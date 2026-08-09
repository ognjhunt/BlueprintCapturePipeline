from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.articulated_workspace_clearance import (
    ArticulatedWorkspaceClearanceError,
    evaluate_frozen_door_state_clearance,
    inventory_sage_sweep_obstacles,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


HINGE = (0.0, 0.0, 1.25)
CLOSED_ENDPOINT = (0.7, 0.0, 1.25)
VERTICAL = (0.9, 1.6)
HALF_THICKNESS = 0.05
FROZEN_STATES = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]


def _collision_box(
    stage: Usd.Stage,
    path: str,
    *,
    center: tuple[float, float, float],
    half: tuple[float, float, float],
) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    cx, cy, cz = center
    hx, hy, hz = half
    points = [
        Gf.Vec3f(cx + sx * hx, cy + sy * hy, cz + sz * hz)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
    mesh.CreateFaceVertexIndicesAttr(
        [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
    )
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_scene(tmp_path: Path, *, obstacle_angle_deg: float | None) -> tuple[Path, Path]:
    scene = tmp_path / "sage_collision.usda"
    stage = Usd.Stage.CreateNew(str(scene))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    _collision_box(
        stage, "/Root/target_refrigerator", center=(0.35, -0.35, 0.8), half=(0.35, 0.3, 0.8)
    )
    if obstacle_angle_deg is not None:
        theta = math.radians(obstacle_angle_deg)
        _collision_box(
            stage,
            "/Root/chair_obstacle",
            center=(0.68 * math.cos(theta), 0.68 * math.sin(theta), 1.2),
            half=(0.02, 0.02, 0.2),
        )
    _collision_box(stage, "/Root/far_wall", center=(3.0, 3.0, 1.0), half=(0.1, 0.1, 1.0))
    stage.GetRootLayer().Save()

    identity = {
        "schema_version": "interiorgs_sage_collision_identity.v1",
        "whole_object_matches": [{"prim_path": "/Root/target_refrigerator"}],
        "source_files": {
            "sage_collision_usd": {"sha256": _sha256(scene)},
        },
        "receipt_digest": "",
    }
    identity["receipt_digest"] = canonical_digest(identity, digest_field="receipt_digest")
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    return scene, identity_path


def _inventory(scene: Path, identity_path: Path) -> dict:
    return inventory_sage_sweep_obstacles(
        sage_collision_usd_path=scene,
        target_collision_identity_receipt_path=identity_path,
        hinge_origin_world_m=list(HINGE),
        closed_endpoint_world_m=list(CLOSED_ENDPOINT),
        member_vertical_interval_m=list(VERTICAL),
    )


def _evaluate(scene: Path, inventory: dict, **overrides):
    arguments = {
        "sage_collision_usd_path": scene,
        "obstacle_inventory": inventory,
        "hinge_origin_world_m": list(HINGE),
        "closed_endpoint_world_m": list(CLOSED_ENDPOINT),
        "member_vertical_interval_m": list(VERTICAL),
        "member_half_thickness_m": HALF_THICKNESS,
        "door_state_angles_degrees": list(FROZEN_STATES),
        "required_maximum_angle_degrees": 55.0,
        "static_box_obstacles": [],
    }
    arguments.update(overrides)
    return evaluate_frozen_door_state_clearance(**arguments)


def test_matrix_clear_scene_reports_twelve_clear_states(tmp_path: Path) -> None:
    scene, identity = _fixture_scene(tmp_path, obstacle_angle_deg=None)

    receipt = _evaluate(scene, _inventory(scene, identity))

    assert receipt["schema_version"] == "articulated_door_state_clearance.v1"
    assert receipt["status"] == "door_state_matrix_clearance_candidate_only"
    rows = receipt["door_state_rows"]
    assert [row["angle_degrees"] for row in rows] == FROZEN_STATES
    assert all(row["clear"] is True for row in rows)
    assert receipt["claim_boundary"]["replacement_self_geometry_bound"] is False
    assert receipt["claim_boundary"]["franka_base_bound"] is False
    assert receipt["claim_boundary"]["clear_result_is_not_native_dynamic_qualification"] is True
    assert receipt["receipt_digest"].startswith("sha256:")


def test_matrix_blocks_on_sage_contact_in_late_states(tmp_path: Path) -> None:
    scene, identity = _fixture_scene(tmp_path, obstacle_angle_deg=53.0)

    receipt = _evaluate(scene, _inventory(scene, identity))

    assert receipt["status"] == "blocked_by_door_state_contact"
    by_angle = {row["angle_degrees"]: row for row in receipt["door_state_rows"]}
    assert by_angle[45.0]["clear"] is True
    assert by_angle[50.0]["clear"] is False
    assert by_angle[55.0]["clear"] is False
    assert "/Root/chair_obstacle" in by_angle[50.0]["sage_contact_prim_paths"]
    assert receipt["first_contact"]["angle_degrees"] == 50.0


def test_matrix_blocks_on_static_replacement_box_contact(tmp_path: Path) -> None:
    scene, identity = _fixture_scene(tmp_path, obstacle_angle_deg=None)

    receipt = _evaluate(
        scene,
        _inventory(scene, identity),
        static_box_obstacles=[
            {
                "label": "authored_lower_door_slab",
                "obstacle_class": "replacement_lower_door",
                "aabb_min": [0.0, -0.05, 0.8],
                "aabb_max": [0.7, 0.05, 1.0],
            }
        ],
    )

    assert receipt["status"] == "blocked_by_door_state_contact"
    first = receipt["first_contact"]
    assert first["angle_degrees"] == 0.0
    assert first["obstacle_class"] == "replacement_lower_door"
    assert receipt["claim_boundary"]["replacement_self_geometry_bound"] is True


def test_matrix_rejects_underspecified_state_list(tmp_path: Path) -> None:
    scene, identity = _fixture_scene(tmp_path, obstacle_angle_deg=None)
    inventory = _inventory(scene, identity)

    with pytest.raises(ArticulatedWorkspaceClearanceError) as excinfo:
        _evaluate(scene, inventory, door_state_angles_degrees=[0.0, 30.0, 55.0])
    assert any(
        "door_state_angles_below_minimum_count" in error for error in excinfo.value.errors
    )

    with pytest.raises(ArticulatedWorkspaceClearanceError) as excinfo:
        _evaluate(
            scene,
            inventory,
            door_state_angles_degrees=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 50.0],
        )
    assert any(
        "door_state_angles_not_strictly_increasing" in error
        for error in excinfo.value.errors
    )

    with pytest.raises(ArticulatedWorkspaceClearanceError) as excinfo:
        _evaluate(
            scene,
            inventory,
            door_state_angles_degrees=[1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
        )
    assert any(
        "door_state_angles_must_start_closed" in error for error in excinfo.value.errors
    )

    with pytest.raises(ArticulatedWorkspaceClearanceError) as excinfo:
        _evaluate(
            scene,
            inventory,
            door_state_angles_degrees=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 33.0, 36.0, 40.0, 45.0, 50.0],
        )
    assert any(
        "door_state_angles_do_not_reach_required_maximum" in error
        for error in excinfo.value.errors
    )


def test_matrix_rejects_tampered_collision_source(tmp_path: Path) -> None:
    scene, identity = _fixture_scene(tmp_path, obstacle_angle_deg=None)
    inventory = _inventory(scene, identity)
    scene.write_bytes(scene.read_bytes() + b"\n# tampered\n")

    with pytest.raises(ArticulatedWorkspaceClearanceError) as excinfo:
        _evaluate(scene, inventory)

    assert any(
        "collision_source_digest_mismatch" in error for error in excinfo.value.errors
    )
