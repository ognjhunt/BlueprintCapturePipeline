from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_droid_grasp_swept_volume import (
    GRIPPER_PREFIX,
    PAD_PATHS,
    materialize_native_droid_grasp_swept_volume,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _cube(stage, path, *, center, size):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _robot(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/panda")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, f"{GRIPPER_PREFIX}base_link")
    _cube(
        stage,
        f"{GRIPPER_PREFIX}base_link/base_collision",
        center=(0.0625, 0.0, 0.0),
        size=(0.125, 0.08, 0.08),
    )
    for side, y in (("left", 0.052), ("right", -0.052)):
        _cube(
            stage,
            PAD_PATHS[side],
            center=(0.13, y, 0.0),
            size=(0.04, 0.02, 0.04),
        )
        _cube(
            stage,
            f"{GRIPPER_PREFIX}{side}_inner_knuckle/collision",
            center=(0.0825, y * 0.5, 0.0),
            size=(0.089, 0.03, 0.04),
        )
    stage.GetRootLayer().Save()
    return path


def _task(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    door = UsdGeom.Xform.Define(stage, "/Asset/links/door")
    UsdPhysics.RigidBodyAPI.Apply(door.GetPrim())
    _cube(
        stage,
        "/Asset/links/door/geometry/disc",
        center=(0.0, 0.02, 0.0),
        size=(0.3, 0.06, 0.3),
    )
    patch = _cube(
        stage,
        "/Asset/links/door/grasp_collision_patches/right_outer_rim",
        center=(0.0, 0.024, 0.0),
        size=(0.01, 0.05, 0.05),
    )
    patch.GetPrim().SetCustomDataByKey(
        "blueprint:graspAffordanceRole", "parallel_jaw_outer_rim_patch"
    )
    stage.GetRootLayer().Save()
    return path


def _receipts(tmp_path: Path, task: Path) -> tuple[Path, Path]:
    registered = {
        "schema_version": "registered_replacement_asset.v1",
        "scene_id": "scene",
        "task_id": "task",
        "asset_id": "asset",
        "task_freeze_digest": "sha256:" + "a" * 64,
        "output_usd": {
            "path": str(task),
            "size_bytes": task.stat().st_size,
            "sha256": _sha(task),
        },
        "receipt_digest": "",
    }
    registered["receipt_digest"] = canonical_digest(
        registered, digest_field="receipt_digest"
    )
    registered_path = tmp_path / "registered.json"
    registered_path.write_text(json.dumps(registered), encoding="utf-8")
    affordance = {
        "schema_version": "paired_target_interaction_affordance_candidate.v1",
        "status": "candidate_geometry_materialized_requires_native_contact",
        "scene_id": "scene",
        "task_id": "task",
        "asset_id": "asset",
        "registered_asset": {"receipt_digest": registered["receipt_digest"]},
        "candidate": {
            "grasp_collision_patch_prim_path": (
                "/Asset/links/door/grasp_collision_patches/right_outer_rim"
            ),
            "contact_point_to_grasp_collider_surface_m": 0.0,
            "contact_point_registered_stage_m": [0.005, -0.001, 0.0],
            "gripper_approach_axis_registered_stage": [0.0, 1.0, 0.0],
            "pinch_axis_registered_stage": [0.0, 0.0, 1.0],
            "grasp_lateral_outward_unit_registered_stage": [1.0, 0.0, 0.0],
        },
        "receipt_digest": "",
    }
    affordance["receipt_digest"] = canonical_digest(
        affordance, digest_field="receipt_digest"
    )
    affordance_path = tmp_path / "affordance.json"
    affordance_path.write_text(json.dumps(affordance), encoding="utf-8")
    return registered_path, affordance_path


def test_exact_robot_bytes_derive_a_margin_beyond_first_clear_pose(
    tmp_path: Path,
) -> None:
    robot = _robot(tmp_path / "robot.usda")
    task = _task(tmp_path / "task.usda")
    registered, affordance = _receipts(tmp_path, task)

    receipt = materialize_native_droid_grasp_swept_volume(
        robot_usd_path=robot,
        expected_robot_sha256=_sha(robot),
        robot_asset_uri="https://example.invalid/pinned-robot.usd",
        registered_asset_receipt_path=registered,
        interaction_affordance_path=affordance,
        output_path=tmp_path / "sweep.json",
        search_step_m=0.001,
        clearance_margin_m=0.004,
        maximum_standoff_m=0.03,
    )

    assert receipt["status"] == "conservative_open_gripper_standoff_qualified"
    minimum = receipt["minimum_collision_free_outward_standoff_m"]
    assert minimum > 0.0
    assert receipt["selected_outward_standoff_m"] == pytest.approx(
        minimum + 0.004
    )
    assert receipt["selected_sample"]["forbidden_collision_count"] == 0
    assert receipt["last_blocked_sample"]["forbidden_collision_count"] > 0
    assert all(
        value > 0.0 for value in receipt["pad_patch_approach_overlap_m"].values()
    )
    assert receipt["lateral_outward_grasp_frame_unit"] == [-1.0, 0.0, 0.0]
    assert receipt["lateral_outward_unit_world"] == pytest.approx(
        [1.0, 0.0, 0.0]
    )
    assert receipt["selected_lateral_tcp_surface_offset_m"] == pytest.approx(
        0.02
    )
    assert receipt["pad_lateral_surface_support_m"] == pytest.approx(
        {"left": 0.02, "right": 0.02}
    )


def test_robot_digest_mismatch_refuses(tmp_path: Path) -> None:
    robot = _robot(tmp_path / "robot.usda")
    task = _task(tmp_path / "task.usda")
    registered, affordance = _receipts(tmp_path, task)
    with pytest.raises(ValueError, match="droid_grasp_robot_identity_invalid"):
        materialize_native_droid_grasp_swept_volume(
            robot_usd_path=robot,
            expected_robot_sha256="sha256:" + "0" * 64,
            robot_asset_uri="https://example.invalid/pinned-robot.usd",
            registered_asset_receipt_path=registered,
            interaction_affordance_path=affordance,
            output_path=tmp_path / "sweep.json",
        )
