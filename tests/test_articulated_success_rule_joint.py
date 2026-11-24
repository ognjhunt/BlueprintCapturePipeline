"""An articulated job's success rule says which joint it is for, of what kind, and the unit of every number.

Brought from the $0 stress test of 2026-11-24 (``test_an_articulated_rule_says_which_joint_and_in_what_unit``): the
shipped ``opening.success_interval`` [0.216, 0.36] is metres for a drawer and radians for a door, and the cabinet has
three drawers; the allow-list dropped ``target_joint`` and ``locked_joints``, no field gave the joint's kind,
``units`` said "metres, radians, seconds" beside hertz fields, and ``window_samples`` had no rate. The joint is read
from the shipped object file, never from the plan, and named by its prim path in scene.usda. Offline, $0.
"""
# Covers: src/blueprint_pipeline/scene_package_stage.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pxr")

from blueprint_pipeline import scene_package_stage as stage  # noqa: E402

#: The articulated contract the matrix's drawer cabinet seals (tests/test_team_scene_packet._articulated_scene),
#: copied from the half record the v1 row "articulated-not-run-joint-as-authored" compiled.
DRAWER_CRITERIA = {
    "assembly_root": {"mode": "required"},
    "hold": {"maximum_settled_target_speed": 0.02, "mode": "required", "window_samples": 15},
    "locked_joints": {"joint_ids": ["drawer_0_fixed", "drawer_2_fixed"], "mode": "required", "motion_tolerance": 0.01},
    "motion": {"movement_epsilon": 0.0036},
    "opening": {"joint_hard_limits": [0.0, 0.36], "mode": "required", "reset_position": 0.0,
                "success_interval": [0.216, 0.36]},
    "release_and_retreat": {"mode": "ignored"},
    "reset": {"tolerance": 0.005},
    "safety": {"mode": "required"},
    "target_joint": {"joint_id": "task_part_joint", "joint_ids": ["task_part_joint"]},
}
POSE = {"position_world_m": [0.4, 0.1, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}


def _room(path: Path) -> None:
    from pxr import Usd, UsdGeom
    room = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(room, UsdGeom.Tokens.z)
    room.SetDefaultPrim(UsdGeom.Xform.Define(room, "/Root").GetPrim())
    room.GetRootLayer().Save()


def _object(path: Path, *, target: str) -> None:
    """A cabinet whose task joint is ``target`` (``prismatic`` or ``revolute``), with two fixed drawers."""
    from pxr import Usd, UsdGeom, UsdPhysics
    asset = Usd.Stage.CreateNew(str(path))
    asset.SetDefaultPrim(UsdGeom.Xform.Define(asset, "/Asset").GetPrim())
    UsdGeom.Xform.Define(asset, "/Asset/joints")
    (UsdPhysics.PrismaticJoint if target == "prismatic" else UsdPhysics.RevoluteJoint).Define(
        asset, "/Asset/joints/task_part_joint")
    for name in ("drawer_0_fixed", "drawer_2_fixed"):
        UsdPhysics.FixedJoint.Define(asset, f"/Asset/joints/{name}")
    asset.GetRootLayer().Save()


def _plan(packet: Path, *, target: str = "prismatic", criteria=None) -> dict:
    packet.mkdir(parents=True, exist_ok=True)
    for name in ("collision.usda", "appearance.usda"):
        _room(packet / name)
    _object(packet / "replacement.usda", target=target)
    rows = [{"semantic_role": role, "usd_path": name, "pose_world": pose}
            for role, name, pose in (("scene_collision", "collision.usda", {**POSE, "position_world_m": [0, 0, 0]}),
                                     ("scene_appearance", "appearance.usda", {**POSE, "position_world_m": [0, 0, 0]}),
                                     ("task_object", "replacement.usda", POSE))]
    return {"objects": rows, "cadence": {"control_frequency_hz": 15.0, "physics_frequency_hz": 120.0,
                                         "episode_length_seconds": 31.0},
            "task_spec": {"task_success_contract": {"criteria": criteria or DRAWER_CRITERIA}}}


def test_an_articulated_rule_says_which_joint_and_in_what_unit(tmp_path):
    rule = stage.neutral_rule(_plan(tmp_path), packet=tmp_path)
    kept = rule["criteria"]
    assert kept["target_joint"] == {"prim_path": "/World/Object/Asset/joints/task_part_joint", "kind": "prismatic",
                                    "unit": "metres"}
    assert kept["locked_joints"] == {"mode": "required", "motion_tolerance": 0.01, "joints": [
        {"prim_path": "/World/Object/Asset/joints/drawer_0_fixed", "kind": "fixed"},
        {"prim_path": "/World/Object/Asset/joints/drawer_2_fixed", "kind": "fixed"}]}
    # Blueprint's joint ids are not shipped: only prim paths of the shipped file.
    assert "joint_id" not in json.dumps(rule)
    units = rule["units"]
    assert units["criteria.opening.success_interval"] == "metres (the target joint is prismatic)"
    assert units["criteria.hold.maximum_settled_target_speed"] == "metres per second (the target joint is prismatic)"
    assert units["criteria.locked_joints.motion_tolerance"] == stage.LOCKED_JOINT_UNIT
    assert units["*window_samples"] == "control steps, 15 a second (timing.control_frequency_hz)"
    # Every unit the file uses, and only those: the hertz and seconds of its timing, the metres of its start.
    assert units["*_hz"] == "hertz" and units["*_seconds"] == "seconds" and units["*_m"] == "metres"
    assert "*_n" not in units and "*_rad" not in units
    # The stress test's own reading of the shipped criteria.
    text = json.dumps(kept)
    assert "task_part_joint" in text and "prismatic" in text


def test_a_door_says_radians_and_the_composed_stage_has_the_joint(tmp_path):
    from pxr import Usd, UsdPhysics
    rule = stage.neutral_rule(_plan(tmp_path, target="revolute"), packet=tmp_path)
    assert rule["criteria"]["target_joint"]["kind"] == "revolute" and rule["criteria"]["target_joint"]["unit"] == "radians"
    assert rule["units"]["criteria.opening.success_interval"] == "radians (the target joint is revolute)"
    assert rule["units"]["criteria.hold.maximum_settled_target_speed"] == "radians per second (the target joint is revolute)"
    # The prim path is where scene.usda puts the joint.
    rows = [{**row, "file": row["usd_path"]} for row in stage.placements(_plan(tmp_path / "again", target="revolute"))]
    stage.write_stage(tmp_path / "again" / "scene.usda", rows, axis="Z")
    composed = Usd.Stage.Open(str(tmp_path / "again" / "scene.usda"))
    assert composed.GetPrimAtPath(rule["criteria"]["target_joint"]["prim_path"]).IsA(UsdPhysics.RevoluteJoint)


def test_a_joint_the_file_does_not_name_is_left_out_and_counted(tmp_path):
    criteria = {**DRAWER_CRITERIA, "target_joint": {"joint_id": "joint_nobody_wrote", "joint_ids": ["x"]},
                "locked_joints": {**DRAWER_CRITERIA["locked_joints"], "joint_ids": ["drawer_0_fixed", "gone"]}}
    rule = stage.neutral_rule(_plan(tmp_path, criteria=criteria), packet=tmp_path)
    assert "target_joint" not in rule["criteria"]
    assert rule["criteria"]["locked_joints"]["joints"] == [
        {"prim_path": "/World/Object/Asset/joints/drawer_0_fixed", "kind": "fixed"}]
    # With no target joint read, the interval's unit is said as what it is.
    assert rule["units"]["criteria.opening.success_interval"] == stage.UNKNOWN_JOINT_UNIT
    # Without the object's file, both blocks go whole, as before.
    kept, left_out = stage.neutral_criteria(DRAWER_CRITERIA)
    assert not {"target_joint", "locked_joints"} & set(kept) and left_out >= 2


def test_a_rigid_rule_lists_its_own_units_and_what_its_window_counts():
    rule = {"criteria": {"settling": {"window_samples": 20, "position_tolerance_m": 0.005,
                                      "orientation_tolerance_rad": 0.03},
                         "terminal_task_contact": {"maximum_task_contact_force_n": 40.0}},
            "timing": {"control_frequency_hz": 20.0, "episode_length_seconds": 13.0},
            "object_estimates_from_video": {"mass_kg": [0.1, 0.2], "static_friction": [0.4, 0.6]},
            "criteria_basis": {"goal_radius_m": "assumed"}}
    assert stage.rule_units(rule) == {
        "*_hz": "hertz", "*_kg": "kilograms", "*_m": "metres", "*_n": "newtons", "*_rad": "radians",
        "*_seconds": "seconds", "*window_samples": "control steps, 20 a second (timing.control_frequency_hz)"}
