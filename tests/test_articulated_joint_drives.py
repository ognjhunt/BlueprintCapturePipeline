from __future__ import annotations

import json
from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.articulated_joint_drives import (
    JOINT_DRIVE_SCHEMA_VERSION,
    ArticulatedJointDriveError,
    author_articulated_joint_drives,
)


def _asset(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for link in ("cabinet", "upper_door", "lower_door"):
        UsdPhysics.RigidBodyAPI.Apply(
            UsdGeom.Xform.Define(stage, f"/Asset/{link}").GetPrim()
        )
    for name, body in (("upper_door_hinge", "upper_door"), ("lower_door_hinge", "lower_door")):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Asset/joints/{name}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/cabinet")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/Asset/{body}")])
        joint.CreateAxisAttr().Set("Z")
        joint.CreateLowerLimitAttr().Set(0.0)
        joint.CreateUpperLimitAttr().Set(90.0)
    stage.GetRootLayer().Save()
    return path


def _author(tmp_path: Path, **overrides):
    arguments = {
        "source_usd_path": _asset(tmp_path / "asset.usda"),
        "destination": tmp_path / "driven.usda",
        "drives": [
            {
                "joint_prim_path": "/Asset/joints/upper_door_hinge",
                "role": "task_joint_free_with_friction",
                "damping": 12.0,
                "max_force": 250.0,
            },
            {
                "joint_prim_path": "/Asset/joints/lower_door_hinge",
                "role": "locked_joint_held_closed",
                "stiffness": 4000.0,
                "damping": 400.0,
                "target_position_degrees": 0.0,
                "max_force": 800.0,
            },
        ],
    }
    arguments.update(overrides)
    return author_articulated_joint_drives(**arguments)


def test_the_task_joint_gets_friction_not_a_position_servo(tmp_path: Path) -> None:
    """The task scores the door staying open after release.

    A position drive would hold it there regardless of the robot, so the
    commanded hinge gets damping only: it resists motion and stays where it is
    left, and the arm has to do the work.
    """

    receipt = _author(tmp_path)

    stage = Usd.Stage.Open(receipt["driven_usd_path"])
    joint = stage.GetPrimAtPath("/Asset/joints/upper_door_hinge")
    drive = UsdPhysics.DriveAPI.Get(joint, "angular")
    assert drive
    assert float(drive.GetStiffnessAttr().Get() or 0.0) == 0.0
    assert float(drive.GetDampingAttr().Get()) == 12.0
    row = next(r for r in receipt["drives"] if r["role"] == "task_joint_free_with_friction")
    assert row["stiffness"] == 0.0
    assert row["position_servo_enabled"] is False
    assert row["holds_position_without_actuation"] is False
    assert row["resists_velocity_without_position_target"] is True


def test_the_locked_joint_is_held_closed(tmp_path: Path) -> None:
    receipt = _author(tmp_path)

    stage = Usd.Stage.Open(receipt["driven_usd_path"])
    drive = UsdPhysics.DriveAPI.Get(
        stage.GetPrimAtPath("/Asset/joints/lower_door_hinge"), "angular"
    )
    assert float(drive.GetStiffnessAttr().Get()) == 4000.0
    assert float(drive.GetTargetPositionAttr().Get()) == 0.0
    row = next(r for r in receipt["drives"] if r["role"] == "locked_joint_held_closed")
    assert row["position_servo_enabled"] is True
    assert row["holds_position_without_actuation"] is True
    assert row["resists_velocity_without_position_target"] is False


def test_a_task_joint_given_position_stiffness_fails_closed(tmp_path: Path) -> None:
    """Stiffness on the commanded joint would score the drive, not the robot."""

    with pytest.raises(ArticulatedJointDriveError) as excinfo:
        _author(
            tmp_path,
            drives=[
                {
                    "joint_prim_path": "/Asset/joints/upper_door_hinge",
                    "role": "task_joint_free_with_friction",
                    "stiffness": 900.0,
                    "damping": 12.0,
                }
            ],
        )

    assert any(
        "task_joint_must_not_be_position_servoed" in error
        for error in excinfo.value.errors
    )


def test_a_missing_joint_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ArticulatedJointDriveError) as excinfo:
        _author(
            tmp_path,
            drives=[
                {
                    "joint_prim_path": "/Asset/joints/nope",
                    "role": "locked_joint_held_closed",
                    "stiffness": 10.0,
                }
            ],
        )

    assert any("joint_not_found" in error for error in excinfo.value.errors)


def test_a_drive_with_no_resistance_fails_closed(tmp_path: Path) -> None:
    """A drive with zero stiffness and zero damping does nothing at all."""

    with pytest.raises(ArticulatedJointDriveError) as excinfo:
        _author(
            tmp_path,
            drives=[
                {
                    "joint_prim_path": "/Asset/joints/upper_door_hinge",
                    "role": "task_joint_free_with_friction",
                    "damping": 0.0,
                }
            ],
        )

    assert any("drive_has_no_effect" in error for error in excinfo.value.errors)


def test_articulation_and_source_are_untouched(tmp_path: Path) -> None:
    source = _asset(tmp_path / "asset.usda")
    before = source.read_bytes()

    receipt = _author(tmp_path, source_usd_path=source)

    assert source.read_bytes() == before
    stage = Usd.Stage.Open(receipt["driven_usd_path"])
    assert len([p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]) == 2
    assert receipt["preserved"]["assembly_joint_count"] == 2
    assert receipt["schema_version"] == JOINT_DRIVE_SCHEMA_VERSION


def test_authoring_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    first = _author(tmp_path, destination=tmp_path / "a.usda")
    second = _author(tmp_path, destination=tmp_path / "b.usda")

    assert first["driven_usd_sha256"] == second["driven_usd_sha256"]
    stored = json.loads(Path(first["receipt_path"]).read_text(encoding="utf-8"))
    assert stored == first
