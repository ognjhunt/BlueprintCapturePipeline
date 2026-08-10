from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.articulated_native_probe import (
    ARTICULATED_NATIVE_PROBE_SCHEMA_VERSION,
    ArticulatedNativeProbeError,
    materialize_articulated_native_probe,
)


UPPER = (0.939981249, 1.631869998)
PIVOT = (-0.356966056, 0.350000041)


def _candidate(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(asset.GetPrim())
    for link, z0, z1 in (
        ("cabinet", 0.0, 1.632),
        ("upper_door", UPPER[0], UPPER[1]),
        ("lower_door", 0.0, UPPER[0]),
    ):
        xform = UsdGeom.Xform.Define(stage, f"/Asset/{link}")
        UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr().Set(10.0)
        mesh = UsdGeom.Mesh.Define(stage, f"/Asset/{link}/geom")
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(x, y, z)
                for x in (-0.35, 0.35)
                for y in (-0.3, 0.3)
                for z in (z0, z1)
            ]
        )
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr(
            [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
        )
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    for name, body1, z in (
        ("upper_door_hinge", "/Asset/upper_door", 1.28),
        ("lower_door_hinge", "/Asset/lower_door", 0.47),
    ):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Asset/joints/{name}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/cabinet")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(PIVOT[0], PIVOT[1], z))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(PIVOT[0], PIVOT[1], z))
        joint.CreateAxisAttr().Set("Z")
        joint.CreateLowerLimitAttr().Set(0.0)
        joint.CreateUpperLimitAttr().Set(90.0)
    stage.GetRootLayer().Save()
    return path


def _probe(tmp_path: Path, **overrides):
    arguments = {
        "candidate_usd_path": _candidate(tmp_path / "candidate.usda"),
        "destination": tmp_path / "probe",
        "task_joint_prim_path": "/Asset/joints/upper_door_hinge",
        "locked_joint_prim_paths": ["/Asset/joints/lower_door_hinge"],
        "commanded_sweep_degrees": [0.0, 15.0, 30.0, 45.0, 55.0],
        "reset_joint_positions_rad": {
            "/Asset/joints/upper_door_hinge": 0.0,
            "/Asset/joints/lower_door_hinge": 0.0,
        },
        "locked_joint_motion_tolerance_rad": 0.001,
        "settle_samples": 40,
        "control_frequency_hz": 15.0,
    }
    arguments.update(overrides)
    return materialize_articulated_native_probe(**arguments)


def test_probe_freezes_every_required_native_readback(tmp_path: Path) -> None:
    receipt = _probe(tmp_path)

    assert receipt["schema_version"] == ARTICULATED_NATIVE_PROBE_SCHEMA_VERSION
    assert receipt["status"] == "frozen_not_executed"
    required = set(receipt["required_readbacks"])
    assert {
        "articulation_root_identity",
        "joint_count_and_types",
        "task_joint_identity",
        "locked_joint_identity",
        "joint_axis_and_limits",
        "locked_joint_motion_within_tolerance",
        "commanded_sweep_reaches_maximum",
        "contact_stability",
        "no_initial_penetration",
        "reset_replay_within_tolerance",
        "deterministic_final_state",
    } <= required
    assert receipt["expected"]["assembly_joint_count"] == 2
    assert receipt["expected"]["articulation_root_prim_path"] == "/Asset"
    assert receipt["expected"]["task_joint_axis"] == "Z"
    assert receipt["expected"]["task_joint_limits_deg"] == [0.0, 90.0]
    assert receipt["expected"]["maximum_commanded_degrees"] == 55.0
    assert receipt["claim_boundary"]["frozen_before_execution"] is True
    assert receipt["claim_boundary"]["native_simulator_qualified"] is False
    assert receipt["receipt_digest"].startswith("sha256:")


def test_probe_writes_a_loadable_blank_stage_diagnostic(tmp_path: Path) -> None:
    receipt = _probe(tmp_path)

    blank = Path(receipt["stages"]["blank_stage"]["path"])
    assert blank.is_file()
    stage = Usd.Stage.Open(str(blank))
    assert stage is not None
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    assert len(scenes) == 1
    assert not [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]


def test_probe_articulation_stage_references_the_exact_candidate(
    tmp_path: Path,
) -> None:
    receipt = _probe(tmp_path)

    articulation = Path(receipt["stages"]["articulation_stage"]["path"])
    stage = Usd.Stage.Open(str(articulation))
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    assert len(joints) == 2
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert len(roots) == 1
    assert receipt["candidate_usd_sha256"].startswith("sha256:")
    for row in receipt["stages"].values():
        assert row["sha256"].startswith("sha256:")


def test_probe_rejects_a_task_joint_that_is_not_in_the_candidate(
    tmp_path: Path,
) -> None:
    with pytest.raises(ArticulatedNativeProbeError) as excinfo:
        _probe(tmp_path, task_joint_prim_path="/Asset/joints/missing")

    assert any("task_joint_not_found" in error for error in excinfo.value.errors)


def test_probe_rejects_a_sweep_beyond_the_authored_limit(tmp_path: Path) -> None:
    with pytest.raises(ArticulatedNativeProbeError) as excinfo:
        _probe(tmp_path, commanded_sweep_degrees=[0.0, 45.0, 120.0])

    assert any(
        "commanded_sweep_outside_joint_limits" in error for error in excinfo.value.errors
    )


def test_probe_rejects_a_reset_outside_the_joint_limits(tmp_path: Path) -> None:
    with pytest.raises(ArticulatedNativeProbeError) as excinfo:
        _probe(
            tmp_path,
            reset_joint_positions_rad={
                "/Asset/joints/upper_door_hinge": -0.5,
                "/Asset/joints/lower_door_hinge": 0.0,
            },
        )

    assert any(
        "reset_position_outside_joint_limits" in error for error in excinfo.value.errors
    )


def test_probe_spec_is_deterministic(tmp_path: Path) -> None:
    first = _probe(tmp_path / "a")
    second = _probe(tmp_path / "b")

    assert first["expected"] == second["expected"]
    assert first["stages"]["blank_stage"]["sha256"] == (
        second["stages"]["blank_stage"]["sha256"]
    )
    assert math.isclose(
        first["settle"]["window_seconds"], second["settle"]["window_seconds"]
    )


def test_probe_spec_file_round_trips(tmp_path: Path) -> None:
    receipt = _probe(tmp_path)

    spec_path = Path(receipt["spec_path"])
    assert spec_path.is_file()
    assert json.loads(spec_path.read_text(encoding="utf-8")) == receipt


def test_probe_time_actuation_lives_in_the_overlay_not_the_asset(
    tmp_path: Path,
) -> None:
    """The asset must not carry a servo the task would then be scoring."""

    receipt = _probe(
        tmp_path,
        probe_drive_stiffness=900.0,
        probe_drive_damping=90.0,
        probe_drive_max_force=400.0,
    )

    overlay = Path(receipt["stages"]["articulation_stage"]["path"]).read_text()
    assert "PhysicsDriveAPI:angular" in overlay
    assert "drive:angular:physics:stiffness = 900.0" in overlay
    assert receipt["probe_drive"]["joint_prim_path"] == (
        "/Asset/joints/upper_door_hinge"
    )
    # the candidate copy the overlay references is untouched by the actuation
    candidate = Path(receipt["stages"]["candidate_copy"]["path"]).read_text()
    assert "PhysicsDriveAPI" not in candidate


def test_a_probe_without_actuation_writes_no_drive(tmp_path: Path) -> None:
    receipt = _probe(tmp_path)

    overlay = Path(receipt["stages"]["articulation_stage"]["path"]).read_text()
    assert "PhysicsDriveAPI" not in overlay
    assert receipt["probe_drive"] is None
