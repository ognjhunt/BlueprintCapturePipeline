from __future__ import annotations

import json
import math
import importlib.util
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


def _production_shaped_washer_candidate(
    path: Path,
    *,
    additional_kinematic_link: str | None = None,
) -> Path:
    """A small fixture with the topology of Scene 840920's washer twin.

    The production failure only occurs when the fixed cabinet is a kinematic
    link inside a five-joint articulation.  A two-door toy can pass all the
    former tests while missing that condition, so this fixture carries the
    actual link and joint roles used by the scene-bound task packet.
    """

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(asset.GetPrim())
    links = {
        "body": (0.0, 0.0, 0.0),
        "door": (0.0, -0.302, 0.43),
        "latch": (0.23, -0.302, 0.43),
        "drum": (0.0, -0.02, 0.43),
        "selector": (-0.08, -0.295, 0.75),
        "drawer": (0.15, -0.295, 0.75),
    }
    for link, position in links.items():
        xform = UsdGeom.Xform.Define(stage, f"/Asset/links/{link}")
        UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr().Set(1.0)
        if link == "body" or link == additional_kinematic_link:
            UsdPhysics.RigidBodyAPI(xform.GetPrim()).CreateKinematicEnabledAttr().Set(True)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        collider = UsdGeom.Cube.Define(stage, f"/Asset/links/{link}/collision")
        collider.CreateSizeAttr().Set(0.1)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())

    def revolute(name: str, body0: str, body1: str, lower: float, upper: float) -> None:
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Asset/joints/{name}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/Asset/links/{body0}")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/Asset/links/{body1}")])
        joint.CreateAxisAttr().Set("X")
        joint.CreateLowerLimitAttr().Set(lower)
        joint.CreateUpperLimitAttr().Set(upper)

    revolute("door_hinge", "body", "door", 0.0, 68.75493621826172)
    revolute("latch_coupler", "door", "latch", -5.729578, 5.729578)
    revolute("drum_bearing", "body", "drum", -180.0, 180.0)
    revolute("selector_axis", "body", "selector", -183.3465, 183.3465)
    drawer = UsdPhysics.PrismaticJoint.Define(stage, "/Asset/joints/drawer_slide")
    drawer.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/links/body")])
    drawer.CreateBody1Rel().SetTargets([Sdf.Path("/Asset/links/drawer")])
    drawer.CreateAxisAttr().Set("X")
    drawer.CreateLowerLimitAttr().Set(0.0)
    drawer.CreateUpperLimitAttr().Set(0.2)
    stage.GetRootLayer().Save()
    return path


def _production_shaped_probe(tmp_path: Path, **overrides):
    arguments = {
        "candidate_usd_path": _production_shaped_washer_candidate(
            tmp_path / "production_washer.usda"
        ),
        "destination": tmp_path / "probe",
        "task_joint_prim_path": "/Asset/joints/door_hinge",
        "locked_joint_prim_paths": [
            "/Asset/joints/drawer_slide",
            "/Asset/joints/drum_bearing",
            "/Asset/joints/latch_coupler",
            "/Asset/joints/selector_axis",
        ],
        "commanded_sweep_degrees": [0.0, 15.0, 30.0, 45.0, 55.0],
        "reset_joint_positions_rad": {
            "/Asset/joints/door_hinge": 0.0,
            "/Asset/joints/drawer_slide": 0.0,
            "/Asset/joints/drum_bearing": 0.0,
            "/Asset/joints/latch_coupler": 0.0,
            "/Asset/joints/selector_axis": 0.0,
        },
        "locked_joint_motion_tolerance_rad": 0.001,
        "settle_samples": 40,
        "control_frequency_hz": 15.0,
    }
    arguments.update(overrides)
    return materialize_articulated_native_probe(**arguments)


def _articulated_worker_module():
    worker = Path(__file__).resolve().parents[1] / (
        "scripts/run_adp009d_articulated_isaac_worker.py"
    )
    spec = importlib.util.spec_from_file_location("articulated_isaac_worker_test", worker)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_production_shaped_washer_uses_a_digest_bound_homogeneous_runtime_overlay(
    tmp_path: Path,
) -> None:
    """The live Scene 840920 base is kinematic, which PhysX cannot articulate."""

    receipt = _production_shaped_probe(tmp_path)

    representation = receipt["runtime_representation"]
    assert representation == {
        "schema_version": "adp009d_physx_articulation_representation.v1",
        "mode": "dynamic_articulation_with_world_fixed_base_anchor",
        "candidate_bytes_modified": False,
        "candidate_structural_topology_preserved": True,
        "physx_homogeneous_articulation_required": True,
        "articulation_body_prim_paths": [
            "/Asset/links/body",
            "/Asset/links/door",
            "/Asset/links/drawer",
            "/Asset/links/drum",
            "/Asset/links/latch",
            "/Asset/links/selector",
        ],
        "authored_kinematic_articulation_body_prim_paths": ["/Asset/links/body"],
        "runtime_dynamic_override_body_prim_paths": ["/Asset/links/body"],
        "fixed_base_body_prim_path": "/Asset/links/body",
        "fixed_base_anchor_prim_path": "/BlueprintProbeRuntime/fixed_base_anchor",
    }
    # The immutable candidate remains honest about how it was authored.
    candidate = Usd.Stage.Open(str(receipt["stages"]["candidate_copy"]["path"]))
    authored_body = UsdPhysics.RigidBodyAPI(candidate.GetPrimAtPath("/Asset/links/body"))
    assert authored_body.GetKinematicEnabledAttr().Get() is True

    stage = Usd.Stage.Open(str(receipt["stages"]["articulation_stage"]["path"]))
    runtime_body = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath("/Asset/links/body"))
    assert runtime_body.GetKinematicEnabledAttr().Get() is False
    anchor = stage.GetPrimAtPath("/BlueprintProbeRuntime/fixed_base_anchor")
    assert anchor.IsA(UsdPhysics.FixedJoint)
    assert UsdPhysics.FixedJoint(anchor).GetBody0Rel().GetTargets() == []
    assert UsdPhysics.FixedJoint(anchor).GetBody1Rel().GetTargets() == [
        Sdf.Path("/Asset/links/body")
    ]
    # The probe's temporary world anchor sits outside the authored asset and
    # cannot change its five-joint structural topology.
    asset_joints = [
        prim for prim in stage.Traverse() if str(prim.GetPath()).startswith("/Asset/")
        and prim.IsA(UsdPhysics.Joint)
    ]
    assert {str(prim.GetPath()) for prim in asset_joints} == {
        "/Asset/joints/door_hinge",
        "/Asset/joints/latch_coupler",
        "/Asset/joints/drum_bearing",
        "/Asset/joints/selector_axis",
        "/Asset/joints/drawer_slide",
    }


def test_runtime_readback_rejects_a_reintroduced_kinematic_body(
    tmp_path: Path,
) -> None:
    """Mutation proof for the exact PhysX failure we observed on Vast."""

    receipt = _production_shaped_probe(tmp_path)
    stage = Usd.Stage.Open(str(receipt["stages"]["articulation_stage"]["path"]))
    worker = _articulated_worker_module()

    passed = worker._physx_homogeneous_articulation_readback(
        stage, receipt["runtime_representation"], UsdPhysics
    )
    assert passed["passed"] is True

    # This is the live failure shape: a kinematic link inside the articulation
    # makes PhysX decline to create its tensor articulation view.  The helper
    # must detect it before the opaque backend AttributeError.
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath("/Asset/links/body")).CreateKinematicEnabledAttr().Set(True)
    failed = worker._physx_homogeneous_articulation_readback(
        stage, receipt["runtime_representation"], UsdPhysics
    )
    assert failed["passed"] is False
    assert failed["observed_runtime_kinematic_enabled"]["/Asset/links/body"] is True


def test_probe_rejects_nonhomogeneous_production_shaped_kinematic_articulation(
    tmp_path: Path,
) -> None:
    candidate = _production_shaped_washer_candidate(
        tmp_path / "two_kinematic_links.usda", additional_kinematic_link="selector"
    )
    with pytest.raises(ArticulatedNativeProbeError) as excinfo:
        _production_shaped_probe(tmp_path, candidate_usd_path=candidate)

    assert (
        "articulated_native_probe_nonhomogeneous_kinematic_articulation"
        in excinfo.value.errors
    )


def test_production_shaped_probe_rejects_an_unclassified_locked_joint(
    tmp_path: Path,
) -> None:
    with pytest.raises(ArticulatedNativeProbeError) as excinfo:
        _production_shaped_probe(
            tmp_path,
            locked_joint_prim_paths=[
                "/Asset/joints/drawer_slide",
                "/Asset/joints/drum_bearing",
                "/Asset/joints/selector_axis",
            ],
        )

    assert "articulated_native_probe_joint_partition_incomplete" in excinfo.value.errors


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
