from __future__ import annotations

import json
from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.articulated_controls_probe import (
    CONTROLS_PROBE_SCHEMA_VERSION,
    REQUIRED_CONTROLS_READBACKS,
    ArticulatedControlsProbeError,
    build_articulated_controls_probe,
)


def _twin(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(asset.GetPrim())
    for link in ("cabinet", "upper_door"):
        UsdPhysics.RigidBodyAPI.Apply(
            UsdGeom.Xform.Define(stage, f"/Asset/{link}").GetPrim()
        )
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/joints/upper_door_hinge")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/cabinet")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/Asset/upper_door")])
    joint.CreateAxisAttr().Set("Z")
    joint.CreateLowerLimitAttr().Set(0.0)
    joint.CreateUpperLimitAttr().Set(90.0)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(-0.357, 0.35, 1.276))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateDampingAttr().Set(3.0)
    drive.CreateMaxForceAttr().Set(300.0)
    stage.GetRootLayer().Save()
    return path


def _build(tmp_path: Path, **overrides):
    arguments = {
        "twin_usd_path": _twin(tmp_path / "twin.usda"),
        "destination": tmp_path / "probe",
        "task_joint_prim_path": "/Asset/joints/upper_door_hinge",
        "task_link_prim_path": "/Asset/upper_door",
        "handle_grasp_point_local_m": [0.1375, 0.3276, 1.023],
        "hinge_point_local_m": [-0.357, 0.35, 1.276],
        "hinge_axis_local": [0.0, 0.0, 1.0],
        "target_open_angle_degrees": 50.0,
        "success_angle_window_degrees": [45.0, 55.0],
        "positive_force_schedule": [
            {"until_angle_degrees": 6.0, "handle_force_n": 30.0},
            {"until_angle_degrees": 40.0, "handle_force_n": 3.0},
        ],
        "seal_breakaway_torque_n_m": 12.0,
        "seal_angular_width_degrees": 5.0,
    }
    arguments.update(overrides)
    return build_articulated_controls_probe(**arguments)


def test_the_probe_freezes_both_controls_before_anything_runs(tmp_path: Path) -> None:
    """A positive alone proves nothing: the door might open on its own."""

    receipt = _build(tmp_path)

    assert receipt["status"] == "frozen_not_executed"
    assert set(receipt["controls"]) == {"zero_action_negative", "forced_positive"}
    assert receipt["schema_version"] == CONTROLS_PROBE_SCHEMA_VERSION


def test_the_negative_applies_no_force_at_all(tmp_path: Path) -> None:
    receipt = _build(tmp_path)

    assert receipt["controls"]["zero_action_negative"]["applied_handle_force_n"] == 0.0
    assert (
        receipt["controls"]["zero_action_negative"]["expected_outcome"]
        == "door_does_not_reach_success_window"
    )


def test_the_positive_eases_off_before_it_releases(tmp_path: Path) -> None:
    """One constant force cannot both break this seal and stop in the window.

    The 24 N needed to crack the gasket leaves the door coasting 85 degrees
    past release; anything gentle enough to stop in time never opens it. The
    schedule is the resolution, and it is also what the measured trace looks
    like.
    """

    schedule = _build(tmp_path)["controls"]["forced_positive"]["force_schedule"]

    assert len(schedule) >= 2
    assert schedule[0]["handle_force_n"] > schedule[-1]["handle_force_n"]
    assert schedule[0]["hinge_torque_n_m"] > 12.0


def test_a_release_past_the_success_window_fails_closed(tmp_path: Path) -> None:
    """Releasing at the far edge guarantees the coast overshoots it."""

    with pytest.raises(ArticulatedControlsProbeError) as excinfo:
        _build(
            tmp_path,
            positive_force_schedule=[
                {"until_angle_degrees": 6.0, "handle_force_n": 30.0},
                {"until_angle_degrees": 80.0, "handle_force_n": 3.0},
            ],
        )

    assert any("release_after_success_window" in e for e in excinfo.value.errors)


def test_the_positive_releases_before_the_window_is_judged(tmp_path: Path) -> None:
    """Holding force through the measurement would score the pusher, not the door.

    The whole question is whether the door stays where it is put, so the force
    has to stop and the settle window has to be measured after it does.
    """

    positive = _build(tmp_path)["controls"]["forced_positive"]

    assert positive["release_before_settle"] is True
    assert positive["settle_steps"] > 0
    assert positive["expected_outcome"] == "door_holds_inside_success_window"


def test_the_applied_force_must_beat_the_seal_it_has_to_break(tmp_path: Path) -> None:
    """A positive that cannot break its own gasket is a broken experiment."""

    with pytest.raises(ArticulatedControlsProbeError) as excinfo:
        _build(
            tmp_path,
            positive_force_schedule=[
                {"until_angle_degrees": 6.0, "handle_force_n": 5.0}
            ],
        )

    assert any("force_below_seal_breakaway" in e for e in excinfo.value.errors)


def test_a_success_window_outside_the_authored_limit_fails_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ArticulatedControlsProbeError) as excinfo:
        _build(tmp_path, success_angle_window_degrees=[85.0, 120.0])

    assert any("window_beyond_authored_limit" in e for e in excinfo.value.errors)


def test_a_twin_that_already_has_a_physics_scene_is_not_given_a_second(
    tmp_path: Path,
) -> None:
    """Two PhysicsScenes stop the run, and the message does not say why.

    PhysX reports "Physics scenes stepping is not the same" and shuts the app
    down a fraction of a second later. It reads as a stepping-configuration
    warning, not as "your overlay duplicated a prim the asset already had",
    and it costs a launch to find out.
    """

    source = _twin(tmp_path / "with_scene.usda")
    stage = Usd.Stage.Open(str(source))
    UsdPhysics.Scene.Define(stage, "/Asset/PhysicsScene")
    stage.GetRootLayer().Save()

    receipt = _build(tmp_path, twin_usd_path=source, destination=tmp_path / "p2")

    composed = Usd.Stage.Open(receipt["stages"]["controls_stage"]["path"])
    scenes = [str(p.GetPath()) for p in composed.Traverse() if p.IsA(UsdPhysics.Scene)]
    assert scenes == ["/Asset/PhysicsScene"]
    assert receipt["physics_scene"]["authored_by_probe"] is False
    assert receipt["physics_scene"]["prim_path"] == "/Asset/PhysicsScene"


def test_a_twin_without_a_physics_scene_gets_exactly_one(tmp_path: Path) -> None:
    receipt = _build(tmp_path)

    composed = Usd.Stage.Open(receipt["stages"]["controls_stage"]["path"])
    scenes = [str(p.GetPath()) for p in composed.Traverse() if p.IsA(UsdPhysics.Scene)]
    assert len(scenes) == 1
    assert receipt["physics_scene"]["authored_by_probe"] is True


def test_the_probe_stages_the_twin_it_pinned(tmp_path: Path) -> None:
    receipt = _build(tmp_path)

    staged = Path(receipt["stages"]["controls_stage"]["path"])
    assert staged.is_file()
    twin = Path(receipt["stages"]["twin_copy"]["path"])
    assert twin.is_file()
    opened = Usd.Stage.Open(str(staged))
    joint = opened.GetPrimAtPath("/Asset/joints/upper_door_hinge")
    assert joint.IsValid()


def test_every_required_readback_is_declared(tmp_path: Path) -> None:
    receipt = _build(tmp_path)

    assert set(receipt["required_readbacks"]) == set(REQUIRED_CONTROLS_READBACKS)


def test_the_receipt_round_trips_and_is_deterministic(tmp_path: Path) -> None:
    first = _build(tmp_path)
    stored = json.loads(
        (Path(first["stages"]["controls_stage"]["path"]).parent
         / "articulated_controls_probe_spec.json").read_text(encoding="utf-8")
    )
    assert stored == first
    second = _build(tmp_path, destination=tmp_path / "probe2")
    assert (
        first["stages"]["controls_stage"]["sha256"]
        == second["stages"]["controls_stage"]["sha256"]
    )
