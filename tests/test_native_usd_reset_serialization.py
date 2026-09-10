"""Replay the native USD value types that caused V22 reset-channel gaps."""

import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.policy_scientific_reset import (
    _native,
    _native_usd_attribute,
    compare_reset_readbacks,
    read_native_reset_channels,
    seal_reset_readback,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Gf.Vec3f(0, 0, -1), [0.0, 0.0, -1.0]),
        (Gf.Vec3d(1, 2, 3), [1.0, 2.0, 3.0]),
        (Gf.Matrix4d(1), np.eye(4).tolist()),
        (Vt.Vec3fArray([Gf.Vec3f(1, 2, 3)]), [[1.0, 2.0, 3.0]]),
        (Vt.IntArray([0, 1, 2]), [0, 1, 2]),
        (Gf.Quatf(1, Gf.Vec3f(0.1, 0.2, 0.3)),
            {"real": 1.0, "imaginary": list(Gf.Vec3f(0.1, 0.2, 0.3))}),
    ],
)
def test_native_usd_sequences_preserve_numeric_values(value, expected):
    assert _native(value) == expected


def test_unknown_and_nonfinite_native_values_remain_explicit_failures():
    with pytest.raises(TypeError):
        _native(object())
    with pytest.raises(ValueError):
        _native(Gf.Vec3d(0, float("nan"), 1))
    with pytest.raises(ValueError, match="native_value_missing"):
        _native(None)


def test_only_exact_unauthored_usd_center_of_mass_fallback_is_retained_as_nonnumeric():
    stage = Usd.Stage.CreateInMemory()
    mass = UsdPhysics.MassAPI.Apply(UsdGeom.Cube.Define(stage, "/cube").GetPrim())
    attribute = mass.GetCenterOfMassAttr()
    fallback = _native_usd_attribute(attribute)
    assert fallback == {"source": "usd_schema_fallback", "usd_type": "point3f",
        "value_tokens": ["-inf", "-inf", "-inf"], "measured_numeric_value": False}
    assert _native_usd_attribute(mass.GetPrincipalAxesAttr()) == {
        "real": 0.0, "imaginary": [0.0, 0.0, 0.0]}
    attribute.Set(Gf.Vec3f(1, 2, 3))
    assert _native_usd_attribute(attribute) == [1.0, 2.0, 3.0]
    attribute.Set(Gf.Vec3f(-float("inf")))
    with pytest.raises(ValueError):
        _native_usd_attribute(attribute)


def _native_usd_fixture(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    physics = UsdPhysics.Scene.Define(stage, "/World/physics")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    subject = UsdGeom.Cube.Define(stage, "/World/envs/env_0/task_object")
    subject.AddTranslateOp().Set(Gf.Vec3d(1.2, -0.4, 0.1))
    UsdPhysics.CollisionAPI.Apply(subject.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(subject.GetPrim())
    UsdPhysics.MassAPI.Apply(subject.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/room")
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    light = UsdLux.DomeLight.Define(stage, "/World/key_light")
    light.CreateColorAttr(Gf.Vec3f(1.0, 0.5, 0.25))
    light.CreateIntensityAttr(1000.0)

    omni, omni_usd = ModuleType("omni"), ModuleType("omni.usd")
    omni.__path__ = []
    omni.usd = omni_usd
    omni_usd.get_context = lambda: NS(get_stage=lambda: stage)
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.usd", omni_usd)

    pose = np.array([[1.2, -0.4, 0.1, 1.0, 0.0, 0.0, 0.0]])
    body = NS(data=NS(root_pose_w=pose, root_vel_w=np.zeros((1, 6))),
        root_physx_view=NS(get_masses=lambda: np.array([[1.25]]),
            get_inertias=lambda: np.ones((1, 9)),
            get_material_properties=lambda: np.array([[0.5, 0.4, 0.0]])))
    robot = NS(data=NS(joint_pos=np.zeros((1, 7)), joint_vel=np.zeros((1, 7)),
        root_pose_w=pose, root_vel_w=np.zeros((1, 6)),
        joint_limits=np.array([[[-1.0, 1.0]] * 7])))
    contact = NS(data=NS(net_forces_w=np.zeros((1, 1, 3)),
        force_matrix_w=np.zeros((1, 1, 1, 3))))
    built = NS(env=NS(unwrapped=NS(
        scene={"robot": robot, "task_object": body, "contact": contact},
        sim=NS(get_physics_dt=lambda: 1 / 120), step_dt=1 / 15)),
        plan={"objects": [{"name": "task_object", "object_type": "RIGID", "task_subject": True},
            {"name": "room", "object_type": "USD"}], "scenario": {}},
        scene_asset_names={"task_object": "task_object", "room": "room"},
        contact_sensor_names={"support": ["contact"]})
    episode = NS(read_control_observation_metadata=lambda: {
        "calibrations": {"external": {"resolution": [64, 32]}, "wrist": {"resolution": [64, 32]}}})
    return NS(stage=stage, built=built, episode=episode, points=points, light=light)


def test_real_usd_stage_retains_all_reset_channels_and_detects_changed_light(monkeypatch):
    fixture = _native_usd_fixture(monkeypatch)

    def read(candidate):
        return seal_reset_readback(binding={"candidate_id": candidate, "cell_id": "anchor", "seed": 31,
            "task_spec_digest": "sha256:" + "1" * 64,
            "resolved_scenario_digest": "sha256:" + "2" * 64},
            **read_native_reset_channels(fixture.built, fixture.episode))

    left = read("pi05_droid")
    assert left["complete"] is True and left["gaps"] == []
    observed = left["observed"]
    assert observed["physics"]["scene_attributes"]["/World/physics"]["physics:gravityDirection"] == [0.0, 0.0, -1.0]
    assert observed["scene_assets"]["task_object"]["world_transform"][3][:3] == [1.2, -0.4, 0.1]
    assert observed["lighting"]["/World/key_light"]["inputs"]["inputs:color"] == [1.0, 0.5, 0.25]
    assert observed["colliders"]["/World/envs/env_0/room"]["geometry_attribute_digests"]["points"] == canonical_digest({"value": fixture.points})
    properties = observed["colliders"]["/World/envs/env_0/task_object"]["properties"]
    assert properties["physics:centerOfMass"]["measured_numeric_value"] is False
    assert properties["physics:principalAxes"] == {"real": 0.0, "imaginary": [0.0, 0.0, 0.0]}
    assert compare_reset_readbacks(left, read("groot_n17_droid"))["status"] == "matched"

    fixture.light.GetIntensityAttr().Set(1001.0)
    parity = compare_reset_readbacks(left, read("groot_n17_droid"))
    assert parity["status"] == "mismatch"
    assert parity["comparison_eligible"] is False
    assert any(path.startswith("/lighting/") for path in parity["mismatches"])
