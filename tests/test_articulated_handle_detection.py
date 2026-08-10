from __future__ import annotations

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.articulated_control_planner import (
    ArticulatedControlPlannerError,
    detect_articulated_handle_grasp_point,
)


def _box(stage, path, lo, hi):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(x, y, z)
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(
        [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
    )
    return mesh


def _door(path, *, handle_lo, handle_hi):
    """A slab door with a hinge on -x and a protruding handle."""

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
    _box(stage, "/Asset/upper_door/panel", (-0.35, 0.20, 0.9), (0.35, 0.31, 1.6))
    _box(stage, "/Asset/upper_door/handle", handle_lo, handle_hi)
    _box(stage, "/Asset/upper_door/liner", (-0.30, 0.10, 1.0), (0.30, 0.17, 1.5))
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/joints/hinge")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/cabinet")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/Asset/upper_door")])
    joint.CreateAxisAttr().Set("Z")
    joint.CreateLowerLimitAttr().Set(0.0)
    joint.CreateUpperLimitAttr().Set(90.0)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(-0.35, 0.31, 1.25))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.35, 0.31, 1.25))
    stage.GetRootLayer().Save()
    return path


def _detect(tmp_path, **overrides):
    arguments = {
        "usd_path": _door(
            tmp_path / "door.usda",
            handle_lo=(0.10, 0.31, 1.00),
            handle_hi=(0.28, 0.35, 1.05),
        ),
        "member_prim_path": "/Asset/upper_door",
        "hinge_point_world_m": [-0.35, 0.31, 1.25],
        "hinge_axis_world": [0.0, 0.0, 1.0],
    }
    arguments.update(overrides)
    return detect_articulated_handle_grasp_point(**arguments)


def test_the_handle_is_the_part_that_protrudes_outward(tmp_path) -> None:
    """A handle is the thing that sticks out the side you pull from.

    Picking by name would only work on assets that happen to be labelled,
    and generated twins are not.
    """

    found = _detect(tmp_path)

    assert found["handle_prim_path"] == "/Asset/upper_door/handle"
    assert found["grasp_point_world_m"][1] == pytest.approx(0.33, abs=0.01)
    assert found["lever_arm_m"] == pytest.approx(0.54, abs=0.02)


def test_interior_parts_are_never_mistaken_for_a_handle(tmp_path) -> None:
    """The liner protrudes too - inward. Grabbing it means reaching through the door."""

    found = _detect(tmp_path)

    assert "liner" not in found["handle_prim_path"]


def test_a_door_with_no_protruding_part_fails_closed(tmp_path) -> None:
    """A flush panel has nothing to grasp, and guessing a point would hide that."""

    with pytest.raises(ArticulatedControlPlannerError) as excinfo:
        _detect(
            tmp_path,
            usd_path=_door(
                tmp_path / "flush.usda",
                handle_lo=(0.10, 0.28, 1.00),
                handle_hi=(0.28, 0.30, 1.05),
            ),
        )

    assert any("no_protruding_handle" in e for e in excinfo.value.errors)


def test_the_grasp_point_sits_on_the_handle_not_inside_it(tmp_path) -> None:
    found = _detect(tmp_path)

    point = found["grasp_point_world_m"]
    assert 0.10 <= point[0] <= 0.28
    assert 1.00 <= point[2] <= 1.05


def test_detection_is_deterministic(tmp_path) -> None:
    assert _detect(tmp_path) == _detect(tmp_path)


def test_outward_is_derived_from_the_door_plane_not_its_centroid(tmp_path) -> None:
    """A hinge anchors on the door's front face, not through its middle.

    The real 840796 twin does exactly this, and taking the direction from the
    hinge to the panel centroid then tilts "outward" by nearly thirty degrees -
    far enough that the handle stops reading as protruding at all and a real,
    grabbable handle is reported as a flush panel. The door's own plane is what
    fixes the normal: its widest span perpendicular to the hinge axis.
    """

    path = tmp_path / "front_anchored.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(
        UsdGeom.Xform.Define(stage, "/Asset/upper_door").GetPrim()
    )
    _box(stage, "/Asset/upper_door/panel", (-0.358, 0.199, 0.92), (0.355, 0.306, 1.63))
    _box(stage, "/Asset/upper_door/pull", (-0.003, 0.306, 1.00), (0.242, 0.349, 1.04))
    # Six small hinge-side fittings, which drag any unweighted centroid sideways.
    for index in range(6):
        height = 1.00 + 0.1 * index
        _box(
            stage,
            f"/Asset/upper_door/fitting_{index:02d}",
            (-0.357, 0.177, height),
            (-0.344, 0.195, height + 0.006),
        )
    stage.GetRootLayer().Save()

    found = detect_articulated_handle_grasp_point(
        usd_path=path,
        member_prim_path="/Asset/upper_door",
        hinge_point_world_m=[-0.357, 0.350, 1.276],
        hinge_axis_world=[0.0, 0.0, 1.0],
    )

    assert found["outward_normal_world"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
    assert found["handle_prim_paths"] == ["/Asset/upper_door/pull"]
    assert found["protrusion_m"] == pytest.approx(0.043, abs=1e-3)
