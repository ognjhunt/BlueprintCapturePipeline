from __future__ import annotations

from pathlib import Path

import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.handle_standoff import (
    HANDLE_STANDOFF_SCHEMA_VERSION,
    HandleStandoffError,
    author_handle_standoff,
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
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return mesh


def _door(path: Path) -> Path:
    """A door whose handle is moulded flush to the panel, as generated twins are."""

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    link = UsdGeom.Xform.Define(stage, "/Asset/upper_door")
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr().Set(11.0)
    _box(stage, "/Asset/upper_door/panel", (-0.35, 0.20, 0.92), (0.35, 0.306, 1.63))
    _box(stage, "/Asset/upper_door/handle", (-0.003, 0.306, 1.004), (0.242, 0.349, 1.041))
    stage.GetRootLayer().Save()
    return path


def _author(tmp_path: Path, **overrides):
    arguments = {
        "source_usd_path": _door(tmp_path / "door.usda"),
        "destination": tmp_path / "standoff.usda",
        "handle_prim_paths": ["/Asset/upper_door/handle"],
        "panel_face_offset_m": 0.306,
        "outward_normal": [0.0, 1.0, 0.0],
        "standoff_m": 0.030,
        "post_fraction": 0.12,
    }
    arguments.update(overrides)
    return author_handle_standoff(**arguments)


def test_the_bar_ends_up_with_room_for_fingers_behind_it(tmp_path: Path) -> None:
    """A flush handle can only be pinched; a standing bar can be hooked."""

    receipt = _author(tmp_path)

    assert receipt["achieved_clearance_m"] == pytest.approx(0.030, abs=1e-6)
    assert receipt["schema_version"] == HANDLE_STANDOFF_SCHEMA_VERSION
    stage = Usd.Stage.Open(receipt["standoff_usd_path"])
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bar = stage.GetPrimAtPath("/Asset/upper_door/handle")
    bounds = cache.ComputeWorldBound(bar).ComputeAlignedRange()
    assert float(bounds.GetMin()[1]) == pytest.approx(0.336, abs=1e-3)


def test_the_bar_is_held_up_by_posts_not_floating(tmp_path: Path) -> None:
    """A bar with nothing joining it to the door is a bar that falls off.

    It would also read as graspable while being physically unsupported, which
    is worse than the flush handle it replaced.
    """

    receipt = _author(tmp_path)

    stage = Usd.Stage.Open(receipt["standoff_usd_path"])
    posts = [
        str(p.GetPath())
        for p in Usd.PrimRange(stage.GetPrimAtPath("/Asset/upper_door"))
        if p.IsA(UsdGeom.Mesh) and "post" in p.GetName()
    ]
    assert len(posts) == 2
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for path in posts:
        bounds = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        # Each post must actually span the gap it is bridging.
        assert float(bounds.GetMin()[1]) <= 0.3061
        assert float(bounds.GetMax()[1]) >= 0.3359


def test_the_posts_leave_a_reachable_span_between_them(tmp_path: Path) -> None:
    """Posts at the ends are supports; posts everywhere are a solid block again."""

    receipt = _author(tmp_path)

    assert receipt["clear_span_m"] > 0.15
    assert receipt["clear_span_m"] < 0.245


def test_a_standoff_smaller_than_a_finger_fails_closed(tmp_path: Path) -> None:
    """Moving a handle 3mm off the door buys nothing and hides that it did."""

    with pytest.raises(HandleStandoffError) as excinfo:
        _author(tmp_path, standoff_m=0.003)

    assert any("standoff_below_finger_clearance" in e for e in excinfo.value.errors)


def test_the_link_mass_and_colliders_survive(tmp_path: Path) -> None:
    receipt = _author(tmp_path)

    stage = Usd.Stage.Open(receipt["standoff_usd_path"])
    link = stage.GetPrimAtPath("/Asset/upper_door")
    assert UsdPhysics.MassAPI(link).GetMassAttr().Get() == 11.0
    bar = stage.GetPrimAtPath("/Asset/upper_door/handle")
    assert bar.HasAPI(UsdPhysics.CollisionAPI)


def test_the_source_is_never_written_over(tmp_path: Path) -> None:
    source = _door(tmp_path / "door.usda")
    before = source.read_bytes()

    _author(tmp_path, source_usd_path=source)

    assert source.read_bytes() == before


def test_authoring_is_deterministic(tmp_path: Path) -> None:
    first = _author(tmp_path, destination=tmp_path / "a.usda")
    second = _author(tmp_path, destination=tmp_path / "b.usda")

    assert first["standoff_usd_sha256"] == second["standoff_usd_sha256"]
