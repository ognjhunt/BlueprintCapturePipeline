"""Tests for blueprint_pipeline.splat_scene_analysis."""
from __future__ import annotations

import numpy as np

from blueprint_pipeline.gaussian_splat_decode import SplatData
import math

from blueprint_pipeline.splat_scene_analysis import (
    DEFAULT_CAMERA_IDS,
    analyze_scene,
    derive_eval_cameras,
    suggest_robot_start,
)


def _box_room(nx=40, ny=60, nz=18) -> SplatData:
    """Synthetic room: x in [0,5], y in [0,8], z in [0,2.5] (z is the smallest extent
    => up axis). Splats cover the floor + 4 walls + ceiling, floor densest."""
    pts = []
    xs = np.linspace(0, 5, nx)
    ys = np.linspace(0, 8, ny)
    zs = np.linspace(0, 2.5, nz)
    # floor (dense) and ceiling
    for x in xs:
        for y in ys:
            pts.append((x, y, 0.0))
            pts.append((x, y, 0.0))  # floor doubled => denser bottom
            pts.append((x, y, 2.5))
    # walls
    for x in xs:
        for z in zs:
            pts.append((x, 0.0, z))
            pts.append((x, 8.0, z))
    for y in ys:
        for z in zs:
            pts.append((0.0, y, z))
            pts.append((5.0, y, z))
    arr = np.array(pts, dtype=np.float32)
    n = arr.shape[0]
    return SplatData(
        count=n,
        xyz=arr,
        opacity=np.full(n, 6.0, dtype=np.float32),  # sigmoid(6) ~ 1 => visible
        f_dc=np.zeros((n, 3), dtype=np.float32),
        scales=np.full((n, 3), -3.0, dtype=np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        properties=(),
    )


def test_up_axis_floor_and_center_detected() -> None:
    geom = analyze_scene(_box_room())
    assert geom.up_axis == 2  # z is the smallest extent
    assert geom.up_sign == 1.0  # floor (denser) at low z
    assert abs(geom.floor - 0.0) < 0.2
    assert abs(geom.ceiling - 2.5) < 0.3
    # horizontal center near room middle
    assert abs(geom.center[0] - 2.5) < 0.6
    assert abs(geom.center[1] - 4.0) < 0.8
    assert set(geom.horizontal_axes) == {0, 1}
    assert geom.radius > 2.0


def test_suggested_start_on_floor_within_footprint() -> None:
    geom = analyze_scene(_box_room())
    start = geom.suggested_start
    pos = start["position"]
    assert abs(pos[2] - geom.floor) < 1e-6  # standing on the floor
    assert geom.aabb_min[0] - 0.5 <= pos[0] <= geom.aabb_max[0] + 0.5
    assert geom.aabb_min[1] - 0.5 <= pos[1] <= geom.aabb_max[1] + 0.5
    assert "facing_yaw_deg" in start
    # eye position is above the floor along +z
    assert start["eye_position"][2] > geom.floor


def test_derive_eval_cameras_complete_and_finite() -> None:
    geom = analyze_scene(_box_room())
    cams = derive_eval_cameras(geom)
    assert [c["id"] for c in cams] == list(DEFAULT_CAMERA_IDS)
    for cam in cams:
        spec = cam["spec"]
        for key in ("pos", "target", "up"):
            assert len(spec[key]) == 3
            assert all(np.isfinite(spec[key]))
        assert spec["fov"] > 0
        # camera should not coincide with its target
        assert np.linalg.norm(np.array(spec["pos"]) - np.array(spec["target"])) > 1e-3


def test_up_axis_override() -> None:
    geom = analyze_scene(_box_room(), up_axis=1)
    assert geom.up_axis == 1
    assert set(geom.horizontal_axes) == {0, 2}


def test_task_aware_start_faces_target() -> None:
    room = _box_room()
    geom = analyze_scene(room)
    target = [5.0, 4.0, 0.6]  # near the +x wall, mid-room
    start = suggest_robot_start(room, geom, task_target=target, standoff=1.5)
    assert start["task_target"] == [5.0, 4.0, 0.6]
    assert start["standoff_distance"] == 1.5
    pos = start["position"]
    expected = math.degrees(math.atan2(target[1] - pos[1], target[0] - pos[0]))
    # facing yaw aims at the target (equal modulo 360)
    assert abs(((start["facing_yaw_deg"] - expected + 180) % 360) - 180) < 1e-3
    # the start stands roughly a standoff away from the target horizontally
    d = math.hypot(target[0] - pos[0], target[1] - pos[1])
    assert 0.4 < d < 4.0


def test_focus_point_aims_task_cameras() -> None:
    import numpy as np

    geom = analyze_scene(_box_room())
    focus = [5.0, 4.0, 0.8]
    cams = {c["id"]: c for c in derive_eval_cameras(geom, focus_point=focus)}
    np.testing.assert_allclose(cams["task_focus"]["spec"]["target"], focus, atol=1e-6)
    # wrist aims at the focus point in the horizontal plane (its up-coord is floor-biased)
    wt = cams["wrist"]["spec"]["target"]
    np.testing.assert_allclose([wt[0], wt[1]], [focus[0], focus[1]], atol=1e-6)
    # without a focus point, task cameras default to the scene center (unchanged behavior)
    default_cams = {c["id"]: c for c in derive_eval_cameras(geom)}
    np.testing.assert_allclose(
        default_cams["task_focus"]["spec"]["target"], geom.center, atol=1e-6
    )
