"""A room scanned as one mesh keeps the station's footprint clear of its own furniture.

capture-447's collider is one mesh, ``/Root/mesh_0000_GLTF`` (3.25 M points, 9.2 m across): the
obstacle-box index drops its whole-room box past its 6 m cap, so the search had no obstacles and ranked
stances inside the dining table. A smaller room keeps that box and refuses every stance; a shell name
("scene") drops it by name. The room's own triangles decide instead.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("pxr")

from blueprint_pipeline import team_embodiment_stance as stance_module  # noqa: E402
from blueprint_pipeline.team_embodiment import TeamEmbodimentError  # noqa: E402
from blueprint_pipeline.team_embodiment_stance import ground_stance  # noqa: E402
from tests.test_team_embodiment import write_room_usd  # noqa: E402
from tests.test_team_scene_packet import _box as usd_box  # noqa: E402
from tests.test_team_scene_packet import _stage  # noqa: E402

# capture-447's oval dining table (its registered support) and the tray on it, in metres.
SUPPORT_MIN, SUPPORT_MAX, TABLE_TOP = (-1.00, -1.06), (0.29, 0.51), 0.8039
SUPPORT_TOP = 0.8189
START = [-0.40, -0.62, SUPPORT_TOP + 0.02]
TRAY_HALF = (0.26, 0.155, 0.02)
ROOM = ((-5.88, -4.31), (3.36, 4.90))  # the production mesh's plan bounds
FOOTPRINT = [0.15, 0.15]
CLEARANCE_M = 0.08  # the stand validator's obstacle clearance


def _oval(n: int = 96) -> np.ndarray:
    cx, cy = (SUPPORT_MIN[0] + SUPPORT_MAX[0]) / 2, (SUPPORT_MIN[1] + SUPPORT_MAX[1]) / 2
    a, b = (SUPPORT_MAX[0] - SUPPORT_MIN[0]) / 2, (SUPPORT_MAX[1] - SUPPORT_MIN[1]) / 2
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + a * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** 0.5,
                     cy + b * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** 0.5], axis=1)


def _in_oval(points: np.ndarray) -> np.ndarray:
    cx, cy = (SUPPORT_MIN[0] + SUPPORT_MAX[0]) / 2, (SUPPORT_MIN[1] + SUPPORT_MAX[1]) / 2
    a, b = (SUPPORT_MAX[0] - SUPPORT_MIN[0]) / 2, (SUPPORT_MAX[1] - SUPPORT_MIN[1]) / 2
    return (np.abs(points[:, 0] - cx) / a) ** 4 + (np.abs(points[:, 1] - cy) / b) ** 4 <= 1.0


def _table(outline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An open-bottomed table as a scan sees it: its sides and its top, no underside."""

    n = len(outline)
    centre = outline.mean(axis=0)
    points = np.vstack([np.c_[outline, np.zeros(n)], np.c_[outline, np.full(n, TABLE_TOP)], [[*centre, TABLE_TOP]]])
    k = np.arange(n)
    faces = np.vstack([np.c_[k, (k + 1) % n, n + (k + 1) % n], np.c_[k, n + (k + 1) % n, n + k],
                       np.c_[np.full(n, 2 * n), n + k, n + (k + 1) % n]])
    return points, faces


def _legged_table(top: float = TABLE_TOP) -> tuple[np.ndarray, np.ndarray]:
    """The same oval table on four legs, as a camera at standing height scans it.

    Its top and the 3 cm edge of its slab, never the slab's underside, and the floor seen between its legs.
    """

    outline = _oval()
    n, centre = len(outline), outline.mean(axis=0)
    points = np.vstack([np.c_[outline, np.full(n, top - 0.03)], np.c_[outline, np.full(n, top)], [[*centre, top]]])
    k = np.arange(n)
    rows = [(points, np.vstack([np.c_[k, (k + 1) % n, n + (k + 1) % n], np.c_[k, n + (k + 1) % n, n + k],
                                np.c_[np.full(n, 2 * n), n + k, n + (k + 1) % n]]))]
    rows += [_box((centre[0] + sx * 0.45 - 0.03, centre[1] + sy * 0.55 - 0.03, 0.0),
                  (centre[0] + sx * 0.45 + 0.03, centre[1] + sy * 0.55 + 0.03, top - 0.03), bottom=False)
             for sx in (-1, 1) for sy in (-1, 1)]
    offsets = np.cumsum([0] + [len(p) for p, _ in rows[:-1]])
    return np.vstack([p for p, _ in rows]), np.vstack([f + first for (_, f), first in zip(rows, offsets, strict=True)])


def _box(low: tuple[float, ...], high: tuple[float, ...], *, bottom: bool = True) -> tuple[np.ndarray, np.ndarray]:
    (x0, y0, z0), (x1, y1, z1) = low, high
    points = np.array([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                       (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], dtype=float)
    quads = [[4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]] + ([[0, 3, 2, 1]] if bottom else [])
    return points, np.array([t for q in quads for t in ([q[0], q[1], q[2]], [q[0], q[2], q[3]])])


def _floor(scale: float, keep: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """A tessellated floor (over the index's 5000-face split cap), less any cell ``keep`` refuses."""

    (x0, y0), (x1, y1) = ((v * scale for v in corner) for corner in ROOM)
    xs, ys = np.arange(x0, x1 + 1e-9, 0.1 * scale), np.arange(y0, y1 + 1e-9, 0.1 * scale)
    gx, gy = np.meshgrid(xs, ys)
    points = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
    nx = len(xs)
    a = (np.arange(len(ys) - 1)[:, None] * nx + np.arange(nx - 1)[None, :]).ravel()
    faces = np.r_[np.c_[a, a + 1, a + nx + 1], np.c_[a, a + nx + 1, a + nx]]
    if keep is not None:
        faces = faces[keep(points[faces].mean(axis=1)[:, :2])]
    return points, faces


def _ceiling(scale: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """The floor's tessellation at ``height``, every other row 2 cm low: a scanned ceiling's noise."""

    points, faces = _floor(scale)
    rows = np.round((points[:, 1] - points[:, 1].min()) / (0.1 * scale)).astype(int)
    return np.c_[points[:, :2], height - 0.02 * (rows % 2)], faces


def write_scanned_room(path: Path, *, name: str = "mesh_0000_GLTF", scale: float = 1.0,
                       parts: list[tuple[np.ndarray, np.ndarray]] | None = None, floor_keep: Any = None,
                       table: tuple[np.ndarray, np.ndarray] | None = None, wall_height_m: float = 2.6) -> Path:
    """One collision mesh under ``/Root``, as website normalization writes a reconstructed room."""

    from pxr import Usd, UsdGeom, UsdPhysics, Vt

    (x0, y0), (x1, y1) = ((v * scale for v in corner) for corner in ROOM)
    rows = [_floor(scale, floor_keep), _table(_oval()) if table is None else table, *(parts or [])]
    h = wall_height_m
    for low, high in (((x0, y0, 0.0), (x0 + 0.05, y1, h)), ((x1 - 0.05, y0, 0.0), (x1, y1, h)),
                      ((x0, y0, 0.0), (x1, y0 + 0.05, h)), ((x0, y1 - 0.05, 0.0), (x1, y1, h))):
        rows.append(_box(low, high))
    points, faces, first = [], [], 0
    for p, f in rows:
        points.append(p)
        faces.append(f + first)
        first += len(p)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Root").GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Root/" + name)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.vstack(points).astype(np.float32)))
    faces = np.vstack(faces).astype(np.int32)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    stage.GetRootLayer().Save()
    return path


def _search(path: Path, **overrides: Any) -> Any:
    return ground_stance(**{"collision_path": path, "collision_sha256": "sha256:" + "0" * 64,
                            "offset": [0.0, 0.0, 0.0], "subject_center": START,
                            "subject_minimum": [START[i] - TRAY_HALF[i] for i in range(3)],
                            "subject_maximum": [START[i] + TRAY_HALF[i] for i in range(3)],
                            "footprint_half_extent_m": FOOTPRINT, "root_world_z_m": SUPPORT_TOP, **overrides})


def text_parses(monkeypatch: Any, suffix: str) -> tuple[list[str], list[str]]:
    """The callers of every ``Usd.Stage.Open`` of a path ending in ``suffix``, and of those that read the file.

    An open reads the file when no stage of it is alive (``Sdf.Layer.Find`` has no layer for it).
    """

    import sys

    from pxr import Sdf, Usd

    opened = Usd.Stage.Open
    opens: list[str] = []
    parsed: list[str] = []

    def counting(*args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], str) and args[0].endswith(suffix):
            caller = sys._getframe(1).f_code.co_name
            opens.append(caller)
            if Sdf.Layer.Find(args[0]) is None:
                parsed.append(caller)
        return opened(*args, **kwargs)

    monkeypatch.setattr(Usd.Stage, "Open", staticmethod(counting))
    return opens, parsed


def footprint_samples(stance: dict[str, Any], *, grow_m: float = CLEARANCE_M) -> np.ndarray:
    """Points filling the station's footprint, grown by ``grow_m``, in plan."""

    x, y = stance["position_world_m"][:2]
    yaw = stance["yaw_rad"]
    hx, hy = (float(v) + grow_m for v in stance["footprint_half_extent_m"])
    u, v = (grid.ravel() for grid in np.meshgrid(np.linspace(-hx, hx, 15), np.linspace(-hy, hy, 15)))
    return np.c_[x + math.cos(yaw) * u - math.sin(yaw) * v, y + math.sin(yaw) * u + math.cos(yaw) * v]


@pytest.mark.parametrize(("name", "scale"), [
    ("mesh_0000_GLTF", 1.0),   # production: one mesh past the 6 m box cap, so the index had no obstacle
    ("mesh_0000_GLTF", 0.6),   # a smaller room: the index's one whole-room box covered every stance
    ("mesh_0000_scene", 1.0),  # a shell name: the index excluded the room by name
])
def test_a_single_mesh_room_keeps_every_station_off_its_table(tmp_path, name, scale) -> None:
    from blueprint_pipeline.scene_placement.usd_index import UsdSceneSpatialIndex

    room = write_scanned_room(tmp_path / "collision.usda", name=name, scale=scale)
    # The box index alone never sees the table: no box, or one box that is the whole room.
    boxes = UsdSceneSpatialIndex(usd_path=str(room)).obstacle_boxes()
    assert all(box.size()[0] > 5.0 and box.size()[1] > 5.0 for box in boxes)
    stances = _search(room, ranked=24)
    assert len(stances) == 24
    for stance in stances:
        assert not _in_oval(footprint_samples(stance)).any(), stance["position_world_m"]
        assert stance["position_world_m"][2] == pytest.approx(SUPPORT_TOP)  # the mount level with the support
    measured = stances[0]["room_mesh_clearance"]
    assert measured["room_mesh_prims"] == ["/Root/" + name]
    # The column reaches the tray's top, above the mount level with the support.
    assert measured["occupied_stances"] > 0
    assert measured["column_top_above_floor_m"] == pytest.approx(START[2] + TRAY_HALF[2])
    # The closest clear stance stands just off the table's near edge, facing the tray.
    x, y, _ = stances[0]["position_world_m"]
    assert math.hypot(x - START[0], y - START[1]) < 0.9


BENCH_LOW, BENCH_HIGH = (0.55, -1.00, 0.0), (1.05, -0.50, 0.45)  # a low bench beside the table


def _on_bench(xy: np.ndarray) -> np.ndarray:
    return ((xy[:, 0] >= BENCH_LOW[0]) & (xy[:, 0] <= BENCH_HIGH[0])
            & (xy[:, 1] >= BENCH_LOW[1]) & (xy[:, 1] <= BENCH_HIGH[1]))


def _below_the_table(body: str) -> dict[str, Any]:
    """A search whose station root stands below the dining table's top."""

    from blueprint_pipeline.scene_placement.robot_profile import UNITREE_G1_PROFILE

    if body == "g1":  # the G1 screening search: its pelvis, 0.79 m, is under a 0.80 m table top
        return {"root_world_z_m": None, "root_height_m": UNITREE_G1_PROFILE.pelvis_height_m,
                "footprint_half_extent_m": list(UNITREE_G1_PROFILE.footprint_half_extent_xyz[:2])}
    if body == "mobile":  # a free base's own low root
        return {"root_world_z_m": None, "root_height_m": 0.35}
    # The tray ends on the bench: the mount is level with the lower support, facing the whole region.
    target = [0.80, -0.75, BENCH_HIGH[2] + TRAY_HALF[2]]
    lower = [min(START[i], target[i]) - TRAY_HALF[i] for i in range(3)]
    upper = [max(START[i], target[i]) + TRAY_HALF[i] for i in range(3)]
    return {"root_world_z_m": BENCH_HIGH[2], "subject_center": [(a + b) / 2 for a, b in zip(lower, upper, strict=True)],
            "subject_minimum": lower, "subject_maximum": upper}


@pytest.mark.parametrize("side_m", [5.0, 7.0])  # the room mesh's box: kept (covers every stance), or dropped
@pytest.mark.parametrize("body", ["g1", "mobile", "franka_lower_support"])
def test_a_station_below_the_tables_top_never_stands_under_it(tmp_path, side_m, body) -> None:
    from blueprint_pipeline.scene_placement.usd_index import UsdSceneSpatialIndex

    # One dense mesh: the floor scanned under the table too, seen between its legs.
    room = write_scanned_room(tmp_path / "collision.usda", scale=side_m / (ROOM[1][0] - ROOM[0][0]),
                              table=_legged_table(), parts=[_box(BENCH_LOW, BENCH_HIGH, bottom=False)])
    sizes = [max(box.size()) for box in UsdSceneSpatialIndex(usd_path=str(room)).obstacle_boxes()]
    assert sizes == ([pytest.approx(side_m, abs=0.05)] if side_m < 6.0 else [])
    search = _below_the_table(body)
    stances = _search(room, ranked=24, **search)
    assert len(stances) == 24
    for stance in stances:
        grown = footprint_samples(stance)
        assert not _in_oval(grown).any() and not _on_bench(grown).any(), stance["position_world_m"]
        # On the floor: the scene's own floor, the root at the body's height above it or the mount's level.
        assert stance["floor_z_m"] == pytest.approx(0.0, abs=0.01)
        assert stance["position_world_m"][2] == pytest.approx(
            search["root_world_z_m"] if search["root_world_z_m"] is not None
            else stance["floor_z_m"] + search["root_height_m"])
    top = search.get("subject_maximum", [START[i] + TRAY_HALF[i] for i in range(3)])[2]
    assert stances[0]["room_mesh_clearance"]["column_top_above_floor_m"] == pytest.approx(top - stances[0]["floor_z_m"])


def test_a_station_never_stands_inside_furniture_the_scan_never_saw_beneath(tmp_path) -> None:
    # A tall cabinet beside the table, where the closest stance would stand: its sides clear a footprint
    # inside it and its top is above the mount, but the scan saw no floor under it.
    low, high = (-0.9, -2.2, 0.0), (0.1, -1.2, 1.6)

    def outside(xy: np.ndarray) -> np.ndarray:
        return ~((xy[:, 0] > low[0]) & (xy[:, 0] < high[0]) & (xy[:, 1] > low[1]) & (xy[:, 1] < high[1]))

    room = write_scanned_room(tmp_path / "collision.usda", parts=[_box(low, high, bottom=False)], floor_keep=outside)
    stances = _search(room, ranked=24)
    for stance in stances:
        samples = footprint_samples(stance, grow_m=0.0)
        assert outside(samples).all() and not _in_oval(samples).any(), stance["position_world_m"]
    assert stances[0]["room_mesh_clearance"]["floor_unobserved_stances"] > 0


@pytest.mark.parametrize("side_m", [5.0, 7.0])
@pytest.mark.parametrize("body", ["franka", "g1", "mobile"])
def test_a_station_never_stands_inside_closed_furniture_over_rebuilt_floor(tmp_path, side_m, body) -> None:
    # The same cabinet, 1.8 m tall (a fridge, a pantry), its floor rebuilt by the reconstruction: a footprint
    # inside it crosses none of its sides, its top is above the task, and the floor under it is observed.
    low, high = (-0.9, -2.2, 0.0), (0.1, -1.2, 1.8)

    def inside(xy: np.ndarray) -> np.ndarray:
        return (xy[:, 0] > low[0]) & (xy[:, 0] < high[0]) & (xy[:, 1] > low[1]) & (xy[:, 1] < high[1])

    room = write_scanned_room(tmp_path / "collision.usda", scale=side_m / (ROOM[1][0] - ROOM[0][0]),
                              parts=[_box(low, high, bottom=False)])
    stances = _search(room, ranked=24, **({} if body == "franka" else _below_the_table(body)))
    assert len(stances) == 24
    for stance in stances:
        samples = footprint_samples(stance, grow_m=0.0)
        assert not inside(samples).any() and not _in_oval(samples).any(), stance["position_world_m"]
        assert stance["floor_z_m"] == pytest.approx(0.0, abs=0.01)
    measured = stances[0]["room_mesh_clearance"]
    assert measured["roofed_stances"] > 0 and measured["floor_unobserved_stances"] == 0
    assert measured["roof_top_above_floor_m"] == pytest.approx(2.05)


@pytest.mark.parametrize("overhead", ["low_ceiling", "ceiling_fan", "lamp"])
def test_a_ceiling_a_fan_or_a_lamp_never_roofs_the_station(tmp_path, overhead) -> None:
    # Over open floor: a 2.0 m ceiling with scan noise (below the headroom; it stays 25 cm under the room's
    # top), a 1 m fan at 2.15 m (above the headroom) and a 20 cm lamp at 1.5 m (over part of a footprint at
    # most) leave every stance of the same room without them.
    height = 2.0 if overhead == "low_ceiling" else 2.6
    plain = _search(write_scanned_room(tmp_path / "plain.usda", wall_height_m=height), ranked=24)
    x, y, _ = plain[0]["position_world_m"]
    parts = {"low_ceiling": [_ceiling(1.0, height)],
             "ceiling_fan": [_box((x - 0.5, y - 0.5, 2.15), (x + 0.5, y + 0.5, 2.2))],
             "lamp": [_box((x - 0.1, y - 0.1, 1.45), (x + 0.1, y + 0.1, 1.55))]}[overhead]
    stances = _search(write_scanned_room(tmp_path / "collision.usda", parts=parts, wall_height_m=height), ranked=24)
    assert [s["position_world_m"] for s in stances] == [s["position_world_m"] for s in plain]
    measured = stances[0]["room_mesh_clearance"]
    assert measured["roofed_stances"] == 0
    assert measured["roof_top_above_floor_m"] == pytest.approx(min(2.05, height - 0.25))


def test_a_single_mesh_room_with_no_clear_space_refuses_the_station_by_name(tmp_path) -> None:
    # No floor was reconstructed anywhere near the task: no station can be shown to stand clear.
    room = write_scanned_room(tmp_path / "collision.usda",
                              floor_keep=lambda xy: np.hypot(xy[:, 0] - START[0], xy[:, 1] - START[1]) > 2.4)
    with pytest.raises(TeamEmbodimentError) as refused:
        _search(room)
    assert refused.value.blockers == ["team_embodiment_room_mesh_no_clear_stance"]


def test_scenes_with_their_own_floor_prim_keep_the_box_search(tmp_path) -> None:
    room = write_room_usd(tmp_path / "collision.usda")
    found = ground_stance(collision_path=room, collision_sha256="sha256:" + "0" * 64, offset=[0.0, 0.0, 0.0],
                                 subject_center=[2.0, 0.0, 0.8], subject_minimum=[1.95, -0.05, 0.75],
                                 subject_maximum=[2.05, 0.05, 0.85], footprint_half_extent_m=[0.15, 0.15],
                                 root_world_z_m=0.75)
    assert "room_mesh_clearance" not in found
    # A floorless scene's own fixtures are not the room either: the object rests on the counter, above it.
    bare = _stage(tmp_path / "bare.usda", "/Root")
    usd_box(bare, "/Root/Counter", (1.0, -0.4, 0.0), (1.6, 0.4, 0.9))
    usd_box(bare, "/Root/Shelf", (-1.5, 1.0, 0.0), (-0.5, 1.4, 1.8))
    bare.GetRootLayer().Save()
    meshes: list[tuple[str, int]] = []
    vertices, faces = stance_module._world_triangles(tmp_path / "bare.usda", [0.0, 0.0, 0.0], meshes)
    assert stance_module._room_mesh_occupancy(vertices, faces, meshes, floor_z=0.0, tolerance=0.05, column_top_m=0.9,
                                       centre=[1.3, 0.0], subject_top_z=1.0, reach_m=2.0) is None


def test_a_stance_search_reads_a_text_room_from_disk_once(tmp_path, monkeypatch) -> None:
    # A website collider is USDA text (capture-447's: 292 MB, parsed at about twelve times its size). The
    # search re-opens it to leave the room out of the box index; with its first stage let go, that open
    # parsed the whole file a second time, holding the room's triangles meanwhile.
    room = write_scanned_room(tmp_path / "collision.usda")
    opens, parsed = text_parses(monkeypatch, "collision.usda")
    stances = _search(room, ranked=24)
    assert "_stage_without" in opens and parsed == ["ground_stance"]
    assert stances[0]["room_mesh_clearance"]["room_mesh_prims"] == ["/Root/mesh_0000_GLTF"]
    # The box search of a scene with its own floor re-opens it too (the index), from the same parse.
    floored = write_room_usd(tmp_path / "floored.usda")
    _, floor_parsed = text_parses(monkeypatch, "floored.usda")
    found = ground_stance(collision_path=floored, collision_sha256="sha256:" + "0" * 64, offset=[0.0, 0.0, 0.0],
                          subject_center=[2.0, 0.0, 0.8], subject_minimum=[1.95, -0.05, 0.75],
                          subject_maximum=[2.05, 0.05, 0.85], footprint_half_extent_m=[0.15, 0.15],
                          root_world_z_m=0.75)
    assert "room_mesh_clearance" not in found and floor_parsed == ["ground_stance"]
