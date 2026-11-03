"""Where a robot starts: a clear spot on the scene's floor, facing the task.

Beta journey step: L5 team policy evaluation. A mobile manipulator or a
humanoid stands on the floor itself; a fixed arm stands on a mount whose top is
level with the task's support (``team_scene_packet``). Either starts where a
person would stand to do the job: the clear floor spot closest to the task
object, facing it, with nothing that reaches the floor under its footprint
below its root (``scene_placement.validation.validate_stand_pose``, the
validator the G1 humanoid lane uses). The floor is the dominant low horizontal
band of the scene's own collision mesh (``external_scene_robot_placement``).
A task part that moves, such as a door or a drawer, also keeps the station out
of its way: no stance the part strikes on its way to success, or anywhere on a
fall gravity carries it through, and a stance it never reaches is preferred.
A room scanned as one mesh (a website capture's collider: floor, furniture and
walls in one prim) has no per-object boxes to clip against, so its own
triangles are measured instead: nothing of it may rise into the station's
column above the footprint, from the floor to the higher of the station's root
and the task's own top, the floor must be observed under the footprint, and no
surface of it below the headroom may roof the whole footprint, as a closed
cabinet's top does over floor the reconstruction rebuilt inside it
(``_room_mesh_occupancy``).
A website collider normalized before 2026-11-03 is a USDA text file
(capture-447's: 292 MB) that OpenUSD parses again at every open finding no stage
of it alive, so a search reads it through its read scope's one binary copy
(``usd_binary_read_cache``), the caller's when it has one.
Host-side and deterministic (OpenUSD), so a robot with no clear stance is
refused by name before any GPU time is spent.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .team_embodiment import TeamEmbodimentError
from .usd_binary_read_cache import read_scoped, readable

SCHEMA = "blueprint.team_free_base_stance.v1"
SEARCH_RADIUS_M = 1.6
GRID_STEP_M = 0.05
#: A task supported by the floor itself (an installed appliance, a floor
#: cabinet) puts a fixed arm's mount on the floor; this absorbs floor-fit noise.
FLOOR_MOUNT_TOLERANCE_M = 0.02
#: A collision mesh holding at least this share of the floor band's area, rising
#: above it and surrounding the task, is the room itself (a scan): a fixture on
#: a separate floor prim holds a small share (its bottom face), a floor prim
#: rises nowhere, and a fixture the task object rests on ends below the object.
ROOM_MESH_FLOOR_SHARE = 0.5
#: Room-mesh geometry this far above the floor enters a station's column; floor
#: noise below it does not (the external placement probe's lower obstacle bound).
ROOM_MESH_FLOOR_CLEARANCE_M = 0.04
#: The column above a footprint reaches the station's root (a fixed arm's mount
#: top) and the task's own top, and at least the stand validator's own
#: floor-obstacle band. Below the task's top, a room surface over the footprint
#: is the furniture the task stands at (a table top over legs the scan saw floor
#: between), never headroom; above it (a wall cabinet, a lamp) it stays the
#: validator's overhead clutter.
ROOM_MESH_MINIMUM_COLUMN_M = 0.40
#: A footprint whose centre and four corners all stand under room-mesh surfaces
#: lower than this above the floor is inside furniture (a wardrobe's or a
#: fridge's top over its side walls, with floor rebuilt beneath it), not on
#: open floor. A ceiling fan or a lamp hung over a walkway clears it; a lamp, a
#: shelf or a wall cabinet lower down covers part of a footprint at most.
ROOM_MESH_HEADROOM_M = 2.05
#: The headroom stays this far below the room's own top (its ceiling, or its
#: walls' top), so a low or noisy ceiling never roofs every station.
ROOM_MESH_CEILING_MARGIN_M = 0.25
ROOM_MESH_RULE = "footprint_column_clear_of_room_mesh_floor_observed_and_unroofed"
#: Points, or faces, converted at once while a collider's triangles are read (``_world_triangles``).
POINT_BLOCK = 1 << 18


def _scene_collision(plan: Mapping[str, Any], packet_dir: Path) -> tuple[Path, str, list[float]]:
    rows = [row for row in plan.get("objects") or [] if row.get("semantic_role") == "scene_collision"]
    if len(rows) != 1:
        raise TeamEmbodimentError(["team_embodiment_scene_collision_unavailable"])
    pose = rows[0].get("pose_world") if isinstance(rows[0].get("pose_world"), Mapping) else {}
    orientation = [float(v) for v in pose.get("orientation_xyzw") or []]
    if len(orientation) != 4 or not all(math.isclose(a, b, abs_tol=1e-9)
                                        for a, b in zip(orientation, [0.0, 0.0, 0.0, 1.0], strict=True)):
        raise TeamEmbodimentError(["team_embodiment_scene_collision_pose_unsupported"])
    path = Path(packet_dir).joinpath(*PurePosixPath(str(rows[0].get("usd_path") or "")).parts)
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if digest is None or digest != rows[0].get("sha256"):
        raise TeamEmbodimentError(["team_embodiment_scene_collision_unavailable"])
    return path, digest, [float(v) for v in pose["position_world_m"]]


def _transformed_points(points: Any, matrix: Any, offset: list[float]) -> Any:
    """``Gf.Matrix4d.Transform`` of every float point, plus ``offset``, as one array.

    Bit for bit what ``[float(v) + offset[axis] for axis, v in enumerate(matrix.Transform(point))]`` gives
    for each point (Python binds that call to the double overload): the row vector times the matrix
    summed in double in the same order, ``GfProject`` by the fourth component, then the offset added.
    A scene collision mesh holds millions of points, and the per-point Python loop took about a minute
    per stance search.
    """

    import numpy as np

    single = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    m = [[float(matrix[row][column]) for column in range(4)] for row in range(4)]
    shift = np.asarray(offset[:3], dtype=np.float64)
    world = np.empty((len(single), 3))
    # A block of points at a time (the same arithmetic per point): a scanned room's millions of points
    # never hold their double-precision temporaries all at once.
    for start in range(0, len(single), POINT_BLOCK):
        p = single[start:start + POINT_BLOCK].astype(np.float64)
        homogeneous = [p[:, 0] * m[0][column] + p[:, 1] * m[1][column] + p[:, 2] * m[2][column] + m[3][column]
                       for column in range(4)]
        w = homogeneous[3]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            inverse = np.where(w != 0.0, 1.0 / np.where(w != 0.0, w, 1.0), 1.0)
            projected = np.stack([inverse * homogeneous[axis] for axis in range(3)], axis=1)
        world[start:start + len(p)] = projected + shift
    return world


def _fan_triangles(counts: Any, indices: Any, first: int) -> Any:
    """Each face's fan ``(i0, ik, ik+1)``, in face order, offset by ``first``; None for negative counts.

    Built a block of faces at a time into one array, with the same values as all at once.
    """

    import numpy as np

    sizes = np.asarray(counts, dtype=np.int64).reshape(-1)
    if (sizes < 0).any():
        return None
    order = np.asarray(indices).reshape(-1)
    starts = np.cumsum(sizes) - sizes
    fans = np.maximum(sizes - 2, 0)
    fan_first = np.cumsum(fans) - fans
    triangles = np.empty((int(fans.sum()), 3), dtype=np.int64)
    for begin in range(0, len(sizes), POINT_BLOCK):
        block = fans[begin:begin + POINT_BLOCK]
        face = np.repeat(np.arange(begin, begin + len(block)), block)
        k = np.arange(int(block.sum()), dtype=np.int64) - np.repeat(np.cumsum(block) - block, block) + 1
        corner = starts[face]
        if len(corner) and int((corner + k + 1).max()) >= len(order):
            raise IndexError("face vertex index out of range")
        rows = np.stack([order[corner], order[corner + k], order[corner + k + 1]], axis=1).astype(np.int64)
        offset = int(fan_first[begin]) if len(block) else 0
        triangles[offset:offset + len(rows)] = rows + first
    return triangles


def _world_triangles(path: Path, offset: list[float], meshes: list[tuple[str, int]] | None = None, *,
                     stage: Any = None) -> tuple[Any, Any]:
    """Every collision mesh triangle in the scene's world frame (Z up, meters).

    ``meshes``, when given, receives each mesh's prim path and triangle count, in face order. ``stage``,
    when given, is ``path`` already open.
    """

    import numpy as np
    from pxr import Usd, UsdGeom, Vt

    stage = Usd.Stage.Open(readable(path)) if stage is None else stage
    if (UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or (
            stage.HasAuthoredMetadata("metersPerUnit")
            and not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0))):
        raise TeamEmbodimentError(["team_embodiment_scene_collision_frame_unsupported"])
    cache = UsdGeom.XformCache()
    vertices: list[Any] = []
    faces: list[Any] = []
    first = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points, counts = mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue
        matrix = cache.GetLocalToWorldTransform(prim)
        vertices.append(_transformed_points(points, matrix, offset) if isinstance(points, Vt.Vec3fArray)
                        else np.asarray([[float(v) + offset[axis] for axis, v in enumerate(matrix.Transform(point))]
                                         for point in points], dtype=float).reshape(-1, 3))
        triangles = _fan_triangles(counts, indices, first)
        if triangles is None:
            # Negative face sizes are not valid USD; walk them exactly as authored.
            rows, cursor = [], 0
            for count in counts:
                rows.extend((first + indices[cursor], first + indices[cursor + k], first + indices[cursor + k + 1])
                            for k in range(1, int(count) - 1))
                cursor += int(count)
            triangles = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
        faces.append(triangles)
        if meshes is not None:
            meshes.append((str(prim.GetPath()), len(triangles)))
        first += len(points)
    if not faces or not sum(len(rows) for rows in faces):
        raise TeamEmbodimentError(["team_embodiment_scene_collision_unavailable"])
    # One mesh (a scanned room) is returned as built: no concatenated or converted copy of it.
    return ((vertices[0] if len(vertices) == 1 else np.concatenate(vertices)).astype(float, copy=False),
            (faces[0] if len(faces) == 1 else np.concatenate(faces)).astype(np.int64, copy=False))


def _strikes(points: Any, *, base: Any, yaw: float, root: float, footprint: list[float],
             robot_id: str, margin_m: float) -> bool:
    """Whether moving-part samples enter the station's mount or the robot's fixed volumes on it.

    A robot with no measured fixed volumes keeps only its mount (or footprint) clear; its own body is unassessed.
    """

    import numpy as np

    from .scene_physics_preflight import ROBOT_FIXED_VOLUMES, robot_fixed_volume_extent, robot_fixed_volume_mask

    if points is None or not len(points):
        return False
    modeled = robot_id in ROBOT_FIXED_VOLUMES
    reach = max(robot_fixed_volume_extent(robot_id) if modeled else 0.0, math.hypot(*footprint)) + margin_m
    near = points[(np.abs(points[:, 0] - base[0]) <= reach) & (np.abs(points[:, 1] - base[1]) <= reach)]
    if not len(near):
        return False
    relative = near - base
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    local = np.c_[cos_yaw * relative[:, 0] + sin_yaw * relative[:, 1],
                  cos_yaw * relative[:, 1] - sin_yaw * relative[:, 0], relative[:, 2]]
    mount = ((np.abs(local[:, 0]) <= footprint[0] + margin_m) & (np.abs(local[:, 1]) <= footprint[1] + margin_m)
             & (local[:, 2] >= -root - margin_m) & (local[:, 2] <= margin_m))
    return bool(mount.any() or (modeled and robot_fixed_volume_mask(robot_id, local, margin_m).any()))


class _PlanBroadPhase:
    """Triangles whose plan bounds meet a box, indexed in numpy.

    Every triangle that meets a footprint has plan bounds that meet the footprint's, so the exact test run
    on these misses none. The external placement probe's own grid indexes walk every triangle in Python,
    which the hundreds of thousands of a scanned room near a task make slow. Here a triangle no wider than
    the cell (its widest such triangle, 2 to 25 cm) is filed under the cell of its lower plan corner, sorted
    by row then column so that a box's cells in one row are one run; a wider one is checked on its own.
    """

    ROW = 1 << 31
    MAXIMUM_CELL_M = 0.25

    def __init__(self, triangles: Any) -> None:
        import numpy as np

        self.triangles = triangles
        self.low, self.high = triangles[:, :, :2].min(axis=1), triangles[:, :, :2].max(axis=1)
        width = (self.high - self.low).max(axis=1)
        wide = width > self.MAXIMUM_CELL_M
        narrow = np.flatnonzero(~wide)
        self.cell = max(0.02, float(width[narrow].max()) if len(narrow) else 0.0)
        cells = np.floor(self.low[narrow] / self.cell).astype(np.int64)
        keys = cells[:, 0] * self.ROW + cells[:, 1]
        order = np.argsort(keys, kind="stable")
        self.narrow, self.keys, self.wide = narrow[order], keys[order], np.flatnonzero(wide)

    def __call__(self, minimum: Any, maximum: Any) -> Any:
        import numpy as np

        lower, upper = np.subtract(minimum, 1e-9), np.add(maximum, 1e-9)
        first = np.floor(lower / self.cell).astype(np.int64) - 1
        last = np.floor(upper / self.cell).astype(np.int64)
        rows = np.arange(first[0], last[0] + 1, dtype=np.int64) * self.ROW
        begin = np.searchsorted(self.keys, rows + first[1], side="left")
        count = np.searchsorted(self.keys, rows + last[1], side="right") - begin
        index = np.concatenate([self.narrow[np.repeat(begin - np.cumsum(count) + count, count)
                                            + np.arange(int(count.sum()))], self.wide])
        index = index[(self.high[index] >= lower).all(axis=1) & (self.low[index] <= upper).all(axis=1)]
        return self.triangles[index]


def _room_mesh_occupancy(vertices: Any, faces: Any, meshes: list[tuple[str, int]], *, floor_z: float,
                         tolerance: float, column_top_m: float, centre: list[float], subject_top_z: float,
                         reach_m: float, measures: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """The collision meshes that are the room itself, and their geometry around the search; None for none.

    The obstacle-box index (``UsdSceneSpatialIndex.obstacle_boxes``) sees a scanned room as one box that is
    the whole room: excluded by a shell name, dropped past its size cap, or, kept, covering every stance. It
    never sees the table in it. A mesh is the room when it holds at least ``ROOM_MESH_FLOOR_SHARE`` of the
    floor band's area, rises above the floor, and surrounds the task: its bounds span the subject's
    ``centre`` in plan and rise above the subject's top (a fixture the object rests on ends below it). Its
    triangles that rise above the floor and start no higher than ``column_top_m`` above it, near the search
    (``reach_m`` around ``centre``), are the obstacles a footprint's column is measured against; the scene's
    triangles at floor height there are the floor it must stand on (``external_scene_robot_placement``'s
    probe and support sampler). Its triangles there whose surface reaches between the column top and the
    headroom (``ROOM_MESH_HEADROOM_M``, kept ``ROOM_MESH_CEILING_MARGIN_M`` below the room's top: the height
    under which 99% of its non-floor area ends, so a floater above a low ceiling cannot lift it) are the
    roofs a footprint must not stand wholly under, measured at each point beneath them.

    Every triangle is measured in blocks (``triangle_measures``, or ``measures`` already taken of these
    ones) and only the ones kept are gathered, so a scanned room's millions of triangles are never held
    as corners at once: the same values, in bounded memory.
    """

    import numpy as np

    from .external_scene_robot_placement import TRIANGLE_BLOCK, triangle_measures, triangles_of

    measures = triangle_measures(vertices, faces) if measures is None else measures
    double_area, valid, normal_z = measures["double_area"], measures["valid"], measures["normal_z"]
    lowest, highest = measures["lowest_z"], measures["highest_z"]
    # The floor band exactly as the external placement probe excludes it from its obstacles.
    floor = valid & (normal_z >= 0.9) & (np.abs(measures["centroid_z"] - floor_z) <= tolerance)
    rising = ~floor & (highest >= floor_z + ROOM_MESH_FLOOR_CLEARANCE_M)
    owner = np.repeat(np.arange(len(meshes)), [count for _, count in meshes])
    area = np.bincount(owner[floor], weights=double_area[floor], minlength=len(meshes))
    rises = np.bincount(owner[rising], minlength=len(meshes)) > 0
    room = rises & (area >= ROOM_MESH_FLOOR_SHARE * area.sum()) & (area > 0.0)
    first = np.cumsum([count for _, count in meshes]) - [count for _, count in meshes]
    for index in np.flatnonzero(room):
        low, high = np.full(3, np.inf), np.full(3, -np.inf)
        for start in range(first[index], first[index] + meshes[index][1], TRIANGLE_BLOCK):
            points = triangles_of(vertices, faces, slice(start, min(start + TRIANGLE_BLOCK,
                                                                     first[index] + meshes[index][1]))).reshape(-1, 3)
            low, high = np.minimum(low, points.min(axis=0)), np.maximum(high, points.max(axis=0))
        room[index] = bool((low[:2] <= centre[:2]).all() and (high[:2] >= centre[:2]).all()
                           and high[2] > subject_top_z)
    if not room.any():
        return None

    def near(mask: Any) -> Any:
        # The masked triangles whose plan bounds reach the search, gathered a block at a time.
        index = np.flatnonzero(mask)
        kept = np.zeros(len(index), dtype=bool)
        for start in range(0, len(index), TRIANGLE_BLOCK):
            rows = triangles_of(vertices, faces, index[start:start + TRIANGLE_BLOCK])
            low, high = rows[:, :, :2].min(axis=1), rows[:, :, :2].max(axis=1)
            kept[start:start + TRIANGLE_BLOCK] = ((high >= np.subtract(centre, reach_m)).all(axis=1)
                                                  & (low <= np.add(centre, reach_m)).all(axis=1))
        mask = mask.copy()
        mask[index] = kept
        return mask

    column = near(room[owner] & rising & (lowest <= floor_z + column_top_m))
    # The floor observed under a footprint: the floor band, and any triangle lying wholly within the floor's
    # height band (a scanned floor's noise tilts its small triangles past the horizontal test).
    ground = near(floor | ((lowest >= floor_z - tolerance) & (highest <= floor_z + tolerance)))
    upper = room[owner] & ~floor & valid
    order = np.argsort(highest[upper], kind="stable")
    ends, weights = highest[upper][order], np.cumsum(double_area[upper][order])
    room_top = float(ends[min(int(np.searchsorted(weights, 0.99 * weights[-1])), len(ends) - 1)]) - floor_z
    roof_top = min(ROOM_MESH_HEADROOM_M, room_top - ROOM_MESH_CEILING_MARGIN_M)
    # Plan-degenerate (vertical) triangles cover no point; a roof is what a footprint stands under. A triangle
    # reaching into the band is kept whole and measured at each point (``_roofed``): a sloped ceiling roofs
    # only where its surface is below the headroom, however it is tessellated.
    roof = near(room[owner] & rising & (np.abs(measures["cross_z"]) > 1e-12) & (highest > floor_z + column_top_m)
                & (lowest <= floor_z + roof_top))
    return {"prims": [meshes[index][0] for index in np.flatnonzero(room)],
            "obstacles": _PlanBroadPhase(np.ascontiguousarray(triangles_of(vertices, faces, column)[:, :, :2])),
            "obstacle_triangles": int(column.sum()),
            "floor": _PlanBroadPhase(triangles_of(vertices, faces, ground)), "floor_triangles": int(ground.sum()),
            "roof": _PlanBroadPhase(triangles_of(vertices, faces, roof)) if roof.any() else None,
            "roof_band_z": (floor_z + column_top_m, floor_z + roof_top),
            "roof_triangles": int(roof.sum()), "roof_top_above_floor_m": float(roof_top)}


def _roofed(roof: Any, band: tuple[float, float], point: tuple[float, float]) -> bool:
    """Whether a roof triangle's surface lies over ``point`` (its edges included) within the ``band`` heights.

    The band is open below (the column top: lower, a triangle crossing the footprint already refused it)
    and closed above (the headroom).
    """

    rows = roof(point, point)
    if not len(rows):
        return False
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = (rows[:, corner].T for corner in range(3))
    px, py = point
    determinant = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    first = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / determinant
    second = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / determinant
    third = 1.0 - first - second
    over = (first >= -1e-9) & (second >= -1e-9) & (third >= -1e-9)
    height = first * az + second * bz + third * cz
    return bool((over & (height > band[0]) & (height <= band[1])).any())


def _room_mesh_refusal(room: Mapping[str, Any], *, x: float, y: float, yaw: float, footprint: list[float],
                       clearance_m: float, floor_z: float, tolerance: float) -> str | None:
    """Why a room mesh leaves no space for this stance, or None.

    ``occupied``: a room-mesh triangle crosses the footprint's column, the footprint grown by the stand
    validator's obstacle clearance (``_triangle_footprint_overlap_count``, conservative). ``floor_unobserved``:
    the footprint's centre or a corner has no floor under it, as inside furniture a scan never saw beneath
    (``_footprint_support_report``). ``roofed``: a room surface below the headroom lies over the centre and
    every corner, as inside closed furniture whose floor the reconstruction rebuilt; a lamp or a shelf over
    part of the footprint is the validator's overhead clutter.
    """

    import numpy as np

    from .external_scene_robot_placement import _footprint_support_report, _triangle_footprint_overlap_count

    cosine, sine = math.cos(yaw), math.sin(yaw)
    grown = (float(footprint[0]) + clearance_m, float(footprint[1]) + clearance_m)
    half = (abs(cosine) * grown[0] + abs(sine) * grown[1], abs(sine) * grown[0] + abs(cosine) * grown[1])
    if _triangle_footprint_overlap_count(obstacle_triangles_xy=room["obstacles"]((x - half[0], y - half[1]),
                                                                                 (x + half[0], y + half[1])),
                                         position=(x, y), yaw=yaw, half_extent_xy=grown):
        return "occupied"
    # The sampler's own points (centre and four corners); only triangles whose bounds hold one can support it.
    hx, hy = float(footprint[0]), float(footprint[1])
    points = [(x + cosine * u - sine * v, y + sine * u + cosine * v)
              for u, v in ((0.0, 0.0), (hx, hy), (hx, -hy), (-hx, hy), (-hx, -hy))]
    nearby = np.concatenate([room["floor"](point, point) for point in points])
    support = _footprint_support_report(support_triangles=nearby, position=(x, y), yaw=yaw, half_extent_xy=(hx, hy),
                                        floor_z=floor_z, height_tolerance=tolerance)
    if not support["full_sample_support_candidate"]:
        return "floor_unobserved"
    roof = room["roof"]
    return ("roofed" if roof is not None and all(_roofed(roof, room["roof_band_z"], point) for point in points)
            else None)


def _stage_without(path: Path, prims: list[str]) -> Any:
    """The collision stage with ``prims`` deactivated in its session layer (the file is unchanged)."""

    from pxr import Usd

    stage = Usd.Stage.Open(readable(path))
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in prims:
            stage.GetPrimAtPath(prim).SetActive(False)
    return stage


@read_scoped
def ground_stance(*, collision_path: Path, collision_sha256: str, offset: list[float],
                  subject_center: list[float], subject_minimum: list[float], subject_maximum: list[float],
                  footprint_half_extent_m: list[float], root_height_m: float | None = None,
                  root_world_z_m: float | None = None, task_motion: Mapping[str, Any] | None = None,
                  ranked: int = 0, retain_large_collision_bounds: bool = False) -> Any:
    """The closest clear floor stance facing the subject, or raise by name (no GPU).

    The root stands ``root_height_m`` above the floor, or at ``root_world_z_m``.
    ``task_motion`` holds world samples of a moving task part, for ``robot_id``'s
    fixed volumes on the station: ``path_points`` (reset to the success
    threshold) and ``fall_points`` (where gravity carries it) block a stance, and
    a stance ``limit_points`` (its whole range) reaches ranks after the rest.
    With ``ranked``, the ``ranked`` best clear stances are returned in order instead.
    ``retain_large_collision_bounds`` keeps long retained collision fixtures in
    the footprint check; the default preserves existing fixed-arm behavior.
    A room scanned as one mesh is left out of the box index and measured by its
    own triangles (``_room_mesh_occupancy``); with no stance clear of it, the
    search refuses by name (``team_embodiment_room_mesh_no_clear_stance``).
    """

    from pxr import Usd

    from .external_scene_robot_placement import (
        ExternalSceneRobotPlacementError,
        _infer_horizontal_support_surface,
        triangle_measures,
    )
    from .scene_placement.types import SceneObject
    from .scene_placement.usd_index import UsdSceneSpatialIndex
    from .scene_placement.validation import DEFAULT_VALIDATION_MIN_OBSTACLE_CLEARANCE_M, validate_stand_pose

    # Every read below (the triangles, ``_stage_without``, the box index) opens the scope's binary copy of a
    # USDA text collider, read once (``usd_binary_read_cache``); the stance names the file by its digest.
    # Held to the end of the search, so a file no copy is made of is parsed once too.
    opened = Path(readable(collision_path))
    stage = Usd.Stage.Open(str(opened))
    meshes: list[tuple[str, int]] = []
    vertices, faces = _world_triangles(opened, offset, meshes, stage=stage)
    # Measured once, in blocks, for the floor and the room mesh alike (``triangle_measures``).
    measures = triangle_measures(vertices, faces)
    try:
        floor_z, _, floor_fit = _infer_horizontal_support_surface(stage_vertices=vertices, faces=faces,
                                                                  measures=measures, return_support_triangles=False)
    except ExternalSceneRobotPlacementError as exc:
        raise TeamEmbodimentError(["team_embodiment_scene_floor_not_found"]) from exc
    root = float(root_height_m) if root_height_m is not None else float(root_world_z_m) - float(floor_z)
    if root_height_m is None and -FLOOR_MOUNT_TOLERANCE_M <= root <= 0.0:
        root = 0.0
    if not math.isfinite(root) or root < 0.0 or (root_height_m is not None and root <= 0.0):
        raise TeamEmbodimentError(["team_embodiment_stance_height_invalid"])
    def shifted(point: Any) -> tuple[float, ...]:
        return tuple(float(v) + offset[axis] for axis, v in enumerate(point))

    start = [float(v) for v in subject_center[:3]]
    clearance_m = float(DEFAULT_VALIDATION_MIN_OBSTACLE_CLEARANCE_M)
    tolerance = float(floor_fit["height_tolerance_stage_units"])
    # A free base's root, or a mount level with the lower of two supports, sits below the task's table:
    # the column still reaches the task's top, so no station stands under that table.
    column_top = max(root, ROOM_MESH_MINIMUM_COLUMN_M, float(subject_maximum[2]) - float(floor_z))
    room = _room_mesh_occupancy(
        vertices, faces, meshes, floor_z=float(floor_z), tolerance=tolerance,
        column_top_m=column_top, centre=start[:2], subject_top_z=float(subject_maximum[2]),
        reach_m=SEARCH_RADIUS_M + math.hypot(*(float(v) + clearance_m for v in footprint_half_extent_m[:2]))
        + GRID_STEP_M, measures=measures)
    del measures
    source = ({"usd_path": str(opened)} if room is None else {"stage": _stage_without(opened, room["prims"])})
    obstacles = [replace(box, bbox_min=shifted(box.bbox_min), bbox_max=shifted(box.bbox_max),
                         centroid=shifted(box.centroid))
                 for box in UsdSceneSpatialIndex(**source,
                     **({"max_obstacle_box_size": math.inf} if retain_large_collision_bounds else {})).obstacle_boxes()]
    subject = SceneObject(id="task_object", label="task object", centroid=tuple(start),
                          bbox_min=tuple(float(v) for v in subject_minimum),
                          bbox_max=tuple(float(v) for v in subject_maximum))
    footprint = (*(float(v) for v in footprint_half_extent_m), root)
    motion = dict(task_motion or {})
    if motion:
        import numpy as np

        from .task_evaluation_robot_placement_geometry import TASK_MOTION_CLEARANCE_M

        robot_id = str(motion.get("robot_id") or "")
        clearance = {key: np.asarray(motion.get(key, np.zeros((0, 3))), dtype=float).reshape(-1, 3)
                     for key in ("path_points", "fall_points", "limit_points")}
    steps, best, clear, struck = math.floor(SEARCH_RADIUS_M / GRID_STEP_M + 1e-9), None, 0, 0
    refused = {"occupied": 0, "floor_unobserved": 0, "roofed": 0}
    candidates: list[Any] = []
    for ix in range(-steps, steps + 1):
        for iy in range(-steps, steps + 1):
            x, y = start[0] + ix * GRID_STEP_M, start[1] + iy * GRID_STEP_M
            distance = math.hypot(x - start[0], y - start[1])
            if distance > SEARCH_RADIUS_M:
                continue
            yaw = math.atan2(start[1] - y, start[0] - x)
            if not validate_stand_pose((x, y, floor_z + root), yaw, subject, obstacles, floor_z,
                                       footprint_half_extent=footprint, pelvis_height=root).ok:
                continue
            refusal = None if room is None else _room_mesh_refusal(
                room, x=x, y=y, yaw=yaw, footprint=list(footprint[:2]), clearance_m=clearance_m,
                floor_z=float(floor_z), tolerance=tolerance)
            if refusal is not None:
                refused[refusal] += 1
                continue
            reached = False
            if motion:
                # The path takes the sampling margin; a fall or the whole range counts actual overlap.
                place = {"base": np.asarray([x, y, floor_z + root]), "yaw": yaw, "root": root,
                         "footprint": list(footprint[:2]), "robot_id": robot_id}
                if (_strikes(clearance["path_points"], margin_m=TASK_MOTION_CLEARANCE_M, **place)
                        or _strikes(clearance["fall_points"], margin_m=0.0, **place)):
                    struck += 1
                    continue
                reached = _strikes(clearance["limit_points"], margin_m=0.0, **place)
            clear += 1
            candidates.append(((reached, distance, ix, iy), x, y, yaw))
            if best is None or (reached, distance, ix, iy) < best[0]:
                best = ((reached, distance, ix, iy), x, y, yaw)
    if best is None:
        # Never a stance inside the room's own furniture: no clear space is a refusal by name.
        blockers = (["team_embodiment_task_motion_no_clear_stance"] if struck else []) + (
            ["team_embodiment_room_mesh_no_clear_stance"] if any(refused.values()) else [])
        raise TeamEmbodimentError(blockers or ["team_embodiment_free_base_no_clear_stance"])
    measured = None if room is None else {
        "rule": ROOM_MESH_RULE, "room_mesh_prims": list(room["prims"]),
        "column_top_above_floor_m": float(column_top), "floor_clearance_m": ROOM_MESH_FLOOR_CLEARANCE_M,
        "obstacle_clearance_m": clearance_m, "obstacle_triangles": room["obstacle_triangles"],
        "floor_triangles": room["floor_triangles"], "occupied_stances": refused["occupied"],
        "floor_unobserved_stances": refused["floor_unobserved"],
        "roof_top_above_floor_m": room["roof_top_above_floor_m"], "roof_triangles": room["roof_triangles"],
        "roofed_stances": refused["roofed"]}
    if ranked:
        return [_stance(row, floor_z=floor_z, root=root, footprint_half_extent_m=footprint_half_extent_m, clear=clear,
                        struck=struck, motion=motion, robot_id=robot_id if motion else None,
                        collision_sha256=collision_sha256, room=measured) for row in sorted(candidates)[:ranked]]
    return _stance(best, floor_z=floor_z, root=root, footprint_half_extent_m=footprint_half_extent_m, clear=clear,
                   struck=struck, motion=motion, robot_id=robot_id if motion else None,
                   collision_sha256=collision_sha256, room=measured)


def _stance(row: Any, *, floor_z: float, root: float, footprint_half_extent_m: list[float], clear: int, struck: int,
            motion: Mapping[str, Any], robot_id: str | None, collision_sha256: str,
            room: Mapping[str, Any] | None = None) -> dict[str, Any]:
    (reached, *_), x, y, yaw = row
    stance = {"schema_version": SCHEMA, "floor_z_m": float(floor_z),
              "position_world_m": [x, y, float(floor_z) + root],
              "orientation_xyzw": [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)], "yaw_rad": yaw,
              "footprint_half_extent_m": [float(v) for v in footprint_half_extent_m],
              "clear_stances": clear, "search_radius_m": SEARCH_RADIUS_M, "grid_step_m": GRID_STEP_M,
              **({"task_motion_clearance": {"robot_id": robot_id, "struck_stances": struck,
                                            "reached_beyond_task_range": reached}} if motion else {}),
              **({"room_mesh_clearance": dict(room)} if room is not None else {}),
              "scene_collision_sha256": collision_sha256,
              "validator": "scene_placement.validation.validate_stand_pose", "stance_digest": ""}
    stance["stance_digest"] = canonical_digest(stance, digest_field="stance_digest")
    return stance


def free_base_stance(manifest: Mapping[str, Any], *, scene_plan: Mapping[str, Any],
                     packet_dir: str | Path) -> dict[str, Any]:
    """The closest clear floor stance facing the task object, or raise by name (no GPU)."""

    base = manifest["base"]
    task = scene_plan.get("task_spec") if isinstance(scene_plan.get("task_spec"), Mapping) else {}
    start, bounds = task.get("start_pose_world"), task.get("subject_collision_bounds_scoring_frame_m")
    if (not isinstance(start, list) or len(start) < 3 or not isinstance(bounds, Mapping)
            or len(bounds.get("minimum") or []) != 3 or len(bounds.get("maximum") or []) != 3):
        raise TeamEmbodimentError(["team_embodiment_task_bounds_unavailable"])
    collision, digest, offset = _scene_collision(scene_plan, Path(packet_dir))
    return ground_stance(
        collision_path=collision, collision_sha256=digest, offset=offset, subject_center=list(start[:3]),
        subject_minimum=[float(start[i]) + float(bounds["minimum"][i]) for i in range(3)],
        subject_maximum=[float(start[i]) + float(bounds["maximum"][i]) for i in range(3)],
        footprint_half_extent_m=list(base["footprint_half_extent_m"]), root_height_m=float(base["root_height_m"]))


__all__ = ["SCHEMA", "free_base_stance", "ground_stance"]
