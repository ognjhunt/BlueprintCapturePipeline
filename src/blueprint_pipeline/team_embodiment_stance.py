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
A fixed arm's pedestal (``PEDESTAL_SEARCH``) is an authored box, not a scanned
body: it keeps the arm's own scene margin from the room and the task object
(``PEDESTAL_CLEARANCE_M``), never the humanoid's 8 cm clearance and 0.4-1.2 m
standoff the stand validator defaults to (capture-447: that standoff alone
kept its pedestal 0.75 m or more from the tray's centre). The standoff is
measured from the box the object sweeps from its start to its target, so a
pedestal never stands on the place. A search given the task's points
(``task_points_world_m``: a carry's start and target) ranks stances by the
farther of them, and its first ranked stances stand at least a footprint
apart (``spread_ranked``), so a carry tried from the first few is tried from
different spots, never from eight neighbouring cells on one side of the
table (capture-447's eight, all 1.02-1.29 m from the place). Every search records the refused stances that ranked ahead of
the chosen one and what refused each (``nearer_refusals``): the validator's
reason, or the room-mesh triangles in the way, with their heights against
the mount and, just ahead of the chosen stance, their bounds.
Named movable obstacles (``movable_obstacles``: a chair's measured bounds)
are never removed here. For each, the search reports the stances that would
be clear were it moved (``movable_obstacle_counterfactual``); it never
chooses one, since the scene the robot runs in still holds the obstacle.
Host-side and deterministic (OpenUSD), so a robot with no clear stance is
refused by name before any GPU time is spent.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
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
#: An authored pedestal keeps from the scan, and from the task object, what the arm's own links keep from
#: the scan along the carry (``robot_arm_path_clearance.DEFAULT_MARGIN_M``). The stand validator's 8 cm
#: covers a scanned humanoid mesh rendering into a cabinet; a box the pipeline authors has no such mesh.
PEDESTAL_CLEARANCE_M = 0.02
#: A fixed arm's station search: the pedestal's clearance from every obstacle, the room mesh and the task
#: object alike (the standoff's lower end), and no upper end inside the search disc. The validator's own
#: standoff, 0.4-1.2 m from the task object, is a humanoid's arm span; a free base keeps it. That standoff
#: was also all that kept a pedestal off a support the box index drops past its 6 m size cap (a long
#: counter run), so the pedestal keeps long fixtures in its footprint check, as the G1's search does.
PEDESTAL_SEARCH = {"clearance_m": PEDESTAL_CLEARANCE_M, "standoff_range_m": (PEDESTAL_CLEARANCE_M, SEARCH_RADIUS_M),
                   "retain_large_collision_bounds": True}
RANK_BY_SUBJECT = "closest_to_subject_centre"
RANK_BY_TASK_POINTS = "farthest_task_point_from_root_smallest"
SPREAD_RULE = "ranked_stances_a_footprint_apart_best_first"
#: Refused stances that ranked just ahead of the chosen one, recorded with what refused each.
NEARER_REFUSALS = 8
#: A blocker whose top is within this of the station's root stands at the mount's height (a table's top
#: beside a mount level with it); lower is under it (a seat), higher over it (a chair's back, a wall).
BLOCKER_MOUNT_BAND_M = 0.05
MOVABLE_RULE = "clear_if_named_obstacles_moved_never_chosen"
#: Counterfactual stances recorded, best first.
MOVABLE_STANCES = 8
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
    # Kept with their heights: a refusal names the triangles in the way, and a named movable obstacle's
    # bounds are measured against them (``_room_mesh_blockers``).
    return {"prims": [meshes[index][0] for index in np.flatnonzero(room)],
            "obstacles": _PlanBroadPhase(np.ascontiguousarray(triangles_of(vertices, faces, column))),
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


def _room_mesh_blockers(room: Mapping[str, Any], *, x: float, y: float, yaw: float, footprint: list[float],
                        clearance_m: float) -> Any:
    """The room-mesh triangles (world corners) crossing the footprint's column, grown by ``clearance_m``.

    The external placement probe's conservative test (``_triangle_footprint_overlap_count``): a triangle
    whose bounds in the footprint's own frame meet the grown footprint.
    """

    cosine, sine = math.cos(yaw), math.sin(yaw)
    grown = (float(footprint[0]) + clearance_m, float(footprint[1]) + clearance_m)
    half = (abs(cosine) * grown[0] + abs(sine) * grown[1], abs(sine) * grown[0] + abs(cosine) * grown[1])
    rows = room["obstacles"]((x - half[0], y - half[1]), (x + half[0], y + half[1]))
    if not len(rows):
        return rows
    dx, dy = rows[:, :, 0] - x, rows[:, :, 1] - y
    local_x, local_y = cosine * dx + sine * dy, -sine * dx + cosine * dy
    return rows[(local_x.max(axis=1) >= -grown[0]) & (local_x.min(axis=1) <= grown[0])
                & (local_y.max(axis=1) >= -grown[1]) & (local_y.min(axis=1) <= grown[1])]


def _room_mesh_refusal(room: Mapping[str, Any], *, x: float, y: float, yaw: float, footprint: list[float],
                       clearance_m: float, floor_z: float, tolerance: float, moved: Any = None,
                       blockers: Any = None) -> str | None:
    """Why a room mesh leaves no space for this stance, or None.

    ``occupied``: a room-mesh triangle crosses the footprint's column, the footprint grown by the stand
    validator's obstacle clearance (``_room_mesh_blockers``, conservative). ``floor_unobserved``:
    the footprint's centre or a corner has no floor under it, as inside furniture a scan never saw beneath
    (``_footprint_support_report``). ``roofed``: a room surface below the headroom lies over the centre and
    every corner, as inside closed furniture whose floor the reconstruction rebuilt; a lamp or a shelf over
    part of the footprint is the validator's overhead clutter. ``moved``, when given, marks the blocking
    triangles a counterfactual leaves out (``_movable_obstacles``); every other check stands. ``blockers``
    are this stance's ``_room_mesh_blockers`` when the caller already has them.
    """

    import numpy as np

    from .external_scene_robot_placement import _footprint_support_report

    if blockers is None:
        blockers = _room_mesh_blockers(room, x=x, y=y, yaw=yaw, footprint=footprint, clearance_m=clearance_m)
    if moved is not None and len(blockers):
        blockers = blockers[~moved(blockers)]
    if len(blockers):
        return "occupied"
    cosine, sine = math.cos(yaw), math.sin(yaw)
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


def _movable_obstacles(rows: Any) -> tuple[list[dict[str, Any]], int]:
    """The named movable obstacles a search may report counterfactuals for, and how many were unusable.

    Each is ``{"obstacle_id", "label", "bounds_world_m": {"minimum", "maximum"}, "basis"}``: a named piece
    of furniture's measured world bounds and where they come from. An entry without an id or with
    non-finite or empty bounds is left out and counted; it never refuses the search.
    """

    kept, ignored = [], 0
    for row in rows or ():
        try:
            bounds = row["bounds_world_m"]
            low, high = [float(v) for v in bounds["minimum"]], [float(v) for v in bounds["maximum"]]
            name = str(row["obstacle_id"]).strip()
            valid = (name and len(low) == len(high) == 3 and all(map(math.isfinite, low + high))
                     and all(a < b for a, b in zip(low, high, strict=True)))
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            ignored += 1
            continue
        kept.append({"obstacle_id": name, "label": str(row.get("label") or ""),
                     "bounds_world_m": {"minimum": low, "maximum": high}, "basis": str(row.get("basis") or "")})
    return kept, ignored


def _inside(named: list[dict[str, Any]], low: Any, high: Any) -> Any:
    """For each box ``low``-``high`` (rows of world corners), the index of a named obstacle holding it, or -1."""

    import numpy as np

    which = np.full(len(low), -1, dtype=np.int64)
    for index, row in reversed(list(enumerate(named))):
        bounds = row["bounds_world_m"]
        held = (low >= np.asarray(bounds["minimum"])).all(axis=1) & (high <= np.asarray(bounds["maximum"])).all(axis=1)
        which[held] = index
    return which


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
                  ranked: int = 0, retain_large_collision_bounds: bool = False,
                  clearance_m: float | None = None, standoff_range_m: Sequence[float] | None = None,
                  task_points_world_m: Sequence[Sequence[float]] | None = None,
                  movable_obstacles: Sequence[Mapping[str, Any]] | None = None) -> Any:
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
    ``clearance_m`` (the footprint's clearance from every obstacle) and
    ``standoff_range_m`` (its gap to the subject) default to the stand
    validator's; a fixed arm's pedestal passes ``PEDESTAL_SEARCH``.
    ``task_points_world_m`` ranks by the farther of them from the root instead of
    by the distance to ``subject_center``, and spreads the ranked stances a
    footprint apart, best first (``spread_ranked``). ``movable_obstacles`` only
    adds a counterfactual to the record (``_movable_obstacles``).
    """

    import numpy as np
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
    clearance = float(DEFAULT_VALIDATION_MIN_OBSTACLE_CLEARANCE_M if clearance_m is None else clearance_m)
    standoff = None if standoff_range_m is None else (float(standoff_range_m[0]), float(standoff_range_m[1]))
    if not math.isfinite(clearance) or clearance < 0.0 or (standoff is not None and not standoff[0] <= standoff[1]):
        raise TeamEmbodimentError(["team_embodiment_stance_clearance_invalid"])
    points = None if not task_points_world_m else np.asarray(
        [[float(v) for v in point[:3]] for point in task_points_world_m], dtype=float)
    named, ignored = _movable_obstacles(movable_obstacles)
    tolerance = float(floor_fit["height_tolerance_stage_units"])
    # A free base's root, or a mount level with the lower of two supports, sits below the task's table:
    # the column still reaches the task's top, so no station stands under that table.
    column_top = max(root, ROOM_MESH_MINIMUM_COLUMN_M, float(subject_maximum[2]) - float(floor_z))
    room = _room_mesh_occupancy(
        vertices, faces, meshes, floor_z=float(floor_z), tolerance=tolerance,
        column_top_m=column_top, centre=start[:2], subject_top_z=float(subject_maximum[2]),
        reach_m=SEARCH_RADIUS_M + math.hypot(*(float(v) + clearance for v in footprint_half_extent_m[:2]))
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
    # A subject that is the object at the first task point is carried through the others: the stance keeps its
    # standoff from the whole swept box, so a pedestal never stands on the place or across the carry between.
    # A subject spanning the task (an object ending on another surface) already holds it.
    swept = []
    if points is not None and len(points) > 1 and math.dist(points[0], start) <= 1e-6:
        shifts = points - points[0]
        swept = [SceneObject(id="task_object_swept", label="task object at every task point", centroid=tuple(start),
                             bbox_min=tuple(float(v) for v in np.min(np.add(subject_minimum, shifts), axis=0)),
                             bbox_max=tuple(float(v) for v in np.max(np.add(subject_maximum, shifts), axis=0)))]
    validator = {"footprint_half_extent": footprint, "pelvis_height": root,
                 **({} if clearance_m is None else {"min_obstacle_clearance_m": clearance}),
                 **({} if standoff is None else {"standoff_range": standoff}),
                 **({"standoff_obstacles": swept} if swept else {})}
    # A box obstacle inside a named movable obstacle's bounds, by index (the counterfactual leaves it out).
    box_owner = (_inside(named, np.asarray([box.bbox_min for box in obstacles], dtype=float).reshape(-1, 3),
                         np.asarray([box.bbox_max for box in obstacles], dtype=float).reshape(-1, 3))
                 if named else None)
    motion = dict(task_motion or {})
    if motion:
        from .task_evaluation_robot_placement_geometry import TASK_MOTION_CLEARANCE_M

        robot_id = str(motion.get("robot_id") or "")
        clearance_points = {key: np.asarray(motion.get(key, np.zeros((0, 3))), dtype=float).reshape(-1, 3)
                            for key in ("path_points", "fall_points", "limit_points")}

    def struck_by_motion(x: float, y: float, yaw: float) -> tuple[bool, bool]:
        # Whether the moving part strikes this station, and whether its whole range reaches it.
        if not motion:
            return False, False
        # The path takes the sampling margin; a fall or the whole range counts actual overlap.
        place = {"base": np.asarray([x, y, floor_z + root]), "yaw": yaw, "root": root,
                 "footprint": list(footprint[:2]), "robot_id": robot_id}
        if (_strikes(clearance_points["path_points"], margin_m=TASK_MOTION_CLEARANCE_M, **place)
                or _strikes(clearance_points["fall_points"], margin_m=0.0, **place)):
            return True, False
        return False, _strikes(clearance_points["limit_points"], margin_m=0.0, **place)

    def moved_set(x: float, y: float, yaw: float, verdict: Any) -> list[str] | None:
        # The named obstacles whose moving would clear this refused stance, or None (``_movable_obstacles``).
        ids: set[int] = set()
        if verdict is not None:
            failed = {code.split(":", 1)[0] for code in verdict.failures}
            if not failed <= {"clips", "clearance"}:
                return None
            blocking = {oid for oid, _ in [*verdict.clipping, *verdict.near_clearance]}
            owners = [int(box_owner[i]) for i, box in enumerate(obstacles) if box.id in blocking]
            if not owners or min(owners) < 0:
                return None
            ids.update(owners)
            kept = [box for i, box in enumerate(obstacles) if box_owner[i] < 0]
            if not validate_stand_pose((x, y, floor_z + root), yaw, subject, kept, floor_z, **validator).ok:
                return None
        if room is not None:
            rows = _room_mesh_blockers(room, x=x, y=y, yaw=yaw, footprint=list(footprint[:2]), clearance_m=clearance)
            owners_of_rows = _inside(named, rows.min(axis=1), rows.max(axis=1)) if len(rows) else np.zeros(0, int)
            ids.update(int(v) for v in owners_of_rows[owners_of_rows >= 0])
            if _room_mesh_refusal(room, x=x, y=y, yaw=yaw, footprint=list(footprint[:2]), clearance_m=clearance,
                                  floor_z=float(floor_z), tolerance=tolerance,
                                  moved=lambda blockers: _inside(named, blockers.min(axis=1),
                                                                 blockers.max(axis=1)) >= 0) is not None:
                return None
        if not ids or struck_by_motion(x, y, yaw)[0]:
            return None
        return [named[i]["obstacle_id"] for i in sorted(ids)]

    steps, clear, struck = math.floor(SEARCH_RADIUS_M / GRID_STEP_M + 1e-9), 0, 0
    refused = {"occupied": 0, "floor_unobserved": 0, "roofed": 0}
    candidates: list[Any] = []
    # Each refusal: its rank key, its reason, where it stood, and the validator's verdict or what was in the way.
    refusals: list[tuple[Any, str, float, float, float, Any]] = []
    counterfactual: list[tuple[Any, float, float, list[str]]] = []
    for ix in range(-steps, steps + 1):
        for iy in range(-steps, steps + 1):
            x, y = start[0] + ix * GRID_STEP_M, start[1] + iy * GRID_STEP_M
            distance = math.hypot(x - start[0], y - start[1])
            if distance > SEARCH_RADIUS_M:
                continue
            yaw = math.atan2(start[1] - y, start[0] - x)
            score = distance if points is None else float(
                np.sqrt(((points - [x, y, float(floor_z) + root]) ** 2).sum(axis=1)).max())
            key = (score, distance, ix, iy)
            verdict = validate_stand_pose((x, y, floor_z + root), yaw, subject, obstacles, floor_z, **validator)
            if not verdict.ok:
                refusals.append((key, verdict.failures[0].split("(", 1)[0].split(":", 1)[0], x, y, yaw, verdict))
                if named and (moved := moved_set(x, y, yaw, verdict)) is not None:
                    counterfactual.append((key, x, y, moved))
                continue
            blockers = None if room is None else _room_mesh_blockers(
                room, x=x, y=y, yaw=yaw, footprint=list(footprint[:2]), clearance_m=clearance)
            refusal = None if room is None else _room_mesh_refusal(
                room, x=x, y=y, yaw=yaw, footprint=list(footprint[:2]), clearance_m=clearance,
                floor_z=float(floor_z), tolerance=tolerance, blockers=blockers)
            if refusal is not None:
                refused[refusal] += 1
                # What was in the way, kept as its bounds only: a scanned room puts thousands of triangles there.
                refusals.append((key, refusal, x, y, yaw, (len(blockers), blockers.reshape(-1, 3).min(axis=0),
                                                           blockers.reshape(-1, 3).max(axis=0))
                                 if refusal == "occupied" else None))
                if named and refusal == "occupied" and (moved := moved_set(x, y, yaw, None)) is not None:
                    counterfactual.append((key, x, y, moved))
                continue
            hit, reached = struck_by_motion(x, y, yaw)
            if hit:
                struck += 1
                refusals.append((key, "task_motion_strike", x, y, yaw, None))
                continue
            clear += 1
            candidates.append(((reached, *key), x, y, yaw))
    if not candidates:
        # Never a stance inside the room's own furniture: no clear space is a refusal by name.
        blockers = (["team_embodiment_task_motion_no_clear_stance"] if struck else []) + (
            ["team_embodiment_room_mesh_no_clear_stance"] if any(refused.values()) else [])
        raise TeamEmbodimentError(blockers or ["team_embodiment_free_base_no_clear_stance"])
    candidates.sort()
    best = candidates[0]
    measured = None if room is None else {
        "rule": ROOM_MESH_RULE, "room_mesh_prims": list(room["prims"]),
        "column_top_above_floor_m": float(column_top), "floor_clearance_m": ROOM_MESH_FLOOR_CLEARANCE_M,
        "obstacle_clearance_m": clearance, "obstacle_triangles": room["obstacle_triangles"],
        "floor_triangles": room["floor_triangles"], "occupied_stances": refused["occupied"],
        "floor_unobserved_stances": refused["floor_unobserved"],
        "roof_top_above_floor_m": room["roof_top_above_floor_m"], "roof_triangles": room["roof_triangles"],
        "roofed_stances": refused["roofed"]}
    search = {"obstacle_clearance_m": clearance,
              "standoff_range_m": None if standoff is None else list(standoff),
              "ranking": ({"rule": RANK_BY_SUBJECT} if points is None else
                          {"rule": RANK_BY_TASK_POINTS, "task_points_world_m": points.tolist(),
                           "spread": {"rule": SPREAD_RULE, "separation_m": _separation(footprint_half_extent_m)}}),
              **({"standoff_swept_bounds_world_m": {"minimum": list(swept[0].bbox_min),
                                                    "maximum": list(swept[0].bbox_max)}} if swept else {}),
              "nearer_refusals": _nearer_refusals(refusals, best=best[0][1:], floor_z=float(floor_z), root=root,
                                                  footprint=list(footprint[:2]), room_measured=room is not None,
                                                  ranked_by_points=points is not None),
              **({"movable_obstacle_counterfactual": _counterfactual_record(
                  named, ignored, counterfactual, best=best[0][1:], ranked_by_points=points is not None)}
                 if named or ignored else {})}
    common = {"floor_z": floor_z, "root": root, "footprint_half_extent_m": footprint_half_extent_m, "clear": clear,
              "struck": struck, "motion": motion, "robot_id": robot_id if motion else None,
              "collision_sha256": collision_sha256, "room": measured, "search": search}
    if ranked:
        rows = candidates if points is None else spread_ranked(candidates, separation=_separation(
            footprint_half_extent_m), count=ranked)
        return [_stance(row, **common) for row in rows[:ranked]]
    return _stance(best, **common)


def _separation(footprint_half_extent_m: Sequence[float]) -> float:
    """A footprint's width: ranked stances this far apart share no floor."""

    return 2.0 * max(float(v) for v in footprint_half_extent_m[:2])


def spread_ranked(candidates: Sequence[Any], *, separation: float, count: int) -> list[Any]:
    """``candidates`` (best first, rows ``(key, x, y, yaw)``) with the first ``count`` a footprint apart.

    Each next stance is the best one at least ``separation`` from every one taken; once none is, the rest
    follow best first. The best stance is always first, so a choice of the first that passes is unchanged.
    """

    import numpy as np

    taken: list[Any] = []
    skipped: list[Any] = []
    where = np.empty((0, 2))
    for row in candidates:
        if len(taken) < count and not (np.hypot(where[:, 0] - row[1], where[:, 1] - row[2]) < separation - 1e-9).any():
            taken.append(row)
            where = np.vstack([where, [row[1], row[2]]])
        else:
            skipped.append(row)
    return taken + skipped


def _nearer_refusals(refusals: list[Any], *, best: Any, floor_z: float, root: float, footprint: list[float],
                     room_measured: bool, ranked_by_points: bool) -> dict[str, Any]:
    """The refused stances that ranked ahead of the chosen one: why each was refused, and what was in the way.

    ``by_reason`` counts them all. A room-mesh ``occupied`` refusal is also counted by where the top of the
    triangles in its way stands against the station's root (``blocker_tops``): a table top at the support's
    height is at the mount, a chair's seat or a stool below it, a chair's back, a counter or a wall above.
    ``frontier`` lists the refused stances that ranked just ahead of the chosen one, a footprint apart, best
    first, with the blocking triangles' world bounds and heights: what kept the station from standing there.
    """

    ahead = sorted((row for row in refusals if row[0] < best), key=lambda row: row[0])
    counts: dict[str, int] = {}
    tops = {"below_mount": 0, "at_mount": 0, "above_mount": 0}
    for _, reason, _, _, _, detail in ahead:
        counts[reason] = counts.get(reason, 0) + 1
        if reason == "occupied":
            top = float(detail[2][2]) - floor_z - root
            tops["below_mount" if top < -BLOCKER_MOUNT_BAND_M else
                 "above_mount" if top > BLOCKER_MOUNT_BAND_M else "at_mount"] += 1
    taken: list[int] = []
    for index in range(len(ahead) - 1, -1, -1):
        if len(taken) == NEARER_REFUSALS:
            break
        x, y = ahead[index][2], ahead[index][3]
        if all(math.hypot(x - ahead[other][2], y - ahead[other][3]) >= _separation(footprint) - 1e-9
               for other in taken):
            taken.append(index)
    frontier = []
    for index in sorted(taken):
        key, reason, x, y, _, detail = ahead[index]
        entry: dict[str, Any] = {"position_world_m": [round(x, 4), round(y, 4)], "reason": reason,
                                 "subject_distance_m": round(key[1], 4),
                                 **({"farthest_task_point_m": round(key[0], 4)} if ranked_by_points else {})}
        if reason == "occupied":
            count, low, high = detail
            entry["blockers"] = {"triangles": int(count),
                                 "bounds_world_m": {"minimum": [round(float(v), 4) for v in low],
                                                    "maximum": [round(float(v), 4) for v in high]},
                                 "bottom_above_floor_m": round(float(low[2]) - floor_z, 4),
                                 "top_above_floor_m": round(float(high[2]) - floor_z, 4)}
        elif detail is not None:
            entry["validator_failures"] = list(detail.failures)
        frontier.append(entry)
    return {"stances": len(ahead), "by_reason": counts,
            **({"blocker_tops": {**tops, "mount_band_m": BLOCKER_MOUNT_BAND_M}} if room_measured else {}),
            "frontier": frontier}


def _counterfactual_record(named: list[dict[str, Any]], ignored: int, rows: list[Any], *, best: Any,
                           ranked_by_points: bool) -> dict[str, Any]:
    """Refused stances that would be clear were the named obstacles moved: reported, never chosen."""

    rows = sorted(rows, key=lambda row: row[0])
    return {"rule": MOVABLE_RULE, "chosen": False, "membership": "triangle_or_box_bounds_within_named_bounds",
            "obstacles": [dict(row) for row in named], "ignored_obstacles": ignored,
            "clear_if_moved_stances": len(rows),
            "ahead_of_chosen_stances": sum(1 for row in rows if row[0] < best),
            "stances": [{"position_world_m": [round(x, 4), round(y, 4)], "moved": list(moved),
                         "subject_distance_m": round(key[1], 4),
                         **({"farthest_task_point_m": round(key[0], 4)} if ranked_by_points else {})}
                        for key, x, y, moved in rows[:MOVABLE_STANCES]]}


def _stance(row: Any, *, floor_z: float, root: float, footprint_half_extent_m: list[float], clear: int, struck: int,
            motion: Mapping[str, Any], robot_id: str | None, collision_sha256: str,
            room: Mapping[str, Any] | None = None, search: Mapping[str, Any] | None = None) -> dict[str, Any]:
    (reached, *_), x, y, yaw = row
    stance = {"schema_version": SCHEMA, "floor_z_m": float(floor_z),
              "position_world_m": [x, y, float(floor_z) + root],
              "orientation_xyzw": [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)], "yaw_rad": yaw,
              "footprint_half_extent_m": [float(v) for v in footprint_half_extent_m],
              "clear_stances": clear, "search_radius_m": SEARCH_RADIUS_M, "grid_step_m": GRID_STEP_M,
              **({"task_motion_clearance": {"robot_id": robot_id, "struck_stances": struck,
                                            "reached_beyond_task_range": reached}} if motion else {}),
              **({"room_mesh_clearance": dict(room)} if room is not None else {}),
              **({"search": dict(search)} if search is not None else {}),
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


__all__ = ["PEDESTAL_CLEARANCE_M", "PEDESTAL_SEARCH", "SCHEMA", "free_base_stance", "ground_stance",
           "spread_ranked"]
