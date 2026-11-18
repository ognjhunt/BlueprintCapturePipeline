"""A destination surface the scan does not show, estimated from what the video does show.

Beta journey step: compile (``website_task_preparation``), before the collider is sealed. Glass and
other see-through or shiny surfaces leave little or nothing in an Atlas collider: test job 4's glass
wall shelf ("move the lamp from the round bedside table to the glass shelf on the wall") had level
collider faces over about 5% of its visible area, so no footprint rested there and the job stopped on
``task_destination_surface_contact_required``.

When the job's one "on" destination has near-level collider faces over less than
``RECONSTRUCTED_MIN_SHARE`` of its visible area, and no surface carries the object's footprint at its
grounded centre, its top is estimated from the destination's own masks and the frames' cameras. No
model is asked: the masks already locate it in every frame they were made for.

- Plane: the level plane on which the destination's outline, cast from each frame's camera, lands in
  the same place in most frames, searched over its visible height range and ``HEIGHT_MARGIN_M``
  around it. Where its masks hold depth (the surface itself, edges, brackets, mounting points), the
  depth's level band must sit at that plane.
- Outline: the plan cells that most of the frames seeing them place inside their mask (the frames'
  outline), the largest connected piece of it. Any reconstructed wall or other standing surface the
  scan shows in the slab's layer is cut out of it, and its convex hull is the surface: a wall shelf's
  back edge so ends at the wall, where a mask that also took a strip of the wall above the shelf cast
  it behind (test job 4: the frames' own depth and the collider put the wall 4 cm inside that outline).
- Checks: the frames' outline, cast back into each frame, meets that frame's mask within
  ``REPROJECTION_MAX_SHARE`` of its diagonal (mean boundary distance) and ``FRAME_MIN_IOU``, in at least ``MIN_FRAMES``
  frames and ``MIN_AGREEING_SHARE`` of the frames that show it; the heights the frames agree on span
  no more than ``MAX_HEIGHT_SPAN_M``; no plane tilted along the surface's length more than
  ``MAX_TILT_DEGREES`` agrees better than the level one by more than ``AGREEMENT_SHARE``, and depth
  that shows a level band in the masks puts it at the plane; the surface's sides are plausible
  (``SIDE_RANGE_M``) and so is its cast area against the masks (``AREA_RATIO_RANGE``). Any refusal is
  recorded and nothing is added.
- Use: a slab ``THICKNESS_M`` thick under the plane is appended to the collider (every existing face
  keeps its index), written once per input and sealed in the base scene with its own digest and a
  ``scene_estimate:destination_surface_estimated`` row; the destination is grounded on it. The look
  (the splat) is unchanged.

Nothing here blocks: a surface that cannot be estimated leaves the collider as it was, the record says
why, and compile raises the same code as before, so the room package still ships. Every value is an
estimate from the video, never a measurement.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .atlas_scene_estimates import DESTINATION_SURFACE_ESTIMATED
from .core.handoff_lease import handoff_effect
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .website_support_geometry import MergedCollider, SupportOutline, footprint_at, search_window_m, support_under
from .website_track_mask_codec import decode_track_mask

REVISION = "website_unscanned_surface.v1"
#: Below this share of the destination's visible plan area under near-level collider faces at its visible top,
#: the scan did not show the surface.
RECONSTRUCTED_MIN_SHARE = 0.5
#: Plan points per side of the destination's visible area that the share is read at.
COVERAGE_GRID_STEPS = 33
#: The plan cell the outlines are cast onto.
CELL_M = 0.01
#: How far below and above its visible height range the plane is searched.
HEIGHT_MARGIN_M = 0.3
COARSE_STEP_M, HEIGHT_STEP_M = 0.02, 0.0025
#: A plane nearer a camera than this is seen edge-on from it: that frame does not vote at that height.
CAMERA_CLEARANCE_M = 0.1
#: Mask pixels this near the image border are where the frame cuts the surface off, never its edge.
BORDER_PX = 2
#: The widest plan area cast at one height: a plane where the frames' outlines spread wider is not the surface.
MAX_SPAN_M = 6.0
#: Frames that must agree, and the share of the frames showing the surface that must.
MIN_FRAMES, MIN_AGREEING_SHARE = 3, 2 / 3
#: A frame agrees when the outline cast back into it overlaps its mask by this much (intersection over
#: union) and their boundaries lie within ``REPROJECTION_MAX_SHARE`` of the frame's diagonal of each other on average.
FRAME_MIN_IOU = 0.5
#: The share of each frame's diagonal: 14.7 pixels of a 1280 x 720 geometry frame, 3 cm at 1.3 m with a 620-pixel
#: focal length.
REPROJECTION_MAX_SHARE = 0.01
#: The surface's cast area against each mask's, median over the frames: at least half of what the frames
#: show is left after the wall, and the hull does not swell past half as much again.
AREA_RATIO_RANGE = (0.5, 1.5)
#: As ``website_support_geometry.FITTED_MAX_TILT_DEGREES``: steeper is not a surface an object rests on.
MAX_TILT_DEGREES = 5.0
#: The tilts along the surface's length tried for the level check (``_tilt_search``).
TILT_STEPS_DEGREES = (-8.0, -4.0, 0.0, 4.0, 8.0)
#: Agreements within this share of each other are as good as each other: the heights that close to the best
#: one bound the plane's height (their span within ``MAX_HEIGHT_SPAN_M``, or the cameras are too close
#: together to fix it), and a steeper plane must do better by more than this to say the surface is not level.
AGREEMENT_SHARE, MAX_HEIGHT_SPAN_M = 0.05, 0.06
#: Depth on glass slants from frame to frame (10 to 20 degrees on test job 4's shelf): it places the plane's
#: height, never its tilt. Its level band must sit this near the plane the outlines agree on.
DEPTH_AGREEMENT_M = 0.03
#: A band of depth steeper than this is the wall seen through the glass, not the surface.
DEPTH_BAND_MAX_TILT_DEGREES = 45.0
#: Depth points that make a band, from at least two frames.
DEPTH_MIN_POINTS = 50
#: The outline's shortest and longest side, in metres: a ledge narrower than a hand holds nothing, and a
#: destination surface is furniture-sized.
SIDE_RANGE_M = (0.05, 4.0)
#: The estimated slab's thickness under its top, about a glass shelf's.
THICKNESS_M = 0.01
#: Collider faces steeper than this standing in the slab's layer are walls the outline stops at.
STANDING_MIN_TILT_DEGREES = 60.0
COLLIDER_NAME, RECEIPT_NAME = "collider_with_estimated_surface.glb", "estimated_surface.json"
#: The README's reason (``website_company_simulation``) and the basis recorded.
BASIS = "visible_edges_cast_from_frame_cameras"


def _plan(up: int) -> list[int]:
    return [axis for axis in range(3) if axis != up]


def reconstructed_share(collider: Any, low: Sequence[float], high: Sequence[float], *, up: int, up_sign: int,
                        meters_per_unit: float, window_m: float) -> float:
    """The share of the destination's visible plan rectangle under a gentle collider face within ``window_m``
    of its visible top (``high`` when up is positive, ``low`` when negative)."""
    plan = _plan(up)
    read = MergedCollider(collider, up=up, flipped=up_sign == -1)
    top = up_sign * (high if up_sign == 1 else low)[up]
    window = window_m / meters_per_unit
    centres = read.centres
    near = ((np.abs(centres[:, up] - top) <= window)
            & (centres[:, plan[0]] >= low[plan[0]] - window) & (centres[:, plan[0]] <= high[plan[0]] + window)
            & (centres[:, plan[1]] >= low[plan[1]] - window) & (centres[:, plan[1]] <= high[plan[1]] + window))
    if not near.any():
        return 0.0
    triangles = read.triangles(read.gentle[near])
    grid = np.array([[x, y] for x in np.linspace(low[plan[0]], high[plan[0]], COVERAGE_GRID_STEPS)
                     for y in np.linspace(low[plan[1]], high[plan[1]], COVERAGE_GRID_STEPS)])
    # The reflection of ``up`` leaves the plan axes as they are.
    return float(SupportOutline(triangles, np.zeros((0, 2, 3)), 0.0, up=up).covers(grid).mean())


def _views(track: Mapping[str, Any], frames: Sequence[Mapping[str, Any]], matrix: np.ndarray, *, up: int,
           up_sign: int) -> list[dict[str, Any]]:
    """Each masked frame: its mask, its camera in the runtime frame, and the mask's depth points lifted there."""
    by_id = {frame["frame_id"]: frame for frame in frames}
    to_source = np.linalg.inv(matrix)
    views = []
    for observation in track.get("observations") or []:
        frame = by_id.get(observation.get("source_frame_id"))
        if frame is None:
            continue
        mask = decode_track_mask(observation)
        if not mask.any():
            continue
        if _sha256_file(Path(frame["geometry_path"])) != frame["geometry_digest"]:
            raise ValueError("task_target_geometry_changed")
        world_from_camera = np.asarray(frame["world_from_camera"], dtype=float)
        camera_from_runtime = np.linalg.inv(world_from_camera) @ to_source
        intrinsics = np.asarray(frame["intrinsics"], dtype=float)
        with np.load(frame["geometry_path"], allow_pickle=False) as geometry:
            depth, valid = geometry["depth_m"], geometry["valid_mask"]
        if depth.shape != mask.shape:
            raise ValueError("task_target_mask_geometry_mismatch")
        ys, xs = np.nonzero(mask & valid)
        camera = (np.stack([xs, ys, np.ones_like(xs)], axis=1) @ np.linalg.inv(intrinsics).T) * depth[ys, xs, None]
        source = camera @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]
        centre = matrix[:3, :3] @ world_from_camera[:3, 3] + matrix[:3, 3]
        views.append({"frame_id": frame["frame_id"], "mask": mask, "intrinsics": intrinsics,
                      "camera_from_runtime": camera_from_runtime, "height": up_sign * float(centre[up]),
                      "depth_points": source @ matrix[:3, :3].T + matrix[:3, 3]})
    return views


def _project(view: Mapping[str, Any], points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel columns and rows of runtime ``points`` in a view, and whether each lies in front, inside the image."""
    camera = points @ view["camera_from_runtime"][:3, :3].T + view["camera_from_runtime"][:3, 3]
    depth = camera[:, 2]
    front = depth > 1e-6
    pixels = camera @ view["intrinsics"].T
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = pixels[:, 0] / depth, pixels[:, 1] / depth
    height, width = view["mask"].shape
    seen = front & (u >= BORDER_PX) & (u <= width - 1 - BORDER_PX) & (v >= BORDER_PX) & (v <= height - 1 - BORDER_PX)
    return u, v, seen


def _rays(view: Mapping[str, Any], columns: np.ndarray, rows: np.ndarray, height: float, *, up: int,
          up_sign: int) -> tuple[np.ndarray, np.ndarray]:
    """Where the rays through pixels meet the level plane at ``height``: runtime points, and whether they do."""
    rotation = view["camera_from_runtime"][:3, :3]
    inverse = np.linalg.inv(rotation)
    origin = -inverse @ view["camera_from_runtime"][:3, 3]
    directions = (np.stack([columns, rows, np.ones_like(columns)], axis=1).astype(float)
                  @ np.linalg.inv(view["intrinsics"]).T) @ inverse.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (up_sign * height - origin[up]) / directions[:, up]
    hits = np.isfinite(t) & (t > 0)
    return origin + np.where(hits, t, 0.0)[:, None] * directions, hits


def _cells(views: list[dict[str, Any]], height: float, *, up: int, up_sign: int,
           meters_per_unit: float) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    """The plan grid the frames' outlines land on at ``height``: cell centres, origin and shape. None when the
    outlines spread wider than ``MAX_SPAN_M`` (or no frame casts its outline onto the plane)."""
    plan, cell = _plan(up), CELL_M / meters_per_unit
    landed = []
    for view in views:
        if abs(view["height"] - height) < CAMERA_CLEARANCE_M / meters_per_unit:
            continue
        rows, columns = np.nonzero(view["mask"])
        step = max(1, len(rows) // 400)
        points, hits = _rays(view, columns[::step], rows[::step], height, up=up, up_sign=up_sign)
        landed.append(points[hits][:, plan])
    landed = [points for points in landed if len(points)]
    if not landed:
        return None
    every = np.concatenate(landed)
    low, high = np.quantile(every, 0.01, axis=0) - 5 * cell, np.quantile(every, 0.99, axis=0) + 5 * cell
    if np.any(high - low > MAX_SPAN_M / meters_per_unit):
        return None
    shape = tuple(int(math.ceil(value)) for value in (high - low) / cell)
    xs, ys = (low[index] + (np.arange(shape[index]) + 0.5) * cell for index in range(2))
    centres = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    return centres, low, shape


def _votes(views: list[dict[str, Any]], centres: np.ndarray, height: Any, *, up: int,
           up_sign: int, meters_per_unit: float) -> tuple[np.ndarray, np.ndarray]:
    """For each view and cell at ``height`` (one, or one per cell): whether the view sees the cell, and whether its
    mask holds it."""
    points = np.zeros((len(centres), 3))
    points[:, _plan(up)], points[:, up] = centres, up_sign * np.asarray(height, dtype=float)
    seen = np.zeros((len(views), len(centres)), dtype=bool)
    inside = np.zeros_like(seen)
    for index, view in enumerate(views):
        if abs(view["height"] - float(np.mean(height))) < CAMERA_CLEARANCE_M / meters_per_unit:
            continue
        u, v, visible = _project(view, points)
        seen[index] = visible
        inside[index, visible] = view["mask"][np.rint(v[visible]).astype(int), np.rint(u[visible]).astype(int)]
    return seen, inside


def _agreement(seen: np.ndarray, inside: np.ndarray) -> tuple[float, np.ndarray]:
    """How well the frames' outlines land on one place: their pooled overlap with the cells most frames seeing
    each place inside their mask (intersection over union, within each frame's view), and those cells."""
    held, viewed = inside.sum(axis=0), seen.sum(axis=0)
    fused = (held >= 2) & (2 * held >= viewed)
    union = int(((inside | fused[None]) & seen).sum())
    return (float((inside & fused[None]).sum()) / union if union else 0.0), fused


def _score(views, height, *, up, up_sign, meters_per_unit):
    grid = _cells(views, height, up=up, up_sign=up_sign, meters_per_unit=meters_per_unit)
    if grid is None:
        return 0.0, None
    centres, low, shape = grid
    seen, inside = _votes(views, centres, height, up=up, up_sign=up_sign, meters_per_unit=meters_per_unit)
    score, fused = _agreement(seen, inside)
    return score, (centres, low, shape, fused)


def _search_height(views, low: float, high: float, *, up, up_sign, meters_per_unit) -> tuple[float, float, list]:
    """The height where the outlines agree best: coarse steps over the range, then fine steps around the best."""
    scale = 1 / meters_per_unit
    coarse = np.arange(low - HEIGHT_MARGIN_M * scale, high + HEIGHT_MARGIN_M * scale + 1e-9, COARSE_STEP_M * scale)
    curve = [(float(h), _score(views, h, up=up, up_sign=up_sign, meters_per_unit=meters_per_unit)[0])
             for h in coarse]
    best = max(curve, key=lambda row: row[1])[0]
    fine = np.arange(best - COARSE_STEP_M * scale, best + COARSE_STEP_M * scale + 1e-9, HEIGHT_STEP_M * scale)
    refined = [(float(h), _score(views, h, up=up, up_sign=up_sign, meters_per_unit=meters_per_unit)[0])
               for h in fine]
    height, score = max(refined, key=lambda row: row[1])
    return height, score, sorted(curve + refined)


def _tilt_search(views, height, grid, *, up, up_sign, meters_per_unit) -> dict[str, Any]:
    """The best of a few planes through the outline at ``height`` tilted along its length.

    Along its length only: across its depth the frames see the surface's front edge as well as its top, and from
    viewpoints on one side a slab's edge and a tilt towards the cameras cast the same outlines (test job 4, and a
    level 1 cm slab here, both fitted best at 8 degrees that way)."""
    centres, _, _, fused = grid
    middle = centres[fused].mean(axis=0)
    along = np.linalg.svd(centres[fused] - middle, full_matrices=False)[2][0]
    best = (-1.0, 0.0)
    for offset in (-HEIGHT_STEP_M / meters_per_unit, 0.0, HEIGHT_STEP_M / meters_per_unit):
        for slope in np.tan(np.deg2rad(TILT_STEPS_DEGREES)):
            heights = height + offset + ((centres - middle) @ along) * slope
            seen, inside = _votes(views, centres, heights, up=up, up_sign=up_sign, meters_per_unit=meters_per_unit)
            score = _agreement(seen, inside)[0]
            if score > best[0] + 1e-9:
                best = (score, float(slope))
    return {"score": best[0], "tilt_degrees": float(np.degrees(np.arctan(abs(best[1])))), "axis": "length",
            "steps_degrees": list(TILT_STEPS_DEGREES)}


def _standing_cells(collider: Any, height: float, centres: np.ndarray, low: np.ndarray, shape: tuple[int, int], *,
                    up: int, up_sign: int, meters_per_unit: float) -> np.ndarray:
    """Cells a reconstructed wall or other standing surface crosses within the slab's layer, sliced at its middle."""
    plan, cell = _plan(up), CELL_M / meters_per_unit
    middle, half = up_sign * (height - THICKNESS_M / meters_per_unit / 2), THICKNESS_M / meters_per_unit / 2
    vertices, faces = np.asarray(collider.vertices, dtype=float), np.asarray(collider.faces, dtype=np.int64)
    high = low + np.asarray(shape) * cell
    blocked = np.zeros(shape, dtype=bool)
    for start in range(0, len(faces), 1 << 18):
        triangles = vertices[faces[start:start + (1 << 18)]]
        span_low, span_high = triangles.min(axis=1), triangles.max(axis=1)
        near = ((span_low[:, up] <= middle + half) & (span_high[:, up] >= middle - half)
                & (span_high[:, plan[0]] >= low[0]) & (span_low[:, plan[0]] <= high[0])
                & (span_high[:, plan[1]] >= low[1]) & (span_low[:, plan[1]] <= high[1]))
        if not near.any():
            continue
        triangles = triangles[near]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        length = np.linalg.norm(normals, axis=1)
        steep = (length > 0) & (np.abs(normals[:, up]) <= np.cos(np.deg2rad(STANDING_MIN_TILT_DEGREES)) * length)
        for triangle in triangles[steep]:
            points = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                da, db = triangle[a, up] - middle, triangle[b, up] - middle
                if da == db:
                    continue
                t = da / (da - db)
                if 0.0 <= t <= 1.0:
                    points.append(triangle[a] + t * (triangle[b] - triangle[a]))
            if len(points) < 2:
                points = [point for point in triangle if abs(point[up] - middle) <= half]
            if not points:
                continue
            ends = np.asarray(points)[:, plan]
            samples = max(2, int(math.ceil(np.linalg.norm(ends[-1] - ends[0]) / (cell / 2))) + 1)
            line = ends[0] + np.linspace(0.0, 1.0, samples)[:, None] * (ends[-1] - ends[0])
            index = np.floor((line - low) / cell).astype(int)
            keep = (index >= 0).all(axis=1) & (index[:, 0] < shape[0]) & (index[:, 1] < shape[1])
            blocked[index[keep, 0], index[keep, 1]] = True
    return blocked


def _beside_standing(piece: np.ndarray, standing: np.ndarray) -> np.ndarray:
    """The outline's largest piece on one side of the standing surfaces, reaching up to them.

    The cut is widened by a cell first, so a wall crossing the grid diagonally leaves no corner gap a piece
    behind it could join through; the kept piece then grows back by that cell, never onto the wall."""
    from scipy import ndimage

    block = np.ones((3, 3), dtype=bool)
    kept = _largest_piece(piece & ~ndimage.binary_dilation(standing, block))
    return _largest_piece(piece & ~standing & ndimage.binary_dilation(kept, block)) if kept.any() else kept


def _largest_piece(cells: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    labels, count = ndimage.label(cells)
    if not count:
        return cells
    sizes = np.bincount(labels.ravel())[1:]
    return ndimage.binary_fill_holes(labels == int(np.argmax(sizes)) + 1)


def _hull(cells: np.ndarray, low: np.ndarray, cell: float) -> np.ndarray:
    """The convex hull of the cells' corners, counter-clockwise in plan."""
    from scipy.spatial import ConvexHull

    index = np.argwhere(cells)
    corners = np.unique(np.concatenate([index + offset for offset in ((0, 0), (0, 1), (1, 0), (1, 1))]), axis=0)
    points = low + corners * cell
    hull = ConvexHull(points)
    polygon = points[hull.vertices]
    area = 0.5 * np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - np.roll(polygon[:, 0], -1) * polygon[:, 1])
    return polygon if area > 0 else polygon[::-1]


def _area(polygon: np.ndarray) -> float:
    return float(0.5 * abs(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                                  - np.roll(polygon[:, 0], -1) * polygon[:, 1])))


def _inside(polygon: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Whether plan ``points`` lie inside the counter-clockwise convex ``polygon``."""
    inside = np.ones(len(points), dtype=bool)
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = end - start
        inside &= edge[0] * (points[:, 1] - start[1]) - edge[1] * (points[:, 0] - start[0]) >= 0
    return inside


def _sides(polygon: np.ndarray) -> list[float]:
    """The sides of the smallest rectangle around the convex ``polygon``, shortest first."""
    best = None
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        direction = (end - start) / max(np.linalg.norm(end - start), 1e-12)
        across = np.array([-direction[1], direction[0]])
        extents = [float(np.ptp(polygon @ axis)) for axis in (direction, across)]
        if best is None or extents[0] * extents[1] < best[0] * best[1]:
            best = extents
    return sorted(best)


def _boundary(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    edge = mask & ~ndimage.binary_erosion(mask)
    edge[:BORDER_PX + 1], edge[-BORDER_PX - 1:], edge[:, :BORDER_PX + 1], edge[:, -BORDER_PX - 1:] = False, False, False, False
    return edge


def _reprojection(view: Mapping[str, Any], polygon: np.ndarray, height: float, *, up: int,
                  up_sign: int) -> dict[str, Any]:
    """The outline cast back into one frame against its mask: overlap, area ratio and mean boundary distance."""
    from scipy import ndimage

    mask = view["mask"]
    # Only the box around the mask and the outline's own corners is cast; a corner behind the camera casts all.
    corners = np.zeros((len(polygon), 3))
    corners[:, _plan(up)], corners[:, up] = polygon, up_sign * height
    u, v, _ = _project(view, corners)
    camera = corners @ view["camera_from_runtime"][:3, :3].T + view["camera_from_runtime"][:3, 3]
    rows, columns = np.nonzero(mask)
    if (camera[:, 2] > 1e-6).all():
        rows, columns = np.r_[rows, np.clip(v, -1, mask.shape[0])], np.r_[columns, np.clip(u, -1, mask.shape[1])]
        top, left = (max(0, int(np.floor(value.min())) - 2 * BORDER_PX) for value in (rows, columns))
        bottom, right = (min(size, int(np.ceil(value.max())) + 2 * BORDER_PX + 1)
                         for value, size in ((rows, mask.shape[0]), (columns, mask.shape[1])))
    else:
        top, left, bottom, right = 0, 0, mask.shape[0], mask.shape[1]
    grid_rows, grid_columns = np.mgrid[top:bottom, left:right]
    points, hits = _rays(view, grid_columns.ravel(), grid_rows.ravel(), height, up=up, up_sign=up_sign)
    cast = np.zeros_like(mask)
    cast[top:bottom, left:right] = (hits & _inside(polygon, points[:, _plan(up)])).reshape(bottom - top, right - left)
    union = int((cast | mask).sum())
    row = {"frame_id": view["frame_id"], "iou": float((cast & mask).sum()) / union if union else 0.0,
           "area_ratio": float(cast.sum()) / max(int(mask.sum()), 1)}
    cast_edge, mask_edge = _boundary(cast), _boundary(mask)
    row["boundary_px"] = None  # no edge of one of them inside the frame: nothing to measure
    if cast_edge.any() and mask_edge.any():
        to_mask, to_cast = ndimage.distance_transform_edt(~mask_edge), ndimage.distance_transform_edt(~cast_edge)
        row["boundary_px"] = float((to_mask[cast_edge].mean() + to_cast[mask_edge].mean()) / 2)
    row["agrees"] = (row["iou"] >= FRAME_MIN_IOU and row["boundary_px"] is not None
                     and row["boundary_px"] <= REPROJECTION_MAX_SHARE * float(np.hypot(*mask.shape)))
    return row


def _depth_anchor(views: list[dict[str, Any]], height: float, *, up: int, up_sign: int,
                  meters_per_unit: float) -> dict[str, Any]:
    """The level band the masks' depth shows, if any: where it lies against ``height`` and how level it is."""
    points = [view["depth_points"] for view in views]
    counts = [len(value) for value in points]
    if sum(counts) < DEPTH_MIN_POINTS:
        return {"status": "no_depth", "points": int(sum(counts))}
    every = np.concatenate(points)
    heights = np.sort(up_sign * every[:, up])
    band = 2 * DEPTH_AGREEMENT_M / meters_per_unit
    ends = np.searchsorted(heights, heights + band, side="right")
    first = int(np.argmax(ends - np.arange(len(heights))))
    held = int(ends[first] - first)
    if held < len(heights) / 2:
        # Through glass the depth reads the wall or the room behind: no level band to anchor on.
        return {"status": "no_level_band", "points": len(heights), "largest_band_share": held / len(heights)}
    centre = float(np.median(heights[first:ends[first]]))
    near = np.abs(up_sign * every[:, up] - centre) <= band / 2
    chosen = every[near][:: max(1, int(near.sum()) // 20000)]
    # The band's own plane: the direction it is thinnest along. A wall seen through the glass is thin across.
    normal = np.linalg.svd(chosen - chosen.mean(axis=0), full_matrices=False)[2][-1]
    tilt = float(np.degrees(np.arccos(min(1.0, abs(normal[up])))))
    frames = sum(1 for value in points if (np.abs(up_sign * value[:, up] - centre) <= band / 2).any())
    row = {"points": len(heights), "band_share": held / len(heights), "band_frames": frames,
           "band_z_world_m": centre * meters_per_unit, "offset_from_plane_m": (centre - height) * meters_per_unit,
           "tilt_degrees": tilt}
    if tilt > DEPTH_BAND_MAX_TILT_DEGREES or frames < 2:
        # Nearer standing than level, or one frame's: the wall behind the glass, not a level band.
        return {"status": "no_level_band", **row}
    return {"status": "level_band", **row}


def estimate_surface(*, track: Mapping[str, Any], frames: Sequence[Mapping[str, Any]], matrix: np.ndarray,
                     collider: Any, low: Sequence[float], high: Sequence[float], up: int, up_sign: int,
                     meters_per_unit: float) -> dict[str, Any]:
    """The destination's top estimated from its masks (``low``/``high``: its visible runtime bounds).

    ``status`` is ``estimated``, with ``polygon`` (plan, runtime units) and ``height`` (along up, runtime units), or
    ``not_estimated`` with its ``reasons``; either way with the evidence each check read, in metres (heights as the
    simulator's z, ``top_z_world_m``)."""
    mpu = meters_per_unit
    views = _views(track, frames, matrix, up=up, up_sign=up_sign)
    record: dict[str, Any] = {"basis": BASIS, "masked_frames": len(views), "physical_measurement": False}
    if len(views) < MIN_FRAMES:
        return {**record, "status": "not_estimated", "reasons": ["too_few_masked_frames"]}
    visible = sorted(up_sign * float(value) for value in (low[up], high[up]))
    height, score, curve = _search_height(views, *visible, up=up, up_sign=up_sign, meters_per_unit=mpu)
    grid = _score(views, height, up=up, up_sign=up_sign, meters_per_unit=mpu)[1]
    close = [value for value, agreement in curve if agreement >= (1 - AGREEMENT_SHARE) * score]
    record.update(top_z_world_m=height * mpu, agreement=score, height_span_m=(max(close) - min(close)) * mpu,
                  height_curve=[[round(value * mpu, 4), round(agreement, 4)] for value, agreement in curve])
    if grid is None or score <= 0:
        return {**record, "status": "not_estimated", "reasons": ["outlines_never_meet"]}
    centres, origin, shape, fused = grid
    cell = CELL_M / mpu
    piece = _largest_piece(fused.reshape(shape))
    # The frames' own outline, before the scan's walls are read: what every agreement check is about.
    outline = _hull(piece, origin, cell)
    rows = [_reprojection(view, outline, height, up=up, up_sign=up_sign) for view in views]
    agreeing = [row for row in rows if row["agrees"]]
    finite = [row["boundary_px"] for row in rows if row["boundary_px"] is not None]
    record["frames_outline"] = {"area_m2": _area(outline) * mpu * mpu,
                                "median_boundary_px": float(np.median(finite)) if finite else None,
                                "median_iou": float(np.median([row["iou"] for row in rows])),
                                "agreeing_frames": len(agreeing), "frames": rows}
    kept = _beside_standing(piece, _standing_cells(collider, height, centres, origin, shape, up=up,
                                                   up_sign=up_sign, meters_per_unit=mpu))
    record["cut_at_standing_surfaces_m2"] = float(piece.sum() - kept.sum()) * CELL_M * CELL_M
    if kept.sum() < 3:
        return {**record, "status": "not_estimated", "reasons": ["no_outline_left_beside_the_scan"]}
    polygon = _hull(kept, origin, cell)
    sides = [value * mpu for value in _sides(polygon)]
    ratios = [_reprojection(view, polygon, height, up=up, up_sign=up_sign)["area_ratio"] for view in views]
    depth = _depth_anchor(views, height, up=up, up_sign=up_sign, meters_per_unit=mpu)
    tilt = _tilt_search(views, height, grid, up=up, up_sign=up_sign, meters_per_unit=mpu)
    record.update(sides_m=sides, area_m2=_area(polygon) * mpu * mpu, median_area_ratio=float(np.median(ratios)),
                  depth_anchor=depth, tilt_search={**tilt, "level_agreement": score})
    reasons = []
    if (max(close) - min(close)) * mpu > MAX_HEIGHT_SPAN_M:
        reasons.append("height_not_fixed_by_the_frames")
    if len(agreeing) < max(MIN_FRAMES, math.ceil(MIN_AGREEING_SHARE * len(rows) - 1e-9)):
        reasons.append("frames_disagree_with_their_outline")
    if tilt["tilt_degrees"] > MAX_TILT_DEGREES and tilt["score"] > (1 + AGREEMENT_SHARE) * score:
        reasons.append("outlines_fit_a_tilted_plane")
    if depth["status"] == "level_band" and abs(depth["offset_from_plane_m"]) > DEPTH_AGREEMENT_M:
        reasons.append("depth_disagrees_with_the_outline")
    if not (SIDE_RANGE_M[0] <= sides[0] and sides[1] <= SIDE_RANGE_M[1]):
        reasons.append("surface_size_implausible")
    if not AREA_RATIO_RANGE[0] <= record["median_area_ratio"] <= AREA_RATIO_RANGE[1]:
        reasons.append("surface_size_disagrees_with_the_masks")
    if reasons:
        return {**record, "status": "not_estimated", "reasons": reasons}
    return {**record, "status": "estimated", "polygon": polygon.tolist(), "height": height}


def _slab(polygon: np.ndarray, height: float, *, up: int, up_sign: int,
          meters_per_unit: float) -> tuple[np.ndarray, np.ndarray]:
    """A closed slab ``THICKNESS_M`` thick under ``polygon`` at ``height``: vertices and outward-facing faces."""
    plan, count = _plan(up), len(polygon)
    top = np.zeros((count + 1, 3))
    top[:count, plan], top[count, plan] = polygon, polygon.mean(axis=0)
    top[:, up] = up_sign * height
    bottom = top.copy()
    bottom[:, up] = up_sign * (height - THICKNESS_M / meters_per_unit)
    vertices = np.vstack([top, bottom])
    faces, wanted = [], []
    lift = np.zeros(3)
    lift[up] = up_sign
    for index in range(count):
        following = (index + 1) % count
        faces += [[count, index, following], [2 * count + 1, count + 1 + following, count + 1 + index]]
        wanted += [lift, -lift]
        outward = np.zeros(3)
        outward[plan] = (polygon[index] + polygon[following]) / 2 - polygon.mean(axis=0)
        faces += [[index, count + 1 + index, count + 1 + following], [index, count + 1 + following, following]]
        wanted += [outward, outward]
    faces = np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    flip = (normals * np.asarray(wanted)).sum(axis=1) < 0
    faces[flip] = faces[flip][:, ::-1]
    return vertices, faces


def _with_slab(collider: Any, vertices: np.ndarray, faces: np.ndarray) -> Any:
    """The collider with the slab's faces appended after its own, its visual extended to them."""
    import trimesh

    combined_vertices = np.vstack([np.asarray(collider.vertices, dtype=np.float64), vertices])
    combined_faces = np.vstack([np.asarray(collider.faces, dtype=np.int64), faces + len(collider.vertices)])
    visual, kind = getattr(collider, "visual", None), getattr(getattr(collider, "visual", None), "kind", None)
    grey = np.array([128, 128, 128, 255], dtype=np.uint8)
    options: dict[str, Any] = {}
    if kind == "vertex":
        options["vertex_colors"] = np.vstack([visual.vertex_colors, np.tile(grey, (len(vertices), 1))])
    elif kind == "face":
        options["face_colors"] = np.vstack([visual.face_colors, np.tile(grey, (len(faces), 1))])
    elif kind == "texture" and getattr(visual, "uv", None) is not None:
        options["visual"] = trimesh.visual.TextureVisuals(
            uv=np.vstack([visual.uv, np.zeros((len(vertices), 2))]), material=visual.material)
    return trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces, process=False, **options)


def _seal(*, collider: Any, base_scene: Mapping[str, Any], vertices: np.ndarray, faces: np.ndarray,
          summary: Mapping[str, Any], output_root: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Write the collider with the slab once per input, and the base scene that binds it."""
    import trimesh

    binding = {"revision": REVISION, "source_collision_mesh_digest": base_scene["collision_mesh_digest"],
               "slab_vertices": np.round(vertices, 9).tolist(), "slab_faces": faces.tolist()}
    root = Path(output_root) / "collider" / ("estimated-surface-" + canonical_digest(binding)[7:31])
    receipt_path, path = root / RECEIPT_NAME, root / COLLIDER_NAME
    receipt = None
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text())
        if (prior.get("binding") == binding and prior.get("digest") == canonical_digest(prior, digest_field="digest")
                and path.is_file() and _sha256_file(path) == prior["output"]["sha256"]):
            receipt = prior
    if receipt is None:
        root.mkdir(parents=True, exist_ok=True)
        staged = root / (COLLIDER_NAME + ".partial")
        _with_slab(collider, vertices, faces).export(str(staged), file_type="glb")
        with handoff_effect("shared_artifact_promotion"):
            os.replace(staged, path)
        receipt = {"schema_version": REVISION, "binding": binding, "output": {"path": str(path),
                   "sha256": _sha256_file(path)}, "splat_unchanged": True, "physical_measurement": False}
        receipt["digest"] = canonical_digest(receipt, digest_field="digest")
        from .common import write_json
        with handoff_effect("shared_artifact_promotion"):
            write_json(receipt_path, receipt)
    digest = receipt["output"]["sha256"]
    row = {"code": DESTINATION_SURFACE_ESTIMATED, "source": "website_unscanned_surface", "evidence": dict(summary)}
    sealed = {**base_scene, "collision_mesh_path": str(path), "collision_mesh_digest": digest,
              "collision_binding_id": "website-collider-" + digest[7:39],
              "estimated_surface": {"schema_version": REVISION, "receipt_path": str(receipt_path),
                                    "receipt_digest": receipt["digest"],
                                    "source_collision_mesh_path": base_scene["collision_mesh_path"],
                                    "source_collision_mesh_digest": base_scene["collision_mesh_digest"],
                                    "source_collision_binding_id": base_scene.get("collision_binding_id")},
              "scene_estimates": [*(row_ for row_ in base_scene.get("scene_estimates") or []
                                    if not isinstance(row_, Mapping) or row_.get("code") != row["code"]), row]}
    from .common import write_json
    write_json(Path(output_root) / "base_scene.json", sealed)
    mesh = trimesh.load(str(path), force="mesh", process=False)
    return mesh, sealed, {"collision_mesh_digest": digest, "receipt_path": str(receipt_path),
                          "receipt_digest": receipt["digest"],
                          "source_collision_mesh_digest": base_scene["collision_mesh_digest"]}


def estimate_unscanned_destination(*, collider: Any, base_scene: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                                   frames: Sequence[Mapping[str, Any]], matrix: np.ndarray,
                                   subject: tuple[Sequence[float], Sequence[float]] | None, up: int, up_sign: int,
                                   meters_per_unit: float, placement_residual_m: float | None,
                                   output_root: Path) -> tuple[Any, Mapping[str, Any], list, dict[str, Any]]:
    """Compile's one call: the collider, base scene and destination rows to place on, and the record to keep.

    Unchanged, with no record, unless the job has one "on" destination with masks, no surface under the object's
    footprint (``subject``: its bounds, when known) at the destination's grounded centre, and level collider faces
    over less than ``RECONSTRUCTED_MIN_SHARE`` of its visible area. Then the record says what was estimated, or
    why nothing was; only an estimate that passes changes the collider, the base scene (written to
    ``output_root``) and the row, which is grounded on the estimated surface from then on
    (``estimated_surface_runtime``)."""
    from .website_task_preparation import _grounded_destination

    row = rows[0] if len(rows) == 1 else {}
    if (collider is None or row.get("placement_relation") != "on" or not isinstance(row.get("track"), Mapping)
            or "estimated_surface_runtime" in row):
        return collider, base_scene, list(rows), {}
    low, high, position = _grounded_destination(row, matrix, up=up, up_sign=up_sign)
    options = {"up": up, "meters_per_unit": meters_per_unit, "up_sign": up_sign,
               "placement_residual_m": placement_residual_m}
    if subject is not None and support_under(collider, *footprint_at(*subject, position, up=up, up_sign=up_sign),
                                             **options) is not None:
        return collider, base_scene, list(rows), {}
    share = reconstructed_share(collider, low, high, up=up, up_sign=up_sign, meters_per_unit=meters_per_unit,
                                window_m=search_window_m(placement_residual_m))
    if share >= RECONSTRUCTED_MIN_SHARE:
        return collider, base_scene, list(rows), {}
    record: dict[str, Any] = {"schema_version": REVISION, "target_id": row["target_id"],
                              "reconstructed_share": share, "reconstructed_min_share": RECONSTRUCTED_MIN_SHARE}
    try:
        estimate = estimate_surface(track=row["track"], frames=frames, matrix=np.asarray(matrix, dtype=float),
                                    collider=collider, low=low, high=high, up=up, up_sign=up_sign,
                                    meters_per_unit=meters_per_unit)
    except (np.linalg.LinAlgError, ArithmeticError) as exc:
        estimate = {"status": "not_estimated", "reasons": [f"estimate_failed:{type(exc).__name__}"]}
    record.update({key: value for key, value in estimate.items() if key not in {"polygon", "height"}})
    if estimate["status"] != "estimated":
        return collider, base_scene, list(rows), {"destination_surface_estimate": record}
    polygon, height = np.asarray(estimate["polygon"]), float(estimate["height"])
    vertices, faces = _slab(polygon, height, up=up, up_sign=up_sign, meters_per_unit=meters_per_unit)
    summary = {"target_id": row["target_id"], "basis": BASIS, "top_z_world_m": record["top_z_world_m"],
               "sides_m": record["sides_m"], "median_boundary_px": record["frames_outline"]["median_boundary_px"],
               "agreeing_frames": record["frames_outline"]["agreeing_frames"], "masked_frames": record["masked_frames"]}
    collider, base_scene, sealed = _seal(collider=collider, base_scene=base_scene, vertices=vertices, faces=faces,
                                         summary=summary, output_root=output_root)
    bounds = [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]
    estimated = {**row, "estimated_surface_runtime": bounds}
    record.update(flag=DESTINATION_SURFACE_ESTIMATED, collider=sealed, look="unchanged",
                  outline_runtime=polygon.tolist(), thickness_m=THICKNESS_M,
                  slab_faces=[int(len(collider.faces) - len(faces)), int(len(collider.faces))])
    if subject is not None:
        _, _, centre = _grounded_destination(estimated, matrix, up=up, up_sign=up_sign)
        footprint = footprint_at(*subject, centre, up=up, up_sign=up_sign)
        record["subject_footprint"] = {
            "sides_m": sorted(float((footprint[1][axis] - footprint[0][axis]) * meters_per_unit)
                              for axis in _plan(up)),
            "supported_at_centre": support_under(collider, *footprint, **options) is not None}
    return collider, base_scene, [estimated], {"destination_surface_estimate": record}
