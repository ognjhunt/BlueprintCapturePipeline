"""CPU spawn check of the Unitree G1 at its reset pose, before any paid simulator time.

On 2026-09-27 the only real G1 trial (DP policy, scene 841757) blew up at spawn: robot-vs-background
contact of 104 to 8,677 N in steps 1-9 (0.02-0.18 s), then the robot out of the workspace for the rest
of the episode; the policy's second query already saw a state far out of distribution. The body began
inside the room's geometry. The stance search (``beta_policy_screening_g1_prepare.g1_stance_search``)
clears only the G1's floor footprint; this module measures the whole body.

It is the arm rest-overlap check of ``scene_physics_preflight`` applied to the G1: the robot USD's
bodies, colliders and joints (``load_asset_model``, instance proxies included), posed at the packet's
reset joints with the root body (the pelvis) at the stance (``body_poses``), against the static
background collision near it (``load_background_triangles``, ``sample_surface``, ``penetration``), with
the shortest push that clears it (``clearing_translation``). The feet are meant to stand on the floor:
a sole at most ``FOOT_SUPPORT_TOLERANCE_M`` below an upward-facing surface stands on it. Any other
penetration beyond the arm check's rest tolerance is overlap, and so is a sole sunk further. As in the
arm check, a collider wholly behind a surface that crosses none of it is not seen.

``status`` is ``clear``, ``overlap`` or ``unmeasured``. A body with no collider prim has no contact in
the engine either: it is listed in ``bodies_without_colliders`` and changes nothing. A capsule is read as
its bounding cylinder. A collider of a shape the measure has no solid for, or an input it cannot read,
makes the reading ``unmeasured`` with its reason. Only a measured overlap protects money: planning funds
no G1 row whose packet reads ``overlap`` (``g1_spawn_overlap``), as ``require_scene_physics_admission``
refuses an arm packet with rest overlap, since a run that starts inside the room cannot measure the
policy. ``unmeasured``, or no report at all, is a flag (``g1_spawn_clearance_unmeasured``) carried in the
row's ``setup_quality_flags`` and its result summary; the row is still bound and quoted. A foot with
no floor found under its sole within ``SOLE_SEARCH_M``, or a sole more than ``FOOT_SUPPORT_TOLERANCE_M``
above it, is flagged the same way (``g1_spawn_foot_unsupported``): a hole or steep patch in a scanned
floor is not an overlap, but the robot may tip or drop at spawn.
A scene whose floor is a development-only estimate (a flat floor standing in for an uneven scanned one,
``atlas_world_levelling``) is flagged too (``g1_scene_floor_estimated``, ``packet_floor_estimate``): every
reading here is then against that flat floor, so clear and supported say nothing about the floor as scanned.
Nothing here runs a simulator, calls a provider or changes an input file.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .scene_physics_preflight import (
    DEFAULT_REST_TOLERANCE_M,
    DEFAULT_SAMPLE_SPACING_M,
    MAXIMUM_BACKGROUND_SAMPLES,
    ScenePhysicsPreflightError,
    _clip_large_triangles,
    _file_sha256,
    _placed,
    _pose_matrix,
    _reachable_triangles,
    _summary,
    body_poses,
    clearing_translation,
    load_asset_model,
    load_background_triangles,
    penetration,
    sample_surface,
    triangle_normals,
)
from .usd_binary_read_cache import read_scoped

SCHEMA_VERSION = "native_g1_spawn_clearance.v1"
FILENAME = SCHEMA_VERSION + ".json"
OVERLAP_REASON = "g1_spawn_overlap"
UNMEASURED_REASON = "g1_spawn_clearance_unmeasured"
#: A foot with no floor found under its sole, or a sole above the floor by more than the support tolerance:
#: the robot may tip or drop at spawn. The overlap measure cannot see a missing floor, so this is a flag the
#: row carries (``setup_quality_flags``), never a gate.
UNSUPPORTED_FOOT_REASON = "g1_spawn_foot_unsupported"
#: The scene's floor is a development-only estimate: in the collider copy a flat floor at the uneven scanned
#: band's median height replaced the floor (``atlas_world_levelling`` fallback floor). The overlap measure and
#: the foot support read against that flat floor, so a G1 result there would overstate itself. A flag the row
#: carries (``setup_quality_flags``), never a gate.
FLOOR_ESTIMATED_REASON = "g1_scene_floor_estimated"
#: Money: a stance with no collision floor under either sole, and nothing of the scene near the robot,
#: is refused before any rental. The library job's first scored G1 run (2026-11-08) spawned so on every
#: start offset and fell through the scene in every cell it ran.
FLOOR_ABSENT_REASON = "g1_spawn_floor_absent"
EMPTY_BACKGROUND_NOTICE = "g1_spawn_background_empty_near_robot"
DIAGNOSTIC_SCHEMA_VERSION = "native_g1_spawn_background_diagnostic.v1"
DIAGNOSTIC_FILENAME = DIAGNOSTIC_SCHEMA_VERSION + ".json"
CLAIM_BOUNDARY = ("cpu_geometry_at_the_reset_pose_only: the G1 USD's colliders against sampled static scene "
                  "collision before the first physics step; no simulator ran; nothing about contact after "
                  "spawn, balance, reach or the task")
DISCLOSED_ASSUMPTION = ("G1 spawn clearance: the robot USD's colliders measured on CPU at the reset pose against "
                        "the scene's static collision; a foot up to 2 cm into an upward-facing surface reads as "
                        "standing on it, anything else inside the room as overlap; planning funds no G1 run "
                        "that reads overlap and flags one it could not measure. No simulator ran.")
#: The G1's soles (``official_g1_policy_handoff.FOOT_BODY_NAMES``).
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
#: A sole this far below an upward-facing surface stands on it. The stance puts the pelvis at the G1's
#: nominal 0.79 m (``UNITREE_G1_PROFILE.pelvis_height_m``) above a floor height inferred from the
#: reconstructed mesh (``external_scene_robot_placement._infer_horizontal_support_surface``, in 5 cm
#: bands). Neither is measured from this USD's legs at the reset posture, so a sole that starts a little
#: in the floor is what the estimate gives, a floor contact that pushes the feet up, not a body inside the
#: room. 2 cm is a chosen bound, not a measured one; past it the foot counts like any other body.
FOOT_SUPPORT_TOLERANCE_M = 0.02
#: An upward-facing surface as the floor inference reads one: a triangle normal within about 26 degrees of
#: vertical (either winding).
FLOOR_NORMAL_MIN_Z = 0.9
#: How far above or below a sole an upward-facing surface under it is read as the floor it stands on.
SOLE_SEARCH_M = 0.10
#: Other stances measured when the chosen one overlaps, in batches that share one background read.
MAXIMUM_FALLBACK_STANCES = 32
FALLBACK_BATCH = 8
WORST_BODIES = 5
_CODE = re.compile(r"[a-z][A-Za-z0-9_:.,\-]{1,191}")


def _code(exc: BaseException, fallback: str) -> str:
    return str(exc) if _CODE.fullmatch(str(exc)) else fallback


def _tolerances(spacing: float) -> dict[str, Any]:
    return {"sample_spacing_m": spacing, "overlap_tolerance_m": DEFAULT_REST_TOLERANCE_M,
            "foot_support_tolerance_m": FOOT_SUPPORT_TOLERANCE_M, "foot_bodies": list(FOOT_BODIES)}


def _unmeasured(reason: str, spacing: float, *, error: BaseException | None = None, **extra: Any) -> dict[str, Any]:
    # A read error may pass: such a report is never kept (``retryable``), so the next pass measures again.
    return {"status": "unmeasured", "reason": reason, "retryable": isinstance(error, OSError), "max_depth_m": None,
            "overlapping_points": None, "worst_bodies": [], "clearing_push": None, "foot_support": None,
            "bodies_without_colliders": {"names": [], "count": 0}, "unreadable_colliders": [],
            "skipped_colliders": [], "background_samples": 0, "notices": [], **_tolerances(spacing), **extra}


def _model(robot_usd: str | Path, reset: Mapping[str, float]):
    # A capsule is read as its bounding cylinder; any other shape the measure has no solid for is listed.
    model = load_asset_model(robot_usd, instance_proxies=True, lenient_shapes=True)
    movable = {joint.joint_id for joint in model.joints if joint.kind != "fixed"}
    unmatched = sorted(set(reset) - movable)
    if unmatched:
        # Joints the USD does not have would leave the body at another pose than the run's reset.
        raise ScenePhysicsPreflightError("g1_spawn_reset_joints_unmatched:" + ",".join(unmatched[:3]))
    return model


def _placements(model: Any, reset: Mapping[str, float], pose: Mapping[str, Sequence[float]]):
    # The pose places the articulation's root body (the pelvis), as the run's root state does; with the
    # root body at its default prim's origin the two readings agree.
    world = _pose_matrix(pose) @ np.linalg.inv(model.asset_from_body[model.root_body])
    return list(_placed(model.colliders, {body: world @ matrix for body, matrix in body_poses(model, reset).items()}))


def _near(triangles: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    if not len(triangles):
        return triangles
    centroids = triangles.mean(axis=1)
    reach = np.linalg.norm(triangles - centroids[:, None], axis=2).max(axis=1)
    return triangles[np.linalg.norm(centroids - centre, axis=1) <= radius + reach]


def _measure(placed: list, triangles: np.ndarray, spacing: float, unreadable: list[str]) -> dict[str, Any]:
    """One stance: background inside the body's solids, the feet's floor contact set apart as support."""
    from scipy.spatial import cKDTree

    centres = np.asarray([world[:3, 3] for _, world in placed])
    bounds = np.asarray([collider.bound_radius for collider, _ in placed])
    low, high = (centres - bounds[:, None]).min(axis=0), (centres + bounds[:, None]).max(axis=0)
    near = _near(triangles, (low + high) / 2, float(np.linalg.norm(high - low) / 2))
    near = _clip_large_triangles(_reachable_triangles(near, placed), low, high)
    points, owners = sample_surface(near, spacing, maximum_samples=MAXIMUM_BACKGROUND_SAMPLES)
    names = np.asarray([collider.body.rsplit("/", 1)[-1] for collider, _ in placed] + [""])
    feet = sorted(set(FOOT_BODIES) & set(names.tolist()))
    notices = [] if feet else ["g1_spawn_foot_bodies_absent"]
    # Each collider's box corners in the world: a foot's sole is its lowest one.
    corners = [collider.local_corners() @ world[:3, :3].T + world[:3, 3] for collider, world in placed]
    sole = np.asarray([c[:, 2].min() for c in corners] + [np.nan])
    upward = (np.abs(triangle_normals(near)[owners][:, 2]) >= FLOOR_NORMAL_MIN_Z if len(points)
              else np.zeros(0, dtype=bool))
    if len(points):
        depth, owner = penetration(points, placed, tree=cKDTree(points))
    else:
        notices.append("g1_spawn_background_empty_near_robot")
        depth, owner = np.zeros(0), np.zeros(0, dtype=np.int64)
    # A floor point inside a foot reads by how far the sole is below it, not by its nearest face: a sole
    # sunk past half the foot's thickness is nearer its top face.
    on_foot = np.isin(names[owner], FOOT_BODIES) & upward
    sink = np.where(on_foot, points[:, 2] - sole[owner] if len(points) else 0.0, 0.0)
    support = on_foot & (depth > 0) & (sink <= FOOT_SUPPORT_TOLERANCE_M)
    blocking = np.where(support, -np.inf, np.where(on_foot, np.maximum(depth, sink), depth))
    summary = _summary(points, blocking, owner, placed, DEFAULT_REST_TOLERANCE_M)
    inside = blocking > DEFAULT_REST_TOLERANCE_M
    # A sunk sole is lifted along the floor's normal; the shortest-exit push is for the rest of the body.
    body_inside = inside & ~on_foot
    push = clearing_translation(points, placed, body_inside)
    by_body = {body: dict(row) for body, row in summary["penetration_by_body"].items()}
    soles, sunk = {}, float(sink[inside & on_foot].max()) if np.any(inside & on_foot) else 0.0
    for name in feet:
        # The highest upward-facing surface within reach under the sole: a foot wholly below the floor has
        # no floor point inside it, so the gap catches what the solid test cannot.
        box = np.concatenate([c for c, n in zip(corners, names) if n == name])
        bottom, top = box.min(axis=0), box.max(axis=0)
        under = (upward & np.all((points[:, :2] >= bottom[:2]) & (points[:, :2] <= top[:2]), axis=1)
                 & (np.abs(points[:, 2] - bottom[2]) <= SOLE_SEARCH_M)) if len(points) else upward
        floor = float(points[under, 2].max()) if under.any() else None
        soles[name] = {"sole_z_m": round(float(bottom[2]), 6), "floor_z_m": None if floor is None else round(floor, 6),
                       "gap_m": None if floor is None else round(float(bottom[2]) - floor, 6)}
        if floor is not None and floor - bottom[2] > FOOT_SUPPORT_TOLERANCE_M:
            row = by_body.setdefault(name, {"points": 0, "max_depth_m": 0.0})
            row["max_depth_m"] = max(row["max_depth_m"], floor - float(bottom[2]))
            sunk = max(sunk, floor - float(bottom[2]))
    if sunk and not body_inside.any():
        push = {"distance_m": sunk, "direction_world": [0.0, 0.0, 1.0]}  # Lift the soles onto the floor.
    worst = sorted(({"body": body, **row} for body, row in by_body.items()),
                   key=lambda row: (-row["max_depth_m"], row["body"]))[:WORST_BODIES]
    overlap = bool(inside.any() or sunk)
    # A measured overlap stands whatever else could not be read; a clear reading needs every collider read.
    status = "overlap" if overlap else "unmeasured" if unreadable else "clear"
    return {"status": status,
            "reason": OVERLAP_REASON if overlap else "g1_spawn_colliders_unreadable" if unreadable else None,
            "retryable": False, "max_depth_m": round(max(0.0, summary["max_depth_m"], sunk), 6),
            "overlapping_points": int(inside.sum()),
            "worst_bodies": [{**row, "max_depth_m": round(row["max_depth_m"], 6)} for row in worst],
            "clearing_push": {"distance_m": round(push["distance_m"], 6),
                              "direction_world": [round(v, 6) for v in push["direction_world"]]},
            "penetration_height_range_m": summary["penetration_height_range_m"],
            "foot_support": {"bodies_found": feet, "points": int(support.sum()),
                             "max_sink_m": round(float(sink[support].max()), 6) if support.any() else 0.0,
                             "lift_m": round(sunk, 6), "soles": soles},
            "unreadable_colliders": unreadable, "background_triangles_reached": int(len(near)),
            "background_samples": int(len(points)),
            "notices": notices + [f"{UNSUPPORTED_FOOT_REASON}:{name}" for name in _unsupported(soles)],
            **_tolerances(spacing)}


def _unsupported(soles: Mapping[str, Any]) -> list[str]:
    """The feet with no floor found under the sole, or a sole above the floor beyond the support tolerance."""
    out = []
    for name, sole in sorted(soles.items()):
        gap = sole.get("gap_m") if isinstance(sole, Mapping) else None
        if (not isinstance(sole, Mapping) or sole.get("floor_z_m") is None
                or not isinstance(gap, (int, float)) or gap > FOOT_SUPPORT_TOLERANCE_M):
            out.append(name)
    return out


def unsupported_feet(report: Mapping[str, Any]) -> list[str]:
    """The report's unsupported feet, from its sole readings (so a report written before the flag counts)."""
    soles = (report.get("foot_support") or {}).get("soles") if isinstance(report.get("foot_support"), Mapping) else None
    return _unsupported(soles) if isinstance(soles, Mapping) else []


def floor_absent(report: Mapping[str, Any] | None) -> bool:
    """Whether a stance's report found no floor under either sole (``unsupported_feet``, both with no floor)."""
    if not isinstance(report, Mapping):
        return False
    soles = (report.get("foot_support") or {}).get("soles") if isinstance(report.get("foot_support"), Mapping) else None
    return (isinstance(soles, Mapping) and len(soles) >= 2
            and all(isinstance(sole, Mapping) and sole.get("floor_z_m") is None for sole in soles.values()))


def background_diagnostic(background_usd: str | Path, *, spawn_xy: Sequence[float],
                          background_pose_world: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Aggregate facts about a packet's scene collision as the spawn check loads it; never mesh bytes.

    For each mesh (by index, never by name): whether it carries an enabled collision API, its face count,
    the scale of its world transform and its world bounds. Over the collision meshes: the faces whose xy
    bounds come within 1 and 2 m of the spawn, a histogram of their mid heights, and the faces over the
    spawn point itself with their height range (what could hold the robot up).
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    from .scene_physics_preflight import _usd_matrix
    from .usd_binary_read_cache import readable

    stage = Usd.Stage.Open(readable(background_usd))
    world_from_stage = None if background_pose_world is None else _pose_matrix(background_pose_world)
    cache, spawn = UsdGeom.XformCache(), np.asarray(spawn_xy[:2], dtype=float)
    meshes, near_one, near_two, heights, under = [], 0, 0, [], []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        collision = bool(prim.HasAPI(UsdPhysics.CollisionAPI)
                         and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False)
        points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=float).reshape(-1, 3)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
        matrix = _usd_matrix(cache, prim)
        if world_from_stage is not None:
            matrix = world_from_stage @ matrix
        world = points @ matrix[:3, :3].T + matrix[:3, 3] if len(points) else points
        row = {"index": len(meshes), "collision_enabled": collision, "faces": int(len(counts)),
               "transform_scale": [round(float(v), 6) for v in np.linalg.norm(matrix[:3, :3], axis=0)],
               "world_bounds_m": ({"minimum": [round(float(v), 4) for v in world.min(axis=0)],
                                   "maximum": [round(float(v), 4) for v in world.max(axis=0)]} if len(world) else None)}
        meshes.append(row)
        if collision and len(world) and len(counts):
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            faces = [world[indices[start:start + count]] for start, count in zip(starts, counts) if count >= 3]
            if not faces:
                continue
            low = np.stack([face.min(axis=0) for face in faces])
            high = np.stack([face.max(axis=0) for face in faces])
            # Distance in xy from the spawn to each face's bounds: a large floor face counts where it lies.
            gap = np.maximum(np.maximum(low[:, :2] - spawn, spawn - high[:, :2]), 0.0)
            distance = np.linalg.norm(gap, axis=1)
            near_one += int((distance <= 1.0).sum())
            near_two += int((distance <= 2.0).sum())
            heights.extend(float(z) for z in ((low[:, 2] + high[:, 2]) / 2)[distance <= 2.0])
            under.extend((float(a), float(b)) for a, b in zip(low[distance == 0.0, 2], high[distance == 0.0, 2]))
    histogram = None
    if heights:
        counts, edges = np.histogram(np.asarray(heights), bins=np.arange(-1.0, 3.01, 0.25))
        histogram = {"edges_m": [round(float(v), 2) for v in edges], "faces": [int(v) for v in counts],
                     "below": int(sum(h < -1.0 for h in heights)), "above": int(sum(h > 3.0 for h in heights))}
    value = {"schema_version": DIAGNOSTIC_SCHEMA_VERSION, "spawn_xy_m": [float(v) for v in spawn],
             "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
             "stage_meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
             "meshes": meshes, "collision_faces_within_1m_xy": near_one, "collision_faces_within_2m_xy": near_two,
             "collision_face_height_histogram_within_2m": histogram,
             "collision_faces_over_spawn_xy": len(under),
             "collision_height_range_over_spawn_m": ([round(min(a for a, _ in under), 4), round(max(b for _, b in under), 4)]
                                                     if under else None),
             "claim_boundary": "aggregate_geometry_counts_only: no vertex, face or name leaves the host"}
    value["report_digest"] = canonical_digest(value, digest_field="report_digest")
    return value


def ensure_background_diagnostic(packet_dir: str | Path, plan_digest: str) -> dict[str, Any] | None:
    """Write the packet's background diagnostic once when its spawn report found no floor; never raises."""
    from .deployment_embodiment_preparation_contract import write_once

    root = Path(packet_dir)
    path = root / DIAGNOSTIC_FILENAME
    try:
        if path.is_file() and not path.is_symlink():
            return json.loads(path.read_text(encoding="utf-8"))
        report = _plan_report(root, plan_digest)
        if not floor_absent(report):
            return None
        plan = json.loads((root / "native_task_arena_scene_plan.v1.json").read_text(encoding="utf-8"))
        collisions = [row for row in plan["objects"] if row.get("semantic_role") == "scene_collision"]
        background = _bound_file(root, collisions[0]["usd_path"], collisions[0]["sha256"])
        value = background_diagnostic(background, spawn_xy=report["base_pose_world"]["position_world_m"],
                                      background_pose_world=collisions[0].get("pose_world"))
        write_once(path, value)
        return value
    except Exception:  # noqa: BLE001 - a diagnostic never blocks planning
        return None


def _measure_stances(model: Any, reset: Mapping[str, float], poses: Sequence[Mapping[str, Any]],
                     background_usd: str | Path, background_pose_world: Mapping[str, Any] | None,
                     spacing: float) -> list[dict[str, Any]]:
    """Each pose's measurement, the background read once for all of them.

    A body with no collider prim has no contact in the engine either: it is listed and skipped.
    """
    covered = {collider.body for collider in model.colliders}
    unreadable = list(model.unreadable_colliders)
    covered |= {body for body in model.asset_from_body for path in unreadable if path.startswith(body + "/")}
    bare = sorted(body.rsplit("/", 1)[-1] for body in model.asset_from_body if body not in covered)
    common = {"skipped_colliders": list(model.skipped_colliders), "unreadable_colliders": unreadable,
              "bodies_without_colliders": {"names": bare, "count": len(bare)}}
    if not model.colliders:  # Nothing measured: none at all, or none readable.
        reason = "g1_spawn_colliders_unreadable" if unreadable else "g1_spawn_robot_colliders_absent"
        return [_unmeasured(reason, spacing, **common) for _ in poses]
    placements = [_placements(model, reset, pose) for pose in poses]
    centres = np.asarray([world[:3, 3] for placed in placements for _, world in placed])
    centre = centres.mean(axis=0)
    radius = float(np.linalg.norm(centres - centre, axis=1).max() + max(c.bound_radius for c in model.colliders))
    try:
        triangles = load_background_triangles(
            background_usd, centre=centre, radius=radius,
            world_from_stage=None if background_pose_world is None else _pose_matrix(background_pose_world))
    except Exception as exc:  # noqa: BLE001 - an unreadable room is unmeasured, never a failed preparation
        return [_unmeasured(_code(exc, "g1_spawn_background_unreadable"), spacing, error=exc, **common)
                for _ in poses]
    rows = []
    for placed in placements:
        try:
            rows.append({**_measure(placed, triangles, spacing, unreadable), **common})
        except Exception as exc:  # noqa: BLE001 - e.g. more background samples than one check may hold
            rows.append(_unmeasured(_code(exc, "g1_spawn_measurement_failed"), spacing, error=exc, **common))
    return rows


def choose_spawn_stance(*, robot_usd: str | Path, joint_reset_positions: Mapping[str, float],
                        stances: Sequence[Mapping[str, Any]], chosen: int, fallback_order: Sequence[int],
                        background_usd: str | Path, background_pose_world: Mapping[str, Any] | None = None,
                        sample_spacing_m: float = DEFAULT_SAMPLE_SPACING_M) -> tuple[int, dict[str, Any]]:
    """The stance to spawn at and its measurement: ``chosen``, unless its body overlaps the room.

    Then the first of ``fallback_order`` (at most ``MAXIMUM_FALLBACK_STANCES``) that measures clear;
    with none, ``chosen`` and its overlap. Each stance is a pose of the root body
    (``position_world_m``, ``orientation_xyzw``). Never raises on unreadable geometry: that stance is
    ``unmeasured`` with its reason, and only a clear one replaces ``chosen``.
    """
    reset = {str(name): float(value) for name, value in joint_reset_positions.items()}
    selection = {"source": "stance_search", "search_rank_chosen": chosen, "search_rank_used": chosen,
                 "fallback_used": False, "fallback_stances_measured": 0}
    try:
        model = _model(robot_usd, reset)
    except Exception as exc:  # noqa: BLE001 - USD errors (Tf.ErrorException) included: unmeasured, by name
        return chosen, {**_unmeasured(_code(exc, "g1_spawn_robot_model_unreadable"), sample_spacing_m, error=exc),
                        "stance_selection": selection}

    def measure(indices: Sequence[int]) -> list[dict[str, Any]]:
        return _measure_stances(model, reset, [stances[i] for i in indices], background_usd,
                                background_pose_world, sample_spacing_m)

    [first] = measure([chosen])
    if first["status"] != "overlap":
        return chosen, {**first, "stance_selection": selection}
    order = [index for index in fallback_order if index != chosen][:MAXIMUM_FALLBACK_STANCES]
    for start in range(0, len(order), FALLBACK_BATCH):
        batch = order[start:start + FALLBACK_BATCH]
        rows = measure(batch)
        selection["fallback_stances_measured"] += len(batch)
        used = next((pair for pair in zip(batch, rows) if pair[1]["status"] == "clear"), None)
        if used is not None:
            selection.update(search_rank_used=used[0], fallback_used=True,
                             chosen_overlap={key: first[key] for key in ("max_depth_m", "worst_bodies", "clearing_push")})
            return used[0], {**used[1], "stance_selection": selection}
    return chosen, {**first, "stance_selection": selection}


def measure_poses(*, robot_usd: str | Path, joint_reset_positions: Mapping[str, float],
                  poses: Sequence[Mapping[str, Any]], background_usd: str | Path,
                  background_pose_world: Mapping[str, Any] | None = None,
                  sample_spacing_m: float = DEFAULT_SAMPLE_SPACING_M) -> list[dict[str, Any]]:
    """The spawn check at each root pose, in order, the robot and the background read once; never raises.

    Each report is what ``choose_spawn_stance`` measures for one stance, without a selection: a navigation
    packet's start offsets are each measured, none replaces another.
    """
    reset = {str(name): float(value) for name, value in joint_reset_positions.items()}
    try:
        model = _model(robot_usd, reset)
    except Exception as exc:  # noqa: BLE001 - USD errors (Tf.ErrorException) included: unmeasured, by name
        return [_unmeasured(_code(exc, "g1_spawn_robot_model_unreadable"), sample_spacing_m, error=exc) for _ in poses]
    return _measure_stances(model, reset, list(poses), background_usd, background_pose_world, sample_spacing_m)


#: A navigation packet's spawn check at each start offset (``native_g1_beta_authority.NAVIGATION_START_OFFSETS``),
#: kept beside the packet. Planning funds no cell whose offset reads overlap and flags the rest as the packet's
#: own report is flagged (``beta_policy_screening_g1.navigation_start_offsets``).
START_OFFSETS_SCHEMA_VERSION = "native_g1_navigation_start_offsets.v1"
START_OFFSETS_FILENAME = START_OFFSETS_SCHEMA_VERSION + ".json"


def packet_start_offsets(packet_dir: str | Path, *,
                         sample_spacing_m: float = DEFAULT_SAMPLE_SPACING_M) -> dict[str, Any]:
    """Measure a navigation packet's G1 at each start offset of its plan's own robot pose, sealed to that plan."""
    from .native_g1_beta_authority import NAVIGATION_START_OFFSETS, apply_start_offset, start_offset_parameters

    root = Path(packet_dir)
    plan = json.loads((root / "native_task_arena_scene_plan.v1.json").read_text(encoding="utf-8"))
    robot = plan["robot"]
    collisions = [row for row in plan.get("objects") or [] if row.get("semantic_role") == "scene_collision"]
    if len(collisions) != 1:
        raise ScenePhysicsPreflightError("g1_spawn_packet_scene_collision_count_invalid")
    reset = robot["joint_reset_positions_rad"]
    poses = [apply_start_offset(robot["base_pose_world"], start_offset_parameters(ordinal))
             for ordinal in range(len(NAVIGATION_START_OFFSETS))]
    try:
        robot_usd = _bound_file(root, robot["usd_path"], robot["usd_sha256"])
        background = _bound_file(root, collisions[0]["usd_path"], collisions[0]["sha256"])
    except ScenePhysicsPreflightError as exc:
        reports = [_unmeasured(str(exc), sample_spacing_m) for _ in poses]
    else:
        reports = measure_poses(robot_usd=robot_usd, joint_reset_positions=reset, poses=poses,
                                background_usd=background, background_pose_world=collisions[0].get("pose_world"),
                                sample_spacing_m=sample_spacing_m)
    rows = []
    for (offset_id, forward, lateral), pose, report in zip(NAVIGATION_START_OFFSETS, poses, reports, strict=True):
        rows.append({**report, "offset_id": offset_id, "robot_start_forward_m": forward,
                     "robot_start_lateral_m": lateral, "unsupported_feet": unsupported_feet(report),
                     **binding(robot_usd_sha256=robot["usd_sha256"], scene_collision_sha256=collisions[0]["sha256"],
                               base_pose_world=pose, joint_reset_positions=reset)})
    value = {"schema_version": START_OFFSETS_SCHEMA_VERSION, "scene_plan_digest": plan["plan_digest"],
             "offsets": rows, "claim_boundary": CLAIM_BOUNDARY, "simulator_executed": False,
             "provider_mutation_performed": False}
    value["report_digest"] = canonical_digest(value, digest_field="report_digest")
    return value


@read_scoped
def ensure_packet_start_offsets(packet_dir: str | Path) -> dict[str, Any]:
    """The packet's per-offset spawn report, measured and written beside it once.

    A report already there is returned only while planning's own reader accepts it for the packet's plan
    (``start_offset_reports``: schema, digest, plan and offsets). Any other (changed, another plan's or
    unreadable) is set aside in the packet's directory, as G1 prepare keeps an incomplete packet
    (``beta_policy_screening_g1_prepare._preserve_incomplete``; a link is refused), and measured again. A
    report in which a read error left an offset unmeasured is returned unwritten, so a later call measures again.
    """
    from .deployment_embodiment_preparation_contract import write_once

    root = Path(packet_dir)
    path = root / START_OFFSETS_FILENAME
    if path.exists() or path.is_symlink():
        plan = json.loads((root / "native_task_arena_scene_plan.v1.json").read_text(encoding="utf-8"))
        current = _start_offsets_report(root, plan["plan_digest"])
        if current is not None:
            return current
        from .beta_policy_screening_g1_prepare import _preserve_incomplete

        _preserve_incomplete(path)
    value = packet_start_offsets(root)
    if not any(row["retryable"] for row in value["offsets"]):
        write_once(path, value)
    return value


def _start_offsets_report(packet_dir: str | Path, plan_digest: str) -> dict[str, Any] | None:
    """The packet's sealed per-offset report of this plan, its offsets in table order; None for anything else."""
    from .native_g1_beta_authority import NAVIGATION_START_OFFSETS

    path = Path(packet_dir) / START_OFFSETS_FILENAME
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if not path.is_symlink() else None
    except (OSError, ValueError):
        return None
    if (not isinstance(value, dict) or value.get("schema_version") != START_OFFSETS_SCHEMA_VERSION
            or value.get("report_digest") != canonical_digest(value, digest_field="report_digest")
            or value.get("scene_plan_digest") != plan_digest or not isinstance(value.get("offsets"), list)
            or [row.get("offset_id") if isinstance(row, dict) else None for row in value["offsets"]]
            != [row[0] for row in NAVIGATION_START_OFFSETS]):
        return None
    return value


def start_offset_reports(packet_dir: str | Path, plan_digest: str) -> list[dict[str, Any]] | None:
    """The packet's sealed per-offset reports of this plan, one per offset in table order; None for anything else."""
    value = _start_offsets_report(packet_dir, plan_digest)
    return None if value is None else value["offsets"]


def seal(report: Mapping[str, Any]) -> dict[str, Any]:
    """The report as the packet keeps it: schema, claim boundary and digest."""
    value = {**dict(report), "schema_version": SCHEMA_VERSION, "claim_boundary": CLAIM_BOUNDARY,
             "simulator_executed": False, "provider_mutation_performed": False}
    value["report_digest"] = canonical_digest(value, digest_field="report_digest")
    return value


def binding(*, robot_usd_sha256: str, scene_collision_sha256: str, base_pose_world: Mapping[str, Any],
            joint_reset_positions: Mapping[str, float]) -> dict[str, Any]:
    """What a measurement was taken of; the packet's plan digest is added once the packet exists."""
    return {"robot_usd_sha256": robot_usd_sha256, "scene_collision_sha256": scene_collision_sha256,
            "base_pose_world": {key: [float(v) for v in base_pose_world[key]]
                                for key in ("position_world_m", "orientation_xyzw")},
            "joint_reset_positions_digest": canonical_digest(dict(joint_reset_positions))}


def _bound_file(root: Path, relative: Any, sha256: Any) -> Path:
    path = Path(str(relative or ""))
    path = (path if path.is_absolute() else root / path).resolve()
    if root.resolve() not in path.parents or not path.is_file() or _file_sha256(path) != sha256:
        raise ScenePhysicsPreflightError("g1_spawn_packet_file_unbound")
    return path


def packet_spawn_clearance(packet_dir: str | Path, *, sample_spacing_m: float = DEFAULT_SAMPLE_SPACING_M) -> dict[str, Any]:
    """Measure a prepared G1 packet's own stance and reset: its plan's robot against its scene collision."""
    root = Path(packet_dir)
    plan = json.loads((root / "native_task_arena_scene_plan.v1.json").read_text(encoding="utf-8"))
    robot = plan["robot"]
    collisions = [row for row in plan.get("objects") or [] if row.get("semantic_role") == "scene_collision"]
    if len(collisions) != 1:
        raise ScenePhysicsPreflightError("g1_spawn_packet_scene_collision_count_invalid")
    reset = robot["joint_reset_positions_rad"]
    bound = binding(robot_usd_sha256=robot["usd_sha256"], scene_collision_sha256=collisions[0]["sha256"],
                    base_pose_world=robot["base_pose_world"], joint_reset_positions=reset)
    try:
        robot_usd = _bound_file(root, robot["usd_path"], robot["usd_sha256"])
        background = _bound_file(root, collisions[0]["usd_path"], collisions[0]["sha256"])
    except ScenePhysicsPreflightError as exc:
        report = _unmeasured(str(exc), sample_spacing_m)
    else:
        _, report = choose_spawn_stance(robot_usd=robot_usd, joint_reset_positions=reset,
                                        stances=[robot["base_pose_world"]], chosen=0, fallback_order=[],
                                        background_usd=background,
                                        background_pose_world=collisions[0].get("pose_world"),
                                        sample_spacing_m=sample_spacing_m)
    report["stance_selection"] = {"source": "packet_robot_base_pose", "fallback_used": False,
                                  "fallback_stances_measured": 0}
    return seal({**report, **bound, "scene_plan_digest": plan["plan_digest"]})


@read_scoped
def ensure_packet_spawn_clearance(packet_dir: str | Path) -> dict[str, Any]:
    """The packet's spawn report, measured and written beside it once (a packet prepared before the check).

    A report a read error left unmeasured is returned unwritten, so a later call measures again.
    """
    from .deployment_embodiment_preparation_contract import write_once

    path = Path(packet_dir) / FILENAME
    if path.is_file() and not path.is_symlink():
        return json.loads(path.read_text(encoding="utf-8"))
    report = packet_spawn_clearance(packet_dir)
    if not report["retryable"]:
        write_once(path, report)
    return report


def _plan_report(packet_dir: str | Path, plan_digest: str) -> dict[str, Any] | None:
    """The packet's sealed report of this plan that reads clear or overlap; None for anything else."""
    path = Path(packet_dir) / FILENAME
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if not path.is_symlink() else None
    except (OSError, ValueError):
        return None
    if (not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION
            or value.get("report_digest") != canonical_digest(value, digest_field="report_digest")
            or value.get("scene_plan_digest") != plan_digest or value.get("status") not in ("clear", "overlap")):
        return None
    return value


def packet_floor_absent(packet_dir: str | Path, plan_digest: str) -> bool:
    """Whether the packet's own sealed spawn report of this plan found no floor under either sole."""
    return floor_absent(_plan_report(packet_dir, plan_digest))


def spawn_setup_flags(packet_dir: str | Path, plan_digest: str) -> list[str]:
    """The flags a funded row carries from its packet's spawn check: unmeasured, or an unsupported foot."""
    report = _plan_report(packet_dir, plan_digest)
    if report is None:
        return [UNMEASURED_REASON]
    return [UNSUPPORTED_FOOT_REASON] if unsupported_feet(report) else []


def scene_floor_estimate(task_spec: Mapping[str, Any]) -> dict[str, Any] | None:
    """The floor estimate a packet's task carries from its scene's levelling, or None when it carries none.

    A navigation task embeds its compiled spec, whose metric-scale evidence carries the base scene's
    levelling (``website_locomotion_task.scene_facts``): ``levelling_floor_estimate``, with the estimate's
    ``claim``, ``basis`` and ``band_spread_m``, exactly when the floor is a flat stand-in. A rigid task's
    packet carries no levelling record, so it reads None.
    """
    compiled = task_spec.get("compiled_task_spec") if isinstance(task_spec, Mapping) else None
    scale = compiled.get("metric_scale") if isinstance(compiled, Mapping) else None
    evidence = scale.get("evidence") if isinstance(scale, Mapping) else None
    for row in evidence if isinstance(evidence, list) else []:
        estimate = row.get("levelling_floor_estimate") if isinstance(row, Mapping) else None
        if isinstance(estimate, Mapping):
            return dict(estimate)
    return None


def packet_floor_estimate(packet_dir: str | Path, plan_digest: str) -> dict[str, Any] | None:
    """The floor estimate the packet's own plan of ``plan_digest`` carries (``scene_floor_estimate``), else None."""
    path = Path(packet_dir) / "native_task_arena_scene_plan.v1.json"
    try:
        plan = json.loads(path.read_text(encoding="utf-8")) if not path.is_symlink() else None
    except (OSError, ValueError):
        return None
    if not isinstance(plan, dict) or plan.get("plan_digest") != plan_digest:
        return None
    return scene_floor_estimate(plan.get("task_spec") or {})


def floor_estimate_assumption(estimate: Mapping[str, Any]) -> str:
    """The disclosed assumption a G1 preparation records for a scene floor that is an estimate."""
    spread = estimate.get("band_spread_m")
    try:
        low, high = (float(value) for value in spread)
        band = f"the scanned floor band spans {low:+.3f} m to {high:+.3f} m about that height (5th to 95th percentile)"
    except (TypeError, ValueError):
        band = "the scanned floor band's spread is not recorded"
    return (f"Scene floor: {estimate.get('claim')} ({estimate.get('basis')}). In the scene collision a flat floor "
            f"at the uneven scanned floor's median height stands in for it; {band}. The spawn check, each start "
            "offset and the foot support read against that flat floor, so a clear or supported reading does not "
            "describe the floor as scanned.")


def spawn_clearance_status(packet_dir: str | Path, plan_digest: str) -> str:
    """What planning reads for this packet: ``clear`` or ``overlap`` from a sealed report of the plan.

    Anything else (no report, another plan's, a changed one, or one that could not measure) is
    ``unmeasured``. Only ``overlap`` refuses money (``OVERLAP_REASON``); ``unmeasured`` travels with
    the row and its result as ``UNMEASURED_REASON`` in ``setup_quality_flags``.
    """
    report = _plan_report(packet_dir, plan_digest)
    return "unmeasured" if report is None else report["status"]


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="CPU spawn check of a prepared native G1 packet; no simulator.")
    parser.add_argument("--packet", required=True, help="prepared native G1 packet directory")
    parser.add_argument("--write", action="store_true", help=f"write {FILENAME} beside the packet when it has none")
    args = parser.parse_args(argv)
    try:
        report = ensure_packet_spawn_clearance(args.packet) if args.write else packet_spawn_clearance(args.packet)
    except (ScenePhysicsPreflightError, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "refused", "error": _code(exc, type(exc).__name__)}))
        return 1
    print(json.dumps({key: report.get(key) for key in ("status", "reason", "max_depth_m", "worst_bodies",
                                                         "clearing_push", "unreadable_colliders",
                                                         "bodies_without_colliders", "report_digest")},
                     sort_keys=True))
    return {"clear": 0, "overlap": 2}.get(report["status"], 3)


if __name__ == "__main__":
    raise SystemExit(main())
