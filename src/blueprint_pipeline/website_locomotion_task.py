"""Compile a website job whose robot moves itself, and hold it only on a named missing piece.

A job whose resolved profile needs locomotion and no manipulation (for example
``goal_region_navigation``) keeps its scene as filmed: nothing is removed,
authored or registered as a subject. The handoff compiles its task spec from the
preserved scene's own facts (``task_spec_compiler``):

- the floor from the levelled collider;
- the route the camera walked, as the start and goal when the brief gives no anchor;
- every non-floor surface as an obstacle the robot must not touch;
- Atlas's scale estimate, recorded as not independently validated.

The goal is pulled back along the filmed route until a robot standing there
clears every obstacle. Each guess is flagged.

Robot and policy choice then read the spec's ``requirements``, never its kind:
bodies that can do the job (``EmbodimentProfile.meets``), and the catalog's
enabled policies that run on those bodies and serve the requirement. When none
can run here, the job holds with ``website_task_runnable_policy_missing``, which
names each body, each policy and each missing piece. The hold keeps the job;
nothing is discarded or rewritten.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .website_locomotion_hold import HOLD_CODE, runnable_policy_gap_from_options

FACTS_SCHEMA = "website_locomotion_scene_facts.v1"
SPEC_FILE = "compiled_task_spec.json"
#: Clearance a standing robot needs from any obstacle when its body declares no footprint.
ROBOT_CLEARANCE_RADIUS_M = 0.35
#: Obstacles are surfaces a walking body can touch: above the floor band, below head height.
OBSTACLE_BAND_M = (0.05, 1.9)
#: The site's route is the filmed one, end to end. Clearing an anchor of obstacles may move it at most this
#: share of the filmed path's length. capture-5b067498 (2026-11-09): the first clear point lay 2.7 m down a
#: 3.3 m walk, half a metre from the goal, and a run there would have scored a quarter-metre shuffle. Beyond
#: the share, the anchor is the best-cleared filmed point inside it, named and flagged: the task stays the
#: site's, and the robot's own spawn check decides whether it can stand there.
ANCHOR_SEARCH_FRACTION = 0.25
NEAR_START = "filmed_path_near_start_clearance_below_robot_radius"
NEAR_END = "filmed_path_near_end_clearance_below_robot_radius"
#: How much room each filmed point has, heights only, written beside the compiled spec (``route_record``).
ROUTE_FILE = "route_clearance.json"
ROUTE_SCHEMA = "website_locomotion_route_clearance.v1"
ROUTE_POINTS = 64
#: The Pipeline pieces a locomotion policy still needs beyond a registered body and policy.
NAVIGATION_SCENE_PACKET = "navigation_scene_packet_from_compiled_spec"
#: The pieces a G1 navigation run needs beyond the compiled spec, in build order
#: (docs/G1_NAVIGATION_FROM_COMPILED_SPEC.md). A hold names each one not yet built, so it
#: shrinks as each lands and never claims a route that does not exist.
NAVIGATION_PIPELINE_PIECES = (
    "navigation_task_spec_from_compiled_spec",   # website_navigation_packet (built)
    "navigation_scene_assets",                    # Isaac appearance/collision for a scene kept as filmed
    "arena_navigation_scene_runtime",             # scene plan, runtime and readback without a task object
    "g1_navigation_preparation",                  # G1 stance at the compiled start, its nominal cell, registration
    "navigation_source_identity",                 # this job's website intake receipt (website_navigation_intake)
)
#: The screening source a navigation scene half binds (``website_navigation_scene_inputs.screening_source``)
#: is the scene intent of the job's website navigation intake, bound by the intake receipt beside its record
#: (``website_navigation_intake``). The piece is built for a job only once that job's receipt exists, so the
#: hold names it for every job without one (no intake yet, or one the website refused).
SOURCE_IDENTITY_PIECE = "navigation_source_identity"
#: What the G1 lane itself still adds (its perturbed start cells, planning and launch) shows as the
#: catalog rows' own task-kind support (``_catalog_task_kind_gap``) and the embodiment's missing work.
#: The pieces whose code exists. One that stands for a job's own output (``NAVIGATION_PIECE_OUTPUTS``)
#: is built for a job only once its handoff wrote that output (``navigation_pieces_built``).
#: ``arena_navigation_scene_runtime`` is code alone: the arena runs a scene with no task object (part B,
#: #3431), for every job; the scene plan and runtime contract a job compiles are its scene half's.
BUILT_NAVIGATION_PIECES = frozenset({"navigation_task_spec_from_compiled_spec", "navigation_scene_assets",
                                     "arena_navigation_scene_runtime", "g1_navigation_preparation",
                                     SOURCE_IDENTITY_PIECE})
#: Each job-made piece and the handoff output it stands for (``website_scene_handoff``):
#: ``navigation_task_spec.json``, ``navigation_scene_assets.json``, the scene-inputs record
#: (``website_navigation_scene_inputs``: the G1 at the compiled start) with its team registry row, and
#: that record's website intake receipt (``website_navigation_intake``).
NAVIGATION_PIECE_OUTPUTS = {"navigation_task_spec_from_compiled_spec": "navigation_task_spec",
                            "navigation_scene_assets": "navigation_scene_assets",
                            "g1_navigation_preparation": "navigation_scene_inputs",
                            SOURCE_IDENTITY_PIECE: "navigation_intake_receipt"}
NAVIGATION_TASK_FILE = "navigation_task_spec.json"


def navigation_pieces_built(outputs: Mapping[str, bool]) -> frozenset[str]:
    """What exists for one job: each built piece its handoff wrote the output of, and the code-only ones.

    The source identity counts only for a job whose own intake receipt exists (``SOURCE_IDENTITY_PIECE``).
    """
    return frozenset(piece for piece in BUILT_NAVIGATION_PIECES
                     if piece not in NAVIGATION_PIECE_OUTPUTS or outputs.get(NAVIGATION_PIECE_OUTPUTS[piece]) is True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _stage(points: np.ndarray) -> np.ndarray:
    """Levelled Y-up -> collision-stage Z-up, the transform the collider compiler uses."""
    return np.column_stack((points[:, 0], -points[:, 2], points[:, 1]))


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", points - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
    return np.linalg.norm(points - (a + ab * t[:, None]), axis=1)


def _obstacle_distances(point_xy: Sequence[float], triangles_xy: np.ndarray) -> np.ndarray:
    """Horizontal distance from a point to each obstacle triangle (0 inside one)."""
    p = np.broadcast_to(np.asarray(point_xy, dtype=np.float64), (len(triangles_xy), 2))
    a, b, c = triangles_xy[:, 0], triangles_xy[:, 1], triangles_xy[:, 2]

    def cross(u, v, w):
        return (v[:, 0] - u[:, 0]) * (w[:, 1] - u[:, 1]) - (v[:, 1] - u[:, 1]) * (w[:, 0] - u[:, 0])

    d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
    # A vertical face projects to a segment (zero area): every point on its line would pass the
    # sign test, so such a triangle is never "inside" and its edge distance stands.
    area = np.abs(cross(a, b, c))
    inside = (area > 1e-12) & ~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))
    edges = np.minimum(np.minimum(_segment_distance(p, a, b), _segment_distance(p, b, c)), _segment_distance(p, c, a))
    return np.where(inside, 0.0, edges)


def obstacle_distance(point_xy: Sequence[float], triangles_xy: np.ndarray) -> float:
    """Horizontal distance from a point to the nearest obstacle triangle (0 inside one)."""
    if len(triangles_xy) == 0:
        return math.inf
    return float(np.min(_obstacle_distances(point_xy, triangles_xy)))


def _along(path: Sequence[Sequence[float]]) -> list[float]:
    points = np.asarray([row[:2] for row in path], dtype=np.float64).reshape(-1, 2)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1) if len(points) > 1 else np.zeros(0)
    return [0.0, *np.cumsum(steps).tolist()]


def route_anchors(path: Sequence[Sequence[float]], obstacles_xy: np.ndarray,
                  clearance_m: float = ROBOT_CLEARANCE_RADIUS_M) -> dict[str, Any]:
    """Where the route starts and ends when a filmed end is not clear of obstacles; {} when both are.

    The first (last) filmed point a robot can stand at clear, when it lies within
    ``ANCHOR_SEARCH_FRACTION`` of the path's length from that end. Otherwise the best-cleared
    filmed point within that share, named ``NEAR_START`` / ``NEAR_END``: the route stays the filmed one.
    """
    points = [list(point) for point in path]
    if len(points) < 2:
        return {}
    cleared = "filmed_path_cleared_of_obstacles"
    along = _along(points)
    reach = ANCHOR_SEARCH_FRACTION * along[-1]
    room = [obstacle_distance(point[:2], obstacles_xy) for point in points]
    anchors: dict[str, Any] = {}
    first = next((index for index, value in enumerate(room) if value >= clearance_m), None)
    if first is None or along[first] > reach:
        near = [index for index in range(len(points)) if along[index] <= reach] or [0]
        best = max(near, key=lambda index: (room[index], -index))
        anchors.update(start_world_m=points[best], start_source=NEAR_START)
    elif first:
        anchors.update(start_world_m=points[first], start_source=cleared)
    last = next((index for index in range(len(points) - 1, -1, -1) if room[index] >= clearance_m), None)
    if last is None or along[-1] - along[last] > reach:
        near = [index for index in range(len(points)) if along[-1] - along[index] <= reach] or [len(points) - 1]
        best = max(near, key=lambda index: (room[index], index))
        anchors.update(goal_world_m=points[best], goal_source=NEAR_END)
    elif last != len(points) - 1:
        anchors.update(goal_world_m=points[last], goal_source=cleared)
    return anchors


def route_record(facts: Mapping[str, Any]) -> dict[str, Any]:
    """How much room each filmed point has, and where the anchors fell; heights only, evidence, never a gate.

    Per point (at most ``ROUTE_POINTS``, evenly taken): the distance along the path, the horizontal
    distance to the nearest obstacle, and that obstacle's height range above the floor.
    """
    path = [list(point) for point in facts["filmed_path_world_m"]]
    obstacles, heights = facts.get("_obstacles_xy"), facts.get("_obstacles_z")
    value: dict[str, Any] = {"schema_version": ROUTE_SCHEMA, "facts_digest": facts["facts_digest"],
                             "robot_clearance_radius_m": ROBOT_CLEARANCE_RADIUS_M,
                             "anchor_search_fraction": ANCHOR_SEARCH_FRACTION,
                             "obstacle_band_m": list(OBSTACLE_BAND_M), "points": []}
    if obstacles is None or len(path) < 2:
        return value
    along = _along(path)
    taken = sorted({int(round(index)) for index in np.linspace(0, len(path) - 1, min(len(path), ROUTE_POINTS))})
    for index in taken:
        row: dict[str, Any] = {"along_m": round(along[index], 3), "clearance_m": None}
        if len(obstacles):
            distances = _obstacle_distances(path[index][:2], obstacles)
            nearest = int(np.argmin(distances))
            row["clearance_m"] = round(float(distances[nearest]), 3)
            if heights is not None:
                row.update(nearest_low_m=round(float(heights[nearest][0]), 3),
                           nearest_high_m=round(float(heights[nearest][1]), 3))
        value["points"].append(row)
    anchors = route_anchors(path, obstacles)
    value.update(path_length_m=round(along[-1], 3), obstacle_triangle_count=int(len(obstacles)),
                 start_source=anchors.get("start_source", "filmed_path_start"),
                 goal_source=anchors.get("goal_source", "filmed_path_endpoint"))
    return value


def scene_facts(*, base_scene: Mapping[str, Any], camera_path_levelled_m: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Floor, obstacles and the filmed route of a preserved scene, in stage metres (Z up)."""
    from .external_scene_collision_candidate import _flatten_glb
    from .external_scene_robot_placement import _infer_horizontal_support_surface, triangle_measures

    glb = Path(base_scene["collision_mesh_path"])
    if glb.is_symlink() or _sha256(glb) != base_scene["collision_mesh_digest"]:
        raise ValueError("website_locomotion_collider_changed")
    mpu = float(base_scene["meters_per_unit"])
    vertices, faces, _ = _flatten_glb(glb)
    stage = _stage(np.asarray(vertices, dtype=np.float64)) * mpu
    faces = np.asarray(faces, dtype=np.int64)
    measures = triangle_measures(stage, faces)
    floor_z, _, surface = _infer_horizontal_support_surface(stage_vertices=stage, faces=faces, measures=measures,
                                                           return_support_triangles=False)
    corners = stage[faces]
    low, high = corners[:, :, 2].min(axis=1), corners[:, :, 2].max(axis=1)
    floor_band = ((measures["normal_z"] >= 0.9)
                  & (np.abs(measures["centroid_z"] - floor_z) <= float(surface["height_tolerance_stage_units"]) * mpu))
    touching = measures["valid"] & ~floor_band & (high >= floor_z + OBSTACLE_BAND_M[0]) \
        & (low <= floor_z + OBSTACLE_BAND_M[1])
    path = _stage(np.asarray(camera_path_levelled_m, dtype=np.float64).reshape(-1, 3)) if len(
        camera_path_levelled_m) else np.zeros((0, 3))
    # A flat floor standing in for an uneven scanned one (``atlas_world_levelling``) is a development-only
    # estimate; the spec carries it so the G1 lane can flag it. Absent, the facts are as before.
    # So is a flat floor filled under the filmed path, where the scan kept none (``walked_floor``).
    levelling = base_scene.get("levelling") or {}
    fallback, walked = levelling.get("fallback_floor"), levelling.get("walked_floor")
    estimate = ({key: fallback.get(key) for key in ("claim", "basis", "band_spread_m")}
                if isinstance(fallback, Mapping) else None)
    if isinstance(walked, Mapping):
        estimate = ({**estimate, "basis": f"{estimate.get('basis')}+{walked.get('basis')}"} if estimate
                    else {"claim": walked.get("claim"), "basis": walked.get("basis"), "band_spread_m": None})
    floor_estimate = {"levelling_floor_estimate": estimate} if estimate else {}
    facts = {
        "schema_version": FACTS_SCHEMA,
        "collision_mesh_digest": base_scene["collision_mesh_digest"],
        "floor_height_world_m": float(floor_z),
        "filmed_path_world_m": path.tolist(),
        "obstacle_triangle_count": int(touching.sum()),
        "metric_scale": {"status": "provider_declared_not_independently_validated",
                         "evidence": [{"source": str(base_scene.get("scale_authority")),
                                       "meters_per_unit": mpu,
                                       "levelling_tilt_degrees": (base_scene.get("levelling") or {}).get("tilt_degrees"),
                                       **floor_estimate}]},
    }
    facts["facts_digest"] = canonical_digest(facts, digest_field="facts_digest")
    return {**facts, "_obstacles_xy": corners[touching][:, :, :2],
            "_obstacles_z": np.column_stack([low[touching], high[touching]]) - float(floor_z)}


def clear_goal_along_path(path: Sequence[Sequence[float]], obstacles_xy: np.ndarray,
                          clearance_m: float = ROBOT_CLEARANCE_RADIUS_M) -> tuple[list[float] | None, int]:
    """The last filmed position a robot can stand at clear of obstacles, and how many it skipped."""
    for skipped, point in enumerate(reversed(list(path))):
        if obstacle_distance(point[:2], obstacles_xy) >= clearance_m:
            return list(point), skipped
    return None, len(path)


def compile_locomotion_spec(*, task_context: Mapping[str, Any], facts: Mapping[str, Any],
                            requirements: Mapping[str, Any]) -> dict[str, Any]:
    from .task_spec_compiler import compile_task_spec

    path = facts["filmed_path_world_m"]
    obstacles = facts.get("_obstacles_xy")
    anchors = route_anchors(path, obstacles) if obstacles is not None and len(path) >= 2 else {}
    scene = {"scene_revision_digest": facts["facts_digest"], "floor_height_world_m": facts["floor_height_world_m"],
             "metric_scale": facts["metric_scale"], "filmed_path_world_m": path, "anchors": anchors}
    brief = {"site_id": task_context["scene_id"], "task_id": "website-" + task_context["context_digest"][7:27],
             "sentence": task_context["description"]}
    return compile_task_spec(brief=brief, scene=scene, task_requirements=dict(requirements))


#: What each catalog policy can do, as data with its evidence (``policy_requirements_served``).
SERVED_PATH = Path(__file__).resolve().parents[2] / "config" / "policy_requirements_served.v1.json"


def policy_requirements_served(path: Path = SERVED_PATH) -> dict[str, dict[str, bool]]:
    value = json.loads(Path(path).read_text())
    if value.get("schema_version") != "blueprint.policy_requirements_served.v1":
        raise ValueError("policy_requirements_served_schema_invalid")
    return {row["candidate_id"]: {"locomote": row["locomote"] is True, "manipulate": row["manipulate"] is True}
            for row in value["policies"]}


def _policy_serves(row: Mapping[str, Any], requirements: Mapping[str, Any],
                   served: Mapping[str, Mapping[str, bool]]) -> bool:
    """Whether a catalog policy does the required work, from its declared row only.

    A policy with no declaration serves nothing: an undeclared ability is never assumed.
    """
    declared = served.get(row["candidate_id"])
    return declared is not None and all(declared.get(key) is True for key in ("locomote", "manipulate")
                                        if requirements.get(key))


def _catalog_task_kind_gap(row: Mapping[str, Any], requirements: Mapping[str, Any]) -> list[str]:
    """Whether the catalog row lists the task kind this job compiles to, as the screening plan reads it.

    Only a walk with nothing handled compiles to a kind here (``goal_region_navigation``); the codes are
    the screening plan's own (``beta_policy_screening_plan``), so a hold names what planning would refuse.
    """
    if not requirements.get("locomote") or requirements.get("manipulate"):
        return []
    from .native_g1_navigation_goal import NAVIGATION_TASK_KIND

    kinds = row.get("supported_task_kinds") or []
    if NAVIGATION_TASK_KIND in kinds:
        return []
    return ["catalog:task_kind_not_supported_by_candidate" if kinds else "catalog:task_kind_support_unverified"]


def runnable_options(requirements: Mapping[str, Any], *, catalog: Mapping[str, Any] | None = None,
                     served: Mapping[str, Mapping[str, bool]] | None = None,
                     built_pieces: frozenset[str] | None = None) -> dict[str, Any]:
    """Bodies and policies that could do this job, each with what it still lacks here.

    ``built_pieces`` is what exists for this job (``navigation_pieces_built`` from its handoff's
    outputs); it is recorded so the hold contract re-checks the same pieces. Without it no job output
    is known, so only the pieces that are code alone count as built.
    """
    built = navigation_pieces_built({}) if built_pieces is None else frozenset(built_pieces)
    from .task_evaluation_embodiment_registry import EMBODIMENT_REGISTRY

    served = policy_requirements_served() if served is None else served
    if catalog is None:
        from .beta_policy_candidate_catalog import load_catalog
        catalog = load_catalog(Path(__file__).resolve().parents[2] / "config" / "beta_policy_candidates.v1.json")
    bodies = [row for row in EMBODIMENT_REGISTRY if row.meets(requirements)]
    by_profile = {key: row for row in bodies for key in row.policy_robot_profile_ids}
    policies = []
    # A walk with no manipulation is the navigation profile: a body may declare a narrow development-only
    # allowance for it (``EmbodimentProfile.development_run_missing``), as the G1 does
    # (``G1_NAVIGATION_DEVELOPMENT_RUN``). The items it stands in for on this candidate leave ``missing`` and
    # are recorded STOOD_IN with their basis and owner decision, never satisfied; the option's result is
    # recorded development_only. Every other requirement reads the body's full gaps.
    navigation = bool(requirements.get("locomote")) and not requirements.get("manipulate")
    for row in catalog["candidates"]:
        body = by_profile.get((row.get("interface") or {}).get("robot_profile_id"))
        if body is None or not (row.get("eligibility") or {}).get("enabled") or not _policy_serves(row, requirements, served):
            continue
        body_missing, allowance = (body.development_run_missing("goal_region_navigation",
                                                                candidate_id=row["candidate_id"])
                                   if navigation else (body.missing, None))
        missing = [*(f"embodiment:{item}" for item in body_missing),
                   *(f"pipeline:{piece}" for piece in NAVIGATION_PIPELINE_PIECES if piece not in built)] \
            if requirements.get("locomote") else [*(f"embodiment:{item}" for item in body.missing)]
        missing += [f"catalog:{code}" for code in (row.get("eligibility") or {}).get("blockers") or []]
        missing += _catalog_task_kind_gap(row, requirements)
        option = {"candidate_id": row["candidate_id"], "embodiment_id": body.embodiment_id,
                  "runnable": not missing, "missing": missing}
        if allowance is not None:
            # Its result is recorded development_only, and each item it stands in for is named STOOD_IN.
            option.update(claim_ceiling=allowance.claim_ceiling,
                          development_run_allowance=allowance.record(row["candidate_id"]))
        policies.append(option)
    return {"requirements": dict(requirements),
            **({"navigation_pieces_built": sorted(built)} if requirements.get("locomote") else {}),
            "embodiments": [{"embodiment_id": row.embodiment_id, "robot": row.robot, "mobility": row.mobility,
                             "status": row.status, "missing": list(row.missing)} for row in bodies],
            "policies": policies}


def runnable_policy_gap(*, task_context: Mapping[str, Any], requirements: Mapping[str, Any],
                        spec_digest: str, task_semantics: Mapping[str, Any] | None = None,
                        source_video_digest: str | None = None,
                        options: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """The typed hold when no policy can run this compiled task here; ``None`` when one can."""
    options = dict(options) if options is not None else runnable_options(requirements)
    return runnable_policy_gap_from_options(task_context=task_context, requirements=requirements,
                                            spec_digest=spec_digest, options=options, task_semantics=task_semantics,
                                            source_video_digest=source_video_digest)


def retained_spec(root: Path) -> dict[str, Any] | None:
    path = Path(root) / SPEC_FILE
    if not path.is_file() or path.is_symlink():
        return None
    from .task_spec_compiler import validate_task_spec
    return validate_task_spec(json.loads(path.read_text()))


__all__ = [
    "BUILT_NAVIGATION_PIECES", "HOLD_CODE", "NAVIGATION_PIECE_OUTPUTS", "NAVIGATION_PIPELINE_PIECES",
    "NAVIGATION_SCENE_PACKET", "NAVIGATION_TASK_FILE", "SOURCE_IDENTITY_PIECE", "SPEC_FILE", "clear_goal_along_path", "compile_locomotion_spec", "route_anchors", "route_record", "ROUTE_FILE",
    "navigation_pieces_built", "obstacle_distance", "policy_requirements_served", "retained_spec",
    "runnable_options", "runnable_policy_gap", "scene_facts",
]
