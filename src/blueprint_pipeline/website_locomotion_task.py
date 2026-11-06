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
#: The Pipeline pieces a locomotion policy still needs beyond a registered body and policy.
NAVIGATION_SCENE_PACKET = "navigation_scene_packet_from_compiled_spec"
#: The pieces a G1 navigation run needs beyond the compiled spec, in build order
#: (docs/G1_NAVIGATION_FROM_COMPILED_SPEC.md). A hold names each one not yet built, so it
#: shrinks as each lands and never claims a route that does not exist.
NAVIGATION_PIPELINE_PIECES = (
    "navigation_task_spec_from_compiled_spec",   # website_navigation_packet (built)
    "navigation_scene_assets",                    # Isaac appearance/collision for a scene kept as filmed
    "arena_navigation_scene_runtime",             # scene plan, runtime and readback without a task object
    "g1_navigation_preparation",                  # G1 stance at the compiled start, cells, registration
)
BUILT_NAVIGATION_PIECES = frozenset({"navigation_task_spec_from_compiled_spec"})
NAVIGATION_TASK_FILE = "navigation_task_spec.json"


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


def obstacle_distance(point_xy: Sequence[float], triangles_xy: np.ndarray) -> float:
    """Horizontal distance from a point to the nearest obstacle triangle (0 inside one)."""
    if len(triangles_xy) == 0:
        return math.inf
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
    return float(np.min(np.where(inside, 0.0, edges)))


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
    facts = {
        "schema_version": FACTS_SCHEMA,
        "collision_mesh_digest": base_scene["collision_mesh_digest"],
        "floor_height_world_m": float(floor_z),
        "filmed_path_world_m": path.tolist(),
        "obstacle_triangle_count": int(touching.sum()),
        "metric_scale": {"status": "provider_declared_not_independently_validated",
                         "evidence": [{"source": str(base_scene.get("scale_authority")),
                                       "meters_per_unit": mpu,
                                       "levelling_tilt_degrees": (base_scene.get("levelling") or {}).get("tilt_degrees")}]},
    }
    facts["facts_digest"] = canonical_digest(facts, digest_field="facts_digest")
    return {**facts, "_obstacles_xy": corners[touching][:, :, :2]}


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
    goal, skipped = (clear_goal_along_path(path, obstacles) if obstacles is not None and len(path) >= 2
                     else (None, 0))
    start = None
    if obstacles is not None and len(path) >= 2:
        start = next((list(point) for point in path if obstacle_distance(point[:2], obstacles)
                      >= ROBOT_CLEARANCE_RADIUS_M), None)
    cleared = "filmed_path_cleared_of_obstacles"
    anchors = {**({"goal_world_m": goal, "goal_source": cleared} if goal is not None and skipped else {}),
               **({"start_world_m": start, "start_source": cleared}
                  if start is not None and start != list(path[0]) else {})}
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


def runnable_options(requirements: Mapping[str, Any], *, catalog: Mapping[str, Any] | None = None,
                     served: Mapping[str, Mapping[str, bool]] | None = None) -> dict[str, Any]:
    """Bodies and policies that could do this job, each with what it still lacks here."""
    from .task_evaluation_embodiment_registry import EMBODIMENT_REGISTRY

    served = policy_requirements_served() if served is None else served
    if catalog is None:
        from .beta_policy_candidate_catalog import load_catalog
        catalog = load_catalog(Path(__file__).resolve().parents[2] / "config" / "beta_policy_candidates.v1.json")
    bodies = [row for row in EMBODIMENT_REGISTRY if row.meets(requirements)]
    by_profile = {key: row for row in bodies for key in row.policy_robot_profile_ids}
    policies = []
    for row in catalog["candidates"]:
        body = by_profile.get((row.get("interface") or {}).get("robot_profile_id"))
        if body is None or not (row.get("eligibility") or {}).get("enabled") or not _policy_serves(row, requirements, served):
            continue
        missing = [*(f"embodiment:{item}" for item in body.missing),
                   *(f"pipeline:{piece}" for piece in NAVIGATION_PIPELINE_PIECES if piece not in BUILT_NAVIGATION_PIECES)] \
            if requirements.get("locomote") else [*(f"embodiment:{item}" for item in body.missing)]
        missing += [f"catalog:{code}" for code in (row.get("eligibility") or {}).get("blockers") or []]
        policies.append({"candidate_id": row["candidate_id"], "embodiment_id": body.embodiment_id,
                         "runnable": not missing, "missing": missing})
    return {"requirements": dict(requirements),
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
    "BUILT_NAVIGATION_PIECES", "HOLD_CODE", "NAVIGATION_PIPELINE_PIECES", "NAVIGATION_SCENE_PACKET",
    "NAVIGATION_TASK_FILE", "SPEC_FILE", "clear_goal_along_path", "compile_locomotion_spec",
    "obstacle_distance", "policy_requirements_served", "retained_spec", "runnable_options", "runnable_policy_gap",
    "scene_facts",
]
