"""Dynamic robot-placement pipeline (task + any scene -> where the robot stands).

Public surface for the whole package. The flow is: a spatial index enumerates scene
objects (USD walk or perception unprojection), a resolver picks the one object the task
acts on, and the placement solver finds the open side to stand on. Every step is
backend-swappable and dependency-light: importing this package pulls in NO isaacsim, NO
google-genai, NO torch, NO GPU — the heavy bits (USD ``pxr``, the Gemini call, the PhysX
probe) are all lazy and/or injected.

``build_scene_index`` is a small factory over the two index backends; ``place_robot_for_task``
is the end-to-end orchestrator (objects -> resolve_target -> compute_stand_pose) that most
callers want.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .perception_index import PerceptionSceneSpatialIndex
from .perception_fusion import (
    MultiViewPerceptionSceneSpatialIndex,
    aabb_iou,
    fuse_scene_objects,
)
from .perception_views import (
    assemble_views,
    generate_view_ring,
    view_ring_for_bounds,
)
from .perception_adapter import (
    build_perception_view,
    build_perception_views,
    build_perception_views_from_frames,
    depth_provider_from_map,
    detections_from_sam3,
)
from .placement import compute_stand_pose
from .validation import (
    DEFAULT_VALIDATION_FOOTPRINT_HALF_EXTENT,
    DEFAULT_VALIDATION_MAX_FACING_ERROR_DEG,
    DEFAULT_VALIDATION_PELVIS_HEIGHT_M,
    DEFAULT_VALIDATION_STANDOFF_RANGE,
    PLACEMENT_VALIDATION_SCHEMA_VERSION,
    PlacementVerdict,
    build_placement_validation_report,
    placement_verdict_to_dict,
    scene_object_to_dict,
    validate_placement,
    validate_stand_pose,
    write_placement_validation_report,
)
from .target_resolver import (
    GenerateFn,
    resolve_target,
    resolve_target_by_label,
)
from .types import Probe, SceneObject, SceneSpatialIndex, StandPose
from .usd_index import (
    UsdSceneSpatialIndex,
    _clean_label,
    _is_excluded,
    _objects_from_bounds,
)

__all__ = [
    "SceneObject",
    "StandPose",
    "SceneSpatialIndex",
    "Probe",
    "UsdSceneSpatialIndex",
    "PerceptionSceneSpatialIndex",
    "MultiViewPerceptionSceneSpatialIndex",
    "fuse_scene_objects",
    "aabb_iou",
    "generate_view_ring",
    "view_ring_for_bounds",
    "assemble_views",
    "detections_from_sam3",
    "depth_provider_from_map",
    "build_perception_view",
    "build_perception_views",
    "build_perception_views_from_frames",
    "resolve_target",
    "resolve_target_by_label",
    "compute_stand_pose",
    "validate_stand_pose",
    "validate_placement",
    "PlacementVerdict",
    "PLACEMENT_VALIDATION_SCHEMA_VERSION",
    "DEFAULT_VALIDATION_FOOTPRINT_HALF_EXTENT",
    "DEFAULT_VALIDATION_MAX_FACING_ERROR_DEG",
    "DEFAULT_VALIDATION_PELVIS_HEIGHT_M",
    "DEFAULT_VALIDATION_STANDOFF_RANGE",
    "build_placement_validation_report",
    "placement_verdict_to_dict",
    "scene_object_to_dict",
    "write_placement_validation_report",
    "build_scene_index",
    "place_robot_for_task",
    "place_and_validate_robot_for_task",
    # pure helpers re-exported for tests / advanced callers
    "_clean_label",
    "_is_excluded",
    "_objects_from_bounds",
]


def build_scene_index(backend: str, **kw) -> SceneSpatialIndex:
    """Construct a spatial index for the chosen ``backend``.

    ``backend`` is ``"usd"`` (walk a stage / .usd path) or ``"perception"`` (unproject
    2D detections + depth). Keyword args are forwarded verbatim to the backend's
    constructor, so this is just a thin, swap-friendly entry point — callers name the
    source of truth and get a :class:`SceneSpatialIndex` without importing the concrete
    class.
    """
    key = (backend or "").strip().lower()
    if key == "usd":
        return UsdSceneSpatialIndex(**kw)
    if key == "perception":
        return PerceptionSceneSpatialIndex(**kw)
    raise ValueError(f"unknown scene-index backend: {backend!r} (expected 'usd' or 'perception')")


def place_robot_for_task(
    index: SceneSpatialIndex,
    task: str,
    *,
    probe: Probe,
    generate: Optional[GenerateFn] = None,
    **place_kw,
) -> StandPose:
    """End-to-end: enumerate objects, resolve the task's target, solve the stand pose.

    Pipeline (each stage swappable / injectable):

    1. ``index.objects()`` enumerates the scene's actionable objects.
    2. :func:`resolve_target` picks the one the task acts on. ``generate`` is the
       injectable VLM text call; when it is ``None`` we resolve purely by label
       (:func:`resolve_target_by_label`) so the orchestrator runs with no model in the
       loop — the hermetic-test and offline path.
    3. :func:`compute_stand_pose` probes for the open side and returns the pelvis pose.

    ``place_kw`` is forwarded to :func:`compute_stand_pose` (pelvis_height, floor_z,
    standing_distance, ...). Raises ``LookupError`` if no object matches the task — the
    caller cannot place a robot for a target that isn't in the scene.
    """
    objects = list(index.objects())
    if generate is None:
        target = resolve_target_by_label(task, objects)
    else:
        target = resolve_target(task, objects, generate=generate)
    if target is None:
        raise LookupError(f"no scene object matched task: {task!r}")
    return compute_stand_pose(target, probe=probe, **place_kw)


def place_and_validate_robot_for_task(
    index: SceneSpatialIndex,
    task: str,
    *,
    probe: Probe,
    floor_z: float,
    generate: Optional[GenerateFn] = None,
    validation_kw: Optional[dict] = None,
    **place_kw,
) -> Tuple[StandPose, PlacementVerdict]:
    """:func:`place_robot_for_task` + geometric self-validation of the solved pose.

    Returns ``(pose, verdict)``. The verdict checks the SOLVED pose against the same scene
    object catalog used to place it (clip / on-floor / facing / standoff), so a caller knows
    immediately whether the pose is usable — instead of discovering a clip in a render. The
    object catalog is enumerated once and feeds both placement and validation.

    ``floor_z`` is required (both the solver and the validator work in the floor frame). Do NOT
    also pass ``floor_z`` in ``place_kw``. ``validation_kw`` forwards extra args to
    :func:`validate_stand_pose` (e.g. ``footprint_half_extent``, ``standoff_range``,
    ``standoff_obstacles``). Raises ``LookupError`` when no object matches the task.
    """
    objects = list(index.objects())
    if generate is None:
        target = resolve_target_by_label(task, objects)
    else:
        target = resolve_target(task, objects, generate=generate)
    if target is None:
        raise LookupError(f"no scene object matched task: {task!r}")
    pose = compute_stand_pose(target, probe=probe, floor_z=floor_z, **place_kw)
    verdict = validate_placement(pose, target, objects, floor_z=floor_z, **(validation_kw or {}))
    return pose, verdict
