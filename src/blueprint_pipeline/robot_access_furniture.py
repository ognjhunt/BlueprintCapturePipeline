"""Movable furniture moved out of a fixed arm's way in the simulation, only with the site owner's approval.

Beta journey step: L5 team policy evaluation. capture-447ae4f4's Franka could
not stand within reach of its dining table, most likely because of the chairs
around it (#3355). The station search already says which refused stances would
be clear were named obstacles moved (``team_embodiment_stance``
``movable_obstacle_counterfactual``), and never chooses one, since the scene
the robot runs in still holds the obstacle. This module is the option that
can: off by default, and only for the job's own site owner.

1. **Measure.** The scene preparation lists the items its own removal analysis
   labels movable furniture: a ``movable_object`` the analysis kept as an
   obstacle or background (never a task object, the support, the destination
   or a person) whose label names a chair or a stool (``FURNITURE_KINDS``).
   Their bounds are the visible masked surfaces lifted with the model's
   estimated depth and cameras (``website_task_masks.estimate_target_bounds``)
   in the simulator frame (``movable_furniture_record``). A wall, a fixed
   counter, fixed clutter or anything else is never listed. An item whose
   bounds meet the task's support in plan is refused (``eligible_obstacles``),
   so the table, its legs and the task object can never go with it.
2. **Propose.** When the team half's carry search finds no stance the Franka
   carries from, the first search names those items as movable obstacles. Of
   the stances it reports clear were some of them moved, the one needing the
   fewest items whose pick and place pass the arm's CPU reach bound
   (``robot_arm_path_clearance.rigid_reach_shortfall``) is the typed proposal
   (``PROPOSAL_SCHEMA``): which items, why, and the stance it enables. Nothing
   changes: the half compiles exactly as before, with the proposal recorded.
3. **Ask the owner.** The screening timer's pass sends each such proposal to
   the WebApp over the signed Pipeline sync transport, as the success
   thresholds go (``website_owner_success_confirmation``):
   ``POST /api/internal/pipeline/task-evaluation-scene-intents/<intent>/robot-access-proposal``.
   The WebApp shows it on the owner's job link only, in plain words, and answers
   with the owner's append-only decision bound to the proposal digest. No
   decision is no change. ``approved`` writes one sealed record under the
   request's own intent (``<intent>/robot-access-removal-confirmations/<digest>.json``),
   bound to the intent, the revision, the collider and the exact items;
   ``declined`` writes nothing and holds the job by name (``DECLINED_HOLD``).
4. **Rebuild on approval.** The request is compiled once more. The compile
   finds the record for exactly this binding, writes a derived copy of the
   collider without the faces lying wholly inside the approved items' bounds
   and rising above the floor band (``write_collider_without``; the sealed
   collider is never edited), reads it back, searches the stance again on it,
   tries the carry again on the CPU and checks the object still rests where the
   flat-support check found it, all before any GPU time. The half records the
   named model (``MODEL``) and the assumption. The visual scene is unchanged:
   the existing removal path erases objects from the frames before
   reconstruction (a new paid Atlas edit and scene build), so the items still
   show in the simulation's camera views (``APPEARANCE``).

Only a rigid pick and place is covered. The owner's approval is a modelling
confirmation, never a spend approval; nothing here starts compute, calls a
provider or moves money.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

MODEL = "named_movable_furniture_moved_for_robot_access"
FURNITURE_SCHEMA = "blueprint.movable_furniture.v1"
FURNITURE_RULE = "analysis_movable_object_kept_as_obstacle_named_chair_or_stool"
FURNITURE_FRAME = "simulator_world_z_up_estimated_m"
PROPOSAL_SCHEMA = "blueprint.robot_access_removal_proposal.v1"
PROPOSAL_RULE = "fewest_named_items_whose_moving_clears_a_stance_within_the_arms_reach_bound"
NOTICE_SCHEMA = "website_robot_access_proposal.v1"
RECEIPT_SCHEMA = "website_robot_access_proposal_receipt.v1"
DECISION_SCHEMA = "website_robot_access_decision.v1"
CONFIRMATION_SCHEMA = "task_evaluation_robot_access_removal_confirmation.v1"
CONFIRMATION_DIRECTORY = "robot-access-removal-confirmations"
CONFIRMATION_SCOPE = "simulation_collision_modelling_only"
WEBSITE_CHANNEL = "website"
#: The named hold a declined proposal leaves on the team packet request's status.
DECLINED_HOLD = "team_scene_robot_access_declined_by_owner"
#: The team packet status field that keeps the latest exchange for its request.
STATUS_FIELD = "website_robot_access"
#: The approvals a completed request was already compiled again for.
REQUEUE_FIELD = "robot_access_confirmation_requeued"
#: The derived collider without the approved items, under the team half.
DIRECTORY = "robot-access"
#: The words an analysis label uses for a piece of movable furniture, and the kind each names.
FURNITURE_KINDS = {"chair": "chair", "chairs": "chair", "armchair": "chair", "armchairs": "chair",
                   "stool": "stool", "stools": "stool", "barstool": "stool", "barstools": "stool"}
#: The analysis's own reading of an item that stays in the scene and is no part of the task.
ELIGIBLE_ROLES = frozenset({"obstacle", "background"})
ELIGIBLE_EFFECTS = frozenset({"static_obstacle", "unrelated"})
#: A face rises above the floor band the stance search measures obstacles from
#: (``team_embodiment_stance.ROOM_MESH_FLOOR_CLEARANCE_M``): the floor under an item always stays.
FLOOR_BAND_M = 0.04
#: At most this many items in one proposal.
MAXIMUM_ITEMS = 4
APPEARANCE = {"status": "unchanged", "items_still_visible_in_camera_views": True,
              "reason": "the_existing_removal_path_erases_objects_from_frames_before_reconstruction_only"}
ASSUMPTION = "assumes_the_named_movable_furniture_is_moved_out_of_the_robots_way"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,131}\Z")
_TARGET = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_INSTANT = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z\Z")
_CODE = re.compile(r"[a-z][a-z0-9_]{0,99}\Z")
DECISION_FIELDS = frozenset({"schema_version", "intent_id", "intent_digest", "request_id", "capture_id",
                             "proposal_digest", "configured_scene_revision_digest", "removal_binding_digest",
                             "decision", "decided_via", "decided_at", "decision_digest"})
RECORD_FIELDS = frozenset({"schema_version", "scope", "model", "intent_id", "intent_digest", "owner",
                           "authenticated_issuer", "configured_scene_revision_digest", "removal_binding",
                           "removal_binding_digest", "items", "proposal_digest", "decision_digest", "approved_by",
                           "approval_evidence", "issuer_channel", "operator_uid", "confirmed_at",
                           "spend_authorized", "provider_mutation_performed", "confirmation_digest"})
MAX_EXCHANGES_PER_PASS = 4
TIMEOUT_SECONDS = 5
#: (first wait, longest wait) in seconds before a request is asked again, by its last outcome; the wait
#: doubles while the same outcome repeats for the same proposal.
BACKOFF = {"pending": (120, 3600), "declined": (1800, 86400), "refused": (900, 86400), "failed": (120, 21600),
           "approved": (3600, 86400), "already_approved": (3600, 86400), "intent_unbound": (3600, 86400),
           "intent_revoked": (3600, 86400)}
Transport = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError("robot_access_" + code)


def _bounds(value: Any) -> tuple[list[float], list[float]] | None:
    try:
        low, high = [float(v) for v in value["minimum"]], [float(v) for v in value["maximum"]]
    except (KeyError, TypeError, ValueError):
        return None
    if len(low) != 3 or len(high) != 3 or not all(map(math.isfinite, low + high)) or any(
            a >= b for a, b in zip(low, high, strict=True)):
        return None
    return low, high


# ----------------------------------------------------------------------------- measure


def furniture_kind(row: Mapping[str, Any]) -> str | None:
    """The kind of movable furniture the analysis's own label names (``FURNITURE_KINDS``), else None."""

    words = re.findall(r"[a-z]+", " ".join(str(row.get(key) or "") for key in (
        "semantic_label", "segmentation_prompt")).lower())
    return next((FURNITURE_KINDS[word] for word in words if word in FURNITURE_KINDS), None)


def _refusal(row: Mapping[str, Any]) -> str | None:
    if row.get("target_class") != "movable_object":
        return "analysis_class_not_movable_object"
    if row.get("target_role") not in ELIGIBLE_ROLES:
        return "analysis_role_not_obstacle_or_background"
    if row.get("disposition") != "keep" or row.get("task_effect") not in ELIGIBLE_EFFECTS:
        return "analysis_effect_not_a_kept_obstacle"
    return None


def movable_furniture_record(targets: Sequence[Mapping[str, Any]],
                             to_world: Callable[[Mapping[str, Any]], tuple[Sequence[float], Sequence[float]]]
                             ) -> dict[str, Any] | None:
    """The items the scene's own analysis labels movable furniture, with their bounds in the simulator frame.

    ``targets`` are the preparation's tracked analysis targets (``website_task_masks``, bound to the
    geometry: ``estimated_visible_bounds``); ``to_world`` maps such bounds to the simulator frame. A
    target whose label names no chair or stool is not listed at all; one that does but that the
    analysis did not keep as a movable obstacle or background is listed under ``excluded`` with why.
    None when the analysis names no chair or stool. People are never listed.
    """

    items: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in targets:
        if row.get("target_class") == "person" or row.get("target_role") == "person":
            continue
        kind = furniture_kind(row)
        if kind is None:
            continue
        name = str(row.get("target_id") or "")
        reason = _refusal(row) or (None if _TARGET.fullmatch(name) else "analysis_target_id_invalid")
        bounds = None
        if reason is None:
            try:
                low, high = to_world(row["estimated_visible_bounds"])
                bounds = _bounds({"minimum": low, "maximum": high})
            except (KeyError, TypeError, ValueError, IndexError):
                bounds = None
            reason = None if bounds is not None else "bounds_unmeasured"
        if reason is not None:
            excluded.append({"target_id": name[:128], "kind": kind, "reason": reason})
            continue
        assert bounds is not None
        items.append({"obstacle_id": name, "label": str(row.get("semantic_label") or kind)[:160], "kind": kind,
                      "segmentation_prompt": str(row.get("segmentation_prompt") or "")[:160],
                      "bounds_world_m": {"minimum": bounds[0], "maximum": bounds[1]},
                      "basis": "visible_masked_surfaces_with_model_estimated_depth_and_cameras"})
    if not items and not excluded:
        return None
    return {"schema_version": FURNITURE_SCHEMA, "rule": FURNITURE_RULE, "frame": FURNITURE_FRAME, "items": items,
            "excluded": excluded, "complete_object_dimensions": False, "physical_measurement_proven": False}


def movable_furniture_for_capture(capture_root: str | Path, *, task_id: str) -> dict[str, Any]:
    """The capture's website preparation's movable furniture, when it is the preparation of this task.

    ``{"status": "measured", "record": ...}``, or ``{"status": <why not>}``: a preparation made before
    the measurement existed, or one that names no chair or stool, has none, and nothing is proposed.
    """

    root = Path(capture_root) / "pipeline"
    try:
        context = json.loads((root / "website_task_context.json").read_text(encoding="utf-8"))
        preparation = json.loads((root / "website_scene_preparation" / "preparation.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "preparation_unreadable"}
    digest = str(context.get("context_digest") or "")
    if (not isinstance(preparation, dict) or preparation.get("digest") != canonical_digest(preparation, digest_field="digest")
            or (preparation.get("binding") or {}).get("task_context_digest") != digest
            or task_id != "website-" + digest[7:27]):
        return {"status": "preparation_not_this_task"}
    record = preparation.get("movable_furniture")
    if not isinstance(record, Mapping) or record.get("schema_version") != FURNITURE_SCHEMA:
        return {"status": "not_recorded_by_preparation"}
    return {"status": "measured", "record": json.loads(json.dumps(record)),
            "preparation_digest": preparation["digest"]}


def eligible_obstacles(record: Mapping[str, Any] | None, *, supports: Sequence[Mapping[str, Any]]
                       ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The measured items a station search may name as movable, and those it may not, with why.

    ``supports`` are the task's support planes (``bounds_min_xyz_m``/``bounds_max_xyz_m``): an item
    whose bounds meet one in plan is refused, so the support, its legs and the object on it can never
    be inside the bounds of anything moved.
    """

    named: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    boxes = []
    for support in supports:
        box = _bounds({"minimum": support.get("bounds_min_xyz_m"), "maximum": support.get("bounds_max_xyz_m")})
        if box is not None:
            boxes.append(box)
    for row in (record or {}).get("items") or []:
        bounds = _bounds(row.get("bounds_world_m")) if isinstance(row, Mapping) else None
        name = str(row.get("obstacle_id") or "") if isinstance(row, Mapping) else ""
        if bounds is None or not _TARGET.fullmatch(name) or row.get("kind") not in set(FURNITURE_KINDS.values()):
            refused.append({"obstacle_id": name[:128], "reason": "item_invalid"})
            continue
        low, high = bounds
        if not boxes or any(low[0] <= b[1][0] and high[0] >= b[0][0] and low[1] <= b[1][1] and high[1] >= b[0][1]
                            for b in boxes):
            refused.append({"obstacle_id": name, "reason": "overlaps_task_support" if boxes else "support_unknown"})
            continue
        named.append({"obstacle_id": name, "label": str(row.get("label") or row["kind"])[:160], "kind": row["kind"],
                      "bounds_world_m": {"minimum": low, "maximum": high}, "basis": str(row.get("basis") or "")})
    return named, refused


# ----------------------------------------------------------------------------- propose


def _base_pose(x: float, y: float, z: float, centre: Sequence[float]) -> dict[str, list[float]]:
    yaw = math.atan2(float(centre[1]) - y, float(centre[0]) - x)
    return {"position_world_m": [x, y, z], "orientation_xyzw": [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]}


def removal_binding(*, revision_digest: str, collision_sha256: str, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What an approval binds in the scene: the revision, the collider the items are taken from, and the items."""

    rows = [{key: json.loads(json.dumps(row[key])) for key in ("obstacle_id", "label", "kind", "bounds_world_m")}
            for row in items]
    return {"configured_scene_revision_digest": revision_digest, "scene_collision_sha256": collision_sha256,
            "items": rows, "model": MODEL}


def station_proposal(*, stances: Sequence[Mapping[str, Any]], carry: Mapping[str, Any],
                     named: Sequence[Mapping[str, Any]], refused: Sequence[Mapping[str, Any]],
                     subject_center: Sequence[float], task: Mapping[str, Any], root_z: float,
                     revision_digest: str, collision_sha256: str, support_label: str) -> dict[str, Any]:
    """The typed proposal for a carry no ranked stance clears, or why there is none (CPU only).

    ``stances`` are the first search's ranked stances, made with ``named`` as movable obstacles;
    ``carry`` is the carry search's record (``team_scene_packet._carrying_station``); ``task`` holds the
    rigid task's ``start_pose_world``, ``target_position_world_m``, ``contact_point_m`` and
    ``subject_bounds_m``. Of the stances clear were named items moved (best first, as the search ranks
    them), the one whose pick and place both lie within the Franka's reach bound and that needs the
    fewest items is proposed.
    """

    from .robot_arm_path_clearance import RobotArmPathClearanceError, rigid_reach_shortfall

    why = {"reason": "no_ranked_stance_carries", "carry_cleared": bool(carry.get("cleared")),
           "stances_tried": len(carry.get("results") or []),
           "reach_shortfall": (carry.get("reach_shortfall") or {}).get("status")}
    base = {"schema_version": PROPOSAL_SCHEMA, "model": MODEL, "rule": PROPOSAL_RULE, "why": why,
            "refused_items": [dict(row) for row in refused], "chosen": False}
    if not named:
        return {**base, "status": "not_proposed", "reason": "no_named_movable_furniture"}
    counterfactual = ((stances[0].get("search") or {}).get("movable_obstacle_counterfactual") or {}) if stances else {}
    by_id = {row["obstacle_id"]: row for row in named}
    tried: list[dict[str, Any]] = []
    for rank, row in enumerate(counterfactual.get("stances") or []):
        moved = [name for name in row.get("moved") or [] if name in by_id]
        if not moved or len(moved) != len(row.get("moved") or []) or len(moved) > MAXIMUM_ITEMS:
            continue
        x, y = (float(v) for v in row["position_world_m"])
        pose = _base_pose(x, y, root_z, subject_center)
        try:
            reach = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=pose,
                                          start_pose_world=task["start_pose_world"],
                                          target_position_world=task["target_position_world_m"],
                                          contact_point_m=task["contact_point_m"],
                                          subject_bounds_m=task.get("subject_bounds_m"))
        except (KeyError, TypeError, ValueError, RobotArmPathClearanceError):
            continue
        tried.append({"rank": rank, "position_world_m": pose["position_world_m"], "moved": moved,
                      "farthest_task_point_m": row.get("farthest_task_point_m"),
                      "reach": {key: reach[key] for key in ("status", "reach_bound_m", "beyond")}
                      | {"phases": [{key: phase[key] for key in ("phase", "distance_m")} for phase in reach["phases"]]}})
    within = [row for row in tried if row["reach"]["status"] == "within_bound"]
    if not within:
        return {**base, "status": "not_proposed", "tried": tried[:8],
                "reason": ("no_stance_clear_if_named_items_moved" if not tried
                           else "no_stance_clear_if_named_items_moved_within_reach")}
    chosen = min(within, key=lambda row: (len(row["moved"]), row["rank"]))
    items = [by_id[name] for name in sorted(chosen["moved"])]
    binding = removal_binding(revision_digest=revision_digest, collision_sha256=collision_sha256, items=items)
    return {**base, "status": "proposed", "items": [dict(row) for row in items],
            "support_label": str(support_label or "the support")[:160],
            "enabled_stance": {key: chosen[key] for key in ("position_world_m", "farthest_task_point_m", "reach")},
            "removal_binding": binding, "removal_binding_digest": canonical_digest(binding),
            "appearance": dict(APPEARANCE), "claim_ceiling": "cpu_bounded_estimate"}


# ----------------------------------------------------------------------------- rebuild


def write_collider_without(source: str | Path, destination: str | Path, *, items: Sequence[Mapping[str, Any]],
                           floor_z: float) -> dict[str, Any]:
    """A derived copy of the scene collider without the faces of ``items`` (the sealed file is never edited).

    A collision mesh face goes when every one of its points lies inside one item's bounds (the
    station search's own membership, ``triangle_or_box_bounds_within_named_bounds``) and it rises
    ``FLOOR_BAND_M`` above ``floor_z`` (the floor under an item always stays). Points are kept; face
    counts and indices, and any uniform or face-varying primvar, normals, hole index or face subset,
    are cut or renumbered with them. Read from the compile's binary copy, edited in memory and
    exported to ``destination`` (``.usdc``).
    """

    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Vt

    from .scene_physics_preflight import _usd_matrix
    from .usd_binary_read_cache import editable

    opened = editable(source)
    layer = Sdf.Layer.OpenAsAnonymous(opened) if isinstance(opened, str) else opened
    stage = Usd.Stage.Open(layer)
    if (not stage or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0)
            or UsdGeom.GetStageUpAxis(stage) != "Z"):
        raise ValueError("robot_access_collider_frame_invalid")
    boxes = [(np.asarray(row["bounds_world_m"]["minimum"], dtype=float),
              np.asarray(row["bounds_world_m"]["maximum"], dtype=float)) for row in items]
    removed_by_item = [0] * len(items)
    removed_low = [np.full(3, np.inf) for _ in items]
    removed_high = [np.full(3, -np.inf) for _ in items]
    meshes: list[dict[str, Any]] = []
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        mesh = UsdGeom.Mesh(prim)
        for attribute in (mesh.GetPointsAttr(), mesh.GetFaceVertexCountsAttr(), mesh.GetFaceVertexIndicesAttr()):
            if attribute.GetNumTimeSamples():
                raise ValueError("robot_access_collider_time_sampled")
        stored = mesh.GetPointsAttr().Get()
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
        if stored is None or not len(counts):
            continue
        if (counts < 3).any() or int(counts.sum()) != len(indices):
            raise ValueError("robot_access_collider_faces_invalid")
        matrix = _usd_matrix(cache, prim)
        world = np.asarray(stored, dtype=np.float32).reshape(-1, 3).astype(float) @ matrix[:3, :3].T + matrix[:3, 3]
        del stored
        if len(indices) and (indices.min() < 0 or indices.max() >= len(world)):
            raise ValueError("robot_access_collider_faces_invalid")
        starts = np.cumsum(counts) - counts
        rises = np.maximum.reduceat(world[indices, 2], starts) >= float(floor_z) + FLOOR_BAND_M
        owner = np.full(len(counts), -1, dtype=np.int64)
        for index, (low, high) in reversed(list(enumerate(boxes))):
            inside = np.all((world >= low) & (world <= high), axis=1)
            held = np.logical_and.reduceat(inside[indices], starts) & rises
            owner[held] = index
        remove = owner >= 0
        if not remove.any():
            continue
        for index in range(len(items)):
            faces = np.flatnonzero(owner == index)
            if not len(faces):
                continue
            removed_by_item[index] += int(len(faces))
            corners = world[indices[np.concatenate([np.arange(starts[f], starts[f] + counts[f]) for f in faces])]]
            removed_low[index] = np.minimum(removed_low[index], corners.min(axis=0))
            removed_high[index] = np.maximum(removed_high[index], corners.max(axis=0))
        keep = ~remove
        keep_corner = np.repeat(keep, counts)
        renumber = np.cumsum(keep) - 1
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(counts[keep].astype(np.int32)))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(indices[keep_corner].astype(np.int32)))
        holes = mesh.GetHoleIndicesAttr()
        if holes.HasAuthoredValue() and len(holes.Get() or []):
            kept_holes = [int(renumber[h]) for h in holes.Get() if 0 <= int(h) < len(keep) and keep[int(h)]]
            holes.Set(Vt.IntArray(kept_holes))

        def cut(values: Any, interpolation: str, element_size: int = 1) -> Any:
            # A per-face or per-corner value goes with its face; anything else is unchanged.
            if interpolation == UsdGeom.Tokens.uniform:
                mask = keep
            elif interpolation == UsdGeom.Tokens.faceVarying:
                mask = keep_corner
            else:
                return None
            mask = np.repeat(mask, max(1, int(element_size)))
            if len(values) != len(mask):
                raise ValueError("robot_access_collider_primvar_size_invalid")
            return [value for value, kept in zip(values, mask, strict=True) if kept]

        normals = mesh.GetNormalsAttr()
        if normals.HasAuthoredValue() and normals.Get() is not None:
            kept_normals = cut(list(normals.Get()), mesh.GetNormalsInterpolation())
            if kept_normals is not None:
                normals.Set(type(normals.Get())(kept_normals))
        for primvar in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
            values = primvar.Get()
            if values is None:
                continue
            interpolation = primvar.GetInterpolation()
            if primvar.IsIndexed():
                kept_indices = cut(list(primvar.GetIndices()), interpolation)
                if kept_indices is not None:
                    primvar.SetIndices(Vt.IntArray(kept_indices))
                continue
            kept_values = cut(list(values), interpolation, primvar.GetElementSize())
            if kept_values is not None:
                primvar.Set(type(values)(kept_values))
        for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh):
            if subset.GetElementTypeAttr().Get() != UsdGeom.Tokens.face:
                continue
            faces = subset.GetIndicesAttr().Get() or []
            subset.GetIndicesAttr().Set(Vt.IntArray([int(renumber[f]) for f in faces
                                                     if 0 <= int(f) < len(keep) and keep[int(f)]]))
        meshes.append({"prim_path": prim.GetPath().pathString, "faces": int(len(counts)),
                       "faces_removed": int(remove.sum())})
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not layer.Export(str(output)):
        raise ValueError("robot_access_collider_export_failed")
    del stage
    return {"edit": "faces_wholly_inside_named_item_bounds_above_floor_band_removed", "floor_z_m": float(floor_z),
            "floor_band_m": FLOOR_BAND_M, "meshes": meshes, "faces_removed": int(sum(removed_by_item)),
            "items": [{"obstacle_id": row["obstacle_id"], "faces_removed": removed_by_item[index],
                       **({"removed_bounds_world_m": {"minimum": removed_low[index].tolist(),
                                                      "maximum": removed_high[index].tolist()}}
                          if removed_by_item[index] else {})}
                      for index, row in enumerate(items)]}


def faces_left_inside(collider: str | Path, *, items: Sequence[Mapping[str, Any]], floor_z: float) -> int:
    """How many of the derived collider's own triangles still lie inside an item's bounds above the floor band."""

    import numpy as np

    from .team_embodiment_stance import _world_triangles

    vertices, faces = _world_triangles(Path(collider), [0.0, 0.0, 0.0])
    corners = vertices[faces]
    low, high = corners.min(axis=1), corners.max(axis=1)
    rises = high[:, 2] >= float(floor_z) + FLOOR_BAND_M
    left = np.zeros(len(faces), dtype=bool)
    for row in items:
        minimum = np.asarray(row["bounds_world_m"]["minimum"], dtype=float)
        maximum = np.asarray(row["bounds_world_m"]["maximum"], dtype=float)
        left |= (low >= minimum).all(axis=1) & (high <= maximum).all(axis=1)
    return int((left & rises).sum())


# ----------------------------------------------------------------------------- the owner's record


def _safe(path: Path) -> None:
    _require(path.is_absolute() and ".." not in path.parts
             and not any(p.is_symlink() for p in (path, *path.parents)), "store_unsafe")


def _valid_record(path: Path, intent: Mapping[str, Any]) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file() or path.suffix != ".json":
        return None
    try:
        row = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    if not (isinstance(row, dict) and set(row) == RECORD_FIELDS and row.get("schema_version") == CONFIRMATION_SCHEMA
            and row.get("scope") == CONFIRMATION_SCOPE and row.get("model") == MODEL
            and row.get("confirmation_digest") == canonical_digest(row, digest_field="confirmation_digest")
            and path.name == str(row["confirmation_digest"]).removeprefix("sha256:") + ".json"
            and row["intent_id"] == intent["intent_id"] and row["intent_digest"] == intent["intent_digest"]
            and row["owner"] == intent["request"]["owner"] and row["approved_by"] == row["owner"]["user_id"]
            and row["authenticated_issuer"] == intent["authenticated_issuer"]
            and row["issuer_channel"] == WEBSITE_CHANNEL and type(row["operator_uid"]) is int
            and row["spend_authorized"] is False and row["provider_mutation_performed"] is False
            and isinstance(row["removal_binding"], dict)
            and row["removal_binding_digest"] == canonical_digest(row["removal_binding"])
            and row["items"] == row["removal_binding"].get("items")
            and row["configured_scene_revision_digest"] == row["removal_binding"].get("configured_scene_revision_digest")
            and isinstance(row["approval_evidence"], str) and bool(row["approval_evidence"].strip())):
        return None
    return row


def confirmations(directory: str | Path, intent: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every valid sealed approval of this intent's owner, earliest first; none once revoked."""

    directory = Path(directory)
    _safe(directory)
    store = directory / CONFIRMATION_DIRECTORY
    if (directory / "revoked.json").exists() or store.is_symlink() or not store.is_dir():
        return []
    rows = [row for row in (_valid_record(path, intent) for path in sorted(store.glob("*.json"))) if row]
    return sorted(rows, key=lambda row: (row["confirmed_at"], row["confirmation_digest"]))


def find_confirmation(directory: str | Path, intent: Mapping[str, Any], *,
                      removal_binding_digest: str) -> dict[str, Any] | None:
    """The earliest valid approval of exactly this removal binding for this intent, else None."""

    return next((row for row in confirmations(directory, intent)
                 if row["removal_binding_digest"] == removal_binding_digest), None)


def record_website_approval(*, intent_root: str | Path, intent: Mapping[str, Any], proposal: Mapping[str, Any],
                            notice: Mapping[str, Any], decision: Mapping[str, Any],
                            now: datetime | None = None) -> dict[str, Any]:
    """Write the owner's approval under the intent, exclusively and under the intake lock.

    ``proposal`` is the team half's recorded proposal and ``notice`` what the owner saw; the decision
    must approve exactly that notice. An existing approval of the same binding is returned as is.
    """

    from . import task_evaluation_scene_intake as intake

    _require(decision.get("decision") == "approved" and decision.get("proposal_digest") == notice["proposal_digest"]
             and notice["removal_binding_digest"] == proposal["removal_binding_digest"]
             and canonical_digest(proposal["removal_binding"]) == proposal["removal_binding_digest"],
             "decision_not_this_proposal")
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    root = Path(intent_root)
    _safe(root)
    root = intake._root(root)
    with intake._lock(root):
        directory = root / str(intent["intent_id"])
        _safe(directory)
        current = intake._read(directory / "intent.json", "intent_digest")
        _require(current == dict(intent) and not (directory / "revoked.json").exists(), "intent_changed_or_revoked")
        _require(os.geteuid() == directory.stat().st_uid, "issuer_user_mismatch")
        existing = find_confirmation(directory, current, removal_binding_digest=proposal["removal_binding_digest"])
        if existing is not None:
            return {"status": "already_approved", **existing}
        owner = dict(current["request"]["owner"])
        binding = json.loads(json.dumps(proposal["removal_binding"]))
        value = intake._seal({
            "schema_version": CONFIRMATION_SCHEMA, "scope": CONFIRMATION_SCOPE, "model": MODEL,
            "intent_id": current["intent_id"], "intent_digest": current["intent_digest"], "owner": owner,
            "authenticated_issuer": current["authenticated_issuer"],
            "configured_scene_revision_digest": binding["configured_scene_revision_digest"],
            "removal_binding": binding, "removal_binding_digest": proposal["removal_binding_digest"],
            "items": binding["items"], "proposal_digest": notice["proposal_digest"],
            "decision_digest": decision["decision_digest"], "approved_by": owner["user_id"],
            "approval_evidence": (f"site owner approved moving {len(binding['items'])} item(s) out of the robot's way "
                                  f"in the simulation on the website job page (owner link): webapp decision "
                                  f"{decision['decision_digest']} at {decision['decided_at']}"),
            "issuer_channel": WEBSITE_CHANNEL, "operator_uid": os.geteuid(),
            "confirmed_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"), "spend_authorized": False,
            "provider_mutation_performed": False}, "confirmation_digest")
        path = directory / CONFIRMATION_DIRECTORY / (value["confirmation_digest"].removeprefix("sha256:") + ".json")
        path.parent.mkdir(mode=0o750, exist_ok=True)
        _safe(path.parent)
        intake.write_exclusive(path, value)
        _require(find_confirmation(directory, current, removal_binding_digest=proposal["removal_binding_digest"])
                 is not None, "record_invalid")
    return {"status": "approved", "record_path": str(path), **value}


# ----------------------------------------------------------------------------- the website exchange


def proposal_notice(*, intent: Mapping[str, Any], task_id: str, proposal: Mapping[str, Any]) -> dict[str, Any]:
    """What the WebApp shows the owner: the items, by the analysis's own words, sealed with the binding."""

    notice = {"schema_version": NOTICE_SCHEMA, "intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"],
              "task_id": str(task_id), "configured_scene_revision_digest":
                  proposal["removal_binding"]["configured_scene_revision_digest"],
              "removal_binding_digest": proposal["removal_binding_digest"],
              "items": [{"target_id": row["obstacle_id"], "label": row["label"], "kind": row["kind"]}
                        for row in proposal["items"]],
              "support_label": proposal["support_label"], "reason": proposal["why"]["reason"],
              "appearance": proposal["appearance"]["status"]}
    notice["proposal_digest"] = cross_runtime_canonical_digest(notice)
    return notice


def send_proposal(notice: Mapping[str, Any]) -> dict[str, Any]:
    """POST the proposal with the existing signed Pipeline -> WebApp transport; return the WebApp's answer."""

    from .safe_outbound_http import pinned_api_policy, request
    from .task_evaluation_launch_webapp_sync import load_pipeline_sync_token
    from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url

    configured = os.getenv("PIPELINE_SYNC_WEBAPP_URL", "").strip()
    _require(bool(configured), "webapp_url_missing")
    parsed = urlsplit(validated_https_sync_url(configured))
    origin = f"{parsed.scheme}://{parsed.netloc}"
    body = json.dumps(dict(notice), separators=(",", ":"), allow_nan=False).encode()
    url = (f"{origin}/api/internal/pipeline/task-evaluation-scene-intents/"
           f"{quote(str(notice['intent_id']), safe='')}/robot-access-proposal")
    try:
        response = request(url, method="POST", data=body,
                           headers=_pipeline_sync_headers(load_pipeline_sync_token(), body),
                           timeout_seconds=TIMEOUT_SECONDS,
                           policy=pinned_api_policy(origin, max_response_bytes=20_000), max_response_bytes=20_000)
    except HTTPError as exc:
        code = None
        try:
            failure = json.loads(exc.read(4096))
            code = failure.get("code") if isinstance(failure, dict) else None
        except (ValueError, OSError):
            pass
        finally:
            exc.close()
        suffix = f":{code}" if isinstance(code, str) and _CODE.fullmatch(code) else ""
        raise ValueError(f"robot_access_http_{exc.code}{suffix}") from None
    value = json.loads(response.body)
    _require(isinstance(value, dict), "receipt_invalid")
    return value


def validated_receipt(value: Any, notice: Mapping[str, Any]) -> dict[str, Any]:
    """The WebApp's answer for exactly this proposal; anything else refuses (and so applies nothing)."""

    _require(isinstance(value, Mapping) and set(value) == {"schema_version", "status", "intent_id", "proposal_digest",
                                                             "decision"}
             and value["schema_version"] == RECEIPT_SCHEMA and value["intent_id"] == notice["intent_id"]
             and value["proposal_digest"] == notice["proposal_digest"]
             and value["status"] in {"pending", "approved", "declined"}, "receipt_invalid")
    decision = value["decision"]
    if value["status"] == "pending":
        _require(decision is None, "receipt_invalid")
        return dict(value)
    _require(isinstance(decision, Mapping) and set(decision) == DECISION_FIELDS
             and decision["schema_version"] == DECISION_SCHEMA and decision["decision"] == value["status"]
             and all(decision[field] == notice[field] for field in (
                 "intent_id", "intent_digest", "proposal_digest", "configured_scene_revision_digest",
                 "removal_binding_digest"))
             and decision["decided_via"] == "owner_job_link"
             and all(isinstance(decision[field], str) and _ID.fullmatch(decision[field])
                     for field in ("request_id", "capture_id"))
             and isinstance(decision["decided_at"], str) and _INSTANT.fullmatch(decision["decided_at"])
             and isinstance(decision["decision_digest"], str) and _DIGEST.fullmatch(decision["decision_digest"])
             and decision["decision_digest"] == cross_runtime_canonical_digest(
                 dict(decision), digest_field="decision_digest"), "decision_invalid")
    return {**value, "decision": dict(decision)}


def half_proposal(path: Any) -> dict[str, Any] | None:
    """The robot-access proposal a compiled team half recorded, when it is still only proposed; else None."""

    from .team_scene_packet import read_team_scene_inputs

    try:
        if not path:
            return None
        read_team_scene_inputs(str(path))
        value = json.loads(Path(str(path)).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    access = value.get("robot_access_removal") if isinstance(value, Mapping) else None
    if not isinstance(access, Mapping) or access.get("status") != "proposed":
        return None
    try:
        if canonical_digest(access["removal_binding"]) != access["removal_binding_digest"]:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return json.loads(json.dumps(dict(access)))


def _completed_with_proposal(status: Mapping[str, Any]) -> bool:
    return status.get("status") == "completed" and bool(status.get("team_scene_inputs_path"))


def _approval_of(path: Any, request: Mapping[str, Any]) -> dict[str, Any] | None:
    from .team_scene_packet_worker import owner_success_confirmation_scope

    proposal = half_proposal(path)
    if proposal is None:
        return None
    try:
        scope, _unbound = owner_success_confirmation_scope(request)
        return None if scope is None else find_confirmation(
            scope["directory"], scope["intent"], removal_binding_digest=proposal["removal_binding_digest"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def approved_proposal(path: Any, request: Mapping[str, Any]) -> bool:
    """Whether the site owner approved the proposal the half at ``path`` records (it is then compiled again)."""

    return _approval_of(path, request) is not None


def untried_confirmations(status: Mapping[str, Any], request: Mapping[str, Any]) -> list[str]:
    """The owner's approvals of a completed half's own proposal that no compile has tried yet."""

    if not _completed_with_proposal(status):
        return []
    row = _approval_of(status.get("team_scene_inputs_path"), request)
    tried = status.get(REQUEUE_FIELD)
    if row is None or row["confirmation_digest"] in (tried if isinstance(tried, list) else []):
        return []
    return [row["confirmation_digest"]]


def _now() -> float:
    return time.time()


def _instant(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _next_check(previous: Mapping[str, Any] | None) -> float | None:
    try:
        return datetime.strptime(str((previous or {}).get("next_check_at")), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _kind(outcome: Mapping[str, Any]) -> str:
    if outcome.get("status") == "failed" and str(outcome.get("error", "")).startswith("robot_access_http_4"):
        return "refused"
    return str(outcome.get("status"))


def _scheduled(outcome: dict[str, Any], previous: Mapping[str, Any] | None, now: float) -> dict[str, Any]:
    previous = previous or {}
    attempt = previous.get("attempt")
    repeated = (_kind(previous) == _kind(outcome) and previous.get("proposal_digest") == outcome.get("proposal_digest")
                and type(attempt) is int and attempt > 0)
    attempt = attempt + 1 if repeated else 1
    first, longest = BACKOFF.get(_kind(outcome), BACKOFF["failed"])
    wait = min(longest, first * 2 ** min(attempt - 1, 20))
    return {**outcome, "attempt": attempt, "checked_at": _instant(now), "next_check_at": _instant(now + wait)}


def _error_code(exc: BaseException) -> str:
    text = str(exc).split(":", 2)
    code = ":".join(text[:2]) if text[0].startswith("robot_access_http_") else text[0]
    return code[:160] if re.fullmatch(r"[a-z][a-z0-9_.:-]{0,159}", code) else type(exc).__name__


def exchange_robot_access_decision(packet_root: Path, request_key: str, *, transport: Transport | None = None,
                                   now: float | None = None) -> dict[str, Any] | None:
    """Ask the WebApp about one completed half's robot-access proposal, and record an approval.

    None when the request's half holds no open proposal, or the request is not due. The outcome is
    what the request's status keeps under ``STATUS_FIELD``; ``declined`` carries ``DECLINED_HOLD``.
    Any error is this request's ``failed`` outcome; it never leaves this function.
    """

    from .team_scene_packet_worker import _read_json, owner_success_confirmation_scope, validated_request

    now = _now() if now is None else now
    try:
        status = _read_json(packet_root / "status" / f"{request_key}.json") or {}
    except Exception:  # noqa: BLE001 - an unreadable status is not this exchange's to decide
        return None
    if not _completed_with_proposal(status) or (packet_root / "pending" / f"{request_key}.json").exists():
        return None
    proposal = half_proposal(status.get("team_scene_inputs_path"))
    if proposal is None:
        return None
    previous = status.get(STATUS_FIELD) if isinstance(status.get(STATUS_FIELD), dict) else None
    next_check = _next_check(previous)
    if (next_check is not None and now < next_check
            and (previous or {}).get("removal_binding_digest") == proposal["removal_binding_digest"]):
        return None
    base: dict[str, Any] = {"removal_binding_digest": proposal["removal_binding_digest"]}
    try:
        request = validated_request(_read_json(packet_root / "processed" / f"{request_key}.json"))
        scope, unbound = owner_success_confirmation_scope(request)
        if scope is None:
            return _scheduled({**base, "status": "intent_unbound", "reason": unbound}, previous, now)
        intent, directory = scope["intent"], Path(scope["directory"])
        if (directory / "revoked.json").exists():
            return _scheduled({**base, "status": "intent_revoked"}, previous, now)
        notice = proposal_notice(intent=intent, task_id=proposal["task_id"], proposal=proposal)
        base["proposal_digest"] = notice["proposal_digest"]
        existing = find_confirmation(directory, intent, removal_binding_digest=proposal["removal_binding_digest"])
        if existing is not None:
            return _scheduled({**base, "status": "already_approved",
                               "confirmation_digest": existing["confirmation_digest"]}, previous, now)
        receipt = validated_receipt((transport or send_proposal)(notice), notice)
        outcome: dict[str, Any] = {**base, "status": receipt["status"]}
        decision = receipt["decision"]
        if decision is not None:
            outcome["decision_digest"] = decision["decision_digest"]
        if receipt["status"] == "declined":
            outcome["hold"] = DECLINED_HOLD
        elif receipt["status"] == "approved":
            written = record_website_approval(intent_root=directory.parent, intent=intent, proposal=proposal,
                                              notice=notice, decision=decision)
            outcome.update(record=written["status"], confirmation_digest=written["confirmation_digest"])
        return _scheduled(outcome, previous, now)
    except Exception as exc:  # noqa: BLE001 - one job's exchange never stops the pass
        return _scheduled({**base, "status": "failed", "error": _error_code(exc)}, previous, now)


def exchange_robot_access_decisions(packet_root: str | Path, *, transport: Transport | None = None,
                                    limit: int = MAX_EXCHANGES_PER_PASS, now: float | None = None
                                    ) -> dict[str, dict[str, Any]]:
    """This pass's exchange for up to ``limit`` due completed halves with an open proposal, by request key."""

    from .team_scene_packet_worker import _read_json

    now = _now() if now is None else now
    root = Path(packet_root)
    open_keys = []
    for path in sorted((root / "processed").glob("*.json")):
        try:
            status = _read_json(root / "status" / path.name) or {}
        except Exception:  # noqa: BLE001 - one unreadable status never stops the pass
            continue
        if _completed_with_proposal(status) and not (root / "pending" / path.name).exists():
            open_keys.append(path.stem)
    if not open_keys:
        return {}
    start = int(now // 60) % len(open_keys)
    outcomes: dict[str, dict[str, Any]] = {}
    for key in open_keys[start:] + open_keys[:start]:
        if len(outcomes) >= max(0, limit):
            break
        outcome = exchange_robot_access_decision(root, key, transport=transport, now=now)
        if outcome is not None:
            outcomes[key] = outcome
    return outcomes


__all__ = [
    "APPEARANCE",
    "ASSUMPTION",
    "CONFIRMATION_DIRECTORY",
    "CONFIRMATION_SCHEMA",
    "DECISION_SCHEMA",
    "DECLINED_HOLD",
    "FURNITURE_KINDS",
    "MODEL",
    "NOTICE_SCHEMA",
    "PROPOSAL_SCHEMA",
    "RECEIPT_SCHEMA",
    "REQUEUE_FIELD",
    "STATUS_FIELD",
    "approved_proposal",
    "confirmations",
    "eligible_obstacles",
    "exchange_robot_access_decision",
    "exchange_robot_access_decisions",
    "faces_left_inside",
    "find_confirmation",
    "furniture_kind",
    "half_proposal",
    "movable_furniture_for_capture",
    "movable_furniture_record",
    "proposal_notice",
    "record_website_approval",
    "removal_binding",
    "send_proposal",
    "station_proposal",
    "untried_confirmations",
    "validated_receipt",
    "write_collider_without",
]
