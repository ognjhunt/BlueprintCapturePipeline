"""Generated views of weakly observed regions, checked against the capture, for the room's look.

Owner decision, 2026-11-18: AI-generated Atlas views may enter the refined splat's input (the room's
look) when the real capture and its depth are the base and the generated views agree with them.
A website video often sees the room from one direction: test job ``itest-c0cbbf20`` saw its table-top
items with a 2.9 to 3.2 degree spread of view angles across 85 views, and a shelf with 1.4, and its
splat is blurry and streaky there. Only new angles fix that.

The weak-region map (``pipeline/atlas/weak_regions.json``, schema ``atlas_weak_regions.v1``, written by
its own builder) names those regions in the posed cameras' world. For each (at most ``MAX_REGIONS``,
weakest angle spread first; at most ``MAX_VIEWS`` views a job) cameras orbit the region's centroid
about the vertical by the region's ``orbit_deg``, starting from the filmed camera in ``seen_by`` that
looks most directly at it: the same distance and intrinsics (on Atlas's 1280x720 target grid), aimed
at the centroid. ``atlasGenerate`` makes the views from the real photographs that saw the region and
their own depth, as completion's views are made (``atlas_scene_generate.prepare_rgbd_context``), one
request a region, each through the job's paid admission and reservation and inside its allowance.

Every returned view is rendered against those filmed frames and their depth (``atlas_gap_composite``);
it is dropped when it contradicts them where they observe the scene well, or overlaps them too little
to tell. The views that pass join the real frames in the refined-splat request with the region they
were made for as their generated part (``keep_mask``), so ``atlas_native_splat`` posts them only
through its whole-view review and falls back to the real frames alone when that form fails. What was
generated, checked, dropped and admitted is recorded once, before the splat is bought, and read back
on every later tick: a rerun buys nothing new.

No map, an empty map or a splat already bought: nothing is generated and the job runs as before.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .atlas_automation_plan import GENERATED_VIEWS_IN_LOOK
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file

WEAK_REGIONS_SCHEMA = "atlas_weak_regions.v1"
#: The weak-region map, relative to the capture root (the website job's Atlas root is ``pipeline/atlas``).
WEAK_REGIONS_RELATIVE = Path("pipeline/atlas/weak_regions.json")
SCHEMA = "atlas_look_views.v1"
ROOT, PLAN, RECORD = "look_views", "look_views_plan.json", "look_views.json"
#: Bounds per job (owner's brief, 2026-11-18): weakest regions first.
MAX_REGIONS, MAX_VIEWS = 6, 24
#: Views a region gets at most: two orbit magnitudes, each side of its filmed camera.
VIEWS_PER_REGION = 4
#: The real photographs a region's request carries and its views are checked against, nearest the
#: filmed camera first. One depth-context view accompanies each target (``target_batches``): 8 + 2 x 4 < 64.
CONTEXT_FRAMES_PER_REGION = 8
#: The consistency rule. A view passes when the filmed frames observe at least this share of it well
#: (``atlas_gap_composite.well_observed``) and it contradicts them on at most this share of that part.
#: Starting values chosen on 2026-11-18, not yet calibrated on a job's retained views: the tolerances
#: themselves are ``atlas_gap_composite``'s, calibrated on capture-447ae4f4.
MIN_CHECKED_FRACTION = 0.10
MAX_CONTRADICTED_FRACTION = 0.10
TARGET_GRID = (1280, 720)
PROMPT = ("The exact photographed room seen from a slightly different angle, with the same layout, materials, "
          "lighting, fixed fixtures and existing possessions in their photographed positions. No additional "
          "rooms, fixtures, furniture, possessions or decorations.")
NONE = {"views": [], "record": None}


# --- the weak-region map (the other builder's contract, read here without importing it) ------------------------

def _finite(value, size=None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, list, tuple)):
        return False
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    return ((size is None or len(values) == size)
            and all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) for v in values))


def _usable(row) -> bool:
    return (isinstance(row, dict) and isinstance(row.get("id"), str) and _finite(row.get("centroid"), 3)
            and _finite(row.get("extent"), 3) and all(v >= 0 for v in row["extent"])
            and _finite(row.get("orbit_deg"), 2) and isinstance(row.get("seen_by"), list)
            and all(isinstance(key, str) for key in row["seen_by"]))


def load_weak_regions(capture_root) -> list[dict] | None:
    """The map's usable regions; None when there is no map or it does not read as ``atlas_weak_regions.v1``."""
    path = Path(capture_root) / WEAK_REGIONS_RELATIVE
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if (not isinstance(value, dict) or WEAK_REGIONS_SCHEMA not in (value.get("schema_version"), value.get("schema"))
            or not isinstance(value.get("regions"), list)):
        return None
    return [row for row in value["regions"] if _usable(row)]


# --- cameras -----------------------------------------------------------------------------------------------------

def orbit_angles(orbit_deg) -> list[float]:
    """Signed orbit angles, nearest first: each side of the filmed camera at both of the map's magnitudes."""
    magnitudes = sorted({round(abs(float(v)), 6) for v in orbit_deg} - {0.0})
    return [sign * magnitude for magnitude in magnitudes for sign in (-1, 1)]


def landscape_camera(frame):
    """The filmed camera on Atlas's target grid: its rays kept, quarter-turned if portrait, principal point centred."""
    from .atlas_scene_chisel import oriented_camera

    camera, _ = oriented_camera(frame)
    k = camera["intrinsics"]
    scale = min(TARGET_GRID[0] / k["width"], TARGET_GRID[1] / k["height"])
    k.update(fx=k["fx"] * scale, fy=k["fy"] * scale, cx=TARGET_GRID[0] / 2, cy=TARGET_GRID[1] / 2,
             width=TARGET_GRID[0], height=TARGET_GRID[1])
    return camera


def _facing(frame, centroid) -> float:
    direction = np.asarray(centroid, float) - np.asarray(frame["position"], float)
    length = np.linalg.norm(direction)
    return float(-np.asarray(frame["rotation"])[:, 2] @ direction / length) if length > 1e-9 else -1.0


def orbit_cameras(regions, frames, *, max_regions=MAX_REGIONS, max_views=MAX_VIEWS):
    """Cameras orbiting each weak region's centroid, and the regions given none with the reason.

    The base is the filmed camera in ``seen_by`` that looks most directly at the centroid. Each new
    camera turns the base position about the vertical through the centroid (Atlas worlds are Y up),
    so it keeps the base's distance and height, and looks at the centroid with the base's intrinsics.
    """
    from scipy.spatial.transform import Rotation
    from .atlas_completion_cameras import aimed_rotation

    by_id = {frame["frame_id"]: frame for frame in frames}
    ordered = sorted(regions, key=lambda row: (row.get("angle_spread_deg")
                                               if _finite(row.get("angle_spread_deg")) else math.inf, row["id"]))
    # A region none of whose cameras this pose holds takes no slot.
    skipped = [{"region_id": row["id"], "reason": "seen_by_cameras_not_posed"} for row in ordered
               if not any(key in by_id for key in row["seen_by"])]
    ordered = [row for row in ordered if any(key in by_id for key in row["seen_by"])]
    chosen = ordered[:max_regions]
    skipped += [{"region_id": row["id"], "reason": "over_region_limit"} for row in ordered[max_regions:]]
    per_region = max(1, min(VIEWS_PER_REGION, max_views // max(1, len(chosen))))
    targets = []
    for region in chosen:
        seen = [by_id[key] for key in dict.fromkeys(region["seen_by"]) if key in by_id]
        centroid = np.asarray(region["centroid"], float)
        base = max(seen, key=lambda frame: _facing(frame, centroid))
        offset = np.asarray(base["position"], float) - centroid
        if np.linalg.norm(offset) < 1e-9:
            skipped.append({"region_id": region["id"], "reason": "filmed_camera_at_centroid"})
            continue
        context = sorted(seen, key=lambda frame: (float(np.linalg.norm(np.asarray(frame["position"])
                                                                       - base["position"])), frame["frame_id"]))
        context_ids = [frame["frame_id"] for frame in context[:CONTEXT_FRAMES_PER_REGION]]
        made = 0
        for angle in orbit_angles(region["orbit_deg"]):
            if made >= per_region or len(targets) >= max_views:
                break
            position = centroid + Rotation.from_euler("y", angle, degrees=True).apply(offset)
            rotation = aimed_rotation(position, centroid)
            if rotation is None:
                continue
            camera = landscape_camera(base)
            camera["extrinsics"].update(position=position.tolist(),
                                        quaternion=Rotation.from_matrix(rotation).as_quat().tolist())
            identity = canonical_digest({"region": region["id"], "centroid": centroid.tolist(),
                                         "base": base["frame_id"], "orbit_degrees": angle, "camera": camera})
            targets.append({"camera_id": "look-" + identity[7:23], "region_id": region["id"],
                            "base_frame_id": base["frame_id"], "orbit_degrees": angle,
                            "distance": float(np.linalg.norm(offset)), "target_world_estimate": centroid.tolist(),
                            "region_extent": [float(v) for v in region["extent"]],
                            "context_frame_ids": context_ids, "camera": camera,
                            "camera_location_observed_free": None, "collision_qualified": False})
            made += 1
        if not made:
            skipped.append({"region_id": region["id"], "reason": "no_orbit_camera"})
    return targets, skipped


def region_footprint(target, camera=None) -> np.ndarray:
    """The pixels of the region's box (centroid, extent) in the view: the part the view was generated for."""
    from .atlas_depth_splat import _rotation_matrix

    camera = camera or target["camera"]
    k, e = camera["intrinsics"], camera["extrinsics"]
    half = np.asarray(target["region_extent"], float) / 2
    corners = np.asarray(target["target_world_estimate"], float) + half * np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    local = (corners - np.asarray(e["position"], float)) @ _rotation_matrix(e["quaternion"])
    mask = np.zeros((int(k["height"]), int(k["width"])), bool)
    depth = -local[:, 2]
    if not np.any(depth > 1e-6):
        return mask
    depth = np.maximum(depth, 1e-6)
    u = local[:, 0] / depth * k["fx"] + k["cx"]
    v = -local[:, 1] / depth * k["fy"] + k["cy"]
    left, right = int(np.clip(np.floor(u.min()), 0, k["width"])), int(np.clip(np.ceil(u.max()), 0, k["width"]))
    top, bottom = int(np.clip(np.floor(v.min()), 0, k["height"])), int(np.clip(np.ceil(v.max()), 0, k["height"]))
    mask[top:bottom, left:right] = True
    return mask


# --- the consistency check ---------------------------------------------------------------------------------------

def check_view(view, references, footprint=None) -> dict:
    """Whether a generated view agrees with the filmed frames and their depth where they overlap.

    ``atlas_gap_composite.composite_view`` renders the references into the view and replaces, among the
    pixels they observe well, those whose depth (beyond 3.94%) or blurred colour contradicts the view.
    Every pixel is checked: nothing is admitted in advance. The view fails when the references observe
    too little of it well, or it contradicts them on too much of that, or on too much of the region it
    was made for (``footprint``): a small object it erased or moved is most of its region.
    """
    from .atlas_gap_composite import composite_view

    shape = np.shape(view["depth"])
    built, receipt = composite_view({**view, "keep_mask": np.zeros(shape, bool)}, references)
    total, well = receipt["total_pixel_count"], receipt["well_observed_pixel_count"]
    checked = well / total if total else 0.0
    contradicted = receipt["replaced_pixel_count"] / well if well else None
    in_region = None
    if footprint is not None and np.any(footprint):
        # Pixels the composite took from the capture (replaced or feathered), within the region.
        changed = (np.any(np.asarray(built["rgb"]) != np.asarray(view["rgb"]), axis=-1)
                   | (np.asarray(built["depth"]) != np.asarray(view["depth"])))
        in_region = float(changed[footprint].mean())
    reason = ("too_little_overlap_with_filmed_frames" if checked < MIN_CHECKED_FRACTION
              else "contradicts_filmed_frames" if contradicted > MAX_CONTRADICTED_FRACTION
              else "contradicts_filmed_frames_in_region" if (in_region or 0) > MAX_CONTRADICTED_FRACTION else None)
    return {"passed": reason is None, **({"reason": reason} if reason else {}),
            "checked_fraction": round(checked, 6),
            "contradicted_fraction": None if contradicted is None else round(contradicted, 6),
            "region_contradicted_fraction": None if in_region is None else round(in_region, 6),
            **{key: receipt[key] for key in ("observed_basis", "total_pixel_count", "well_observed_pixel_count",
                                             "depth_contradiction_pixel_count", "colour_contradiction_pixel_count",
                                             "replaced_pixel_count", "reference_frame_ids", "rgb_sha256",
                                             "depth_sha256")},
            "tolerances": "atlas_gap_composite.v2"}


# --- generated views as frames -----------------------------------------------------------------------------------

def _native_outputs(stage: Path) -> dict:
    result = json.loads((stage / "result.json").read_text())
    if result.get("status") != "completed" or result.get("digest") != canonical_digest(result, digest_field="digest"):
        raise ValueError("atlas_look_views_generated_result_invalid")
    return result


def _native_pixels(stage: Path, index: int, key: str) -> np.ndarray:
    """One returned view's pixels from its digest-bound output, read when used and never kept."""
    import cv2
    from .atlas_depth_splat import read_exr_depth

    result = _native_outputs(stage)
    rows = {row["role"]: row for row in result["outputs"]}
    role = {"rgb": "imageAsset", "depth": "depth-depthAsset", "confidence": "depth-confidenceAsset"}[key]
    row = rows.get(f"frames-{index}-{role}")
    if row is None and key == "confidence":
        depth = _native_pixels(stage, index, "depth")
        return (np.isfinite(depth) & (depth > 0)).astype(np.float32)
    path = stage / row["path"]
    if path.is_symlink() or not path.resolve().is_relative_to(stage.resolve()) or _sha256_file(path) != row["sha256"]:
        raise ValueError("atlas_look_views_generated_output_changed")
    if key == "depth":
        return read_exr_depth(path)
    pixels = cv2.imread(str(path), cv2.IMREAD_COLOR if key == "rgb" else cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise ValueError("atlas_look_views_generated_output_undecodable")
    return np.ascontiguousarray(pixels[..., ::-1]) if key == "rgb" else pixels


def splat_frame(view, stage: Path):
    """A passing view as a refined-splat input frame: its region is the generated part (``keep_mask``)."""
    from .atlas_depth_splat import LazyFrame, _rotation_matrix

    camera = view["camera"]
    fields = {"frame_id": view["frame_id"], "intrinsics": dict(camera["intrinsics"]),
              "extrinsics": dict(camera["extrinsics"]), "rotation": _rotation_matrix(camera["extrinsics"]["quaternion"]),
              "position": np.asarray(camera["extrinsics"]["position"], float),
              "keep_mask": region_footprint(view, camera)}
    index = view["native_frame_index"]
    pixels = {key: (lambda key=key: _native_pixels(stage, index, key)) for key in ("rgb", "depth", "confidence")}
    return LazyFrame(fields, pixels, ["frame_id", "rgb", "depth", "confidence", "intrinsics", "extrinsics",
                                      "rotation", "position", "keep_mask"])


# --- the stage ---------------------------------------------------------------------------------------------------

def _capture_root(output_root: Path) -> Path | None:
    output_root = Path(output_root)
    return output_root.parent.parent if output_root.name == "atlas" and output_root.parent.name == "pipeline" else None


def _seal(path: Path, value: dict) -> dict:
    value["digest"] = canonical_digest(value, digest_field="digest")
    write_json(path, value)
    return value


def _retained(root: Path, pose) -> dict:
    """The sealed record's passing views; nothing is bought or generated again."""
    record = json.loads((root / RECORD).read_text())
    if record.get("digest") != canonical_digest(record, digest_field="digest") or (root / RECORD).is_symlink():
        raise ValueError("atlas_look_views_record_changed")
    if record.get("pose_result_digest") != pose["digest"]:
        return {"views": [], "record": record}
    passed = set(record.get("passed_view_ids") or [])
    views = [splat_frame(row, root / row["stage_relative_path"]) for row in record.get("views") or []
             if row["frame_id"] in passed]
    return {"views": views, "record": record}


def _record(root: Path, plan, *, status, regions_digest, pose, skipped, batches=(), views=(), **extra) -> dict:
    passed = [row["frame_id"] for row in views if row["check"]["passed"]]
    return _seal(root / RECORD, {
        "schema_version": SCHEMA, "status": status, "weak_regions_digest": regions_digest,
        "pose_result_digest": pose["digest"], "plan_digest": (plan or {}).get("digest"),
        "limits": {"max_regions": MAX_REGIONS, "max_views": MAX_VIEWS, "views_per_region": VIEWS_PER_REGION,
                   "context_frames_per_region": CONTEXT_FRAMES_PER_REGION},
        "check_rule": {"min_checked_fraction": MIN_CHECKED_FRACTION,
                       "max_contradicted_fraction": MAX_CONTRADICTED_FRACTION,
                       "basis": "starting values 2026-11-18; atlas_gap_composite tolerances from capture-447ae4f4"},
        "skipped_regions": list(skipped), "batches": list(batches), "views": list(views),
        "passed_view_ids": passed, "dropped_view_count": len(views) - len(passed),
        "generated_content": True, "claim_ceiling": "development_only", **extra})


def look_views_for_splat(*, frames, pose, output_root, policy, task_context, source_digest, configuration,
                         task_runner, bought, check=None, max_polls=2) -> dict:
    """``{"views": frames to send beside the real ones, "record": the sealed record or None}``, or processing.

    ``configuration`` is a callable returning the job's current allowance, read only when a map exists.
    """
    from .atlas_pricing import headroom, render_cost, submitted
    from .atlas_scene_completion import load_generated_views
    from .atlas_scene_generate import prepare_rgbd_context
    from .hard_stops import is_wall
    from .website_atlas_tasks import validate_task_request

    output_root = Path(output_root)
    root = output_root / ROOT
    if (root / RECORD).is_file():
        return _retained(root, pose)
    if bought:
        return dict(NONE)
    sealed = json.loads((root / PLAN).read_text()) if (root / PLAN).is_file() else None
    if sealed is not None:
        # The cameras were decided once: a later map never re-derives them into another purchase.
        if sealed.get("digest") != canonical_digest(sealed, digest_field="digest") or (root / PLAN).is_symlink():
            raise ValueError("atlas_look_views_plan_changed")
        targets, skipped, regions_digest = sealed["targets"], sealed["skipped_regions"], sealed["weak_regions_digest"]
    else:
        capture_root = _capture_root(output_root)
        regions = load_weak_regions(capture_root) if capture_root is not None else None
        if not regions:
            return dict(NONE)
        root.mkdir(parents=True, exist_ok=True)
        regions_digest = _sha256_file(capture_root / WEAK_REGIONS_RELATIVE)
        targets, skipped = orbit_cameras(regions, frames)
    if sealed is not None and sealed.get("pose_result_digest") != pose["digest"]:
        record = _record(root, sealed, status="plan_changed", regions_digest=regions_digest, pose=pose,
                         skipped=skipped)
        return {"views": [], "record": record}
    if not targets or "rgbd_view_cost_usd" not in policy:
        record = _record(root, None, status="no_views" if not targets else "rgbd_view_price_unreviewed",
                         regions_digest=regions_digest, pose=pose, skipped=skipped)
        return {"views": [], "record": record}
    by_id = {frame["frame_id"]: frame for frame in frames}
    groups = {}
    for target in targets:
        groups.setdefault(target["region_id"], []).append(target)
    specs = []
    for index, group in enumerate(groups.values()):
        context = [by_id[key] for key in group[0]["context_frame_ids"]]
        contexts, uploads, _ = prepare_rgbd_context(frames=context, root=root / "context" / f"{index:02d}")
        request = {"contextFrames": contexts, "targetCameras": [row["camera"] for row in group], "prompt": PROMPT,
                   "enhancePrompt": False, "returnDepth": True}
        validate_task_request("atlasGenerate", request)
        quote = {"schema_version": "website_atlas_task_price_review.v1", "task": "atlasGenerate",
                 "request_digest": canonical_digest(request), "maximum_cost_usd": render_cost(policy, len(group)),
                 **{key: policy[key] for key in ("rate_card_reference", "reviewed_by", "reviewed_on")}}
        specs.append((group, request, uploads, root / "generated" / f"{index:02d}", quote))
    plan = {"schema_version": SCHEMA + ".plan", "weak_regions_digest": regions_digest,
            "pose_result_digest": pose["digest"], "targets": targets, "skipped_regions": skipped,
            "batches": [{"region_id": group[0]["region_id"], "frame_ids": [row["camera_id"] for row in group],
                         "stage_relative_path": str(stage.relative_to(root)), "request_digest": quote["request_digest"],
                         "maximum_cost_usd": quote["maximum_cost_usd"]} for group, _, _, stage, quote in specs]}
    plan["digest"] = canonical_digest(plan, digest_field="digest")
    if sealed is not None and sealed != plan:
        # The filmed frames' photographs or depth changed under a sealed plan: nothing more is bought for it.
        record = _record(root, plan, status="plan_changed", regions_digest=regions_digest, pose=pose,
                         skipped=skipped)
        return {"views": [], "record": record}
    write_json(root / PLAN, plan)
    from .atlas_native_splat import _provider_failed_and_exhausted

    pending = [quote["maximum_cost_usd"] for _, _, _, stage, quote in specs if not submitted(stage)]
    if pending:
        forecast = headroom(configuration(), stages={"look_view_batches": len(pending), "whole_view_review": 1},
                            cost_usd=sum(pending), final_output_root=output_root, context=task_context,
                            source_digest=source_digest)
        if forecast is not None:
            write_json(root / "request-budget-preflight.json", forecast)
        if forecast is not None and not forecast["sufficient_for_known_calls"]:
            record = _record(root, plan, status="allowance_insufficient", regions_digest=regions_digest, pose=pose,
                             skipped=skipped, forecast=forecast)
            return {"views": [], "record": record}
    batches, views = [], []
    for index, (group, request, uploads, stage, quote) in enumerate(specs):
        row = {"region_id": group[0]["region_id"], "stage_relative_path": str(stage.relative_to(root)),
               "request_digest": quote["request_digest"], "maximum_cost_usd": quote["maximum_cost_usd"]}
        if _provider_failed_and_exhausted(stage):
            batches.append({**row, "status": "failed", "reason": "atlas_task_operation_failed"})
            continue
        if check is not None and not submitted(stage):
            check("refined_splat_look_views")
        try:
            state = task_runner(task="atlasGenerate", request=request, uploads=uploads, output_root=stage,
                                source_video_digest=source_digest, task_context=task_context, price_review=quote,
                                max_polls=max_polls)
        except ValueError as exc:
            if is_wall(exc):
                raise
            # Flagged and the job goes on: the region keeps its filmed views only.
            batches.append({**row, "status": "failed", "reason": str(exc)[:200]})
            continue
        if state["status"] != "completed":
            return {"status": "processing", "stage": "refined_splat_look_views",
                    "operation_id": state.get("operation_id")}
        native, returned, _ = load_generated_views(stage, [target["camera_id"] for target in group])
        batches.append({**row, "status": "completed", "native_result_digest": native["digest"],
                        "billing": native.get("billing")})
        from .atlas_gap_composite import observation_references
        references = observation_references([by_id[key] for key in group[0]["context_frame_ids"]])
        for native_index, (view, target) in enumerate(zip(returned, group, strict=True)):
            footprint = region_footprint(target, view)
            entry = {"frame_id": target["camera_id"], "region_id": target["region_id"],
                     "base_frame_id": target["base_frame_id"], "orbit_degrees": target["orbit_degrees"],
                     "distance": target["distance"], "target_world_estimate": target["target_world_estimate"],
                     "region_extent": target["region_extent"], "camera": {"intrinsics": view["intrinsics"],
                                                                         "extrinsics": view["extrinsics"]},
                     "stage_relative_path": row["stage_relative_path"], "native_frame_index": native_index,
                     "native_result_digest": native["digest"], "region_pixel_count": int(footprint.sum())}
            if not footprint.any():
                entry["check"] = {"passed": False, "reason": "weak_region_not_in_view"}
            elif footprint.all():
                entry["check"] = {"passed": False, "reason": "weak_region_fills_view"}
            else:
                entry["check"] = check_view(view, references, footprint)
            views.append(entry)
        del returned, references
    record = _record(root, plan, status="checked", regions_digest=regions_digest, pose=pose, skipped=skipped,
                     batches=batches, views=views,
                     bought_cost_usd=round(sum(row["maximum_cost_usd"] for row in batches
                                               if row["status"] == "completed"), 6))
    return _retained(root, pose) if record["passed_view_ids"] else {"views": [], "record": record}


def disclosure(look, native, output_root: Path) -> dict:
    """What the result says about the views: how many were made and passed, and which the look holds."""
    record = look["record"]
    passed = set(record.get("passed_view_ids") or [])
    rows = (native.get("splat_manifest") or {}).get("frames") or []
    in_look = sorted({row["frame_id"] for row in rows
                      if row.get("frame_id") in passed and row.get("whole_view_review")})
    return {"status": record["status"], "record_digest": record["digest"],
            "record_relative_path": str((Path(output_root) / ROOT / RECORD).relative_to(output_root)),
            "generated_view_count": len(record.get("views") or []), "passed_view_count": len(passed),
            "request_form": (native.get("refined_splat") or {}).get("request_form"), "in_look_view_ids": in_look,
            **({"generated_views_in": GENERATED_VIEWS_IN_LOOK} if in_look else {})}


def bound_look_views(*, splat, source, appearance, job_dir: Path) -> dict | None:
    """The look's checked generated views, when a preserved scene's refined splat is bound to them; else None.

    The splat must be the reviewed-whole form on the source pose, and every view it was sent must be a
    filmed frame or a view the sealed record passed and the whole-view review admitted.
    """
    from .atlas_refined_splat_forms import WHOLE

    disclosed = (appearance or {}).get("look_views") or {}
    if disclosed.get("generated_views_in") != GENERATED_VIEWS_IN_LOOK:
        return None
    path = (Path(job_dir) / str(disclosed.get("record_relative_path") or "")).resolve()
    if not path.is_relative_to(Path(job_dir).resolve()) or not path.is_file():
        raise ValueError("atlas_look_views_not_bound")
    record = json.loads(path.read_text())
    if (record.get("digest") != canonical_digest(record, digest_field="digest")
            or record["digest"] != disclosed.get("record_digest") or record.get("pose_result_digest") != source["digest"]):
        raise ValueError("atlas_look_views_not_bound")
    captured = {row.get("frame_id") for row in
                ((source.get("operation_receipt") or {}).get("input_usage") or {}).get("frames") or []}
    passed, rows = set(record.get("passed_view_ids") or []), splat.get("frames") or []
    generated = [row for row in rows if isinstance(row, dict) and row.get("frame_id") in passed]
    if (splat.get("splat_kind") != "atlas_refined" or splat.get("task") != "posedImages2Splats"
            or splat.get("request_form") != WHOLE or splat.get("pose_result_digest") != source.get("digest")
            or not generated or any((row.get("whole_view_review") or {}).get("admit_whole") is not True
                                    for row in generated)
            or not all(isinstance(row, dict) and (row in generated or (
                row.get("frame_id") in captured and "whole_view_review" not in row and "gap_composite" not in row))
                for row in rows)):
        raise ValueError("atlas_look_views_not_bound")
    return {"generated_views_in": GENERATED_VIEWS_IN_LOOK, "look_views_digest": record["digest"],
            "view_count": len({row["frame_id"] for row in generated}),
            "basis": "filmed_frames_with_generated_views_checked_against_them"}
