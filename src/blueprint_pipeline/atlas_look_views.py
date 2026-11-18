"""Generated views of weakly observed regions, checked against the capture, for the room's look.

Owner decision, 2026-11-18: AI-generated Atlas views may enter the refined splat's input (the room's
look) when the real capture and its depth are the base and the generated views agree with them.
A website video often sees the room from one direction: test job ``itest-c0cbbf20`` saw its table-top
items with a 2.9 to 3.2 degree spread of view angles across 85 views, and a shelf with 1.4, and its
splat is blurry and streaky there. Only new angles fix that.

The weak-region map (``pipeline/atlas/weak_regions.json``, schema ``atlas_weak_regions.v1``, written by
its own builder) names those regions in the posed cameras' world. For each (at most ``MAX_REGIONS``,
in the map's order; at most ``MAX_VIEWS`` views a job) cameras orbit the region's centroid
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
#: Bounds per job (owner's brief, 2026-11-18), regions in the map's order.
MAX_REGIONS, MAX_VIEWS = 6, 24
#: Views a region gets at most: two orbit magnitudes, each side of its filmed camera.
VIEWS_PER_REGION = 4
#: A region's generation context is not bounded here: it fills the served model's rollout
#: (``atlas_scene_chisel.ROLLOUT_VIEWS`` less two views per target) with the region's own views, then
#: the job's task views, then the wider room (``generation_context``). This bound is the consistency
#: check's only: each view is composited against at most this many of the real frames that saw its
#: region, nearest its filmed camera first. That is a review, not reconstruction context, kept small
#: for the room step's runtime (``_judged``).
CHECK_FRAMES_PER_VIEW = 8
#: The consistency rule. A view passes when the filmed frames observe at least this share of it well
#: (``atlas_gap_composite.well_observed``) and it contradicts them on at most this share of that part.
#: Starting values chosen on 2026-11-18, not yet calibrated on a job's retained views: the tolerances
#: themselves are ``atlas_gap_composite``'s, calibrated on capture-447ae4f4.
#: Views of one region checked at once; the room step is one serial process on a 4-CPU worker.
CHECK_WORKERS = 4
MIN_CHECKED_FRACTION = 0.10
MAX_CONTRADICTED_FRACTION = 0.10
TARGET_GRID = (1280, 720)
PROMPT = ("The exact photographed room seen from a slightly different angle, with the same layout, materials, "
          "lighting, fixed fixtures and existing possessions in their photographed positions. No additional "
          "rooms, fixtures, furniture, possessions or decorations.")
NONE = {"views": [], "record": None}


# --- the weak-region map (``atlas_coverage_map``, written before the splat) -------------------------------------

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


def _mapped_regions(capture_root: Path):
    """``atlas_coverage_map``'s regions and the pose it mapped, read by its own sealed reader; (None, None) if none.

    The reader checks the seal, the arrays' digest and that the map is ``mapped``; a map it refuses
    gives no views, and the look keeps its filmed frames.
    """
    from .atlas_coverage_map import load_weak_regions, read_weak_regions

    try:
        regions = load_weak_regions(capture_root)
        record = read_weak_regions(Path(capture_root) / WEAK_REGIONS_RELATIVE.parent) if regions else None
    except (OSError, ValueError, KeyError, TypeError):
        return None, None
    if not regions or record is None:
        return None, None
    return [row for row in regions if _usable(row)], record.get("pose_result_digest")


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


def generation_context(seen, pool, *, base, task_frame_ids=(), targets: int) -> list[str]:
    """The real frames a region's ``atlasGenerate`` request carries, filling the provider's rollout.

    The served model takes the context photographs plus two views per target within
    ``atlas_scene_chisel.ROLLOUT_VIEWS`` (``target_batches``). The region's own ``seen_by`` views come
    first, nearest its filmed camera; then the job's task views; then the wider room, each next the
    pool's view farthest from those already chosen. All are distinct posed views of ``pool``; a frame
    that would not fit whole is passed over for a smaller one.
    """
    from .atlas_scene_chisel import ROLLOUT_VIEWS, photograph_count

    capacity = ROLLOUT_VIEWS - 2 * targets
    by_id = {frame["frame_id"]: frame for frame in pool}
    origin = np.asarray(base["position"], float)
    own = sorted(seen, key=lambda frame: (float(np.linalg.norm(np.asarray(frame["position"]) - origin)),
                                          frame["frame_id"]))
    ordered = [*own, *(by_id[key] for key in task_frame_ids if key in by_id)]
    chosen, used = [], 0

    def take(frame):
        nonlocal used
        if frame["frame_id"] in {row["frame_id"] for row in chosen}:
            return
        count = photograph_count(frame)
        if used + count <= capacity:
            chosen.append(frame)
            used += count
    for frame in ordered:
        take(frame)
    rest = [frame for frame in pool if frame["frame_id"] not in {row["frame_id"] for row in chosen}]
    while rest and used < capacity:
        placed = np.array([np.asarray(row["position"], float) for row in chosen] or [origin])
        far = max(rest, key=lambda frame: (float(np.min(np.linalg.norm(
            placed - np.asarray(frame["position"], float), axis=1))), frame["frame_id"]))
        rest.remove(far)
        take(far)
    return [frame["frame_id"] for frame in chosen]


def orbit_cameras(regions, frames, *, max_regions=MAX_REGIONS, max_views=MAX_VIEWS, task_frame_ids=()):
    """Cameras orbiting each weak region's centroid, and the regions given none with the reason.

    ``frames`` are the real views available to generation and the check: posed, sent to the splat and
    never held out from it. Regions are taken in the map's order (``atlas_coverage_map``: those the
    job's route views saw first, then the nearest), up to ``max_regions`` with a view among them. The
    base is the filmed camera in ``seen_by`` that looks most directly at the centroid. Each new
    camera turns the base position about the vertical through the centroid (Atlas worlds are Y up),
    so it keeps the base's distance and height, and looks at the centroid with the base's intrinsics.
    """
    from scipy.spatial.transform import Rotation
    from .atlas_completion_cameras import aimed_rotation

    by_id = {frame["frame_id"]: frame for frame in frames}
    ordered = list(regions)
    # A region none of whose cameras is available takes no slot.
    skipped = [{"region_id": row["id"], "reason": "seen_by_cameras_not_available"} for row in ordered
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
        nearest = sorted(seen, key=lambda frame: (float(np.linalg.norm(np.asarray(frame["position"])
                                                                       - base["position"])), frame["frame_id"]))
        check_ids = [frame["frame_id"] for frame in nearest[:CHECK_FRAMES_PER_VIEW]]
        start, made = len(targets), 0
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
                            "check_frame_ids": check_ids,
                            "seen_by_frame_ids": [frame["frame_id"] for frame in seen], "camera": camera,
                            "camera_location_observed_free": None, "collision_qualified": False})
            made += 1
        if not made:
            skipped.append({"region_id": region["id"], "reason": "no_orbit_camera"})
            continue
        context_ids = generation_context(seen, frames, base=base, task_frame_ids=task_frame_ids, targets=made)
        for target in targets[start:]:
            target["context_frame_ids"] = context_ids
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

def check_view(view, references, footprint=None, rendered=None) -> dict:
    """Whether a generated view agrees with the filmed frames and their depth where they overlap.

    ``atlas_gap_composite.composite_view`` renders the references into the view and replaces, among the
    pixels they observe well, those whose depth (beyond 3.94%) or blurred colour contradicts the view.
    Every pixel is checked: nothing is admitted in advance. The view fails when the references observe
    too little of it well, or it contradicts them on too much of that, or on too much of the region it
    was made for (``footprint``): a small object it erased or moved is most of its region.
    """
    from .atlas_gap_composite import composite_view

    shape = np.shape(view["depth"])
    built, receipt = composite_view({**view, "keep_mask": np.zeros(shape, bool)}, references, rendered=rendered)
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


def _check_and_composite(view, references, footprint):
    """``check_view`` and, for a passing view, its composite around the region: one render of the references."""
    from .atlas_gap_composite import composite_view, render_observed

    rendered = render_observed(references, {"intrinsics": view["intrinsics"], "extrinsics": view["extrinsics"]})
    result = check_view(view, references, footprint, rendered=rendered)
    if not result["passed"]:
        return result, None
    return result, composite_view({**view, "keep_mask": footprint}, references, rendered=rendered)


def _judged(group, returned, native, stage: Path, root: Path, batch, by_id):
    """Each returned view checked against its region's real frames, in ``CHECK_WORKERS`` threads; entries."""
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from .atlas_depth_splat import frame_arrays, write_exr_depth
    from .atlas_gap_composite import observation_references

    # A review, not reconstruction context: at most CHECK_FRAMES_PER_VIEW of the real frames that saw the
    # region, smoothed once before the threads share them.
    references = observation_references([by_id[key] for key in group[0]["check_frame_ids"]])
    for reference in references:
        frame_arrays(reference, "depth")

    def judge(item):
        native_index, (view, target) = item
        footprint = region_footprint(target, view)
        entry = {"frame_id": target["camera_id"], "region_id": target["region_id"],
                 "base_frame_id": target["base_frame_id"], "orbit_degrees": target["orbit_degrees"],
                 "distance": target["distance"], "target_world_estimate": target["target_world_estimate"],
                 "region_extent": target["region_extent"], "context_frame_ids": target["context_frame_ids"],
                 "check_frame_ids": target["check_frame_ids"],
                 "seen_by_frame_ids": target["seen_by_frame_ids"],
                 "camera": {"intrinsics": view["intrinsics"], "extrinsics": view["extrinsics"]},
                 "stage_relative_path": batch["stage_relative_path"], "native_frame_index": native_index,
                 "native_result_digest": native["digest"], "region_pixel_count": int(footprint.sum())}
        if not footprint.any():
            entry["check"] = {"passed": False, "reason": "weak_region_not_in_view"}
            return entry
        if footprint.all():
            entry["check"] = {"passed": False, "reason": "weak_region_fills_view"}
            return entry
        entry["check"], composite = _check_and_composite(view, references, footprint)
        if composite is not None:
            built, receipt = composite
            out = root / "composite" / target["camera_id"]
            out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(built["rgb"], np.uint8)).save(out / "rgb.png")
            write_exr_depth(out / "depth.exr", np.asarray(built["depth"], np.float32))
            entry["composite"] = {"rgb_relative_path": str((out / "rgb.png").relative_to(root)),
                                  "rgb_sha256": _sha256_file(out / "rgb.png"),
                                  "depth_relative_path": str((out / "depth.exr").relative_to(root)),
                                  "depth_sha256": _sha256_file(out / "depth.exr")}
            entry["gap_composite"] = receipt
        return entry
    with ThreadPoolExecutor(max_workers=CHECK_WORKERS) as pool:
        return list(pool.map(judge, enumerate(zip(returned, group, strict=True))))


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


def reference_frame_ids(view, sent=None) -> list[str]:
    """The real frames sent beside a view that overlap it, at most ``CHECK_FRAMES_PER_VIEW``.

    The frames its region's check used (the nearest that saw the region), where they are sent too; else, with fewer than
    ``atlas_gap_composite.MIN_VIEWS`` of them, the sent ``seen_by`` frames nearest the view's camera.
    The whole-view review pairs the view among these alone, not every real frame.
    """
    from .atlas_gap_composite import MIN_VIEWS

    if sent is None:
        return list(view["check_frame_ids"])
    real = {frame["frame_id"]: frame for frame in sent if frame.get("keep_mask") is None}
    chosen = [key for key in view["check_frame_ids"] if key in real]
    if len(chosen) < MIN_VIEWS:
        position = np.asarray(view["camera"]["extrinsics"]["position"], float)
        nearest = sorted((key for key in view["seen_by_frame_ids"] if key in real), key=lambda key: (
            float(np.linalg.norm(np.asarray(real[key]["position"], float) - position)), key))
        chosen = list(dict.fromkeys([*chosen, *nearest]))
    return chosen[:CHECK_FRAMES_PER_VIEW]


def _composite_pixels(root: Path, view, key: str) -> np.ndarray:
    from .atlas_depth_splat import read_exr_depth

    composite = view["composite"]
    path = root / composite[key + "_relative_path"]
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or _sha256_file(path) != composite[
            key + "_sha256"]:
        raise ValueError("atlas_look_views_composite_changed")
    if key == "depth":
        return read_exr_depth(path)
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def splat_frame(view, root: Path, sent=None):
    """A passing view as a refined-splat input frame: its region is the generated part (``keep_mask``).

    Its pixels are its composite, already corrected where it contradicts the real frames overlapping
    it (``gap_composite``), so the splat path posts it as it is and renders nothing again.
    """
    from .atlas_depth_splat import LazyFrame, _rotation_matrix

    camera, stage = view["camera"], root / view["stage_relative_path"]
    fields = {"frame_id": view["frame_id"], "intrinsics": dict(camera["intrinsics"]),
              "extrinsics": dict(camera["extrinsics"]), "rotation": _rotation_matrix(camera["extrinsics"]["quaternion"]),
              "position": np.asarray(camera["extrinsics"]["position"], float),
              "keep_mask": region_footprint(view, camera), "gap_composite": view["gap_composite"],
              "overlapping_frame_ids": reference_frame_ids(view, sent)}
    index = view["native_frame_index"]
    pixels = {"rgb": lambda: _composite_pixels(root, view, "rgb"), "depth": lambda: _composite_pixels(root, view, "depth"),
              "confidence": lambda: _native_pixels(stage, index, "confidence")}
    return LazyFrame(fields, pixels, ["frame_id", "rgb", "depth", "confidence", "intrinsics", "extrinsics",
                                      "rotation", "position", "keep_mask", "gap_composite", "overlapping_frame_ids"])


# --- the stage ---------------------------------------------------------------------------------------------------

def _capture_root(output_root: Path) -> Path | None:
    output_root = Path(output_root)
    return output_root.parent.parent if output_root.name == "atlas" and output_root.parent.name == "pipeline" else None


def _seal(path: Path, value: dict) -> dict:
    value["digest"] = canonical_digest(value, digest_field="digest")
    write_json(path, value)
    return value


def _retained(root: Path, pose, sent=None) -> dict:
    """The sealed record's passing views; nothing is bought or generated again."""
    record = json.loads((root / RECORD).read_text())
    if record.get("digest") != canonical_digest(record, digest_field="digest") or (root / RECORD).is_symlink():
        raise ValueError("atlas_look_views_record_changed")
    if record.get("pose_result_digest") != pose["digest"]:
        return {"views": [], "record": record}
    passed = set(record.get("passed_view_ids") or [])
    views = [splat_frame(row, root, sent) for row in record.get("views") or [] if row["frame_id"] in passed]
    return {"views": views, "record": record}


def _record(root: Path, plan, *, status, regions_digest, pose, skipped, batches=(), views=(), **extra) -> dict:
    passed = [row["frame_id"] for row in views if row["check"]["passed"]]
    return _seal(root / RECORD, {
        "schema_version": SCHEMA, "status": status, "weak_regions_digest": regions_digest,
        "pose_result_digest": pose["digest"], "plan_digest": (plan or {}).get("digest"),
        "limits": {"max_regions": MAX_REGIONS, "max_views": MAX_VIEWS, "views_per_region": VIEWS_PER_REGION,
                   "check_frames_per_view": CHECK_FRAMES_PER_VIEW,
                   "generation_context": "atlas_scene_chisel.ROLLOUT_VIEWS less two views per target"},
        "check_rule": {"min_checked_fraction": MIN_CHECKED_FRACTION,
                       "max_contradicted_fraction": MAX_CONTRADICTED_FRACTION,
                       "basis": "starting values 2026-11-18; atlas_gap_composite tolerances from capture-447ae4f4"},
        "skipped_regions": list(skipped), "batches": list(batches), "views": list(views),
        "passed_view_ids": passed, "dropped_view_count": len(views) - len(passed),
        "generated_content": True, "claim_ceiling": "development_only", **extra})


class _Passthrough(Exception):
    """A checkpoint's own exception (a withdrawal, the tick's budget), carried past the stage's fallback."""

    def __init__(self, error: BaseException):
        super().__init__(str(error))
        self.error = error


def _in_flight(root: Path) -> bool:
    """Whether any request under ``root`` was sent and has neither a result nor a proven provider failure."""
    from .atlas_native_splat import _provider_failed_and_exhausted
    from .atlas_pricing import submitted

    return any(submitted(stage) and not (stage / "result.json").is_file() and not _provider_failed_and_exhausted(stage)
               for stage in sorted((root / "generated").glob("*")))


def look_views_for_splat(*, frames, pose, output_root, policy, task_context, source_digest, configuration,
                         task_runner, bought, check=None, max_polls=2, sent=None, heldout_frame_ids=(),
                         task_frame_ids=()) -> dict:
    """``{"views": frames to send beside the real ones, "record": the sealed record or None}``, or processing.

    ``configuration`` is a callable returning the job's current allowance, read only when a map exists.
    ``sent`` are the real frames the splat is sent; each view names those overlapping it
    (``reference_frame_ids``). Generation context and the check draw only on the real frames sent, never
    on ``heldout_frame_ids``: the held-out score measures the look against views nothing generated saw.
    ``task_frame_ids`` (the job's route views) follow each region's own views in its context.
    A wall (money, privacy, rights) and a checkpoint's own stop are raised; any
    other error is recorded and the splat goes on with its real frames, unless a request it sent is still
    live, which is then raised so the next tick polls and settles it.
    """
    from .hard_stops import is_wall

    output_root = Path(output_root)
    root = output_root / ROOT
    if (root / RECORD).is_file():
        return _retained(root, pose, sent)
    if bought:
        return dict(NONE)

    def checkpoint(stage):
        if check is not None:
            try:
                check(stage)
            except BaseException as exc:  # noqa: BLE001 - re-raised as it was, past the fallback below
                raise _Passthrough(exc) from exc
    try:
        heldout = set(heldout_frame_ids)
        pool = [frame for frame in (frames if sent is None else sent)
                if frame.get("keep_mask") is None and frame["frame_id"] not in heldout]
        return _stage(frames=pool, pose=pose, output_root=output_root, root=root, policy=policy,
                      task_context=task_context, source_digest=source_digest, configuration=configuration,
                      task_runner=task_runner, checkpoint=checkpoint, max_polls=max_polls, sent=sent,
                      task_frame_ids=[key for key in task_frame_ids if key not in heldout])
    except _Passthrough as wrapped:
        raise wrapped.error from None
    except Exception as exc:  # noqa: BLE001 - only walls and live requests may hold the splat
        if is_wall(exc) or _in_flight(root):
            raise
        root.mkdir(parents=True, exist_ok=True)
        record = _record(root, None, status="failed", regions_digest=None, pose=pose, skipped=[],
                         reason=(str(exc) if str(exc).startswith("atlas_") else type(exc).__name__)[:200])
        return {"views": [], "record": record}


def _stage(*, frames, pose, output_root, root, policy, task_context, source_digest, configuration, task_runner,
           checkpoint, max_polls, sent, task_frame_ids=()):
    from .atlas_native_splat import _provider_failed_and_exhausted
    from .atlas_pricing import headroom, render_cost, submitted
    from .atlas_scene_completion import load_generated_views
    from .atlas_scene_chisel import ROLLOUT_VIEWS
    from .hard_stops import is_wall
    from .website_atlas_tasks import validate_task_request

    sealed = json.loads((root / PLAN).read_text()) if (root / PLAN).is_file() else None
    if sealed is not None:
        # The cameras were decided once: a later map never re-derives them into another purchase.
        if sealed.get("digest") != canonical_digest(sealed, digest_field="digest") or (root / PLAN).is_symlink():
            raise ValueError("atlas_look_views_plan_changed")
        targets, skipped, regions_digest = sealed["targets"], sealed["skipped_regions"], sealed["weak_regions_digest"]
    else:
        capture_root = _capture_root(output_root)
        regions, mapped_pose = _mapped_regions(capture_root) if capture_root is not None else (None, None)
        if not regions:
            return dict(NONE)
        root.mkdir(parents=True, exist_ok=True)
        regions_digest = _sha256_file(capture_root / WEAK_REGIONS_RELATIVE)
        if mapped_pose != pose["digest"]:
            # A map that names no pose, or another pose's world: its centroids are not in these cameras' frame.
            record = _record(root, None, status="weak_regions_pose_mismatch", regions_digest=regions_digest,
                             pose=pose, skipped=[], weak_regions_pose_result_digest=mapped_pose)
            return {"views": [], "record": record}
        targets, skipped = orbit_cameras(regions, frames, task_frame_ids=task_frame_ids)
    # A plan sealed on another pose buys nothing more; what it sent is still polled and settled.
    stale = sealed is not None and sealed.get("pose_result_digest") != pose["digest"]
    priced = "rgbd_view_cost_usd" in policy
    by_id = {frame["frame_id"]: frame for frame in frames}
    groups = {}
    for target in targets:
        groups.setdefault(target["region_id"], []).append(target)
    review_record = {key: policy[key] for key in ("rate_card_reference", "reviewed_by", "reviewed_on") if key in policy}
    specs, not_bought, photographs = [], [], {}
    for index, group in enumerate(groups.values()):
        stage = root / "generated" / f"{index:02d}"
        retained = _sent_binding(stage) if submitted(stage) else None
        if retained is not None:
            # A request already sent is polled exactly as it was sent, at the price it was reserved under,
            # whatever the rate card or the frames say now: a paid operation is never abandoned.
            request, uploads = retained["request_template"], retained["uploads"]
            quote = {"schema_version": "website_atlas_task_price_review.v1", "task": "atlasGenerate",
                     "request_digest": canonical_digest(request), "maximum_cost_usd": retained["maximum_cost_usd"],
                     **review_record}
        elif stale or not priced:
            not_bought.append({"region_id": group[0]["region_id"], "status": "not_bought",
                               "reason": "plan_pose_changed" if stale else "rgbd_view_price_unreviewed"})
            continue
        else:
            contexts, uploads = _context([by_id[key] for key in group[0]["context_frame_ids"]], root, photographs)
            if len(contexts) + 2 * len(group) > ROLLOUT_VIEWS:
                raise ValueError("atlas_look_views_rollout_capacity_exceeded")
            request = {"contextFrames": contexts, "targetCameras": [row["camera"] for row in group],
                       "prompt": PROMPT, "enhancePrompt": False, "returnDepth": True}
            validate_task_request("atlasGenerate", request)
            quote = {"schema_version": "website_atlas_task_price_review.v1", "task": "atlasGenerate",
                     "request_digest": canonical_digest(request), "maximum_cost_usd": render_cost(policy, len(group)),
                     **review_record}
        specs.append((group, request, uploads, stage, quote))
    if sealed is None:
        if not specs:
            record = _record(root, None, status="no_views" if not targets else "rgbd_view_price_unreviewed",
                             regions_digest=regions_digest, pose=pose, skipped=skipped)
            return {"views": [], "record": record}
        plan = {"schema_version": SCHEMA + ".plan", "weak_regions_digest": regions_digest,
                "pose_result_digest": pose["digest"], "targets": targets, "skipped_regions": skipped,
                "batches": [{"region_id": group[0]["region_id"], "frame_ids": [row["camera_id"] for row in group],
                             "stage_relative_path": str(stage.relative_to(root)),
                             "request_digest": quote["request_digest"], "maximum_cost_usd": quote["maximum_cost_usd"]}
                            for group, _, _, stage, quote in specs]}
        plan["digest"] = canonical_digest(plan, digest_field="digest")
        write_json(root / PLAN, plan)
    else:
        plan = sealed
    pending = [quote["maximum_cost_usd"] for _, _, _, stage, quote in specs if not submitted(stage)]
    if pending:
        forecast = headroom(configuration(), stages={"look_view_batches": len(pending), "whole_view_review": 1},
                            cost_usd=sum(pending), final_output_root=output_root, context=task_context,
                            source_digest=source_digest)
        if forecast is not None:
            write_json(root / "request-budget-preflight.json", forecast)
        if forecast is not None and not forecast["sufficient_for_known_calls"]:
            # Only requests not sent yet are dropped; any already sent are still polled and settled.
            specs = [spec for spec in specs if submitted(spec[3])]
            not_bought.append({"status": "not_bought", "reason": "allowance_insufficient", "forecast": forecast})
            if not specs:
                record = _record(root, plan, status="allowance_insufficient", regions_digest=regions_digest,
                                 pose=pose, skipped=skipped, forecast=forecast)
                return {"views": [], "record": record}
    # Every region's request is sent (or polled) before any view is checked, so each view is checked once.
    batches, completed = [], []
    for group, request, uploads, stage, quote in specs:
        row = {"region_id": group[0]["region_id"], "stage_relative_path": str(stage.relative_to(root)),
               "request_digest": quote["request_digest"], "maximum_cost_usd": quote["maximum_cost_usd"]}
        if _provider_failed_and_exhausted(stage):
            batches.append({**row, "status": "failed", "reason": "atlas_task_operation_failed"})
            continue
        if not submitted(stage):
            checkpoint("refined_splat_look_views")
        try:
            state = task_runner(task="atlasGenerate", request=request, uploads=uploads, output_root=stage,
                                source_video_digest=source_digest, task_context=task_context, price_review=quote,
                                max_polls=max_polls)
        except ValueError as exc:
            if is_wall(exc) or _in_flight(root):
                raise
            # Flagged and the job goes on: the region keeps its filmed views only.
            batches.append({**row, "status": "failed", "reason": str(exc)[:200]})
            continue
        batches.append({**row, "status": state["status"], "operation_id": state.get("operation_id")})
        completed.append((group, stage, batches[-1]))
    waiting = [row for row in batches if row["status"] not in ("completed", "failed")]
    if waiting:
        return {"status": "processing", "stage": "refined_splat_look_views", "operation_id": waiting[0]["operation_id"]}
    bought_cost = round(sum(row["maximum_cost_usd"] for row in batches if row["status"] == "completed"), 6)
    if stale:
        record = _record(root, plan, status="plan_changed", regions_digest=regions_digest, pose=pose,
                         skipped=skipped, batches=[*batches, *not_bought], bought_cost_usd=bought_cost)
        return {"views": [], "record": record}
    views = []
    for group, stage, row in completed:
        row.pop("operation_id", None)
        native, returned, _ = load_generated_views(stage, [target["camera_id"] for target in group])
        row.update(native_result_digest=native["digest"], billing=native.get("billing"))
        views.extend(_judged(group, returned, native, stage, root, row, by_id))
        del returned
    record = _record(root, plan, status="checked", regions_digest=regions_digest, pose=pose, skipped=skipped,
                     batches=[*batches, *not_bought], views=views, bought_cost_usd=bought_cost)
    return _retained(root, pose, sent) if record["passed_view_ids"] else {"views": [], "record": record}


def _context(frames, root: Path, photographs: dict):
    """Context photographs and their own depth for ``frames``, in order; each frame's prepared once a job.

    ``atlas_scene_generate.prepare_rgbd_context`` per frame, kept under the frame's own directory so the
    regions that share a view share its files; the asset ids are numbered in this request's order.
    """
    from .atlas_scene_generate import prepare_rgbd_context

    contexts, uploads = [], {}
    for frame in frames:
        key = frame["frame_id"]
        if key not in photographs:
            name = canonical_digest({"frame_id": key})[7:23]
            photographs[key] = prepare_rgbd_context(frames=[frame], root=root / "context" / name)[:2]
        own, files = photographs[key]
        for index, row in enumerate(own):
            number = len(contexts)
            for role in ("rgb", "depth"):
                uploads[f"context-{number}-{role}"] = files[f"context-{index}-{role}"]
            contexts.append({"imageAsset": {"assetId": f"upload:context-{number}-rgb"},
                             "depth": {"depthAsset": {"assetId": f"upload:context-{number}-depth"}},
                             "camera": row["camera"]})
    return contexts, uploads


def _sent_binding(stage: Path) -> dict | None:
    """The request, uploads and reserved price a stage's submission was sent with, or None if it names none."""
    path = stage / "atlas_pose/submission.json"
    try:
        binding = json.loads(path.read_text()).get("binding") if path.is_file() and not path.is_symlink() else None
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(binding, dict) or not all(key in binding for key in ("request_template", "uploads",
                                                                          "maximum_cost_usd")):
        return None
    return binding


def disclosure(look, native, output_root: Path) -> dict:
    """What the result says about the views: how many were made and passed, and which the look holds."""
    record = look["record"]
    passed = set(record.get("passed_view_ids") or [])
    rows = (native.get("splat_manifest") or {}).get("frames") or []
    posted = {row.get("frame_id") for row in rows if row.get("whole_view_review")}
    in_look = [key for key in record.get("passed_view_ids") or [] if key in posted]
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
