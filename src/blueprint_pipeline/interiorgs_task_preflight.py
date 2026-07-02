"""CPU preflight for task placement on an InteriorGS 3DGS (splat) scene.

This is the splat-scene sibling of :mod:`kitchen_task_scaling_preflight`: given a
scene directory (``3dgs*.ply`` + ``labels.json`` + ``structure.json`` +
``task_targets*.json``) and a robot profile, it runs the SAME dynamic-placement
chain the kitchen USD lane uses — task → target resolution → stance solving →
placement validation → task-framing cameras — entirely on CPU, and normalizes the
results into fail-closed gates. A task that passes here is ready for a paid render
(reference Spark splat render locally, Isaac/NuRec on GPU); a task that fails is
diagnosed for free.

What is different from the USD lane, and why:

* The spatial index is :class:`InteriorGSSceneSpatialIndex` (ground-truth label
  boxes), not a USD stage walk — a splat has no prims to enumerate.
* The stance probe is structure-aware (:func:`build_interiorgs_probe`): a splat
  has no collision shell, so walls come from ``structure.json`` and stances are
  constrained to the target's room polygon.
* Target resolution tries the INSTANCE tier first (``pot_88`` → ins_id 88 —
  InteriorGS task prompts embed instance names) before the label/VLM tiers.
* The splat itself is only consulted for cheap chunk-bounds statistics
  (:func:`gaussian_splat_decode.read_compressed_ply_chunk_bounds`) to cross-check
  that labels, structure, and splat agree on the world frame and floor height.

Truth boundary: passing gates prove geometry-consistent placement intent against
the labeled boxes. They do NOT prove the splat renders, that physics holds, or
that manipulation succeeds — those remain render/GPU-lane claims.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .gaussian_splat_decode import (
    convert_to_standard_ply,
    read_compressed_ply_chunk_bounds,
    read_standard_3dgs_ply,
)
from .splat_occupancy import (
    build_floor_occupancy_grid,
    refine_coarse_obstacles,
    wall_boxes_from_splat,
)
from .scene_placement import (
    DEFAULT_ROBOT_ID,
    InteriorGSSceneSpatialIndex,
    RobotProfile,
    SceneObject,
    build_interiorgs_probe,
    compute_stand_pose,
    get_robot_profile,
    ring_scan_stand_pose,
    placement_verdict_to_dict,
    robot_profile_from_json_file,
    scene_object_to_dict,
    stance_task_cameras,
    supporting_fixtures_for,
    to_splat_render_specs,
    validate_stand_pose,
)
from .scene_placement.interiorgs_index import DEFAULT_ANKLE_CLEARANCE_M
from .scene_placement.target_resolver import (
    is_openable_target,
    resolve_target_by_instance,
    resolve_target_by_label,
)

PREFLIGHT_SCHEMA_VERSION = "interiorgs_task_preflight.v1"
GATE_SET_VERSION = "interiorgs_task_preflight_gates.v1"
RENDER_HARNESS_REL = "tools/splat_render/render_splat.mjs"

# Gap between the two floor estimates (label boxes vs splat chunk bounds) beyond
# which we assume the sidecars and the splat are NOT in the same world frame.
FLOOR_CONSISTENCY_TOL_M = 0.25
# Fraction of label centroids that must fall inside the (slightly inflated) splat
# AABB for the label/splat frames to count as aligned.
LABEL_BOUNDS_MIN_FRACTION = 0.95
LABEL_BOUNDS_INFLATE_M = 0.5


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _gate(name: str, passed: bool, *, evidence: Mapping[str, Any] | None = None) -> dict:
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "evidence": dict(evidence or {}),
    }


def _skipped_gate(name: str, *, evidence: Mapping[str, Any] | None = None) -> dict:
    return {"name": name, "status": "SKIPPED", "evidence": dict(evidence or {})}


# ----------------------------- scene asset discovery -----------------------------

def discover_scene_assets(scene_dir: str | Path) -> dict[str, Optional[Path]]:
    """Locate the splat + sidecars inside an InteriorGS scene directory."""
    scene_dir = Path(scene_dir)
    splat: Optional[Path] = None
    for pattern in ("*compressed.ply", "*.ply"):
        candidates = sorted(scene_dir.glob(pattern))
        if candidates:
            splat = candidates[0]
            break
    labels: Optional[Path] = None
    for name in ("labels.json", "labels.bootstrap.json"):
        candidate = scene_dir / name
        if candidate.is_file():
            labels = candidate
            break
    structure = scene_dir / "structure.json"
    task_files = sorted(scene_dir.glob("task_targets*.json"))
    return {
        "splat": splat,
        "labels": labels,
        "structure": structure if structure.is_file() else None,
        "task_file": task_files[0] if task_files else None,
    }


def load_task_specs(path: str | Path) -> list[dict[str, Any]]:
    """Task specs from a ``task_targets*.json`` file.

    Abstract group ids (``pick_place_manipulation``) carry no object reference, so
    they are marked ``abstract`` and excluded from placement by default; prompt
    tasks ("Pick up pot_88 …") are the placement-relevant ones.
    """
    payload = json.loads(Path(path).read_text())
    tasks = payload.get("tasks") if isinstance(payload, dict) else payload
    specs: list[dict[str, Any]] = []
    for entry in tasks or []:
        if not isinstance(entry, dict):
            continue
        task_id = str(entry.get("task_id", "")).strip()
        if not task_id:
            continue
        specs.append(
            {
                "task_id": task_id,
                "source": str(entry.get("source", "")),
                "abstract": " " not in task_id,
            }
        )
    return specs


def select_task_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    only: Sequence[str] = (),
    limit: int | None = None,
    include_abstract: bool = False,
) -> list[dict[str, Any]]:
    out = [dict(s) for s in specs if include_abstract or not s.get("abstract")]
    if only:
        needles = [n.strip().lower() for n in only if n.strip()]
        out = [s for s in out if any(n in s["task_id"].lower() for n in needles)]
    if limit is not None and limit >= 0:
        out = out[:limit]
    return out


# ----------------------------- scene-level gates -----------------------------

def evaluate_scene_gates(
    *,
    splat_path: Optional[Path],
    index: Optional[InteriorGSSceneSpatialIndex],
) -> tuple[list[dict], dict[str, Any]]:
    """Cross-check splat / labels / structure agree before spending per-task work."""
    gates: list[dict] = []
    context: dict[str, Any] = {}

    chunk_bounds = None
    if splat_path is None or not splat_path.is_file():
        gates.append(_gate("splat_asset_present", False, evidence={"splat": str(splat_path)}))
    else:
        gates.append(_gate("splat_asset_present", True, evidence={"splat": str(splat_path)}))
        try:
            chunk_bounds = read_compressed_ply_chunk_bounds(splat_path)
        except Exception as exc:  # noqa: BLE001 - decode failure is a gate, not a crash
            gates.append(
                _gate("splat_chunk_bounds_readable", False, evidence={"error": str(exc)})
            )
        else:
            aabb_min, aabb_max = chunk_bounds.aabb()
            context["splat_aabb_min"] = [round(float(v), 4) for v in aabb_min]
            context["splat_aabb_max"] = [round(float(v), 4) for v in aabb_max]
            context["splat_floor_z_estimate"] = round(chunk_bounds.floor_z_estimate(), 4)
            gates.append(
                _gate(
                    "splat_chunk_bounds_readable",
                    True,
                    evidence={
                        "chunk_count": chunk_bounds.chunk_count,
                        "vertex_count": chunk_bounds.vertex_count,
                        "aabb_min": context["splat_aabb_min"],
                        "aabb_max": context["splat_aabb_max"],
                        "floor_z_estimate": context["splat_floor_z_estimate"],
                    },
                )
            )

    if index is None:
        gates.append(_gate("labels_loaded", False, evidence={"reason": "labels.json missing/unreadable"}))
        return gates, context

    objects = index.objects()
    gates.append(
        _gate(
            "labels_loaded",
            bool(objects),
            evidence={"object_count": len(objects), "floor_z": index.floor_z},
        )
    )
    if index.structure is not None:
        gates.append(
            _gate(
                "structure_loaded",
                bool(index.structure.rooms) and bool(index.structure.wall_boxes),
                evidence={
                    "room_count": len(index.structure.rooms),
                    "wall_count": len(index.structure.wall_boxes),
                    "hole_count": len(index.structure.holes),
                },
            )
        )
    else:
        gates.append(_skipped_gate("structure_loaded", evidence={"reason": "no structure.json"}))

    if chunk_bounds is not None and objects:
        aabb_min, aabb_max = chunk_bounds.aabb()
        lo = [float(v) - LABEL_BOUNDS_INFLATE_M for v in aabb_min]
        hi = [float(v) + LABEL_BOUNDS_INFLATE_M for v in aabb_max]
        inside = sum(
            1
            for obj in objects
            if all(lo[i] <= float(obj.centroid[i]) <= hi[i] for i in range(3))
        )
        fraction = inside / len(objects)
        gates.append(
            _gate(
                "labels_within_splat_bounds",
                fraction >= LABEL_BOUNDS_MIN_FRACTION,
                evidence={
                    "inside": inside,
                    "total": len(objects),
                    "fraction": round(fraction, 4),
                    "min_fraction": LABEL_BOUNDS_MIN_FRACTION,
                },
            )
        )
        splat_floor = chunk_bounds.floor_z_estimate()
        delta = abs(splat_floor - index.floor_z)
        gates.append(
            _gate(
                "floor_consistency",
                delta <= FLOOR_CONSISTENCY_TOL_M,
                evidence={
                    "labels_floor_z": round(index.floor_z, 4),
                    "splat_floor_z_estimate": round(splat_floor, 4),
                    "delta_m": round(delta, 4),
                    "tolerance_m": FLOOR_CONSISTENCY_TOL_M,
                },
            )
        )
    else:
        gates.append(_skipped_gate("labels_within_splat_bounds"))
        gates.append(_skipped_gate("floor_consistency"))
    return gates, context


# ----------------------------- per-task evaluation -----------------------------

def _xy_gap_to_box(
    f_min: tuple[float, float],
    f_max: tuple[float, float],
    obj: SceneObject,
) -> float:
    dx = max(obj.bbox_min[0] - f_max[0], f_min[0] - obj.bbox_max[0], 0.0)
    dy = max(obj.bbox_min[1] - f_max[1], f_min[1] - obj.bbox_max[1], 0.0)
    return math.hypot(dx, dy)


def _reach_envelope_check(
    pose_position: tuple[float, float, float],
    target: SceneObject,
    *,
    floor_z: float,
    profile: RobotProfile,
    openable: bool = False,
) -> dict[str, Any]:
    """Heuristic single-arm reach check for the solved stance.

    The affordance proxy is the target centroid. Reach budget = arm span +
    forward shoulder offset + end-effector slack; height window = [ankle-ish,
    shoulder + arm]. This mirrors the kitchen lane's reach gate at preview
    fidelity — a FAIL means the target is geometrically out of envelope from the
    stance (e.g. a ceiling-mounted unit), not that IK failed.
    """
    px, py, _ = (float(v) for v in pose_position)
    hx = float(profile.footprint_half_extent_xyz[0])
    hy = float(profile.footprint_half_extent_xyz[1])
    gap = _xy_gap_to_box((px - hx, py - hy), (px + hx, py + hy), target)
    reach_budget = (
        float(profile.arm_span_m)
        + float(profile.shoulder_forward_offset_m)
        + float(profile.max_effector_to_affordance_m)
    )
    if openable:
        # The stance deliberately backs off by the door/drawer swing margin; the
        # robot closes that distance as the fixture opens, so the margin counts
        # toward reach rather than against it.
        reach_budget += max(0.0, float(profile.openable_standoff_extra_m))
    shoulder_z = floor_z + float(profile.pelvis_height_m) + float(profile.shoulder_above_root_m)
    z_lo = floor_z + 0.10
    z_hi = shoulder_z + float(profile.arm_span_m)
    affordance_z = float(target.centroid[2])
    horizontal_ok = gap <= reach_budget
    vertical_ok = z_lo <= affordance_z <= z_hi
    return {
        "ok": horizontal_ok and vertical_ok,
        "gap_to_target_m": round(gap, 4),
        "reach_budget_m": round(reach_budget, 4),
        "affordance_z": round(affordance_z, 4),
        "reach_z_window": [round(z_lo, 4), round(z_hi, 4)],
        "horizontal_ok": horizontal_ok,
        "vertical_ok": vertical_ok,
    }


def evaluate_task(
    index: InteriorGSSceneSpatialIndex,
    task_id: str,
    *,
    profile: RobotProfile,
    obstacles: Optional[List[SceneObject]] = None,
    region_of=None,
    generate=None,
) -> dict[str, Any]:
    """Run resolution → stance → validation → cameras for one task; return the report.

    ``region_of`` is an optional ``(x, y) tuple -> Optional[int]`` floor-region
    lookup that supersedes structure.json room polygons (used when the scene has
    no structure file and connectivity comes from splat free-space components).
    """
    objects = index.objects()
    gates: list[dict] = []

    target = resolve_target_by_instance(task_id, objects)
    method = "instance"
    if target is None:
        target = resolve_target_by_label(task_id, objects)
        method = "label"
    gates.append(
        _gate(
            "target_resolved",
            target is not None,
            evidence=(
                {"method": method, "target_id": target.id, "label": target.label}
                if target is not None
                else {"reason": "no object matched task"}
            ),
        )
    )
    report: dict[str, Any] = {"task_id": task_id, "gates": gates}
    if target is None:
        report["all_gates_passed"] = False
        return report
    report["target"] = scene_object_to_dict(target)

    floor_z = index.floor_z
    if obstacles is None:
        obstacles = index.obstacle_boxes()
    # Same walk-over rule as the probe: carpets/rugs under the ankle band are not
    # placement obstacles (the shared validator has no ankle rule of its own).
    walkable_top = floor_z + DEFAULT_ANKLE_CLEARANCE_M
    placement_obstacles = [o for o in obstacles if o.max_z() > walkable_top]
    fixtures = supporting_fixtures_for(target, placement_obstacles)
    probe = build_interiorgs_probe(
        index,
        target=target,
        robot_profile=profile,
        standoff_obstacles=fixtures,
        obstacle_boxes=placement_obstacles,
        region_of=region_of,
    )
    openable = is_openable_target(target)
    # The validator measures standoff as the FOOTPRINT-edge -> box-edge gap, while
    # the solver's standing_distance is pelvis-center -> box-edge. Derive a standing
    # distance that lands the footprint gap just above the profile's standoff floor
    # (plus the near-clearance margin), instead of trusting the raw profile default.
    standoff_lo = float(profile.standoff_range_m[0])
    front_half = max(
        float(profile.footprint_half_extent_xyz[0]),
        float(profile.footprint_half_extent_xyz[1]),
    )
    standing_distance = max(
        float(profile.standing_distance_m),
        standoff_lo + front_half + float(profile.min_obstacle_clearance_m) + 0.05,
    )
    pose = compute_stand_pose(
        target,
        probe=probe,
        floor_z=floor_z,
        standing_distance=standing_distance,
        include_diagonals=True,
        openable_target=openable,
        robot_profile=profile,
    )
    stance_method = "ray_probe"
    if not pose.clear:
        # Ray probing marches straight through the target's center, which fails
        # for targets pinned against a wall (the ray clips the wall band even
        # though a laterally-offset spot is open). Scan the full standoff
        # annulus before declaring the target unplaceable.
        effective_sd = standing_distance + (
            float(profile.openable_standoff_extra_m) if openable else 0.0
        )
        ring_pose = ring_scan_stand_pose(
            target,
            probe=probe,
            floor_z=floor_z,
            standing_distance=effective_sd,
            max_standing_distance=float(profile.standoff_range_m[1]) + front_half,
            robot_profile=profile,
        )
        if ring_pose.clear:
            pose = ring_pose
            stance_method = "ring_scan"
    def _validate(candidate):
        return validate_stand_pose(
            candidate.position,
            candidate.yaw,
            target,
            placement_obstacles,
            floor_z,
            standoff_obstacles=fixtures or None,
            robot_profile=profile,
        )

    verdict = _validate(pose)
    if pose.clear and not verdict.ok and stance_method == "ray_probe":
        # A probe-clear ray stance can still fail validation (e.g. the nearest
        # ray spot sits past the standoff ceiling in a crowded entryway). The
        # annulus usually holds a compliant spot the rays skipped.
        effective_sd = standing_distance + (
            float(profile.openable_standoff_extra_m) if openable else 0.0
        )
        ring_pose = ring_scan_stand_pose(
            target,
            probe=probe,
            floor_z=floor_z,
            standing_distance=effective_sd,
            max_standing_distance=float(profile.standoff_range_m[1]) + front_half,
            robot_profile=profile,
        )
        if ring_pose.clear:
            ring_verdict = _validate(ring_pose)
            if ring_verdict.ok:
                pose, verdict, stance_method = ring_pose, ring_verdict, "ring_scan"

    report["stance"] = {
        "method": stance_method,
        "position": [round(float(v), 4) for v in pose.position],
        "yaw_rad": round(float(pose.yaw), 4),
        "clear": bool(pose.clear),
        "standoff_m": round(float(pose.standoff_m), 4),
        "openable_target": openable,
        "notes": pose.notes,
    }
    gates.append(
        _gate("stance_found_clear", bool(pose.clear), evidence={"notes": pose.notes})
    )
    report["placement_validation"] = placement_verdict_to_dict(verdict)
    report["standoff_fixture_ids"] = [f.id for f in fixtures]
    gates.append(
        _gate(
            "placement_validated",
            bool(verdict.ok),
            evidence={"failures": list(verdict.failures)},
        )
    )

    structure = index.structure
    room_lookup = region_of
    room_source = "free_space_components"
    if room_lookup is None and structure is not None and structure.rooms:
        room_lookup = structure.room_index_of_point
        room_source = "structure_rooms"
    if room_lookup is not None:
        stance_room = room_lookup((pose.position[0], pose.position[1]))
        target_room = room_lookup(
            (float(target.centroid[0]), float(target.centroid[1]))
        )
        same_room = stance_room is not None and (
            target_room is None or stance_room == target_room
        )
        gates.append(
            _gate(
                "stance_in_target_room",
                same_room,
                evidence={
                    "stance_room": stance_room,
                    "target_room": target_room,
                    "source": room_source,
                },
            )
        )
    else:
        gates.append(_skipped_gate("stance_in_target_room"))

    reach = _reach_envelope_check(
        pose.position, target, floor_z=floor_z, profile=profile, openable=openable
    )
    gates.append(_gate("target_within_reach_envelope", bool(reach.pop("ok")), evidence=reach))

    ceiling = None
    if structure is not None:
        ceiling = floor_z + structure.wall_height_m

    def _eye_clear(point) -> bool:
        # A camera eye is clear when it sits inside no obstacle/wall volume.
        x, y, z = (float(v) for v in point)
        for obs in placement_obstacles:
            if (
                obs.bbox_min[0] <= x <= obs.bbox_max[0]
                and obs.bbox_min[1] <= y <= obs.bbox_max[1]
                and obs.bbox_min[2] <= z <= obs.bbox_max[2]
            ):
                return False
        return True

    cameras = stance_task_cameras(
        pose, target, floor_z=floor_z, robot_profile=profile, ceiling_z=ceiling,
        eye_clear_fn=_eye_clear,
    )
    cams_finite = all(
        math.isfinite(float(v))
        for cam in cameras.values()
        for key in ("eye", "target")
        for v in cam[key]  # type: ignore[union-attr]
    )
    gates.append(
        _gate("cameras_built", bool(cameras) and cams_finite, evidence={"camera_ids": sorted(cameras)})
    )
    report["cameras"] = {
        cam_id: {
            "eye": [round(float(v), 4) for v in cam["eye"]],  # type: ignore[union-attr]
            "target": [round(float(v), 4) for v in cam["target"]],  # type: ignore[union-attr]
            "vfov_rad": round(float(cam["vfov"]), 4),  # type: ignore[arg-type]
            "width": cam["width"],
            "height": cam["height"],
        }
        for cam_id, cam in cameras.items()
    }
    report["splat_render_cameras"] = to_splat_render_specs(cameras)
    report["all_gates_passed"] = all(g["status"] == "PASS" for g in gates)
    return report


# ----------------------------- splat-occupancy refinement -----------------------------

def build_refined_obstacles(
    index: InteriorGSSceneSpatialIndex,
    splat_path: Path,
    cache_dir: Path,
    *,
    repo_root: Path | None = None,
) -> tuple[Optional[List[SceneObject]], dict[str, Any], Any]:
    """Decode the splat (cached) and carve jumbo label AABBs into occupancy columns.

    Coarse label boxes around L-shaped fixtures swallow the open floor in front
    of them (the whole-kitchen ``cupboard`` box); the splat's actual mass carves
    them back to the occupied footprint. Fail-open: when the decode toolchain is
    unavailable the caller falls back to the coarse boxes with an explicit note —
    never a fabricated refinement.

    Returns ``(refined_obstacles, report, decoded_splat)``; the decoded
    :class:`SplatData` is reused by callers for occupancy walls / free-space
    connectivity so the PLY is only decoded once per run.
    """
    ensure_dir(cache_dir)
    decoded = cache_dir / "decoded_standard_3dgs.ply"
    if not decoded.is_file():
        status = convert_to_standard_ply(
            splat_path, decoded, repo_root=repo_root or _repo_root()
        )
        if status.get("status") != "completed":
            return None, {"status": "blocked", **status}, None
    try:
        splat = read_standard_3dgs_ply(decoded)
    except Exception as exc:  # noqa: BLE001 - surfaces as a skipped refinement
        return (
            None,
            {"status": "blocked", "blockers": ["decoded_ply_unreadable"], "error": str(exc)},
            None,
        )
    refined, report = refine_coarse_obstacles(
        index.obstacle_boxes(), splat, floor_z=index.floor_z
    )
    return refined, {"status": "completed", **report}, splat


# ----------------------------- optional local splat render -----------------------------

def render_task_views(
    splat_path: Path,
    task_report: Mapping[str, Any],
    out_dir: Path,
    *,
    repo_root: Path | None = None,
    node: str = "node",
    width: int = 1280,
    height: int = 960,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Render the task's stance cameras against the real splat via the local Spark harness.

    Local, GPU-free (headless Chromium), fail-closed. This proves the FRAMING
    against the real capture; it is labeled ``reference_spark_renderer``, not an
    Isaac render.
    """
    root = repo_root or _repo_root()
    harness = root / RENDER_HARNESS_REL
    if not harness.is_file():
        return {"status": "blocked", "blockers": ["splat_render_harness_missing"], "harness": str(harness)}
    cameras = task_report.get("splat_render_cameras") or []
    if not cameras:
        return {"status": "blocked", "blockers": ["no_cameras_in_task_report"]}
    ensure_dir(out_dir)
    cameras_path = out_dir / "cameras.json"
    cameras_path.write_text(json.dumps(cameras, indent=2))
    cmd = [
        node,
        str(harness),
        "--splat",
        str(splat_path),
        "--out",
        str(out_dir),
        "--cameras",
        str(cameras_path),
        "--width",
        str(int(width)),
        "--height",
        str(int(height)),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    except FileNotFoundError:
        return {"status": "blocked", "blockers": ["node_runtime_unavailable"]}
    except subprocess.TimeoutExpired:
        return {"status": "blocked", "blockers": ["splat_render_timeout"]}
    manifest: dict[str, Any] = {}
    try:
        manifest = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001 - stdout may carry no JSON on failure
        manifest = {}
    if proc.returncode != 0 or manifest.get("status") not in {"completed", "ok"}:
        return {
            "status": "blocked",
            "blockers": ["splat_render_failed"],
            "returncode": proc.returncode,
            "stderr_tail": (proc.stderr or "")[-2000:],
            "harness_manifest": manifest,
        }
    return {
        "status": "completed",
        "rendered_by": "reference_spark_renderer",
        "out_dir": str(out_dir),
        "harness_manifest": manifest,
    }


# ----------------------------- orchestration -----------------------------

def run_preflight(
    *,
    scene_dir: str | Path | None = None,
    splat: str | Path | None = None,
    labels: str | Path | None = None,
    structure: str | Path | None = None,
    task_file: str | Path | None = None,
    tasks: Sequence[str] = (),
    limit: int | None = None,
    include_abstract: bool = False,
    robot_id: str | None = None,
    robot_profile_json: str | Path | None = None,
    out_dir: str | Path,
    splat_refine: bool = True,
    bootstrap_missing_sidecars: bool = False,
    bootstrap_detector=None,
    render_views: bool = False,
    render_limit: int = 1,
    render_timeout_seconds: int = 900,
    generate=None,
) -> dict[str, Any]:
    """End-to-end CPU preflight; writes ``preflight_manifest.json`` under ``out_dir``.

    With ``bootstrap_missing_sidecars`` a bare PLY is enough: missing labels /
    tasks are generated by :func:`splat_scene_bootstrap.bootstrap_scene_sidecars`
    (Spark views + VLM detection + splat depth), and missing structure geometry
    is substituted at runtime from splat occupancy (wall boxes + free-space
    connectivity), so the SAME gate chain runs label-free.
    """
    out_path = Path(out_dir)
    ensure_dir(out_path)
    assets = discover_scene_assets(scene_dir) if scene_dir else {}
    splat_path = Path(splat) if splat else assets.get("splat")
    labels_path = Path(labels) if labels else assets.get("labels")
    structure_path = Path(structure) if structure else assets.get("structure")
    task_path = Path(task_file) if task_file else assets.get("task_file")

    bootstrap_report: Optional[dict[str, Any]] = None
    if (
        bootstrap_missing_sidecars
        and labels_path is None
        and splat_path is not None
        and Path(splat_path).is_file()
    ):
        from .splat_scene_bootstrap import bootstrap_scene_sidecars

        bootstrap_report = bootstrap_scene_sidecars(
            splat_path, out_path / "bootstrap", detector=bootstrap_detector
        )
        if bootstrap_report.get("status") == "completed":
            labels_path = Path(bootstrap_report["labels_path"])
            if task_path is None:
                task_path = Path(bootstrap_report["task_targets_path"])

    if robot_profile_json:
        profile = robot_profile_from_json_file(robot_profile_json)
    else:
        profile = get_robot_profile(robot_id or DEFAULT_ROBOT_ID)

    manifest: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "gate_set_version": GATE_SET_VERSION,
        "generated_at": utc_now_iso(),
        "scene_dir": str(scene_dir) if scene_dir else None,
        "assets": {
            "splat": str(splat_path) if splat_path else None,
            "labels": str(labels_path) if labels_path else None,
            "structure": str(structure_path) if structure_path else None,
            "task_file": str(task_path) if task_path else None,
        },
        "robot_profile_id": profile.robot_id,
        "truth_boundary": (
            "CPU geometry preflight against labeled boxes + structure polygons. "
            "Proves placement intent, not rendering, physics, or manipulation success."
        ),
    }
    if bootstrap_report is not None:
        manifest["sidecar_bootstrap"] = {
            k: v for k, v in bootstrap_report.items() if k != "steps"
        }
        manifest["sidecar_bootstrap"]["steps"] = {
            name: step.get("status") if isinstance(step, dict) else step
            for name, step in (bootstrap_report.get("steps") or {}).items()
        }

    index: Optional[InteriorGSSceneSpatialIndex] = None
    if labels_path is not None and Path(labels_path).is_file():
        # Bootstrapped labels are unprojected detections whose box bottoms are
        # noisier than curated exports; trust the splat's own floor estimate.
        floor_override = (
            bootstrap_report.get("floor_z")
            if bootstrap_report is not None and bootstrap_report.get("status") == "completed"
            else None
        )
        try:
            index = InteriorGSSceneSpatialIndex(
                labels_path,
                structure_path if structure_path and Path(structure_path).is_file() else None,
                floor_z=floor_override,
            )
        except Exception as exc:  # noqa: BLE001 - surfaces as a failed scene gate
            manifest["labels_error"] = str(exc)

    scene_gates, scene_context = evaluate_scene_gates(splat_path=splat_path, index=index)
    manifest["scene_gates"] = scene_gates
    manifest["scene_context"] = scene_context
    scene_ok = all(g["status"] in {"PASS", "SKIPPED"} for g in scene_gates)
    manifest["scene_gates_passed"] = scene_ok

    refined_obstacles: Optional[List[SceneObject]] = None
    decoded_splat = None
    if splat_refine and index is not None and splat_path is not None and Path(splat_path).is_file():
        refined_obstacles, refine_report, decoded_splat = build_refined_obstacles(
            index, Path(splat_path), out_path / "cache"
        )
        manifest["splat_occupancy_refinement"] = refine_report
    elif not splat_refine:
        manifest["splat_occupancy_refinement"] = {"status": "disabled"}

    # No structure.json: substitute wall geometry + floor connectivity from splat
    # occupancy so the probe still cannot step through walls or across rooms.
    region_of = None
    if index is not None and index.structure is None and decoded_splat is not None:
        splat_walls = wall_boxes_from_splat(decoded_splat, floor_z=index.floor_z)
        base_obstacles = (
            refined_obstacles if refined_obstacles is not None else index.obstacle_boxes()
        )
        refined_obstacles = list(base_obstacles) + splat_walls
        grid = build_floor_occupancy_grid(decoded_splat, floor_z=index.floor_z)
        region_of = grid.region_of_fn()
        manifest["occupancy_structure_substitute"] = {
            "wall_boxes": len(splat_walls),
            "free_space_regions": True,
        }

    task_reports: list[dict[str, Any]] = []
    if index is not None:
        specs: list[dict[str, Any]] = []
        if task_path is not None and Path(task_path).is_file():
            specs = load_task_specs(task_path)
        elif tasks:
            specs = [{"task_id": t, "source": "cli", "abstract": False} for t in tasks]
        selected = select_task_specs(
            specs, only=tasks if task_path is not None else (), limit=limit,
            include_abstract=include_abstract,
        )
        for spec in selected:
            task_reports.append(
                evaluate_task(
                    index,
                    spec["task_id"],
                    profile=profile,
                    obstacles=refined_obstacles,
                    region_of=region_of,
                    generate=generate,
                )
            )
    manifest["tasks"] = task_reports
    passed = [t for t in task_reports if t.get("all_gates_passed")]
    manifest["summary"] = {
        "tasks_evaluated": len(task_reports),
        "tasks_passed": len(passed),
        "tasks_failed": len(task_reports) - len(passed),
        "scene_gates_passed": scene_ok,
    }

    # Persist CPU placement proof before optional rendering. Local Spark rendering
    # is advisory and may be slow or unavailable; it must not hide the preflight
    # manifest needed for provider handoff.
    manifest_path = out_path / "preflight_manifest.json"
    write_json(manifest_path, manifest)

    if render_views and splat_path is not None and passed:
        renders: list[dict[str, Any]] = []
        for task_report in passed[: max(0, int(render_limit))]:
            slug = "".join(
                ch if ch.isalnum() else "_" for ch in task_report["task_id"].lower()
            )[:60].strip("_")
            renders.append(
                {
                    "task_id": task_report["task_id"],
                    **render_task_views(
                        splat_path,
                        task_report,
                        out_path / "renders" / slug,
                        timeout_seconds=render_timeout_seconds,
                    ),
                }
            )
        manifest["splat_renders"] = renders

    write_json(manifest_path, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU task-placement preflight for an InteriorGS 3DGS scene",
    )
    parser.add_argument("--scene-dir", help="scene directory holding the .ply + sidecars")
    parser.add_argument("--splat", help="explicit path to the 3DGS .ply")
    parser.add_argument("--labels", help="explicit path to labels.json")
    parser.add_argument("--structure", help="explicit path to structure.json")
    parser.add_argument("--task-file", help="explicit path to task_targets*.json")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="task_id substring filter (repeatable); with no --task-file, a literal task string",
    )
    parser.add_argument("--limit", type=int, default=None, help="max tasks to evaluate")
    parser.add_argument(
        "--include-abstract",
        action="store_true",
        help="also evaluate abstract group task ids (no object reference)",
    )
    parser.add_argument("--robot-id", default=None, help="registered robot profile id")
    parser.add_argument("--robot-profile-json", default=None, help="robot profile JSON file")
    parser.add_argument("--out", required=True, help="output directory for the manifest")
    parser.add_argument(
        "--no-splat-refine",
        action="store_true",
        help="skip carving jumbo label boxes with splat occupancy (use coarse AABBs)",
    )
    parser.add_argument(
        "--bootstrap-missing-sidecars",
        action="store_true",
        help=(
            "when labels.json is missing, auto-generate labels + tasks from the "
            "bare PLY (Spark views + Gemini detection + splat depth)"
        ),
    )
    parser.add_argument(
        "--render-views",
        action="store_true",
        help="render stance cameras for passing tasks via the local Spark splat harness",
    )
    parser.add_argument("--render-limit", type=int, default=1)
    parser.add_argument(
        "--render-timeout-seconds",
        type=int,
        default=900,
        help="outer timeout for each optional local splat render command",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.scene_dir and not (args.splat or args.labels):
        raise SystemExit("provide --scene-dir or explicit --splat/--labels paths")
    manifest = run_preflight(
        scene_dir=args.scene_dir,
        splat=args.splat,
        labels=args.labels,
        structure=args.structure,
        task_file=args.task_file,
        tasks=args.task,
        limit=args.limit,
        include_abstract=args.include_abstract,
        robot_id=args.robot_id,
        robot_profile_json=args.robot_profile_json,
        out_dir=args.out,
        splat_refine=not args.no_splat_refine,
        bootstrap_missing_sidecars=args.bootstrap_missing_sidecars,
        render_views=args.render_views,
        render_limit=args.render_limit,
        render_timeout_seconds=args.render_timeout_seconds,
    )
    summary = manifest["summary"]
    print(
        json.dumps(
            {
                "scene_gates_passed": summary["scene_gates_passed"],
                "tasks_evaluated": summary["tasks_evaluated"],
                "tasks_passed": summary["tasks_passed"],
                "manifest": str(Path(args.out) / "preflight_manifest.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
