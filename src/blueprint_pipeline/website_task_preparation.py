"""Bridge website capture outputs into the production scene-configuration path.

Steps 10-14 of the website design already exist as the Task Evaluation
scene-configuration run: Content Agents/Astra replacement authoring, SimReady
static and native Isaac qualification, scene assembly, configured-scene
publication, and the Franka DROID controls. This module only compiles the
website preparation (confirmed task, SAM 3.1 masks, MapAnything estimates,
clean-plate removal manifest, Marble world) into that path's typed inputs.
Every scale here is estimated; nothing is promoted to a physical measurement.
"""

from __future__ import annotations

import base64
import io
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from . import task_evaluation_scene_intake as intake
from .adp009d_newton_gripper_drive import FULL_STROKE_M
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .external_scene_frame_registration import _axis_rotations, _sample, _trimmed_rmse
from .local_reconstruction_adapters import _sha256_file
from .website_assembly_coverage import assembly_constraints, assembly_contract, coverage_blockers, coverage_matches
from .website_task_masks import decode_track_mask, estimate_target_bounds
from .website_support_geometry import support_under

SCHEMA_VERSION = "website_scene_preparation.v1"
CLAIM_CEILING = "development_only"
_UP_INDEX = {"Y": 1, "-Y": 1, "Z": 2}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
# Robotiq 2F-85 datasheet: 85 mm stroke, 20-235 N grip force. Reference only.
GRIPPER = {"model": "robotiq_2f85", "full_stroke_m": FULL_STROKE_M,
           "grip_force_n": {"lower": 20.0, "upper": 235.0}}
SUCCESS = {"control_frequency_hz": 15, "maximum_episode_seconds": 24.0, "minimum_lift_m": 0.05,
           "pregrasp_clearance_m": 0.1, "minimum_planar_displacement_m": 0.1,
           "maximum_final_planar_target_error_m": 0.05, "maximum_retries": 0, "maximum_regrasps": 0}
ARTICULATED_SUCCESS = {"control_frequency_hz": 15, "maximum_episode_seconds": 30,
                       "minimum_opening_fraction_of_estimated_stroke": 0.6, "minimum_hold_seconds": 1.0,
                       "maximum_retries": 0}
# Usable travel priors for an unobserved mechanism: a drawer on ordinary slides
# opens about three quarters of its depth; a door swings about a right angle.
DRAWER_USABLE_STROKE_FRACTION_OF_DEPTH = 0.75
DOOR_USABLE_SWING_RAD = math.pi / 2
# Wide rigid-object priors: estimated bounds only, never measured values.
PHYSICS_PRIORS = {"density_kg_m3": [50.0, 2500.0], "envelope_fill": [0.2, 1.0],
                  "static_friction": [0.2, 0.9], "dynamic_friction": [0.15, 0.8], "restitution": [0.0, 0.3]}
DIMENSION_RELATIVE_ERROR = 0.25


def _source_points(source_geometry: Mapping[str, Any], *, per_frame: int = 4000) -> np.ndarray:
    points = []
    for frame in source_geometry["frames"]:
        if _sha256_file(Path(frame["geometry_path"])) != frame["geometry_digest"]:
            raise ValueError("website_source_geometry_changed")
        with np.load(frame["geometry_path"], allow_pickle=False) as geometry:
            depth, valid = geometry["depth_m"], geometry["valid_mask"]
        ys, xs = np.nonzero(valid)
        if not len(xs):
            continue
        keep = np.random.default_rng(len(points)).choice(len(xs), min(per_frame, len(xs)), replace=False)
        xs, ys = xs[keep], ys[keep]
        pixels = np.stack([xs, ys, np.ones_like(xs)], axis=1)
        camera = (pixels @ np.linalg.inv(np.asarray(frame["intrinsics"], dtype=np.float64)).T) * depth[ys, xs, None]
        pose = np.asarray(frame["world_from_camera"], dtype=np.float64)
        points.append(camera @ pose[:3, :3].T + pose[:3, 3])
    if not points:
        raise ValueError("website_source_geometry_empty")
    return np.concatenate(points)


def _mesh_vertices(path: Path) -> np.ndarray:
    import trimesh

    loaded = trimesh.load(str(path), force="mesh", process=False)
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    if len(vertices) < 8:
        raise ValueError("website_base_mesh_empty")
    return vertices


def _refine_registration(source: np.ndarray, target: np.ndarray, scale: float,
                         rotation: np.ndarray, translation: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Trimmed similarity ICP; a candidate still needs fit and ambiguity checks."""
    tree = cKDTree(target)
    initial_scale = scale
    for _ in range(30):
        moved = scale * (source @ rotation.T) + translation
        distances, indices = tree.query(moved, k=1)
        keep = distances <= np.quantile(distances, 0.8)
        a, b = source[keep], target[indices[keep]]
        ac, bc = a.mean(axis=0), b.mean(axis=0)
        a, b = a - ac, b - bc
        u, singular, vt = np.linalg.svd(b.T @ a)
        if singular[1] <= 1e-12:
            break
        signs = np.ones(3)
        signs[-1] = np.linalg.det(u @ vt)
        next_rotation = (u * signs) @ vt
        next_scale = float(np.dot(singular, signs) / np.sum(a * a))
        if not 0.5 * initial_scale <= next_scale <= 2 * initial_scale:
            break
        next_translation = bc - next_scale * (next_rotation @ ac)
        next_moved = next_scale * (source @ next_rotation.T) + next_translation
        change = np.max(np.linalg.norm(next_moved - moved, axis=1))
        scale, rotation, translation = next_scale, next_rotation, next_translation
        if change < 1e-7:
            break
    return scale, rotation, translation


_ROLLS = (0.0, 90.0, -90.0, 180.0)
ANCHOR_MAX_ROTATION_DEGREES = 45.0
ANCHOR_SCALE_RATIO_BOUNDS = (0.75, 4.0 / 3.0)
GROUND_PLANE_MAX_RESIDUAL_M = 0.35


def _roll(degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_degrees(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(a @ b.T) - 1.0) / 2.0))))


def _about(axis: int, degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    i, j = [k for k in range(3) if k != axis]
    turn = np.eye(3)
    turn[i, i], turn[i, j], turn[j, i], turn[j, j] = c, -s, s, c
    return turn


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-9:
        raise ValueError("website_registration_anchor_frame_invalid")
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


GROUND_SEARCH_MAX_TILT_DEGREES = 45.0
GROUND_SEARCH_TOLERANCE_M = 0.03


def _observed_declared_ground(source: np.ndarray, camera_to_world: np.ndarray, offset_m: float) -> np.ndarray | None:
    """Unit downward normal of the dominant downward-facing plane, if it lies at the declared camera height.

    A generated world that declares a ground plane at ``y = offset`` is
    gravity-levelled: its origin is the first view's camera centre but its
    axes are not the camera's tilted axes. The source must show that floor
    (below the camera, within the declared-scale tolerance of the declared
    height) for the camera's tilt to be removed; otherwise ``None``.
    """
    centre, camera_down = camera_to_world[:3, 3], camera_to_world[:3, 1]
    low, high = (bound * offset_m for bound in ANCHOR_SCALE_RATIO_BOUNDS)
    rng = np.random.default_rng(604)
    best, best_count = None, int(0.03 * len(source))
    for _ in range(2000):
        a, b, c = source[rng.choice(len(source), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) <= 1e-9:
            continue
        normal = normal / np.linalg.norm(normal)
        normal = normal if normal @ camera_down >= 0 else -normal
        if normal @ camera_down < math.cos(math.radians(GROUND_SEARCH_MAX_TILT_DEGREES)) or normal @ (a - centre) <= 0:
            continue
        count = int(np.sum(np.abs((source - a) @ normal) <= GROUND_SEARCH_TOLERANCE_M))
        if count > best_count:
            best, best_count = (a, normal), count
    if best is None:
        return None
    inliers = source[np.abs((source - best[0]) @ best[1]) <= GROUND_SEARCH_TOLERANCE_M]
    normal = np.linalg.svd(inliers - inliers.mean(axis=0))[2][-1]
    normal = normal if normal @ camera_down >= 0 else -normal
    # The dominant downward-facing surface is the floor only at the declared height.
    return normal if low <= float(normal @ (inliers.mean(axis=0) - centre)) <= high else None


def _anchor_prior(source_geometry: Mapping[str, Any], anchor: Mapping[str, Any]) -> tuple[float, np.ndarray]:
    """The provider world is anchored at its first input view's camera, in the same OpenCV convention."""
    frame = next((row for row in source_geometry["frames"] if row["frame_id"] == anchor.get("frame_id")), None)
    if frame is None:
        raise ValueError("website_registration_anchor_frame_missing")
    mpu = anchor.get("meters_per_unit")
    if isinstance(mpu, bool) or not isinstance(mpu, (int, float)) or not math.isfinite(mpu) or mpu <= 0:
        raise ValueError("website_base_scene_scale_invalid")
    if anchor.get("up_axis") not in _UP_INDEX:
        raise ValueError("website_registration_anchor_frame_invalid")
    return float(mpu), np.linalg.inv(np.asarray(frame["world_from_camera"], dtype=np.float64))


def register_source_to_runtime(*, source_geometry: Mapping[str, Any], collision_mesh_path: Path,
                               sample_cap: int = 20000, anchor: Mapping[str, Any] | None = None,
                               focus_bounds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Similarity-register estimated source points to the reconstructed collider frame.

    With ``anchor`` (the provider's declared scale, ground plane and the source
    frame it was generated from), the prior pose is refined and its deviation
    reported; an unconstrained pose that clearly fits better is a conflict, not
    a silent override. Without it the unconstrained search must be unambiguous.
    """
    # Refuse a missing declared anchor before decoding every depth map and
    # searching collider poses. A surrogate camera cannot establish the
    # provider's first-view frame.
    anchor_pose = _anchor_prior(source_geometry, anchor) if anchor is not None else None
    source = _sample(_source_points(source_geometry), cap=sample_cap, seed=601)
    target = _sample(_mesh_vertices(collision_mesh_path), cap=sample_cap, seed=602)
    target_tree = cKDTree(target)
    target_extent = np.percentile(target, 99, axis=0) - np.percentile(target, 1, axis=0)
    # Model camera frames are not limited to axis permutations. Principal-axis
    # seeds cover arbitrary orientation; ICP refines candidate correspondences.
    source_axes = np.linalg.eigh(np.cov(source.T))[1]
    target_axes = np.linalg.eigh(np.cov(target.T))[1]
    source_axes[:, -1] *= np.linalg.det(source_axes)
    target_axes[:, -1] *= np.linalg.det(target_axes)
    rotations = _axis_rotations()
    rotations += [target_axes @ turn @ source_axes.T for turn in _axis_rotations()]
    rows = []
    fit_source = _sample(source, cap=4000, seed=603)

    def score(scale, rotation, translation):
        moved = scale * (source @ rotation.T) + translation
        forward = _trimmed_rmse(target_tree.query(moved, k=1)[0], 0.8)
        reverse = _trimmed_rmse(cKDTree(moved).query(target, k=1)[0], 0.8)
        return (math.sqrt((forward**2 + reverse**2) / 2.0), scale, rotation, translation)

    for rotation in rotations:
        moved = source @ rotation.T
        source_extent = np.percentile(moved, 99, axis=0) - np.percentile(moved, 1, axis=0)
        scale = float(np.median(target_extent / np.maximum(source_extent, 1e-9)))
        moved = moved * scale
        translation = np.median(target, axis=0) - np.median(moved, axis=0)
        rows.append(score(scale, rotation, translation))
    # An already exact frame conversion needs no iterative optimization.
    if min(row[0] for row in rows) > 1e-7 * np.linalg.norm(target_extent):
        rows += [score(*_refine_registration(fit_source, target, *row[1:])) for row in list(rows)]
    rows.sort(key=lambda row: row[0])
    extent_norm = float(np.linalg.norm(target_extent))

    def moved_by(row):
        return row[1] * (source @ row[2].T) + row[3]

    def distinct_from(reference, candidates):
        # Seeds converging to the same pose are one hypothesis, not ambiguity.
        base_moved = moved_by(reference)
        return [row for row in candidates
                if np.sqrt(np.mean(np.square(moved_by(row) - base_moved))) > 0.005 * extent_norm]

    anchor_report = ground = None
    if anchor is None:
        best = rows[0]
        distinct = distinct_from(best, rows[1:])
        if not distinct:
            raise ValueError("website_registration_ambiguous")
        ratio = distinct[0][0] / max(best[0], 1e-12)
        if ratio < 1.2:
            raise ValueError("website_registration_ambiguous")
        metres_per_runtime_unit = 1.0 / best[1]
    else:
        mpu, camera_from_world = anchor_pose
        up = _UP_INDEX[anchor["up_axis"]]
        offset = anchor.get("ground_plane_offset_m")
        floor = (None if offset is None else
                 _observed_declared_ground(source, np.linalg.inv(camera_from_world), float(offset)))
        if floor is None:
            # No observed floor at the declared height: the camera frame is the only prior.
            level, turn = np.eye(3), _roll
        else:
            # Remove the first camera's pitch and roll: the declared floor
            # normal becomes the world's down axis; heading stays ambiguous.
            down = np.zeros(3)
            down[up] = 1.0 if anchor["up_axis"] == "-Y" else -1.0
            level = _rotation_between(camera_from_world[:3, :3] @ floor, down)
            turn = lambda degrees: _about(up, degrees)  # noqa: E731
        anchored = []
        for roll in _ROLLS:
            rotation = turn(roll) @ level
            prior = (1.0 / mpu, rotation @ camera_from_world[:3, :3], (rotation @ camera_from_world[:3, 3]) / mpu)
            anchored.append((score(*_refine_registration(fit_source, target, *prior)), roll, prior))
        anchored.sort(key=lambda row: row[0][0])
        best, roll, prior = anchored[0]
        distinct = distinct_from(best, [row[0] for row in anchored[1:]])
        ratio = distinct[0][0] / max(best[0], 1e-12) if distinct else math.inf
        if ratio < 1.2:
            raise ValueError("website_registration_ambiguous")
        def same_hypothesis(row):
            return (_rotation_degrees(row[2], best[2]) <= 10.0 and abs(math.log(row[1] / best[1])) <= math.log(1.1)
                    and float(np.linalg.norm(row[3] - best[3])) <= 0.1 * extent_norm)
        better = [row for row in rows if row[0] < best[0]]
        # An unconstrained pose that fits materially and clearly better than
        # the anchored one means the world does not follow its declared anchor.
        # Only poses at a scale the anchor admits are evidence: a generated
        # world that extends past the observed footage rewards inflated-scale
        # poses, which the declared scale already refuses.
        if any(row[0] * 1.2 < best[0] and best[0] - row[0] > 0.005 * extent_norm
               and ANCHOR_SCALE_RATIO_BOUNDS[0] <= row[1] * mpu <= ANCHOR_SCALE_RATIO_BOUNDS[1]
               for row in better if not same_hypothesis(row)):
            raise ValueError("website_registration_conflicts_provider_anchor")
        # The same pose reached from an unconstrained seed is the same
        # hypothesis; keep whichever solution of it converged better.
        same_pose = [row for row in better if same_hypothesis(row)]
        if same_pose:
            best = min(same_pose, key=lambda row: row[0])
        anchor_report = {"kind": anchor.get("kind"), "frame_id": anchor["frame_id"], "roll_degrees": roll,
                         "hypothesis_axis": "camera_optical" if floor is None else "declared_up",
                         "levelled_tilt_degrees": _rotation_degrees(level, np.eye(3)),
                         "rotation_deviation_degrees": _rotation_degrees(best[2], prior[1]),
                         "translation_deviation_m": float(np.linalg.norm(best[3] - prior[2]) * mpu),
                         "scale_ratio_to_declared": float(best[1] * mpu),
                         "declared_meters_per_unit": mpu}
        if (anchor_report["rotation_deviation_degrees"] > ANCHOR_MAX_ROTATION_DEGREES
                or not ANCHOR_SCALE_RATIO_BOUNDS[0] <= anchor_report["scale_ratio_to_declared"] <= ANCHOR_SCALE_RATIO_BOUNDS[1]):
            raise ValueError("website_registration_anchor_deviation")
        metres_per_runtime_unit = mpu
        if offset is not None:
            down = (1 if anchor["up_axis"] == "-Y" else -1) * moved_by(best)[:, up]
            floor_raw = float(np.percentile(down, 98))
            fraction = float(np.mean(np.abs(down - floor_raw) <= 0.10 / mpu))
            ground = {"declared_offset_m": float(offset), "observed_floor_offset_m": floor_raw * mpu,
                      "residual_m": abs(floor_raw * mpu - float(offset)), "floor_point_fraction": fraction,
                      "checked": fraction >= 0.03}
            if ground["checked"] and ground["residual_m"] > GROUND_PLANE_MAX_RESIDUAL_M:
                raise ValueError("website_registration_ground_plane_inconsistent")
    if best[0] > 0.1 * extent_norm:
        raise ValueError("website_registration_poor_fit")
    task_region = None
    if focus_bounds is not None:
        low, high = (np.asarray(focus_bounds[key], dtype=np.float64) for key in ("minimum", "maximum"))
        center, radius = (low + high) / 2.0, max(1.0, float(np.linalg.norm(high - low)))
        near = source[np.linalg.norm(source - center, axis=1) <= radius]
        task_region = {"radius_m": radius, "point_count": int(len(near)), "trimmed_rmse_m": None}
        if len(near) >= 50:
            moved = best[1] * (near @ best[2].T) + best[3]
            task_region["trimmed_rmse_m"] = _trimmed_rmse(target_tree.query(moved, k=1)[0], 0.8) * metres_per_runtime_unit
    matrix = np.eye(4)
    matrix[:3, :3] = best[1] * best[2]
    matrix[:3, 3] = best[3]
    return {"schema_version": "website_source_registration.v1", "source_to_runtime": matrix.tolist(),
            "scale": best[1], "rotation": best[2].tolist(), "translation": best[3].tolist(),
            "trimmed_rmse_runtime_units": best[0], "trimmed_rmse_m": best[0] * metres_per_runtime_unit,
            "runner_up_ratio": ratio, "anchor": anchor_report, "ground_plane": ground, "task_region": task_region,
            "source_geometry_digest": source_geometry["digest"],
            "collision_mesh_digest": _sha256_file(collision_mesh_path),
            "scale_status": "provider_declared_anchor" if anchor else "estimated_registration",
            "physical_scale_measured": False, "physical_registration_proven": False}


def _runtime_bounds(bounds: Mapping[str, Any], matrix: np.ndarray) -> tuple[list[float], list[float]]:
    low, high = np.asarray(bounds["minimum"], dtype=np.float64), np.asarray(bounds["maximum"], dtype=np.float64)
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    moved = corners @ matrix[:3, :3].T + matrix[:3, 3]
    return moved.min(axis=0).tolist(), moved.max(axis=0).tolist()


def screen_physics(dimensions_m: Sequence[float], *, gripper: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Estimate object properties; test grasp sensitivity only for an explicitly supplied gripper."""
    volume = float(np.prod(dimensions_m))
    density, fill = PHYSICS_PRIORS["density_kg_m3"], PHYSICS_PRIORS["envelope_fill"]
    mass = [max(0.005, volume * density[0] * fill[0]), max(0.01, volume * density[1] * fill[1])]
    bounds = {"mass_kg": mass, "static_friction": list(PHYSICS_PRIORS["static_friction"]),
              "dynamic_friction": list(PHYSICS_PRIORS["dynamic_friction"]),
              "restitution": list(PHYSICS_PRIORS["restitution"])}
    if gripper is None:
        return {"basis": "estimated", "dimensions_m": [float(v) for v in dimensions_m],
                "envelope_volume_m3": volume, "bounds": bounds, "priors": PHYSICS_PRIORS,
                "sensitivity": "awaiting_robot_team_selection", "measurement_escalation": None}
    gravity, safety = 9.81, 2.0
    best = mass[0] * gravity * safety / (2 * bounds["static_friction"][1])
    worst = mass[1] * gravity * safety / (2 * bounds["static_friction"][0])
    available = gripper["grip_force_n"]["upper"]
    sorted_dims = sorted(dimensions_m)
    grasp_width_upper = sorted_dims[0] * (1 + DIMENSION_RELATIVE_ERROR)
    escalation = None
    if grasp_width_upper > gripper["full_stroke_m"]:
        sensitivity = "blocked_by_estimate"
        escalation = {"property": "smallest_dimension_m", "instrument": "tape_measure",
                      "reason": "The estimated object width may exceed the reference gripper stroke."}
    elif worst <= available:
        sensitivity = "robust_within_range"
    elif best <= available:
        sensitivity = "outcome_depends_on_estimate"
        escalation = {"property": "mass_kg", "instrument": "kitchen_scale",
                      "reason": "Grip hold succeeds at the light end of the mass range and fails at the heavy end."}
    else:
        sensitivity = "blocked_by_estimate"
        escalation = {"property": "mass_kg", "instrument": "kitchen_scale",
                      "reason": "Even the lightest estimate exceeds the reference grip force."}
    return {"basis": "estimated", "dimensions_m": [float(v) for v in dimensions_m], "envelope_volume_m3": volume,
            "bounds": bounds, "priors": PHYSICS_PRIORS, "gripper": dict(gripper),
            "hold_force_required_n": {"best_case": best, "worst_case": worst, "safety_factor": safety},
            "sensitivity": sensitivity, "measurement_escalation": escalation}


def screen_articulated_physics(dimensions_m: Sequence[float], *, joint_type: str,
                               gripper: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Estimate assembly properties; the pull outcome depends on unobserved slide friction."""
    dims = [float(value) for value in dimensions_m]
    volume = max(dims[0] * dims[1] * dims[2], 1e-6)
    bounds = {
        "mass_kg": [round(max(4.0, 60.0 * volume), 3), round(max(12.0, 250.0 * volume), 3)],
        "task_part_mass_kg": [0.5, 6.0],
        "static_friction": [0.3, 0.8], "dynamic_friction": [0.2, 0.6], "restitution": [0.0, 0.2],
        # Passive slide/hinge resistance the robot must overcome; unobserved.
        "joint_friction": [1.0, 15.0], "joint_damping": [1.0, 30.0],
        "handle_thickness_m": [0.008, 0.02], "handle_length_fraction_of_front_width": [0.5, 0.8],
    }
    value = {"basis": "estimated", "dimensions_m": dims, "joint_type": joint_type, "bounds": bounds,
             "dimension_relative_error": DIMENSION_RELATIVE_ERROR, "physical_measurement_proven": False,
             "measurement_escalation": {"property": "joint_friction",
                                        "reason": "drawer_slide_resistance_not_observable_in_footage"}}
    if gripper is None:
        value["sensitivity"] = "awaiting_robot_team_selection"
        return value
    value["gripper"] = dict(gripper)
    if bounds["handle_thickness_m"][1] * (1 + DIMENSION_RELATIVE_ERROR) > gripper["full_stroke_m"]:
        value["sensitivity"] = "blocked_by_estimate"
        value["measurement_escalation"] = {"property": "handle_thickness_m", "reason": "handle_may_exceed_gripper_stroke"}
    else:
        value["sensitivity"] = "outcome_depends_on_estimate"
    return value


def estimate_front_normal(track: Mapping[str, Any], frames_by_id: Mapping[str, Mapping[str, Any]],
                          source_to_runtime: np.ndarray, center_runtime: Sequence[float], *, up: int,
                          runtime_to_sim: np.ndarray) -> dict[str, Any]:
    """Estimate which way the assembly faces: toward the camera of its fullest observation.

    A drawer front is only observed from the side the operator filmed. The
    horizontal direction from the assembly centre to that camera is the front
    normal estimate; it is an estimate from registered camera poses, never a
    measured orientation.
    """
    best = max(track["observations"], key=lambda row: sum(run["length"] for run in row["runs"]))
    frame = frames_by_id[best["source_frame_id"]]
    camera_source = np.asarray(frame["world_from_camera"], dtype=float)[:3, 3]
    camera_runtime = (np.asarray(source_to_runtime, dtype=float) @ [*camera_source, 1.0])[:3]
    direction = camera_runtime - np.asarray(center_runtime, dtype=float)
    direction[up] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("website_front_normal_undetermined")
    direction = direction / norm
    rotation = np.asarray(runtime_to_sim, dtype=float)[:3, :3]
    sim = rotation @ direction
    sim = sim / max(float(np.linalg.norm(sim)), 1e-12)
    return {"estimated_front_normal_world": [float(v) for v in sim], "basis": "registered_camera_pose_of_fullest_observation",
            "source_frame_id": frame["frame_id"], "timestamp_seconds": frame.get("timestamp_seconds"),
            "physical_orientation_measured": False}


def levelled_body_bounds(body: Mapping[str, Any], source_to_runtime: np.ndarray, *, up: int) -> tuple[list, list]:
    """Runtime bounds of the whole body standing upright, yawed to its levelled front normal.

    The body box's own up axis comes from camera orientation and is an
    estimate; the registered runtime up axis is the gravity reference. The
    box keeps its estimated depth, width and height and its corner centre.
    """
    linear = np.asarray(source_to_runtime, dtype=float)[:3, :3]
    scale = abs(float(np.linalg.det(linear))) ** (1 / 3)
    corners = np.asarray(body["corners"], dtype=float) @ linear.T + np.asarray(source_to_runtime)[:3, 3]
    normal = linear @ np.asarray(body["front_normal"], dtype=float)
    normal[up] = 0.0
    if float(np.linalg.norm(normal)) < 1e-9:
        raise ValueError("website_front_normal_undetermined")
    normal = np.abs(normal / np.linalg.norm(normal))
    a, b = [axis for axis in range(3) if axis != up]
    depth, width, height = (float(body[key]) * scale for key in ("depth_m", "width_m", "height_m"))
    half = np.zeros(3)
    half[a] = (depth * normal[a] + width * normal[b]) / 2
    half[b] = (depth * normal[b] + width * normal[a]) / 2
    half[up] = height / 2
    center = corners.mean(axis=0)
    return (center - half).tolist(), (center + half).tolist()


def body_front_normal(body: Mapping[str, Any], source_to_sim_linear: np.ndarray) -> dict[str, Any]:
    """The closed front plane's outward normal, levelled, in the simulator frame."""
    sim = np.asarray(source_to_sim_linear, dtype=float) @ np.asarray(body["front_normal"], dtype=float)
    sim[2] = 0.0
    norm = float(np.linalg.norm(sim))
    if norm < 1e-9:
        raise ValueError("website_front_normal_undetermined")
    return {"estimated_front_normal_world": [float(v) for v in sim / norm],
            "basis": "closed_front_plane_of_whole_object_coverage", "physical_orientation_measured": False}


def _thumbnail(track: Mapping[str, Any], frames_by_id: Mapping[str, Mapping[str, Any]], output: Path) -> dict[str, Any]:
    best = max(track["observations"], key=lambda item: sum(run["length"] for run in item["runs"]))
    frame = frames_by_id[best["source_frame_id"]]
    if _sha256_file(Path(frame["image_path"])) != frame["image_digest"]:
        raise ValueError("website_source_image_changed")
    mask = decode_track_mask(best)
    ys, xs = np.nonzero(mask)
    with Image.open(frame["image_path"]) as image:
        image = image.convert("RGB")
        if mask.shape != (image.height, image.width):
            raise ValueError("website_thumbnail_mask_mismatch")
        pad_x, pad_y = int(0.25 * (xs.max() - xs.min() + 1)), int(0.25 * (ys.max() - ys.min() + 1))
        box = (max(0, xs.min() - pad_x), max(0, ys.min() - pad_y),
               min(image.width, xs.max() + 1 + pad_x), min(image.height, ys.max() + 1 + pad_y))
        crop = image.crop(box)
        crop.thumbnail((480, 300))
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
    output.write_bytes(buffer.getvalue())
    return {"path": str(output), "digest": _sha256_file(output), "width": crop.width, "height": crop.height,
            "png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "source_frame_id": best["source_frame_id"], "source_image_digest": frame["image_digest"],
            "crop_box_xyxy": [int(v) for v in box], "consent": "operator_listing_approval_required"}


def compile_website_scene_preparation(*, task_context: Mapping[str, Any], task_masks: Mapping[str, Any],
                                      removal_manifest: Mapping[str, Any], source_geometry: Mapping[str, Any],
                                      base_scene: Mapping[str, Any], output_root: Path, spend: Mapping[str, Any],
                                      now: float) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    if task_masks.get("deferred_target_ids"):
        raise ValueError("website_static_task_masks_pending")
    for value, field in ((task_context, "context_digest"), (task_masks, "digest"), (source_geometry, "digest")):
        if value.get(field) != canonical_digest(value, digest_field=field):
            raise ValueError("website_preparation_input_digest_mismatch")
    if task_context.get("confirmed") is not True:
        raise ValueError("website_preparation_task_not_confirmed")
    if task_masks.get("source_geometry_digest") != source_geometry["digest"]:
        raise ValueError("website_preparation_geometry_binding_mismatch")
    request_id, capture_id = str(task_context["request_id"]), str(task_context["capture_id"])
    if not _IDENTIFIER.fullmatch(capture_id) or not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", request_id):
        raise ValueError("website_identity_invalid")
    development_seed = base_scene.get("mode") == "development_fixture_seed"
    if development_seed:
        from .website_development_test import enabled
        if (not enabled(task_context["context_digest"])
                or base_scene.get("reconstruction_blocker") != "website_reconstruction_failed"
                or base_scene.get("provider") != "blueprint_authored_development_seed"):
            raise ValueError("website_development_fixture_seed_not_authorized")
    for path_key, digest_key in (("collision_mesh_path", "collision_mesh_digest"),) if development_seed else (
            ("splat_path", "splat_digest"), ("collision_mesh_path", "collision_mesh_digest")):
        if _sha256_file(Path(base_scene[path_key])) != base_scene[digest_key]:
            raise ValueError("website_base_scene_changed")
    up = _UP_INDEX[base_scene["up_axis"]]
    up_sign = -1 if base_scene["up_axis"] == "-Y" else 1
    targets = {row["target_id"]: row for row in task_masks["targets"]}
    removed = [row for row in removal_manifest["entries"] if row.get("task_effect") == "manipulated"]
    if len(removed) != 1 or removed[0]["target_id"] not in targets:
        raise ValueError("website_single_manipulated_subject_required")
    subject_entry, subject_target = removed[0], targets[removed[0]["target_id"]]
    declared_mpu = base_scene.get("meters_per_unit")
    if isinstance(declared_mpu, bool) or (declared_mpu is not None and (
            not isinstance(declared_mpu, (int, float)) or not math.isfinite(declared_mpu) or declared_mpu <= 0)):
        raise ValueError("website_base_scene_scale_invalid")
    anchor = base_scene.get("anchor")
    if anchor is not None:
        if declared_mpu is None:
            raise ValueError("website_base_scene_scale_invalid")
        anchor = {**anchor, "meters_per_unit": declared_mpu, "up_axis": base_scene["up_axis"],
                  "ground_plane_offset_m": base_scene.get("ground_plane_offset_m")}
    independent_object = False
    try:
        if development_seed:
            raise ValueError("website_reconstruction_failed")
        registration = register_source_to_runtime(source_geometry=source_geometry,
            collision_mesh_path=Path(base_scene["collision_mesh_path"]),
            anchor=anchor, focus_bounds=subject_target["estimated_visible_bounds"])
    except ValueError as exc:
        from .website_development_test import enabled
        from .website_object_local_frame import REGISTRATION_REFUSALS, estimated_object_frame
        if str(exc) not in REGISTRATION_REFUSALS or not enabled(task_context["context_digest"]):
            raise
        registration = estimated_object_frame(track=subject_target["track"],
            source_geometry=source_geometry, registration_blocker=str(exc))
        blockers.append(str(exc))
        independent_object = True
        up, up_sign, declared_mpu = 2, 1, 1.0
    # With browser video, use MapAnything's estimated metres to scale the
    # generated world. A provider's declared factor is an estimate too, and
    # the two must agree; neither is a physical measurement.
    mpu = 1.0 / registration["scale"] if declared_mpu is None else float(declared_mpu)
    if not independent_object and declared_mpu is not None and anchor is None and not 0.75 <= registration["scale"] * mpu <= 4.0 / 3.0:
        blockers.append("website_registration_scale_conflicts_declared")
    runtime_to_sim = np.eye(4)
    runtime_to_sim[:3, :3] = mpu * (np.array([[1, 0, 0], [0, 0, -up_sign], [0, up_sign, 0]])
                                   if up == 1 else np.eye(3))
    registration_path = output_root / "registration.json"
    write_json(registration_path, registration)
    matrix = np.asarray(registration["source_to_runtime"])
    frames_by_id = {frame["frame_id"]: frame for frame in source_geometry["frames"]}
    articulation_kind = str(subject_entry.get("articulation_kind") or subject_target.get("articulation_kind") or "")
    articulated = articulation_kind in {"prismatic", "revolute"}
    coverage = subject_target.get("authoring_coverage") if articulated else None
    coverage_bound = coverage is not None and coverage_matches(coverage, target=subject_target,
                                                               source_geometry=source_geometry)
    body = coverage["body_bounds"] if coverage_bound and coverage["status"] == "complete" else None
    if body is not None:
        subject_min, subject_max = levelled_body_bounds(body, matrix, up=up)
    else:
        subject_bounds = estimate_target_bounds(subject_target["track"], source_geometry["frames"],
                                                source_to_target=matrix)
        subject_min, subject_max = subject_bounds["minimum"], subject_bounds["maximum"]
    import trimesh
    collider = None if independent_object else trimesh.load(base_scene["collision_mesh_path"], force="mesh", process=False)
    support = (None if independent_object else
               support_under(collider, subject_min, subject_max, up=up, meters_per_unit=mpu, up_sign=up_sign))
    snap = 0.0
    if support is None:
        blockers.append("support_surface_not_found_under_subject")
    else:
        snap = support["top_runtime_units"] - (subject_min if up_sign == 1 else subject_max)[up]
        subject_min[up] += snap
        subject_max[up] += snap
    sim_min, sim_max = _runtime_bounds({"minimum": subject_min, "maximum": subject_max}, runtime_to_sim)
    dimensions_m = [sim_max[i] - sim_min[i] for i in range(3)]
    sim_support = (_runtime_bounds({"minimum": support["aabb_min"], "maximum": support["aabb_max"]}, runtime_to_sim)
                   if support else (None, None))
    destination_rows = [row for row in task_masks["targets"] if row.get("target_role") == "destination"]
    destination = None
    if articulated:
        # The moving part stays inside its assembly: no destination pose exists.
        # The whole assembly is removed, rebuilt with one task joint, and
        # placed back on the support beneath its footprint.
        articulated_part = str(subject_entry.get("articulated_part") or subject_target.get("articulated_part") or "")
        if not articulated_part:
            blockers.append("website_articulated_part_label_required")
        if destination_rows:
            blockers.append("website_articulated_task_has_no_destination")
        # A rebuilt assembly needs the whole object, not the surfaces that
        # happened to fall in the depth-sampling frames: those bounds miss the
        # body hidden in its cabinet and mix open and closed states, and a
        # thin front passes a check against them. Until the subject carries
        # views chosen to cover every part and state of the object, with the
        # body's depth observed, the build must not be bought.
        if coverage is not None and not coverage_bound:
            blockers.append("website_assembly_coverage_binding_mismatch")
        if body is None:
            blockers.append("website_assembly_whole_object_coverage_required")
            if coverage_bound:
                blockers.extend(coverage_blockers(coverage, target_id=subject_entry["target_id"], several=False))
    elif len(destination_rows) == 1 and not independent_object:
        low, high = _runtime_bounds(destination_rows[0]["estimated_visible_bounds"], matrix)
        position = [(low[i] + high[i]) / 2 for i in range(3)]
        position[up] = (high if up_sign == 1 else low)[up]
        relation = destination_rows[0].get("placement_relation")
        contact_bound = False
        if relation not in {"on", "inside"}:
            blockers.append("task_destination_relation_required")
        elif relation == "inside":
            blockers.append("task_destination_interior_geometry_required")
        elif support is not None:
            # Test the subject's footprint at the destination against real
            # triangles, not the destination mask's enclosing box.
            destination_min, destination_max = np.asarray(subject_min).copy(), np.asarray(subject_max).copy()
            for axis in range(3):
                if axis != up:
                    half_width = (subject_max[axis] - subject_min[axis]) / 2
                    destination_min[axis], destination_max[axis] = position[axis] - half_width, position[axis] + half_width
            height = subject_max[up] - subject_min[up]
            destination_min[up] = position[up] - (height if up_sign == -1 else 0)
            destination_max[up] = position[up] + (height if up_sign == 1 else 0)
            destination_support = support_under(collider, destination_min, destination_max, up=up,
                                                meters_per_unit=mpu, up_sign=up_sign)
            if destination_support is None:
                blockers.append("task_destination_surface_contact_required")
            elif destination_support["face_indices"] != support["face_indices"]:
                blockers.append("task_distinct_destination_surface_binding_required")
            else:
                position[up] = destination_support["top_runtime_units"]
                contact_bound = True
        destination = {"relation": relation, "visible_label": destination_rows[0].get("semantic_label") or destination_rows[0]["target_id"],
                       **({"mode": "existing_support_surface"} if relation == "on" else {}),
                       "position_world_m": (runtime_to_sim @ [*position, 1.0])[:3].tolist(),
                       "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                       "basis": "registered_estimated_visible_bounds_and_collider_contact" if contact_bound
                       else "registered_estimated_visible_bounds"}
    else:
        blockers.append("task_destination_pose_required")
    track = subject_target["track"]
    authoring_provider = spend.get("authoring_provider", "openai")
    if authoring_provider not in {"openai", "anthropic"}:
        raise ValueError("website_authoring_provider_invalid")
    contract = None
    if body is not None:
        from .authoring_frame_budget import FrameBudgetError, fit_reference_frames
        scale = abs(float(np.linalg.det(runtime_to_sim[:3, :3] @ matrix[:3, :3]))) ** (1 / 3)
        try:
            # Frames go to the builder as provider-sized derivatives of the
            # retained upright views; both digests stay on each row.
            contract = fit_reference_frames(
                assembly_contract(coverage, articulation_kind=articulation_kind, source_to_simulator_scale=scale),
                provider=authoring_provider, output_root=output_root / "reference_frames")
        except (ValueError, FrameBudgetError) as exc:
            blockers.append(str(exc))
    authoring_frames = [{"path": row["path"], "sha256": row["sha256"], "role": "observed_source",
                         "frame_id": row["frame_id"], "reason": row["reason"],
                         "source_sha256": row["transmission"]["source_sha256"]}
                        for row in (contract or {}).get("reference_frames") or []]
    for observation in [] if body is not None else track["observations"]:
        frame = frames_by_id[observation["source_frame_id"]]
        authoring_frames.append({"path": frame["image_path"], "sha256": frame["image_digest"],
                                 "role": "observed_source", "frame_id": frame["frame_id"]})
    from .task_evaluation_scene_configuration_submission_records import (
        articulated_stage_three_configuration, stage_three_configuration,
    )
    mechanism = None
    if articulated:
        physics = screen_articulated_physics(dimensions_m, joint_type=articulation_kind)
        if body is not None:
            front = body_front_normal(body, runtime_to_sim[:3, :3] @ matrix[:3, :3])
            normal = front["estimated_front_normal_world"]
            extent = [float(body[key]) * scale for key in ("depth_m", "width_m", "height_m")]
        else:
            front = estimate_front_normal(track, frames_by_id, matrix,
                                          [(subject_min[i] + subject_max[i]) / 2 for i in range(3)],
                                          up=up, runtime_to_sim=runtime_to_sim)
            normal = front["estimated_front_normal_world"]
            extent = [abs(normal[0]) * dimensions_m[0] + abs(normal[1]) * dimensions_m[1],
                      abs(normal[1]) * dimensions_m[0] + abs(normal[0]) * dimensions_m[1], dimensions_m[2]]
        depth_m = extent[0]
        from .website_articulated_mass import articulated_mass_bounds
        masses = articulated_mass_bounds(subject_target, joint_type=articulation_kind, body_extent_m=extent,
            fixed_part_ids=[row["part_id"] for row in (contract or {}).get("required_parts") or []
                            if row["role"] == "fixed_interior" and articulation_kind == "revolute"],
            object_spec=subject_target.get("object_spec"))
        physics["bounds"].update({key.removesuffix("_bounds"): value for key, value in masses.items()
                                  if key.endswith("_mass_kg_bounds") or key == "mass_kg_bounds"})
        physics["mass_authority"] = masses["mass_authority"]
        # A published size that contradicts the measured body, or unreconciled
        # research, holds the build; it is never silently resolved here.
        blockers.extend((subject_target.get("object_spec") or {}).get("blockers") or [])
        travel = ({"estimated_usable_stroke_m": round(DRAWER_USABLE_STROKE_FRACTION_OF_DEPTH * depth_m, 4)}
                  if articulation_kind == "prismatic" else {"estimated_usable_swing_rad": DOOR_USABLE_SWING_RAD})
        mechanism = {"assembly_label": subject_entry.get("semantic_label") or subject_entry["target_id"],
                     "part_label": articulated_part or "unspecified part", "joint_type": articulation_kind,
                     **travel, "travel_authority": ("object_prior_estimate_from_observed_body_depth" if body is not None
                                                    else "object_prior_estimate_from_estimated_visible_bounds"),
                     "estimated_front_normal_world": normal, "front_normal_basis": front["basis"],
                     "lock_status": "unknown",
                     "part_observed_open_in_footage": bool(coverage_bound and coverage["part_observed_open"]),
                     "observation_timestamps_seconds": sorted({float(frames_by_id[row["source_frame_id"]].get("timestamp_seconds") or 0.0)
                                                              for row in track["observations"]}),
                     "physical_measurement_proven": False}
        if body is not None:
            # The website admits a rebuilt hinged assembly only with this evidence.
            mechanism["whole_object_coverage"] = {
                "schema_version": coverage["schema_version"], "status": "complete", "digest": coverage["digest"],
                "reference_frame_count": len(coverage["selected_frames"])}
        authoring_configuration = articulated_stage_three_configuration(
            scene_id=task_context["scene_id"],
            replacement_identity={"id": "website-subject-" + task_context["context_digest"][7:27], "version": "v1"},
            source_instance_id=subject_entry["target_id"],
            authoring_target=subject_entry.get("semantic_label") or subject_entry["target_id"],
            source_min=sim_min, source_max=sim_max, dimension_tolerance=DIMENSION_RELATIVE_ERROR,
            physics_bounds={key + "_bounds": value for key, value in physics["bounds"].items()},
            mechanism=mechanism,
        )
        if "fixed_part_mass_kg_bounds" in masses:
            authoring_configuration["required_output"]["fixed_part_mass_kg_bounds"] = masses["fixed_part_mass_kg_bounds"]
        authoring_configuration.update(mass_authority=masses["mass_authority"],
                                       mass_bounds_provenance=masses["provenance"],
                                       mass_source_urls=masses["source_urls"])
        if contract is not None:
            authoring_configuration.update(contract)
    else:
        physics = screen_physics(dimensions_m)
        authoring_configuration = stage_three_configuration(
            scene_id=task_context["scene_id"],
            replacement_identity={"id": "website-subject-" + task_context["context_digest"][7:27], "version": "v1"},
            source_instance_id=subject_entry["target_id"],
            authoring_target=subject_entry.get("semantic_label") or subject_entry["target_id"],
            source_min=sim_min, source_max=sim_max, dimension_tolerance=DIMENSION_RELATIVE_ERROR,
            physics_bounds={key + "_bounds": value for key, value in physics["bounds"].items()},
        )
    authoring_configuration.update(
        source_object_identity=subject_entry["target_id"],
        source_observation_kind="website_capture_frames", dimension_authority="estimated",
        appearance_inputs=("digest_bound_upright_coverage_frames" if body is not None
                           else "digest_bound_original_capture_frames"),
        geometry_support=("whole_body_bounds_from_closed_front_and_open_interior_depth" if body is not None
                          else "unregistered_object_local_estimated_visible_bounds" if independent_object
                          else "registered_partial_visible_bounds_from_estimated_source_geometry"),
        construction_constraints={
            "confirmed_task": task_context["description"],
            "operator_answers": task_context.get("operator_answers") or {},
            "task_context_digest": task_context["context_digest"],
            "subject_target_id": subject_entry["target_id"], "destination": destination,
            **({"mechanism": mechanism} if mechanism else {}),
            **(assembly_constraints(coverage, task_part=articulated_part) if body is not None else {}),
            "rebuild_only_this_subject": True, "non_target_scene_objects_remain_in_background": True,
            "complete_object_dimensions_observed": body is not None,
            "unknown_surfaces": "Generated completion must remain an explicit assumption.",
        },
    )
    if contract is not None:
        # Hold what the builder would refuse before anything is bought.
        from .task_object_articulated_packaging import plan_articulated_assembly
        from .task_object_astra_authoring import AssetAuthoringError
        try:
            plan_articulated_assembly(authoring_configuration)
        except AssetAuthoringError as exc:
            blockers.append("website_assembly_builder_refused:" + str(exc))
    thumbnail = _thumbnail(track, frames_by_id, output_root / "thumbnail.png")
    # Capture consent permits scene preparation; it does not manufacture a
    # paid simulation authorization or accept provider terms on the owner's behalf.
    owner = dict(spend.get("owner") or {})
    consent = dict(spend.get("consent") or {})
    if authoring_provider == "anthropic":
        # This must come from the website's fresh owner-authorized execution
        # authority. A local key or worker environment cannot opt a scene in.
        terms = spend.get("anthropic_provider_terms_reference")
        if (not isinstance(terms, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", terms) is None
                or not isinstance(consent.get("provider_terms_reference"), str)
                or not consent["provider_terms_reference"]):
            blockers.append("website_anthropic_provider_terms_authority_required")
        authoring_configuration["authoring_model_provider"] = "anthropic"
    if any(spend.get(key) is not None for key in ("authoring_agent_runtime", "authoring_model", "agents_api_policy")):
        if (authoring_provider != "openai"
                or spend.get("authoring_agent_runtime") != "openai_agents_api"
                or spend.get("authoring_model") != "gpt-6-sol"
                or not isinstance(spend.get("agents_api_policy"), Mapping)
                or spend.get("authority_digest") != canonical_digest(spend, digest_field="authority_digest")
                or spend.get("scene_id") != task_context["scene_id"]
                or spend.get("capture_id") != task_context["capture_id"]
                or spend.get("task_context_digest") != task_context["context_digest"]):
            blockers.append("website_agents_api_signed_authority_required")
        else:
            authoring_configuration.update(authoring_model_provider="openai",
                authoring_agent_runtime="openai_agents_api", authoring_model="gpt-6-sol",
                agents_api_policy=dict(spend["agents_api_policy"]))
    if not owner or not consent:
        blockers.append("website_scene_execution_authority_required")
    rights = task_context.get("capture_rights") or {}
    if rights.get("derived_scene_generation_allowed") is not True:
        blockers.append("website_scene_processing_rights_required")
    request = {
        "schema_version": intake.REQUEST_SCHEMA, "submission_id": capture_id, "owner": owner,
        "source": ({"kind": "mesh", "binding_id": base_scene["collision_binding_id"],
                    "content_digest": base_scene["collision_mesh_digest"]} if development_seed else
                   {"kind": "gaussian_splat", "binding_id": base_scene["splat_binding_id"],
                    "content_digest": base_scene["splat_digest"]}),
        # The preparation below binds the reconstructed collider and estimated
        # registration. This is not an owner-uploaded companion mesh or an
        # owner declaration of its coordinate frame.
        "task": {"task_id": "website-" + task_context["context_digest"][7:27],
                 "strategy": "articulated_open_close" if articulated else "pick_and_place",
                 "subject": {"description": subject_entry.get("semantic_label") or subject_entry["target_id"],
                             "aabb_min_xyz": sim_min, "aabb_max_xyz": sim_max,
                             "coordinate_frame": "Z_up_estimated_meters",
                             "geometry_origin": "removed_before_reconstruction",
                             "complete_object_dimensions": body is not None},
                 "support": {"description": subject_entry.get("support_label") or "support surface under subject",
                             "aabb_min_xyz": sim_support[0], "aabb_max_xyz": sim_support[1]},
                 **({"articulation": mechanism, "success": dict(ARTICULATED_SUCCESS)} if articulated else
                    {"destination": destination or {"needs_input": "task_destination_pose_required"},
                     "success": dict(SUCCESS)})},
        "execution": {"purpose": "scene_preparation", "max_total_spend_usd": spend["max_total_spend_usd"],
                      "max_paid_attempts": spend["max_paid_attempts"], "max_retries": 0,
                      "expires_at_epoch": spend["expires_at_epoch"],
                      "allowed_providers": (["vast", "openai", "anthropic"]
                                            if authoring_provider == "anthropic" else ["vast", "openai"]),
                      "policy_candidates": [],
                      "claim_scope": CLAIM_CEILING},
        "consent": consent,
    }
    if not blockers:
        try:
            intake.validate_request(request, now=now)
        except intake.SceneIntakeError as exc:
            blockers.append(str(exc))
    compose_back = {"replacement_asset_id": "website-subject-" + task_context["context_digest"][7:27],
                    "pose_world": {"position": [(subject_min[i] + subject_max[i]) / 2 for i in range(3)],
                                   "orientation_xyzw": [0.0, 0.0, 0.0, 1.0], "unit": base_scene["up_axis"] + "-up runtime units",
                                   "meters_per_unit": mpu, "support_snap_runtime_units": snap},
                    "replacement_asset_frame_registration_uri": str(registration_path)}
    composed = dict(removal_manifest)
    composed["entries"] = [dict(row, compose_back=compose_back) if row["target_id"] == subject_entry["target_id"] else dict(row)
                           for row in removal_manifest["entries"]]
    write_json(output_root / "removal_manifest.composed.json", composed)
    value = {
        "schema_version": SCHEMA_VERSION, "status": "needs_input" if blockers else "intake_ready",
        "blockers": blockers, "claim_ceiling": CLAIM_CEILING,
        **({"authoring_provider_terms_reference": terms} if authoring_provider == "anthropic" and
            isinstance(terms, str) else {}),
        "binding": {"task_context_digest": task_context["context_digest"], "task_masks_digest": task_masks["digest"],
                    "source_geometry_digest": source_geometry["digest"],
                    "removal_manifest_digest": canonical_digest(removal_manifest),
                    **({"development_seed_mesh_digest": base_scene["collision_mesh_digest"]} if development_seed else
                       {"splat_digest": base_scene["splat_digest"]}),
                    "collision_mesh_digest": base_scene["collision_mesh_digest"],
                    "provider": base_scene.get("provider"), "operation_id": base_scene.get("operation_id")},
        "registration": registration, "coordinate_frame": {"declared_meters_per_unit": mpu,
                                                           "scale_authority": ("model_estimated_object_frame" if independent_object
                                                                               else base_scene.get("scale_authority") or "registration_estimate"),
                                                           "placement_uncertainty_m": (registration.get("task_region") or {}).get("trimmed_rmse_m"),
                                                           "declared_up_axis": "Z" if independent_object else base_scene["up_axis"],
                                                           "physical_scale_measured": False,
                                                           "task_coordinates": "Z_up_estimated_meters",
                                                           "runtime_to_simulator": runtime_to_sim.tolist()},
        "subject": request["task"]["subject"], "support": support, "destination": destination,
        "authoring_inputs": {"adapter": "astra_articulated_replacement" if articulated else "content_agents_rigid_replacement",
                             "source_frames": authoring_frames,
                             "configuration": authoring_configuration,
                             "metric_envelope": {"minimum_xyz_m": sim_min,
                                                 "maximum_xyz_m": sim_max,
                                                 "maximum_dimension_relative_error": DIMENSION_RELATIVE_ERROR},
                             "dimension_authority": "estimated"},
        "physics": physics, "thumbnail": {key: value for key, value in thumbnail.items() if key != "png_base64"},
        "intake_request": request, "compose_back": compose_back,
        "recipe_plan": {"observed_appearance_object_removal": "satisfied_by_website_clean_plate_before_reconstruction",
                        "collision_object_excision": "not_required_subject_absent_from_reconstruction",
                        **({"articulated_replacement_authoring": "astra_articulated_replacement",
                            "replacement_static_qualification": "simready_static_articulated_qualification"}
                           if articulated else
                           {"rigid_replacement_authoring": "content_agents_rigid_replacement",
                            "replacement_static_qualification": "simready_static_rigid_qualification"}),
                        "replacement_native_import_qualification": "simready_native_import_qualification",
                        "scene_assembly": "native_task_scene_assembly"},
        "provider_mutation_performed": False,
    }
    value["digest"] = canonical_digest(value, digest_field="digest")
    write_json(output_root / "preparation.json", value)
    (output_root / "thumbnail.b64").write_text(thumbnail["png_base64"])
    return value
