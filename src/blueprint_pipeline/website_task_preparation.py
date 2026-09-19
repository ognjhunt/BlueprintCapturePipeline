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
from .adp009d_policy_candidate_admission import EXPECTED_CANDIDATES
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .external_scene_frame_registration import _axis_rotations, _sample, _trimmed_rmse
from .local_reconstruction_adapters import _sha256_file
from .website_task_masks import decode_track_mask
from .website_support_geometry import support_under

SCHEMA_VERSION = "website_scene_preparation.v1"
CLAIM_CEILING = "development_only"
_UP_INDEX = {"Y": 1, "Z": 2}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
# Robotiq 2F-85 datasheet: 85 mm stroke, 20-235 N grip force. Reference only.
GRIPPER = {"model": "robotiq_2f85", "full_stroke_m": FULL_STROKE_M,
           "grip_force_n": {"lower": 20.0, "upper": 235.0}}
SUCCESS = {"control_frequency_hz": 15, "maximum_episode_seconds": 24.0, "minimum_lift_m": 0.05,
           "pregrasp_clearance_m": 0.1, "minimum_planar_displacement_m": 0.1,
           "maximum_final_planar_target_error_m": 0.05, "maximum_retries": 0, "maximum_regrasps": 0}
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


def register_source_to_runtime(*, source_geometry: Mapping[str, Any], collision_mesh_path: Path,
                               sample_cap: int = 20000) -> dict[str, Any]:
    """Similarity-register estimated source points to the reconstructed collider frame."""
    source = _sample(_source_points(source_geometry), cap=sample_cap, seed=601)
    target = _sample(_mesh_vertices(collision_mesh_path), cap=sample_cap, seed=602)
    target_tree = cKDTree(target)
    target_extent = np.percentile(target, 99, axis=0) - np.percentile(target, 1, axis=0)
    rows = []
    for rotation in _axis_rotations():
        moved = source @ rotation.T
        source_extent = np.percentile(moved, 99, axis=0) - np.percentile(moved, 1, axis=0)
        scale = float(np.median(target_extent / np.maximum(source_extent, 1e-9)))
        moved = moved * scale
        translation = np.median(target, axis=0) - np.median(moved, axis=0)
        moved = moved + translation
        forward = _trimmed_rmse(target_tree.query(moved, k=1)[0], 0.8)
        reverse = _trimmed_rmse(cKDTree(moved).query(target, k=1)[0], 0.8)
        rows.append((math.sqrt((forward**2 + reverse**2) / 2.0), scale, rotation, translation))
    rows.sort(key=lambda row: row[0])
    best, runner_up = rows[0], rows[1]
    ratio = runner_up[0] / max(best[0], 1e-12)
    if ratio < 1.2:
        raise ValueError("website_registration_ambiguous")
    if best[0] > 0.1 * float(np.linalg.norm(target_extent)):
        raise ValueError("website_registration_poor_fit")
    matrix = np.eye(4)
    matrix[:3, :3] = best[1] * best[2]
    matrix[:3, 3] = best[3]
    return {"schema_version": "website_source_registration.v1", "source_to_runtime": matrix.tolist(),
            "scale": best[1], "rotation": best[2].tolist(), "translation": best[3].tolist(),
            "trimmed_rmse_runtime_units": best[0], "runner_up_ratio": ratio,
            "source_geometry_digest": source_geometry["digest"],
            "collision_mesh_digest": _sha256_file(collision_mesh_path),
            "scale_status": "estimated_registration", "physical_scale_measured": False,
            "physical_registration_proven": False}


def _runtime_bounds(bounds: Mapping[str, Any], matrix: np.ndarray) -> tuple[list[float], list[float]]:
    low, high = np.asarray(bounds["minimum"], dtype=np.float64), np.asarray(bounds["maximum"], dtype=np.float64)
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    moved = corners @ matrix[:3, :3].T + matrix[:3, 3]
    return moved.min(axis=0).tolist(), moved.max(axis=0).tolist()


def screen_physics(dimensions_m: Sequence[float]) -> dict[str, Any]:
    """Estimated property ranges plus a grasp-hold sensitivity screen against the reference gripper."""
    volume = float(np.prod(dimensions_m))
    density, fill = PHYSICS_PRIORS["density_kg_m3"], PHYSICS_PRIORS["envelope_fill"]
    mass = [max(0.005, volume * density[0] * fill[0]), max(0.01, volume * density[1] * fill[1])]
    bounds = {"mass_kg": mass, "static_friction": list(PHYSICS_PRIORS["static_friction"]),
              "dynamic_friction": list(PHYSICS_PRIORS["dynamic_friction"]),
              "restitution": list(PHYSICS_PRIORS["restitution"])}
    gravity, safety = 9.81, 2.0
    best = mass[0] * gravity * safety / (2 * bounds["static_friction"][1])
    worst = mass[1] * gravity * safety / (2 * bounds["static_friction"][0])
    available = GRIPPER["grip_force_n"]["upper"]
    sorted_dims = sorted(dimensions_m)
    grasp_width_upper = sorted_dims[0] * (1 + DIMENSION_RELATIVE_ERROR)
    escalation = None
    if grasp_width_upper > GRIPPER["full_stroke_m"]:
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
            "bounds": bounds, "priors": PHYSICS_PRIORS, "gripper": GRIPPER,
            "hold_force_required_n": {"best_case": best, "worst_case": worst, "safety_factor": safety},
            "sensitivity": sensitivity, "measurement_escalation": escalation}


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
    for path_key, digest_key in (("splat_path", "splat_digest"), ("collision_mesh_path", "collision_mesh_digest")):
        if _sha256_file(Path(base_scene[path_key])) != base_scene[digest_key]:
            raise ValueError("website_base_scene_changed")
    up = _UP_INDEX[base_scene["up_axis"]]
    registration = register_source_to_runtime(source_geometry=source_geometry,
                                              collision_mesh_path=Path(base_scene["collision_mesh_path"]))
    # With browser video, use MapAnything's estimated metres to scale the
    # generated world. A provider's nominal unit is not a physical measurement.
    mpu = (1.0 / registration["scale"] if base_scene.get("meters_per_unit") is None
           else float(base_scene["meters_per_unit"]))
    if not math.isfinite(mpu) or mpu <= 0 or isinstance(base_scene.get("meters_per_unit"), bool):
        raise ValueError("website_base_scene_scale_invalid")
    runtime_to_sim = np.eye(4)
    runtime_to_sim[:3, :3] = mpu * (np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                                   if up == 1 else np.eye(3))
    registration_path = output_root / "registration.json"
    write_json(registration_path, registration)
    matrix = np.asarray(registration["source_to_runtime"])
    targets = {row["target_id"]: row for row in task_masks["targets"]}
    frames_by_id = {frame["frame_id"]: frame for frame in source_geometry["frames"]}
    removed = [row for row in removal_manifest["entries"] if row.get("task_effect") == "manipulated"]
    if len(removed) != 1 or removed[0]["target_id"] not in targets:
        raise ValueError("website_single_manipulated_subject_required")
    subject_entry, subject_target = removed[0], targets[removed[0]["target_id"]]
    subject_min, subject_max = _runtime_bounds(subject_target["estimated_visible_bounds"], matrix)
    import trimesh
    collider = trimesh.load(base_scene["collision_mesh_path"], force="mesh", process=False)
    support = support_under(collider, subject_min, subject_max, up=up, meters_per_unit=mpu)
    snap = 0.0
    if support is None:
        blockers.append("support_surface_not_found_under_subject")
    else:
        snap = support["top_runtime_units"] - subject_min[up]
        subject_min[up] += snap
        subject_max[up] += snap
    sim_min, sim_max = _runtime_bounds({"minimum": subject_min, "maximum": subject_max}, runtime_to_sim)
    dimensions_m = [sim_max[i] - sim_min[i] for i in range(3)]
    sim_support = (_runtime_bounds({"minimum": support["aabb_min"], "maximum": support["aabb_max"]}, runtime_to_sim)
                   if support else (None, None))
    destination_rows = [row for row in task_masks["targets"] if row.get("target_role") == "destination"]
    destination = None
    if len(destination_rows) == 1:
        low, high = _runtime_bounds(destination_rows[0]["estimated_visible_bounds"], matrix)
        position = [(low[i] + high[i]) / 2 for i in range(3)]
        position[up] = high[up]
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
            destination_min[up] = position[up]
            destination_max[up] = position[up] + subject_max[up] - subject_min[up]
            destination_support = support_under(collider, destination_min, destination_max, up=up, meters_per_unit=mpu)
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
    authoring_frames = []
    for observation in track["observations"]:
        frame = frames_by_id[observation["source_frame_id"]]
        authoring_frames.append({"path": frame["image_path"], "sha256": frame["image_digest"],
                                 "role": "observed_source", "frame_id": frame["frame_id"]})
    physics = screen_physics(dimensions_m)
    from .task_evaluation_scene_configuration_submission_records import stage_three_configuration
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
        appearance_inputs="digest_bound_original_capture_frames",
        geometry_support="registered_partial_visible_bounds_from_estimated_source_geometry",
        construction_constraints={
            "confirmed_task": task_context["description"],
            "operator_answers": task_context.get("operator_answers") or {},
            "task_context_digest": task_context["context_digest"],
            "subject_target_id": subject_entry["target_id"], "destination": destination,
            "rebuild_only_this_subject": True, "non_target_scene_objects_remain_in_background": True,
            "complete_object_dimensions_observed": False,
            "unknown_surfaces": "Generated completion must remain an explicit assumption.",
        },
    )
    thumbnail = _thumbnail(track, frames_by_id, output_root / "thumbnail.png")
    # Capture consent permits scene preparation; it does not manufacture a
    # paid simulation authorization or accept provider terms on the owner's behalf.
    owner = dict(spend.get("owner") or {})
    consent = dict(spend.get("consent") or {})
    if not owner or not consent:
        blockers.append("website_scene_execution_authority_required")
    rights = task_context.get("capture_rights") or {}
    if rights.get("derived_scene_generation_allowed") is not True:
        blockers.append("website_scene_processing_rights_required")
    request = {
        "schema_version": intake.REQUEST_SCHEMA, "submission_id": capture_id, "owner": owner,
        "source": {"kind": "gaussian_splat", "binding_id": base_scene["splat_binding_id"],
                   "content_digest": base_scene["splat_digest"],
                   "collision_mesh": {"binding_id": base_scene["collision_binding_id"],
                                      "content_digest": base_scene["collision_mesh_digest"],
                                      "rights_reference": task_context["context_digest"],
                                      "frame_relation": "owner_declared_common_frame"}},
        "task": {"task_id": "website-" + task_context["context_digest"][7:27], "strategy": "pick_and_place",
                 "subject": {"description": subject_entry.get("semantic_label") or subject_entry["target_id"],
                             "aabb_min_xyz": sim_min, "aabb_max_xyz": sim_max,
                             "coordinate_frame": "Z_up_estimated_meters",
                             "geometry_origin": "removed_before_reconstruction",
                             "complete_object_dimensions": False},
                 "support": {"description": subject_entry.get("support_label") or "support surface under subject",
                             "aabb_min_xyz": sim_support[0], "aabb_max_xyz": sim_support[1]},
                 "destination": destination or {"needs_input": "task_destination_pose_required"},
                 "success": dict(SUCCESS)},
        "execution": {"max_total_spend_usd": spend["max_total_spend_usd"],
                      "max_paid_attempts": spend["max_paid_attempts"], "max_retries": 0,
                      "expires_at_epoch": spend["expires_at_epoch"], "allowed_providers": ["vast", "openai"],
                      "policy_candidates": [{"id": name, "artifact_digest": EXPECTED_CANDIDATES[name]["checkpoint_inventory_digest"]}
                                            for name in intake.SUPPORTED_POLICY_CANDIDATE_IDS],
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
        "binding": {"task_context_digest": task_context["context_digest"], "task_masks_digest": task_masks["digest"],
                    "source_geometry_digest": source_geometry["digest"],
                    "removal_manifest_digest": canonical_digest(removal_manifest),
                    "splat_digest": base_scene["splat_digest"], "collision_mesh_digest": base_scene["collision_mesh_digest"],
                    "provider": base_scene.get("provider"), "operation_id": base_scene.get("operation_id")},
        "registration": registration, "coordinate_frame": {"declared_meters_per_unit": mpu,
                                                           "declared_up_axis": base_scene["up_axis"],
                                                           "physical_scale_measured": False,
                                                           "task_coordinates": "Z_up_estimated_meters",
                                                           "runtime_to_simulator": runtime_to_sim.tolist()},
        "subject": request["task"]["subject"], "support": support, "destination": destination,
        "authoring_inputs": {"adapter": "content_agents_rigid_replacement", "source_frames": authoring_frames,
                             "configuration": authoring_configuration,
                             "metric_envelope": {"minimum_xyz_m": sim_min,
                                                 "maximum_xyz_m": sim_max,
                                                 "maximum_dimension_relative_error": DIMENSION_RELATIVE_ERROR},
                             "dimension_authority": "estimated"},
        "physics": physics, "thumbnail": {key: value for key, value in thumbnail.items() if key != "png_base64"},
        "intake_request": request, "compose_back": compose_back,
        "recipe_plan": {"observed_appearance_object_removal": "satisfied_by_website_clean_plate_before_reconstruction",
                        "collision_object_excision": "not_required_subject_absent_from_reconstruction",
                        "rigid_replacement_authoring": "content_agents_rigid_replacement",
                        "replacement_static_qualification": "simready_static_rigid_qualification",
                        "replacement_native_import_qualification": "simready_native_import_qualification",
                        "scene_assembly": "native_task_scene_assembly"},
        "provider_mutation_performed": False,
    }
    value["digest"] = canonical_digest(value, digest_field="digest")
    write_json(output_root / "preparation.json", value)
    (output_root / "thumbnail.b64").write_text(thumbnail["png_base64"])
    return value
