"""Atlas scene completion using the existing planner, task and review adapters.

Generated RGBD is consumed by reconstruction, never promoted to capture truth.
The source image edits and the earlier incomplete reconstruction remain reusable.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file

COMPLETION_REVISION = "v4"


def camera_candidates(frames):
    """A scene-relative candidate pool; the planner chooses count and viewpoints.

    Translation follows actual scene depth, never an assumed metre scale. Wide
    target cameras preserve the input vertical field of view on Atlas's grid.
    These are generation hypotheses, not collision-cleared physical cameras.
    """
    candidates = []
    for frame in frames:
        depth = frame["depth"]
        valid = np.isfinite(depth) & (depth > 0) & (frame["confidence"] > 0)
        if not np.any(valid):
            raise ValueError("atlas_completion_no_context_depth")
        distance = float(np.median(depth[valid]))
        camera = {"intrinsics": deepcopy(frame["intrinsics"]), "extrinsics": deepcopy(frame["extrinsics"])}
        k = camera["intrinsics"]
        scale = 720 / k["height"]
        k.update(width=1280, height=720, fx=k["fx"] * scale, fy=k["fy"] * scale, cx=640, cy=360)
        for name, delta in (("wide", (0, 0, 0)), ("left", (-.08, 0, 0)),
                            ("right", (.08, 0, 0)), ("lower", (0, -.08, 0)), ("higher", (0, .08, 0))):
            target = deepcopy(camera)
            position = frame["position"] + frame["rotation"] @ (np.array(delta) * distance)
            target["extrinsics"]["position"] = position.tolist()
            candidates.append({"camera_id": frame["frame_id"] + "-" + name,
                               "source_frame_id": frame["frame_id"], "purpose": name,
                               "context_valid_depth_fraction": float(np.mean(valid)),
                               "translation_fraction_of_visible_depth": list(delta), "camera": target})
    # Cover the environment around the observed free-space camera path, rather
    # than only extrapolating the downward source views. Count follows angular
    # coverage and overlap, not a fixed quota of source or generated images.
    from scipy.spatial.transform import Rotation
    vfov = math.radians(75)
    focal = 720 / (2 * math.tan(vfov / 2))
    hfov = 2 * math.atan(1280 / (2 * focal))
    overlap_stride = .65
    yaw_count = math.ceil(2 * math.pi / (hfov * overlap_stride))
    pitch_count = math.ceil((math.pi - vfov) / (vfov * overlap_stride)) + 1
    position = np.median([frame["position"] for frame in frames], axis=0).tolist()
    for yi in range(yaw_count):
        yaw = 2 * math.pi * yi / yaw_count
        for pi, pitch in enumerate(np.linspace(-math.pi / 2 + vfov / 2, math.pi / 2 - vfov / 2, pitch_count)):
            candidates.append({"camera_id": f"room-{yi}-{pi}", "source_frame_id": frames[0]["frame_id"],
                "purpose": "required_full_room_coverage", "inferred_surface_views": True,
                "camera": {"intrinsics": {"width": 1280, "height": 720, "fx": focal, "fy": focal, "cx": 640, "cy": 360},
                           "extrinsics": {"position": position,
                               "quaternion": Rotation.from_euler("YX", [yaw, float(pitch)]).as_quat().tolist(),
                               "coordinateSystem": "rub"}}})
    return candidates


def load_generated_views(root: Path, frame_ids):
    """Read exact generated RGB/depth/cameras, including optional confidence."""
    import cv2
    from .atlas_depth_splat import posed_frame, read_exr_depth

    result = json.loads((root / "result.json").read_text())
    if result.get("status") != "completed" or result.get("digest") != canonical_digest(result, digest_field="digest"):
        raise ValueError("atlas_completion_generated_result_invalid")
    outputs = {}
    for row in result["outputs"]:
        path = root / row["path"]
        if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or _sha256_file(path) != row["sha256"]):
            raise ValueError("atlas_completion_generated_output_changed")
        outputs[row["role"]] = path
    returned = result["operation"]["response"]["frames"]
    if len(returned) != len(frame_ids):
        raise ValueError("atlas_completion_generated_count_changed")
    frames, images = [], []
    for index, (frame_id, row) in enumerate(zip(frame_ids, returned, strict=True)):
        prefix = f"frames-{index}-"
        image = outputs[prefix + "imageAsset"]
        rgb = cv2.imread(str(image), cv2.IMREAD_COLOR)
        depth = read_exr_depth(outputs[prefix + "depth-depthAsset"])
        confidence_path = outputs.get(prefix + "depth-confidenceAsset")
        confidence = (cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED) if confidence_path else
                      (np.isfinite(depth) & (depth > 0)).astype(np.float32))
        if rgb is None or confidence is None:
            raise ValueError("atlas_completion_generated_output_undecodable")
        frames.append(posed_frame(frame_id=frame_id, rgb=rgb[..., ::-1], depth=depth,
                                  confidence=confidence, camera=row["camera"]))
        images.append({"frame_id": frame_id, "image_path": str(image), "image_digest": _sha256_file(image),
                       "novel_view": True})
    return result, frames, images


def complete_scene(*, pose_root, pose, frames, review, root, context, configuration,
                   task_runner, planner_runner, release_blockers, max_polls, checkpoint=None):
    """Plan -> generate RGBD -> independent review, all resumable under one cap."""
    from .website_image_completion import verify_completed_background

    root.mkdir(parents=True, exist_ok=True)
    from .atlas_context_fov import verify_pose_context_fov
    fov = verify_pose_context_fov(pose_root=pose_root, frames=frames, output_root=root)
    candidates = camera_candidates(frames)
    originals = {}
    survey_frames = []
    usage = pose["operation_receipt"]["input_usage"]["frames"]
    for index, row in enumerate(usage):
        asset = next(value for value in row["outputs"] if value["role"] == "image")
        path = pose_root / asset["path"]
        if (path.is_symlink() or not path.resolve().is_relative_to(pose_root.resolve())
                or _sha256_file(path) != asset["sha256"]):
            raise ValueError("atlas_completion_context_changed")
        originals[row["frame_id"]] = {"image_path": str(path), "image_digest": asset["sha256"]}
        survey_frames.append({"frame_id": row["frame_id"], "timestamp_seconds": index,
                              "sharpness": 0, "clipped_fraction": 0,
                              "source_image_path": str(path), "source_image_digest": asset["sha256"]})
    survey = {"source_video_digest": pose["binding"]["source_video_digest"], "heldout_pixels_included": False,
              "display_rotation_degrees": 0, "frames": survey_frames,
              "content_basis": "independently_reviewed_generated_background"}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    survey_path = root / "survey.json"
    if survey_path.exists() and json.loads(survey_path.read_text()) != survey:
        raise ValueError("atlas_completion_survey_changed")
    write_json(survey_path, survey)
    feedback_path = root / "plan-inputs.json"
    feedback = {"pose_result_digest": pose["digest"], "context_fov_audit_digest": fov["digest"],
                "candidate_camera_ids": candidates,
                "removal_instruction": review["automation"]["removal_instruction"],
                "uncaptured_regions": review.get("uncaptured_regions", []), "previews": [],
                "input_basis": "generated_background_not_raw_capture", "claim_ceiling": "development_only"}
    baseline_path = root.parent / "background_splat/atlas_depth_splat_manifest.json"
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text())
        if baseline.get("digest") != canonical_digest(baseline, digest_field="digest"):
            raise ValueError("atlas_completion_baseline_changed")
        if baseline.get("pose_result_digest") == pose["digest"]:
            feedback["baseline_geometry_gaps"] = {key: baseline[key] for key in
                ("digest", "frames", "cross_view_agreement", "gaussian_count")}
        else:
            # A full-view re-pose has a new coordinate frame. Preserve the old
            # cropped baseline, but do not present its holes as current geometry.
            feedback["earlier_baseline_digest"] = baseline["digest"]
            feedback["baseline_geometry_comparable"] = False
        registration_path = root.parent / "background_mesh/registered/atlas_mesh_registration.json"
        if "baseline_geometry_gaps" in feedback and registration_path.is_file():
            registration = json.loads(registration_path.read_text())
            if (registration.get("digest") != canonical_digest(registration, digest_field="digest")
                    or registration.get("splat_manifest_digest") != baseline["digest"]):
                raise ValueError("atlas_completion_baseline_mesh_changed")
            feedback["baseline_geometry_gaps"]["mesh_area_kept_fraction"] = registration["area_kept_fraction"]
    if feedback_path.exists():
        saved = json.loads(feedback_path.read_text())
        if {key: saved.get(key) for key in feedback} != feedback:
            raise ValueError("atlas_completion_plan_inputs_changed")
        feedback = saved
    else:
        feedback["remaining_budget_usd_at_plan"] = configuration["remaining_cost_usd"]
        write_json(feedback_path, feedback)
    if checkpoint:
        checkpoint("automation_completion_plan")
    if planner_runner is None:
        raise ValueError("atlas_completion_canonical_planner_required")
    planned = planner_runner(survey_path=survey_path, task_context=context, configuration=configuration,
                             output_root=root / "planner", completion=feedback)
    proposal = planned["proposal"]
    if proposal["status"] != "proposed":
        raise ValueError("atlas_completion_planner_abstained")
    by_id = {candidate["camera_id"]: candidate for candidate in candidates}
    ids = proposal["selected_camera_ids"]
    if len(ids) != len(set(ids)) or any(key not in by_id for key in ids):
        raise ValueError("atlas_completion_camera_selection_invalid")
    selected = [by_id[key] for key in ids]
    required = {row["camera_id"] for row in candidates if row["purpose"] == "required_full_room_coverage"}
    if not required <= set(ids):
        raise ValueError("atlas_completion_full_room_camera_coverage_missing")
    from .atlas_scene_preservation import generate_masked_completion
    _, blockers = release_blockers()
    if blockers:
        raise ValueError("atlas_completion_release_not_admitted:" + ",".join(blockers))
    completed, generation = generate_masked_completion(frames=frames, selected=selected, candidates=candidates,
        root=root, pose=pose, context=context, configuration=configuration, task_runner=task_runner,
        max_polls=max_polls, removed_objects=[obj["label"] for obj in review["automation"]["objects"]
            if obj["disposition"] == "remove"], checkpoint=checkpoint)
    if completed is None:
        return None, generation
    generated, images = completed
    for image in images:
        source = originals[image["source_frame_id"]]
        image.update(original_image_path=source["image_path"], original_image_digest=source["image_digest"])
    targets = [{"target_id": obj["target_id"], "semantic_label": obj["label"],
                "task_effect": "manipulated" if obj["disposition"] == "remove" else "support",
                "disposition": obj["disposition"]} for obj in review["automation"]["objects"]]
    plan = {"targets": targets, "task_context_sha256": hashlib.sha256(json.dumps(
        dict(context), sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    _, blockers = release_blockers()
    if blockers:
        raise ValueError("atlas_completion_review_release_not_admitted:" + ",".join(blockers))
    if checkpoint:
        checkpoint("automation_completion_review")
    visual = verify_completed_background(frames=images, original_frames=images, plan=plan,
        output_root=root / "review", task_context=context, source_video_digest=survey["source_video_digest"])
    report = {"status": "passed" if visual["status"] == "passed" else "needs_input",
              "planner": planned, "context_fov_audit_digest": fov["digest"],
              "generation_result_digest": generation["digest"], "generated_roots": generation["generated_roots"],
              "generation": generation, "preservation_paths": generation["preservation_paths"],
              "generated_frame_ids": [row["frame_id"] for row in generated], "review": visual,
              "selected_camera_ids": ids, "target_cameras": [row["camera"] for row in selected],
              "gap_assessment": proposal["gap_assessment"], "remaining_unknowns": proposal["remaining_unknowns"],
              "generated_geometry_ready": visual["status"] == "passed", "completion_extent": "full_room_directional_views",
              "claim_ceiling": "development_only",
              "simulation_qualified": False, "metric_measurement_proven": False, "physical_evidence": False}
    if visual["status"] != "passed":
        report["reason"] = "atlas_completion_visual_review_required"
    report["digest"] = canonical_digest(report, digest_field="digest")
    write_json(root / "completion.json", report)
    return (generated if visual["status"] == "passed" else None), report
