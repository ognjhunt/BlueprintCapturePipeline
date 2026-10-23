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

COMPLETION_REVISION = "v14"


def camera_candidates(frames):
    """A scene-relative candidate pool; the planner chooses count and viewpoints.

    Original-centre rotations follow the source field of view and need no
    estimated-depth RGB warp. Additional, modest translated hypotheses reveal
    disocclusions between neighbouring captured positions; they are not claimed
    to be observed or collision-cleared camera locations.
    A calibrated pixel/camera quarter-turn fits portrait photos to the native
    landscape grid without broadening the photographed horizontal field of view.
    These are generation hypotheses, not collision-cleared physical cameras.
    """
    from scipy.spatial.transform import Rotation
    from .atlas_scene_chisel import oriented_photograph

    candidates = []
    for frame in frames:
        depth = frame["depth"]
        valid = np.isfinite(depth) & (depth > 0) & (frame["confidence"] > 0)
        if not np.any(valid):
            raise ValueError("atlas_completion_no_context_depth")
        _, camera, rotated = oriented_photograph(frame)
        k = camera["intrinsics"]
        scale = min(1280 / k['width'], 720 / k['height'])
        k.update(fx=k['fx'] * scale, fy=k['fy'] * scale,
                 cx=(k['cx'] - k['width'] / 2) * scale + 640,
                 cy=(k['cy'] - k['height'] / 2) * scale + 360, width=1280, height=720)
        source_k = frame['intrinsics']
        yaw_step = .15 * 2 * math.atan(source_k['width'] / (2 * source_k['fx']))
        pitch_step = .15 * 2 * math.atan(source_k['height'] / (2 * source_k['fy']))
        roll = Rotation.from_euler('Z', 90, degrees=True).as_matrix() if rotated else np.eye(3)
        for name, angles in (("wide", (0, 0)), ("left", (yaw_step, 0)),
                             ("right", (-yaw_step, 0)), ("lower", (0, -pitch_step)),
                             ("higher", (0, pitch_step))):
            target = deepcopy(camera)
            if name != 'wide':
                target['extrinsics']['quaternion'] = Rotation.from_matrix(
                    frame['rotation'] @ Rotation.from_euler('YX', angles).as_matrix() @ roll).as_quat().tolist()
            candidates.append({"camera_id": frame["frame_id"] + "-" + name,
                               "source_frame_id": frame["frame_id"], "purpose": name,
                               "context_valid_depth_fraction": float(np.mean(valid)),
                               "rotation_radians": list(angles), "camera": target})
    wide = [row for row in candidates if row['purpose'] == 'wide']
    for first, second in zip(wide, wide[1:]):
        start = np.asarray(first['camera']['extrinsics']['position'])
        end = np.asarray(second['camera']['extrinsics']['position'])
        if np.linalg.norm(end - start) < 1e-6:
            continue
        target = deepcopy(first)
        target.update(camera_id=first['camera_id'] + '-translated', purpose='translated',
                      translation_basis='quarter_of_adjacent_recorded_camera_baseline')
        target['camera']['extrinsics']['position'] = (start + .25 * (end - start)).tolist()
        candidates.append(target)
    from .atlas_completion_cameras import disocclusion_candidates
    candidates.extend(disocclusion_candidates(frames, wide))
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
                   task_runner, planner_runner, release_blockers, max_polls, checkpoint=None,
                   existing_geometry_frames=(), expected_frames=(), splat_root=None,
                   reference_frames=None, coverage_feedback=None):
    """Plan -> generate RGBD -> independent review, all resumable under one cap."""
    from .website_image_completion import verify_completed_background

    root.mkdir(parents=True, exist_ok=True)
    from .atlas_context_fov import verify_pose_context_fov
    fov = verify_pose_context_fov(pose_root=pose_root, frames=frames, output_root=root)
    camera_pool = camera_candidates(frames)
    # The initial pass already surveyed the full topology. A feedback pass
    # checks every source/reviewed view, then aims at its remaining holes;
    # re-surveying every old translated hypothesis would repeat finished work.
    candidates = ([row for row in camera_pool if row['purpose'] == 'wide']
                  if splat_root is not None else list(camera_pool))
    if expected_frames:
        for expected in expected_frames:
            target = next(row for row in camera_candidates([expected]) if row['purpose'] == 'wide')
            target['source_frame_id'] = min(frames, key=lambda f: np.linalg.norm(
                f['position'] - expected['position']))['frame_id']
            target['purpose'] = 'wide'
            candidates.append(target)
            camera_pool.append(target)
    from .atlas_scene_preservation import observed_scene_bounds
    extent = observed_scene_bounds(frames)
    from .atlas_scene_gaps import survey_gaps, select_gap_mask, target_gap_ids, context_supported_candidates
    geometry_frames = [*frames, *existing_geometry_frames]
    survey_options = ({'expected_frames': expected_frames, 'splat_root': splat_root}
                      if expected_frames or splat_root is not None else {})
    context_frames = frames if reference_frames is None else reference_frames
    gaps = survey_gaps(frames=context_frames, geometry_frames=geometry_frames, candidates=candidates,
                       root=root / 'gaps', **survey_options)
    from .atlas_completion_cameras import gap_target_candidates
    aimed = gap_target_candidates(frames, camera_pool, gaps, **(
        {'expected_frames': expected_frames} if expected_frames else {}))
    if aimed:
        targeted = survey_gaps(frames=context_frames, geometry_frames=geometry_frames,
                              candidates=aimed, root=root / 'gap-targets', **survey_options)
        _, rejected_aims = context_supported_candidates(aimed, targeted)
        rejected_ids = set(rejected_aims)
        missing = {key for row in aimed if row['camera_id'] in rejected_ids for key in row['source_gap_ids']}
        # Keep the same named holes. Try a wider, source-centred view when
        # looking from farther away exposes the back of a known surface.
        if missing:
            retry_survey = {**gaps, 'views': [{**view, 'gaps': [gap for gap in view['gaps']
                                                             if gap['gap_id'] in missing]}
                                            for view in gaps['views']]}
            fallback = gap_target_candidates(frames, camera_pool, retry_survey, source_centred=True,
                **({'expected_frames': expected_frames} if expected_frames else {}))
            if fallback:
                fallback_gaps = survey_gaps(frames=context_frames, geometry_frames=geometry_frames,
                    candidates=fallback, root=root / 'gap-target-context-fallback', **survey_options)
                aimed.extend(fallback)
                targeted = {**targeted, 'views': [*targeted['views'], *fallback_gaps['views']],
                    'previews': [*targeted['previews'], *fallback_gaps['previews']],
                    'binding_digest': canonical_digest({'initial': targeted['binding_digest'],
                                                       'fallback': fallback_gaps['binding_digest']})}
        candidates.extend(aimed)
        gaps = {**gaps, 'views': [*gaps['views'], *targeted['views']],
                'previews': [*gaps['previews'], *targeted['previews']],
                'targeted_preview_count': len(targeted['previews']),
                'binding_digest': canonical_digest({'base': gaps['binding_digest'],
                                                    'targeted': targeted['binding_digest']})}
        gaps['digest'] = canonical_digest(gaps, digest_field='digest')
        write_json(root / 'gap-target-survey.json', gaps)
    if not any(row['gaps'] for row in gaps['views']):
        report = {'status': 'passed', 'generation_result_digest': gaps['digest'], 'generated_roots': [],
                  'generated_frame_ids': [], 'preservation_paths': [str(root / 'gaps/survey.json')],
                  'review': {'status': 'not_required', 'reason': 'no_eligible_geometry_gaps'},
                  'gap_survey_digest': gaps['digest'], 'completion_extent': 'identified_missing_geometry_only',
                  'generation_extent': extent, 'generated_geometry_ready': True, 'claim_ceiling': 'development_only'}
        report.update(whole_scene_complete=False, coverage_basis='surveyed_camera_gaps_only')
        report['digest'] = canonical_digest(report, digest_field='digest')
        write_json(root / 'completion.json', report)
        return [], report
    candidates, unsupported = context_supported_candidates(candidates, gaps)
    usable = {row['camera_id'] for row in candidates}
    with_gaps = {row['camera_id'] for row in gaps['views'] if row['gap_pixel_count']}
    # Show the actual proposed camera views, including dedicated gap views.
    # The source-camera overlay cannot reveal a bad new viewpoint.
    previews = [row for row in gaps['previews'] if row['camera_id'] in usable & with_gaps]
    if not usable & with_gaps:
        raise ValueError('atlas_completion_no_context_supported_gap_camera')
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
                "generation_extent": extent, "scope": "missing_surfaces_within_captured_scene",
                "removal_instruction": review["automation"]["removal_instruction"],
                "uncaptured_regions": review.get("uncaptured_regions", []), "previews": previews,
                'unsupported_camera_ids': unsupported,
                'gap_survey': {k: v for k, v in gaps.items() if k != 'previews'},
                "input_basis": "generated_background_not_raw_capture", "claim_ceiling": "development_only"}
    if coverage_feedback is not None:
        feedback['post_export_coverage'] = coverage_feedback
    if expected_frames:
        from .atlas_scene_chisel import prepare_source_photographs
        photos = prepare_source_photographs(frames=expected_frames, root=root / 'reviewed-references')
        feedback['previews'].extend({'path': p['image_path'], 'sha256': p['image_digest'],
            'source_frame_id': p['source_frame_id'], 'camera_id': p['source_frame_id'],
            'kind': 'reviewed_rgbd_reference_depth_is_a_hypothesis'} for p in photos)
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
    if proposal['status'] == 'sufficient' and not proposal['selected_camera_ids']:
        report = {'status': 'passed', 'generation_result_digest': gaps['digest'],
                  'generated_roots': [], 'generated_frame_ids': [], 'planner': planned,
                  'preservation_paths': [str(root / 'gaps/survey.json'), str(survey_path), str(feedback_path)],
                  'review': {'status': 'not_required', 'reason': 'agent_found_no_material_task_gap'},
                  'gap_survey_digest': gaps['digest'], 'remaining_unknowns': proposal['remaining_unknowns'],
                  'gap_assessment': proposal['gap_assessment'], 'whole_scene_complete': False,
                  'coverage_basis': 'task_prioritized_review_of_surveyed_export',
                  'generated_geometry_ready': True, 'claim_ceiling': 'development_only'}
        report['digest'] = canonical_digest(report, digest_field='digest')
        write_json(root / 'completion.json', report)
        return [], report
    if proposal["status"] != "proposed":
        raise ValueError("atlas_completion_planner_abstained")
    by_id = {candidate["camera_id"]: candidate for candidate in candidates}
    ids = proposal["selected_camera_ids"]
    if not ids or len(ids) != len(set(ids)) or any(key not in by_id for key in ids):
        raise ValueError("atlas_completion_camera_selection_invalid")
    selected = [by_id[key] for key in ids]
    entries = proposal.get('view_gap_reasons', [])
    reasons = {row['camera_id']: row['reason'] for row in entries}
    if (not proposal.get('scene_caption', '').strip() or len(reasons) != len(entries)
            or set(reasons) != set(ids) or any(not value.strip() for value in reasons.values())):
        raise ValueError('atlas_completion_task_gap_plan_missing')
    gap_views = {row['camera_id']: row for row in gaps['views']}
    entries_by_id = {row['camera_id']: row for row in entries}
    for index, view in enumerate(selected):
        target_view = gap_views[view['camera_id']]
        requested = entries_by_id[view['camera_id']].get('gap_ids', [])
        view.update(select_gap_mask(target_view, target_gap_ids(target_view, view, requested),
                                   root / 'gaps' / f'selected-{index:04d}.png'))
    from .atlas_scene_generate import generate_rgbd_completion
    _, blockers = release_blockers()
    if blockers:
        raise ValueError("atlas_completion_release_not_admitted:" + ",".join(blockers))
    completed, generation = generate_rgbd_completion(frames=frames, selected=selected, candidates=candidates,
        root=root, pose=pose, context=context, configuration=configuration, task_runner=task_runner,
        max_polls=max_polls, removed_objects=[obj["label"] for obj in review["automation"]["objects"]
            if obj["disposition"] == "remove"], checkpoint=checkpoint,
        scene_caption=proposal['scene_caption'], view_gap_reasons=reasons, geometry_frames=geometry_frames,
        **({'reference_frames': reference_frames} if reference_frames is not None else {}),
        **({'exported_gap_feedback': True} if splat_root is not None else {}))
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
    # Inspect the provider output before source-priority compositing can hide a
    # changed fixture or invented possession in its known-photo footprint.
    review_images = [{**image, 'image_path': image.get('native_image_path', image['image_path']),
                      'image_digest': image.get('native_image_digest', image['image_digest'])} for image in images]
    visual = verify_completed_background(frames=review_images, original_frames=review_images, plan=plan,
        output_root=root / "review", task_context=context, source_video_digest=survey["source_video_digest"])
    report = {"status": "passed" if visual["status"] == "passed" else "needs_input",
              "planner": planned, "context_fov_audit_digest": fov["digest"],
              "generation_result_digest": generation["digest"], "generated_roots": generation["generated_roots"],
              "generation": generation, "preservation_paths": generation["preservation_paths"],
              "generated_frame_ids": [row["frame_id"] for row in generated], "review": visual,
              "selected_camera_ids": ids, "target_cameras": [row["camera"] for row in selected],
              "gap_assessment": proposal["gap_assessment"], "remaining_unknowns": proposal["remaining_unknowns"],
              "generated_geometry_ready": visual["status"] == "passed", "completion_extent": "identified_missing_geometry_only",
              'gap_survey_digest': gaps['digest'],
              "generation_extent": extent, "scene_caption": proposal['scene_caption'],
              "view_gap_reasons": reasons,
              "claim_ceiling": "development_only",
              "whole_scene_complete": False, "coverage_basis": "surveyed_camera_gaps_only",
              "simulation_qualified": False, "metric_measurement_proven": False, "physical_evidence": False}
    if visual["status"] != "passed":
        report["reason"] = "atlas_completion_visual_review_required"
    report["digest"] = canonical_digest(report, digest_field="digest")
    write_json(root / "completion.json", report)
    return (generated if visual["status"] == "passed" else None), report
