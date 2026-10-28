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


def _context_binding(context_basis, review, pose, context):
    if context_basis == 'generated_background':
        return {}
    from .atlas_automation_plan import PRESERVED_SCENE_ORIGINS

    automation = review.get('automation') or {}
    objects = automation.get('objects')
    if (context_basis != 'preserved_capture'
            or review.get('subject_origin') not in PRESERVED_SCENE_ORIGINS
            or not automation
            or not isinstance(objects, list)
            or any(not isinstance(row, dict) or row.get('disposition') != 'keep' for row in objects)
            or automation.get('removal_instruction', '').strip()
            or review.get('transient_frame_ids')):
        raise ValueError('atlas_completion_preserved_context_invalid')
    return {'context_basis': context_basis, 'source_pose_digest': pose['digest'],
            'source_video_digest': pose['binding']['source_video_digest'],
            'task_context_digest': context['context_digest']}


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
                   reference_frames=None, coverage_feedback=None, source_survey_path=None,
                   recorded_frames=(), context_pose_root=None, context_pose_digest=None,
                   context_basis='generated_background'):
    """Plan -> generate RGBD -> independent review, all resumable under one cap."""
    from .website_image_completion import verify_completed_background

    binding = _context_binding(context_basis, review, pose, context)
    root.mkdir(parents=True, exist_ok=True)
    # Completed native generation and its admission remain reusable. A new
    # source-first decision governs future purchases, not another purchase of
    # the same images or a replay of their finished geometry work.
    saved_report = root / 'completion.json'
    if saved_report.is_file():
        saved = json.loads(saved_report.read_text())
        if (saved.get('context_basis', 'generated_background') != context_basis
                or binding and any(saved.get(key) != value for key, value in binding.items())):
            raise ValueError('atlas_completion_context_basis_changed')
        native_roots = saved.get('generated_roots', [])
        if (saved.get('status') == 'passed' and native_roots
                and all((Path(p) / 'result.json').is_file() for p in native_roots)):
            from .atlas_scene_gaps import retained_completion
            return retained_completion(root=root, pose_digest=pose['digest'])
    from .atlas_context_fov import verify_pose_context_fov
    pixel_root = Path(context_pose_root) if context_pose_root is not None else pose_root
    pixel_pose = (json.loads((pixel_root / 'atlas_posed_rgbd_result.json').read_text())
                  if context_pose_root is not None else pose)
    if context_pose_root is not None and pixel_pose.get('digest') != canonical_digest(pixel_pose, digest_field='digest'):
        raise ValueError('atlas_completion_refreshed_context_changed')
    if context_pose_root is not None and (
            pixel_pose.get('digest') != context_pose_digest
            or pixel_pose['binding']['source_video_digest'] != pose['binding']['source_video_digest']
            or {row['frame_id'] for row in pixel_pose['operation_receipt']['input_usage']['frames']}
            != {frame['frame_id'] for frame in frames}):
        raise ValueError('atlas_completion_refreshed_context_identity_changed')
    fov = verify_pose_context_fov(pose_root=pixel_root, frames=frames, output_root=root)
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
            target['expected_frame_id'] = expected['frame_id']
            candidates.append(target)
            camera_pool.append(target)
        # Every reference is surveyed in its own camera. Other candidate views
        # use the nearest reviewed RGBD camera as their depth hypothesis instead
        # of taking the minimum of contradictory estimates from all views.
        # All approved photographs/depths still accompany the planner and Atlas.
        from .atlas_depth_splat import _rotation_matrix
        scales = [float(np.median(f['depth'][np.isfinite(f['depth']) & (f['depth'] > 0)
                                            & (f['confidence'] > 0)])) for f in expected_frames]
        scale = max(float(np.median(scales)), np.finfo(float).eps)
        for target in camera_pool:
            camera = target['camera']['extrinsics']
            rotation = _rotation_matrix(camera['quaternion'])
            closest = min(expected_frames, key=lambda f:
                np.linalg.norm(f['position'] - camera['position']) / scale +
                np.linalg.norm(f['rotation'] - rotation))
            target.setdefault('expected_frame_id', closest['frame_id'])
    from .atlas_scene_preservation import observed_scene_bounds
    extent = observed_scene_bounds([*frames, *recorded_frames])
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
    if not any(row['gaps'] for row in gaps['views']) and coverage_feedback is None:
        report = {'status': 'passed', 'generation_result_digest': gaps['digest'], 'generated_roots': [],
                  'generated_frame_ids': [], 'preservation_paths': [str(root / 'gaps/survey.json')],
                  'review': {'status': 'not_required', 'reason': 'no_eligible_geometry_gaps'},
                  'gap_survey_digest': gaps['digest'], 'completion_extent': 'identified_missing_geometry_only',
                  'generation_extent': extent, 'generated_geometry_ready': True, 'claim_ceiling': 'development_only',
                  **binding}
        report.update(whole_scene_complete=False, coverage_basis='surveyed_camera_gaps_only')
        report['digest'] = canonical_digest(report, digest_field='digest')
        write_json(root / 'completion.json', report)
        return [], report
    candidates, unsupported = context_supported_candidates(candidates, gaps)
    usable = {row['camera_id'] for row in candidates}
    with_gaps = {row['camera_id'] for row in gaps['views'] if row['gap_pixel_count']}
    # Show the actual proposed camera views, including dedicated gap views.
    # The source-camera overlay cannot reveal a bad new viewpoint.
    preview_ids = usable if coverage_feedback is not None else usable & with_gaps
    previews = [row for row in gaps['previews'] if row['camera_id'] in preview_ids]
    if not preview_ids:
        raise ValueError('atlas_completion_no_context_supported_gap_camera')
    originals = {}
    survey_frames = []
    usage = pixel_pose["operation_receipt"]["input_usage"]["frames"]
    for index, row in enumerate(usage):
        asset = next(value for value in row["outputs"] if value["role"] == "image")
        path = pixel_root / asset["path"]
        if (path.is_symlink() or not path.resolve().is_relative_to(pixel_root.resolve())
                or _sha256_file(path) != asset["sha256"]):
            raise ValueError("atlas_completion_context_changed")
        originals[row["frame_id"]] = {"image_path": str(path), "image_digest": asset["sha256"]}
        survey_frames.append({"frame_id": row["frame_id"], "timestamp_seconds": index,
                              "sharpness": 0, "clipped_fraction": 0,
                              "source_image_path": str(path), "source_image_digest": asset["sha256"]})
    survey = {"source_video_digest": pose["binding"]["source_video_digest"], "heldout_pixels_included": False,
              "display_rotation_degrees": 0, "frames": survey_frames,
              "content_basis": ('preserved_capture_context' if binding else
                                'independently_reviewed_generated_background')}
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
                "input_basis": ('preserved_capture_context_with_estimated_depth' if binding else
                                'generated_background_not_raw_capture'),
                "claim_ceiling": "development_only", **binding}
    if coverage_feedback is not None:
        feedback['post_export_coverage'] = coverage_feedback
    if expected_frames:
        from .atlas_scene_chisel import prepare_source_photographs
        photos = prepare_source_photographs(frames=expected_frames, root=root / 'reviewed-references')
        feedback['previews'].extend({'path': p['image_path'], 'sha256': p['image_digest'],
            'source_frame_id': p['source_frame_id'], 'camera_id': p['source_frame_id'],
            'kind': 'reviewed_rgbd_reference_depth_is_a_hypothesis'} for p in photos)
    from .atlas_source_gap_evidence import source_gallery, source_actions
    gallery, source_previews = source_gallery(
        survey_path=source_survey_path, source_video_digest=pose['binding']['source_video_digest'],
        conditioned_frame_ids=[f['frame_id'] for f in [*frames, *recorded_frames]], root=root / 'original-capture')
    feedback['original_capture_gallery'] = gallery
    feedback['previews'].extend(source_previews)
    # Source review can be added to a retained, unfinished pass without
    # overwriting its older planner input or repurchasing finished native work.
    feedback_path = root / ('plan-inputs-source-' + gallery['survey_digest'][7:23] + '.json')
    if coverage_feedback is not None:
        # Keep the previous planner input/receipt while reusing native repairs.
        feedback_path = feedback_path.with_name(feedback_path.stem + '-surfaces-v1.json')
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
    prior_feedback = root / ('plan-inputs-source-' + gallery['survey_digest'][7:23] + '.json')
    if coverage_feedback is not None and not feedback_path.exists() and prior_feedback.is_file():
        saved = json.loads(prior_feedback.read_text())
        if {key: saved.get(key) for key in feedback} == feedback:
            # A review-policy upgrade retains the exact evidence/budget snapshot;
            # the live reservation still rechecks the current remaining allowance.
            write_json(feedback_path, saved)
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
    if coverage_feedback is not None:
        from .atlas_scene_planner import CoverageCompletionProposal
        proposal = CoverageCompletionProposal.model_validate(proposal).model_dump(mode='json')
        camera_ids = {row['camera_id'] for row in gaps['views']}
        source_ids = set(gallery['reviewed_source_frame_ids'])
        gap_ids = {gap['gap_id'] for row in gaps['views'] for gap in row['gaps']}
        if any(not set(row['evidence_camera_ids']) <= camera_ids
               or not set(row['source_frame_ids']) <= source_ids
               or not set(row['gap_ids']) <= gap_ids
               for row in proposal['environment_surfaces']):
            raise ValueError('atlas_coverage_environment_evidence_invalid')
    if proposal['status'] == 'sufficient' and not proposal['selected_camera_ids']:
        report = {'status': 'passed', 'generation_result_digest': gaps['digest'],
                  'generated_roots': [], 'generated_frame_ids': [], 'planner': planned,
                  'preservation_paths': [str(root / 'gaps/survey.json'), str(survey_path), str(feedback_path)],
                  'review': {'status': 'not_required', 'reason': 'agent_found_no_material_task_gap'},
                  'gap_survey_digest': gaps['digest'], 'remaining_unknowns': proposal['remaining_unknowns'],
                  'gap_assessment': proposal['gap_assessment'], 'whole_scene_complete': False,
                  'environment_surfaces': proposal.get('environment_surfaces', []),
                  'coverage_basis': 'task_prioritized_review_of_surveyed_export',
                  'generated_geometry_ready': True, 'claim_ceiling': 'development_only', **binding}
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
    audit = source_actions(entries, gallery)
    write_json(root / 'source-evidence.json', audit)
    if audit['additional_source_frame_ids'] or audit['unresolved_gap_ids']:
        report = {'schema_version': 'atlas_source_gap_action.v1',
                  'status': ('source_recovery_required' if audit['additional_source_frame_ids']
                             else 'source_review_required'),
                  'reason': ('atlas_recorded_source_views_must_be_recovered_before_generation'
                             if audit['additional_source_frame_ids'] else 'atlas_source_visibility_uncertain'),
                  'source_evidence': audit, 'planner': planned, 'new_atlas_generation_calls': 0,
                  'claim_ceiling': 'development_only', **binding}
        report['digest'] = canonical_digest(report, digest_field='digest')
        write_json(root / 'source-action.json', report)
        return None, report
    # Empty space is not missing material. Keep its evidence and exclude it
    # from the actual masks handed to the generation adapter.
    entries = [{**row, 'gap_ids': [key for key in row['gap_ids']
                 if key not in audit['physical_opening_gap_ids'].get(row['camera_id'], [])]}
               for row in entries]
    entries = [row for row in entries if row['gap_ids']]
    ids = [row['camera_id'] for row in entries]
    selected = [by_id[key] for key in ids]
    reasons = {row['camera_id']: row['reason'] for row in entries}
    if not selected:
        report = {'status': 'passed', 'generation_result_digest': audit['digest'],
                  'generated_roots': [], 'generated_frame_ids': [], 'planner': planned,
                  'preservation_paths': [str(root / 'source-evidence.json')],
                  'review': {'status': 'not_required', 'reason': 'source_review_found_physical_openings'},
                  'source_evidence': audit, 'new_atlas_generation_calls': 0,
                  'generated_geometry_ready': True, 'whole_scene_complete': False,
                  'claim_ceiling': 'development_only', **binding}
        report['digest'] = canonical_digest(report, digest_field='digest')
        write_json(root / 'completion.json', report)
        return [], report
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
        **({'extent_frames': [*frames, *recorded_frames]} if recorded_frames else {}),
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
    if binding:
        plan['context_basis'] = context_basis
    _, blockers = release_blockers()
    if blockers:
        raise ValueError("atlas_completion_review_release_not_admitted:" + ",".join(blockers))
    if checkpoint:
        checkpoint("automation_completion_review")
    # Inspect the provider output before source-priority compositing can hide a
    # changed fixture or invented possession in its known-photo footprint.
    review_images = [{**image, 'image_path': image.get('native_image_path', image['image_path']),
                      'image_digest': image.get('native_image_digest', image['image_digest'])} for image in images]
    capture_references = [{"frame_id": row["source_frame_id"], "image_path": row["path"],
                           "image_digest": row["sha256"]} for row in source_previews]
    # Raw capture is review-only context. It may contain the removed targets;
    # it must never become an unclean input to native background generation.
    visual = verify_completed_background(frames=review_images,
        original_frames=[*review_images, *capture_references], plan=plan,
        output_root=root / "review", task_context=context, source_video_digest=survey["source_video_digest"])
    report = {"status": "passed" if visual["status"] == "passed" else "needs_input",
              "planner": planned, "context_fov_audit_digest": fov["digest"],
              "generation_result_digest": generation["digest"], "generated_roots": generation["generated_roots"],
              "generation": generation,
              "preservation_paths": [*generation["preservation_paths"], str(root / 'source-evidence.json')],
              "generated_frame_ids": [row["frame_id"] for row in generated], "review": visual,
              "selected_camera_ids": ids, "target_cameras": [row["camera"] for row in selected],
              "gap_assessment": proposal["gap_assessment"], "remaining_unknowns": proposal["remaining_unknowns"],
              "environment_surfaces": proposal.get("environment_surfaces", []),
              "generated_geometry_ready": visual["status"] == "passed", "completion_extent": "identified_missing_geometry_only",
              'gap_survey_digest': gaps['digest'],
              'source_evidence': audit,
              "generation_extent": extent, "scene_caption": proposal['scene_caption'],
              "view_gap_reasons": reasons,
              "claim_ceiling": "development_only",
              "whole_scene_complete": False, "coverage_basis": "surveyed_camera_gaps_only",
              "simulation_qualified": False, "metric_measurement_proven": False, "physical_evidence": False,
              **binding}
    if visual["status"] != "passed":
        report["reason"] = "atlas_completion_visual_review_required"
    report["digest"] = canonical_digest(report, digest_field="digest")
    write_json(root / "completion.json", report)
    return (generated if visual["status"] == "passed" else None), report
