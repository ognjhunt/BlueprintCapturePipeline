"""Production Atlas preparation controller for one website capture.

Source review, adaptive inputs, admitted posing, generated RGBD and portable
downloads run on the worker. The public API cannot create a splat, so the worker
fuses Atlas's own posed depth into a draft splat and meshes it with splats2Mesh
once a reviewed mesh price exists; neither becomes a metric or simulator-ready result.
Website uploads reach it through the worker in ``website_atlas_preview``. It is
the capture's only paid reconstruction lane: once its object-removed background
is complete it also poses the selected original views, and scene preparation
reads both (``website_atlas_scene_source``) without buying anything itself.
"""
from __future__ import annotations

import argparse
import fcntl
from copy import deepcopy
import json
import os
from pathlib import Path
import time

from .website_task_context import load_atlas_configuration as load_configuration
from .common import write_json
from .atlas_automation_plan import PRESERVED_SCENE_ORIGINS
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .atlas_preparation_context import requires_task_semantic_analysis


#: Who read the rate card, and when. Renewing it re-confirms the same prices.
_REVIEW_RECORD = ("rate_card_reference", "reviewed_by", "reviewed_on")
#: The per-review cap a job was admitted under. A cap raised later serves only the reviews this
#: one refuses, so a review bought before the raise is found again under its own receipt.
ADMITTED_REVIEW_CAP = "admitted_maximum_review_cost_usd"


def resumed_policy(stored, current):
    """The policy fields a job that already bought work under ``stored`` takes from ``current``.

    The authority and every price must be unchanged; the review record may be renewed. The
    per-review cap may rise but never fall below the admitted one: the authority's total still
    bounds what the job can spend.
    """
    admitted = stored.get(ADMITTED_REVIEW_CAP, stored["policy"]["maximum_review_cost_usd"])
    if (stored["authority"] != current["authority"]
            or current["policy"]["maximum_review_cost_usd"] < admitted
            or any(current["policy"].get(key) != value for key, value in stored["policy"].items()
                   if key not in (*_REVIEW_RECORD, "maximum_review_cost_usd"))):
        raise ValueError("atlas_runtime_authority_or_policy_changed")
    if ADMITTED_REVIEW_CAP in stored or current["policy"]["maximum_review_cost_usd"] != admitted:
        return {"policy": current["policy"], ADMITTED_REVIEW_CAP: admitted}
    return {"policy": current["policy"]}


def _require_current_price_review(policy) -> None:
    """Frame selection and posing refuse a stale price review; refuse it before buying a review."""
    from datetime import datetime, timezone
    from .website_atlas_pose import _reviewed_recently
    if not _reviewed_recently(policy.get("reviewed_on"), datetime.now(timezone.utc).date()):
        raise ValueError("atlas_runtime_price_review_expired")


def generation_request(*, pose, inputs, review):
    """Cover reviewed task AND room anchors at their observed camera locations."""
    from .website_worldlabs import _atlas_posed_context
    from .website_atlas_tasks import validate_task_request
    context = _atlas_posed_context(operation=pose["operation"], expected_frame_count=len(inputs["frames"]))
    by_id = {row["frame_id"]: frame for row, frame in zip(inputs["frames"], context, strict=True)}
    ids = list(dict.fromkeys(key for group in review["coverage_requirements"] for key in group["required_frame_ids"]))
    targets = []
    for key in ids:
        camera = deepcopy(by_id[key]["camera"])
        k = camera["intrinsics"]
        scale = 720 / k["height"]
        k.update(width=1280, height=720, fx=k["fx"] * scale, fy=k["fy"] * scale, cx=640, cy=360)
        targets.append(camera)
    request = {"contextFrames": context, "targetCameras": targets, "returnDepth": True,
               "prompt": "The same captured scene: " + review["scene_state"], "enhancePrompt": True}
    validate_task_request("atlasGenerate", request)
    return request


#: Owner direction 2026-10-22 (accuracy first): candidates every 0.1 s inside the
#: reviewed still interval, coarsened only to keep the second review within the
#: planner's image budget (16k + 3k tokens per image, under 1M).
ACCURACY_CANDIDATE_STEP_SECONDS = 0.1
MAXIMUM_REVIEW_FRAMES = 300


def accuracy_survey(*, source_video: Path, base_root: Path, base_survey: Path, review, output_root: Path) -> Path:
    """The reviewed still interval densified for accuracy; built once, then reused on resume."""
    from .atlas_video_extension import extend_survey
    extended_path = output_root / "extended_survey" / "atlas_video_survey.json"
    if not extended_path.exists():
        indexes = list((base_root / "source").rglob("decoded_observation_index.json"))
        if len(indexes) != 1:
            raise ValueError("atlas_runtime_decoded_index_ambiguous")
        span = sum(end - start for start, end in review["static_intervals_seconds"])
        step = max(ACCURACY_CANDIDATE_STEP_SECONDS, span / MAXIMUM_REVIEW_FRAMES)
        extend_survey(source_video=source_video, base_survey=base_survey, decoded_index=indexes[0], review=review,
                      output_root=extended_path.parent, step_seconds=step, gap_threshold_seconds=2 * step)
    return extended_path


def _source_pose(*, pose_root: Path, inputs, review, policy, task_context, check, experimental, max_polls,
                 reference_inputs=None, configuration=None, allow_scale_review=True):
    """The selected original views' own Atlas pose and the scale its reviewed references give.

    Retained receipts make a resume free. A refused purchase leaves the
    background intact and is reported, not raised.
    """
    from .atlas_scene_automation import measurement_plan
    from .atlas_scene_measurements import estimate_sizes, load_views
    from .paid_resource_allocator import run_sponsored_website_atlas_pose
    from .website_atlas_pose import ATLAS_PRICE_REVIEW_ENV, request_options
    from .atlas_source_pose_inputs import prepare_source_pose
    from .atlas_context_fov import verify_pose_context_fov

    parent_root = pose_root
    pose_root, inputs = prepare_source_pose(pose_root)
    count = len(inputs["frames"])
    quote = {"schema_version": "website_atlas_pose_price_review.v1", "endpoint": "/api/v2/tasks:images2PosedRGBD",
        "maximum_cost_usd": round(count * policy["pose_input_cost_usd"], 6), "max_input_images": count,
        "target_resolutions": [[720, 1280], [1280, 720]], "request_options": request_options(),
        **{k: policy[k] for k in _REVIEW_RECORD}}
    try:
        if reference_inputs is not None and (
                reference_inputs.get('digest') != canonical_digest(reference_inputs, digest_field='digest')
                or reference_inputs.get('heldout_pixels_included') is not False
                or reference_inputs.get('binding', {}).get('source_video_digest') != inputs['binding']['source_video_digest']
                or not {r['frame_id'] for r in reference_inputs['frames']} <= {r['frame_id'] for r in inputs['frames']}):
            raise ValueError('atlas_source_reference_inputs_changed')
        os.environ[ATLAS_PRICE_REVIEW_ENV] = json.dumps(quote)
        if not (pose_root / "atlas_posed_rgbd_result.json").is_file():
            _require_current_price_review(policy)
            check("source_pose")
        posed = run_sponsored_website_atlas_pose(output_root=pose_root, capture_id=task_context["capture_id"],
            task_context=task_context, experimental=experimental, max_polls=max_polls)
        if posed["status"] != "completed":
            return {"status": "processing", "stage": "source_pose", "operation_id": posed.get("operation_id")}
        pose, frames = load_views(pose_root, inputs)
        verify_pose_context_fov(pose_root=pose_root, frames=frames, output_root=pose_root)
        full_views = frames
        measurement_path = pose_root / 'scene_measurements.json'
        measurement_input_digest = reference_inputs['digest'] if reference_inputs is not None else None
        if reference_inputs is not None:
            # Retained reviews may name pixels on an earlier provider crop.
            # Register that exact grid to the full posed view; never reinterpret
            # its normalized coordinates as coordinates on the wider source.
            # The reference coordinates still name their original images.
            # Additional task-state views use their own full-view sources.
            from .atlas_context_fov import full_view_sources
            refs = {r['frame_id']: r for r in reference_inputs['frames']}
            reference_views = reference_inputs
            if {r['frame_id'] for r in inputs['frames']} - refs.keys():
                reference_views = {**reference_inputs, 'frames': [
                    refs.get(r['frame_id'], r) for r in full_view_sources(inputs, pose_root)]}
                reference_views.pop('full_view_sources', None)
            _, frames = load_views(pose_root, reference_views)
            measurement_path = pose_root / ('scene_measurements_reference-' + reference_inputs['digest'][7:23] + '.json')
        plan = measurement_plan(review)
        from .atlas_reference_masks import segment_scale_references
        reference_masks = segment_scale_references(frames=frames, references=plan.get("scale_references", []),
            root=pose_root / "reference_masks", task_context=task_context,
            source_digest=(inputs.get("binding") or {}).get("source_video_digest"), checkpoint=check)
        # The task's own supports and destinations, from the clean plate's retained
        # tracks, read only. They are no part of a measurement's binding: a retained
        # measurement is read back as it was written.
        from .atlas_scene_measurements import surfaces_on_posed_views
        from .website_task_masks import retained_task_surfaces
        pipeline_root = next((p for p in (parent_root, *parent_root.parents) if p.name == "pipeline"), None)
        selection = (retained_task_surfaces(pipeline_root=pipeline_root,
                                            source_video_digest=(inputs.get("binding") or {}).get(
                                                "source_video_digest"))
                     if pipeline_root is not None else
                     {"status": "unavailable", "reason": "clean_plate_task_masks_unavailable"})
        surfaces = surfaces_on_posed_views(full_views, inputs, selection)
        surface_source = {key: value for key, value in selection.items()
                          if key not in ("targets", "source_frame_registry")}
        # The selected original views: their camera height and named surfaces are scale cues too.
        measurements = estimate_sizes(frames=frames, automation=plan, pose_digest=pose["digest"],
                                      output=measurement_path, reference_masks=reference_masks,
                                      camera_height=True, surfaces=surfaces, surface_source=surface_source)
        if configuration is not None and allow_scale_review:
            from .atlas_scale_review import needs_scale_review, repair_source_scale, rests_on_height_priors
            from .paid_resource_allocator import run_sponsored_website_atlas_plan
            if needs_scale_review(measurements):
                try:
                    from .website_operation_coverage_review import scale_operation_coverage_options
                    semantic_options = scale_operation_coverage_options(job_dir=parent_root.parent,
                        task_context=task_context, source_video_digest=inputs["binding"]["source_video_digest"])
                    reviewed = repair_source_scale(pose_root=pose_root,
                        pose=pose, inputs=inputs, automation=plan, previous=measurements,
                        task_context=task_context, configuration=configuration, check=check,
                        planner_runner=lambda **kw: run_sponsored_website_atlas_plan(experimental=experimental, **kw),
                        surfaces=surfaces, surface_source=surface_source, **semantic_options)
                except ValueError as exc:
                    # The review found no reference to add: the height priors' guesstimate stands.
                    if str(exc) != "atlas_scale_review_abstained" or not rests_on_height_priors(measurements):
                        raise
                else:
                    (measurements, measurement_path), measurement_input_digest = reviewed, inputs['digest']
                if measurements.get("status") != "estimated":
                    return {"status": "needs_input", "reason": "atlas_scale_review_unresolved",
                            "scene_measurements_digest": measurements["digest"]}
    except ValueError as exc:
        return {"status": "needs_input", "reason": str(exc)}
    return {"status": "completed", "result_digest": pose["digest"], "operation_id": pose["operation"]["id"],
            "pose_relative_path": str(pose_root.relative_to(parent_root.parent)),
            'measurements_relative_path': str(measurement_path.relative_to(pose_root)),
            **({'measurement_reference_inputs_digest': measurement_input_digest} if measurement_input_digest else {}),
            "submitted_frame_count": count, "scene_measurements_digest": measurements["digest"],
            "meters_per_unit_estimate": measurements.get("meters_per_unit_estimate"),
            "scale_status": measurements.get("scale_status")}


def run_atlas_pipeline(*, output_root: Path, **options):
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run_atlas_pipeline(output_root=output_root, **options)
        if (result.get("mesh_available") and not result.get('observed_scene_only')
                and requires_task_semantic_analysis(options.get("task_context", {}))):
            from .website_operation_coverage_review import clarify_operation_coverage
            clarify_operation_coverage(job_dir=output_root, task_context=options["task_context"],
                source_video_digest=result["source_video_digest"], experimental=options.get("experimental", False),
                invoker=options.get("invoker"), checkpoint=options.get("checkpoint"))
        return result


def _job_edit_scope(output_root: Path, requested: str | None) -> str:
    """The job's edit scope, chosen once: a resume can never change which views were cleaned."""
    from .atlas_scene_automation import EDIT_SCOPES
    path = output_root / "edit_scope.json"
    stored = json.loads(path.read_text()).get("edit_scope") if path.is_file() else None
    if requested is not None and requested not in EDIT_SCOPES:
        raise ValueError("atlas_edit_scope_invalid")
    if stored is not None and requested is not None and requested != stored:
        raise ValueError("atlas_edit_scope_changed")
    scope = stored or requested or EDIT_SCOPES[0]
    if stored is None:
        write_json(path, {"schema_version": "atlas_edit_scope.v1", "edit_scope": scope})
    return scope


def _preserved_scene(*, pose_root, output_root, inputs, review, policy, task_context,
                     source_digest, configuration, check, experimental, max_polls, model,
                     source_survey_path=None, reconstruction_only=False, task_evidence_limitation=None):
    """Reconstruct an environment-preserving task from full-view observations.

    Selection and scale still use the production agents and validators. Missing
    surfaces enter the existing source-first bounded completion/recovery seams.
    Original observations and the first incomplete export remain retained.
    """
    from .atlas_source_pose_inputs import prepare_source_pose
    from .atlas_depth_splat import write_depth_splat, load_posed_rgbd, splats2mesh_attempt
    from .atlas_mesh_registration import register_atlas_mesh
    from .atlas_result_bundle import export_pose_bundle
    from .paid_resource_allocator import run_sponsored_website_atlas_task, run_sponsored_website_atlas_automation

    if ((not reconstruction_only and review.get("subject_origin") not in PRESERVED_SCENE_ORIGINS)
            or not review.get("automation")
            or any(row.get("disposition") != "keep" for row in review["automation"]["objects"])
            or review["automation"].get("removal_instruction", "").strip()
            or review.get("transient_frame_ids")):
        raise ValueError("atlas_synthetic_subject_requires_preserved_capture"
                         if review.get("subject_origin") == "synthetic_design"
                         else "atlas_scene_requires_preserved_capture")
    source_pose = _source_pose(pose_root=pose_root, inputs=inputs, review=review, policy=policy,
        task_context=task_context, check=check, experimental=experimental, max_polls=max_polls,
        configuration=configuration, **({'allow_scale_review': False} if reconstruction_only else {}))
    if source_pose["status"] == "processing":
        return source_pose
    source_root, _ = prepare_source_pose(pose_root)
    if not (source_root / "atlas_posed_rgbd_result.json").is_file():
        return {"status": "needs_input", "stage": "source_pose", "source_pose": source_pose,
                "reason": source_pose.get("reason", "atlas_source_pose_unavailable"),
                "blockers": [source_pose.get("reason", "atlas_source_pose_unavailable")]}
    pose, frames = load_posed_rgbd(source_root)
    splat_root = output_root / "splat"
    manifest = splat_root / "atlas_depth_splat_manifest.json"
    splat = (json.loads(manifest.read_text()) if manifest.is_file()
             else write_depth_splat(frames=frames, result=pose, output_root=splat_root))
    blockers = ["metric_scene_registration_unproven"]
    mesh_root = None
    completion = None
    generated_roots, completion_paths = [], []
    completion_stopped = False
    admitted_completion = False
    # A scale/authority refusal may still leave useful retained observations.
    # Export those locally, but never continue purchasing after that refusal.
    if source_pose["status"] != "completed":
        blockers.append(source_pose.get("reason", "atlas_source_pose_unavailable"))
    elif not reconstruction_only and "mesh_export_cost_usd" not in policy:
        blockers.append("atlas_mesh_export_price_unreviewed")
    elif 'rgbd_view_cost_usd' in policy and not reconstruction_only:
        if source_survey_path is None:
            completion_stopped = True
            blockers.append('atlas_original_capture_survey_required')
        else:
            def completion_checkpoint(stage, **_details):
                check(stage)
            completion = run_sponsored_website_atlas_automation(preserved_capture=True,
                pose_root=source_root, pose=pose, frames=frames, review=review,
                root=output_root / 'preserved_completion', job_root=output_root, context=task_context,
                configuration=load_configuration(task_context, source_digest), splat_root=splat_root,
                source_survey_path=source_survey_path, checkpoint=completion_checkpoint,
                experimental=experimental, max_polls=max_polls)
            write_json(output_root / 'preserved_completion/controller_state.json', completion)
            if completion['status'] == 'processing':
                return completion
            completion_stopped = completion['status'] != 'completed'
            if completion_stopped:
                blockers.append(completion.get('reason', 'atlas_preserved_completion_not_admitted'))
            elif completion.get('generated_geometry_admitted'):
                admitted_completion = True
                splat_root, splat = Path(completion['splat_root']), completion['splat_manifest']
                generated_roots = [Path(p) for p in completion['generated_roots']]
                completion_paths = [Path(p) for p in completion['analysis_paths']]
    if (not reconstruction_only and source_pose['status'] == 'completed' and 'mesh_export_cost_usd' in policy
            and not completion_stopped):
        mesh_destination = (output_root / 'preserved_completion/mesh' / splat['digest'][7:23]
                            if admitted_completion else output_root / 'mesh')
        spec, mesh_root = splats2mesh_attempt(splat_root, mesh_destination)
        quote = {"schema_version": "website_atlas_task_price_review.v1", "task": "splats2Mesh",
            "request_digest": canonical_digest(spec["request"]), "maximum_cost_usd": policy["mesh_export_cost_usd"],
            **{key: policy[key] for key in _REVIEW_RECORD}}
        check("mesh")
        mesh = run_sponsored_website_atlas_task(task="splats2Mesh", request=spec["request"], uploads=spec["uploads"],
            source_video_digest=source_digest, output_root=mesh_root, task_context=task_context,
            price_review=quote, experimental=experimental, max_polls=max_polls)
        if mesh["status"] != "completed":
            return {"status": "processing", "stage": "mesh", "operation_id": mesh.get("operation_id")}
        if not (mesh_root / "registered/atlas_mesh_registration.json").is_file():
            outputs = json.loads((mesh_root / "result.json").read_text())["outputs"]
            register_atlas_mesh(glb=mesh_root / next(row["path"] for row in outputs if row["role"] == "mesh"),
                splat_root=splat_root, output_root=mesh_root / "registered", reference_frames=frames,
                reference_pose_digest=pose["digest"])
    analysis = [output_root / "atlas_scene_review.json"]
    if reconstruction_only and task_evidence_limitation is not None:
        limitation_path = output_root / 'task_evidence_limitation.json'
        write_json(limitation_path, task_evidence_limitation)
        analysis.append(limitation_path)
    measurement_path = source_root / source_pose.get("measurements_relative_path", "scene_measurements.json")
    if not measurement_path.resolve().is_relative_to(source_root.resolve()) or measurement_path.is_symlink():
        raise ValueError("atlas_preserved_scene_measurement_path_invalid")
    if measurement_path.is_file():
        analysis.append(measurement_path)
    analysis.extend(completion_paths)
    if completion is not None:
        analysis.append(output_root / 'preserved_completion/controller_state.json')
    bundle = export_pose_bundle(pose_root=source_root, output_root=output_root / "downloads",
        splat_root=splat_root, mesh_root=mesh_root, analysis_paths=analysis,
        **({'generated_roots': generated_roots} if generated_roots else {}))
    native_completion = ((completion or {}).get('completion') or completion or {})
    result = {"schema_version": "website_atlas_pipeline_result.v1", "status": "needs_input",
        "source_video_digest": source_digest, "task_context_digest": task_context["context_digest"],
        "subject_origin": review["subject_origin"], "captured_scene_preserved": True,
        "object_removal_performed": False, "image_inpainting_performed": False,
        "generated_completion_performed": bool(native_completion.get('generated_roots')),
        "generated_geometry_admitted": admitted_completion, "uncaptured_regions": review["uncaptured_regions"],
        "geometry_authority": ('captured_rgb_depth_with_bounded_generated_completion_estimates'
                               if admitted_completion else 'captured_rgb_derived_depth_and_reconstructed_mesh_estimates'),
        "source_pose": source_pose, "download": bundle, "model": model,
        "task_and_surrounding_scene_covered": not reconstruction_only and not completion_stopped, "blockers": blockers,
        "splat_available": True, "splat_kind": splat["splat_kind"], "splat_manifest_digest": splat["digest"],
        "mesh_available": mesh_root is not None, "simulation_qualified": False,
        "claim_ceiling": "development_only", "physical_evidence": False,
        "scale_status": source_pose.get("scale_status", "unscaled_provider_world_units"),
        "billing_status": "provider_usage_unreported"}
    if completion is not None:
        result['scene_completion'] = {'status': completion['status'], 'reason': completion.get('reason'),
            'failure_class': completion.get('failure_class'),
            'generated_geometry_admitted': admitted_completion,
            'remaining_unknowns': native_completion.get('remaining_unknowns', []),
            **({'admission_digest': completion['admission']['digest'],
                'admission_relative_path': str(Path(completion['admission_path']).relative_to(output_root))}
               if admitted_completion else {})}
    if reconstruction_only:
        result.update(observed_scene_only=True, native_preparation_eligible=False, execution_authority_granted=False)
    result["digest"] = canonical_digest(result, digest_field="digest")
    write_json(output_root / "result.json", result)
    return result


def _run_atlas_pipeline(*, source_video: Path, output_root: Path, task_context, experimental=False,
                       invoker=None, max_polls=2, refinement_feedback: Path | None = None, checkpoint=None,
                       edit_scope: str | None = None):
    # ``checkpoint(stage)`` runs before every paid stage; raising there (a
    # withdrawal) stops the job before it buys or uploads anything more.
    check = checkpoint or (lambda _stage: None)
    from .atlas_video_inputs import survey_video, prepare_video_inputs, takes_reviewed_background
    from .atlas_scene_planner import proposal_review
    from .paid_resource_allocator import (run_sponsored_website_atlas_plan, run_sponsored_website_atlas_pose,
                                          run_sponsored_website_atlas_task)
    from .website_atlas_pose import ATLAS_PRICE_REVIEW_ENV, ATLAS_PROVIDER_RETENTION_ENV, request_options, atlas_api_config
    from .atlas_result_bundle import export_pose_bundle
    from .atlas_depth_splat import build_depth_splat
    from .atlas_mesh_registration import register_atlas_mesh

    output_root.mkdir(parents=True, exist_ok=True)
    source_digest = _sha256_file(source_video)
    from .atlas_observed_room import SELECTION as ROOM_SELECTION, run_observed_room, retain_selection
    if (output_root / ROOM_SELECTION).is_file():
        return run_observed_room(output_root=output_root, source_digest=source_digest, task_context=task_context,
            max_polls=max_polls, checkpoint=checkpoint, experimental=experimental)
    saved = output_root / "configuration.json"
    # Retained work may read its original allowance, never issue a new one.
    current = (load_configuration(task_context, source_digest, create=False) if saved.is_file()
               else load_configuration(task_context, source_digest))
    atlas_api_config()  # Refuse missing/unsafe production secrets before buying a review.
    reviews = output_root / "scene_planner"
    selection_path = output_root / "selection_binding.json"
    if saved.exists():
        stored = config = json.loads(saved.read_text())
        if not reviews.exists() and not selection_path.exists():
            # The scene review is the first purchase, and a lapsed price review stops a
            # job before it: a job that bought nothing takes the current configuration whole.
            config = current
        else:
            # A resumed job may gain a reviewed price for a stage it has not bought, a
            # renewed review of the same prices or a raised review cap; any changed
            # price or authority refuses.
            config = {**config, **resumed_policy(config, current)}
        config = {**config, "policy": current["policy"]}
        if config != stored:
            write_json(saved, config)
    else:
        config = current
        write_json(saved, config)
    policy = config["policy"]
    edit_scope = _job_edit_scope(output_root, edit_scope)
    os.environ[ATLAS_PROVIDER_RETENTION_ENV] = current["authority"]["provider_retention"]
    feedback = None
    if refinement_feedback is not None:
        from .atlas_refinement import load_refinement_feedback
        feedback = load_refinement_feedback(refinement_feedback, source_digest=source_digest,
                                            context_digest=task_context["context_digest"])
        if output_root.resolve() == Path(feedback["parent_root"]):
            raise ValueError("atlas_refinement_requires_new_job")
        binding_path = output_root / "refinement_binding.json"
        binding = {"feedback_digest": feedback["digest"], "parent_result_digest": feedback["parent_result_digest"]}
        if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
            raise ValueError("atlas_refinement_binding_changed")
        if not binding_path.exists():
            write_json(binding_path, binding)
    reuse_options = {}
    if feedback is not None:
        parent_background = Path(feedback['parent_root']) / 'automation/background/prepared_background.json'
        if parent_background.is_file():
            reuse_options['prior_background_path'] = parent_background
    base_root = output_root / "survey"
    survey_path = base_root / "atlas_video_survey.json"
    if feedback is not None:
        from .atlas_video_extension import extend_survey
        parent = Path(feedback["parent_root"])
        extended_root = output_root / "extended_survey"
        survey_path = extended_root / "atlas_video_survey.json"
        if not survey_path.exists():
            indexes = list((parent / "survey/source").rglob("decoded_observation_index.json"))
            if len(indexes) != 1:
                raise ValueError("atlas_runtime_decoded_index_ambiguous")
            extend_survey(source_video=source_video, base_survey=parent / "survey/atlas_video_survey.json",
                decoded_index=indexes[0], review=feedback["prior_review"], output_root=extended_root)
    elif not survey_path.exists():
        survey_video(source_video=source_video, output_root=base_root, capture_id=task_context["capture_id"])
    survey = json.loads(survey_path.read_text())
    if survey["source_video_digest"] != source_digest:
        raise ValueError("atlas_runtime_source_changed")
    # Reserve room for a second review, depth views and the mesh export. These are
    # dollar allowances, not arbitrary limits on the number of source images.
    # The saved configuration preserves the original authority and prices;
    # completed purchases and held unknown calls consume today's allowance.
    budget = (current["remaining_cost_usd"] - 2 * policy["maximum_review_cost_usd"]
              - policy.get("mesh_export_cost_usd", 0))
    if budget < 2 * policy["pose_input_cost_usd"] + 2 * policy["rgbd_view_cost_usd"]:
        raise ValueError("atlas_runtime_budget_insufficient")
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        survey_path = output_root / selection["survey_relative_path"]
        if not survey_path.resolve().is_relative_to(output_root.resolve()):
            raise ValueError("atlas_runtime_selection_escape")
        survey = json.loads(survey_path.read_text())
        review = selection["review"]
        receipt = selection["planner_receipt"]
    else:
        check("plan")
        _require_current_price_review(policy)
        plan_options = {"feedback": feedback} if feedback is not None else {}
        receipt = run_sponsored_website_atlas_plan(survey_path=survey_path, task_context=task_context,
            configuration=config, output_root=reviews, invoker=invoker, experimental=experimental, **plan_options)
        if feedback is None and receipt.get('proposal', {}).get('status') == 'abstained':
            # The review has settled or remains held against the original cap.
            # Select source inputs against the current remaining allowance.
            room_configuration = load_configuration(task_context, source_digest, create=False)
            try:
                retain_selection(output_root=output_root, survey_path=survey_path, survey=survey, receipt=receipt,
                    task_context=task_context, configuration={**room_configuration, 'policy': config['policy']})
            except ValueError as exc:
                if str(exc).startswith(('atlas_runtime_', 'atlas_observed_room_planner_receipt_changed')):
                    raise
                # The room also lacks a usable source selection. Retain that
                # limit and the original task abstention; buy no blind retry.
                write_json(output_root / 'observed_room/selection_limit.json', {
                    'status': 'needs_input', 'reason': str(exc),
                    'task_planner_receipt_digest': canonical_digest(receipt), 'provider_mutation_performed': False})
            else:
                return run_observed_room(output_root=output_root, source_digest=source_digest, task_context=task_context,
                    max_polls=max_polls, checkpoint=checkpoint, experimental=experimental)
        review = proposal_review(receipt["proposal"], survey=survey, policy=policy, pose_budget=budget, feedback=feedback, reference_sources=receipt.get("reference_sources", []))
        if feedback is None:
            subject_origin = review.get("subject_origin")
            # Every frame of the reviewed still interval becomes a candidate; the
            # denser survey is reviewed again before anything is selected or sent.
            survey_path = accuracy_survey(source_video=source_video, base_root=base_root, base_survey=survey_path,
                                          review=review, output_root=output_root)
            survey = json.loads(survey_path.read_text())
            check("plan")
            _require_current_price_review(policy)
            receipt = run_sponsored_website_atlas_plan(survey_path=survey_path, task_context=task_context,
                configuration=config, output_root=reviews, invoker=invoker, experimental=experimental)
            review = proposal_review(receipt["proposal"], survey=survey, policy=policy, pose_budget=budget, reference_sources=receipt.get("reference_sources", []))
            if subject_origin is not None and review.get("subject_origin") != subject_origin:
                raise ValueError("atlas_scene_subject_origin_changed")
    if review.get("action") == "keep":
        result = {"schema_version": "atlas_refinement_decision.v1", "status": "needs_input",
            "action": "keep", "gap_assessment": review["gap_assessment"],
            "parent_result_digest": feedback["parent_result_digest"], "parent_root": feedback["parent_root"],
            "planner_receipt": receipt, "claim_ceiling": "development_only", "simulation_qualified": False}
        write_json(output_root / "result.json", result)
        return result
    pose_root = output_root / "pose"
    pose_root.mkdir(exist_ok=True)
    inputs = prepare_video_inputs(survey_path=survey_path, review=review, output_root=pose_root)
    if not selection_path.exists():
        write_json(selection_path, {"survey_relative_path": str(survey_path.relative_to(output_root)),
                                   "review": review, "planner_receipt": receipt})
    from .website_task_semantics import retain_task_semantics
    retain_task_semantics(job_dir=output_root, receipt=receipt, task_context=task_context,
                          source_video_digest=source_digest)
    write_json(output_root / "atlas_scene_review.json", review)
    if review.get("subject_origin") in PRESERVED_SCENE_ORIGINS:
        from .atlas_source_gap_evidence import combine_source_surveys
        source_survey = combine_source_surveys(survey_paths=[
            (Path(feedback['parent_root']) / 'survey/atlas_video_survey.json'
             if feedback is not None else base_root / 'atlas_video_survey.json'), survey_path],
            output=output_root / 'capture_source_survey.json')
        return _preserved_scene(pose_root=pose_root, output_root=output_root, inputs=inputs, review=review,
            policy=policy, task_context=task_context, source_digest=source_digest,
            configuration=load_configuration(task_context, source_digest), check=check,
            experimental=experimental, max_polls=max_polls, model=receipt["model"],
            source_survey_path=source_survey)
    if takes_reviewed_background(review):
        # The editor/reviewer works from source pixels. Atlas receives no RGB
        # until the complete cleaned set passes; do not buy a raw-scene pose or
        # synthetic views merely to start image removal. People and other moving
        # things in the edited views are erased here too; the raw path below
        # never receives a frame showing one.
        from .paid_resource_allocator import run_sponsored_website_atlas_automation
        from .atlas_source_gap_evidence import combine_source_surveys
        full_source_survey = combine_source_surveys(
            survey_paths=[(Path(feedback['parent_root']) / 'survey/atlas_video_survey.json'
                           if feedback is not None else base_root / 'atlas_video_survey.json'), survey_path],
            output=output_root / 'capture_source_survey.json')
        check("reviewed_background")
        automation = run_sponsored_website_atlas_automation(
            pose_root=None, inputs=inputs, review=review, output_root=output_root / "automation",
            task_context=task_context, configuration=load_configuration(task_context, source_digest),
            experimental=experimental, max_polls=max_polls, checkpoint=checkpoint, edit_scope=edit_scope,
            source_survey_path=full_source_survey,
            **reuse_options)
        if automation.get('status') in {'source_recovery_required', 'source_review_required'}:
            return automation
        if automation["status"] == "processing":
            return automation
        complete = automation["status"] == "completed"
        source_pose = None
        if complete:
            # Scene preparation lifts task masks onto observed depth and needs a
            # scale: the original views' own pose, bought once here and shared.
            source_pose = _source_pose(pose_root=pose_root, inputs=inputs, review=review, policy=policy,
                task_context=task_context, check=check, experimental=experimental, max_polls=max_polls,
                configuration=config)
            if source_pose["status"] == "processing":
                return source_pose
        return finalize_background_result(output_root=output_root, source_digest=source_digest,
            task_context=task_context, automation=automation, source_pose=source_pose,
            model=receipt["model"], edit_scope=edit_scope)
    count = len(inputs["frames"])
    target_count = len({key for group in review["coverage_requirements"] for key in group["required_frame_ids"]})
    planned_cost = round(count * policy["pose_input_cost_usd"] + target_count * policy["rgbd_view_cost_usd"], 6)
    if planned_cost > budget + 1e-9:
        raise ValueError("atlas_runtime_budget_insufficient_for_pose_and_generation")
    pose_quote = {"schema_version": "website_atlas_pose_price_review.v1", "endpoint": "/api/v2/tasks:images2PosedRGBD",
        "maximum_cost_usd": round(count * policy["pose_input_cost_usd"], 6), "max_input_images": count,
        "target_resolutions": [[720, 1280], [1280, 720]], "request_options": request_options(),
        **{k: policy[k] for k in ("rate_card_reference", "reviewed_by", "reviewed_on")}}
    # Validate all deterministic quote/camera options before reserving a call.
    os.environ[ATLAS_PRICE_REVIEW_ENV] = json.dumps(pose_quote)
    check("pose")
    posed = run_sponsored_website_atlas_pose(output_root=pose_root, capture_id=task_context["capture_id"],
        task_context=task_context, experimental=experimental, max_polls=max_polls)
    if posed["status"] != "completed":
        return {"status": "processing", "stage": "pose", "operation_id": posed.get("operation_id")}
    pose = json.loads((pose_root / "atlas_posed_rgbd_result.json").read_text())
    request = generation_request(pose=pose, inputs=inputs, review=review)
    cost = round(len(request["targetCameras"]) * policy["rgbd_view_cost_usd"], 6)
    quote = {"schema_version": "website_atlas_task_price_review.v1", "task": "atlasGenerate",
        "request_digest": canonical_digest(request), "maximum_cost_usd": cost,
        **{k: policy[k] for k in ("rate_card_reference", "reviewed_by", "reviewed_on")}}
    generated_root = output_root / "generated"
    check("generated_rgbd")
    generated = run_sponsored_website_atlas_task(task="atlasGenerate", request=request,
        source_video_digest=source_digest, output_root=generated_root, task_context=task_context,
        price_review=quote, experimental=experimental, max_polls=max_polls)
    if generated["status"] != "completed":
        return {"status": "processing", "stage": "generated_rgbd", "operation_id": generated.get("operation_id")}
    splat_root = output_root / "splat"
    splat_manifest = splat_root / "atlas_depth_splat_manifest.json"
    splat = (json.loads(splat_manifest.read_text()) if splat_manifest.exists()
             else build_depth_splat(pose_root, splat_root))
    blockers = ["metric_scene_registration_unproven"]
    mesh_root = None
    if "mesh_export_cost_usd" in policy:
        from .atlas_depth_splat import splats2mesh_attempt

        spec, mesh_root = splats2mesh_attempt(splat_root, output_root / "mesh")
        mesh_quote = {"schema_version": "website_atlas_task_price_review.v1", "task": "splats2Mesh",
            "request_digest": canonical_digest(spec["request"]), "maximum_cost_usd": policy["mesh_export_cost_usd"],
            **{k: policy[k] for k in ("rate_card_reference", "reviewed_by", "reviewed_on")}}
        check("mesh")
        mesh = run_sponsored_website_atlas_task(task="splats2Mesh", request=spec["request"], uploads=spec["uploads"],
            source_video_digest=source_digest, output_root=mesh_root, task_context=task_context,
            price_review=mesh_quote, experimental=experimental, max_polls=max_polls)
        if mesh["status"] != "completed":
            return {"status": "processing", "stage": "mesh", "operation_id": mesh.get("operation_id")}
        if not (mesh_root / "registered" / "atlas_mesh_registration.json").exists():
            from .atlas_depth_splat import load_posed_rgbd
            reference_pose, reference_frames = load_posed_rgbd(pose_root)
            if reference_pose["digest"] != pose["digest"]:
                raise ValueError("atlas_mesh_depth_filter_pose_binding_invalid")
            outputs = json.loads((mesh_root / "result.json").read_text())["outputs"]
            register_atlas_mesh(glb=mesh_root / next(row["path"] for row in outputs if row["role"] == "mesh"),
                                splat_root=splat_root, output_root=mesh_root / "registered",
                                reference_frames=reference_frames, reference_pose_digest=reference_pose["digest"])
    else:
        blockers.insert(0, "atlas_mesh_export_price_unreviewed")
    bundle = export_pose_bundle(pose_root=pose_root, output_root=output_root / "downloads",
                               generated_roots=[generated_root], splat_root=splat_root, mesh_root=mesh_root)
    result = {"schema_version": "website_atlas_pipeline_result.v1", "status": "needs_input",
        "source_video_digest": source_digest, "task_context_digest": task_context["context_digest"],
        "pose": posed, "generated_result_digest": generated["digest"], "download": bundle,
        "task_and_surrounding_scene_covered": True, "model": receipt["model"],
        "blockers": blockers, "splat_available": True, "splat_kind": splat["splat_kind"],
        "splat_manifest_digest": splat["digest"], "mesh_available": mesh_root is not None, "simulation_qualified": False,
        "claim_ceiling": "development_only", "physical_evidence": False,
        "scale_status": "unscaled_provider_world_units", "billing_status": "provider_usage_unreported"}
    if review.get("automation") is not None:
        from .paid_resource_allocator import run_sponsored_website_atlas_automation
        automation = run_sponsored_website_atlas_automation(pose_root=pose_root, inputs=inputs, review=review,
            output_root=output_root / "automation", task_context=task_context,
            configuration=load_configuration(task_context, source_digest), experimental=experimental, max_polls=max_polls,
            checkpoint=checkpoint, edit_scope=edit_scope, **reuse_options)
        result["automation"] = automation
        result["size_estimates"] = automation.get("measurements")
        if automation["status"] == "processing":
            result.update(status="processing", stage="automation", operation_id=automation.get("operation_id"))
        analysis = [p for p in (output_root / "automation/scene_measurements.json", output_root / "automation/result.json") if p.is_file()]
        background_download = (automation.get("background") or {}).get("download")
        if background_download:
            background_zip = Path(background_download["zip_path"])
            if _sha256_file(background_zip) != background_download["zip_sha256"]:
                raise ValueError("atlas_background_bundle_changed")
            analysis.append(background_zip)
        result["download"] = export_pose_bundle(pose_root=pose_root, output_root=output_root / "downloads",
            generated_roots=[generated_root], splat_root=splat_root, mesh_root=mesh_root, analysis_paths=analysis)
    if feedback is not None:
        result["refinement"] = {"parent_result_digest": feedback["parent_result_digest"],
            "feedback_digest": feedback["digest"], "gap_assessment": review["gap_assessment"],
            "prior_input_count": len(feedback["prior_selected_frame_ids"]), "new_input_count": count}
    result["digest"] = canonical_digest(result, digest_field="digest")
    write_json(output_root / "result.json", result)
    return result


def finalize_background_result(*, output_root, source_digest, task_context, automation, source_pose, model, edit_scope=None):
    """The same durable handoff for a fresh job or a retained completed background."""
    background = automation.get("background") or {}
    complete = automation["status"] == "completed"
    result = {"schema_version": "website_atlas_pipeline_result.v1", "status": "needs_input",
              "source_video_digest": source_digest, "task_context_digest": task_context["context_digest"],
              "automation": automation, "download": background.get("download"), "edit_scope": edit_scope,
              "task_and_surrounding_scene_covered": True, "model": model,
              "blockers": (["metric_scene_registration_unproven"] if complete else
                           [background.get("reason", "atlas_background_review_required")]),
              "splat_available": complete, "splat_kind": "depth_fused_draft" if complete else None,
              "splat_manifest_digest": background.get("splat_manifest_digest"),
              "mesh_available": background.get("mesh_available", False),
              "scene_measurements": automation.get("measurements"),
              "source_pose": source_pose,
              "simulation_qualified": False, "claim_ceiling": "development_only",
              "physical_evidence": False, "scale_status": "unscaled_provider_world_units",
              "billing_status": "provider_usage_unreported"}
    result["digest"] = canonical_digest(result, digest_field="digest")
    write_json(output_root / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--maximum-runtime-seconds", type=int, default=1200)
    parser.add_argument("--experimental", action="store_true")
    parser.add_argument("--refinement-feedback", type=Path,
                        help="Digest-bound parent result and derived previews; uses a fresh revision job and the same allowance")
    from .atlas_scene_automation import EDIT_SCOPES
    parser.add_argument("--edit-scope", choices=EDIT_SCOPES,
                        help="Views the editor cleans; fixed for the job at its first run (trial: all_selected_views)")
    args = vars(parser.parse_args())
    from .website_task_context import load_current_website_task_context
    context = load_current_website_task_context(request_id=args.pop("request_id"), capture_id=args.pop("capture_id"),
        scene_id=args.pop("scene_id"), purpose="scene_preparation")
    seconds = args.pop("maximum_runtime_seconds")
    if not 1 <= seconds <= 86400:
        raise ValueError("atlas_runtime_deadline_invalid")
    deadline = time.monotonic() + seconds
    while True:
        result = run_atlas_pipeline(task_context=context, **args)
        print(json.dumps(result), flush=True)
        if result["status"] != "processing" or time.monotonic() >= deadline:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
