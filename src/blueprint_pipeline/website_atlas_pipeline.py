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
import inspect
from copy import deepcopy
import json
import os
from pathlib import Path
import time

from .website_task_context import load_atlas_configuration as load_configuration
from .common import write_json
from .atlas_automation_plan import PRESERVED_SCENE_ORIGINS
from .atlas_pricing import admitted_final_outputs, render_cost, final_output_cost
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .atlas_preparation_context import requires_task_semantic_analysis


#: Who read the rate card, and when. Renewing it re-confirms the same prices.
_REVIEW_RECORD = ("rate_card_reference", "reviewed_by", "reviewed_on")
#: Every per-review cap a job has run under, lowest first. Each receipt binds the cap it was
#: priced under, so a review bought under any of them is found again rather than bought twice.
REVIEW_CAPS = "review_caps_usd"


# The owner may raise a running job's allowance by amending its own canonical
# authorization; who, what source, which terms and retention stay fixed.
_RAISABLE_LIMITS = ("maximum_cost_usd", "maximum_requests")
_AMENDMENT_RECORD = ("approval_reference", "expires_at_epoch", "authorization_digest")

#: The host's route for a new job, set on the worker unit and unset by default (``full``).
#: ``geometry_only`` keeps the scene as filmed for the robot fit check: no object removal,
#: image editing, generated completion or refined splat, only the pose, the scale review and
#: the mesh.
ROUTE_ENV = "BLUEPRINT_WEBSITE_ATLAS_ROUTE"
ROUTES = ("full", "geometry_only")


def _job_route(output_root: Path, *, purchased: bool) -> str:
    """The job's route, chosen once: a geometry-only job records it before its first purchase.

    A job that has bought anything without that record is on the full route, so a
    full-route job writes nothing and keeps its directory as it was.
    """
    path = output_root / "route.json"
    if path.is_file():
        route = json.loads(path.read_text()).get("route")
    else:
        route = "full" if purchased else (os.getenv(ROUTE_ENV, "").strip() or "full")
    if route not in ROUTES:
        raise ValueError("atlas_route_invalid")
    if route != "full" and not path.is_file():
        write_json(path, {"schema_version": "website_atlas_route.v1", "route": route})
    return route


def _route_policy(policy, route: str):
    """The rate card a route can buy from: geometry only never buys the refined splat, so never holds it back."""
    if route == "geometry_only":
        return {key: value for key, value in policy.items() if key != "refined_splat_cost_usd"}
    return policy


def _same_authority(stored, current):
    amendable = (*_RAISABLE_LIMITS, *_AMENDMENT_RECORD)
    if ({k: v for k, v in stored.items() if k not in amendable}
            != {k: v for k, v in current.items() if k not in amendable}):
        return False
    # A limit the owner removed is raised; dollars stay bounded by WebApp admission.
    return all(key not in current or current[key] == stored[key]
               or (type(current[key]) in (int, float) and type(stored[key]) in (int, float)
                   and current[key] >= stored[key])
               for key in _RAISABLE_LIMITS if key in stored)


def resumed_policy(stored, current):
    """The policy fields a job that already bought work under ``stored`` takes from ``current``.

    Every price must be unchanged; the review record may be renewed. The authority may only
    be the same grant with limits the owner raised: canonical admission keeps counting what
    the job already used, so a raise continues the job instead of buying it again. The
    per-review cap may rise but never fall below one the job ran under.
    """
    cap = current["policy"]["maximum_review_cost_usd"]
    caps = stored.get(REVIEW_CAPS, [stored["policy"]["maximum_review_cost_usd"]])
    if (not _same_authority(stored["authority"], current["authority"]) or cap < caps[-1]
            or any(current["policy"].get(key) != value for key, value in stored["policy"].items()
                   if key not in (*_REVIEW_RECORD, "maximum_review_cost_usd"))):
        raise ValueError("atlas_runtime_authority_or_policy_changed")
    caps = [*caps, cap] if cap > caps[-1] else caps
    return {"policy": current["policy"], REVIEW_CAPS: caps} if len(caps) > 1 else {"policy": current["policy"]}


def _current_configuration(task_context, source_digest, resumed):
    """The current allowance for a later stage, keeping every review cap the job ran under."""
    value = load_configuration(task_context, source_digest)
    return {**value, REVIEW_CAPS: resumed[REVIEW_CAPS]} if REVIEW_CAPS in resumed else value


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


def _job_edit_scope(output_root: Path, requested: str | None, *, default="required_and_additional") -> str:
    """The job's edit scope, chosen once: a resume can never change which views were cleaned."""
    from .atlas_scene_automation import EDIT_SCOPES
    path = output_root / "edit_scope.json"
    stored = json.loads(path.read_text()).get("edit_scope") if path.is_file() else None
    if requested is not None and requested not in EDIT_SCOPES:
        raise ValueError("atlas_edit_scope_invalid")
    if stored is not None and requested is not None and requested != stored:
        raise ValueError("atlas_edit_scope_changed")
    scope = stored or requested or default
    if stored is None:
        write_json(path, {"schema_version": "atlas_edit_scope.v1", "edit_scope": scope})
    return scope


def _splat_input_options(pose_root, *, review, policy, edit_scope):
    """Keep historical inputs immutable; new refined-splat jobs pose denser real footage.

    A removal job that cleans every selected view keeps 16 to 30 of them: each is
    also an edit. A preserved scene's refined splat trains on its posed selection,
    so that selection fills toward ``SPLAT_FRAME_TARGET`` sharp views spread in
    time. Every new selection's added views also clear the clip's relative
    sharpness floor (``RELATIVE_SHARPNESS_FRACTION``). A retained selection (or
    its inputs, before the plan is written) keeps the profile it recorded.
    """
    from .atlas_video_inputs import GEOMETRY_INPUTS_FILE, RELATIVE_SHARPNESS_FRACTION, takes_reviewed_background
    from .website_atlas_pose import SPLAT_FRAME_TARGET
    plan, inputs = pose_root / "atlas_view_selection.json", pose_root / GEOMETRY_INPUTS_FILE
    if plan.is_file() or inputs.is_file():
        recorded = (json.loads(plan.read_text()).get("frame_policy", {}) if plan.is_file()
                    else json.loads(inputs.read_text()).get("binding", {}))
        dense, target = recorded.get("minimum_source_frames") == 16, recorded.get("splat_frame_target")
        sharpness = recorded.get("relative_sharpness_fraction")
    else:
        splat = "refined_splat_cost_usd" in policy
        dense = splat and edit_scope == "all_selected_views" and takes_reviewed_background(review)
        target = SPLAT_FRAME_TARGET if splat and review.get("subject_origin") in PRESERVED_SCENE_ORIGINS else None
        sharpness = RELATIVE_SHARPNESS_FRACTION
    profile = {"relative_sharpness_fraction": sharpness} if sharpness else {}
    if dense:
        return {"minimum_frames": 16, "maximum_frames": 30, **profile}
    return {**({"splat_frame_target": target} if target else {}), **profile}


REFINED_APPEARANCE, DRAFT_APPEARANCE = "atlas_refined_splat", "depth_fused_draft_fallback"


def _splat_input_selection(*, frames, pose, draft, output_root, review, source_root, bought):
    """The posed views the refined splat is sent, decided once before its purchase and recorded.

    ``atlas_splat_input_quality.select_splat_frames`` reads the draft's per-pair
    agreement and the posed cameras. A splat bought before that record existed
    keeps the views it was sent (None: every posed view); a recorded selection
    is read back, never re-derived.
    """
    from .atlas_splat_input_quality import SELECTION_FILE, scene_depth, select_splat_frames
    from .atlas_video_inputs import GEOMETRY_INPUTS_FILE, WORKING_AREA_ROLES
    path = output_root / SELECTION_FILE
    if path.is_symlink():
        raise ValueError("atlas_splat_input_selection_invalid")
    if path.is_file():
        record = json.loads(path.read_text())
        if (record.get("digest") != canonical_digest(record, digest_field="digest")
                or record.get("pose_result_digest") != pose["digest"]
                or record.get("draft_splat_manifest_digest") != draft["digest"]
                or record.get("posed_frame_ids") != [frame["frame_id"] for frame in frames]):
            raise ValueError("atlas_splat_input_selection_changed")
        return record
    if bought:
        return None
    inputs_path = source_root / GEOMETRY_INPUTS_FILE
    rows = json.loads(inputs_path.read_text()).get("frames", []) if inputs_path.is_file() else []
    agreement = draft.get("cross_view_agreement") if draft.get("pose_result_digest") == pose["digest"] else None
    record = select_splat_frames(frames=frames, agreement=agreement,
        timestamps={row["frame_id"]: row.get("timestamp_seconds") for row in rows},
        sharpness={row["frame_id"]: row.get("sharpness") for row in rows
                   if isinstance(row.get("sharpness"), (int, float))},
        scene_depth=scene_depth(frames),
        pose_refinement_accepted=pose["operation"]["response"].get("poseRefinementAccepted"),
        route_frame_ids={key for group in review.get("coverage_requirements") or []
                         if group.get("role") in WORKING_AREA_ROLES for key in group.get("required_frame_ids", [])})
    record.pop("digest")
    record.update(pose_result_digest=pose["digest"], draft_splat_manifest_digest=draft["digest"])
    record["digest"] = canonical_digest(record, digest_field="digest")
    write_json(path, record)
    return record


def _heldout_quality(*, frames, selection, splat_root):
    """Score the returned splat against the views held back from it, once; record and flag only."""
    from .atlas_splat_input_quality import SCORE_FILE, heldout_quality
    path = Path(splat_root) / SCORE_FILE
    if path.is_file() and not path.is_symlink():
        return json.loads(path.read_text())
    try:
        from .atlas_depth_splat import decode_spz
        splat = decode_spz((Path(splat_root) / "atlas_refined_splat.spz").read_bytes())
        record = heldout_quality(splat=splat, frames=frames, selection=selection)
    except Exception as exc:  # noqa: BLE001 - a score never blocks the splat it describes
        record = {"schema_version": "atlas_splat_heldout_score.v1", "status": "not_scored",
                  "reason": type(exc).__name__, "selection_digest": selection.get("digest")}
    write_json(path, record)
    return record


def _preserved_appearance(*, frames, pose, draft, output_root, policy, task_context, source_digest,
                          configuration, check, experimental, max_polls, admitted_completion,
                          review=None, source_root=None, rebuild=False):
    """Atlas's refined splat of the captured frames, or the draft with the reason it is the fallback.

    The draft is the fallback only when the job's Atlas money cannot cover the refined splat and
    the mesh after it, when Atlas fails every attempt, or when admitted generated geometry would
    be left out of a captured-frames-only request. A splat already bought is always adopted.
    A new purchase sends the views ``_splat_input_selection`` keeps (consistent, with
    parallax, a few held back to score the result).

    A ``rebuild`` replaces a refined splat the job already has, so it never falls back
    to the draft: before buying, it needs World Labs to have accepted pose refinement,
    enough consistent views and the money for the splat and mesh, or it stops by name
    with nothing more bought.
    """
    from .atlas_native_splat import _form_roots, _started, run_native_splat
    from .atlas_pricing import splat_cost
    from .paid_resource_allocator import run_sponsored_website_atlas_task

    root = output_root / "refined_splat"
    bought = any(_started(form_root) for _, form_root in _form_roots(root))
    if admitted_completion and not bought:
        return {"status": "completed", "basis": DRAFT_APPEARANCE,
                "fallback_reason": "atlas_refined_splat_would_omit_admitted_completion"}
    selection = (_splat_input_selection(frames=frames, pose=pose, draft=draft, output_root=output_root,
                                        review=review or {}, source_root=source_root, bought=bought)
                 if source_root is not None else None)
    if rebuild and not bought:
        flags = set((selection or {}).get("flags") or [])
        stop = ("atlas_splat_rebuild_input_selection_missing" if selection is None
                else "atlas_splat_rebuild_pose_refinement_not_accepted"
                if selection.get("pose_refinement_accepted") is not True
                else "atlas_splat_rebuild_consistent_views_insufficient"
                if flags & {"splat_consistent_frames_below_floor", "splat_consistent_frames_below_provider_minimum"}
                else None)
        if stop:
            return {"status": "needs_input", "stage": "refined_splat", "reason": stop}
    if not bought:
        remaining = _current_configuration(task_context, source_digest, configuration).get("remaining_cost_usd")
        needed = splat_cost(policy) + float(policy.get("mesh_export_cost_usd", 0))
        if isinstance(remaining, bool) or not isinstance(remaining, (int, float)) or remaining < needed:
            if rebuild:
                return {"status": "needs_input", "stage": "refined_splat",
                        "reason": "atlas_splat_rebuild_allowance_insufficient"}
            return {"status": "completed", "basis": DRAFT_APPEARANCE,
                    "fallback_reason": "atlas_refined_splat_unaffordable"}
    sent = frames if selection is None else [f for f in frames if f["frame_id"] in set(selection["sent_frame_ids"])]
    check("refined_splat")
    try:
        native = run_native_splat(frames=sent, result=pose, draft=draft, output_root=root, policy=policy,
            task_context=task_context, max_polls=max_polls,
            task_runner=lambda **kw: run_sponsored_website_atlas_task(experimental=experimental, **kw),
            **({"input_selection_digest": selection["digest"]} if selection is not None else {}))
    except ValueError as exc:
        if str(exc) != "atlas_task_operation_failed" or rebuild:
            raise
        return {"status": "completed", "basis": DRAFT_APPEARANCE,
                "fallback_reason": "atlas_refined_splat_provider_failed"}
    if native["status"] != "completed":
        return {"status": "processing", "stage": "refined_splat", "operation_id": native.get("operation_id")}
    appearance = {"status": "completed", "basis": REFINED_APPEARANCE, "splat_root": native["splat_root"],
                  "splat_manifest": native["splat_manifest"]}
    if selection is not None:
        appearance["splat_input"] = {key: selection[key] for key in ("digest", "flags")}
        appearance["splat_input"].update(sent_frame_count=len(selection["sent_frame_ids"]),
            left_out_frame_count=len(selection["left_out"]), heldout_frame_ids=selection["heldout_frame_ids"])
        if selection["heldout_frame_ids"]:
            score = _heldout_quality(frames=frames, selection=selection, splat_root=native["splat_root"])
            appearance["heldout_quality"] = {key: score.get(key) for key in (
                "status", "median_heldout_psnr_db", "median_training_psnr_db", "flags", "reason") if key in score}
    return appearance


def _preserved_scene(*, pose_root, output_root, inputs, review, policy, task_context,
                     source_digest, configuration, check, experimental, max_polls, model,
                     source_survey_path=None, reconstruction_only=False, task_evidence_limitation=None,
                     geometry_only=False, rebuild=False):
    """Reconstruct an environment-preserving task from full-view observations.

    Selection and scale still use the production agents and validators. Missing
    surfaces enter the existing source-first bounded completion/recovery seams.
    Original observations and the first incomplete export remain retained.

    ``geometry_only`` keeps any task's scene as filmed, a captured subject
    included, and buys only the pose, the scale review and the mesh: no
    completion. Its inputs already left out every frame showing a transient.

    ``rebuild`` (``atlas_splat_rebuild``) buys a fresh pose, then the refined
    splat and the mesh, and no completion: the parent's completion admitted no
    generated geometry. It stops by name, with the parent's scene left as it was,
    when its pose or money cannot support the splat.
    """
    from .atlas_source_pose_inputs import prepare_source_pose
    from .atlas_depth_splat import write_depth_splat, load_posed_rgbd, splats2mesh_attempt
    from .atlas_mesh_registration import register_atlas_mesh
    from .atlas_result_bundle import export_pose_bundle
    from .paid_resource_allocator import run_sponsored_website_atlas_task, run_sponsored_website_atlas_automation

    if not geometry_only and ((not reconstruction_only and review.get("subject_origin") not in PRESERVED_SCENE_ORIGINS)
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
    if source_survey_path is not None:
        # The gallery must also hold the task-source views this source pose was given.
        from .atlas_source_gap_evidence import with_task_source_views
        from .atlas_video_inputs import GEOMETRY_INPUTS_FILE
        retained_inputs = source_root / GEOMETRY_INPUTS_FILE
        source_survey_path = with_task_source_views(
            survey_path=source_survey_path,
            source_inputs=json.loads(retained_inputs.read_text()) if retained_inputs.is_file() else {},
            clean_root=pose_root.parent.parent / "clean_plate",
            output=output_root / "capture_source_survey_with_task_views.json")
    if not (source_root / "atlas_posed_rgbd_result.json").is_file():
        return {"status": "needs_input", "stage": "source_pose", "source_pose": source_pose,
                "reason": source_pose.get("reason", "atlas_source_pose_unavailable"),
                "blockers": [source_pose.get("reason", "atlas_source_pose_unavailable")]}
    if rebuild and ((json.loads((source_root / "atlas_posed_rgbd_result.json").read_text()).get("operation") or {})
                    .get("response") or {}).get("poseRefinementAccepted") is not True:
        # A rebuild buys its splat only on a pose World Labs refined (``_preserved_appearance``), so it
        # stops here, before decoding and fusing every posed view for a draft it would not use.
        # capture-5b067498 (2026-11-09): 105 views, refinement not accepted, and the fusion was killed
        # out of memory each tick instead of reaching this stop.
        return {"status": "needs_input", "stage": "source_pose",
                "reason": "atlas_splat_rebuild_pose_refinement_not_accepted"}
    repose = None
    if not (rebuild or geometry_only or reconstruction_only) and "refined_splat_cost_usd" in policy:
        # A new job's one windowed re-pose before its refined splat (``website_atlas_windowed_repose``).
        repose = _windowed_repose(output_root=output_root, first_pose_root=pose_root, source_root=source_root,
            review=review, source_pose=source_pose, policy=policy, task_context=task_context,
            source_digest=source_digest, configuration=configuration, check=check, experimental=experimental,
            max_polls=max_polls)
        if repose is not None and repose.get("status") == "processing":
            return repose
        if repose is not None and repose.get("adopted"):
            pose_root, inputs, review = repose["pose_root"], repose["inputs"], repose["review"]
            source_pose, source_root = repose["source_pose"], repose["source_root"]
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
    elif 'rgbd_view_cost_usd' in policy and not reconstruction_only and not geometry_only and not rebuild:
        if source_survey_path is None:
            completion_stopped = True
            blockers.append('atlas_original_capture_survey_required')
        else:
            def completion_checkpoint(stage, **_details):
                check(stage)
            # Time-spread views add splat density, not viewpoints: completion's
            # photographs, generation context and registration keep the rest.
            plan = pose_root / "atlas_view_selection.json"
            spread = ({row["frame_id"] for row in json.loads(plan.read_text()).get("diagnostics", [])
                       if row.get("decision") == "time_spread_view"} if plan.is_file() else set())
            reference = ({"reference_frame_ids": [f["frame_id"] for f in frames if f["frame_id"] not in spread]}
                         if spread else {})
            completion = run_sponsored_website_atlas_automation(preserved_capture=True, **reference,
                pose_root=source_root, pose=pose, frames=frames, review=review,
                root=output_root / 'preserved_completion', job_root=output_root, context=task_context,
                configuration=_current_configuration(task_context, source_digest, configuration), splat_root=splat_root,
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
    # The scene's appearance is Atlas's refined splat whenever its money allows (owner,
    # 2026-11-07); the depth-fused draft stays the gap and coverage evidence and is the
    # appearance only as a named fallback.
    appearance = None
    if (not reconstruction_only and not geometry_only and source_pose['status'] == 'completed'
            and not completion_stopped and 'refined_splat_cost_usd' in policy):
        appearance = _preserved_appearance(frames=frames, pose=pose, draft=splat, output_root=output_root,
            policy=policy, task_context=task_context, source_digest=source_digest, configuration=configuration,
            check=check, experimental=experimental, max_polls=max_polls, admitted_completion=admitted_completion,
            review=review, source_root=source_root, rebuild=rebuild)
        if appearance['status'] in ('processing', 'needs_input'):
            return appearance
        if appearance['basis'] == REFINED_APPEARANCE:
            splat_root, splat = Path(appearance['splat_root']), appearance['splat_manifest']
    if rebuild and (appearance or {}).get('basis') != REFINED_APPEARANCE:
        # A rebuild only ever replaces the scene with a new refined splat; nothing else is bought.
        return {'status': 'needs_input', 'stage': 'source_pose' if source_pose['status'] != 'completed'
                else 'refined_splat', 'reason': source_pose.get('reason') if source_pose['status'] != 'completed'
                else 'atlas_splat_rebuild_refined_splat_unpriced'}
    if (not reconstruction_only and source_pose['status'] == 'completed' and 'mesh_export_cost_usd' in policy
            and not completion_stopped):
        mesh_destination = (output_root / 'mesh-refined' / splat['digest'][7:23]
                            if (appearance or {}).get('basis') == REFINED_APPEARANCE
                            else output_root / 'preserved_completion/mesh' / splat['digest'][7:23]
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
    if repose is not None:
        # The re-pose's window (its derived review inside) and its outcome travel with the scene.
        analysis += [path for path in (output_root / repose["root"] / name for name in repose["records"])
                     if path.is_file()]
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
    if appearance is not None:
        result['appearance'] = {key: appearance[key] for key in (
            'basis', 'fallback_reason', 'splat_input', 'heldout_quality') if key in appearance}
    if repose is not None:
        result['windowed_repose'] = repose['disclosure']
    if completion is not None:
        result['scene_completion'] = {'status': completion['status'], 'reason': completion.get('reason'),
            'failure_class': completion.get('failure_class'),
            'generated_geometry_admitted': admitted_completion,
            'remaining_unknowns': native_completion.get('remaining_unknowns', []),
            # Generated views made and reviewed but adding nothing: named, with the
            # recovery receipt that shows it, so the scene source can go on unchanged.
            **({'generated_geometry_rejected': completion['generated_geometry_rejected'],
                'geometry_recovery': {key: completion['geometry_recovery'][key]
                                      for key in ('digest', 'exportable_samples', 'added_geometry_pixels')}}
               if completion.get('generated_geometry_rejected') else {}),
            **({'admission_digest': completion['admission']['digest'],
                'admission_relative_path': str(Path(completion['admission_path']).relative_to(output_root))}
               if admitted_completion else {})}
    if geometry_only:
        # A captured subject stays where it was filmed; a robot check reading this must cut it out first.
        result.update(atlas_route="geometry_only",
                      subject_in_reconstruction=review.get("subject_origin") not in PRESERVED_SCENE_ORIGINS)
    if reconstruction_only:
        result.update(observed_scene_only=True, native_preparation_eligible=False, execution_authority_granted=False)
    result["digest"] = canonical_digest(result, digest_field="digest")
    write_json(output_root / "result.json", result)
    return result


def _windowed_repose(*, output_root, first_pose_root, source_root, review, source_pose, policy, task_context,
                     source_digest, configuration, check, experimental, max_polls):
    """A new job's one windowed re-pose before its refined splat, under the term it recorded.

    None when the job has no enabled term, its first pose was refined, or nothing was
    tried. Otherwise ``processing`` while the re-pose runs, or the recorded outcome:
    ``adopted`` with the re-pose's pose root, inputs, derived review, source pose and
    source root, or not adopted (the job goes on with its first pose, exactly as before).
    Every decision is recorded once and read back (``website_atlas_windowed_repose``).
    """
    from . import website_atlas_windowed_repose as reposes
    from .atlas_pricing import admitted_final_outputs, final_output_cost
    from .atlas_source_pose_inputs import prepare_source_pose
    from .atlas_splat_rebuild import WINDOW_FILE
    from .atlas_splat_rebuild_window import first_pose_window
    from .atlas_video_inputs import prepare_video_inputs

    term = reposes.read_term(output_root)
    first = json.loads((source_root / "atlas_posed_rgbd_result.json").read_text())
    if (term is None or not term.get("enabled")
            or ((first.get("operation") or {}).get("response") or {}).get("poseRefinementAccepted") is True):
        return None
    binding = {"first_pose_result_digest": first["digest"], "term_digest": term["digest"]}
    root = output_root / reposes.ROOT
    selection = json.loads((output_root / "selection_binding.json").read_text())
    survey_path = output_root / selection["survey_relative_path"]
    if survey_path.is_symlink() or not survey_path.resolve().is_relative_to(output_root.resolve()):
        raise ValueError("atlas_runtime_selection_escape")
    window = first_pose_window(source_root=source_root, binding=binding, review=review,
                               survey=json.loads(survey_path.read_text()), pose_root=root)
    # The selection profile the first pose recorded: its splat target, sharpness floor and input bound.
    profile = _splat_input_options(first_pose_root, review=review, policy=policy, edit_scope="required_and_additional")

    def finish(outcome, ready=None):
        value = {"root": reposes.ROOT, "records": (WINDOW_FILE, reposes.OUTCOME_FILE),
                 "disclosure": reposes.disclosure(outcome, window), "adopted": outcome["status"] == "adopted"}
        if not value["adopted"]:
            return value
        if ready is None:
            # Read back on a later tick: the retained selection and pose, nothing bought again.
            inputs_ = prepare_video_inputs(survey_path=survey_path, review=window["review"], output_root=root,
                                           **profile)
            ready = (inputs_, _source_pose(pose_root=root, inputs=inputs_, review=window["review"], policy=policy,
                task_context=task_context, check=check, experimental=experimental, max_polls=max_polls,
                configuration=configuration, allow_scale_review=False))
        retry_inputs, retry_pose = ready
        if retry_pose.get("status") != "completed" or retry_pose.get("result_digest") != outcome[
                "retry_pose_result_digest"]:
            raise ValueError("website_atlas_windowed_repose_changed")
        retry_root, _ = prepare_source_pose(root)
        return {**value, "pose_root": root, "inputs": retry_inputs, "review": window["review"],
                "source_pose": retry_pose, "source_root": retry_root}

    recorded = reposes.read_outcome(output_root, binding)
    if recorded is not None:
        return finish(recorded)
    if window["status"] != "windowed":
        return finish(reposes.record_outcome(output_root, binding, "no_window", reason=window.get("reason"),
                                             flags=window["flags"]))
    price = float(policy["pose_input_cost_usd"])
    if price != term["per_view_cost_usd"]:
        return finish(reposes.record_outcome(output_root, binding, "price_changed",
                                             reason="pose_price_differs_from_term"))
    try:
        retry_inputs = prepare_video_inputs(survey_path=survey_path, review=window["review"], output_root=root,
                                            **profile)
        retry_root, prepared = prepare_source_pose(root)
    except ValueError as exc:
        return finish(reposes.record_outcome(output_root, binding, "selection_refused",
                                             reason=str(exc).partition(":")[0]))
    count = len(prepared["frames"])
    first_count = len(first["operation_receipt"]["input_usage"]["frames"])
    if count > term["max_input_images"] or count > first_count:
        return finish(reposes.record_outcome(output_root, binding, "inputs_over_term_bound", input_count=count,
                                             first_pose_input_count=first_count))
    bought = (retry_root / "atlas_posed_rgbd_result.json").is_file() or any(
        (retry_root / "atlas_pose").glob("*.json"))
    if not bought:
        # Money before the paid stage, as for any: the re-pose and the final outputs still to buy.
        remaining = _current_configuration(task_context, source_digest, configuration).get("remaining_cost_usd")
        needed = round(count * price + final_output_cost(policy, admitted_final_outputs(output_root)), 6)
        if isinstance(remaining, bool) or not isinstance(remaining, (int, float)) or remaining < needed:
            return finish(reposes.record_outcome(output_root, binding, "allowance_insufficient",
                                                 reason="remaining_allowance_below_repose_and_final_outputs",
                                                 needed_usd=needed))
    retry_pose = _source_pose(pose_root=root, inputs=retry_inputs, review=window["review"], policy=policy,
        task_context=task_context, check=check, experimental=experimental, max_polls=max_polls,
        configuration=configuration, allow_scale_review=False)
    if retry_pose["status"] == "processing":
        return {**retry_pose, "stage": "windowed_source_pose"}
    posed = retry_root / "atlas_posed_rgbd_result.json"
    if retry_pose["status"] != "completed" or not posed.is_file():
        return finish(reposes.record_outcome(output_root, binding, "repose_unavailable",
                                             reason=retry_pose.get("reason", "atlas_source_pose_unavailable")))
    result = json.loads(posed.read_text())
    details = {"retry_pose_result_digest": result["digest"], "input_count": count,
               "scale_status": retry_pose.get("scale_status"), "first_scale_status": source_pose.get("scale_status")}
    if ((result.get("operation") or {}).get("response") or {}).get("poseRefinementAccepted") is not True:
        return finish(reposes.record_outcome(output_root, binding, "not_accepted",
                                             reason="repose_refinement_not_accepted", **details))
    unscaled = "unscaled_provider_world_units"
    if retry_pose.get("scale_status") == unscaled and source_pose.get("scale_status") != unscaled:
        # No scale review is bought for the re-pose: it is used only where its own measurement scales it.
        return finish(reposes.record_outcome(output_root, binding, "scale_unresolved",
                                             reason="repose_measurement_unscaled", **details))
    return finish(reposes.record_outcome(output_root, binding, "adopted", **details), (retry_inputs, retry_pose))


def _run_splat_rebuild(*, job_dir, task_context, policy, held_policy, configuration, source_digest, check,
                       experimental, max_polls):
    """Advance the operator's reviewed rebuild (``atlas_splat_rebuild``) in its own root.

    A new selection under the current policy, a new source pose, then the refined
    splat and the mesh, each through the job's own allowance and paid admission.
    Returns the result the job publishes: the rebuilt one, or on a stop the parent's.

    A windowed retry first reads the earlier rebuild's pose (``atlas_splat_rebuild_window``):
    its selection draws only from that pose's consistent moving footage, under the review
    the window derives. With no usable window it stops by name before buying anything.
    """
    from . import atlas_splat_rebuild as rebuilds
    from .atlas_video_inputs import prepare_video_inputs

    request, root, parent = rebuilds.load_request(job_dir)
    finished = rebuilds.retained_outcome(job_dir, root, parent)
    if finished is not None:
        return finished
    selection = json.loads((job_dir / "selection_binding.json").read_text())
    survey_path = job_dir / selection["survey_relative_path"]
    if survey_path.is_symlink() or not survey_path.resolve().is_relative_to(job_dir.resolve()):
        raise ValueError("atlas_runtime_selection_escape")
    review, pose_root = selection["review"], root / "pose"
    pose_root.mkdir(parents=True, exist_ok=True)
    window = None
    if request.get("attempt") == rebuilds.WINDOW_ATTEMPT:
        from .atlas_splat_rebuild_window import pose_window
        window = pose_window(job_dir=job_dir, request=request, review=review,
                             survey=json.loads(survey_path.read_text()), pose_root=pose_root)
        if window["status"] != "windowed":
            return rebuilds.record_stop(root, request, parent, {
                "stage": "pose_window", "reason": "atlas_splat_rebuild_pose_window_" + window["status"],
                "flags": window["flags"]})
        review = window["review"]
    # The rebuilt bundle carries the reviewed scene the parent's did (a windowed retry: the one it derived).
    write_json(root / "atlas_scene_review.json", review)
    try:
        try:
            inputs = prepare_video_inputs(survey_path=survey_path, review=review, output_root=pose_root,
                **_splat_input_options(pose_root, review=review, policy=held_policy,
                                       edit_scope="required_and_additional"))
        except ValueError as exc:
            if window is None:
                raise
            # The window's footage cannot carry the reviewed selection: the retry stops, nothing bought.
            return rebuilds.record_stop(root, request, parent, {
                "stage": "pose_window", "reason": "atlas_splat_rebuild_pose_window_selection_refused",
                "flags": [*window["flags"], str(exc).partition(":")[0]]})
        result = _preserved_scene(pose_root=pose_root, output_root=root, inputs=inputs, review=review,
            policy=policy, task_context=task_context, source_digest=source_digest, configuration=configuration,
            check=check, experimental=experimental, max_polls=max_polls, model=parent.get("model"), rebuild=True)
    except ValueError as exc:
        if not str(exc).startswith(rebuilds.STOP_PREFIXES):
            raise
        result = {"status": "needs_input", "reason": str(exc).partition(":")[0]}
    if result.get("status") == "processing":
        return result
    if not result.get("download") or (result.get("appearance") or {}).get("basis") != REFINED_APPEARANCE:
        return rebuilds.record_stop(root, request, parent, result)
    return rebuilds.publish(job_dir, root, request, parent, result, window=window)


def _checkpoint_with_details(checkpoint):
    """Keep stage-only source checks compatible with diagnostic progress details."""
    if checkpoint is None:
        return None
    try:
        signature = inspect.signature(checkpoint)
    except (TypeError, ValueError):
        signature = None  # Opaque callables retain the documented stage-only contract.

    def check(stage, **details):
        if details and signature is not None:
            try:
                signature.bind(stage, **details)
            except TypeError:
                pass
            else:
                return checkpoint(stage, **details)
        # Never catch/retry an exception from the actual source/withdrawal check.
        return checkpoint(stage)

    return check


def _run_atlas_pipeline(*, source_video: Path, output_root: Path, task_context, experimental=False,
                       invoker=None, max_polls=2, refinement_feedback: Path | None = None, checkpoint=None,
                       edit_scope: str | None = None):
    # ``checkpoint(stage)`` runs before every paid stage; raising there (a
    # withdrawal) stops the job before it buys or uploads anything more.
    checkpoint = _checkpoint_with_details(checkpoint)
    check = checkpoint or (lambda _stage, **_details: None)
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
    # Before anything new is admitted, report every closed Atlas attempt's outcome so a
    # failed or refused hold World Labs never charged stops counting against the job.
    from .atlas_spend_settlement import settle_closed_atlas_attempts_under
    settle_closed_atlas_attempts_under(output_root, task_context=task_context)
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
    route = _job_route(output_root, purchased=reviews.exists() or selection_path.exists())
    held = _route_policy(policy, route)
    default_scope = ("all_selected_views" if "refined_splat_cost_usd" in held
                     and not selection_path.exists() and not reviews.exists() else "required_and_additional")
    edit_scope = _job_edit_scope(output_root, edit_scope, default=default_scope)
    os.environ[ATLAS_PROVIDER_RETENTION_ENV] = current["authority"]["provider_retention"]
    from .atlas_splat_rebuild import REQUEST_FILE as REBUILD_REQUEST
    if refinement_feedback is None and (output_root / REBUILD_REQUEST).is_file():
        # An operator's reviewed rebuild of this ready job's refined splat (atlas_splat_rebuild).
        return _run_splat_rebuild(job_dir=output_root, task_context=task_context, policy=policy, held_policy=held,
            configuration=_current_configuration(task_context, source_digest, config), source_digest=source_digest,
            check=check, experimental=experimental, max_polls=max_polls)
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
              - final_output_cost(held, admitted_final_outputs(output_root)))
    # A new job's term for one windowed re-pose before its refined splat (owner 2026-11-09): decided
    # once before its first purchase, and its maximum held back from the selection. A job that bought
    # before this release has none and keeps the quote it was admitted under.
    from .website_atlas_windowed_repose import decide_term, held_back_usd
    repose_term = decide_term(output_root, policy=held, purchased=reviews.exists() or selection_path.exists(),
                              available_usd=budget,
                              minimum_usd=2 * policy["pose_input_cost_usd"] + render_cost(policy, 2))
    budget -= held_back_usd(repose_term)
    # Size a new selection only; a retained one already spent its reviews, and
    # holding them back again would stop a resumed job on money it already paid.
    if not selection_path.exists() and budget < 2 * policy["pose_input_cost_usd"] + render_cost(policy, 2):
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
    inputs = prepare_video_inputs(survey_path=survey_path, review=review, output_root=pose_root,
        exclude_transients=route == "geometry_only",
        **_splat_input_options(pose_root, review=review, policy=held, edit_scope=edit_scope))
    if not selection_path.exists():
        write_json(selection_path, {"survey_relative_path": str(survey_path.relative_to(output_root)),
                                   "review": review, "planner_receipt": receipt})
    from .website_task_semantics import retain_task_semantics
    retain_task_semantics(job_dir=output_root, receipt=receipt, task_context=task_context,
                          source_video_digest=source_digest)
    write_json(output_root / "atlas_scene_review.json", review)
    if feedback is None:
        # A task with no native profile, or no installed runtime, holds here, before any
        # World Labs stage: the review has read the whole job (website_early_task_hold).
        from .website_early_task_hold import hold_after_review
        held = hold_after_review(output_root=output_root, task_context=task_context,
                                 source_video_digest=source_digest)
        if held is not None:
            return held
    if route == "geometry_only":
        # The scene stays as filmed, subject included: nothing is removed or edited,
        # and no generated completion or refined splat is bought.
        return _preserved_scene(pose_root=pose_root, output_root=output_root, inputs=inputs, review=review,
            policy=policy, task_context=task_context, source_digest=source_digest,
            configuration=_current_configuration(task_context, source_digest, config), check=check,
            experimental=experimental, max_polls=max_polls, model=receipt["model"], geometry_only=True)
    if review.get("subject_origin") in PRESERVED_SCENE_ORIGINS:
        from .atlas_source_gap_evidence import combine_source_surveys
        source_survey = combine_source_surveys(survey_paths=[
            (Path(feedback['parent_root']) / 'survey/atlas_video_survey.json'
             if feedback is not None else base_root / 'atlas_video_survey.json'), survey_path],
            output=output_root / 'capture_source_survey.json')
        return _preserved_scene(pose_root=pose_root, output_root=output_root, inputs=inputs, review=review,
            policy=policy, task_context=task_context, source_digest=source_digest,
            configuration=_current_configuration(task_context, source_digest, config), check=check,
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
            task_context=task_context, configuration=_current_configuration(task_context, source_digest, config),
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
    planned_cost = round(count * policy["pose_input_cost_usd"] + render_cost(policy, target_count), 6)
    pose_bought = ((pose_root / "atlas_pose/submission.json").exists()
                   or (pose_root / "atlas_posed_rgbd_result.json").exists())
    if not pose_bought and planned_cost > budget + 1e-9:
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
    cost = render_cost(policy, len(request["targetCameras"]))
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
            configuration=_current_configuration(task_context, source_digest, config), experimental=experimental,
            max_polls=max_polls,
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
              "splat_available": complete,
              "splat_kind": background.get("splat_kind", "depth_fused_draft") if complete else None,
              "splat_manifest_digest": background.get("splat_manifest_digest"),
              "mesh_available": background.get("mesh_available", False),
              "scene_measurements": automation.get("measurements"),
              "source_pose": source_pose,
              "simulation_qualified": False, "claim_ceiling": "development_only",
              "physical_evidence": False, "scale_status": "unscaled_provider_world_units",
              "billing_status": "provider_usage_unreported"}
    if complete and background.get("refined_splat"):
        result["refined_splat"] = background["refined_splat"]
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
