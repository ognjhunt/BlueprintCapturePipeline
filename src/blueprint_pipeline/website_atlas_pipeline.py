"""Production Atlas preparation controller for one website capture.

Source review, adaptive inputs, admitted posing, generated RGBD and portable
downloads run on the worker. The public API cannot create a splat, so the worker
fuses Atlas's own posed depth into a draft splat and meshes it with splats2Mesh
once a reviewed mesh price exists; neither becomes a metric or simulator-ready result.
Website uploads reach it only through the side-lane worker in
``website_atlas_preview``; the main preparation never waits on it.
"""
from __future__ import annotations

import argparse
import fcntl
from copy import deepcopy
import json
import os
from pathlib import Path
import time

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file


def load_configuration(context, source_digest):
    from .website_task_context import website_webapp_request
    value = website_webapp_request(capture_id=context["capture_id"], operation="atlas-configuration", payload={
        "request_id": context["request_id"], "scene_id": context["scene_id"],
        "task_context_digest": context["context_digest"], "source_video_digest": source_digest})
    if (value.get("schema_version") != "website_atlas_configuration.v1"
            or value.get("source_video_digest") != source_digest
            or value.get("task_context_digest") != context["context_digest"]
            or value.get("dispatch_authorized") is not False):
        raise ValueError("atlas_runtime_configuration_invalid")
    return value


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


def run_atlas_pipeline(*, output_root: Path, **options):
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_atlas_pipeline(output_root=output_root, **options)


def _run_atlas_pipeline(*, source_video: Path, output_root: Path, task_context, experimental=False,
                       invoker=None, max_polls=2, refinement_feedback: Path | None = None, checkpoint=None):
    # ``checkpoint(stage)`` runs before every paid stage; raising there (a
    # withdrawal) stops the job before it buys or uploads anything more.
    check = checkpoint or (lambda _stage: None)
    from .atlas_video_inputs import survey_video, prepare_video_inputs
    from .atlas_scene_planner import proposal_review
    from .paid_resource_allocator import (run_sponsored_website_atlas_plan, run_sponsored_website_atlas_pose,
                                          run_sponsored_website_atlas_task)
    from .website_atlas_pose import ATLAS_PRICE_REVIEW_ENV, ATLAS_PROVIDER_RETENTION_ENV, request_options, atlas_api_config
    from .atlas_result_bundle import export_pose_bundle
    from .atlas_depth_splat import build_depth_splat
    from .atlas_mesh_registration import register_atlas_mesh

    output_root.mkdir(parents=True, exist_ok=True)
    source_digest = _sha256_file(source_video)
    current = load_configuration(task_context, source_digest)
    atlas_api_config()  # Refuse missing/unsafe production secrets before buying a review.
    saved = output_root / "configuration.json"
    if saved.exists():
        config = json.loads(saved.read_text())
        # A resumed job may gain a reviewed price for a stage it has not bought;
        # any changed price or authority still refuses.
        if (config["authority"] != current["authority"]
                or any(current["policy"].get(key) != value for key, value in config["policy"].items())):
            raise ValueError("atlas_runtime_authority_or_policy_changed")
        if config["policy"] != current["policy"]:
            config = {**config, "policy": current["policy"]}
            write_json(saved, config)
    else:
        config = current
        write_json(saved, config)
    policy = config["policy"]
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
    reviews = output_root / "scene_planner"
    # Reserve room for a second review, depth views and the mesh export. These are
    # dollar allowances, not arbitrary limits on the number of source images.
    budget = (config["remaining_cost_usd"] - 2 * policy["maximum_review_cost_usd"]
              - policy.get("mesh_export_cost_usd", 0))
    if budget < 2 * policy["pose_input_cost_usd"] + 2 * policy["rgbd_view_cost_usd"]:
        raise ValueError("atlas_runtime_budget_insufficient")
    selection_path = output_root / "selection_binding.json"
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
        plan_options = {"feedback": feedback} if feedback is not None else {}
        receipt = run_sponsored_website_atlas_plan(survey_path=survey_path, task_context=task_context,
            configuration=config, output_root=reviews, invoker=invoker, experimental=experimental, **plan_options)
        review = proposal_review(receipt["proposal"], survey=survey, policy=policy, pose_budget=budget, feedback=feedback)
        if feedback is None:
            # Every frame of the reviewed still interval becomes a candidate; the
            # denser survey is reviewed again before anything is selected or sent.
            survey_path = accuracy_survey(source_video=source_video, base_root=base_root, base_survey=survey_path,
                                          review=review, output_root=output_root)
            survey = json.loads(survey_path.read_text())
            check("plan")
            receipt = run_sponsored_website_atlas_plan(survey_path=survey_path, task_context=task_context,
                configuration=config, output_root=reviews, invoker=invoker, experimental=experimental)
            review = proposal_review(receipt["proposal"], survey=survey, policy=policy, pose_budget=budget)
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
    write_json(output_root / "atlas_scene_review.json", review)
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
        spec = json.loads((splat_root / "splats2mesh_task_spec.json").read_text())
        mesh_quote = {"schema_version": "website_atlas_task_price_review.v1", "task": "splats2Mesh",
            "request_digest": canonical_digest(spec["request"]), "maximum_cost_usd": policy["mesh_export_cost_usd"],
            **{k: policy[k] for k in ("rate_card_reference", "reviewed_by", "reviewed_on")}}
        mesh_root = output_root / "mesh"
        check("mesh")
        mesh = run_sponsored_website_atlas_task(task="splats2Mesh", request=spec["request"], uploads=spec["uploads"],
            source_video_digest=source_digest, output_root=mesh_root, task_context=task_context,
            price_review=mesh_quote, experimental=experimental, max_polls=max_polls)
        if mesh["status"] != "completed":
            return {"status": "processing", "stage": "mesh", "operation_id": mesh.get("operation_id")}
        if not (mesh_root / "registered" / "atlas_mesh_registration.json").exists():
            outputs = json.loads((mesh_root / "result.json").read_text())["outputs"]
            register_atlas_mesh(glb=mesh_root / next(row["path"] for row in outputs if row["role"] == "mesh"),
                                splat_root=splat_root, output_root=mesh_root / "registered")
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
    if feedback is not None:
        result["refinement"] = {"parent_result_digest": feedback["parent_result_digest"],
            "feedback_digest": feedback["digest"], "gap_assessment": review["gap_assessment"],
            "prior_input_count": len(feedback["prior_selected_frame_ids"]), "new_input_count": count}
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
