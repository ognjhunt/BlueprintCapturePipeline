"""Website capture derivatives into the existing construction queue, ADP-030/day 28.

This CPU builder preserves authenticated intent, estimated geometry and separate
object provenance. It never reconstructs, removes an object again, or allocates.
"""
from __future__ import annotations

import math
from pathlib import Path

from . import task_evaluation_scene_configuration_submission_records as records
from .decision_evidence_contracts import canonical_digest
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from .task_evaluation_completed_scene_transaction import completed_submission_transaction
from .task_evaluation_launch_preparation_contract import (
    launch_preparation_request_digest, validate_launch_preparation_request,
)
from .task_evaluation_scene_configuration_submission_inputs import (
    Staging, checked_file, read, release_inputs, require, sha, slug,
)
from .website_native_background import prepare_construction_stages, construction_rights_admission, PREFIX


def verified_submission_inputs(task, *, now=None):
    from .task_evaluation_scene_owner_authority import reopen_scene_intent
    intent = reopen_scene_intent(task["scene_intent_authority"], now=now)
    paths = {key: checked_file(task[key]["path"], task[key])
             for key in ("preparation", "runtime_inputs", "task_context")}
    preparation = read(paths["preparation"], digest_field="digest")
    context = read(paths["task_context"], digest_field="context_digest")
    require(preparation["intake_request"] == intent["request"], "website_intent_preparation_mismatch")
    # Consent was valid when the authenticated intent was accepted. Current
    # expiry and revocation were independently checked by reopen_scene_intent.
    rights = construction_rights_admission(preparation=preparation, task_context=context,
                                          now=intent["accepted_at_epoch"])
    construction = prepare_construction_stages(runtime_inputs_path=paths["runtime_inputs"],
                                               preparation_path=paths["preparation"])
    return intent, preparation, context, rights, construction, paths


def validate_website_publication(*, root, manifest, request):
    """Reopen real owner authority before publishing the exact derivative set."""
    from .task_evaluation_launch_preparation_worker import collect_preparation_references
    from .task_evaluation_scene_configuration_submission_publication import _path
    from .task_evaluation_scene_configuration_stage_configuration import validate_immutable_stage_configurations
    from .website_native_inputs import validate_website_native_inputs

    task = read(_path(root, "provenance/website_task_request.v1.json"))
    intent, preparation, context, _, _, paths = verified_submission_inputs(task)
    require(request.get("scene_intent_digest") == intent["intent_digest"]
            and request["task"]["identity"]["id"] == intent["request"]["task"]["task_id"]
            and request["scene"]["source_manifest"]["digest"] == sha(paths["preparation"])
            and read(_path(root, "website/task_context.json")) == context,
            "website_publication_authority_mismatch")
    by_uri = {row["uri"]: row for row in manifest["files"]}
    def materialized(ref):
        row = by_uri[ref["uri"]]
        require(all(row[key] == ref[key] for key in ("digest", "size_bytes")), "website_publication_reference_changed")
        return _path(root, row["relative_path"])
    recipe = read(materialized(request["construction"]["recipe"]), digest_field="recipe_digest")
    refs = collect_preparation_references(request)
    configurations = {stage["stage_id"]: read(materialized(stage["configuration"])) for stage in recipe["stage_sequence"]}
    envelope = {"request": request, "recipe": recipe, "materialized_references": [
        {**ref, "materialized_path": str(materialized(ref)), "full_byte_service_account_readback_passed": True} for ref in refs]}
    validate_immutable_stage_configurations(envelope=envelope, configurations=configurations)
    validate_website_native_inputs(envelope=envelope, configurations=configurations, require_render_inputs=False)
    allowed = {by_uri[ref["uri"]]["relative_path"] for ref in refs}
    allowed.update(by_uri[stage["configuration"]["uri"]]["relative_path"] for stage in recipe["stage_sequence"])
    allowed.update({"provenance/website_task_request.v1.json", "website/task_context.json",
                    "scene_configuration_preparation_request.v1.json"})
    require({row["relative_path"] for row in manifest["files"]} == allowed,
            "website_publication_unreferenced_bytes")


@completed_submission_transaction(task_relative_path="provenance/website_task_request.v1.json")
def materialize_website_submission(*, task, deploy_receipt_path, release_provenance_path,
                                  release_environment_path, runtime_publication_root,
                                  expected_production_commit, namespace_timestamp,
                                  release_admission_mode, staging_root):
    intent, preparation, context, rights, construction, paths = verified_submission_inputs(task)
    owner_task = intent["request"]["task"]
    from .website_development_test import environment, LABEL
    development = environment(preparation)
    scene_id = construction["scene_identity"]["id"]
    commit = expected_production_commit
    deploy, toolchain, renderer = release_inputs(deploy_path=Path(deploy_receipt_path),
        provenance_path=Path(release_provenance_path), publication_root=Path(runtime_publication_root),
        commit=commit, release_admission_mode=release_admission_mode)
    team = "website-" + intent["intent_digest"][7:31]
    namespace = slug(f"{team}-{commit}-{namespace_timestamp.lower()}")
    run_id = slug(f"{team}-{commit[:8]}-{namespace_timestamp.lower()}")
    stage = Staging(Path(staging_root), namespace)
    manifest_ref = stage.copy(paths["preparation"], "website/preparation.json")
    stage.copy(paths["task_context"], "website/task_context.json")
    stage.json("provenance/website_task_request.v1.json", task)
    rights_ref = stage.json("rights/admission.json", rights)
    human_ref = stage.json("rights/owner_consent.json", {
        "schema_version": "website_scene_owner_authority.v1", "owner": intent["request"]["owner"],
        "consent": intent["request"]["consent"], "intent_digest": intent["intent_digest"],
        "task_context_digest": context["context_digest"], "physical_metrology_claimed": False})
    terms_ref = stage.json("rights/terms.json", {
        "schema_version": "website_scene_processing_terms.v1",
        "rights_reference": rights["consent"]["rights_reference"],
        "provider_terms_reference": rights["consent"]["provider_terms_reference"],
        "public_redistribution_allowed": False, "provider_training_allowed": False})
    refs = {}
    for index, row in enumerate(construction["references"]):
        path = checked_file(row["path"], {"sha256": row["digest"], "size_bytes": row["size_bytes"]})
        refs[row["contract_path"]] = stage.copy(path, f"website/inputs/{index:03d}{path.suffix}")
    runtime = read(paths["runtime_inputs"], digest_field="digest")
    normalization_ref = stage.copy(Path(runtime["collision"]["normalization_path"]), "geometry/normalization.json")
    website = {key: refs[PREFIX + "." + key] for key in ("runtime_inputs", "appearance", "observations", "candidate")}
    website["frames"] = [refs[PREFIX + f".frames.{i}"] for i in range(
        sum(key.startswith(PREFIX + ".frames.") for key in refs))]
    subject = runtime["subject"]
    lower, upper = subject["aabb_min_xyz"], subject["aabb_max_xyz"]
    support = owner_task["support"]
    support_record = records.support_plane_input(scene_id=scene_id, instance_id="website-support",
        semantic_label=support["description"], sage_prim_path="/Root",
        bounds_min=support["aabb_min_xyz"], bounds_max=support["aabb_max_xyz"])
    support_record.update(authority="authored_development_surface" if development else "registered_estimated_capture_and_reconstruction",
                          physical_scale_measured=False, source_face_indices=preparation["support"]["face_indices"])
    from .task_evaluation_surface_target import derive_surface_target, surface_execution_limits
    destination = owner_task["destination"]
    require(destination.get("mode") == "existing_support_surface", "website_destination_asset_required")
    target = derive_surface_target(destination={**destination, "kind": "green_region",
        "radius_m": math.hypot(*[(upper[i] - lower[i]) / 2 for i in (0, 1)]) + 0.01},
        support=support_record, source_min=lower, source_max=upper, support_instance_id="website-support")
    position = destination["position_world_m"]
    task_identity = {"id": owner_task["task_id"], "version": "v1"}
    template, success, execution = records.pick_and_place_task_records(
        task_identity=task_identity, object_identity=construction["subject_identity"],
        start_center=[(a + b) / 2 for a, b in zip(lower, upper, strict=True)],
        target_center=[position[0], position[1], position[2] + (upper[2] - lower[2]) / 2],
        source_min=lower, source_max=upper, grasp_axis=2, grasp_sign=1.0,
        success=surface_execution_limits(success=owner_task["success"], support=support_record),
        resolved_seed=1, jaw_axis=min(range(2), key=lambda i: upper[i] - lower[i]))
    # The estimated property ranges, the grasp-hold sensitivity verdict and the
    # registration's task-region residual travel with the task, so a result can
    # abstain from a feasibility claim the estimate cannot support.
    physics = preparation["physics"]
    template.update(instruction=(LABEL + " " if development else "") + context["description"], instruction_subject_label=subject["description"],
                    visible_target_label=destination["visible_label"], surface_target=target,
                    dimension_authority="estimated", physical_world_truth_claimed=False,
                    physical_property_screen={
                        "basis": physics["basis"], "dimensions_m": physics["dimensions_m"],
                        "bounds": physics["bounds"], "sensitivity": physics["sensitivity"],
                        "measurement_escalation": physics.get("measurement_escalation"),
                        "reference_gripper": physics.get("gripper"),
                        "feasibility_claim_allowed": physics["sensitivity"] == "robust_within_range"},
                    placement_uncertainty_m=preparation["coordinate_frame"].get("placement_uncertainty_m"),
                    scale_authority=preparation["coordinate_frame"].get("scale_authority", "registration_estimate"))
    if development:
        template["test_environment"] = development
    template["owner_success_contract_authority"] = {"confirmation_status": "confirmed",
        "accepted_by": intent["request"]["owner"]["user_id"],
        "authority_reference": "scene-intent:" + intent["intent_digest"]}
    template["success"]["surface_target"] = target
    success["surface_target"] = target
    selection_ref = stage.json("configuration/subject.json", {
        "schema_version": "task_evaluation_source_object_selection.v1", "scene_id": scene_id,
        "source_object_id": construction["configurations"][2]["source_object_identity"],
        "review_label": subject["description"], "geometry_origin": "removed_before_reconstruction",
        "aabb_min_xyz_m": lower, "aabb_max_xyz_m": upper,
        "complete_object_geometry": False, "source_object_is_physics_authority": False})
    def plan(name, **fields):
        return stage.json(f"configuration/{name}.json", {"schema_version": f"website_{name}.v1",
            "status": "execute_during_scene_configuration_run", "scene_id": scene_id,
            "physical_metrology_claimed": False, **fields})
    renderer_ref = plan("renderer_qualification", appearance_qualified=False, browser_preview_qualifies=False)
    metric_ref = plan("metric_registration", **preparation["coordinate_frame"],
                      registration_authority="registered_model_estimates")
    support_ref = stage.json("configuration/support.json", support_record)
    robot_ref = plan("robot_mount_interface", supported_robot_classes=["fixed_arm"], robot_qualified=False)
    workspace_ref = plan("workspace_clearance", workspace_clearance_qualified=False)
    camera_ref = stage.json("configuration/camera.json",
        records.camera_calibration_plan(scene_id=scene_id, strategy="pick_and_place"))
    config_refs = [stage.json(f"configuration/stage_{i + 1}.json", value)
                   for i, value in enumerate(construction["configurations"])]
    output_identity = {"id": scene_id + "-configured", "version": intent["intent_digest"][7:19]}
    recipe = records.recipe(recipe_id=run_id + "-recipe", team_namespace=team,
        scene_identity=construction["scene_identity"], task_identity=task_identity,
        subject_identity=construction["subject_identity"], output_identity=output_identity,
        source_manifest_digest=manifest_ref["digest"], rights_admission_digest=rights_ref["digest"],
        stage_configuration_references=config_refs, supplemental_destination=None)
    for original, configured in zip(recipe["stage_sequence"], construction["stage_sequence"], strict=True):
        original.update(configured)
    recipe["recipe_digest"] = canonical_digest(recipe, digest_field="recipe_digest")
    recipe_ref = stage.json("configuration/recipe.json", recipe)
    release_ref = stage.json("release/binding.json", records.exact_production_release_binding(
        team_namespace=team, scene_identity=construction["scene_identity"], source_commit=commit,
        deploy_receipt=deploy, deploy_receipt_sha256=sha(Path(deploy_receipt_path)),
        release_environment_sha256=sha(Path(release_environment_path)),
        scene_configuration_publication=toolchain, splat_render_publication=renderer,
        release_admission_mode=release_admission_mode))
    health_ref = stage.json("runtime/health.json", records.runtime_health_protocol(source_commit=commit))
    request = {"schema_version": "task_evaluation_launch_preparation_request.v1", "run_mode": "scene_configuration",
        "expected_production_commit": commit, "preparation_id": run_id + "-preparation",
        "team_namespace": team, "run_id": run_id, "scene_intent_digest": intent["intent_digest"],
        "scene": {"mode": "configure_source_scene", "identity": construction["scene_identity"],
            "source_manifest": manifest_ref, "website_native_inputs": website,
            "appearance": {"kind": "textured_usd" if development else "gaussian_splat", "representation": website["appearance"],
                           "renderer_qualification": renderer_ref},
            "geometry": {"kind": "other_derived", "collision": refs["scene.geometry.collision"], "validation": normalization_ref},
            "registration": {"metric_registration": metric_ref, "support_plane": support_ref,
                "robot_mount_interface": robot_ref, "workspace_clearance": workspace_ref, "camera_calibration": camera_ref},
            "rights": {"admission": rights_ref, "evidence": [
                {"role": "publisher_terms", "artifact": terms_ref}, {"role": "human_authority_record", "artifact": human_ref}],
                "source_bytes_redistributable": False, "provider_disclosure_scope": "derived_only"}},
        "construction": {"mode": "production_recipe", "recipe": recipe_ref, "output_identity": output_identity},
        "task": {"identity": task_identity, "binding_mode": "define_configuration_template",
            "kind": "rigid_relocation", "strategy": "pick_and_place", "surface_target": target,
            "subject": {"mode": "construct_from_scene_object", "identity": construction["subject_identity"],
                "representation_kind": "simready_usd", "source_object": selection_ref,
                "rights_admission": rights_ref, "provider_disclosure_allowed": True},
            "definition": stage.json("configuration/task.json", template),
            "success_criteria": stage.json("configuration/success.json", success),
            "execution": stage.json("configuration/execution.json", execution)},
        "sensors": {"configuration": camera_ref},
        "runtime": {"identity": {"id": "task-evaluation-scene-configuration-provider", "version": commit[:8]},
            "oci_image": NATIVE_TASK_ARENA_IMAGE, "entrypoint": ["/opt/blueprint/run-task-evaluation-scene-configuration"],
            "health_protocol": health_ref, "requirements": {"cpu_cores": 8, "memory_gib": 32, "gpu_count": 1, "disk_gib": 64},
            "network": {"default": "deny", "allowlist": ["api.openai.com"]}, "secret_refs": ["secret-file:openai_api_key"],
            "mounts": [{"source": release_ref, "container_path": "/inputs/release-binding.json", "mode": "read_only"},
                       {"container_path": "/outputs", "mode": "output"}], "output_limit_bytes": 20_000_000_000},
        "execution_adapter": {"kind": "scene_configuration_pipeline", "version": "v1", "runtime_source_bundle": release_ref},
        "publication": {"input_namespace": namespace, "service_account_readback_required": True}, "spend": records.spend_block(construction["configurations"][2]["authoring_backend"])}
    request["replacement_authoring_backend"] = construction["configurations"][2]["authoring_backend"]
    require(request["spend"]["hard_cap_usd"] <= intent["request"]["execution"]["max_total_spend_usd"],
            "website_native_construction_budget_exceeds_authority")
    validate_launch_preparation_request(request)
    from .website_native_inputs import validate_website_native_inputs
    envelope = {"request": request, "recipe": recipe, "materialized_references": stage.reference_rows(request)}
    configurations = {f"stage-{i + 1}": value for i, value in enumerate(construction["configurations"])}
    from .task_evaluation_scene_configuration_stage_configuration import validate_immutable_stage_configurations
    validate_immutable_stage_configurations(envelope=envelope, configurations=configurations)
    validate_website_native_inputs(envelope=envelope, configurations=configurations, require_render_inputs=False)
    stage.json("scene_configuration_preparation_request.v1.json", request)
    inventory = {"schema_version": "task_evaluation_scene_configuration_submission_manifest.v1",
        "status": "validated_pending_production_publication_and_submission", "source_commit": commit,
        "input_namespace": namespace, "release_admission_mode": release_admission_mode, "claim_ceiling": "development_only",
        "source": "website_capture_derivatives", "request_digest": launch_preparation_request_digest(request),
        "files": list(stage.files.values()), "raw_source_upload_allowed": False, "provider_allocated": False,
        "native_qualification_claimed": False, "physical_metrology_claimed": False}
    inventory["manifest_digest"] = canonical_digest(inventory, digest_field="manifest_digest")
    stage.json("bundle_manifest.v1.json", inventory)
    return {"staging_root": str(stage.root), "input_namespace": namespace, "status": inventory["status"],
            "request_digest": inventory["request_digest"], "manifest_digest": inventory["manifest_digest"]}
