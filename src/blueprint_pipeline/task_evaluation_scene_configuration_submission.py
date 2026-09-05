"""Assemble one exact-release scene-construction request; never allocate or upload.

Native qualification, appearance review, robot controls and policy execution are
downstream stages. This assembler validates their inputs, not their outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from . import task_evaluation_scene_configuration_submission_records as records
from .decision_evidence_contracts import canonical_digest
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from .task_evaluation_launch_preparation_contract import (
    launch_preparation_request_digest,
    validate_launch_preparation_request,
)
from .task_evaluation_passive_destination_placement_proposal import (
    derive_passive_destination_placement_proposal,
)
from .task_evaluation_scene_configuration_disclosure import resolve_scene_configuration_disclosure
from .task_evaluation_scene_configuration_source_preflight import (
    validate_scene_configuration_source_bindings,
)
from .task_evaluation_scene_configuration_sam31_plan import (
    PROFILE_ENV, build_sam31_preparation_plan,
)
from .task_evaluation_scene_configuration_stage_configuration import (
    SAM31_MASK_SOURCE, SAM31_SELECTION_RULE,
    _bounded_pair,
    validate_immutable_stage_configurations,
)
from .task_evaluation_scene_configuration_submission_inputs import (
    SceneConfigurationSubmissionError as SceneConfigurationSubmissionError,
    Staging,
    checked_file,
    read,
    release_inputs,
    require,
    sha,
    slug,
    source_inputs,
)
from .task_evaluation_scene_construction_recipe import validate_scene_construction_recipe


def _validate_task(task: dict[str, Any]) -> None:
    require(task.get("schema_version") == "task_evaluation_minimal_task_request.v1"
            and task.get("strategy") == "pick_and_place", "task_request_invalid")
    for field in ("subject", "support", "destination", "success", "human_authority"):
        require(isinstance(task.get(field), dict), "task_request_invalid")
    for identity in (task.get("scene_identity"), task.get("task_identity"),
                     task.get("output_identity"), task["subject"].get("replacement_identity")):
        require(isinstance(identity, dict) and set(identity) == {"id", "version"},
                "identity_invalid")
        slug(identity["id"])
        slug(identity["version"])
    require(task["destination"].get("relation") == "inside", "task_request_invalid")
    physics = task["subject"].get("physics_bounds")
    require(isinstance(physics, dict), "task_subject_physics_bounds_invalid")
    for field in ("mass_kg_bounds", "static_friction_bounds", "dynamic_friction_bounds",
                  "restitution_bounds"):
        require(_bounded_pair(physics.get(field), positive_lower=field == "mass_kg_bounds"),
                "task_subject_physics_bounds_invalid")
    require(float(physics["dynamic_friction_bounds"][0]) <=
            float(physics["static_friction_bounds"][1]),
            "task_subject_friction_bounds_infeasible")
    for obj in ("subject", "support"):
        require(bool(str(task[obj].get("source_instance_id") or "").strip()), "task_object_missing")
    for obj, field in (("subject", "review_label"), ("subject", "authoring_target"),
                       ("destination", "visible_label")):
        require(isinstance(task[obj].get(field), str) and bool(task[obj][field].strip()),
                "task_object_missing")
    for field in ("accepted_by", "accepted_on", "authority_reference"):
        require(isinstance(task["human_authority"].get(field), str) and
                bool(task["human_authority"][field].strip()), "human_authority_missing")
    authority = task["human_authority"]
    for field in ("private_derived_frame_disclosure_authorized", "provider_retention_terms_accepted",
                  "provider_training_terms_accepted"):
        require(authority.get(field) is True, "provider_authority_missing")
    require(authority.get("provider_training_authorized") is False,
            "provider_training_forbidden")
    require(task.get("appearance_removal_method") in {"registered_source_bounds", "sam31"},
            "requested_appearance_method_not_implemented")
    for field in ("control_frequency_hz", "maximum_episode_seconds", "minimum_lift_m",
                  "pregrasp_clearance_m", "minimum_planar_displacement_m",
                  "maximum_final_planar_target_error_m"):
        value = task["success"].get(field)
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and
                math.isfinite(value) and value > 0, "task_success_bounds_invalid")
    for field in ("retreat_clearance_m", "drop_minimum_fall_m", "maximum_task_contact_force_n",
                  "collision_failure_minimum_force_n"):
        if field in task["success"]:
            value = task["success"][field]
            require(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value > 0, "task_success_bounds_invalid")
    if "forbidden_contact_classes" in task["success"]:
        classes = task["success"]["forbidden_contact_classes"]
        require(isinstance(classes, list) and bool(classes) and
                all(isinstance(value, str) and bool(value.strip()) for value in classes),
                "task_success_contact_classes_invalid")
    for field in ("maximum_retries", "maximum_regrasps"):
        value = task["success"].get(field)
        require(type(value) is int and value == 0, "retry_contract_amendment_required")
    require(type(task.get("resolved_seed", 1)) is int and task.get("resolved_seed", 1) > 0,
            "resolved_seed_invalid")


def _destination(path: Path, subject_min: list, subject_max: list) -> tuple[dict, dict, dict]:
    result = read(path, digest_field="result_digest")
    require(result.get("schema_version") == "task_evaluation_passive_destination_simready.v1"
            and result.get("status") == "static_qualified_pending_native_import_and_placement",
            "destination_result_invalid")
    paths = {key: checked_file(Path(result[key]["path"]), result[key])
             for key in ("asset", "authoring_receipt", "static_qualification", "rights_admission")}
    static = read(paths["static_qualification"], digest_field="result_digest")
    authoring = read(paths["authoring_receipt"], digest_field="result_digest")
    rights = read(paths["rights_admission"], digest_field="rights_admission_digest")
    identity = result["destination_identity"]
    require(static.get("replacement_identity") == identity and
            authoring.get("replacement_identity") == identity and
            rights.get("destination_identity") == identity and
            rights.get("status") == "admitted" and
            rights.get("private_provider_processing_allowed") is True and
            rights.get("provider_training_allowed") is False and
            rights.get("public_redistribution_allowed") is False, "destination_identity_or_rights_invalid")
    require(result.get("static_result_digest") == static["result_digest"] and
            static.get("authored_structure_statically_qualified") is True and
            not static.get("structural_findings") and
            all(static.get("replacement_usd", {}).get(k) == result["asset"][k] and
                authoring.get("output_usd", {}).get(k) == result["asset"][k]
                for k in ("sha256", "size_bytes")), "destination_asset_binding_mismatch")
    bounds = result["interior_bounds_body_frame_m"]
    interior = [float(bounds["maximum"][i]) - float(bounds["minimum"][i]) for i in range(3)]
    subject = [float(subject_max[i]) - float(subject_min[i]) for i in range(3)]
    # The proposal aligns the tray's long axis with the support. Only admit the
    # source-aligned subject orientation that this template actually authors.
    require(all(math.isfinite(x) and x > 0 for x in interior + subject) and
            all(subject[i] <= interior[i] for i in range(3)),
            "destination_cannot_contain_subject")
    structure = static["observed_structure"]
    support = result.get("intended_support_prim_paths", [])
    colliders = result.get("intended_support_collision_prim_paths", [])
    require(len(support) == 1 and support[0] in structure["rigid_body_paths"] and
            len(colliders) == 1 and colliders[0] in structure["collision_prim_paths"],
            "destination_support_identity_invalid")
    return result, static, paths


def materialize_scene_configuration_submission(
    *, task_request_path: str | Path, installation_receipt_path: str | Path,
    publisher_intake_path: str | Path, source_preparation_receipt_path: str | Path,
    destination_simready_result_path: str | Path, deploy_receipt_path: str | Path,
    release_provenance_path: str | Path, release_environment_path: str | Path,
    runtime_publication_root: str | Path, rights_evidence: dict[str, Any],
    staging_root: str | Path, expected_production_commit: str,
    namespace_timestamp: str,
    sam31_server_profile_path: str | Path | None = None,
    release_admission_mode: str = "promoted",
) -> dict[str, Any]:
    """Validate evidence joins, retain exact inputs, then emit the production request."""
    commit = expected_production_commit
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "release_commit_invalid")
    require(re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", namespace_timestamp) is not None,
            "namespace_timestamp_invalid")
    task = read(task_request_path)
    _validate_task(task)
    team, prefix = slug(task["team_namespace"]), slug(task["run_prefix"])
    namespace = f"{prefix}-{commit}-{namespace_timestamp}"
    run_id = f"{prefix}-{commit[:8]}-{namespace_timestamp.lower()}-scene-configuration"
    deploy, toolchain, renderer = release_inputs(
        deploy_path=Path(deploy_receipt_path), provenance_path=Path(release_provenance_path),
        publication_root=Path(runtime_publication_root), commit=commit,
        release_admission_mode=release_admission_mode)
    inputs = source_inputs(
        installation_path=Path(installation_receipt_path), publisher_path=Path(publisher_intake_path),
        preparation_path=Path(source_preparation_receipt_path), task=task, commit=commit)
    scene_id = inputs["scene_id"]
    source = inputs["identities"]["subject"]
    support_source = inputs["identities"]["support"]
    subject_target, support_target = source["receipt"]["target"], support_source["receipt"]["target"]
    lower, upper = subject_target["world_aabb_min_m"], subject_target["world_aabb_max_m"]
    source_object = records.source_object_selection(
        scene_id=scene_id, instance_id=str(task["subject"]["source_instance_id"]),
        semantic_label=subject_target["semantic_label"], review_label=task["subject"]["review_label"],
        aabb_min=lower, aabb_max=upper, collision_prim_path=source["match"]["prim_path"],
        task_family=task["strategy"])
    support = records.support_plane_input(
        scene_id=scene_id, instance_id=str(task["support"]["source_instance_id"]),
        semantic_label=support_target["semantic_label"], sage_prim_path=support_source["match"]["prim_path"],
        bounds_min=support_target["world_aabb_min_m"], bounds_max=support_target["world_aabb_max_m"])
    destination, static, destination_paths = _destination(
        Path(destination_simready_result_path), lower, upper)
    proposal = derive_passive_destination_placement_proposal(
        support_plane=support, subject_selection=source_object,
        destination_identity=destination["destination_identity"],
        destination_static_qualification=static,
        clearance_gap_m=task["destination"]["clearance_gap_m"],
        support_edge_margin_m=task["destination"]["support_edge_margin_m"])
    # A rotated destination needs a separately authored subject orientation;
    # never silently reuse the source-aligned grasp/template in that case.
    require(abs(proposal["derivation"]["yaw_rad"]) < 1e-9,
            "rotated_destination_subject_orientation_not_authored")
    start = source_object["center_xyz_m"]
    target = list(proposal["pose_world"]["position_world_m"])
    interior = destination["interior_bounds_body_frame_m"]
    for axis in range(2):
        target[axis] += (interior["minimum"][axis] + interior["maximum"][axis]) / 2.0
    target[2] += interior["minimum"][2] + (upper[2] - lower[2]) / 2.0
    grasp_axis = "xyz".index(proposal["derivation"]["long_axis"])
    grasp_sign = -1.0 if proposal["derivation"]["side"] == "positive" else 1.0
    template, success, execution = records.pick_and_place_task_records(
        task_identity=task["task_identity"], object_identity=task["subject"]["replacement_identity"],
        start_center=start, target_center=target, source_min=lower, source_max=upper,
        grasp_axis=grasp_axis, grasp_sign=grasp_sign, success=task["success"],
        resolved_seed=task.get("resolved_seed", 1))
    instruction = (f"Pick up the {task['subject']['review_label'].replace('_', ' ')}, "
                   f"place it fully inside the {task['destination']['visible_label']}, "
                   "release it, and move the gripper clear.")
    require(task.get("instruction", instruction) == instruction, "instruction_semantics_mismatch")
    # Confirmation refers to the retained owner task request. Do not invent
    # confirmation for a request that contains only provider-processing rights.
    if task["human_authority"].get("task_success_contract_confirmed") is True:
        template["owner_success_contract_authority"] = {
            "confirmation_status": "confirmed",
            "accepted_by": task["human_authority"]["accepted_by"],
            "authority_reference": task["human_authority"]["authority_reference"],
        }
    if "success_contract_authority" in task:
        require(isinstance(task["success_contract_authority"], dict), "task_success_authority_invalid")
        template["owner_success_contract_authority"] = dict(task["success_contract_authority"])
    template["instruction"] = instruction
    template["instruction_subject_label"] = task["subject"]["review_label"].replace("_", " ")
    template["visible_target_label"] = task["destination"]["visible_label"]
    template["claim_boundary"] = {
        "native_grasp_qualified": False, "robot_reachability_established": False,
        "policy_execution_authorized": False,
    }
    sam_plan = None
    if task["appearance_removal_method"] == "sam31":
        configured_profile = sam31_server_profile_path or os.environ.get(PROFILE_ENV)
        require(bool(configured_profile), "sam31_server_profile_missing")
        sam_plan = build_sam31_preparation_plan(
            source_commit=commit, task=task,
            host_inputs={
                "task_request": Path(task_request_path),
                "installation_receipt": Path(installation_receipt_path),
                "publisher_intake": Path(publisher_intake_path),
                "source_preparation_receipt": Path(source_preparation_receipt_path),
                "interiorgs_terms": Path(rights_evidence["interiorgs_terms"]),
            },
            source_min=lower, source_max=upper, server_profile_path=Path(configured_profile))
    stage = Staging(Path(staging_root), namespace)
    sam_plan_ref = (stage.json("configuration/sam31_preparation_plan.v1.json", sam_plan)
                    if sam_plan is not None else None)
    # Original bytes are staged solely for control-plane readback. Their URIs
    # continue to point to the publisher, and publication_allowed is false.
    raw_refs = {}
    for role, row in inputs["raw"].items():
        raw_refs[role] = stage.copy(row["path"], f"source/{role}/{row['path'].name}",
                                    publisher_uri=row["publisher_url"])
    stage.copy(Path(task_request_path), "provenance/task_request.v1.json")
    stage.copy(Path(installation_receipt_path), "provenance/source_installation.v1.json")
    stage.copy(Path(publisher_intake_path), "provenance/publisher_intake.v1.json")
    stage.copy(Path(source_preparation_receipt_path), "provenance/source_preparation.v1.json")
    for path, _ in inputs["artifacts"]:
        stage.copy(path, f"validation/{path.name}")
    subject_validation = stage.files[stage.prefix + f"validation/{source['path'].name}"]
    validation_ref = {k: subject_validation[k] for k in ("uri", "digest", "size_bytes")}
    source_ref = stage.json("configuration/source_object_selection.v1.json", source_object)
    evidence_refs = {}
    for key in ("interiorgs_terms", "interiorgs_readme", "sage_readme"):
        evidence_refs[key] = stage.copy(Path(rights_evidence[key]),
                                       f"rights/{key}{Path(rights_evidence[key]).suffix}")
    revisions = {key: inputs["raw"][role]["publisher_revision"]
                 for key, role in (("interiorgs", "appearance_3dgs"),
                                   ("sage_collision", "collision_usd"),
                                   ("sage_usdz", "publisher_scene_usdz"))}
    rights = records.rights_admission(
        scene_id=scene_id, publisher_revisions=revisions,
        terms_sha256=evidence_refs["interiorgs_terms"]["digest"],
        interiorgs_readme_sha256=evidence_refs["interiorgs_readme"]["digest"],
        sage_readme_sha256=evidence_refs["sage_readme"]["digest"],
        human_authority=task["human_authority"])
    rights_ref = stage.json("rights/rights_admission.v1.json", rights)
    human_ref = stage.json("rights/human_authority.v1.json", task["human_authority"])
    manifest = {
        "schema_version": "task_evaluation_scene_source_manifest.v1",
        "status": "candidate_source_bytes_retained", "scene_id": scene_id,
        "publisher_scene_id": scene_id, "coordinate_system": records.COORDINATE_SYSTEM,
        "artifacts": [
            {"role": role, "sha256": raw_refs[key]["digest"], "size_bytes": raw_refs[key]["size_bytes"],
             "publisher_url": raw_refs[key]["uri"], "publisher_revision": inputs["raw"][key]["publisher_revision"]}
            for role, key in (("interiorgs_source_splat", "appearance_3dgs"),
                              ("sage_collision_source", "collision_usd"))],
        "source_task_object": {"publisher_instance_id": str(task["subject"]["source_instance_id"]),
                               "source_aabb_min_xyz_m": lower, "source_aabb_max_xyz_m": upper},
        "source_collision_object": {
            "prim_path": source["match"]["prim_path"],
            "aabb_min_xyz_m": source["match"]["world_aabb_min_m"],
            "aabb_max_xyz_m": source["match"]["world_aabb_max_m"],
            "point_count": source["match"]["point_count"], "face_count": source["match"]["face_count"]},
    }
    manifest_ref = stage.json("scene/source_scene_manifest.v1.json", manifest)
    dest_refs = {key: stage.copy(path, f"destination/{key}{path.suffix}")
                 for key, path in destination_paths.items()}
    dest_result_ref = stage.copy(Path(destination_simready_result_path),
                                "destination/simready_result.v1.json")
    stage.json("configuration/destination_placement_proposal.v1.json", proposal)
    supplemental = {"identity": destination["destination_identity"], "relation": "inside",
                    **dest_refs, "simready_result": dest_result_ref}
    tolerance = records.metric_envelope_tolerance(
        source_min=lower, source_max=upper, target_match=source["match"])
    replacement = task["subject"]["replacement_identity"]
    configs = [
        records.stage_one_configuration(scene_id=scene_id, source_object=source_object,
            support_label=support_target["semantic_label"], human_authority=task["human_authority"]),
        records.stage_two_configuration(scene_id=scene_id,
            collision_source_digest=raw_refs["collision_usd"]["digest"], target_match=source["match"],
            support_prim_path=support["sage_prim_path"]),
        records.stage_three_configuration(scene_id=scene_id, replacement_identity=replacement,
            source_instance_id=str(task["subject"]["source_instance_id"]),
            authoring_target=task["subject"]["authoring_target"], source_min=lower, source_max=upper,
            dimension_tolerance=tolerance, physics_bounds=task["subject"]["physics_bounds"]),
        records.stage_four_configuration(replacement_identity=replacement, dimension_tolerance=tolerance),
        records.stage_five_configuration(replacement_identity=replacement),
        records.stage_six_configuration(scene_identity=task["scene_identity"], support_plane=support,
                                       start_center=start, bottom_z=lower[2]),
    ]
    if sam_plan_ref is not None:
        configs[0]["gaussian_cutout"] = {
            "selection_rule": SAM31_SELECTION_RULE, "retained_rows_must_remain_byte_exact": True}
        configs[0]["required_views"]["minimum"] = 16
        configs[0]["required_views"]["mask_source"] = SAM31_MASK_SOURCE
        configs[0]["sam31_review_kind"] = "ai"
        configs[0]["sam31_preparation_plan"] = sam_plan_ref
    config_refs = [stage.json(f"configuration/stage_{i + 1}.v1.json", v)
                   for i, v in enumerate(configs)]
    recipe = records.recipe(
        recipe_id=run_id + "-recipe", team_namespace=team, scene_identity=task["scene_identity"],
        task_identity=task["task_identity"], subject_identity=replacement,
        source_manifest_digest=manifest_ref["digest"], rights_admission_digest=rights_ref["digest"],
        output_identity=task["output_identity"], stage_configuration_references=config_refs,
        supplemental_destination=supplemental)
    validate_scene_construction_recipe(recipe)
    recipe_ref = stage.json("configuration/scene_construction_recipe.v1.json", recipe)
    release = records.exact_production_release_binding(
        team_namespace=team, scene_identity=task["scene_identity"], source_commit=commit,
        deploy_receipt=deploy, deploy_receipt_sha256=sha(Path(deploy_receipt_path)),
        release_environment_sha256=sha(Path(release_environment_path)),
        scene_configuration_publication=toolchain, splat_render_publication=renderer,
        release_admission_mode=release_admission_mode)
    release_ref = stage.json("release/exact_production_release_binding.v1.json", release)
    camera_ref = stage.json("configuration/camera_calibration_plan.v1.json",
        records.camera_calibration_plan(scene_id=scene_id, strategy=task["strategy"]))
    request = {
        "schema_version": "task_evaluation_launch_preparation_request.v1",
        "run_mode": "scene_configuration", "expected_production_commit": commit,
        "preparation_id": run_id + "-preparation", "team_namespace": team, "run_id": run_id,
        "scene": {
            "mode": "configure_source_scene", "identity": task["scene_identity"],
            "source_manifest": manifest_ref,
            "appearance": {"kind": "interiorgs", "representation": raw_refs["appearance_3dgs"],
                "renderer_qualification": stage.json("configuration/renderer_qualification_plan.v1.json",
                                                     records.renderer_qualification_plan())},
            "geometry": {"kind": "sage_derived", "collision": raw_refs["collision_usd"],
                         "validation": validation_ref},
            "registration": {
                "metric_registration": stage.json("configuration/metric_registration_input.v1.json",
                    records.metric_registration_input(scene_id=scene_id)),
                "support_plane": stage.json("configuration/support_plane_input.v1.json", support),
                "robot_mount_interface": stage.json("configuration/robot_mount_interface_plan.v1.json",
                    records.robot_mount_interface_plan(scene_id=scene_id, strategy=task["strategy"])),
                "workspace_clearance": stage.json("configuration/workspace_clearance_plan.v1.json", {
                    "schema_version": "task_evaluation_workspace_clearance_plan.v1",
                    "status": "execute_during_scene_configuration_run",
                    "scene_id": scene_id, "workspace_clearance_qualified": False,
                    "support_bounds_min_xyz_m": support["bounds_min_xyz_m"],
                    "support_bounds_max_xyz_m": support["bounds_max_xyz_m"],
                    "all_task_waypoints_must_be_validated": True}),
                "camera_calibration": camera_ref},
            "rights": {"admission": rights_ref, "evidence": [
                {"role": "publisher_terms", "artifact": evidence_refs["interiorgs_terms"]},
                {"role": "publisher_readme", "artifact": evidence_refs["interiorgs_readme"]},
                {"role": "publisher_readme", "artifact": evidence_refs["sage_readme"]},
                {"role": "human_authority_record", "artifact": human_ref}],
                "source_bytes_redistributable": False, "provider_disclosure_scope": "derived_only"}},
        "construction": {"mode": "production_recipe", "recipe": recipe_ref,
                         "output_identity": task["output_identity"]},
        "task": {"identity": task["task_identity"], "binding_mode": "define_configuration_template",
                 "kind": "rigid_relocation", "strategy": task["strategy"],
                 "subject": {"mode": "construct_from_scene_object", "identity": replacement,
                    "representation_kind": "simready_usd", "source_object": source_ref,
                    "rights_admission": rights_ref, "provider_disclosure_allowed": True},
                 "definition": stage.json("configuration/task_template.v1.json", template),
                 "success_criteria": stage.json("configuration/task_success_criteria.v1.json", success),
                 "execution": stage.json("configuration/task_execution_spec.v1.json", execution),
                 "destination": {
                     "schema_version": "task_evaluation_rigid_destination_asset.v1",
                     "identity": destination["destination_identity"], "relation": "inside",
                     "visible_label": task["destination"]["visible_label"],
                     **{k: dest_refs[k] for k in ("asset", "rights_admission", "static_qualification")},
                     "pose_world": proposal["pose_world"], "native_probe": proposal["native_probe"],
                     "provider_disclosure_allowed": True}},
        "sensors": {"configuration": camera_ref},
        "runtime": {
            "identity": {"id": "task-evaluation-scene-configuration-provider", "version": commit[:8]},
            "oci_image": NATIVE_TASK_ARENA_IMAGE,
            "entrypoint": ["/opt/blueprint/run-task-evaluation-scene-configuration"],
            "health_protocol": stage.json("release/runtime_health_protocol.v1.json",
                                         records.runtime_health_protocol(source_commit=commit)),
            "requirements": {"cpu_cores": 8, "memory_gib": 32, "gpu_count": 1, "disk_gib": 64},
            "network": {"default": "deny", "allowlist": ["api.openai.com"]},
            "secret_refs": ["secret-file:openai_api_key"],
            "mounts": [{"source": release_ref, "container_path": "/inputs/release-binding.json",
                        "mode": "read_only"}, {"container_path": "/outputs", "mode": "output"}],
            "output_limit_bytes": 20_000_000_000},
        "execution_adapter": {"kind": "scene_configuration_pipeline", "version": "v1",
                              "runtime_source_bundle": release_ref},
        "publication": {"input_namespace": namespace, "service_account_readback_required": True},
        "spend": records.spend_block(),
    }
    if sam_plan_ref is not None:
        request["runtime"]["mounts"].append({
            "source": sam_plan_ref, "container_path": "/inputs/sam31-preparation-plan.json",
            "mode": "read_only"})
    validate_launch_preparation_request(request)
    references = stage.reference_rows(request)
    decision = resolve_scene_configuration_disclosure(
        stage_one_configuration=configs[0], rights_admission=rights)
    require(decision["render_execution_site"] == "control_plane" and
            decision["source_appearance_bytes_to_provider"] is False, "raw_disclosure_forbidden")
    envelope = {"request": request, "recipe": recipe, "materialized_references": references,
                "render_inputs_result": {"disclosure_decision": decision}}
    configuration_map = {row["stage_id"]: cfg for row, cfg in zip(recipe["stage_sequence"], configs, strict=True)}
    validate_immutable_stage_configurations(envelope=envelope, configurations=configuration_map)
    validate_scene_configuration_source_bindings(envelope=envelope, configurations=configuration_map)
    stage.json("scene_configuration_preparation_request.v1.json", request)
    inventory = {"schema_version": "task_evaluation_scene_configuration_submission_manifest.v1",
                 "status": "validated_pending_production_publication_and_submission",
                 "source_commit": commit, "input_namespace": namespace,
                 "release_admission_mode": release_admission_mode, "claim_ceiling": "development_only",
                 "request_digest": launch_preparation_request_digest(request),
                 "files": list(stage.files.values()),
                 "raw_source_upload_allowed": False,
                 "native_qualification_claimed": False, "provider_allocated": False,
                 "manifest_digest": ""}
    inventory["manifest_digest"] = canonical_digest(inventory, digest_field="manifest_digest")
    stage.json("bundle_manifest.v1.json", inventory)
    return {"staging_root": str(stage.root), "input_namespace": namespace,
            "request_digest": inventory["request_digest"], "manifest_digest": inventory["manifest_digest"],
            "status": inventory["status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task-request", "installation-receipt", "publisher-intake",
                 "source-preparation-receipt", "destination-simready-result",
                 "deploy-receipt", "release-provenance", "release-environment"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("runtime-publication-root", "staging-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("expected-production-commit", "namespace-timestamp"):
        parser.add_argument("--" + name, required=True)
    for name in ("interiorgs-terms", "interiorgs-readme", "sage-readme"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--sam31-server-profile-path", type=Path)
    parser.add_argument("--release-admission-mode", choices=("promoted", "development_iteration"),
                        default="promoted")
    args = vars(parser.parse_args())
    evidence = {name: args.pop(name) for name in ("interiorgs_terms", "interiorgs_readme", "sage_readme")}
    for name in ("task_request", "installation_receipt", "publisher_intake",
                 "source_preparation_receipt", "destination_simready_result",
                 "deploy_receipt", "release_provenance", "release_environment"):
        args[name + "_path"] = args.pop(name)
    print(json.dumps(materialize_scene_configuration_submission(**args, rights_evidence=evidence),
                     sort_keys=True))


if __name__ == "__main__":
    main()
