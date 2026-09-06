"""Assemble one scene-configuration submission from owner-provided completed bytes.

The owner uploads a finished 3DGS appearance asset and a companion collision
mesh, names the movable subject and its support, declares a coordinate frame,
and consents to private provider processing.  This builder turns those exact
owned inputs into a ``scene_configuration_preparation_request.v1`` plus a
``bundle_manifest.v1`` inventory, without an ``installation_receipt``,
``publisher_intake``, InteriorGS/SAGE evidence, or a SAM31 removal plan.

Nothing here manufactures a measurement.  The declared metric frame is retained
as *declared*, never *measured*; collision, renderer, native-import and grasp
qualifications are expressed as run-produced plans marked not-yet-qualified; and
the owner's raw bytes are never marked redistributable or provider-uploadable.
"""

from __future__ import annotations

import math
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
from .task_evaluation_scene_configuration_submission_inputs import (
    Staging,
    checked_file,
    read,
    release_inputs,
    require,
    sha,
    slug,
)

from .task_evaluation_scene_construction_recipe import (
    SCHEMA_VERSION as RECIPE_SCHEMA, CAPABILITY_ORDER as COMPLETED_CONSTRUCTION_CAPABILITIES,
)
from .task_evaluation_completed_scene_transaction import completed_submission_transaction

_APPEARANCE_KIND = {"gaussian_splat": "gaussian_splat", "mesh": "other_observed"}


def _finite_vector(value: Any, length: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != length:
        return None
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            return None
        out.append(float(item))
    return out


def _center(minimum: list[float], maximum: list[float]) -> list[float]:
    return [(minimum[i] + maximum[i]) / 2.0 for i in range(3)]


def _plan(schema: str, **fields: Any) -> dict[str, Any]:
    return {"schema_version": schema, "status": "execute_during_scene_configuration_run", **fields}


def _owner_terms(scene_id: str, owner: dict, human_authority: dict, rights_reference: str,
                 collision_rights_reference: str | None) -> dict[str, Any]:
    return {"schema_version": "task_evaluation_completed_scene_owner_terms.v1", "status": "owner_declared",
        "scene_id": scene_id, "owner": dict(owner), "owner_rights_reference": rights_reference,
        "collision_rights_reference": collision_rights_reference,
        "source_is_owner_provided_completed_asset": True, "captured_observation_supplied": False,
        "physical_metrology_claimed": False, "public_redistribution_allowed": False,
        "authorized_by": human_authority["accepted_by"], "accepted_on": human_authority["accepted_on"],
        "declared_scope": "owner authorizes internal development-only private processing of this owned asset"}


def _rights_admission(scene_id: str, owner: dict, human_authority: dict, rights_reference: str,
                      collision_rights_reference: str | None) -> dict[str, Any]:
    return {"schema_version": "task_evaluation_completed_scene_rights_admission.v1",
        "status": "admitted_for_internal_development", "program_id": "arm-decision-proof-v1",
        "product": "Task Evaluation Run", "scene_id": scene_id, "source": "owner_provided_completed_asset",
        "owner": dict(owner), "owner_rights_reference": rights_reference,
        "collision_rights_reference": collision_rights_reference,
        "declared_use_scope": "internal_noncommercial_research_and_development_Task_Evaluation_Run",
        "private_provider_processing_allowed": bool(human_authority["private_derived_frame_disclosure_authorized"]),
        "provider_training_allowed": False, "public_redistribution_allowed": False,
        "source_bytes_redistributable": False, "physical_metrology_claimed": False,
        "provider_disclosure": {"raw_owner_source_bytes_may_be_uploaded": False,
            "minimum_digest_bound_derived_runtime_bytes_may_be_privately_processed": True,
            "provider_training_allowed": False, "public_redistribution_allowed": False,
            "provider_retention_rule": "bounded_to_the_exact_Task_Evaluation_Run_then_governed_teardown",
            "network_egress_rule": "fail_closed_except_pinned_runtime_dependencies_and_governed_result_sync"},
        "authority_records": [{"authority_kind": "authenticated_owner_consent",
            "authority_reference": human_authority["authority_reference"],
            "authorized_by": human_authority["accepted_by"], "recorded_on": human_authority["accepted_on"],
            "declared_scope": f"internal_and_development_only_use_of_owner_scene_{scene_id}"}],
        "claim_boundary": ("Owner-authorized internal development-only processing of an owner-provided "
            "completed asset. Not a redistribution, dataset-publication, or commercial-use grant.")}


def _source_object(scene_id: str, obj: dict, review_label: str) -> dict[str, Any]:
    minimum = [float(v) for v in obj["aabb_min_xyz_m"]]
    maximum = [float(v) for v in obj["aabb_max_xyz_m"]]
    return {"schema_version": "task_evaluation_completed_scene_object_selection.v1",
        "status": "frozen_before_scene_configuration_run", "scene_id": scene_id,
        "source_object_id": str(obj["source_object_id"]), "review_label": review_label,
        "geometry_origin": "owner_provided_asset", "appearance_source": "owner_provided_completed_asset",
        "aabb_min_xyz_m": minimum, "aabb_max_xyz_m": maximum, "center_xyz_m": _center(minimum, maximum),
        "point_count": int(obj["point_count"]), "face_count": int(obj["face_count"]),
        "physical_object_identity_proven": False, "source_object_is_physics_authority": False}


@completed_submission_transaction
def materialize_completed_scene_submission(
    *, binding: dict[str, Any], task: dict[str, Any], task_request_path: str | Path,
    deploy_receipt_path: str | Path, release_provenance_path: str | Path,
    release_environment_path: str | Path, runtime_publication_root: str | Path,
    expected_production_commit: str, namespace_timestamp: str, release_admission_mode: str,
    staging_root: str | Path, scene_intent_digest: str,
) -> dict[str, Any]:
    """Validate owned inputs, stage exact bytes, and emit the production request."""
    commit = expected_production_commit
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "completed_release_commit_invalid")
    require(re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", namespace_timestamp) is not None,
            "completed_namespace_timestamp_invalid")
    deploy, toolchain, renderer = release_inputs(
        deploy_path=Path(deploy_receipt_path), provenance_path=Path(release_provenance_path),
        publication_root=Path(runtime_publication_root), commit=commit,
        release_admission_mode=release_admission_mode)
    references = binding["references"]
    appearance_src = checked_file(references["primary"]["path"], references["primary"])
    collision_src = checked_file(references["collision"]["path"], references["collision"])
    owner = binding["owner"]
    human_authority = task["human_authority"]
    frame = task["coordinate_frame"]
    scene_id = task["scene_identity"]["id"]
    team, prefix = slug(task["team_namespace"]), slug(task["run_prefix"])
    namespace = f"{prefix}-{commit}-{namespace_timestamp}"
    run_id = f"{prefix}-{commit[:8]}-{namespace_timestamp.lower()}-scene-configuration"

    subject, support = task["subject"], task["support"]
    lower = [float(v) for v in subject["aabb_min_xyz_m"]]
    upper = [float(v) for v in subject["aabb_max_xyz_m"]]
    start = _center(lower, upper)
    from .task_evaluation_scene_configuration_submission import _destination
    destination_result_path = checked_file(task["destination"]["simready_result"]["path"],
                                           task["destination"]["simready_result"])
    destination_result, destination_static, destination_paths = _destination(destination_result_path, lower, upper)
    destination_identity = destination_result["destination_identity"]
    # The owner supplies the destination pose. Derive the object target from
    # its admitted interior, preserving the owner's orientation and placement.
    from scipy.spatial.transform import Rotation
    orientation = task["destination"]["orientation_xyzw"]
    interior = destination_result["interior_bounds_body_frame_m"]
    local_target = [(interior["minimum"][i] + interior["maximum"][i]) / 2 for i in range(3)]
    local_target[2] = interior["minimum"][2] + (upper[2] - lower[2]) / 2
    target = (Rotation.from_quat(orientation).apply(local_target)
              + task["destination"]["position_world_m"]).tolist()
    template, success, execution = records.pick_and_place_task_records(
        task_identity=task["task_identity"], object_identity=subject["identity"],
        start_center=start, target_center=target, source_min=lower, source_max=upper,
        grasp_axis=int(task["grasp"]["axis"]), grasp_sign=float(task["grasp"]["sign"]),
        success=task["success"], resolved_seed=1)
    label = str(subject["review_label"]).replace("_", " ")
    instruction = (f"Pick up the {label}, place it {task['destination']['relation']} the "
                   f"{task['destination']['visible_label']}, release it, and move the gripper clear.")
    template.update(instruction=instruction, instruction_subject_label=label,
                    visible_target_label=task["destination"]["visible_label"],
                    claim_boundary={"native_grasp_qualified": False, "robot_reachability_established": False,
                                    "policy_execution_authorized": False, "physical_world_truth_claimed": False})
    template["owner_success_contract_authority"] = {"confirmation_status": "confirmed",
        "accepted_by": human_authority["accepted_by"], "authority_reference": human_authority["authority_reference"]}

    from .task_evaluation_owner_source_store import source_uri
    from .task_evaluation_completed_scene_publication import SCHEMA as INTAKE_SCHEMA, RELATIVE_PATH
    stage = Staging(Path(staging_root), namespace)
    filenames = binding["source_filenames"]
    primary_uri = source_uri(references["primary"]["sha256"], filenames["primary"])
    collision_uri = source_uri(references["collision"]["sha256"], filenames["collision"])
    appearance_ref = stage.copy(appearance_src, "source/appearance/" + primary_uri.rsplit("/", 1)[-1],
        publisher_uri=primary_uri)
    collision_ref = appearance_ref if binding["source_kind"] == "mesh" else stage.copy(
        collision_src, "source/collision/" + collision_uri.rsplit("/", 1)[-1], publisher_uri=collision_uri)
    raw_appearance_ref = appearance_ref
    host_sources = [appearance_ref] if collision_ref == appearance_ref else [appearance_ref, collision_ref]
    splat_normalization = None
    if task.get("splat_normalization") is not None:
        ref = task["splat_normalization"]
        path = checked_file(ref["path"], ref)
        splat_normalization = read(path, digest_field="normalization_digest")
        converted = checked_file(path.parent / splat_normalization["output"]["relative_path"], splat_normalization["output"])
        appearance_ref = stage.copy(converted, "source/normalized_appearance/normalized.ply",
            publisher_uri=source_uri(splat_normalization["output"]["sha256"], "normalized.ply"))
        host_sources.append(appearance_ref)
    stage.json(RELATIVE_PATH, {"schema_version": INTAKE_SCHEMA,
        "scene_intent_authority": task["scene_intent_authority"],
        "source_binding": task["source_binding"], "intent_digest": scene_intent_digest,
        "artifacts": host_sources})
    raw_collision_ref = collision_ref
    normalization_path = checked_file(task["geometry_normalization"]["path"], task["geometry_normalization"])
    normalization = read(normalization_path, digest_field="normalization_digest")
    normalized_path = checked_file(normalization_path.parent / normalization["output"]["relative_path"],
                                   normalization["output"])
    collision_ref = stage.copy(normalized_path, "geometry/normalized_scene.usda")
    validation_ref = stage.copy(normalization_path, "geometry/mesh_normalization.v1.json")
    stage.copy(Path(task_request_path), "provenance/completed_task_request.v1.json")
    manifest = {"schema_version": "task_evaluation_completed_scene_source_manifest.v1",
        "status": "candidate_source_bytes_retained", "scene_id": scene_id,
        "source": "owner_provided_completed_asset", "captured_observation_supplied": False,
        "coordinate_system": {"declared_meters_per_unit": frame["declared_meters_per_unit"],
            "declared_up_axis": frame["declared_up_axis"], "physical_scale_measured": False,
            "physical_metrology_claimed": False},
        "artifacts": [
            {"role": "owner_appearance_source", "kind": task["appearance_kind"],
             "sha256": raw_appearance_ref["digest"], "size_bytes": raw_appearance_ref["size_bytes"]},
            {"role": "owner_collision_source", "kind": "collision_mesh",
             "sha256": raw_collision_ref["digest"], "size_bytes": raw_collision_ref["size_bytes"]},
            {"role": "normalized_owner_collision", "kind": "collision_mesh",
             "sha256": collision_ref["digest"], "size_bytes": collision_ref["size_bytes"]}],
        "source_task_object": {"source_object_id": subject["source_object_id"],
            "runtime_prim_path": subject["runtime_prim_path"],
            "source_aabb_min_xyz_m": lower, "source_aabb_max_xyz_m": upper},
        "source_support_object": {"source_object_id": support["source_object_id"],
            "runtime_prim_path": support["runtime_prim_path"],
            "aabb_min_xyz_m": [float(v) for v in support["aabb_min_xyz_m"]],
            "aabb_max_xyz_m": [float(v) for v in support["aabb_max_xyz_m"]]}}
    manifest["artifacts"][0].update(provider_upload_allowed=False,
        splat_count=binding["inspection"].get("appearance", {}).get("retained_gaussian_count"))
    manifest["runtime_appearance_role"] = "owner_appearance_source"
    if splat_normalization is not None:
        manifest["runtime_appearance_role"] = "normalized_owner_appearance"
        manifest["appearance_normalization"] = splat_normalization
        manifest["artifacts"].append({"role": "normalized_owner_appearance",
            "sha256": appearance_ref["digest"], "size_bytes": appearance_ref["size_bytes"],
            "splat_count": splat_normalization["retained_gaussian_count"], "provider_upload_allowed": False})
    manifest_ref = stage.json("scene/source_scene_manifest.v1.json", manifest)

    rights = _rights_admission(scene_id, owner, human_authority, binding["source_content_digest"],
                               task.get("collision_rights_reference"))
    rights_ref = stage.json("rights/rights_admission.v1.json", rights)
    human_ref = stage.json("rights/human_authority.v1.json", human_authority)
    terms_ref = stage.json("rights/owner_source_terms.v1.json",
        _owner_terms(scene_id, owner, human_authority, binding["source_content_digest"],
                     task.get("collision_rights_reference")))
    source_object_ref = stage.json("configuration/source_object_selection.v1.json",
                                   _source_object(scene_id, subject, label))
    renderer_ref = stage.json("configuration/renderer_qualification_plan.v1.json",
        _plan("task_evaluation_renderer_qualification_plan.v1",
              appearance_source="owner_provided_completed_asset", appearance_kind=task["appearance_kind"],
              browser_preview_qualifies=False, appearance_qualified=False,
              required_bindings=["renderer_name", "renderer_version", "environment_digest", "camera_pose",
                                 "camera_intrinsics", "source_appearance_digest", "rendered_image_digests",
                                 "fidelity_qualification_receipt"]))
    metric_ref = stage.json("configuration/metric_registration_input.v1.json",
        _plan("task_evaluation_completed_scene_metric_registration_input.v1", scene_id=scene_id,
              declared_meters_per_unit=frame["declared_meters_per_unit"],
              declared_up_axis=frame["declared_up_axis"], registration_authority="owner_declared_frame",
              physical_metrology_claimed=False, physical_scale_measured=False,
              production_validation_required=True))
    support_ref = stage.json("configuration/support_plane_input.v1.json",
        {"schema_version": "task_evaluation_completed_scene_support_plane_input.v1",
         "status": "frozen_candidate_pending_production_validation", "scene_id": scene_id,
         "source_object_id": support["source_object_id"], "review_label": support.get("label", ""),
         "bounds_min_xyz_m": [float(v) for v in support["aabb_min_xyz_m"]],
         "bounds_max_xyz_m": [float(v) for v in support["aabb_max_xyz_m"]],
         "top_z_m": float(support["aabb_max_xyz_m"][2]),
         "required_validation": ["planarity", "finite_bounds", "support_contact",
                                 "target_region_inside_bounds"]})
    robot_ref = stage.json("configuration/robot_mount_interface_plan.v1.json",
        _plan("task_evaluation_completed_scene_robot_mount_interface_plan.v1", scene_id=scene_id,
              scene_base_frame=f"owner_scene_{scene_id}_world",
              supported_robot_classes=["fixed_arm", "mobile_manipulator"],
              minimum_non_target_clearance_m=0.03,
              configuration_run_must_not_claim_any_robot_qualified=True))
    workspace_ref = stage.json("configuration/workspace_clearance_plan.v1.json",
        _plan("task_evaluation_completed_scene_workspace_clearance_plan.v1", scene_id=scene_id,
              workspace_clearance_qualified=False,
              support_bounds_min_xyz_m=[float(v) for v in support["aabb_min_xyz_m"]],
              support_bounds_max_xyz_m=[float(v) for v in support["aabb_max_xyz_m"]],
              all_task_waypoints_must_be_validated=True))
    camera_ref = stage.json("configuration/camera_calibration_plan.v1.json",
        records.camera_calibration_plan(scene_id=scene_id, strategy="pick_and_place"))

    destination_refs = {key: stage.copy(path, f"destination/{key}{path.suffix}")
                        for key, path in destination_paths.items()}
    destination_result_ref = stage.copy(destination_result_path, "destination/simready_result.v1.json")
    native_probe = {"schema_version": "task_evaluation_rigid_destination_native_probe_configuration.v1",
        "placement_support_scene_prim_paths": [support["runtime_prim_path"]],
        "qualification_limits": {"maximum_penetration_m": 0.005, "minimum_support_contact_force_n": 0.05,
            "maximum_forbidden_contact_force_n": 5.0, "settle_translation_tolerance_m": 0.01,
            "settle_rotation_tolerance_rad": 0.08, "reset_translation_tolerance_m": 0.005,
            "reset_rotation_tolerance_rad": 0.04,
            "minimum_camera_pixels": {"external": 64, "wrist": 32, "overview": 64}},
        "settle_sample_count": 3, "settle_steps_per_sample": 30}

    definition_ref = stage.json("configuration/task_template.v1.json", template)
    success_ref = stage.json("configuration/task_success_criteria.v1.json", success)
    execution_ref = stage.json("configuration/task_execution_spec.v1.json", execution)
    from .task_evaluation_completed_scene_recipe import construction_recipe, stage_configurations
    configurations = stage_configurations(task=task, collision_digest=collision_ref["digest"])
    config_refs = [stage.json(f"configuration/stage_{i + 1}.v1.json", value)
                   for i, value in enumerate(configurations)]
    recipe = construction_recipe(run_id=run_id, task=task, source_manifest_digest=manifest_ref["digest"],
        rights_admission_digest=rights_ref["digest"], configurations=config_refs,
        supplemental_destination={"identity": destination_identity, "relation": task["destination"]["relation"],
            **destination_refs, "simready_result": destination_result_ref})
    recipe_ref = stage.json("configuration/scene_construction_recipe.v1.json", recipe)
    release = records.exact_production_release_binding(
        team_namespace=team, scene_identity=task["scene_identity"], source_commit=commit,
        deploy_receipt=deploy, deploy_receipt_sha256=sha(Path(deploy_receipt_path)),
        release_environment_sha256=sha(Path(release_environment_path)),
        scene_configuration_publication=toolchain, splat_render_publication=renderer,
        release_admission_mode=release_admission_mode)
    release_ref = stage.json("release/exact_production_release_binding.v1.json", release)
    health_ref = stage.json("release/runtime_health_protocol.v1.json",
                            records.runtime_health_protocol(source_commit=commit))

    request = {
        "schema_version": "task_evaluation_launch_preparation_request.v1", "run_mode": "scene_configuration",
        "expected_production_commit": commit, "preparation_id": run_id + "-preparation",
        "team_namespace": team, "run_id": run_id, "scene_intent_digest": scene_intent_digest,
        "scene": {"mode": "configure_source_scene", "identity": task["scene_identity"],
            "source_manifest": manifest_ref,
            "appearance": {"kind": task["appearance_kind"], "representation": appearance_ref,
                           "renderer_qualification": renderer_ref},
            "geometry": {"kind": "observed_mesh", "collision": collision_ref, "validation": validation_ref},
            "registration": {"metric_registration": metric_ref, "support_plane": support_ref,
                             "robot_mount_interface": robot_ref, "workspace_clearance": workspace_ref,
                             "camera_calibration": camera_ref},
            "rights": {"admission": rights_ref, "evidence": [
                {"role": "publisher_terms", "artifact": terms_ref},
                {"role": "human_authority_record", "artifact": human_ref}],
                "source_bytes_redistributable": False, "provider_disclosure_scope": "derived_only"}},
        "construction": {"mode": "production_recipe", "recipe": recipe_ref,
                         "output_identity": task["output_identity"]},
        "task": {"identity": task["task_identity"], "binding_mode": "define_configuration_template",
            "kind": "rigid_relocation", "strategy": "pick_and_place",
            "subject": {"mode": "construct_from_scene_object", "identity": subject["identity"],
                        "representation_kind": "simready_usd", "source_object": source_object_ref,
                        "rights_admission": rights_ref, "provider_disclosure_allowed": True},
            "definition": definition_ref, "success_criteria": success_ref, "execution": execution_ref,
            "destination": {"schema_version": "task_evaluation_rigid_destination_asset.v1",
                "identity": destination_identity, "relation": task["destination"]["relation"],
                "visible_label": task["destination"]["visible_label"],
                **{key: destination_refs[key] for key in ("asset", "rights_admission", "static_qualification")},
                "pose_world": {"position_world_m": task["destination"]["position_world_m"],
                               "orientation_xyzw": [float(v) for v in task["destination"]["orientation_xyzw"]]},
                "native_probe": native_probe, "provider_disclosure_allowed": True}},
        "sensors": {"configuration": camera_ref},
        "runtime": {"identity": {"id": "task-evaluation-scene-configuration-provider", "version": commit[:8]},
            "oci_image": NATIVE_TASK_ARENA_IMAGE,
            "entrypoint": ["/opt/blueprint/run-task-evaluation-scene-configuration"],
            "health_protocol": health_ref,
            "requirements": {"cpu_cores": 8, "memory_gib": 32, "gpu_count": 1, "disk_gib": 64},
            "network": {"default": "deny", "allowlist": ["api.openai.com"]},
            "secret_refs": ["secret-file:openai_api_key"],
            "mounts": [{"source": release_ref, "container_path": "/inputs/release-binding.json",
                        "mode": "read_only"}, {"container_path": "/outputs", "mode": "output"}],
            "output_limit_bytes": 20_000_000_000},
        "execution_adapter": {"kind": "scene_configuration_pipeline", "version": "v1",
                              "runtime_source_bundle": release_ref},
        "publication": {"input_namespace": namespace, "service_account_readback_required": True},
        "spend": records.spend_block()}
    validate_launch_preparation_request(request)
    from .task_evaluation_scene_configuration_stage_configuration import validate_immutable_stage_configurations
    validate_immutable_stage_configurations(envelope={"request": request, "recipe": recipe,
        "materialized_references": stage.reference_rows(request)},
        configurations={f"stage-{i + 1}": value for i, value in enumerate(configurations)})
    for row in stage.reference_rows(request):
        checked_file(Path(row["materialized_path"]), {"sha256": row["digest"], "size_bytes": row["size_bytes"]})
    stage.json("scene_configuration_preparation_request.v1.json", request)
    inventory = {"schema_version": "task_evaluation_scene_configuration_submission_manifest.v1",
        "status": "validated_pending_production_publication_and_submission", "source_commit": commit,
        "input_namespace": namespace, "release_admission_mode": release_admission_mode,
        "claim_ceiling": "development_only", "source": "owner_provided_completed_asset",
        "request_digest": launch_preparation_request_digest(request), "files": list(stage.files.values()),
        "raw_source_upload_allowed": False, "captured_observation_supplied": False,
        "native_qualification_claimed": False, "physical_metrology_claimed": False,
        "provider_allocated": False, "manifest_digest": ""}
    inventory["manifest_digest"] = canonical_digest(inventory, digest_field="manifest_digest")
    stage.json("bundle_manifest.v1.json", inventory)
    return {"staging_root": str(stage.root), "input_namespace": namespace,
            "request_digest": inventory["request_digest"], "manifest_digest": inventory["manifest_digest"],
            "status": inventory["status"]}


__all__ = ["materialize_completed_scene_submission", "RECIPE_SCHEMA",
           "COMPLETED_CONSTRUCTION_CAPABILITIES"]
