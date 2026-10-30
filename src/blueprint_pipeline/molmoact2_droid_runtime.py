"""Pinned MolmoAct2 DROID binding for the existing Panda–Robotiq beta station."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .molmoact2_droid_assets import (
    INTERFACE_ID, MODEL_ID, MODEL_REVISION, SOURCE_REVISION, NORM_SHA256, release_inventory,
)
from .molmoact2_droid_policy import MolmoAct2DROIDBinding

CANDIDATE_ID = "molmoact2_droid"
PROFILE_ID = "molmoact2_droid.v1"
MODEL_SOURCE = "https://huggingface.co/" + MODEL_ID
MINIMUM_GPU_MEMORY_GB = 48  # Conservative policy+Isaac floor; not measured peak memory.
PORT = 8203
ASSUMPTION_REFERENCE = ("initial_simulation_assumption:existing_DROID_station_external_and_wrist_mounts;"
    "aspect_preserving640x360_RGB_not_site_or_training_calibration;"
    "publisher_original_DROID15Hz_normalized_closure;execute8_of15_explicit_harness_choice")


def checkpoint_identity() -> dict[str, Any]:
    return {"model_id": MODEL_ID, "revision": MODEL_REVISION,
            "files": [{"path": row["path"], "size_bytes": row["bytes"],
                       "sha256": "sha256:" + (row["publisher_lfs_sha256"] or row["locally_verified_sha256"])}
                      for row in sorted(release_inventory()["files"], key=lambda row: row["path"])
                      if row["required_for_inference"]]}


def interface_contract() -> dict[str, Any]:
    return {"policy_interface_id": INTERFACE_ID, "robot_profile_id": "droid_franka_panda_robotiq_2f85_v1",
        "observation_profile_id": "molmoact2_droid_two_camera_state8.v1",
        "action_schema_id": "droid_absolute_joint_position_v1", "control_hz": 15,
        "action_horizon": 15, "execution_horizon": 8, "camera_roles": ["external", "wrist"],
        "image_width": 640, "image_height": 360, "image_encoding": "rgb_uint8",
        "image_preprocessing": "resize_with_padding", "history_frames": 1,
        "normalization_digest": "sha256:" + NORM_SHA256}


def policy_binding(policy: Mapping[str, Any]) -> MolmoAct2DROIDBinding:
    if (not isinstance(policy, Mapping) or set(policy) != {"policy_id", "policy_interface_id", "model_identity_digest",
            "source_revision", "binding"} or policy["policy_id"] != PROFILE_ID
            or policy["policy_interface_id"] != INTERFACE_ID or policy["source_revision"] != SOURCE_REVISION
            or policy["model_identity_digest"] != canonical_digest(checkpoint_identity())):
        raise ValueError("molmoact2_droid_policy_identity_invalid")
    binding = MolmoAct2DROIDBinding(**dict(policy["binding"]))
    binding.validate()
    if binding.execution_horizon != 8 or binding.camera_binding_reference != ASSUMPTION_REFERENCE:
        raise ValueError("molmoact2_droid_policy_binding_invalid")
    return binding


def serialized_binding(policy: Mapping[str, Any]) -> dict[str, Any]:
    return asdict(policy_binding(policy))


def candidate_execution_spec(*, scene_plan: Mapping[str, Any], contract: Mapping[str, Any], source_commit: str) -> dict[str, Any]:
    from .adp009d_policy_episode import maximum_policy_queries_for_task_spec
    from .droid_policy_canary_embodiment import concrete_droid_task_instruction
    from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE

    cameras = [row for row in scene_plan.get("cameras", []) if row.get("role") in {"external", "wrist"}
               and row.get("policy_input") is True]
    if {row["role"] for row in cameras} != {"external", "wrist"} or len(cameras) != 2:
        raise ValueError("molmoact2_droid_required_policy_camera_unavailable")
    configuration = {"scene_plan_digest": scene_plan["plan_digest"],
        "robot_profile_id": interface_contract()["robot_profile_id"], "cameras": cameras,
        "camera_binding_reference": ASSUMPTION_REFERENCE}
    binding = MolmoAct2DROIDBinding(canonical_digest(configuration), ASSUMPTION_REFERENCE)
    policy = {"policy_id": PROFILE_ID, "policy_interface_id": INTERFACE_ID,
        "model_identity_digest": canonical_digest(checkpoint_identity()), "source_revision": SOURCE_REVISION,
        "binding": asdict(binding)}
    inventory = release_inventory()
    runtime = {"source_commit": source_commit, "container_image": NATIVE_TASK_ARENA_IMAGE,
        "checkpoint_identity": checkpoint_identity(), "configuration": configuration,
        "policy_identity": binding.identity(), "observation_adapter": interface_contract(),
        "action_adapter": "molmoact2_absolute_joint_position_normalized_closure_execute8_of15"}
    rights = {"scene_id": scene_plan["scene_id"], "task_id": scene_plan["task_id"],
        "scene_plan_digest": scene_plan["plan_digest"], "source_license": inventory["license"],
        "license_evidence_url": inventory["license_evidence_url"],
        "license_evidence_sha256": inventory["license_evidence_sha256"],
        "publisher_license_statement": inventory["license_statement"], "hf_license_metadata_present": False,
        "authorization_basis": "user_requested_public_policy_screening",
        "checkpoint_provider_use_status": "allowed", "checkpoint_redistribution_status": "allowed",
        "rights_ready": True, "secret_material_recorded": False}
    rights["rights_receipt_digest"] = canonical_digest(rights, digest_field="rights_receipt_digest")
    spec = {"schema_version": "native_task_arena_policy_canary_execution_spec.v1", "candidate_id": CANDIDATE_ID,
        "policy_interface_id": INTERFACE_ID, "evaluation_backend": "robolab.v1",
        "execution_authority": "internal_policy_canary_unqualified", "claim_ceiling": "diagnostic_policy_execution",
        "scene_id": scene_plan["scene_id"], "task_id": scene_plan["task_id"], "scene_plan_digest": scene_plan["plan_digest"],
        "prompt": concrete_droid_task_instruction({**scene_plan["task_spec"], "task_id": scene_plan["task_id"]}),
        "policy_endpoint": {"host": "127.0.0.1", "port": PORT}, "policy_spec": policy,
        "candidate_rights_binding": rights, "checkpoint_digest": canonical_digest(checkpoint_identity()),
        "checkpoint_identity_basis": "pinned_file_inventory_verified_before_worker_load", "weights_identity_verified": False,
        "runtime_identity": runtime, "runtime_identity_digest": canonical_digest(runtime),
        "worker_identity_requirement": "none", "require_observed_eef_support": False,
        "max_policy_queries": maximum_policy_queries_for_task_spec(scene_plan["task_spec"], open_loop_horizon=8),
        "open_loop_horizon": 8, "ranking_permitted": False, "qualification_permitted": False,
        "scene_promotion_permitted": False, "scoring_authority": "deterministic_simulator_state",
        "task_success_contract": dict(contract), "task_success_contract_digest": contract["contract_digest"]}
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    validate_execution_spec(spec)
    return spec


def validate_execution_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    binding = policy_binding(spec["policy_spec"])
    runtime = spec.get("runtime_identity") or {}
    rights = spec.get("candidate_rights_binding") or {}
    inventory = release_inventory()
    if (spec.get("candidate_id") != CANDIDATE_ID or spec.get("policy_interface_id") != INTERFACE_ID
            or spec.get("evaluation_backend") != "robolab.v1"
            or spec.get("checkpoint_digest") != canonical_digest(checkpoint_identity())
            or spec.get("policy_endpoint") != {"host": "127.0.0.1", "port": PORT} or spec.get("open_loop_horizon") != 8
            or spec.get("execution_spec_digest") != canonical_digest(spec, digest_field="execution_spec_digest")
            or runtime.get("checkpoint_identity") != checkpoint_identity() or runtime.get("policy_identity") != binding.identity()
            or spec.get("runtime_identity_digest") != canonical_digest(runtime)
            or binding.configuration_digest != canonical_digest(runtime.get("configuration") or {})
            or (runtime.get("configuration") or {}).get("scene_plan_digest") != spec.get("scene_plan_digest")
            or rights.get("rights_receipt_digest") != canonical_digest(rights, digest_field="rights_receipt_digest")
            or rights.get("rights_ready") is not True or rights.get("source_license") != "Apache-2.0"
            or rights.get("license_evidence_sha256") != inventory["license_evidence_sha256"]
            or rights.get("scene_plan_digest") != spec.get("scene_plan_digest")
            or rights.get("checkpoint_provider_use_status") != "allowed"):
        raise ValueError("molmoact2_droid_execution_binding_invalid")
    return dict(spec)
