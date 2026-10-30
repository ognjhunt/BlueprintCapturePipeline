"""Pinned Cosmos Nano DROID execution through the ordinary beta simulator.

The historical WAM client retains a 16-row conditioning prefix. Simulator
execution instead retains all 32 native rows and executes eight at 15 Hz. The
three real policy cameras and the existing 224-pixel transport preprocessing
remain explicit; NVIDIA's pinned server composes and resizes them to 540x640.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .cosmos_edge_droid_policy_runtime import COSMOS_FRAMEWORK_REVISION, canonical_sha256
from .cosmos_nano_droid_policy_runtime import MODEL_ID, MODEL_REVISION, CosmosNanoDroidPolicySpec
from .cosmos_edge_beta_runtime import CosmosEdgeSimulatorClient
from .decision_evidence_contracts import canonical_digest

CANDIDATE_ID = "cosmos3_nano_policy_droid"
INTERFACE_ID = "cosmos_nano_droid.v1"
CHECKPOINT_DIGEST = "sha256:1f87ceffb8f1db9e0a3b6cf76e23d64e2d01047c8e39ffe92691f386aa2a3e88"
SNAPSHOT_MANIFEST_SHA256 = "1f87ceffb8f1db9e0a3b6cf76e23d64e2d01047c8e39ffe92691f386aa2a3e88"
SNAPSHOT_STAGED_NAME = "cosmos_nano_policy_snapshot_manifest.json"
MINIMUM_GPU_MEMORY_GB = 80  # Conservative policy+Isaac allocation floor, not a measured model minimum.


def snapshot_manifest_path() -> Path:
    return Path(__file__).parent / "_vendor" / "cosmos-nano-droid-snapshot.v1.json"


def verified_snapshot_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {key: val for key, val in value.items() if key != "manifest_sha256"}
    if (value.get("manifest_sha256") != SNAPSHOT_MANIFEST_SHA256
            or canonical_sha256(body) != SNAPSHOT_MANIFEST_SHA256
            or value.get("model_id") != MODEL_ID or value.get("model_revision") != MODEL_REVISION):
        raise ValueError("cosmos_nano_beta_snapshot_manifest_mismatch")
    return value


def interface_contract() -> dict[str, Any]:
    return {
        "policy_interface_id": INTERFACE_ID,
        "robot_profile_id": "droid_franka_panda_robotiq_2f85_v1",
        "observation_profile_id": "droid_three_camera_robot_state_v1",
        "action_schema_id": "droid_absolute_joint_position_v1",
        "control_hz": 15, "action_horizon": 32, "execution_horizon": 8,
        "camera_roles": ["external", "wrist", "exterior_right"],
        "image_width": 224, "image_height": 224, "image_encoding": "rgb_uint8",
        "image_preprocessing": "resize_with_padding", "history_frames": 1,
        "normalization_digest": None,
    }


def candidate_execution_spec(*, scene_plan: Mapping[str, Any], contract: Mapping[str, Any],
                             source_commit: str) -> dict[str, Any]:
    from .adp009d_policy_episode import maximum_policy_queries_for_task_spec
    from .droid_policy_canary_embodiment import concrete_droid_task_instruction
    from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE

    roles = {row.get("role") for row in scene_plan.get("cameras", []) if row.get("policy_input") is True}
    if not set(interface_contract()["camera_roles"]).issubset(roles):
        raise ValueError("cosmos_nano_required_policy_camera_unavailable")
    verified_snapshot_manifest(snapshot_manifest_path())
    policy = CosmosNanoDroidPolicySpec(snapshot_manifest_sha256=SNAPSHOT_MANIFEST_SHA256)
    policy.validate()
    from .cosmos_droid_bundle import runtime_contract
    runtime = {
        "provisioning_identity": runtime_contract(CANDIDATE_ID),
        "source_commit": source_commit, "container_image": NATIVE_TASK_ARENA_IMAGE,
        "candidate_source": {"repository": "https://github.com/NVIDIA/cosmos-framework",
                             "revision": COSMOS_FRAMEWORK_REVISION},
        "policy_identity": policy.server_metadata(), "checkpoint_inventory_digest": CHECKPOINT_DIGEST,
        "observation_adapter": interface_contract(), "action_adapter": "cosmos_native32_execute8_joint_position",
    }
    rights = {"scene_id": scene_plan["scene_id"], "task_id": scene_plan["task_id"],
              "scene_plan_digest": scene_plan["plan_digest"], "source_license": "OpenMDW-1.1",
              "checkpoint_provider_use_status": "allowed", "checkpoint_redistribution_status": "allowed",
              "rights_ready": True, "secret_material_recorded": False}
    rights["rights_receipt_digest"] = canonical_digest(rights, digest_field="rights_receipt_digest")
    spec = {
        "schema_version": "native_task_arena_policy_canary_execution_spec.v1",
        "candidate_id": CANDIDATE_ID, "policy_interface_id": INTERFACE_ID,
        "evaluation_backend": "robolab.v1", "execution_authority": "internal_policy_canary_unqualified",
        "claim_ceiling": "diagnostic_policy_execution", "scene_id": scene_plan["scene_id"],
        "task_id": scene_plan["task_id"], "scene_plan_digest": scene_plan["plan_digest"],
        "prompt": concrete_droid_task_instruction({**scene_plan["task_spec"], "task_id": scene_plan["task_id"]}),
        "policy_endpoint": {"host": "127.0.0.1", "port": 8000}, "policy_spec": asdict(policy),
        "candidate_rights_binding": rights, "checkpoint_digest": CHECKPOINT_DIGEST,
        "runtime_identity": runtime, "runtime_identity_digest": canonical_digest(runtime),
        "worker_identity_requirement": "none", "require_observed_eef_support": False,
        "max_policy_queries": maximum_policy_queries_for_task_spec(scene_plan["task_spec"], open_loop_horizon=8),
        "open_loop_horizon": 8, "ranking_permitted": False, "qualification_permitted": False,
        "scene_promotion_permitted": False, "scoring_authority": "deterministic_simulator_state",
        "task_success_contract": dict(contract), "task_success_contract_digest": contract["contract_digest"],
    }
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    validate_execution_spec(spec)
    return spec


def validate_execution_spec(spec: Mapping[str, Any]) -> None:
    policy = CosmosNanoDroidPolicySpec(**dict(spec["policy_spec"]))
    policy.validate()
    from .cosmos_droid_bundle import runtime_contract
    expected = CosmosNanoDroidPolicySpec(snapshot_manifest_sha256=SNAPSHOT_MANIFEST_SHA256)
    if (spec.get("runtime_identity", {}).get("provisioning_identity") != runtime_contract(CANDIDATE_ID)
            or spec.get("candidate_id") != CANDIDATE_ID or spec.get("policy_interface_id") != INTERFACE_ID
            or spec.get("evaluation_backend") != "robolab.v1"
            or spec.get("checkpoint_digest") != CHECKPOINT_DIGEST
            or policy.snapshot_manifest_sha256 != SNAPSHOT_MANIFEST_SHA256
            or policy.server_metadata() != expected.server_metadata()
            or spec.get("policy_endpoint") != {"host": "127.0.0.1", "port": 8000}
            or spec.get("open_loop_horizon") != 8
            or spec.get("execution_spec_digest") != canonical_digest(spec, digest_field="execution_spec_digest")):
        raise ValueError("cosmos_nano_beta_execution_binding_invalid")


class CosmosNanoSimulatorClient(CosmosEdgeSimulatorClient):
    spec_type = CosmosNanoDroidPolicySpec
    policy_id = MODEL_ID
