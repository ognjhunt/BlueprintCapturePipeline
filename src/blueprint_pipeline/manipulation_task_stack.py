"""Build manipulation-task contracts and default policy eval packets."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .lucky_g1_reference_adapter import run_lucky_g1_reference_adapter
from .manipulation_physics_simulator_command import (
    run_mujoco_manipulation_physics,
    write_mujoco_tote_asset,
)


OBJECT_CONTRACT_SCHEMA_VERSION = "robot_eval_manipulation_object_contracts.v1"
TASK_REQUEST_SCHEMA_VERSION = "robot_eval_manipulation_task_request.v1"
POLICY_ADAPTER_SCHEMA_VERSION = "robot_eval_manipulation_policy_adapter_contract.v1"
DEFAULT_POLICY_TRACE_SCHEMA_VERSION = "robot_eval_default_manipulation_policy_trace.v1"
EVAL_REPORT_SCHEMA_VERSION = "robot_eval_manipulation_eval_report.v1"
STACK_MANIFEST_SCHEMA_VERSION = "robot_eval_manipulation_stack_manifest.v1"
POLICY_TIER_MATRIX_SCHEMA_VERSION = "robot_eval_manipulation_policy_tier_matrix.v1"

DEFAULT_OUTPUT_RELATIVE = "pipeline/simulation_automation/manipulation_task_stack"
DEFAULT_OBJECT_ID = "simready_tote_001"
DEFAULT_TASK_ID = "mobile_pick_carry_place_tote"
DEFAULT_POLICY_ID = "blueprint_default_phase_manipulation_policy"

CLAIM_BOUNDARY = {
    "manipulation_task_contract_ready": False,
    "default_policy_trace_generated": False,
    "team_policy_endpoint_execution_proven": False,
    "simulator_physics_execution_proven": False,
    "grasp_physics_validated": False,
    "carry_physics_validated": False,
    "robot_team_policy_quality_proven": False,
    "physical_robot_readiness_proven": False,
    "public_claim_upgrade_allowed": False,
}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "enabled"}


def _number(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _pose(value: Any, default: Sequence[float]) -> list[float]:
    if isinstance(value, Mapping):
        return _pose(value.get("xyz") or value.get("position") or value.get("pose"), default)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = list(value)
        if len(parts) >= 2:
            x = _number(parts[0], float(default[0]))
            y = _number(parts[1], float(default[1]))
            z = _number(parts[2], float(default[2])) if len(parts) >= 3 else float(default[2])
            yaw = _number(parts[3], 0.0) if len(parts) >= 4 else 0.0
            return [float(x), float(y), float(z), float(yaw)]
    return [float(default[0]), float(default[1]), float(default[2]), float(default[3])]


def _load_optional_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return {}
    payload = read_json_any(resolved)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _tote_template(
    *,
    object_id: str,
    object_asset_path: str | None,
    object_pose: Sequence[float],
) -> dict[str, Any]:
    return {
        "schema_version": "robot_eval_manipulation_object_contract.v1",
        "object_id": object_id,
        "object_class": "tote",
        "asset": {
            "asset_kind": "simready_or_mujoco_object",
            "uri": object_asset_path,
            "asset_status": "provided" if object_asset_path else "missing_asset_uri",
        },
        "pose_xyz_yaw": list(object_pose),
        "physical_properties": {
            "mass_kg": 1.25,
            "center_of_mass": [0.0, 0.0, 0.18],
            "inertia_status": "template_estimate",
            "static_friction": 0.7,
            "dynamic_friction": 0.55,
        },
        "geometry": {
            "bounding_box_m": [0.60, 0.40, 0.32],
            "collider": "box_proxy_plus_handle_affordances",
            "stable_support_faces": ["bottom"],
            "forbidden_grasp_zones": ["bottom_support_face", "sharp_or_unverified_edges"],
        },
        "affordances": [
            {
                "affordance_id": "left_rim",
                "type": "rim_or_handle_grasp",
                "side": "left",
                "local_position_xyz": [0.0, 0.22, 0.27],
                "approach_vector": [0.0, -1.0, 0.0],
                "allowed_end_effectors": ["left_hand", "right_hand", "bimanual"],
                "confidence": "template_inferred",
            },
            {
                "affordance_id": "right_rim",
                "type": "rim_or_handle_grasp",
                "side": "right",
                "local_position_xyz": [0.0, -0.22, 0.27],
                "approach_vector": [0.0, 1.0, 0.0],
                "allowed_end_effectors": ["left_hand", "right_hand", "bimanual"],
                "confidence": "template_inferred",
            },
        ],
        "success_thresholds": {
            "pregrasp_distance_m": 0.75,
            "lift_height_delta_m": 0.18,
            "minimum_hold_time_s": 1.0,
            "max_carry_tilt_degrees": 18.0,
            "max_drop_height_m": 0.08,
            "placement_tolerance_xy_m": 0.45,
            "placement_tolerance_yaw_rad": 0.7,
        },
        "contract_source": "class_template:tote.v1",
        "verification_status": "template_inferred_requires_asset_preflight",
    }


def _contract_from_payload(
    *,
    payload: Mapping[str, Any],
    object_id: str,
    object_asset_path: str | None,
    object_pose: Sequence[float],
) -> dict[str, Any]:
    contract = dict(payload)
    contract.setdefault("schema_version", "robot_eval_manipulation_object_contract.v1")
    contract.setdefault("object_id", object_id)
    contract.setdefault("pose_xyz_yaw", list(object_pose))
    asset = _mapping(contract.get("asset"))
    if object_asset_path:
        asset.setdefault("uri", object_asset_path)
    asset.setdefault("asset_kind", "simready_or_mujoco_object")
    asset.setdefault("asset_status", "provided" if asset.get("uri") else "missing_asset_uri")
    contract["asset"] = asset
    return contract


def build_manipulation_object_contract(
    *,
    object_id: str = DEFAULT_OBJECT_ID,
    object_class: str = "tote",
    object_asset_path: str | None = None,
    object_pose: Sequence[float] = (2.0, 4.0, 0.16, 0.0),
    object_contract: Mapping[str, Any] | None = None,
    allow_template_inference: bool = True,
) -> dict[str, Any]:
    if object_contract:
        contract = _contract_from_payload(
            payload=object_contract,
            object_id=object_id,
            object_asset_path=object_asset_path,
            object_pose=object_pose,
        )
    elif object_class == "tote" and allow_template_inference:
        contract = _tote_template(
            object_id=object_id,
            object_asset_path=object_asset_path,
            object_pose=object_pose,
        )
    else:
        contract = {
            "schema_version": "robot_eval_manipulation_object_contract.v1",
            "object_id": object_id,
            "object_class": object_class,
            "asset": {
                "asset_kind": "simready_or_mujoco_object",
                "uri": object_asset_path,
                "asset_status": "provided" if object_asset_path else "missing_asset_uri",
            },
            "pose_xyz_yaw": list(object_pose),
            "affordances": [],
            "success_thresholds": {},
            "contract_source": "missing_object_class_contract",
        }

    blockers: list[str] = []
    affordances = contract.get("affordances")
    thresholds = _mapping(contract.get("success_thresholds"))
    physical = _mapping(contract.get("physical_properties"))
    asset = _mapping(contract.get("asset"))
    geometry = _mapping(contract.get("geometry"))
    if not isinstance(affordances, Sequence) or isinstance(affordances, (str, bytes)) or not affordances:
        blockers.append("manipulation_affordances_missing")
    if not thresholds:
        blockers.append("manipulation_success_thresholds_missing")
    for key in ("mass_kg", "center_of_mass"):
        if key not in physical:
            blockers.append(f"physical_property_{key}_missing")
    if not geometry.get("bounding_box_m"):
        blockers.append("object_bounding_box_missing")
    if not asset.get("uri"):
        blockers.append("simready_object_asset_uri_missing")
    status = "ready" if not blockers else "blocked"
    return {
        **contract,
        "status": status,
        "blockers": blockers,
        "contract_ready_for_scored_manipulation": not blockers,
        "claim_boundary": {
            **CLAIM_BOUNDARY,
            "manipulation_task_contract_ready": not blockers,
            "public_claim_upgrade_allowed": False,
        },
    }


def _phase(
    phase_id: str,
    *,
    action: str,
    target: Any,
    expected_evidence: Sequence[str],
    success: bool,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "phase_id": phase_id,
        "action": action,
        "target": target,
        "expected_evidence": list(expected_evidence),
        "status": "completed" if success else "blocked",
        "success": success,
        "duration_s": float(duration_s),
    }


def _distance_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def build_manipulation_task_request(
    *,
    task_id: str = DEFAULT_TASK_ID,
    object_contract: Mapping[str, Any],
    start_pose: Sequence[float] = (0.0, 0.0, 0.793, 0.0),
    return_pose: Sequence[float] | None = None,
    instruction: str | None = None,
) -> dict[str, Any]:
    object_id = _string(object_contract.get("object_id")) or DEFAULT_OBJECT_ID
    return_pose = list(return_pose or start_pose)
    return {
        "schema_version": TASK_REQUEST_SCHEMA_VERSION,
        "task_id": task_id,
        "task_kind": "mobile_manipulation_pick_carry_place",
        "instruction": instruction
        or f"Navigate to {object_id}, pick it up, carry it to the start point, and place it down.",
        "object_id": object_id,
        "object_class": object_contract.get("object_class"),
        "start_pose_xyz_yaw": list(start_pose),
        "object_pose_xyz_yaw": object_contract.get("pose_xyz_yaw"),
        "return_pose_xyz_yaw": list(return_pose),
        "required_phases": [
            "navigate_to_object",
            "pregrasp_stance",
            "reach",
            "close_grip",
            "lift",
            "verify_grasp",
            "carry_to_return_pose",
            "place",
            "release",
            "verify_placement",
        ],
        "required_observations": [
            "base_pose",
            "head_or_scene_camera",
            "wrist_or_hand_camera",
            "end_effector_pose",
            "object_pose",
            "object_contact_state",
            "hand_joint_state",
        ],
        "success_metrics": [
            "phase_completion",
            "object_lifted",
            "grasp_held",
            "object_carried_without_drop",
            "placement_accuracy",
            "robot_fall",
            "forbidden_collision",
            "timeout",
        ],
    }


def build_policy_adapter_contract(
    *,
    task_request: Mapping[str, Any],
    object_contract: Mapping[str, Any],
    team_policy_endpoint: str | None = None,
    default_policy_enabled: bool = True,
    lucky_reference_enabled: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": POLICY_ADAPTER_SCHEMA_VERSION,
        "status": "ready",
        "task_id": task_request.get("task_id"),
        "object_id": object_contract.get("object_id"),
        "policy_submission_modes": [
            {
                "mode": "default_phase_policy",
                "tier": 1,
                "enabled": bool(default_policy_enabled),
                "policy_id": DEFAULT_POLICY_ID,
                "boundary": (
                    "Reference baseline for packet/evaluator development. It emits phase "
                    "and action traces, but does not prove physics by itself."
                ),
            },
            {
                "mode": "lucky_g1_reference_adapter",
                "tier": 2,
                "enabled": bool(lucky_reference_enabled),
                "source_repo": "https://github.com/luckyrobots/g1-manipulation-challenge",
                "expected_policy_assets": ["walker.onnx", "right_reacher.onnx", "grab_or_attach_logic"],
                "blueprint_fallback_reference": "mujoco_weld_grasp_tote_physics_command",
                "boundary": (
                    "Lucky-compatible G1 reference lane. When Lucky assets are present, "
                    "Pipeline can call that adapter; otherwise Blueprint's MuJoCo physics "
                    "command proves the evaluator's grasp/carry/place physics contract."
                ),
            },
            {
                "mode": "policy_api_endpoint",
                "tier": 3,
                "enabled": bool(team_policy_endpoint),
                "endpoint_url": team_policy_endpoint,
                "vla_adapter_supported": True,
                "compatible_external_policy_families": ["team_endpoint", "openpi_pi05_or_pi0_5", "pi0_6_style_vla"],
                "request_shape": {
                    "task_request": TASK_REQUEST_SCHEMA_VERSION,
                    "object_contract": "robot_eval_manipulation_object_contract.v1",
                    "observations": task_request.get("required_observations"),
                },
                "response_shape": {
                    "attempt_id": "string",
                    "status": "completed|failed|blocked",
                    "phase_trace": "array",
                    "action_trace": "array",
                    "metrics": "object",
                    "artifact_paths": "object",
                },
            },
        ],
        "required_trace_fields": [
            "phase_id",
            "action",
            "target",
            "status",
            "success",
            "expected_evidence",
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }

def build_default_manipulation_policy_trace(
    *,
    task_request: Mapping[str, Any],
    object_contract: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    contract_ready = object_contract.get("contract_ready_for_scored_manipulation") is True
    thresholds = _mapping(object_contract.get("success_thresholds"))
    start_pose = _pose(task_request.get("start_pose_xyz_yaw"), (0.0, 0.0, 0.793, 0.0))
    object_pose = _pose(task_request.get("object_pose_xyz_yaw"), (2.0, 4.0, 0.16, 0.0))
    return_pose = _pose(task_request.get("return_pose_xyz_yaw"), start_pose)
    approach_distance = _distance_xy(start_pose, object_pose)
    return_distance = _distance_xy(object_pose, return_pose)
    lift_delta = float(thresholds.get("lift_height_delta_m") or 0.18)
    placement_tolerance = float(thresholds.get("placement_tolerance_xy_m") or 0.45)
    can_attempt = contract_ready
    phase_specs = [
        _phase(
            "navigate_to_object",
            action="navigate",
            target=object_pose,
            expected_evidence=["base_pose_trace", "clearance_trace"],
            success=can_attempt,
            duration_s=max(2.0, approach_distance / 0.45),
        ),
        _phase(
            "pregrasp_stance",
            action="align_base_and_torso",
            target={"object_id": object_contract.get("object_id")},
            expected_evidence=["base_pose", "wrist_camera_frame"],
            success=can_attempt,
            duration_s=1.0,
        ),
        _phase(
            "reach",
            action="reach_to_affordance",
            target={"affordance_id": "right_rim"},
            expected_evidence=["end_effector_pose", "wrist_camera_frame"],
            success=can_attempt,
            duration_s=1.5,
        ),
        _phase(
            "close_grip",
            action="close_hand_or_gripper",
            target={"allowed_end_effectors": ["right_hand", "bimanual"]},
            expected_evidence=["hand_joint_state", "object_contact_state"],
            success=can_attempt,
            duration_s=0.7,
        ),
        _phase(
            "lift",
            action="lift_object",
            target={"lift_height_delta_m": lift_delta},
            expected_evidence=["object_pose_delta", "contact_state"],
            success=can_attempt,
            duration_s=1.2,
        ),
        _phase(
            "verify_grasp",
            action="hold_and_verify",
            target={"minimum_hold_time_s": thresholds.get("minimum_hold_time_s", 1.0)},
            expected_evidence=["object_attached_or_supported", "drop_event_absent"],
            success=can_attempt,
            duration_s=float(thresholds.get("minimum_hold_time_s") or 1.0),
        ),
        _phase(
            "carry_to_return_pose",
            action="navigate_with_object",
            target=return_pose,
            expected_evidence=["base_pose_trace", "object_pose_trace", "drop_event_absent"],
            success=can_attempt,
            duration_s=max(2.0, return_distance / 0.35),
        ),
        _phase(
            "place",
            action="lower_object_to_target",
            target={"target_pose": return_pose, "placement_tolerance_xy_m": placement_tolerance},
            expected_evidence=["object_pose", "end_effector_pose"],
            success=can_attempt,
            duration_s=1.4,
        ),
        _phase(
            "release",
            action="open_hand_or_gripper",
            target={"object_id": object_contract.get("object_id")},
            expected_evidence=["hand_joint_state", "object_contact_state"],
            success=can_attempt,
            duration_s=0.5,
        ),
        _phase(
            "verify_placement",
            action="verify_object_pose",
            target={"target_pose": return_pose, "placement_tolerance_xy_m": placement_tolerance},
            expected_evidence=["object_final_pose", "placement_error"],
            success=can_attempt,
            duration_s=0.8,
        ),
    ]
    action_trace = [
        {
            "step": index,
            "phase_id": phase["phase_id"],
            "command_kind": phase["action"],
            "target": phase["target"],
            "status": phase["status"],
            "success": phase["success"],
        }
        for index, phase in enumerate(phase_specs)
    ]
    success = can_attempt and all(phase["success"] for phase in phase_specs)
    blockers = list(object_contract.get("blockers") or []) if not contract_ready else []
    return {
        "schema_version": DEFAULT_POLICY_TRACE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed_reference_trace" if success else "blocked_contract_not_ready",
        "policy_id": DEFAULT_POLICY_ID,
        "policy_kind": "default_phase_mobile_manipulation",
        "default_test_policy": True,
        "robot_team_policy_execution_proven": False,
        "simulator_physics_execution_proven": False,
        "task_id": task_request.get("task_id"),
        "object_id": object_contract.get("object_id"),
        "attempt_id": f"default_manipulation_{task_request.get('task_id')}",
        "success": success,
        "blockers": blockers,
        "phase_trace": phase_specs,
        "action_trace": action_trace,
        "metrics": {
            "phase_count": len(phase_specs),
            "completed_phase_count": sum(1 for phase in phase_specs if phase["success"]),
            "approach_distance_m": round(approach_distance, 6),
            "return_distance_m": round(return_distance, 6),
            "estimated_cycle_time_s": round(sum(float(phase["duration_s"]) for phase in phase_specs), 6),
            "object_lifted": success,
            "grasp_held": success,
            "object_carried_without_drop": success,
            "placement_accuracy_m": 0.0 if success else None,
            "forbidden_collision_count": 0 if success else None,
            "object_drop_count": 0 if success else None,
        },
        "claim_boundary": {
            **CLAIM_BOUNDARY,
            "default_policy_trace_generated": True,
            "manipulation_task_contract_ready": contract_ready,
        },
    }


def build_policy_tier_matrix(
    *,
    task_request: Mapping[str, Any],
    object_contract: Mapping[str, Any],
    policy_adapter_contract: Mapping[str, Any],
    default_policy_trace: Mapping[str, Any],
    physics_output: Mapping[str, Any],
    lucky_reference_output: Mapping[str, Any] | None = None,
    generated_at: str,
) -> dict[str, Any]:
    endpoint_mode = next(
        (
            mode
            for mode in policy_adapter_contract.get("policy_submission_modes", [])
            if isinstance(mode, Mapping) and mode.get("mode") == "policy_api_endpoint"
        ),
        {},
    )
    default_complete = default_policy_trace.get("status") == "completed_reference_trace"
    physics_complete = physics_output.get("status") == "complete"
    lucky_reference_output = _mapping(lucky_reference_output)
    return {
        "schema_version": POLICY_TIER_MATRIX_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "ready",
        "task_id": task_request.get("task_id"),
        "object_id": object_contract.get("object_id"),
        "tiers": [
            {
            "tier": 1,
            "tier_id": "default_phase_policy",
            "status": "complete" if default_complete else "blocked",
            "ready": default_complete,
            "testable_in_pipeline": True,
            "policy_id": default_policy_trace.get("policy_id"),
                "default_policy_trace_generated": bool(default_policy_trace),
                "simulator_physics_execution_proven": False,
                "robot_team_policy_quality_proven": False,
            },
            {
            "tier": 2,
            "tier_id": "lucky_g1_reference_or_blueprint_physics",
            "status": "complete" if physics_complete else "blocked",
            "ready": physics_complete,
            "testable_in_pipeline": True,
            "source_repo": "https://github.com/luckyrobots/g1-manipulation-challenge",
                "official_lucky_adapter_status": lucky_reference_output.get("status")
                or "not_requested",
                "official_lucky_walker_reacher_policy_assets_executed": lucky_reference_output.get(
                    "official_lucky_walker_reacher_policy_assets_executed"
                )
                is True,
                "official_lucky_pick_place_physics_validated": _mapping(
                    lucky_reference_output.get("claim_boundary")
                ).get("official_lucky_pick_place_physics_validated")
                is True,
                "blueprint_physics_fallback_used": physics_complete,
                "manipulation_capable_g1_model_loaded": physics_output.get(
                    "manipulation_capable_g1_model_loaded"
                )
                is True,
                "controller_drove_actuators": physics_output.get("controller_drove_actuators")
                is True,
                "g1_reference_manipulation_physics_executed": physics_output.get(
                    "g1_reference_manipulation_physics_executed"
                )
                is True,
                "simulator_physics_execution_proven": physics_output.get(
                    "simulator_physics_execution_proven"
                )
                is True,
                "grasp_physics_validated": physics_output.get("grasp_physics_validated") is True,
                "carry_physics_validated": physics_output.get("carry_physics_validated") is True,
                "placement_physics_validated": physics_output.get("placement_physics_validated") is True,
                "full_unitree_g1_dexterous_hand_policy_proven": False,
                "artifacts": {
                    **_mapping(physics_output.get("artifacts")),
                    **_mapping(lucky_reference_output.get("artifacts")),
                },
            },
            {
            "tier": 3,
            "tier_id": "team_policy_endpoint_or_vla_adapter",
            "status": "ready_for_endpoint" if endpoint_mode.get("enabled") else "ready_unconfigured",
            "ready": endpoint_mode.get("enabled") is True,
            "testable_in_pipeline": True,
            "endpoint_configured": endpoint_mode.get("enabled") is True,
                "vla_adapter_supported": endpoint_mode.get("vla_adapter_supported") is True,
                "simulator_physics_execution_proven": False,
                "team_policy_endpoint_execution_proven": False,
                "robot_team_policy_quality_proven": False,
                "boundary": "Requires a live endpoint or submitted policy artifact before execution proof.",
            },
        ],
        "claim_boundary": {
            **CLAIM_BOUNDARY,
            "manipulation_task_contract_ready": object_contract.get("status") == "ready",
            "default_policy_trace_generated": default_complete,
            "simulator_physics_execution_proven": physics_output.get(
                "simulator_physics_execution_proven"
            )
            is True,
            "grasp_physics_validated": physics_output.get("grasp_physics_validated") is True,
            "carry_physics_validated": physics_output.get("carry_physics_validated") is True,
            "team_policy_endpoint_execution_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def build_manipulation_eval_report(
    *,
    task_request: Mapping[str, Any],
    object_contract: Mapping[str, Any],
    policy_adapter_contract: Mapping[str, Any],
    default_policy_trace: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    contract_ready = object_contract.get("contract_ready_for_scored_manipulation") is True
    default_ready = default_policy_trace.get("status") == "completed_reference_trace"
    blockers = []
    if not contract_ready:
        blockers.extend(str(item) for item in object_contract.get("blockers", []))
    if not default_ready:
        blockers.append("default_manipulation_policy_trace_not_complete")
    return {
        "schema_version": EVAL_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "ready_for_policy_comparison" if contract_ready else "blocked",
        "task_id": task_request.get("task_id"),
        "object_id": object_contract.get("object_id"),
        "contract_ready": contract_ready,
        "default_policy_available": default_ready,
        "team_policy_endpoint_supported": any(
            mode.get("mode") == "policy_api_endpoint"
            for mode in policy_adapter_contract.get("policy_submission_modes", [])
            if isinstance(mode, Mapping)
        ),
        "scorecard": {
            "default_policy_reference_score": 1.0 if default_ready else 0.0,
            "required_team_metrics": task_request.get("success_metrics"),
            "comparison_basis": (
                "Teams are compared against the same object contract, task request, "
                "phase trace schema, and scoring thresholds. Default policy is a reference "
                "baseline, not a robot-team-quality claim."
            ),
        },
        "blockers": blockers,
        "claim_boundary": {
            **CLAIM_BOUNDARY,
            "manipulation_task_contract_ready": contract_ready,
            "default_policy_trace_generated": bool(default_policy_trace),
            "public_claim_upgrade_allowed": False,
        },
    }


def build_manipulation_task_stack(
    *,
    capture_root: str | Path,
    output_dir: str | Path | None = None,
    object_id: str = DEFAULT_OBJECT_ID,
    object_class: str = "tote",
    object_asset_path: str | None = None,
    object_contract_path: str | Path | None = None,
    job_request_path: str | Path | None = None,
    start_pose: Sequence[float] = (0.0, 0.0, 0.793, 0.0),
    object_pose: Sequence[float] = (2.0, 4.0, 0.16, 0.0),
    return_pose: Sequence[float] | None = None,
    team_policy_endpoint: str | None = None,
    default_policy_enabled: bool = True,
    lucky_reference_enabled: bool = True,
    run_lucky_reference_adapter: bool = False,
    lucky_reference_root: str | Path | None = None,
    fetch_lucky_reference: bool = False,
    run_physics_sim: bool = True,
    allow_template_inference: bool = True,
) -> dict[str, Any]:
    root = Path(capture_root).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else root / DEFAULT_OUTPUT_RELATIVE
    ensure_dir(out_dir)
    generated_at = utc_now_iso()
    job_request = _load_optional_json(job_request_path)
    request_manipulation = _mapping(
        job_request.get("manipulation_task")
        or job_request.get("manipulationTask")
        or job_request.get("default_test_policy")
    )
    object_id = _string(request_manipulation.get("object_id")) or object_id
    object_class = _string(request_manipulation.get("object_class")) or object_class
    if request_manipulation.get("object_asset_path") or request_manipulation.get("objectAssetPath"):
        object_asset_path = _string(
            request_manipulation.get("object_asset_path") or request_manipulation.get("objectAssetPath")
        )
    start_pose = _pose(request_manipulation.get("start_pose"), start_pose)
    object_pose = _pose(request_manipulation.get("object_pose"), object_pose)
    return_pose = _pose(request_manipulation.get("return_pose"), return_pose or start_pose)
    if request_manipulation.get("team_policy_endpoint"):
        team_policy_endpoint = _string(request_manipulation.get("team_policy_endpoint"))
    if request_manipulation.get("run_physics_sim") is not None:
        run_physics_sim = _boolish(request_manipulation.get("run_physics_sim"))
    object_contract_payload = _load_optional_json(object_contract_path)
    if not object_contract_payload:
        inline_contract = request_manipulation.get("object_contract") or request_manipulation.get("objectContract")
        object_contract_payload = dict(inline_contract) if isinstance(inline_contract, Mapping) else {}

    generated_object_asset: dict[str, Any] = {}
    if not object_asset_path and object_class == "tote" and allow_template_inference:
        generated_object_asset = write_mujoco_tote_asset(
            output_dir=out_dir / "assets",
            object_id=object_id,
            object_pose=object_pose,
            generated_at=generated_at,
        )
        object_asset_path = str(generated_object_asset["asset_path"])

    object_contract = build_manipulation_object_contract(
        object_id=object_id,
        object_class=object_class,
        object_asset_path=object_asset_path,
        object_pose=object_pose,
        object_contract=object_contract_payload,
        allow_template_inference=allow_template_inference,
    )
    task_request = build_manipulation_task_request(
        task_id=_string(request_manipulation.get("task_id")) or DEFAULT_TASK_ID,
        object_contract=object_contract,
        start_pose=start_pose,
        return_pose=return_pose,
        instruction=_string(request_manipulation.get("instruction")) or None,
    )
    policy_adapter_contract = build_policy_adapter_contract(
        task_request=task_request,
        object_contract=object_contract,
        team_policy_endpoint=team_policy_endpoint,
        default_policy_enabled=default_policy_enabled,
        lucky_reference_enabled=lucky_reference_enabled,
    )
    default_policy_trace = (
        build_default_manipulation_policy_trace(
            task_request=task_request,
            object_contract=object_contract,
            generated_at=generated_at,
        )
        if default_policy_enabled
        else {
            "schema_version": DEFAULT_POLICY_TRACE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "not_requested",
            "policy_id": DEFAULT_POLICY_ID,
            "default_test_policy": False,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    )
    if run_physics_sim and object_contract.get("status") == "ready":
        physics_output = run_mujoco_manipulation_physics(
            capture_root=root,
            output_dir=out_dir / "mujoco_manipulation_physics",
            object_id=object_id,
            task_id=_string(task_request.get("task_id")) or DEFAULT_TASK_ID,
            object_pose=object_pose,
            return_pose=return_pose,
            object_mass_kg=float(_mapping(object_contract.get("physical_properties")).get("mass_kg") or 1.25),
        )
    else:
        physics_output = {
            "schema_version": "mujoco_manipulation_physics_output.v1",
            "generated_at": generated_at,
            "status": "not_requested" if not run_physics_sim else "blocked_contract_not_ready",
            "simulator_physics_execution_proven": False,
            "grasp_physics_validated": False,
            "carry_physics_validated": False,
            "placement_physics_validated": False,
            "artifacts": {},
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    if run_lucky_reference_adapter and lucky_reference_enabled:
        lucky_reference_output = run_lucky_g1_reference_adapter(
            capture_root=root,
            output_dir=out_dir / "lucky_g1_reference_adapter",
            lucky_root=lucky_reference_root,
            fetch_if_missing=fetch_lucky_reference,
        )
    else:
        lucky_reference_output = {
            "schema_version": "lucky_g1_reference_adapter_manifest.v1",
            "generated_at": generated_at,
            "status": "not_requested" if not run_lucky_reference_adapter else "disabled",
            "official_lucky_walker_reacher_policy_assets_executed": False,
            "lucky_g1_reference_adapter_ready": False,
            "artifacts": {},
            "claim_boundary": {
                "official_lucky_walker_reacher_policy_assets_executed": False,
                "official_lucky_pick_place_physics_validated": False,
                "blueprint_tote_task_validated_by_lucky_assets": False,
            },
        }

    policy_tier_matrix = build_policy_tier_matrix(
        task_request=task_request,
        object_contract=object_contract,
        policy_adapter_contract=policy_adapter_contract,
        default_policy_trace=default_policy_trace,
        physics_output=physics_output,
        lucky_reference_output=lucky_reference_output,
        generated_at=generated_at,
    )
    eval_report = build_manipulation_eval_report(
        task_request=task_request,
        object_contract=object_contract,
        policy_adapter_contract=policy_adapter_contract,
        default_policy_trace=default_policy_trace,
        generated_at=generated_at,
    )

    object_contracts = {
        "schema_version": OBJECT_CONTRACT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "ready" if object_contract.get("status") == "ready" else "blocked",
        "object_contract_count": 1,
        "contracts": [object_contract],
        "claim_boundary": {
            **CLAIM_BOUNDARY,
            "manipulation_task_contract_ready": object_contract.get("status") == "ready",
        },
    }
    artifacts = {
        "manipulation_object_contracts": str(out_dir / "manipulation_object_contracts.json"),
        "manipulation_task_request": str(out_dir / "manipulation_task_request.json"),
        "manipulation_policy_adapter_contract": str(
            out_dir / "manipulation_policy_adapter_contract.json"
        ),
        "manipulation_policy_tier_matrix": str(out_dir / "manipulation_policy_tier_matrix.json"),
        "default_manipulation_policy_trace": str(out_dir / "default_manipulation_policy_trace.json"),
        "manipulation_eval_report": str(out_dir / "manipulation_eval_report.json"),
        "manipulation_physics_output": str(
            physics_output.get("output_path") or out_dir / "mujoco_manipulation_physics" / "manipulation_physics_output.json"
        ),
    }
    if generated_object_asset:
        artifacts["mujoco_tote_object_asset"] = str(generated_object_asset["asset_path"])
        artifacts["mujoco_tote_object_asset_manifest"] = str(generated_object_asset["manifest_path"])
    if physics_output.get("artifacts", {}).get("manipulation_contact_manifest"):
        artifacts["manipulation_contact_manifest"] = str(
            physics_output["artifacts"]["manipulation_contact_manifest"]
        )
    if physics_output.get("artifacts", {}).get("manipulation_video_manifest"):
        artifacts["manipulation_video_manifest"] = str(
            physics_output["artifacts"]["manipulation_video_manifest"]
        )
    if physics_output.get("artifacts", {}).get("manipulation_overview_video"):
        artifacts["manipulation_overview_video"] = str(
            physics_output["artifacts"]["manipulation_overview_video"]
        )
    if physics_output.get("artifacts", {}).get("mujoco_tote_visual_mesh"):
        artifacts["mujoco_tote_visual_mesh"] = str(
            physics_output["artifacts"]["mujoco_tote_visual_mesh"]
        )
    if physics_output.get("artifacts", {}).get("mujoco_g1_manipulation_model_manifest"):
        artifacts["mujoco_g1_manipulation_model_manifest"] = str(
            physics_output["artifacts"]["mujoco_g1_manipulation_model_manifest"]
        )
    if lucky_reference_output.get("output_path"):
        artifacts["lucky_g1_reference_adapter_manifest"] = str(
            lucky_reference_output["output_path"]
        )
    for key, value in _mapping(lucky_reference_output.get("artifacts")).items():
        if value:
            artifacts[key] = str(value)
    physics_complete = physics_output.get("status") == "complete"
    manifest = {
        "schema_version": STACK_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "complete"
        if eval_report.get("status") == "ready_for_policy_comparison" and physics_complete
        else "blocked",
        "capture_root": str(root),
        "output_dir": str(out_dir),
        "task_id": task_request.get("task_id"),
        "object_id": object_contract.get("object_id"),
        "object_class": object_contract.get("object_class"),
        "default_policy_available": default_policy_trace.get("status") == "completed_reference_trace",
        "lucky_g1_reference_adapter_ready": bool(lucky_reference_enabled),
        "team_policy_endpoint_configured": bool(team_policy_endpoint),
        "manipulation_capable_g1_model_loaded": physics_output.get(
            "manipulation_capable_g1_model_loaded"
        )
        is True,
        "controller_drove_actuators": physics_output.get("controller_drove_actuators") is True,
        "g1_reference_manipulation_physics_executed": physics_output.get(
            "g1_reference_manipulation_physics_executed"
        )
        is True,
        "official_lucky_adapter_status": lucky_reference_output.get("status"),
        "official_lucky_walker_reacher_policy_assets_executed": lucky_reference_output.get(
            "official_lucky_walker_reacher_policy_assets_executed"
        )
        is True,
        "official_lucky_pick_place_physics_validated": _mapping(
            lucky_reference_output.get("claim_boundary")
        ).get("official_lucky_pick_place_physics_validated")
        is True,
        "simulator_physics_execution_proven": physics_output.get("simulator_physics_execution_proven") is True,
        "grasp_physics_validated": physics_output.get("grasp_physics_validated") is True,
        "carry_physics_validated": physics_output.get("carry_physics_validated") is True,
        "placement_physics_validated": physics_output.get("placement_physics_validated") is True,
        "artifacts": artifacts,
        "blockers": []
        if eval_report.get("status") == "ready_for_policy_comparison" and physics_complete
        else list(eval_report.get("blockers") or [])
        + ([] if physics_complete else ["manipulation_physics_simulator_not_complete"]),
        "claim_boundary": {
            **CLAIM_BOUNDARY,
            "manipulation_task_contract_ready": object_contract.get("status") == "ready",
            "default_policy_trace_generated": bool(default_policy_trace),
            "manipulation_capable_g1_proxy_model_executed": physics_output.get(
                "manipulation_capable_g1_model_loaded"
            )
            is True,
            "controller_drove_actuators": physics_output.get("controller_drove_actuators")
            is True,
            "official_lucky_walker_reacher_policy_assets_executed": lucky_reference_output.get(
                "official_lucky_walker_reacher_policy_assets_executed"
            )
            is True,
            "simulator_physics_execution_proven": physics_output.get(
                "simulator_physics_execution_proven"
            )
            is True,
            "grasp_physics_validated": physics_output.get("grasp_physics_validated") is True,
            "carry_physics_validated": physics_output.get("carry_physics_validated") is True,
            "robot_team_policy_quality_proven": False,
            "physical_robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    write_json(Path(artifacts["manipulation_object_contracts"]), object_contracts)
    write_json(Path(artifacts["manipulation_task_request"]), task_request)
    write_json(Path(artifacts["manipulation_policy_adapter_contract"]), policy_adapter_contract)
    write_json(Path(artifacts["manipulation_policy_tier_matrix"]), policy_tier_matrix)
    write_json(Path(artifacts["default_manipulation_policy_trace"]), default_policy_trace)
    write_json(Path(artifacts["manipulation_eval_report"]), eval_report)
    manifest_path = out_dir / "manipulation_stack_manifest.json"
    write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--object-id", default=DEFAULT_OBJECT_ID)
    parser.add_argument("--object-class", default="tote")
    parser.add_argument("--object-asset-path")
    parser.add_argument("--object-contract")
    parser.add_argument("--job-request")
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--start-z", type=float, default=0.793)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--object-x", type=float, default=2.0)
    parser.add_argument("--object-y", type=float, default=4.0)
    parser.add_argument("--object-z", type=float, default=0.16)
    parser.add_argument("--object-yaw", type=float, default=0.0)
    parser.add_argument("--return-x", type=float)
    parser.add_argument("--return-y", type=float)
    parser.add_argument("--return-z", type=float, default=0.793)
    parser.add_argument("--return-yaw", type=float, default=0.0)
    parser.add_argument("--team-policy-endpoint")
    parser.add_argument("--disable-default-policy", action="store_true")
    parser.add_argument("--disable-lucky-reference", action="store_true")
    parser.add_argument("--run-lucky-reference-adapter", action="store_true")
    parser.add_argument("--lucky-reference-root")
    parser.add_argument("--fetch-lucky-reference", action="store_true")
    parser.add_argument("--no-physics-sim", action="store_true")
    parser.add_argument("--no-template-inference", action="store_true")
    args = parser.parse_args(argv)
    start_pose = [args.start_x, args.start_y, args.start_z, args.start_yaw]
    object_pose = [args.object_x, args.object_y, args.object_z, args.object_yaw]
    return_pose = (
        [args.return_x, args.return_y, args.return_z, args.return_yaw]
        if args.return_x is not None and args.return_y is not None
        else start_pose
    )
    result = build_manipulation_task_stack(
        capture_root=args.capture_root,
        output_dir=args.output_dir,
        object_id=args.object_id,
        object_class=args.object_class,
        object_asset_path=args.object_asset_path,
        object_contract_path=args.object_contract,
        job_request_path=args.job_request,
        start_pose=start_pose,
        object_pose=object_pose,
        return_pose=return_pose,
        team_policy_endpoint=args.team_policy_endpoint,
        default_policy_enabled=not args.disable_default_policy,
        lucky_reference_enabled=not args.disable_lucky_reference,
        run_lucky_reference_adapter=args.run_lucky_reference_adapter,
        lucky_reference_root=args.lucky_reference_root,
        fetch_lucky_reference=args.fetch_lucky_reference,
        run_physics_sim=not args.no_physics_sim,
        allow_template_inference=not args.no_template_inference,
    )
    print(result["manifest_path"])
    print(result["status"])
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
