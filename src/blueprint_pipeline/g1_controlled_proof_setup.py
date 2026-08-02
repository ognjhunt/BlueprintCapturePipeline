"""Generate the controlled Unitree G1 evidence packet for live robot-eval proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, optional_read_json, utc_now_iso, write_json, write_text


G1_CONTROLLED_PROOF_SETUP_SCHEMA_VERSION = "g1_controlled_proof_setup.v1"
CONTROLLED_FIELD_ANCHOR_REQUEST_SCHEMA_VERSION = "controlled_field_anchor_request_packet.v1"
ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION = "accepted_real_world_anchor.v1"
CONTROLLED_ANCHOR_JOIN_KEYS = (
    "scenario_eval_run_id",
    "policy_id",
    "task_id",
    "scenario_variation_instance_id",
)
OWNER_EVIDENCE_CLAIM_REQUIREMENTS = (
    "real_robot_pov",
    "action_logs",
    "timestamp_alignment",
    "hardware_validation",
    "contact_collision_logs",
    "policy_metrics",
    "robot_team_review",
)
DEFAULT_ROBOT_PROFILE_ID = "unitree_g1_humanoid"
DEFAULT_ROBOT_MAKE_MODEL = "Unitree G1"
DEFAULT_POLICY_ID = "unitree_rl_gym_g1_mujoco_policy_candidate"
OFFICIAL_UNITREE_G1_POLICY_CANDIDATE_SCHEMA_VERSION = (
    "official_unitree_g1_policy_candidate.v1"
)
DEFAULT_LOW_COST_GPU_TYPE_ID = "NVIDIA RTX A4000"
DEFAULT_PROVIDER_MAX_BUDGET_USD = 2.0
DEFAULT_PROVIDER_HARD_TIMEOUT_SECONDS = 120
DEFAULT_PROVIDER_WATCHDOG_TTL_SECONDS = 180
UNITREE_RL_GYM_MAIN_REF = "276801e46c5d433564f24658bac64f254b7d2d4b"
UNITREE_RL_LAB_MAIN_REF = "4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"
UNITREE_MUJOCO_MAIN_REF = "ae6a8403e272733e9996ef59990880330496177f"


OFFICIAL_UNITREE_G1_POLICY_SOURCES = (
    {
        "name": "unitree_rl_lab",
        "url": "https://github.com/unitreerobotics/unitree_rl_lab",
        "recommended_ref": UNITREE_RL_LAB_MAIN_REF,
        "license": "Apache-2.0",
        "candidate_use": "G1 29-DOF RL policy training, MuJoCo sim2sim, and sim2real deployment candidate.",
        "relevant_paths": [
            "deploy/robots/g1_29dof",
        ],
        "primary_source_references": [
            "README: supports Unitree G1-29dof robots and documents Unitree-G1-29dof-Velocity play/train tasks",
        ],
        "source_note": (
            "Official Unitree repository describes compiling g1_29dof, running "
            "unitree_mujoco for sim2sim, then using g1_ctrl for sim2real."
        ),
    },
    {
        "name": "unitree_rl_gym",
        "url": "https://github.com/unitreerobotics/unitree_rl_gym",
        "recommended_ref": UNITREE_RL_GYM_MAIN_REF,
        "license": "BSD-3-Clause",
        "candidate_use": "G1 MuJoCo and physical deployment workflow candidate.",
        "relevant_paths": [
            "deploy/deploy_mujoco",
            "deploy/deploy_real",
            "deploy/pre_train/g1",
        ],
        "primary_source_references": [
            "README: supports G1 and documents Train -> Play -> Sim2Sim -> Sim2Real",
            "deploy/deploy_real/README.md: physical deploy supports Unitree G1 with g1.yaml",
        ],
        "source_note": (
            "Official Unitree repository documents Train -> Play -> Sim2Sim -> "
            "Sim2Real and a G1 MuJoCo config."
        ),
    },
    {
        "name": "unitree_mujoco",
        "url": "https://github.com/unitreerobotics/unitree_mujoco",
        "recommended_ref": UNITREE_MUJOCO_MAIN_REF,
        "license": "BSD-3-Clause",
        "candidate_use": "Official Unitree SDK2/MuJoCo bridge for controller sim-to-real verification.",
        "relevant_paths": [
            "simulate",
            "simulate_python",
            "unitree_robots/g1",
        ],
        "primary_source_references": [
            "Repository: official Unitree MuJoCo bridge used for SDK2/MuJoCo controller verification",
        ],
        "source_note": (
            "Official Unitree MuJoCo simulator is built around Unitree SDK2 "
            "messages and includes G1 low-level message support."
        ),
    },
)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_string(*values: Any, default: str) -> str:
    for value in values:
        text = _string(value)
        if text:
            return text
    return default


def _safe_id(value: str, fallback: str) -> str:
    text = _string(value)
    if not text:
        return fallback
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in text).strip("-") or fallback


def _find_job_request_path(capture_root: Path, job_id: str | None) -> Path | None:
    jobs_root = capture_root / "pipeline" / "robot_eval_jobs"
    if job_id:
        candidate = jobs_root / job_id / "job_request.json"
        return candidate if candidate.is_file() else None
    if not jobs_root.is_dir():
        return None
    for candidate in sorted(jobs_root.glob("*/job_request.json")):
        return candidate
    return None


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _webapp_route_proof_score(payload: Mapping[str, Any] | None) -> tuple[int, int, int, str]:
    if not payload:
        return (-1, -1, -1, "")
    boundary = _mapping(payload.get("proof_boundary"))
    pipeline_forward = _mapping(payload.get("pipeline_forward"))
    pipeline_intake = _mapping(payload.get("pipeline_intake"))
    return (
        1 if payload.get("status") == "forwarded_to_pipeline_intake" else 0,
        1 if boundary.get("production_live_webapp_forwarding_proven") is True else 0,
        1
        if pipeline_forward.get("accepted") is True
        and pipeline_intake.get("accepted") is True
        and not _as_list(pipeline_intake.get("input_blockers"))
        else 0,
        _string(payload.get("generated_at")),
    )


def _find_webapp_route_job_request(
    capture_root: Path,
    job_id: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    proof_dir = capture_root / "pipeline" / "webapp_route_forwarding_proof"
    default_path = proof_dir / "webapp_route_forwarding_proof.json"
    candidates = [default_path]
    if proof_dir.is_dir():
        for candidate in sorted(proof_dir.glob("webapp_route_forwarding_proof*.json")):
            if candidate not in candidates:
                candidates.append(candidate)

    selected_request: dict[str, Any] | None = None
    selected_path: str | None = None
    selected_score = (-1, -1, -1, "")
    for candidate in candidates:
        payload = optional_read_json(candidate)
        if not payload:
            continue
        request = _mapping(payload.get("job_request"))
        if not request:
            continue
        if job_id and _string(request.get("job_id")) != _string(job_id):
            continue
        score = _webapp_route_proof_score(payload)
        if score > selected_score:
            selected_request = request
            selected_path = str(candidate)
            selected_score = score
    return selected_request, selected_path


def _job_context(capture_root: Path, job_id: str | None) -> dict[str, Any]:
    request_path = _find_job_request_path(capture_root, job_id)
    request = optional_read_json(request_path) if request_path else None
    job_request_source = "robot_eval_jobs" if request else "missing"
    route_request_path: str | None = None
    if not request:
        route_request, route_request_path = _find_webapp_route_job_request(capture_root, job_id)
        if route_request:
            request = route_request
            job_request_source = "webapp_route_forwarding_proof"
    requested_tasks = request.get("requested_tasks") if isinstance(request, Mapping) else None
    first_task = (
        dict(requested_tasks[0])
        if isinstance(requested_tasks, list) and isinstance(requested_tasks[0], Mapping)
        else {}
    )
    scenario_ids = first_task.get("scenario_ids") if isinstance(first_task, Mapping) else None
    first_scenario_id = (
        _string(scenario_ids[0])
        if isinstance(scenario_ids, list) and scenario_ids
        else ""
    )
    source = _mapping(request.get("source")) if isinstance(request, Mapping) else {}
    selection_state = _mapping(source.get("selection_state"))
    site_package = _mapping(request.get("site_package")) if isinstance(request, Mapping) else {}
    resolved_job_id = _first_string(
        job_id,
        request.get("job_id") if isinstance(request, Mapping) else None,
        default="robot-eval-job-id",
    )
    task_id = _first_string(
        first_task.get("task_id"),
        selection_state.get("task_id"),
        default="walk_to_target",
    )
    scenario_id = _first_string(
        first_scenario_id,
        selection_state.get("scenario_id"),
        default=f"{_safe_id(task_id, 'task')}_scenario",
    )
    return {
        "job_request_path": str(request_path) if request_path else route_request_path,
        "job_request_found": bool(
            request_path and request_path.is_file() or route_request_path
        ),
        "job_request_source": job_request_source,
        "job_id": resolved_job_id,
        "task_id": task_id,
        "scenario_id": scenario_id,
        "scenario_variation_instance_id": scenario_id,
        "scenario_eval_run_id": f"{_safe_id(resolved_job_id, 'job')}-{_safe_id(scenario_id, 'scenario')}",
        "robot_profile_id": _first_string(
            selection_state.get("robot_profile_id"),
            default=DEFAULT_ROBOT_PROFILE_ID,
        ),
        "site_slug": _first_string(
            site_package.get("site_slug"),
            selection_state.get("site_slug"),
            default="site-slug",
        ),
        "site_submission_id": _first_string(
            site_package.get("site_submission_id"),
            selection_state.get("site_submission_id"),
            default="site-submission-id",
        ),
        "buyer_request_id": _first_string(
            site_package.get("buyer_request_id"),
            selection_state.get("buyer_request_id"),
            request.get("buyer_request_id") if isinstance(request, Mapping) else None,
            default="buyer-request-id",
        ),
        "capture_job_id": _first_string(
            site_package.get("capture_job_id"),
            selection_state.get("capture_job_id"),
            default="capture-job-id",
        ),
        "capture_id": _first_string(
            site_package.get("capture_id"),
            selection_state.get("capture_id"),
            default=capture_root.name,
        ),
    }


def _owner_attestation_template(role: str) -> dict[str, Any]:
    return {
        "status": "not_signed",
        "role": role,
        "attested_by": "<operator-or-owner-id>",
        "statement": "<signed statement that the referenced evidence is from this physical Unitree G1 run>",
        "accepted_claim_boundary": (
            "This attests only to the referenced physical run evidence, not to public readiness."
        ),
        "signed_at_utc": "<timestamp>",
    }


def _proof_boundary() -> dict[str, bool]:
    return {
        "template_is_not_proof": True,
        "generated_world_rank_fidelity_result_proven": False,
        "non_ranking_operational_claim_validated": False,
        "real_robot_pov_evidence_proven": False,
        "robot_team_policy_performance_proven": False,
        "production_runpod_worker_execution_proven": False,
        "customer_through_website_testing_ready": False,
        "public_claim_upgrade_allowed": False,
    }


def _physical_robot_run_manifest(context: Mapping[str, Any]) -> dict[str, Any]:
    job_id = _string(context.get("job_id"))
    return {
        "schema_version": "physical_robot_run_package.v1",
        "status": "template_external_operator_input_required",
        "job_id": job_id,
        "run_id": f"unitree-g1-controlled-run-{_safe_id(job_id, 'job')}",
        "robot_make_model": DEFAULT_ROBOT_MAKE_MODEL,
        "robot_profile_id": DEFAULT_ROBOT_PROFILE_ID,
        "robot_serial_or_fleet_id": "<unitree-g1-serial-or-fleet-id>",
        "site_or_lab_location_id": "<controlled-test-site-or-lab-id>",
        "operator_attestation": _owner_attestation_template("operator"),
        "hardware_owner_attestation": _owner_attestation_template("hardware_owner"),
        "start_time_utc": "<timestamp>",
        "end_time_utc": "<timestamp>",
        "task_id": context["task_id"],
        "scenario_id": context["scenario_id"],
        "scenario_variation_id": context["scenario_variation_instance_id"],
        "scenario_eval_run_id": context["scenario_eval_run_id"],
        "action_log_refs": {
            "robot_state_log_uri": "<physical-g1-state-log-uri>",
            "command_log_uri": "<physical-g1-command-log-uri>",
            "action_trace_uri": "<physical-g1-action-trace-uri>",
        },
        "outcome_ledger_ref": "<deployment-outcome-ledger-uri>",
        "required_before_claim_upgrade": [
            "operator and hardware-owner attestation signed",
            "real G1 action/state logs uploaded",
            "real outcome ledger completed for the same task and scenario variation",
        ],
        "proof_boundary": _proof_boundary(),
    }


def _deployment_outcome_manifest(context: Mapping[str, Any]) -> dict[str, Any]:
    job_id = _string(context.get("job_id"))
    return {
        "schema_version": "deployment_outcome_manifest.v1",
        "status": "template_external_operator_input_required",
        "job_id": job_id,
        "records": [
            {
                "outcome_id": f"unitree-g1-outcome-{_safe_id(job_id, 'job')}",
                "job_id": job_id,
                "policy_id": DEFAULT_POLICY_ID,
                "task_id": context["task_id"],
                "scenario_id": context["scenario_id"],
                "scenario_variation_instance_id": context["scenario_variation_instance_id"],
                "scenario_eval_run_id": context["scenario_eval_run_id"],
                "anchor_schema_version": ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION,
                "anchor_join_key": {
                    "scenario_eval_run_id": context["scenario_eval_run_id"],
                    "policy_id": DEFAULT_POLICY_ID,
                    "task_id": context["task_id"],
                    "scenario_variation_instance_id": context[
                        "scenario_variation_instance_id"
                    ],
                },
                "anchor_status": "template_external_operator_input_required",
                "review_status": "not_reviewed",
                "actual_status": "<passed|failed|aborted>",
                "actual_success": "<true|false>",
                "failure_mode_ids": [],
                "cycle_time_seconds": "<measured-cycle-time>",
                "intervention_count": "<operator-intervention-count>",
                "reviewer_decision": {
                    "safety_review_decision": "not_reviewed",
                    "policy_review_decision": "not_reviewed",
                    "accepted_for_calibration": False,
                },
                "evidence_refs": {
                    "physical_robot_run_manifest": "<physical-robot-run-manifest-uri>",
                    "operator_log": "<operator-log-uri>",
                    "video_review": "<review-video-uri>",
                },
                "owner_evidence_refs": {
                    "physical_robot_run_manifest": "<physical-robot-run-manifest-uri>",
                    "operator_log": "<operator-log-uri>",
                    "video_review": "<review-video-uri>",
                },
                "operator_attestation": _owner_attestation_template("operator"),
            }
        ],
        "proof_boundary": _proof_boundary(),
    }


def _controlled_field_anchor_request_packet(context: Mapping[str, Any]) -> dict[str, Any]:
    join_key = {
        "scenario_eval_run_id": context["scenario_eval_run_id"],
        "policy_id": DEFAULT_POLICY_ID,
        "task_id": context["task_id"],
        "scenario_variation_instance_id": context["scenario_variation_instance_id"],
    }
    return {
        "schema_version": CONTROLLED_FIELD_ANCHOR_REQUEST_SCHEMA_VERSION,
        "status": "not_requested_for_sim_only",
        "job_id": context["job_id"],
        "job_context": {
            "job_id": context["job_id"],
            "task_id": context["task_id"],
            "scenario_id": context["scenario_id"],
            "scenario_eval_run_id": context["scenario_eval_run_id"],
            "scenario_variation_instance_id": context["scenario_variation_instance_id"],
            "robot_profile_id": context["robot_profile_id"],
        },
        "accepted_anchor_schema_version": ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION,
        "required_exact_join_keys": list(CONTROLLED_ANCHOR_JOIN_KEYS),
        "anchor_join_key": join_key,
        "loose_or_inferred_matches_allowed_for_calibration": False,
        "operator_site_checklist": {
            "controlled_area_verified": False,
            "floor_clear_for_g1_walk": False,
            "bystander_exclusion_zone_marked": False,
            "emergency_stop_operator_present": True,
            "site_or_lab_location_id": "<controlled-test-site-or-lab-id>",
        },
        "allowed_task_set": [
            {
                "task_id": context["task_id"],
                "scenario_id": context["scenario_id"],
                "scenario_eval_run_id": context["scenario_eval_run_id"],
                "scenario_variation_instance_id": context[
                    "scenario_variation_instance_id"
                ],
                "policy_id": DEFAULT_POLICY_ID,
                "allowed": True,
            }
        ],
        "exclusion_and_abort_criteria": {
            "excluded_task_ids": ["<any-task-not-listed-in-allowed_task_set>"],
            "abort_conditions": [
                "loss_of_comms",
                "unexpected_human_entry",
                "fall_detected",
                "contact_force_exceeds_threshold",
                "operator_estop",
            ],
            "loose_or_inferred_anchor_matches_allowed": False,
        },
        "robot_calibration_refs": {
            "robot_state_log": "robot_state_log.jsonl",
            "hardware_validation": "hardware_validation.json",
            "contact_collision_log": "contact_collision_log.json",
        },
        "camera_calibration_refs": {
            "robot_camera_video": "robot_camera_video.mp4",
            "timestamp_alignment": "timestamp_alignment.json",
            "camera_mount_or_sensor_ids": ["<g1-head-or-body-camera-id>"],
        },
        "actual_outcome": {
            "actual_status": "<passed|failed|aborted>",
            "actual_success": "<true|false>",
            "cycle_time_seconds": "<measured-cycle-time>",
            "intervention_count": "<operator-intervention-count>",
        },
        "reviewer_decision": {
            "safety_review_decision": "not_reviewed",
            "policy_review_decision": "not_reviewed",
            "accepted_for_calibration": False,
        },
        "owner_evidence": {
            "required_before_physical_claim": list(OWNER_EVIDENCE_CLAIM_REQUIREMENTS),
            "operator_attestation_required": True,
            "hardware_owner_attestation_required": True,
            "safety_reviewer_attestation_required": True,
            "robot_team_review_attestation_required": True,
            "explicit_signed_attestations_required": True,
            "physical_robot_camera_action_and_timestamp_refs_required": True,
        },
        "timestamps": {
            "start_time_utc": "<timestamp>",
            "end_time_utc": "<timestamp>",
        },
        "artifact_provenance": {
            "source": "g1_controlled_proof_setup_template",
            "capture_root": context.get("capture_root"),
        },
        "blockers": [],
        "proof_boundary": {
            **_proof_boundary(),
            "accepted_anchors_can_calibrate_evaluator_ranking_against_supplied_outcomes": False,
            "broad_deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "future_real_world_success_proven": False,
            "sim_only_beta_ranking_blocked": False,
            "physical_evidence_not_requested_for_sim_only": True,
        },
    }


def _real_robot_pov_manifest(context: Mapping[str, Any]) -> dict[str, Any]:
    job_id = _string(context.get("job_id"))
    return {
        "schema_version": "real_robot_pov_manifest.v1",
        "status": "template_external_operator_input_required",
        "job_id": job_id,
        "timestamp_alignment": "<mapping from physical camera timestamps to G1 action log timestamps>",
        "records": [
            {
                "evidence_id": f"unitree-g1-pov-{_safe_id(job_id, 'job')}",
                "job_id": job_id,
                "task_id": context["task_id"],
                "scenario_id": context["scenario_id"],
                "scenario_variation_instance_id": context["scenario_variation_instance_id"],
                "scenario_eval_run_id": context["scenario_eval_run_id"],
                "robot_camera_video_uri": "<physical-g1-pov-camera-video-uri>",
                "camera_mount_or_sensor_ids": ["<g1-head-or-body-camera-id>"],
                "action_log_uri": "<physical-g1-action-log-uri>",
                "timestamp_alignment": "<camera-to-action-log-alignment-uri-or-description>",
                "owner_evidence_refs": {
                    "physical_robot_run_manifest": "<physical-robot-run-manifest-uri>",
                    "deployment_outcome_manifest": "<deployment-outcome-manifest-uri>",
                },
                "operator_attestation": _owner_attestation_template("operator"),
            }
        ],
        "claim_boundary": "Simulator POV frames cannot satisfy this manifest.",
        "proof_boundary": _proof_boundary(),
    }


def _non_ranking_operational_claim_package(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "reviewed_non_ranking_operational_claim_package.v1",
        "status": "template_external_safety_review_required",
        "job_id": context["job_id"],
        "robot_make_model": DEFAULT_ROBOT_MAKE_MODEL,
        "robot_profile_id": DEFAULT_ROBOT_PROFILE_ID,
        "robot_id": "<unitree-g1-serial-or-fleet-id>",
        "task_id": context["task_id"],
        "scenario_id": context["scenario_id"],
        "scenario_variation_id": context["scenario_variation_instance_id"],
        "reviewer_id": "<qualified-safety-reviewer-id>",
        "accepted_safety_thresholds": {
            "max_speed_mps": "<reviewed-threshold>",
            "min_human_clearance_m": "<reviewed-threshold>",
            "max_contact_force_n": "<reviewed-threshold>",
            "emergency_stop_required": True,
        },
        "stop_conditions": [
            "loss_of_comms",
            "unexpected_human_entry",
            "fall_detected",
            "contact_force_exceeds_threshold",
            "operator_estop",
        ],
        "contact_or_collision_log_refs": ["<contact-collision-log-uri>"],
        "physics_or_hardware_validation_refs": ["<hardware-safety-validation-uri>"],
        "review_decision": "not_reviewed",
        "review_timestamp_utc": "<timestamp>",
        "operator_attestation": _owner_attestation_template("safety_reviewer"),
        "proof_boundary": _proof_boundary(),
    }


def _policy_package(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "robot_team_policy_package.v1",
        "status": "template_non_default_policy_execution_required",
        "job_id": context["job_id"],
        "policy_id": DEFAULT_POLICY_ID,
        "policy_owner": "Unitree open-source RL example candidate, owner must review before use",
        "robot_make_model": DEFAULT_ROBOT_MAKE_MODEL,
        "robot_profile_id": DEFAULT_ROBOT_PROFILE_ID,
        "source_repositories": list(OFFICIAL_UNITREE_G1_POLICY_SOURCES),
        "policy_package": {
            "sim_controller_plugin": {
                "simulator_framework": "mujoco",
                "plugin_uri": "https://github.com/unitreerobotics/unitree_rl_gym/tree/main/deploy/deploy_mujoco",
                "source_repository": "https://github.com/unitreerobotics/unitree_rl_gym",
                "source_ref": UNITREE_RL_GYM_MAIN_REF,
                "expected_config": "g1.yaml",
                "physical_deploy_command_template": (
                    "python deploy/deploy_real/deploy_real.py ${BLUEPRINT_G1_NET_INTERFACE:-enp3s0} g1.yaml"
                ),
                "sim_bridge_repository": "https://github.com/unitreerobotics/unitree_mujoco",
                "sim_bridge_ref": UNITREE_MUJOCO_MAIN_REF,
                "execution_trace_refs": ["<non-default-g1-policy-execution-trace-uri>"],
                "metric_refs": ["<policy-metrics-uri>"],
                "owner_acceptance_or_review": "<robot-team-review-uri>",
            },
            "recorded_action_trace": {
                "trace_manifest_uri": "<physical-or-reviewed-sim-action-trace-manifest-uri>",
                "timestamp_alignment": "<policy-trace-to-scenario-eval-run-alignment>",
            },
        },
        "scenario_variation_ids": [context["scenario_variation_instance_id"]],
        "claim_boundary": (
            "This package names a non-default G1 policy candidate. It is not performance proof "
            "until the policy is executed and metrics are tied to this job/scenario."
        ),
        "proof_boundary": _proof_boundary(),
    }


def _official_g1_policy_candidate(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": OFFICIAL_UNITREE_G1_POLICY_CANDIDATE_SCHEMA_VERSION,
        "status": "candidate_selected_execution_required",
        "job_id": context["job_id"],
        "policy_id": DEFAULT_POLICY_ID,
        "candidate_owner": "Unitree official open-source examples; Blueprint must execute and review before use.",
        "robot_make_model": DEFAULT_ROBOT_MAKE_MODEL,
        "robot_profile_id": DEFAULT_ROBOT_PROFILE_ID,
        "task_id": context["task_id"],
        "scenario_id": context["scenario_id"],
        "scenario_variation_ids": [context["scenario_variation_instance_id"]],
        "source_repositories": list(OFFICIAL_UNITREE_G1_POLICY_SOURCES),
        "selected_modalities": [
            "sim_controller_plugin",
        ],
        "candidate_package": {
            "selected_default": {
                "source_repository": "https://github.com/unitreerobotics/unitree_rl_gym",
                "source_ref": UNITREE_RL_GYM_MAIN_REF,
                "source_paths": [
                    "deploy/deploy_mujoco",
                    "deploy/deploy_real",
                    "deploy/pre_train/g1",
                ],
                "physical_deploy_config": "deploy/deploy_real/configs/g1.yaml",
                "physical_deploy_command_template": (
                    "python deploy/deploy_real/deploy_real.py ${BLUEPRINT_G1_NET_INTERFACE:-enp3s0} g1.yaml"
                ),
                "selection_reason": (
                    "Official Unitree G1-supported RL Gym workflow documents "
                    "Train -> Play -> Sim2Sim -> Sim2Real and physical deploy with g1.yaml."
                ),
            },
            "sim_controller_plugin": {
                "simulator_framework": "mujoco",
                "plugin_uri": "https://github.com/unitreerobotics/unitree_rl_gym/tree/main/deploy/deploy_mujoco",
                "alternate_plugin_uri": "https://github.com/unitreerobotics/unitree_rl_lab/tree/main/deploy/robots/g1_29dof",
                "expected_robot_config": "g1.yaml or g1_29dof controller config",
                "requires_unitree_mujoco_bridge": True,
                "unitree_mujoco_bridge_ref": UNITREE_MUJOCO_MAIN_REF,
            },
        },
        "execution_required_before_performance_claim": [
            "fetch or vendor the selected Unitree policy/controller source at a pinned commit",
            "run the policy/controller against this job/scenario in MuJoCo or physical G1",
            "write action traces and metrics tied to the scenario variation",
            "record robot-team owner acceptance or review",
        ],
        "proof_boundary": {
            "candidate_selection_is_not_policy_performance": True,
            "robot_team_policy_performance_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
            "non_ranking_operational_claim_validated": False,
            "real_robot_pov_evidence_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _live_closure_evidence(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "live_robot_eval_closure_evidence.v1",
        "status": "template_external_closure_evidence_required",
        "job_id": context["job_id"],
        "review_acceptance": {
            "status": "not_accepted",
            "accepted": False,
            "reviewer": "<reviewer-id>",
            "operator_attestation": _owner_attestation_template("reviewer"),
        },
        "delivery": {
            "signed_urls": ["<signed-customer-delivery-url>"],
            "storage_upload_performed": False,
            "entitlement_verified": False,
        },
        "safety_contact_physics": {
            "physics_contact_validated": False,
            "non_ranking_operational_claim_validated": False,
            "rank_fidelity_result_proven": False,
            "non_ranking_operational_claim_uri_or_path": "<reviewed-safety-validation-package-uri>",
            "contact_validation_uri_or_path": "<contact-validation-report-uri>",
            "operator_attestation": _owner_attestation_template("safety_reviewer"),
        },
        "rights_privacy": {
            "status": "not_reviewed",
            "external_use_allowed": False,
            "evidence_uri": "<rights-privacy-review-uri>",
        },
        "webapp_upstream": {
            "site_submission_id": context["site_submission_id"],
            "request_id": "<production-webapp-request-id>",
            "buyer_request_id": context["buyer_request_id"],
            "capture_job_id": context["capture_job_id"],
            "pipeline_intake_request_id": "<pipeline-intake-request-id>",
            "sync_status": "<succeeded>",
            "production_forward_url": "<production-forward-url>",
            "request_timestamp_utc": "<timestamp>",
            "response_status_code": "<202>",
        },
        "proof_boundary": _proof_boundary(),
    }


def _runpod_low_cost_plan(context: Mapping[str, Any], provider_launch_request_path: Path) -> dict[str, Any]:
    maximum_seconds = DEFAULT_PROVIDER_WATCHDOG_TTL_SECONDS
    pod_hourly_rate_reference_usd = 0.25
    serverless_cost_per_second_reference_usd = 0.00016
    expected_pod_compute_cost = round(pod_hourly_rate_reference_usd * maximum_seconds / 3600, 4)
    expected_serverless_compute_cost = round(
        serverless_cost_per_second_reference_usd * maximum_seconds, 4
    )
    return {
        "schema_version": "g1_runpod_low_cost_launch_plan.v1",
        "status": "blocked_until_provider_inputs_ready_and_owner_sets_env_gates",
        "job_id": context["job_id"],
        "provider_launch_request_path": str(provider_launch_request_path),
        "preferred_gpu_type_id": DEFAULT_LOW_COST_GPU_TYPE_ID,
        "preferred_serverless_gpu_tier": {
            "gpu_type_ids": [
                "NVIDIA RTX A4000",
                "NVIDIA RTX A4500",
                "NVIDIA RTX 4000 Ada Generation",
            ],
            "memory_gb": 16,
            "reference_cost_per_second_usd": serverless_cost_per_second_reference_usd,
            "source": "RunPod serverless pricing docs: A4000, A4500, RTX 4000 16 GB tier",
        },
        "fallback_gpu_type_ids": [
            "NVIDIA RTX A4500",
            "NVIDIA RTX 4000 Ada Generation",
            "NVIDIA RTX A5000",
            "NVIDIA L4",
        ],
        "max_budget_usd": DEFAULT_PROVIDER_MAX_BUDGET_USD,
        "hard_timeout_seconds": DEFAULT_PROVIDER_HARD_TIMEOUT_SECONDS,
        "external_watchdog_ttl_seconds": DEFAULT_PROVIDER_WATCHDOG_TTL_SECONDS,
        "max_active_workers": 1,
        "expected_max_pod_compute_cost_usd_reference": expected_pod_compute_cost,
        "expected_max_serverless_compute_cost_usd_reference": (
            expected_serverless_compute_cost
        ),
        "expected_cost_basis": (
            "RunPod public docs currently list NVIDIA RTX A4000 as a Pod GPU type and "
            "the A4000/A4500/RTX 4000 serverless tier at 0.00016 USD/sec. This plan "
            "caps the first proof run at three minutes and still requires live pricing "
            "verification before launch."
        ),
        "required_env": [
            "RUNPOD_API_KEY",
            "BLUEPRINT_ALLOW_RUNPOD_API_CALLS=true",
            "BLUEPRINT_ALLOW_GPU_PROVIDER_LAUNCH=true",
        ],
        "known_current_blockers": [
            "provider_launch_request_must_be_status_request_manifest_ready",
            "worker_image_ref_must_be_published_and_provider_fetchable",
            "worker_manifest_and_capture_bundle_must_be_provider_fetchable",
            "artifact_output_uri_must_be_provider_writeable",
            "pod_shutdown_or_termination_proof_required_after_launch",
        ],
        "proof_boundary": _proof_boundary(),
    }


def _placeholder_guard_script(paths: Mapping[str, str]) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

PACKET_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
CAPTURE_ROOT="${{BLUEPRINT_CAPTURE_ROOT:-{paths['capture_root']}}}"
MANIFEST_PATH="${{BLUEPRINT_LIVE_PIPELINE_MANIFEST:-$CAPTURE_ROOT/pipeline/live_pipeline_control_plane/live_pipeline_control_plane_manifest.json}}"
WEBAPP_JOB_REQUEST="${{BLUEPRINT_WEBAPP_JOB_REQUEST:-{paths['job_request']}}}"

if [[ "${{BLUEPRINT_ALLOW_STAGING_G1_CONTROLLED_RUN_INPUTS:-}}" != "true" ]]; then
  echo "Set BLUEPRINT_ALLOW_STAGING_G1_CONTROLLED_RUN_INPUTS=true after replacing every placeholder." >&2
  exit 2
fi

if grep -R --line-number '<[^>][^>]*>' "$PACKET_DIR"/*.json >/tmp/blueprint-g1-placeholder-scan.txt 2>/dev/null; then
  cat /tmp/blueprint-g1-placeholder-scan.txt >&2
  echo "Refusing to stage G1 evidence templates while placeholders remain." >&2
  exit 2
fi

blueprint-intake-live-pipeline-inputs \\
  --manifest-path "$MANIFEST_PATH" \\
  --webapp-job-request "$WEBAPP_JOB_REQUEST" \\
  --policy-package "$PACKET_DIR/robot_team_policy_package.unitree_rl_gym.template.json" \\
  --real-robot-pov "$PACKET_DIR/real_robot_pov_manifest.template.json" \\
  --deployment-outcomes "$PACKET_DIR/deployment_outcome_manifest.template.json" \\
  --live-closure-evidence "$PACKET_DIR/live_eval_closure_evidence.template.json" \\
  --stage-policy-package \\
  --stage-real-robot-pov \\
  --stage-deployment-outcomes \\
  --stage-live-closure-evidence \\
  --overwrite \\
  --output-path "$PACKET_DIR/live_pipeline_input_intake_audit.json" \\
  --staged-inputs-path "$CAPTURE_ROOT/pipeline/live_pipeline_staged_inputs.json"
"""


def _assemble_evidence_script(paths: Mapping[str, str], context: Mapping[str, Any]) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

PACKET_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
CAPTURE_ROOT="${{BLUEPRINT_CAPTURE_ROOT:-{paths['capture_root']}}}"
EVIDENCE_DIR="${{BLUEPRINT_G1_EVIDENCE_DIR:-$PACKET_DIR/physical_g1_evidence_drop}}"
INPUT_CONFIG="${{BLUEPRINT_G1_INPUT_CONFIG:-$EVIDENCE_DIR/g1_controlled_run_inputs.json}}"

mkdir -p "$EVIDENCE_DIR"
if [[ ! -f "$INPUT_CONFIG" ]]; then
  python -m blueprint_pipeline.g1_controlled_run_evidence write-template \\
    --capture-root "$CAPTURE_ROOT" \\
    --job-id "{context['job_id']}" \\
    --output-path "$INPUT_CONFIG"
  echo "Wrote input template to $INPUT_CONFIG. Add physical G1 evidence files, then rerun this script." >&2
  exit 2
fi

python -m blueprint_pipeline.g1_controlled_run_evidence assemble \\
  --capture-root "$CAPTURE_ROOT" \\
  --job-id "{context['job_id']}" \\
  --evidence-dir "$EVIDENCE_DIR" \\
  --input-config "$INPUT_CONFIG" \\
  --output-dir "$PACKET_DIR/assembled_live_inputs"
"""


def _runpod_launch_script(paths: Mapping[str, str]) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

PACKET_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROVIDER_LAUNCH_REQUEST="${{BLUEPRINT_GPU_PROVIDER_LAUNCH_REQUEST:-{paths['provider_launch_request']}}}"
GPU_TYPE_ID="${{BLUEPRINT_RUNPOD_GPU_TYPE_ID:-{DEFAULT_LOW_COST_GPU_TYPE_ID}}}"

if [[ -z "${{RUNPOD_API_KEY:-}}" && -z "${{RUNPOD_API_KEY_FILE:-}}" ]]; then
  echo "Missing RUNPOD_API_KEY or RUNPOD_API_KEY_FILE. Keep the secret in the shell or a local ignored file only." >&2
  exit 2
fi
if [[ "${{BLUEPRINT_ALLOW_RUNPOD_API_CALLS:-}}" != "true" ]]; then
  echo "Set BLUEPRINT_ALLOW_RUNPOD_API_CALLS=true to allow the adapter to call RunPod." >&2
  exit 2
fi
if [[ "${{BLUEPRINT_ALLOW_GPU_PROVIDER_LAUNCH:-}}" != "true" ]]; then
  echo "Set BLUEPRINT_ALLOW_GPU_PROVIDER_LAUNCH=true after confirming spend and shutdown monitoring." >&2
  exit 2
fi

python - "$PROVIDER_LAUNCH_REQUEST" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "request_manifest_ready":
    raise SystemExit(f"Provider launch request is not ready: {{payload.get('status')}}")
PY

blueprint-run-runpod-provider-adapter \\
  --provider-launch-request "$PROVIDER_LAUNCH_REQUEST" \\
  --output-path "$PACKET_DIR/runpod_provider_adapter_result.live.json" \\
  --mode on-demand-pod \\
  --gpu-type-id "$GPU_TYPE_ID" \\
  --pod-name "blueprint-g1-${{BLUEPRINT_ROBOT_EVAL_JOB_SHORT_ID:-controlled-proof}}" \\
  --timeout-seconds 30 \\
  --allow-runpod-api-call

python -m blueprint_pipeline.runpod_live_execution_proof \\
  --provider-launch-request "$PROVIDER_LAUNCH_REQUEST" \\
  --adapter-result "$PACKET_DIR/runpod_provider_adapter_result.live.json" \\
  --output-path "$PACKET_DIR/runpod_live_execution_proof.json" \\
  --stop-pod \\
  --timeout-seconds 30 \\
  --allow-runpod-api-call
"""


def _webapp_forwarding_script(paths: Mapping[str, str], context: Mapping[str, Any]) -> str:
    output = str(Path(paths["capture_root"]) / "pipeline" / "webapp_route_forwarding_proof" / "webapp_route_forwarding_proof.json")
    return f"""#!/usr/bin/env bash
set -euo pipefail

WEBAPP_ROOT="${{BLUEPRINT_WEBAPP_ROOT:-../Blueprint-WebApp}}"
CAPTURE_ROOT="${{BLUEPRINT_CAPTURE_ROOT:-{paths['capture_root']}}}"
OUTPUT_PATH="${{BLUEPRINT_WEBAPP_ROUTE_FORWARDING_PROOF_OUTPUT:-{output}}}"
WEBAPP_PRODUCTION_URL="${{BLUEPRINT_WEBAPP_PRODUCTION_URL:-}}"

if [[ -z "$WEBAPP_PRODUCTION_URL" && -z "${{ROBOT_EVAL_JOB_REQUEST_FORWARD_URL:-}}" ]]; then
  echo "Missing ROBOT_EVAL_JOB_REQUEST_FORWARD_URL for the target Pipeline intake." >&2
  exit 2
fi
if [[ -z "$WEBAPP_PRODUCTION_URL" && -z "${{ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN:-}}" ]]; then
  echo "Missing ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN. Keep it in the shell only." >&2
  exit 2
fi

cd "$WEBAPP_ROOT"
if [[ -n "$WEBAPP_PRODUCTION_URL" ]]; then
  npm run pipeline:first-gpu:production-route-proof -- \\
    --capture-root "$CAPTURE_ROOT" \\
    --output "$OUTPUT_PATH" \\
    --webapp-url "$WEBAPP_PRODUCTION_URL" \\
    --site-slug "{context['site_slug']}" \\
    --task-id "{context['task_id']}" \\
    --scenario-id "{context['scenario_id']}" \\
    --robot-profile-id "{DEFAULT_ROBOT_PROFILE_ID}" \\
    --policy-id "{DEFAULT_POLICY_ID}"
else
  npm run pipeline:forwarding:preflight -- --require-forwarding
  npm run pipeline:first-gpu:route-forwarding-proof -- \\
    --capture-root "$CAPTURE_ROOT" \\
    --output "$OUTPUT_PATH" \\
    --forward-url "$ROBOT_EVAL_JOB_REQUEST_FORWARD_URL" \\
    --site-slug "{context['site_slug']}" \\
    --task-id "{context['task_id']}" \\
    --scenario-id "{context['scenario_id']}" \\
    --robot-profile-id "{DEFAULT_ROBOT_PROFILE_ID}" \\
    --policy-id "{DEFAULT_POLICY_ID}" \\
    --source-kind "owner_agent_codex_request"
fi
"""


def _readme(context: Mapping[str, Any], paths: Mapping[str, str]) -> str:
    return f"""# Unitree G1 Controlled Proof Setup

Status: external operator evidence required.

This packet explicitly selects Unitree G1 for the controlled humanoid proof lane. Blueprint's general robot default remains Franka Panda. The selection does not prove generated-world rank fidelity by itself.

## Order

1. Fill the field-run config and run `field_run_capture_kit/run_g1_field_capture.sh` on the controlled Unitree G1 lab machine for job `{context['job_id']}`, task `{context['task_id']}`, scenario `{context['scenario_id']}`.
2. Review/update the physical-run evidence files in `physical_g1_evidence_drop/`, including `g1_controlled_run_inputs.json`, safety review files, policy metrics, and robot-team acceptance.
3. Run `assembled_live_inputs/stage_assembled_g1_live_inputs.sh` with `BLUEPRINT_ALLOW_STAGING_G1_CONTROLLED_RUN_INPUTS=true`, or fill the raw JSON templates and run `stage_g1_live_inputs.sh`.
4. Publish and verify the MuJoCo worker image, then rerun provider input setup until the provider launch request is `request_manifest_ready`.
5. Export `RUNPOD_API_KEY` or `RUNPOD_API_KEY_FILE`, plus `BLUEPRINT_ALLOW_RUNPOD_API_CALLS=true` and `BLUEPRINT_ALLOW_GPU_PROVIDER_LAUNCH=true`, then run `runpod_live_low_cost_launch.sh`.
6. Run `run_webapp_production_forwarding_proof.sh` against the production/staging forward URL and token to prove customer-through-website request creation and Pipeline intake.

## Boundaries

- Simulator POV is not real robot POV.
- A config that names Unitree G1 is not generated-world rank fidelity.
- The default smoke policy is not robot-team policy performance.
- Dry-run RunPod adapter output is not production provider execution.
- Local WebApp route proof is not production customer-through-website proof.

Key paths:

- Setup manifest: `{paths['setup_manifest']}`
- Controlled anchor request packet: `{paths['controlled_field_anchor_request_packet']}`
- Field-run capture kit: `{paths['field_run_capture_kit']}`
- Evidence assembler: `{paths['assemble_script']}`
- Stage script: `{paths['stage_script']}`
- RunPod script: `{paths['runpod_script']}`
- WebApp script: `{paths['webapp_script']}`
"""


def build_g1_controlled_proof_setup(
    *,
    capture_root: str | Path,
    job_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(capture_root).expanduser().resolve()
    context = _job_context(root, job_id)
    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else root / "pipeline" / "g1_controlled_proof_setup"
    )
    ensure_dir(output_root)
    provider_launch_request_path = (
        root
        / "pipeline"
        / "robot_eval_jobs"
        / _string(context["job_id"])
        / "gpu_provider_launch_request.json"
    )
    job_request_path = _string(context.get("job_request_path")) or str(
        root / "pipeline" / "robot_eval_jobs" / _string(context["job_id"]) / "job_request.json"
    )
    artifacts = {
        "setup_manifest": output_root / "g1_controlled_proof_setup_manifest.json",
        "controlled_field_anchor_request_packet": output_root
        / "controlled_field_anchor_request_packet.template.json",
        "physical_robot_run_manifest": output_root / "physical_robot_run_manifest.template.json",
        "deployment_outcome_manifest": output_root / "deployment_outcome_manifest.template.json",
        "real_robot_pov_manifest": output_root / "real_robot_pov_manifest.template.json",
        "reviewed_non_ranking_operational_claim_package": output_root
        / "reviewed_non_ranking_operational_claim_package.template.json",
        "official_g1_policy_candidate": output_root
        / "official_unitree_g1_policy_candidate.json",
        "robot_team_policy_package": output_root
        / "robot_team_policy_package.unitree_rl_gym.template.json",
        "live_closure_evidence": output_root / "live_eval_closure_evidence.template.json",
        "runpod_low_cost_launch_plan": output_root / "runpod_low_cost_launch_plan.json",
        "runpod_adapter_dry_run_result": output_root / "runpod_provider_adapter_result.dry_run.json",
        "runpod_live_execution_proof": output_root / "runpod_live_execution_proof.json",
        "field_run_capture_kit": output_root
        / "field_run_capture_kit"
        / "g1_field_run_capture_kit_manifest.json",
        "assemble_script": output_root / "assemble_g1_evidence.sh",
        "evidence_input_template": output_root
        / "physical_g1_evidence_drop"
        / "g1_controlled_run_inputs.json",
        "stage_script": output_root / "stage_g1_live_inputs.sh",
        "runpod_script": output_root / "runpod_live_low_cost_launch.sh",
        "webapp_script": output_root / "run_webapp_production_forwarding_proof.sh",
        "readme": output_root / "README.md",
    }
    write_json(
        artifacts["controlled_field_anchor_request_packet"],
        _controlled_field_anchor_request_packet(context),
    )
    write_json(artifacts["physical_robot_run_manifest"], _physical_robot_run_manifest(context))
    write_json(artifacts["deployment_outcome_manifest"], _deployment_outcome_manifest(context))
    write_json(artifacts["real_robot_pov_manifest"], _real_robot_pov_manifest(context))
    write_json(artifacts["reviewed_non_ranking_operational_claim_package"], _non_ranking_operational_claim_package(context))
    write_json(artifacts["official_g1_policy_candidate"], _official_g1_policy_candidate(context))
    write_json(artifacts["robot_team_policy_package"], _policy_package(context))
    write_json(artifacts["live_closure_evidence"], _live_closure_evidence(context))
    write_json(
        artifacts["runpod_low_cost_launch_plan"],
        _runpod_low_cost_plan(context, provider_launch_request_path),
    )
    path_strings = {
        "capture_root": str(root),
        "job_request": job_request_path,
        "provider_launch_request": str(provider_launch_request_path),
        "setup_manifest": str(artifacts["setup_manifest"]),
        "controlled_field_anchor_request_packet": str(
            artifacts["controlled_field_anchor_request_packet"]
        ),
        "stage_script": str(artifacts["stage_script"]),
        "runpod_script": str(artifacts["runpod_script"]),
        "webapp_script": str(artifacts["webapp_script"]),
        "assemble_script": str(artifacts["assemble_script"]),
        "field_run_capture_kit": str(artifacts["field_run_capture_kit"]),
    }
    ensure_dir(artifacts["evidence_input_template"].parent)
    write_text(artifacts["stage_script"], _placeholder_guard_script(path_strings))
    write_text(artifacts["assemble_script"], _assemble_evidence_script(path_strings, context))
    write_text(artifacts["runpod_script"], _runpod_launch_script(path_strings))
    write_text(artifacts["webapp_script"], _webapp_forwarding_script(path_strings, context))
    for script_path in (
        artifacts["stage_script"],
        artifacts["assemble_script"],
        artifacts["runpod_script"],
        artifacts["webapp_script"],
    ):
        script_path.chmod(0o755)
    from .g1_field_run_capture import build_g1_field_run_capture_kit

    field_run_kit = build_g1_field_run_capture_kit(
        capture_root=root,
        job_id=_string(context["job_id"]),
        output_dir=output_root / "field_run_capture_kit",
    )
    write_text(artifacts["readme"], _readme(context, path_strings))
    manifest = {
        "schema_version": G1_CONTROLLED_PROOF_SETUP_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "setup_ready_external_operator_inputs_required",
        "capture_root": str(root),
        "output_dir": str(output_root),
        "default_robot": {
            "make_model": DEFAULT_ROBOT_MAKE_MODEL,
            "robot_profile_id": DEFAULT_ROBOT_PROFILE_ID,
            "proof_target": "physical Unitree G1 controlled run",
        },
        "job_context": context,
        "field_run_capture_kit": field_run_kit,
        "required_to_prove": {
            "generated_world_rank_fidelity": str(artifacts["physical_robot_run_manifest"]),
            "non_ranking_operational_claim": str(artifacts["reviewed_non_ranking_operational_claim_package"]),
            "real_robot_pov": str(artifacts["real_robot_pov_manifest"]),
            "robot_team_policy_performance": str(artifacts["robot_team_policy_package"]),
            "production_runpod_worker_execution": str(artifacts["runpod_low_cost_launch_plan"]),
            "customer_through_website_testing_ready": str(artifacts["webapp_script"]),
        },
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "proof_boundary": _proof_boundary(),
    }
    write_json(artifacts["setup_manifest"], manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    manifest = build_g1_controlled_proof_setup(
        capture_root=args.capture_root,
        job_id=args.job_id,
        output_dir=args.output_dir,
    )
    print(json.dumps({"manifest": manifest["artifacts"]["setup_manifest"], "status": manifest["status"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
