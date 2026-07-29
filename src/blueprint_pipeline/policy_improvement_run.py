"""Deprecated internal policy-candidate experiment manifest builder.

This module preserves the historical Task Evaluation Run, data-export, and
policy-autoresearch artifact composition as an internal compatibility contract.
It is intentionally model-agnostic and source-code-optional: robot
teams can start with a black-box policy API/container, a configurable adapter
surface, or full source/training access.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json, write_text
from .local_capture import resolve_local_capture_context
from .rl_post_training_handoff import build_rl_post_training_handoff_packet


POLICY_IMPROVEMENT_RUN_SCHEMA_VERSION = "policy_improvement_run_offer.v2"
DEFAULT_OUTPUT_DIR_NAME = "policy_improvement_run"

ACCESS_LEVELS = {
    "black_box": {
        "label": "Black-box policy API or container",
        "source_code_required": False,
        "allowed_work": [
            "baseline evaluation",
            "failure diagnosis",
            "wrapper or action-interface tuning",
            "scenario and curriculum generation",
            "sealed regression evaluation",
        ],
        "typical_artifact": "evidence report plus wrapper/config recommendations",
    },
    "config_adapter": {
        "label": "Config, adapter, or task-head access",
        "source_code_required": False,
        "allowed_work": [
            "baseline evaluation",
            "failure diagnosis",
            "adapter or task-head post-training",
            "distilled skill candidate generation",
            "sealed regression evaluation",
        ],
        "typical_artifact": "versioned adapter, task head, or distilled skill package",
    },
    "source_training": {
        "label": "Source and training access",
        "source_code_required": True,
        "allowed_work": [
            "baseline evaluation",
            "failure diagnosis",
            "training recipe changes",
            "policy architecture or reward-development changes outside sealed scoring",
            "complete policy candidate generation",
            "sealed regression evaluation",
        ],
        "typical_artifact": "adapter, task head, distilled skill, or complete policy candidate",
    },
}

IMPROVEMENT_TARGETS = ("adapter", "task_head", "distilled_skill", "complete_policy")

CLAIM_BOUNDARY: dict[str, Any] = {
    "artifact_purpose": "policy_improvement_run_offer",
    "extends_task_evaluation_run": True,
    "extends_post_training_data_package": True,
    "model_backend_agnostic": True,
    "customer_supplied_policy_friendly": True,
    "source_code_required_by_default": False,
    "development_scenarios_may_be_used_for_training": True,
    "sealed_audit_scenarios_must_not_be_used_for_training": True,
    "sim_heldout_success_is_not_rank_fidelity_result": True,
    "wam_heldout_success_is_not_rank_fidelity_result": True,
    "generated_wam_rollouts_are_model_derived_support_artifacts": True,
    "customer_specific_srcc_requires_real_world_validation_rollouts": True,
    "customer_specific_srcc_claimed": False,
    "simulator_execution_proven": False,
    "robot_policy_execution_proven": False,
    "real_world_outcome_proven": False,
    "rank_fidelity_result_proven": False,
    "non_ranking_operational_claim_proven": False,
    "public_claim_upgrade_allowed": False,
}

PRIVATE_HARDWARE_INTEGRATION_MODES: dict[str, dict[str, Any]] = {
    "reference_public_robot": {
        "label": "Reference public robot",
        "default_site_ip_protection_level": "blueprint_hosted",
        "customer_private_robot_assets_required": False,
        "blueprint_hosts_robot_asset": False,
        "customer_hosts_private_runtime": False,
        "typical_use": "Unitree G1 or other public/reference embodiment for plumbing and demos.",
    },
    "private_asset_hosted_by_blueprint": {
        "label": "Private robot asset hosted by Blueprint",
        "default_site_ip_protection_level": "blueprint_hosted",
        "customer_private_robot_assets_required": True,
        "blueprint_hosts_robot_asset": True,
        "customer_hosts_private_runtime": False,
        "typical_use": (
            "The robot team supplies an NDA-bound URDF/MJCF/USD, limits, cameras, "
            "and action contract so Blueprint can compose the robot into a private run."
        ),
    },
    "customer_hosted_sealed_eval_capsule": {
        "label": "Customer-hosted sealed eval capsule",
        "default_site_ip_protection_level": "sealed_eval_capsule",
        "customer_private_robot_assets_required": False,
        "blueprint_hosts_robot_asset": False,
        "customer_hosts_private_runtime": True,
        "typical_use": (
            "Closed-stack robot teams keep private models and simulators in their "
            "environment while running a least-privilege Blueprint eval packet."
        ),
    },
    "physical_robot_evidence_bridge": {
        "label": "Physical robot evidence bridge",
        "default_site_ip_protection_level": "redacted_anchor_packet",
        "customer_private_robot_assets_required": False,
        "blueprint_hosts_robot_asset": False,
        "customer_hosts_private_runtime": True,
        "typical_use": (
            "The customer runs a hardware bridge and returns camera/action/outcome "
            "evidence joined to Blueprint scenario_eval_run_id values."
        ),
    },
}

SITE_IP_PROTECTION_LEVELS: dict[str, dict[str, Any]] = {
    "blueprint_hosted": {
        "label": "Blueprint-hosted harness",
        "raw_capture_shared": False,
        "full_scene_mesh_shared": False,
        "full_scoring_harness_shared": False,
        "sealed_audit_scenarios_disclosed": False,
    },
    "sealed_eval_capsule": {
        "label": "Sealed eval capsule",
        "raw_capture_shared": False,
        "full_scene_mesh_shared": False,
        "full_scoring_harness_shared": False,
        "sealed_audit_scenarios_disclosed": False,
    },
    "redacted_anchor_packet": {
        "label": "Redacted anchor packet",
        "raw_capture_shared": False,
        "full_scene_mesh_shared": False,
        "full_scoring_harness_shared": False,
        "sealed_audit_scenarios_disclosed": False,
    },
}

DEFAULT_PRIVATE_HARDWARE_INTEGRATION_MODE = "customer_hosted_sealed_eval_capsule"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _safe_id(value: Any, *, fallback: str = "item") -> str:
    text = _string(value) or fallback
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in text)
    return "_".join(part for part in cleaned.split("_") if part) or fallback


def _read_optional_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _include_if_file(
    included: dict[str, str],
    *,
    key: str,
    base_dir: Path,
    path: Path,
) -> None:
    if path.is_file():
        included[key] = _relative_to(base_dir, path)


def _first_string(*values: Any) -> str:
    for value in values:
        text = _string(value)
        if text:
            return text
    return ""


def _split_counts(matrix: Mapping[str, Any]) -> dict[str, int]:
    counts = {
        "development": 0,
        "validation": 0,
        "heldout": 0,
        "sealed_audit": 0,
        "unknown": 0,
    }
    runs = matrix.get("runs")
    if not isinstance(runs, list):
        return counts
    for run in runs:
        if not isinstance(run, Mapping):
            counts["unknown"] += 1
            continue
        split = _safe_id(
            run.get("split")
            or run.get("scenario_split")
            or run.get("scenarioSplit")
            or run.get("eval_split")
            or run.get("evalSplit"),
            fallback="unknown",
        )
        if split in {"train", "training", "dev", "development", "autoresearch"}:
            counts["development"] += 1
        elif split in {"val", "validation"}:
            counts["validation"] += 1
        elif split in {"heldout", "holdout", "test"}:
            counts["heldout"] += 1
        elif split in {"sealed", "sealed_audit", "audit"}:
            counts["sealed_audit"] += 1
        else:
            counts["unknown"] += 1
    return counts


def _failure_mode_summary(labels: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[Mapping[str, Any]] = []
    for key in ("labels", "failure_labels", "accepted_failure_labels"):
        value = labels.get(key)
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, Mapping))
    if not rows:
        value = labels.get("failures")
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, Mapping))

    counts: dict[str, int] = {}
    for row in rows:
        mode = _first_string(
            row.get("failure_mode_id"),
            row.get("failure_mode"),
            row.get("label"),
            row.get("reason"),
            "unlabeled_failure",
        )
        counts[mode] = counts.get(mode, 0) + 1
    return {
        "label_count": int(labels.get("label_count") or len(rows)),
        "dominant_failure_modes": [
            {"failure_mode": mode, "count": count}
            for mode, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def _scorecard_summary(evaluation_result: Mapping[str, Any]) -> dict[str, Any]:
    scorecard = _mapping(evaluation_result.get("standard_policy_scorecard"))
    cycle_time = _mapping(scorecard.get("cycle_time"))
    return {
        "status": evaluation_result.get("status") or "missing",
        "success_rate": _float(scorecard.get("success_rate")),
        "cycle_time_mean_seconds": _float(cycle_time.get("mean_seconds")),
        "cycle_time_sample_count": cycle_time.get("sample_count"),
        "intervention_rate": _float(scorecard.get("intervention_rate")),
        "sim_vs_real_calibration_score": _float(
            scorecard.get("sim_vs_real_calibration_score")
        ),
        "required_scenario_eval_run_ids": scorecard.get("required_scenario_eval_run_ids")
        or [],
        "completed_scenario_eval_run_ids": scorecard.get("completed_scenario_eval_run_ids")
        or [],
    }


def _policy_autoresearch_summary(
    *,
    report: Mapping[str, Any],
    candidate: Mapping[str, Any],
    heldout: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = _float(report.get("baseline_heldout_success_rate"))
    best = _float(report.get("best_heldout_success_rate"))
    delta = None if baseline is None or best is None else round(best - baseline, 6)
    return {
        "status": report.get("status") or candidate.get("status") or "missing",
        "candidate_status": candidate.get("status") or "missing",
        "baseline_heldout_success_rate": baseline,
        "best_heldout_success_rate": best,
        "heldout_success_rate_delta": delta,
        "target_success_reached": report.get("target_success_reached") is True,
        "safety_contact_gate_passed": heldout.get("safety_contact_gate_passed") is True,
        "frozen_verifier_sha256": (
            report.get("frozen_verifier_sha256")
            or candidate.get("frozen_verifier_sha256")
            or heldout.get("frozen_verifier_sha256")
        ),
        "promoted_artifact_kind": _first_string(
            candidate.get("promoted_artifact_kind"),
            candidate.get("artifact_kind"),
            "policy_candidate_package",
        ),
        "simulator_execution_proven": candidate.get("simulator_execution_proven") is True,
        "rank_fidelity_result_proven": candidate.get("rank_fidelity_result_proven") is True,
        "public_claim_upgrade_allowed": candidate.get("public_claim_upgrade_allowed") is True,
    }


def _success_claim_ledger_from_sources(*sources: Mapping[str, Any]) -> dict[str, Any]:
    for source in sources:
        if source.get("schema_version") == "success_claim_ledger.v1":
            return dict(source)
        ledger = _mapping(source.get("success_claim_ledger"))
        if ledger:
            return ledger
    return {}


def _private_hardware_integration_plan(
    *,
    job_request: Mapping[str, Any],
    mode: str | None,
    site_ip_protection_level: str | None,
    robot_embodiment_pack_ref: str | None,
    customer_hosted_connector_ref: str | None,
) -> dict[str, Any]:
    request_integration = _mapping(
        job_request.get("private_hardware_integration")
        or job_request.get("hardware_integration")
        or {}
    )
    requested_mode = _first_string(
        mode,
        request_integration.get("integration_mode"),
        request_integration.get("mode"),
        DEFAULT_PRIVATE_HARDWARE_INTEGRATION_MODE,
    )
    if requested_mode not in PRIVATE_HARDWARE_INTEGRATION_MODES:
        raise ValueError(f"Unsupported private hardware integration mode: {requested_mode}")

    mode_profile = dict(PRIVATE_HARDWARE_INTEGRATION_MODES[requested_mode])
    protection_level = _first_string(
        site_ip_protection_level,
        request_integration.get("site_ip_protection_level"),
        request_integration.get("ip_protection_level"),
        mode_profile.get("default_site_ip_protection_level"),
    )
    if protection_level not in SITE_IP_PROTECTION_LEVELS:
        raise ValueError(f"Unsupported site IP protection level: {protection_level}")

    supplied_pack = _first_string(
        robot_embodiment_pack_ref,
        request_integration.get("robot_embodiment_pack_ref"),
        request_integration.get("robotEmbodimentPackRef"),
    )
    connector_ref = _first_string(
        customer_hosted_connector_ref,
        request_integration.get("customer_hosted_connector_ref"),
        request_integration.get("customerHostedConnectorRef"),
    )
    protection = dict(SITE_IP_PROTECTION_LEVELS[protection_level])

    execution_blockers: list[str] = []
    if mode_profile.get("customer_private_robot_assets_required") and not supplied_pack:
        execution_blockers.append("missing_robot_embodiment_pack_ref")
    if mode_profile.get("customer_hosts_private_runtime") and not connector_ref:
        execution_blockers.append("missing_customer_hosted_connector_ref")

    return {
        "schema_version": "private_hardware_integration_plan.v1",
        "integration_mode": requested_mode,
        "integration_label": mode_profile["label"],
        "site_ip_protection_level": protection_level,
        "site_ip_protection_label": protection["label"],
        "recommended_default_for_closed_hardware": "customer_hosted_sealed_eval_capsule",
        "robot_embodiment_pack_ref": supplied_pack or None,
        "customer_hosted_connector_ref": connector_ref or None,
        "blueprint_ip_controls": {
            "raw_capture_bundle_shared_with_customer": protection["raw_capture_shared"],
            "full_resolution_scene_mesh_shared_by_default": protection[
                "full_scene_mesh_shared"
            ],
            "full_scoring_harness_shared_by_default": protection[
                "full_scoring_harness_shared"
            ],
            "sealed_audit_scenarios_disclosed_to_customer": protection[
                "sealed_audit_scenarios_disclosed"
            ],
            "exported_packet_is_least_privilege": True,
            "signed_expiring_artifact_urls_required": True,
            "packet_watermarking_or_request_binding_required": True,
            "customer_visible_packet_fields": [
                "task_id",
                "scenario_eval_run_id",
                "redacted_scene_anchors_or_proxy_assets",
                "observation_schema",
                "action_schema",
                "success_criteria",
                "cycle_time_and_intervention_thresholds",
                "evidence_envelope_contract",
            ],
            "withheld_by_default": [
                "raw_capture_bundle",
                "full_site_geometry_or_dense_scene_assets",
                "capturer_or_site_private_metadata",
                "full_scoring_harness_implementation",
                "sealed_audit_scenario_seeds",
                "hidden_failure_labels_or_verifier_weights",
            ],
        },
        "customer_hardware_controls": {
            "customer_private_robot_model_may_remain_customer_side": bool(
                mode_profile.get("customer_hosts_private_runtime")
            ),
            "customer_private_robot_assets_required_by_blueprint": bool(
                mode_profile.get("customer_private_robot_assets_required")
            ),
            "blueprint_hosts_customer_robot_asset": bool(
                mode_profile.get("blueprint_hosts_robot_asset")
            ),
            "customer_hosts_private_runtime_or_hardware_bridge": bool(
                mode_profile.get("customer_hosts_private_runtime")
            ),
            "private_robot_asset_inputs_if_shared": [
                "URDF_MJCF_or_USD",
                "kinematic_and_dynamic_limits",
                "collision_meshes_or_proxy_collision_shapes",
                "camera_frames_intrinsics_extrinsics",
                "sensor_topics_or_observation_schema",
                "action_command_schema_units_frequency_limits",
                "controller_reset_and_safety_envelope",
            ],
        },
        "required_connector_evidence": [
            "camera_video_or_frame_refs_by_scenario_eval_run_id",
            "action_or_skill_logs_with_timestamps",
            "robot_state_or_joint_state_logs_when_available",
            "observation_action_alignment_summary",
            "outcome_labels_and_failure_modes",
            "checksums_for_returned_artifacts",
            "owner_or_operator_attestation",
        ],
        "claim_boundary": {
            "customer_hosted_connector_outputs_are_owner_evidence": True,
            "customer_hosted_connector_does_not_export_blueprint_raw_scene_ip": True,
            "robot_model_or_urdf_presence_alone_is_not_hardware_readiness": True,
            "generated_world_rank_fidelity_requires_accepted_real_robot_evidence": True,
            "blueprint_scene_packet_is_not_unbounded_site_asset_delivery": True,
        },
        "execution_status": "ready_for_contract_review"
        if not execution_blockers
        else "blocked_missing_private_hardware_inputs",
        "execution_blockers": execution_blockers,
        "typical_use": mode_profile["typical_use"],
    }


def _customer_inputs(
    *,
    job_request: Mapping[str, Any],
    policy_package: Mapping[str, Any],
    customer_policy_ref: str | None,
    embodiment: str | None,
    action_interface: str | None,
    target_task: str | None,
    success_threshold: float | None,
    cycle_time_threshold_seconds: float | None,
) -> dict[str, dict[str, Any]]:
    request_policy = _mapping(job_request.get("policy_package") or job_request.get("policy"))
    request_robot = _mapping(job_request.get("robot") or job_request.get("robot_profile"))
    request_task = _mapping(job_request.get("task") or job_request.get("target_task"))
    request_thresholds = _mapping(
        job_request.get("thresholds")
        or job_request.get("success_thresholds")
        or job_request.get("task_thresholds")
    )
    package_policy_id = _first_string(
        policy_package.get("policy_id"),
        policy_package.get("policyId"),
        policy_package.get("name"),
    )
    base_policy = _first_string(
        customer_policy_ref,
        request_policy.get("policy_id"),
        request_policy.get("policy_uri"),
        request_policy.get("policy_api_endpoint"),
        package_policy_id,
    )
    robot_embodiment = _first_string(
        embodiment,
        request_robot.get("embodiment"),
        request_robot.get("robot_model"),
        request_robot.get("robot_profile_id"),
        job_request.get("robot_profile_id"),
    )
    interface = _first_string(
        action_interface,
        request_policy.get("action_interface"),
        request_policy.get("action_space"),
        policy_package.get("action_interface"),
        policy_package.get("action_space"),
    )
    task = _first_string(
        target_task,
        request_task.get("task_id"),
        request_task.get("task_statement"),
        job_request.get("task_id"),
        job_request.get("target_task"),
    )
    success = (
        success_threshold
        if success_threshold is not None
        else _float(
            request_thresholds.get("success_rate")
            or request_thresholds.get("target_success_rate")
            or job_request.get("target_success_rate")
        )
    )
    cycle_time = (
        cycle_time_threshold_seconds
        if cycle_time_threshold_seconds is not None
        else _float(
            request_thresholds.get("cycle_time_seconds")
            or request_thresholds.get("max_cycle_time_seconds")
            or job_request.get("cycle_time_threshold_seconds")
        )
    )
    return {
        "base_policy_or_model": {"status": "present" if base_policy else "missing", "value": base_policy},
        "robot_embodiment": {
            "status": "present" if robot_embodiment else "missing",
            "value": robot_embodiment,
        },
        "action_interface": {"status": "present" if interface else "missing", "value": interface},
        "target_task": {"status": "present" if task else "missing", "value": task},
        "success_threshold": {"status": "present" if success is not None else "missing", "value": success},
        "cycle_time_threshold_seconds": {
            "status": "present" if cycle_time is not None else "missing",
            "value": cycle_time,
        },
    }



def _readiness_ladder(
    *,
    customer_inputs: Mapping[str, Mapping[str, Any]],
    included_artifacts: Mapping[str, str],
    split_counts: Mapping[str, int],
    post_training_package: Mapping[str, Any],
    policy_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks = [
        (
            "customer_policy_intake",
            all(value.get("status") == "present" for value in customer_inputs.values()),
            [
                f"missing_customer_input_{key}"
                for key, value in customer_inputs.items()
                if value.get("status") != "present"
            ],
        ),
        (
            "task_evaluation_run_contract",
            "scenario_eval_matrix" in included_artifacts
            and int(split_counts.get("heldout") or 0)
            + int(split_counts.get("sealed_audit") or 0)
            > 0,
            [
                blocker
                for blocker, failed in (
                    ("missing_scenario_eval_matrix", "scenario_eval_matrix" not in included_artifacts),
                    (
                        "missing_heldout_or_sealed_audit_split",
                        int(split_counts.get("heldout") or 0)
                        + int(split_counts.get("sealed_audit") or 0)
                        == 0,
                    ),
                )
                if failed
            ],
        ),
        (
            "baseline_evaluation",
            "normalized_attempt_trace" in included_artifacts
            and "evaluation_result" in included_artifacts,
            [
                blocker
                for blocker, failed in (
                    ("baseline_normalized_attempt_trace_missing", "normalized_attempt_trace" not in included_artifacts),
                    ("evaluation_result_missing", "evaluation_result" not in included_artifacts),
                )
                if failed
            ],
        ),
        (
            "failure_diagnosis",
            "failure_labels" in included_artifacts,
            [] if "failure_labels" in included_artifacts else ["failure_labels_missing"],
        ),
        (
            "post_training_data_package",
            post_training_package.get("status") == "export_ready_review_required",
            []
            if post_training_package.get("status") == "export_ready_review_required"
            else ["post_training_data_package_export_not_ready"],
        ),
        (
            "policy_autoresearch",
            "policy_autoresearch_report" in included_artifacts,
            []
            if "policy_autoresearch_report" in included_artifacts
            else ["policy_autoresearch_report_missing"],
        ),
        (
            "candidate_promotion",
            policy_summary.get("candidate_status")
            in {"promoted_sim_only_policy_candidate", "promoted_wam_policy_candidate"}
            and policy_summary.get("safety_contact_gate_passed") is True,
            [
                blocker
                for blocker, failed in (
                    (
                        "promoted_policy_candidate_missing",
                        policy_summary.get("candidate_status")
                        not in {
                            "promoted_sim_only_policy_candidate",
                            "promoted_wam_policy_candidate",
                        },
                    ),
                    (
                        "policy_candidate_safety_contact_gate_not_passed",
                        policy_summary.get("safety_contact_gate_passed") is not True,
                    ),
                )
                if failed
            ],
        ),
        (
            "customer_review_package",
            "policy_candidate_package" in included_artifacts
            and "heldout_eval_result" in included_artifacts,
            [
                blocker
                for blocker, failed in (
                    ("policy_candidate_package_missing", "policy_candidate_package" not in included_artifacts),
                    ("heldout_eval_result_missing", "heldout_eval_result" not in included_artifacts),
                )
                if failed
            ],
        ),
    ]
    return [
        {
            "stage": stage,
            "status": "ready" if ready else "blocked",
            "blockers": blockers,
        }
        for stage, ready, blockers in checks
    ]


def _webapp_summary_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    policy_summary = _mapping(manifest.get("policy_autoresearch_summary"))
    baseline = _mapping(manifest.get("baseline_evaluation_summary"))
    customer_inputs = _mapping(manifest.get("customer_inputs"))
    private_hardware = _mapping(manifest.get("private_hardware_integration"))
    return {
        "schema_version": "policy_improvement_run_webapp_summary.v1",
        "product_family": "legacy_policy_improvement_run",
        "deprecated": True,
        "replacement_product": "task_evaluation_run",
        "scene_id": manifest.get("scene_id"),
        "capture_id": manifest.get("capture_id"),
        "status": manifest.get("status"),
        "blockers": manifest.get("blockers") or [],
        "customer_input_status": {
            key: _mapping(value).get("status") for key, value in customer_inputs.items()
        },
        "baseline_success_rate": baseline.get("success_rate"),
        "candidate_success_rate": policy_summary.get("best_heldout_success_rate"),
        "heldout_success_rate_delta": policy_summary.get("heldout_success_rate_delta"),
        "safety_contact_gate_passed": policy_summary.get("safety_contact_gate_passed"),
        "readiness_ladder": manifest.get("readiness_ladder") or [],
        "artifact_uris": {
            "manifest": manifest.get("manifest_path"),
            "brief": manifest.get("brief_path"),
        },
        "private_hardware_integration": {
            "integration_mode": private_hardware.get("integration_mode"),
            "site_ip_protection_level": private_hardware.get("site_ip_protection_level"),
            "execution_status": private_hardware.get("execution_status"),
            "execution_blockers": private_hardware.get("execution_blockers") or [],
            "blueprint_raw_capture_shared": _mapping(
                private_hardware.get("blueprint_ip_controls")
            ).get("raw_capture_bundle_shared_with_customer"),
            "full_scoring_harness_shared": _mapping(
                private_hardware.get("blueprint_ip_controls")
            ).get("full_scoring_harness_shared_by_default"),
        },
        "safe_for_firestore": True,
        "dense_or_secret_payloads_included": False,
        "claim_boundary": manifest.get("claim_boundary") or {},
    }

def _status_and_blockers(
    *,
    customer_inputs: Mapping[str, Mapping[str, Any]],
    included_artifacts: Mapping[str, str],
    split_counts: Mapping[str, int],
    policy_summary: Mapping[str, Any],
    post_training_package: Mapping[str, Any],
) -> tuple[str, list[str]]:
    blockers = [
        f"missing_customer_input_{key}"
        for key, value in customer_inputs.items()
        if value.get("status") != "present"
    ]
    if "scenario_eval_matrix" not in included_artifacts:
        blockers.append("missing_scenario_eval_matrix")
    if int(split_counts.get("heldout") or 0) + int(split_counts.get("sealed_audit") or 0) == 0:
        blockers.append("missing_heldout_or_sealed_audit_split")

    if blockers:
        return "blocked_missing_policy_improvement_inputs", blockers

    if "normalized_attempt_trace" not in included_artifacts:
        return "ready_for_baseline_evaluation", ["baseline_normalized_attempt_trace_missing"]
    if "failure_labels" not in included_artifacts:
        return "ready_for_failure_diagnosis", ["failure_labels_missing"]
    if post_training_package.get("status") != "export_ready_review_required":
        return "ready_for_post_training_data_package", [
            "post_training_data_package_export_not_ready"
        ]
    if "policy_autoresearch_report" not in included_artifacts:
        return "ready_for_policy_autoresearch", ["policy_autoresearch_report_missing"]
    promoted_statuses = {
        "promoted_sim_only_policy_candidate",
        "promoted_wam_policy_candidate",
    }
    if policy_summary.get("candidate_status") not in promoted_statuses:
        return "completed_no_promoted_candidate", ["promoted_policy_candidate_missing"]
    if policy_summary.get("safety_contact_gate_passed") is not True:
        return "blocked_candidate_failed_safety_contact_gate", [
            "policy_candidate_safety_contact_gate_not_passed"
        ]
    return "improvement_candidate_ready_for_customer_review", []


def _markdown_brief(manifest: Mapping[str, Any]) -> str:
    product = _mapping(manifest.get("product"))
    access = _mapping(manifest.get("access_model"))
    summary = _mapping(manifest.get("policy_autoresearch_summary"))
    boundary = _mapping(manifest.get("claim_boundary"))
    private_hardware = _mapping(manifest.get("private_hardware_integration"))
    ip_controls = _mapping(private_hardware.get("blueprint_ip_controls"))
    handoff = _mapping(manifest.get("rl_post_training_handoff"))
    concurrent_ab = _mapping(handoff.get("concurrent_baseline_ab"))
    bottleneck = _mapping(handoff.get("bottleneck_stage_detection"))
    speed_plan = _mapping(handoff.get("speed_curriculum_plan"))
    action_chunk_qa = _mapping(handoff.get("action_chunk_continuity_qa"))
    safety_ledger = _mapping(handoff.get("intervention_safety_ledger"))
    lines = [
        "# Policy Improvement Run",
        "",
        f"Status: `{manifest.get('status')}`",
        "",
        "## Offer",
        "",
        _string(product.get("one_line")),
        "",
        "## Access",
        "",
        f"- Level: `{access.get('access_level')}`",
        f"- Source code required: `{access.get('source_code_required')}`",
        "",
        "## Private Hardware / IP Controls",
        "",
        f"- Integration mode: `{private_hardware.get('integration_mode')}`",
        f"- Site IP protection: `{private_hardware.get('site_ip_protection_level')}`",
        f"- Raw capture shared: `{ip_controls.get('raw_capture_bundle_shared_with_customer')}`",
        f"- Full scoring harness shared: `{ip_controls.get('full_scoring_harness_shared_by_default')}`",
        f"- Execution status: `{private_hardware.get('execution_status')}`",
        "",
        "## Evidence",
        "",
        f"- Baseline heldout success: `{summary.get('baseline_heldout_success_rate')}`",
        f"- Best heldout success: `{summary.get('best_heldout_success_rate')}`",
        f"- Heldout delta: `{summary.get('heldout_success_rate_delta')}`",
        f"- Safety/contact gate passed: `{summary.get('safety_contact_gate_passed')}`",
        f"- Concurrent baseline A/B: `{concurrent_ab.get('status')}`",
        f"- Dominant bottleneck stage: `{bottleneck.get('dominant_stage')}`",
        f"- Speed curriculum: `{speed_plan.get('status')}`",
        f"- Action-chunk continuity QA: `{action_chunk_qa.get('status')}`",
        f"- Intervention/safety ledger events: `{safety_ledger.get('event_count')}`",
        "",
        "## Boundary",
        "",
        f"- Sim heldout success is generated-world rank-fidelity result: `{not boundary.get('sim_heldout_success_is_not_rank_fidelity_result', True)}`",
        f"- generated-world rank fidelity proven: `{boundary.get('rank_fidelity_result_proven')}`",
        f"- Public claim upgrade allowed: `{boundary.get('public_claim_upgrade_allowed')}`",
    ]
    blockers = manifest.get("blockers")
    if isinstance(blockers, list) and blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    return "\n".join(lines) + "\n"


def build_policy_improvement_run_offer(
    *,
    capture_root: str | Path,
    job_dir: str | Path,
    output_dir: str | Path | None = None,
    access_level: str = "black_box",
    customer_policy_ref: str | None = None,
    embodiment: str | None = None,
    action_interface: str | None = None,
    target_task: str | None = None,
    success_threshold: float | None = None,
    cycle_time_threshold_seconds: float | None = None,
    hardware_integration_mode: str | None = None,
    site_ip_protection_level: str | None = None,
    robot_embodiment_pack_ref: str | None = None,
    customer_hosted_connector_ref: str | None = None,
    improvement_targets: Sequence[str] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if access_level not in ACCESS_LEVELS:
        raise ValueError(f"Unsupported access level: {access_level}")
    targets = tuple(improvement_targets or ("adapter", "task_head", "distilled_skill"))
    unknown_targets = sorted(set(targets) - set(IMPROVEMENT_TARGETS))
    if unknown_targets:
        raise ValueError(f"Unsupported improvement target(s): {', '.join(unknown_targets)}")

    context = resolve_local_capture_context(capture_root)
    resolved_job_dir = Path(job_dir).resolve()
    resolved_output_dir = (
        Path(output_dir).resolve()
        if output_dir
        else resolved_job_dir / DEFAULT_OUTPUT_DIR_NAME
    )
    ensure_dir(resolved_output_dir)
    generated = generated_at or utc_now_iso()

    included_artifacts: dict[str, str] = {}
    job_artifacts = {
        "job_request": "job_request.json",
        "scenario_eval_matrix": "scenario_eval_matrix.json",
        "normalized_attempt_trace": "normalized_attempt_trace.json",
        "failure_labels": "failure_labels.json",
        "policy_package_manifest": "policy_package_manifest.json",
        "policy_execution_manifest": "policy_execution_manifest.json",
        "policy_execution_trace": "policy_execution_trace.json",
        "task_eval_run_report": "task_eval_run_report.json",
        "success_claim_ledger": "success_claim_ledger.json",
        "training_request": "training_request.json",
        "training_result": "training_result.json",
        "evaluation_result": "evaluation_result.json",
        "evaluation_substrate_registry": "evaluation_substrate_registry.json",
        "wam_evaluation_request": "wam_evaluation_request.json",
        "wam_rollout_manifest": "wam_rollout_manifest.json",
        "wam_rollout_results": "wam_rollout_results.json",
        "vision_success_labels": "vision_success_labels.json",
        "policy_ranking_scorecard": "policy_ranking_scorecard.json",
        "wam_eval_claim_boundary": "wam_eval_claim_boundary.json",
        "real_world_validation_followup_request": "real_world_validation_followup_request.json",
        "srcc_validation_plan": "srcc_validation_plan.json",
        "candidate_selection_report": "candidate_selection_report.json",
        "candidate_selection_report_markdown": "candidate_selection_report.md",
        "post_training_data_package_export_manifest": (
            "post_training_data_package_export_manifest.json"
        ),
        "policy_autoresearch_report": "policy_autoresearch/policy_autoresearch_report.json",
        "policy_candidate_package": "policy_autoresearch/policy_candidate_package.json",
        "heldout_eval_result": "policy_autoresearch/heldout_eval_result.json",
        "agent_idea_tree": "policy_autoresearch/agent_idea_tree.json",
        "followup_real_world_validation_request": (
            "policy_autoresearch/followup_real_world_validation_request.json"
        ),
        "budget_ledger": "policy_autoresearch/budget_ledger.json",
        "live_eval_closure_manifest": "live_eval_closure_manifest.json",
        "proof_boundary": "proof_boundary.json",
        "intervention_safety_ledger": "intervention_safety_ledger.json",
        "safety_events_ledger": "safety_events_ledger.json",
    }
    for key, relative_path in job_artifacts.items():
        _include_if_file(
            included_artifacts,
            key=key,
            base_dir=resolved_output_dir,
            path=resolved_job_dir / relative_path,
        )

    pipeline_artifacts = {
        "site_card": "robot_eval_dataset/site_card.json",
        "task_cards": "robot_eval_dataset/task_cards.json",
        "scenario_cards": "robot_eval_dataset/scenario_cards.json",
        "eval_cards": "robot_eval_dataset/eval_cards.json",
        "rights_packet": "robot_eval_dataset/rights_packet.json",
        "proof_boundaries": "robot_eval_dataset/proof_boundaries.json",
    }
    for key, relative_path in pipeline_artifacts.items():
        _include_if_file(
            included_artifacts,
            key=key,
            base_dir=resolved_output_dir,
            path=context.pipeline_root / relative_path,
        )

    job_request = _read_optional_mapping(resolved_job_dir / "job_request.json")
    scenario_matrix = _read_optional_mapping(resolved_job_dir / "scenario_eval_matrix.json")
    labels = _read_optional_mapping(resolved_job_dir / "failure_labels.json")
    policy_package = _read_optional_mapping(resolved_job_dir / "policy_package_manifest.json")
    post_training_package = _read_optional_mapping(
        resolved_job_dir / "post_training_data_package_export_manifest.json"
    )
    policy_report = _read_optional_mapping(
        resolved_job_dir / "policy_autoresearch" / "policy_autoresearch_report.json"
    )
    candidate_package = _read_optional_mapping(
        resolved_job_dir / "policy_autoresearch" / "policy_candidate_package.json"
    )
    heldout_result = _read_optional_mapping(
        resolved_job_dir / "policy_autoresearch" / "heldout_eval_result.json"
    )
    task_eval_run_report = _read_optional_mapping(
        resolved_job_dir / "task_eval_run_report.json"
    )
    direct_success_claim_ledger = _read_optional_mapping(
        resolved_job_dir / "success_claim_ledger.json"
    )
    wam_scorecard = _read_optional_mapping(resolved_job_dir / "policy_ranking_scorecard.json")
    wam_claim_boundary = _read_optional_mapping(resolved_job_dir / "wam_eval_claim_boundary.json")
    candidate_selection_report = _read_optional_mapping(
        resolved_job_dir / "candidate_selection_report.json"
    )
    evaluation_result = _read_optional_mapping(resolved_job_dir / "evaluation_result.json")
    normalized_trace = _read_optional_mapping(resolved_job_dir / "normalized_attempt_trace.json")
    policy_execution_trace = _read_optional_mapping(resolved_job_dir / "policy_execution_trace.json")
    safety_events = _read_optional_mapping(resolved_job_dir / "intervention_safety_ledger.json")
    if not safety_events:
        safety_events = _read_optional_mapping(resolved_job_dir / "safety_events_ledger.json")
    success_claim_ledger = _success_claim_ledger_from_sources(
        direct_success_claim_ledger,
        task_eval_run_report,
        candidate_package,
        heldout_result,
    )

    customer_input_summary = _customer_inputs(
        job_request=job_request,
        policy_package=policy_package,
        customer_policy_ref=customer_policy_ref,
        embodiment=embodiment,
        action_interface=action_interface,
        target_task=target_task,
        success_threshold=success_threshold,
        cycle_time_threshold_seconds=cycle_time_threshold_seconds,
    )
    private_hardware_plan = _private_hardware_integration_plan(
        job_request=job_request,
        mode=hardware_integration_mode,
        site_ip_protection_level=site_ip_protection_level,
        robot_embodiment_pack_ref=robot_embodiment_pack_ref,
        customer_hosted_connector_ref=customer_hosted_connector_ref,
    )
    split_counts = _split_counts(scenario_matrix)
    policy_summary = _policy_autoresearch_summary(
        report=policy_report,
        candidate=candidate_package,
        heldout=heldout_result,
    )
    status, blockers = _status_and_blockers(
        customer_inputs=customer_input_summary,
        included_artifacts=included_artifacts,
        split_counts=split_counts,
        policy_summary=policy_summary,
        post_training_package=post_training_package,
    )
    readiness_ladder = _readiness_ladder(
        customer_inputs=customer_input_summary,
        included_artifacts=included_artifacts,
        split_counts=split_counts,
        post_training_package=post_training_package,
        policy_summary=policy_summary,
    )

    access = dict(ACCESS_LEVELS[access_level])
    access["access_level"] = access_level
    boundary = dict(CLAIM_BOUNDARY)
    boundary["simulator_execution_proven"] = bool(
        candidate_package.get("simulator_execution_proven") is True
    )
    boundary["robot_policy_execution_proven"] = bool(
        candidate_package.get("robot_policy_execution_proven") is True
    )
    boundary["rank_fidelity_result_proven"] = bool(
        candidate_package.get("rank_fidelity_result_proven") is True
    )
    boundary["public_claim_upgrade_allowed"] = bool(
        candidate_package.get("public_claim_upgrade_allowed") is True
    )
    boundary["wam_evaluation_substrate"] = wam_claim_boundary.get("evaluation_substrate")
    boundary["wam_scorecard_included"] = bool(wam_scorecard)
    boundary["customer_specific_srcc_claimed"] = False
    boundary["blueprint_raw_capture_exported_to_customer_by_default"] = False
    boundary["blueprint_full_scoring_harness_exported_to_customer_by_default"] = False
    boundary["customer_hosted_connector_outputs_are_owner_evidence"] = True

    rl_handoff = build_rl_post_training_handoff_packet(
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        job_id=resolved_job_dir.name,
        generated_at=generated,
        job_request=job_request,
        scenario_matrix=scenario_matrix,
        trace=normalized_trace,
        labels=labels,
        evaluation_result=evaluation_result,
        policy_package=policy_package,
        policy_report=policy_report,
        candidate_package=candidate_package,
        heldout_result=heldout_result,
        policy_execution_trace=policy_execution_trace,
        safety_events=safety_events,
        source_artifacts=included_artifacts,
    )
    task_eval_handoff_path = resolved_job_dir / "rl_post_training_handoff_packet.json"
    write_json(task_eval_handoff_path, rl_handoff)
    write_json(resolved_output_dir / "rl_post_training_handoff_packet.json", rl_handoff)
    included_artifacts["rl_post_training_handoff_packet"] = "rl_post_training_handoff_packet.json"
    included_artifacts["task_eval_rl_post_training_handoff_packet"] = _relative_to(
        resolved_output_dir,
        task_eval_handoff_path,
    )
    boundary["concurrent_ab_required_for_candidate_improvement_claim"] = True
    boundary["candidate_improvement_claim_allowed"] = (
        _mapping(rl_handoff.get("concurrent_baseline_ab")).get("candidate_claim_allowed")
        is True
    )
    boundary["rl_post_training_handoff_included"] = True

    # An "improvement candidate ready for customer review" status is itself an
    # improvement claim, so it must not survive a missing/non-positive heldout delta
    # or an unsupported concurrent-A/B comparison.
    improvement_claim_blockers: list[str] = []
    heldout_delta = policy_summary.get("heldout_success_rate_delta")
    if heldout_delta is None:
        improvement_claim_blockers.append("heldout_success_rate_delta_missing")
    elif float(heldout_delta) <= 0:
        improvement_claim_blockers.append(
            f"heldout_success_rate_delta_not_positive:{heldout_delta}"
        )
    if not boundary["candidate_improvement_claim_allowed"]:
        improvement_claim_blockers.append("concurrent_ab_candidate_claim_not_allowed")
    if not success_claim_ledger:
        improvement_claim_blockers.append(
            "success_claim_ledger_missing_for_candidate_improvement_claim"
        )
    elif success_claim_ledger.get("schema_version") != "success_claim_ledger.v1":
        improvement_claim_blockers.append(
            "success_claim_ledger_schema_unrecognized_for_candidate_improvement_claim"
        )
    boundary["success_claim_ledger_included"] = bool(success_claim_ledger)
    boundary["success_claim_ledger_highest_truthful_claim"] = success_claim_ledger.get(
        "highest_truthful_claim"
    )
    boundary["improvement_claim_blockers"] = improvement_claim_blockers
    if status == "improvement_candidate_ready_for_customer_review" and improvement_claim_blockers:
        status = "blocked_improvement_claim_unsupported"
        blockers = [*blockers, *improvement_claim_blockers]

    manifest: dict[str, Any] = {
        "schema_version": POLICY_IMPROVEMENT_RUN_SCHEMA_VERSION,
        "deprecation": {
            "deprecated": True,
            "compatibility_only": True,
            "replacement_product": "Task Evaluation Run",
            "default_orchestration_enabled": False,
            "candidate_generation_is_internal_experiment": True,
            "training_or_improvement_implied": False,
        },
        "generated_at": generated,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "job_dir": str(resolved_job_dir),
        "product": {
            "name": "Policy Improvement Run",
            "product_family": "task_eval_to_policy_improvement",
            "one_line": (
                "Evaluate a customer-supplied robot policy against a real-site "
                "Task Evaluation Run using replaceable evaluation substrates, turn "
                "failures into twin/cousin curricula, post-train a bounded candidate, "
                "and report sealed before/after evidence."
            ),
            "primary_sale": "policy lift toward a pilot gate",
            "not_sold_as": [
                "foundation-model ownership",
                "unbounded raw scene or scoring-harness export",
                "generated-world rank-fidelity result",
                "physical safety certification",
                "guaranteed site production readiness",
            ],
            "extends": ["Task Evaluation Run", "Post-Training Data Package"],
        },
        "status": status,
        "blockers": blockers,
        "customer_inputs": customer_input_summary,
        "access_model": access,
        "private_hardware_integration": private_hardware_plan,
        "improvement_targets": list(targets),
        "workflow": [
            "baseline_evaluation",
            "concurrent_frozen_baseline_ab_reservation",
            "substrate_selection_or_wam_fixture_evaluation",
            "dominant_failure_mode_diagnosis",
            "bottleneck_substage_detection",
            "speed_curriculum_planning",
            "action_chunk_continuity_qa",
            "intervention_safety_ledger_review",
            "digital_twin_and_cousin_scenario_generation",
            "curriculum_build",
            "candidate_post_training_or_policy_lift",
            "sealed_scenario_evaluation",
            "improved_artifact_and_evidence_report",
        ],
        "task_evaluation_run_parity": {
            "baseline_eval_result_required": True,
            "scenario_eval_matrix_required": True,
            "normalized_attempt_trace_required": True,
            "standard_policy_scorecard_projected": True,
            "customer_handoff_ready_only_after_review": True,
            "webapp_projection_safe": True,
        },
        "baseline_evaluation_summary": _scorecard_summary(evaluation_result),
        "scenario_split_policy": {
            "development_visible_to_autoresearch": True,
            "validation_limited_feedback": True,
            "heldout_or_sealed_required_for_promotion": True,
            "sealed_audit_inaccessible_to_training": True,
            "split_counts": split_counts,
        },
        "failure_mode_summary": _failure_mode_summary(labels),
        "post_training_package_summary": {
            "status": post_training_package.get("status") or "missing",
            "package_type": post_training_package.get("package_type"),
            "export_policy": post_training_package.get("export_policy") or {},
            "manifest_counts": post_training_package.get("manifest_counts") or {},
        },
        "policy_autoresearch_summary": policy_summary,
        "success_claim_summary": {
            "success_claim_ledger_included": bool(success_claim_ledger),
            "highest_truthful_claim": success_claim_ledger.get("highest_truthful_claim"),
            "ledger_path": included_artifacts.get("success_claim_ledger")
            or included_artifacts.get("task_eval_run_report"),
        },
        "rl_post_training_handoff": rl_handoff,
        "wam_evaluation_summary": {
            "status": wam_scorecard.get("status") or "missing",
            "evaluation_substrate": wam_scorecard.get("evaluation_substrate")
            or wam_claim_boundary.get("evaluation_substrate"),
            "top_policy_id": wam_scorecard.get("top_policy_id"),
            "candidate_selection_status": candidate_selection_report.get("status")
            or "missing",
            "candidate_selection_top_policy_id": candidate_selection_report.get("top_policy_id"),
            "candidate_selection_runner_up_policy_id": candidate_selection_report.get(
                "runner_up_policy_id"
            ),
            "candidate_selection_margin": candidate_selection_report.get("margin"),
            "ranking_ambiguous": _mapping(
                candidate_selection_report.get("selection")
            ).get("ranking_ambiguous"),
            "candidate_shortlist": candidate_selection_report.get("candidate_shortlist") or [],
            "decisive_scenario_count": len(
                candidate_selection_report.get("decisive_scenarios") or []
            ),
            "failure_cluster_count": len(
                candidate_selection_report.get("failure_clusters") or []
            ),
            "candidate_selection_report_path": (
                "candidate_selection_report.json" if candidate_selection_report else None
            ),
            "policy_count": wam_scorecard.get("policy_count"),
            "scenario_attempt_count": wam_scorecard.get("scenario_attempt_count"),
            "customer_specific_srcc_claimed": False,
            "claim_boundary": {
                "generated_wam_rollouts_are_model_derived_support_artifacts": True,
                "passing_wam_heldout_eval_is_not_rank_fidelity_result": True,
                "customer_specific_srcc_requires_real_world_validation_rollouts": True,
            },
        },
        "readiness_ladder": readiness_ladder,
        "deliverables": [
            "baseline_eval_report",
            "rl_post_training_handoff_packet",
            "concurrent_baseline_ab_plan",
            "bottleneck_stage_detection",
            "speed_curriculum_plan",
            "action_chunk_continuity_qa",
            "intervention_safety_ledger",
            "private_hardware_integration_plan",
            "wam_rollout_and_policy_ranking_scorecard_when_requested",
            "failure_mode_report",
            "twin_and_cousin_scenario_curriculum",
            "post_training_data_package",
            "policy_candidate_package",
            "sealed_heldout_eval_result",
            "customer_evidence_report",
            "real_world_validation_followup_request",
        ],
        "commercial_tiers": [
            {
                "tier": "baseline_eval_and_failure_diagnosis",
                "typical_when": "team needs a credible before score and failure taxonomy",
            },
            {
                "tier": "policy_lift_sprint",
                "typical_when": "team has a borderline policy and needs a better candidate",
            },
            {
                "tier": "pilot_gate_package",
                "typical_when": "team needs before/after evidence for a site pilot decision",
            },
        ],
        "included_artifacts": included_artifacts,
        "manifest_path": "policy_improvement_run_offer.json",
        "brief_path": "policy_improvement_run_offer.md",
        "private_hardware_integration_path": "private_hardware_integration_plan.json",
        "claim_boundary": boundary,
    }
    manifest["webapp_summary_projection"] = _webapp_summary_projection(manifest)

    write_json(resolved_output_dir / "policy_improvement_run_offer.json", manifest)
    write_json(
        resolved_output_dir / "private_hardware_integration_plan.json",
        private_hardware_plan,
    )
    write_json(
        resolved_output_dir / "policy_improvement_run_webapp_summary.json",
        manifest["webapp_summary_projection"],
    )
    write_text(resolved_output_dir / "policy_improvement_run_offer.md", _markdown_brief(manifest))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "DEPRECATED compatibility command: build an internal policy-candidate "
            "experiment manifest. Use a Task Evaluation Run for customer decisions."
        )
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--access-level",
        choices=sorted(ACCESS_LEVELS),
        default="black_box",
    )
    parser.add_argument("--customer-policy-ref")
    parser.add_argument("--embodiment")
    parser.add_argument("--action-interface")
    parser.add_argument("--target-task")
    parser.add_argument("--success-threshold", type=float)
    parser.add_argument("--cycle-time-threshold-seconds", type=float)
    parser.add_argument(
        "--hardware-integration-mode",
        choices=sorted(PRIVATE_HARDWARE_INTEGRATION_MODES),
    )
    parser.add_argument(
        "--site-ip-protection-level",
        choices=sorted(SITE_IP_PROTECTION_LEVELS),
    )
    parser.add_argument("--robot-embodiment-pack-ref")
    parser.add_argument("--customer-hosted-connector-ref")
    parser.add_argument(
        "--improvement-target",
        action="append",
        choices=IMPROVEMENT_TARGETS,
        dest="improvement_targets",
    )
    args = parser.parse_args(argv)
    result = build_policy_improvement_run_offer(
        capture_root=args.capture_root,
        job_dir=args.job_dir,
        output_dir=args.output_dir,
        access_level=args.access_level,
        customer_policy_ref=args.customer_policy_ref,
        embodiment=args.embodiment,
        action_interface=args.action_interface,
        target_task=args.target_task,
        success_threshold=args.success_threshold,
        cycle_time_threshold_seconds=args.cycle_time_threshold_seconds,
        hardware_integration_mode=args.hardware_integration_mode,
        site_ip_protection_level=args.site_ip_protection_level,
        robot_embodiment_pack_ref=args.robot_embodiment_pack_ref,
        customer_hosted_connector_ref=args.customer_hosted_connector_ref,
        improvement_targets=args.improvement_targets,
    )
    manifest_dir = Path(args.output_dir or Path(args.job_dir) / DEFAULT_OUTPUT_DIR_NAME)
    print(f"[policy-improvement-run] manifest={manifest_dir / 'policy_improvement_run_offer.json'}")
    print(f"[policy-improvement-run] status={result['status']}")
    return 0 if not str(result["status"]).startswith("blocked") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
