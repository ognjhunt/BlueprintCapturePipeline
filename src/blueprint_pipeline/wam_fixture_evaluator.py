"""Deterministic fixture WAM evaluator for local robot-eval job tests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json, write_text
from .wam_eval_substrate import (
    WAM_EVALUATION_SUBSTRATES,
    build_wam_eval_claim_boundary,
    build_wam_evaluation_request,
    normalize_evaluation_substrate,
    write_evaluation_substrate_registry,
)
from .wam_provider_runtime import (
    classical_sim_cross_check_plan as _classical_sim_cross_check_plan,
    customer_validation_envelope as _customer_validation_envelope,
    live_provider_gate_blockers as _live_provider_gate_blockers,
    normalize_provider_rollouts as _normalize_provider_rollouts,
    policy_interface_binding as _policy_interface_binding,
    production_ops_manifest as _production_ops_manifest,
    provider_artifact_upload_proof as _provider_artifact_upload_proof,
    provider_auth_status as _provider_auth_status,
    provider_cost_ledger as _provider_cost_ledger,
    provider_execution_manifest as _provider_execution_manifest,
    provider_runtime_package as _provider_runtime_package,
    real_world_anchor_manifest as _real_world_anchor_manifest,
    run_provider_command as _run_provider_command,
    substrate_provider_command as _substrate_provider_command,
    vision_review_queue as _vision_review_queue,
)
from .wam_vision_success_judge import build_fixture_vision_success_labels


WAM_ROLLOUT_MANIFEST_SCHEMA_VERSION = "wam_rollout_manifest.v1"
WAM_ROLLOUT_RESULTS_SCHEMA_VERSION = "wam_rollout_results.v1"
POLICY_RANKING_SCORECARD_SCHEMA_VERSION = "policy_ranking_scorecard.v1"
REAL_WORLD_VALIDATION_FOLLOWUP_SCHEMA_VERSION = "real_world_validation_followup_request.v1"
SRCC_VALIDATION_PLAN_SCHEMA_VERSION = "srcc_validation_plan.v1"
NORMALIZED_ATTEMPT_TRACE_SCHEMA_VERSION = "robot_eval_job_normalized_attempt_trace.v1"
FAILURE_LABELS_SCHEMA_VERSION = "robot_eval_job_failure_labels.v1"
PREDICTION_OUTCOME_LEDGER_SCHEMA_VERSION = "robot_eval_job_prediction_outcome_ledger.v1"
CALIBRATION_REPORT_SCHEMA_VERSION = "robot_eval_job_calibration_report.v1"
BREAKAGE_LIBRARY_SCHEMA_VERSION = "robot_eval_job_breakage_library.v1"

WAM_ARTIFACT_PATHS = {
    "evaluation_substrate_registry": "evaluation_substrate_registry.json",
    "wam_evaluation_request": "wam_evaluation_request.json",
    "wam_rollout_manifest": "wam_rollout_manifest.json",
    "wam_rollout_results": "wam_rollout_results.json",
    "vision_success_labels": "vision_success_labels.json",
    "normalized_attempt_trace": "normalized_attempt_trace.json",
    "failure_labels": "failure_labels.json",
    "prediction_outcome_ledger": "prediction_outcome_ledger.json",
    "calibration_report": "calibration_report.json",
    "breakage_library": "breakage_library.json",
    "policy_ranking_scorecard": "policy_ranking_scorecard.json",
    "wam_eval_claim_boundary": "wam_eval_claim_boundary.json",
    "real_world_validation_followup_request": "real_world_validation_followup_request.json",
    "srcc_validation_plan": "srcc_validation_plan.json",
    "wam_provider_runtime_package": "wam_provider_runtime_package.json",
    "wam_provider_execution_manifest": "wam_provider_execution_manifest.json",
    "wam_provider_cost_control_ledger": "wam_provider_cost_control_ledger.json",
    "wam_provider_artifact_upload_proof": "wam_provider_artifact_upload_proof.json",
    "wam_policy_interface_binding": "wam_policy_interface_binding.json",
    "wam_vision_success_review_queue": "wam_vision_success_review_queue.json",
    "wam_real_world_validation_anchor_manifest": "wam_real_world_validation_anchor_manifest.json",
    "wam_customer_validation_envelope": "wam_customer_validation_envelope.json",
    "wam_production_ops_manifest": "wam_production_ops_manifest.json",
    "wam_classical_sim_cross_check_plan": "wam_classical_sim_cross_check_plan.json",
    "customer_handoff_report": "customer_handoff_report.json",
    "customer_handoff_report_markdown": "customer_handoff_report.md",
}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_string(item) for item in value if _string(item)]
    return []


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_id(value: Any, *, fallback: str = "item") -> str:
    text = _string(value) or fallback
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in text)
    return "_".join(part for part in cleaned.split("_") if part) or fallback


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _matrix_runs(matrix: Mapping[str, Any]) -> list[Dict[str, Any]]:
    runs = matrix.get("runs")
    if not isinstance(runs, list):
        return []
    normalized: list[Dict[str, Any]] = []
    for index, raw in enumerate(runs, start=1):
        if not isinstance(raw, Mapping):
            continue
        run = dict(raw)
        run_id = _string(run.get("scenario_eval_run_id") or run.get("scenarioEvalRunId"))
        if not run_id:
            run_id = f"scenario_eval_run_{index:04d}"
        run["scenario_eval_run_id"] = run_id
        normalized.append(run)
    return normalized


def _policy_candidates(
    *,
    request: Mapping[str, Any],
    policy_manifest: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    wam_request = _mapping(request.get("wam_evaluation") or request.get("wamEvaluation"))
    raw = (
        request.get("policy_candidates")
        or request.get("policyCandidates")
        or request.get("policies")
        or request.get("checkpoints")
        or wam_request.get("policy_candidates")
        or wam_request.get("policyCandidates")
        or wam_request.get("policies")
        or wam_request.get("checkpoints")
    )
    candidates: list[Dict[str, Any]] = []
    if isinstance(raw, list):
        for index, item in enumerate(raw, start=1):
            payload = _mapping(item)
            policy_id = (
                _string(payload.get("policy_id") or payload.get("policyId"))
                or f"policy_candidate_{index:02d}"
            )
            capabilities = _string_list(
                payload.get("capabilities")
                or payload.get("policy_capabilities")
                or payload.get("policyCapabilities")
                or payload.get("supported_failure_modes")
                or payload.get("supportedFailureModes")
            )
            candidates.append(
                {
                    **payload,
                    "policy_id": policy_id,
                    "display_name": _string(payload.get("display_name") or payload.get("name"))
                    or policy_id,
                    "capabilities": sorted(set(capabilities)),
                }
            )
    if candidates:
        return candidates
    selected = _string_list(policy_manifest.get("selected_modalities"))
    policy_id = _string(policy_manifest.get("policy_id") or policy_manifest.get("policyId"))
    if not policy_id:
        policy_id = selected[0] if selected else "policy_package_candidate"
    return [
        {
            "policy_id": policy_id,
            "display_name": policy_id,
            "capabilities": _string_list(policy_manifest.get("policy_capabilities")),
            "source": "policy_package_manifest",
        }
    ]


def _risk_profile(run: Mapping[str, Any]) -> Dict[str, Any]:
    text = " ".join(
        _string(run.get(key))
        for key in (
            "scenario_eval_run_id",
            "scenario_id",
            "task_id",
            "variation_name",
        )
    ).lower()
    required: list[str] = []
    failures: list[str] = []
    if any(marker in text for marker in ("blocked", "obstacle", "narrow", "clearance")):
        required.append("clearance_aware_navigation")
        failures.append("blocked_path_or_clearance_failure")
    if any(marker in text for marker in ("human", "forklift", "crossing", "dynamic")):
        required.append("dynamic_obstacle_yield")
        failures.append("dynamic_agent_safety_failure")
    if any(marker in text for marker in ("occlusion", "glare", "missing_label", "wrong_object")):
        required.append("visual_recheck")
        failures.append("perception_ambiguity_failure")
    if any(marker in text for marker in ("grasp", "place", "object_rotation", "cart_shifted")):
        required.append("grasp_alignment_correction")
        failures.append("manipulation_alignment_failure")
    return {
        "required_capabilities": sorted(set(required)),
        "candidate_failure_modes": sorted(set(failures)),
    }


def _policy_supports(policy: Mapping[str, Any], capability: str) -> bool:
    capabilities = set(_string_list(policy.get("capabilities")))
    return "all" in capabilities or capability in capabilities


def _forced_failures(policy: Mapping[str, Any], run: Mapping[str, Any]) -> bool:
    profile = _mapping(policy.get("fixture_success_profile") or policy.get("success_profile"))
    fail_variations = set(_string_list(profile.get("fail_variation_names")))
    fail_runs = set(_string_list(profile.get("fail_scenario_eval_run_ids")))
    variation_name = _string(run.get("variation_name") or run.get("variationName"))
    run_id = _string(run.get("scenario_eval_run_id"))
    return variation_name in fail_variations or run_id in fail_runs


def _rollout_for_run(
    *,
    job_dir: Path,
    substrate: str,
    policy: Mapping[str, Any],
    run: Mapping[str, Any],
    index: int,
    generated_at: str,
) -> Dict[str, Any]:
    policy_id = _string(policy.get("policy_id")) or "policy"
    run_id = _string(run.get("scenario_eval_run_id"))
    risk = _risk_profile(run)
    missing = [
        capability
        for capability in _string_list(risk.get("required_capabilities"))
        if not _policy_supports(policy, capability)
    ]
    forced_failure = _forced_failures(policy, run)
    ood_flags: list[str] = []
    variation_name = _string(run.get("variation_name") or run.get("variationName"))
    if any(marker in variation_name.lower() for marker in ("wrong_object", "glare", "missing_label")):
        ood_flags.append("vision_distribution_shift")
    if forced_failure:
        ood_flags.append("fixture_forced_failure")
    uncertainty = min(
        0.95,
        round(
            0.12
            + 0.11 * len(missing)
            + 0.08 * len(_string_list(risk.get("required_capabilities")))
            + 0.18 * len(ood_flags),
            6,
        ),
    )
    success = not missing and not forced_failure and uncertainty < 0.75
    failure_modes = [] if success else _string_list(risk.get("candidate_failure_modes"))
    if forced_failure and "fixture_policy_failure" not in failure_modes:
        failure_modes.append("fixture_policy_failure")
    rollout_id = f"wam_{_safe_id(policy_id)}_{_safe_id(run_id)}"
    attempt_id = f"{rollout_id}_attempt"
    media_dir = job_dir / "wam_rollouts"
    ensure_dir(media_dir)
    support_manifest_path = media_dir / f"{rollout_id}.json"
    support_manifest = {
        "schema_version": "fixture_wam_rollout_support_manifest.v1",
        "generated_at": generated_at,
        "rollout_id": rollout_id,
        "evaluation_substrate": substrate,
        "policy_id": policy_id,
        "scenario_eval_run_id": run_id,
        "generated_video_available": False,
        "deterministic_fixture_frames": [
            {
                "frame_index": 0,
                "description": "initial captured-site conditioned observation",
            },
            {
                "frame_index": 1,
                "description": "policy action applied through fixture WAM transition",
            },
            {
                "frame_index": 2,
                "description": "fixture outcome frame used by the vision success judge",
            },
        ],
        "claim_boundary": {
            "support_manifest_not_video_truth": True,
            "model_derived_support_artifact": True,
            "raw_capture_evidence": False,
        },
    }
    write_json(support_manifest_path, support_manifest)
    return {
        "rollout_id": rollout_id,
        "attempt_id": attempt_id,
        "generated_at": generated_at,
        "evaluation_substrate": substrate,
        "simulator_engine": substrate,
        "policy_id": policy_id,
        "policy_display_name": _string(policy.get("display_name")) or policy_id,
        "scenario_eval_run_id": run_id,
        "scenario_variation_instance_id": run.get("scenario_variation_instance_id")
        or run.get("scenarioVariationInstanceId"),
        "task_id": _string(run.get("task_id") or run.get("taskId")),
        "scenario_id": _string(run.get("scenario_id") or run.get("scenarioId")),
        "variation_name": variation_name or None,
        "rollout_index": index,
        "predicted_success": success,
        "required_policy_capabilities": _string_list(risk.get("required_capabilities")),
        "policy_capabilities": _string_list(policy.get("capabilities")),
        "missing_policy_capabilities": missing,
        "failure_mode_ids": failure_modes,
        "uncertainty_score": uncertainty,
        "ood_flags": ood_flags,
        "metrics": {
            "cycle_time_seconds": round(18.0 + index * 0.05 + len(missing) * 2.0, 6),
            "intervention_count": 0 if success else 1,
            "contact_event_count": 0 if success else int("clearance_aware_navigation" in missing),
            "safety_event_count": 0 if "dynamic_obstacle_yield" not in missing else 1,
            "world_model_uncertainty": uncertainty,
            "ood_flag_count": len(ood_flags),
        },
        "artifact_paths": {
            "rollout_support_manifest": str(support_manifest_path.relative_to(job_dir)),
        },
        "claim_boundary": {
            "model_derived_support_artifact": True,
            "raw_capture_evidence": False,
            "simulator_execution_proven": False,
            "robot_policy_execution_proven": False,
            "real_world_outcome_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _rollout_manifest(
    *,
    job_id: str,
    substrate: str,
    rollouts: Sequence[Mapping[str, Any]],
    generated_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": WAM_ROLLOUT_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "completed" if rollouts else "blocked_missing_rollouts",
        "evaluation_substrate": substrate,
        "rollout_count": len(rollouts),
        "rollouts": [dict(rollout) for rollout in rollouts],
        "artifact_dir": "wam_rollouts",
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def _rollout_results(
    *,
    job_id: str,
    substrate: str,
    rollouts: Sequence[Mapping[str, Any]],
    generated_at: str,
) -> Dict[str, Any]:
    success_count = sum(1 for rollout in rollouts if bool(rollout.get("predicted_success")))
    return {
        "schema_version": WAM_ROLLOUT_RESULTS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "completed" if rollouts else "blocked_missing_rollouts",
        "evaluation_substrate": substrate,
        "rollout_count": len(rollouts),
        "predicted_success_count": success_count,
        "predicted_failure_count": len(rollouts) - success_count,
        "predicted_success_rate": round(success_count / len(rollouts), 6) if rollouts else 0.0,
        "failure_mode_ids": sorted(
            {mode for rollout in rollouts for mode in _string_list(rollout.get("failure_mode_ids"))}
        ),
        "ood_rollout_count": sum(1 for rollout in rollouts if _string_list(rollout.get("ood_flags"))),
        "rollouts": [dict(rollout) for rollout in rollouts],
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def _normalized_attempt_trace(
    *,
    substrate: str,
    labels: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    label_rows = [
        dict(label)
        for label in labels.get("labels", []) or []
        if isinstance(label, Mapping)
    ]
    attempts: list[Dict[str, Any]] = []
    for label in label_rows:
        success = bool(label.get("task_success"))
        attempts.append(
            {
                "attempt_id": label.get("attempt_id"),
                "rollout_id": label.get("rollout_id"),
                "scenario_eval_run_id": label.get("scenario_eval_run_id"),
                "scenario_variation_instance_id": label.get("scenario_variation_instance_id"),
                "task_id": label.get("task_id"),
                "scenario_id": label.get("scenario_id"),
                "variation_name": label.get("variation_name"),
                "policy_id": label.get("policy_id"),
                "evaluation_substrate": substrate,
                "simulator_engine": substrate,
                "status": "completed" if success else "failed",
                "success": success,
                "task_success": success,
                "failure_mode_ids": _string_list(label.get("failure_mode_ids")),
                "metrics": {
                    "world_model_uncertainty": _number(label.get("uncertainty_score")),
                    "intervention_count": 0 if success else 1,
                    "contact_event_count": 0,
                    "safety_event_count": 0,
                    "cycle_time_seconds": 20.0,
                },
                "artifact_paths": {"vision_success_label": "vision_success_labels.json"},
                "claim_boundary": {
                    "generated_wam_attempt": True,
                    "model_derived_support_artifact": True,
                    "simulator_execution_proven": False,
                    "robot_policy_execution_proven": False,
                    "robot_readiness_proven": False,
                },
            }
        )
    successful = [attempt for attempt in attempts if attempt["success"]]
    failed = [attempt for attempt in attempts if not attempt["success"]]
    run_ids = sorted({_string(attempt.get("scenario_eval_run_id")) for attempt in attempts if attempt.get("scenario_eval_run_id")})
    return {
        "schema_version": NORMALIZED_ATTEMPT_TRACE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if attempts else "blocked_missing_attempts",
        "runner": f"{substrate}_evaluator",
        "evaluation_substrate": substrate,
        "attempt_count": len(attempts),
        "successful_task_attempt_count": len(successful),
        "failed_task_attempt_count": len(failed),
        "task_success_rate": round(len(successful) / len(attempts), 6) if attempts else 0.0,
        "task_success_summary": {
            "attempt_count": len(attempts),
            "successful_attempt_count": len(successful),
            "failed_attempt_count": len(failed),
            "task_success_rate": round(len(successful) / len(attempts), 6) if attempts else 0.0,
        },
        "covered_scenario_eval_run_ids": run_ids,
        "missing_scenario_eval_run_ids": [],
        "scenario_eval_run_coverage_complete": bool(run_ids),
        "attempts": attempts,
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def _failure_labels(
    *,
    substrate: str,
    trace: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    attempts = [item for item in trace.get("attempts", []) or [] if isinstance(item, Mapping)]
    failures = [attempt for attempt in attempts if not bool(attempt.get("success"))]
    return {
        "schema_version": FAILURE_LABELS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "review_required" if failures else "no_failures_labeled",
        "evaluation_substrate": substrate,
        "label_count": len(failures),
        "failed_attempt_count": len(failures),
        "covered_failed_attempt_ids": sorted(_string(attempt.get("attempt_id")) for attempt in failures),
        "missing_failed_attempt_ids": [],
        "covered_failed_scenario_eval_run_ids": sorted(
            {
                _string(attempt.get("scenario_eval_run_id"))
                for attempt in failures
                if attempt.get("scenario_eval_run_id")
            }
        ),
        "missing_failed_scenario_eval_run_ids": [],
        "failed_run_label_coverage_complete": True,
        "labels": [
            {
                "label_id": f"wam_failure_label_{index:04d}",
                "attempt_id": attempt.get("attempt_id"),
                "rollout_id": attempt.get("rollout_id"),
                "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
                "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
                "variation_name": attempt.get("variation_name"),
                "task_id": attempt.get("task_id"),
                "scenario_id": attempt.get("scenario_id"),
                "policy_id": attempt.get("policy_id"),
                "failure_mode_ids": _string_list(attempt.get("failure_mode_ids")),
                "failure_reason": "fixture_wam_predicted_task_failure",
                "source": "vision_success_labels",
                "status": "review_required",
                "proof_effect": "none_until_review_accepted_or_real_world_validation_supplied",
            }
            for index, attempt in enumerate(failures, start=1)
        ],
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def _prediction_ledgers(
    *,
    substrate: str,
    trace: Mapping[str, Any],
    generated_at: str,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    attempts = [item for item in trace.get("attempts", []) or [] if isinstance(item, Mapping)]
    records = [
        {
            "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
            "variation_name": attempt.get("variation_name"),
            "task_id": attempt.get("task_id"),
            "scenario_id": attempt.get("scenario_id"),
            "policy_id": attempt.get("policy_id"),
            "evaluation_substrate": attempt.get("evaluation_substrate"),
            "predicted_status": "passed" if attempt.get("success") else "failed",
            "predicted_success": bool(attempt.get("success")),
            "failure_mode_ids": _string_list(attempt.get("failure_mode_ids")),
            "world_model_uncertainty": _mapping(attempt.get("metrics")).get(
                "world_model_uncertainty"
            ),
            "actual_status": "needs_real_world_validation",
            "source": f"{substrate}_eval",
        }
        for attempt in attempts
    ]
    prediction = {
        "schema_version": PREDICTION_OUTCOME_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if records else "not_available",
        "evaluation_substrate": substrate,
        "record_count": len(records),
        "records": records,
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }
    calibration = {
        "schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "needs_real_world_outcomes",
        "evaluation_substrate": substrate,
        "record_count": len(records),
        "records": records,
        "sim_vs_real_calibration_score": None,
        "srcc_validation_status": "not_measured",
        "customer_specific_srcc_claimed": False,
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }
    failures = [record for record in records if not record["predicted_success"]]
    breakage = {
        "schema_version": BREAKAGE_LIBRARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "review_required" if failures else "no_breakages_recorded",
        "evaluation_substrate": substrate,
        "record_count": len(failures),
        "records": failures,
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }
    return prediction, calibration, breakage


def _write_wam_artifacts(job_dir: Path, payloads: Mapping[str, Mapping[str, Any]]) -> None:
    for key, payload in payloads.items():
        write_json(job_dir / WAM_ARTIFACT_PATHS[key], payload)


def _policy_scorecard(
    *,
    substrate: str,
    labels: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    label_rows = [dict(item) for item in labels.get("labels", []) or [] if isinstance(item, Mapping)]
    by_policy: Dict[str, list[Dict[str, Any]]] = {}
    for label in label_rows:
        by_policy.setdefault(_string(label.get("policy_id")) or "policy", []).append(label)
    rows: list[Dict[str, Any]] = []
    for policy_id, policy_labels in by_policy.items():
        success_count = sum(1 for label in policy_labels if bool(label.get("task_success")))
        uncertainties = [_number(label.get("uncertainty_score")) for label in policy_labels]
        rows.append(
            {
                "policy_id": policy_id,
                "attempt_count": len(policy_labels),
                "predicted_success_count": success_count,
                "predicted_failure_count": len(policy_labels) - success_count,
                "predicted_success_rate": round(success_count / len(policy_labels), 6)
                if policy_labels
                else 0.0,
                "mean_uncertainty": round(sum(uncertainties) / len(uncertainties), 6)
                if uncertainties
                else None,
                "ood_flag_count": sum(1 for label in policy_labels if _string_list(label.get("ood_flags"))),
                "failure_taxonomy": sorted(
                    {
                        mode
                        for label in policy_labels
                        for mode in _string_list(label.get("failure_mode_ids"))
                    }
                ),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            -_number(row.get("predicted_success_rate")),
            _number(row.get("mean_uncertainty"), 1.0),
            _string(row.get("policy_id")),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return {
        "schema_version": POLICY_RANKING_SCORECARD_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if ranked else "blocked_missing_labels",
        "evaluation_substrate": substrate,
        "ranking_basis": "fixture_vision_success_labels_over_model_derived_wam_rollouts",
        "policy_count": len(ranked),
        "scenario_attempt_count": len(label_rows),
        "policy_rankings": ranked,
        "top_policy_id": ranked[0]["policy_id"] if ranked else None,
        "failure_taxonomy": sorted(
            {
                mode
                for label in label_rows
                for mode in _string_list(label.get("failure_mode_ids"))
            }
        ),
        "uncertainty_ood_summary": {
            "ood_label_count": sum(1 for label in label_rows if _string_list(label.get("ood_flags"))),
            "mean_uncertainty": round(
                sum(_number(label.get("uncertainty_score")) for label in label_rows)
                / len(label_rows),
                6,
            )
            if label_rows
            else None,
        },
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def _claim_boundary(*, substrate: str, generated_at: str) -> Dict[str, Any]:
    return build_wam_eval_claim_boundary(substrate=substrate, generated_at=generated_at)


def _real_world_validation_followup(
    *,
    job_id: str,
    substrate: str,
    scorecard: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": REAL_WORLD_VALIDATION_FOLLOWUP_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "requested_real_world_validation_anchors",
        "evaluation_substrate": substrate,
        "top_policy_id": scorecard.get("top_policy_id"),
        "requested_anchor_rollouts": [
            "real_world_rollouts_for_top_ranked_policy",
            "real_world_rollouts_for_low_ranked_policy",
            "real_world_rollouts_for_high_uncertainty_or_ood_scenarios",
        ],
        "minimum_validation_requirements": {
            "paired_real_outcome_records_required": True,
            "exact_scenario_eval_run_id_join_required": True,
            "policy_or_checkpoint_ids_required": True,
            "owner_evidence_or_operator_attestation_required": True,
        },
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def _srcc_validation_plan(*, job_id: str, substrate: str, generated_at: str) -> Dict[str, Any]:
    return {
        "schema_version": SRCC_VALIDATION_PLAN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "requires_real_world_rollout_anchors",
        "evaluation_substrate": substrate,
        "metrics_to_compute_when_anchors_exist": [
            "spearman_rank_correlation",
            "pearson_success_rate_correlation",
            "mean_absolute_success_rate_error",
            "maximum_rank_violation",
            "failure_mode_agreement",
        ],
        "customer_specific_srcc_claimed": False,
        "blocked_report_reason": "missing_paired_real_world_rollout_outcomes",
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def _customer_handoff_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# WAM Policy Evaluation Handoff",
            "",
            f"Status: `{report.get('status')}`",
            f"Evaluation substrate: `{report.get('evaluation_substrate')}`",
            f"Top policy: `{report.get('top_policy_id')}`",
            "",
            "Generated rollouts and fixture labels are support artifacts. They do not prove real-world success, deployment readiness, robot safety, or customer-specific SRCC.",
            "",
        ]
    )


def _customer_handoff_report(
    *,
    job_id: str,
    substrate: str,
    scorecard: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "wam_customer_handoff_report.v1",
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "generated",
        "evaluation_substrate": substrate,
        "top_policy_id": scorecard.get("top_policy_id"),
        "artifact_paths": {
            key: value
            for key, value in WAM_ARTIFACT_PATHS.items()
            if key not in {"customer_handoff_report_markdown"}
        },
        "reader_boundary": (
            "Generated WAM rollouts are model-derived support artifacts, not raw truth or "
            "deployment approval."
        ),
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def _blocked_wam_artifacts(
    *,
    job_dir: Path,
    job_id: str,
    substrate: str,
    generated_at: str,
    blockers: Sequence[str],
) -> Dict[str, Any]:
    registry = write_evaluation_substrate_registry(job_dir, generated_at=generated_at)
    matrix = _read_optional_mapping(job_dir / "scenario_eval_matrix.json")
    policy_manifest = _read_optional_mapping(job_dir / "policy_package_manifest.json")
    request_payload = _read_optional_mapping(job_dir / "job_request.json")
    policies = _policy_candidates(request=request_payload, policy_manifest=policy_manifest)
    policy_binding = _policy_interface_binding(
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        policy_manifest=policy_manifest,
        policies=policies,
        generated_at=generated_at,
    )
    runtime_package = _provider_runtime_package(
        capture_root=job_dir.parents[2] if len(job_dir.parents) >= 3 else job_dir,
        job_dir=job_dir,
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        scenario_eval_run_count=len(_matrix_runs(matrix)),
        policies=policies,
        generated_at=generated_at,
        artifact_output_uri=None,
        budget_usd=None,
    )
    provider_execution = _provider_execution_manifest(
        substrate=substrate,
        generated_at=generated_at,
        status="blocked",
        command_used=False,
        blockers=blockers,
    )
    provider_cost = _provider_cost_ledger(
        substrate=substrate,
        generated_at=generated_at,
        budget_usd=None,
        status="blocked",
    )
    provider_upload = _provider_artifact_upload_proof(
        substrate=substrate,
        generated_at=generated_at,
        artifact_output_uri=None,
    )
    request = build_wam_evaluation_request(
        job_id=job_id,
        substrate=substrate,
        generated_at=generated_at,
        status="blocked",
        blockers=blockers,
    )
    empty_rollout_manifest = {
        "schema_version": WAM_ROLLOUT_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "blocked",
        "evaluation_substrate": substrate,
        "blockers": list(blockers),
        "rollout_count": 0,
        "rollouts": [],
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }
    empty_results = {
        "schema_version": WAM_ROLLOUT_RESULTS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "blocked",
        "evaluation_substrate": substrate,
        "blockers": list(blockers),
        "rollout_count": 0,
        "rollouts": [],
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }
    labels = build_fixture_vision_success_labels(
        rollout_results=empty_results,
        generated_at=generated_at,
    )
    trace = _normalized_attempt_trace(substrate=substrate, labels=labels, generated_at=generated_at)
    failure_labels = _failure_labels(
        substrate=substrate,
        trace=trace,
        generated_at=generated_at,
    )
    scorecard = _policy_scorecard(substrate=substrate, labels=labels, generated_at=generated_at)
    claim_boundary = _claim_boundary(substrate=substrate, generated_at=generated_at)
    review_queue = _vision_review_queue(
        substrate=substrate,
        labels=labels,
        generated_at=generated_at,
    )
    followup = _real_world_validation_followup(
        job_id=job_id,
        substrate=substrate,
        scorecard=scorecard,
        generated_at=generated_at,
    )
    srcc_plan = _srcc_validation_plan(job_id=job_id, substrate=substrate, generated_at=generated_at)
    anchor_manifest = _real_world_anchor_manifest(
        job_dir=job_dir,
        substrate=substrate,
        scorecard=scorecard,
        generated_at=generated_at,
    )
    validation_envelope = _customer_validation_envelope(
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        scorecard=scorecard,
        anchor_manifest=anchor_manifest,
        generated_at=generated_at,
    )
    production_ops = _production_ops_manifest(
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        provider_execution=provider_execution,
        generated_at=generated_at,
        artifact_output_uri=None,
        budget_usd=None,
    )
    cross_check_plan = _classical_sim_cross_check_plan(
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        scorecard=scorecard,
        generated_at=generated_at,
    )
    payloads = {
        "wam_provider_runtime_package": runtime_package,
        "wam_provider_execution_manifest": provider_execution,
        "wam_provider_cost_control_ledger": provider_cost,
        "wam_provider_artifact_upload_proof": provider_upload,
        "wam_policy_interface_binding": policy_binding,
        "wam_evaluation_request": request,
        "wam_rollout_manifest": empty_rollout_manifest,
        "wam_rollout_results": empty_results,
        "vision_success_labels": labels,
        "normalized_attempt_trace": trace,
        "failure_labels": failure_labels,
        "policy_ranking_scorecard": scorecard,
        "wam_eval_claim_boundary": claim_boundary,
        "wam_vision_success_review_queue": review_queue,
        "real_world_validation_followup_request": followup,
        "srcc_validation_plan": srcc_plan,
        "wam_real_world_validation_anchor_manifest": anchor_manifest,
        "wam_customer_validation_envelope": validation_envelope,
        "wam_production_ops_manifest": production_ops,
        "wam_classical_sim_cross_check_plan": cross_check_plan,
    }
    _write_wam_artifacts(job_dir, payloads)
    return {
        "status": "blocked",
        "blockers": list(blockers),
        "evaluation_substrate_registry": registry,
        **payloads,
        "artifact_paths": dict(WAM_ARTIFACT_PATHS),
    }


def run_wam_eval_job(
    *,
    capture_root: str | Path,
    job_dir: str | Path,
    evaluation_substrate: str = "fixture_wam",
    allow_live_provider: bool = False,
    provider_command: str | None = None,
    artifact_output_uri: str | None = None,
    budget_usd: float | None = None,
    max_retries: int = 0,
    timeout_seconds: int = 120,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Run the deterministic local WAM evaluator for an existing robot-eval job."""

    resolved_capture_root = Path(capture_root).resolve()
    resolved_job_dir = Path(job_dir).resolve()
    ensure_dir(resolved_job_dir)
    generated = generated_at or utc_now_iso()
    substrate = normalize_evaluation_substrate(evaluation_substrate)
    job_id = resolved_job_dir.name
    if substrate not in WAM_EVALUATION_SUBSTRATES:
        return _blocked_wam_artifacts(
            job_dir=resolved_job_dir,
            job_id=job_id,
            substrate=substrate,
            generated_at=generated,
            blockers=["evaluation_substrate_is_not_wam"],
        )

    matrix = _read_optional_mapping(resolved_job_dir / "scenario_eval_matrix.json")
    policy_manifest = _read_optional_mapping(resolved_job_dir / "policy_package_manifest.json")
    request_payload = _read_optional_mapping(resolved_job_dir / "job_request.json")
    runs = _matrix_runs(matrix)
    policies = _policy_candidates(request=request_payload, policy_manifest=policy_manifest)
    blockers: list[str] = []
    if not runs:
        blockers.append("scenario_eval_matrix_missing_or_empty")
    if not policies:
        blockers.append("policy_candidates_missing")
    if blockers:
        return _blocked_wam_artifacts(
            job_dir=resolved_job_dir,
            job_id=job_id,
            substrate=substrate,
            generated_at=generated,
            blockers=blockers,
        )

    registry = write_evaluation_substrate_registry(resolved_job_dir, generated_at=generated)
    policy_binding = _policy_interface_binding(
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        policy_manifest=policy_manifest,
        policies=policies,
        generated_at=generated,
    )
    provider_command_text = _substrate_provider_command(substrate, provider_command)
    provider_runtime_package = _provider_runtime_package(
        capture_root=resolved_capture_root,
        job_dir=resolved_job_dir,
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        scenario_eval_run_count=len(runs),
        policies=policies,
        generated_at=generated,
        artifact_output_uri=artifact_output_uri,
        budget_usd=budget_usd,
    )
    provider_runtime_package_path = (
        resolved_job_dir / WAM_ARTIFACT_PATHS["wam_provider_runtime_package"]
    )
    write_json(provider_runtime_package_path, provider_runtime_package)
    request = build_wam_evaluation_request(
        job_id=job_id,
        substrate=substrate,
        policy_ids=[_string(policy.get("policy_id")) for policy in policies],
        generated_at=generated,
    )
    rollouts: list[Dict[str, Any]] = []
    provider_execution_detail: Dict[str, Any] = {}
    provider_payload: Dict[str, Any] = {}
    provider_execution_status = "not_required_fixture"
    provider_execution_blockers: list[str] = []
    provider_command_used = False
    if substrate == "fixture_wam":
        for policy in policies:
            for run in runs:
                rollouts.append(
                    _rollout_for_run(
                        job_dir=resolved_job_dir,
                        substrate=substrate,
                        policy=policy,
                        run=run,
                        index=len(rollouts) + 1,
                        generated_at=generated,
                    )
                )
    else:
        provider_execution_blockers.extend(
            _live_provider_gate_blockers(allow_live_provider=allow_live_provider)
        )
        auth_status = _provider_auth_status(substrate)
        if not provider_command_text:
            provider_execution_blockers.append(
                f"{substrate}_provider_adapter_not_configured_for_local_run"
            )
        if not auth_status["auth_available"]:
            provider_execution_blockers.append(f"{substrate}_auth_env_missing")
        if provider_execution_blockers:
            return _blocked_wam_artifacts(
                job_dir=resolved_job_dir,
                job_id=job_id,
                substrate=substrate,
                generated_at=generated,
                blockers=provider_execution_blockers,
            )
        output_path = resolved_job_dir / "wam_provider" / "wam_provider_output.json"
        attempts = 0
        last_status = "blocked"
        last_payload: Any = {}
        last_detail: Dict[str, Any] = {}
        for attempt in range(max(0, max_retries) + 1):
            attempts = attempt + 1
            last_status, last_payload, last_detail = _run_provider_command(
                command_text=provider_command_text,
                runtime_package_path=provider_runtime_package_path,
                output_path=output_path,
                substrate=substrate,
                artifact_output_uri=artifact_output_uri,
                timeout_seconds=timeout_seconds,
            )
            rollouts = _normalize_provider_rollouts(
                payload=last_payload,
                substrate=substrate,
                generated_at=generated,
            )
            if last_status == "completed" and rollouts:
                break
        provider_command_used = True
        provider_execution_status = "completed" if last_status == "completed" and rollouts else "blocked"
        provider_payload = _mapping(last_payload)
        provider_execution_detail = {
            **last_detail,
            "normalized_rollout_count": len(rollouts),
            "attempt_count": attempts,
        }
        provider_execution_blockers.extend(_string_list(last_detail.get("blockers")))
        if not rollouts:
            provider_execution_blockers.append("wam_provider_output_missing_rollouts")
    rollout_manifest = _rollout_manifest(
        job_id=job_id,
        substrate=substrate,
        rollouts=rollouts,
        generated_at=generated,
    )
    rollout_results = _rollout_results(
        job_id=job_id,
        substrate=substrate,
        rollouts=rollouts,
        generated_at=generated,
    )
    labels = build_fixture_vision_success_labels(
        rollout_results=rollout_results,
        generated_at=generated,
    )
    trace = _normalized_attempt_trace(substrate=substrate, labels=labels, generated_at=generated)
    failure_labels = _failure_labels(
        substrate=substrate,
        trace=trace,
        generated_at=generated,
    )
    prediction, calibration, breakage = _prediction_ledgers(
        substrate=substrate,
        trace=trace,
        generated_at=generated,
    )
    scorecard = _policy_scorecard(substrate=substrate, labels=labels, generated_at=generated)
    claim_boundary = _claim_boundary(substrate=substrate, generated_at=generated)
    if provider_command_used and provider_execution_status == "completed":
        claim_boundary = {**claim_boundary, "live_provider_calls_performed": True}
    review_queue = _vision_review_queue(
        substrate=substrate,
        labels=labels,
        generated_at=generated,
    )
    followup = _real_world_validation_followup(
        job_id=job_id,
        substrate=substrate,
        scorecard=scorecard,
        generated_at=generated,
    )
    srcc_plan = _srcc_validation_plan(job_id=job_id, substrate=substrate, generated_at=generated)
    anchor_manifest = _real_world_anchor_manifest(
        job_dir=resolved_job_dir,
        substrate=substrate,
        scorecard=scorecard,
        generated_at=generated,
    )
    validation_envelope = _customer_validation_envelope(
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        scorecard=scorecard,
        anchor_manifest=anchor_manifest,
        generated_at=generated,
    )
    provider_execution = _provider_execution_manifest(
        substrate=substrate,
        generated_at=generated,
        status=provider_execution_status
        if provider_execution_status != "not_required_fixture"
        else "not_required_fixture",
        command_used=provider_command_used,
        detail=provider_execution_detail,
        blockers=provider_execution_blockers,
        attempt_count=1
        if substrate == "fixture_wam"
        else int(provider_execution_detail.get("attempt_count") or 1),
        max_retries=max_retries,
    )
    provider_cost = _provider_cost_ledger(
        substrate=substrate,
        generated_at=generated,
        budget_usd=budget_usd,
        status=provider_execution_status,
        duration_seconds=_number(provider_execution_detail.get("duration_seconds"), None),
    )
    provider_upload = _provider_artifact_upload_proof(
        substrate=substrate,
        generated_at=generated,
        artifact_output_uri=artifact_output_uri,
        provider_payload=provider_payload,
    )
    production_ops = _production_ops_manifest(
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        provider_execution=provider_execution,
        generated_at=generated,
        artifact_output_uri=artifact_output_uri,
        budget_usd=budget_usd,
    )
    cross_check_plan = _classical_sim_cross_check_plan(
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        scorecard=scorecard,
        generated_at=generated,
    )
    handoff = _customer_handoff_report(
        job_id=job_id,
        substrate=substrate,
        scorecard=scorecard,
        generated_at=generated,
    )
    payloads = {
        "wam_provider_runtime_package": provider_runtime_package,
        "wam_provider_execution_manifest": provider_execution,
        "wam_provider_cost_control_ledger": provider_cost,
        "wam_provider_artifact_upload_proof": provider_upload,
        "wam_policy_interface_binding": policy_binding,
        "wam_evaluation_request": request,
        "wam_rollout_manifest": rollout_manifest,
        "wam_rollout_results": rollout_results,
        "vision_success_labels": labels,
        "normalized_attempt_trace": trace,
        "failure_labels": failure_labels,
        "prediction_outcome_ledger": prediction,
        "calibration_report": calibration,
        "breakage_library": breakage,
        "policy_ranking_scorecard": scorecard,
        "wam_eval_claim_boundary": claim_boundary,
        "wam_vision_success_review_queue": review_queue,
        "real_world_validation_followup_request": followup,
        "srcc_validation_plan": srcc_plan,
        "wam_real_world_validation_anchor_manifest": anchor_manifest,
        "wam_customer_validation_envelope": validation_envelope,
        "wam_production_ops_manifest": production_ops,
        "wam_classical_sim_cross_check_plan": cross_check_plan,
        "customer_handoff_report": handoff,
    }
    _write_wam_artifacts(resolved_job_dir, payloads)
    write_text(
        resolved_job_dir / WAM_ARTIFACT_PATHS["customer_handoff_report_markdown"],
        _customer_handoff_markdown(handoff),
    )
    return {
        "status": "completed" if rollouts else "blocked",
        "blockers": provider_execution_blockers,
        "evaluation_substrate": substrate,
        "evaluation_substrate_registry": registry,
        **payloads,
        "artifact_paths": dict(WAM_ARTIFACT_PATHS),
        "claim_boundary": claim_boundary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic fixture WAM eval job")
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--evaluation-substrate", default="fixture_wam")
    parser.add_argument("--allow-live-provider", action="store_true")
    parser.add_argument("--provider-command")
    parser.add_argument("--artifact-output-uri")
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args(argv)
    result = run_wam_eval_job(
        capture_root=args.capture_root,
        job_dir=args.job_dir,
        evaluation_substrate=args.evaluation_substrate,
        allow_live_provider=args.allow_live_provider,
        provider_command=args.provider_command,
        artifact_output_uri=args.artifact_output_uri,
        budget_usd=args.budget_usd,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"[wam-eval] status={result['status']}")
    print(f"[wam-eval] job_dir={Path(args.job_dir).resolve()}")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
