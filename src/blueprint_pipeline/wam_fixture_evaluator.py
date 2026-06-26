"""Deterministic fixture WAM evaluator for local robot-eval job tests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json, write_text
from .failure_diagnosis_contract import (
    FAILURE_LABEL_PROOF_EFFECT,
    dedupe as _dedupe_refs,
    evidence_refs as _failure_evidence_refs,
    failure_root_cause_category as _failure_root_cause_category,
    frame_or_clip_refs as _failure_frame_or_clip_refs,
    remediation_candidate as _failure_remediation_candidate,
    review_status_for_failure_label as _failure_review_status,
)
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
from .wam_vision_success_judge import (
    FIXTURE_VISUAL_REVIEW_BLOCKER,
    FIXTURE_VISUAL_SMOKE_STATUS,
    build_fixture_vision_success_labels,
)


WAM_ROLLOUT_MANIFEST_SCHEMA_VERSION = "wam_rollout_manifest.v1"
WAM_ROLLOUT_RESULTS_SCHEMA_VERSION = "wam_rollout_results.v1"
POLICY_RANKING_SCORECARD_SCHEMA_VERSION = "policy_ranking_scorecard.v1"
POLICY_RANKING_TIE_BAND = 0.05
POLICY_RANKING_HIGH_UNCERTAINTY_THRESHOLD = 0.65
POLICY_RANKING_HIGH_OOD_RATE_THRESHOLD = 0.5
REAL_WORLD_VALIDATION_FOLLOWUP_SCHEMA_VERSION = "real_world_validation_followup_request.v1"
SRCC_VALIDATION_PLAN_SCHEMA_VERSION = "srcc_validation_plan.v1"
NORMALIZED_ATTEMPT_TRACE_SCHEMA_VERSION = "robot_eval_job_normalized_attempt_trace.v1"
FAILURE_LABELS_SCHEMA_VERSION = "robot_eval_job_failure_labels.v1"
PREDICTION_OUTCOME_LEDGER_SCHEMA_VERSION = "robot_eval_job_prediction_outcome_ledger.v1"
CALIBRATION_REPORT_SCHEMA_VERSION = "robot_eval_job_calibration_report.v1"
BREAKAGE_LIBRARY_SCHEMA_VERSION = "robot_eval_job_breakage_library.v1"
ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION = "accepted_real_world_anchor.v1"
ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS = (
    "scenario_eval_run_id",
    "policy_id",
    "task_id",
    "scenario_variation_instance_id",
)
CANDIDATE_SELECTION_REPORT_SCHEMA_VERSION = "wam_candidate_selection_report.v1"
CANDIDATE_SELECTION_AMBIGUITY_SUCCESS_RATE_MARGIN = 0.05
CANDIDATE_SELECTION_HIGH_UNCERTAINTY_THRESHOLD = 0.5

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
    "candidate_selection_report": "candidate_selection_report.json",
    "candidate_selection_report_markdown": "candidate_selection_report.md",
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


def _optional_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ordered_unique_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = _string(value)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def _dedupe(values: Sequence[str]) -> list[str]:
    return _ordered_unique_strings(values)


def _safe_id(value: Any, *, fallback: str = "item") -> str:
    text = _string(value) or fallback
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in text)
    return "_".join(part for part in cleaned.split("_") if part) or fallback


def _visual_review_gate_from_labels(labels: Mapping[str, Any]) -> Dict[str, Any]:
    label_rows = [row for row in labels.get("labels", []) or [] if isinstance(row, Mapping)]
    visual_smoke_statuses = _string_list(labels.get("visual_smoke_statuses"))
    if not visual_smoke_statuses:
        visual_smoke_statuses = sorted(
            {
                _string(row.get("visual_smoke_status"))
                for row in label_rows
                if _string(row.get("visual_smoke_status"))
            }
        )
    visual_rollout_useful = bool(
        labels.get("visual_rollout_useful_for_task_success_review")
    ) or bool(
        label_rows
        and all(
            bool(row.get("visual_rollout_useful_for_task_success_review"))
            for row in label_rows
        )
    )
    fixture_only = bool(labels.get("fixture_evaluator_only")) or any(
        bool(row.get("fixture_evaluator_only")) for row in label_rows
    )
    review_grade_success_labels = bool(
        labels.get("review_grade_success_labels")
        and visual_rollout_useful
        and not fixture_only
    )
    blockers = _string_list(labels.get("visual_review_blockers"))
    for row in label_rows:
        blockers.extend(_string_list(row.get("visual_review_blockers")))
    if fixture_only and FIXTURE_VISUAL_REVIEW_BLOCKER not in blockers:
        blockers.append(FIXTURE_VISUAL_REVIEW_BLOCKER)
    if not visual_rollout_useful and not blockers:
        blockers.append("generated_rollout_visual_smoke_missing_or_failed")
    blockers = sorted(set(blockers))
    status = (
        "review_grade_success_labels_available"
        if review_grade_success_labels
        else "fixture_evaluator_only"
        if fixture_only
        else "blocked_visual_review_required"
    )
    return {
        "status": status,
        "visual_smoke_status": labels.get("visual_smoke_status")
        or (
            visual_smoke_statuses[0]
            if len(visual_smoke_statuses) == 1
            else "mixed_visual_smoke_statuses"
            if visual_smoke_statuses
            else FIXTURE_VISUAL_SMOKE_STATUS
        ),
        "visual_smoke_statuses": visual_smoke_statuses,
        "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
        "review_grade_visual_evidence_available": bool(
            labels.get("review_grade_visual_evidence_available") or visual_rollout_useful
        ),
        "review_grade_success_labels": review_grade_success_labels,
        "fixture_evaluator_only": fixture_only,
        "blockers": blockers,
    }


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
            "rank_fidelity_result_proven": False,
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
                "confidence": label.get("confidence"),
                "evidence_refs": _failure_evidence_refs(
                    label,
                    extra_refs=("vision_success_labels.json",),
                ),
                "source_trace_refs": _dedupe_refs(["vision_success_labels.json"]),
                "frame_or_clip_refs": _failure_frame_or_clip_refs(label),
                "visual_smoke_ref": label.get("visual_smoke_ref")
                or label.get("visualSmokeRef"),
                "visual_smoke_status": label.get("visual_smoke_status"),
                "visual_rollout_useful_for_task_success_review": bool(
                    label.get("visual_rollout_useful_for_task_success_review")
                ),
                "visual_review_blockers": _string_list(label.get("visual_review_blockers")),
                "fixture_evaluator_only": bool(label.get("fixture_evaluator_only")),
                "review_grade_visual_evidence_available": bool(
                    label.get("review_grade_visual_evidence_available")
                ),
                "review_grade_success_label": bool(label.get("review_grade_success_label")),
                "generated_wam_rollout": True,
                "model_derived_support_artifact": True,
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
                    "visual_smoke_required_for_review_grade_success_label": True,
                    "visual_rollout_useful_for_task_success_review": bool(
                        label.get("visual_rollout_useful_for_task_success_review")
                    ),
                    "fixture_evaluator_only": bool(label.get("fixture_evaluator_only")),
                    "review_grade_success_label": bool(
                        label.get("review_grade_success_label")
                    ),
                    "simulator_execution_proven": False,
                    "robot_policy_execution_proven": False,
                    "rank_fidelity_result_proven": False,
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
    labels: list[Dict[str, Any]] = []
    labels_missing_failure_modes: list[str] = []
    labels_missing_evidence_refs: list[str] = []
    labels_missing_review_status: list[str] = []
    nonreviewable_labels: list[str] = []
    visual_smoke_statuses: list[str] = []
    visual_review_blockers: list[str] = []
    for index, attempt in enumerate(failures, start=1):
        label_id = f"wam_failure_label_{index:04d}"
        failure_mode_ids = _string_list(attempt.get("failure_mode_ids"))
        frame_refs = _failure_frame_or_clip_refs(attempt)
        source_trace_refs = _dedupe_refs(
            [
                "normalized_attempt_trace.json",
                *_string_list(attempt.get("source_trace_refs")),
                "vision_success_labels.json",
            ]
        )
        evidence_refs = _failure_evidence_refs(
            attempt,
            extra_refs=tuple(source_trace_refs),
        )
        visual_smoke_ref = (
            _string(attempt.get("visual_smoke_ref") or attempt.get("visualSmokeRef")) or None
        )
        review_status = _failure_review_status(
            supplied_review_status=attempt.get("review_status"),
            supplied_status=attempt.get("status"),
            generated_rollout=True,
            frame_or_clip_ref_count=len(frame_refs),
        )
        root_cause_category = _failure_root_cause_category(
            failure_mode_ids,
            ood_flags=_string_list(attempt.get("ood_flags")),
            failure_reason="fixture_wam_predicted_task_failure",
        )
        unknown_when_evidence_weak = bool(
            not frame_refs or not evidence_refs or review_status == "non_reviewable_failure_hypothesis"
        )
        visual_smoke_status = (
            _string(attempt.get("visual_smoke_status")) or FIXTURE_VISUAL_SMOKE_STATUS
        )
        visual_rollout_useful = bool(
            attempt.get("visual_rollout_useful_for_task_success_review")
        )
        attempt_visual_blockers = _string_list(attempt.get("visual_review_blockers"))
        fixture_only = bool(attempt.get("fixture_evaluator_only"))
        if fixture_only and FIXTURE_VISUAL_REVIEW_BLOCKER not in attempt_visual_blockers:
            attempt_visual_blockers.append(FIXTURE_VISUAL_REVIEW_BLOCKER)
        if not visual_rollout_useful and not attempt_visual_blockers:
            attempt_visual_blockers.append("generated_rollout_visual_smoke_missing_or_failed")
        visual_smoke_statuses.append(visual_smoke_status)
        visual_review_blockers.extend(attempt_visual_blockers)
        label = {
            "label_id": label_id,
            "attempt_id": attempt.get("attempt_id"),
            "rollout_id": attempt.get("rollout_id"),
            "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
            "variation_name": attempt.get("variation_name"),
            "task_id": attempt.get("task_id"),
            "scenario_id": attempt.get("scenario_id"),
            "policy_id": attempt.get("policy_id"),
            "evaluation_substrate": substrate,
            "failure_mode_ids": failure_mode_ids,
            "failure_reason": "fixture_wam_predicted_task_failure",
            "source": "vision_success_labels",
            "evidence_refs": evidence_refs,
            "source_trace_refs": source_trace_refs,
            "frame_or_clip_refs": frame_refs,
            "visual_smoke_ref": visual_smoke_ref,
            "confidence": attempt.get("confidence"),
            "status": "review_required",
            "review_status": review_status,
            "reviewer_acceptance_required": True,
            "root_cause_category": root_cause_category,
            "remediation_candidate": _failure_remediation_candidate(
                root_cause_category,
                failure_mode_ids,
            ),
            "unknown_when_evidence_weak": unknown_when_evidence_weak,
            "non_reviewable_failure_hypothesis": (
                review_status == "non_reviewable_failure_hypothesis"
            ),
            "visual_smoke_status": visual_smoke_status,
            "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
            "visual_review_blockers": sorted(set(attempt_visual_blockers)),
            "fixture_evaluator_only": fixture_only,
            "review_grade_failure_diagnosis": False,
            "authoritative_failure_diagnosis": False,
            "generated_wam_rollout": True,
            "model_derived_support_artifact": True,
            "proof_effect": FAILURE_LABEL_PROOF_EFFECT,
        }
        if not failure_mode_ids:
            labels_missing_failure_modes.append(label_id)
        if not evidence_refs:
            labels_missing_evidence_refs.append(label_id)
        if not review_status:
            labels_missing_review_status.append(label_id)
        if review_status == "non_reviewable_failure_hypothesis":
            nonreviewable_labels.append(label_id)
        if fixture_only or not visual_rollout_useful:
            nonreviewable_labels.append(label_id)
        labels.append(label)
    coverage_blockers = []
    if labels_missing_failure_modes:
        coverage_blockers.append("failure_labels_missing_failure_mode_ids")
    if labels_missing_evidence_refs:
        coverage_blockers.append("failure_labels_missing_evidence_refs")
    if labels_missing_review_status:
        coverage_blockers.append("failure_labels_missing_review_status")
    deduped_nonreviewable_labels = sorted(set(nonreviewable_labels))
    visual_review_blockers = sorted(set(visual_review_blockers))
    visual_rollout_useful_for_review = bool(failures) and not visual_review_blockers and all(
        bool(attempt.get("visual_rollout_useful_for_task_success_review"))
        for attempt in failures
    )
    return {
        "schema_version": FAILURE_LABELS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "review_required" if failures else "no_failures_labeled",
        "evaluation_substrate": substrate,
        "visual_smoke_status": visual_smoke_statuses[0]
        if len(set(visual_smoke_statuses)) == 1
        else "mixed_visual_smoke_statuses"
        if visual_smoke_statuses
        else FIXTURE_VISUAL_SMOKE_STATUS,
        "visual_smoke_statuses": sorted(set(visual_smoke_statuses)),
        "visual_rollout_useful_for_task_success_review": visual_rollout_useful_for_review,
        "visual_review_blockers": visual_review_blockers,
        "fixture_evaluator_only": any(bool(attempt.get("fixture_evaluator_only")) for attempt in failures),
        "review_grade_failure_diagnosis": False,
        "authoritative_failure_diagnosis": False,
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
        "failure_diagnosis_coverage_complete": not coverage_blockers,
        "failure_diagnosis_review_complete": not deduped_nonreviewable_labels,
        "failure_diagnosis_complete": bool(
            failures and not coverage_blockers and not deduped_nonreviewable_labels
        )
        if failures
        else True,
        "failure_diagnosis_blockers": [
            *coverage_blockers,
            *visual_review_blockers,
            *(
                ["failure_labels_nonreviewable_failure_hypotheses"]
                if deduped_nonreviewable_labels
                else []
            ),
        ],
        "labels_missing_failure_mode_ids": labels_missing_failure_modes,
        "labels_missing_evidence_refs": labels_missing_evidence_refs,
        "labels_missing_review_status": labels_missing_review_status,
        "nonreviewable_failure_hypothesis_label_ids": deduped_nonreviewable_labels,
        "labels": labels,
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def _prediction_ledgers(
    *,
    substrate: str,
    trace: Mapping[str, Any],
    failure_labels: Mapping[str, Any] | None = None,
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
        "status": "not_measured",
        "evaluation_substrate": substrate,
        "record_count": len(records),
        "records": records,
        "accepted_anchor_schema": {
            "schema_version": ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION,
            "join_keys": list(ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS),
            "required_prediction_fields": [
                *ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS,
                "predicted_success",
            ],
            "required_actual_fields": [
                *ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS,
                "actual_success",
                "owner_evidence_or_operator_attestation",
            ],
        },
        "accepted_anchor_count": 0,
        "sim_vs_real_calibration_score": None,
        "spearman_rank_correlation": None,
        "pearson_success_rate_correlation": None,
        "mean_maximum_rank_violation": None,
        "mmrv": None,
        "mean_absolute_success_rate_error": None,
        "confidence_intervals": {},
        "blockers": ["insufficient_anchor_count", "unmatched_prediction_rows"]
        if records
        else ["insufficient_anchor_count"],
        "srcc_validation_status": "not_measured",
        "customer_specific_srcc_claimed": False,
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }
    failures = [record for record in records if not record["predicted_success"]]
    label_rows = [
        dict(label)
        for label in (failure_labels or {}).get("labels", []) or []
        if isinstance(label, Mapping)
    ]
    labels_by_attempt = {
        _string(label.get("attempt_id")): label
        for label in label_rows
        if _string(label.get("attempt_id"))
    }
    labels_by_run = {
        _string(label.get("scenario_eval_run_id")): label
        for label in label_rows
        if _string(label.get("scenario_eval_run_id"))
    }
    aggregation_map: Dict[tuple[str, str, str, str, str], Dict[str, Any]] = {}
    dominant_map: Dict[str, Dict[str, Any]] = {}
    for record in failures:
        label = labels_by_run.get(_string(record.get("scenario_eval_run_id"))) or labels_by_attempt.get(
            _string(record.get("attempt_id"))
        )
        failure_mode_ids = _string_list(
            (label or {}).get("failure_mode_ids") if label else record.get("failure_mode_ids")
        ) or ["unknown_failure_mode"]
        root_cause = _string((label or {}).get("root_cause_category")) or _failure_root_cause_category(
            failure_mode_ids,
            failure_reason=_string((label or {}).get("failure_reason")),
        )
        evidence_refs = _failure_evidence_refs(label or record)
        media_refs = _failure_frame_or_clip_refs(label or record)
        exemplar = {
            "attempt_id": (label or {}).get("attempt_id") or record.get("attempt_id"),
            "scenario_eval_run_id": record.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": record.get("scenario_variation_instance_id"),
            "variation_name": record.get("variation_name"),
            "policy_id": record.get("policy_id"),
            "task_id": record.get("task_id"),
            "scenario_id": record.get("scenario_id"),
            "failure_mode_ids": failure_mode_ids,
            "root_cause_category": root_cause,
            "evidence_refs": evidence_refs,
            "frame_or_clip_refs": media_refs,
            "visual_smoke_ref": (label or {}).get("visual_smoke_ref"),
            "review_status": (label or {}).get("review_status"),
        }
        for failure_mode_id in failure_mode_ids:
            key = (
                _string(record.get("policy_id")) or "unknown_policy",
                _string(record.get("task_id")) or "unknown_task",
                _string(record.get("scenario_id")) or "unknown_scenario",
                failure_mode_id,
                root_cause,
            )
            bucket = aggregation_map.setdefault(
                key,
                {
                    "policy_id": key[0],
                    "task_id": key[1],
                    "scenario_id": key[2],
                    "failure_mode_id": key[3],
                    "root_cause_category": key[4],
                    "failed_attempt_count": 0,
                    "scenario_eval_run_ids": [],
                    "exemplar_failed_attempts": [],
                    "media_refs": [],
                    "evidence_refs": [],
                },
            )
            bucket["failed_attempt_count"] += 1
            bucket["scenario_eval_run_ids"] = _dedupe_refs(
                [
                    *bucket["scenario_eval_run_ids"],
                    _string(record.get("scenario_eval_run_id")),
                ]
            )
            if len(bucket["exemplar_failed_attempts"]) < 3:
                bucket["exemplar_failed_attempts"].append(exemplar)
            bucket["media_refs"] = _dedupe_refs([*bucket["media_refs"], *media_refs])
            bucket["evidence_refs"] = _dedupe_refs([*bucket["evidence_refs"], *evidence_refs])
            dominant = dominant_map.setdefault(
                failure_mode_id,
                {
                    "failure_mode_id": failure_mode_id,
                    "failed_attempt_count": 0,
                    "root_cause_categories": [],
                    "exemplar_failed_attempts": [],
                    "media_refs": [],
                    "evidence_refs": [],
                },
            )
            dominant["failed_attempt_count"] += 1
            dominant["root_cause_categories"] = _dedupe_refs(
                [*dominant["root_cause_categories"], root_cause]
            )
            if len(dominant["exemplar_failed_attempts"]) < 3:
                dominant["exemplar_failed_attempts"].append(exemplar)
            dominant["media_refs"] = _dedupe_refs([*dominant["media_refs"], *media_refs])
            dominant["evidence_refs"] = _dedupe_refs([*dominant["evidence_refs"], *evidence_refs])
    aggregations = sorted(
        aggregation_map.values(),
        key=lambda row: (
            -int(row["failed_attempt_count"]),
            _string(row["policy_id"]),
            _string(row["task_id"]),
            _string(row["scenario_id"]),
            _string(row["failure_mode_id"]),
            _string(row["root_cause_category"]),
        ),
    )
    dominant_failure_modes = sorted(
        dominant_map.values(),
        key=lambda row: (-int(row["failed_attempt_count"]), _string(row["failure_mode_id"])),
    )
    breakage = {
        "schema_version": BREAKAGE_LIBRARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "review_required" if failures else "no_breakages_recorded",
        "evaluation_substrate": substrate,
        "record_count": len(failures),
        "records": failures,
        "aggregation_keys": [
            "policy_id",
            "task_id",
            "scenario_id",
            "failure_mode_id",
            "root_cause_category",
        ],
        "aggregation_count": len(aggregations),
        "aggregations": aggregations,
        "dominant_failure_modes": dominant_failure_modes,
        "dominant_failure_mode_id": dominant_failure_modes[0]["failure_mode_id"]
        if dominant_failure_modes
        else None,
        "source_failure_labels": "failure_labels.json",
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
    required_scenario_eval_run_ids: Sequence[str] = (),
    policy_ids: Sequence[str] = (),
) -> Dict[str, Any]:
    label_rows = [dict(item) for item in labels.get("labels", []) or [] if isinstance(item, Mapping)]
    visual_review_gate = _visual_review_gate_from_labels(labels)
    by_policy: Dict[str, list[Dict[str, Any]]] = {}
    for label in label_rows:
        by_policy.setdefault(_string(label.get("policy_id")) or "policy", []).append(label)
    required_run_ids = _ordered_unique_strings(required_scenario_eval_run_ids)
    if not required_run_ids:
        required_run_ids = _ordered_unique_strings(
            [
                label.get("scenario_eval_run_id")
                for label in label_rows
                if _string(label.get("scenario_eval_run_id"))
            ]
        )
    declared_policy_ids = _ordered_unique_strings(
        [
            *policy_ids,
            *[
                _string(label.get("policy_id")) or "policy"
                for label in label_rows
            ],
        ]
    )
    rows: list[Dict[str, Any]] = []
    per_policy_coverage: list[Dict[str, Any]] = []
    missing_by_policy: Dict[str, list[str]] = {}
    extra_by_policy: Dict[str, list[str]] = {}
    attempt_count_by_policy: Dict[str, int] = {}
    duplicate_required_attempts_by_policy: Dict[str, list[str]] = {}
    required_run_set = set(required_run_ids)
    for policy_id in declared_policy_ids:
        policy_labels = by_policy.get(policy_id, [])
        observed_run_ids = _ordered_unique_strings(
            [label.get("scenario_eval_run_id") for label in policy_labels]
        )
        run_attempt_counts = {
            run_id: sum(
                1
                for label in policy_labels
                if _string(label.get("scenario_eval_run_id")) == run_id
            )
            for run_id in observed_run_ids
        }
        covered_required_ids = [
            run_id for run_id in required_run_ids if run_id in set(observed_run_ids)
        ]
        missing_ids = [run_id for run_id in required_run_ids if run_id not in set(observed_run_ids)]
        extra_ids = sorted(set(observed_run_ids) - required_run_set) if required_run_ids else []
        duplicate_required_ids = [
            run_id
            for run_id, count in run_attempt_counts.items()
            if run_id in required_run_set and count > 1
        ]
        attempt_count = len(policy_labels)
        expected_attempt_count = len(required_run_ids)
        policy_coverage_complete = bool(
            required_run_ids
            and not missing_ids
            and not extra_ids
            and not duplicate_required_ids
            and attempt_count == expected_attempt_count
        )
        missing_by_policy[policy_id] = missing_ids
        extra_by_policy[policy_id] = extra_ids
        attempt_count_by_policy[policy_id] = attempt_count
        duplicate_required_attempts_by_policy[policy_id] = duplicate_required_ids
        per_policy_coverage.append(
            {
                "policy_id": policy_id,
                "required_scenario_eval_run_ids": list(required_run_ids),
                "covered_scenario_eval_run_ids": covered_required_ids,
                "missing_scenario_eval_run_ids": missing_ids,
                "extra_scenario_eval_run_ids": extra_ids,
                "attempt_count": attempt_count,
                "expected_attempt_count": expected_attempt_count,
                "duplicate_required_scenario_eval_run_ids": duplicate_required_ids,
                "coverage_complete": policy_coverage_complete,
            }
        )
        success_count = sum(1 for label in policy_labels if bool(label.get("task_success")))
        uncertainties = [
            value
            for label in policy_labels
            for value in [_optional_number(label.get("uncertainty_score"))]
            if value is not None
        ]
        ood_flag_count = sum(1 for label in policy_labels if _string_list(label.get("ood_flags")))
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
                "ood_flag_count": ood_flag_count,
                "ood_rate": round(ood_flag_count / len(policy_labels), 6)
                if policy_labels
                else 0.0,
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
    score_range_blockers: list[str] = []
    for label in label_rows:
        uncertainty = _optional_number(label.get("uncertainty_score"))
        if uncertainty is not None and not 0.0 <= uncertainty <= 1.0:
            score_range_blockers.append("uncertainty_score_out_of_range")
        confidence = _optional_number(label.get("confidence"))
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            score_range_blockers.append("confidence_score_out_of_range")
    score_ranges_valid = not score_range_blockers
    coverage_complete = bool(
        declared_policy_ids
        and required_run_ids
        and all(item["coverage_complete"] for item in per_policy_coverage)
    )
    top_policy_margin: float | None = None
    if len(ranked) >= 2:
        top_policy_margin = round(
            _number(ranked[0].get("predicted_success_rate"))
            - _number(ranked[1].get("predicted_success_rate")),
            6,
        )
    ranking_ambiguous = bool(
        len(ranked) >= 2
        and top_policy_margin is not None
        and top_policy_margin <= POLICY_RANKING_TIE_BAND
    )
    uncertainty_penalty_applied = any(
        _optional_number(row.get("mean_uncertainty")) is not None
        and _number(row.get("mean_uncertainty")) >= POLICY_RANKING_HIGH_UNCERTAINTY_THRESHOLD
        for row in ranked
    )
    ood_blockers = [
        f"policy:{row['policy_id']}:ood_rate_high"
        for row in ranked
        if _number(row.get("ood_rate")) >= POLICY_RANKING_HIGH_OOD_RATE_THRESHOLD
        and int(row.get("ood_flag_count") or 0) > 0
    ]
    comparison_blockers: list[str] = []
    if not label_rows:
        comparison_blockers.append("policy_labels_missing")
    if len(declared_policy_ids) < 2:
        comparison_blockers.append("policy_comparison_requires_at_least_two_candidates")
    if not required_run_ids:
        comparison_blockers.append("required_scenario_eval_run_ids_missing")
    if any(missing_by_policy.values()):
        comparison_blockers.append("policy_coverage_missing_required_scenario_eval_run_ids")
    if any(extra_by_policy.values()):
        comparison_blockers.append("policy_coverage_contains_unknown_scenario_eval_run_ids")
    if any(duplicate_required_attempts_by_policy.values()):
        comparison_blockers.append("policy_coverage_duplicate_required_scenario_attempts")
    if required_run_ids and any(
        count != len(required_run_ids) for count in attempt_count_by_policy.values()
    ):
        comparison_blockers.append("policy_attempt_count_not_equal_required_scenario_count")
    if not score_ranges_valid:
        comparison_blockers.extend(score_range_blockers)
    comparison_blockers = _dedupe(comparison_blockers)
    if comparison_blockers:
        status = "blocked_inconclusive_ranking"
    elif ranking_ambiguous:
        status = "completed_ambiguous_ranking"
    elif uncertainty_penalty_applied or ood_blockers:
        status = "completed_low_confidence_ranking"
    else:
        status = "completed"
    single_best_policy_claimed = bool(
        ranked and not comparison_blockers and not ranking_ambiguous
    )
    evaluator_top_policy_id = ranked[0]["policy_id"] if ranked else None
    confidence_level = "blocked"
    if not comparison_blockers:
        if ranking_ambiguous:
            confidence_level = "ambiguous"
        elif uncertainty_penalty_applied or ood_blockers:
            confidence_level = "low"
        else:
            confidence_level = "medium_evaluator_only"
    review_grade_policy_ranking = bool(
        status == "completed"
        and visual_review_gate["review_grade_success_labels"]
        and visual_review_gate["visual_rollout_useful_for_task_success_review"]
    )
    return {
        "schema_version": POLICY_RANKING_SCORECARD_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "evaluation_substrate": substrate,
        "ranking_basis": "fixture_vision_success_labels_over_model_derived_wam_rollouts",
        "visual_smoke_status": visual_review_gate["visual_smoke_status"],
        "visual_smoke_statuses": visual_review_gate["visual_smoke_statuses"],
        "visual_rollout_useful_for_task_success_review": visual_review_gate[
            "visual_rollout_useful_for_task_success_review"
        ],
        "visual_review_blockers": visual_review_gate["blockers"],
        "fixture_evaluator_only": visual_review_gate["fixture_evaluator_only"],
        "review_grade_visual_evidence_available": visual_review_gate[
            "review_grade_visual_evidence_available"
        ],
        "review_grade_success_labels": visual_review_gate["review_grade_success_labels"],
        "review_grade_policy_ranking": review_grade_policy_ranking,
        "review_grade_policy_ranking_status": "completed"
        if review_grade_policy_ranking
        else "blocked_visual_review_required",
        "comparison_contract": {
            "primary_eval_question": (
                "which policy_or_checkpoint performs better inside this configured evaluator"
            ),
            "comparison_scope": "configured_evaluator_only",
            "same_scenario_eval_matrix_required": True,
            "same_observation_and_label_protocol_required": True,
            "ranking_metrics": [
                "predicted_success_rate",
                "mean_uncertainty",
                "failure_taxonomy",
            ],
            "validation_metrics_when_real_anchors_exist": [
                "spearman_rank_correlation",
                "pearson_success_rate_correlation",
                "mean_maximum_rank_violation",
                "mean_absolute_success_rate_error",
            ],
            "traditional_sim_cross_check_optional": True,
            "evaluation_readiness_claimed": False,
            "external_deployment_grade_claimed": False,
            "single_best_policy_claim_requires_margin_above_tie_band": True,
            "review_grade_policy_ranking_requires_passed_visual_smoke": True,
            "fixture_evaluator_only_ranking_is_not_review_grade": True,
        },
        "policy_count": len(ranked),
        "scenario_attempt_count": len(label_rows),
        "required_scenario_eval_run_ids": list(required_run_ids),
        "per_policy_coverage": per_policy_coverage,
        "coverage_complete": coverage_complete,
        "missing_by_policy": missing_by_policy,
        "extra_by_policy": extra_by_policy,
        "attempt_count_by_policy": attempt_count_by_policy,
        "comparison_blockers": comparison_blockers,
        "score_ranges_valid": score_ranges_valid,
        "score_range_blockers": _dedupe(score_range_blockers),
        "policy_rankings": ranked,
        "evaluator_top_policy_id": evaluator_top_policy_id,
        "top_policy_id": evaluator_top_policy_id if single_best_policy_claimed else None,
        "single_best_policy_claimed": single_best_policy_claimed,
        "ranking_confidence": {
            "top_policy_margin": top_policy_margin,
            "tie_band": POLICY_RANKING_TIE_BAND,
            "ranking_ambiguous": ranking_ambiguous,
            "uncertainty_penalty_applied": uncertainty_penalty_applied,
            "ood_blockers": ood_blockers,
            "confidence_level": confidence_level,
            "real_world_calibration_metrics": {
                "spearman_rank_correlation": "not_measured",
                "pearson_success_rate_correlation": "not_measured",
                "mean_maximum_rank_violation": "not_measured",
            },
        },
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
        "claim_boundary": {
            **_claim_boundary(substrate=substrate, generated_at=generated_at),
            "visual_smoke_required_for_review_grade_policy_ranking": True,
            "visual_rollout_useful_for_task_success_review": visual_review_gate[
                "visual_rollout_useful_for_task_success_review"
            ],
            "fixture_evaluator_only": visual_review_gate["fixture_evaluator_only"],
            "review_grade_policy_ranking": review_grade_policy_ranking,
        },
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
            "mean_maximum_rank_violation",
            "failure_mode_agreement",
        ],
        "accepted_anchor_join_keys": list(ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS),
        "customer_specific_srcc_claimed": False,
        "blocked_report_reason": "missing_paired_real_world_rollout_outcomes",
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def _policy_metadata(policies: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for policy in policies:
        policy_id = _string(policy.get("policy_id") or policy.get("policyId"))
        if not policy_id:
            continue
        metadata[policy_id] = {
            "policy_id": policy_id,
            "display_name": _string(policy.get("display_name") or policy.get("name")) or policy_id,
            "adapter_id": _string(
                policy.get("adapter_id")
                or policy.get("adapterId")
                or policy.get("policy_adapter_id")
                or policy.get("policyAdapterId")
            )
            or None,
            "checkpoint_id": _string(
                policy.get("checkpoint_id")
                or policy.get("checkpointId")
                or policy.get("checkpoint")
            )
            or None,
        }
    return metadata


def _policy_ranking_rows(scorecard: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows = [dict(item) for item in scorecard.get("policy_rankings", []) or [] if isinstance(item, Mapping)]
    return sorted(
        rows,
        key=lambda row: (
            _number(row.get("rank"), 999999),
            -_number(row.get("predicted_success_rate")),
            _string(row.get("policy_id")),
        ),
    )


def _candidate_selection_summary(scorecard: Mapping[str, Any]) -> Dict[str, Any]:
    ranked = _policy_ranking_rows(scorecard)
    if not ranked:
        return {
            "status": "blocked_missing_ranking_evidence",
            "top_policy_id": None,
            "runner_up_policy_id": None,
            "margin": None,
            "ranking_ambiguous": True,
            "tie_or_ambiguity_status": "no_candidate_ranking_available",
            "candidate_shortlist": [],
            "ambiguity_reasons": ["policy_ranking_scorecard_missing_or_empty"],
            "policy_rankings": [],
        }
    if len(ranked) == 1:
        only = ranked[0]
        return {
            "status": "single_candidate_no_comparative_ranking",
            "top_policy_id": None,
            "runner_up_policy_id": None,
            "margin": None,
            "ranking_ambiguous": True,
            "tie_or_ambiguity_status": "single_candidate_no_comparison",
            "candidate_shortlist": [only],
            "ambiguity_reasons": ["only_one_policy_candidate_was_evaluated"],
            "policy_rankings": ranked,
        }

    top = ranked[0]
    runner_up = ranked[1]
    success_margin = round(
        _number(top.get("predicted_success_rate"))
        - _number(runner_up.get("predicted_success_rate")),
        6,
    )
    uncertainty_delta = None
    if top.get("mean_uncertainty") is not None and runner_up.get("mean_uncertainty") is not None:
        uncertainty_delta = round(
            _number(runner_up.get("mean_uncertainty"))
            - _number(top.get("mean_uncertainty")),
            6,
        )
    shortlist = [
        row
        for row in ranked
        if round(
            _number(top.get("predicted_success_rate"))
            - _number(row.get("predicted_success_rate")),
            6,
        )
        < CANDIDATE_SELECTION_AMBIGUITY_SUCCESS_RATE_MARGIN
    ]
    ambiguous = success_margin < CANDIDATE_SELECTION_AMBIGUITY_SUCCESS_RATE_MARGIN
    return {
        "status": "ambiguous_candidate_shortlist" if ambiguous else "clear_winner",
        "top_policy_id": None if ambiguous else top.get("policy_id"),
        "runner_up_policy_id": runner_up.get("policy_id"),
        "margin": {
            "predicted_success_rate": success_margin,
            "mean_uncertainty_advantage": uncertainty_delta,
            "ambiguity_threshold": CANDIDATE_SELECTION_AMBIGUITY_SUCCESS_RATE_MARGIN,
        },
        "ranking_ambiguous": ambiguous,
        "tie_or_ambiguity_status": "ambiguous" if ambiguous else "clear",
        "candidate_shortlist": shortlist if ambiguous else [],
        "ambiguity_reasons": ["top_two_success_rates_within_threshold"] if ambiguous else [],
        "policy_rankings": ranked,
    }


def _scenario_metadata(matrix: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for run in _matrix_runs(matrix):
        run_id = _string(run.get("scenario_eval_run_id"))
        if not run_id:
            continue
        metadata[run_id] = {
            "scenario_eval_run_id": run_id,
            "scenario_variation_instance_id": run.get("scenario_variation_instance_id")
            or run.get("scenarioVariationInstanceId"),
            "task_id": _string(run.get("task_id") or run.get("taskId")) or None,
            "scenario_id": _string(run.get("scenario_id") or run.get("scenarioId")) or None,
            "variation_name": _string(run.get("variation_name") or run.get("variationName"))
            or None,
            "split": _string(run.get("split")) or None,
        }
    return metadata


def _label_rows(labels: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return [dict(item) for item in labels.get("labels", []) or [] if isinstance(item, Mapping)]


def _label_evidence_ref(label: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "artifact_path": "vision_success_labels.json",
        "label_id": label.get("label_id"),
        "attempt_id": label.get("attempt_id"),
        "rollout_id": label.get("rollout_id"),
        "scenario_eval_run_id": label.get("scenario_eval_run_id"),
        "policy_id": label.get("policy_id"),
    }


def _scenario_matrix_coverage(
    *,
    matrix: Mapping[str, Any],
    labels: Mapping[str, Any],
    scorecard: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata = _scenario_metadata(matrix)
    rows = _label_rows(labels)
    matrix_run_ids = sorted(metadata)
    covered_run_ids = sorted(
        {_string(label.get("scenario_eval_run_id")) for label in rows if label.get("scenario_eval_run_id")}
    )
    required_run_ids = matrix_run_ids or covered_run_ids
    missing_run_ids = sorted(set(required_run_ids) - set(covered_run_ids))
    policy_count = int(_number(scorecard.get("policy_count"), 0) or 0)
    expected_attempt_count = len(required_run_ids) * policy_count if required_run_ids and policy_count else None
    observed_attempt_count = len(rows)
    return {
        "scenario_eval_run_count": len(required_run_ids),
        "policy_count": policy_count,
        "expected_candidate_attempt_count": expected_attempt_count,
        "observed_candidate_attempt_count": observed_attempt_count,
        "coverage_complete": bool(
            required_run_ids
            and not missing_run_ids
            and (expected_attempt_count is None or observed_attempt_count >= expected_attempt_count)
        ),
        "required_scenario_eval_run_ids": required_run_ids,
        "covered_scenario_eval_run_ids": covered_run_ids,
        "missing_scenario_eval_run_ids": missing_run_ids,
        "coverage_source": "scenario_eval_matrix" if matrix_run_ids else "vision_success_labels",
    }


def _decisive_scenarios(
    *,
    matrix: Mapping[str, Any],
    labels: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    metadata = _scenario_metadata(matrix)
    by_run: Dict[str, list[Dict[str, Any]]] = {}
    for label in _label_rows(labels):
        run_id = _string(label.get("scenario_eval_run_id"))
        if run_id:
            by_run.setdefault(run_id, []).append(label)
    decisive: list[Dict[str, Any]] = []
    for run_id, rows in sorted(by_run.items()):
        outcomes = []
        successes = []
        failures = []
        for label in sorted(rows, key=lambda item: _string(item.get("policy_id"))):
            policy_id = _string(label.get("policy_id")) or "policy"
            success = bool(label.get("task_success"))
            if success:
                successes.append(policy_id)
            else:
                failures.append(policy_id)
            outcomes.append(
                {
                    "policy_id": policy_id,
                    "task_success": success,
                    "uncertainty_score": _number(label.get("uncertainty_score")),
                    "failure_mode_ids": _string_list(label.get("failure_mode_ids")),
                    "ood_flags": _string_list(label.get("ood_flags")),
                    "evidence_ref": _label_evidence_ref(label),
                }
            )
        if successes and failures:
            decisive.append(
                {
                    **metadata.get(run_id, {"scenario_eval_run_id": run_id}),
                    "successful_policy_ids": successes,
                    "failed_policy_ids": failures,
                    "policy_outcomes": outcomes,
                    "failure_mode_ids": sorted(
                        {mode for outcome in outcomes for mode in outcome["failure_mode_ids"]}
                    ),
                    "exemplar_evidence_refs": [
                        outcome["evidence_ref"] for outcome in outcomes[:4]
                    ],
                }
            )
    return decisive


def _high_uncertainty_scenarios(labels: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows = [
        label
        for label in _label_rows(labels)
        if _number(label.get("uncertainty_score")) >= CANDIDATE_SELECTION_HIGH_UNCERTAINTY_THRESHOLD
    ]
    rows = sorted(
        rows,
        key=lambda label: (
            -_number(label.get("uncertainty_score")),
            _string(label.get("scenario_eval_run_id")),
            _string(label.get("policy_id")),
        ),
    )
    return [
        {
            "scenario_eval_run_id": label.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": label.get("scenario_variation_instance_id"),
            "policy_id": label.get("policy_id"),
            "uncertainty_score": _number(label.get("uncertainty_score")),
            "ood_flags": _string_list(label.get("ood_flags")),
            "review_status": "needs_review",
            "evidence_ref": _label_evidence_ref(label),
        }
        for label in rows
    ]


def _ood_blockers(labels: Mapping[str, Any]) -> list[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for label in _label_rows(labels):
        for flag in _string_list(label.get("ood_flags")):
            row = grouped.setdefault(
                flag,
                {
                    "ood_flag": flag,
                    "count": 0,
                    "scenario_eval_run_ids": set(),
                    "affected_policy_ids": set(),
                    "exemplar_evidence_refs": [],
                },
            )
            row["count"] += 1
            if label.get("scenario_eval_run_id"):
                row["scenario_eval_run_ids"].add(_string(label.get("scenario_eval_run_id")))
            if label.get("policy_id"):
                row["affected_policy_ids"].add(_string(label.get("policy_id")))
            if len(row["exemplar_evidence_refs"]) < 3:
                row["exemplar_evidence_refs"].append(_label_evidence_ref(label))
    return [
        {
            **{key: value for key, value in row.items() if key not in {"scenario_eval_run_ids", "affected_policy_ids"}},
            "scenario_eval_run_ids": sorted(row["scenario_eval_run_ids"]),
            "affected_policy_ids": sorted(row["affected_policy_ids"]),
        }
        for row in sorted(grouped.values(), key=lambda item: (-item["count"], item["ood_flag"]))
    ]


def _failure_hook_template(failure_mode_id: str) -> Dict[str, list[str]]:
    templates = {
        "blocked_path_or_clearance_failure": {
            "data_to_collect": [
                "robot POV clips through blocked and narrow-clearance approaches",
                "depth, pose, near-miss, and contact annotations at obstacle boundaries",
            ],
            "scenario_variants_to_add": [
                "narrow aisle clearance sweeps",
                "partially blocked path variants",
                "movable obstacle offsets near the target approach",
            ],
        },
        "dynamic_agent_safety_failure": {
            "data_to_collect": [
                "robot POV and third-person clips with humans or carts crossing the route",
                "time-aligned agent trajectories and yield-distance labels",
            ],
            "scenario_variants_to_add": [
                "human crossing timing offsets",
                "forklift or cart crossing speed variants",
                "late-yield and stop-go interaction cases",
            ],
        },
        "perception_ambiguity_failure": {
            "data_to_collect": [
                "multi-angle robot POV clips for visually similar targets",
                "object identity, occlusion, glare, and missing-label annotations",
            ],
            "scenario_variants_to_add": [
                "glare and low-light target views",
                "partial occlusion variants",
                "wrong-object distractor placements",
            ],
        },
        "manipulation_alignment_failure": {
            "data_to_collect": [
                "hand-camera clips of grasp, place, and object-rotation attempts",
                "object pose, gripper pose, slip, and final-placement labels",
            ],
            "scenario_variants_to_add": [
                "object rotation variants",
                "shifted cart or bin target poses",
                "grasp approach angle sweeps",
            ],
        },
        "wam_ood_uncertain": {
            "data_to_collect": [
                "paired real rollout anchors for high-uncertainty generated scenarios",
                "operator review labels explaining whether the generated observation is usable",
            ],
            "scenario_variants_to_add": [
                "near-distribution versions of the OOD scenario",
                "single-factor OOD ablations for glare, occlusion, or target ambiguity",
            ],
        },
        "fixture_policy_failure": {
            "data_to_collect": [
                "policy command traces and robot POV clips around the forced failure case",
                "review notes confirming whether the fixture failure matches a real failure",
            ],
            "scenario_variants_to_add": [
                "direct regression case for the failing scenario_eval_run_id",
                "one-factor neighboring variants around the failing setup",
            ],
        },
        "unknown_needs_review": {
            "data_to_collect": [
                "human-reviewed rollout clips with failure reason annotations",
                "policy action traces, observations, and task-state snapshots near failure",
            ],
            "scenario_variants_to_add": [
                "minimal reproduction variants once the reviewer labels the failure",
                "neighboring scenario variants that isolate the suspected cause",
            ],
        },
    }
    return templates.get(failure_mode_id, templates["unknown_needs_review"])


def _retry_policy_refs(
    policy_ids: Sequence[str],
    policy_metadata: Mapping[str, Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    refs: list[Dict[str, Any]] = []
    for policy_id in _string_list(policy_ids):
        metadata = _mapping(policy_metadata.get(policy_id))
        refs.append(
            {
                "policy_id": policy_id,
                "display_name": metadata.get("display_name") or policy_id,
                "adapter_id": metadata.get("adapter_id"),
                "checkpoint_id": metadata.get("checkpoint_id"),
                "retry_reason": "rerun_after_failure_cluster_data_package_update",
            }
        )
    return refs


def _failure_clusters(
    *,
    failure_labels: Mapping[str, Any],
    selection: Mapping[str, Any],
    policy_metadata: Mapping[str, Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for label in [
        dict(item)
        for item in failure_labels.get("labels", []) or []
        if isinstance(item, Mapping)
    ]:
        modes = _string_list(label.get("failure_mode_ids")) or ["unknown_needs_review"]
        for mode in modes:
            row = grouped.setdefault(
                mode,
                {
                    "failure_mode_id": mode,
                    "count": 0,
                    "affected_policy_ids": set(),
                    "scenario_eval_run_ids": set(),
                    "exemplar_evidence_refs": [],
                },
            )
            row["count"] += 1
            if label.get("policy_id"):
                row["affected_policy_ids"].add(_string(label.get("policy_id")))
            if label.get("scenario_eval_run_id"):
                row["scenario_eval_run_ids"].add(_string(label.get("scenario_eval_run_id")))
            if len(row["exemplar_evidence_refs"]) < 3:
                row["exemplar_evidence_refs"].append(
                    {
                        "artifact_path": "failure_labels.json",
                        "label_id": label.get("label_id"),
                        "attempt_id": label.get("attempt_id"),
                        "rollout_id": label.get("rollout_id"),
                        "scenario_eval_run_id": label.get("scenario_eval_run_id"),
                        "policy_id": label.get("policy_id"),
                    }
                )
    fallback_policy_ids = [
        _string(row.get("policy_id"))
        for row in selection.get("candidate_shortlist", []) or []
        if isinstance(row, Mapping) and _string(row.get("policy_id"))
    ]
    if not fallback_policy_ids and selection.get("top_policy_id"):
        fallback_policy_ids = [_string(selection.get("top_policy_id"))]
    clusters: list[Dict[str, Any]] = []
    for mode, row in sorted(grouped.items(), key=lambda item: (-item[1]["count"], item[0])):
        affected_policy_ids = sorted(row["affected_policy_ids"]) or fallback_policy_ids
        template = _failure_hook_template(mode)
        weak = mode == "unknown_needs_review"
        clusters.append(
            {
                "cluster_id": f"failure_cluster_{_safe_id(mode)}",
                "failure_mode_id": mode if not weak else None,
                "diagnosis": "unknown_needs_review"
                if weak
                else "failure_mode_observed_root_cause_needs_review",
                "evidence_strength": "weak" if weak else "label_only_needs_review",
                "count": row["count"],
                "affected_policy_ids": affected_policy_ids,
                "scenario_eval_run_ids": sorted(row["scenario_eval_run_ids"]),
                "exemplar_evidence_refs": row["exemplar_evidence_refs"],
                "post_training_data_package_hooks": {
                    **template,
                    "policy_adapter_or_checkpoint_to_retry": _retry_policy_refs(
                        affected_policy_ids,
                        policy_metadata,
                    ),
                },
            }
        )
    return clusters


def _dominant_failure_modes_from_clusters(
    clusters: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    if not clusters:
        return []
    return [
        {
            "failure_mode_id": cluster.get("failure_mode_id") or "unknown_needs_review",
            "count": int(_number(cluster.get("count"), 0) or 0),
            "diagnosis": cluster.get("diagnosis") or "unknown_needs_review",
            "evidence_strength": cluster.get("evidence_strength") or "weak",
        }
        for cluster in clusters
    ]


def _candidate_selection_report(
    *,
    job_id: str,
    substrate: str,
    matrix: Mapping[str, Any],
    policies: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Any],
    failure_labels: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    followup: Mapping[str, Any],
    anchor_manifest: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    selection = _candidate_selection_summary(scorecard)
    policy_meta = _policy_metadata(policies)
    decisive = _decisive_scenarios(matrix=matrix, labels=labels)
    clusters = _failure_clusters(
        failure_labels=failure_labels,
        selection=selection,
        policy_metadata=policy_meta,
    )
    coverage = _scenario_matrix_coverage(matrix=matrix, labels=labels, scorecard=scorecard)
    high_uncertainty = _high_uncertainty_scenarios(labels)
    ood_blockers = _ood_blockers(labels)
    exemplar_refs: list[Dict[str, Any]] = []
    for scenario in decisive[:4]:
        exemplar_refs.extend(
            ref
            for ref in scenario.get("exemplar_evidence_refs", []) or []
            if isinstance(ref, Mapping)
        )
    for cluster in clusters[:4]:
        exemplar_refs.extend(
            ref
            for ref in cluster.get("exemplar_evidence_refs", []) or []
            if isinstance(ref, Mapping)
        )
    usable_anchor_count = int(_number(anchor_manifest.get("usable_anchor_count"), 0) or 0)
    claim_boundary = {
        **_claim_boundary(substrate=substrate, generated_at=generated_at),
        "boundary_statement": "do not use for generated-world rank-fidelity result",
        "do_not_use_as_rank_fidelity_result": True,
        "rank_fidelity_result_claimed": False,
        "accepted_anchor_success_claimed": False,
        "best_policy_statement_scope": "configured_evaluator_only",
    }
    return {
        "schema_version": CANDIDATE_SELECTION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": selection["status"],
        "evaluation_substrate": substrate,
        "primary_eval_question": "which policy performed best in this evaluator, and what broke",
        "selection": selection,
        "top_policy_id": selection.get("top_policy_id"),
        "runner_up_policy_id": selection.get("runner_up_policy_id"),
        "margin": selection.get("margin"),
        "tie_or_ambiguity_status": selection.get("tie_or_ambiguity_status"),
        "candidate_shortlist": selection.get("candidate_shortlist"),
        "scenario_matrix_coverage": coverage,
        "decisive_scenarios": decisive,
        "high_uncertainty_scenarios": high_uncertainty,
        "ood_blockers": ood_blockers,
        "dominant_failure_modes": _dominant_failure_modes_from_clusters(clusters),
        "failure_clusters": clusters,
        "failure_evidence_status": "unknown_needs_review"
        if clusters and all(cluster.get("evidence_strength") == "weak" for cluster in clusters)
        else ("no_failures_observed_in_evaluator" if not clusters else "label_only_needs_review"),
        "exemplar_evidence_refs": exemplar_refs[:10],
        "real_world_validation_followup_request": {
            "artifact_path": "real_world_validation_followup_request.json",
            "status": followup.get("status") or "requested_real_world_validation_anchors",
            "triggered_by_no_real_world_anchors": usable_anchor_count == 0,
            "usable_anchor_count": usable_anchor_count,
            "requested_anchor_rollouts": _string_list(followup.get("requested_anchor_rollouts")),
            "minimum_validation_requirements": followup.get("minimum_validation_requirements")
            or {},
        },
        "real_world_validation_requests": [
            {
                "request_type": "paired_real_world_rollout_anchors",
                "status": followup.get("status") or "requested_real_world_validation_anchors",
                "artifact_path": "real_world_validation_followup_request.json",
                "needed_before_external_claims": True,
            }
        ],
        "artifact_paths": {
            "policy_ranking_scorecard": "policy_ranking_scorecard.json",
            "vision_success_labels": "vision_success_labels.json",
            "failure_labels": "failure_labels.json",
            "wam_rollout_results": "wam_rollout_results.json",
            "real_world_validation_followup_request": (
                "real_world_validation_followup_request.json"
            ),
            "wam_real_world_validation_anchor_manifest": (
                "wam_real_world_validation_anchor_manifest.json"
            ),
        },
        "claim_boundary": claim_boundary,
    }


def _candidate_selection_markdown(report: Mapping[str, Any]) -> str:
    selection = _mapping(report.get("selection"))
    margin = _mapping(report.get("margin"))
    top_policy = report.get("top_policy_id") or "ambiguous, use shortlist"
    shortlist = [
        _string(row.get("policy_id"))
        for row in report.get("candidate_shortlist", []) or []
        if isinstance(row, Mapping) and _string(row.get("policy_id"))
    ]
    lines = [
        "# WAM Candidate Selection Report",
        "",
        f"Status: `{report.get('status')}`",
        f"Evaluation substrate: `{report.get('evaluation_substrate')}`",
        f"Top policy: `{top_policy}`",
        f"Runner-up: `{report.get('runner_up_policy_id')}`",
        f"Predicted success-rate margin: `{margin.get('predicted_success_rate')}`",
        f"Tie or ambiguity status: `{report.get('tie_or_ambiguity_status')}`",
        "",
        "Boundary: do not use for generated-world rank-fidelity result.",
        "",
    ]
    if shortlist:
        lines.extend(
            [
                "## Candidate Shortlist",
                "",
                *[f"- `{policy_id}`" for policy_id in shortlist],
                "",
            ]
        )
    decisive = report.get("decisive_scenarios", []) or []
    lines.extend(["## Decisive Scenarios", ""])
    if decisive:
        for scenario in decisive[:8]:
            if not isinstance(scenario, Mapping):
                continue
            lines.append(
                f"- `{scenario.get('scenario_eval_run_id')}`: "
                f"passed={scenario.get('successful_policy_ids')} "
                f"failed={scenario.get('failed_policy_ids')}"
            )
    else:
        lines.append("- None found in this evaluator run.")
    lines.append("")
    clusters = report.get("failure_clusters", []) or []
    lines.extend(["## Failure Clusters", ""])
    if clusters:
        for cluster in clusters[:8]:
            if not isinstance(cluster, Mapping):
                continue
            lines.append(
                f"- `{cluster.get('cluster_id')}`: {cluster.get('diagnosis')} "
                f"({cluster.get('count')} labels)"
            )
    else:
        lines.append("- No failed evaluator labels were produced.")
    lines.extend(
        [
            "",
            "## Real-World Validation",
            "",
            (
                "- Follow-up request: "
                f"`{_mapping(report.get('real_world_validation_followup_request')).get('status')}`"
            ),
            "",
        ]
    )
    if selection.get("ambiguity_reasons"):
        lines.extend(
            [
                "## Ambiguity Reasons",
                "",
                *[f"- `{reason}`" for reason in _string_list(selection.get("ambiguity_reasons"))],
                "",
            ]
        )
    return "\n".join(lines)


def _customer_handoff_markdown(report: Mapping[str, Any]) -> str:
    visual_gate = _mapping(report.get("visual_reviewability_gate"))
    blockers = _string_list(visual_gate.get("blockers"))
    return "\n".join(
        [
            "# WAM Policy Evaluation Handoff",
            "",
            f"Status: `{report.get('status')}`",
            f"Evaluation substrate: `{report.get('evaluation_substrate')}`",
            f"Top policy: `{report.get('top_policy_id')}`",
            f"Visual review gate: `{visual_gate.get('status')}`",
            f"Visual review blockers: `{', '.join(blockers) if blockers else 'none'}`",
            "",
            (
                "This ranks policies inside the configured evaluator. Generated rollouts "
                "and fixture labels are support artifacts; they do not prove real-world "
                "success, generated-world rank-fidelity result, robot safety, or customer-specific SRCC."
            ),
            "",
        ]
    )


def _customer_handoff_report(
    *,
    job_id: str,
    substrate: str,
    scorecard: Mapping[str, Any],
    candidate_selection_report: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    visual_gate = {
        "status": scorecard.get("review_grade_policy_ranking_status")
        or "blocked_visual_review_required",
        "visual_smoke_status": scorecard.get("visual_smoke_status")
        or FIXTURE_VISUAL_SMOKE_STATUS,
        "visual_rollout_useful_for_task_success_review": bool(
            scorecard.get("visual_rollout_useful_for_task_success_review")
        ),
        "review_grade_policy_ranking": bool(scorecard.get("review_grade_policy_ranking")),
        "fixture_evaluator_only": bool(scorecard.get("fixture_evaluator_only")),
        "blockers": _string_list(scorecard.get("visual_review_blockers")),
    }
    return {
        "schema_version": "wam_customer_handoff_report.v1",
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "generated",
        "evaluation_substrate": substrate,
        "top_policy_id": candidate_selection_report.get("top_policy_id"),
        "candidate_selection_report_path": "candidate_selection_report.json",
        "candidate_selection_summary": {
            "status": candidate_selection_report.get("status"),
            "top_policy_id": candidate_selection_report.get("top_policy_id"),
            "runner_up_policy_id": candidate_selection_report.get("runner_up_policy_id"),
            "margin": candidate_selection_report.get("margin"),
            "tie_or_ambiguity_status": candidate_selection_report.get(
                "tie_or_ambiguity_status"
            ),
            "candidate_shortlist": candidate_selection_report.get("candidate_shortlist"),
        },
        "legacy_scorecard_top_policy_id": scorecard.get("top_policy_id"),
        "visual_reviewability_gate": visual_gate,
        "artifact_paths": {
            key: value
            for key, value in WAM_ARTIFACT_PATHS.items()
            if key not in {"customer_handoff_report_markdown", "candidate_selection_report_markdown"}
        },
        "reader_boundary": (
            "Generated WAM rollouts are model-derived support artifacts, not raw truth or "
            "generated-world rank-fidelity result. Fixture-only labels and rankings are not review-grade "
            "task-success evidence unless an explicit visual smoke artifact says the "
            "rollout is useful for task-success review."
        ),
        "claim_boundary": {
            **_claim_boundary(substrate=substrate, generated_at=generated_at),
            "visual_smoke_required_for_review_grade_policy_ranking": True,
            "fixture_evaluator_only": bool(scorecard.get("fixture_evaluator_only")),
            "review_grade_policy_ranking": bool(scorecard.get("review_grade_policy_ranking")),
        },
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
    scorecard = _policy_scorecard(
        substrate=substrate,
        labels=labels,
        generated_at=generated_at,
        required_scenario_eval_run_ids=[
            _string(run.get("scenario_eval_run_id")) for run in _matrix_runs(matrix)
        ],
        policy_ids=[_string(policy.get("policy_id")) for policy in policies],
    )
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
    candidate_report = _candidate_selection_report(
        job_id=job_id,
        substrate=substrate,
        matrix=matrix,
        policies=policies,
        labels=labels,
        failure_labels=failure_labels,
        scorecard=scorecard,
        followup=followup,
        anchor_manifest=anchor_manifest,
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
    handoff = _customer_handoff_report(
        job_id=job_id,
        substrate=substrate,
        scorecard=scorecard,
        candidate_selection_report=candidate_report,
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
        "candidate_selection_report": candidate_report,
        "customer_handoff_report": handoff,
    }
    _write_wam_artifacts(job_dir, payloads)
    write_text(
        job_dir / WAM_ARTIFACT_PATHS["candidate_selection_report_markdown"],
        _candidate_selection_markdown(candidate_report),
    )
    write_text(
        job_dir / WAM_ARTIFACT_PATHS["customer_handoff_report_markdown"],
        _customer_handoff_markdown(handoff),
    )
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
        failure_labels=failure_labels,
        generated_at=generated,
    )
    scorecard = _policy_scorecard(
        substrate=substrate,
        labels=labels,
        generated_at=generated,
        required_scenario_eval_run_ids=[
            _string(run.get("scenario_eval_run_id")) for run in runs
        ],
        policy_ids=[_string(policy.get("policy_id")) for policy in policies],
    )
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
    candidate_report = _candidate_selection_report(
        job_id=job_id,
        substrate=substrate,
        matrix=matrix,
        policies=policies,
        labels=labels,
        failure_labels=failure_labels,
        scorecard=scorecard,
        followup=followup,
        anchor_manifest=anchor_manifest,
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
        candidate_selection_report=candidate_report,
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
        "candidate_selection_report": candidate_report,
        "customer_handoff_report": handoff,
    }
    _write_wam_artifacts(resolved_job_dir, payloads)
    write_text(
        resolved_job_dir / WAM_ARTIFACT_PATHS["candidate_selection_report_markdown"],
        _candidate_selection_markdown(candidate_report),
    )
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
