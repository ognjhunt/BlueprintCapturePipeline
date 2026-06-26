"""Evaluation substrate registry for WAM and simulator-backed robot eval jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .common import utc_now_iso, write_json


EVALUATION_SUBSTRATE_REGISTRY_SCHEMA_VERSION = "evaluation_substrate_registry.v1"
EVALUATION_SUBSTRATE_REQUEST_SCHEMA_VERSION = "wam_evaluation_request.v1"
WAM_EVAL_CLAIM_BOUNDARY_SCHEMA_VERSION = "wam_eval_claim_boundary.v1"

SUPPORTED_EVALUATION_SUBSTRATES = (
    "fixture_wam",
    "cosmos3_wam",
    "oscar_wam",
    "classical_sim_mujoco",
    "classical_sim_isaac",
    "recorded_trace",
)

WAM_EVALUATION_SUBSTRATES = {"fixture_wam", "cosmos3_wam", "oscar_wam"}
CLASSICAL_SIM_EVALUATION_SUBSTRATES = {"classical_sim_mujoco", "classical_sim_isaac"}

SUBSTRATE_ALIASES = {
    "fixture": "fixture_wam",
    "fixture_local": "fixture_wam",
    "wam_fixture": "fixture_wam",
    "local_wam": "fixture_wam",
    "cosmos": "cosmos3_wam",
    "cosmos3": "cosmos3_wam",
    "cosmos_3": "cosmos3_wam",
    "oscar": "oscar_wam",
    "mujoco": "classical_sim_mujoco",
    "classical_mujoco": "classical_sim_mujoco",
    "sim_mujoco": "classical_sim_mujoco",
    "isaac": "classical_sim_isaac",
    "isaac_sim": "classical_sim_isaac",
    "isaac_lab_arena": "classical_sim_isaac",
    "classical_isaac": "classical_sim_isaac",
    "trace": "recorded_trace",
    "recorded_action_trace": "recorded_trace",
}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_string(item) for item in value if _string(item)]
    return []


def _substrate_key(value: Any) -> str:
    return _string(value).lower().replace("-", "_").replace(" ", "_")


def normalize_evaluation_substrate(
    value: Any,
    *,
    simulator_engine: Any | None = None,
    default: str = "",
) -> str:
    """Normalize a requested substrate while preserving legacy simulator aliases."""

    raw = _substrate_key(value) or _substrate_key(simulator_engine)
    if not raw:
        return default
    normalized = SUBSTRATE_ALIASES.get(raw, raw)
    if normalized not in SUPPORTED_EVALUATION_SUBSTRATES:
        raise ValueError(f"Unsupported evaluation substrate: {value or simulator_engine}")
    return normalized


def is_wam_evaluation_substrate(value: Any) -> bool:
    substrate = normalize_evaluation_substrate(value, default="")
    return substrate in WAM_EVALUATION_SUBSTRATES


def is_classical_sim_evaluation_substrate(value: Any) -> bool:
    substrate = normalize_evaluation_substrate(value, default="")
    return substrate in CLASSICAL_SIM_EVALUATION_SUBSTRATES


def requested_evaluation_substrate(
    request: Mapping[str, Any],
    *,
    explicit: Any | None = None,
) -> str:
    """Return a substrate only when the request or caller explicitly asks for one."""

    if _string(explicit):
        return normalize_evaluation_substrate(explicit)
    execution_request = _mapping(request.get("execution_request") or request.get("executionRequest"))
    wam_request = _mapping(
        request.get("wam_evaluation")
        or request.get("wamEvaluation")
        or execution_request.get("wam_evaluation")
        or execution_request.get("wamEvaluation")
    )
    substrate = (
        _field(request, "evaluation_substrate", "evaluationSubstrate")
        or _field(execution_request, "evaluation_substrate", "evaluationSubstrate")
        or _field(wam_request, "evaluation_substrate", "evaluationSubstrate", "substrate")
    )
    return normalize_evaluation_substrate(substrate, default="") if _string(substrate) else ""


def legacy_simulator_substrate(simulator: Any) -> str:
    return normalize_evaluation_substrate("", simulator_engine=simulator, default="")


def _field(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def build_evaluation_substrate_registry(*, generated_at: str | None = None) -> Dict[str, Any]:
    generated = generated_at or utc_now_iso()
    entries = {
        "fixture_wam": {
            "substrate": "fixture_wam",
            "family": "world_action_model",
            "provider_kind": "repo_local_fixture",
            "local_available": True,
            "live_provider_required": False,
            "command_surface": "blueprint-run-wam-fixture-evaluator",
            "rollout_artifact_family": "wam_rollout_manifest",
            "success_judge": "fixture_vision_success_judge",
            "proof_ceiling": "model_derived_support_artifact_only",
        },
        "cosmos3_wam": {
            "substrate": "cosmos3_wam",
            "family": "world_action_model",
            "provider_kind": "replaceable_live_or_owner_adapter",
            "local_available": False,
            "live_provider_required": True,
            "command_surface": "blocked_until_provider_adapter_configured",
            "rollout_artifact_family": "wam_rollout_manifest",
            "success_judge": "vision_success_labeler_adapter",
            "proof_ceiling": "model_derived_support_artifact_until_real_validation",
        },
        "oscar_wam": {
            "substrate": "oscar_wam",
            "family": "world_action_model",
            "provider_kind": "replaceable_live_or_owner_adapter",
            "local_available": False,
            "live_provider_required": True,
            "command_surface": "blocked_until_provider_adapter_configured",
            "rollout_artifact_family": "wam_rollout_manifest",
            "success_judge": "vision_success_labeler_adapter",
            "proof_ceiling": "model_derived_support_artifact_until_real_validation",
        },
        "classical_sim_mujoco": {
            "substrate": "classical_sim_mujoco",
            "family": "classical_simulation",
            "provider_kind": "mujoco_or_owner_command",
            "local_available": False,
            "live_provider_required": False,
            "command_surface": "blueprint-run-policy-autoresearch-mujoco-evaluator",
            "rollout_artifact_family": "normalized_attempt_trace",
            "success_judge": "state_metrics_or_review_labels",
            "proof_ceiling": "simulator_proof_only_when_owner_command_evidence_exists",
        },
        "classical_sim_isaac": {
            "substrate": "classical_sim_isaac",
            "family": "classical_simulation",
            "provider_kind": "isaac_or_owner_command",
            "local_available": False,
            "live_provider_required": False,
            "command_surface": "owner_gpu_or_provider_command",
            "rollout_artifact_family": "normalized_attempt_trace",
            "success_judge": "state_metrics_or_review_labels",
            "proof_ceiling": "simulator_proof_only_when_owner_command_evidence_exists",
        },
        "recorded_trace": {
            "substrate": "recorded_trace",
            "family": "recorded_or_replay_trace",
            "provider_kind": "customer_or_owner_trace",
            "local_available": True,
            "live_provider_required": False,
            "command_surface": "trace_ingest",
            "rollout_artifact_family": "normalized_attempt_trace",
            "success_judge": "review_labels_or_outcome_records",
            "proof_ceiling": "trace_coverage_until_owner_policy_or_real_outcome_proof",
        },
    }
    return {
        "schema_version": EVALUATION_SUBSTRATE_REGISTRY_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "available",
        "default_primary_substrate": "fixture_wam",
        "supported_substrates": list(SUPPORTED_EVALUATION_SUBSTRATES),
        "compatibility_aliases": dict(SUBSTRATE_ALIASES),
        "entries": entries,
        "contract": {
            "primary_eval_question": (
                "which policy_or_checkpoint performs better inside the configured evaluator"
            ),
            "policy_comparison_is_evaluator_bounded": True,
            "policy_comparison_requires_same_scenario_matrix": True,
            "traditional_sim_is_cross_check_not_required_authority": True,
            "mmrv_pearson_spearman_are_validation_metrics_when_anchors_exist": True,
            "substrates_are_replaceable": True,
            "simulator_engine_aliases_preserved": True,
            "generated_rollouts_are_model_derived_support_artifacts": True,
            "customer_specific_srcc_requires_real_world_validation_rollouts": True,
            "passing_wam_eval_is_not_deployment_or_safety_approval": True,
        },
    }


def write_evaluation_substrate_registry(job_dir: Path, *, generated_at: str | None = None) -> Dict[str, Any]:
    registry = build_evaluation_substrate_registry(generated_at=generated_at)
    write_json(job_dir / "evaluation_substrate_registry.json", registry)
    return registry


def build_wam_evaluation_request(
    *,
    job_id: str,
    substrate: str,
    scenario_eval_matrix_path: str = "scenario_eval_matrix.json",
    policy_package_manifest_path: str = "policy_package_manifest.json",
    policy_ids: Sequence[str] = (),
    generated_at: str | None = None,
    status: str = "planned",
    blockers: Sequence[str] = (),
) -> Dict[str, Any]:
    generated = generated_at or utc_now_iso()
    return {
        "schema_version": EVALUATION_SUBSTRATE_REQUEST_SCHEMA_VERSION,
        "generated_at": generated,
        "job_id": job_id,
        "status": status,
        "evaluation_substrate": substrate,
        "substrate_family": "world_action_model"
        if substrate in WAM_EVALUATION_SUBSTRATES
        else "non_wam_compatibility",
        "scenario_eval_matrix_path": scenario_eval_matrix_path,
        "policy_package_manifest_path": policy_package_manifest_path,
        "policy_ids": _string_list(policy_ids),
        "blockers": list(blockers),
        "outputs_expected": {
            "wam_rollout_manifest": "wam_rollout_manifest.json",
            "wam_rollout_results": "wam_rollout_results.json",
            "vision_success_labels": "vision_success_labels.json",
            "normalized_attempt_trace": "normalized_attempt_trace.json",
            "failure_labels": "failure_labels.json",
            "policy_ranking_scorecard": "policy_ranking_scorecard.json",
            "wam_eval_claim_boundary": "wam_eval_claim_boundary.json",
            "real_world_validation_followup_request": (
                "real_world_validation_followup_request.json"
            ),
            "candidate_selection_report": "candidate_selection_report.json",
            "candidate_selection_report_markdown": "candidate_selection_report.md",
        },
        "claim_boundary": {
            "request_is_not_provider_execution": True,
            "primary_proof_target": "policy_comparison_within_configured_evaluator",
            "policy_ranking_is_evaluator_bounded": True,
            "policy_ranking_compares_policies_on_same_scenario_matrix": True,
            "policy_ranking_requires_symmetric_policy_scenario_coverage": True,
            "single_best_policy_claim_requires_margin_above_tie_band": True,
            "traditional_sim_is_optional_cross_check_for_wam_eval": True,
            "generated_rollouts_are_model_derived_support_artifacts": True,
            "generated_rollout_success_labels_require_visual_smoke_status": True,
            "visual_smoke_required_for_review_grade_policy_ranking": True,
            "raw_capture_evidence_upstream_only": True,
            "customer_specific_srcc_claimed": False,
            "spearman_pearson_mmrv_status": "not_measured_until_real_anchors_exist",
            "passing_wam_eval_is_not_rank_fidelity_result": True,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def build_wam_eval_claim_boundary(*, substrate: str, generated_at: str) -> Dict[str, Any]:
    return {
        "schema_version": WAM_EVAL_CLAIM_BOUNDARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "evaluation_substrate": substrate,
        "artifact_purpose": "wam_policy_evaluation_support",
        "primary_proof_target": "policy_comparison_within_configured_evaluator",
        "policy_ranking_is_evaluator_bounded": True,
        "policy_ranking_compares_policies_on_same_scenario_matrix": True,
        "policy_ranking_requires_symmetric_policy_scenario_coverage": True,
        "single_best_policy_claim_requires_margin_above_tie_band": True,
        "policy_ranking_is_not_evaluation_readiness": True,
        "traditional_sim_is_optional_cross_check_for_wam_eval": True,
        "mmrv_pearson_spearman_require_validation_anchors": True,
        "spearman_pearson_mmrv_status": "not_measured_until_real_anchors_exist",
        "forward_inverse_consistency_is_reliability_signal_not_success_label": True,
        "forward_inverse_consistency_requires_reviewable_generated_rollout": True,
        "generated_rollouts_are_model_derived_support_artifacts": True,
        "generated_rollouts_are_raw_capture_evidence": False,
        "generated_rollout_success_labels_require_visual_smoke_status": True,
        "visual_smoke_required_for_review_grade_policy_ranking": True,
        "visual_smoke_required_for_review_grade_failure_diagnosis": True,
        "fixture_wam_is_deterministic_local_test_substrate": substrate == "fixture_wam",
        "fixture_evaluator_only": substrate == "fixture_wam",
        "live_provider_calls_performed": False,
        "customer_specific_srcc_claimed": False,
        "customer_specific_srcc_requires_real_world_validation_rollouts": True,
        "passing_wam_heldout_eval_is_not_rank_fidelity_result": True,
        "simulator_execution_proven": False,
        "robot_policy_execution_proven": False,
        "real_world_outcome_proven": False,
        "rank_fidelity_result_proven": False,
        "non_ranking_operational_claim_validated": False,
        "public_claim_upgrade_allowed": False,
    }
