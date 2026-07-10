"""Buyer-facing webapp projection for one robot-eval job."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, TypedDict

from .buyer_claim_ceiling import build_buyer_claim_ceiling
from .failure_diagnosis_contract import build_failure_diagnosis_audit
from .robot_eval_closure_common import (
    POLICY_MODALITY_ORDER,
    _artifact_paths,
    _boolish,
    _capture_root_from_job_dir,
    _explicitly_blocked_scenario_eval_run_records,
    _mapping,
    _read_optional_mapping,
    _scenario_eval_matrix_runs,
    _string,
    _string_list,
    _valid_explicitly_blocked_scenario_eval_run_ids,
)
from .robot_eval_execution import POLICY_ACTION_SCHEMA_ID, POLICY_OBSERVATION_SCHEMA_ID


WEBAPP_ROBOT_EVAL_STATUS_PROJECTION_SCHEMA_VERSION = (
    "webapp_robot_eval_status_projection.v1"
)


class WebappRobotEvalStatusProjection(TypedDict, total=False):
    schema_version: str
    generated_at: str
    job_id: str
    status: str
    blockers: List[str]
    claim_boundary: Dict[str, Any]


def _webapp_robot_eval_status_projection(
    *,
    job_dir: Path,
    job_id: str,
    scene_id: str,
    capture_id: str,
    status: str,
    blockers: Sequence[str],
    request: Mapping[str, Any],
    scenario_eval_matrix: Mapping[str, Any],
    simulator_result: Mapping[str, Any],
    copied_artifacts: Mapping[str, Mapping[str, Any]],
    robot_pov_manifest: Mapping[str, Any],
    policy_manifest: Mapping[str, Any],
    policy_execution_manifest: Mapping[str, Any],
    evaluation_result: Mapping[str, Any],
    proof_boundary: Mapping[str, Any],
    live_closure: Mapping[str, Any],
    data_package_export: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    artifact_paths = _artifact_paths(job_dir)
    task_eval_run_report = _read_optional_mapping(job_dir / "task_eval_run_report.json")
    data_package_consent_evidence = _mapping(data_package_export.get("consent_evidence"))
    revocation_takedown = _mapping(data_package_export.get("revocation_takedown"))
    data_package_blockers = _string_list(data_package_export.get("blockers"))
    consent_revoked = bool(
        _boolish(data_package_consent_evidence.get("consent_revoked"))
        or _boolish(revocation_takedown.get("consent_revoked"))
    )
    revocation_required = bool(
        consent_revoked
        or revocation_takedown.get("status") == "takedown_required"
        or data_package_export.get("status")
        == "blocked_consent_revoked_takedown_required"
        or "consent:consent_revoked_takedown_required" in data_package_blockers
    )
    # TOCTOU guard: re-read consent LIVE at this buyer-facing emit point so a
    # revocation that landed after the upstream export manifest was written still
    # blocks the projection. A live read can only ADD a revocation, never clear
    # an inherited one.
    _live_capture_root = _capture_root_from_job_dir(job_dir)
    if _live_capture_root is not None:
        from .consent_takedown import read_consent_state

        if read_consent_state(_live_capture_root).get("state") == "revoked":
            consent_revoked = True
            revocation_required = True
    webapp_takedown_executed = revocation_takedown.get("webapp_takedown_executed") is True
    hosted_session_takedown_executed = (
        revocation_takedown.get("hosted_session_takedown_executed") is True
    )
    downstream_takedown_artifacts = _mapping(
        revocation_takedown.get("downstream_takedown_artifacts")
        or data_package_export.get("downstream_takedown_artifacts")
    )
    trace = _mapping(copied_artifacts.get("normalized_attempt_trace")) or _read_optional_mapping(
        job_dir / "normalized_attempt_trace.json"
    )
    labels = _mapping(copied_artifacts.get("failure_labels")) or _read_optional_mapping(
        job_dir / "failure_labels.json"
    )
    policy_trace = _read_optional_mapping(job_dir / "policy_execution_trace.json")
    failure_diagnosis_audit = build_failure_diagnosis_audit(
        labels_payload=labels,
        trace_payload=trace,
        policy_trace_payload=policy_trace,
    )
    batch_closure = _read_optional_mapping(job_dir / "simulator_command_batch_closure_manifest.json")
    batch_trace_manifest = _read_optional_mapping(
        job_dir / "simulator_command_batch_trace_package_manifest.json"
    )
    remote_cloud_closure = _read_optional_mapping(
        job_dir / "remote_cloud_execution_closure_manifest.json"
    )
    robot_team_grade_closure = _read_optional_mapping(
        job_dir / "robot_team_grade_eval_closure_manifest.json"
    )
    task_success_summary = _mapping(
        trace.get("task_success_summary")
        or simulator_result.get("task_success_summary")
        or evaluation_result.get("task_success_summary")
    )
    required_count = int(
        batch_closure.get("required_scenario_eval_run_count")
        or trace.get("required_scenario_eval_run_count")
        or scenario_eval_matrix.get("scenario_eval_run_count")
        or len(_scenario_eval_matrix_runs(scenario_eval_matrix))
        or 0
    )
    covered_count = int(
        batch_closure.get("covered_scenario_eval_run_count")
        or trace.get("covered_scenario_eval_run_count")
        or len(simulator_result.get("covered_scenario_eval_run_ids") or [])
        or 0
    )
    missing_count = int(
        batch_closure.get("missing_scenario_eval_run_count")
        or trace.get("missing_scenario_eval_run_count")
        or len(simulator_result.get("missing_scenario_eval_run_ids") or [])
        or max(0, required_count - covered_count)
    )
    coverage_complete = bool(
        batch_closure.get("scenario_eval_run_coverage_complete")
        or trace.get("scenario_eval_run_coverage_complete")
        or simulator_result.get("scenario_eval_run_coverage_complete")
    )
    matrix_run_ids = [
        run_id
        for run_id in (
            _string(_mapping(run).get("scenario_eval_run_id"))
            for run in _scenario_eval_matrix_runs(scenario_eval_matrix)
        )
        if run_id
    ]
    required_run_ids = _string_list(
        batch_closure.get("required_scenario_eval_run_ids")
        or trace.get("required_scenario_eval_run_ids")
        or simulator_result.get("required_scenario_eval_run_ids")
        or matrix_run_ids
    )
    covered_run_ids = _string_list(
        batch_closure.get("covered_scenario_eval_run_ids")
        or trace.get("covered_scenario_eval_run_ids")
        or simulator_result.get("covered_scenario_eval_run_ids")
    )
    missing_run_ids = _string_list(
        batch_closure.get("missing_scenario_eval_run_ids")
        or trace.get("missing_scenario_eval_run_ids")
        or simulator_result.get("missing_scenario_eval_run_ids")
    )
    if not missing_run_ids and required_run_ids and covered_run_ids:
        missing_run_ids = sorted(set(required_run_ids) - set(covered_run_ids))
    blocked_run_records = _explicitly_blocked_scenario_eval_run_records(
        batch_closure,
        trace,
        simulator_result,
    )
    explicitly_blocked_run_ids, invalid_blocked_run_ids = (
        _valid_explicitly_blocked_scenario_eval_run_ids(
            blocked_run_records,
            missing_run_ids=missing_run_ids,
        )
    )
    covered_or_blocked_run_ids = sorted(set(covered_run_ids) | set(explicitly_blocked_run_ids))
    selected_scenario_runs_closed = bool(
        required_count
        and (
            (coverage_complete and missing_count == 0)
            or (required_run_ids and not set(required_run_ids) - set(covered_or_blocked_run_ids))
        )
        and not invalid_blocked_run_ids
    )
    digital_twin_fidelity = _mapping(batch_closure.get("digital_twin_fidelity_qa"))
    policy_interface = _mapping(policy_manifest.get("interface_contract"))
    selected_modalities = _string_list(
        policy_manifest.get("selected_modalities")
        or policy_manifest.get("selected_policy_modalities")
        or request.get("selected_policy_modalities")
        or request.get("policy_modalities")
    )
    supported_modalities = _string_list(
        policy_manifest.get("supported_modalities") or policy_manifest.get("modalities")
    )
    if not supported_modalities:
        supported_modalities = list(POLICY_MODALITY_ORDER)
    simulator_execution_proven = proof_boundary.get("simulator_execution_proven") is True
    robot_policy_execution_proven = (
        policy_execution_manifest.get("robot_policy_execution_proven") is True
        or proof_boundary.get("robot_policy_execution_proven") is True
    )
    real_world_outcome_proven = proof_boundary.get("real_world_outcome_proven") is True
    physics_contact_validated = proof_boundary.get("physics_contact_validated") is True
    non_ranking_operational_claim_validated = (
        proof_boundary.get("non_ranking_operational_claim_validated") is True
    )
    proof_public_claim = proof_boundary.get("public_claim_upgrade_allowed") is True
    rank_fidelity_result_proven = proof_boundary.get("rank_fidelity_result_proven") is True
    success_claim_ledger = _mapping(task_eval_run_report.get("success_claim_ledger"))
    buyer_claim_ceiling = build_buyer_claim_ceiling(
        success_claim_ledger=success_claim_ledger,
        proof_boundary={
            "live_simulator_execution_proven": _mapping(
                live_closure.get("proof_boundary")
            ).get("simulator_execution_proven")
            is True,
            "live_policy_execution_proven": _mapping(
                live_closure.get("proof_boundary")
            ).get("robot_policy_execution_proven")
            is True,
        },
        live_closure=live_closure,
        buyer_copy_inputs={
            "request": {
                "buyer_facing_copy": request.get("buyer_facing_copy"),
                "marketing_copy": request.get("marketing_copy"),
                "report_copy": request.get("report_copy"),
                "public_claims": request.get("public_claims"),
            },
            "task_eval_run_report": {
                "buyer_facing_copy": task_eval_run_report.get("buyer_facing_copy"),
                "marketing_copy": task_eval_run_report.get("marketing_copy"),
                "report_copy": task_eval_run_report.get("report_copy"),
                "public_claims": task_eval_run_report.get("public_claims"),
            },
        },
    )
    machine_trace_complete = bool(
        batch_closure.get("machine_trace_package_complete")
        or simulator_result.get("machine_trace_package_complete")
    )
    robot_team_package_complete = bool(
        batch_closure.get("robot_team_grade_package_complete")
        or simulator_result.get("robot_team_grade_package_complete")
    )
    if revocation_required:
        buyer_display_state = "blocked_consent_revoked_takedown_required"
    elif blockers:
        buyer_display_state = "blocked"
    elif robot_team_package_complete:
        buyer_display_state = "robot_team_package_ready_for_review"
    elif machine_trace_complete or simulator_execution_proven:
        buyer_display_state = "simulator_results_ready_review_required"
    else:
        buyer_display_state = "awaiting_pipeline_evidence"
    return {
        "schema_version": WEBAPP_ROBOT_EVAL_STATUS_PROJECTION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "status": status,
        "state": "blocked" if blockers or revocation_required else "completed",
        "buyer_display_state": buyer_display_state,
        "webapp_role": "display_status_and_proof_boundaries_only",
        "provider_complexity_hidden": True,
        "provider_details_exposed": False,
        "scenario_batch": {
            "status": scenario_eval_matrix.get("status"),
            "scenario_eval_run_count": required_count,
            "target_scenario_eval_run_count": scenario_eval_matrix.get(
                "target_scenario_eval_run_count"
            ),
            "base_scenario_eval_run_count": scenario_eval_matrix.get(
                "base_scenario_eval_run_count"
            ),
            "scenario_eval_batch_expanded": bool(
                scenario_eval_matrix.get("scenario_eval_batch_expanded")
            ),
            "target_scenario_eval_run_count_satisfied": bool(
                scenario_eval_matrix.get("target_scenario_eval_run_count_satisfied")
            ),
            "episode_authoring_contract": _mapping(
                scenario_eval_matrix.get("episode_authoring_contract")
            ),
            "covered_scenario_eval_run_count": covered_count,
            "missing_scenario_eval_run_count": missing_count,
            "explicitly_blocked_scenario_eval_run_count": len(explicitly_blocked_run_ids),
            "selected_scenario_runs_closed": selected_scenario_runs_closed,
            "scenario_eval_run_coverage_complete": coverage_complete,
            "scenario_eval_matrix_path": artifact_paths.get("scenario_eval_matrix"),
        },
        "trace_package": {
            "status": batch_trace_manifest.get("status")
            or trace.get("status")
            or simulator_result.get("status"),
            "machine_trace_package_complete": machine_trace_complete,
            "attempt_trace_path": artifact_paths.get("normalized_attempt_trace"),
            "robot_pov_observation_manifest_path": artifact_paths.get(
                "robot_pov_observation_manifest"
            ),
            "robot_camera_profile_registry_path": artifact_paths.get(
                "robot_camera_profile_registry"
            ),
            "robot_camera_profile_launch_readiness_path": artifact_paths.get(
                "robot_camera_profile_launch_readiness"
            ),
            "robot_pov_observation_candidate_set_path": artifact_paths.get(
                "robot_pov_observation_candidate_set"
            ),
            "selected_initial_policy_observation_path": artifact_paths.get(
                "selected_initial_policy_observation"
            ),
            "robot_pov_frame_sequence_manifest_path": artifact_paths.get(
                "robot_pov_frame_sequence_manifest"
            ),
            "third_person_video_manifest_path": artifact_paths.get("clips_manifest"),
            "contact_stream_path": _mapping(batch_trace_manifest.get("artifact_paths")).get(
                "contact_stream_jsonl"
            ),
        },
        "task_metrics": {
            "evaluation_status": evaluation_result.get("status"),
            "task_success_rate": task_success_summary.get("task_success_rate")
            or task_success_summary.get("success_rate")
            or trace.get("task_success_rate"),
            "successful_attempt_count": task_success_summary.get("successful_attempt_count")
            or trace.get("successful_task_attempt_count"),
            "failed_attempt_count": task_success_summary.get("failed_attempt_count")
            or trace.get("failed_task_attempt_count")
            or labels.get("failed_attempt_count"),
            "task_success_label_provenance_counts": task_success_summary.get(
                "task_success_label_provenance_counts"
            )
            or trace.get("task_success_label_provenance_counts")
            or {},
            "task_success_label_provenance_disclosures": task_success_summary.get(
                "task_success_label_provenance_disclosures"
            )
            or {},
            "generated_video_vlm_judged_attempt_count": task_success_summary.get(
                "generated_video_vlm_judged_attempt_count"
            )
            or 0,
            "success_rate_requires_provenance_disclosure": bool(
                task_success_summary.get("success_rate_requires_provenance_disclosure")
                or trace.get("success_rate_requires_provenance_disclosure")
            ),
            "success_rate_provenance_disclosed": bool(
                task_success_summary.get("success_rate_provenance_disclosed")
                or trace.get("success_rate_provenance_disclosed")
            ),
            "success_rate_buyer_display_allowed": bool(
                task_success_summary.get("success_rate_buyer_display_allowed")
                or trace.get("success_rate_buyer_display_allowed")
            ),
            "success_rate_buyer_display_blockers": _string_list(
                task_success_summary.get("success_rate_buyer_display_blockers")
                or trace.get("success_rate_buyer_display_blockers")
            ),
            "metric_coverage_complete": bool(
                batch_closure.get("metric_coverage_complete")
                or simulator_result.get("metric_coverage_complete")
            ),
            "failure_label_coverage_complete": bool(
                batch_closure.get("failure_label_coverage_complete")
                or failure_diagnosis_audit.get("failure_diagnosis_coverage_complete")
            ),
            "failure_diagnosis_coverage_complete": bool(
                failure_diagnosis_audit.get("failure_diagnosis_coverage_complete")
            ),
            "failure_diagnosis_complete": bool(
                failure_diagnosis_audit.get("failure_diagnosis_complete")
            ),
            "failure_diagnosis_blockers": _string_list(
                failure_diagnosis_audit.get("blockers")
            ),
        },
        "task_eval_run_report": {
            "status": task_eval_run_report.get("status") or "not_available",
            "evidence_level": task_eval_run_report.get("evidence_level"),
            "highest_truthful_claim": buyer_claim_ceiling["highest_truthful_claim"],
            "blockers": _string_list(task_eval_run_report.get("blockers")),
            "report_path": artifact_paths.get("task_eval_run_report"),
            "bare_success_booleans_forbidden": bool(
                _mapping(task_eval_run_report.get("claim_boundary")).get(
                    "bare_success_booleans_forbidden"
                )
            ),
        },
        "buyer_claim_ceiling": buyer_claim_ceiling,
        "batch_closure": {
            "status": batch_closure.get("status") or "not_available",
            "batch_execution_status": batch_closure.get("batch_execution_status"),
            "machine_trace_package_complete": machine_trace_complete,
            "robot_team_grade_package_complete": robot_team_package_complete,
            "robot_team_grade_blockers": _string_list(
                batch_closure.get("robot_team_grade_blockers")
                or simulator_result.get("robot_team_handoff_blockers")
            ),
            "batch_closure_manifest_path": artifact_paths.get(
                "simulator_command_batch_closure_manifest"
            ),
            "batch_trace_package_manifest_path": artifact_paths.get(
                "simulator_command_batch_trace_package_manifest"
            ),
        },
        "digital_twin_fidelity": {
            "status": digital_twin_fidelity.get("status") or "not_available",
            "machine_fidelity_audit_complete": bool(
                digital_twin_fidelity.get("machine_fidelity_audit_complete")
            ),
            "robot_team_grade_fidelity_passed": bool(
                digital_twin_fidelity.get("robot_team_grade_fidelity_passed")
            ),
            "blockers": _string_list(digital_twin_fidelity.get("blockers")),
        },
        "policy_interface": {
            "status": policy_manifest.get("status") or "contract_declared",
            "selected_modalities": selected_modalities,
            "supported_modalities": supported_modalities,
            "observation_schema_id": policy_interface.get("observation_schema_id")
            or POLICY_OBSERVATION_SCHEMA_ID,
            "action_schema_id": policy_interface.get("action_schema_id")
            or POLICY_ACTION_SCHEMA_ID,
            "reproducible_replay_required": bool(
                policy_interface.get("reproducible_replay_required")
                or policy_interface.get("reproducible_replay_contract")
            ),
            "robot_policy_execution_proven": robot_policy_execution_proven,
        },
        "closure_audit": {
            "live_eval_closure_status": live_closure.get("status"),
            "selected_scenario_coverage_closed": selected_scenario_runs_closed,
            "machine_trace_package_complete": machine_trace_complete,
            "robot_team_grade_package_complete": robot_team_package_complete,
            "post_training_data_package_status": data_package_export.get("status"),
            "no_readiness_claim_upgrade_without_evidence": not proof_public_claim
            or rank_fidelity_result_proven,
        },
        "rights_privacy_takedown": {
            "status": revocation_takedown.get("status")
            or ("takedown_required" if revocation_required else "not_required"),
            "consent_revoked": consent_revoked,
            "consent_revoked_at": data_package_consent_evidence.get("consent_revoked_at")
            or revocation_takedown.get("consent_revoked_at"),
            "local_package_access_revoked": bool(
                _boolish(revocation_takedown.get("local_package_access_revoked"))
                or revocation_required
            ),
            "delivery_blocked_by_consent_revocation": bool(
                _boolish(revocation_takedown.get("delivery_blocked"))
                or revocation_required
            ),
            "signed_access_revoked_by_consent": bool(
                _boolish(revocation_takedown.get("signed_access_revoked"))
                or revocation_required
            ),
            "downstream_takedown_required": bool(
                _boolish(revocation_takedown.get("downstream_takedown_required"))
                or revocation_required
            ),
            "webapp_takedown_executed": webapp_takedown_executed,
            "hosted_session_takedown_executed": hosted_session_takedown_executed,
            "webapp_or_hosted_takedown_execution_proven": bool(
                webapp_takedown_executed and hosted_session_takedown_executed
            ),
            "required_actions": _string_list(revocation_takedown.get("required_actions")),
            "downstream_unexecuted_actions": _string_list(
                revocation_takedown.get("downstream_unexecuted_actions")
            ),
            "revocation_takedown_manifest_path": artifact_paths.get(
                "revocation_takedown_manifest"
            )
            or revocation_takedown.get("path"),
            "webapp_rights_privacy_takedown_notice_path": artifact_paths.get(
                "webapp_rights_privacy_takedown_notice"
            )
            or downstream_takedown_artifacts.get(
                "webapp_rights_privacy_takedown_notice"
            ),
            "hosted_session_takedown_request_path": artifact_paths.get(
                "hosted_session_takedown_request"
            )
            or downstream_takedown_artifacts.get("hosted_session_takedown_request"),
        },
        "remote_cloud_execution": {
            "status": remote_cloud_closure.get("status") or "not_available",
            "contract_ready_for_remote_runtime": bool(
                remote_cloud_closure.get("contract_ready_for_remote_runtime")
            ),
            "remote_cloud_execution_proven": bool(
                remote_cloud_closure.get("remote_cloud_execution_proven")
            ),
            "clean_shutdown_proven": bool(
                remote_cloud_closure.get("clean_shutdown_proven")
            ),
            "live_provider_calls_performed": bool(
                remote_cloud_closure.get("live_provider_calls_performed")
            ),
            "blockers": _string_list(remote_cloud_closure.get("blockers")),
            "closure_manifest_path": artifact_paths.get(
                "remote_cloud_execution_closure_manifest"
            ),
        },
        "robot_team_grade_eval_closure": {
            "status": robot_team_grade_closure.get("status") or "not_available",
            "sim_only_beta_core_complete": bool(
                robot_team_grade_closure.get("sim_only_beta_core_complete")
            ),
            "robot_team_grade_evaluation_complete": bool(
                robot_team_grade_closure.get("robot_team_grade_evaluation_complete")
            ),
            "evaluation_readiness_complete": bool(
                robot_team_grade_closure.get("evaluation_readiness_complete")
            ),
            "blocked_requirement_ids": _string_list(
                robot_team_grade_closure.get("blocked_requirement_ids")
            ),
            "all_blocked_requirement_ids": _string_list(
                robot_team_grade_closure.get("all_blocked_requirement_ids")
            ),
            "sim_only_beta_blocked_requirement_ids": _string_list(
                robot_team_grade_closure.get("sim_only_beta_blocked_requirement_ids")
            ),
            "sim_only_customer_handoff_complete": bool(
                robot_team_grade_closure.get("sim_only_customer_handoff_complete")
            ),
            "sim_only_customer_handoff_blocked_requirement_ids": _string_list(
                robot_team_grade_closure.get(
                    "sim_only_customer_handoff_blocked_requirement_ids"
                )
            ),
            "robot_team_grade_blocked_requirement_ids": _string_list(
                robot_team_grade_closure.get("robot_team_grade_blocked_requirement_ids")
            ),
            "evaluation_readiness_blocked_requirement_ids": _string_list(
                robot_team_grade_closure.get("evaluation_readiness_blocked_requirement_ids")
            ),
            "closure_manifest_path": artifact_paths.get(
                "robot_team_grade_eval_closure_manifest"
            ),
        },
        "proof_boundary": {
            "simulator_execution_proven": simulator_execution_proven,
            "robot_policy_execution_proven": robot_policy_execution_proven,
            "real_world_outcome_proven": real_world_outcome_proven,
            "physics_contact_validated": physics_contact_validated,
            "non_ranking_operational_claim_validated": (
                non_ranking_operational_claim_validated
            ),
            "rank_fidelity_result_proven": rank_fidelity_result_proven,
            "public_claim_upgrade_allowed": proof_public_claim,
        },
        "artifact_paths": {
            key: value
            for key, value in artifact_paths.items()
            if key
            in {
                "scenario_eval_matrix",
                "simulator_command_batch_closure_manifest",
                "simulator_command_batch_trace_package_manifest",
                "normalized_attempt_trace",
                "failure_labels",
                "robot_pov_observation_manifest",
                "robot_camera_profile_registry",
                "robot_camera_profile_launch_readiness",
                "robot_pov_frame_sequence_manifest",
                "policy_package_manifest",
                "policy_execution_manifest",
                "evaluation_result",
                "task_eval_run_report",
                "proof_boundary",
                "job_run_manifest",
                "gpu_startup_pipeline_plan",
                "post_training_data_package_export_manifest",
                "customer_handoff_report",
                "delivery_manifest",
                "signed_access_manifest",
                "revocation_takedown_manifest",
                "webapp_rights_privacy_takedown_notice",
                "hosted_session_takedown_request",
                "review_resolution_ledger",
                "accepted_failure_labels",
                "webapp_robot_eval_status_projection",
                "remote_cloud_execution_closure_manifest",
                "robot_team_grade_eval_closure_manifest",
            }
        },
        "buyer_display_guardrails": {
            "must_not_display_as": [
                "generated_world_rank_fidelity",
                "evaluation_readiness",
                "policy_quality_certification",
                "deployment_approval",
                "physical_robot_readiness",
            ],
            "provider_commands_exposed": False,
            "provider_credentials_exposed": False,
            "readiness_claim_upgrade_allowed": proof_public_claim,
            "buyer_facing_claim_ceiling_pinned_to_highest_truthful_claim": (
                buyer_claim_ceiling[
                    "buyer_facing_claim_ceiling_pinned_to_highest_truthful_claim"
                ]
            ),
            "live_simulator_execution_claim_allowed": buyer_claim_ceiling[
                "live_simulator_execution_claim_allowed"
            ],
            "live_policy_execution_claim_allowed": buyer_claim_ceiling[
                "live_policy_execution_claim_allowed"
            ],
            "signed_delivery_access_is_package_access_only": True,
            "consent_revocation_blocks_downstream_use": revocation_required,
            "webapp_or_hosted_takedown_execution_proven": bool(
                webapp_takedown_executed and hosted_session_takedown_executed
            ),
        },
    }

build_webapp_robot_eval_status_projection = _webapp_robot_eval_status_projection

__all__ = [
    "WEBAPP_ROBOT_EVAL_STATUS_PROJECTION_SCHEMA_VERSION",
    "WebappRobotEvalStatusProjection",
    "build_webapp_robot_eval_status_projection",
]
