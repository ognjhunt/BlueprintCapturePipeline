"""Typed buyer-facing projection and robot-team closure decisions.

The functions here are deterministic artifact consumers. They centralize the
claim ceiling, failure diagnosis, consent revocation, scenario coverage, and
delivery/closure state machines without running providers or upgrading proof.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, TypedDict

from .failure_diagnosis_contract import build_failure_diagnosis_audit
from .robot_eval_claim_contracts import ROBOT_EVAL_JOB_CLAIM_BOUNDARY
from .robot_eval_status_projection import (  # noqa: F401 - preserve established helpers
    WEBAPP_ROBOT_EVAL_STATUS_PROJECTION_SCHEMA_VERSION,
    _artifact_paths,
    _boolish,
    _capture_root_from_job_dir,
    _dedupe,
    _explicitly_blocked_scenario_eval_run_records,
    _mapping,
    _number,
    _read_optional_mapping,
    _scenario_eval_matrix_runs,
    _string,
    _string_list,
    _valid_explicitly_blocked_scenario_eval_run_ids,
    _webapp_robot_eval_status_projection,
)


ROBOT_TEAM_GRADE_EVAL_CLOSURE_SCHEMA_VERSION = "robot_team_grade_eval_closure.v1"
POLICY_MODALITY_ORDER = (
    "policy_api_endpoint",
    "docker_container",
    "recorded_action_trace",
    "high_level_skill_trace",
    "teleop_demo",
    "sim_controller_plugin",
)
CLAIM_BOUNDARY = ROBOT_EVAL_JOB_CLAIM_BOUNDARY


class WebappRobotEvalStatusProjection(TypedDict, total=False):
    schema_version: str
    generated_at: str
    job_id: str
    status: str
    blockers: List[str]
    claim_boundary: Dict[str, Any]


class RobotTeamGradeEvalClosure(TypedDict, total=False):
    schema_version: str
    generated_at: str
    job_id: str
    status: str
    blockers: List[str]
    claim_boundary: Dict[str, Any]




def _robot_team_grade_eval_closure_manifest(
    *,
    job_dir: Path,
    job_id: str,
    scene_id: str,
    capture_id: str,
    status: str,
    blockers: Sequence[str],
    scenario_eval_matrix: Mapping[str, Any],
    simulator_result: Mapping[str, Any],
    copied_artifacts: Mapping[str, Mapping[str, Any]],
    robot_pov_manifest: Mapping[str, Any],
    policy_manifest: Mapping[str, Any],
    policy_execution_manifest: Mapping[str, Any],
    evaluation_result: Mapping[str, Any],
    proof_boundary: Mapping[str, Any],
    live_closure: Mapping[str, Any],
    remote_cloud_closure: Mapping[str, Any],
    webapp_status_projection: Mapping[str, Any],
    data_package_export: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    artifact_paths = _artifact_paths(job_dir)
    task_eval_run_report = _read_optional_mapping(job_dir / "task_eval_run_report.json")
    trace = _mapping(copied_artifacts.get("normalized_attempt_trace")) or _read_optional_mapping(
        job_dir / "normalized_attempt_trace.json"
    )
    labels = _mapping(copied_artifacts.get("failure_labels")) or _read_optional_mapping(
        job_dir / "failure_labels.json"
    )
    policy_trace = _read_optional_mapping(job_dir / "policy_execution_trace.json")
    batch_closure = _read_optional_mapping(job_dir / "simulator_command_batch_closure_manifest.json")
    digital_twin_fidelity_artifact = _read_optional_mapping(
        job_dir / "simulator_command_digital_twin_fidelity_qa.json"
    )
    visual_media_coverage = _read_optional_mapping(
        job_dir / "simulator_command_batch_visual_media_coverage.json"
    )
    artifact_checksums = _read_optional_mapping(
        job_dir / "simulator_command_batch_artifact_checksums.json"
    )
    batch_metrics = _read_optional_mapping(job_dir / "simulator_command_batch_metrics.json")
    simulator_batch_labels = _read_optional_mapping(
        job_dir / "simulator_command_batch_failure_labels.json"
    )
    batch_trace_manifest = _read_optional_mapping(
        job_dir / "simulator_command_batch_trace_package_manifest.json"
    )
    calibration_report = _read_optional_mapping(job_dir / "sim_vs_real_calibration_report.json")
    policy_ranking_scorecard = _read_optional_mapping(job_dir / "policy_ranking_scorecard.json")
    candidate_selection_report = _read_optional_mapping(job_dir / "candidate_selection_report.json")
    wam_claim_boundary = _read_optional_mapping(job_dir / "wam_eval_claim_boundary.json")
    policy_comparison_policy_count = int(
        _number(policy_ranking_scorecard.get("policy_count"), 0) or 0
    )
    policy_comparison_status = _string(policy_ranking_scorecard.get("status"))
    policy_comparison_completed_statuses = {
        "completed",
        "completed_low_confidence_ranking",
        "completed_ambiguous_ranking",
        "completed_visual_review_required",
    }
    policy_comparison_required_run_ids = _string_list(
        policy_ranking_scorecard.get("required_scenario_eval_run_ids")
    )
    policy_comparison_missing_by_policy = {
        _string(policy_id): _string_list(missing)
        for policy_id, missing in _mapping(
            policy_ranking_scorecard.get("missing_by_policy")
        ).items()
    }
    policy_comparison_extra_by_policy = {
        _string(policy_id): _string_list(extra)
        for policy_id, extra in _mapping(policy_ranking_scorecard.get("extra_by_policy")).items()
    }
    policy_comparison_attempt_count_by_policy = {
        _string(policy_id): int(_number(count, 0) or 0)
        for policy_id, count in _mapping(
            policy_ranking_scorecard.get("attempt_count_by_policy")
        ).items()
    }
    policy_comparison_coverage_rows = [
        _mapping(item)
        for item in policy_ranking_scorecard.get("per_policy_coverage", []) or []
        if isinstance(item, Mapping)
    ]
    policy_comparison_coverage_complete = bool(
        policy_ranking_scorecard.get("coverage_complete")
    )
    policy_comparison_score_ranges_valid = bool(
        policy_ranking_scorecard.get("score_ranges_valid")
    )
    policy_comparison_blockers = _string_list(
        policy_ranking_scorecard.get("comparison_blockers")
    )
    policy_comparison_visual_review_blockers = _string_list(
        policy_ranking_scorecard.get("visual_review_blockers")
    )
    policy_comparison_review_grade_complete = bool(
        policy_ranking_scorecard.get("review_grade_policy_ranking") is True
        and policy_ranking_scorecard.get(
            "visual_rollout_useful_for_task_success_review"
        )
        is True
    )
    policy_comparison_explicit_evaluator_only = bool(
        policy_ranking_scorecard.get("fixture_evaluator_only") is True
        or policy_ranking_scorecard.get("simulator_evaluator_only") is True
        or policy_ranking_scorecard.get("evaluator_only") is True
    )
    policy_comparison_missing_required_scenarios = sorted(
        {
            run_id
            for missing in policy_comparison_missing_by_policy.values()
            for run_id in missing
        }
    )
    policy_comparison_extra_scenarios = sorted(
        {
            run_id
            for extra in policy_comparison_extra_by_policy.values()
            for run_id in extra
        }
    )
    policy_comparison_attempt_count_mismatch = bool(
        policy_comparison_required_run_ids
        and any(
            count != len(policy_comparison_required_run_ids)
            for count in policy_comparison_attempt_count_by_policy.values()
        )
    )
    policy_comparison_boundary = _mapping(policy_ranking_scorecard.get("claim_boundary"))
    policy_comparison_contract = _mapping(policy_ranking_scorecard.get("comparison_contract"))
    policy_comparison_non_overclaiming_boundary = bool(
        policy_comparison_boundary.get("policy_ranking_is_evaluator_bounded") is True
        and policy_comparison_boundary.get("policy_ranking_is_not_evaluation_readiness") is True
        and policy_comparison_boundary.get("rank_fidelity_result_proven") is False
        and policy_comparison_boundary.get("public_claim_upgrade_allowed") is False
        and policy_comparison_contract.get("evaluation_readiness_claimed") is False
        and policy_comparison_contract.get("external_deployment_grade_claimed") is False
    )
    policy_comparison_symmetric_coverage = bool(
        policy_comparison_coverage_complete
        and policy_comparison_coverage_rows
        and all(
            set(_string_list(row.get("required_scenario_eval_run_ids")))
            == set(policy_comparison_required_run_ids)
            and bool(row.get("coverage_complete"))
            for row in policy_comparison_coverage_rows
        )
    )
    evaluator_policy_comparison_contract_complete = bool(
        policy_comparison_status in policy_comparison_completed_statuses
        and policy_comparison_policy_count >= 2
        and policy_comparison_symmetric_coverage
        and policy_comparison_score_ranges_valid
        and not policy_comparison_missing_required_scenarios
        and not policy_comparison_extra_scenarios
        and not policy_comparison_attempt_count_mismatch
        and not policy_comparison_blockers
        and policy_comparison_non_overclaiming_boundary
    )
    evaluator_policy_comparison_complete = bool(
        evaluator_policy_comparison_contract_complete
        and (
            policy_comparison_review_grade_complete
            or policy_comparison_explicit_evaluator_only
        )
    )
    digital_twin_fidelity = digital_twin_fidelity_artifact or _mapping(
        batch_closure.get("digital_twin_fidelity_qa")
    )
    policy_interface = _mapping(policy_manifest.get("interface_contract"))
    scorecard = _mapping(evaluation_result.get("standard_policy_scorecard"))
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
    uncovered_or_unblocked_run_ids = sorted(set(required_run_ids) - set(covered_or_blocked_run_ids))
    missing_without_explicit_blockers = sorted(
        set(missing_run_ids) - set(explicitly_blocked_run_ids)
    )
    selected_scenario_runs_closed = bool(
        required_count
        and (
            (coverage_complete and missing_count == 0)
            or (
                required_run_ids
                and not uncovered_or_unblocked_run_ids
                and not missing_without_explicit_blockers
                and not invalid_blocked_run_ids
            )
        )
    )
    no_claim_upgrade = (
        not bool(proof_boundary.get("public_claim_upgrade_allowed"))
        and not bool(proof_boundary.get("rank_fidelity_result_proven"))
    )
    live_closure_gates = _mapping(live_closure.get("gates"))

    def live_gate_summary(gate_id: str) -> Dict[str, Any]:
        gate = _mapping(live_closure_gates.get(gate_id))
        return {
            "gate_id": gate_id,
            "present": bool(gate),
            "passed": bool(gate.get("passed")),
            "blockers": _string_list(gate.get("blockers")),
            "evidence": _mapping(gate.get("evidence")),
        }

    webapp_upstream_gate = live_gate_summary("webapp_upstream_truth")
    rights_privacy_gate = live_gate_summary("rights_privacy_scope")
    review_acceptance_gate = live_gate_summary("review_acceptance")
    signed_delivery_gate = live_gate_summary("signed_delivery_access")
    data_package_export_ready = (
        data_package_export.get("status") == "export_ready_review_required"
    )
    customer_handoff_artifacts = {
        "post_training_data_package_export_manifest": artifact_paths.get(
            "post_training_data_package_export_manifest"
        ),
        "proof_boundary": artifact_paths.get("proof_boundary"),
        "live_eval_closure_manifest": artifact_paths.get("live_eval_closure_manifest"),
        "customer_handoff_report": artifact_paths.get("customer_handoff_report"),
        "task_eval_run_report": artifact_paths.get("task_eval_run_report"),
        "delivery_manifest": artifact_paths.get("delivery_manifest"),
        "signed_access_manifest": artifact_paths.get("signed_access_manifest"),
        "review_resolution_ledger": artifact_paths.get("review_resolution_ledger"),
        "accepted_failure_labels": artifact_paths.get("accepted_failure_labels"),
    }
    required_customer_handoff_artifacts = {
        "post_training_data_package_export_manifest",
        "proof_boundary",
        "live_eval_closure_manifest",
        "customer_handoff_report",
        "task_eval_run_report",
        "delivery_manifest",
    }
    missing_customer_handoff_artifacts = sorted(
        key
        for key in required_customer_handoff_artifacts
        if not customer_handoff_artifacts.get(key)
    )
    signed_delivery_record_present = bool(
        customer_handoff_artifacts.get("signed_access_manifest")
        or signed_delivery_gate["passed"]
    )
    customer_handoff_boundary_clean = bool(
        no_claim_upgrade
        and not bool(proof_boundary.get("deployment_approval_proven"))
        and not bool(proof_boundary.get("physical_robot_readiness_proven"))
        and not bool(proof_boundary.get("delivery_access_is_deployment_approval"))
        and not bool(proof_boundary.get("package_delivery_is_deployment_approval"))
    )
    customer_handoff_ready = bool(
        data_package_export_ready
        and not missing_customer_handoff_artifacts
        and webapp_upstream_gate["passed"]
        and rights_privacy_gate["passed"]
        and review_acceptance_gate["passed"]
        and signed_delivery_gate["passed"]
        and signed_delivery_record_present
        and customer_handoff_boundary_clean
    )

    def requirement(
        requirement_id: str,
        *,
        title: str,
        passed: bool,
        blockers: Sequence[str] = (),
        evidence_paths: Sequence[str] = (),
        sim_only_beta_required: bool = True,
        sim_only_customer_handoff_required: bool = False,
        robot_team_grade_required: bool = True,
        evaluation_readiness_required: bool = True,
        notes: Sequence[str] = (),
    ) -> Dict[str, Any]:
        deduped_blockers = _dedupe(blockers)
        return {
            "requirement_id": requirement_id,
            "title": title,
            "status": "passed" if passed else "blocked",
            "passed": bool(passed),
            "sim_only_beta_required": bool(sim_only_beta_required),
            "sim_only_customer_handoff_required": bool(
                sim_only_customer_handoff_required
            ),
            "robot_team_grade_required": bool(robot_team_grade_required),
            "evaluation_readiness_required": bool(evaluation_readiness_required),
            "blockers": deduped_blockers,
            "evidence_paths": [path for path in evidence_paths if path],
            "notes": [note for note in notes if note],
        }

    def first_present(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    def failure_label_rows(payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
        return [
            dict(label)
            for label in payload.get("labels", []) or []
            if isinstance(label, Mapping)
        ]

    def scenario_failure_trace_from_labels(
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        attempts: list[Dict[str, Any]] = []
        seen_attempt_keys: set[str] = set()
        for index, label in enumerate(failure_label_rows(payload), start=1):
            scenario_eval_run_id = _string(
                label.get("scenario_eval_run_id") or label.get("scenarioEvalRunId")
            )
            attempt_id = _string(
                label.get("attempt_id")
                or label.get("attemptId")
                or label.get("policy_attempt_id")
                or label.get("policyAttemptId")
            )
            if not scenario_eval_run_id and not attempt_id:
                continue
            attempt_key = attempt_id or scenario_eval_run_id
            if attempt_key in seen_attempt_keys:
                continue
            seen_attempt_keys.add(attempt_key)
            attempt: Dict[str, Any] = {
                "status": "failed",
                "success": False,
                "failure_label_id": _string(label.get("label_id") or label.get("labelId"))
                or f"sim_failure_label_{index:04d}",
                "failure_mode_ids": _string_list(
                    label.get("failure_mode_ids")
                    or label.get("failureModeIds")
                    or label.get("failure_modes")
                    or label.get("failureModes")
                ),
            }
            if scenario_eval_run_id:
                attempt["scenario_eval_run_id"] = scenario_eval_run_id
            if attempt_id:
                attempt["attempt_id"] = attempt_id
            attempts.append(attempt)
        required_run_ids = sorted(
            {
                _string(
                    attempt.get("scenario_eval_run_id") or attempt.get("scenarioEvalRunId")
                )
                for attempt in attempts
                if _string(
                    attempt.get("scenario_eval_run_id") or attempt.get("scenarioEvalRunId")
                )
            }
        )
        return {
            "status": "completed_with_failures" if attempts else "completed",
            "attempts": attempts,
            "required_scenario_eval_run_ids": required_run_ids,
            "covered_scenario_eval_run_ids": required_run_ids,
            "missing_scenario_eval_run_ids": [],
            "scenario_eval_run_coverage_complete": bool(required_run_ids),
        }

    simulator_batch_failure_trace = scenario_failure_trace_from_labels(
        simulator_batch_labels
    )
    simulator_batch_failed_attempt_count = _number(
        simulator_batch_labels.get("failed_attempt_count")
    )
    simulator_batch_metric_failed_attempt_count = _number(
        batch_metrics.get("failed_attempt_count")
    )
    simulator_batch_zero_failures_reviewed = bool(
        _string(simulator_batch_labels.get("status"))
        in {"no_failures_labeled", "zero_failures_reviewed"}
        and simulator_batch_failed_attempt_count == 0
        and simulator_batch_metric_failed_attempt_count == 0
        and int(_number(batch_metrics.get("attempt_count"), 0) or 0) > 0
        and batch_metrics.get("scenario_eval_run_coverage_complete") is True
    )
    use_simulator_batch_failure_labels = bool(
        (
            failure_label_rows(simulator_batch_labels)
            and simulator_batch_failure_trace.get("attempts")
        )
        or simulator_batch_zero_failures_reviewed
    )
    failure_diagnosis_labels = (
        simulator_batch_labels if use_simulator_batch_failure_labels else labels
    )
    failure_diagnosis_label_artifact_key = (
        "simulator_command_batch_failure_labels"
        if use_simulator_batch_failure_labels
        else "failure_labels"
    )
    failure_diagnosis_label_artifact_path = artifact_paths.get(
        failure_diagnosis_label_artifact_key
    )
    failure_diagnosis_trace = (
        simulator_batch_failure_trace if use_simulator_batch_failure_labels else trace
    )
    failure_diagnosis_policy_trace = (
        {} if use_simulator_batch_failure_labels else policy_trace
    )
    failure_diagnosis_audit = build_failure_diagnosis_audit(
        labels_payload=failure_diagnosis_labels,
        trace_payload=failure_diagnosis_trace,
        policy_trace_payload=failure_diagnosis_policy_trace,
    )

    metric_fields = {
        key
        for key, value in {
            "task_success_rate": first_present(
                trace.get("task_success_rate"),
                task_success_summary.get("task_success_rate"),
                task_success_summary.get("success_rate"),
            ),
            "successful_attempt_count": first_present(
                trace.get("successful_task_attempt_count"),
                task_success_summary.get("successful_attempt_count"),
            ),
            "failed_attempt_count": first_present(
                trace.get("failed_task_attempt_count"),
                task_success_summary.get("failed_attempt_count"),
            ),
            "goal_reached": task_success_summary.get("goal_reached_attempt_count"),
            "cycle_time": _mapping(scorecard.get("cycle_time")).get("sample_count"),
            "fall_count": task_success_summary.get("fall_attempt_count"),
            "clearance": task_success_summary.get("min_clearance_m"),
            "contacts": first_present(
                task_success_summary.get("scene_contact_attempt_count"),
                _mapping(scorecard.get("collision_risk")).get("event_count"),
            ),
            "near_misses": task_success_summary.get("near_miss_event_count"),
            "path_deviation": task_success_summary.get("max_path_deviation_m"),
            "stuck_behavior": task_success_summary.get("stuck_attempt_count"),
            "policy_instability": task_success_summary.get(
                "policy_instability_attempt_count"
            ),
            "collision_risk": _mapping(scorecard.get("collision_risk")).get("event_count"),
            "unsafe_proximity": first_present(
                task_success_summary.get("near_miss_event_count"),
                _mapping(scorecard.get("unsafe_proximity")).get("event_count"),
            ),
        }.items()
        if value is not None
    }
    batch_metric_keys = set(_string_list(batch_metrics.get("required_metric_keys")))
    for row in batch_metrics.get("attempt_metric_rows", []) or []:
        if isinstance(row, Mapping):
            batch_metric_keys.update(_string(key) for key in row.keys() if _string(key))
    batch_metric_aliases = {
        "goal_reached": {"goal_reached"},
        "fall_count": {"fall_count"},
        "clearance": {"min_clearance_m", "clearance_threshold_m"},
        "contacts": {
            "robot_scene_contact_event_count",
            "collision_response_event_count",
        },
        "near_misses": {"near_miss_event_count"},
        "path_deviation": {
            "actual_path_distance_m",
            "path_efficiency_ratio",
            "max_path_deviation_m",
            "mean_path_deviation_m",
        },
        "stuck_behavior": {"stuck_event_count"},
        "policy_instability": {"policy_instability_detected"},
    }
    metric_fields.update(
        field
        for field, aliases in batch_metric_aliases.items()
        if batch_metric_keys.intersection(aliases)
    )
    required_metric_fields = {
        "task_success_rate",
        "successful_attempt_count",
        "failed_attempt_count",
        "goal_reached",
        "cycle_time",
        "fall_count",
        "clearance",
        "contacts",
        "near_misses",
        "path_deviation",
        "stuck_behavior",
        "policy_instability",
        "collision_risk",
        "unsafe_proximity",
    }
    missing_metric_fields = sorted(required_metric_fields - metric_fields)
    metric_coverage_blockers = []
    if not artifact_paths.get("simulator_command_batch_metrics"):
        metric_coverage_blockers.append("batch_metrics_artifact_missing")
    elif not batch_metrics:
        metric_coverage_blockers.append("batch_metrics_manifest_missing_or_empty")
    else:
        expected_metric_row_count = required_count or covered_count
        attempt_metric_row_count = int(
            _number(batch_metrics.get("attempt_metric_row_count"), 0) or 0
        )
        missing_metric_row_count = int(
            _number(batch_metrics.get("missing_metric_row_count"), 0) or 0
        )
        if batch_metrics.get("metric_coverage_complete") is not True:
            metric_coverage_blockers.append("batch_metric_coverage_incomplete")
        if expected_metric_row_count and attempt_metric_row_count != expected_metric_row_count:
            metric_coverage_blockers.append("batch_metric_row_count_mismatch")
        if missing_metric_row_count:
            metric_coverage_blockers.append("batch_metric_rows_missing_required_keys")
    full_trace_evidence = {
        "normalized_attempt_trace": artifact_paths.get("normalized_attempt_trace"),
        "robot_pov_observation_manifest": artifact_paths.get("robot_pov_observation_manifest"),
        "robot_pov_observation_candidate_set": artifact_paths.get(
            "robot_pov_observation_candidate_set"
        ),
        "selected_initial_policy_observation": artifact_paths.get(
            "selected_initial_policy_observation"
        ),
        "robot_pov_frame_sequence_manifest": artifact_paths.get(
            "robot_pov_frame_sequence_manifest"
        ),
        "failure_labels": artifact_paths.get("failure_labels"),
        "metrics": artifact_paths.get("simulator_command_batch_metrics"),
        "visual_media_coverage": artifact_paths.get(
            "simulator_command_batch_visual_media_coverage"
        ),
        "artifact_checksums": artifact_paths.get(
            "simulator_command_batch_artifact_checksums"
        ),
        "batch_trace_package_manifest": artifact_paths.get(
            "simulator_command_batch_trace_package_manifest"
        ),
        "contact_stream": artifact_paths.get("simulator_command_batch_contact_stream"),
        "planner_state_stream": artifact_paths.get("simulator_command_batch_planner_state"),
        "control_stream": artifact_paths.get("simulator_command_batch_control_stream"),
        "third_person_video_manifest": artifact_paths.get("clips_manifest")
        or artifact_paths.get("simulator_command_batch_visual_media_coverage"),
    }
    missing_trace_parts = sorted(
        key for key, value in full_trace_evidence.items() if not value
    )
    visual_media_blockers = []
    if not visual_media_coverage:
        visual_media_blockers.append("visual_media_coverage_manifest_missing")
    else:
        if visual_media_coverage.get("all_required_runs_have_visual_recording") is not True:
            visual_media_blockers.append("visual_media_coverage_not_complete_for_all_runs")
        if visual_media_coverage.get("all_required_runs_have_robot_pov_video") is not True:
            visual_media_blockers.append("robot_pov_video_coverage_not_complete")
        if visual_media_coverage.get("all_required_runs_have_third_person_video") is not True:
            visual_media_blockers.append("third_person_video_coverage_not_complete")
    stream_coverage_blockers = []
    if artifact_paths.get("simulator_command_batch_trace_package_manifest") and not batch_trace_manifest:
        stream_coverage_blockers.append("batch_trace_package_manifest_missing_or_empty")
    if batch_trace_manifest:
        if batch_trace_manifest.get("contact_stream_record_count") is None:
            stream_coverage_blockers.append("contact_stream_record_count_missing")
        if batch_trace_manifest.get("planner_state_coverage_complete") is not True:
            stream_coverage_blockers.append("planner_state_coverage_not_complete")
        if batch_trace_manifest.get("control_stream_coverage_complete") is not True:
            stream_coverage_blockers.append("control_stream_coverage_not_complete")
    checksum_blockers = []
    required_checksum_artifacts = {
        "attempt_trace_jsonl",
        "contact_stream_jsonl",
        "planner_state_jsonl",
        "control_stream_jsonl",
        "metrics",
        "failure_labels",
        "visual_media_coverage",
    }
    if artifact_paths.get("simulator_command_batch_artifact_checksums") and not artifact_checksums:
        checksum_blockers.append("artifact_checksums_manifest_missing_or_empty")
    if artifact_checksums:
        checksum_artifacts = _mapping(artifact_checksums.get("artifacts"))
        missing_checksum_artifacts = sorted(
            required_checksum_artifacts - set(checksum_artifacts)
        )
        checksum_blockers.extend(
            f"artifact_checksum_missing_{key}" for key in missing_checksum_artifacts
        )
        checksum_blockers.extend(
            f"artifact_checksum_artifact_absent_{key}"
            for key in sorted(required_checksum_artifacts & set(checksum_artifacts))
            if _mapping(checksum_artifacts.get(key)).get("present") is not True
        )
    digital_twin_fidelity_blockers = []
    if not artifact_paths.get("simulator_command_digital_twin_fidelity_qa"):
        digital_twin_fidelity_blockers.append("digital_twin_fidelity_qa_artifact_missing")
    elif not digital_twin_fidelity_artifact:
        digital_twin_fidelity_blockers.append(
            "digital_twin_fidelity_qa_artifact_missing_or_empty"
        )
    if not bool(digital_twin_fidelity.get("robot_team_grade_fidelity_passed")):
        digital_twin_fidelity_blockers.extend(
            _string_list(digital_twin_fidelity.get("blockers"))
            or ["digital_twin_fidelity_qa_not_passed"]
        )
    selected_policy_modalities = _string_list(policy_manifest.get("selected_modalities"))
    policy_modalities = _mapping(policy_manifest.get("modalities"))
    selected_policy_statuses = {
        modality: _string(_mapping(policy_modalities.get(modality)).get("status"))
        for modality in selected_policy_modalities
    }
    selected_policy_missing_inputs = {
        modality: _string_list(_mapping(policy_modalities.get(modality)).get("missing_inputs"))
        for modality in selected_policy_modalities
    }
    invalid_selected_policy_modalities = sorted(
        modality
        for modality, modality_status in selected_policy_statuses.items()
        if modality_status in {"", "blocked", "not_selected"}
        or selected_policy_missing_inputs.get(modality)
    )
    docker_runtime = _mapping(
        _mapping(_mapping(policy_modalities.get("docker_container")).get("interface_contract")).get(
            "container_runtime"
        )
    )
    selected_docker_runtime_not_versioned = bool(
        "docker_container" in selected_policy_modalities
        and not docker_runtime.get("versioned_runtime_image_proven")
    )
    policy_interface_blockers = []
    if not (
        policy_interface.get("observation_schema_id")
        and policy_interface.get("action_schema_id")
        and policy_interface.get("reproducible_replay_required")
    ):
        policy_interface_blockers.append("policy_interface_contract_incomplete")
    if not selected_policy_modalities:
        policy_interface_blockers.append("policy_package_no_selected_modality")
    if _string(policy_manifest.get("status")) == "blocked" or policy_manifest.get(
        "missing_inputs"
    ):
        policy_interface_blockers.append("policy_package_validation_blocked")
    if invalid_selected_policy_modalities:
        policy_interface_blockers.append("policy_interface_selected_modalities_invalid")
    if selected_docker_runtime_not_versioned:
        policy_interface_blockers.append("policy_docker_container_runtime_image_not_versioned")
    policy_interface_ready = not policy_interface_blockers
    failure_diagnosis_blockers = _string_list(failure_diagnosis_audit.get("blockers"))
    failure_diagnosis_complete = bool(
        failure_diagnosis_labels
        and failure_diagnosis_audit.get("failure_diagnosis_complete")
    )
    task_metric_closure_complete = bool(
        not missing_metric_fields
        and failure_diagnosis_labels
        and not metric_coverage_blockers
        and failure_diagnosis_complete
    )
    full_trace_package_complete = bool(
        not missing_trace_parts
        and not visual_media_blockers
        and not stream_coverage_blockers
        and not checksum_blockers
    )
    required_closure_artifacts = {
        "scenario_eval_matrix",
        "live_eval_closure_manifest",
        "proof_boundary",
        "post_training_data_package_export_manifest",
    }
    missing_closure_artifacts = sorted(
        key for key in required_closure_artifacts if not artifact_paths.get(key)
    )
    requirements = [
        requirement(
            "batch_scenario_execution",
            title="Every selected scenario_eval_run_id ran or is explicitly blocked",
            passed=selected_scenario_runs_closed,
            blockers=[]
            if selected_scenario_runs_closed
            else [
                *(["scenario_eval_run_coverage_incomplete"] if not coverage_complete else []),
                *(["scenario_eval_matrix_empty"] if not required_count else []),
                *(
                    ["missing_scenario_eval_run_ids_not_listed"]
                    if missing_count and not missing_run_ids
                    else []
                ),
                *(
                    ["scenario_eval_run_missing_without_explicit_blockers"]
                    if missing_without_explicit_blockers
                    else []
                ),
                *(
                    ["scenario_eval_run_blocker_records_missing_required_fields"]
                    if invalid_blocked_run_ids
                    else []
                ),
                *(
                    ["scenario_eval_run_covered_or_blocked_set_incomplete"]
                    if uncovered_or_unblocked_run_ids
                    else []
                ),
            ],
            evidence_paths=[
                artifact_paths.get("scenario_eval_matrix"),
                artifact_paths.get("simulator_command_batch_closure_manifest"),
            ],
            notes=[
                f"required={required_count}",
                f"covered={covered_count}",
                f"missing={missing_count}",
                f"explicitly_blocked={len(explicitly_blocked_run_ids)}",
            ],
        ),
        requirement(
            "task_success_metrics",
            title="Task success metrics and failure labels are computed",
            passed=not missing_metric_fields
            and bool(failure_diagnosis_labels)
            and not metric_coverage_blockers,
            blockers=[
                *[f"missing_metric_{field}" for field in missing_metric_fields],
                *(["failure_labels_missing"] if not failure_diagnosis_labels else []),
                *metric_coverage_blockers,
            ],
            evidence_paths=[
                artifact_paths.get("normalized_attempt_trace"),
                failure_diagnosis_label_artifact_path,
                artifact_paths.get("simulator_command_batch_metrics"),
                artifact_paths.get("evaluation_result"),
            ],
        ),
        requirement(
            "failure_diagnosis",
            title="Failure diagnosis labels are evidence-backed and accepted or reviewable",
            passed=failure_diagnosis_complete,
            blockers=[
                *(["failure_labels_missing"] if not failure_diagnosis_labels else []),
                *failure_diagnosis_blockers,
            ],
            evidence_paths=[
                artifact_paths.get("normalized_attempt_trace"),
                failure_diagnosis_label_artifact_path,
            ],
            notes=[
                f"label_source={failure_diagnosis_label_artifact_key}",
                "failure_diagnosis_coverage_complete="
                f"{bool(failure_diagnosis_audit.get('failure_diagnosis_coverage_complete'))}",
                "failure_diagnosis_review_complete="
                f"{bool(failure_diagnosis_audit.get('failure_diagnosis_review_complete'))}",
                "zero_failures_reviewed="
                f"{bool(failure_diagnosis_audit.get('zero_failures_reviewed'))}",
            ],
        ),
        requirement(
            "digital_twin_fidelity_qa",
            title="Visual and collision parity has a digital-twin QA result",
            passed=bool(digital_twin_fidelity.get("robot_team_grade_fidelity_passed"))
            and not digital_twin_fidelity_blockers,
            blockers=digital_twin_fidelity_blockers,
            evidence_paths=[
                artifact_paths.get("simulator_command_digital_twin_fidelity_qa"),
                artifact_paths.get("simulator_command_batch_closure_manifest"),
            ],
            sim_only_beta_required=False,
        ),
        requirement(
            "robot_team_policy_interface",
            title="Policy interface schemas and replay contract are declared",
            passed=policy_interface_ready,
            blockers=policy_interface_blockers,
            evidence_paths=[
                artifact_paths.get("policy_package_manifest"),
                artifact_paths.get("policy_execution_manifest"),
            ],
            notes=[
                f"selected_modalities={','.join(selected_policy_modalities)}",
                *[
                    f"{modality}_status={status}"
                    for modality, status in selected_policy_statuses.items()
                ],
            ],
            sim_only_beta_required=False,
        ),
        requirement(
            "full_trace_package",
            title="Trace package has POV, third-person, contacts, metrics, labels, and manifests",
            passed=(
                not missing_trace_parts
                and not visual_media_blockers
                and not stream_coverage_blockers
                and not checksum_blockers
            ),
            blockers=[
                *[f"missing_trace_artifact_{key}" for key in missing_trace_parts],
                *visual_media_blockers,
                *stream_coverage_blockers,
                *checksum_blockers,
            ],
            evidence_paths=[value for value in full_trace_evidence.values() if value],
        ),
        requirement(
            "remote_cloud_execution_path",
            title="Provider/worker path has pinned inputs, cost controls, timeout, and shutdown proof",
            passed=bool(
                remote_cloud_closure.get("remote_cloud_execution_proven")
                and remote_cloud_closure.get("clean_shutdown_proven")
            ),
            blockers=_string_list(remote_cloud_closure.get("blockers"))
            or ["remote_cloud_execution_not_proven"],
            evidence_paths=[artifact_paths.get("remote_cloud_execution_closure_manifest")],
            sim_only_beta_required=False,
        ),
        requirement(
            "end_to_end_webapp_flow",
            title="Pipeline emits WebApp-safe proof-boundary status without provider complexity",
            passed=bool(
                webapp_status_projection
                and webapp_status_projection.get("provider_complexity_hidden") is True
                and webapp_status_projection.get("provider_details_exposed") is False
            ),
            blockers=[]
            if (
                webapp_status_projection
                and webapp_status_projection.get("provider_complexity_hidden") is True
                and webapp_status_projection.get("provider_details_exposed") is False
            )
            else ["webapp_status_projection_missing_or_provider_details_exposed"],
            evidence_paths=[artifact_paths.get("webapp_robot_eval_status_projection")],
        ),
        requirement(
            "closure_audit",
            title="Closure gate preserves artifact presence and blocks unsupported claim upgrades",
            passed=bool(
                no_claim_upgrade
                and not blockers
                and selected_scenario_runs_closed
                and task_metric_closure_complete
                and full_trace_package_complete
                and not missing_closure_artifacts
            ),
            blockers=[
                *(["job_blockers_present"] if blockers else []),
                *(["readiness_claim_upgrade_present"] if not no_claim_upgrade else []),
                *(
                    ["selected_scenario_run_closure_incomplete"]
                    if not selected_scenario_runs_closed
                    else []
                ),
                *(
                    ["task_metric_closure_incomplete"]
                    if not task_metric_closure_complete
                    else []
                ),
                *(
                    ["full_trace_package_incomplete"]
                    if not full_trace_package_complete
                    else []
                ),
                *[
                    f"closure_artifact_missing_{artifact_key}"
                    for artifact_key in missing_closure_artifacts
                ],
            ],
            evidence_paths=[
                artifact_paths.get("live_eval_closure_manifest"),
                artifact_paths.get("proof_boundary"),
                artifact_paths.get("post_training_data_package_export_manifest"),
            ],
        ),
        requirement(
            "package_delivery_handoff",
            title="Customer handoff references package export, rights/privacy, review, and signed access",
            passed=customer_handoff_ready,
            blockers=[]
            if customer_handoff_ready
            else [
                *(
                    ["post_training_data_package_export_not_ready"]
                    if not data_package_export_ready
                    else []
                ),
                *[
                    f"customer_handoff_artifact_missing_{artifact_key}"
                    for artifact_key in missing_customer_handoff_artifacts
                ],
                *(
                    ["signed_access_manifest_missing"]
                    if not signed_delivery_record_present
                    else []
                ),
                *(
                    ["webapp_upstream_truth_gate_missing"]
                    if not webapp_upstream_gate["present"]
                    else []
                ),
                *(
                    ["rights_privacy_scope_gate_missing"]
                    if not rights_privacy_gate["present"]
                    else []
                ),
                *(
                    ["review_acceptance_gate_missing"]
                    if not review_acceptance_gate["present"]
                    else []
                ),
                *(
                    ["signed_delivery_access_gate_missing"]
                    if not signed_delivery_gate["present"]
                    else []
                ),
                *[
                    f"webapp_upstream_truth:{blocker}"
                    for blocker in webapp_upstream_gate["blockers"]
                ],
                *[
                    f"rights_privacy_scope:{blocker}"
                    for blocker in rights_privacy_gate["blockers"]
                ],
                *[
                    f"review_acceptance:{blocker}"
                    for blocker in review_acceptance_gate["blockers"]
                ],
                *[
                    f"signed_delivery_access:{blocker}"
                    for blocker in signed_delivery_gate["blockers"]
                ],
                *(
                    ["package_delivery_boundary_allows_claim_upgrade"]
                    if not customer_handoff_boundary_clean
                    else []
                ),
            ],
            evidence_paths=[
                customer_handoff_artifacts.get("post_training_data_package_export_manifest"),
                customer_handoff_artifacts.get("proof_boundary"),
                customer_handoff_artifacts.get("live_eval_closure_manifest"),
                customer_handoff_artifacts.get("customer_handoff_report"),
                customer_handoff_artifacts.get("delivery_manifest"),
                customer_handoff_artifacts.get("signed_access_manifest"),
            ],
            sim_only_beta_required=False,
            sim_only_customer_handoff_required=True,
            robot_team_grade_required=False,
            evaluation_readiness_required=False,
            notes=[
                "sim_only_beta_core_evidence_is_separate_from_customer_delivery",
                "signed_delivery_proves_package_access_not_deployment_approval",
            ],
        ),
        requirement(
            "evaluator_bounded_policy_comparison",
            title="Policy ranking compares candidate policies inside the configured evaluator",
            passed=evaluator_policy_comparison_complete,
            blockers=[]
            if evaluator_policy_comparison_complete
            else [
                *(
                    ["policy_ranking_scorecard_missing_or_incomplete"]
                    if policy_comparison_status not in policy_comparison_completed_statuses
                    else []
                ),
                *(
                    ["policy_comparison_requires_at_least_two_candidates"]
                    if policy_comparison_policy_count < 2
                    else []
                ),
                *(
                    ["policy_comparison_required_scenario_eval_run_ids_missing"]
                    if not policy_comparison_required_run_ids
                    else []
                ),
                *(
                    ["policy_comparison_policy_coverage_not_symmetric"]
                    if not policy_comparison_symmetric_coverage
                    else []
                ),
                *(
                    ["policy_comparison_missing_required_scenario_eval_run_ids"]
                    if policy_comparison_missing_required_scenarios
                    else []
                ),
                *(
                    ["policy_comparison_extra_scenario_eval_run_ids"]
                    if policy_comparison_extra_scenarios
                    else []
                ),
                *(
                    ["policy_comparison_attempt_count_mismatch"]
                    if policy_comparison_attempt_count_mismatch
                    else []
                ),
                *(
                    ["policy_comparison_score_ranges_invalid"]
                    if not policy_comparison_score_ranges_valid
                    else []
                ),
                *(
                    ["policy_ranking_boundary_not_evaluator_bounded"]
                    if not policy_comparison_non_overclaiming_boundary
                    else []
                ),
                *[
                    f"policy_ranking_scorecard_blocker:{blocker}"
                    for blocker in policy_comparison_blockers
                ],
                *(
                    [
                        f"policy_ranking_visual_review_blocker:{blocker}"
                        for blocker in (
                            policy_comparison_visual_review_blockers
                            or ["policy_ranking_visual_review_gate_not_passed"]
                        )
                    ]
                    if evaluator_policy_comparison_contract_complete
                    and not policy_comparison_review_grade_complete
                    and not policy_comparison_explicit_evaluator_only
                    else []
                ),
            ],
            evidence_paths=[
                artifact_paths.get("policy_ranking_scorecard"),
                artifact_paths.get("candidate_selection_report"),
                artifact_paths.get("wam_eval_claim_boundary"),
            ],
            sim_only_beta_required=False,
            robot_team_grade_required=False,
            evaluation_readiness_required=False,
        ),
        requirement(
            "sim_vs_real_calibration_path",
            title="Real-world anchors calibrate evaluator ranking before external claims",
            passed=bool(calibration_report.get("sim_vs_real_calibration_score") is not None),
            blockers=_string_list(calibration_report.get("blockers"))
            or ["sim_vs_real_calibration_not_required_for_sim_only_beta"],
            evidence_paths=[artifact_paths.get("sim_vs_real_calibration_report")],
            sim_only_beta_required=False,
            robot_team_grade_required=False,
            evaluation_readiness_required=True,
        ),
    ]
    sim_only_required = [item for item in requirements if item["sim_only_beta_required"]]
    sim_only_customer_handoff_required = [
        item for item in requirements if item["sim_only_customer_handoff_required"]
    ]
    robot_team_required = [item for item in requirements if item["robot_team_grade_required"]]
    deployment_required = [
        item for item in requirements if item["evaluation_readiness_required"]
    ]
    all_blocked_requirement_ids = [
        item["requirement_id"] for item in requirements if not item["passed"]
    ]
    sim_only_beta_blocked_requirement_ids = [
        item["requirement_id"]
        for item in sim_only_required
        if not item["passed"]
    ]
    sim_only_customer_handoff_blocked_requirement_ids = [
        item["requirement_id"]
        for item in sim_only_customer_handoff_required
        if not item["passed"]
    ]
    robot_team_grade_blocked_requirement_ids = [
        item["requirement_id"]
        for item in robot_team_required
        if not item["passed"]
    ]
    evaluation_readiness_blocked_requirement_ids = [
        item["requirement_id"]
        for item in deployment_required
        if not item["passed"]
    ]
    return {
        "schema_version": ROBOT_TEAM_GRADE_EVAL_CLOSURE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "job_status": status,
        "status": "robot_team_grade_complete"
        if all(item["passed"] for item in robot_team_required)
        else "blocked_robot_team_grade_requirements",
        "requirement_count": len(requirements),
        "passed_requirement_count": sum(1 for item in requirements if item["passed"]),
        "blocked_requirement_ids": robot_team_grade_blocked_requirement_ids,
        "all_blocked_requirement_ids": all_blocked_requirement_ids,
        "sim_only_beta_blocked_requirement_ids": sim_only_beta_blocked_requirement_ids,
        "sim_only_customer_handoff_blocked_requirement_ids": (
            sim_only_customer_handoff_blocked_requirement_ids
        ),
        "robot_team_grade_blocked_requirement_ids": robot_team_grade_blocked_requirement_ids,
        "evaluation_readiness_blocked_requirement_ids": (
            evaluation_readiness_blocked_requirement_ids
        ),
        "sim_only_beta_core_complete": all(item["passed"] for item in sim_only_required),
        "sim_only_customer_handoff_complete": all(
            item["passed"] for item in sim_only_customer_handoff_required
        ),
        "robot_team_grade_evaluation_complete": all(
            item["passed"] for item in robot_team_required
        ),
        "evaluation_readiness_complete": all(item["passed"] for item in deployment_required),
        "primary_proof_target": "policy_comparison_within_configured_evaluator",
        "evaluator_bounded_policy_comparison_complete": evaluator_policy_comparison_complete,
        "policy_comparison_summary": {
            "status": policy_comparison_status or "not_available",
            "policy_count": policy_comparison_policy_count,
            "top_policy_id": policy_ranking_scorecard.get("top_policy_id"),
            "evaluator_top_policy_id": policy_ranking_scorecard.get(
                "evaluator_top_policy_id"
            ),
            "single_best_policy_claimed": bool(
                policy_ranking_scorecard.get("single_best_policy_claimed")
            ),
            "ranking_basis": policy_ranking_scorecard.get("ranking_basis"),
            "required_scenario_eval_run_ids": policy_comparison_required_run_ids,
            "coverage_complete": policy_comparison_coverage_complete,
            "symmetric_policy_coverage": policy_comparison_symmetric_coverage,
            "missing_by_policy": policy_comparison_missing_by_policy,
            "extra_by_policy": policy_comparison_extra_by_policy,
            "attempt_count_by_policy": policy_comparison_attempt_count_by_policy,
            "comparison_blockers": policy_comparison_blockers,
            "visual_smoke_status": policy_ranking_scorecard.get("visual_smoke_status"),
            "visual_rollout_useful_for_task_success_review": bool(
                policy_ranking_scorecard.get(
                    "visual_rollout_useful_for_task_success_review"
                )
            ),
            "visual_review_blockers": policy_comparison_visual_review_blockers,
            "fixture_evaluator_only": bool(
                policy_ranking_scorecard.get("fixture_evaluator_only")
            ),
            "simulator_evaluator_only": bool(
                policy_ranking_scorecard.get("simulator_evaluator_only")
            ),
            "explicit_evaluator_only": policy_comparison_explicit_evaluator_only,
            "candidate_behavior_distinctness_proven": bool(
                policy_ranking_scorecard.get("candidate_behavior_distinctness_proven")
            ),
            "review_grade_policy_ranking": policy_comparison_review_grade_complete,
            "score_ranges_valid": policy_comparison_score_ranges_valid,
            "ranking_confidence": _mapping(policy_ranking_scorecard.get("ranking_confidence")),
            "policy_ranking_is_evaluator_bounded": bool(
                policy_comparison_boundary.get("policy_ranking_is_evaluator_bounded")
            ),
            "non_overclaiming_claim_boundary": policy_comparison_non_overclaiming_boundary,
            "claim_boundary_path": "wam_eval_claim_boundary.json" if wam_claim_boundary else None,
            "candidate_selection_report_path": (
                "candidate_selection_report.json" if candidate_selection_report else None
            ),
            "candidate_selection_status": candidate_selection_report.get("status"),
            "candidate_selection_top_policy_id": candidate_selection_report.get(
                "top_policy_id"
            ),
            "candidate_selection_evaluator_top_policy_id": (
                candidate_selection_report.get("evaluator_top_policy_id")
            ),
            "candidate_selection_tie_or_ambiguity_status": (
                candidate_selection_report.get("tie_or_ambiguity_status")
            ),
        },
        "scenario_execution_summary": {
            "required_scenario_eval_run_count": required_count,
            "covered_scenario_eval_run_count": covered_count,
            "missing_scenario_eval_run_count": missing_count,
            "explicitly_blocked_scenario_eval_run_count": len(explicitly_blocked_run_ids),
            "selected_scenario_runs_closed": selected_scenario_runs_closed,
            "required_scenario_eval_run_ids": required_run_ids,
            "covered_scenario_eval_run_ids": covered_run_ids,
            "missing_scenario_eval_run_ids": missing_run_ids,
            "explicitly_blocked_scenario_eval_run_ids": explicitly_blocked_run_ids,
            "missing_without_explicit_blockers": missing_without_explicit_blockers,
            "invalid_explicit_blocker_record_run_ids": invalid_blocked_run_ids,
            "uncovered_or_unblocked_scenario_eval_run_ids": uncovered_or_unblocked_run_ids,
        },
        "closure_audit_summary": {
            "no_readiness_claim_upgrade_without_evidence": no_claim_upgrade,
            "selected_scenario_runs_closed": selected_scenario_runs_closed,
            "task_metric_closure_complete": task_metric_closure_complete,
            "failure_diagnosis_coverage_complete": bool(
                failure_diagnosis_audit.get("failure_diagnosis_coverage_complete")
            ),
            "failure_diagnosis_complete": failure_diagnosis_complete,
            "failure_diagnosis_blockers": failure_diagnosis_blockers,
            "failure_diagnosis_label_source_artifact": (
                failure_diagnosis_label_artifact_path
            ),
            "canonical_failure_labels_artifact": artifact_paths.get("failure_labels"),
            "simulator_batch_failure_labels_artifact": artifact_paths.get(
                "simulator_command_batch_failure_labels"
            ),
            "batch_metric_keys_used_for_task_metric_closure": sorted(batch_metric_keys),
            "full_trace_package_complete": full_trace_package_complete,
            "missing_required_artifacts": missing_closure_artifacts,
            "sim_only_customer_handoff_complete": customer_handoff_ready,
        },
        "package_delivery_handoff_summary": {
            "sim_only_customer_handoff_complete": customer_handoff_ready,
            "post_training_data_package_export_ready": data_package_export_ready,
            "task_eval_run_report_status": task_eval_run_report.get("status")
            or "not_available",
            "task_eval_run_report_evidence_level": task_eval_run_report.get(
                "evidence_level"
            ),
            "customer_handoff_artifacts": customer_handoff_artifacts,
            "missing_customer_handoff_artifacts": missing_customer_handoff_artifacts,
            "webapp_upstream_truth": {
                key: value
                for key, value in webapp_upstream_gate.items()
                if key != "evidence"
            },
            "rights_privacy_scope": {
                key: value
                for key, value in rights_privacy_gate.items()
                if key != "evidence"
            },
            "review_acceptance": {
                key: value
                for key, value in review_acceptance_gate.items()
                if key != "evidence"
            },
            "signed_delivery_access": {
                key: value
                for key, value in signed_delivery_gate.items()
                if key != "evidence"
            },
            "signed_delivery_record_present": signed_delivery_record_present,
            "signed_delivery_proves_package_access_not_deployment_approval": True,
            "customer_handoff_boundary_clean": customer_handoff_boundary_clean,
        },
        "policy_interface_summary": {
            "policy_package_status": policy_manifest.get("status"),
            "selected_modalities": selected_policy_modalities,
            "selected_modality_statuses": selected_policy_statuses,
            "selected_modality_missing_inputs": selected_policy_missing_inputs,
            "invalid_selected_modalities": invalid_selected_policy_modalities,
            "docker_container_runtime_image_versioned": bool(
                docker_runtime.get("versioned_runtime_image_proven")
            )
            if "docker_container" in selected_policy_modalities
            else None,
            "policy_interface_ready": policy_interface_ready,
            "blockers": policy_interface_blockers,
        },
        "requirements": requirements,
        "artifact_paths": {
            key: value
            for key, value in artifact_paths.items()
            if key
            in {
                "scenario_eval_matrix",
                "normalized_attempt_trace",
                "failure_labels",
                "simulator_command_batch_failure_labels",
                "robot_pov_observation_manifest",
                "robot_pov_observation_candidate_set",
                "selected_initial_policy_observation",
                "robot_pov_frame_sequence_manifest",
                "simulator_command_batch_trace_package_manifest",
                "simulator_command_batch_closure_manifest",
                "remote_cloud_execution_closure_manifest",
                "webapp_robot_eval_status_projection",
                "live_eval_closure_manifest",
                "proof_boundary",
                "sim_vs_real_calibration_report",
                "post_training_data_package_export_manifest",
                "task_eval_run_report",
                "customer_handoff_report",
                "delivery_manifest",
                "signed_access_manifest",
                "review_resolution_ledger",
                "accepted_failure_labels",
                "policy_ranking_scorecard",
                "candidate_selection_report",
                "candidate_selection_report_markdown",
                "wam_eval_claim_boundary",
            }
        },
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "primary_proof_target": "policy_comparison_within_configured_evaluator",
            "evaluator_bounded_policy_comparison_complete": evaluator_policy_comparison_complete,
            "review_grade_policy_comparison_complete": policy_comparison_review_grade_complete,
            "visual_smoke_required_for_review_grade_policy_ranking": True,
            "evaluator_bounded_policy_comparison_requires_symmetric_coverage": True,
            "evaluator_bounded_policy_comparison_single_best_policy_claimed": bool(
                policy_ranking_scorecard.get("single_best_policy_claimed")
            ),
            "policy_ranking_is_not_evaluation_readiness": True,
            "traditional_sim_is_optional_cross_check_for_wam_eval": True,
            "spearman_pearson_mmrv_status": "not_measured_until_real_anchors_exist",
            "sim_only_beta_core_complete": all(item["passed"] for item in sim_only_required),
            "sim_only_customer_handoff_complete": customer_handoff_ready,
            "signed_delivery_access_proven": signed_delivery_gate["passed"],
            "review_acceptance_proven": review_acceptance_gate["passed"],
            "rights_privacy_scope_proven": rights_privacy_gate["passed"],
            "webapp_upstream_truth_grounded": webapp_upstream_gate["passed"],
            "customer_handoff_ready": customer_handoff_ready,
            "delivery_access_is_deployment_approval": False,
            "package_delivery_is_deployment_approval": False,
            "deployment_approval_proven": False,
            "physical_robot_readiness_proven": False,
            "safety_validation_proven": False,
            "robot_team_grade_evaluation_complete": all(
                item["passed"] for item in robot_team_required
            ),
            "evaluation_readiness_complete": all(
                item["passed"] for item in deployment_required
            ),
            "public_claim_upgrade_allowed": False,
            "generated_world_rank_fidelity_claimed": False,
        },
    }

build_webapp_robot_eval_status_projection = _webapp_robot_eval_status_projection
build_robot_team_grade_eval_closure = _robot_team_grade_eval_closure_manifest

__all__ = [
    "ROBOT_TEAM_GRADE_EVAL_CLOSURE_SCHEMA_VERSION",
    "WEBAPP_ROBOT_EVAL_STATUS_PROJECTION_SCHEMA_VERSION",
    "RobotTeamGradeEvalClosure",
    "WebappRobotEvalStatusProjection",
    "build_robot_team_grade_eval_closure",
    "build_webapp_robot_eval_status_projection",
]
