"""Shared deterministic helpers for robot-eval closure projections."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .common import read_json_any


POLICY_MODALITY_ORDER = (
    "policy_api_endpoint",
    "docker_container",
    "recorded_action_trace",
    "high_level_skill_trace",
    "teleop_demo",
    "sim_controller_plugin",
)


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _number(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "passed",
        "success",
        "succeeded",
    }


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence):
        return [_string(item) for item in value if _string(item)]
    return []


def _dedupe(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = read_json_any(path)
    except Exception:
        return {}
    return _mapping(payload)


def _scenario_eval_matrix_runs(scenario_eval_matrix: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        dict(run)
        for run in scenario_eval_matrix.get("runs", []) or []
        if isinstance(run, Mapping) and _string(run.get("scenario_eval_run_id"))
    ]

def _artifact_paths(job_dir: Path) -> Dict[str, str]:
    names = [
        "job_request.json",
        "job_request_source.json",
        "job_request_enrichment_manifest.json",
        "job_validation.json",
        "job_plan.json",
        "agent_orchestration_plan.json",
        "scheduler_decision.json",
        "worker_launch_plan.json",
        "gpu_startup_pipeline_plan.json",
        "gpu_provisioning_request.json",
        "gpu_provider_launch_request.json",
        "gpu_provider_launcher_result.json",
        "gpu_provider_launcher.stdout.log",
        "gpu_provider_launcher.stderr.log",
        "runpod_provider_adapter_result.json",
        "worker_manifest.json",
        "gpu_cost_control_ledger.json",
        "remote_cloud_execution_closure_manifest.json",
        "robot_team_grade_eval_closure_manifest.json",
        "sim_only_provider_execution_plan.json",
        "sim_only_provider_preflight.json",
        "sim_only_provider_runtime_manifest.json",
        "sim_only_provider_cost_ledger.json",
        "sim_only_provider_artifacts_manifest.json",
        "gpu_provisioning_result.json",
        "simulator_service_request.json",
        "simulator_service_result.json",
        "simulator_provider_adapter_manifest.json",
        "simulator_command_artifacts_manifest.json",
        "simulator_command_digital_twin_fidelity_qa.json",
        "simulator_command_batch_trace_package_manifest.json",
        "simulator_command_batch_attempt_trace.jsonl",
        "simulator_command_batch_contact_stream.jsonl",
        "simulator_command_batch_planner_state.jsonl",
        "simulator_command_batch_control_stream.jsonl",
        "simulator_command_batch_metrics.json",
        "simulator_command_batch_failure_labels.json",
        "simulator_command_batch_visual_media_coverage.json",
        "simulator_command_batch_artifact_checksums.json",
        "simulator_command_batch_closure_manifest.json",
        "scenario_eval_matrix.json",
        "policy_package_manifest.json",
        "evaluation_substrate_registry.json",
        "wam_evaluation_request.json",
        "wam_rollout_manifest.json",
        "wam_rollout_results.json",
        "vision_success_labels.json",
        "policy_ranking_scorecard.json",
        "wam_eval_claim_boundary.json",
        "real_world_validation_followup_request.json",
        "srcc_validation_plan.json",
        "candidate_selection_report.json",
        "candidate_selection_report.md",
        "robot_pov_observation_manifest.json",
        "robot_camera_profile_registry.json",
        "robot_camera_profile_launch_readiness.json",
        "owner_robot_camera_calibration_request.json",
        "robot_pov_observation_candidate_set.json",
        "selected_initial_policy_observation.json",
        "robot_pov_observations.jsonl",
        "robot_pov_frame_sequence_manifest.json",
        "robot_pov_render_storyboard.json",
        "policy_execution_manifest.json",
        "policy_execution_trace.json",
        "policy_execution_trace.jsonl",
        "training_request.json",
        "training_result.json",
        "evaluation_request.json",
        "evaluation_result.json",
        "task_eval_run_report.json",
        "robot_eval_report.json",
        "arena_eval_schedule.json",
        "arena_eval_retry_queue.json",
        "arena_eval_cost_ledger.json",
        "arena_eval_resume_manifest.json",
        "policy_adapter_manifest.json",
        "arena_result_ingest_ledger.json",
        "arena_artifact_checksums.json",
        "arena_eval_metrics.json",
        "normalized_attempt_trace.json",
        "failure_labels.json",
        "clips_manifest.json",
        "rollout_vision_labels.json",
        "review_resolution_ledger.json",
        "accepted_failure_labels.json",
        "prediction_outcome_ledger.json",
        "calibration_report.json",
        "breakage_library.json",
        "deployment_outcome_ledger.json",
        "sim_vs_real_calibration_report.json",
        "prediction_vs_actual_deployment_summary.json",
        "real_world_validation_followup_plan.json",
        "real_world_validation_followup_request_queue.json",
        "live_eval_closure_manifest.json",
        "arena_rerun_plan.json",
        "arena_rerun_lineage.json",
        "customer_handoff_report.json",
        "customer_handoff_report.md",
        "delivery_manifest.json",
        "signed_access_manifest.json",
        "revocation_takedown_manifest.json",
        "webapp_rights_privacy_takedown_notice.json",
        "hosted_session_takedown_request.json",
        "live_operator_ledger.json",
        "startup_architecture_audit.json",
        "worker_runtime_manifest.json",
        "worker_runtime_preflight.json",
        "worker_runtime_preflight.stdout.log",
        "worker_runtime_preflight.stderr.log",
        "dataset_card.json",
        "license_manifest.json",
        "package_index.json",
        "checksums.json",
        "archive_manifest.json",
        "post_training_data_package_export_manifest.json",
        "webapp_robot_eval_status_projection.json",
        "proof_boundary.json",
        "job_claim.json",
        "job_commit.json",
        "job_run_manifest.json",
        "blocked_manifest.json",
    ]
    paths: Dict[str, str] = {}
    for name in names:
        if not (job_dir / name).is_file():
            continue
        path = Path(name)
        key = path.stem
        if path.suffix == ".md":
            key = f"{key}_markdown"
        paths[key] = name
    return paths


def _explicitly_blocked_scenario_eval_run_records(
    *sources: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for source in sources:
        for key in (
            "explicitly_blocked_scenario_eval_runs",
            "blocked_scenario_eval_runs",
            "blocked_scenario_eval_run_records",
            "scenario_eval_run_blockers",
        ):
            raw_records = source.get(key)
            if isinstance(raw_records, Mapping):
                raw_records = [
                    {"scenario_eval_run_id": run_id, **_mapping(record)}
                    for run_id, record in raw_records.items()
                ]
            if isinstance(raw_records, Sequence) and not isinstance(raw_records, (str, bytes)):
                for raw_record in raw_records:
                    if isinstance(raw_record, Mapping):
                        records.append(dict(raw_record))
                    elif _string(raw_record):
                        records.append({"scenario_eval_run_id": _string(raw_record)})
    return records


def _valid_explicitly_blocked_scenario_eval_run_ids(
    records: Sequence[Mapping[str, Any]],
    *,
    missing_run_ids: Sequence[str],
) -> tuple[List[str], List[str]]:
    missing = set(missing_run_ids)
    valid_ids: set[str] = set()
    invalid_ids: set[str] = set()
    for record in records:
        run_id = _string(
            record.get("scenario_eval_run_id")
            or record.get("scenarioEvalRunId")
            or record.get("run_id")
        )
        if not run_id or (missing and run_id not in missing):
            continue
        blockers = _string_list(
            record.get("blockers")
            or record.get("blocker_ids")
            or record.get("failure_mode_ids")
        )
        reason = _string(record.get("reason") or record.get("blocked_reason"))
        stage = _string(record.get("stage") or record.get("blocked_stage"))
        if blockers and reason and stage:
            valid_ids.add(run_id)
        else:
            invalid_ids.add(run_id)
    return sorted(valid_ids), sorted(invalid_ids - valid_ids)


def _capture_root_from_job_dir(job_dir: Path) -> Path | None:
    """Find the capture root (the dir carrying the consent source) above a job dir.

    Robust to nesting depth: walk up until a directory carries the capture's
    consent source or descriptor. Returns None when no capture root is found.
    """
    for parent in (Path(job_dir), *Path(job_dir).parents):
        if (
            (parent / "raw" / "rights_consent.json").exists()
            or (parent / "capture_descriptor.json").exists()
            or (parent / "raw" / "manifest.json").exists()
        ):
            return parent
    return None
