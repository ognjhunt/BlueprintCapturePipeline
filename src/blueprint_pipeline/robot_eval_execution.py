"""Robot-eval execution, robot-POV, and deployment calibration helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from . import robot_eval_calibration as _calibration
from .common import ensure_dir, read_json_any, resolve_gs_uri_to_path, write_json
from .robot_initial_observation import build_initial_observation_source_resolution
from .security_controls import (
    SecurityValidationError,
    fetch_bounded_https,
    json_shape_within_limits,
    origins_from_env,
)


ROBOT_POV_OBSERVATION_SCHEMA_VERSION = "robot_pov_observation_manifest.v1"
ROBOT_POV_FRAME_SEQUENCE_SCHEMA_VERSION = "robot_pov_frame_sequence_manifest.v1"
ROBOT_POV_RENDER_STORYBOARD_SCHEMA_VERSION = "robot_pov_render_storyboard.v1"
SCENARIO_EVAL_MATRIX_SCHEMA_VERSION = "robot_eval_scenario_eval_matrix.v1"
POLICY_EXECUTION_MANIFEST_SCHEMA_VERSION = "robot_policy_execution_manifest.v1"
POLICY_EXECUTION_TRACE_SCHEMA_VERSION = "robot_policy_execution_trace.v1"
POLICY_OBSERVATION_SCHEMA_ID = "blueprint.robot_eval.observation.v1"
POLICY_ACTION_SCHEMA_ID = "blueprint.robot_eval.action_trace.v1"
POLICY_OBSERVATION_SCHEMA_REF = "blueprint://schemas/robot_eval_observation.v1"
POLICY_ACTION_SCHEMA_REF = "blueprint://schemas/robot_eval_action_trace.v1"
DEPLOYMENT_OUTCOME_LEDGER_SCHEMA_VERSION = "deployment_outcome_ledger.v1"
SIM_VS_REAL_CALIBRATION_SCHEMA_VERSION = "sim_vs_real_calibration_report.v1"
PREDICTION_VS_ACTUAL_DEPLOYMENT_SCHEMA_VERSION = "prediction_vs_actual_deployment_summary.v1"
REAL_WORLD_VALIDATION_FOLLOWUP_PLAN_SCHEMA_VERSION = "real_world_validation_followup_plan.v1"
SIMULATOR_COMMAND_ARTIFACTS_SCHEMA_VERSION = "simulator_command_artifacts.v1"
ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION = _calibration.ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION
ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS = _calibration.ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS
MIN_ACCEPTED_ANCHOR_COUNT_FOR_CALIBRATION = _calibration.MIN_ACCEPTED_ANCHOR_COUNT_FOR_CALIBRATION
MIN_POLICY_GROUP_COUNT_FOR_CALIBRATION = _calibration.MIN_POLICY_GROUP_COUNT_FOR_CALIBRATION
MIN_POLICY_CHECKPOINT_COUNT_FOR_PUBLIC_RANK_FIDELITY = (
    _calibration.MIN_POLICY_CHECKPOINT_COUNT_FOR_PUBLIC_RANK_FIDELITY
)
MIN_CRITERION_COUNT_FOR_PUBLIC_RANK_FIDELITY = (
    _calibration.MIN_CRITERION_COUNT_FOR_PUBLIC_RANK_FIDELITY
)
MIN_REGISTERED_SPLIT_COUNT_FOR_PUBLIC_RANK_FIDELITY = (
    _calibration.MIN_REGISTERED_SPLIT_COUNT_FOR_PUBLIC_RANK_FIDELITY
)
MIN_MATCHED_TRIALS_PER_CELL_FOR_PUBLIC_RANK_FIDELITY = (
    _calibration.MIN_MATCHED_TRIALS_PER_CELL_FOR_PUBLIC_RANK_FIDELITY
)
DEFAULT_CALIBRATION_BOOTSTRAP_SEED = _calibration.DEFAULT_CALIBRATION_BOOTSTRAP_SEED
DEFAULT_CALIBRATION_BOOTSTRAP_REPLICATES = _calibration.DEFAULT_CALIBRATION_BOOTSTRAP_REPLICATES
RANK_FIDELITY_CLAIM_ELIGIBILITY_SCHEMA_VERSION = (
    _calibration.RANK_FIDELITY_CLAIM_ELIGIBILITY_SCHEMA_VERSION
)
UNIT_OF_ANALYSIS_FIELDS = _calibration.UNIT_OF_ANALYSIS_FIELDS

# Backward-compatible private aliases.  New code imports the typed calibration
# module directly; legacy callers keep a stable surface during the split.
_prediction_index = _calibration._prediction_index
_prediction_for_actual = _calibration._prediction_for_actual
_predicted_success = _calibration._predicted_success
_actual_success = _calibration._actual_success
_failure_ids = _calibration._failure_ids
_actual_signal_present = _calibration._actual_signal_present
_anchor_variation_instance_id = _calibration._anchor_variation_instance_id
_anchor_key = _calibration._anchor_key
_anchor_key_dict = _calibration._anchor_key_dict
_missing_anchor_key_fields = _calibration._missing_anchor_key_fields
_anchor_record_status = _calibration._anchor_record_status
_anchor_record_is_stale = _calibration._anchor_record_is_stale
_accepted_review_value = _calibration._accepted_review_value
_anchor_review_accepted = _calibration._anchor_review_accepted
_attestation_signed = _calibration._attestation_signed
_physical_evidence_requested = _calibration._physical_evidence_requested
_physical_evidence_present = _calibration._physical_evidence_present
_prediction_anchor_rows = _calibration._prediction_anchor_rows
_prediction_anchor_index = _calibration._prediction_anchor_index
_average_ranks = _calibration._average_ranks
_pearson = _calibration._pearson
_policy_anchor_summaries = _calibration.policy_anchor_summaries
_summaries_with_rank_position_diagnostics = _calibration._summaries_with_rank_position_diagnostics
_simpler_pairwise_margin_rank_violations = _calibration._simpler_pairwise_margin_rank_violations
_calibration_metrics_from_policy_summaries = _calibration.calibration_metrics_from_policy_summaries
_macro_calibration_estimand = _calibration._macro_calibration_estimand
_registered_split_estimands = _calibration._registered_split_estimands
_percentile = _calibration._percentile
_bootstrap_confidence_intervals = _calibration._bootstrap_confidence_intervals
_rank_fidelity_claim_eligibility = _calibration.evaluate_rank_fidelity_claim_eligibility
_accepted_anchor_calibration = _calibration.build_accepted_anchor_calibration
BATCH_TRACE_ARTIFACT_JOB_NAMES = {
    "attempt_trace_jsonl": "simulator_command_batch_attempt_trace.jsonl",
    "contact_stream_jsonl": "simulator_command_batch_contact_stream.jsonl",
    "planner_state_jsonl": "simulator_command_batch_planner_state.jsonl",
    "control_stream_jsonl": "simulator_command_batch_control_stream.jsonl",
    "metrics": "simulator_command_batch_metrics.json",
    "failure_labels": "simulator_command_batch_failure_labels.json",
    "visual_media_coverage": "simulator_command_batch_visual_media_coverage.json",
    "visual_review_ledger": "simulator_command_batch_visual_review_ledger.json",
    "artifact_checksums": "simulator_command_batch_artifact_checksums.json",
}

POLICY_MODALITIES = (
    "policy_api_endpoint",
    "docker_container",
    "recorded_action_trace",
    "high_level_skill_trace",
    "teleop_demo",
    "sim_controller_plugin",
)
DEFAULT_TEST_POLICY_ID = "blueprint_default_walk_to_target_test_policy"
DEFAULT_POLICY_API_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
DEFAULT_POLICY_API_MAX_REQUEST_BYTES = 4 * 1024 * 1024

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "robot_eval_execution_and_calibration_support",
    "repo_local_default": True,
    "external_policy_calls_allowed_only_with_runtime_gates": True,
    "robot_pov_generated_from_capture_task_scenario_context": True,
    "generated_robot_pov_is_support_artifact_not_raw_capture_evidence": True,
    "robot_policy_execution_proven": False,
    "simulator_execution_proven": False,
    "real_world_outcome_proven": False,
    "rank_fidelity_result_proven": False,
    "non_ranking_operational_claim_validated": False,
    "public_claim_upgrade_allowed": False,
}


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
    return str(value).strip().lower() in {"1", "true", "yes", "y", "passed", "success", "succeeded"}


def _safe_id(value: Any) -> str:
    text = _string(value).lower()
    return "".join(char if char.isalnum() else "_" for char in text).strip("_") or "unknown"


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence):
        return [_string(item) for item in value if _string(item)]
    return []


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = read_json_any(path)
    except Exception:
        return {}
    return _mapping(payload)


def _relative_to(root: Path, value: Any) -> str:
    path = value if isinstance(value, Path) else Path(str(value))
    try:
        return os.path.relpath(path.resolve(), root.resolve())
    except Exception:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_trace_base_dir(simulator_payload: Mapping[str, Any]) -> Path | None:
    artifact_paths = _mapping(simulator_payload.get("artifact_paths"))
    manifest_ref = _string(artifact_paths.get("batch_trace_package_manifest"))
    if not manifest_ref or "://" in manifest_ref:
        return None
    manifest_path = Path(manifest_ref)
    if manifest_path.is_file():
        return manifest_path.parent
    return None


def _copy_command_batch_trace_artifacts(
    *,
    job_dir: Path,
    trace_package: Mapping[str, Any],
    source_base_dir: Path | None,
) -> tuple[Dict[str, str], Dict[str, Any]]:
    source_artifact_paths = _mapping(trace_package.get("artifact_paths"))
    copied: Dict[str, str] = {}
    records: Dict[str, Any] = {}
    for artifact_key, job_name in BATCH_TRACE_ARTIFACT_JOB_NAMES.items():
        source_ref = _string(source_artifact_paths.get(artifact_key))
        if not source_ref:
            records[artifact_key] = {
                "status": "missing_source_ref",
                "source_ref": None,
                "job_artifact": job_name,
            }
            continue
        if "://" in source_ref:
            records[artifact_key] = {
                "status": "remote_source_not_copied",
                "source_ref": source_ref,
                "job_artifact": job_name,
            }
            continue
        source_path = Path(source_ref)
        if not source_path.is_absolute() and source_base_dir is not None:
            source_path = source_base_dir / source_path
        destination = job_dir / job_name
        if source_path.is_file():
            ensure_dir(destination.parent)
            if source_path.resolve() != destination.resolve():
                shutil.copyfile(source_path, destination)
            copied[artifact_key] = job_name
            records[artifact_key] = {
                "status": "copied",
                "source_ref": source_ref,
                "job_artifact": job_name,
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256_file(destination),
            }
        else:
            records[artifact_key] = {
                "status": "missing_source_file",
                "source_ref": source_ref,
                "resolved_source_path": str(source_path),
                "job_artifact": job_name,
            }
    return copied, records


def default_test_policy_package_from_request(job_request: Mapping[str, Any]) -> Dict[str, Any]:
    """Return policy-package modality gated Blueprint default test policies."""

    policy_package = _mapping(job_request.get("policy_package") or job_request.get("policyPackage"))
    raw = _mapping(
        job_request.get("default_test_policy")
        or job_request.get("defaultTestPolicy")
        or policy_package.get("default_test_policy")
        or policy_package.get("defaultTestPolicy")
    )
    use_default = _boolish(
        job_request.get("use_default_test_policy")
        or job_request.get("useDefaultTestPolicy")
        or policy_package.get("use_default_test_policy")
        or policy_package.get("useDefaultTestPolicy")
    )
    if raw and raw.get("enabled") is False:
        return {}
    if not raw and not use_default:
        return {}

    policy_kind = _string(raw.get("policy_kind") or raw.get("policyKind") or raw.get("kind"))
    manipulation_policy_kinds = {
        "mobile_manipulation_pick_carry_place",
        "default_mobile_manipulation",
        "pick_carry_place",
        "pick_carry_place_tote",
        "default_pick_carry_place_tote",
    }
    if policy_kind and policy_kind not in {
        "walk_to_target",
        "default_walk_to_target",
        *manipulation_policy_kinds,
    }:
        return {}
    if policy_kind in manipulation_policy_kinds:
        object_id = _string(raw.get("object_id") or raw.get("objectId"))
        object_class = _string(raw.get("object_class") or raw.get("objectClass")) or "tote"
        task_id = _string(raw.get("task_id") or raw.get("taskId")) or "mobile_pick_carry_place_tote"
        policy_id = _string(raw.get("policy_id") or raw.get("policyId")) or (
            "blueprint_default_phase_manipulation_policy"
        )
        return {
            "high_level_skill_trace": {
                "policy_id": policy_id,
                "policy_kind": "mobile_manipulation_pick_carry_place",
                "task_id": task_id,
                "object_id": object_id,
                "object_class": object_class,
                "blueprint_default_test_policy": True,
                "blockers": [] if object_id else ["default_manipulation_policy_object_id_missing"],
                "ordered_skill_sequence": [
                    {"skill_id": "navigate_to_object", "name": "navigate_to_object"},
                    {"skill_id": "pregrasp_stance", "name": "pregrasp_stance"},
                    {"skill_id": "reach", "name": "reach"},
                    {"skill_id": "close_grip", "name": "close_grip"},
                    {"skill_id": "lift", "name": "lift"},
                    {"skill_id": "verify_grasp", "name": "verify_grasp"},
                    {"skill_id": "carry_to_return_pose", "name": "carry_to_return_pose"},
                    {"skill_id": "place", "name": "place"},
                    {"skill_id": "release", "name": "release"},
                    {"skill_id": "verify_placement", "name": "verify_placement"},
                ],
                "claim_boundary": {
                    "default_test_policy_execution_contract": True,
                    "default_manipulation_policy": True,
                    "robot_team_policy_execution_proven": False,
                    "simulator_physics_execution_proven": False,
                    "grasp_physics_validated": False,
                    "generated_world_rank_fidelity_result_proven": False,
                    "public_claim_upgrade_allowed": False,
                },
            }
        }
    target = (
        _string(
            raw.get("target")
            or raw.get("target_pose_id")
            or raw.get("targetPoseId")
            or raw.get("goal_pose_id")
            or raw.get("goalPoseId")
        )
        or "walk_to_target_pose"
    )
    policy_id = _string(raw.get("policy_id") or raw.get("policyId")) or DEFAULT_TEST_POLICY_ID
    return {
        "high_level_skill_trace": {
            "policy_id": policy_id,
            "policy_kind": "walk_to_target",
            "target": target,
            "blueprint_default_test_policy": True,
            "ordered_skill_sequence": [
                {
                    "skill_id": "walk_to_target",
                    "name": "walk_to_target",
                    "target": target,
                }
            ],
            "claim_boundary": {
                "default_test_policy_execution_contract": True,
                "robot_team_policy_execution_proven": False,
                "robot_team_policy_quality_proven": False,
                "rank_fidelity_result_proven": False,
                "public_claim_upgrade_allowed": False,
            },
        }
    }


def _read_optional_any(path: Path) -> Any:
    try:
        return read_json_any(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _load_real_robot_pov_payload(
    *,
    capture_root: Path,
    job_dir: Path,
    job_request: Mapping[str, Any],
) -> tuple[Dict[str, Any], Path | None, str | None]:
    inline = job_request.get("real_robot_pov") or job_request.get("realRobotPov")
    if isinstance(inline, Mapping):
        return dict(inline), None, "job_request_inline_real_robot_pov"
    for ref in (
        job_request.get("real_robot_pov_manifest_uri"),
        job_request.get("realRobotPovManifestUri"),
    ):
        loaded = _load_reference_json(ref, capture_root=capture_root, job_dir=job_dir)
        if isinstance(loaded, Mapping):
            return dict(loaded), None, "job_request_real_robot_pov_manifest_ref"
    job_id = job_dir.name
    for path in (
        job_dir / "real_robot_pov_manifest.json",
        capture_root / "pipeline" / "robot_eval_inputs" / job_id / "real_robot_pov_manifest.json",
        capture_root / "pipeline" / "robot_eval_inputs" / "real_robot_pov_manifest.json",
    ):
        loaded = _read_optional_any(path)
        if isinstance(loaded, Mapping) and _records_from_payload(loaded):
            return dict(loaded), path, "capture_robot_eval_inputs_real_robot_pov_manifest"
    return {}, None, None


def _real_robot_pov_record_value(record: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = _string(record.get(key))
        if value:
            return value
    return ""


def _real_robot_pov_evidence(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "evidence_id": _real_robot_pov_record_value(record, "evidence_id", "evidenceId"),
        "robot_camera_video_uri": _real_robot_pov_record_value(
            record,
            "robot_camera_video_uri",
            "robotCameraVideoUri",
            "camera_video_uri",
            "video_uri",
        ),
        "action_log_uri": _real_robot_pov_record_value(
            record,
            "action_log_uri",
            "actionLogUri",
            "robot_action_log_uri",
        ),
        "robot_state_log_uri": _real_robot_pov_record_value(
            record,
            "robot_state_log_uri",
            "robotStateLogUri",
        ),
        "owner_evidence_refs": _mapping(
            record.get("owner_evidence_refs")
            or record.get("ownerEvidenceRefs")
            or record.get("evidence_refs")
            or record.get("evidenceRefs")
        ),
        "operator_attestation": record.get("operator_attestation")
        or record.get("operatorAttestation")
        or record.get("owner_attestation")
        or record.get("ownerAttestation"),
        "timestamp_alignment": _real_robot_pov_record_value(
            record,
            "timestamp_alignment",
            "timestampAlignment",
        ),
        "present": bool(record),
        "robot_pov_evidence_proven": bool(
            _real_robot_pov_record_value(
                record,
                "robot_camera_video_uri",
                "robotCameraVideoUri",
                "camera_video_uri",
                "video_uri",
            )
            and _real_robot_pov_record_value(
                record,
                "action_log_uri",
                "actionLogUri",
                "robot_action_log_uri",
            )
        ),
    }


def _real_robot_pov_index(
    records: Sequence[Mapping[str, Any]],
) -> Dict[tuple[str, str], Dict[str, Any]]:
    index: Dict[tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        task_id = _string(record.get("task_id") or record.get("taskId"))
        scenario_id = _string(record.get("scenario_id") or record.get("scenarioId"))
        run_id = _string(record.get("scenario_eval_run_id") or record.get("scenarioEvalRunId"))
        variation_id = _string(
            record.get("scenario_variation_instance_id")
            or record.get("scenarioVariationInstanceId")
        )
        if run_id and variation_id:
            index[(run_id, variation_id)] = dict(record)
        if run_id:
            index.setdefault((run_id, ""), dict(record))
        if task_id and scenario_id and variation_id:
            index.setdefault((f"{task_id}:{scenario_id}", variation_id), dict(record))
    return index


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    path.write_text(
        "\n".join(json.dumps(dict(row), sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _local_reference_path(
    value: Any,
    *,
    capture_root: Path,
    job_dir: Path,
) -> Path | None:
    text = _string(value)
    if not text:
        return None
    if text.startswith("file://"):
        return Path(text[7:]).expanduser()
    if text.startswith("gs://"):
        default_gcs_root = (
            capture_root.parents[3] if len(capture_root.parents) > 3 else capture_root
        )
        return resolve_gs_uri_to_path(text, Path(os.getenv("GCS_ROOT", str(default_gcs_root))))
    if "://" in text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    job_candidate = job_dir / path
    if job_candidate.exists():
        return job_candidate
    return capture_root / path


def _load_reference_json(
    value: Any,
    *,
    capture_root: Path,
    job_dir: Path,
) -> Any:
    path = _local_reference_path(value, capture_root=capture_root, job_dir=job_dir)
    if path is None or not path.is_file():
        return None
    return _read_optional_any(path)


def _cards_by_id(cards_payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in cards_payload.get("cards", []) or []:
        if isinstance(item, Mapping):
            scenario_id = _string(item.get("scenario_id") or item.get("task_id"))
            if scenario_id:
                out[scenario_id] = dict(item)
    return out


def _requested_scenarios(
    request: Mapping[str, Any],
    scenario_cards: Mapping[str, Any],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    requested_tasks = request.get("requested_tasks") or request.get("requestedTasks") or []
    explicit_requested_tasks = (
        ("requested_tasks" in request or "requestedTasks" in request)
        and isinstance(
            requested_tasks,
            Sequence,
        )
        and not isinstance(
            requested_tasks,
            (str, bytes, bytearray),
        )
    )
    for task in requested_tasks if explicit_requested_tasks else []:
        if not isinstance(task, Mapping):
            continue
        task_id = _string(task.get("task_id") or task.get("taskId"))
        scenario_ids = _string_list(task.get("scenario_ids") or task.get("scenarioIds"))
        if not scenario_ids:
            scenario_ids = [
                _string(card.get("scenario_id"))
                for card in scenario_cards.get("cards", []) or []
                if isinstance(card, Mapping) and _string(card.get("task_id")) == task_id
            ]
        for scenario_id in scenario_ids:
            rows.append({"task_id": task_id, "scenario_id": scenario_id})
    if explicit_requested_tasks:
        return [row for row in rows if row["scenario_id"]]
    for item in scenario_cards.get("cards", []) or []:
        if isinstance(item, Mapping):
            rows.append(
                {
                    "task_id": _string(item.get("task_id")),
                    "scenario_id": _string(item.get("scenario_id")),
                }
            )
    return [row for row in rows if row["scenario_id"]]


def _requested_scenario_eval_run_filters(request: Mapping[str, Any]) -> List[Dict[str, str]]:
    raw_filters = (
        request.get("requested_scenario_eval_runs")
        or request.get("requestedScenarioEvalRuns")
        or []
    )
    if isinstance(raw_filters, Mapping):
        raw_filters = [raw_filters]
    if not isinstance(raw_filters, Sequence) or isinstance(raw_filters, (str, bytes, bytearray)):
        return []
    filters: List[Dict[str, str]] = []
    for item in raw_filters:
        if not isinstance(item, Mapping):
            continue
        row = {
            "scenario_eval_run_id": _string(
                item.get("scenario_eval_run_id") or item.get("scenarioEvalRunId")
            ),
            "scenario_variation_instance_id": _string(
                item.get("scenario_variation_instance_id")
                or item.get("scenarioVariationInstanceId")
            ),
            "variation_name": _string(item.get("variation_name") or item.get("variationName")),
            "task_id": _string(item.get("task_id") or item.get("taskId")),
            "scenario_id": _string(item.get("scenario_id") or item.get("scenarioId")),
            "source_followup_action_id": _string(
                item.get("source_followup_action_id") or item.get("sourceFollowupActionId")
            ),
        }
        if any(row.values()):
            filters.append(row)
    return filters


def _run_matches_requested_filter(run: Mapping[str, Any], filter_row: Mapping[str, str]) -> bool:
    for field in (
        "task_id",
        "scenario_id",
        "scenario_eval_run_id",
        "scenario_variation_instance_id",
        "variation_name",
    ):
        expected = _string(filter_row.get(field))
        if expected and expected != _string(run.get(field)):
            return False
    return True


def _scenario_card_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cards = payload.get("cards")
    if not isinstance(cards, list):
        return []
    return [dict(card) for card in cards if isinstance(card, Mapping)]


def _load_scenario_variation_instances(capture_root: Path) -> Dict[str, Any]:
    return _read_optional_mapping(
        capture_root / "pipeline" / "simulation_automation" / "scenario_variation_instances.json"
    )


def _scenario_variation_rows_by_scenario(
    variation_payload: Mapping[str, Any],
) -> Dict[tuple[str, str], List[Dict[str, Any]]]:
    by_key: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for item in variation_payload.get("instances", []) or []:
        if not isinstance(item, Mapping):
            continue
        task_id = _string(item.get("task_id"))
        scenario_id = _string(item.get("scenario_id"))
        if not scenario_id:
            continue
        by_key.setdefault((task_id, scenario_id), []).append(dict(item))
    return by_key


def _scenario_eval_run_id(
    *,
    task_id: str,
    scenario_id: str,
    variation_name: str,
    index: int,
) -> str:
    return (
        f"{_safe_id(task_id or 'task')}_{_safe_id(scenario_id)}_"
        f"{_safe_id(variation_name or 'base')}_run_{index:04d}"
    )


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _requested_scenario_eval_run_target_count(
    job_request: Mapping[str, Any],
) -> tuple[int | None, str | None]:
    execution_request = _mapping(
        job_request.get("execution_request") or job_request.get("executionRequest")
    )
    containers: list[tuple[str, Mapping[str, Any]]] = [
        ("job_request", job_request),
        ("execution_request", execution_request),
        (
            "execution_request.scenario_batch",
            _mapping(
                execution_request.get("scenario_batch") or execution_request.get("scenarioBatch")
            ),
        ),
        (
            "execution_request.scenario_matrix",
            _mapping(
                execution_request.get("scenario_matrix") or execution_request.get("scenarioMatrix")
            ),
        ),
        (
            "execution_request.simulator_routing",
            _mapping(
                execution_request.get("simulator_routing")
                or execution_request.get("simulatorRouting")
            ),
        ),
    ]
    keys = (
        "target_scenario_eval_run_count",
        "targetScenarioEvalRunCount",
        "requested_scenario_eval_run_count",
        "requestedScenarioEvalRunCount",
        "scenario_eval_run_count",
        "scenarioEvalRunCount",
        "scenario_count",
        "scenarioCount",
        "arena_scenario_count",
        "arenaScenarioCount",
        "batch_size",
        "batchSize",
    )
    for source_name, container in containers:
        for key in keys:
            value = _positive_int(container.get(key))
            if value is not None:
                return value, f"{source_name}.{key}"
    return None, None


def _policy_candidate_rows(job_request: Mapping[str, Any]) -> list[Dict[str, Any]]:
    execution_request = _mapping(
        job_request.get("execution_request") or job_request.get("executionRequest")
    )
    wam_request = _mapping(
        job_request.get("wam_evaluation")
        or job_request.get("wamEvaluation")
        or execution_request.get("wam_evaluation")
        or execution_request.get("wamEvaluation")
    )
    raw = (
        job_request.get("policy_candidates")
        or job_request.get("policyCandidates")
        or job_request.get("policies")
        or job_request.get("checkpoints")
        or execution_request.get("policy_candidates")
        or execution_request.get("policyCandidates")
        or wam_request.get("policy_candidates")
        or wam_request.get("policyCandidates")
        or wam_request.get("policies")
        or wam_request.get("checkpoints")
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    candidates: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, start=1):
        payload = _mapping(item)
        policy_id = (
            _string(
                payload.get("policy_id")
                or payload.get("policyId")
                or payload.get("candidate_id")
                or payload.get("candidateId")
                or payload.get("id")
            )
            or f"policy_candidate_{index:02d}"
        )
        if policy_id in seen:
            continue
        seen.add(policy_id)
        candidates.append(
            {
                **payload,
                "policy_id": policy_id,
                "display_name": _string(payload.get("display_name") or payload.get("name"))
                or policy_id,
                "candidate_index": index,
                "observation_protocol_id": (
                    _string(
                        payload.get("observation_protocol_id")
                        or payload.get("observationProtocolId")
                    )
                    or POLICY_OBSERVATION_SCHEMA_ID
                ),
                "action_protocol_id": (
                    _string(payload.get("action_protocol_id") or payload.get("actionProtocolId"))
                    or POLICY_ACTION_SCHEMA_ID
                ),
            }
        )
    return candidates


def _policy_comparison_flag(job_request: Mapping[str, Any]) -> bool | None:
    execution_request = _mapping(
        job_request.get("execution_request") or job_request.get("executionRequest")
    )
    containers = (job_request, execution_request)
    keys = (
        "policy_comparison_mode",
        "policyComparisonMode",
        "compare_policies",
        "comparePolicies",
        "candidate_policy_comparison",
        "candidatePolicyComparison",
    )
    for container in containers:
        for key in keys:
            if key in container:
                return _boolish(container.get(key))
    return None


def _wam_matrix_policy_expansion_blocked(job_request: Mapping[str, Any]) -> bool:
    execution_request = _mapping(
        job_request.get("execution_request") or job_request.get("executionRequest")
    )
    substrate = (
        _string(
            job_request.get("evaluation_substrate")
            or job_request.get("evaluationSubstrate")
            or execution_request.get("evaluation_substrate")
            or execution_request.get("evaluationSubstrate")
        )
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    return substrate in {
        "fixture_wam",
        "wam_fixture",
        "local_wam",
        "cosmos3_wam",
        "cosmos_wam",
        "oscar_wam",
    }


def _policy_comparison_rows(
    runs: Sequence[Mapping[str, Any]],
    *,
    candidates: Sequence[Mapping[str, Any]],
    job_request: Mapping[str, Any],
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    flag = _policy_comparison_flag(job_request)
    requested = bool(flag) if flag is not None else len(candidates) >= 2
    disabled_for_wam = _wam_matrix_policy_expansion_blocked(job_request)
    candidate_ids = [_string(candidate.get("policy_id")) for candidate in candidates]
    base_run_ids = [
        _string(run.get("scenario_eval_run_id"))
        for run in runs
        if _string(run.get("scenario_eval_run_id"))
    ]
    if not requested or disabled_for_wam or len(candidates) < 2 or not runs:
        blockers: list[str] = []
        if requested and len(candidates) < 2:
            blockers.append("policy_comparison_requires_at_least_two_candidates")
        if requested and disabled_for_wam:
            blockers.append("wam_substrate_uses_internal_policy_candidate_loop")
        return [dict(run) for run in runs], {
            "enabled": False,
            "requested": requested,
            "disabled_for_wam_substrate": disabled_for_wam,
            "candidate_count": len(candidates),
            "candidate_ids": candidate_ids,
            "base_scenario_eval_run_count": len(base_run_ids),
            "base_scenario_eval_run_ids": base_run_ids,
            "expanded_scenario_eval_run_ids": base_run_ids,
            "blockers": blockers,
        }

    observation_protocol_id = POLICY_OBSERVATION_SCHEMA_ID
    action_protocol_id = POLICY_ACTION_SCHEMA_ID
    expanded: list[Dict[str, Any]] = []
    for base_index, run in enumerate(runs, start=1):
        base = dict(run)
        base_run_id = _string(base.get("base_scenario_eval_run_id")) or _string(
            base.get("scenario_eval_run_id")
        )
        for candidate_index, candidate in enumerate(candidates, start=1):
            policy_id = (
                _string(candidate.get("policy_id")) or f"policy_candidate_{candidate_index:02d}"
            )
            candidate_run_id = f"{base_run_id}__policy_{_safe_id(policy_id)}"
            expanded.append(
                {
                    **base,
                    "scenario_eval_run_id": candidate_run_id,
                    "base_scenario_eval_run_id": base_run_id,
                    "policy_id": policy_id,
                    "policy_candidate_id": policy_id,
                    "policy_candidate_display_name": _string(candidate.get("display_name"))
                    or policy_id,
                    "policy_candidate_index": candidate_index,
                    "policy_comparison_base_run_index": base_index,
                    "policy_comparison_candidate_run": True,
                    "policy_comparison_reference_only": bool(
                        candidate.get("reference_only") or candidate.get("referenceOnly")
                    ),
                    "observation_protocol_id": observation_protocol_id,
                    "action_protocol_id": action_protocol_id,
                    "policy_observation_protocol_id": observation_protocol_id,
                    "policy_action_protocol_id": action_protocol_id,
                    "same_observation_action_protocol_required": True,
                }
            )
    return expanded, {
        "enabled": True,
        "requested": requested,
        "disabled_for_wam_substrate": False,
        "candidate_count": len(candidates),
        "candidate_ids": candidate_ids,
        "base_scenario_eval_run_count": len(base_run_ids),
        "base_scenario_eval_run_ids": base_run_ids,
        "expanded_scenario_eval_run_ids": [
            _string(run.get("scenario_eval_run_id"))
            for run in expanded
            if _string(run.get("scenario_eval_run_id"))
        ],
        "observation_protocol_id": observation_protocol_id,
        "action_protocol_id": action_protocol_id,
        "same_observation_protocol_required": True,
        "same_action_protocol_required": True,
        "same_observation_action_protocol_required": True,
        "blockers": [],
    }


def _pose_triplet(value: Any) -> list[float] | None:
    if isinstance(value, Mapping):
        xyz = value.get("xyz") or value.get("pose") or value.get("position")
        nested = _pose_triplet(xyz)
        if nested is not None:
            return nested
        x = value.get("x")
        y = value.get("y")
        z = value.get("z", 0.793)
        try:
            return [round(float(x), 6), round(float(y), 6), round(float(z), 6)]
        except (TypeError, ValueError):
            return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) < 2:
            return None
        try:
            x = float(value[0])
            y = float(value[1])
            z = float(value[2]) if len(value) > 2 else 0.793
        except (TypeError, ValueError):
            return None
        return [round(x, 6), round(y, 6), round(z, 6)]
    return None


def _first_valid_candidate(candidates: Any) -> dict[str, Any] | None:
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes, bytearray)):
        return None
    for item in candidates:
        candidate = _mapping(item)
        pose = _pose_triplet(
            candidate.get("pose_xyz") or candidate.get("xyz") or candidate.get("pose")
        )
        if pose is None:
            continue
        if candidate.get("validated") is False:
            continue
        status = _string(candidate.get("validation_status"))
        if status and status.startswith("blocked"):
            continue
        return {**candidate, "pose_xyz": pose}
    return None


def _scenario_card_spawn_target_context(scenario_card: Mapping[str, Any] | None) -> dict[str, Any]:
    if not scenario_card:
        return {
            "validated_spawn_target_pair": False,
            "blockers": ["scenario_card_missing"],
        }
    spawn_candidate = _first_valid_candidate(scenario_card.get("spawn_candidates"))
    target_candidate = _first_valid_candidate(scenario_card.get("target_candidates"))
    spawn_pose = _pose_triplet(_mapping(spawn_candidate).get("pose_xyz"))
    target_pose = _pose_triplet(_mapping(target_candidate).get("pose_xyz"))
    blockers: list[str] = []
    if spawn_pose is None:
        blockers.append("scenario_card_validated_spawn_candidate_missing")
    if target_pose is None:
        blockers.append("scenario_card_validated_target_candidate_missing")
    pair_valid = not blockers
    return {
        "validated_spawn_target_pair": pair_valid,
        "spawn_pose": spawn_pose,
        "target_pose": target_pose,
        "spawn_candidate_id": _string(_mapping(spawn_candidate).get("zone_id")) or None,
        "target_candidate_id": _string(_mapping(target_candidate).get("zone_id")) or None,
        "spawn_candidate": spawn_candidate,
        "target_candidate": target_candidate,
        "source": "scenario_card_validated_site_zone_pair"
        if pair_valid
        else "missing_scenario_card_site_zone_pair",
        "blockers": blockers,
        "claim_boundary": "scenario-card spawn target candidates are finite site-coordinate eval inputs, not navigation safety proof",
    }


def _merge_semantic_spawn_target(
    run: Mapping[str, Any],
    scenario_card: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    out = dict(run)
    context = _scenario_card_spawn_target_context(scenario_card)
    concrete_mutation = _mapping(out.get("concrete_mutation"))
    if context["validated_spawn_target_pair"]:
        if _pose_triplet(concrete_mutation.get("spawn_pose")) is None:
            concrete_mutation["spawn_pose"] = context["spawn_pose"]
        if _pose_triplet(concrete_mutation.get("target_pose")) is None:
            concrete_mutation["target_pose"] = context["target_pose"]
        concrete_mutation.setdefault("spawn_candidate_id", context["spawn_candidate_id"])
        concrete_mutation.setdefault("target_candidate_id", context["target_candidate_id"])
    out["concrete_mutation"] = concrete_mutation
    out["semantic_spawn_target"] = context
    out["semantic_spawn_target_source"] = context["source"]
    out["semantic_spawn_target_validated"] = bool(context["validated_spawn_target_pair"])
    out["semantic_spawn_target_blockers"] = list(context.get("blockers") or [])
    return out


def _mutation_pose(run: Mapping[str, Any], *keys: str) -> tuple[list[float] | None, str | None]:
    concrete_mutation = _mapping(run.get("concrete_mutation"))
    for key in keys:
        pose = _pose_triplet(run.get(key))
        if pose is not None:
            return pose, key
        pose = _pose_triplet(concrete_mutation.get(key))
        if pose is not None:
            return pose, f"concrete_mutation.{key}"
    return None, None


def _stable_scenario_seed(run: Mapping[str, Any], *, ordinal: int, repeat_index: int) -> int:
    explicit = _positive_int(run.get("deterministic_seed") or run.get("episode_seed"))
    if explicit is not None:
        return explicit
    raw = ":".join(
        [
            _string(run.get("task_id")) or "task",
            _string(run.get("scenario_id")) or "scenario",
            _string(run.get("scenario_variation_instance_id")) or "base",
            _string(run.get("variation_name")) or "base_capture_layout",
            str(ordinal),
            str(repeat_index),
        ]
    )
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)


def _fallback_spawn_target(seed: int) -> tuple[list[float], list[float]]:
    start_x = round(((seed % 1600) - 800) / 100.0, 3)
    start_y = round((((seed >> 8) % 1600) - 800) / 100.0, 3)
    target_x = round(((((seed >> 16) % 1600) - 800) / 100.0), 3)
    target_y = round(((((seed >> 24) % 1600) - 800) / 100.0), 3)
    if abs(target_x - start_x) + abs(target_y - start_y) < 1.0:
        target_x = round(start_x + 2.0, 3)
        target_y = round(start_y - 2.0, 3)
    return [start_x, start_y, 0.793], [target_x, target_y, 0.793]


def _with_deterministic_scenario_fields(
    run: Mapping[str, Any],
    *,
    ordinal: int,
    repeat_index: int,
    batch_source_run_id: str | None = None,
) -> Dict[str, Any]:
    out = dict(run)
    seed = _stable_scenario_seed(out, ordinal=ordinal, repeat_index=repeat_index)
    fallback_spawn, fallback_target = _fallback_spawn_target(seed)
    spawn_pose, spawn_source = _mutation_pose(
        out, "spawn_pose", "start_pose", "robot_spawn_pose", "start_xyz", "spawn_xyz"
    )
    target_pose, target_source = _mutation_pose(
        out, "target_pose", "goal_pose", "robot_target_pose", "target_xyz", "goal_xyz"
    )
    semantic_spawn_target = _mapping(out.get("semantic_spawn_target"))
    semantic_validated = bool(semantic_spawn_target.get("validated_spawn_target_pair"))
    if spawn_pose is None:
        spawn_pose = fallback_spawn
        spawn_source = "deterministic_seed_fallback"
    if target_pose is None:
        target_pose = fallback_target
        target_source = "deterministic_seed_fallback"
    deterministic_fallback_used = (
        spawn_source == "deterministic_seed_fallback"
        or target_source == "deterministic_seed_fallback"
    )
    concrete_mutation = {
        **_mapping(out.get("concrete_mutation")),
        "spawn_pose": spawn_pose,
        "target_pose": target_pose,
        "deterministic_seed": seed,
    }
    out.update(
        {
            "episode_id": out.get("episode_id")
            or f"episode_{_safe_id(_string(out.get('scenario_eval_run_id')))}",
            "deterministic_seed": seed,
            "episode_seed": seed,
            "spawn_pose": spawn_pose,
            "target_pose": target_pose,
            "start_xyz": spawn_pose,
            "target_xyz": target_pose,
            "route_waypoints": out.get("route_waypoints") or [spawn_pose, target_pose],
            "concrete_mutation": concrete_mutation,
            "batch_ordinal": ordinal,
            "batch_repeat_index": repeat_index,
            "batch_source_scenario_eval_run_id": batch_source_run_id
            or _string(out.get("scenario_eval_run_id")),
            "spawn_goal_variation_seed_frozen": True,
            "validated_spawn_target_pair": bool(
                semantic_validated and not deterministic_fallback_used
            ),
            "semantic_spawn_target_source": out.get("semantic_spawn_target_source"),
            "deterministic_spawn_target_fallback_used": deterministic_fallback_used,
            "deterministic_scenario_parameters": {
                "schema_version": "deterministic_scenario_parameters.v1",
                "spawn_pose_source": spawn_source,
                "target_pose_source": target_source,
                "semantic_spawn_target_source": out.get("semantic_spawn_target_source"),
                "semantic_spawn_target_validated": bool(
                    semantic_validated and not deterministic_fallback_used
                ),
                "deterministic_fallback_used": deterministic_fallback_used,
                "semantic_spawn_target_blockers": list(
                    out.get("semantic_spawn_target_blockers") or []
                ),
                "deterministic_seed_source": "sha256_task_scenario_variation_ordinal",
                "runtime_spawn_goal_variation_mutation_allowed": False,
            },
        }
    )
    return out


def _expand_scenario_eval_runs_to_target_count(
    runs: Sequence[Mapping[str, Any]],
    *,
    target_count: int | None,
    exact_filters_requested: bool,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    if not runs:
        return [], {
            "expanded": False,
            "base_scenario_eval_run_count": 0,
            "target_scenario_eval_run_count": target_count,
            "target_scenario_eval_run_count_satisfied": False,
        }
    if exact_filters_requested or target_count is None or target_count <= len(runs):
        enriched = [
            _with_deterministic_scenario_fields(run, ordinal=index, repeat_index=0)
            for index, run in enumerate(runs, start=1)
        ]
        return enriched, {
            "expanded": False,
            "base_scenario_eval_run_count": len(runs),
            "target_scenario_eval_run_count": target_count or len(runs),
            "target_scenario_eval_run_count_satisfied": len(enriched)
            >= (target_count or len(runs)),
            "expansion_skipped_reason": "exact_run_filters_requested"
            if exact_filters_requested and target_count is not None
            else None,
        }
    base_runs = [dict(run) for run in runs]
    expanded: list[Dict[str, Any]] = []
    for ordinal in range(1, target_count + 1):
        base = dict(base_runs[(ordinal - 1) % len(base_runs)])
        repeat_index = (ordinal - 1) // len(base_runs)
        source_run_id = _string(base.get("scenario_eval_run_id"))
        if repeat_index:
            base["scenario_eval_run_id"] = _scenario_eval_run_id(
                task_id=_string(base.get("task_id")),
                scenario_id=_string(base.get("scenario_id")),
                variation_name=_string(base.get("variation_name")) or "base_capture_layout",
                index=ordinal,
            )
            base.pop("episode_id", None)
        expanded.append(
            _with_deterministic_scenario_fields(
                base,
                ordinal=ordinal,
                repeat_index=repeat_index,
                batch_source_run_id=source_run_id,
            )
        )
    return expanded, {
        "expanded": True,
        "base_scenario_eval_run_count": len(base_runs),
        "target_scenario_eval_run_count": target_count,
        "target_scenario_eval_run_count_satisfied": len(expanded) == target_count,
    }


def build_scenario_eval_matrix(
    *,
    capture_root: str | Path,
    job_dir: str | Path,
    job_request: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    """Expand requested task/scenario scope into concrete eval runs.

    Scenario-family variation instances are generated by the simulation
    automation lane. This job-level matrix makes those variation runs explicit
    for robot POV generation, policy adapters, simulator adapters, and coverage
    auditing. It is still only an execution contract; it is not simulator,
    policy, or real-world proof.
    """

    capture_path = Path(capture_root).resolve()
    resolved_job_dir = Path(job_dir).resolve()
    robot_eval_dir = capture_path / "pipeline" / "robot_eval_dataset"
    scenario_cards = _read_optional_mapping(robot_eval_dir / "scenario_cards.json")
    requested = _requested_scenarios(job_request, scenario_cards)
    requested_eval_run_filters = _requested_scenario_eval_run_filters(job_request)
    target_scenario_eval_run_count, target_scenario_eval_run_count_source = (
        _requested_scenario_eval_run_target_count(job_request)
    )
    scenario_card_rows = _scenario_card_rows(scenario_cards)
    scenario_cards_by_id = {
        _string(card.get("scenario_id")): card
        for card in scenario_card_rows
        if _string(card.get("scenario_id"))
    }
    scenario_card_task_ids = {
        _string(card.get("task_id")) for card in scenario_card_rows if _string(card.get("task_id"))
    }
    requested_task_ids = [
        _string(task.get("task_id") or task.get("taskId"))
        for task in (job_request.get("requested_tasks") or job_request.get("requestedTasks") or [])
        if isinstance(task, Mapping) and _string(task.get("task_id") or task.get("taskId"))
    ]
    variation_payload = _load_scenario_variation_instances(capture_path)
    variations_by_scenario = _scenario_variation_rows_by_scenario(variation_payload)

    runs: List[Dict[str, Any]] = []
    missing_variation_scenarios: List[str] = []
    invalid_requested_rows: List[Dict[str, Any]] = []
    unknown_requested_task_ids: List[str] = []
    unknown_requested_scenario_ids: List[str] = []
    requested_scenario_task_mismatches: List[Dict[str, str]] = []
    for task_id in requested_task_ids:
        if task_id not in scenario_card_task_ids and task_id not in unknown_requested_task_ids:
            unknown_requested_task_ids.append(task_id)
    for row in requested:
        task_id = row["task_id"]
        scenario_id = row["scenario_id"]
        scenario_card = scenario_cards_by_id.get(scenario_id)
        if not scenario_card:
            if scenario_id not in unknown_requested_scenario_ids:
                unknown_requested_scenario_ids.append(scenario_id)
            invalid_requested_rows.append(
                {
                    "task_id": task_id,
                    "scenario_id": scenario_id,
                    "reason": "scenario_id_not_in_scenario_cards",
                }
            )
            continue
        scenario_task_id = _string(scenario_card.get("task_id"))
        if task_id and scenario_task_id and task_id != scenario_task_id:
            mismatch = {
                "task_id": task_id,
                "scenario_id": scenario_id,
                "scenario_card_task_id": scenario_task_id,
            }
            requested_scenario_task_mismatches.append(mismatch)
            invalid_requested_rows.append(
                {
                    **mismatch,
                    "reason": "requested_task_id_mismatches_scenario_card",
                }
            )
            continue
    valid_requested = [
        row
        for row in requested
        if row["scenario_id"] in scenario_cards_by_id
        and (
            not row["task_id"]
            or _string(scenario_cards_by_id[row["scenario_id"]].get("task_id")) == row["task_id"]
        )
    ]
    for requested_index, row in enumerate(valid_requested, start=1):
        task_id = row["task_id"]
        scenario_id = row["scenario_id"]
        scenario_card = scenario_cards_by_id.get(scenario_id)
        variations = variations_by_scenario.get(
            (task_id, scenario_id)
        ) or variations_by_scenario.get(("", scenario_id))
        if not variations:
            missing_variation_scenarios.append(scenario_id)
            runs.append(
                _merge_semantic_spawn_target(
                    {
                        "scenario_eval_run_id": _scenario_eval_run_id(
                            task_id=task_id,
                            scenario_id=scenario_id,
                            variation_name="base_capture_layout",
                            index=requested_index,
                        ),
                        "task_id": task_id,
                        "scenario_id": scenario_id,
                        "scenario_variation_instance_id": None,
                        "variation_name": "base_capture_layout",
                        "baseline_capture_layout": True,
                        "concrete_mutation": {},
                        "engine_mutations": {},
                        "robot_pov_required": True,
                        "policy_attempt_required": True,
                        "simulator_rollout_required": True,
                        "review_required": True,
                        "claim_boundary": "base_scenario_eval_run_is_contract_not_execution_proof",
                    },
                    scenario_card,
                )
            )
            continue
        for variation_index, variation in enumerate(variations, start=1):
            variation_name = _string(variation.get("variation_name"))
            runs.append(
                _merge_semantic_spawn_target(
                    {
                        "scenario_eval_run_id": _scenario_eval_run_id(
                            task_id=task_id,
                            scenario_id=scenario_id,
                            variation_name=variation_name,
                            index=variation_index,
                        ),
                        "task_id": task_id,
                        "scenario_id": scenario_id,
                        "scenario_variation_instance_id": _string(variation.get("instance_id"))
                        or None,
                        "variation_name": variation_name,
                        "baseline_capture_layout": False,
                        "concrete_mutation": _mapping(variation.get("concrete_mutation")),
                        "engine_mutations": _mapping(variation.get("engine_mutations")),
                        "robot_pov_required": True,
                        "policy_attempt_required": True,
                        "simulator_rollout_required": True,
                        "review_required": True,
                        "claim_boundary": "scenario_variation_eval_run_is_contract_not_execution_proof",
                    },
                    scenario_card,
                )
            )

    unmatched_requested_eval_run_filters: List[Dict[str, str]] = []
    if requested_eval_run_filters:
        filtered_runs: List[Dict[str, Any]] = []
        seen_run_ids: set[str] = set()
        for filter_row in requested_eval_run_filters:
            matches = [run for run in runs if _run_matches_requested_filter(run, filter_row)]
            if not matches:
                unmatched_requested_eval_run_filters.append(dict(filter_row))
                continue
            for match in matches:
                run_id = _string(match.get("scenario_eval_run_id"))
                if run_id in seen_run_ids:
                    continue
                seen_run_ids.add(run_id)
                filtered_runs.append(
                    {
                        **dict(match),
                        "requested_scenario_eval_run_filter": {
                            key: value for key, value in filter_row.items() if value
                        },
                    }
                )
        runs = filtered_runs
    runs, batch_expansion = _expand_scenario_eval_runs_to_target_count(
        runs,
        target_count=target_scenario_eval_run_count,
        exact_filters_requested=bool(requested_eval_run_filters),
    )
    policy_candidates = _policy_candidate_rows(job_request)
    runs, policy_comparison_expansion = _policy_comparison_rows(
        runs,
        candidates=policy_candidates,
        job_request=job_request,
    )

    required_names = _string_list(variation_payload.get("required_variation_names"))
    covered_names = sorted(
        {
            _string(run.get("variation_name"))
            for run in runs
            if _string(run.get("variation_name")) and not run.get("baseline_capture_layout")
        }
    )
    missing_required = sorted(set(required_names) - set(covered_names)) if required_names else []
    fallback_spawn_target_run_ids = [
        _string(run.get("scenario_eval_run_id"))
        for run in runs
        if run.get("deterministic_spawn_target_fallback_used") is True
        and _string(run.get("scenario_eval_run_id"))
    ]
    missing_semantic_spawn_target_run_ids = [
        _string(run.get("scenario_eval_run_id"))
        for run in runs
        if run.get("validated_spawn_target_pair") is not True
        and _string(run.get("scenario_eval_run_id"))
    ]
    semantic_spawn_target_coverage_complete = (
        bool(runs) and not missing_semantic_spawn_target_run_ids
    )
    matrix_blockers: List[str] = []
    if unknown_requested_task_ids:
        matrix_blockers.append("scenario_eval_matrix_unknown_requested_tasks")
    if unknown_requested_scenario_ids:
        matrix_blockers.append("scenario_eval_matrix_unknown_requested_scenarios")
    if requested_scenario_task_mismatches:
        matrix_blockers.append("scenario_eval_matrix_requested_task_scenario_mismatch")
    if not runs:
        matrix_blockers.append("scenario_eval_matrix_missing_requested_scenarios")
    if unmatched_requested_eval_run_filters:
        matrix_blockers.append("scenario_eval_matrix_unknown_requested_eval_runs")
    if missing_semantic_spawn_target_run_ids:
        matrix_blockers.append("scenario_eval_matrix_semantic_spawn_target_missing")
    if matrix_blockers:
        matrix_status = "blocked_invalid_requested_scope"
    else:
        matrix_status = "completed"
    manifest = {
        "schema_version": SCENARIO_EVAL_MATRIX_SCHEMA_VERSION,
        "generated_at": generated_at,
        "capture_root": str(capture_path),
        "job_dir": str(resolved_job_dir),
        "status": matrix_status,
        "blockers": matrix_blockers,
        "requested_scenario_count": len(requested),
        "valid_requested_scenario_count": len(valid_requested),
        "invalid_requested_scenario_count": len(invalid_requested_rows),
        "requested_scenario_eval_run_filter_count": len(requested_eval_run_filters),
        "requested_scenario_eval_run_filters": requested_eval_run_filters,
        "target_scenario_eval_run_count": batch_expansion["target_scenario_eval_run_count"],
        "target_scenario_eval_run_count_source": target_scenario_eval_run_count_source,
        "base_scenario_eval_run_count": batch_expansion["base_scenario_eval_run_count"],
        "scenario_eval_batch_expanded": batch_expansion["expanded"],
        "target_scenario_eval_run_count_satisfied": batch_expansion[
            "target_scenario_eval_run_count_satisfied"
        ],
        "scenario_eval_batch_expansion": batch_expansion,
        "policy_comparison_mode": bool(policy_comparison_expansion.get("enabled")),
        "policy_comparison_requested": bool(policy_comparison_expansion.get("requested")),
        "policy_comparison_candidate_count": int(
            policy_comparison_expansion.get("candidate_count") or 0
        ),
        "policy_comparison_candidate_ids": _string_list(
            policy_comparison_expansion.get("candidate_ids")
        ),
        "policy_comparison_base_scenario_eval_run_count": int(
            policy_comparison_expansion.get("base_scenario_eval_run_count") or 0
        ),
        "policy_comparison_base_scenario_eval_run_ids": _string_list(
            policy_comparison_expansion.get("base_scenario_eval_run_ids")
        ),
        "policy_comparison_expanded_scenario_eval_run_ids": _string_list(
            policy_comparison_expansion.get("expanded_scenario_eval_run_ids")
        ),
        "policy_comparison_observation_protocol_id": policy_comparison_expansion.get(
            "observation_protocol_id"
        ),
        "policy_comparison_action_protocol_id": policy_comparison_expansion.get(
            "action_protocol_id"
        ),
        "policy_comparison_same_observation_protocol_required": bool(
            policy_comparison_expansion.get("same_observation_protocol_required")
        ),
        "policy_comparison_same_action_protocol_required": bool(
            policy_comparison_expansion.get("same_action_protocol_required")
        ),
        "policy_comparison_same_observation_action_protocol_required": bool(
            policy_comparison_expansion.get("same_observation_action_protocol_required")
        ),
        "policy_comparison_expansion": policy_comparison_expansion,
        "unmatched_requested_scenario_eval_run_filter_count": len(
            unmatched_requested_eval_run_filters
        ),
        "unmatched_requested_scenario_eval_run_filters": (unmatched_requested_eval_run_filters),
        "scenario_eval_run_count": len(runs),
        "variation_instance_count": int(variation_payload.get("instance_count") or 0),
        "required_variation_names": required_names,
        "variation_names_covered": covered_names,
        "missing_required_variation_names": missing_required,
        "missing_variation_scenarios": missing_variation_scenarios,
        "semantic_spawn_target_coverage_complete": semantic_spawn_target_coverage_complete,
        "deterministic_fallback_spawn_target_run_count": len(fallback_spawn_target_run_ids),
        "fallback_spawn_target_run_ids": fallback_spawn_target_run_ids,
        "missing_semantic_spawn_target_run_ids": missing_semantic_spawn_target_run_ids,
        "unknown_requested_task_ids": unknown_requested_task_ids,
        "unknown_requested_scenario_ids": unknown_requested_scenario_ids,
        "requested_scenario_task_mismatches": requested_scenario_task_mismatches,
        "invalid_requested_rows": invalid_requested_rows,
        "runs": runs,
        "source_artifacts": {
            "scenario_cards": "../robot_eval_dataset/scenario_cards.json"
            if scenario_cards
            else None,
            "scenario_variation_instances": "../simulation_automation/scenario_variation_instances.json"
            if variation_payload
            else None,
        },
        "episode_authoring_contract": {
            "schema_version": "scenario_eval_episode_authoring_contract.v1",
            "spawn_target_variation_seed_handling": "deterministic_frozen_matrix_rows",
            "runtime_spawn_goal_variation_mutation_allowed": False,
            "scenario_eval_run_id_exact_coverage_required": True,
            "semantic_spawn_target_coverage_required": True,
            "semantic_spawn_target_coverage_complete": semantic_spawn_target_coverage_complete,
            "deterministic_spawn_target_fallback_allowed_for_beta_release": False,
            "target_scenario_eval_run_count": batch_expansion["target_scenario_eval_run_count"],
            "target_scenario_eval_run_count_satisfied": batch_expansion[
                "target_scenario_eval_run_count_satisfied"
            ],
        },
        "robot_pov_generation_required_for_each_run": True,
        "policy_attempt_required_for_each_run": True,
        "simulator_rollout_required_for_each_run": True,
        "simulator_execution_proven": False,
        "robot_policy_execution_proven": False,
        "rank_fidelity_result_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    manifest["deterministic_fingerprint"] = hashlib.sha256(
        json.dumps({"runs": runs}, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    write_json(resolved_job_dir / "scenario_eval_matrix.json", manifest)
    return manifest


def _attempt_video_index(attempt_trace: Mapping[str, Any]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for attempt in attempt_trace.get("attempts", []) or []:
        if not isinstance(attempt, Mapping):
            continue
        scenario_id = _string(attempt.get("scenario_id"))
        video_path = _string(
            attempt.get("video_path")
            or _mapping(attempt.get("artifact_paths")).get("robot_pov_video")
            or _mapping(attempt.get("artifact_paths")).get("video")
        )
        if scenario_id and video_path and scenario_id not in index:
            index[scenario_id] = video_path
    return index


def _requested_eval_runs(
    *,
    requested: Sequence[Mapping[str, str]],
    scenario_eval_matrix: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    matrix_runs = [
        dict(item)
        for item in scenario_eval_matrix.get("runs", []) or []
        if isinstance(item, Mapping)
    ]
    if matrix_runs:
        return matrix_runs
    return [
        {
            "scenario_eval_run_id": _scenario_eval_run_id(
                task_id=_string(row.get("task_id")),
                scenario_id=_string(row.get("scenario_id")),
                variation_name="base_capture_layout",
                index=index,
            ),
            "task_id": _string(row.get("task_id")),
            "scenario_id": _string(row.get("scenario_id")),
            "scenario_variation_instance_id": None,
            "variation_name": "base_capture_layout",
            "baseline_capture_layout": True,
            "concrete_mutation": {},
            "engine_mutations": {},
        }
        for index, row in enumerate(requested, start=1)
    ]


def _write_observation_png(path: Path, lines: Sequence[str]) -> bool:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    ensure_dir(path.parent)
    image = Image.new("RGB", (960, 540), (24, 29, 35))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 936, 516), outline=(90, 160, 220), width=3)
    draw.rectangle((64, 330, 896, 460), outline=(175, 190, 205), width=2)
    draw.line((480, 120, 480, 465), fill=(210, 210, 130), width=2)
    draw.text((64, 56), "Blueprint robot POV observation", fill=(245, 248, 250))
    y = 104
    for line in lines[:10]:
        draw.text((64, y), line[:110], fill=(220, 230, 238))
        y += 34
    image.save(path)
    return True


def _write_robot_pov_frame_sequence(
    *,
    frame_dir: Path,
    observation_id: str,
    lines: Sequence[str],
) -> List[Dict[str, Any]]:
    phases = [
        (
            "approach",
            [
                "phase: approach",
                "camera_motion: base moves toward target approach region",
                "risk_focus: path clearance, human/forklift proximity, occlusion",
            ],
        ),
        (
            "inspect",
            [
                "phase: inspect",
                "camera_motion: front_rgbd centers target and nearby distractors",
                "risk_focus: missing label, glare, wrong object nearby",
            ],
        ),
        (
            "act",
            [
                "phase: act",
                "camera_motion: target interaction and recovery envelope",
                "risk_focus: object drop, collision risk, timeout recovery",
            ],
        ),
    ]
    frames: List[Dict[str, Any]] = []
    for index, (phase, phase_lines) in enumerate(phases, start=1):
        path = frame_dir / observation_id / f"{index:03d}_{phase}.png"
        written = _write_observation_png(
            path,
            [
                f"observation_id: {observation_id}",
                *phase_lines,
                *lines,
            ],
        )
        if written:
            frames.append(
                {
                    "frame_index": index,
                    "phase": phase,
                    "path": path,
                    "camera_state": {
                        "frame": "site_coordinate_frame",
                        "pose_source": "deterministic_storyboard_default",
                        "synthetic_local_render": True,
                    },
                }
            )
    return frames


def build_robot_pov_observation_bundle(
    *,
    capture_root: str | Path,
    job_dir: str | Path,
    job_request: Mapping[str, Any],
    generated_at: str,
    attempt_trace: Mapping[str, Any] | None = None,
    scenario_eval_matrix: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build deterministic robot-POV observation packets for every requested scenario."""

    capture_path = Path(capture_root).resolve()
    resolved_job_dir = Path(job_dir).resolve()
    robot_eval_dir = capture_path / "pipeline" / "robot_eval_dataset"
    scenario_cards = _read_optional_mapping(robot_eval_dir / "scenario_cards.json")
    task_cards = _read_optional_mapping(robot_eval_dir / "task_cards.json")
    episode_spec = _read_optional_mapping(
        capture_path / "pipeline" / "simulation_automation" / "episode_spec.v1.json"
    )
    scenarios_by_id = _cards_by_id(scenario_cards)
    tasks_by_id = _cards_by_id(task_cards)
    requested = _requested_scenarios(job_request, scenario_cards)
    scenario_eval_matrix_payload = scenario_eval_matrix or _read_optional_mapping(
        resolved_job_dir / "scenario_eval_matrix.json"
    )
    eval_runs = _requested_eval_runs(
        requested=requested,
        scenario_eval_matrix=scenario_eval_matrix_payload,
    )
    robot_profile = _mapping(job_request.get("robot_profile") or job_request.get("robotProfile"))
    robot_profile_id = _string(robot_profile.get("robot_profile_id") or robot_profile.get("id"))
    video_index = _attempt_video_index(attempt_trace or {})
    real_pov_payload, real_pov_manifest_path, real_pov_source = _load_real_robot_pov_payload(
        capture_root=capture_path,
        job_dir=resolved_job_dir,
        job_request=job_request,
    )
    real_pov_records = _records_from_payload(real_pov_payload)
    real_pov_index = _real_robot_pov_index(real_pov_records)

    observations: List[Dict[str, Any]] = []
    frame_sequences: List[Dict[str, Any]] = []
    storyboards: List[Dict[str, Any]] = []
    frame_dir = resolved_job_dir / "robot_pov"
    for index, row in enumerate(eval_runs, start=1):
        scenario_id = _string(row.get("scenario_id"))
        task_id = _string(row.get("task_id"))
        scenario_eval_run_id = _string(row.get("scenario_eval_run_id")) or _scenario_eval_run_id(
            task_id=task_id,
            scenario_id=scenario_id,
            variation_name=_string(row.get("variation_name")) or "base_capture_layout",
            index=index,
        )
        variation_name = _string(row.get("variation_name")) or "base_capture_layout"
        variation_instance_id = _string(row.get("scenario_variation_instance_id")) or None
        scenario = scenarios_by_id.get(scenario_id, {})
        task = tasks_by_id.get(task_id, {})
        observation_id = (
            f"robot_pov_{_safe_id(task_id)}_{_safe_id(scenario_id)}_{_safe_id(variation_name)}"
        )
        frame_path = frame_dir / f"{observation_id}.png"
        lines = [
            f"task_id: {task_id or 'unknown'}",
            f"scenario_id: {scenario_id}",
            f"scenario_eval_run_id: {scenario_eval_run_id}",
            f"variation: {variation_name}",
            f"robot_profile_id: {robot_profile_id or 'unknown'}",
            f"task: {_string(task.get('task_statement')) or 'from task card'}",
            f"normal: {_string(_mapping(scenario.get('normal_scenario')).get('statement'))}",
            f"variation: {_string(_mapping(scenario.get('variation')).get('statement'))}",
            f"edge_case: {_string(_mapping(scenario.get('edge_case')).get('statement'))}",
        ]
        frame_written = _write_observation_png(frame_path, lines)
        sequence_frames = _write_robot_pov_frame_sequence(
            frame_dir=frame_dir,
            observation_id=observation_id,
            lines=lines,
        )
        relative_sequence_paths = [
            _relative_to(resolved_job_dir, _mapping(frame).get("path"))
            for frame in sequence_frames
            if isinstance(_mapping(frame).get("path"), Path)
        ]
        sequence_id = f"robot_pov_sequence_{_safe_id(observation_id)}"
        storyboard_id = f"robot_pov_storyboard_{_safe_id(observation_id)}"
        real_record = (
            real_pov_index.get((scenario_eval_run_id, variation_instance_id or ""))
            or real_pov_index.get((scenario_eval_run_id, ""))
            or real_pov_index.get((f"{task_id}:{scenario_id}", variation_instance_id or ""))
            or {}
        )
        real_evidence = _real_robot_pov_evidence(real_record)
        frame_sequences.append(
            {
                "sequence_id": sequence_id,
                "observation_id": observation_id,
                "task_id": task_id,
                "scenario_id": scenario_id,
                "scenario_eval_run_id": scenario_eval_run_id,
                "scenario_variation_instance_id": variation_instance_id,
                "variation_name": variation_name,
                "frame_count": len(relative_sequence_paths),
                "frame_paths": relative_sequence_paths,
                "phases": [frame.get("phase") for frame in sequence_frames],
                "local_render_generated": bool(relative_sequence_paths),
                "robot_pov_evidence_proven": False,
                "claim_boundary": "generated_frame_sequence_not_raw_robot_camera_video",
            }
        )
        storyboards.append(
            {
                "storyboard_id": storyboard_id,
                "observation_id": observation_id,
                "sequence_id": sequence_id,
                "task_id": task_id,
                "scenario_id": scenario_id,
                "scenario_eval_run_id": scenario_eval_run_id,
                "scenario_variation_instance_id": variation_instance_id,
                "variation_name": variation_name,
                "frames": [
                    {
                        "frame_index": frame.get("frame_index"),
                        "phase": frame.get("phase"),
                        "frame_path": _relative_to(resolved_job_dir, _mapping(frame).get("path"))
                        if isinstance(_mapping(frame).get("path"), Path)
                        else None,
                        "camera_state": frame.get("camera_state"),
                    }
                    for frame in sequence_frames
                ],
                "render_intent": "local_robot_pov_support_storyboard_for_policy_adapter_inputs",
                "robot_pov_evidence_proven": False,
                "claim_boundary": "storyboard_is_generated_support_artifact_not_real_robot_video",
            }
        )
        observations.append(
            {
                "observation_id": observation_id,
                "task_id": task_id,
                "scenario_id": scenario_id,
                "scenario_eval_run_id": scenario_eval_run_id,
                "scenario_variation_instance_id": variation_instance_id,
                "variation_name": variation_name,
                "concrete_mutation": _mapping(row.get("concrete_mutation")),
                "engine_mutations": _mapping(row.get("engine_mutations")),
                "robot_profile_id": robot_profile_id or None,
                "sequence_index": index,
                "camera": {
                    "name": "front_rgbd",
                    "frame": "site_coordinate_frame",
                    "resolution": {"width": 960, "height": 540},
                    "horizontal_fov_degrees": 75.0,
                    "mount": "robot_front",
                    "extrinsics_source": "episode_spec_or_deterministic_default",
                },
                "observation_schema": {
                    "schema_id": POLICY_OBSERVATION_SCHEMA_ID,
                    "schema_ref": POLICY_OBSERVATION_SCHEMA_REF,
                    "rgb": "image/png",
                    "depth": "optional_depth_map_or_missing",
                    "robot_state": ["base_pose", "joint_state_optional", "gripper_state_optional"],
                    "task_instruction": "task_card.task_statement",
                },
                "expected_action_schema": {
                    "schema_id": POLICY_ACTION_SCHEMA_ID,
                    "schema_ref": POLICY_ACTION_SCHEMA_REF,
                    "required_fields": [
                        "scenario_eval_run_id",
                        "scenario_variation_instance_id",
                        "task_id",
                        "scenario_id",
                        "status",
                        "success",
                        "actions",
                        "metrics",
                        "failure_mode_ids",
                    ],
                },
                "generated_frame_path": _relative_to(resolved_job_dir, frame_path)
                if frame_written
                else None,
                "render_sequence_id": sequence_id,
                "render_frame_paths": relative_sequence_paths,
                "render_storyboard_id": storyboard_id,
                "real_robot_pov_evidence": real_evidence,
                "sim_or_real_video_path": real_evidence.get("robot_camera_video_uri")
                or video_index.get(scenario_id),
                "source_artifacts": {
                    "scenario_card": "pipeline/robot_eval_dataset/scenario_cards.json",
                    "task_card": "pipeline/robot_eval_dataset/task_cards.json",
                    "episode_spec": "pipeline/simulation_automation/episode_spec.v1.json"
                    if episode_spec
                    else None,
                    "scenario_eval_matrix": "pipeline/robot_eval_jobs/"
                    f"{resolved_job_dir.name}/scenario_eval_matrix.json",
                },
                "claim_boundary": (
                    "generated_robot_pov_observation_packet_not_raw_robot_camera_evidence"
                ),
            }
        )

    required_scenario_eval_run_ids = sorted(
        {
            _string(item.get("scenario_eval_run_id"))
            for item in observations
            if _string(item.get("scenario_eval_run_id"))
        }
    )
    real_robot_pov_covered_scenario_eval_run_ids = sorted(
        {
            _string(item.get("scenario_eval_run_id"))
            for item in observations
            if bool(_mapping(item.get("real_robot_pov_evidence")).get("robot_pov_evidence_proven"))
            and _string(item.get("scenario_eval_run_id"))
        }
    )
    missing_real_robot_pov_scenario_eval_run_ids = [
        run_id
        for run_id in required_scenario_eval_run_ids
        if run_id not in real_robot_pov_covered_scenario_eval_run_ids
    ]
    robot_pov_evidence_proven = (
        bool(observations) and not missing_real_robot_pov_scenario_eval_run_ids
    )
    initial_observation_resolution = build_initial_observation_source_resolution(
        capture_root=capture_path,
        job_dir=resolved_job_dir,
        job_request=job_request,
        generated_at=generated_at,
        scenario_cards=scenario_cards,
        task_cards=task_cards,
        scenario_eval_matrix=scenario_eval_matrix_payload,
        observations=observations,
    )
    initial_candidate_set = _mapping(initial_observation_resolution.get("candidate_set"))
    selected_initial_observation = _mapping(
        initial_observation_resolution.get("selected_initial_policy_observation")
    )
    camera_profile_registry = _mapping(initial_candidate_set.get("camera_profile_registry"))
    camera_profile_launch_readiness = _mapping(
        initial_candidate_set.get("camera_profile_launch_readiness")
    )
    manifest = {
        "schema_version": ROBOT_POV_OBSERVATION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if observations else "blocked_missing_scenarios",
        "capture_root": str(capture_path),
        "job_dir": str(resolved_job_dir),
        "observation_count": len(observations),
        "scenario_eval_run_count": len(eval_runs),
        "scenario_eval_matrix_path": "scenario_eval_matrix.json"
        if (resolved_job_dir / "scenario_eval_matrix.json").is_file()
        else None,
        "local_render_sequence_count": len(frame_sequences),
        "local_render_frame_count": sum(
            int(sequence.get("frame_count") or 0) for sequence in frame_sequences
        ),
        "policy_adapter_input_contract": {
            "schema_version": "robot_policy_adapter_input_contract.v1",
            "observation_schema_id": POLICY_OBSERVATION_SCHEMA_ID,
            "observation_schema_ref": POLICY_OBSERVATION_SCHEMA_REF,
            "action_schema_id": POLICY_ACTION_SCHEMA_ID,
            "action_schema_ref": POLICY_ACTION_SCHEMA_REF,
            "scenario_eval_run_id_exact_coverage_required": True,
            "scenario_variation_instance_id_exact_coverage_required": True,
            "runtime_spawn_goal_variation_mutation_allowed": False,
        },
        "robot_profile": robot_profile,
        "observations": observations,
        "initial_observation_source_resolver": {
            "candidate_set_path": "robot_pov_observation_candidate_set.json",
            "selected_initial_policy_observation_path": (
                "selected_initial_policy_observation.json"
            ),
            "camera_profile_registry_path": "robot_camera_profile_registry.json",
            "camera_profile_launch_readiness_path": ("robot_camera_profile_launch_readiness.json"),
            "owner_robot_camera_calibration_request_path": (
                "owner_robot_camera_calibration_request.json"
            ),
            "camera_profile_count": camera_profile_registry.get("profile_count"),
            "camera_profile_launch_readiness_status": camera_profile_launch_readiness.get("status"),
            "camera_profile_ready_for_launch": camera_profile_launch_readiness.get(
                "ready_for_launch"
            ),
            "camera_profile_launch_ready_profile_count": camera_profile_launch_readiness.get(
                "launch_ready_profile_count"
            ),
            "camera_profile_smoke_only_profile_count": camera_profile_launch_readiness.get(
                "smoke_only_profile_count"
            ),
            "candidate_count": initial_candidate_set.get("candidate_count"),
            "selected_candidate_id": initial_candidate_set.get("selected_candidate_id"),
            "selected_source_kind": initial_candidate_set.get("selected_source_kind"),
            "selected_status": selected_initial_observation.get("status"),
            "source_qa_path": "initial_policy_observation_source_qa.json",
            "contact_sheet_path": "initial_policy_observation_contact_sheet.jpg",
            "recapture_guidance_path": "initial_policy_observation_recapture_guidance.json",
            "paid_provider_calls_performed": False,
        },
        "robot_camera_profile_launch_readiness": {
            "path": "robot_camera_profile_launch_readiness.json",
            "status": camera_profile_launch_readiness.get("status"),
            "launch_mode": camera_profile_launch_readiness.get("launch_mode"),
            "ready_for_launch": camera_profile_launch_readiness.get("ready_for_launch"),
            "all_profiles_launch_ready": camera_profile_launch_readiness.get(
                "all_profiles_launch_ready"
            ),
            "owner_calibration_request_packet_path": camera_profile_launch_readiness.get(
                "owner_calibration_request_packet_path"
            ),
            "blockers": camera_profile_launch_readiness.get("blockers") or [],
        },
        "robot_pov_generated": bool(observations),
        "generated_robot_pov_support_available": bool(observations),
        "real_robot_pov_manifest_path": _relative_to(
            resolved_job_dir.parent,
            real_pov_manifest_path,
        )
        if real_pov_manifest_path
        else None,
        "real_robot_pov_source": real_pov_source,
        "real_robot_pov_evidence_record_count": len(real_pov_records),
        "real_robot_pov_action_log_record_count": sum(
            1
            for record in real_pov_records
            if _real_robot_pov_record_value(
                record,
                "action_log_uri",
                "actionLogUri",
                "robot_action_log_uri",
            )
        ),
        "real_robot_pov_covered_scenario_eval_run_ids": (
            real_robot_pov_covered_scenario_eval_run_ids
        ),
        "missing_real_robot_pov_scenario_eval_run_ids": (
            missing_real_robot_pov_scenario_eval_run_ids
        ),
        "sim_or_real_robot_pov_video_available": any(
            _string(item.get("sim_or_real_video_path")) for item in observations
        ),
        "robot_pov_evidence_proven": robot_pov_evidence_proven,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    frame_sequence_manifest = {
        "schema_version": ROBOT_POV_FRAME_SEQUENCE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if frame_sequences else "blocked_missing_scenarios_or_renderer",
        "capture_root": str(capture_path),
        "job_dir": str(resolved_job_dir),
        "sequence_count": len(frame_sequences),
        "total_frame_count": sum(
            int(sequence.get("frame_count") or 0) for sequence in frame_sequences
        ),
        "sequences": frame_sequences,
        "local_robot_pov_render_generated": any(
            bool(sequence.get("local_render_generated")) for sequence in frame_sequences
        ),
        "robot_pov_evidence_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    storyboard_manifest = {
        "schema_version": ROBOT_POV_RENDER_STORYBOARD_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if storyboards else "blocked_missing_scenarios_or_renderer",
        "capture_root": str(capture_path),
        "job_dir": str(resolved_job_dir),
        "storyboard_count": len(storyboards),
        "storyboards": storyboards,
        "local_robot_pov_render_generated": any(
            bool(storyboard.get("frames")) for storyboard in storyboards
        ),
        "robot_pov_evidence_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(resolved_job_dir / "robot_pov_observation_manifest.json", manifest)
    write_json(resolved_job_dir / "robot_pov_frame_sequence_manifest.json", frame_sequence_manifest)
    write_json(resolved_job_dir / "robot_pov_render_storyboard.json", storyboard_manifest)
    _write_jsonl(resolved_job_dir / "robot_pov_observations.jsonl", observations)
    return manifest


def _modality_payload(policy_package: Mapping[str, Any], modality: str) -> Dict[str, Any]:
    camel = {
        "policy_api_endpoint": "policyApiEndpoint",
        "docker_container": "dockerContainer",
        "recorded_action_trace": "recordedActionTrace",
        "high_level_skill_trace": "highLevelSkillTrace",
        "teleop_demo": "teleopDemo",
        "sim_controller_plugin": "simControllerPlugin",
    }[modality]
    return _mapping(policy_package.get(modality) or policy_package.get(camel))


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if any(
                marker in key_text.lower() for marker in ("token", "secret", "password", "auth")
            ):
                out[key_text] = "<redacted>"
            else:
                out[key_text] = _redact(child)
        return out
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _command_from_payload(
    *,
    modality: str,
    payload: Mapping[str, Any],
    commands: Mapping[str, str],
) -> str:
    return _string(
        commands.get(modality)
        or payload.get("execution_command")
        or payload.get("executionCommand")
        or payload.get("adapter_command")
        or payload.get("adapterCommand")
    )


def _normalize_policy_attempts(
    *,
    payload: Any,
    modality: str,
    observations: Sequence[Mapping[str, Any]],
    generated_at: str,
) -> List[Dict[str, Any]]:
    raw_attempts: List[Mapping[str, Any]] = []
    if isinstance(payload, Mapping):
        for key in ("attempts", "actions", "skills", "episodes", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                raw_attempts.extend(item for item in value if isinstance(item, Mapping))
        if not raw_attempts and payload:
            raw_attempts = [payload]
    elif isinstance(payload, list):
        raw_attempts = [item for item in payload if isinstance(item, Mapping)]

    if not raw_attempts and modality == "high_level_skill_trace":
        raw_attempts = [{"status": "completed", "actions": []}]

    attempts: List[Dict[str, Any]] = []
    if not observations:
        observations = [{"observation_id": "observation_1", "scenario_id": "", "task_id": ""}]
    if modality == "high_level_skill_trace" and len(raw_attempts) == 1 and len(observations) > 1:
        only = raw_attempts[0]
        has_explicit_scope = any(
            _string(only.get(key))
            for key in (
                "observation_id",
                "scenario_id",
                "scenarioId",
                "scenario_eval_run_id",
                "scenarioEvalRunId",
                "scenario_variation_instance_id",
                "scenarioVariationInstanceId",
            )
        )
        if not has_explicit_scope:
            raw_attempts = [dict(only) for _ in observations]
    for index, raw in enumerate(raw_attempts or [{}], start=1):
        observation = observations[(index - 1) % len(observations)]
        status = _string(raw.get("status") or raw.get("result") or "completed").lower()
        success_raw = raw.get("success")
        success = (
            _boolish(success_raw)
            if success_raw is not None
            else status
            in {
                "completed",
                "success",
                "succeeded",
                "passed",
            }
        )
        attempts.append(
            {
                "attempt_id": _string(raw.get("attempt_id") or raw.get("attemptId"))
                or f"{modality}_attempt_{index:04d}",
                "modality": modality,
                "observation_id": _string(raw.get("observation_id"))
                or _string(observation.get("observation_id")),
                "scenario_id": _string(raw.get("scenario_id") or raw.get("scenarioId"))
                or _string(observation.get("scenario_id")),
                "scenario_eval_run_id": _string(
                    raw.get("scenario_eval_run_id") or raw.get("scenarioEvalRunId")
                )
                or _string(observation.get("scenario_eval_run_id")),
                "scenario_variation_instance_id": _string(
                    raw.get("scenario_variation_instance_id")
                    or raw.get("scenarioVariationInstanceId")
                )
                or _string(observation.get("scenario_variation_instance_id"))
                or None,
                "variation_name": _string(raw.get("variation_name") or raw.get("variationName"))
                or _string(observation.get("variation_name"))
                or None,
                "task_id": _string(raw.get("task_id") or raw.get("taskId"))
                or _string(observation.get("task_id")),
                "policy_id": _string(raw.get("policy_id") or raw.get("policyId") or modality),
                "policy_kind": _string(raw.get("policy_kind") or raw.get("policyKind")) or None,
                "policy_scope": _string(raw.get("policy_scope") or raw.get("policyScope"))
                or "robot_team_policy",
                "target": _string(raw.get("target") or raw.get("targetPoseId")) or None,
                "status": status,
                "success": bool(success),
                "actions": raw.get("actions") if isinstance(raw.get("actions"), list) else [],
                "skills": raw.get("skills") if isinstance(raw.get("skills"), list) else [],
                "metrics": _mapping(raw.get("metrics")),
                "artifact_paths": _mapping(raw.get("artifact_paths") or raw.get("artifactPaths")),
                "generated_at": generated_at,
                "claim_boundary": "policy_submission_trace_not_rank_fidelity_proof",
            }
        )
    return attempts


def _policy_run_coverage(
    attempts: Sequence[Mapping[str, Any]],
    required_run_ids: Sequence[str],
) -> Dict[str, Any]:
    required = [run_id for run_id in required_run_ids if run_id]
    covered = sorted(
        {
            _string(attempt.get("scenario_eval_run_id") or attempt.get("scenarioEvalRunId"))
            for attempt in attempts
            if _string(attempt.get("scenario_eval_run_id") or attempt.get("scenarioEvalRunId"))
        }
    )
    missing = sorted(set(required) - set(covered)) if required else []
    return {
        "required_scenario_eval_run_count": len(required),
        "covered_scenario_eval_run_count": len(covered),
        "missing_scenario_eval_run_count": len(missing),
        "covered_scenario_eval_run_ids": covered,
        "missing_scenario_eval_run_ids": missing,
        "scenario_eval_run_coverage_complete": bool(required) and not missing,
    }


def _run_command(
    *,
    command_text: str,
    output_path: Path,
    observation_manifest_path: Path,
    modality: str,
    timeout_seconds: int,
) -> tuple[str, Any, Dict[str, Any]]:
    command = shlex.split(command_text)
    ensure_dir(output_path.parent)
    env = {
        **os.environ,
        "BLUEPRINT_POLICY_OBSERVATION_MANIFEST": str(observation_manifest_path),
        "BLUEPRINT_POLICY_EXECUTION_OUTPUT": str(output_path),
        "BLUEPRINT_POLICY_MODALITY": modality,
    }
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except FileNotFoundError:
        return "blocked", None, {"blockers": ["missing_policy_command_dependency"]}
    except subprocess.TimeoutExpired as exc:
        return "failed", None, {"blockers": ["policy_command_timeout"], "stdout": exc.stdout or ""}
    payload = _read_optional_any(output_path) if output_path.is_file() else None
    if payload is None and completed.stdout.strip().startswith(("{", "[")):
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = None
    status = "completed" if completed.returncode == 0 and payload is not None else "failed"
    detail = {
        "command": command,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "exit_code": completed.returncode,
        "blockers": []
        if status == "completed"
        else [f"policy_command_exit:{completed.returncode}"],
    }
    return status, payload, detail


def _call_policy_api(
    *,
    endpoint: str,
    observation_manifest: Mapping[str, Any],
    timeout_seconds: int,
) -> tuple[str, Any, Dict[str, Any]]:
    data = json.dumps({"observation_manifest": observation_manifest}).encode("utf-8")
    if len(data) > DEFAULT_POLICY_API_MAX_REQUEST_BYTES:
        return "failed", None, {"blockers": ["policy_api_request_too_large"]}
    try:
        response = fetch_bounded_https(
            endpoint,
            method="POST",
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout_seconds=max(1, min(int(timeout_seconds), 30)),
            max_bytes=DEFAULT_POLICY_API_MAX_RESPONSE_BYTES,
            allowed_origins=origins_from_env("BLUEPRINT_POLICY_ENDPOINT_ALLOWED_ORIGINS"),
            allowed_content_types=("application/json",),
            max_redirects=1,
        )
        payload = json.loads(response.body.decode("utf-8"))
        if not json_shape_within_limits(payload, max_depth=32, max_items=100_000):
            raise SecurityValidationError("policy API JSON exceeds shape limits")
        return "completed", payload, {"http_status": response.status, "blockers": []}
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        return (
            "failed",
            None,
            {
                "blockers": ["policy_api_call_failed"],
                "error_type": type(exc).__name__,
            },
        )


def _docker_command(payload: Mapping[str, Any]) -> str:
    image = _string(payload.get("image_ref") or payload.get("imageRef"))
    entrypoint = _string(payload.get("entrypoint"))
    if not image:
        return ""
    base = f"docker run --rm -i {shlex.quote(image)}"
    return f"{base} {entrypoint}" if entrypoint else base


def _replay_reference_payload(
    *,
    modality: str,
    payload: Mapping[str, Any],
    capture_root: Path,
    job_dir: Path,
) -> Any:
    keys = {
        "recorded_action_trace": ("trace_manifest_uri", "traceManifestUri"),
        "teleop_demo": ("demo_artifact_uri", "demoArtifactUri"),
        "sim_controller_plugin": ("plugin_uri", "pluginUri"),
        "policy_api_endpoint": (
            "response_manifest_uri",
            "responseManifestUri",
            "local_response_path",
        ),
        "docker_container": ("output_manifest_uri", "outputManifestUri", "local_output_path"),
    }.get(modality, ())
    for key in keys:
        loaded = _load_reference_json(payload.get(key), capture_root=capture_root, job_dir=job_dir)
        if loaded is not None:
            return loaded
    if modality == "high_level_skill_trace":
        sequence = (
            payload.get("ordered_skill_sequence") or payload.get("orderedSkillSequence") or []
        )
        return {"attempts": [{"status": "completed", "skills": list(sequence), "success": True}]}
    return None


def _is_default_test_policy_payload(modality: str, payload: Mapping[str, Any]) -> bool:
    return modality == "high_level_skill_trace" and bool(
        payload.get("blueprint_default_test_policy")
        or payload.get("blueprintDefaultTestPolicy")
        or _string(payload.get("policy_id") or payload.get("policyId")) == DEFAULT_TEST_POLICY_ID
    )


def _default_test_policy_execution_payload(
    *,
    payload: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    policy_kind = (
        _string(payload.get("policy_kind") or payload.get("policyKind")) or "walk_to_target"
    )
    policy_id = (
        _string(payload.get("policy_id") or payload.get("policyId")) or DEFAULT_TEST_POLICY_ID
    )
    if policy_kind == "mobile_manipulation_pick_carry_place":
        task_id = _string(payload.get("task_id") or payload.get("taskId")) or (
            "mobile_pick_carry_place_tote"
        )
        object_id = _string(payload.get("object_id") or payload.get("objectId"))
        object_class = _string(payload.get("object_class") or payload.get("objectClass")) or "tote"
        if not object_id:
            return {
                "schema_version": "blueprint_default_test_policy_execution.v1",
                "status": "blocked_missing_manipulation_object_id",
                "policy_id": policy_id,
                "policy_kind": "mobile_manipulation_pick_carry_place",
                "task_id": task_id,
                "object_id": "",
                "object_class": object_class,
                "attempts": [],
                "blockers": ["default_manipulation_policy_object_id_missing"],
                "claim_boundary": {
                    "default_test_policy_execution_proven": False,
                    "default_manipulation_policy": True,
                    "robot_team_policy_execution_proven": False,
                    "robot_team_policy_quality_proven": False,
                    "simulator_physics_execution_proven": False,
                    "grasp_physics_validated": False,
                    "rank_fidelity_result_proven": False,
                    "public_claim_upgrade_allowed": False,
                },
            }
        raw_phases = payload.get("ordered_skill_sequence")
        phases = (
            [dict(item) for item in raw_phases if isinstance(item, Mapping)]
            if isinstance(raw_phases, list)
            else []
        )
        if not phases:
            phases = [
                {"skill_id": name, "name": name}
                for name in (
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
                )
            ]
        attempts: List[Dict[str, Any]] = []
        for index, observation in enumerate(observations, start=1):
            scenario_eval_run_id = _string(observation.get("scenario_eval_run_id"))
            actions = [
                {
                    "action": _string(phase.get("skill_id") or phase.get("name")),
                    "target": {"object_id": object_id, "object_class": object_class},
                    "status": "completed",
                    "evidence_scope": "job_default_manipulation_policy_reference_trace",
                }
                for phase in phases
            ]
            attempts.append(
                {
                    "attempt_id": f"{policy_id}_{_safe_id(scenario_eval_run_id or str(index))}",
                    "observation_id": _string(observation.get("observation_id")),
                    "scenario_id": _string(observation.get("scenario_id")),
                    "scenario_eval_run_id": scenario_eval_run_id,
                    "scenario_variation_instance_id": _string(
                        observation.get("scenario_variation_instance_id")
                    )
                    or None,
                    "variation_name": _string(observation.get("variation_name")) or None,
                    "task_id": _string(observation.get("task_id")) or task_id,
                    "policy_id": policy_id,
                    "policy_scope": "blueprint_default_test_policy",
                    "policy_kind": "mobile_manipulation_pick_carry_place",
                    "target": object_id,
                    "status": "completed",
                    "success": True,
                    "actions": actions,
                    "skills": [
                        {
                            "skill_id": _string(phase.get("skill_id") or phase.get("name")),
                            "name": _string(phase.get("name") or phase.get("skill_id")),
                            "target": object_id,
                            "status": "completed",
                        }
                        for phase in phases
                    ],
                    "metrics": {
                        "default_test_policy": True,
                        "default_manipulation_policy": True,
                        "phase_count": len(phases),
                        "completed_phase_count": len(phases),
                        "reference_trace_only": True,
                        "simulator_physics_execution_proven": False,
                        "grasp_physics_validated": False,
                    },
                }
            )
        return {
            "schema_version": "blueprint_default_test_policy_execution.v1",
            "status": "completed" if attempts else "blocked_missing_observations",
            "policy_id": policy_id,
            "policy_kind": "mobile_manipulation_pick_carry_place",
            "task_id": task_id,
            "object_id": object_id,
            "object_class": object_class,
            "attempts": attempts,
            "claim_boundary": {
                "default_test_policy_execution_proven": bool(attempts),
                "default_manipulation_policy": True,
                "robot_team_policy_execution_proven": False,
                "robot_team_policy_quality_proven": False,
                "simulator_physics_execution_proven": False,
                "grasp_physics_validated": False,
                "rank_fidelity_result_proven": False,
                "public_claim_upgrade_allowed": False,
            },
        }

    target = (
        _string(
            payload.get("target")
            or payload.get("target_pose_id")
            or payload.get("targetPoseId")
            or payload.get("goal_pose_id")
            or payload.get("goalPoseId")
        )
        or "walk_to_target_pose"
    )
    attempts: List[Dict[str, Any]] = []
    for index, observation in enumerate(observations, start=1):
        scenario_eval_run_id = _string(observation.get("scenario_eval_run_id"))
        attempts.append(
            {
                "attempt_id": f"{policy_id}_{_safe_id(scenario_eval_run_id or str(index))}",
                "observation_id": _string(observation.get("observation_id")),
                "scenario_id": _string(observation.get("scenario_id")),
                "scenario_eval_run_id": scenario_eval_run_id,
                "scenario_variation_instance_id": _string(
                    observation.get("scenario_variation_instance_id")
                )
                or None,
                "variation_name": _string(observation.get("variation_name")) or None,
                "task_id": _string(observation.get("task_id")),
                "policy_id": policy_id,
                "policy_scope": "blueprint_default_test_policy",
                "policy_kind": "walk_to_target",
                "target": target,
                "status": "completed",
                "success": True,
                "actions": [
                    {
                        "action": "walk_to_target",
                        "target": target,
                        "status": "completed",
                        "evidence_scope": "job_default_test_policy",
                    }
                ],
                "skills": [
                    {
                        "skill_id": "walk_to_target",
                        "name": "walk_to_target",
                        "target": target,
                        "status": "completed",
                    }
                ],
                "metrics": {
                    "default_test_policy": True,
                    "walk_to_target_completed": True,
                },
            }
        )
    return {
        "schema_version": "blueprint_default_test_policy_execution.v1",
        "status": "completed" if attempts else "blocked_missing_observations",
        "policy_id": policy_id,
        "policy_kind": "walk_to_target",
        "target": target,
        "attempts": attempts,
        "claim_boundary": {
            "default_test_policy_execution_proven": bool(attempts),
            "robot_team_policy_execution_proven": False,
            "robot_team_policy_quality_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def build_policy_execution_bundle(
    *,
    capture_root: str | Path,
    job_dir: str | Path,
    job_request: Mapping[str, Any],
    observation_manifest: Mapping[str, Any],
    allow_policy_execution: bool = False,
    allow_reference_replay: bool = True,
    policy_execution_commands: Mapping[str, str] | None = None,
    timeout_seconds: int = 120,
    generated_at: str,
) -> Dict[str, Any]:
    """Execute or replay robot-team policy submissions into normalized traces."""

    capture_path = Path(capture_root).resolve()
    resolved_job_dir = Path(job_dir).resolve()
    policy_package = _mapping(job_request.get("policy_package") or job_request.get("policyPackage"))
    for modality, payload in default_test_policy_package_from_request(job_request).items():
        policy_package.setdefault(modality, payload)
    commands = dict(policy_execution_commands or {})
    env_allows = _boolish(os.getenv("BLUEPRINT_ALLOW_POLICY_EXECUTION"))
    observation_manifest_path = resolved_job_dir / "robot_pov_observation_manifest.json"
    observations = [
        dict(item)
        for item in observation_manifest.get("observations", []) or []
        if isinstance(item, Mapping)
    ]
    required_run_ids = [
        _string(observation.get("scenario_eval_run_id"))
        for observation in observations
        if _string(observation.get("scenario_eval_run_id"))
    ]
    modality_results: Dict[str, Dict[str, Any]] = {}
    all_attempts: List[Dict[str, Any]] = []

    for modality in POLICY_MODALITIES:
        payload = _modality_payload(policy_package, modality)
        if not payload:
            modality_results[modality] = {
                "status": "not_selected",
                "execution_performed": False,
                "attempt_count": 0,
                "missing_inputs": [],
                "claim_boundary": dict(CLAIM_BOUNDARY),
            }
            continue
        command_text = _command_from_payload(
            modality=modality,
            payload=payload,
            commands=commands,
        )
        default_test_policy = _is_default_test_policy_payload(modality, payload)
        if (
            not command_text
            and modality == "docker_container"
            and allow_policy_execution
            and env_allows
        ):
            command_text = _docker_command(payload)

        payload_result: Any = None
        detail: Dict[str, Any] = {}
        execution_performed = False
        if default_test_policy and not command_text:
            if not allow_policy_execution or not env_allows:
                modality_results[modality] = {
                    "status": "blocked_policy_execution_gate",
                    "execution_performed": False,
                    "attempt_count": 0,
                    "reference": _redact(payload),
                    "launch_reviewable_without_execution": True,
                    "default_test_policy": True,
                    "default_test_policy_execution_proven": False,
                    "robot_team_policy_execution_proven": False,
                    "blockers": [
                        "Set BLUEPRINT_ALLOW_POLICY_EXECUTION=true and pass allow_policy_execution.",
                    ],
                    "claim_boundary": {
                        **dict(CLAIM_BOUNDARY),
                        "default_test_policy_execution_proven": False,
                        "robot_team_policy_execution_proven": False,
                        "reviewable_policy_adapter_pack_is_not_execution_proof": True,
                    },
                }
                continue
            payload_result = _default_test_policy_execution_payload(
                payload=payload,
                observations=observations,
            )
            status = (
                "completed"
                if payload_result.get("status") == "completed"
                else "blocked_missing_policy_execution_trace"
            )
            payload_blockers = [
                _string(item) for item in payload_result.get("blockers", []) if _string(item)
            ]
            detail = {
                "adapter": "blueprint_default_walk_to_target_policy",
                "blockers": []
                if status == "completed"
                else payload_blockers or ["default_policy_observations_missing"],
            }
            execution_performed = True
        elif command_text:
            if not allow_policy_execution or not env_allows:
                modality_results[modality] = {
                    "status": "blocked_policy_execution_gate",
                    "execution_performed": False,
                    "attempt_count": 0,
                    "reference": _redact(payload),
                    "launch_reviewable_without_execution": True,
                    "blockers": [
                        "Set BLUEPRINT_ALLOW_POLICY_EXECUTION=true and pass allow_policy_execution.",
                    ],
                    "claim_boundary": {
                        **dict(CLAIM_BOUNDARY),
                        "reviewable_policy_adapter_pack_is_not_execution_proof": True,
                    },
                }
                continue
            output_path = resolved_job_dir / "policy_execution" / f"{modality}_output.json"
            status, payload_result, detail = _run_command(
                command_text=command_text,
                output_path=output_path,
                observation_manifest_path=observation_manifest_path,
                modality=modality,
                timeout_seconds=timeout_seconds,
            )
            execution_performed = True
        elif modality == "policy_api_endpoint" and allow_policy_execution and env_allows:
            endpoint = _string(
                payload.get("endpoint_url") or payload.get("endpointUrl") or payload.get("url")
            )
            status, payload_result, detail = _call_policy_api(
                endpoint=endpoint,
                observation_manifest=observation_manifest,
                timeout_seconds=timeout_seconds,
            )
            execution_performed = True
        elif not allow_reference_replay:
            status = "blocked_reference_replay_gate"
            payload_result = None
            detail = {"blockers": ["reference_replay_disabled_by_validation_or_rights_gate"]}
        else:
            payload_result = _replay_reference_payload(
                modality=modality,
                payload=payload,
                capture_root=capture_path,
                job_dir=resolved_job_dir,
            )
            status = (
                "completed_reference_replay" if payload_result is not None else "reference_ready"
            )
            detail = {
                "blockers": [] if payload_result is not None else ["local_reference_not_available"]
            }

        attempts = (
            _normalize_policy_attempts(
                payload=payload_result,
                modality=modality,
                observations=observations,
                generated_at=generated_at,
            )
            if payload_result is not None
            else []
        )
        all_attempts.extend(attempts)
        coverage = _policy_run_coverage(attempts, required_run_ids)
        execution_proven_for_modality = execution_performed and status == "completed"
        modality_results[modality] = {
            "status": status,
            "execution_performed": execution_performed,
            "reference_replayed": payload_result is not None and not execution_performed,
            "launch_reviewable_without_execution": bool(
                not execution_performed
                and payload
                and status
                in {
                    "reference_ready",
                    "completed_reference_replay",
                    "blocked_policy_execution_gate",
                }
            ),
            "default_test_policy": default_test_policy,
            "default_test_policy_execution_proven": (
                default_test_policy and execution_proven_for_modality
            ),
            "robot_team_policy_execution_proven": (
                execution_proven_for_modality and not default_test_policy
            ),
            "attempt_count": len(attempts),
            **coverage,
            "reference": _redact(payload),
            "detail": detail,
            "robot_policy_execution_proven": execution_proven_for_modality,
            "policy_submission_trace_available": bool(attempts),
            "claim_boundary": {
                **dict(CLAIM_BOUNDARY),
                "robot_policy_execution_proven": execution_proven_for_modality,
                "default_test_policy_execution_proven": (
                    default_test_policy and execution_proven_for_modality
                ),
                "robot_team_policy_execution_proven": (
                    execution_proven_for_modality and not default_test_policy
                ),
                "reviewable_policy_adapter_pack_is_not_execution_proof": True,
            },
        }

    execution_proven = any(
        bool(item.get("robot_policy_execution_proven")) for item in modality_results.values()
    )
    default_test_execution_proven = any(
        bool(item.get("default_test_policy_execution_proven")) for item in modality_results.values()
    )
    robot_team_policy_execution_proven = any(
        bool(item.get("robot_team_policy_execution_proven")) for item in modality_results.values()
    )
    aggregate_coverage = _policy_run_coverage(all_attempts, required_run_ids)
    trace = {
        "schema_version": POLICY_EXECUTION_TRACE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if all_attempts else "blocked_missing_policy_execution_trace",
        "attempt_count": len(all_attempts),
        "attempts": all_attempts,
        **aggregate_coverage,
        "robot_policy_execution_proven": execution_proven,
        "default_test_policy_execution_proven": default_test_execution_proven,
        "robot_team_policy_execution_proven": robot_team_policy_execution_proven,
        "rank_fidelity_result_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "robot_policy_execution_proven": execution_proven,
            "default_test_policy_execution_proven": default_test_execution_proven,
            "robot_team_policy_execution_proven": robot_team_policy_execution_proven,
        },
    }
    manifest = {
        "schema_version": POLICY_EXECUTION_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if all_attempts else "blocked",
        "selected_modalities": [
            modality
            for modality, result in modality_results.items()
            if result.get("status") != "not_selected"
        ],
        "env_BLUEPRINT_ALLOW_POLICY_EXECUTION": env_allows,
        "allow_policy_execution_flag": bool(allow_policy_execution),
        "modality_results": modality_results,
        "attempt_count": len(all_attempts),
        **aggregate_coverage,
        "policy_execution_trace_path": "policy_execution_trace.json",
        "reviewable_policy_adapter_modes": [
            modality
            for modality, result in modality_results.items()
            if result.get("launch_reviewable_without_execution")
        ],
        "policy_adapter_pack_contract": {
            "schema_version": "robot_team_policy_adapter_pack_contract.v1",
            "same_observation_action_contract_for_all_modes": True,
            "supported_modalities": list(POLICY_MODALITIES),
            "selected_modalities": [
                modality
                for modality, result in modality_results.items()
                if result.get("status") != "not_selected"
            ],
            "reviewable_policy_adapter_modes": [
                modality
                for modality, result in modality_results.items()
                if result.get("launch_reviewable_without_execution")
            ],
            "execution_claim_requires_gated_policy_execution": True,
        },
        "robot_policy_execution_proven": execution_proven,
        "default_test_policy_execution_proven": default_test_execution_proven,
        "robot_team_policy_execution_proven": robot_team_policy_execution_proven,
        "rank_fidelity_result_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "robot_policy_execution_proven": execution_proven,
            "default_test_policy_execution_proven": default_test_execution_proven,
            "robot_team_policy_execution_proven": robot_team_policy_execution_proven,
        },
    }
    write_json(resolved_job_dir / "policy_execution_manifest.json", manifest)
    write_json(resolved_job_dir / "policy_execution_trace.json", trace)
    _write_jsonl(resolved_job_dir / "policy_execution_trace.jsonl", all_attempts)
    return {"manifest": manifest, "trace": trace}


def _records_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, Mapping):
        for key in (
            "records",
            "attempts",
            "episodes",
            "results",
            "actual_outcomes",
            "actualOutcomes",
            "outcomes",
            "pilot_runs",
            "runs",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, Mapping)]
        if payload:
            return [dict(payload)]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    return []


def _task_success_label_provenance(
    *,
    record: Mapping[str, Any],
    task_outcome: Mapping[str, Any],
    simulator: str,
) -> Dict[str, Any]:
    explicit = _mapping(
        record.get("task_success_label_provenance")
        or record.get("success_label_provenance")
        or task_outcome.get("task_success_label_provenance")
        or task_outcome.get("success_label_provenance")
    )
    if explicit:
        return {
            "schema_version": "task_success_label_provenance.v1",
            "provenance_type": _string(explicit.get("provenance_type"))
            or _string(explicit.get("type"))
            or "declared_by_simulator_output",
            "label_source": _string(explicit.get("label_source") or explicit.get("source"))
            or "declared_by_simulator_output",
            "evidence_artifact_type": _string(explicit.get("evidence_artifact_type"))
            or _string(explicit.get("evidence_type"))
            or "declared",
            "buyer_disclosure": _string(explicit.get("buyer_disclosure"))
            or "Task success uses the provenance declared by the simulator output.",
            "generated_video_vlm_judge": bool(explicit.get("generated_video_vlm_judge")),
            "simulator_physics_or_trace": bool(explicit.get("simulator_physics_or_trace")),
            "real_world_outcome": bool(explicit.get("real_world_outcome")),
            "success_label_disclosed_to_buyer": True,
            "public_claim_upgrade_allowed": False,
        }

    success_label = _mapping(
        record.get("wam_success_label")
        or record.get("generated_video_success_label")
        or task_outcome.get("wam_success_label")
        or task_outcome.get("generated_video_success_label")
    )
    label_source = (
        _string(record.get("success_label_source"))
        or _string(record.get("label_source"))
        or _string(task_outcome.get("success_label_source"))
        or _string(task_outcome.get("label_source"))
        or _string(success_label.get("label_source"))
        or _string(success_label.get("labeler"))
    )
    generated_video = bool(
        record.get("wam_success_label_from_generated_video")
        or record.get("success_label_from_generated_video")
        or record.get("generated_video_success_label_passed") is not None
        or task_outcome.get("wam_success_label_from_generated_video")
        or task_outcome.get("success_label_from_generated_video")
        or success_label.get("success_label_from_generated_video")
        or success_label.get("wam_success_label_from_generated_video")
        or "generated_video" in label_source
        or "video_frame_judge" in label_source
        or "vlm" in label_source
    )
    artifact_paths = _mapping(record.get("artifact_paths") or record.get("artifactPaths"))
    has_trace_or_physics = bool(
        task_outcome.get("goal_reached") is not None
        or task_outcome.get("final_target_error_m") is not None
        or task_outcome.get("min_clearance_m") is not None
        or record.get("contact_trace")
        or record.get("actions")
        or artifact_paths.get("scene_trace")
        or artifact_paths.get("policy_trace")
    )
    if generated_video:
        return {
            "schema_version": "task_success_label_provenance.v1",
            "provenance_type": "generated_video_vlm_judge",
            "label_source": label_source or "generated_video_success_label",
            "evidence_artifact_type": "model_derived_generated_video",
            "buyer_disclosure": (
                "Success labels are semantic judgments over model-derived generated "
                "rollout video, not measured physical robot success or simulator "
                "contact-state proof."
            ),
            "generated_video_vlm_judge": True,
            "simulator_physics_or_trace": False,
            "real_world_outcome": False,
            "success_label_disclosed_to_buyer": True,
            "public_claim_upgrade_allowed": False,
        }
    if has_trace_or_physics:
        return {
            "schema_version": "task_success_label_provenance.v1",
            "provenance_type": "simulator_trace_or_physics",
            "label_source": label_source or f"{simulator}_command_output",
            "evidence_artifact_type": "simulator_state_contact_or_route_trace",
            "buyer_disclosure": (
                "Success labels are derived from simulator trace/state outputs; "
                "they are not physical robot or live-site success proof."
            ),
            "generated_video_vlm_judge": False,
            "simulator_physics_or_trace": True,
            "real_world_outcome": False,
            "success_label_disclosed_to_buyer": True,
            "public_claim_upgrade_allowed": False,
        }
    return {
        "schema_version": "task_success_label_provenance.v1",
        "provenance_type": "legacy_success_boolean",
        "label_source": label_source or f"{simulator}_command_output",
        "evidence_artifact_type": "legacy_boolean_without_detailed_trace",
        "buyer_disclosure": (
            "Success labels come from a legacy simulator/provider boolean without "
            "detailed trace provenance; display only with this limitation."
        ),
        "generated_video_vlm_judge": False,
        "simulator_physics_or_trace": False,
        "real_world_outcome": False,
        "success_label_disclosed_to_buyer": True,
        "public_claim_upgrade_allowed": False,
    }


def _simulator_attempts_from_payload(
    *,
    payload: Any,
    simulator: str,
    generated_at: str,
) -> List[Dict[str, Any]]:
    records = _records_from_payload(payload)
    attempts: List[Dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        status = _string(record.get("status") or record.get("result") or "completed").lower()
        explicit_success = record.get("success")
        # `success` may fall back to the episode status (runtime completion), but
        # `task_success` must never: an episode that merely finished without an explicit
        # task verdict fails closed instead of inheriting "completed" as task success.
        success = (
            _boolish(explicit_success)
            if explicit_success is not None
            else status in {"completed", "success", "succeeded", "passed"}
        )
        task_outcome = _mapping(record.get("task_outcome") or record.get("taskOutcome"))
        task_success_raw = (
            record.get("task_success")
            if record.get("task_success") is not None
            else task_outcome.get("task_success")
        )
        task_success_explicit = task_success_raw is not None or explicit_success is not None
        task_success = (
            _boolish(task_success_raw)
            if task_success_raw is not None
            else _boolish(explicit_success)
            if explicit_success is not None
            else False
        )
        failure_ids = _failure_ids(record, "failure_mode_ids", "failure_modes", "failures")
        if not failure_ids:
            failure_ids = _failure_ids(
                task_outcome, "failure_mode_ids", "failure_modes", "failures"
            )
        if (not success or not task_success) and not failure_ids:
            failure_ids = [
                _string(record.get("failure_reason"))
                or (
                    "task_success_not_reported_failing_closed"
                    if not task_success_explicit
                    else "simulator_failure"
                )
            ]
        label_provenance = _task_success_label_provenance(
            record=record,
            task_outcome=task_outcome,
            simulator=simulator,
        )
        attempts.append(
            {
                "attempt_id": _string(record.get("attempt_id") or record.get("attemptId"))
                or f"{simulator}_attempt_{index:04d}",
                "episode_id": _string(record.get("episode_id") or record.get("episodeId"))
                or f"{simulator}_episode_{index:04d}",
                "scenario_id": _string(record.get("scenario_id") or record.get("scenarioId")),
                "scenario_run_id": _string(
                    record.get("scenario_run_id") or record.get("scenarioRunId")
                )
                or f"{simulator}_scenario_run_{index:04d}",
                "scenario_eval_run_id": _string(
                    record.get("scenario_eval_run_id") or record.get("scenarioEvalRunId")
                )
                or None,
                "scenario_variation_instance_id": _string(
                    record.get("scenario_variation_instance_id")
                    or record.get("scenarioVariationInstanceId")
                )
                or None,
                "variation_name": _string(
                    record.get("variation_name") or record.get("variationName")
                )
                or None,
                "task_id": _string(record.get("task_id") or record.get("taskId")),
                "policy_id": _string(record.get("policy_id") or record.get("policyId")),
                "engine": simulator,
                "runner": "command_adapter",
                "status": status,
                "success": bool(success and task_success),
                "task_success": bool(task_success),
                "task_success_explicit": bool(task_success_explicit),
                "task_success_label_provenance": label_provenance,
                "task_status": _string(
                    record.get("task_status")
                    or record.get("taskStatus")
                    or task_outcome.get("task_status")
                )
                or ("passed" if task_success else "failed_task_criteria"),
                "failure_reason": _string(record.get("failure_reason") or record.get("reason"))
                or _string(task_outcome.get("failure_reason"))
                or None,
                "failure_mode_ids": failure_ids,
                "metrics": _mapping(record.get("metrics")),
                "task_outcome": task_outcome,
                "spawn_pose": record.get("spawn_pose") or record.get("spawnPose"),
                "target_pose": record.get("target_pose") or record.get("targetPose"),
                "final_pose": record.get("final_pose") or record.get("finalPose"),
                "deterministic_seed": record.get("deterministic_seed")
                or _mapping(record.get("metrics")).get("deterministic_seed"),
                "route_source": _string(record.get("route_source") or record.get("routeSource"))
                or None,
                "route_strategy": _string(
                    record.get("route_strategy") or record.get("routeStrategy")
                )
                or None,
                "route_waypoints": record.get("route_waypoints")
                if isinstance(record.get("route_waypoints"), list)
                else [],
                "action_trace": record.get("actions")
                if isinstance(record.get("actions"), list)
                else [],
                "contact_trace": record.get("contact_trace")
                if isinstance(record.get("contact_trace"), list)
                else [],
                "safety_events": record.get("safety_events")
                if isinstance(record.get("safety_events"), list)
                else [],
                "video_path": _string(record.get("video_path") or record.get("videoPath")) or None,
                "artifact_paths": _mapping(
                    record.get("artifact_paths") or record.get("artifactPaths")
                ),
                "generated_at": generated_at,
                "claim_boundary": "simulator_command_output_not_real_robot_deployment_proof",
            }
        )
    return attempts


def _task_success_summary_from_attempts(attempts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    successful = [attempt for attempt in attempts if bool(attempt.get("task_success"))]
    failed = [attempt for attempt in attempts if not bool(attempt.get("task_success"))]
    failure_mode_counts: Dict[str, int] = {}
    task_outcomes = [_mapping(attempt.get("task_outcome")) for attempt in attempts]
    provenance_counts: Dict[str, int] = {}
    provenance_disclosures: Dict[str, str] = {}
    generated_video_vlm_judged_attempt_count = 0
    undisclosed_attempt_ids: List[str] = []

    def outcome_has(key: str) -> bool:
        return any(key in outcome for outcome in task_outcomes)

    for attempt in attempts:
        provenance = _mapping(attempt.get("task_success_label_provenance"))
        provenance_type = _string(provenance.get("provenance_type")) or "missing"
        provenance_counts[provenance_type] = provenance_counts.get(provenance_type, 0) + 1
        disclosure = _string(provenance.get("buyer_disclosure"))
        if disclosure:
            provenance_disclosures.setdefault(provenance_type, disclosure)
        if provenance.get("generated_video_vlm_judge") is True:
            generated_video_vlm_judged_attempt_count += 1
        if not disclosure:
            undisclosed_attempt_ids.append(
                _string(attempt.get("attempt_id")) or f"attempt_{len(undisclosed_attempt_ids) + 1}"
            )

    for attempt in failed:
        for failure_mode in _string_list(attempt.get("failure_mode_ids")):
            failure_mode_counts[failure_mode] = failure_mode_counts.get(failure_mode, 0) + 1
    final_errors = [
        value
        for value in (
            _number(_mapping(attempt.get("task_outcome")).get("final_target_error_m"))
            for attempt in attempts
        )
        if value is not None
    ]
    path_deviations = [
        value
        for value in (
            _number(_mapping(attempt.get("task_outcome")).get("max_path_deviation_m"))
            for attempt in attempts
        )
        if value is not None
    ]
    clearance_values = [
        value
        for value in (
            _number(_mapping(attempt.get("task_outcome")).get("min_clearance_m"))
            for attempt in attempts
        )
        if value is not None
    ]
    success_rate_provenance_disclosed = bool(attempts) and not undisclosed_attempt_ids
    success_rate_buyer_display_blockers = (
        []
        if success_rate_provenance_disclosed
        else ["task_success_label_provenance_missing"]
        if attempts
        else ["task_success_attempts_not_available"]
    )
    return {
        "schema_version": "robot_eval_task_success_summary.v1",
        "status": "completed" if attempts else "not_available",
        "attempt_count": len(attempts),
        "successful_attempt_count": len(successful),
        "failed_attempt_count": len(failed),
        "task_success_rate": round(len(successful) / len(attempts), 6) if attempts else None,
        "task_success_label_provenance_counts": dict(sorted(provenance_counts.items())),
        "task_success_label_provenance_disclosures": dict(sorted(provenance_disclosures.items())),
        "generated_video_vlm_judged_attempt_count": generated_video_vlm_judged_attempt_count,
        "success_rate_requires_provenance_disclosure": True,
        "success_rate_provenance_disclosed": success_rate_provenance_disclosed,
        "success_rate_buyer_display_allowed": success_rate_provenance_disclosed,
        "success_rate_buyer_display_blockers": success_rate_buyer_display_blockers,
        "undisclosed_success_label_attempt_ids": undisclosed_attempt_ids,
        "failed_scenario_eval_run_ids": sorted(
            _string(attempt.get("scenario_eval_run_id"))
            for attempt in failed
            if _string(attempt.get("scenario_eval_run_id"))
        ),
        "failure_mode_counts": dict(sorted(failure_mode_counts.items())),
        "near_miss_attempt_count": sum(
            1
            for attempt in attempts
            if int(_mapping(attempt.get("task_outcome")).get("near_miss_event_count") or 0) > 0
        )
        if outcome_has("near_miss_event_count")
        else None,
        "near_miss_event_count": sum(
            int(_mapping(attempt.get("task_outcome")).get("near_miss_event_count") or 0)
            for attempt in attempts
        )
        if outcome_has("near_miss_event_count")
        else None,
        "min_clearance_m": min(clearance_values) if clearance_values else None,
        "clearance_threshold_m": min(
            (
                value
                for value in (
                    _number(_mapping(attempt.get("task_outcome")).get("clearance_threshold_m"))
                    for attempt in attempts
                )
                if value is not None
            ),
            default=None,
        ),
        "fall_attempt_count": sum(
            1
            for attempt in attempts
            if bool(_mapping(attempt.get("task_outcome")).get("fall_detected"))
        )
        if outcome_has("fall_detected")
        else None,
        "stuck_attempt_count": sum(
            1
            for attempt in attempts
            if bool(_mapping(attempt.get("task_outcome")).get("stuck_detected"))
        )
        if outcome_has("stuck_detected")
        else None,
        "policy_instability_attempt_count": sum(
            1
            for attempt in attempts
            if bool(_mapping(attempt.get("task_outcome")).get("policy_instability_detected"))
        )
        if outcome_has("policy_instability_detected")
        else None,
        "scene_contact_attempt_count": sum(
            1
            for attempt in attempts
            if int(
                _mapping(attempt.get("task_outcome")).get("robot_scene_contact_event_count") or 0
            )
            > 0
        )
        if outcome_has("robot_scene_contact_event_count")
        else None,
        "endpoint_clean_attempt_count": sum(
            1
            for attempt in attempts
            if bool(_mapping(attempt.get("task_outcome")).get("endpoint_clean"))
        ),
        "goal_reached_attempt_count": sum(
            1
            for attempt in attempts
            if bool(_mapping(attempt.get("task_outcome")).get("goal_reached"))
        ),
        "max_final_target_error_m": max(final_errors) if final_errors else None,
        "max_path_deviation_m": max(path_deviations) if path_deviations else None,
        "task_success_boundary": (
            "Task success is normalized separately from simulator command completion; "
            "failed attempts remain valid evidence when coverage is complete."
        ),
    }


def build_simulator_command_artifacts(
    *,
    job_dir: str | Path,
    simulator: str,
    simulator_output: Any,
    generated_at: str,
) -> Dict[str, Any]:
    """Normalize simulator command output into evaluator/package artifacts."""

    resolved_job_dir = Path(job_dir).resolve()
    attempts = _simulator_attempts_from_payload(
        payload=simulator_output,
        simulator=simulator,
        generated_at=generated_at,
    )
    simulator_payload = _mapping(simulator_output)
    simulator_output_execution_proven = simulator_payload.get("simulator_execution_proven")
    simulator_execution_proven = (
        simulator_output_execution_proven is True
        if simulator_output_execution_proven is not None
        else bool(attempts)
    )
    required_scenario_eval_run_ids = _string_list(
        simulator_payload.get("required_scenario_eval_run_ids")
    )
    if not required_scenario_eval_run_ids:
        matrix_path = resolved_job_dir / "scenario_eval_matrix.json"
        if matrix_path.is_file():
            try:
                matrix_payload = _mapping(read_json_any(matrix_path))
            except Exception:
                matrix_payload = {}
            raw_runs = matrix_payload.get("runs")
            if isinstance(raw_runs, Sequence) and not isinstance(raw_runs, (str, bytes)):
                required_scenario_eval_run_ids = [
                    run_id
                    for run_id in (
                        _string(_mapping(raw_run).get("scenario_eval_run_id"))
                        for raw_run in raw_runs
                    )
                    if run_id
                ]
    covered_scenario_eval_run_ids = sorted(
        {
            _string(attempt.get("scenario_eval_run_id"))
            for attempt in attempts
            if _string(attempt.get("scenario_eval_run_id"))
        }
    )
    duplicate_scenario_eval_run_ids = sorted(
        {
            run_id
            for run_id in required_scenario_eval_run_ids
            if required_scenario_eval_run_ids.count(run_id) > 1
        }
    )
    missing_scenario_eval_run_ids = sorted(
        set(required_scenario_eval_run_ids) - set(covered_scenario_eval_run_ids)
    )
    if not required_scenario_eval_run_ids:
        missing_scenario_eval_run_ids = _string_list(
            simulator_payload.get("missing_scenario_eval_run_ids")
        )
    attempt_count_matches_matrix_count = not required_scenario_eval_run_ids or len(attempts) == len(
        required_scenario_eval_run_ids
    )
    scenario_eval_run_id_coverage_exact = not required_scenario_eval_run_ids or (
        set(covered_scenario_eval_run_ids) == set(required_scenario_eval_run_ids)
        and len(covered_scenario_eval_run_ids) == len(required_scenario_eval_run_ids)
    )
    scenario_eval_run_coverage_complete = (
        bool(required_scenario_eval_run_ids)
        and attempt_count_matches_matrix_count
        and scenario_eval_run_id_coverage_exact
        and not missing_scenario_eval_run_ids
        and not duplicate_scenario_eval_run_ids
    )
    status = "completed" if attempts else "blocked_missing_simulator_attempts"
    if attempts and required_scenario_eval_run_ids and not scenario_eval_run_coverage_complete:
        status = "blocked_incomplete_scenario_eval_run_coverage"
    if attempts and not simulator_execution_proven:
        status = "blocked_simulator_execution_not_proven"
    failures = [attempt for attempt in attempts if not bool(attempt.get("success"))]
    task_success_summary = _task_success_summary_from_attempts(attempts)
    failed_attempt_ids = sorted(
        _string(attempt.get("attempt_id"))
        for attempt in failures
        if _string(attempt.get("attempt_id"))
    )
    failed_scenario_eval_run_ids = sorted(
        _string(attempt.get("scenario_eval_run_id"))
        for attempt in failures
        if _string(attempt.get("scenario_eval_run_id"))
    )
    trace = {
        "schema_version": "robot_eval_simulator_command_normalized_attempt_trace.v1",
        "generated_at": generated_at,
        "status": status,
        "backend": simulator,
        "attempt_count": len(attempts),
        "required_scenario_eval_run_count": len(required_scenario_eval_run_ids),
        "covered_scenario_eval_run_count": len(covered_scenario_eval_run_ids),
        "missing_scenario_eval_run_count": len(missing_scenario_eval_run_ids),
        "attempt_count_matches_matrix_count": attempt_count_matches_matrix_count,
        "scenario_eval_run_id_coverage_exact": scenario_eval_run_id_coverage_exact,
        "duplicate_scenario_eval_run_ids": duplicate_scenario_eval_run_ids,
        "required_scenario_eval_run_ids": required_scenario_eval_run_ids,
        "covered_scenario_eval_run_ids": covered_scenario_eval_run_ids,
        "missing_scenario_eval_run_ids": missing_scenario_eval_run_ids,
        "scenario_eval_run_coverage_complete": scenario_eval_run_coverage_complete,
        "task_success_summary": task_success_summary,
        "successful_task_attempt_count": task_success_summary["successful_attempt_count"],
        "failed_task_attempt_count": task_success_summary["failed_attempt_count"],
        "task_success_rate": task_success_summary["task_success_rate"],
        "task_success_label_provenance_counts": task_success_summary[
            "task_success_label_provenance_counts"
        ],
        "generated_video_vlm_judged_attempt_count": task_success_summary[
            "generated_video_vlm_judged_attempt_count"
        ],
        "success_rate_requires_provenance_disclosure": task_success_summary[
            "success_rate_requires_provenance_disclosure"
        ],
        "success_rate_provenance_disclosed": task_success_summary[
            "success_rate_provenance_disclosed"
        ],
        "success_rate_buyer_display_allowed": task_success_summary[
            "success_rate_buyer_display_allowed"
        ],
        "success_rate_buyer_display_blockers": task_success_summary[
            "success_rate_buyer_display_blockers"
        ],
        "attempts": attempts,
        "result_ingested": bool(attempts),
        "simulator_execution_proven": simulator_execution_proven,
        "robot_policy_execution_proven": False,
        "rank_fidelity_result_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }

    def _failure_label(attempt: Mapping[str, Any]) -> Dict[str, Any]:
        task_outcome = _mapping(attempt.get("task_outcome"))
        artifact_paths = _mapping(attempt.get("artifact_paths"))
        evidence_refs: List[Dict[str, Any]] = []
        for key in (
            "scene_trace",
            "spawn_trace",
            "policy_trace",
            "sim_robot_pov_evidence",
        ):
            path = artifact_paths.get(key)
            if path:
                evidence_refs.append({"kind": key, "path": path})
        frames = artifact_paths.get("frames")
        if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes)):
            evidence_refs.append(
                {
                    "kind": "rendered_episode_frames",
                    "frame_count": len(frames),
                    "sample_paths": list(frames[:3]),
                }
            )
        criteria_metric_keys = (
            "goal_reached",
            "endpoint_clean",
            "spawn_clean",
            "timeout",
            "fall_detected",
            "stuck_detected",
            "policy_instability_detected",
            "final_target_error_m",
            "goal_tolerance_m",
            "min_clearance_m",
            "clearance_threshold_m",
            "clearance_threshold_violation",
            "robot_scene_contact_event_count",
            "near_miss_event_count",
            "progress_to_goal_ratio",
            "path_efficiency_ratio",
            "cycle_time_seconds",
        )
        success_criteria = _mapping(task_outcome.get("success_criteria"))
        if not success_criteria:
            derived_criteria_sources = {
                "goal_reached_within_tolerance": "goal_reached",
                "endpoint_clean": "endpoint_clean",
                "spawn_clean": "spawn_clean",
                "no_timeout": "timeout",
                "no_fall_detected": "fall_detected",
                "no_stuck_or_no_progress": "stuck_detected",
                "no_policy_instability": "policy_instability_detected",
                "no_clearance_near_miss": "clearance_threshold_violation",
            }
            for criterion, source_key in derived_criteria_sources.items():
                if source_key not in task_outcome:
                    continue
                value = bool(task_outcome.get(source_key))
                success_criteria[criterion] = not value if criterion.startswith("no_") else value
        failure_mode_ids = _string_list(attempt.get("failure_mode_ids"))
        return {
            "label_id": f"label_{_safe_id(_string(attempt.get('attempt_id')))}",
            "attempt_id": attempt.get("attempt_id"),
            "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
            "variation_name": attempt.get("variation_name"),
            "task_id": attempt.get("task_id"),
            "scenario_id": attempt.get("scenario_id"),
            "policy_id": attempt.get("policy_id"),
            "label": "failure",
            "label_source": "deterministic_simulator_command_state_contact_route_trace",
            "status": "deterministically_labeled_failure",
            "task_success": bool(attempt.get("task_success")),
            "task_status": attempt.get("task_status"),
            "failure_mode_ids": failure_mode_ids,
            "primary_failure_mode": failure_mode_ids[0] if failure_mode_ids else None,
            "failure_reason": attempt.get("failure_reason"),
            "criteria_results": {
                "success_criteria": success_criteria,
                "metrics": {
                    key: task_outcome.get(key)
                    for key in criteria_metric_keys
                    if key in task_outcome
                },
            },
            "task_outcome": task_outcome,
            "evidence_refs": evidence_refs,
            "review_status": "available_for_human_audit_not_required_for_sim_only_metric",
            "proof_effect": "sim_only_metric_input_not_real_rank_fidelity",
        }

    labels = {
        "schema_version": "robot_eval_simulator_command_failure_labels.v1",
        "generated_at": generated_at,
        "status": "review_required" if failures else "no_failures_labeled",
        "label_count": len(failures),
        "failed_attempt_count": len(failures),
        "covered_failed_attempt_ids": failed_attempt_ids,
        "missing_failed_attempt_ids": [],
        "covered_failed_scenario_eval_run_ids": failed_scenario_eval_run_ids,
        "missing_failed_scenario_eval_run_ids": [],
        "failed_run_label_coverage_complete": True,
        "task_success_summary": task_success_summary,
        "labels": [_failure_label(attempt) for attempt in failures],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }

    def _visual_review(attempt: Mapping[str, Any]) -> Dict[str, Any]:
        task_outcome = _mapping(attempt.get("task_outcome"))
        artifact_paths = _mapping(attempt.get("artifact_paths"))
        media_refs: List[Dict[str, Any]] = []
        for key in ("video_path", "sim_robot_pov_evidence", "overview_video", "robot_pov_video"):
            value = artifact_paths.get(key) or attempt.get(key)
            if value:
                media_refs.append({"kind": key, "path": value})
        frames = artifact_paths.get("frames")
        if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes)):
            media_refs.append(
                {
                    "kind": "rendered_episode_frames",
                    "frame_count": len(frames),
                    "sample_paths": list(frames[:3]),
                }
            )
        criteria = _mapping(task_outcome.get("success_criteria"))
        if not criteria:
            criteria = {
                "goal_reached": bool(task_outcome.get("goal_reached")),
                "endpoint_clean": bool(task_outcome.get("endpoint_clean")),
                "spawn_clean": bool(task_outcome.get("spawn_clean", True)),
                "no_timeout": not bool(task_outcome.get("timeout")),
                "no_fall_detected": not bool(task_outcome.get("fall_detected")),
                "no_stuck_or_no_progress": not bool(task_outcome.get("stuck_detected")),
                "no_policy_instability": not bool(task_outcome.get("policy_instability_detected")),
                "no_clearance_near_miss": not bool(
                    task_outcome.get("clearance_threshold_violation")
                ),
            }
        accepted = bool(attempt.get("task_success"))
        return {
            "review_id": f"visual_review_{_safe_id(_string(attempt.get('attempt_id')))}",
            "attempt_id": attempt.get("attempt_id"),
            "episode_id": attempt.get("episode_id"),
            "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
            "variation_name": attempt.get("variation_name"),
            "task_id": attempt.get("task_id"),
            "scenario_id": attempt.get("scenario_id"),
            "decision": "success" if accepted else "failure",
            "success": accepted,
            "failure_mode_ids": attempt.get("failure_mode_ids") or [],
            "criteria_results": {
                "success_criteria": criteria,
                "task_outcome": task_outcome,
            },
            "media_refs": media_refs,
            "media_evidence_present": bool(media_refs),
            "confidence": "high" if media_refs else "medium_trace_only",
            "confidence_score": 0.92 if media_refs else 0.72,
            "review_status": "accepted_deterministic_simulator_visual_review",
            "human_review_status": "not_required_for_sim_only_failure_packaging",
            "claim_boundary": "accepted_simulator_review_labels_success_or_failure_only_not_robot_capability_claim",
        }

    visual_review_records = [_visual_review(attempt) for attempt in attempts]
    visual_review_ledger = {
        "schema_version": "robot_eval_simulator_visual_review_ledger.v1",
        "generated_at": generated_at,
        "status": "accepted" if visual_review_records else "not_available",
        "review_count": len(visual_review_records),
        "attempt_count": len(attempts),
        "accepted_review_count": sum(
            1
            for record in visual_review_records
            if record["review_status"] == "accepted_deterministic_simulator_visual_review"
        ),
        "success_count": sum(1 for record in visual_review_records if record["success"]),
        "failure_count": sum(1 for record in visual_review_records if not record["success"]),
        "media_backed_review_count": sum(
            1 for record in visual_review_records if record["media_evidence_present"]
        ),
        "visual_review_coverage_complete": bool(attempts)
        and len(visual_review_records) == len(attempts),
        "records": visual_review_records,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    prediction_records = [
        {
            "scenario_id": attempt.get("scenario_id"),
            "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
            "variation_name": attempt.get("variation_name"),
            "task_id": attempt.get("task_id"),
            "policy_id": attempt.get("policy_id"),
            "predicted_status": "passed" if attempt.get("success") else "failed",
            "predicted_success": bool(attempt.get("success")),
            "predicted_task_success": bool(attempt.get("task_success")),
            "predicted_cycle_time_seconds": _number(
                _mapping(attempt.get("metrics")).get("cycle_time_seconds")
            ),
            "predicted_final_target_error_m": _number(
                _mapping(attempt.get("task_outcome")).get("final_target_error_m")
            ),
            "predicted_endpoint_clean": _mapping(attempt.get("task_outcome")).get("endpoint_clean"),
            "failure_mode_ids": attempt.get("failure_mode_ids") or [],
            "source": f"{simulator}_command_output",
            "actual_status": "needs_actual_outcome",
        }
        for attempt in attempts
    ]
    prediction_ledger = {
        "schema_version": "robot_eval_simulator_prediction_outcome_ledger.v1",
        "generated_at": generated_at,
        "status": "completed" if attempts else "not_available",
        "record_count": len(prediction_records),
        "records": prediction_records,
        "task_success_summary": task_success_summary,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    calibration_report = {
        "schema_version": "robot_eval_simulator_calibration_report.v1",
        "generated_at": generated_at,
        "status": "needs_real_world_outcomes" if attempts else "not_available",
        "record_count": len(prediction_records),
        "records": prediction_records,
        "task_success_summary": task_success_summary,
        "sim_vs_real_calibration_score": None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    breakage_library = {
        "schema_version": "robot_eval_simulator_breakage_library.v1",
        "generated_at": generated_at,
        "status": "review_required" if failures else "no_breakages_recorded",
        "record_count": len(failures),
        "records": [
            {
                "scenario_id": attempt.get("scenario_id"),
                "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
                "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
                "variation_name": attempt.get("variation_name"),
                "task_id": attempt.get("task_id"),
                "failure_mode_ids": attempt.get("failure_mode_ids") or [],
                "failure_reason": attempt.get("failure_reason"),
                "task_status": attempt.get("task_status"),
                "task_outcome": attempt.get("task_outcome") or {},
                "review_required": True,
            }
            for attempt in failures
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    command_batch_trace_package = _mapping(simulator_payload.get("batch_trace_package"))
    command_batch_closure_manifest = _mapping(simulator_payload.get("batch_closure_manifest"))
    command_batch_trace_copied_paths: Dict[str, str] = {}
    command_batch_trace_copy_records: Dict[str, Any] = {}
    source_base_dir = _source_trace_base_dir(simulator_payload)
    if command_batch_trace_package:
        command_batch_trace_copied_paths, command_batch_trace_copy_records = (
            _copy_command_batch_trace_artifacts(
                job_dir=resolved_job_dir,
                trace_package=command_batch_trace_package,
                source_base_dir=source_base_dir,
            )
        )
        command_batch_trace_package = {
            **command_batch_trace_package,
            "source_artifact_paths": dict(
                _mapping(command_batch_trace_package.get("artifact_paths"))
            ),
            "artifact_paths": dict(command_batch_trace_copied_paths),
            "job_artifact_copy_status": "copied"
            if set(BATCH_TRACE_ARTIFACT_JOB_NAMES).issubset(set(command_batch_trace_copied_paths))
            else "partial_or_missing",
            "job_artifact_copy_records": command_batch_trace_copy_records,
        }
    simulator_artifact_paths = _mapping(simulator_payload.get("artifact_paths"))
    digital_twin_qa_job_name = "simulator_command_digital_twin_fidelity_qa.json"
    digital_twin_qa_copy_record: Dict[str, Any] = {
        "status": "missing_source_ref",
        "source_ref": None,
        "job_artifact": digital_twin_qa_job_name,
    }
    digital_twin_qa_source_ref = _string(simulator_artifact_paths.get("digital_twin_fidelity_qa"))
    if digital_twin_qa_source_ref:
        if "://" in digital_twin_qa_source_ref:
            digital_twin_qa_copy_record = {
                "status": "remote_source_not_copied",
                "source_ref": digital_twin_qa_source_ref,
                "job_artifact": digital_twin_qa_job_name,
            }
        else:
            digital_twin_qa_source_path = Path(digital_twin_qa_source_ref)
            if not digital_twin_qa_source_path.is_absolute() and source_base_dir is not None:
                digital_twin_qa_source_path = source_base_dir / digital_twin_qa_source_path
            digital_twin_qa_destination = resolved_job_dir / digital_twin_qa_job_name
            if digital_twin_qa_source_path.is_file():
                ensure_dir(digital_twin_qa_destination.parent)
                if digital_twin_qa_source_path.resolve() != digital_twin_qa_destination.resolve():
                    shutil.copyfile(digital_twin_qa_source_path, digital_twin_qa_destination)
                digital_twin_qa_copy_record = {
                    "status": "copied",
                    "source_ref": digital_twin_qa_source_ref,
                    "job_artifact": digital_twin_qa_job_name,
                    "sha256": _sha256_file(digital_twin_qa_destination),
                }
            else:
                digital_twin_qa_copy_record = {
                    "status": "missing_source_file",
                    "source_ref": digital_twin_qa_source_ref,
                    "job_artifact": digital_twin_qa_job_name,
                }
    artifact_paths = {
        "normalized_attempt_trace": "normalized_attempt_trace.json",
        "failure_labels": "failure_labels.json",
        "visual_review_ledger": "visual_review_ledger.json",
        "prediction_outcome_ledger": "prediction_outcome_ledger.json",
        "calibration_report": "calibration_report.json",
        "breakage_library": "breakage_library.json",
    }
    if command_batch_trace_package:
        artifact_paths["simulator_command_batch_trace_package_manifest"] = (
            "simulator_command_batch_trace_package_manifest.json"
        )
        for artifact_key, job_name in command_batch_trace_copied_paths.items():
            artifact_paths[f"simulator_command_batch_{artifact_key}"] = job_name
    if command_batch_closure_manifest:
        artifact_paths["simulator_command_batch_closure_manifest"] = (
            "simulator_command_batch_closure_manifest.json"
        )
    if digital_twin_qa_copy_record.get("status") == "copied":
        artifact_paths["simulator_command_digital_twin_fidelity_qa"] = digital_twin_qa_job_name
    manifest = {
        "schema_version": SIMULATOR_COMMAND_ARTIFACTS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "simulator": simulator,
        "simulator_execution_proven": simulator_execution_proven,
        "attempt_count": len(attempts),
        "required_scenario_eval_run_count": len(required_scenario_eval_run_ids),
        "covered_scenario_eval_run_count": len(covered_scenario_eval_run_ids),
        "missing_scenario_eval_run_count": len(missing_scenario_eval_run_ids),
        "attempt_count_matches_matrix_count": attempt_count_matches_matrix_count,
        "scenario_eval_run_id_coverage_exact": scenario_eval_run_id_coverage_exact,
        "duplicate_scenario_eval_run_ids": duplicate_scenario_eval_run_ids,
        "scenario_eval_run_coverage_complete": scenario_eval_run_coverage_complete,
        "task_success_summary": task_success_summary,
        "successful_task_attempt_count": task_success_summary["successful_attempt_count"],
        "failed_task_attempt_count": task_success_summary["failed_attempt_count"],
        "task_success_rate": task_success_summary["task_success_rate"],
        "visual_review_count": visual_review_ledger["review_count"],
        "visual_review_coverage_complete": visual_review_ledger["visual_review_coverage_complete"],
        "artifact_paths": artifact_paths,
        "command_batch_trace_package_status": command_batch_trace_package.get("status"),
        "command_batch_trace_job_artifact_copy_status": command_batch_trace_package.get(
            "job_artifact_copy_status"
        ),
        "command_batch_trace_job_artifacts_copied": bool(
            command_batch_trace_package
            and set(BATCH_TRACE_ARTIFACT_JOB_NAMES).issubset(set(command_batch_trace_copied_paths))
        ),
        "simulator_command_digital_twin_fidelity_qa_copy_record": (digital_twin_qa_copy_record),
        "command_batch_closure_status": command_batch_closure_manifest.get("status"),
        "machine_trace_package_complete": command_batch_closure_manifest.get(
            "machine_trace_package_complete"
        ),
        "robot_team_grade_package_complete": command_batch_closure_manifest.get(
            "robot_team_grade_package_complete"
        ),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(resolved_job_dir / "normalized_attempt_trace.json", trace)
    write_json(resolved_job_dir / "failure_labels.json", labels)
    write_json(resolved_job_dir / "visual_review_ledger.json", visual_review_ledger)
    write_json(resolved_job_dir / "prediction_outcome_ledger.json", prediction_ledger)
    write_json(resolved_job_dir / "calibration_report.json", calibration_report)
    write_json(resolved_job_dir / "breakage_library.json", breakage_library)
    if command_batch_trace_package:
        write_json(
            resolved_job_dir / "simulator_command_batch_trace_package_manifest.json",
            command_batch_trace_package,
        )
    if command_batch_closure_manifest:
        write_json(
            resolved_job_dir / "simulator_command_batch_closure_manifest.json",
            command_batch_closure_manifest,
        )
    write_json(resolved_job_dir / "simulator_command_artifacts_manifest.json", manifest)
    return {
        "manifest": manifest,
        "normalized_attempt_trace": trace,
        "failure_labels": labels,
        "visual_review_ledger": visual_review_ledger,
        "prediction_outcome_ledger": prediction_ledger,
        "calibration_report": calibration_report,
        "breakage_library": breakage_library,
        "simulator_command_batch_trace_package_manifest": command_batch_trace_package,
        "simulator_command_batch_closure_manifest": command_batch_closure_manifest,
    }


def _load_actual_outcome_payload(
    *,
    capture_root: Path,
    job_dir: Path,
    job_request: Mapping[str, Any],
) -> tuple[Any, str | None]:
    for key in (
        "actual_outcomes",
        "actualOutcomes",
        "real_world_outcomes",
        "realWorldOutcomes",
        "deployment_outcomes",
        "deploymentOutcomes",
    ):
        value = job_request.get(key)
        if isinstance(value, (Mapping, list)):
            return value, f"job_request_inline_{key}"
    explicit_refs = [
        job_request.get("actual_outcome_manifest_uri"),
        job_request.get("actualOutcomeManifestUri"),
        job_request.get("deployment_outcome_manifest_uri"),
        job_request.get("deploymentOutcomeManifestUri"),
    ]
    for ref in explicit_refs:
        loaded = _load_reference_json(ref, capture_root=capture_root, job_dir=job_dir)
        if loaded is not None:
            return loaded, "job_request_outcome_manifest_ref"
    for path in (
        capture_root / "pipeline" / "robot_eval_inputs" / "deployment_outcome_manifest.json",
        capture_root / "pipeline" / "robot_eval_inputs" / "actual_outcome_manifest.json",
    ):
        loaded = _read_optional_any(path)
        if loaded is not None and _records_from_payload(loaded):
            return loaded, "capture_robot_eval_inputs_outcome_manifest"
    inbox_payload = _load_actual_outcome_inbox(
        capture_root=capture_root,
        job_dir=job_dir,
    )
    if inbox_payload.get("records"):
        return inbox_payload, "deployment_outcome_inbox"
    return None, None


def _load_actual_outcome_inbox(*, capture_root: Path, job_dir: Path) -> Dict[str, Any]:
    job_id = job_dir.name
    inboxes = (
        capture_root / "pipeline" / "robot_eval_inputs" / job_id / "deployment_outcomes" / "inbox",
        capture_root / "pipeline" / "robot_eval_inputs" / job_id / "actual_outcomes" / "inbox",
        capture_root / "pipeline" / "robot_eval_inputs" / "deployment_outcomes" / "inbox",
        capture_root / "pipeline" / "robot_eval_inputs" / "actual_outcomes" / "inbox",
        job_dir / "deployment_outcomes" / "inbox",
        job_dir / "actual_outcomes" / "inbox",
    )
    records: List[Dict[str, Any]] = []
    source_files: List[str] = []
    blockers: List[str] = []
    for inbox in inboxes:
        if not inbox.is_dir():
            continue
        for path in sorted(inbox.glob("*.json")):
            if not path.is_file() or path.name.startswith("."):
                continue
            try:
                payload = read_json_any(path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                blockers.append(f"{path.name}:read_failed:{type(exc).__name__}")
                continue
            rows = _records_from_payload(payload)
            if not rows:
                blockers.append(f"{path.name}:no_outcome_records")
                continue
            for row in rows:
                row.setdefault("source_outcome_file", str(path))
                records.append(row)
            source_files.append(str(path))
    return {
        "schema_version": "deployment_outcome_inbox.v1",
        "status": "ready" if records else "empty",
        "record_count": len(records),
        "source_files": source_files,
        "records": records,
        "blockers": blockers,
        "claim_boundary": "deployment_outcome_inbox_is_owner_supplied_actual_outcome_input",
    }


def _attestation_present(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if not isinstance(value, Mapping):
        return False
    return bool(
        _string(value.get("attested_by") or value.get("attestedBy") or value.get("operator_id"))
        and _string(
            value.get("attestation")
            or value.get("statement")
            or value.get("accepted_claim_boundary")
            or value.get("acceptedClaimBoundary")
        )
    )


def _outcome_owner_evidence(actual: Mapping[str, Any]) -> Dict[str, Any]:
    evidence_refs = _mapping(
        actual.get("evidence_refs")
        or actual.get("evidenceRefs")
        or actual.get("owner_evidence_refs")
        or actual.get("ownerEvidenceRefs")
    )
    owner_evidence_uri = _string(
        actual.get("evidence_uri")
        or actual.get("evidenceUri")
        or actual.get("pilot_log_uri")
        or actual.get("pilotLogUri")
        or actual.get("owner_system_proof_uri")
        or actual.get("ownerSystemProofUri")
    )
    attestation = _mapping(
        actual.get("operator_attestation")
        or actual.get("operatorAttestation")
        or actual.get("owner_attestation")
        or actual.get("ownerAttestation")
    )
    owner_attestation = _mapping(
        actual.get("hardware_owner_attestation")
        or actual.get("hardwareOwnerAttestation")
        or actual.get("owner_attestation")
        or actual.get("ownerAttestation")
    )
    return {
        "owner_evidence_present": bool(
            evidence_refs or owner_evidence_uri or _attestation_present(attestation)
        ),
        "owner_evidence_refs": evidence_refs,
        "owner_evidence_uri": owner_evidence_uri or None,
        "operator_attestation": attestation,
        "owner_attestation": owner_attestation,
        "signed_operator_attestation_present": _attestation_signed(attestation)
        or _attestation_signed(owner_attestation),
    }


def _followup_action_context(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "record_id": _string(row.get("record_id")),
        "task_id": _string(row.get("task_id")),
        "scenario_id": _string(row.get("scenario_id")),
        "scenario_eval_run_id": _string(row.get("scenario_eval_run_id")) or None,
        "scenario_variation_instance_id": _string(row.get("scenario_variation_instance_id"))
        or None,
        "variation_name": _string(row.get("variation_name")) or None,
        "policy_id": _string(row.get("policy_id")) or None,
        "prediction_match_level": _string(row.get("prediction_match_level")) or None,
        "predicted_success": row.get("predicted_success"),
        "actual_success": row.get("actual_success"),
        "predicted_failures": _string_list(row.get("predicted_failures")),
        "actual_failures": _string_list(row.get("actual_failures")),
    }


def _build_real_world_validation_followup_plan(
    *,
    rows: Sequence[Mapping[str, Any]],
    generated_at: str,
    outcome_source: str | None,
    calibration_status: str,
) -> Dict[str, Any]:
    actions: List[Dict[str, Any]] = []
    blockers: List[str] = []

    def add_action(
        row: Mapping[str, Any],
        *,
        action_type: str,
        reasons: Sequence[str],
        details: Mapping[str, Any],
    ) -> None:
        record_id = _string(row.get("record_id")) or "deployment_outcome"
        actions.append(
            {
                "action_id": _safe_id(f"{record_id}_{action_type}_{len(actions) + 1:04d}"),
                "action_type": action_type,
                "status": "queued_for_review",
                "reasons": sorted(set(_string_list(reasons))),
                **_followup_action_context(row),
                **dict(details),
                "claim_boundary": (
                    "followup_action_is_deterministic_plan_not_proof_of_rerun_or_fix"
                ),
            }
        )

    for row in rows:
        missed_failures = _string_list(row.get("missed_failures"))
        unmatched_actual = not bool(row.get("matched_prediction"))
        weak_prediction_match = bool(row.get("matched_prediction")) and not bool(
            row.get("exact_prediction_match")
        )
        actual_failed = row.get("actual_success") is False
        missing_actual_signal = not bool(row.get("actual_result_signal_present"))
        rerun_reasons: List[str] = []
        if actual_failed:
            rerun_reasons.append("actual_failed")
        if missed_failures:
            rerun_reasons.append("missed_failures")
        if unmatched_actual:
            rerun_reasons.append("unmatched_actual")
        if weak_prediction_match:
            rerun_reasons.append("weak_prediction_match")
        if missing_actual_signal:
            rerun_reasons.append("missing_actual_result_signal")
        if rerun_reasons:
            add_action(
                row,
                action_type="rerun_scenario_eval",
                reasons=rerun_reasons,
                details={
                    "recommended_next_step": (
                        "rerun_policy_on_same_task_scenario_variation_after_review"
                    ),
                    "rerun_inputs": {
                        "scenario_eval_run_id": row.get("scenario_eval_run_id"),
                        "scenario_variation_instance_id": row.get("scenario_variation_instance_id"),
                        "task_id": row.get("task_id"),
                        "scenario_id": row.get("scenario_id"),
                        "policy_id": row.get("policy_id"),
                    },
                },
            )
        if missed_failures:
            add_action(
                row,
                action_type="update_scenario_library_for_missed_failures",
                reasons=("missed_failures",),
                details={
                    "missed_failures": missed_failures,
                    "recommended_library_change": (
                        "add_or_update_scenario_family_variation_for_missed_failures"
                    ),
                },
            )
        tuning_needed = (
            bool(row.get("real_world_tuning_needed"))
            or bool(_number(row.get("tuning_hours"), 0.0))
            or bool(int(_number(row.get("tuning_iterations"), 0.0) or 0))
        )
        if tuning_needed:
            add_action(
                row,
                action_type="robot_team_tuning_review",
                reasons=("real_world_tuning_needed",),
                details={
                    "tuning_hours": _number(row.get("tuning_hours"), 0.0),
                    "tuning_iterations": int(_number(row.get("tuning_iterations"), 0.0) or 0),
                    "tuning_notes": _string_list(row.get("tuning_notes")),
                    "recommended_next_step": (
                        "request_robot_team_tuning_notes_and_replay_updated_policy"
                    ),
                },
            )
        site_modifications = row.get("site_modifications") or []
        if not isinstance(site_modifications, list):
            site_modifications = []
        if site_modifications:
            add_action(
                row,
                action_type="site_modification_review",
                reasons=("site_modifications_recorded",),
                details={
                    "site_modifications": site_modifications,
                    "site_modifications_helped": row.get("site_modifications_helped"),
                    "recommended_next_step": (
                        "review_site_modification_effect_and_rerun_representative_scenarios"
                    ),
                },
            )
        if unmatched_actual:
            add_action(
                row,
                action_type="unmatched_actual_review",
                reasons=("unmatched_actual",),
                details={
                    "recommended_next_step": (
                        "repair_prediction_join_keys_before_using_calibration_score"
                    ),
                },
            )

    summary = {
        "action_count": len(actions),
        "scenario_rerun_count": sum(
            1 for action in actions if action.get("action_type") == "rerun_scenario_eval"
        ),
        "scenario_library_update_count": sum(
            1
            for action in actions
            if action.get("action_type") == "update_scenario_library_for_missed_failures"
        ),
        "robot_team_tuning_review_count": sum(
            1 for action in actions if action.get("action_type") == "robot_team_tuning_review"
        ),
        "site_modification_review_count": sum(
            1 for action in actions if action.get("action_type") == "site_modification_review"
        ),
        "unmatched_actual_review_count": sum(
            1 for action in actions if action.get("action_type") == "unmatched_actual_review"
        ),
    }
    status = (
        "not_requested" if not rows else "review_required" if actions else "no_followup_required"
    )
    return {
        "schema_version": REAL_WORLD_VALIDATION_FOLLOWUP_PLAN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "outcome_source": outcome_source,
        "calibration_status": calibration_status,
        "source_artifacts": {
            "deployment_outcome_ledger": "deployment_outcome_ledger.json",
            "sim_vs_real_calibration_report": "sim_vs_real_calibration_report.json",
            "prediction_vs_actual_deployment_summary": (
                "prediction_vs_actual_deployment_summary.json"
            ),
        },
        "summary": summary,
        "follow_up_actions": actions,
        "blockers": blockers,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "artifact_purpose": "real_world_validation_loop_followup_plan",
            "followup_plan_is_not_proof_of_rerun_tuning_or_site_modification_success": True,
            "requires_owner_review_before_public_claim_upgrade": True,
            "sim_only_beta_ranking_blocked": False,
            "external_real_world_calibration_not_requested": not bool(rows),
        },
    }


def build_deployment_validation_bundle(
    *,
    capture_root: str | Path,
    job_dir: str | Path,
    job_request: Mapping[str, Any],
    prediction_ledger: Mapping[str, Any],
    attempt_trace: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    """Ingest real deployment outcomes and compute sim-vs-real calibration."""

    capture_path = Path(capture_root).resolve()
    resolved_job_dir = Path(job_dir).resolve()
    payload, outcome_source = _load_actual_outcome_payload(
        capture_root=capture_path,
        job_dir=resolved_job_dir,
        job_request=job_request,
    )
    actual_records = _records_from_payload(payload)
    write_json(
        resolved_job_dir / "deployment_outcome_intake_manifest.json",
        {
            "schema_version": "deployment_outcome_intake_manifest.v1",
            "generated_at": generated_at,
            "status": "completed" if actual_records else "not_requested",
            "outcome_source": outcome_source,
            "record_count": len(actual_records),
            "real_world_outcome_records_present": bool(actual_records),
            "real_world_outcome_proven": False,
            "source_files": _string_list(_mapping(payload).get("source_files")),
            "blockers": _string_list(_mapping(payload).get("blockers")),
            "claim_boundary": {
                **dict(CLAIM_BOUNDARY),
                "real_world_outcome_records_present": bool(actual_records),
                "real_world_outcome_proven": False,
            },
        },
    )
    predictions = _prediction_index(prediction_ledger, attempt_trace)
    prediction_rows = _prediction_anchor_rows(prediction_ledger, attempt_trace)
    (
        prediction_anchor_index,
        prediction_conflict_ids,
        prediction_incomplete_rows,
    ) = _prediction_anchor_index(prediction_rows)
    rows: List[Dict[str, Any]] = []
    for index, actual in enumerate(actual_records, start=1):
        task_id = _string(actual.get("task_id") or actual.get("taskId"))
        scenario_id = _string(actual.get("scenario_id") or actual.get("scenarioId"))
        scenario_eval_run_id = _string(
            actual.get("scenario_eval_run_id") or actual.get("scenarioEvalRunId")
        )
        scenario_variation_instance_id = _string(
            actual.get("scenario_variation_instance_id")
            or actual.get("scenarioVariationInstanceId")
        )
        policy_id = _string(actual.get("policy_id") or actual.get("policyId"))
        anchor_key = (
            scenario_eval_run_id,
            policy_id,
            task_id,
            scenario_variation_instance_id,
        )
        prediction, prediction_match_level = _prediction_for_actual(
            predictions,
            task_id=task_id,
            scenario_id=scenario_id,
            scenario_eval_run_id=scenario_eval_run_id,
            scenario_variation_instance_id=scenario_variation_instance_id,
        )
        exact_prediction_join_key_present = bool(
            scenario_eval_run_id and scenario_variation_instance_id
        )
        exact_prediction_match = (
            exact_prediction_join_key_present
            and prediction_match_level == "scenario_eval_run_and_variation"
        )
        anchor_prediction = prediction_anchor_index.get(anchor_key)
        predicted_failures = _failure_ids(prediction, "failure_mode_ids", "predicted_failures")
        actual_failures = _failure_ids(actual, "failure_mode_ids", "actual_failures", "failures")
        predicted_success = _predicted_success(anchor_prediction or prediction)
        actual_success = _actual_success(actual)
        actual_result_signal_present = _actual_signal_present(actual)
        site_modifications = (
            actual.get("site_modifications") or actual.get("siteModifications") or []
        )
        owner_evidence = _outcome_owner_evidence(actual)
        unit_source = anchor_prediction or prediction
        row = {
            "record_id": _string(actual.get("outcome_id") or actual.get("record_id"))
            or f"deployment_outcome_{index:04d}",
            "task_id": task_id,
            "scenario_id": scenario_id,
            "scenario_eval_run_id": scenario_eval_run_id or None,
            "scenario_variation_instance_id": scenario_variation_instance_id or None,
            "matched_prediction_scenario_eval_run_id": _string(
                prediction.get("scenario_eval_run_id")
            )
            or None,
            "matched_prediction_scenario_variation_instance_id": _string(
                prediction.get("scenario_variation_instance_id")
            )
            or None,
            "variation_name": _string(actual.get("variation_name") or actual.get("variationName"))
            or _string(prediction.get("variation_name"))
            or None,
            "policy_id": policy_id,
            "checkpoint_id": _string(
                actual.get("checkpoint_id")
                or actual.get("checkpointId")
                or actual.get("policy_checkpoint_id")
                or actual.get("policyCheckpointId")
                or unit_source.get("checkpoint_id")
                or unit_source.get("checkpointId")
                or unit_source.get("policy_checkpoint_id")
                or unit_source.get("policyCheckpointId")
            )
            or None,
            "criterion_id": _string(
                actual.get("criterion_id")
                or actual.get("criterionId")
                or actual.get("success_criterion_id")
                or actual.get("successCriterionId")
                or unit_source.get("criterion_id")
                or unit_source.get("criterionId")
                or unit_source.get("success_criterion_id")
                or unit_source.get("successCriterionId")
            )
            or None,
            "registered_split": _string(
                actual.get("registered_split")
                or actual.get("registeredSplit")
                or actual.get("evaluation_split")
                or actual.get("evaluationSplit")
                or actual.get("split")
                or unit_source.get("registered_split")
                or unit_source.get("registeredSplit")
                or unit_source.get("evaluation_split")
                or unit_source.get("evaluationSplit")
                or unit_source.get("split")
            )
            or None,
            "task_family": _string(
                actual.get("task_family")
                or actual.get("taskFamily")
                or actual.get("registered_task_family")
                or actual.get("registeredTaskFamily")
                or unit_source.get("task_family")
                or unit_source.get("taskFamily")
                or unit_source.get("registered_task_family")
                or unit_source.get("registeredTaskFamily")
            )
            or None,
            "matched_initial_condition_id": _string(
                actual.get("matched_initial_condition_id")
                or actual.get("matchedInitialConditionId")
                or actual.get("initial_condition_id")
                or actual.get("initialConditionId")
                or unit_source.get("matched_initial_condition_id")
                or unit_source.get("matchedInitialConditionId")
                or unit_source.get("initial_condition_id")
                or unit_source.get("initialConditionId")
            )
            or None,
            "anchor_schema_version": ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION,
            "anchor_join_key": _anchor_key_dict(anchor_key),
            "accepted_anchor_join_key_present": not _missing_anchor_key_fields(anchor_key),
            "strict_anchor_prediction_match": bool(anchor_prediction),
            "anchor_status": _string(
                actual.get("anchor_status")
                or actual.get("anchorStatus")
                or actual.get("validation_status")
                or actual.get("validationStatus")
                or actual.get("review_status")
                or actual.get("reviewStatus")
            )
            or None,
            "stale": actual.get("stale")
            if actual.get("stale") is not None
            else actual.get("is_stale")
            if actual.get("is_stale") is not None
            else actual.get("isStale"),
            "prediction_source": (anchor_prediction or prediction).get("source")
            or (anchor_prediction or prediction).get("prediction_source"),
            "predicted_success": predicted_success,
            "actual_success": actual_success,
            "actual_result_signal_present": actual_result_signal_present,
            "predicted_failures": predicted_failures,
            "actual_failures": actual_failures,
            "missed_failures": sorted(set(actual_failures) - set(predicted_failures)),
            "false_alarm_failures": sorted(set(predicted_failures) - set(actual_failures)),
            "predicted_cycle_time_seconds": _number(
                prediction.get("predicted_cycle_time_seconds")
                or prediction.get("cycle_time_seconds")
            ),
            "actual_cycle_time_seconds": _number(
                actual.get("cycle_time_seconds") or actual.get("actualCycleTimeSeconds")
            ),
            "intervention_count": _number(
                actual.get("intervention_count") or actual.get("interventions"),
                0.0,
            ),
            "real_world_tuning_needed": bool(
                actual.get("real_world_tuning_needed")
                or actual.get("realWorldTuningNeeded")
                or actual.get("tuning_notes")
            ),
            "tuning_iterations": int(_number(actual.get("tuning_iterations"), 0.0) or 0),
            "tuning_hours": _number(actual.get("tuning_hours") or actual.get("tuningHours"), 0.0),
            "tuning_notes": _string_list(actual.get("tuning_notes") or actual.get("tuningNotes")),
            "site_modifications": site_modifications
            if isinstance(site_modifications, list)
            else [],
            "site_modifications_helped": actual.get("site_modifications_helped")
            if actual.get("site_modifications_helped") is not None
            else actual.get("siteModificationsHelped"),
            "evidence_refs": owner_evidence["owner_evidence_refs"],
            "owner_evidence_refs": owner_evidence["owner_evidence_refs"],
            "owner_evidence_uri": owner_evidence["owner_evidence_uri"],
            "operator_attestation": owner_evidence["operator_attestation"],
            "owner_attestation": owner_evidence["owner_attestation"],
            "owner_evidence_present": owner_evidence["owner_evidence_present"],
            "signed_operator_attestation_present": owner_evidence[
                "signed_operator_attestation_present"
            ],
            "physical_evidence_required": _physical_evidence_requested(actual),
            "physical_evidence_present": _physical_evidence_present(actual),
            "matched_prediction": bool(prediction),
            "prediction_match_level": prediction_match_level,
            "exact_prediction_join_key_present": exact_prediction_join_key_present,
            "exact_prediction_match": exact_prediction_match,
            "claim_boundary": "real_world_outcome_requires_owner_system_evidence_review",
        }
        rows.append(row)

    study_design = _mapping(
        job_request.get("rank_fidelity_study_design")
        or job_request.get("rankFidelityStudyDesign")
        or job_request.get("sc3_study_design")
        or job_request.get("sc3StudyDesign")
        or _mapping(payload).get("rank_fidelity_study_design")
    )
    anchor_calibration = _accepted_anchor_calibration(
        rows=rows,
        prediction_rows=prediction_rows,
        prediction_anchor_index=prediction_anchor_index,
        prediction_conflict_ids=prediction_conflict_ids,
        prediction_incomplete_rows=prediction_incomplete_rows,
        study_design=study_design,
    )
    score = anchor_calibration.get("sim_vs_real_calibration_score")
    rank_fidelity_claim_eligibility = _mapping(
        anchor_calibration.get("rank_fidelity_claim_eligibility")
    )
    public_rank_fidelity_claim_eligible = (
        rank_fidelity_claim_eligibility.get("public_rank_fidelity_claim_eligible") is True
    )
    status = "completed" if rows else "not_requested"
    owner_evidence_record_count = sum(1 for row in rows if row.get("owner_evidence_present"))
    missing_owner_evidence_record_ids = [
        _string(row.get("record_id")) for row in rows if not row.get("owner_evidence_present")
    ]
    missing_actual_signal_record_ids = [
        _string(row.get("record_id")) for row in rows if not row.get("actual_result_signal_present")
    ]
    unmatched_actual_record_ids = [
        _string(row.get("record_id")) for row in rows if not row.get("matched_prediction")
    ]
    missing_exact_prediction_join_key_record_ids = [
        _string(row.get("record_id"))
        for row in rows
        if not row.get("exact_prediction_join_key_present")
    ]
    weak_prediction_match_record_ids = [
        _string(row.get("record_id"))
        for row in rows
        if row.get("matched_prediction") and not row.get("exact_prediction_match")
    ]
    exact_prediction_record_count = sum(1 for row in rows if row.get("exact_prediction_match"))
    prediction_match_counts = {
        "scenario_eval_run_and_variation": sum(
            1
            for row in rows
            if row.get("prediction_match_level") == "scenario_eval_run_and_variation"
        ),
        "scenario_eval_run": sum(
            1 for row in rows if row.get("prediction_match_level") == "scenario_eval_run"
        ),
        "scenario_variation_instance": sum(
            1 for row in rows if row.get("prediction_match_level") == "scenario_variation_instance"
        ),
        "task_scenario_fallback": sum(
            1 for row in rows if row.get("prediction_match_level") == "task_scenario_fallback"
        ),
        "unmatched": sum(1 for row in rows if row.get("prediction_match_level") == "unmatched"),
    }
    real_world_outcome_records_present = bool(rows)
    real_world_outcome_proven = (
        bool(rows)
        and owner_evidence_record_count == len(rows)
        and not missing_actual_signal_record_ids
    )
    outcome_blockers: List[str] = []
    if rows and missing_owner_evidence_record_ids:
        outcome_blockers.append("deployment_outcomes_missing_owner_evidence")
    if missing_actual_signal_record_ids:
        outcome_blockers.append("deployment_outcomes_missing_actual_result_signal")
    if unmatched_actual_record_ids:
        outcome_blockers.append("deployment_outcomes_missing_matching_prediction")
    if missing_exact_prediction_join_key_record_ids:
        outcome_blockers.append("deployment_outcomes_missing_exact_prediction_join_keys")
    calibration_status = _string(anchor_calibration.get("status")) or (
        "not_measured" if not rows else "blocked_insufficient_anchor_count"
    )
    calibration_blockers = _string_list(anchor_calibration.get("blockers"))
    if (
        calibration_status == "blocked_anchor_quality"
        and weak_prediction_match_record_ids
        and exact_prediction_record_count == 0
        and (
            (missing_exact_prediction_join_key_record_ids and not missing_owner_evidence_record_ids)
            or outcome_source == "deployment_outcome_inbox"
        )
    ):
        calibration_status = "blocked_weak_prediction_matches"
    ledger = {
        "schema_version": DEPLOYMENT_OUTCOME_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "outcome_source": outcome_source,
        "record_count": len(rows),
        "records": rows,
        "matched_prediction_record_count": len(rows) - len(unmatched_actual_record_ids),
        "exact_prediction_record_count": exact_prediction_record_count,
        "missing_exact_prediction_join_key_record_count": len(
            missing_exact_prediction_join_key_record_ids
        ),
        "missing_exact_prediction_join_key_record_ids": (
            missing_exact_prediction_join_key_record_ids
        ),
        "weak_prediction_match_record_count": len(weak_prediction_match_record_ids),
        "weak_prediction_match_record_ids": weak_prediction_match_record_ids,
        "unmatched_actual_record_count": len(unmatched_actual_record_ids),
        "unmatched_actual_record_ids": unmatched_actual_record_ids,
        "accepted_anchor_count": anchor_calibration.get("accepted_anchor_count"),
        "accepted_anchor_blockers": calibration_blockers,
        "rank_fidelity_claim_eligibility": rank_fidelity_claim_eligibility,
        "public_rank_fidelity_claim_eligible": public_rank_fidelity_claim_eligible,
        "prediction_match_counts": prediction_match_counts,
        "real_world_outcome_records_present": real_world_outcome_records_present,
        "owner_evidence_record_count": owner_evidence_record_count,
        "missing_owner_evidence_record_ids": missing_owner_evidence_record_ids,
        "missing_actual_result_signal_record_ids": missing_actual_signal_record_ids,
        "real_world_outcome_proven": real_world_outcome_proven,
        "blockers": [],
        "diagnostic_blockers": outcome_blockers,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "real_world_outcome_records_present": real_world_outcome_records_present,
            "real_world_outcome_proven": real_world_outcome_proven,
            "sim_only_beta_ranking_blocked": False,
            "external_real_world_calibration_not_requested": not real_world_outcome_records_present,
        },
    }
    report = {
        "schema_version": SIM_VS_REAL_CALIBRATION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": calibration_status,
        "outcome_source": outcome_source,
        "accepted_anchor_schema": anchor_calibration.get("accepted_anchor_schema"),
        "paired_record_count": len(
            [
                row
                for row in rows
                if row.get("predicted_success") is not None
                and row.get("actual_success") is not None
            ]
        ),
        "accepted_anchor_count": anchor_calibration.get("accepted_anchor_count"),
        "minimum_accepted_anchor_count": anchor_calibration.get("minimum_accepted_anchor_count"),
        "policy_group_count": anchor_calibration.get("policy_group_count"),
        "minimum_policy_group_count": anchor_calibration.get("minimum_policy_group_count"),
        "policy_checkpoint_group_count": anchor_calibration.get("policy_checkpoint_group_count"),
        "minimum_policy_checkpoint_count_for_public_rank_fidelity": (
            anchor_calibration.get("minimum_policy_checkpoint_count_for_public_rank_fidelity")
        ),
        "sim_vs_real_calibration_score": score,
        "spearman_rank_correlation": anchor_calibration.get("spearman_rank_correlation"),
        "pearson_success_rate_correlation": anchor_calibration.get(
            "pearson_success_rate_correlation"
        ),
        "mean_maximum_rank_violation": anchor_calibration.get("mean_maximum_rank_violation"),
        "mmrv": anchor_calibration.get("mmrv"),
        "mmrv_definition": anchor_calibration.get("mmrv_definition"),
        "maximum_pairwise_real_margin_rank_violation": anchor_calibration.get(
            "maximum_pairwise_real_margin_rank_violation"
        ),
        "mean_normalized_rank_position_error": anchor_calibration.get(
            "mean_normalized_rank_position_error"
        ),
        "maximum_normalized_rank_position_error": anchor_calibration.get(
            "maximum_normalized_rank_position_error"
        ),
        "mean_absolute_success_rate_error": anchor_calibration.get(
            "mean_absolute_success_rate_error"
        ),
        "confidence_intervals": anchor_calibration.get("confidence_intervals") or {},
        "unit_of_analysis_fields": anchor_calibration.get("unit_of_analysis_fields")
        or list(UNIT_OF_ANALYSIS_FIELDS),
        "estimands": anchor_calibration.get("estimands") or {},
        "rank_fidelity_claim_eligibility": rank_fidelity_claim_eligibility,
        "matched_prediction_record_count": len(rows) - len(unmatched_actual_record_ids),
        "exact_prediction_record_count": exact_prediction_record_count,
        "missing_exact_prediction_join_key_record_count": len(
            missing_exact_prediction_join_key_record_ids
        ),
        "missing_exact_prediction_join_key_record_ids": (
            missing_exact_prediction_join_key_record_ids
        ),
        "weak_prediction_match_record_count": len(weak_prediction_match_record_ids),
        "weak_prediction_match_record_ids": weak_prediction_match_record_ids,
        "unmatched_actual_record_count": len(unmatched_actual_record_ids),
        "unmatched_actual_record_ids": unmatched_actual_record_ids,
        "accepted_anchors": anchor_calibration.get("accepted_anchors") or [],
        "rejected_anchors": anchor_calibration.get("rejected_anchors") or [],
        "policy_success_rate_rows": anchor_calibration.get("policy_success_rate_rows") or [],
        "unmatched_prediction_row_count": anchor_calibration.get("unmatched_prediction_row_count"),
        "unmatched_prediction_rows": anchor_calibration.get("unmatched_prediction_rows") or [],
        "stale_anchor_row_count": anchor_calibration.get("stale_anchor_row_count"),
        "stale_anchor_row_ids": anchor_calibration.get("stale_anchor_row_ids") or [],
        "conflicting_anchor_row_count": anchor_calibration.get("conflicting_anchor_row_count"),
        "conflicting_anchor_row_ids": anchor_calibration.get("conflicting_anchor_row_ids") or [],
        "blockers": [],
        "diagnostic_blockers": calibration_blockers,
        "prediction_match_counts": prediction_match_counts,
        "missed_failure_count": sum(len(_string_list(row.get("missed_failures"))) for row in rows),
        "false_alarm_failure_count": sum(
            len(_string_list(row.get("false_alarm_failures"))) for row in rows
        ),
        "site_modification_count": sum(len(row.get("site_modifications") or []) for row in rows),
        "tuning_hours_total": round(
            sum(_number(row.get("tuning_hours"), 0.0) or 0.0 for row in rows), 4
        ),
        "records": rows,
        "real_world_outcome_records_present": real_world_outcome_records_present,
        "owner_evidence_record_count": owner_evidence_record_count,
        "missing_owner_evidence_record_ids": missing_owner_evidence_record_ids,
        "missing_actual_result_signal_record_ids": missing_actual_signal_record_ids,
        "real_world_outcome_proven": real_world_outcome_proven,
        "rank_fidelity_result_proven": public_rank_fidelity_claim_eligible,
        "public_rank_fidelity_claim_eligible": public_rank_fidelity_claim_eligible,
        "deployment_accuracy_claim_supported": False,
        "real_world_success_rate_prediction_claim_supported": False,
        "sim_only_beta_ranking_blocked": False,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "real_world_outcome_records_present": real_world_outcome_records_present,
            "real_world_outcome_proven": real_world_outcome_proven,
            "public_rank_fidelity_claim_eligible": public_rank_fidelity_claim_eligible,
            "deployment_accuracy_claim_supported": False,
            "real_world_success_rate_prediction_claim_supported": False,
            "sim_only_beta_ranking_blocked": False,
            "external_real_world_calibration_not_requested": not real_world_outcome_records_present,
        },
    }
    summary = {
        "schema_version": PREDICTION_VS_ACTUAL_DEPLOYMENT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": calibration_status,
        "outcome_source": outcome_source,
        "what_eval_predicted": [
            {
                "task_id": row.get("task_id"),
                "scenario_id": row.get("scenario_id"),
                "scenario_eval_run_id": row.get("scenario_eval_run_id"),
                "scenario_variation_instance_id": row.get("scenario_variation_instance_id"),
                "variation_name": row.get("variation_name"),
                "predicted_success": row.get("predicted_success"),
                "predicted_failures": row.get("predicted_failures"),
                "prediction_match_level": row.get("prediction_match_level"),
                "exact_prediction_join_key_present": row.get("exact_prediction_join_key_present"),
                "exact_prediction_match": row.get("exact_prediction_match"),
            }
            for row in rows
        ],
        "what_actually_happened": [
            {
                "task_id": row.get("task_id"),
                "scenario_id": row.get("scenario_id"),
                "scenario_eval_run_id": row.get("scenario_eval_run_id"),
                "scenario_variation_instance_id": row.get("scenario_variation_instance_id"),
                "variation_name": row.get("variation_name"),
                "actual_success": row.get("actual_success"),
                "actual_failures": row.get("actual_failures"),
                "prediction_match_level": row.get("prediction_match_level"),
                "exact_prediction_join_key_present": row.get("exact_prediction_join_key_present"),
                "exact_prediction_match": row.get("exact_prediction_match"),
            }
            for row in rows
        ],
        "which_scenarios_predicted_failure": [
            row.get("scenario_id") for row in rows if row.get("predicted_success") is False
        ],
        "which_failures_were_missed": [
            {
                "scenario_id": row.get("scenario_id"),
                "scenario_eval_run_id": row.get("scenario_eval_run_id"),
                "scenario_variation_instance_id": row.get("scenario_variation_instance_id"),
                "variation_name": row.get("variation_name"),
                "missed_failures": row.get("missed_failures"),
            }
            for row in rows
            if row.get("missed_failures")
        ],
        "how_much_real_world_tuning_was_needed": {
            "tuning_hours_total": report["tuning_hours_total"],
            "tuning_iterations_total": sum(int(row.get("tuning_iterations") or 0) for row in rows),
            "records_with_tuning": sum(1 for row in rows if row.get("real_world_tuning_needed")),
        },
        "whether_site_modifications_helped": [
            {
                "scenario_id": row.get("scenario_id"),
                "scenario_eval_run_id": row.get("scenario_eval_run_id"),
                "scenario_variation_instance_id": row.get("scenario_variation_instance_id"),
                "variation_name": row.get("variation_name"),
                "site_modifications": row.get("site_modifications"),
                "site_modifications_helped": row.get("site_modifications_helped"),
            }
            for row in rows
            if row.get("site_modifications")
        ],
        "sim_vs_real_calibration_score": score,
        "accepted_anchor_count": anchor_calibration.get("accepted_anchor_count"),
        "minimum_accepted_anchor_count": anchor_calibration.get("minimum_accepted_anchor_count"),
        "policy_group_count": anchor_calibration.get("policy_group_count"),
        "policy_checkpoint_group_count": anchor_calibration.get("policy_checkpoint_group_count"),
        "spearman_rank_correlation": anchor_calibration.get("spearman_rank_correlation"),
        "pearson_success_rate_correlation": anchor_calibration.get(
            "pearson_success_rate_correlation"
        ),
        "mean_maximum_rank_violation": anchor_calibration.get("mean_maximum_rank_violation"),
        "mmrv": anchor_calibration.get("mmrv"),
        "mmrv_definition": anchor_calibration.get("mmrv_definition"),
        "mean_normalized_rank_position_error": anchor_calibration.get(
            "mean_normalized_rank_position_error"
        ),
        "mean_absolute_success_rate_error": anchor_calibration.get(
            "mean_absolute_success_rate_error"
        ),
        "confidence_intervals": anchor_calibration.get("confidence_intervals") or {},
        "unit_of_analysis_fields": anchor_calibration.get("unit_of_analysis_fields")
        or list(UNIT_OF_ANALYSIS_FIELDS),
        "estimands": anchor_calibration.get("estimands") or {},
        "rank_fidelity_claim_eligibility": rank_fidelity_claim_eligibility,
        "public_rank_fidelity_claim_eligible": public_rank_fidelity_claim_eligible,
        "matched_prediction_record_count": len(rows) - len(unmatched_actual_record_ids),
        "exact_prediction_record_count": exact_prediction_record_count,
        "missing_exact_prediction_join_key_record_count": len(
            missing_exact_prediction_join_key_record_ids
        ),
        "missing_exact_prediction_join_key_record_ids": (
            missing_exact_prediction_join_key_record_ids
        ),
        "weak_prediction_match_record_count": len(weak_prediction_match_record_ids),
        "weak_prediction_match_record_ids": weak_prediction_match_record_ids,
        "unmatched_actual_record_count": len(unmatched_actual_record_ids),
        "unmatched_actual_record_ids": unmatched_actual_record_ids,
        "unmatched_prediction_row_count": anchor_calibration.get("unmatched_prediction_row_count"),
        "unmatched_prediction_rows": anchor_calibration.get("unmatched_prediction_rows") or [],
        "stale_anchor_row_count": anchor_calibration.get("stale_anchor_row_count"),
        "stale_anchor_row_ids": anchor_calibration.get("stale_anchor_row_ids") or [],
        "conflicting_anchor_row_count": anchor_calibration.get("conflicting_anchor_row_count"),
        "conflicting_anchor_row_ids": anchor_calibration.get("conflicting_anchor_row_ids") or [],
        "blockers": [],
        "diagnostic_blockers": calibration_blockers,
        "prediction_match_counts": prediction_match_counts,
        "real_world_outcome_records_present": real_world_outcome_records_present,
        "owner_evidence_record_count": owner_evidence_record_count,
        "missing_owner_evidence_record_ids": missing_owner_evidence_record_ids,
        "missing_actual_result_signal_record_ids": missing_actual_signal_record_ids,
        "real_world_outcome_proven": real_world_outcome_proven,
        "real_world_validation_followup_plan_path": ("real_world_validation_followup_plan.json"),
        "claim_boundary": report["claim_boundary"],
    }
    followup_plan = _build_real_world_validation_followup_plan(
        rows=rows,
        generated_at=generated_at,
        outcome_source=outcome_source,
        calibration_status=calibration_status,
    )
    write_json(resolved_job_dir / "deployment_outcome_ledger.json", ledger)
    write_json(resolved_job_dir / "sim_vs_real_calibration_report.json", report)
    write_json(resolved_job_dir / "prediction_vs_actual_deployment_summary.json", summary)
    write_json(resolved_job_dir / "real_world_validation_followup_plan.json", followup_plan)
    return {
        "ledger": ledger,
        "calibration_report": report,
        "summary": summary,
        "followup_plan": followup_plan,
    }


def fingerprint_execution_artifacts(*payloads: Mapping[str, Any]) -> str:
    encoded = json.dumps([dict(payload) for payload in payloads], sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
