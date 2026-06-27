"""Build a fail-closed Isaac Sim Unitree G1 lane for 3DGS site assets.

This module prepares the realistic Isaac/3DGS/G1 artifact contract without
promoting support artifacts into simulator, controller, WAM/VLA, or deployment
proof. When Isaac Sim or a real controller runtime is unavailable, it still
writes the full job bundle with blocked attempts and exact proof boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .g1_site_3dgs_mujoco_preview import (
    _attempt_visual_asset_mesh_conversion,
    _safe_id,
    build_default_spawns,
    build_default_tasks,
    build_scenario_eval_matrix,
    build_scene_proxies_from_metadata,
    inspect_optional_scene_metadata,
    inspect_scene_asset,
    validate_spawns,
)


ISAAC_REALISTIC_JOB_SCHEMA_VERSION = "isaac_g1_site_3dgs_realistic_eval_job.v1"
ISAAC_RUNTIME_DISCOVERY_SCHEMA_VERSION = "isaac_runtime_discovery.v1"
ISAAC_PROVIDER_PLAN_SCHEMA_VERSION = "isaac_provider_plan.v1"
ISAAC_PHASE_LOG_SCHEMA_VERSION = "isaac_runtime_phase_log.v1"
ISAAC_COST_LEDGER_SCHEMA_VERSION = "isaac_gpu_cost_control_ledger.v1"
ISAAC_TEARDOWN_SCHEMA_VERSION = "isaac_teardown_manifest.v1"
ISAAC_SCENE_INSPECTION_SCHEMA_VERSION = "isaac_scene_asset_inspection.v1"
SPLAT_VISUAL_RENDER_SCHEMA_VERSION = "splat_visual_render_manifest.v1"
USD_SCENE_ASSEMBLY_SCHEMA_VERSION = "usd_scene_assembly_manifest.v1"
COLLIDER_PROXY_SCHEMA_VERSION = "isaac_collider_proxy_plan.v1"
SCENE_CONVERSION_SCHEMA_VERSION = "isaac_scene_conversion_report.v1"
VISUAL_COLLISION_ALIGNMENT_SCHEMA_VERSION = "visual_collision_alignment_manifest.v1"
VISUAL_TRUTH_BOUNDARY_SCHEMA_VERSION = "visual_truth_boundary.v1"
G1_ASSET_SOURCE_SCHEMA_VERSION = "unitree_g1_asset_source_manifest.v1"
G1_CONTROLLER_RUNTIME_SCHEMA_VERSION = "g1_controller_runtime_manifest.v1"
FOOT_CONTACT_TRACE_SCHEMA_VERSION = "foot_contact_trace.v1"
ROOT_MOTION_CONTINUITY_SCHEMA_VERSION = "root_motion_continuity_report.v1"
COLLISION_CONTACT_REPORT_SCHEMA_VERSION = "collision_contact_report.v1"
CONTROLLER_GRADE_PROOF_SCHEMA_VERSION = "controller_grade_proof_manifest.v1"
MANIPULATION_OBJECT_SCHEMA_VERSION = "manipulation_scene_object_manifest.v1"
MANIPULATION_ACTION_SCHEMA_VERSION = "manipulation_action_spec_manifest.v1"
MANIPULATION_CONTACT_TRACE_SCHEMA_VERSION = "manipulation_contact_trace.v1"
OBJECT_MOTION_TRACE_SCHEMA_VERSION = "object_motion_trace.v1"
MANIPULATION_EVAL_SCHEMA_VERSION = "manipulation_success_evaluator_results.v1"
MANIPULATION_TRUTH_BOUNDARY_SCHEMA_VERSION = "manipulation_truth_boundary.v1"
WAM_VLA_DISCOVERY_SCHEMA_VERSION = "real_wam_vla_runtime_discovery.v1"
WAM_VLA_OBSERVATION_SCHEMA_VERSION = "wam_vla_observation_packet.v1"
WAM_VLA_OUTPUT_SCHEMA_VERSION = "wam_vla_policy_outputs.v1"
WAM_VLA_ACTION_TRACE_SCHEMA_VERSION = "wam_vla_action_trace.v1"
WAM_VLA_PROOF_SCHEMA_VERSION = "wam_vla_runtime_proof_manifest.v1"
WAM_VLA_TRUTH_BOUNDARY_SCHEMA_VERSION = "wam_vla_truth_boundary.v1"
CAMERA_MANIFEST_SCHEMA_VERSION = "isaac_g1_camera_manifest.v1"
EPISODE_SPEC_SCHEMA_VERSION = "isaac_g1_episode_spec_manifest.v1"
NORMALIZED_TRACE_SCHEMA_VERSION = "robot_eval_job_normalized_attempt_trace.v1"
FAILURE_LABELS_SCHEMA_VERSION = "robot_eval_job_failure_labels.v1"
WAM_EVALUATOR_THRESHOLDS_SCHEMA_VERSION = "wam_evaluator_thresholds.v1"
WAM_EVALUATOR_TRACE_BINDING_SCHEMA_VERSION = "wam_evaluator_trace_binding.v1"
WAM_EVALUATOR_RESULTS_SCHEMA_VERSION = "wam_evaluator_results.v1"
POLICY_EVALUATION_SUMMARY_SCHEMA_VERSION = "policy_evaluation_summary.v1"
REALISTIC_VIDEO_MANIFEST_SCHEMA_VERSION = "realistic_video_manifest.v1"
ISAAC_PROVIDER_RUNTIME_BUNDLE_SCHEMA_VERSION = "isaac_provider_runtime_bundle.v1"
ISAAC_PROVIDER_EVAL_MANIFEST_SCHEMA_VERSION = "isaac_provider_eval_manifest.v1"
LOCAL_VALIDATION_REPORT_SCHEMA_VERSION = "isaac_realistic_local_validation_report.v1"
LOCAL_PROVIDER_COMMAND_DIAGNOSTIC_SCHEMA_VERSION = (
    "isaac_provider_command_local_diagnostic.v1"
)
ISAAC_PROVIDER_BUNDLE_READINESS_SCHEMA_VERSION = "isaac_provider_bundle_readiness.v1"
ISAAC_G1_SIMULATOR_COMMAND_OUTPUT_SCHEMA_VERSION = "isaac_g1_simulator_command_output.v1"
ISAAC_G1_ARTIFACT_MANIFEST_SCHEMA_VERSION = "isaac_g1_simulator_artifact_manifest.v1"
ISAAC_G1_BATCH_TRACE_PACKAGE_SCHEMA_VERSION = "isaac_g1_batch_trace_package.v1"
ISAAC_G1_BATCH_CLOSURE_SCHEMA_VERSION = "isaac_g1_batch_closure_manifest.v1"

OFFICIAL_ISAAC_G1_ASSET_PATH = "Isaac/Robots/Unitree/G1/g1.usd"
OFFICIAL_ISAAC_G1_DOC_URL = (
    "https://docs.isaacsim.omniverse.nvidia.com/5.0.0/assets/usd_assets_robots.html"
)
DEFAULT_CAMERA_IDS = (
    "head_pov",
    "torso",
    "wrist",
    "third_person",
    "overhead",
    "task_focus",
)
CAMERA_ID_ALIASES = {
    "overview": "third_person",
    "sim_robot_follow_pov": "head_pov",
}
REQUIRED_PHASES = (
    "runner_referencing_official_g1",
    "runner_official_g1_resolved",
    "runner_official_g1_reference_added",
    "runner_robot_api_evidence_collected",
    "runner_scene_loaded",
    "runner_episode_execution_started",
    "runner_episode_execution_completed",
    "runner_artifacts_exported",
    "runner_gpu_teardown_completed",
)
DEFAULT_ISAAC_RUNTIME_IMAGE_REF = "nvcr.io/nvidia/isaac-sim:6.0.0"
ISAAC_WORKER_IMAGE_REF_ENV = "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF"
ISAAC_WORKER_IMAGE_REF_FILE_ENV = "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF_FILE"
DEFAULT_ISAAC_WORKER_IMAGE_REF_FILE = "~/.blueprint-secrets/isaac_eval_worker_image_ref"
ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV = (
    "BLUEPRINT_ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC"
)
DEFAULT_ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC = (
    "output/isaac_worker_image_manifest_diagnostic.json"
)
ALLOW_DIRECT_ISAAC_BASE_IMAGE_RUNPOD_ENV = "BLUEPRINT_ALLOW_DIRECT_ISAAC_BASE_IMAGE_RUNPOD"
RUNPOD_CONTAINER_REGISTRY_AUTH_ID_ENV = "BLUEPRINT_RUNPOD_CONTAINER_REGISTRY_AUTH_ID"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [item for item in (_string(item) for item in value) if item]
    return []


def _normalize_camera_ids(camera_ids: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    normalized: list[str] = []
    aliases: dict[str, str] = {}
    for raw_camera_id in camera_ids:
        camera_id = _string(raw_camera_id)
        if not camera_id:
            continue
        canonical_id = CAMERA_ID_ALIASES.get(camera_id, camera_id)
        if canonical_id != camera_id:
            aliases[camera_id] = canonical_id
        if canonical_id not in normalized:
            normalized.append(canonical_id)
    return normalized, aliases


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
            count += 1
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _json_mapping(path: Path) -> dict[str, Any]:
    try:
        return _mapping(read_json_any(path))
    except Exception:
        return {}


def _scenario_eval_matrix_from_path(
    path: str | Path,
    *,
    camera_ids: Sequence[str],
    job_id: str,
) -> dict[str, Any]:
    matrix_path = Path(path).expanduser().resolve()
    payload = _mapping(read_json_any(matrix_path))
    raw_runs = payload.get("runs")
    runs: list[dict[str, Any]] = []
    missing_id_indexes: list[int] = []
    run_ids: list[str] = []
    if isinstance(raw_runs, Sequence) and not isinstance(raw_runs, (str, bytes, bytearray)):
        for index, raw_run in enumerate(raw_runs, start=1):
            if not isinstance(raw_run, Mapping):
                missing_id_indexes.append(index)
                continue
            row = dict(raw_run)
            run_id = _string(row.get("scenario_eval_run_id") or row.get("scenarioEvalRunId"))
            if run_id:
                row["scenario_eval_run_id"] = run_id
                run_ids.append(run_id)
            else:
                missing_id_indexes.append(index)
            if not isinstance(row.get("camera_ids"), list):
                row["camera_ids"] = list(camera_ids)
            else:
                row["camera_ids"] = _normalize_camera_ids(
                    _string_list(row.get("camera_ids"))
                )[0]
            if not _string(row.get("episode_id")):
                row["episode_id"] = f"{job_id}_isaac_episode_{index:04d}"
            runs.append(row)

    declared_count = _int(payload.get("scenario_eval_run_count"), default=len(runs))
    duplicate_run_ids = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
    blockers: list[str] = []
    if not runs:
        blockers.append("scenario_eval_matrix_contains_no_runs")
    if missing_id_indexes:
        blockers.append("scenario_eval_matrix_missing_scenario_eval_run_id")
    if duplicate_run_ids:
        blockers.append("scenario_eval_matrix_duplicate_scenario_eval_run_id")
    if declared_count != len(runs):
        blockers.append("scenario_eval_matrix_declared_count_mismatch")
    if blockers:
        raise RuntimeError(
            "scenario_eval_matrix is not executable by Isaac command: " + ",".join(blockers)
        )

    return {
        **payload,
        "runs": runs,
        "scenario_eval_run_count": len(runs),
        "source_scenario_eval_matrix_path": str(matrix_path),
        "source_matrix_scenario_eval_run_count": declared_count,
        "matrix_declared_count_matches_rows": declared_count == len(runs),
        "required_scenario_eval_run_ids": run_ids,
        "missing_scenario_eval_run_id_indexes": missing_id_indexes,
        "duplicate_scenario_eval_run_ids": duplicate_run_ids,
    }


def _matrix_required_run_ids(matrix: Mapping[str, Any]) -> list[str]:
    required = _string_list(matrix.get("required_scenario_eval_run_ids"))
    if required:
        return required
    return [
        run_id
        for run_id in (
            _string(_mapping(run).get("scenario_eval_run_id"))
            for run in matrix.get("runs", []) or []
        )
        if run_id
    ]


def _coverage_summary(
    *,
    matrix: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required = _matrix_required_run_ids(matrix)
    covered = sorted(
        {
            _string(attempt.get("scenario_eval_run_id"))
            for attempt in attempts
            if _string(attempt.get("scenario_eval_run_id"))
        }
    )
    duplicates = sorted({run_id for run_id in required if required.count(run_id) > 1})
    missing = sorted(set(required) - set(covered))
    attempt_count_matches = len(attempts) == len(required)
    exact = (
        set(covered) == set(required)
        and len(covered) == len(required)
        and not duplicates
    )
    complete = bool(required) and attempt_count_matches and exact and not missing
    return {
        "required_scenario_eval_run_count": len(required),
        "covered_scenario_eval_run_count": len(covered),
        "missing_scenario_eval_run_count": len(missing),
        "attempt_count_matches_matrix_count": attempt_count_matches,
        "scenario_eval_run_id_coverage_exact": exact,
        "scenario_eval_run_coverage_complete": complete,
        "duplicate_scenario_eval_run_ids": duplicates,
        "required_scenario_eval_run_ids": required,
        "covered_scenario_eval_run_ids": covered,
        "missing_scenario_eval_run_ids": missing,
    }


def _file_ref(path: Path, *, base_dir: Path) -> dict[str, Any]:
    present = path.is_file()
    try:
        relative = os.path.relpath(path.resolve(), base_dir.resolve())
    except Exception:
        relative = str(path)
    return {
        "path": str(path),
        "relative_path": relative,
        "present": present,
        "size_bytes": path.stat().st_size if present else None,
        "sha256": _sha256(path) if present else None,
    }


def _runtime_result_attempts(runtime_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("attempts", "episodes", "results"):
        value = runtime_result.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _write_isaac_batch_trace_package(
    *,
    job_dir: Path,
    generated_at: str,
    attempts: Sequence[Mapping[str, Any]],
    failure_labels: Mapping[str, Any],
    video_manifest: Mapping[str, Any],
    collision_contact_report: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    attempt_trace_path = job_dir / "isaac_batch_attempt_trace.jsonl"
    contact_stream_path = job_dir / "isaac_batch_contact_stream.jsonl"
    planner_state_path = job_dir / "isaac_batch_planner_state.jsonl"
    control_stream_path = job_dir / "isaac_batch_control_stream.jsonl"
    metrics_path = job_dir / "isaac_batch_metrics.json"
    failure_labels_path = job_dir / "isaac_batch_failure_labels.json"
    visual_media_path = job_dir / "isaac_batch_visual_media_coverage.json"
    visual_review_path = job_dir / "isaac_batch_visual_review_ledger.json"
    checksums_path = job_dir / "isaac_batch_artifact_checksums.json"
    manifest_path = job_dir / "isaac_batch_trace_package_manifest.json"

    _write_jsonl(attempt_trace_path, attempts)
    contact_rows: list[dict[str, Any]] = []
    planner_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for attempt in attempts:
        attempt_id = _string(attempt.get("attempt_id"))
        for contact in attempt.get("contact_trace") or []:
            if isinstance(contact, Mapping):
                contact_rows.append({"attempt_id": attempt_id, **dict(contact)})
        planner_rows.append(
            {
                "attempt_id": attempt_id,
                "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
                "route_waypoints": attempt.get("route_waypoints") or [],
                "status": attempt.get("status"),
            }
        )
        for index, action in enumerate(attempt.get("actions") or attempt.get("action_trace") or []):
            if isinstance(action, Mapping):
                control_rows.append({"attempt_id": attempt_id, "action_index": index, **dict(action)})
    _write_jsonl(contact_stream_path, contact_rows)
    _write_jsonl(planner_state_path, planner_rows)
    _write_jsonl(control_stream_path, control_rows)

    metrics = {
        "schema_version": "isaac_g1_batch_metrics.v1",
        "generated_at": generated_at,
        "status": "completed" if attempts else "blocked",
        "attempt_count": len(attempts),
        "successful_attempt_count": sum(1 for attempt in attempts if attempt.get("success") is True),
        "failed_attempt_count": sum(1 for attempt in attempts if attempt.get("success") is not True),
        "contact_count": collision_contact_report.get("contact_count"),
        "collision_dynamics_validated": collision_contact_report.get("collision_dynamics_validated"),
        **dict(coverage),
        "proof_boundary": (
            "Isaac batch metrics summarize Isaac command/runtime attempts only. They do not "
            "prove MuJoCo execution, real robot readiness, safety validation, deployment "
            "approval, WAM consistency, or generated-world rank fidelity."
        ),
    }
    write_json(metrics_path, metrics)
    write_json(failure_labels_path, dict(failure_labels))
    visual_rows = []
    for video in video_manifest.get("videos") or []:
        if isinstance(video, Mapping):
            visual_rows.append(dict(video))
    visual_media = {
        "schema_version": "isaac_g1_batch_visual_media_coverage.v1",
        "generated_at": generated_at,
        "status": "completed" if video_manifest.get("video_count") else "blocked_no_runtime_videos",
        "video_count": video_manifest.get("video_count"),
        "expected_video_count": video_manifest.get("expected_video_count"),
        "all_required_runs_have_visual_recording": (
            int(video_manifest.get("video_count") or 0)
            >= int(video_manifest.get("expected_video_count") or 0)
            and int(video_manifest.get("expected_video_count") or 0) > 0
        ),
        "videos": visual_rows,
    }
    write_json(visual_media_path, visual_media)
    visual_review = {
        "schema_version": "isaac_g1_batch_visual_review_ledger.v1",
        "generated_at": generated_at,
        "status": "review_required" if attempts else "not_available",
        "review_count": len(attempts),
        "attempt_count": len(attempts),
        "accepted_review_count": len(attempts),
        "media_backed_review_count": sum(
            1
            for attempt in attempts
            if _mapping(attempt.get("artifact_paths")).get("video_path")
            or _mapping(attempt.get("artifact_paths")).get("robot_pov_video")
        ),
        "visual_review_coverage_complete": bool(attempts),
        "records": [
            {
                "review_id": f"isaac_visual_review_{_safe_id(attempt.get('attempt_id'))}",
                "attempt_id": attempt.get("attempt_id"),
                "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
                "decision": "success" if attempt.get("success") is True else "failure_or_blocked",
                "success": attempt.get("success") is True,
                "failure_label_ids": attempt.get("failure_label_ids") or [],
                "review_status": "accepted_runtime_trace_review_required_for_claim_upgrade",
            }
            for attempt in attempts
        ],
    }
    write_json(visual_review_path, visual_review)
    checksum_inputs = [
        attempt_trace_path,
        contact_stream_path,
        planner_state_path,
        control_stream_path,
        metrics_path,
        failure_labels_path,
        visual_media_path,
        visual_review_path,
    ]
    checksums = {
        "schema_version": "isaac_g1_batch_artifact_checksums.v1",
        "generated_at": generated_at,
        "files": {path.name: _file_ref(path, base_dir=job_dir) for path in checksum_inputs},
    }
    write_json(checksums_path, checksums)
    artifact_paths = {
        "attempt_trace_jsonl": str(attempt_trace_path),
        "contact_stream_jsonl": str(contact_stream_path),
        "planner_state_jsonl": str(planner_state_path),
        "control_stream_jsonl": str(control_stream_path),
        "metrics": str(metrics_path),
        "failure_labels": str(failure_labels_path),
        "visual_media_coverage": str(visual_media_path),
        "visual_review_ledger": str(visual_review_path),
        "artifact_checksums": str(checksums_path),
        "trace_package_manifest": str(manifest_path),
    }
    manifest = {
        "schema_version": ISAAC_G1_BATCH_TRACE_PACKAGE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if attempts else "blocked",
        "simulator_backend": "isaac_sim",
        "attempt_count": len(attempts),
        **dict(coverage),
        "metric_coverage_complete": bool(attempts),
        "failed_run_label_coverage_complete": True,
        "artifact_paths": artifact_paths,
        "proof_boundary": (
            "Trace package is Isaac simulator evidence and closure input only. It does not "
            "prove MuJoCo execution, physical robot readiness, safety validation, deployment "
            "approval, WAM consistency, or generated-world rank fidelity."
        ),
    }
    write_json(manifest_path, manifest)
    return {
        "manifest": manifest,
        "artifact_paths": artifact_paths,
        "metrics": metrics,
        "visual_media_coverage": visual_media,
        "visual_review_ledger": visual_review,
    }


def _build_isaac_batch_closure_manifest(
    *,
    job_dir: Path,
    generated_at: str,
    attempts: Sequence[Mapping[str, Any]],
    coverage: Mapping[str, Any],
    batch_trace_package: Mapping[str, Any],
    artifact_paths: Mapping[str, str],
    summary: Mapping[str, Any],
    video_manifest: Mapping[str, Any],
    collision_contact_report: Mapping[str, Any],
) -> dict[str, Any]:
    required_artifacts = {
        "normalized_attempt_trace": job_dir / "normalized_attempt_trace.json",
        "failure_labels": job_dir / "failure_labels.json",
        "policy_evaluation_summary": job_dir / "policy_evaluation_summary.json",
        "realistic_video_manifest": job_dir / "realistic_video_manifest.json",
        "g1_locomotion_trace_jsonl": job_dir / "g1_locomotion_trace.jsonl",
        "collision_contact_report": job_dir / "collision_contact_report.json",
        "job_run_manifest": job_dir / "job_run_manifest.json",
    }
    for key, path in artifact_paths.items():
        if key.startswith("batch_"):
            required_artifacts[key] = Path(path)
    artifact_presence = {
        key: _file_ref(path, base_dir=job_dir)
        for key, path in required_artifacts.items()
    }
    missing = [
        key for key, record in artifact_presence.items() if record.get("present") is not True
    ]
    runtime_attempt_rows = [
        attempt for attempt in attempts if attempt.get("status") != "blocked"
    ]
    contact_validated = bool(collision_contact_report.get("collision_dynamics_validated"))
    video_complete = (
        int(video_manifest.get("video_count") or 0)
        >= int(video_manifest.get("expected_video_count") or 0)
        and int(video_manifest.get("expected_video_count") or 0) > 0
    )
    machine_trace_package_complete = (
        bool(coverage.get("scenario_eval_run_coverage_complete"))
        and not missing
        and bool(batch_trace_package)
    )
    robot_team_grade_package_complete = (
        machine_trace_package_complete
        and bool(runtime_attempt_rows)
        and bool(summary.get("official_policy_execution_proven"))
        and bool(summary.get("controller_grade_execution_proven"))
        and contact_validated
        and video_complete
    )
    blockers: list[str] = []
    if not coverage.get("scenario_eval_run_coverage_complete"):
        blockers.append("scenario_eval_run_coverage_incomplete")
    if missing:
        blockers.append("isaac_required_artifacts_missing")
    if not runtime_attempt_rows:
        blockers.append("isaac_runtime_attempts_missing")
    if not summary.get("official_policy_execution_proven"):
        blockers.append("official_policy_execution_not_proven_by_isaac")
    if not summary.get("controller_grade_execution_proven"):
        blockers.append("controller_grade_execution_not_proven_by_isaac")
    if not contact_validated:
        blockers.append("isaac_contact_collision_dynamics_not_validated")
    if not video_complete:
        blockers.append("isaac_video_coverage_not_complete_for_all_runs")
    return {
        "schema_version": ISAAC_G1_BATCH_CLOSURE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed"
        if robot_team_grade_package_complete
        else "completed_with_robot_team_grade_blockers"
        if machine_trace_package_complete
        else "blocked",
        "simulator_backend": "isaac_sim",
        "machine_trace_package_complete": machine_trace_package_complete,
        "robot_team_grade_package_complete": robot_team_grade_package_complete,
        "blockers": sorted(set(blockers)),
        "attempt_count": len(attempts),
        **dict(coverage),
        "runtime_attempt_count": len(runtime_attempt_rows),
        "video_coverage_complete": video_complete,
        "contact_dynamics_validated": contact_validated,
        "artifact_presence": artifact_presence,
        "missing_required_artifacts": missing,
        "policy_interface_boundary": {
            "robot_team_policy_execution_proven": bool(
                summary.get("official_policy_execution_proven")
            ),
            "controller_grade_execution_proven": bool(
                summary.get("controller_grade_execution_proven")
            ),
            "training_grade_policy_rollout_proven": False,
        },
        "claim_boundary": {
            "simulator_backend": "isaac_sim",
            "mujoco_proof_counted_as_isaac_proof": False,
            "simulator_proof_is_not_safety_validation": True,
            "simulator_proof_is_not_physical_robot_readiness": True,
            "generated_world_rank_fidelity_result_proven": False,
            "deployment_approval_proven": False,
        },
    }


def _write_isaac_artifact_manifest(
    *,
    job_dir: Path,
    generated_at: str,
    artifact_paths: Mapping[str, str],
    summary: Mapping[str, Any],
    coverage: Mapping[str, Any],
    batch_closure_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    records = {
        key: _file_ref(Path(path), base_dir=job_dir)
        for key, path in sorted(artifact_paths.items())
        if path
    }
    manifest = {
        "schema_version": ISAAC_G1_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "complete" if records else "blocked_no_artifacts",
        "simulator_backend": "isaac_sim",
        "simulator_version": summary.get("simulator_version"),
        "attempt_count": summary.get("attempted_episode_count"),
        **dict(coverage),
        "artifacts": dict(artifact_paths),
        "files": records,
        "batch_closure_manifest": dict(batch_closure_manifest),
        "proof_boundary": {
            "artifact_manifest_is_not_runtime_proof_by_itself": True,
            "mujoco_artifacts_counted_as_isaac_proof": False,
            "simulator_proof_is_not_safety_validation": True,
            "simulator_proof_is_not_deployment_approval": True,
            "simulator_proof_is_not_physical_robot_readiness": True,
            "generated_world_rank_fidelity_result_proven": False,
        },
    }
    write_json(job_dir / "artifact_manifest.json", manifest)
    return manifest


def _usd_string(value: Any) -> str:
    return _string(value).replace("\\", "\\\\").replace('"', '\\"')


def _usd_identifier(value: Any) -> str:
    ident = _safe_id(value, fallback="item")
    if ident and ident[0].isdigit():
        return f"_{ident}"
    return ident or "item"


def _path_exists(path: str) -> bool:
    return bool(path) and Path(path).expanduser().exists()


def _which_record(command: str) -> dict[str, Any]:
    resolved = shutil.which(command)
    return {"command": command, "path": resolved, "present": bool(resolved)}


def _probe_version(executable: str | None) -> dict[str, Any]:
    if not executable:
        return {"attempted": False, "version": None}
    try:
        completed = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=6,
        )
    except Exception as exc:  # pragma: no cover - environment dependent.
        return {
            "attempted": True,
            "status": "blocked",
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }
    output = (completed.stdout or completed.stderr or "").strip()
    return {
        "attempted": True,
        "status": "completed" if completed.returncode == 0 else "blocked",
        "returncode": completed.returncode,
        "version": output.splitlines()[0] if output else None,
    }


def detect_isaac_runtime(*, generated_at: str | None = None) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    env_roots = {
        name: _string(os.getenv(name))
        for name in (
            "ISAAC_SIM_ROOT",
            "ISAACSIM_PATH",
            "OMNI_ISAAC_HOME",
            "EXP_PATH",
            "CARB_APP_PATH",
        )
    }
    root_candidates = [
        {"env_var": name, "path": value, "exists": _path_exists(value)}
        for name, value in env_roots.items()
        if value
    ]
    executable_candidates = [
        _which_record("isaacsim"),
        _which_record("isaac-sim.sh"),
        _which_record("python.sh"),
    ]
    common_paths = [
        "/Applications/Isaac Sim.app/Contents/MacOS/Isaac Sim",
        "/opt/nvidia/isaac-sim/isaac-sim.sh",
        "/isaac-sim/python.sh",
    ]
    executable_candidates.extend(
        {
            "command": path,
            "path": path if Path(path).exists() else None,
            "present": Path(path).exists(),
        }
        for path in common_paths
    )
    python_modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("isaacsim", "omni", "pxr")
    }
    selected_executable = next(
        (_string(item.get("path")) for item in executable_candidates if item.get("present")),
        "",
    )
    version_probe = _probe_version(selected_executable) if selected_executable else {
        "attempted": False,
        "version": None,
    }
    available = bool(
        selected_executable
        or any(item["exists"] for item in root_candidates)
        or python_modules.get("isaacsim")
    )
    blockers = [] if available else ["isaac_sim_runtime_unavailable"]
    if available and not selected_executable and not python_modules.get("isaacsim"):
        blockers.append("isaac_runtime_root_present_but_launch_surface_unverified")
    return {
        "schema_version": ISAAC_RUNTIME_DISCOVERY_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "available" if available and not blockers else "blocked",
        "host_platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "isaac_runtime_available": available and not blockers,
        "local_isaac_runtime_ready": available and not blockers,
        "selected_executable": selected_executable or None,
        "version_probe": version_probe,
        "version": version_probe.get("version"),
        "root_candidates": root_candidates,
        "executable_candidates": executable_candidates,
        "python_module_probe": python_modules,
        "blockers": blockers,
        "proof_boundary": (
            "Runtime discovery only proves local launch surfaces. Isaac scene loading, "
            "G1 asset loading, controller execution, contact dynamics, and video proof "
            "require separate runtime phase artifacts."
        ),
    }


def _secret_file_status(name: str, default_path: str) -> dict[str, Any]:
    explicit = _string(os.getenv(name))
    selected = explicit or default_path
    path = Path(selected).expanduser()
    return {
        "env_var": name,
        "path": str(path),
        "path_source": "env" if explicit else "default_blueprint_secret_file_path",
        "present": path.is_file(),
        "raw_secret_value_recorded": False,
    }


def _configured_isaac_worker_image_ref() -> dict[str, Any]:
    explicit = _string(os.getenv(ISAAC_WORKER_IMAGE_REF_ENV))
    if explicit:
        return {
            "image_ref": explicit,
            "source": ISAAC_WORKER_IMAGE_REF_ENV,
            "configured": True,
            "image_ref_file": None,
            "image_ref_file_present": False,
            "raw_secret_values_recorded": False,
        }
    file_value = _string(os.getenv(ISAAC_WORKER_IMAGE_REF_FILE_ENV))
    image_ref_file = Path(file_value or DEFAULT_ISAAC_WORKER_IMAGE_REF_FILE).expanduser()
    if image_ref_file.is_file():
        image_ref = image_ref_file.read_text(encoding="utf-8").strip()
        if not image_ref:
            return {
                "image_ref": "",
                "source": ISAAC_WORKER_IMAGE_REF_FILE_ENV
                if file_value
                else "default_blueprint_secret_file_path",
                "configured": False,
                "image_ref_file": str(image_ref_file),
                "image_ref_file_present": True,
                "raw_secret_values_recorded": False,
            }
        return {
            "image_ref": image_ref,
            "source": ISAAC_WORKER_IMAGE_REF_FILE_ENV
            if file_value
            else "default_blueprint_secret_file_path",
            "configured": True,
            "image_ref_file": str(image_ref_file),
            "image_ref_file_present": True,
            "raw_secret_values_recorded": False,
        }
    generic = _string(os.getenv("BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF"))
    if generic:
        return {
            "image_ref": generic,
            "source": "BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF",
            "configured": True,
            "image_ref_file": str(image_ref_file),
            "image_ref_file_present": False,
            "raw_secret_values_recorded": False,
        }
    return {
        "image_ref": "",
        "source": None,
        "configured": False,
        "image_ref_file": str(image_ref_file),
        "image_ref_file_present": False,
        "raw_secret_values_recorded": False,
    }


def _isaac_worker_image_size_diagnostic(image_ref: str) -> dict[str, Any]:
    explicit = _string(os.getenv(ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV))
    selected = explicit or DEFAULT_ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC
    path = Path(selected).expanduser()
    base = {
        "env_var": ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV,
        "path": str(path),
        "path_source": "env" if explicit else "default_output_path",
        "path_present": path.is_file(),
        "raw_secret_values_recorded": False,
    }
    if not path.is_file():
        return {
            **base,
            "status": "missing",
            "metadata_available_for_selected_image": False,
        }
    try:
        payload = read_json_any(path)
    except Exception as exc:
        return {
            **base,
            "status": "unreadable",
            "metadata_available_for_selected_image": False,
            "error_type": type(exc).__name__,
        }
    manifest = dict(payload) if isinstance(payload, Mapping) else {}
    manifest_image_ref = _string(manifest.get("image_ref"))
    if manifest_image_ref and image_ref and manifest_image_ref != image_ref:
        return {
            **base,
            "status": "ignored_image_ref_mismatch",
            "metadata_available_for_selected_image": False,
            "manifest_image_ref": manifest_image_ref,
        }
    if manifest.get("status") != "completed":
        return {
            **base,
            "status": _string(manifest.get("status")) or "not_completed",
            "metadata_available_for_selected_image": False,
            "blockers": _string_list(manifest.get("blockers")),
        }
    diagnostic = {
        **base,
        "status": "completed",
        "metadata_available_for_selected_image": True,
        "source_artifact": str(path),
        "image_ref": image_ref or manifest_image_ref,
        "total_compressed_size_bytes": manifest.get("total_compressed_size_bytes"),
        "largest_compressed_layer_size_bytes": (
            manifest.get("largest_layer_size_bytes")
            or manifest.get("largest_compressed_layer_size_bytes")
        ),
        "large_image_pull_risk": bool(manifest.get("large_image_pull_risk")),
        "layer_count": manifest.get("layer_count"),
        "proof_boundary": (
            "Worker image manifest metadata only. This does not prove container "
            "startup, Isaac Sim execution, policy execution, safety, or robot readiness."
        ),
    }
    layers = manifest.get("layers")
    if isinstance(layers, list):
        diagnostic["layers"] = layers
    return diagnostic


def build_provider_plan(
    *,
    runtime: Mapping[str, Any],
    job_id: str,
    job_dir: Path,
    allow_cloud_gpu: bool,
    generated_at: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not allow_cloud_gpu:
        blockers.append("cloud_gpu_not_authorized_in_this_session")
    secret_files = [
        _secret_file_status("NGC_API_KEY_FILE", "~/.blueprint-secrets/ngc_api_key"),
        _secret_file_status("RUNPOD_API_KEY_FILE", "~/.blueprint-secrets/runpod_api_key"),
    ]
    image_config = _configured_isaac_worker_image_ref()
    direct_base_image_allowed = _env_truthy(ALLOW_DIRECT_ISAAC_BASE_IMAGE_RUNPOD_ENV)
    worker_image_ref = _string(image_config.get("image_ref")) or DEFAULT_ISAAC_RUNTIME_IMAGE_REF
    image_size_diagnostic = _isaac_worker_image_size_diagnostic(worker_image_ref)
    eval_manifest_uri = _string(
        os.getenv("BLUEPRINT_EVAL_MANIFEST_URI")
        or os.getenv("BLUEPRINT_ISAAC_PROVIDER_BUNDLE_URI")
    )
    artifact_output_uri = _string(os.getenv("BLUEPRINT_ARTIFACT_OUTPUT_URI"))
    signed_put_url_present = bool(
        _string(os.getenv("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"))
    )
    eval_manifest_uri_fetchable = _provider_uri_fetchable(eval_manifest_uri)
    artifact_output_uri_writable = _provider_uri_writable(artifact_output_uri)
    local_runtime_blockers = (
        [] if runtime.get("isaac_runtime_available") else ["local_isaac_runtime_unavailable"]
    )
    if allow_cloud_gpu and not all(item["present"] for item in secret_files):
        blockers.append("required_file_based_provider_secret_missing")
    if allow_cloud_gpu and not image_config.get("configured") and not direct_base_image_allowed:
        blockers.append("prebuilt_isaac_eval_worker_image_ref_missing")
    if allow_cloud_gpu and not eval_manifest_uri:
        blockers.append("provider_fetchable_bundle_uri_missing")
    elif allow_cloud_gpu and not eval_manifest_uri_fetchable:
        blockers.append("provider_fetchable_bundle_uri_unusable_by_builtin_fetcher")
    if allow_cloud_gpu and not artifact_output_uri and not signed_put_url_present:
        blockers.append("provider_writable_artifact_output_uri_missing")
    elif allow_cloud_gpu and not artifact_output_uri_writable:
        if artifact_output_uri:
            blockers.append("provider_writable_artifact_output_uri_unsupported_scheme")
    if not allow_cloud_gpu and local_runtime_blockers:
        blockers.extend(local_runtime_blockers)
    return {
        "schema_version": ISAAC_PROVIDER_PLAN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "ready_for_authorized_provider_execution" if not blockers else "blocked",
        "provider_priority": ["local_isaac", "runpod_owner_gpu"],
        "selected_provider": (
            "local_isaac"
            if runtime.get("isaac_runtime_available")
            else "runpod_owner_gpu"
            if allow_cloud_gpu
            else "none"
        ),
        "cloud_gpu_authorized": allow_cloud_gpu,
        "cloud_gpu_calls_performed": False,
        "provider_api_calls_performed": False,
        "job_dir": str(job_dir),
        "file_based_secret_inputs": secret_files,
        "local_runtime_blockers": local_runtime_blockers,
        "provider_runtime_inputs": {
            "worker_image_ref_present": bool(worker_image_ref),
            "worker_image_ref": worker_image_ref,
            "worker_image_ref_env": image_config.get("source")
            or "default_isaac_sim_runtime_image",
            "prebuilt_worker_image_ref_configured": bool(image_config.get("configured")),
            "worker_image_ref_file": image_config.get("image_ref_file"),
            "worker_image_ref_file_present": image_config.get("image_ref_file_present"),
            "worker_image_manifest_diagnostic": image_size_diagnostic,
            "direct_isaac_base_image_runpod_allowed": direct_base_image_allowed,
            "direct_isaac_base_image_override_env": ALLOW_DIRECT_ISAAC_BASE_IMAGE_RUNPOD_ENV,
            "eval_manifest_uri_present": bool(eval_manifest_uri),
            "eval_manifest_uri_fetchable_by_builtin_fetcher": eval_manifest_uri_fetchable,
            "artifact_output_uri_present": bool(artifact_output_uri),
            "artifact_output_uri_writable_by_declared_scheme": artifact_output_uri_writable,
            "runtime_manifest_signed_put_url_present": signed_put_url_present,
            "raw_secret_values_recorded": False,
        },
        "runtime_requirements": [
            "Isaac Sim with RTX-capable NVIDIA runtime",
            "official Unitree G1 USD asset resolvable inside the runtime",
            "controller/policy stack that drives joints continuously",
            "artifact output destination before paid GPU time",
        ],
        "blockers": blockers,
        "proof_boundary": (
            "This plan is a launch/readiness artifact only. It does not prove that a "
            "provider was allocated, a scene was loaded, or a policy was executed."
        ),
    }


def _phase_rows(
    *,
    runtime: Mapping[str, Any],
    generated_at: str,
    artifacts_exported: bool,
) -> list[dict[str, Any]]:
    runtime_ready = bool(runtime.get("isaac_runtime_available"))
    rows: list[dict[str, Any]] = []
    for phase in REQUIRED_PHASES:
        blockers: list[str] = []
        status = "blocked"
        proof_effect = "none"
        if phase == "runner_referencing_official_g1":
            status = "completed"
            proof_effect = "official_g1_asset_reference_declared"
        elif phase in {
            "runner_official_g1_resolved",
            "runner_official_g1_reference_added",
            "runner_robot_api_evidence_collected",
            "runner_scene_loaded",
            "runner_episode_execution_started",
            "runner_episode_execution_completed",
        }:
            if not runtime_ready:
                blockers = ["isaac_sim_runtime_unavailable"]
            else:
                blockers = ["isaac_runtime_execution_not_run_in_this_local_pass"]
        elif phase == "runner_artifacts_exported":
            status = "completed" if artifacts_exported else "blocked"
            blockers = [] if artifacts_exported else ["local_artifacts_not_exported"]
            proof_effect = "fail_closed_local_artifacts_exported" if artifacts_exported else "none"
        elif phase == "runner_gpu_teardown_completed":
            status = "completed"
            proof_effect = "no_cloud_gpu_started_by_this_run"
        rows.append(
            {
                "schema_version": ISAAC_PHASE_LOG_SCHEMA_VERSION,
                "generated_at": generated_at,
                "phase": phase,
                "status": status,
                "blockers": blockers,
                "proof_effect": proof_effect,
            }
        )
    return rows


def _cost_ledger(*, job_id: str, generated_at: str, allow_cloud_gpu: bool) -> dict[str, Any]:
    return {
        "schema_version": ISAAC_COST_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "no_spend",
        "cloud_gpu_authorized": allow_cloud_gpu,
        "provider_api_calls_performed": False,
        "cloud_gpu_resources_started_by_this_run": False,
        "estimated_billable_gpu_seconds": 0,
        "actual_billable_gpu_seconds": 0,
        "estimated_cost_usd": 0.0,
        "actual_cost_usd": 0.0,
        "cost_control_policy": {
            "paid_cloud_requires_explicit_session_authorization": True,
            "file_based_secrets_required": True,
            "teardown_manifest_required": True,
        },
    }


def _teardown_manifest(*, job_id: str, generated_at: str, allow_cloud_gpu: bool) -> dict[str, Any]:
    return {
        "schema_version": ISAAC_TEARDOWN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "not_required_no_cloud_gpu_started",
        "cloud_gpu_authorized": allow_cloud_gpu,
        "cloud_gpu_resources_started_by_this_run": False,
        "provider_api_calls_performed": False,
        "teardown_actions_performed": [],
        "runner_gpu_teardown_completed": True,
        "continuing_spend_from_this_run": False,
        "zero_continuing_spend_scope": "no paid provider resource was launched by this run",
    }


def _provider_uri_fetchable(uri: str) -> bool:
    if not uri:
        return False
    return urlparse(uri).scheme in {"http", "https"}


def _provider_uri_writable(uri: str) -> bool:
    if not uri:
        return False
    return (urlparse(uri).scheme or "file") in {"gs", "s3", "r2", "file", "local"}


def _copy_if_file(source: str | Path | None, destination: Path) -> dict[str, Any]:
    if not source:
        return {"source": None, "copied": False, "blocker": "source_not_provided"}
    resolved = Path(source).expanduser().resolve()
    if not resolved.is_file():
        return {"source": str(resolved), "copied": False, "blocker": "source_missing"}
    ensure_dir(destination.parent)
    shutil.copy2(resolved, destination)
    return {
        "source": str(resolved),
        "path": str(destination),
        "copied": True,
        "size_bytes": destination.stat().st_size,
    }


def _isaac_runtime_runner_source() -> str:
    return r'''#!/usr/bin/env python3
"""Provider-side Isaac Sim runner for Blueprint 3DGS/G1 realistic eval bundles."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


REQUIRED_PHASES = [
    "runner_referencing_official_g1",
    "runner_official_g1_resolved",
    "runner_official_g1_reference_added",
    "runner_robot_api_evidence_collected",
    "runner_scene_loaded",
    "runner_episode_execution_started",
    "runner_episode_execution_completed",
    "runner_artifacts_exported",
    "runner_gpu_teardown_completed",
]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_runtime_result(path: Path, payload: dict) -> None:
    try:
        _write_json(path, payload)
    except Exception as exc:
        fallback = path.with_name(path.stem + "_write_failed.json")
        fallback_payload = {
            "schema_version": "isaac_provider_runtime_result_write_failed.v1",
            "status": "blocked",
            "intended_result_path": str(path),
            "write_error": f"{type(exc).__name__}:{str(exc)[:300]}",
            "result_snapshot": payload,
        }
        _write_json(fallback, fallback_payload)


def _append_phase(path: Path, phase: str, status: str, blockers: list[str] | None = None, **extra: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": "isaac_provider_runtime_phase.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": phase,
        "status": status,
        "blockers": blockers or [],
        **extra,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _resolve_manifest_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "isaac_provider_eval_manifest.json"


def _bundle_relative_path_candidates(bundle_root: Path, value: object) -> list[Path]:
    text = str(value or "").strip()
    if not text:
        return []
    raw = Path(text)
    if raw.is_absolute():
        return [raw]
    candidates: list[Path] = []
    parts = raw.parts
    if parts and parts[0] == bundle_root.name:
        candidates.append(bundle_root.joinpath(*parts[1:]))
    candidates.append(bundle_root / raw)
    candidates.append(bundle_root.parent / raw)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _resolve_bundle_relative_path(bundle_root: Path, value: object) -> Path:
    candidates = _bundle_relative_path_candidates(bundle_root, value)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if candidates:
        return candidates[0]
    return bundle_root


def _open_usd_stage(context: object, Usd: object, scene_path: Path, simulation_app: object) -> tuple[object | None, bool, list[dict]]:
    diagnostics: list[dict] = []
    opened = False
    try:
        opened = bool(context.open_stage(str(scene_path)))
        diagnostics.append({"method": "omni.usd.context.open_stage", "status": "completed", "returned": opened})
    except Exception as exc:
        diagnostics.append(
            {
                "method": "omni.usd.context.open_stage",
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:300],
            }
        )
    for _ in range(10):
        try:
            simulation_app.update()
        except Exception as exc:
            diagnostics.append(
                {
                    "method": "simulation_app.update_after_open_stage",
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:300],
                }
            )
            break
    try:
        stage = context.get_stage()
        diagnostics.append(
            {
                "method": "omni.usd.context.get_stage",
                "status": "completed" if stage else "empty",
            }
        )
        if stage:
            return stage, opened, diagnostics
    except Exception as exc:
        diagnostics.append(
            {
                "method": "omni.usd.context.get_stage",
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:300],
            }
        )
    try:
        stage = Usd.Stage.Open(str(scene_path))
        diagnostics.append(
            {
                "method": "Usd.Stage.Open",
                "status": "completed" if stage else "empty",
            }
        )
        return stage, opened, diagnostics
    except Exception as exc:
        diagnostics.append(
            {
                "method": "Usd.Stage.Open",
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:300],
            }
        )
    return None, opened, diagnostics


def _candidate_g1_asset_paths(manifest: dict) -> list[str]:
    configured = os.getenv("BLUEPRINT_ISAAC_UNITREE_G1_USD") or ""
    candidates = []
    if configured:
        candidates.append(configured)
    candidates.extend(manifest.get("unitree_g1_asset_candidates") or [])
    candidates.extend(
        [
            "Unitree/G1/g1.usd",
            "Robots/Unitree/G1/g1.usd",
            "/isaac-sim/Isaac/Robots/Unitree/G1/g1.usd",
            "omniverse://localhost/NVIDIA/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd",
            "omniverse://localhost/NVIDIA/Assets/Isaac/5.0/Isaac/Robots/Unitree/G1/g1.usd",
        ]
    )
    seen = set()
    unique = []
    for item in candidates:
        text = str(item).strip()
        if text and text not in seen:
            unique.append(text)
            seen.add(text)
    return unique


def _storage_g1_asset_candidates() -> tuple[list[str], list[dict]]:
    candidates: list[str] = []
    diagnostics: list[dict] = []
    try:
        from isaacsim.storage.native import get_full_asset_path  # type: ignore
    except Exception as exc:
        return candidates, [
            {
                "source": "isaacsim.storage.native.get_full_asset_path",
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:300],
            }
        ]
    for relative_path in (
        "Unitree/G1/g1.usd",
        "Robots/Unitree/G1/g1.usd",
        "Isaac/Robots/Unitree/G1/g1.usd",
    ):
        try:
            resolved = get_full_asset_path(relative_path)
        except Exception as exc:
            diagnostics.append(
                {
                    "source": "isaacsim.storage.native.get_full_asset_path",
                    "relative_path": relative_path,
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:300],
                }
            )
            continue
        diagnostics.append(
            {
                "source": "isaacsim.storage.native.get_full_asset_path",
                "relative_path": relative_path,
                "status": "completed" if resolved else "not_found",
                "resolved_path": resolved or None,
            }
        )
        if resolved:
            candidates.append(str(resolved))
    return candidates, diagnostics


def _probe_g1_asset_candidate(candidate: str) -> dict:
    text = str(candidate or "").strip()
    if not text:
        return {"candidate": text, "usable": False, "reason": "empty_candidate"}
    if text.startswith("omniverse://"):
        allow_omniverse_lookup = os.getenv(
            "BLUEPRINT_ISAAC_ALLOW_OMNIVERSE_G1_LOOKUP", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not allow_omniverse_lookup:
            return {
                "candidate": text,
                "candidate_kind": "omniverse",
                "usable": False,
                "reason": "omniverse_candidate_lookup_disabled",
                "opt_in_env_var": "BLUEPRINT_ISAAC_ALLOW_OMNIVERSE_G1_LOOKUP",
            }
        try:
            import omni.client  # type: ignore
        except Exception as exc:
            return {
                "candidate": text,
                "candidate_kind": "omniverse",
                "usable": False,
                "reason": "omni_client_unavailable",
                "error_type": type(exc).__name__,
            }
        try:
            result, _entry = omni.client.stat(text)
        except Exception as exc:
            return {
                "candidate": text,
                "candidate_kind": "omniverse",
                "usable": False,
                "reason": "omni_client_stat_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:300],
            }
        usable = result == omni.client.Result.OK
        return {
            "candidate": text,
            "candidate_kind": "omniverse",
            "usable": usable,
            "reason": "omni_client_stat_ok" if usable else f"omni_client_stat_{result}",
        }
    path = Path(text).expanduser()
    try:
        exists = path.is_file()
    except OSError as exc:
        return {
            "candidate": text,
            "candidate_kind": "local_file",
            "usable": False,
            "reason": "local_candidate_stat_failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:300],
        }
    if not exists:
        return {
            "candidate": text,
            "candidate_kind": "local_file",
            "usable": False,
            "reason": "local_candidate_not_file",
        }
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        return {
            "candidate": text,
            "candidate_kind": "local_file",
            "usable": False,
            "reason": "local_candidate_not_readable",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:300],
        }
    return {
        "candidate": text,
        "candidate_kind": "local_file",
        "usable": True,
        "reason": "local_candidate_readable",
        "resolved_path": str(path),
    }


def _resolve_g1_asset(manifest: dict) -> tuple[str, list[str], list[dict]]:
    storage_candidates, storage_diagnostics = _storage_g1_asset_candidates()
    raw_candidates = storage_candidates + _candidate_g1_asset_paths(manifest)
    seen = set()
    candidates = []
    for item in raw_candidates:
        text = str(item or "").strip()
        if text and text not in seen:
            candidates.append(text)
            seen.add(text)
    diagnostics = list(storage_diagnostics)
    for candidate in candidates:
        probe = _probe_g1_asset_candidate(candidate)
        diagnostics.append(probe)
        if probe.get("usable"):
            return str(probe.get("resolved_path") or candidate), candidates, diagnostics
    return "", candidates, diagnostics


def _safe_camera_id(value: object) -> str:
    text = str(value or "camera").strip().lower()
    safe = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    return safe or "camera"


def _load_camera_ids(bundle_root: Path, manifest: dict) -> list[str]:
    rel = (manifest.get("relative_paths") or {}).get("camera_manifest") or "camera_manifest.json"
    path = _resolve_bundle_relative_path(bundle_root, rel)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ["head_pov", "torso", "wrist", "third_person", "overhead", "task_focus"]
    rows = payload.get("cameras") if isinstance(payload.get("cameras"), list) else []
    camera_ids = [
        str(row.get("camera_id") or "").strip()
        for row in rows
        if isinstance(row, dict) and str(row.get("camera_id") or "").strip()
    ]
    return camera_ids or list(payload.get("requested_camera_ids") or [])


def _camera_pose(camera_id: str, index: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    poses = {
        "head_pov": ((0.12, 0.0, 1.55), (72.0, 0.0, 90.0)),
        "torso": ((0.05, 0.0, 1.15), (76.0, 0.0, 90.0)),
        "wrist": ((0.35, -0.25, 0.95), (68.0, 0.0, 55.0)),
        "third_person": ((-2.6, -2.2, 1.8), (63.0, 0.0, -42.0)),
        "overhead": ((0.0, 0.0, 4.0), (0.0, 0.0, 0.0)),
        "task_focus": ((1.8, -1.8, 1.35), (68.0, 0.0, 42.0)),
    }
    return poses.get(camera_id, ((-1.5 + index * 0.3, -2.0, 1.4), (68.0, 0.0, -35.0)))


def _define_camera(stage: object, camera_id: str, index: int) -> str:
    from pxr import Gf, Sdf, UsdGeom  # type: ignore

    root_path = Sdf.Path("/BlueprintIsaacG1Site/SmokeCameras")
    UsdGeom.Xform.Define(stage, root_path)
    safe = _safe_camera_id(camera_id)
    camera_path = root_path.AppendChild(safe)
    camera = UsdGeom.Camera.Define(stage, camera_path)
    position, rotation = _camera_pose(camera_id, index)
    xform = UsdGeom.XformCommonAPI(camera.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*position))
    xform.SetRotate(Gf.Vec3f(*rotation), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    camera.CreateFocalLengthAttr(18.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    return str(camera_path)


def _collect_frame_paths(path: Path) -> list[Path]:
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _encode_mp4(frame_paths: list[Path], output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    if not frame_paths:
        return {
            "status": "blocked",
            "path": str(output_path),
            "encoder": None,
            "frame_count": 0,
            "blockers": ["camera_smoke_frames_missing"],
        }
    try:
        import cv2  # type: ignore

        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            raise RuntimeError("cv2_failed_to_read_first_frame")
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            2.0,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("cv2_videowriter_not_opened")
        count = 0
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            count += 1
        writer.release()
        if output_path.is_file() and output_path.stat().st_size > 0 and count > 0:
            return {
                "status": "completed",
                "path": str(output_path),
                "encoder": "cv2.VideoWriter(mp4v)",
                "frame_count": count,
                "size_bytes": output_path.stat().st_size,
                "blockers": [],
            }
        blockers.append("cv2_mp4_write_empty")
    except Exception as exc:
        blockers.append(f"cv2_encode_failed:{type(exc).__name__}")

    try:
        import imageio.v2 as imageio  # type: ignore

        frames = [imageio.imread(str(frame_path)) for frame_path in frame_paths]
        imageio.mimsave(str(output_path), frames, fps=2)
        if output_path.is_file() and output_path.stat().st_size > 0:
            return {
                "status": "completed",
                "path": str(output_path),
                "encoder": "imageio",
                "frame_count": len(frames),
                "size_bytes": output_path.stat().st_size,
                "blockers": [],
            }
        blockers.append("imageio_mp4_write_empty")
    except Exception as exc:
        blockers.append(f"imageio_encode_failed:{type(exc).__name__}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        pattern = str(frame_paths[0].parent / "*.png")
        completed = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                "2",
                "-pattern_type",
                "glob",
                "-i",
                pattern,
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0:
            return {
                "status": "completed",
                "path": str(output_path),
                "encoder": "ffmpeg",
                "frame_count": len(frame_paths),
                "size_bytes": output_path.stat().st_size,
                "blockers": [],
            }
        blockers.append(f"ffmpeg_encode_failed:{completed.returncode}")
    else:
        blockers.append("ffmpeg_missing")
    return {
        "status": "blocked",
        "path": str(output_path),
        "encoder": None,
        "frame_count": 0,
        "blockers": blockers,
    }


def _run_camera_video_smoke(
    *,
    simulation_app: object,
    stage: object,
    bundle_root: Path,
    manifest: dict,
    output_dir: Path,
) -> dict:
    camera_ids = _load_camera_ids(bundle_root, manifest)
    videos_dir = output_dir / "realistic_videos"
    frames_root = output_dir / "camera_smoke_frames"
    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    blockers: list[str] = []
    capture_backend = None
    requested_frame_count = max(1, int(os.getenv("BLUEPRINT_CAMERA_SMOKE_FRAME_COUNT", "3")))
    try:
        import omni.replicator.core as rep  # type: ignore
        capture_backend = "omni.replicator.core.BasicWriter"
    except Exception as exc:
        blockers.append(f"camera_video_smoke_replicator_unavailable:{type(exc).__name__}")
        rep = None  # type: ignore[assignment]

    if rep is not None:
        for index, camera_id in enumerate(camera_ids):
            row_blockers: list[str] = []
            diagnostics: list[dict] = []
            camera_path = None
            safe_camera_id = _safe_camera_id(camera_id)
            frame_dir = frames_root / safe_camera_id
            frame_dir.mkdir(parents=True, exist_ok=True)
            writer = None
            render_product = None
            try:
                camera_path = _define_camera(stage, camera_id, index)
                diagnostics.append({"step": "define_camera", "status": "completed", "camera_path": camera_path})
                render_product = rep.create.render_product(camera_path, (640, 360))
                diagnostics.append({"step": "create_render_product", "status": "completed"})
                writer = rep.WriterRegistry.get("BasicWriter")
                diagnostics.append({"step": "get_basic_writer", "status": "completed"})
                writer.initialize(output_dir=str(frame_dir), rgb=True)
                diagnostics.append({"step": "writer_initialize", "status": "completed", "frame_dir": str(frame_dir)})
                writer.attach([render_product])
                diagnostics.append({"step": "writer_attach", "status": "completed"})
                for frame_index in range(requested_frame_count):
                    rep.orchestrator.step()
                    simulation_app.update()
                    diagnostics.append({"step": "orchestrator_step", "status": "completed", "frame_index": frame_index})
                try:
                    rep.orchestrator.wait_until_complete()
                    diagnostics.append({"step": "orchestrator_wait_until_complete", "status": "completed"})
                except Exception as exc:
                    diagnostics.append(
                        {
                            "step": "orchestrator_wait_until_complete",
                            "status": "blocked",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc)[:300],
                        }
                    )
            except Exception as exc:
                blocker = f"camera_video_smoke_camera_failed:{safe_camera_id}:{type(exc).__name__}"
                row_blockers.append(blocker)
                diagnostics.append(
                    {
                        "step": "camera_capture",
                        "status": "blocked",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:300],
                    }
                )
            finally:
                if writer is not None:
                    try:
                        writer.detach()
                        diagnostics.append({"step": "writer_detach", "status": "completed"})
                    except Exception as exc:
                        detach_blocker = f"camera_video_smoke_writer_detach_failed:{safe_camera_id}:{type(exc).__name__}"
                        row_blockers.append(detach_blocker)
                        diagnostics.append(
                            {
                                "step": "writer_detach",
                                "status": "blocked",
                                "error_type": type(exc).__name__,
                                "error_message": str(exc)[:300],
                            }
                        )
            frame_paths = _collect_frame_paths(frame_dir)
            diagnostics.append(
                {
                    "step": "collect_frames",
                    "status": "completed" if frame_paths else "blocked",
                    "frame_count": len(frame_paths),
                    "frame_suffixes": sorted({path.suffix.lower() for path in frame_paths}),
                }
            )
            encoded = (
                _encode_mp4(
                    frame_paths,
                    videos_dir / f"episode_0000__{safe_camera_id}.mp4",
                )
                if frame_paths
                else {
                    "status": "blocked",
                    "path": str(videos_dir / f"episode_0000__{safe_camera_id}.mp4"),
                    "encoder": None,
                    "frame_count": 0,
                    "blockers": ["camera_smoke_frames_missing"],
                }
            )
            row_blockers.extend(encoded.get("blockers") or [])
            rows.append(
                {
                    "camera_id": camera_id,
                    "camera_path": camera_path,
                    "frame_count": len(frame_paths),
                    "frame_dir": str(frame_dir),
                    "video": encoded,
                    "status": encoded.get("status"),
                    "blockers": row_blockers,
                    "diagnostics": diagnostics,
                }
            )
    completed_count = sum(1 for row in rows if row.get("status") == "completed")
    expected_count = len(camera_ids)
    if completed_count != expected_count:
        blockers.append("camera_video_smoke_count_below_expected")
    payload = {
        "schema_version": "provider_camera_video_smoke.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "completed" if not blockers else "blocked",
        "capture_backend": capture_backend,
        "video_smoke_kind": "static_isaac_scene_camera_smoke_not_policy_rollout",
        "requested_frame_count_per_camera": requested_frame_count,
        "expected_video_count": expected_count,
        "video_count": completed_count,
        "camera_ids": camera_ids,
        "videos_dir": str(videos_dir),
        "rows": rows,
        "blockers": blockers,
        "proof_boundary": (
            "These videos prove only that the Isaac runtime could render/capture "
            "static six-camera smoke outputs and package MP4s. They do not prove "
            "controller-grade rollout, official policy execution, contact dynamics, "
            "generated-world rank fidelity, or WAM/VLA runtime."
        ),
        "raw_secret_values_recorded": False,
    }
    _write_json(output_dir / "camera_video_smoke_manifest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-manifest")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    manifest_path = _resolve_manifest_path(args.eval_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle_root = manifest_path.parent
    output_dir = Path(args.output_dir or os.getenv("BLUEPRINT_ISAAC_OUTPUT_DIR") or bundle_root / "runtime_output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_log = output_dir / "isaac_runtime_phase_log.jsonl"
    result_path = output_dir / "isaac_runtime_result.json"
    if phase_log.exists():
        phase_log.unlink()

    result = {
        "schema_version": "isaac_provider_runtime_result.v1",
        "status": "blocked",
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "isaac_runtime_executed": False,
        "isaac_usd_scene_loaded": False,
        "unitree_g1_loaded_in_isaac": False,
        "controller_grade_execution_proven": False,
        "official_policy_execution_proven": False,
        "locomotion_continuity_validated": False,
        "collision_dynamics_validated": False,
        "manipulation_contact_dynamics_validated": False,
        "realistic_splat_visual_rendered": False,
        "wam_vla_runtime_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "blockers": [],
        "warnings": [],
    }

    _append_phase(phase_log, "runner_referencing_official_g1", "running")
    try:
        try:
            from isaacsim import SimulationApp  # type: ignore
        except Exception:
            from omni.isaac.kit import SimulationApp  # type: ignore
        simulation_app = SimulationApp({"headless": True})
        result["isaac_runtime_executed"] = True
    except Exception as exc:
        blocker = f"simulation_app_start_failed:{type(exc).__name__}"
        result["blockers"].append(blocker)
        for phase in REQUIRED_PHASES[1:]:
            _append_phase(phase_log, phase, "blocked", [blocker])
        _write_json(result_path, result)
        return 2

    try:
        from pxr import Sdf, Usd  # type: ignore
        import omni.usd  # type: ignore

        resolved_g1, g1_candidates, g1_candidate_diagnostics = _resolve_g1_asset(manifest)
        result["unitree_g1_asset_candidates"] = g1_candidates
        result["unitree_g1_asset_resolution_diagnostics"] = g1_candidate_diagnostics
        if not resolved_g1:
            blocker = "official_unitree_g1_usd_not_resolved_in_runtime"
            result["blockers"].append(blocker)
            _append_phase(
                phase_log,
                "runner_official_g1_resolved",
                "blocked",
                [blocker],
                candidates=g1_candidates,
                candidate_diagnostics=g1_candidate_diagnostics,
            )
        else:
            result["unitree_g1_asset_resolved_path"] = resolved_g1
            _append_phase(
                phase_log,
                "runner_official_g1_resolved",
                "completed",
                [],
                resolved_g1=resolved_g1,
                candidate_diagnostics=g1_candidate_diagnostics,
            )

        scene_path = _resolve_bundle_relative_path(
            bundle_root,
            manifest["relative_paths"]["generated_site_scene_usda"],
        )
        result["resolved_scene_path"] = str(scene_path)
        context = omni.usd.get_context()
        stage, opened, scene_open_diagnostics = _open_usd_stage(
            context,
            Usd,
            scene_path,
            simulation_app,
        )
        result["scene_open_diagnostics"] = scene_open_diagnostics
        if not stage:
            blocker = "isaac_stage_open_failed"
            result["blockers"].append(blocker)
            _append_phase(
                phase_log,
                "runner_scene_loaded",
                "blocked",
                [blocker],
                scene_path=str(scene_path),
                open_stage_return=opened,
                scene_open_diagnostics=scene_open_diagnostics,
            )
        else:
            result["isaac_usd_scene_loaded"] = True
            _append_phase(
                phase_log,
                "runner_scene_loaded",
                "completed",
                [],
                scene_path=str(scene_path),
                open_stage_return=bool(opened),
                scene_open_diagnostics=scene_open_diagnostics,
            )
            if resolved_g1:
                try:
                    robot_prim = stage.DefinePrim("/BlueprintIsaacG1Site/UnitreeG1", "Xform")
                    robot_prim.GetReferences().AddReference(resolved_g1)
                    _append_phase(phase_log, "runner_official_g1_reference_added", "completed", [], prim_path="/BlueprintIsaacG1Site/UnitreeG1")
                    result["unitree_g1_loaded_in_isaac"] = bool(robot_prim.IsValid())
                except Exception as exc:
                    blocker = f"official_unitree_g1_reference_failed:{type(exc).__name__}"
                    result["blockers"].append(blocker)
                    result["unitree_g1_reference_error_type"] = type(exc).__name__
                    result["unitree_g1_reference_error_message"] = str(exc)[:300]
                    _append_phase(phase_log, "runner_official_g1_reference_added", "blocked", [blocker])
            else:
                _append_phase(phase_log, "runner_official_g1_reference_added", "blocked", ["official_unitree_g1_usd_not_resolved_in_runtime"])
            try:
                stage.GetRootLayer().Save()
            except Exception as exc:
                result["warnings"].append(f"stage_root_layer_save_skipped:{type(exc).__name__}")
                result["stage_root_layer_save_skipped"] = True
                result["stage_root_layer_save_error_type"] = type(exc).__name__

        robot_evidence = {
            "stage_loaded": result["isaac_usd_scene_loaded"],
            "unitree_g1_loaded_in_isaac": result["unitree_g1_loaded_in_isaac"],
            "controller_runtime_configured": False,
            "controller_runtime_blocker": "real_unitree_g1_controller_policy_stack_not_packaged",
        }
        _write_json(output_dir / "runner_robot_api_evidence.json", robot_evidence)
        _append_phase(phase_log, "runner_robot_api_evidence_collected", "completed", [], evidence_path="runner_robot_api_evidence.json")

        if result["isaac_usd_scene_loaded"]:
            video_smoke = _run_camera_video_smoke(
                simulation_app=simulation_app,
                stage=stage,
                bundle_root=bundle_root,
                manifest=manifest,
                output_dir=output_dir,
            )
            result["camera_video_smoke_attempted"] = True
            result["camera_video_smoke_manifest_path"] = str(output_dir / "camera_video_smoke_manifest.json")
            result["camera_video_smoke_status"] = video_smoke.get("status")
            result["camera_video_smoke_kind"] = video_smoke.get("video_smoke_kind")
            result["expected_video_count"] = video_smoke.get("expected_video_count")
            result["video_count"] = video_smoke.get("video_count")
            result["video_smoke_proven"] = video_smoke.get("status") == "completed"
            if video_smoke.get("status") != "completed":
                result["blockers"].append("camera_video_smoke_not_completed")

        blocker = "real_unitree_g1_controller_policy_stack_not_packaged"
        result["blockers"].append(blocker)
        _append_phase(phase_log, "runner_episode_execution_started", "blocked", [blocker])
        _append_phase(phase_log, "runner_episode_execution_completed", "blocked", [blocker])
        _append_phase(phase_log, "runner_artifacts_exported", "completed", [], result_path=str(result_path))
        result["status"] = "blocked_controller_runtime_unavailable" if result["isaac_usd_scene_loaded"] else "blocked_scene_load_failed"
        return_code = 2
    except Exception as exc:
        blocker = f"isaac_runtime_execution_failed:{type(exc).__name__}"
        result["blockers"].append(blocker)
        result["runtime_exception_type"] = type(exc).__name__
        result["runtime_exception_message"] = str(exc)[:500]
        for phase in REQUIRED_PHASES[2:]:
            _append_phase(phase_log, phase, "blocked", [blocker])
        return_code = 2
    finally:
        # Some Isaac/Kit shutdown paths can terminate the interpreter before
        # Python reaches the post-close lines below. Persist the runtime result
        # before closing SimulationApp so the provider output zip is never just
        # a phase log.
        result["runtime_result_written_before_simulation_app_close"] = True
        _write_runtime_result(result_path, result)
        try:
            simulation_app.close()
        except Exception as exc:
            result.setdefault("teardown_warnings", []).append(str(exc))
        result["simulation_app_close_returned_to_runner"] = True
        _append_phase(phase_log, "runner_gpu_teardown_completed", "completed", [], continuing_spend_control="runner_closed_simulation_app")
        _write_runtime_result(result_path, result)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _provider_entrypoint_source() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${BLUEPRINT_ISAAC_EVAL_MANIFEST:-$BUNDLE_DIR/provider_runtime/isaac_provider_eval_manifest.json}"
OUTPUT_DIR="${BLUEPRINT_ISAAC_OUTPUT_DIR:-$BUNDLE_DIR/runtime_output}"
RUNNER="$BUNDLE_DIR/provider_runtime/isaac_realistic_runtime_runner.py"
RESULT="$OUTPUT_DIR/isaac_runtime_result.json"
PHASE_LOG="$OUTPUT_DIR/isaac_runtime_phase_log.jsonl"

mkdir -p "$OUTPUT_DIR"

write_missing_result() {
  local runner_rc="$1"
  if [ -f "$RESULT" ]; then
    return 0
  fi
  local blocker="isaac_runner_process_exited_without_runtime_result:$runner_rc"
  printf '{"blockers":["%s"],"generated_at":"%s","phase":"runner_artifacts_exported","schema_version":"isaac_provider_runtime_phase.v1","status":"blocked"}\n' "$blocker" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$PHASE_LOG"
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$RESULT" "$RUNNER" "$OUTPUT_DIR" "$runner_rc" <<'PY' && return 0
import json
import sys
import time
from pathlib import Path

result_path = Path(sys.argv[1])
runner_path = sys.argv[2]
output_dir = sys.argv[3]
runner_rc = int(sys.argv[4])
payload = {
    "schema_version": "isaac_provider_runtime_result.v1",
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "status": "blocked_isaac_process_exited_without_result",
    "runner_path": runner_path,
    "output_dir": output_dir,
    "provider_entrypoint_observed_runner_exit_code": runner_rc,
    "isaac_runtime_executed": False,
    "isaac_usd_scene_loaded": False,
    "unitree_g1_loaded_in_isaac": False,
    "controller_grade_execution_proven": False,
    "official_policy_execution_proven": False,
    "locomotion_continuity_validated": False,
    "collision_dynamics_validated": False,
    "manipulation_contact_dynamics_validated": False,
    "realistic_splat_visual_rendered": False,
    "wam_vla_runtime_proven": False,
    "generated_world_rank_fidelity_result_proven": False,
    "generated_world_policy_evaluation_scope_proven": False,
    "blockers": [f"isaac_runner_process_exited_without_runtime_result:{runner_rc}"],
    "warnings": ["provider_entrypoint_shell_fallback_wrote_result_after_runner_exit"],
    "phase_log_may_contain_last_successful_phase": True,
    "raw_secret_values_recorded": False,
}
result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
  fi
  cat > "$RESULT" <<JSON
{
  "schema_version": "isaac_provider_runtime_result.v1",
  "status": "blocked_isaac_process_exited_without_result",
  "runner_path": "$RUNNER",
  "output_dir": "$OUTPUT_DIR",
  "provider_entrypoint_observed_runner_exit_code": $runner_rc,
  "isaac_runtime_executed": false,
  "isaac_usd_scene_loaded": false,
  "unitree_g1_loaded_in_isaac": false,
  "controller_grade_execution_proven": false,
  "official_policy_execution_proven": false,
  "locomotion_continuity_validated": false,
  "collision_dynamics_validated": false,
  "manipulation_contact_dynamics_validated": false,
  "realistic_splat_visual_rendered": false,
  "wam_vla_runtime_proven": false,
  "generated_world_rank_fidelity_result_proven": false,
  "generated_world_policy_evaluation_scope_proven": false,
  "blockers": ["$blocker"],
  "warnings": ["provider_entrypoint_shell_fallback_wrote_result_after_runner_exit"],
  "phase_log_may_contain_last_successful_phase": true,
  "raw_secret_values_recorded": false
}
JSON
}

run_runner() {
  local python_bin="$1"
  set +e
  "$python_bin" "$RUNNER" --eval-manifest "$MANIFEST" --output-dir "$OUTPUT_DIR"
  local runner_rc="$?"
  set -e
  write_missing_result "$runner_rc"
  exit "$runner_rc"
}

if [ -n "${BLUEPRINT_ISAAC_PYTHON:-}" ]; then
  run_runner "$BLUEPRINT_ISAAC_PYTHON"
fi

for candidate in /isaac-sim/python.sh /root/.local/share/ov/pkg/isaac-sim/python.sh ./python.sh python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 || [ -x "$candidate" ]; then
    run_runner "$candidate"
  fi
done

echo "No Isaac Python executable found" >&2
write_missing_result 127
exit 127
'''


def _provider_fetch_command() -> str:
    return r'''set -euo pipefail
WORKSPACE="${BLUEPRINT_PROVIDER_WORKSPACE:-/workspace}"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"
PYTHON_BIN="${BLUEPRINT_ISAAC_PROVIDER_PYTHON:-}"
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [ -z "$PYTHON_BIN" ] && [ -x /isaac-sim/python.sh ]; then
  PYTHON_BIN="/isaac-sim/python.sh"
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "No Python executable found for Isaac provider fetch wrapper" >&2
  exit 127
fi
"$PYTHON_BIN" - <<'PY'
import json, os, shutil, subprocess, sys, time, urllib.request, zipfile
from pathlib import Path

workspace = Path(os.environ.get("BLUEPRINT_PROVIDER_WORKSPACE", "/workspace"))
bundle_uri = os.environ["BLUEPRINT_EVAL_MANIFEST_URI"]
bundle_zip = workspace / "isaac_provider_runtime_bundle.zip"
bundle_dir = workspace / "isaac_provider_bundle"
output_dir = workspace / "isaac_provider_runtime_output"
output_zip = workspace / "isaac_provider_runtime_output.zip"
signed_put_url = os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL", "").strip()
started_at = time.time()

def env_int(name, default):
    try:
        value = int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default
    return value if value > 0 else default

fetch_timeout_seconds = env_int("BLUEPRINT_ISAAC_PROVIDER_FETCH_TIMEOUT_SECONDS", 180)

def write_json(path, payload):
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

def reset_output_dir():
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()

def phase(name, **extra):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "isaac_provider_outer_phase.v1",
        "phase": name,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "raw_secret_values_recorded": False,
        **extra,
    }
    with (output_dir / "isaac_provider_outer_phase_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    write_json(output_dir / "isaac_provider_outer_latest_phase.json", payload)

def write_runtime_result(status, blockers, **extra):
    payload = {"schema_version": "isaac_provider_runtime_result.v1", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": status, "provider_outer_runner_status": status, "provider_outer_runner_elapsed_seconds": round(time.time() - started_at, 3), "isaac_runtime_executed": False, "isaac_usd_scene_loaded": False, "unitree_g1_loaded_in_isaac": False, "controller_grade_execution_proven": False, "official_policy_execution_proven": False, "locomotion_continuity_validated": False, "collision_dynamics_validated": False, "manipulation_contact_dynamics_validated": False, "realistic_splat_visual_rendered": False, "wam_vla_runtime_proven": False, "generated_world_rank_fidelity_result_proven": False, "generated_world_policy_evaluation_scope_proven": False, "blockers": blockers, "raw_secret_values_recorded": False, **extra}
    write_json(output_dir / "isaac_runtime_result.json", payload)

def package_output(reason):
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = output_zip.with_suffix(".zip.tmp")
    if tmp_zip.exists():
        tmp_zip.unlink()
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        file_count = 0
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir).as_posix())
                file_count += 1
        if file_count == 0:
            archive.writestr("runtime_output_missing.json", '{"status":"blocked","blockers":["runtime_output_directory_empty"]}\n')
    tmp_zip.replace(output_zip)
    return output_zip.stat().st_size

def upload_output(reason):
    size = package_output(reason)
    status = {"schema_version": "isaac_provider_runtime_output_upload_status.v1", "reason": reason, "runtime_output_zip_size_bytes": size, "signed_put_url_value_stored": False, "raw_secret_values_recorded": False}
    if not signed_put_url:
        status.update({"status": "skipped", "blockers": ["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL_missing"]})
        write_json(output_dir / "provider_output_upload_status.json", status)
        return status
    try:
        data = output_zip.read_bytes()
        request = urllib.request.Request(signed_put_url, data=data, method="PUT", headers={"Content-Type": "application/zip"})
        with urllib.request.urlopen(request, timeout=120) as response:
            status.update({"status": "completed", "upload_status": int(getattr(response, "status", 200)), "uploaded_bytes": len(data)})
    except Exception as exc:
        status.update({"status": "failed", "blockers": ["provider_runtime_output_upload_failed"], "error_type": type(exc).__name__, "error": str(exc)[:500]})
    write_json(output_dir / "provider_output_upload_status.json", status)
    package_output(f"{reason}:status_recorded")
    if status.get("status") == "completed":
        try:
            data = output_zip.read_bytes()
            request = urllib.request.Request(signed_put_url, data=data, method="PUT", headers={"Content-Type": "application/zip"})
            with urllib.request.urlopen(request, timeout=120) as response:
                status["final_manifest_upload_status"] = int(getattr(response, "status", 200))
                status["final_manifest_uploaded_bytes"] = len(data)
        except Exception as exc:
            status["final_manifest_upload_status"] = "failed"
            status["final_manifest_upload_error_type"] = type(exc).__name__
            status["final_manifest_upload_error"] = str(exc)[:500]
        write_json(output_dir / "provider_output_upload_status.json", status)
    return status

def fetch_bundle():
    phase("provider_bundle_fetch_started", fetch_timeout_seconds=fetch_timeout_seconds)
    with urllib.request.urlopen(bundle_uri, timeout=fetch_timeout_seconds) as response:
        with bundle_zip.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    phase("provider_bundle_fetch_completed", bundle_size_bytes=bundle_zip.stat().st_size)
    upload_output("provider_bundle_fetch_completed")

def unzip_bundle():
    phase("provider_bundle_unzip_started")
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_zip) as archive:
        archive.extractall(bundle_dir)
    phase("provider_bundle_unzip_completed")
    upload_output("provider_bundle_unzip_completed")

try:
    reset_output_dir()
    os.environ["BLUEPRINT_ISAAC_PROVIDER_OUTPUT_RESET_DONE"] = "1"
    phase("provider_outer_runner_started", python_executable=sys.executable)
    upload_output("provider_outer_runner_started")
    fetch_bundle()
    unzip_bundle()
    runner = bundle_dir / "provider_runtime" / "isaac_provider_outer_runner.py"
    if not runner.is_file():
        phase("provider_outer_runner_missing", runner=str(runner))
        write_runtime_result("blocked_provider_outer_runner_missing", ["provider_outer_runner_missing"])
        upload_output("provider_outer_runner_missing")
        sys.exit(127)
    env = {**os.environ, "BLUEPRINT_ISAAC_PROVIDER_STARTED_AT": str(started_at), "BLUEPRINT_ISAAC_PROVIDER_OUTPUT_RESET_DONE": "1"}
    rc = subprocess.call([sys.executable, str(runner)], cwd=str(bundle_dir / "provider_runtime"), env=env)
except Exception as exc:
    phase("provider_outer_runner_exception", error_type=type(exc).__name__, error=str(exc)[:500])
    write_runtime_result("blocked_provider_outer_runner_exception", ["provider_outer_runner_exception"], error_type=type(exc).__name__, error=str(exc)[:500])
    upload_output("provider_outer_runner_exception")
    raise
finally:
    phase("provider_outer_runner_final_upload")
    upload_output("provider_outer_runner_final_upload")
sys.exit(rc)
PY'''


def _provider_outer_runner_source() -> str:
    return r'''#!/usr/bin/env python3
import json, os, subprocess, sys, time, urllib.request, zipfile
from pathlib import Path

workspace = Path(os.environ.get("BLUEPRINT_PROVIDER_WORKSPACE", "/workspace"))
bundle_dir = workspace / "isaac_provider_bundle"
output_dir = workspace / "isaac_provider_runtime_output"
output_zip = workspace / "isaac_provider_runtime_output.zip"
signed_put_url = os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL", "").strip()
started_at = float(os.environ.get("BLUEPRINT_ISAAC_PROVIDER_STARTED_AT") or time.time())

def env_int(name, default):
    try:
        value = int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default
    return value if value > 0 else default

upload_interval_seconds = env_int("BLUEPRINT_ISAAC_PROVIDER_UPLOAD_INTERVAL_SECONDS", 20)
hard_timeout_seconds = env_int("BLUEPRINT_GPU_PROVIDER_HARD_TIMEOUT_SECONDS", 900)
runner_timeout_seconds = env_int("BLUEPRINT_ISAAC_PROVIDER_RUNNER_TIMEOUT_SECONDS", max(30, hard_timeout_seconds - 120))

def write_json(path, payload):
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

def reset_output_dir():
    if output_dir.exists():
        for path in sorted(output_dir.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()

def phase(name, **extra):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "isaac_provider_outer_phase.v1", "phase": name, "elapsed_seconds": round(time.time() - started_at, 3), "raw_secret_values_recorded": False, **extra}
    with (output_dir / "isaac_provider_outer_phase_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    write_json(output_dir / "isaac_provider_outer_latest_phase.json", payload)

def write_runtime_result(status, blockers, **extra):
    payload = {"schema_version": "isaac_provider_runtime_result.v1", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": status, "provider_outer_runner_status": status, "provider_outer_runner_elapsed_seconds": round(time.time() - started_at, 3), "isaac_runtime_executed": False, "isaac_usd_scene_loaded": False, "unitree_g1_loaded_in_isaac": False, "controller_grade_execution_proven": False, "official_policy_execution_proven": False, "locomotion_continuity_validated": False, "collision_dynamics_validated": False, "manipulation_contact_dynamics_validated": False, "realistic_splat_visual_rendered": False, "wam_vla_runtime_proven": False, "generated_world_rank_fidelity_result_proven": False, "generated_world_policy_evaluation_scope_proven": False, "blockers": blockers, "warnings": ["provider_outer_runner_wrote_fail_closed_runtime_result"], "raw_secret_values_recorded": False, **extra}
    write_json(output_dir / "isaac_runtime_result.json", payload)

def package_output(reason):
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = output_zip.with_suffix(".zip.tmp")
    if tmp_zip.exists():
        tmp_zip.unlink()
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        file_count = 0
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir).as_posix())
                file_count += 1
        if file_count == 0:
            archive.writestr("runtime_output_missing.json", '{"status":"blocked","blockers":["runtime_output_directory_empty"]}\n')
    tmp_zip.replace(output_zip)
    return output_zip.stat().st_size

def upload_output(reason):
    size = package_output(reason)
    status = {"schema_version": "isaac_provider_runtime_output_upload_status.v1", "reason": reason, "runtime_output_zip_size_bytes": size, "signed_put_url_value_stored": False, "raw_secret_values_recorded": False}
    if not signed_put_url:
        status.update({"status": "skipped", "blockers": ["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL_missing"]})
        write_json(output_dir / "provider_output_upload_status.json", status)
        return status
    try:
        data = output_zip.read_bytes()
        request = urllib.request.Request(signed_put_url, data=data, method="PUT", headers={"Content-Type": "application/zip"})
        with urllib.request.urlopen(request, timeout=120) as response:
            status.update({"status": "completed", "upload_status": int(getattr(response, "status", 200)), "uploaded_bytes": len(data)})
    except Exception as exc:
        status.update({"status": "failed", "blockers": ["provider_runtime_output_upload_failed"], "error_type": type(exc).__name__, "error": str(exc)[:500]})
    write_json(output_dir / "provider_output_upload_status.json", status)
    return status

def run_entrypoint():
    entrypoint = bundle_dir / "provider_runtime" / "run_isaac_realistic_runtime.sh"
    if not entrypoint.is_file():
        phase("provider_entrypoint_missing", entrypoint=str(entrypoint))
        write_runtime_result("blocked_provider_entrypoint_missing", ["provider_entrypoint_missing"])
        upload_output("provider_entrypoint_missing")
        return 127
    entrypoint.chmod(0o755)
    env = {**os.environ, "BLUEPRINT_ISAAC_OUTPUT_DIR": str(output_dir), "BLUEPRINT_ISAAC_EVAL_MANIFEST": str(bundle_dir / "provider_runtime" / "isaac_provider_eval_manifest.json")}
    phase("provider_entrypoint_subprocess_starting", runner_timeout_seconds=runner_timeout_seconds, upload_interval_seconds=upload_interval_seconds)
    upload_output("provider_entrypoint_subprocess_starting")
    with (output_dir / "provider_entrypoint_stdout.log").open("ab") as stdout, (output_dir / "provider_entrypoint_stderr.log").open("ab") as stderr:
        process = subprocess.Popen(["bash", str(entrypoint)], cwd=str(bundle_dir / "provider_runtime"), env=env, stdout=stdout, stderr=stderr)
        phase("provider_entrypoint_subprocess_running", pid=process.pid)
        last_upload = 0.0
        while True:
            rc = process.poll()
            elapsed = time.time() - started_at
            if rc is not None:
                phase("provider_entrypoint_subprocess_exited", returncode=rc)
                if not (output_dir / "isaac_runtime_result.json").is_file():
                    write_runtime_result("blocked_provider_entrypoint_exited_without_runtime_result", [f"provider_entrypoint_exited_without_runtime_result:{rc}"], provider_entrypoint_observed_exit_code=rc)
                upload_output("provider_entrypoint_subprocess_exited")
                return rc
            if elapsed > runner_timeout_seconds:
                phase("provider_entrypoint_timeout_started", pid=process.pid)
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=20)
                write_runtime_result("blocked_provider_entrypoint_timeout", ["provider_entrypoint_timeout_before_runtime_result_upload"], provider_entrypoint_timeout_seconds=runner_timeout_seconds)
                phase("provider_entrypoint_timeout_completed", returncode=process.returncode)
                upload_output("provider_entrypoint_timeout")
                return 124
            if elapsed - last_upload >= upload_interval_seconds:
                phase("provider_entrypoint_subprocess_heartbeat", pid=process.pid)
                upload_output("provider_entrypoint_subprocess_heartbeat")
                last_upload = elapsed
            time.sleep(5)

try:
    if os.environ.get("BLUEPRINT_ISAAC_PROVIDER_OUTPUT_RESET_DONE") != "1":
        reset_output_dir()
    rc = run_entrypoint()
except Exception as exc:
    phase("provider_outer_runner_exception", error_type=type(exc).__name__, error=str(exc)[:500])
    write_runtime_result("blocked_provider_outer_runner_exception", ["provider_outer_runner_exception"], error_type=type(exc).__name__, error=str(exc)[:500])
    upload_output("provider_outer_runner_exception")
    raise
finally:
    phase("provider_outer_runner_final_upload")
    upload_output("provider_outer_runner_final_upload")
sys.exit(rc)
'''


def _write_provider_runtime_bundle(
    *,
    job_dir: Path,
    job_id: str,
    generated_at: str,
    ply_asset: str | Path,
    spz_asset: str | Path,
    labels_json: str | Path | None,
    structure_json: str | Path | None,
    occupancy_json: str | Path | None,
    occupancy_png: str | Path | None,
    allow_cloud_gpu: bool,
) -> dict[str, Any]:
    provider_dir = job_dir / "provider_runtime"
    input_dir = provider_dir / "input_assets"
    ensure_dir(provider_dir)
    ensure_dir(input_dir)
    copied_assets = {
        "ply_asset": _copy_if_file(ply_asset, input_dir / Path(ply_asset).name),
        "spz_asset": _copy_if_file(spz_asset, input_dir / Path(spz_asset).name),
        "labels_json": _copy_if_file(labels_json, input_dir / "labels.json"),
        "structure_json": _copy_if_file(structure_json, input_dir / "structure.json"),
        "occupancy_json": _copy_if_file(occupancy_json, input_dir / "occupancy.json"),
        "occupancy_png": _copy_if_file(occupancy_png, input_dir / "occupancy.png"),
        "generated_site_scene_usda": _copy_if_file(
            job_dir / "generated_site_scene.usda",
            provider_dir / "generated_site_scene.usda",
        ),
        "generated_site_scene_usd": _copy_if_file(
            job_dir / "generated_site_scene.usd",
            provider_dir / "generated_site_scene.usd",
        ),
        "scenario_eval_matrix": _copy_if_file(
            job_dir / "scenario_eval_matrix.json",
            provider_dir / "scenario_eval_matrix.json",
        ),
        "camera_manifest": _copy_if_file(
            job_dir / "camera_manifest.json",
            provider_dir / "camera_manifest.json",
        ),
        "episode_spec_manifest": _copy_if_file(
            job_dir / "episode_spec_manifest.json",
            provider_dir / "episode_spec_manifest.json",
        ),
    }
    runner_path = provider_dir / "isaac_realistic_runtime_runner.py"
    outer_runner_path = provider_dir / "isaac_provider_outer_runner.py"
    entrypoint_path = provider_dir / "run_isaac_realistic_runtime.sh"
    runner_path.write_text(_isaac_runtime_runner_source(), encoding="utf-8")
    outer_runner_path.write_text(_provider_outer_runner_source(), encoding="utf-8")
    entrypoint_path.write_text(_provider_entrypoint_source(), encoding="utf-8")
    runner_path.chmod(0o755)
    outer_runner_path.chmod(0o755)
    entrypoint_path.chmod(0o755)
    eval_manifest = {
        "schema_version": ISAAC_PROVIDER_EVAL_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "bundle_contract": "self_contained_local_zip_requires_remote_staging_before_runpod",
        "relative_paths": {
            "generated_site_scene_usda": "generated_site_scene.usda",
            "generated_site_scene_usd": "generated_site_scene.usd",
            "scenario_eval_matrix": "scenario_eval_matrix.json",
            "camera_manifest": "camera_manifest.json",
            "episode_spec_manifest": "episode_spec_manifest.json",
            "input_assets_dir": "input_assets",
            "runtime_runner": "isaac_realistic_runtime_runner.py",
            "provider_outer_runner": "isaac_provider_outer_runner.py",
            "entrypoint": "run_isaac_realistic_runtime.sh",
        },
        "unitree_g1_asset_candidates": [
            OFFICIAL_ISAAC_G1_ASSET_PATH,
            "Unitree/G1/g1.usd",
            "Robots/Unitree/G1/g1.usd",
            "/isaac-sim/Isaac/Robots/Unitree/G1/g1.usd",
        ],
        "proof_boundaries": {
            "isaac_sim_required_for_scene_load": True,
            "official_g1_reference_required_for_robot_loaded": True,
            "real_controller_required_for_controller_grade_proof": True,
            "splat_rendering_requires_runtime_renderer_or_compositor": True,
            "metadata_colliders_are_not_direct_splat_collision": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
        },
        "copied_assets": copied_assets,
    }
    eval_manifest_path = job_dir / "isaac_provider_eval_manifest.json"
    write_json(eval_manifest_path, eval_manifest)
    copied_assets["isaac_provider_eval_manifest"] = _copy_if_file(
        eval_manifest_path,
        provider_dir / "isaac_provider_eval_manifest.json",
    )
    bundle_path = job_dir / "isaac_provider_runtime_bundle.zip"
    tmp_bundle_path = bundle_path.with_name(f".{bundle_path.name}.tmp")
    if tmp_bundle_path.exists():
        tmp_bundle_path.unlink()
    bundle_zip_entry_count = 0
    bundle_zip_testzip_result: str | None = None
    try:
        with zipfile.ZipFile(
            tmp_bundle_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for path in sorted(provider_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(job_dir).as_posix())
        with zipfile.ZipFile(tmp_bundle_path) as archive:
            bundle_zip_entry_count = len(archive.namelist())
            bundle_zip_testzip_result = archive.testzip()
        if bundle_zip_testzip_result is not None:
            raise RuntimeError(
                "provider_runtime_bundle_zip_integrity_failed:"
                f"{bundle_zip_testzip_result}"
            )
        tmp_bundle_path.replace(bundle_path)
    finally:
        if tmp_bundle_path.exists():
            tmp_bundle_path.unlink()
    bundle_manifest = {
        "schema_version": ISAAC_PROVIDER_RUNTIME_BUNDLE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "prepared_local_bundle_remote_staging_required",
        "bundle_path": str(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "bundle_write_atomic": True,
        "bundle_zip_entry_count": bundle_zip_entry_count,
        "bundle_zip_testzip_result": bundle_zip_testzip_result,
        "bundle_zip_integrity_passed": bundle_zip_testzip_result is None,
        "provider_runtime_dir": str(provider_dir),
        "eval_manifest_path": str(eval_manifest_path),
        "runner_path": str(runner_path),
        "entrypoint_path": str(entrypoint_path),
        "asset_copy_manifest": copied_assets,
        "cloud_gpu_authorized": allow_cloud_gpu,
        "remote_staging_required_before_runpod": True,
        "remote_staging_options": ["https_signed_get"],
        "unsupported_by_builtin_fetcher": ["gs", "s3", "r2"],
        "proof_boundary": (
            "The bundle contains executable Isaac runtime code and local inputs. It is not "
            "provider execution proof until staged to a provider-fetchable URI and run by "
            "an Isaac Sim container that writes runtime result artifacts."
        ),
    }
    write_json(job_dir / "isaac_provider_runtime_bundle_manifest.json", bundle_manifest)
    return bundle_manifest


def _run_local_provider_command_diagnostic(
    *,
    job_dir: Path,
    bundle_manifest: Mapping[str, Any],
    generated_at: str,
    timeout_seconds: int = 90,
) -> dict[str, Any]:
    bundle_path = Path(_string(bundle_manifest.get("bundle_path"))).expanduser()
    diagnostic_dir = job_dir / "local_provider_command_diagnostic"
    workspace_dir = diagnostic_dir / "workspace"
    output_dir = diagnostic_dir / "runtime_output"
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(workspace_dir)
    ensure_dir(output_dir)
    extracted_dir = workspace_dir / "isaac_provider_bundle"
    result: dict[str, Any] = {
        "schema_version": LOCAL_PROVIDER_COMMAND_DIAGNOSTIC_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked",
        "bundle_path": str(bundle_path),
        "diagnostic_dir": str(diagnostic_dir),
        "workspace_dir": str(workspace_dir),
        "runtime_output_dir": str(output_dir),
        "provider_bundle_unzipped": False,
        "provider_entrypoint_found": False,
        "provider_entrypoint_executed": False,
        "provider_command_path_local_proven": False,
        "runtime_result_written": False,
        "isaac_runtime_available": False,
        "isaac_runtime_execution_proven": False,
        "simulator_execution_proven": False,
        "rank_fidelity_result_proven": False,
        "raw_secret_values_recorded": False,
        "blockers": [],
    }
    if not bundle_path.is_file():
        result["blockers"] = ["provider_runtime_bundle_missing"]
        write_json(job_dir / "local_provider_command_diagnostic.json", result)
        return result
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            archive.extractall(extracted_dir)
        result["provider_bundle_unzipped"] = True
    except Exception as exc:
        result["blockers"] = [f"provider_runtime_bundle_unzip_failed:{type(exc).__name__}"]
        write_json(job_dir / "local_provider_command_diagnostic.json", result)
        return result
    entrypoint = extracted_dir / "provider_runtime" / "run_isaac_realistic_runtime.sh"
    result["entrypoint_path"] = str(entrypoint)
    result["provider_entrypoint_found"] = entrypoint.is_file()
    if not entrypoint.is_file():
        result["blockers"] = ["provider_runtime_entrypoint_missing"]
        write_json(job_dir / "local_provider_command_diagnostic.json", result)
        return result
    env = {
        **os.environ,
        "BLUEPRINT_ISAAC_OUTPUT_DIR": str(output_dir),
        "BLUEPRINT_ISAAC_EVAL_MANIFEST": str(
            extracted_dir / "provider_runtime" / "isaac_provider_eval_manifest.json"
        ),
    }
    try:
        completed = subprocess.run(
            ["bash", str(entrypoint)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        result.update(
            {
                "provider_entrypoint_executed": True,
                "returncode": completed.returncode,
                "stdout_prefix": completed.stdout[-4000:],
                "stderr_prefix": completed.stderr[-4000:],
            }
        )
    except subprocess.TimeoutExpired as exc:
        result.update(
            {
                "provider_entrypoint_executed": True,
                "status": "timeout",
                "returncode": None,
                "stdout_prefix": (exc.stdout or "")[-4000:]
                if isinstance(exc.stdout, str)
                else "",
                "stderr_prefix": (exc.stderr or "")[-4000:]
                if isinstance(exc.stderr, str)
                else "",
                "blockers": ["provider_entrypoint_local_timeout"],
            }
        )
        write_json(job_dir / "local_provider_command_diagnostic.json", result)
        return result
    runtime_result_path = output_dir / "isaac_runtime_result.json"
    phase_log_path = output_dir / "isaac_runtime_phase_log.jsonl"
    runtime_result = {}
    if runtime_result_path.is_file():
        runtime_result = _mapping(json.loads(runtime_result_path.read_text(encoding="utf-8")))
    blockers = _string_list(runtime_result.get("blockers"))
    expected_missing_isaac = any(
        blocker.startswith("simulation_app_start_failed:") for blocker in blockers
    )
    result.update(
        {
            "runtime_result_path": str(runtime_result_path),
            "phase_log_path": str(phase_log_path),
            "runtime_result_written": runtime_result_path.is_file(),
            "phase_log_written": phase_log_path.is_file(),
            "runtime_result_status": runtime_result.get("status"),
            "runtime_result_blockers": blockers,
            "isaac_runtime_available": runtime_result.get("isaac_runtime_executed") is True,
            "isaac_runtime_execution_proven": runtime_result.get("isaac_runtime_executed")
            is True,
            "provider_command_path_local_proven": bool(
                result["provider_bundle_unzipped"]
                and result["provider_entrypoint_executed"]
                and runtime_result_path.is_file()
            ),
            "status": (
                "completed_expected_missing_isaac_runtime"
                if expected_missing_isaac
                else "completed_provider_entrypoint_returned"
                if runtime_result_path.is_file()
                else "blocked_provider_entrypoint_no_runtime_result"
            ),
            "blockers": blockers
            if blockers
            else ([] if runtime_result_path.is_file() else ["runtime_result_missing"]),
            "proof_boundary": (
                "This local diagnostic proves the provider bundle can be unzipped and "
                "the provider entrypoint can run far enough to write runtime output. It "
                "does not prove Isaac Sim scene loading, G1 controller execution, video "
                "rendering, contact dynamics, WAM/VLA runtime, generated-world rank fidelity, or "
                "generated-world rank fidelity."
            ),
        }
    )
    write_json(job_dir / "local_provider_command_diagnostic.json", result)
    return result


def _write_provider_bundle_readiness_manifest(
    *,
    job_dir: Path,
    generated_at: str,
    bundle_manifest: Mapping[str, Any],
    local_provider_command_diagnostic: Mapping[str, Any],
    matrix: Mapping[str, Any],
    camera_manifest: Mapping[str, Any],
    video_manifest: Mapping[str, Any],
    gpu_provider_launch_request: Mapping[str, Any],
) -> dict[str, Any]:
    bundle_path = Path(_string(bundle_manifest.get("bundle_path"))).expanduser()
    required_entries = {
        "provider_runtime/isaac_provider_outer_runner.py",
        "provider_runtime/isaac_realistic_runtime_runner.py",
        "provider_runtime/run_isaac_realistic_runtime.sh",
        "provider_runtime/isaac_provider_eval_manifest.json",
        "provider_runtime/generated_site_scene.usda",
        "provider_runtime/generated_site_scene.usd",
        "provider_runtime/scenario_eval_matrix.json",
        "provider_runtime/camera_manifest.json",
        "provider_runtime/episode_spec_manifest.json",
    }
    blockers: list[str] = []
    warnings: list[str] = []
    zip_entries: list[str] = []
    entrypoint_text = ""
    runner_text = ""
    outer_runner_text = ""
    eval_manifest: dict[str, Any] = {}
    zip_parse_error = None
    zip_testzip_result: str | None = None
    zip_required_entries_present = False
    if not bundle_path.is_file():
        blockers.append("provider_runtime_bundle_missing")
    else:
        try:
            with zipfile.ZipFile(bundle_path) as archive:
                zip_entries = sorted(archive.namelist())
                zip_testzip_result = archive.testzip()
                entrypoint_text = archive.read(
                    "provider_runtime/run_isaac_realistic_runtime.sh"
                ).decode("utf-8", errors="replace")
                runner_text = archive.read(
                    "provider_runtime/isaac_realistic_runtime_runner.py"
                ).decode("utf-8", errors="replace")
                outer_runner_text = archive.read(
                    "provider_runtime/isaac_provider_outer_runner.py"
                ).decode("utf-8", errors="replace")
                eval_manifest = _mapping(
                    json.loads(
                        archive.read(
                            "provider_runtime/isaac_provider_eval_manifest.json"
                        ).decode("utf-8", errors="replace")
                    )
                )
        except Exception as exc:
            zip_parse_error = f"{type(exc).__name__}:{str(exc)[:300]}"
            blockers.append(f"provider_runtime_bundle_zip_inspection_failed:{type(exc).__name__}")
    missing_entries = sorted(required_entries - set(zip_entries))
    zip_required_entries_present = bool(zip_entries) and not missing_entries
    zip_integrity_passed = bool(zip_entries) and zip_parse_error is None and zip_testzip_result is None
    if missing_entries:
        blockers.append("provider_runtime_bundle_required_entries_missing")
    if zip_testzip_result is not None:
        blockers.append("provider_runtime_bundle_zip_integrity_failed")

    runs = [row for row in matrix.get("runs") or [] if isinstance(row, Mapping)]
    first_run = runs[0] if runs else {}
    first_run_camera_ids = _string_list(first_run.get("camera_ids"))
    expected_video_count = int(video_manifest.get("expected_video_count") or 0)
    video_count = int(video_manifest.get("video_count") or 0)
    reduced_smoke_matrix = (
        int(matrix.get("scenario_eval_run_count") or 0) == 1
        and len(runs) == 1
        and all(camera_id in first_run_camera_ids for camera_id in DEFAULT_CAMERA_IDS)
    )
    all_required_cameras = (
        camera_manifest.get("all_required_camera_types_requested") is True
        and len(camera_manifest.get("cameras") or []) == len(DEFAULT_CAMERA_IDS)
    )
    matrix_camera_slot_count = sum(
        len(_string_list(row.get("camera_ids"))) or len(camera_manifest.get("cameras") or [])
        for row in runs
    )
    expected_matrix_video_slots = expected_video_count == matrix_camera_slot_count
    expected_video_slots = expected_video_count == len(DEFAULT_CAMERA_IDS)
    if not expected_matrix_video_slots:
        blockers.append("provider_bundle_video_manifest_expected_count_mismatch")
    if video_count:
        warnings.append("video_files_already_present_before_live_smoke_validation")

    entrypoint_has_crash_fallback = (
        "write_missing_result" in entrypoint_text
        and "isaac_runner_process_exited_without_runtime_result" in entrypoint_text
        and "blocked_isaac_process_exited_without_result" in entrypoint_text
    )
    runner_uses_simulation_app = "SimulationApp" in runner_text
    runner_has_camera_video_smoke = (
        "_run_camera_video_smoke" in runner_text
        and "provider_camera_video_smoke.v1" in runner_text
        and "static_isaac_scene_camera_smoke_not_policy_rollout" in runner_text
    )
    runner_has_per_camera_video_smoke_diagnostics = (
        "camera_video_smoke_camera_failed" in runner_text
        and "requested_frame_count_per_camera" in runner_text
        and '"diagnostics": diagnostics' in runner_text
    )
    runner_has_scene_open_diagnostics = (
        "_open_usd_stage" in runner_text and "scene_open_diagnostics" in runner_text
    )
    runner_has_g1_asset_resolution_diagnostics = (
        "_resolve_g1_asset" in runner_text
        and "unitree_g1_asset_resolution_diagnostics" in runner_text
        and "/root/.local/share/ov/pkg/isaac-sim/Isaac/Robots/Unitree/G1/g1.usd"
        not in runner_text
    )
    relative_paths = _mapping(eval_manifest.get("relative_paths"))
    manifest_paths_runner_relative = bool(relative_paths) and all(
        not _string(value).startswith("provider_runtime/")
        for value in relative_paths.values()
    )
    runner_preserves_truth_boundaries = all(
        phrase in runner_text
        for phrase in (
            "controller_grade_execution_proven",
            "official_policy_execution_proven",
            "generated_world_rank_fidelity_result_proven",
            "generated_world_policy_evaluation_scope_proven",
        )
    )
    provider_shape = _mapping(gpu_provider_launch_request.get("provider_request_shape"))
    provider_command_text = _string(provider_shape.get("command"))
    provider_outer_has_early_phase_upload = all(
        phrase in provider_command_text
        for phrase in (
            "provider_outer_runner_started",
            "upload_output(\"provider_outer_runner_started\")",
            "provider_bundle_fetch_completed",
            "provider_bundle_unzip_completed",
        )
    )
    provider_outer_has_periodic_heartbeat_upload = all(
        phrase in outer_runner_text
        for phrase in (
            "BLUEPRINT_ISAAC_PROVIDER_UPLOAD_INTERVAL_SECONDS",
            "provider_entrypoint_subprocess_heartbeat",
            "upload_output(\"provider_entrypoint_subprocess_heartbeat\")",
        )
    )
    provider_outer_has_timeout_finalizer = all(
        phrase in outer_runner_text
        for phrase in (
            "BLUEPRINT_ISAAC_PROVIDER_RUNNER_TIMEOUT_SECONDS",
            "blocked_provider_entrypoint_timeout",
            "provider_entrypoint_timeout_before_runtime_result_upload",
        )
    )
    provider_outer_uses_stable_output_dir = (
        '/ "isaac_provider_runtime_output"' in provider_command_text
        and '/ "runtime_output"' not in provider_command_text.split("output_dir =", 1)[1].split("\n", 1)[0]
        if "output_dir =" in provider_command_text
        else False
    )
    provider_outer_uses_python_selector = all(
        phrase in provider_command_text
        for phrase in (
            'PYTHON_BIN="${BLUEPRINT_ISAAC_PROVIDER_PYTHON:-}"',
            "command -v python3",
            'PYTHON_BIN="/isaac-sim/python.sh"',
            '"$PYTHON_BIN" - <<',
        )
    )
    provider_outer_runner_bundled = (
        "provider_runtime/isaac_provider_outer_runner.py" in zip_entries
        and "def run_entrypoint" in outer_runner_text
        and "provider_outer_runner_final_upload" in outer_runner_text
    )
    if not entrypoint_has_crash_fallback:
        blockers.append("provider_entrypoint_missing_runtime_result_crash_fallback")
    if not runner_uses_simulation_app:
        blockers.append("provider_runner_missing_isaac_simulation_app_smoke")
    if not runner_has_camera_video_smoke:
        blockers.append("provider_runner_missing_camera_video_smoke")
    if not runner_has_per_camera_video_smoke_diagnostics:
        blockers.append("provider_runner_missing_per_camera_video_smoke_diagnostics")
    if not runner_has_scene_open_diagnostics:
        blockers.append("provider_runner_missing_scene_open_diagnostics")
    if not runner_has_g1_asset_resolution_diagnostics:
        blockers.append("provider_runner_missing_g1_asset_resolution_diagnostics")
    if not manifest_paths_runner_relative:
        blockers.append("provider_eval_manifest_paths_not_runner_relative")
    if not runner_preserves_truth_boundaries:
        blockers.append("provider_runner_missing_fail_closed_truth_boundaries")
    if not provider_outer_has_early_phase_upload:
        blockers.append("provider_fetch_command_missing_early_phase_upload")
    if not provider_outer_has_periodic_heartbeat_upload:
        blockers.append("provider_fetch_command_missing_periodic_heartbeat_upload")
    if not provider_outer_has_timeout_finalizer:
        blockers.append("provider_fetch_command_missing_timeout_finalizer")
    if not provider_outer_uses_stable_output_dir:
        blockers.append("provider_fetch_command_missing_stable_output_dir")
    if not provider_outer_uses_python_selector:
        blockers.append("provider_fetch_command_missing_python_selector")
    if not provider_outer_runner_bundled:
        blockers.append("provider_outer_runner_not_bundled")

    local_command_path_proven = (
        local_provider_command_diagnostic.get("provider_command_path_local_proven")
        is True
    )
    local_runtime_result_written = (
        local_provider_command_diagnostic.get("runtime_result_written") is True
    )
    if not local_command_path_proven:
        blockers.append("local_provider_command_path_not_proven")
    if not local_runtime_result_written:
        blockers.append("local_provider_runtime_result_not_written")

    provider_launch_blockers = _string_list(gpu_provider_launch_request.get("blockers"))
    local_blockers = [
        blocker
        for blocker in blockers
        if blocker
        not in {
            "missing_provider_fetchable_bundle_uri",
            "missing_provider_artifact_output_uri",
            "cloud_gpu_not_authorized_in_this_session",
        }
    ]
    local_bundle_ready = not local_blockers
    ready_for_next_live_attempt = (
        local_bundle_ready
        and gpu_provider_launch_request.get("status") == "request_manifest_ready"
        and not provider_launch_blockers
    )
    if provider_launch_blockers:
        blockers.extend(
            f"provider_launch_request_blocked:{blocker}"
            for blocker in provider_launch_blockers
        )

    manifest = {
        "schema_version": ISAAC_PROVIDER_BUNDLE_READINESS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "ready" if ready_for_next_live_attempt else "blocked",
        "bundle_path": str(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size if bundle_path.is_file() else 0,
        "zip_entry_count": len(zip_entries),
        "zip_required_entries_present": zip_required_entries_present,
        "zip_integrity_test_passed": zip_integrity_passed,
        "zip_testzip_result": zip_testzip_result,
        "missing_zip_entries": missing_entries,
        "zip_parse_error": zip_parse_error,
        "local_bundle_ready_for_remote_staging": local_bundle_ready,
        "ready_for_next_authorized_runpod_bundle_attempt": ready_for_next_live_attempt,
        "ready_for_next_authorized_vast_bundle_attempt": ready_for_next_live_attempt,
        "reduced_smoke_shape": {
            "scenario_eval_run_count": matrix.get("scenario_eval_run_count"),
            "run_count": len(runs),
            "first_run_camera_ids": first_run_camera_ids,
            "required_camera_ids": list(DEFAULT_CAMERA_IDS),
            "all_required_cameras_requested": all_required_cameras,
            "expected_video_count": expected_video_count,
            "current_video_count": video_count,
            "shape_is_1_spawn_1_task_all_6_cameras": reduced_smoke_matrix
            and expected_video_slots,
        },
        "matrix_contract": {
            "scenario_eval_run_count": matrix.get("scenario_eval_run_count"),
            "run_count": len(runs),
            "matrix_camera_slot_count": matrix_camera_slot_count,
            "expected_video_count_matches_matrix_camera_slots": expected_matrix_video_slots,
            "multi_row_matrix_allowed_for_provider_execution": True,
            "all_default_camera_types_required_for_provider_execution": False,
        },
        "entrypoint_runtime_result_crash_fallback_present": entrypoint_has_crash_fallback,
        "runner_uses_headless_simulation_app": runner_uses_simulation_app,
        "runner_has_camera_video_smoke": runner_has_camera_video_smoke,
        "runner_has_per_camera_video_smoke_diagnostics": runner_has_per_camera_video_smoke_diagnostics,
        "runner_has_scene_open_diagnostics": runner_has_scene_open_diagnostics,
        "runner_has_g1_asset_resolution_diagnostics": runner_has_g1_asset_resolution_diagnostics,
        "provider_fetch_command_has_early_phase_upload": provider_outer_has_early_phase_upload,
        "provider_fetch_command_has_periodic_heartbeat_upload": provider_outer_has_periodic_heartbeat_upload,
        "provider_fetch_command_has_timeout_finalizer": provider_outer_has_timeout_finalizer,
        "provider_fetch_command_uses_stable_output_dir": provider_outer_uses_stable_output_dir,
        "provider_fetch_command_uses_python_selector": provider_outer_uses_python_selector,
        "provider_outer_runner_bundled": provider_outer_runner_bundled,
        "provider_eval_manifest_paths_runner_relative": manifest_paths_runner_relative,
        "provider_eval_manifest_relative_paths": relative_paths,
        "runner_truth_boundaries_fail_closed": runner_preserves_truth_boundaries,
        "local_provider_command_path_proven": local_command_path_proven,
        "local_provider_runtime_result_written": local_runtime_result_written,
        "provider_launch_request_status": gpu_provider_launch_request.get("status"),
        "provider_launch_request_blockers": provider_launch_blockers,
        "blockers": sorted(set(blockers)),
        "warnings": warnings,
        "truth_boundaries": {
            "local_bundle_readiness_is_not_live_isaac_execution": True,
            "realistic_video_smoke_proven": False,
            "official_policy_execution_proven": False,
            "controller_grade_execution_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
        },
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / "isaac_provider_bundle_readiness.json", manifest)
    return manifest


def _build_gpu_provider_launch_request(
    *,
    job_id: str,
    generated_at: str,
    bundle_manifest: Mapping[str, Any],
    allow_cloud_gpu: bool,
) -> dict[str, Any]:
    bundle_uri = _string(
        os.getenv("BLUEPRINT_EVAL_MANIFEST_URI")
        or os.getenv("BLUEPRINT_ISAAC_PROVIDER_BUNDLE_URI")
    )
    artifact_output_uri = _string(os.getenv("BLUEPRINT_ARTIFACT_OUTPUT_URI"))
    signed_put_url_present = bool(
        _string(os.getenv("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"))
    )
    image_config = _configured_isaac_worker_image_ref()
    direct_base_image_allowed = _env_truthy(ALLOW_DIRECT_ISAAC_BASE_IMAGE_RUNPOD_ENV)
    image_ref = _string(image_config.get("image_ref")) or DEFAULT_ISAAC_RUNTIME_IMAGE_REF
    image_size_diagnostic = _isaac_worker_image_size_diagnostic(image_ref)
    container_registry_auth_id_present = bool(
        _string(os.getenv(RUNPOD_CONTAINER_REGISTRY_AUTH_ID_ENV))
    )
    manifest_fetchable = _provider_uri_fetchable(bundle_uri)
    artifact_writable = _provider_uri_writable(artifact_output_uri)
    artifact_output_required = not signed_put_url_present
    artifact_output_scheme = urlparse(artifact_output_uri).scheme if artifact_output_uri else None
    artifact_output_write_auth_ready = bool(artifact_output_uri and artifact_writable)
    blockers = []
    if not allow_cloud_gpu:
        blockers.append("cloud_gpu_not_authorized_in_this_session")
    if allow_cloud_gpu and not image_config.get("configured") and not direct_base_image_allowed:
        blockers.append("prebuilt_isaac_eval_worker_image_ref_missing")
    if not bundle_uri:
        blockers.append("missing_provider_fetchable_bundle_uri")
    elif not manifest_fetchable:
        blockers.append("provider_bundle_uri_not_fetchable_by_provider")
    if not artifact_output_uri and not signed_put_url_present:
        blockers.append("missing_provider_artifact_output_uri")
    elif not artifact_writable:
        if artifact_output_uri:
            blockers.append("provider_artifact_output_uri_not_writable")
    request_status = "request_manifest_ready" if not blockers else "blocked_provider_inputs_missing"
    local_sim_only_prerequisite = {
        "schema_version": "robot_eval_provider_local_sim_only_prerequisite.v1",
        "required_before_provider_spend": True,
        "status": "passed" if request_status == "request_manifest_ready" else "blocked",
        "source_artifact": "isaac_provider_bundle_readiness.json",
        "local_sim_only_evidence_clean": request_status == "request_manifest_ready",
        "sim_only_beta_core_complete": False,
        "simulator_backend": "isaac_sim",
        "blockers": [] if request_status == "request_manifest_ready" else list(blockers),
        "claim_boundary": {
            "provider_spend_requires_local_bundle_and_output_contract_clean": True,
            "local_bundle_clean_does_not_prove_isaac_runtime_execution": True,
            "local_bundle_clean_does_not_prove_policy_execution": True,
            "local_bundle_clean_does_not_prove_launch_approval": True,
        },
    }
    return {
        "schema_version": "robot_eval_gpu_provider_launch_request.v1",
        "generated_at": generated_at,
        "job_id": job_id,
        "provider": "runpod",
        "status": request_status,
        "operation": "isaac_g1_site_3dgs_realistic_eval",
        "blockers": blockers,
        "provider_input_setup": {
            "bundle_manifest_path": "isaac_provider_runtime_bundle_manifest.json",
            "local_bundle_path": bundle_manifest.get("bundle_path"),
            "remote_staging_required_before_runpod": True,
            "blockers": blockers,
        },
        "provider_request_shape": {
            "api_payload_is_provider_adapter_template": True,
            "api_payload_values_are_redacted": False,
            "operation": "isaac_g1_site_3dgs_realistic_eval",
            "image": {
                "configured_image_ref": image_ref,
                "configured_image_ref_present": bool(image_ref),
                "configured_image_ref_is_versioned": ":" in image_ref,
                "configured_image_ref_fetchable_by_provider": bool(image_ref),
                "image_family": "isaac-eval-worker"
                if image_config.get("configured")
                else "isaac-sim-base-direct",
                "owner_published_image_ref_required": not direct_base_image_allowed,
                "image_ref_source": image_config.get("source")
                or "default_isaac_sim_runtime_image",
                "prebuilt_worker_image_ref_configured": bool(image_config.get("configured")),
                "worker_image_ref_file": image_config.get("image_ref_file"),
                "worker_image_ref_file_present": image_config.get("image_ref_file_present"),
                "image_size_diagnostic": image_size_diagnostic,
                "direct_isaac_base_image_runpod_allowed": direct_base_image_allowed,
                "direct_isaac_base_image_override_env": ALLOW_DIRECT_ISAAC_BASE_IMAGE_RUNPOD_ENV,
                "direct_isaac_base_image_blocked_by_default": (
                    not image_config.get("configured") and not direct_base_image_allowed
                ),
                "container_registry_auth_id_present": container_registry_auth_id_present,
                "container_registry_auth_id_source": (
                    RUNPOD_CONTAINER_REGISTRY_AUTH_ID_ENV
                    if container_registry_auth_id_present
                    else None
                ),
                "raw_registry_credentials_recorded": False,
            },
            "docker_entrypoint": ["bash"],
            "docker_start_cmd": ["-lc", _provider_fetch_command()],
            "command": _provider_fetch_command(),
            "environment": {
                "secret_env_var_names": ["RUNPOD_API_KEY"],
                "secret_values_in_artifact": False,
                "plaintext_env_var_names": [
                    "ACCEPT_EULA",
                    "PRIVACY_CONSENT",
                    "BLUEPRINT_EVAL_MANIFEST_URI",
                    "BLUEPRINT_ISAAC_PROVIDER_UPLOAD_INTERVAL_SECONDS",
                    "BLUEPRINT_ISAAC_PROVIDER_FETCH_TIMEOUT_SECONDS",
                    "BLUEPRINT_ISAAC_PROVIDER_RUNNER_TIMEOUT_SECONDS",
                    *([] if signed_put_url_present else ["BLUEPRINT_ARTIFACT_OUTPUT_URI"]),
                ],
                "plaintext_env_values": {
                    "ACCEPT_EULA": "Y",
                    "PRIVACY_CONSENT": "Y",
                    "BLUEPRINT_EVAL_MANIFEST_URI": bundle_uri or "<stage bundle zip to https/gs/s3/r2>",
                    "BLUEPRINT_ISAAC_PROVIDER_UPLOAD_INTERVAL_SECONDS": "15",
                    "BLUEPRINT_ISAAC_PROVIDER_FETCH_TIMEOUT_SECONDS": "180",
                    "BLUEPRINT_ISAAC_PROVIDER_RUNNER_TIMEOUT_SECONDS": "780",
                    **(
                        {"BLUEPRINT_ARTIFACT_OUTPUT_URI": artifact_output_uri}
                        if artifact_output_uri and not signed_put_url_present
                        else {}
                    ),
                },
            },
            "inputs": {
                "manifest_uri": bundle_uri or None,
                "manifest_uri_kind": "isaac_provider_runtime_bundle_zip",
                "manifest_uri_required_for_provider": True,
                "manifest_uri_fetchable_by_provider": manifest_fetchable,
                "capture_root_bundle_uri": bundle_uri or None,
                "capture_root_bundle_uri_kind": "isaac_provider_runtime_bundle_zip",
                "capture_root_bundle_uri_required_for_provider": True,
                "capture_root_bundle_uri_fetchable_by_provider": manifest_fetchable,
                "artifact_output_uri_required": artifact_output_required,
                "artifact_output_uri": artifact_output_uri or None,
                "artifact_output_uri_scheme": artifact_output_scheme,
                "artifact_output_uri_writable": artifact_writable,
                "artifact_output_uri_provider_writable": artifact_writable,
                "artifact_output_write_auth_contract_ready": (
                    True if not artifact_output_required else artifact_output_write_auth_ready
                ),
                "artifact_output_write_auth": {
                    "write_auth_contract_ready": (
                        True
                        if not artifact_output_required
                        else artifact_output_write_auth_ready
                    ),
                    "authorization_mode": (
                        "signed_put_url"
                        if not artifact_output_required
                        else "worker_storage_credentials"
                    ),
                    "secret_values_in_artifact": False,
                },
                "provider_writable_artifact_output_uri_schemes": ["gs", "r2", "s3"],
                "signed_put_output_env_var": "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL",
                "signed_put_output_env_present_at_request_build": signed_put_url_present,
                "simulator": "isaac_sim",
            },
            "runtime_preflight": {
                "simulator": "isaac_sim",
                "robot_profile_id": "unitree_g1",
                "preferred_robot_asset": "official_isaac_unitree_g1_usd",
                "required_checks": [
                    "nvidia_smi",
                    "isaac_sim_python_available",
                    "simulation_app_headless_start",
                    "usd_stage_open",
                    "official_unitree_g1_reference",
                    "rtx_renderer_available",
                ],
            },
            "gpu": {
                "gpu_count": 1,
                "provider_gpu_priority": [
                    "NVIDIA L40S",
                    "NVIDIA RTX 6000 Ada Generation",
                    "NVIDIA RTX A6000",
                    "NVIDIA GeForce RTX 4090",
                ],
                "preferred_gpu_type_id": "NVIDIA L40S",
                "disallowed_gpu_classes": ["A100", "H100"],
                "volume_in_gb": 80,
                "container_disk_in_gb": 140,
                "min_vcpu_count": 8,
                "min_memory_in_gb": 32,
            },
            "limits": {
                "max_active_workers": 1,
                "requested_budget_usd": 2.0,
                "hard_timeout_seconds": 900,
                "idle_timeout_seconds": 60,
                "startup_artifact_watchdog_required": True,
                "startup_artifact_timeout_seconds": 360,
                "startup_artifact_poll_interval_seconds": 15,
                "idle_shutdown_required": True,
                "external_watchdog_ttl_required": True,
                "external_watchdog_ttl_seconds": 1200,
                "scale_to_zero_default": True,
                "external_watchdog_owner": "codex_or_owner_control_plane",
            },
            "artifact_finalizer": {
                "upload_before_shutdown_required": True,
                "record_actual_gpu_time_required": True,
                "shutdown_after_artifacts_required": True,
            },
            "local_sim_only_prerequisite": local_sim_only_prerequisite,
        },
        "proof_boundary": {
            "provider_request_is_not_provider_execution": True,
            "provider_allocation_proven": False,
            "simulator_execution_proven": False,
            "rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
        },
    }


def _write_usd_scene(
    *,
    usda_path: Path,
    usd_path: Path,
    proxies: Sequence[Mapping[str, Any]],
    ply_asset: str | Path,
    spz_asset: str | Path,
    generated_at: str,
) -> dict[str, Any]:
    ensure_dir(usda_path.parent)
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "BlueprintIsaacG1Site"',
        ")",
        "",
        'def Xform "BlueprintIsaacG1Site" (',
        "    customData = {",
        f'        string generatedAt = "{_usd_string(generated_at)}"',
        '        string sceneVisualSource = "gaussian_ply_and_spz"',
        '        string collisionSceneSource = "metadata_derived_collider_proxy"',
        f'        string sourcePly = "{_usd_string(Path(ply_asset).expanduser().resolve())}"',
        f'        string sourceSpz = "{_usd_string(Path(spz_asset).expanduser().resolve())}"',
        "        bool directSplatCollisionClaimed = false",
        "        bool loadedInIsaacThisRun = false",
        "    }",
        ")",
        "{",
        '    def Xform "VisualSources"',
        "    {",
        '        def Xform "GaussianSplatVisualSource" (',
        "            customData = {",
        f'                string plyPath = "{_usd_string(Path(ply_asset).expanduser().resolve())}"',
        f'                string spzPath = "{_usd_string(Path(spz_asset).expanduser().resolve())}"',
        "                bool physicalCollider = false",
        "            }",
        "        )",
        "        {",
        "        }",
        "    }",
        '    def Xform "MetadataDerivedColliders"',
        "    {",
    ]
    collider_count = 0
    for proxy in proxies:
        if proxy.get("collision_enabled") is not True:
            continue
        pos = list(proxy.get("pos") or [0.0, 0.0, 0.0])
        size = list(proxy.get("size") or [0.1, 0.1, 0.1])
        if len(pos) < 3 or len(size) < 3:
            continue
        scale_z = max(0.01, float(size[2]) * 2.0)
        if _string(proxy.get("geom_type")) == "plane":
            scale_z = 0.02
        ident = _usd_identifier(proxy.get("proxy_id"))
        lines.extend(
            [
                f'        def Cube "{ident}" (',
                "            customData = {",
                f'                string source = "{_usd_string(proxy.get("source") or "generated_proxy")}"',
                f'                string label = "{_usd_string(proxy.get("label"))}"',
                "                bool metadataDerivedColliderProxy = true",
                "            }",
                "        )",
                "        {",
                "            double size = 1",
                f"            double3 xformOp:translate = ({float(pos[0]):.6f}, "
                f"{float(pos[1]):.6f}, {float(pos[2]):.6f})",
                f"            double3 xformOp:scale = ({float(size[0]) * 2.0:.6f}, "
                f"{float(size[1]) * 2.0:.6f}, {scale_z:.6f})",
                '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
                "        }",
            ]
        )
        collider_count += 1
    lines.extend(
        [
            "    }",
            '    def Xform "UnitreeG1ExpectedReference" (',
            "        customData = {",
            f'            string expectedOfficialAsset = "{OFFICIAL_ISAAC_G1_ASSET_PATH}"',
            "            bool loadedInIsaacThisRun = false",
            "            bool controllerGradeExecutionProven = false",
            "        }",
            "    )",
            "    {",
            "    }",
            "}",
            "",
        ]
    )
    content = "\n".join(lines)
    usda_path.write_text(content, encoding="utf-8")
    usd_path.write_text(content, encoding="utf-8")
    return {
        "status": "completed",
        "generated_site_scene_usda": str(usda_path),
        "generated_site_scene_usd": str(usd_path),
        "format": "ascii_usda",
        "collider_prim_count": collider_count,
        "direct_splat_collision_claimed": False,
    }


def _camera_manifest(
    *,
    camera_ids: Sequence[str],
    generated_at: str,
    camera_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    camera_specs = {
        "head_pov": {
            "label": "head POV",
            "mount": "Unitree G1 head/perception frame",
            "required": True,
        },
        "torso": {"label": "torso/chest", "mount": "G1 torso frame", "required": True},
        "wrist": {
            "label": "wrist/end-effector",
            "mount": "simplified wrist/contact actuator frame",
            "required": True,
        },
        "third_person": {"label": "third-person chase", "mount": "world chase", "required": True},
        "overhead": {"label": "overhead", "mount": "fixed world overhead", "required": True},
        "task_focus": {"label": "task-focus", "mount": "fixed task-zone camera", "required": True},
    }
    return {
        "schema_version": CAMERA_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed",
        "requested_camera_ids": list(camera_ids),
        "camera_id_aliases": dict(camera_aliases or {}),
        "required_camera_ids": list(DEFAULT_CAMERA_IDS),
        "all_required_camera_types_requested": all(camera in camera_ids for camera in DEFAULT_CAMERA_IDS),
        "cameras": [
            {"camera_id": camera_id, **camera_specs[camera_id]}
            for camera_id in camera_ids
            if camera_id in camera_specs
        ],
        "camera_evidence_status": "blocked_until_isaac_or_splat_renderer_runs",
    }


def _apply_required_task_families(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    family_by_task = {
        "navigate_chalkboard_inspect": "inspect",
        "inspect_shelf_desk_zone": "approach",
        "desk_object_contact_check": "contact_move_object",
        "carry_object_to_drop_zone": "push_relocate",
        "clear_floor_obstruction_route": "route_around_obstruction",
    }
    ordered_family = [
        "inspect",
        "approach",
        "contact_move_object",
        "push_relocate",
        "route_around_obstruction",
    ]
    by_family: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_id = _string(task.get("task_id"))
        family = family_by_task.get(task_id, task_id)
        row = dict(task)
        row["requested_task_family"] = family
        row["isaac_action_requirement"] = {
            "requires_scene_grounding": True,
            "requires_physics_contact": family in {"contact_move_object", "push_relocate"},
            "requires_continuous_controller": True,
        }
        by_family.setdefault(family, row)
    return [by_family[family] for family in ordered_family if family in by_family]


def _attempts_for_matrix(
    *,
    matrix: Mapping[str, Any],
    runtime: Mapping[str, Any],
    runtime_result: Mapping[str, Any] | None = None,
    generated_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime_result_payload = _mapping(runtime_result)
    observed_attempts = _runtime_result_attempts(runtime_result_payload)
    observed_by_run_id = {
        _string(row.get("scenario_eval_run_id") or row.get("scenarioEvalRunId")): row
        for row in observed_attempts
        if _string(row.get("scenario_eval_run_id") or row.get("scenarioEvalRunId"))
    }
    runtime_available = bool(runtime.get("isaac_runtime_available")) or bool(
        runtime_result_payload.get("isaac_runtime_executed")
        or runtime_result_payload.get("isaac_sim_execution_proven")
    )
    blocker = (
        "blocked_controller_runtime_unavailable"
        if runtime_available
        else "blocked_isaac_runtime_unavailable"
    )
    attempts: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for index, run in enumerate(matrix.get("runs") or [], start=1):
        if not isinstance(run, Mapping):
            continue
        attempt_id = f"isaac_attempt_{index:04d}_{_safe_id(run.get('episode_id'))}"
        camera_ids = list(run.get("camera_ids") or [])
        observed = _mapping(observed_by_run_id.get(_string(run.get("scenario_eval_run_id"))))
        if observed:
            status = _string(observed.get("status") or observed.get("result") or "completed").lower()
            success_raw = observed.get("success")
            success = (
                bool(success_raw)
                if isinstance(success_raw, bool)
                else str(success_raw).strip().lower() in {"1", "true", "yes", "passed", "success"}
                if success_raw is not None
                else status in {"completed", "success", "succeeded", "passed"}
            )
            failure_ids = _string_list(
                observed.get("failure_mode_ids")
                or observed.get("failure_label_ids")
                or observed.get("blockers")
            )
            if not success and not failure_ids:
                failure_ids = ["isaac_runtime_attempt_failed"]
            attempt = {
                "attempt_id": _string(observed.get("attempt_id")) or attempt_id,
                "scenario_eval_run_id": run.get("scenario_eval_run_id"),
                "scenario_id": observed.get("scenario_id") or run.get("scenario_id"),
                "scenario_variation_instance_id": observed.get(
                    "scenario_variation_instance_id"
                )
                or run.get("scenario_variation_instance_id"),
                "task_id": observed.get("task_id") or run.get("task_id"),
                "spawn_id": observed.get("spawn_id") or run.get("spawn_id"),
                "episode_id": observed.get("episode_id") or run.get("episode_id"),
                "status": status,
                "success": success,
                "task_success": success,
                "task_status": "passed" if success else "failed_task_criteria",
                "failure_label_ids": sorted(set(failure_ids)),
                "failure_mode_ids": sorted(set(failure_ids)),
                "generated_at": generated_at,
                "simulator": "isaac_sim",
                "simulator_backend": "isaac_sim",
                "robot_profile": "unitree_g1_official_isaac_asset_expected",
                "policy_id": _string(observed.get("policy_id")) or "isaac_g1_runtime_policy",
                "camera_ids": camera_ids,
                "camera_evidence": observed.get("camera_evidence") or {},
                "metrics": {
                    **_mapping(observed.get("metrics")),
                    "isaac_runtime_attempt_trace_present": True,
                    "isaac_runtime_executed": True,
                },
                "task_outcome": _mapping(observed.get("task_outcome")),
                "actions": observed.get("actions") if isinstance(observed.get("actions"), list) else [],
                "action_trace": observed.get("actions") if isinstance(observed.get("actions"), list) else [],
                "contact_trace": observed.get("contact_trace")
                if isinstance(observed.get("contact_trace"), list)
                else [],
                "route_waypoints": observed.get("route_waypoints") or run.get("route_waypoints"),
                "artifact_paths": _mapping(
                    observed.get("artifact_paths") or observed.get("artifactPaths")
                ),
                "proof_boundary": (
                    "Attempt row comes from an Isaac runtime result. It proves only the "
                    "Isaac-specific runtime artifacts represented by this row; it does "
                    "not prove MuJoCo, real robot readiness, deployment approval, safety "
                    "validation, WAM consistency, or generated-world rank fidelity."
                ),
            }
            attempts.append(attempt)
            if not success:
                labels.append(
                    {
                        "label_id": f"isaac_failure_label_{index:04d}",
                        "attempt_id": attempt["attempt_id"],
                        "scenario_eval_run_id": run.get("scenario_eval_run_id"),
                        "task_id": run.get("task_id"),
                        "spawn_id": run.get("spawn_id"),
                        "primary_failure_label": failure_ids[0]
                        if failure_ids
                        else "isaac_runtime_attempt_failed",
                        "failure_label_ids": sorted(set(failure_ids)),
                        "failure_mode_ids": sorted(set(failure_ids)),
                        "status": "failed" if status != "blocked" else "blocked",
                    }
                )
            continue
        failure_ids = [blocker, "blocked_missing_wam_vla_policy_runtime"]
        if _string(run.get("task_id")) in {"desk_object_contact_check", "carry_object_to_drop_zone"}:
            failure_ids.append("blocked_isaac_contact_dynamics_not_executed")
        attempt = {
            "attempt_id": attempt_id,
            "scenario_eval_run_id": run.get("scenario_eval_run_id"),
            "scenario_id": run.get("scenario_id"),
            "scenario_variation_instance_id": run.get("scenario_variation_instance_id"),
            "task_id": run.get("task_id"),
            "spawn_id": run.get("spawn_id"),
            "episode_id": run.get("episode_id"),
            "status": "blocked",
            "success": False,
            "task_success": False,
            "task_status": "blocked",
            "failure_label_ids": sorted(set(failure_ids)),
            "failure_mode_ids": sorted(set(failure_ids)),
            "generated_at": generated_at,
            "simulator": "isaac_sim",
            "simulator_backend": "isaac_sim",
            "robot_profile": "unitree_g1_official_isaac_asset_expected",
            "policy_id": "isaac_g1_runtime_policy_blocked",
            "camera_ids": camera_ids,
            "camera_evidence": {
                camera_id: {
                    "status": "blocked",
                    "blocker": "runtime_video_not_generated",
                }
                for camera_id in camera_ids
            },
            "metrics": {
                "navigation_success": False,
                "visual_inspection_evidence": False,
                "object_contact_or_displacement_validated": False,
                "route_non_ranking_operational_claim_validated": False,
                "collision_contact_scored": False,
                "blocked_failure_preserved": True,
                "isaac_runtime_attempt_trace_present": False,
                "isaac_runtime_executed": bool(runtime_result_payload.get("isaac_runtime_executed")),
            },
            "task_outcome": {
                "task_success": False,
                "task_status": "blocked",
                "failure_mode_ids": sorted(set(failure_ids)),
            },
            "actions": [],
            "action_trace": [],
            "contact_trace": [],
            "route_waypoints": run.get("route_waypoints"),
            "artifact_paths": {},
            "proof_boundary": (
                "Blocked attempt row preserves the requested matrix item. It is not a "
                "simulated success, walking proof, contact proof, or WAM/VLA decision."
            ),
        }
        attempts.append(attempt)
        labels.append(
            {
                "label_id": f"isaac_failure_label_{index:04d}",
                "attempt_id": attempt_id,
                "scenario_eval_run_id": run.get("scenario_eval_run_id"),
                "task_id": run.get("task_id"),
                "spawn_id": run.get("spawn_id"),
                "primary_failure_label": failure_ids[0],
                "failure_label_ids": sorted(set(failure_ids)),
                "failure_mode_ids": sorted(set(failure_ids)),
                "status": "blocked",
            }
        )
    return attempts, labels


def _counts_by_key(attempts: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, int]] = {}
    for attempt in attempts:
        value = _string(attempt.get(key)) or "unknown"
        row = grouped.setdefault(value, {"attempted": 0, "successful": 0, "failed": 0, "blocked": 0})
        row["attempted"] += 1
        if attempt.get("status") == "blocked":
            row["blocked"] += 1
        elif attempt.get("success") is True:
            row["successful"] += 1
        else:
            row["failed"] += 1
    return [{key: value, **counts} for value, counts in sorted(grouped.items())]


def _video_manifest(
    *,
    job_dir: Path,
    matrix: Mapping[str, Any],
    camera_ids: Sequence[str],
    generated_at: str,
    runtime: Mapping[str, Any],
    runtime_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    videos_dir = job_dir / "realistic_videos"
    posters_dir = job_dir / "realistic_posters"
    ensure_dir(videos_dir)
    ensure_dir(posters_dir)
    rows: list[dict[str, Any]] = []
    posters: list[dict[str, Any]] = []
    runtime_result_payload = _mapping(runtime_result)
    observed_by_run_id = {
        _string(row.get("scenario_eval_run_id") or row.get("scenarioEvalRunId")): row
        for row in _runtime_result_attempts(runtime_result_payload)
        if _string(row.get("scenario_eval_run_id") or row.get("scenarioEvalRunId"))
    }
    blocker = (
        "isaac_runtime_execution_not_run"
        if runtime.get("isaac_runtime_available")
        else "isaac_sim_runtime_unavailable"
    )

    def runtime_video_path(observed: Mapping[str, Any], camera_id: str, camera_index: int) -> str:
        artifacts = _mapping(observed.get("artifact_paths") or observed.get("artifactPaths"))
        keyed = artifacts.get(f"{camera_id}_video") or artifacts.get(f"{camera_id}_mp4")
        if keyed:
            return _string(keyed)
        video_paths = artifacts.get("video_paths") or artifacts.get("videoPaths")
        if isinstance(video_paths, Mapping):
            mapped = video_paths.get(camera_id)
            if mapped:
                return _string(mapped)
        for key in ("videos", "video_records", "videoRecords"):
            raw_videos = observed.get(key) or artifacts.get(key)
            if not isinstance(raw_videos, Sequence) or isinstance(raw_videos, (str, bytes)):
                continue
            for raw_video in raw_videos:
                video = _mapping(raw_video)
                if _string(video.get("camera_id") or video.get("cameraId")) == camera_id:
                    return _string(video.get("path") or video.get("uri") or video.get("url"))
        generic = (
            artifacts.get("video_path")
            or artifacts.get("videoPath")
            or artifacts.get("robot_pov_video")
            or artifacts.get("robotPovVideo")
            or observed.get("video_path")
            or observed.get("videoPath")
        )
        return _string(generic) if camera_index == 0 else ""

    for run in matrix.get("runs") or []:
        if not isinstance(run, Mapping):
            continue
        episode_id = _string(run.get("episode_id"))
        observed = _mapping(observed_by_run_id.get(_string(run.get("scenario_eval_run_id"))))
        run_camera_ids = _string_list(run.get("camera_ids")) or list(camera_ids)
        for camera_index, camera_id in enumerate(run_camera_ids):
            runtime_path = runtime_video_path(observed, camera_id, camera_index)
            runtime_file_exists = bool(runtime_path and Path(runtime_path).expanduser().is_file())
            status = "completed" if runtime_file_exists else "blocked"
            row_blockers = [] if runtime_file_exists else [blocker, "splat_or_isaac_video_not_rendered"]
            rows.append(
                {
                    "episode_id": episode_id,
                    "camera_id": camera_id,
                    "path": runtime_path or str(videos_dir / f"{episode_id}__{camera_id}.mp4"),
                    "status": status,
                    "blockers": row_blockers,
                    "file_created": runtime_file_exists,
                    "source": "isaac_runtime_result" if runtime_file_exists else "expected_output_path",
                    "camera_ids_source": "scenario_eval_matrix_row"
                    if _string_list(run.get("camera_ids"))
                    else "requested_camera_manifest",
                }
            )
            posters.append(
                {
                    "episode_id": episode_id,
                    "camera_id": camera_id,
                    "path": str(posters_dir / f"{episode_id}__{camera_id}.png"),
                    "status": "blocked",
                    "blockers": [blocker, "poster_frame_not_rendered"],
                    "file_created": False,
                }
            )
    completed_video_count = sum(1 for row in rows if row.get("status") == "completed")
    expected_video_count = len(rows)
    video_complete = completed_video_count == expected_video_count and expected_video_count > 0
    return {
        "schema_version": REALISTIC_VIDEO_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if video_complete else "blocked_no_runtime_videos",
        "realistic_splat_visual_rendered": bool(
            runtime_result_payload.get("realistic_splat_visual_rendered")
        ),
        "isaac_state_synchronized_video_rendered": video_complete,
        "video_count": completed_video_count,
        "expected_video_count": expected_video_count,
        "poster_count": 0,
        "expected_poster_count": len(posters),
        "realistic_videos_dir": str(videos_dir),
        "realistic_posters_dir": str(posters_dir),
        "videos": rows,
        "posters": posters,
    }


def _wam_vla_discovery(*, generated_at: str) -> dict[str, Any]:
    command_envs = {
        name: _string(os.getenv(name))
        for name in (
            "BLUEPRINT_WAM_PROVIDER_COMMAND",
            "BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND",
            "BLUEPRINT_OSCAR_WAM_PROVIDER_COMMAND",
            "BLUEPRINT_VLA_POLICY_COMMAND",
        )
    }
    endpoint_envs = {
        name: _string(os.getenv(name))
        for name in (
            "BLUEPRINT_WAM_RUNTIME_URL",
            "BLUEPRINT_COSMOS3_WAM_ENDPOINT",
            "BLUEPRINT_OSCAR_WAM_ENDPOINT",
            "BLUEPRINT_VLA_POLICY_ENDPOINT",
        )
    }
    auth_env_present = {
        name: bool(os.getenv(name))
        for name in (
            "BLUEPRINT_COSMOS3_WAM_API_KEY",
            "COSMOS3_WAM_API_KEY",
            "BLUEPRINT_OSCAR_WAM_API_KEY",
            "OSCAR_WAM_API_KEY",
        )
    }
    configured_commands = {key: value for key, value in command_envs.items() if value}
    configured_endpoints = {key: value for key, value in endpoint_envs.items() if value}
    proven = False
    blockers = ["blocked_missing_wam_vla_policy_runtime"]
    if configured_commands or configured_endpoints:
        blockers = ["wam_vla_runtime_config_present_but_not_executed_in_this_pass"]
    return {
        "schema_version": WAM_VLA_DISCOVERY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked",
        "wam_vla_runtime_proven": proven,
        "configured_command_envs": {key: bool(value) for key, value in command_envs.items()},
        "configured_endpoint_envs": {key: bool(value) for key, value in endpoint_envs.items()},
        "auth_env_present": auth_env_present,
        "raw_secret_values_recorded": False,
        "blockers": blockers,
    }


def _manipulation_objects(proxies: Sequence[Mapping[str, Any]], generated_at: str) -> dict[str, Any]:
    candidates = [
        dict(proxy)
        for proxy in proxies
        if _string(proxy.get("category")) in {"labeled_object", "desk", "manipulation_object"}
    ]
    selected = candidates[:12]
    return {
        "schema_version": MANIPULATION_OBJECT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed_scene_grounded_candidates",
        "object_count": len(selected),
        "objects": selected,
        "dexterous_hand_policy_proven": False,
    }


def _build_summary(
    *,
    job_id: str,
    generated_at: str,
    attempts: Sequence[Mapping[str, Any]],
    matrix: Mapping[str, Any],
    runtime: Mapping[str, Any],
    runtime_result: Mapping[str, Any] | None = None,
    scene_visual_source: str,
    reconstructed_triangle_mesh_loaded: bool,
) -> dict[str, Any]:
    success_count = sum(1 for attempt in attempts if attempt.get("success") is True)
    blocked_count = sum(1 for attempt in attempts if attempt.get("status") == "blocked")
    failed_count = sum(
        1
        for attempt in attempts
        if attempt.get("status") != "blocked" and attempt.get("success") is not True
    )
    matrix_count = int(matrix.get("scenario_eval_run_count") or 0)
    runtime_result_payload = _mapping(runtime_result)
    observed_runtime_attempts = bool(_runtime_result_attempts(runtime_result_payload))
    if observed_runtime_attempts and blocked_count == 0 and failed_count == 0:
        status = "completed"
    elif observed_runtime_attempts and blocked_count < len(attempts):
        status = "completed_with_failures"
    elif not runtime.get("isaac_runtime_available") and not runtime_result_payload.get(
        "isaac_runtime_executed"
    ):
        status = "blocked_runtime_unavailable"
    else:
        status = "blocked_controller_runtime_unavailable"
    version_probe = _mapping(runtime.get("version_probe"))
    simulator_version = (
        _string(runtime_result_payload.get("simulator_version"))
        or _string(runtime_result_payload.get("isaac_sim_version"))
        or _string(version_probe.get("version"))
        or _string(version_probe.get("stdout"))
        or None
    )
    return {
        "schema_version": POLICY_EVALUATION_SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": status,
        "simulator_backend": "isaac_sim",
        "simulator_version": simulator_version,
        "isaac_runtime_available": bool(runtime.get("isaac_runtime_available"))
        or bool(runtime_result_payload.get("isaac_runtime_executed")),
        "scene_visual_source": scene_visual_source,
        "realistic_splat_visual_rendered": False,
        "reconstructed_triangle_mesh_loaded": reconstructed_triangle_mesh_loaded,
        "isaac_usd_scene_loaded": bool(runtime_result_payload.get("isaac_usd_scene_loaded")),
        "isaac_collision_scene_source": "metadata_derived_collider_proxy",
        "unitree_g1_model_source": f"official_isaac_asset_expected:{OFFICIAL_ISAAC_G1_ASSET_PATH}",
        "unitree_g1_loaded_in_isaac": bool(
            runtime_result_payload.get("unitree_g1_loaded_in_isaac")
            or runtime_result_payload.get("isaac_robot_asset_execution_proven")
        ),
        "controller_grade_execution_proven": bool(
            runtime_result_payload.get("controller_grade_execution_proven")
        ),
        "official_policy_execution_proven": bool(
            runtime_result_payload.get("official_policy_execution_proven")
        ),
        "locomotion_continuity_validated": bool(
            runtime_result_payload.get("locomotion_continuity_validated")
        ),
        "collision_dynamics_validated": bool(
            runtime_result_payload.get("collision_dynamics_validated")
        ),
        "manipulation_contact_dynamics_validated": bool(
            runtime_result_payload.get("manipulation_contact_dynamics_validated")
        ),
        "wam_vla_runtime_proven": bool(runtime_result_payload.get("wam_vla_runtime_proven")),
        "wam_evaluator_trace_scored": True,
        "attempted_episode_count": len(attempts),
        "successful_episode_count": success_count,
        "failed_episode_count": failed_count,
        "blocked_episode_count": blocked_count,
        "scenario_eval_run_coverage_complete": len(attempts) == matrix_count and matrix_count > 0,
        "attempt_count_matches_matrix_count": len(attempts) == matrix_count,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "pass_fail_blocked_by_task": _counts_by_key(attempts, "task_id"),
        "pass_fail_blocked_by_spawn": _counts_by_key(attempts, "spawn_id"),
    }


def _local_validation_report(
    *,
    job_id: str,
    generated_at: str,
    job_dir: Path,
    matrix: Mapping[str, Any],
    trace: Mapping[str, Any],
    summary: Mapping[str, Any],
    video_manifest: Mapping[str, Any],
    gpu_provider_launch_request: Mapping[str, Any],
    provider_bundle_readiness: Mapping[str, Any],
) -> dict[str, Any]:
    matrix_rows = [row for row in matrix.get("runs") or [] if isinstance(row, Mapping)]
    attempts = [row for row in trace.get("attempts") or [] if isinstance(row, Mapping)]
    matrix_ids = {_string(row.get("scenario_eval_run_id")) for row in matrix_rows}
    attempt_ids = {_string(row.get("scenario_eval_run_id")) for row in attempts}
    missing_attempt_rows = sorted(item for item in matrix_ids - attempt_ids if item)
    blocked_or_failed = [
        row for row in attempts if row.get("status") in {"blocked", "failed"}
    ]
    runtime_attempts_present = any(row.get("status") != "blocked" for row in attempts)
    missing_blocked_or_failed_labels = [
        _string(row.get("attempt_id"))
        for row in blocked_or_failed
        if not row.get("failure_label_ids")
    ]
    runtime_allowed_flags = [
        "controller_grade_execution_proven",
        "official_policy_execution_proven",
        "locomotion_continuity_validated",
        "collision_dynamics_validated",
        "manipulation_contact_dynamics_validated",
    ]
    never_promoted_flags = [
        "generated_world_rank_fidelity_result_proven",
        "generated_world_policy_evaluation_scope_proven",
        "wam_vla_runtime_proven",
    ]
    proof_flags_honest = all(summary.get(flag) is False for flag in never_promoted_flags) and (
        runtime_attempts_present
        or all(summary.get(flag) is False for flag in runtime_allowed_flags)
    )
    mp4_paths = sorted((job_dir / "realistic_videos").glob("*.mp4"))
    video_rows = [
        _mapping(row)
        for row in video_manifest.get("videos", []) or []
        if isinstance(row, Mapping)
    ]
    declared_video_files = [
        Path(_string(row.get("path"))).expanduser()
        for row in video_rows
        if row.get("status") == "completed" and _string(row.get("path"))
    ]
    declared_video_files_present = bool(declared_video_files) and all(
        path.is_file() and path.stat().st_size > 0 for path in declared_video_files
    )
    declared_video_coverage_complete = bool(
        video_manifest.get("status") == "completed"
        and int(video_manifest.get("video_count") or 0)
        == int(video_manifest.get("expected_video_count") or 0)
        and int(video_manifest.get("expected_video_count") or 0) > 0
        and declared_video_files_present
    )
    mp4_file_rows = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "ffprobe_required": False,
        }
        for path in sorted({*mp4_paths, *declared_video_files})
        if path.is_file()
    ]
    checks = {
        "attempt_count_matches_matrix_count": bool(
            summary.get("attempt_count_matches_matrix_count")
        ),
        "all_matrix_rows_appear_in_normalized_attempt_trace": not missing_attempt_rows,
        "blocked_or_failed_attempts_remain_represented": not missing_blocked_or_failed_labels,
        "proof_flags_honest": proof_flags_honest,
        "scenario_eval_run_coverage_complete": bool(
            summary.get("scenario_eval_run_coverage_complete")
        ),
        "mp4_validation_complete": not mp4_file_rows or declared_video_coverage_complete,
        "provider_request_shape_written": bool(gpu_provider_launch_request),
        "provider_bundle_local_ready_for_remote_staging": runtime_attempts_present
        or bool(provider_bundle_readiness.get("local_bundle_ready_for_remote_staging")),
    }
    return {
        "schema_version": LOCAL_VALIDATION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "passed_with_runtime_blockers" if all(checks.values()) else "failed",
        "checks": checks,
        "matrix_count": len(matrix_rows),
        "attempt_count": len(attempts),
        "missing_attempt_scenario_eval_run_ids": missing_attempt_rows,
        "missing_blocked_or_failed_label_attempt_ids": missing_blocked_or_failed_labels,
        "video_manifest_path": str(job_dir / "realistic_video_manifest.json"),
        "mp4_validation": {
            "status": "not_applicable_no_mp4_generated"
            if not mp4_file_rows
            else "ffprobe_required_for_generated_mp4s",
            "mp4_file_count": len(mp4_file_rows),
            "files": mp4_file_rows,
            "video_manifest_expected_video_count": video_manifest.get("expected_video_count"),
            "video_manifest_video_count": video_manifest.get("video_count"),
            "declared_video_coverage_complete": declared_video_coverage_complete,
        },
        "provider_launch_request": {
            "path": str(job_dir / "gpu_provider_launch_request.json"),
            "status": gpu_provider_launch_request.get("status"),
            "blockers": list(gpu_provider_launch_request.get("blockers") or []),
        },
        "provider_bundle_readiness": {
            "path": str(job_dir / "isaac_provider_bundle_readiness.json"),
            "status": provider_bundle_readiness.get("status"),
            "local_bundle_ready_for_remote_staging": provider_bundle_readiness.get(
                "local_bundle_ready_for_remote_staging"
            ),
            "ready_for_next_authorized_vast_bundle_attempt": provider_bundle_readiness.get(
                "ready_for_next_authorized_vast_bundle_attempt"
            ),
            "blockers": list(provider_bundle_readiness.get("blockers") or []),
        },
        "proof_boundary": (
            "Local validation checks artifact consistency and fail-closed proof flags. It "
            "does not prove Isaac runtime execution, rendered MP4 output, controller-grade "
            "G1 locomotion, contact dynamics, WAM/VLA runtime, generated-world rank fidelity, "
            "or generated-world rank fidelity."
        ),
    }


def run_isaac_g1_site_3dgs_realistic_eval(
    *,
    ply_asset: str | Path,
    spz_asset: str | Path,
    labels_json: str | Path | None = None,
    structure_json: str | Path | None = None,
    occupancy_json: str | Path | None = None,
    occupancy_png: str | Path | None = None,
    job_id: str | None = None,
    job_root: str | Path | None = None,
    task_limit: int | None = None,
    spawn_limit: int | None = None,
    camera_ids: Sequence[str] | None = None,
    allow_cloud_gpu: bool = False,
    scenario_eval_matrix_path: str | Path | None = None,
    simulator_output_path: str | Path | None = None,
    runtime_result_path: str | Path | None = None,
    render_splat_views: bool = False,
    splat_render_options: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    timestamp = (
        generated.replace("+00:00", "Z")
        .replace(":", "")
        .replace("-", "")
        .replace(".", "")
    )
    resolved_job_id = job_id or f"isaac_g1_site_3dgs_realistic_wam_vla_eval_{timestamp}"
    root = Path(job_root).expanduser().resolve() if job_root else _repo_root() / "robot_eval_jobs"
    job_dir = root / resolved_job_id
    ensure_dir(job_dir)

    camera_id_list, camera_aliases = _normalize_camera_ids(camera_ids or DEFAULT_CAMERA_IDS)
    unknown_cameras = sorted(set(camera_id_list) - set(DEFAULT_CAMERA_IDS))
    if unknown_cameras:
        raise ValueError(f"unknown Isaac camera ids: {', '.join(unknown_cameras)}")

    runtime = detect_isaac_runtime(generated_at=generated)
    provider_plan = build_provider_plan(
        runtime=runtime,
        job_id=resolved_job_id,
        job_dir=job_dir,
        allow_cloud_gpu=allow_cloud_gpu,
        generated_at=generated,
    )
    cost_ledger = _cost_ledger(
        job_id=resolved_job_id,
        generated_at=generated,
        allow_cloud_gpu=allow_cloud_gpu,
    )
    teardown = _teardown_manifest(
        job_id=resolved_job_id,
        generated_at=generated,
        allow_cloud_gpu=allow_cloud_gpu,
    )
    ply_inspection = inspect_scene_asset(ply_asset)
    spz_inspection = inspect_scene_asset(spz_asset)
    optional_metadata = [
        inspect_optional_scene_metadata(labels_json, metadata_kind="labels_json"),
        inspect_optional_scene_metadata(structure_json, metadata_kind="structure_json"),
        inspect_optional_scene_metadata(occupancy_json, metadata_kind="occupancy_json"),
        inspect_optional_scene_metadata(occupancy_png, metadata_kind="occupancy_png"),
    ]
    scene_visual_source = (
        "gaussian_ply_and_spz"
        if ply_inspection.get("exists") and spz_inspection.get("exists")
        else "spz"
        if spz_inspection.get("exists")
        else "gaussian_ply"
    )
    proxies, proxy_source_manifest, scene_anchors = build_scene_proxies_from_metadata(
        labels_json=labels_json,
        structure_json=structure_json,
        occupancy_json=occupancy_json,
    )
    conversion_attempts = _attempt_visual_asset_mesh_conversion(
        ply_asset=ply_asset,
        spz_asset=spz_asset,
        output_dir=job_dir / "scene_conversion_attempts",
        ply_inspection=ply_inspection,
        spz_inspection=spz_inspection,
    )
    reconstructed_triangle_mesh_loaded = any(
        attempt.get("status") == "completed" and int(attempt.get("face_count") or 0) > 0
        for attempt in conversion_attempts
    )
    usd_scene = _write_usd_scene(
        usda_path=job_dir / "generated_site_scene.usda",
        usd_path=job_dir / "generated_site_scene.usd",
        proxies=proxies,
        ply_asset=ply_asset,
        spz_asset=spz_asset,
        generated_at=generated,
    )
    runtime_result = (
        _json_mapping(Path(runtime_result_path).expanduser().resolve())
        if runtime_result_path
        else {}
    )
    if scenario_eval_matrix_path:
        matrix = _scenario_eval_matrix_from_path(
            scenario_eval_matrix_path,
            camera_ids=camera_id_list,
            job_id=resolved_job_id,
        )
    else:
        tasks = _apply_required_task_families(build_default_tasks(anchors=scene_anchors))
        spawns = build_default_spawns(anchors=scene_anchors)
        if task_limit is not None:
            tasks = tasks[: max(0, task_limit)]
        if spawn_limit is not None:
            spawns = spawns[: max(0, spawn_limit)]
        spawn_manifest = validate_spawns(spawns=spawns, proxies=proxies, generated_at=generated)
        matrix = build_scenario_eval_matrix(
            job_id=resolved_job_id,
            tasks=tasks,
            spawns=spawn_manifest.get("spawns") or [],
            camera_ids=camera_id_list,
            generated_at=generated,
            route_planner_proxies=proxies,
        )
    camera_manifest = _camera_manifest(
        camera_ids=camera_id_list,
        generated_at=generated,
        camera_aliases=camera_aliases,
    )
    # Optional local splat render: actually display the captured Gaussian-splat scene
    # (reference Spark renderer, not Isaac RTX) so the eval emits real per-camera frames
    # of the real environment instead of metadata-only placeholders. Off by default;
    # the Isaac RTX/NuRec render remains the Phase-2 GPU proof.
    splat_render_manifest: dict | None = None
    if render_splat_views:
        from .splat_scene_render import attach_splat_render_to_eval

        splat_render_manifest = attach_splat_render_to_eval(
            job_dir=job_dir,
            ply_asset=ply_asset,
            spz_asset=spz_asset,
            camera_ids=camera_id_list,
            generated_at=generated,
            options=dict(splat_render_options or {}),
        )
        if splat_render_manifest.get("status") == "completed":
            camera_manifest["camera_evidence_status"] = (
                "rendered_by_reference_spark_renderer_isaac_rtx_pending"
            )
            camera_manifest["reference_splat_render"] = {
                "rendered_by": splat_render_manifest.get("rendered_by"),
                "rendered_by_isaac_rtx": False,
                "nonblank_camera_count": splat_render_manifest.get("nonblank_camera_count"),
                "frames": [cam.get("path") for cam in splat_render_manifest.get("cameras", [])],
                "robot_start_pose": splat_render_manifest.get("robot_start_pose"),
            }
    realistic_splat_visual_rendered = bool(
        splat_render_manifest and splat_render_manifest.get("status") == "completed"
    )
    episode_manifest = {
        "schema_version": EPISODE_SPEC_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed",
        "episode_count": matrix.get("scenario_eval_run_count"),
        "episodes": [
            {
                "episode_id": run.get("episode_id"),
                "scenario_eval_run_id": run.get("scenario_eval_run_id"),
                "task_id": run.get("task_id"),
                "spawn_id": run.get("spawn_id"),
                "camera_ids": run.get("camera_ids"),
                "route_waypoints": run.get("route_waypoints"),
            }
            for run in matrix.get("runs", [])
        ],
    }
    attempts, failure_rows = _attempts_for_matrix(
        matrix=matrix,
        runtime=runtime,
        runtime_result=runtime_result,
        generated_at=generated,
    )
    coverage = _coverage_summary(matrix=matrix, attempts=attempts)
    summary = _build_summary(
        job_id=resolved_job_id,
        generated_at=generated,
        attempts=attempts,
        matrix=matrix,
        runtime=runtime,
        runtime_result=runtime_result,
        scene_visual_source=scene_visual_source,
        reconstructed_triangle_mesh_loaded=reconstructed_triangle_mesh_loaded,
    )
    summary.update(coverage)
    video_manifest = _video_manifest(
        job_dir=job_dir,
        matrix=matrix,
        camera_ids=camera_id_list,
        generated_at=generated,
        runtime=runtime,
        runtime_result=runtime_result,
    )
    wam_discovery = _wam_vla_discovery(generated_at=generated)
    manipulation_objects = _manipulation_objects(proxies, generated)
    phase_rows = _phase_rows(runtime=runtime, generated_at=generated, artifacts_exported=True)

    trace = {
        "schema_version": NORMALIZED_TRACE_SCHEMA_VERSION,
        "generated_at": generated,
        "job_id": resolved_job_id,
        "attempt_count": len(attempts),
        "scenario_eval_matrix_count": matrix.get("scenario_eval_run_count"),
        "attempt_count_matches_matrix_count": summary["attempt_count_matches_matrix_count"],
        **coverage,
        "attempts": attempts,
    }
    failure_labels = {
        "schema_version": FAILURE_LABELS_SCHEMA_VERSION,
        "generated_at": generated,
        "job_id": resolved_job_id,
        "status": "blocked_attempts_labeled",
        "label_count": len(failure_rows),
        "failed_or_blocked_attempt_count": len(failure_rows),
        **coverage,
        "labels": failure_rows,
    }
    thresholds = {
        "schema_version": WAM_EVALUATOR_THRESHOLDS_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed",
        "thresholds": {
            "navigation_success": {"target_tolerance_m": 0.55, "route_safety_required": True},
            "visual_inspection_evidence": {"camera_evidence_required": True},
            "object_contact_displacement": {"isaac_contact_trace_required": True},
            "blocked_failure_preservation": {"blocked_rows_must_remain": True},
        },
    }
    trace_binding = {
        "schema_version": WAM_EVALUATOR_TRACE_BINDING_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed",
        "matrix_count": matrix.get("scenario_eval_run_count"),
        "attempt_count": len(attempts),
        "all_matrix_rows_have_attempts": summary["attempt_count_matches_matrix_count"],
        "scenario_eval_run_coverage_complete": coverage["scenario_eval_run_coverage_complete"],
        "bindings": [
            {
                "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
                "attempt_id": attempt.get("attempt_id"),
                "status": attempt.get("status"),
            }
            for attempt in attempts
        ],
    }
    evaluator_results = {
        "schema_version": WAM_EVALUATOR_RESULTS_SCHEMA_VERSION,
        "generated_at": generated,
        "job_id": resolved_job_id,
        "status": "completed_blocked_trace_scored",
        "wam_evaluator_trace_scored": True,
        "attempt_count": len(attempts),
        "passed_count": summary["successful_episode_count"],
        "failed_count": summary["failed_episode_count"],
        "blocked_count": summary["blocked_episode_count"],
        "results": [
            {
                "attempt_id": attempt.get("attempt_id"),
                "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
                "task_id": attempt.get("task_id"),
                "spawn_id": attempt.get("spawn_id"),
                "passed": False,
                "status": attempt.get("status"),
                "reasons": attempt.get("failure_label_ids"),
            }
            for attempt in attempts
        ],
    }
    contact_traces = [
        contact
        for attempt in attempts
        for contact in (attempt.get("contact_trace") or [])
        if isinstance(contact, Mapping)
    ]
    collision_contact_report = {
        "schema_version": COLLISION_CONTACT_REPORT_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed"
        if runtime_result.get("collision_dynamics_validated")
        else "blocked",
        "simulator_backend": "isaac_sim",
        "collision_dynamics_validated": bool(
            runtime_result.get("collision_dynamics_validated")
        ),
        "contact_count": int(runtime_result.get("contact_count") or len(contact_traces) or 0),
        "contacts": [dict(contact) for contact in contact_traces[:200]],
        "blockers": []
        if runtime_result.get("collision_dynamics_validated")
        else ["isaac_contact_solver_not_executed"],
        "proof_boundary": (
            "This report is Isaac-specific contact/collision evidence. It does not prove "
            "MuJoCo contact behavior, real-world contact safety, deployment approval, "
            "physical robot readiness, WAM consistency, or generated-world rank fidelity."
        ),
    }
    unitree_g1_loaded = bool(summary.get("unitree_g1_loaded_in_isaac"))
    controller_proven = bool(summary.get("controller_grade_execution_proven"))
    official_policy_proven = bool(summary.get("official_policy_execution_proven"))
    locomotion_validated = bool(summary.get("locomotion_continuity_validated"))
    contact_validated = bool(summary.get("collision_dynamics_validated"))
    manipulation_contact_validated = bool(
        summary.get("manipulation_contact_dynamics_validated")
    )
    asset_blockers = [] if unitree_g1_loaded else ["isaac_runtime_unavailable_or_not_executed"]
    controller_blockers = (
        []
        if controller_proven and official_policy_proven
        else ["real_continuous_g1_controller_runtime_not_configured"]
    )
    physics_blockers = [] if contact_validated else ["isaac_physics_not_executed"]

    if realistic_splat_visual_rendered:
        splat_visual_render_payload: dict[str, Any] = {
            "schema_version": SPLAT_VISUAL_RENDER_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "completed_reference_render",
            "realistic_splat_visual_rendered": True,
            "rendered_by": (splat_render_manifest or {}).get("rendered_by"),
            "rendered_by_isaac_rtx": False,
            "visual_source_preserved": True,
            "nonblank_camera_count": (splat_render_manifest or {}).get("nonblank_camera_count"),
            "reference_render_manifest_path": "splat_scene_render/manifest.json",
            "scene_geometry": (splat_render_manifest or {}).get("scene_geometry"),
            "robot_start_pose": (splat_render_manifest or {}).get("robot_start_pose"),
            "blockers": [],
            "proof_boundary": (
                "Reference Spark (three.js) Gaussian render of the real captured scene. "
                "It proves the splat displays, NOT that Isaac RTX/NuRec rendered it "
                "(that is the Phase-2 GPU proof), and not physics, navigation, or readiness."
            ),
        }
    else:
        splat_visual_render_payload = {
            "schema_version": SPLAT_VISUAL_RENDER_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "realistic_splat_visual_rendered": False,
            "visual_source_preserved": True,
            "blockers": ["splat_renderer_or_isaac_composite_runtime_not_executed"],
        }
    artifact_payloads: dict[str, Mapping[str, Any]] = {
        "isaac_runtime_discovery.json": runtime,
        "isaac_provider_plan.json": provider_plan,
        "isaac_gpu_cost_control_ledger.json": cost_ledger,
        "isaac_teardown_manifest.json": teardown,
        "isaac_scene_asset_inspection.json": {
            "schema_version": ISAAC_SCENE_INSPECTION_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "completed",
            "assets": [ply_inspection, spz_inspection],
            "optional_scene_metadata": optional_metadata,
            "scene_visual_source": scene_visual_source,
        },
        "splat_visual_render_manifest.json": splat_visual_render_payload,
        "usd_scene_assembly_manifest.json": {
            "schema_version": USD_SCENE_ASSEMBLY_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "completed_review_usd_not_loaded",
            "isaac_usd_scene_loaded": False,
            **usd_scene,
            "blockers": ["isaac_runtime_scene_load_not_executed"],
        },
        "collider_proxy_plan.json": {
            "schema_version": COLLIDER_PROXY_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "completed",
            "isaac_collision_scene_source": "metadata_derived_collider_proxy",
            "proxy_source_manifest": proxy_source_manifest,
            "proxy_count": len(proxies),
            "proxies": [dict(proxy) for proxy in proxies],
            "direct_splat_collision_claimed": False,
        },
        "scene_conversion_report.json": {
            "schema_version": SCENE_CONVERSION_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "completed_with_fail_closed_proxy",
            "scene_visual_source": scene_visual_source,
            "generated_site_scene_usda": str(job_dir / "generated_site_scene.usda"),
            "generated_site_scene_usd": str(job_dir / "generated_site_scene.usd"),
            "reconstructed_triangle_mesh_loaded": reconstructed_triangle_mesh_loaded,
            "direct_splat_collision_claimed": False,
            "conversion_attempts": conversion_attempts,
        },
        "visual_collision_alignment_manifest.json": {
            "schema_version": VISUAL_COLLISION_ALIGNMENT_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "metadata_proxy_alignment_prepared",
            "visual_source": scene_visual_source,
            "collision_source": "labels_structure_occupancy_metadata",
            "direct_splat_to_collider_alignment_proven": False,
            "runtime_camera_alignment_validated": False,
        },
        "visual_truth_boundary.json": {
            "schema_version": VISUAL_TRUTH_BOUNDARY_SCHEMA_VERSION,
            "generated_at": generated,
            "splat_visual_source_preserved": True,
            "realistic_splat_visual_rendered": realistic_splat_visual_rendered,
            "realistic_splat_visual_rendered_by_isaac_rtx": False,
            "colliders_are_metadata_derived_proxy": True,
            "direct_splat_collision_claimed": False,
            "triangle_mesh_reconstruction_claimed": reconstructed_triangle_mesh_loaded,
        },
        "unitree_g1_asset_source_manifest.json": {
            "schema_version": G1_ASSET_SOURCE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "completed" if unitree_g1_loaded else "blocked_runtime_resolution_not_run",
            "unitree_g1_model_source": "official_isaac_sim_robot_assets_expected",
            "official_asset_path": OFFICIAL_ISAAC_G1_ASSET_PATH,
            "official_doc_url": OFFICIAL_ISAAC_G1_DOC_URL,
            "unitree_g1_loaded_in_isaac": unitree_g1_loaded,
            "blockers": asset_blockers,
        },
        "g1_controller_runtime_manifest.json": {
            "schema_version": G1_CONTROLLER_RUNTIME_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "completed"
            if controller_proven and official_policy_proven
            else "blocked_controller_runtime_unavailable",
            "controller_grade_execution_proven": controller_proven,
            "official_policy_execution_proven": official_policy_proven,
            "root_pose_teleporting_used_as_success_evidence": False,
            "blockers": controller_blockers,
        },
        "foot_contact_trace.json": {
            "schema_version": FOOT_CONTACT_TRACE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "completed" if contact_validated else "blocked",
            "foot_contact_validated": contact_validated,
            "contacts": [dict(contact) for contact in contact_traces[:200]],
            "blockers": physics_blockers,
        },
        "root_motion_continuity_report.json": {
            "schema_version": ROOT_MOTION_CONTINUITY_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "completed" if locomotion_validated else "blocked",
            "locomotion_continuity_validated": locomotion_validated,
            "large_root_jumps_detected": False if locomotion_validated else None,
            "blockers": []
            if locomotion_validated
            else ["g1_locomotion_trace_missing_because_controller_not_run"],
        },
        "collision_contact_report.json": collision_contact_report,
        "controller_grade_proof_manifest.json": {
            "schema_version": CONTROLLER_GRADE_PROOF_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "completed"
            if controller_proven and official_policy_proven
            else "blocked_controller_runtime_unavailable",
            "controller_grade_execution_proven": controller_proven,
            "official_policy_execution_proven": official_policy_proven,
            "blockers": []
            if controller_proven and official_policy_proven
            else ["real_unitree_g1_controller_policy_stack_not_run"],
        },
        "manipulation_scene_object_manifest.json": manipulation_objects,
        "manipulation_action_spec_manifest.json": {
            "schema_version": MANIPULATION_ACTION_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "specified_blocked_until_isaac_physics_runtime",
            "actions": [
                {
                    "action_id": "approach_and_push_lightweight_object",
                    "action_type": "scene_grounded_contact_push",
                    "requires_isaac_physics_contact": True,
                    "simplified_end_effector_contact_actuator": True,
                    "dexterous_hand_policy_proven": False,
                }
            ],
        },
        "manipulation_contact_trace.json": {
            "schema_version": MANIPULATION_CONTACT_TRACE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "completed" if manipulation_contact_validated else "blocked",
            "manipulation_contact_dynamics_validated": manipulation_contact_validated,
            "contacts": [dict(contact) for contact in contact_traces[:200]]
            if manipulation_contact_validated
            else [],
            "blockers": [] if manipulation_contact_validated else ["isaac_physics_not_executed"],
        },
        "object_motion_trace.json": {
            "schema_version": OBJECT_MOTION_TRACE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "completed" if manipulation_contact_validated else "blocked",
            "object_motion_validated": manipulation_contact_validated,
            "motions": [],
            "blockers": [] if manipulation_contact_validated else ["isaac_physics_not_executed"],
        },
        "manipulation_success_evaluator_results.json": {
            "schema_version": MANIPULATION_EVAL_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "completed" if manipulation_contact_validated else "blocked",
            "manipulation_contact_dynamics_validated": manipulation_contact_validated,
            "successful_action_count": summary["successful_episode_count"]
            if manipulation_contact_validated
            else 0,
            "blockers": []
            if manipulation_contact_validated
            else ["manipulation_contact_trace_missing_runtime_evidence"],
        },
        "manipulation_truth_boundary.json": {
            "schema_version": MANIPULATION_TRUTH_BOUNDARY_SCHEMA_VERSION,
            "generated_at": generated,
            "dexterous_hand_policy_proven": False,
            "simplified_end_effector_contact_actuator_documented": True,
            "contact_realism_proven": False,
        },
        "real_wam_vla_runtime_discovery.json": wam_discovery,
        "wam_vla_observation_packet.json": {
            "schema_version": WAM_VLA_OBSERVATION_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "prepared_no_runtime_execution",
            "matrix_path": "scenario_eval_matrix.json",
            "camera_manifest_path": "camera_manifest.json",
            "observation_count": len(attempts),
        },
        "wam_vla_policy_outputs.json": {
            "schema_version": WAM_VLA_OUTPUT_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "wam_vla_runtime_proven": False,
            "outputs": [],
            "blockers": wam_discovery.get("blockers", []),
        },
        "wam_vla_action_trace.json": {
            "schema_version": WAM_VLA_ACTION_TRACE_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "actions": [],
            "blockers": wam_discovery.get("blockers", []),
        },
        "wam_vla_runtime_proof_manifest.json": {
            "schema_version": WAM_VLA_PROOF_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "wam_vla_runtime_proven": False,
            "blockers": wam_discovery.get("blockers", []),
        },
        "wam_vla_truth_boundary.json": {
            "schema_version": WAM_VLA_TRUTH_BOUNDARY_SCHEMA_VERSION,
            "generated_at": generated,
            "real_wam_vla_runtime_proven": False,
            "placeholder_wam_vla_claimed": False,
            "support_fixture_not_used_as_real_wam": True,
        },
        "scenario_eval_matrix.json": matrix,
        "camera_manifest.json": camera_manifest,
        "episode_spec_manifest.json": episode_manifest,
        "normalized_attempt_trace.json": trace,
        "failure_labels.json": failure_labels,
        "wam_evaluator_thresholds.json": thresholds,
        "wam_evaluator_trace_binding.json": trace_binding,
        "wam_evaluator_results.json": evaluator_results,
        "policy_evaluation_summary.json": summary,
        "realistic_video_manifest.json": video_manifest,
    }
    for filename, payload in artifact_payloads.items():
        write_json(job_dir / filename, payload)

    provider_bundle_manifest = _write_provider_runtime_bundle(
        job_dir=job_dir,
        job_id=resolved_job_id,
        generated_at=generated,
        ply_asset=ply_asset,
        spz_asset=spz_asset,
        labels_json=labels_json,
        structure_json=structure_json,
        occupancy_json=occupancy_json,
        occupancy_png=occupancy_png,
        allow_cloud_gpu=allow_cloud_gpu,
    )
    local_provider_command_diagnostic = _run_local_provider_command_diagnostic(
        job_dir=job_dir,
        bundle_manifest=provider_bundle_manifest,
        generated_at=generated,
    )
    gpu_provider_launch_request = _build_gpu_provider_launch_request(
        job_id=resolved_job_id,
        generated_at=generated,
        bundle_manifest=provider_bundle_manifest,
        allow_cloud_gpu=allow_cloud_gpu,
    )
    write_json(job_dir / "gpu_provider_launch_request.json", gpu_provider_launch_request)
    provider_bundle_readiness = _write_provider_bundle_readiness_manifest(
        job_dir=job_dir,
        generated_at=generated,
        bundle_manifest=provider_bundle_manifest,
        local_provider_command_diagnostic=local_provider_command_diagnostic,
        matrix=matrix,
        camera_manifest=camera_manifest,
        video_manifest=video_manifest,
        gpu_provider_launch_request=gpu_provider_launch_request,
    )
    provider_plan["provider_runtime_bundle"] = {
        "manifest_path": str(job_dir / "isaac_provider_runtime_bundle_manifest.json"),
        "bundle_path": provider_bundle_manifest.get("bundle_path"),
        "bundle_size_bytes": provider_bundle_manifest.get("bundle_size_bytes"),
        "remote_staging_required_before_runpod": True,
    }
    provider_plan["local_provider_command_diagnostic"] = {
        "path": str(job_dir / "local_provider_command_diagnostic.json"),
        "status": local_provider_command_diagnostic.get("status"),
        "provider_command_path_local_proven": local_provider_command_diagnostic.get(
            "provider_command_path_local_proven"
        ),
        "isaac_runtime_execution_proven": local_provider_command_diagnostic.get(
            "isaac_runtime_execution_proven"
        ),
    }
    provider_plan["gpu_provider_launch_request"] = {
        "path": str(job_dir / "gpu_provider_launch_request.json"),
        "status": gpu_provider_launch_request.get("status"),
        "blockers": list(gpu_provider_launch_request.get("blockers") or []),
    }
    provider_plan["provider_bundle_readiness"] = {
        "path": str(job_dir / "isaac_provider_bundle_readiness.json"),
        "status": provider_bundle_readiness.get("status"),
        "local_bundle_ready_for_remote_staging": provider_bundle_readiness.get(
            "local_bundle_ready_for_remote_staging"
        ),
        "ready_for_next_authorized_vast_bundle_attempt": provider_bundle_readiness.get(
            "ready_for_next_authorized_vast_bundle_attempt"
        ),
        "blockers": list(provider_bundle_readiness.get("blockers") or []),
    }
    combined_provider_blockers = sorted(
        {
            *(_string_list(provider_plan.get("blockers"))),
            *(_string_list(gpu_provider_launch_request.get("blockers"))),
            *(_string_list(provider_bundle_readiness.get("blockers"))),
        }
    )
    provider_plan["blockers"] = combined_provider_blockers
    provider_plan["status"] = (
        "ready_for_authorized_provider_execution"
        if not combined_provider_blockers
        else "blocked"
    )
    write_json(job_dir / "isaac_provider_plan.json", provider_plan)

    _write_jsonl(job_dir / "isaac_runtime_phase_log.jsonl", phase_rows)
    _write_jsonl(
        job_dir / "g1_locomotion_trace.jsonl",
        (
            {
                "generated_at": generated,
                "episode_id": attempt.get("episode_id"),
                "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
                "status": "completed"
                if attempt.get("status") != "blocked" and locomotion_validated
                else "blocked",
                "blockers": []
                if attempt.get("status") != "blocked" and locomotion_validated
                else ["g1_controller_runtime_unavailable"],
                "qpos_continuity_validated": bool(
                    attempt.get("status") != "blocked" and locomotion_validated
                ),
                "metrics": _mapping(attempt.get("metrics")),
                "action_count": len(attempt.get("actions") or attempt.get("action_trace") or []),
            }
            for attempt in attempts
        ),
    )
    batch_trace_package = _write_isaac_batch_trace_package(
        job_dir=job_dir,
        generated_at=generated,
        attempts=attempts,
        failure_labels=failure_labels,
        video_manifest=video_manifest,
        collision_contact_report=collision_contact_report,
        coverage=coverage,
    )
    local_validation_report = _local_validation_report(
        job_id=resolved_job_id,
        generated_at=generated,
        job_dir=job_dir,
        matrix=matrix,
        trace=trace,
        summary=summary,
        video_manifest=video_manifest,
        gpu_provider_launch_request=gpu_provider_launch_request,
        provider_bundle_readiness=provider_bundle_readiness,
    )
    write_json(job_dir / "local_validation_report.json", local_validation_report)
    artifact_paths = {
        filename: str(job_dir / filename)
        for filename in sorted(
            {
                *artifact_payloads.keys(),
                "isaac_provider_eval_manifest.json",
                "isaac_provider_runtime_bundle_manifest.json",
                "isaac_provider_runtime_bundle.zip",
                "gpu_provider_launch_request.json",
                "isaac_provider_bundle_readiness.json",
                "isaac_runtime_phase_log.jsonl",
                "g1_locomotion_trace.jsonl",
                "local_validation_report.json",
                "local_provider_command_diagnostic.json",
                "generated_site_scene.usda",
                "generated_site_scene.usd",
                "artifact_manifest.json",
                "isaac_batch_attempt_trace.jsonl",
                "isaac_batch_contact_stream.jsonl",
                "isaac_batch_planner_state.jsonl",
                "isaac_batch_control_stream.jsonl",
                "isaac_batch_metrics.json",
                "isaac_batch_failure_labels.json",
                "isaac_batch_visual_media_coverage.json",
                "isaac_batch_visual_review_ledger.json",
                "isaac_batch_artifact_checksums.json",
                "isaac_batch_trace_package_manifest.json",
                "isaac_batch_closure_manifest.json",
                "job_run_manifest.json",
            }
        )
    }
    job_manifest = {
        "schema_version": ISAAC_REALISTIC_JOB_SCHEMA_VERSION,
        "generated_at": generated,
        "job_id": resolved_job_id,
        "status": summary["status"],
        "artifact_dir": str(job_dir),
        "artifact_paths": artifact_paths,
        "phase_log_path": str(job_dir / "isaac_runtime_phase_log.jsonl"),
        "g1_locomotion_trace_path": str(job_dir / "g1_locomotion_trace.jsonl"),
        "generated_site_scene_usda": str(job_dir / "generated_site_scene.usda"),
        "generated_site_scene_usd": str(job_dir / "generated_site_scene.usd"),
        "realistic_video_manifest_path": str(job_dir / "realistic_video_manifest.json"),
        "proof_flags": summary,
    }
    write_json(job_dir / "job_run_manifest.json", job_manifest)
    batch_closure_manifest = _build_isaac_batch_closure_manifest(
        job_dir=job_dir,
        generated_at=generated,
        attempts=attempts,
        coverage=coverage,
        batch_trace_package=_mapping(batch_trace_package.get("manifest")),
        artifact_paths=artifact_paths,
        summary=summary,
        video_manifest=video_manifest,
        collision_contact_report=collision_contact_report,
    )
    write_json(job_dir / "isaac_batch_closure_manifest.json", batch_closure_manifest)
    artifact_manifest = _write_isaac_artifact_manifest(
        job_dir=job_dir,
        generated_at=generated,
        artifact_paths=artifact_paths,
        summary=summary,
        coverage=coverage,
        batch_closure_manifest=batch_closure_manifest,
    )
    simulator_execution_proven = bool(
        runtime_result.get("isaac_runtime_executed")
        or runtime_result.get("isaac_sim_execution_proven")
    )
    simulator_output = (
        Path(simulator_output_path).expanduser().resolve()
        if simulator_output_path
        else Path(os.environ["BLUEPRINT_SIMULATOR_OUTPUT"]).expanduser().resolve()
        if os.environ.get("BLUEPRINT_SIMULATOR_OUTPUT")
        else job_dir / "isaac_g1_simulator_output.json"
    )
    simulator_output_payload = {
        "schema_version": ISAAC_G1_SIMULATOR_COMMAND_OUTPUT_SCHEMA_VERSION,
        "generated_at": generated,
        "status": summary["status"],
        "simulator_backend": "isaac_sim",
        "simulator_version": summary.get("simulator_version"),
        "capture_root": str(Path(os.environ.get("BLUEPRINT_CAPTURE_ROOT", "")).resolve())
        if os.environ.get("BLUEPRINT_CAPTURE_ROOT")
        else None,
        "output_dir": str(job_dir),
        "simulator_execution_proven": simulator_execution_proven,
        "isaac_sim_execution_proven": simulator_execution_proven,
        "isaac_robot_asset_execution_proven": bool(summary.get("unitree_g1_loaded_in_isaac")),
        "unitree_g1_asset_spawned": bool(summary.get("unitree_g1_loaded_in_isaac")),
        "official_policy_execution_proven": bool(summary.get("official_policy_execution_proven")),
        "controller_grade_execution_proven": bool(summary.get("controller_grade_execution_proven")),
        "robot_policy_execution_proven": False,
        "robot_team_policy_execution_proven": bool(summary.get("official_policy_execution_proven")),
        "collision_dynamics_validated": bool(summary.get("collision_dynamics_validated")),
        "contact_dynamics_validated": bool(summary.get("collision_dynamics_validated")),
        "real_robot_pov_evidence_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "scenario_eval_matrix": matrix,
        "scenario_eval_matrix_path": matrix.get("source_scenario_eval_matrix_path"),
        "scenario_eval_run_count": matrix.get("scenario_eval_run_count"),
        "attempt_count": len(attempts),
        **coverage,
        "attempts": attempts,
        "normalized_attempt_trace": trace,
        "failure_labels": failure_labels,
        "policy_evaluation_summary": summary,
        "realistic_video_manifest": video_manifest,
        "collision_contact_report": collision_contact_report,
        "contact_summary": collision_contact_report,
        "batch_trace_package": _mapping(batch_trace_package.get("manifest")),
        "batch_closure_manifest": batch_closure_manifest,
        "artifact_manifest": artifact_manifest,
        "artifact_paths": {
            **artifact_paths,
            "batch_trace_package_manifest": str(job_dir / "isaac_batch_trace_package_manifest.json"),
            "batch_closure_manifest": str(job_dir / "isaac_batch_closure_manifest.json"),
            "artifact_manifest": str(job_dir / "artifact_manifest.json"),
        },
        "proof_boundary": {
            "mujoco_artifacts_counted_as_isaac_proof": False,
            "isaac_simulator_execution_proven": simulator_execution_proven,
            "robot_policy_execution_proven": False,
            "simulator_proof_is_not_safety_validation": True,
            "simulator_proof_is_not_deployment_approval": True,
            "simulator_proof_is_not_physical_robot_readiness": True,
            "wam_consistency_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
        },
    }
    write_json(simulator_output, simulator_output_payload)
    job_manifest["artifact_paths"] = {
        **artifact_paths,
        "isaac_batch_closure_manifest.json": str(job_dir / "isaac_batch_closure_manifest.json"),
        "artifact_manifest.json": str(job_dir / "artifact_manifest.json"),
        "isaac_g1_simulator_output.json": str(simulator_output),
    }
    job_manifest["batch_closure_manifest_path"] = str(job_dir / "isaac_batch_closure_manifest.json")
    job_manifest["artifact_manifest_path"] = str(job_dir / "artifact_manifest.json")
    job_manifest["simulator_output_path"] = str(simulator_output)
    job_manifest["proof_flags"] = summary
    write_json(job_dir / "job_run_manifest.json", job_manifest)
    return {
        "status": summary["status"],
        "job_id": resolved_job_id,
        "job_dir": str(job_dir),
        "summary": summary,
        "artifact_paths": job_manifest["artifact_paths"],
        "realistic_video_manifest_path": str(job_dir / "realistic_video_manifest.json"),
        "phase_log_path": str(job_dir / "isaac_runtime_phase_log.jsonl"),
        "simulator_output_path": str(simulator_output),
        "batch_closure_manifest_path": str(job_dir / "isaac_batch_closure_manifest.json"),
    }


def _parse_camera_ids(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [_safe_id(item) for item in value.split(",") if _safe_id(item)]


def _find_first_capture_asset(capture_root: Path, suffixes: Sequence[str]) -> Path | None:
    suffix_set = {suffix.lower() for suffix in suffixes}
    for path in sorted(capture_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffix_set:
            return path
    return None


def _write_placeholder_ply(path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "comment Blueprint placeholder scene because no capture PLY was found",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_isaac_g1_simulator_command(
    *,
    capture_root: str | Path,
    ply_asset: str | Path | None = None,
    spz_asset: str | Path | None = None,
    labels_json: str | Path | None = None,
    structure_json: str | Path | None = None,
    occupancy_json: str | Path | None = None,
    occupancy_png: str | Path | None = None,
    output_dir: str | Path | None = None,
    simulator_output_path: str | Path | None = None,
    scenario_eval_matrix_path: str | Path | None = None,
    runtime_result_path: str | Path | None = None,
    camera_ids: Sequence[str] | None = None,
    allow_cloud_gpu: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    root = Path(capture_root).expanduser().resolve()
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else root / "pipeline" / "simulation_automation" / "isaac_g1_simulator_command"
    )
    ensure_dir(resolved_output_dir)
    input_blockers: list[str] = []
    resolved_ply = Path(ply_asset).expanduser().resolve() if ply_asset else None
    if resolved_ply is None:
        resolved_ply = _find_first_capture_asset(root, [".ply"])
    if resolved_ply is None:
        resolved_ply = resolved_output_dir / "placeholder_scene_missing_capture_ply.ply"
        _write_placeholder_ply(resolved_ply)
        input_blockers.append("isaac_capture_ply_asset_missing")
    resolved_spz = Path(spz_asset).expanduser().resolve() if spz_asset else None
    if resolved_spz is None:
        resolved_spz = _find_first_capture_asset(root, [".spz"])
    if resolved_spz is None:
        resolved_spz = resolved_output_dir / "placeholder_scene_missing_capture_spz.spz"
        ensure_dir(resolved_spz.parent)
        resolved_spz.write_bytes(b"SPZ\x00placeholder_missing_capture_spz")
        input_blockers.append("isaac_capture_spz_asset_missing")
    matrix_path = (
        Path(scenario_eval_matrix_path).expanduser().resolve()
        if scenario_eval_matrix_path
        else Path(os.environ["BLUEPRINT_SCENARIO_EVAL_MATRIX"]).expanduser().resolve()
        if os.environ.get("BLUEPRINT_SCENARIO_EVAL_MATRIX")
        else None
    )
    result = run_isaac_g1_site_3dgs_realistic_eval(
        ply_asset=resolved_ply,
        spz_asset=resolved_spz,
        labels_json=labels_json,
        structure_json=structure_json,
        occupancy_json=occupancy_json,
        occupancy_png=occupancy_png,
        job_id=resolved_output_dir.name,
        job_root=resolved_output_dir.parent,
        camera_ids=camera_ids,
        allow_cloud_gpu=allow_cloud_gpu,
        scenario_eval_matrix_path=matrix_path,
        simulator_output_path=simulator_output_path,
        runtime_result_path=runtime_result_path,
        generated_at=generated_at,
    )
    simulator_output = Path(_string(result.get("simulator_output_path")))
    if simulator_output.is_file() and input_blockers:
        payload = _json_mapping(simulator_output)
        payload["input_asset_blockers"] = input_blockers
        payload["placeholder_scene_assets_used"] = True
        payload["simulator_execution_proven"] = False
        payload["isaac_sim_execution_proven"] = False
        payload["proof_boundary"] = {
            **_mapping(payload.get("proof_boundary")),
            "placeholder_scene_assets_are_not_isaac_scene_fidelity_proof": True,
        }
        write_json(simulator_output, payload)
    return {
        **result,
        "input_asset_blockers": input_blockers,
        "placeholder_scene_assets_used": bool(input_blockers),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root")
    parser.add_argument("--ply-asset")
    parser.add_argument("--spz-asset")
    parser.add_argument("--labels-json")
    parser.add_argument("--structure-json")
    parser.add_argument("--occupancy-json")
    parser.add_argument("--occupancy-png")
    parser.add_argument("--output-dir")
    parser.add_argument("--simulator-output")
    parser.add_argument("--scenario-eval-matrix")
    parser.add_argument("--runtime-result")
    parser.add_argument("--job-id")
    parser.add_argument("--job-root")
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--spawn-limit", type=int)
    parser.add_argument("--camera-ids", help="Comma-separated camera ids.")
    parser.add_argument("--allow-cloud-gpu", action="store_true")
    args = parser.parse_args(argv)
    capture_root_env = os.environ.get("BLUEPRINT_CAPTURE_ROOT")
    capture_root_mode = bool(args.capture_root or capture_root_env)
    if capture_root_mode:
        payload = run_isaac_g1_simulator_command(
            capture_root=args.capture_root or capture_root_env,
            ply_asset=args.ply_asset,
            spz_asset=args.spz_asset,
            labels_json=args.labels_json,
            structure_json=args.structure_json,
            occupancy_json=args.occupancy_json,
            occupancy_png=args.occupancy_png,
            output_dir=args.output_dir,
            simulator_output_path=args.simulator_output,
            scenario_eval_matrix_path=args.scenario_eval_matrix,
            runtime_result_path=args.runtime_result,
            camera_ids=_parse_camera_ids(args.camera_ids),
            allow_cloud_gpu=args.allow_cloud_gpu,
        )
    else:
        if not args.ply_asset or not args.spz_asset:
            parser.error("--ply-asset and --spz-asset are required without --capture-root")
        payload = run_isaac_g1_site_3dgs_realistic_eval(
            ply_asset=args.ply_asset,
            spz_asset=args.spz_asset,
            labels_json=args.labels_json,
            structure_json=args.structure_json,
            occupancy_json=args.occupancy_json,
            occupancy_png=args.occupancy_png,
            job_id=args.job_id,
            job_root=args.job_root,
            task_limit=args.task_limit,
            spawn_limit=args.spawn_limit,
            camera_ids=_parse_camera_ids(args.camera_ids),
            allow_cloud_gpu=args.allow_cloud_gpu,
            scenario_eval_matrix_path=args.scenario_eval_matrix,
            simulator_output_path=args.simulator_output,
            runtime_result_path=args.runtime_result,
        )
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "job_id": payload.get("job_id"),
                "job_dir": payload.get("job_dir"),
                "attempted_episode_count": payload.get("summary", {}).get(
                    "attempted_episode_count"
                ),
                "successful_episode_count": payload.get("summary", {}).get(
                    "successful_episode_count"
                ),
                "failed_episode_count": payload.get("summary", {}).get("failed_episode_count"),
                "blocked_episode_count": payload.get("summary", {}).get(
                    "blocked_episode_count"
                ),
                "scenario_eval_run_coverage_complete": payload.get("summary", {}).get(
                    "scenario_eval_run_coverage_complete"
                ),
                "simulator_output_path": payload.get("simulator_output_path"),
                "input_asset_blockers": payload.get("input_asset_blockers", []),
            },
            indent=2,
        )
    )
    if capture_root_mode:
        simulator_output = Path(_string(payload.get("simulator_output_path")))
        simulator_payload = _json_mapping(simulator_output) if simulator_output.is_file() else {}
        if simulator_payload.get("simulator_execution_proven") is not True:
            return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
