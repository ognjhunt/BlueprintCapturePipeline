"""Run local preflight, current package pipeline, and agent review end to end."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import logging
from pathlib import Path
from typing import Any, Mapping
from typing import List, Optional

from .agent_runtime.orchestrator import run_agent_review
from .agent_runtime.openai_phase2 import OpenAIPhase2Config
from .capture_orchestrator import (
    PipelineConfig,
    run_capture_pipeline,
)
from .common import PipelineError, read_json_any, utc_now_iso, write_json
from .evaluation_prep_stage import run_evaluation_prep_stage
from .logging_utils import log_event
from .local_capture import resolve_local_capture_context
from .materialization import materialize_capture_bundle
from .preflight_capture import build_capture_preflight_report
from .robot_eval_job_orchestrator import (
    build_robot_eval_job,
    run_robot_eval_job_request_inbox,
)
from .synthesis.cosmos_benchmark import run_cosmos_zero_shot_validation_lane


logger = logging.getLogger(__name__)

RUN_E2E_STAGE_LEDGER_FILENAME = "run_e2e_stage_ledger.json"
RUN_E2E_STAGE_LEDGER_SCHEMA_VERSION = "run_e2e_stage_ledger.v1"
RUN_E2E_INPUT_FINGERPRINT_SCHEMA_VERSION = "run_e2e_capture_input_fingerprint.v1"
RUN_E2E_STAGE_RESULT_SNAPSHOT_MAX_BYTES = 512_000
_SENSITIVE_STAGE_SNAPSHOT_KEY_MARKERS = (
    "api_key",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
_RUN_E2E_STAGE_ORDER = (
    "preflight",
    "materialization",
    "capture_pipeline",
    "agent_review",
    "evaluation_prep",
    "cosmos_validation",
    "robot_eval",
)


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_job_id(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-" for char in value
    )
    return cleaned.strip("-_") or "run-e2e-robot-eval"


def _run_e2e_stage_ledger_path(capture_root: Path) -> Path:
    return capture_root / "pipeline" / RUN_E2E_STAGE_LEDGER_FILENAME


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_input_fingerprint_source(
    *,
    capture_root: Path,
    role: str,
    path: Path,
) -> dict[str, Any]:
    relative_path = path.relative_to(capture_root) if path.is_relative_to(capture_root) else path
    exists = path.is_file()
    return {
        "role": role,
        "relative_path": str(relative_path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": _sha_file(path) if exists else None,
    }


def _capture_input_fingerprint(context: Any) -> dict[str, Any]:
    capture_root = Path(context.capture_root)
    raw_root = Path(getattr(context, "raw_root", capture_root / "raw"))
    sources = [
        _capture_input_fingerprint_source(
            capture_root=capture_root,
            role="capture_descriptor",
            path=Path(context.descriptor_path),
        ),
        _capture_input_fingerprint_source(
            capture_root=capture_root,
            role="raw_manifest",
            path=raw_root / "manifest.json",
        ),
    ]
    payload = {
        "schema_version": RUN_E2E_INPUT_FINGERPRINT_SCHEMA_VERSION,
        "sources": sources,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "fingerprint_sha256": sha256(encoded).hexdigest()}


def _new_run_e2e_stage_ledger(
    *,
    capture_root: Path,
    provider: str,
    pipeline_lane: str,
    run_evaluation_prep: bool,
    run_cosmos_validation: bool,
    robot_eval_requested: bool,
    capture_input_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    generated_at = utc_now_iso()
    return {
        "schema_version": RUN_E2E_STAGE_LEDGER_SCHEMA_VERSION,
        "status": "running",
        "started_at": generated_at,
        "updated_at": generated_at,
        "capture_root": str(capture_root),
        "provider": provider,
        "pipeline_lane": pipeline_lane,
        "capture_input_fingerprint": dict(capture_input_fingerprint),
        "requested": {
            "evaluation_prep": run_evaluation_prep,
            "cosmos_validation": run_cosmos_validation,
            "robot_eval": robot_eval_requested,
        },
        "stage_order": list(_RUN_E2E_STAGE_ORDER),
        "current_stage": None,
        "last_completed_stage": None,
        "failed_stage": None,
        "stages": {
            name: {"name": name, "status": "pending"}
            for name in _RUN_E2E_STAGE_ORDER
        },
    }


def _run_e2e_resume_requested(
    *,
    run_evaluation_prep: bool,
    run_cosmos_validation: bool,
    robot_eval_requested: bool,
) -> dict[str, bool]:
    return {
        "evaluation_prep": run_evaluation_prep,
        "cosmos_validation": run_cosmos_validation,
        "robot_eval": robot_eval_requested,
    }


def _read_run_e2e_stage_ledger(capture_root: Path) -> dict[str, Any]:
    path = _run_e2e_stage_ledger_path(capture_root)
    if not path.is_file():
        return {}
    try:
        loaded = read_json_any(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _resume_compatible_run_e2e_stage_ledger(
    *,
    capture_root: Path,
    provider: str,
    pipeline_lane: str,
    requested: Mapping[str, bool],
    capture_input_fingerprint: Mapping[str, Any],
) -> dict[str, Any] | None:
    existing = _read_run_e2e_stage_ledger(capture_root)
    if not existing:
        return None
    if existing.get("schema_version") != RUN_E2E_STAGE_LEDGER_SCHEMA_VERSION:
        return None
    if existing.get("capture_root") != str(capture_root):
        return None
    if existing.get("provider") != provider:
        return None
    if existing.get("pipeline_lane") != pipeline_lane:
        return None
    if existing.get("capture_input_fingerprint") != dict(capture_input_fingerprint):
        return None
    if existing.get("requested") != dict(requested):
        return None

    now = utc_now_iso()
    resumed = dict(existing)
    resumed["status"] = "running"
    resumed["updated_at"] = now
    resumed["current_stage"] = None
    resumed["failed_stage"] = None
    resumed["resume_completed_stages_requested"] = True
    resumed["resume_started_at"] = now
    resumed["resumed_from_status"] = existing.get("status")
    return resumed


def _write_run_e2e_stage_ledger(
    capture_root: Path,
    ledger: Mapping[str, Any],
) -> None:
    write_json(_run_e2e_stage_ledger_path(capture_root), ledger)


def _stage_result_status(value: Any) -> str | None:
    if isinstance(value, Mapping):
        status = _string(value.get("status"))
        return status or None
    return None


def _json_safe_stage_result_snapshot(value: Any) -> Any | None:
    try:
        encoded = json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return None
    if len(encoded.encode("utf-8")) > RUN_E2E_STAGE_RESULT_SNAPSHOT_MAX_BYTES:
        return None
    return _redact_stage_result_snapshot(json.loads(encoded))


def _redact_stage_result_snapshot(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if any(
                marker in key_text.lower()
                for marker in _SENSITIVE_STAGE_SNAPSHOT_KEY_MARKERS
            ):
                redacted[key_text] = "<redacted>"
            else:
                redacted[key_text] = _redact_stage_result_snapshot(child)
        return redacted
    if isinstance(value, list):
        return [_redact_stage_result_snapshot(item) for item in value]
    return value


def _completed_stage_resume_snapshot(
    ledger: Mapping[str, Any],
    *,
    stage: str,
) -> Any | None:
    stages = ledger.get("stages")
    if not isinstance(stages, Mapping):
        return None
    entry = stages.get(stage)
    if not isinstance(entry, Mapping):
        return None
    if entry.get("status") != "completed":
        return None
    if not entry.get("resume_result_snapshot_available"):
        return None
    if "result_snapshot" not in entry:
        return None
    return _redact_stage_result_snapshot(entry.get("result_snapshot"))


def _mark_run_e2e_stage(
    ledger: dict[str, Any],
    *,
    capture_root: Path,
    stage: str,
    status: str,
    detail: str | None = None,
    artifacts: Mapping[str, Any] | None = None,
    result_snapshot: Any | None = None,
    resume_used: bool = False,
    error: BaseException | None = None,
) -> None:
    now = utc_now_iso()
    stages = ledger.setdefault("stages", {})
    entry = dict(stages.get(stage) if isinstance(stages.get(stage), Mapping) else {})
    entry.setdefault("name", stage)
    entry["status"] = status
    if status == "running":
        entry.setdefault("started_at", now)
        ledger["status"] = "running"
        ledger["current_stage"] = stage
    elif status == "completed":
        entry.setdefault("started_at", now)
        if resume_used:
            entry.setdefault("completed_at", now)
        else:
            entry["completed_at"] = now
        ledger["last_completed_stage"] = stage
        ledger["current_stage"] = None
        if result_snapshot is not None:
            entry["result_snapshot"] = _redact_stage_result_snapshot(result_snapshot)
            entry["resume_result_snapshot_available"] = True
        elif "result_snapshot" not in entry:
            entry["resume_result_snapshot_available"] = False
    elif status == "skipped":
        entry["skipped_at"] = now
    elif status == "failed":
        entry.setdefault("started_at", now)
        entry["failed_at"] = now
        ledger["status"] = "failed"
        ledger["failed_stage"] = stage
        ledger["current_stage"] = None
    if detail:
        entry["detail"] = detail
    if artifacts:
        entry["artifacts"] = dict(artifacts)
    if resume_used:
        entry["resume_used"] = True
        entry["resumed_at"] = now
    if error is not None:
        entry["error_type"] = type(error).__name__
        entry["error"] = str(error)
        ledger["last_error_type"] = type(error).__name__
        ledger["last_error"] = str(error)
    stages[stage] = entry
    ledger["updated_at"] = now
    _write_run_e2e_stage_ledger(capture_root, ledger)


def _complete_run_e2e_stage_ledger(
    ledger: dict[str, Any],
    *,
    capture_root: Path,
) -> None:
    completed_at = utc_now_iso()
    ledger["status"] = "completed"
    ledger["completed_at"] = completed_at
    ledger["updated_at"] = completed_at
    ledger["current_stage"] = None
    _write_run_e2e_stage_ledger(capture_root, ledger)


def _robot_eval_job_id(
    *,
    explicit_job_id: str | None,
    request_path: str | Path,
) -> str:
    if explicit_job_id:
        return _safe_job_id(explicit_job_id)
    return _safe_job_id(Path(request_path).stem)


def _read_job_artifact(job_result: Mapping[str, Any], name: str) -> dict[str, Any]:
    job_dir = _string(job_result.get("job_dir"))
    if not job_dir:
        return {}
    path = Path(job_dir) / name
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return payload if isinstance(payload, dict) else {}


def _robot_eval_provider_runtime_summary(
    job_result: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not job_result:
        return None
    provider_request = _read_job_artifact(job_result, "gpu_provider_launch_request.json")
    cost_ledger = _read_job_artifact(job_result, "gpu_cost_control_ledger.json")
    remote_closure = _read_job_artifact(
        job_result,
        "remote_cloud_execution_closure_manifest.json",
    )
    provider_race_handoff = _read_job_artifact(
        job_result,
        "gpu_provider_race_handoff.json",
    )
    provider_race_launcher_result = _read_job_artifact(
        job_result,
        "gpu_provider_race_launcher_result.json",
    )
    prelaunch_guard = _mapping(provider_request.get("prelaunch_spend_guard"))
    provider_race = _mapping(
        prelaunch_guard.get("provider_race") or provider_request.get("provider_race")
    )
    provider_race_launcher_contract = _mapping(provider_race.get("launcher_contract"))
    launcher_ready = (
        provider_race_launcher_result.get("status") == "ready_for_live_provider_race"
    )
    runtime_wired = bool(
        provider_race.get("customer_path_provider_failover_runtime_wired")
        or provider_race_handoff.get("customer_path_provider_failover_runtime_wired")
        or launcher_ready
    )
    runtime_status = (
        provider_race_launcher_result.get("status")
        or provider_race_handoff.get("customer_path_provider_failover_runtime_status")
        or provider_race_handoff.get("status")
        or provider_race.get("customer_path_provider_failover_runtime_status")
        or _mapping(provider_race.get("runtime_readiness")).get("status")
    )
    return {
        "schema_version": "run_e2e_robot_eval_provider_runtime_summary.v1",
        "job_id": job_result.get("job_id"),
        "job_dir": job_result.get("job_dir"),
        "provider": provider_request.get("provider"),
        "gpu_provider_launch_request_path": (
            str(Path(str(job_result.get("job_dir"))) / "gpu_provider_launch_request.json")
            if job_result.get("job_dir")
            else None
        ),
        "gpu_provider_launch_request_status": provider_request.get("status"),
        "gpu_provider_launch_request_reason": provider_request.get("reason"),
        "prelaunch_spend_guard": prelaunch_guard or None,
        "provider_race": provider_race or None,
        "provider_race_required_for_customer_path": bool(
            provider_race.get("race_required_for_customer_path")
            or provider_race_handoff.get("provider_race_required_for_customer_path")
        ),
        "customer_path_provider_failover_wired": bool(
            provider_race.get("customer_path_provider_failover_wired")
            or provider_race_handoff.get("customer_path_provider_failover_wired")
            or provider_race_handoff.get("customer_path_provider_failover_handoff_wired")
        ),
        "customer_path_provider_failover_runtime_wired": runtime_wired,
        "customer_path_provider_failover_runtime_status": runtime_status,
        "provider_race_handoff_status": provider_race_handoff.get("status"),
        "provider_race_launcher_result_status": provider_race_launcher_result.get(
            "status"
        ),
        "provider_race_launcher_ready": launcher_ready,
        "provider_race_execution_proven": (
            provider_race_launcher_result.get("provider_race_execution_proven") is True
        ),
        "provider_race_launcher_available": bool(
            provider_race.get("provider_race_runtime_launcher_available")
            or provider_race_launcher_contract.get("provider_race_launcher_available")
            or provider_race_handoff.get("provider_race_runtime_launcher_available")
            or provider_race_launcher_result.get("provider_race_launcher_available")
        ),
        "provider_race_launcher_command": (
            provider_race.get("provider_race_launcher_command")
            or provider_race_launcher_contract.get("provider_race_launcher_command")
            or provider_race_handoff.get("launcher_command")
        ),
        "serial_provider_launch_blocked_unless_override": bool(
            provider_race.get("customer_path_serial_launch_blocked_unless_override")
            or provider_race_handoff.get("serial_provider_launch_default_allowed")
            is False
        ),
        "live_provider_calls_performed": bool(
            provider_request.get("live_provider_calls_performed")
            or cost_ledger.get("live_provider_calls_performed")
        ),
        "gpu_cost_control_ledger_status": cost_ledger.get("status"),
        "remote_cloud_execution_closure_status": remote_closure.get("status"),
        "remote_cloud_execution_proven": bool(
            remote_closure.get("remote_cloud_execution_proven")
        ),
        "claim_boundary": {
            "run_e2e_robot_eval_handoff_is_not_provider_execution": True,
            "provider_race_launcher_result_is_not_provider_execution": bool(
                provider_race_launcher_result
            ),
            "provider_race_execution_proven": (
                provider_race_launcher_result.get("provider_race_execution_proven")
                is True
            ),
            "live_provider_calls_performed": bool(
                provider_request.get("live_provider_calls_performed")
                or cost_ledger.get("live_provider_calls_performed")
            ),
            "remote_cloud_execution_proven": bool(
                remote_closure.get("remote_cloud_execution_proven")
            ),
        },
    }


def run_end_to_end(
    *,
    capture_root: str,
    provider: str,
    openai_phase2_config: Optional[OpenAIPhase2Config] = None,
    pipeline_lane: str = "current",
    run_evaluation_prep: bool = False,
    evaluation_prep_provider: str = "manual",
    run_cosmos_validation: bool = False,
    robot_eval_job_request: str | None = None,
    robot_eval_job_id: str | None = None,
    robot_eval_request_inbox: str | None = None,
    robot_eval_provisioner: str = "fixture_local",
    robot_eval_simulator: str = "fixture",
    robot_eval_evaluation_substrate: str | None = None,
    robot_eval_budget_usd: float | None = None,
    allow_robot_eval_gpu_provisioning: bool = False,
    allow_robot_eval_simulator_execution: bool = False,
    resume_completed_stages: bool = False,
) -> dict:
    if robot_eval_job_request and robot_eval_request_inbox:
        raise PipelineError(
            "Pass either robot_eval_job_request or robot_eval_request_inbox, not both."
        )
    log_event(
        logger,
        logging.INFO,
        "run_e2e.started",
        capture_root=capture_root,
        provider=provider,
        pipeline_lane=pipeline_lane,
        run_evaluation_prep=run_evaluation_prep,
        run_cosmos_validation=run_cosmos_validation,
    )
    context = resolve_local_capture_context(capture_root)
    capture_input_fingerprint = _capture_input_fingerprint(context)
    robot_eval_requested = bool(robot_eval_job_request or robot_eval_request_inbox)
    resume_requested = _run_e2e_resume_requested(
        run_evaluation_prep=run_evaluation_prep,
        run_cosmos_validation=run_cosmos_validation,
        robot_eval_requested=robot_eval_requested,
    )
    stage_ledger = (
        _resume_compatible_run_e2e_stage_ledger(
            capture_root=context.capture_root,
            provider=provider,
            pipeline_lane=pipeline_lane,
            requested=resume_requested,
            capture_input_fingerprint=capture_input_fingerprint,
        )
        if resume_completed_stages
        else None
    )
    if stage_ledger is None:
        stage_ledger = _new_run_e2e_stage_ledger(
            capture_root=context.capture_root,
            provider=provider,
            pipeline_lane=pipeline_lane,
            run_evaluation_prep=run_evaluation_prep,
            run_cosmos_validation=run_cosmos_validation,
            robot_eval_requested=robot_eval_requested,
            capture_input_fingerprint=capture_input_fingerprint,
        )
        if resume_completed_stages:
            stage_ledger["resume_completed_stages_requested"] = True
            stage_ledger["resume_status"] = "no_compatible_completed_stage_ledger"
    _write_run_e2e_stage_ledger(context.capture_root, stage_ledger)

    def _run_stage(
        stage: str,
        callback: Any,
        *,
        artifacts_from_result: Any = None,
    ) -> Any:
        if resume_completed_stages:
            resumed_result = _completed_stage_resume_snapshot(stage_ledger, stage=stage)
            if resumed_result is not None:
                _mark_run_e2e_stage(
                    stage_ledger,
                    capture_root=context.capture_root,
                    stage=stage,
                    status="completed",
                    detail=_stage_result_status(resumed_result),
                    result_snapshot=resumed_result,
                    resume_used=True,
                )
                stage_ledger["resume_used_count"] = (
                    int(stage_ledger.get("resume_used_count") or 0) + 1
                )
                _write_run_e2e_stage_ledger(context.capture_root, stage_ledger)
                return resumed_result
        _mark_run_e2e_stage(
            stage_ledger,
            capture_root=context.capture_root,
            stage=stage,
            status="running",
        )
        try:
            stage_result = callback()
        except Exception as exc:
            _mark_run_e2e_stage(
                stage_ledger,
                capture_root=context.capture_root,
                stage=stage,
                status="failed",
                error=exc,
            )
            raise
        artifacts = (
            artifacts_from_result(stage_result)
            if callable(artifacts_from_result)
            else None
        )
        _mark_run_e2e_stage(
            stage_ledger,
            capture_root=context.capture_root,
            stage=stage,
            status="completed",
            detail=_stage_result_status(stage_result),
            artifacts=artifacts,
            result_snapshot=_json_safe_stage_result_snapshot(stage_result),
        )
        return stage_result

    def _preflight_stage() -> dict[str, Any]:
        preflight_result = build_capture_preflight_report(context.capture_root)
        if preflight_result.get("missing_required_inputs"):
            missing_inputs = [
                str(item) for item in preflight_result["missing_required_inputs"]
            ]
            log_event(
                logger,
                logging.WARNING,
                "run_e2e.preflight_failed",
                capture_root=str(context.capture_root),
                provider=provider,
                missing_required_input_count=len(missing_inputs),
                missing_required_inputs=missing_inputs,
            )
            missing = ",".join(
                str(item) for item in preflight_result["missing_required_inputs"]
            )
            raise PipelineError(
                f"Preflight failed; missing required inputs: {missing}"
            )
        return preflight_result

    preflight = _run_stage("preflight", _preflight_stage)
    log_event(
        logger,
        logging.INFO,
        "run_e2e.preflight_completed",
        capture_root=str(context.capture_root),
        provider=provider,
        preflight_status=preflight.get("status"),
    )

    if context.raw_complete_path.is_file():
        def _materialization_stage() -> dict[str, Any]:
            log_event(
                logger,
                logging.INFO,
                "run_e2e.materialization_started",
                capture_root=str(context.capture_root),
                raw_prefix_uri=context.raw_prefix_uri,
            )
            materialization_result = materialize_capture_bundle(
                bucket=context.bucket,
                scene_id=context.scene_id,
                capture_id=context.capture_id,
                gcs_root=context.storage_root,
                raw_prefix_uri=context.raw_prefix_uri,
            )
            log_event(
                logger,
                logging.INFO,
                "run_e2e.materialization_completed",
                capture_root=str(context.capture_root),
                raw_prefix_uri=context.raw_prefix_uri,
            )
            return (
                dict(materialization_result)
                if isinstance(materialization_result, Mapping)
                else {"status": "completed"}
            )

        _run_stage(
            "materialization",
            _materialization_stage,
            artifacts_from_result=lambda _result: {
                "raw_prefix_uri": context.raw_prefix_uri,
                "descriptor_uri": context.descriptor_uri,
            },
        )
    elif not context.descriptor_path.is_file():
        def _missing_descriptor_stage() -> None:
            log_event(
                logger,
                logging.WARNING,
                "run_e2e.descriptor_missing",
                capture_root=str(context.capture_root),
                raw_complete_path=str(context.raw_complete_path),
                descriptor_path=str(context.descriptor_path),
            )
            raise PipelineError(
                "Descriptor is missing and raw/capture_upload_complete.json was not found."
            )

        _run_stage("materialization", _missing_descriptor_stage)
    else:
        _mark_run_e2e_stage(
            stage_ledger,
            capture_root=context.capture_root,
            stage="materialization",
            status="skipped",
            detail="descriptor_already_present",
        )

    pipeline = _run_stage(
        "capture_pipeline",
        lambda: run_capture_pipeline(
            descriptor_gcs_uri=context.descriptor_uri,
            lane=pipeline_lane,
            config=PipelineConfig(gcs_root=context.storage_root),
        ),
        artifacts_from_result=lambda value: {
            "descriptor_uri": context.descriptor_uri,
            "lanes": value.get("lanes") if isinstance(value, Mapping) else None,
        },
    )
    review = _run_stage(
        "agent_review",
        lambda: run_agent_review(
            capture_root=context.capture_root,
            provider_name=provider,
            mode="qualification",
            openai_phase2_config=openai_phase2_config,
        ),
        artifacts_from_result=lambda value: {
            "final_memo_path": (
                value.get("final_memo_path") if isinstance(value, Mapping) else None
            ),
            "final_bundle_path": (
                value.get("final_bundle_path") if isinstance(value, Mapping) else None
            ),
        },
    )
    evaluation_prep_result = (
        _run_stage(
            "evaluation_prep",
            lambda: run_evaluation_prep_stage(
                capture_root=context.capture_root,
                provider_name=evaluation_prep_provider,
            ),
            artifacts_from_result=lambda value: {
                "manifest_path": (
                    value.get("manifest_path") if isinstance(value, Mapping) else None
                )
            },
        )
        if run_evaluation_prep
        else None
    )
    if not run_evaluation_prep:
        _mark_run_e2e_stage(
            stage_ledger,
            capture_root=context.capture_root,
            stage="evaluation_prep",
            status="skipped",
            detail="not_requested",
        )
    cosmos_validation = (
        _run_stage(
            "cosmos_validation",
            lambda: run_cosmos_zero_shot_validation_lane(
                capture_root=context.capture_root,
                descriptor_gcs_uri=context.descriptor_uri,
                cfg=PipelineConfig(gcs_root=context.storage_root),
            ),
        )
        if run_cosmos_validation
        else None
    )
    if not run_cosmos_validation:
        _mark_run_e2e_stage(
            stage_ledger,
            capture_root=context.capture_root,
            stage="cosmos_validation",
            status="skipped",
            detail="not_requested",
        )
    result = {
        "schema_version": "v1",
        "capture_root": str(context.capture_root),
        "provider": provider,
        "preflight_status": preflight.get("status"),
        "pipeline_status": pipeline.get("status"),
        "pipeline_lanes": pipeline.get("lanes"),
        "pipeline_summary": review.get("artifacts", {}).get("readiness_report"),
        "final_memo_path": review.get("final_memo_path"),
        "final_bundle_path": review.get("final_bundle_path"),
        "evaluation_prep": evaluation_prep_result,
        "webapp_sync_result": (
            evaluation_prep_result.get("webapp_sync_result")
            if isinstance(evaluation_prep_result, dict)
            else None
        ),
        "site_package_manifest": (
            evaluation_prep_result.get("site_package_manifest")
            if isinstance(evaluation_prep_result, dict)
            else None
        ),
        "hosted_review_readiness": (
            evaluation_prep_result.get("hosted_review_readiness")
            if isinstance(evaluation_prep_result, dict)
            else None
        ),
        "proof_pack_manifest": (
            evaluation_prep_result.get("proof_pack_manifest")
            if isinstance(evaluation_prep_result, dict)
            else None
        ),
        "proof_path_status": (
            evaluation_prep_result.get("proof_path_status")
            if isinstance(evaluation_prep_result, dict)
            else None
        ),
        "cosmos_validation": cosmos_validation,
        "run_e2e_stage_ledger_path": str(
            _run_e2e_stage_ledger_path(context.capture_root)
        ),
    }
    robot_eval_job = None
    robot_eval_inbox = None
    robot_eval_provider_runtime = None
    if robot_eval_job_request:
        def _robot_eval_job_stage() -> dict[str, Any]:
            return build_robot_eval_job(
                capture_root=context.capture_root,
                job_request=robot_eval_job_request,
                job_id=_robot_eval_job_id(
                    explicit_job_id=robot_eval_job_id,
                    request_path=robot_eval_job_request,
                ),
                provisioner=robot_eval_provisioner,
                simulator=robot_eval_simulator,
                evaluation_substrate=robot_eval_evaluation_substrate,
                allow_gpu_provisioning=allow_robot_eval_gpu_provisioning,
                allow_simulator_execution=allow_robot_eval_simulator_execution,
                budget_usd=robot_eval_budget_usd,
            )

        robot_eval_job = _run_stage(
            "robot_eval",
            _robot_eval_job_stage,
            artifacts_from_result=lambda value: {
                "mode": "job_request",
                "job_id": value.get("job_id") if isinstance(value, Mapping) else None,
                "job_dir": value.get("job_dir") if isinstance(value, Mapping) else None,
                "manifest_path": (
                    value.get("manifest_path") if isinstance(value, Mapping) else None
                ),
            },
        )
        robot_eval_provider_runtime = _robot_eval_provider_runtime_summary(
            robot_eval_job
        )
    elif robot_eval_request_inbox:
        def _robot_eval_inbox_stage() -> dict[str, Any]:
            return run_robot_eval_job_request_inbox(
                capture_root=context.capture_root,
                inbox_dir=robot_eval_request_inbox,
                provisioner=robot_eval_provisioner,
                simulator=robot_eval_simulator,
                evaluation_substrate=robot_eval_evaluation_substrate,
                allow_gpu_provisioning=allow_robot_eval_gpu_provisioning,
                allow_simulator_execution=allow_robot_eval_simulator_execution,
                budget_usd=robot_eval_budget_usd,
            )

        robot_eval_inbox = _run_stage(
            "robot_eval",
            _robot_eval_inbox_stage,
            artifacts_from_result=lambda value: {
                "mode": "request_inbox",
                "status": (
                    value.get("status") if isinstance(value, Mapping) else None
                ),
                "inbox_dir": robot_eval_request_inbox,
            },
        )
    else:
        _mark_run_e2e_stage(
            stage_ledger,
            capture_root=context.capture_root,
            stage="robot_eval",
            status="skipped",
            detail="not_requested",
        )
    result.update(
        {
            "robot_eval_job": robot_eval_job,
            "robot_eval_request_inbox": robot_eval_inbox,
            "robot_eval_provider_runtime": robot_eval_provider_runtime,
        }
    )
    _complete_run_e2e_stage_ledger(
        stage_ledger,
        capture_root=context.capture_root,
    )
    log_event(
        logger,
        logging.INFO,
        "run_e2e.completed",
        capture_root=str(context.capture_root),
        provider=provider,
        preflight_status=result.get("preflight_status"),
        pipeline_status=result.get("pipeline_status"),
        pipeline_lanes=result.get("pipeline_lanes"),
        evaluation_prep_enabled=run_evaluation_prep,
        cosmos_validation_enabled=run_cosmos_validation,
        robot_eval_job_requested=bool(robot_eval_job_request),
        robot_eval_request_inbox_requested=bool(robot_eval_request_inbox),
    )
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a local capture through the current capture-to-package review path"
    )
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument(
        "--provider",
        required=True,
        choices=("local", "claude", "openai"),
        help=(
            "Agent-review provider. Use local for deterministic no-LLM contract "
            "runs; claude/openai may use configured external review providers."
        ),
    )
    parser.add_argument(
        "--pipeline-lane",
        default="current",
        choices=(
            "current",
            "qualification",
            "evaluation_prep",
            "simulation_automation",
            "scene_memory",
            "retrieval_index",
            "frame_alignment",
            "synthesis_coverage_validation",
            "cosmos_single_capture_smoke",
            "all",
        ),
    )
    parser.add_argument("--openai-phase2-mode", choices=("disabled", "codex_cli"))
    parser.add_argument("--openai-phase2-model")
    parser.add_argument("--openai-phase2-codex-bin")
    parser.add_argument("--openai-phase2-timeout-seconds", type=int)
    parser.add_argument("--openai-phase2-reasoning-effort")
    evaluation_prep_group = parser.add_mutually_exclusive_group()
    evaluation_prep_group.add_argument(
        "--run-evaluation-prep",
        dest="run_evaluation_prep",
        action="store_true",
        default=True,
        help=(
            "Run evaluation prep and WebApp sync handoff. This is the default for "
            "the operator CLI; kept for explicitness and backward-compatible scripts."
        ),
    )
    evaluation_prep_group.add_argument(
        "--skip-evaluation-prep",
        dest="run_evaluation_prep",
        action="store_false",
        help=(
            "Developer-only escape hatch: stop after agent review and mark "
            "evaluation_prep skipped in the run_e2e stage ledger."
        ),
    )
    parser.add_argument("--evaluation-prep-provider", default="manual")
    parser.add_argument(
        "--run-cosmos-validation",
        action="store_true",
        help="Legacy optional Cosmos validation path; not part of the current default pipeline.",
    )
    parser.add_argument(
        "--resume-completed-stages",
        action="store_true",
        help=(
            "Reuse compatible completed run_e2e stage snapshots when retrying a "
            "previous local run."
        ),
    )
    parser.add_argument(
        "--robot-eval-job-request",
        help="Optional robot_eval_job_request JSON to hand into the Task Evaluation Run job path.",
    )
    parser.add_argument("--robot-eval-job-id")
    parser.add_argument(
        "--robot-eval-request-inbox",
        help="Optional directory of robot eval job requests to process fail-closed.",
    )
    parser.add_argument("--robot-eval-provisioner", default="fixture_local")
    parser.add_argument("--robot-eval-simulator", default="fixture")
    parser.add_argument("--robot-eval-evaluation-substrate")
    parser.add_argument("--robot-eval-budget-usd", type=float)
    parser.add_argument(
        "--allow-robot-eval-gpu-provisioning",
        action="store_true",
        help="Forward the explicit GPU provisioning gate to the robot-eval job builder.",
    )
    parser.add_argument(
        "--allow-robot-eval-simulator-execution",
        action="store_true",
        help="Forward the explicit simulator execution gate to the robot-eval job builder.",
    )
    args = parser.parse_args(argv)

    openai_phase2_config = None
    if any(
        [
            args.openai_phase2_mode,
            args.openai_phase2_model,
            args.openai_phase2_codex_bin,
            args.openai_phase2_timeout_seconds,
            args.openai_phase2_reasoning_effort,
        ]
    ):
        env_default = OpenAIPhase2Config.from_env()
        openai_phase2_config = OpenAIPhase2Config(
            mode=args.openai_phase2_mode or env_default.mode,
            model=args.openai_phase2_model or env_default.model,
            codex_bin=args.openai_phase2_codex_bin or env_default.codex_bin,
            timeout_seconds=int(args.openai_phase2_timeout_seconds or env_default.timeout_seconds),
            reasoning_effort=args.openai_phase2_reasoning_effort or env_default.reasoning_effort,
        )

    try:
        result = run_end_to_end(
            capture_root=args.capture_root,
            provider=args.provider,
            openai_phase2_config=openai_phase2_config,
            pipeline_lane=args.pipeline_lane,
            run_evaluation_prep=bool(args.run_evaluation_prep),
            evaluation_prep_provider=args.evaluation_prep_provider,
            run_cosmos_validation=bool(args.run_cosmos_validation),
            robot_eval_job_request=args.robot_eval_job_request,
            robot_eval_job_id=args.robot_eval_job_id,
            robot_eval_request_inbox=args.robot_eval_request_inbox,
            robot_eval_provisioner=args.robot_eval_provisioner,
            robot_eval_simulator=args.robot_eval_simulator,
            robot_eval_evaluation_substrate=args.robot_eval_evaluation_substrate,
            robot_eval_budget_usd=args.robot_eval_budget_usd,
            allow_robot_eval_gpu_provisioning=bool(
                args.allow_robot_eval_gpu_provisioning
            ),
            allow_robot_eval_simulator_execution=bool(
                args.allow_robot_eval_simulator_execution
            ),
            resume_completed_stages=bool(args.resume_completed_stages),
        )
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "run_e2e.failed",
            capture_root=args.capture_root,
            provider=args.provider,
            reason=str(exc),
        )
        print(f"[run-e2e] FAILED: {exc}")
        return 1

    print(f"[run-e2e] preflight_status={result['preflight_status']}")
    print(f"[run-e2e] pipeline_status={result['pipeline_status']}")
    print(f"[run-e2e] pipeline_lanes={result.get('pipeline_lanes')}")
    print(f"[run-e2e] final_memo={result['final_memo_path']}")
    print(f"[run-e2e] final_bundle={result['final_bundle_path']}")
    if result.get("evaluation_prep"):
        print(f"[run-e2e] evaluation_prep={result['evaluation_prep']['manifest_path']}")
    if result.get("cosmos_validation"):
        print(f"[run-e2e] cosmos_validation={result['cosmos_validation']['status']}")
    if result.get("robot_eval_job"):
        print(f"[run-e2e] robot_eval_job={result['robot_eval_job']['manifest_path']}")
    if result.get("robot_eval_request_inbox"):
        print(
            "[run-e2e] robot_eval_request_inbox="
            f"{result['robot_eval_request_inbox']['status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
