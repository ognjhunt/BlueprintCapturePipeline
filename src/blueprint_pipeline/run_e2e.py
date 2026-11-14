"""Run a website capture's preflight, materialization and site preparation end to end."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from typing import List, Optional

from .handoff_job_state import required_stage_result_blocker
from .core.handoff_lease import handoff_lease_checkpoint, guarded_handoff_write_json as write_json
from .capture_orchestrator import (
    PipelineConfig,
    run_capture_pipeline,
)
from .core.common import PipelineError, read_json_any, utc_now_iso
from .core.lane_resume import (
    CAPTURE_INPUT_FINGERPRINT_SCHEMA_VERSION,
    capture_input_fingerprint,
)
from .core.logging_utils import log_event
from .local_capture import resolve_local_capture_context
from .materialization import materialize_capture_bundle
from .core.pipeline_settings import PipelineSettings
from .preflight_capture import build_capture_preflight_report
from .website_robot_fit_hold import preparation_binding_digest
from .website_held_handoff_replay import preparation_binding as held_replay_binding
from .website_task_target_choice import correction_binding_digest
from .core.stage_outcome import stage_ledger_outcome_kind


logger = logging.getLogger(__name__)

RUN_E2E_STAGE_LEDGER_FILENAME = "run_e2e_stage_ledger.json"
RUN_E2E_STAGE_LEDGER_SCHEMA_VERSION = "run_e2e_stage_ledger.v1"
RUN_E2E_RUN_SUMMARY_FILENAME = "run_summary.json"
RUN_E2E_RUN_SUMMARY_SCHEMA_VERSION = "pipeline_run_summary.v1"
RUN_E2E_INPUT_FINGERPRINT_SCHEMA_VERSION = CAPTURE_INPUT_FINGERPRINT_SCHEMA_VERSION
RUN_E2E_STAGE_RESULT_SNAPSHOT_MAX_BYTES = 512_000
RUN_E2E_PIPELINE_LANE = "qualification"
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
    "task_evaluation_supervisor",
)
# The ``requested`` record every website ledger on the host carries; a retry resumes
# only a ledger whose record matches, so it stays byte-identical.
_RUN_E2E_REQUESTED = {
    "task_evaluation_supervisor": True,
    "agent_review": False,
    "evaluation_prep": False,
    "support_validation": False,
    "robot_eval": False,
}
_WEBSITE_NATIVE_HANDOFF = {
    "status": "skipped",
    "reason": "website_scene_preparation_uses_native_handoff",
    "native_execution_complete": False,
    "proof_state_mutated_by_agent": False,
}


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _completed_website_preparation_only(value: Any) -> bool:
    """Recognize the admitted website preparation receipt, never a full run.

    The qualification producer emits this receipt only after its current source,
    grant/privacy checks and native handoff/explicit preparation hold succeed.
    Generic scientific-envelope clarification is not another preparation gate.
    """
    if not isinstance(value, Mapping) or value.get("status") != "completed":
        return False
    if value.get("lanes") != ["qualification"]:
        return False
    rows = value.get("results")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        return False
    row = rows[0]
    preparation = row.get("website_scene_preparation")
    return (
        row.get("lane") == "qualification"
        and row.get("status") == "completed"
        and row.get("native_execution_complete") is False
        and isinstance(preparation, Mapping)
        and bool(preparation)
    )


def _run_e2e_stage_ledger_path(capture_root: Path) -> Path:
    return capture_root / "pipeline" / RUN_E2E_STAGE_LEDGER_FILENAME


def _run_e2e_summary_path(capture_root: Path) -> Path:
    return capture_root / "pipeline" / RUN_E2E_RUN_SUMMARY_FILENAME


def _elapsed_seconds(started_at: Any, ended_at: Any) -> float | None:
    if not isinstance(started_at, str) or not isinstance(ended_at, str):
        return None
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        ended = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, round((ended - started).total_seconds(), 6))


def _run_summary_from_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    stage_rows: list[dict[str, Any]] = []
    stages = ledger.get("stages")
    stages = stages if isinstance(stages, Mapping) else {}
    for stage_name in ledger.get("stage_order") or []:
        entry = stages.get(stage_name)
        entry = entry if isinstance(entry, Mapping) else {}
        stage_rows.append(
            {
                "stage": stage_name,
                "status": entry.get("status") or "pending",
                "outcome_kind": entry.get("outcome_kind"),
                "started_at": entry.get("started_at"),
                "ended_at": (
                    entry.get("completed_at") or entry.get("failed_at") or entry.get("skipped_at")
                ),
                "duration_seconds": entry.get("duration_seconds"),
                "resume_used": entry.get("resume_used") is True,
            }
        )
    return {
        "schema_version": RUN_E2E_RUN_SUMMARY_SCHEMA_VERSION,
        "status": ledger.get("status"),
        "capture_root": ledger.get("capture_root"),
        "provider": ledger.get("provider"),
        "pipeline_lane": ledger.get("pipeline_lane"),
        "started_at": ledger.get("started_at"),
        "completed_at": ledger.get("completed_at"),
        "updated_at": ledger.get("updated_at"),
        "failed_stage": ledger.get("failed_stage"),
        "stage_timings": stage_rows,
        "spend": dict(ledger.get("spend") or {}),
        "claim_boundary": {
            "requested_budget_is_not_actual_spend": True,
            "live_provider_calls_are_explicit": True,
            "missing_actual_gpu_seconds_are_not_zero": True,
        },
    }


def _capture_input_fingerprint(context: Any) -> dict[str, Any]:
    capture_root = Path(context.capture_root)
    return capture_input_fingerprint(
        capture_root=capture_root,
        descriptor_path=Path(context.descriptor_path),
        raw_root=Path(getattr(context, "raw_root", capture_root / "raw")),
    )


def _new_run_e2e_stage_ledger(
    *,
    capture_root: Path,
    provider: str,
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
        "pipeline_lane": RUN_E2E_PIPELINE_LANE,
        "capture_input_fingerprint": dict(capture_input_fingerprint),
        "requested": dict(_RUN_E2E_REQUESTED),
        "stage_order": list(_RUN_E2E_STAGE_ORDER),
        "current_stage": None,
        "last_completed_stage": None,
        "failed_stage": None,
        "stages": {name: {"name": name, "status": "pending"} for name in _RUN_E2E_STAGE_ORDER},
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
    handoff_lease_checkpoint("run_e2e_stage_publication")
    write_json(_run_e2e_stage_ledger_path(capture_root), ledger)
    write_json(_run_e2e_summary_path(capture_root), _run_summary_from_ledger(ledger))


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
            if any(marker in key_text.lower() for marker in _SENSITIVE_STAGE_SNAPSHOT_KEY_MARKERS):
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
    input_binding_digest: str | None = None,
) -> Any | None:
    stages = ledger.get("stages")
    if not isinstance(stages, Mapping):
        return None
    entry = stages.get(stage)
    if not isinstance(entry, Mapping):
        return None
    if input_binding_digest is not None and entry.get("input_binding_digest") != (
        input_binding_digest
    ):
        return None
    if entry.get("status") != "completed":
        return None
    if not entry.get("resume_result_snapshot_available"):
        return None
    if "result_snapshot" not in entry:
        return None
    snapshot = _redact_stage_result_snapshot(entry.get("result_snapshot"))
    if required_stage_result_blocker(stage, snapshot):
        return None
    return snapshot


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
    input_binding_digest: str | None = None,
    error: BaseException | None = None,
) -> None:
    now = utc_now_iso()
    stages = ledger.setdefault("stages", {})
    entry = dict(stages.get(stage) if isinstance(stages.get(stage), Mapping) else {})
    if input_binding_digest is not None and entry.get("input_binding_digest") != (
        input_binding_digest
    ):
        entry = {}
    entry.setdefault("name", stage)
    entry["status"] = status
    if input_binding_digest is not None:
        entry["input_binding_digest"] = input_binding_digest
    if status != "running":
        entry["outcome_kind"] = stage_ledger_outcome_kind(
            status=status,
            detail=detail,
        ).value
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
        entry["duration_seconds"] = _elapsed_seconds(
            entry.get("started_at"),
            entry.get("completed_at"),
        )
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
        entry["duration_seconds"] = _elapsed_seconds(
            entry.get("started_at"),
            entry.get("failed_at"),
        )
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
    else:
        entry.pop("resume_used", None)
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


def run_end_to_end(
    *,
    capture_root: str,
    provider: str,
    resume_completed_stages: bool = False,
) -> dict:
    log_event(
        logger,
        logging.INFO,
        "run_e2e.started",
        capture_root=capture_root,
        provider=provider,
        pipeline_lane=RUN_E2E_PIPELINE_LANE,
    )
    context = resolve_local_capture_context(capture_root)
    capture_input_fingerprint = _capture_input_fingerprint(context)
    stage_ledger = (
        _resume_compatible_run_e2e_stage_ledger(
            capture_root=context.capture_root,
            provider=provider,
            pipeline_lane=RUN_E2E_PIPELINE_LANE,
            requested=_RUN_E2E_REQUESTED,
            capture_input_fingerprint=capture_input_fingerprint,
        )
        if resume_completed_stages
        else None
    )
    if stage_ledger is None:
        stage_ledger = _new_run_e2e_stage_ledger(
            capture_root=context.capture_root,
            provider=provider,
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
        input_binding_digest: str | None = None,
    ) -> Any:
        handoff_lease_checkpoint(f"run_e2e_before:{stage}")
        # Handoff staging can replace descriptor/QA files with the original cloud
        # records on every delivery. Rebuild these cheap local projections from
        # verified raw inputs; a saved result alone cannot restore their files.
        if resume_completed_stages and stage != "materialization":
            resumed_result = _completed_stage_resume_snapshot(
                stage_ledger,
                stage=stage,
                input_binding_digest=input_binding_digest,
            )
            if stage == "capture_pipeline" and isinstance(resumed_result, Mapping):
                if (resumed_result.get("artifact_purpose") != "evaluation_preparation"
                        or resumed_result.get("robot_evaluation_performed") is not False):
                    resumed_result = None
            if resumed_result is not None:
                _mark_run_e2e_stage(
                    stage_ledger,
                    capture_root=context.capture_root,
                    stage=stage,
                    status="completed",
                    detail=_stage_result_status(resumed_result),
                    result_snapshot=resumed_result,
                    resume_used=True,
                    input_binding_digest=input_binding_digest,
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
            input_binding_digest=input_binding_digest,
        )
        try:
            stage_result = callback()
            handoff_lease_checkpoint(f"run_e2e_after:{stage}")
            blocker = required_stage_result_blocker(stage, stage_result)
            if blocker:
                raise PipelineError(blocker)
        except Exception as exc:
            _mark_run_e2e_stage(
                stage_ledger,
                capture_root=context.capture_root,
                stage=stage,
                status="failed",
                input_binding_digest=input_binding_digest,
                error=exc,
            )
            raise
        artifacts = artifacts_from_result(stage_result) if callable(artifacts_from_result) else None
        _mark_run_e2e_stage(
            stage_ledger,
            capture_root=context.capture_root,
            stage=stage,
            status="completed",
            detail=_stage_result_status(stage_result),
            artifacts=artifacts,
            result_snapshot=_json_safe_stage_result_snapshot(stage_result),
            input_binding_digest=input_binding_digest,
        )
        return stage_result

    def _preflight_stage() -> dict[str, Any]:
        preflight_result = build_capture_preflight_report(context.capture_root)
        if preflight_result.get("missing_required_inputs"):
            missing_inputs = [str(item) for item in preflight_result["missing_required_inputs"]]
            log_event(
                logger,
                logging.WARNING,
                "run_e2e.preflight_failed",
                capture_root=str(context.capture_root),
                provider=provider,
                missing_required_input_count=len(missing_inputs),
                missing_required_inputs=missing_inputs,
            )
            missing = ",".join(str(item) for item in preflight_result["missing_required_inputs"])
            raise PipelineError(f"Preflight failed; missing required inputs: {missing}")
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

    # A site's correction of the object the robot moves reruns preparation
    # from its retained stage outputs; a confirmation of the guess does not.
    # So does a change of the catalog or team registry a robot fit hold was
    # decided against (website_robot_fit_hold); nothing else changes it.
    # And a held job's replay once the fact it waited on arrived (website_held_handoff_replay).
    # The lane ledger keys on the same binding, so its marker never skips a rerun this stage makes.
    capture_binding = held_replay_binding(Path(context.capture_root) / "pipeline", preparation_binding_digest(
        Path(context.capture_root) / "pipeline",
        correction_binding_digest(Path(context.capture_root) / "pipeline")))
    pipeline = _run_stage(
        "capture_pipeline",
        lambda: run_capture_pipeline(
            descriptor_gcs_uri=context.descriptor_uri,
            lane=RUN_E2E_PIPELINE_LANE,
            config=PipelineConfig(gcs_root=context.storage_root),
            **({"input_binding_digest": capture_binding} if capture_binding is not None else {}),
        ),
        artifacts_from_result=lambda value: {
            "descriptor_uri": context.descriptor_uri,
            "lanes": value.get("lanes") if isinstance(value, Mapping) else None,
        },
        input_binding_digest=capture_binding,
    )
    # Website preparation's native handoff owns execution admission; nothing else runs here.
    if not _completed_website_preparation_only(pipeline):
        error = PipelineError("required_stage_not_complete:capture_pipeline:website_scene_preparation_missing")
        _mark_run_e2e_stage(
            stage_ledger, capture_root=context.capture_root,
            stage="task_evaluation_supervisor", status="failed", error=error,
        )
        raise error
    supervisor = dict(_WEBSITE_NATIVE_HANDOFF)
    _mark_run_e2e_stage(
        stage_ledger, capture_root=context.capture_root,
        stage="task_evaluation_supervisor", status="skipped",
        detail=supervisor["reason"], result_snapshot=supervisor,
    )
    result = {
        "schema_version": "v1",
        "capture_root": str(context.capture_root),
        "provider": provider,
        "preflight_status": preflight.get("status"),
        "pipeline_status": pipeline.get("status"),
        "pipeline_lanes": pipeline.get("lanes"),
        "task_evaluation_supervisor": supervisor,
        "run_e2e_stage_ledger_path": str(_run_e2e_stage_ledger_path(context.capture_root)),
    }
    _complete_run_e2e_stage_ledger(
        stage_ledger,
        capture_root=context.capture_root,
    )
    result["run_summary_path"] = str(_run_e2e_summary_path(context.capture_root))
    log_event(
        logger,
        logging.INFO,
        "run_e2e.completed",
        capture_root=str(context.capture_root),
        provider=provider,
        preflight_status=result.get("preflight_status"),
        pipeline_status=result.get("pipeline_status"),
        pipeline_lanes=result.get("pipeline_lanes"),
    )
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a local website capture through preflight, materialization and site preparation"
    )
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument(
        "--provider",
        required=True,
        choices=("local", "claude", "openai"),
        help="Provider name recorded in the run_e2e stage ledger.",
    )
    parser.add_argument(
        "--resume-completed-stages",
        action="store_true",
        help=(
            "Reuse compatible completed run_e2e stage snapshots when retrying a previous local run."
        ),
    )
    args = parser.parse_args(argv)

    try:
        PipelineSettings.from_env()
    except ValueError as exc:
        log_event(
            logger,
            logging.ERROR,
            "run_e2e.settings_invalid",
            capture_root=args.capture_root,
            provider=args.provider,
            reason=str(exc),
        )
        print(f"[run-e2e] FAILED: {exc}")
        return 1

    try:
        result = run_end_to_end(
            capture_root=args.capture_root,
            provider=args.provider,
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
    supervisor_result = result.get("task_evaluation_supervisor") or {}
    print(f"[run-e2e] task_evaluation_supervisor={supervisor_result.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
