"""Pull BlueprintCapture bridge handoffs from Pub/Sub and run the pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from google.cloud import storage

from .common import PipelineError, utc_now_iso
from .run_e2e import run_end_to_end

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HandoffMessage:
    bucket: str
    scene_id: str
    capture_id: str
    raw_prefix_uri: str
    pipeline_handoff_uri: str | None
    robot_eval_job_request_uri: str | None = None
    robot_eval_request_inbox_uri: str | None = None
    robot_eval_job_id: str | None = None
    robot_eval_provisioner: str | None = None
    robot_eval_simulator: str | None = None
    robot_eval_evaluation_substrate: str | None = None
    robot_eval_budget_usd: float | None = None

    @property
    def capture_prefix(self) -> str:
        return f"scenes/{self.scene_id}/captures/{self.capture_id}"


def parse_handoff_payload(payload: bytes | str | Mapping[str, Any]) -> HandoffMessage:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise PipelineError(f"Pub/Sub handoff payload is not valid JSON: {exc}") from exc
    else:
        data = dict(payload)

    bucket = _required_string(data, "bucket")
    scene_id = _required_string(data, "scene_id")
    capture_id = _required_string(data, "capture_id")
    raw_prefix_uri = _required_string(data, "raw_prefix_uri")
    if raw_prefix_uri != f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/raw":
        raise PipelineError(
            "Pub/Sub handoff raw_prefix_uri does not match bucket/scene/capture identity: "
            f"{raw_prefix_uri}"
        )

    pipeline_handoff_uri = data.get("pipeline_handoff_uri")
    if pipeline_handoff_uri is not None and not isinstance(pipeline_handoff_uri, str):
        raise PipelineError("Pub/Sub handoff pipeline_handoff_uri must be a string when present.")

    robot_eval_job_request_uri = _optional_string(
        data,
        "robot_eval_job_request_uri",
        "robot_eval_job_request_path",
    )
    robot_eval_request_inbox_uri = _optional_string(
        data,
        "robot_eval_request_inbox_uri",
        "robot_eval_request_inbox_path",
    )
    robot_eval_budget_usd = _optional_number(data, "robot_eval_budget_usd")

    return HandoffMessage(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        raw_prefix_uri=raw_prefix_uri,
        pipeline_handoff_uri=pipeline_handoff_uri,
        robot_eval_job_request_uri=robot_eval_job_request_uri,
        robot_eval_request_inbox_uri=robot_eval_request_inbox_uri,
        robot_eval_job_id=_optional_string(data, "robot_eval_job_id"),
        robot_eval_provisioner=_optional_string(data, "robot_eval_provisioner"),
        robot_eval_simulator=_optional_string(data, "robot_eval_simulator"),
        robot_eval_evaluation_substrate=_optional_string(
            data,
            "robot_eval_evaluation_substrate",
        ),
        robot_eval_budget_usd=robot_eval_budget_usd,
    )


def stage_handoff_capture(
    handoff: HandoffMessage,
    *,
    storage_root: Path,
    storage_client: storage.Client | None = None,
) -> Path:
    client = storage_client or storage.Client()
    capture_root = storage_root / handoff.bucket / handoff.capture_prefix
    capture_root.mkdir(parents=True, exist_ok=True)

    blobs = list(client.list_blobs(handoff.bucket, prefix=f"{handoff.capture_prefix}/"))
    if not blobs:
        raise PipelineError(f"No objects found for handoff prefix: {handoff.capture_prefix}/")

    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        destination = storage_root / handoff.bucket / blob.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination))

    if not (capture_root / "raw" / "capture_upload_complete.json").is_file():
        raise PipelineError(
            "Staged handoff capture is missing raw/capture_upload_complete.json; "
            f"capture_root={capture_root}"
        )
    if not (capture_root / "pipeline_handoff.json").is_file():
        # Real iOS bundles never upload pipeline_handoff.json (XR-03); synthesize it from the
        # provenance already carried by raw/manifest.json + raw/capture_context.json so the
        # capture_job_id / site_submission_id / buyer_request_id data contract stays intact.
        _synthesize_pipeline_handoff(handoff, capture_root=capture_root)
    return capture_root


def _read_optional_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _first_non_empty(*sources: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _first_bool(*sources: Mapping[str, Any], keys: Sequence[str]) -> bool | None:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, bool):
                return value
    return None


def _first_list(*sources: Mapping[str, Any], keys: Sequence[str]) -> list[Any]:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, list):
                return value
    return []


def _synthesize_pipeline_handoff(handoff: HandoffMessage, *, capture_root: Path) -> Path:
    """Materialize pipeline_handoff.json from raw sidecars when the iOS bundle omits it.

    We never invent provenance: values come only from raw/manifest.json and
    raw/capture_context.json, both of which the iOS app already writes.
    """

    raw_root = capture_root / "raw"
    manifest = _read_optional_json_object(raw_root / "manifest.json")
    context = _read_optional_json_object(raw_root / "capture_context.json")

    site_submission_id = _first_non_empty(
        manifest, context, keys=("site_submission_id", "siteSubmissionId")
    )
    buyer_request_id = _first_non_empty(
        manifest, context, keys=("buyer_request_id", "buyerRequestId")
    )
    capture_job_id = _first_non_empty(manifest, context, keys=("capture_job_id", "captureJobId"))
    request_id = buyer_request_id or capture_job_id

    requested_outputs: list[str] = []
    for source in (manifest, context):
        for key in ("requested_outputs", "requestedOutputs", "requested_lanes", "requestedLanes"):
            value = source.get(key)
            if isinstance(value, list):
                for item in value:
                    text = str(item).strip()
                    if text and text not in requested_outputs:
                        requested_outputs.append(text)

    payload: dict[str, Any] = {
        "schema_version": "pipeline_handoff.v1",
        "synthesized": True,
        "synthesized_from": ["raw/manifest.json", "raw/capture_context.json"],
        "scene_id": handoff.scene_id,
        "capture_id": handoff.capture_id,
        "bucket": handoff.bucket,
        "raw_prefix_uri": handoff.raw_prefix_uri,
        "site_submission_id": site_submission_id,
        "buyer_request_id": buyer_request_id,
        "capture_job_id": capture_job_id,
        "owner_system": {
            "owner_system": "blueprint_capture",
            "request_id": request_id,
            "site_submission_id": site_submission_id,
            "buyer_request_id": buyer_request_id,
            "capture_job_id": capture_job_id,
        },
    }
    if requested_outputs:
        payload["requested_outputs"] = requested_outputs

    destination = capture_root / "pipeline_handoff.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(
        "pubsub_handoff.synthesized_pipeline_handoff",
        extra={
            "scene_id": handoff.scene_id,
            "capture_id": handoff.capture_id,
            "capture_job_id": capture_job_id,
        },
    )
    return destination


def _control_plane_handoff_payload(
    handoff: HandoffMessage,
    *,
    capture_root: Path,
) -> dict[str, Any]:
    pipeline_handoff = _read_optional_json_object(capture_root / "pipeline_handoff.json")
    manifest = _read_optional_json_object(capture_root / "raw" / "manifest.json")
    context = _read_optional_json_object(capture_root / "raw" / "capture_context.json")
    owner_system = pipeline_handoff.get("owner_system")
    owner = dict(owner_system) if isinstance(owner_system, Mapping) else {}
    sources = (pipeline_handoff, owner, manifest, context)
    pipeline_handoff_uri = handoff.pipeline_handoff_uri or str(
        capture_root / "pipeline_handoff.json"
    )
    capture_descriptor_uri = (
        str(capture_root / "capture_descriptor.json")
        if (capture_root / "capture_descriptor.json").is_file()
        else None
    )
    payload: dict[str, Any] = {
        "bucket": handoff.bucket,
        "scene_id": handoff.scene_id,
        "capture_id": handoff.capture_id,
        "raw_prefix_uri": handoff.raw_prefix_uri,
        "pipeline_handoff_uri": pipeline_handoff_uri,
        "capture_descriptor_uri": capture_descriptor_uri,
        "capture_root": str(capture_root),
    }
    for output_key, keys in (
        ("site_submission_id", ("site_submission_id", "siteSubmissionId")),
        ("buyer_request_id", ("buyer_request_id", "buyerRequestId")),
        ("capture_job_id", ("capture_job_id", "captureJobId")),
        ("site_slug", ("site_slug", "siteSlug")),
    ):
        value = _first_non_empty(*sources, keys=keys)
        if value:
            payload[output_key] = value
    requested_outputs = _first_list(
        *sources,
        keys=("requested_outputs", "requestedOutputs"),
    )
    if requested_outputs:
        payload["requested_outputs"] = requested_outputs
    requested_lanes = _first_list(
        *sources,
        keys=("requested_lanes", "requestedLanes"),
    )
    if requested_lanes:
        payload["requested_lanes"] = requested_lanes
    robot_eval_requested = _first_bool(
        *sources,
        keys=("robot_eval_dataset_requested", "robotEvalDatasetRequested"),
    )
    if robot_eval_requested is not None:
        payload["robot_eval_dataset_requested"] = robot_eval_requested
    return payload


def _stage_control_plane_input(
    *,
    handoff: HandoffMessage,
    capture_root: Path,
    manifest_path: str | Path,
    work_dir: str | Path | None,
    staged_inputs_path: str | Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    from .live_pipeline_intake_service import stage_capture_handoff_for_control_plane

    payload = _control_plane_handoff_payload(handoff, capture_root=capture_root)
    result = stage_capture_handoff_for_control_plane(
        payload=payload,
        capture_root=capture_root,
        manifest_path=manifest_path,
        work_dir=work_dir,
        overwrite=overwrite,
        staged_inputs_path=staged_inputs_path,
    )
    if result.get("status") != "staged_for_control_plane":
        blockers = result.get("input_blockers") or result.get("blockers") or []
        raise PipelineError(
            "Pub/Sub handoff could not stage control-plane input: "
            + ", ".join(str(blocker) for blocker in blockers)
        )
    return result


JOB_LEDGER_FILENAME = "pipeline_job_ledger.json"
JOB_LEDGER_SCHEMA_VERSION = "pipeline_job_ledger.v1"
JOB_STATUS_SCHEMA_VERSION = "pipeline_job_status.v1"
PROVIDER_OPS_STATUS_SCHEMA_VERSION = "provider_ops_status.v1"
_JOB_RETRYABLE_STATUSES = {"processing", "failed_retryable"}
_PROVIDER_STATUS_FILENAMES = {
    "wam_compute_run_result.json",
    "runpod_wam_async_poll_manifest.json",
    "runpod_wam_async_create_manifest.json",
    "vast_wam_async_poll_manifest.json",
    "vast_wam_async_create_manifest.json",
    "vast_provider_adapter_result.json",
    "remote_cloud_execution_closure_manifest.json",
    "provider_reliability_manifest.json",
    "runpod_wam_provider_reliability_manifest.json",
    "gpu_provider_launch_request.json",
}
_PROVIDER_STATUS_FIELD_NAMES = {
    "continuing_spend_from_this_run",
    "teardown_status",
    "provider_phase",
    "provider_command_status",
    "output_availability",
    "provider_runtime_output_zip_path",
    "provider_output_validation_status",
}


def _read_job_ledger(capture_root: Path) -> dict[str, Any]:
    ledger_path = capture_root / JOB_LEDGER_FILENAME
    if not ledger_path.is_file():
        return {}
    try:
        loaded = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_job_ledger(capture_root: Path, ledger: Mapping[str, Any]) -> None:
    (capture_root / JOB_LEDGER_FILENAME).write_text(
        json.dumps(dict(ledger), indent=2, sort_keys=True), encoding="utf-8"
    )


def _string(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _relative_to(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _attempt_history(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    history = ledger.get("attempt_history")
    if not isinstance(history, list):
        return []
    return [dict(item) for item in history if isinstance(item, Mapping)]


def _looks_like_provider_status_artifact(path: Path, payload: Mapping[str, Any]) -> bool:
    if path.name in _PROVIDER_STATUS_FILENAMES:
        return True
    if any(key in payload for key in _PROVIDER_STATUS_FIELD_NAMES):
        return True
    schema = _string(payload.get("schema_version"))
    return bool(
        schema
        and any(
            token in schema
            for token in ("provider", "runpod_wam", "vast_wam", "wam_compute")
        )
    )


def _provider_status_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for key in (
        "blockers",
        "provider_command_blockers",
        "runtime_result_blockers",
        "completion_blockers",
    ):
        for blocker in _string_list(payload.get(key)):
            if blocker not in blockers:
                blockers.append(blocker)
    nested_validation = payload.get("provider_output_validation")
    if isinstance(nested_validation, Mapping):
        for blocker in _string_list(nested_validation.get("blockers")):
            if blocker not in blockers:
                blockers.append(blocker)
    return blockers


def _provider_status_row(
    *,
    capture_root: Path,
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    teardown_status = _string(payload.get("teardown_status")) or _string(
        payload.get("teardown_action")
    )
    provider_phase = (
        _string(payload.get("provider_phase"))
        or _string(payload.get("pod_status"))
        or _string(payload.get("instance_status"))
        or _string(payload.get("status"))
    )
    continuing_spend = _bool_or_none(payload.get("continuing_spend_from_this_run"))
    return {
        "artifact_path": _relative_to(capture_root, path),
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
        "provider": payload.get("provider"),
        "provider_phase": provider_phase or None,
        "provider_command_status": payload.get("provider_command_status"),
        "runtime_result_status": payload.get("runtime_result_status"),
        "output_availability": payload.get("output_availability"),
        "output_zip_present": payload.get("output_zip_present"),
        "provider_runtime_output_zip_path": payload.get(
            "provider_runtime_output_zip_path"
        )
        or payload.get("output_zip_path")
        or payload.get("output_path"),
        "provider_output_validation_status": payload.get(
            "provider_output_validation_status"
        ),
        "teardown_status": teardown_status or None,
        "teardown_performed": payload.get("teardown_performed"),
        "continuing_spend_from_this_run": continuing_spend,
        "blockers": _provider_status_blockers(payload),
    }


def _provider_ops_status(capture_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if capture_root.is_dir():
        for path in sorted(capture_root.rglob("*.json")):
            if path.name == JOB_LEDGER_FILENAME:
                continue
            payload = _read_optional_json_object(path)
            if not payload or not _looks_like_provider_status_artifact(path, payload):
                continue
            rows.append(
                _provider_status_row(
                    capture_root=capture_root,
                    path=path,
                    payload=payload,
                )
            )
    continuing_spend_any = any(
        row.get("continuing_spend_from_this_run") is True for row in rows
    )
    blocker_count = sum(len(row.get("blockers") or []) for row in rows)
    if continuing_spend_any:
        status = "running_spend_attention_required"
    elif rows and blocker_count:
        status = "blocked_or_review_required"
    elif rows:
        status = "observed"
    else:
        status = "not_observed"
    return {
        "schema_version": PROVIDER_OPS_STATUS_SCHEMA_VERSION,
        "status": status,
        "provider_artifact_count": len(rows),
        "continuing_spend_from_this_run": continuing_spend_any,
        "teardown_attention_required": continuing_spend_any,
        "blocker_count": blocker_count,
        "provider_statuses": rows,
        "claim_boundary": {
            "ops_status_only": True,
            "status_query_is_not_provider_execution": True,
            "provider_runtime_success_is_not_task_success": True,
            "continuing_spend_true_requires_operator_poll_or_teardown": True,
        },
    }


def read_handoff_job_status(
    *,
    storage_root: Path,
    bucket: str,
    scene_id: str,
    capture_id: str,
) -> dict[str, Any]:
    """Read durable local job state for a staged Pub/Sub handoff capture."""

    capture_root = storage_root / bucket / "scenes" / scene_id / "captures" / capture_id
    ledger = _read_job_ledger(capture_root)
    run_e2e_stage_ledger = _read_optional_json_object(
        capture_root / "pipeline" / "run_e2e_stage_ledger.json"
    )
    staged_capture_present = capture_root.is_dir()
    upload_complete_present = (
        capture_root / "raw" / "capture_upload_complete.json"
    ).is_file()
    pipeline_handoff_present = (capture_root / "pipeline_handoff.json").is_file()
    provider_ops_status = _provider_ops_status(capture_root)
    if ledger:
        status = str(ledger.get("status") or "unknown").strip() or "unknown"
    elif staged_capture_present:
        status = "not_started"
    else:
        status = "not_staged"
    return {
        "schema_version": JOB_STATUS_SCHEMA_VERSION,
        "status": status,
        "bucket": bucket,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "capture_root": str(capture_root),
        "staged_capture_present": staged_capture_present,
        "upload_complete_present": upload_complete_present,
        "pipeline_handoff_present": pipeline_handoff_present,
        "job_ledger_present": bool(ledger),
        "attempt_count": int(ledger.get("attempt_count") or 0) if ledger else 0,
        "run_e2e_status": ledger.get("run_e2e_status") if ledger else None,
        "run_e2e_stage_ledger_present": bool(run_e2e_stage_ledger),
        "run_e2e_stage_ledger_path": str(
            capture_root / "pipeline" / "run_e2e_stage_ledger.json"
        ),
        "run_e2e_stage_status": run_e2e_stage_ledger.get("status")
        if run_e2e_stage_ledger
        else None,
        "run_e2e_current_stage": run_e2e_stage_ledger.get("current_stage")
        if run_e2e_stage_ledger
        else None,
        "run_e2e_failed_stage": run_e2e_stage_ledger.get("failed_stage")
        if run_e2e_stage_ledger
        else None,
        "run_e2e_last_completed_stage": run_e2e_stage_ledger.get(
            "last_completed_stage"
        )
        if run_e2e_stage_ledger
        else None,
        "run_e2e_stage_ledger": run_e2e_stage_ledger or None,
        "provider_ops_status": provider_ops_status,
        "provider_runtime_status": provider_ops_status.get("status"),
        "provider_runtime_artifact_count": provider_ops_status.get(
            "provider_artifact_count"
        ),
        "continuing_spend_from_this_run": provider_ops_status.get(
            "continuing_spend_from_this_run"
        ),
        "teardown_attention_required": provider_ops_status.get(
            "teardown_attention_required"
        ),
        "started_at": ledger.get("started_at") if ledger else None,
        "updated_at": ledger.get("updated_at") if ledger else None,
        "last_attempt_started_at": (
            ledger.get("last_attempt_started_at") if ledger else None
        ),
        "completed_at": ledger.get("completed_at") if ledger else None,
        "last_failed_at": ledger.get("last_failed_at") if ledger else None,
        "last_error_type": ledger.get("last_error_type") if ledger else None,
        "last_error": ledger.get("last_error") if ledger else None,
        "retry_expected_on_redelivery": status in _JOB_RETRYABLE_STATUSES,
        "completed_redelivery_is_noop": status == "completed",
        "attempt_history": _attempt_history(ledger),
        "ledger": ledger,
    }


def process_handoff_payload(
    payload: bytes | str | Mapping[str, Any],
    *,
    storage_root: Path,
    provider: str,
    run_e2e: Callable[..., dict[str, Any]] = run_end_to_end,
    storage_client: storage.Client | None = None,
    run_evaluation_prep: bool = True,
    run_e2e_enabled: bool = True,
    stage_control_plane: bool = False,
    control_plane_manifest_path: str | Path | None = None,
    control_plane_work_dir: str | Path | None = None,
    control_plane_staged_inputs_path: str | Path | None = None,
    overwrite_control_plane_input: bool = False,
) -> dict[str, Any]:
    handoff = parse_handoff_payload(payload)
    capture_root = stage_handoff_capture(
        handoff,
        storage_root=storage_root,
        storage_client=storage_client,
    )

    # Idempotency: Pub/Sub is at-least-once, so a redelivered message for a
    # capture that already completed must not re-run (and re-bill) the
    # pipeline. A "processing" marker from a crashed run is retried.
    ledger = _read_job_ledger(capture_root)
    if ledger.get("status") == "completed":
        logger.info(
            "pubsub_handoff.skipped_already_processed",
            extra={
                "scene_id": handoff.scene_id,
                "capture_id": handoff.capture_id,
            },
        )
        return {
            "schema_version": "v1",
            "status": "skipped_already_processed",
            "bucket": handoff.bucket,
            "scene_id": handoff.scene_id,
            "capture_id": handoff.capture_id,
            "capture_root": str(capture_root),
            "job_ledger": ledger,
        }

    attempt_count = int(ledger.get("attempt_count") or 0) + 1
    previous_history = _attempt_history(ledger)
    job_started_at = str(ledger.get("started_at") or "").strip() or utc_now_iso()
    attempt_started_at = utc_now_iso()
    _write_job_ledger(
        capture_root,
        {
            "schema_version": JOB_LEDGER_SCHEMA_VERSION,
            "status": "processing",
            "scene_id": handoff.scene_id,
            "capture_id": handoff.capture_id,
            "attempt_count": attempt_count,
            "started_at": job_started_at,
            "updated_at": attempt_started_at,
            "last_attempt_started_at": attempt_started_at,
            "attempt_history": previous_history,
        },
    )
    run_kwargs: dict[str, Any] = {
        "capture_root": str(capture_root),
        "provider": provider,
        "run_evaluation_prep": run_evaluation_prep,
        "resume_completed_stages": True,
    }
    robot_eval_job_request = _resolve_staged_handoff_path(
        handoff.robot_eval_job_request_uri,
        handoff=handoff,
        capture_root=capture_root,
        storage_root=storage_root,
        expect_directory=False,
    )
    robot_eval_request_inbox = _resolve_staged_handoff_path(
        handoff.robot_eval_request_inbox_uri,
        handoff=handoff,
        capture_root=capture_root,
        storage_root=storage_root,
        expect_directory=True,
    )
    if robot_eval_job_request is not None:
        run_kwargs["robot_eval_job_request"] = str(robot_eval_job_request)
    if robot_eval_request_inbox is not None:
        run_kwargs["robot_eval_request_inbox"] = str(robot_eval_request_inbox)
    if robot_eval_job_request is not None or robot_eval_request_inbox is not None:
        run_kwargs.update(
            {
                "robot_eval_job_id": handoff.robot_eval_job_id,
                "robot_eval_provisioner": (
                    handoff.robot_eval_provisioner or "fixture_local"
                ),
                "robot_eval_simulator": handoff.robot_eval_simulator or "fixture",
                "robot_eval_evaluation_substrate": (
                    handoff.robot_eval_evaluation_substrate
                ),
                "robot_eval_budget_usd": handoff.robot_eval_budget_usd,
                "allow_robot_eval_gpu_provisioning": False,
                "allow_robot_eval_simulator_execution": False,
            }
        )
    control_plane_staging: dict[str, Any] | None = None
    failure_stage = "run_e2e"
    try:
        if stage_control_plane:
            failure_stage = "control_plane_staging"
            if control_plane_manifest_path is None:
                raise PipelineError(
                    "Pub/Sub handoff control-plane staging requires a manifest path."
                )
            control_plane_staging = _stage_control_plane_input(
                handoff=handoff,
                capture_root=capture_root,
                manifest_path=control_plane_manifest_path,
                work_dir=control_plane_work_dir,
                staged_inputs_path=control_plane_staged_inputs_path,
                overwrite=overwrite_control_plane_input,
            )
        failure_stage = "run_e2e"
        if run_e2e_enabled:
            result = run_e2e(**run_kwargs)
        else:
            result = {
                "status": "skipped",
                "reason": "run_e2e_disabled_after_control_plane_staging",
            }
    except Exception as exc:
        failed_at = utc_now_iso()
        failure_record = {
            "attempt_number": attempt_count,
            "status": "failed_retryable",
            "stage": failure_stage,
            "started_at": attempt_started_at,
            "failed_at": failed_at,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_job_ledger(
            capture_root,
            {
                "schema_version": JOB_LEDGER_SCHEMA_VERSION,
                "status": "failed_retryable",
                "scene_id": handoff.scene_id,
                "capture_id": handoff.capture_id,
                "attempt_count": attempt_count,
                "started_at": job_started_at,
                "updated_at": failed_at,
                "last_attempt_started_at": attempt_started_at,
                "last_failed_at": failed_at,
                "last_error_type": type(exc).__name__,
                "last_error": str(exc),
                "attempt_history": [*previous_history, failure_record],
            },
        )
        raise
    completed_at = utc_now_iso()
    control_plane_staging_status = (
        str(control_plane_staging.get("status") or "") or None
        if control_plane_staging
        else None
    )
    control_plane_staging_path = (
        str((control_plane_staging.get("webapp_staging") or {}).get("target_path") or "")
        or None
        if control_plane_staging
        else None
    )
    completion_record = {
        "attempt_number": attempt_count,
        "status": "completed",
        "stage": "run_e2e",
        "started_at": attempt_started_at,
        "completed_at": completed_at,
        "run_e2e_status": str(result.get("status") or "") or None,
    }
    if control_plane_staging:
        completion_record.update(
            {
                "control_plane_staging_status": control_plane_staging_status,
                "control_plane_staging_path": control_plane_staging_path,
            }
        )
    ledger_update = {
        "schema_version": JOB_LEDGER_SCHEMA_VERSION,
        "status": "completed",
        "scene_id": handoff.scene_id,
        "capture_id": handoff.capture_id,
        "attempt_count": attempt_count,
        "started_at": job_started_at,
        "updated_at": completed_at,
        "last_attempt_started_at": attempt_started_at,
        "completed_at": completed_at,
        "run_e2e_status": str(result.get("status") or "") or None,
        "last_error_type": None,
        "last_error": None,
        "attempt_history": [*previous_history, completion_record],
    }
    if control_plane_staging:
        ledger_update.update(
            {
                "control_plane_staging_status": control_plane_staging_status,
                "control_plane_staging_path": control_plane_staging_path,
            }
        )
    _write_job_ledger(
        capture_root,
        ledger_update,
    )
    return {
        "schema_version": "v1",
        "status": "processed",
        "bucket": handoff.bucket,
        "scene_id": handoff.scene_id,
        "capture_id": handoff.capture_id,
        "capture_root": str(capture_root),
        "run_e2e": result,
        "control_plane_staging": control_plane_staging,
    }


def pull_and_process(
    *,
    subscription: str,
    storage_root: Path,
    provider: str,
    max_messages: int,
    run_evaluation_prep: bool = True,
    run_e2e_enabled: bool = True,
    stage_control_plane: bool = False,
    control_plane_manifest_path: str | Path | None = None,
    control_plane_work_dir: str | Path | None = None,
    control_plane_staged_inputs_path: str | Path | None = None,
    overwrite_control_plane_input: bool = False,
) -> int:
    from google.cloud import pubsub_v1

    subscriber = pubsub_v1.SubscriberClient()
    response = subscriber.pull(
        request={
            "subscription": subscription,
            "max_messages": max_messages,
        },
        timeout=30,
    )
    ack_ids: list[str] = []
    for received in response.received_messages:
        message = received.message
        logger.info(
            "pubsub_handoff.received",
            extra={
                "message_id": message.message_id,
                "attributes": dict(message.attributes),
            },
        )
        # One poison message must not block the batch: successes still ack;
        # the failed message stays un-acked so the subscription's retry /
        # dead-letter policy owns redelivery. The job ledger written by
        # process_handoff_payload makes redelivered completions no-ops.
        try:
            process_handoff_payload(
                message.data,
                storage_root=storage_root,
                provider=provider,
                run_evaluation_prep=run_evaluation_prep,
                run_e2e_enabled=run_e2e_enabled,
                stage_control_plane=stage_control_plane,
                control_plane_manifest_path=control_plane_manifest_path,
                control_plane_work_dir=control_plane_work_dir,
                control_plane_staged_inputs_path=control_plane_staged_inputs_path,
                overwrite_control_plane_input=overwrite_control_plane_input,
            )
        except Exception:
            logger.exception(
                "pubsub_handoff.processing_failed",
                extra={"message_id": message.message_id},
            )
            continue
        ack_ids.append(received.ack_id)

    if ack_ids:
        subscriber.acknowledge(request={"subscription": subscription, "ack_ids": ack_ids})
    return len(ack_ids)


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(f"Pub/Sub handoff missing required string: {key}")
    return value.strip()


def _optional_string(data: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise PipelineError(f"Pub/Sub handoff {key} must be a string when present.")
        if value.strip():
            return value.strip()
    return None


def _optional_number(data: Mapping[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise PipelineError(f"Pub/Sub handoff {key} must be a number when present.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError as exc:
            raise PipelineError(
                f"Pub/Sub handoff {key} must be a number when present."
            ) from exc
    raise PipelineError(f"Pub/Sub handoff {key} must be a number when present.")


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_staged_handoff_path(
    value: str | None,
    *,
    handoff: HandoffMessage,
    capture_root: Path,
    storage_root: Path,
    expect_directory: bool,
) -> Path | None:
    if not value:
        return None
    if value.startswith("gs://"):
        prefix = f"gs://{handoff.bucket}/"
        if not value.startswith(prefix):
            raise PipelineError(
                "Pub/Sub handoff robot eval path must reference the handoff bucket."
            )
        local_path = storage_root / handoff.bucket / value[len(prefix) :]
    else:
        path = Path(value)
        local_path = path if path.is_absolute() else capture_root / path
    if expect_directory:
        if not local_path.is_dir():
            raise PipelineError(
                f"Staged robot eval request inbox is missing: {local_path}"
            )
    elif not local_path.is_file():
        raise PipelineError(f"Staged robot eval job request is missing: {local_path}")
    return local_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pull BlueprintCapture Pub/Sub handoffs and run blueprint_pipeline.run_e2e."
    )
    parser.add_argument("--subscription")
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument(
        "--provider",
        default="openai",
        choices=("local", "claude", "openai"),
        help=(
            "Agent-review provider for run_e2e. Production defaults to openai; "
            "local is a deterministic no-LLM contract lane."
        ),
    )
    parser.add_argument("--max-messages", type=int, default=1)
    parser.add_argument("--skip-evaluation-prep", action="store_true")
    parser.add_argument(
        "--stage-control-plane",
        action="store_true",
        default=_env_truthy("BLUEPRINT_PUBSUB_HANDOFF_STAGE_CONTROL_PLANE"),
        help="Stage converted capture handoffs into the live control-plane inbox.",
    )
    parser.add_argument(
        "--control-plane-manifest",
        default=os.getenv("BLUEPRINT_CONTROL_PLANE_OUTPUT_PATH"),
        help="Path to live_pipeline_control_plane_manifest.json for inbox staging.",
    )
    parser.add_argument(
        "--control-plane-work-dir",
        default=os.getenv("BLUEPRINT_LIVE_PIPELINE_INTAKE_WORK_DIR"),
        help="Directory for Pub/Sub-to-control-plane staging candidates.",
    )
    parser.add_argument(
        "--control-plane-staged-inputs-path",
        default=os.getenv("BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH"),
        help="Optional live_pipeline_staged_inputs.json path to update during staging.",
    )
    parser.add_argument(
        "--overwrite-control-plane-input",
        action="store_true",
        default=_env_truthy("BLUEPRINT_LIVE_PIPELINE_INTAKE_OVERWRITE"),
    )
    parser.add_argument(
        "--skip-run-e2e",
        action="store_true",
        default=_env_truthy("BLUEPRINT_PUBSUB_HANDOFF_SKIP_RUN_E2E"),
        help="Only stage the capture/control-plane input; do not run run_e2e in the listener.",
    )
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--bucket")
    parser.add_argument("--scene-id")
    parser.add_argument("--capture-id")
    args = parser.parse_args(argv)

    storage_root = args.storage_root or Path(tempfile.gettempdir()) / "blueprint-pubsub-handoffs"
    if args.status:
        missing = [
            name
            for name, value in (
                ("--bucket", args.bucket),
                ("--scene-id", args.scene_id),
                ("--capture-id", args.capture_id),
            )
            if not value
        ]
        if missing:
            parser.error("--status requires " + ", ".join(missing))
        print(
            json.dumps(
                read_handoff_job_status(
                    storage_root=storage_root,
                    bucket=str(args.bucket),
                    scene_id=str(args.scene_id),
                    capture_id=str(args.capture_id),
                ),
                sort_keys=True,
            )
        )
        return 0

    if not args.subscription:
        parser.error("--subscription is required unless --status is used")
    if args.stage_control_plane and not args.control_plane_manifest:
        parser.error("--stage-control-plane requires --control-plane-manifest")
    if args.skip_run_e2e and not args.stage_control_plane:
        parser.error("--skip-run-e2e requires --stage-control-plane")
    processed = pull_and_process(
        subscription=args.subscription,
        storage_root=storage_root,
        provider=args.provider,
        max_messages=max(1, args.max_messages),
        run_evaluation_prep=not args.skip_evaluation_prep,
        run_e2e_enabled=not args.skip_run_e2e,
        stage_control_plane=args.stage_control_plane,
        control_plane_manifest_path=args.control_plane_manifest,
        control_plane_work_dir=args.control_plane_work_dir,
        control_plane_staged_inputs_path=args.control_plane_staged_inputs_path,
        overwrite_control_plane_input=args.overwrite_control_plane_input,
    )
    print(json.dumps({"processed": processed, "storage_root": str(storage_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
