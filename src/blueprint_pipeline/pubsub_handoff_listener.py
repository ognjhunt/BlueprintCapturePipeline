"""Pull BlueprintCapture bridge handoffs from Pub/Sub and run the pipeline."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import socket
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping, Sequence

import google.auth
from google.cloud import storage

from .common import PipelineError, utc_now_iso, write_json
from .run_e2e import run_end_to_end
from .core.security_controls import (
    SecurityValidationError,
    contained_path,
    prove_path_contained,
    strict_gcs_bucket,
    strict_identifier,
)

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
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PipelineError("Pub/Sub handoff payload is not valid UTF-8.") from exc
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise PipelineError(f"Pub/Sub handoff payload is not valid JSON: {exc}") from exc
    else:
        data = dict(payload)

    try:
        bucket = strict_gcs_bucket(_required_string(data, "bucket"))
        scene_id = strict_identifier(_required_string(data, "scene_id"), field="scene_id")
        capture_id = strict_identifier(
            _required_string(data, "capture_id"),
            field="capture_id",
        )
        robot_eval_job_id = _optional_string(data, "robot_eval_job_id")
        if robot_eval_job_id:
            robot_eval_job_id = strict_identifier(
                robot_eval_job_id,
                field="robot_eval_job_id",
            )
    except SecurityValidationError as exc:
        raise PipelineError(f"Invalid Pub/Sub handoff identity: {exc}") from exc
    raw_prefix_uri = _required_string(data, "raw_prefix_uri")
    if raw_prefix_uri != f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/raw":
        raise PipelineError(
            "Pub/Sub handoff raw_prefix_uri does not match bucket/scene/capture identity: "
            f"{raw_prefix_uri}"
        )

    pipeline_handoff_uri = data.get("pipeline_handoff_uri")
    if pipeline_handoff_uri is not None and not isinstance(pipeline_handoff_uri, str):
        raise PipelineError("Pub/Sub handoff pipeline_handoff_uri must be a string when present.")
    expected_pipeline_handoff_uri = (
        f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/pipeline_handoff.json"
    )
    if pipeline_handoff_uri is not None and pipeline_handoff_uri != expected_pipeline_handoff_uri:
        raise PipelineError(
            "Pub/Sub handoff pipeline_handoff_uri does not match bucket/scene/capture identity."
        )

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
        robot_eval_job_id=robot_eval_job_id,
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
    resolved_storage_root = storage_root.resolve()
    bucket_root = contained_path(
        resolved_storage_root,
        handoff.bucket,
        field="Pub/Sub staging bucket path",
    )
    capture_root = contained_path(
        bucket_root,
        "scenes",
        handoff.scene_id,
        "captures",
        handoff.capture_id,
        field="Pub/Sub capture staging path",
    )
    capture_root.mkdir(parents=True, exist_ok=True)

    blobs = list(client.list_blobs(handoff.bucket, prefix=f"{handoff.capture_prefix}/"))
    if not blobs:
        raise PipelineError(f"No objects found for handoff prefix: {handoff.capture_prefix}/")

    for blob in blobs:
        blob_name = str(blob.name or "")
        expected_prefix = f"{handoff.capture_prefix}/"
        if not blob_name.startswith(expected_prefix):
            raise PipelineError("Pub/Sub blob escaped the declared capture prefix")
        blob_path = PurePosixPath(blob_name)
        if (
            blob_path.is_absolute()
            or any(part in {"", ".", ".."} for part in blob_path.parts)
            or "\\" in blob_name
            or "\x00" in blob_name
        ):
            raise PipelineError("Pub/Sub blob name contains an unsafe path")
        if blob_name.endswith("/"):
            continue
        try:
            destination = contained_path(
                bucket_root,
                *blob_path.parts,
                field="Pub/Sub blob destination",
            )
            prove_path_contained(
                capture_root,
                destination,
                field="Pub/Sub blob capture destination",
            )
        except SecurityValidationError as exc:
            raise PipelineError(str(exc)) from exc
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
    write_json(destination, payload)
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
JOB_OUTPUT_COMMIT_FILENAME = "pipeline_job_output_commit.json"
JOB_LEDGER_SCHEMA_VERSION = "pipeline_job_ledger.v1"
JOB_OUTPUT_COMMIT_SCHEMA_VERSION = "pipeline_job_output_commit.v1"
JOB_STATUS_SCHEMA_VERSION = "pipeline_job_status.v1"
PROVIDER_OPS_STATUS_SCHEMA_VERSION = "provider_ops_status.v1"
DEFAULT_JOB_LEASE_SECONDS = 900
DEFAULT_ACK_DEADLINE_SECONDS = 600
DEFAULT_MAX_DELIVERY_ATTEMPTS = 5
_JOB_RETRYABLE_STATUSES = {
    "processing",
    "failed_retryable",
    "retryable_blocked",
    "lease_active_retryable",
}
_HANDOFF_TERMINAL_SUCCESS_STATUSES = {
    "completed",
    "ok",
    "qualified",
    "processed",
    "skipped",
    "succeeded",
}
_ROBOT_EVAL_RETRYABLE_STATUSES = {
    "blocked",
    "retryable_blocked",
    "fatal_infrastructure",
    "blocked_all_requests_retryable",
}
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


RECONSTRUCTION_POLICY_ROOT_ENV = "BLUEPRINT_CAPTURE_RECONSTRUCTION_POLICY_ROOT"
RECONSTRUCTION_QUEUE_ROOT_ENV = "BLUEPRINT_CAPTURE_RECONSTRUCTION_QUEUE_ROOT"
RECONSTRUCTION_SOURCE_COMMIT_ENV = "BLUEPRINT_CAPTURE_RECONSTRUCTION_SOURCE_COMMIT_SHA"


def _enqueue_capture_reconstruction_if_configured(
    *,
    handoff: HandoffMessage,
    capture_root: Path,
) -> dict[str, Any]:
    """Queue this capture's 3DGS launch when a site/task policy admits it.

    Abstention is an ordinary outcome, not a delivery failure: a capture whose
    site/task has no registered policy, or whose raw bytes disagree with the
    device hash manifest, is recorded and left alone.  Nacking here would
    dead-letter a message the listener actually handled correctly, and the
    queue itself is idempotent, so a genuine redelivery cannot double-book.
    """

    policy_root = str(os.getenv(RECONSTRUCTION_POLICY_ROOT_ENV) or "").strip()
    queue_root = str(os.getenv(RECONSTRUCTION_QUEUE_ROOT_ENV) or "").strip()
    if not policy_root or not queue_root:
        return {"status": "not_configured", "enqueued": False}

    from .capture_reconstruction_launch_dispatcher import (
        CaptureReconstructionLaunchError,
        enqueue_capture_reconstruction,
    )

    payload = {
        "bucket": handoff.bucket,
        "scene_id": handoff.scene_id,
        "capture_id": handoff.capture_id,
        "raw_prefix_uri": handoff.raw_prefix_uri,
    }
    try:
        receipt = enqueue_capture_reconstruction(
            capture_root=capture_root,
            payload=payload,
            policy_root=policy_root,
            queue_root=queue_root,
            source_commit_sha=str(
                os.getenv(RECONSTRUCTION_SOURCE_COMMIT_ENV) or ""
            ).strip(),
            requested_at=utc_now_iso(),
        )
    except CaptureReconstructionLaunchError as exc:
        return {
            "status": "abstained",
            "enqueued": False,
            "capture_id": handoff.capture_id,
            "blockers": [str(exc)],
        }
    return {
        "status": receipt["status"],
        "enqueued": not receipt["already_exists"],
        "already_exists": receipt["already_exists"],
        "capture_id": receipt["capture_id"],
        "capture_digest": receipt["capture_digest"],
        "idempotency_key": receipt["idempotency_key"],
        "queue_path": receipt["queue_path"],
        "provider_mutation_performed": False,
    }


def _read_job_ledger(capture_root: Path) -> dict[str, Any]:
    ledger_path = capture_root / JOB_LEDGER_FILENAME
    if not ledger_path.is_file():
        return {}
    try:
        loaded = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema_version": JOB_LEDGER_SCHEMA_VERSION,
            "status": "corrupt",
            "ledger_read_error": type(exc).__name__,
        }
    if not isinstance(loaded, dict):
        return {
            "schema_version": JOB_LEDGER_SCHEMA_VERSION,
            "status": "corrupt",
            "ledger_read_error": "not_mapping",
        }
    return loaded


@contextmanager
def _locked_job_ledger(capture_root: Path) -> Iterator[dict[str, Any]]:
    """Hold the per-capture ledger lock while reading or committing state.

    ``flock`` supplies cross-process exclusion. ``write_json`` supplies the
    same-filesystem temp/fsync/replace commit, so a killed writer leaves either
    the prior complete ledger or the complete next revision.
    """

    capture_root.mkdir(parents=True, exist_ok=True)
    lock_path = capture_root / f".{JOB_LEDGER_FILENAME}.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield _read_job_ledger(capture_root)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _commit_job_ledger(
    capture_root: Path,
    ledger: Mapping[str, Any],
    *,
    previous_revision: int,
) -> dict[str, Any]:
    committed = {
        **dict(ledger),
        "schema_version": JOB_LEDGER_SCHEMA_VERSION,
        "revision": previous_revision + 1,
    }
    write_json(capture_root / JOB_LEDGER_FILENAME, committed)
    return committed


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_at(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _lease_owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"


def _claim_job_lease(
    capture_root: Path,
    *,
    scene_id: str,
    capture_id: str,
    owner: str,
    lease_seconds: int,
    now: datetime | None = None,
) -> tuple[str, dict[str, Any]]:
    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    with _locked_job_ledger(capture_root) as ledger:
        revision = int(ledger.get("revision") or 0)
        status = _string(ledger.get("status"))
        if status == "corrupt":
            return "corrupt", dict(ledger)
        if status == "completed":
            return "completed", dict(ledger)
        expires_at = _parse_utc_timestamp(ledger.get("lease_expires_at"))
        if (
            status == "processing"
            and expires_at is not None
            and expires_at > current_time
            and _string(ledger.get("lease_owner")) != owner
        ):
            return "active", dict(ledger)

        attempt_count = int(ledger.get("attempt_count") or 0) + 1
        started_at = _string(ledger.get("started_at")) or _iso_at(current_time)
        token = uuid.uuid4().hex
        claimed = _commit_job_ledger(
            capture_root,
            {
                "status": "processing",
                "scene_id": scene_id,
                "capture_id": capture_id,
                "attempt_count": attempt_count,
                "started_at": started_at,
                "updated_at": _iso_at(current_time),
                "last_attempt_started_at": _iso_at(current_time),
                "attempt_history": _attempt_history(ledger),
                "lease_owner": owner,
                "lease_token": token,
                "lease_acquired_at": _iso_at(current_time),
                "lease_heartbeat_at": _iso_at(current_time),
                "lease_expires_at": _iso_at(
                    current_time + timedelta(seconds=max(1, lease_seconds))
                ),
                "recovered_expired_lease": status == "processing",
                "previous_lease_owner": ledger.get("lease_owner")
                if status == "processing"
                else None,
            },
            previous_revision=revision,
        )
        return "claimed", claimed


def _heartbeat_job_lease(
    capture_root: Path,
    *,
    owner: str,
    token: str,
    lease_seconds: int,
) -> bool:
    now = datetime.now(timezone.utc)
    with _locked_job_ledger(capture_root) as ledger:
        if (
            ledger.get("status") != "processing"
            or _string(ledger.get("lease_owner")) != owner
            or _string(ledger.get("lease_token")) != token
        ):
            return False
        revision = int(ledger.get("revision") or 0)
        _commit_job_ledger(
            capture_root,
            {
                **ledger,
                "updated_at": _iso_at(now),
                "lease_heartbeat_at": _iso_at(now),
                "lease_expires_at": _iso_at(
                    now + timedelta(seconds=max(1, lease_seconds))
                ),
            },
            previous_revision=revision,
        )
    return True


def _finish_job_lease(
    capture_root: Path,
    *,
    owner: str,
    token: str,
    update: Mapping[str, Any],
) -> dict[str, Any]:
    with _locked_job_ledger(capture_root) as ledger:
        if (
            ledger.get("status") != "processing"
            or _string(ledger.get("lease_owner")) != owner
            or _string(ledger.get("lease_token")) != token
        ):
            raise PipelineError("Pub/Sub job ledger lease ownership was lost before commit.")
        revision = int(ledger.get("revision") or 0)
        return _commit_job_ledger(
            capture_root,
            {
                **ledger,
                **dict(update),
                "lease_owner": None,
                "lease_token": None,
                "lease_expires_at": None,
            },
            previous_revision=revision,
        )


class _JobLeaseHeartbeat:
    def __init__(
        self,
        *,
        capture_root: Path,
        owner: str,
        token: str,
        lease_seconds: int,
    ) -> None:
        self.capture_root = capture_root
        self.owner = owner
        self.token = token
        self.lease_seconds = max(1, lease_seconds)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        interval = max(1.0, min(float(self.lease_seconds) / 3.0, 60.0))
        while not self._stop.wait(interval):
            if not _heartbeat_job_lease(
                self.capture_root,
                owner=self.owner,
                token=self.token,
                lease_seconds=self.lease_seconds,
            ):
                return

    def __enter__(self) -> "_JobLeaseHeartbeat":
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


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


def _output_commit(
    capture_root: Path,
    *,
    scene_id: str,
    capture_id: str,
) -> dict[str, Any]:
    commit = _read_optional_json_object(capture_root / JOB_OUTPUT_COMMIT_FILENAME)
    if (
        commit.get("schema_version") != JOB_OUTPUT_COMMIT_SCHEMA_VERSION
        or commit.get("status") != "committed"
        or commit.get("scene_id") != scene_id
        or commit.get("capture_id") != capture_id
        or not _string(commit.get("result_sha256"))
    ):
        return {}
    return commit


def _write_output_commit(
    capture_root: Path,
    *,
    scene_id: str,
    capture_id: str,
    attempt_count: int,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    encoded = json.dumps(dict(result), sort_keys=True, default=str).encode("utf-8")
    commit = {
        "schema_version": JOB_OUTPUT_COMMIT_SCHEMA_VERSION,
        "status": "committed",
        "committed_at": utc_now_iso(),
        "scene_id": scene_id,
        "capture_id": capture_id,
        "attempt_count": attempt_count,
        "result_sha256": sha256(encoded).hexdigest(),
        "result_status": result.get("status"),
        "commit_is_idempotency_evidence_not_task_success": True,
    }
    write_json(capture_root / JOB_OUTPUT_COMMIT_FILENAME, commit)
    return commit


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

    try:
        safe_bucket = strict_gcs_bucket(bucket)
        safe_scene_id = strict_identifier(scene_id, field="scene_id")
        safe_capture_id = strict_identifier(capture_id, field="capture_id")
        bucket_root = contained_path(
            storage_root,
            safe_bucket,
            field="Pub/Sub status bucket path",
        )
        capture_root = contained_path(
            bucket_root,
            "scenes",
            safe_scene_id,
            "captures",
            safe_capture_id,
            field="Pub/Sub status capture path",
        )
    except SecurityValidationError as exc:
        raise PipelineError(f"Invalid Pub/Sub job status identity: {exc}") from exc
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


def _handoff_capture_root(
    handoff: HandoffMessage,
    *,
    storage_root: Path,
) -> Path:
    try:
        bucket_root = contained_path(
            storage_root.resolve(),
            handoff.bucket,
            field="Pub/Sub lease bucket path",
        )
        return contained_path(
            bucket_root,
            "scenes",
            handoff.scene_id,
            "captures",
            handoff.capture_id,
            field="Pub/Sub lease capture path",
        )
    except SecurityValidationError as exc:
        raise PipelineError(str(exc)) from exc


def _handoff_result_disposition(result: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Map pipeline/job results to Pub/Sub acknowledgement semantics."""

    statuses: list[str] = []
    for value in (
        result.get("status"),
        result.get("pipeline_status"),
        _mapping(result.get("robot_eval_job")).get("status"),
        _mapping(result.get("robot_eval_request_inbox")).get("status"),
    ):
        normalized = _string(value).lower()
        if normalized:
            statuses.append(normalized)
    retryable = [
        status
        for status in statuses
        if status in _ROBOT_EVAL_RETRYABLE_STATUSES
        or "retryable" in status
        or status.startswith("blocked")
        or status.startswith("failed")
    ]
    if retryable:
        return "retryable_blocked", retryable
    if statuses and all(
        status in _HANDOFF_TERMINAL_SUCCESS_STATUSES
        or status.startswith("completed")
        or status.endswith("_completed")
        or status in {
            "fixture_evaluation_completed",
            "simulator_command_completed",
            "skipped_already_processed",
        }
        for status in statuses
    ):
        return "terminal_success", []
    # run_e2e historically has no top-level status. Presence of its canonical
    # pipeline/final-artifact fields is the terminal-success contract.
    if result.get("pipeline_status") and result.get("final_bundle_path"):
        return "terminal_success", []
    return "retryable_blocked", ["pipeline_result_terminal_state_not_proven"]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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
    lease_owner: str | None = None,
    lease_seconds: int = DEFAULT_JOB_LEASE_SECONDS,
) -> dict[str, Any]:
    handoff = parse_handoff_payload(payload)
    capture_root = _handoff_capture_root(handoff, storage_root=storage_root)
    owner = lease_owner or _lease_owner()
    claim_status, ledger = _claim_job_lease(
        capture_root,
        scene_id=handoff.scene_id,
        capture_id=handoff.capture_id,
        owner=owner,
        lease_seconds=lease_seconds,
    )
    if claim_status == "completed":
        commit = _output_commit(
            capture_root,
            scene_id=handoff.scene_id,
            capture_id=handoff.capture_id,
        )
        if not commit:
            return {
                "schema_version": "v1",
                "status": "completed_output_commit_missing_retryable",
                "queue_disposition": "retryable",
                "bucket": handoff.bucket,
                "scene_id": handoff.scene_id,
                "capture_id": handoff.capture_id,
                "capture_root": str(capture_root),
                "job_ledger": ledger,
                "blockers": ["completed_handoff_output_commit_missing_or_invalid"],
            }
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
            "queue_disposition": "terminal_success",
            "output_commit": commit,
            "job_ledger": ledger,
        }
    if claim_status == "active":
        return {
            "schema_version": "v1",
            "status": "lease_active_retryable",
            "queue_disposition": "retryable",
            "bucket": handoff.bucket,
            "scene_id": handoff.scene_id,
            "capture_id": handoff.capture_id,
            "capture_root": str(capture_root),
            "job_ledger": ledger,
            "blockers": ["handoff_job_active_lease"],
        }
    if claim_status == "corrupt":
        return {
            "schema_version": "v1",
            "status": "job_ledger_corrupt_retryable",
            "queue_disposition": "retryable",
            "bucket": handoff.bucket,
            "scene_id": handoff.scene_id,
            "capture_id": handoff.capture_id,
            "capture_root": str(capture_root),
            "job_ledger": ledger,
            "blockers": ["handoff_job_ledger_corrupt"],
        }

    attempt_count = int(ledger.get("attempt_count") or 0)
    previous_history = _attempt_history(ledger)
    job_started_at = _string(ledger.get("started_at")) or utc_now_iso()
    attempt_started_at = _string(ledger.get("last_attempt_started_at")) or utc_now_iso()
    token = _string(ledger.get("lease_token"))
    recovered_commit = _output_commit(
        capture_root,
        scene_id=handoff.scene_id,
        capture_id=handoff.capture_id,
    )
    if ledger.get("recovered_expired_lease") is True and recovered_commit:
        recovered_at = utc_now_iso()
        recovered_record = {
            "attempt_number": attempt_count,
            "status": "completed_from_output_commit",
            "stage": "output_commit_recovery",
            "started_at": attempt_started_at,
            "completed_at": recovered_at,
            "result_sha256": recovered_commit.get("result_sha256"),
        }
        completed_ledger = _finish_job_lease(
            capture_root,
            owner=owner,
            token=token,
            update={
                "status": "completed",
                "updated_at": recovered_at,
                "completed_at": recovered_at,
                "output_commit_status": "committed",
                "output_commit_path": JOB_OUTPUT_COMMIT_FILENAME,
                "attempt_history": [*previous_history, recovered_record],
            },
        )
        return {
            "schema_version": "v1",
            "status": "skipped_committed_output_recovered",
            "queue_disposition": "terminal_success",
            "bucket": handoff.bucket,
            "scene_id": handoff.scene_id,
            "capture_id": handoff.capture_id,
            "capture_root": str(capture_root),
            "output_commit": recovered_commit,
            "job_ledger": completed_ledger,
        }
    control_plane_staging: dict[str, Any] | None = None
    reconstruction_enqueue: dict[str, Any] | None = None
    failure_stage = "stage_handoff_capture"
    try:
        with _JobLeaseHeartbeat(
            capture_root=capture_root,
            owner=owner,
            token=token,
            lease_seconds=lease_seconds,
        ):
            staged_capture_root = stage_handoff_capture(
                handoff=handoff,
                storage_root=storage_root,
                storage_client=storage_client,
            )
            run_kwargs: dict[str, Any] = {
                "capture_root": str(staged_capture_root),
                "provider": provider,
                "run_evaluation_prep": run_evaluation_prep,
                "resume_completed_stages": True,
            }
            robot_eval_job_request = _resolve_staged_handoff_path(
                handoff.robot_eval_job_request_uri,
                handoff=handoff,
                capture_root=staged_capture_root,
                storage_root=storage_root,
                expect_directory=False,
            )
            robot_eval_request_inbox = _resolve_staged_handoff_path(
                handoff.robot_eval_request_inbox_uri,
                handoff=handoff,
                capture_root=staged_capture_root,
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
                        "robot_eval_provisioner": handoff.robot_eval_provisioner
                        or "fixture_local",
                        "robot_eval_simulator": handoff.robot_eval_simulator
                        or "fixture",
                        "robot_eval_evaluation_substrate": handoff.robot_eval_evaluation_substrate,
                        "robot_eval_budget_usd": handoff.robot_eval_budget_usd,
                        "allow_robot_eval_gpu_provisioning": False,
                        "allow_robot_eval_simulator_execution": False,
                    }
                )
            if stage_control_plane:
                failure_stage = "control_plane_staging"
                if control_plane_manifest_path is None:
                    raise PipelineError(
                        "Pub/Sub handoff control-plane staging requires a manifest path."
                    )
                control_plane_staging = _stage_control_plane_input(
                    handoff=handoff,
                    capture_root=staged_capture_root,
                    manifest_path=control_plane_manifest_path,
                    work_dir=control_plane_work_dir,
                    staged_inputs_path=control_plane_staged_inputs_path,
                    overwrite=overwrite_control_plane_input,
                )
                failure_stage = "reconstruction_launch_enqueue"
                reconstruction_enqueue = _enqueue_capture_reconstruction_if_configured(
                    handoff=handoff,
                    capture_root=staged_capture_root,
                )
            failure_stage = "run_e2e"
            result = (
                run_e2e(**run_kwargs)
                if run_e2e_enabled
                else {
                    "status": "skipped",
                    "reason": "run_e2e_disabled_after_control_plane_staging",
                }
            )
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
        _finish_job_lease(
            capture_root,
            owner=owner,
            token=token,
            update={
                "status": "failed_retryable",
                "updated_at": failed_at,
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
    disposition, result_blockers = _handoff_result_disposition(result)
    terminal_success = disposition == "terminal_success"
    output_commit = (
        _write_output_commit(
            capture_root,
            scene_id=handoff.scene_id,
            capture_id=handoff.capture_id,
            attempt_count=attempt_count,
            result=result,
        )
        if terminal_success
        else None
    )
    completion_record = {
        "attempt_number": attempt_count,
        "status": "completed" if terminal_success else "retryable_blocked",
        "stage": "run_e2e",
        "started_at": attempt_started_at,
        "completed_at": completed_at,
        "run_e2e_status": str(result.get("status") or "") or None,
        "queue_disposition": disposition,
        "output_commit_status": "committed" if output_commit else None,
        "output_commit_path": JOB_OUTPUT_COMMIT_FILENAME if output_commit else None,
        "blockers": result_blockers,
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
        "status": "completed" if terminal_success else "retryable_blocked",
        "scene_id": handoff.scene_id,
        "capture_id": handoff.capture_id,
        "attempt_count": attempt_count,
        "started_at": job_started_at,
        "updated_at": completed_at,
        "last_attempt_started_at": attempt_started_at,
        "completed_at": completed_at,
        "run_e2e_status": str(result.get("status") or "") or None,
        "last_error_type": None if terminal_success else "RetryableBlockedResult",
        "last_error": None if terminal_success else ",".join(result_blockers),
        "retry_blockers": result_blockers,
        "queue_disposition": disposition,
        "output_commit_status": "committed" if output_commit else None,
        "output_commit_path": JOB_OUTPUT_COMMIT_FILENAME if output_commit else None,
        "output_result_sha256": output_commit.get("result_sha256")
        if output_commit
        else None,
        "attempt_history": [*previous_history, completion_record],
    }
    if control_plane_staging:
        ledger_update.update(
            {
                "control_plane_staging_status": control_plane_staging_status,
                "control_plane_staging_path": control_plane_staging_path,
            }
        )
    _finish_job_lease(
        capture_root,
        owner=owner,
        token=token,
        update=ledger_update,
    )
    return {
        "schema_version": "v1",
        "status": "processed" if terminal_success else "retryable_blocked",
        "queue_disposition": "terminal_success" if terminal_success else "retryable",
        "blockers": result_blockers,
        "bucket": handoff.bucket,
        "scene_id": handoff.scene_id,
        "capture_id": handoff.capture_id,
        "capture_root": str(capture_root),
        "run_e2e": result,
        "control_plane_staging": control_plane_staging,
        "reconstruction_enqueue": reconstruction_enqueue,
        "output_commit": output_commit,
    }


class _AckDeadlineHeartbeat:
    """Keep a synchronous pull message leased while its durable job lease runs."""

    def __init__(
        self,
        *,
        subscriber: Any,
        subscription: str,
        ack_id: str,
        ack_deadline_seconds: int,
    ) -> None:
        self.subscriber = subscriber
        self.subscription = subscription
        self.ack_id = ack_id
        self.ack_deadline_seconds = max(10, min(600, ack_deadline_seconds))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _modify(self, seconds: int) -> None:
        modify = getattr(self.subscriber, "modify_ack_deadline", None)
        if not callable(modify):
            return
        modify(
            request={
                "subscription": self.subscription,
                "ack_ids": [self.ack_id],
                "ack_deadline_seconds": seconds,
            }
        )

    def _run(self) -> None:
        interval = max(5.0, min(float(self.ack_deadline_seconds) / 2.0, 60.0))
        while not self._stop.wait(interval):
            try:
                self._modify(self.ack_deadline_seconds)
            except Exception:  # noqa: BLE001 - durable lease remains authoritative
                logger.exception("pubsub_handoff.ack_deadline_extension_failed")

    def __enter__(self) -> "_AckDeadlineHeartbeat":
        try:
            self._modify(self.ack_deadline_seconds)
        except Exception:  # noqa: BLE001 - durable lease prevents duplicate execution
            logger.exception("pubsub_handoff.initial_ack_deadline_extension_failed")
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def nack(self) -> None:
        try:
            self._modify(0)
        except Exception:  # noqa: BLE001 - leaving unacked still preserves retry semantics
            logger.exception("pubsub_handoff.explicit_nack_failed")


def _write_delivery_evidence(
    *,
    storage_root: Path,
    message: Any,
    received: Any,
    disposition: str,
    blockers: Sequence[str],
) -> Path:
    raw_data = bytes(message.data) if isinstance(message.data, bytes) else str(
        message.data
    ).encode("utf-8", errors="replace")
    digest = sha256(raw_data).hexdigest()
    message_id = _string(getattr(message, "message_id", None))
    record_path = (
        storage_root
        / ".pubsub_delivery_evidence"
        / disposition
        / f"{digest[:24]}.json"
    )
    write_json(
        record_path,
        {
            "schema_version": "pubsub_delivery_failure_evidence.v1",
            "generated_at": utc_now_iso(),
            "status": disposition,
            "queue_disposition": disposition,
            "message_id": message_id or None,
            "payload_sha256": digest,
            "payload_byte_count": len(raw_data),
            "raw_payload_stored": False,
            "delivery_attempt": getattr(received, "delivery_attempt", None),
            "blockers": list(blockers),
        },
    )
    return record_path


def _canonical_subscription_resource(subscription: str) -> str:
    value = _string(subscription)
    parts = value.split("/")
    if len(parts) == 4 and parts[0] == "projects" and parts[2] == "subscriptions":
        if parts[1] and parts[3]:
            return value
    if not value or "/" in value:
        raise PipelineError("Pub/Sub subscription must be a short id or full resource name")
    project = _string(os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT"))
    if not project:
        _credentials, default_project = google.auth.default()
        project = _string(default_project)
    if not project:
        raise PipelineError("Pub/Sub project id could not be resolved for short subscription id")
    return f"projects/{project}/subscriptions/{value}"


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
    ack_deadline_seconds: int = DEFAULT_ACK_DEADLINE_SECONDS,
    max_delivery_attempts: int = DEFAULT_MAX_DELIVERY_ATTEMPTS,
) -> int:
    from google.cloud import pubsub_v1

    subscriber = pubsub_v1.SubscriberClient()
    subscription_resource = _canonical_subscription_resource(subscription)
    response = subscriber.pull(
        request={
            "subscription": subscription_resource,
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
        # Contract-invalid payloads are permanent and can be acknowledged after
        # typed logging. Retryable work is explicitly nacked; the subscription's
        # configured dead-letter policy owns exhausted delivery routing.
        try:
            parse_handoff_payload(message.data)
        except PipelineError as exc:
            evidence_path = _write_delivery_evidence(
                storage_root=storage_root,
                message=message,
                received=received,
                disposition="permanent_invalid",
                blockers=[str(exc)],
            )
            logger.error(
                "pubsub_handoff.permanent_invalid",
                extra={
                    "message_id": message.message_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "queue_disposition": "permanent_invalid_ack",
                    "failure_evidence_path": str(evidence_path),
                },
            )
            ack_ids.append(received.ack_id)
            continue
        heartbeat = _AckDeadlineHeartbeat(
            subscriber=subscriber,
            subscription=subscription_resource,
            ack_id=received.ack_id,
            ack_deadline_seconds=ack_deadline_seconds,
        )
        try:
            with heartbeat:
                result = process_handoff_payload(
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
            delivery_attempt = getattr(received, "delivery_attempt", None)
            if isinstance(delivery_attempt, int) and delivery_attempt >= max_delivery_attempts:
                _write_delivery_evidence(
                    storage_root=storage_root,
                    message=message,
                    received=received,
                    disposition="retry_exhausted_pending_pubsub_dlq",
                    blockers=["handoff_processing_exception"],
                )
            logger.exception(
                "pubsub_handoff.processing_failed",
                extra={
                    "message_id": message.message_id,
                    "queue_disposition": "retryable_nack",
                    "delivery_attempt": getattr(received, "delivery_attempt", None),
                    "max_delivery_attempts": max_delivery_attempts,
                },
            )
            heartbeat.nack()
            continue
        if result.get("queue_disposition") == "retryable" or result.get(
            "status"
        ) in _JOB_RETRYABLE_STATUSES:
            delivery_attempt = getattr(received, "delivery_attempt", None)
            if isinstance(delivery_attempt, int) and delivery_attempt >= max_delivery_attempts:
                _write_delivery_evidence(
                    storage_root=storage_root,
                    message=message,
                    received=received,
                    disposition="retry_exhausted_pending_pubsub_dlq",
                    blockers=[str(item) for item in result.get("blockers") or []],
                )
            logger.warning(
                "pubsub_handoff.retryable_result",
                extra={
                    "message_id": message.message_id,
                    "status": result.get("status"),
                    "blockers": result.get("blockers") or [],
                    "delivery_attempt": getattr(received, "delivery_attempt", None),
                    "max_delivery_attempts": max_delivery_attempts,
                    "dead_letter_policy_owns_exhausted_delivery": True,
                },
            )
            heartbeat.nack()
            continue
        ack_ids.append(received.ack_id)

    if ack_ids:
        subscriber.acknowledge(request={"subscription": subscription_resource, "ack_ids": ack_ids})
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
        prefix = f"gs://{handoff.bucket}/{handoff.capture_prefix}/"
        if not value.startswith(prefix):
            raise PipelineError(
                "Pub/Sub handoff robot eval path must remain in the staged capture prefix."
            )
        relative = PurePosixPath(value[len(f"gs://{handoff.bucket}/") :])
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise PipelineError("Pub/Sub handoff robot eval path is unsafe.")
        local_path = contained_path(
            storage_root / handoff.bucket,
            *relative.parts,
            field="staged robot eval path",
        )
    else:
        path = Path(value)
        if path.is_absolute():
            raise PipelineError("Pub/Sub handoff robot eval path may not be absolute.")
        local_path = contained_path(
            capture_root,
            *path.parts,
            field="staged robot eval path",
        )
    try:
        local_path = prove_path_contained(
            capture_root,
            local_path,
            field="staged robot eval path",
        )
    except SecurityValidationError as exc:
        raise PipelineError(str(exc)) from exc
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
    acknowledged = pull_and_process(
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
    print(
        json.dumps(
            {
                "acknowledged": acknowledged,
                "acknowledged_means_terminal_success_or_permanent_invalid": True,
                "storage_root": str(storage_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
