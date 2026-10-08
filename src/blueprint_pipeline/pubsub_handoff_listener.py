"""Pull BlueprintCapture bridge handoffs from Pub/Sub and run the pipeline."""

from __future__ import annotations
from .task_evaluation_scene_retirement_access import scene_participant

import argparse
import fcntl
import json
import logging
import os
import re
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
from .pubsub_handoff_disk_admission import (  # noqa: F401 - native bodies resolve this live namespace
    HandoffStagingCapacityError,
    download_with_reservation,
    finish_staging_capacity_blocked,
    staging_manifest_row as _staging_manifest_row,
)
from .decision_evidence_contracts import canonical_digest
from .run_e2e import run_end_to_end
from .handoff_job_state import (  # noqa: F401 - compatibility exports and shared durability primitives
    JOB_LEDGER_FILENAME,
    JOB_LEDGER_SCHEMA_VERSION,
    JOB_OUTPUT_COMMIT_FILENAME,
    JOB_OUTPUT_COMMIT_SCHEMA_VERSION,
    _existing_job_ledger_lock,
    _output_commit,
    _read_job_ledger,
    _retained_required_stage_blockers,
    required_stage_result_blocker,
)
from .core.security_controls import (
    SecurityValidationError,
    contained_path,
    prove_path_contained,
    strict_gcs_bucket,
    strict_identifier,
)
from .website_capture_entry import is_website_capture_manifest

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
    source_finalize: Mapping[str, str] | None = None
    source_membership_selector: Mapping[str, Any] | None = None

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
    source_finalize = data.get("source_finalize")
    if source_finalize is not None:
        expected_marker = f"scenes/{scene_id}/captures/{capture_id}/raw/capture_upload_complete.json"
        if (type(source_finalize) is not dict
                or set(source_finalize) != {"bucket", "object_name", "generation", "event_id", "event_source"}
                or any(type(value) is not str for value in source_finalize.values())
                or source_finalize["bucket"] != bucket
                or source_finalize["object_name"] != expected_marker
                or re.fullmatch(r"[1-9][0-9]{0,19}", source_finalize["generation"]) is None
                or not 0 < len(source_finalize["event_id"].encode("utf-8")) <= 256
                or not 0 < len(source_finalize["event_source"].encode("utf-8")) <= 1024
                or "\x00" in source_finalize["event_id"]
                or "\x00" in source_finalize["event_source"]):
            raise PipelineError("Pub/Sub handoff source finalize identity invalid.")

    source_membership_selector = data.get("source_membership_selector")
    if source_membership_selector is not None:
        if source_finalize is None or type(source_membership_selector) is not dict or set(source_membership_selector) != {
            "object_name", "generation", "size_bytes", "sha256"
        }:
            raise PipelineError("Pub/Sub handoff source membership selector invalid.")
        import hashlib
        delivery_key = hashlib.sha256(json.dumps([
            bucket, source_finalize["object_name"], source_finalize["generation"]
        ], separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        expected_member = (f"scenes/{scene_id}/captures/{capture_id}/deliveries/"
                           f"{delivery_key}/capture_delivery_membership.json")
        if (source_membership_selector["object_name"] != expected_member
                or type(source_membership_selector["generation"]) is not str
                or re.fullmatch(r"[1-9][0-9]{0,19}", source_membership_selector["generation"]) is None
                or type(source_membership_selector["size_bytes"]) is not int
                or not 0 < source_membership_selector["size_bytes"] <= 65536
                or type(source_membership_selector["sha256"]) is not str
                or re.fullmatch(r"sha256:[0-9a-f]{64}", source_membership_selector["sha256"]) is None):
            raise PipelineError("Pub/Sub handoff source membership selector invalid.")

    # Selected deliveries use the immutable handoff beside their exact validated
    # membership record. A URI alone never selects a delivery or another owner.
    selected_handoff_uri = (
        f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/deliveries/"
        f"{delivery_key}/pipeline_handoff.json"
        if source_membership_selector is not None else None
    )
    if pipeline_handoff_uri is not None and pipeline_handoff_uri not in (
        expected_pipeline_handoff_uri, selected_handoff_uri,
    ):
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
        source_finalize=source_finalize,
        source_membership_selector=source_membership_selector,
    )


@scene_participant('storage_root')
def stage_handoff_capture(handoff: HandoffMessage, *, storage_root: Path, storage_client: storage.Client | None=None) -> Path:
    import sys
    from .pubsub_handoff_scene_operations import _stage_handoff_capture_body
    return _stage_handoff_capture_body(sys.modules[__name__], handoff, storage_root=storage_root, storage_client=storage_client)


def _previous_staging_rows(capture_root: Path, *, handoff: HandoffMessage) -> dict[str, dict[str, Any]]:
    """Rows of this capture's last staging manifest, keyed by object name.

    A manifest for another bucket or prefix, or one that is unreadable, proves
    nothing, so every object downloads again.
    """

    manifest = _read_optional_json_object(capture_root / STAGING_MANIFEST_FILENAME)
    objects = manifest.get("objects")
    if (
        manifest.get("schema_version") != STAGING_MANIFEST_SCHEMA_VERSION
        or manifest.get("bucket") != handoff.bucket
        or manifest.get("prefix") != f"{handoff.capture_prefix}/"
        or not isinstance(objects, list)
    ):
        return {}
    return {
        row["name"]: dict(row)
        for row in objects
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    }


def _staged_copy_is_current(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    destination: Path,
) -> bool:
    """Whether the object is unchanged since it was staged and its local copy is intact.

    Unchanged means the listing still reports the generation and size recorded
    at the last successful staging. An object whose generation or size is
    unknown is never assumed unchanged.

    The local copy counts as intact when it is a regular file of that same
    size; its bytes are not hashed. A local edit that keeps the size is
    therefore not undone by restaging. Staging's consumers catch that case:
    run_e2e always reruns materialization, which rebuilds the descriptor and
    QA projections, and the raw verifier checks raw bytes against
    raw/hashes.json.
    """

    generation = current.get("generation")
    size = current.get("size")
    if generation is None or size is None:
        return False
    previous_size = previous.get("size")
    if (
        previous.get("generation") != generation
        or not isinstance(previous_size, int)
        or isinstance(previous_size, bool)
        or previous_size != size
    ):
        return False
    try:
        return destination.is_file() and destination.stat().st_size == size
    except OSError:
        return False


def _preserve_local_website_derivatives(capture_root: Path, uploaded_names: set[str], prefix: str) -> None:
    """Recover the legacy SAM helper's local-only frames without relaxing raw checks.

    GCS is the completed upload authority. Never move an uploaded member or a
    symlink; retain displaced local artifacts and their hashes under pipeline/.
    Called only by staging under the existing job lease.
    """
    if not is_website_capture_manifest(_read_optional_json_object(capture_root / "raw/manifest.json")):
        return
    source = capture_root / "raw/object_index_artifacts"
    if not source.exists() and not source.is_symlink():
        return
    cloud_prefix = f"{prefix}/raw/object_index_artifacts"
    if any(name == cloud_prefix or name.startswith(cloud_prefix + "/") for name in uploaded_names):
        return  # Canonical upload violations still fail the raw verifier.
    if source.is_symlink() or not source.is_dir() or any(p.is_symlink() for p in source.rglob("*")):
        raise PipelineError("website_local_derivative_recovery_unsafe")
    members = {p.relative_to(source).as_posix(): sha256(p.read_bytes()).hexdigest()
               for p in sorted(source.rglob("*")) if p.is_file()}
    archive = contained_path(capture_root, "pipeline", "recovered_raw_derivatives",
                             field="website derivative recovery archive")
    archive.mkdir(parents=True, exist_ok=True)
    recovery = Path(tempfile.mkdtemp(prefix="sam3-", dir=archive))
    write_json(recovery / "receipt.json", {
        "schema_version": "website_local_derivative_recovery.v1",
        "source": "raw/object_index_artifacts", "reason": "legacy_sam_helper_local_outputs",
        "not_present_in_completed_upload": True, "member_sha256": members,
        "claim_ceiling": "development_only", "recovered_at": utc_now_iso(),
    })
    source.rename(recovery / "object_index_artifacts")


def _read_optional_json_object(path: Path) -> dict[str, Any]:
    """The JSON object at path, or {} when it is missing or unreadable.

    ValueError covers both invalid JSON and bytes that are not UTF-8.
    """

    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _set_aside(path: Path, label: str) -> Path:
    """Rename a record to <stem>.<label>-<UTC compact time><suffix>, keeping its bytes.

    Never overwrites: an existing name gets a numeric suffix. Callers hold the
    capture's ledger lock.
    """

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = path.with_name(f"{path.stem}.{label}-{stamp}{path.suffix}")
    counter = 1
    while target.exists() or target.is_symlink():
        target = path.with_name(f"{path.stem}.{label}-{stamp}-{counter}{path.suffix}")
        counter += 1
    path.rename(target)
    return target


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
    import sys
    from .pubsub_handoff_scene_operations import __synthesize_pipeline_handoff_body
    return __synthesize_pipeline_handoff_body(sys.modules[__name__], handoff, capture_root=capture_root)


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
    from .task_evaluation_scene_retirement_generations import capture_birth_input_path

    descriptor_path = capture_birth_input_path(capture_root, "capture_descriptor.json")
    capture_descriptor_uri = str(descriptor_path) if descriptor_path.is_file() else None
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


JOB_STATUS_SCHEMA_VERSION = "pipeline_job_status.v1"
TERMINAL_AUTHORITY_STATUS = "terminal_authority_ended"
JOB_TERMINAL_RECEIPT_FILENAME = "pipeline_job_terminal_receipt.json"
JOB_TERMINAL_RECEIPT_SCHEMA_VERSION = "pipeline_job_terminal_receipt.v1"
JOB_ACK_RECEIPT_FILENAME = "pipeline_job_ack_receipt.json"
JOB_ACK_RECEIPT_SCHEMA_VERSION = "pubsub_handoff_ack_receipt.v1"
# The only outcomes an ack receipt may record; retirement matches them to the ledger.
_ACK_RECEIPT_DISPOSITIONS = frozenset({"terminal_success", TERMINAL_AUTHORITY_STATUS})
STAGING_MANIFEST_FILENAME = "pipeline_staging_manifest.json"
STAGING_MANIFEST_SCHEMA_VERSION = "pipeline_handoff_staging_manifest.v1"
PROVIDER_OPS_STATUS_SCHEMA_VERSION = "provider_ops_status.v1"
DEFAULT_JOB_LEASE_SECONDS = 900
DEFAULT_ACK_DEADLINE_SECONDS = 600
DEFAULT_MAX_DELIVERY_ATTEMPTS = 5
RETRY_DEFER_SECONDS = 600
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
# The WebApp ends a website scene's authority with a typed 409 refusal, which
# website_task_context raises as ValueError("website_control_<op>_http_409:<code>").
# Retrying the same handoff cannot revive an expired consent or a revoked
# source. Other 409 codes (task_brief_missing, idempotency_conflict, ...) can be
# fixed by a person, so they stay retryable.
AUTHORITY_ENDING_CODES = frozenset({"consent_expired", "source_revoked"})
# The same token boundary on both sides: no identifier character or hyphen may
# touch the typed refusal, so "...:source_revokedX" is not "source_revoked".
# Group 1 is the refusing operation (e.g. scene-sponsorship), group 2 the code.
_AUTHORITY_ENDING_RE = re.compile(
    r"(?<![A-Za-z0-9_-])website_control_([a-z0-9-]+)_http_409:("
    + "|".join(re.escape(code) for code in sorted(AUTHORITY_ENDING_CODES))
    + r")(?![A-Za-z0-9_-])"
)
_AUTHORITY_ENDING_CHAIN_LIMIT = 16


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
                os.getenv(RECONSTRUCTION_SOURCE_COMMIT_ENV)
                or os.getenv("BLUEPRINT_SOURCE_COMMIT")
                or ""
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


class HandoffCaptureRetired(PipelineError):
    """The capture's workspace was retired (or removed) while this process reached for its lock."""


def _lock_names_capture(lock_file: Any, lock_path: Path, capture_root: Path) -> bool:
    """Whether the lock file held open is still the one the capture's path names."""

    try:
        held = os.fstat(lock_file.fileno())
        named = os.stat(lock_path)
    except OSError:
        return False
    return (held.st_dev, held.st_ino) == (named.st_dev, named.st_ino) and capture_root.is_dir()


@contextmanager
def _locked_job_ledger(capture_root: Path, *, create: bool = True) -> Iterator[dict[str, Any]]:
    """Hold the per-capture ledger lock while reading or committing state.

    ``flock`` supplies cross-process exclusion. ``write_json`` supplies the
    same-filesystem temp/fsync/replace commit, so a killed writer leaves either
    the prior complete ledger or the complete next revision. Scene workspace
    retirement takes this same lock and removes the workspace under it, so a
    lock acquired after that no longer names the capture: that raises
    ``HandoffCaptureRetired`` rather than recreating a retired workspace, as
    does ``create=False`` when the capture is already gone.
    """

    if create:
        capture_root.mkdir(parents=True, exist_ok=True)
    lock_path = capture_root / f".{JOB_LEDGER_FILENAME}.lock"
    try:
        lock_file = lock_path.open("a+b")
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HandoffCaptureRetired("pubsub_handoff_capture_retired") from exc
    with lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if not _lock_names_capture(lock_file, lock_path, capture_root):
                raise HandoffCaptureRetired("pubsub_handoff_capture_retired")
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
    payload_sha256: str | None = None,
    producer_delivery_key: str | None = None,
    create_capture_root: bool = True,
    retired_ended_payload_sha256s: Sequence[str] = (),
    retired_ended_producer_delivery_keys: Sequence[str] = (),
) -> tuple[str, dict[str, Any]]:
    if producer_delivery_key is not None and re.fullmatch(r"sha256:[0-9a-f]{64}", producer_delivery_key) is None:
        raise PipelineError('capture_delivery_key_invalid')
    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    with _locked_job_ledger(capture_root, create=create_capture_root) as ledger:
        revision = int(ledger.get("revision") or 0)
        status = _string(ledger.get("status"))
        if status == "corrupt":
            return "corrupt", dict(ledger)
        if producer_delivery_key is not None and ledger and ledger.get('producer_delivery_key') != producer_delivery_key:
            return 'source_conflict', dict(ledger)
        history = _attempt_history(ledger)
        if not ledger and retired_ended_payload_sha256s:
            history.extend({"status": TERMINAL_AUTHORITY_STATUS, "payload_sha256": digest,
                            "source": "scene_retirement_receipt"}
                           for digest in sorted(set(retired_ended_payload_sha256s))
                           if re.fullmatch(r"[0-9a-f]{64}", digest))
        if not ledger and retired_ended_producer_delivery_keys:
            history.extend({'status': TERMINAL_AUTHORITY_STATUS,
                            'producer_delivery_key': key,
                            'source': 'scene_retirement_receipt'}
                           for key in sorted(set(retired_ended_producer_delivery_keys))
                           if re.fullmatch(r"sha256:[0-9a-f]{64}", key))
        if producer_delivery_key is not None and producer_delivery_key in (
                _ended_delivery_keys(ledger) | set(retired_ended_producer_delivery_keys)):
            return 'terminal', dict(ledger)
        # A payload whose run ended for lost authority never runs again, even
        # while a later payload reopened the job and is running or retrying.
        # Only a completed job answers a redelivery from its output commit.
        if (
            status != "completed"
            and payload_sha256
            and payload_sha256 in _ended_payload_digests(ledger)
        ):
            return "terminal", dict(ledger)
        if status == TERMINAL_AUTHORITY_STATUS:
            if producer_delivery_key is not None:
                return 'terminal', dict(ledger)
            ended_by = _string(ledger.get("terminal_payload_sha256"))
            # Without both digests nothing proves this is a new request, so the
            # ending stands (a redelivery must not re-run an ended scene).
            if not payload_sha256 or not ended_by:
                return "terminal", dict(ledger)
            # A different message is a new request for this capture, for example
            # after the website renewed consent. Keep the ending in the history.
            history.append({
                "attempt_number": int(ledger.get("attempt_count") or 0) + 1,
                "status": "reopened_after_terminal_authority",
                "reopened_at": _iso_at(current_time),
                "terminal_code": ledger.get("terminal_code"),
                "terminal_payload_sha256": ended_by,
                "payload_sha256": payload_sha256,
            })
            # The ended run's receipt is kept, but never under the live name,
            # which only ever describes the ledger's current ending.
            live_receipt = capture_root / JOB_TERMINAL_RECEIPT_FILENAME
            if live_receipt.exists() or live_receipt.is_symlink():
                _set_aside(live_receipt, "superseded")
        if status == "completed":
            if not _retained_required_stage_blockers(capture_root):
                return "completed", dict(ledger)
            # Older acknowledgements ignored required-stage result failures.
            # Preserve the receipt; only the incomplete stage loses resume eligibility.
            old_commit = _read_optional_json_object(capture_root / JOB_OUTPUT_COMMIT_FILENAME)
            if old_commit:
                write_json(capture_root / JOB_OUTPUT_COMMIT_FILENAME, {
                    **old_commit, "status": "superseded_failed_lanes", "superseded_at": _iso_at(current_time)})
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
                "attempt_history": history,
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
                **({'producer_delivery_key': producer_delivery_key,
                    'source_payload_sha256': payload_sha256}
                   if producer_delivery_key is not None else {}),
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
    after_commit: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Commit the lease's final ledger; after_commit runs under the same lock.

    after_commit runs after the commit. If it raises, the commit stands (the
    ledger is already durable) and the exception propagates to the caller.
    The terminal path relies on this: a terminal receipt that failed to write
    leaves the message unacknowledged, and the redelivery repairs the receipt
    from the committed ledger.
    """

    with _locked_job_ledger(capture_root) as ledger:
        if (
            ledger.get("status") != "processing"
            or _string(ledger.get("lease_owner")) != owner
            or _string(ledger.get("lease_token")) != token
        ):
            raise PipelineError("Pub/Sub job ledger lease ownership was lost before commit.")
        revision = int(ledger.get("revision") or 0)
        committed = _commit_job_ledger(
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
        if after_commit is not None:
            after_commit(committed)
        return committed


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


def _ended_payload_digests(ledger: Mapping[str, Any]) -> set[str]:
    """Every payload digest whose run this capture ended for lost authority."""

    digests = {_string(ledger.get("terminal_payload_sha256"))}
    for row in _attempt_history(ledger):
        if row.get("status") == TERMINAL_AUTHORITY_STATUS:
            digests.add(_string(row.get("payload_sha256")))
        elif row.get("status") == "reopened_after_terminal_authority":
            digests.add(_string(row.get("terminal_payload_sha256")))
    digests.discard("")
    return digests


def _ended_delivery_keys(ledger: Mapping[str, Any]) -> set[str]:
    keys = {_string(ledger.get('terminal_producer_delivery_key'))}
    for row in _attempt_history(ledger):
        if row.get('status') == TERMINAL_AUTHORITY_STATUS:
            keys.add(_string(row.get('producer_delivery_key')))
        elif row.get('status') == 'reopened_after_terminal_authority':
            keys.add(_string(row.get('terminal_producer_delivery_key')))
    keys.discard('')
    return keys


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
    # Present means an intact receipt of the ledger's current ending, not just a file.
    terminal_receipt_present = _terminal_receipt_current(
        _read_optional_json_object(capture_root / JOB_TERMINAL_RECEIPT_FILENAME), ledger
    )
    ack_receipt = _read_optional_json_object(capture_root / JOB_ACK_RECEIPT_FILENAME) or None
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
        "terminal_code": ledger.get("terminal_code") if ledger else None,
        "terminal_operation": ledger.get("terminal_operation") if ledger else None,
        "terminal_receipt_present": terminal_receipt_present,
        "ack_receipt": ack_receipt,
        # An authority ending is terminal: a redelivery is acknowledged, not retried.
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

    if "task_evaluation_supervisor" in result:
        blocker = required_stage_result_blocker(
            "task_evaluation_supervisor", result["task_evaluation_supervisor"]
        )
        if blocker:
            return "retryable_blocked", [blocker]
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
        or status == "completed_with_lane_failures"
    ]
    if retryable:
        return "retryable_blocked", retryable
    if "pipeline_status" in result:
        blocker = required_stage_result_blocker(
            "capture_pipeline", {"status": result["pipeline_status"]}
        )
        if blocker:
            return "retryable_blocked", [blocker]
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


def authority_ending(exc: BaseException) -> tuple[str, str] | None:
    """(refusing operation, code) of the WebApp refusal that ended this job, if any.

    Walks the exception chain (__cause__, then __context__), at most 16 links, and
    guards against cycles. Only exact typed WebApp 409 codes qualify. The token may
    sit inside a longer message, because a StageError joins its blockers, but it
    must be whole.
    """

    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(_AUTHORITY_ENDING_CHAIN_LIMIT):
        if current is None or id(current) in seen:
            return None
        seen.add(id(current))
        try:
            text = str(current)
        except Exception:  # noqa: BLE001 - an unprintable error carries no typed code
            text = ""
        match = _AUTHORITY_ENDING_RE.search(text)
        if match:
            return match.group(1), match.group(2)
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return None


def authority_ending_code(exc: BaseException) -> str | None:
    """The WebApp authority code that permanently ended this job, if any."""

    ending = authority_ending(exc)
    return ending[1] if ending else None


def payload_sha256(payload: bytes | str | Mapping[str, Any]) -> str:
    """Hex sha256 of the message bytes, as `_write_delivery_evidence` already records it.

    str -> utf-8 bytes; Mapping -> json.dumps(sort_keys=True, separators=(",", ":")).
    """

    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8", errors="replace")
    else:
        raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()


def _write_terminal_receipt(
    capture_root: Path,
    *,
    handoff: HandoffMessage,
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Record, from the committed terminal ledger, why this job ended."""

    receipt: dict[str, Any] = {
        "schema_version": JOB_TERMINAL_RECEIPT_SCHEMA_VERSION,
        "status": "authority_ended",
        "code": ledger.get("terminal_code"),
        "terminal_operation": ledger.get("terminal_operation"),
        "bucket": handoff.bucket,
        "scene_id": handoff.scene_id,
        "capture_id": handoff.capture_id,
        "attempt_count": int(ledger.get("attempt_count") or 0),
        "payload_sha256": ledger.get("terminal_payload_sha256"),
        "ended_at": ledger.get("terminal_at"),
        "error": str(ledger.get("last_error") or "")[:500],
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(capture_root / JOB_TERMINAL_RECEIPT_FILENAME, receipt)
    return receipt


def _terminal_receipt_current(receipt: Mapping[str, Any], ledger: Mapping[str, Any]) -> bool:
    """Whether receipt is an intact record of the ledger's current authority ending."""

    ended_by = _string(ledger.get("terminal_payload_sha256"))
    return bool(
        _string(ledger.get("status")) == TERMINAL_AUTHORITY_STATUS
        and ended_by
        and receipt.get("schema_version") == JOB_TERMINAL_RECEIPT_SCHEMA_VERSION
        and receipt.get("status") == "authority_ended"
        and receipt.get("payload_sha256") == ended_by
        and receipt.get("code") == ledger.get("terminal_code")
        and receipt.get("receipt_digest") == canonical_digest(receipt, digest_field="receipt_digest")
    )


def _replace_terminal_receipt(
    capture_root: Path,
    *,
    handoff: HandoffMessage,
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the ledger's receipt, first setting aside whatever held the live name.

    Callers hold the capture's ledger lock.
    """

    live = capture_root / JOB_TERMINAL_RECEIPT_FILENAME
    if live.exists() or live.is_symlink():
        _set_aside(live, "superseded")
    return _write_terminal_receipt(capture_root, handoff=handoff, ledger=ledger)


def _repair_terminal_receipt(capture_root: Path, *, handoff: HandoffMessage) -> None:
    """Under the ledger lock, make the live receipt describe the current ending.

    Covers a process that died between the terminal ledger commit and its
    receipt, and a receipt left behind by an earlier ending. Takes the lock
    without creating anything, so a capture retired after the claim stays
    retired.
    """

    with _existing_job_ledger_lock(capture_root) as state:
        if state != "ledger_present":
            return
        ledger = _read_job_ledger(capture_root)
        if (
            _string(ledger.get("status")) != TERMINAL_AUTHORITY_STATUS
            or not _string(ledger.get("terminal_code"))
            or not _string(ledger.get("terminal_payload_sha256"))
        ):
            return
        receipt = _read_optional_json_object(capture_root / JOB_TERMINAL_RECEIPT_FILENAME)
        if not _terminal_receipt_current(receipt, ledger):
            _replace_terminal_receipt(capture_root, handoff=handoff, ledger=ledger)


def _terminal_authority_result(
    handoff: HandoffMessage,
    *,
    capture_root: Path,
    ledger: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": "v1",
        "status": status,
        "queue_disposition": TERMINAL_AUTHORITY_STATUS,
        "bucket": handoff.bucket,
        "scene_id": handoff.scene_id,
        "capture_id": handoff.capture_id,
        "capture_root": str(capture_root),
        "blockers": [_string(ledger.get("terminal_code")) or TERMINAL_AUTHORITY_STATUS],
        "job_ledger": dict(ledger),
    }


def _finish_terminal_authority_ending(
    capture_root: Path,
    *,
    handoff: HandoffMessage,
    owner: str,
    token: str,
    operation: str,
    code: str,
    error: BaseException,
    stage: str,
    attempt_count: int,
    attempt_started_at: str,
    previous_history: Sequence[Mapping[str, Any]],
    payload_digest: str,
    producer_delivery_key: str | None = None,
) -> dict[str, Any]:
    """End the job for good: the website ended this scene's authority.

    The ledger commits first and the receipt follows under the same lock. A
    crash before the receipt is written leaves a terminal ledger, and the next
    redelivery writes the receipt from it.
    """

    ended_at = utc_now_iso()
    ledger = _finish_job_lease(
        capture_root,
        owner=owner,
        token=token,
        after_commit=lambda committed: _replace_terminal_receipt(
            capture_root, handoff=handoff, ledger=committed
        ),
        update={
            "status": TERMINAL_AUTHORITY_STATUS,
            "terminal_code": code,
            "terminal_operation": operation,
            "terminal_at": ended_at,
            "updated_at": ended_at,
            "terminal_payload_sha256": payload_digest,
            **({'terminal_producer_delivery_key': producer_delivery_key}
               if producer_delivery_key is not None else {}),
            "last_error_type": type(error).__name__,
            "last_error": str(error)[:500],
            "queue_disposition": TERMINAL_AUTHORITY_STATUS,
            "attempt_history": [
                *previous_history,
                {
                    "attempt_number": attempt_count,
                    "status": TERMINAL_AUTHORITY_STATUS,
                    "stage": stage,
                    "started_at": attempt_started_at,
                    "ended_at": ended_at,
                    "code": code,
                    "operation": operation,
                    "payload_sha256": payload_digest,
                    **({'producer_delivery_key': producer_delivery_key}
                       if producer_delivery_key is not None else {}),
                },
            ],
        },
    )
    logger.warning(
        "pubsub_handoff.terminal_authority_ended",
        extra={
            "scene_id": handoff.scene_id,
            "capture_id": handoff.capture_id,
            "terminal_code": code,
            "stage": stage,
        },
    )
    return _terminal_authority_result(
        handoff,
        capture_root=capture_root,
        ledger=ledger,
        status=TERMINAL_AUTHORITY_STATUS,
    )


@scene_participant('storage_root')
def process_handoff_payload(payload: bytes | str | Mapping[str, Any], *, storage_root: Path, provider: str, run_e2e: Callable[..., dict[str, Any]]=run_end_to_end, storage_client: storage.Client | None=None, run_evaluation_prep: bool=True, run_e2e_enabled: bool=True, stage_control_plane: bool=False, control_plane_manifest_path: str | Path | None=None, control_plane_work_dir: str | Path | None=None, control_plane_staged_inputs_path: str | Path | None=None, overwrite_control_plane_input: bool=False, lease_owner: str | None=None, lease_seconds: int=DEFAULT_JOB_LEASE_SECONDS, payload_digest: str | None=None) -> dict[str, Any]:
    import sys
    from .pubsub_handoff_scene_operations import _process_handoff_payload_body
    try:
        return _process_handoff_payload_body(sys.modules[__name__], payload, storage_root=storage_root, provider=provider, run_e2e=run_e2e, storage_client=storage_client, run_evaluation_prep=run_evaluation_prep, run_e2e_enabled=run_e2e_enabled, stage_control_plane=stage_control_plane, control_plane_manifest_path=control_plane_manifest_path, control_plane_work_dir=control_plane_work_dir, control_plane_staged_inputs_path=control_plane_staged_inputs_path, overwrite_control_plane_input=overwrite_control_plane_input, lease_owner=lease_owner, lease_seconds=lease_seconds, payload_digest=payload_digest)
    finally:
        # Separate durable wake-up only; never let delivery reopen or fail processing.
        try:
            from .website_preparation_status import retain_preparation_wakeup
            handoff = parse_handoff_payload(payload)
            root = _handoff_capture_root(handoff, storage_root=storage_root)
            if root.is_dir():
                retain_preparation_wakeup(root)
        except Exception:
            logger.debug("pubsub_handoff.preparation_wakeup_unavailable")


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

    def defer_retry(self) -> None:
        """Keep blocked work unacked while giving other handoffs a pull window.

        A zero-second nack makes an expensive blocked scene immediately eligible
        again. One bounded Pub/Sub ack lease leaves it retryable without letting
        that same message dominate the next timer invocation. Pub/Sub redelivers
        it after the lease expires (or routes it to the configured dead letter
        topic after its delivery limit). If the lease change fails, the message
        remains unacked and its existing lease still expires normally.
        """
        try:
            self._modify(RETRY_DEFER_SECONDS)
        except Exception:  # noqa: BLE001 - an unacked message still retries
            logger.exception("pubsub_handoff.retry_deferral_failed")


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


def _write_ack_receipt(
    *,
    capture_root: Path,
    subscription: str,
    message_id: str | None,
    payload_digest: str,
    delivery_attempt: int | None,
    disposition: str,
) -> str | None:
    """Replace the capture's ack receipt; call only after acknowledge returned.

    Returns None once written. Returns "capture_absent" or "ledger_absent",
    writing nothing, when the capture has no ledger: a workspace retired after
    its cloud copy was verified must stay retired.
    """

    with _existing_job_ledger_lock(capture_root) as state:
        if state != "ledger_present":
            return state
        path = capture_root / JOB_ACK_RECEIPT_FILENAME
        previous = _read_optional_json_object(path)
        if not previous and path.is_file():
            _set_aside(path, "unreadable")  # a receipt that cannot be read is kept, not overwritten
        previous_count = previous.get("acknowledgement_count")
        if not isinstance(previous_count, int) or isinstance(previous_count, bool) or previous_count < 0:
            previous_count = 0
        write_json(
            path,
            {
                "schema_version": JOB_ACK_RECEIPT_SCHEMA_VERSION,
                "subscription": subscription,
                "message_id": message_id,
                "payload_sha256": payload_digest,
                "delivery_attempt": delivery_attempt,
                "disposition": disposition,
                "acknowledged_at": utc_now_iso(),
                "acknowledgement_count": previous_count + 1,
            },
        )
    return None


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


@scene_participant('storage_root')
def pull_and_process(*, subscription: str, storage_root: Path, provider: str, max_messages: int, run_evaluation_prep: bool=True, run_e2e_enabled: bool=True, stage_control_plane: bool=False, control_plane_manifest_path: str | Path | None=None, control_plane_work_dir: str | Path | None=None, control_plane_staged_inputs_path: str | Path | None=None, overwrite_control_plane_input: bool=False, ack_deadline_seconds: int=DEFAULT_ACK_DEADLINE_SECONDS, max_delivery_attempts: int=DEFAULT_MAX_DELIVERY_ATTEMPTS) -> int:
    import sys
    from .pubsub_handoff_scene_operations import _pull_and_process_body
    return _pull_and_process_body(sys.modules[__name__], subscription=subscription, storage_root=storage_root, provider=provider, max_messages=max_messages, run_evaluation_prep=run_evaluation_prep, run_e2e_enabled=run_e2e_enabled, stage_control_plane=stage_control_plane, control_plane_manifest_path=control_plane_manifest_path, control_plane_work_dir=control_plane_work_dir, control_plane_staged_inputs_path=control_plane_staged_inputs_path, overwrite_control_plane_input=overwrite_control_plane_input, ack_deadline_seconds=ack_deadline_seconds, max_delivery_attempts=max_delivery_attempts)


def _record_acknowledgement(
    result: Mapping[str, Any],
    *,
    subscription: str,
    message: Any,
    received: Any,
    payload_digest: str,
) -> None:
    """Write the capture's ack receipt; call only after acknowledge returned.

    A receipt never claims an ack that Pub/Sub did not accept. The ack is
    already final, so a receipt that cannot be written is logged, never raised.
    A capture with no ledger (its workspace was retired) gets no receipt.
    """

    capture_root = _string(result.get("capture_root"))
    if not capture_root:
        return
    message_id = _string(getattr(message, "message_id", None)) or None
    disposition = _string(result.get("queue_disposition"))
    if disposition not in _ACK_RECEIPT_DISPOSITIONS:
        logger.warning(
            "pubsub_handoff.ack_receipt_skipped_disposition_unrecognized",
            extra={"message_id": message_id, "queue_disposition": disposition or None},
        )
        return
    delivery_attempt = getattr(received, "delivery_attempt", None)
    try:
        skipped = _write_ack_receipt(
            capture_root=Path(capture_root),
            subscription=subscription,
            message_id=message_id,
            payload_digest=payload_digest,
            delivery_attempt=delivery_attempt
            if isinstance(delivery_attempt, int) and not isinstance(delivery_attempt, bool)
            else None,
            disposition=disposition,
        )
    except (OSError, ValueError):
        logger.exception("pubsub_handoff.ack_receipt_write_failed", extra={"message_id": message_id})
        return
    if skipped == "capture_absent":
        logger.warning("pubsub_handoff.ack_receipt_skipped_capture_absent", extra={"message_id": message_id})
    elif skipped == "ledger_absent":
        logger.warning("pubsub_handoff.ack_receipt_skipped_ledger_absent", extra={"message_id": message_id})


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
                "acknowledged_means_terminal_or_permanent_invalid": True,
                "storage_root": str(storage_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
