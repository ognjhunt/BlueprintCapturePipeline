"""One-shot Vast execution for an immutable semantic-teacher editor bundle.

This module intentionally has no CLI.  The only production caller is the
shared ``paid_resource_allocator gpu-canary`` branch, which supplies the paid
resource admission grant and an already-armed independent watchdog.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any
import urllib.error
import zipfile

from .common import ensure_dir, redacted_failure_detail, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .gpu_render_providers import GpuRenderProvider, RenderLaunchSpec
from .paid_lane_guard import (
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    close_pending_teardown,
    load_pending_teardowns,
    mark_pending_teardown_ambiguous,
    open_pending_teardown,
)
from .paid_provider_lane_lease import (
    acquire_paid_provider_lane_lease,
    build_paid_provider_lane_reconciliation,
    release_paid_provider_lane_lease,
)
from .paid_resource_admission import (
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .safe_outbound_http import presigned_transfer_policy, request as safe_http_request
from .semantic_teacher_image_edit_bundle import MANIFEST, RUNTIME_REQUEST
from .semantic_teacher_image_edit_paid_authority import (
    MAX_ATTEMPT_SPEND_USD,
    MAX_TTL_SECONDS,
    consume_semantic_teacher_image_edit_paid_authority_once,
    validate_semantic_teacher_image_edit_paid_authority,
)
from .semantic_teacher_image_edit_paid_lane import (
    BILLING_SCHEMA_VERSION,
    RUNTIME_MEDIA_GAP_SCHEMA_VERSION,
    SECRET_REDACTION_SCHEMA_VERSION,
    materialize_semantic_teacher_image_edit_result,
    materialize_semantic_teacher_provider_zero_receipt,
)
from .semantic_teacher_image_edit_worker import RUNTIME_RESULT_SCHEMA_VERSION
from .task_evaluation_artifact_manifest import (
    ADAPTER_RESULT_NAME,
    PROVIDER_EVIDENCE_DIRNAME,
    PROVIDER_RUN_DIRNAME,
    TEARDOWN_MANIFEST_NAME,
    seal_lane_terminal_artifacts,
)
from .vast_independent_watchdog_control import write_started_vast_instance_id
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


EXECUTION_SCHEMA_VERSION = "semantic_teacher_image_edit_vast_execution.v1"
TEARDOWN_SCHEMA_VERSION = "semantic_teacher_image_edit_vast_teardown.v1"
PAID_LANE = "semantic_teacher_image_edit_gpu_canary"
PROBE_KIND = "semantic-teacher-image-edit"
NAME_PREFIX = "blueprint-semantic-teacher-"
OBJECT_STORE_KEY_PREFIX = "blueprint/adp009d/semantic-teacher-image-edit"
INPUT_GET_ENV = "BLUEPRINT_SEMANTIC_TEACHER_INPUT_BUNDLE_GET_URL"
OUTPUT_PUT_ENV = "BLUEPRINT_SEMANTIC_TEACHER_OUTPUT_PUT_URL"
TOKEN_ENV = "BLUEPRINT_IMAGE_EDITOR_TOKEN"
MAX_RESULT_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
# Poll the exact provider contract separately from the object-store result. Two
# consecutive API-confirmed absences leave a final grace window for a just-
# uploaded object while avoiding a full-TTL wait after a dead container.
PROVIDER_LIVENESS_POLL_SECONDS = 30.0
PROVIDER_ABSENCE_CONFIRMATIONS_REQUIRED = 2
MAX_OUTPUT_MEMBERS = 3_000
MAX_OUTPUT_MEMBER_BYTES = 128 * 1024 * 1024
_PINNED_IMAGE = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")
_TEACHER_FRAME = re.compile(r"tasks/[^/\\]+/[0-9]{5}\.png")
_SAFE_PROVIDER_BLOCKERS = frozenset(
    {
        "no_vast_offer_matching_rate_and_gpu_memory",
        "vast_instance_not_created",
        "vast_offer_search_failed",
        "vast_maximum_create_attempts_invalid",
        "vast_create_outcome_ambiguous",
    }
)
_SAFE_PROVIDER_HTTP_BLOCKER = re.compile(r"^vast_create_http_error:[0-9]{3}$")


class SemanticTeacherImageEditVastError(ValueError):
    """The one-shot provider lifecycle was not safe to enter."""


def _sanitized_provider_blockers(value: Any) -> list[str]:
    """Retain only stable Vast reason codes, never provider prose or secrets."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    sanitized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        if item in _SAFE_PROVIDER_BLOCKERS or _SAFE_PROVIDER_HTTP_BLOCKER.fullmatch(
            item
        ):
            sanitized.append(item)
    return sorted(set(sanitized))


def _confirmed_no_allocation(
    launch_result: Mapping[str, Any], *, provider_mutations: int
) -> bool:
    """Require an explicit, internally consistent zero-allocation outcome."""

    return bool(
        launch_result.get("status") == "blocked"
        and not launch_result.get("instance_id")
        and launch_result.get("allocation_created") is False
        and launch_result.get("allocation_outcome_ambiguous") is not True
        and provider_mutations == 0
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticTeacherImageEditVastError(code) from exc
    if path.is_symlink() or not isinstance(value, dict):
        raise SemanticTeacherImageEditVastError(code)
    return value


def _write_receipt(path: Path, value: dict[str, Any], *, digest_field: str) -> dict[str, Any]:
    value[digest_field] = canonical_digest(value, digest_field=digest_field)
    write_json(path, value)
    return value


def _materialize_secret_redaction(
    *,
    output_path: Path,
    runtime_output_root: Path,
    terminal_result_path: Path,
    archive_path: Path,
    authority_digest: str,
    bundle_sha256: str,
    instance_id: str,
    secret_values: tuple[str, ...],
) -> dict[str, Any]:
    stdout = runtime_output_root / "runtime_stdout.log"
    stderr = runtime_output_root / "runtime_stderr.log"
    paths = (stdout, stderr, terminal_result_path, archive_path)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise SemanticTeacherImageEditVastError(
            "semantic_teacher_secret_redaction_inputs_missing"
        )
    patterns = tuple(value.encode("utf-8") for value in secret_values if value) + (
        b"Authorization: Bearer ",
        b'"authorization"',
        b"signature=",
        b"X-Amz-Signature",
    )
    found = False
    scanned: list[dict[str, Any]] = []
    for path in paths:
        payload = path.read_bytes()
        found = found or any(pattern in payload for pattern in patterns)
        scanned.append(
            {
                "role": path.name,
                "size_bytes": len(payload),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
        )
    result = {
        "schema_version": SECRET_REDACTION_SCHEMA_VERSION,
        "status": "blocked" if found else "passed",
        "authority_digest": authority_digest,
        "bundle_sha256": bundle_sha256,
        "run_instance_id": instance_id,
        "stdout_sha256": _sha256(stdout),
        "stderr_sha256": _sha256(stderr),
        "stdout_scanned": True,
        "stderr_scanned": True,
        "runtime_result_scanned": True,
        "output_archive_scanned": True,
        "scanned_artifacts": scanned,
        "secret_values_found": found,
        "raw_secret_values_recorded": False,
        "redaction_digest": "",
    }
    return _write_receipt(output_path, result, digest_field="redaction_digest")


def _materialize_runtime_media_gap(
    *,
    runtime_output_root: Path,
    authority: Mapping[str, Any],
    receipt: Mapping[str, Any],
    instance_id: str,
    gap_type: str,
    reason_code: str,
) -> tuple[dict[str, Any], Path]:
    """Retain an allocated-run output gap without inventing request outcomes."""

    ensure_dir(runtime_output_root)
    for name in ("runtime_stdout.log", "runtime_stderr.log"):
        path = runtime_output_root / name
        if not path.exists():
            path.write_text(
                "runtime output unavailable; see typed runtime media gap\n",
                encoding="utf-8",
            )
    partial_png_inventory: list[dict[str, Any]] = []
    for path in sorted(runtime_output_root.rglob("*.png")):
        relative = path.relative_to(runtime_output_root).as_posix()
        if (
            path.is_symlink()
            or not path.is_file()
            or not _TEACHER_FRAME.fullmatch(relative)
        ):
            continue
        partial_png_inventory.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    gap = {
        "schema_version": RUNTIME_MEDIA_GAP_SCHEMA_VERSION,
        "status": "blocked_runtime_result_missing",
        "gap_type": gap_type,
        "reason_code": reason_code,
        "authority_digest": authority.get("authorization_digest"),
        "bundle_sha256": (receipt.get("bundle") or {}).get("sha256"),
        "runtime_request_digest": receipt.get("runtime_request_digest"),
        "backend_entry_digest": receipt.get("backend_entry_digest"),
        "run_instance_id": instance_id,
        "attempted_request_count_known": False,
        "attempted_request_count_upper_bound": authority.get("camera_count"),
        "runtime_result_present": False,
        "media_complete": False,
        "teacher_frames_qualified": False,
        "partial_png_inventory": partial_png_inventory,
        "raw_secret_values_recorded": False,
    }
    path = runtime_output_root / f"{RUNTIME_MEDIA_GAP_SCHEMA_VERSION}.json"
    return _write_receipt(path, gap, digest_field="gap_digest"), path


def _materialize_failure_archive(
    *, runtime_output_root: Path, output_path: Path
) -> Path:
    """Deterministically retain host closeout logs, gap, and partial PNGs."""

    members = [
        path
        for path in sorted(runtime_output_root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            relative = path.relative_to(runtime_output_root).as_posix()
            if relative not in {
                "runtime_stdout.log",
                "runtime_stderr.log",
                f"{RUNTIME_MEDIA_GAP_SCHEMA_VERSION}.json",
                f"{RUNTIME_RESULT_SCHEMA_VERSION}.json",
            } and not _TEACHER_FRAME.fullmatch(relative):
                continue
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, path.read_bytes())
    return output_path


def _seal_terminal_execution(
    *,
    root: Path,
    result: dict[str, Any],
    instance_ids: list[str],
    provider_zero_verified: bool,
    continuing_spend: bool,
    teardown_actions: list[Mapping[str, Any]],
    evidence_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Seal the allocator terminal paths without inventing provider evidence."""

    provider_run = root / PROVIDER_RUN_DIRNAME
    evidence_root = root / PROVIDER_EVIDENCE_DIRNAME
    ensure_dir(provider_run)
    ensure_dir(evidence_root)
    write_json(
        provider_run / ADAPTER_RESULT_NAME,
        {
            "schema_version": "semantic_teacher_image_edit_vast_adapter_result.v1",
            "status": result.get("status"),
            "instance_ids": instance_ids,
            "provider_mutations_performed": result.get(
                "provider_mutations_performed", 0
            ),
            "provider_zero_verified": provider_zero_verified,
            "continuing_spend_from_this_run": continuing_spend,
            "blockers": list(result.get("blockers") or []),
            "raw_secret_values_recorded": False,
        },
    )
    write_json(
        provider_run / TEARDOWN_MANIFEST_NAME,
        {
            "schema_version": "vast_teardown_manifest.v1",
            "generated_at": utc_now_iso(),
            "status": "completed" if provider_zero_verified else "blocked",
            "vast_instance_ids": instance_ids,
            "teardown_actions_performed": [dict(item) for item in teardown_actions],
            "continuing_spend_from_this_run": continuing_spend,
            "provider_zero_api_confirmed": provider_zero_verified,
            "raw_secret_values_recorded": False,
        },
    )
    retained: dict[str, dict[str, Any]] = {}
    extra_artifact_roots: dict[str, Path] = {}
    for role, path in sorted(evidence_paths.items()):
        if path.is_file() and not path.is_symlink():
            retained[role] = {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            extra_artifact_roots[role] = path
        elif path.is_dir() and not path.is_symlink():
            extra_artifact_roots[role] = path
    write_json(
        evidence_root / "semantic_teacher_closeout_index.json",
        {
            "schema_version": "semantic_teacher_image_edit_closeout_index.v1",
            "authorization_digest": result.get("authorization_digest"),
            "bundle_sha256": result.get("bundle_sha256"),
            "runtime_result_digest": result.get("runtime_result_digest"),
            "runtime_media_gap_digest": result.get("runtime_media_gap_digest"),
            "provider_zero_digest": result.get("provider_zero_digest"),
            "retained_files": retained,
            "raw_secret_values_recorded": False,
        },
    )
    sealed = seal_lane_terminal_artifacts(
        result,
        attempt_root=root,
        lane=PAID_LANE,
        extra_artifact_roots=extra_artifact_roots,
        binding={
            "authorization_digest": result.get("authorization_digest"),
            "bundle_sha256": result.get("bundle_sha256"),
            "provider": "vast",
        },
    )
    return _write_receipt(
        root / "semantic_teacher_image_edit_vast_execution.json",
        sealed,
        digest_field="execution_digest",
    )


def _close_blocked_before_launch(
    *,
    root: Path,
    reason: str,
    authority: Mapping[str, Any],
    receipt: Mapping[str, Any],
    consumption: Mapping[str, Any],
    provider: GpuRenderProvider,
    initial_scoped_inventory: Mapping[str, Any],
    initial_global_inventory: Mapping[str, Any],
    watchdog_closer: Callable[..., Mapping[str, Any]] | None,
    staging_dir: Path,
    cleanup_required: bool,
    object_store_cleaner: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    """Close an armed watchdog and re-prove both zero scopes before launch."""

    if cleanup_required:
        try:
            cleanup = dict(object_store_cleaner(staging_dir))
        except Exception as exc:  # noqa: BLE001 - retain the cleanup gap
            cleanup = {
                "status": "blocked",
                "all_objects_absent": False,
                "signed_url_files_removed": False,
                "blockers": [f"cleanup_failed:{redacted_failure_detail(exc)}"],
                "raw_secret_values_recorded": False,
            }
    else:
        cleanup = {
            "schema_version": "wam_provider_object_store_cleanup.v1",
            "status": "completed",
            "exact_object_count": 0,
            "all_objects_absent": True,
            "signed_url_files_removed": True,
            "blockers": [],
            "raw_secret_values_recorded": False,
        }
        ensure_dir(staging_dir)
        write_json(staging_dir / "wam_provider_object_store_cleanup.json", cleanup)
    try:
        final_scoped = dict(provider.billable_inventory(name_prefix=NAME_PREFIX))
    except Exception as exc:  # noqa: BLE001
        final_scoped = {
            "api_confirmed": False,
            "live_resource_count": None,
            "resources": [],
            "error_type": type(exc).__name__,
        }
    try:
        final_global = dict(provider.billable_inventory(name_prefix=""))
    except Exception as exc:  # noqa: BLE001
        final_global = {
            "api_confirmed": False,
            "live_resource_count": None,
            "resources": [],
            "error_type": type(exc).__name__,
        }
    provider_zero = _api_zero(final_scoped) and _api_zero(final_global)
    if watchdog_closer is None:
        watchdog = {
            "status": "blocked",
            "blockers": ["semantic_teacher_watchdog_closer_missing"],
            "raw_secret_values_recorded": False,
        }
    else:
        try:
            watchdog = dict(
                watchdog_closer(
                    instance_ids=[],
                    provider_teardown_completed=provider_zero,
                    provider_allocation_impossible=True,
                )
            )
        except Exception as exc:  # noqa: BLE001
            watchdog = {
                "status": "blocked",
                "blockers": [f"watchdog_close_failed:{redacted_failure_detail(exc)}"],
                "raw_secret_values_recorded": False,
            }
    write_json(root / "independent_watchdog.json", watchdog)
    zero_receipt = _write_receipt(
        root / "no_allocation_provider_zero_receipt.json",
        {
            "schema_version": "semantic_teacher_image_edit_no_allocation_provider_zero.v1",
            "status": "provider_zero" if provider_zero else "provider_nonzero",
            "provider": "vast",
            "run_instance_ids": [],
            "initial_scoped_inventory": dict(initial_scoped_inventory),
            "initial_global_inventory": dict(initial_global_inventory),
            "final_scoped_inventory": final_scoped,
            "final_global_inventory": final_global,
            "double_lane_and_global_api_zero_confirmed": bool(
                _api_zero(initial_scoped_inventory)
                and _api_zero(initial_global_inventory)
                and provider_zero
            ),
            "provider_zero_api_confirmed": provider_zero,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "independent_watchdog_status": watchdog.get("status"),
            "total_cost_usd": 0.0,
            "continuing_spend_from_this_run": not provider_zero,
            "raw_secret_values_recorded": False,
            "confirmed_at": utc_now_iso(),
        },
        digest_field="provider_zero_digest",
    )
    blockers = [reason]
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("semantic_teacher_object_store_cleanup_not_proven")
    if not provider_zero:
        blockers.append("semantic_teacher_provider_zero_not_proven")
    if watchdog.get("status") not in {
        "provider_terminal",
        "cancelled_no_allocation",
    }:
        blockers.append("semantic_teacher_independent_watchdog_not_closed")
    result = {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "status": "blocked",
            "source_commit_sha": authority.get("source_commit_sha"),
            "authorization_digest": authority.get("authorization_digest"),
            "authorization_consumption": dict(consumption),
            "bundle_sha256": (receipt.get("bundle") or {}).get("sha256"),
            "provider": "vast",
            "allocation_count": 0,
            "maximum_create_attempts": 1,
            "create_attempt_count": 0,
            "automatic_retry_count": 0,
            "retry_cap": 0,
            "provider_mutations_performed": 0,
            "cost_usd": 0.0,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "provider_zero_verified": provider_zero,
            "provider_zero_digest": zero_receipt["provider_zero_digest"],
            "continuing_spend_from_this_run": not provider_zero,
            "blockers": sorted(set(blockers)),
            "raw_secret_values_recorded": False,
        }
    return _seal_terminal_execution(
        root=root,
        result=result,
        instance_ids=[],
        provider_zero_verified=provider_zero,
        continuing_spend=not provider_zero,
        teardown_actions=[],
        evidence_paths={
            "no_allocation_provider_zero": (
                root / "no_allocation_provider_zero_receipt.json"
            ),
            "independent_watchdog": root / "independent_watchdog.json",
            "object_store_cleanup": (
                staging_dir / "wam_provider_object_store_cleanup.json"
            ),
        },
    )


def _api_zero(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("api_confirmed") is True
        and value.get("live_resource_count") == 0
        and value.get("resources") == []
    )


def _read_token_file(path: str | Path) -> str:
    token_path = Path(path).expanduser().resolve()
    try:
        metadata = token_path.stat()
        raw = token_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SemanticTeacherImageEditVastError(
            "semantic_teacher_openai_token_file_invalid"
        ) from exc
    token = raw.strip()
    if (
        token_path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o077
        or metadata.st_size > 16 * 1024
        or not token
        or "\n" in token
        or "\r" in token
    ):
        raise SemanticTeacherImageEditVastError(
            "semantic_teacher_openai_token_file_invalid"
        )
    return token


def _provider_instance_absent(provider: Any, instance_id: str) -> bool:
    """Return true only for an exact API-confirmed provider absence."""

    try:
        inspection = dict(provider.inspect(instance_id))
    except (OSError, RuntimeError, ValueError):
        return False
    return bool(
        inspection.get("status") == "absent"
        and inspection.get("api_confirmed") is True
        and inspection.get("provider_absence_confirmed") is True
        and not inspection.get("blockers")
    )


def _default_result_fetcher(url: str) -> bytes:
    try:
        response = safe_http_request(
            url,
            method="GET",
            timeout_seconds=60,
            policy=presigned_transfer_policy(
                url, max_response_bytes=MAX_RESULT_ARCHIVE_BYTES
            ),
            max_response_bytes=MAX_RESULT_ARCHIVE_BYTES,
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError("semantic_teacher_output_not_ready") from exc
        raise
    if response.status == 404:
        raise FileNotFoundError("semantic_teacher_output_not_ready")
    if response.status != 200:
        raise SemanticTeacherImageEditVastError(
            f"semantic_teacher_output_http:{response.status}"
        )
    return bytes(response.body)


def _validate_bundle_runtime_bindings(
    bundle: Path,
    *,
    receipt: Mapping[str, Any],
    authority: Mapping[str, Any],
    checkout_source_commit: str,
    runtime_image_identity: str,
) -> dict[str, Any]:
    """Reopen the committed archive instead of trusting its outer receipt."""

    try:
        with zipfile.ZipFile(bundle) as archive:
            request_value = json.loads(archive.read(RUNTIME_REQUEST))
            manifest_value = json.loads(archive.read(MANIFEST))
    except (KeyError, OSError, UnicodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise SemanticTeacherImageEditVastError(
            "semantic_teacher_bundle_runtime_binding_invalid"
        ) from exc
    if not isinstance(request_value, dict) or not isinstance(manifest_value, dict):
        raise SemanticTeacherImageEditVastError(
            "semantic_teacher_bundle_runtime_binding_invalid"
        )
    backend = request_value.get("backend")
    backend = backend if isinstance(backend, Mapping) else {}
    registry_entry = backend.get("registry_entry")
    registry_entry = registry_entry if isinstance(registry_entry, Mapping) else {}
    execution = backend.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    tasks = request_value.get("tasks")
    if not isinstance(tasks, list):
        raise SemanticTeacherImageEditVastError(
            "semantic_teacher_bundle_runtime_binding_invalid"
        )
    order: list[dict[str, Any]] = []
    total_cameras = 0
    task_ids: set[str] = set()
    for task in tasks:
        task = task if isinstance(task, Mapping) else {}
        task_id = str(task.get("task_id") or "")
        frames = task.get("frames")
        if not task_id or task_id in task_ids or not isinstance(frames, list):
            raise SemanticTeacherImageEditVastError(
                "semantic_teacher_bundle_runtime_order_invalid"
            )
        task_ids.add(task_id)
        camera_ids: list[str] = []
        for index, frame in enumerate(frames):
            frame = frame if isinstance(frame, Mapping) else {}
            camera_id = str(frame.get("camera_id") or "")
            if (
                frame.get("frame_index") != index
                or not camera_id
                or camera_id in camera_ids
            ):
                raise SemanticTeacherImageEditVastError(
                    "semantic_teacher_bundle_runtime_order_invalid"
                )
            camera_ids.append(camera_id)
        total_cameras += len(camera_ids)
        order.append({"task_id": task_id, "camera_ids": camera_ids})
    order_digest = canonical_digest({"tasks": order})
    expected_backend = str(receipt.get("backend_entry_digest") or "")
    expected_model = str(authority.get("model_snapshot") or "")
    if (
        request_value.get("source_commit_sha") != checkout_source_commit
        or request_value.get("request_digest")
        != canonical_digest(request_value, digest_field="request_digest")
        or request_value.get("request_digest")
        != receipt.get("runtime_request_digest")
        or backend.get("backend_entry_digest") != expected_backend
        or expected_backend != authority.get("backend_entry_digest")
        or expected_backend != canonical_digest(registry_entry)
        or execution.get("model_snapshot") != expected_model
        or not expected_model
        or execution.get("runtime_image_identity") != runtime_image_identity
        or execution.get("adapter_id") != authority.get("adapter_id")
        or not str(authority.get("adapter_id") or "")
        or len(tasks) != receipt.get("task_count")
        or total_cameras != receipt.get("camera_count")
        or manifest_value.get("source_commit_sha") != checkout_source_commit
        or manifest_value.get("runtime_request_digest")
        != receipt.get("runtime_request_digest")
        or manifest_value.get("backend_entry_digest") != expected_backend
        or manifest_value.get("task_count") != receipt.get("task_count")
        or manifest_value.get("camera_count") != receipt.get("camera_count")
        or manifest_value.get("automatic_retry_count") != 0
        or authority.get("runtime_image_identity") != runtime_image_identity
    ):
        raise SemanticTeacherImageEditVastError(
            "semantic_teacher_bundle_runtime_binding_invalid"
        )
    return {
        "runtime_request_digest": request_value["request_digest"],
        "backend_entry_digest": expected_backend,
        "adapter_id": execution["adapter_id"],
        "model_snapshot": expected_model,
        "task_camera_order": order,
        "task_camera_order_digest": order_digest,
    }


def _watchdog_valid(
    watchdog: Mapping[str, Any], *, now_epoch: float, hard_ttl_seconds: int
) -> bool:
    try:
        pid = int(watchdog.get("watchdog_pid") or watchdog.get("pid") or 0)
        started = float(
            watchdog.get("watchdog_started_epoch")
            or watchdog.get("started_epoch")
            or 0
        )
        deadline = float(
            watchdog.get("watchdog_deadline_epoch")
            or watchdog.get("deadline_epoch")
            or 0
        )
    except (TypeError, ValueError):
        return False
    if (
        watchdog.get("status") != "armed"
        or watchdog.get("independent_process") is not True
        or not str(
            watchdog.get("pod_name_prefix") or ""
        ).startswith(NAME_PREFIX)
        or pid <= 0
        or started <= 0
        or started > now_epoch
        or deadline <= now_epoch
        or deadline - started < hard_ttl_seconds
    ):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _bootstrap_script() -> str:
    """Return the exact remote transport with one worker invocation."""

    return r'''set -euo pipefail
umask 077
bundle_path=/work/semantic_teacher_image_edit_provider_bundle.zip
bundle_root=/work/semantic_teacher_image_edit_provider_bundle
output_root=/work/semantic_teacher_image_edit_runtime_output
output_zip=/work/semantic_teacher_image_edit_runtime_output.zip
secret_root=/run/blueprint-secrets
secret_file="$secret_root/image_editor_token"
cleanup_secret() {
  unset BLUEPRINT_IMAGE_EDITOR_TOKEN || true
  rm -f "$secret_file"
}
trap cleanup_secret EXIT
mkdir -p "$(dirname "$bundle_path")"
python - "$bundle_path" <<'PY'
import hashlib, os, sys
from pathlib import Path
from urllib.request import Request, urlopen
url = os.environ["BLUEPRINT_SEMANTIC_TEACHER_INPUT_BUNDLE_GET_URL"]
if not url.startswith("https://"):
    raise SystemExit("semantic_teacher_bundle_url_invalid")
with urlopen(Request(url, method="GET"), timeout=180) as response:
    status = int(getattr(response, "status", 0))
    payload = response.read(2147483649)
if status != 200 or len(payload) > 2147483648:
    raise SystemExit(f"semantic_teacher_bundle_download_failed:{status}")
expected = os.environ["BLUEPRINT_SEMANTIC_TEACHER_BUNDLE_DIGEST"].removeprefix("sha256:")
if hashlib.sha256(payload).hexdigest() != expected:
    raise SystemExit("semantic_teacher_bundle_digest_mismatch")
Path(sys.argv[1]).write_bytes(payload)
PY
rm -rf "$bundle_root" "$output_root" "$output_zip"
mkdir -p "$bundle_root" "$output_root" "$secret_root"
python - "$bundle_path" "$bundle_root" <<'PY'
import stat, sys, zipfile
from pathlib import Path
archive_path, root = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
with zipfile.ZipFile(archive_path) as archive:
    infos = archive.infolist()
    if len(infos) > 3000:
        raise SystemExit("semantic_teacher_bundle_member_count_invalid")
    for info in infos:
        relative = Path(info.filename)
        mode = info.external_attr >> 16
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or info.is_dir()
            or stat.S_ISLNK(mode)
            or info.file_size > 134217728
        ):
            raise SystemExit("semantic_teacher_bundle_member_invalid")
        destination = (root / relative).resolve()
        if root not in destination.parents:
            raise SystemExit("semantic_teacher_bundle_member_invalid")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, destination.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
PY
printf '%s' "$BLUEPRINT_IMAGE_EDITOR_TOKEN" > "$secret_file"
chmod 600 "$secret_file"
unset BLUEPRINT_IMAGE_EDITOR_TOKEN
set +e
BLUEPRINT_IMAGE_EDITOR_TOKEN="$(<"$secret_file")" \
BLUEPRINT_SEMANTIC_TEACHER_OUTPUT_DIR="$output_root" \
bash "$bundle_root/provider_runtime/run_semantic_teacher_image_edit.sh" \
  >"$output_root/runtime_stdout.log" 2>"$output_root/runtime_stderr.log"
worker_status=$?
set -e
cleanup_secret
python - "$output_root" "$output_zip" <<'PY'
import re, stat, sys, zipfile
from pathlib import Path
root, destination = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
frame = re.compile(r"tasks/[^/\\]+/[0-9]{5}\.png")
fixed = {
    "semantic_teacher_image_edit_runtime_result.v1.json",
    "runtime_stdout.log",
    "runtime_stderr.log",
}
files = [path for path in sorted(root.rglob("*")) if path.is_file()]
for path in files:
    relative = path.relative_to(root).as_posix()
    if relative not in fixed and not frame.fullmatch(relative):
        raise SystemExit("semantic_teacher_output_member_not_allowlisted")
with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        relative = path.relative_to(root).as_posix()
        info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = (stat.S_IFREG | 0o600) << 16
        archive.writestr(info, path.read_bytes())
PY
python - "$output_zip" <<'PY'
import os, sys
from pathlib import Path
from urllib.request import Request, urlopen
url = os.environ["BLUEPRINT_SEMANTIC_TEACHER_OUTPUT_PUT_URL"]
payload = Path(sys.argv[1]).read_bytes()
if not url.startswith("https://"):
    raise SystemExit("semantic_teacher_output_url_invalid")
request = Request(
    url,
    data=payload,
    method="PUT",
    headers={"Content-Type": "application/zip"},
)
with urlopen(request, timeout=180) as response:
    status = int(getattr(response, "status", 0))
    response.read(1025)
if status not in {200, 201, 204}:
    raise SystemExit(f"semantic_teacher_output_upload_failed:{status}")
PY
exit "$worker_status"
'''


def _extract_and_validate_output(
    archive_bytes: bytes,
    *,
    output_root: Path,
    secret_values: tuple[str, ...],
    expected_task_count: int,
    expected_camera_count: int,
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if not archive_bytes or len(archive_bytes) > MAX_RESULT_ARCHIVE_BYTES:
        raise SemanticTeacherImageEditVastError(
            "semantic_teacher_output_archive_invalid"
        )
    forbidden = tuple(value.encode("utf-8") for value in secret_values if value)
    if any(value in archive_bytes for value in forbidden):
        raise SemanticTeacherImageEditVastError(
            "semantic_teacher_output_contains_secret"
        )
    archive_path = output_root.parent / "semantic_teacher_image_edit_runtime_output.zip"
    archive_path.write_bytes(archive_bytes)
    ensure_dir(output_root)
    total_size = 0
    observed: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_OUTPUT_MEMBERS:
            raise SemanticTeacherImageEditVastError(
                "semantic_teacher_output_archive_invalid"
            )
        for info in infos:
            relative = Path(info.filename)
            mode = info.external_attr >> 16
            name = relative.as_posix()
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or info.is_dir()
                or stat.S_ISLNK(mode)
                or name in observed
                or info.file_size > MAX_OUTPUT_MEMBER_BYTES
                or (
                    name
                    not in {
                        f"{RUNTIME_RESULT_SCHEMA_VERSION}.json",
                        "runtime_stdout.log",
                        "runtime_stderr.log",
                    }
                    and not _TEACHER_FRAME.fullmatch(name)
                )
            ):
                raise SemanticTeacherImageEditVastError(
                    "semantic_teacher_output_member_not_allowlisted"
                )
            total_size += info.file_size
            if total_size > MAX_RESULT_ARCHIVE_BYTES:
                raise SemanticTeacherImageEditVastError(
                    "semantic_teacher_output_archive_invalid"
                )
            observed.add(name)
            destination = (output_root / relative).resolve()
            if output_root not in destination.parents:
                raise SemanticTeacherImageEditVastError(
                    "semantic_teacher_output_member_not_allowlisted"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.read(info)
            if any(value in payload for value in forbidden):
                raise SemanticTeacherImageEditVastError(
                    "semantic_teacher_output_contains_secret"
                )
            destination.write_bytes(payload)
    runtime_path = output_root / f"{RUNTIME_RESULT_SCHEMA_VERSION}.json"
    runtime = _read(runtime_path, code="semantic_teacher_runtime_result_invalid")
    if (
        runtime.get("schema_version") != RUNTIME_RESULT_SCHEMA_VERSION
        or runtime.get("result_digest")
        != canonical_digest(runtime, digest_field="result_digest")
        or runtime.get("status")
        != "completed_unreviewed_semantic_teacher_candidates"
        or runtime.get("task_count") != expected_task_count
        or runtime.get("request_count") != expected_camera_count
        or runtime.get("retry_count") != 0
        or runtime.get("raw_secret_values_recorded") is not False
        or runtime.get("appearance_qualified") is not False
        or runtime.get("source_runtime_request_digest")
        != expected_binding.get("runtime_request_digest")
        or runtime.get("backend_entry_digest")
        != expected_binding.get("backend_entry_digest")
        or runtime.get("adapter_id") != expected_binding.get("adapter_id")
        or runtime.get("model_snapshot") != expected_binding.get("model_snapshot")
    ):
        raise SemanticTeacherImageEditVastError(
            "semantic_teacher_runtime_result_invalid"
        )
    referenced: set[str] = set()
    observed_order: list[dict[str, Any]] = []
    for task in runtime.get("tasks") or []:
        observed_camera_ids: list[str] = []
        for frame in task.get("frames") or []:
            observed_camera_ids.append(str(frame.get("camera_id") or ""))
            record = frame.get("semantic_teacher_frame") or {}
            name = str(record.get("relative_path") or "")
            path = (output_root / name).resolve()
            if (
                not _TEACHER_FRAME.fullmatch(name)
                or output_root not in path.parents
                or not path.is_file()
                or path.stat().st_size != record.get("size_bytes")
                or _sha256(path) != record.get("sha256")
            ):
                raise SemanticTeacherImageEditVastError(
                    "semantic_teacher_runtime_frame_inventory_invalid"
                )
            referenced.add(name)
        observed_order.append(
            {
                "task_id": str(task.get("task_id") or ""),
                "camera_ids": observed_camera_ids,
            }
        )
    observed_frames = {name for name in observed if _TEACHER_FRAME.fullmatch(name)}
    if len(referenced) != expected_camera_count or referenced != observed_frames:
        raise SemanticTeacherImageEditVastError(
            "semantic_teacher_runtime_frame_inventory_invalid"
        )
    if (
        observed_order != expected_binding.get("task_camera_order")
        or canonical_digest({"tasks": observed_order})
        != expected_binding.get("task_camera_order_digest")
    ):
        raise SemanticTeacherImageEditVastError(
            "semantic_teacher_runtime_camera_order_invalid"
        )
    return runtime


def _execute_semantic_teacher_image_edit_vast(
    *,
    authority_path: str | Path,
    bundle_path: str | Path,
    bundle_receipt_path: str | Path,
    checkout_source_commit: str,
    job_dir: str | Path,
    token_file: str | Path,
    runtime_image_identity: str,
    preflight: Mapping[str, Any],
    provider: GpuRenderProvider,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    object_store_stager: Callable[..., Mapping[str, Any]] = (
        stage_wam_provider_bundle_object_store
    ),
    object_store_cleaner: Callable[..., Mapping[str, Any]] = (
        cleanup_staged_wam_provider_objects
    ),
    result_fetcher: Callable[[str], bytes] = _default_result_fetcher,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    watchdog_validator: Callable[[Mapping[str, Any], float, int], bool]
    | None = None,
    watchdog_instance_binder: Callable[[Path, int], None] = (
        write_started_vast_instance_id
    ),
    watchdog_closer: Callable[..., Mapping[str, Any]] | None = None,
    excluded_machine_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Run one immutable bundle; always tear down compute and staged objects."""

    require_paid_resource_admission_grant(
        paid_resource_admission_grant, resource_class="gpu_render"
    )
    root = Path(job_dir).expanduser().resolve()
    ensure_dir(root)
    authority_file = Path(authority_path).expanduser().resolve()
    bundle = Path(bundle_path).expanduser().resolve()
    receipt_file = Path(bundle_receipt_path).expanduser().resolve()
    authority = _read(
        authority_file, code="semantic_teacher_paid_authority_invalid"
    )
    receipt = _read(
        receipt_file, code="semantic_teacher_bundle_receipt_invalid"
    )
    hourly_cap = authority.get("maximum_hourly_rate_usd")
    spend_cap = authority.get("hard_total_spend_cap_usd")
    hard_ttl = authority.get("hard_ttl_seconds")
    vast_spend_upper_bound = authority.get("vast_spend_upper_bound_usd")
    hosted_spend_upper_bound = authority.get(
        "hosted_editor_spend_upper_bound_usd"
    )
    if (
        provider.name != "vast"
        or not _PINNED_IMAGE.fullmatch(runtime_image_identity)
        or isinstance(hourly_cap, bool)
        or not isinstance(hourly_cap, (int, float))
        or not math.isfinite(float(hourly_cap))
        or not 0 < float(hourly_cap) <= MAX_ATTEMPT_SPEND_USD
        or isinstance(spend_cap, bool)
        or not isinstance(spend_cap, (int, float))
        or not math.isfinite(float(spend_cap))
        or not 0 < float(spend_cap) <= MAX_ATTEMPT_SPEND_USD
        or isinstance(hard_ttl, bool)
        or not isinstance(hard_ttl, int)
        or not 1 <= hard_ttl <= MAX_TTL_SECONDS
        or float(hourly_cap) * hard_ttl / 3600 > float(spend_cap)
        or authority.get("maximum_provider_allocations") != 1
        or authority.get("maximum_automatic_retries") != 0
        or authority.get("automatic_paid_retry_authorized") is not False
        or authority.get("runtime_image_identity") != runtime_image_identity
        or isinstance(vast_spend_upper_bound, bool)
        or not isinstance(vast_spend_upper_bound, (int, float))
        or not math.isfinite(float(vast_spend_upper_bound))
        or float(vast_spend_upper_bound)
        < float(hourly_cap) * hard_ttl / 3600
        or isinstance(hosted_spend_upper_bound, bool)
        or not isinstance(hosted_spend_upper_bound, (int, float))
        or not math.isfinite(float(hosted_spend_upper_bound))
        or float(hosted_spend_upper_bound) <= 0
        or float(vast_spend_upper_bound) + float(hosted_spend_upper_bound)
        > float(spend_cap)
    ):
        raise SemanticTeacherImageEditVastError(
            "semantic_teacher_execution_bounds_invalid"
        )
    validate_semantic_teacher_image_edit_paid_authority(
        authority,
        bundle_path=bundle,
        bundle_receipt=receipt,
        source_commit_sha=checkout_source_commit,
        backend_entry_digest=str(receipt.get("backend_entry_digest") or ""),
        task_count=int(receipt.get("task_count") or 0),
        camera_count=int(receipt.get("camera_count") or 0),
        maximum_hourly_rate_usd=float(hourly_cap),
        hard_total_spend_cap_usd=float(spend_cap),
        hard_ttl_seconds=hard_ttl,
    )
    if (
        receipt.get("source_commit_sha") != checkout_source_commit
        or (receipt.get("bundle") or {}).get("sha256") != _sha256(bundle)
        or (receipt.get("bundle") or {}).get("size_bytes") != bundle.stat().st_size
    ):
        raise SemanticTeacherImageEditVastError(
            "semantic_teacher_bundle_binding_invalid"
        )
    runtime_binding = _validate_bundle_runtime_bindings(
        bundle,
        receipt=receipt,
        authority=authority,
        checkout_source_commit=checkout_source_commit,
        runtime_image_identity=runtime_image_identity,
    )
    runtime_request_path = root / Path(RUNTIME_REQUEST).name
    with zipfile.ZipFile(bundle) as archive:
        runtime_request_path.write_bytes(archive.read(RUNTIME_REQUEST))
    watchdog = preflight.get("watchdog")
    watchdog = watchdog if isinstance(watchdog, Mapping) else {}
    started_at = float(clock())
    validator = watchdog_validator or (
        lambda value, now, ttl: _watchdog_valid(
            value, now_epoch=now, hard_ttl_seconds=ttl
        )
    )
    if not validator(watchdog, started_at, hard_ttl):
        raise SemanticTeacherImageEditVastError(
            "semantic_teacher_independent_watchdog_not_live"
        )
    scoped_before = provider.billable_inventory(name_prefix=NAME_PREFIX)
    global_before = provider.billable_inventory(name_prefix="")
    if not _api_zero(scoped_before) or not _api_zero(global_before):
        raise SemanticTeacherImageEditVastError(
            "semantic_teacher_provider_not_zero_before_launch"
        )
    consumption = consume_semantic_teacher_image_edit_paid_authority_once(
        authority, source_commit_sha=checkout_source_commit
    )
    staging_dir = root / "object_store_staging"
    if consumption.get("status") != "consumed":
        return _close_blocked_before_launch(
            root=root,
            reason=str(
                (consumption.get("blockers") or [
                    "semantic_teacher_authority_consumption_blocked"
                ])[0]
            ),
            authority=authority,
            receipt=receipt,
            consumption=consumption,
            provider=provider,
            initial_scoped_inventory=scoped_before,
            initial_global_inventory=global_before,
            watchdog_closer=watchdog_closer,
            staging_dir=staging_dir,
            cleanup_required=False,
            object_store_cleaner=object_store_cleaner,
        )
    try:
        token = _read_token_file(token_file)
    except SemanticTeacherImageEditVastError as exc:
        return _close_blocked_before_launch(
            root=root,
            reason=str(exc),
            authority=authority,
            receipt=receipt,
            consumption=consumption,
            provider=provider,
            initial_scoped_inventory=scoped_before,
            initial_global_inventory=global_before,
            watchdog_closer=watchdog_closer,
            staging_dir=staging_dir,
            cleanup_required=False,
            object_store_cleaner=object_store_cleaner,
        )
    try:
        staging = dict(
            object_store_stager(
                job_dir=staging_dir,
                bundle_path=bundle,
                key_prefix=OBJECT_STORE_KEY_PREFIX,
                output_content_type="application/zip",
                expiration_seconds=max(hard_ttl + 1800, 7200),
            )
        )
    except Exception as exc:  # noqa: BLE001 - close watchdog and prove zero
        ensure_dir(staging_dir)
        staging = {
            "status": "blocked",
            "blockers": [
                f"semantic_teacher_object_store_staging_failed:{redacted_failure_detail(exc)}"
            ],
        }
    if staging.get("status") != "completed":
        return _close_blocked_before_launch(
            root=root,
            reason=str(
                (staging.get("blockers") or [
                    "semantic_teacher_object_store_staging_blocked"
                ])[0]
            ),
            authority=authority,
            receipt=receipt,
            consumption=consumption,
            provider=provider,
            initial_scoped_inventory=scoped_before,
            initial_global_inventory=global_before,
            watchdog_closer=watchdog_closer,
            staging_dir=staging_dir,
            cleanup_required=True,
            object_store_cleaner=object_store_cleaner,
        )
    try:
        input_url = (staging_dir / "provider_bundle_url.txt").read_text(
            encoding="utf-8"
        ).strip()
        output_put_url = (staging_dir / "provider_output_put_url.txt").read_text(
            encoding="utf-8"
        ).strip()
        output_get_url = (staging_dir / "provider_output_get_url.txt").read_text(
            encoding="utf-8"
        ).strip()
    except OSError as exc:
        del exc
        return _close_blocked_before_launch(
            root=root,
            reason="semantic_teacher_staging_urls_invalid",
            authority=authority,
            receipt=receipt,
            consumption=consumption,
            provider=provider,
            initial_scoped_inventory=scoped_before,
            initial_global_inventory=global_before,
            watchdog_closer=watchdog_closer,
            staging_dir=staging_dir,
            cleanup_required=True,
            object_store_cleaner=object_store_cleaner,
        )
    if not all(value.startswith("https://") for value in (input_url, output_put_url, output_get_url)):
        return _close_blocked_before_launch(
            root=root,
            reason="semantic_teacher_staging_urls_invalid",
            authority=authority,
            receipt=receipt,
            consumption=consumption,
            provider=provider,
            initial_scoped_inventory=scoped_before,
            initial_global_inventory=global_before,
            watchdog_closer=watchdog_closer,
            staging_dir=staging_dir,
            cleanup_required=True,
            object_store_cleaner=object_store_cleaner,
        )

    pending_dir = root / "pending_teardowns"
    lease_dir = root / "leases"
    ensure_dir(pending_dir)
    ensure_dir(lease_dir)
    reconciliation = build_paid_provider_lane_reconciliation(
        provider="vast",
        lane=PAID_LANE,
        provider_inventory=global_before,
        open_pending_teardowns=load_pending_teardowns(registry_dir=pending_dir),
    )
    lease = acquire_paid_provider_lane_lease(
        provider="vast",
        lane=PAID_LANE,
        job_dir=str(root),
        ttl_seconds=hard_ttl,
        lease_dir=lease_dir,
        reconciliation=reconciliation,
    )
    if lease.get("status") != "acquired":
        return _close_blocked_before_launch(
            root=root,
            reason="semantic_teacher_paid_lane_not_acquired",
            authority=authority,
            receipt=receipt,
            consumption=consumption,
            provider=provider,
            initial_scoped_inventory=scoped_before,
            initial_global_inventory=global_before,
            watchdog_closer=watchdog_closer,
            staging_dir=staging_dir,
            cleanup_required=True,
            object_store_cleaner=object_store_cleaner,
        )
    request_digest = str(receipt.get("runtime_request_digest") or "")
    watchdog_prefix = str(
        watchdog.get("pod_name_prefix") or ""
    )
    name = f"{watchdog_prefix}{request_digest.removeprefix('sha256:')[:12]}"
    pending = open_pending_teardown(
        provider="vast",
        lane=PAID_LANE,
        run_id=str(authority.get("authorization_digest") or request_digest),
        resource_name=name,
        job_dir=root,
        max_age_seconds=hard_ttl + 1800,
        registry_dir=pending_dir,
    )
    pending_path = Path(str(pending["path"]))
    pending_resolved = False
    instance_id: str | None = None
    launch_result: dict[str, Any] = {}
    runtime_result: dict[str, Any] | None = None
    runtime_gap_type: str | None = None
    runtime_gap_reason = ""
    blockers: list[str] = []
    provider_mutations = 0
    allocation_count = 0
    terminate_result: dict[str, Any] = {"status": "not_required"}
    cleanup: dict[str, Any] = {}
    watchdog_close: dict[str, Any] = {}
    scoped_after: dict[str, Any] = {}
    global_after: dict[str, Any] = {}
    try:
        price = float(preflight.get("on_demand_price_usd_per_hour") or 0)
        if not 0 < price <= float(hourly_cap):
            blockers.append("semantic_teacher_preflight_price_invalid")
            cancel_pending_teardown(
                pending_path,
                reason="preflight_price_invalid_no_allocation",
                evidence={"provider_mutations_performed": 0},
            )
            pending_resolved = True
        else:
            try:
                scoped_launch = dict(
                    provider.billable_inventory(name_prefix=NAME_PREFIX)
                )
                global_launch = dict(provider.billable_inventory(name_prefix=""))
            except Exception as exc:  # noqa: BLE001 - no launch on inventory gap
                scoped_launch = {}
                global_launch = {}
                blockers.append(
                    f"semantic_teacher_prelaunch_inventory_failed:{redacted_failure_detail(exc)}"
                )
            if not _api_zero(scoped_launch) or not _api_zero(global_launch):
                if not any(
                    item.startswith("semantic_teacher_prelaunch_inventory_failed:")
                    for item in blockers
                ):
                    blockers.append(
                        "semantic_teacher_provider_not_zero_immediately_before_launch"
                    )
                cancel_pending_teardown(
                    pending_path,
                    reason="provider_nonzero_immediately_before_launch",
                    evidence={"provider_mutations_performed": 0},
                )
                pending_resolved = True
            else:
                write_json(root / "prelaunch_scoped_inventory.json", scoped_launch)
                write_json(root / "prelaunch_global_inventory.json", global_launch)
        if not blockers:
            env = {
                INPUT_GET_ENV: input_url,
                OUTPUT_PUT_ENV: output_put_url,
                TOKEN_ENV: token,
                "BLUEPRINT_SEMANTIC_TEACHER_BUNDLE_DIGEST": str(
                    (receipt.get("bundle") or {}).get("sha256") or ""
                ),
                "BLUEPRINT_SEMANTIC_TEACHER_RUNTIME_REQUEST_DIGEST": request_digest,
                "BLUEPRINT_SEMANTIC_TEACHER_SOURCE_COMMIT": checkout_source_commit,
            }
            spec = RenderLaunchSpec(
                name=name,
                image=runtime_image_identity,
                env=env,
                bootstrap_argv=["-lc", _bootstrap_script()],
                entrypoint=["bash"],
                container_disk_gb=max(
                    16,
                    int(preflight.get("container_disk_bytes") or 0) // 1024**3,
                ),
                volume_gb=0,
                max_hourly_rate_usd=price,
                min_gpu_ram_mb=max(
                    1,
                    int(preflight.get("gpu_memory_bytes") or 0) // 1_000_000,
                ),
                requires_rtx=False,
                vast_launch_mode="args",
                excluded_machine_ids=tuple(
                    int(value) for value in excluded_machine_ids
                ),
            )
            provider_request = provider.build_request(spec, root)
            provider_request["maximum_create_attempts"] = 1
            provider_request["prelaunch_spend_guard"] = {
                "schema_version": "semantic_teacher_image_edit_prelaunch_spend_guard.v1",
                "required_before_provider_launch": True,
                "can_launch": True,
                "blockers": [],
                "maximum_hourly_rate_usd": float(hourly_cap),
                "hard_total_spend_cap_usd": float(spend_cap),
                "hard_ttl_seconds": hard_ttl,
                "maximum_provider_allocations": 1,
                "automatic_retry_count": 0,
                "maximum_create_attempts": 1,
                "authorization_digest": authority.get("authorization_digest"),
                "bundle_sha256": (receipt.get("bundle") or {}).get("sha256"),
            }
            launch_result = dict(
                provider.launch(
                    root,
                    provider_request,
                    cold=True,
                    allow_cold_fallback=False,
                    paid_resource_admission_grant=paid_resource_admission_grant,
                )
            )
            provider_blockers = _sanitized_provider_blockers(
                launch_result.get("blockers")
            )
            blockers.extend(provider_blockers)
            confirmed_no_allocation = _confirmed_no_allocation(
                launch_result, provider_mutations=provider_mutations
            )
            create_attempt_count = launch_result.get("create_attempt_count")
            maximum_create_attempts = launch_result.get("maximum_create_attempts")
            create_attempt_contract_valid = bool(
                isinstance(maximum_create_attempts, int)
                and not isinstance(maximum_create_attempts, bool)
                and maximum_create_attempts == 1
                and isinstance(create_attempt_count, int)
                and not isinstance(create_attempt_count, bool)
                and (
                    create_attempt_count == 1
                    or (create_attempt_count == 0 and confirmed_no_allocation)
                )
            )
            if not create_attempt_contract_valid:
                blockers.append(
                    "semantic_teacher_vast_create_attempt_contract_invalid"
                )
            allocation_outcome_ambiguous = bool(
                launch_result.get("allocation_outcome_ambiguous") is True
                or (
                    launch_result.get("status") != "launched"
                    and not confirmed_no_allocation
                )
                or (
                    launch_result.get("status") == "launched"
                    and not launch_result.get("instance_id")
                )
            )
            if allocation_outcome_ambiguous:
                launch_result["allocation_outcome_ambiguous"] = True
                provider_mutations += 1
                mark_pending_teardown_ambiguous(
                    pending_path,
                    reason="semantic_teacher_vast_create_outcome_ambiguous",
                    evidence={"blockers": provider_blockers},
                )
                blockers.append("semantic_teacher_vast_create_outcome_ambiguous")
            elif launch_result.get("status") != "launched" or not launch_result.get(
                "instance_id"
            ):
                cancel_pending_teardown(
                    pending_path,
                    reason="provider_confirmed_no_allocation",
                    evidence={"status": launch_result.get("status")},
                )
                pending_resolved = True
                blockers.append("semantic_teacher_vast_instance_not_created")
            else:
                instance_id = str(launch_result["instance_id"])
                allocation_count = 1
                provider_mutations += 1
                bind_pending_teardown_instance(pending_path, instance_id)
                started_instance_path = str(
                    watchdog.get("started_instance_id_path") or ""
                )
                if started_instance_path:
                    binding_path = Path(started_instance_path).expanduser()
                    if not binding_path.is_absolute() or binding_path.is_symlink():
                        blockers.append(
                            "semantic_teacher_watchdog_instance_binding_invalid"
                        )
                    else:
                        watchdog_instance_binder(binding_path, int(instance_id))
                output_archive: bytes | None = None
                provider_absence_confirmations = 0
                provider_absence_confirmed = False
                liveness_checked_at = float(clock())
                while float(clock()) - started_at <= hard_ttl:
                    try:
                        output_archive = result_fetcher(output_get_url)
                        break
                    except (FileNotFoundError, TimeoutError):
                        now = float(clock())
                        if now - started_at >= hard_ttl:
                            break
                        if now - liveness_checked_at >= PROVIDER_LIVENESS_POLL_SECONDS:
                            liveness_checked_at = now
                            if _provider_instance_absent(provider, instance_id):
                                provider_absence_confirmations += 1
                            else:
                                provider_absence_confirmations = 0
                            if (
                                provider_absence_confirmations
                                >= PROVIDER_ABSENCE_CONFIRMATIONS_REQUIRED
                            ):
                                provider_absence_confirmed = True
                                break
                        sleeper(
                            min(
                                5.0,
                                max(
                                    0.0,
                                    hard_ttl - (float(clock()) - started_at),
                                ),
                            )
                        )
                if provider_absence_confirmed:
                    blockers.append("semantic_teacher_provider_instance_vanished")
                    runtime_gap_type = "provider_instance_vanished"
                    runtime_gap_reason = (
                        "provider_confirmed_absent_before_any_output_appeared"
                    )
                elif output_archive is None:
                    blockers.append("semantic_teacher_output_timeout")
                    runtime_gap_type = "runtime_timeout"
                    runtime_gap_reason = "output_download_timed_out_after_allocation"
                else:
                    try:
                        runtime_result = _extract_and_validate_output(
                            output_archive,
                            output_root=root / "runtime_output",
                            secret_values=(
                                token,
                                input_url,
                                output_put_url,
                                output_get_url,
                            ),
                            expected_task_count=int(receipt["task_count"]),
                            expected_camera_count=int(receipt["camera_count"]),
                            expected_binding=runtime_binding,
                        )
                    except (
                        OSError,
                        zipfile.BadZipFile,
                        SemanticTeacherImageEditVastError,
                    ) as exc:
                        blockers.append(str(exc))
                        runtime_gap_type = "runtime_output_malformed"
                        runtime_gap_reason = str(exc)
    finally:
        if instance_id is not None:
            try:
                terminate_result = dict(provider.terminate(instance_id))
                provider_mutations += 1
            except Exception as exc:  # noqa: BLE001 - continue terminal closeout
                terminate_result = {
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                }
                blockers.append("semantic_teacher_vast_terminate_failed")
        try:
            cleanup = dict(object_store_cleaner(staging_dir))
        except Exception as exc:  # noqa: BLE001 - continue watchdog/provider-zero
            cleanup = {
                "status": "blocked",
                "all_objects_absent": False,
                "signed_url_files_removed": False,
                "blockers": [
                    f"semantic_teacher_object_store_cleanup_failed:{redacted_failure_detail(exc)}"
                ],
                "raw_secret_values_recorded": False,
            }
            blockers.append("semantic_teacher_object_store_cleanup_failed")
        try:
            scoped_after = dict(
                provider.billable_inventory(name_prefix=NAME_PREFIX)
            )
        except Exception as exc:  # noqa: BLE001 - retain typed zero gap
            scoped_after = {
                "api_confirmed": False,
                "live_resource_count": None,
                "resources": [],
                "error_type": type(exc).__name__,
            }
            blockers.append("semantic_teacher_scoped_inventory_failed")
        try:
            global_after = dict(provider.billable_inventory(name_prefix=""))
        except Exception as exc:  # noqa: BLE001 - retain typed zero gap
            global_after = {
                "api_confirmed": False,
                "live_resource_count": None,
                "resources": [],
                "error_type": type(exc).__name__,
            }
            blockers.append("semantic_teacher_global_inventory_failed")
        provider_zero = _api_zero(scoped_after) and _api_zero(global_after)
        teardown_passed = bool(
            provider_zero
            and cleanup.get("all_objects_absent") is True
            and (
                instance_id is None
                or terminate_result.get("status")
                in {"stopped", "terminated", "deleted"}
            )
        )
        teardown = _write_receipt(
            root / "teardown_receipt.json",
            {
                "schema_version": TEARDOWN_SCHEMA_VERSION,
                "status": "PASS" if teardown_passed else "FAIL",
                "provider": "vast",
                "authorization_digest": authority.get("authorization_digest"),
                "bundle_sha256": (receipt.get("bundle") or {}).get("sha256"),
                "instance_id": instance_id,
                "terminate_result": terminate_result,
                "all_staged_objects_absent": cleanup.get("all_objects_absent"),
                "scoped_provider_zero": _api_zero(scoped_after),
                "global_provider_zero": _api_zero(global_after),
                "continuing_spend_from_this_run": not provider_zero,
                "timestamp": utc_now_iso(),
            },
            digest_field="teardown_digest",
        )
        if instance_id is not None and teardown_passed:
            close_pending_teardown(pending_path, teardown)
        elif (
            instance_id is None
            and launch_result.get("allocation_outcome_ambiguous") is True
            and provider_zero
        ):
            cancel_pending_teardown(
                pending_path,
                reason="provider_zero_resolved_ambiguous_create",
                evidence={"teardown_digest": teardown["teardown_digest"]},
            )
            pending_resolved = True
        elif instance_id is None and not pending_resolved:
            try:
                cancel_pending_teardown(
                    pending_path,
                    reason="provider_allocation_not_started",
                    evidence={"provider_mutations_performed": provider_mutations},
                )
                pending_resolved = True
            except Exception as exc:  # noqa: BLE001 - retain pending gap
                blockers.append(
                    f"semantic_teacher_pending_teardown_cancel_failed:{redacted_failure_detail(exc)}"
                )
        if watchdog_closer is None:
            watchdog_close = {
                "status": "blocked",
                "blockers": ["semantic_teacher_watchdog_closer_missing"],
                "raw_secret_values_recorded": False,
            }
        else:
            try:
                watchdog_close = dict(
                    watchdog_closer(
                        instance_ids=[instance_id] if instance_id else [],
                        provider_teardown_completed=provider_zero,
                        provider_allocation_impossible=(
                            instance_id is None
                            and launch_result.get("allocation_outcome_ambiguous")
                            is not True
                        ),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - terminal evidence must survive
                watchdog_close = {
                    "status": "blocked",
                    "blockers": [
                        f"semantic_teacher_watchdog_close_failed:{redacted_failure_detail(exc)}"
                    ],
                    "raw_secret_values_recorded": False,
                }
        write_json(root / "independent_watchdog.json", watchdog_close)
        reconciliation_after = build_paid_provider_lane_reconciliation(
            provider="vast",
            lane=PAID_LANE,
            provider_inventory=global_after,
            open_pending_teardowns=load_pending_teardowns(registry_dir=pending_dir),
        )
        release = release_paid_provider_lane_lease(
            lease,
            reason="semantic_teacher_image_edit_terminal",
            provider_mutation_started=instance_id is not None,
            terminal_reconciliation=reconciliation_after,
            lease_dir=lease_dir,
        )
        if not teardown_passed:
            blockers.append("semantic_teacher_teardown_verification_failed")
        if instance_id is not None and release.get("released") is not True:
            blockers.append("semantic_teacher_paid_lane_release_blocked")

    runtime_output_root = root / "runtime_output"
    terminal_result_path = runtime_output_root / f"{RUNTIME_RESULT_SCHEMA_VERSION}.json"
    archive_path = root / "semantic_teacher_image_edit_runtime_output.zip"
    scan_archive_path = archive_path
    runtime_media_gap: dict[str, Any] | None = None
    if instance_id is not None and runtime_result is None:
        runtime_media_gap, terminal_result_path = _materialize_runtime_media_gap(
            runtime_output_root=runtime_output_root,
            authority=authority,
            receipt=receipt,
            instance_id=instance_id,
            gap_type=runtime_gap_type or "runtime_output_missing",
            reason_code=runtime_gap_reason or "runtime_result_missing_after_allocation",
        )
        scan_archive_path = _materialize_failure_archive(
            runtime_output_root=runtime_output_root,
            output_path=root
            / "semantic_teacher_image_edit_runtime_failure_artifacts.zip",
        )

    duration = max(0.0, float(clock()) - started_at)
    price = float(preflight.get("on_demand_price_usd_per_hour") or 0)
    vast_actual = launch_result.get("actual_cost_usd")
    vast_actual_valid = bool(
        isinstance(vast_actual, (int, float))
        and not isinstance(vast_actual, bool)
        and math.isfinite(float(vast_actual))
        and float(vast_actual) >= 0
    )
    runtime_billing = (
        runtime_result.get("provider_billing")
        if isinstance(runtime_result, Mapping)
        else None
    )
    runtime_billing = runtime_billing if isinstance(runtime_billing, Mapping) else {}
    hosted_actual = runtime_billing.get("actual_cost_usd")
    hosted_actual_valid = bool(
        isinstance(hosted_actual, (int, float))
        and not isinstance(hosted_actual, bool)
        and math.isfinite(float(hosted_actual))
        and float(hosted_actual) >= 0
    )
    vast_ledger_cost = (
        float(vast_actual)
        if vast_actual_valid
        else (float(vast_spend_upper_bound) if instance_id else 0.0)
    )
    attempted_request_count = (
        int(runtime_result.get("attempted_request_count"))
        if isinstance(runtime_result, Mapping)
        else None
    )
    maximum_cost_per_request = float(
        authority.get("maximum_cost_per_request_usd") or 0
    )
    editor_request_cost = (
        0.0
        if instance_id is None
        else (
            float(hosted_actual)
            if hosted_actual_valid
            else (
                attempted_request_count * maximum_cost_per_request
                if runtime_result is not None
                else float(hosted_spend_upper_bound)
            )
        )
    )
    hosted_ledger_cost = editor_request_cost
    compute_cost = vast_ledger_cost
    cost = editor_request_cost + compute_cost
    if cost > float(spend_cap):
        blockers.append("semantic_teacher_budget_exhausted")
    if runtime_media_gap is not None:
        billing_attempt_fields: dict[str, Any] = {
            "status": "conservative_upper_bound_runtime_result_missing",
            "attempted_request_count_known": False,
            "attempted_request_count_upper_bound": authority.get("camera_count"),
            "editor_request_cost_basis": (
                "full_authorized_upper_bound_due_to_unknown_attempt_count"
            ),
        }
    else:
        billing_attempt_fields = {
            "status": "completed",
            "attempted_request_count_known": True,
            "attempted_request_count": attempted_request_count or 0,
            "editor_request_cost_basis": (
                "provider_actual"
                if hosted_actual_valid
                else "attempted_request_count_times_authorized_maximum"
            ),
        }
    billing = _write_receipt(
        root / "billing_receipt.json",
        {
            "schema_version": BILLING_SCHEMA_VERSION,
            **billing_attempt_fields,
            "provider": "vast",
            "authority_digest": authority.get("authorization_digest"),
            "bundle_sha256": (receipt.get("bundle") or {}).get("sha256"),
            "runtime_request_digest": receipt.get("runtime_request_digest"),
            "backend_entry_digest": receipt.get("backend_entry_digest"),
            "pricing_binding_digest": authority.get("pricing_binding_digest"),
            "run_instance_id": instance_id or "",
            "allocation_count": allocation_count,
            "automatic_retry_count": 0,
            "duration_seconds": duration,
            "hourly_rate_usd": price if instance_id else 0.0,
            "vast_actual_cost_usd": float(vast_actual) if vast_actual_valid else None,
            "vast_spend_upper_bound_usd": float(vast_spend_upper_bound),
            "vast_ledger_cost_usd": vast_ledger_cost,
            "hosted_editor_actual_cost_usd": (
                float(hosted_actual) if hosted_actual_valid else None
            ),
            "hosted_editor_spend_upper_bound_usd": float(
                hosted_spend_upper_bound
            ),
            "hosted_editor_ledger_cost_usd": hosted_ledger_cost,
            "maximum_cost_per_request_usd": maximum_cost_per_request,
            "editor_request_cost_usd": editor_request_cost,
            "compute_cost_usd": compute_cost,
            "actual_total_cost_usd": (
                float(vast_actual) + float(hosted_actual)
                if vast_actual_valid and hosted_actual_valid
                else None
            ),
            "cost_usd": cost,
            "ledger_basis": (
                "actual"
                if vast_actual_valid and hosted_actual_valid
                else "conservative_upper_bound_for_missing_actual_component"
            ),
            "hard_total_spend_cap_usd": float(spend_cap),
            "within_cap": cost <= float(spend_cap),
            "raw_secret_values_recorded": False,
        },
        digest_field="billing_digest",
    )
    provider_zero = _api_zero(scoped_after) and _api_zero(global_after)
    scoped_inventory = {
        **scoped_after,
        "provider": "vast",
        "queried_instance_ids": [instance_id] if instance_id else [],
        "absent_instance_ids": [instance_id] if instance_id and provider_zero else [],
        "raw_secret_values_recorded": False,
    }
    global_inventory = {
        **global_after,
        "provider": "vast",
        "raw_secret_values_recorded": False,
    }
    scoped_inventory_path = root / "scoped_provider_inventory.json"
    global_inventory_path = root / "global_provider_inventory.json"
    write_json(scoped_inventory_path, scoped_inventory)
    write_json(global_inventory_path, global_inventory)
    secret_redaction: dict[str, Any] | None = None
    secret_redaction_path = root / "secret_redaction.json"
    if (
        instance_id
        and scan_archive_path.is_file()
        and runtime_output_root.is_dir()
        and terminal_result_path.is_file()
    ):
        try:
            secret_redaction = _materialize_secret_redaction(
                output_path=secret_redaction_path,
                runtime_output_root=runtime_output_root,
                terminal_result_path=terminal_result_path,
                archive_path=scan_archive_path,
                authority_digest=str(authority.get("authorization_digest") or ""),
                bundle_sha256=str((receipt.get("bundle") or {}).get("sha256") or ""),
                instance_id=instance_id,
                secret_values=(token, input_url, output_put_url, output_get_url),
            )
        except (OSError, SemanticTeacherImageEditVastError) as exc:
            blockers.append(f"semantic_teacher_secret_redaction_failed:{exc}")
    if secret_redaction is not None and secret_redaction.get("status") != "passed":
        blockers.append("semantic_teacher_secret_redaction_failed")
    provider_zero_receipt: dict[str, Any] | None = None
    if (
        instance_id
        and (runtime_result is not None or runtime_media_gap is not None)
        and secret_redaction is not None
        and terminal_result_path.is_file()
    ):
        try:
            provider_zero_receipt = materialize_semantic_teacher_provider_zero_receipt(
                authority_path=authority_file,
                bundle_receipt_path=receipt_file,
                terminal_result_path=terminal_result_path,
                billing_receipt_path=root / "billing_receipt.json",
                scoped_inventory_path=scoped_inventory_path,
                global_inventory_path=global_inventory_path,
                object_store_cleanup_path=(
                    staging_dir / "wam_provider_object_store_cleanup.json"
                ),
                independent_watchdog_path=root / "independent_watchdog.json",
                secret_redaction_path=secret_redaction_path,
                stdout_log_path=root / "runtime_output/runtime_stdout.log",
                stderr_log_path=root / "runtime_output/runtime_stderr.log",
                output_path=root / "provider_zero_receipt.json",
            )
        except (OSError, ValueError) as exc:
            blockers.append(f"semantic_teacher_provider_zero_receipt_failed:{exc}")
    elif instance_id is None:
        provider_zero_receipt = _write_receipt(
            root / "no_allocation_provider_zero_receipt.json",
            {
                "schema_version": (
                    "semantic_teacher_image_edit_no_allocation_provider_zero.v1"
                ),
                "status": "provider_zero" if provider_zero else "provider_nonzero",
                "provider": "vast",
                "run_instance_ids": [],
                "scoped_inventory": scoped_inventory,
                "global_inventory": global_inventory,
                "provider_zero_api_confirmed": provider_zero,
                "all_staged_objects_absent": cleanup.get("all_objects_absent"),
                "independent_watchdog_status": watchdog_close.get("status"),
                "total_cost_usd": 0.0,
                "continuing_spend_from_this_run": not provider_zero,
                "raw_secret_values_recorded": False,
                "confirmed_at": utc_now_iso(),
            },
            digest_field="provider_zero_digest",
        )
    retained_result: dict[str, Any] | None = None
    if runtime_result is not None and provider_zero_receipt is not None:
        try:
            retained_result = materialize_semantic_teacher_image_edit_result(
                runtime_output_root=root / "runtime_output",
                runtime_request_path=runtime_request_path,
                bundle_receipt_path=receipt_file,
                authority_path=authority_file,
                billing_receipt_path=root / "billing_receipt.json",
                scoped_inventory_path=scoped_inventory_path,
                global_inventory_path=global_inventory_path,
                object_store_cleanup_path=(
                    staging_dir / "wam_provider_object_store_cleanup.json"
                ),
                watchdog_receipt_path=root / "independent_watchdog.json",
                secret_redaction_path=secret_redaction_path,
                provider_zero_path=root / "provider_zero_receipt.json",
                expected_task_count=int(receipt["task_count"]),
                expected_camera_count=int(receipt["camera_count"]),
                output_path=root
                / "semantic_teacher_image_edit_result_import.v1.json",
            )
        except (OSError, ValueError) as exc:
            blockers.append(f"semantic_teacher_result_import_failed:{exc}")
    no_allocation_occurred = bool(
        instance_id is None
        and allocation_count == 0
        and provider_mutations == 0
        and launch_result.get("allocation_outcome_ambiguous") is not True
    )
    watchdog_closed = bool(
        watchdog_close.get("status") == "provider_terminal"
        or (
            watchdog_close.get("status") == "cancelled_no_allocation"
            and no_allocation_occurred
        )
    )
    if not watchdog_closed:
        blockers.append("semantic_teacher_independent_watchdog_not_closed")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("semantic_teacher_object_store_cleanup_not_proven")
    if not provider_zero:
        blockers.append("semantic_teacher_provider_zero_not_proven")
    result = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": (
            "completed"
            if runtime_result is not None and retained_result is not None and not blockers
            else "blocked"
        ),
        "source_commit_sha": checkout_source_commit,
        "authorization_digest": authority.get("authorization_digest"),
        "authorization_consumption": dict(consumption),
        "bundle_sha256": (receipt.get("bundle") or {}).get("sha256"),
        "bundle_size_bytes": (receipt.get("bundle") or {}).get("size_bytes"),
        "runtime_request_digest": receipt.get("runtime_request_digest"),
        "backend_entry_digest": receipt.get("backend_entry_digest"),
        "adapter_id": runtime_binding["adapter_id"],
        "model_snapshot": runtime_binding["model_snapshot"],
        "task_camera_order": runtime_binding["task_camera_order"],
        "task_camera_order_digest": runtime_binding[
            "task_camera_order_digest"
        ],
        "task_count": receipt.get("task_count"),
        "camera_count": receipt.get("camera_count"),
        "provider": "vast",
        "runtime_image_identity": runtime_image_identity,
        "excluded_machine_ids": sorted(
            set(int(value) for value in excluded_machine_ids)
        ),
        "instance_id": instance_id,
        "allocation_count": allocation_count,
        "maximum_create_attempts": 1,
        "create_attempt_count": launch_result.get("create_attempt_count", 0),
        "automatic_retry_count": 0,
        "retry_cap": 0,
        "provider_mutations_performed": provider_mutations,
        "provider_mutation_outcome_ambiguous": bool(
            launch_result.get("allocation_outcome_ambiguous") is True
        ),
        "runtime_result_digest": (
            runtime_result.get("result_digest") if runtime_result else None
        ),
        "runtime_media_gap_digest": (
            runtime_media_gap.get("gap_digest") if runtime_media_gap else None
        ),
        "result_import_digest": (
            retained_result.get("result_import_digest") if retained_result else None
        ),
        "billing_digest": billing["billing_digest"],
        "cost_usd": cost,
        "teardown_digest": teardown["teardown_digest"],
        "provider_zero_digest": (
            provider_zero_receipt.get("provider_zero_digest")
            if provider_zero_receipt
            else None
        ),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "provider_zero_verified": provider_zero,
        "continuing_spend_from_this_run": not provider_zero,
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "raw_secret_values_recorded": False,
        "visual_reviewed": False,
        "appearance_qualified": False,
        "physical_evidence_claimed": False,
    }
    evidence_paths = {
        "billing_receipt": root / "billing_receipt.json",
        "provider_zero_receipt": (
            root / "provider_zero_receipt.json"
            if instance_id
            else root / "no_allocation_provider_zero_receipt.json"
        ),
        "teardown_receipt": root / "teardown_receipt.json",
        "independent_watchdog": root / "independent_watchdog.json",
        "secret_redaction": root / "secret_redaction.json",
        "scoped_provider_inventory": scoped_inventory_path,
        "global_provider_inventory": global_inventory_path,
        "object_store_cleanup": (
            staging_dir / "wam_provider_object_store_cleanup.json"
        ),
        "runtime_output": runtime_output_root,
        "runtime_output_archive": archive_path,
        "runtime_failure_archive": (
            root / "semantic_teacher_image_edit_runtime_failure_artifacts.zip"
        ),
        "result_import": (
            root / "semantic_teacher_image_edit_result_import.v1.json"
        ),
    }
    return _seal_terminal_execution(
        root=root,
        result=result,
        instance_ids=[instance_id] if instance_id else [],
        provider_zero_verified=provider_zero,
        continuing_spend=not provider_zero,
        teardown_actions=[terminate_result] if instance_id else [],
        evidence_paths=evidence_paths,
    )


def run_semantic_teacher_image_edit_vast(
    args: Any,
    *,
    checkout_commit: str,
    preflight: Mapping[str, Any],
    provider: GpuRenderProvider,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    watchdog_closer: Callable[..., Mapping[str, Any]],
    object_store_stager: Callable[..., Mapping[str, Any]] = (
        stage_wam_provider_bundle_object_store
    ),
    object_store_cleaner: Callable[..., Mapping[str, Any]] = (
        cleanup_staged_wam_provider_objects
    ),
    result_fetcher: Callable[[str], bytes] = _default_result_fetcher,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    watchdog_validator: Callable[[Mapping[str, Any], float, int], bool]
    | None = None,
    watchdog_instance_binder: Callable[[Path, int], None] = (
        write_started_vast_instance_id
    ),
) -> dict[str, Any]:
    """Allocator-facing adapter for ``gpu-canary semantic-teacher-image-edit``.

    The shared allocator owns argument parsing, checkout cleanliness, admission,
    and watchdog arming.  This hook owns the lifecycle beginning with the live
    preflight snapshot and ending with provider/object-store zero.
    """

    return _execute_semantic_teacher_image_edit_vast(
        authority_path=args.semantic_teacher_attempt_authority,
        bundle_path=args.semantic_teacher_bundle,
        bundle_receipt_path=args.semantic_teacher_bundle_receipt,
        checkout_source_commit=checkout_commit,
        job_dir=args.semantic_teacher_job_dir,
        token_file=args.semantic_teacher_token_file,
        runtime_image_identity=args.semantic_teacher_runtime_image_identity,
        preflight=preflight,
        provider=provider,
        paid_resource_admission_grant=paid_resource_admission_grant,
        object_store_stager=object_store_stager,
        object_store_cleaner=object_store_cleaner,
        result_fetcher=result_fetcher,
        sleeper=sleeper,
        clock=clock,
        watchdog_validator=watchdog_validator,
        watchdog_instance_binder=watchdog_instance_binder,
        watchdog_closer=watchdog_closer,
        excluded_machine_ids=getattr(
            args, "semantic_teacher_excluded_machine_id", []
        ),
    )


__all__ = [
    "EXECUTION_SCHEMA_VERSION",
    "NAME_PREFIX",
    "PAID_LANE",
    "PROBE_KIND",
    "SemanticTeacherImageEditVastError",
    "run_semantic_teacher_image_edit_vast",
]
