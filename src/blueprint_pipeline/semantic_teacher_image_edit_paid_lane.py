"""Fail-closed, provider-neutral seams for the semantic-teacher paid lane.

This module deliberately performs no allocation or network access. The shared
``paid_resource_allocator gpu-canary`` entrypoint can call these seams for its
dry run, no-allocation closeout, retained-output import, and post-teardown
provider-zero receipt.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .semantic_teacher_image_edit_paid_authority import (
    validate_semantic_teacher_image_edit_paid_authority,
)
from .semantic_teacher_image_edit_worker import RUNTIME_RESULT_SCHEMA_VERSION


DRY_RUN_SCHEMA_VERSION = "semantic_teacher_image_edit_allocator_dry_run.v1"
PROVIDER_ZERO_SCHEMA_VERSION = "semantic_teacher_image_edit_provider_zero.v1"
RESULT_IMPORT_SCHEMA_VERSION = "semantic_teacher_image_edit_result_import.v1"
NO_ALLOCATION_SCHEMA_VERSION = "semantic_teacher_image_edit_no_allocation_closeout.v1"
WATCHDOG_SCHEMA_VERSION = "semantic_teacher_image_edit_watchdog.v1"
CLEANUP_SCHEMA_VERSION = "semantic_teacher_image_edit_object_store_cleanup.v1"
SECRET_REDACTION_SCHEMA_VERSION = "semantic_teacher_image_edit_secret_redaction.v1"
BILLING_SCHEMA_VERSION = "semantic_teacher_image_edit_billing.v1"
RUNTIME_MEDIA_GAP_SCHEMA_VERSION = "semantic_teacher_image_edit_runtime_media_gap.v1"
_FRAME_PATH = re.compile(r"tasks/[^/\\]+/[0-9]{5}\.png")


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
        raise ValueError(code) from exc
    if path.is_symlink() or not isinstance(value, dict):
        raise ValueError(code)
    return value


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    result["relative_path" if root is not None else "path"] = (
        path.relative_to(root).as_posix() if root is not None else str(path)
    )
    return result


def _api_zero(inventory: Any) -> bool:
    return (
        isinstance(inventory, Mapping)
        and inventory.get("api_confirmed") is True
        and inventory.get("live_resource_count") == 0
        and inventory.get("resources") == []
    )


def _billing_within_authority(
    billing: Mapping[str, Any], authority: Mapping[str, Any]
) -> bool:
    fields = (
        billing.get("editor_request_cost_usd"),
        billing.get("compute_cost_usd"),
        billing.get("cost_usd"),
        billing.get("maximum_cost_per_request_usd"),
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in fields
    ):
        return False
    editor, compute, total, per_request = (float(value) for value in fields)
    common_valid = (
        math.isclose(editor + compute, total, rel_tol=0, abs_tol=1e-9)
        and compute <= float(authority.get("maximum_compute_cost_usd") or -1)
        and total <= float(authority.get("hard_total_spend_cap_usd") or -1)
    )
    if not common_valid:
        return False
    attempted_known = billing.get("attempted_request_count_known")
    if attempted_known is False:
        attempted_upper_bound = billing.get("attempted_request_count_upper_bound")
        camera_count = authority.get("camera_count")
        hosted_upper_bound = authority.get("hosted_editor_spend_upper_bound_usd")
        return (
            "attempted_request_count" not in billing
            and billing.get("status")
            == "conservative_upper_bound_runtime_result_missing"
            and isinstance(attempted_upper_bound, int)
            and not isinstance(attempted_upper_bound, bool)
            and attempted_upper_bound == camera_count
            and _finite_nonnegative(hosted_upper_bound)
            and math.isclose(
                editor, float(hosted_upper_bound), rel_tol=0, abs_tol=1e-9
            )
            and math.isclose(
                editor,
                attempted_upper_bound * per_request,
                rel_tol=0,
                abs_tol=1e-9,
            )
            and billing.get("editor_request_cost_basis")
            == "full_authorized_upper_bound_due_to_unknown_attempt_count"
        )
    attempted = billing.get("attempted_request_count")
    return (
        attempted_known in {None, True}
        and isinstance(attempted, int)
        and not isinstance(attempted, bool)
        and attempted >= 0
        and editor <= attempted * per_request
    )


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _runtime_media_gap_valid(
    gap: Mapping[str, Any],
    *,
    authority_digest: Any,
    bundle_sha256: Any,
    runtime_request_digest: Any,
    backend_entry_digest: Any,
    instance_id: str,
    camera_count: Any,
) -> bool:
    partial_inventory = gap.get("partial_png_inventory")
    if not isinstance(partial_inventory, list):
        return False
    seen_paths: set[str] = set()
    for item in partial_inventory:
        if not isinstance(item, Mapping):
            return False
        relative_path = item.get("relative_path")
        if (
            not isinstance(relative_path, str)
            or not _FRAME_PATH.fullmatch(relative_path)
            or relative_path in seen_paths
            or not isinstance(item.get("size_bytes"), int)
            or isinstance(item.get("size_bytes"), bool)
            or item["size_bytes"] <= 0
            or not isinstance(item.get("sha256"), str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", item["sha256"])
        ):
            return False
        seen_paths.add(relative_path)
    return (
        gap.get("schema_version") == RUNTIME_MEDIA_GAP_SCHEMA_VERSION
        and gap.get("status") == "blocked_runtime_result_missing"
        and gap.get("gap_type")
        in {"runtime_timeout", "runtime_output_missing", "runtime_output_malformed"}
        and isinstance(gap.get("reason_code"), str)
        and bool(gap["reason_code"].strip())
        and gap.get("authority_digest") == authority_digest
        and gap.get("bundle_sha256") == bundle_sha256
        and gap.get("runtime_request_digest") == runtime_request_digest
        and gap.get("backend_entry_digest") == backend_entry_digest
        and gap.get("run_instance_id") == instance_id
        and gap.get("attempted_request_count_known") is False
        and "attempted_request_count" not in gap
        and gap.get("attempted_request_count_upper_bound") == camera_count
        and gap.get("runtime_result_present") is False
        and gap.get("media_complete") is False
        and gap.get("teacher_frames_qualified") is False
        and gap.get("raw_secret_values_recorded") is False
        and gap.get("gap_digest")
        == canonical_digest(gap, digest_field="gap_digest")
    )


def prepare_semantic_teacher_image_edit_allocator_dry_run(
    *,
    authority_path: str | Path,
    bundle_path: str | Path,
    bundle_receipt_path: str | Path,
    checkout_source_commit: str,
    live_inventory: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate immutable admission without token lookup, staging, or allocation."""

    authority_file = Path(authority_path).expanduser().resolve()
    receipt_file = Path(bundle_receipt_path).expanduser().resolve()
    authority = _read(authority_file, code="semantic_teacher_dry_run_authority_invalid")
    receipt = _read(receipt_file, code="semantic_teacher_dry_run_receipt_invalid")
    validate_semantic_teacher_image_edit_paid_authority(
        authority,
        bundle_path=bundle_path,
        bundle_receipt=receipt,
        source_commit_sha=checkout_source_commit,
        backend_entry_digest=str(authority.get("backend_entry_digest") or ""),
        task_count=int(authority.get("task_count") or 0),
        camera_count=int(authority.get("camera_count") or 0),
        maximum_hourly_rate_usd=float(authority.get("maximum_hourly_rate_usd") or 0),
        hard_total_spend_cap_usd=float(
            authority.get("hard_total_spend_cap_usd") or 0
        ),
        hard_ttl_seconds=int(authority.get("hard_ttl_seconds") or 0),
    )
    if not _api_zero(live_inventory):
        raise ValueError("semantic_teacher_dry_run_provider_inventory_not_zero")
    rehearsal = receipt.get("rehearsal")
    if (
        not isinstance(rehearsal, Mapping)
        or rehearsal.get("status") != "passed"
        or rehearsal.get("token_lookup_performed") is not False
        or rehearsal.get("upload_performed") is not False
        or rehearsal.get("provider_mutations_performed") != 0
    ):
        raise ValueError("semantic_teacher_dry_run_rehearsal_invalid")
    result: dict[str, Any] = {
        "schema_version": DRY_RUN_SCHEMA_VERSION,
        "status": "dry_run_ready",
        "source_commit_sha": checkout_source_commit,
        "authorization_digest": authority["authorization_digest"],
        "bundle_sha256": authority["bundle"]["sha256"],
        "bundle_size_bytes": authority["bundle"]["size_bytes"],
        "backend_entry_digest": authority["backend_entry_digest"],
        "task_count": authority["task_count"],
        "camera_count": authority["camera_count"],
        "maximum_provider_allocations": 1,
        "automatic_retry_count": 0,
        "provider_inventory_api_zero": True,
        "token_lookup_performed": False,
        "object_store_staging_performed": False,
        "watchdog_armed": False,
        "provider_mutations_performed": 0,
        "paid_inference_performed": False,
        "dry_run_digest": "",
    }
    result["dry_run_digest"] = canonical_digest(result, digest_field="dry_run_digest")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError("semantic_teacher_dry_run_output_exists")
    ensure_dir(destination.parent)
    write_json(destination, result)
    return result


def materialize_semantic_teacher_no_allocation_closeout(
    *,
    dry_run_path: str | Path,
    watchdog_closeout_path: str | Path,
    reason: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Retain a terminal teardown when allocation never became possible."""

    dry_run = _read(
        Path(dry_run_path).expanduser().resolve(),
        code="semantic_teacher_no_allocation_dry_run_invalid",
    )
    if (
        dry_run.get("schema_version") != DRY_RUN_SCHEMA_VERSION
        or dry_run.get("dry_run_digest")
        != canonical_digest(dry_run, digest_field="dry_run_digest")
        or dry_run.get("provider_mutations_performed") != 0
        or not reason.strip()
    ):
        raise ValueError("semantic_teacher_no_allocation_input_invalid")
    watchdog_path = Path(watchdog_closeout_path).expanduser().resolve()
    watchdog = _read(
        watchdog_path, code="semantic_teacher_no_allocation_watchdog_invalid"
    )
    inventories = [
        watchdog.get("initial_inventory"),
        watchdog.get("initial_global_inventory"),
        watchdog.get("final_inventory"),
        watchdog.get("final_global_inventory"),
    ]
    if (
        watchdog.get("schema_version") != "vast_independent_watchdog_handoff.v1"
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("watchdog_armed_before_allocation") is not True
        or watchdog.get("provider_absence_confirmed") is not True
        or watchdog.get("provider_mutations_performed") != 0
        or watchdog.get("raw_secret_values_recorded") is not False
        or not all(_api_zero(inventory) for inventory in inventories)
    ):
        raise ValueError("semantic_teacher_no_allocation_watchdog_invalid")
    result: dict[str, Any] = {
        "schema_version": NO_ALLOCATION_SCHEMA_VERSION,
        "status": "closed_without_allocation",
        "reason": reason.strip(),
        "source_dry_run_digest": dry_run["dry_run_digest"],
        "watchdog_closeout": _record(watchdog_path),
        "allocation_count": 0,
        "watchdog_status": "provider_terminal",
        "double_lane_and_global_api_zero_confirmed": True,
        "object_store_objects_created": 0,
        "all_staged_objects_absent": True,
        "provider_mutations_performed": 0,
        "cost_usd": 0.0,
        "continuing_spend_from_this_run": False,
        "closed_at": utc_now_iso(),
        "closeout_digest": "",
    }
    result["closeout_digest"] = canonical_digest(result, digest_field="closeout_digest")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError("semantic_teacher_no_allocation_output_exists")
    ensure_dir(destination.parent)
    write_json(destination, result)
    return result


def materialize_semantic_teacher_provider_zero_receipt(
    *,
    authority_path: str | Path,
    bundle_receipt_path: str | Path,
    terminal_result_path: str | Path,
    billing_receipt_path: str | Path,
    scoped_inventory_path: str | Path,
    global_inventory_path: str | Path,
    object_store_cleanup_path: str | Path,
    independent_watchdog_path: str | Path,
    secret_redaction_path: str | Path,
    stdout_log_path: str | Path,
    stderr_log_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind API-confirmed resource zero, staged-object zero, watchdog, and cost."""

    paths = {
        "authority": Path(authority_path).expanduser().resolve(),
        "bundle_receipt": Path(bundle_receipt_path).expanduser().resolve(),
        "terminal_result": Path(terminal_result_path).expanduser().resolve(),
        "billing": Path(billing_receipt_path).expanduser().resolve(),
        "scoped_inventory": Path(scoped_inventory_path).expanduser().resolve(),
        "global_inventory": Path(global_inventory_path).expanduser().resolve(),
        "object_store_cleanup": Path(object_store_cleanup_path).expanduser().resolve(),
        "independent_watchdog": Path(independent_watchdog_path).expanduser().resolve(),
        "secret_redaction": Path(secret_redaction_path).expanduser().resolve(),
        "stdout_log": Path(stdout_log_path).expanduser().resolve(),
        "stderr_log": Path(stderr_log_path).expanduser().resolve(),
    }
    values = {
        role: _read(path, code=f"semantic_teacher_provider_zero_{role}_invalid")
        for role, path in paths.items()
        if role not in {"stdout_log", "stderr_log"}
    }
    if any(
        path.is_symlink() or not path.is_file()
        for role, path in paths.items()
        if role in {"stdout_log", "stderr_log"}
    ):
        raise ValueError("semantic_teacher_provider_zero_logs_invalid")
    authority = values["authority"]
    bundle = values["bundle_receipt"]
    terminal = values["terminal_result"]
    billing = values["billing"]
    scoped_inventory = values["scoped_inventory"]
    global_inventory = values["global_inventory"]
    object_store_cleanup = values["object_store_cleanup"]
    independent_watchdog = values["independent_watchdog"]
    secret_redaction = values["secret_redaction"]
    authority_digest = authority.get("authorization_digest")
    bundle_sha256 = (authority.get("bundle") or {}).get("sha256")
    runtime_request_digest = authority.get("runtime_request_digest")
    backend_entry_digest = authority.get("backend_entry_digest")
    instance_ids = [str(value) for value in independent_watchdog.get("instance_ids") or []]
    instance_id = instance_ids[0] if len(instance_ids) == 1 else ""
    terminal_is_runtime_result = (
        terminal.get("schema_version") == RUNTIME_RESULT_SCHEMA_VERSION
    )
    terminal_is_runtime_gap = (
        terminal.get("schema_version") == RUNTIME_MEDIA_GAP_SCHEMA_VERSION
    )
    terminal_evidence_digest = (
        terminal.get("result_digest")
        if terminal_is_runtime_result
        else terminal.get("gap_digest")
    )
    terminal_valid = (
        terminal_is_runtime_result
        and terminal.get("result_digest")
        == canonical_digest(terminal, digest_field="result_digest")
        and terminal.get("source_runtime_request_digest") == runtime_request_digest
        and terminal.get("backend_entry_digest") == backend_entry_digest
    ) or (
        terminal_is_runtime_gap
        and _runtime_media_gap_valid(
            terminal,
            authority_digest=authority_digest,
            bundle_sha256=bundle_sha256,
            runtime_request_digest=runtime_request_digest,
            backend_entry_digest=backend_entry_digest,
            instance_id=instance_id,
            camera_count=authority.get("camera_count"),
        )
    )
    billing_attempt_binding_valid = (
        terminal_is_runtime_result
        and billing.get("attempted_request_count_known") in {None, True}
        and billing.get("attempted_request_count")
        == terminal.get("attempted_request_count")
    ) or (
        terminal_is_runtime_gap
        and billing.get("attempted_request_count_known") is False
        and "attempted_request_count" not in billing
        and billing.get("attempted_request_count_upper_bound")
        == terminal.get("attempted_request_count_upper_bound")
    )
    total_cost_usd = billing.get("cost_usd")
    cleanup_valid = (
        object_store_cleanup.get("schema_version")
        == "wam_provider_object_store_cleanup.v1"
        and object_store_cleanup.get("status") == "completed"
        and object_store_cleanup.get("all_objects_absent") is True
        and object_store_cleanup.get("signed_url_files_removed") is True
        and object_store_cleanup.get("blockers") == []
        and object_store_cleanup.get("raw_secret_values_recorded") is False
    )
    watchdog_valid = (
        independent_watchdog.get("schema_version")
        == "vast_independent_watchdog_handoff.v1"
        and independent_watchdog.get("status") == "provider_terminal"
        and independent_watchdog.get("watchdog_armed_before_allocation") is True
        and independent_watchdog.get("provider_absence_confirmed") is True
        and independent_watchdog.get("raw_secret_values_recorded") is False
    )
    redaction_valid = (
        secret_redaction.get("schema_version") == SECRET_REDACTION_SCHEMA_VERSION
        and secret_redaction.get("status") == "passed"
        and secret_redaction.get("authority_digest") == authority_digest
        and secret_redaction.get("bundle_sha256") == bundle_sha256
        and secret_redaction.get("run_instance_id") == instance_id
        and secret_redaction.get("stdout_sha256") == _sha256(paths["stdout_log"])
        and secret_redaction.get("stderr_sha256") == _sha256(paths["stderr_log"])
        and secret_redaction.get("stdout_scanned") is True
        and secret_redaction.get("stderr_scanned") is True
        and secret_redaction.get("secret_values_found") is False
        and secret_redaction.get("raw_secret_values_recorded") is False
        and secret_redaction.get("redaction_digest")
        == canonical_digest(secret_redaction, digest_field="redaction_digest")
    )
    inventory_ids = [
        str(value) for value in scoped_inventory.get("queried_instance_ids") or []
    ]
    absent_ids = [
        str(value) for value in scoped_inventory.get("absent_instance_ids") or []
    ]
    if (
        authority.get("authorization_digest")
        != canonical_digest(authority, digest_field="authorization_digest")
        or bundle.get("receipt_digest")
        != canonical_digest(bundle, digest_field="receipt_digest")
        or (bundle.get("bundle") or {}).get("sha256") != bundle_sha256
        or bundle.get("runtime_request_digest") != runtime_request_digest
        or bundle.get("backend_entry_digest") != backend_entry_digest
        or not terminal_valid
        or billing.get("schema_version") != BILLING_SCHEMA_VERSION
        or billing.get("status")
        not in {"completed", "conservative_upper_bound_runtime_result_missing"}
        or billing.get("authority_digest") != authority_digest
        or billing.get("bundle_sha256") != bundle_sha256
        or billing.get("runtime_request_digest") != runtime_request_digest
        or billing.get("backend_entry_digest") != backend_entry_digest
        or billing.get("run_instance_id") != instance_id
        or not billing_attempt_binding_valid
        or billing.get("pricing_binding_digest")
        != authority.get("pricing_binding_digest")
        or billing.get("maximum_cost_per_request_usd")
        != authority.get("maximum_cost_per_request_usd")
        or not _billing_within_authority(billing, authority)
        or billing.get("raw_secret_values_recorded") is not False
        or billing.get("billing_digest")
        != canonical_digest(billing, digest_field="billing_digest")
        or not _api_zero(scoped_inventory)
        or not _api_zero(global_inventory)
        or not str(scoped_inventory.get("provider") or "").strip()
        or global_inventory.get("provider") != scoped_inventory.get("provider")
        or inventory_ids != [instance_id]
        or absent_ids != [instance_id]
        or not cleanup_valid
        or not watchdog_valid
        or not redaction_valid
        or not instance_id
        or len(set(instance_ids)) != 1
        or not str(authority_digest).startswith("sha256:")
        or not str(bundle_sha256).startswith("sha256:")
        or not str(runtime_request_digest).startswith("sha256:")
        or not str(backend_entry_digest).startswith("sha256:")
        or not str(terminal_evidence_digest).startswith("sha256:")
        or isinstance(total_cost_usd, bool)
        or not isinstance(total_cost_usd, (int, float))
        or not math.isfinite(float(total_cost_usd))
        or float(total_cost_usd) < 0
    ):
        raise ValueError("semantic_teacher_provider_zero_inputs_invalid")
    result: dict[str, Any] = {
        "schema_version": PROVIDER_ZERO_SCHEMA_VERSION,
        "status": "provider_zero",
        "authority_digest": authority_digest,
        "bundle_sha256": bundle_sha256,
        "runtime_request_digest": runtime_request_digest,
        "backend_entry_digest": backend_entry_digest,
        "terminal_evidence_kind": (
            "runtime_result" if terminal_is_runtime_result else "runtime_media_gap"
        ),
        "terminal_evidence_digest": terminal_evidence_digest,
        "terminal_result_digest": (
            terminal_evidence_digest if terminal_is_runtime_result else None
        ),
        "runtime_media_gap_digest": (
            terminal_evidence_digest if terminal_is_runtime_gap else None
        ),
        "attempted_request_count_known": not terminal_is_runtime_gap,
        "attempted_request_count_upper_bound": (
            terminal.get("attempted_request_count_upper_bound")
            if terminal_is_runtime_gap
            else terminal.get("attempted_request_count")
        ),
        "run_instance_ids": instance_ids,
        "scoped_provider_inventory": _record(paths["scoped_inventory"]),
        "global_provider_inventory": _record(paths["global_inventory"]),
        "provider_zero_api_confirmed": True,
        "retained_records": {role: _record(path) for role, path in paths.items()},
        "object_store_cleanup": _record(paths["object_store_cleanup"]),
        "all_staged_objects_absent": True,
        "independent_watchdog": _record(paths["independent_watchdog"]),
        "secret_redaction": _record(paths["secret_redaction"]),
        "raw_secret_values_recorded": False,
        "total_cost_usd": float(total_cost_usd),
        "continuing_spend_from_this_run": False,
        "confirmed_at": utc_now_iso(),
        "provider_zero_digest": "",
    }
    result["provider_zero_digest"] = canonical_digest(
        result, digest_field="provider_zero_digest"
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError("semantic_teacher_provider_zero_output_exists")
    ensure_dir(destination.parent)
    write_json(destination, result)
    return result


def materialize_semantic_teacher_image_edit_result(
    *,
    runtime_output_root: str | Path,
    runtime_request_path: str | Path,
    bundle_receipt_path: str | Path,
    authority_path: str | Path,
    billing_receipt_path: str | Path,
    scoped_inventory_path: str | Path,
    global_inventory_path: str | Path,
    object_store_cleanup_path: str | Path,
    watchdog_receipt_path: str | Path,
    secret_redaction_path: str | Path,
    provider_zero_path: str | Path,
    expected_task_count: int,
    expected_camera_count: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Retain every generated PNG and closeout receipt under an exact allowlist."""

    root = Path(runtime_output_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("semantic_teacher_result_root_invalid")
    runtime_result_path = root / f"{RUNTIME_RESULT_SCHEMA_VERSION}.json"
    runtime_result = _read(
        runtime_result_path, code="semantic_teacher_runtime_result_invalid"
    )
    runtime_request_file = Path(runtime_request_path).expanduser().resolve()
    bundle_file = Path(bundle_receipt_path).expanduser().resolve()
    authority_file = Path(authority_path).expanduser().resolve()
    runtime_request = _read(
        runtime_request_file, code="semantic_teacher_runtime_request_invalid"
    )
    bundle = _read(bundle_file, code="semantic_teacher_bundle_receipt_invalid")
    authority = _read(authority_file, code="semantic_teacher_authority_invalid")
    backend = runtime_request.get("backend")
    if (
        runtime_request.get("request_digest")
        != canonical_digest(runtime_request, digest_field="request_digest")
        or not isinstance(backend, Mapping)
        or authority.get("authorization_digest")
        != canonical_digest(authority, digest_field="authorization_digest")
        or bundle.get("receipt_digest")
        != canonical_digest(bundle, digest_field="receipt_digest")
        or authority.get("bundle_receipt_digest") != bundle.get("receipt_digest")
        or (authority.get("bundle") or {}).get("sha256")
        != (bundle.get("bundle") or {}).get("sha256")
        or authority.get("runtime_request_digest") != runtime_request.get("request_digest")
        or authority.get("backend_entry_digest") != backend.get("backend_entry_digest")
        or bundle.get("runtime_request_digest") != runtime_request.get("request_digest")
        or bundle.get("backend_entry_digest") != backend.get("backend_entry_digest")
    ):
        raise ValueError("semantic_teacher_result_execution_binding_invalid")
    if (
        runtime_result.get("schema_version") != RUNTIME_RESULT_SCHEMA_VERSION
        or runtime_result.get("result_digest")
        != canonical_digest(runtime_result, digest_field="result_digest")
        or runtime_result.get("status")
        != "completed_unreviewed_semantic_teacher_candidates"
        or runtime_result.get("task_count") != expected_task_count
        or runtime_result.get("request_count") != expected_camera_count
        or runtime_result.get("retry_count") != 0
        or runtime_result.get("raw_secret_values_recorded") is not False
        or runtime_result.get("source_runtime_request_digest")
        != runtime_request["request_digest"]
        or runtime_result.get("backend_entry_digest")
        != authority["backend_entry_digest"]
        or runtime_result.get("attempted_request_count") != expected_camera_count
        or runtime_result.get("successful_request_count") != expected_camera_count
    ):
        raise ValueError("semantic_teacher_runtime_result_invalid")
    expected_frames = {
        str(frame["semantic_teacher_frame"]["relative_path"]): frame[
            "semantic_teacher_frame"
        ]
        for task in runtime_result.get("tasks") or []
        for frame in task.get("frames") or []
    }
    observed_files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    observed_frames: dict[str, dict[str, Any]] = {}
    log_paths = {
        "stdout": root / "runtime_stdout.log",
        "stderr": root / "runtime_stderr.log",
    }
    if any(path.is_symlink() or not path.is_file() for path in log_paths.values()):
        raise ValueError("semantic_teacher_result_logs_missing")
    for path in observed_files:
        relative = path.relative_to(root).as_posix()
        if relative == runtime_result_path.name or relative in {
            "runtime_stdout.log",
            "runtime_stderr.log",
        }:
            continue
        if not _FRAME_PATH.fullmatch(relative):
            raise ValueError("semantic_teacher_result_file_not_allowlisted")
        observed_frames[relative] = _record(path, root=root)
    if (
        len(observed_frames) != expected_camera_count
        or set(observed_frames) != set(expected_frames)
        or any(
            observed_frames[name]["sha256"] != expected_frames[name].get("sha256")
            or observed_frames[name]["size_bytes"]
            != expected_frames[name].get("size_bytes")
            for name in observed_frames
        )
    ):
        raise ValueError("semantic_teacher_result_frame_inventory_invalid")
    closeout_paths = {
        "billing": Path(billing_receipt_path).expanduser().resolve(),
        "scoped_inventory": Path(scoped_inventory_path).expanduser().resolve(),
        "global_inventory": Path(global_inventory_path).expanduser().resolve(),
        "object_store_cleanup": Path(object_store_cleanup_path).expanduser().resolve(),
        "independent_watchdog": Path(watchdog_receipt_path).expanduser().resolve(),
        "secret_redaction": Path(secret_redaction_path).expanduser().resolve(),
        "provider_zero": Path(provider_zero_path).expanduser().resolve(),
    }
    closeouts = {
        name: _read(path, code=f"semantic_teacher_{name}_receipt_invalid")
        for name, path in closeout_paths.items()
    }
    billing = closeouts["billing"]
    cleanup = closeouts["object_store_cleanup"]
    scoped_inventory = closeouts["scoped_inventory"]
    global_inventory = closeouts["global_inventory"]
    watchdog = closeouts["independent_watchdog"]
    redaction = closeouts["secret_redaction"]
    zero = closeouts["provider_zero"]
    retained = zero.get("retained_records")
    if (
        billing.get("schema_version") != BILLING_SCHEMA_VERSION
        or billing.get("billing_digest")
        != canonical_digest(billing, digest_field="billing_digest")
        or billing.get("authority_digest") != authority["authorization_digest"]
        or billing.get("bundle_sha256") != authority["bundle"]["sha256"]
        or billing.get("runtime_request_digest") != runtime_request["request_digest"]
        or billing.get("backend_entry_digest") != authority["backend_entry_digest"]
        or not _billing_within_authority(billing, authority)
        or not _api_zero(scoped_inventory)
        or not _api_zero(global_inventory)
        or cleanup.get("schema_version") != "wam_provider_object_store_cleanup.v1"
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or watchdog.get("schema_version") != "vast_independent_watchdog_handoff.v1"
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("watchdog_armed_before_allocation") is not True
        or redaction.get("schema_version") != SECRET_REDACTION_SCHEMA_VERSION
        or redaction.get("redaction_digest")
        != canonical_digest(redaction, digest_field="redaction_digest")
        or redaction.get("stdout_sha256") != _sha256(log_paths["stdout"])
        or redaction.get("stderr_sha256") != _sha256(log_paths["stderr"])
        or redaction.get("secret_values_found") is not False
        or zero.get("schema_version") != PROVIDER_ZERO_SCHEMA_VERSION
        or zero.get("provider_zero_digest")
        != canonical_digest(zero, digest_field="provider_zero_digest")
        or zero.get("authority_digest") != authority["authorization_digest"]
        or zero.get("bundle_sha256") != authority["bundle"]["sha256"]
        or zero.get("runtime_request_digest") != runtime_request["request_digest"]
        or zero.get("backend_entry_digest") != authority["backend_entry_digest"]
        or zero.get("terminal_result_digest") != runtime_result["result_digest"]
        or zero.get("provider_zero_api_confirmed") is not True
        or zero.get("continuing_spend_from_this_run") is not False
        or not isinstance(retained, Mapping)
        or any(
            retained.get(role) != _record(path)
            for role, path in {
                "authority": authority_file,
                "bundle_receipt": bundle_file,
                "terminal_result": runtime_result_path,
                **closeout_paths,
                "stdout_log": log_paths["stdout"],
                "stderr_log": log_paths["stderr"],
            }.items()
            if role != "provider_zero"
        )
    ):
        raise ValueError("semantic_teacher_closeout_receipts_invalid")
    result: dict[str, Any] = {
        "schema_version": RESULT_IMPORT_SCHEMA_VERSION,
        "status": "retained_unreviewed_semantic_teacher_candidates",
        "runtime_result": _record(runtime_result_path),
        "runtime_request": _record(runtime_request_file),
        "bundle_receipt": _record(bundle_file),
        "authority": _record(authority_file),
        "teacher_frames": list(observed_frames.values()),
        "task_count": expected_task_count,
        "camera_count": expected_camera_count,
        "billing_receipt": _record(closeout_paths["billing"]),
        "scoped_inventory": _record(closeout_paths["scoped_inventory"]),
        "global_inventory": _record(closeout_paths["global_inventory"]),
        "object_store_cleanup": _record(closeout_paths["object_store_cleanup"]),
        "independent_watchdog": _record(closeout_paths["independent_watchdog"]),
        "secret_redaction": _record(closeout_paths["secret_redaction"]),
        "provider_zero": _record(closeout_paths["provider_zero"]),
        "runtime_stdout": _record(log_paths["stdout"]),
        "runtime_stderr": _record(log_paths["stderr"]),
        "all_generated_teacher_pngs_retained": True,
        "all_staged_objects_absent": True,
        "continuing_spend_from_this_run": False,
        "visual_reviewed": False,
        "appearance_qualified": False,
        "physical_evidence_claimed": False,
        "result_import_digest": "",
    }
    result["result_import_digest"] = canonical_digest(
        result, digest_field="result_import_digest"
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError("semantic_teacher_result_import_output_exists")
    ensure_dir(destination.parent)
    write_json(destination, result)
    return result


__all__ = [
    "materialize_semantic_teacher_image_edit_result",
    "materialize_semantic_teacher_no_allocation_closeout",
    "materialize_semantic_teacher_provider_zero_receipt",
    "prepare_semantic_teacher_image_edit_allocator_dry_run",
]
