"""Seal an expired retained Arena worker from independent watchdog evidence."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .core.common import ensure_dir, utc_now_iso, write_json
from .native_task_arena_paid_authority import AUTHORITY_SCHEMA_VERSION
from .native_task_arena_paid_authority import (
    PROVIDER_ZERO_SCHEMA_VERSION as ARENA_PROVIDER_ZERO_SCHEMA_VERSION,
)
from .paid_attempt_authority import valid_adp_paid_provider_zero


CLOSEOUT_SCHEMA_VERSION = "native_task_arena_expired_warm_closeout.v1"
WATCHDOG_SCHEMA_VERSION = "groot_oscar_runpod_canary_watchdog.v1"
WATCHDOG_SUPERSESSION_SCHEMA_VERSION = "vast_independent_watchdog_supersession.v1"
FAILED_WATCHDOG_RECOVERY_SCHEMA_VERSION = (
    "native_task_arena_failed_watchdog_recovery_closeout.v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if not isinstance(value, dict):
        raise ValueError(code)
    return value


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _bound_path(raw: Any, *, code: str) -> Path:
    path = Path(str(raw or "")).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(code)
    return path.resolve()


def materialize_expired_warm_closeout(
    *,
    authority_path: str | Path,
    retained_result_path: str | Path,
    provider_zero_guard_path: str | Path,
    output_dir: str | Path,
    watchdog_supersession_path: str | Path | None = None,
    successor_watchdog_path: str | Path | None = None,
) -> dict[str, Any]:
    """Derive terminal records only after the retained worker is proven absent.

    The retained result remains immutable.  This function writes a new result,
    adapter, and teardown whose provenance binds the originals and the
    independent hard-TTL watchdog.  It performs no provider mutation.
    """

    authority_file = Path(authority_path).expanduser().resolve()
    retained_result_file = Path(retained_result_path).expanduser().resolve()
    authority = _read(authority_file, "expired_warm_authority_unreadable")
    retained = _read(retained_result_file, "expired_warm_result_unreadable")
    adapter_file = _bound_path(
        retained.get("adapter_result_path"), code="expired_warm_adapter_unbound"
    )
    teardown_file = _bound_path(
        retained.get("teardown_manifest_path"), code="expired_warm_teardown_unbound"
    )
    retained_watchdog_file = _bound_path(
        retained.get("watchdog_receipt_path"), code="expired_warm_watchdog_unbound"
    )
    supersession_file: Path | None = None
    supersession: dict[str, Any] | None = None
    if watchdog_supersession_path is not None or successor_watchdog_path is not None:
        if watchdog_supersession_path is None or successor_watchdog_path is None:
            raise ValueError("native_task_arena_expired_warm_closeout_invalid")
        supersession_file = Path(watchdog_supersession_path).expanduser().resolve()
        watchdog_file = Path(successor_watchdog_path).expanduser().resolve()
        if (
            supersession_file.is_symlink()
            or not supersession_file.is_file()
            or watchdog_file.is_symlink()
            or not watchdog_file.is_file()
        ):
            raise ValueError("native_task_arena_expired_warm_closeout_invalid")
        supersession = _read(
            supersession_file, "expired_warm_watchdog_supersession_unreadable"
        )
    else:
        watchdog_file = retained_watchdog_file
    guard_file = Path(provider_zero_guard_path).expanduser().resolve()
    cleanup_file = _bound_path(
        retained.get("object_store_cleanup_path"), code="expired_warm_cleanup_unbound"
    )
    adapter = _read(adapter_file, "expired_warm_adapter_unreadable")
    teardown = _read(teardown_file, "expired_warm_teardown_unreadable")
    retained_watchdog = _read(
        retained_watchdog_file, "expired_warm_watchdog_unreadable"
    )
    watchdog = _read(watchdog_file, "expired_warm_watchdog_unreadable")
    guard = _read(guard_file, "expired_warm_provider_zero_guard_unreadable")
    cleanup = _read(cleanup_file, "expired_warm_cleanup_unreadable")

    instance_ids = list(adapter.get("vast_instance_ids") or [])
    warm_session = retained.get("warm_session")
    warm_session = dict(warm_session) if isinstance(warm_session, Mapping) else {}
    instance_id = warm_session.get("instance_id")
    final_inventory = watchdog.get("final_inventory")
    recorded_teardown = watchdog.get("recorded_vast_instance_teardown")
    guard_rows = {
        row.get("provider"): row
        for row in guard.get("inventory_results") or []
        if isinstance(row, Mapping)
    }
    try:
        guard_after_watchdog = datetime.fromisoformat(
            str(guard.get("generated_at") or "").replace("Z", "+00:00")
        ) >= datetime.fromisoformat(
            str(watchdog.get("completed_at") or "").replace("Z", "+00:00")
        )
    except ValueError:
        guard_after_watchdog = False
    supersession_valid = supersession is None
    if supersession is not None:
        successor_out_dir = Path(
            str(supersession.get("successor_watchdog_out_dir") or "")
        ).expanduser()
        provider_inspections = (
            supersession.get("provider_inspect_before"),
            supersession.get("provider_inspect_successor_armed"),
            supersession.get("provider_inspect_after_transfer"),
        )
        try:
            successor_deadline_is_later = float(
                supersession.get("successor_watchdog_deadline_epoch") or 0
            ) > float(supersession.get("predecessor_watchdog_deadline_epoch") or 0)
            successor_started_id_matches = (
                (successor_out_dir / "started_vast_instance_id.txt")
                .read_text(encoding="utf-8")
                .strip()
                == str(instance_id)
            )
        except (OSError, UnicodeError, TypeError, ValueError, OverflowError):
            successor_deadline_is_later = False
            successor_started_id_matches = False
        supersession_valid = bool(
            supersession.get("schema_version")
            == WATCHDOG_SUPERSESSION_SCHEMA_VERSION
            and supersession.get("status") == "superseded"
            and supersession.get("instance_id") == instance_id
            and supersession.get("predecessor_watchdog_pid")
            == warm_session.get("watchdog_pid")
            and supersession.get("predecessor_watchdog_deadline_epoch")
            == warm_session.get("watchdog_deadline_epoch")
            and supersession.get("predecessor_watchdog_retired") is True
            and supersession.get("successor_watchdog_pid") == watchdog.get("pid")
            and supersession.get("successor_watchdog_deadline_epoch")
            == watchdog.get("deadline_epoch")
            and successor_deadline_is_later
            and successor_out_dir.is_absolute()
            and successor_out_dir.resolve() == watchdog_file.parent
            and watchdog.get("watchdog_out_dir") == str(successor_out_dir.resolve())
            and retained_watchdog.get("schema_version") == WATCHDOG_SCHEMA_VERSION
            and retained_watchdog.get("pid") == warm_session.get("watchdog_pid")
            and retained_watchdog.get("deadline_epoch")
            == warm_session.get("watchdog_deadline_epoch")
            and retained_watchdog.get("watchdog_out_dir")
            == warm_session.get("watchdog_out_dir")
            and retained_watchdog.get("pod_name_prefix")
            == warm_session.get("watchdog_pod_name_prefix")
            and supersession.get("provider_instance_running_after_transfer") is True
            and all(
                isinstance(row, Mapping)
                and row.get("api_confirmed") is True
                and str(row.get("instance_id") or "") == str(instance_id)
                and str(row.get("actual_status") or "").lower()
                in {"running", "active"}
                for row in provider_inspections
            )
            and successor_started_id_matches
        )
    if (
        authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or authority.get("authorization_digest")
        != canonical_digest(authority, digest_field="authorization_digest")
        or retained.get("schema_version") != "native_task_arena_vast_run.v1"
        or retained.get("status") not in {"blocked", "completed"}
        or retained.get("authorization_consumption", {}).get("status") != "consumed"
        or retained.get("authorization_consumption", {}).get("authorization_digest")
        != authority.get("authorization_digest")
        or retained.get("continuing_spend_from_this_run") is not True
        or retained.get("all_staged_objects_absent") is not True
        or warm_session.get("status") != "ready"
        or warm_session.get("continuing_spend") is not True
        or isinstance(instance_id, bool)
        or not isinstance(instance_id, int)
        or instance_ids != [instance_id]
        or adapter.get("schema_version") != "vast_provider_adapter_result.v1"
        or adapter.get("retained_owned") is not True
        or adapter.get("continuing_spend_from_this_run") is not True
        or teardown.get("schema_version") != "vast_teardown_manifest.v1"
        or teardown.get("status") != "retained_owned"
        or teardown.get("vast_instance_ids") != [instance_id]
        or teardown.get("continuing_spend_from_this_run") is not True
        or watchdog.get("schema_version") != WATCHDOG_SCHEMA_VERSION
        or watchdog.get("provider") != "vast"
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("provider_absence_confirmed") is not True
        or not isinstance(final_inventory, Mapping)
        or final_inventory.get("api_confirmed") is not True
        or final_inventory.get("live_resource_count") != 0
        or not isinstance(recorded_teardown, Mapping)
        # Vast's exact-id inspection response serializes this identifier as a
        # decimal string while the allocation/teardown manifests use an int.
        # Normalize only that representation; never accept a different id.
        or str(recorded_teardown.get("instance_id") or "") != str(instance_id)
        or recorded_teardown.get("provider_absence_confirmed") is not True
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or guard_file.is_symlink()
        or not guard_file.is_file()
        or guard.get("schema_version") != "gpu_spend_guard.v1"
        or guard.get("status") != "passed"
        or guard.get("provider_zero_verified") is not True
        or guard.get("live_instance_count") != 0
        or guard.get("blockers") != []
        or guard_rows.get("vast", {}).get("status") != "succeeded"
        or guard_rows.get("vast", {}).get("row_count") != 0
        or not guard_after_watchdog
        or not supersession_valid
    ):
        raise ValueError("native_task_arena_expired_warm_closeout_invalid")

    root = Path(output_dir).expanduser().resolve()
    if root.exists() or root.is_symlink():
        raise ValueError("native_task_arena_expired_warm_closeout_output_exists")
    ensure_dir(root)
    generated_at = utc_now_iso()

    terminal_watchdog = dict(watchdog)
    terminal_watchdog.update(
        {
            "global_inventory_informational_only": True,
            "final_global_inventory": {
                "status": "observed",
                "provider": "vast",
                "live_resource_count": 0,
                "resources": [],
                "api_confirmed": True,
                "provider_zero_guard": _record(guard_file),
                "raw_provider_response_recorded": False,
            },
        }
    )
    terminal_watchdog_path = root / "groot_oscar_runpod_canary_watchdog.json"
    write_json(terminal_watchdog_path, terminal_watchdog)

    terminal_teardown = {
        "schema_version": "vast_teardown_manifest.v1",
        "generated_at": generated_at,
        "status": "completed",
        "vast_instance_ids": [instance_id],
        "teardown_actions_performed": list(watchdog.get("terminations") or []),
        "runner_gpu_teardown_completed": True,
        "provider_instance_absent": True,
        "continuing_spend_from_this_run": False,
        "retention_authorized": False,
        "zero_continuing_spend_scope": "independent_watchdog_exact_instance_and_lane_inventory",
        "retained_teardown": _record(teardown_file),
        "independent_watchdog": _record(terminal_watchdog_path),
        "raw_secret_values_recorded": False,
    }
    terminal_teardown_path = root / "vast_teardown_manifest.json"
    write_json(terminal_teardown_path, terminal_teardown)

    terminal_adapter = dict(adapter)
    terminal_adapter.update(
        {
            "generated_at": generated_at,
            "status": "completed",
            "reason": "retained_worker_closed_by_independent_hard_ttl_watchdog",
            "continuing_spend_from_this_run": False,
            "retained_owned": False,
            "retained_adapter_result": _record(adapter_file),
            "independent_watchdog": _record(terminal_watchdog_path),
            "raw_secret_values_recorded": False,
        }
    )
    terminal_adapter_path = root / "vast_provider_adapter_result.json"
    write_json(terminal_adapter_path, terminal_adapter)

    terminal_result = dict(retained)
    terminal_warm_session = dict(warm_session)
    terminal_warm_session.update(
        {"status": "provider_terminal", "continuing_spend": False}
    )
    terminal_result.update(
        {
            "generated_at": generated_at,
            "adapter_result_path": str(terminal_adapter_path),
            "teardown_manifest_path": str(terminal_teardown_path),
            "warm_session": terminal_warm_session,
            "watchdog_receipt_path": str(terminal_watchdog_path),
            "independent_watchdog": dict(terminal_watchdog),
            "independent_watchdog_close": dict(terminal_watchdog),
            "continuing_spend_from_this_run": False,
            "expired_warm_session_closeout": {
                "schema_version": CLOSEOUT_SCHEMA_VERSION,
                "retained_result": _record(retained_result_file),
                "retained_adapter": _record(adapter_file),
                "retained_teardown": _record(teardown_file),
            "independent_watchdog": _record(watchdog_file),
            "watchdog_supersession": (
                _record(supersession_file) if supersession_file is not None else None
            ),
                "provider_zero_guard": _record(guard_file),
                "provider_mutation_performed": False,
            },
        }
    )
    terminal_result_path = root / "adp_arena_vast_result.json"
    write_json(terminal_result_path, terminal_result)

    receipt = {
        "schema_version": CLOSEOUT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed",
        "authority": _record(authority_file),
        "retained_result": _record(retained_result_file),
        "terminal_result": _record(terminal_result_path),
        "terminal_adapter": _record(terminal_adapter_path),
        "terminal_teardown": _record(terminal_teardown_path),
        "independent_watchdog": _record(watchdog_file),
        "watchdog_supersession": (
            _record(supersession_file) if supersession_file is not None else None
        ),
        "terminal_watchdog": _record(terminal_watchdog_path),
        "provider_zero_guard": _record(guard_file),
        "provider_instance_id": instance_id,
        "provider_instance_absent": True,
        "continuing_spend_from_this_run": False,
        "provider_mutation_performed": False,
        "blockers": [],
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    write_json(root / "native_task_arena_expired_warm_closeout.v1.json", receipt)
    return receipt


def _timestamp(value: Any, *, code: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(code) from exc
    if parsed.tzinfo is None:
        raise ValueError(code)
    return parsed.astimezone(timezone.utc)


def _session_recovery_observation(
    path: Path, *, instance_id: int, deadline_epoch: float
) -> dict[str, Any]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("failed_watchdog_recovery_session_excerpt_invalid") from exc
    if len(rows) != 4 or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("failed_watchdog_recovery_session_excerpt_invalid")
    terminate_call, terminate_result, inspect_call, inspect_result = rows
    terminate_input = str((terminate_call.get("payload") or {}).get("input") or "")
    inspect_input = str((inspect_call.get("payload") or {}).get("input") or "")
    terminal_item = (terminate_result.get("payload") or {}).get("item") or {}
    readback_item = (inspect_result.get("payload") or {}).get("item") or {}
    try:
        terminate_response = json.loads(str(terminal_item.get("stdout") or ""))
        readback_response = json.loads(str(readback_item.get("stdout") or ""))
    except json.JSONDecodeError as exc:
        raise ValueError("failed_watchdog_recovery_session_excerpt_invalid") from exc
    inspect = readback_response.get("inspect")
    inventory = readback_response.get("inventory")
    terminate_at = _timestamp(
        terminate_result.get("timestamp"),
        code="failed_watchdog_recovery_session_excerpt_invalid",
    )
    readback_at = _timestamp(
        inspect_result.get("timestamp"),
        code="failed_watchdog_recovery_session_excerpt_invalid",
    )
    expected_instance = str(instance_id)
    if (
        terminate_call.get("type") != "response_item"
        or (terminate_call.get("payload") or {}).get("type") != "custom_tool_call"
        or "VastRenderProvider" not in terminate_input
        or f'terminate(\\"{expected_instance}\\")' not in terminate_input
        or terminate_result.get("type") != "event_msg"
        or terminal_item.get("type") != "CommandExecution"
        or terminal_item.get("exit_code") != 0
        or str(terminal_item.get("stderr") or "") != ""
        or terminate_response != {"http": 200, "status": "stopped"}
        or terminate_at.timestamp() <= deadline_epoch
        or inspect_call.get("type") != "response_item"
        or (inspect_call.get("payload") or {}).get("type") != "custom_tool_call"
        or "VastRenderProvider" not in inspect_input
        or f'inspect(\\"{expected_instance}\\")' not in inspect_input
        or "billable_inventory(name_prefix=\\\"\\\")" not in inspect_input
        or inspect_result.get("type") != "event_msg"
        or readback_item.get("type") != "CommandExecution"
        or readback_item.get("exit_code") != 0
        or str(readback_item.get("stderr") or "") != ""
        or not isinstance(inspect, Mapping)
        or inspect.get("status") != "absent"
        or inspect.get("provider") != "vast"
        or str(inspect.get("instance_id") or "") != expected_instance
        or inspect.get("api_confirmed") is not True
        or inspect.get("provider_absence_confirmed") is not True
        or inspect.get("raw_provider_response_recorded") is not False
        or not isinstance(inventory, Mapping)
        or inventory.get("status") != "observed"
        or inventory.get("provider") != "vast"
        or inventory.get("name_prefix") != ""
        or inventory.get("api_confirmed") is not True
        or inventory.get("live_resource_count") != 0
        or inventory.get("resources") != []
        or inventory.get("raw_provider_response_recorded") is not False
        or readback_at < terminate_at
    ):
        raise ValueError("failed_watchdog_recovery_session_excerpt_invalid")
    return {
        "terminate_response": terminate_response,
        "terminate_observed_at": terminate_at.isoformat(),
        "readback": readback_response,
        "readback_observed_at": readback_at.isoformat(),
        "source_excerpt": _record(path),
        "source_line_sha256": [
            "sha256:" + hashlib.sha256(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            for row in rows
        ],
    }


def _exact_absence_observation(path: Path, *, instance_id: int) -> datetime:
    value = _read(path, "failed_watchdog_recovery_absence_observation_invalid")
    inspect = value.get("inspect_result")
    if (
        value.get("schema_version") != "vast_exact_instance_absence_observation.v1"
        or value.get("provider") != "vast"
        or value.get("instance_id") != instance_id
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
        or not isinstance(inspect, Mapping)
        or inspect.get("status") != "absent"
        or inspect.get("provider") != "vast"
        or str(inspect.get("instance_id") or "") != str(instance_id)
        or inspect.get("api_confirmed") is not True
        or inspect.get("provider_absence_confirmed") is not True
        or inspect.get("raw_provider_response_recorded") is not False
        or value.get("raw_secret_values_recorded") is not False
    ):
        raise ValueError("failed_watchdog_recovery_absence_observation_invalid")
    return _timestamp(
        value.get("observed_at"),
        code="failed_watchdog_recovery_absence_observation_invalid",
    )


def materialize_failed_watchdog_recovery_closeout(
    *,
    authority_path: str | Path,
    retained_result_path: str | Path,
    termination_session_excerpt_path: str | Path,
    exact_absence_observation_paths: list[str | Path],
    provider_zero_path: str | Path,
    official_billing_response_path: str | Path,
    provider_billing_source_receipt_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Seal a retained worker after its watchdog died without terminal evidence.

    This recovery never rewrites the stale watchdog and never claims it performed
    teardown. It binds the observed canonical terminate call, repeated exact-id
    absence inspections, later authenticated global zero, and official billing.
    """

    authority_file = Path(authority_path).expanduser().resolve()
    retained_file = Path(retained_result_path).expanduser().resolve()
    session_file = Path(termination_session_excerpt_path).expanduser().resolve()
    zero_file = Path(provider_zero_path).expanduser().resolve()
    billing_file = Path(official_billing_response_path).expanduser().resolve()
    billing_source_file = Path(
        provider_billing_source_receipt_path
    ).expanduser().resolve()
    absence_files = [Path(path).expanduser().resolve() for path in exact_absence_observation_paths]
    if len(absence_files) != 2 or len(set(absence_files)) != 2:
        raise ValueError("failed_watchdog_recovery_absence_observation_invalid")
    authority = _read(authority_file, "failed_watchdog_recovery_authority_invalid")
    retained = _read(retained_file, "failed_watchdog_recovery_result_invalid")
    adapter_file = _bound_path(
        retained.get("adapter_result_path"), code="failed_watchdog_recovery_adapter_invalid"
    )
    teardown_file = _bound_path(
        retained.get("teardown_manifest_path"), code="failed_watchdog_recovery_teardown_invalid"
    )
    watchdog_file = _bound_path(
        retained.get("watchdog_receipt_path"), code="failed_watchdog_recovery_watchdog_invalid"
    )
    cleanup_file = _bound_path(
        retained.get("object_store_cleanup_path"), code="failed_watchdog_recovery_cleanup_invalid"
    )
    adapter = _read(adapter_file, "failed_watchdog_recovery_adapter_invalid")
    teardown = _read(teardown_file, "failed_watchdog_recovery_teardown_invalid")
    watchdog = _read(watchdog_file, "failed_watchdog_recovery_watchdog_invalid")
    cleanup = _read(cleanup_file, "failed_watchdog_recovery_cleanup_invalid")
    zero = _read(zero_file, "failed_watchdog_recovery_provider_zero_invalid")
    billing = _read(billing_file, "failed_watchdog_recovery_billing_invalid")
    billing_source = _read(
        billing_source_file, "failed_watchdog_recovery_billing_source_invalid"
    )
    warm = retained.get("warm_session")
    warm = dict(warm) if isinstance(warm, Mapping) else {}
    instance_id = warm.get("instance_id")
    deadline_epoch = watchdog.get("deadline_epoch")
    if isinstance(instance_id, bool) or not isinstance(instance_id, int):
        raise ValueError("failed_watchdog_recovery_result_invalid")
    try:
        deadline_epoch = float(deadline_epoch)
    except (TypeError, ValueError) as exc:
        raise ValueError("failed_watchdog_recovery_watchdog_invalid") from exc
    session_observation = _session_recovery_observation(
        session_file, instance_id=instance_id, deadline_epoch=deadline_epoch
    )
    absence_times = [
        _exact_absence_observation(path, instance_id=instance_id)
        for path in absence_files
    ]
    zero_at = _timestamp(
        zero.get("observed_at_utc"),
        code="failed_watchdog_recovery_provider_zero_invalid",
    )
    billing_rows = [
        row
        for row in billing.get("results") or []
        if isinstance(row, Mapping) and row.get("source") == f"instance-{instance_id}"
    ]
    linked_billing = [
        row
        for row in billing_source.get("sources") or []
        if isinstance(row, Mapping)
        and row.get("provider") == "vast"
        and Path(str(row.get("retained_path") or "")).expanduser().resolve()
        == billing_file
        and row.get("response_digest") == _sha256(billing_file)
        and row.get("response_size_bytes") == billing_file.stat().st_size
    ]
    billing_amount = billing_rows[0].get("amount") if len(billing_rows) == 1 else None
    if (
        authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or authority.get("authorization_digest")
        != canonical_digest(authority, digest_field="authorization_digest")
        or retained.get("schema_version") != "native_task_arena_vast_run.v1"
        or retained.get("status") not in {"blocked", "completed"}
        or retained.get("authorization_consumption", {}).get("status") != "consumed"
        or retained.get("authorization_consumption", {}).get("authorization_digest")
        != authority.get("authorization_digest")
        or retained.get("continuing_spend_from_this_run") is not True
        or retained.get("all_staged_objects_absent") is not True
        or warm.get("status") != "ready"
        or warm.get("continuing_spend") is not True
        or warm.get("watchdog_pid") != watchdog.get("pid")
        or warm.get("watchdog_deadline_epoch") != deadline_epoch
        or adapter.get("schema_version") != "vast_provider_adapter_result.v1"
        or adapter.get("vast_instance_ids") != [instance_id]
        or adapter.get("retained_owned") is not True
        or adapter.get("continuing_spend_from_this_run") is not True
        or teardown.get("schema_version") != "vast_teardown_manifest.v1"
        or teardown.get("status") != "retained_owned"
        or teardown.get("vast_instance_ids") != [instance_id]
        or teardown.get("continuing_spend_from_this_run") is not True
        or watchdog.get("schema_version") != WATCHDOG_SCHEMA_VERSION
        or watchdog.get("provider") != "vast"
        or watchdog.get("status") != "armed"
        or watchdog.get("independent_process") is not True
        or watchdog.get("provider_mutations_performed") != 0
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or not valid_adp_paid_provider_zero(zero)
        or any(observed.timestamp() <= deadline_epoch for observed in absence_times)
        or zero_at < max(absence_times)
        or billing_source.get("status") != "reconciled"
        or len(linked_billing) != 1
        or len(billing_rows) != 1
        or isinstance(billing_amount, bool)
        or not isinstance(billing_amount, (int, float))
        or float(billing_amount) < 0
    ):
        raise ValueError("native_task_arena_failed_watchdog_recovery_invalid")

    root = Path(output_dir).expanduser().resolve()
    if root.exists() or root.is_symlink():
        raise ValueError("native_task_arena_failed_watchdog_recovery_output_exists")
    ensure_dir(root)
    generated_at = utc_now_iso()
    recovery = {
        "schema_version": FAILED_WATCHDOG_RECOVERY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed",
        "reason": "independent_watchdog_caller_exit_survival_failed",
        "provider": "vast",
        "provider_instance_id": instance_id,
        "stale_armed_watchdog": _record(watchdog_file),
        "watchdog_performed_teardown": False,
        "canonical_termination_observation": session_observation,
        "exact_absence_observations": [_record(path) for path in absence_files],
        "provider_zero": _record(zero_file),
        "official_billing_response": _record(billing_file),
        "provider_billing_source_receipt": _record(billing_source_file),
        "official_billing_amount_usd": float(billing_amount),
        "provider_instance_absent": True,
        "continuing_spend_from_this_run": False,
        "provider_mutation_performed_by_materializer": False,
        "raw_secret_values_recorded": False,
        "receipt_digest": "",
    }
    recovery["receipt_digest"] = canonical_digest(
        recovery, digest_field="receipt_digest"
    )
    recovery_path = root / "native_task_arena_failed_watchdog_recovery.v1.json"
    write_json(recovery_path, recovery)

    terminal_teardown = {
        "schema_version": "vast_teardown_manifest.v1",
        "generated_at": generated_at,
        "status": "completed",
        "vast_instance_ids": [instance_id],
        "teardown_actions_performed": [
            {
                "action": "canonical_external_termination_recovered_from_session_evidence",
                "instance_id": instance_id,
                "http_status_code": 200,
                "provider_status": "stopped",
            }
        ],
        "runner_gpu_teardown_completed": True,
        "provider_instance_absent": True,
        "continuing_spend_from_this_run": False,
        "retention_authorized": False,
        "zero_continuing_spend_scope": "exact_instance_readback_and_global_provider_zero",
        "retained_teardown": _record(teardown_file),
        "failed_watchdog_recovery": _record(recovery_path),
        "raw_secret_values_recorded": False,
    }
    terminal_teardown_path = root / "vast_teardown_manifest.json"
    write_json(terminal_teardown_path, terminal_teardown)

    terminal_adapter = dict(adapter)
    terminal_adapter.update(
        {
            "generated_at": generated_at,
            "status": "completed",
            "reason": "retained_worker_closed_by_failed_watchdog_recovery",
            "continuing_spend_from_this_run": False,
            "retained_owned": False,
            "retained_adapter_result": _record(adapter_file),
            "failed_watchdog_recovery": _record(recovery_path),
            "raw_secret_values_recorded": False,
        }
    )
    terminal_adapter_path = root / "vast_provider_adapter_result.json"
    write_json(terminal_adapter_path, terminal_adapter)

    terminal_result = dict(retained)
    terminal_warm = dict(warm)
    terminal_warm.update({"status": "provider_terminal", "continuing_spend": False})
    terminal_result.update(
        {
            "generated_at": generated_at,
            "adapter_result_path": str(terminal_adapter_path),
            "teardown_manifest_path": str(terminal_teardown_path),
            "warm_session": terminal_warm,
            "continuing_spend_from_this_run": False,
            "failed_watchdog_recovery_closeout": _record(recovery_path),
        }
    )
    terminal_result_path = root / "adp_arena_vast_result.json"
    write_json(terminal_result_path, terminal_result)

    arena_provider_zero = {
        "schema_version": ARENA_PROVIDER_ZERO_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed_recovered_provider_zero",
        "attempt_authority": _record(authority_file),
        "attempt_authority_digest": authority["authorization_digest"],
        "terminal_result": _record(terminal_result_path),
        "provider_adapter": _record(terminal_adapter_path),
        "teardown": _record(terminal_teardown_path),
        "watchdog": _record(watchdog_file),
        "failed_watchdog_recovery": _record(recovery_path),
        "object_store_cleanup": _record(cleanup_file),
        "estimated_cost_usd": terminal_result.get("estimated_cost_usd"),
        "provider_zero_confirmed": True,
        "inventory": [],
        "inventory_scope": (
            "failed_watchdog_exact_instance_and_fresh_global_zero_recovery"
        ),
        "recovered_global_provider_zero": _record(zero_file),
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "receipt_digest": "",
    }
    arena_provider_zero["receipt_digest"] = canonical_digest(
        arena_provider_zero, digest_field="receipt_digest"
    )
    arena_provider_zero_path = root / "native_task_arena_provider_zero.v1.json"
    write_json(arena_provider_zero_path, arena_provider_zero)

    receipt = dict(recovery)
    receipt.update(
        {
            "terminal_result": _record(terminal_result_path),
            "terminal_adapter": _record(terminal_adapter_path),
            "terminal_teardown": _record(terminal_teardown_path),
            "arena_provider_zero": _record(arena_provider_zero_path),
            "receipt_digest": "",
        }
    )
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    write_json(root / "native_task_arena_failed_watchdog_closeout.v1.json", receipt)
    return receipt


__all__ = [
    "CLOSEOUT_SCHEMA_VERSION",
    "FAILED_WATCHDOG_RECOVERY_SCHEMA_VERSION",
    "materialize_expired_warm_closeout",
    "materialize_failed_watchdog_recovery_closeout",
]
