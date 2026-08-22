"""Seal an expired retained Arena worker from independent watchdog evidence."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .core.common import ensure_dir, utc_now_iso, write_json
from .native_task_arena_paid_authority import AUTHORITY_SCHEMA_VERSION


CLOSEOUT_SCHEMA_VERSION = "native_task_arena_expired_warm_closeout.v1"
WATCHDOG_SCHEMA_VERSION = "groot_oscar_runpod_canary_watchdog.v1"


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
    watchdog_file = _bound_path(
        retained.get("watchdog_receipt_path"), code="expired_warm_watchdog_unbound"
    )
    guard_file = Path(provider_zero_guard_path).expanduser().resolve()
    cleanup_file = _bound_path(
        retained.get("object_store_cleanup_path"), code="expired_warm_cleanup_unbound"
    )
    adapter = _read(adapter_file, "expired_warm_adapter_unreadable")
    teardown = _read(teardown_file, "expired_warm_teardown_unreadable")
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


__all__ = ["CLOSEOUT_SCHEMA_VERSION", "materialize_expired_warm_closeout"]
