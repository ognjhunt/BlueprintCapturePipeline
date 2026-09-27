"""Prove a refused Vast create did not allocate a policy canary instance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .vast_create_failure_diagnosis import definite_create_refusal_without_instance


def proven_provider_null_closeout(
    allocator: Mapping[str, Any],
    *,
    root: Path,
    record_file: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Return exact retained records only when refusal, teardown and watchdog agree."""

    closeout = allocator.get("provider_closeout")
    watchdog = allocator.get("independent_watchdog_close")
    if not isinstance(closeout, Mapping) or not isinstance(watchdog, Mapping):
        return None
    watchdog_status = watchdog.get("status")
    watchdog_closed = watchdog_status == "cancelled_no_allocation"
    watchdog_retained = (
        watchdog_status == "retained_until_hard_ttl"
        and watchdog.get("reason") == "provider_allocation_identity_ambiguous"
        and watchdog.get("watchdog_retention_liveness_confirmed") is True
        and watchdog.get("watchdog_armed_before_allocation") is True
        and watchdog.get("instance_ids") == []
    )
    if not (
        allocator.get("status") == "blocked"
        and allocator.get("scientific_attempt_started") is False
        and allocator.get("candidate_policy_queried") is False
        and allocator.get("continuing_spend_from_this_run") is False
        and allocator.get("all_staged_objects_absent") is True
        and closeout.get("provider_zero_confirmed") is True
        and (watchdog_closed or watchdog_retained)
        and watchdog.get("provider_mutations_performed") == 0
    ):
        return None
    records: dict[str, Mapping[str, Any]] = {}
    for key, path_value in (
        ("adapter_result", allocator.get("adapter_result_path")),
        ("teardown_manifest", allocator.get("teardown_manifest_path")),
    ):
        if not isinstance(path_value, str) or not path_value:
            return None
        path = Path(path_value)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            return None
        record = record_file(path)
        if closeout.get(key) != record:
            return None
        records[key] = record
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(value, Mapping):
            return None
        if key == "adapter_result" and not definite_create_refusal_without_instance(value):
            return None
        if key == "teardown_manifest" and not (
            value.get("status") == "completed"
            and value.get("vast_instance_ids") == []
            and value.get("continuing_spend_from_this_run") is False
        ):
            return None
    return {
        "provider_adapter": dict(records["adapter_result"]),
        "teardown": dict(records["teardown_manifest"]),
        "watchdog_status": watchdog_status,
        "watchdog_retained_until_hard_ttl": watchdog_retained,
    }


__all__ = ["proven_provider_null_closeout"]
