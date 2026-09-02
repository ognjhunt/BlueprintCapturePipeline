"""Terminal evidence adapter for official billing of policy canary sessions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def policy_canary_terminal_evidence(
    *,
    instance_id: int,
    result_path: Path,
    result: Mapping[str, Any],
    result_bytes: bytes,
    json_file: Callable[..., tuple[Path, dict[str, Any], bytes]],
    record: Callable[[Path, bytes], dict[str, Any]],
    error_factory: Callable[[str], Exception],
) -> dict[str, Any] | None:
    if (
        result.get("schema_version")
        != "native_task_arena_policy_canary_session_result.v1"
        or result_path.name != "allocator_result.json"
    ):
        return None
    closeout = result.get("provider_closeout")
    watchdog = result.get("independent_watchdog")
    instance_ids = result.get("vast_instance_ids")
    watchdog_instance_lineage_valid = True
    if instance_ids is None and isinstance(watchdog, Mapping):
        # The canonical paid allocator owns the provider identity and already
        # seals it in the caller-surviving watchdog close receipt.  Early
        # policy-canary results did not duplicate that identity at top level,
        # so billing must consume the authoritative closure field instead of
        # waiting forever for a redundant projection that cannot appear after
        # teardown.
        instance_ids = watchdog.get("instance_ids")
        watchdog_instance_lineage_valid = bool(
            watchdog.get("status") == "provider_terminal"
            and watchdog.get("provider_absence_confirmed") is True
        )
    elif instance_ids is None:
        watchdog_instance_lineage_valid = False
    if (
        instance_ids != [instance_id]
        or result.get("status") not in {"completed", "blocked"}
        or result.get("retry_cap") != 0
        or result.get("continuing_spend_from_this_run") is not False
        or not watchdog_instance_lineage_valid
        or not isinstance(closeout, Mapping)
        or closeout.get("provider_zero_confirmed") is not True
        or closeout.get("warm_session_retained") is not False
        or closeout.get("all_staged_objects_absent") is not True
    ):
        raise error_factory("vast_official_terminal_result_invalid")
    paths = {
        "provider_adapter_result": Path(str(result.get("adapter_result_path") or "")),
        "teardown_manifest": Path(str(result.get("teardown_manifest_path") or "")),
        "artifact_manifest": Path(str(result.get("artifact_manifest_path") or "")),
        "post_teardown_provider_zero": (
            result_path.parent / "post_teardown_global_provider_zero.json"
        ),
    }
    loaded = {
        role: json_file(path, code=f"vast_official_policy_canary_{role}_invalid")
        for role, path in paths.items()
    }
    adapter = loaded["provider_adapter_result"][1]
    teardown = loaded["teardown_manifest"][1]
    zero = loaded["post_teardown_provider_zero"][1]
    if (
        adapter.get("vast_instance_ids") != [instance_id]
        or adapter.get("continuing_spend_from_this_run") is not False
        or teardown.get("vast_instance_ids") != [instance_id]
        or teardown.get("continuing_spend_from_this_run") is not False
        or teardown.get("runner_gpu_teardown_completed") is not True
        or zero.get("schema_version")
        != "task_evaluation_policy_canary_vast_provider_zero.v1"
        or zero.get("provider_zero_verified") is not True
        or zero.get("live_instance_count") != 0
    ):
        raise error_factory("vast_official_terminal_result_invalid")
    terminal = {
        "terminal_status": result["status"],
        "provider_absence_confirmed": True,
        "provider_zero_verified": True,
        "continuing_spend_from_this_run": False,
        "retry_cap": 0,
        "terminal_result": record(result_path, result_bytes),
    }
    for role, (path, _value, payload) in loaded.items():
        terminal[role] = record(path, payload)
    return terminal


__all__ = ["policy_canary_terminal_evidence"]
