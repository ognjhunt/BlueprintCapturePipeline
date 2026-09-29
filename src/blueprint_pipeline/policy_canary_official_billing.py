"""Terminal evidence adapter for official billing of policy canary sessions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .policy_canary_late_watchdog import late_watchdog_instance
from .policy_canary_staged_object_absence import billing_staged_objects_absent


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
    late_watchdog_path: Path | None = None
    if instance_ids is None and isinstance(watchdog, Mapping):
        # The canonical paid allocator owns the provider identity and already
        # seals it in the caller-surviving watchdog close receipt.  Early
        # policy-canary results did not duplicate that identity at top level,
        # so billing must consume the authoritative closure field instead of
        # waiting forever for a redundant projection that cannot appear after
        # teardown.
        if watchdog.get("status") == "retained_until_hard_ttl":
            late = late_watchdog_instance(
                result_path=result_path, result=result,
                read_json=lambda path, code: json_file(path, code=code)[1],
                error_factory=error_factory,
            )
            instance_ids = [late[0]] if late is not None else None
            late_watchdog_path = late[1] if late is not None else None
            watchdog_instance_lineage_valid = late is not None
        else:
            instance_ids = watchdog.get("instance_ids")
            watchdog_instance_lineage_valid = bool(
                watchdog.get("status") == "provider_terminal"
                and watchdog.get("provider_absence_confirmed") is True
            )
    elif instance_ids is None:
        watchdog_instance_lineage_valid = False
    # A streamed attempt whose lane deferred its cleanup is proven absent by
    # the digest-bound proof resume writes (review C2).
    staged_absent, absence_proof_path = billing_staged_objects_absent(result)
    if (
        instance_ids != [instance_id]
        or result.get("status") not in {"completed", "blocked"}
        or result.get("retry_cap") != 0
        or result.get("continuing_spend_from_this_run") is not False
        or not watchdog_instance_lineage_valid
        or not isinstance(closeout, Mapping)
        or closeout.get("provider_zero_confirmed") is not True
        or closeout.get("warm_session_retained") is not False
        or not staged_absent
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
    if late_watchdog_path is not None:
        path, _value, payload = json_file(
            late_watchdog_path, code="vast_official_policy_canary_late_watchdog_invalid"
        )
        terminal["independent_watchdog_terminal"] = record(path, payload)
    if absence_proof_path is not None:
        path, _value, payload = json_file(
            absence_proof_path, code="vast_official_policy_canary_staged_object_absence_proof_invalid"
        )
        terminal["staged_object_absence_proof"] = record(path, payload)
    return terminal


__all__ = ["policy_canary_terminal_evidence"]
