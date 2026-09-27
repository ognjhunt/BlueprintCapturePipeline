"""Provider identity and sealed no-query billing resume for retained policy runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_policy_canary_session import LEARNED_ROLLOUT_COUNT
from .policy_canary_late_watchdog import late_watchdog_instance


def adapter_instance_ids(
    adapter: Mapping[str, Any], *, result_path: Path | None = None,
    read_json: Callable[[Path, str], Mapping[str, Any]],
    error_factory: Callable[[str], Exception],
) -> list[int]:
    values = adapter.get("vast_instance_ids")
    watchdog = adapter.get("independent_watchdog")
    if values is None and isinstance(watchdog, Mapping) and (
        watchdog.get("status") == "provider_terminal"
        and watchdog.get("provider_absence_confirmed") is True
    ):
        values = watchdog.get("instance_ids")
    if values is None and result_path is not None:
        late = late_watchdog_instance(
            result_path=result_path, result=adapter,
            read_json=read_json, error_factory=error_factory,
        )
        values = [late[0]] if late is not None else None
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        return []
    return list(values)


def retained_sparse_billing_gap(
    root: Path, *, read_json: Callable[[Path, str], Mapping[str, Any]],
    sealed_provider_zero: Callable[[Path], Mapping[str, Any] | None],
) -> bool:
    """Allow only a sealed old no-query allocation to finish billing without rerental."""

    required = (
        "allocator_result.json", "allocator_invocation_started.json",
        "allocator_invocation_finished.json", "policy_canary_session_authority.json",
        "policy_canary_terminal_result.json", "post_teardown_global_provider_zero.json",
        "dispatch_pending.json",
        "bundle/native_task_arena_policy_canary_session_bundle_receipt.v1.json",
    )
    if (root / "official_billing_reconciliation.json").exists() or not all(
        (root / name).is_file() for name in required
    ):
        return False
    pending = read_json(root / "dispatch_pending.json", "policy_canary_dispatch_pending_invalid")
    joined = read_json(root / "policy_canary_terminal_result.json", "policy_canary_terminal_result_invalid")
    adapter = read_json(root / "allocator_result.json", "policy_canary_allocator_result_invalid")
    closeout = adapter.get("provider_closeout")
    episodes = joined.get("episodes")
    no_query_rows = (
        isinstance(episodes, list)
        and len(episodes) == LEARNED_ROLLOUT_COUNT
        and all(
            isinstance(row, Mapping)
            and row.get("status") == "blocked"
            and row.get("candidate_policy_queried") is False
            and row.get("arm_moved") is False
            and row.get("typed_harness_failure") == "cell_not_completed_before_terminal_failure"
            and isinstance((row.get("visual_evidence") or {}).get("media_gap"), Mapping)
            for row in episodes
        )
    )
    return bool(
        pending.get("status") == "awaiting_official_billing"
        and joined.get("status") == "blocked"
        and joined.get("candidate_policy_queried") is False
        and joined.get("result_digest") == canonical_digest(joined, digest_field="result_digest")
        and no_query_rows
        and adapter.get("status") == "blocked"
        and adapter.get("retry_cap") == 0
        and adapter.get("continuing_spend_from_this_run") is False
        and isinstance(closeout, Mapping)
        and closeout.get("provider_zero_confirmed") is True
        and closeout.get("warm_session_retained") is False
        and closeout.get("all_staged_objects_absent") is True
        and sealed_provider_zero(root / "post_teardown_global_provider_zero.json") is not None
    )


__all__ = ["adapter_instance_ids", "retained_sparse_billing_gap"]
