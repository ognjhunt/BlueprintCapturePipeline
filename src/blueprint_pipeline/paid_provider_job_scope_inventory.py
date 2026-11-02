"""Read the live Vast inventory for a job-scoped provider-zero receipt.

Kept apart from ``paid_provider_job_scope`` on purpose: that module's pure
receipt checks are imported by the control-plane launch reconciler, and this
reader's default collector pulls in the GPU provider adapters. Living here, the
reader never puts the hot lane on the control plane's import graph
(tests/test_live_pipeline_import_isolation.py).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paid_provider_job_scope import (
    ProviderJobScopeError,
    _positive_rate,
    build_job_provider_zero_receipt,
    load_spend_guard,
)


def collect_job_scoped_vast_provider_zero(
    *,
    job_key: str,
    added_burn_usd_per_hour: float,
    own_label_prefixes: Sequence[str] = (),
    guard_path: str | Path | None = None,
    inventory_collector: Callable[[], Mapping[str, Any]] | None = None,
    global_receipt_collector: Callable[..., Mapping[str, Any]] | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """One live Vast reading, returned as the strictest receipt it supports.

    Empty account: the classic ``adp_paid_provider_zero.v1`` receipt, exactly as
    before.  Otherwise: a job-scoped receipt if, and only if, everything live is
    another job's owned, labeled work and the fleet budget has room for this
    job's own instance at ``added_burn_usd_per_hour`` (its authorized maximum
    hourly rate).
    """

    if not _positive_rate(added_burn_usd_per_hour):
        raise ProviderJobScopeError(["job_scope_incoming_hourly_rate_missing"])

    if inventory_collector is None:
        from .gpu_render_providers import VastRenderProvider

        def inventory_collector() -> Mapping[str, Any]:
            return VastRenderProvider().billable_inventory(name_prefix="")

    observed = dict(inventory_collector())
    resources = observed.get("resources")
    if (
        observed.get("provider") != "vast"
        or observed.get("status") != "observed"
        or observed.get("api_confirmed") is not True
        or observed.get("blockers") not in (None, [])
        or not isinstance(resources, list)
        or any(not isinstance(row, Mapping) for row in resources)
        or observed.get("live_resource_count") != len(resources)
        or observed.get("raw_provider_response_recorded") is not False
    ):
        raise ProviderJobScopeError(["job_scope_provider_inventory_query_failed"])
    if not resources:
        if global_receipt_collector is None:
            from .adp_task_evaluation_abstention import collect_vast_provider_zero_receipt

            global_receipt_collector = collect_vast_provider_zero_receipt
        return dict(global_receipt_collector(inventory_collector=lambda: observed))
    guard, record = load_spend_guard(guard_path)
    observed_at = (now or (lambda: datetime.now(timezone.utc)))()
    return build_job_provider_zero_receipt(
        job_key=job_key, live_inventory=resources, guard=guard, guard_record=record,
        observed_at=observed_at, own_label_prefixes=own_label_prefixes,
        added_burn_usd_per_hour=float(added_burn_usd_per_hour),
    )


__all__ = ["collect_job_scoped_vast_provider_zero"]
