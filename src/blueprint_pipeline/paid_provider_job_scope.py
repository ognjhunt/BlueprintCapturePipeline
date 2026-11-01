"""Job-scoped provider zero: let website jobs run paid GPU work side by side.

Before this module, the website scene-configuration path asked for an
*account-wide* Vast provider-zero reading before staging a new activation
(``adp_paid_provider_zero.v1``: ``global_live_resource_count == 0``), and scene
progression's release/recovery reconciliation asked the GPU spend guard for
``live_instance_count == 0`` across the whole fleet.  Both are correct for a
single job and both serialize every website job behind every other one: while
any instance is live anywhere on the account, every other job waits.

What those checks protect is narrower than "the account is empty":

1. **This job's own spend is accounted.** No instance attributable to this job
   is still billing when it starts or is reconciled.
2. **Nothing on the account is unaccounted.** Every live instance has a live
   owner (the spend guard marks it ``protected``) and a Blueprint label, and no
   orphan is waiting to be reaped.
3. **The fleet stays inside its money cap.** The spend guard's fleet budget
   (live-instance ceiling, burn per hour, daily and total spend) passes, with
   headroom for what this job will add.

Every one of those can be proven while *other* jobs are running.  This module
proves them from two read-only observations: the live, API-confirmed Vast
inventory and the reap-mode GPU spend guard report.  The account-wide fleet
budget, the spend guard's orphan reaping and each attempt's own hard cap, TTL
and watchdog stay exactly as they are; this only replaces "nobody is running"
with "nobody unaccounted is running and there is budget for one more".

The scope is opt-in (``BLUEPRINT_PAID_PROVIDER_ZERO_SCOPE=job``).  The default
stays ``global`` so a deploy changes nothing until the owner turns it on.  When
the account *is* empty, callers still emit the classic global receipt, so a
job-scoped receipt only ever appears while other jobs are live.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .vast_evidence_contracts import (
    VAST_PROVIDER_ZERO_API_CALL,
    valid_vast_provider_zero_api_call,
)

JOB_PROVIDER_ZERO_SCHEMA_VERSION = "adp_paid_provider_job_zero.v1"
GLOBAL_PROVIDER_ZERO_SCHEMA_VERSION = "adp_paid_provider_zero.v1"
SPEND_GUARD_SCHEMA_VERSION = "gpu_spend_guard.v1"

SCOPE_ENV = "BLUEPRINT_PAID_PROVIDER_ZERO_SCOPE"
SCOPE_GLOBAL = "global"
SCOPE_JOB = "job"
GUARD_REPORT_ENV = "BLUEPRINT_GPU_SPEND_GUARD_REPORT"
DEFAULT_GUARD_REPORT = (
    "/var/lib/blueprint/pipeline-control-plane/gpu_spend_guard/latest.json"
)
#: The spend guard timer refreshes about once a minute; paid admission already
#: treats a report older than five minutes as stale.
DEFAULT_MAX_GUARD_AGE_SECONDS = 300
_CLOCK_SKEW_SECONDS = 60
#: Only Blueprint lanes stamp this; anything else on the account is somebody
#: else's (or nobody's) and must not be launched beside.
BLUEPRINT_LABEL_PREFIX = "blueprint-"


class ProviderJobScopeError(RuntimeError):
    """A job-scoped provider-zero reading could not be proven."""

    def __init__(self, blockers: Sequence[str]) -> None:
        self.blockers = [str(item) for item in blockers] or ["job_scope_unproven"]
        super().__init__(";".join(self.blockers))


def provider_zero_scope(environ: Mapping[str, str] | None = None) -> str:
    """``global`` unless the owner explicitly asked for ``job``.

    Anything that is not exactly ``job`` (a typo, an empty value) keeps the
    stricter account-wide rule: a misconfiguration must never widen spend.
    """

    raw = str((environ if environ is not None else os.environ).get(SCOPE_ENV) or "")
    return SCOPE_JOB if raw.strip().lower() == SCOPE_JOB else SCOPE_GLOBAL


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _owned_by_job(
    *,
    instance_id: str,
    name: str,
    own_label_prefixes: Sequence[str],
    own_instance_ids: Iterable[str],
) -> bool:
    return instance_id in set(own_instance_ids) or any(
        prefix and name.startswith(prefix) for prefix in own_label_prefixes
    )


def evaluate_job_scope(
    *,
    guard: Mapping[str, Any],
    now: datetime,
    own_label_prefixes: Sequence[str] = (),
    own_instance_ids: Iterable[Any] = (),
    live_inventory: Sequence[Mapping[str, Any]] | None = None,
    required_providers: Sequence[str] = ("vast",),
    max_guard_age_seconds: int = DEFAULT_MAX_GUARD_AGE_SECONDS,
    not_before: datetime | None = None,
    not_before_subject: str = "launch",
    added_instances: int = 1,
    added_burn_usd_per_hour: float = 0.0,
) -> dict[str, Any]:
    """Decide whether this job may proceed beside whatever else is live.

    ``guard`` is a reap-mode ``gpu_spend_guard.v1`` report.  ``live_inventory``
    is an optional fresher API reading (rows with ``instance_id`` and ``name``);
    every live row in it must also be a protected instance in the guard, so an
    instance created after the guard's snapshot waits for the next refresh
    instead of being assumed someone else's.

    ``added_instances`` and ``added_burn_usd_per_hour`` are the headroom this
    job needs. With both zero this is a *closure* reading (after teardown): it
    admits no new spend, so the fleet budget and the guard's admission status
    do not apply -- exactly as the account-wide closure check never looked at
    them -- while ownership, attribution and reaping still do.
    """

    blockers: list[str] = []
    admission = int(added_instances) > 0 or float(added_burn_usd_per_hour) > 0
    own_prefixes = [str(item) for item in own_label_prefixes if str(item or "")]
    own_ids = {str(item) for item in own_instance_ids if str(item or "")}

    # --- the guard itself: fresh, reaping, and not blocked for any reason
    if guard.get("schema_version") != SPEND_GUARD_SCHEMA_VERSION:
        blockers.append("job_scope_spend_guard_schema_invalid")
    generated_at = _timestamp(guard.get("generated_at"))
    if generated_at is None:
        blockers.append("job_scope_spend_guard_timestamp_invalid")
    else:
        age = (now - generated_at).total_seconds()
        if age < -_CLOCK_SKEW_SECONDS or age > max_guard_age_seconds:
            blockers.append("job_scope_spend_guard_stale")
        if not_before is not None and generated_at < not_before:
            blockers.append(f"job_scope_spend_guard_predates_{not_before_subject}")
    if guard.get("reap_mode") is not True:
        # Orphan reaping is what makes "every live instance has an owner" a
        # standing property rather than one observation.
        blockers.append("job_scope_spend_guard_reap_mode_missing")
    if admission and (guard.get("status") != "passed" or guard.get("blockers") not in ([], ())):
        # Covers billing reconciliation, inventory query failures, unknown
        # cost rates and the paid-spend admission lock.
        blockers.append("job_scope_spend_guard_blocked")
    inventories = {
        str(row.get("provider")): row
        for row in guard.get("inventory_results") or []
        if isinstance(row, Mapping)
    }
    if not required_providers:
        blockers.append("job_scope_required_provider_scope_missing")
    for provider in required_providers:
        inventory = inventories.get(str(provider))
        if inventory is None or inventory.get("status") != "succeeded":
            blockers.append(f"job_scope_provider_inventory_unverified:{provider}")
    if guard.get("reap_candidate_ids") not in ([], ()):
        blockers.append("job_scope_orphan_reap_candidates_remaining")
    for result in guard.get("reap_results") or []:
        if isinstance(result, Mapping) and result.get("status") != "terminated":
            blockers.append("job_scope_orphan_reap_not_confirmed")

    # --- the account-wide money cap, with room for this job
    budget = guard.get("fleet_budget")
    budget = budget if isinstance(budget, Mapping) else {}
    live_count = guard.get("live_instance_count")
    burn = _number(guard.get("total_burn_per_hour_usd"))
    max_live = budget.get("max_live_instances")
    max_burn = _number(budget.get("max_burn_usd_per_hour"))
    if admission and budget.get("status") != "passed":
        blockers.append("job_scope_fleet_budget_not_passed")
    if not admission:
        pass
    elif isinstance(max_live, bool) or not isinstance(max_live, int) or max_live < 1:
        # Without a configured ceiling the fleet budget is not a cap at all, and
        # the account-wide zero is the only thing bounding concurrent spend.
        blockers.append("job_scope_fleet_live_instance_ceiling_missing")
    elif isinstance(live_count, bool) or not isinstance(live_count, int):
        blockers.append("job_scope_fleet_live_instance_count_invalid")
    elif live_count + max(0, int(added_instances)) > max_live:
        blockers.append("job_scope_fleet_live_instance_headroom_exhausted")
    if not admission:
        pass
    elif max_burn is None:
        blockers.append("job_scope_fleet_burn_ceiling_missing")
    elif burn is None:
        blockers.append("job_scope_fleet_burn_unknown")
    elif burn + max(0.0, float(added_burn_usd_per_hour)) > max_burn:
        blockers.append("job_scope_fleet_burn_headroom_exhausted")
    max_daily = _number(budget.get("max_daily_spend_usd"))
    daily = _number(budget.get("daily_spend_usd"))
    if admission and max_daily is not None and (daily is None or daily >= max_daily):
        blockers.append("job_scope_fleet_daily_spend_headroom_exhausted")

    # --- every live instance: attributable, owned, and not this job's
    guard_live: dict[str, Mapping[str, Any]] = {}
    for row in guard.get("instances") or []:
        if not isinstance(row, Mapping) or row.get("live") is not True:
            continue
        guard_live[str(row.get("id") or "")] = row
    if isinstance(live_count, int) and not isinstance(live_count, bool) and live_count != len(guard_live):
        blockers.append("job_scope_spend_guard_live_inventory_inconsistent")
    job_rows: list[dict[str, Any]] = []
    foreign_rows: list[dict[str, Any]] = []

    def _classify(instance_id: str, name: str, *, protected: bool, source: str) -> None:
        row = {"instance_id": instance_id, "name": name, "source": source}
        if _owned_by_job(
            instance_id=instance_id, name=name,
            own_label_prefixes=own_prefixes, own_instance_ids=own_ids,
        ):
            blockers.append(f"job_scope_own_resource_live:{instance_id}")
            job_rows.append(row)
            return
        if not name.startswith(BLUEPRINT_LABEL_PREFIX):
            # Unlabeled or non-Blueprint spend cannot be attributed to another
            # job, so it might be this one's. Same rule as the adapter's
            # prelaunch inventory guard.
            blockers.append(f"job_scope_unattributable_live_resource:{instance_id}")
        elif not protected:
            # Live, labeled, but no live owner yet: either an orphan the reaper
            # has not reached or a create that has not recorded its owner. Wait.
            blockers.append(f"job_scope_unowned_live_resource:{instance_id}")
        foreign_rows.append({**row, "protected": protected})

    for instance_id, row in sorted(guard_live.items()):
        _classify(
            instance_id, str(row.get("name") or ""),
            protected=row.get("protected") is True, source="spend_guard",
        )
    if live_inventory is not None:
        for row in live_inventory:
            instance_id = str(row.get("instance_id") or row.get("id") or "")
            name = str(row.get("name") or row.get("label") or "")
            if instance_id in guard_live:
                continue
            # Newer than the guard snapshot: never assume it belongs to
            # someone else until the guard has seen and attributed it.
            blockers.append(f"job_scope_live_resource_not_in_spend_guard:{instance_id}")
            _classify(instance_id, name, protected=False, source="live_inventory")

    return {
        "scope": SCOPE_JOB,
        "mode": "admission" if admission else "closure",
        "verified": not blockers,
        "blockers": sorted(set(blockers)),
        "own_label_prefixes": own_prefixes,
        "own_instance_ids": sorted(own_ids),
        "job_live_resource_count": len(job_rows),
        "job_live_resources": job_rows,
        "foreign_live_resource_count": len(foreign_rows),
        "foreign_live_resources": foreign_rows,
        "fleet": {
            "live_instance_count": live_count,
            "max_live_instances": max_live,
            "total_burn_per_hour_usd": burn,
            "max_burn_usd_per_hour": max_burn,
            "daily_spend_usd": daily,
            "max_daily_spend_usd": max_daily,
            "added_instances": int(added_instances),
            "added_burn_usd_per_hour": float(added_burn_usd_per_hour),
        },
        "spend_guard_generated_at": guard.get("generated_at"),
    }


def load_spend_guard(path: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the reap-mode spend guard report and a content record of it."""

    resolved = Path(
        path or os.environ.get(GUARD_REPORT_ENV) or DEFAULT_GUARD_REPORT
    ).expanduser()
    try:
        if resolved.is_symlink():
            raise ProviderJobScopeError(["job_scope_spend_guard_unreadable"])
        payload = resolved.read_bytes()
        value = json.loads(payload)
    except (OSError, ValueError) as exc:
        raise ProviderJobScopeError(["job_scope_spend_guard_unreadable"]) from exc
    if not isinstance(value, Mapping):
        raise ProviderJobScopeError(["job_scope_spend_guard_unreadable"])
    record = {
        "path": str(resolved),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "generated_at": value.get("generated_at"),
    }
    return dict(value), record


def build_job_provider_zero_receipt(
    *,
    job_key: str,
    live_inventory: Sequence[Mapping[str, Any]],
    guard: Mapping[str, Any],
    guard_record: Mapping[str, Any],
    observed_at: datetime,
    own_label_prefixes: Sequence[str] = (),
    added_instances: int = 1,
    added_burn_usd_per_hour: float = 0.0,
    api_command: Sequence[str] = VAST_PROVIDER_ZERO_API_CALL,
) -> dict[str, Any]:
    """Seal an ``adp_paid_provider_job_zero.v1`` receipt or refuse.

    The receipt deliberately carries ``provider_zero: False`` and a different
    schema: any consumer that has not been taught job scope keeps rejecting it.
    """

    if not str(job_key or "").strip():
        raise ProviderJobScopeError(["job_scope_job_key_missing"])
    evaluation = evaluate_job_scope(
        guard=guard, now=observed_at, own_label_prefixes=own_label_prefixes,
        live_inventory=live_inventory, added_instances=added_instances,
        added_burn_usd_per_hour=added_burn_usd_per_hour,
    )
    if not evaluation["verified"]:
        raise ProviderJobScopeError(evaluation["blockers"])
    inventory = [
        {
            "instance_id": str(row.get("instance_id") or row.get("id") or ""),
            "name": str(row.get("name") or row.get("label") or ""),
        }
        for row in live_inventory
    ]
    receipt: dict[str, Any] = {
        "schema_version": JOB_PROVIDER_ZERO_SCHEMA_VERSION,
        "provider": "vast",
        "observed_at_utc": observed_at.astimezone(timezone.utc).isoformat(),
        "api_command": list(api_command),
        "api_confirmed": True,
        "scope": {
            "kind": SCOPE_JOB,
            "job_key": str(job_key),
            "own_label_prefixes": evaluation["own_label_prefixes"],
        },
        "job_live_resource_count": 0,
        "job_provider_zero": True,
        "provider_zero": False,
        "global_live_resource_count": len(inventory),
        "inventory": inventory,
        "foreign_live_resources_owned": True,
        "spend_guard": {
            "path": str(guard_record.get("path") or ""),
            "sha256": str(guard_record.get("sha256") or ""),
            "generated_at": guard.get("generated_at"),
        },
        "fleet": evaluation["fleet"],
        "blockers": [],
        "stderr_present": False,
        "raw_secret_values_recorded": False,
        "provider_zero_digest": "",
    }
    receipt["provider_zero_digest"] = canonical_digest(
        receipt, digest_field="provider_zero_digest"
    )
    return receipt


def validate_job_provider_zero_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Re-check a sealed job-scoped receipt without trusting its producer."""

    receipt = dict(value)
    scope = receipt.get("scope")
    scope = scope if isinstance(scope, Mapping) else {}
    fleet = receipt.get("fleet")
    fleet = fleet if isinstance(fleet, Mapping) else {}
    guard = receipt.get("spend_guard")
    guard = guard if isinstance(guard, Mapping) else {}
    inventory = receipt.get("inventory")
    prefixes = [str(item) for item in scope.get("own_label_prefixes") or []]
    if (
        receipt.get("schema_version") != JOB_PROVIDER_ZERO_SCHEMA_VERSION
        or receipt.get("provider") != "vast"
        or receipt.get("api_confirmed") is not True
        or not valid_vast_provider_zero_api_call(receipt.get("api_command"))
        or receipt.get("raw_secret_values_recorded") is not False
        or receipt.get("stderr_present") is not False
        or receipt.get("provider_zero_digest")
        != canonical_digest(receipt, digest_field="provider_zero_digest")
        or scope.get("kind") != SCOPE_JOB
        or not str(scope.get("job_key") or "").strip()
        or not str(guard.get("sha256") or "").startswith("sha256:")
        or _timestamp(guard.get("generated_at")) is None
        or _timestamp(receipt.get("observed_at_utc")) is None
    ):
        raise ProviderJobScopeError(["job_scope_receipt_invalid"])
    max_live = fleet.get("max_live_instances")
    live = fleet.get("live_instance_count")
    added = fleet.get("added_instances")
    if (
        receipt.get("job_live_resource_count") != 0
        or receipt.get("job_provider_zero") is not True
        or receipt.get("foreign_live_resources_owned") is not True
        or receipt.get("blockers") not in ([], ())
        or not isinstance(inventory, list)
        or receipt.get("global_live_resource_count") != len(inventory)
        or any(
            not isinstance(row, Mapping)
            or not str(row.get("name") or "").startswith(BLUEPRINT_LABEL_PREFIX)
            or any(p and str(row.get("name")).startswith(p) for p in prefixes)
            for row in inventory
        )
        or any(isinstance(item, bool) or not isinstance(item, int) for item in (max_live, live, added))
        or live + added > max_live
    ):
        raise ProviderJobScopeError(["job_scope_receipt_not_job_zero"])
    return receipt


def collect_job_scoped_vast_provider_zero(
    *,
    job_key: str,
    own_label_prefixes: Sequence[str] = (),
    guard_path: str | Path | None = None,
    inventory_collector: Callable[[], Mapping[str, Any]] | None = None,
    global_receipt_collector: Callable[..., Mapping[str, Any]] | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """One live Vast reading, returned as the strictest receipt it supports.

    Empty account: the classic ``adp_paid_provider_zero.v1`` receipt, exactly as
    before.  Otherwise: a job-scoped receipt if, and only if, everything live is
    another job's owned, labeled work and the fleet budget has room.
    """

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
    )


__all__ = [
    "BLUEPRINT_LABEL_PREFIX",
    "DEFAULT_GUARD_REPORT",
    "GLOBAL_PROVIDER_ZERO_SCHEMA_VERSION",
    "JOB_PROVIDER_ZERO_SCHEMA_VERSION",
    "SCOPE_ENV",
    "SCOPE_GLOBAL",
    "SCOPE_JOB",
    "ProviderJobScopeError",
    "build_job_provider_zero_receipt",
    "collect_job_scoped_vast_provider_zero",
    "evaluate_job_scope",
    "load_spend_guard",
    "provider_zero_scope",
    "validate_job_provider_zero_receipt",
]
