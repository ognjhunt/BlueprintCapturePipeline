"""What bounds a site's screening money once testing is no longer rationed (owner decision 2026-11-11).

The owner's words: "yes replace them - $300 monthly limit, $100 fuse". A website job's plan now
budgets everything the job's one dollar ceiling leaves after its build
(``beta_policy_screening_plan.build_job_ceiling_screening_plan``), with no $15 plan size and no
$30 site allowance below it. The coordinator still keeps one site figure (``_budget``'s
``limit_usd``, which every publication carries as ``budget.limit_usd``), and every reservation
and refill checks it. That figure is the largest job ceiling (``maximum_cost_usd``) any of the
site's campaigns recorded when it was enqueued (``SITE_LIMIT_FIELD``); a campaign that recorded none
(one enqueued before this change, or a plan made without a website job ceiling, which no production
path builds) counts at the coordinator's ``SITE_LIMIT``, $30, as it was planned. So a site with one
older campaign and a $25 ceiling has a $30 site figure: the job's ceiling still binds every plan
(its budget) and every launch (``require_current_job_ceiling``). The figure adds no limit of its own.
A site past it holds only itself (the coordinator's ``beta_screening_site_budget_exceeded``).

When a reservation, a refill or the scripted reference check finds no room in a plan sized to the
job's ceiling, the reason it records is the ceiling's own (``JOB_CEILING_DEFERRED``), never a site
budget (``budget_reason``).

All of this runs only with the switch at ``fuse`` (``website_job_limits``). At ``nested`` (the
shipped value) every site's limit is ``SITE_LIMIT`` ($30) whatever its campaigns recorded, nothing
is recorded, and reasons are the site budget's, all as before 2026-11-11; a site that recorded a
higher ceiling under ``fuse`` and committed past $30 is then held (``site_limit``). Reading a
recorded figure, and a retained plan sized to its job's ceiling (``plan_site_limit``), never raises
in either mode.
"""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

#: Recorded on a ledger campaign enqueued from a plan sized to its website job's ceiling.
SITE_LIMIT_FIELD = "site_limit_usd"
#: The site allowance before 2026-11-11, for campaigns that recorded no ceiling and plans without one.
LEGACY_SITE_LIMIT_USD = Decimal("30")
#: ``beta_policy_screening_plan.JOB_CEILING_BASIS`` and ``JOB_CEILING_DEFERRED``, named here rather than
#: imported so this module reaches no GPU lane (``tests/test_live_pipeline_import_isolation.py``); a test
#: pins them to the plan's own.
JOB_CEILING_BASIS = "website_job_ceiling"
JOB_CEILING_DEFERRED = "website_job_ceiling_insufficient"


def _ceiling_maximum(plan: Mapping[str, Any]) -> Decimal | None:
    """The job ceiling's ``maximum_cost_usd`` a plan was sized under, or None for a plan without one."""
    if plan.get("budget_basis") != JOB_CEILING_BASIS:
        return None
    record = plan.get("website_job_ceiling")
    summary = record.get("summary") if isinstance(record, Mapping) else None
    value = summary.get("maximum_cost_usd") if isinstance(summary, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    amount = Decimal(str(value))
    return amount if amount.is_finite() and amount > 0 else None


def plan_site_limit(plan: Mapping[str, Any], legacy: Decimal = LEGACY_SITE_LIMIT_USD) -> Decimal:
    """The most a plan's budget may be: its job's ceiling, else ``legacy`` (the coordinator's ``SITE_LIMIT``)."""
    maximum = _ceiling_maximum(plan)
    return legacy if maximum is None else maximum


def campaign_fields(plan: Mapping[str, Any]) -> dict[str, str]:
    """The ledger fields a new campaign records: its job ceiling, when it was planned within one (``fuse`` only)."""
    from .website_job_limits import fuse

    maximum = _ceiling_maximum(plan) if fuse() else None
    return {} if maximum is None else {SITE_LIMIT_FIELD: format(maximum.normalize(), "f")}


def site_limit(ledger: Mapping[str, Any], site_id: str, legacy: Decimal = LEGACY_SITE_LIMIT_USD) -> Decimal:
    """The site's screening limit: the largest figure any of its campaigns recorded.

    A campaign without a recorded figure counts at ``legacy`` (the coordinator's ``SITE_LIMIT``), as it
    was planned. A site with no campaign yet has ``legacy`` (nothing reserves before a campaign exists).
    A recorded figure is a decimal string; one that cannot be read counts as ``legacy``.
    """
    from .website_job_limits import fuse

    if not fuse():
        return legacy  # nested: $30 for every site, as before; a site past it is held.
    figures = []
    for state in ledger.get("campaigns", {}).values():
        if state.get("site_id") != site_id:
            continue
        figures.append(_recorded(state.get(SITE_LIMIT_FIELD)) or legacy)
    return max(figures, default=legacy)


def _recorded(value: Any) -> Decimal | None:
    try:
        amount = Decimal(value) if isinstance(value, str) else None
    except ArithmeticError:
        return None
    return amount if amount is not None and amount.is_finite() and amount > 0 else None


def budget_reason(plan: Mapping[str, Any], code: str) -> str:
    """``code`` for a plan without a job ceiling, or at ``nested``; the ceiling's own reason for a plan sized to one."""
    from .website_job_limits import fuse

    if not fuse():
        return code
    return JOB_CEILING_DEFERRED if _ceiling_maximum(plan) is not None else code


__all__ = ["LEGACY_SITE_LIMIT_USD", "SITE_LIMIT_FIELD", "budget_reason", "campaign_fields", "plan_site_limit",
           "site_limit"]
