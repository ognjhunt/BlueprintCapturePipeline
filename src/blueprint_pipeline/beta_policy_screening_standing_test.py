"""The standing-test allowance: money for testing a policy on a job after the job's window.

Plan 1.6 (``docs/launch-readiness/2026-11-11-evaluation-service-plan.md``). A job's own screening
money (the site's $30, ``SITE_LIMIT``, inside the job's one dollar ceiling) is preparation money and
ends with the job's window. A later test of a newly released policy on a stored job is paid from this
separate allowance instead:

- **Per policy per job: ``POLICY_CAP_USD`` ($2.00, owner approval 2026-11-11).** What one policy (its
  ``policy_identity``, under any candidate id) may hold or spend on one job across every standing test,
  settled attempts at their settled cost.
  A run whose quote is above what is left is shortened to fit by the plan's own rule
  (``beta_policy_screening_plan.fit_run_to_cap``, the scripted reference row's rule), never raised.
- **Monthly total across all jobs: required, unset by default.** The owner has not decided it. It is
  read from a root-owned file (``ALLOWANCE_FILE_ENV``, default ``DEFAULT_ALLOWANCE_FILE``); while the
  file is absent or invalid every standing test is refused (``MONTHLY_TOTAL_UNSET`` /
  ``MONTHLY_TOTAL_INVALID``) and nothing is planned, enqueued or spent. A standing test's caps count
  against the UTC month its campaign was enqueued in, from the moment it is enqueued.
- **Inside the account month and the spend guard.** A standing campaign launches through the same
  coordinator, allocator and paid admission as any campaign, so the $200 account month
  (``account_monthly_spend``), the spend guard and the provider credit checks decide every launch
  unchanged. The monthly total may not exceed the account month's limit when that file is readable.
- **The site's permission is the WebApp's.** The WebApp's Atlas configuration read (the same read that
  carries ``job_ceiling``, ``beta_policy_screening_job_ceiling.read_job_ceiling_summary``) may carry one
  optional top-level field, ``standing_tests_allowed``. Only the JSON value ``true`` permits a standing
  test; absent or anything else is no. It is read before planning and again before every paid launch.

A standing campaign is marked by its plan (``STANDING_TEST_FIELD``, digest-bound) and, from it, by its
ledger state (``FUNDING_FIELD``). Its money is outside the site's allowance and the job's ceiling
(``beta_policy_screening_coordinator._budget`` and ``local_screening_holds`` leave it out), so a standing
test never takes a job's preparation money and the job's own figures never pay for one. Everything that
stops a run stops it: the WebApp's execution control (withdrawal, the owner's run closure, a refused or
expired authority), consent read through ``require_current_source`` and the site withdrawal check before
every reservation.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
from typing import Any

#: Owner approval, 2026-11-11: a standing test may hold or spend at most $2.00 per policy per job.
POLICY_CAP_USD = Decimal("2.00")
POLICY_CAP_DECISION = "owner approval 2026-11-11: USD 2.00 per policy per job"
ALLOWANCE_FILE_ENV = "BLUEPRINT_STANDING_TEST_ALLOWANCE_FILE"
DEFAULT_ALLOWANCE_FILE = Path("/etc/blueprint/standing-test-allowance.v1.json")
ALLOWANCE_SCHEMA = "blueprint.standing_test_allowance.v1"
_ALLOWANCE_KEYS = frozenset({"schema_version", "monthly_total_usd", "approved_by", "approved_at",
                             "authorization_reference"})
#: The WebApp's one field for the site's agreement to tests after the window: only ``true`` permits.
PERMISSION_FIELD = "standing_tests_allowed"
RECORD_SCHEMA = "blueprint.beta_screening_standing_test.v1"
_RECORD_KEYS = frozenset({"schema_version", "policy_cap_usd", "policy_cap_decision", "monthly_total_usd",
                          "month", "request_id", "authorization_reference"})
#: The ledger state's funding mark, copied from the plan at enqueue.
FUNDING_FIELD = "funding"
STANDING = "standing_test"
#: The plan's own mark (``beta_policy_screening_plan.STANDING_TEST_FIELD``), digest-bound.
STANDING_TEST_FIELD_NAME = "standing_test"

MONTHLY_TOTAL_UNSET = "standing_test_monthly_total_unset"
MONTHLY_TOTAL_INVALID = "standing_test_monthly_total_invalid"
NOT_ALLOWED = "standing_test_not_allowed_by_site"
POLICY_CAP_INSUFFICIENT = "standing_test_policy_cap_insufficient"
MONTHLY_TOTAL_INSUFFICIENT = "standing_test_monthly_total_insufficient"
CANNOT_FIT = "standing_test_run_cannot_fit_policy_cap"
_MONEY = re.compile(r"(?:0|[1-9][0-9]{0,5})\.[0-9]{2}")
_MONTH = re.compile(r"[0-9]{4}-(?:0[1-9]|1[0-2])")


def month_of(moment: datetime | None = None) -> str:
    return (moment or datetime.now(timezone.utc)).strftime("%Y-%m")


def allowance_path(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    configured = str(env.get(ALLOWANCE_FILE_ENV) or "").strip()
    return Path(configured) if configured else DEFAULT_ALLOWANCE_FILE


def monthly_total_usd(*, month: str | None = None, environ: Mapping[str, str] | None = None) -> Decimal:
    """The owner's monthly total for standing tests; refuses with a typed code while it is unset."""
    path = allowance_path(environ)
    if not path.is_symlink() and not path.exists():
        raise ValueError(MONTHLY_TOTAL_UNSET)
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16384:
            raise ValueError(MONTHLY_TOTAL_INVALID)
        value = json.loads(path.read_text(encoding="utf-8"))
        total = value.get("monthly_total_usd") if isinstance(value, dict) else None
        if (not isinstance(value, dict) or set(value) != _ALLOWANCE_KEYS or value["schema_version"] != ALLOWANCE_SCHEMA
                or not isinstance(total, str) or not _MONEY.fullmatch(total) or Decimal(total) <= 0
                or any(not isinstance(value[key], str) or not 1 <= len(value[key]) <= 300
                       for key in ("approved_by", "approved_at", "authorization_reference"))):
            raise ValueError(MONTHLY_TOTAL_INVALID)
        amount = Decimal(total)
    except (OSError, ValueError, UnicodeDecodeError, InvalidOperation):
        raise ValueError(MONTHLY_TOTAL_INVALID) from None
    from .account_monthly_spend import configured_policy, effective_limit

    policy, _error, _path = configured_policy(environ)
    if policy is not None and amount > effective_limit(policy, month or month_of()):
        raise ValueError(MONTHLY_TOTAL_INVALID)  # never more than the account month itself
    return amount


def site_allows(extras: Mapping[str, Any]) -> bool:
    """The WebApp's answer, read beside the job's ceiling: only ``true`` is yes."""
    return extras.get(PERMISSION_FIELD) is True


def require_site_permission(*, context: Mapping[str, Any], capture_root: str | Path) -> None:
    """Read the WebApp's permission now (the job-ceiling read); refuse unless it says ``true``."""
    from .beta_policy_screening_job_ceiling import read_job_ceiling_summary

    extras: dict[str, Any] = {}
    read_job_ceiling_summary(context=context, capture_root=capture_root, extras=extras)
    if not site_allows(extras):
        raise ValueError(NOT_ALLOWED)


def standing_deadline(context: Mapping[str, Any]) -> float | None:
    """A standing test's own end from the current source: the consent's expiry only.

    The job's execution deadline is its preparation window, which a standing test is for running after;
    the WebApp's execution control answers with the standing authority's deadline instead.
    """
    from .beta_policy_screening_control import _deadline

    rights = context.get("capture_rights") or {}
    return _deadline({"capture_rights": {"consent_expires_at_epoch": rights.get("consent_expires_at_epoch")}})


def require_launch_authority(plan: Mapping[str, Any], *, context: Mapping[str, Any]) -> float | None:
    """Before a standing campaign's paid launch: the allowance is still set and the site still agrees."""
    from .team_scene_packet_worker import resolve_capture_root

    if plan.get(STANDING_TEST_FIELD_NAME) is None:
        raise ValueError("beta_screening_standing_test_plan_invalid")
    monthly_total_usd()
    require_site_permission(context=context, capture_root=resolve_capture_root(plan["source"]))
    return standing_deadline(context)


def is_standing(state: Mapping[str, Any]) -> bool:
    """A ledger campaign (or a plan) funded from the standing-test allowance."""
    return state.get(FUNDING_FIELD) == STANDING or bool(state.get(STANDING_TEST_FIELD_NAME))


def funding_fields(plan: Mapping[str, Any]) -> dict[str, str]:
    """What the coordinator records on a new campaign's ledger state for its plan's funding."""
    return {FUNDING_FIELD: STANDING} if plan.get(STANDING_TEST_FIELD_NAME) else {}


def _row_usage(row: Mapping[str, Any], cap: Decimal) -> Decimal:
    from .beta_policy_screening_coordinator import _amount
    from .beta_policy_screening_result import QUEUED_STATUSES
    from .beta_policy_screening_settlement import settled_cost

    used = Decimal(0)
    attempts = [*row.get("attempt_history", []), *([row["attempt"]] if row.get("attempt") else [])]
    for attempt in attempts:
        if attempt["reservation_status"] == "reserved":
            used += _amount(attempt["maximum_cost_usd"])
        elif attempt["reservation_status"] == "committed":
            settled = settled_cost(attempt)
            used += _amount(attempt["maximum_cost_usd"]) if settled is None else settled
    if row.get("status") in QUEUED_STATUSES:
        used += cap  # funded at enqueue, not reserved yet
    return used


def standing_usage(queue_root: str | Path | None, *, month: str) -> dict[str, Any]:
    """What standing tests hold: ``month_usd`` for ``month``, ``by_policy[(site_id, candidate_id)]`` and
    ``rows``: each campaign row's ``(site_id, candidate_id, policy_identity, used)``. A row whose campaign
    catalog was not kept has no identity (``None``) and is counted by its candidate id alone."""
    from .beta_policy_screening_coordinator import _ledger, _lock, _plan, _root
    from .beta_policy_screening_plan import money
    from .beta_policy_screening_tested_pairs import campaign_catalog_rows, policy_identity

    root = _root(queue_root)
    with _lock(root):
        ledger = _ledger(root)
    month_usd, by_policy, rows = Decimal(0), {}, []
    for campaign_id, state in ledger["campaigns"].items():
        if not is_standing(state):
            continue
        plan = _plan(root, campaign_id, state["plan_digest"])
        catalog_rows = campaign_catalog_rows(root, plan)
        caps = {row["candidate_id"]: money(row["max_cost_usd"]) for row in plan["candidates"] if "max_cost_usd" in row}
        for candidate_id, row in state["candidates"].items():
            used = _row_usage(row, caps.get(candidate_id, Decimal(0)))
            key = (state["site_id"], candidate_id)
            by_policy[key] = by_policy.get(key, Decimal(0)) + used
            source = catalog_rows.get(candidate_id)
            rows.append((state["site_id"], candidate_id,
                         None if source is None else policy_identity(source)["policy_identity"], used))
            if str(state["created_at"])[:7] == month:
                month_usd += used
    return {"month": month, "month_usd": month_usd, "by_policy": by_policy, "rows": rows}


def policy_used(usage: Mapping[str, Any], *, site_id: str, candidate_id: str, identity: str) -> Decimal:
    """What one policy holds on one job against ``POLICY_CAP_USD``: every standing row of the same policy
    identity, under any candidate id, and every row under the same candidate id (so a row whose identity is
    unknown is never left out). Each row counts once."""
    return sum((used for row_site, row_candidate, row_identity, used in usage["rows"]
                if row_site == site_id and (row_identity == identity or row_candidate == candidate_id)), Decimal(0))


def _deferred(row: Mapping[str, Any], code: str) -> dict[str, Any]:
    kept = {key: row[key] for key in ("candidate_id", "display_name", "configuration_digest",
                                      "execution_evidence_kind", "reach_rank") if key in row}
    return {**kept, "status": "deferred", "blockers": [code]}


def standing_plan(plan: Mapping[str, Any], *, policy_left: Mapping[str, Decimal], month_left: Decimal,
                  record: Mapping[str, Any]) -> dict[str, Any]:
    """A planned limited campaign re-funded from the standing allowance: each runnable row fitted to what
    its policy has left of ``POLICY_CAP_USD`` and, in plan order, to what the month has left."""
    from .beta_policy_screening_g1 import BACKEND as G1_BACKEND
    from .beta_policy_screening_plan import (
        STANDING_TEST_FIELD, WAITLISTED, fit_run_to_cap, money, validate_screening_plan,
    )
    from .decision_evidence_contracts import canonical_digest

    value = json.loads(json.dumps(plan))
    left, planned, rows = Decimal(month_left), Decimal(0), []
    for row in value["candidates"]:
        if row["status"] not in {"selected", WAITLISTED}:
            rows.append(row)
            continue
        cap = min(POLICY_CAP_USD, Decimal(policy_left.get(row["candidate_id"], POLICY_CAP_USD)))
        spec, quote = row["execution_spec"], row["cost_quote"]
        fitted = None
        if cap > 0 and not (spec.get("backend") == G1_BACKEND and money(quote["maximum_cost_usd"]) > cap):
            fitted = fit_run_to_cap(spec, quote, cap, reason="fitted_to_standing_test_policy_cap",
                                    suffix="-standing-cap")
        if fitted is None:
            rows.append(_deferred(row, POLICY_CAP_INSUFFICIENT if cap < POLICY_CAP_USD else CANNOT_FIT))
            continue
        spec, quote = fitted
        maximum = money(quote["maximum_cost_usd"])
        if maximum > left:
            rows.append(_deferred(row, MONTHLY_TOTAL_INSUFFICIENT))
            continue
        left -= maximum
        planned += maximum
        rows.append({**row, "status": "selected", "blockers": [], "execution_spec": spec,
                     "admission_digest": canonical_digest(spec), "cost_quote": quote,
                     "max_cost_usd": quote["maximum_cost_usd"]})
    value["candidates"] = rows
    value.update(budget_usd=f"{planned:.2f}", maximum_planned_cost_usd=f"{planned:.2f}",
                 **{STANDING_TEST_FIELD: dict(record)})
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return validate_screening_plan(value) if planned > 0 else value


def standing_record(*, monthly_total: Decimal, month: str, request_id: str,
                    authorization_reference: str) -> dict[str, Any]:
    return {"schema_version": RECORD_SCHEMA, "policy_cap_usd": f"{POLICY_CAP_USD:.2f}",
            "policy_cap_decision": POLICY_CAP_DECISION, "monthly_total_usd": f"{monthly_total:.2f}",
            "month": month, "request_id": request_id, "authorization_reference": authorization_reference}


def validate_standing_plan(plan: Mapping[str, Any]) -> None:
    """A retained standing plan: its record, every funded row inside its cap, nothing waiting on money."""
    from .beta_policy_screening_known_good_control import is_control
    from .beta_policy_screening_plan import (
        _JOB_CEILING_FIELDS, STANDING_TEST_FIELD, WAITLISTED, BetaPolicyScreeningPlanError, money,
    )

    record = plan[STANDING_TEST_FIELD]
    if (not isinstance(record, dict) or set(record) != _RECORD_KEYS or record["schema_version"] != RECORD_SCHEMA
            or not isinstance(record["policy_cap_usd"], str) or not _MONEY.fullmatch(record["policy_cap_usd"])
            or not isinstance(record["monthly_total_usd"], str) or not _MONEY.fullmatch(record["monthly_total_usd"])
            or not isinstance(record["month"], str) or not _MONTH.fullmatch(record["month"])
            or any(not isinstance(record[key], str) or not 1 <= len(record[key]) <= 300
                   for key in ("policy_cap_decision", "request_id", "authorization_reference"))
            or _JOB_CEILING_FIELDS & set(plan)
            or any(is_control(row) or row["status"] == WAITLISTED for row in plan["candidates"])
            or any(money(row["max_cost_usd"]) > money(record["policy_cap_usd"])
                   for row in plan["candidates"] if row["status"] == "selected")
            or money(plan["budget_usd"]) != money(plan["maximum_planned_cost_usd"])):
        raise BetaPolicyScreeningPlanError("beta_screening_standing_test_plan_invalid")


__all__ = ["ALLOWANCE_FILE_ENV", "ALLOWANCE_SCHEMA", "CANNOT_FIT", "DEFAULT_ALLOWANCE_FILE", "FUNDING_FIELD",
           "MONTHLY_TOTAL_INSUFFICIENT", "MONTHLY_TOTAL_INVALID", "MONTHLY_TOTAL_UNSET", "NOT_ALLOWED",
           "PERMISSION_FIELD", "POLICY_CAP_DECISION", "POLICY_CAP_INSUFFICIENT", "POLICY_CAP_USD", "STANDING",
           "funding_fields", "is_standing", "month_of", "monthly_total_usd", "require_launch_authority",
           "policy_used", "require_site_permission", "site_allows", "standing_deadline", "standing_plan", "standing_record",
           "standing_usage", "validate_standing_plan"]
