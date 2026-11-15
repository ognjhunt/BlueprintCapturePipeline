"""The early trial's frozen screening ledger, read-only.

The owner dropped Blueprint's own early trial (2026-11-14); nothing writes this ledger any more,
but the host still keeps its campaigns and attempts. Two kept readers need it until the retained
data is gone: withdrawal deletion binds a withdrawn site's attempts through it
(``website_capture_withdrawal_screening``, ``website_capture_withdrawal_deletion``), and the
account spend meter counts its attempts (``paid_spend_meter_sources``). These are the pieces they
read, moved unchanged from the trial's coordinator, result, settlement, selection and review-video
modules. The store helpers it shares with the scene stage's holds live in ``scene_stage_holds``.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .scene_stage_holds import DEFAULT_QUEUE_ROOT, QUEUE_ROOT_ENV, ScreeningCoordinatorError, _read

# The bulk volume new attempts are reserved under (the unit sets
# /mnt/blueprint-work/beta-policy-screening-work): each attempt's sealed bundle, allocator job
# directory and launch receipts live in ``<work root>/attempts/<attempt id>``. The ledger, plans
# and publications stay in the queue root, beside a pointer per attempt. The ledger records each
# attempt's directory, so an attempt reserved without it (unset) keeps ``<queue>/attempts/<id>``.
WORK_ROOT_ENV = "BLUEPRINT_BETA_POLICY_SCREENING_WORK_ROOT"
ATTEMPT_DIR_FIELD = "attempt_dir"
LEDGER_SCHEMA = "blueprint.beta_policy_screening_ledger.v1"
LEDGER_FILENAME = "screening_ledger.v1.json"
#: The site's screening allowance across its campaigns (raised from $15 on 2026-11-05, owner
#: approval). Each website job's ceiling still bounds what a job spends in all.
SITE_LIMIT = Decimal("30")
SETTLEMENT_HELD_FIELD = "settlement_held"
ACTIVE_STATUSES = frozenset({"preparing", "launching", "running", "reconciliation_required"})
SETTLEMENT_SCHEMA = "blueprint.beta_policy_screening_settlement.v1"
SETTLEMENT_BASIS = "adapter_budget_ledger_estimate"
#: Where a campaign's candidate selection was kept (``queue_root/selections``).
SELECTION_DIRECTORY = "selections"
#: Each capture's retained review videos (``captures/*/pipeline/<dir>``) and the lock withdrawal takes on them.
REVIEW_VIDEOS_DIRNAME = "beta_screening_review_videos"
REVIEW_VIDEOS_LOCK_NAME = "retention.lock"
_MONEY = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]{1,6})?")


def money(value: Decimal) -> str:
    """Decimal money on the wire; never round a commitment down."""
    if not value.is_finite() or value < 0 or value != value.quantize(Decimal("0.000001")):
        raise ValueError("beta_screening_money_precision_invalid")
    return format(value, ".6f").rstrip("0").rstrip(".") if value else "0"


def _amount(value: Any) -> Decimal:
    if not isinstance(value, str) or not _MONEY.fullmatch(value) or len(value) > 32:
        raise ScreeningCoordinatorError("beta_screening_money_invalid")
    return Decimal(value)


def settled_cost(attempt: Mapping[str, Any]) -> Decimal | None:
    """What a committed attempt's recorded settlement counts, or None to count its whole maximum."""
    settled = attempt.get("settled_cost_usd")
    if settled is None:
        return None
    receipt = attempt.get("settlement_receipt")
    try:
        if (not isinstance(settled, str) or not _MONEY.fullmatch(settled) or not isinstance(receipt, Mapping)
                or attempt.get("settlement_basis") != SETTLEMENT_BASIS
                or receipt.get("schema_version") != SETTLEMENT_SCHEMA
                or receipt.get("settlement_basis") != SETTLEMENT_BASIS
                or receipt.get("settled_cost_usd") != settled
                or receipt.get("settled_at") != attempt.get("settled_at")
                or receipt.get("attempt_id") != attempt.get("attempt_id")
                or receipt.get("maximum_cost_usd") != attempt.get("maximum_cost_usd")
                or receipt.get("receipt_digest") != attempt.get("settlement_receipt_digest")
                or receipt["receipt_digest"] != canonical_digest(dict(receipt), digest_field="receipt_digest")):
            return None
        value, maximum = Decimal(settled), Decimal(str(attempt["maximum_cost_usd"]))
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return None
    return value if 0 <= value <= maximum else None


def _budget(ledger: Mapping[str, Any], site_id: str, *, campaign_id: str | None = None) -> dict[str, Any]:
    """What the site (or one campaign) holds, as funding counts it.

    A committed attempt counts at its recorded settlement, else at its whole
    maximum; a reserved one at its maximum.
    """
    reserved, spent = Decimal(0), Decimal(0)
    for key, campaign in ledger["campaigns"].items():
        if campaign["site_id"] != site_id or campaign_id is not None and key != campaign_id:
            continue
        for row in campaign["candidates"].values():
            attempts = [*row.get("attempt_history", []), *([row["attempt"]] if row.get("attempt") else [])]
            for attempt in attempts:
                cap = _amount(attempt["maximum_cost_usd"])
                state = attempt["reservation_status"]
                if state == "reserved":
                    reserved += cap
                elif state == "committed":
                    settled = settled_cost(attempt)
                    spent += cap if settled is None else settled
                elif state != "released":
                    raise ScreeningCoordinatorError("beta_screening_reservation_state_invalid")
    return {"currency": "USD", "limit_usd": money(SITE_LIMIT),
            "committed_upper_bound_usd": money(reserved + spent),
            "spent_upper_bound_usd": money(spent), "reserved_usd": money(reserved)}


def _ledger(root: Path) -> dict[str, Any]:
    path = root / LEDGER_FILENAME
    if not path.exists():
        if any((root / "attempts").iterdir()):
            raise ScreeningCoordinatorError("beta_screening_ledger_missing_after_attempt")
        return {"schema_version": LEDGER_SCHEMA, "campaigns": {}}
    ledger = _read(path)
    if (ledger.get("schema_version") != LEDGER_SCHEMA or not isinstance(ledger.get("campaigns"), dict)
            or ledger.get("ledger_digest") != canonical_digest(ledger, digest_field="ledger_digest")):
        raise ScreeningCoordinatorError("beta_screening_ledger_invalid")
    for site_id in {row["site_id"] for row in ledger["campaigns"].values()}:
        if _amount(_budget(ledger, site_id)["committed_upper_bound_usd"]) > SITE_LIMIT:
            raise ScreeningCoordinatorError("beta_screening_site_budget_exceeded")
    return ledger


def _attempt_path(root: Path, attempt: Mapping[str, Any]) -> Path:
    """The attempt's directory: the one the ledger recorded when it was reserved, else beside the ledger."""
    recorded = attempt.get(ATTEMPT_DIR_FIELD)
    if recorded is None:
        return root / "attempts" / attempt["attempt_id"]
    path = Path(recorded) if isinstance(recorded, str) else Path()
    if (not path.is_absolute() or ".." in path.parts or path.name != attempt["attempt_id"]
            or path.parent.name != "attempts"):
        raise ScreeningCoordinatorError("beta_screening_attempt_path_unsafe")
    return path


__all__ = [
    "ACTIVE_STATUSES", "ATTEMPT_DIR_FIELD", "DEFAULT_QUEUE_ROOT", "LEDGER_FILENAME", "LEDGER_SCHEMA",
    "QUEUE_ROOT_ENV", "REVIEW_VIDEOS_DIRNAME", "REVIEW_VIDEOS_LOCK_NAME", "SELECTION_DIRECTORY", "SETTLEMENT_HELD_FIELD",
    "ScreeningCoordinatorError", "WORK_ROOT_ENV", "money", "settled_cost",
]
