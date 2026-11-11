"""Standing-policy extensions of a website scene's spend ceiling (owner decision 2026-11-10).

A job's dollar ceiling is its expected budget, not a wall that needs a
person. When a reservation does not fit, and the job's signed grant says
``automatic_budget_extension: true``, the ceiling rises by exactly the
shortfall, provided that

- the account's monthly admission would admit that amount now: a fresh,
  bound account report (``account_monthly_spend``), no admission blocker,
  and the month's limit less what the report counts less every standing
  extension issued since the report was written covers the shortfall;
- the job has paid attempts left (the caller asks only after its attempt
  checks passed);
- the new ceiling stays within the schema bound every scene ceiling has.

Anything else, a stale or unreadable account report included, issues
nothing: the job waits exactly as before and its next timed re-ask asks
again. An extension is permission, not spend: every paid launch is still
checked at ``paid_resource_admission.require_paid_resource_admission``,
which stays the hard stop.

Records live in their own store beside the owner's
(``execution-budget-extensions/``), with their own sealed chain, written
under the same intake lock. The owner's chain and its reader are
untouched, so an older release ignores this store and sees the owner's
ceiling. The effective ceiling is the larger of the two, so it never
decreases. A grant without the field (every grant issued before this
existed) never extends.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import datetime
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

SCHEMA = "task_evaluation_scene_standing_budget_extension.v1"
DIRECTORY = "standing-budget-extensions"
CHANNEL = "standing_policy"
GRANT_FIELD = "automatic_budget_extension"
REASON = "reservation_exceeds_ceiling"
#: The bound every scene ceiling already has (``_limits`` in the owner's chain); not a new figure.
SCHEMA_CEILING_USD = Decimal(1000)
_MICRO = Decimal("0.000001")
_FIELDS = frozenset({
    "schema_version", "issuer_channel", "intent_id", "intent_digest", "grant_authority_digest", "sequence",
    "predecessor_digest", "reason", "attempt_id", "amount_usd", "prior_ceiling_usd", "owner_ceiling_usd",
    "resulting_ceiling_usd", "account_reading", "issued_at_epoch", "provider_mutation_performed",
    "historical_reservations_released", "extension_digest",
})


def _usd(value: Any) -> Decimal:
    return Decimal(str(value))


def _float(value: Decimal) -> float:
    return float(value.quantize(_MICRO, rounding=ROUND_CEILING))


def _records(directory: Path, intent: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every standing record of this intent, chain-validated, in sequence order."""
    from . import task_evaluation_scene_intent_contracts as contracts
    root = directory / DIRECTORY
    contracts._require(not root.is_symlink(), "standing_extension_store_unsafe")
    if not root.exists():
        return []
    rows = []
    for path in root.iterdir():
        contracts._require(path.is_file() and not path.is_symlink() and path.suffix == ".json",
                           "standing_extension_record_invalid")
        row = contracts._read(path, "extension_digest")
        contracts._require(set(row) == _FIELDS and row["schema_version"] == SCHEMA
                           and row["issuer_channel"] == CHANNEL
                           and row["intent_id"] == intent["intent_id"]
                           and row["intent_digest"] == intent["intent_digest"]
                           and path.name == row["extension_digest"].removeprefix("sha256:") + ".json"
                           and row["provider_mutation_performed"] is False
                           and row["historical_reservations_released"] is False
                           and isinstance(row["account_reading"], dict)
                           and all(contracts._number(row[k]) for k in (
                               "amount_usd", "prior_ceiling_usd", "owner_ceiling_usd", "resulting_ceiling_usd",
                               "issued_at_epoch")),
                           "standing_extension_record_invalid")
        rows.append(row)
    rows.sort(key=lambda row: row["sequence"])
    predecessor, ceiling = intent["intent_digest"], None
    for index, row in enumerate(rows, 1):
        amount, prior, result = _usd(row["amount_usd"]), _usd(row["prior_ceiling_usd"]), _usd(row["resulting_ceiling_usd"])
        contracts._require(row["sequence"] == index and row["predecessor_digest"] == predecessor
                           and amount > 0 and result == prior + amount and result <= SCHEMA_CEILING_USD
                           and prior >= _usd(row["owner_ceiling_usd"])
                           and (ceiling is None or prior >= ceiling),
                           "standing_extension_chain_invalid")
        predecessor, ceiling = row["extension_digest"], result
    return rows


def effective_scene_budget(directory: str | Path, intent: Mapping[str, Any]) -> dict[str, Any]:
    """The owner's effective budget, with the spend ceiling raised by any standing extension.

    ``extension_digest`` stays the owner's chain head (what an attempt row
    binds); the standing chain is reported beside it.
    """
    from .task_evaluation_scene_execution_budget_evidence import effective_execution_budget
    directory = Path(directory)
    owner = effective_execution_budget(directory, intent)
    rows = _records(directory, intent)
    budget = {**owner, "owner_max_total_spend_usd": owner["max_total_spend_usd"],
              "standing_extension_count": len(rows),
              "standing_extension_digest": rows[-1]["extension_digest"] if rows else None}
    if rows and _usd(rows[-1]["resulting_ceiling_usd"]) > _usd(owner["max_total_spend_usd"]):
        budget["max_total_spend_usd"] = rows[-1]["resulting_ceiling_usd"]
    return budget


def granted_authority(request: Mapping[str, Any], queue_root: str | Path, now: float) -> dict[str, Any] | None:
    """The job's retained signed grant when it allows automatic extension, else ``None``.

    Read from the grant the job was prepared under (retained on this host and
    bound by its digest). Only a website scene-preparation intent has one.
    """
    if (request.get("execution") or {}).get("purpose") != "scene_preparation":
        return None
    from .website_preparation_authority import _retained_sources
    try:
        _context, authority, _valid_until = _retained_sources(request=request, queue_root=queue_root, now=now)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return dict(authority) if authority.get(GRANT_FIELD) is True else None


def _epoch(value: Any) -> float | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.timestamp() if parsed.tzinfo is not None else None


def _issued_since(root: Path, *, period: str, since: float) -> Decimal:
    """Standing extensions every intent in this root issued against this month after ``since``."""
    from . import task_evaluation_scene_intent_contracts as contracts
    total = Decimal(0)
    for path in root.glob("scene-*/" + DIRECTORY + "/*.json"):
        row = contracts._read(path, "extension_digest")
        reading = row.get("account_reading") or {}
        contracts._require(row.get("schema_version") == SCHEMA and contracts._number(reading.get("read_at_epoch"))
                           and contracts._number(row.get("amount_usd")), "standing_extension_record_invalid")
        if reading.get("period") == period and reading["read_at_epoch"] >= since:
            total += _usd(row["amount_usd"])
    return total


def account_reading(root: Path, *, wall: float) -> tuple[dict[str, Any] | None, str | None]:
    """The account reading an extension would be admitted against, or ``(None, why not)``.

    No limit configured is not a reading: nothing is admitted against it.
    """
    from . import account_monthly_spend as account
    from .paid_spend_meter import money, month_of
    policy, error, _ = account.configured_policy()
    if policy is None:
        return None, error or account.LIMIT_NOT_CONFIGURED
    blockers = account.admission_blockers(now=wall)
    if blockers:
        return None, blockers[0]
    try:
        report = account.read_report(Path(policy["report_path"]))
        period = month_of(wall)
        limit = account.effective_limit(policy, period)
    except (OSError, ValueError, UnicodeDecodeError, ArithmeticError):
        return None, account.REPORT_INVALID
    counted, generated = money(report.get("counted_usd")), _epoch(report.get("generated_at"))
    if counted is None or generated is None:
        return None, account.REPORT_INVALID
    issued = _issued_since(root, period=period, since=generated)
    return {"period": period, "limit_usd": str(limit), "counted_usd": str(counted),
            "report_generated_at": report.get("generated_at"), "policy_sha256": policy["policy_sha256"],
            "standing_extensions_since_report_usd": str(issued),
            "headroom_usd": str(limit - counted - issued), "read_at_epoch": wall}, None


def extend_for_shortfall(*, root: Path, directory: Path, intent: Mapping[str, Any], budget: Mapping[str, Any],
                         required_usd: Decimal, attempt_id: str, now: float,
                         wall: float | None = None) -> dict[str, Any] | None:
    """Under the intake lock: raise the ceiling to ``required_usd``, or return ``None`` and issue nothing.

    ``budget`` is ``effective_scene_budget`` read under the same lock, and the
    caller has already passed its paid-attempt checks for this reservation.
    """
    from . import task_evaluation_scene_intent_contracts as contracts
    from .task_evaluation_scene_intent_storage import write_exclusive
    ceiling = _usd(budget["max_total_spend_usd"])
    shortfall = (required_usd - ceiling).quantize(_MICRO, rounding=ROUND_CEILING)
    if shortfall <= 0 or ceiling + shortfall > SCHEMA_CEILING_USD:
        return None
    grant = granted_authority(intent["request"], root, now)
    if grant is None:
        return None
    reading, _why = account_reading(root, wall=time.time() if wall is None else wall)
    if reading is None or shortfall > Decimal(reading["headroom_usd"]):
        return None
    rows = _records(directory, intent)
    value = contracts._seal({
        "schema_version": SCHEMA, "issuer_channel": CHANNEL, "intent_id": intent["intent_id"],
        "intent_digest": intent["intent_digest"], "grant_authority_digest": grant.get("authority_digest"),
        "sequence": len(rows) + 1,
        "predecessor_digest": rows[-1]["extension_digest"] if rows else intent["intent_digest"],
        "reason": REASON, "attempt_id": attempt_id, "amount_usd": float(shortfall),
        "prior_ceiling_usd": float(ceiling), "owner_ceiling_usd": budget["owner_max_total_spend_usd"],
        "resulting_ceiling_usd": float(ceiling + shortfall), "account_reading": reading,
        "issued_at_epoch": now, "provider_mutation_performed": False,
        "historical_reservations_released": False}, "extension_digest")
    store = directory / DIRECTORY
    contracts._require(not store.is_symlink(), "standing_extension_store_unsafe")
    store.mkdir(mode=0o750, exist_ok=True)
    write_exclusive(store / (value["extension_digest"].removeprefix("sha256:") + ".json"), value)
    return effective_scene_budget(directory, intent)


__all__ = [
    "CHANNEL",
    "DIRECTORY",
    "GRANT_FIELD",
    "SCHEMA",
    "account_reading",
    "effective_scene_budget",
    "extend_for_shortfall",
    "granted_authority",
]
