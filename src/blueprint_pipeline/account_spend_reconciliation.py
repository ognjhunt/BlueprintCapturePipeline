"""Month-scoped, per provider: settled ledger spend against the provider's billing delta.

G16 / WP-13 / O-13. The ledger (``paid_spend_meter``: what Blueprint's own
records settled with a provider this UTC month) and the provider's bill (the
same month's reconciled reading, ``account_monthly_spend_report``) were on
different scopes and compared nowhere. This compares them per provider, on the
month, inside the account monthly spend report.

Tolerance (owner, O-13): the larger of ``$5`` or 10 % of the provider's billing
delta. Outside it is a warning (``account_spend_reconciliation_outside_tolerance``),
never a refusal: nothing here admits, refuses or changes a count. A provider
whose billing for the month is not on the host, or was not read as observed
this month, reads ``unknown`` with the reason; it is never compared as zero.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from .paid_spend_meter import money, usd

SCHEMA = "blueprint.account_monthly_spend_reconciliation.v1"
TOLERANCE_FLOOR_USD = Decimal("5")
TOLERANCE_FRACTION = Decimal("0.10")


def reconcile(providers: Mapping[str, Mapping[str, Any]], *, period: str | None) -> dict[str, Any]:
    """One row per provider of the account report: ``within_tolerance``, ``outside_tolerance`` or ``unknown``."""

    rows: dict[str, dict[str, Any]] = {}
    for name, row in sorted(providers.items()):
        if not isinstance(row, Mapping):
            continue
        ledger = money(row.get("metered_settled_usd")) or Decimal(0)
        billing = money(row.get("reconciled_month_usd")) if row.get("reconciled_month_usd") is not None else None
        result: dict[str, Any] = {"ledger_settled_usd": usd(ledger),
                                  "billing_month_usd": None if billing is None else usd(billing)}
        if billing is None or row.get("status") != "observed":
            result.update(status="unknown",
                          reason=str(row.get("reason") or f"billing_reading_{row.get('status') or 'missing'}"))
        else:
            tolerance = max(TOLERANCE_FLOOR_USD, billing * TOLERANCE_FRACTION)
            difference = ledger - billing
            result.update(difference_usd=usd(difference), tolerance_usd=usd(tolerance),
                          status="within_tolerance" if abs(difference) <= tolerance else "outside_tolerance")
        rows[str(name)] = result
    outside = sorted(name for name, row in rows.items() if row["status"] == "outside_tolerance")
    unknown = sorted(name for name, row in rows.items() if row["status"] == "unknown")
    return {"schema_version": SCHEMA, "period": period, "providers": rows, "outside_tolerance": outside,
            "unknown": unknown, "tolerance_rule": "max(USD 5, 10% of the billing delta); outside is a warning"}


def main(argv: list[str] | None = None) -> int:
    """Print the reconciliation from the current account report (reads only)."""

    from .account_monthly_spend import configured_policy, read_report, report_path

    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    policy, _error, _config = configured_policy()
    path = report_path(policy)
    try:
        report = read_report(path) if path is not None else None
    except (OSError, ValueError, UnicodeDecodeError):
        report = None
    if not isinstance(report, Mapping):
        print(json.dumps({"status": "unknown", "reason": "account_monthly_spend_report_unreadable",
                          "report_path": str(path) if path else None}, indent=2))
        return 1
    value = report.get("reconciliation") or reconcile(report.get("providers") or {}, period=report.get("period"))
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


__all__ = ["SCHEMA", "TOLERANCE_FLOOR_USD", "TOLERANCE_FRACTION", "reconcile"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
