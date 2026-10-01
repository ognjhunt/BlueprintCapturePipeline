"""Sanitized upsert plan for the existing authorized Notion publisher.

This module never calls Notion. The publisher must query Source key, update the
existing page (including clearing corrected amounts), then read back its digest.
"""

from __future__ import annotations

DAILY_PAGE = "3ec80154-161d-817d-b48d-fa0adfb30c1a"
DATA_SOURCE = "22a9c161-930d-4c6f-b7d7-de7cb4a4a065"


def build_notion_projection(snapshot: dict) -> dict:
    if snapshot.get("evidence_origin") != "retained_provider_response":
        return {
            "status": "unverified_source",
            "page_id": DAILY_PAGE,
            "data_source_id": DATA_SOURCE,
            "upserts": [],
        }
    upserts = []
    publication_gaps = []
    for row in snapshot.get("rows", []):
        if not row["stable_correction_identity"]:
            publication_gaps.append(
                {
                    "source_key": row["source_key"],
                    "gap": "stable_charge_identity_missing_supersession_unresolved",
                }
            )
            continue
        kind = row["kind"]
        notion_kind = (
            "Daily usage"
            if kind == "actual"
            and row["grain"] == "interval"
            and row["reporting_day"]
            and not row["partial"]
            else "Invoice expense"
            if kind == "invoice_reference"
            else "Coverage gap"
        )
        gap = []
        if row["allocation_gap"]:
            gap.append(row["allocation_gap"])
        if row["partial"]:
            gap.append("source_row_partial")
        if row["provider_freshness"] != "current":
            gap.append("provider_freshness_" + row["provider_freshness"])
        if kind == "cumulative_actual":
            gap.append("cumulative_cohort_not_daily_usage")
        if row["grain"] == "year":
            gap.append("year_grain_no_daily_breakdown")
        # No cash/prepaid/reservation rows are synthesized. Estimates remain separate.
        amount = row["amount"] if row["currency"] == "USD" and not row["partial"] else None
        properties = {
            "Entry": f"{row['provider']} / {row['service']} / source interval",
            "Source key": row["source_key"],
            "Provider": row["provider"],
            "Service": row["service"],
            "Account / project": " / ".join(v for v in (row["account"], row["project"]) if v)
            or "Unknown",
            "Kind": notion_kind,
            "Amount": amount,
            "Currency": "USD" if row["currency"] == "USD" else "Unverified",
            "Basis": "Estimated"
            if kind == "estimate"
            else "Actual"
            if amount is not None
            else "Unknown",
            "Expense rollup": "Reference only",
            "Expense MTD": None,
            "MTD status": "Incomplete",
            "Payment status": "Not applicable",
            "Scope": "Blueprint context",
            "Freshness": "Current source check"
            if row["provider_freshness"] == "current"
            else "Dated evidence"
            if row["provider_freshness"] == "stale"
            else "Missing",
            "Coverage gap": "; ".join(gap),
            "Workflow": "Existing provider billing evidence",
            "Reconciliation": f"revision={row['revision_id']}; sourceTZ={row['source_timezone']}; snapshot={snapshot['snapshot_digest']}",
            "Next action": "Reconcile retained source scope and coverage; no provider calls or new access.",
        }
        for prop, value in (
            ("Period start", row["interval_start"]),
            ("Period end", row["interval_end"]),
            ("Source observed", row["source_observed_at"]),
            ("Source checked", snapshot["generated_at"]),
            ("Day", row["reporting_day"]),
        ):
            properties[f"date:{prop}:start"] = value
            properties[f"date:{prop}:is_datetime"] = 0 if prop == "Day" else 1
        upserts.append(
            {
                "source_key": row["source_key"],
                "revision_id": row["revision_id"],
                "properties": properties,
                "clear_properties": [k for k, v in properties.items() if v is None],
                "provenance": row["provenance"],
            }
        )
    return {
        "status": "plan_only_partial",
        "page_id": DAILY_PAGE,
        "data_source_id": DATA_SOURCE,
        "snapshot_digest": snapshot["snapshot_digest"],
        "source_collected_at": snapshot["source_collected_at"],
        "upserts": upserts,
        "gaps": snapshot["coverage"]["gaps"],
        "daily_total": None,
        "publication_gaps": publication_gaps,
        "local_metered_calls": snapshot.get(
            "local_metered_calls", {"coverage": "unavailable", "billed_cost": None}
        ),
        "warning": "Coverage incomplete. Missing daily usage is unknown; provider freshness is separate from refresh time.",
        "publication_contract": "upsert_by_source_key_clear_corrected_values_and_read_back_revision",
    }
