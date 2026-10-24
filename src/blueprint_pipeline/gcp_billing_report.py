"""Validate an operator-retained official GCP project billing CSV.

This is a bounded bridge for projects without a historical BigQuery export.
It does not query billing, estimate posted charges, or refresh source timestamps.
The root-managed receipt retains the original CSV and report context together.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlsplit

SCHEMA = "blueprint.gcp_official_billing_report.v1"
MAX_AGE = timedelta(hours=24)
MAX_BYTES = 2 * 1024 * 1024


class GCPBillingReportError(ValueError):
    """The retained report cannot establish the requested billing scope."""


def read_report(
    path: str | Path,
    *,
    billing_account_id: str,
    project_ids: list[str],
    cohort_start: datetime,
    now: datetime,
) -> tuple[float, datetime, dict]:
    """Return exact posted net charges, source age, and retained audit bytes.

    The provider report has calendar-date precision. Only cohorts beginning at
    midnight on the same calendar date are supported; partial-day cohorts fail.
    Report end dates may include the current month's future dates, as Google's
    monthly chart does; captured_at remains the actual observation time.
    """
    try:
        source = Path(path)
        info = source.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size > MAX_BYTES
            or stat.S_IMODE(info.st_mode) & 0o022
            or info.st_uid not in {0, os.geteuid()}
        ):
            raise GCPBillingReportError("gcp_billing_report_file_unsafe")
        raw = source.read_bytes()
        if len(raw) > MAX_BYTES:
            raise GCPBillingReportError("gcp_billing_report_too_large")
        report = json.loads(raw)
        if not isinstance(report, dict) or report.get("schema_version") != SCHEMA:
            raise GCPBillingReportError("gcp_billing_report_schema_invalid")
        if (
            not re.fullmatch(r"[A-Z0-9]{6}(?:-[A-Z0-9]{6}){2}", billing_account_id)
            or report.get("billing_account_id") != billing_account_id
            or not project_ids
            or len(set(project_ids)) != len(project_ids)
            or report.get("currency") != "USD"
            or report.get("source_kind") != "operator_downloaded_official_report"
            or report.get("charge_type") != "posted_net_cost"
        ):
            raise GCPBillingReportError("gcp_billing_report_scope_invalid")
        url = urlsplit(report["source_url"])
        if (
            url.scheme != "https"
            or url.netloc != "console.cloud.google.com"
            or url.path.split(";", 1)[0] != f"/billing/{billing_account_id}/reports"
            or url.fragment
        ):
            raise GCPBillingReportError("gcp_billing_report_source_invalid")
        context = report["report_context_text"]
        if (
            not isinstance(context, str)
            or report["source_url"] not in context
            or any(project not in context for project in project_ids)
        ):
            raise GCPBillingReportError("gcp_billing_report_context_invalid")
        captured = datetime.fromisoformat(report["captured_at"].replace("Z", "+00:00"))
        if captured.tzinfo is None:
            raise GCPBillingReportError("gcp_billing_report_timestamp_invalid")
        captured = captured.astimezone(timezone.utc)
        if captured > now or now - captured > MAX_AGE:
            raise GCPBillingReportError("gcp_billing_report_stale_or_future")
        start = date.fromisoformat(report["period_start_date"])
        end = date.fromisoformat(report["period_end_date"])
        if (
            cohort_start.time().isoformat() != "00:00:00"
            or start != cohort_start.date()
            or end < captured.date()
            or end.year != captured.year
            or end.month != captured.month
        ):
            raise GCPBillingReportError("gcp_billing_report_period_mismatch")
        csv_text = report["csv_utf8"]
        if hashlib.sha256(csv_text.encode("utf-8")).hexdigest() != report["csv_sha256"]:
            raise GCPBillingReportError("gcp_billing_report_csv_digest_mismatch")
        rows = csv.DictReader(io.StringIO(csv_text))
        required = {"Project ID", "Project number", "Unrounded subtotal ($)", "Other savings ($)"}
        if not required.issubset(rows.fieldnames or []) or len(rows.fieldnames or []) != len(
            set(rows.fieldnames or [])
        ):
            raise GCPBillingReportError("gcp_billing_report_columns_invalid")
        projects, summaries = {}, {}
        for row in rows:
            if None in row:
                raise GCPBillingReportError("gcp_billing_report_row_invalid")
            amount = Decimal(row["Unrounded subtotal ($)"])
            if not amount.is_finite():
                raise GCPBillingReportError("gcp_billing_report_amount_invalid")
            project = row["Project ID"]
            if project:
                if project in projects or not re.fullmatch(r"[0-9]+", row["Project number"]):
                    raise GCPBillingReportError("gcp_billing_report_project_invalid")
                projects[project] = amount
            else:
                label = row["Other savings ($)"]
                if label not in {"Subtotal", "Tax", "Filtered total"} or label in summaries:
                    raise GCPBillingReportError("gcp_billing_report_summary_invalid")
                summaries[label] = amount
        total = sum(projects.values(), Decimal(0))
        if (
            set(projects) != set(project_ids)
            or total < 0
            or set(summaries) != {"Subtotal", "Tax", "Filtered total"}
            or summaries["Subtotal"] != total
            or summaries["Tax"] < 0
            or summaries["Filtered total"] != total + summaries["Tax"]
        ):
            raise GCPBillingReportError("gcp_billing_report_total_or_projects_mismatch")
        total = summaries["Filtered total"]
        if total > Decimal("1000000000"):
            raise GCPBillingReportError("gcp_billing_report_amount_invalid")
        audit = {
            "provider": "gcp",
            "endpoint": report["source_url"],
            "evidence_origin": report["source_kind"],
            "source_captured_at": captured.isoformat(),
            "period_start_date": start.isoformat(),
            "period_end_date": end.isoformat(),
            "period_precision": "provider_calendar_date",
            "project_ids": sorted(projects),
            "csv_sha256": report["csv_sha256"],
            "response_digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "response_size_bytes": len(raw),
            "response_bytes": raw,
        }
        return float(total), captured, audit
    except GCPBillingReportError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, InvalidOperation) as exc:
        raise GCPBillingReportError("gcp_billing_report_invalid_or_missing") from exc
