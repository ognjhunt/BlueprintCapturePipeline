"""Development-only reports protect cumulative scope and source freshness."""

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from blueprint_pipeline.gcp_billing_report import GCPBillingReportError, SCHEMA, read_report
from blueprint_pipeline.provider_billing_reconciler import reconcile_provider_billing

NOW = datetime(2026, 10, 24, 2, tzinfo=timezone.utc)
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
ACCOUNT = "AAAAAA-BBBBBB-CCCCCC"
URL = f"https://console.cloud.google.com/billing/{ACCOUNT}/reports;grouping=GROUP_BY_PROJECT"
CSV = (
    "Project name,Project ID,Project number,Cost ($),Savings programs ($),Other savings ($),Unrounded subtotal ($),Subtotal ($),Percent change\n"
    "test,test-project,123456789,8.12,0,0,8.123456,8.12,New\n"
    ",,,,,Subtotal,8.123456,8.12\n,,,,,Tax,0.000000,0.00\n,,,,,Filtered total,8.123456,8.12\n"
)


def _receipt(tmp_path, **changes):
    report = {
        "schema_version": SCHEMA,
        "source_kind": "operator_downloaded_official_report",
        "charge_type": "posted_net_cost",
        "currency": "USD",
        "billing_account_id": ACCOUNT,
        "source_url": URL,
        "report_context_text": URL + " test-project 123456789",
        "captured_at": (NOW - timedelta(hours=1)).isoformat(),
        "period_start_date": "2026-01-01",
        "period_end_date": "2026-10-31",
        "csv_utf8": CSV,
        "csv_sha256": hashlib.sha256(CSV.encode()).hexdigest(),
    }
    report.update(changes)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    path.chmod(0o600)
    return path


def _read(path, **changes):
    args = dict(
        billing_account_id=ACCOUNT, project_ids=["test-project"], cohort_start=START, now=NOW
    )
    args.update(changes)
    return read_report(path, **args)


def test_retains_exact_unrounded_total_and_original_bytes(tmp_path):
    path = _receipt(tmp_path)
    total, captured, audit = _read(path)
    assert total == 8.123456
    assert captured == NOW - timedelta(hours=1)
    assert audit["response_bytes"] == path.read_bytes()
    assert audit["evidence_origin"] == "operator_downloaded_official_report"


@pytest.mark.parametrize(
    "changes",
    [
        {"captured_at": (NOW - timedelta(hours=25)).isoformat()},
        {"captured_at": (NOW + timedelta(seconds=1)).isoformat()},
        {"captured_at": "2026-10-24T01:00:00"},
        {"currency": "EUR"},
        {"charge_type": "forecast"},
        {"billing_account_id": "DDDDDD-EEEEEE-FFFFFF"},
        {"period_start_date": "2026-10-01"},
        {"period_end_date": "2026-10-23"},
        {"period_end_date": "2026-11-30"},
        {"csv_sha256": "0" * 64},
        {"source_url": "https://example.com/reports"},
        {"report_context_text": "unbound"},
    ],
)
def test_wrong_stale_or_unbound_reports_refuse(tmp_path, changes):
    with pytest.raises(GCPBillingReportError):
        _read(_receipt(tmp_path, **changes))


@pytest.mark.parametrize(
    "csv_text",
    [
        CSV.replace("test-project", "wrong-project"),
        CSV.replace("8.123456", "NaN"),
        CSV.replace("8.123456", "-8.123456"),
        CSV.replace("Subtotal,8.123456", "Subtotal,9.123456"),
        CSV + "test,test-project,123456789,8.12,0,0,8.123456,8.12,New\n",
        CSV.replace("Unrounded subtotal ($)", "Unrounded subtotal (EUR)"),
        CSV.replace("Project name,", "Project ID,"),
        CSV.replace(",,,,,Tax,0.000000,0.00\n", ""),
    ],
)
def test_invalid_csv_cannot_supply_a_provider_total(tmp_path, csv_text):
    path = _receipt(
        tmp_path, csv_utf8=csv_text, csv_sha256=hashlib.sha256(csv_text.encode()).hexdigest()
    )
    with pytest.raises(GCPBillingReportError):
        _read(path)


def test_partial_day_or_incomplete_project_scope_refuses(tmp_path):
    path = _receipt(tmp_path)
    for changes in (
        {"cohort_start": START + timedelta(hours=1)},
        {"project_ids": ["test-project", "another-project"]},
    ):
        with pytest.raises(GCPBillingReportError):
            _read(path, **changes)


def test_symlink_and_writable_report_refuse(tmp_path):
    path = _receipt(tmp_path)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(GCPBillingReportError):
        _read(link)
    path.chmod(0o666)
    with pytest.raises(GCPBillingReportError):
        _read(path)


def test_reconciler_retains_original_source_age_and_blocks_after_expiry(tmp_path):
    from scripts import gpu_spend_guard as guard

    path = _receipt(tmp_path)
    args = dict(
        secrets_dir=tmp_path / "missing-secrets",
        billing_export_path=tmp_path / "export.json",
        audit_root=tmp_path / "audit",
        start_at=START.isoformat(),
        gcp_report_receipt=path,
        gcp_billing_account_id=ACCOUNT,
        gcp_project_ids=["test-project"],
    )
    result = reconcile_provider_billing(**args, now=NOW)
    assert result["provider_totals_usd"] == {"gcp": 8.123456}
    export = json.loads((tmp_path / "export.json").read_text())
    assert export["generated_at"] == (NOW - timedelta(hours=1)).isoformat()
    live = guard.GpuInstance(
        provider="gcp", id="stopped", name="stopped", state="terminated", booted=False, live=True
    )
    accepted = guard.reconcile_billing_export(
        billing_export_path=tmp_path / "export.json",
        instances=[live],
        now=NOW.timestamp(),
        required=True,
    )
    assert accepted["status"] == "reconciled"
    receipt = json.loads(open(result["source_receipt_path"]).read())
    assert receipt["sources"][0]["source_captured_at"] == export["generated_at"]
    before = (tmp_path / "export.json").read_bytes()
    with pytest.raises(RuntimeError, match="no_provider_covered"):
        reconcile_provider_billing(**args, now=NOW + timedelta(days=1))
    assert (tmp_path / "export.json").read_bytes() == before
    expired = guard.reconcile_billing_export(
        billing_export_path=tmp_path / "export.json",
        instances=[live],
        now=(NOW + timedelta(days=1)).timestamp(),
        required=True,
    )
    assert "provider_billing_export_stale_or_invalid_time" in expired["blockers"]
