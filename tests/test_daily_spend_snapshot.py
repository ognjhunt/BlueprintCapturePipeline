import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from blueprint_pipeline.daily_spend_snapshot import (
    _hash,
    chicago_day,
    project_daily_snapshot,
    write_daily_snapshot,
)
from blueprint_pipeline.metered_call_evidence import digest, read_attempts
from blueprint_pipeline.notion_spend_projection import build_notion_projection
from blueprint_pipeline.provider_billing_reconciler import reconcile_provider_billing
from tests.test_provider_billing_reconciler import _secrets, _Transport

NOW = datetime(2026, 10, 1, 18, tzinfo=timezone.utc)


def _inputs(amount=4, currency="USD", id=7):
    export = {
        "currency": "USD",
        "generated_at": NOW.isoformat(),
        "provider_totals_usd": {"vast": amount},
    }
    receipt = {
        "generated_at": NOW.isoformat(),
        "cohort_start_at": "2026-01-01T00:00:00Z",
        "cohort_end_at": NOW.isoformat(),
        "receipt_digest": "sha256:test",
    }
    source = {
        "provider": "vast",
        "endpoint": "https://console.vast.ai/api/v0/charges/",
        "response_digest": "sha256:response",
        "request_query_digest": "sha256:query",
    }
    row = {
        "id": id,
        "day": 1790812800,
        "start": "2026-10-01T07:00:00Z",
        "end": "2026-10-01T08:00:00Z",
        "source": "instance-private",
        "project_id": "project-private",
        "amount": amount,
        "currency": currency,
        "items": [
            {"type": name, "amount": amount if name == "gpu" else 0}
            for name in ("gpu", "disk", "bwd", "bwu")
        ],
    }
    return export, receipt, [(source, {"success": True, "results": [row], "next_token": None})]


def _snapshot(inputs=None, **kwargs):
    return project_daily_snapshot(*(inputs or _inputs()), now=NOW, **kwargs)


def _vast(snapshot):
    return next(r for r in snapshot["rows"] if r["service"] == "charges")


def test_offline_daily_rows_preserve_components_provenance_and_unknown_global_coverage():
    inputs = _inputs()
    original = deepcopy(inputs)
    snapshot = _snapshot(inputs)
    row = _vast(snapshot)
    assert row["amount"] == 4 and row["components"] == {
        "gpu": 4,
        "disk": 0,
        "download": 0,
        "upload": 0,
    }
    assert row["reporting_day"] == "2026-10-01" and row["source_day"] == 1790812800
    assert row["provider_freshness"] == "unknown"
    assert snapshot["coverage"]["daily_total"] is None and snapshot["provider_requests_added"] == 0
    assert snapshot["observed_daily_slices"] == [
        {
            "provider": "vast",
            "day": "2026-10-01",
            "currency": "USD",
            "amount": 4,
            "partial_day_coverage": True,
        }
    ]
    assert inputs == original
    assert "project-private" not in json.dumps(snapshot) and "instance-private" not in json.dumps(
        snapshot
    )


@pytest.mark.parametrize("currency", [None, "", "US dollars"])
def test_null_or_invalid_currency_never_zero(currency):
    snapshot = _snapshot(_inputs(currency=currency))
    assert _vast(snapshot)["currency"] is None and _vast(snapshot)["billed_cost"] is None
    assert _vast(snapshot)["partial"] and snapshot["observed_daily_slices"] == []


def test_partial_pages_components_and_unknown_charge_id_are_inspectable_not_complete():
    inputs = _inputs(id=None)
    inputs[2][0][1]["next_token"] = "private-cursor"
    inputs[2][0][1]["results"][0]["items"][0]["amount"] = 9
    snapshot = _snapshot(inputs)
    assert _vast(snapshot)["partial"] and _vast(snapshot)["billed_cost"] is None
    assert snapshot["observed_daily_slices"] == []
    assert "vast_pagination_partial_or_unknown" in snapshot["coverage"]["gaps"]
    assert "private-cursor" not in json.dumps(snapshot)


def test_duplicate_credit_corrections_and_posting_changes():
    first = _snapshot()
    second = _snapshot(_inputs(-1), previous=first)
    assert _vast(first)["source_key"] == _vast(second)["source_key"]
    assert _vast(first)["revision_id"] != _vast(second)["revision_id"]
    assert _vast(second)["adjustment_kind"] == "credit"
    assert len(second["revision_history"]) == 4
    assert second["posting_changes"][0]["amount"] == -5
    assert second["posting_changes"][0]["reporting_day"] is None
    inputs = _inputs()
    inputs[2][0][1]["results"] *= 2
    assert len(_snapshot(inputs)["observed_daily_slices"]) == 1
    assert _snapshot(inputs)["observed_daily_slices"][0]["amount"] == 4
    inputs[2][0][1]["results"][1] = deepcopy(inputs[2][0][1]["results"][1])
    inputs[2][0][1]["results"][1]["amount"] = 9
    conflict = _snapshot(inputs)
    assert _vast(conflict)["amount"] is None and _vast(conflict)["billed_cost"] is None
    assert conflict["observed_daily_slices"] == []
    inputs[2][0][1]["results"].append(inputs[2][0][1]["results"][0])
    assert _vast(_snapshot(inputs))["amount"] is None  # A later identical row cannot erase the conflict.


@pytest.mark.parametrize(
    "start,end,day",
    [
        ("2026-03-08T06:00:00Z", "2026-03-09T05:00:00Z", "2026-03-08"),
        ("2026-11-01T05:00:00Z", "2026-11-02T06:00:00Z", "2026-11-01"),
        ("2026-10-01T00:00:00Z", "2026-10-02T00:00:00Z", None),
        (None, None, None),
        ("2026-10-01", "2026-10-02", None),
    ],
)
def test_utc_chicago_dst_exact_allocation_only(start, end, day):
    assert chicago_day(start, end) == day


def test_year_grain_runpod_and_stale_do_source_are_not_daily_or_refreshed_by_pointer():
    export, receipt, responses = _inputs()
    responses += [
        (
            {"provider": "runpod", "endpoint": "https://rest.runpod.io/v1/billing/pods"},
            [{"amount": 2}],
        ),
        (
            {
                "provider": "digitalocean",
                "endpoint": "https://api.digitalocean.com/v2/customers/my/balance",
            },
            {"generated_at": "2026-09-20T12:00:00Z"},
        ),
        (
            {
                "provider": "digitalocean",
                "endpoint": "https://api.digitalocean.com/v2/customers/my/invoices",
            },
            {
                "invoices": [],
                "invoice_preview": {"invoice_period": "2026-10", "amount": 2},
                "links": {"pages": {}},
            },
        ),
    ]
    snapshot = _snapshot((export, receipt, responses))
    runpod = next(r for r in snapshot["rows"] if r["provider"] == "runpod")
    assert (
        runpod["grain"] == "year"
        and runpod["reporting_day"] is None
        and runpod["interval_start"] is None
        and runpod["billed_cost"] is None
    )
    preview = next(r for r in snapshot["rows"] if r["service"] == "invoice_preview")
    assert preview["kind"] == "estimate" and preview["provider_freshness"] == "stale"
    assert snapshot["generated_at"] == NOW.isoformat()


def test_invoice_preview_overlap_on_later_page_and_aws_exclusion():
    export, receipt, responses = _inputs()
    export["provider_totals_usd"]["aws"] = 99
    endpoint = "https://api.digitalocean.com/v2/customers/my/invoices"
    responses += [
        ({"provider": "aws", "endpoint": "https://ce.us-east-1.amazonaws.com"}, {"Amount": 99}),
        (
            {"provider": "digitalocean", "endpoint": endpoint},
            {
                "invoices": [],
                "invoice_preview": {"invoice_period": "2026-10", "amount": 2},
                "links": {"pages": {"next": "opaque"}},
            },
        ),
        (
            {"provider": "digitalocean", "endpoint": endpoint},
            {
                "invoices": [{"invoice_uuid": "final", "invoice_period": "2026-10", "amount": 2}],
                "links": {"pages": {}},
            },
        ),
    ]
    snapshot = _snapshot((export, receipt, responses))
    assert "digitalocean_preview_overlaps_finalized_invoice" in snapshot["coverage"]["gaps"]
    assert all(r["provider"] != "aws" for r in snapshot["rows"])
    assert next(r for r in snapshot["rows"] if r["service"] == "invoice_preview")["partial"]
    assert len(snapshot["observed_daily_slices"]) == 1  # Vast only; invoice not added.


def test_bound_offline_export_digest_and_fixture_publication_guard(tmp_path):
    export = tmp_path / "provider_billing_export.json"
    source = tmp_path / "audit" / "provider_billing_source_receipt.json"
    source.parent.mkdir()
    exp, receipt, responses = _inputs()
    exp["schema_version"] = "blueprint.provider_billing_export.v1"
    receipt["schema_version"] = "blueprint.provider_billing_source_receipt.v1"
    receipt["provider_totals_usd"] = exp["provider_totals_usd"]
    raw = json.dumps(responses[0][1]).encode()
    response = source.parent / "response-001-vast.json"
    response.write_bytes(raw)
    receipt["sources"] = [
        responses[0][0]
        | {
            "retained_path": "/original/response-001-vast.json",
            "response_size_bytes": len(raw),
            "response_digest": digest(raw),
        }
    ]
    receipt["receipt_digest"] = _hash({k: v for k, v in receipt.items() if k != "receipt_digest"})
    source.write_text(json.dumps(receipt))
    export.write_text(json.dumps(exp))
    assert write_daily_snapshot(export, source, source.parent)["status"] == "written"
    snapshot = json.loads((tmp_path / "daily_spend_snapshot.json").read_text())
    assert snapshot["notion_projection"]["upserts"] == []
    response.write_bytes(b"{}")
    assert write_daily_snapshot(export, source, source.parent)["status"] == "unavailable"
    assert json.loads(export.read_text()) == exp


def test_existing_reconciler_keeps_call_count_export_and_admission_contract(tmp_path):
    transport = _Transport(digitalocean_generated_at=NOW.isoformat())
    result = reconcile_provider_billing(
        secrets_dir=_secrets(tmp_path),
        billing_export_path=tmp_path / "guard" / "provider_billing_export.json",
        audit_root=tmp_path / "guard" / "billing-audit",
        start_at="2026-01-01T00:00:00Z",
        now=NOW,
        transport=transport,
    )
    assert len(transport.requests) == 7
    assert not any("amazonaws.com" in url for url, _ in transport.requests)
    assert len(read_attempts(tmp_path / "guard" / "metered-call-events")) == 7
    assert result["daily_spend_snapshot"]["status"] == "written"
    snapshot = json.loads((tmp_path / "guard" / "daily_spend_snapshot.json").read_text())
    assert (
        snapshot["evidence_origin"] == "development_only"
        and snapshot["notion_projection"]["upserts"] == []
    )


def test_notion_plan_dedup_corrections_and_partial_source_day_guard():
    snapshot = _snapshot(evidence_origin="retained_provider_response")
    plan = build_notion_projection(snapshot)
    assert plan["status"] == "plan_only_partial" and plan["daily_total"] is None
    assert all(row["properties"]["Expense rollup"] == "Reference only" for row in plan["upserts"])
    assert all(
        row["properties"]["Kind"] not in {"Cash payment", "Prepaid funding", "Reservation estimate"}
        for row in plan["upserts"]
    )
    corrected = _snapshot(_inputs(currency=None), evidence_origin="retained_provider_response")
    row = next(
        r
        for r in build_notion_projection(corrected)["upserts"]
        if r["properties"]["Service"] == "charges"
    )
    assert row["properties"]["Amount"] is None and "Amount" in row["clear_properties"]
    assert row["source_key"] == _vast(snapshot)["source_key"]
    inputs = _inputs()
    inputs[2][0][1]["results"][0]["start"] = "2026-10-01T00:00:00Z"
    inputs[2][0][1]["results"][0]["end"] = "2026-10-02T00:00:00Z"
    assert (
        next(
            r
            for r in build_notion_projection(
                _snapshot(inputs, evidence_origin="retained_provider_response")
            )["upserts"]
            if r["properties"]["Service"] == "charges"
        )["properties"]["Kind"]
        == "Coverage gap"
    )
