"""Metered spend, stage 1: every paid admission and its settlement, per job, summing to the totals."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from blueprint_pipeline import paid_spend_meter as meter
from blueprint_pipeline import paid_spend_meter_sources as sources
from blueprint_pipeline.beta_policy_screening_coordinator import LEDGER_FILENAME, LEDGER_SCHEMA
from blueprint_pipeline.beta_policy_screening_settlement import SETTLEMENT_BASIS, SETTLEMENT_SCHEMA
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    build_paid_lane_admission,
    require_paid_resource_admission,
)

NOW = datetime(2026, 11, 20, 16, 0, tzinfo=timezone.utc).timestamp()
LAST_MONTH = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc).timestamp()


@pytest.fixture
def journal(tmp_path, monkeypatch):
    root = tmp_path / "paid-spend-meter"
    monkeypatch.setenv(meter.METER_ROOT_ENV, str(root))
    monkeypatch.delenv(meter.JOB_ID_ENV, raising=False)
    return root


def _events(root: Path) -> list[dict]:
    events, gaps = meter.read_journal(root)
    assert gaps == []
    return events


def test_every_grant_is_journaled_against_the_job_it_names(journal, monkeypatch):
    admission = {**build_paid_lane_admission(resource_class="vast_provider_adapter"),
                 "meter": {"job_id": "site-req-1", "meter_key": "attempt-1", "held_usd": "4.00"}}
    require_paid_resource_admission(admission, resource_class="vast_provider_adapter",
                                    expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    monkeypatch.setenv(meter.JOB_ID_ENV, "req-2")
    require_paid_resource_admission(build_paid_lane_admission(resource_class="openai_api_candidate"),
                                    resource_class="openai_api_candidate",
                                    expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    monkeypatch.delenv(meter.JOB_ID_ENV)
    require_paid_resource_admission({"schema_version": "example.v1", "status": "admitted", "blockers": []},
                                    resource_class="cpu_build", expected_schema_version="example.v1")

    events = _events(journal)
    assert [(row["job_id"], row["resource_class"], row["provider"], row["held_usd"]) for row in events] == [
        ("req-1", "vast_provider_adapter", "vast", "4.00"),
        ("req-2", "openai_api_candidate", "openai", None),
        (meter.UNATTRIBUTED, "cpu_build", None, None),
    ]
    assert all(row["event"] == "admission" and row["schema_version"] == meter.EVENT_SCHEMA for row in events)


def test_an_admission_that_carries_its_request_id_is_metered_against_it(journal, monkeypatch):
    monkeypatch.setenv(meter.JOB_ID_ENV, "req-from-env")
    admission = {**build_paid_lane_admission(resource_class="provider_reconstruction_api"),
                 "request_id": "req-9", "allocation_binding_digest": "sha256:" + "a" * 64}
    require_paid_resource_admission(admission, resource_class="provider_reconstruction_api",
                                    expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    [row] = _events(journal)
    assert (row["job_id"], row["meter_key"], row["provider"]) == ("req-9", "sha256:" + "a" * 64, "world_labs")


def test_a_refused_admission_is_not_metered(journal):
    with pytest.raises(PaidResourceAdmissionBlocked):
        require_paid_resource_admission(build_paid_lane_admission(resource_class="vast_provider_adapter",
                                                                  blockers=["budget_missing"]),
                                        resource_class="vast_provider_adapter",
                                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    assert _events(journal) == []


def test_an_unwritable_meter_never_blocks_the_admission(tmp_path, monkeypatch):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("x")
    monkeypatch.setenv(meter.METER_ROOT_ENV, str(blocked))
    grant = require_paid_resource_admission(build_paid_lane_admission(resource_class="vast_provider_adapter"),
                                            resource_class="vast_provider_adapter",
                                            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    assert grant.resource_class == "vast_provider_adapter"


def test_no_meter_root_off_the_host_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv(meter.METER_ROOT_ENV, raising=False)
    monkeypatch.setattr(meter, "DEFAULT_METER_ROOT", tmp_path / "absent" / "paid-spend-meter")
    assert meter.meter_root() is None
    require_paid_resource_admission(build_paid_lane_admission(resource_class="vast_provider_adapter"),
                                    resource_class="vast_provider_adapter",
                                    expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    assert not (tmp_path / "absent").exists()


def test_journal_folds_admission_and_settlement_and_counts_repeats_once(tmp_path):
    root = tmp_path / "meter"
    meter.record_admission(job_id="req-1", meter_key="k1", resource_class="vast_provider_adapter",
                           held_usd="6.00", root=root, now=NOW - 100)
    meter.record_admission(job_id="req-1", meter_key="k1", resource_class="vast_provider_adapter",
                           held_usd="6.00", root=root, now=NOW - 90)
    meter.record_settlement(job_id="req-1", meter_key="k1", settled_usd="2.50", basis="fixture",
                            provider="vast", root=root, now=NOW - 50)
    meter.record_admission(job_id="req-1", meter_key="k2", resource_class="vast_provider_adapter",
                           held_usd="3.00", root=root, now=NOW - 40)
    meter.record_admission(job_id="req-2", meter_key="k3", resource_class="cpu_build", root=root, now=NOW - 30)
    rows = {row["meter_key"]: row for row in meter.journal_rows(_events(root))}
    assert rows["k1"]["status"] == "settled" and rows["k1"]["settled_usd"] == Decimal("2.50")
    assert rows["k2"]["status"] == "held" and rows["k2"]["held_usd"] == Decimal("3.00")
    assert rows["k3"]["status"] == "unpriced"
    view = meter.meter_view(rows.values(), now=NOW)
    jobs = {row["job_id"]: row for row in view["jobs"]}
    assert jobs["req-1"]["settled_usd"] == 2.5 and jobs["req-1"]["held_usd"] == 3.0
    assert jobs["req-1"]["admissions"] == 2 and jobs["req-2"]["unpriced_admissions"] == 1
    with pytest.raises(ValueError):
        meter.record_settlement(job_id="req-1", meter_key="k2", settled_usd="-1", basis="fixture", root=root)


def _settled_attempt(attempt_id: str, cap: str, settled: str, at: str) -> dict:
    receipt = {"schema_version": SETTLEMENT_SCHEMA, "settlement_basis": SETTLEMENT_BASIS, "settled_cost_usd": settled,
               "settled_at": at, "attempt_id": attempt_id, "maximum_cost_usd": cap}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return {"attempt_id": attempt_id, "maximum_cost_usd": cap, "reservation_status": "committed",
            "settled_cost_usd": settled, "settlement_basis": SETTLEMENT_BASIS, "settled_at": at,
            "settlement_receipt": receipt, "settlement_receipt_digest": receipt["receipt_digest"]}


def write_screening_ledger(queue: Path) -> None:
    queue.mkdir(parents=True, exist_ok=True)
    ledger = {"schema_version": LEDGER_SCHEMA, "campaigns": {
        "campaign-a": {"site_id": "site-req-1", "candidates": {
            "cand-1": {"attempt": {"attempt_id": "screen-1", "maximum_cost_usd": "4.00",
                                   "reservation_status": "reserved"}},
            "cand-2": {"attempt": _settled_attempt("screen-2", "4.00", "1.25", "2026-11-05T10:00:00+00:00"),
                       "attempt_history": [{"attempt_id": "screen-0", "maximum_cost_usd": "4.00",
                                            "reservation_status": "released"}]},
            "cand-3": {"attempt": {"attempt_id": "screen-3", "maximum_cost_usd": "4.00",
                                   "reservation_status": "committed"}},
        }},
        "campaign-b": {"site_id": "site-req-2", "candidates": {
            "cand-1": {"attempt": _settled_attempt("screen-4", "4.00", "0.91", "2026-10-29T10:00:00+00:00")},
        }},
    }}
    ledger["ledger_digest"] = canonical_digest(ledger, digest_field="ledger_digest")
    (queue / LEDGER_FILENAME).write_text(json.dumps(ledger))


def test_screening_attempts_are_read_from_the_screening_ledger(tmp_path):
    write_screening_ledger(tmp_path / "queue")
    rows, gaps = sources.screening_rows(tmp_path / "queue")
    assert gaps == []
    by_key = {row["meter_key"]: row for row in rows}
    assert by_key["screen-1"]["status"] == "held" and by_key["screen-1"]["held_usd"] == Decimal("4.00")
    assert by_key["screen-2"]["status"] == "settled" and by_key["screen-2"]["settled_usd"] == Decimal("1.25")
    assert by_key["screen-3"]["status"] == "held"  # committed without a verified settlement keeps its cap
    assert by_key["screen-0"]["status"] == "released"
    assert {row["job_id"] for row in rows} == {"req-1", "req-2"}
    view = meter.meter_view(rows, now=NOW)
    jobs = {row["job_id"]: row for row in view["jobs"]}
    assert jobs["req-1"] == {**jobs["req-1"], "settled_usd": 1.25, "held_usd": 8.0, "metered_usd": 9.25}
    # Settled in October: not this month's spend, and nothing held.
    assert "req-2" not in jobs


def test_a_screening_ledger_with_a_bad_digest_is_a_gap_not_a_number(tmp_path):
    queue = tmp_path / "queue"
    write_screening_ledger(queue)
    ledger = json.loads((queue / LEDGER_FILENAME).read_text())
    ledger["ledger_digest"] = "sha256:" + "0" * 64
    (queue / LEDGER_FILENAME).write_text(json.dumps(ledger))
    assert sources.screening_rows(queue) == ([], ["paid_spend_meter_screening_ledger_invalid"])


def test_per_job_metering_sums_match_the_totals(tmp_path):
    root = tmp_path / "meter"
    write_screening_ledger(tmp_path / "queue")
    meter.record_admission(job_id="req-1", meter_key="k1", resource_class="openai_api_candidate",
                           held_usd="2.00", root=root, now=NOW - 400)
    meter.record_admission(job_id="req-3", meter_key="k2", resource_class="vast_provider_adapter",
                           held_usd="5.00", root=root, now=NOW - 300)
    meter.record_settlement(job_id="req-3", meter_key="k2", settled_usd="3.333333", basis="fixture",
                            provider="vast", root=root, now=NOW - 200)
    meter.record_admission(job_id="req-4", meter_key="old", resource_class="vast_provider_adapter",
                           held_usd="1.00", root=root, now=LAST_MONTH)
    world_labs = {"complete": True, "charges": [
        {"at_epoch": NOW - 1000, "usd": 5.5, "held": False, "job_id": "site-req-1", "attempt": "atlas:a"},
        {"at_epoch": NOW - 900, "usd": 1.7, "held": True, "job_id": "site-req-3", "attempt": "atlas:b"},
    ]}
    env = {meter.METER_ROOT_ENV: str(root), meter.SCREENING_QUEUE_ROOT_ENV: str(tmp_path / "queue")}
    view = sources.collect_meter_view(now=NOW, world_labs=world_labs, environ=env)
    assert view["gaps"] == [] and view["complete"] is True
    totals = view["totals"]
    for key in ("settled_usd", "held_usd", "metered_usd"):
        assert round(sum(job[key] for job in view["jobs"]), 6) == totals[key]
    for key in ("admissions", "unpriced_admissions"):
        assert sum(job[key] for job in view["jobs"]) == totals[key]
    for source, amounts in totals["by_source"].items():
        for key in ("settled_usd", "held_usd"):
            assert round(sum(job["by_source"].get(source, {}).get(key, 0) for job in view["jobs"]), 6) == amounts[key]
    assert totals["settled_usd"] == round(1.25 + 3.333333 + 5.5, 6)
    # Held now, whatever month: screening 8.00, journal 2.00 + last month's open 1.00, Atlas 1.70.
    assert totals["held_usd"] == round(8.0 + 2.0 + 1.0 + 1.7, 6)
    assert view["_exact"]["held_outside_provider_readings_usd"] == Decimal("11.00")
    assert totals["by_provider_settled_usd"] == {"vast": round(1.25 + 3.333333, 6), "world_labs": 5.5}
    jobs = {row["job_id"]: row for row in view["jobs"]}
    assert jobs["req-1"]["by_source"]["atlas"]["settled_usd"] == 5.5


def test_job_key_reads_the_website_request_id():
    assert meter.job_key("site-abc123") == "abc123"
    assert meter.job_key("abc123") == "abc123"
    assert meter.job_key("") == meter.UNATTRIBUTED
    assert meter.job_key("../etc") == meter.UNATTRIBUTED


def test_unobserved_atlas_attempts_are_a_gap(tmp_path):
    view = sources.collect_meter_view(now=NOW, world_labs=None, environ={meter.METER_ROOT_ENV: str(tmp_path / "m")})
    assert view["gaps"] == ["paid_spend_meter_atlas_attempts_unobserved"] and view["complete"] is False
