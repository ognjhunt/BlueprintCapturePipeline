"""Owner decision 2026-11-10: a settled GPU run keeps its final Vast bill, not its whole allowance.

A charge is final when the teardown proves every instance destroyed, Vast's
charges were collected at least 48 h after the teardown with no page
missing, and the instance's charge rows cover every day of the run. Anything
short of that keeps the hold as it was. Every figure and identifier here is
synthetic; the row and gap shapes are the ones the host's snapshot carries.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from blueprint_pipeline import scene_attempt_measured_compute as measured
from blueprint_pipeline import task_evaluation_terminal_scene_attempt_settlement as settlement
from blueprint_pipeline.daily_spend_snapshot import _identity
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_artisan_proven_authoring_spend import (  # noqa: F401 - ``site`` is a fixture
    AGREEING, _artisan_runtime, _hold, _pre_gate_refusal, _request, _terminal_attempt, site,
)

STARTED = datetime(2026, 11, 1, 22, 0, tzinfo=timezone.utc)
DESTROYED = STARTED + timedelta(hours=3)  # crosses into 11-02


def _iso(moment):
    return moment.isoformat()


def _closed_run(run):
    """The run's own records: dispatch start, and a teardown that destroyed instance 7."""
    started = {"schema_version": "task_evaluation_launch_started.v1", "launch_id": run.name,
               "started_at": _iso(STARTED)}
    started["started_digest"] = canonical_digest(started, digest_field="started_digest")
    (run / "launch_started.json").write_text(json.dumps(started))
    (run / "teardown.json").write_text(json.dumps({
        "schema_version": "vast_teardown_manifest.v1", "status": "completed", "vast_instance_ids": [7],
        "continuing_spend_from_this_run": False, "runner_gpu_teardown_completed": True,
        "generated_at": _iso(DESTROYED)}))


def _row(day, amount, *, currency="USD", resource="instance-7"):
    start = datetime(2026, 11, day, tzinfo=timezone.utc)
    return {"provider": "vast", "service": "charges", "kind": "actual", "resource": _identity(resource),
            "amount": amount, "currency": currency, "interval_start": _iso(start),
            "interval_end": _iso(start + timedelta(days=1)), "partial": True}


# The gaps every snapshot on the host names today; none says a Vast charge page is missing.
LIVE_GAPS = ("all_provider_all_day_coverage_unknown", "vast_invalid_amount_currency_or_interval",
             "vast_provider_freshness_unknown", "vast_stable_charge_id_missing_corrections_unknown")


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    billing = tmp_path / "billing"
    billing.mkdir()
    monkeypatch.setenv("BLUEPRINT_GPU_BILLING_EXPORT", str(billing / "provider_billing_export.json"))

    def write(rows, *, after_destroy=timedelta(hours=48), collected_after_destroy=None, gaps=LIVE_GAPS,
              drop=()):
        measured._CACHE.clear()
        collected = after_destroy if collected_after_destroy is None else collected_after_destroy
        value = {"generated_at": _iso(DESTROYED + after_destroy),
                 "source_collected_at": _iso(DESTROYED + collected),
                 "coverage": {"partial": True, "gaps": list(gaps)},
                 "snapshot_digest": "sha256:" + "5" * 64, "rows": rows}
        for key in drop:
            del value[key]
        (billing / measured.SNAPSHOT_FILENAME).write_text(json.dumps(value))
    return write


FINAL = [_row(1, 0.25), _row(2, 0.16), _row(2, 3.0, resource="instance-8")]


def test_final_bills_replace_only_the_gpu_allowance(site, snapshot):  # noqa: F811 - the imported fixture
    rows = {
        "pre_gate": _terminal_attempt(site, "pregate", _request("pregate", authoring=15.0, total=20.0),
                                      _pre_gate_refusal, after=_closed_run),
        "parent": _terminal_attempt(site, "parent", _request("parent", authoring=15.0, total=20.0),
            lambda output, request: _artisan_runtime(output, request, runs=AGREEING), after=_closed_run),
        "no_completion": _terminal_attempt(site, "died", _request("died", authoring=15.0, total=20.0),
            lambda output, request: _artisan_runtime(output, request, baseline=18.41,
                                                     runs=[([3.0, 4.0], None, True)]), after=_closed_run),
    }
    before = {name: _hold(site, path)["retained_spend_usd"] for name, path in rows.items()}
    assert before == {"pre_gate": 5.0, "parent": 18.76, "no_completion": 20.0}
    snapshot(FINAL)
    holds = {name: _hold(site, path) for name, path in rows.items()}
    assert {name: (hold["basis"], hold["retained_spend_usd"]) for name, hold in holds.items()} == {
        "pre_gate": (settlement.MEASURED_UNENTERED_BASIS, 0.41),               # $0.41 billed, no model spend
        "parent": (settlement.MEASURED_PROVEN_AUTHORING_BASIS, 14.17),         # $13.76 metered + $0.41 billed
        "no_completion": (settlement.MEASURED_COMPUTE_BASIS, 15.41),           # $15 model allowance kept
    }
    evidence = holds["parent"]["evidence"]
    assert evidence["billed_usd"] == 0.41 and evidence["vast_instance_ids"] == [7]
    assert evidence["final_after_destroy_seconds"] == 48 * 3600 and evidence["compute_allowance_kept"] is False
    assert evidence["snapshot_source_collected_at"] == _iso(DESTROYED + timedelta(hours=48))


def test_a_bill_above_the_allowance_is_kept_up_to_the_reservation(site, snapshot):  # noqa: F811 - the imported fixture
    row = _terminal_attempt(site, "pregate", _request("pregate", authoring=15.0, total=20.0), _pre_gate_refusal,
                            after=_closed_run)
    snapshot([_row(1, 7.5), _row(2, 0.5)])
    assert _hold(site, row)["retained_spend_usd"] == 8.0
    snapshot([_row(1, 30.0), _row(2, 0.5)])
    assert _hold(site, row)["retained_spend_usd"] == 20.0


@pytest.mark.parametrize("case", ["47h", "missing_day", "no_currency", "no_rows", "no_snapshot", "teardown_open",
                                  "no_start_record", "early_collection_projected_again", "vast_page_missing",
                                  "vast_source_missing", "no_collection_time", "no_coverage"])
def test_a_charge_that_is_not_final_keeps_the_hold(site, snapshot, monkeypatch, case):  # noqa: F811 - the imported fixture
    def run_records(run):
        _closed_run(run)
        if case == "teardown_open":
            value = json.loads((run / "teardown.json").read_text())
            value["continuing_spend_from_this_run"] = True
            (run / "teardown.json").write_text(json.dumps(value))
        if case == "no_start_record":
            (run / "launch_started.json").unlink()
    row = _terminal_attempt(site, "pregate", _request("pregate", authoring=15.0, total=20.0), _pre_gate_refusal,
                            after=run_records)
    rows = {"missing_day": [_row(2, 0.16)], "no_currency": [_row(1, 0.25), _row(2, 0.16, currency=None)],
            "no_rows": []}.get(case, FINAL)
    snapshot(rows, **{
        "47h": {"after_destroy": timedelta(hours=47)},
        # Vast was asked one hour after the destroy; a failed later read projected that answer again at 72 h.
        "early_collection_projected_again": {"after_destroy": timedelta(hours=72),
                                             "collected_after_destroy": timedelta(hours=1)},
        "vast_page_missing": {"gaps": LIVE_GAPS + ("vast_pagination_partial_or_unknown",)},
        "vast_source_missing": {"gaps": LIVE_GAPS + ("vast_source_missing",)},
        "no_collection_time": {"drop": ("source_collected_at",)},
        "no_coverage": {"drop": ("coverage",)},
    }.get(case, {}))
    if case == "no_snapshot":
        monkeypatch.delenv("BLUEPRINT_GPU_BILLING_EXPORT")
        monkeypatch.setattr(measured, "HOST_BILLING_EXPORT", site["tmp"] / "absent" / "export.json")
    hold = _hold(site, row)
    expected = (("terminal_launch_unreconciled", 20.0) if case == "teardown_open"  # an open teardown keeps it all
                else ("native_allowance_with_unentered_authoring", 5.0))
    assert (hold["basis"], hold["retained_spend_usd"]) == expected
