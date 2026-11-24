"""The account's monthly limit and the spend journal it reads (the $0 stress run of round 2, 2026-11-24).

A grant that names its cost is a hold until it settles or a bill can show it; a grant that cannot be journaled is not
admitted; the report names how far it read the journal, so no clock hides a grant and none counts twice; a busy
month's journal rolls to a new file before its readers' cap; last month's report on the 1st reads as stale.

No provider, no network, nothing paid for: the policy, the reports and the journal are temporary files the real
readers and writers load, and the report is rebuilt by the real ``account_monthly_spend_report.refresh_report`` (no
billing export, no provider observation).
"""
# Covers: src/blueprint_pipeline/paid_spend_meter.py
# Covers: src/blueprint_pipeline/paid_spend_meter_sources.py
# Covers: src/blueprint_pipeline/account_monthly_spend.py
# Covers: src/blueprint_pipeline/account_monthly_spend_report.py
# Covers: src/blueprint_pipeline/paid_resource_admission.py
from __future__ import annotations

import errno
import json
import multiprocessing
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from blueprint_pipeline import account_monthly_spend as account
from blueprint_pipeline import account_monthly_spend_report as report_builder
from blueprint_pipeline import paid_spend_meter as meter
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    SPEND_EFFECT_STOP,
    PaidResourceAdmissionBlocked,
    build_paid_lane_admission,
    require_paid_resource_admission,
    require_paid_resource_budget,
)

NOV_29 = datetime(2026, 11, 29, 12, 0, tzinfo=timezone.utc).timestamp()


def _config(tmp_path: Path) -> dict[str, str]:
    report = tmp_path / "report.json"
    config = tmp_path / "account_monthly_spend_limit.v1.json"
    shipped = json.loads(account.SHIPPED_CONFIG_PATH.read_text())
    config.write_text(json.dumps({**shipped, "report_path": str(report)}))
    return {account.CONFIG_ENV: str(config), meter.METER_ROOT_ENV: str(tmp_path / "meter")}


def _write_report(env: dict[str, str], *, counted: str, generated: float, period_of: float | None = None) -> None:
    """A report as one written by hand: it names nothing of what it read of the journal."""
    policy = account.load_policy(Path(env[account.CONFIG_ENV]))
    period = account.month_of(generated if period_of is None else period_of)
    path = Path(policy["report_path"])
    path.unlink(missing_ok=True)
    path.write_text(json.dumps({
        "schema_version": account.REPORT_SCHEMA, "cap_status": "set", "policy_sha256": policy["policy_sha256"],
        "counted_usd": counted, "limit_usd": str(account.effective_limit(policy, period)), "period": period,
        "generated_at": datetime.fromtimestamp(generated, timezone.utc).isoformat().replace("+00:00", "Z")}))
    os.chmod(path, 0o644)


@pytest.fixture
def env(tmp_path, monkeypatch):
    for name in ("BLUEPRINT_GCP_MONTHLY_BUDGET_REPORT", "BLUEPRINT_GCP_BUDGET_GUARD_CONFIG", meter.JOB_ID_ENV,
                 account.ROOT_ENV, "BLUEPRINT_GPU_BILLING_EXPORT", "BLUEPRINT_PROVIDER_BILLING_AUDIT_ROOT"):
        monkeypatch.delenv(name, raising=False)
    values = _config(tmp_path)
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(meter.SCREENING_QUEUE_ROOT_ENV, str(tmp_path / "no-screening"))
    return values


def _admit(cost, *, binding: str, job: str = "job-1", resource_class: str = "openai_api_candidate"):
    """A website-preparation-shaped grant through the real chokepoint: it names its cost, no held price."""
    admission = {**build_paid_lane_admission(resource_class=resource_class), "allocation_binding_digest": binding,
                 "maximum_cost_usd": cost, "meter": {"job_id": job, "provider": "openai"}}
    return require_paid_resource_admission(admission, resource_class=resource_class,
                                           expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                                           maximum_cost_usd=cost)


def _settled_month(env, usd: str, at: float) -> None:
    """``usd`` already settled this month on the meter (the only figure the report counts with no billing read)."""
    root = Path(env[meter.METER_ROOT_ENV])
    meter.record_admission(job_id="earlier", meter_key="earlier-call", resource_class="openai_api_candidate",
                           provider="openai", root=root, now=at)
    meter.record_settlement(job_id="earlier", meter_key="earlier-call", settled_usd=usd, basis="test",
                            provider="openai", root=root, now=at)


def _binding(index: int) -> str:
    return f"sha256:{index:064d}"


# --- a grant that names its cost counts until it settles ---------------------------------------------------------

def test_a_grant_admitted_before_one_report_still_counts_after_the_next(env, monkeypatch):
    """A website preparation grant names its cost but no held price, so its journal row was 'unpriced': it counted
    only until the next report (written every minute), and that report counted it as nothing. At $380 of $400, a
    $12 call a minute, each after a fresh report, was judged at $380 every time. Now it is a hold."""
    _settled_month(env, "380", NOV_29 - 3600)
    admitted = 0
    for minute in range(5):
        moment = NOV_29 + 60 * minute
        monkeypatch.setattr(account, "_clock", lambda moment=moment: moment)
        written = report_builder.refresh_report(now=moment - 1)
        assert Decimal(str(written["counted_usd"])) == Decimal("380") + 12 * admitted
        try:
            _admit(12, binding=_binding(minute))
            admitted += 1
        except PaidResourceAdmissionBlocked as refused:
            assert refused.blockers == [account.LIMIT_REACHED]
    assert admitted == 1


def test_a_settled_grant_counts_at_its_settlement_and_frees_the_rest(env, monkeypatch):
    """Control: once the grant settles at $3, the next report counts $383, and another $12 call fits."""
    _settled_month(env, "380", NOV_29 - 3600)
    monkeypatch.setattr(account, "_clock", lambda: NOV_29)
    report_builder.refresh_report(now=NOV_29 - 1)
    _admit(12, binding=_binding(1))
    meter.record_settlement(job_id="job-1", meter_key=_binding(1), settled_usd="3", basis="test", provider="openai",
                            root=Path(env[meter.METER_ROOT_ENV]), now=NOV_29 + 30)
    monkeypatch.setattr(account, "_clock", lambda: NOV_29 + 60)
    assert Decimal(str(report_builder.refresh_report(now=NOV_29 + 59)["counted_usd"])) == Decimal("383")
    _admit(12, binding=_binding(2))  # $383 + $12 = $395


def test_a_reservation_judged_again_is_not_counted_beside_its_own_hold(env, monkeypatch):
    """The report holds the grant at $12; judging the same reservation again (a retained hold rechecked) adds its
    $12 once, not twice: $388 + $12 fits $400 before and after the report is rewritten."""
    _settled_month(env, "388", NOV_29 - 3600)
    monkeypatch.setattr(account, "_clock", lambda: NOV_29)
    report_builder.refresh_report(now=NOV_29 - 1)
    _admit(12, binding=_binding(7))
    report = report_builder.refresh_report(now=NOV_29 + 1)
    assert report["counted_usd"] == 400 and report["meter"]["journal"]["holds_usd"] == {_binding(7): "12"}
    require_paid_resource_budget(resource_class="openai_api_candidate", maximum_cost_usd=12,
                                 allocation_binding_digest=_binding(7))
    with pytest.raises(PaidResourceAdmissionBlocked, match=account.LIMIT_REACHED):
        require_paid_resource_budget(resource_class="openai_api_candidate", maximum_cost_usd=1,
                                     allocation_binding_digest=_binding(8))


def test_a_grant_whose_bill_can_show_it_is_left_to_the_bill(env, monkeypatch):
    """A grant held at its cost is left to its provider's bill once a reading taken a day after it can show it, so
    a call that never settles does not count for the rest of the month on top of its bill."""
    _settled_month(env, "100", NOV_29 - 3600)
    monkeypatch.setattr(account, "_clock", lambda: NOV_29)
    report_builder.refresh_report(now=NOV_29 - 1)
    _admit(12, binding=_binding(3))
    readings = {"openai": {"status": "observed", "month_usd": Decimal("130"),
                           "observed_at": datetime.fromtimestamp(NOV_29 + 3600, timezone.utc).isoformat()}}
    assert report_builder.bill_readings(readings) == {"openai": NOV_29 + 3600}
    from blueprint_pipeline.paid_spend_meter_sources import collect_meter_view
    soon = collect_meter_view(now=NOV_29 + 3600, bills=report_builder.bill_readings(readings))
    assert soon["totals"]["held_usd"] == 12
    later = collect_meter_view(now=NOV_29 + 86400, bills={"openai": NOV_29 + 86400})
    assert later["totals"]["held_usd"] == 0


# --- a grant the journal cannot record is not admitted -----------------------------------------------------------

def _full_disk(monkeypatch):
    real_write = os.write

    def full(descriptor, data):
        if b'"event":"admission"' in data:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_write(descriptor, data)
    monkeypatch.setattr(meter.os, "write", full)


def test_a_grant_the_journal_cannot_record_is_not_admitted(env, monkeypatch):
    """Disk full: the journal write failed and ``meter_chokepoint_admission`` swallowed it ("metering is
    visibility"), but the account's limit reads that journal, so every grant was judged at the report's stale count.
    Now the grant is refused by name; a clear stop still goes through."""
    _settled_month(env, "380", NOV_29 - 3600)
    monkeypatch.setattr(account, "_clock", lambda: NOV_29)
    _write_report(env, counted="380.00", generated=NOV_29 - 30)
    _full_disk(monkeypatch)
    for index in range(4):
        with pytest.raises(PaidResourceAdmissionBlocked) as refused:
            _admit(12, binding=_binding(index))
        assert refused.value.blockers == [meter.JOURNAL_UNWRITABLE]
    stop = {**build_paid_lane_admission(resource_class="vast_provider_adapter"), "spend_effect": SPEND_EFFECT_STOP}
    require_paid_resource_admission(stop, resource_class="vast_provider_adapter",
                                    expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                                    spend_effect=SPEND_EFFECT_STOP)


def test_a_process_with_no_meter_root_journals_nothing_and_is_not_refused(tmp_path, monkeypatch):
    """Control: off the host there is no meter root and nothing to count; the grant is admitted as before."""
    monkeypatch.delenv(meter.METER_ROOT_ENV, raising=False)
    monkeypatch.setattr(meter, "DEFAULT_METER_ROOT", tmp_path / "absent" / "paid-spend-meter")
    for name in (account.CONFIG_ENV, "BLUEPRINT_GCP_MONTHLY_BUDGET_REPORT", "BLUEPRINT_GCP_BUDGET_GUARD_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(account, "INSTALLED_CONFIG_PATH", tmp_path / "no-limit.json")
    _full_disk(monkeypatch)
    _admit(12, binding=_binding(1))


# --- no clock hides a grant -------------------------------------------------------------------------------------

def test_a_report_stamped_ahead_of_the_journal_clock_hides_the_grants_in_between(env, monkeypatch):
    """A report up to 60 s in the future is accepted, and ``admitted_since`` counted only grants journaled at or after
    its stamp: with the report's clock 50 s ahead, two $12 calls 10 s apart at $380 of $400 both passed."""
    _write_report(env, counted="380.00", generated=NOV_29 + 50)
    admitted = 0
    for second in (0, 10):
        monkeypatch.setattr(account, "_clock", lambda second=second: NOV_29 + second)
        try:
            _admit(12, binding=_binding(second))
            admitted += 1
        except PaidResourceAdmissionBlocked:
            pass
    assert admitted == 1


def test_a_report_that_read_the_journal_counts_by_what_it_read_whatever_its_clock(env, monkeypatch):
    """The report names how far it read each journal file: a grant after that counts, one before it is in its count,
    and neither clock decides. Stamped 50 s ahead of the journal's clock, it still sees both grants once."""
    _settled_month(env, "370", NOV_29 - 3600)
    monkeypatch.setattr(account, "_clock", lambda: NOV_29)
    _write_report(env, counted="370.00", generated=NOV_29 - 30)
    _admit(12, binding=_binding(1))
    report = report_builder.refresh_report(now=NOV_29 + 50)  # the report's clock, 50 s ahead
    assert report["counted_usd"] == 382 and report["meter"]["journal"]["read_bytes"]
    monkeypatch.setattr(account, "_clock", lambda: NOV_29 + 10)
    _admit(12, binding=_binding(2))  # $382 + $12 = $394: after the report's read, counted next
    monkeypatch.setattr(account, "_clock", lambda: NOV_29 + 20)
    with pytest.raises(PaidResourceAdmissionBlocked, match=account.LIMIT_REACHED):
        _admit(12, binding=_binding(3))  # $394 + $12 > $400
    _admit(6, binding=_binding(4))  # $394 + $6 = $400: fits, the first grant not counted twice


# --- a busy month rolls to a new file ---------------------------------------------------------------------------

def test_a_month_journal_past_its_size_cap_stops_admissions_and_the_writer_rolls_before_it(env, monkeypatch):
    """Over ``MAX_JOURNAL_FILE_BYTES`` a month's file is unreadable and every new paid admission is refused
    (``journal_unreadable``), and the writer kept appending to the same file, so nothing cleared it until the month
    ended. Now the writer starts the next file before a file would pass ``ROLL_JOURNAL_FILE_BYTES``, readers read
    every file of the month, and a file past ``NEAR_CAP_JOURNAL_FILE_BYTES`` is flagged and paged before the cap."""
    monkeypatch.setattr(account, "_clock", lambda: NOV_29)
    root = Path(env[meter.METER_ROOT_ENV])
    row = meter.record_settlement(job_id="itest-0123abcd", meter_key=meter.ARTISAN_OPENAI_METER_PREFIX + "c" * 32,
                                  settled_usd="0.0421", basis="artisan_openai_call", provider="openai", root=root,
                                  now=NOV_29 - 60)
    journal = next((root / "events" / "2026-11").glob("uid-*.jsonl"))
    line = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with journal.open("ab") as stream:  # an old writer that never rolled: past the cap
        stream.write(line * (meter.MAX_JOURNAL_FILE_BYTES // len(line) + 1))
    _write_report(env, counted="10.00", generated=NOV_29 - 30)
    with pytest.raises(PaidResourceAdmissionBlocked) as refused:
        _admit(1, binding="sha256:" + "a" * 64)
    assert account.JOURNAL_UNREADABLE in refused.value.blockers
    meter.record_settlement(job_id="x", meter_key="y", settled_usd="0", basis="t", provider="openai", root=root,
                            now=NOV_29)
    files = {path.name: path for path in (root / "events" / "2026-11").glob("uid-*.jsonl")}
    rolled = files.pop(f"uid-{os.geteuid()}.1.jsonl")
    assert list(files) == [journal.name] and rolled.stat().st_size < meter.ROLL_JOURNAL_FILE_BYTES
    # The report flags the unreadable file, and the alert pages.
    view = report_builder.refresh_report(now=NOV_29)
    assert "paid_spend_meter_journal_file_unreadable" in view["gaps"]
    alerts = account.account_alerts({**account.observe_account_spend(now=NOV_29), "blockers": []})
    assert account.JOURNAL_UNREADABLE_CODE in {row["code"] for row in alerts}
    assert account.JOURNAL_UNREADABLE_CODE in account.ACCOUNT_PAGE_CODES


def test_a_writer_rolls_before_the_cap_and_every_file_of_the_month_counts(env, monkeypatch):
    """With a small roll size, grants land in several files; the admissions after a report count every file, and a
    file past the near-cap mark is flagged (``NEAR_CAP_GAP``, a warning) while it still reads."""
    monkeypatch.setattr(meter, "ROLL_JOURNAL_FILE_BYTES", 900)
    monkeypatch.setattr(meter, "NEAR_CAP_JOURNAL_FILE_BYTES", 600)
    monkeypatch.setattr(account, "_clock", lambda: NOV_29)
    _write_report(env, counted="300.00", generated=NOV_29 - 30)
    for index in range(6):
        _admit(10, binding=_binding(index))
    files = sorted((Path(env[meter.METER_ROOT_ENV]) / "events" / "2026-11").glob("uid-*.jsonl"))
    assert len(files) >= 3 and all(path.stat().st_size <= 900 for path in files)
    assert account.admitted_since_report({"generated_at": "2026-11-29T11:59:30Z"}, now=NOV_29) == 60
    with pytest.raises(PaidResourceAdmissionBlocked, match=account.LIMIT_REACHED):
        _admit(41, binding=_binding(9))  # $300 + $60 + $41
    events, gaps = meter.read_journal(Path(env[meter.METER_ROOT_ENV]))
    assert len([row for row in events if row["event"] == "admission"]) == 6 and meter.NEAR_CAP_GAP in gaps
    alerts = account.account_alerts({"cap_status": "set", "period": "2026-11", "blockers": [], "complete": False,
                                     "gaps": [meter.NEAR_CAP_GAP], "counted_usd": 360, "limit_usd": "400"})
    assert account.JOURNAL_NEAR_CAP_CODE in {row["code"] for row in alerts}
    assert account.JOURNAL_NEAR_CAP_CODE in account.ACCOUNT_WARN_CODES


# --- the month's turn -------------------------------------------------------------------------------------------

def test_month_rollover_refuses_as_stale_until_the_new_months_report_and_counts_only_its_own_grants(env, monkeypatch):
    """At 00:00:05 UTC on 1 December the November report is stale (a new month): it read as missing or invalid,
    because November's limit is not December's. A December report then judges December grants against $300;
    November's last grants stay in November."""
    nov_30 = datetime(2026, 11, 30, 23, 59, 50, tzinfo=timezone.utc).timestamp()
    dec_1 = datetime(2026, 12, 1, 0, 0, 5, tzinfo=timezone.utc).timestamp()
    _write_report(env, counted="100.00", generated=nov_30 - 5)
    monkeypatch.setattr(account, "_clock", lambda: nov_30)
    _admit(12, binding="sha256:" + "1" * 64)
    monkeypatch.setattr(account, "_clock", lambda: dec_1)
    with pytest.raises(PaidResourceAdmissionBlocked) as stale:
        _admit(12, binding="sha256:" + "2" * 64)
    assert stale.value.blockers == [account.REPORT_STALE]
    _write_report(env, counted="295.00", generated=dec_1 - 1)
    with pytest.raises(PaidResourceAdmissionBlocked) as full:
        _admit(6, binding="sha256:" + "3" * 64)  # 295 + 6 > December's 300
    assert full.value.blockers == [account.LIMIT_REACHED]
    _admit(5, binding="sha256:" + "4" * 64)  # exactly 300: fits
    assert account.admitted_since_report({"generated_at": "2026-12-01T00:00:04Z"}, now=dec_1, environ=env) == 5


def _contend(env: dict, start: float, index: int, out) -> None:
    os.environ.update(env)
    account._clock = lambda: start
    try:
        _admit(12, binding=_binding(index), job=f"job-{index}")
        out.put("admitted")
    except PaidResourceAdmissionBlocked as exc:
        out.put(",".join(exc.blockers))
    except Exception as exc:  # noqa: BLE001
        out.put(f"error:{type(exc).__name__}:{exc}")


def test_concurrent_admissions_are_judged_one_at_a_time_and_none_deadlocks(env):
    """Control: eight processes at once, each a $12 call at $380 of $400, while a ninth journals settlements
    (journal lock only). Exactly one fits; every process finishes."""
    _write_report(env, counted="380.00", generated=NOV_29 - 30)
    context = multiprocessing.get_context("fork")
    out = context.Queue()
    workers = [context.Process(target=_contend, args=(env, NOV_29, index, out)) for index in range(8)]

    def settle(env=env):
        root = Path(env[meter.METER_ROOT_ENV])
        for index in range(200):
            meter.record_settlement(job_id="s", meter_key=f"k{index}", settled_usd="0.01", basis="t",
                                    provider="openai", root=root, now=NOV_29)
    settler = context.Process(target=settle)
    for process in (*workers, settler):
        process.start()
    for process in (*workers, settler):
        process.join(timeout=60)
        assert not process.is_alive(), "a process is still waiting: deadlock or lock starvation"
    results = [out.get(timeout=5) for _ in workers]
    assert results.count("admitted") == 1, results
    assert all(value == "admitted" or value == account.LIMIT_REACHED for value in results), results
