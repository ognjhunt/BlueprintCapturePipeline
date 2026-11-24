"""The account's monthly limit counts the call it admits and every call admitted since its report.

From the $0 stress test of 2026-11-24 (``tests/stress/test_stress_money.py`` on the stress branch, copied here): at
$399.99 of November's $400 a $6, $12 or $40 call was admitted, because the check took no amount, so the month could
end at $439.99; and every admission made after one spend report was judged at that report's count for as long as
the report was accepted (15 minutes). Now a call that names its maximum cost must fit under the limit with it, as
the job's $40 cap admits a call, and each grant journaled since the report counts. A caller that knows no amount
keeps the rule it had, and a stop is never refused for a budget reason. No provider, no network, nothing paid for:
the policy, the report and the journal are temporary files the real readers load.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from blueprint_pipeline import account_monthly_spend as account
from blueprint_pipeline import paid_spend_meter as meter
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    build_paid_lane_admission,
    require_paid_resource_admission,
    require_paid_resource_budget,
)

NOVEMBER_29 = datetime(2026, 11, 29, 12, 0, tzinfo=timezone.utc).timestamp()
REACHED = account.LIMIT_REACHED


def _limit(tmp_path: Path, *, counted: str, generated_ago_seconds: float, now: float = NOVEMBER_29) -> dict:
    """The shipped limit with its report under ``tmp_path``: a report counting ``counted``, and an empty journal."""
    report = tmp_path / "report.json"
    config = tmp_path / "account_monthly_spend_limit.v1.json"
    shipped = json.loads(account.SHIPPED_CONFIG_PATH.read_text())
    config.write_text(json.dumps({**shipped, "report_path": str(report)}))
    policy = account.load_policy(config)
    period = account.month_of(now)
    generated = datetime.fromtimestamp(now - generated_ago_seconds, timezone.utc).isoformat().replace("+00:00", "Z")
    report.write_text(json.dumps({"schema_version": account.REPORT_SCHEMA, "cap_status": "set",
                                  "policy_sha256": policy["policy_sha256"], "counted_usd": counted,
                                  "limit_usd": str(account.effective_limit(policy, period)), "period": period,
                                  "generated_at": generated}))
    os.chmod(report, 0o644)
    return {account.CONFIG_ENV: str(config), meter.METER_ROOT_ENV: str(tmp_path / "meter")}


@pytest.fixture
def month(tmp_path, monkeypatch):
    """The chokepoint's own environment (it reads the process's): November's limit, a report, a journal, and the
    account's clock at ``NOVEMBER_29``. Call it with the report's count; it can be called again to rewrite it."""

    monkeypatch.setattr(account, "_clock", lambda: NOVEMBER_29)
    for name in ("BLUEPRINT_GCP_MONTHLY_BUDGET_REPORT", "BLUEPRINT_GCP_BUDGET_GUARD_CONFIG", meter.JOB_ID_ENV,
                 account.ROOT_ENV):
        monkeypatch.delenv(name, raising=False)

    def at(counted: str, *, generated_ago_seconds: float = 60) -> dict:
        env = _limit(tmp_path, counted=counted, generated_ago_seconds=generated_ago_seconds)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return env

    return at


def _admit(cost=None, *, resource_class: str = "evaluator_api", **extra):
    return require_paid_resource_admission({**build_paid_lane_admission(resource_class=resource_class), **extra},
                                           resource_class=resource_class,
                                           expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                                           maximum_cost_usd=cost)


def _stop(cost=None, **extra):
    admission = {**build_paid_lane_admission(resource_class="gpu_render"), "spend_effect": "stop", **extra}
    return require_paid_resource_admission(admission, resource_class="gpu_render",
                                           expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                                           spend_effect="stop", maximum_cost_usd=cost)


def _journal(tmp_path: Path) -> list[dict]:
    events, gaps = meter.read_journal(tmp_path / "meter")
    assert gaps == []
    return events


# ----------------------------- the stress reproductions -----------------------------


def test_the_shipped_november_limit_is_400_and_december_300():
    policy = account.load_policy(account.SHIPPED_CONFIG_PATH)
    assert account.effective_limit(policy, "2026-11") == Decimal("400")
    assert account.effective_limit(policy, "2026-12") == Decimal("300")


def test_at_the_limit_a_new_admission_is_refused(tmp_path):
    env = _limit(tmp_path, counted="400.00", generated_ago_seconds=60)
    assert account.admission_blockers(now=NOVEMBER_29, environ=env) == [REACHED]


@pytest.mark.parametrize("quote", ["6.00", "12.00", "40.00"])
def test_a_call_whose_quote_does_not_fit_under_the_monthly_limit_is_refused(tmp_path, quote):
    """AGENTS.md: the job cap admits a call only when spent + held + this call's quote fit; the account's limit now
    asks the same. One cent under the month's $400, a $40 job's next call is refused, so the month cannot end past
    its limit. (The stress test could ask without an amount only; the check now takes one.)"""
    env = _limit(tmp_path, counted="399.99", generated_ago_seconds=60)
    blockers = account.admission_blockers(now=NOVEMBER_29, environ=env, maximum_cost_usd=quote)
    assert blockers == [REACHED], (
        f"a ${quote} call is admitted at $399.99 of $400: {blockers} (counted + quote = "
        f"${Decimal('399.99') + Decimal(quote)})")


def test_admissions_made_after_the_report_count_against_the_limit(tmp_path):
    """The report is accepted for 15 minutes and the guard rewrites it about once a minute. Admissions made after it
    are in none of its figures, so each grant journaled since now counts: ten $5 calls, one a minute, at $395 of $400
    and judged against one report, admit one. (The stress test journaled nothing; a grant is journaled here as the
    chokepoint journals it.)"""
    env = _limit(tmp_path, counted="395.00", generated_ago_seconds=0)
    admitted = 0
    for minute in range(10):
        moment = NOVEMBER_29 + 60 * minute
        if account.admission_blockers(now=moment, environ=env, maximum_cost_usd="5.00") == []:
            admitted += 1
            meter.meter_chokepoint_admission(build_paid_lane_admission(resource_class="evaluator_api"),
                                             resource_class="evaluator_api",
                                             schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION, environ=env,
                                             now=moment, maximum_cost_usd="5.00")
    assert admitted == 1, f"{admitted} admissions in 10 minutes were each judged at $395 of $400"


def test_a_report_of_last_month_refuses_on_the_first_of_the_month(tmp_path):
    """Control: at 00:05 UTC on 1 December the November report is stale; money fails closed until a new one."""
    env = _limit(tmp_path, counted="10.00", generated_ago_seconds=0, now=NOVEMBER_29)
    december = datetime(2026, 12, 1, 0, 5, tzinfo=timezone.utc).timestamp()
    for cost in (None, 1):
        assert account.admission_blockers(now=december, environ=env, maximum_cost_usd=cost) in (
            [account.REPORT_STALE], [account.REPORT_INVALID])


def test_the_job_cap_file_is_40_and_the_scene_budget_never_passes_it():
    from blueprint_pipeline.task_evaluation_scene_intake import scene_spend_budget
    from blueprint_pipeline.website_job_spend_cap import per_job_cap_usd
    assert per_job_cap_usd() == Decimal("40")
    budget = scene_spend_budget({"request": {"execution": {"max_total_spend_usd": 400, "max_paid_attempts": 3}}})
    assert budget["max_total_spend_usd"] == Decimal("40")


# ----------------------------- the call's own cost -----------------------------


def test_a_call_that_fits_exactly_is_admitted_and_a_cent_more_is_not(tmp_path):
    env = _limit(tmp_path, counted="390.00", generated_ago_seconds=60)
    assert account.admission_blockers(now=NOVEMBER_29, environ=env, maximum_cost_usd="10.00") == []
    assert account.admission_blockers(now=NOVEMBER_29, environ=env, maximum_cost_usd="10.01") == [REACHED]


@pytest.mark.parametrize("quote", [6.0, 12.0, 40.0])
def test_the_precheck_and_the_chokepoint_refuse_a_quote_that_does_not_fit(month, tmp_path, quote):
    month("399.99")
    with pytest.raises(PaidResourceAdmissionBlocked) as precheck:
        require_paid_resource_budget(resource_class="evaluator_api", maximum_cost_usd=quote)
    assert precheck.value.blockers == [REACHED]
    with pytest.raises(PaidResourceAdmissionBlocked) as told:
        _admit(quote)
    assert told.value.blockers == [REACHED]
    # A website preparation receipt carries its own quote: the chokepoint reads it without being told.
    with pytest.raises(PaidResourceAdmissionBlocked) as receipt:
        _admit(maximum_cost_usd=quote)
    assert receipt.value.blockers == [REACHED]
    assert _journal(tmp_path) == []  # nothing refused is journaled


def test_the_larger_of_the_callers_cost_and_the_admissions_own_is_judged(month):
    month("390.00")
    with pytest.raises(PaidResourceAdmissionBlocked, match=REACHED):
        _admit(1.0, maximum_cost_usd=12.0)
    with pytest.raises(PaidResourceAdmissionBlocked, match=REACHED):
        _admit(12.0, maximum_cost_usd=1.0)
    assert _admit(10.0, maximum_cost_usd=1.0).resource_class == "evaluator_api"


@pytest.mark.parametrize("cost", [-1, float("nan"), float("inf"), True, "six dollars"])
def test_a_cost_that_is_no_amount_refuses_new_spend_by_name(month, cost):
    """A figure that cannot be judged never lets the call be judged without it: money fails closed."""
    month("10.00")
    with pytest.raises(PaidResourceAdmissionBlocked) as precheck:
        require_paid_resource_budget(resource_class="evaluator_api", maximum_cost_usd=cost)
    assert precheck.value.blockers == [account.CALL_COST_INVALID]
    with pytest.raises(PaidResourceAdmissionBlocked) as receipt:
        _admit(maximum_cost_usd=cost)
    assert receipt.value.blockers == [account.CALL_COST_INVALID]


# ----------------------------- the calls admitted since the report -----------------------------


def test_a_burst_through_the_chokepoint_never_takes_the_month_past_its_limit(month, tmp_path):
    month("395.00", generated_ago_seconds=0)
    outcomes = []
    for _ in range(10):
        try:
            _admit(2.0)
            outcomes.append("granted")
        except PaidResourceAdmissionBlocked as exc:
            outcomes.append(",".join(exc.blockers))
    assert outcomes == ["granted", "granted"] + [REACHED] * 8  # $395 + $2 + $2 = $399; a third would be $401
    assert [row[meter.MAXIMUM_COST_FIELD] for row in _journal(tmp_path)] == ["2.0", "2.0"]


def _admit_in_a_process(start, results) -> None:
    start.wait(30)
    try:
        _admit(5.0)
        results.put("granted")
    except PaidResourceAdmissionBlocked as exc:
        results.put(",".join(exc.blockers))


def test_processes_admitting_at_once_are_judged_one_after_another(month, tmp_path):
    """Between the account check and the grant's journal line one lock is held (``admission_lock``): six processes
    released together at $395 of $400 with a $5 call each get one grant, never one each at the same count."""
    month("395.00", generated_ago_seconds=0)
    context = multiprocessing.get_context("fork")
    start, results = context.Event(), context.Queue()
    workers = [context.Process(target=_admit_in_a_process, args=(start, results)) for _ in range(6)]
    for worker in workers:
        worker.start()
    start.set()
    outcomes = sorted(results.get(timeout=120) for _ in workers)
    for worker in workers:
        worker.join(30)
    assert outcomes == [REACHED] * 5 + ["granted"]
    assert [row[meter.MAXIMUM_COST_FIELD] for row in _journal(tmp_path)] == ["5.0"]


def test_a_grant_taken_again_for_the_same_reservation_counts_once(month):
    """A retained hold whose dispatch bought nothing pays the next one under the same binding: one reservation, so its
    earlier grant since the report is not counted beside its own cost; a different reservation is."""
    month("370.00", generated_ago_seconds=0)
    first, second, third = ("sha256:" + digit * 64 for digit in "abc")
    for _ in range(3):
        _admit(12.0, allocation_binding_digest=first)  # $370 + $12 each time, never + $24 + $36
    _admit(12.0, allocation_binding_digest=second)  # $370 + $12 + $12 = $394
    # The precheck of the first reservation again: $370 + $12 (second) + $12 = $394, room left.
    require_paid_resource_budget(resource_class="evaluator_api", maximum_cost_usd=12.0,
                                 allocation_binding_digest=first)
    with pytest.raises(PaidResourceAdmissionBlocked, match=REACHED):
        _admit(12.0, allocation_binding_digest=third)  # $370 + $24 + $12 = $406


def _read_through(env: dict, report_counted: str) -> None:
    """Rewrite the report as one that read the journal as it stands now (``meter.journal.read_bytes``)."""
    marks: dict = {}
    meter.read_journal(Path(env[meter.METER_ROOT_ENV]), marks=marks)
    path = Path(account.load_policy(Path(env[account.CONFIG_ENV]))["report_path"])
    report = json.loads(path.read_text())
    report.update(counted_usd=report_counted, meter={"journal": {"period": account.month_of(NOVEMBER_29),
                                                                 "read_bytes": marks, "holds_usd": {}}})
    path.write_text(json.dumps(report))
    os.chmod(path, 0o644)


def test_a_grant_made_before_the_report_is_not_added_on_top_of_it(month, tmp_path):
    """A grant the report read (its count holds it) is not counted again by the admissions after it."""
    env = month("395.00", generated_ago_seconds=0)
    meter.meter_chokepoint_admission(build_paid_lane_admission(resource_class="evaluator_api"),
                                     resource_class="evaluator_api", schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                                     now=NOVEMBER_29 - 30, maximum_cost_usd=5.0)
    _read_through(env, "395.00")
    assert _admit(5.0).resource_class == "evaluator_api"


def test_a_report_that_says_nothing_of_what_it_read_counts_the_grants_within_the_accepted_skew(month, tmp_path):
    """Stress round 2: a report written up to 60 s ahead of the journal's clock is accepted; one that names no read of
    the journal counts every grant from its stamp less those 60 s, so none between is hidden (and one may count
    twice, never less)."""
    month("390.00", generated_ago_seconds=-50)  # stamped 50 s ahead
    meter.meter_chokepoint_admission(build_paid_lane_admission(resource_class="evaluator_api"),
                                     resource_class="evaluator_api", schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                                     now=NOVEMBER_29 - 5, maximum_cost_usd=5.0)
    with pytest.raises(PaidResourceAdmissionBlocked, match=REACHED):
        _admit(6.0)  # $390 + $5 + $6
    assert _admit(5.0).resource_class == "evaluator_api"  # $390 + $5 + $5 = $400: fits


def test_a_journaled_quote_is_a_hold_until_it_settles_or_its_bill_can_show_it(tmp_path):
    """Stress round 2: a grant's quote only counted until the next report, so at $380 of $400 one $12 call a report
    was judged at $380 each time. Now it is a hold until its settlement is journaled or a bill of its provider read a
    day after it can show it, and past its month it is the call it was."""
    root = tmp_path / "meter"
    meter.meter_chokepoint_admission(build_paid_lane_admission(resource_class="openai_api_candidate"),
                                     resource_class="openai_api_candidate",
                                     schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                                     environ={meter.METER_ROOT_ENV: str(root)}, now=NOVEMBER_29, maximum_cost_usd=12)
    events, _gaps = meter.read_journal(root)
    rows = meter.journal_rows(events)
    assert [(row["status"], row["held_usd"], row["hold_basis"]) for row in rows] == [
        ("held", Decimal(12), meter.UNPRICED_HOLD_BASIS)]
    view = meter.meter_view(rows, now=NOVEMBER_29)
    assert (view["totals"]["held_usd"], view["totals"]["settled_usd"], view["totals"]["unpriced_admissions"]) == (12, 0, 0)
    [key] = view["journal"]["holds_usd"]
    assert view["journal"]["holds_usd"][key] == "12"
    # A bill of its provider read an hour after it cannot show it yet; one read a day after can.
    early = meter.meter_view(rows, now=NOVEMBER_29, bills={"openai": NOVEMBER_29 + 3600})["totals"]
    late = meter.meter_view(rows, now=NOVEMBER_29, bills={"openai": NOVEMBER_29 + meter.BILL_COVERS_AFTER_SECONDS})
    assert early["held_usd"] == 12 and late["totals"]["held_usd"] == 0 and late["totals"]["unpriced_admissions"] == 1
    assert late["journal"]["holds_usd"] == {}
    # Another provider's bill shows nothing of it.
    other = meter.meter_view(rows, now=NOVEMBER_29, bills={"vast": NOVEMBER_29 + 2 * meter.BILL_COVERS_AFTER_SECONDS})
    assert other["totals"]["held_usd"] == 12
    # December: the call it was, never an old hold for someone to settle.
    december = meter.meter_view(rows, now=NOVEMBER_29 + 3 * 86400)
    assert december["totals"]["held_usd"] == 0 and december["old_unsettled_holds"] == []
    # Settled: its settlement counts, not its quote.
    meter.record_settlement(job_id=rows[0]["job_id"], meter_key=rows[0]["meter_key"], settled_usd="4.25",
                            basis="test", provider="openai", root=root, now=NOVEMBER_29 + 60)
    settled = meter.meter_view(meter.journal_rows(meter.read_journal(root)[0]), now=NOVEMBER_29 + 60)["totals"]
    assert (settled["held_usd"], settled["settled_usd"]) == (0, 4.25)
    assert meter.admitted_since(root, NOVEMBER_29, now=NOVEMBER_29) == Decimal(12)
    assert meter.admitted_since(root, NOVEMBER_29 + 1, now=NOVEMBER_29) == 0


# ----------------------------- a caller with no amount -----------------------------


def test_a_caller_with_no_amount_keeps_the_rule_it_had(tmp_path):
    """No amount is made up: a call that names none is refused only once the month reaches its limit, as before."""
    env = _limit(tmp_path, counted="399.99", generated_ago_seconds=60)
    assert account.admission_blockers(now=NOVEMBER_29, environ=env) == []
    assert account.admission_blockers(now=NOVEMBER_29, environ=env, maximum_cost_usd=None) == []
    env = _limit(tmp_path, counted="400.00", generated_ago_seconds=60)
    assert account.admission_blockers(now=NOVEMBER_29, environ=env) == [REACHED]


def test_an_unpriced_grant_is_admitted_as_before_and_adds_nothing_for_the_next_call(month, tmp_path):
    month("399.00", generated_ago_seconds=0)
    for resource_class in ("vast_provider_adapter", "openai_api_candidate", "fullenrich_work_email"):
        assert _admit(resource_class=resource_class).resource_class == resource_class
    require_paid_resource_budget(resource_class="openai_api_candidate")
    assert all(meter.MAXIMUM_COST_FIELD not in row for row in _journal(tmp_path))
    # Nothing is made up for them: a $1 call still fits ($399 + $1).
    assert _admit(1.0).resource_class == "evaluator_api"


def test_a_caller_with_no_amount_is_still_judged_with_the_calls_admitted_since_the_report(month):
    month("395.00", generated_ago_seconds=0)
    require_paid_resource_budget(resource_class="openai_api_candidate")
    _admit(5.0)  # $395 + $5 = $400: it fits, and the month is then at its limit
    with pytest.raises(PaidResourceAdmissionBlocked) as precheck:
        require_paid_resource_budget(resource_class="openai_api_candidate")
    assert precheck.value.blockers == [REACHED]
    with pytest.raises(PaidResourceAdmissionBlocked, match=REACHED):
        _admit(resource_class="vast_provider_adapter")


# ----------------------------- a stop -----------------------------


@pytest.mark.parametrize("counted", ["399.99", "450.00"])
def test_a_stop_is_never_refused_for_a_budget_reason_whatever_cost_it_names(month, tmp_path, counted):
    month(counted)
    for cost in (None, 40.0, -1, "not money"):
        assert _stop(cost).resource_class == "gpu_render"
        assert _stop(maximum_cost_usd=cost).resource_class == "gpu_render"
    assert {(row["spend_effect"], row.get(meter.MAXIMUM_COST_FIELD)) for row in _journal(tmp_path)} == {("stop", None)}
    with pytest.raises(PaidResourceAdmissionBlocked, match=REACHED):
        _admit(40.0, resource_class="gpu_render")


def test_stops_since_the_report_count_nothing_for_the_next_call(month):
    month("390.00", generated_ago_seconds=0)
    for _ in range(3):
        _stop(40.0, maximum_cost_usd=40.0)
    assert _admit(10.0).resource_class == "evaluator_api"  # $390 + $10: the stops added nothing


def test_a_stop_waits_on_no_lock_and_new_spend_does(month):
    month("399.99")
    stopped, admitted = threading.Event(), threading.Event()
    with meter.admission_lock():  # another admission is being judged
        threading.Thread(target=lambda: (_stop(), stopped.set()), daemon=True).start()
        assert stopped.wait(30), "a stop waited for the account check of another admission"
        threading.Thread(target=lambda: (_admit(), admitted.set()), daemon=True).start()
        assert not admitted.wait(0.5)
    assert admitted.wait(30)


# ----------------------------- callers that pass their cost -----------------------------


def test_a_website_preparation_quote_must_fit_before_the_website_is_asked(month, tmp_path, monkeypatch):
    """The precheck is asked for the purchase's quote; the grant is judged at the receipt's and journaled at it."""
    from blueprint_pipeline import website_task_context as context

    month("395.00", generated_ago_seconds=0)
    asked = []

    def webapp(**kwargs):
        asked.append(kwargs)
        spend = kwargs["payload"]["spend"]
        return {**spend, "schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION, "status": "admitted", "blockers": [],
                "expires_at_epoch": time.time() + 600, "external_disclosure_allowed": True}

    monkeypatch.setattr(context, "preparation_webapp_request", webapp)
    task_context = {"context_digest": "sha256:" + "1" * 64, "request_id": "req-1", "scene_id": "site-req-1",
                    "capture_id": "walkthrough-req-1"}

    def reserve(cost: float, binding: str):
        return context.reserve_website_preparation_spend(
            task_context=task_context, binding_digest="sha256:" + binding * 64, maximum_cost_usd=cost,
            request_count=1, resource_class="evaluator_api", provider="google")

    with pytest.raises(PaidResourceAdmissionBlocked, match=REACHED):
        reserve(6.0, "2")
    assert asked == []  # refused before any remote hold
    reserve(5.0, "3")
    assert len(asked) == 1
    [row] = _journal(tmp_path)
    assert (row["job_id"], row["meter_key"], row[meter.MAXIMUM_COST_FIELD]) == ("req-1", "sha256:" + "3" * 64, "5.0")
    with pytest.raises(PaidResourceAdmissionBlocked, match=REACHED):
        reserve(0.01, "4")  # $395 + the $5 since the report: the month is at its limit
    assert len(asked) == 1


def test_a_remote_cpu_attempt_is_judged_at_its_worst_case(month):
    from blueprint_pipeline import remote_cpu_job_allocator as allocator

    month("396.00")
    assert allocator.admit_remote_cpu_job(blockers=[], binding={"attempt_id": "a"}, execute=True,
                                          maximum_cost_usd=4.01)[1] is None
    assert allocator.admit_remote_cpu_job(blockers=[], binding={"attempt_id": "a"}, execute=True,
                                          maximum_cost_usd=4.0)[1] is not None


def test_a_findall_request_is_judged_at_its_own_ceiling(month):
    from tests.test_parallel_findall_admission import prepared

    from blueprint_pipeline import parallel_findall_admission as issuer

    month("398.50")
    with pytest.raises(PaidResourceAdmissionBlocked, match=REACHED):
        issuer.admit_exact_request(prepared(maximum_cost_usd="2.00"), blockers=[])
    assert issuer.admit_exact_request(prepared(maximum_cost_usd="1.00"), blockers=[]) is not None


@pytest.mark.parametrize(("counted", "admitted"), [("397.75", True), ("397.76", False)])
def test_a_scene_configuration_attempt_is_judged_at_its_hard_cap(month, tmp_path, monkeypatch, counted, admitted):
    """The attempt's whole hard cap ($2.25 in the fixture) must fit under the month."""
    from tests.test_task_evaluation_scene_configuration_allocator import (
        _args, _expected_source, _identity, _prepared_bundle)

    from blueprint_pipeline import task_evaluation_scene_configuration_allocator as lane

    month(counted)
    monkeypatch.setattr(lane, "resolve_host_resident_bundle_receipt", lambda _path: {"blockers": []})
    monkeypatch.setattr(lane, "load_scene_configuration_provider_bundle_receipt",
                        lambda _path, **_kwargs: _prepared_bundle())
    monkeypatch.setattr(lane, "validate_scene_configuration_paid_authority", lambda *_args, **_kwargs: None)
    runs = []
    monkeypatch.setattr(lane, "run_scene_configuration_vast", lambda **kwargs: runs.append(kwargs) or {"status": "completed"})
    args = _args(tmp_path, execute=True)
    code = lane.run_scene_configuration_allocator_probe(args, control_plane_identity_probe=_identity,
                                                        expected_source_commit_probe=_expected_source)
    assert (code == 0, len(runs)) == ((True, 1) if admitted else (False, 0))
    if not admitted:
        assert json.loads(Path(args.adapter_output).read_text())["blockers"] == [REACHED]


# ----------------------------- every account's grants are read (Codex review of #3716) -----------------------------


def test_each_accounts_journal_is_readable_by_every_account_and_an_older_strict_one_is_opened_up(tmp_path):
    """Root-run operator commands and the ``blueprint`` services journal under different accounts. A grant one
    journals must count for an admission the other judges, so each file is readable by every account: a new one is
    made so, and one made earlier under the old rule is opened up by its own writer at its next grant."""
    root = tmp_path / "meter"
    meter.record_admission(job_id="req-1", meter_key="first", resource_class="evaluator_api", root=root,
                           now=NOVEMBER_29, maximum_cost_usd="5.00")
    path = root / "events" / account.month_of(NOVEMBER_29) / f"uid-{os.geteuid()}.jsonl"
    assert os.stat(path).st_mode & 0o777 == meter.JOURNAL_FILE_MODE == 0o644
    os.chmod(path, 0o600)  # as the earlier rule left it
    meter.record_admission(job_id="req-1", meter_key="second", resource_class="evaluator_api", root=root,
                           now=NOVEMBER_29 + 1, maximum_cost_usd="5.00")
    assert os.stat(path).st_mode & 0o777 == 0o644


@pytest.mark.parametrize("damage", ["another_accounts_file", "not_text", "too_large", "not_a_file"])
def test_a_journal_file_that_cannot_be_read_whole_refuses_new_spend_rather_than_counting_nothing(
        tmp_path, monkeypatch, damage):
    """Codex review of #3716: a file of the month this process cannot read was skipped, so a root-issued $5 grant was
    $0 to a service, which could then admit past the limit. Now the month cannot be counted, so new spend waits
    (``JOURNAL_UNREADABLE``), the test-job door refuses too, and a stop still goes through."""
    env = _limit(tmp_path, counted="100.00", generated_ago_seconds=0)
    directory = tmp_path / "meter" / "events" / account.month_of(NOVEMBER_29)
    directory.mkdir(parents=True)
    other = directory / "uid-0.jsonl"
    if damage == "not_a_file":
        other.mkdir()
    else:
        other.write_bytes(b"\xff\xfe not text" if damage == "not_text" else b"")
    if damage == "too_large":
        monkeypatch.setattr(meter, "MAX_JOURNAL_FILE_BYTES", -1)
    if damage == "another_accounts_file":  # mode 0600 under another account: this process is refused, as on the host
        real = os.open

        def refused(path, flags, *args, **kwargs):
            if str(path).endswith("uid-0.jsonl") and not flags & (os.O_WRONLY | os.O_RDWR):
                raise PermissionError(13, "Permission denied", str(path))
            return real(path, flags, *args, **kwargs)
        monkeypatch.setattr(os, "open", refused)
    assert account.admission_blockers(now=NOVEMBER_29, environ=env, maximum_cost_usd="1.00") == [
        account.JOURNAL_UNREADABLE]
    assert account.admission_blockers(now=NOVEMBER_29, environ=env) == [account.JOURNAL_UNREADABLE]
    with pytest.raises(meter.JournalUnreadable):
        meter.admitted_since(tmp_path / "meter", NOVEMBER_29 - 60, now=NOVEMBER_29)


def test_the_test_job_door_waits_when_a_journal_file_cannot_be_read(tmp_path, monkeypatch):
    from blueprint_pipeline import internal_test_job_door as door

    env = _limit(tmp_path, counted="100.00", generated_ago_seconds=0)
    monkeypatch.setattr(account, "_clock", lambda: NOVEMBER_29)
    directory = tmp_path / "meter" / "events" / account.month_of(NOVEMBER_29)
    directory.mkdir(parents=True)
    (directory / "uid-0.jsonl").mkdir()
    assert door.month_refusal(environ=env) == "test_job_month_closed:" + account.JOURNAL_UNREADABLE
