"""Reviewer probe (stage 8c, finding 1): a plan-time *selected* reference holds money a waitlisted AI row needs."""
from copy import deepcopy
from decimal import Decimal

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline.beta_policy_screening_plan import validate_screening_plan
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_beta_policy_screening_coordinator import plan as queue_plan
from tests.test_beta_policy_screening_known_good_control import CONTROL, with_control


def _plan(reference):
    value = deepcopy(queue_plan(count=2, cost="4.00"))
    if reference:
        value = deepcopy(with_control(value))  # reference $3 selected: what the planner does at $7 with 2 x $4 robots
    value["budget_usd"] = "7.00"
    value["candidates"][-1].update(status="waitlisted", blockers=["site_screening_budget_insufficient"])
    value["maximum_planned_cost_usd"] = "7.00" if reference else "4.00"
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return validate_screening_plan(value)


def _attempt(cid, maximum, status, settled=None):
    a = {"attempt_id": cid + "-a1", "maximum_cost_usd": maximum, "reservation_status": status}
    if settled:
        a["settled_cost_usd"] = settled
    return a


@pytest.mark.parametrize("reference", [False, True])
def test_settled_leftover_funds_the_second_robot(tmp_path, monkeypatch, reference):
    monkeypatch.setattr(coordinator, "settled_cost", lambda a: Decimal(a["settled_cost_usd"]) if "settled_cost_usd" in a else None)
    coordinator.enqueue_screening_campaign(_plan(reference), queue_root=tmp_path)
    with coordinator._lock(tmp_path):
        ledger = coordinator._ledger(tmp_path)
        rows = ledger["campaigns"]["campaign-1"]["candidates"]
        rows["candidate-000"].update(status="completed", attempt=_attempt("c0", "4.00", "committed", "1.00"))
        if reference:
            # The reference launched first (plan order) and is still running, holding its $3.
            rows[CONTROL].update(status="running", attempt=_attempt("ctl", "3.00", "reserved"))
        coordinator._save(tmp_path, ledger)
    coordinator._refill_waitlist(tmp_path)
    second = coordinator._ledger(tmp_path)["campaigns"]["campaign-1"]["candidates"]["candidate-001"]
    print("reference" if reference else "no-reference", "->", second["status"], second["blockers"])
    if reference:
        # The reference now settles at its full $3: nothing can release money any more.
        with coordinator._lock(tmp_path):
            ledger = coordinator._ledger(tmp_path)
            ledger["campaigns"]["campaign-1"]["candidates"][CONTROL].update(
                status="completed", attempt=_attempt("ctl", "3.00", "committed", "3.00"))
            coordinator._save(tmp_path, ledger)
        coordinator._refill_waitlist(tmp_path)
        second = coordinator._ledger(tmp_path)["campaigns"]["campaign-1"]["candidates"]["candidate-001"]
        print("after reference settles $3 ->", second["status"], second["blockers"])
    assert second["status"] == "pending", "the reference cost the job its second AI robot"


def test_a_selected_reference_launches_although_no_ai_robot_ever_launches(tmp_path, monkeypatch, summaries):
    from tests.test_beta_policy_screening_known_good_control import PerRow, run
    coordinator.enqueue_screening_campaign(with_control(queue_plan(count=1)), queue_root=tmp_path)
    backend = PerRow({})
    run(tmp_path, backend, passes=1)  # max_launches=1: the reference goes first
    # The AI robot's quote then lapses (an hour passes while the reference runs): it can never launch.
    real = coordinator._quote_current
    monkeypatch.setattr(coordinator, "_quote_current", lambda c: real(c) if c.get("known_good_control") else False)
    [publication] = run(tmp_path, backend, passes=4)
    print("launched:", backend.order, {r["candidate_id"]: (r["status"], r.get("blockers")) for r in publication["candidates"]})
    assert backend.order != [CONTROL], "the reference launched and paid while no AI robot was ever tested"


from tests.test_beta_policy_screening_known_good_control import summaries  # noqa: E402,F401
