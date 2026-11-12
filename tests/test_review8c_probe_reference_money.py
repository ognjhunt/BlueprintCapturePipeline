"""Reviewer probe (finding 8): random campaigns/attempts/settlements against control_gate; total <= $3.00."""
import random
from copy import deepcopy
from decimal import Decimal

from blueprint_pipeline import beta_policy_screening_known_good_control as control
from blueprint_pipeline import beta_policy_screening_site_limit  # noqa: F401
from tests.test_beta_policy_screening_coordinator import plan as queue_plan
from tests.test_beta_policy_screening_known_good_control import CONTROL, _summary, settled, with_control  # noqa: F401


def test_random_sequences_never_exceed_three_dollars(settled, monkeypatch):  # noqa: F811
    from blueprint_pipeline import beta_policy_screening_site_limit as site_limit
    monkeypatch.setattr(site_limit, "budget_reason", lambda plan, code: code)
    rng = random.Random(3)
    base = queue_plan(count=1)
    plans = {}
    worst = Decimal(0)
    for trial in range(400):
        ledger = {"campaigns": {}}
        actual = Decimal(0)
        for step in range(rng.randint(2, 8)):
            cid = f"campaign-{step}"
            cost = f"{Decimal(rng.randint(10, 300)) / 100:.2f}"
            # a re-plan may pick another scripted body: the row id differs, the check is the same
            plans[cid] = p = with_control(base, cost=cost, campaign_id=cid)
            ledger["campaigns"][cid] = {"site_id": p["site_id"], "plan_digest": p["plan_digest"],
                                        "candidates": {CONTROL: {"status": "pending", "known_good_control": True},
                                                       "candidate-000": {"status": "pending"}}}
            decision = control.control_gate(ledger, cid, p, plans=lambda k, d: plans[k])
            row = ledger["campaigns"][cid]["candidates"][CONTROL]
            if decision != "proceed":
                continue
            end = rng.choice(["interrupted", "released", "unsettled", "classed", "running"])
            attempt = {"maximum_cost_usd": cost, "reservation_status": "reserved"}
            if end == "running":
                row.update(status="running", attempt=attempt)
                actual += Decimal(cost)  # may still spend all of it
                continue
            if end == "released":
                attempt["reservation_status"] = "released"
                row.update(status="blocked", blockers=["x"], attempt=attempt)
                continue
            attempt["reservation_status"] = "committed"
            if end != "unsettled":
                spent = Decimal(rng.randint(0, int(Decimal(cost) * 100))) / 100
                attempt["settled_cost_usd"] = f"{spent:.2f}"
            else:
                spent = Decimal(cost)
            actual += spent
            if end == "classed":
                row.update(status="completed", attempt=attempt, summary=_summary(3, rng.randint(0, 3), scripted=True, planned=3))
            else:
                row.update(status="blocked", blockers=["beta_screening_native_result_missing_or_invalid"], attempt=attempt)
            # running rows of earlier campaigns end later, in a random order
            for other in ledger["campaigns"].values():
                r = other["candidates"][CONTROL]
                if r["status"] == "running" and rng.random() < 0.5:
                    r["attempt"]["reservation_status"] = "committed"
                    r.update(status="blocked", blockers=["x"])
        worst = max(worst, actual)
        assert actual <= Decimal("3.00"), (trial, actual)
    print("worst total", worst)
