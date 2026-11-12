"""Reviewer probe (finding 2): a reference body that cannot reach is skipped; one that can is used."""
from blueprint_pipeline.website_candidate_feasibility import reach_rank
from tests.test_beta_policy_screening_known_good_control import catalog_with_bodies, website_plan, FRANKA_BODY

PANDA = "franka_emika_panda_robotiq_2f85_sim"
CANNOT = reach_rank({"status": "infeasible", "blockers": ["robot_reach_insufficient:place:1.086m_vs_0.990m"]})
FLAGGED = {"tier": 3, "basis": "own_arm_solve_flagged", "feasibility_status": "feasible", "reach_source": None,
           "solve_source": None, "solve_status": "flagged", "report_digest": None, "reasons": [], "feasibility_digest": None}


def _ref(ranks):
    catalog, caps, quotes = catalog_with_bodies([PANDA, FRANKA_BODY])
    caps = {**caps, "candidate_reach_rank": ranks}
    row = website_plan(catalog, caps, quotes)["candidates"][0]
    return row["candidate_id"], row["status"], row["known_good_control"].get("unavailable")


def test_first_body_cannot_reach_second_used():
    got = _ref({"blueprint_scripted_" + PANDA: CANNOT})
    print(got)
    assert got[:2] == ("blueprint_scripted_" + FRANKA_BODY, "selected")


def test_both_cannot_reach():
    got = _ref({"blueprint_scripted_" + PANDA: CANNOT, "blueprint_scripted_" + FRANKA_BODY: CANNOT})
    print(got)
    assert got[1:] == ("unavailable", "reach")


def test_reaching_and_flagged_and_unmeasured_bodies_not_excluded():
    for rank in (None, reach_rank(None), FLAGGED):
        got = _ref({} if rank is None else {"blueprint_scripted_" + PANDA: rank})
        print(got)
        assert got[:2] == ("blueprint_scripted_" + PANDA, "selected")
