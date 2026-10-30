"""After the published result, a deterministic rule chooses and names the deployment partners; no human input."""
from __future__ import annotations

import json
import random

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import website_candidate_selection as selection
from blueprint_pipeline.beta_policy_candidate_catalog import catalog_candidate, load_default_catalog
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.deployment_embodiment_support import support_catalog
from tests.test_beta_policy_screening_coordinator import Backend, native_result, plan as coordinator_plan, publication

SHA = "sha256:" + "a" * 64


def result_row(candidate_id, status="completed", *, succeeded=0, scored=0, void=0, blockers=(), kind=None):
    configured = catalog_candidate(load_default_catalog(), candidate_id)
    row = {"candidate_id": candidate_id, "configuration_digest": configured["configuration_digest"],
           "display_name": configured["display_name"], "status": status, "blockers": list(blockers),
           "episodes_planned": 10 if status == "completed" else 0, "episodes_scored": scored,
           "episodes_succeeded": succeeded, "episodes_failed": scored - succeeded, "episodes_void": void,
           "success_rate": succeeded / scored if scored else None,
           "session_result_digest": SHA if status == "completed" else None}
    if kind:
        row["execution_evidence_kind"] = kind
    return row


def campaign(rows, plan_status=None):
    """A terminal publication and its plan; the plan lists candidates in priority order."""
    plan_status = plan_status or {}
    plan = {"campaign_id": "campaign-1", "plan_digest": SHA, "catalog_digest": load_default_catalog()["catalog_digest"],
            "candidates": [{"candidate_id": row["candidate_id"],
                            "status": plan_status.get(row["candidate_id"],
                                                      "selected" if row["status"] == "completed" else "unavailable"),
                            "blockers": list(row["blockers"])} for row in rows]}
    published = {"campaign_id": "campaign-1", "site_id": "site-job1", "publication_digest": SHA, "revision": 4,
                 "status": "completed" if any(row["status"] == "completed" for row in rows) else "blocked",
                 "claim_ceiling": "simulation_estimate", "candidates": rows}
    return plan, published


def choose(rows, **kwargs):
    plan, published = campaign(rows, **kwargs)
    return selection.select_candidate(publication=published, plan=plan, catalog=load_default_catalog(),
                                      catalog_basis="campaign_catalog", support=support_catalog())


def test_choice_is_deterministic_by_published_scores_with_priority_breaking_ties_and_partners_named():
    rows = [result_row("fanuc_m710ic_45m__pi05_custom_checkpoint", succeeded=7, scored=10),
            result_row("jacobi_fanuc_m710ic_45m", succeeded=7, scored=10),
            result_row("pi05_droid", succeeded=6, scored=10),
            result_row("blueprint_scripted_fanuc_m710ic_45m_robotiq_2f85_sim", succeeded=10, scored=10,
                       kind="scripted_simulator_state_baseline"),
            result_row("dreamlabs_dream1", "unavailable", blockers=["candidate_disabled"]),
            result_row("groot_n17_droid", "unavailable", blockers=["candidate_infeasible_measured_geometry",
                                                                   "team_embodiment_station_out_of_reach"])]
    value = choose(rows, plan_status={"groot_n17_droid": "infeasible"})
    assert value["status"] == "chosen" and value["decided_by"] == "deterministic_rule_no_human_input"
    chosen = value["chosen"]
    assert chosen["candidate_id"] == "fanuc_m710ic_45m__pi05_custom_checkpoint"
    assert value["tied_with"] == ["jacobi_fanuc_m710ic_45m"]
    assert (chosen["configuration_id"], chosen["policy_route_id"]) == ("fanuc_m710ic_45m", "pi05_custom_checkpoint")
    assert chosen["deployment_companies"] == ["Jacobi Robotics", "Delta Technology"]
    assert chosen["related_teams"] == ["Jacobi Robotics", "Delta Technology"] and chosen["partner_blockers"] == []
    shown = {row["candidate_id"]: row for row in value["candidates"]}
    assert set(shown) == {row["candidate_id"] for row in rows}
    assert shown["blueprint_scripted_fanuc_m710ic_45m_robotiq_2f85_sim"]["choice_blocker"] == \
        "scripted_baseline_reference_only"
    assert shown["groot_n17_droid"]["plan_status"] == "infeasible"
    assert shown["groot_n17_droid"]["blockers"][-1] == "team_embodiment_station_out_of_reach"
    assert shown["dreamlabs_dream1"]["choice_blocker"] == "not_evaluated"
    assert [shown[key].get("result_rank") for key in ("fanuc_m710ic_45m__pi05_custom_checkpoint",
                                                       "jacobi_fanuc_m710ic_45m", "pi05_droid")] == [1, 2, 3]
    # The publication's row order never changes the choice or the record.
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    plan, published = campaign(rows, plan_status={"groot_n17_droid": "infeasible"})
    published["candidates"] = shuffled
    again = selection.select_candidate(publication=published, plan=plan, catalog=load_default_catalog(),
                                       catalog_basis="campaign_catalog", support=support_catalog())
    assert again["chosen"] == chosen and again["tied_with"] == value["tied_with"]
    assert value["selection_digest"] == canonical_digest(value, digest_field="selection_digest")


def test_more_scored_episodes_then_fewer_void_episodes_break_equal_success_rates():
    value = choose([result_row("pi05_droid", succeeded=4, scored=8, void=2),
                    result_row("groot_n17_droid", succeeded=5, scored=10)])
    assert value["chosen"]["candidate_id"] == "groot_n17_droid" and value["tied_with"] == []
    assert value["chosen"]["partner_blockers"] == ["no_recorded_deployment_partner_for_configuration"]
    assert value["chosen"]["deployment_companies"] == [] and value["chosen"]["configuration_id"] is None


def test_no_evaluated_candidate_chooses_nothing_and_keeps_every_reason():
    rows = [result_row("pi05_droid", "unavailable", blockers=["runtime_candidate_binding_missing"]),
            result_row("jacobi_fanuc_m710ic_45m", "unavailable", blockers=["prepared_robot_policy_registration_missing"]),
            result_row("groot_n17_droid", "deferred", blockers=["site_screening_budget_insufficient"])]
    value = choose(rows)
    assert value["status"] == "no_evaluated_candidate" and value["chosen"] is None and value["tied_with"] == []
    assert all(row["blockers"] and row["choice_blocker"] == "not_evaluated" for row in value["candidates"])


def test_evaluated_candidates_without_a_success_choose_nothing():
    value = choose([result_row("pi05_droid", succeeded=0, scored=10),
                    result_row("groot_n17_droid", succeeded=0, scored=0, void=10)])
    assert value["status"] == "no_successful_candidate" and value["chosen"] is None
    assert {row["candidate_id"]: row["choice_blocker"] for row in value["candidates"]} == {
        "pi05_droid": None, "groot_n17_droid": "no_scored_episodes"}


def test_selection_is_retained_only_after_the_terminal_result_is_published(tmp_path):
    value = coordinator_plan()
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=Backend({
        "status": "completed", "provider_allocation_performed": True, "provider_zero_verified": True,
        "result": native_result(value)}))
    assert selection.retain_selections(queue_root=tmp_path) == {"written": [], "terminal_published": 0}
    published = publication(tmp_path)
    assert coordinator.acknowledge_publication(value["campaign_id"], published["publication_digest"],
                                               queue_root=tmp_path)
    assert selection.retain_selections(queue_root=tmp_path) == {"written": ["campaign-1"], "terminal_published": 1}
    record = json.loads((tmp_path / "selections" / "campaign-1.json").read_text())
    assert record["status"] == "chosen" and record["publication_digest"] == published["publication_digest"]
    assert record["chosen"]["candidate_id"] == "candidate-000" and record["chosen"]["success_rate"] == 0.7
    # This plan's catalog was not retained, so the current catalog is used and says so.
    assert record["catalog_basis"] == "current_default_catalog_not_campaign_catalog"
    assert record["chosen"]["partner_blockers"] == ["catalog_row_unavailable"]
    assert selection.retain_selections(queue_root=tmp_path) == {"written": [], "terminal_published": 1}


def test_the_plans_own_effective_catalog_is_retained_for_the_partner_match(tmp_path):
    catalog = load_default_catalog()
    selection.retain_campaign_catalog(queue_root=tmp_path, campaign_id="campaign-1", catalog=catalog)
    plan = {"campaign_id": "campaign-1", "catalog_digest": catalog["catalog_digest"]}
    assert selection._campaign_catalog(tmp_path, plan) == (catalog, "campaign_catalog")
    changed = {**plan, "catalog_digest": SHA}
    assert selection._campaign_catalog(tmp_path, changed)[1] == "current_default_catalog_not_campaign_catalog"
