"""The beta queue cannot duplicate spend, exceed the site allowance or upgrade ADP evidence."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import fcntl
import json
from pathlib import Path

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline.beta_policy_screening_plan import validate_screening_plan
from blueprint_pipeline.beta_policy_screening_result import candidate_result_summary
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    BETA_ADMISSION_SCHEMA_VERSION, BETA_RUN_KIND, RESULT_SCHEMA_VERSION, validate_session_result,
)
from tests.test_native_task_arena_policy_canary_session import _activation


def digest(value):
    return canonical_digest({"value": value})


def seal(value, field):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def plan(*, campaign_id="campaign-1", site_id="site-job-1", cost="4.00", count=1):
    contract = _activation()["task_success_contract"]
    scene_digest = digest(campaign_id)
    cells = []
    for index in range(10):
        scenario = {"family": "canonical_anchor", "ordinal": index, "parameters": {}}
        cells.append({"cell_id": f"screening.{site_id}.{index:03}", "seed": 1000 + index,
                      "partition": "screening", "family": "canonical_anchor", "resolved_scenario": scenario,
                      "cell_spec_digest": canonical_digest(scenario),
                      "resolved_scenario_digest": canonical_digest(scenario)})
    receipt = {"path": "/operator/sealed.json", "sha256": digest("sealed"), "size_bytes": 42}
    rows = []
    for index in range(count):
        candidate_id = f"candidate-{index:03}"
        configuration = digest(candidate_id)
        spec = {"schema_version": "blueprint.beta_policy_screening_execution.v1",
                "candidate_id": candidate_id, "policy_execution_spec": receipt, "checkpoint_inventory": None,
                "configuration_digest": configuration, "runtime_digest": digest("runtime"),
                "backend": "robolab.v1", "max_hourly_rate_usd": "0.01", "hard_ttl_seconds": 1800,
                "implementation_commit": "a" * 40, "task_success_contract_digest": contract["contract_digest"]}
        quote = seal({"schema_version": "beta_policy_screening_cost_quote.v1", "quote_id": candidate_id,
                      "price_source": "operator_resource_bound", "configuration_digest": configuration,
                      "scene_revision_digest": scene_digest, "episode_count": 10, "maximum_cost_usd": cost,
                      "valid_until": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}, "quote_digest")
        rows.append({"candidate_id": candidate_id, "display_name": candidate_id, "configuration_digest": configuration, "status": "selected",
                     "blockers": [], "execution_spec": spec, "admission_digest": canonical_digest(spec),
                     "cost_quote": quote, "max_cost_usd": cost, "episode_count": 10, "scenario_cells": cells})
    value = {"schema_version": "beta_policy_screening_plan.v1", "campaign_id": campaign_id,
             "site_id": site_id, "scene_id": site_id, "capture_id": "walkthrough-job-1", "task_id": "task-1",
             "scene_revision_digest": scene_digest, "task_success_contract_digest": contract["contract_digest"],
             "robot_free_probe_digest": digest("probe"), "scene_inputs": receipt, "catalog_digest": digest("catalog"),
             "scenario_cells": cells, "scenario_plan_digest": canonical_digest({"cells": cells}), "budget_usd": "15.00",
             "maximum_planned_cost_usd": f"{Decimal(cost) * count:.2f}", "purpose": "screening",
             "claim_ceiling": "diagnostic_policy_execution", "candidates": rows,
             "source": {"scene_intent_id": "scene-intent-1", "scene_intent_digest": digest("intent"),
                        "capture_id": "walkthrough-job-1", "task_id": "task-1",
                        "task_context_digest": digest("context"), "scene_revision_digest": scene_digest}}
    return validate_screening_plan(seal(value, "plan_digest"))


def native_result(value, candidate=None):
    candidate = candidate or value["candidates"][0]
    contract = _activation()["task_success_contract"]
    binding = {"schema_version": BETA_ADMISSION_SCHEMA_VERSION, "candidate_id": candidate["candidate_id"],
               "admission_digest": candidate["admission_digest"], "plan_digest": value["plan_digest"],
               "configuration_digest": candidate["configuration_digest"], "scene_revision_digest": value["scene_revision_digest"],
               "task_success_contract_digest": contract["contract_digest"],
               "robot_free_probe_receipt_digest": value["robot_free_probe_digest"],
               "evaluation_purpose": "screening", "backend": "robolab.v1"}
    episodes = []
    for index, cell in enumerate(candidate["scenario_cells"]):
        score = {"status": "scored", "task_succeeded": index < 7}
        episodes.append({**cell, "candidate_id": candidate["candidate_id"], "run_kind": BETA_RUN_KIND,
                         "claim_ceiling": "diagnostic_policy_execution", "ranking_eligible": False,
                         "status": "completed", "candidate_policy_queried": True, "actions_reached_robot": True,
                         "arm_moved": True, "checkpoint_digest": digest("checkpoint"),
                         "runtime_identity_digest": candidate["execution_spec"]["runtime_digest"],
                         **{field: digest(field) for field in ("lossless_frame_manifest_digest", "review_video_digest",
                              "returned_action_sequence_digest", "action_delivery_readback_digest", "state_trace_digest",
                              "contact_force_digest", "task_object_trajectory_digest")},
                         "deterministic_score_digest": digest(score), "scoring_authority": "deterministic_simulator_state",
                         "episode": {"score": score}})
    result = {"schema_version": RESULT_SCHEMA_VERSION, "status": "completed_unqualified", "run_kind": BETA_RUN_KIND,
              "claim_ceiling": "diagnostic_policy_execution", "candidate_ids": [candidate["candidate_id"]],
              "task_success_contract": contract, "task_success_contract_digest": contract["contract_digest"],
              "beta_screening_admission": binding, "scene_revision_digest": value["scene_revision_digest"],
              "plan_digest": value["plan_digest"], "configuration_digest": candidate["configuration_digest"],
              "episodes_per_policy": 10, "learned_policy_rollout_count": 10, "retry_cap": 0,
              "warm_session_open_count": 1, "provider_allocations_observed": 1, "episodes": episodes,
              "scene_promotion_performed": False, "official_ranking_performed": False,
              "session_closeout": {"provider_allocations_observed": 1, "teardown_completed": True,
                                   "provider_zero_confirmed": True}}
    return validate_session_result(seal(result, "result_digest"))


class Backend:
    def __init__(self, observation=None):
        self.observation = observation or {"status": "running"}
        self.preparations = self.launches = self.reconciliations = 0

    def prepare_candidate(self, **kwargs):
        self.preparations += 1
        return {"sealed": "local-inputs"}

    def launch_candidate(self, **kwargs):
        self.launches += 1
        if isinstance(self.observation, Exception):
            raise self.observation
        return self.observation

    def reconcile_candidate(self, **kwargs):
        self.reconciliations += 1
        if isinstance(self.observation, Exception):
            raise self.observation
        return self.observation


def publication(root, campaign_id="campaign-1"):
    return next(row for row in coordinator.pending_publications(queue_root=root) if row["campaign_id"] == campaign_id)


def test_site_budget_is_shared_across_scene_revisions_and_uncertain_launches(tmp_path):
    first, second = plan(cost="8.00"), plan(campaign_id="campaign-2", cost="8.00")
    for value in (first, second):
        coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    backend = Backend(RuntimeError("provider response lost"))
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend, max_launches=2)
    assert backend.launches == 1
    assert publication(tmp_path)["status"] == "reconciliation_required"
    assert publication(tmp_path)["budget"]["reserved_usd"] == "8"
    assert publication(tmp_path, "campaign-2")["candidates"][0]["blockers"] == ["beta_screening_site_budget_exhausted"]
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    assert backend.launches == 1 and backend.reconciliations == 1


def test_no_allocation_requires_explicit_proof_to_release(tmp_path):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = Backend({"status": "blocked", "provider_zero_verified": True,
                       "provider_allocation_performed": False})
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    assert publication(tmp_path)["budget"]["reserved_usd"] == "4"
    backend.observation["no_allocation_proven"] = True
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    assert publication(tmp_path)["status"] == "blocked"
    assert publication(tmp_path)["budget"]["committed_upper_bound_usd"] == "0"
    assert backend.launches == 1


def test_allocated_failure_charges_bound_and_cannot_claim_later_no_allocation(tmp_path):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = Backend({"status": "running", "provider_allocation_performed": True})
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    backend.observation = {"status": "blocked", "provider_allocation_performed": False, "no_allocation_proven": True}
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    assert publication(tmp_path)["budget"]["reserved_usd"] == "4"
    backend.observation = {"status": "blocked", "provider_zero_verified": True}
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    value = publication(tmp_path)
    assert value["budget"]["spent_upper_bound_usd"] == "4" and value["budget"]["reserved_usd"] == "0"
    assert value["status"] == "blocked"


def test_closed_native_result_within_budget_is_the_jobs_evaluation_result(tmp_path):
    value = plan()
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    backend = Backend({"status": "completed", "provider_allocation_performed": True,
                       "provider_zero_verified": True, "result": native_result(value)})
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    published = publication(tmp_path)
    row = published["candidates"][0]
    assert published["status"] == "completed"
    assert (row["episodes_scored"], row["episodes_succeeded"], row["episodes_failed"], row["success_rate"]) == (10, 7, 3, 0.7)
    # One tier: the automatic run is the job's result, stated as an estimate.
    assert published["schema_version"] == "beta_policy_screening_publication.v2"
    assert published["purpose"] == "job_evaluation" and published["claim_ceiling"] == "simulation_estimate"
    assert "recommendation_eligible" not in published
    assert published["budget"]["spent_upper_bound_usd"] == "4" and published["budget"]["limit_usd"] == "15"
    # ADP separation: the runtime session stays below qualified evidence.
    assert value["claim_ceiling"] == "diagnostic_policy_execution"
    assert published["source"]["capture_id"] != published["site_id"]
    assert published["publication_digest"] == cross_runtime_canonical_digest(published, digest_field="publication_digest")
    assert coordinator.acknowledge_publication(value["campaign_id"], published["publication_digest"], queue_root=tmp_path)
    assert coordinator.pending_publications(queue_root=tmp_path) == []


@pytest.mark.parametrize("tamper", ["plan", "runtime", "scenario", "score", "cell", "media"])
def test_other_execution_or_changed_score_cannot_be_counted(tamper):
    value = plan()
    result = native_result(value)
    if tamper == "plan":
        result["plan_digest"] = result["beta_screening_admission"]["plan_digest"] = digest("other")
    elif tamper == "runtime":
        for row in result["episodes"]:
            row["runtime_identity_digest"] = digest("other")
    elif tamper == "scenario":
        result["episodes"][0]["resolved_scenario"] = {"other": True}
        result["episodes"][0]["resolved_scenario_digest"] = canonical_digest({"other": True})
    elif tamper == "score":
        result["episodes"][0]["episode"]["score"]["task_succeeded"] = False
    elif tamper == "cell":
        result["episodes"][0]["seed"] += 100
    else:
        result["episodes"][0].pop("review_video_digest")
    seal(result, "result_digest")
    with pytest.raises(ValueError):
        candidate_result_summary(result, plan=value, candidate=value["candidates"][0])


def test_other_campaign_budget_change_gets_new_revision_and_old_ack_cannot_hide_it(tmp_path):
    first, second = plan(), plan(campaign_id="campaign-2")
    for value in (first, second):
        coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    old = publication(tmp_path, "campaign-2")
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=Backend(), max_campaigns=1)
    new = publication(tmp_path, "campaign-2")
    assert new["revision"] > old["revision"]
    assert new["budget"]["reserved_usd"] == "4"
    assert not coordinator.acknowledge_publication("campaign-2", old["publication_digest"], queue_root=tmp_path)
    assert publication(tmp_path, "campaign-2") == new


def test_acknowledged_bytes_from_an_older_contract_republish_as_a_new_revision(tmp_path):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    current = publication(tmp_path)
    assert coordinator.acknowledge_publication("campaign-1", current["publication_digest"], queue_root=tmp_path)
    with coordinator._lock(tmp_path):
        ledger = coordinator._ledger(tmp_path)
        ledger["campaigns"]["campaign-1"]["published_digest"] = digest("acknowledged-v1-bytes")
        ledger["ledger_digest"] = canonical_digest(ledger, digest_field="ledger_digest")
        coordinator._write(tmp_path / coordinator.LEDGER_FILENAME, ledger)
    resent = publication(tmp_path)
    assert resent["revision"] == current["revision"] + 1
    assert coordinator.acknowledge_publication("campaign-1", resent["publication_digest"], queue_root=tmp_path)
    assert coordinator.pending_publications(queue_root=tmp_path) == []


def test_replay_is_idempotent_and_changed_plan_cannot_reset_budget(tmp_path):
    value = plan()
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    assert coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)["status"] == "already_enqueued"
    with pytest.raises(ValueError, match="identity_changed"):
        coordinator.enqueue_screening_campaign(plan(cost="3.00"), queue_root=tmp_path)


def test_preparation_failure_releases_reservation_and_redacts_private_error(tmp_path):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    class FailedPreparation(Backend):
        def prepare_candidate(self, **kwargs):
            raise RuntimeError("https://private.example/?token=credential")
    backend = FailedPreparation()
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    value = publication(tmp_path)
    assert backend.launches == 0 and value["budget"]["committed_upper_bound_usd"] == "0"
    assert value["candidates"][0]["blockers"] == ["beta_screening_preparation_failed"]
    # Only the attempt keeps the failure behind the generic code; the publication never does.
    row = coordinator._ledger(tmp_path)["campaigns"][value["campaign_id"]]["candidates"][
        value["candidates"][0]["candidate_id"]]
    assert row["attempt"]["error"] == {"type": "builtins.RuntimeError",
                                       "message": "https://private.example/?<redacted>"}
    assert "credential" not in json.dumps(value) and "error" not in value["candidates"][0]


@pytest.mark.parametrize("raised", [OSError(28, "No space left on device"), RuntimeError("vast API HTTP 429")])
def test_capacity_preflight_failure_keeps_its_bounded_cause_on_the_attempt(tmp_path, raised):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)

    class FailedPreflight(Backend):
        def preflight_candidate_capacity(self, **kwargs):
            raise raised
    backend = FailedPreflight()
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    value = publication(tmp_path)
    assert backend.launches == 0 and value["budget"]["committed_upper_bound_usd"] == "0"
    assert value["candidates"][0]["blockers"] == ["beta_screening_capacity_preflight_failed"]
    attempt = coordinator._ledger(tmp_path)["campaigns"][value["campaign_id"]]["candidates"][
        value["candidates"][0]["candidate_id"]]["attempt"]
    assert attempt["reservation_status"] == "released"
    assert attempt["error"] == {"type": f"builtins.{type(raised).__name__}", "message": str(raised)}
    assert "error" not in value["candidates"][0]


def test_quote_expired_after_prepare_does_not_launch(tmp_path, monkeypatch):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    checks = iter([True, False])
    monkeypatch.setattr(coordinator, "_quote_current", lambda _: next(checks))
    backend = Backend()
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    assert backend.preparations == 1 and backend.launches == 0
    assert publication(tmp_path)["candidates"][0]["status"] == "deferred"
    assert publication(tmp_path)["budget"]["reserved_usd"] == "0"


def test_missing_ledger_after_attempt_fails_closed(tmp_path):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=Backend())
    (tmp_path / coordinator.LEDGER_FILENAME).unlink()
    with pytest.raises(ValueError, match="ledger_missing_after_attempt"):
        coordinator.enqueue_screening_campaign(plan(campaign_id="campaign-2"), queue_root=tmp_path)


def test_another_worker_cannot_launch_concurrently(tmp_path):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    with (tmp_path / ".worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        backend = Backend()
        assert coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)["status"] == "worker_busy"
        assert backend.launches == 0


def test_malformed_response_remains_reserved_and_never_relaunches(tmp_path):
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = Backend()
    backend.observation = None
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend)
    assert backend.launches == 1 and publication(tmp_path)["budget"]["reserved_usd"] == "4"


def test_forty_candidates_are_not_hard_coded_to_pair_or_team200(tmp_path):
    coordinator.enqueue_screening_campaign(plan(cost="0.25", count=40), queue_root=tmp_path)
    backend = Backend({"status": "blocked", "no_allocation_proven": True, "provider_allocation_performed": False})
    outcome = coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend, max_launches=100)
    assert outcome["launches_started"] == backend.launches == 40
    assert len(publication(tmp_path)["candidates"]) == 40
    assert publication(tmp_path)["budget"]["committed_upper_bound_usd"] == "0"


def test_uncertain_campaign_does_not_starve_other_sites(tmp_path):
    for index in range(3):
        coordinator.enqueue_screening_campaign(plan(campaign_id=f"campaign-{index}", site_id=f"site-job-{index}"),
                                               queue_root=tmp_path)
    backend = Backend(RuntimeError("uncertain response"))
    visited = [coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path, executor=backend, max_campaigns=1)["campaign_ids"]
               for _ in range(4)]
    assert visited == [["campaign-0"], ["campaign-1"], ["campaign-2"], ["campaign-0"]]
    assert backend.launches == 3 and backend.reconciliations == 1


def test_default_executor_is_the_canonical_allocator_bridge(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_launch as bridge
    backend = Backend({"status": "blocked", "no_allocation_proven": True, "provider_allocation_performed": False})
    for method in ("prepare_candidate", "launch_candidate", "reconcile_candidate"):
        monkeypatch.setattr(bridge, method, getattr(backend, method))
    monkeypatch.setattr(bridge, "preflight_candidate_capacity", lambda **_: {"capacity_available": True, "blockers": []})
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=tmp_path)
    assert backend.preparations == backend.launches == 1


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_shared_wire_fixture_is_valid_and_its_nonterminating_fraction_digest_is_stable(version):
    # v2 is emitted; retained v1 records stay readable by the website.
    from jsonschema import Draft202012Validator, FormatChecker
    repository = Path(__file__).parents[1]
    value = json.loads((repository / f"tests/fixtures/beta_policy_screening_publication.{version}.json").read_text())
    schema = json.loads((repository / f"docs/schemas/beta_policy_screening_publication.{version}.schema.json").read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)
    assert value["candidates"][0]["success_rate"] == 2 / 3
    assert value["publication_digest"] == cross_runtime_canonical_digest(value, digest_field="publication_digest")


def test_default_gate_never_prepares_or_reserves_without_exact_website_ack(tmp_path):
    value = plan()
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    backend = Backend()
    for _ in range(2):
        coordinator.process_screening_queue(queue_root=tmp_path, executor=backend)
    assert (backend.preparations, backend.launches, backend.reconciliations) == (0, 0, 0)
    assert publication(tmp_path)["budget"]["committed_upper_bound_usd"] == "0"
    assert not coordinator.acknowledge_publication(value["campaign_id"], digest("wrong-ack"), queue_root=tmp_path)
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend)
    assert backend.preparations == backend.launches == 0
    current = publication(tmp_path)
    assert coordinator.acknowledge_publication(value["campaign_id"], current["publication_digest"], queue_root=tmp_path)
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend)
    assert backend.preparations == backend.launches == 1
    assert publication(tmp_path)["budget"]["reserved_usd"] == "4"


def test_failed_website_delivery_blocks_new_spend_but_never_blocks_reconciliation(tmp_path):
    from blueprint_pipeline.beta_policy_screening_worker import deliver_publications
    value = plan()
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    backend = Backend(RuntimeError("uncertain response"))
    def rejected(_):
        raise ValueError("beta_screening_publication_http_409")
    assert deliver_publications(queue_root=tmp_path, transport=rejected)["pending"] == 1
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend)
    assert backend.preparations == backend.launches == 0
    published = publication(tmp_path)
    coordinator.acknowledge_publication(value["campaign_id"], published["publication_digest"], queue_root=tmp_path)
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend)
    assert publication(tmp_path)["status"] == "reconciliation_required"
    assert deliver_publications(queue_root=tmp_path, transport=rejected)["pending"] == 1
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend)
    assert backend.launches == 1 and backend.reconciliations == 1
    assert publication(tmp_path)["budget"]["reserved_usd"] == "4"


def test_restarted_preparing_attempt_requires_ack_of_its_latest_progress(tmp_path):
    value = plan()
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    old = publication(tmp_path)
    coordinator.acknowledge_publication(value["campaign_id"], old["publication_digest"], queue_root=tmp_path)
    assert coordinator._reserve(tmp_path, value, value["candidates"][0]) is not None
    backend = Backend()
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend)
    assert backend.preparations == backend.launches == 0
    assert publication(tmp_path)["candidates"][0]["status"] == "preparing"
    current = publication(tmp_path)
    coordinator.acknowledge_publication(value["campaign_id"], current["publication_digest"], queue_root=tmp_path)
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend)
    assert backend.preparations == backend.launches == 1
