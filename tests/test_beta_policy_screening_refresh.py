"""Immutable quote refresh cannot renew consent, budgets, or uncertain launches."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_refresh as refresh
from blueprint_pipeline import beta_policy_screening_worker as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_beta_policy_screening_coordinator import Backend, native_result, plan, seal
from tests.test_beta_policy_screening_capacity_coordinator import CapacityBackend, RaceBackend


@pytest.fixture
def clock(monkeypatch):
    class Clock(datetime):
        instant = datetime.now(timezone.utc)

        @classmethod
        def now(cls, tz=None):
            return cls.instant.astimezone(tz) if tz else cls.instant.replace(tzinfo=None)
    monkeypatch.setattr(coordinator, "datetime", Clock)
    monkeypatch.setattr(refresh, "datetime", Clock)
    return Clock


def source_authority(tmp_path, monkeypatch, value):
    """Use the production reopen/join; stub only current website and retained authority."""
    from blueprint_pipeline import task_evaluation_scene_owner_authority as owner
    root = tmp_path / "intake"
    directory = root / value["source"]["scene_intent_id"]
    directory.mkdir(parents=True)
    intent = {"intent_id": value["source"]["scene_intent_id"],
              "intent_digest": value["source"]["scene_intent_digest"],
              "request": {"submission_id": value["capture_id"], "task": {"task_id": value["task_id"]}}}
    (directory / "intent.json").write_text(json.dumps(intent))
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(root))
    calls = []
    def reopen(reference):
        calls.append(reference)
        assert Path(reference["path"]) == directory / "intent.json"
        assert reference["size_bytes"] == (directory / "intent.json").stat().st_size
        return deepcopy(intent)
    monkeypatch.setattr(owner, "reopen_scene_intent", reopen)
    monkeypatch.setattr(worker, "require_current_source", lambda *_args, **_kwargs: {"scene_id": value["site_id"]})
    return calls


def builder(clock, *, cost=None, change=None):
    calls = []
    def build(previous, next_id, _inputs):
        calls.append(next_id)
        result = deepcopy(previous)
        result["campaign_id"] = next_id
        for row in result["candidates"]:
            if row["status"] != "selected":
                continue
            row["cost_quote"]["valid_until"] = (clock.instant + timedelta(days=1)).isoformat()
            if cost:
                row["max_cost_usd"] = row["cost_quote"]["maximum_cost_usd"] = cost
            seal(row["cost_quote"], "quote_digest")
        result["maximum_planned_cost_usd"] = f"{sum(float(row['max_cost_usd']) for row in result['candidates'] if row['status'] == 'selected'):.2f}"
        if change:
            change(result)
        return seal(result, "plan_digest")
    return build, calls


def process(root, backend, **kwargs):
    return coordinator.process_screening_queue(queue_root=root, executor=backend,
        require_publication_ack=False, **kwargs)


def row(root, campaign_id="campaign-1", candidate_id="candidate-000"):
    return coordinator._ledger(root)["campaigns"][campaign_id]["candidates"][candidate_id]


@pytest.mark.parametrize("raced_slot", [False, True])
def test_twenty_five_hour_outage_refreshes_one_unspent_generation_and_one_allocation(
        tmp_path, monkeypatch, clock, raced_slot):
    value = plan()
    authority = source_authority(tmp_path, monkeypatch, value)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    original_bytes = (tmp_path / "plans/campaign-1.json").read_bytes()
    backend = RaceBackend() if raced_slot else CapacityBackend()
    process(tmp_path, backend)
    previous = deepcopy(row(tmp_path))
    assert previous["status"] == "preparing"
    clock.instant += timedelta(hours=25)
    process(tmp_path, backend)
    assert row(tmp_path)["blockers"] == ["beta_screening_cost_quote_expired"]
    build, builds = builder(clock)
    outcome = refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    assert outcome["refresh_status"] == "refreshed" and len(builds) == len(authority) == 1
    current_id = outcome["campaign_id"]
    current = coordinator._plan(tmp_path, current_id)
    assert current["catalog_digest"] == value["catalog_digest"] and current["source"] == value["source"]
    assert (tmp_path / "plans/campaign-1.json").read_bytes() == original_bytes
    assert row(tmp_path)["attempt"].get("prepared") == previous["attempt"].get("prepared")
    assert row(tmp_path).get("attempt_history") == previous.get("attempt_history")
    assert coordinator._budget(coordinator._ledger(tmp_path), value["site_id"])["committed_upper_bound_usd"] == "4"
    backend.busy = False
    process(tmp_path, backend)
    assert backend.launches == (2 if raced_slot else 1)
    assert row(tmp_path, current_id)["status"] == "running"
    for _ in range(4):
        refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
        process(tmp_path, backend)
    assert len(builds) == 1 and len(list((tmp_path / "plans").glob("*.json"))) == 2
    assert backend.launches == (2 if raced_slot else 1)
    assert coordinator._budget(coordinator._ledger(tmp_path), value["site_id"])["committed_upper_bound_usd"] == "4"


def test_completed_and_uncertain_candidates_stay_in_original_attempts(tmp_path, monkeypatch, clock):
    value = plan(count=3)
    source_authority(tmp_path, monkeypatch, value)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    launched = []
    class Mixed(Backend):
        def launch_candidate(self, **kwargs):
            key = kwargs["candidate"]["candidate_id"]
            launched.append((kwargs["plan"]["campaign_id"], key))
            if key == "candidate-000":
                return {"status": "completed", "provider_allocation_performed": True,
                        "provider_zero_verified": True, "result": native_result(value, value["candidates"][0])}
            if key == "candidate-001":
                raise OSError("lost response")
            return {"status": "running", "provider_allocation_performed": True}
        def reconcile_candidate(self, **kwargs):
            return {"status": "reconciliation_required"}
    backend = Mixed()
    process(tmp_path, backend, max_launches=2)
    completed = deepcopy(row(tmp_path))
    clock.instant += timedelta(hours=25)
    process(tmp_path, backend)
    uncertain = deepcopy(row(tmp_path, candidate_id="candidate-001"))
    build, builds = builder(clock)
    outcome = refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    current_id = outcome["campaign_id"]
    assert row(tmp_path) == completed
    assert row(tmp_path, candidate_id="candidate-001") == uncertain
    assert row(tmp_path, current_id)["status"] == "completed" and "attempt" not in row(tmp_path, current_id)
    assert row(tmp_path, current_id, "candidate-001")["prior_campaign_id"] == "campaign-1"
    for _ in range(3):
        process(tmp_path, backend)
        refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    assert launched == [("campaign-1", "candidate-000"), ("campaign-1", "candidate-001"),
                        (current_id, "candidate-002")]
    budget = coordinator._budget(coordinator._ledger(tmp_path), value["site_id"])
    assert budget["committed_upper_bound_usd"] == "12" and budget["spent_upper_bound_usd"] == "4"
    # A later real result joins from the old attempt; adoption never duplicates its charge.
    coordinator._observe(tmp_path, value, value["candidates"][1], {"status": "completed",
        "provider_allocation_performed": True, "provider_zero_verified": True,
        "result": native_result(value, value["candidates"][1])})
    refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    assert row(tmp_path, current_id, "candidate-001")["status"] == "completed"
    assert "attempt" not in row(tmp_path, current_id, "candidate-001") and len(builds) == 1
    assert coordinator._budget(coordinator._ledger(tmp_path), value["site_id"])["committed_upper_bound_usd"] == "12"


def test_failed_ledger_commit_cannot_expose_adopted_completed_candidate_as_pending(tmp_path, monkeypatch, clock):
    value = plan(count=2)
    source_authority(tmp_path, monkeypatch, value)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    backend = Backend({"status": "completed", "provider_allocation_performed": True,
        "provider_zero_verified": True, "result": native_result(value)})
    process(tmp_path, backend)
    clock.instant += timedelta(hours=25)
    process(tmp_path, backend)
    before = (tmp_path / coordinator.LEDGER_FILENAME).read_bytes()
    build, builds = builder(clock)
    save = coordinator._save
    def crash(*_args):
        raise OSError("crash before atomic ledger replace")
    monkeypatch.setattr(coordinator, "_save", crash)
    with pytest.raises(OSError):
        refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    assert (tmp_path / coordinator.LEDGER_FILENAME).read_bytes() == before
    assert refresh.active_campaign_id(tmp_path, "campaign-1") == "campaign-1"
    assert len(list((tmp_path / "plans").glob("*.json"))) == 2
    monkeypatch.setattr(coordinator, "_save", save)
    clock.instant += timedelta(seconds=301)
    outcome = refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    assert len(builds) == 1
    assert row(tmp_path, outcome["campaign_id"])["status"] == "completed"
    assert row(tmp_path, outcome["campaign_id"], "candidate-001")["status"] == "preparing"
    assert "attempt" not in row(tmp_path, outcome["campaign_id"])


@pytest.mark.parametrize("changed", ["runtime", "policy_bytes", "configuration", "source", "implementation", "cells"])
def test_refresh_refuses_changed_immutable_bindings(tmp_path, monkeypatch, clock, changed):
    value = plan()
    source_authority(tmp_path, monkeypatch, value)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    clock.instant += timedelta(hours=25)
    def change(fresh):
        candidate = fresh["candidates"][0]
        if changed == "runtime":
            candidate["execution_spec"]["runtime_digest"] = "sha256:" + "b" * 64
        elif changed == "policy_bytes":
            candidate["execution_spec"]["policy_execution_spec"]["sha256"] = "sha256:" + "b" * 64
        elif changed == "implementation":
            candidate["execution_spec"]["implementation_commit"] = "b" * 40
        elif changed == "configuration":
            candidate["configuration_digest"] = candidate["execution_spec"]["configuration_digest"] = "sha256:" + "b" * 64
            candidate["cost_quote"]["configuration_digest"] = candidate["configuration_digest"]
            seal(candidate["cost_quote"], "quote_digest")
        elif changed == "source":
            fresh["source"]["scene_intent_digest"] = "sha256:" + "b" * 64
        else:
            fresh["scenario_cells"][0]["seed"] += 1
            fresh["scenario_plan_digest"] = canonical_digest({"cells": fresh["scenario_cells"]})
        candidate["admission_digest"] = canonical_digest(candidate["execution_spec"])
    build, _ = builder(clock, change=change)
    with pytest.raises(ValueError, match="refresh_|selected_cells_invalid|scenario_plan_invalid"):
        refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    assert len(coordinator._ledger(tmp_path)["campaigns"]) == 1
    assert row(tmp_path)["status"] == "pending" and "attempt" not in row(tmp_path)


def test_quote_increase_above_remaining_site_budget_has_explicit_action(tmp_path, monkeypatch, clock):
    value = plan(cost="8.00")
    source_authority(tmp_path, monkeypatch, value)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    other = plan(campaign_id="other-revision", cost="8.00")
    coordinator.enqueue_screening_campaign(other, queue_root=tmp_path)
    coordinator._reserve(tmp_path, other, other["candidates"][0])
    clock.instant += timedelta(hours=25)
    build, builds = builder(clock, cost="9.00")
    outcome = refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    assert outcome["refresh_status"] == "cost_above_remaining_site_budget"
    assert outcome["next_action"] == "retain_site_commitments_and_review_cost_or_existing_authority"
    assert coordinator._budget(coordinator._ledger(tmp_path), value["site_id"])["reserved_usd"] == "8"
    assert "attempt" not in row(tmp_path) and len(list((tmp_path / "plans").glob("*.json"))) == 2
    for _ in range(3):
        refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    assert len(builds) == 1


@pytest.mark.parametrize("fault", ["expired", "revoked", "current_rights"])
def test_refresh_reopens_real_canonical_authority_and_backs_off_on_stop(tmp_path, monkeypatch, clock, fault):
    from blueprint_pipeline import task_evaluation_scene_owner_authority as owner
    from tests.test_scene_intent_retained_readers import retained_intent
    directory, intent = retained_intent.__wrapped__(tmp_path / "intake", monkeypatch)
    value = plan()
    value.update(capture_id=intent["request"]["submission_id"], task_id=intent["request"]["task"]["task_id"])
    value["source"].update(capture_id=value["capture_id"], task_id=value["task_id"],
        scene_intent_id=intent["intent_id"], scene_intent_digest=intent["intent_digest"])
    seal(value, "plan_digest")
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path / "screening")
    monkeypatch.setattr(worker, "require_current_source", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(owner.time, "time", lambda: 1000 if fault == "expired" else 201)
    if fault == "revoked":
        (directory / "revoked.json").write_text("{}")
    elif fault == "current_rights":
        def denied(*_args, **_kwargs):
            raise ValueError("beta_screening_current_capture_consent_required")
        monkeypatch.setattr(worker, "require_current_source", denied)
    clock.instant += timedelta(hours=25)
    build, builds = builder(clock)
    root = tmp_path / "screening"
    with pytest.raises(ValueError, match="authority_expired|authority_revoked|capture_consent_required"):
        refresh.maybe_refresh_campaign(queue_root=root, base_campaign_id="campaign-1", build=build)
    assert refresh.maybe_refresh_campaign(queue_root=root, base_campaign_id="campaign-1", build=build)["status"] == "already_enqueued"
    assert builds == [] and len(list((root / "plans").glob("*.json"))) == 1
    assert not list((root / "attempts").iterdir())


def test_new_private_hosted_credential_refreshes_only_pinned_affected_candidate(tmp_path, monkeypatch, clock):
    from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog, seal_catalog
    from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs
    from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
    from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture
    from tests.test_beta_policy_screening_hosted import TOKEN, hosted_plan
    ready, _specs, settings = hosted_plan(tmp_path, monkeypatch)
    candidates = []
    template = next(item for item in load_default_catalog()["candidates"] if item["candidate_id"] == "pi05_droid")
    for key in ("hosted_alpha", "hosted_beta"):
        configured = deepcopy(template)
        configured.update(candidate_id=key, display_name=key)
        configured["interface"]["policy_interface_id"] = "openpi_droid_https.v1"
        configured["policy"].update(kind="hosted", source="https://policy.example.com", revision="release-1",
                                     digest_kind="provider_metadata")
        configured["eligibility"]["requires_credentials"] = ["HOSTED_DROID_CREDENTIAL"]
        candidates.append(configured)
    catalog = seal_catalog({**load_default_catalog(), "candidates": candidates})
    missing = tmp_path / "operator" / "later-private-alpha"
    settings["hosted_alpha"]["credential_file"] = str(missing)
    profile = profile_for_capture(str(tmp_path / "capture"))
    profile["screening_candidates"] = settings
    root = tmp_path / "queue"
    coordinator._root(root)
    calls = []
    def build(previous, next_id, directory):
        calls.append(next_id)
        facts = build_planning_inputs(catalog=catalog, profile=profile, capture_root=tmp_path / "capture",
            source=ready["source"], inputs_dir=directory)
        if previous is None:
            refresh.retain_planning_bindings(root, next_id, capabilities=facts["capabilities"],
                expectations=facts["refresh_expectations"])
        return build_screening_plan(catalog, facts["scene"], campaign_id=next_id,
            capabilities=facts["capabilities"], cost_quotes=facts["cost_quotes"])
    initial = build(None, "campaign-1", root / "inputs/campaign-1")
    assert initial["candidates"][0]["status"] == "unavailable"
    assert initial["candidates"][1]["status"] == "selected"
    source_authority(tmp_path, monkeypatch, initial)
    coordinator.enqueue_screening_campaign(initial, queue_root=root)
    other = plan(campaign_id="unrelated-site", site_id="site-unrelated")
    coordinator.enqueue_screening_campaign(other, queue_root=root)
    before = deepcopy(coordinator._ledger(root)["campaigns"])
    old_bytes = (root / "plans/campaign-1.json").read_bytes()
    missing.write_text("second-fake-private-credential")
    missing.chmod(0o600)
    outcome = refresh.maybe_refresh_campaign(queue_root=root, base_campaign_id="campaign-1", build=build)
    new_id = outcome["campaign_id"]
    assert outcome["refresh_status"] == "refreshed" and len(calls) == 2
    state = coordinator._ledger(root)["campaigns"]
    assert state[new_id]["candidates"]["hosted_alpha"]["status"] == "preparing"
    assert state[new_id]["candidates"]["hosted_beta"]["status"] == "blocked"
    assert state["campaign-1"]["candidates"]["hosted_beta"] == before["campaign-1"]["candidates"]["hosted_beta"]
    assert state["unrelated-site"] == before["unrelated-site"]
    assert (root / "plans/campaign-1.json").read_bytes() == old_bytes
    for path in [*(root / "plans").glob("*.json"), *(root / "planning_bindings").glob("*.json"),
                 root / coordinator.LEDGER_FILENAME]:
        assert TOKEN not in path.read_text() and "second-fake-private-credential" not in path.read_text()
    for _ in range(3):
        refresh.maybe_refresh_campaign(queue_root=root, base_campaign_id="campaign-1", build=build)
    assert len(calls) == 2


def test_legacy_missing_runtime_cannot_acquire_new_execution_identity_from_a_label(tmp_path, monkeypatch, clock):
    selected = plan()
    unavailable = deepcopy(selected)
    unavailable["candidates"] = [{key: value for key, value in selected["candidates"][0].items()
                                  if key in {"candidate_id", "configuration_digest", "display_name"}}]
    unavailable["candidates"][0].update(status="unavailable", blockers=["runtime_candidate_binding_missing"])
    unavailable["maximum_planned_cost_usd"] = "0.00"
    seal(unavailable, "plan_digest")
    source_authority(tmp_path, monkeypatch, unavailable)
    coordinator.enqueue_screening_campaign(unavailable, queue_root=tmp_path)
    def build(_previous, next_id, _directory):
        fresh = deepcopy(selected)
        fresh["campaign_id"] = next_id
        return seal(fresh, "plan_digest")
    with pytest.raises(ValueError, match="new_configuration_identity_required"):
        refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    assert len(coordinator._ledger(tmp_path)["campaigns"]) == 1
    assert not list((tmp_path / "attempts").iterdir())


def test_source_worker_recovers_one_quote_after_twenty_five_hours_and_requires_new_ack(tmp_path, monkeypatch, clock):
    from tests import test_beta_policy_screening_worker as fixture
    monkeypatch.setattr(fixture, "datetime", clock)
    world = fixture.prepared_source_queue(tmp_path, monkeypatch)
    kwargs = {"queue_root": world["screening_queue"], "packet_queue_root": world["packet_queue"]}
    [initial] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    root = world["screening_queue"]
    value = coordinator._plan(root, initial["campaign_id"])
    # This test follows one affected candidate; keep the other original
    # candidate pending without fabricating a completion or a paid outcome.
    backend = CapacityBackend()
    worker.deliver_publications(queue_root=root, transport=lambda _: None)
    coordinator.process_screening_queue(queue_root=root, executor=backend, max_launches=1)
    assert backend.launches == 0
    original = (root / "plans" / (initial["campaign_id"] + ".json")).read_bytes()
    source_authority(tmp_path, monkeypatch, value)
    clock.instant += timedelta(hours=25)
    [outcome] = worker.enqueue_prepared_scenes(**kwargs)["sources"]
    assert outcome["refresh_status"] == "refreshed" and len(world["planning_calls"]) == 2
    backend.busy = False
    coordinator.process_screening_queue(queue_root=root, executor=backend, max_launches=1)
    assert backend.launches == 0  # An old website acknowledgment cannot start this generation.
    worker.deliver_publications(queue_root=root, transport=lambda _: None)
    coordinator.process_screening_queue(queue_root=root, executor=backend, max_launches=1)
    assert backend.launches == 1
    for _ in range(3):
        worker.enqueue_prepared_scenes(**kwargs)
    assert len(world["planning_calls"]) == 2
    assert (root / "plans" / (initial["campaign_id"] + ".json")).read_bytes() == original
    assert len(list((root / "plans").glob("*.json"))) == 2


@pytest.mark.parametrize("artifact", ["allocator_process.json", "invocation/manifest.json", "invocation/provider_create_intent.json"])
def test_preparing_label_cannot_release_or_refresh_an_unproven_invocation(tmp_path, monkeypatch, clock, artifact):
    value = plan()
    source_authority(tmp_path, monkeypatch, value)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    attempt = coordinator._reserve(tmp_path, value, value["candidates"][0])
    evidence = tmp_path / "attempts" / attempt["attempt_id"] / artifact
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}")
    clock.instant += timedelta(hours=25)
    build, builds = builder(clock)
    refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    backend = Backend()
    process(tmp_path, backend)
    retained = row(tmp_path)
    assert retained["status"] == "reconciliation_required" and retained["attempt"]["reservation_status"] == "reserved"
    assert coordinator._budget(coordinator._ledger(tmp_path), value["site_id"])["reserved_usd"] == "4"
    assert builds == [] and backend.preparations == backend.launches == 0
    assert evidence.read_text() == "{}"


def test_two_generations_keep_unknown_result_joined_to_original_attempt(tmp_path, monkeypatch, clock):
    value = plan(count=2)
    source_authority(tmp_path, monkeypatch, value)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    backend = Backend(OSError("unknown allocation"))
    process(tmp_path, backend, max_launches=1)
    original = deepcopy(row(tmp_path)["attempt"])
    build, builds = builder(clock)
    generations = []
    for _ in range(2):
        clock.instant += timedelta(hours=25)
        result = refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
        generations.append(result["campaign_id"])
        assert row(tmp_path, result["campaign_id"])["prior_campaign_id"] == "campaign-1"
        assert row(tmp_path)["attempt"] == original
        assert coordinator._budget(coordinator._ledger(tmp_path), value["site_id"])["reserved_usd"] == "8"
    assert len(set(generations)) == len(builds) == 2 and backend.launches == 1
    coordinator._observe(tmp_path, value, value["candidates"][0], {"status": "completed",
        "provider_allocation_performed": True, "provider_zero_verified": True, "result": native_result(value)})
    refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    adopted = row(tmp_path, generations[-1])
    assert adopted["status"] == "completed" and adopted["adopted_from_campaign_id"] == "campaign-1"
    assert "attempt" not in adopted and len(builds) == 2


@pytest.mark.parametrize("fault", ["naive_backoff", "wrong_cached_campaign"])
def test_refresh_metadata_cannot_change_its_generation_binding(tmp_path, monkeypatch, clock, fault):
    value = plan()
    source_authority(tmp_path, monkeypatch, value)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    clock.instant += timedelta(hours=25)
    build, _ = builder(clock)
    checks = tmp_path / "refresh_checks"
    checks.mkdir()
    if fault == "naive_backoff":
        (checks / "campaign-1.json").write_text(json.dumps({"retry_not_before": "2100-01-01T00:00:00"}))
    else:
        identity = canonical_digest({"base_campaign_id": "campaign-1", "supersedes": value["plan_digest"], "generation": 1})[7:39]
        next_id = "screening-refresh-" + identity
        cached = build(value, "different-generation", tmp_path / "unused")
        (checks / (next_id + ".plan.json")).write_text(json.dumps(cached))
    with pytest.raises(ValueError, match="refresh_backoff_invalid|refresh_generation_invalid"):
        refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    assert len(coordinator._ledger(tmp_path)["campaigns"]) == 1 and "attempt" not in row(tmp_path)


@pytest.mark.parametrize("expire_original_first", [False, True])
def test_later_original_quote_expiry_refreshes_from_actual_unspent_origin(
        tmp_path, monkeypatch, clock, expire_original_first):
    value = plan(count=2)
    beta = value["candidates"][1]
    beta["cost_quote"]["valid_until"] = (clock.instant + timedelta(days=1)).isoformat()
    seal(beta["cost_quote"], "quote_digest")
    seal(value, "plan_digest")
    source_authority(tmp_path, monkeypatch, value)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    original_bytes = (tmp_path / "plans/campaign-1.json").read_bytes()
    backend = CapacityBackend()
    process(tmp_path, backend)
    original_beta = deepcopy(row(tmp_path, candidate_id="candidate-001")["attempt"])
    build, builds = builder(clock)
    clock.instant += timedelta(hours=2)
    first = refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    first_id = first["campaign_id"]
    assert row(tmp_path, first_id, "candidate-001")["prior_campaign_id"] == "campaign-1"
    assert row(tmp_path, candidate_id="candidate-001")["attempt"] == original_beta
    assert coordinator._budget(coordinator._ledger(tmp_path), value["site_id"])["reserved_usd"] == "8"
    first_bytes = (tmp_path / "plans" / (first_id + ".json")).read_bytes()
    clock.instant += timedelta(hours=25)
    if expire_original_first:
        process(tmp_path, backend)
        assert row(tmp_path, candidate_id="candidate-001")["status"] == "deferred"
    commitments = []
    save = coordinator._save
    def observe_commit(root, ledger):
        commitments.append(coordinator._budget(ledger, value["site_id"])["reserved_usd"])
        return save(root, ledger)
    with monkeypatch.context() as scoped:
        scoped.setattr(coordinator, "_save", observe_commit)
        second = refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    assert commitments == ["8"]
    second_id = second["campaign_id"]
    assert second_id != first_id and len(builds) == 2
    assert row(tmp_path, second_id, "candidate-001")["status"] == "preparing"
    assert row(tmp_path, second_id, "candidate-001")["refreshed_from_campaign_id"] == "campaign-1"
    retained = row(tmp_path, candidate_id="candidate-001")
    assert retained["attempt"]["attempt_id"] == original_beta["attempt_id"]
    assert retained["attempt"]["reservation_status"] == "released" and retained["superseded_by"] == second_id
    assert coordinator._budget(coordinator._ledger(tmp_path), value["site_id"])["reserved_usd"] == "8"
    assert (tmp_path / "plans/campaign-1.json").read_bytes() == original_bytes
    assert (tmp_path / "plans" / (first_id + ".json")).read_bytes() == first_bytes
    backend.busy = False
    backend.observation = {"status": "running", "provider_allocation_performed": True}
    process(tmp_path, backend, max_launches=2)
    assert backend.launches == 2
    for _ in range(3):
        refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
        process(tmp_path, backend, max_launches=2)
    assert backend.launches == len(builds) == 2
    assert coordinator._budget(coordinator._ledger(tmp_path), value["site_id"])["reserved_usd"] == "8"


def test_origin_that_launched_after_partial_refresh_never_becomes_new_retry_authority(tmp_path, monkeypatch, clock):
    value = plan(count=2)
    value["candidates"][1]["cost_quote"]["valid_until"] = (clock.instant + timedelta(days=1)).isoformat()
    seal(value["candidates"][1]["cost_quote"], "quote_digest")
    seal(value, "plan_digest")
    source_authority(tmp_path, monkeypatch, value)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    capacity = CapacityBackend()
    process(tmp_path, capacity)
    build, _ = builder(clock)
    clock.instant += timedelta(hours=2)
    refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    unknown = Backend(OSError("original provider response is unknown"))
    coordinator._advance_candidate(tmp_path, value, value["candidates"][1], executor=unknown, require_publication_ack=False)
    original = deepcopy(row(tmp_path, candidate_id="candidate-001"))
    assert original["status"] == "reconciliation_required" and unknown.launches == 1
    clock.instant += timedelta(hours=25)
    outcome = refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    assert row(tmp_path, outcome["campaign_id"], "candidate-001")["prior_campaign_id"] == "campaign-1"
    assert row(tmp_path, candidate_id="candidate-001") == original
    assert coordinator._budget(coordinator._ledger(tmp_path), value["site_id"])["reserved_usd"] == "8"
    assert unknown.launches == 1


def test_availability_only_generation_does_not_strand_later_original_quote_expiry(tmp_path, monkeypatch, clock):
    ready = plan(count=2)
    initial = deepcopy(ready)
    initial["candidates"][0] = {key: value for key, value in initial["candidates"][0].items()
                              if key in {"candidate_id", "configuration_digest", "display_name"}}
    initial["candidates"][0].update(status="unavailable", blockers=["credential_unavailable:PINNED_POLICY_CREDENTIAL"])
    initial["maximum_planned_cost_usd"] = "4.00"
    initial["candidates"][1]["cost_quote"]["valid_until"] = (clock.instant + timedelta(days=1)).isoformat()
    seal(initial["candidates"][1]["cost_quote"], "quote_digest")
    seal(initial, "plan_digest")
    source_authority(tmp_path, monkeypatch, initial)
    coordinator.enqueue_screening_campaign(initial, queue_root=tmp_path)
    refresh.retain_planning_bindings(tmp_path, "campaign-1", capabilities={"candidate_bindings": {
        candidate["candidate_id"]: {"execution_spec": candidate["execution_spec"]} for candidate in ready["candidates"]}})
    capacity = CapacityBackend()
    process(tmp_path, capacity)
    rebuild, builds = builder(clock)
    def build(_previous, next_id, directory):
        return rebuild(ready, next_id, directory)
    first = refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    assert row(tmp_path, first["campaign_id"], "candidate-001")["prior_campaign_id"] == "campaign-1"
    clock.instant += timedelta(hours=25)
    process(tmp_path, capacity)
    second = refresh.maybe_refresh_campaign(queue_root=tmp_path, base_campaign_id="campaign-1", build=build)
    assert second["campaign_id"] != first["campaign_id"] and len(builds) == 2
    assert row(tmp_path, second["campaign_id"], "candidate-001")["status"] == "preparing"
    assert row(tmp_path, second["campaign_id"], "candidate-001")["refreshed_from_campaign_id"] == "campaign-1"
    assert coordinator._budget(coordinator._ledger(tmp_path), initial["site_id"])["reserved_usd"] == "8"
