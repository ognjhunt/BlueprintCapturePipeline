from __future__ import annotations

import json
import fcntl

import pytest

from blueprint_pipeline.gpu_campaign_state_machine import (
    CampaignBlocked,
    CampaignConfig,
    CampaignMachine,
    validate_same_allocation_canary_handoff,
    validate_smoke_result,
)


class FakeProvider:
    provider_name = "fake-gcp"

    def __init__(self, *, fail_stage=None, live=None, ambiguous_teardown=False):
        self.fail_stage = fail_stage
        self.live = list(live or [])
        self.ambiguous_teardown = ambiguous_teardown
        self.calls = []

    def inventory(self, allocation_key):
        self.calls.append(("inventory", allocation_key))
        return list(self.live)

    def allocate(self, config):
        self.calls.append(("allocate", config["campaign_id"]))
        self.live = [{"allocation_id": "vm-1"}]
        return {"status": "completed", "allocation_id": "vm-1"}

    def run_stage(self, allocation_id, stage, *, deadline_seconds, config):
        self.calls.append((stage, deadline_seconds))
        if stage == self.fail_stage:
            return {"status": "blocked", "reason": "fixture_failure"}
        if stage == "smoke":
            return {
                "status": "passed",
                "command_return_code": 0,
                "simulator_steps": 3,
                "manifest_valid": True,
                "learned_policy_request_response_valid": True,
                "fresh_policy_conditioning_valid": True,
                "action_trace_nonempty": True,
                "real_task_executor_measurement": True,
                "artifact_output_present": True,
                "learned_policy_action_count": 3,
                "action_sources": ["groot_policy_server"] * 3,
            }
        return {"status": "passed", "stage": stage}

    def retrieve(self, allocation_id, config):
        self.calls.append(("retrieve", allocation_id))
        return {"status": "retrieved", "artifact_sha256": "a" * 64}

    def terminate(self, allocation_id):
        self.calls.append(("terminate", allocation_id))
        if not self.ambiguous_teardown:
            self.live = []
        return {"status": "delete_requested"}

    def inspect(self, allocation_id):
        self.calls.append(("inspect", allocation_id))
        return {"absent": not self.live, "http": 404 if not self.live else 200}


def config(**overrides):
    values = dict(
        campaign_id="campaign-1",
        allocation_key="blueprint-g4",
        source_sha="5" * 40,
        image_digest="sha256:" + "7" * 64,
        hourly_rate_usd=4.5,
        max_provider_seconds=3900,
        spend_authorization_usd=20,
        prior_exposure_usd=10,
    )
    values.update(overrides)
    return CampaignConfig(**values)


def test_full_run_checkpoints_and_tears_down(tmp_path):
    provider = FakeProvider()
    result = CampaignMachine(
        config=config(), adapter=provider, state_dir=tmp_path, teardown_owner="owner-1"
    ).run()
    assert result["status"] == "completed"
    assert result["teardown"]["status"] == "passed"
    assert result["final_inventory"] == []
    assert result["completed_stages"][-1] == "artifact_retrieval"
    manifest = json.loads((tmp_path / "immutable_config_manifest.json").read_text())
    assert manifest["config_sha256"]


def test_smoke_failure_blocks_episodes_and_still_tears_down(tmp_path):
    provider = FakeProvider(fail_stage="smoke")
    result = CampaignMachine(
        config=config(), adapter=provider, state_dir=tmp_path, teardown_owner="owner-1"
    ).run()
    assert result["status"] == "blocked"
    assert not any(call[0] == "episodes" for call in provider.calls)
    assert ("terminate", "vm-1") in provider.calls


def test_duplicate_allocation_is_refused_before_mutation(tmp_path):
    provider = FakeProvider(live=[{"allocation_id": "already-live"}])
    result = CampaignMachine(
        config=config(), adapter=provider, state_dir=tmp_path, teardown_owner="owner-1"
    ).run()
    assert result["status"] == "blocked"
    assert "duplicate_paid_allocation_detected" in result["blockers"]
    assert not any(call[0] == "allocate" for call in provider.calls)


def test_budget_is_admitted_on_worst_case_not_expected_spend(tmp_path):
    with pytest.raises(CampaignBlocked, match="maximum_exceeds"):
        CampaignMachine(
            config=config(prior_exposure_usd=19),
            adapter=FakeProvider(),
            state_dir=tmp_path,
        ).run()


def test_teardown_ambiguity_remains_blocked(tmp_path):
    result = CampaignMachine(
        config=config(),
        adapter=FakeProvider(ambiguous_teardown=True),
        state_dir=tmp_path,
        teardown_owner="owner-1",
    ).run()
    assert result["status"] == "blocked"
    assert "provider_teardown_ambiguous" in result["blockers"]


def test_resume_rejects_different_teardown_owner(tmp_path):
    first = CampaignMachine(
        config=config(), adapter=FakeProvider(), state_dir=tmp_path, teardown_owner="owner-1"
    )
    first.run()
    with pytest.raises(CampaignBlocked, match="owned_by_another"):
        CampaignMachine(
            config=config(), adapter=FakeProvider(), state_dir=tmp_path, teardown_owner="owner-2"
        ).run()


def test_running_checkpoint_auto_adopts_recorded_teardown_owner_and_resumes(tmp_path):
    cfg = config()
    provider = FakeProvider()
    machine = CampaignMachine(
        config=cfg, adapter=provider, state_dir=tmp_path, teardown_owner="owner-1"
    )
    state, _ = machine._load()
    state["allocation_id"] = "vm-1"
    provider.live = [{"allocation_id": "vm-1"}]
    machine._checkpoint(state, "allocation", {"status": "completed", "allocation_id": "vm-1"})
    resumed = CampaignMachine(config=cfg, adapter=provider, state_dir=tmp_path)
    result = resumed.run()
    assert resumed.teardown_owner == "owner-1"
    assert result["status"] == "completed"
    assert not any(call[0] == "allocate" for call in provider.calls)


def test_same_allocation_canary_handoff_schema():
    cfg = config(reuse_validated_same_allocation_canary=True)
    evidence = {
        "schema_version": "same_allocation_canary_handoff.v1",
        "source_sha": cfg.source_sha,
        "image_digest": cfg.image_digest,
        "allocation_key": cfg.allocation_key,
        "allocation_id": "vm-1",
        "launch_nonce": "nonce-1",
        "runtime_health_passed": True,
        "review_media_valid": True,
        "allocation_still_owned": True,
        "teardown_requested": False,
    }
    assert validate_same_allocation_canary_handoff(cfg, evidence) == []
    evidence["review_media_valid"] = False
    assert "same_allocation_canary_review_media_not_valid" in (
        validate_same_allocation_canary_handoff(cfg, evidence)
    )


def test_smoke_evidence_rejects_surrogate_actions_and_missing_steps():
    blockers = validate_smoke_result(
        {
            "status": "passed",
            "command_return_code": 0,
            "simulator_steps": 2,
            "manifest_valid": True,
            "learned_policy_request_response_valid": True,
            "fresh_policy_conditioning_valid": True,
            "action_trace_nonempty": True,
            "real_task_executor_measurement": True,
            "artifact_output_present": True,
            "learned_policy_action_count": 3,
            "action_sources": ["surrogate_fixture"] * 3,
        }
    )
    assert "smoke_real_simulator_steps_below_three" in blockers
    assert "smoke_action_sources_not_real" in blockers


def test_stage_deadline_is_enforced_even_if_adapter_reports_passed(tmp_path):
    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    provider = FakeProvider()
    original = provider.run_stage

    def slow_stage(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.now += 601
        return result

    provider.run_stage = slow_stage
    result = CampaignMachine(
        config=config(),
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
        clock=clock,
    ).run()
    assert result["status"] == "blocked"
    assert "campaign_stage_deadline_exceeded:host_ready" in result["blockers"]


def test_second_controller_cannot_race_the_teardown_owner(tmp_path):
    lock_path = tmp_path / "campaign_controller.lock"
    lock_path.touch()
    with lock_path.open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(CampaignBlocked, match="already_running"):
            CampaignMachine(config=config(), adapter=FakeProvider(), state_dir=tmp_path).run()
