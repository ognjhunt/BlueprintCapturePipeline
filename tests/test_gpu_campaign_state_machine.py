from __future__ import annotations

import json
import fcntl
import math
import threading
import time
from datetime import datetime, timezone

import pytest

from blueprint_pipeline.gpu_campaign_state_machine import (
    CampaignBlocked,
    CampaignConfig,
    CampaignMachine,
    validate_same_allocation_canary_handoff,
    validate_preloaded_image_evidence,
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
        if stage == "image_ready":
            return {
                "status": "passed",
                "image_digest": config["image_digest"],
                "local_digest_inspect_passed": True,
                "digest_already_local_at_allocation": True,
                "cold_pull_performed_during_campaign": False,
                "host_image_id": config["image_residency_evidence"]["host_image_id"],
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
        image_total_compressed_bytes=47_101_357_226,
        image_largest_layer_bytes=14_083_497_680,
        image_residency_evidence={
            "schema_version": "preloaded_worker_image.v1",
            "source_sha": "5" * 40,
            "image_digest": "sha256:" + "7" * 64,
            "allocation_key": "blueprint-g4",
            "host_image_id": "g4-host-image-immutable-1",
            "image_present_before_allocation": True,
            "local_digest_inspect_passed": True,
            "runtime_health_preflight_passed": True,
            "cold_pull_required_during_campaign": False,
            "host_self_test_sha256": "a" * 64,
            "runtime_health_sha256": "b" * 64,
        },
    )
    values.update(overrides)
    return CampaignConfig(**values)


def test_large_image_requires_exact_preallocation_residency_evidence(tmp_path):
    with pytest.raises(CampaignBlocked, match="preload_evidence_missing"):
        CampaignMachine(
            config=config(image_residency_evidence=None),
            adapter=FakeProvider(),
            state_dir=tmp_path,
        ).run()


def test_campaign_requires_at_least_one_full_episode_seed_before_allocation(tmp_path):
    provider = FakeProvider()

    with pytest.raises(CampaignBlocked, match="campaign_episode_seeds_missing"):
        CampaignMachine(
            config=config(episode_seeds=()),
            adapter=provider,
            state_dir=tmp_path,
        ).run()

    assert not any(call[0] == "allocate" for call in provider.calls)


@pytest.mark.parametrize(
    ("overrides", "blocker"),
    [
        ({"prior_exposure_usd": -1}, "campaign_prior_exposure_invalid"),
        ({"spend_authorization_usd": -1}, "campaign_spend_authorization_invalid"),
    ],
)
def test_campaign_rejects_negative_budget_inputs_before_allocation(tmp_path, overrides, blocker):
    provider = FakeProvider()

    with pytest.raises(CampaignBlocked, match=blocker):
        CampaignMachine(
            config=config(**overrides),
            adapter=provider,
            state_dir=tmp_path,
        ).run()

    assert not any(call[0] == "allocate" for call in provider.calls)


def test_registry_availability_is_not_preloaded_image_evidence():
    cfg = config()
    blockers = validate_preloaded_image_evidence(
        cfg,
        {
            "schema_version": "preloaded_worker_image.v1",
            "source_sha": cfg.source_sha,
            "image_digest": cfg.image_digest,
            "allocation_key": cfg.allocation_key,
            "host_image_id": "host-1",
            "registry_manifest_resolves": True,
        },
    )
    assert "large_image_preload_image_present_before_allocation_not_proven" in blockers
    assert "large_image_cold_pull_still_required" in blockers


def test_image_ready_rejects_paid_cold_pull_and_tears_down(tmp_path):
    provider = FakeProvider()
    original = provider.run_stage

    def cold_pull(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[1] == "image_ready":
            result["digest_already_local_at_allocation"] = False
            result["cold_pull_performed_during_campaign"] = True
        return result

    provider.run_stage = cold_pull
    result = CampaignMachine(
        config=config(), adapter=provider, state_dir=tmp_path, teardown_owner="owner-1"
    ).run()
    assert result["status"] == "blocked"
    assert "large_image_paid_cold_pull_detected" in result["blockers"][0]
    assert ("terminate", "vm-1") in provider.calls


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


def test_allocation_rpc_is_bounded_by_paid_lifetime(tmp_path, monkeypatch):
    observed: list[tuple[float, str]] = []
    original = CampaignMachine._call_with_deadline

    def record_deadline(timeout_seconds, operation, *args):
        observed.append((timeout_seconds, operation.__name__))
        return original(timeout_seconds, operation, *args)

    monkeypatch.setattr(
        CampaignMachine,
        "_call_with_deadline",
        staticmethod(record_deadline),
    )
    result = CampaignMachine(
        config=config(max_provider_seconds=123),
        adapter=FakeProvider(),
        state_dir=tmp_path,
        teardown_owner="owner-1",
    ).run()

    assert result["status"] == "completed"
    assert (123.0, "allocate") in observed


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
    assert result["teardown"]["status"] == "blocked"
    assert result["teardown"]["billing_stopped"] is False
    assert "provider_final_inventory_not_zero" in result["blockers"]
    assert not any(call[0] == "allocate" for call in provider.calls)


def test_provider_native_id_is_checkpointed_and_torn_down(tmp_path):
    class NativeIdProvider(FakeProvider):
        def allocate(self, config):
            self.calls.append(("allocate", config["campaign_id"]))
            self.live = [{"id": "native-vm-1"}]
            return {"status": "completed", "id": "native-vm-1"}

    provider = NativeIdProvider(fail_stage="host_ready")
    result = CampaignMachine(
        config=config(),
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
    ).run()
    assert result["allocation_id"] == "native-vm-1"
    assert ("terminate", "native-vm-1") in provider.calls
    assert result["teardown"]["billing_stopped"] is True


def test_lost_allocate_response_deletes_inventory_discovered_allocation(tmp_path):
    class LostResponseProvider(FakeProvider):
        def allocate(self, config):
            self.calls.append(("allocate", config["campaign_id"]))
            self.live = [{"id": "created-but-response-lost"}]
            raise ConnectionError("create_response_lost")

    provider = LostResponseProvider()
    result = CampaignMachine(
        config=config(),
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
    ).run()

    assert result["status"] == "blocked"
    assert ("terminate", "created-but-response-lost") in provider.calls
    assert result["teardown"]["status"] == "passed"
    assert result["teardown"]["billing_stopped"] is True
    assert result["final_inventory"] == []


def test_missing_allocate_id_deletes_inventory_discovered_allocation(tmp_path):
    class MissingIdProvider(FakeProvider):
        def allocate(self, config):
            self.calls.append(("allocate", config["campaign_id"]))
            self.live = [{"allocation_id": "created-with-missing-response-id"}]
            return {"status": "completed"}

    provider = MissingIdProvider()
    result = CampaignMachine(
        config=config(),
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
    ).run()

    assert result["status"] == "blocked"
    assert ("terminate", "created-with-missing-response-id") in provider.calls
    assert result["teardown"]["billing_stopped"] is True


def test_ambiguous_create_empty_inventory_stays_retryable_until_discovered(tmp_path):
    class EventuallyVisibleProvider(FakeProvider):
        post_create_inventory_calls = 0

        def allocate(self, config):
            self.calls.append(("allocate", config["campaign_id"]))
            self.live = [{"id": "eventually-visible-vm"}]
            raise ConnectionError("create_response_lost")

        def inventory(self, allocation_key):
            self.calls.append(("inventory", allocation_key))
            if self.live:
                self.post_create_inventory_calls += 1
                if self.post_create_inventory_calls == 1:
                    return []
            return list(self.live)

    provider = EventuallyVisibleProvider()
    first = CampaignMachine(
        config=config(),
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
    ).run()
    assert first["status"] == "blocked"
    assert first["teardown"]["status"] == "blocked"
    assert first["teardown"]["billing_stopped"] is False

    second = CampaignMachine(
        config=config(),
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
    ).run()
    assert ("terminate", "eventually-visible-vm") in provider.calls
    assert second["teardown"]["status"] == "passed"
    assert second["teardown"]["billing_stopped"] is True


def test_budget_is_admitted_on_worst_case_not_expected_spend(tmp_path):
    with pytest.raises(CampaignBlocked, match="maximum_exceeds"):
        CampaignMachine(
            config=config(prior_exposure_usd=19),
            adapter=FakeProvider(),
            state_dir=tmp_path,
        ).run()


@pytest.mark.parametrize(
    ("overrides", "blocker"),
    [
        ({"source_sha": "z" * 40}, "source_sha_invalid"),
        ({"source_sha": "A" * 40}, "source_sha_invalid"),
        ({"image_digest": "sha256:" + "z" * 64}, "image_digest_invalid"),
        ({"image_digest": "sha256:" + "A" * 64}, "image_digest_invalid"),
    ],
)
def test_runtime_identity_validation_matches_lowercase_hex_schema(overrides, blocker):
    assert blocker in config(**overrides).validate()


def test_teardown_ambiguity_remains_blocked(tmp_path):
    result = CampaignMachine(
        config=config(),
        adapter=FakeProvider(ambiguous_teardown=True),
        state_dir=tmp_path,
        teardown_owner="owner-1",
    ).run()
    assert result["status"] == "blocked"
    assert "provider_teardown_ambiguous" in result["blockers"]


def test_teardown_call_is_actually_interrupted_at_deadline():
    with pytest.raises(CampaignBlocked, match="teardown_deadline_exceeded"):
        CampaignMachine._call_with_deadline(0.01, time.sleep, 0.2)


def test_teardown_deadline_is_bounded_off_main_thread():
    outcome = []

    def invoke():
        try:
            CampaignMachine._call_with_deadline(0.01, time.sleep, 0.2)
        except CampaignBlocked as exc:
            outcome.append(str(exc))

    worker = threading.Thread(target=invoke)
    worker.start()
    worker.join(0.2)
    assert not worker.is_alive()
    assert outcome == ["campaign_teardown_deadline_exceeded"]


def test_teardown_timeout_is_checkpointed_as_billing_ambiguous(tmp_path, monkeypatch):
    provider = FakeProvider()
    machine = CampaignMachine(
        config=config(),
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
    )
    original = machine._call_with_deadline

    def timeout_on_terminate(timeout_seconds, operation, *args):
        if operation == provider.terminate:
            raise CampaignBlocked("campaign_teardown_deadline_exceeded")
        return original(timeout_seconds, operation, *args)

    monkeypatch.setattr(machine, "_call_with_deadline", timeout_on_terminate)
    result = machine.run()
    assert result["status"] == "blocked"
    assert result["teardown"]["status"] == "blocked"
    assert result["teardown"]["billing_stopped"] is False
    assert any("campaign_teardown_deadline_exceeded" in item for item in result["blockers"])


def test_teardown_exception_never_persists_false_completion_and_is_retried(tmp_path):
    class RaisingTeardownProvider(FakeProvider):
        should_raise = True

        def terminate(self, allocation_id):
            self.calls.append(("terminate", allocation_id))
            if self.should_raise:
                raise ConnectionError("provider_api_disconnected")
            self.live = []
            return {"status": "delete_requested"}

    provider = RaisingTeardownProvider()
    machine = CampaignMachine(
        config=config(), adapter=provider, state_dir=tmp_path, teardown_owner="owner-1"
    )
    first = machine.run()
    assert first["status"] == "blocked"
    assert first["teardown"]["status"] == "blocked"
    assert first["teardown"]["billing_stopped"] is False
    provider.should_raise = False
    retried = CampaignMachine(
        config=config(), adapter=provider, state_dir=tmp_path, teardown_owner="owner-1"
    ).run()
    assert retried["status"] == "blocked"
    assert retried["teardown"]["status"] == "passed"
    assert retried["teardown"]["billing_stopped"] is True
    assert provider.calls.count(("terminate", "vm-1")) == 2


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
    state["provider_started_at_epoch_seconds"] = 1000.0
    provider.live = [{"allocation_id": "vm-1"}]
    machine._checkpoint(state, "allocation", {"status": "completed", "allocation_id": "vm-1"})
    resumed = CampaignMachine(
        config=cfg, adapter=provider, state_dir=tmp_path, wall_clock=lambda: 1001.0
    )
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
        "teardown_owner": "owner-1",
        "provider_started_at_epoch_seconds": 1000.0,
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


def test_same_allocation_canary_adopts_inventory_without_allocating(tmp_path):
    cfg = config(
        reuse_validated_same_allocation_canary=True,
        canary_handoff={
            "schema_version": "same_allocation_canary_handoff.v1",
            "source_sha": "5" * 40,
            "image_digest": "sha256:" + "7" * 64,
            "allocation_key": "blueprint-g4",
            "allocation_id": "vm-1",
            "launch_nonce": "nonce-1",
            "teardown_owner": "owner-1",
            "provider_started_at_epoch_seconds": 1000.0,
            "runtime_health_passed": True,
            "review_media_valid": True,
            "allocation_still_owned": True,
            "teardown_requested": False,
        },
    )
    provider = FakeProvider(live=[{"allocation_id": "vm-1"}])
    result = CampaignMachine(
        config=cfg, adapter=provider, state_dir=tmp_path, wall_clock=lambda: 1001.0
    ).run()
    assert result["status"] == "completed"
    assert result["stage_results"]["allocation"]["adopted_canary_handoff"] is True
    assert result["stage_results"]["canary"]["reused_handoff"] is True
    assert not any(call[0] == "allocate" for call in provider.calls)


def test_same_allocation_canary_checkpoints_owned_id_before_inventory_rpc(tmp_path):
    cfg = config(
        reuse_validated_same_allocation_canary=True,
        canary_handoff={
            "schema_version": "same_allocation_canary_handoff.v1",
            "source_sha": "5" * 40,
            "image_digest": "sha256:" + "7" * 64,
            "allocation_key": "blueprint-g4",
            "allocation_id": "vm-1",
            "launch_nonce": "nonce-1",
            "teardown_owner": "owner-1",
            "provider_started_at_epoch_seconds": 1000.0,
            "runtime_health_passed": True,
            "review_media_valid": True,
            "allocation_still_owned": True,
            "teardown_requested": False,
        },
    )

    class CheckpointAwareProvider(FakeProvider):
        first_inventory_saw_checkpoint = False

        def inventory(self, allocation_key):
            if not self.calls:
                persisted = json.loads((tmp_path / "campaign_state.json").read_text())
                self.first_inventory_saw_checkpoint = persisted["allocation_id"] == "vm-1"
            return super().inventory(allocation_key)

    provider = CheckpointAwareProvider(live=[{"allocation_id": "vm-1"}])
    result = CampaignMachine(
        config=cfg,
        adapter=provider,
        state_dir=tmp_path,
        wall_clock=lambda: 1001.0,
    ).run()

    assert result["status"] == "completed"
    assert provider.first_inventory_saw_checkpoint is True


def test_same_allocation_canary_blocks_if_handoff_is_not_in_inventory(tmp_path):
    cfg = config(
        reuse_validated_same_allocation_canary=True,
        canary_handoff={
            "schema_version": "same_allocation_canary_handoff.v1",
            "source_sha": "5" * 40,
            "image_digest": "sha256:" + "7" * 64,
            "allocation_key": "blueprint-g4",
            "allocation_id": "vm-1",
            "launch_nonce": "nonce-1",
            "teardown_owner": "owner-1",
            "provider_started_at_epoch_seconds": 1000.0,
            "runtime_health_passed": True,
            "review_media_valid": True,
            "allocation_still_owned": True,
            "teardown_requested": False,
        },
    )
    provider = FakeProvider()
    result = CampaignMachine(
        config=cfg,
        adapter=provider,
        state_dir=tmp_path,
        wall_clock=lambda: 1001.0,
    ).run()
    assert result["status"] == "blocked"
    assert "same_allocation_handoff_inventory_mismatch" in result["blockers"]
    assert not any(call[0] == "allocate" for call in provider.calls)
    assert ("terminate", "vm-1") in provider.calls
    assert result["teardown"]["billing_stopped"] is True


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


def test_stage_deadline_is_capped_to_remaining_paid_lifetime(tmp_path):
    provider = FakeProvider()
    machine = CampaignMachine(
        config=config(max_provider_seconds=100),
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
        wall_clock=lambda: 1000.0,
    )
    state, _ = machine._load()
    state["allocation_id"] = "vm-1"
    state["provider_started_at_epoch_seconds"] = 950.0
    provider.live = [{"allocation_id": "vm-1"}]
    machine._checkpoint(state, "allocation", {"status": "completed", "allocation_id": "vm-1"})

    result = machine.run()

    assert result["status"] == "completed"
    assert ("host_ready", 50) in provider.calls
    assert result["stage_results"]["host_ready"]["authorized_deadline_seconds"] == 50


def test_resume_preserves_original_provider_lifetime_and_tears_down(tmp_path):
    cfg = config(max_provider_seconds=100)
    provider = FakeProvider()
    machine = CampaignMachine(
        config=cfg,
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
        wall_clock=lambda: 1000.0,
    )
    state, _ = machine._load()
    state["allocation_id"] = "vm-1"
    state["provider_started_at_epoch_seconds"] = 800.0
    provider.live = [{"allocation_id": "vm-1"}]
    machine._checkpoint(state, "allocation", {"status": "completed", "allocation_id": "vm-1"})

    result = CampaignMachine(
        config=cfg,
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
        wall_clock=lambda: 1000.0,
    ).run()

    assert result["status"] == "blocked"
    assert "campaign_provider_lifetime_exceeded" in result["blockers"]
    assert not any(call[0] == "host_ready" for call in provider.calls)
    assert ("terminate", "vm-1") in provider.calls


def test_resume_with_allocation_but_no_start_time_fails_closed(tmp_path):
    cfg = config()
    provider = FakeProvider(live=[{"allocation_id": "vm-1"}])
    machine = CampaignMachine(
        config=cfg, adapter=provider, state_dir=tmp_path, teardown_owner="owner-1"
    )
    state, _ = machine._load()
    state["allocation_id"] = "vm-1"
    machine._checkpoint(state, "allocation", {"status": "completed", "allocation_id": "vm-1"})

    result = machine.run()

    assert result["status"] == "blocked"
    assert "campaign_provider_start_time_missing" in result["blockers"]
    assert ("terminate", "vm-1") in provider.calls


def test_invalid_resume_config_still_tears_down_recorded_allocation(tmp_path):
    original = config()
    provider = FakeProvider(live=[{"allocation_id": "vm-1"}])
    machine = CampaignMachine(
        config=original,
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
    )
    state, _ = machine._load()
    state["allocation_id"] = "vm-1"
    state["provider_started_at_epoch_seconds"] = 1000.0
    machine._checkpoint(state, "allocation", {"status": "completed", "allocation_id": "vm-1"})

    result = CampaignMachine(
        config=config(source_sha="not-a-sha", allocation_key="changed-invalid-key"),
        adapter=provider,
        state_dir=tmp_path,
    ).run()

    assert result["status"] == "blocked"
    assert any(
        blocker.startswith("campaign_config_invalid:source_sha_invalid")
        for blocker in result["blockers"]
    )
    assert ("terminate", "vm-1") in provider.calls
    assert ("inventory", original.allocation_key) in provider.calls
    assert result["teardown"]["billing_stopped"] is True


def test_invalid_resume_config_tears_down_pending_create_without_checkpointed_id(
    tmp_path,
):
    original = config()
    provider = FakeProvider(live=[{"allocation_id": "vm-created-before-response-loss"}])
    machine = CampaignMachine(
        config=original,
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
    )
    state, _ = machine._load()
    state["allocation_mutation_pending"] = True
    state["provider_started_at_epoch_seconds"] = 1000.0
    machine._write(machine.state_path, state)

    result = CampaignMachine(
        config=config(source_sha="not-a-sha", allocation_key="changed-invalid-key"),
        adapter=provider,
        state_dir=tmp_path,
    ).run()

    assert result["status"] == "blocked"
    assert ("inventory", original.allocation_key) in provider.calls
    assert ("terminate", "vm-created-before-response-loss") in provider.calls
    assert result["teardown"]["billing_stopped"] is True
    assert result["allocation_mutation_pending"] is False


def test_changed_valid_resume_config_still_tears_down_checkpoint(tmp_path):
    original = config()
    provider = FakeProvider(live=[{"allocation_id": "vm-1"}])
    machine = CampaignMachine(
        config=original,
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
    )
    state, _ = machine._load()
    state["allocation_id"] = "vm-1"
    state["provider_started_at_epoch_seconds"] = 1000.0
    machine._checkpoint(state, "allocation", {"status": "completed", "allocation_id": "vm-1"})

    result = CampaignMachine(
        config=config(max_provider_seconds=3800),
        adapter=provider,
        state_dir=tmp_path,
    ).run()

    assert result["status"] == "blocked"
    assert "immutable_campaign_config_changed" in result["blockers"]
    assert ("terminate", "vm-1") in provider.calls
    assert result["teardown"]["billing_stopped"] is True


def test_invalid_config_does_not_repeat_proven_complete_teardown(tmp_path):
    provider = FakeProvider()
    original = CampaignMachine(
        config=config(),
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
    ).run()
    terminate_count = provider.calls.count(("terminate", "vm-1"))
    assert original["status"] == "completed"

    with pytest.raises(CampaignBlocked, match="source_sha_invalid"):
        CampaignMachine(
            config=config(source_sha="invalid"),
            adapter=provider,
            state_dir=tmp_path,
        ).run()

    assert provider.calls.count(("terminate", "vm-1")) == terminate_count


def test_corrupt_config_manifest_cannot_hide_checkpointed_allocation(tmp_path):
    provider = FakeProvider(live=[{"allocation_id": "vm-1"}])
    machine = CampaignMachine(
        config=config(),
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
    )
    state, _ = machine._load()
    state["allocation_id"] = "vm-1"
    state["provider_started_at_epoch_seconds"] = 1000.0
    machine._checkpoint(state, "allocation", {"status": "completed", "allocation_id": "vm-1"})
    (tmp_path / "immutable_config_manifest.json").write_text("{corrupt")

    result = CampaignMachine(
        config=config(source_sha="invalid"),
        adapter=provider,
        state_dir=tmp_path,
    ).run()

    assert result["status"] == "blocked"
    assert ("terminate", "vm-1") in provider.calls
    assert result["teardown"]["billing_stopped"] is True


def test_corrupt_config_manifest_with_valid_resumed_config_still_tears_down(tmp_path):
    provider = FakeProvider(live=[{"allocation_id": "vm-1"}])
    machine = CampaignMachine(
        config=config(),
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
    )
    state, _ = machine._load()
    state["allocation_id"] = "vm-1"
    state["provider_started_at_epoch_seconds"] = 1000.0
    machine._checkpoint(state, "allocation", {"status": "completed", "allocation_id": "vm-1"})
    (tmp_path / "immutable_config_manifest.json").write_text("{corrupt")

    result = CampaignMachine(
        config=config(),
        adapter=provider,
        state_dir=tmp_path,
    ).run()

    assert result["status"] == "blocked"
    assert "immutable_campaign_config_manifest_unreadable" in result["blockers"]
    assert ("terminate", "vm-1") in provider.calls
    assert result["teardown"]["billing_stopped"] is True


def test_validation_exception_still_tears_down_checkpointed_allocation(tmp_path):
    provider = FakeProvider(live=[{"allocation_id": "vm-1"}])
    machine = CampaignMachine(
        config=config(),
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
    )
    state, _ = machine._load()
    state["allocation_id"] = "vm-1"
    state["provider_started_at_epoch_seconds"] = 1000.0
    machine._checkpoint(state, "allocation", {"status": "completed", "allocation_id": "vm-1"})
    malformed_deadlines = dict(config().stage_deadlines_seconds)
    malformed_deadlines["host_ready"] = "bad"  # type: ignore[assignment]

    result = CampaignMachine(
        config=config(stage_deadlines_seconds=malformed_deadlines),
        adapter=provider,
        state_dir=tmp_path,
    ).run()

    assert result["status"] == "blocked"
    assert any(
        blocker.startswith("campaign_config_invalid:campaign_config_validation_exception")
        for blocker in result["blockers"]
    )
    assert ("terminate", "vm-1") in provider.calls
    assert result["teardown"]["billing_stopped"] is True


def test_malformed_resumed_teardown_deadline_uses_recorded_deadline(tmp_path):
    provider = FakeProvider(live=[{"allocation_id": "vm-1"}])
    machine = CampaignMachine(
        config=config(),
        adapter=provider,
        state_dir=tmp_path,
        teardown_owner="owner-1",
    )
    state, _ = machine._load()
    state["allocation_id"] = "vm-1"
    state["provider_started_at_epoch_seconds"] = 1000.0
    machine._checkpoint(state, "allocation", {"status": "completed", "allocation_id": "vm-1"})
    malformed_deadlines = dict(config().stage_deadlines_seconds)
    malformed_deadlines["teardown"] = "bad"  # type: ignore[assignment]

    result = CampaignMachine(
        config=config(stage_deadlines_seconds=malformed_deadlines),
        adapter=provider,
        state_dir=tmp_path,
    ).run()

    assert ("terminate", "vm-1") in provider.calls
    assert result["teardown"]["billing_stopped"] is True
    assert result["teardown"]["deadline_seconds"] == 300


def test_second_controller_cannot_race_the_teardown_owner(tmp_path):
    lock_path = tmp_path / "campaign_controller.lock"
    lock_path.touch()
    with lock_path.open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(CampaignBlocked, match="already_running"):
            CampaignMachine(config=config(), adapter=FakeProvider(), state_dir=tmp_path).run()


@pytest.mark.parametrize(
    ("field", "blocker"),
    [
        ("hourly_rate_usd", "paid_runtime_bound_invalid"),
        ("prior_exposure_usd", "campaign_prior_exposure_invalid"),
        ("spend_authorization_usd", "campaign_spend_authorization_invalid"),
        ("image_total_compressed_bytes", "image_closure_size_missing"),
        ("image_largest_layer_bytes", "image_closure_size_missing"),
    ],
)
def test_nonfinite_budget_and_image_numbers_fail_closed(field, blocker):
    cfg = config(**{field: math.nan})
    assert blocker in cfg.validate()


def test_nonserializable_stage_result_cannot_preempt_teardown(tmp_path):
    class NonserializableProvider(FakeProvider):
        def run_stage(self, allocation_id, stage, *, deadline_seconds, config):
            result = super().run_stage(
                allocation_id, stage, deadline_seconds=deadline_seconds, config=config
            )
            if stage == "host_ready":
                result["observed_at"] = datetime.now(timezone.utc)
            return result

    provider = NonserializableProvider()
    result = CampaignMachine(config=config(), adapter=provider, state_dir=tmp_path).run()

    assert result["status"] == "blocked"
    assert ("terminate", "vm-1") in provider.calls
    assert result["teardown"]["billing_stopped"] is True
    assert any(
        item.startswith("state_persistence_exception:TypeError") for item in result["blockers"]
    )
