"""AUT-08 local invocation power-loss boundaries, with no provider calls."""

import json
import threading
import time

import pytest

from blueprint_pipeline import allocator_invocation as jobs


def retained(tmp_path):
    authority = {"authority_digest": "sha256:" + "a" * 64, "resource_name": "exact-owned-resource"}
    bundle = {"bundle_sha256": "sha256:" + "b" * 64}
    path = jobs.retain_invocation(attempt_dir=tmp_path, binding={
        **authority, **bundle, "provider": "vast", "prepared_digest": "sha256:" + "c" * 64},
        argv=["gpu-canary", "--execute"])
    return path, authority, bundle


def test_pre_spawn_crash_is_fenced_and_delayed_spawn_cannot_create(tmp_path):
    path, _, _ = retained(tmp_path)
    proof = jobs.fence_unstarted(path)
    assert proof["status"] == "fenced_unspent"
    assert proof["provider_create_prevented"] is True
    assert jobs.run_invocation(path, allocator=lambda args: pytest.fail("fenced invocation ran")) == 2
    assert jobs.fence_unstarted(path) == proof


def test_live_lock_blocks_fence_without_using_pid_or_global_inventory(tmp_path):
    path, _, _ = retained(tmp_path)
    started, finish = threading.Event(), threading.Event()

    def allocator(args):
        started.set()
        assert finish.wait(2)
        return 0

    thread = threading.Thread(target=lambda: jobs.run_invocation(path, allocator=allocator))
    thread.start()
    try:
        assert started.wait(2)
        assert jobs.fence_unstarted(path)["status"] == "running"
    finally:
        finish.set()
        thread.join(2)
    assert jobs.fence_unstarted(path)["status"] == "fenced_unspent"


def test_revocation_between_wrapper_start_and_create_refuses_provider_effect(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_worker as worker
    authority = {"authority_digest": "sha256:" + "a" * 64, "resource_name": "exact-owned-resource"}
    bundle = {"bundle_sha256": "sha256:" + "b" * 64}
    path = jobs.retain_invocation(attempt_dir=tmp_path, binding={**authority, **bundle, "provider": "vast"},
        argv=["gpu-canary"], control={"source": {}, "site_id": "site-one", "prepared_digest": "sha256:" + "c" * 64,
            "initial_observed_at_epoch": time.time(), "execution_deadline_epoch": None})
    observations = iter([{}, ValueError("beta_screening_current_capture_consent_required")])

    def current(*args, **kwargs):
        outcome = next(observations)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(worker, "require_current_source", current)
    with pytest.raises(ValueError, match="unproven_before_create"):
        jobs.run_invocation(path, allocator=lambda a: jobs.before_provider_create(authority=authority, bundle=bundle))
    assert not (path.parent / "provider_create_intent.json").exists()
    assert (path.parent / "stop_requested.json").exists()


def test_create_response_loss_preserves_exact_identity_and_never_restarts(tmp_path):
    path, authority, bundle = retained(tmp_path)
    calls = []

    def allocator(args):
        jobs.before_provider_create(authority=authority, bundle=bundle)
        calls.append("provider_create")
        raise RuntimeError("lost response after create")

    with pytest.raises(RuntimeError):
        jobs.run_invocation(path, allocator=allocator)
    result = jobs.fence_unstarted(path)
    assert result["status"] == "reconciliation_required"
    assert result["next_action"] == "reconcile_exact_provider_identity"
    assert jobs.run_invocation(path, allocator=lambda args: calls.append("duplicate") or 0) == 2
    assert calls == ["provider_create"]
    intention = json.loads((path.parent / "provider_create_intent.json").read_text())
    assert intention["resource_name"] == authority["resource_name"]
    assert intention["binding"]["bundle_sha256"] == bundle["bundle_sha256"]


def test_wrong_authority_or_bundle_refuses_before_create(tmp_path):
    path, authority, bundle = retained(tmp_path)

    def allocator(args):
        with pytest.raises(ValueError, match="create_binding_invalid"):
            jobs.before_provider_create(authority={**authority, "resource_name": "unrelated"}, bundle=bundle)
        with pytest.raises(ValueError, match="create_binding_invalid"):
            jobs.before_provider_create(authority=authority, bundle={"bundle_sha256": "changed"})
        return 2

    jobs.run_invocation(path, allocator=allocator)
    assert not (path.parent / "provider_create_intent.json").exists()
    assert jobs.fence_unstarted(path)["status"] == "fenced_unspent"


def test_second_create_boundary_is_refused_even_with_same_identity(tmp_path):
    path, authority, bundle = retained(tmp_path)

    def allocator(args):
        jobs.before_provider_create(authority=authority, bundle=bundle)
        with pytest.raises(FileExistsError):
            jobs.before_provider_create(authority=authority, bundle=bundle)
        return 0

    assert jobs.run_invocation(path, allocator=allocator) == 0


def test_completed_result_prevents_any_new_start(tmp_path):
    path, _, _ = retained(tmp_path)
    (tmp_path / "allocator_result.json").write_text("{}")
    assert jobs.run_invocation(path, allocator=lambda args: pytest.fail("completed result restarted")) == 2


def test_changed_manifest_is_refused_before_job_claim(tmp_path):
    path, _, _ = retained(tmp_path)
    value = json.loads(path.read_text())
    value["argv"].append("--different-configuration")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="manifest_invalid"):
        jobs.fence_unstarted(path)


def test_job_writable_invocation_namespace_cannot_supply_fence_proof(tmp_path):
    path, _, _ = retained(tmp_path)
    path.parent.chmod(0o777)
    with pytest.raises(ValueError, match="manifest_invalid"):
        jobs.fence_unstarted(path)


def test_manifest_fsync_before_launch_intent_is_recoverable(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_launch as launch
    path, _, _ = retained(tmp_path)
    prepared = {"prepared_digest": jobs.manifest(path)["binding"]["prepared_digest"]}
    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    result = launch.reconcile_candidate(plan={}, candidate={}, attempt_dir=tmp_path, prepared=prepared)
    assert result["retryable_unstarted_refusal"] is True
    assert jobs.run_invocation(path, allocator=lambda a: pytest.fail("fenced late start")) == 2


def test_unverified_no_create_receipt_never_releases_site_reservation(tmp_path):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from tests.test_beta_policy_screening_coordinator import Backend, plan, publication
    backend = Backend({"status": "blocked", "no_allocation_proven": True,
        "provider_allocation_performed": False, "provider_zero_verified": True,
        "retryable_unstarted_refusal": True, "no_policy_api_calls_proven": True,
        "unstarted_refusal_receipt": {"path": "/forged/no_create_fence.json"}})
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend, require_publication_ack=False)
    assert publication(tmp_path)["status"] == "reconciliation_required"
    assert publication(tmp_path)["budget"]["reserved_usd"] == "4"


def test_standalone_provider_gate_keeps_existing_authority(monkeypatch):
    monkeypatch.delenv(jobs.ENV, raising=False)
    jobs.before_provider_create(authority={}, bundle={})


def test_coordinator_transfers_original_reservation_only_after_exact_fence(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from blueprint_pipeline import beta_policy_screening_launch as launch
    from tests.test_beta_policy_screening_coordinator import Backend, plan, publication

    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)

    class CrashBackend(Backend):
        def prepare_candidate(self, **kwargs):
            return {"prepared_digest": "sha256:" + "c" * 64}

        def launch_candidate(self, **kwargs):
            self.launches += 1
            directory = kwargs["attempt_dir"]
            path = jobs.retain_invocation(attempt_dir=directory,
                binding={"prepared_digest": kwargs["prepared"]["prepared_digest"]}, argv=["gpu-canary"])
            launch._write(directory / "allocator_launch_intent.json", {
                "prepared_digest": kwargs["prepared"]["prepared_digest"], "invocation_manifest": str(path)})
            raise RuntimeError("power loss after intent fsync")

        def reconcile_candidate(self, **kwargs):
            return launch.reconcile_candidate(**kwargs)

    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = CrashBackend()
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend, require_publication_ack=False)
    assert publication(tmp_path)["budget"]["reserved_usd"] == "4"
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend, require_publication_ack=False)
    state = coordinator._ledger(tmp_path)["campaigns"]["campaign-1"]["candidates"]["candidate-000"]
    assert state["status"] == "preparing" and state["attempt"]["generation"] == 1
    assert state["attempt_history"][0]["reservation_status"] == "released"
    assert state["attempt_history"][0]["no_policy_api_calls_proven"] is True
    assert publication(tmp_path)["budget"]["reserved_usd"] == "4"
    assert publication(tmp_path)["budget"]["spent_upper_bound_usd"] == "0"
    assert backend.launches == 1
