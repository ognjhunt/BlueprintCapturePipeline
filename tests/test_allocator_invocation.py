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


def test_stop_after_the_create_intent_is_ignored_so_the_result_is_always_written(tmp_path):
    """Once the adapter has torn down, a SIGTERM must not kill the watchdog close or the result write."""
    import signal
    from blueprint_pipeline import beta_policy_screening_launch as launch
    path, authority, bundle = retained(tmp_path)
    before = signal.getsignal(signal.SIGTERM)
    written = []

    def allocator(args):
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler), "the default action would kill the allocator before its result"
        jobs.before_provider_create(authority=authority, bundle=bundle)
        # The adapter tore down and restored this handler; a stop lands now.
        handler(signal.SIGTERM, None)
        handler(signal.SIGTERM, None)
        launch._write(tmp_path / "allocator_result.json", {"status": "blocked"})
        written.append(True)
        return 0

    assert jobs.run_invocation(path, allocator=allocator) == 0
    assert written == [True] and (tmp_path / "allocator_result.json").is_file()
    assert signal.getsignal(signal.SIGTERM) == before


def test_first_stop_before_any_create_unwinds_once_and_the_job_is_fenced(tmp_path):
    import signal
    path, _, _ = retained(tmp_path)

    def allocator(args):
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler), "the default action kills without unwinding"
        try:
            handler(signal.SIGTERM, None)
        except KeyboardInterrupt:
            handler(signal.SIGTERM, None)  # A repeat cannot interrupt the unwinding.
            raise
        pytest.fail("a stop before any create must unwind the allocator")

    with pytest.raises(KeyboardInterrupt):
        jobs.run_invocation(path, allocator=allocator)
    assert not (path.parent / "provider_create_intent.json").exists()
    assert jobs.fence_unstarted(path)["status"] == "fenced_unspent"


def test_stop_signal_record_is_claimed_once(tmp_path):
    assert jobs.record_stop_signal(tmp_path, sender="allocator_monitor", reason="revoked", pid=7) is True
    assert jobs.record_stop_signal(tmp_path, sender="screening_reconcile", reason="revoked", pid=7) is False
    assert json.loads((tmp_path / jobs.STOP_SIGNAL_RECORD).read_text())["sender"] == "allocator_monitor"


def test_create_refused_on_a_read_without_answer_is_retried_not_blocked(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_launch as launch
    from blueprint_pipeline import beta_policy_screening_worker as worker
    authority = {"authority_digest": "sha256:" + "a" * 64, "resource_name": "exact-owned-resource"}
    bundle = {"bundle_sha256": "sha256:" + "b" * 64}
    prepared = {"prepared_digest": "sha256:" + "c" * 64}
    path = jobs.retain_invocation(attempt_dir=tmp_path, binding={**authority, **bundle, "provider": "vast", **prepared},
        argv=["gpu-canary"], control={"source": {}, "site_id": "site-one", **prepared,
            "initial_observed_at_epoch": time.time(), "execution_deadline_epoch": None})
    reads = iter([{}, ValueError("website_control_task-context_http_503:unclassified")])

    def current(*args, **kwargs):
        outcome = next(reads, None)
        if outcome is None:
            pytest.fail("a finished allocator was read again")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(worker, "require_current_source", current)

    def allocator(args):
        # The adapter's pre-create hook refuses; the adapter writes its no-create result.
        with pytest.raises(ValueError, match="control_unavailable_before_create"):
            jobs.before_provider_create(authority=authority, bundle=bundle)
        launch._write(tmp_path / "allocator_result.json", {"status": "blocked", "provider_mutations_performed": 0,
            "provider_create_attempted": False, "blockers": ["g1_beta_paid_precreate_recheck_failed"]})
        return 2

    assert jobs.run_invocation(path, allocator=allocator) == 2
    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    result = launch.reconcile_candidate(plan={}, candidate={}, attempt_dir=tmp_path, prepared=prepared)
    assert result["retryable_unstarted_refusal"] is True and result["no_allocation_proven"] is True
    assert result["unstarted_refusal_receipt"]["path"] == str(path.parent / "no_create_fence.json")


def test_unspent_recovery_publishes_the_wait_code_the_website_accepts(tmp_path, monkeypatch):
    """The WebApp refuses a running->preparing regression unless it carries the capacity-wait code."""
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from blueprint_pipeline import beta_policy_screening_launch as launch
    from tests.test_beta_policy_screening_coordinator import Backend, plan

    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)

    class SpawnedThenLost(Backend):
        def prepare_candidate(self, **kwargs):
            return {"prepared_digest": "sha256:" + "c" * 64}

        def launch_candidate(self, **kwargs):
            directory = kwargs["attempt_dir"]
            path = jobs.retain_invocation(attempt_dir=directory,
                binding={"prepared_digest": kwargs["prepared"]["prepared_digest"]}, argv=["gpu-canary"])
            launch._write(directory / "allocator_launch_intent.json", {
                "prepared_digest": kwargs["prepared"]["prepared_digest"], "invocation_manifest": str(path)})
            return {"status": "running", "handle": {"pid": 1}, "blockers": []}

        def reconcile_candidate(self, **kwargs):
            return launch.reconcile_candidate(**kwargs)

    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = SpawnedThenLost()
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend, require_publication_ack=False)
    row = coordinator._ledger(tmp_path)["campaigns"]["campaign-1"]["candidates"]["candidate-000"]
    assert row["status"] == "running"
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend, require_publication_ack=False)
    row = coordinator._ledger(tmp_path)["campaigns"]["campaign-1"]["candidates"]["candidate-000"]
    assert row["status"] == "preparing" and row["attempt"]["generation"] == 1
    assert coordinator.CAPACITY_WAIT_BLOCKER in row["blockers"]


def _fresh_attempt_backend(monkeypatch, *, self_fenced):
    """Each attempt spawns an allocator that refuses before any create.

    ``self_fenced``: its create-time control read got no answer, so it fenced
    its own job and wrote its no-create result. Otherwise it was lost before
    create (a crash or a deploy), and recovery fences the job.
    """
    from blueprint_pipeline import beta_policy_screening_launch as launch
    from tests.test_beta_policy_screening_coordinator import Backend

    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)

    class FreshAttempts(Backend):
        def prepare_candidate(self, **kwargs):
            self.preparations += 1
            return {"prepared_digest": "sha256:" + "c" * 64}

        def launch_candidate(self, **kwargs):
            self.launches += 1
            directory = kwargs["attempt_dir"]
            path = jobs.retain_invocation(attempt_dir=directory,
                binding={"prepared_digest": kwargs["prepared"]["prepared_digest"]}, argv=["gpu-canary"])
            launch._write(directory / "allocator_launch_intent.json", {
                "prepared_digest": kwargs["prepared"]["prepared_digest"], "invocation_manifest": str(path)})
            if self_fenced:
                assert jobs.fence_unstarted(path)["status"] == "fenced_unspent"  # The allocator's own fence.
                launch._write(directory / "allocator_result.json", {"status": "blocked",
                    "provider_mutations_performed": 0, "provider_create_attempted": False,
                    "blockers": ["beta_screening_control_unavailable_before_create"]})
            return {"status": "running", "handle": {"pid": 1}, "blockers": []}

        def reconcile_candidate(self, **kwargs):
            self.reconciliations += 1
            return launch.reconcile_candidate(**kwargs)

    return FreshAttempts()


def test_fresh_attempts_after_unanswered_create_reads_are_capped_at_three(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from tests.test_beta_policy_screening_coordinator import plan, publication

    monkeypatch.setattr(coordinator, "CAPACITY_RETRY_SECONDS", 0)
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = _fresh_attempt_backend(monkeypatch, self_fenced=True)

    def row():
        return coordinator._ledger(tmp_path)["campaigns"]["campaign-1"]["candidates"]["candidate-000"]

    for retry in range(1, coordinator.TRANSPORT_RETRY_CAP + 1):
        coordinator.process_screening_queue(queue_root=tmp_path, executor=backend, require_publication_ack=False)
        coordinator.process_screening_queue(queue_root=tmp_path, executor=backend, require_publication_ack=False)
        current = row()
        assert current["status"] == "preparing" and current["transport_retries"] == retry
        assert current["attempt"]["generation"] == retry
        assert publication(tmp_path)["budget"]["reserved_usd"] == "4"
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend, require_publication_ack=False)
    coordinator.process_screening_queue(queue_root=tmp_path, executor=backend, require_publication_ack=False)
    current = row()
    assert backend.launches == 4 and backend.preparations == 4
    assert current["status"] == "blocked" and current["blockers"] == [coordinator.TRANSPORT_RETRY_CAP_BLOCKER]
    attempts = [*current["attempt_history"], current["attempt"]]
    assert len(attempts) == 4 and all(item["reservation_status"] == "released" for item in attempts)
    # The fourth job keeps its fence; nothing can start it later.
    fence = tmp_path / "attempts" / current["attempt"]["attempt_id"] / "invocation" / "no_create_fence.json"
    assert fence.is_file() and current["attempt"]["unstarted_refusal_receipt"]["path"] == str(fence)
    budget = publication(tmp_path)["budget"]
    assert budget["reserved_usd"] == "0" and budget["spent_upper_bound_usd"] == "0"


def test_lost_allocator_recovery_does_not_spend_the_unanswered_read_budget(tmp_path, monkeypatch):
    """A crash or a deploy before create is not a control refusal; it keeps its existing recovery."""
    from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
    from tests.test_beta_policy_screening_coordinator import plan

    monkeypatch.setattr(coordinator, "CAPACITY_RETRY_SECONDS", 0)
    coordinator.enqueue_screening_campaign(plan(), queue_root=tmp_path)
    backend = _fresh_attempt_backend(monkeypatch, self_fenced=False)
    for _ in range(coordinator.TRANSPORT_RETRY_CAP + 1):
        coordinator.process_screening_queue(queue_root=tmp_path, executor=backend, require_publication_ack=False)
        coordinator.process_screening_queue(queue_root=tmp_path, executor=backend, require_publication_ack=False)
    row = coordinator._ledger(tmp_path)["campaigns"]["campaign-1"]["candidates"]["candidate-000"]
    assert row["status"] == "preparing" and row.get("transport_retries", 0) == 0
    assert row["attempt"]["generation"] == coordinator.TRANSPORT_RETRY_CAP + 1


def test_monitor_start_without_answer_fences_the_job_for_a_counted_retry(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_control as control
    from blueprint_pipeline import beta_policy_screening_launch as launch
    from blueprint_pipeline import beta_policy_screening_worker as worker
    prepared = {"prepared_digest": "sha256:" + "c" * 64}
    path = jobs.retain_invocation(attempt_dir=tmp_path, binding={**prepared, "provider": "vast"},
        argv=["gpu-canary"], control={"source": {}, "site_id": "site-one", **prepared,
            "initial_observed_at_epoch": time.time(), "execution_deadline_epoch": None})

    def unanswered(*args, **kwargs):
        raise ValueError("website_control_task-context_http_503:unclassified")

    monkeypatch.setattr(worker, "require_current_source", unanswered)
    with pytest.raises(ValueError, match=control.CONTROL_UNAVAILABLE_BEFORE_START):
        jobs.run_invocation(path, allocator=lambda argv: pytest.fail("started without a current answer"))
    assert (path.parent / "no_create_fence.json").is_file()
    assert not (path.parent / "provider_create_intent.json").exists()
    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    result = launch.reconcile_candidate(plan={}, candidate={}, attempt_dir=tmp_path, prepared=prepared)
    assert result["retryable_unstarted_refusal"] is True and result["retry_cause"] == "control_unanswered"


def test_definite_start_refusal_is_not_fenced_as_a_retry(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_worker as worker
    prepared = {"prepared_digest": "sha256:" + "c" * 64}
    path = jobs.retain_invocation(attempt_dir=tmp_path, binding={**prepared, "provider": "vast"},
        argv=["gpu-canary"], control={"source": {}, "site_id": "site-one", **prepared,
            "initial_observed_at_epoch": time.time(), "execution_deadline_epoch": None})

    def forbidden(*args, **kwargs):
        raise ValueError("website_control_task-context_http_403:unclassified")

    monkeypatch.setattr(worker, "require_current_source", forbidden)
    with pytest.raises(ValueError, match="unproven_before_start"):
        jobs.run_invocation(path, allocator=lambda argv: pytest.fail("started without a current answer"))
    assert not (path.parent / "no_create_fence.json").exists()
