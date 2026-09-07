"""Compose real owner admission with the existing real dispatcher resume rehearsal."""
import pytest

from blueprint_pipeline import task_evaluation_scene_execution_authority as authority
from blueprint_pipeline import task_evaluation_scene_intake as intake
from tests import test_task_evaluation_policy_canary_dispatcher as rehearsal
from tests.test_task_evaluation_scene_intake import stage, attempt, request


@pytest.mark.parametrize("ending", ["expiry", "revocation"])
def test_dispatch_billing_and_delivery_resume_after_owner_authority_ends(tmp_path, monkeypatch, ending):
    owner_root = tmp_path / "owner"
    intent = stage(owner_root)
    reserved = attempt(owner_root, intent, commit=rehearsal.COMMIT[0], cost=4)
    binding = authority.bind_scene_attempt(reserved)
    monkeypatch.setenv(intake.CLIENTS_ENV, "webapp")
    clock = [102]
    admissions = []
    real_dispatch = rehearsal.dispatch_policy_canary_activation
    real_require = authority.require_scene_execution_authority

    def admission(setup, **kwargs):
        admissions.append(clock[0])
        return real_require({**setup, **binding}, queue_root=owner_root, now=clock[0], **kwargs)

    monkeypatch.setattr("blueprint_pipeline.task_evaluation_policy_canary_dispatcher.require_scene_execution_authority", admission)

    def dispatch(**kwargs):
        original = kwargs["allocator_runner"]
        def allocate(argv):
            result = original(argv)  # injected fixture allocator, never a provider
            if ending == "revocation":
                intake.revoke_scene_intent(queue_root=owner_root, intent_id=intent["intent_id"],
                    intent_digest=intent["intent_digest"], owner=request()["owner"], now=103)
                clock[0] = 104
            else:
                clock[0] = 1001
            return result
        return real_dispatch(**{**kwargs, "allocator_runner": allocate})

    monkeypatch.setattr(rehearsal, "dispatch_policy_canary_activation", dispatch)
    # This helper exercises accepted output -> billing pending twice -> posted
    # billing -> sealed delivery, with immutable result and one allocator call.
    rehearsal.test_live_shaped_result_waits_for_billing_and_never_launches_twice(tmp_path, monkeypatch)
    assert admissions == [102, 102]
    with pytest.raises(authority.SceneExecutionAuthorityError, match="revoked|expired"):
        real_require({**binding, "source_commit": rehearsal.COMMIT},
            source_commit=rehearsal.COMMIT, provider="vast", maximum_spend_usd=4,
            queue_root=owner_root, now=clock[0])


@pytest.mark.parametrize("ending", ["expiry", "revocation"])
def test_authority_ending_during_preallocation_progress_stops_allocator(tmp_path, monkeypatch, ending):
    owner_root = tmp_path / "owner"
    intent = stage(owner_root)
    reserved = attempt(owner_root, intent, commit=rehearsal.COMMIT[0], cost=4)
    binding = authority.bind_scene_attempt(reserved)
    monkeypatch.setenv(intake.CLIENTS_ENV, "webapp")
    clock = [102]
    allocations = []
    real_dispatch = rehearsal.dispatch_policy_canary_activation
    real_require = authority.require_scene_execution_authority
    def admission(setup, **kwargs):
        return real_require({**setup, **binding}, queue_root=owner_root, now=clock[0], **kwargs)
    monkeypatch.setattr("blueprint_pipeline.task_evaluation_policy_canary_dispatcher.require_scene_execution_authority", admission)
    def dispatch(**kwargs):
        allocator = kwargs["allocator_runner"]
        progress = kwargs.get("progress_sync_runner", lambda **_values: {"status": "succeeded"})
        def sync(**values):
            result = progress(**values)
            if values["progress"]["phase"] == "provider_allocating":
                if ending == "revocation":
                    intake.revoke_scene_intent(queue_root=owner_root, intent_id=intent["intent_id"],
                        intent_digest=intent["intent_digest"], owner=request()["owner"], now=103)
                    clock[0] = 104
                else:
                    clock[0] = 1001
            return result
        def allocate(argv):
            allocations.append(argv)
            return allocator(argv)
        return real_dispatch(**{**kwargs, "allocator_runner": allocate, "progress_sync_runner": sync})
    monkeypatch.setattr(rehearsal, "dispatch_policy_canary_activation", dispatch)
    with pytest.raises(authority.SceneExecutionAuthorityError, match="expired|revoked"):
        rehearsal.test_live_shaped_result_waits_for_billing_and_never_launches_twice(tmp_path, monkeypatch)
    assert allocations == []
    assert not (tmp_path / "dispatch-live" / "allocator_invocation_started.json").exists()


def test_failed_delivery_after_teardown_resumes_only_publication(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_policy_canary_dispatcher as dispatcher
    calls = []
    def failed_sync(**kwargs):
        calls.append(kwargs["run_id"])
        return {"status": "failed"}
    run = rehearsal.materialize_canary_root(tmp_path, monkeypatch, sync_runner=failed_sync)
    root = run["root"]
    assert run["receipt"]["status"] == "awaiting_website_sync_or_notification"
    assert (root / "dispatch_pending.json").is_file()
    assert not (root / "dispatch_receipt.json").exists()
    frozen = {path: path.read_bytes() for path in (
        run["joined_path"], root / "official_billing_reconciliation.json",
        root / "teardown.json", root / "post_teardown_global_provider_zero.json",
        root / "artifacts/result_delivery/delivery.json",
    )}
    real_sync = rehearsal._echoing_website(monkeypatch)
    def recovered_sync(**kwargs):
        calls.append(kwargs["run_id"])
        return real_sync(**kwargs)
    setup = run["setup"]
    receipt = dispatcher._resume_materialized_policy_canary_delivery(root=root, setup=setup,
        runtime_inputs={"configuration_digest": run["joined"]["configuration_digest"],
            "task_success_contract": setup["task_success_contract"],
            "task_success_contract_digest": setup["task_success_contract_digest"]},
        authority={"authority_digest": "sha256:" + "1" * 64},
        bundle={"bundle_sha256": "sha256:" + "2" * 64},
        adapter={"teardown_manifest_path": str(root / "teardown.json")}, sync_runner=recovered_sync)
    assert receipt["allocator_invoked"] is False
    assert receipt["teardown"]["teardown_completed"] is True
    assert len(calls) == 2
    assert {path: path.read_bytes() for path in frozen} == frozen
    assert (root / "dispatch_receipt.json").is_file()


def test_owner_binding_upgrade_preserves_preexisting_started_authority(tmp_path, monkeypatch):
    """A release upgrade cannot rewrite an already started run's authority digest."""
    import json
    from blueprint_pipeline import task_evaluation_policy_canary_dispatcher as dispatcher
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    owner_root = tmp_path / "owner"
    intent = stage(owner_root)
    reserved = attempt(owner_root, intent, commit=rehearsal.COMMIT[0], cost=4)
    monkeypatch.setenv(intake.ROOT_ENV, str(owner_root))
    monkeypatch.setenv(intake.CLIENTS_ENV, "webapp")
    monkeypatch.setattr(authority.time, "time", lambda: 102)
    validate = dispatcher.validate_policy_canary_execution_setup
    monkeypatch.setattr(dispatcher, "validate_policy_canary_execution_setup",
        lambda value: {**validate(value), **authority.bind_scene_attempt(reserved)})
    original_write = dispatcher._write_exclusive
    retained = []
    def simulate_old_producer(path, value):
        if path.name == "policy_canary_session_authority.json" and not retained:
            value.pop("scene_execution_owner", None)
            value["authority_digest"] = canonical_digest(value, digest_field="authority_digest")
            original_write(path, value)
            retained.append((path, path.read_bytes()))
            return
        return original_write(path, value)
    monkeypatch.setattr(dispatcher, "_write_exclusive", simulate_old_producer)
    rehearsal.test_live_shaped_result_waits_for_billing_and_never_launches_twice(tmp_path, monkeypatch)
    path, original = retained[0]
    assert path.read_bytes() == original
    assert "scene_execution_owner" not in json.loads(original)
