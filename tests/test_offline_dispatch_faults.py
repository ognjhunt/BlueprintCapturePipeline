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
    """An owned legacy attempt needs real delivery proof without new allocation."""
    import hashlib
    import io
    import json
    from functools import partial
    from pathlib import Path
    from urllib.parse import urlsplit
    from blueprint_pipeline import task_evaluation_policy_canary_dispatcher as dispatcher
    from blueprint_pipeline import task_evaluation_owner_delivery_readback as owner_delivery
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from tests.test_task_evaluation_owner_delivery_readback import _inputs

    # The earlier test grafted an owner onto a deliberately unowned rehearsal,
    # whose projection omitted run_id and whose delivery had no artifact rows.
    # Retain a real completed-scene factory's owner/preparation lineage instead.
    fixture_root = tmp_path / "owner-fixture"
    fixture_root.mkdir()
    owner_inputs, owner_root, intent = _inputs(fixture_root, monkeypatch)
    reserved = attempt(owner_root, intent, attempt_id="started-compatibility",
                       commit=rehearsal.COMMIT[0], cost=4, now=intent["accepted_at_epoch"] + 2)
    monkeypatch.setattr(authority.time, "time", lambda: intent["accepted_at_epoch"] + 3)
    validate = dispatcher.validate_policy_canary_execution_setup
    monkeypatch.setattr(dispatcher, "validate_policy_canary_execution_setup",
        lambda value: {**validate(value), **authority.bind_scene_attempt(reserved),
                       "capture_session_id": owner_inputs["setup"]["capture_session_id"]})
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

    payloads = {"a" * 32: b"retained episode evidence", "b" * 32: b"retained native report"}
    artifacts = [{"artifact_id": key, "digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
                  "size_bytes": len(raw)} for key, raw in payloads.items()]
    http_state = {"mode": "wrong_owner", "gets": [], "posts": []}
    class Response(io.BytesIO):
        status = 200
    def opener(call, *, timeout):
        if call.method == "POST":
            body = json.loads(call.data)
            assert body["schema_version"] == "task_evaluation_delivery_readback_request.v2"
            assert call.get_header("X-blueprint-pipeline-signature").startswith("sha256=")
            http_state["posts"].append(body)
            result = {"schema_version": "task_evaluation_delivery_readback.v1", "status": "verified",
                **{k: body[k] for k in ("run_id", "request_digest", "configuration_digest", "owner_user_id",
                                       "team_namespace", "result_delivery_digest", "policy_canary_projection_digest")},
                "inbox": {"status": "verified", "run_id": body["run_id"],
                    "owner_user_id": "wrong-owner" if http_state["mode"] == "wrong_owner" else body["owner_user_id"],
                    "team_namespace": body["team_namespace"], "projection_digest": body["policy_canary_projection_digest"],
                    "source": "website_owner_run_index_readback"},
                "ephemeral_downloads": [{"artifact_id": row["artifact_id"], "sha256": row["digest"],
                    "size_bytes": row["size_bytes"], "download_url": "/api/task-evaluation-result-downloads/result/"
                    + row["artifact_id"] + "?signature=ephemeral-fixture"} for row in artifacts if row["artifact_id"] in body["artifact_ids"]]}
            return Response(json.dumps(result).encode())
        key = urlsplit(call.full_url).path.rsplit("/", 1)[-1]
        http_state["gets"].append((http_state["mode"], key))
        return Response(b"wrong bytes" if http_state["mode"] == "bad_bytes" else payloads[key])
    reader = partial(owner_delivery.verify_website_delivery, endpoint_url="https://website.example/readback",
                     token="fixture-token", opener=opener)
    monkeypatch.setattr(owner_delivery, "verify_website_delivery", reader)
    real_dispatch = rehearsal.dispatch_policy_canary_activation
    dispatch_calls = []
    def dispatch(**kwargs):
        dispatch_calls.append(kwargs)
        if len(dispatch_calls) < 3:
            return real_dispatch(**kwargs)
        # Fill the publication/byte contract at the external edges; retain the
        # real owned-delivery validator and actual streamed SHA-256 verification.
        monkeypatch.setattr(dispatcher, "_projection", lambda **values: {
            "run_id": values["result"]["run_id"], "projection_digest": "sha256:" + "e" * 64})
        monkeypatch.setattr(dispatcher, "materialize_policy_canary_website_delivery",
            lambda *, run_root, delivery: {**delivery, "artifacts": artifacts})
        original_sync = kwargs["sync_runner"]
        def sync(**values):
            return {**original_sync(**values), **{k: values[k] for k in ("run_id", "request_digest", "configuration_digest")},
                    "result_delivery_digest": values["result_delivery"]["delivery_digest"],
                    "policy_canary_projection_digest": values["policy_canary_result"]["projection_digest"]}
        kwargs = {**kwargs, "sync_runner": sync}
        root = Path(kwargs["output_root"])
        for mode in ("wrong_owner", "bad_bytes", "verified"):
            http_state["mode"] = mode
            result = real_dispatch(**kwargs)
            path, original = retained[0]
            assert path.read_bytes() == original
            if mode != "verified":
                assert result["status"] == "awaiting_website_download_readback"
                assert not (root / "dispatch_receipt.json").exists()
                assert not (root / "artifacts/result_delivery/owner_delivery_readback.json").exists()
            else:
                proof_path = Path(result["owner_delivery_readback"]["path"])
                proof = json.loads(proof_path.read_text())
                assert result["owner_delivery_readback"]["sha256"] == "sha256:" + hashlib.sha256(proof_path.read_bytes()).hexdigest()
                assert proof["every_artifact_downloaded_and_hashed"] is True
                assert {row["artifact_id"] for row in proof["artifacts"]} == set(payloads)
                assert "ephemeral-fixture" not in proof_path.read_text()
        return result
    monkeypatch.setattr(rehearsal, "dispatch_policy_canary_activation", dispatch)
    rehearsal.test_live_shaped_result_waits_for_billing_and_never_launches_twice(tmp_path, monkeypatch)
    path, original = retained[0]
    assert path.read_bytes() == original
    assert "scene_execution_owner" not in json.loads(original)
    assert len(http_state["posts"]) == 3
    assert all(body["owner_user_id"] == "u1" and body["team_namespace"].startswith("scene-")
               for body in http_state["posts"])
    assert all(mode != "wrong_owner" for mode, _ in http_state["gets"])
    assert {(mode, key) for mode, key in http_state["gets"] if mode == "verified"} == {
        ("verified", key) for key in payloads}
