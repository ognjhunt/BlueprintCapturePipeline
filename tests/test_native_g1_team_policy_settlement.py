"""Real selected bundle/worker/charge/registry contracts meet at settlement."""

import json
from pathlib import Path
import shutil

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest as digest
from blueprint_pipeline import native_g1_team_policy_settlement as settlement
from blueprint_pipeline import native_g1_private_review_ingest as ingest
from blueprint_pipeline.native_g1_team_policy_dispatcher import START_SCHEMA, FINAL_SCHEMA
from blueprint_pipeline.native_g1_team_policy_preparation import SCHEMA as PREPARATION_SCHEMA
from tests.test_native_g1_team_review_evidence import _completed
from tests.test_native_g1_official_billing import _selected_case, _selected_billing_source, _write
from tests.test_native_g1_private_review_ingest import _Response
from tests.test_native_g1_team_policy_run_request import NOW


def _seal(path, value, field):
    value[field] = digest(value, digest_field=field)
    _write(path, value)
    return value


def _case(tmp_path, monkeypatch):
    args, bundle, old_adapter, old_output = _completed(tmp_path, monkeypatch)
    intent_path = args["authority_arguments"]["intent_path"]
    intent = json.loads(intent_path.read_text())
    work = tmp_path / "work"
    directory = work / intent["intent_id"]
    directory.mkdir(parents=True)
    shutil.move(str(old_adapter.parent), directory / "run")
    run = directory / "run"
    adapter_path = run / old_adapter.name
    adapter = json.loads(adapter_path.read_text())
    attempt = run / "attempts/attempt_001"
    output = attempt / "immutable_execution"
    native_path = output / "native_g1_team_provider_result.v1.json"
    native = json.loads(native_path.read_text())
    native["claim_ceiling"] = "development_only"
    native["result_digest"] = canonical_digest(native, digest_field="result_digest")
    terminal_path, financial = _selected_case(run)
    _write(native_path, native)
    financial["native_control_result_digest"] = native["result_digest"]
    financial["bundle_sha256"] = bundle["bundle_sha256"]
    artifact_path = Path(financial["artifact_manifest_path"])
    artifact = json.loads(artifact_path.read_text())
    artifact["binding"]["bundle_sha256"] = bundle["bundle_sha256"]
    artifact["manifest_digest"] = canonical_digest(artifact, digest_field="manifest_digest")
    _write(artifact_path, artifact)
    _write(terminal_path, financial)
    _write(attempt / "adp_arena_vast_result.json", financial)
    adapter.update(financial)
    adapter["allocation_binding_digest"] = "sha256:" + "a" * 64
    _write(adapter_path, adapter)
    shutil.move(str(Path(bundle["receipt_path"]).parent), directory / "bundle")
    bundle_path = directory / "bundle/native_g1_team_provider_bundle.v1.json"
    bundle.update(bundle_path=str(directory / "bundle" / Path(bundle["bundle_path"]).name), receipt_path=str(bundle_path))
    _write(bundle_path, bundle)
    import zipfile
    from blueprint_pipeline.native_g1_team_provider_bundle import PACKET_RELATIVE_PATH
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        packet = json.loads(archive.read(PACKET_RELATIVE_PATH))
    from blueprint_pipeline.native_g1_team_paid_policy import CONSUMPTION_FILENAME
    _write(run / CONSUMPTION_FILENAME, {
        "schema_version": "native_g1_team_paid_attempt_consumption.v1", "status": "consumed", "retry_cap": 0,
        "execution_packet_digest": packet["packet_digest"], "orchestrator_source_commit": packet["implementation_commit"],
        "allocation_binding_digest": adapter["allocation_binding_digest"],
    })
    prepared = _seal(directory / "preparation.json", {
        "schema_version": PREPARATION_SCHEMA, "status": "bundle_prepared_not_executed",
        "intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"],
        "implementation_commit": packet["implementation_commit"], "bundle_receipt_path": str(bundle_path),
        "operator_approval_digest": packet["operator_approval"]["approval_digest"],
        "policy_profile_digest": packet["policy_profile_digest"], "objective_id": packet["objective_id"],
        "authorization": intent["request"]["authorization"], "claim_ceiling": "development_only",
        "provider_mutation_performed": False,
        **{key: bundle[key] for key in ("bundle_sha256", "manifest_digest", "execution_packet_digest")},
    }, "preparation_digest")
    started = _seal(directory / "execution_started.json", {
        "schema_version": START_SCHEMA, "status": "execution_started_once", "retry_cap": 0,
        "intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"],
        "preparation_digest": prepared["preparation_digest"], "implementation_commit": packet["implementation_commit"],
        "started_at_epoch": NOW,
    }, "start_digest")
    _seal(directory / "dispatch_final.json", {
        "schema_version": FINAL_SCHEMA, "status": "controller_completed_pending_billing_and_private_delivery",
        "intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"], "start_digest": started["start_digest"],
        "implementation_commit": packet["implementation_commit"], "paid_allocator_exit_code": 0,
        "adapter_status": "completed", "adapter_result_path": str(adapter_path), "adapter_result_digest": digest(adapter),
        "controller_reported_episode_verified": True, "run_teardown_confirmed_by_adapter": True,
    }, "dispatch_digest")
    _write(attempt / "vast_provider_run/vast_startup_probe_manifest.json", {
        "schema_version": "vast_startup_probe_manifest.v1", "status": "completed",
        "create_request_summary": {"label": "blueprint-native-task-arena-g1-team-one"},
        "last_instance_payload": {"instances": {"id": 52686067, "label": "blueprint-native-task-arena-g1-team-one"}},
    })
    source = _selected_billing_source(tmp_path / "audit")
    zero = {"schema_version": "gpu_spend_guard.v1", "provider_zero_verified": True, "live_instance_count": 0}
    zero["receipt_digest"] = canonical_digest(zero, digest_field="receipt_digest")
    calls = []
    class Opener:
        def open(self, request, *, timeout):
            payload = json.loads(request.data)
            calls.append(payload)
            return _Response(request.full_url, {"status": "ingested", "run_id": payload["run_id"], "review_digest": payload["review"]["review_digest"]})
    monkeypatch.setattr(ingest.urllib.request, "build_opener", lambda *_args: Opener())
    return {
        "intent_path": intent_path, "work_root": work, "billing_audit_root": source.parent.parent,
        "result_root": tmp_path / "results", "webapp_url": "https://tryblueprint.io", "sync_token": "test-private-token",
        "collect_zero": lambda: zero,
    }, directory, calls


def test_selected_settlement_uses_real_charge_and_delivery_and_reopens_on_replay(tmp_path, monkeypatch):
    args, directory, calls = _case(tmp_path, monkeypatch)
    monkeypatch.setattr("blueprint_pipeline.native_g1_team_policy_run_request.time.time", lambda: NOW + 7200)
    result = settlement.settle_g1_team_policy(**args)
    assert result["status"] == "delivered_owner_only"
    assert result["official_total_usd"] == 0.5
    assert result["public_redistribution_authorized"] is False
    assert len(calls) == 1
    assert calls[0]["review"]["episodes"][0]["score"]["outcome"] == "failure"
    assert calls[0]["review"]["episodes"][0]["policy_query_count"] == 2
    assert settlement.settle_g1_team_policy(**args) == result
    assert len(calls) == 1
    Path(json.loads((directory / "run/adp_arena_vast_result.json").read_text())["object_store_cleanup_path"]).write_text("{}")
    with pytest.raises(ValueError):
        settlement.settle_g1_team_policy(**args)


def test_selected_settlement_pending_zero_does_not_deliver_and_checks_fresh_after_cached_zero(tmp_path, monkeypatch):
    args, directory, calls = _case(tmp_path, monkeypatch)
    blocked = {"provider_zero_verified": False, "blockers": ["other_provider_active"]}
    result = settlement.settle_g1_team_policy(**{**args, "collect_zero": lambda: blocked})
    assert result["status"] == "awaiting_global_provider_zero"
    assert calls == []
    settlement.settle_g1_team_policy(**args)
    result = settlement.settle_g1_team_policy(**{**args, "collect_zero": lambda: blocked})
    assert result["status"] == "awaiting_global_provider_zero"
    assert len(calls) == 1


@pytest.mark.parametrize("fault", ["adapter_digest", "packet_intent", "startup_label", "started_after_expiry"])
def test_selected_settlement_refuses_mismatched_start_input_and_provider(tmp_path, monkeypatch, fault):
    args, directory, calls = _case(tmp_path, monkeypatch)
    final_path = directory / "dispatch_final.json"
    final = json.loads(final_path.read_text())
    if fault == "adapter_digest":
        final["adapter_result_digest"] = "sha256:" + "b" * 64
        _seal(final_path, final, "dispatch_digest")
    elif fault == "packet_intent":
        intent = json.loads(args["intent_path"].read_text())
        intent["request"]["owner"]["user_id"] = "other-owner"
        args["intent_path"].chmod(0o600)
        _seal(args["intent_path"], intent, "intent_digest")
    elif fault == "startup_label":
        path = directory / "run/attempts/attempt_001/vast_provider_run/vast_startup_probe_manifest.json"
        value = json.loads(path.read_text())
        value["last_instance_payload"]["instances"]["label"] = "other"
        _write(path, value)
    else:
        path = directory / "execution_started.json"
        value = json.loads(path.read_text())
        value["started_at_epoch"] = NOW + 7200
        _seal(path, value, "start_digest")
        final["start_digest"] = value["start_digest"]
        _seal(final_path, final, "dispatch_digest")
    with pytest.raises(ValueError):
        settlement.settle_g1_team_policy(**args)
    assert calls == []


def test_selected_settlement_charge_pending_then_resumes_interrupted_registration(tmp_path, monkeypatch):
    args, directory, calls = _case(tmp_path, monkeypatch)
    audit = args["billing_audit_root"]
    hidden = audit.with_name("billing-not-posted")
    audit.rename(hidden)
    result = settlement.settle_g1_team_policy(**args)
    assert result["status"] == "awaiting_posted_official_billing"
    assert calls == []
    assert not (directory / "private_delivery.json").exists()
    hidden.rename(audit)
    real_ingest = settlement.ingest_g1_private_review
    monkeypatch.setattr(settlement, "ingest_g1_private_review", lambda **kwargs: (_ for _ in ()).throw(OSError("fake interruption")))
    with pytest.raises(OSError, match="fake interruption"):
        settlement.settle_g1_team_policy(**args)
    assert (directory / "private_delivery.json").is_file()
    assert not (directory / "private_ingest.json").exists()
    monkeypatch.setattr(settlement, "ingest_g1_private_review", real_ingest)
    assert settlement.settle_g1_team_policy(**args)["status"] == "delivered_owner_only"
    assert len(calls) == 1


def test_selected_settlement_queue_skips_unfinished_start_and_pending_charge(tmp_path, monkeypatch):
    queue, work = tmp_path / "queue", tmp_path / "work"
    seen = []
    for suffix in ("a", "b", "c"):
        identity = "g1-team-policy-" + suffix * 64
        _write(queue / identity / "intent.json", {})
        if suffix != "a":
            _seal(work / identity / "dispatch_final.json", {
                "status": "controller_completed_pending_billing_and_private_delivery",
            }, "dispatch_digest")
        else:
            _write(work / identity / "execution_started.json", {})
    def settle(**kwargs):
        seen.append(kwargs["intent_path"].parent.name)
        return {"status": "awaiting_posted_official_billing" if len(seen) == 1 else "delivered_owner_only"}
    monkeypatch.setattr(settlement, "settle_g1_team_policy", settle)
    result = settlement.settle_pending_g1_team_policies(queue_root=queue, work_root=work)
    assert result["status"] == "delivered_owner_only"
    assert seen == ["g1-team-policy-" + "b" * 64, "g1-team-policy-" + "c" * 64]
