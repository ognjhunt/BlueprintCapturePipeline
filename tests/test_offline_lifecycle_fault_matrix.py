"""ADP-004/009D offline transitions; all inventories and providers are synthetic."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import paid_lane_guard as guard
from blueprint_pipeline import paid_provider_allocation_lifecycle as lifecycle
from blueprint_pipeline import task_evaluation_scene_execution_authority as authority
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline.provider_reliability_manifest import build_teardown_proof
from tests.test_task_evaluation_scene_intake import stage, attempt


def proof(instance="owned", provider="vast"):
    return build_teardown_proof(provider=provider, allocation_id=instance,
        terminate_requested=True, provider_terminal_status="not_found",
        verified_at="2026-09-07T12:00:00Z", status_source="provider_api")


@pytest.mark.parametrize("provider,instance", [("vast", "other"), ("runpod", "owned")])
def test_wrong_resource_cannot_close_pending_obligation(tmp_path, provider, instance):
    record = guard.open_pending_teardown(provider="vast", lane="offline", run_id="one",
        instance_id="owned", registry_dir=tmp_path)
    result = guard.close_pending_teardown(record["path"], proof(instance, provider))
    assert result["status"] == "open"
    assert result["close_refused_reason"] == "teardown_proof_identity_mismatch"
    assert json.loads(Path(record["path"]).read_text())["instance_id"] == "owned"


def test_restart_cannot_rebind_known_instance(tmp_path):
    record = guard.open_pending_teardown(provider="vast", lane="offline", run_id="one",
        registry_dir=tmp_path)
    guard.bind_pending_teardown_instance(record["path"], "owned")
    before = Path(record["path"]).read_bytes()
    with pytest.raises(ValueError, match="instance_identity_conflict"):
        guard.bind_pending_teardown_instance(record["path"], "other")
    assert Path(record["path"]).read_bytes() == before


@pytest.mark.parametrize("observation", [None, {}, {"http": 503},
    {"status": "observed", "http": 200, "desiredStatus": "running"},
    {"status": "absent", "http": 200}])
def test_timeout_then_restart_teardown_retains_exact_obligation(tmp_path, observation):
    record = guard.open_pending_teardown(provider="vast", lane="offline", run_id="one",
        instance_id="owned", registry_dir=tmp_path)
    other = guard.open_pending_teardown(provider="vast", lane="offline", run_id="other",
        instance_id="other", registry_dir=tmp_path)
    other_bytes = Path(other["path"]).read_bytes()
    calls = []
    class Provider:
        name = "vast"
        def terminate(self, instance):
            calls.append(("terminate", instance))
            raise TimeoutError("lost delete acknowledgement")
        def inspect(self, instance):
            calls.append(("inspect", instance))
            if observation is None:
                raise TimeoutError("inventory unavailable")
            return observation
    def settle():
        return lifecycle.finalize_known_allocation(provider_obj=Provider(), instance_id="owned",
            pending_path=record["path"], reason="offline-restart",
            teardown_proof_builder=lifecycle.teardown_proof_from_attempt,
            close_pending=guard.close_pending_teardown,
            release_lane=lambda *_a, **_k: {"all_providers_terminal": True,
                "results": [{"status": "released"}]})
    assert settle()["terminal"] is False
    assert json.loads(Path(record["path"]).read_text())["status"] == "open"
    observation = {"http": 404}
    assert settle()["terminal"] is True
    assert json.loads(Path(record["path"]).read_text())["status"] == "closed"
    assert calls == [("terminate", "owned"), ("inspect", "owned")] * 2
    assert Path(other["path"]).read_bytes() == other_bytes


@pytest.mark.parametrize("moment", [999, 1000, 1001])
def test_reservation_restart_expiry_blocks_new_action_without_resetting_accounting(tmp_path, monkeypatch, moment):
    intent = stage(tmp_path)
    reserved = attempt(tmp_path, intent)
    monkeypatch.setenv(intake.CLIENTS_ENV, "webapp")
    profile = {**authority.bind_scene_attempt(reserved), "source_commit": reserved["source_commit"],
        "allocator": {"max_spend_usd": 2, "argv": ["--provider", "vast"]}}
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    blockers = authority.scene_execution_authority_blockers(profile, queue_root=tmp_path, now=moment)
    assert blockers == ([] if moment < 1000 else ["scene_execution_owner_expired"])
    assert authority.scene_execution_authority_blockers(profile, reopen_records=False, now=moment) == []
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("field", ["runtime_digest", "input_digest"])
def test_restart_cannot_inherit_authority_with_changed_inputs(tmp_path, monkeypatch, field):
    intent = stage(tmp_path)
    reserved = attempt(tmp_path, intent)
    monkeypatch.setenv(intake.CLIENTS_ENV, "webapp")
    profile = {**authority.bind_scene_attempt(reserved), "source_commit": reserved["source_commit"],
        "allocator": {"max_spend_usd": 2, "argv": ["--provider", "vast"]}}
    profile["scene_attempt_binding"][field] = "sha256:" + "1" * 64
    assert authority.scene_execution_authority_blockers(profile, queue_root=tmp_path, now=102) == ["scene_execution_owner_record_mismatch"]


def test_delayed_billing_replay_does_not_reprice_retained_charge(tmp_path):
    from tests.test_vast_official_billing_extractor import (
        _fixture, _refresh_response_binding, INSTANCE_A, LABEL_A,
    )
    from blueprint_pipeline.vast_official_billing_extractor import (
        extract_vast_official_instance_charge, VastOfficialBillingExtractionError,
    )
    fixture = _fixture(tmp_path)
    response = fixture["responses"][0]
    original = response.read_bytes()
    value = json.loads(original)
    posted = value["results"].pop(0)
    response.write_text(json.dumps(value))
    _refresh_response_binding(fixture, 0)
    def extract():
        return extract_vast_official_instance_charge(
            provider_billing_source_receipt_path=fixture["receipt"], instance_id=INSTANCE_A,
            launch_label=LABEL_A)
    with pytest.raises(VastOfficialBillingExtractionError, match="unposted"):
        extract()
    # A new posted response is separately retained; the old unposted bytes survive.
    pending = tmp_path / "retained-unposted.json"
    pending.write_bytes(response.read_bytes())
    value["results"].append(posted)
    response.write_text(json.dumps(value))
    _refresh_response_binding(fixture, 0)
    charge = extract()
    assert charge["official_charge_usd"] == 0.123
    assert extract() == charge
    assert json.loads(pending.read_text())["results"] == [value["results"][0]]
    value["results"].append(posted)
    response.write_text(json.dumps(value))
    _refresh_response_binding(fixture, 0)
    with pytest.raises(VastOfficialBillingExtractionError, match="duplicate"):
        extract()
    assert charge["official_charge_usd"] == 0.123


@pytest.mark.parametrize("field,value", [("source", "instance-999"), ("metadata", {"label": "wrong"})])
def test_billing_wrong_identity_is_not_zero_dollars(tmp_path, field, value):
    from tests.test_vast_official_billing_extractor import _fixture, _refresh_response_binding, INSTANCE_A, LABEL_A
    from blueprint_pipeline.vast_official_billing_extractor import extract_vast_official_instance_charge, VastOfficialBillingExtractionError
    fixture = _fixture(tmp_path)
    response = fixture["responses"][0]
    data = json.loads(response.read_text())
    data["results"][0][field] = value
    response.write_text(json.dumps(data))
    _refresh_response_binding(fixture, 0)
    with pytest.raises(VastOfficialBillingExtractionError, match="identity_invalid"):
        extract_vast_official_instance_charge(provider_billing_source_receipt_path=fixture["receipt"],
            instance_id=INSTANCE_A, launch_label=LABEL_A)


def test_duplicate_pending_open_cannot_reset_started_identity_or_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(guard.time, "time", lambda: 100)
    kwargs = dict(provider="vast", lane="offline", run_id="one", registry_dir=tmp_path)
    record = guard.open_pending_teardown(**kwargs)
    guard.bind_pending_teardown_instance(record["path"], "owned")
    before = Path(record["path"]).read_bytes()
    monkeypatch.setattr(guard.time, "time", lambda: 900)
    reopened = guard.open_pending_teardown(**kwargs)
    assert reopened["instance_id"] == "owned"
    assert reopened["started_at_epoch"] == 100
    assert Path(record["path"]).read_bytes() == before


def test_ambiguous_create_cannot_be_cancelled_as_unallocated(tmp_path):
    record = guard.open_pending_teardown(provider="vast", lane="offline", run_id="one", registry_dir=tmp_path)
    guard.mark_pending_teardown_ambiguous(record["path"], reason="create_ack_lost")
    cancelled = guard.cancel_pending_teardown(record["path"], reason="missing_local_result")
    assert cancelled["status"] == "open"
    assert cancelled["allocation_outcome_ambiguous"] is True


def test_concurrent_binding_cannot_replace_first_instance(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    record = guard.open_pending_teardown(provider="vast", lane="offline", run_id="one", registry_dir=tmp_path)
    def bind(identifier):
        try:
            return guard.bind_pending_teardown_instance(record["path"], identifier)["instance_id"]
        except ValueError:
            return "refused"
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(bind, ["owned-a", "owned-b"]))
    assert outcomes.count("refused") == 1
    retained = json.loads(Path(record["path"]).read_text())
    assert retained["instance_id"] in outcomes


@pytest.mark.parametrize("start,end", [(200, 100), (1, 2), (9_999_999_999, 9_999_999_999)])
def test_charge_outside_requested_billing_period_is_refused(tmp_path, start, end):
    from tests.test_vast_official_billing_extractor import _fixture, _refresh_response_binding, INSTANCE_A, LABEL_A
    from blueprint_pipeline.vast_official_billing_extractor import extract_vast_official_instance_charge, VastOfficialBillingExtractionError
    fixture = _fixture(tmp_path)
    response = fixture["responses"][0]
    data = json.loads(response.read_text())
    data["results"][0].update(start=start, end=end)
    response.write_text(json.dumps(data))
    _refresh_response_binding(fixture, 0)
    with pytest.raises(VastOfficialBillingExtractionError, match="period_invalid"):
        extract_vast_official_instance_charge(provider_billing_source_receipt_path=fixture["receipt"],
            instance_id=INSTANCE_A, launch_label=LABEL_A)


def test_out_of_order_posted_pages_bind_identity_not_position(tmp_path):
    from tests.test_vast_official_billing_extractor import _fixture, _refresh_response_binding, INSTANCE_A, LABEL_A
    from blueprint_pipeline.vast_official_billing_extractor import extract_vast_official_instance_charge
    fixture = _fixture(tmp_path)
    kwargs = dict(provider_billing_source_receipt_path=fixture["receipt"], instance_id=INSTANCE_A, launch_label=LABEL_A)
    original = extract_vast_official_instance_charge(**kwargs)
    first, second = fixture["responses"]
    a, b = json.loads(first.read_text()), json.loads(second.read_text())
    a["results"], b["results"] = b["results"], a["results"][::-1]
    first.write_text(json.dumps(a))
    second.write_text(json.dumps(b))
    for index in (0, 1):
        _refresh_response_binding(fixture, index)
    reordered = extract_vast_official_instance_charge(**kwargs)
    assert reordered["provider_instance_id"] == original["provider_instance_id"] == INSTANCE_A
    assert reordered["official_charge_usd"] == original["official_charge_usd"] == 0.123
    assert reordered["official_billing_response"]["source_index"] == 1
    assert original["official_billing_response"]["source_index"] == 0
