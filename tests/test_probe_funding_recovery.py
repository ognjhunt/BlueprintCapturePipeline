"""Historical funding readback never replenishes a probe's grant or dispatches."""

import json

import pytest

from blueprint_pipeline import probe_funding_recovery as funding
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest


def request():
    return {"capture_id": "capture", "operation": "preparation-spend", "payload": {
        "request_id": "request", "scene_id": "site-request", "spend": {
            "task_context_digest": "sha256:" + "a" * 64,
            "allocation_binding_digest": "sha256:" + "b" * 64,
            "resource_class": "gpu_render", "provider": "vast", "maximum_cost_usd": 1., "request_count": 1}}}


def observation(command, *, status="reserved", **extra):
    return {"schema_version": "website_preparation_spend_observation.v1", "capture_id": "capture",
            "request_id": "request", "scene_id": "site-request", "status": status,
            "command_digest": cross_runtime_canonical_digest(command),
            "allocation_binding_digest": command["allocation_binding_digest"], "dispatch_authorized": False,
            "observed_at_epoch": 1000,
            **({"admission": {**command, "status": "admitted", "blockers": [],
                "schema_version": "paid_lane_admission.v1", "external_disclosure_allowed": True}}
               if status == "reserved" else {}), **extra}


def test_request_is_exclusive_and_refusal_is_exact_and_typed(tmp_path):
    value = request()
    funding.retain_request(tmp_path, value)
    with pytest.raises(FileExistsError):
        funding.retain_request(tmp_path, value)
    assert funding.funding_state(tmp_path)["status"] == "unknown"
    funding.retain_refusal(tmp_path, "website_control_preparation-spend_http_503:unclassified")
    assert funding.funding_state(tmp_path)["status"] == "unknown"
    funding.retain_refusal(tmp_path, "website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted")
    assert funding.funding_state(tmp_path)["status"] == "refused"
    refused = json.loads((tmp_path / funding.REFUSAL_FILENAME).read_text())
    refused["request_digest"] = "wrong"
    (tmp_path / funding.REFUSAL_FILENAME).write_text(json.dumps(refused))
    with pytest.raises(ValueError, match="refusal_invalid"):
        funding.funding_state(tmp_path)


@pytest.mark.parametrize("fault", ["capture_id", "command_digest", "allocation_binding_digest", "dispatch_authorized",
                                  "admission", "observed_at_epoch"])
def test_readback_must_bind_exact_request_and_never_be_dispatch_authority(tmp_path, fault):
    funding.retain_request(tmp_path, request())
    retained = funding.request_intent(tmp_path)
    command = retained["payload"]["spend"]
    value = observation(command)
    value[fault] = True if fault == "dispatch_authorized" else -1 if fault == "observed_at_epoch" else {}
    with pytest.raises(ValueError, match="observation"):
        funding.validate_observation(value, request=retained, now=1001)


def test_not_reserved_snapshot_keeps_unknown_slot_and_readback_has_bounded_backoff(tmp_path):
    funding.retain_request(tmp_path, request())
    calls = []
    def reader(**kwargs):
        calls.append(kwargs)
        assert kwargs["payload"]["create"] is False
        return observation(kwargs["payload"]["spend"], status="not_reserved")
    first = funding.reconcile_funding(tmp_path, reader=reader, now=1001)
    assert first["status"] == "unknown" and first["funded"] is None
    assert first["next_retry_at_epoch"] == 1061 and len(calls) == 1
    assert funding.reconcile_funding(tmp_path, reader=reader, now=1060) == first
    second = funding.reconcile_funding(tmp_path, reader=reader, now=1061)
    assert second["next_retry_at_epoch"] == 1181 and len(calls) == 2
    assert not (tmp_path / funding.FUNDING_FILENAME).exists()


def test_missing_readback_cannot_synthesize_no_funding(tmp_path):
    funding.retain_request(tmp_path, request())
    def unavailable(**kwargs):
        raise OSError("no connection")
    result = funding.reconcile_funding(tmp_path, reader=unavailable, now=1001)
    assert result["status"] == "unknown" and result["funded"] is None
    assert result["last_remote_status"] == "unavailable"
    assert not (tmp_path / funding.FUNDING_FILENAME).exists()


def test_default_readback_is_disabled_until_server_contract_verified(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_context
    funding.retain_request(tmp_path, request())
    calls = []
    def send(**kwargs):
        calls.append(kwargs)
        return observation(kwargs["payload"]["spend"])
    monkeypatch.setattr(website_task_context, "website_webapp_request", send)
    monkeypatch.delenv(funding.READBACK_ENV, raising=False)
    for _ in range(2):
        result = funding.reconcile_funding(tmp_path, now=1001)
        assert result["status"] == "unknown" and result["funded"] is None
        assert result["last_remote_status"] == "contract_not_enabled"
    assert calls == [] and not (tmp_path / funding.RECONCILIATION_FILENAME).exists()
    monkeypatch.setenv(funding.READBACK_ENV, "true")
    result = funding.reconcile_funding(tmp_path, now=1001)
    assert len(calls) == 1 and calls[0]["payload"]["create"] is False
    assert result["funded"] is True and result["status"] == "unknown"
    assert not (tmp_path / funding.FUNDING_FILENAME).exists()
