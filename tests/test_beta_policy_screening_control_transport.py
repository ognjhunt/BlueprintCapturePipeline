"""Signed screening control transport using a fake service."""

import json
from types import SimpleNamespace

import pytest
from blueprint_pipeline import beta_policy_screening_control_contracts as contract
from blueprint_pipeline import beta_policy_screening_control_transport as transport
from tests.test_beta_policy_screening_control_contracts import binding, response


def test_signed_request_uses_existing_control_plane_transport_and_actual_deadline(monkeypatch):
    from blueprint_pipeline import safe_outbound_http, task_evaluation_launch_webapp_sync
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://example.com/api/internal/pipeline/sync")
    monkeypatch.setattr(task_evaluation_launch_webapp_sync, "load_pipeline_sync_token", lambda: "fake-controller-token")
    monkeypatch.setattr(contract.time, "time", lambda: 1001)
    calls = []
    def request(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(body=json.dumps(response()).encode())
    monkeypatch.setattr(safe_outbound_http, "request", request)
    value = transport.request_control(binding=binding(), execution_deadline_epoch=2500.25)
    assert value["execution_deadline_at_ms"] == 2000_000
    url, call = calls[0]
    assert url == "https://example.com/api/internal/pipeline/beta-policy-screening-runs/campaign-1/execution-control"
    payload = json.loads(call["data"])
    assert payload == {"schema_version": "blueprint.beta_policy_screening_execution_control_request.v1",
                      **binding(), "execution_deadline_at_ms": 2500250}
    assert call["timeout_seconds"] == 10 and "X-Blueprint-Pipeline-Signature" in call["headers"]
    assert "fake-controller-token" not in call["data"].decode()


def test_current_response_cannot_extend_the_requested_smaller_deadline(monkeypatch):
    monkeypatch.setattr(contract.time, "time", lambda: 1001)
    with pytest.raises(ValueError, match="deadline_extended"):
        transport.request_control(binding=binding(), execution_deadline_epoch=1500,
                                  transport=lambda body: response(deadline=2000))
