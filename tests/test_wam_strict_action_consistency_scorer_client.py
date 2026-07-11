from __future__ import annotations

import json

import pytest

from blueprint_pipeline import wam_strict_action_consistency_scorer_client as client


def test_strict_scorer_client_uses_https_file_token_and_strict_request(
    tmp_path, monkeypatch
) -> None:
    token = tmp_path / "token"
    token.write_text("secret-token")
    request_payload = {
        "strict_action_aware_consistency": {"required": True},
        "rollouts": [{"rollout_id": "r1"}],
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {"status": "completed", "rollout_checks": [{"rollout_id": "r1"}]}
            ).encode()

    def fake_open(request, timeout, policy):
        assert request.full_url == "https://scorer.example/v1/score"
        assert request.headers["Authorization"] == "Bearer secret-token"
        assert json.loads(request.data) == request_payload
        assert timeout == 600.0
        assert policy.allowed_hosts == frozenset({"scorer.example"})
        return Response()

    monkeypatch.setattr(client.safe_outbound_http, "_open_with_policy", fake_open)
    result = client.score_via_service(
        request_payload,
        url="https://scorer.example/v1/score",
        token_file=token,
    )
    assert result["status"] == "completed"
    assert result["client_transport"]["raw_token_recorded"] is False


def test_strict_scorer_client_rejects_insecure_remote_and_non_strict_request(
    tmp_path,
) -> None:
    token = tmp_path / "token"
    token.write_text("x")
    with pytest.raises(ValueError, match="requires_https"):
        client.score_via_service({}, url="http://scorer.example", token_file=token)
    with pytest.raises(ValueError, match="strict_action_aware"):
        client.score_via_service({}, url="https://scorer.example", token_file=token)
