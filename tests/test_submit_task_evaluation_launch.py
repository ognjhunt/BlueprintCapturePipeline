from __future__ import annotations

import hmac
import json
from datetime import datetime, timezone

import pytest

from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    canonical_digest,
    validate_launch_request,
)
from scripts import submit_task_evaluation_launch as submitter

NOW = datetime(2026, 8, 11, 15, 0, 0, tzinfo=timezone.utc)
SHA = "sha256:" + "a" * 64


def _profile() -> dict:
    return {
        "profile_id": "adp009d-840313-franka-controls-" + "b" * 40,
        "profile_digest": "sha256:" + "c" * 64,
        "source_bundle": {
            "bundle_id": "adp009d-840313-interiorgs-sage-v1",
            "source_kind": "interiorgs_sage",
            "uri": "https://example.invalid/bundle.json",
            "digest": "sha256:" + "d" * 64,
        },
        "evaluation_run_spec": {
            "uri": "https://example.invalid/spec.json",
            "digest": "sha256:" + "e" * 64,
        },
        "required_controls": {
            "canonical_allocator": "python -m blueprint_pipeline.paid_resource_allocator gpu-canary",
            "secret_profile_id": "canonical-vast-adp",
            "watchdog_required": True,
            "artifact_storage_required": True,
            "teardown_required": True,
            "provider_zero_required": True,
            "webapp_status_sync_required": True,
            "retry_cap": 0,
        },
        "claim_ceiling": "development_only",
    }


def _request() -> dict:
    return submitter.build_launch_request(
        profile=_profile(),
        launch_id="adp009d-840313-controls-web-1",
        run_id="adp009d-840313-controls-run-1",
        actor_id="temporary-task-evaluation-launch-lab",
        actor_role="ops",
        rights_scope="interiorgs_sage_simulator_evaluation",
        rights_uri="https://example.invalid/rights.json",
        rights_digest=SHA,
        max_spend_usd=6.0,
        authority_window_hours=3.0,
        now=NOW,
    )


def test_built_request_passes_the_dispatcher_contract() -> None:
    """The tool must produce a request the production dispatcher accepts, or it
    is not exercising the same path the website uses."""
    request = _request()

    assert validate_launch_request(request) == []
    assert request["request_digest"] == canonical_digest(
        request, digest_field="request_digest"
    )
    assert request["authorization"]["spend"]["max_spend_usd"] == 6.0
    assert request["authorization"]["spend"]["expires_at"] == "2026-08-11T18:00:00.000Z"
    # The request may never carry allocator arguments or an execute flag: the
    # dispatcher owns both.
    serialized = json.dumps(request)
    assert "--execute" not in serialized
    assert "allocator" not in request


def test_signature_matches_the_intake_canonical_payload() -> None:
    """The intake signs `{timestamp}.{client_id}.{nonce}.` plus the raw body and
    parses the timestamp with fromisoformat, so epoch seconds are rejected."""
    body = b'{"launch_id":"x"}'
    headers = submitter.signed_headers(
        secret="s3cret", client_id="blueprint-webapp", body=body, now=NOW, nonce="n" * 12
    )

    timestamp = headers[submitter.TIMESTAMP_HEADER]
    assert datetime.fromisoformat(timestamp) == NOW
    expected = hmac.new(
        b"s3cret",
        f"{timestamp}.blueprint-webapp.{'n' * 12}.".encode() + body,
        "sha256",
    ).hexdigest()
    assert headers[submitter.SIGNATURE_HEADER] == f"sha256={expected}"
    # Nonce must satisfy the intake's character and length rule.
    nonce = headers[submitter.NONCE_HEADER]
    assert 8 <= len(nonce) <= 160

    with pytest.raises(submitter.LaunchSubmissionError):
        submitter.signed_headers(
            secret="", client_id="c", body=body, now=NOW, nonce="n" * 12
        )


def test_submission_never_echoes_the_signing_secret(capsys, tmp_path, monkeypatch) -> None:
    """A launch tool that prints its secret on failure would leak it into logs."""
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile()), encoding="utf-8")
    secret_path = tmp_path / "secret"
    secret_path.write_text("super-secret-value\n", encoding="utf-8")

    monkeypatch.setattr(
        submitter,
        "submit",
        lambda **kwargs: {"http_status": 401, "body": "invalid intake signature"},
    )
    exit_code = submitter.main(
        [
            "--profile", str(profile_path),
            "--endpoint", "http://127.0.0.1:8765/api/live-pipeline/task-evaluation-launches",
            "--secret-file", str(secret_path),
            "--launch-id", "launch-1",
            "--run-id", "run-1",
            "--actor-id", "ops-1",
            "--rights-scope", "interiorgs_sage_simulator_evaluation",
            "--rights-uri", "https://example.invalid/rights.json",
            "--rights-digest", SHA,
            "--max-spend-usd", "6",
        ]
    )

    assert exit_code == 2
    output = capsys.readouterr().out
    assert "super-secret-value" not in output
    assert json.loads(output)["status"] == "rejected"
    assert json.loads(output)["provider_mutation_performed_by_this_tool"] is False
