from __future__ import annotations

import hmac
import json
from datetime import datetime, timezone

import pytest

from scripts import submit_task_evaluation_launch as submitter
from scripts import submit_task_evaluation_launch_via_webapp as canonical_webapp


NOW = datetime(2026, 8, 24, 0, 30, 0, tzinfo=timezone.utc)
SHA = "sha256:" + "a" * 64
WEBAPP_REQUEST_KEYS = {
    "confirm_execution",
    "launch_id",
    "run_id",
    "profile_id",
    "profile_digest",
    "rights",
    "spend",
}


def _profile() -> dict[str, object]:
    return {
        "profile_id": "arena-controls-live-scene-840920-c84",
        "profile_digest": "sha256:" + "c" * 64,
        # These fields belong to the published profile. The WebApp looks the
        # profile up by ID+digest; the launch-only request must not duplicate
        # them or try to construct Pipeline's expanded intake request itself.
        "source_bundle": {"digest": "sha256:" + "d" * 64},
        "evaluation_run_spec": {"digest": "sha256:" + "e" * 64},
        "required_controls": {"retry_cap": 0},
        "claim_ceiling": "development_only",
    }


def _request() -> dict[str, object]:
    return submitter.build_launch_request(
        profile=_profile(),
        launch_id="adp-arena-controls-scene-840920-c84",
        run_id="scene-840920-task-a-c84",
        rights_scope="internal ADP simulator evaluation",
        rights_uri="gs://blueprint-evidence/rights.json",
        rights_digest=SHA,
        max_spend_usd=2.0,
        authority_window_hours=3.0,
        now=NOW,
    )


def test_builds_exact_webapp_launch_only_body() -> None:
    request = _request()

    assert set(request) == WEBAPP_REQUEST_KEYS
    assert request == {
        "confirm_execution": True,
        "launch_id": "adp-arena-controls-scene-840920-c84",
        "run_id": "scene-840920-task-a-c84",
        "profile_id": "arena-controls-live-scene-840920-c84",
        "profile_digest": "sha256:" + "c" * 64,
        "rights": {
            "scope": "internal ADP simulator evaluation",
            "evidence": {
                "uri": "gs://blueprint-evidence/rights.json",
                "digest": SHA,
            },
        },
        "spend": {
            "max_spend_usd": 2.0,
            "expires_at": "2026-08-24T03:30:00.000Z",
        },
    }

    # These are Pipeline intake fields. Sending any of them to the strict
    # WebApp schema is the regression that caused the production rejection.
    obsolete_fields = {
        "schema_version",
        "idempotency_key",
        "launch_profile_id",
        "launch_profile_digest",
        "source_bundle",
        "evaluation_run_spec",
        "required_controls",
        "claim_ceiling",
        "authorization",
        "request_digest",
    }
    assert not obsolete_fields.intersection(request)


def test_body_passes_strict_webapp_client_contract_and_full_intake_fails(
    tmp_path,
) -> None:
    request = _request()
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    validated, _body = canonical_webapp.read_exact_launch_request(request_path)
    assert validated == request

    # This represents the obsolete helper shape: the strict WebApp contract
    # must refuse expanded Pipeline intake fields rather than silently accept
    # two different authorities at one public boundary.
    request_path.write_text(
        json.dumps(
            {
                **request,
                "schema_version": "task_evaluation_launch_request.v1",
                "authorization": {"execution": {"approved": True}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        canonical_webapp.WebAppLaunchSubmissionError,
        match="launch_request_fields_invalid",
    ):
        canonical_webapp.read_exact_launch_request(request_path)


def test_signs_exact_webapp_canonical_bytes_and_sets_idempotency() -> None:
    body = b'{"confirm_execution":true,"launch_id":"launch-1"}'
    headers = submitter.signed_headers(
        secret="s3cret",
        body=body,
        now=NOW,
        nonce="0123456789abcdef",
        launch_id="launch-1",
    )

    timestamp = headers[submitter.TIMESTAMP_HEADER]
    expected = hmac.new(
        b"s3cret",
        (
            f"{timestamp}.blueprint-production-runner.0123456789abcdef."
        ).encode()
        + body,
        "sha256",
    ).hexdigest()
    assert headers == {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Blueprint-Launch-Timestamp": timestamp,
        "X-Blueprint-Launch-Nonce": "0123456789abcdef",
        "X-Blueprint-Launch-Client-Id": "blueprint-production-runner",
        "X-Blueprint-Launch-Signature": f"sha256={expected}",
        "Idempotency-Key": "launch-1",
    }
    assert not any(name.startswith("X-Blueprint-Pipeline-") for name in headers)

    with pytest.raises(submitter.LaunchSubmissionError):
        submitter.signed_headers(
            secret="",
            body=body,
            now=NOW,
            nonce="0123456789abcdef",
            launch_id="launch-1",
        )


def test_main_posts_and_records_the_same_exact_webapp_body(
    capsys, tmp_path, monkeypatch
) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile()), encoding="utf-8")
    secret_path = tmp_path / "secret"
    secret_path.write_text("super-secret-value\n", encoding="utf-8")
    request_path = tmp_path / "request.json"
    observed: dict[str, object] = {}

    def fake_submit(**kwargs):  # type: ignore[no-untyped-def]
        observed.update(kwargs)
        return {"http_status": 202, "body": {"status": "queued_in_pipeline"}}

    monkeypatch.setattr(submitter, "submit", fake_submit)
    exit_code = submitter.main(
        [
            "--profile",
            str(profile_path),
            "--endpoint",
            "http://127.0.0.1:8765/api/internal/task-evaluation-launch-submissions",
            "--secret-file",
            str(secret_path),
            "--launch-id",
            "launch-1",
            "--run-id",
            "run-1",
            "--rights-scope",
            "internal ADP simulator evaluation",
            "--rights-uri",
            "gs://blueprint-evidence/rights.json",
            "--rights-digest",
            SHA,
            "--max-spend-usd",
            "2",
            "--request-out",
            str(request_path),
        ]
    )

    assert exit_code == 0
    submitted_body = observed["body"]
    assert isinstance(submitted_body, bytes)
    assert submitted_body == request_path.read_bytes()
    submitted_request = json.loads(submitted_body)
    assert set(submitted_request) == WEBAPP_REQUEST_KEYS
    headers = observed["headers"]
    assert isinstance(headers, dict)
    assert headers["Idempotency-Key"] == submitted_request["launch_id"]
    assert headers["X-Blueprint-Launch-Client-Id"] == "blueprint-production-runner"
    assert "X-Blueprint-Pipeline-Signature" not in headers

    output = capsys.readouterr().out
    assert "super-secret-value" not in output
    result = json.loads(output)
    assert result["status"] == "submitted"
    assert result["client_id"] == "blueprint-production-runner"
    assert result["provider_mutation_performed_by_this_tool"] is False


def test_rejected_submission_never_echoes_secret(capsys, tmp_path, monkeypatch) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile()), encoding="utf-8")
    secret_path = tmp_path / "secret"
    secret_path.write_text("super-secret-value\n", encoding="utf-8")

    monkeypatch.setattr(
        submitter,
        "submit",
        lambda **kwargs: {"http_status": 401, "body": "invalid WebApp signature"},
    )
    exit_code = submitter.main(
        [
            "--profile",
            str(profile_path),
            "--endpoint",
            "http://127.0.0.1:8765/api/internal/task-evaluation-launch-submissions",
            "--secret-file",
            str(secret_path),
            "--launch-id",
            "launch-1",
            "--run-id",
            "run-1",
            "--rights-scope",
            "internal ADP simulator evaluation",
            "--rights-uri",
            "gs://blueprint-evidence/rights.json",
            "--rights-digest",
            SHA,
            "--max-spend-usd",
            "2",
        ]
    )

    assert exit_code == 2
    output = capsys.readouterr().out
    assert "super-secret-value" not in output
    assert json.loads(output)["status"] == "rejected"
