from __future__ import annotations

import hashlib
import hmac
import json
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import submit_task_evaluation_launch_via_webapp as submitter


SECRET = "launch-only-secret-value-0123456789abcdef"
DIGEST = "sha256:" + "a" * 64
PROFILE_DIGEST = "sha256:" + "b" * 64


def _request() -> dict[str, object]:
    return {
        "launch_id": "scene-840920-artifixer-launch-1",
        "run_id": "scene-840920-artifixer-run-1",
        "profile_id": "scene-840920-artifixer-profile-1",
        "profile_digest": PROFILE_DIGEST,
        "rights": {
            "scope": "noncommercial internal ADP-009 production qualification",
            "evidence": {"uri": "gs://evidence/rights.json", "digest": DIGEST},
        },
        "spend": {
            "max_spend_usd": 2.0,
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        },
        "confirm_execution": True,
    }


def _queue_receipt(request: dict[str, object], request_digest: str) -> dict[str, object]:
    return {
        "schema_version": "task_evaluation_launch_queue_receipt.v1",
        "status": "queued",
        "already_exists": False,
        "launch_id": request["launch_id"],
        "run_id": request["run_id"],
        "request_digest": request_digest,
        "launch_profile_id": request["profile_id"],
        "launch_profile_digest": request["profile_digest"],
        "queue_path": "/var/lib/blueprint/pipeline-control-plane/pending/request.json",
        "provider_mutation_performed": False,
    }


def _web_receipt(
    request: dict[str, object],
    *,
    request_digest: str = DIGEST,
    already_exists: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": "task_evaluation_launch_web_receipt.v1",
        "status": "queued_in_pipeline",
        "already_exists": already_exists,
        "launch_id": request["launch_id"],
        "run_id": request["run_id"],
        "request_digest": request_digest,
        "forward": {
            "status": "forwarded",
            "performed": True,
            "required": True,
            "endpoint_configured": True,
            "http_status": 202,
            "queue_receipt": _queue_receipt(request, request_digest),
            "pipeline_intake_status": "accepted",
        },
        "provider_mutation_performed_inside_web_request": False,
        "submission_channel": "production_webapp_service_api",
    }


class _Response:
    def __init__(self, status: int, payload: dict[str, object]) -> None:
        self.status = status
        self.payload = json.dumps(payload, separators=(",", ":")).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, amount: int = -1) -> bytes:
        return self.payload if amount < 0 else self.payload[:amount]


def _files(tmp_path: Path) -> tuple[Path, Path, Path, bytes, dict[str, object]]:
    request = _request()
    body = (json.dumps(request, indent=2) + "\n").encode()
    request_path = tmp_path / "request.json"
    request_path.write_bytes(body)
    secret_path = tmp_path / "secret"
    secret_path.write_text(SECRET + "\n", encoding="utf-8")
    secret_path.chmod(0o440)
    return request_path, secret_path, tmp_path / "receipt.json", body, request


def test_signs_and_posts_the_exact_request_bytes(monkeypatch, tmp_path, capsys) -> None:
    request_path, secret_path, receipt_path, body, request = _files(tmp_path)
    observed: dict[str, object] = {}

    def urlopen(http_request, timeout):  # type: ignore[no-untyped-def]
        observed["body"] = http_request.data
        observed["headers"] = dict(http_request.header_items())
        observed["timeout"] = timeout
        return _Response(202, _web_receipt(request))

    monkeypatch.setattr(submitter.urllib.request, "urlopen", urlopen)
    assert (
        submitter.main(
            [
                "--request",
                str(request_path),
                "--secret-file",
                str(secret_path),
                "--receipt-out",
                str(receipt_path),
            ]
        )
        == 0
    )

    assert observed["body"] == body
    headers = {key.lower(): value for key, value in observed["headers"].items()}
    assert headers["idempotency-key"] == request["launch_id"]
    assert headers["x-blueprint-launch-client-id"] == "blueprint-production-runner"
    timestamp = headers["x-blueprint-launch-timestamp"]
    nonce = headers["x-blueprint-launch-nonce"]
    expected = hmac.new(
        SECRET.encode(),
        f"{timestamp}.blueprint-production-runner.{nonce}.".encode() + body,
        "sha256",
    ).hexdigest()
    assert headers["x-blueprint-launch-signature"] == f"sha256={expected}"

    evidence = json.loads(receipt_path.read_text())
    assert evidence["status"] == "submitted"
    assert evidence["submitted_body_digest"] == ("sha256:" + hashlib.sha256(body).hexdigest())
    assert evidence["webapp_request_digest"] == DIGEST
    assert evidence["request_timestamp"] == timestamp
    assert evidence["request_nonce"] == nonce
    assert evidence["idempotency_key"] == request["launch_id"]
    assert evidence["webapp_receipt"]["launch_id"] == request["launch_id"]
    assert stat_mode(receipt_path) == 0o440
    assert SECRET not in receipt_path.read_text()
    assert SECRET not in capsys.readouterr().out


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_body_tamper_changes_the_signature_and_receipt_mismatch_fails() -> None:
    request = _request()
    original = json.dumps(request, separators=(",", ":")).encode()
    changed = original.replace(b"run-1", b"run-2")
    headers = submitter.signed_headers(
        secret=SECRET.encode(),
        body=original,
        timestamp="2026-08-16T20:30:00.000+00:00",
        nonce="0123456789abcdef",
        launch_id=str(request["launch_id"]),
    )
    changed_headers = submitter.signed_headers(
        secret=SECRET.encode(),
        body=changed,
        timestamp="2026-08-16T20:30:00.000+00:00",
        nonce="0123456789abcdef",
        launch_id=str(request["launch_id"]),
    )
    assert headers[submitter.SIGNATURE_HEADER] != changed_headers[submitter.SIGNATURE_HEADER]

    receipt = _web_receipt(request)
    receipt["run_id"] = "different-run"
    with pytest.raises(
        submitter.WebAppLaunchSubmissionError,
        match="webapp_receipt_run_id_mismatch",
    ):
        submitter.validate_webapp_receipt(
            status_code=202,
            response_body=json.dumps(receipt).encode(),
            request=request,
            allow_replay=False,
        )


def test_rejects_queue_request_digest_mismatch() -> None:
    request = _request()
    receipt = _web_receipt(request)
    receipt["forward"]["queue_receipt"]["request_digest"] = (  # type: ignore[index]
        "sha256:" + "c" * 64
    )
    with pytest.raises(
        submitter.WebAppLaunchSubmissionError,
        match="webapp_queue_receipt_request_digest_mismatch",
    ):
        submitter.validate_webapp_receipt(
            status_code=202,
            response_body=json.dumps(receipt).encode(),
            request=request,
            allow_replay=False,
        )


def test_http_200_requires_explicit_replay_mode() -> None:
    request = _request()
    replay = _web_receipt(request, already_exists=True)
    response = json.dumps(replay).encode()
    with pytest.raises(
        submitter.WebAppLaunchSubmissionError,
        match="webapp_replay_requires_explicit_flag",
    ):
        submitter.validate_webapp_receipt(
            status_code=200,
            response_body=response,
            request=request,
            allow_replay=False,
        )
    assert (
        submitter.validate_webapp_receipt(
            status_code=200,
            response_body=response,
            request=request,
            allow_replay=True,
        )["already_exists"]
        is True
    )


def test_http_error_and_secret_never_leak(monkeypatch, tmp_path, capsys) -> None:
    request_path, secret_path, receipt_path, _body, _request_value = _files(tmp_path)

    def urlopen(http_request, timeout):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(
            http_request.full_url,
            401,
            "rejected " + SECRET,
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(submitter.urllib.request, "urlopen", urlopen)
    assert (
        submitter.main(
            [
                "--request",
                str(request_path),
                "--secret-file",
                str(secret_path),
                "--receipt-out",
                str(receipt_path),
            ]
        )
        == 2
    )
    output = capsys.readouterr().out
    assert json.loads(output)["blockers"] == ["webapp_http_error_401"]
    assert SECRET not in output
    assert not receipt_path.exists()


def test_receipt_with_secret_shaped_extra_field_is_refused(monkeypatch, tmp_path, capsys) -> None:
    request_path, secret_path, receipt_path, _body, request = _files(tmp_path)
    bad_receipt = _web_receipt(request)
    bad_receipt["forward"]["queue_receipt"]["queue_path"] = SECRET  # type: ignore[index]
    monkeypatch.setattr(
        submitter.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(202, bad_receipt),
    )
    assert (
        submitter.main(
            [
                "--request",
                str(request_path),
                "--secret-file",
                str(secret_path),
                "--receipt-out",
                str(receipt_path),
            ]
        )
        == 2
    )
    output = capsys.readouterr().out
    assert "webapp_response_reflected_secret" in output
    assert SECRET not in output
    assert not receipt_path.exists()


def test_existing_receipt_fails_before_network(monkeypatch, tmp_path) -> None:
    request_path, secret_path, receipt_path, _body, _request_value = _files(tmp_path)
    receipt_path.write_text("user-owned\n", encoding="utf-8")
    called = False

    def urlopen(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        raise AssertionError("network must not run")

    monkeypatch.setattr(submitter.urllib.request, "urlopen", urlopen)
    assert (
        submitter.main(
            [
                "--request",
                str(request_path),
                "--secret-file",
                str(secret_path),
                "--receipt-out",
                str(receipt_path),
            ]
        )
        == 2
    )
    assert called is False
    assert receipt_path.read_text() == "user-owned\n"


def test_secret_file_may_be_group_readable_but_not_group_writable(tmp_path) -> None:
    secret_path = tmp_path / "secret"
    secret_path.write_text(SECRET, encoding="utf-8")
    secret_path.chmod(0o440)
    assert submitter.read_private_secret_file(secret_path) == SECRET.encode()
    secret_path.chmod(0o460)
    with pytest.raises(
        submitter.WebAppLaunchSubmissionError,
        match="launch_submit_secret_file_not_private",
    ):
        submitter.read_private_secret_file(secret_path)


def test_only_https_endpoints_are_admitted() -> None:
    with pytest.raises(
        submitter.WebAppLaunchSubmissionError,
        match="launch_submit_endpoint_not_https",
    ):
        submitter.post_signed_launch(
            endpoint="http://tryblueprint.io/api/internal/task-evaluation-launch-submissions",
            headers={},
            body=b"{}",
            timeout_seconds=1,
        )
