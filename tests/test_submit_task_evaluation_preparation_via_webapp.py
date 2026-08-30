from __future__ import annotations

import hashlib
import hmac
import json
import urllib.error
from pathlib import Path

import pytest
import rfc8785

from scripts import submit_task_evaluation_preparation_via_webapp as submitter


SECRET = "preparation-service-secret-0123456789abcdef"


def _digest(value: dict[str, object], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    body = rfc8785.dumps(unsigned)
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _request() -> dict[str, object]:
    return {
        "schema_version": "task_evaluation_launch_preparation_request.v1",
        "run_mode": "scene_configuration",
        "expected_production_commit": "a" * 40,
        "preparation_id": "scene-839873-configuration-001",
        "team_namespace": "blueprint-adp",
        "run_id": "scene-839873-configuration-run-001",
        "scene": {"mode": "configure_source_scene"},
        "construction": {"mode": "production_recipe"},
        "task": {"kind": "rigid_relocation"},
        "sensors": {"configuration": "bound"},
        "runtime": {"identity": "native-arena"},
        "execution_adapter": {"kind": "scene_configuration_pipeline"},
        "publication": {"input_namespace": "scene-839873"},
        "spend": {"retry_cap": 0},
    }


def _intake_receipt(request: dict[str, object]) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema_version": "task_evaluation_launch_preparation_intake_receipt.v1",
        "status": "queued_for_no_spend_preparation",
        "accepted": True,
        "already_exists": False,
        "preparation_id": request["preparation_id"],
        "run_id": request["run_id"],
        "team_namespace": request["team_namespace"],
        "request_digest": _digest(request, "request_digest"),
        "expected_production_commit": request["expected_production_commit"],
        "provider_mutation_performed_inside_http_request": False,
        "catalog_mutation_performed_inside_http_request": False,
        "paid_execution_requested": False,
        "canonical_allocator_required_for_later_execution": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = _digest(receipt, "receipt_digest")
    return receipt


def _web_receipt(
    request: dict[str, object], *, already_exists: bool = False
) -> dict[str, object]:
    return {
        "schema_version": "task_evaluation_launch_preparation_web_receipt.v1",
        "status": "queued_for_no_spend_preparation",
        "already_exists": already_exists,
        "preparation_id": request["preparation_id"],
        "run_id": request["run_id"],
        "team_namespace": request["team_namespace"],
        "request_digest": _digest(request, "request_digest"),
        "expected_production_commit": request["expected_production_commit"],
        "pipeline": {
            "status": "forwarded",
            "performed": True,
            "http_status": 202,
            "receipt": _intake_receipt(request),
        },
        "provider_mutation_performed_inside_web_request": False,
        "catalog_mutation_performed_inside_web_request": False,
        "paid_execution_requested": False,
        "preparation_is_not_execution": True,
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
    request_path = tmp_path / "preparation.json"
    request_path.write_bytes(body)
    secret_path = tmp_path / "secret"
    secret_path.write_text(SECRET + "\n", encoding="utf-8")
    secret_path.chmod(0o440)
    return request_path, secret_path, tmp_path / "receipt.json", body, request


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_signs_and_posts_exact_preparation_bytes(monkeypatch, tmp_path, capsys) -> None:
    request_path, secret_path, receipt_path, body, request = _files(tmp_path)
    observed: dict[str, object] = {}

    def urlopen(http_request, timeout):  # type: ignore[no-untyped-def]
        observed["body"] = http_request.data
        observed["headers"] = dict(http_request.header_items())
        observed["timeout"] = timeout
        observed["reserved"] = receipt_path.is_file()
        observed["mode"] = _mode(receipt_path)
        return _Response(202, _web_receipt(request))

    monkeypatch.setattr(submitter.urllib.request, "urlopen", urlopen)
    assert submitter.main([
        "--request", str(request_path),
        "--secret-file", str(secret_path),
        "--receipt-out", str(receipt_path),
    ]) == 0

    assert observed["body"] == body
    assert observed["reserved"] is True
    assert observed["mode"] == 0o000
    headers = {key.lower(): value for key, value in observed["headers"].items()}
    assert headers["idempotency-key"] == request["preparation_id"]
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
    assert evidence["submitted_body_digest"] == (
        "sha256:" + hashlib.sha256(body).hexdigest()
    )
    assert evidence["webapp_request_digest"] == _digest(request, "request_digest")
    assert evidence["idempotency_key"] == request["preparation_id"]
    assert evidence["provider_mutation_performed_by_this_tool"] is False
    assert evidence["catalog_mutation_performed_by_this_tool"] is False
    assert evidence["paid_execution_requested_by_this_tool"] is False
    assert _mode(receipt_path) == 0o440
    assert SECRET not in receipt_path.read_text()
    assert SECRET not in capsys.readouterr().out


def test_receipt_identity_or_digest_mismatch_fails() -> None:
    request = _request()
    receipt = _web_receipt(request)
    receipt["request_digest"] = "sha256:" + "f" * 64
    with pytest.raises(
        submitter.WebAppPreparationSubmissionError,
        match="webapp_preparation_receipt_request_digest_mismatch",
    ):
        submitter.validate_webapp_preparation_receipt(
            status_code=202,
            response_body=json.dumps(receipt).encode(),
            request=request,
            allow_replay=False,
        )


def test_cross_runtime_digest_uses_ecmascript_number_serialization() -> None:
    value = {"whole_float": 1.0, "fraction": 0.8, "request_digest": ""}
    assert submitter._canonical_artifact_digest(value, "request_digest") == (
        "sha256:" + hashlib.sha256(b'{"fraction":0.8,"whole_float":1}').hexdigest()
    )


def test_intake_receipt_digest_mismatch_fails() -> None:
    request = _request()
    receipt = _web_receipt(request)
    receipt["pipeline"]["receipt"]["receipt_digest"] = "sha256:" + "f" * 64  # type: ignore[index]
    with pytest.raises(
        submitter.WebAppPreparationSubmissionError,
        match="webapp_preparation_intake_receipt_digest_mismatch",
    ):
        submitter.validate_webapp_preparation_receipt(
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
        submitter.WebAppPreparationSubmissionError,
        match="webapp_preparation_replay_requires_explicit_flag",
    ):
        submitter.validate_webapp_preparation_receipt(
            status_code=200,
            response_body=response,
            request=request,
            allow_replay=False,
        )
    assert submitter.validate_webapp_preparation_receipt(
        status_code=200,
        response_body=response,
        request=request,
        allow_replay=True,
    )["already_exists"] is True


def test_unknown_request_field_fails_before_network(monkeypatch, tmp_path) -> None:
    request_path, secret_path, receipt_path, _body, request = _files(tmp_path)
    request["unexpected"] = True
    request_path.write_text(json.dumps(request), encoding="utf-8")
    called = False

    def urlopen(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        raise AssertionError("network must not run")

    monkeypatch.setattr(submitter.urllib.request, "urlopen", urlopen)
    assert submitter.main([
        "--request", str(request_path),
        "--secret-file", str(secret_path),
        "--receipt-out", str(receipt_path),
    ]) == 2
    assert called is False
    assert not receipt_path.exists()


def test_scene_configuration_accepts_explicit_paused_ungraded_review_override(
    tmp_path,
) -> None:
    request = _request()
    request["appearance_review_override"] = {
        "mode": "paused_ungraded",
        "scope": "artifixer_appearance_only",
        "ungraded_publication_acknowledged": True,
        "review_provider_call_permitted": False,
        "warning_label": "Visual review paused - appearance ungraded",
    }
    body = (json.dumps(request, indent=2) + "\n").encode()
    request_path = tmp_path / "paused-ungraded-preparation.json"
    request_path.write_bytes(body)

    parsed, exact_body = submitter.read_exact_preparation_request(request_path)

    assert parsed == request
    assert exact_body == body


def test_existing_receipt_fails_before_network(monkeypatch, tmp_path) -> None:
    request_path, secret_path, receipt_path, _body, _request = _files(tmp_path)
    receipt_path.write_text("user-owned\n", encoding="utf-8")
    called = False

    def urlopen(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        raise AssertionError("network must not run")

    monkeypatch.setattr(submitter.urllib.request, "urlopen", urlopen)
    assert submitter.main([
        "--request", str(request_path),
        "--secret-file", str(secret_path),
        "--receipt-out", str(receipt_path),
    ]) == 2
    assert called is False
    assert receipt_path.read_text() == "user-owned\n"


def test_http_error_does_not_leak_secret(monkeypatch, tmp_path, capsys) -> None:
    request_path, secret_path, receipt_path, _body, _request = _files(tmp_path)

    def urlopen(http_request, timeout):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(
            http_request.full_url,
            401,
            "rejected " + SECRET,
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(submitter.urllib.request, "urlopen", urlopen)
    assert submitter.main([
        "--request", str(request_path),
        "--secret-file", str(secret_path),
        "--receipt-out", str(receipt_path),
    ]) == 2
    output = capsys.readouterr().out
    assert json.loads(output)["blockers"] == ["webapp_preparation_http_error_401"]
    assert SECRET not in output
    assert not receipt_path.exists()
