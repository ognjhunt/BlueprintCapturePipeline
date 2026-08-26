from __future__ import annotations

import hmac
import json
from pathlib import Path

from scripts import read_task_evaluation_preparation_status_via_webapp as reader
from scripts import submit_task_evaluation_preparation_via_webapp as submitter


SECRET = "preparation-service-secret-0123456789abcdef"


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


def _web_status(
    request: dict[str, object], state: str, *, blockers: list[str] | None = None
) -> dict[str, object]:
    digest = submitter._canonical_artifact_digest(request, "request_digest")
    pipeline: dict[str, object] = {
        "schema_version": "task_evaluation_launch_preparation_status.v1",
        "status": state,
        "preparation_id": request["preparation_id"],
        "run_mode": request["run_mode"],
        "run_id": request["run_id"],
        "team_namespace": request["team_namespace"],
        "expected_production_commit": request["expected_production_commit"],
        "request_digest": digest,
        "provider_mutation_performed_by_status_read": False,
        "provider_mutation_performed_by_worker": False,
        "catalog_mutation_performed_by_worker": False,
        "paid_execution_requested": False,
        "blockers": blockers or [],
    }
    if state == "materialized":
        pipeline.update({
            "source_commit": request["expected_production_commit"],
            "result_digest": "sha256:" + "b" * 64,
            "reference_count": 29,
            "full_byte_service_account_readback_passed": True,
        })
    return {
        "schema_version": "task_evaluation_launch_preparation_web_status.v1",
        "preparation_id": request["preparation_id"],
        "run_id": request["run_id"],
        "team_namespace": request["team_namespace"],
        "request_digest": digest,
        "expected_production_commit": request["expected_production_commit"],
        "state": state,
        "pipeline": pipeline,
        "provider_mutation_performed_by_status_read": False,
        "paid_execution_requested": False,
        "preparation_is_not_execution": True,
    }


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.status = 200
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
    return request_path, secret_path, tmp_path / "status.json", body, request


def test_polls_signed_website_status_until_full_byte_materialization(
    monkeypatch, tmp_path
) -> None:
    request_path, secret_path, receipt_path, _body, request = _files(tmp_path)
    responses = iter([_web_status(request, "processing"), _web_status(request, "materialized")])
    observed: list[dict[str, object]] = []

    def urlopen(http_request, timeout):  # type: ignore[no-untyped-def]
        observed.append({
            "method": http_request.method,
            "url": http_request.full_url,
            "headers": dict(http_request.header_items()),
            "timeout": timeout,
        })
        return _Response(next(responses))

    monkeypatch.setattr(reader.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(reader.time, "sleep", lambda _seconds: None)
    assert reader.main([
        "--request", str(request_path),
        "--secret-file", str(secret_path),
        "--receipt-out", str(receipt_path),
        "--poll-interval-seconds", "0.001",
    ]) == 0
    assert len(observed) == 2
    for call in observed:
        assert call["method"] == "GET"
        assert str(call["url"]).endswith("/scene-839873-configuration-001")
        headers = {key.lower(): value for key, value in call["headers"].items()}
        timestamp = headers["x-blueprint-launch-timestamp"]
        nonce = headers["x-blueprint-launch-nonce"]
        expected = hmac.new(
            SECRET.encode(),
            f"{timestamp}.blueprint-production-runner.{nonce}.".encode(),
            "sha256",
        ).hexdigest()
        assert headers["x-blueprint-launch-signature"] == f"sha256={expected}"
    evidence = json.loads(receipt_path.read_text())
    assert evidence["status"] == "materialized"
    assert evidence["poll_count"] == 2
    assert evidence["full_byte_service_account_readback_passed"] is True
    assert evidence["provider_mutation_performed_by_this_tool"] is False
    assert receipt_path.stat().st_mode & 0o777 == 0o440


def test_terminal_blocked_status_is_sealed_without_claiming_readback(
    monkeypatch, tmp_path
) -> None:
    request_path, secret_path, receipt_path, _body, request = _files(tmp_path)
    monkeypatch.setattr(
        reader.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(
            _web_status(request, "blocked", blockers=["construction_failed"])
        ),
    )
    assert reader.main([
        "--request", str(request_path),
        "--secret-file", str(secret_path),
        "--receipt-out", str(receipt_path),
    ]) == 3
    evidence = json.loads(receipt_path.read_text())
    assert evidence["status"] == "blocked"
    assert evidence["blockers"] == ["construction_failed"]
    assert evidence["full_byte_service_account_readback_passed"] is False


def test_identity_mismatch_fails_and_does_not_publish_receipt(
    monkeypatch, tmp_path
) -> None:
    request_path, secret_path, receipt_path, _body, request = _files(tmp_path)
    status = _web_status(request, "materialized")
    status["run_id"] = "different-run"
    monkeypatch.setattr(
        reader.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(status),
    )
    assert reader.main([
        "--request", str(request_path),
        "--secret-file", str(secret_path),
        "--receipt-out", str(receipt_path),
    ]) == 2
    assert not receipt_path.exists()


def test_existing_status_receipt_fails_before_network(monkeypatch, tmp_path) -> None:
    request_path, secret_path, receipt_path, _body, _request = _files(tmp_path)
    receipt_path.write_text("user-owned\n", encoding="utf-8")
    called = False

    def urlopen(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        raise AssertionError("network must not run")

    monkeypatch.setattr(reader.urllib.request, "urlopen", urlopen)
    assert reader.main([
        "--request", str(request_path),
        "--secret-file", str(secret_path),
        "--receipt-out", str(receipt_path),
    ]) == 2
    assert called is False
    assert receipt_path.read_text() == "user-owned\n"
