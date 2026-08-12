import json
import hmac
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import live_pipeline_intake_service as intake_service
from blueprint_pipeline import task_evaluation_terminal_resource_release as release_module
from blueprint_pipeline.common import write_json
from blueprint_pipeline.live_pipeline_intake_service import create_app
from blueprint_pipeline.task_evaluation_launch_dispatcher import canonical_digest
from blueprint_pipeline.task_evaluation_terminal_resource_release import (
    dispatch_terminal_resource_release,
    process_terminal_resource_release_queue,
    stage_terminal_resource_release_request,
)


def _digest(char: str) -> str:
    return f"sha256:{char * 64}"


def _request() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "task_evaluation_terminal_resource_release_request.v1",
        "release_id": "launch-001-vast-47508030",
        "launch_id": "launch-001",
        "run_id": "run-001",
        "request_digest": _digest("a"),
        "control_plane_terminal_blocker": {
            "schema_version": "task_evaluation_launch_control_plane_blocker.v1",
            "status": "blocked",
            "code": "control_plane_terminal_receipt_missing_after_spend_authority_expiry",
            "launch_id": "launch-001",
            "run_id": "run-001",
            "request_digest": _digest("a"),
            "spend_authority_expires_at": "2026-08-10T12:30:00.000Z",
            "observed_at_iso": "2026-08-10T12:31:00.000Z",
            "pipeline_terminal_receipt_observed": False,
            "provider_mutation_performed_by_webapp": False,
            "paid_execution_retry_performed": False,
            "execution_result": "not_observed",
            "scripted_positive_controls_result": "not_observed",
            "learned_policy_result": "not_observed",
        },
        "provider": "vast",
        "instance_id": "47508030",
        "expected_label": "blueprint-adp009d-1786496624",
        "authorization": {
            "actor": {"id": "founder-001", "role": "admin"},
            "authorized_at": "2026-08-11T12:00:00.000Z",
            "action": "terminal_provider_record_release",
            "approved": True,
            "max_additional_spend_usd": 0,
            "retry_cap": 0,
        },
        "provider_mutation_performed_inside_web_request": False,
        "automatic_retry_performed": False,
        "claim_ceiling": "operational_resource_release_only",
    }
    value["terminal_resource_release_digest"] = canonical_digest(
        value, digest_field="terminal_resource_release_digest"
    )
    return value


def _zero(path: Path) -> dict[str, object]:
    report = {
        "schema_version": "gpu_spend_guard.v1",
        "provider_zero_verified": True,
        "provider_zero": {"status": "verified", "blockers": []},
    }
    write_json(path, report)
    return {"exit_code": 0, "report": report, "raw_process_output_recorded": False}


def test_exact_terminal_release_requires_label_then_proves_exact_absence_and_global_zero(
    tmp_path: Path,
) -> None:
    request_path = tmp_path / "request.json"
    write_json(request_path, _request())

    class Provider:
        name = "vast"

        def __init__(self) -> None:
            self.inspect_calls = 0
            self.terminated: list[str] = []

        def inspect(self, instance_id: str) -> dict[str, object]:
            assert instance_id == "47508030"
            self.inspect_calls += 1
            if self.inspect_calls == 1:
                return {
                    "status": "observed", "provider": "vast", "api_confirmed": True,
                    "instance_id": instance_id, "name": "blueprint-adp009d-1786496624",
                    "desiredStatus": "exited", "provider_absence_confirmed": False,
                }
            return {
                "status": "absent", "provider": "vast", "api_confirmed": True,
                "instance_id": instance_id, "provider_absence_confirmed": True,
            }

        def terminate(self, instance_id: str) -> dict[str, object]:
            self.terminated.append(instance_id)
            return {"status": "terminated", "http": 204}

    provider = Provider()
    receipt = dispatch_terminal_resource_release(
        request_path=request_path,
        state_root=tmp_path / "state",
        provider_factory=lambda name: provider,
        provider_zero_guard=_zero,
    )

    assert receipt["status"] == "completed"
    assert provider.terminated == ["47508030"]
    assert receipt["exact_provider_absence_confirmed"] is True
    assert receipt["provider_zero_verified"] is True
    assert receipt["execution_result"] == "not_observed"
    assert receipt["automatic_retry_performed"] is False


def test_label_or_live_state_mismatch_never_deletes_the_provider_record(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    write_json(request_path, _request())

    class Provider:
        def inspect(self, _instance_id: str) -> dict[str, object]:
            return {
                "status": "observed", "provider": "vast", "api_confirmed": True,
                "instance_id": "47508030", "name": "different-label", "desiredStatus": "exited",
            }

        def terminate(self, _instance_id: str) -> dict[str, object]:
            raise AssertionError("label mismatch must not delete")

    receipt = dispatch_terminal_resource_release(
        request_path=request_path,
        state_root=tmp_path / "state",
        provider_factory=lambda name: Provider(),
        provider_zero_guard=_zero,
    )

    assert receipt["status"] == "blocked"
    assert receipt["provider_mutations_performed"] == 0
    assert "terminal_resource_release_expected_label_mismatch" in receipt["blockers"]


def test_queue_is_immutable_and_worker_calls_only_the_canonical_allocator_shape(tmp_path: Path) -> None:
    queue = tmp_path / "queue"
    request = _request()
    staged = stage_terminal_resource_release_request(value=request, queue_root=queue)
    assert staged["status"] == "queued"
    assert stage_terminal_resource_release_request(value=request, queue_root=queue)["already_exists"] is True

    invocations: list[dict[str, object]] = []

    def fake_allocator(*, request_path: Path, state_root: Path) -> dict[str, object]:
        invocations.append({"request_path": request_path, "state_root": state_root})
        return {"schema_version": "task_evaluation_terminal_resource_release_receipt.v1", "status": "completed"}

    result = process_terminal_resource_release_queue(
        queue_root=queue,
        state_root=tmp_path / "state",
        dispatcher=fake_allocator,
    )
    assert result["status"] == "completed"
    assert len(invocations) == 1
    assert list((queue / "completed").glob("*.json"))


def test_default_queue_worker_never_calls_a_provider_and_uses_the_canonical_allocator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = tmp_path / "queue"
    stage_terminal_resource_release_request(value=_request(), queue_root=queue)
    invocations: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs) -> SimpleNamespace:
        invocations.append(argv)
        output = Path(argv[argv.index("--terminal-resource-release-output") + 1])
        write_json(output, {
            "schema_version": "task_evaluation_terminal_resource_release_receipt.v1",
            "status": "completed",
        })
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(release_module.subprocess, "run", fake_run)
    result = process_terminal_resource_release_queue(
        queue_root=queue, state_root=tmp_path / "state",
    )

    assert result["status"] == "completed"
    assert invocations[0][1:4] == [
        "-m", "blueprint_pipeline.paid_resource_allocator", "gpu-canary",
    ]
    assert "--terminal-resource-release" in invocations[0]
    assert "--execute" in invocations[0]


def test_canonical_allocator_refuses_to_release_without_execute_and_uses_release_only_path(
    tmp_path: Path, monkeypatch
) -> None:
    request_path = tmp_path / "request.json"
    output = tmp_path / "allocator-receipt.json"
    write_json(request_path, _request())
    calls: list[dict[str, object]] = []

    def fake_dispatch(*, request_path: str, state_root: Path) -> dict[str, object]:
        calls.append({"request_path": request_path, "state_root": state_root})
        return {"status": "completed", "provider_mutations_performed": 1}

    monkeypatch.setattr(allocator, "dispatch_terminal_resource_release", fake_dispatch)
    with pytest.raises(SystemExit):
        allocator.main([
            "gpu-canary", "--terminal-resource-release", str(request_path),
            "--terminal-resource-release-output", str(output),
        ])
    assert calls == []
    with pytest.raises(SystemExit):
        allocator.main([
            "gpu-canary", "--terminal-resource-release", str(request_path),
            "--terminal-resource-release-output", str(output), "--execute",
        ])
    assert calls == []
    monkeypatch.setenv(allocator.TERMINAL_RESOURCE_RELEASE_WORKER_ENV, "true")
    assert allocator.main([
        "gpu-canary", "--terminal-resource-release", str(request_path),
        "--terminal-resource-release-output", str(output), "--execute",
    ]) == 0
    assert calls == [{"request_path": str(request_path), "state_root": output.parent.resolve()}]
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "completed"


def test_signed_pipeline_intake_only_stages_the_release_request_before_worker_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = tmp_path / "queue"
    token = "test-release-intake-token"
    monkeypatch.setenv(
        intake_service.INTAKE_CLIENT_SECRETS_ENV,
        json.dumps({"blueprint-webapp": token}),
    )
    monkeypatch.setenv(
        intake_service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"),
    )
    monkeypatch.setenv(intake_service.INTAKE_WORK_DIR_ENV, str(tmp_path / "intake-work"))
    monkeypatch.setenv(
        intake_service.TASK_EVALUATION_TERMINAL_RESOURCE_RELEASE_QUEUE_ROOT_ENV, str(queue),
    )
    monkeypatch.setenv(
        intake_service.TASK_EVALUATION_TERMINAL_RESOURCE_RELEASE_TRIGGER_SYSTEMD_UNIT_ENV,
        "blueprint-task-evaluation-terminal-resource-release.service",
    )
    monkeypatch.setenv(
        intake_service.TASK_EVALUATION_TERMINAL_RESOURCE_RELEASE_ALLOW_TRIGGER_ENV, "true",
    )
    intake_service._INTAKE_NONCE_CACHE.clear()
    calls: list[list[str]] = []
    monkeypatch.setattr(
        intake_service.subprocess,
        "run",
        lambda argv, **_kwargs: calls.append(list(argv)) or SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )
    request = _request()
    body = json.dumps(request, separators=(",", ":"))
    timestamp = datetime.now(timezone.utc).isoformat()
    nonce = "terminal-release-nonce-1"
    signature = hmac.new(
        token.encode("utf-8"),
        f"{timestamp}.blueprint-webapp.{nonce}.{body}".encode("utf-8"),
        "sha256",
    ).hexdigest()
    response = TestClient(create_app()).post(
        "/api/live-pipeline/task-evaluation-terminal-resource-releases",
        data=body,
        headers={
            "content-type": "application/json",
            "x-blueprint-pipeline-timestamp": timestamp,
            "x-blueprint-pipeline-client-id": "blueprint-webapp",
            "x-blueprint-pipeline-nonce": nonce,
            "x-blueprint-pipeline-signature": f"sha256={signature}",
        },
    )

    assert response.status_code == 202
    assert response.json()["schema_version"] == "task_evaluation_terminal_resource_release_intake_receipt.v1"
    assert response.json()["provider_mutation_performed_inside_http_request"] is False
    assert list((queue / "pending").glob("*.json"))
    assert calls == [[
        "systemctl", "start", "--no-block", "blueprint-task-evaluation-terminal-resource-release.service",
    ]]
