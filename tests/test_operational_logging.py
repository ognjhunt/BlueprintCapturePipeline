from __future__ import annotations

import io
import json
import logging
import shlex
import sys
from http import HTTPStatus
from pathlib import Path
from types import MethodType, SimpleNamespace
from urllib.error import HTTPError

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline import (
    capture_orchestrator,
    privacy_runner_service,
    run_e2e,
    runpod_provider_adapter,
    video_to_world_runner_service,
)
from blueprint_pipeline.common import PipelineError
from blueprint_pipeline.runtime_service_app import create_runtime_app
from blueprint_pipeline.robot_eval_provider_launcher import run_gpu_provider_launcher
from blueprint_pipeline.runpod_provider_adapter import RUNPOD_API_GATE_ENV, RUNPOD_API_KEY_ENV


def _records(caplog: pytest.LogCaptureFixture, event: str) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if getattr(record, "blueprint_event", None) == event
    ]


def _event_names(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        str(getattr(record, "blueprint_event"))
        for record in caplog.records
        if getattr(record, "blueprint_event", None)
    ]


def _capture_root(tmp_path: Path) -> Path:
    root = tmp_path / "storage" / "bucket" / "scenes" / "site-1" / "captures" / "cap-1"
    root.mkdir(parents=True)
    return root


def test_run_e2e_logs_start_completion_and_preflight_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    capture_root = _capture_root(tmp_path)
    (capture_root / "capture_descriptor.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        run_e2e,
        "build_capture_preflight_report",
        lambda _root: {"status": "ready", "missing_required_inputs": []},
    )
    monkeypatch.setattr(
        run_e2e,
        "run_capture_pipeline",
        lambda **kwargs: {"status": "completed", "lanes": [kwargs["lane"]]},
    )
    monkeypatch.setattr(
        run_e2e,
        "run_agent_review",
        lambda **_kwargs: {
            "artifacts": {"readiness_report": "ready.md"},
            "final_memo_path": "memo.md",
            "final_bundle_path": "bundle.zip",
        },
    )

    with caplog.at_level(logging.INFO, logger="blueprint_pipeline.run_e2e"):
        result = run_e2e.run_end_to_end(
            capture_root=str(capture_root),
            provider="openai",
            pipeline_lane="all",
        )

    assert result["pipeline_status"] == "completed"
    assert _event_names(caplog)[:2] == ["run_e2e.started", "run_e2e.preflight_completed"]
    completed = _records(caplog, "run_e2e.completed")[-1]
    assert completed.capture_root == str(capture_root)
    assert completed.provider == "openai"
    assert completed.pipeline_status == "completed"
    assert completed.pipeline_lanes == ["all"]

    caplog.clear()
    monkeypatch.setattr(
        run_e2e,
        "build_capture_preflight_report",
        lambda _root: {"missing_required_inputs": ["raw/video.mov"]},
    )
    with caplog.at_level(logging.INFO, logger="blueprint_pipeline.run_e2e"):
        with pytest.raises(PipelineError):
            run_e2e.run_end_to_end(capture_root=str(capture_root), provider="openai")

    failure = _records(caplog, "run_e2e.preflight_failed")[-1]
    assert failure.capture_root == str(capture_root)
    assert failure.missing_required_input_count == 1


def test_capture_pipeline_logs_lane_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    descriptor_path = (
        tmp_path
        / "bucket"
        / "scenes"
        / "scene-1"
        / "captures"
        / "capture-1"
        / "capture_descriptor.json"
    )
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        capture_orchestrator,
        "resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )
    monkeypatch.setattr(
        capture_orchestrator,
        "resolve_requested_lanes",
        lambda **_kwargs: ["qualification", "evaluation_prep"],
    )
    monkeypatch.setattr(
        capture_orchestrator,
        "run_qualification_pipeline",
        lambda **_kwargs: {"status": "completed", "lane": "qualification"},
    )
    monkeypatch.setattr(
        capture_orchestrator,
        "run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "eval.json")},
    )

    with caplog.at_level(logging.INFO, logger="blueprint_pipeline.capture_orchestrator"):
        result = capture_orchestrator.run_capture_pipeline(
            descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
            config=capture_orchestrator.PipelineConfig(gcs_root=tmp_path),
        )

    assert result["status"] == "completed"
    assert _event_names(caplog).count("capture_pipeline.lane_started") == 2
    assert _event_names(caplog).count("capture_pipeline.lane_completed") == 2
    completed = _records(caplog, "capture_pipeline.completed")[-1]
    assert completed.lanes == ["qualification", "evaluation_prep"]
    assert completed.result_count == 2


class _RuntimeBackend:
    base_url = "http://runtime.test"
    ws_base_url = "ws://runtime.test"

    def runtime_info(self, *, service_version: str) -> dict[str, object]:
        return {
            "service": "test-runtime",
            "runtime_kind": "native_world_model",
            "production_grade": False,
            "readiness": {"model_ready": True, "checkpoint_ready": True},
            "service_version": service_version,
        }

    def register_site_world_package(self, **_kwargs: object) -> dict[str, object]:
        return {"site_world_id": "siteworld-1", "status": "registered"}

    def load_site_world(self, site_world_id: str) -> dict[str, object]:
        if site_world_id == "missing":
            raise FileNotFoundError(site_world_id)
        return {"site_world_id": site_world_id}

    def load_site_world_health(self, site_world_id: str) -> dict[str, object]:
        if site_world_id == "missing":
            raise FileNotFoundError(site_world_id)
        return {"site_world_id": site_world_id, "status": "healthy"}

    def create_session(self, site_world_id: str, **kwargs: object) -> dict[str, object]:
        return {
            "session_id": str(kwargs.get("session_id") or "session-1"),
            "site_world_id": site_world_id,
        }

    def reset_session(self, session_id: str, **_kwargs: object) -> dict[str, object]:
        return {"session_id": session_id, "status": "reset"}

    def step_session(self, session_id: str, *, action: object) -> dict[str, object]:
        return {"session_id": session_id, "status": "stepped", "action": action}

    def session_state(self, session_id: str) -> dict[str, object]:
        return {"session_id": session_id, "status": "active"}

    def control_session(self, session_id: str, *, control: dict[str, object]) -> dict[str, object]:
        return {"session_id": session_id, "control": control}

    def render_bytes(self, _session_id: str, _camera_id: str) -> bytes:
        return b"\x89PNG\r\n\x1a\n"

    def media_response(
        self,
        _session_id: str,
        *,
        camera_id: str,
        chunk_id: str | None,
    ) -> dict[str, object]:
        return {
            "content": b"media",
            "media_type": "application/octet-stream",
            "headers": {"x-camera-id": camera_id, "x-chunk-id": chunk_id or ""},
        }

    def drain_media_events(self, _session_id: str) -> list[dict[str, object]]:
        return []

    def explorer_render(self, session_id: str, **_kwargs: object) -> dict[str, object]:
        return {"session_id": session_id, "frame_path": "/tmp/frame.png"}

    def explorer_frame_bytes(self, _session_id: str, _camera_id: str) -> bytes:
        return b"\x89PNG\r\n\x1a\n"


def test_runtime_service_logs_request_success_and_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = create_runtime_app(backend=_RuntimeBackend(), title="test-runtime")
    client = TestClient(app)
    payload = {
        "spec": {},
        "registration": {"site_world_id": "siteworld-1"},
        "health": {},
    }

    with caplog.at_level(logging.INFO, logger="blueprint_pipeline.runtime_service_app"):
        assert client.post("/v1/site-worlds", json=payload).status_code == 200
        assert client.get("/v1/site-worlds/missing").status_code == 404
        session = client.post(
            "/v1/site-worlds/siteworld-1/sessions",
            json={
                "robot_profile_id": "robot-1",
                "task_id": "task-1",
                "scenario_id": "scenario-1",
                "start_state_id": "start-1",
            },
        )
        assert session.status_code == 200
        assert client.post("/v1/sessions/session-1/step", json={"action": [1]}).status_code == 200

    assert _records(caplog, "runtime_service.site_world_registered")[-1].site_world_id == (
        "siteworld-1"
    )
    failure = _records(caplog, "runtime_service.request_failed")[-1]
    assert failure.status_code == 404
    assert failure.route == "get_site_world"
    created = _records(caplog, "runtime_service.session_created")[-1]
    assert created.site_world_id == "siteworld-1"
    assert created.session_id == "session-1"
    assert _records(caplog, "runtime_service.session_stepped")[-1].session_id == "session-1"


class _Headers(dict[str, str]):
    def get(self, key: str, default: str | None = None) -> str | None:
        return super().get(key, default)


def _handler(handler_cls, *, path: str, body: object = None, headers: dict[str, str] | None = None):
    handler = object.__new__(handler_cls)
    raw = b"" if body is None else json.dumps(body).encode("utf-8")
    response: dict[str, object] = {"headers": []}
    handler.path = path
    handler.headers = _Headers(headers or {})
    if body is not None and "Content-Length" not in handler.headers:
        handler.headers["Content-Length"] = str(len(raw))
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()

    def send_response(self, status: int) -> None:  # type: ignore[no-untyped-def]
        response["status"] = status

    def send_header(self, key: str, value: str) -> None:  # type: ignore[no-untyped-def]
        response["headers"].append((key, value))

    def end_headers(self) -> None:  # type: ignore[no-untyped-def]
        response["ended"] = True

    handler.send_response = MethodType(send_response, handler)
    handler.send_header = MethodType(send_header, handler)
    handler.end_headers = MethodType(end_headers, handler)
    return handler, response


def test_runner_wrappers_log_rejections_and_completion_without_tokens(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("PRIVACY_RUNNER_KIND", "SAM3")
    monkeypatch.setenv("PRIVACY_RUNNER_TOKEN", "secret-token")
    monkeypatch.setattr(
        privacy_runner_service,
        "execute_privacy_service_request",
        lambda kind, body: {"status": "succeeded", "kind": kind, "body": body},
    )

    with caplog.at_level(logging.INFO):
        handler, response = _handler(privacy_runner_service._Handler, path="/run", body={})
        handler.do_POST()
        assert response["status"] == HTTPStatus.UNAUTHORIZED

        handler, response = _handler(
            privacy_runner_service._Handler,
            path="/run",
            body={"input": "video"},
            headers={"Authorization": "Bearer secret-token"},
        )
        handler.do_POST()
        assert response["status"] == HTTPStatus.OK

    assert _records(caplog, "privacy_runner.request_rejected")[-1].reason == "unauthorized"
    assert _records(caplog, "privacy_runner.request_completed")[-1].runner_kind == "sam3"
    assert "secret-token" not in caplog.text

    caplog.clear()
    monkeypatch.setenv("VIDEO_TO_WORLD_RUNNER_TOKEN", "video-secret")
    monkeypatch.setattr(
        video_to_world_runner_service,
        "execute_video_to_world_request",
        lambda body: {"status": "failed", "body": body},
    )

    with caplog.at_level(logging.INFO):
        handler, response = _handler(
            video_to_world_runner_service._Handler,
            path="/run",
            body=None,
            headers={"Authorization": "Bearer video-secret", "Content-Length": "9"},
        )
        handler.rfile = io.BytesIO(b"{bad-json")
        handler.do_POST()
        assert response["status"] == HTTPStatus.BAD_REQUEST

        handler, response = _handler(
            video_to_world_runner_service._Handler,
            path="/run",
            body={},
            headers={"Authorization": "Bearer video-secret"},
        )
        handler.do_POST()
        assert response["status"] == HTTPStatus.BAD_GATEWAY

    assert _records(caplog, "video_to_world_runner.request_rejected")[-1].reason == (
        "invalid_json"
    )
    completed = _records(caplog, "video_to_world_runner.request_completed")[-1]
    assert completed.runner == "video_to_world"
    assert completed.result_status == "failed"
    assert "video-secret" not in caplog.text


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ready_provider_launch_request(path: Path) -> Path:
    _write_json(
        path,
        {
            "schema_version": "robot_eval_gpu_provider_launch_request.v1",
            "job_id": "launcher-job-1",
            "provider": "runpod",
            "status": "request_manifest_ready",
            "provider_request_shape": {
                "api_payload_is_provider_adapter_template": True,
                "image": {
                    "owner_published_image_ref_required": True,
                    "configured_image_ref": "registry.example/worker:2026-06-12",
                    "configured_image_ref_is_versioned": True,
                },
                "environment": {
                    "secret_env_var_names": ["RUNPOD_API_KEY"],
                    "secret_values_in_artifact": False,
                },
                "inputs": {
                    "manifest_uri_required_for_provider": True,
                    "manifest_uri": "r2://bucket/jobs/launcher-job-1/worker_manifest.json",
                    "manifest_uri_fetchable_by_provider": True,
                    "artifact_output_uri_required": True,
                    "artifact_output_uri": "r2://bucket/jobs/launcher-job-1",
                },
                "limits": {
                    "hard_timeout_seconds": 120,
                    "idle_timeout_seconds": 60,
                    "external_watchdog_ttl_seconds": 180,
                    "max_active_workers": 1,
                },
            },
        },
    )
    return path


def _ready_runpod_request(path: Path) -> Path:
    _write_json(
        path,
        {
            "schema_version": "robot_eval_gpu_provider_launch_request.v1",
            "job_id": "runpod-job-1",
            "provider": "runpod",
            "status": "request_manifest_ready",
            "provider_request_shape": {
                "api_payload_is_provider_adapter_template": True,
                "image": {
                    "configured_image_ref": "registry.example/worker:2026-06-12",
                    "configured_image_ref_is_versioned": True,
                    "configured_image_ref_fetchable_by_provider": True,
                },
                "command": "blueprint-run-robot-eval-worker --manifest ${BLUEPRINT_EVAL_MANIFEST_URI}",
                "environment": {"secret_values_in_artifact": False},
                "inputs": {
                    "manifest_uri": "r2://bucket/jobs/runpod-job-1/worker_manifest.json",
                    "manifest_uri_fetchable_by_provider": True,
                    "artifact_output_uri_required": True,
                    "artifact_output_uri": "r2://bucket/jobs/runpod-job-1",
                },
                "gpu": {"preferred_gpu_class": "NVIDIA RTX A6000"},
                "limits": {
                    "hard_timeout_seconds": 120,
                    "idle_timeout_seconds": 60,
                    "external_watchdog_ttl_seconds": 180,
                    "max_active_workers": 1,
                },
            },
        },
    )
    return path


def _python_command(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def test_provider_adapters_log_blocked_completed_and_redacted_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider_request = _ready_provider_launch_request(tmp_path / "provider_request.json")

    with caplog.at_level(logging.INFO):
        blocked = run_gpu_provider_launcher(
            provider_launch_request_path=provider_request,
            allow_provider_launch=False,
        )

    assert blocked["status"] == "blocked"
    blocked_record = _records(caplog, "robot_eval_provider_launcher.blocked")[-1]
    assert blocked_record.job_id == "launcher-job-1"
    assert blocked_record.provider == "runpod"
    assert blocked_record.blocker_count >= 1

    caplog.clear()
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVIDER_LAUNCH", "true")
    monkeypatch.setenv("RUNPOD_API_KEY", "secret-runpod-key")
    with caplog.at_level(logging.INFO):
        completed = run_gpu_provider_launcher(
            provider_launch_request_path=provider_request,
            allow_provider_launch=True,
            provider_launch_command=_python_command("import os; print(os.environ['RUNPOD_API_KEY'])"),
        )

    assert completed["status"] == "completed"
    assert _records(caplog, "robot_eval_provider_launcher.completed")[-1].exit_code == 0
    assert "secret-runpod-key" not in caplog.text

    caplog.clear()
    runpod_request = _ready_runpod_request(tmp_path / "runpod_request.json")
    with caplog.at_level(logging.INFO):
        dry_run = runpod_provider_adapter.run_runpod_provider_adapter(
            provider_launch_request_path=runpod_request,
            mode="dry-run",
            endpoint_id="endpoint-123",
        )
    assert dry_run["status"] == "dry_run_ready"
    assert _records(caplog, "runpod_provider_adapter.completed")[-1].mode == "dry-run"

    caplog.clear()
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(RUNPOD_API_KEY_ENV, "secret-runpod-key")

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        raise HTTPError(
            request.full_url,
            401,
            "unauthorized secret-runpod-key",
            hdrs=None,
            fp=SimpleNamespace(read=lambda: b"bad secret-runpod-key"),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with caplog.at_level(logging.INFO):
        failed = runpod_provider_adapter.run_runpod_provider_adapter(
            provider_launch_request_path=runpod_request,
            mode="serverless-run",
            allow_runpod_api_call=True,
            endpoint_id="endpoint-123",
        )

    assert failed["status"] == "failed"
    failure = _records(caplog, "runpod_provider_adapter.failed")[-1]
    assert failure.job_id == "runpod-job-1"
    assert failure.http_status_code == 401
    assert "secret-runpod-key" not in caplog.text
