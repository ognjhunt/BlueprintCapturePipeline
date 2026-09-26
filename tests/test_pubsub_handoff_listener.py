import json
import logging
import shutil
import threading
import types
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest
import google.cloud

import blueprint_pipeline.pubsub_handoff_listener as listener_module
import blueprint_pipeline.site_package_orchestrator as orchestrator
from blueprint_pipeline.capture_orchestrator import run_capture_pipeline
from blueprint_pipeline.common import PipelineError, StageError
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.live_pipeline_control_plane import (
    LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION,
)
from blueprint_pipeline.pubsub_handoff_listener import (
    HandoffMessage,
    main,
    parse_handoff_payload,
    pull_and_process,
    process_handoff_payload,
    read_handoff_job_status,
    stage_handoff_capture,
)


# Real iOS raw bundle namelist per CaptureRawContractV3Validator (no pipeline_handoff.json).
_IOS_MANIFEST = {
    "scene_id": "scene-1",
    "capture_id": "capture-1",
    "site_submission_id": "site-submission-scene-1",
    "buyer_request_id": "req-scene-1",
    "capture_job_id": "capture-job-scene-1",
    "requested_outputs": ["robot_eval_dataset", "task_evaluation_run"],
}
_IOS_CONTEXT = {
    "scene_id": "scene-1",
    "capture_id": "capture-1",
    "site_submission_id": "site-submission-scene-1",
    "buyer_request_id": "req-scene-1",
    "capture_job_id": "capture-job-scene-1",
}


def _ios_bundle_blobs(prefix: str) -> "list[FakeBlob]":
    return [
        FakeBlob(f"{prefix}/raw/manifest.json", json.dumps(_IOS_MANIFEST).encode("utf-8")),
        FakeBlob(f"{prefix}/raw/capture_context.json", json.dumps(_IOS_CONTEXT).encode("utf-8")),
        FakeBlob(f"{prefix}/raw/hashes.json", b"{}"),
        FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
        FakeBlob(f"{prefix}/raw/arkit/frames.jsonl", b"{}\n"),
        FakeBlob(f"{prefix}/raw/walkthrough.mov", b"\x00\x00"),
    ]


def _robot_eval_dataset_blobs(prefix: str) -> "list[FakeBlob]":
    return [
        FakeBlob(
            f"{prefix}/pipeline/robot_eval_dataset/task_cards.json",
            json.dumps(
                {
                    "cards": [
                        {
                            "task_id": "scene_anchor_geometry_0",
                            "description": "Walk to the selected scene anchor.",
                        }
                    ]
                }
            ).encode("utf-8"),
        ),
        FakeBlob(
            f"{prefix}/pipeline/robot_eval_dataset/scenario_cards.json",
            json.dumps(
                {
                    "cards": [
                        {
                            "task_id": "scene_anchor_geometry_0",
                            "scenario_id": "scenario_scene_anchor_geometry_0_unitree_g1",
                        }
                    ]
                }
            ).encode("utf-8"),
        ),
        FakeBlob(
            f"{prefix}/pipeline/robot_eval_dataset/robot_eval_dataset_manifest.json",
            b'{"schema_version":"robot_eval_dataset_manifest.v1"}',
        ),
    ]


class FakeBlob:
    def __init__(
        self,
        name: str,
        data: bytes,
        *,
        size: int | None = None,
        generation: int | None = None,
        md5_hash: str | None = None,
        crc32c: str | None = None,
    ) -> None:
        self.name = name
        self._data = data
        self.size = size
        self.generation = generation
        self.md5_hash = md5_hash
        self.crc32c = crc32c
        self.download_count = 0

    def download_to_filename(self, destination: str) -> None:
        self.download_count += 1
        Path(destination).write_bytes(self._data)


class FakeStorageClient:
    def __init__(self, blobs: list[FakeBlob]) -> None:
        self._blobs = blobs

    def bucket(self, _name: str):
        return object()

    def list_blobs(self, _bucket: str, prefix: str):
        return [blob for blob in self._blobs if blob.name.startswith(prefix)]


class FakeSubscriber:
    def __init__(self, received_messages: list[object]) -> None:
        self.received_messages = received_messages
        self.acknowledged: list[str] = []
        self.acknowledge_requests: list[dict] = []
        self.ack_deadline_requests: list[dict] = []
        self.pull_requests: list[dict] = []

    def pull(self, *, request: dict, timeout: int) -> object:
        self.pull_requests.append({"request": request, "timeout": timeout})
        return types.SimpleNamespace(received_messages=self.received_messages)

    def acknowledge(self, *, request: dict) -> None:
        self.acknowledge_requests.append(request)
        self.acknowledged.extend(request["ack_ids"])

    def modify_ack_deadline(self, *, request: dict) -> None:
        self.ack_deadline_requests.append(request)


def test_parse_handoff_payload_requires_identity_consistency() -> None:
    payload = {
        "bucket": "capture-bucket",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        "pipeline_handoff_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/pipeline_handoff.json",
    }

    handoff = parse_handoff_payload(json.dumps(payload).encode("utf-8"))

    assert handoff == HandoffMessage(
        bucket="capture-bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        raw_prefix_uri="gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        pipeline_handoff_uri="gs://capture-bucket/scenes/scene-1/captures/capture-1/pipeline_handoff.json",
    )


def test_parse_handoff_payload_blocks_mismatched_raw_prefix() -> None:
    with pytest.raises(PipelineError, match="raw_prefix_uri does not match"):
        parse_handoff_payload(
            {
                "bucket": "capture-bucket",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "raw_prefix_uri": "gs://capture-bucket/scenes/other/captures/capture-1/raw",
            }
        )


def test_parse_handoff_payload_rejects_invalid_utf8_as_permanent_input_error() -> None:
    with pytest.raises(PipelineError, match="not valid UTF-8"):
        parse_handoff_payload(b"\xff\xfe")


def test_process_handoff_stages_capture_and_runs_e2e(tmp_path: Path) -> None:
    prefix = "scenes/scene-1/captures/capture-1"
    client = FakeStorageClient(
        [
            FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
            FakeBlob(f"{prefix}/raw/manifest.json", b"{}"),
            FakeBlob(f"{prefix}/pipeline_handoff.json", b"{}"),
            FakeBlob(f"{prefix}/capture_descriptor.json", b"{}"),
        ]
    )
    calls = []

    def fake_run_e2e(**kwargs):
        calls.append(kwargs)
        return {"status": "ok"}

    result = process_handoff_payload(
        {
            "bucket": "capture-bucket",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        },
        storage_root=tmp_path,
        provider="openai",
        run_e2e=fake_run_e2e,
        storage_client=client,  # type: ignore[arg-type]
    )

    capture_root = tmp_path / "capture-bucket" / prefix
    assert result["status"] == "processed"
    assert calls == [
        {
            "capture_root": str(capture_root),
            "provider": "openai",
            "run_evaluation_prep": True,
            "resume_completed_stages": True,
        }
    ]
    assert (capture_root / "raw" / "capture_upload_complete.json").is_file()
    assert (capture_root / "pipeline_handoff.json").is_file()


def test_process_handoff_threads_robot_eval_request_without_live_spend(
    tmp_path: Path,
) -> None:
    prefix = "scenes/scene-1/captures/capture-1"
    robot_request_key = f"{prefix}/pipeline/robot_eval_requests/request-1.json"
    client = FakeStorageClient(
        [
            FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
            FakeBlob(f"{prefix}/raw/manifest.json", b"{}"),
            FakeBlob(f"{prefix}/pipeline_handoff.json", b"{}"),
            FakeBlob(f"{prefix}/capture_descriptor.json", b"{}"),
            FakeBlob(robot_request_key, b'{"job_id":"request-1"}'),
        ]
    )
    calls = []

    def fake_run_e2e(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "robot_eval_job": {"status": "blocked"}}

    result = process_handoff_payload(
        {
            "bucket": "capture-bucket",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
            "robot_eval_job_request_uri": f"gs://capture-bucket/{robot_request_key}",
            "robot_eval_job_id": "customer-job-1",
            "robot_eval_provisioner": "runpod",
            "robot_eval_simulator": "mujoco",
            "robot_eval_evaluation_substrate": "wam",
            "robot_eval_budget_usd": 5.0,
        },
        storage_root=tmp_path,
        provider="openai",
        run_e2e=fake_run_e2e,
        storage_client=client,  # type: ignore[arg-type]
    )

    capture_root = tmp_path / "capture-bucket" / prefix
    staged_request = capture_root / "pipeline" / "robot_eval_requests" / "request-1.json"
    assert result["status"] == "retryable_blocked"
    assert result["queue_disposition"] == "retryable"
    assert staged_request.is_file()
    assert calls == [
        {
            "capture_root": str(capture_root),
            "provider": "openai",
            "run_evaluation_prep": True,
            "resume_completed_stages": True,
            "robot_eval_job_request": str(staged_request),
            "robot_eval_job_id": "customer-job-1",
            "robot_eval_provisioner": "runpod",
            "robot_eval_simulator": "mujoco",
            "robot_eval_evaluation_substrate": "wam",
            "robot_eval_budget_usd": 5.0,
            "allow_robot_eval_gpu_provisioning": False,
            "allow_robot_eval_simulator_execution": False,
        }
    ]
    ledger = json.loads(
        (capture_root / "pipeline_job_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["status"] == "retryable_blocked"
    assert ledger["retry_blockers"] == ["blocked"]

    def restored_run_e2e(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "robot_eval_job": {"status": "completed"}}

    retried = process_handoff_payload(
        {
            "bucket": "capture-bucket",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
            "robot_eval_job_request_uri": f"gs://capture-bucket/{robot_request_key}",
            "robot_eval_job_id": "customer-job-1",
        },
        storage_root=tmp_path,
        provider="openai",
        run_e2e=restored_run_e2e,
        storage_client=client,  # type: ignore[arg-type]
    )
    assert retried["status"] == "processed"
    assert len(calls) == 2


def test_process_handoff_stages_control_plane_inbox_without_running_e2e(
    tmp_path: Path,
) -> None:
    prefix = "scenes/scene-1/captures/capture-1"
    client = FakeStorageClient(
        [
            *_ios_bundle_blobs(prefix),
            *_robot_eval_dataset_blobs(prefix),
            FakeBlob(f"{prefix}/capture_descriptor.json", b'{"scene_id":"scene-1"}'),
        ]
    )
    configured_capture_root = tmp_path / "configured-single-capture-root"
    configured_capture_root.mkdir()
    inbox_dir = tmp_path / "control-plane-inbox"
    manifest_path = tmp_path / "control-plane" / "live_pipeline_control_plane_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION,
                "capture_root": str(configured_capture_root),
                "job_request_inbox": str(inbox_dir),
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict] = []

    def fake_run_e2e(**kwargs):
        calls.append(kwargs)
        return {"status": "unexpected"}

    result = process_handoff_payload(
        {
            "bucket": "capture-bucket",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        },
        storage_root=tmp_path,
        provider="openai",
        run_e2e=fake_run_e2e,
        storage_client=client,  # type: ignore[arg-type]
        run_e2e_enabled=False,
        stage_control_plane=True,
        control_plane_manifest_path=manifest_path,
        control_plane_work_dir=tmp_path / "incoming-pubsub-handoffs",
        overwrite_control_plane_input=True,
    )

    capture_root = tmp_path / "capture-bucket" / prefix
    staged_requests = sorted(inbox_dir.glob("*.json"))
    assert result["status"] == "processed"
    assert result["run_e2e"]["status"] == "skipped"
    assert calls == []
    assert result["control_plane_staging"]["status"] == "staged_for_control_plane"
    assert len(staged_requests) == 1
    staged = json.loads(staged_requests[0].read_text(encoding="utf-8"))
    job_request = staged["job_request"]
    assert staged["source_kind"] == "capture_pipeline_handoff"
    assert job_request["site_package"]["capture_root"] == str(capture_root.resolve())
    assert job_request["source"]["selection_state"]["task_id"] == "scene_anchor_geometry_0"
    assert job_request["owner_system"]["site_submission_id"] == "site-submission-scene-1"
    ledger = json.loads(
        (capture_root / "pipeline_job_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["status"] == "completed"
    assert ledger["run_e2e_status"] == "skipped"
    assert ledger["control_plane_staging_status"] == "staged_for_control_plane"
    assert ledger["control_plane_staging_path"] == str(staged_requests[0])


def test_redelivered_completed_handoff_is_idempotent(tmp_path: Path) -> None:
    prefix = "scenes/scene-1/captures/capture-1"
    client = FakeStorageClient(
        [
            FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
            FakeBlob(f"{prefix}/pipeline_handoff.json", b"{}"),
        ]
    )
    payload = {
        "bucket": "capture-bucket",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
    }
    calls: list[dict] = []

    def fake_run_e2e(**kwargs):
        calls.append(kwargs)
        return {"status": "ok"}

    first = process_handoff_payload(
        payload,
        storage_root=tmp_path,
        provider="openai",
        run_e2e=fake_run_e2e,
        storage_client=client,  # type: ignore[arg-type]
    )
    second = process_handoff_payload(
        payload,
        storage_root=tmp_path,
        provider="openai",
        run_e2e=fake_run_e2e,
        storage_client=client,  # type: ignore[arg-type]
    )

    assert first["status"] == "processed"
    assert second["status"] == "skipped_already_processed"
    assert len(calls) == 1
    capture_root = tmp_path / "capture-bucket" / prefix
    ledger = json.loads(
        (capture_root / "pipeline_job_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["status"] == "completed"
    assert ledger["attempt_count"] == 1
    commit = json.loads(
        (capture_root / "pipeline_job_output_commit.json").read_text(encoding="utf-8")
    )
    assert commit["status"] == "committed"
    assert ledger["output_result_sha256"] == commit["result_sha256"]

    status = read_handoff_job_status(
        storage_root=tmp_path,
        bucket="capture-bucket",
        scene_id="scene-1",
        capture_id="capture-1",
    )
    assert status["schema_version"] == "pipeline_job_status.v1"
    assert status["status"] == "completed"
    assert status["attempt_count"] == 1
    assert status["run_e2e_status"] == "ok"
    assert status["completed_redelivery_is_noop"] is True
    assert status["retry_expected_on_redelivery"] is False
    assert status["last_error"] is None
    assert status["attempt_history"] == [
        {
            "attempt_number": 1,
            "completed_at": ledger["completed_at"],
            "run_e2e_status": "ok",
            "stage": "run_e2e",
            "started_at": ledger["last_attempt_started_at"],
            "status": "completed",
            "queue_disposition": "terminal_success",
            "output_commit_status": "committed",
            "output_commit_path": "pipeline_job_output_commit.json",
            "blockers": [],
        }
    ]


def test_expired_lease_recovers_existing_output_commit_without_rerun(
    tmp_path: Path,
) -> None:
    prefix = "scenes/scene-1/captures/capture-1"
    client = FakeStorageClient(
        [
            FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
            FakeBlob(f"{prefix}/pipeline_handoff.json", b"{}"),
        ]
    )
    payload = {
        "bucket": "capture-bucket",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
    }
    capture_root = tmp_path / "capture-bucket" / prefix
    assert process_handoff_payload(
        payload,
        storage_root=tmp_path,
        provider="openai",
        run_e2e=lambda **_kwargs: {"status": "ok"},
        storage_client=client,  # type: ignore[arg-type]
    )["status"] == "processed"
    ledger_path = capture_root / "pipeline_job_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    listener_module.write_json(
        ledger_path,
        {
            **ledger,
            "status": "processing",
            "lease_owner": "crashed-worker",
            "lease_token": "dead-token",
            "lease_expires_at": "2020-01-01T00:00:00Z",
        },
    )
    calls: list[str] = []
    recovered = process_handoff_payload(
        payload,
        storage_root=tmp_path,
        provider="openai",
        run_e2e=lambda **_kwargs: calls.append("rerun") or {"status": "ok"},
        storage_client=client,  # type: ignore[arg-type]
        lease_owner="recovery-worker",
    )
    assert recovered["status"] == "skipped_committed_output_recovered"
    assert recovered["queue_disposition"] == "terminal_success"
    assert calls == []
    final_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert final_ledger["status"] == "completed"
    assert final_ledger["attempt_count"] == 2
    assert final_ledger["attempt_history"][-1]["status"] == (
        "completed_from_output_commit"
    )


def test_read_handoff_job_status_reports_not_staged(tmp_path: Path) -> None:
    status = read_handoff_job_status(
        storage_root=tmp_path,
        bucket="capture-bucket",
        scene_id="scene-1",
        capture_id="capture-1",
    )

    assert status["status"] == "not_staged"
    assert status["staged_capture_present"] is False
    assert status["job_ledger_present"] is False
    assert status["attempt_count"] == 0
    assert status["terminal_code"] is None
    assert status["terminal_receipt_present"] is False
    assert status["ack_receipt"] is None


def test_crashed_processing_run_is_retried_not_skipped(tmp_path: Path) -> None:
    prefix = "scenes/scene-1/captures/capture-1"
    client = FakeStorageClient(
        [
            FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
            FakeBlob(f"{prefix}/pipeline_handoff.json", b"{}"),
        ]
    )
    payload = {
        "bucket": "capture-bucket",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
    }
    boom_calls: list[dict] = []

    def crashing_run_e2e(**kwargs):
        boom_calls.append(kwargs)
        raise RuntimeError("pod died mid-run")

    with pytest.raises(RuntimeError):
        process_handoff_payload(
            payload,
            storage_root=tmp_path,
            provider="openai",
            run_e2e=crashing_run_e2e,
            storage_client=client,  # type: ignore[arg-type]
        )
    capture_root = tmp_path / "capture-bucket" / prefix
    ledger = json.loads(
        (capture_root / "pipeline_job_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["status"] == "failed_retryable"
    assert ledger["attempt_count"] == 1
    assert ledger["last_error_type"] == "RuntimeError"
    assert ledger["last_error"] == "pod died mid-run"
    assert ledger["attempt_history"] == [
        {
            "attempt_number": 1,
            "error": "pod died mid-run",
            "error_type": "RuntimeError",
            "failed_at": ledger["last_failed_at"],
            "stage": "run_e2e",
            "started_at": ledger["last_attempt_started_at"],
            "status": "failed_retryable",
        }
    ]
    status = read_handoff_job_status(
        storage_root=tmp_path,
        bucket="capture-bucket",
        scene_id="scene-1",
        capture_id="capture-1",
    )
    assert status["status"] == "failed_retryable"
    assert status["retry_expected_on_redelivery"] is True
    assert status["last_error_type"] == "RuntimeError"
    assert status["last_error"] == "pod died mid-run"
    assert status["attempt_history"] == ledger["attempt_history"]

    def ok_run_e2e(**kwargs):
        return {"status": "ok"}

    retried = process_handoff_payload(
        payload,
        storage_root=tmp_path,
        provider="openai",
        run_e2e=ok_run_e2e,
        storage_client=client,  # type: ignore[arg-type]
    )
    assert retried["status"] == "processed"
    ledger = json.loads(
        (capture_root / "pipeline_job_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["status"] == "completed"
    assert ledger["attempt_count"] == 2
    assert ledger["last_error"] is None
    assert ledger["last_error_type"] is None
    assert [attempt["status"] for attempt in ledger["attempt_history"]] == [
        "failed_retryable",
        "completed",
    ]
    assert ledger["attempt_history"][0]["error"] == "pod died mid-run"
    assert ledger["attempt_history"][1]["run_e2e_status"] == "ok"


def test_main_status_mode_prints_job_status_without_subscription(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture-bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    capture_root.mkdir(parents=True)
    (capture_root / "pipeline_job_ledger.json").write_text(
        json.dumps(
            {
                "schema_version": "pipeline_job_ledger.v1",
                "status": "completed",
                "attempt_count": 3,
                "run_e2e_status": "ok",
            }
        ),
        encoding="utf-8",
    )
    (capture_root / "pipeline").mkdir(exist_ok=True)
    (capture_root / "pipeline" / "run_e2e_stage_ledger.json").write_text(
        json.dumps(
            {
                "schema_version": "run_e2e_stage_ledger.v1",
                "status": "completed",
                "current_stage": None,
                "failed_stage": None,
                "last_completed_stage": "robot_eval",
                "stages": {
                    "robot_eval": {
                        "name": "robot_eval",
                        "status": "completed",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "--status",
            "--storage-root",
            str(tmp_path),
            "--bucket",
            "capture-bucket",
            "--scene-id",
            "scene-1",
            "--capture-id",
            "capture-1",
        ]
    ) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "completed"
    assert printed["attempt_count"] == 3
    assert printed["run_e2e_status"] == "ok"
    assert printed["run_e2e_stage_ledger_present"] is True
    assert printed["run_e2e_stage_status"] == "completed"
    assert printed["run_e2e_last_completed_stage"] == "robot_eval"
    assert printed["run_e2e_failed_stage"] is None
    assert printed["run_e2e_stage_ledger"]["stages"]["robot_eval"]["status"] == "completed"
    assert printed["provider_runtime_status"] == "not_observed"
    assert printed["continuing_spend_from_this_run"] is False
    assert printed["teardown_attention_required"] is False


def test_main_status_mode_surfaces_provider_spend_and_teardown_attention(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture-bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    provider_dir = capture_root / "pipeline" / "robot_eval_job" / "provider_job"
    provider_dir.mkdir(parents=True)
    (capture_root / "pipeline_job_ledger.json").write_text(
        json.dumps(
            {
                "schema_version": "pipeline_job_ledger.v1",
                "status": "processing",
                "attempt_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (provider_dir / "runpod_wam_async_poll_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "runpod_wam_async_poll_manifest.v1",
                "status": "running",
                "provider_command_status": "running",
                "pod_status": "RUNNING",
                "provider_runtime_output_zip_path": str(
                    provider_dir / "runpod_provider_runtime_output.zip"
                ),
                "output_zip_present": False,
                "runtime_result_status": None,
                "teardown_status": "not_requested",
                "continuing_spend_from_this_run": True,
                "provider_command_blockers": ["runtime_output_not_ready"],
                "raw_secret_values_recorded": False,
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "--status",
            "--storage-root",
            str(tmp_path),
            "--bucket",
            "capture-bucket",
            "--scene-id",
            "scene-1",
            "--capture-id",
            "capture-1",
        ]
    ) == 0

    printed = json.loads(capsys.readouterr().out)
    provider_ops = printed["provider_ops_status"]
    assert printed["provider_runtime_status"] == "running_spend_attention_required"
    assert printed["continuing_spend_from_this_run"] is True
    assert printed["teardown_attention_required"] is True
    assert provider_ops["provider_artifact_count"] == 1
    assert provider_ops["provider_statuses"][0]["artifact_path"] == (
        "pipeline/robot_eval_job/provider_job/runpod_wam_async_poll_manifest.json"
    )
    assert provider_ops["provider_statuses"][0]["provider_phase"] == "RUNNING"
    assert provider_ops["provider_statuses"][0]["teardown_status"] == "not_requested"
    assert provider_ops["provider_statuses"][0]["continuing_spend_from_this_run"] is True
    assert "runtime_output_not_ready" in provider_ops["provider_statuses"][0]["blockers"]


def test_main_status_mode_surfaces_retryable_failure(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture-bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    capture_root.mkdir(parents=True)
    (capture_root / "pipeline_job_ledger.json").write_text(
        json.dumps(
            {
                "schema_version": "pipeline_job_ledger.v1",
                "status": "failed_retryable",
                "attempt_count": 2,
                "last_error_type": "PipelineError",
                "last_error": "missing descriptor",
                "last_failed_at": "2026-07-04T00:00:00+00:00",
                "attempt_history": [
                    {
                        "attempt_number": 2,
                        "status": "failed_retryable",
                        "stage": "run_e2e",
                        "error_type": "PipelineError",
                        "error": "missing descriptor",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "--status",
            "--storage-root",
            str(tmp_path),
            "--bucket",
            "capture-bucket",
            "--scene-id",
            "scene-1",
            "--capture-id",
            "capture-1",
        ]
    ) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "failed_retryable"
    assert printed["retry_expected_on_redelivery"] is True
    assert printed["last_error_type"] == "PipelineError"
    assert printed["last_error"] == "missing descriptor"
    assert printed["attempt_history"][0]["status"] == "failed_retryable"


def test_pull_and_process_acks_successes_and_permanent_invalid_payload(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pubsub_v1 = types.SimpleNamespace()
    monkeypatch.setattr(google.cloud, "pubsub_v1", pubsub_v1, raising=False)

    def received(ack_id: str, message_id: str, data: bytes) -> object:
        return types.SimpleNamespace(
            ack_id=ack_id,
            message=types.SimpleNamespace(
                message_id=message_id,
                data=data,
                attributes={},
            ),
        )

    payload_one = json.dumps(
        {
            "bucket": "capture-bucket",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        }
    ).encode("utf-8")
    payload_two = json.dumps(
        {
            "bucket": "capture-bucket",
            "scene_id": "scene-2",
            "capture_id": "capture-2",
            "raw_prefix_uri": "gs://capture-bucket/scenes/scene-2/captures/capture-2/raw",
        }
    ).encode("utf-8")
    subscriber = FakeSubscriber(
        [
            received("ack-good-1", "msg-good-1", payload_one),
            received("ack-poison", "msg-poison", b"{not-json"),
            received("ack-good-2", "msg-good-2", payload_two),
        ]
    )
    processed_payloads: list[bytes] = []

    def fake_process_handoff_payload(payload: bytes, **_kwargs: object) -> dict:
        if payload == b"{not-json":
            raise PipelineError("bad payload")
        processed_payloads.append(payload)
        return {"status": "processed"}

    monkeypatch.setattr(
        pubsub_v1,
        "SubscriberClient",
        lambda: subscriber,
        raising=False,
    )
    monkeypatch.setattr(
        listener_module,
        "process_handoff_payload",
        fake_process_handoff_payload,
    )

    acknowledged = pull_and_process(
        subscription="projects/p/subscriptions/s",
        storage_root=tmp_path,
        provider="openai",
        max_messages=3,
    )

    assert acknowledged == 3
    assert processed_payloads == [payload_one, payload_two]
    assert subscriber.acknowledged == ["ack-good-1", "ack-poison", "ack-good-2"]
    permanent_invalid = list(
        (tmp_path / ".pubsub_delivery_evidence" / "permanent_invalid").glob("*.json")
    )
    assert len(permanent_invalid) == 1
    invalid_record = json.loads(permanent_invalid[0].read_text(encoding="utf-8"))
    assert invalid_record["raw_payload_stored"] is False
    assert invalid_record["payload_sha256"]
    assert any(
        request["ack_deadline_seconds"] > 0
        for request in subscriber.ack_deadline_requests
    )


def test_pull_and_process_expands_short_subscription_with_adc_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pubsub_v1 = types.SimpleNamespace()
    subscriber = FakeSubscriber([])
    monkeypatch.setattr(google.cloud, "pubsub_v1", pubsub_v1, raising=False)
    monkeypatch.setattr(pubsub_v1, "SubscriberClient", lambda: subscriber, raising=False)
    monkeypatch.setattr(listener_module.google.auth, "default", lambda: (object(), "blueprint-8c1ca"))

    assert pull_and_process(
        subscription="blueprint-pipeline-handoff-listener",
        storage_root=tmp_path,
        provider="openai",
        max_messages=1,
    ) == 0
    assert subscriber.pull_requests[0]["request"]["subscription"] == (
        "projects/blueprint-8c1ca/subscriptions/blueprint-pipeline-handoff-listener"
    )


def test_concurrent_duplicate_delivery_observes_active_lease_and_executes_once(
    tmp_path: Path,
) -> None:
    prefix = "scenes/scene-1/captures/capture-1"
    client = FakeStorageClient(
        [
            FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
            FakeBlob(f"{prefix}/pipeline_handoff.json", b"{}"),
        ]
    )
    payload = {
        "bucket": "capture-bucket",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
    }
    entered = threading.Event()
    release = threading.Event()
    results: list[dict] = []
    calls: list[str] = []

    def slow_run(**_kwargs: object) -> dict:
        calls.append("run")
        entered.set()
        assert release.wait(timeout=5)
        return {"status": "ok"}

    thread = threading.Thread(
        target=lambda: results.append(
            process_handoff_payload(
                payload,
                storage_root=tmp_path,
                provider="openai",
                run_e2e=slow_run,
                storage_client=client,  # type: ignore[arg-type]
                lease_owner="worker-one",
            )
        )
    )
    thread.start()
    assert entered.wait(timeout=5)
    duplicate = process_handoff_payload(
        payload,
        storage_root=tmp_path,
        provider="openai",
        run_e2e=slow_run,
        storage_client=client,  # type: ignore[arg-type]
        lease_owner="worker-two",
    )
    release.set()
    thread.join(timeout=5)

    assert duplicate["status"] == "lease_active_retryable"
    assert duplicate["queue_disposition"] == "retryable"
    assert results[0]["status"] == "processed"
    assert calls == ["run"]


def test_expired_job_lease_is_recovered_with_new_attempt(tmp_path: Path) -> None:
    capture_root = (
        tmp_path
        / "capture-bucket"
        / "scenes"
        / "scene-1"
        / "captures"
        / "capture-1"
    )
    now = datetime(2026, 7, 9, 12, tzinfo=timezone.utc)
    first_status, first = listener_module._claim_job_lease(
        capture_root,
        scene_id="scene-1",
        capture_id="capture-1",
        owner="worker-one",
        lease_seconds=30,
        now=now,
    )
    active_status, _ = listener_module._claim_job_lease(
        capture_root,
        scene_id="scene-1",
        capture_id="capture-1",
        owner="worker-two",
        lease_seconds=30,
        now=now + timedelta(seconds=20),
    )
    recovered_status, recovered = listener_module._claim_job_lease(
        capture_root,
        scene_id="scene-1",
        capture_id="capture-1",
        owner="worker-two",
        lease_seconds=30,
        now=now + timedelta(seconds=31),
    )

    assert first_status == "claimed"
    assert first["attempt_count"] == 1
    assert active_status == "active"
    assert recovered_status == "claimed"
    assert recovered["attempt_count"] == 2
    assert recovered["recovered_expired_lease"] is True
    assert recovered["previous_lease_owner"] == "worker-one"


def test_corrupt_job_ledger_fails_closed_without_execution_or_overwrite(
    tmp_path: Path,
) -> None:
    prefix = "scenes/scene-1/captures/capture-1"
    capture_root = tmp_path / "capture-bucket" / prefix
    capture_root.mkdir(parents=True)
    ledger_path = capture_root / "pipeline_job_ledger.json"
    ledger_path.write_bytes(b'{"status":"processing"')
    calls: list[str] = []
    result = process_handoff_payload(
        {
            "bucket": "capture-bucket",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        },
        storage_root=tmp_path,
        provider="openai",
        run_e2e=lambda **_kwargs: calls.append("run") or {"status": "ok"},
        storage_client=FakeStorageClient([]),  # type: ignore[arg-type]
    )
    assert result["status"] == "job_ledger_corrupt_retryable"
    assert result["queue_disposition"] == "retryable"
    assert calls == []
    assert ledger_path.read_bytes() == b'{"status":"processing"'


def test_pull_and_process_defers_retryable_result_without_ack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pubsub_v1 = types.SimpleNamespace()
    monkeypatch.setattr(google.cloud, "pubsub_v1", pubsub_v1, raising=False)
    payload = json.dumps(
        {
            "bucket": "capture-bucket",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        }
    ).encode()
    received = types.SimpleNamespace(
        ack_id="ack-retry",
        delivery_attempt=5,
        message=types.SimpleNamespace(message_id="msg-retry", data=payload, attributes={}),
    )
    subscriber = FakeSubscriber([received])
    monkeypatch.setattr(
        pubsub_v1,
        "SubscriberClient",
        lambda: subscriber,
        raising=False,
    )
    monkeypatch.setattr(
        listener_module,
        "process_handoff_payload",
        lambda *_args, **_kwargs: {
            "status": "retryable_blocked",
            "queue_disposition": "retryable",
            "blockers": ["dependency_unavailable"],
        },
    )

    assert pull_and_process(
        subscription="projects/p/subscriptions/s",
        storage_root=tmp_path,
        provider="openai",
        max_messages=1,
    ) == 0
    assert subscriber.acknowledged == []
    assert subscriber.ack_deadline_requests[-1]["ack_deadline_seconds"] == 600
    assert all(
        request["ack_deadline_seconds"] != 0
        for request in subscriber.ack_deadline_requests
    )
    exhausted = list(
        (
            tmp_path
            / ".pubsub_delivery_evidence"
            / "retry_exhausted_pending_pubsub_dlq"
        ).glob("*.json")
    )
    assert len(exhausted) == 1
    record = json.loads(exhausted[0].read_text(encoding="utf-8"))
    assert record["delivery_attempt"] == 5
    assert record["blockers"] == ["dependency_unavailable"]


def test_retryable_old_scene_is_deferred_and_new_scene_can_finish_next_pull(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pubsub_v1 = types.SimpleNamespace()
    monkeypatch.setattr(google.cloud, "pubsub_v1", pubsub_v1, raising=False)

    def handoff(scene: str, ack_id: str) -> object:
        payload = json.dumps(
            {
                "bucket": "capture-bucket",
                "scene_id": scene,
                "capture_id": f"capture-{scene}",
                "raw_prefix_uri": (
                    f"gs://capture-bucket/scenes/{scene}/captures/capture-{scene}/raw"
                ),
            }
        ).encode()
        return types.SimpleNamespace(
            ack_id=ack_id,
            delivery_attempt=1,
            message=types.SimpleNamespace(message_id=f"msg-{scene}", data=payload, attributes={}),
        )

    class BatchedSubscriber(FakeSubscriber):
        def __init__(self) -> None:
            super().__init__([])
            self.batches = [[handoff("old", "ack-old")], [handoff("new", "ack-new")]]

        def pull(self, *, request: dict, timeout: int) -> object:
            self.pull_requests.append({"request": request, "timeout": timeout})
            return types.SimpleNamespace(received_messages=self.batches.pop(0))

    subscriber = BatchedSubscriber()
    monkeypatch.setattr(pubsub_v1, "SubscriberClient", lambda: subscriber, raising=False)
    processed: list[str] = []

    def process(payload: bytes, **_kwargs: object) -> dict:
        scene = json.loads(payload)["scene_id"]
        processed.append(scene)
        if scene == "old":
            return {"status": "retryable_blocked", "queue_disposition": "retryable"}
        return {"status": "processed", "queue_disposition": "terminal_success"}

    monkeypatch.setattr(listener_module, "process_handoff_payload", process)

    kwargs = {
        "subscription": "projects/p/subscriptions/s",
        "storage_root": tmp_path,
        "provider": "openai",
        "max_messages": 1,
    }
    assert pull_and_process(**kwargs) == 0
    assert subscriber.acknowledged == []
    assert subscriber.ack_deadline_requests[-1]["ack_ids"] == ["ack-old"]
    assert subscriber.ack_deadline_requests[-1]["ack_deadline_seconds"] == 600
    assert pull_and_process(**kwargs) == 1
    assert processed == ["old", "new"]
    assert subscriber.acknowledged == ["ack-new"]


def test_processing_exception_is_deferred_without_ack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pubsub_v1 = types.SimpleNamespace()
    monkeypatch.setattr(google.cloud, "pubsub_v1", pubsub_v1, raising=False)
    payload = json.dumps(
        {
            "bucket": "capture-bucket",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        }
    ).encode()
    subscriber = FakeSubscriber(
        [
            types.SimpleNamespace(
                ack_id="ack-error",
                delivery_attempt=2,
                message=types.SimpleNamespace(message_id="msg-error", data=payload, attributes={}),
            )
        ]
    )
    monkeypatch.setattr(pubsub_v1, "SubscriberClient", lambda: subscriber, raising=False)

    def fail(*_args: object, **_kwargs: object) -> dict:
        raise RuntimeError("temporary failure")

    monkeypatch.setattr(listener_module, "process_handoff_payload", fail)
    assert pull_and_process(
        subscription="projects/p/subscriptions/s",
        storage_root=tmp_path,
        provider="openai",
        max_messages=1,
    ) == 0
    assert subscriber.acknowledged == []
    assert subscriber.ack_deadline_requests[-1]["ack_deadline_seconds"] == 600


def test_stage_handoff_synthesizes_missing_pipeline_handoff(tmp_path: Path) -> None:
    """XR-03: a real iOS bundle (no hand-authored pipeline_handoff.json) stages without error."""

    prefix = "scenes/scene-1/captures/capture-1"
    handoff = HandoffMessage(
        bucket="capture-bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        raw_prefix_uri="gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        pipeline_handoff_uri=(
            "gs://capture-bucket/scenes/scene-1/captures/capture-1/pipeline_handoff.json"
        ),
    )
    client = FakeStorageClient(_ios_bundle_blobs(prefix))

    capture_root = stage_handoff_capture(handoff, storage_root=tmp_path, storage_client=client)  # type: ignore[arg-type]

    synthesized = capture_root / "pipeline_handoff.json"
    assert synthesized.is_file(), "stage must synthesize pipeline_handoff.json for real iOS bundles"
    payload = json.loads(synthesized.read_text())
    assert payload["scene_id"] == "scene-1"
    assert payload["capture_id"] == "capture-1"
    assert payload["site_submission_id"] == "site-submission-scene-1"
    assert payload["buyer_request_id"] == "req-scene-1"
    assert payload["capture_job_id"] == "capture-job-scene-1"
    assert payload["owner_system"]["request_id"] == "req-scene-1"
    assert payload["synthesized"] is True


def test_stage_handoff_preserves_hand_authored_pipeline_handoff(tmp_path: Path) -> None:
    """A bundle that already carries pipeline_handoff.json is never overwritten by synthesis."""

    prefix = "scenes/scene-1/captures/capture-1"
    handoff = HandoffMessage(
        bucket="capture-bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        raw_prefix_uri="gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        pipeline_handoff_uri=None,
    )
    hand_authored = {"owner_system": {"request_id": "hand-authored"}, "hand_authored": True}
    blobs = _ios_bundle_blobs(prefix) + [
        FakeBlob(f"{prefix}/pipeline_handoff.json", json.dumps(hand_authored).encode("utf-8")),
    ]
    client = FakeStorageClient(blobs)

    capture_root = stage_handoff_capture(handoff, storage_root=tmp_path, storage_client=client)  # type: ignore[arg-type]

    payload = json.loads((capture_root / "pipeline_handoff.json").read_text())
    assert payload == hand_authored


def test_stage_handoff_missing_upload_complete_still_raises(tmp_path: Path) -> None:
    """Synthesis must not paper over a genuinely broken bundle (no capture_upload_complete.json)."""

    prefix = "scenes/scene-1/captures/capture-1"
    handoff = HandoffMessage(
        bucket="capture-bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        raw_prefix_uri="gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        pipeline_handoff_uri=None,
    )
    blobs = [
        FakeBlob(f"{prefix}/raw/manifest.json", json.dumps(_IOS_MANIFEST).encode("utf-8")),
        FakeBlob(f"{prefix}/raw/capture_context.json", json.dumps(_IOS_CONTEXT).encode("utf-8")),
    ]
    client = FakeStorageClient(blobs)

    with pytest.raises(PipelineError, match="capture_upload_complete.json"):
        stage_handoff_capture(handoff, storage_root=tmp_path, storage_client=client)  # type: ignore[arg-type]


def test_website_upload_runs_preparation_before_legacy_robot_job_conversion(tmp_path: Path) -> None:
    prefix = "scenes/scene-1/captures/capture-1"
    manifest = {"scene_id": "scene-1", "capture_id": "capture-1",
                "capture_source": "browser_self_capture", "site_submission_id": "request-1"}
    client = FakeStorageClient([
        FakeBlob(f"{prefix}/raw/manifest.json", json.dumps(manifest).encode()),
        FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
        FakeBlob(f"{prefix}/raw/walkthrough.mov", b"video"),
    ])
    calls = []
    def prepare(**kwargs):
        calls.append(kwargs)
        return {"status": "completed"}
    payload = {"bucket": "capture-bucket", "scene_id": "scene-1", "capture_id": "capture-1",
               "raw_prefix_uri": f"gs://capture-bucket/{prefix}/raw"}
    kwargs = dict(storage_root=tmp_path, provider="openai", run_e2e=prepare,
                  storage_client=client, stage_control_plane=True, run_e2e_enabled=False)
    first = process_handoff_payload(payload, **kwargs)
    second = process_handoff_payload(payload, **kwargs)
    assert first["status"] == "processed"
    assert first["control_plane_staging"] is None
    assert second["status"] == "skipped_already_processed"
    assert len(calls) == 1
    assert calls[0]["resume_completed_stages"] is True
    assert calls[0]["pipeline_lane"] == "qualification"
    assert calls[0]["run_evaluation_prep"] is False
    assert "robot_eval_job_request" not in calls[0]


def test_app_filmed_site_capture_runs_website_preparation_not_device_job_conversion(tmp_path: Path) -> None:
    prefix = "scenes/site-req-1/captures/walkthrough-req-1"
    manifest = {
        "scene_id": "site-req-1", "capture_id": "walkthrough-req-1",
        "capture_source": "iphone", "capture_profile_id": "iphone_arkit_lidar",
        "site_submission_id": "req-1",
        "site_self_capture": {
            "schema_version": "site_self_capture.v1", "authored_by": "blueprint_webapp",
            "site_filmed_itself": True, "capture_job_exists": False,
            "request_id": "req-1", "client": "ios_app_clip",
        },
    }
    client = FakeStorageClient([
        FakeBlob(f"{prefix}/raw/manifest.json", json.dumps(manifest).encode()),
        FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
        FakeBlob(f"{prefix}/raw/walkthrough.mov", b"video"),
    ])
    calls = []

    def prepare(**kwargs):
        calls.append(kwargs)
        return {"status": "completed"}

    payload = {"bucket": "capture-bucket", "scene_id": "site-req-1", "capture_id": "walkthrough-req-1",
               "raw_prefix_uri": f"gs://capture-bucket/{prefix}/raw"}
    result = process_handoff_payload(payload, storage_root=tmp_path, provider="openai", run_e2e=prepare,
                                     storage_client=client, stage_control_plane=True, run_e2e_enabled=False)
    assert result["status"] == "processed"
    assert result["control_plane_staging"] is None
    assert len(calls) == 1
    assert calls[0]["pipeline_lane"] == "qualification"
    assert calls[0]["run_evaluation_prep"] is False


def test_forged_site_marker_does_not_leave_the_device_lane(tmp_path: Path) -> None:
    prefix = "scenes/site-req-1/captures/walkthrough-req-1"
    manifest = {
        "scene_id": "site-req-1", "capture_id": "walkthrough-req-1",
        "capture_source": "iphone", "site_submission_id": "req-1",
        "site_self_capture": {"schema_version": "site_self_capture.v1", "authored_by": "ios_app_clip",
                              "site_filmed_itself": True, "capture_job_exists": False, "request_id": "req-1"},
    }
    client = FakeStorageClient([
        FakeBlob(f"{prefix}/raw/manifest.json", json.dumps(manifest).encode()),
        FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
        FakeBlob(f"{prefix}/raw/walkthrough.mov", b"video"),
    ])
    calls = []
    payload = {"bucket": "capture-bucket", "scene_id": "site-req-1", "capture_id": "walkthrough-req-1",
               "raw_prefix_uri": f"gs://capture-bucket/{prefix}/raw"}
    result = process_handoff_payload(payload, storage_root=tmp_path, provider="openai",
                                     run_e2e=lambda **kwargs: calls.append(kwargs) or {"status": "completed"},
                                     storage_client=client, stage_control_plane=False, run_e2e_enabled=False)
    # Device lane with the e2e run disabled: nothing is prepared as a website capture.
    assert calls == []
    assert result["status"] != "failed"


def test_partial_pipeline_failures_are_not_acknowledged_as_success():
    disposition, blockers = listener_module._handoff_result_disposition({
        "pipeline_status": "completed_with_lane_failures", "final_bundle_path": "exists"})
    assert disposition == "retryable_blocked"
    assert blockers == ["completed_with_lane_failures"]


def test_old_failed_lane_completion_is_reopened_under_existing_lease(tmp_path):
    root = tmp_path / "capture"
    (root / "pipeline").mkdir(parents=True)
    (root / "pipeline_job_ledger.json").write_text(json.dumps({"status": "completed", "attempt_count": 6}))
    (root / "pipeline_job_output_commit.json").write_text(json.dumps({"status": "committed", "result_sha256": "retained"}))
    (root / "pipeline/run_e2e_stage_ledger.json").write_text(json.dumps({"stages": {
        "capture_pipeline": {"result_snapshot": {"status": "completed_with_lane_failures"}}}}))
    status, ledger = listener_module._claim_job_lease(root, scene_id="scene-1", capture_id="cap-1",
                                                     owner="worker", lease_seconds=900)
    assert status == "claimed"
    assert ledger["attempt_count"] == 7
    retained = json.loads((root / "pipeline_job_output_commit.json").read_text())
    assert retained["result_sha256"] == "retained"
    assert retained["status"] == "superseded_failed_lanes"


def test_website_staging_preserves_local_sam_derivatives_outside_raw(tmp_path):
    prefix = "scenes/scene-1/captures/capture-1"
    root = tmp_path / "capture-bucket" / prefix
    stray = root / "raw/object_index_artifacts/sam3_frames/frame_000000.png"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"retained derived frame")
    blobs = [FakeBlob(f"{prefix}/raw/manifest.json", b'{"capture_source":"browser_self_capture"}'),
             FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
             FakeBlob(f"{prefix}/raw/walkthrough.mov", b"original video"),
             FakeBlob(f"{prefix}/pipeline_handoff.json", b"{}")]
    handoff = HandoffMessage(bucket="capture-bucket", scene_id="scene-1", capture_id="capture-1",
        raw_prefix_uri=f"gs://capture-bucket/{prefix}/raw", pipeline_handoff_uri=None)
    for _ in range(2):
        stage_handoff_capture(handoff, storage_root=tmp_path, storage_client=FakeStorageClient(blobs))
    assert (root / "raw/walkthrough.mov").read_bytes() == b"original video"
    assert not (root / "raw/object_index_artifacts").exists()
    archives = list((root / "pipeline/recovered_raw_derivatives").glob("*/receipt.json"))
    assert len(archives) == 1
    assert (archives[0].parent / "object_index_artifacts/sam3_frames/frame_000000.png").read_bytes() == b"retained derived frame"
    assert json.loads(archives[0].read_text())["member_sha256"]["sam3_frames/frame_000000.png"]


def test_recovery_never_reclassifies_uploaded_derivatives_or_device_raw(tmp_path):
    root = tmp_path
    source = root / "raw/object_index_artifacts"
    source.mkdir(parents=True)
    manifest = root / "raw/manifest.json"
    manifest.write_text('{"capture_source":"browser_self_capture"}')
    listener_module._preserve_local_website_derivatives(root, {"prefix/raw/object_index_artifacts/file.png"}, "prefix")
    assert source.exists()
    manifest.write_text('{"capture_source":"ios"}')
    listener_module._preserve_local_website_derivatives(root, set(), "prefix")
    assert source.exists()


def test_recovery_refuses_symlinks_without_moving_anything(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/manifest.json").write_text('{"capture_source":"browser_self_capture"}')
    source = tmp_path / "raw/object_index_artifacts"
    source.symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(PipelineError, match="recovery_unsafe"):
        listener_module._preserve_local_website_derivatives(tmp_path, set(), "prefix")
    assert source.is_symlink()


# ---------------------------------------------------------------------------
# Website authority endings (consent expired, source revoked)
# ---------------------------------------------------------------------------

from tests.test_qualification_coverage_edges import (  # noqa: E402
    _descriptor as _qualification_descriptor,
    _patch_pipeline_side_effects,
    _write_descriptor,
)

PAYLOAD = {
    "bucket": "capture-bucket",
    "scene_id": "scene-1",
    "capture_id": "capture-1",
    "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
}
PAYLOAD_BYTES = json.dumps(PAYLOAD).encode("utf-8")


@pytest.mark.parametrize("code", ["consent_expired", "source_revoked"])
def test_authority_ending_code_walks_the_exception_chain(code):
    try:
        try:
            raise ValueError(f"website_control_scene-sponsorship_http_409:{code}")
        except ValueError as inner:
            raise listener_module.PipelineError("website_task_context failed") from inner
    except listener_module.PipelineError as outer:
        assert listener_module.authority_ending_code(outer) == code


@pytest.mark.parametrize("message", [
    "website_control_scene-sponsorship_http_409:task_brief_missing",
    "website_control_scene-sponsorship_http_503:consent_expired",
    "website_control_scene-sponsorship_http_409:consent_expired_soon",
    "consent_expired",
    "not_website_control_scene-sponsorship_http_409:consent_expired",
    "website_control_scene-sponsorship_http_409:source_revokedX",
    "website_control_scene-sponsorship_http_409:source_revoked-x",
    "website_control_scene-sponsorship_http_409:consent_expiredZ",
])
def test_other_failures_are_not_authority_endings(message):
    assert listener_module.authority_ending_code(ValueError(message)) is None


@pytest.mark.parametrize("message,code", [
    ("website_control_prepared-scene_http_409:source_revoked,task_brief_missing", "source_revoked"),
    ("held (website_control_task-context_http_409:consent_expired)", "consent_expired"),
    ("website_control_task-context_http_409:consent_expired: request refused", "consent_expired"),
])
def test_authority_endings_are_found_between_separators(message, code):
    assert listener_module.authority_ending_code(StageError("website_scene_preparation", message)) == code


def test_authority_ending_code_follows_implicit_context_and_stops_on_cycles():
    try:
        try:
            raise ValueError("website_control_prepared-scene_http_409:source_revoked")
        except ValueError:
            raise RuntimeError("preparation held")  # implicit __context__, no __cause__
    except RuntimeError as outer:
        assert listener_module.authority_ending_code(outer) == "source_revoked"

    first, second = RuntimeError("first"), RuntimeError("second")
    first.__cause__, second.__cause__ = second, first
    assert listener_module.authority_ending_code(first) is None

    deepest = ValueError("website_control_task-context_http_409:consent_expired")
    chain = deepest
    for index in range(20):
        wrapper = PipelineError(f"wrapper {index}")
        wrapper.__cause__ = chain
        chain = wrapper
    assert listener_module.authority_ending_code(chain) is None


def _website_qualification_descriptor(storage_root: Path) -> str:
    return _write_descriptor(storage_root, _qualification_descriptor(
        capture_source="unknown", capture_modality="video_only",
        requested_outputs=["preview_simulation"],
        metadata={"capture_entry_source": "browser_self_capture",
                  "capture_rights": {"derived_scene_generation_allowed": True}}))


def test_a_real_task_context_refusal_is_recognized_as_an_authority_ending(tmp_path, monkeypatch):
    """The WebApp's 409 travels through the real transport and qualification lane."""
    import blueprint_pipeline.website_task_context as website_task_context

    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://tryblueprint.io/api/internal/pipeline/sync")
    monkeypatch.setattr(website_task_context, "load_pipeline_sync_token", lambda: "test-secret")

    def refuse(url, **_kwargs):
        raise HTTPError(url, 409, "request refused", {}, BytesIO(b'{"code":"consent_expired"}'))

    monkeypatch.setattr(website_task_context, "safe_request", refuse)
    storage_root = tmp_path / "gcs"
    descriptor_uri = _website_qualification_descriptor(storage_root)
    with pytest.raises(PipelineError) as failure:
        run_capture_pipeline(
            descriptor_gcs_uri=descriptor_uri, lane="qualification",
            config=types.SimpleNamespace(gcs_root=storage_root, runtime_preflight_enabled=False))
    assert str(failure.value) == "website_control_task-context_http_409:consent_expired"
    assert listener_module.authority_ending_code(failure.value) == "consent_expired"


def test_a_held_preparation_stage_is_recognized_as_an_authority_ending(tmp_path, monkeypatch):
    """A refusal folded into preparation blockers surfaces as a StageError, then a PipelineError."""
    import blueprint_pipeline.website_scene_handoff as website_scene_handoff

    storage_root = tmp_path / "gcs"
    descriptor_uri = _website_qualification_descriptor(storage_root)
    _patch_pipeline_side_effects(monkeypatch)
    monkeypatch.setattr(orchestrator, "load_current_website_task_context", lambda **_: {
        "description": "Pick the box", "confirmed": True,
        "capture_rights": {"derived_scene_generation_allowed": True}})
    monkeypatch.setattr(orchestrator, "load_website_scene_sponsorship", lambda **_: {"sponsor": "blueprint"})
    monkeypatch.setattr(orchestrator, "run_clean_plate_stage", lambda **_: {
        "status": "noop", "privacy_status": "no_people_detected", "privacy_verified": True})
    # prepare_website_scene_handoff catches the transport's ValueError and keeps it as a blocker.
    monkeypatch.setattr(website_scene_handoff, "prepare_website_scene_handoff", lambda **_: {
        "status": "intake_ready",
        "runtime_inputs": {"status": "awaiting_inputs",
                           "blockers": ["website_control_prepared-scene_http_409:source_revoked"]}})
    with pytest.raises(PipelineError) as failure:
        run_capture_pipeline(
            descriptor_gcs_uri=descriptor_uri, lane="qualification",
            config=types.SimpleNamespace(gcs_root=storage_root, runtime_preflight_enabled=False))
    assert isinstance(failure.value.__cause__, StageError)
    assert listener_module.authority_ending_code(failure.value) == "source_revoked"


def test_payload_digest_matches_the_recorded_delivery_evidence(tmp_path):
    message = types.SimpleNamespace(message_id="m1", data=PAYLOAD_BYTES, attributes={})
    evidence = listener_module._write_delivery_evidence(
        storage_root=tmp_path, message=message, received=types.SimpleNamespace(delivery_attempt=1),
        disposition="permanent_invalid", blockers=[])
    recorded = json.loads(evidence.read_text(encoding="utf-8"))["payload_sha256"]
    assert listener_module.payload_sha256(PAYLOAD_BYTES) == recorded
    assert listener_module.payload_sha256(PAYLOAD_BYTES.decode("utf-8")) == recorded
    assert listener_module.payload_sha256({"b": 1, "a": 2}) == sha256(b'{"a":2,"b":1}').hexdigest()


SUBSCRIPTION = "projects/p/subscriptions/s"
_CAPTURE_PREFIX = "scenes/scene-1/captures/capture-1"
_WEBSITE_MANIFEST = {"scene_id": "scene-1", "capture_id": "capture-1",
                     "capture_source": "browser_self_capture", "site_submission_id": "request-1"}
_ORIGINAL_PROCESS_HANDOFF_PAYLOAD = listener_module.process_handoff_payload


def _website_bundle_blobs(prefix: str = _CAPTURE_PREFIX) -> "list[FakeBlob]":
    return [
        FakeBlob(f"{prefix}/raw/manifest.json", json.dumps(_WEBSITE_MANIFEST).encode("utf-8")),
        FakeBlob(f"{prefix}/raw/capture_upload_complete.json", b"{}"),
        FakeBlob(f"{prefix}/raw/walkthrough.mov", b"video"),
    ]


def _received(*, ack_id: str, data: bytes, delivery_attempt: int = 1) -> object:
    return types.SimpleNamespace(
        ack_id=ack_id,
        delivery_attempt=delivery_attempt,
        message=types.SimpleNamespace(message_id=f"msg-{ack_id}", data=data, attributes={}),
    )


def _install_fake_pubsub(monkeypatch, subscriber, *, storage_client=None, run_e2e=None) -> "list[dict]":
    """Point pull_and_process at a fake subscriber and the real handoff processor.

    pull_and_process has no storage or run_e2e seams of its own, so the real
    process_handoff_payload is wrapped with them. Returns every result it produced.
    """

    monkeypatch.setattr(google.cloud, "pubsub_v1",
                        types.SimpleNamespace(SubscriberClient=lambda: subscriber), raising=False)
    results: list[dict] = []

    def process(payload, **kwargs):
        if storage_client is not None:
            kwargs["storage_client"] = storage_client
        if run_e2e is not None:
            kwargs["run_e2e"] = run_e2e
        result = _ORIGINAL_PROCESS_HANDOFF_PAYLOAD(payload, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(listener_module, "process_handoff_payload", process)
    return results


def _pull(storage_root: Path) -> int:
    # The deployed listener stages control-plane input and skips run_e2e for
    # device captures; website captures still run their preparation.
    return pull_and_process(subscription=SUBSCRIPTION, storage_root=storage_root, provider="openai",
                            max_messages=1, stage_control_plane=True, run_e2e_enabled=False)


def _capture_root(storage_root: Path) -> Path:
    return storage_root / "capture-bucket" / _CAPTURE_PREFIX


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _expired(*_args, **_kwargs):
    try:
        raise ValueError("website_control_scene-sponsorship_http_409:consent_expired")
    except ValueError as exc:
        raise listener_module.PipelineError("website scene failed") from exc


class _ListingForbidden:
    def list_blobs(self, *_args, **_kwargs):
        pytest.fail("a capture whose authority ended must not be staged again")


def test_consent_expired_finishes_the_job_as_terminal_and_is_acknowledged(tmp_path, monkeypatch):
    subscriber = FakeSubscriber([_received(ack_id="a1", data=PAYLOAD_BYTES, delivery_attempt=1)])
    results = _install_fake_pubsub(monkeypatch, subscriber,
                                   storage_client=FakeStorageClient(_website_bundle_blobs()), run_e2e=_expired)

    acknowledged = _pull(tmp_path)

    assert acknowledged == 1 and subscriber.acknowledged == ["a1"]
    assert results[0]["status"] == "terminal_authority_ended"
    assert results[0]["queue_disposition"] == "terminal_authority_ended"
    assert results[0]["blockers"] == ["consent_expired"]
    ledger = _read(_capture_root(tmp_path) / "pipeline_job_ledger.json")
    assert ledger["status"] == "terminal_authority_ended"
    assert ledger["terminal_code"] == "consent_expired"
    assert ledger["terminal_payload_sha256"] == sha256(PAYLOAD_BYTES).hexdigest()
    assert ledger["queue_disposition"] == "terminal_authority_ended"
    assert ledger["updated_at"] == ledger["terminal_at"]
    assert ledger["last_error_type"] == "PipelineError"
    assert ledger["last_error"] == "website scene failed"
    assert ledger["lease_owner"] is None and ledger["lease_expires_at"] is None
    assert ledger["terminal_operation"] == "scene-sponsorship"
    assert ledger["attempt_history"] == [{
        "attempt_number": 1, "status": "terminal_authority_ended", "stage": "run_e2e",
        "started_at": ledger["last_attempt_started_at"], "ended_at": ledger["terminal_at"],
        "code": "consent_expired", "operation": "scene-sponsorship",
        "payload_sha256": sha256(PAYLOAD_BYTES).hexdigest(),
    }]
    receipt = _read(_capture_root(tmp_path) / "pipeline_job_terminal_receipt.json")
    assert receipt["status"] == "authority_ended" and receipt["receipt_digest"].startswith("sha256:")
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
    assert {key: value for key, value in receipt.items() if key != "receipt_digest"} == {
        "schema_version": "pipeline_job_terminal_receipt.v1", "status": "authority_ended",
        "code": "consent_expired", "terminal_operation": "scene-sponsorship",
        "bucket": "capture-bucket", "scene_id": "scene-1", "capture_id": "capture-1", "attempt_count": 1,
        "payload_sha256": sha256(PAYLOAD_BYTES).hexdigest(), "ended_at": ledger["terminal_at"],
        "error": "website scene failed",
    }
    assert not (tmp_path / ".pubsub_delivery_evidence").exists()
    ack = _read(_capture_root(tmp_path) / "pipeline_job_ack_receipt.json")
    assert ack["disposition"] == "terminal_authority_ended"
    assert ack["payload_sha256"] == receipt["payload_sha256"]


def test_redelivered_terminal_capture_is_acknowledged_without_staging(tmp_path, monkeypatch):
    first = FakeSubscriber([_received(ack_id="a1", data=PAYLOAD_BYTES)])
    _install_fake_pubsub(monkeypatch, first,
                         storage_client=FakeStorageClient(_website_bundle_blobs()), run_e2e=_expired)
    assert _pull(tmp_path) == 1
    ledger_path = _capture_root(tmp_path) / "pipeline_job_ledger.json"
    terminal_ledger = _read(ledger_path)

    second = FakeSubscriber([_received(ack_id="a2", data=PAYLOAD_BYTES, delivery_attempt=2)])
    results = _install_fake_pubsub(monkeypatch, second, storage_client=_ListingForbidden(),
                                   run_e2e=lambda **_: pytest.fail("a terminal capture must not run again"))

    assert _pull(tmp_path) == 1
    assert second.acknowledged == ["a2"]
    second_result_status = results[0]["status"]
    assert second_result_status == "skipped_terminal_authority_ended"
    assert results[0]["queue_disposition"] == "terminal_authority_ended"
    assert results[0]["blockers"] == ["consent_expired"]
    assert _read(ledger_path) == terminal_ledger
    ack = _read(_capture_root(tmp_path) / "pipeline_job_ack_receipt.json")
    assert ack["disposition"] == "terminal_authority_ended"
    assert (ack["message_id"], ack["delivery_attempt"], ack["acknowledgement_count"]) == ("msg-a2", 2, 2)


def test_a_redelivery_repairs_a_terminal_receipt_lost_to_a_crash(tmp_path, monkeypatch):
    first = FakeSubscriber([_received(ack_id="a1", data=PAYLOAD_BYTES)])
    _install_fake_pubsub(monkeypatch, first,
                         storage_client=FakeStorageClient(_website_bundle_blobs()), run_e2e=_expired)
    assert _pull(tmp_path) == 1
    receipt_path = _capture_root(tmp_path) / "pipeline_job_terminal_receipt.json"
    original = receipt_path.read_bytes()
    receipt_path.unlink()  # the ledger committed; the process died before the receipt

    second = FakeSubscriber([_received(ack_id="a2", data=PAYLOAD_BYTES, delivery_attempt=2)])
    _install_fake_pubsub(monkeypatch, second, storage_client=_ListingForbidden())
    assert _pull(tmp_path) == 1
    assert receipt_path.read_bytes() == original


def test_a_new_payload_reopens_a_terminal_capture(tmp_path, monkeypatch):
    first = FakeSubscriber([_received(ack_id="a1", data=PAYLOAD_BYTES)])
    _install_fake_pubsub(monkeypatch, first,
                         storage_client=FakeStorageClient(_website_bundle_blobs()), run_e2e=_expired)
    assert _pull(tmp_path) == 1
    reopened_payload = json.dumps({
        **PAYLOAD,
        "pipeline_handoff_uri": f"gs://capture-bucket/{_CAPTURE_PREFIX}/pipeline_handoff.json",
    }).encode("utf-8")
    run_e2e_calls: list[dict] = []

    def prepare(**kwargs):
        run_e2e_calls.append(kwargs)
        return {"status": "completed"}

    second = FakeSubscriber([_received(ack_id="a2", data=reopened_payload)])
    results = _install_fake_pubsub(monkeypatch, second,
                                   storage_client=FakeStorageClient(_website_bundle_blobs()), run_e2e=prepare)

    assert _pull(tmp_path) == 1
    assert len(run_e2e_calls) == 1  # the reopened job runs
    assert results[0]["status"] == "processed"
    ledger = _read(_capture_root(tmp_path) / "pipeline_job_ledger.json")
    assert ledger["status"] == "completed"
    assert ledger["attempt_count"] == 2
    assert [row["status"] for row in ledger["attempt_history"]] == [
        "terminal_authority_ended", "reopened_after_terminal_authority", "completed"]
    reopened = ledger["attempt_history"][1]
    assert reopened["attempt_number"] == 2
    assert reopened["terminal_code"] == "consent_expired"
    assert reopened["terminal_payload_sha256"] == sha256(PAYLOAD_BYTES).hexdigest()
    assert reopened["payload_sha256"] == sha256(reopened_payload).hexdigest()
    # The earlier ending's receipt is kept as evidence, but never under the live name.
    capture_root = _capture_root(tmp_path)
    assert not (capture_root / "pipeline_job_terminal_receipt.json").exists()
    superseded = list(capture_root.glob("pipeline_job_terminal_receipt.superseded-*.json"))
    assert len(superseded) == 1
    kept = _read(superseded[0])
    assert kept["payload_sha256"] == sha256(PAYLOAD_BYTES).hexdigest()
    assert kept["receipt_digest"] == canonical_digest(kept, digest_field="receipt_digest")
    status = read_handoff_job_status(storage_root=tmp_path, bucket="capture-bucket",
                                     scene_id="scene-1", capture_id="capture-1")
    assert status["terminal_receipt_present"] is False


def test_non_terminal_409_stays_retryable(tmp_path, monkeypatch):
    def held(**_kwargs):
        try:
            raise ValueError("website_control_task-context_http_409:task_brief_missing")
        except ValueError as exc:
            raise listener_module.PipelineError(str(exc)) from exc

    subscriber = FakeSubscriber([_received(ack_id="a1", data=PAYLOAD_BYTES)])
    _install_fake_pubsub(monkeypatch, subscriber,
                         storage_client=FakeStorageClient(_website_bundle_blobs()), run_e2e=held)

    assert _pull(tmp_path) == 0
    assert subscriber.acknowledged == []
    ledger = _read(_capture_root(tmp_path) / "pipeline_job_ledger.json")
    assert ledger["status"] == "failed_retryable"
    assert "terminal_code" not in ledger
    assert not (_capture_root(tmp_path) / "pipeline_job_terminal_receipt.json").exists()


def test_a_terminal_ledger_without_a_matching_digest_is_not_reopened(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    (root / "pipeline_job_ledger.json").write_text(json.dumps({
        "status": "terminal_authority_ended", "terminal_code": "source_revoked",
        "terminal_payload_sha256": "a" * 64, "attempt_count": 1}), encoding="utf-8")
    claim = listener_module._claim_job_lease
    same, _ = claim(root, scene_id="s", capture_id="c", owner="w", lease_seconds=60, payload_sha256="a" * 64)
    unknown, _ = claim(root, scene_id="s", capture_id="c", owner="w", lease_seconds=60)
    assert (same, unknown) == ("terminal", "terminal")
    assert _read(root / "pipeline_job_ledger.json")["status"] == "terminal_authority_ended"
    reopened, ledger = claim(root, scene_id="s", capture_id="c", owner="w", lease_seconds=60,
                             payload_sha256="b" * 64)
    assert reopened == "claimed"
    assert ledger["attempt_count"] == 2
    assert ledger["attempt_history"][-1]["status"] == "reopened_after_terminal_authority"


def test_ack_receipt_is_written_after_acknowledge_returns(tmp_path, monkeypatch):
    receipt_path = _capture_root(tmp_path) / "pipeline_job_ack_receipt.json"
    subscriber = FakeSubscriber([
        _received(ack_id="a1", data=PAYLOAD_BYTES, delivery_attempt=3),
        _received(ack_id="poison", data=b"{not-json"),
    ])
    present_during_acknowledge: list[tuple[list[str], bool]] = []
    record_acknowledgement = subscriber.acknowledge

    def acknowledge(*, request):
        present_during_acknowledge.append((request["ack_ids"], receipt_path.exists()))
        record_acknowledgement(request=request)

    subscriber.acknowledge = acknowledge
    _install_fake_pubsub(monkeypatch, subscriber, storage_client=FakeStorageClient(_website_bundle_blobs()),
                         run_e2e=lambda **_: {"status": "completed"})

    assert _pull(tmp_path) == 2
    assert subscriber.acknowledged == ["a1", "poison"]
    # a1's receipt appears only after a1's own acknowledgement returned.
    assert present_during_acknowledge == [(["a1"], False), (["poison"], True)]
    ack = _read(receipt_path)
    assert datetime.fromisoformat(ack["acknowledged_at"]).tzinfo is not None
    assert ack == {
        "schema_version": "pubsub_handoff_ack_receipt.v1",
        "subscription": SUBSCRIPTION,
        "message_id": "msg-a1",
        "payload_sha256": sha256(PAYLOAD_BYTES).hexdigest(),
        "delivery_attempt": 3,
        "disposition": "terminal_success",
        "acknowledged_at": ack["acknowledged_at"],
        "acknowledgement_count": 1,
    }
    # The permanently invalid payload has no capture root and keeps only its delivery evidence.
    assert list(tmp_path.rglob("pipeline_job_ack_receipt.json")) == [receipt_path]
    assert len(list((tmp_path / ".pubsub_delivery_evidence" / "permanent_invalid").glob("*.json"))) == 1

    redelivery = FakeSubscriber([_received(ack_id="a2", data=PAYLOAD_BYTES, delivery_attempt=4)])
    _install_fake_pubsub(monkeypatch, redelivery, storage_client=_ListingForbidden())
    assert _pull(tmp_path) == 1
    ack = _read(receipt_path)
    assert (ack["message_id"], ack["delivery_attempt"], ack["disposition"], ack["acknowledgement_count"]) == (
        "msg-a2", 4, "terminal_success", 2)


def test_no_ack_receipt_when_acknowledge_fails(tmp_path, monkeypatch):
    subscriber = FakeSubscriber([_received(ack_id="a1", data=PAYLOAD_BYTES)])
    subscriber.acknowledge = lambda **_k: (_ for _ in ()).throw(RuntimeError("pubsub down"))
    _install_fake_pubsub(monkeypatch, subscriber,
                         storage_client=FakeStorageClient(_website_bundle_blobs()), run_e2e=_expired)
    with pytest.raises(RuntimeError):
        _pull(tmp_path)
    assert not (_capture_root(tmp_path) / "pipeline_job_ack_receipt.json").exists()
    # The job itself still ended; only the acknowledgement is unproven.
    assert _read(_capture_root(tmp_path) / "pipeline_job_ledger.json")["status"] == "terminal_authority_ended"


def test_retryable_results_leave_no_ack_receipt(tmp_path, monkeypatch):
    def held(**_kwargs):
        raise listener_module.PipelineError("website_control_task-context_http_409:task_brief_missing")

    subscriber = FakeSubscriber([_received(ack_id="a1", data=PAYLOAD_BYTES)])
    _install_fake_pubsub(monkeypatch, subscriber,
                         storage_client=FakeStorageClient(_website_bundle_blobs()), run_e2e=held)
    assert _pull(tmp_path) == 0
    assert not (_capture_root(tmp_path) / "pipeline_job_ack_receipt.json").exists()


@pytest.mark.parametrize("damage", ["unwritable", "malformed"])
def test_one_bad_ack_receipt_does_not_cost_the_others(tmp_path, monkeypatch, damage):
    other_prefix = "scenes/scene-2/captures/capture-2"
    other_payload = json.dumps({
        "bucket": "capture-bucket", "scene_id": "scene-2", "capture_id": "capture-2",
        "raw_prefix_uri": f"gs://capture-bucket/{other_prefix}/raw"}).encode("utf-8")
    damaged = _capture_root(tmp_path) / "pipeline_job_ack_receipt.json"
    if damage == "unwritable":
        damaged.mkdir(parents=True)  # a directory where the receipt file belongs: the write fails
    else:
        damaged.parent.mkdir(parents=True)
        damaged.write_bytes(b"\xff\xfe not utf-8")  # reading the previous count raises UnicodeDecodeError
    subscriber = FakeSubscriber([_received(ack_id="a1", data=PAYLOAD_BYTES),
                                 _received(ack_id="a2", data=other_payload)])
    _install_fake_pubsub(
        monkeypatch, subscriber,
        storage_client=FakeStorageClient([*_website_bundle_blobs(), *_website_bundle_blobs(other_prefix)]),
        run_e2e=lambda **_: {"status": "completed"})

    assert _pull(tmp_path) == 2
    assert subscriber.acknowledged == ["a1", "a2"]
    if damage == "unwritable":
        assert damaged.is_dir()
    else:
        # The unreadable receipt is kept under another name, never overwritten.
        kept = list(damaged.parent.glob("pipeline_job_ack_receipt.unreadable-*.json"))
        assert len(kept) == 1 and kept[0].read_bytes() == b"\xff\xfe not utf-8"
        assert _read(damaged)["message_id"] == "msg-a1"
    other = _read(tmp_path / "capture-bucket" / other_prefix / "pipeline_job_ack_receipt.json")
    assert other["message_id"] == "msg-a2" and other["acknowledgement_count"] == 1


def _staging_handoff() -> HandoffMessage:
    return HandoffMessage(bucket="capture-bucket", scene_id="scene-1", capture_id="capture-1",
                          raw_prefix_uri=f"gs://capture-bucket/{_CAPTURE_PREFIX}/raw",
                          pipeline_handoff_uri=None)


def test_staging_manifest_records_cloud_identity(tmp_path):
    video = FakeBlob(f"{_CAPTURE_PREFIX}/raw/walkthrough.mov", b"video", size=5,
                     generation=1790000000000001, md5_hash="bWQ1LWJhc2U2NA==", crc32c="Y3JjMzJj")
    complete = FakeBlob(f"{_CAPTURE_PREFIX}/raw/capture_upload_complete.json", b"{}", size=2, generation=7)
    folder_marker = FakeBlob(f"{_CAPTURE_PREFIX}/raw/", b"")
    client = FakeStorageClient([video, complete, folder_marker])

    capture_root = stage_handoff_capture(_staging_handoff(), storage_root=tmp_path, storage_client=client)

    manifest = _read(capture_root / "pipeline_staging_manifest.json")
    assert manifest["schema_version"] == "pipeline_handoff_staging_manifest.v1"
    assert manifest["bucket"] == "capture-bucket"
    assert manifest["prefix"] == f"{_CAPTURE_PREFIX}/"
    assert datetime.fromisoformat(manifest["staged_at"]).tzinfo is not None
    assert manifest["objects"] == [
        {"name": f"{_CAPTURE_PREFIX}/raw/walkthrough.mov", "relative_path": "raw/walkthrough.mov",
         "size": 5, "generation": "1790000000000001", "md5_hash": "bWQ1LWJhc2U2NA==", "crc32c": "Y3JjMzJj"},
        {"name": f"{_CAPTURE_PREFIX}/raw/capture_upload_complete.json",
         "relative_path": "raw/capture_upload_complete.json",
         "size": 2, "generation": "7", "md5_hash": None, "crc32c": None},
    ]
    # Local derivations are not cloud objects and are not in the manifest.
    assert (capture_root / "pipeline_handoff.json").is_file()


def test_unchanged_objects_are_not_downloaded_again(tmp_path):
    blob = FakeBlob(f"{_CAPTURE_PREFIX}/raw/capture_upload_complete.json", b"{}", size=2, generation=7)
    unversioned = FakeBlob(f"{_CAPTURE_PREFIX}/raw/walkthrough.mov", b"video")  # generation/size unknown
    client = FakeStorageClient([blob, unversioned])
    handoff = _staging_handoff()

    capture_root = stage_handoff_capture(handoff, storage_root=tmp_path, storage_client=client)
    first = _read(capture_root / "pipeline_staging_manifest.json")
    stage_handoff_capture(handoff, storage_root=tmp_path, storage_client=client)

    assert blob.download_count == 1
    assert unversioned.download_count == 2  # a blob whose generation or size is unknown always downloads
    second = _read(capture_root / "pipeline_staging_manifest.json")
    assert second["objects"] == first["objects"]  # the skipped blob keeps its row


def test_changed_generation_downloads_again(tmp_path):
    name = f"{_CAPTURE_PREFIX}/raw/capture_upload_complete.json"
    handoff = _staging_handoff()
    capture_root = stage_handoff_capture(
        handoff, storage_root=tmp_path,
        storage_client=FakeStorageClient([FakeBlob(name, b"{}", size=2, generation=7)]))

    replaced = FakeBlob(name, b"[]", size=2, generation=8)
    stage_handoff_capture(handoff, storage_root=tmp_path, storage_client=FakeStorageClient([replaced]))
    assert replaced.download_count == 1
    assert (capture_root / "raw/capture_upload_complete.json").read_bytes() == b"[]"
    assert _read(capture_root / "pipeline_staging_manifest.json")["objects"][0]["generation"] == "8"

    # Same generation, but the local copy no longer has the staged size: download it again.
    (capture_root / "raw/capture_upload_complete.json").write_bytes(b"[ ]")
    stage_handoff_capture(handoff, storage_root=tmp_path, storage_client=FakeStorageClient([replaced]))
    assert replaced.download_count == 2
    assert (capture_root / "raw/capture_upload_complete.json").read_bytes() == b"[]"


def test_status_reports_an_authority_ending_and_its_acknowledgement(tmp_path, monkeypatch):
    subscriber = FakeSubscriber([_received(ack_id="a1", data=PAYLOAD_BYTES)])
    _install_fake_pubsub(monkeypatch, subscriber,
                         storage_client=FakeStorageClient(_website_bundle_blobs()), run_e2e=_expired)
    assert _pull(tmp_path) == 1

    status = read_handoff_job_status(storage_root=tmp_path, bucket="capture-bucket",
                                     scene_id="scene-1", capture_id="capture-1")

    assert status["status"] == "terminal_authority_ended"
    assert status["terminal_code"] == "consent_expired"
    assert status["terminal_operation"] == "scene-sponsorship"
    assert status["terminal_receipt_present"] is True
    assert status["ack_receipt"]["disposition"] == "terminal_authority_ended"
    assert status["ack_receipt"]["acknowledgement_count"] == 1
    assert status["retry_expected_on_redelivery"] is False
    assert status["provider_ops_status"]["provider_artifact_count"] == 0


def test_an_ack_receipt_never_recreates_a_workspace_retired_after_the_ack(tmp_path, monkeypatch):
    scene_dir = tmp_path / "capture-bucket" / "scenes" / "scene-1"
    subscriber = FakeSubscriber([_received(ack_id="a1", data=PAYLOAD_BYTES)])
    record_acknowledgement = subscriber.acknowledge

    def acknowledge_then_retire(*, request):
        record_acknowledgement(request=request)
        shutil.rmtree(scene_dir)  # retirement wins the race to the capture

    subscriber.acknowledge = acknowledge_then_retire
    _install_fake_pubsub(monkeypatch, subscriber, storage_client=FakeStorageClient(_website_bundle_blobs()),
                         run_e2e=lambda **_: {"status": "completed"})

    assert _pull(tmp_path) == 1
    assert subscriber.acknowledged == ["a1"]
    assert not scene_dir.exists()


@pytest.mark.parametrize("workspace", ["absent", "no_ledger"])
def test_an_ack_for_a_capture_without_a_ledger_writes_nothing(tmp_path, monkeypatch, caplog, workspace):
    capture_root = _capture_root(tmp_path)
    if workspace == "no_ledger":
        capture_root.mkdir(parents=True)
    subscriber = FakeSubscriber([_received(ack_id="a1", data=PAYLOAD_BYTES)])
    monkeypatch.setattr(google.cloud, "pubsub_v1",
                        types.SimpleNamespace(SubscriberClient=lambda: subscriber), raising=False)
    # A redelivery answered for a retired scene still names its (gone) capture root.
    monkeypatch.setattr(listener_module, "process_handoff_payload", lambda *_args, **_kwargs: {
        "status": "skipped_retired_terminal", "queue_disposition": "terminal_success",
        "capture_root": str(capture_root)})
    caplog.set_level(logging.WARNING, logger=listener_module.logger.name)

    assert _pull(tmp_path) == 1
    assert subscriber.acknowledged == ["a1"]
    if workspace == "absent":
        assert not (tmp_path / "capture-bucket").exists()
    else:
        assert list(capture_root.iterdir()) == []
    expected = {"absent": "pubsub_handoff.ack_receipt_skipped_capture_absent",
                "no_ledger": "pubsub_handoff.ack_receipt_skipped_ledger_absent"}[workspace]
    assert [record.getMessage() for record in caplog.records
            if record.getMessage().startswith("pubsub_handoff.ack_receipt_skipped")] == [expected]


def test_an_undecodable_staging_manifest_skips_nothing_and_is_replaced(tmp_path):
    blob = FakeBlob(f"{_CAPTURE_PREFIX}/raw/capture_upload_complete.json", b"{}", size=2, generation=7)
    manifest = _capture_root(tmp_path) / "pipeline_staging_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"\xff\xfe not utf-8")

    stage_handoff_capture(_staging_handoff(), storage_root=tmp_path, storage_client=FakeStorageClient([blob]))

    assert blob.download_count == 1
    assert _read(manifest)["objects"][0]["generation"] == "7"


@pytest.mark.parametrize("damaged", ["pipeline_job_ack_receipt.json", "pipeline_job_ledger.json"])
def test_status_survives_an_undecodable_record(tmp_path, capsys, damaged):
    capture_root = _capture_root(tmp_path)
    capture_root.mkdir(parents=True)
    (capture_root / "pipeline_job_ledger.json").write_text(json.dumps({
        "schema_version": "pipeline_job_ledger.v1", "status": "completed", "attempt_count": 1}), encoding="utf-8")
    (capture_root / damaged).write_bytes(b"\xff\xfe not utf-8")

    assert main(["--status", "--storage-root", str(tmp_path), "--bucket", "capture-bucket",
                 "--scene-id", "scene-1", "--capture-id", "capture-1"]) == 0

    printed = json.loads(capsys.readouterr().out)
    if damaged == "pipeline_job_ack_receipt.json":
        assert printed["status"] == "completed"
        assert printed["ack_receipt"] is None
    else:
        assert printed["status"] == "corrupt"  # the existing fail-closed ledger state


def test_each_message_is_acknowledged_as_soon_as_it_finishes(tmp_path, monkeypatch):
    other_prefix = "scenes/scene-2/captures/capture-2"
    other_payload = json.dumps({
        "bucket": "capture-bucket", "scene_id": "scene-2", "capture_id": "capture-2",
        "raw_prefix_uri": f"gs://capture-bucket/{other_prefix}/raw"}).encode("utf-8")
    subscriber = FakeSubscriber([_received(ack_id="a1", data=PAYLOAD_BYTES),
                                 _received(ack_id="poison", data=b"{not-json"),
                                 _received(ack_id="a2", data=other_payload)])
    seen_when_processing: list[tuple[list[str], bool]] = []

    def prepare(**_kwargs):
        first_receipt = _capture_root(tmp_path) / "pipeline_job_ack_receipt.json"
        seen_when_processing.append((list(subscriber.acknowledged), first_receipt.exists()))
        return {"status": "completed"}

    _install_fake_pubsub(
        monkeypatch, subscriber,
        storage_client=FakeStorageClient([*_website_bundle_blobs(), *_website_bundle_blobs(other_prefix)]),
        run_e2e=prepare)

    assert _pull(tmp_path) == 3
    # An ack ID is spent the moment its message finishes, so a later long-running
    # message cannot outlive the earlier message's ack deadline.
    assert seen_when_processing == [([], False), (["a1", "poison"], True)]
    assert [request["ack_ids"] for request in subscriber.acknowledge_requests] == [["a1"], ["poison"], ["a2"]]


def _acknowledged_capture_with_ledger(tmp_path: Path) -> Path:
    capture_root = _capture_root(tmp_path)
    capture_root.mkdir(parents=True)
    (capture_root / "pipeline_job_ledger.json").write_text('{"status": "completed"}', encoding="utf-8")
    (capture_root / ".pipeline_job_ledger.json.lock").touch()
    return capture_root


@pytest.mark.parametrize("disposition,recorded", [
    ("terminal_success", True),
    ("terminal_authority_ended", True),
    ("terminal_mystery", False),
    (None, False),
])
def test_ack_receipts_record_only_known_dispositions(tmp_path, monkeypatch, caplog, disposition, recorded):
    capture_root = _acknowledged_capture_with_ledger(tmp_path)
    result = {"status": "processed", "capture_root": str(capture_root)}
    if disposition is not None:
        result["queue_disposition"] = disposition
    subscriber = FakeSubscriber([_received(ack_id="a1", data=PAYLOAD_BYTES)])
    monkeypatch.setattr(google.cloud, "pubsub_v1",
                        types.SimpleNamespace(SubscriberClient=lambda: subscriber), raising=False)
    monkeypatch.setattr(listener_module, "process_handoff_payload", lambda *_args, **_kwargs: result)
    caplog.set_level(logging.WARNING, logger=listener_module.logger.name)

    assert _pull(tmp_path) == 1

    receipt = capture_root / "pipeline_job_ack_receipt.json"
    if recorded:
        assert _read(receipt)["disposition"] == disposition
    else:
        assert not receipt.exists()
        assert any(record.getMessage() == "pubsub_handoff.ack_receipt_skipped_disposition_unrecognized"
                   for record in caplog.records)


def test_authority_ending_keeps_the_refusing_operation():
    held = StageError("website_scene_preparation", "website_control_prepared-scene_http_409:source_revoked")
    assert listener_module.authority_ending(held) == ("prepared-scene", "source_revoked")
    assert listener_module.authority_ending(ValueError("website_control_task-context_http_409:task_brief_missing")) is None


def _second_payload() -> bytes:
    return json.dumps({
        **PAYLOAD,
        "pipeline_handoff_uri": f"gs://capture-bucket/{_CAPTURE_PREFIX}/pipeline_handoff.json",
    }).encode("utf-8")


def test_a_payload_that_already_ended_stays_ended_after_a_later_ending(tmp_path, monkeypatch):
    for ack_id, payload in (("p1", PAYLOAD_BYTES), ("p2", _second_payload())):
        subscriber = FakeSubscriber([_received(ack_id=ack_id, data=payload)])
        _install_fake_pubsub(monkeypatch, subscriber,
                             storage_client=FakeStorageClient(_website_bundle_blobs()), run_e2e=_expired)
        assert _pull(tmp_path) == 1
    ledger_path = _capture_root(tmp_path) / "pipeline_job_ledger.json"
    ended_twice = _read(ledger_path)
    assert ended_twice["terminal_payload_sha256"] == sha256(_second_payload()).hexdigest()

    redelivered = FakeSubscriber([_received(ack_id="p1-again", data=PAYLOAD_BYTES, delivery_attempt=2)])
    results = _install_fake_pubsub(monkeypatch, redelivered, storage_client=_ListingForbidden(),
                                   run_e2e=lambda **_: pytest.fail("an ended payload must not run again"))

    assert _pull(tmp_path) == 1
    assert results[0]["status"] == "skipped_terminal_authority_ended"
    assert _read(ledger_path) == ended_twice


def _live_terminal_receipt(storage_root: Path) -> Path:
    return _capture_root(storage_root) / "pipeline_job_terminal_receipt.json"


def _status(storage_root: Path) -> dict:
    return read_handoff_job_status(storage_root=storage_root, bucket="capture-bucket",
                                   scene_id="scene-1", capture_id="capture-1")


def test_a_reopened_ending_that_crashed_before_its_receipt_is_repaired(tmp_path, monkeypatch):
    first = FakeSubscriber([_received(ack_id="p1", data=PAYLOAD_BYTES)])
    _install_fake_pubsub(monkeypatch, first,
                         storage_client=FakeStorageClient(_website_bundle_blobs()), run_e2e=_expired)
    assert _pull(tmp_path) == 1

    # P2 reopens the capture and ends too, but the process dies before its receipt.
    write_receipt = listener_module._write_terminal_receipt

    def killed(*_args, **_kwargs):
        raise RuntimeError("process killed before the terminal receipt")

    monkeypatch.setattr(listener_module, "_write_terminal_receipt", killed)
    second = FakeSubscriber([_received(ack_id="p2", data=_second_payload())])
    _install_fake_pubsub(monkeypatch, second,
                         storage_client=FakeStorageClient(_website_bundle_blobs()), run_e2e=_expired)
    assert _pull(tmp_path) == 0
    ledger = _read(_capture_root(tmp_path) / "pipeline_job_ledger.json")
    assert ledger["terminal_payload_sha256"] == sha256(_second_payload()).hexdigest()
    assert _status(tmp_path)["terminal_receipt_present"] is False

    monkeypatch.setattr(listener_module, "_write_terminal_receipt", write_receipt)
    redelivered = FakeSubscriber([_received(ack_id="p2-again", data=_second_payload(), delivery_attempt=2)])
    _install_fake_pubsub(monkeypatch, redelivered, storage_client=_ListingForbidden())
    assert _pull(tmp_path) == 1

    receipt = _read(_live_terminal_receipt(tmp_path))
    assert receipt["payload_sha256"] == sha256(_second_payload()).hexdigest()
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
    assert _status(tmp_path)["terminal_receipt_present"] is True


def test_a_stale_receipt_under_the_live_name_is_set_aside_and_rewritten(tmp_path, monkeypatch):
    for ack_id, payload in (("p1", PAYLOAD_BYTES), ("p2", _second_payload())):
        subscriber = FakeSubscriber([_received(ack_id=ack_id, data=payload)])
        _install_fake_pubsub(monkeypatch, subscriber,
                             storage_client=FakeStorageClient(_website_bundle_blobs()), run_e2e=_expired)
        assert _pull(tmp_path) == 1
    capture_root = _capture_root(tmp_path)
    (p1_receipt,) = capture_root.glob("pipeline_job_terminal_receipt.superseded-*.json")
    stale = p1_receipt.read_bytes()
    _live_terminal_receipt(tmp_path).write_bytes(stale)  # P1's receipt back under the live name
    assert _status(tmp_path)["terminal_receipt_present"] is False

    redelivered = FakeSubscriber([_received(ack_id="p2-again", data=_second_payload(), delivery_attempt=2)])
    _install_fake_pubsub(monkeypatch, redelivered, storage_client=_ListingForbidden())
    assert _pull(tmp_path) == 1

    assert _read(_live_terminal_receipt(tmp_path))["payload_sha256"] == sha256(_second_payload()).hexdigest()
    kept = sorted(capture_root.glob("pipeline_job_terminal_receipt.superseded-*.json"))
    assert [path.read_bytes() for path in kept].count(stale) == 2  # both copies of P1's receipt are kept
    assert _status(tmp_path)["terminal_receipt_present"] is True


def test_a_terminal_ending_whose_lease_was_lost_is_not_acknowledged(tmp_path, monkeypatch):
    ledger_path = _capture_root(tmp_path) / "pipeline_job_ledger.json"

    def lease_lost_then_expired(**_kwargs):
        # Another worker recovered this job's lease while this one was still running.
        listener_module.write_json(ledger_path, {**_read(ledger_path), "lease_owner": "other-worker",
                                                 "lease_token": "other-token"})
        _expired()

    subscriber = FakeSubscriber([_received(ack_id="a1", data=PAYLOAD_BYTES)])
    _install_fake_pubsub(monkeypatch, subscriber, storage_client=FakeStorageClient(_website_bundle_blobs()),
                         run_e2e=lease_lost_then_expired)

    assert _pull(tmp_path) == 0
    assert subscriber.acknowledged == []
    ledger = _read(ledger_path)
    assert (ledger["status"], ledger["lease_owner"]) == ("processing", "other-worker")
    assert "terminal_code" not in ledger
    assert not _live_terminal_receipt(tmp_path).exists()
    assert not (_capture_root(tmp_path) / "pipeline_job_ack_receipt.json").exists()
