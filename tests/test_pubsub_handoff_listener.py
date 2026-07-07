import json
import types
from pathlib import Path

import pytest

import blueprint_pipeline.pubsub_handoff_listener as listener_module
from blueprint_pipeline.common import PipelineError
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
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def download_to_filename(self, destination: str) -> None:
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
        self.pull_requests: list[dict] = []

    def pull(self, *, request: dict, timeout: int) -> object:
        self.pull_requests.append({"request": request, "timeout": timeout})
        return types.SimpleNamespace(received_messages=self.received_messages)

    def acknowledge(self, *, request: dict) -> None:
        self.acknowledged.extend(request["ack_ids"])


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
            "robot_eval_job_id": "customer job 1",
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
    assert result["status"] == "processed"
    assert staged_request.is_file()
    assert calls == [
        {
            "capture_root": str(capture_root),
            "provider": "openai",
            "run_evaluation_prep": True,
            "resume_completed_stages": True,
            "robot_eval_job_request": str(staged_request),
            "robot_eval_job_id": "customer job 1",
            "robot_eval_provisioner": "runpod",
            "robot_eval_simulator": "mujoco",
            "robot_eval_evaluation_substrate": "wam",
            "robot_eval_budget_usd": 5.0,
            "allow_robot_eval_gpu_provisioning": False,
            "allow_robot_eval_simulator_execution": False,
        }
    ]


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
        }
    ]


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


def test_pull_and_process_acks_successes_while_leaving_poison_unacked(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pubsub_v1 = pytest.importorskip("google.cloud.pubsub_v1")

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

    monkeypatch.setattr(pubsub_v1, "SubscriberClient", lambda: subscriber)
    monkeypatch.setattr(
        listener_module,
        "process_handoff_payload",
        fake_process_handoff_payload,
    )

    processed = pull_and_process(
        subscription="projects/p/subscriptions/s",
        storage_root=tmp_path,
        provider="openai",
        max_messages=3,
    )

    assert processed == 2
    assert processed_payloads == [payload_one, payload_two]
    assert subscriber.acknowledged == ["ack-good-1", "ack-good-2"]


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
