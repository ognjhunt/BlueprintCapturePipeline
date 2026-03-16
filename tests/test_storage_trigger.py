import base64
import importlib.util
import json
from pathlib import Path


def _load_storage_trigger_module():
    module_path = Path(__file__).resolve().parents[1] / "functions" / "storage_trigger.py"
    spec = importlib.util.spec_from_file_location("storage_trigger_test_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_on_swap_dispatch_launches_cloud_run_job(monkeypatch) -> None:
    storage_trigger = _load_storage_trigger_module()
    captured: dict[str, object] = {}

    monkeypatch.setenv("PIPELINE_EXECUTION_MODE", "cloud_run_job")
    monkeypatch.setenv("PIPELINE_RUN_JOB_NAME", "blueprint-pipeline")
    monkeypatch.setenv("PIPELINE_RUN_JOB_REGION", "us-central1")
    monkeypatch.setenv("PIPELINE_PROJECT_ID", "blueprint-8c1ca")

    def _launch(payload):
        captured["payload"] = dict(payload)
        return "cloud_run_job:op-123"

    monkeypatch.setattr(storage_trigger, "_launch_cloud_run_job", _launch)

    payload = {
        "descriptor_gcs_uri": "gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        "bucket": "bucket",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
    }
    event = {
        "data": base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8"),
    }

    storage_trigger.on_swap_dispatch(event, context=None)

    assert captured["payload"] == payload


def test_on_storage_finalize_retries_when_capture_bundle_not_ready(monkeypatch) -> None:
    storage_trigger = _load_storage_trigger_module()

    monkeypatch.setattr(
        storage_trigger,
        "capture_materialization_readiness",
        lambda **_kwargs: {"ready": False, "issues": ["missing_manifest"]},
    )

    event = {
        "bucket": "bucket",
        "name": "scenes/scene-1/captures/capture-1/raw/capture_upload_complete.json",
    }

    try:
        storage_trigger.on_storage_finalize(event, context=None)
    except RuntimeError as exc:
        assert "missing_manifest" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected readiness failure to raise")


def test_on_storage_finalize_ignores_bridge_primary_raw_completion(monkeypatch) -> None:
    storage_trigger = _load_storage_trigger_module()
    dispatched: list[dict[str, object]] = []

    monkeypatch.setenv("SWAP_TRIGGER_USE_CAPTURE_BRIDGE_HANDOFF", "true")
    monkeypatch.setenv("SWAP_TRIGGER_DISPATCH_MODE", "pubsub")
    monkeypatch.setattr(storage_trigger, "_dispatch_payload", lambda payload: dispatched.append(dict(payload)) or "ok")

    event = {
        "bucket": "bucket",
        "name": "scenes/scene-1/captures/capture-1/raw/capture_upload_complete.json",
    }

    storage_trigger.on_storage_finalize(event, context=None)

    assert dispatched == []


def test_on_storage_finalize_ignores_bridge_primary_descriptor(monkeypatch) -> None:
    storage_trigger = _load_storage_trigger_module()
    dispatched: list[dict[str, object]] = []

    monkeypatch.setenv("SWAP_TRIGGER_USE_CAPTURE_BRIDGE_HANDOFF", "true")
    monkeypatch.setattr(storage_trigger, "_dispatch_payload", lambda payload: dispatched.append(dict(payload)) or "ok")

    event = {
        "bucket": "bucket",
        "name": "scenes/scene-1/captures/capture-1/capture_descriptor.json",
    }

    storage_trigger.on_storage_finalize(event, context=None)

    assert dispatched == []
