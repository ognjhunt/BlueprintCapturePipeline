"""The capture app's path (swap dispatch, Cloud Run job) was removed: every entry records why and runs nothing."""

from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path

import pytest

from blueprint_pipeline import capture_orchestrator as co

ROOT = Path(__file__).resolve().parents[1]

# extractFrames' message for a website upload, reduced to the fields dispatch reads.
WEBSITE_PAYLOAD = {
    "schema_version": "v1",
    "bucket": "bucket",
    "scene_id": "site-r1",
    "capture_id": "walkthrough-r1",
    "capture_descriptor_uri": "gs://bucket/scenes/site-r1/captures/walkthrough-r1/deliveries/k/capture_descriptor.json",
    "raw_prefix_uri": "gs://bucket/scenes/site-r1/captures/walkthrough-r1/raw",
    "requested_outputs": ["qualification"],
}
DEVICE_PAYLOAD = {
    "bucket": "bucket",
    "scene_id": "scene-1",
    "capture_id": "capture-1",
    "descriptor_gcs_uri": "gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
}


def _storage_trigger():
    spec = importlib.util.spec_from_file_location("storage_trigger_closed_test", ROOT / "functions" / "storage_trigger.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event(payload: dict) -> dict:
    return {"data": base64.b64encode(json.dumps(payload).encode()).decode()}


@pytest.fixture
def trigger(monkeypatch):
    module = _storage_trigger()
    launched: list[dict] = []
    ran: list[dict] = []
    monkeypatch.setenv("PIPELINE_EXECUTION_MODE", "cloud_run_job")
    monkeypatch.setattr(module, "_launch_cloud_run_job", lambda payload: launched.append(dict(payload)) or "cloud_run_job:op")
    monkeypatch.setattr("blueprint_pipeline.capture_orchestrator.run_capture_pipeline",
                        lambda **kwargs: ran.append(kwargs) or {"status": "completed"})
    module.launched, module.ran = launched, ran
    return module


@pytest.mark.parametrize("variant", [
    {},
    {"scene_id": None, "capture_id": None},  # named only by its URIs
    {"scene_id": "site-r1", "capture_id": "supplement-" + "a" * 40,
     "capture_descriptor_uri": None, "raw_prefix_uri": None},
])
def test_swap_dispatch_acknowledges_and_drops_a_website_capture(trigger, monkeypatch, variant) -> None:
    payload = {key: value for key, value in {**WEBSITE_PAYLOAD, **variant}.items() if value is not None}
    trigger.on_swap_dispatch(_event(payload), context=None)  # returns: the delivery is acknowledged
    assert trigger.on_swap_dispatch_http(type("R", (), {"get_json": lambda self, silent=False: payload})()) == ("ok", 200)
    monkeypatch.setenv("PIPELINE_EXECUTION_MODE", "inline")
    assert trigger._execute_pipeline_payload(payload) == "skipped:website_capture"
    assert trigger.launched == [] and trigger.ran == []


def test_dispatch_never_queues_a_website_capture(trigger, monkeypatch) -> None:
    queued: list[dict] = []
    monkeypatch.setattr(trigger, "_dispatch_pubsub", lambda payload: queued.append(payload) or "pubsub:1")
    monkeypatch.setattr(trigger, "_dispatch_cloud_tasks", lambda payload: queued.append(payload) or "tasks:1")
    for mode in ("pubsub", "cloud_tasks", "direct"):
        monkeypatch.setenv("SWAP_TRIGGER_ALLOW_DIRECT", "true")
        assert trigger._dispatch_payload(dict(WEBSITE_PAYLOAD), mode=mode) == "skipped:website_capture"
    assert queued == [] and trigger.ran == []


def test_raw_completion_of_a_website_capture_is_not_materialized_for_the_app_path(trigger, monkeypatch) -> None:
    monkeypatch.delenv("SWAP_TRIGGER_USE_CAPTURE_BRIDGE_HANDOFF", raising=False)
    touched: list[str] = []
    monkeypatch.setattr(trigger, "capture_materialization_readiness", lambda **_: touched.append("readiness"))
    monkeypatch.setattr(trigger, "materialize_capture_bundle", lambda **_: touched.append("materialize"))
    monkeypatch.setattr(trigger, "_dispatch_payload", lambda payload, **_: touched.append("dispatch"))
    trigger.on_storage_finalize(
        {"bucket": "bucket", "name": "scenes/site-r1/captures/walkthrough-r1/raw/capture_upload_complete.json"},
        None)
    assert touched == []


@pytest.mark.parametrize("old_switch", [None, "enabled"])
def test_device_capture_path_runs_nothing_whatever_the_old_switch_says(trigger, monkeypatch, caplog, old_switch) -> None:
    if old_switch:
        monkeypatch.setenv("BLUEPRINT_DEVICE_CAPTURE_PIPELINE", old_switch)
    with caplog.at_level("INFO"):
        trigger.on_swap_dispatch(_event(DEVICE_PAYLOAD), context=None)
    assert trigger.launched == []
    records = [json.loads(r.getMessage()) for r in caplog.records if r.getMessage().startswith("{")]
    assert [(r["schema_version"], r["reason"], r["capture_id"], r["removed_on"]) for r in records] == [
        ("device_capture_pipeline_closed.v1", "device_capture_pipeline_removed", "capture-1", "2026-11-11")]
    monkeypatch.setenv("PIPELINE_EXECUTION_MODE", "inline")
    assert trigger.run_capture_pipeline(descriptor_gcs_uri="gs://b/x")["status"] == "skipped"
    assert trigger._execute_pipeline_payload(dict(DEVICE_PAYLOAD)) == "skipped:device_capture_pipeline_removed"
    monkeypatch.setenv("PIPELINE_EXECUTION_MODE", "cloud_run_job")
    assert trigger._execute_pipeline_payload(dict(DEVICE_PAYLOAD)) == "skipped:device_capture_pipeline_removed"
    assert trigger.launched == [] and trigger.ran == []


@pytest.mark.parametrize("argv", [
    ["--bucket", "bucket", "--scene-id", "scene-1", "--capture-id", "capture-1"],
    ["--descriptor-gcs-uri", "gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json", "--lane", "all"],
    ["--bucket", "bucket", "--scene-id", "scene-1", "--capture-id", "capture-1", "--an-unknown-flag", "x"],
])
def test_cloud_run_job_entry_exits_zero_with_one_record(monkeypatch, capsys, argv) -> None:
    monkeypatch.setenv("BLUEPRINT_DEVICE_CAPTURE_PIPELINE", "enabled")  # no longer read
    assert co.main(argv) == 0
    record = json.loads(capsys.readouterr().out)
    assert (record["schema_version"], record["reason"], record["entry"], record["capture_id"], record["status"]) == (
        "device_capture_pipeline_closed.v1", "device_capture_pipeline_removed", "cloud_run_job", "capture-1", "skipped")


def test_cloud_run_job_entry_with_no_arguments_still_exits_zero(monkeypatch, capsys) -> None:
    for name in ("PIPELINE_DESCRIPTOR_GCS_URI", "PIPELINE_BUCKET", "PIPELINE_SCENE_ID", "PIPELINE_CAPTURE_ID"):
        monkeypatch.delenv(name, raising=False)
    assert co.main([]) == 0
    assert json.loads(capsys.readouterr().out)["capture_id"] is None
