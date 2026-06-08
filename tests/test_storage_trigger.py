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


def test_on_swap_dispatch_inline_accepts_capture_descriptor_uri_alias(monkeypatch) -> None:
    storage_trigger = _load_storage_trigger_module()
    captured: dict[str, object] = {}

    monkeypatch.setenv("PIPELINE_EXECUTION_MODE", "inline")
    monkeypatch.setattr(
        storage_trigger,
        "run_capture_pipeline",
        lambda **kwargs: captured.update(kwargs) or {"status": "completed"},
    )

    payload = {
        "capture_descriptor_uri": "gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
    }
    event = {
        "data": base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8"),
    }

    storage_trigger.on_swap_dispatch(event, context=None)

    assert captured == {
        "descriptor_gcs_uri": "gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
    }


def test_on_swap_dispatch_http_accepts_capture_descriptor_uri_alias(monkeypatch) -> None:
    storage_trigger = _load_storage_trigger_module()
    captured: dict[str, object] = {}

    class _Request:
        def get_json(self, *, silent: bool = False):  # noqa: ARG002
            return {
                "capture_descriptor_uri": (
                    "gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json"
                ),
                "scene_id": "scene-1",
                "capture_id": "capture-1",
            }

    def _execute(payload):
        captured["payload"] = dict(payload)
        return "inline:completed"

    monkeypatch.setattr(storage_trigger, "_execute_pipeline_payload", _execute)

    response = storage_trigger.on_swap_dispatch_http(_Request())

    assert response == ("ok", 200)
    assert captured["payload"]["capture_descriptor_uri"] == (
        "gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json"
    )


def test_on_swap_dispatch_inline_runs_robot_eval_outputs_from_bridge_handoff(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GCS_ROOT", str(tmp_path))
    monkeypatch.setenv("PIPELINE_EXECUTION_MODE", "inline")
    storage_trigger = _load_storage_trigger_module()
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_outputs": ["qualification", "robot_eval_dataset", "task_evaluation_run"],
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, object]] = []

    def _qualification(**kwargs):
        calls.append(("qualification", kwargs["requested_lanes"]))
        return {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        }

    def _evaluation_prep(**kwargs):
        calls.append(("evaluation_prep", kwargs["capture_root"]))
        return {"manifest_path": str(tmp_path / "evaluation_prep_manifest.json")}

    def _simulation_automation(**kwargs):
        calls.append(("simulation_automation", kwargs["capture_root"]))
        return {
            "manifest_path": str(tmp_path / "simulation_automation_run_manifest.json"),
            "plan_path": str(tmp_path / "simulation_automation_plan.json"),
            "status": "blocked",
        }

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        _qualification,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        _evaluation_prep,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.build_simulation_automation",
        _simulation_automation,
    )

    payload = {
        "capture_descriptor_uri": "gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        "requested_lanes": ["evaluation_prep", "robot_eval_dataset", "task_evaluation_run"],
        "scene_id": "scene-1",
        "capture_id": "capture-1",
    }
    event = {
        "data": base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8"),
    }

    storage_trigger.on_swap_dispatch(event, context=None)

    assert calls == [
        ("qualification", ["qualification", "evaluation_prep", "simulation_automation"]),
        ("evaluation_prep", descriptor_path.parent),
        ("simulation_automation", descriptor_path.parent),
    ]


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
