import importlib.util
from pathlib import Path

import pytest


def _load_storage_trigger_module():
    module_path = Path(__file__).resolve().parents[1] / "functions" / "storage_trigger.py"
    spec = importlib.util.spec_from_file_location("storage_trigger_test_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_on_storage_finalize_publishes_handoff_for_bridge_primary_raw_completion(monkeypatch) -> None:
    """XR-02: bridge-primary raw completion must emit exactly one canonical handoff (not a black hole)."""

    storage_trigger = _load_storage_trigger_module()
    published: list[dict[str, object]] = []

    monkeypatch.setenv("SWAP_TRIGGER_USE_CAPTURE_BRIDGE_HANDOFF", "true")
    monkeypatch.setenv("SWAP_TRIGGER_DISPATCH_MODE", "pubsub")
    # Descriptor-topic dispatch must NOT be used for the bridge handoff.
    monkeypatch.setattr(
        storage_trigger,
        "_dispatch_payload",
        lambda payload: (_ for _ in ()).throw(AssertionError("descriptor dispatch must not run for bridge handoff")),
    )
    monkeypatch.setattr(
        storage_trigger,
        "_publish_handoff_message",
        lambda payload: published.append(dict(payload)) or "pubsub:handoff-1",
    )

    event = {
        "bucket": "bucket",
        "name": "scenes/scene-1/captures/capture-1/raw/capture_upload_complete.json",
    }

    storage_trigger.on_storage_finalize(event, context=None)

    assert len(published) == 1
    payload = published[0]
    assert payload["bucket"] == "bucket"
    assert payload["scene_id"] == "scene-1"
    assert payload["capture_id"] == "capture-1"
    assert payload["raw_prefix_uri"] == "gs://bucket/scenes/scene-1/captures/capture-1/raw"
    assert payload["pipeline_handoff_uri"] == (
        "gs://bucket/scenes/scene-1/captures/capture-1/pipeline_handoff.json"
    )


def test_raw_completion_handoff_retains_original_finalize_version(monkeypatch) -> None:
    from types import SimpleNamespace
    from blueprint_pipeline.pubsub_handoff_listener import parse_handoff_payload

    storage_trigger = _load_storage_trigger_module()
    published = []
    monkeypatch.setenv("SWAP_TRIGGER_USE_CAPTURE_BRIDGE_HANDOFF", "true")
    monkeypatch.setenv("SWAP_TRIGGER_DISPATCH_MODE", "pubsub")
    monkeypatch.setattr(storage_trigger, "_publish_handoff_message",
                        lambda payload: published.append(payload) or "pubsub:handoff-2")
    event = {"bucket": "bucket", "name": "scenes/scene-1/captures/capture-1/raw/capture_upload_complete.json",
             "generation": "17000000000000000001"}
    context = SimpleNamespace(event_id="finalize-123", event_type="google.storage.object.finalize")

    storage_trigger.on_storage_finalize(event, context)

    assert len(published) == 1
    assert published[0]["source_finalize"] == {
        "bucket": event["bucket"], "object_name": event["name"],
        "generation": event["generation"], "event_id": context.event_id,
        "event_source": context.event_type,
    }
    assert parse_handoff_payload(published[0]).source_finalize == published[0]["source_finalize"]


def test_raw_completion_handoff_refuses_unusable_event_generation_before_publish(monkeypatch) -> None:
    from types import SimpleNamespace

    storage_trigger = _load_storage_trigger_module()
    published = []
    monkeypatch.setenv("SWAP_TRIGGER_USE_CAPTURE_BRIDGE_HANDOFF", "true")
    monkeypatch.setenv("SWAP_TRIGGER_DISPATCH_MODE", "pubsub")
    monkeypatch.setattr(storage_trigger, "_publish_handoff_message", lambda payload: published.append(payload))
    event = {"bucket": "bucket", "name": "scenes/scene-1/captures/capture-1/raw/capture_upload_complete.json",
             "generation": "01"}
    context = SimpleNamespace(event_id="finalize-123", event_type="google.storage.object.finalize")
    with pytest.raises(ValueError, match="capture_finalize_source_invalid"):
        storage_trigger.on_storage_finalize(event, context)
    assert not published


def test_build_handoff_payload_matches_listener_contract() -> None:
    """XR-04 contract: the handoff publisher emits exactly what parse_handoff_payload accepts."""

    from blueprint_pipeline.pubsub_handoff_listener import HandoffMessage, parse_handoff_payload

    storage_trigger = _load_storage_trigger_module()
    payload = storage_trigger._build_handoff_payload(
        bucket="capture-bucket",
        scene_id="scene-1",
        capture_id="capture-1",
    )

    handoff = parse_handoff_payload(payload)
    assert handoff == HandoffMessage(
        bucket="capture-bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        raw_prefix_uri="gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        pipeline_handoff_uri="gs://capture-bucket/scenes/scene-1/captures/capture-1/pipeline_handoff.json",
    )


def test_on_storage_finalize_ignores_bridge_primary_descriptor(monkeypatch) -> None:
    storage_trigger = _load_storage_trigger_module()
    dispatched: list[dict[str, object]] = []
    published: list[dict[str, object]] = []

    monkeypatch.setenv("SWAP_TRIGGER_USE_CAPTURE_BRIDGE_HANDOFF", "true")
    monkeypatch.setattr(storage_trigger, "_dispatch_payload", lambda payload: dispatched.append(dict(payload)) or "ok")
    monkeypatch.setattr(
        storage_trigger,
        "_publish_handoff_message",
        lambda payload: published.append(dict(payload)) or "ok",
    )

    event = {
        "bucket": "bucket",
        "name": "scenes/scene-1/captures/capture-1/capture_descriptor.json",
    }

    storage_trigger.on_storage_finalize(event, context=None)

    # iOS never uploads capture_descriptor.json; when the bridge is primary the raw-completion
    # event is the ingest path, so a descriptor finalize stays a no-op (no descriptor dispatch,
    # no duplicate handoff).
    assert dispatched == []
    assert published == []
