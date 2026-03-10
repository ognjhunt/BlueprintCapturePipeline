"""Tests for async storage trigger dispatch behavior."""

from __future__ import annotations

import base64
import json

import pytest

from functions import storage_trigger


def test_parse_descriptor_path_valid() -> None:
    parsed = storage_trigger.parse_descriptor_path(
        "scenes/scene_a/captures/cap_b/capture_descriptor.json"
    )
    assert parsed == {
        "scene_id": "scene_a",
        "capture_id": "cap_b",
        "object_name": "scenes/scene_a/captures/cap_b/capture_descriptor.json",
    }


def test_parse_descriptor_path_invalid() -> None:
    assert storage_trigger.parse_descriptor_path("scenes/scene_a/raw/manifest.json") is None


def test_parse_raw_upload_complete_path_valid() -> None:
    parsed = storage_trigger.parse_raw_upload_complete_path(
        "scenes/scene_a/captures/cap_b/raw/capture_upload_complete.json"
    )
    assert parsed == {
        "scene_id": "scene_a",
        "capture_id": "cap_b",
        "object_name": "scenes/scene_a/captures/cap_b/raw/capture_upload_complete.json",
    }


def test_on_storage_finalize_ignores_non_descriptor(monkeypatch) -> None:
    called = {"count": 0}

    def _fake_dispatch(payload):  # noqa: ANN001
        called["count"] += 1
        return "ok"

    monkeypatch.setattr(storage_trigger, "_dispatch_payload", _fake_dispatch)

    storage_trigger.on_storage_finalize(
        {"bucket": "bucket", "name": "scenes/a/raw/manifest.json"},
        None,
    )

    assert called["count"] == 0


def test_on_storage_finalize_materializes_from_raw_completion(monkeypatch) -> None:
    captured = {}

    def _fake_dispatch(payload):  # noqa: ANN001
        captured.update(payload)
        return "ok"

    def _fake_materialize(**kwargs):  # noqa: ANN001
        return {
            "descriptor_uri": "gs://bucket/scenes/scene_1/captures/cap_1/capture_descriptor.json"
        }

    monkeypatch.setattr(storage_trigger, "_dispatch_payload", _fake_dispatch)
    monkeypatch.setattr(storage_trigger, "materialize_capture_bundle", _fake_materialize)

    storage_trigger.on_storage_finalize(
        {
            "bucket": "bucket",
            "name": "scenes/scene_1/captures/cap_1/raw/capture_upload_complete.json",
        },
        None,
    )

    assert captured["scene_id"] == "scene_1"
    assert captured["capture_id"] == "cap_1"
    assert captured["descriptor_gcs_uri"].endswith("/capture_descriptor.json")


def test_on_storage_finalize_dispatches_payload(monkeypatch) -> None:
    captured = {}

    def _fake_dispatch(payload):  # noqa: ANN001
        captured.update(payload)
        return "pubsub:123"

    monkeypatch.setattr(storage_trigger, "_dispatch_payload", _fake_dispatch)

    storage_trigger.on_storage_finalize(
        {
            "bucket": "bucket",
            "name": "scenes/scene_1/captures/cap_1/capture_descriptor.json",
        },
        None,
    )

    assert captured["scene_id"] == "scene_1"
    assert captured["capture_id"] == "cap_1"
    assert captured["descriptor_gcs_uri"] == (
        "gs://bucket/scenes/scene_1/captures/cap_1/capture_descriptor.json"
    )


def test_dispatch_payload_direct_mode(monkeypatch) -> None:
    captured = {}

    def _fake_run(*, descriptor_gcs_uri: str):
        captured["descriptor"] = descriptor_gcs_uri

    monkeypatch.setattr(storage_trigger, "run_capture_pipeline", _fake_run)
    monkeypatch.setenv("SWAP_TRIGGER_ALLOW_DIRECT", "true")

    result = storage_trigger._dispatch_payload(
        {"descriptor_gcs_uri": "gs://bucket/scenes/s/captures/c/capture_descriptor.json"},
        mode="direct",
    )

    assert result == "direct:completed"
    assert captured["descriptor"].startswith("gs://bucket/")


def test_dispatch_payload_direct_mode_blocked_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SWAP_TRIGGER_ALLOW_DIRECT", raising=False)
    with pytest.raises(RuntimeError):
        storage_trigger._dispatch_payload(
            {"descriptor_gcs_uri": "gs://bucket/scenes/s/captures/c/capture_descriptor.json"},
            mode="direct",
        )


def test_dispatch_payload_default_pubsub(monkeypatch) -> None:
    called = {"count": 0}

    def _fake_pubsub(payload):  # noqa: ANN001
        called["count"] += 1
        return "pubsub:message"

    monkeypatch.setattr(storage_trigger, "_dispatch_pubsub", _fake_pubsub)
    monkeypatch.delenv("SWAP_TRIGGER_DISPATCH_MODE", raising=False)

    result = storage_trigger._dispatch_payload(
        {"descriptor_gcs_uri": "gs://bucket/scenes/s/captures/c/capture_descriptor.json"}
    )

    assert result == "pubsub:message"
    assert called["count"] == 1


def test_on_swap_dispatch_runs_pipeline(monkeypatch) -> None:
    captured = {}

    def _fake_run(*, descriptor_gcs_uri: str):
        captured["descriptor"] = descriptor_gcs_uri

    monkeypatch.setattr(storage_trigger, "run_capture_pipeline", _fake_run)

    payload = {
        "descriptor_gcs_uri": "gs://bucket/scenes/scene_a/captures/cap_b/capture_descriptor.json"
    }
    event = {"data": base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")}

    storage_trigger.on_swap_dispatch(event, None)

    assert captured["descriptor"] == payload["descriptor_gcs_uri"]


def test_on_swap_dispatch_http(monkeypatch) -> None:
    captured = {}

    def _fake_run(*, descriptor_gcs_uri: str):
        captured["descriptor"] = descriptor_gcs_uri

    monkeypatch.setattr(storage_trigger, "run_capture_pipeline", _fake_run)

    class _Req:
        def __init__(self, payload):
            self._payload = payload

        def get_json(self, silent: bool = True):  # noqa: ARG002
            return self._payload

    ok_resp = storage_trigger.on_swap_dispatch_http(
        _Req({"descriptor_gcs_uri": "gs://bucket/scenes/s/captures/c/capture_descriptor.json"})
    )
    bad_resp = storage_trigger.on_swap_dispatch_http(_Req({}))

    assert ok_resp == ("ok", 200)
    assert bad_resp == ("Missing descriptor_gcs_uri", 400)
    assert captured["descriptor"].startswith("gs://bucket/")
