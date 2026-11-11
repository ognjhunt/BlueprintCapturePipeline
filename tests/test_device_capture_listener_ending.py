"""A capture-app (device-format) handoff at the host listener is acknowledged and ended.

The capture app's processing path was removed on 2026-11-11. Such a message gets one
permanent-invalid evidence record and an acknowledgement: nothing is claimed, staged or
run, and it is never retried or dead-lettered.
"""

from __future__ import annotations

import json

import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from tests.test_pubsub_handoff_listener import PAYLOAD_BYTES, FakeSubscriber, _install_fake_pubsub, _received

DEVICE_PAYLOAD = {
    "bucket": "capture-bucket",
    "scene_id": "scene-1",
    "capture_id": "capture-1",
    "raw_prefix_uri": "gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
}


@pytest.mark.parametrize("delivery_attempt", [1, 5])
def test_a_device_format_handoff_is_acknowledged_with_one_evidence_record(tmp_path, monkeypatch, delivery_attempt):
    data = json.dumps(DEVICE_PAYLOAD).encode("utf-8")
    subscriber = FakeSubscriber([_received(ack_id="device", data=data, delivery_attempt=delivery_attempt)])
    results = _install_fake_pubsub(
        monkeypatch, subscriber,
        storage_client=object(),  # never used: nothing is staged
        run_e2e=lambda **_k: pytest.fail("a device-format handoff must run nothing"))
    assert listener.pull_and_process(subscription="projects/p/subscriptions/s", storage_root=tmp_path,
                                     provider="openai", max_messages=1, max_delivery_attempts=5) == 1
    assert subscriber.acknowledged == ["device"]
    assert subscriber.ack_deadline_requests == []  # not deferred for a redelivery
    assert results == []  # ended before the processor
    path, = (tmp_path / ".pubsub_delivery_evidence" / "permanent_invalid").glob("*.json")
    evidence = json.loads(path.read_text())
    assert evidence["blockers"] == ["device_capture_pipeline_removed"]
    assert evidence["payload_sha256"] == listener.payload_sha256(data)
    assert not (tmp_path / "capture-bucket").exists()  # nothing claimed or staged
    assert not list((tmp_path / ".pubsub_delivery_evidence").glob("retry_exhausted*"))


def test_a_direct_call_ends_a_device_format_handoff_before_any_claim(tmp_path):
    result = listener.process_handoff_payload(
        json.dumps(DEVICE_PAYLOAD), storage_root=tmp_path, provider="openai",
        run_e2e=lambda **_k: pytest.fail("a device-format handoff must run nothing"), storage_client=object())
    assert (result["status"], result["queue_disposition"], result["blockers"]) == (
        "device_capture_pipeline_removed", "permanent_invalid", ["device_capture_pipeline_removed"])
    assert "capture_root" not in result
    assert not (tmp_path / "capture-bucket").exists()


@pytest.mark.parametrize("ids", [("site-r1", "walkthrough-r1"), ("site-r1", "supplement-" + "a" * 40)])
def test_a_website_handoff_is_not_ended_by_the_device_rule(ids):
    from blueprint_pipeline.pubsub_handoff_scene_operations import device_capture_removed_result

    scene_id, capture_id = ids
    payload = {"bucket": "capture-bucket", "scene_id": scene_id, "capture_id": capture_id,
               "raw_prefix_uri": f"gs://capture-bucket/scenes/{scene_id}/captures/{capture_id}/raw"}
    assert device_capture_removed_result(listener.parse_handoff_payload(json.dumps(payload))) is None
    assert device_capture_removed_result(listener.parse_handoff_payload(PAYLOAD_BYTES)) is None
