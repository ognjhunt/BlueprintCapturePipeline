"""Original owner reads never turn a delivered marker into local birth authority."""

import json
import time
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest


def observation():
    request_id = "req-1"
    prefix = "scenes/site-req-1/captures/walkthrough-req-1"
    value = {
        "schema_version": "website_capture_owner_observation.v1",
        "request_id": request_id,
        "scene_id": "site-req-1",
        "capture_id": "walkthrough-req-1",
        "bucket": "capture-bucket",
        "raw_prefix_uri": f"gs://capture-bucket/{prefix}/raw",
        "capture_owner": {"user_id": "owner-1", "basis": "inboundRequests.account_owner_uid"},
        "source_document": {"collection": "inboundRequests", "document_id": request_id,
                            "update_time": {"seconds": 1700000000, "nanoseconds": 1}},
        "ownership_record": {"claimed_at_iso": None},
        "consent_attestation": {"granted": True, "statement_version": "2026-09-18.v1",
                                "recorded_at_iso": "2026-09-28T00:00:00Z"},
        "capture_rights": {"derived_scene_generation_allowed": True,
                           "data_licensing_allowed": False,
                           "capture_contributor_payout_eligible": False,
                           "consent_status": "granted", "consent_revoked": False,
                           "consent_scope": ["derived_scene_generation", "robot_evaluation"],
                           "statement_version": "2026-09-18.v1",
                           "recorded_at_iso": "2026-09-28T00:00:00Z"},
        "completion_marker": {"object_name": f"{prefix}/raw/capture_upload_complete.json",
                              "generation": "17000000000000000001", "size_bytes": 30,
                              "sha256": "sha256:" + "a" * 64},
        "producer_delivery": {"kind": "website_browser_capture_delivery",
                              "delivery_key": "sha256:" + "b" * 64,
                              "server_record": {"object_name": f"{prefix}/upload/producer_deliveries/browser-video-17000000000000000000.json",
                                                "generation": "17000000000000000002", "size_bytes": 500,
                                                "sha256": "sha256:" + "c" * 64},
                              "raw_video": {"object_name": f"{prefix}/raw/walkthrough.mov",
                                            "generation": "17000000000000000000", "size_bytes": 9,
                                            "crc32c": "AAAAAA=="}},
        "observed_at_epoch": int(time.time()),
        "valid_until_epoch": int(time.time()) + 60,
    }
    value["source_projection_digest"] = cross_runtime_canonical_digest({
        key: value[key] for key in ("request_id", "scene_id", "capture_id", "bucket",
                                    "raw_prefix_uri", "capture_owner", "ownership_record",
                                    "consent_attestation", "capture_rights", "completion_marker",
                                    "producer_delivery")})
    value["observation_digest"] = cross_runtime_canonical_digest(value)
    return value


def test_observer_refuses_wrong_generation_and_rights():
    from blueprint_pipeline.capture_original_owner_observer import validate_observation

    value = observation()
    assert validate_observation(value, bucket="capture-bucket", scene_id="site-req-1",
                                capture_id="walkthrough-req-1",
                                marker_generation="17000000000000000001") == value
    with pytest.raises(ValueError):
        validate_observation(value, bucket="capture-bucket", scene_id="site-req-1",
                             capture_id="walkthrough-req-1", marker_generation="17000000000000000002")
    value["capture_rights"]["consent_revoked"] = True
    value["observation_digest"] = cross_runtime_canonical_digest(value, digest_field="observation_digest")
    with pytest.raises(ValueError):
        validate_observation(value, bucket="capture-bucket", scene_id="site-req-1",
                             capture_id="walkthrough-req-1", marker_generation="17000000000000000001")


@pytest.mark.parametrize("purpose", [None, "scene_preparation"])
def test_selected_delivery_refuses_before_lease_when_owner_api_missing(tmp_path: Path, monkeypatch, purpose):
    from blueprint_pipeline import pubsub_handoff_listener as listener
    from blueprint_pipeline import capture_original_owner_observer as observer

    calls = []

    def unavailable(**_kwargs):
        calls.append(_kwargs)
        raise ValueError("capture_owner_unavailable")

    monkeypatch.setattr(observer, "load_original_owner_observation", unavailable)
    payload = {"bucket": "capture-bucket", "scene_id": "site-req-1",
               "capture_id": "walkthrough-req-1",
               "raw_prefix_uri": "gs://capture-bucket/scenes/site-req-1/captures/walkthrough-req-1/raw",
               "source_finalize": {"bucket": "capture-bucket",
                                   "object_name": "scenes/site-req-1/captures/walkthrough-req-1/raw/capture_upload_complete.json",
                                   "generation": "17000000000000000001", "event_id": "evt-1",
                                   "event_source": "storage"}}
    result = listener.process_handoff_payload(payload, storage_root=tmp_path, provider="openai",
                                               run_e2e=lambda **_: pytest.fail("ran"),
                                               expected_preparation_purpose=purpose)
    assert result["queue_disposition"] == "retryable"
    assert calls[0].get("expected_purpose") == purpose
    assert not (tmp_path / "capture-bucket" / "scenes" / "site-req-1" / "captures" /
                "walkthrough-req-1").exists()


@pytest.mark.parametrize("purpose", [None, "scene_preparation"])
def test_native_signed_owner_read_keeps_unclaimed_preparation_scope(tmp_path, monkeypatch, purpose):
    from types import SimpleNamespace
    from blueprint_pipeline import pubsub_handoff_listener as listener
    from blueprint_pipeline import capture_original_owner_observer as observer

    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://tryblueprint.io/api/internal/pipeline/sync")
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "fixture-secret")
    calls = []

    def respond(_url, **kwargs):
        command = json.loads(kwargs["data"])
        calls.append(command)
        value = observation()
        value["capture_owner"] = None
        if command.get("purpose"):
            value["purpose"] = command["purpose"]
        value["source_projection_digest"] = cross_runtime_canonical_digest({
            key: value[key] for key in (*observer._SOURCE_KEYS, *(("purpose",) if "purpose" in value else ()))})
        value["observation_digest"] = cross_runtime_canonical_digest(value, digest_field="observation_digest")
        return SimpleNamespace(status=200, body=json.dumps(value).encode())

    monkeypatch.setattr(observer, "safe_request", respond)
    payload = {"bucket": "capture-bucket", "scene_id": "site-req-1", "capture_id": "walkthrough-req-1",
               "raw_prefix_uri": "gs://capture-bucket/scenes/site-req-1/captures/walkthrough-req-1/raw",
               "source_finalize": {"bucket": "capture-bucket",
                                   "object_name": "scenes/site-req-1/captures/walkthrough-req-1/raw/capture_upload_complete.json",
                                   "generation": "17000000000000000001", "event_id": "evt-1", "event_source": "storage"}}
    result = listener.process_handoff_payload(payload, storage_root=tmp_path, provider="openai",
        expected_preparation_purpose=purpose, run_e2e=lambda **_: pytest.fail("ran"))
    assert calls[0].get("purpose") == purpose
    assert result["status"] == ("capture_original_birth_unavailable_retryable" if purpose
                                else "capture_owner_observation_unavailable_retryable")
    assert not (tmp_path / "capture-bucket").exists()


def test_signed_read_keeps_exact_generation_and_response_bound(monkeypatch):
    from types import SimpleNamespace
    from blueprint_pipeline import capture_original_owner_observer as observer

    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://tryblueprint.io/api/internal/pipeline/sync")
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "fixture-secret")
    calls = []

    def respond(url, **kwargs):
        calls.append((url, kwargs))
        body = json.loads(kwargs["data"])
        assert body == {"request_id": "req-1", "scene_id": "site-req-1",
                        "completion_marker_generation": "17000000000000000001",
                        "remaining_timeout_ms": 4321}
        assert kwargs["headers"]["X-Blueprint-Pipeline-Signature"].startswith("sha256=")
        return SimpleNamespace(status=200, body=json.dumps(observation()).encode())

    monkeypatch.setattr(observer, "safe_request", respond)
    got = observer.load_original_owner_observation(
        bucket="capture-bucket", scene_id="site-req-1", capture_id="walkthrough-req-1",
        marker_generation="17000000000000000001", remaining_timeout_ms=4321)
    assert got["capture_owner"]["user_id"] == "owner-1"
    assert calls[0][0].endswith("/creator-captures/walkthrough-req-1/capture-owner")
    assert calls[0][1]["max_response_bytes"] == 65536
    assert calls[0][1]["timeout_seconds"] == 4.321
    observed, size = observer.load_original_owner_observation(
        bucket="capture-bucket", scene_id="site-req-1", capture_id="walkthrough-req-1",
        marker_generation="17000000000000000001", remaining_timeout_ms=4321,
        include_response_bytes=True)
    assert observed == got and size == len(json.dumps(observation()).encode())

    monkeypatch.setattr(observer, "safe_request", lambda *_args, **_kwargs: SimpleNamespace(
        status=200, body=b"{" + b" " * 65536 + b"}"))
    with pytest.raises(observer.CaptureOwnerObservationError):
        observer.load_original_owner_observation(
            bucket="capture-bucket", scene_id="site-req-1", capture_id="walkthrough-req-1",
            marker_generation="17000000000000000001")


def test_signed_read_rejects_duplicate_keys_even_when_outer_digest_is_valid(monkeypatch):
    from types import SimpleNamespace
    from blueprint_pipeline import capture_original_owner_observer as observer

    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://tryblueprint.io/api/internal/pipeline/sync")
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "fixture-secret")
    raw = json.dumps(observation()).encode().replace(
        b'"capture_owner":', b'"capture_owner":{"user_id":"forged"},"capture_owner":', 1)
    monkeypatch.setattr(observer, "safe_request", lambda *_args, **_kwargs: SimpleNamespace(status=200, body=raw))
    with pytest.raises(observer.CaptureOwnerObservationError, match="capture_owner_response_duplicate_key"):
        observer.load_original_owner_observation(
            bucket="capture-bucket", scene_id="site-req-1", capture_id="walkthrough-req-1",
            marker_generation="17000000000000000001")


def test_valid_observation_still_cannot_enter_legacy_birth_or_direct_stage(tmp_path, monkeypatch):
    from blueprint_pipeline import pubsub_handoff_listener as listener
    from blueprint_pipeline import capture_original_owner_observer as observer
    from blueprint_pipeline.common import PipelineError

    monkeypatch.setattr(observer, "load_original_owner_observation", lambda **_: observation())
    payload = {"bucket": "capture-bucket", "scene_id": "site-req-1",
               "capture_id": "walkthrough-req-1",
               "raw_prefix_uri": "gs://capture-bucket/scenes/site-req-1/captures/walkthrough-req-1/raw",
               "source_finalize": {"bucket": "capture-bucket",
                                   "object_name": "scenes/site-req-1/captures/walkthrough-req-1/raw/capture_upload_complete.json",
                                   "generation": "17000000000000000001", "event_id": "evt-1",
                                   "event_source": "storage"}}
    result = listener.process_handoff_payload(payload, storage_root=tmp_path, provider="openai",
                                               run_e2e=lambda **_: pytest.fail("ran"))
    assert result["status"] == "capture_original_birth_unavailable_retryable"
    handoff = listener.parse_handoff_payload(payload)
    with pytest.raises(PipelineError, match="capture_original_birth_unavailable"):
        listener.stage_handoff_capture(handoff, storage_root=tmp_path)
    assert not (tmp_path / "capture-bucket" / "scenes" / "site-req-1" / "captures" /
                "walkthrough-req-1").exists()
