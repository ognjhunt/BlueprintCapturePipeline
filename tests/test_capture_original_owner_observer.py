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
@pytest.mark.parametrize("error_kind", ["typed", "generic", "unsafe_typed"])
def test_selected_delivery_refuses_before_lease_when_owner_api_missing(tmp_path: Path, monkeypatch, purpose, error_kind):
    from blueprint_pipeline import pubsub_handoff_listener as listener
    from blueprint_pipeline import capture_original_owner_observer as observer

    calls = []

    def unavailable(**_kwargs):
        calls.append(_kwargs)
        if error_kind == "typed":
            raise observer.CaptureOwnerObservationError("capture_owner_purpose_mismatch")
        if error_kind == "unsafe_typed":
            raise observer.CaptureOwnerObservationError("private response token=secret")
        raise ValueError("private response token=secret")

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
    assert result["owner_observation_reason"] == ("capture_owner_purpose_mismatch" if error_kind == "typed"
                                                   else "capture_owner_response_unavailable")
    assert "private" not in json.dumps(result) and "secret" not in json.dumps(result)
    assert calls[0].get("expected_purpose") == purpose
    assert not (tmp_path / "capture-bucket" / "scenes" / "site-req-1" / "captures" /
                "walkthrough-req-1").exists()


@pytest.mark.parametrize("purpose", [None, "scene_preparation"])
def test_native_signed_owner_read_admits_unclaimed_capture_only_in_preparation_scope(tmp_path, monkeypatch, purpose):
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
    # The plain read refuses an unclaimed owner; the listener then reads in preparation scope.
    assert [call.get("purpose") for call in calls[1:2]] == (["scene_preparation"] if purpose is None else [])
    assert result["status"] == "capture_original_birth_unavailable_retryable"
    assert not (tmp_path / "capture-bucket").exists()


def test_native_owner_read_never_moves_a_claimed_capture_into_preparation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from blueprint_pipeline import pubsub_handoff_listener as listener
    from blueprint_pipeline import capture_original_owner_observer as observer

    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://tryblueprint.io/api/internal/pipeline/sync")
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "fixture-secret")
    calls = []

    def respond(_url, **kwargs):
        command = json.loads(kwargs["data"])
        calls.append(command)
        if not command.get("purpose"):
            return SimpleNamespace(status=503, body=b"{}")
        value = observation()  # claimed owner-1
        value["purpose"] = command["purpose"]
        value["source_projection_digest"] = cross_runtime_canonical_digest({
            key: value[key] for key in (*observer._SOURCE_KEYS, "purpose")})
        value["observation_digest"] = cross_runtime_canonical_digest(value, digest_field="observation_digest")
        return SimpleNamespace(status=200, body=json.dumps(value).encode())

    monkeypatch.setattr(observer, "safe_request", respond)
    payload = {"bucket": "capture-bucket", "scene_id": "site-req-1", "capture_id": "walkthrough-req-1",
               "raw_prefix_uri": "gs://capture-bucket/scenes/site-req-1/captures/walkthrough-req-1/raw",
               "source_finalize": {"bucket": "capture-bucket",
                                   "object_name": "scenes/site-req-1/captures/walkthrough-req-1/raw/capture_upload_complete.json",
                                   "generation": "17000000000000000001", "event_id": "evt-1", "event_source": "storage"}}
    result = listener.process_handoff_payload(payload, storage_root=tmp_path, provider="openai",
                                              run_e2e=lambda **_: pytest.fail("ran"))
    assert [call.get("purpose") for call in calls] == [None, "scene_preparation"]
    assert result["status"] == "capture_owner_observation_unavailable_retryable"
    assert result["owner_observation_reason"] == "capture_owner_response_unavailable"
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


def supplemental_observation():
    value = observation()
    original = value["capture_id"]
    child_id = "supplement-" + "d" * 40
    # Every produced object moves together; no original artifact is overwritten.
    def replace(item):
        if isinstance(item, str):
            return item.replace(original, child_id)
        if isinstance(item, dict):
            return {key: replace(child) for key, child in item.items()}
        return item
    value = replace(value)
    value["capture_owner"]["basis"] = "inboundRequests.capture_owner_uid"
    def source(capture, digest):
        return {"capture_id": capture, "raw_bundle_digest": digest,
                "raw_manifest_uri": f"gs://capture-bucket/scenes/site-req-1/captures/{capture}/raw/manifest.json",
                "upload_completion_digest": value["completion_marker"]["sha256"],
                "verification_status": "pending_pipeline_storage_readback",
                "completion_generation": value["completion_marker"]["generation"],
                "completion_record_generation": value["producer_delivery"]["server_record"]["generation"],
                "manifest_generation": "17000000000000000003"}
    child = source(child_id, value["producer_delivery"]["delivery_key"])
    parent = source(original, "sha256:" + "e" * 64)
    value["capture_binding"] = {"schema_version": "website_capture_continuation.v1",
        "original_capture_id": original, "capture_id": child_id, "coordinate_frames_independent": True,
        "lineage": [{"child": child, "parent": parent, "supplement": {
            "parent_capture_id": original, "parent_bundle_digest": parent["raw_bundle_digest"],
            "parent_manifest_uri": parent["raw_manifest_uri"], "coverage_digest": "f" * 64}}]}
    return seal_supplemental_observation(value)


def seal_supplemental_observation(value):
    from blueprint_pipeline.capture_original_owner_observer import _SOURCE_KEYS
    value["source_projection_digest"] = cross_runtime_canonical_digest({key: value[key] for key in
        (*_SOURCE_KEYS, "capture_binding")})
    value["observation_digest"] = cross_runtime_canonical_digest(value, digest_field="observation_digest")
    return value


def test_supplemental_owner_observation_retains_same_job_lineage_and_new_source():
    from blueprint_pipeline.capture_original_owner_observer import validate_observation
    value = supplemental_observation()
    assert validate_observation(value, bucket="capture-bucket", scene_id=value["scene_id"],
        capture_id=value["capture_id"], marker_generation=value["completion_marker"]["generation"]) == value


@pytest.mark.parametrize("fault", ["missing", "cross_job", "wrong_marker", "wrong_source", "changed_parent", "cycle"])
def test_supplemental_owner_observer_refuses_unbound_lineage_even_with_resealed_envelope(fault):
    from blueprint_pipeline.capture_original_owner_observer import validate_observation
    value = supplemental_observation()
    row = value["capture_binding"]["lineage"][0]
    if fault == "cross_job":
        row["parent"]["capture_id"] = "walkthrough-other"
    if fault == "wrong_marker":
        row["child"]["upload_completion_digest"] = "sha256:" + "0" * 64
    if fault == "wrong_source":
        row["child"]["raw_bundle_digest"] = "sha256:" + "0" * 64
    if fault == "changed_parent":
        row["parent"]["raw_bundle_digest"] = "sha256:" + "0" * 64
    if fault == "cycle":
        row["parent"] = dict(row["child"])
    seal_supplemental_observation(value)
    if fault == "missing":
        value.pop("capture_binding")
    with pytest.raises(ValueError):
        validate_observation(value, bucket="capture-bucket", scene_id=value["scene_id"],
            capture_id=value["capture_id"], marker_generation=value["completion_marker"]["generation"])


def test_supplement_owner_read_derives_request_from_scene_and_requires_signed_binding(monkeypatch):
    from types import SimpleNamespace
    from blueprint_pipeline import capture_original_owner_observer as observer
    value = supplemental_observation()
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://tryblueprint.io/api/internal/pipeline/sync")
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "fixture-secret")
    calls = []
    def respond(url, **kwargs):
        calls.append(json.loads(kwargs["data"]))
        return SimpleNamespace(status=200, body=json.dumps(value).encode())
    monkeypatch.setattr(observer, "safe_request", respond)
    assert observer.load_original_owner_observation(bucket="capture-bucket", scene_id=value["scene_id"],
        capture_id=value["capture_id"], marker_generation=value["completion_marker"]["generation"]) == value
    assert calls[0]["request_id"] == "req-1"


def test_repair_keeps_unprocessed_original_as_history_without_inventing_completion():
    from blueprint_pipeline.capture_original_owner_observer import validate_observation
    value = supplemental_observation()
    parent = value["capture_binding"]["lineage"][0]["parent"]
    parent.update(source_state="retained_unprocessed", upload_completion_digest=None,
                  completion_generation=None, completion_record_generation=None, manifest_generation=None)
    seal_supplemental_observation(value)
    assert validate_observation(value, bucket="capture-bucket", scene_id=value["scene_id"],
        capture_id=value["capture_id"], marker_generation=value["completion_marker"]["generation"]) == value
    # The current source must still have completed producer/marker proof.
    value["capture_binding"]["lineage"][0]["child"].update(source_state="retained_unprocessed", completion_generation=None)
    seal_supplemental_observation(value)
    with pytest.raises(ValueError):
        validate_observation(value, bucket="capture-bucket", scene_id=value["scene_id"],
            capture_id=value["capture_id"], marker_generation=value["completion_marker"]["generation"])
