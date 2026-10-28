"""The website's short-view source set survives the real consumer boundaries."""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import google_crc32c
import pytest

from blueprint_pipeline import website_recording_sources as sources
from blueprint_pipeline import website_recording_composition as composition
from blueprint_pipeline import website_task_context as context
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def vector():
    return json.loads((Path(__file__).parent / "fixtures/website-additional-views-vector.json").read_text())


def validate(binding):
    return sources.recording_sources(binding, request_id="req1", scene_id="site-req1",
        capture_id=binding["capture_id"], bucket="synthetic")


def test_actual_webapp_vector_keeps_both_sources_in_current_task_context():
    value = vector()["context"]
    actual = context.validate_website_task_context(value, request_id="req1", scene_id="site-req1", capture_id=value["capture_id"])
    assert actual == value
    rows = validate(actual["capture_binding"])
    assert [row["capture_id"] for row in rows] == ["walkthrough-req1", value["capture_id"]]
    assert rows[0]["raw_video"]["generation"] == "123"
    assert actual["capture_binding"]["coordinate_frames_independent"] is True


def test_verification_enrichment_does_not_change_the_immutable_method_sources():
    binding = vector()["context"]["capture_binding"]
    original = validate(binding)
    binding["lineage"][0]["child"]["verification_status"] = "verified"
    binding["lineage"][0]["parent"]["verification_status"] = "verified"
    assert validate(binding) == original


@pytest.mark.parametrize("fault", ["drop_original", "reorder", "different_bucket", "other_job", "wrong_generation",
                                 "unknown_mode", "scope_changed", "extra_source", "missing_video"])
def test_an_incomplete_or_changed_source_set_never_buys_or_decodes(fault):
    value = vector()["context"]["capture_binding"]
    entry = value["lineage"][0]
    if fault == "drop_original":
        value["recording_capture_ids"] = value["recording_capture_ids"][1:]
    elif fault == "reorder":
        value["recording_capture_ids"].reverse()
    elif fault == "different_bucket":
        entry["parent"]["raw_manifest_uri"] = entry["parent"]["raw_manifest_uri"].replace("gs://synthetic/", "gs://other/")
        entry["supplement"]["parent_manifest_uri"] = entry["parent"]["raw_manifest_uri"]
    elif fault == "other_job":
        entry["parent"]["raw_video"]["object_name"] = "scenes/site-other/captures/walkthrough-other/raw/walkthrough.mov"
    elif fault == "wrong_generation":
        entry["parent"]["raw_video"]["generation"] = "unknown"
    elif fault == "unknown_mode":
        entry["recording_scope"] = "registered_geometry"
    elif fault == "scope_changed":
        entry["recording_scope"] = "complete_walkthrough"
    elif fault == "extra_source":
        value["recording_capture_ids"].insert(0, "walkthrough-other")
    else:
        entry["parent"].pop("raw_video")
    with pytest.raises(ValueError, match="binding_invalid"):
        validate(value)


def test_v2_cannot_be_downgraded_to_ignore_the_original(tmp_path):
    value = vector()["context"]
    value["capture_binding"]["schema_version"] = "website_capture_continuation.v1"
    value["capture_binding"].pop("recording_capture_ids")
    value["context_digest"] = canonical_digest(value, digest_field="context_digest")
    with pytest.raises(ValueError, match="continuation_invalid"):
        context.validate_website_task_context(value, request_id="req1", scene_id="site-req1", capture_id=value["capture_id"])
    with pytest.raises(ValueError, match="binding_invalid"):
        sources.resolve_recording_video(task_context=value, capture_root=tmp_path, source_video=None, bucket="synthetic")


def source_set(tmp_path, monkeypatch):
    value = vector()["context"]
    root = tmp_path / f"synthetic/scenes/site-req1/captures/{value['capture_id']}"
    (root / "raw").mkdir(parents=True)
    payloads = {"walkthrough-req1": b"original recording", value["capture_id"]: b"extra angle"}
    for entry in value["capture_binding"]["lineage"]:
        for row in (entry["child"], entry["parent"]):
            data = payloads[row["capture_id"]]
            row["raw_video"].update(size_bytes=len(data),
                crc32c=base64.b64encode(google_crc32c.Checksum(data).digest()).decode(),
                sha256="sha256:" + hashlib.sha256(data).hexdigest())
    child = root / "raw/walkthrough.mov"
    child.write_bytes(payloads[value["capture_id"]])
    downloaded, processed = [], []

    def download(row, target):
        downloaded.append(deepcopy(row))
        target.write_bytes(payloads[row["capture_id"]])

    def compose(**kwargs):
        processed.append(kwargs)
        path = kwargs["output_root"] / "method.mp4"
        path.write_bytes(b"derived method")
        return {"source_video": str(path)}

    monkeypatch.setattr(composition, "compose_recordings", compose)
    return value, root, child, payloads, downloaded, processed, download


def test_reuses_verified_child_and_fetches_only_the_exact_missing_original(tmp_path, monkeypatch):
    value, root, child, payloads, downloaded, processed, download = source_set(tmp_path, monkeypatch)
    actual = sources.resolve_recording_video(task_context=value, capture_root=root, source_video=child, bucket="synthetic", download=download)
    assert actual.read_bytes() == b"derived method"
    assert [(row["capture_id"], row["raw_video"]["generation"]) for row in downloaded] == [("walkthrough-req1", "123")]
    assert [row["capture_id"] for row in processed[0]["recordings"]] == ["walkthrough-req1", value["capture_id"]]
    assert [Path(row["source_video"]).read_bytes() for row in processed[0]["recordings"]] == list(payloads.values())
    assert child.read_bytes() == b"extra angle"
    sources.resolve_recording_video(task_context=value, capture_root=root, source_video=child, bucket="synthetic", download=download)
    assert len(downloaded) == 1


def test_corrupt_download_never_reaches_method_input_and_original_stays_saved(tmp_path, monkeypatch):
    value, root, child, _, _, processed, _ = source_set(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="source_changed"):
        sources.resolve_recording_video(task_context=value, capture_root=root, source_video=child, bucket="synthetic",
            download=lambda _row, target: target.write_bytes(b"different bytes"))
    assert not processed and child.read_bytes() == b"extra angle"
    assert not list(root.rglob(".download-*"))


@pytest.mark.parametrize("fault", ["revoked", "wrong_bucket", "output_symlink"])
def test_rights_and_output_paths_are_checked_before_download(tmp_path, monkeypatch, fault):
    value, root, child, _, downloaded, processed, download = source_set(tmp_path, monkeypatch)
    bucket = "synthetic"
    if fault == "revoked":
        value["capture_rights"]["derived_scene_generation_allowed"] = False
    elif fault == "wrong_bucket":
        bucket = "different"
    else:
        outside = tmp_path / "outside"
        outside.mkdir()
        (root / "pipeline").mkdir()
        (root / "pipeline/recording_inputs").symlink_to(outside)
    with pytest.raises(ValueError):
        sources.resolve_recording_video(task_context=value, capture_root=root, source_video=child, bucket=bucket, download=download)
    assert not downloaded and not processed


def test_original_and_complete_walkthrough_updates_keep_the_existing_path(tmp_path):
    path = tmp_path / "original.mov"
    assert sources.resolve_recording_video(task_context={}, capture_root=tmp_path, source_video=path, bucket="synthetic") == path


@pytest.mark.slow
def test_real_videos_reach_existing_website_preparation_as_one_bound_method_input(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tests.test_website_recording_composition import media
    from tests.test_qualification_coverage_edges import _descriptor
    from blueprint_pipeline import site_package_orchestrator as orchestrator, website_scene_handoff
    from blueprint_pipeline.local_reconstruction_adapters import _probe_video, _sha256_file

    _, ffprobe, clips = media.__wrapped__(tmp_path)
    task = vector()["context"]
    task.update(purpose="scene_preparation", confirmed=False, confirmed_at=None)
    capture_id = task["capture_id"]
    storage = tmp_path / "environment-specific-storage-root"
    root = storage / f"scenes/site-req1/captures/{capture_id}"
    raw = []
    for source, clip in zip((task["capture_binding"]["lineage"][0]["parent"], task["capture_binding"]["lineage"][0]["child"]), clips, strict=True):
        data = Path(clip["source_video"]).read_bytes()
        target = storage / source["raw_video"]["object_name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        raw.append(target)
        source["raw_video"].update(size_bytes=len(data), crc32c=base64.b64encode(google_crc32c.Checksum(data).digest()).decode(),
                                   sha256="sha256:" + hashlib.sha256(data).hexdigest())
    task["context_digest"] = canonical_digest(task, digest_field="context_digest")
    prefix = f"gs://synthetic/scenes/site-req1/captures/{capture_id}"
    descriptor = _descriptor(scene_id="site-req1", capture_id=capture_id, site_submission_id="req1",
        raw_prefix_uri=prefix + "/raw", raw_video_uri=prefix + "/raw/walkthrough.mov", frames_index_uri=None,
        capture_source="unknown", capture_modality="video_only", requested_outputs=["preview_simulation"],
        metadata={"capture_entry_source": "browser_self_capture"})
    path = root / "capture_descriptor.json"
    path.write_text(json.dumps(descriptor.to_dict()))
    original_hashes = [_sha256_file(value) for value in raw]
    monkeypatch.setenv("BLUEPRINT_WEBSITE_ATLAS_ROUTING", "off")
    monkeypatch.setattr(orchestrator, "load_current_website_task_context", lambda **_: task)
    monkeypatch.setattr(orchestrator, "load_website_scene_sponsorship", lambda **_: {"purpose": "scene_preparation"})
    received = []

    def prepare(**kwargs):
        method = kwargs["website_source_video"]
        received.append(method)
        assert method.is_relative_to(root / "pipeline") and method not in raw
        assert _probe_video(method, ffprobe)["frame_count"] == 12
        receipt = json.loads((method.parent / "composition.json").read_text())
        assert [value["capture_id"] for value in receipt["sources"]] == ["walkthrough-req1", capture_id]
        assert receipt["source_video_digest"] == _sha256_file(method)
        return {"status": "noop", "privacy_status": "website_capture_admitted", "privacy_verified": True}

    monkeypatch.setattr(orchestrator, "run_clean_plate_stage", prepare)
    monkeypatch.setattr(website_scene_handoff, "prepare_website_scene_handoff", lambda **_: {
        "status": "intake_ready", "website_intake_outbox": {"state": "forward_pending"}})
    result = orchestrator.run_qualification_pipeline(descriptor_gcs_uri=prefix + "/capture_descriptor.json",
        config=SimpleNamespace(gcs_root=storage))
    assert result["status"] == "completed" and result["native_execution_complete"] is False
    assert len(received) == 1 and [_sha256_file(value) for value in raw] == original_hashes
    assert json.loads(path.read_text())["raw_video_uri"] == descriptor.raw_video_uri
