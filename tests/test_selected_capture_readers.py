"""Reader fault injection at the membership seam; not protected birth proof.

Actual no-override protected staging is separately tested in
test_selected_capture_inputs.py on the ordinary Linux CI runner.
"""
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_retirement_generations as generations
from blueprint_pipeline.capture_bridge import CaptureDescriptor
from blueprint_pipeline.capture_orchestrator import PipelineConfig, run_capture_pipeline
from blueprint_pipeline.common import PipelineError, read_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.frames_layout import load_frames_layout, read_frame_bytes, iter_frame_payloads
from blueprint_pipeline.site_package_orchestrator import run_qualification_pipeline
from tests.test_frames_layout import build_packed_capture, FRAME_PAYLOADS


def reader_fixture(tmp_path, monkeypatch):
    root = tmp_path / "bucket/scenes/site-r1/captures/capture"
    selected = root / "deliveries" / ("a" * 64)
    build_packed_capture(selected / "frames")
    descriptor = {
        "schema_version": "v1", "scene_id": "site-r1", "capture_id": "capture",
        "capture_source": "unknown", "capture_tier": "candidate",
        "raw_prefix_uri": "gs://bucket/scenes/site-r1/captures/capture/raw",
        "frames_index_uri": "gs://bucket/scenes/site-r1/captures/capture/frames/index.jsonl",
        "metadata": {"capture_entry_source": "browser_self_capture"},
    }
    (selected / "capture_descriptor.json").write_text(json.dumps(descriptor))
    members = {path.relative_to(selected).as_posix(): {
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    } for path in selected.rglob("*") if path.is_file()}

    def selected_member(path, relative):
        assert Path(path) == root
        if relative not in members:
            raise ValueError("scene_capture_selected_input_missing")
        return selected / relative, members[relative]

    @contextmanager
    def unit_descriptor(path):
        # Isolate bytes/digest lifetime from the protected-ancestry admission.
        # No production guard is changed and this unit test claims no birth.
        fd = os.open(path, os.O_RDONLY)
        try:
            yield fd, os.fstat(fd)
        finally:
            os.close(fd)

    monkeypatch.setattr(generations, "_capture_birth_input", selected_member)
    monkeypatch.setattr(generations, "_opened", unit_descriptor)
    return root, selected, members


def test_selected_packed_reader_checks_every_archive_and_retains_canonical_root(tmp_path, monkeypatch):
    root, selected, _ = reader_fixture(tmp_path, monkeypatch)
    layout = load_frames_layout(root / "frames")
    assert layout.capture_root == root
    assert dict((record.member_name, payload) for record, payload in iter_frame_payloads(root / "frames")) == FRAME_PAYLOADS
    (selected / "frames/frames_000.tar").write_bytes(b"changed archive")
    with pytest.raises(ValueError, match="scene_capture_selected_input_changed"):
        read_frame_bytes(layout, layout.records[0])


def test_missing_selected_packing_manifest_cannot_enter_legacy_fallback(tmp_path, monkeypatch):
    root, selected, _ = reader_fixture(tmp_path, monkeypatch)
    (selected / "frames/packing_manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_frames_layout(root / "frames")


def test_index_replacement_after_path_verification_cannot_be_consumed(tmp_path, monkeypatch):
    root, selected, _ = reader_fixture(tmp_path, monkeypatch)
    opened = generations._opened
    @contextmanager
    def replace_after_verified_path(path):
        with opened(path) as descriptor:
            yield descriptor
        if path.name == "index.jsonl":
            path.write_text('{"frame_id":"replacement"}\n')
    monkeypatch.setattr(generations, "_opened", replace_after_verified_path)
    with pytest.raises(ValueError, match="scene_capture_selected_input_changed"):
        load_frames_layout(root / "frames")
    assert "replacement" in (selected / "frames/index.jsonl").read_text()


def test_archive_and_descriptor_read_use_verified_snapshot_after_path_replacement(tmp_path, monkeypatch):
    root, selected, _ = reader_fixture(tmp_path, monkeypatch)
    layout = load_frames_layout(root / "frames")
    opened = generations._opened
    @contextmanager
    def replace_after_snapshot(path):
        with opened(path) as descriptor:
            yield descriptor
        if path.suffix == ".tar" or path.name == "capture_descriptor.json":
            path.write_bytes(b"replaced after verified snapshot")
    monkeypatch.setattr(generations, "_opened", replace_after_snapshot)
    assert read_frame_bytes(layout, layout.records[0]) == FRAME_PAYLOADS["000001.jpg"]
    descriptor = CaptureDescriptor.from_file(selected / "capture_descriptor.json")
    assert descriptor.capture_id == "capture"
    with pytest.raises(ValueError, match="scene_capture_selected_input_changed"):
        read_json(selected / "capture_descriptor.json")


def test_selected_descriptor_cannot_displace_canonical_withdrawal_root(tmp_path, monkeypatch):
    root, selected, _ = reader_fixture(tmp_path, monkeypatch)
    tombstone = root.parent.parent / "website_withdrawal/tombstone.json"
    tombstone.parent.mkdir()
    value = {"schema_version": "website_capture_withdrawal_tombstone.v1",
             "request_id": "r1", "scene_id": "site-r1",
             "consent_revoked_at": "2026-10-07T00:00:00Z"}
    value["digest"] = canonical_digest(value, digest_field="digest")
    tombstone.write_text(json.dumps(value))
    def forbidden(**_kwargs):
        pytest.fail("withdrawn capture cannot reach current context, sponsorship, provider or sync")
    monkeypatch.setattr("blueprint_pipeline.site_package_orchestrator.load_current_website_task_context", forbidden)
    monkeypatch.setattr("blueprint_pipeline.site_package_orchestrator.load_website_scene_sponsorship", forbidden)
    uri = "gs://bucket/scenes/site-r1/captures/capture/deliveries/" + ("a" * 64) + "/capture_descriptor.json"
    with pytest.raises(PipelineError, match="website_capture_withdrawn"):
        run_qualification_pipeline(descriptor_gcs_uri=uri, config=PipelineConfig(gcs_root=tmp_path))


def test_selected_descriptor_keeps_lane_resume_and_raw_fingerprint_root_canonical(tmp_path, monkeypatch):
    root, _, _ = reader_fixture(tmp_path, monkeypatch)
    calls = []
    def fingerprint(**kwargs):
        calls.append(kwargs)
        raise OSError("unit stop before any lane")
    monkeypatch.setattr("blueprint_pipeline.capture_orchestrator.lane_ledger_input_fingerprint", fingerprint)
    monkeypatch.setattr("blueprint_pipeline.capture_orchestrator.run_qualification_pipeline", lambda **_: {"status": "completed"})
    uri = "gs://bucket/scenes/site-r1/captures/capture/deliveries/" + ("a" * 64) + "/capture_descriptor.json"
    run_capture_pipeline(descriptor_gcs_uri=uri, requested_lanes=["qualification"], config=PipelineConfig(gcs_root=tmp_path))
    assert len(calls) == 1 and calls[0]["capture_root"] == root


def test_saved_layout_cannot_read_a_later_delivery_with_same_frame_names(tmp_path, monkeypatch):
    root, selected, members = reader_fixture(tmp_path, monkeypatch)
    layout = load_frames_layout(root / "frames")
    later = selected.parent / ("b" * 64)
    later.mkdir()
    def later_member(path, relative):
        assert Path(path) == root
        return later / relative, members[relative]
    monkeypatch.setattr(generations, "_capture_birth_input", later_member)
    with pytest.raises(ValueError, match="scene_capture_selected_input_mismatch"):
        read_frame_bytes(layout, layout.records[0])


def test_generic_json_reader_preserves_shallow_relative_delivery_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("deliveries/key/capture_descriptor.json")
    path.parent.mkdir(parents=True)
    path.write_text('{"legacy":"ordinary relative JSON"}')
    assert read_json(path) == {"legacy": "ordinary relative JSON"}
