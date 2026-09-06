"""Completed splat input remains exact asset evidence, not capture or physics truth."""
import hashlib
import json
import struct
import pytest

from blueprint_pipeline.provided_scene_splat import inspect_splat
from blueprint_pipeline.capture_upload_intake import process_capture_upload_submission
from tests.test_capture_upload_intake import _submission, _opener


def splat_bytes():
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 64\n"
    header += "".join("property float " + name + "\n" for name in names) + "end_header\n"
    rows = [struct.pack("<14f", i % 4, (i // 4) % 4, i // 16, 0, 0, 0, 2, -3, -3, -3, 1, 0, 0, 0)
            for i in range(64)]
    return header.encode() + b"".join(rows)


def test_completed_splat_upload_and_whole_asset_survey_keep_proof_boundaries(tmp_path):
    data = splat_bytes()
    submission = _submission(data)
    submission["request"].update(capture_authority_profile="provided_scene_splat", source_type="provided_scene_splat",
        original_file={"original_filename": "room.ply", "size_bytes": len(data), "media_type": "application/octet-stream"},
        available_sensor_streams=[{"stream_type": "provided_geometry", "status": "available"}],
        coordinate_frame_declaration={"meters_per_unit": 1, "up_axis": "Z"},
        permitted_evidence_uses=["appearance_review"])
    receipt = process_capture_upload_submission(submission, store_root=tmp_path,
        allowed_hosts=["download.example.test"], transfer_opener=_opener(data, {}),
        malware_scanner=lambda _: {"status": "passed", "scanner": "fixture"})
    assert receipt["admission_status"] == "accepted", receipt
    assert receipt["claim_ceiling"]["captured_observation_review"] is False
    assert receipt["claim_ceiling"]["metric_geometry"] is False
    from blueprint_pipeline.reconstruction_control_plane import _source_binding
    source = _source_binding(capture_store_root=tmp_path, capture_session_id=submission["capture_session_id"],
                             intake_id=submission["request"]["intake_id"])
    report = inspect_splat(source["object_path"], coordinate_frame_declaration={"meters_per_unit": 1, "up_axis": "Z"})
    assert report["asset_digest"] == "sha256:" + hashlib.sha256(data).hexdigest()
    assert report["retained_gaussian_count"] == 64
    assert report["whole_retained_splat_surveyed"] is True
    assert report["renderer_qualified"] is False and report["collision_qualified"] is False
    assert source["object_path"].read_bytes() == data
    assert json.loads((source["artifact_root"] / "capture_intake_envelope.json").read_text())["available_sensor_streams"][0]["stream_type"] == "scene_splat"


def test_splat_refuses_missing_scale_or_extra_bytes(tmp_path):
    path = tmp_path / "room.ply"
    path.write_bytes(splat_bytes())
    with pytest.raises(ValueError, match="declared_frame_required"):
        inspect_splat(path, coordinate_frame_declaration={})
    path.write_bytes(splat_bytes() + b"unbound")
    with pytest.raises(ValueError, match="layout_invalid"):
        inspect_splat(path, coordinate_frame_declaration={"meters_per_unit": 1, "up_axis": "Z"})
