from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.native_360_normalization import (
    Native360NormalizationError,
    build_native_360_probe_receipt,
    normalize_native_360_capture,
)


CAPTURE_DIGEST = "sha256:" + "a" * 64
IMPLEMENTATION_DIGEST = "sha256:" + "b" * 64
RUNTIME_DIGEST = "sha256:" + "c" * 64
SOURCE_COMMIT = "d" * 40
MASK_DIGEST = "sha256:" + "e" * 64
CALIBRATION_DIGEST = "sha256:" + "f" * 64
AUTHORITY = {
    "source_capture_rights_valid": True,
    "consent_valid": True,
    "privacy_review_valid": True,
    "retention_authorized": True,
    "local_processing_authorized": True,
    "provider_upload_authorized": False,
    "paid_compute_authorized": False,
}


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _schema() -> dict:
    return json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/native_360_normalization.v1.schema.json"
        ).read_text(encoding="utf-8")
    )


def _rig_calibration(lens_id: str, *, width: int = 3840) -> dict:
    return {
        "lens_id": lens_id,
        "intrinsics": {
            "fx": 1900.0,
            "fy": 1901.0,
            "cx": width / 2,
            "cy": 1920.0,
            "width": width,
            "height": 3840,
        },
        "distortion": {
            "model": "opencv_fisheye",
            "coefficients": [0.01, -0.001, 0.0001, -0.00001],
        },
        "valid_pixel_mask_digest": MASK_DIGEST,
        "calibration_source": "official_sdk_sidecar",
        "calibration_source_digest": CALIBRATION_DIGEST,
    }


def _fixture(
    root: Path,
    *,
    rear_pts: list[float] | None = None,
    rear_calibration_width: int = 3840,
) -> tuple[dict, dict]:
    source = root / "native" / "capture.insv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"immutable-native-insv-fixture")
    source_digest = _digest(source)
    metadata = {
        "schema_version": "native_360_camera_metadata.v1",
        "source_capture_digest": CAPTURE_DIGEST,
        "camera_model": "Insta360 fixture",
        "capture_mode": "dual_fisheye_video",
        "firmware_version": "fixture-1",
        "coordinate_frame_declaration": {
            "units": "meters",
            "handedness": "right_handed",
            "camera_axes": "+x right, +y down, +z forward",
            "rig_frame": "front_lens_optical_center",
        },
        "segments": [
            {
                "sequence_index": 0,
                "segment_id": "segment-0000",
                "files": [
                    {
                        "relative_path": "native/capture.insv",
                        "original_filename": "capture.insv",
                        "size_bytes": source.stat().st_size,
                        "digest": source_digest,
                        "lens_streams": [
                            {"lens_id": "front", "stream_index": 0},
                            {"lens_id": "rear", "stream_index": 1},
                        ],
                    }
                ],
            }
        ],
        "lens_calibrations": [
            _rig_calibration("front"),
            _rig_calibration("rear", width=rear_calibration_width),
        ],
        "rig_extrinsics": {
            "T_front_rear": [
                [1.0, 0.0, 0.0, 0.06],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "calibration_source": "official_sdk_sidecar",
            "calibration_source_digest": CALIBRATION_DIGEST,
        },
        "imu": {"status": "unavailable"},
        "gyro": {"status": "unavailable"},
    }
    front_pts = [0.0, 0.033333, 0.066667]
    receipt = build_native_360_probe_receipt(
        source_file_digest=source_digest,
        runtime_identity="ffprobe-fixture",
        runtime_digest=RUNTIME_DIGEST,
        streams=[
            {
                "stream_index": 0,
                "media_type": "video",
                "codec_name": "hevc",
                "width": 3840,
                "height": 3840,
                "time_base": "1/90000",
                "pts_seconds": front_pts,
                "metadata": {"lens": "front"},
            },
            {
                "stream_index": 1,
                "media_type": "video",
                "codec_name": "hevc",
                "width": 3840,
                "height": 3840,
                "time_base": "1/90000",
                "pts_seconds": rear_pts or front_pts,
                "metadata": {"lens": "rear"},
            },
        ],
        format_metadata={"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
    )
    return metadata, {"native/capture.insv": receipt}


def _normalize(
    capture_root: Path,
    output_root: Path,
    metadata: dict,
    receipts: dict,
    *,
    authority: dict | None = None,
    timestamp: str = "2026-07-30T12:00:00-05:00",
    maximum_source_bytes: int = 1024,
) -> dict:
    return normalize_native_360_capture(
        capture_root=capture_root,
        output_root=output_root,
        intake_id="native-360-fixture",
        capture_digest=CAPTURE_DIGEST,
        camera_metadata=metadata,
        probe_receipts_by_path=receipts,
        source_commit_sha=SOURCE_COMMIT,
        implementation_digest=IMPLEMENTATION_DIGEST,
        authority_used=authority or AUTHORITY,
        timestamp=timestamp,
        maximum_source_bytes=maximum_source_bytes,
    )


def test_native_360_normalization_is_idempotent_and_preserves_source(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    output_root = tmp_path / "output"
    metadata, receipts = _fixture(capture_root)
    source = capture_root / "native/capture.insv"
    before = source.read_bytes()

    first = _normalize(capture_root, output_root, metadata, receipts)
    second = _normalize(
        capture_root,
        output_root,
        metadata,
        receipts,
        timestamp="2099-01-01T00:00:00Z",
    )

    assert first == second
    assert source.read_bytes() == before
    assert first["status"] == "normalized"
    assert first["claim_ceiling"] == "calibrated_camera_rig"
    assert first["proof_effect"] == "calibrated_native_360_rig_only"
    assert first["metric_scale_status"] == "not_established"
    assert first["camera_trajectory_status"] == "not_established"
    assert first["appearance_reconstruction_proven"] is False
    assert first["collision_geometry_proven"] is False
    assert first["original_native_bytes_modified"] is False
    assert first["timestamp"] == "2026-07-30T17:00:00Z"

    artifact_root = next(output_root.glob("native_360_normalization_*"))
    rig = json.loads(
        (artifact_root / "camera_360_rig_declaration.json").read_text(encoding="utf-8")
    )
    binding = json.loads(
        (artifact_root / "dual_fisheye_stream_binding.json").read_text(
            encoding="utf-8"
        )
    )
    assert binding["all_segments_synchronized"] is True
    assert len(binding["segments"][0]["frame_pairs"]) == 3
    assert rig["agent_may_alter_calibration"] is False
    validator = jsonschema.Draft202012Validator(
        _schema(), format_checker=jsonschema.FormatChecker()
    )
    for artifact in (receipts["native/capture.insv"], rig, binding, first):
        validator.validate(artifact)
    for reference in first["artifact_references"].values():
        path = artifact_root / reference["relative_path"]
        assert _digest(path) == reference["digest"]


def test_native_360_unsynchronized_streams_abstain_without_rebinding(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    metadata, receipts = _fixture(
        capture_root,
        rear_pts=[0.0, 0.0345, 0.068],
    )

    result = _normalize(capture_root, tmp_path / "output", metadata, receipts)

    assert result["status"] == "blocked"
    assert result["claim_ceiling"] == "decoded_native_container"
    assert result["proof_effect"] == "none"
    assert "native_360_lens_streams_unsynchronized:0" in result["blockers"]
    artifact_root = next((tmp_path / "output").glob("native_360_normalization_*"))
    binding = json.loads(
        (artifact_root / "dual_fisheye_stream_binding.json").read_text(
            encoding="utf-8"
        )
    )
    assert binding["agent_may_rebind_lens_streams"] is False


def test_native_360_calibration_dimensions_must_match_probed_stream(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    metadata, receipts = _fixture(capture_root, rear_calibration_width=4000)

    result = _normalize(capture_root, tmp_path / "output", metadata, receipts)

    assert result["status"] == "blocked"
    assert (
        "native_360_calibration_stream_dimensions_mismatch:0:rear"
        in result["blockers"]
    )
    assert result["metric_geometry_proven"] is False


def test_native_360_missing_calibration_fails_closed_to_container_claim(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    metadata, receipts = _fixture(capture_root)
    metadata["lens_calibrations"] = metadata["lens_calibrations"][:1]

    result = _normalize(capture_root, tmp_path / "output", metadata, receipts)

    assert result["status"] == "blocked"
    assert "native_360_complete_lens_calibration_missing" in result["blockers"]
    assert result["claim_ceiling"] == "decoded_native_container"


def test_native_360_requires_explicit_non_provider_authority(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    metadata, receipts = _fixture(capture_root)
    authority = dict(AUTHORITY)
    authority["provider_upload_authorized"] = True

    with pytest.raises(Native360NormalizationError, match="native_360_authority_invalid"):
        _normalize(
            capture_root,
            tmp_path / "output",
            metadata,
            receipts,
            authority=authority,
        )


def test_native_360_rejects_unbound_or_tampered_probe_receipt(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    metadata, receipts = _fixture(capture_root)
    receipts["stale.insv"] = receipts["native/capture.insv"]
    with pytest.raises(Native360NormalizationError, match="native_360_unbound_probe_receipt"):
        _normalize(capture_root, tmp_path / "output-a", metadata, receipts)

    metadata, receipts = _fixture(tmp_path / "capture-b")
    receipts["native/capture.insv"] = dict(receipts["native/capture.insv"])
    receipts["native/capture.insv"]["probe_receipt_digest"] = "sha256:" + "0" * 64
    with pytest.raises(Native360NormalizationError, match="native_360_probe_receipt_invalid"):
        _normalize(tmp_path / "capture-b", tmp_path / "output-b", metadata, receipts)


def test_native_360_rejects_path_traversal_symlink_and_oversized_source(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    metadata, receipts = _fixture(capture_root)
    metadata["segments"][0]["files"][0]["relative_path"] = "../escape.insv"
    with pytest.raises(Native360NormalizationError, match="relative_path_unsafe"):
        _normalize(capture_root, tmp_path / "output-a", metadata, receipts)

    capture_root = tmp_path / "capture-b"
    metadata, receipts = _fixture(capture_root)
    source = capture_root / "native/capture.insv"
    external = tmp_path / "external.insv"
    external.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(external)
    with pytest.raises(Native360NormalizationError, match="source_symlink_forbidden"):
        _normalize(capture_root, tmp_path / "output-b", metadata, receipts)

    capture_root = tmp_path / "capture-c"
    metadata, receipts = _fixture(capture_root)
    with pytest.raises(Native360NormalizationError, match="source_oversized"):
        _normalize(
            capture_root,
            tmp_path / "output-c",
            metadata,
            receipts,
            maximum_source_bytes=1,
        )


def test_native_360_rejects_invalid_timestamp_and_duplicate_pts(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    metadata, receipts = _fixture(capture_root)
    with pytest.raises(Native360NormalizationError, match="native_360_timestamp_invalid"):
        _normalize(
            capture_root,
            tmp_path / "output",
            metadata,
            receipts,
            timestamp="2026-07-30T12:00:00",
        )

    source_digest = receipts["native/capture.insv"]["source_file_digest"]
    with pytest.raises(
        Native360NormalizationError,
        match="native_360_pts_not_strictly_increasing:stream_0",
    ):
        build_native_360_probe_receipt(
            source_file_digest=source_digest,
            runtime_identity="ffprobe-fixture",
            runtime_digest=RUNTIME_DIGEST,
            streams=[
                {
                    "stream_index": 0,
                    "media_type": "video",
                    "codec_name": "hevc",
                    "width": 3840,
                    "height": 3840,
                    "pts_seconds": [0.0, 0.0],
                }
            ],
            format_metadata={},
        )
