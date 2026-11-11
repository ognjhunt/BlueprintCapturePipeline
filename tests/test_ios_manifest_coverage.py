from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path


from blueprint_pipeline import ios_manifest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _bundle_hash(artifacts: dict[str, str]) -> str:
    canonical = "\n".join(f"{path}:{artifacts[path]}" for path in sorted(artifacts))
    return sha256(canonical.encode("utf-8")).hexdigest()


def test_v3_raw_manifest_verifies_hashes_and_fails_mismatch(tmp_path: Path) -> None:
    gcs_root = tmp_path / "gcs"
    raw = gcs_root / "bucket" / "raw"
    _write_json(
        raw / "manifest.json",
        {
            "schema_version": "v3",
            "capture_schema_version": "3.0.0",
            "scene_id": "site",
            "capture_id": "capture",
            "video_uri": "walkthrough.mov",
            "capture_source": "fixture",
            "capture_tier_hint": "test_fixture",
            "capture_profile_id": "fixture_video_only",
            "capture_capabilities": {},
            "coordinate_frame_session_id": "cfs-fixture-1",
            "capture_start_epoch_ms": 1_700_000_000_000,
            "app_version": "1.0.0",
            "app_build": "1",
            "ios_version": "18.0",
            "ios_build": "22A",
            "hardware_model_identifier": "FixtureDevice1,1",
            "device_model_marketing": "Fixture Device",
            "has_lidar": False,
            "depth_supported": False,
            "fps_source": 30.0,
            "width": 1920,
            "height": 1080,
            "rights_profile": "test_fixture",
            "requested_outputs": ["qualification"],
        },
    )
    (raw / "walkthrough.mov").write_bytes(b"video")
    _write_json(
        raw / "capture_upload_complete.json",
        {
            "schema_version": "v1",
            "scene_id": "site",
            "capture_id": "capture",
            "raw_prefix": "raw",
            "completed_at": "2026-07-09T12:00:00Z",
            "status": "complete",
        },
    )
    artifacts = {
        "capture_upload_complete.json": _sha_file(raw / "capture_upload_complete.json"),
        "manifest.json": _sha_file(raw / "manifest.json"),
        "walkthrough.mov": _sha_file(raw / "walkthrough.mov"),
    }
    _write_json(
        raw / "hashes.json",
        {
            "schema_version": "v1",
            "bundle_sha256": _bundle_hash(artifacts),
            "artifacts": artifacts,
        },
    )

    report = ios_manifest.verify_raw_bundle_hashes_path(raw)
    assert report["valid"] is True
    assert report["bundle_sha256_matches"] is True

    (raw / "walkthrough.mov").write_bytes(b"corrupted")
    mismatch = ios_manifest.verify_raw_bundle_hashes_path(raw)
    assert mismatch["valid"] is False
    assert "hash_mismatch:walkthrough.mov" in mismatch["errors"]
    assert "bundle_sha256_mismatch" in mismatch["errors"]

    _write_json(raw / "hashes.json", {"schema_version": "v1", "bundle_sha256": "abc", "artifacts": {}})
    missing_coverage = ios_manifest.verify_raw_bundle_hashes_path(raw)
    assert "hash_coverage_missing:manifest.json" in missing_coverage["errors"]
