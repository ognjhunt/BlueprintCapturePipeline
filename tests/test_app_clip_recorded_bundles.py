"""The iOS recorder's own Raw Contract V3.2 bundles, as the pipeline receives them.

Owner-directed App Clip capture work (website self-capture lane, ADP-009B,
``development_only``). It unblocks no ADP gate: these bundles prove plumbing
between the phone, the WebApp and this pipeline, not a real site.

``tests/fixtures/app_clip_recorded_v3_2/bundles.tar.gz`` holds
``{lidar,nonlidar}/raw``: the raw prefix a Blueprint-WebApp test
(``site-capture-recorded-app-bundles.test.ts``) left in its fake bucket after
completing a bundle through the capture link. One archive rather than a
directory per file keeps this fixture to a single changed path. The bundle
itself was recorded by BlueprintCapture's shared recorder driven by its
synthetic camera, the way the App Clip records (ARKit poses and intrinsics,
depth and confidence on the LiDAR profile, no IMU with the absence declared),
finalized by the app's V3.2 finalizer (``SyntheticCaptureBundleTests``). The
server wrote its own files (manifest, rights, context, intake, hashes, marker).
Nothing in them is a measurement of a real space.

Source: the BlueprintCapture CI artifact ``synthetic-raw-bundles`` of run
35802438288 (commit 7f533a1), completed by the WebApp test with
``BLUEPRINT_EXPORT_COMPLETED_BUNDLES`` and packed with sorted entries, zero
mtimes and owner 0:0, so the same bundles always give the same archive bytes.
Do not edit the archive by hand: ``hashes.json`` covers every file.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from blueprint_pipeline.ios_manifest import verify_canonical_raw_bundle_path
from blueprint_pipeline.local_reconstruction_adapters import LocalArkitMetricScaffoldAdapter
from blueprint_pipeline.materialization import _capture_modality
from blueprint_pipeline.website_capture_entry import (
    SITE_SELF_CAPTURE,
    capture_entry_source,
    is_server_authored_site_self_capture,
    is_website_capture_manifest,
)

ARCHIVE = Path(__file__).parent / "fixtures" / "app_clip_recorded_v3_2" / "bundles.tar.gz"
PROFILES = [
    pytest.param("lidar", "iphone_arkit_lidar", id="lidar"),
    pytest.param("nonlidar", "iphone_arkit_non_lidar", id="nonlidar"),
]


def _raw_copy(profile: str, tmp_path: Path) -> Path:
    """A private extraction, so nothing a check writes can touch the fixture."""
    destination = tmp_path / "recorded"
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        archive.extractall(destination, filter="data")
    return destination / profile / "raw"


def _manifest(raw: Path) -> dict:
    return json.loads((raw / "manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(("profile", "profile_id"), PROFILES)
def test_recorded_bundle_is_verified_canonical_v3_intake(profile: str, profile_id: str, tmp_path: Path) -> None:
    raw = _raw_copy(profile, tmp_path)
    manifest = _manifest(raw)

    report = verify_canonical_raw_bundle_path(
        raw,
        expected_scene_id=manifest["scene_id"],
        expected_capture_id=manifest["capture_id"],
    )

    assert report["errors"] == []
    assert report["status"] == "verified"
    assert report["valid_for_derivation"] is True
    assert report["current_schema"] is True
    assert manifest["schema_version"] == "v3"
    assert manifest["capture_schema_version"] == "3.2.0"
    assert manifest["capture_profile_id"] == profile_id
    # The clip records no IMU and says so; nothing stands in for it.
    assert manifest["capture_capabilities"]["device_imu"] is False
    assert manifest["capture_capabilities"]["device_imu_unavailable_reason"] == "app_clip_runtime"
    assert not (raw / "motion.jsonl").exists()


@pytest.mark.parametrize(("profile", "profile_id"), PROFILES)
def test_recorded_bundle_routes_to_the_website_lane_with_iphone_treatment(
    profile: str, profile_id: str, tmp_path: Path
) -> None:
    manifest = _manifest(_raw_copy(profile, tmp_path))

    assert is_server_authored_site_self_capture(manifest)
    assert capture_entry_source(manifest) == SITE_SELF_CAPTURE
    assert is_website_capture_manifest(manifest)
    # iPhone treatment: the source stays "iphone" and the profile decides.
    assert manifest["capture_source"] == "iphone"
    assert _capture_modality(manifest, {}, "iphone", [], profile == "lidar") == profile_id


@pytest.mark.parametrize(("profile", "profile_id"), PROFILES)
def test_a_changed_byte_is_quarantined(profile: str, profile_id: str, tmp_path: Path) -> None:
    raw = _raw_copy(profile, tmp_path)
    manifest = _manifest(raw)
    poses = raw / "arkit" / "poses.jsonl"
    poses.write_bytes(poses.read_bytes() + b"\n")

    report = verify_canonical_raw_bundle_path(
        raw,
        expected_scene_id=manifest["scene_id"],
        expected_capture_id=manifest["capture_id"],
    )

    assert report["status"] == "quarantined"
    assert report["valid_for_derivation"] is False


def test_recorded_lidar_bundle_passes_the_strict_v32_metric_scaffold(tmp_path: Path) -> None:
    raw = _raw_copy("lidar", tmp_path)
    manifest = _manifest(raw)
    report = verify_canonical_raw_bundle_path(
        raw,
        expected_scene_id=manifest["scene_id"],
        expected_capture_id=manifest["capture_id"],
    )
    assert report["status"] == "verified"

    result = LocalArkitMetricScaffoldAdapter().execute(
        intake_id="app-clip-recorded-lidar",
        # The verified intake digest, as the identity form the adapter binds.
        capture_digest=f"sha256:{report['intake_digest']}",
        capture_root=raw,
        output_root=tmp_path / "derived",
        rights_and_retention={"external_processing": False},
        maximum_frames=4,
    )

    assert result["camera_solution"]["status"] == "raw_contract_3_2_verified"
    assert result["validation_metrics"]["depth_confidence_pair_count"] >= 1
    # A synthetic camera proves the contract, never a metric claim.
    assert result["claim_ceiling"]["metric_scale"] is False
    assert result["claim_ceiling"]["collision_geometry"] is False
    assert result["claim_ceiling"]["physical_task_success"] is False
