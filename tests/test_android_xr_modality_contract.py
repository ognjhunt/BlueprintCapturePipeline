from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.capture_bridge import CaptureDescriptor, build_capture_bundle_constraints
from blueprint_pipeline.capture_orchestrator import resolve_requested_lanes
from blueprint_pipeline.retrieval_index_stage import run_retrieval_index_stage

import pytest

pytestmark = pytest.mark.slow


def _android_xr_descriptor_payload() -> dict:
    return {
        "schema_version": "v1",
        "scene_id": "scene-xr",
        "capture_id": "capture-xr",
        "capture_source": "glasses",
        "source_device": "android_xr_glasses",
        "capture_tier": "tier2_glasses",
        "capture_modality": "android_xr_video_only",
        "capture_profile_id": "android_xr_glasses",
        "capture_capabilities": {
            "camera_pose": False,
            "camera_intrinsics": False,
            "depth": False,
            "depth_confidence": False,
            "geospatial": False,
            "motion_authoritative": False,
        },
        "raw_prefix_uri": "gs://bucket/scenes/scene-xr/captures/capture-xr/raw",
        "raw_video_uri": "gs://bucket/scenes/scene-xr/captures/capture-xr/raw/walkthrough.mp4",
        "frames_index_uri": "gs://bucket/scenes/scene-xr/captures/capture-xr/frames/index.jsonl",
        "geometry_ready": True,
        "geometry_source": "arcore",
        "quoted_payout_cents": 750,
        "quality": {
            "world_model_candidate": True,
            "geometry_ready": True,
            "geometry_source": "arcore",
            "provider_ready": True,
        },
        "scene_memory_capture": {
            "world_model_candidate": True,
            "geometry_expected_downstream": True,
            "geometry_source": "arcore",
        },
        "capture_rights": {
            "derived_scene_generation_allowed": True,
            "capture_contributor_payout_eligible": True,
            "consent_status": "documented",
            "permission_document_uri": "gs://local-blueprint/rights/consent-packet.pdf",
        },
        "capture_mode": {
            "requested_mode": "site_world_candidate",
            "resolved_mode": "site_world_candidate",
        },
        "site_identity": {
            "site_id": "site-xr",
            "site_id_source": "test",
        },
        "requested_outputs": ["preview_simulation", "scene_memory"],
    }


def test_capture_descriptor_preserves_android_xr_video_only_and_strips_false_claims() -> None:
    descriptor = CaptureDescriptor.from_dict(_android_xr_descriptor_payload())

    assert descriptor.capture_source == "glasses"
    assert descriptor.capture_profile_id == "android_xr_glasses"
    assert descriptor.capture_modality == "android_xr_video_only"
    assert descriptor.evidence_tier == "pre_screen_video"
    assert descriptor.geometry_ready is False
    assert descriptor.geometry_source is None
    assert descriptor.quoted_payout_cents is None
    assert descriptor.capture_capabilities["camera_pose"] is False
    assert descriptor.capture_capabilities["depth"] is False
    assert descriptor.capture_capabilities["geospatial"] is False
    assert descriptor.metadata["capture_rights"]["capture_contributor_payout_eligible"] is False
    assert descriptor.metadata["scene_memory_capture"]["world_model_candidate"] is False
    assert descriptor.metadata["scene_memory_capture"]["geometry_expected_downstream"] is False
    assert descriptor.metadata["scene_memory_capture"]["geometry_source"] is None

    constraints = build_capture_bundle_constraints(descriptor)
    assert constraints["capture_profile_id"] == "android_xr_glasses"
    assert constraints["capture_modality"] == "android_xr_video_only"
    assert constraints["capture_capabilities"]["camera_pose"] is False
    assert "quoted_payout_cents" not in constraints


def test_android_xr_video_only_descriptor_resolves_to_qualification_only(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "bucket" / "scenes" / "scene-xr" / "captures" / "capture-xr" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(json.dumps(_android_xr_descriptor_payload()), encoding="utf-8")

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-xr/captures/capture-xr/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == ["qualification"]


def test_retrieval_index_skips_android_xr_video_only_until_new_geometry_contract(tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-xr" / "captures" / "capture-xr"
    capture_root.mkdir(parents=True)
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps(_android_xr_descriptor_payload()),
        encoding="utf-8",
    )

    result = run_retrieval_index_stage(capture_root=capture_root, embedding_model=object())

    assert result == {
        "status": "skipped",
        "reason": "android_xr_video_only_requires_explicit_geometry_contract",
        "capture_id": "capture-xr",
    }
    assert not (capture_root / "pipeline" / "geometry" / "geometry_summary.json").exists()
