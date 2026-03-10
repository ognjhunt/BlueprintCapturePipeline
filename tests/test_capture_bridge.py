"""Tests for capture descriptor parsing and bridge adapters."""

import pytest

from blueprint_pipeline.capture_bridge import (
    CaptureDescriptor,
    build_capture_bundle_constraints,
    build_scene_manifest_seed,
    build_scene_request_from_descriptor,
)


def _sample_descriptor_payload() -> dict:
    return {
        "schema_version": "v1",
        "scene_id": "scene_kitchen_001",
        "capture_id": "capture_2026_02_16",
        "capture_source": "iphone",
        "capture_tier": "tier1_iphone",
        "raw_prefix_uri": "gs://bucket/scenes/scene_kitchen_001/iphone/capture_2026_02_16/raw",
        "frames_index_uri": "gs://bucket/scenes/scene_kitchen_001/captures/capture_2026_02_16/frames/index.jsonl",
        "keyframe_uri": "gs://bucket/scenes/scene_kitchen_001/images/capture_2026_02_16_keyframe.jpg",
        "quality": {"pose_match_rate": 0.97, "p95_pose_delta_sec": 0.02},
        "swap_focus": ["kitchen"],
        "intended_space_type": "kitchen",
        "capture_bundle": {
            "arkit_poses_uri": "gs://bucket/.../arkit/poses.jsonl",
            "arkit_intrinsics_uri": "gs://bucket/.../arkit/intrinsics.json",
        },
        "articulation_hints": [{"label": "drawer", "joint_type": "prismatic"}],
    }


def test_capture_descriptor_parses_alias_fields() -> None:
    descriptor = CaptureDescriptor.from_dict(_sample_descriptor_payload())

    assert descriptor.environment_type_hint == "kitchen"
    assert descriptor.arkit_poses_uri == "gs://bucket/.../arkit/poses.jsonl"
    assert descriptor.arkit_intrinsics_uri == "gs://bucket/.../arkit/intrinsics.json"
    assert descriptor.requested_lanes == ["qualification"]


def test_capture_descriptor_infers_nurec_mode() -> None:
    payload = _sample_descriptor_payload()
    payload.pop("nurec_mode", None)
    descriptor = CaptureDescriptor.from_dict(payload)
    assert descriptor.nurec_mode == "mono_pose_assisted"


def test_capture_descriptor_normalizes_bedroom_aliases() -> None:
    payload = _sample_descriptor_payload()
    payload["environment_type_hint"] = "bed room"
    payload["swap_focus"] = ["bedroom", "auto", "warehouse"]

    descriptor = CaptureDescriptor.from_dict(payload)
    assert descriptor.environment_type_hint == "bedroom"
    assert descriptor.swap_focus == ["bedroom", "warehouse"]


def test_build_capture_bundle_constraints_embeds_metadata() -> None:
    descriptor = CaptureDescriptor.from_dict(_sample_descriptor_payload())
    bundle = build_capture_bundle_constraints(
        descriptor,
        descriptor_uri="gs://bucket/scenes/scene_kitchen_001/captures/capture_2026_02_16/capture_descriptor.json",
        qa_report_uri="gs://bucket/scenes/scene_kitchen_001/captures/capture_2026_02_16/qa_report.json",
        qa_report={"status": "passed"},
    )

    assert bundle["capture_id"] == "capture_2026_02_16"
    assert bundle["capture_bundle"]["arkit_poses_uri"].endswith("poses.jsonl")
    assert bundle["qa"]["status"] == "passed"


def test_build_scene_request_from_descriptor() -> None:
    descriptor = CaptureDescriptor.from_dict(_sample_descriptor_payload())
    payload = build_scene_request_from_descriptor(descriptor)

    assert payload["source_mode"] == "image"
    assert payload["constraints"]["capture_bundle"]["capture_id"] == descriptor.capture_id
    assert payload["image"]["gcs_uri"].endswith("_keyframe.jpg")


def test_scene_manifest_seed_marks_separate_assets() -> None:
    descriptor = CaptureDescriptor.from_dict(_sample_descriptor_payload())
    seed = build_scene_manifest_seed(
        descriptor,
        manipulation_candidates=[{"label": "dishwasher", "instance_id": "dw_1"}],
    )

    assert seed["scene_shell"]["proxy_collision_enabled"] is True
    assert all(item["must_be_separate_asset"] for item in seed["manipulation_candidates"])


def test_capture_descriptor_rejects_unsupported_schema() -> None:
    payload = _sample_descriptor_payload()
    payload["schema_version"] = "v2"
    with pytest.raises(ValueError):
        CaptureDescriptor.from_dict(payload)


def test_capture_descriptor_normalizes_requested_lanes() -> None:
    payload = _sample_descriptor_payload()
    payload["requested_lanes"] = ["qualification", "advanced_geometry", "all", "qualification"]

    descriptor = CaptureDescriptor.from_dict(payload)

    assert descriptor.requested_lanes == ["qualification", "advanced_geometry"]
