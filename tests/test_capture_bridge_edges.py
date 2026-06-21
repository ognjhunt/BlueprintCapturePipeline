from __future__ import annotations

import json
from typing import Any

import pytest

import blueprint_pipeline.capture_bridge as cb


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "v1",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "capture_source": "iphone",
        "capture_tier": "tier1_iphone",
        "raw_prefix_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw",
        "frames_index_uri": "gs://bucket/scenes/scene-1/captures/capture-1/frames/index.jsonl",
    }
    payload.update(overrides)
    return payload


def test_capture_bridge_normalizer_edges() -> None:
    assert cb._normalize_environment_hint("bed room") == "bedroom"
    assert cb._normalize_environment_hint("warehouse_floor") == "warehouse"
    assert cb._normalize_environment_hint("custom_lab") == "custom_lab"
    assert cb._normalize_swap_focus("kitchen") == ["kitchen"]
    assert cb._normalize_swap_focus(("warehouse", "kitchen")) == ["warehouse", "kitchen"]
    assert cb._normalize_swap_focus(123) == []
    assert cb._normalize_requested_lanes("retrieval_index") == ["qualification", "retrieval_index"]
    assert cb._normalize_requested_lanes("current") == [
        "qualification",
        "evaluation_prep",
        "simulation_automation",
    ]
    assert cb._normalize_requested_lanes(["", "task_evaluation_run"]) == [
        "qualification",
        "evaluation_prep",
        "simulation_automation",
    ]
    assert cb._normalize_requested_lanes(123) == ["qualification"]
    assert cb._normalize_capture_tier("") == "tier2_glasses"
    assert cb._normalize_capture_tier("tier2_android_phone") == "tier2_android"
    assert cb._infer_capture_source("android_phone", "") == "android"
    assert cb._infer_capture_source("", "tier2_glasses") == "glasses"
    assert cb._infer_capture_source("", "tier2_android") == "android"
    assert cb._infer_capture_source("", "tier1_iphone") == "iphone"
    assert cb._normalize_string_list(" one ") == ["one"]
    assert cb._normalize_string_list(42) == ["42"]
    assert cb._normalize_uncertainty_priors({"": 1, "ok": "2.5", "bad": "nope"}) == {"ok": 2.5}
    assert cb._normalize_scaffolding_validation({"scale_anchor_count": "bad", "validated_metric_bundle": True}) == {
        "validated_metric_bundle": True
    }
    assert cb._normalize_orientation_size({"width": "bad", "h": 3}) == {}
    assert cb._normalize_rotation_degrees(None) is None
    assert cb._normalize_rotation_degrees("bad") is None
    assert cb._normalize_rotation_degrees(359) == 359
    assert cb._first_nonzero_int("bad", 0, "") is None


def test_capture_orientation_evidence_and_modality_edges() -> None:
    orientation = cb._normalize_capture_orientation(
        {
            "encoded_width": 640,
            "encoded_height": 480,
            "declaredCaptureWidth": 480,
            "declaredCaptureHeight": 640,
            "displayOrientation": "Landscape",
            "rotationDegrees": 90,
            "normalization_applied": True,
            "source": "metadata",
            "preserve_original_display_orientation": False,
            "probe_details": {"tool": "ffprobe"},
        }
    )

    assert orientation["display_orientation"] == "landscape"
    assert orientation["display_size"] == {"width": 480, "height": 640}
    assert orientation["encoded_size"] == {"width": 640, "height": 480}
    assert orientation["normalization_applied"] is True
    assert orientation["source"] == "metadata"
    assert orientation["preserve_original_display_orientation"] is False
    assert orientation["probe_details"] == {"tool": "ffprobe"}
    camel_orientation = cb._normalize_capture_orientation(
        {"normalizationApplied": True, "preserveOriginalDisplayOrientation": True}
    )
    assert camel_orientation["normalization_applied"] is True
    assert camel_orientation["preserve_original_display_orientation"] is True
    assert cb._resolve_evidence_tier("glasses_with_validated_scaffolding", "", {}) == "video_with_validated_scaffolding"
    assert cb._resolve_evidence_tier("pre_screen_video", "", {}) == "pre_screen_video"
    assert cb._resolve_evidence_tier(None, "glasses_plus_scaffolding", {}) == "video_with_validated_scaffolding"
    assert cb._resolve_capture_modality(
        raw_modality=None,
        capture_source="iphone",
        quality={},
        scaffolding_used=[],
        has_metric_arkit_bundle=True,
    ) == "iphone_arkit_lidar"
    assert cb._resolve_capture_modality(
        raw_modality=None,
        capture_source="glasses",
        quality={},
        scaffolding_used=["scale"],
        has_metric_arkit_bundle=False,
    ) == "glasses_plus_scaffolding"
    assert cb._resolve_capture_modality(
        raw_modality=None,
        capture_source="glasses",
        quality={},
        scaffolding_used=[],
        has_metric_arkit_bundle=False,
    ) == "glasses_video_only"
    assert cb._resolve_capture_modality(
        raw_modality=None,
        capture_source="android",
        quality={},
        scaffolding_used=["scale"],
        has_metric_arkit_bundle=False,
    ) == "android_plus_scaffolding"
    assert cb._resolve_capture_modality(
        raw_modality=None,
        capture_source="android",
        quality={},
        scaffolding_used=[],
        has_metric_arkit_bundle=False,
    ) == "android_video_only"
    assert cb._resolve_capture_modality(
        raw_modality=None,
        capture_source="unknown",
        quality={},
        scaffolding_used=[],
        has_metric_arkit_bundle=False,
    ) == "iphone_arkit_lidar"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"schema_version": "v2"}, "Unsupported capture descriptor schema_version"),
        (_payload(scene_id=""), "scene_id is required"),
        (_payload(capture_id=""), "capture_id is required"),
        (_payload(raw_prefix_uri=""), "raw_prefix_uri is required"),
        (_payload(frames_index_uri=""), "frames_index_uri is required"),
    ],
)
def test_capture_descriptor_validation_edges(payload: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cb.CaptureDescriptor.from_dict(payload)


def test_capture_descriptor_android_profile_and_metadata_edges() -> None:
    android = cb.CaptureDescriptor.from_dict(
        _payload(
            capture_source="glasses",
            capture_tier="tier2_glasses",
            capture_profile_id="android_xr_glasses",
            capture_capabilities={"camera_pose": True, "pose_rows": 10},
            quality={"geometry_ready": True, "provider_ready": True},
            metadata={"capture_rights": {"capture_contributor_payout_eligible": True}},
        )
    )
    assert android.capture_modality == "android_xr_video_only"
    assert android.capture_capabilities["camera_pose"] is False
    assert android.capture_capabilities["pose_rows"] == 0
    assert android.quality["provider_ready"] is False
    assert android.metadata["capture_rights"]["capture_contributor_payout_eligible"] is False

    profiled = cb.CaptureDescriptor.from_dict(
        _payload(
            capture_profile_id="iphone_arkit_lidar",
            capture_capabilities={"camera_pose": True},
        )
    )
    assert profiled.metadata["capture_profile_id"] == "iphone_arkit_lidar"
    assert profiled.metadata["capture_capabilities"] == {"camera_pose": True}


def test_capture_descriptor_serialization_and_scene_builders(tmp_path) -> None:
    descriptor = cb.CaptureDescriptor.from_json(
        json.dumps(
            _payload(
                keyframe_uri="gs://bucket/keyframe.jpg",
                world_model_video_uri="gs://bucket/world.mov",
                arkit_poses_uri="gs://bucket/poses.jsonl",
                arkit_intrinsics_uri="gs://bucket/intrinsics.json",
                arkit_depth_prefix_uri="gs://bucket/depth",
                arkit_confidence_prefix_uri="gs://bucket/confidence",
                capture_capabilities={"camera_pose": True},
                depth_conditioning={"status": "available"},
                capture_orientation={"display_orientation": "portrait"},
                manipulation_candidates=[{"object_id": "drawer"}],
                articulation_hints=[{"joint": "slide"}],
            )
        )
    )
    descriptor_path = tmp_path / "capture_descriptor.json"
    descriptor_path.write_text(json.dumps(_payload()), encoding="utf-8")
    assert cb.CaptureDescriptor.from_file(descriptor_path).scene_id == "scene-1"

    serialized = descriptor.to_dict()
    assert serialized["capture_capabilities"] == {"camera_pose": True}
    assert serialized["depth_conditioning"] == {"status": "available"}
    assert serialized["capture_orientation"] == {"display_orientation": "portrait"}
    assert descriptor.preferred_world_model_video_uri == "gs://bucket/world.mov"
    constraints = cb.build_capture_bundle_constraints(
        descriptor,
        descriptor_uri="gs://bucket/descriptor.json",
        qa_report={"status": "passed"},
    )
    assert constraints["capture_bundle"]["arkit_poses_uri"] == "gs://bucket/poses.jsonl"
    assert constraints["qa"] == {"status": "passed"}
    request = cb.build_scene_request_from_descriptor(descriptor)
    assert request["image"]["gcs_uri"] == "gs://bucket/keyframe.jpg"
    assert request["constraints"]["capture_bundle"]["capture_capabilities"] == {"camera_pose": True}

    no_keyframe = cb.CaptureDescriptor.from_dict(_payload())
    with pytest.raises(ValueError, match="keyframe URI is required"):
        cb.build_scene_request_from_descriptor(no_keyframe)

    seed = cb.build_scene_manifest_seed(
        descriptor,
        manipulation_candidates=[{"object_id": "door"}],
        articulation_hints=[{"joint": "hinge"}],
    )
    assert [item["object_id"] for item in seed["manipulation_candidates"]] == ["drawer", "door"]
    assert all(item["must_be_separate_asset"] for item in seed["manipulation_candidates"])
    assert seed["articulation_hints"] == [{"joint": "slide"}, {"joint": "hinge"}]
