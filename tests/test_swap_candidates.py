"""Tests for swap candidate policy selection."""

from blueprint_pipeline.capture_bridge import CaptureDescriptor
from blueprint_pipeline.swap_candidates import build_swap_candidates_payload


def _descriptor() -> CaptureDescriptor:
    return CaptureDescriptor.from_dict(
        {
            "schema_version": "v1",
            "scene_id": "scene_1",
            "capture_id": "cap_1",
            "capture_source": "iphone",
            "capture_tier": "tier1_iphone",
            "raw_prefix_uri": "gs://bucket/scenes/scene_1/iphone/cap_1/raw",
            "frames_index_uri": "gs://bucket/scenes/scene_1/captures/cap_1/frames/index.jsonl",
            "nurec_mode": "mono_pose_assisted",
            "manipulation_candidates": [{"instance_id": "obj_box", "label": "box"}],
            "articulation_hints": [{"instance_id": "obj_drawer", "label": "drawer"}],
        }
    )


def test_candidate_generation_mixed_labels_and_roles() -> None:
    object_index = [
        {
            "id": "obj_drawer",
            "label": "kitchen drawer",
            "pointCloudFile": "obj_drawer.ply",
            "boundingBox": {
                "center": [1.0, 0.5, 2.0],
                "extents": [0.8, 0.4, 0.6],
                "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "orientationQuaternion": [1, 0, 0, 0],
            },
        },
        {
            "id": "obj_box",
            "label": "plastic tote box",
            "pointCloudFile": "obj_box.ply",
            "boundingBox": {
                "center": [0.0, 0.2, 1.0],
                "extents": [0.5, 0.3, 0.4],
                "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "orientationQuaternion": [1, 0, 0, 0],
            },
        },
        {"id": "obj_wall", "label": "wall segment"},
    ]

    payload = build_swap_candidates_payload(descriptor=_descriptor(), object_index_entries=object_index)
    candidates = payload["candidates"]

    assert len(candidates) == 2
    by_id = {item["object_id"]: item for item in candidates}

    assert by_id["obj_drawer"]["sim_role"] == "articulated_furniture"
    assert by_id["obj_drawer"]["articulation"]["required"] is True
    assert by_id["obj_box"]["sim_role"] == "manipulable_object"
    assert by_id["obj_box"]["articulation"]["required"] is False
    assert all(item["must_be_separate_asset"] is True for item in candidates)
