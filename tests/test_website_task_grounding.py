"""ADP-009B/day 14: exact-image evidence resolves coarse video localization."""
from __future__ import annotations

import pytest
from blueprint_pipeline.website_task_grounding import validate_grounding


def candidate(**overrides):
    return {"visible": True, "confidence": 0.92, "box_xywh_normalized": [0.2, 0.3, 0.2, 0.1],
            "segmentation_prompt": "white book", "reason": "The object directly beneath the blue case.", **overrides}


def test_grounding_refines_visual_evidence_without_rewriting_the_task():
    target = {"target_id": "support", "disposition": "keep", "task_effect": "static_contact",
              "semantic_label": "white container under blue case", "spatial_evidence": [{"timestamp_seconds": 8}]}
    result = validate_grounding(candidate(), target=target, timestamp=8.133, frame_id="decoded-244", image_digest="sha256:image")
    assert result["target_id"] == "support" and result["disposition"] == "keep"
    assert result["semantic_label"] == target["semantic_label"]
    assert result["spatial_evidence"] == [{"timestamp_seconds": 8.133, "box_xywh_normalized": [0.2, 0.3, 0.2, 0.1]}]
    assert result["segmentation_prompt"] == "white book"
    assert result["grounding"]["source_frame_id"] == "decoded-244"
    assert result["grounding"]["image_digest"] == "sha256:image"


@pytest.mark.parametrize("changes", [{"visible": False}, {"confidence": 0.4}, {"confidence": True},
    {"box_xywh_normalized": [0.9, 0.2, 0.3, 0.2]}, {"box_xywh_normalized": [0, 0, float("nan"), 1]},
    {"segmentation_prompt": ""}, {"reason": ""}])
def test_uncertain_or_invalid_grounding_cannot_authorize_a_mask(changes):
    with pytest.raises(ValueError, match="grounding"):
        validate_grounding(candidate(**changes), target={}, timestamp=8, frame_id="frame", image_digest="image")


def test_concept_recovery_binds_unedited_crop_and_keeps_full_frame_coordinates(tmp_path, monkeypatch):
    from PIL import Image
    from blueprint_pipeline import website_task_grounding as module
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file

    video = tmp_path / "video.mp4"
    video.write_bytes(b"retained video")
    def extract(args, **kwargs):
        Image.new("RGB", (100, 200), "white").save(args[-1])
    monkeypatch.setattr(module.subprocess, "run", extract)
    calls = []
    def retain(**kwargs):
        calls.append(kwargs)
        return {"observation": candidate()}
    monkeypatch.setattr(module, "retained_gemini_call", retain)
    target = {"target_id": "support", "disposition": "keep", "spatial_evidence": [
        {"timestamp_seconds": 8, "box_xywh_normalized": [0.2, 0.3, 0.2, 0.1]}]}
    target["grounding"] = {"source_image_path": "/private/local/path.png", "image_digest": "sha256:source"}
    result = module.ground_task_target(target=target, tracks=[],
        registry=[{"source_frame_id": "frame", "model_frame_index": 240, "decoded_pts_seconds": 8}],
        video={"path": str(video), "sha256": _sha256_file(video)},
        task_context={"description": "Keep the support under the blue object"}, output_root=tmp_path / "ground",
        failed_segmentation_prompt="white container")
    binding = calls[0]["binding"]
    recovery = binding["concept_recovery"]
    assert recovery["failed_prompt"] == "white container"
    crop = tmp_path / "ground" / "support-240-concept.png"
    assert recovery["crop_digest"] == _sha256_file(crop)
    assert recovery["crop_box_pixels"] == (16, 55, 44, 84)
    assert "FIRST, full image" in binding["prompt"]
    assert "/private/local/path.png" not in binding["prompt"]
    assert result["target_id"] == "support" and result["disposition"] == "keep"
    assert result["spatial_evidence"][0]["box_xywh_normalized"] == [0.2, 0.3, 0.2, 0.1]
