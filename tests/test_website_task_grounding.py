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
