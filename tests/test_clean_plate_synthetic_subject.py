"""A task that adds a new object absent from the video (a synthetic ball)."""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline.clean_plate_removal_analysis_gemini import (
    build_removal_plan,
    parse_removal_plan_response,
    parse_synthetic_subject,
    validate_removal_plan,
)
from blueprint_pipeline.clean_plate_removal_response_schema import RESPONSE_SCHEMA
from blueprint_pipeline.clean_plate_stage import _build_removal_manifest

TASK = "Pick up a synthetic ball resting on the floor and place it on the tabletop above the cabinets."


def _target(target_id, role, effect, **extra):
    return {"target_id": target_id, "semantic_label": target_id, "segmentation_prompt": target_id,
            "target_class": "fixed_clutter", "target_role": role, "task_effect": effect,
            "disposition": "keep", "rebuild_intent": "none", "decision_reason": "task surface",
            "task_basis_quote": "", "clarification_question": "", "articulated_part": "",
            "articulation_kind": "", "placement_relation": "", "spatial_evidence": [], "confidence": 0.9, **extra}


def _response(**subject):
    targets = [_target("floor", "support", "static_contact"),
               _target("desk_top", "destination", "static_contact", placement_relation="on",
                       task_basis_quote="on the tabletop above the cabinets")]
    design = {"semantic_label": "ball", "shape": "sphere", "dimensions_m": [0.065, 0.065, 0.065],
              "mass_kg": 0.06, "task_basis_quote": "a synthetic ball", "start_support_target_id": "floor",
              "decision_reason": "The task adds a ball that the video does not show.", **subject}
    return json.dumps({"targets": targets, "synthetic_subject": design})


def _parse(text):
    targets = parse_removal_plan_response(text, strict=True, task_description=TASK)
    return targets, parse_synthetic_subject(text, targets=targets, task_description=TASK)


def test_a_task_adding_a_ball_keeps_every_captured_object_and_carries_the_design():
    targets, subject = _parse(_response())
    assert subject["shape"] == "sphere" and subject["dimensions_m"] == [0.065] * 3
    assert subject["design_basis"] == "synthetic_design_choice" and subject["physical_measurement"] is False
    plan = build_removal_plan(targets=targets, synthetic_subject=subject, status="completed", model="m",
                              processing="agentic")
    assert validate_removal_plan(plan) == [] and plan["movable_removal_count"] == 0
    manifest = _build_removal_manifest(plan)
    assert manifest["entries"] == [] and manifest["synthetic_subject"] == subject
    # A response without the key is a captured-subject task, unchanged.
    assert parse_synthetic_subject(json.dumps({"targets": []}), targets=[], task_description=TASK) is None
    assert "synthetic_subject" not in build_removal_plan(targets=[], status="completed", model="m", processing="p")
    schema = RESPONSE_SCHEMA["properties"]["synthetic_subject"]
    assert set(schema["required"]) == set(schema["properties"]) and "synthetic_subject" not in RESPONSE_SCHEMA["required"]


@pytest.mark.parametrize("change, error", [
    ({"task_basis_quote": "a rubber ball"}, "synthetic_subject_task_basis_missing"),
    ({"start_support_target_id": "desk_top"}, "synthetic_subject_start_support_invalid"),
    ({"start_support_target_id": "wall"}, "synthetic_subject_start_support_invalid"),
    ({"dimensions_m": [0.065, 0.065, 0.03]}, "synthetic_subject_shape_dimensions_invalid"),
    ({"dimensions_m": [2.0, 2.0, 2.0]}, "synthetic_subject_design_out_of_range"),
    ({"shape": "torus"}, "synthetic_subject_shape_invalid"),
    ({"decision_reason": ""}, "removal_analysis_task_reason_required"),
])
def test_a_synthetic_design_that_misreads_the_task_or_its_support_is_refused(change, error):
    with pytest.raises(ValueError, match=error):
        _parse(_response(**change))


def test_a_visible_stand_in_cannot_be_both_captured_subject_and_synthetic():
    data = json.loads(_response())
    data["targets"].append({**_target("backpack", "task_object", "manipulated"), "target_class": "movable_object",
                            "disposition": "remove", "rebuild_intent": "rebuild_and_compose",
                            "task_basis_quote": "a synthetic ball",
                            "spatial_evidence": [{"timestamp_seconds": 1.0, "box_xywh_normalized": [0, 0, .1, .1]}]})
    with pytest.raises(ValueError, match="conflicts_with_captured_subject"):
        _parse(json.dumps(data))
