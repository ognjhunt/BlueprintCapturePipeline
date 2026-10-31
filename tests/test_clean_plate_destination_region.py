# Covers (for impacted-test selection):
#   src/blueprint_pipeline/clean_plate_removal_response_schema.py
#   src/blueprint_pipeline/clean_plate_removal_analysis_gemini.py
#   src/blueprint_pipeline/clean_plate_stage.py
"""A task that names a place on its destination surface (capture-447ae4f4, 2026-10-31).

"place it at the other end of the same table" grounded the whole surface, whose
middle lay under the start. The analysis may now return that place as an
optional destination region; a malformed one is refused and recorded, never
repaired with another paid turn and never a hold.
"""
from __future__ import annotations

import hashlib
import json
import sys
import types as module_types

import pytest

from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis
from blueprint_pipeline.clean_plate_removal_response_schema import (
    PRIOR_RESPONSE_SCHEMA,
    RESPONSE_SCHEMA,
    validate_destination_region,
)
from blueprint_pipeline.clean_plate_stage import _build_removal_manifest
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

TASK = ("Pick up the wooden tray from the dining table and place it at the other end "
        "of the same table.")


def _targets(**destination):
    tray = {"target_id": "tray", "semantic_label": "wooden tray", "segmentation_prompt": "wooden tray",
            "target_class": "movable_object", "target_role": "task_object", "task_effect": "manipulated",
            "disposition": "remove", "rebuild_intent": "rebuild_and_compose", "decision_reason": "moved",
            "task_basis_quote": "the wooden tray", "clarification_question": "", "articulated_part": "",
            "articulation_kind": "", "placement_relation": "", "confidence": 0.9,
            "spatial_evidence": [{"timestamp_seconds": 2.0, "box_xywh_normalized": [0.4, 0.5, 0.15, 0.1]}]}
    table = {**tray, "target_id": "table", "semantic_label": "dining table", "segmentation_prompt": "wooden table",
             "target_class": "fixed_clutter", "target_role": "destination", "task_effect": "static_contact",
             "disposition": "keep", "rebuild_intent": "none", "placement_relation": "on",
             "task_basis_quote": "at the other end of the same table", "decision_reason": "the place",
             "spatial_evidence": [{"timestamp_seconds": 2.0, "box_xywh_normalized": [0.1, 0.45, 0.8, 0.4]}],
             **destination}
    return [tray, table]


def _region(**change):
    return {"destination_target_id": "table", "region_basis_quote": "the other end",
            "spatial_evidence": [{"timestamp_seconds": 2.0, "box_xywh_normalized": [0.65, 0.5, 0.2, 0.2]}],
            **change}


def _response(region=None, **destination):
    return json.dumps({"targets": _targets(**destination),
                       **({"destination_region": region} if region is not None else {})})


def _parse(text):
    targets = analysis.parse_removal_plan_response(text, strict=True, task_description=TASK)
    return targets, analysis.parse_destination_region(text, targets=targets, task_description=TASK)


def test_a_named_place_is_parsed_and_carried_to_the_removal_manifest():
    targets, (region, refusal) = _parse(_response(_region(region_basis_quote=" the other end ")))
    assert refusal is None
    assert region == {"destination_target_id": "table", "region_basis_quote": "the other end",
                      "spatial_evidence": [{"timestamp_seconds": 2.0, "box_xywh_normalized": [0.65, 0.5, 0.2, 0.2]}]}
    plan = analysis.build_removal_plan(targets=targets, destination_region=region, status="completed",
                                       model="m", processing="static")
    assert analysis.validate_removal_plan(plan) == [] and plan["destination_region"] == region
    assert _build_removal_manifest(plan)["destination_region"] == region


def test_a_task_naming_no_place_gets_no_region_and_an_unchanged_plan():
    targets, parsed = _parse(_response())
    assert parsed == (None, None)
    plan = analysis.build_removal_plan(targets=targets, status="completed", model="m", processing="static")
    assert "destination_region" not in plan and "destination_region_refusal" not in plan
    assert "destination_region" not in _build_removal_manifest(plan)


@pytest.mark.parametrize("region, destination, code", [
    (_region(region_basis_quote="the far end"), {}, "destination_region_basis_missing"),
    (_region(region_basis_quote=""), {}, "destination_region_basis_missing"),
    (_region(destination_target_id="tray"), {}, "destination_region_target_invalid"),
    (_region(destination_target_id="wall"), {}, "destination_region_target_invalid"),
    (_region(), {"placement_relation": "inside"}, "destination_region_target_invalid"),
    (_region(spatial_evidence=[]), {}, "destination_region_evidence_invalid"),
    (_region(spatial_evidence=[{"timestamp_seconds": 2.0, "box_xywh_normalized": [0.9, 0.5, 0.2, 0.2]}]), {},
     "destination_region_evidence_invalid"),
    (_region(spatial_evidence=[{"timestamp_seconds": 2.0, "box_xywh_normalized": [0.65, 0.5, 0.0, 0.2]}]), {},
     "destination_region_evidence_invalid"),
    (_region(spatial_evidence=[{"timestamp_seconds": -1, "box_xywh_normalized": [0.65, 0.5, 0.2, 0.2]}]), {},
     "destination_region_evidence_invalid"),
    (_region(spatial_evidence=[{"timestamp_seconds": True, "box_xywh_normalized": [0.65, 0.5, 0.2, 0.2]}]), {},
     "destination_region_evidence_invalid"),
    (_region(spatial_evidence=[{"timestamp_seconds": 2.0, "box_xywh_normalized": [0.65, 0.5, 0.2]}]), {},
     "destination_region_evidence_invalid"),
    (_region(spatial_evidence=[{"timestamp_seconds": 3.0, "box_xywh_normalized": [0.65, 0.5, 0.2, 0.2]}]), {},
     "destination_region_outside_destination_box"),
    (_region(spatial_evidence=[{"timestamp_seconds": 2.0, "box_xywh_normalized": [0.0, 0.0, 0.08, 0.08]}]), {},
     "destination_region_outside_destination_box"),
    (_region(note="the far side"), {}, "destination_region_invalid"),
    ("the other end", {}, "destination_region_invalid"),
])
def test_a_malformed_region_is_refused_and_the_targets_stand(region, destination, code):
    targets, (parsed, refusal) = _parse(_response(region, **destination))
    assert parsed is None and refusal == code
    assert [row["target_id"] for row in targets] == ["tray", "table"]


def test_a_retained_plan_region_is_checked_against_its_own_targets():
    targets, (region, _) = _parse(_response(_region()))
    plan = analysis.build_removal_plan(targets=targets, destination_region=region, status="completed",
                                       model="m", processing="static")
    assert analysis.validate_removal_plan({**plan, "destination_region": {**region, "destination_target_id": "x"}}) \
        == ["destination_region_target_invalid"]
    # Without the task text only the verbatim quote check is skipped.
    assert validate_destination_region(region, targets=targets, task_description=None) == []


def _analysis(tmp_path, monkeypatch, text):
    google = sys.modules.get("google") or module_types.ModuleType("google")
    genai = module_types.ModuleType("google.genai")
    genai.types = module_types.ModuleType("google.genai.types")
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setattr(google, "genai", genai, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai.types)
    video = tmp_path / "walkthrough.mp4"
    video.write_bytes(b"video")
    monkeypatch.setenv(analysis.GATE_ENV, "1")
    monkeypatch.setattr(analysis, "_api_key", lambda: ("fixture", "fixture"))
    monkeypatch.setattr(analysis, "_video_processing", lambda path, mode: ("static", 10.0))
    monkeypatch.setattr(analysis, "_invoke_agentic_video",
                        lambda **_kwargs: {"text": text, "video_processing": {"mode": "static"}})
    repairs = []
    monkeypatch.setattr(analysis, "_invoke_text_repair", lambda **kwargs: repairs.append(kwargs))
    context = {"description": TASK, "confirmed": True}
    return analysis._analyze_removal_targets(video_path=video, task_context=context), repairs


def test_the_one_paid_answer_carries_its_region_without_another_turn(tmp_path, monkeypatch):
    plan, repairs = _analysis(tmp_path, monkeypatch, _response(_region()))
    assert plan["status"] == "completed" and repairs == []
    assert plan["destination_region"]["region_basis_quote"] == "the other end"


def test_a_malformed_region_is_recorded_without_a_repair_turn_or_a_hold(tmp_path, monkeypatch):
    plan, repairs = _analysis(tmp_path, monkeypatch, _response(_region(region_basis_quote="the far end")))
    assert plan["status"] == "completed" and plan["blockers"] == [] and repairs == []
    assert "destination_region" not in plan
    assert plan["destination_region_refusal"] == "destination_region_basis_missing"


def test_the_region_is_optional_and_the_prior_contract_is_kept_exactly():
    schema = RESPONSE_SCHEMA["properties"]["destination_region"]
    assert RESPONSE_SCHEMA["required"] == ["targets"]
    assert set(schema["required"]) == set(schema["properties"]) == {
        "destination_target_id", "region_basis_quote", "spatial_evidence"}
    assert schema["properties"]["spatial_evidence"]["minItems"] == 1
    # The schema and instruction every retained analysis was bought under (origin/main).
    assert canonical_digest(PRIOR_RESPONSE_SCHEMA) == (
        "sha256:27a13097d4b46ad8c387436bde563587208c5d666f8e6bfce4ba605a57813db7")
    assert hashlib.sha256(analysis.PRIOR_PROMPT_INSTRUCTION.encode()).hexdigest() == (
        "ad873396b1e3877b029e000403e53aaeb8065d7dfec00cadf56613a851893f75")
    assert analysis.DESTINATION_REGION_INSTRUCTION in analysis.PROMPT_INSTRUCTION
    assert analysis.DESTINATION_REGION_INSTRUCTION in analysis._video_prompt("static", {"description": TASK})
