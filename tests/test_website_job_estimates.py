# Covers (for impacted-test selection):
#   src/blueprint_pipeline/clean_plate_removal_response_schema.py
#   src/blueprint_pipeline/clean_plate_removal_analysis_gemini.py
#   src/blueprint_pipeline/clean_plate_stage.py
"""Where a site never states a fact a deployment company needs, the packet gives Blueprint's best honest estimate.

Owner decision 2026-11-19. The video analysis Blueprint already buys (one Gemini call per job) also reads what each
object is mostly made of and how much of its box that fills, and estimates how often a site like the one filmed runs
the job. No new call, and an analysis bought before these were asked for is never bought again.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis
from blueprint_pipeline.clean_plate_removal_response_schema import (
    MATERIALS, RESPONSE_SCHEMA, RIGIDITY_RESPONSE_SCHEMA,
)
from blueprint_pipeline.clean_plate_stage import _build_removal_manifest
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

TASK = "Move the lamp from the round bedside table to the glass shelf on the wall."
OPERATIONS = {"per_hour": [1, 4], "hours_per_day": [0.5, 2], "days_per_week": [5, 7],
              "basis": "A hotel room's housekeeping moves a lamp now and then while turning the room."}


def _response(*, operations=None, **reading):
    target = {"target_id": "table_lamp", "semantic_label": "table lamp", "segmentation_prompt": "table lamp",
              "target_class": "movable_object", "target_role": "task_object", "task_effect": "manipulated",
              "disposition": "remove", "rebuild_intent": "rebuild_and_compose", "decision_reason": "moved",
              "task_basis_quote": "Move the lamp", "clarification_question": "", "articulated_part": "",
              "articulation_kind": "", "placement_relation": "", "confidence": 0.9,
              "spatial_evidence": [{"timestamp_seconds": 1.0, "box_xywh_normalized": [0.4, 0.5, 0.1, 0.1]}],
              "rigidity": "rigid", "rigidity_reason": "metal stand", **reading}
    return json.dumps({"targets": [target], **({"operations_estimate": operations} if operations else {})})


def _manifest(text):
    targets = analysis.parse_removal_plan_response(text, strict=True, task_description=TASK)
    plan = analysis.build_removal_plan(targets=targets, status="completed", model="m", processing="static",
                                       operations_estimate=analysis.parse_operations_estimate(text))
    return targets, plan, _build_removal_manifest(plan)


def test_the_estimates_ride_the_existing_call_and_the_earlier_request_stays_exact():
    target = RESPONSE_SCHEMA["properties"]["targets"]["items"]
    assert target["properties"]["material"]["enum"] == list(MATERIALS)
    assert target["required"][-2:] == ["material", "solid_fraction"]
    assert set(RESPONSE_SCHEMA["properties"]["operations_estimate"]["required"]) == {
        "per_hour", "hours_per_day", "days_per_week", "basis"}
    assert "operations_estimate" not in RESPONSE_SCHEMA["required"]  # optional: it never fails an answer
    assert analysis.ESTIMATE_INSTRUCTION in analysis.PROMPT_INSTRUCTION
    assert analysis.PROMPT_REVISION == "removal_analysis.r4"
    # The schema and instruction every analysis from 2026-11-03 to 2026-11-19 was bought under (origin/main).
    assert canonical_digest(RIGIDITY_RESPONSE_SCHEMA) == (
        "sha256:209408cb8c89f8f45047a5ff421e75746d737cd596ebd61bc340ec7ebb43b7c3")
    assert hashlib.sha256(analysis.RIGIDITY_PROMPT_INSTRUCTION.encode()).hexdigest() == (
        "4f32ead6cc59465919c322830691a4c36a81266835768df4c9d55ffeebe32162")
    # The furniture analysis the owner approves keeps its own exact request.
    binding, _cost = analysis.movable_obstacle_request(source_digest="sha256:" + "a" * 64, duration_seconds=10.0,
                                                       processing="static", task_context={"description": TASK},
                                                       review_items=[])
    assert binding["response_schema"] == RIGIDITY_RESPONSE_SCHEMA


@pytest.mark.parametrize("reading,kept", [
    ({"material": "metal", "solid_fraction": [0.003, 0.015]}, True),
    ({"material": "unknown", "solid_fraction": [0.1, 0.5]}, True),
    ({"material": "unobtainium", "solid_fraction": [0.1, 0.5]}, False),
    ({"material": "metal", "solid_fraction": [0, 0.5]}, False),       # a share of nothing is no reading
    ({"material": "metal", "solid_fraction": [0.6, 0.2]}, False),
    ({"material": "metal", "solid_fraction": [0.2, 1.5]}, False),
    ({"material": "metal", "solid_fraction": [True, 0.5]}, False),
    ({}, False),
])
def test_the_material_and_fill_ride_into_the_manifest_only_as_given(reading, kept):
    targets, _plan, manifest = _manifest(_response(**reading))
    entry = manifest["entries"][0]
    if kept:
        assert entry["material"] == reading["material"] and entry["solid_fraction"] == reading["solid_fraction"]
    else:
        # Nothing is repaired or made up, and the answer itself still stands.
        assert "material" not in targets[0] and "material" not in entry and "solid_fraction" not in entry
    assert entry["rigidity"] == "rigid"


@pytest.mark.parametrize("operations,kept", [
    (OPERATIONS, True),
    ({**OPERATIONS, "per_hour": [0, 2]}, True),                       # none in some hours is a fair estimate
    ({**OPERATIONS, "hours_per_day": [20, 30]}, False),
    ({**OPERATIONS, "days_per_week": [8, 9]}, False),
    ({**OPERATIONS, "basis": " "}, False),
    ({key: value for key, value in OPERATIONS.items() if key != "per_hour"}, False),
    (None, False),
])
def test_how_often_the_job_runs_rides_into_the_manifest_as_an_estimate(operations, kept):
    _targets, plan, manifest = _manifest(_response(operations=operations))
    if kept:
        assert manifest["operations_estimate"] == plan["operations_estimate"] == {
            **{key: [float(v) for v in operations[key]] for key in ("per_hour", "hours_per_day", "days_per_week")},
            "basis": operations["basis"], "source": "video_analysis_estimate"}
    else:
        assert "operations_estimate" not in plan and "operations_estimate" not in manifest


def test_an_answer_bought_before_the_estimates_still_decides_the_job():
    assert analysis.analysis_revision({"prompt_revision": "removal_analysis.r3"}) == analysis.RIGIDITY_PROMPT_REVISION
    assert analysis.analysis_revision({"prompt_template_sha256": hashlib.sha256(
        analysis.RIGIDITY_PROMPT_INSTRUCTION.encode()).hexdigest()}) == analysis.RIGIDITY_PROMPT_REVISION
    for status, blockers in (("completed", []), ("blocked", ["task_target_requires_clarification:x"])):
        assert analysis.binds_current_job({"status": status, "blockers": blockers,
                                           "prompt_revision": analysis.RIGIDITY_PROMPT_REVISION})


#: The receipt origin/main writes for test_website_gemini_receipts' static fixture under the rigidity revision.
RETAINED_BEFORE_ESTIMATES = "b1c93c909e1d2bb2a677ca302a71279de464c40306dd0086640a6d1645ae4940"


@pytest.mark.pins("blueprint_pipeline.website_gemini_receipts")
def test_an_analysis_bought_before_the_estimates_is_never_bought_again(tmp_path, monkeypatch):
    from tests.test_website_gemini_receipts import context, setup_call
    _args, calls = setup_call(tmp_path / "receipts", monkeypatch)
    video = tmp_path / "video.mov"
    video.write_bytes(b"original video")
    monkeypatch.setattr(analysis, "_video_processing", lambda *a: ("static", 13.525))
    monkeypatch.setattr(analysis, "_analyze_removal_targets", lambda **_k: pytest.fail("must not call the provider"))
    result = {"status": "blocked", "targets": ["box"], "blockers": ["task_target_requires_clarification:x"]}
    root = tmp_path / "receipts"
    root.mkdir()
    (root / f"{RETAINED_BEFORE_ESTIMATES}.json").write_text(json.dumps(
        {"status": "completed", "request_digest": "sha256:" + RETAINED_BEFORE_ESTIMATES, "result": result,
         "result_digest": canonical_digest(result)}))
    assert analysis.analyze_removal_targets(video_path=video, task_context=context(), output_root=root) == result
    assert [path.name for path in root.glob("*.json")] == [f"{RETAINED_BEFORE_ESTIMATES}.json"]
    assert calls == []
