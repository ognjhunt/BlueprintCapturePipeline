# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_profile_contracts.py
#   src/blueprint_pipeline/clean_plate_removal_response_schema.py
#   src/blueprint_pipeline/clean_plate_removal_analysis_gemini.py
#   src/blueprint_pipeline/clean_plate_stage.py
#   src/blueprint_pipeline/website_task_preparation.py
#   src/blueprint_pipeline/website_scene_handoff.py
#   src/blueprint_pipeline/website_task_context_contracts.py
"""A soft task object is decided by the object analysis's own judgment, not a word list.

The capture's clean-plate analysis (one Gemini call it already buys) now also classifies
each object rigid, deformable or unknown, with its stated evidence. Deformable holds;
rigid goes on; unknown goes on and is flagged. The label word list decides only for an
analysis bought before the classification existed, and every reading records its source.
No new paid call, and an analysis bought under the earlier request is never bought again.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis
from blueprint_pipeline import website_scene_handoff as handoff
from blueprint_pipeline import website_task_preparation as preparation
from blueprint_pipeline.clean_plate_removal_response_schema import (
    PRIOR_RESPONSE_SCHEMA, REGION_RESPONSE_SCHEMA, RESPONSE_SCHEMA,
)
from blueprint_pipeline.clean_plate_stage import _build_removal_manifest
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_profile_contracts import task_object_rigidity
from blueprint_pipeline.website_task_context_contracts import holds_native_task_preparation
from tests.test_website_task_shape_holds import DEFORMABLE, Reached, _context, _removal
from tests.test_website_scene_handoff import ATLAS, atlas_ready, inputs
from tests.test_website_task_preparation import _arguments

TASK = "Move the plush toy from the bed to the shelf."


@pytest.mark.parametrize("entry,rigidity,source,noun", [
    # The analysis decides, whatever the label says.
    ({"semantic_label": "plush toy", "rigidity": "deformable", "rigidity_reason": "soft fabric squashes"},
     "deformable", "object_analysis", None),
    ({"semantic_label": "hard-shell camera bag", "rigidity": "rigid", "rigidity_reason": "moulded case"},
     "rigid", "object_analysis", "bag"),
    ({"semantic_label": "folded bath towel", "rigidity": "unknown", "rigidity_reason": ""},
     "unknown", "object_analysis", "towel"),
    # An analysis bought before the classification: the label word list is the backstop.
    ({"semantic_label": "folded bath towel"}, "deformable", "label_word_backstop", "towel"),
    ({"semantic_label": "white ceramic mug"}, "unknown", "label_word_backstop", None),
    ({"semantic_label": "white ceramic mug", "rigidity": "squishy"}, "unknown", "label_word_backstop", None),
])
def test_the_reading_records_its_class_source_and_evidence(entry, rigidity, source, noun):
    value = task_object_rigidity({"target_id": "obj", **entry})
    assert (value["rigidity"], value["source"], value["label_word_screen"]) == (rigidity, source, noun)
    assert value["field"] == ("rigidity" if source == "object_analysis" else "semantic_label")
    assert value["reason"] == (entry.get("rigidity_reason") or None if source == "object_analysis" else None)
    assert ("flag" in value) is (rigidity == "unknown")


def _classified(label, rigidity, reason="the visible evidence"):
    return _removal({"semantic_label": label, "rigidity": rigidity, "rigidity_reason": reason})


def test_a_soft_object_the_word_list_misses_is_held_by_the_analysis(tmp_path, monkeypatch):
    arguments = _arguments(tmp_path, removal_manifest=_classified("plush toy", "deformable", "soft fabric squashes"))
    _context(arguments, "rigid_pick_and_place")
    monkeypatch.setattr(preparation, "register_source_to_runtime", lambda **_: pytest.fail("not authored rigid"))
    result = preparation.compile_website_scene_preparation(**arguments)
    assert result["blockers"] == [DEFORMABLE] and result["intake_request"] is None
    gap = result["native_task_preparation_gap"]
    assert gap["subject_basis"] == "manipulated_target_rigidity_classification" and gap["deformable_noun"] is None
    assert gap["task_object_rigidity"]["source"] == "object_analysis"
    assert gap["task_object_rigidity"]["reason"] == "soft fabric squashes"
    assert "'plush toy'" in gap["reason"] and "object analysis classified it as deformable" in gap["reason"]


@pytest.mark.parametrize("label,rigidity", [("hard-shell camera bag", "rigid"), ("folded bath towel", "unknown"),
                                            ("white ceramic mug", "rigid")])
def test_rigid_or_unknown_goes_on_even_where_the_word_list_reads_soft(tmp_path, monkeypatch, label, rigidity):
    arguments = _arguments(tmp_path, removal_manifest=_classified(label, rigidity))
    _context(arguments, "rigid_pick_and_place")

    def registration(**_):
        raise Reached()

    monkeypatch.setattr(preparation, "register_source_to_runtime", registration)
    with pytest.raises(Reached):
        preparation.compile_website_scene_preparation(**arguments)


@pytest.fixture
def ready(tmp_path, monkeypatch):
    value = atlas_ready(tmp_path)
    monkeypatch.setattr(ATLAS, lambda **_: value)
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry",
                        lambda **kw: {**kw["task_masks"], "source_geometry_digest": "atlas-geometry"})
    return value


def _handoff(tmp_path, monkeypatch, entry):
    kwargs = inputs(tmp_path)
    context = kwargs["descriptor"]["metadata"]["site_task_context"]
    context.update(schema_version="website_site_task_context.v1", task_family="rigid_pick_and_place",
                   description="Put it on the shelf.")
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    kwargs["clean_plate"]["task_masks"] = {"targets": [{"target_id": entry["target_id"]}]}
    write_json(tmp_path / "pipeline/removal.json", {"entries": [{"task_effect": "manipulated", **entry}]})
    return kwargs, context


def test_the_handoff_holds_a_classified_soft_object_and_re_derives_it(tmp_path, monkeypatch, ready):
    kwargs, context = _handoff(tmp_path, monkeypatch, {"target_id": "toy", "semantic_label": "plush toy",
                                                       "rigidity": "deformable", "rigidity_reason": "squashes"})
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                        lambda **_: pytest.fail("a held object buys no coverage"))
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert result["blockers"] == [DEFORMABLE] and holds_native_task_preparation(result, context)

    def tampered(change):
        value = deepcopy(result)
        change(value["native_task_preparation_gap"])
        value["digest"] = canonical_digest(value, digest_field="digest")
        return holds_native_task_preparation(value, context)

    assert not tampered(lambda gap: gap["task_object_rigidity"].update(rigidity="rigid"))
    assert not tampered(lambda gap: gap["task_object_rigidity"].update(source="label_word_backstop"))
    assert not tampered(lambda gap: gap.update(subject_basis="manipulated_target_label_text_screen"))
    assert not tampered(lambda gap: gap.update(deformable_noun="towel"))
    # A hold written before the classification (no reading) still reads by its label noun.
    legacy = deepcopy(result)
    gap = legacy["native_task_preparation_gap"]
    gap.pop("task_object_rigidity")
    gap.update(subject_label="folded bath towel", deformable_noun="towel",
               subject_basis="manipulated_target_label_text_screen")
    legacy["digest"] = canonical_digest(legacy, digest_field="digest")
    assert holds_native_task_preparation(legacy, context)
    gap["subject_label"] = "plush toy"
    legacy["digest"] = canonical_digest(legacy, digest_field="digest")
    assert not holds_native_task_preparation(legacy, context)


@pytest.mark.parametrize("entry,flagged", [
    ({"semantic_label": "mug", "rigidity": "unknown", "rigidity_reason": "only its handle is seen"}, True),
    ({"semantic_label": "mug", "rigidity": "rigid", "rigidity_reason": "glazed ceramic"}, False),
    ({"semantic_label": "mug"}, True),  # unclassified: the backstop finds no soft noun, so unknown
])
def test_the_handoff_records_the_reading_and_flags_unknown_without_holding(tmp_path, monkeypatch, ready,
                                                                          entry, flagged):
    kwargs, _ = _handoff(tmp_path, monkeypatch, {"target_id": "mug", **entry})
    reached = []
    # The job goes on past the subject holds to coverage, spec research and compile; compile
    # is stubbed to stop there with a held preparation, so the handoff record is returned.
    for name in ("website_assembly_coverage.attach_assembly_coverage", "website_object_spec_research.attach_object_specs"):
        monkeypatch.setattr("blueprint_pipeline." + name, lambda **kw: reached.append(1) or kw["task_masks"])
    monkeypatch.setattr("blueprint_pipeline.website_preparation_revision.bind_preparation", lambda value, **_: value)
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", lambda **_: {
        "status": "needs_input", "blockers": ["fixture_stop"], "digest": "sha256:" + "0" * 64, "thumbnail": None,
        "native_task_preparation_gap": {"code": "fixture_stop"}})
    result = handoff.prepare_website_scene_handoff(**kwargs)
    assert reached == [1, 1] and result["blockers"] == ["fixture_stop"]
    reading = result["task_object_rigidity"]
    assert reading["target_id"] == "mug" and reading["rigidity"] == entry.get("rigidity", "unknown")
    assert reading["source"] == ("object_analysis" if "rigidity" in entry else "label_word_backstop")
    assert result.get("flags") == (["task_object_rigidity_unknown"] if flagged else None)
    assert json.loads((tmp_path / "pipeline/website_scene_preparation/handoff.json").read_text()) == result


def _response(**rigidity):
    target = {"target_id": "toy", "semantic_label": "plush toy", "segmentation_prompt": "plush toy",
              "target_class": "movable_object", "target_role": "task_object", "task_effect": "manipulated",
              "disposition": "remove", "rebuild_intent": "rebuild_and_compose", "decision_reason": "moved",
              "task_basis_quote": "the plush toy", "clarification_question": "", "articulated_part": "",
              "articulation_kind": "", "placement_relation": "", "confidence": 0.9,
              "spatial_evidence": [{"timestamp_seconds": 1.0, "box_xywh_normalized": [0.4, 0.5, 0.1, 0.1]}],
              **rigidity}
    return json.dumps({"targets": [target]})


@pytest.mark.parametrize("given,kept", [
    ({"rigidity": "deformable", "rigidity_reason": "soft fabric squashes"}, True),
    ({"rigidity": "Rigid", "rigidity_reason": "hard plastic"}, True),
    ({"rigidity": "squishy", "rigidity_reason": "?"}, False),
    ({}, False),
])
def test_the_analysis_answer_carries_its_classification_into_the_manifest(given, kept):
    targets = analysis.parse_removal_plan_response(_response(**given), strict=True, task_description=TASK)
    plan = analysis.build_removal_plan(targets=targets, status="completed", model="m", processing="static")
    entry = _build_removal_manifest(plan)["entries"][0]
    if kept:
        assert entry["rigidity"] == given["rigidity"].lower() and entry["rigidity_reason"] == given["rigidity_reason"]
    else:
        # No class is no class: never a fabricated "unknown", and older manifests keep their bytes.
        assert "rigidity" not in targets[0] and "rigidity" not in entry and "rigidity_reason" not in entry


def test_the_classification_rides_the_existing_call_and_earlier_requests_stay_exact():
    target = RESPONSE_SCHEMA["properties"]["targets"]["items"]
    assert target["properties"]["rigidity"]["enum"] == ["rigid", "deformable", "unknown"]
    assert target["required"][-2:] == ["rigidity", "rigidity_reason"]
    assert analysis.RIGIDITY_INSTRUCTION in analysis.PROMPT_INSTRUCTION
    # The schema and instruction every analysis since 2026-10-31 was bought under (origin/main).
    assert canonical_digest(REGION_RESPONSE_SCHEMA) == (
        "sha256:9bc486a6642244be7d1637bec92634e9dcc51d6070b8168b851ee73b37383c14")
    assert hashlib.sha256(analysis.REGION_PROMPT_INSTRUCTION.encode()).hexdigest() == (
        "6a94ad5e25cc5d9dd74c86e52866126590df40117105ee9e5f3f37d11f279167")
    assert canonical_digest(PRIOR_RESPONSE_SCHEMA) == (
        "sha256:27a13097d4b46ad8c387436bde563587208c5d666f8e6bfce4ba605a57813db7")
    assert hashlib.sha256(analysis.PRIOR_PROMPT_INSTRUCTION.encode()).hexdigest() == (
        "ad873396b1e3877b029e000403e53aaeb8065d7dfec00cadf56613a851893f75")


#: The receipt origin/main writes for test_website_gemini_receipts' static fixture, before rigidity.
RETAINED_BEFORE_RIGIDITY = "70b0d68dcf3886ecc0f767a0bbae1a2fb25f6b070751c6a9f17feb8433b06b1f"


@pytest.mark.parametrize("status", ["completed", "submitting"])
def test_an_analysis_bought_before_rigidity_is_never_bought_again(tmp_path, monkeypatch, status):
    from tests.test_website_gemini_receipts import context, setup_call
    _args, calls = setup_call(tmp_path / "receipts", monkeypatch)
    video = tmp_path / "video.mov"
    video.write_bytes(b"original video")
    monkeypatch.setattr(analysis, "_video_processing", lambda *a: ("static", 13.525))
    monkeypatch.setattr(analysis, "_analyze_removal_targets", lambda **_k: pytest.fail("must not call the provider"))
    result = {"status": "completed", "targets": ["box"]}
    root = tmp_path / "receipts"
    root.mkdir()
    (root / f"{RETAINED_BEFORE_RIGIDITY}.json").write_text(json.dumps(
        {"status": status, "request_digest": "sha256:" + RETAINED_BEFORE_RIGIDITY,
         **({"result": result, "result_digest": canonical_digest(result)} if status == "completed" else {})}))
    kwargs = dict(video_path=video, task_context=context(), output_root=root)
    if status == "completed":
        # Reused as it stands, unclassified: the word list backstops its task object.
        assert analysis.analyze_removal_targets(**kwargs) == result
    else:
        with pytest.raises(ValueError, match="requires_reconciliation"):
            analysis.analyze_removal_targets(**kwargs)
    assert calls == [] and [path.name for path in root.glob("*.json")] == [f"{RETAINED_BEFORE_RIGIDITY}.json"]
