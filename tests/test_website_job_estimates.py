# Covers (for impacted-test selection):
#   src/blueprint_pipeline/clean_plate_removal_response_schema.py
#   src/blueprint_pipeline/clean_plate_removal_analysis_gemini.py
#   src/blueprint_pipeline/clean_plate_stage.py
#   src/blueprint_pipeline/website_job_estimates.py
#   src/blueprint_pipeline/website_task_preparation.py
#   src/blueprint_pipeline/website_robot_fit.py
#   src/blueprint_pipeline/website_packet_facts.py
"""Where a site never states a fact a deployment company needs, the packet gives Blueprint's best honest estimate.

Owner decision 2026-11-19. The video analysis Blueprint already buys (one Gemini call per job) also reads what each
object is mostly made of and how much of its box that fills, and estimates how often a site like the one filmed runs
the job. No new call, and an analysis bought before these were asked for is never bought again. The preparation
turns them into the object's weight (a published weight first, else its size × fill × material density) and keeps
the job's operations estimate; the robot fit and the company's job page read both, labelled as estimates with their
basis, and the site's own answers replace them wherever it gave them (the website holds the answers).
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


# --- the weight: published, designed, or from size and material -------------------------------------------------

from blueprint_pipeline import website_job_estimates as estimates  # noqa: E402

LAMP_BOX = [0.4, 0.37, 0.67]
METAL = {"material": "metal", "solid_fraction": [0.003, 0.015]}


def _spec(match="exact_model", value=4.2, basis="label_read"):
    spec = {"schema_version": "website_object_spec.v1", "target_id": "table_lamp", "identity": {"basis": basis},
            "specs": {"net_weight": {"value": value, "unit": "kg", "match": match,
                                     "source_urls": ["https://maker.example/lamp"]}}}
    spec["digest"] = canonical_digest(spec, digest_field="digest")
    return spec


def test_the_weight_is_the_box_times_the_fill_times_the_density_across_the_scale_band():
    value = estimates.weight_estimate(dimensions_m=LAMP_BOX, relative_error=0.25, entry=METAL)
    volume = 0.4 * 0.37 * 0.67
    assert value["status"] == "estimated" and value["basis"] == estimates.PHYSICS
    assert value["best_kg"] == pytest.approx(volume * (0.003 * 0.015) ** 0.5 * (2700 * 7900) ** 0.5, rel=2e-3)
    assert value["range_kg"] == [pytest.approx(volume * 0.75 ** 3 * 0.003 * 2700, rel=2e-3),
                                 pytest.approx(volume / 0.75 ** 3 * 0.015 * 7900, rel=2e-3)]
    assert value["best_kg"] == 3.07 and value["range_kg"] == [0.339, 27.9]  # a lamp, not 1 to 250 kg
    assert value["material"] == "metal" and value["density_kg_m3"] == [2700.0, 7900.0]
    assert value["claim"] == "estimate_not_weighed"


def test_every_material_the_analysis_may_name_has_a_density():
    assert set(estimates.MATERIAL_DENSITY_KG_M3) == set(MATERIALS) - {"unknown"}
    assert all(0 < low < high for low, high in estimates.MATERIAL_DENSITY_KG_M3.values())


@pytest.mark.parametrize("spec,basis", [
    (_spec(), estimates.PUBLISHED),
    (_spec(match="model_family"), estimates.PHYSICS),         # the exact model's weight only
    (_spec(basis="unknown"), estimates.PHYSICS),               # an unidentified object's specs are not its own
    ({**_spec(), "digest": "sha256:" + "0" * 64}, estimates.PHYSICS),  # an unsealed record publishes nothing
])
def test_a_published_weight_for_the_exact_model_comes_first(spec, basis):
    value = estimates.weight_estimate(dimensions_m=LAMP_BOX, relative_error=0.25, entry=METAL,
                                      target={"target_id": "table_lamp"}, object_spec=spec)
    assert value["basis"] == basis
    if basis == estimates.PUBLISHED:
        assert value["best_kg"] == 4.2 and value["range_kg"] == [3.78, 4.62]


@pytest.mark.parametrize("entry,box,reason", [
    ({}, LAMP_BOX, "no_material_reading"),
    ({"material": "metal"}, LAMP_BOX, "no_material_reading"),
    ({"material": "unknown", "solid_fraction": [0.1, 0.2]}, LAMP_BOX, "material_not_seen"),
    (METAL, [], "no_object_box"),
    (METAL, [0.4, 0.0, 0.67], "no_object_box"),
])
def test_without_a_published_weight_or_a_material_reading_it_says_why(entry, box, reason):
    assert estimates.weight_estimate(dimensions_m=box, relative_error=0.25, entry=entry) == {
        "status": "not_estimated", "reason": reason}


def test_a_designed_object_weighs_what_its_design_says():
    value = estimates.weight_estimate(dimensions_m=LAMP_BOX, relative_error=0.25, entry={}, designed_kg=0.43)
    assert value["basis"] == estimates.DESIGNED and value["range_kg"] == [0.43, 0.43]


def test_the_preparation_records_both_estimates(tmp_path):
    from tests.test_website_task_preparation import REMOVAL, _compile
    removal = {**REMOVAL, "entries": [{**REMOVAL["entries"][0], **METAL}], "operations_estimate": OPERATIONS}
    value = _compile(tmp_path, removal_manifest=removal)
    assert value["status"] == "intake_ready", value["blockers"]
    dims = [b - a for a, b in zip(value["subject"]["aabb_min_xyz"], value["subject"]["aabb_max_xyz"])]
    assert value["weight_estimate"] == estimates.weight_estimate(dimensions_m=dims, relative_error=0.25, entry=METAL)
    assert value["operations_estimate"]["per_hour"] == [1.0, 4.0]
    assert value["operations_estimate"]["source"] == "video_analysis_estimate"
    # The simulation's own prior is unchanged: the estimate informs a company, it does not rebuild the object.
    assert value["physics"]["bounds"]["mass_kg"] == _compile(tmp_path)["physics"]["bounds"]["mass_kg"]
    # An analysis bought before the material was asked for: no estimate, and the reason why.
    assert _compile(tmp_path)["weight_estimate"] == {"status": "not_estimated", "reason": "no_material_reading"}
    assert "operations_estimate" not in _compile(tmp_path)


def test_the_robot_fit_weighs_the_object_by_the_estimate_unless_the_site_stated_it():
    from blueprint_pipeline.website_robot_fit import _mass
    band = (0.75, 1 / 0.75)
    estimated = estimates.weight_estimate(dimensions_m=LAMP_BOX, relative_error=0.25, entry=METAL)
    prep = {"weight_estimate": estimated, "physics": {"basis": "estimated", "bounds": {"mass_kg": [1.0, 250.0]}}}
    mass, basis, record = _mass(prep, band)
    # The estimate already spans the scale band: it is not widened again.
    assert mass == [0.339, 27.9] and basis == "estimated"
    assert record["mass_estimate_basis"] == estimates.PHYSICS and record["mass_widened_by"] is None
    stated = {**prep, "authoring_inputs": {"configuration": {"construction_constraints": {
        "operator_task_details": {"item_weight": "2 kg"}}}}}
    assert _mass(stated, band)[1] == "owner_stated"
    older = {**prep, "weight_estimate": {"status": "not_estimated", "reason": "no_material_reading"}}
    assert _mass(older, band)[0] == [pytest.approx(0.421875), pytest.approx(250 / 0.421875)]


# --- the company's job page's words ---------------------------------------------------------------------------------------

PRIOR = {"physics": {"bounds": {"mass_kg": [0.999, 250.0]}}}


def test_the_weight_line_gives_the_estimate_with_its_range_its_basis_and_that_it_was_not_weighed():
    estimate = estimates.weight_estimate(dimensions_m=LAMP_BOX, relative_error=0.25, entry=METAL)
    assert estimates.weight_line({**PRIOR, "weight_estimate": estimate}) == (
        "Weight: about 3.1 kg (0.34 to 28 kg). Blueprint's estimate from its size in the video, across the scale "
        "band, and its likely material: mostly metal, filling 0.3 to 1.5 % of its box for a typical object of its "
        "kind, as Blueprint's video analysis read it. An estimate, not weighed. The simulation uses a wider prior "
        "from its size alone: 1 to 250 kg.")
    published = estimates.weight_estimate(dimensions_m=LAMP_BOX, relative_error=0.25, entry={},
                                          target={"target_id": "table_lamp"}, object_spec=_spec())
    assert estimates.weight_line({**PRIOR, "weight_estimate": published}) == (
        "Weight: about 4.2 kg (3.8 to 4.6 kg), the weight its maker publishes for the exact model, give or take "
        "10 %. An estimate, not weighed.")


def test_the_weight_line_says_in_plain_words_when_there_is_no_estimate():
    assert estimates.weight_line(PRIOR) == (
        "Weight: not estimated: this job was prepared before Blueprint estimated weights. The simulation uses a "
        "wide prior from its size alone: 1 to 250 kg.")
    missing = {**PRIOR, "weight_estimate": {"status": "not_estimated", "reason": "no_material_reading"}}
    assert estimates.weight_line(missing).startswith(
        "Weight: not estimated: no weight is published for its exact model, and Blueprint's video analysis did not "
        "read what it is made of")
    assert estimates.weight_line({}) is None  # nothing is built, nothing is weighed


RATES = {"operations_estimate": {**OPERATIONS, "source": "video_analysis_estimate"}}


def test_how_often_the_job_runs_is_blueprints_labelled_estimate_naming_the_question_each_line_answers():
    assert estimates.operations_lines(RATES) == [
        ("How many an hour: 1 to 4 (Blueprint's estimate; the site did not say).", "items_per_hour"),
        ("How many hours a day the job runs: 0.5 to 2 (Blueprint's estimate; the site did not say).", "hours_per_day"),
        ("How many days a week it runs: 5 to 7 (Blueprint's estimate; the site did not say).", None),
        ("What that estimate rests on: A hotel room's housekeeping moves a lamp now and then while turning the room. "
         "Blueprint's video analysis made it from the job's words and the place filmed; nobody at the site checked "
         "it.", None)]
    assert estimates.operations_lines({"subject": {}}) == []  # no estimate: the page says the site did not state it


def test_the_company_page_facts_carry_the_labelled_estimates_each_naming_the_question_it_answers():
    from blueprint_pipeline.website_packet_facts import measured_and_estimated
    estimate = estimates.weight_estimate(dimensions_m=LAMP_BOX, relative_error=0.25, entry=METAL)
    prep = {**PRIOR, "weight_estimate": estimate, "operations_estimate": RATES["operations_estimate"]}
    estimated = measured_and_estimated(prep, None, None)[1]
    weight = next(row for row in estimated if row["text"].startswith("Weight: "))
    assert weight["answers"] == "item_weight" and weight["text"].startswith("Weight: about 3.1 kg (0.34 to 28 kg).")
    assert {"text": "How many an hour: 1 to 4 (Blueprint's estimate; the site did not say).",
            "answers": "items_per_hour"} in estimated
    assert not any("Mass estimate:" in row["text"] for row in estimated)
    # No estimate (an analysis bought before it was asked for): no rate line, and the weight says why in plain words.
    estimated = measured_and_estimated(PRIOR, None, None)[1]
    assert not any(row["text"].startswith("How many") for row in estimated)
    assert any(row["text"].startswith("Weight: not estimated: this job was prepared before") for row in estimated)
