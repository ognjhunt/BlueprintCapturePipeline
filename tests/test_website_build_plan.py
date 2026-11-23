"""The build plan (owner direction 2026-11-23): a model says what a job's simulation needs, plain code checks each
entry, and with several objects the most important one the job can build is taken out of the scan alone."""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline import website_build_plan as build_plan
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_build_plan_notes import build_plan_notes, plan_notes, sealed_decision
from tests.test_website_atlas_pipeline import _bought_review, _job, _preparation

CONTEXT = {"capture_id": "c", "context_digest": "task",
           "description": "Move the blue tote of parts (about 4 kg) from the bench to the shelf.",
           "success_criteria": "The tote rests on the shelf."}
ENTRIES = [{"target_id": "tote", "semantic_label": "blue tote", "task_effect": "manipulated", "rigidity": "rigid",
            "rigidity_reason": "plastic"},
           {"target_id": "carton", "semantic_label": "carton", "task_effect": "manipulated", "rigidity": "rigid",
            "rigidity_reason": "cardboard"},
           {"target_id": "shelf", "semantic_label": "shelf", "task_effect": "static_contact"}]
EDITOR = _bought_review()["automation"]["objects"]  # table, centerpiece (removed); chair_left, floor (kept)


def _row(target, name, *, role="moved_object", rigidity="rigid", sourcing="parametric", ids=(), size=None, mass=None,
         variety=None, shape=None):
    return {"target_id": target, "name": name, "role": role, "variety": variety, "rigidity": rigidity,
            "size_m": size, "mass_kg": mass, "sourcing": sourcing,
            "parametric_shape": shape or ("carton" if sourcing == "parametric" else None),
            "reason": f"The job needs the {name}.", "editor_object_ids": list(ids)}


PLAN = {"objects": [
    _row("tote", "blue tote", ids=["centerpiece"], variety="one tote of parts", shape="tote",
         mass={"low": 3.5, "high": 4.5, "basis": "site_words", "quote": "about 4 kg"},
         size={"low": .4, "high": .6, "basis": "estimated", "from": "the tote's extent in the analysis"}),
    _row("carton", "carton", sourcing="generate", ids=["table"]),
    _row("shelf", "shelf", role="destination", sourcing="keep_as_filmed"),
    _row(build_plan.NOT_IN_VIDEO, "pallet", role="fixture", sourcing="library"),
]}


#: ``_job``'s Atlas context states no words, so the plan it is given quotes none (a quote would drop its entry).
FILMED_PLAN = {"objects": [{**PLAN["objects"][0], "mass_kg": None}, *PLAN["objects"][1:]]}


def _inputs(entries=ENTRIES, editor=EDITOR):
    return build_plan.plan_inputs(task_context=CONTEXT, targets=entries, editor_objects=editor)


def _answering(monkeypatch, answer):
    asked = []

    def ask(**kwargs):
        asked.append(kwargs)
        if isinstance(answer, BaseException):
            raise answer
        return {"answer": answer}
    monkeypatch.setattr(build_plan, "ask_gemini", ask)
    return asked


def test_a_valid_plan_is_kept_in_the_models_order_and_quotes_only_the_jobs_words():
    plan = build_plan.validate_plan(PLAN, inputs=_inputs(), task_context=CONTEXT)
    assert [row["target_id"] for row in plan["objects"]] == ["tote", "carton", "shelf", "not_in_video"]
    assert "dropped" not in plan
    assert plan["objects"][0]["mass_kg"] == {"low": 3.5, "high": 4.5, "basis": "site_words", "quote": "about 4 kg"}
    assert plan["objects"][0]["size_m"]["basis"] == "estimated" and plan["objects"][1]["size_m"] is None


@pytest.mark.parametrize("bad, reason", [
    ({**PLAN["objects"][1], "target_id": "ghost"}, "target_unknown_or_repeated"),
    ({**PLAN["objects"][1], "target_id": "tote"}, "target_unknown_or_repeated"),
    ({**PLAN["objects"][1], "role": "robot"}, "field_off_list"),
    ({**PLAN["objects"][1], "sourcing": "buy"}, "field_off_list"),
    ({**PLAN["objects"][1], "rigidity": "soft"}, "soft_or_loose_built_rigid"),
    ({**PLAN["objects"][1], "reason": ""}, "reason"),
    ({**PLAN["objects"][1], "extra": 1}, "shape"),
    ({**PLAN["objects"][1], "mass_kg": {"low": 2, "high": 3, "basis": "site_words", "quote": "weighs 2 to 3 kg"}},
     "range_quote_not_in_job"),
    ({**PLAN["objects"][1], "mass_kg": {"low": 2, "high": 3, "basis": "estimated"}}, "range_basis"),
    ({**PLAN["objects"][1], "size_m": {"low": 3, "high": 2, "basis": "estimated", "from": "x"}}, "range_out_of_bounds"),
    ({**PLAN["objects"][1], "size_m": {"low": 1, "high": 90, "basis": "estimated", "from": "x"}},
     "range_out_of_bounds"),
    ({**PLAN["objects"][1], "editor_object_ids": ["centerpiece"]}, "editor_objects"),  # already the tote's
    ({**PLAN["objects"][1], "editor_object_ids": ["sofa"]}, "editor_objects"),
    ({**PLAN["objects"][3], "editor_object_ids": ["floor"]}, "editor_objects"),  # not in the video, yet shown
    # A parametric shape names one of the generator's shapes, for a rigid object it sources parametric, and only then.
    ({**PLAN["objects"][1], "sourcing": "parametric"}, "parametric_shape"),
    ({**PLAN["objects"][1], "parametric_shape": "tote"}, "parametric_shape"),
    ({**PLAN["objects"][1], "sourcing": "parametric", "parametric_shape": "crate"}, "parametric_shape"),
    ({**PLAN["objects"][1], "sourcing": "parametric", "parametric_shape": "tote", "rigidity": "articulated"},
     "parametric_shape"),
])
def test_a_bad_entry_is_dropped_alone_and_nothing_is_repaired(bad, reason):
    answer = {"objects": [PLAN["objects"][0], bad, PLAN["objects"][2]]}
    plan = build_plan.validate_plan(answer, inputs=_inputs(), task_context=CONTEXT)
    assert [row["target_id"] for row in plan["objects"]] == ["tote", "shelf"]
    assert plan["dropped"] == [{"index": 1, "reason": reason}]


def test_only_an_answer_with_no_object_list_is_no_plan_and_a_long_one_is_cut_at_its_limit():
    for answer in (None, [], {"object": []}, {"objects": {}}):
        with pytest.raises(ValueError, match=build_plan.INVALID):
            build_plan.validate_plan(answer, inputs=_inputs(), task_context=CONTEXT)
    many = [_row(build_plan.NOT_IN_VIDEO, f"thing {n}", role="other", sourcing="library") for n in range(10)]
    plan = build_plan.validate_plan({"objects": many}, inputs=_inputs(), task_context=CONTEXT)
    assert len(plan["objects"]) == build_plan.MAX_OBJECTS
    assert plan["dropped"] == [{"index": n, "reason": "over_max_objects"} for n in (8, 9)]


def test_the_model_is_shown_labels_and_ids_only_and_never_a_person():
    inputs = _inputs(ENTRIES + [{"target_id": "p", "target_class": "person", "semantic_label": "person"}])
    assert [row["target_id"] for row in inputs["analysis_targets"]] == ["tote", "carton", "shelf"]
    assert {row["object_id"]: row["editor_removes"] for row in inputs["editor_objects"]} == {
        "table": True, "centerpiece": True, "chair_left": False, "floor": False}
    assert inputs["job"] == {"description": CONTEXT["description"], "success_criteria": CONTEXT["success_criteria"]}


def test_the_built_object_is_the_first_the_job_moves_and_can_build():
    plan = build_plan.validate_plan(PLAN, inputs=_inputs(), task_context=CONTEXT)
    assert build_plan.built_object(plan, entries=ENTRIES, editor_objects=EDITOR) == {
        "target_id": "tote", "name": "blue tote", "sourcing": "parametric", "parametric_shape": "tote",
        "editor_object_ids": ["centerpiece"]}
    # Kept as filmed, soft in the analysis's reading, shown by no row the editor removes, or not the job's own object:
    # the next one in order.
    for first in ({**PLAN["objects"][0], "sourcing": "keep_as_filmed"}, {**PLAN["objects"][0], "role": "destination"},
                  {**PLAN["objects"][0], "editor_object_ids": ["floor"]}):
        reordered = {"objects": [first, *PLAN["objects"][1:]]}
        assert build_plan.built_object(reordered, entries=ENTRIES, editor_objects=EDITOR)["target_id"] == "carton"
    soft = [{**ENTRIES[0], "rigidity": "deformable", "rigidity_reason": "a bag"}, *ENTRIES[1:]]
    assert build_plan.built_object(plan, entries=soft, editor_objects=EDITOR)["target_id"] == "carton"
    assert build_plan.built_object({"objects": PLAN["objects"][2:]}, entries=ENTRIES, editor_objects=EDITOR) is None
    assert build_plan.built_object(None, entries=ENTRIES, editor_objects=EDITOR) is None


def test_the_partial_review_erases_only_the_built_objects_rows_with_fixed_words():
    review = _bought_review()
    partial = build_plan.partial_review(review, ["centerpiece"])
    assert {row["target_id"]: row["disposition"] for row in partial["automation"]["objects"]} == {
        "table": "keep", "centerpiece": "remove", "chair_left": "keep", "floor": "keep"}
    assert partial["automation"]["removal_instruction"] == build_plan.PARTIAL_REMOVAL_INSTRUCTION
    assert "the planner's words" not in json.dumps(partial)
    assert review["automation"]["objects"][0]["disposition"] == "remove"  # the bought review is unchanged


def test_several_job_objects_take_the_plans_one_out_and_keep_the_rest_as_filmed(tmp_path, monkeypatch):
    review = _bought_review()
    root, run, automated, posed = _job(tmp_path, monkeypatch, review, clean_plate=("objects_removed", ENTRIES))
    _preparation(tmp_path)
    asked = _answering(monkeypatch, FILMED_PLAN)
    run()
    record = json.loads((root / "job_object_rule.json").read_text())
    assert record["schema_version"] == build_plan.DECISION_SCHEMA and record["applied"] is False
    assert "job_objects_kept" not in record and record["digest"] == canonical_digest(record, digest_field="digest")
    assert record["built"] == {"target_id": "tote", "name": "blue tote", "sourcing": "parametric",
                               "parametric_shape": "tote", "editor_object_ids": ["centerpiece"]}
    assert record["kept_in_scan_target_ids"] == ["carton"]
    assert record["build_plan"]["plan"]["objects"][0]["target_id"] == "tote" and "fallback" not in record["build_plan"]
    # The editor erases the tote's rows only, with the fixed words; the bought review is unchanged.
    assert len(automated) == 1 and not posed
    consumed = automated[0]["review"]
    assert [row["disposition"] for row in consumed["automation"]["objects"]] == ["keep", "remove", "keep", "keep"]
    assert consumed["automation"]["removal_instruction"] == build_plan.PARTIAL_REMOVAL_INSTRUCTION
    assert json.loads((root / "atlas_scene_review.json").read_text()) == review
    # Sealed once: a rerun asks nothing and takes the same route.
    run()
    assert len(asked) == 1 and automated[1]["review"] == consumed
    # The handoff builds the tote alone: the carton stays in the room as filmed.
    removal = build_plan.apply_to_removal({"entries": ENTRIES}, tmp_path)
    assert [(row["target_id"], row["task_effect"]) for row in removal["entries"]] == [
        ("tote", "manipulated"), ("carton", "kept_in_scan"), ("shelf", "static_contact")]
    # The tote is sourced as a parametric shape: compile is told, and authors it as one at no cost.
    assert removal["parametric_subject"] == {"target_id": "tote", "shape": "tote"}
    assert sealed_decision(tmp_path) == record
    assert plan_notes(record) == [
        "What this simulation needs, as Blueprint's model read the job: blue tote (one tote of parts), carton, shelf "
        "and pallet.",
        "The plan takes this out of the scan to build as its own object: blue tote (one tote of parts).",
        "The plan keeps these in the scan as filmed, not built on their own: carton.",
        "Not in the video, so not in this simulation: pallet."]
    assert build_plan_notes(tmp_path) == {"plan_notes": plan_notes(record)}


# Review of #3711: a task family (whose runtime may be missing, shipping the room) and a navigation job never author
# one object, so the plan takes nothing out of the scan for them: every object stays in the room as filmed, as before.
@pytest.mark.parametrize("task_family, builds", [(None, True), ("pick_place", True), ("stacking", False),
                                                 ("goal_region_navigation", False)])
def test_only_the_one_object_route_takes_the_plans_object_out(tmp_path, task_family, builds):
    from blueprint_pipeline import website_atlas_pipeline as atlas
    context = {"capture_id": "c", "context_digest": "task", **({"task_family": task_family} if task_family else {})}
    assert atlas._authors_one_object(tmp_path, context, None) is builds


def test_a_family_job_keeps_every_object_as_filmed_and_still_records_its_plan(tmp_path, monkeypatch):
    from blueprint_pipeline import website_atlas_pipeline as atlas
    monkeypatch.setattr(atlas, "_authors_one_object", lambda *_args: False)
    review = _bought_review()
    root, run, automated, posed = _job(tmp_path, monkeypatch, review, clean_plate=("objects_removed", ENTRIES))
    _preparation(tmp_path)
    _answering(monkeypatch, FILMED_PLAN)
    run()
    record = json.loads((root / "job_object_rule.json").read_text())
    assert record["applied"] is True and record["job_objects_kept"] == "several_job_objects" and "built" not in record
    assert record["build_plan"]["plan"]["objects"][0]["target_id"] == "tote"
    assert not automated and len(posed) == 1


@pytest.mark.parametrize("answer", [{"nope": []}, RuntimeError("no answer"),
                                    {"objects": FILMED_PLAN["objects"][2:]}])
def test_no_plan_or_no_buildable_object_keeps_todays_route_for_several_objects(tmp_path, monkeypatch, answer):
    review = _bought_review()
    root, run, automated, posed = _job(tmp_path, monkeypatch, review, clean_plate=("objects_removed", ENTRIES))
    _preparation(tmp_path)
    _answering(monkeypatch, answer)
    run()
    record = json.loads((root / "job_object_rule.json").read_text())
    assert record["applied"] is True and record["job_objects_kept"] == "several_job_objects" and "built" not in record
    assert (record["build_plan"]["plan"] is None) is (not isinstance(answer, dict) or "objects" not in answer)
    if isinstance(answer, RuntimeError):
        assert record["build_plan"]["fallback"].endswith(":RuntimeError")
    elif record["build_plan"]["plan"] is None:
        assert record["build_plan"]["fallback"] == f"{build_plan.INVALID}:shape"
    assert not automated and len(posed) == 1
    assert all(row["disposition"] == "keep" for row in posed[0]["review"]["automation"]["objects"])
    assert build_plan.apply_to_removal({"entries": ENTRIES}, tmp_path) == {"entries": ENTRIES}
    notes = plan_notes(sealed_decision(tmp_path))
    assert not any("takes this out" in note for note in notes)
    if notes:  # the planner's removals were not applied: every object the plan names stays as filmed
        assert notes[1:] == ["Not in the video, so not in this simulation: pallet."]


def test_one_rigid_object_keeps_todays_removal_and_records_its_plan(tmp_path, monkeypatch):
    review = _bought_review()
    entries = [ENTRIES[0], ENTRIES[2]]
    root, run, automated, posed = _job(tmp_path, monkeypatch, review, clean_plate=("objects_removed", entries))
    _preparation(tmp_path)
    _answering(monkeypatch, {"objects": [FILMED_PLAN["objects"][0], PLAN["objects"][2]]})
    run()
    record = json.loads((root / "job_object_rule.json").read_text())
    assert record["applied"] is False and "built" not in record and record["build_plan"]["plan"]["objects"]
    assert len(automated) == 1 and automated[0]["review"] == review and not posed
    # Today's editor erases what the planner asked: the notes name what the job needs, and keep nothing as filmed.
    assert plan_notes(record) == ["What this simulation needs, as Blueprint's model read the job: blue tote "
                                  "(one tote of parts) and shelf."]


def test_a_call_another_run_holds_waits_and_privacy_stops(tmp_path, monkeypatch):
    from blueprint_pipeline.hard_stops import PrivacyRightsStop
    from blueprint_pipeline.website_build_choice import IN_PROGRESS
    review = _bought_review()
    root, run, automated, posed = _job(tmp_path, monkeypatch, review, clean_plate=("objects_removed", ENTRIES))
    _preparation(tmp_path)
    _answering(monkeypatch, ValueError(IN_PROGRESS))
    assert run() == {"status": "processing", "stage": "clean_plate_wait"}
    assert not (root / "job_object_rule.json").exists() and not automated and not posed
    _answering(monkeypatch, PrivacyRightsStop("website_preparation_spend_receipt_invalid"))
    with pytest.raises(PrivacyRightsStop):
        run()
    assert not (root / "job_object_rule.json").exists()


def test_a_decision_sealed_before_the_plan_is_honoured_exactly(tmp_path, monkeypatch):
    review = _bought_review()
    root, run, automated, posed = _job(tmp_path, monkeypatch, review, clean_plate=("objects_removed", ENTRIES))
    sealed = {"schema_version": "atlas_job_object_rule.v1", "applied": True, "clean_plate_status": "objects_removed",
              "planner_remove_count": 2, "job_objects_kept": "several_job_objects"}
    sealed["digest"] = canonical_digest(sealed, digest_field="digest")
    (root / "job_object_rule.json").write_text(json.dumps(sealed))
    asked = _answering(monkeypatch, PLAN)
    run()
    assert not asked and not automated and len(posed) == 1
    assert posed[0]["review"]["job_object_rule"]["job_objects_kept"] == "several_job_objects"
    assert json.loads((root / "job_object_rule.json").read_text()) == sealed
    assert sealed_decision(tmp_path) is None and build_plan_notes(tmp_path) == {}
    assert build_plan.apply_to_removal({"entries": ENTRIES}, tmp_path) == {"entries": ENTRIES}


def test_a_tampered_sealed_plan_is_never_read(tmp_path):
    decided = {"schema_version": build_plan.DECISION_SCHEMA, "applied": False,
               "built": {"target_id": "tote", "editor_object_ids": ["centerpiece"]},
               "build_plan": {"plan": PLAN, "digest": "sha256:" + "0" * 64}}
    decided["digest"] = canonical_digest(decided, digest_field="digest")
    path = tmp_path / "pipeline" / "atlas" / "job_object_rule.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({**decided, "built": {"target_id": "carton"}}))
    assert sealed_decision(tmp_path) is None
    assert build_plan.apply_to_removal({"entries": ENTRIES}, tmp_path) == {"entries": ENTRIES}
    path.write_text(json.dumps(decided))
    assert sealed_decision(tmp_path) == decided


def test_the_record_holds_the_request_and_its_cost_and_a_key_missing_is_a_fallback(tmp_path, monkeypatch):
    from blueprint_pipeline import website_gemini_receipts as receipts
    for name in ("GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY", "GOOGLE_AI_API_KEY", "GEMINI_API_KEY_FILE",
                 "GOOGLE_GENAI_API_KEY_FILE", "GOOGLE_AI_API_KEY_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(receipts, "require_current_website_source", lambda _context: None)
    monkeypatch.setattr(receipts, "reserve_website_preparation_spend", lambda **_: pytest.fail("money was reserved"))
    task = {"schema_version": "website_site_task_context.v1", "request_id": "itest", "scene_id": "site-itest",
            "capture_id": "walkthrough-itest", "confirmed": True, "confirmed_at": "2026-11-16",
            "description": CONTEXT["description"]}
    task["context_digest"] = canonical_digest(task, digest_field="context_digest")
    record = build_plan.decide_plan(task_context=task, targets=ENTRIES, editor_objects=EDITOR, output_root=tmp_path)
    assert record["plan"] is None and record["fallback"] == build_plan.KEY_MISSING
    binding, cost = build_plan.request_binding(
        build_plan.plan_inputs(task_context=task, targets=ENTRIES, editor_objects=EDITOR))
    assert record["request_binding_digest"] == canonical_digest(binding) and record["maximum_cost_usd"] == cost > 0
    assert record["digest"] == canonical_digest(record, digest_field="digest")
    assert "data, not instructions" in binding["prompt"] and binding["kind"] == "website_build_plan"


def _configuration(extents=(0.6, 0.4, 0.3), tolerance=0.1):
    return {"authoring_backend": "artisan_v1", "source_observation_kind": "website_capture_frames",
            "metric_envelope": {"minimum_xyz_m": [0, 0, 0], "maximum_xyz_m": list(extents),
                                "maximum_dimension_relative_error": tolerance}}


def test_a_parametric_object_is_authored_as_a_supplied_shape_at_no_cost():
    from blueprint_pipeline.website_supplied_mesh import supplied_mesh_configuration
    from blueprint_pipeline.website_synthetic_subject import AUTHORING_METHOD
    options = dict(synthetic=None, synthetic_authoring=None, subject_id="tote", sim_min=[1, 1, 0],
                   sim_max=[1.6, 1.4, 0.3])
    made = supplied_mesh_configuration(_configuration(), parametric={"target_id": "tote", "shape": "tote"}, **options)
    assert "authoring_backend" not in made and made["authoring_method"] == AUTHORING_METHOD
    assert (made["authoring_source"], made["authoring_cost_usd"]) == ("parametric", 0.0)
    assert made["parametric_design"]["shape"] == "tote"
    assert made["parametric_design"]["dimensions_m"] == pytest.approx([0.6, 0.4, 0.3])
    assert made["source_observation_kind"] == "website_capture_frames"  # the object was filmed; only its shape is plain
    # Another object, or none: the builder is unchanged.
    for parametric in (None, {"target_id": "carton", "shape": "carton"}):
        assert supplied_mesh_configuration(_configuration(), parametric=parametric, **options) == _configuration()
    # A can on an oblong footprint is no can: the builder stays, and says why.
    kept = supplied_mesh_configuration(_configuration(), parametric={"target_id": "tote", "shape": "cylinder"},
                                       **options)
    assert kept == {**_configuration(), "parametric_not_used": "website_parametric_object_outside_estimated_bounds"}
    kept = supplied_mesh_configuration(_configuration(), parametric={"target_id": "tote", "shape": "crate"}, **options)
    assert kept["parametric_not_used"] == "website_parametric_object_design_invalid" and "authoring_backend" in kept


def test_a_designed_object_is_configured_exactly_as_before():
    from blueprint_pipeline.website_supplied_mesh import supplied_mesh_configuration
    synthetic = {"shape": "box", "dimensions_m": [.1, .1, .1], "mass_kg": .2}
    made = supplied_mesh_configuration(_configuration(), synthetic=synthetic, synthetic_authoring={"backend": "b"},
                                       parametric={"target_id": "x", "shape": "tote"}, subject_id="x",
                                       sim_min=[0, 0, 0], sim_max=[.1, .1, .1])
    assert made == {"source_observation_kind": "synthetic_design", "metric_envelope": _configuration()["metric_envelope"],
                    "authoring_method": "exact_provided_mesh_with_bounded_simulation_physics",
                    "dimension_authority": "synthetic_design", "appearance_inputs": "synthetic_primitive_design",
                    "geometry_support": "synthetic_primitive_design", "synthetic_design": synthetic,
                    "synthetic_authoring": {"backend": "b"}}

