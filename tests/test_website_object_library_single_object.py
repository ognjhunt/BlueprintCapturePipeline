"""Library-first for the one-object job (owner, 2026-11-24: "yes extend library-first to single-object jobs next"):
the object today's one-object route removes from the scan and authors is tried as a licensed library model, then as
the plan's free parametric shape, then by ARTISAN as today. What is removed from the scan, and the object the
success rule scores, are unchanged; the plan keeping the object as filmed, a route that cannot be read, and a job
compiled before this route change nothing.

Offline: the library tests' fixture catalog and mesh, an injected fetcher and model answer; no Gemini key is set and
nothing is paid for.
"""
# Covers: src/blueprint_pipeline/website_object_library.py
# Covers: src/blueprint_pipeline/website_build_plan.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import website_object_library as library
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_build_plan import apply_to_removal
from blueprint_pipeline.website_build_plan_notes import DECISION_SCHEMA
from blueprint_pipeline.website_supplied_mesh import LIBRARY_SUBJECT, PARAMETRIC_SUBJECT, supplied_mesh_configuration
from tests.test_website_object_library import THUMB, TASK, Fetcher, _asking, _box_obj, _configuration, _index, _record

pytest.importorskip("pxr")

ENTRY = {"target_id": "tote", "semantic_label": "storage tote", "task_effect": "manipulated", "rigidity": "rigid",
         "rigidity_reason": "plastic", "spatial_evidence": []}
REMOVAL = {"schema_version": "clean_plate_removal_manifest.v1",
           "entries": [ENTRY, {"target_id": "shelf", "task_effect": "static_contact"}]}
ROW = {"target_id": "tote", "name": "storage tote", "role": "moved_object", "variety": None, "rigidity": "rigid",
       "size_m": {"low": 0.5, "high": 0.7, "basis": "estimated", "from": "the analysis"}, "mass_kg": None,
       "sourcing": "generate", "parametric_shape": None, "reason": "The job moves the tote.", "editor_object_ids": []}


def _sealed_plan(capture: Path, rows=(ROW,), *, tamper: bool = False) -> None:
    """The Atlas job's sealed decision for a one-object job: the plan is recorded, no object taken out alone."""
    decided = {"schema_version": DECISION_SCHEMA, "applied": False, "clean_plate_status": "objects_removed",
               "planner_remove_count": 1, "build_plan": {"plan": {"objects": [dict(row) for row in rows]}}}
    decided["digest"] = canonical_digest(decided, digest_field="digest")
    if tamper:
        decided["applied"] = True
    (capture / "pipeline" / "atlas").mkdir(parents=True, exist_ok=True)
    (capture / "pipeline" / "atlas" / "job_object_rule.json").write_text(json.dumps(decided))


def _attach(tmp_path: Path, removal=REMOVAL, *, answer=None, fetch=None):
    ask, asked = _asking(answer or {"candidate_id": "c01", "reason": "The same stackable tote."})
    out = library.attach_library_subject(dict(removal), capture_root=tmp_path, task_context=TASK,
                                         output_root=tmp_path / "handoff" / "object_library",
                                         entries=library.catalog(_index(tmp_path)), fetch=fetch or Fetcher(), ask=ask)
    return out, asked


def _compile(removal, *, extents=(0.6, 0.4, 0.3), root=None, mechanism=False):
    configuration = {**_configuration(), **({"mechanism": {}} if mechanism else {})}
    return supplied_mesh_configuration(configuration, synthetic=None, synthetic_authoring=None,
                                       parametric=removal.get(PARAMETRIC_SUBJECT), subject_id="tote",
                                       sim_min=[0, 0, 0], sim_max=list(extents), library=removal.get(LIBRARY_SUBJECT),
                                       library_root=root)


def test_a_single_rigid_object_with_a_library_match_is_built_from_the_model_at_no_cost(tmp_path):
    from blueprint_pipeline.website_synthetic_subject import AUTHORING_METHOD
    _sealed_plan(tmp_path)
    removal, asked = _attach(tmp_path)
    subject = removal[LIBRARY_SUBJECT]
    assert subject["route"] == "one_object" and subject["row_source"] == "plan" and "not_used" not in subject
    assert subject["attribution"]["title"] == "Sterilite Storage Tote, Blue" and len(asked) == 1
    # The same object, kept where it was: nothing else in the removal changes.
    assert removal["entries"] == REMOVAL["entries"]
    made = _compile(removal, root=tmp_path / "model")
    assert "authoring_backend" not in made and made["authoring_method"] == AUTHORING_METHOD
    assert made["authoring_source"] == "library" and made["authoring_cost_usd"] == 0.0
    # Sealed once: a rerun asks nothing.
    again, asked_again = _attach(tmp_path)
    assert again == removal and not asked_again


@pytest.mark.parametrize("sourcing, shape, extents, expected", [
    ("parametric", "carton", (0.6, 0.4, 0.3), "parametric"),   # no model of it: the plan's shape, which fits
    ("parametric", "cylinder", (0.8, 0.3, 0.3), "artisan"),    # a shape that does not fit: today's builder
    ("generate", None, (0.6, 0.4, 0.3), "artisan")])           # no shape: today's builder
def test_with_no_library_match_the_parametric_shape_then_artisan_as_today(tmp_path, sourcing, shape, extents, expected):
    row = {**ROW, "name": "storage crate", "sourcing": sourcing, "parametric_shape": shape}
    _sealed_plan(tmp_path, [row])
    removal = apply_to_removal(REMOVAL, tmp_path)
    assert (PARAMETRIC_SUBJECT in removal) is (sourcing == "parametric")
    removal, _ = _attach(tmp_path, removal, answer={"candidate_id": "none", "reason": "No crate among them."})
    assert removal[LIBRARY_SUBJECT]["not_used"] == library.NO_MATCH
    made = _compile(removal, extents=extents, root=tmp_path / "model")
    assert made["library_not_used"] == library.NO_MATCH
    if expected == "parametric":
        assert made["authoring_source"] == "parametric" and "authoring_backend" not in made
    else:
        assert made["authoring_backend"] == "artisan_v1" and "library_design" not in made


@pytest.mark.parametrize("case", ["keep_as_filmed", "compiled_before", "no_decision", "tampered", "two_objects",
                                  "designed", "unreadable_entries"])
def test_what_the_plan_keeps_as_filmed_or_a_route_that_cannot_be_read_changes_nothing(tmp_path, case):
    removal = dict(REMOVAL)
    if case == "keep_as_filmed":
        _sealed_plan(tmp_path, [{**ROW, "sourcing": "keep_as_filmed"}])
    elif case == "compiled_before":
        _sealed_plan(tmp_path, [{**ROW, "sourcing": "parametric", "parametric_shape": "carton"}])
        (tmp_path / "handoff").mkdir()
        (tmp_path / "handoff" / "preparation.json").write_text("{}")  # compiled for today's builder
        removal = apply_to_removal(removal, tmp_path)
        assert PARAMETRIC_SUBJECT in removal
    elif case != "no_decision":
        _sealed_plan(tmp_path, tamper=case == "tampered")
        if case == "two_objects":
            removal["entries"] = [ENTRY, {**ENTRY, "target_id": "carton"}]
        elif case == "designed":
            removal["synthetic_subject"] = {"shape": "box"}
        elif case == "unreadable_entries":
            removal["entries"] = {"tote": ENTRY}
    out, asked = _attach(tmp_path, removal)
    assert LIBRARY_SUBJECT not in out and PARAMETRIC_SUBJECT not in out and not asked
    assert out == {key: value for key, value in removal.items() if key != PARAMETRIC_SUBJECT}
    assert _compile(out) == _configuration()  # today's builder, exactly
    record = tmp_path / "handoff" / "object_library" / library.SUBJECT_RECORD
    if case in ("keep_as_filmed", "compiled_before"):  # the skip is still recorded, with its reason
        reason = library.KEEPS_AS_FILMED if case == "keep_as_filmed" else library.ROUTE_ALREADY_COMPILED
        assert json.loads(record.read_text())["not_used"] == reason
    else:
        assert not record.exists()


@pytest.mark.parametrize("entry, row", [({**ENTRY, "articulated_part": "lid"}, None),
                                        (ENTRY, {**ROW, "rigidity": "articulated"}),
                                        ({**ENTRY, "rigidity": "deformable"}, None)])
def test_an_opening_or_soft_object_keeps_todays_route(tmp_path, entry, row):
    _sealed_plan(tmp_path, [row] if row else [])
    removal, asked = _attach(tmp_path, {**REMOVAL, "entries": [entry]})
    assert removal[LIBRARY_SUBJECT]["not_used"] == library.NOT_RIGID and not asked
    assert _compile(removal, mechanism=True)["authoring_backend"] == "artisan_v1"


def test_with_no_plan_row_the_analysis_names_the_object_and_the_fallback_never_guesses_a_size(tmp_path):
    _sealed_plan(tmp_path, [])
    removal, asked = _attach(tmp_path)
    assert removal[LIBRARY_SUBJECT]["row_source"] == "analysis" and asked[0]["binding"]["prompt"].count("storage tote")
    # Without the model, a plan-less object has no size to check, so the conservative fallback takes nothing.
    record = library.decide_library_object(task_context=TASK, plan_row=library.single_object_row(
        {"schema_version": DECISION_SCHEMA, "build_plan": {"plan": {"objects": []}}}, REMOVAL)[0],
        output_root=tmp_path / "keyless", entries=library.catalog(_index(tmp_path)), fetch=Fetcher())
    assert record["fallback"] == library.KEY_MISSING and record["chosen"] is None


def test_the_one_object_library_model_reserves_no_artisan_money(tmp_path, monkeypatch):
    """End to end through compile and the native submission: the attempt's spend has no ARTISAN allocation (the
    supplied-mesh stage calls no provider), where today's builder reserves its $12."""
    import tests.test_website_native_submission as submission
    from blueprint_pipeline.website_native_submission import materialize_website_submission
    monkeypatch.delenv("BLUEPRINT_TASK_OBJECT_AUTHORING_BACKEND", raising=False)

    def request(root):
        return json.loads((root / "scene_configuration_preparation_request.v1.json").read_text())
    (tmp_path / "today").mkdir()
    kwargs, _ = submission.setup(tmp_path / "today", monkeypatch, authoring_backend="artisan_v1")
    materialize_website_submission(**kwargs)
    today = request(kwargs["staging_root"])
    preparation = json.loads(Path(kwargs["task"]["preparation"]["path"]).read_text())
    low, high = preparation["subject"]["aabb_min_xyz"], preparation["subject"]["aabb_max_xyz"]
    size = [round(b - a, 6) for a, b in zip(low, high, strict=True)]
    # A library model of the filmed cup, the same size; the plan names the cup and its route authors it.
    mesh = _box_obj(*size)
    fixture = tmp_path / "catalog"
    fixture.mkdir()
    import hashlib
    record = {**_record("Ceramic_Cup", "Ceramic Cup", size=size),
              "mesh": {"sha256": hashlib.sha256(mesh).hexdigest(), "size_bytes": len(mesh), "faces": 12}}
    entries = library.catalog(_index(fixture, [record]))
    capture = tmp_path / "library-capture"
    _sealed_plan(capture, [{**ROW, "target_id": "cup-1", "name": "cup", "size_m": None}])

    def fetch(url, target, *, sha256, max_bytes):
        Path(target).write_bytes(mesh if url.endswith(".obj") else THUMB)
    ask, _ = _asking({"candidate_id": "c01", "reason": "The same cup."})
    real = submission.compile_website_scene_preparation

    def compile_with_library(**args):
        args["removal_manifest"] = library.attach_library_subject(
            args["removal_manifest"], capture_root=capture, task_context=TASK,
            output_root=tmp_path / "handoff" / "object_library", entries=entries, fetch=fetch, ask=ask)
        return real(**args)
    monkeypatch.setattr(submission, "compile_website_scene_preparation", compile_with_library)
    (tmp_path / "library").mkdir()
    kwargs, _ = submission.setup(tmp_path / "library", monkeypatch, authoring_backend="artisan_v1")
    materialize_website_submission(**kwargs)
    sourced = request(kwargs["staging_root"])
    assert today["replacement_authoring_backend"] == "artisan_v1"
    assert today["spend"]["external_service_caps"]["openai"]["maximum_cost_usd"] == 12.0
    assert "replacement_authoring_backend" not in sourced
    openai = sourced["spend"]["external_service_caps"]["openai"]
    assert openai["stage_max_cost_usd"]["content_agents"] < 1 and openai["maximum_cost_usd"] < 1
    assert sourced["spend"]["hard_cap_usd"] <= today["spend"]["hard_cap_usd"] - 11.5
    recipe = json.loads((kwargs["staging_root"] / "configuration/recipe.json").read_text())
    third = recipe["stage_sequence"][2]
    assert third["adapter"]["id"] == "provided_mesh_rigid_authoring" and third["execution_class"] == "no_spend"
    configuration = json.loads(Path(kwargs["task"]["preparation"]["path"]).read_text())["authoring_inputs"][
        "configuration"]
    assert configuration["library_design"]["attribution"]["title"] == "Ceramic Cup"
    assert configuration["source_object_identity"] == preparation["authoring_inputs"]["configuration"][
        "source_object_identity"]  # the same object the success rule scores
