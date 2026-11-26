"""Several objects per job, stage 5: a job's free further objects are taken out of the scan with its own object and
built, and an object taken out and then not built is never said to be in the scan.

A decision sealed under v3 (``website_build_plan_notes.DECISION_SCHEMA``) takes out of the scan, before the room is
reconstructed, each further object sure of a free build then (a rigid shape that is not a can); the handoff reads
compile's further objects from it (``website_further_objects.sealed_inputs``), compile builds and composes them, and
the scene half and the company's package carry them. A paid one stays in the scan as filmed, named, under a reason
that claims no limit refused it. A decision sealed under v2 keeps its route byte for byte.

Fixtures: ``tests.test_website_further_objects``'s cup on its table with more objects beside it, and its scene half
and package. Offline: no provider, no network, nothing paid.
"""
# Covers: src/blueprint_pipeline/website_build_plan.py
# Covers: src/blueprint_pipeline/website_build_plan_notes.py
# Covers: src/blueprint_pipeline/website_extra_objects.py
# Covers: src/blueprint_pipeline/website_further_objects.py
# Covers: src/blueprint_pipeline/website_atlas_pipeline.py
# Covers: src/blueprint_pipeline/website_scene_handoff.py
# Covers: src/blueprint_pipeline/website_half_preparation.py
# Covers: src/blueprint_pipeline/website_packet_facts.py
# Covers: src/blueprint_pipeline/atlas_scene_estimates.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import website_build_plan as build_plan
from blueprint_pipeline import website_further_objects as further
from blueprint_pipeline import website_library_catalog as library_catalog
from blueprint_pipeline import website_task_preparation as preparation
from blueprint_pipeline.atlas_scene_estimates import preparation_record_flags
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_build_plan_notes import (DECISION_SCHEMA, JOB_OBJECT_RECORD, plan_notes,
                                                         sealed_decision)
from tests import test_website_further_objects as fo

pytest.importorskip("pxr")

V2 = "atlas_job_object_rule.v2"
#: The plan the job's model returns: the cup first (the scored object), then the further objects it names.
SOURCING = {"cup-1": ("generate", None), "artisan-1": ("generate", None), "can-1": ("parametric", "cylinder")}


@pytest.fixture(autouse=True)
def _catalog(monkeypatch):
    """The deployed catalog allows the test model under the license it was chosen under (rights, read now); and the
    fixture's objects are labelled in plain object words, so the README names each by its own label."""
    monkeypatch.setattr(library_catalog, "catalog", lambda *_, **__: [{"id": fo.ENTRY_ID, "license": "CC-BY-4.0"}])
    monkeypatch.setitem(fo.LABELS, "artisan-1", "toy")
    monkeypatch.setitem(fo.LABELS, "overlap-1", "bowl")
    monkeypatch.setitem(fo.LABELS, "can-1", "can")


def _rows(names: list[str]) -> list[dict]:
    rows = []
    for name in ["cup-1", *names]:
        sourcing, shape = SOURCING.get(name, ("parametric", "carton"))
        rows.append({**fo._plan_row(name, sourcing=sourcing, shape=shape)})
    return rows


def _review(names: list[str]) -> dict:
    """The Atlas planner's review: one editor row per object, every one asked to be removed."""
    objects = [{"target_id": "e-" + name, "label": fo.LABELS.get(name, name), "segmentation_prompt": name,
                "disposition": "remove", "regions": [{"frame_id": "f0", "box_xywh_normalized": [.1, .1, .2, .2]}]}
               for name in ["cup-1", *names]]
    return {"subject_origin": "captured", "uncaptured_regions": [],
            "coverage_requirements": [{"role": "task_object", "label": "t", "required_frame_ids": ["f0"]}],
            "automation": {"scale_references": [], "objects": objects, "removal_instruction": "the planner's words",
                           "mask_margin_fraction": .01}}


def _reviewed(capture: Path, arguments: dict, names: list[str], monkeypatch, *, schema: str | None = None):
    """The Atlas job's job-object decision, made as the job makes it (``job_object_review``) on the clean plate's
    entries and a plan the model returns, and the review the scene editor consumes. ``schema``: a decision sealed
    earlier under that schema, written before the job reads it (with no ``further``, as v2 sealed it)."""
    from blueprint_pipeline import website_atlas_pipeline as atlas
    clean = capture / "pipeline" / "clean_plate"
    write_json(clean / "clean_plate_stage_manifest.json", {"status": "objects_removed"})
    write_json(clean / "removal_manifest.json", arguments["removal_manifest"])
    monkeypatch.setattr(build_plan, "ask_gemini", lambda **_: {"answer": {"objects": _rows(names)}})
    root = capture / "pipeline" / "atlas"
    root.mkdir(parents=True, exist_ok=True)
    if schema is not None:
        entries = arguments["removal_manifest"]["entries"]
        record = build_plan.decide_plan(task_context=arguments["task_context"], targets=entries,
                                        editor_objects=_review(names)["automation"]["objects"],
                                        output_root=root / "build_plan")
        built = build_plan.built_objects(record["plan"], entries=entries,
                                         editor_objects=_review(names)["automation"]["objects"])
        decided = {"schema_version": schema, "applied": False, "clean_plate_status": "objects_removed",
                   "planner_remove_count": len(names) + 1, "build_plan": record, "built": built[0],
                   "kept_in_scan_target_ids": sorted(names)}
        decided["digest"] = canonical_digest(decided, digest_field="digest")
        write_json(root / JOB_OBJECT_RECORD, decided)
    consumed, waiting = atlas.job_object_review(root, _review(names), arguments["task_context"])
    assert waiting is False
    return json.loads((root / JOB_OBJECT_RECORD).read_text()), consumed


# --- a decision sealed under v2 keeps its route exactly --------------------------------------------------------------

def test_a_v2_decision_keeps_its_route_byte_for_byte(tmp_path, monkeypatch):
    arguments = fo._job(tmp_path / "fixture", ["carton-1", "mug-1"])
    capture = tmp_path / "capture"
    decided, consumed = _reviewed(capture, arguments, ["carton-1", "mug-1"], monkeypatch, schema=V2)
    assert decided["schema_version"] == V2 and "further" not in decided
    assert sealed_decision(capture) == decided  # still read: its plan notes, library route and names stand
    # The editor erases the built object's rows alone, with the words it always had.
    assert [row["disposition"] for row in consumed["automation"]["objects"]] == ["remove", "keep", "keep"]
    assert consumed["automation"]["removal_instruction"] == build_plan.PARTIAL_REMOVAL_INSTRUCTION
    # Every further object stays in the scan, marked exactly as before stage 5.
    removal = arguments["removal_manifest"]
    before = {**removal, "entries": [{**row, "task_effect": "kept_in_scan", "kept_in_scan_reason": "build_plan"}
                                     if row["target_id"] != "cup-1" and row.get("task_effect") == "manipulated"
                                     else row for row in removal["entries"]]}
    marked = build_plan.apply_to_removal(removal, capture)
    assert json.dumps(marked, sort_keys=True) == json.dumps(before, sort_keys=True)
    # No further object reaches compile, so its preparation is as before.
    assert further.sealed_inputs(decided, entries=removal["entries"], task_context=arguments["task_context"],
                                 library_root=tmp_path / "library") == []
    value = preparation.compile_website_scene_preparation(**{**arguments, "removal_manifest": marked})
    assert value["status"] == "intake_ready", value["blockers"]
    assert "further_objects" not in value and "additional_objects" not in value["intake_request"]["task"]
    assert value["binding"]["removal_manifest_digest"] == canonical_digest(before)
    # Its plan notes are the stage 4 notes.
    assert plan_notes(decided)[1:] == [
        "The plan takes this out of the scan to build as its own object: cup.",
        "The plan keeps these in the scan as filmed, not built on their own: carton and mug box."]


# --- taken out of the scan, then not built: said so, never "in the scan" --------------------------------------------

def _decided(capture: Path, arguments: dict, names: list[str]) -> dict:
    """A v3 decision for ``names`` sealed as ``job_object_review`` seals one, without running the Atlas job."""
    entries = arguments["removal_manifest"]["entries"]
    editor = _review(names)["automation"]["objects"]
    inputs = build_plan.plan_inputs(task_context=arguments["task_context"], targets=entries, editor_objects=editor)
    plan = build_plan.validate_plan({"objects": _rows(names)}, inputs=inputs, task_context=arguments["task_context"])
    every = build_plan.built_objects(plan, entries=entries, editor_objects=editor)
    rows = build_plan.further_removal(plan, every, entries=entries)
    decided = {"schema_version": DECISION_SCHEMA, "applied": False, "build_plan": {"plan": plan, "digest": "x"},
               "built": every[0], "further": rows,
               "kept_in_scan_target_ids": sorted(row["target_id"] for row in rows if not row["removed"])}
    decided["digest"] = canonical_digest(decided, digest_field="digest")
    write_json(capture / "pipeline" / "atlas" / JOB_OBJECT_RECORD, decided)
    return decided


def test_an_object_taken_out_and_not_built_is_said_to_be_in_neither_the_scan_nor_the_simulation(tmp_path,
                                                                                                monkeypatch):
    capture = tmp_path / "capture"
    names = ["carton-1", "static-1", "overlap-1", "artisan-1"]
    arguments = fo._job(tmp_path / "fixture", names)
    decided = _decided(capture, arguments, names)
    original = further.static_check

    def failing(path, **options):
        return "no_collider" if "rank-03" in str(path) else original(path, **options)
    monkeypatch.setattr(further, "static_check", failing)
    marked = build_plan.apply_to_removal(arguments["removal_manifest"], capture)
    inputs = further.sealed_inputs(decided, entries=arguments["removal_manifest"]["entries"],
                                   task_context=arguments["task_context"], library_root=tmp_path / "library")
    assert [(row["rank"], row.get(further.NOT_BUILT)) for row in inputs] == [
        (2, None), (3, None), (4, None),
        (5, "further_object_needs_artisan:website_extra_object_no_free_source")]
    value = preparation.compile_website_scene_preparation(**{**arguments, "removal_manifest": marked},
                                                           further_objects=inputs)
    assert value["status"] == "intake_ready", value["blockers"]  # the scored object's job goes on
    record = value["further_objects"]
    assert [row["target_id"] for row in record["objects"]] == ["carton-1"]
    assert record["removed_not_built"] == [
        {"rank": 3, "target_id": "static-1", "reason": "further_object_static_check_failed:no_collider"},
        {"rank": 4, "target_id": "overlap-1", "reason": "further_object_overlaps:scored_object_start"}]
    assert record["left_in_scan"] == [
        {"rank": 5, "target_id": "artisan-1", "reason": "further_object_needs_artisan:website_extra_object_no_free_source"}]
    assert record["flags"] == [further.LEFT_IN_SCAN, further.REMOVED_NOT_BUILT]
    assert {further.LEFT_IN_SCAN, further.REMOVED_NOT_BUILT} <= preparation_record_flags(value)
    # The company's page says it in plain words that never put the object in the scan.
    from blueprint_pipeline.website_packet_facts import ESTIMATE_SENTENCES
    assert ESTIMATE_SENTENCES[further.REMOVED_NOT_BUILT] == (
        "One or more of the job's other objects were taken out of the scan to be built as objects of their own, "
        "but could not be built, so they are not in this simulation.")
    assert "scan shows" not in ESTIMATE_SENTENCES[further.REMOVED_NOT_BUILT]
    # The package's source names each, and its plan note says what became of each.
    from blueprint_pipeline import website_half_preparation as half_preparation
    told = half_preparation.further_objects(value, capture)["further_objects"]
    assert [(row["name"], row["reason"]) for row in told["removed_not_built"]] == [
        ("crate", "further_object_static_check_failed:no_collider"),
        ("bowl", "further_object_overlaps:scored_object_start")]
    notes = plan_notes(decided, None, told)
    assert notes[1:] == [
        "The plan takes this out of the scan to build as its own object: cup.",
        "Also built as its own object, where the video shows it: carton.",
        "Taken out of the scan to be built as its own object, but it could not be built, so it is not in this "
        "simulation: crate, because its file did not pass Blueprint's static check.",
        "Taken out of the scan to be built as its own object, but it could not be built, so it is not in this "
        "simulation: bowl, because where the video shows it, it would overlap the job's object.",
        "Left in the scan as filmed, not built on its own: toy, because it needs a model Blueprint would "
        "generate, and none was made for it."]
    assert not any("Left in the scan" in note and ("crate" in note or "bowl" in note) for note in notes)


def test_without_its_record_a_v3_plan_note_says_the_plan_takes_them_out_and_never_keeps_them(tmp_path):
    capture = tmp_path / "capture"
    names = ["carton-1", "mug-1", "artisan-1"]
    decided = _decided(capture, fo._job(tmp_path / "fixture", names), names)
    assert plan_notes(decided)[1:] == [
        "The plan takes this out of the scan to build as its own object: cup.",
        "The plan also takes these out of the scan to build as their own objects: carton and mug box.",
        "The plan keeps these in the scan as filmed, not built on their own: toy."]
    one = {**decided, "further": decided["further"][1:]}
    one["digest"] = canonical_digest(one, digest_field="digest")
    assert plan_notes(one)[2] == "The plan also takes this out of the scan to build as its own object: mug box."


def test_a_further_object_read_otherwise_than_the_decision_read_it_is_never_built(tmp_path):
    capture = tmp_path / "capture"
    names = ["carton-1", "mug-1"]
    arguments = fo._job(tmp_path / "fixture", names)
    decided = _decided(capture, arguments, names)
    # The clean plate no longer names the carton as an object the job moves: the mug box moves up a rank.
    entries = [row for row in arguments["removal_manifest"]["entries"] if row["target_id"] != "carton-1"]
    inputs = further.sealed_inputs(decided, entries=entries, task_context=arguments["task_context"],
                                   library_root=tmp_path / "library")
    assert [(row["rank"], row["target_id"], row.get(further.NOT_BUILT)) for row in inputs] == [
        (2, "carton-1", "further_object_input_invalid"), (3, "mug-1", "further_object_input_invalid")]
    # A decision sealed earlier, or none, names no further object.
    for other in (None, {**decided, "schema_version": V2}):
        assert further.sealed_inputs(other, entries=entries, task_context=arguments["task_context"],
                                     library_root=tmp_path / "library") == []
