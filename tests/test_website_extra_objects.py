"""Several objects per job, stage 1: the build plan's ranked list of every object it would build, and for each one
after the first the free source it could use, as a record only. A job still builds its first object alone, and
nothing here reserves, spends, asks a model or fetches a file."""
# Covers: src/blueprint_pipeline/website_build_plan.py
# Covers: src/blueprint_pipeline/website_extra_objects.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import website_build_budget as budget
from blueprint_pipeline import website_build_plan as build_plan
from blueprint_pipeline import website_extra_objects as extra
from blueprint_pipeline import website_library_asset as asset
from blueprint_pipeline import website_object_library as library
from blueprint_pipeline import website_task_context as task_context
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_object_authoring_backend import ARTISAN_BUDGET_ENV

CONTEXT = {"capture_id": "c", "request_id": "r1", "context_digest": "sha256:" + "1" * 64,
           "description": "Move the blue tote, the carton and the cabinet's things from the bench to the shelf."}


def _entry(target: str, label: str, **fields) -> dict:
    return {"target_id": target, "semantic_label": label, "task_effect": "manipulated", "rigidity": "rigid",
            "rigidity_reason": "hard plastic", **fields}


ENTRIES = [_entry("tote", "blue tote"), _entry("carton", "carton"),
           _entry("cabinet", "cabinet", articulated_part="door", articulation_kind="revolute"),
           _entry("bag", "bag of rags", rigidity="deformable", rigidity_reason="cloth"),
           {"target_id": "shelf", "semantic_label": "shelf", "task_effect": "static_contact"}]
EDITOR = [*({"target_id": "e-" + name, "label": name, "disposition": "remove"}
            for name in ("tote", "carton", "cabinet", "bag")),
          {"target_id": "e-shelf", "label": "shelf", "disposition": "keep"}]


def _row(target: str, name: str, *, sourcing: str = "parametric", shape: str | None = None, rigidity: str = "rigid",
         role: str = "moved_object", ids: list[str] | None = None) -> dict:
    return {"target_id": target, "name": name, "role": role, "variety": None, "rigidity": rigidity, "size_m": None,
            "mass_kg": None, "sourcing": sourcing, "parametric_shape": shape, "reason": f"The job needs the {name}.",
            "editor_object_ids": ["e-" + target] if ids is None else ids}


def _plan(*rows: dict) -> dict:
    inputs = build_plan.plan_inputs(task_context=CONTEXT, targets=ENTRIES, editor_objects=EDITOR)
    plan = build_plan.validate_plan({"objects": list(rows)}, inputs=inputs, task_context=CONTEXT)
    assert "dropped" not in plan, plan
    return plan


TOTE = _row("tote", "blue tote", shape="tote")
SHELF = _row("shelf", "shelf", role="destination", sourcing="keep_as_filmed", ids=[])
PLAN = _plan(TOTE, SHELF, _row("carton", "carton", sourcing="generate"),
             _row("bag", "bag of rags", rigidity="soft", sourcing="keep_as_filmed"),
             _row("cabinet", "cabinet", rigidity="articulated", sourcing="generate"))
CATALOG = [{"id": "gso:carton", "license": "CC-BY-4.0"}]
MESH = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


def _sources(plan: dict, root: Path, **options) -> dict:
    return extra.extra_object_sources(plan, entries=ENTRIES, editor_objects=EDITOR, task_context=CONTEXT,
                                      library_root=root, **{"catalog_entries": CATALOG, **options})


def _seal(library_root: Path, row: dict, *, at: str | None = None, **fields) -> dict:
    """The object's library pick as the library step seals one, in its own directory (``at``: another target's):
    its reading, its credit and its fetched file."""
    root = extra.object_root(library_root, at or row["target_id"])
    (root / "files").mkdir(parents=True, exist_ok=True)
    mesh = root / "files" / "model.obj"
    mesh.write_bytes(MESH)
    subject = {"schema_version": library.SUBJECT_SCHEMA, "target_id": row["target_id"], "route": "extra",
               "reading_digest": library.reading_digest(row, CONTEXT),
               "attribution": {"entry_id": "gso:carton", "license": "CC-BY-4.0", "title": "Carton", "format": "obj"},
               "files": [{"role": "mesh", "path": str(mesh), "sha256": hashlib.sha256(MESH).hexdigest(),
                          "size_bytes": len(MESH), "suffix": ".obj"}],
               "pick": {"source": "model", "reason": "The same kind of object."}, **fields}
    subject["digest"] = canonical_digest(subject, digest_field="digest")
    (root / library.SUBJECT_RECORD).write_text(json.dumps(subject))
    return subject


def test_the_plan_ranks_every_object_it_would_build_and_its_first_is_still_the_one_built():
    ranked = build_plan.built_objects(PLAN, entries=ENTRIES, editor_objects=EDITOR)
    # The plan's order, without the destination it keeps as filmed and the soft bag.
    assert [row["target_id"] for row in ranked] == ["tote", "carton", "cabinet"]
    assert ranked[0] == {"target_id": "tote", "name": "blue tote", "sourcing": "parametric",
                         "parametric_shape": "tote", "editor_object_ids": ["e-tote"]}
    assert build_plan.built_object(PLAN, entries=ENTRIES, editor_objects=EDITOR) == ranked[0]
    reordered = {"objects": PLAN["objects"][::-1]}
    assert [row["target_id"] for row in build_plan.built_objects(reordered, entries=ENTRIES, editor_objects=EDITOR)
            ] == ["cabinet", "carton", "tote"]
    assert build_plan.built_object(reordered, entries=ENTRIES, editor_objects=EDITOR)["target_id"] == "cabinet"
    for nothing in (None, {"objects": [SHELF]}):
        assert build_plan.built_objects(nothing, entries=ENTRIES, editor_objects=EDITOR) == []
        assert build_plan.built_object(nothing, entries=ENTRIES, editor_objects=EDITOR) is None


def test_one_function_ranks_every_built_object_as_the_intake_request_does():
    from blueprint_pipeline.task_evaluation_scene_intent_contracts import ADDITIONAL_OBJECT_RANKS
    assert [build_plan.built_rank(position) for position in range(build_plan.MAX_OBJECTS)] == [1, 2, 3, 4, 5, 6, 7, 8]
    # The scored object is 1; every further object the plan can name fits the contract's ranks.
    assert (build_plan.built_rank(1), build_plan.built_rank(build_plan.MAX_OBJECTS - 1)) == ADDITIONAL_OBJECT_RANKS


def test_the_built_object_still_stops_at_the_first_it_can_build():
    # As before, a row after the one it builds is never read: a malformed later row changes nothing.
    plan = {"objects": [*PLAN["objects"][:1], {"target_id": "carton"}]}
    assert build_plan.built_object(plan, entries=ENTRIES, editor_objects=EDITOR)["target_id"] == "tote"
    with pytest.raises(KeyError):
        build_plan.built_objects(plan, entries=ENTRIES, editor_objects=EDITOR)


def test_a_job_with_one_object_to_build_has_no_extra_object_and_nothing_is_written(tmp_path):
    root = tmp_path / "object_library"
    for plan, first in ((_plan(TOTE, SHELF), {"target_id": "tote", "name": "blue tote"}), (None, None)):
        record = _sources(plan, root)
        assert record["first_object"] == first and record["objects"] == []
        assert record["free_count"] == record["artisan_count"] == 0 and record["artisan_maximum_cost_usd"] == 0
    assert not root.exists()


@pytest.mark.parametrize("sourcing, shape, sealed, source, why", [
    # A library model the plan also gave a shape: the model first, the shape if its proportions miss.
    ("parametric", "carton", True, extra.LIBRARY, None),
    # A box is made to the object's estimated bounds, so it always fits them.
    ("parametric", "open_bin", False, extra.PARAMETRIC, None),
    ("parametric", "tray", False, extra.PARAMETRIC, None),
    # A can's diameter is its footprint's mean side: it can miss the bounds and leave the object to ARTISAN.
    ("parametric", "cylinder", False, extra.ARTISAN, extra.CYLINDER_ONLY),
    # A library model with nothing to fall back on can miss the bounds too.
    ("library", None, True, extra.ARTISAN, extra.LIBRARY_WITHOUT_FALLBACK),
    ("generate", None, False, extra.ARTISAN, extra.NO_FREE_SOURCE),
    ("library", None, False, extra.ARTISAN, extra.NO_FREE_SOURCE),
])
def test_an_extra_object_is_free_only_with_a_library_model_and_a_shape_or_a_shape_that_is_not_a_can(
        tmp_path, monkeypatch, sourcing, shape, sealed, source, why):
    monkeypatch.delenv(ARTISAN_BUDGET_ENV, raising=False)
    carton = _row("carton", "carton", sourcing=sourcing, shape=shape)
    plan = _plan(TOTE, carton)
    if sealed:
        subject = _seal(tmp_path, plan["objects"][1])
    record = _sources(plan, tmp_path)
    assert record["first_object"] == {"target_id": "tote", "name": "blue tote"}
    (row,) = record["objects"]
    # Ranked as the intake request ranks it: the scored object is 1, so the first further object is 2.
    assert row["rank"] == 2 and row["target_id"] == "carton" and row["plan_sourcing"] == sourcing
    assert row["source"] == source and row["free"] is (source != extra.ARTISAN)
    assert row["parametric_shape"] == shape
    assert row["library"] == ({"status": "sealed_and_fetched", "entry_id": "gso:carton", "license": "CC-BY-4.0",
                               "subject_digest": subject["digest"]} if sealed
                              else {"status": "not_used", "reason": extra.NOT_PICKED})
    if source == extra.LIBRARY:
        assert row["fallback_shape"] == shape and "needs_artisan" not in row
    if source == extra.ARTISAN:
        assert row["needs_artisan"] == {"reason": why, "maximum_cost_usd": 12.0, "reserved": False}
    assert (record["free_count"], record["artisan_count"]) == ((0, 1) if source == extra.ARTISAN else (1, 0))
    assert record["artisan_maximum_cost_usd"] == (12.0 if source == extra.ARTISAN else 0)
    assert "flags" not in row


def test_a_library_model_whose_fallback_is_a_can_is_free_by_the_rule_and_flagged(tmp_path):
    plan = _plan(TOTE, _row("carton", "carton", shape="cylinder"))
    _seal(tmp_path, plan["objects"][1])
    (row,) = _sources(plan, tmp_path)["objects"]
    assert row["source"] == extra.LIBRARY and row["free"] is True and row["fallback_shape"] == "cylinder"
    assert row["flags"] == [extra.CYLINDER_FALLBACK]


@pytest.mark.parametrize("case", ["rights_refused_now", "license_changed", "other_reading", "not_used", "file_changed",
                                  "file_missing", "tampered", "another_targets_pick"])
def test_a_library_model_counts_only_when_sealed_for_this_reading_fetched_and_still_allowed(tmp_path, case):
    plan = _plan(TOTE, _row("carton", "carton", shape="cylinder"))
    carton = plan["objects"][1]
    catalog = CATALOG
    root = extra.object_root(tmp_path, "carton")
    if case == "other_reading":
        _seal(tmp_path, {**carton, "name": "cardboard carton"})
    elif case == "not_used":
        _seal(tmp_path, carton, not_used="website_object_library_no_match", attribution=None, files=None)
    else:
        subject = _seal(tmp_path, carton)
        if case == "rights_refused_now":
            catalog = [{"id": "gso:carton", "license": None, "license_refused": "license_off_list"}]
        elif case == "license_changed":
            catalog = [{"id": "gso:carton", "license": "CC0-1.0"}]
        elif case == "file_changed":
            Path(subject["files"][0]["path"]).write_bytes(MESH + b"# changed\n")
        elif case == "file_missing":
            Path(subject["files"][0]["path"]).unlink()
        elif case == "tampered":
            written = json.loads((root / library.SUBJECT_RECORD).read_text())
            (root / library.SUBJECT_RECORD).write_text(json.dumps({**written, "route": "built"}))
        elif case == "another_targets_pick":
            # The tote's pick in the carton's directory, and the carton's own in the tote's: each read by its target.
            (root / library.SUBJECT_RECORD).unlink()
            _seal(tmp_path, plan["objects"][0], at="carton")
            _seal(tmp_path, carton, at="tote")
    (row,) = _sources(plan, tmp_path, catalog_entries=catalog)["objects"]
    # Without the model, the can alone is left: ARTISAN, said why.
    assert row["library"]["status"] == "not_used" and row["source"] == extra.ARTISAN
    assert row["needs_artisan"]["reason"] == extra.CYLINDER_ONLY
    assert row["library"]["reason"] == {
        "rights_refused_now": "website_object_library_license_refused_now:license_off_list",
        "license_changed": "website_object_library_license_changed:CC-BY-4.0->CC0-1.0",
        "other_reading": extra.READING_CHANGED, "not_used": "website_object_library_no_match",
        "file_changed": extra.FILES_CHANGED, "file_missing": extra.FILES_CHANGED, "tampered": extra.NOT_PICKED,
        "another_targets_pick": extra.NOT_PICKED}[case]


def test_an_object_that_opens_is_never_free(tmp_path):
    # The analysis saw a door the plan's rigid shape would drop: supplied geometry is one rigid body.
    plan = _plan(TOTE, _row("cabinet", "cabinet", shape="carton"))
    _seal(tmp_path, plan["objects"][1])
    (row,) = _sources(plan, tmp_path)["objects"]
    assert row["source"] == extra.ARTISAN and row["needs_artisan"]["reason"] == extra.OPENS
    assert row["library"]["status"] == "sealed_and_fetched"


def test_the_record_is_deterministic_and_reserves_spends_asks_and_fetches_nothing(tmp_path, monkeypatch):
    def forbidden(*_, **__):
        pytest.fail("a record of free sources reserves, spends, asks and fetches nothing")

    for module, name in ((library, "ask_gemini"), (library, "decide_library_object"), (asset, "download"),
                         (asset, "fetch_pinned"), (task_context, "reserve_website_preparation_spend"),
                         (budget, "admit_extra_objects")):
        monkeypatch.setattr(module, name, forbidden)
    plan = _plan(TOTE, _row("carton", "carton", shape="carton"), _row("cabinet", "cabinet", sourcing="generate"))
    _seal(tmp_path, plan["objects"][1])
    written = sorted(tmp_path.rglob("*"))
    first = _sources(plan, tmp_path)
    again = extra.extra_object_sources(plan, entries=ENTRIES[::-1], editor_objects=EDITOR[::-1], task_context=CONTEXT,
                                       library_root=tmp_path, catalog_entries=CATALOG)
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)
    assert first["digest"] == canonical_digest(first, digest_field="digest")
    assert [(row["rank"], row["target_id"], row["source"]) for row in first["objects"]] == [
        (2, "carton", extra.LIBRARY), (3, "cabinet", extra.ARTISAN)]
    assert sorted(tmp_path.rglob("*")) == written  # nothing written
    # Each object's pick has its own directory, keyed by its target.
    assert extra.object_root(tmp_path, "carton") != extra.object_root(tmp_path, "cabinet")
    assert extra.object_root(tmp_path, "carton") == extra.object_root(tmp_path, "carton")
