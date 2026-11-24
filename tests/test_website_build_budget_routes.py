"""The build budget reads what compile builds each extra object with, and a retained hold is never a grant after the
site withdrew (the $0 stress run of round 2, 2026-11-24).

The real reservation path (``reserve_website_preparation_spend`` and its paid chokepoint) onto the budget tests'
stand-in WebApp ledger (``tests/test_website_build_budget.Ledger``: $8 scan + $20 scene intake held, so $12 is left
under the $40). No provider, no network, nothing paid for. ``admit_extra_objects`` has no live caller yet (one object
per job today): these close its gaps before the several-objects build calls it.
"""
# Covers: src/blueprint_pipeline/website_build_budget.py
# Covers: src/blueprint_pipeline/website_supplied_mesh.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import paid_resource_admission as admission
from blueprint_pipeline import website_build_budget as budget
from blueprint_pipeline import website_task_context
from blueprint_pipeline.hard_stops import PrivacyRightsStop
from tests import test_website_build_budget as budget_tests
from tests.test_website_build_budget import CONTEXT, OBJECTS, PLAN_DIGEST, Ledger, _compiled, _object

#: The budget tests' stand-in WebApp ledger, as a fixture of this module too.
ledger = budget_tests.ledger

pytest.importorskip("pxr")


def _admit(root, objects, configurations=None, digest=PLAN_DIGEST):
    return budget.admit_extra_objects(objects, task_context=CONTEXT, build_plan_digest=digest, root=root,
                                      configurations=_compiled(objects) if configurations is None else configurations)


def test_library_and_parametric_objects_never_reserve(tmp_path, ledger):
    """Control: a free object reserves nothing wherever it ranks; only the generated one asks."""
    objects = [_object("tote", "blue tote", "library"), _object("bin", "open bin", "parametric"),
               _object("mug", "mug", "library"), _object("carton", "carton", "generate")]
    record = _admit(tmp_path, objects)
    assert [row["status"] for row in record["objects"]] == [budget.FIRST, budget.FREE, budget.FREE, budget.ADMITTED]
    assert [spend["allocation_binding_digest"] for spend in ledger.reservations()] == [
        record["objects"][3]["binding_digest"]]


def _library_built(tmp_path: Path, target: str):
    """Compile's configuration of an object the plan calls ``generate`` but the library builds from a free model."""
    from blueprint_pipeline import website_library_asset as asset
    from blueprint_pipeline import website_object_library as library
    from blueprint_pipeline.website_supplied_mesh import supplied_mesh_configuration
    from tests.test_website_object_library import TASK, Fetcher, _asking, _configuration, _index
    row = {**_object(target, "blue storage tote", "generate"),
           "size_m": {"low": 0.5, "high": 0.7, "basis": "estimated", "from": "the analysis"}}
    ask, _ = _asking({"candidate_id": "c01", "reason": "The same stackable tote."})
    choice = library.decide_library_object(task_context=TASK, plan_row=row, output_root=tmp_path / "lib",
                                           entries=library.catalog(_index(tmp_path)), fetch=Fetcher(), ask=ask)
    assert choice["chosen"] is not None  # a free licensed model of it exists
    subject = {"target_id": target, "attribution": choice["chosen"],
               "files": asset.fetch_files(choice["chosen"], tmp_path / "lib" / "files", fetch=Fetcher())}
    made = supplied_mesh_configuration(_configuration(), synthetic=None, synthetic_authoring=None, parametric=None,
                                       subject_id=target, sim_min=[0, 0, 0], sim_max=[0.6, 0.4, 0.3], library=subject,
                                       library_root=tmp_path / "model")
    assert made["authoring_source"] == "library"
    return row, made


def test_an_object_the_plan_sources_generate_but_the_library_has_reserves_nothing(tmp_path, ledger):
    """Library first "whatever the plan's sourcing said", but the budget read only the plan's ``sourcing``: an extra
    object the plan calls ``generate`` that the library builds from a free model still reserved ARTISAN's $12, and
    with $12 left under the $40 it took the room a paid object ranked below it needed. Now the budget reads compile's
    result for each object."""
    row, made = _library_built(tmp_path, "tote2")
    objects = [_object("tote", "blue tote", "generate"), row, _object("crate", "crate", "generate")]
    configurations = {"tote": {"authoring_backend": "artisan_v1"}, "tote2": made,
                      "crate": {"authoring_backend": "artisan_v1"}}
    record = _admit(tmp_path / "budget", objects, configurations)
    statuses = {entry["name"]: (entry["status"], entry.get("route")) for entry in record["objects"]}
    assert statuses["blue storage tote"] == (budget.FREE, "library")
    assert statuses["crate"] == (budget.ADMITTED, "generate") and len(ledger.reservations()) == 1


def test_a_parametric_extra_object_that_falls_back_to_artisan_is_reserved_first(tmp_path, ledger):
    """The budget counted a ``parametric`` object free; compile falls back to ARTISAN when the shape does not fit the
    estimated bounds (a cylinder on an oblong footprint), and that object ran ARTISAN with no reservation of its own.
    Now compile's result is read: it is reserved like any paid object."""
    from blueprint_pipeline.website_supplied_mesh import supplied_mesh_configuration
    from tests.test_website_object_library import _configuration
    objects = [_object("tote", "blue tote", "generate"),
               {**_object("can", "paint can", "parametric"), "parametric_shape": "cylinder"}]
    made = supplied_mesh_configuration(_configuration(), synthetic=None, synthetic_authoring=None,
                                       parametric={"target_id": "can", "shape": "cylinder"}, subject_id="can",
                                       sim_min=[0, 0, 0], sim_max=[0.8, 0.3, 0.3], library=None, library_root=None)
    assert made["authoring_backend"] == "artisan_v1" and made["parametric_not_used"]
    record = _admit(tmp_path, objects, {"can": made})
    can = record["objects"][1]
    assert (can["status"], can["route"]) == (budget.ADMITTED, "generate")
    assert [spend["maximum_cost_usd"] for spend in ledger.reservations()] == [12.0]
    assert budget.authoring_allocation(can, None) == 12.0


def test_an_object_compile_has_not_decided_is_never_budgeted(tmp_path, ledger):
    """An extra object with no compile result (or one this cannot read) is refused before anything is asked: it is
    never counted free, so never authored with ARTISAN unreserved; a free entry has no ARTISAN allocation either."""
    objects = [_object("tote", "blue tote", "generate"), _object("bin", "open bin", "parametric")]
    for configurations in ({}, {"bin": {"authoring_backend": "something_else"}}, {"bin": "parametric"}):
        with pytest.raises(ValueError, match=budget.OBJECTS_INVALID):
            _admit(tmp_path, objects, configurations)
    assert ledger.asked == []
    record = _admit(tmp_path, objects)
    assert record["objects"][1]["status"] == budget.FREE
    with pytest.raises(ValueError, match=budget.OBJECTS_INVALID):
        budget.authoring_allocation(record["objects"][1], None)


def test_a_new_plan_does_not_hold_a_second_reservation_for_the_same_object(tmp_path, ledger):
    """The binding included the sealed plan's digest, and nothing released a row: planned again (a new Atlas result,
    a correction) the same carton was asked for a second $12 while the first row stayed held, and with $12 left it was
    refused at the cap and kept as filmed, its first $12 still held for nothing. Now the job and the object key it."""
    objects = OBJECTS[:2]  # the first object and the carton (paid)
    first = _admit(tmp_path, objects)
    assert first["objects"][1]["status"] == budget.ADMITTED
    replanned = _admit(tmp_path, objects, digest="sha256:" + "b" * 64)
    held = [key for key in ledger.rows if key != "scan"]
    assert replanned["objects"][1]["status"] == budget.ADMITTED and len(held) == 1
    assert replanned["build_plan_digest"] == "sha256:" + "b" * 64
    assert len(ledger.reservations()) == 1  # never asked again


def test_a_retained_reservation_is_not_admitted_again_after_the_site_withdrew(tmp_path, ledger, monkeypatch):
    """Privacy: a reservation written before the site withdrew was "admitted again" by the local recheck alone (the
    account's month and the resource class), with no read of the capture's current source, so the several-objects
    build would author the object (ARTISAN, paid) after the withdrawal. Now the source is read first, as
    ``retained_gemini_call`` reads it, and its privacy stop is raised."""
    objects = OBJECTS[:2]
    _admit(tmp_path, objects)

    def withdrawn(**_kwargs):
        raise PrivacyRightsStop("website_control_task-context_http_409:source_revoked")
    monkeypatch.setattr(website_task_context, "load_current_website_task_context", withdrawn)
    asked = len(ledger.asked)
    with pytest.raises(PrivacyRightsStop):
        _admit(tmp_path, objects)
    assert len(ledger.asked) == asked  # nothing asked or settled under the earlier hold
    # The hold stays written for whatever ends the job (withdrawal deletion); this run wrote no new record over it.
    assert json.loads((tmp_path / budget.BUDGET_RECORD).read_text())["objects"][1]["status"] == budget.ADMITTED


def test_a_retained_reservation_for_a_changed_task_context_waits_for_the_fresh_one(tmp_path, ledger, monkeypatch):
    """The website's current context differs from the run's: the hold is not used under the stale one, and the job
    retries with the fresh context (``website_scene_sponsorship_binding_invalid``)."""
    objects = OBJECTS[:2]
    _admit(tmp_path, objects)
    monkeypatch.setattr(website_task_context, "load_current_website_task_context",
                        lambda **_: {**CONTEXT, "context_digest": "sha256:" + "9" * 64})
    with pytest.raises(ValueError, match="website_scene_sponsorship_binding_invalid"):
        _admit(tmp_path, objects)


def test_a_retained_reservation_is_counted_once_at_the_accounts_month(tmp_path, monkeypatch):
    """Control (double counting): with the real account limit, a retained reservation's recheck excludes its own
    journaled grant, so the hold is judged once: $388 + $12 fits $400 on the first run and on the retry."""
    from tests.test_account_monthly_journal import NOV_29, _config, _write_report
    from blueprint_pipeline import account_monthly_spend as account
    value = Ledger()
    monkeypatch.setattr(website_task_context, "preparation_webapp_request", value)
    monkeypatch.setattr(website_task_context, "load_current_website_task_context", lambda **_: dict(CONTEXT))
    monkeypatch.setattr(admission, "_monthly_budget_blockers", lambda _resource_class: [])
    monkeypatch.delenv("BLUEPRINT_ARTISAN_AUTHORING_MAX_COST_USD", raising=False)
    env = _config(tmp_path)
    for name, item in env.items():
        monkeypatch.setenv(name, item)
    monkeypatch.setattr(account, "_clock", lambda: NOV_29)
    _write_report(env, counted="388.00", generated=NOV_29 - 30)
    objects = OBJECTS[:2]
    assert _admit(tmp_path / "b", objects)["objects"][1]["status"] == budget.ADMITTED
    assert _admit(tmp_path / "b", objects)["objects"][1]["status"] == budget.ADMITTED
    journal = Path(env["BLUEPRINT_PAID_SPEND_METER_ROOT"]) / "events" / "2026-11"
    rows = [json.loads(line) for path in journal.glob("*.jsonl") for line in path.read_text().splitlines()]
    assert len([row for row in rows if row["event"] == "admission"]) == 1


def test_the_one_object_parametric_shape_reserves_no_artisan_money_end_to_end(tmp_path, monkeypatch):
    """Control, end to end through compile and the native submission (as the library test does): the plan's carton
    shape for the job's cup authors as supplied geometry, and the attempt's spend carries no ARTISAN allocation."""
    import tests.test_website_native_submission as submission
    from blueprint_pipeline.website_native_submission import materialize_website_submission
    from blueprint_pipeline.website_supplied_mesh import PARAMETRIC_SUBJECT
    monkeypatch.delenv("BLUEPRINT_TASK_OBJECT_AUTHORING_BACKEND", raising=False)
    real = submission.compile_website_scene_preparation

    def compile_with_shape(**args):
        args["removal_manifest"] = {**args["removal_manifest"],
                                    PARAMETRIC_SUBJECT: {"target_id": "cup-1", "shape": "carton"}}
        return real(**args)
    monkeypatch.setattr(submission, "compile_website_scene_preparation", compile_with_shape)
    kwargs, _ = submission.setup(tmp_path, monkeypatch, authoring_backend="artisan_v1")
    materialize_website_submission(**kwargs)
    request = json.loads((kwargs["staging_root"] / "scene_configuration_preparation_request.v1.json").read_text())
    configuration = json.loads(Path(kwargs["task"]["preparation"]["path"]).read_text())["authoring_inputs"][
        "configuration"]
    if configuration.get("authoring_source") != "parametric":
        pytest.skip(f"the fixture's cup is no carton: {configuration.get('parametric_not_used')}")
    assert "replacement_authoring_backend" not in request
    assert request["spend"]["external_service_caps"]["openai"]["maximum_cost_usd"] < 1
