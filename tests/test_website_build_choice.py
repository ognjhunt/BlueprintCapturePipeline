"""What to build for a job: one model decision from the pipeline's menu, today's rules as the fallback.

No live model calls: the one Gemini call is replaced by its answer, or runs through the real retained receipts with
the reservation and the provider stubbed.
"""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline import website_build_choice as choice
from blueprint_pipeline import website_scene_handoff as handoff
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_website_scene_handoff import ATLAS, atlas_ready, inputs, room_recorder

DESCRIPTION = "Move the box onto the shelf"
DECIDED = "pipeline/website_scene_preparation/build_choice"


@pytest.fixture
def ready(tmp_path, monkeypatch):
    """The capture's Atlas scene is ready (``test_website_scene_handoff.ready_atlas``)."""
    value = atlas_ready(tmp_path)
    monkeypatch.setattr(ATLAS, lambda **_: value)
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry",
                        lambda **kw: {**kw["task_masks"], "source_geometry_digest": "atlas-geometry"})
    return value


def _job(tmp_path, *, family="rigid_pick_and_place", mechanism="", destinations=("shelf",),
         description=DESCRIPTION):
    kwargs = inputs(tmp_path)
    context = kwargs["descriptor"]["metadata"]["site_task_context"]
    context.update(task_family=family, description=description, context_digest="sha256:" + "d" * 64)
    kwargs["clean_plate"]["task_masks"] = {"targets": [
        {"target_id": "box", "articulation_kind": mechanism},
        *({"target_id": key, "target_role": "destination", "placement_relation": "on"} for key in destinations)]}
    write_json(tmp_path / "pipeline/removal.json", {"entries": [
        {"target_id": "box", "task_effect": "manipulated", "semantic_label": "box", "articulation_kind": mechanism,
         "articulated_part": "lid" if mechanism else ""}]})
    return kwargs


def _to_compile(monkeypatch):
    """The object path up to compile: coverage and specs pass through, compile records its arguments and stops."""
    monkeypatch.setattr("blueprint_pipeline.website_target_background_worker.queue_correction",
                        lambda **_: {"status": "ready"})
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
                        lambda **kw: kw["task_masks"])
    monkeypatch.setattr("blueprint_pipeline.website_object_spec_research.attach_object_specs",
                        lambda **kw: kw["task_masks"])
    compiled = []

    def compile_(**kw):
        compiled.append({key: value for key, value in kw.items() if not callable(value)})
        return {"status": "needs_input", "blockers": ["website_task_context_family_invalid"], "digest": "x",
                "thumbnail": None}
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", compile_)
    monkeypatch.setattr("blueprint_pipeline.website_preparation_revision.bind_preparation", lambda value, **_: value)
    return compiled


def _answering(monkeypatch, answer, asked=None):
    asked = [] if asked is None else asked

    def ask(**kw):
        asked.append(kw)
        if isinstance(answer, BaseException):
            raise answer
        return {"answer": answer, "frames": []}
    monkeypatch.setattr(choice, "ask_gemini", ask)
    return asked


def _answer(build, destination=None, quote="Move the box"):
    return {"build": build, "destination_target_id": destination, "job_quote": quote,
            "reasons": "The job moves the box.", "not_built": []}


def _record(tmp_path):
    return json.loads((tmp_path / DECIDED / choice.RECORD).read_text())


def test_the_menu_offers_what_the_pipeline_can_build_for_this_job_and_says_why_not_the_rest(tmp_path):
    kwargs = _job(tmp_path, mechanism="prismatic")
    context = kwargs["descriptor"]["metadata"]["site_task_context"]
    removal = json.loads((tmp_path / "pipeline/removal.json").read_text())
    targets = kwargs["clean_plate"]["task_masks"]["targets"]
    rules = choice.rules_route(task_context=context, removal=removal, targets=targets, flags=[], task_semantics=None,
                               source_video_digest=None, geometry_only=False)
    # Declared pick-and-place, an opening part in the analysis: today's rule ships the room for the mismatch.
    assert rules == {"build": choice.ROOM_ONLY, "destination_target_id": None, "basis": handoff.MISMATCH}
    menu = choice.build_menu(task_context=context, removal=removal, targets=targets, rules=rules)
    assert choice.available_builds(menu) == [choice.RIGID, choice.ARTICULATED, choice.RULE_WITHHELD, choice.ROOM_ONLY]
    pouring = next(row for row in menu["builds"] if row["build"] == "pouring")
    assert pouring["available"] is False and "own registered task" in pouring["why_not"]
    assert menu["objects"][0]["opening_joint"] == "prismatic" and menu["destinations"][0]["target_id"] == "shelf"
    assert menu["menu_digest"] == canonical_digest(menu, digest_field="menu_digest")


def test_a_family_job_may_only_keep_its_family_or_take_the_room_as_filmed(tmp_path):
    kwargs = _job(tmp_path, family="wiping")
    context = kwargs["descriptor"]["metadata"]["site_task_context"]
    removal = json.loads((tmp_path / "pipeline/removal.json").read_text())
    targets = kwargs["clean_plate"]["task_masks"]["targets"]
    rules = choice.rules_route(task_context=context, removal=removal, targets=targets, flags=[], task_semantics=None,
                               source_video_digest=None, geometry_only=False)
    assert rules["build"] == "wiping"
    menu = choice.build_menu(task_context=context, removal=removal, targets=targets, rules=rules)
    assert choice.available_builds(menu) == ["wiping", choice.ROOM_ONLY]


def test_a_valid_menu_choice_is_honoured(tmp_path, monkeypatch, ready):
    """The mismatch rule would ship the room; the model reads the job as the lid it opens: that is built."""
    kwargs = _job(tmp_path, mechanism="prismatic", destinations=(), description="Open the box lid")
    compiled = _to_compile(monkeypatch)
    packaged = room_recorder(monkeypatch)
    _answering(monkeypatch, _answer(choice.ARTICULATED, quote="Open the box lid"))
    handoff.prepare_website_scene_handoff(**kwargs, allow_build_choice=True)
    assert packaged == []
    [call] = compiled
    assert call["build_choice"]["build"] == choice.ARTICULATED and "rule_withheld" not in call
    record = _record(tmp_path)
    assert (record["source"], record["honoured"], record["rules"]["build"]) == ("model", True, choice.ROOM_ONLY)
    assert record["choice"]["reasons"] and record["digest"] == canonical_digest(record, digest_field="digest")


def test_the_room_as_filmed_is_honoured_and_buys_nothing_for_the_object(tmp_path, monkeypatch, ready):
    kwargs = _job(tmp_path)
    compiled = _to_compile(monkeypatch)
    packaged = room_recorder(monkeypatch)
    _answering(monkeypatch, _answer(choice.ROOM_ONLY))
    result = handoff.prepare_website_scene_handoff(**kwargs, allow_build_choice=True)
    assert result["status"] == "room_package_written" and compiled == []
    assert [(call["task_profile"], call["flags"]) for call in packaged] == [("rigid_pick_place", [choice.ROOM_FLAG])]
    assert result["build_choice"]["build"] == choice.ROOM_ONLY and result["build_choice"]["rules_build"] == choice.RIGID


def test_a_chosen_rule_withheld_build_carries_its_own_sealed_reason(tmp_path, monkeypatch, ready):
    kwargs = _job(tmp_path)
    compiled = _to_compile(monkeypatch)
    _answering(monkeypatch, _answer(choice.RULE_WITHHELD, destination="shelf"))
    handoff.prepare_website_scene_handoff(**kwargs, allow_build_choice=True)
    [call] = compiled
    assert call["rule_withheld"] == {"code": choice.RULE_WITHHELD_CODE, "task_id": "website-" + "d" * 20}
    from blueprint_pipeline.website_company_simulation import RULE_WITHHELD_REASONS
    assert choice.RULE_WITHHELD_CODE in RULE_WITHHELD_REASONS


@pytest.mark.parametrize("answer, why", [
    (_answer("pouring"), "build_not_on_menu"),
    (_answer(choice.RIGID, destination="table"), "destination_not_listed"),
    (_answer(choice.ARTICULATED), "build_not_on_menu"),  # no joint type in the analysis
    (_answer(choice.ROOM_ONLY, destination="shelf"), "destination_for_a_build_without_one"),
    (_answer(choice.RIGID, quote="Throw the box away"), "job_quote_not_in_job"),
    ({"build": choice.ROOM_ONLY}, "shape"),
    (None, "shape"),
])
def test_an_invalid_choice_falls_back_to_today_s_rules(tmp_path, monkeypatch, ready, answer, why):
    kwargs = _job(tmp_path, destinations=("shelf", "cart"))
    compiled = _to_compile(monkeypatch)
    _answering(monkeypatch, answer)
    with_choice = handoff.prepare_website_scene_handoff(**kwargs, allow_build_choice=True)
    record = _record(tmp_path)
    assert record["source"] == "rules" and record["fallback"] == f"{choice.INVALID}:{why}"
    assert record["honoured"] is False and record["build"] == choice.RIGID
    assert "build_choice" not in compiled[0] and "build_choice" not in with_choice
    without = handoff.prepare_website_scene_handoff(**kwargs)
    assert with_choice == without and compiled[0] == compiled[1]


def test_a_call_that_fails_for_no_wall_falls_back_and_a_money_stop_stays_a_stop(tmp_path, monkeypatch, ready):
    from blueprint_pipeline.website_gemini_receipts import CREDIT_DEPLETED, CreditDepleted
    kwargs = _job(tmp_path)
    compiled = _to_compile(monkeypatch)
    _answering(monkeypatch, TimeoutError("no answer"))
    handoff.prepare_website_scene_handoff(**kwargs, allow_build_choice=True)
    assert len(compiled) == 1 and "build_choice" not in compiled[0]
    record = _record(tmp_path)
    assert (record["source"], record["fallback_error_type"], record["honoured"]) == ("rules", "TimeoutError", False)
    _answering(monkeypatch, CreditDepleted())
    result = handoff.prepare_website_scene_handoff(**kwargs, allow_build_choice=True)
    assert len(compiled) == 1  # nothing more was prepared
    assert (result["status"], result["blockers"]) == ("awaiting_inputs", [CREDIT_DEPLETED])
    _answering(monkeypatch, ValueError(choice.IN_PROGRESS))  # another run holds the call: wait for it
    result = handoff.prepare_website_scene_handoff(**kwargs, allow_build_choice=True)
    assert result["blockers"] == [choice.IN_PROGRESS] and len(compiled) == 1


@pytest.mark.parametrize("family, mechanism, destinations, removed, answer", [
    # A modelled one-object job: pick and place onto its one destination.
    ("rigid_pick_and_place", "", ("shelf",), None, _answer(choice.RIGID, "shelf")),
    # A job no profile models, one rigid object: built with Blueprint's rule withheld.
    ("unimplemented_task_family", "", ("shelf",), None, _answer(choice.RULE_WITHHELD, "shelf")),
    # The mechanism-mismatch room.
    ("articulated_open_close", "", (), None, _answer(choice.ROOM_ONLY)),
    # A job the analysis found several objects of: one build only, nothing is asked.
    ("rigid_pick_and_place", "", ("shelf",), 2, None),
])
def test_jobs_whose_choice_matches_today_s_rules_are_unchanged(tmp_path, monkeypatch, ready, family,
                                                               mechanism, destinations, removed, answer):
    kwargs = _job(tmp_path, family=family, mechanism=mechanism, destinations=destinations)
    if removed:
        write_json(tmp_path / "pipeline/removal.json", {"entries": [
            {"target_id": f"box-{index}", "task_effect": "manipulated"} for index in range(removed)]})
        kwargs["clean_plate"]["task_masks"]["targets"] += [{"target_id": f"box-{index}"} for index in range(removed)]
    compiled = _to_compile(monkeypatch)
    packaged = room_recorder(monkeypatch)
    asked = _answering(monkeypatch, answer)
    with_choice = handoff.prepare_website_scene_handoff(**kwargs, allow_build_choice=True)
    without = handoff.prepare_website_scene_handoff(**kwargs)
    assert with_choice == without
    assert compiled[::2] == compiled[1::2] and packaged[::2] == packaged[1::2]
    if answer is None:
        assert asked == [] and not (tmp_path / DECIDED / choice.RECORD).exists()
    else:
        assert len(asked) == 1 and _record(tmp_path)["agrees_with_rules"] is True


def test_a_rerun_makes_no_new_call(tmp_path, monkeypatch):
    """The real retained call: the first run reserves and asks once; every rerun reads its receipt."""
    from tests.test_website_gemini_receipts import context, setup_call
    _, calls = setup_call(tmp_path, monkeypatch)
    monkeypatch.setattr("blueprint_pipeline.clean_plate_removal_analysis_gemini._api_key", lambda: ("key", "env"))
    task_context = context()
    removal = {"entries": [{"target_id": "box", "task_effect": "manipulated", "semantic_label": "box"}]}
    targets = [{"target_id": "box"}, {"target_id": "shelf", "target_role": "destination", "placement_relation": "on"}]
    rules = {"build": choice.RIGID, "destination_target_id": "shelf", "basis": "job_profile"}
    import blueprint_pipeline.website_gemini_receipts as receipts
    real = receipts.retained_gemini_call

    def retained(**kw):
        kw["preflight"] = lambda: None

        def invoke():
            calls.append(("invoke", None))
            return {"answer": _answer(choice.RULE_WITHHELD, "shelf"), "frames": []}
        kw["invoke"] = invoke
        return real(**kw)
    monkeypatch.setattr(receipts, "retained_gemini_call", retained)
    records = [choice.decide_build(task_context=task_context, removal=removal, targets=targets, rules=rules,
                                   output_root=tmp_path / "decision") for _ in range(3)]
    assert [row[0] for row in calls] == ["reserve", "invoke"]
    assert records[0] == records[1] == records[2]
    assert records[0]["honoured"] is True and records[0]["build"] == choice.RULE_WITHHELD
    assert records[0]["maximum_cost_usd"] <= 0.05


def test_compile_builds_the_chosen_mechanism_and_destination_and_nothing_else_changes(tmp_path):
    from tests.test_website_task_preparation import _arguments, preparation
    arguments = _arguments(tmp_path)
    context, masks = arguments["task_context"], arguments["task_masks"]
    context["task_family"] = "rigid_pick_and_place"
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    next(row for row in masks["targets"] if row["target_id"] == "cup-1")["articulation_kind"] = "prismatic"
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    held = preparation.compile_website_scene_preparation(**arguments)
    assert held["blockers"] == [handoff.MISMATCH]  # today's rule
    built = preparation.compile_website_scene_preparation(
        **arguments, build_choice={"build": choice.RIGID, "destination_target_id": "tray-1"})
    assert built["status"] == "intake_ready" and built["destination"]["visible_label"] == "tray-1"
    assert "native_task_preparation_gap" not in built and built["intake_request"] is not None
    guessed = preparation.compile_website_scene_preparation(
        **arguments, build_choice={"build": choice.RIGID, "destination_target_id": None})
    assert (guessed.get("destination") or {}).get("visible_label") != "tray-1"
    with pytest.raises(ValueError, match="website_build_choice_binding_invalid"):
        preparation.compile_website_scene_preparation(
            **arguments, build_choice={"build": choice.ARTICULATED, "destination_target_id": "tray-1"})
    next(row for row in masks["targets"] if row["target_id"] == "cup-1")["articulation_kind"] = ""
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    with pytest.raises(ValueError, match="website_build_choice_binding_invalid"):  # no joint type to open
        preparation.compile_website_scene_preparation(
            **arguments, build_choice={"build": choice.ARTICULATED, "destination_target_id": None})
