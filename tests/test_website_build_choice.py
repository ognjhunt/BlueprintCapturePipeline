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
    assert pouring["available"] is False and "needs no choice" in pouring["why_not"]
    assert menu["objects"][0]["opening_joint"] == "prismatic" and menu["destinations"][0]["target_id"] == "shelf"
    assert menu["menu_digest"] == canonical_digest(menu, digest_field="menu_digest")


def _rules(tmp_path, kwargs, **extra):
    removal = json.loads((tmp_path / "pipeline/removal.json").read_text())
    return choice.rules_route(task_context=kwargs["descriptor"]["metadata"]["site_task_context"], removal=removal,
                              targets=kwargs["clean_plate"]["task_masks"]["targets"], flags=extra.pop("flags", []),
                              task_semantics=None, source_video_digest=None,
                              geometry_only=extra.pop("geometry_only", False))


@pytest.mark.parametrize("family", ["wiping", "navigation"])
def test_family_and_navigation_jobs_never_wait_on_the_model(tmp_path, monkeypatch, ready, family):
    """Blocker 2: no judgement is left for a job whose words name its own task, so no call is made at all."""
    kwargs = _job(tmp_path, family=family)
    assert _rules(tmp_path, kwargs) is None
    monkeypatch.setattr(choice, "ask_gemini", lambda **_: pytest.fail("a family or navigation job asked the model"))
    packaged = room_recorder(monkeypatch)
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", lambda **_: pytest.fail("compiled"))
    monkeypatch.setattr("blueprint_pipeline.website_target_background_worker.queue_correction",
                        lambda **_: {"status": "ready"})
    with_choice = handoff.prepare_website_scene_handoff(**kwargs, allow_build_choice=True)
    without = handoff.prepare_website_scene_handoff(**kwargs)
    assert with_choice == without and packaged[0] == packaged[1]
    assert not (tmp_path / DECIDED / choice.RECORD).exists()


def test_a_designed_object_or_the_geometry_only_route_never_reaches_the_model(tmp_path):
    """Blocker 1 (and geometry-only): a designed object's build is the rules', with no call to go wrong."""
    from tests.test_website_task_preparation import SYNTHETIC_BALL
    kwargs = _job(tmp_path, family="articulated_open_close", destinations=("shelf", "cart"))
    assert _rules(tmp_path, kwargs, geometry_only=True) is None
    write_json(tmp_path / "pipeline/removal.json", {"entries": [], "synthetic_subject": dict(SYNTHETIC_BALL)})
    assert _rules(tmp_path, kwargs) is None


def test_a_valid_menu_choice_is_honoured(tmp_path, monkeypatch, ready):
    """The mismatch rule would ship the room; the model reads the job as the lid it opens: that is built."""
    kwargs = _job(tmp_path, mechanism="prismatic", destinations=(), description="Open the box lid")
    compiled = _to_compile(monkeypatch)
    packaged = room_recorder(monkeypatch)
    _answering(monkeypatch, _answer(choice.ARTICULATED, quote="Open the box lid"))
    handoff.prepare_website_scene_handoff(**kwargs, allow_build_choice=True)
    # The room-only package the mismatch rule would have shipped is not written. This fixture's compile stops on a
    # blocker that is no wall, so the built room's floor is written flagged with it (owner, 2026-11-23).
    assert [call["flags"] for call in packaged] == [["object_not_built", "website_task_context_family_invalid"]]
    assert all(choice.ROOM_FLAG not in call["flags"] for call in packaged)
    [call] = compiled
    assert call["build_choice"]["build"] == choice.ARTICULATED and "rule_withheld" not in call
    record = _record(tmp_path)
    assert (record["source"], record["honoured"], record["rules_prediction"]["build"]) == (
        "model", True, choice.ROOM_ONLY)
    assert record["choice"]["reasons"] and record["digest"] == canonical_digest(record, digest_field="digest")
    assert record["shipped"]["model_choice_built"] is True


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
    (_answer(choice.RIGID, quote="the box"), "job_quote_too_short"),
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
    assert record["honoured"] is False and record["rules_prediction"]["build"] == choice.RIGID
    assert "build_choice" not in compiled[0] and "build_choice" not in with_choice
    without = handoff.prepare_website_scene_handoff(**kwargs)
    assert with_choice == without and compiled[0] == compiled[1]


def test_a_failed_or_unfunded_call_falls_back_to_the_rules_and_only_privacy_or_a_held_call_stops(
        tmp_path, monkeypatch, ready):
    """Blocker 2: a call refused for money leaves the rules, which spend nothing to decide; the job goes on."""
    from blueprint_pipeline.hard_stops import MoneyStop, PrivacyRightsStop
    from blueprint_pipeline.website_gemini_receipts import CreditDepleted
    kwargs = _job(tmp_path)
    compiled = _to_compile(monkeypatch)
    for error, kind in ((TimeoutError("no answer"), "TimeoutError"), (CreditDepleted(), "CreditDepleted"),
                        (MoneyStop("website_preparation_spend_receipt_invalid"), "MoneyStop")):
        _answering(monkeypatch, error)
        handoff.prepare_website_scene_handoff(**kwargs, allow_build_choice=True)
        record = _record(tmp_path)
        assert (record["source"], record["fallback_error_type"], record["honoured"]) == ("rules", kind, False)
        assert "build_choice" not in compiled[-1]
    assert len(compiled) == 3  # each run went on to compile under the rules
    _answering(monkeypatch, ValueError(choice.IN_PROGRESS))  # another run holds the call: wait for it
    result = handoff.prepare_website_scene_handoff(**kwargs, allow_build_choice=True)
    assert result["blockers"] == [choice.IN_PROGRESS] and len(compiled) == 3
    _answering(monkeypatch, PrivacyRightsStop("website_preparation_spend_receipt_invalid"))
    result = handoff.prepare_website_scene_handoff(**kwargs, allow_build_choice=True)
    assert result["status"] == "awaiting_inputs" and len(compiled) == 3


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
        assert len(asked) == 1 and _record(tmp_path)["agrees_with_rules_prediction"] is True


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
    assert records[0]["honoured"] is True and records[0]["choice"]["build"] == choice.RULE_WITHHELD
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


def _retained(monkeypatch, calls, answer):
    """The real retained receipts with the reservation stubbed; ``answer`` is what the provider would say."""
    import blueprint_pipeline.website_gemini_receipts as receipts
    real = receipts.retained_gemini_call

    def retained(**kw):
        preflight = kw["preflight"]

        def checked():
            try:
                preflight()
            except Exception as exc:  # noqa: BLE001 - no key or client here; only the frames check is under test
                if str(exc) == choice.FRAMES_UNREADABLE:
                    raise
        kw["preflight"] = checked
        kw["invoke"] = lambda: calls.append(("invoke", None)) or {"answer": answer(), "frames": []}
        return real(**kw)
    monkeypatch.setattr(receipts, "retained_gemini_call", retained)
    monkeypatch.setattr("blueprint_pipeline.clean_plate_removal_analysis_gemini._api_key", lambda: ("key", "env"))


def test_a_rerun_whose_source_video_is_gone_reuses_the_stored_answer(tmp_path, monkeypatch):
    from tests.test_website_gemini_receipts import context, setup_call
    _, calls = setup_call(tmp_path, monkeypatch)
    _retained(monkeypatch, calls, lambda: _answer(choice.RULE_WITHHELD, "shelf"))
    decoded = []
    monkeypatch.setattr(choice, "decode_frames", lambda video, times, root: decoded.append(times) or [
        {"timestamp_seconds": t, "path": str(video), "sha256": "sha256:" + "0" * 64} for t in times])
    removal = {"entries": [{"target_id": "box", "task_effect": "manipulated",
                            "spatial_evidence": [{"timestamp_seconds": 2.5, "box_xywh_normalized": [0, 0, 1, 1]}]}]}
    targets = [{"target_id": "box"}, {"target_id": "shelf", "target_role": "destination"}]
    rules = {"build": choice.RIGID, "destination_target_id": "shelf", "basis": "job_profile"}
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    args = dict(task_context=context(), removal=removal, targets=targets, rules=rules, output_root=tmp_path / "d")
    first = choice.decide_build(**args, source_video=video)
    again = choice.decide_build(**args, source_video=None)  # the video was retired: the same answer, no call
    assert [row[0] for row in calls] == ["reserve", "invoke"] and decoded == [[2.5]]
    assert first["choice"] == again["choice"] and again["request_binding_digest"] == first["request_binding_digest"]
    other = choice.decide_build(**{**args, "output_root": tmp_path / "e"}, source_video=None)  # nothing stored
    assert other["fallback"] == choice.FRAMES_UNREADABLE and len(calls) == 2  # nothing reserved or sent


def test_a_choice_the_build_cannot_satisfy_falls_back_now_and_is_never_served_again(tmp_path, monkeypatch, ready):
    """Blocker 1: compile refuses the chosen build; this run goes on under the rules, never held, the record says
    what shipped, and the next run asks again with that build off the menu (a new request, not the stored answer)."""
    kwargs = _job(tmp_path, mechanism="prismatic", destinations=(), description="Open the box lid all the way")
    compiled = _to_compile(monkeypatch)
    inner = handoff.compile_website_scene_preparation

    def refusing(**kw):
        if "build_choice" in kw:
            raise ValueError(choice.BINDING_INVALID)
        return inner(**kw)
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", refusing)
    packaged = room_recorder(monkeypatch)
    asked = _answering(monkeypatch, _answer(choice.ARTICULATED, quote="Open the box lid"))
    result = handoff.prepare_website_scene_handoff(**kwargs, allow_build_choice=True)
    assert result == handoff.prepare_website_scene_handoff(**kwargs)  # the rules' outcome: the mismatch room
    assert result["status"] == "room_package_written" and [call["flags"] for call in packaged][0] == [handoff.MISMATCH]
    record = _record(tmp_path)
    assert (record["rejected"], record["honoured"]) == (choice.BINDING_INVALID, False)
    assert record["shipped"]["status"] == "room_package_written" and record["shipped"]["model_choice_built"] is False
    [rejected] = choice.read_rejected(tmp_path / DECIDED)
    assert rejected["build"] == choice.ARTICULATED
    handoff.prepare_website_scene_handoff(**kwargs, allow_build_choice=True)
    assert len(asked) == 2 and asked[1]["binding"] != asked[0]["binding"]  # decided again, the build off the menu
    assert '"build": "articulated_open_close"' in asked[1]["binding"]["prompt"]
    record = _record(tmp_path)
    assert record["fallback"] == f"{choice.INVALID}:build_not_on_menu" and compiled == []
    assert "an earlier choice of it could not be built" in asked[1]["binding"]["prompt"]


def test_a_soft_object_is_never_offered_as_a_rigid_body(tmp_path):
    """Blocker 3: the analysis's reading of the object as soft is a fact, like its joint type."""
    kwargs = _job(tmp_path)
    removal = json.loads((tmp_path / "pipeline/removal.json").read_text())
    removal["entries"][0].update(rigidity="deformable", rigidity_reason="it drapes")
    write_json(tmp_path / "pipeline/removal.json", removal)
    rules = _rules(tmp_path, kwargs)
    assert rules["build"] == choice.ROOM_ONLY  # today's soft-object room
    context = kwargs["descriptor"]["metadata"]["site_task_context"]
    menu = choice.build_menu(task_context=context, removal=removal,
                             targets=kwargs["clean_plate"]["task_masks"]["targets"], rules=rules)
    assert choice.available_builds(menu) == [choice.ROOM_ONLY]
    assert "soft" in next(row for row in menu["builds"] if row["build"] == choice.RIGID)["why_not"]
    assert choice.decide_build(task_context=context, removal=removal, rules=rules, output_root=tmp_path / "d",
                               targets=kwargs["clean_plate"]["task_masks"]["targets"],
                               ask=lambda **_: pytest.fail("a menu of one build asked the model")) is None


def test_the_room_chosen_for_a_job_with_no_runtime_keeps_its_flag(tmp_path, monkeypatch, ready):
    kwargs = _job(tmp_path)
    _to_compile(monkeypatch)
    packaged = room_recorder(monkeypatch)
    monkeypatch.setattr("blueprint_pipeline.website_early_task_hold.task_family_runtime_gap", lambda *_, **__: {})
    _answering(monkeypatch, _answer(choice.ROOM_ONLY))
    handoff.prepare_website_scene_handoff(**kwargs, allow_build_choice=True)
    assert packaged[0]["flags"] == [choice.ROOM_FLAG, handoff.RUNTIME_GAP]


# Owner direction 2026-11-23: the success rule from the site's own words, not one template for every job.
def _stated(**fields):
    return {name: {"value": value, "quote": quote} for name, (value, quote) in fields.items()}


def test_the_site_s_words_set_the_numbers_of_the_chosen_rule_each_quoted_and_held_to_its_range(tmp_path):
    context = {"description": "Slide the sheet tray onto the rack within 3 cm of the back stop, it can take up to "
                              "a minute", "success_criteria": {"successDefinition": "Open the oven door fully"}}
    kept, dropped = choice.validate_success_rule(_stated(
        maximum_final_planar_target_error_m=(0.03, "within 3 cm of the back stop"),
        maximum_episode_seconds=(60, "it can take up to a minute"),
        minimum_lift_m=(0.0, "Slide the sheet tray onto the rack")), build=choice.RIGID, task_context=context)
    assert kept == {"maximum_final_planar_target_error_m": {"value": 0.03, "quote": "within 3 cm of the back stop"},
                    "maximum_episode_seconds": {"value": 60.0, "quote": "it can take up to a minute"}}
    # A lift is never lowered below the scripted run's own lift: the slide is still scored as a carry.
    assert dropped == {"minimum_lift_m": "out_of_range"}
    kept, dropped = choice.validate_success_rule(_stated(
        minimum_opening_fraction_of_estimated_stroke=(0.95, "Open the oven door fully"),
        minimum_hold_seconds=(2.5, "oven door"), maximum_episode_seconds=(45.5, "up to a minute"),
        closing=(1.0, "Open the oven door fully")), build=choice.ARTICULATED, task_context=context)
    assert kept == {"minimum_opening_fraction_of_estimated_stroke": {"value": 0.95, "quote": "Open the oven door fully"}}
    assert dropped == {"minimum_hold_seconds": "quote_too_short", "maximum_episode_seconds": "not_whole_seconds",
                       "closing": "not_a_field_of_this_rule"}
    assert choice.validate_success_rule(_stated(minimum_lift_m=(0.1, "lift it over the rim")), build=choice.RIGID,
                                        task_context=context) == ({}, {"minimum_lift_m": "quote_not_in_job"})
    # A build with no rule of Blueprint's has no numbers to set.
    assert choice.validate_success_rule({"minimum_lift_m": None}, build=choice.RULE_WITHHELD,
                                        task_context=context) == ({}, {})
    assert choice.validate_success_rule("fully", build=choice.RIGID, task_context=context) == (
        {}, {"success_rule": "shape"})


def test_a_bad_number_never_costs_the_build_choice_and_an_answer_without_numbers_still_counts(tmp_path, monkeypatch,
                                                                                            ready):
    kwargs = _job(tmp_path)
    compiled = _to_compile(monkeypatch)
    _answering(monkeypatch, {**_answer(choice.RIGID, "shelf"), "success_rule": _stated(
        maximum_final_planar_target_error_m=(0.03, "Move the box onto the shelf"), minimum_lift_m=(9.0, "Move the box"))})
    handoff.prepare_website_scene_handoff(**kwargs, allow_build_choice=True)
    record = _record(tmp_path)
    # The model agreed with the rules' build: nothing is honoured, but the numbers its words state still reach compile.
    assert (record["source"], record["honoured"]) == ("model", False)
    assert record["choice"]["success_rule_dropped"] == {"minimum_lift_m": "out_of_range"}
    [call] = compiled
    assert call["success_rule"]["build"] == choice.RIGID and "build_choice" not in call
    assert call["success_rule"]["fields"] == {"maximum_final_planar_target_error_m": {
        "value": 0.03, "quote": "Move the box onto the shelf"}}
    _answering(monkeypatch, _answer(choice.RIGID, "shelf"))  # the earlier prompt's answer shape
    (tmp_path / DECIDED / "receipts").exists()
    assert choice.validate_choice(_answer(choice.RIGID, "shelf"), menu=choice.build_menu(
        task_context=kwargs["descriptor"]["metadata"]["site_task_context"],
        removal=json.loads((tmp_path / "pipeline/removal.json").read_text()),
        targets=kwargs["clean_plate"]["task_masks"]["targets"], rules=_rules(tmp_path, kwargs)),
        task_context=kwargs["descriptor"]["metadata"]["site_task_context"])["success_rule"] == {}


def test_the_menu_tells_the_model_each_number_s_default_and_range_for_the_builds_it_offers(tmp_path):
    kwargs = _job(tmp_path)
    removal = json.loads((tmp_path / "pipeline/removal.json").read_text())
    menu = choice.build_menu(task_context=kwargs["descriptor"]["metadata"]["site_task_context"], removal=removal,
                             targets=kwargs["clean_plate"]["task_masks"]["targets"], rules=_rules(tmp_path, kwargs))
    from blueprint_pipeline.website_task_preparation import SUCCESS
    assert list(menu["success_rule_fields"]) == [choice.RIGID]  # no joint type: no opening rule on offer
    lift = menu["success_rule_fields"][choice.RIGID]["minimum_lift_m"]
    assert (lift["default"], lift["minimum"]) == (SUCCESS["minimum_lift_m"], SUCCESS["minimum_lift_m"])
    binding, _ = choice.request_binding(menu=menu, times=[], source_video_digest=None)
    assert "success_rule_fields" in binding["prompt"] and "production target" in binding["prompt"]


def test_compile_scores_the_numbers_the_site_s_words_state_and_records_each_number_s_basis(tmp_path):
    from blueprint_pipeline.website_task_preparation import ARTICULATED_SUCCESS, SUCCESS

    def _rule_success(**kw):
        return choice.success_with_site_numbers(**kw, defaults={"rigid": SUCCESS, "opening": ARTICULATED_SUCCESS})
    destination = {"radius_m": 0.08}
    stated = {"build": choice.RIGID, "fields": {
        "maximum_final_planar_target_error_m": {"value": 0.03, "quote": "within 3 cm"},
        "maximum_episode_seconds": {"value": 60.0, "quote": "up to a minute"}}}
    success, basis = _rule_success(articulated=False, destination=destination, success_rule=stated, rule_withheld=None)
    assert success == {**SUCCESS, "maximum_final_planar_target_error_m": 0.03, "maximum_episode_seconds": 60}
    assert basis["applied"] is True and basis["fields"]["minimum_lift_m"] == {"value": 0.05, "basis": "assumed"}
    assert basis["fields"]["maximum_final_planar_target_error_m"]["basis"] == choice.SUCCESS_RULE_SITE
    # A looser tolerance than the goal area is held to it: the object still ends on its destination.
    loose = {"build": choice.RIGID, "fields": {"maximum_final_planar_target_error_m": {"value": 0.3, "quote": "q"}}}
    success, basis = _rule_success(articulated=False, destination=destination, success_rule=loose, rule_withheld=None)
    assert success["maximum_final_planar_target_error_m"] == 0.08
    assert basis["fields"]["maximum_final_planar_target_error_m"]["held_to"] == "destination_area"
    # No reading of the site's words, a rule withheld, or another build: today's numbers, and nothing new recorded.
    assert _rule_success(articulated=False, destination=destination, success_rule=None, rule_withheld=None) == (
        {**SUCCESS, "maximum_final_planar_target_error_m": 0.08}, None)
    withheld, basis = _rule_success(articulated=False, destination=None, success_rule=stated,
                                    rule_withheld={"code": "x"})
    assert withheld == SUCCESS and basis["applied"] is False
    opening, basis = _rule_success(articulated=True, destination=None, success_rule=stated, rule_withheld=None)
    assert basis["applied"] is False and "minimum_opening_fraction_of_estimated_stroke" in basis["fields"]
    # A value the second check refuses sets nothing, recorded.
    bad = {"build": choice.RIGID, "fields": {"minimum_lift_m": {"value": 0.01, "quote": "q"}}}
    assert _rule_success(articulated=False, destination=None, success_rule=bad, rule_withheld=None)[1]["refused"]


def test_the_company_rule_says_which_numbers_the_site_stated_and_never_quotes_the_site(tmp_path):
    from blueprint_pipeline.website_company_simulation import success_basis
    preparation = tmp_path / "pipeline/website_scene_preparation/preparation.json"
    write_json(preparation, {"success_rule_basis": {"build": choice.RIGID, "applied": True, "fields": {
        "minimum_lift_m": {"value": 0.05, "basis": "assumed"},
        "maximum_episode_seconds": {"value": 60, "basis": choice.SUCCESS_RULE_SITE, "quote": "Acme Bakery, a minute"},
        "maximum_final_planar_target_error_m": {"value": 0.08, "basis": "destination_area"}}}})
    assert success_basis(tmp_path) == {"minimum_lift_m": "assumed", "maximum_episode_seconds": "site_stated",
                                       "maximum_final_planar_target_error_m": "destination_area"}
    write_json(preparation, {})
    assert success_basis(tmp_path) is None
