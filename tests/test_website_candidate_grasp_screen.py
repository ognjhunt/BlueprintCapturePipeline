"""Each AI robot's own hand against the task object: a flag on that robot's result, never a stop."""
from __future__ import annotations

from copy import deepcopy

import pytest

from blueprint_pipeline import website_candidate_grasp_screen as grasp
from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
from blueprint_pipeline.beta_policy_screening_publication_flags import row_flags
from blueprint_pipeline.website_candidate_feasibility import scene_check

# capture-447ae4f4's tray, from its compiled task spec's collision box.
TRAY_TASK_SPEC = {"subject_collision_bounds_scoring_frame_m": {
    "minimum": [-0.15668, -0.26056, -0.010372], "maximum": [0.15668, 0.26056, 0.034556]}}
# A crate whose smallest side (12 cm) is wider than an 85 mm hand even before the size error.
CRATE_TASK_SPEC = {"subject_collision_bounds_scoring_frame_m": {
    "minimum": [-0.2, -0.15, 0.0], "maximum": [0.2, 0.15, 0.12]}}


def _screens(task_spec, profile=None):
    return grasp.grasp_screens(catalog=load_default_catalog(), profile=profile or {}, task_spec=task_spec)


def test_the_droid_rows_are_tested_with_their_own_robotiq_hand_and_others_are_not_assessed() -> None:
    screens = _screens(TRAY_TASK_SPEC)
    droid = screens["pi05_droid"]
    assert droid["status"] == "assessed"
    row = droid["grippers"][0]
    assert {key: row[key] for key in ("model", "full_stroke_m", "grip_force_recorded", "sensitivity")} == {
        "model": "robotiq_2f85", "full_stroke_m": 0.085, "grip_force_recorded": False,
        "sensitivity": "hold_unassessed_grip_force_not_declared"}
    # The screen's own test passes on the 4.5 cm side, but a hand from above closes across 31 cm:
    # 447's tray can be held only by pinching its rim.
    assert row["closes_across_object"] is False and row["horizontal_width_upper_m"] == pytest.approx(0.31336 * 1.25)
    assert droid["flags"] == [grasp.FLAG_EDGE]
    assert screens["humanoidarena_pi05_g1_dex3_sonic"] == {
        "schema_version": grasp.SCHEMA_VERSION, "status": "not_assessed", "reason": "no_gripper_recorded", "flags": []}


def test_an_object_wider_than_the_hand_flags_that_robot() -> None:
    droid = _screens(CRATE_TASK_SPEC)["pi05_droid"]
    assert droid["flags"] == [grasp.FLAG_WIDER] and droid["grippers"][0]["sensitivity"] == "blocked_by_estimate"


@pytest.mark.parametrize("force, flag", [(1000.0, []), (40.0, [grasp.FLAG_DEPENDS]), (0.01, [grasp.FLAG_HEAVIER])])
def test_a_recorded_grip_force_is_tested_against_the_weight_estimate(force, flag) -> None:
    hand = {"model": "registered/gripper", "full_stroke_m": 0.5, "grip_force_n": {"upper": force}}
    assert grasp.grasp_screen([0.31, 0.52, 0.045], [hand])["flags"] == flag


def test_a_robot_with_two_hands_is_flagged_only_when_neither_takes_the_object() -> None:
    narrow = {"model": "a/left", "full_stroke_m": 0.05, "grip_force_n": None}
    wide = {"model": "a/right", "full_stroke_m": 0.20, "grip_force_n": None}
    assert grasp.grasp_screen([0.4, 0.3, 0.12], [narrow, wide])["flags"] == []
    assert grasp.grasp_screen([0.4, 0.3, 0.12], [narrow, dict(wide, full_stroke_m=0.06)])["flags"] == [grasp.FLAG_WIDER]


def test_the_flag_lands_on_that_robots_result_and_the_robot_is_still_planned() -> None:
    from tests.test_beta_policy_screening_website_rows import website_plans

    catalog = load_default_catalog()
    screens = _screens(CRATE_TASK_SPEC)
    fresh, _ = website_plans(catalog, capabilities={"candidate_grasp_screen": screens})
    rows = {row["candidate_id"]: row for row in fresh["candidates"]}
    droid = rows["pi05_droid"]
    without, _ = website_plans(catalog)
    assert droid["status"] == {row["candidate_id"]: row for row in without["candidates"]}["pi05_droid"]["status"]
    assert droid["grasp_screen"]["flags"] == [grasp.FLAG_WIDER]
    assert scene_check(droid) == ("flagged", [grasp.FLAG_WIDER])
    assert row_flags(droid, {})["scene_check_flags"] == [grasp.FLAG_WIDER]
    # A row with nothing recorded says nothing.
    g1 = rows.get("humanoidarena_pi05_g1_dex3_sonic")
    if g1 is not None and "grasp_screen" in g1:
        assert g1["grasp_screen"]["status"] == "not_assessed" and "scene_check_flags" not in row_flags(g1, {})


def test_a_malformed_record_is_refused_by_the_plan() -> None:
    from blueprint_pipeline.beta_policy_screening_plan import BetaPolicyScreeningPlanError
    from tests.test_beta_policy_screening_website_rows import website_plans

    screens = deepcopy(_screens(CRATE_TASK_SPEC))
    screens["pi05_droid"]["flags"] = ["made_up_code"]
    with pytest.raises(BetaPolicyScreeningPlanError, match="beta_screening_grasp_screen_invalid"):
        website_plans(load_default_catalog(), capabilities={"candidate_grasp_screen": screens})


@pytest.mark.parametrize("dimensions, flags", [
    ([0.06, 0.06, 0.06], []),                          # a 6 cm box: the hand closes across it
    ([0.31336, 0.52113, 0.04493], [grasp.FLAG_EDGE]),  # 447's tray: only its rim fits the hand
    ([0.40, 0.40, 0.40], [grasp.FLAG_WIDER]),          # a 40 cm ball: wider than the hand, no thin edge
])
def test_the_grip_the_robot_will_use_is_tested(dimensions, flags) -> None:
    robotiq = grasp.candidate_grippers({"interface": {"robot_profile_id": "droid_franka_panda_robotiq_2f85_v1"}}, None)
    record = grasp.grasp_screen(dimensions, robotiq, approach="top_down")
    assert record["flags"] == flags and record["approach"] == "top_down"


def test_the_approach_is_read_from_the_jobs_carry_check(tmp_path) -> None:
    import json

    native = tmp_path / "pipeline" / "website_scene_preparation" / "native"
    native.mkdir(parents=True)
    (native / "capability_verdict.json").write_text(json.dumps({"reach": {"grasp": "top_down"}}))
    assert grasp.recorded_approach(tmp_path) == "top_down"
    assert grasp.recorded_approach(tmp_path / "missing") is None


def test_no_object_box_means_no_record() -> None:
    assert grasp.grasp_screens(catalog=load_default_catalog(), profile={}, task_spec={}) is None


def test_a_failing_grasp_screen_never_stops_planning(monkeypatch) -> None:
    def broken(*_args, **_kwargs):
        raise OSError("robot_spec_catalog unreadable")
    monkeypatch.setattr(grasp, "candidate_grippers", broken)
    screens = _screens(TRAY_TASK_SPEC)
    assert screens["pi05_droid"] == {"schema_version": grasp.SCHEMA_VERSION, "status": "not_assessed",
                                     "reason": "grasp_screen_failed:OSError", "flags": []}
    assert grasp.validate_grasp_screen(screens["pi05_droid"])["flags"] == []


@pytest.mark.parametrize("mode", ["nested", "fuse"])
def test_every_plan_the_planner_builds_with_grasp_records_validates_against_the_plan_schema(tmp_path, monkeypatch, mode):
    """Review 8d (3): the published plan schema accepts each record exactly as the code writes it."""
    import json
    from pathlib import Path

    from jsonschema import Draft202012Validator

    from blueprint_pipeline import website_job_limits
    from tests.test_beta_policy_screening_website_rows import website_plans

    config = tmp_path / "website_job_limits.v1.json"
    config.write_text(json.dumps({"schema_version": website_job_limits.SCHEMA, "mode": mode}), encoding="utf-8")
    monkeypatch.setenv(website_job_limits.CONFIG_ENV, str(config))
    schema = json.loads((Path(__file__).resolve().parents[1]
                         / "docs/schemas/beta_policy_screening_plan.v1.schema.json").read_text())
    validator = Draft202012Validator(schema)
    catalog = load_default_catalog()
    failing = _screens(TRAY_TASK_SPEC)
    failing[next(iter(failing))] = grasp.not_assessed(RuntimeError("x"))
    records = [_screens(TRAY_TASK_SPEC), _screens(CRATE_TASK_SPEC), failing,
               _screens(CRATE_TASK_SPEC, {"screening_candidates": {}}), None]
    built = 0
    for screens in records:
        for remaining in (0, 5, 15, 40):
            capabilities = {"candidate_grasp_screen": screens} if screens else {}
            for plan in website_plans(catalog, remaining=remaining, capabilities=capabilities):
                errors = sorted(error.message[:200] for error in validator.iter_errors(plan))
                assert errors == [], (remaining, errors[:3])
                built += 1
                if screens:
                    assert any("grasp_screen" in row for row in plan["candidates"])
    assert built == 40
    assert {row["status"] for row in failing.values()} == {"assessed", "not_assessed"}
