# Covers (for impacted-test selection):
#   src/blueprint_pipeline/deterministic_refusal.py
#   src/blueprint_pipeline/atlas_scene_estimates.py
#   src/blueprint_pipeline/atlas_scene_planner.py
#   src/blueprint_pipeline/atlas_video_inputs.py
#   src/blueprint_pipeline/website_task_preparation.py
#   src/blueprint_pipeline/beta_policy_screening_plan.py
#   src/blueprint_pipeline/beta_policy_screening_publication_flags.py
#   src/blueprint_pipeline/beta_policy_screening_result.py
"""Reconstruction refusals that continue on a labelled estimate, and those the same inputs repeat (2026-11-10).

An estimate is named once (``atlas_scene_estimates``) and reaches the site's result as a flag in
publication v3's ``scene_feasibility``; a refusal decided only by retained inputs and code is a
``DeterministicRefusal`` and is never re-run on a timer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator, FormatChecker

from blueprint_pipeline import atlas_scene_estimates as estimates
from blueprint_pipeline import atlas_video_inputs as video
from blueprint_pipeline import deterministic_refusal as deterministic
from blueprint_pipeline.beta_policy_screening_plan import BetaPolicyScreeningPlanError, validate_screening_plan
from blueprint_pipeline.beta_policy_screening_publication_flags import scene_feasibility
from blueprint_pipeline.beta_policy_screening_result import screening_publication
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

ROOT = Path(__file__).parents[1]


# --- deterministic refusals ----------------------------------------------------------------------

def test_a_deterministic_refusal_is_its_code_and_keys_its_inputs_and_step_revision(monkeypatch):
    from blueprint_pipeline import atlas_mesh_registration
    error = deterministic.DeterministicRefusal("atlas_mesh_not_registered_to_splat", step="atlas_mesh_registration",
                                               inputs={"source_glb_sha256": "sha256:" + "1" * 64})
    assert str(error) == "atlas_mesh_not_registered_to_splat" and isinstance(error, ValueError)
    record = error.record()
    assert deterministic.unchanged(record)
    assert not deterministic.unchanged({**record, "inputs": {"source_glb_sha256": "sha256:" + "2" * 64}})
    monkeypatch.setattr(atlas_mesh_registration, "REGISTRATION_REVISION", "atlas_mesh_registration.r2")
    assert not deterministic.unchanged(record)


@pytest.mark.parametrize("record", [None, {}, {"schema_version": deterministic.SCHEMA, "step": "unknown_step"},
                                    "sha256:" + "0" * 64])
def test_a_record_this_release_cannot_read_never_holds_a_job_back(record):
    assert deterministic.unchanged(record) is False


def test_a_deterministic_refusal_is_found_behind_the_errors_raised_from_it():
    refusal = deterministic.DeterministicRefusal("website_front_normal_undetermined", step="website_front_normal",
                                                 inputs={"geometry_sha256": "sha256:" + "3" * 64})
    try:
        try:
            raise refusal
        except ValueError as exc:
            raise RuntimeError("handoff failed") from exc
    except RuntimeError as outer:
        assert deterministic.find(outer) is refusal
    assert deterministic.find(ValueError("atlas_mesh_not_registered_to_splat")) is None


def test_an_object_filmed_only_from_above_asks_for_its_front_and_is_not_retried_on_a_timer():
    from blueprint_pipeline.website_task_preparation import body_front_normal
    with pytest.raises(deterministic.DeterministicRefusal, match="website_front_normal_undetermined") as raised:
        body_front_normal({"front_normal": [0.0, 0.0, 1.0]}, np.eye(3))
    assert raised.value.step == "website_front_normal" and deterministic.unchanged(raised.value.record())


# --- the scene review's citations ----------------------------------------------------------------

def _citing_review(tmp_path):
    from tests.test_atlas_scene_planner import fixture
    _, survey, proposal, policy = fixture(tmp_path)
    proposal["subject_origin"] = "no_manipulated_subject"
    proposal["coverage_requirements"][0]["role"] = "task_region"
    proposal["coverage_requirements"][0]["required_frame_ids"] = ["1"]
    proposal["coverage_requirements"][1]["required_frame_ids"] = ["1"]
    proposal["anchor_frame_id"] = "1"
    proposal["unsafe_frame_ids"] = ["0"]
    proposal["automation"] = {"scale_references": [], "objects": [
        {"target_id": "floor", "label": "floor", "segmentation_prompt": "floor", "disposition": "keep",
         "regions": [{"frame_id": "0", "box_xywh_normalized": [0, 0, 1, 1]},
                     {"frame_id": "1", "box_xywh_normalized": [0, 0, 1, 1]}]}],
        "removal_instruction": "", "mask_margin_fraction": .01}
    return survey, proposal, policy


def test_a_review_citing_an_unsafe_frame_beside_a_safe_one_drops_that_citation_and_flags_it(tmp_path):
    # capture-5b067498 (2026-11-06) failed the job on such a citation; it now continues, flagged.
    from blueprint_pipeline import atlas_scene_planner as planner
    survey, proposal, policy = _citing_review(tmp_path)
    review = planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=4)
    assert review["automation_citations_dropped"] == ["0"]
    assert [r["frame_id"] for r in review["automation"]["objects"][0]["regions"]] == ["1"]
    rows = estimates.scene_estimates(levelling={}, registration={}, review=review)
    assert [row["code"] for row in rows] == [estimates.CITATIONS]


def test_a_review_whose_object_rests_only_on_unsafe_frames_refuses_once_not_on_a_timer(tmp_path):
    from blueprint_pipeline import atlas_scene_planner as planner
    survey, proposal, policy = _citing_review(tmp_path)
    proposal["automation"]["objects"][0]["regions"] = proposal["automation"]["objects"][0]["regions"][:1]
    with pytest.raises(deterministic.DeterministicRefusal, match="atlas_automation_frames_not_selected") as raised:
        planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=4)
    assert raised.value.inputs == {"proposal_digest": canonical_digest(proposal), "survey_digest": survey["digest"]}


# --- the frame budget ----------------------------------------------------------------------------

def test_a_surrounding_view_past_the_frame_budget_is_left_out_and_reported_uncaptured(monkeypatch):
    frames = [{"frame_id": str(i), "sharpness": 50, "clipped_fraction": 0} for i in range(6)]
    monkeypatch.setattr(video, "_overlap", lambda a, b: (100, 25) if abs(a - b) == 1 else (0, 0))
    requirements = [{"role": "task_object", "label": "dishwasher", "required_frame_ids": ["2"]},
                    {"role": "surrounding_scene", "label": "counter", "required_frame_ids": ["1"]},
                    {"role": "surrounding_scene", "label": "far wall", "required_frame_ids": ["5"]}]
    # Before 2026-11-10: atlas_coverage_exceeds_frame_budget, which the same video repeats.
    selected, diagnostics = video.select_coverage_views(frames=frames, features=list(range(6)), maximum_frames=3,
                                                        requirements=requirements, anchor_id="2")
    assert selected == [1, 2]
    left_out = [row for row in diagnostics if row.get("decision") == video.OVER_FRAME_BUDGET]
    assert [row["frame_id"] for row in left_out] == ["5"] and video.OVER_FRAME_BUDGET in video.SURROUNDING_GAP_DECISIONS
    rows = estimates.scene_estimates(levelling={}, registration={}, selection={"diagnostics": diagnostics})
    assert rows == [{"code": estimates.SURROUNDING, "source": "view_selection", "evidence": {"frame_ids": ["5"]}}]


def test_a_working_area_view_past_the_frame_budget_still_refuses(monkeypatch):
    frames = [{"frame_id": str(i), "sharpness": 50, "clipped_fraction": 0} for i in range(6)]
    monkeypatch.setattr(video, "_overlap", lambda a, b: (100, 25) if abs(a - b) == 1 else (0, 0))
    requirements = [{"role": "task_object", "label": "dishwasher", "required_frame_ids": ["0", "5"]},
                    {"role": "surrounding_scene", "label": "counter", "required_frame_ids": ["1"]}]
    with pytest.raises(ValueError, match="atlas_coverage_exceeds_frame_budget"):
        video.select_coverage_views(frames=frames, features=list(range(6)), maximum_frames=3,
                                    requirements=requirements, anchor_id="0")


# --- from the scene's records to the site's result -----------------------------------------------

def test_each_estimate_is_named_once_and_a_base_scene_from_before_keeps_its_floor_flag():
    levelling = {"level_estimate": {"code": "scene_estimate:level"}, "appearance_estimate": {"code": "x"},
                 "fallback_floor": {"claim": "development_only_estimate"}}
    registration = {"registration_estimate": {"median_residual": 0.2}, "support_estimate": {"basis": "b"}}
    rows = estimates.scene_estimates(levelling=levelling, registration=registration)
    assert [row["code"] for row in rows] == [estimates.LEVEL, estimates.APPEARANCE, estimates.FLOOR,
                                            estimates.COLLIDER, estimates.COLLIDER]
    assert estimates.base_scene_flags({estimates.FIELD: rows}) == sorted(
        {estimates.LEVEL, estimates.APPEARANCE, estimates.FLOOR, estimates.COLLIDER})
    # A base scene written before ``scene_estimates`` existed: its flat floor stand-in is still named.
    assert estimates.base_scene_flags({"levelling": {"walked_floor": {}}}) == [estimates.FLOOR]
    assert estimates.base_scene_flags({"levelling": {"up_basis": "observed_floor"}}) == []


def test_the_capture_s_base_scene_gives_its_flags_and_an_unreadable_one_gives_none(tmp_path):
    root = tmp_path / "pipeline/website_scene_preparation"
    root.mkdir(parents=True)
    (root / "base_scene.json").write_text(json.dumps({estimates.FIELD: [{"code": estimates.LEVEL}]}))
    assert estimates.capture_scene_flags(tmp_path) == [estimates.LEVEL]
    (root / "base_scene.json").write_text("{not json")
    assert estimates.capture_scene_flags(tmp_path) == []


def test_a_plan_carries_the_estimate_flags_to_the_v3_scene_feasibility_and_one_without_publishes_as_before():
    from tests.test_beta_policy_screening_coordinator import plan as make_plan, seal
    before = validate_screening_plan(make_plan())
    flags = [estimates.COLLIDER, estimates.LEVEL]
    value = validate_screening_plan(seal({**make_plan(), "scene_estimate_flags": flags}, "plan_digest"))
    state = {"revision": 1, "updated_at": "2026-11-10T00:00:00Z",
             "candidates": {row["candidate_id"]: {"status": "pending", "blockers": []} for row in value["candidates"]}}
    budget = {"currency": "USD", "site_limit_usd": "30.00", "planned_usd": "0.00", "spent_usd": "0.00"}
    publication = screening_publication(plan=value, state=state, budget=budget)
    assert publication["scene_feasibility"] == {"status": "flagged", "flags": flags}
    assert screening_publication(plan=before, state=state, budget=budget)["scene_feasibility"]["flags"] == []
    schema = json.loads((ROOT / "docs/schemas/beta_policy_screening_publication.v3.schema.json").read_text())
    Draft202012Validator(schema["properties"]["scene_feasibility"], format_checker=FormatChecker()).validate(
        publication["scene_feasibility"])
    assert scene_feasibility([], estimates=["not a code!"])["flags"] != []  # carried as untyped, never as text
    with pytest.raises(BetaPolicyScreeningPlanError, match="beta_screening_plan_schema_invalid"):
        validate_screening_plan(seal({**make_plan(), "scene_estimate_flags": ["Not A Code"]}, "plan_digest"))
