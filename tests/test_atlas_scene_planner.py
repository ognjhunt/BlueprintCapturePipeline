"""Model review cannot waive source/privacy/coverage, budget or durable intent."""
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from blueprint_pipeline import atlas_scene_planner as planner
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def fixture(tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (20, 30), "white").save(image)
    survey = {"source_video_digest": "sha256:" + "a" * 64, "heldout_pixels_included": False,
        "display_rotation_degrees": 90, "frames": [{"frame_id": str(i), "timestamp_seconds": i,
        "source_image_path": str(image), "source_image_digest": _sha256_file(image),
        "sharpness": 50, "clipped_fraction": 0} for i in range(3)]}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    path = tmp_path / "survey.json"
    path.write_text(json.dumps(survey))
    proposal = {"status": "proposed", "scene_state": "same open appliance", "static_intervals_seconds": [[0, 1.5]],
        "anchor_frame_id": "1", "coverage_requirements": [
            {"role": "task_object", "label": "object", "required_frame_ids": ["1"]},
            {"role": "surrounding_scene", "label": "room", "required_frame_ids": ["0"]}],
        "unsafe_frame_ids": ["2"], "uncaptured_regions": ["rear"],
        "selection_policy": {"minimum_sharpness": 20, "maximum_clipped_fraction": .5, "minimum_overlap_inliers": 24},
        "summary": "room and object visible"}
    policy = {"model": "gpt-6.1-sol", "maximum_review_cost_usd": .5, "pose_input_cost_usd": .01,
        "rgbd_view_cost_usd": .05,
        "rate_card_reference": "https://example.test/rates", "reviewed_by": "operator", "reviewed_on": "2026-10-21"}
    return path, survey, proposal, policy


def test_review_cannot_include_unsafe_frames_or_omit_room(tmp_path):
    _, survey, proposal, policy = fixture(tmp_path)
    result = planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=4)
    assert result["selection_budget"]["maximum_pose_cost_usd"] == 3.9
    proposal["static_intervals_seconds"] = [[0, 2.5]]
    with pytest.raises(ValueError, match="privacy_invalid"):
        planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=4)
    proposal["static_intervals_seconds"] = [[0, 1.5]]
    proposal["coverage_requirements"] = proposal["coverage_requirements"][:1]
    with pytest.raises(ValueError, match="privacy_invalid"):
        planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=4)


def test_selection_budget_preserves_generated_coverage_before_buying_pose(tmp_path):
    _, survey, proposal, policy = fixture(tmp_path)
    # An anchor used for two roles is generated and charged once.
    proposal["coverage_requirements"][1]["required_frame_ids"].append("1")
    review = planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=.13)
    assert review["selection_budget"]["maximum_pose_cost_usd"] == .03
    with pytest.raises(ValueError, match="budget_insufficient_for_pose_and_generation"):
        planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=.11)


def test_one_source_bound_paid_call_settles_and_resume_never_repurchases(tmp_path, monkeypatch):
    path, _, proposal, policy = fixture(tmp_path)
    from blueprint_pipeline import website_task_context as control
    from blueprint_pipeline import paid_resource_admission as admission
    monkeypatch.setattr(admission, "require_paid_resource_admission_grant", lambda *_, **__: None)
    calls = []
    monkeypatch.setattr(control, "reserve_website_preparation_spend", lambda **kwargs: (calls.append(kwargs) or {}, object()))
    monkeypatch.setattr(control, "website_webapp_request", lambda **kwargs: {"status": "settled"})
    class Invoker:
        def configure_reservation_audit(self, **_):
            pass
        def invoke(self, spec, input_value):
            assert spec.model == policy["model"] and len(input_value[0]["content"]) == 7
            return SimpleNamespace(output=proposal, cost_usd=.03, usage={"input_tokens": 100},
                                   model=spec.model, sdk_version="test")
    options = dict(survey_path=path, task_context={"description": "test task", "context_digest": "sha256:" + "b" * 64,
        "request_id": "r", "scene_id": "s", "capture_id": "c"}, configuration={"policy": policy},
        output_root=tmp_path / "plans", invoker=Invoker(), release_blockers=lambda: ("c" * 40, []))
    result = planner.review_survey(**options)
    assert planner.review_survey(**options) == result
    assert len(calls) == 1 and calls[0]["provider"] == "openai" and calls[0]["source_video_digest"]
    retained = tmp_path / "plans" / (result["binding_digest"][7:] + ".json")
    retained.write_text(json.dumps({"status": "submitting"}))
    with pytest.raises(ValueError, match="reconciliation"):
        planner.review_survey(**options)
    assert len(calls) == 1


def test_unpriced_or_overbudget_model_refused_before_reservation(tmp_path):
    path, _, _, policy = fixture(tmp_path)
    policy["maximum_review_cost_usd"] = .00001
    with pytest.raises(ValueError, match="token_budget_exceeded"):
        planner.review_survey(survey_path=path, task_context={}, configuration={"policy": policy}, output_root=tmp_path)


def test_gap_review_preserves_state_quality_and_original_views(tmp_path):
    _, survey, proposal, policy = fixture(tmp_path)
    from copy import deepcopy
    prior = deepcopy(proposal)
    prior["static_intervals_seconds"] = [[0, 2.5]]
    proposal.update(static_intervals_seconds=[[0, 2.5]], unsafe_frame_ids=[], action="refine",
                    gap_assessment=["Frame 2 covers the edge missing from the draft"], additional_input_frame_ids=["2"])
    feedback = {"prior_review": prior, "prior_selected_frame_ids": ["0", "1"], "digest": "feedback"}
    review = planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=2, feedback=feedback)
    assert review["additional_input_frame_ids"] == ["2"]
    assert review["selection_budget"]["maximum_pose_cost_usd"] == 1.9  # No synthetic view purchase for frame 2.
    for mutation in ({"static_intervals_seconds": [[0, 2.4]]},
                     {"selection_policy": {**proposal["selection_policy"], "minimum_overlap_inliers": 8}},
                     {"action": "keep"}):
        with pytest.raises(ValueError, match="refinement_state_or_selection_invalid"):
            planner.proposal_review({**proposal, **mutation}, survey=survey, policy=policy, pose_budget=2, feedback=feedback)
    feedback["prior_selected_frame_ids"].append("missing")
    with pytest.raises(ValueError, match="refinement_state_or_selection_invalid"):
        planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=2, feedback=feedback)


def test_gap_pixels_are_labeled_derived_and_digest_checked(tmp_path):
    _, survey, _, _ = fixture(tmp_path)
    image = tmp_path / "render.png"
    Image.new("RGB", (20, 30), "red").save(image)
    feedback = {"digest": "feedback", "previews": [{"path": str(image), "sha256": _sha256_file(image),
                                                     "source_frame_id": "1"}]}
    content = planner._input(survey, {"description": "task"}, feedback)[0]["content"]
    assert json.loads(content[-2]["text"])["not_source_observation"] is True
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="preview_changed"):
        planner._input(survey, {"description": "task"}, feedback)


def test_unselected_blurry_candidate_does_not_invalidate_sharp_static_inputs(tmp_path):
    _, survey, proposal, policy = fixture(tmp_path)
    proposal["static_intervals_seconds"] = [[0, 2.5]]
    survey["frames"][2]["sharpness"] = 19  # Existing retained threshold is 20.
    review = planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=2)
    assert review["quality_excluded_frame_ids"] == ["2"]
    assert review["selection_policy"]["minimum_sharpness"] == 20
    # An unsafe frame that passes quality is still a privacy/state refusal.
    survey["frames"][2]["sharpness"] = 20
    with pytest.raises(ValueError, match="privacy_invalid"):
        planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=2)
    survey["frames"][2]["sharpness"] = 19
    proposal["coverage_requirements"][0]["required_frame_ids"].append("2")
    with pytest.raises(ValueError, match="privacy_invalid"):
        planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=2)
