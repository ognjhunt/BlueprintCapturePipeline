"""Model review cannot waive source/privacy/coverage, budget or durable intent."""
import json
from pathlib import Path
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
    proposal = {"status": "proposed", "subject_origin": "captured", "scene_state": "same open appliance", "static_intervals_seconds": [[0, 1.5]],
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
    assert "excluded_frame_ids" not in result and "transient_frame_ids" not in result
    # Owner direction 2026-10-22: one video must work while people keep working
    # in it. An unusable frame inside the interval is left out, not fatal...
    proposal["static_intervals_seconds"] = [[0, 2.5]]
    assert planner.proposal_review(proposal, survey=survey, policy=policy,
                                   pose_budget=4)["excluded_frame_ids"] == ["2"]
    # ...but it can never be a required view.
    proposal["coverage_requirements"][0]["required_frame_ids"].append("2")
    with pytest.raises(ValueError, match="privacy_invalid"):
        planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=4)
    proposal["coverage_requirements"][0]["required_frame_ids"].remove("2")
    proposal["static_intervals_seconds"] = [[0, 1.5]]
    proposal["coverage_requirements"] = proposal["coverage_requirements"][:1]
    with pytest.raises(ValueError, match="privacy_invalid"):
        planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=4)


@pytest.mark.parametrize("violation", [None, "remove", "instruction", "transient", "captured_surrogate", "uncertain"])
@pytest.mark.parametrize("subject_origin", ["synthetic_design", "no_manipulated_subject"])
def test_preserved_task_requires_regions_and_keeps_every_captured_object(tmp_path, violation, subject_origin):
    _, survey, proposal, policy = fixture(tmp_path)
    proposal["subject_origin"] = subject_origin
    proposal["coverage_requirements"][0]["role"] = "task_region"
    proposal["automation"] = {"scale_references": [], "objects": [
        {"target_id": "support", "label": "observed support", "segmentation_prompt": "support",
         "disposition": "keep", "regions": [{"frame_id": "1", "box_xywh_normalized": [0, 0, 1, 1]}]}],
        "removal_instruction": "", "mask_margin_fraction": .01}
    if violation == "remove":
        proposal["automation"]["objects"][0]["disposition"] = "remove"
        proposal["automation"]["removal_instruction"] = "Erase support"
    elif violation == "instruction":
        proposal["automation"]["removal_instruction"] = "Erase something"
    elif violation == "transient":
        proposal["transient_frame_ids"] = ["1"]
    elif violation == "captured_surrogate":
        proposal["coverage_requirements"].append(
            {"role": "task_object", "label": "surrogate", "required_frame_ids": ["1"]})
    elif violation == "uncertain":
        proposal["subject_origin"] = "uncertain"
    if violation:
        with pytest.raises(ValueError, match="preserved_capture|subject_origin_uncertain"):
            planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=4)
    else:
        result = planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=4)
        assert result["subject_origin"] == subject_origin
        assert result["automation"]["objects"][0]["disposition"] == "keep"
        assert result["selection_budget"]["maximum_pose_cost_usd"] == 4


def test_selection_budget_preserves_generated_coverage_before_buying_pose(tmp_path):
    _, survey, proposal, policy = fixture(tmp_path)
    # An anchor used for two roles is generated and charged once.
    proposal["coverage_requirements"][1]["required_frame_ids"].append("1")
    review = planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=.13)
    assert review["selection_budget"]["maximum_pose_cost_usd"] == .03
    with pytest.raises(ValueError, match="budget_insufficient_for_pose_and_generation"):
        planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=.11)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("mode", ["scene", "completion", "export_coverage", "scale", "recovery", "recorded_sources"])
def test_one_source_bound_paid_call_settles_and_resume_never_repurchases(tmp_path, monkeypatch, mode, dense):
    completion_mode = mode in {"completion", "export_coverage"}
    path, survey, proposal, policy = fixture(tmp_path)
    if dense:
        survey["frames"] = [{**survey["frames"][0], "frame_id": str(i), "timestamp_seconds": i / 10}
                            for i in range(173)]
        survey["digest"] = canonical_digest(survey, digest_field="digest")
        path.write_text(json.dumps(survey))
    if mode in {"recovery", "recorded_sources"} and not dense:
        policy["maximum_review_cost_usd"] = 2
    from blueprint_pipeline import website_task_context as control
    from blueprint_pipeline import paid_resource_admission as admission
    monkeypatch.setattr(admission, "require_paid_resource_admission_grant", lambda *_, **__: None)
    calls = []
    monkeypatch.setattr(control, "reserve_website_preparation_spend", lambda **kwargs: (calls.append(kwargs) or {}, object()))
    monkeypatch.setattr(control, "website_webapp_request", lambda **kwargs: {"status": "settled"})
    if completion_mode:
        proposal = {"status": "proposed", "selected_camera_ids": ["a-wide", "b-lower"],
                    "gap_assessment": ["Complete floor"], "remaining_unknowns": ["Hidden wall"],
                    'scene_caption': 'The same room with a continuous bare floor.',
                    'view_gap_reasons': [{'camera_id': 'a-wide', 'reason': 'Keep the task anchor'},
                                        {'camera_id': 'b-lower', 'reason': 'Complete nearby floor'}]}
        if mode == 'export_coverage':
            proposal['environment_surfaces'] = [{'component': 'support assembly', 'surface': 'end face',
                'required_for_local_environment': True, 'coverage': 'material_gap',
                'source_frame_ids': ['0'], 'evidence_camera_ids': ['a-wide'], 'gap_ids': [],
                'reason': 'A large part of the photographed support is absent from the export'}]
    elif mode == "scale":
        proposal = {"status": "abstained", "scale_references": [], "summary": "No defensible visible reference"}
    elif mode == "recovery":
        proposal = {"status": "proposed", "task_extent_summary": "Local working area",
                    "views": [{"frame_id": "0", "regions": [], "reason": "No missing task surface"}],
                    "remaining_required_surfaces": [], "excluded_scene_regions": ["distant scenery"]}
    elif mode == 'recorded_sources':
        proposal = {'views': [{'frame_id': '0', 'decision': 'use', 'fixed_environment_consistent': True,
                              'objects': [], 'reason': 'The fixed support is visible'}],
                    'transients': [], 'summary': 'Recover the recorded support'}
    class Invoker:
        def configure_reservation_audit(self, **_):
            pass
        def invoke(self, spec, input_value):
            content = input_value[0]["content"]
            assert spec.model == policy["model"] and len(content) == 1 + 2 * len(survey["frames"]) + (mode != "scene")
            labels = [json.loads(part["text"]) for part in content if part["type"] == "input_text"]
            assert [row["frame_id"] for row in labels if "frame_id" in row] == [row["frame_id"] for row in survey["frames"]]
            if dense:
                from blueprint_pipeline.openai_prompt_cache import worst_case_reservation_usd
                assert worst_case_reservation_usd(model=spec.model, input_token_ceiling=spec.max_input_tokens,
                    max_output_tokens=spec.max_output_tokens, cache_policy=None) + spec.max_hosted_tool_calls_per_turn * .01 <= policy["maximum_review_cost_usd"]
            no_search = completion_mode or mode in {"recovery", "recorded_sources"}
            assert len(spec.hosted_tools) == (0 if no_search else 1)
            assert spec.max_hosted_tool_calls_per_turn == (0 if no_search else 2) and spec.hosted_tool_call_usd == .01
            if mode == 'export_coverage':
                assert spec.max_output_tokens >= 8000, 'Surface and source decisions were truncated at 4k'
            return SimpleNamespace(output=proposal, cost_usd=.03, usage={"input_tokens": 100},
                                   model=spec.model, sdk_version="test")
    options = dict(survey_path=path, task_context={"description": "test task", "context_digest": "sha256:" + "b" * 64,
        "request_id": "r", "scene_id": "s", "capture_id": "c"}, configuration={"policy": policy},
        output_root=tmp_path / "plans", invoker=Invoker(), release_blockers=lambda: ("c" * 40, []))
    if completion_mode:
        options["completion"] = {"previews": [], "candidate_camera_ids": ["a-wide", "b-lower"]}
        if mode == 'export_coverage':
            options['completion']['post_export_coverage'] = {}
    elif mode == "scale":
        options["scaling"] = {"previews": [], "previous_measurement_digest": "old"}
    elif mode == "recovery":
        options["recovery"] = {"previews": [], "native_frame_ids": ["0"]}
    elif mode == 'recorded_sources':
        options['recovery'] = {'scope': 'recorded_background_sources', 'previews': [], 'requested_frame_ids': ['0']}
    result = planner.review_survey(**options)
    assert planner.review_survey(**options) == result
    assert len(calls) == 1 and calls[0]["provider"] == "openai" and calls[0]["source_video_digest"]
    retained = tmp_path / "plans" / (result["binding_digest"][7:] + ".json")
    retained.write_text(json.dumps({"status": "submitting"}))
    with pytest.raises(ValueError, match="reconciliation"):
        planner.review_survey(**options)
    assert len(calls) == 1


def _abstained_source_review(tmp_path, monkeypatch, *, still_missing=False, change_source=False):
    import base64
    from copy import deepcopy
    from io import BytesIO
    from blueprint_pipeline import website_task_context as control
    from blueprint_pipeline import paid_resource_admission as admission

    path, survey, proposal, policy = fixture(tmp_path)
    source = Path(survey["frames"][0]["source_image_path"])
    Image.new("RGB", (1920, 1080), "white").save(source)
    for row in survey["frames"]:
        row["source_image_digest"] = _sha256_file(source)
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    path.write_text(json.dumps(survey))
    abstained = deepcopy(proposal)
    abstained.update(status="abstained", summary="Small overview previews do not establish a usable object/destination pair",
                     uncaptured_regions=["A clear view of one permitted object and its destination"])
    abstained["coverage_requirements"] = abstained["coverage_requirements"][1:]
    calls = {"reserved": [], "settled": [], "inputs": [], "source": source,
             "source_digest": _sha256_file(source)}
    monkeypatch.setattr(admission, "require_paid_resource_admission_grant", lambda *_, **__: None)
    monkeypatch.setattr(control, "reserve_website_preparation_spend",
                        lambda **kw: (calls["reserved"].append(kw) or {}, object()))
    monkeypatch.setattr(control, "website_webapp_request",
                        lambda **kw: calls["settled"].append(kw) or {"status": "settled"})

    class Invoker:
        def configure_reservation_audit(self, **_):
            pass

        def invoke(self, spec, value):
            parts = value[0]["content"]
            frames = [json.loads(p["text"])["frame_id"] for p in parts
                      if p["type"] == "input_text" and "frame_id" in json.loads(p["text"])]
            sizes = []
            for part in parts:
                if part["type"] == "input_image":
                    with Image.open(BytesIO(base64.b64decode(part["image_url"].partition(",")[2]))) as image:
                        sizes.append(image.size)
            calls["inputs"].append({"frames": frames, "sizes": sizes, "instructions": spec.instructions})
            first = len(calls["inputs"]) == 1
            if first and change_source:
                Image.new("RGB", (1920, 1080), "black").save(source)
            return SimpleNamespace(output=abstained if first or still_missing else proposal,
                                   cost_usd=.03, usage={"input_tokens": 100}, model=spec.model, sdk_version="test")

    options = dict(survey_path=path, task_context={"description": "Move any suitable observed box to an empty spot on the same shelf",
        "context_digest": "sha256:" + "b" * 64, "request_id": "r", "scene_id": "s", "capture_id": "c"},
        configuration={"policy": policy}, output_root=tmp_path / "plans", invoker=Invoker(),
        release_blockers=lambda: ("c" * 40, []))
    return options, calls, survey, policy


def test_abstained_preview_gets_one_retained_higher_detail_source_review(tmp_path, monkeypatch):
    options, calls, survey, _ = _abstained_source_review(tmp_path, monkeypatch)
    result = planner.review_survey(**options)
    assert result["proposal"]["status"] == "proposed"
    assert len(calls["inputs"]) == len(calls["reserved"]) == len(calls["settled"]) == 2
    assert calls["inputs"][0]["sizes"] == [(288, 512)] * 3
    assert calls["inputs"][1]["sizes"] == [(720, 1280)] * 3
    assert all(row["frames"] == [frame["frame_id"] for frame in survey["frames"]] for row in calls["inputs"])
    assert _sha256_file(calls["source"]) == calls["source_digest"]
    assert all(row["source_video_digest"] == survey["source_video_digest"] for row in calls["reserved"])
    assert all(row["maximum_cost_usd"] == options["configuration"]["policy"]["maximum_review_cost_usd"]
               for row in calls["reserved"])
    predecessor = json.loads((options["output_root"] /
                             (result["binding"]["source_review_recovery"]["prior_binding_digest"][7:] + ".json")).read_text())
    assert predecessor["proposal"]["status"] == "abstained" and predecessor["cost_usd"] == .03
    assert result["binding"]["source_review_recovery"]["prior_proposal_digest"] == predecessor["proposal_digest"]
    assert planner.review_survey(**options) == result
    assert len(calls["inputs"]) == len(calls["reserved"]) == 2


def test_unresolved_source_gap_stops_after_one_detail_review_and_is_never_promoted(tmp_path, monkeypatch):
    options, calls, survey, policy = _abstained_source_review(tmp_path, monkeypatch, still_missing=True)
    result = planner.review_survey(**options)
    assert result["proposal"]["status"] == "abstained"
    with pytest.raises(ValueError, match="atlas_scene_planner_abstained"):
        planner.proposal_review(result["proposal"], survey=survey, policy=policy, pose_budget=4)
    assert planner.review_survey(**options) == result
    assert len(calls["inputs"]) == len(calls["reserved"]) == 2


def test_changed_source_refuses_before_the_detail_review_reserves_or_calls(tmp_path, monkeypatch):
    options, calls, _, _ = _abstained_source_review(tmp_path, monkeypatch, change_source=True)
    with pytest.raises(ValueError, match="atlas_scene_planner_source_changed"):
        planner.review_survey(**options)
    assert len(calls["inputs"]) == len(calls["reserved"]) == len(calls["settled"]) == 1
    assert len(list(options["output_root"].glob("*.settlement.json"))) == 1


def test_detail_review_keeps_the_supplied_invokers_existing_cumulative_budget(tmp_path, monkeypatch):
    from blueprint_pipeline.openai_prompt_cache import worst_case_reservation_usd
    options, calls, survey, policy = _abstained_source_review(tmp_path, monkeypatch)
    first_ceiling = worst_case_reservation_usd(model=policy["model"],
        input_token_ceiling=24000 + 3000 * len(survey["frames"]), max_output_tokens=4000, cache_policy=None)
    options["invoker"].config = SimpleNamespace(max_inference_cost_usd=first_ceiling + .02 + .000001)
    with pytest.raises(ValueError, match="atlas_scene_planner_invoker_budget_exhausted"):
        planner.review_survey(**options)
    assert len(calls["inputs"]) == len(calls["reserved"]) == len(calls["settled"]) == 1


def test_full_frame_export_review_reserves_for_each_image_not_each_sheet(tmp_path, monkeypatch):
    path, survey, _, policy = fixture(tmp_path)
    image = tmp_path / 'preview.png'
    Image.new('RGB', (1280, 720), 'white').save(image)
    previews = [{'path': str(image), 'kind': 'geometry_gap_survey'}] * 12
    captured = []
    from blueprint_pipeline import openai_prompt_cache
    monkeypatch.setattr(openai_prompt_cache, 'worst_case_reservation_usd',
                        lambda **kw: captured.append(kw) or 1.)
    # The unaffordable quote stops before any paid reservation/model invocation.
    with pytest.raises(ValueError, match='token_budget_exceeded'):
        planner.review_survey(survey_path=path, task_context={}, configuration={'policy': policy},
            output_root=tmp_path, completion={'scope': 'missing_surfaces_within_captured_scene',
                'post_export_coverage': {}, 'previews': previews})
    assert captured[0]['input_token_ceiling'] >= 24_000 + planner._export_image_token_ceiling(survey, previews)
    assert planner._export_image_token_ceiling(survey, previews) > planner._export_image_token_ceiling(survey, previews[:3])


def test_realistic_retained_full_frame_review_stays_within_one_dollar(tmp_path):
    from blueprint_pipeline.openai_prompt_cache import worst_case_reservation_usd

    source, preview = tmp_path / 'source.png', tmp_path / 'preview.png'
    Image.new('RGB', (1080, 1920), 'white').save(source)
    Image.new('RGB', (1280, 720), 'white').save(preview)
    survey = {'frames': [{'source_image_path': str(source)}] * 7}
    previews = [{'path': str(preview)}] * 77
    bound = 24_000 + planner._export_image_token_ceiling(survey, previews) + 40_000
    assert worst_case_reservation_usd(model='gpt-6.1-sol', input_token_ceiling=bound,
                                      max_output_tokens=13_500, cache_policy=None) < 1
    assert planner._export_image_token_ceiling({'frames': []}, previews[:1]) == 3 * 32 * 18 + 256


def test_large_export_review_fits_its_existing_cap_without_losing_views_or_gaps(tmp_path):
    from blueprint_pipeline.openai_prompt_cache import worst_case_reservation_usd
    source, preview = tmp_path / 'full-source.png', tmp_path / 'derived.png'
    Image.new('RGB', (1080, 1920), 'white').save(source)
    Image.new('RGB', (1280, 720), 'white').save(preview)
    survey = {'display_rotation_degrees': 0, 'frames': [{
        'frame_id': str(i), 'source_image_path': str(source), 'source_image_digest': _sha256_file(source)}
        for i in range(7)]}
    cameras = ['gap-target-source-camera-' + str(i).zfill(5) + '-bounded-environment-surface' for i in range(65)]
    rows = [{'camera_id': key, 'source_frame_id': '0', 'pose_rub': [0., 1., 2., 0., 0., 0., 1.],
             'purpose': 'gap_target', 'source_gap_ids': [key + '-gap-1']} for key in cameras]
    previews = [{'path': str(source if i < 19 else preview),
        'sha256': _sha256_file(source if i < 19 else preview),
        'source_frame_id': str(i), 'camera_id': cameras[i % len(cameras)], 'timestamp_seconds': i,
        'kind': 'original_capture_source_check_only' if i < 19 else 'geometry_gap_survey'} for i in range(94)]
    extra = {'scope': 'missing_surfaces_within_captured_scene', 'post_export_coverage': {},
        'candidate_camera_ids': rows, 'previews': previews,
        'gap_survey': {'views': [{'camera_id': key, 'gaps': [[j, 100, [.1, .2, .3, .4]] for j in range(7)]}
                                for key in cameras]}}
    value, payload, tokens, output, _ = planner._fit_export_review(
        survey, {'description': 'Reconstruct this work area'}, extra, model='gpt-6.1-sol', maximum=.5,
        output_tokens=18250, instructions=planner.COMPLETION_INSTRUCTIONS + planner.POST_EXPORT_COMPLETION_INSTRUCTIONS,
        schema=planner.CoverageCompletionProposal)
    assert worst_case_reservation_usd(model='gpt-6.1-sol', input_token_ceiling=tokens,
                                      max_output_tokens=output, cache_policy=None) <= .5
    assert 8000 <= output <= planner.SDK_MAX_OUTPUT_TOKENS
    column = value['candidate_camera_columns'].index('camera_id')
    assert [row[column] for row in value['candidate_camera_ids']] == cameras
    assert [row[0] for row in value['gap_survey']['views']] == cameras
    assert sum(len(row[2]) for row in value['gap_survey']['views']) == 455
    assert len(value['previews']) == 94
    assert value['planning_preview_layout']['original_long_side'] == 1024
    labels = [json.loads(part['text']) for part in payload[0]['content'] if part['type'] == 'input_text']
    identities = [tile['camera_id'] for label in labels for tile in label.get('derived_reconstruction_previews', [])]
    assert identities == [row['camera_id'] for row in previews[19:]]
    assert sum('original_capture_frame' in row for row in labels) == 19


def test_unpriced_or_overbudget_model_refused_before_reservation(tmp_path):
    path, _, _, policy = fixture(tmp_path)
    policy["maximum_review_cost_usd"] = .00001
    with pytest.raises(ValueError, match="token_budget_exceeded"):
        planner.review_survey(survey_path=path, task_context={}, configuration={"policy": policy}, output_root=tmp_path)


def test_thirteen_view_surface_response_is_not_capped_at_eight_thousand_tokens(tmp_path, monkeypatch):
    path, _, _, policy = fixture(tmp_path)
    policy["maximum_review_cost_usd"] = 1
    from blueprint_pipeline import website_task_context as control, paid_resource_admission as admission
    monkeypatch.setattr(admission, "require_paid_resource_admission_grant", lambda *_, **__: None)
    monkeypatch.setattr(control, "reserve_website_preparation_spend", lambda **kw: ({}, object()))
    monkeypatch.setattr(control, "website_webapp_request", lambda **kw: {"status": "settled"})
    ids = [str(i) for i in range(13)]
    class Invoker:
        def configure_reservation_audit(self, **_):
            pass
        def invoke(self, spec, inputs):
            assert spec.max_output_tokens > 8000, "The retained 13-view response was truncated at 8k"
            return SimpleNamespace(output={"status": "proposed", "task_extent_summary": "Task and context",
                "views": [{"frame_id": fid, "regions": [], "reason": "Reviewed"} for fid in ids],
                "remaining_required_surfaces": [], "excluded_scene_regions": []}, cost_usd=.1,
                usage={"input_tokens": 100, "output_tokens": 9000}, model=spec.model, sdk_version="test")
    result = planner.review_survey(survey_path=path,
        task_context={"description": "task", "context_digest": "c", "capture_id": "c", "scene_id": "s", "request_id": "r"},
        configuration={"policy": policy}, output_root=tmp_path / "plans", invoker=Invoker(),
        release_blockers=lambda: ("a" * 40, []), recovery={"native_frame_ids": ids, "previews": []})
    assert [v["frame_id"] for v in result["proposal"]["views"]] == ids


def cut_off_review(tmp_path, monkeypatch, *, cut_offs, output_tokens=None):
    """A planner whose first ``cut_offs`` responses stop at their output allowance."""
    path, _, proposal, policy = fixture(tmp_path)
    policy["maximum_review_cost_usd"] = 1
    from agents.exceptions import ModelBehaviorError
    from blueprint_pipeline import website_task_context as control, paid_resource_admission as admission
    from blueprint_pipeline.task_evaluation_supervisor.inference_reservations import InferenceReservationAudit
    monkeypatch.setattr(admission, "require_paid_resource_admission_grant", lambda *_, **__: None)
    monkeypatch.setattr(InferenceReservationAudit, "record_completion", lambda self, value: None)
    calls = {"allowances": [], "reserved": [], "settled": []}
    monkeypatch.setattr(control, "reserve_website_preparation_spend",
                        lambda **kw: (calls["reserved"].append(kw["binding_digest"]) or {}, object()))
    monkeypatch.setattr(control, "website_webapp_request",
                        lambda **kw: calls["settled"].append(kw["payload"]["settlement"]) or {"status": "settled"})

    class Invoker:
        def configure_reservation_audit(self, *, record_completion, **_):
            self.record = record_completion

        def invoke(self, spec, inputs):
            calls["allowances"].append(spec.max_output_tokens)
            if len(calls["allowances"]) <= cut_offs:
                # The SDK records the observed charge, then rejects the unparseable JSON.
                self.record({"status": "invalid_structured_output", "reconciled_actual_cost_usd": .02,
                             "usage": {"output_tokens": output_tokens or spec.max_output_tokens,
                                       "cost_is_actual": True}})
                raise ModelBehaviorError("Invalid JSON when parsing a response cut off mid-object")
            return SimpleNamespace(output=proposal, cost_usd=.03, usage={"output_tokens": 10},
                                   model=spec.model, sdk_version="test")
    options = dict(survey_path=path, task_context={"description": "task", "context_digest": "sha256:" + "b" * 64,
        "request_id": "r", "scene_id": "s", "capture_id": "c"}, configuration={"policy": policy},
        output_root=tmp_path / "plans", invoker=Invoker(), release_blockers=lambda: ("c" * 40, []))
    return options, calls


def test_cut_off_review_is_settled_and_retried_with_a_larger_allowance(tmp_path, monkeypatch):
    options, calls = cut_off_review(tmp_path, monkeypatch, cut_offs=1)
    result = planner.review_survey(**options)
    assert result["status"] == "completed" and result["binding"]["max_output_tokens"] == 8000
    assert calls["allowances"] == [4000, 8000] and len(set(calls["reserved"])) == 2
    # The cut-off call is charged what it used plus its hosted-search allowance; nothing is left held.
    assert [row["provider_charge_amount_usd"] for row in calls["settled"]] == [pytest.approx(.04), .03]
    cut_off = json.loads((tmp_path / "plans" / (calls["reserved"][0][7:] + ".json")).read_text())
    assert cut_off["status"] == "truncated" and cut_off["truncation_retry"] == {"max_output_tokens": 8000, "attempt": 1}
    # A restart follows the retained chain without buying either call again.
    assert planner.review_survey(**options) == result
    assert calls["allowances"] == [4000, 8000] and len(calls["settled"]) == 2


def test_cut_off_review_stops_after_two_larger_retries(tmp_path, monkeypatch):
    options, calls = cut_off_review(tmp_path, monkeypatch, cut_offs=3)
    with pytest.raises(ValueError, match="output_truncated_retry_unavailable"):
        planner.review_survey(**options)
    assert calls["allowances"] == [4000, 8000, 16000] and len(calls["settled"]) == 3
    with pytest.raises(ValueError, match="output_truncated_retry_unavailable"):
        planner.review_survey(**options)
    assert len(calls["allowances"]) == 3 and len(calls["settled"]) == 3


def test_retry_loop_keeps_the_allocator_as_the_only_paid_review_caller():
    # Main's release verifier, not PR CI, rejects any other function naming
    # review_survey; a self-call here would block the signed deploy artifact.
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("verify_paid_resource_allocator",
                                                  root / "scripts/verify_paid_resource_allocator.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    paths = ["src/blueprint_pipeline/atlas_scene_planner.py", "src/blueprint_pipeline/paid_resource_allocator.py"]
    sources = {p: (root / p).read_text() for p in paths}
    assert verifier._s3_transport_capability_callers(
        sources, verifier.ATLAS_PLAN_CAPABILITIES) == verifier.APPROVED_ATLAS_PLAN_CALLERS


class AccountingInvoker:
    """Mirrors OpenAIAgentsSDKInvoker's balance: it only grows and is bounded by its budget."""

    def __init__(self, proposal, budget):
        self.proposal, self.balance, self.calls = proposal, 0.0, []
        self.config = SimpleNamespace(max_inference_cost_usd=budget)

    def configure_reservation_audit(self, *, record_completion, restored_reserved_cost_usd, **_):
        if restored_reserved_cost_usd < self.balance:
            raise ValueError("agents_sdk_restored_reservation_cannot_decrease")
        self.balance, self.record = restored_reserved_cost_usd, record_completion

    def invoke(self, spec, inputs):
        from agents.exceptions import ModelBehaviorError
        self.calls.append(spec.max_output_tokens)
        if len(self.calls) == 1:
            self.balance += .02
            self.record({"status": "invalid_structured_output", "reconciled_actual_cost_usd": .02,
                         "usage": {"output_tokens": spec.max_output_tokens, "cost_is_actual": True}})
            raise ModelBehaviorError("Invalid JSON when parsing a response cut off mid-object")
        return SimpleNamespace(output=self.proposal, cost_usd=.03, usage={"output_tokens": 10},
                               model=spec.model, sdk_version="test")


def test_supplied_invoker_keeps_its_balance_across_a_cut_off_retry(tmp_path, monkeypatch):
    options, calls = cut_off_review(tmp_path, monkeypatch, cut_offs=0)
    _, _, proposal, _ = fixture(tmp_path)
    options["invoker"] = invoker = AccountingInvoker(proposal, budget=1)
    result = planner.review_survey(**options)
    # The reused invoker's charge carries into the retry; resetting it would refuse after spending.
    assert result["status"] == "completed" and invoker.calls == [4000, 8000]
    assert invoker.balance >= .02 and len(calls["reserved"]) == 2


def test_supplied_invoker_without_budget_refuses_before_spend(tmp_path, monkeypatch):
    options, calls = cut_off_review(tmp_path, monkeypatch, cut_offs=0)
    _, _, proposal, _ = fixture(tmp_path)
    options["invoker"] = AccountingInvoker(proposal, budget=0.0)
    with pytest.raises(ValueError, match="invoker_budget_exhausted"):
        planner.review_survey(**options)
    assert calls["reserved"] == [] and not list((tmp_path / "plans").glob("*.json"))


def test_invalid_review_that_was_not_cut_off_is_settled_and_repaired_once(tmp_path, monkeypatch):
    options, calls = cut_off_review(tmp_path, monkeypatch, cut_offs=1, output_tokens=900)
    result = planner.review_survey(**options)
    assert result["status"] == "completed" and result["binding"]["validation_repair"]["attempt"] == 1
    assert "was refused (ModelBehaviorError" in result["binding"]["instructions"]
    assert calls["allowances"] == [4000, 4000] and len(set(calls["reserved"])) == 2
    # The refused call is charged what it used plus its hosted-search allowance.
    assert [row["provider_charge_amount_usd"] for row in calls["settled"]] == [pytest.approx(.04), .03]
    refused = json.loads((tmp_path / "plans" / (calls["reserved"][0][7:] + ".json")).read_text())
    assert refused["status"] == "invalid_output" and refused["error"]["type"] == "ModelBehaviorError"
    assert planner.review_survey(**options) == result and len(calls["settled"]) == 2


def test_invalid_review_is_refused_by_name_after_its_one_repair(tmp_path, monkeypatch):
    options, calls = cut_off_review(tmp_path, monkeypatch, cut_offs=2, output_tokens=900)
    for _ in range(2):
        with pytest.raises(ValueError, match="atlas_scene_planner_output_invalid"):
            planner.review_survey(**options)
    assert calls["allowances"] == [4000, 4000] and len(calls["settled"]) == 2


def test_a_call_that_ended_without_a_recorded_outcome_is_held_at_its_reservation_and_repaired(tmp_path, monkeypatch):
    # Production, 2026-10-27: the SDK raised before recording any completion, and the
    # submitting receipt it left refused every later tick (requires_reconciliation).
    options, calls = cut_off_review(tmp_path, monkeypatch, cut_offs=0)
    proposal_invoker = options["invoker"]

    class Stranding:
        def configure_reservation_audit(self, **_):
            pass

        def invoke(self, spec, inputs):
            calls["allowances"].append(spec.max_output_tokens)
            raise RuntimeError("worker stopped mid-call")
    with pytest.raises(RuntimeError):
        planner.review_survey(**{**options, "invoker": Stranding()})
    stranded = tmp_path / "plans" / (calls["reserved"][0][7:] + ".json")
    assert json.loads(stranded.read_text())["status"] == "submitting"

    # The next tick holds the stranded call at its reservation, then repairs it once.
    result = planner.review_survey(**{**options, "invoker": proposal_invoker})
    held = json.loads(stranded.read_text())
    assert held["status"] == "invalid_output" and held["error"]["type"] == "submission_outcome_unknown"
    assert held["usage"]["cost_is_actual"] is False and 0 < held["cost_usd"] <= 1
    assert result["status"] == "completed" and result["binding"]["validation_repair"]["attempt"] == 1
    assert [row["provider_charge_amount_usd"] for row in calls["settled"]] == [held["cost_usd"], .03]


def test_a_source_fix_retries_only_a_failed_review_and_reuses_its_new_result(tmp_path, monkeypatch):
    from agents.exceptions import ModelBehaviorError

    options, calls = cut_off_review(tmp_path, monkeypatch, cut_offs=0)
    repaired_invoker = options["invoker"]

    class MissingCompletion:
        def configure_reservation_audit(self, **_):
            pass

        def invoke(self, spec, inputs):
            calls["allowances"].append(spec.max_output_tokens)
            raise ModelBehaviorError("Invalid JSON without SDK run data")

    failed_options = {**options, "invoker": MissingCompletion()}
    with pytest.raises(ValueError, match="atlas_scene_planner_output_invalid"):
        planner.review_survey(**failed_options)
    originals = {path: path.read_bytes() for path in (tmp_path / "plans").glob("*.json")}
    assert len(calls["allowances"]) == 2
    # An unchanged release cannot buy the same refused answers again.
    with pytest.raises(ValueError, match="atlas_scene_planner_output_invalid"):
        planner.review_survey(**options)
    assert len(calls["allowances"]) == 2

    fixed_options = {**options, "invoker": repaired_invoker, "release_blockers": lambda: ("d" * 40, [])}
    result = planner.review_survey(**fixed_options)
    assert result["status"] == "completed"
    assert result["binding"]["failed_output_recovery"]["failed_source_commit"] == "c" * 40
    assert result["source_commit"] == "d" * 40
    assert len(calls["allowances"]) == 3
    assert all(path.read_bytes() == payload for path, payload in originals.items())
    assert planner.review_survey(**fixed_options) == result
    assert len(calls["allowances"]) == 3
    # A successful review is reused across releases too.
    another_release = {**fixed_options, "release_blockers": lambda: ("e" * 40, [])}
    assert planner.review_survey(**another_release) == result
    assert len(calls["allowances"]) == 3


def test_completion_output_compiles_in_the_actual_strict_agents_sdk():
    from agents import AgentOutputSchema

    compiled = AgentOutputSchema(planner.CompletionProposal)
    assert compiled.is_strict_json_schema()
    value = planner.CompletionProposal.model_validate({
        'status': 'proposed', 'selected_camera_ids': ['a-lower'], 'gap_assessment': ['Floor gap'],
        'remaining_unknowns': [], 'scene_caption': 'Same room with bare continuous floor.',
        'view_gap_reasons': [{'camera_id': 'a-lower', 'reason': 'Continue the observed floor edge'}]})
    assert value.view_gap_reasons[0].camera_id == 'a-lower'


def test_scale_output_compiles_in_the_actual_strict_agents_sdk():
    from agents import AgentOutputSchema
    assert AgentOutputSchema(planner.ScaleProposal).is_strict_json_schema()


def test_compact_completion_keeps_every_exact_gap_and_source_survey_unchanged():
    from copy import deepcopy

    original = {'previews': [], 'candidate_camera_ids': [{'camera_id': 'c', 'source_frame_id': 's',
        'camera': {'extrinsics': {'position': [1., 2., 3.], 'quaternion': [0., 0., 0., 1.]}}}],
        'gap_survey': {'digest': 'bound-survey', 'views': [{'camera_id': 'c', 'source_frame_id': 's',
            'label_path': '/retained/labels.png', 'label_sha256': 'hash', 'gaps': [
                {'label': i, 'gap_id': 'c-gap-' + str(i), 'pixel_count': 100 + i,
                 'box_xywh_normalized': [.1, .2, .3, .4]} for i in range(1, 501)]}]}}
    before = deepcopy(original)
    compact = planner.completion_summary(original)
    assert original == before and compact['gap_survey']['digest'] == 'bound-survey'
    view = compact['gap_survey']['views'][0]
    assert [(view['camera_id'] + '-gap-' + str(r[0]), r[1]) for r in view['gaps']] == [
        (r['gap_id'], r['pixel_count']) for r in original['gap_survey']['views'][0]['gaps']]
    assert len(json.dumps(compact)) < len(json.dumps(original)) / 2
    assert 'label_path' not in view and compact['candidate_camera_ids'][0]['pose_rub'] == [1., 2., 3., 0., 0., 0., 1.]


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


def test_completion_sheets_keep_every_camera_and_leave_source_images_unchanged(tmp_path):
    _, survey, _, _ = fixture(tmp_path)
    previews = []
    for index in range(5):
        path = tmp_path / f'camera-{index}.png'
        Image.new('RGB', (1280, 720), (index * 40, 20, 50)).save(path)
        previews.append({'path': str(path), 'sha256': _sha256_file(path),
                         'source_frame_id': '1', 'camera_id': f'camera-{index}'})
    feedback = {'scope': 'missing_surfaces_within_captured_scene', 'previews': previews}
    content = planner._input(survey, {'description': 'task'}, feedback)[0]['content']
    labels = [json.loads(row['text'])['derived_reconstruction_previews'] for row in content
              if row['type'] == 'input_text' and 'derived_reconstruction_previews' in row['text']]
    assert len(labels) == 2
    assert [row['camera_id'] for sheet in labels for row in sheet] == [f'camera-{i}' for i in range(5)]
    assert all(_sha256_file(Path(row['path'])) == row['sha256'] for row in previews)
    Path(previews[-1]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='preview_changed'):
        planner._input(survey, {'description': 'task'}, feedback)


def test_post_export_reviews_use_separate_readable_geometry_previews(tmp_path):
    import base64
    from io import BytesIO

    _, survey, _, _ = fixture(tmp_path)
    path = tmp_path / 'gap.png'
    Image.new('RGB', (1280, 720), 'magenta').save(path)
    feedback = {'scope': 'missing_surfaces_within_captured_scene', 'post_export_coverage': {},
                'previews': [{'path': str(path), 'sha256': _sha256_file(path),
                              'source_frame_id': 's', 'camera_id': 'side', 'kind': 'geometry_gap_survey'}]}
    content = planner._input(survey, {'description': 'task'}, feedback)[0]['content']
    assert json.loads(content[-2]['text'])['camera_id'] == 'side'
    with Image.open(BytesIO(base64.b64decode(content[-1]['image_url'].split(',')[1]))) as image:
        assert image.size == (1024, 576), 'The material-gap review must not use a 256px tile'


@pytest.mark.parametrize('coverage', ['present', 'minor_defects', 'physical_opening'])
def test_complete_local_environment_may_stop_without_physical_qualification(coverage):
    value = planner.CoverageCompletionProposal.model_validate({
        'status': 'sufficient', 'selected_camera_ids': [], 'gap_assessment': ['Local fixture covered'],
        'remaining_unknowns': ['Unseen interior and physical fit remain unverified'],
        'environment_surfaces': [{'component': 'work fixture', 'surface': 'outer end face',
            'required_for_local_environment': True, 'coverage': coverage,
            'source_frame_ids': ['s'], 'evidence_camera_ids': ['side'], 'gap_ids': [],
            'reason': 'Checked this surface in the exported geometry and source photo'}]})
    assert value.status == 'sufficient'


@pytest.mark.parametrize('change', [
    {'coverage': 'material_gap'}, {'coverage': 'unknown'},
    {'required_for_local_environment': False}, {'evidence_camera_ids': []},
])
def test_enough_task_context_cannot_excuse_an_unfinished_local_fixture(change):
    row = {'component': 'work fixture', 'surface': 'outer end face',
           'required_for_local_environment': True, 'coverage': 'present',
           'source_frame_ids': ['s'], 'evidence_camera_ids': ['side'], 'gap_ids': [],
           'reason': 'Enough context to perform the task'}
    with pytest.raises(ValueError, match='sufficient_with_unresolved_environment_surface'):
        planner.CoverageCompletionProposal.model_validate({
            'status': 'sufficient', 'selected_camera_ids': [], 'gap_assessment': ['Enough task context'],
            'remaining_unknowns': [], 'environment_surfaces': [{**row, **change}]})


def test_post_export_schema_requires_an_actual_surface_assessment():
    from agents import AgentOutputSchema

    assert AgentOutputSchema(planner.CoverageCompletionProposal).is_strict_json_schema()
    with pytest.raises(ValueError, match='environment_surfaces'):
        planner.CoverageCompletionProposal.model_validate({
            'status': 'sufficient', 'selected_camera_ids': [], 'gap_assessment': ['Enough task context'],
            'remaining_unknowns': []})


def test_unselected_blurry_candidate_does_not_invalidate_sharp_static_inputs(tmp_path):
    _, survey, proposal, policy = fixture(tmp_path)
    proposal["static_intervals_seconds"] = [[0, 2.5]]
    survey["frames"][2]["sharpness"] = 19  # Existing retained threshold is 20.
    review = planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=2)
    assert review["quality_excluded_frame_ids"] == ["2"]
    assert review["selection_policy"]["minimum_sharpness"] == 20
    # An unsafe frame that passes quality is left out of selection by the review.
    survey["frames"][2]["sharpness"] = 20
    review = planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=2)
    assert review["excluded_frame_ids"] == ["2"] and "quality_excluded_frame_ids" not in review
    survey["frames"][2]["sharpness"] = 19
    proposal["coverage_requirements"][0]["required_frame_ids"].append("2")
    with pytest.raises(ValueError, match="privacy_invalid"):
        planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=2)


def _automation():
    return {"scale_references": [], "objects": [{"target_id": "obj", "label": "object",
            "segmentation_prompt": "object", "disposition": "remove",
            "regions": [{"frame_id": "1", "box_xywh_normalized": [.2, .2, .5, .5]}]}],
            "removal_instruction": "Remove the object.", "mask_margin_fraction": .01}


def test_people_working_in_the_one_video_stay_usable_and_are_named_for_erasing(tmp_path):
    _, survey, proposal, policy = fixture(tmp_path)
    proposal.update(static_intervals_seconds=[[0, 2.5]], unsafe_frame_ids=[], automation=_automation(),
                    transient_frame_ids=["0", "1"],
                    transients=[{"label": "rolling cart", "segmentation_prompt": "cart"}])
    review = planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=4)
    assert review["transient_frame_ids"] == ["0", "1"]
    assert review["transient_policy"] == "erased_before_reconstruction"
    # People are always looked for, whatever else the planner names.
    assert review["transients"] == [
        {"target_id": "transient-0", "label": "rolling cart", "segmentation_prompt": "cart"},
        {"target_id": "transient-1", "label": "person", "segmentation_prompt": "person"}]
    assert "excluded_frame_ids" not in review
    # A transient named only outside every still interval is not a selected view.
    proposal["static_intervals_seconds"] = [[0, 1.5]]
    proposal["transient_frame_ids"] = ["2"]
    assert "transient_frame_ids" not in planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=4)


@pytest.mark.parametrize("change, reason", [
    ({"transient_frame_ids": ["missing"]}, "transients_invalid"),
    ({"transient_frame_ids": ["0"], "automation": None}, "transients_require_automation"),
])
def test_transients_cannot_be_unusable_unknown_or_unplanned(tmp_path, change, reason):
    _, survey, proposal, policy = fixture(tmp_path)
    proposal.update(static_intervals_seconds=[[0, 2.5]], unsafe_frame_ids=[], automation=_automation(),
                    transient_frame_ids=[], transients=[])
    proposal.update(change)
    if proposal["automation"] is None:
        del proposal["automation"]
    with pytest.raises(ValueError, match=reason):
        planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=4)


def test_an_unsafe_frame_showing_people_is_left_out_not_a_failed_plan(tmp_path):
    """2026-10-23: hands on 22 frames of a different rack state failed a whole plan."""
    _, survey, proposal, policy = fixture(tmp_path)
    proposal.update(static_intervals_seconds=[[0, 2.5]], unsafe_frame_ids=["2"], automation=_automation(),
                    transient_frame_ids=["1", "2"],
                    transients=[{"label": "hands", "segmentation_prompt": "hands"}])
    review = planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=4)
    # The unsafe frame is excluded from the inputs, so it has nothing to erase.
    assert review["excluded_frame_ids"] == ["2"]
    assert review["transient_frame_ids"] == ["1"]
    # Unsafe frames that are all transient leave no transient to erase.
    proposal["transient_frame_ids"] = ["2"]
    assert "transient_frame_ids" not in planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=4)


def test_transient_targets_never_collide_with_reviewed_objects(tmp_path):
    _, survey, proposal, policy = fixture(tmp_path)
    automation = _automation()
    automation["objects"][0]["target_id"] = "transient-0"
    proposal.update(static_intervals_seconds=[[0, 1.5]], automation=automation, transient_frame_ids=["0"])
    with pytest.raises(ValueError, match="transients_invalid"):
        planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=4)


def test_budget_fitted_export_review_is_admitted_by_a_budgeted_invoker(tmp_path, monkeypatch):
    # 2026-10-24 live run: the fitted request was checked with the unfitted quote and refused.
    path, _, _, policy = fixture(tmp_path)
    from blueprint_pipeline import openai_prompt_cache
    from blueprint_pipeline import paid_resource_admission as admission
    from blueprint_pipeline import website_task_context as control
    monkeypatch.setattr(admission, "require_paid_resource_admission_grant", lambda *_, **__: None)
    monkeypatch.setattr(control, "reserve_website_preparation_spend", lambda **kwargs: ({}, object()))
    monkeypatch.setattr(control, "website_webapp_request", lambda **kwargs: {"status": "settled"})
    quotes = []
    monkeypatch.setattr(openai_prompt_cache, "worst_case_reservation_usd",
                        lambda **kw: quotes.append(kw) or (10.0 if len(quotes) == 1 else .1))
    proposal = {"status": "proposed", "selected_camera_ids": ["a-wide", "b-lower"],
                "gap_assessment": ["Complete floor"], "remaining_unknowns": ["Hidden wall"],
                "scene_caption": "The same room with a continuous bare floor.",
                "view_gap_reasons": [{"camera_id": "a-wide", "reason": "Keep the task anchor"},
                                     {"camera_id": "b-lower", "reason": "Complete nearby floor"}],
                "environment_surfaces": [{"component": "support assembly", "surface": "end face",
                    "required_for_local_environment": True, "coverage": "material_gap",
                    "source_frame_ids": ["0"], "evidence_camera_ids": ["a-wide"], "gap_ids": [],
                    "reason": "A large part of the photographed support is absent from the export"}]}

    class BudgetedInvoker:
        config = SimpleNamespace(max_inference_cost_usd=policy["maximum_review_cost_usd"])

        def configure_reservation_audit(self, **_):
            pass

        def invoke(self, spec, input_value):
            return SimpleNamespace(output=proposal, cost_usd=.03, usage={"input_tokens": 100},
                                   model=spec.model, sdk_version="test")

    result = planner.review_survey(
        survey_path=path, task_context={"description": "test task", "context_digest": "sha256:" + "b" * 64,
            "request_id": "r", "scene_id": "s", "capture_id": "c"}, configuration={"policy": policy},
        output_root=tmp_path / "plans", invoker=BudgetedInvoker(), release_blockers=lambda: ("c" * 40, []),
        completion={"previews": [], "candidate_camera_ids": ["a-wide", "b-lower"], "post_export_coverage": {}})
    assert result["status"] == "completed" and quotes[0] != quotes[-1]

@pytest.mark.parametrize("rotation", [0, 90, 45])
def test_dense_input_preflight_prices_actual_rotated_previews_without_resizing_further(tmp_path, rotation):
    image=tmp_path / "recorded.png"
    Image.new("RGB", (1080,1920), "white").save(image)
    survey={"display_rotation_degrees":rotation,"frames":[{"frame_id":"source-0","source_image_path":str(image),
        "source_image_digest":_sha256_file(image)}]}
    payload,tokens,quote=planner.preflight_review_input(survey,{"description":"Review the entire captured room"},None,
        model="gpt-6.1-sol",maximum=1.,output_tokens=4000,instructions=planner.INSTRUCTIONS,schema=planner.CurrentSceneProposal,
        hosted_tool_cost_usd=.02)
    import base64
    from io import BytesIO
    part=next(p for p in payload[0]["content"] if p["type"]=="input_image")
    with Image.open(BytesIO(base64.b64decode(part["image_url"].partition(",")[2]))) as preview:
        assert max(preview.size)==512
        assert min(preview.size)==(512 if rotation==45 else 288)
    assert tokens>24_000 and quote+.02<=1
    assert _sha256_file(image)==survey["frames"][0]["source_image_digest"]


def test_dense_review_source_change_refuses_before_any_paid_reservation(tmp_path, monkeypatch):
    path,survey,_,policy=fixture(tmp_path)
    survey["frames"]=[{**survey["frames"][0],"frame_id":str(i)} for i in range(173)]
    survey["digest"]=canonical_digest(survey,digest_field="digest")
    path.write_text(json.dumps(survey))
    Image.new("RGB",(20,30),"blue").save(survey["frames"][0]["source_image_path"])
    from blueprint_pipeline import website_task_context as control
    monkeypatch.setattr(control,"reserve_website_preparation_spend",lambda **_:pytest.fail("changed input reserved spend"))
    with pytest.raises(ValueError,match="source_changed"):
        planner.review_survey(survey_path=path,task_context={"description":"Read the room"},configuration={"policy":policy},
            output_root=tmp_path/"plans")
