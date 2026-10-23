"""Production camera coverage and routing never imply simulator readiness."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from blueprint_pipeline.website_atlas_pipeline import generation_request


def _reviewed_today():
    # The pipeline checks the review's age against the real date, so fixtures
    # must not carry a fixed day that expires a month later.
    return datetime.now(timezone.utc).date().isoformat()


def frame(x):
    return {"imageAsset": {"assetId": "asset_" + str(x)}, "depth": {"depthAsset": {"assetId": "depth_" + str(x)}},
        "camera": {"intrinsics": {"width": 720, "height": 1280, "fx": 1000, "fy": 1000, "cx": 360, "cy": 640},
                   "extrinsics": {"position": [x, 0, 0], "quaternion": [0, 0, 0, 1], "coordinateSystem": "rub"}}}


def test_generated_depth_targets_keep_task_and_room_and_never_extrapolate_source_camera_positions():
    pose = {"operation": {"id": "operation", "done": True, "response": {"frames": [frame(0), frame(1)]}}}
    original = deepcopy(pose)
    request = generation_request(pose=pose, inputs={"frames": [{"frame_id": "task"}, {"frame_id": "room"}]},
        review={"scene_state": "same object", "coverage_requirements": [
            {"role": "task_object", "required_frame_ids": ["task"]},
            {"role": "surrounding_scene", "required_frame_ids": ["room"]}]})
    assert [r["extrinsics"]["position"] for r in request["targetCameras"]] == [[0, 0, 0], [1, 0, 0]]
    assert all(r["intrinsics"]["width"] == 1280 for r in request["targetCameras"])
    assert request["returnDepth"] and pose == original


def test_refinement_reviews_extended_candidates_and_resumes_without_rebuying_review(tmp_path, monkeypatch):
    from blueprint_pipeline import website_atlas_pipeline as pipeline, atlas_refinement as refinement
    from blueprint_pipeline import atlas_video_extension as extension, atlas_video_inputs as inputs
    from blueprint_pipeline import paid_resource_allocator as allocator, website_atlas_pose as pose_api
    video = tmp_path / "video.mov"
    video.write_bytes(b"source")
    parent = tmp_path / "parent"
    (parent / "survey/source").mkdir(parents=True)
    (parent / "survey/source/decoded_observation_index.json").write_text("{}")
    proposal = {"status": "proposed", "scene_state": "unchanged", "static_intervals_seconds": [[0, 3.5]],
        "anchor_frame_id": "1", "coverage_requirements": [
            {"role": "task_object", "label": "object", "required_frame_ids": ["1"]},
            {"role": "surrounding_scene", "label": "room", "required_frame_ids": ["0"]}],
        "unsafe_frame_ids": [], "uncaptured_regions": ["rear"], "summary": "more angles help rack edges",
        "selection_policy": {"minimum_sharpness": 20, "maximum_clipped_fraction": .5, "minimum_overlap_inliers": 12},
        "action": "refine", "gap_assessment": ["Frame 2 covers rack edges"], "additional_input_frame_ids": ["2"]}
    feedback = {"parent_root": str(parent), "digest": "feedback", "parent_result_digest": "parent-result",
                "prior_review": proposal, "prior_selected_frame_ids": ["0", "1"]}
    monkeypatch.setattr(refinement, "load_refinement_feedback", lambda *a, **k: feedback)
    policy = {"model": "gpt-6.1-sol", "maximum_review_cost_usd": .5, "pose_input_cost_usd": .01,
        "rgbd_view_cost_usd": .05, "mesh_export_cost_usd": .5, "rate_card_reference": "test", "reviewed_by": "test",
        "reviewed_on": _reviewed_today()}
    monkeypatch.setattr(pipeline, "load_configuration", lambda *a: {"policy": policy, "remaining_cost_usd": 3.4,
        "authority": {"provider_retention": "project_library_retention_approved"}})
    monkeypatch.setattr(pose_api, "atlas_api_config", lambda: ("url", "secret"))
    def extend(**kwargs):
        path = kwargs["output_root"]
        path.mkdir()
        (path / "atlas_video_survey.json").write_text(json.dumps({"digest": "survey",
            "source_video_digest": pipeline._sha256_file(video), "frames": [
                {"frame_id": str(i), "timestamp_seconds": i} for i in range(3)]}))
    monkeypatch.setattr(extension, "extend_survey", extend)
    reviews = []
    def plan(**kwargs):
        reviews.append(kwargs)
        return {"proposal": proposal, "model": policy["model"]}
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_plan", plan)
    monkeypatch.setattr(inputs, "prepare_video_inputs", lambda **k: {"frames": [{"frame_id": str(i)} for i in range(3)]})
    poses = []
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_pose", lambda **k: poses.append(k) or {
        "status": "processing", "operation_id": "new-pose"})
    options = dict(source_video=video, output_root=tmp_path / "revision", task_context={"capture_id": "c", "context_digest": "ctx"},
                   refinement_feedback=tmp_path / "feedback.json")
    assert pipeline.run_atlas_pipeline(**options)["stage"] == "pose"
    assert pipeline.run_atlas_pipeline(**options)["stage"] == "pose"
    assert len(reviews) == 1 and reviews[0]["feedback"] is feedback and len(poses) == 2
    with pytest.raises(ValueError, match="requires_new_job"):
        pipeline.run_atlas_pipeline(**{**options, "output_root": parent})


@pytest.mark.parametrize("automation_enabled", [False, True])
@pytest.mark.parametrize("mesh_price", [None, .5])
@pytest.mark.parametrize("over_budget", [False, True])
def test_worker_densifies_reviews_then_resumes_pose_without_replanning(tmp_path, monkeypatch, over_budget, mesh_price, automation_enabled):
    from blueprint_pipeline import website_atlas_pipeline as pipeline, atlas_video_inputs as inputs
    from blueprint_pipeline import atlas_video_extension as extension, paid_resource_allocator as allocator
    from blueprint_pipeline import atlas_result_bundle as bundles, atlas_depth_splat as depth_splat
    from blueprint_pipeline import website_atlas_pose as pose_api
    monkeypatch.setattr(pose_api, "atlas_api_config", lambda: ("https://example.test", "test-non-secret"))
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    video = tmp_path / "source.mov"
    video.write_bytes(b"immutable test capture")
    root = tmp_path / "job"
    policy = {"model": "gpt-6.1-sol", "maximum_review_cost_usd": .5, "pose_input_cost_usd": .01,
        "rgbd_view_cost_usd": .05, "rate_card_reference": "https://example.test/rates", "reviewed_by": "test",
        "reviewed_on": _reviewed_today()}
    if mesh_price:
        policy["mesh_export_cost_usd"] = mesh_price
    config = {"authority": {"provider_retention": "project_library_retention_approved"},
        "policy": policy, "remaining_cost_usd": (1.14 if over_budget else 4.76) + (mesh_price or 0)}
    monkeypatch.setattr(pipeline, "load_configuration", lambda *_: config)
    calls = {"review": 0, "pose": 0, "atlasGenerate": 0, "splat": 0, "splats2Mesh": 0}
    def survey(**kwargs):
        path = kwargs["output_root"]
        (path / "source").mkdir(parents=True)
        (path / "source/decoded_observation_index.json").write_text("{}")
        value = {"source_video_digest": pipeline._sha256_file(video), "frames": [
            {"frame_id": str(i), "timestamp_seconds": i} for i in range(2)], "digest": "base"}
        (path / "atlas_video_survey.json").write_text(json.dumps(value))
    monkeypatch.setattr(inputs, "survey_video", survey)
    proposal = {"status": "proposed", "scene_state": "open appliance", "static_intervals_seconds": [[0, 1.5]],
        "anchor_frame_id": "1", "coverage_requirements": [
            {"role": "task_object", "label": "object", "required_frame_ids": ["1"]},
            {"role": "surrounding_scene", "label": "room", "required_frame_ids": ["0"]}],
        "unsafe_frame_ids": [], "uncaptured_regions": ["rear"], "summary": "reviewed",
        "selection_policy": {"minimum_sharpness": 20, "maximum_clipped_fraction": .5, "minimum_overlap_inliers": 24}}
    automated = []
    if automation_enabled:
        proposal["automation"] = {"scale_references": [], "objects": [], "removal_instruction": "", "mask_margin_fraction": .01}
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_automation",
        lambda **kw: automated.append(kw) or {"status": "relative_sizes_only", "measurements": {"status": "relative_sizes_only"}})
    def review(**kwargs):
        calls["review"] += 1
        return {"proposal": proposal, "model": policy["model"]}
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_plan", review)
    def extend(**kwargs):
        path = kwargs["output_root"]
        path.mkdir()
        value = json.loads(kwargs["base_survey"].read_text())
        value["digest"] = "extended"
        (path / "atlas_video_survey.json").write_text(json.dumps(value))
    monkeypatch.setattr(extension, "extend_survey", extend)
    def prepare(**kwargs):
        if kwargs["review"]["survey_digest"] == "base":
            raise ValueError("atlas_selection_coverage_gap")
        return {"frames": [{"frame_id": str(i)} for i in range(5 if over_budget else 2)]}
    monkeypatch.setattr(inputs, "prepare_video_inputs", prepare)
    def pose(**kwargs):
        calls["pose"] += 1
        if calls["pose"] == 1:
            return {"status": "processing", "operation_id": "pose"}
        value = {"operation": {"id": "pose", "done": True, "response": {"frames": [frame(0), frame(1)]}}}
        (kwargs["output_root"] / "atlas_posed_rgbd_result.json").write_text(json.dumps(value))
        return {"status": "completed", "operation_id": "pose"}
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_pose", pose)
    def generate(**kwargs):
        calls[kwargs["task"]] += 1
        assert kwargs["price_review"]["request_digest"] == canonical_digest(kwargs["request"])
        if kwargs["task"] == "splats2Mesh":
            assert kwargs["price_review"]["maximum_cost_usd"] == mesh_price
            assert kwargs["uploads"]["splat"]["path"].endswith("atlas_depth_splat.spz")
            kwargs["output_root"].mkdir(parents=True, exist_ok=True)
            (kwargs["output_root"] / "result.json").write_text(json.dumps(
                {"outputs": [{"role": "mesh", "path": "outputs/mesh.glb"}]}))
            return {"status": "completed", "digest": "mesh"}
        assert kwargs["price_review"]["maximum_cost_usd"] == .1
        return {"status": "completed", "digest": "generated"}
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_task", generate)
    def splat(pose_root, splat_root):
        calls["splat"] += 1
        splat_root.mkdir(parents=True)
        (splat_root / "splats2mesh_task_spec.json").write_text(json.dumps({
            "request": {"splatAsset": {"assetId": "upload:splat"}},
            "uploads": {"splat": {"path": str(splat_root / "atlas_depth_splat.spz"), "sha256": "sha256:" + "0" * 64}}}))
        manifest = {"splat_kind": "depth_fused_draft", "digest": "splat"}
        (splat_root / "atlas_depth_splat_manifest.json").write_text(json.dumps(manifest))
        return manifest
    monkeypatch.setattr(depth_splat, "build_depth_splat", splat)
    from blueprint_pipeline import atlas_mesh_registration as mesh_registration
    registered = []
    monkeypatch.setattr(mesh_registration, "register_atlas_mesh", lambda **kw: registered.append(kw))
    exported = {}
    monkeypatch.setattr(bundles, "export_pose_bundle", lambda **kw: exported.update(kw) or {"zip_path": "retained.zip"})
    stages = []
    options = {"source_video": video, "output_root": root, "task_context": {"capture_id": "c", "context_digest": "task"},
               "checkpoint": stages.append}
    if over_budget:
        with pytest.raises(ValueError, match="budget_insufficient_for_pose_and_generation"):
            pipeline.run_atlas_pipeline(**options)
        assert calls == {"review": 2, "pose": 0, "atlasGenerate": 0, "splat": 0, "splats2Mesh": 0}
        return
    assert pipeline.run_atlas_pipeline(**options)["stage"] == "pose"
    result = pipeline.run_atlas_pipeline(**options)
    assert calls == {"review": 2, "pose": 2, "atlasGenerate": 1, "splat": 1, "splats2Mesh": 1 if mesh_price else 0}
    assert bool(automated) is automation_enabled
    # A withdrawal is re-checked before every paid stage, including each resumed one.
    assert stages == ["plan", "plan", "pose", "pose", "generated_rgbd"] + (["mesh"] if mesh_price else [])
    assert result["status"] == "needs_input" and result["simulation_qualified"] is False
    assert result["task_and_surrounding_scene_covered"] is True
    assert result["splat_available"] is True and result["mesh_available"] is bool(mesh_price)
    assert (exported["mesh_root"] is not None) is bool(mesh_price) and exported["splat_root"] == root / "splat"
    assert [str(kw["glb"]) for kw in registered] == ([str(root / "mesh/outputs/mesh.glb")] if mesh_price else [])
    # No reviewed mesh price means no mesh purchase, never an implied mesh.
    assert result["blockers"] == ([] if mesh_price else ["atlas_mesh_export_price_unreviewed"]) + [
        "metric_scene_registration_unproven"]
    if mesh_price is None:
        # A price reviewed later resumes the job and buys only the mesh.
        policy["mesh_export_cost_usd"] = .5
        mesh_price = .5
        resumed = pipeline.run_atlas_pipeline(**options)
        assert resumed["mesh_available"] is True and calls["review"] == 2 and calls["splats2Mesh"] == 1
        policy["pose_input_cost_usd"] = .02
        with pytest.raises(ValueError, match="authority_or_policy_changed"):
            pipeline.run_atlas_pipeline(**options)


@pytest.mark.parametrize("review", [
    {"automation": {"objects": [{"disposition": "remove"}]}},
    # One video of people at work: nothing to remove but the people in the edited views.
    {"automation": {"objects": [{"disposition": "keep"}]}, "transient_frame_ids": ["0"],
     "coverage_requirements": [{"role": "task_object", "required_frame_ids": ["0"]}]},
])
def test_removal_automation_precedes_every_atlas_submission(tmp_path, monkeypatch, review):
    from blueprint_pipeline import website_atlas_pipeline as pipeline, atlas_video_inputs as inputs
    from blueprint_pipeline import paid_resource_allocator as allocator, website_atlas_pose as pose_api

    root = tmp_path / "job"
    (root / "survey").mkdir(parents=True)
    video = tmp_path / "source.mov"
    video.write_bytes(b"source")
    survey = {"source_video_digest": pipeline._sha256_file(video)}
    (root / "survey/atlas_video_survey.json").write_text(json.dumps(survey))
    (root / "selection_binding.json").write_text(json.dumps({"survey_relative_path": "survey/atlas_video_survey.json",
        "review": review, "planner_receipt": {"model": "agent"}}))
    config = {"authority": {"provider_retention": "approved"}, "remaining_cost_usd": 10,
              "policy": {"maximum_review_cost_usd": 1, "pose_input_cost_usd": .01, "rgbd_view_cost_usd": .05}}
    monkeypatch.setattr(pipeline, "load_configuration", lambda *_: config)
    monkeypatch.setattr(pose_api, "atlas_api_config", lambda: ("https://example.test", "non-secret"))
    monkeypatch.setattr(inputs, "prepare_video_inputs", lambda **_: {"frames": []})
    for name in ("run_sponsored_website_atlas_pose", "run_sponsored_website_atlas_task"):
        monkeypatch.setattr(allocator, name, lambda **_: pytest.fail("unreviewed RGB reached Atlas"))
    seen = []
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_automation", lambda **kw:
        seen.append(kw) or {"status": "quality_rejected", "background": {"status": "quality_rejected"}})
    result = pipeline.run_atlas_pipeline(source_video=video, output_root=root,
                                        task_context={"capture_id": "c", "context_digest": "task"})
    assert seen[0]["pose_root"] is None and not result["splat_available"]
    assert result["blockers"] == ["atlas_background_review_required"]


@pytest.mark.parametrize("refused", [False, True])
def test_a_completed_background_buys_the_original_views_pose_once_for_scene_preparation(tmp_path, monkeypatch, refused):
    from blueprint_pipeline import website_atlas_pipeline as pipeline, atlas_video_inputs as inputs
    from blueprint_pipeline import paid_resource_allocator as allocator, website_atlas_pose as pose_api
    from blueprint_pipeline import atlas_scene_measurements as measurements, atlas_scene_automation as automation
    from blueprint_pipeline import atlas_source_pose_inputs, atlas_context_fov
    monkeypatch.setattr(atlas_source_pose_inputs, "prepare_source_pose", lambda root: (
        root, {"frames": [{"frame_id": "a"}, {"frame_id": "b"}]}))
    monkeypatch.setattr(atlas_context_fov, "verify_pose_context_fov", lambda **_: None)

    root = tmp_path / "job"
    (root / "survey").mkdir(parents=True)
    video = tmp_path / "source.mov"
    video.write_bytes(b"source")
    (root / "survey/atlas_video_survey.json").write_text(json.dumps({"source_video_digest": pipeline._sha256_file(video)}))
    review = {"automation": {"objects": [{"disposition": "remove"}]}}
    (root / "selection_binding.json").write_text(json.dumps({"survey_relative_path": "survey/atlas_video_survey.json",
        "review": review, "planner_receipt": {"model": "agent"}}))
    policy = {"maximum_review_cost_usd": 1, "pose_input_cost_usd": .01, "rgbd_view_cost_usd": .05,
              "rate_card_reference": "https://example.test/rates", "reviewed_by": "test",
              "reviewed_on": _reviewed_today()}
    config = {"authority": {"provider_retention": "approved"}, "remaining_cost_usd": 10, "policy": policy}
    monkeypatch.setattr(pipeline, "load_configuration", lambda *_: config)
    monkeypatch.setattr(pose_api, "atlas_api_config", lambda: ("https://example.test", "non-secret"))
    monkeypatch.setattr(inputs, "prepare_video_inputs", lambda **_: {"frames": [{"frame_id": "a"}, {"frame_id": "b"}]})
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_automation", lambda **kw: {
        "status": "completed", "measurements": {"status": "relative_sizes_only"},
        "background": {"status": "completed", "splat_manifest_digest": "splat", "mesh_available": True}})
    quotes, poses = [], []

    def pose(**kw):
        quotes.append(json.loads(pipeline.os.environ[pose_api.ATLAS_PRICE_REVIEW_ENV]))
        if refused:
            raise ValueError("website_atlas_allowance_exhausted")
        poses.append(kw)
        if len(poses) == 1:
            return {"status": "processing", "operation_id": "source"}
        (kw["output_root"] / "atlas_posed_rgbd_result.json").write_text("{}")
        return {"status": "completed", "operation_id": "source"}
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_pose", pose)
    monkeypatch.setattr(automation, "measurement_plan", lambda review: {"plan": "reviewed"})
    monkeypatch.setattr(measurements, "load_views", lambda root, value: (
        {"digest": "sha256:pose", "operation": {"id": "source"}}, ["views"]))
    measured, segmented = [], []
    monkeypatch.setattr(measurements, "estimate_sizes", lambda **kw: measured.append(kw) or {
        "digest": "sha256:sizes", "meters_per_unit_estimate": 0.8, "scale_status": "estimated_from_named_assumptions"})
    from blueprint_pipeline import atlas_reference_masks
    monkeypatch.setattr(atlas_reference_masks, "segment_scale_references",
                        lambda **kw: segmented.append(kw) or ["reference masks"])
    stages = []
    options = {"source_video": video, "output_root": root, "checkpoint": stages.append,
               "task_context": {"capture_id": "c", "context_digest": "task"}}
    first = pipeline.run_atlas_pipeline(**options)
    assert quotes[0]["max_input_images"] == 2 and quotes[0]["maximum_cost_usd"] == .02
    assert stages[-1] == "source_pose"
    if refused:
        # A refused purchase keeps the finished background and says why.
        assert first["source_pose"] == {"status": "needs_input", "reason": "website_atlas_allowance_exhausted"}
        assert first["splat_available"] is True and json.loads((root / "result.json").read_text()) == first
        return
    assert first == {"status": "processing", "stage": "source_pose", "operation_id": "source"}
    result = pipeline.run_atlas_pipeline(**options)
    assert result["source_pose"] == {"status": "completed", "result_digest": "sha256:pose", "operation_id": "source",
        "pose_relative_path": "pose",
        "submitted_frame_count": 2, "scene_measurements_digest": "sha256:sizes", "meters_per_unit_estimate": 0.8,
        "scale_status": "estimated_from_named_assumptions"}
    assert measured[0]["automation"] == {"plan": "reviewed"}
    assert measured[0]["output"] == root / "pose" / "scene_measurements.json"
    # Each reference is checked against its named object's SAM mask on the posed views.
    assert segmented[0]["frames"] == ["views"] and segmented[0]["references"] == []
    assert segmented[0]["root"] == root / "pose" / "reference_masks"
    segmented[0]["checkpoint"]("reference_segmentation")  # the job's own withdrawal check
    assert stages[-1] == "reference_segmentation"
    assert measured[0]["reference_masks"] == ["reference masks"]
    assert [kw["output_root"] for kw in poses] == [root / "pose"] * 2
    # The retained pose resumes without another withdrawal check or quote.
    pipeline.run_atlas_pipeline(**options)
    assert len(poses) == 3 and stages.count("source_pose") == 2


def test_a_checkpoint_refusal_stops_the_run_before_the_first_paid_review(tmp_path, monkeypatch):
    from blueprint_pipeline import website_atlas_pipeline as pipeline, atlas_video_inputs as inputs
    from blueprint_pipeline import paid_resource_allocator as allocator, website_atlas_pose as pose_api
    video = tmp_path / "source.mov"
    video.write_bytes(b"capture")
    monkeypatch.setattr(pose_api, "atlas_api_config", lambda: ("https://example.test", "test-non-secret"))
    monkeypatch.setattr(pipeline, "load_configuration", lambda *_: {"authority": {"provider_retention":
        "account_does_not_retain"}, "remaining_cost_usd": 2, "policy": {"maximum_review_cost_usd": .1,
        "pose_input_cost_usd": .01, "rgbd_view_cost_usd": .05}})

    def survey(**kwargs):
        kwargs["output_root"].mkdir(parents=True)
        (kwargs["output_root"] / "atlas_video_survey.json").write_text(json.dumps(
            {"source_video_digest": pipeline._sha256_file(video), "frames": []}))
    monkeypatch.setattr(inputs, "survey_video", survey)
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_plan",
                        lambda **_: pytest.fail("a withdrawn capture must not buy a review"))

    def withdrawn(stage):
        raise RuntimeError(f"withdrawn before {stage}")
    with pytest.raises(RuntimeError, match="withdrawn before plan"):
        pipeline.run_atlas_pipeline(source_video=video, output_root=tmp_path / "job", checkpoint=withdrawn,
                                    task_context={"capture_id": "c", "context_digest": "task"})


def test_a_job_stopped_by_a_lapsed_price_review_resumes_once_the_owner_renews_it(tmp_path, monkeypatch):
    from blueprint_pipeline import website_atlas_pipeline as pipeline, atlas_video_inputs as inputs
    from blueprint_pipeline import paid_resource_allocator as allocator, website_atlas_pose as pose_api
    video = tmp_path / "source.mov"
    video.write_bytes(b"capture")
    today = datetime.now(timezone.utc).date()
    stale = today - timedelta(days=pose_api.PRICE_REVIEW_MAX_AGE_DAYS + 1)
    policy = {"maximum_review_cost_usd": .1, "pose_input_cost_usd": .01, "rgbd_view_cost_usd": .05,
              "rate_card_reference": "https://example.test/rates", "reviewed_by": "owner",
              "reviewed_on": stale.isoformat()}
    live = {"authority": {"provider_retention": "account_does_not_retain"}, "remaining_cost_usd": 2,
            "policy": policy}
    monkeypatch.setattr(pose_api, "atlas_api_config", lambda: ("https://example.test", "test-non-secret"))
    monkeypatch.setattr(pipeline, "load_configuration", lambda *_: deepcopy(live))

    def survey(**kwargs):
        kwargs["output_root"].mkdir(parents=True)
        (kwargs["output_root"] / "atlas_video_survey.json").write_text(json.dumps(
            {"source_video_digest": pipeline._sha256_file(video), "frames": []}))
    monkeypatch.setattr(inputs, "survey_video", survey)
    bought = []

    def plan(**kwargs):
        bought.append(kwargs["configuration"]["policy"])
        kwargs["output_root"].mkdir(parents=True, exist_ok=True)
        raise RuntimeError("scene review bought")
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_plan", plan)
    job = dict(source_video=video, output_root=tmp_path / "job",
               task_context={"capture_id": "c", "context_digest": "task"})
    # Frame selection refuses a stale review only after the paid scene review, so
    # every upload after the review lapses would pay for nothing.
    with pytest.raises(ValueError, match="atlas_runtime_price_review_expired"):
        pipeline.run_atlas_pipeline(**job)
    assert bought == []
    # The owner renews the review, here with a new price as well. Nothing was bought,
    # so the waiting job takes the current policy whole rather than failing for good.
    live["policy"] = {**policy, "maximum_review_cost_usd": .2, "reviewed_on": today.isoformat()}
    with pytest.raises(RuntimeError, match="scene review bought"):
        pipeline.run_atlas_pipeline(**job)
    assert bought == [live["policy"]]
    assert json.loads((tmp_path / "job/configuration.json").read_text())["policy"] == live["policy"]
    # After a purchase, a renewed record of the same prices is still adopted; a
    # changed price would rebind what was bought, so it refuses.
    live["policy"] = {**live["policy"], "reviewed_by": "second owner"}
    with pytest.raises(RuntimeError, match="scene review bought"):
        pipeline.run_atlas_pipeline(**job)
    assert bought[-1]["reviewed_by"] == "second owner"
    live["policy"] = {**live["policy"], "maximum_review_cost_usd": .3}
    with pytest.raises(ValueError, match="atlas_runtime_authority_or_policy_changed"):
        pipeline.run_atlas_pipeline(**job)
    assert len(bought) == 2


def test_every_first_run_densifies_the_still_interval_within_the_review_image_budget(tmp_path, monkeypatch):
    from blueprint_pipeline import website_atlas_pipeline as pipeline, atlas_video_extension as extension
    base_root = tmp_path / "job/survey"
    (base_root / "source").mkdir(parents=True)
    (base_root / "source/decoded_observation_index.json").write_text("{}")
    calls = []

    def extend(**kwargs):
        calls.append(kwargs)
        kwargs["output_root"].mkdir(parents=True, exist_ok=True)
        (kwargs["output_root"] / "atlas_video_survey.json").write_text("{}")
    monkeypatch.setattr(extension, "extend_survey", extend)
    options = dict(source_video=tmp_path / "video.mov", base_root=base_root, base_survey=base_root / "atlas_video_survey.json",
                   output_root=tmp_path / "job")
    path = pipeline.accuracy_survey(review={"static_intervals_seconds": [[0, 8.1]]}, **options)
    assert path == tmp_path / "job/extended_survey/atlas_video_survey.json"
    assert (calls[0]["step_seconds"], calls[0]["gap_threshold_seconds"]) == (.1, .2)
    pipeline.accuracy_survey(review={"static_intervals_seconds": [[0, 8.1]]}, **options)
    assert len(calls) == 1  # a resumed job reuses the densified survey and its review
    path.unlink()
    pipeline.accuracy_survey(review={"static_intervals_seconds": [[0, 40], [50, 70]]}, **options)
    assert calls[1]["step_seconds"] == 60 / pipeline.MAXIMUM_REVIEW_FRAMES == .2  # long stillness stays reviewable


def test_a_job_keeps_the_edit_scope_it_started_with(tmp_path):
    from blueprint_pipeline import website_atlas_pipeline as pipeline

    fresh = tmp_path / "default"
    assert pipeline._job_edit_scope(fresh, None) == "required_and_additional"
    assert pipeline._job_edit_scope(fresh, "required_and_additional") == "required_and_additional"
    with pytest.raises(ValueError, match="edit_scope_changed"):
        pipeline._job_edit_scope(fresh, "all_selected_views")
    trial = tmp_path / "trial"
    assert pipeline._job_edit_scope(trial, "all_selected_views") == "all_selected_views"
    # A resume without the option keeps the trial's scope.
    assert pipeline._job_edit_scope(trial, None) == "all_selected_views"
    with pytest.raises(ValueError, match="edit_scope_invalid"):
        pipeline._job_edit_scope(tmp_path / "bad", "everything")
