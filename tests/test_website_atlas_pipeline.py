"""Production camera coverage and routing never imply simulator readiness."""
from copy import deepcopy
import json

import pytest

from blueprint_pipeline.website_atlas_pipeline import generation_request, atlas_selected


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


def test_production_rollout_is_capture_specific_and_malformed_config_refuses(monkeypatch):
    monkeypatch.setenv("BLUEPRINT_WEBSITE_ATLAS_CAPTURE_IDS_JSON", '["admitted"]')
    assert atlas_selected("admitted") and not atlas_selected("another")
    monkeypatch.setenv("BLUEPRINT_WEBSITE_ATLAS_CAPTURE_IDS_JSON", '{"admitted":true}')
    with pytest.raises(ValueError, match="rollout_invalid"):
        atlas_selected("admitted")


@pytest.mark.parametrize("status", ["processing", "needs_input"])
def test_rgbd_result_never_publishes_a_completed_scene_handoff(tmp_path, status):
    from blueprint_pipeline.website_atlas_pipeline import retain_site_preparation_hold
    from blueprint_pipeline.common import StageError
    result = {"status": status, "blockers": ["atlas_public_splat_creation_api_unavailable"]}
    with pytest.raises(StageError):
        retain_site_preparation_hold(pipeline_dir=tmp_path, atlas_result=result, scene_id="s", capture_id="c")
    assert json.loads((tmp_path / "atlas_preparation_result.json").read_text()) == result
    handoff = tmp_path / "website_scene_preparation/handoff.json"
    if status == "processing":
        assert not handoff.exists()
    else:
        value = json.loads(handoff.read_text())
        assert value["status"] == "needs_input" and value["simulator_ready"] is False
        assert value["atlas_preparation"] == result and value["capture_id"] == "c"


@pytest.mark.parametrize("over_budget", [False, True])
def test_worker_densifies_reviews_then_resumes_pose_without_replanning(tmp_path, monkeypatch, over_budget):
    from blueprint_pipeline import website_atlas_pipeline as pipeline, atlas_video_inputs as inputs
    from blueprint_pipeline import atlas_video_extension as extension, paid_resource_allocator as allocator
    from blueprint_pipeline import atlas_result_bundle as bundles
    from blueprint_pipeline import website_atlas_pose as pose_api
    monkeypatch.setattr(pose_api, "atlas_api_config", lambda: ("https://example.test", "test-non-secret"))
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    video = tmp_path / "source.mov"
    video.write_bytes(b"immutable test capture")
    root = tmp_path / "job"
    policy = {"model": "gpt-6.1-sol", "maximum_review_cost_usd": .5, "pose_input_cost_usd": .01,
        "rgbd_view_cost_usd": .05, "rate_card_reference": "https://example.test/rates", "reviewed_by": "test",
        "reviewed_on": "2026-10-21"}
    config = {"authority": {"provider_retention": "project_library_retention_approved"},
        "policy": policy, "remaining_cost_usd": 1.14 if over_budget else 4.76}
    monkeypatch.setattr(pipeline, "load_configuration", lambda *_: config)
    calls = {"review": 0, "pose": 0, "generation": 0}
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
        calls["generation"] += 1
        assert kwargs["price_review"]["maximum_cost_usd"] == .1
        assert kwargs["price_review"]["request_digest"] == canonical_digest(kwargs["request"])
        return {"status": "completed", "digest": "generated"}
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_task", generate)
    monkeypatch.setattr(bundles, "export_pose_bundle", lambda **_: {"zip_path": "retained.zip"})
    options = {"source_video": video, "output_root": root, "task_context": {"capture_id": "c", "context_digest": "task"}}
    if over_budget:
        with pytest.raises(ValueError, match="budget_insufficient_for_pose_and_generation"):
            pipeline.run_atlas_pipeline(**options)
        assert calls == {"review": 2, "pose": 0, "generation": 0}
        return
    assert pipeline.run_atlas_pipeline(**options)["stage"] == "pose"
    result = pipeline.run_atlas_pipeline(**options)
    assert calls == {"review": 2, "pose": 2, "generation": 1}
    assert result["status"] == "needs_input" and result["simulation_qualified"] is False
    assert result["task_and_surrounding_scene_covered"] is True
    assert result["blockers"][0] == "atlas_public_splat_creation_api_unavailable"
