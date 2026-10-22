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


@pytest.mark.parametrize("mesh_price", [None, .5])
def test_worker_densifies_reviews_then_resumes_pose_without_replanning(tmp_path, monkeypatch, mesh_price):
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
        "reviewed_on": "2026-10-21"}
    if mesh_price:
        policy["mesh_export_cost_usd"] = mesh_price
    config = {"authority": {"provider_retention": "project_library_retention_approved"},
        "policy": policy, "remaining_cost_usd": 4.76}
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
        return {"frames": [{"frame_id": "0"}, {"frame_id": "1"}]}
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
    options = {"source_video": video, "output_root": root, "task_context": {"capture_id": "c", "context_digest": "task"}}
    assert pipeline.run_atlas_pipeline(**options)["stage"] == "pose"
    result = pipeline.run_atlas_pipeline(**options)
    assert calls == {"review": 2, "pose": 2, "atlasGenerate": 1, "splat": 1, "splats2Mesh": 1 if mesh_price else 0}
    assert result["status"] == "needs_input" and result["simulation_qualified"] is False
    assert result["task_and_surrounding_scene_covered"] is True
    assert result["splat_available"] is True and result["mesh_available"] is bool(mesh_price)
    assert (exported["mesh_root"] is not None) is bool(mesh_price) and exported["splat_root"] == root / "splat"
    assert [str(kw["glb"]) for kw in registered] == ([str(root / "mesh/outputs/mesh.glb")] if mesh_price else [])
    # No reviewed mesh price means no mesh purchase, never an implied mesh.
    assert result["blockers"] == ([] if mesh_price else ["atlas_mesh_export_price_unreviewed"]) + [
        "metric_scene_registration_unproven"]
