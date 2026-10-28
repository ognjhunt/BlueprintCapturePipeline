"""Additive tasks reconstruct observed geometry without an editing side route."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

import pytest

from blueprint_pipeline import website_atlas_pipeline as pipeline
from blueprint_pipeline import atlas_depth_splat as splats
from blueprint_pipeline import atlas_result_bundle as bundles
from blueprint_pipeline import atlas_source_pose_inputs as source_inputs
from blueprint_pipeline import atlas_mesh_registration as mesh_registration
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import website_atlas_scene_source as scene_source
from tests.test_website_atlas_scene_source import job_fixture, write, digest
from tests.test_atlas_scene_planner import fixture as planner_fixture


def test_automatic_controller_routes_the_typed_plan_without_a_scene_override(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_video_inputs as inputs, website_atlas_pose as pose_api
    _, survey, proposal, policy = planner_fixture(tmp_path)
    video = tmp_path / "source.mov"
    video.write_bytes(b"fresh source")
    survey["source_video_digest"] = pipeline._sha256_file(video)
    proposal.update(subject_origin="synthetic_design", automation={
        "scale_references": [], "objects": [], "removal_instruction": "", "mask_margin_fraction": .01})
    proposal["coverage_requirements"][0]["role"] = "task_region"
    policy["reviewed_on"] = datetime.now(timezone.utc).date().isoformat()
    monkeypatch.setattr(pose_api, "atlas_api_config", lambda: ("configured", "test-non-secret"))
    monkeypatch.setattr(pipeline, "load_configuration", lambda *a: {
        "policy": policy, "remaining_cost_usd": 4,
        "authority": {"provider_retention": "project_library_retention_approved"}})
    monkeypatch.setattr(inputs, "survey_video", lambda **kw: write(kw["output_root"] / "atlas_video_survey.json", survey))
    monkeypatch.setattr(pipeline, "accuracy_survey", lambda **kw: kw["base_survey"])
    monkeypatch.setattr(inputs, "prepare_video_inputs", lambda **kw: {"frames": []})
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_plan", lambda **kw: {"proposal": proposal, "model": "test"})
    routed = []
    monkeypatch.setattr(pipeline, "_preserved_scene", lambda **kw: routed.append(kw) or {"status": "processing"})
    for name in ("run_sponsored_website_atlas_task", "run_sponsored_website_atlas_automation",
                 "run_sponsored_website_atlas_pose"):
        monkeypatch.setattr(allocator, name, lambda **kw: pytest.fail("Legacy editing/generation route was entered"))
    result = pipeline.run_atlas_pipeline(source_video=video, output_root=tmp_path / "job",
        task_context={"capture_id": "fresh", "context_digest": "task"})
    assert result["status"] == "processing" and len(routed) == 1
    assert routed[0]["review"]["subject_origin"] == "synthetic_design"
    assert routed[0]["task_context"]["capture_id"] == "fresh"


@pytest.mark.parametrize("source_status", ["processing", "needs_input", "completed"])
@pytest.mark.parametrize("cached_splat", [False, True])
def test_preserved_reconstruction_exports_observations_and_never_buys_appearance(
        tmp_path, monkeypatch, source_status, cached_splat):
    pose_root = tmp_path / "pose"
    source_root = pose_root / "full-view-source-v1"
    source_root.mkdir(parents=True)
    pose = {"digest": "posed-source"}
    (source_root / "atlas_posed_rgbd_result.json").write_text(json.dumps(pose))
    (source_root / "scene_measurements.json").write_text("{}")
    (tmp_path / "atlas_scene_review.json").write_text("{}")
    proposal = {"subject_origin": "synthetic_design", "automation": {
        "objects": [{"disposition": "keep"}], "removal_instruction": ""},
        "uncaptured_regions": ["unobserved underside"]}
    calls, exports, decodes = [], [], []
    monkeypatch.setattr(pipeline, "_source_pose", lambda **kw: calls.append("source_pose") or {
        "status": source_status, "operation_id": "source", "reason": "scale_unresolved",
        "scale_status": "estimated_from_named_assumptions"})
    monkeypatch.setattr(source_inputs, "prepare_source_pose", lambda root: (source_root, {}))
    frames = ["observed_depth"]
    def decode(root):
        assert root == source_root
        decodes.append(root)
        return pose, frames
    monkeypatch.setattr(splats, "load_posed_rgbd", decode)
    def fuse(*, frames, result, output_root):
        assert result is pose and frames is retained_frames
        assert output_root == tmp_path / "splat"
        calls.append("fuse")
        return {"digest": "splat", "splat_kind": "depth_fused_draft"}
    retained_frames = frames
    monkeypatch.setattr(splats, "write_depth_splat", fuse)
    monkeypatch.setattr(splats, "build_depth_splat", lambda *a, **kw: pytest.fail("Source RGBD was decoded twice"))
    if cached_splat:
        write(tmp_path / "splat/atlas_depth_splat_manifest.json", {
            "digest": "splat", "splat_kind": "depth_fused_draft"})
    def mesh_attempt(root, output):
        output.mkdir()
        return {"request": {"splat": "bound"}, "uploads": {}}, output
    monkeypatch.setattr(splats, "splats2mesh_attempt", mesh_attempt)
    def task(**kw):
        assert kw["task"] == "splats2Mesh", "No unconditional image generation is authorized"
        calls.append("mesh")
        (kw["output_root"] / "result.json").write_text(json.dumps({
            "outputs": [{"role": "mesh", "path": "scene.glb"}]}))
        return {"status": "completed"}
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_task", task)
    monkeypatch.setattr(mesh_registration, "register_atlas_mesh", lambda **kw: calls.append("register"))
    monkeypatch.setattr(bundles, "export_pose_bundle", lambda **kw: exports.append(kw) or {"zip_path": "scene.zip"})
    stages = []
    result = pipeline._preserved_scene(pose_root=pose_root, output_root=tmp_path, inputs={}, review=proposal,
        policy={"mesh_export_cost_usd": .5, "rate_card_reference": "test", "reviewed_by": "test", "reviewed_on": "test"},
        task_context={"context_digest": "task"}, source_digest="source", configuration={}, check=stages.append,
        experimental=False, max_polls=1, model="test")
    if source_status == "processing":
        assert result["status"] == "processing" and calls == ["source_pose"] and not exports and not decodes
        return
    assert result["captured_scene_preserved"] is True
    assert result["image_inpainting_performed"] is False and result["object_removal_performed"] is False
    assert result["generated_completion_performed"] is False and result["simulation_qualified"] is False
    assert result["uncaptured_regions"] == ["unobserved underside"]
    assert exports[0]["pose_root"] == source_root and "generated_roots" not in exports[0]
    assert decodes == [source_root]
    expected = ["source_pose"] + ([] if cached_splat else ["fuse"])
    if source_status == "completed":
        assert calls == expected + ["mesh", "register"] and stages == ["mesh"]
    else:
        assert calls == expected and not stages
        assert "scale_unresolved" in result["blockers"] and not result["mesh_available"]


@pytest.mark.parametrize("tamper", [None, "removal", "pose"])
def test_production_scene_handoff_accepts_preserved_geometry_without_fabricating_removal(tmp_path, tamper):
    root = job_fixture(tmp_path, same_pose=tamper != "pose")
    result = json.loads((root / "result.json").read_text())
    background = result.pop("automation")["background"]
    result.update(subject_origin="synthetic_design", captured_scene_preserved=True,
        object_removal_performed=tamper == "removal", image_inpainting_performed=False,
        generated_completion_performed=False, mesh_available=True, splat_manifest_digest=background["splat_manifest_digest"])
    shutil.move(root / "automation/background_splat", root / "splat")
    shutil.move(root / "automation/background_mesh", root / "mesh")
    write(root / "result.json", digest(result))
    write(root / "preview_job.json", {"state": "ready", "result_digest": result["digest"]})
    if tamper:
        with pytest.raises(ValueError, match="preserved_scene_contract_invalid|preserved_scene_pose_mismatch"):
            scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    else:
        value = scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
        assert value["status"] == "ready" and value["scene_content"] == "captured_scene_preserved"
        assert value["base_scene"]["anchor"]["basis"] == "same_atlas_pose"
        assert value["base_scene"]["scale_authority"] == "atlas_reference_estimate"
        assert not (root / "automation/background/prepared_background.json").exists()
