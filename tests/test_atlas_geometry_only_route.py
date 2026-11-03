"""A geometry-only website job keeps the scene as filmed and buys only what the robot fit check needs.

Step 2 of the spec-feasibility tier (owner, 2026-11-03): the pose, the scale review and the
mesh, without object removal, image editing, generated completion or the refined splat.
"""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_atlas_pipeline.py
#   src/blueprint_pipeline/atlas_video_inputs.py
#   src/blueprint_pipeline/website_atlas_scene_source.py

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

import pytest

from blueprint_pipeline import atlas_depth_splat as splats
from blueprint_pipeline import atlas_mesh_registration as mesh_registration
from blueprint_pipeline import atlas_result_bundle as bundles
from blueprint_pipeline import atlas_source_pose_inputs as source_inputs
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import website_atlas_pipeline as pipeline
from blueprint_pipeline.atlas_pricing import final_output_cost
from tests.test_atlas_scene_planner import fixture as planner_fixture
from tests.test_website_atlas_scene_source import digest, job_fixture, write


def test_a_new_job_takes_the_hosts_route_and_keeps_it_once_anything_is_bought(tmp_path, monkeypatch):
    monkeypatch.delenv(pipeline.ROUTE_ENV, raising=False)
    assert pipeline._job_route(tmp_path / "unset", purchased=False) == "full"
    # A full-route job records nothing: its directory stays as it was.
    assert not (tmp_path / "unset" / "route.json").exists()

    job = tmp_path / "geometry"
    job.mkdir()
    monkeypatch.setenv(pipeline.ROUTE_ENV, "geometry_only")
    assert pipeline._job_route(job, purchased=False) == "geometry_only"
    monkeypatch.delenv(pipeline.ROUTE_ENV)
    # Its record outlives a change of the host setting.
    assert pipeline._job_route(job, purchased=True) == "geometry_only"

    # A job that bought before any record was kept is on the full route, whatever the host now says.
    monkeypatch.setenv(pipeline.ROUTE_ENV, "geometry_only")
    assert pipeline._job_route(tmp_path / "bought", purchased=True) == "full"

    monkeypatch.setenv(pipeline.ROUTE_ENV, "cheapest")
    with pytest.raises(ValueError, match="atlas_route_invalid"):
        pipeline._job_route(tmp_path / "bogus", purchased=False)


def test_geometry_only_never_holds_back_the_refined_splat():
    policy = {"mesh_export_cost_usd": .5, "refined_splat_cost_usd": 5, "rgbd_view_cost_usd": .05}
    assert final_output_cost(pipeline._route_policy(policy, "full")) == 5.5
    assert final_output_cost(pipeline._route_policy(policy, "geometry_only")) == .5
    assert pipeline._route_policy(policy, "full") is policy


@pytest.mark.parametrize("route", ["geometry_only", "full"])
def test_only_a_geometry_only_job_skips_the_removal_route_for_a_captured_subject(tmp_path, monkeypatch, route):
    from blueprint_pipeline import atlas_video_inputs as inputs, website_atlas_pose as pose_api
    _, survey, proposal, policy = planner_fixture(tmp_path)
    video = tmp_path / "source.mov"
    video.write_bytes(b"fresh source")
    survey["source_video_digest"] = pipeline._sha256_file(video)
    survey["schema_version"] = "atlas_video_survey.v1"
    survey["digest"] = pipeline.canonical_digest(survey, digest_field="digest")
    assert proposal["subject_origin"] == "captured"
    policy["reviewed_on"] = datetime.now(timezone.utc).date().isoformat()
    policy["refined_splat_cost_usd"] = 5
    monkeypatch.setenv(pipeline.ROUTE_ENV, route)
    monkeypatch.setattr(pose_api, "atlas_api_config", lambda: ("configured", "test-non-secret"))
    monkeypatch.setattr(pipeline, "load_configuration", lambda *a: {
        "policy": policy, "remaining_cost_usd": 8, "authority": {"provider_retention": "project_library_retention_approved"}})
    monkeypatch.setattr(inputs, "survey_video", lambda **kw: write(kw["output_root"] / "atlas_video_survey.json", survey))
    monkeypatch.setattr(pipeline, "accuracy_survey", lambda **kw: kw["base_survey"])
    selections = []
    monkeypatch.setattr(inputs, "prepare_video_inputs", lambda **kw: selections.append(kw) or {"frames": []})
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_plan", lambda **kw: {"proposal": proposal, "model": "test"})
    routed = []
    monkeypatch.setattr(pipeline, "_preserved_scene", lambda **kw: routed.append(kw) or {"status": "processing"})

    class FullRoute(Exception):
        pass

    def full_route(**_):
        raise FullRoute

    for name in ("run_sponsored_website_atlas_task", "run_sponsored_website_atlas_automation",
                 "run_sponsored_website_atlas_pose"):
        monkeypatch.setattr(allocator, name, full_route)
    context = {"capture_id": "fresh", "context_digest": "task", "description": "Move the box onto the shelf."}
    if route == "full":
        with pytest.raises(FullRoute):
            pipeline.run_atlas_pipeline(source_video=video, output_root=tmp_path / "job", task_context=context)
        assert not routed and selections[0]["exclude_transients"] is False
        assert not (tmp_path / "job" / "route.json").exists()
        return
    result = pipeline.run_atlas_pipeline(source_video=video, output_root=tmp_path / "job", task_context=context)
    assert result == {"status": "processing"} and len(routed) == 1
    assert routed[0]["geometry_only"] is True and routed[0]["review"]["subject_origin"] == "captured"
    # No editor runs, so a frame showing a person is never posed; and the inputs stay the ordinary
    # (not refined-dense) set, because the refined splat is never bought.
    assert selections[0]["exclude_transients"] is True
    assert "minimum_frames" not in selections[0]
    assert json.loads((tmp_path / "job" / "route.json").read_text())["route"] == "geometry_only"


def test_geometry_only_buys_the_pose_and_the_mesh_and_nothing_that_edits_or_completes(tmp_path, monkeypatch):
    pose_root = tmp_path / "pose"
    source_root = pose_root / "full-view-source-v1"
    source_root.mkdir(parents=True)
    pose = {"digest": "posed-source"}
    (source_root / "atlas_posed_rgbd_result.json").write_text(json.dumps(pose))
    (source_root / "scene_measurements.json").write_text("{}")
    (tmp_path / "atlas_scene_review.json").write_text("{}")
    # The plan would remove the subject and edit out a person; geometry only does neither.
    review = {"subject_origin": "captured", "automation": {
        "objects": [{"disposition": "remove"}], "removal_instruction": "Remove the box"},
        "transient_frame_ids": ["person-in-view"], "uncaptured_regions": ["rear"]}
    calls, pose_options = [], []

    def source_pose(**kw):
        calls.append("source_pose")
        pose_options.append(kw)
        return {"status": "completed", "operation_id": "source", "scale_status": "estimated"}

    monkeypatch.setattr(pipeline, "_source_pose", source_pose)
    monkeypatch.setattr(source_inputs, "prepare_source_pose", lambda root: (source_root, {}))
    monkeypatch.setattr(splats, "load_posed_rgbd", lambda root: (pose, ["observed_depth"]))
    monkeypatch.setattr(splats, "write_depth_splat", lambda **kw: calls.append("fuse") or {
        "digest": "splat", "splat_kind": "depth_fused_draft"})

    def mesh_attempt(root, output):
        output.mkdir()
        return {"request": {"splat": "bound"}, "uploads": {}}, output

    monkeypatch.setattr(splats, "splats2mesh_attempt", mesh_attempt)

    def task(**kw):
        assert kw["task"] == "splats2Mesh"
        calls.append("mesh")
        (kw["output_root"] / "result.json").write_text(json.dumps({"outputs": [{"role": "mesh", "path": "scene.glb"}]}))
        return {"status": "completed"}

    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_task", task)
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_automation",
                        lambda **kw: pytest.fail("Geometry only never buys generated completion"))
    monkeypatch.setattr(mesh_registration, "register_atlas_mesh", lambda **kw: calls.append("register"))
    monkeypatch.setattr(bundles, "export_pose_bundle", lambda **kw: {"zip_path": "scene.zip"})
    policy = {"mesh_export_cost_usd": .5, "rgbd_view_cost_usd": .05, "refined_splat_cost_usd": 5,
              "rate_card_reference": "test", "reviewed_by": "test", "reviewed_on": "test"}
    result = pipeline._preserved_scene(pose_root=pose_root, output_root=tmp_path, inputs={}, review=review,
        policy=policy, task_context={"context_digest": "task"}, source_digest="source", configuration={},
        check=lambda _stage: None, experimental=False, max_polls=1, model="test", geometry_only=True)
    assert calls == ["source_pose", "fuse", "mesh", "register"]
    # The scale review stays: reach and payload need metres.
    assert "allow_scale_review" not in pose_options[0]
    assert result["atlas_route"] == "geometry_only" and result["subject_in_reconstruction"] is True
    assert result["captured_scene_preserved"] is True and result["mesh_available"] is True
    assert result["object_removal_performed"] is False and result["image_inpainting_performed"] is False
    assert result["generated_completion_performed"] is False and "scene_completion" not in result
    assert result["digest"] == pipeline.canonical_digest(result, digest_field="digest")


@pytest.mark.parametrize("tamper", [None, "completion", "subject_flag"])
def test_the_scene_source_hands_on_the_scene_as_filmed_with_its_subject_in_it(tmp_path, tamper):
    from blueprint_pipeline import website_atlas_scene_source as scene_source
    root = job_fixture(tmp_path, same_pose=True)
    result = json.loads((root / "result.json").read_text())
    background = result.pop("automation")["background"]
    result.update(subject_origin="captured", atlas_route="geometry_only", subject_in_reconstruction=True,
        captured_scene_preserved=True, object_removal_performed=False, image_inpainting_performed=False,
        generated_completion_performed=tamper == "completion", mesh_available=True,
        splat_manifest_digest=background["splat_manifest_digest"])
    if tamper == "subject_flag":
        del result["subject_in_reconstruction"]
    shutil.move(root / "automation/background_splat", root / "splat")
    shutil.move(root / "automation/background_mesh", root / "mesh")
    write(root / "result.json", digest(result))
    write(root / "preview_job.json", {"state": "ready", "result_digest": result["digest"]})
    if tamper:
        # Geometry only never completes the scene, and says whether the subject is still in it.
        with pytest.raises(ValueError, match="preserved_scene_contract_invalid"):
            scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
        return
    value = scene_source.atlas_scene_source(capture_root=tmp_path, output_root=tmp_path / "prep")
    assert value["status"] == "ready" and value["scene_content"] == "captured_scene_preserved"
    assert value["atlas_route"] == "geometry_only" and value["subject_in_reconstruction"] is True
    assert value["subject_origin"] == "captured"
    assert not (root / "automation/background/prepared_background.json").exists()
