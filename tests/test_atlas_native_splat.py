"""Refined-splat schema, durable admission, decoding and final-appearance contracts."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import atlas_depth_splat as depth
from blueprint_pipeline import atlas_native_splat as native
from blueprint_pipeline import website_atlas_tasks as tasks
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.atlas_pricing import render_cost, final_output_cost
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import read_standard_3dgs_ply
from tests.test_atlas_depth_splat import camera, frame, plane_depth
pytest_plugins = ["tests.test_website_atlas_tasks"]


def source_frames():
    return [frame(cam, plane_depth(cam), str(i)) for i, cam in enumerate([camera(), camera((.2, 0, 0))])]


def test_new_rate_counts_first_frame_once_and_preserves_historical_quotes():
    policy = {"rgbd_view_cost_usd": .05, "additional_rgbd_view_cost_usd": .01,
              "rgb_view_cost_usd": .04, "additional_rgb_view_cost_usd": .008,
              "refined_splat_cost_usd": 5, "mesh_export_cost_usd": .5}
    assert render_cost(policy, 30) == .34
    assert render_cost(policy, 30, depth=False) == .272
    assert render_cost(policy, 30, legacy_multiplier=2) == .34
    assert final_output_cost(policy) == 5.5
    assert render_cost({"rgbd_view_cost_usd": .05}, 30, legacy_multiplier=2) == 3
    with pytest.raises(ValueError, match="frame_count"):
        render_cost(policy, 0)


def test_native_request_retains_exact_returned_grid_and_linear_depth(tmp_path):
    frames = source_frames()
    request, uploads, provenance = native.prepare_splat_request(frames=frames, root=tmp_path)
    for i, (context, original) in enumerate(zip(request["posedFrames"], frames, strict=True)):
        assert context["camera"] == {"intrinsics": original["intrinsics"], "extrinsics": original["extrinsics"]}
        assert np.array_equal(depth.read_exr_depth(Path(uploads[f"frame-{i}-depth"]["path"])), original["depth"])
        assert provenance[i]["frame_id"] == original["frame_id"]
        assert "maskAsset" not in context  # Every pixel of these frames is fused.
    assert "refine" not in request  # Only the documented, refined-default OpenAPI contract.


def test_native_request_refuses_misaligned_depth_before_runner(tmp_path):
    frames = source_frames()
    frames[0]["depth"] = frames[0]["depth"][:-1]
    with pytest.raises(ValueError, match="grid_changed"):
        native.prepare_splat_request(frames=frames, root=tmp_path)


def test_native_schema_refuses_empty_result_and_published_count_mismatch(case, tmp_path):
    fake, request, run = native_case(case, tmp_path)
    with pytest.raises(ValueError, match="refined_splats_missing"):
        tasks._validate_response("posedImages2Splats", request, {"refinedSplats": {}})
    fake.response = {"refinedSplats": {"1": {"assetId": "asset_splat"}}}
    with pytest.raises(ValueError, match="splat_count_mismatch"):
        run()
    assert fake.posts == 1


def native_case(case, tmp_path):
    fake, body, review, _ = case
    fake.task = "posedImages2Splats"
    posed = body["contextFrames"][0]
    request = {"posedFrames": [posed, {**posed, "camera": {**posed["camera"],
        "extrinsics": {**posed["camera"]["extrinsics"], "position": [.1, 0, 0]}}}]}
    splat, _ = depth.fuse_posed_frames(source_frames())
    fake.asset_bytes["asset_splat"] = depth.encode_spz(splat)
    fake.response = {"refinedSplats": {str(splat.count): {"assetId": "asset_splat"}}}
    quote = {**review, "task": fake.task, "request_digest": canonical_digest(request), "maximum_cost_usd": 5}

    def run():
        return allocator.run_sponsored_website_atlas_task(task=fake.task, request=request, output_root=tmp_path,
            source_video_digest="sha256:" + "b" * 64, task_context=fake.context, price_review=quote,
            transport=fake.transport, sleep=lambda _: None, max_polls=1)
    return fake, request, run


def test_native_operation_uses_shared_reservation_and_replays_without_buying_twice(case, tmp_path):
    fake, _, run = native_case(case, tmp_path)
    result = run()
    assert result["outputs"][0]["path"].endswith(".spz")
    assert result["billing"] == {"status": "unreported", "reserved_usd": 5.0}
    assert result["metric_measurement_proven"] is False
    assert run() == result
    assert fake.posts == fake.reservations == 1


def test_native_pending_operation_is_resumed_by_id(case, tmp_path):
    fake, _, run = native_case(case, tmp_path)
    fake.pending = True
    assert run()["status"] == "processing"
    fake.pending = False
    assert run()["status"] == "completed"
    assert fake.posts == fake.reservations == 1


def test_native_lost_response_adopts_operation_without_a_second_purchase(case, tmp_path):
    from blueprint_pipeline.website_atlas_pose import AtlasTransportError
    fake, _, run = native_case(case, tmp_path)
    fake.outcome = "lost"
    with pytest.raises(AtlasTransportError):
        run()
    assert run()["status"] == "completed"
    assert fake.posts == fake.reservations == 1


def test_native_splat_invalid_output_is_never_published(case, tmp_path):
    fake, _, run = native_case(case, tmp_path)
    fake.asset_bytes["asset_splat"] = b"not a splat"
    with pytest.raises(ValueError, match="splat_not_spz"):
        run()
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("change,error", [
    (lambda r: r["posedFrames"][0]["depth"].clear(), "request_schema_invalid"),
    (lambda r: r["posedFrames"][1]["camera"]["intrinsics"].update(width=720), "grids_must_match"),
])
def test_native_schema_refuses_invalid_input_before_purchase(case, tmp_path, change, error):
    fake, request, run = native_case(case, tmp_path)
    # Separate the two cameras for the tamper, as a real decoded JSON request does.
    request.update(json.loads(json.dumps(request)))
    change(request)
    with pytest.raises(ValueError, match=error):
        run()
    assert fake.posts == fake.reservations == 0


def test_native_manifest_retains_provider_bytes_and_decoded_ply(tmp_path):
    frames = source_frames()
    splat, _ = depth.fuse_posed_frames(frames)
    blob = depth.encode_spz(splat)
    pose = {"digest": "sha256:" + "a" * 64, "operation": {"id": "op_pose"},
            "binding": {"source_video_digest": "sha256:" + "b" * 64}}
    policy = {"refined_splat_cost_usd": 5, "rate_card_reference": "https://example.test/card",
              "reviewed_by": "test", "reviewed_on": datetime.now(timezone.utc).date().isoformat()}
    calls = []

    def runner(**kw):
        from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
        calls.append(kw)
        root = kw["output_root"]
        root.mkdir(parents=True, exist_ok=True)
        path = root / "original.spz"
        path.write_bytes(blob)
        state = {"status": "completed", "operation": {"id": "op_native"},
                "binding": {"source_video_digest": kw["source_video_digest"],
                            "task_context_digest": kw["task_context"]["context_digest"],
                            "request_template": kw["request"], "uploads": kw["uploads"]},
                "outputs": [{"role": f"refinedSplats-{splat.count}", "path": path.name, "sha256": _sha256_file(path)}],
                "billing": {"status": "unreported", "reserved_usd": 5}}
        state["digest"] = canonical_digest(state, digest_field="digest")
        (root / "result.json").write_text(json.dumps(state))
        (root / "atlas_pose").mkdir()
        (root / "atlas_pose/submission.json").write_text(json.dumps(
            {"status": "completed", "result_digest": state["digest"]}))
        return state

    draft = {"digest": "sha256:" + "c" * 64, "voxel_size": .01, "generated_result_digests": ["generated"]}
    result = native.run_native_splat(frames=frames, result=pose, draft=draft, output_root=tmp_path, policy=policy,
                                    task_context={"context_digest": "sha256:" + "d" * 64}, task_runner=runner)
    manifest = result["splat_manifest"]
    assert (tmp_path / "atlas_refined_splat.spz").read_bytes() == blob
    decoded = read_standard_3dgs_ply(tmp_path / "atlas_refined_splat.ply")
    assert decoded.count == splat.count
    assert manifest["splat_kind"] == "atlas_refined" and manifest["photometric_optimization"] is True
    assert manifest["metric_measurement_proven"] is manifest["simulation_qualified"] is False
    assert calls[0]["task"] == "posedImages2Splats"
    assert (tmp_path / "splats2mesh_task_spec.json").is_file()
    # Registration keeps the draft's grid; the draft and its generated inputs stay bound.
    assert manifest["voxel_size"] == .01 and manifest["draft_splat_manifest_digest"] == draft["digest"]
    assert manifest["generated_result_digests"] == ["generated"]
    resumed = native.run_native_splat(frames=frames, result=pose, draft=draft, output_root=tmp_path, policy=policy,
        task_context={"context_digest": "sha256:" + "d" * 64}, task_runner=runner)
    assert resumed == result and len(calls) == 1
    assert np.array_equal(decoded.xyz, depth.decode_spz(blob).xyz)
    (tmp_path / "atlas_refined_splat.spz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="output_changed"):
        native.run_native_splat(frames=frames, result=pose, draft=draft, output_root=tmp_path, policy=policy,
            task_context={"context_digest": "sha256:" + "d" * 64}, task_runner=runner)


def test_refined_request_masks_exactly_the_pixels_the_draft_would_not_fuse(tmp_path):
    from PIL import Image
    frames = source_frames()
    frames[1]["confidence"][:, :10] = 0
    # A recovered view can repeat a generated view's ID; each is posted on its own.
    frames[1]["frame_id"] = frames[0]["frame_id"]
    request, uploads, provenance = native.prepare_splat_request(frames=frames, root=tmp_path)
    assert len(request["posedFrames"]) == 2 and "maskAsset" not in request["posedFrames"][0]
    assert request["posedFrames"][1]["maskAsset"] == {"assetId": "upload:frame-1-mask"}
    mask = np.asarray(Image.open(uploads["frame-1-mask"]["path"]))
    assert np.array_equal(mask == 255, frames[1]["confidence"] > 0) and set(np.unique(mask)) == {0, 255}
    assert provenance[1]["mask_sha256"] == uploads["frame-1-mask"]["sha256"]
    # A view the draft admitted no pixel of is not posted at all.
    empty = {**frames[0], "frame_id": "generated", "confidence": np.zeros_like(frames[0]["confidence"])}
    request, _, provenance = native.prepare_splat_request(frames=[*frames, empty], root=tmp_path / "with-empty")
    assert len(request["posedFrames"]) == 2 and "generated" not in {row["frame_id"] for row in provenance}
    with pytest.raises(ValueError, match="frames_required"):
        native.prepare_splat_request(frames=[frames[0], empty], root=tmp_path / "too-few")


def test_mixed_grids_are_posted_on_one_grid_with_their_rays(tmp_path):
    portrait, landscape = camera(), camera((.2, 0, 0), width=64, height=36)
    frames = [frame(portrait, plane_depth(portrait), "portrait"), frame(landscape, plane_depth(landscape), "landscape")]
    request, uploads, provenance = native.prepare_splat_request(frames=frames, root=tmp_path)
    assert {(c["camera"]["intrinsics"]["width"], c["camera"]["intrinsics"]["height"])
            for c in request["posedFrames"]} == {(1280, 720)}
    assert {row["frame_id"] for row in provenance} == {"portrait", "landscape"}
    assert len(uploads) >= 2 * len(request["posedFrames"])


@pytest.mark.parametrize("state", ["completed", "processing"])
def test_refined_splat_replaces_the_draft_as_the_final_appearance(tmp_path, monkeypatch, state):
    from blueprint_pipeline import atlas_scene_automation as worker, atlas_result_bundle
    from blueprint_pipeline import atlas_background_automation as edits, atlas_scene_measurements as sizes
    from tests.test_atlas_scene_automation import view, plan
    f = view(tmp_path)
    pose = {"digest": "pose", "binding": {"source_video_digest": "source"}}
    monkeypatch.setattr(sizes, "load_views", lambda *a: (pose, [f]))
    monkeypatch.setattr(edits, "run_background", lambda **_: {
        "status": "passed", "provider": "atlas", "stage_roots": [], "result_digests": ["edited-task"],
        "visual_review_digest": "review", "batch_frame_ids": []})
    monkeypatch.setattr(worker, "_edited_frames", lambda _: [f])
    draft = {"digest": "draft", "splat_kind": "depth_fused_draft"}
    # The draft is still built: it is the gap and coverage evidence.
    monkeypatch.setattr(depth, "write_depth_splat", lambda **_: draft)
    observed, exported = [], []

    def refined(**kw):
        observed.append(kw)
        if state == "processing":
            return {"status": "processing", "stage": "refined_splat", "operation_id": "op"}
        return {"status": "completed", "splat_root": str(tmp_path / "refined"),
                "splat_manifest": {"digest": "refined", "splat_kind": "atlas_refined"}}
    monkeypatch.setattr(native, "run_native_splat", refined)
    monkeypatch.setattr(atlas_result_bundle, "export_pose_bundle",
                        lambda **kw: exported.append(kw) or {"zip_path": "refined.zip"})
    result = worker.run_automation(pose_root=tmp_path, inputs={}, review={"automation": plan()},
        output_root=tmp_path / "out", task_context={}, configuration={"policy": {"refined_splat_cost_usd": 5}},
        task_runner=lambda **_: pytest.fail("the refined splat is bought through run_native_splat"),
        pose_runner=None, release_blockers=lambda: ("commit", []))
    assert observed[0]["frames"] == [f] and observed[0]["draft"] is draft
    assert observed[0]["output_root"] == tmp_path / "out/background_splat-refined"
    if state == "processing":
        assert result == {"status": "processing", "stage": "refined_splat", "operation_id": "op"} and not exported
        return
    assert result["status"] == "completed" and exported[0]["splat_root"] == tmp_path / "refined"
    assert result["background"]["splat_kind"] == "atlas_refined"
    assert result["background"]["splat_manifest_digest"] == "refined"


@pytest.mark.parametrize("refined", [False, True])
def test_refined_splat_follows_gap_completion_without_the_draft_coverage_pass(tmp_path, monkeypatch, refined):
    from blueprint_pipeline import atlas_scene_automation as worker, atlas_result_bundle, atlas_scene_coverage
    from blueprint_pipeline import atlas_background_automation as edits, atlas_scene_measurements as sizes
    from blueprint_pipeline import atlas_scene_completion, atlas_recorded_source_recovery, atlas_scene_recovery
    from blueprint_pipeline import atlas_scene_task_regions
    from tests.test_atlas_scene_automation import view, plan
    real, generated = view(tmp_path), view(tmp_path, "gap", .2)
    pose = {"digest": "pose", "binding": {"source_video_digest": "source"}}
    monkeypatch.setattr(sizes, "load_views", lambda *a: (pose, [real]))
    monkeypatch.setattr(edits, "run_background", lambda **_: {
        "status": "fallback_prepared", "frames": ["edited"], "review": {"verdict": "passed"}, "stage_roots": []})
    monkeypatch.setattr(worker, "_fallback_pose", lambda *a: ((pose, [real], tmp_path), {}))
    monkeypatch.setattr(atlas_recorded_source_recovery, "adopt_recorded_recoveries", lambda **_: ([real], []))
    monkeypatch.setattr(atlas_scene_completion, "complete_scene", lambda **_: ([generated], {
        "status": "passed", "generation_result_digest": "gaps", "review": "gap-review", "generated_roots": []}))
    monkeypatch.setattr(atlas_scene_recovery, "reviewed_completion_views", lambda _: ["reviewed-gap-view"])
    monkeypatch.setattr(atlas_scene_task_regions, "plan_task_regions", lambda **_: {})
    monkeypatch.setattr(atlas_scene_recovery, "recover_reviewed_geometry",
                        lambda **_: ([], {"digest": "rec", "preservation_paths": []}))
    monkeypatch.setattr(depth, "write_depth_splat", lambda **kw: {
        "digest": "draft", "generated_result_digests": kw["generated_result_digests"],
        "frames": [{"frame_id": f["frame_id"]} for f in kw["frames"]]})
    coverage = []
    monkeypatch.setattr(atlas_scene_coverage, "refine_scene_coverage",
                        lambda **kw: coverage.append(kw) or (None, {"status": "processing", "stage": "coverage"}))
    observed = []
    monkeypatch.setattr(native, "run_native_splat", lambda **kw: observed.append(kw) or {
        "status": "completed", "splat_root": str(tmp_path / "refined"),
        "splat_manifest": {"digest": "refined", "splat_kind": "atlas_refined"}})
    monkeypatch.setattr(atlas_result_bundle, "export_pose_bundle", lambda **kw: {"zip_path": "refined.zip"})
    policy = {"rgbd_view_cost_usd": .05, **({"refined_splat_cost_usd": 5} if refined else {})}
    result = worker.run_automation(pose_root=tmp_path, inputs={}, review={"automation": plan()},
        output_root=tmp_path / "out", task_context={}, configuration={"policy": policy},
        task_runner=None, pose_runner=None, release_blockers=lambda: ("commit", []))
    if not refined:
        assert len(coverage) == 1 and not observed and result["status"] == "processing"
        return
    # The generated gap views go into the refined splat with the real views.
    assert not coverage and [f["frame_id"] for f in observed[0]["frames"]] == ["a", "gap"]
    assert result["status"] == "completed" and result["background"]["splat_kind"] == "atlas_refined"
