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


def test_real_views_go_in_whole_and_reviewed_views_only_where_admitted(tmp_path):
    from PIL import Image
    frames = source_frames()
    # Atlas's own low depth confidence: the RGB there is still real footage.
    frames[0]["confidence"][:, :10] = 0
    keep = np.zeros(frames[1]["depth"].shape, bool)
    keep[:, 20:] = True
    reviewed = depth.restrict_frame(frames[1], keep)
    # Fusion sees exactly confidence * mask, as before.
    assert np.array_equal(reviewed["confidence"], frames[1]["confidence"] * keep)
    # A recovered view can repeat a generated view's ID; each is posted on its own.
    frames[1] = {**reviewed, "frame_id": frames[0]["frame_id"]}
    request, uploads, provenance = native.prepare_splat_request(frames=frames, root=tmp_path)
    real, partial = request["posedFrames"]
    assert "maskAsset" not in real
    assert real["depth"]["confidenceAsset"] == {"assetId": "upload:frame-0-confidence"}
    confidence = np.asarray(Image.open(uploads["frame-0-confidence"]["path"]))
    assert (confidence[:, :10] == 0).all() and (confidence[:, 10:] == 255).all()
    assert partial["maskAsset"] == {"assetId": "upload:frame-1-mask"}
    assert np.array_equal(np.asarray(Image.open(uploads["frame-1-mask"]["path"])) == 255, keep)
    assert provenance[1]["mask_sha256"] == uploads["frame-1-mask"]["sha256"]
    # A view whose review admitted nothing is not posted; masks compose.
    empty = depth.restrict_frame({**source_frames()[0], "frame_id": "generated"}, np.zeros_like(keep))
    request, _, provenance = native.prepare_splat_request(frames=[*frames, empty], root=tmp_path / "with-empty")
    assert len(request["posedFrames"]) == 2 and "generated" not in {row["frame_id"] for row in provenance}
    with pytest.raises(ValueError, match="frames_required"):
        native.prepare_splat_request(frames=[frames[0], empty], root=tmp_path / "too-few")
    assert np.array_equal(depth.restrict_frame(reviewed, ~keep)["keep_mask"], np.zeros_like(keep))


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
                "splat_manifest": {"digest": "refined", "splat_kind": "atlas_refined"}, "refined_splat": marker}
    marker = {"request_form": "documented_minimal", "left_out_admitted_view_ids": ["gap"],
              "gap_preserving_form_untried": True}
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
    # The adopted form is on the result, not in the bundle: a failed upgrade leaves the bundle unchanged.
    assert result["background"]["refined_splat"] == marker
    assert "refined_splat" not in json.loads((tmp_path / "out/background_review.json").read_text())
    from blueprint_pipeline.website_atlas_pipeline import finalize_background_result
    handoff = finalize_background_result(output_root=tmp_path / "out", source_digest="source",
        task_context={"context_digest": "ctx"}, automation=result, source_pose=None, model="model")
    assert handoff["refined_splat"] == marker


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



POSE = {"digest": "pose", "operation": {"id": "op_pose"}, "binding": {"source_video_digest": "video"}}
DRAFT = {"digest": "draft", "voxel_size": .01}
CONTEXT = {"context_digest": "ctx"}


def keep(columns):
    value = np.zeros((64, 48), bool)
    value[:, columns] = True
    return value


def gap_view(admitted, frame_id="gap", shift=0.0):
    """A generated gap view (or geometry recovered for it: same camera and RGB) admitted in part."""
    cam = camera((0, .2, 0))
    return depth.restrict_frame(frame(cam, plane_depth(cam) + shift, frame_id), admitted)


def recovered(view, admitted, frame_id="recovered"):
    """Recovered geometry repeating another view's camera and RGB with its own depth and mask."""
    whole = {key: value for key, value in view.items() if key != "keep_mask"}
    return {**depth.restrict_frame({**whole, "depth": view["depth"] + .5}, admitted), "frame_id": frame_id}


def scripted_runner(outcomes, calls):
    """The admitted runner as World Labs answers each call: completed, failed, or still processing."""
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    splat, _ = depth.fuse_posed_frames(source_frames())
    blob = depth.encode_spz(splat)

    def runner(**kw):
        calls.append(kw)
        stage, outcome = kw["output_root"], outcomes.pop(0)
        (stage / "atlas_pose").mkdir(parents=True, exist_ok=True)
        binding = {"source_video_digest": kw["source_video_digest"], "request_template": kw["request"],
                   "uploads": kw["uploads"], "task_context_digest": kw["task_context"]["context_digest"]}
        state = {"status": "submitted", "binding": binding}
        if outcome == "completed":
            (stage / "original.spz").write_bytes(blob)
            result = {"status": "completed", "operation": {"id": "op_native"}, "binding": binding,
                      "outputs": [{"role": f"refinedSplats-{splat.count}", "path": "original.spz",
                                   "sha256": _sha256_file(stage / "original.spz")}],
                      "billing": {"status": "unreported", "reserved_usd": 5}}
            result["digest"] = canonical_digest(result, digest_field="digest")
            (stage / "result.json").write_text(json.dumps(result))
            state.update(status="completed", result_digest=result["digest"])
        elif outcome == "failed":
            state.update(status="failed", terminal_operation={"error": {"code": 13}})
        (stage / "atlas_pose/submission.json").write_text(json.dumps(state))
        if outcome == "failed":
            raise ValueError("atlas_task_operation_failed")
        return result if outcome == "completed" else {"status": "processing", "operation_id": "op"}
    return runner


def refined(frames, root, outcomes, calls):
    policy = {"refined_splat_cost_usd": 5, "rate_card_reference": "https://example.test/card",
              "reviewed_by": "test", "reviewed_on": datetime.now(timezone.utc).date().isoformat()}
    return native.run_native_splat(frames=frames, result=POSE, draft=DRAFT, output_root=root, policy=policy,
                                   task_context=CONTEXT, task_runner=scripted_runner(outcomes, calls))


def previous_release(monkeypatch, frames, root, form, outcome):
    """A form the previous release bought without merging duplicate views."""
    with monkeypatch.context() as patch:
        patch.setattr(native, "merge_duplicate_views", lambda frames: [(i, f, []) for i, f in enumerate(frames)])
        policy = {"refined_splat_cost_usd": 5, "rate_card_reference": "https://example.test/card",
                  "reviewed_by": "test", "reviewed_on": datetime.now(timezone.utc).date().isoformat()}
        try:
            return native._run_form(form=form, root=root, frames=frames, result=POSE, draft=DRAFT, policy=policy,
                                    task_context=CONTEXT, task_runner=scripted_runner([outcome], []),
                                    max_polls=1, checkpoint=None)
        except ValueError:
            return None


def tree(root):
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_a_whole_duplicate_absorbs_a_recovered_view_and_takes_its_depth_where_admitted(tmp_path):
    from PIL import Image
    frames = source_frames()
    duplicate = recovered(frames[1], keep(slice(0, 10)))
    duplicate["confidence"] = duplicate["confidence"] * .5
    request, uploads, provenance = native.prepare_splat_request(frames=[*frames, duplicate], root=tmp_path)
    assert len(request["posedFrames"]) == 2 and "maskAsset" not in request["posedFrames"][1]
    posted = depth.read_exr_depth(Path(uploads["frame-1-depth"]["path"]))
    assert np.array_equal(posted[:, :10], duplicate["depth"][:, :10])
    assert np.array_equal(posted[:, 10:], frames[1]["depth"][:, 10:])
    confidence = np.asarray(Image.open(uploads["frame-1-confidence"]["path"]))
    assert (confidence[:, :10] == 128).all() and (confidence[:, 10:] == 255).all()
    assert provenance[1]["frame_id"] == "1"
    assert provenance[1]["merged_duplicates"] == [{"position": 2, "frame_id": "recovered"}]
    assert "merged_duplicates" not in provenance[0]


def test_partial_duplicates_post_one_view_with_the_union_of_their_masks(tmp_path):
    from PIL import Image
    generated = gap_view(keep(slice(20, None)))
    # Recovered views can repeat the generated view's ID.
    duplicate = recovered(gap_view(keep(slice(None))), keep(slice(0, 10)), frame_id="gap")
    request, uploads, provenance = native.prepare_splat_request(
        frames=[source_frames()[0], generated, duplicate], root=tmp_path)
    assert len(request["posedFrames"]) == 2 and request["posedFrames"][1]["maskAsset"]
    mask = np.asarray(Image.open(uploads["frame-1-mask"]["path"])) == 255
    assert np.array_equal(mask, keep(slice(20, None)) | keep(slice(0, 10)))
    posted = depth.read_exr_depth(Path(uploads["frame-1-depth"]["path"]))
    assert np.array_equal(posted[:, :10], duplicate["depth"][:, :10])
    assert np.array_equal(posted[:, 20:], generated["depth"][:, 20:])
    assert provenance[1]["merged_duplicates"] == [{"position": 2, "frame_id": "gap"}]
    # The minimal form still leaves the merged partial view out.
    request, _, _ = native.prepare_splat_request(frames=[*source_frames(), generated, duplicate],
                                                 root=tmp_path / "minimal", form="documented_minimal")
    assert len(request["posedFrames"]) == 2


def test_mixed_grids_merge_duplicates_before_tiling(tmp_path):
    portrait, landscape = camera(), camera((.2, 0, 0), width=64, height=36)
    frames = [frame(portrait, plane_depth(portrait), "portrait"), frame(landscape, plane_depth(landscape), "landscape")]
    alone, _, _ = native.prepare_splat_request(frames=frames, root=tmp_path / "alone")
    twin = {**depth.restrict_frame(frames[1], np.ones((36, 64), bool) & (np.arange(64) < 20)), "frame_id": "twin"}
    request, _, provenance = native.prepare_splat_request(frames=[*frames, twin], root=tmp_path / "merged")
    assert len(request["posedFrames"]) == len(alone["posedFrames"])
    assert {row["frame_id"] for row in provenance} == {"portrait", "landscape"}
    assert all(row["merged_duplicates"] == [{"position": 2, "frame_id": "twin"}]
               for row in provenance if row["frame_id"] == "landscape")


def test_after_provider_failures_masks_follow_then_the_documented_minimal_form(tmp_path):
    frames, calls = [*source_frames(), gap_view(keep(slice(20, None)))], []
    root = tmp_path / "draft-refined"
    assert refined(frames, root, ["failed", "failed", "processing"], calls)["status"] == "processing"
    assert [call["output_root"] for call in calls] == [
        tmp_path / f"draft-refined{suffix}/native" for suffix in ("", "-masks", "-minimal")]
    full, masks, minimal = (call["request"]["posedFrames"] for call in calls)
    assert len(full) == len(masks) == 3 and full[2]["maskAsset"] and masks[2]["maskAsset"]
    assert all("confidenceAsset" in view["depth"] for view in full)
    assert not any("confidenceAsset" in view["depth"] for view in masks)
    # Whole views only, image + depth + camera, exactly the guide's example.
    assert len(minimal) == 2 and all(set(view) == {"imageAsset", "camera", "depth"}
                                     and set(view["depth"]) == {"depthAsset"} for view in minimal)
    assert [json.loads((tmp_path / f"draft-refined{suffix}/input_binding.json").read_text())["request_form"]
            for suffix in ("", "-masks", "-minimal")] == ["reviewed_masks_and_confidence", "reviewed_masks",
                                                          "documented_minimal"]
    assert not (tmp_path / "draft-refined-merged").exists()
    # The live minimal attempt resumes with what it sent; both gap forms were tried.
    done = refined(frames, root, ["completed"], calls)
    assert calls[3]["request"] == calls[2]["request"] and done["splat_root"] == str(tmp_path / "draft-refined-minimal")
    assert done["refined_splat"] == {
        "request_form": "documented_minimal", "splat_root": done["splat_root"],
        "splat_manifest_digest": done["splat_manifest"]["digest"], "left_out_admitted_view_ids": ["gap"],
        "gap_preserving_form_untried": False}
    assert refined(frames, root, [], calls)["splat_manifest"] == done["splat_manifest"] and len(calls) == 4


def test_a_minimal_splat_of_whole_views_records_nothing_left_out(tmp_path):
    calls = []
    done = refined(source_frames(), tmp_path / "draft-refined", ["failed", "failed", "completed"], calls)
    assert len(calls) == 3 and done["refined_splat"]["request_form"] == "documented_minimal"
    assert "left_out_admitted_view_ids" not in done["refined_splat"]
    assert "gap_preserving_form_untried" not in done["refined_splat"]


def test_a_full_form_failed_before_merging_is_evaluated_as_retained_and_never_bought_again(tmp_path, monkeypatch):
    frames = [*source_frames(), gap_view(keep(slice(20, None)))]
    frames.append(recovered(frames[1], keep(slice(0, 10))))
    root = tmp_path / "draft-refined"
    previous_release(monkeypatch, frames, root, "reviewed_masks_and_confidence", "failed")
    retained, calls = tree(root), []
    assert refined(frames, root, ["processing"], calls)["status"] == "processing"
    assert [call["output_root"] for call in calls] == [tmp_path / "draft-refined-masks/native"]
    assert len(calls[0]["request"]["posedFrames"]) == 3  # the recovered duplicate merged
    assert tree(root) == retained and not (tmp_path / "draft-refined-merged").exists()


def test_a_live_full_form_from_before_merging_resumes_with_what_it_sent(tmp_path, monkeypatch):
    frames = [*source_frames(), gap_view(keep(slice(20, None)))]
    frames.append(recovered(frames[1], keep(slice(0, 10))))
    root = tmp_path / "draft-refined"
    previous_release(monkeypatch, frames, root, "reviewed_masks_and_confidence", "processing")
    sent = json.loads((root / "native/atlas_pose/submission.json").read_text())["binding"]["request_template"]
    inputs, calls = tree(root / "inputs"), []
    done = refined(frames, root, ["completed"], calls)
    assert [call["output_root"] for call in calls] == [root / "native"]
    assert calls[0]["request"] == sent and len(sent["posedFrames"]) == 4 and tree(root / "inputs") == inputs
    assert done["splat_root"] == str(root) and done["refined_splat"] == {
        "request_form": "reviewed_masks_and_confidence", "splat_root": str(root),
        "splat_manifest_digest": done["splat_manifest"]["digest"]}


@pytest.mark.parametrize("outcome", ["completed", "failed"])
def test_an_operator_upgrade_sends_the_untried_gap_forms_once(tmp_path, monkeypatch, outcome):
    frames = [*source_frames(), gap_view(keep(slice(20, None)))]
    frames.append(recovered(frames[1], keep(slice(0, 10))))
    root, minimal = tmp_path / "draft-refined", tmp_path / "draft-refined-minimal"
    # The previous release failed the full form, then bought the minimal one.
    previous_release(monkeypatch, frames, root, "reviewed_masks_and_confidence", "failed")
    previous_release(monkeypatch, frames, minimal, "documented_minimal", "completed")
    calls = []
    adopted = refined(frames, root, [], calls)
    # Adopted as it is; nothing upgrades it on its own.
    assert not calls and adopted["splat_root"] == str(minimal)
    marker = adopted["refined_splat"]
    assert marker["left_out_admitted_view_ids"] == ["gap"] and marker["gap_preserving_form_untried"] is True
    with pytest.raises(ValueError, match="upgrade_invalid"):
        native.request_gap_preserving_upgrade(minimal, splat_manifest_digest="sha256:" + "0" * 64, at_epoch=1)
    request = native.request_gap_preserving_upgrade(minimal, splat_manifest_digest=marker["splat_manifest_digest"],
                                                    at_epoch=1)
    assert request["roots"] == ["draft-refined-merged", "draft-refined-masks"]
    upgraded = refined(frames, root, [outcome] if outcome == "completed" else ["failed", "failed"], calls)
    if outcome == "completed":
        assert [call["output_root"] for call in calls] == [tmp_path / "draft-refined-merged/native"]
        posted = calls[0]["request"]["posedFrames"]
        assert len(posted) == 3 and posted[2]["maskAsset"] and posted[2]["depth"]["confidenceAsset"]
        assert upgraded["splat_root"] == str(tmp_path / "draft-refined-merged")
        assert upgraded["refined_splat"]["request_form"] == "reviewed_masks_and_confidence"
        assert "gap_preserving_form_untried" not in upgraded["refined_splat"]
    else:
        assert [call["output_root"] for call in calls] == [
            tmp_path / "draft-refined-merged/native", tmp_path / "draft-refined-masks/native"]
        # The minimal splat stands unchanged, and the upgrade is spent.
        assert upgraded["splat_manifest"] == adopted["splat_manifest"] and upgraded["splat_root"] == str(minimal)
        assert upgraded["refined_splat"]["gap_preserving_form_untried"] is False
        with pytest.raises(ValueError, match="upgrade_unavailable"):
            native.request_gap_preserving_upgrade(minimal, splat_manifest_digest=marker["splat_manifest_digest"],
                                                  at_epoch=2)
    calls.clear()
    assert refined(frames, root, [], calls)["splat_root"] == upgraded["splat_root"] and not calls
