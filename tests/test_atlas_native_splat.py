"""Native splat schema, durable admission, decoding and source-selection contracts."""
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
        assert provenance[i]["source_frame_id"] == original["frame_id"]
    assert "refine" not in request  # Only the documented, refined-default OpenAPI contract.


def test_native_request_refuses_misaligned_depth_before_runner(tmp_path):
    frames = source_frames()
    frames[0]["depth"] = frames[0]["depth"][:-1]
    with pytest.raises(ValueError, match="grid_changed"):
        native.prepare_splat_request(frames=frames, root=tmp_path)


@pytest.mark.parametrize("count", [2, 42])
def test_real_source_selection_expands_minimal_edit_set_without_hidden_or_blurry_views(tmp_path, monkeypatch, count):
    from PIL import Image
    from blueprint_pipeline import atlas_video_inputs as selection
    from blueprint_pipeline.atlas_splat_source_selection import select_splat_sources
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    rng = np.random.default_rng(11)
    rows = []
    for i in range(count):
        path = tmp_path / f"{i}.png"
        pixels = (np.full((64, 64, 3), 100, np.uint8) if i == 11 else
                  rng.integers(20, 230, (64, 64, 3), dtype=np.uint8))
        Image.fromarray(pixels).save(path)
        rows.append({"frame_id": str(i), "timestamp_seconds": float(i),
                     "image_path": str(path), "image_digest": _sha256_file(path)})
    inputs = {"heldout_pixels_included": False, "frames": rows, "digest": "input"}
    review = {"anchor_frame_id": "0", "coverage_requirements": [
        {"role": "task_object", "label": "task", "required_frame_ids": ["0"]},
        {"role": "surrounding_scene", "label": "whole room", "required_frame_ids": [str(count - 1)]}]}
    monkeypatch.setattr(selection, "_features", lambda row, rotation: int(Path(row["source_image_path"]).stem))
    monkeypatch.setattr(selection, "_overlap", lambda a, b: (50, 25.) if abs(a - b) <= 6 else (0, 0.))
    ids, receipt = select_splat_sources(frames=rows, inputs=inputs, review=review, output_root=tmp_path / "selection")
    assert min(16, count) <= len(ids) <= min(30, count) and {"0", str(count - 1)} <= set(ids)
    assert "11" not in ids and set(ids) <= {row["frame_id"] for row in rows}
    assert receipt["generated_views_count_as_real"] is False
    assert receipt["minimum_recommended_views_met"] is (len(ids) >= 16)
    assert select_splat_sources(frames=rows, inputs=inputs, review=review,
                                output_root=tmp_path / "selection")[0] == ids


def test_native_schema_refuses_empty_result_and_published_count_mismatch(case, tmp_path):
    fake, request, run = native_case(case, tmp_path)
    with pytest.raises(ValueError, match="refined_splats_missing"):
        tasks._validate_response("posedImages2Splats", request, {"refinedSplats": {}})
    fake.response = {"refinedSplats": {"1": {"assetId": "asset_splat"}}}
    with pytest.raises(ValueError, match="splat_count_mismatch"):
        run()
    assert fake.posts == 1


def test_native_splat_missing_decoder_refuses_before_purchase(tmp_path, monkeypatch):
    from blueprint_pipeline import gaussian_splat_decode
    monkeypatch.setattr(gaussian_splat_decode, "find_splat_transform_cli", lambda *_: None)
    with pytest.raises(ValueError, match="decoder_required_before_purchase"):
        native.run_native_splat(frames=source_frames(), result={"digest": "pose",
            "binding": {"source_video_digest": "source"}}, output_root=tmp_path, policy={},
            task_context={}, task_runner=lambda **_: pytest.fail("must not purchase"))


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


def test_native_controller_uses_real_views_without_legacy_fusion_or_upfront_gap_generation(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_automation as worker
    from blueprint_pipeline import atlas_scene_measurements, atlas_background_automation, atlas_scene_completion
    from blueprint_pipeline import atlas_splat_source_selection, atlas_result_bundle
    from tests.test_atlas_scene_automation import view, plan
    frames = [view(tmp_path), view(tmp_path, "b", .1)]
    pose = {"digest": "pose", "binding": {"source_video_digest": "video"}}
    inputs = {"digest": "inputs", "frames": [{k: f[k] for k in ("frame_id", "image_path", "image_digest")}
                                              for f in frames]}
    monkeypatch.setattr(atlas_scene_measurements, "load_views", lambda *_: (pose, frames))
    monkeypatch.setattr(atlas_scene_measurements, "estimate_sizes", lambda **_: {})
    monkeypatch.setattr(atlas_splat_source_selection, "select_splat_sources",
                        lambda **_: (["a", "b"], {"selected_real_frame_count": 2}))
    monkeypatch.setattr(atlas_background_automation, "run_background", lambda **_: {
        "status": "passed", "stage_roots": [], "result_digests": [], "visual_review_digest": "review"})
    monkeypatch.setattr(worker, "_edited_frames", lambda _: frames)
    monkeypatch.setattr(depth, "write_depth_splat", lambda **_: pytest.fail("legacy fusion must not gate native Atlas"))
    monkeypatch.setattr(atlas_scene_completion, "complete_scene", lambda **_: pytest.fail("do not buy gap views upfront"))
    observed = []

    def run(**kw):
        observed.append(kw)
        return {"status": "completed", "splat_root": str(tmp_path / "native"),
                "splat_manifest": {"digest": "native", "splat_kind": "atlas_refined"}}
    monkeypatch.setattr(native, "run_native_splat", run)
    monkeypatch.setattr(atlas_result_bundle, "export_pose_bundle", lambda **kw: {"directory": str(kw["splat_root"])})
    result = worker.run_automation(pose_root=tmp_path, inputs=inputs,
        review={"automation": plan(), "coverage_requirements": [{"required_frame_ids": ["a"]}]},
        output_root=tmp_path / "out", task_context={},
        configuration={"policy": {"refined_splat_cost_usd": 5, "rgbd_view_cost_usd": .05}},
        task_runner=None, pose_runner=None, release_blockers=None)
    assert result["status"] == "completed"
    assert observed[0]["frames"] == frames
    assert result["background"]["splat_kind"] == "atlas_refined"
    assert result["background"]["splat_source_selection"]["selected_real_frame_count"] == 2


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


@pytest.mark.slow
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

    result = native.run_native_splat(frames=frames, result=pose, output_root=tmp_path, policy=policy,
                                    task_context={"context_digest": "sha256:" + "d" * 64}, task_runner=runner)
    manifest = result["splat_manifest"]
    assert (tmp_path / "atlas_refined_splat.spz").read_bytes() == blob
    decoded = read_standard_3dgs_ply(tmp_path / "atlas_refined_splat.ply")
    assert decoded.count == splat.count
    assert manifest["splat_kind"] == "atlas_refined" and manifest["photometric_optimization"] is True
    assert manifest["metric_measurement_proven"] is manifest["simulation_qualified"] is False
    assert calls[0]["task"] == "posedImages2Splats"
    assert (tmp_path / "splats2mesh_task_spec.json").is_file()
    resumed = native.run_native_splat(frames=frames, result=pose, output_root=tmp_path, policy=policy,
        task_context={"context_digest": "sha256:" + "d" * 64}, task_runner=runner)
    assert resumed == result and len(calls) == 1
    (tmp_path / "atlas_refined_splat.spz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="output_changed"):
        native.run_native_splat(frames=frames, result=pose, output_root=tmp_path, policy=policy,
            task_context={"context_digest": "sha256:" + "d" * 64}, task_runner=runner)
