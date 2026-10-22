"""Generated views must contribute bound geometry, not just downloaded pictures."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline import atlas_scene_completion as completion
from blueprint_pipeline.atlas_depth_splat import posed_frame
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def frame(key="a", x=0):
    return posed_frame(frame_id=key, rgb=np.full((8, 6, 3), 100, np.uint8),
        depth=np.full((8, 6), 2, np.float32), confidence=np.ones((8, 6), np.float32),
        camera={"intrinsics": {"width": 6, "height": 8, "fx": 5, "fy": 5, "cx": 3, "cy": 4},
                "extrinsics": {"position": [x, 0, 0], "quaternion": [0, 0, 0, 1], "coordinateSystem": "rub"}})


def test_completion_camera_pool_preserves_vertical_rays_and_uses_scene_relative_motion():
    first = frame()
    candidates = completion.camera_candidates([first])
    assert len({r["camera_id"] for r in candidates}) == len(candidates)
    for row in candidates:
        k = row["camera"]["intrinsics"]
        assert (k["width"], k["height"]) == (1280, 720)
        if row["purpose"] != "required_full_room_coverage":
            assert k["fy"] / k["height"] == pytest.approx(first["intrinsics"]["fy"] / 8)
    scaled = deepcopy(first)
    scaled["depth"] *= 10
    scaled["position"] *= 10
    scaled["extrinsics"]["position"] = scaled["position"].tolist()
    for before, after in zip(candidates, completion.camera_candidates([scaled]), strict=True):
        assert np.allclose(np.array(before["camera"]["extrinsics"]["position"]) * 10,
                           after["camera"]["extrinsics"]["position"])
    first["depth"][:] = np.nan
    with pytest.raises(ValueError, match="no_context_depth"):
        completion.camera_candidates([first])


def test_full_room_camera_frusta_cover_every_direction():
    from scipy.spatial.transform import Rotation
    cameras = [row["camera"] for row in completion.camera_candidates([frame()])
               if row["purpose"] == "required_full_room_coverage"]
    directions = np.array([[np.cos(p) * np.sin(y), np.sin(p), np.cos(p) * np.cos(y)]
                           for y in np.linspace(-np.pi, np.pi, 73)
                           for p in np.linspace(-np.pi / 2, np.pi / 2, 37)])
    covered = np.zeros(len(directions), bool)
    for camera in cameras:
        local = directions @ Rotation.from_quat(camera["extrinsics"]["quaternion"]).as_matrix()
        k = camera["intrinsics"]
        covered |= ((local[:, 2] < 0) &
                    (np.abs(local[:, 0]) <= -local[:, 2] * k["width"] / (2 * k["fx"]) + 1e-9) &
                    (np.abs(local[:, 1]) <= -local[:, 2] * k["height"] / (2 * k["fy"]) + 1e-9))
    assert np.all(covered), "A room direction would never be generated"


def test_generated_loader_binds_every_pixel_and_keeps_generated_camera(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_depth_splat
    image, depth = tmp_path / "image.png", tmp_path / "depth.exr"
    Image.fromarray(frame()["rgb"]).save(image)
    depth.write_bytes(b"retained-exr")
    monkeypatch.setattr(atlas_depth_splat, "read_exr_depth", lambda _: np.full((8, 6), 2, np.float32))
    cam = {"intrinsics": frame()["intrinsics"], "extrinsics": frame(x=.2)["extrinsics"]}
    result = {"status": "completed", "operation": {"response": {"frames": [{"camera": cam}]}},
              "outputs": [{"role": role, "path": p.name, "sha256": _sha256_file(p)} for role, p in
                          (("frames-0-imageAsset", image), ("frames-0-depth-depthAsset", depth))]}
    result["digest"] = canonical_digest(result, digest_field="digest")
    (tmp_path / "result.json").write_text(json.dumps(result))
    _, frames, images = completion.load_generated_views(tmp_path, ["novel-a"])
    assert np.allclose(frames[0]["position"], [.2, 0, 0])
    assert frames[0]["frame_id"] == "novel-a" and images[0]["novel_view"] is True
    assert np.all(frames[0]["confidence"] == 1)
    depth.write_bytes(b"different-exr")
    with pytest.raises(ValueError, match="output_changed"):
        completion.load_generated_views(tmp_path, ["novel-a"])


def test_completion_reuploads_exact_reviewed_context_and_uses_planner_selection(tmp_path, monkeypatch):
    from blueprint_pipeline import website_image_completion
    frames = [frame(), frame("b", .2)]
    usage, returned = [], []
    for index, f in enumerate(frames):
        image, depth = tmp_path / f"{index}-image.bin", tmp_path / f"{index}-depth.bin"
        Image.fromarray(f["rgb"]).save(image, format="PNG")
        depth.write_bytes(b"\x76\x2f\x31\x01retained-exr")
        usage.append({"frame_id": f["frame_id"], "outputs": [
            {"role": role, "path": p.name, "sha256": _sha256_file(p)} for role, p in (("image", image), ("depth", depth))]})
        returned.append({"imageAsset": {"assetId": "expired-image"}, "depth": {"depthAsset": {"assetId": "expired-depth"}},
                         "camera": {"intrinsics": f["intrinsics"], "extrinsics": f["extrinsics"]}})
    pose = {"digest": "pose", "binding": {"source_video_digest": "source"},
            "operation": {"id": "pose", "done": True, "response": {"frames": returned}},
            "operation_receipt": {"input_usage": {"frames": usage}}}
    required = [row["camera_id"] for row in completion.camera_candidates(frames)
                if row["purpose"] == "required_full_room_coverage"]
    proposal = {"status": "proposed", "selected_camera_ids": ["a-wide", "b-lower"] + required,
                "gap_assessment": ["Complete floor and empty bay"], "remaining_unknowns": ["Generated rear wall"]}
    def generated_views(root, ids):
        generated = [frame(key) for key in ids]
        images = [{"frame_id": key, "image_path": str(tmp_path / "0-image.bin"),
                   "image_digest": _sha256_file(tmp_path / "0-image.bin"), "novel_view": True} for key in ids]
        return {"digest": "generated-depth"}, generated, images
    monkeypatch.setattr(completion, "load_generated_views", generated_views)
    reviews = []
    monkeypatch.setattr(website_image_completion, "verify_completed_background",
                        lambda **kw: reviews.append(kw) or {"status": "passed"})
    calls = []
    args = dict(pose_root=tmp_path, pose=pose, frames=frames,
        review={"automation": {"objects": [], "removal_instruction": "The dishwasher bay is empty"}},
        root=tmp_path / "complete", context={"description": "test"},
        configuration={"remaining_cost_usd": 5, "policy": {"rgbd_view_cost_usd": .05,
            "rate_card_reference": "rates", "reviewed_on": "2026-10-22", "reviewed_by": "test"}},
        task_runner=lambda **kw: calls.append(kw) or {"status": "completed"},
        planner_runner=lambda **_: {"proposal": proposal},
        release_blockers=lambda: ("commit", []), max_polls=1)
    output, report = completion.complete_scene(**args)
    assert len(output) == len(proposal["selected_camera_ids"]) and report["generated_geometry_ready"] is True
    assert len(calls) == 1 and calls[0]["request"]["returnDepth"] is True
    assert calls[0]["price_review"]["maximum_cost_usd"] == len(output) * .05
    assert len(calls[0]["request"]["targetCameras"]) == len(output)
    assert calls[0]["request"]["contextFrames"][0]["imageAsset"]["assetId"] == "upload:context-0-image"
    assert calls[0]["uploads"]["context-0-image"]["sha256"] == usage[0]["outputs"][0]["sha256"]
    from pathlib import Path
    assert Path(calls[0]["uploads"]["context-0-image"]["path"]).suffix == ".png"
    assert Path(calls[0]["uploads"]["context-0-depth"]["path"]).suffix == ".exr"
    assert reviews[0]["frames"][1]["original_image_path"] == str(tmp_path / "1-image.bin")
    proposal["selected_camera_ids"] = ["a-wide", "b-lower"]
    with pytest.raises(ValueError, match="full_room_camera_coverage_missing"):
        completion.complete_scene(**args)
    assert len(calls) == 1
    proposal["selected_camera_ids"] = ["invented-camera"]
    with pytest.raises(ValueError, match="camera_selection_invalid"):
        completion.complete_scene(**args)
    assert len(calls) == 1


@pytest.mark.parametrize("drop_generated", [False, True])
def test_automation_consumes_generated_depth_and_refuses_a_stale_baseline(tmp_path, monkeypatch, drop_generated):
    from blueprint_pipeline import atlas_scene_automation as worker, atlas_background_automation
    from blueprint_pipeline import atlas_scene_measurements, atlas_depth_splat, atlas_result_bundle
    source, novel = frame(), frame("novel", .1)
    pose = {"digest": "pose", "binding": {"source_video_digest": "source"}}
    monkeypatch.setattr(atlas_scene_measurements, "load_views", lambda *a: (pose, [source]))
    monkeypatch.setattr(atlas_scene_measurements, "estimate_sizes", lambda **_: {})
    monkeypatch.setattr(atlas_background_automation, "run_background", lambda **_: {
        "status": "fallback_prepared", "frames": [], "review": {"status": "passed"}})
    monkeypatch.setattr(worker, "_fallback_pose", lambda *a: ((pose, [source], tmp_path), {"status": "completed"}))
    monkeypatch.setattr(completion, "complete_scene", lambda **_: ([novel], {
        "status": "passed", "generation_result_digest": "generated", "review": {"status": "passed"},
        "generated_root": str(tmp_path / "generated")}))
    def build(**kw):
        assert [f["frame_id"] for f in kw["frames"]] == ["a", "novel"]
        assert kw["generated_result_digests"][-1] == "generated"
        assert kw["output_root"].name == "background_completed_splat-" + completion.COMPLETION_REVISION
        return {"digest": "splat", "generated_result_digests": kw["generated_result_digests"],
                "frames": [{"frame_id": f["frame_id"]} for f in (kw["frames"][:1] if drop_generated else kw["frames"])]}
    monkeypatch.setattr(atlas_depth_splat, "write_depth_splat", build)
    exports = []
    monkeypatch.setattr(atlas_result_bundle, "export_pose_bundle", lambda **kw: exports.append(kw) or {"zip_path": "scene.zip"})
    result = worker.run_automation(pose_root=tmp_path, inputs={},
        review={"automation": {"objects": [], "scale_references": [], "removal_instruction": "Empty bay", "mask_margin_fraction": .01}},
        output_root=tmp_path / "out", task_context={}, configuration={"policy": {"rgbd_view_cost_usd": .05}},
        task_runner=None, pose_runner=None, release_blockers=lambda: ("commit", []))
    if drop_generated:
        assert result["status"] == "needs_input" and "did_not_consume_generated_depth" in result["background"]["reason"]
        assert not exports
    else:
        assert result["status"] == "completed" and result["background"]["reconstruction_frame_count"] == 2
        assert exports[0]["generated_roots"] == [tmp_path / "generated"]
