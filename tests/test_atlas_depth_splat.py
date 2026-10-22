"""No provider calls: Atlas posed RGBD fuses into a draft splat that splats2Mesh accepts."""
import json
import math
import os

import numpy as np
import pytest

from blueprint_pipeline import atlas_depth_splat as depth_splat
from blueprint_pipeline import website_atlas_tasks as tasks
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import read_standard_3dgs_ply


def camera(position=(0.0, 0.0, 0.0), yaw_degrees=0.0, width=48, height=64, focal=60.0):
    half = math.radians(yaw_degrees) / 2
    return {"extrinsics": {"coordinateSystem": "rub", "position": list(position),
                           "quaternion": [0.0, math.sin(half), 0.0, math.cos(half)]},
            "intrinsics": {"fx": focal, "fy": focal, "cx": width / 2, "cy": height / 2,
                           "width": width, "height": height}}


def plane_depth(cam, plane_z=-2.0):
    k, e = cam["intrinsics"], cam["extrinsics"]
    u = (np.arange(k["width"]) + 0.5 - k["cx"]) / k["fx"]
    v = (np.arange(k["height"]) + 0.5 - k["cy"]) / k["fy"]
    rays = np.stack(np.broadcast_arrays(u[None, :], -v[:, None], -1.0), axis=-1)
    rays = rays @ depth_splat._rotation_matrix(e["quaternion"]).T
    return ((plane_z - e["position"][2]) / rays[..., 2]).astype(np.float32)


def frame(cam, depth, frame_id="f0"):
    height, width = depth.shape
    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[..., 0] = 200
    rgb[..., 1] = (np.arange(width) * 4 % 256)[None, :]
    return depth_splat.posed_frame(frame_id=frame_id, rgb=rgb, depth=depth,
                                   confidence=np.full(depth.shape, 255, np.uint8), camera=cam)


def test_overlapping_views_of_one_plane_agree_and_fuse_onto_it():
    first, second = camera(), camera((0.2, 0.0, 0.0), yaw_degrees=10.0)
    frames = [frame(first, plane_depth(first), "a"), frame(second, plane_depth(second), "b")]
    splat, report = depth_splat.fuse_posed_frames(frames)
    assert report["cross_view_agreement"]["median_relative_residual"] < 0.01
    assert {(row["source"], row["target"]) for row in report["cross_view_agreement"]["pairs"]} == {("a", "b"), ("b", "a")}
    assert np.allclose(splat.xyz[:, 2], -2.0, atol=1e-5)
    # Discs lie in the plane: the thin local z axis is rotated onto the plane normal.
    assert np.allclose(np.abs(splat.quats[:, 0]), 1.0, atol=1e-5)
    assert np.all(splat.scales[:, 2] < splat.scales[:, 0])
    assert report["voxel_size"] == pytest.approx(2.0 * np.median([2.0 / 60.0]), rel=0.1)


def test_depth_discontinuities_do_not_become_flying_points():
    cam = camera()
    depth = np.full((64, 48), 2.0, np.float32)
    depth[:, :24] = 1.0
    splat, _ = depth_splat.fuse_posed_frames([frame(cam, depth)])
    z = -splat.xyz[:, 2]
    assert np.all((np.abs(z - 1.0) < 1e-4) | (np.abs(z - 2.0) < 1e-4))


def test_views_whose_depths_disagree_fail_closed():
    first, second = camera(), camera((0.2, 0.0, 0.0), yaw_degrees=10.0)
    frames = [frame(first, plane_depth(first), "a"), frame(second, plane_depth(second) * 1.2, "b")]
    with pytest.raises(ValueError, match="atlas_depth_splat_cross_view_inconsistent"):
        depth_splat.fuse_posed_frames(frames)


def test_spz_round_trip_is_reproducible_and_within_quantization():
    rng = np.random.default_rng(7)
    count = 257
    quats = rng.normal(size=(count, 4))
    splat = depth_splat.SplatData(count=count, xyz=rng.uniform(-3, 3, (count, 3)).astype(np.float32),
                                  opacity=rng.uniform(-2, 4, count).astype(np.float32),
                                  f_dc=rng.uniform(-1.7, 1.7, (count, 3)).astype(np.float32),
                                  scales=rng.uniform(-9, -2, (count, 3)).astype(np.float32),
                                  quats=(quats / np.linalg.norm(quats, axis=1, keepdims=True)).astype(np.float32),
                                  properties=depth_splat.PLY_PROPERTIES)
    data = depth_splat.encode_spz(splat)
    assert data == depth_splat.encode_spz(splat)
    decoded = depth_splat.decode_spz(data)
    assert decoded.count == count
    assert np.max(np.abs(decoded.xyz - splat.xyz)) <= 0.5 / 4096 + 1e-6
    assert np.max(np.abs(decoded.f_dc - splat.f_dc)) <= 0.5 / (0.15 * 255) + 1e-6
    assert np.max(np.abs(decoded.scales - splat.scales)) <= 0.5 / 16 + 1e-6
    alignment = np.abs(np.sum(decoded.quats * splat.quats, axis=1))
    assert np.min(alignment) > 0.99
    splat.xyz[0, 0] = 5000.0
    with pytest.raises(ValueError, match="spz_position_out_of_range"):
        depth_splat.encode_spz(splat)


def test_mesh_spec_renders_capture_cameras_within_atlas_limits(tmp_path):
    spz = tmp_path / "splat.spz"
    spz.write_bytes(b"splat")
    portrait = camera(width=720, height=1280, focal=1284.0)
    frames = [frame(portrait, np.ones((1280, 720), np.float32))]
    spec = depth_splat.splats2mesh_task_spec(frames=frames, spz=spz, source_video_digest="sha256:" + "0" * 64)
    tasks.validate_task_request(spec["task"], spec["request"])
    rendered = spec["request"]["cameras"][0]
    assert (rendered["intrinsics"]["width"], rendered["intrinsics"]["height"]) == (576, 1024)
    assert rendered["intrinsics"]["fx"] == pytest.approx(1284.0 * 0.8)
    assert rendered["intrinsics"]["cy"] == pytest.approx(640.0 * 0.8)
    assert rendered["extrinsics"] == portrait["extrinsics"]
    assert spec["uploads"]["splat"]["sha256"].startswith("sha256:")
    frames.append(frame(camera(), np.ones((64, 48), np.float32)))
    with pytest.raises(ValueError, match="camera_grids_differ"):
        depth_splat.splats2mesh_task_spec(frames=frames, spz=spz, source_video_digest="sha256:" + "0" * 64)


def retained_pose_root(root):
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2

    cameras, usage = [], []
    for index, cam in enumerate([camera(), camera((0.2, 0.0, 0.0), yaw_degrees=10.0)]):
        item = frame(cam, plane_depth(cam), f"f{index}")
        folder = root / "atlas_pose" / "outputs" / f"{index:06d}"
        folder.mkdir(parents=True)
        outputs = []
        for role, suffix, pixels, options in (
                ("image", ".png", np.ascontiguousarray(item["rgb"][..., ::-1]), []),
                ("depth", ".exr", item["depth"], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]),
                ("confidence", ".png", np.full(item["depth"].shape, 255, np.uint8), [])):
            encoded = folder / f"{role}{suffix}"
            assert cv2.imwrite(str(encoded), pixels, options)
            path = encoded.rename(folder / f"{role}.bin")
            outputs.append({"role": role, "path": str(path.relative_to(root)),
                            "sha256": depth_splat._sha256_file(path)})
        cameras.append({"camera": cam})
        usage.append({"frame_id": f"f{index}", "outputs": outputs})
    result = {"binding": {"source_video_digest": "sha256:" + "1" * 64}, "depth_unit": "provider_world_units",
              "scale_status": "unscaled_provider_world_units",
              "operation": {"id": "op_retained", "response": {"frames": cameras}},
              "operation_receipt": {"input_usage": {"frames": usage}}}
    result["digest"] = canonical_digest(result, digest_field="digest")
    (root / depth_splat.POSE_RESULT).write_text(json.dumps(result))
    return root


def test_retained_pose_result_builds_bound_splat_files(tmp_path):
    pose_root = retained_pose_root(tmp_path / "pose")
    manifest = depth_splat.build_depth_splat(pose_root, tmp_path / "splat")
    out = tmp_path / "splat"
    assert manifest["digest"] == canonical_digest(manifest, digest_field="digest")
    assert manifest["claim_scope"] == "development_only" and manifest["metric_measurement_proven"] is False
    for row in manifest["outputs"]:
        assert depth_splat._sha256_file(out / row["path"]) == row["sha256"]
    assert read_standard_3dgs_ply(out / "atlas_depth_splat.ply").count == manifest["gaussian_count"]
    assert depth_splat.decode_spz((out / "atlas_depth_splat.spz").read_bytes()).count == manifest["gaussian_count"]
    spec = json.loads((out / "splats2mesh_task_spec.json").read_text())
    assert spec["source_video_digest"] == "sha256:" + "1" * 64
    assert spec["uploads"]["splat"]["sha256"] == manifest["outputs"][1]["sha256"]


def test_changed_retained_depth_is_refused(tmp_path):
    pose_root = retained_pose_root(tmp_path / "pose")
    depth = pose_root / "atlas_pose" / "outputs" / "000001" / "depth.bin"
    depth.write_bytes(depth.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="atlas_depth_splat_output_changed"):
        depth_splat.load_posed_rgbd(pose_root)


def test_a_nearer_surface_hiding_points_is_occlusion_not_disagreement():
    # Camera b sees a panel at depth 0.5 in front of most of the background camera a sees.
    first, second = camera(), camera((0.5, 0.0, 0.0))
    k = second["intrinsics"]
    panel_x = 0.5 + (np.arange(k["width"]) + 0.5 - k["cx"]) / k["fx"] * 0.5
    occluded = np.where(((panel_x >= 0.35) & (panel_x <= 0.75))[None, :], 0.5, 2.0)
    frames = [frame(first, plane_depth(first), "a"),
              frame(second, np.broadcast_to(occluded, (k["height"], k["width"])).astype(np.float32), "b")]
    _, report = depth_splat.fuse_posed_frames(frames)
    agreement = report["cross_view_agreement"]
    assert agreement["median_relative_residual"] < 0.01
    hidden = next(pair for pair in agreement["pairs"] if (pair["source"], pair["target"]) == ("a", "b"))
    assert hidden["occluded_samples"] > hidden["overlap_samples"]


def test_cross_view_work_is_bounded_to_nearest_views():
    cams = [camera((0.05 * index, 0.0, 0.0), width=16, height=16, focal=20.0) for index in range(12)]
    frames = [frame(cam, plane_depth(cam), f"f{index}") for index, cam in enumerate(cams)]
    _, report = depth_splat.fuse_posed_frames(frames)
    pairs = report["cross_view_agreement"]["pairs"]
    assert len(pairs) == 12 * depth_splat.CROSS_VIEW_NEIGHBOURS
    assert [pair["target"] for pair in pairs if pair["source"] == "f0"] == [f"f{index}" for index in range(1, 9)]


def test_streamed_voxel_sums_match_one_batch_fusion():
    first, second = camera(), camera((0.2, 0.0, 0.0), yaw_degrees=10.0)
    samples = [depth_splat.surface_samples(frame(cam, plane_depth(cam))) for cam in (first, second)]
    streamed, batched = depth_splat.VoxelSums(0.05), depth_splat.VoxelSums(0.05)
    for sample in samples:
        streamed.add(sample)
    batched.add({key: np.concatenate([sample[key] for sample in samples]) for key in samples[0]})
    assert np.array_equal(streamed.keys, batched.keys) and np.allclose(streamed.sums, batched.sums)
