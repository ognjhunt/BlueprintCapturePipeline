"""No provider calls: the draft fusion holds one view at a time and writes the same splat as before.

105 posed 1280x720 views (PR #3467's splat target) were OOM-killed at 5.4 GB on
2026-11-09: every view's pixels were decoded up front and the fusion kept
smoothed and rescaled copies of all of them. ``legacy_fuse`` below is that
all-views fusion, frozen, so the streamed fusion is checked byte for byte.
"""
import gc
import json
import math
import tracemalloc

import numpy as np
import pytest

from blueprint_pipeline import atlas_depth_splat as depth_splat
from tests.atlas_posed_rgbd_synthetic import room_frame, write_room_pose_root


# --- The all-views fusion of origin/main before 2026-11-09, frozen as the reference. ---

def legacy_cross_view_agreement(frames, samples):
    centres = np.array([frame["position"] for frame in frames])
    pairs, residuals = [], []
    for index, (source, sample) in enumerate(zip(frames, samples)):
        picked = sample["points"]
        if not len(picked):
            continue
        order = np.argsort(np.linalg.norm(centres - centres[index], axis=1), kind="stable")
        for target in (frames[i] for i in order[order != index][:depth_splat.CROSS_VIEW_NEIGHBOURS]):
            k = target["intrinsics"]
            local = (picked - target["position"]) @ target["rotation"]
            planar = -local[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                column = np.floor(local[:, 0] / planar * k["fx"] + k["cx"])
                row = np.floor(-local[:, 1] / planar * k["fy"] + k["cy"])
                inside = (planar > 0) & (column >= 0) & (column < k["width"]) & (row >= 0) & (row < k["height"])
            observed = np.full(len(picked), np.nan)
            index_pair = (row[inside].astype(np.int64), column[inside].astype(np.int64))
            observed[inside] = np.where(target["confidence"][index_pair] > 0, target["depth"][index_pair], np.nan)
            seen = np.isfinite(observed) & (observed > 0)
            visible = seen & (observed >= planar * (1.0 - depth_splat.MAX_MEDIAN_CROSS_VIEW_RESIDUAL))
            if visible.sum() < 0.01 * len(picked):
                continue
            relative = np.abs(observed[visible] - planar[visible]) / planar[visible]
            residuals.append(relative)
            pairs.append({"source": source["frame_id"], "target": target["frame_id"],
                          "overlap_samples": int(visible.sum()), "occluded_samples": int((seen & ~visible).sum()),
                          "median_relative_residual": float(np.median(relative))})
    pooled = float(np.median(np.concatenate(residuals))) if residuals else None
    return {"pairs": pairs, "median_relative_residual": pooled, "limit": depth_splat.MAX_MEDIAN_CROSS_VIEW_RESIDUAL}


def legacy_voting_views(frames, index, points, centres):
    if not len(points):
        return []
    count = min(depth_splat.DEPTH_CONSENSUS_RANKING_SAMPLES, len(points))
    pick = points[np.linspace(0, len(points) - 1, count).astype(np.int64)]
    order = np.argsort(np.linalg.norm(centres - centres[index], axis=1), kind="stable")
    counted = []
    with np.errstate(invalid="ignore"):
        for other in order[order != index]:
            counted.append((int((np.abs(depth_splat._seen_ratio(frames[other], pick) - 1.0)
                                 <= depth_splat.DEPTH_CONSENSUS_GATE).sum()), int(other)))
    ranked = sorted(counted, key=lambda row: -row[0])
    return [other for count, other in ranked[:depth_splat.CROSS_VIEW_NEIGHBOURS] if count]


def legacy_consensus_depths(frames):
    frames = [dict(frame) for frame in frames]
    centres = np.array([frame["position"] for frame in frames])
    start, stride = depth_splat.DEPTH_CONSENSUS_STRIDE // 2, depth_splat.DEPTH_CONSENSUS_STRIDE
    voted_by_pass, voters_by_pass = [], []
    for _ in range(depth_splat.DEPTH_CONSENSUS_PASSES):
        ratios, voted, voters = [], [], []
        for index, frame in enumerate(frames):
            k = frame["intrinsics"]
            depth = frame["depth"][start::stride, start::stride].astype(np.float64)
            valid = np.isfinite(depth) & (depth > 0) & (frame["confidence"][start::stride, start::stride] > 0)
            rows, columns = np.nonzero(valid)
            z = depth[rows, columns]
            u = (columns * stride + start + 0.5 - k["cx"]) / k["fx"]
            v = (rows * stride + start + 0.5 - k["cy"]) / k["fy"]
            points = np.stack([u * z, -v * z, -z], axis=1) @ frame["rotation"].T + frame["position"]
            votes = [np.ones(len(z))]
            chosen = legacy_voting_views(frames, index, points, centres)
            for other in chosen:
                seen = depth_splat._seen_ratio(frames[other], points)
                with np.errstate(invalid="ignore"):
                    votes.append(np.where(np.abs(seen - 1.0) <= depth_splat.DEPTH_CONSENSUS_GATE, seen, np.nan))
            voters.append(len(chosen))
            stack = np.stack(votes)
            counted = np.isfinite(stack).sum(axis=0) >= 2
            correction, voted_grid = np.zeros(depth.shape), np.zeros(depth.shape, bool)
            correction[rows[counted], columns[counted]] = np.nanmedian(stack[:, counted], axis=0) - 1.0
            voted_grid[rows[counted], columns[counted]] = True
            grid_depth = np.where(valid, depth, np.nan)
            ratios.append((depth_splat._surface_smoothed(correction, voted_grid, grid_depth), grid_depth))
            voted.append(float(counted.sum()) / depth.size)
        for frame, (ratio, grid_depth) in zip(frames, ratios):
            full = depth_splat._upsampled_on_surface(ratio, grid_depth, frame["depth"])
            frame["depth"] = (frame["depth"] * full).astype(frame["depth"].dtype)
        voted_by_pass.append(round(float(np.median(voted)), 4))
        voters_by_pass.append(float(np.median(voters)))
    return frames, {"basis": "median_of_most_agreeing_views", "passes": depth_splat.DEPTH_CONSENSUS_PASSES,
                    "neighbours": depth_splat.CROSS_VIEW_NEIGHBOURS, "relative_gate": depth_splat.DEPTH_CONSENSUS_GATE,
                    "stride_pixels": stride, "sigma_grid_pixels": depth_splat.DEPTH_CONSENSUS_SIGMA,
                    "ranking_samples": depth_splat.DEPTH_CONSENSUS_RANKING_SAMPLES,
                    "median_voted_grid_share_by_pass": voted_by_pass,
                    "median_voting_views_by_pass": voters_by_pass}


def legacy_fuse(frames, voxel_footprint_multiple=2.0):
    frames = [dict(frame) for frame in frames]
    frames = [{**frame, "depth": depth_splat.smoothed_depth(frame["depth"])} for frame in frames]
    samples = [depth_splat._subsample(depth_splat.surface_samples(frame)) for frame in frames]
    agreement = legacy_cross_view_agreement(frames, samples)
    if (agreement["median_relative_residual"] is not None
            and agreement["median_relative_residual"] > depth_splat.MAX_MEDIAN_CROSS_VIEW_RESIDUAL):
        raise ValueError("atlas_depth_splat_cross_view_inconsistent")
    frames, consensus = legacy_consensus_depths(frames)
    samples = [depth_splat._subsample(depth_splat.surface_samples(frame)) for frame in frames]
    consensus["median_relative_residual_after"] = legacy_cross_view_agreement(frames, samples)["median_relative_residual"]
    footprints = np.concatenate([sample["footprint"] for sample in samples])
    voxel = float(voxel_footprint_multiple * np.median(footprints))
    sums = depth_splat.VoxelSums(voxel)
    for frame in frames:
        sums.add(depth_splat.surface_samples(frame))
    fused = sums.fused()
    report = {"frames": [{"frame_id": frame["frame_id"], "usable_pixel_fraction": sample["usable"] / frame["depth"].size}
                         for frame, sample in zip(frames, samples)],
              "cross_view_agreement": agreement, "voxel_size": voxel,
              "voxel_footprint_multiple": voxel_footprint_multiple,
              "depth_smoothing": {"basis": "local_plane_fit", "sigma_pixels": depth_splat.DEPTH_SMOOTHING_SIGMA_PIXELS,
                                  "relative_residual": depth_splat.DEPTH_SMOOTHING_RESIDUAL,
                                  "relative_tolerance": depth_splat.DEPTH_SMOOTHING_TOLERANCE},
              "depth_consensus": consensus,
              "median_observations_per_gaussian": float(np.median(fused["observations"]))}
    return depth_splat.depth_fused_splat(fused, voxel), report


# --- Helpers ---

def eager(frame):
    """The dict ``posed_frame`` returned for a retained view before views were read lazily."""
    return {key: frame[key] for key in frame}


def assert_same_splat(first, second):
    (splat, report), (reference, reference_report) = first, second
    assert splat.count == reference.count
    for field in ("xyz", "opacity", "f_dc", "scales", "quats"):
        assert getattr(splat, field).tobytes() == getattr(reference, field).tobytes(), field
    assert json.dumps(report) == json.dumps(reference_report)


def room_views(count, width=160, height=96):
    frames = []
    for index in range(count):
        cam, rgb, depth, confidence = room_frame(index, count, width=width, height=height)
        frames.append(depth_splat.posed_frame(frame_id=f"view-{index:03d}", rgb=rgb, depth=depth,
                                              confidence=confidence, camera=cam))
    return frames


# --- Byte identity ---

@pytest.mark.parametrize("resident_bytes,block_bytes", [(0, 1), (3 * 160 * 96 * 4, 200_000), (None, None)])
def test_streamed_fusion_writes_the_all_views_splat_byte_for_byte(monkeypatch, resident_bytes, block_bytes):
    """Rebuilt, partly kept and fully kept working depths, voted one view or many at a time, all agree."""
    if resident_bytes is not None:
        monkeypatch.setattr(depth_splat, "FUSION_RESIDENT_DEPTH_BYTES", resident_bytes)
        monkeypatch.setattr(depth_splat, "CONSENSUS_VOTE_BLOCK_BYTES", block_bytes)
    frames = room_views(10)
    reference = legacy_fuse(frames)
    assert reference[1]["depth_consensus"]["median_voted_grid_share_by_pass"][0] > 0.3, "consensus has work to do"
    assert_same_splat(depth_splat.fuse_posed_frames(frames), reference)


def test_existing_scenes_fuse_exactly_as_before():
    from tests.test_atlas_depth_splat import camera, frame, plane_depth

    first, second = camera(), camera((0.2, 0.0, 0.0), yaw_degrees=10.0)
    occluded = camera((0.5, 0.0, 0.0))
    k = occluded["intrinsics"]
    panel_x = 0.5 + (np.arange(k["width"]) + 0.5 - k["cx"]) / k["fx"] * 0.5
    panel = np.where(((panel_x >= 0.35) & (panel_x <= 0.75))[None, :], 0.5, 2.0)
    scenes = [
        [frame(first, plane_depth(first) * 1.015, "a"), frame(second, plane_depth(second) * 0.985, "b")],
        [frame(first, plane_depth(first), "a"),
         frame(occluded, np.broadcast_to(panel, (k["height"], k["width"])).astype(np.float32), "b")],
        [frame(cam, plane_depth(cam), f"f{index}") for index, cam in enumerate(
            camera((0.05 * index, 0.0, 0.0), width=16, height=16, focal=20.0) for index in range(12))],
    ]
    patch = np.zeros((64, 48), np.float32)
    patch[30:34, 20:24] = 1.0
    scenes.append([*scenes[0], *[{**frame(cam, plane_depth(cam), f"repair-{index}"), "confidence": patch}
                                 for index, cam in enumerate(camera((0.01 * index, 0.02, 0.0)) for index in range(9))]])
    for frames in scenes:
        assert_same_splat(depth_splat.fuse_posed_frames(frames), legacy_fuse(frames))


def test_consensus_depths_rescale_exactly_as_before():
    frames = room_views(8)
    rescaled, report = depth_splat.consensus_depths(frames)
    reference, reference_report = legacy_consensus_depths(frames)
    assert report == reference_report
    for after, expected in zip(rescaled, reference):
        assert after["depth"].tobytes() == expected["depth"].tobytes()
        assert list(after) == list(expected)


def test_a_rescaled_grid_is_the_grid_of_the_rescaled_depth():
    """The consensus grids advance without rebuilding any view's full-resolution depth."""
    rng = np.random.default_rng(3)
    depth = (2.0 + rng.normal(0, 0.05, (96, 160))).astype(np.float32)
    depth[10:30, 40:90] = 1.0
    depth[50:60, :20] = np.nan
    depth[70:, 100:] = 0.0
    confidence = (rng.uniform(size=depth.shape) > 0.1).astype(np.float32)
    folded = depth_splat._folded(depth, confidence)
    grid = depth_splat._consensus_grid(folded)
    ratio = 1.0 + rng.uniform(-0.04, 0.04, grid.shape)
    rescaled = depth_splat._consensus_grid(depth_splat._rescaled(folded, ratio))
    assert np.array_equal(depth_splat._rescaled_grid(grid, ratio, folded.dtype), rescaled, equal_nan=True)
    # The confidence-folded depth rescales exactly as the depth beside its confidence.
    unfolded = depth_splat._rescaled(depth, ratio, confidence)
    assert np.array_equal(depth_splat._folded(unfolded, confidence), depth_splat._rescaled(folded, ratio),
                          equal_nan=True)


# --- Retained views stay in their files ---

def test_retained_views_are_checked_up_front_and_decoded_only_when_read(tmp_path):
    pose_root = write_room_pose_root(tmp_path / "pose", 3, width=64, height=48)
    result, frames = depth_splat.load_posed_rgbd(pose_root)
    assert all(isinstance(frame, depth_splat.PosedFrame) for frame in frames)
    first = frames[0]
    cam, rgb, depth, confidence = room_frame(0, 3, width=64, height=48)
    expected = depth_splat.posed_frame(frame_id="view-000", rgb=rgb, depth=depth.astype(np.float16).astype(np.float32),
                                       confidence=confidence, camera=cam)
    read_rgb, read_depth, read_confidence = depth_splat.frame_arrays(first, "rgb", "depth", "confidence")
    assert first._values.keys().isdisjoint({"rgb", "depth", "confidence"}), "frame_arrays keeps nothing"
    assert np.array_equal(read_rgb, expected["rgb"]) and np.array_equal(read_confidence, expected["confidence"])
    assert np.array_equal(read_depth, expected["depth"], equal_nan=True)
    assert list(first) == list(expected) and len(first) == len(expected) and "depth" in first
    assert first["rgb"] is first["rgb"], "reading a key keeps it, as the dict did"
    # Callers annotate and replace fields as they did on the dict.
    first.update(image_path="image.png")
    first["depth"] = np.zeros((48, 64), np.float32)
    assert first["image_path"] == "image.png" and not depth_splat.frame_arrays(first, "depth")[0].any()
    assert list(first)[-1] == "image_path" and first.get("keep_mask") is None
    assert {**frames[1]}.keys() == expected.keys()
    assert frames[1] != frames[2] and frames[1] == frames[1]


def test_a_retained_file_changed_after_loading_is_refused_when_read(tmp_path):
    pose_root = write_room_pose_root(tmp_path / "pose", 2, width=64, height=48)
    _, frames = depth_splat.load_posed_rgbd(pose_root)
    depth = pose_root / "atlas_pose" / "outputs" / "000001" / "depth.bin"
    depth.write_bytes(depth.read_bytes() + b"\0")
    depth_splat.frame_arrays(frames[1], "rgb")
    with pytest.raises(ValueError, match="atlas_depth_splat_output_changed"):
        depth_splat.frame_arrays(frames[1], "depth")


def test_retained_views_fuse_into_the_splat_their_decoded_pixels_give(tmp_path):
    pose_root = write_room_pose_root(tmp_path / "pose", 6, width=160, height=96)
    _, frames = depth_splat.load_posed_rgbd(pose_root)
    streamed = depth_splat.fuse_posed_frames(frames)
    assert all(frame._values.keys().isdisjoint({"rgb", "depth", "confidence"}) for frame in frames)
    assert_same_splat(streamed, legacy_fuse([eager(frame) for frame in frames]))


def peak_load_and_fuse_bytes(pose_root, load=depth_splat.load_posed_rgbd, fuse=depth_splat.fuse_posed_frames):
    """Traced peak of reading a retained pose and fusing it, with coarse voxels so the voxel sums stay small."""
    gc.collect()
    tracemalloc.start()
    try:
        _, frames = load(pose_root)
        fuse(frames, voxel_footprint_multiple=40.0)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_fusion_memory_does_not_grow_with_the_number_of_views(tmp_path, monkeypatch):
    """Twelve more retained views add far less than half a view's pixels each to the peak.

    Reading and fusing a pose held every view's image, depth and confidence plus a
    smoothed and a rescaled depth of each: about 1.4 views' pixels per added view
    at this size. Now only consensus grids (1/16 of the pixels), fixed-size
    subsamples and voxel sums grow with the views; coarse voxels and small
    subsamples keep those small here. The kept working depths and the vote blocks
    are fixed budgets, set small here.
    """
    monkeypatch.setattr(depth_splat, "FUSION_RESIDENT_DEPTH_BYTES", 0)
    monkeypatch.setattr(depth_splat, "CONSENSUS_VOTE_BLOCK_BYTES", 2**20)
    monkeypatch.setattr(depth_splat, "CROSS_VIEW_SAMPLES", 500)
    width, height = 320, 180
    roots = [write_room_pose_root(tmp_path / f"pose-{views}", 16, width=width, height=height, views=views)
             for views in (2, 4, 16)]
    peak_load_and_fuse_bytes(roots[0])  # first-call allocations (imports, caches) are not the views'
    few, many = peak_load_and_fuse_bytes(roots[1]), peak_load_and_fuse_bytes(roots[2])
    view_bytes = width * height * (3 + 4 + 4)
    assert many - few < 12 * 0.5 * view_bytes, (few, many, view_bytes)


@pytest.mark.slow
def test_a_hundred_atlas_views_fuse_within_a_bounded_peak(tmp_path):
    """105 posed 1280x720 views, as in the OOM of 2026-11-09, fuse well inside the host's memory.

    Measured 2026-11-09 before the views streamed: 2.7 GB traced peak (3.0 GB
    resident), 1.0 GB of it the decoded views alone. Streamed: about 0.8 GB until
    the voxel sums start, and the peak is the voxel sums and splat of this
    3.6 M-Gaussian draft, which follow the scene's voxels, not its views.
    """
    pose_root = write_room_pose_root(tmp_path / "pose", 105)
    gc.collect()
    tracemalloc.start()
    try:
        _, frames = depth_splat.load_posed_rgbd(pose_root)
        loaded = tracemalloc.get_traced_memory()[0]
        splat, report = depth_splat.fuse_posed_frames(frames)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert loaded < 16 * 2**20
    assert report["depth_consensus"]["median_relative_residual_after"] < report["cross_view_agreement"][
        "median_relative_residual"]
    assert splat.count > 1_000_000 and math.isfinite(report["voxel_size"])
    assert peak < 1.75 * 2**30, peak / 2**20
