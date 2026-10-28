"""Memory-safe fusion preserves every sample and the original weighted reduction."""
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import atlas_depth_splat as fusion
from tests.test_atlas_depth_splat import camera, frame, plane_depth, retained_pose_root


def samples(seed, count=237):
    rng = np.random.default_rng(seed)
    cells = rng.integers(-4, 5, (count, 3))
    points = (cells + rng.uniform(.1, .9, (count, 3))) * .01
    normals = rng.normal(size=(count, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return {"points": points, "normals": normals, "colors": rng.uniform(0, 1, (count, 3)),
            "weights": 10. ** rng.uniform(-7, 3, count), "slant": rng.uniform(.001, .03, count)}


def legacy_frame_reduction(sample, voxel):
    """The former per-frame bincount reduction, independent of the new merge."""
    weights = sample["weights"]
    columns = np.concatenate([weights[None], (sample["points"] * weights[:, None]).T,
        (sample["colors"] * weights[:, None]).T, (sample["normals"] * weights[:, None]).T,
        np.ones_like(weights)[None], (sample["slant"] * weights)[None]])
    keys, inverse = np.unique(fusion._voxel_keys(sample["points"], voxel), return_inverse=True)
    return keys, np.array([np.bincount(inverse, weights=row, minlength=len(keys)) for row in columns])


def legacy_merge(tables):
    """Former table concatenation/reduction, retained solely as a small oracle."""
    keys = np.concatenate([keys for keys, _ in tables])
    columns = np.concatenate([columns for _, columns in tables], axis=1)
    unique, inverse = np.unique(keys, return_inverse=True)
    return unique, np.array([np.bincount(inverse, weights=row, minlength=len(unique)) for row in columns])


@pytest.mark.parametrize("disk", [False, True])
def test_chunk_merge_matches_original_weighted_sums_with_overlapping_voxels(tmp_path, disk):
    from contextlib import nullcontext

    batches = [samples(17), samples(18), samples(17)]
    reduced = [legacy_frame_reduction(sample, .01) for sample in batches]
    expected_keys, expected_columns = legacy_merge([reduced[0]])
    with fusion._FusionScratch(tmp_path) if disk else nullcontext(None) as scratch:
        sums = fusion.VoxelSums(.01, scratch=scratch)
        sums.add(batches[0])
        np.testing.assert_array_equal(sums.keys, expected_keys)
        np.testing.assert_array_equal(sums.sums.T, expected_columns)
        sums.add(batches[1])
        sums.add(batches[2])
        expected_keys, expected_columns = legacy_merge([(expected_keys, expected_columns), *reduced[1:]])
        np.testing.assert_array_equal(sums.keys, expected_keys)
        np.testing.assert_array_equal(sums.sums.T, expected_columns)
        if disk:
            assert isinstance(sums._columns, np.memmap)
            assert scratch.peak_bytes >= sums._columns.nbytes


def test_merge_does_not_concatenate_full_quantity_tables(tmp_path, monkeypatch):
    with fusion._FusionScratch(tmp_path) as scratch:
        sums = fusion.VoxelSums(.01, scratch=scratch)
        sums.add(samples(1))
        sums._merge()
        sums.add(samples(2))
        sums.add(samples(3))
        original = np.concatenate

        def only_key_concatenation(arrays, *args, **kwargs):
            arrays = list(arrays)
            assert all(array.ndim == 1 for array in arrays), "Merge duplicated the complete 12-column table"
            return original(arrays, *args, **kwargs)

        monkeypatch.setattr(fusion.np, "concatenate", only_key_concatenation)
        sums._merge()
        assert len(sums.keys) and np.isfinite(sums.sums).all()


def test_fixed_flush_bound_preserves_legacy_streaming_reduction_with_skewed_weights(monkeypatch):
    # The old threshold grew with its table and therefore grouped later frames
    # differently. Exercise that difference without a million-row fixture.
    sums = fusion.VoxelSums(.01)
    monkeypatch.setattr(sums, "MIN_MERGE_ROWS", 16)
    old_keys, old_columns = np.empty(0, np.int64), np.empty((12, 0))
    pending, rows, old_merges = [], 0, 0
    for index in range(30):
        batch = samples(index, count=83)
        sums.add(batch)
        reduced = legacy_frame_reduction(batch, .01)
        pending.append(reduced)
        rows += len(reduced[0])
        if rows >= max(16, len(old_keys) // 2):
            old_keys, old_columns = legacy_merge([(old_keys, old_columns), *pending])
            pending, rows = [], 0
            old_merges += 1
    if pending:
        old_keys, old_columns = legacy_merge([(old_keys, old_columns), *pending])
    assert old_merges < 30  # This actually changes addition grouping.
    np.testing.assert_array_equal(sums.keys, old_keys)
    np.testing.assert_allclose(sums.sums.T, old_columns, rtol=1e-12, atol=1e-12)


def test_pending_reduction_flushes_at_fixed_bound_even_with_a_large_old_table(monkeypatch):
    sums = fusion.VoxelSums(.01)
    sums._keys = np.arange(100, dtype=np.int64)
    sums._columns = np.zeros((12, 100))
    monkeypatch.setattr(sums, "MIN_MERGE_ROWS", 4)
    merges = []
    monkeypatch.setattr(sums, "_merge", lambda: merges.append(sums._pending_rows))
    sums.add(samples(27, count=8))
    assert len(merges) == 1 and merges[0] >= 4


def test_disk_backed_depths_preserve_full_grid_fusion_and_scientific_gates(tmp_path):
    cameras = [camera(), camera((.2, 0, 0), yaw_degrees=10)]
    frames = [frame(cam, plane_depth(cam), f"view-{i}") for i, cam in enumerate(cameras)]
    originals = [{key: value.copy() for key, value in row.items() if isinstance(value, np.ndarray)} for row in frames]
    expected, expected_report = fusion.fuse_posed_frames(frames)
    with fusion._FusionScratch(tmp_path) as scratch:
        actual, report = fusion.fuse_posed_frames(frames, scratch=scratch)
        assert actual.count == expected.count
        for field in ("xyz", "f_dc", "opacity", "scales", "quats"):
            np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), rtol=1e-6, atol=1e-7)
        for field in ("frames", "cross_view_agreement", "voxel_size", "depth_smoothing", "depth_consensus"):
            assert report[field] == expected_report[field]
    for row, original in zip(frames, originals, strict=True):
        for key, value in original.items():
            np.testing.assert_array_equal(row[key], value)


def test_disk_backed_fusion_keeps_the_same_cross_view_rejection(tmp_path):
    first, second = camera(), camera((.2, 0, 0), yaw_degrees=10)
    frames = [frame(first, plane_depth(first), "first"), frame(second, plane_depth(second) * 1.2, "second")]
    with fusion._FusionScratch(tmp_path) as scratch:
        with pytest.raises(ValueError, match="atlas_depth_splat_cross_view_inconsistent"):
            fusion.fuse_posed_frames(frames, scratch=scratch)
    assert not list(tmp_path.glob(".atlas-fusion-*"))


def test_scratch_reserves_zeroed_maps_and_releases_owned_files(tmp_path):
    with fusion._FusionScratch(tmp_path) as scratch:
        root = scratch.root
        mapped = scratch.allocate((17, 3))
        assert isinstance(mapped, np.memmap) and mapped.dtype == np.float64
        assert not np.any(mapped)
        filename = Path(mapped.filename)
        assert filename.is_relative_to(root) and filename.is_file()
        assert scratch.peak_bytes >= mapped.nbytes
        mapped[:] = 7
        scratch.release(mapped)
        assert not filename.exists()
    assert not root.exists()


def test_scratch_budget_refuses_before_mapping_or_creating_large_files(tmp_path, monkeypatch):
    monkeypatch.setattr(fusion._FusionScratch, "MAXIMUM_BYTES", 31)
    with fusion._FusionScratch(tmp_path) as scratch:
        monkeypatch.setattr(fusion.np, "memmap", lambda *a, **k: pytest.fail("An over-budget mapping was attempted"))
        with pytest.raises(ValueError, match="atlas_depth_splat_scratch_budget_exceeded"):
            scratch.allocate((4,), dtype=np.float64)
        with pytest.raises(ValueError, match="atlas_depth_splat_scratch_budget_exceeded"):
            scratch.allocate((2**40, 2**40), dtype=np.float64)
        assert scratch.peak_bytes == 0
    assert not list(tmp_path.glob(".atlas-fusion-*"))


@pytest.mark.parametrize("available_gib", [2, 6])
def test_scratch_honors_shared_device_floor_and_competing_reservations(tmp_path, monkeypatch, available_gib):
    from types import SimpleNamespace
    from blueprint_pipeline import control_plane_disk_budget as budget

    original = budget.reserve_control_plane_disk
    ledger = tmp_path / "ledger"
    gib = 1024**3
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES", str(8 * gib))

    def reserve(*args, **kwargs):
        assert args == ("cpu_prestage",) and kwargs["workload"] == "atlas_depth_fusion"
        assert kwargs["expected_bytes"] == 4 * gib
        return original(*args, **{**kwargs, "reservation_root": ledger,
            "disk_usage": lambda _: SimpleNamespace(total=100*gib, free=(8+available_gib)*gib)})

    monkeypatch.setattr(budget, "reserve_control_plane_disk", reserve)
    if available_gib == 2:
        with pytest.raises(budget.ControlPlaneDiskBudgetError, match="control_plane_disk_budget_exceeded"):
            with fusion._FusionScratch(tmp_path / "first"):
                pytest.fail("The bulk disk floor was consumed")
        assert not list((tmp_path / "first").glob(".atlas-fusion-*"))
    else:
        with fusion._FusionScratch(tmp_path / "first") as first:
            assert first.reservation.path.is_file()
            with pytest.raises(budget.ControlPlaneDiskBudgetError, match="control_plane_disk_budget_exceeded"):
                with fusion._FusionScratch(tmp_path / "second"):
                    pytest.fail("Concurrent scratch stole the first reservation")
        assert not first.reservation.path.exists()
        with fusion._FusionScratch(tmp_path / "second") as second:
            assert second.reservation.path.is_file()
        assert not second.reservation.path.exists()
    assert not list(ledger.glob("*.json"))


@pytest.mark.parametrize("fail", [False, True])
def test_writer_owns_and_cleans_fusion_scratch_on_success_and_failure(tmp_path, monkeypatch, fail):
    pose_root = retained_pose_root(tmp_path / "pose")
    result, frames = fusion.load_posed_rgbd(pose_root)
    output = tmp_path / "splat"
    original = fusion.fuse_posed_frames
    roots, reservations = [], []

    def observe_scratch(frames, **kwargs):
        scratch = kwargs.get("scratch")
        assert isinstance(scratch, fusion._FusionScratch)
        roots.append(scratch.root)
        reservations.append(scratch.reservation.path)
        scratch.allocate((41,), dtype=np.float32)
        if fail:
            raise ValueError("deliberate_fusion_failure")
        return original(frames, **kwargs)

    monkeypatch.setattr(fusion, "fuse_posed_frames", observe_scratch)
    if fail:
        with pytest.raises(ValueError, match="deliberate_fusion_failure"):
            fusion.write_depth_splat(frames=frames, result=result, output_root=output)
    else:
        manifest = fusion.write_depth_splat(frames=frames, result=result, output_root=output)
        assert manifest["gaussian_count"] > 0 and manifest["metric_measurement_proven"] is False
        assert manifest["world_frame"] == "atlas_world_rub_y_up"
        assert manifest["claim_scope"] == "development_only"
    assert roots and all(not root.exists() for root in roots)
    assert reservations and all(not path.exists() for path in reservations)
    assert not list(output.glob(".atlas-fusion-*"))
