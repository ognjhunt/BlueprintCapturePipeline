"""Preserve exact voxel accumulation while avoiding copies of all twelve columns."""
import numpy as np
import pytest

from blueprint_pipeline import atlas_depth_splat as depth


@pytest.mark.parametrize("chunk_rows", [1, 7, 65_536])
def test_chunked_tables_match_the_previous_bincount_order_exactly(monkeypatch, chunk_rows):
    rng = np.random.default_rng(62)
    tables = []
    for count in (17, 24, 1, 19):
        keys = np.sort(rng.choice(40, count, replace=False)).astype(np.int64)
        columns = rng.normal(size=(12, count))
        tables.append((keys, columns))
    expected_keys, expected_columns = depth._sum_by_key(
        np.concatenate([keys for keys, _ in tables]),
        np.concatenate([columns for _, columns in tables], axis=1))
    sums = depth.VoxelSums(.01)
    sums._keys, sums._columns = tables[0]
    sums._pending = tables[1:]
    sums._pending_rows = sum(len(keys) for keys, _ in tables[1:])
    monkeypatch.setattr(sums, "MERGE_CHUNK_ROWS", chunk_rows)
    original = np.concatenate

    def no_column_copy(arrays, *args, **kwargs):
        assert all(array.ndim == 1 for array in arrays), "merge copied the full column tables"
        return original(arrays, *args, **kwargs)

    monkeypatch.setattr(np, "concatenate", no_column_copy)
    np.testing.assert_array_equal(sums.keys, expected_keys)
    np.testing.assert_array_equal(sums.sums, expected_columns.T)
    assert sums._pending == [] and sums._pending_rows == 0


def test_chunking_preserves_existing_totals_across_multiple_flushes(monkeypatch):
    rng = np.random.default_rng(80)
    samples = [{"points": rng.uniform(-1, 1, (400, 3)), "colors": rng.uniform(0, 1, (400, 3)),
                "normals": rng.normal(size=(400, 3)), "weights": rng.uniform(.1, 1, 400),
                "slant": rng.uniform(.001, .01, 400)} for _ in range(5)]
    monkeypatch.setattr(depth.VoxelSums, "MIN_MERGE_ROWS", 1)
    reference, chunked = depth.VoxelSums(.2), depth.VoxelSums(.2)
    previous_keys, previous_columns = np.empty(0, np.int64), np.empty((12, 0))
    for sample in samples:
        # Reconstruct the previous implementation's per-frame reduction and
        # merge order independently of the new merge method.
        weights = sample["weights"]
        columns = np.concatenate([weights[None], (sample["points"] * weights[:, None]).T,
            (sample["colors"] * weights[:, None]).T, (sample["normals"] * weights[:, None]).T,
            np.ones_like(weights)[None], (sample["slant"] * weights)[None]])
        keys, columns = depth._sum_by_key(depth._voxel_keys(sample["points"], reference.voxel), columns)
        previous_keys, previous_columns = depth._sum_by_key(
            np.concatenate([previous_keys, keys]), np.concatenate([previous_columns, columns], axis=1))
        chunked.MERGE_CHUNK_ROWS = 3
        chunked.add(sample)
    np.testing.assert_array_equal(chunked.keys, previous_keys)
    np.testing.assert_array_equal(chunked.sums, previous_columns.T)
