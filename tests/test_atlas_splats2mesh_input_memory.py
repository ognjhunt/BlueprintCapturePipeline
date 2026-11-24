"""No provider calls: the draft's mesh input is merged a column and a run of rows at a time, to the same bits.

itest-cce2beee's 16.2 M-Gaussian draft (2026-11-24, 204 posed 720x1280 views) peaked at
3.34 GiB in ``splats2mesh_input``: it held the positions widened to float64, each grid's
keys with their temporaries, and the six normal products stacked beside their list. A room
of about 400 views (some 30 M Gaussians) would have needed about 6 GiB, above the Atlas
preview lane's 5.5 GB limit. ``legacy_splats2mesh_input`` below is origin/main's before that
change, frozen, so the rework is checked bit for bit.
"""
import dataclasses
import gc
import tracemalloc

import numpy as np
import pytest

from blueprint_pipeline import atlas_depth_splat as depth_splat
from blueprint_pipeline.gaussian_splat_decode import SplatData

# --- origin/main before 2026-11-24's mesh-input rework, frozen as the reference. ---


def legacy_disc_normals(splat):
    w, x, y, z = (splat.quats[:, index].astype(np.float64) for index in range(4))
    normal = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)
    return normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)


def legacy_splats2mesh_input(splat, voxel, *, limit=None):
    limit = depth_splat.SPLATS2MESH_MAX_GAUSSIANS if limit is None else limit
    if splat.count <= limit:
        return splat, voxel
    points = splat.xyz.astype(np.float64)
    for multiple in depth_splat._MESH_INPUT_GRID_MULTIPLES:
        size = voxel * multiple
        keys, inverse = np.unique(depth_splat._voxel_keys(points, size), return_inverse=True)
        if len(keys) <= limit:
            break
    else:
        raise ValueError("atlas_depth_splat_mesh_input_exceeds_limit")
    inverse = inverse.reshape(-1)
    weight = np.clip((splat.opacity_sigmoid.astype(np.float64) - 0.5) / 0.49, 1e-3, 1.0)
    total = np.bincount(inverse, weights=weight, minlength=len(keys))

    def mean(values):
        return np.stack([np.bincount(inverse, weights=weight * values[:, column], minlength=len(keys))
                         for column in range(values.shape[1])], axis=1) / total[:, None]

    normals = legacy_disc_normals(splat)
    pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    second = mean(np.stack([normals[:, i] * normals[:, j] for i, j in pairs], axis=1))
    values, vectors = np.linalg.eigh(second[:, [0, 1, 2, 1, 3, 4, 2, 4, 5]].reshape(-1, 3, 3))
    spans = np.exp(splat.scales[:, :1].astype(np.float64)) / depth_splat.DISC_SIGMA_PER_VOXEL
    fused = {"xyz": mean(points), "rgb": mean(splat.f_dc.astype(np.float64) * depth_splat.SH_C0 + 0.5),
             "normal": vectors[:, :, 2], "normal_agreement": np.clip((3 * values[:, 2] - 1) / 2, 0.0, 1.0),
             "confidence": total / np.bincount(inverse, minlength=len(keys)), "slant": mean(spans)[:, 0]}
    return depth_splat.depth_fused_splat(fused, size), size


# --- Synthetic drafts: discs on a room's surfaces, about one per voxel. ---

def room_draft(count, seed=3):
    """About ``count`` discs on a 4 x 2.5 x 3 m room's floor, two walls and a table, with sub-voxel jitter,
    unnormalised rotations (one of them zero) and spread opacities, colours and sizes; and its voxel."""
    rng = np.random.default_rng(seed)
    planes = [((0, 0, 0), (4, 0, 0), (0, 0, 3)), ((0, 0, 0), (4, 0, 0), (0, 2.5, 0)),
              ((0, 0, 0), (0, 0, 3), (0, 2.5, 0)), ((1, 0.8, 1), (1.2, 0, 0), (0, 0, 0.8))]
    area = sum(np.linalg.norm(u) * np.linalg.norm(v) for _, u, v in planes)
    spacing = float(np.sqrt(area / count))
    points = []
    for origin, u, v in planes:
        u, v = np.asarray(u, float), np.asarray(v, float)
        a, b = np.meshgrid(np.arange(0, 1, spacing / np.linalg.norm(u)), np.arange(0, 1, spacing / np.linalg.norm(v)))
        points.append(np.asarray(origin, float) + a.reshape(-1, 1) * u + b.reshape(-1, 1) * v)
    xyz = np.concatenate(points)
    xyz += rng.normal(0, 0.2 * spacing, xyz.shape)
    n = len(xyz)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats[n // 2] = 0
    sigma = np.log(spacing * depth_splat.DISC_SIGMA_PER_VOXEL)
    splat = SplatData(count=n, xyz=xyz.astype(np.float32), opacity=rng.normal(0.5, 3, n).astype(np.float32),
                      f_dc=rng.normal(0, 1.5, (n, 3)).astype(np.float32),
                      scales=(sigma + rng.normal(0, 0.4, (n, 3))).astype(np.float32), quats=quats,
                      properties=depth_splat.PLY_PROPERTIES)
    return splat, spacing


def assert_same_splat(left, right):
    assert left.count == right.count and left.properties == right.properties
    for field in dataclasses.fields(SplatData):
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), field.name


def traced_peak(call):
    """``call()``'s result and its traced peak above what was traced before it."""
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        result = call()
        return result, tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()


# --- Bit identity ---

@pytest.mark.parametrize("chunk_rows", [1, 7, 1 << 20])
@pytest.mark.parametrize("share", [0.9, 0.3, 0.05])
def test_the_mesh_input_is_merged_a_run_at_a_time_to_the_same_bits(monkeypatch, chunk_rows, share):
    splat, voxel = room_draft(1_500 if chunk_rows < 10 else 60_000)
    limit = int(share * splat.count)
    expected, expected_size = legacy_splats2mesh_input(splat, voxel, limit=limit)
    monkeypatch.setattr(depth_splat, "MESH_INPUT_CHUNK_ROWS", chunk_rows, raising=False)
    merged, size = depth_splat.splats2mesh_input(splat, voxel, limit=limit)
    assert size == expected_size and merged.count <= limit < splat.count
    assert_same_splat(merged, expected)
    assert depth_splat.encode_spz(merged) == depth_splat.encode_spz(expected)


def test_a_draft_within_the_limit_or_beyond_every_grid_is_answered_as_before(monkeypatch):
    splat, voxel = room_draft(2_000)
    monkeypatch.setattr(depth_splat, "MESH_INPUT_CHUNK_ROWS", 300, raising=False)
    same, size = depth_splat.splats2mesh_input(splat, voxel, limit=splat.count)
    assert same is splat and size == voxel
    with pytest.raises(ValueError, match="atlas_depth_splat_mesh_input_exceeds_limit"):
        legacy_splats2mesh_input(splat, voxel, limit=3)
    with pytest.raises(ValueError, match="atlas_depth_splat_mesh_input_exceeds_limit"):
        depth_splat.splats2mesh_input(splat, voxel, limit=3)
    far = dataclasses.replace(splat, xyz=splat.xyz.copy())
    far.xyz[-1] = 1e9
    with pytest.raises(ValueError, match="atlas_depth_splat_scene_exceeds_voxel_grid"):
        legacy_splats2mesh_input(far, voxel, limit=10)
    with pytest.raises(ValueError, match="atlas_depth_splat_scene_exceeds_voxel_grid"):
        depth_splat.splats2mesh_input(far, voxel, limit=10)


# --- Memory ---

def test_merging_the_mesh_input_holds_a_few_columns_of_the_draft_not_its_widened_copies(monkeypatch):
    """Beyond the draft itself: about seven float64 columns of it (the voxel of each Gaussian, its weight,
    its normal, and the column being summed), never its widened positions or six stacked normal products.

    The merged input is about an eighth of the draft, as 2 M is of a 16 M-Gaussian room. The frozen
    implementation traced about 160 bytes a Gaussian here; this one about 60.
    """
    splat, voxel = room_draft(300_000)
    monkeypatch.setattr(depth_splat, "MESH_INPUT_CHUNK_ROWS", 1 << 14, raising=False)
    (merged, _), peak = traced_peak(lambda: depth_splat.splats2mesh_input(splat, voxel, limit=splat.count // 8))
    assert merged.count <= splat.count // 8
    assert peak < 90 * splat.count, (peak / splat.count, peak / 2**20)
