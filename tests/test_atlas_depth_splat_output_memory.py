"""No provider calls: the draft's voxel table, SPZ and PLY are built a run at a time and write the bytes they wrote.

Austin's 204 posed 720x1280 views (2026-11-24) fused into 16.2 M Gaussians and,
replayed offline, peaked at 6.95 GiB, above the Atlas preview lane's 5.5 GB limit:
the voxel table was merged as one table (4.5 GiB), its means and the splat were
built beside it (5.0 GiB), the SPZ was encoded with every field widened at once
(5.2 GiB) and then decoded whole to int64 to check it (6.95 GiB). The ``legacy_*``
functions below are those of origin/main before that day, frozen, so the rework is
checked byte for byte.
"""
import dataclasses
import gc
import gzip
import json
import struct
import tracemalloc

import numpy as np
import pytest

from blueprint_pipeline import atlas_depth_splat as depth_splat
from blueprint_pipeline import gaussian_splat_decode
from blueprint_pipeline.gaussian_splat_decode import SplatData, float_ply_header
from tests.atlas_posed_rgbd_synthetic import write_room_pose_root

# --- origin/main before 2026-11-24, frozen as the reference. ---

class LegacyVoxelSums:
    MIN_MERGE_ROWS = 1_000_000
    MERGE_CHUNK_ROWS = 65_536

    def __init__(self, voxel):
        self.voxel = voxel
        self._keys = np.empty(0, np.int64)
        self._columns = np.empty((12, 0))
        self._pending = []
        self._pending_rows = 0

    def add(self, sample):
        weights = sample["weights"]
        slant = sample.get("slant", np.zeros_like(weights))
        columns = np.concatenate([weights[None], (sample["points"] * weights[:, None]).T,
                                  (sample["colors"] * weights[:, None]).T, (sample["normals"] * weights[:, None]).T,
                                  np.ones_like(weights)[None], (slant * weights)[None]])
        self._pending.append(depth_splat._sum_by_key(depth_splat._voxel_keys(sample["points"], self.voxel), columns))
        self._pending_rows += len(self._pending[-1][0])
        if self._pending_rows >= max(self.MIN_MERGE_ROWS, len(self._keys) // 2):
            self._merge()

    def _merge(self):
        if self._pending:
            tables = [(self._keys, self._columns), *self._pending]
            keys = np.unique(np.concatenate([part_keys for part_keys, _ in tables]))
            columns = np.zeros((12, len(keys)))
            for part_keys, part_columns in tables:
                for start in range(0, len(part_keys), self.MERGE_CHUNK_ROWS):
                    end = start + self.MERGE_CHUNK_ROWS
                    slots = np.searchsorted(keys, part_keys[start:end])
                    for row in range(12):
                        columns[row, slots] += part_columns[row, start:end]
            self._keys, self._columns = keys, columns
            self._pending, self._pending_rows = [], 0

    @property
    def keys(self):
        self._merge()
        return self._keys

    @property
    def sums(self):
        self._merge()
        return self._columns.T

    def fused(self):
        total = self.sums[:, :1]
        normal = self.sums[:, 7:10] / total
        agreement = np.linalg.norm(normal, axis=1)
        return {"xyz": self.sums[:, 1:4] / total, "rgb": self.sums[:, 4:7] / total,
                "normal": normal / np.maximum(agreement, 1e-12)[:, None], "normal_agreement": agreement,
                "confidence": total[:, 0] / self.sums[:, 10], "observations": self.sums[:, 10],
                "slant": self.sums[:, 11] / total[:, 0]}

    def fused_splat(self):
        """What ``fuse_posed_frames`` did with the table: ``fused()``, the median, then ``depth_fused_splat``."""
        fused = self.fused()
        observations = float(np.median(fused.pop("observations")))
        return depth_splat.depth_fused_splat(fused, self.voxel), observations


def legacy_encode_spz(splat):
    degree = depth_splat._sh_degree(splat.sh_rest, splat.count)
    fixed = np.round(splat.xyz.astype(np.float64) * (1 << depth_splat.SPZ_FRACTIONAL_BITS)).astype(np.int64)
    if np.any(np.abs(fixed) >= 1 << 23):
        raise ValueError("atlas_depth_splat_spz_position_out_of_range")
    positions = ((fixed[..., None] & 0xFFFFFF) >> np.array([0, 8, 16])) & 0xFF
    quats = splat.quats.astype(np.float64)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    quats *= np.where(quats[:, :1] < 0, -1.0, 1.0)
    uint8 = depth_splat._uint8
    parts = [
        struct.pack("<IIIBBBB", depth_splat.SPZ_MAGIC, depth_splat.SPZ_VERSION, splat.count, degree,
                    depth_splat.SPZ_FRACTIONAL_BITS, 0, 0),
        positions.astype(np.uint8).tobytes(),
        uint8(255.0 / (1.0 + np.exp(-splat.opacity.astype(np.float64)))).tobytes(),
        uint8(splat.f_dc.astype(np.float64) * (depth_splat.SPZ_COLOR_SCALE * 255.0) + 127.5).tobytes(),
        uint8((splat.scales.astype(np.float64) + 10.0) * 16.0).tobytes(),
        uint8(quats[:, 1:] * 127.5 + 127.5).tobytes(),
    ]
    if degree:
        dim = (degree + 1) ** 2 - 1
        rest = np.asarray(splat.sh_rest, np.float64).reshape(splat.count, 3, dim).transpose(0, 2, 1)
        parts.append(uint8(rest * 128.0 + 128.0).tobytes())
    return gzip.compress(b"".join(parts), compresslevel=9, mtime=0)


def legacy_decode_spz(data):
    raw = gzip.decompress(data)
    magic, version, count, degree, fractional, _flags, _reserved = struct.unpack_from("<IIIBBBB", raw)
    dim = (degree + 1) ** 2 - 1 if degree <= 3 else -1
    if (magic != depth_splat.SPZ_MAGIC or version != depth_splat.SPZ_VERSION or dim < 0
            or len(raw) != 16 + count * (19 + 3 * dim)):
        raise ValueError("atlas_depth_splat_spz_invalid")
    body = np.frombuffer(raw, dtype=np.uint8, offset=16).astype(np.int64)
    position, rest = body[:count * 9].reshape(count, 3, 3), body[count * 9:count * 19]
    fixed = position[..., 0] | (position[..., 1] << 8) | (position[..., 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    alpha = np.clip(rest[:count] / 255.0, 1e-6, 1 - 1e-6)
    colors, scales, rotation = (rest[count + 3 * count * index:count + 3 * count * (index + 1)].reshape(count, 3)
                                for index in range(3))
    xyz = rotation / 127.5 - 1.0
    w = np.sqrt(np.clip(1.0 - np.sum(xyz * xyz, axis=1), 0.0, None))
    sh_rest, properties = None, depth_splat.PLY_PROPERTIES
    if dim:
        coefficients = body[count * 19:].reshape(count, dim, 3).transpose(0, 2, 1).reshape(count, 3 * dim)
        sh_rest = ((coefficients - 128.0) / 128.0).astype(np.float32)
        properties = (*depth_splat.PLY_PROPERTIES, *(f"f_rest_{index}" for index in range(3 * dim)))
    return SplatData(count=count, xyz=(fixed / float(1 << fractional)).astype(np.float32),
                     opacity=np.log(alpha / (1 - alpha)).astype(np.float32),
                     f_dc=((colors / 255.0 - 0.5) / depth_splat.SPZ_COLOR_SCALE).astype(np.float32),
                     scales=(scales / 16.0 - 10.0).astype(np.float32),
                     quats=np.column_stack([w, xyz]).astype(np.float32), properties=properties, sh_rest=sh_rest)


def legacy_round_trip_failed(data, splat):
    """``write_depth_splat``'s check: decode the whole SPZ and compare its count and positions."""
    decoded = legacy_decode_spz(data)
    return bool(decoded.count != splat.count
                or np.max(np.abs(decoded.xyz - splat.xyz)) > 1.0 / (1 << depth_splat.SPZ_FRACTIONAL_BITS))


def legacy_write_float_ply(path, count, order):
    table = np.empty((count, len(order)), dtype="<f4")
    for col, (_, values) in enumerate(order):
        table[:, col] = np.asarray(values, dtype="<f4")
    with open(path, "wb") as handle:
        handle.write(float_ply_header(count, [name for name, _ in order]))
        handle.write(table.tobytes(order="C"))
    return path


# --- Helpers ---

SPLAT_FIELDS = ("xyz", "opacity", "f_dc", "scales", "quats")


def assert_same_splat(splat, reference):
    assert splat.count == reference.count and splat.properties == reference.properties
    for field in (*SPLAT_FIELDS, "sh_rest"):
        value, expected = getattr(splat, field), getattr(reference, field)
        assert (value is None) == (expected is None), field
        if value is not None:
            assert value.dtype == expected.dtype and value.shape == expected.shape, field
            assert value.tobytes() == expected.tobytes(), field


def room_samples(frames=24, points=4_000, seed=5):
    """Overlapping views' surface samples of one room: most voxels are seen by several frames."""
    rng = np.random.default_rng(seed)
    samples = []
    for index in range(frames):
        centre = np.array([0.4 * np.cos(index / 3), 0.1 * index / frames, 0.4 * np.sin(index / 3)])
        normals = rng.normal(size=(points, 3))
        samples.append({"points": centre + rng.uniform(-0.6, 0.6, (points, 3)),
                        "colors": rng.uniform(0, 1, (points, 3)),
                        "normals": normals / np.linalg.norm(normals, axis=1, keepdims=True),
                        "weights": rng.uniform(0.05, 1.0, points), "slant": rng.uniform(0.001, 0.08, points)})
    return samples


def random_splat(count, degree=0, seed=9):
    rng = np.random.default_rng(seed)
    quats = rng.normal(size=(count, 4)).astype(np.float32)
    rest = None
    properties = depth_splat.PLY_PROPERTIES
    if degree:
        dim = (degree + 1) ** 2 - 1
        rest = rng.uniform(-1.2, 1.2, (count, 3 * dim)).astype(np.float32)
        properties = (*properties, *(f"f_rest_{index}" for index in range(3 * dim)))
    return SplatData(count=count, xyz=rng.uniform(-1500, 1500, (count, 3)).astype(np.float32),
                     opacity=rng.normal(0, 4, count).astype(np.float32),
                     f_dc=rng.normal(0, 2, (count, 3)).astype(np.float32),
                     scales=rng.uniform(-11, 6, (count, 3)).astype(np.float32), quats=quats,
                     properties=properties, sh_rest=rest)


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


# --- Byte identity ---

@pytest.mark.parametrize("leaf_rows,min_merge_rows,merge_fraction", [
    (2, 1, 1), (7, 3, 2), (64, 16, 8), (1_000, 50, 4), (None, None, None)])
def test_leaves_add_every_voxel_sum_and_build_every_gaussian_bit_for_bit(
        monkeypatch, leaf_rows, min_merge_rows, merge_fraction):
    """One-row leaves, a leaf per few voxels, merges after every frame or none until the end: all the same bits."""
    if leaf_rows is not None:
        monkeypatch.setattr(depth_splat.VoxelSums, "LEAF_ROWS", leaf_rows)
        monkeypatch.setattr(depth_splat.VoxelSums, "MIN_MERGE_ROWS", min_merge_rows)
        monkeypatch.setattr(depth_splat.VoxelSums, "MERGE_FRACTION", merge_fraction)
    samples = room_samples()
    for legacy_merge_rows in (1, 10**9):
        reference = LegacyVoxelSums(0.04)
        reference.MIN_MERGE_ROWS = legacy_merge_rows
        sums = depth_splat.VoxelSums(0.04)
        for sample in samples:
            reference.add(sample)
            sums.add(sample)
        if leaf_rows is not None and leaf_rows < 100:
            assert len(sums._leaves) > 10, "the table is held in many leaves"
        assert sums.keys.tobytes() == reference.keys.tobytes()
        assert np.ascontiguousarray(sums.sums).tobytes() == np.ascontiguousarray(reference.sums).tobytes()
        splat, observations = sums.fused_splat()
        expected, expected_observations = reference.fused_splat()
        assert_same_splat(splat, expected)
        assert observations == expected_observations
        assert len(sums.keys) == 0, "the table was let go as the splat was built"


@pytest.mark.parametrize("chunk_rows", [1, 7, 1 << 20])
@pytest.mark.parametrize("degree", [0, 1, 3])
def test_spz_is_encoded_decoded_and_checked_a_run_at_a_time_byte_for_byte(monkeypatch, chunk_rows, degree):
    monkeypatch.setattr(depth_splat, "SPZ_CHUNK_ROWS", chunk_rows)
    splat = random_splat(1_500 if chunk_rows < 10 else 20_000, degree)
    data = depth_splat.encode_spz(splat)
    assert data == legacy_encode_spz(splat)
    assert_same_splat(depth_splat.decode_spz(data), legacy_decode_spz(data))
    assert depth_splat._spz_round_trip_failed(data, splat) is legacy_round_trip_failed(data, splat) is False
    # A position moved by more than one fixed-point step, or a count changed, fails both checks alike.
    moved = dataclasses.replace(splat, xyz=splat.xyz.copy())
    moved.xyz[-1, 2] += 2.0 / (1 << depth_splat.SPZ_FRACTIONAL_BITS)
    assert depth_splat._spz_round_trip_failed(data, moved) is legacy_round_trip_failed(data, moved) is True
    fewer = random_splat(splat.count - 1, degree)
    assert depth_splat._spz_round_trip_failed(data, fewer) is legacy_round_trip_failed(data, fewer) is True
    with pytest.raises(ValueError, match="atlas_depth_splat_spz_position_out_of_range"):
        depth_splat.encode_spz(dataclasses.replace(splat, xyz=splat.xyz * 2000))


def test_the_streamed_check_refuses_every_file_the_whole_decode_refused():
    splat = random_splat(3_000, degree=1)
    data = depth_splat.encode_spz(splat)
    raw = gzip.decompress(data)
    miscounted = gzip.compress(raw[:8] + struct.pack("<I", splat.count + 1) + raw[12:], compresslevel=9, mtime=0)
    unversioned = gzip.compress(raw[:4] + struct.pack("<I", 3) + raw[8:], compresslevel=9, mtime=0)
    for invalid in (miscounted, unversioned):
        for check in (depth_splat._spz_round_trip_failed, legacy_round_trip_failed):
            with pytest.raises(ValueError, match="atlas_depth_splat_spz_invalid"):
                check(invalid, splat)
    # A file cut short, or with more after its gzip member, is refused, as decoding it whole refused it.
    for invalid in (data[:-9], data[:len(data) // 2], data + b"more"):
        with pytest.raises(ValueError, match="atlas_depth_splat_spz_invalid"):
            depth_splat._spz_round_trip_failed(invalid, splat)
        with pytest.raises((EOFError, gzip.BadGzipFile)):
            legacy_round_trip_failed(invalid, splat)


@pytest.mark.parametrize("chunk_rows", [1, 5, 1 << 20])
def test_float_ply_is_written_a_run_of_rows_at_a_time_byte_for_byte(tmp_path, monkeypatch, chunk_rows):
    monkeypatch.setattr(gaussian_splat_decode, "FLOAT_PLY_CHUNK_ROWS", chunk_rows)
    for degree in (0, 2):
        splat = random_splat(123, degree)
        written = gaussian_splat_decode.write_standard_3dgs_ply(splat, tmp_path / f"new-{degree}.ply")
        with monkeypatch.context() as legacy:
            legacy.setattr(gaussian_splat_decode, "_write_float_ply", legacy_write_float_ply)
            expected = gaussian_splat_decode.write_standard_3dgs_ply(splat, tmp_path / f"old-{degree}.ply")
        assert written.read_bytes() == expected.read_bytes()


def test_the_draft_writer_writes_the_files_and_manifest_it_wrote_before(tmp_path, monkeypatch):
    """A retained pose's draft, fused with many leaves and written in short runs, is the legacy writer's draft."""
    pose_root = write_room_pose_root(tmp_path / "pose", 6, width=160, height=96)
    result, frames = depth_splat.load_posed_rgbd(pose_root)
    with monkeypatch.context() as legacy:
        legacy.setattr(depth_splat, "VoxelSums", LegacyVoxelSums)
        legacy.setattr(depth_splat, "encode_spz", legacy_encode_spz)
        legacy.setattr(depth_splat, "_spz_round_trip_failed", legacy_round_trip_failed)
        legacy.setattr(gaussian_splat_decode, "_write_float_ply", legacy_write_float_ply)
        expected = depth_splat.write_depth_splat(frames=frames, result=result, output_root=tmp_path / "legacy")
    monkeypatch.setattr(depth_splat.VoxelSums, "LEAF_ROWS", 64)
    monkeypatch.setattr(depth_splat.VoxelSums, "MIN_MERGE_ROWS", 8)
    monkeypatch.setattr(depth_splat, "SPZ_CHUNK_ROWS", 100)
    monkeypatch.setattr(gaussian_splat_decode, "FLOAT_PLY_CHUNK_ROWS", 100)
    manifest = depth_splat.write_depth_splat(frames=frames, result=result, output_root=tmp_path / "new")
    assert manifest["gaussian_count"] > 20 * 64, "the draft spans many leaves and runs"
    assert json.dumps(manifest) == json.dumps(expected)
    for name in ("atlas_depth_splat.ply", "atlas_depth_splat.spz", "atlas_depth_splat_manifest.json"):
        assert (tmp_path / "new" / name).read_bytes() == (tmp_path / "legacy" / name).read_bytes(), name
    spec, legacy_spec = (json.loads((tmp_path / root / "splats2mesh_task_spec.json").read_text())
                         for root in ("new", "legacy"))
    assert spec["uploads"]["splat"]["sha256"] == legacy_spec["uploads"]["splat"]["sha256"]
    assert spec["request"] == legacy_spec["request"]


# --- Memory follows one copy of the table and of the splat ---

def test_filling_the_voxel_table_and_building_its_splat_hold_one_table(monkeypatch):
    """The fill peaks near one table and the splat near one table and the splat it becomes.

    Merging rebuilt the whole table beside itself and its pending frames, and the
    splat was built beside the table and a full set of its means. Leaves of 16 k
    voxels stand in here for the 512 k of a production table about 30 times that.
    """
    monkeypatch.setattr(depth_splat.VoxelSums, "LEAF_ROWS", 1 << 14, raising=False)
    monkeypatch.setattr(depth_splat.VoxelSums, "MIN_MERGE_ROWS", 1 << 9)
    samples = room_samples(frames=48, points=20_000, seed=11)
    voxel = 0.012
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        sums = depth_splat.VoxelSums(voxel)
        for sample in samples:
            sums.add(sample)
        count = len(sums.keys)
        fill = tracemalloc.get_traced_memory()[1] - before
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        splat = (sums.fused_splat()[0] if hasattr(sums, "fused_splat")
                 else depth_splat.depth_fused_splat(sums.fused(), voxel))
        build = tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()
    table = count * (12 + 1) * 8
    splat_bytes = sum(getattr(splat, field).nbytes for field in SPLAT_FIELDS)
    assert count > 500_000 and splat.count == count
    # One table, what waits to merge into its leaves and one frame's reduction.
    assert fill < 1.4 * table, (fill / 2**20, table / 2**20)
    # The splat and its observation counts, allocated whole and filled as the leaves go, and one leaf's means.
    assert build < splat_bytes + count * 8 + 0.2 * table, (build / 2**20, table / 2**20, splat_bytes / 2**20)


def test_encoding_and_checking_the_spz_hold_a_run_of_it_not_a_widened_copy(monkeypatch):
    """Encoding held the positions widened to int64 twice over and every field's bytes twice; the check decoded
    the whole body to int64, eight times the file's raw size, beside the splat it was compared with."""
    monkeypatch.setattr(depth_splat, "SPZ_CHUNK_ROWS", 1 << 14, raising=False)
    splat = random_splat(600_000)
    fields = sum(getattr(splat, field).nbytes for field in SPLAT_FIELDS)
    raw = 16 + splat.count * 19
    data, encode_peak = traced_peak(lambda: depth_splat.encode_spz(splat))
    # The raw body, the compressed file twice (zlib joins its output blocks) and one run of widened fields.
    assert encode_peak < raw + 2 * len(data) + 0.15 * fields, (encode_peak / 2**20, raw / 2**20, len(data) / 2**20)
    decoded, decode_peak = traced_peak(lambda: depth_splat.decode_spz(data))
    assert_same_splat(decoded, legacy_decode_spz(data))
    # The decoded fields, the raw body and one run of it widened.
    assert decode_peak < fields + raw + 0.15 * fields, (decode_peak / 2**20, fields / 2**20)
    if hasattr(depth_splat, "_spz_round_trip_failed"):
        failed, check_peak = traced_peak(lambda: depth_splat._spz_round_trip_failed(data, splat))
        # Inflated a run at a time: not even the raw body is held.
        assert failed is False and check_peak < 0.15 * fields, (check_peak / 2**20, fields / 2**20)


def test_writing_the_draft_holds_one_splat_not_its_interleaved_or_widened_copies(tmp_path, monkeypatch):
    """From the fused splat to the files, ``write_depth_splat`` holds the splat and its SPZ, plus a run of rows."""
    monkeypatch.setattr(depth_splat, "SPZ_CHUNK_ROWS", 1 << 15, raising=False)
    monkeypatch.setattr(gaussian_splat_decode, "FLOAT_PLY_CHUNK_ROWS", 1 << 15, raising=False)
    pose_root = write_room_pose_root(tmp_path / "pose", 2, width=64, height=48)
    result, frames = depth_splat.load_posed_rgbd(pose_root)
    splat = random_splat(600_000)
    splat = dataclasses.replace(splat, xyz=splat.xyz / 1000)
    fields = sum(getattr(splat, field).nbytes for field in SPLAT_FIELDS)
    monkeypatch.setattr(depth_splat, "fuse_posed_frames", lambda frames, **_: (splat, {"voxel_size": 0.01}))
    manifest, peak = traced_peak(lambda: depth_splat.write_depth_splat(
        frames=frames, result=result, output_root=tmp_path / "draft"))
    assert manifest["gaussian_count"] == splat.count
    spz = (tmp_path / "draft" / "atlas_depth_splat.spz").stat().st_size
    # Encoding's raw body and compressed bytes (twice, as zlib joins its output), and a run of rows widened or
    # interleaved: the check inflates and the PLY interleaves a run at a time.
    assert peak < 16 + splat.count * 19 + 2 * spz + 0.2 * fields, (peak / 2**20, fields / 2**20, spz / 2**20)
