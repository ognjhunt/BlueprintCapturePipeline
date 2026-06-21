from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import site_memory_utils as smu


def _pose(tx: float = 0.0, tz: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = tx
    pose[2, 3] = tz
    return pose


def test_jsonl_stats_and_pose_helpers(tmp_path: Path) -> None:
    assert smu.load_jsonl(tmp_path / "missing.jsonl") == []
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('\nnot-json\n["list"]\n{"a":1}\n', encoding="utf-8")
    assert smu.load_jsonl(jsonl) == [{"a": 1}]

    out = tmp_path / "nested" / "out.jsonl"
    smu.write_jsonl(out, [{"b": 2}])
    assert out.read_text(encoding="utf-8").strip() == '{"b":2}'

    assert smu.p95([]) == 0.0
    assert smu.p95([1.0, 2.0, 3.0]) > 2.0
    assert smu.clamp01("bad", default=0.25) == 0.25
    assert smu.clamp01(2.0) == 1.0
    assert smu.pose_matrix("bad") is None
    assert smu.pose_matrix([1, 2, 3]) is None
    assert smu.pose_matrix(np.eye(4)).shape == (4, 4)

    inverse = smu.mat_inv(_pose(tx=2.0))
    assert np.allclose(inverse @ _pose(tx=2.0), np.eye(4))
    assert np.allclose(smu.transform_translation("bad"), np.zeros(3))
    assert np.allclose(smu.transform_translation(_pose(tx=1.0)), [1.0, 0.0, 0.0])
    assert smu.effective_pose({"T_world_camera": "bad"}) is None
    assert np.allclose(smu.effective_pose({"T_site_camera": _pose(tx=3.0)})[:3, 3], [3.0, 0.0, 0.0])
    assert np.allclose(
        smu.effective_pose({"T_world_camera": _pose(tx=1.0), "site_frame_transform": _pose(tx=2.0)})[:3, 3],
        [3.0, 0.0, 0.0],
    )
    assert smu.pose_distance_m(_pose(tx=1.0), _pose(tx=4.0)) == 3.0
    assert smu.rotation_cosine(np.eye(4), np.eye(4)) == 1.0


def test_uri_embedding_and_numeric_array_loaders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage_root = tmp_path / "storage"
    assert smu.gs_uri_to_local("", storage_root=storage_root) is None
    assert smu.gs_uri_to_local("/tmp/local.bin", storage_root=None) == Path("/tmp/local.bin")
    assert smu.gs_uri_to_local("gs://bucket/key.bin", storage_root=None) is None
    assert smu.gs_uri_to_local("gs://bucket", storage_root=storage_root) is None
    assert smu.gs_uri_to_local("gs://bucket/key.bin", storage_root=storage_root) == storage_root / "bucket" / "key.bin"

    assert smu.load_embedding(embedding_uri="gs://bucket/missing.bin", storage_root=storage_root, expected_dim=3) is None
    emb_path = storage_root / "bucket" / "emb.bin"
    emb_path.parent.mkdir(parents=True)
    np.asarray([1.0, 2.0], dtype=np.float32).tofile(emb_path)
    assert smu.load_embedding(embedding_uri="gs://bucket/emb.bin", storage_root=storage_root, expected_dim=3) is None
    np.zeros(3, dtype=np.float32).tofile(emb_path)
    assert smu.load_embedding(embedding_uri="gs://bucket/emb.bin", storage_root=storage_root, expected_dim=3) is None
    np.asarray([3.0, 4.0, 0.0], dtype=np.float32).tofile(emb_path)
    assert np.allclose(
        smu.load_embedding(embedding_uri="gs://bucket/emb.bin", storage_root=storage_root, expected_dim=3),
        [0.6, 0.8, 0.0],
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(smu.np, "fromfile", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("nope")))
        assert smu.load_embedding(embedding_uri="gs://bucket/emb.bin", storage_root=storage_root, expected_dim=3) is None

    assert smu.load_numeric_array("", storage_root=storage_root) is None
    assert smu.load_numeric_array("gs://bucket/missing.npy", storage_root=storage_root) is None
    npy_path = storage_root / "bucket" / "depth.npy"
    np.save(npy_path, np.asarray([[1.0, 2.0]], dtype=np.float32))
    assert smu.load_numeric_array("gs://bucket/depth.npy", storage_root=storage_root).shape == (1, 2)
    with monkeypatch.context() as scoped:
        scoped.setattr(smu.np, "load", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad npy")))
        assert smu.load_numeric_array("gs://bucket/depth.npy", storage_root=storage_root) is None


def test_png_loader_and_depth_fingerprints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    Image = pytest.importorskip("PIL.Image")
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(rgb_path)
    assert smu._load_png_array(rgb_path).shape == (2, 2)
    assert smu._load_png_array(tmp_path / "not-an-image.png") is None

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "PIL":
            raise ImportError("missing pillow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert smu._load_png_array(rgb_path) is None
    monkeypatch.setattr(builtins, "__import__", real_import)

    storage_root = tmp_path / "storage"
    depth_path = storage_root / "bucket" / "depth.npy"
    conf_path = storage_root / "bucket" / "confidence.npy"
    depth_path.parent.mkdir(parents=True)
    np.save(depth_path, np.ones((4, 4), dtype=np.float32))
    np.save(conf_path, np.full((4, 4), 0.8, dtype=np.float32))

    assert smu.geometry_fingerprint(depth_path="", confidence_path="", storage_root=storage_root) == {
        "available": False,
        "representation": "none",
    }
    zero_path = storage_root / "bucket" / "zero.npy"
    np.save(zero_path, np.zeros((2, 2), dtype=np.float32))
    assert smu.geometry_fingerprint(depth_path="gs://bucket/zero.npy", confidence_path="", storage_root=storage_root)[
        "representation"
    ] == "depth_unusable"
    fingerprint = smu.geometry_fingerprint(
        depth_path="gs://bucket/depth.npy",
        confidence_path="gs://bucket/confidence.npy",
        storage_root=storage_root,
        intrinsics={"fx": 100.0, "fy": 101.0, "cx": 2.0, "cy": 3.0},
    )
    assert fingerprint["available"] is True
    assert fingerprint["confidence_mean"] == 0.8
    assert fingerprint["projective_scale"]["fx"] == 100.0

    png16_path = storage_root / "bucket" / "depth16.png"
    Image.fromarray(np.asarray([[1000, 0], [2000, 3000]], dtype=np.uint16)).save(png16_path)
    assert smu.load_numeric_array("gs://bucket/depth16.png", storage_root=storage_root)[0, 0] == 1.0
    png8_path = storage_root / "bucket" / "depth8.png"
    Image.fromarray(np.asarray([[10, 20], [30, 40]], dtype=np.uint8)).save(png8_path)
    assert smu.load_numeric_array("gs://bucket/depth8.png", storage_root=storage_root)[0, 0] == 10.0
    bad_png = tmp_path / "not-an-image.png"
    bad_png.write_text("not really a png", encoding="utf-8")
    assert smu.load_numeric_array(str(bad_png)) is None
    unknown = tmp_path / "unknown.txt"
    unknown.write_text("unknown", encoding="utf-8")
    assert smu.load_numeric_array(str(unknown)) is None


def test_similarity_visibility_backprojection_and_pointcloud(tmp_path: Path) -> None:
    assert smu.fingerprint_similarity({}, {}) == 0.0
    a = {
        "depth_histogram_8": [0.5, 0.5],
        "median_depth_m": 1.0,
        "surface_complexity": 0.2,
        "plane_support_ratio": 0.8,
    }
    b = {
        "depth_histogram_8": [0.25, 0.75],
        "median_depth_m": 2.0,
        "surface_complexity": 0.4,
        "plane_support_ratio": 0.6,
    }
    assert 0.0 < smu.fingerprint_similarity(a, b) < 1.0

    assert smu.visibility_cells_from_record({}) == []
    zero_forward = _pose()
    zero_forward[:3, 2] = 0.0
    assert smu.visibility_cells_from_record({"T_world_camera": zero_forward}) == []
    assert smu.visibility_cells_from_record({"T_world_camera": _pose(), "geometry_fingerprint": "bad"})[:2] == [
        "0,1",
        "0,2",
    ]
    cells = smu.visibility_cells_from_record(
        {"T_world_camera": _pose(), "geometry_fingerprint": {"free_space_extent_m": 1.1}},
        cell_size_m=0.5,
    )
    assert cells == ["0,1", "0,2", "0,3"]

    depth = np.asarray([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    confidence = np.asarray([[0.9, 0.9], [0.2, 0.8]], dtype=np.float32)
    assert smu.backproject_depth_points(depth=depth, intrinsics={}, T_world_camera=np.eye(4)).shape == (0, 4)
    assert smu.backproject_depth_points(
        depth=np.zeros((1, 1), dtype=np.float32),
        intrinsics={"fx": 1.0, "fy": 1.0},
        T_world_camera=np.eye(4),
    ).shape == (0, 4)
    points = smu.backproject_depth_points(
        depth=depth,
        confidence=confidence,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        T_world_camera=np.eye(4),
        sample_step=1,
        min_confidence=0.5,
        static_weight=2.0,
    )
    assert points.shape == (2, 4)
    assert points[0, 3] == pytest.approx(1.8)

    ply_path = tmp_path / "points.ply"
    smu.write_ascii_pointcloud(ply_path, points)
    assert "element vertex 2" in ply_path.read_text(encoding="utf-8")


def test_aggregate_planes_histograms_and_groups(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    emb_path = storage_root / "bucket" / "emb.bin"
    emb_path.parent.mkdir(parents=True)
    embedding = np.zeros(1024, dtype=np.float32)
    embedding[0] = 1.0
    embedding.tofile(emb_path)

    summary = smu.aggregate_chunk_summary(
        [
            {
                "embedding_uri": "gs://bucket/emb.bin",
                "zone_id": "zone-a",
                "staticness_score": 0.8,
                "anchor_observations": ["door", "door", "shelf"],
                "geometry_fingerprint": {
                    "median_depth_m": 1.0,
                    "plane_support_ratio": 0.5,
                    "surface_complexity": 0.2,
                    "depth_histogram_8": [0.2, 0.8],
                },
            },
            {"staticness_score": "bad", "anchor_observations": ["shelf", "bin"]},
        ],
        storage_root=storage_root,
    )

    assert summary["record_count"] == 2
    assert summary["zone_id"] == "zone-a"
    assert summary["anchor_ids"] == ["door", "shelf", "bin"]
    assert summary["embedding_centroid"].shape == (1024,)
    assert summary["embedding_centroid"][0] == 1.0
    assert summary["geometry_fingerprint"]["depth_histogram_8"] == [0.2, 0.8]
    assert smu.aggregate_chunk_summary([], storage_root=storage_root)["embedding_centroid"] is None
    assert smu._mean_histograms([[], []]) == []

    class TruthyEmpty:
        def __bool__(self) -> bool:
            return True

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(())

    assert smu._mean_histograms([TruthyEmpty()]) == []

    assert smu.plane_summaries(np.zeros((0, 4), dtype=np.float32)) == []
    assert smu.plane_summaries(np.zeros((4, 4), dtype=np.float32)) == []
    sparse_y = np.asarray([[0.0, float(y), 0.0, 1.0] for y in range(12)], dtype=np.float32)
    assert smu.plane_summaries(sparse_y) == []
    floor = np.asarray([[x, 0.0, z, 1.0] for x in range(6) for z in range(2)], dtype=np.float32)
    ceiling = np.asarray([[x, 3.0, z, 1.0] for x in range(6) for z in range(2)], dtype=np.float32)
    planes = smu.plane_summaries(np.concatenate([floor, ceiling], axis=0))
    assert {plane["plane_id"] for plane in planes} == {"floor_like", "ceiling_like"}

    assert smu.iter_groups([{"zone": "a"}, {"zone": ""}, {"zone": "a"}, {"zone": "b"}], "zone") == {
        "a": [{"zone": "a"}, {"zone": "a"}],
        "b": [{"zone": "b"}],
    }
