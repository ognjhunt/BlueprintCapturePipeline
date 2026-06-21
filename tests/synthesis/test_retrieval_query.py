"""Tests for retrieval_query.py — K-NN retrieval from site_reference_index.jsonl."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.synthesis import retrieval_query
from blueprint_pipeline.synthesis.retrieval_query import query_site


def _make_record(i: int, tx: float, embedding: np.ndarray | None = None) -> dict:
    T = np.eye(4).tolist()
    T[0][3] = tx
    rec = {
        "reference_id": f"ref-{i:04d}",
        "frame_id": f"{i:06d}",
        "T_world_camera": T,
        "intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0},
        "depth_uri": None,
        "frame_uri": None,
        "embedding_uri": None,
        "site_frame_transform": None,
    }
    if embedding is not None:
        rec["_embedding_inline"] = embedding.tolist()   # for test injection
    return rec


def _write_index(path: Path, records: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestSpatialQuery:
    def test_returns_k_nearest_by_distance(self, tmp_path):
        """10 records spaced 1 m apart; query at 3.1 m → top-3 are indices 2,3,4."""
        records = [_make_record(i, float(i)) for i in range(10)]
        index_path = tmp_path / "site_reference_index.jsonl"
        _write_index(index_path, records)

        target_T = np.eye(4)
        target_T[0, 3] = 3.1

        results = query_site(
            site_index_path=index_path,
            target_T_world_camera=target_T,
            k=3,
            mode="spatial",
        )
        assert len(results) == 3
        translations = [r["T_world_camera"][0][3] for r in results]
        # Nearest to 3.1 are 3, 4, 2 (distances 0.1, 0.9, 1.1)
        assert translations[0] == pytest.approx(3.0)
        assert translations[1] == pytest.approx(4.0)
        assert translations[2] == pytest.approx(2.0)

    def test_max_distance_filter(self, tmp_path):
        """max_distance_m prunes records beyond the radius."""
        records = [_make_record(i, float(i)) for i in range(10)]
        index_path = tmp_path / "site_reference_index.jsonl"
        _write_index(index_path, records)

        target_T = np.eye(4)
        target_T[0, 3] = 0.0

        results = query_site(
            site_index_path=index_path,
            target_T_world_camera=target_T,
            k=10,
            mode="spatial",
            max_distance_m=2.5,
        )
        # Only records at tx=0,1,2 are within 2.5 m
        assert len(results) == 3

    def test_empty_index_returns_empty(self, tmp_path):
        index_path = tmp_path / "site_reference_index.jsonl"
        index_path.write_text("")
        results = query_site(
            site_index_path=index_path,
            target_T_world_camera=np.eye(4),
            k=3,
            mode="spatial",
        )
        assert results == []


class TestEmbeddingQuery:
    def _write_index_with_embeddings(self, tmp_path: Path, embed_dir: Path, k: int = 5):
        """Create k records each with an orthonormal embedding (standard basis vectors)."""
        records = []
        for i in range(k):
            vec = np.zeros(1024, dtype=np.float32)
            vec[i] = 1.0  # orthonormal standard basis vector e_i
            emb_path = embed_dir / f"{i:06d}.bin"
            emb_path.write_bytes(vec.tobytes())

            rec = _make_record(i, float(i))
            rec["embedding_uri"] = str(emb_path)
            records.append(rec)

        index_path = tmp_path / "site_reference_index.jsonl"
        _write_index(index_path, records)
        return index_path

    def test_returns_cosine_nearest(self, tmp_path):
        """Query with e_0 → record 0 should be top-1."""
        embed_dir = tmp_path / "embeddings"
        embed_dir.mkdir()
        index_path = self._write_index_with_embeddings(tmp_path, embed_dir, k=5)

        query_emb = np.zeros(1024, dtype=np.float32)
        query_emb[0] = 1.0  # exact match to record 0

        results = query_site(
            site_index_path=index_path,
            target_T_world_camera=np.eye(4),
            k=1,
            mode="embedding",
            query_embedding=query_emb,
            storage_root=None,
        )
        assert len(results) == 1
        # The top result should be the record with e_0 embedding (at tx=0)
        assert results[0]["T_world_camera"][0][3] == pytest.approx(0.0)

    def test_embedding_mode_requires_query_embedding(self, tmp_path):
        index_path = tmp_path / "idx.jsonl"
        index_path.write_text(json.dumps(_make_record(0, 0.0)) + "\n")
        with pytest.raises(ValueError, match="query_embedding"):
            query_site(
                site_index_path=index_path,
                target_T_world_camera=np.eye(4),
                k=1,
                mode="embedding",
            )


class TestHybridQuery:
    def test_hybrid_returns_k_results(self, tmp_path):
        """Hybrid mode should return up to k results without crashing."""
        embed_dir = tmp_path / "embeddings"
        embed_dir.mkdir()
        records = []
        for i in range(6):
            vec = np.zeros(1024, dtype=np.float32)
            vec[i] = 1.0
            emb_path = embed_dir / f"{i:06d}.bin"
            emb_path.write_bytes(vec.tobytes())
            rec = _make_record(i, float(i))
            rec["embedding_uri"] = str(emb_path)
            records.append(rec)

        index_path = tmp_path / "site_reference_index.jsonl"
        _write_index(index_path, records)

        query_emb = np.zeros(1024, dtype=np.float32)
        query_emb[0] = 1.0

        target_T = np.eye(4)
        target_T[0, 3] = 0.2  # spatially near record 0

        results = query_site(
            site_index_path=index_path,
            target_T_world_camera=target_T,
            k=3,
            mode="hybrid",
            query_embedding=query_emb,
            storage_root=None,
        )
        assert 1 <= len(results) <= 3
        # Record 0 is both spatially nearest and embedding-nearest; should rank first
        assert results[0]["T_world_camera"][0][3] == pytest.approx(0.0)


def test_retrieval_query_edge_paths(tmp_path, monkeypatch):
    index_path = tmp_path / "idx.jsonl"
    base_record = _make_record(0, 0.0)
    _write_index(index_path, [base_record])

    with pytest.raises(ValueError, match="query_embedding"):
        query_site(
            site_index_path=index_path,
            target_T_world_camera=np.eye(4),
            mode="hybrid",
        )

    with pytest.raises(ValueError, match="Unknown query mode"):
        query_site(
            site_index_path=index_path,
            target_T_world_camera=np.eye(4),
            mode="nearest",
        )

    no_pose = {"reference_id": "no-pose"}
    wrong_pose = {"reference_id": "wrong-pose", "T_world_camera": [1.0, 2.0, 3.0]}
    transformed = _make_record(1, 2.0)
    site_frame_transform = np.eye(4)
    site_frame_transform[0, 3] = 10.0
    transformed["site_frame_transform"] = site_frame_transform.tolist()
    direct_pose = {"T_site_camera": np.eye(4).tolist()}

    spatial = retrieval_query._query_spatial(
        records=[no_pose, wrong_pose, transformed],
        t_target=np.array([12.0, 0.0, 0.0]),
        k=2,
        max_distance_m=None,
    )
    assert spatial == [transformed]

    assert retrieval_query._effective_pose(no_pose) is None
    assert retrieval_query._effective_pose(wrong_pose) is None
    assert retrieval_query._effective_pose(direct_pose)[3, 3] == pytest.approx(1.0)
    assert retrieval_query._effective_pose(transformed)[0, 3] == pytest.approx(12.0)

    hybrid_empty = retrieval_query._query_hybrid(
        records=[no_pose, transformed],
        t_target=np.zeros(3),
        query_embedding=np.ones(128, dtype=np.float32),
        storage_root=None,
        bucket=None,
        k=3,
        max_distance_m=0.1,
    )
    assert hybrid_empty == []

    valid_vec = np.zeros(128, dtype=np.float32)
    valid_vec[0] = 1.0
    valid_embedding_path = tmp_path / "valid.bin"
    valid_vec.tofile(valid_embedding_path)
    embedded = _make_record(2, 3.0)
    embedded["embedding_uri"] = str(valid_embedding_path)
    embedding_results = retrieval_query._query_embedding(
        records=[base_record, embedded],
        query_embedding=valid_vec,
        storage_root=None,
        bucket=None,
        k=2,
    )
    assert embedding_results == [embedded]

    assert retrieval_query._load_embedding({}, storage_root=None, bucket=None) is None
    assert retrieval_query._load_embedding(
        {"embedding_uri": str(tmp_path / "missing.bin")},
        storage_root=None,
        bucket=None,
    ) is None

    short_path = tmp_path / "short.bin"
    np.ones(3, dtype=np.float32).tofile(short_path)
    assert retrieval_query._load_embedding(
        {"embedding_uri": str(short_path)},
        storage_root=None,
        bucket=None,
    ) is None

    def _raise_fromfile(*_args, **_kwargs):
        raise OSError("cannot read")

    monkeypatch.setattr(retrieval_query.np, "fromfile", _raise_fromfile)
    assert retrieval_query._load_embedding(
        {"embedding_uri": str(valid_embedding_path)},
        storage_root=None,
        bucket=None,
    ) is None

    bad_index = tmp_path / "bad_index.jsonl"
    bad_index.write_text("\nnot-json\n" + json.dumps(transformed) + "\n", encoding="utf-8")
    assert retrieval_query._load_site_index(tmp_path / "missing.jsonl") == []
    assert retrieval_query._load_site_index(bad_index) == [transformed]

    storage_root = tmp_path / "storage"
    bucket_file = storage_root / "bucket-a" / "nested" / "asset.bin"
    bucket_file.parent.mkdir(parents=True)
    bucket_file.write_bytes(b"asset")
    flat_file = storage_root / "flat.bin"
    flat_file.write_bytes(b"flat")

    assert retrieval_query._uri_to_local(
        "gs://bucket-a/nested/asset.bin",
        storage_root=storage_root,
        bucket="bucket-a",
    ) == bucket_file
    assert retrieval_query._uri_to_local(
        "gs://different/flat.bin",
        storage_root=storage_root,
        bucket="bucket-a",
    ) == flat_file
    assert retrieval_query._uri_to_local(
        "gs://bucket-a/missing.bin",
        storage_root=storage_root,
        bucket="bucket-a",
    ) == storage_root / "bucket-a" / "missing.bin"
    assert retrieval_query._uri_to_local(
        "gs://bucket-a/nested/asset.bin",
        storage_root=None,
        bucket=None,
    ) is None
