"""Tests for retrieval_query.py — K-NN retrieval from site_reference_index.jsonl."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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
