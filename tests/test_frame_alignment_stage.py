from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blueprint_pipeline.frame_alignment_stage import run_frame_alignment_stage


def _write_embedding(path: Path, hot_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vec = np.zeros(1024, dtype=np.float32)
    vec[hot_index] = 1.0
    path.write_bytes(vec.tobytes())


def _write_depth_and_confidence(root: Path, index: int) -> tuple[str, str]:
    depth_path = root / f"depth_{index:06d}.npy"
    confidence_path = root / f"confidence_{index:06d}.npy"
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(depth_path, np.full((24, 32), 2.0 + (index * 0.1), dtype=np.float32))
    np.save(confidence_path, np.full((24, 32), 0.95, dtype=np.float32))
    return str(depth_path), str(confidence_path)


def _record(
    *,
    session_id: str,
    capture_id: str,
    reference_id: str,
    frame_index: int,
    tx: float,
    embedding_uri: str,
    depth_uri: str,
    confidence_uri: str,
) -> dict[str, object]:
    T = np.eye(4, dtype=float)
    T[0, 3] = tx
    return {
        "reference_id": reference_id,
        "site_id": "site-1",
        "capture_id": capture_id,
        "scene_id": "scene-1",
        "pass_id": "pass-1",
        "pass_index": 0,
        "capture_session_id": session_id,
        "coordinate_frame_session_id": session_id,
        "chunk_id": "chunk_000",
        "chunk_order": 0,
        "site_frame_transform": None,
        "frame_id": f"{frame_index:06d}",
        "frame_index": frame_index,
        "t_capture_sec": float(frame_index),
        "T_world_camera": T.tolist(),
        "T_site_camera": None,
        "intrinsics": {"fx": 24.0, "fy": 24.0, "cx": 16.0, "cy": 12.0},
        "depth_uri": depth_uri,
        "confidence_uri": confidence_uri,
        "embedding_uri": embedding_uri,
        "embedding_model_id": "test-model",
        "frame_uri": None,
        "thumbnail_uri": None,
        "privacy_source": "privacy/final_walkthrough.mov",
        "geometry_source": "arkit",
        "quality": {"sharpness_score": 90.0},
        "anchor_observations": ["anchor_entry"],
        "retrieval_signals": {"capture_confidence": 0.95},
        "staticness_score": 0.92,
        "geometry_fingerprint": {
            "available": True,
            "depth_histogram_8": [0.0, 0.0, 0.0, 0.3, 0.4, 0.3, 0.0, 0.0],
            "median_depth_m": 2.2,
            "plane_support_ratio": 0.8,
            "surface_complexity": 0.1,
        },
        "visibility_cells": ["0,0", "1,0"],
        "zone_id": "zone-1",
        "captured_at": "2026-03-20T12:00:00Z",
        "indexed_at": "2026-03-20T12:10:00Z",
    }


def test_frame_alignment_stage_writes_alignment_and_static_map_artifacts(tmp_path: Path) -> None:
    bucket = "bucket"
    site_id = "site-1"
    capture_root = tmp_path / bucket / "scenes" / "scene-1" / "captures" / "capture-query"
    capture_root.mkdir(parents=True, exist_ok=True)
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-query",
                "metadata": {"site_identity": {"site_id": site_id}},
            }
        ),
        encoding="utf-8",
    )

    site_root = tmp_path / bucket / "sites" / site_id / "reference_memory"
    site_root.mkdir(parents=True, exist_ok=True)
    embeddings_root = site_root / "embeddings"
    depth_root = site_root / "depth"

    records: list[dict[str, object]] = []
    for i in range(8):
        ref_emb = embeddings_root / "ref" / f"{i:06d}.bin"
        query_emb = embeddings_root / "query" / f"{i:06d}.bin"
        _write_embedding(ref_emb, i)
        _write_embedding(query_emb, i)
        ref_depth, ref_conf = _write_depth_and_confidence(depth_root / "ref", i)
        query_depth, query_conf = _write_depth_and_confidence(depth_root / "query", i)
        records.append(
            _record(
                session_id="session-ref",
                capture_id="capture-ref",
                reference_id=f"ref-{i}",
                frame_index=i,
                tx=10.0 + (i * 0.2),
                embedding_uri=str(ref_emb),
                depth_uri=ref_depth,
                confidence_uri=ref_conf,
            )
        )
        records.append(
            _record(
                session_id="session-query",
                capture_id="capture-query",
                reference_id=f"query-{i}",
                frame_index=i,
                tx=i * 0.2,
                embedding_uri=str(query_emb),
                depth_uri=query_depth,
                confidence_uri=query_conf,
            )
        )

    site_index_path = site_root / "site_reference_index.jsonl"
    with site_index_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")

    (site_root / "site_reference_manifest.json").write_text(
        json.dumps({"schema_version": "v2", "site_id": site_id, "captures": []}),
        encoding="utf-8",
    )
    (site_root / "site_overlap_graph.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "site_id": site_id,
                "nodes": [],
                "edges": [
                    {
                        "edge_id": "edge_chunk_000_chunk_000",
                        "from_session_id": "session-query",
                        "to_session_id": "session-ref",
                        "accepted_for_alignment": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_frame_alignment_stage(capture_root=capture_root)

    assert result["status"] == "completed"
    assert result["sessions_aligned"] == 2
    assert (site_root / "site_transforms.json").is_file()
    assert (site_root / "alignment_validation.json").is_file()
    assert (site_root / "site_static_map_manifest.json").is_file()
    assert (site_root / "static_memory" / "static_pointcloud.ply").is_file()

    graph = json.loads((site_root / "site_overlap_graph.json").read_text(encoding="utf-8"))
    assert graph["edges"][0]["accepted_for_alignment"] is True

    updated_records = [json.loads(line) for line in site_index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    query_records = [row for row in updated_records if row["coordinate_frame_session_id"] == "session-query"]
    assert query_records
    assert all(row["site_frame_transform"] is not None for row in query_records)
    assert all(row["T_site_camera"] is not None for row in query_records)
