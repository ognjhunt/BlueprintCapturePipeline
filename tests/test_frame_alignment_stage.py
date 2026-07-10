from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import blueprint_pipeline.frame_alignment_stage as fas
from blueprint_pipeline.common import PipelineError
from blueprint_pipeline.frame_alignment_stage import run_frame_alignment_stage
from blueprint_pipeline.local_capture import LocalCaptureContext


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


def _ctx(tmp_path: Path) -> LocalCaptureContext:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-query"
    capture_root.mkdir(parents=True, exist_ok=True)
    return LocalCaptureContext(
        capture_root=capture_root,
        raw_root=capture_root / "raw",
        pipeline_root=capture_root / "pipeline",
        descriptor_path=capture_root / "capture_descriptor.json",
        raw_complete_path=capture_root / "raw" / "capture_upload_complete.json",
        storage_root=tmp_path,
        bucket="bucket",
        scene_id="scene-1",
        capture_id="capture-query",
    )


def _write_descriptor(capture_root: Path, payload: dict[str, object]) -> None:
    capture_root.mkdir(parents=True, exist_ok=True)
    (capture_root / "capture_descriptor.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_site_index(site_root: Path, rows: list[dict[str, object]]) -> Path:
    site_root.mkdir(parents=True, exist_ok=True)
    site_index_path = site_root / "site_reference_index.jsonl"
    with site_index_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return site_index_path


def _patch_expensive_alignment_writers(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "_write_static_memory_artifacts",
        "_update_coverage_map",
        "_write_site_manifest",
        "_write_site_memory_indices",
        "_write_retrieval_validation",
        "_update_site_manifest_alignment",
        "_write_site_reference_summary_projection",
    ):
        monkeypatch.setattr(fas, name, lambda **_: None)


def _pose(tx: float = 0.0) -> list[list[float]]:
    T = np.eye(4, dtype=float)
    T[0, 3] = tx
    return T.tolist()


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
    assert (site_root / "site_reference_summary_projection.json").is_file()
    assert (site_root / "static_memory" / "static_pointcloud.ply").is_file()

    graph = json.loads((site_root / "site_overlap_graph.json").read_text(encoding="utf-8"))
    assert graph["edges"][0]["accepted_for_alignment"] is True

    updated_records = [json.loads(line) for line in site_index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    query_records = [row for row in updated_records if row["coordinate_frame_session_id"] == "session-query"]
    assert query_records
    assert all(row["site_frame_transform"] is not None for row in query_records)
    assert all(row["T_site_camera"] is not None for row in query_records)

    projection = json.loads((site_root / "site_reference_summary_projection.json").read_text(encoding="utf-8"))
    assert projection["storage_class"] == "firestore_summary_safe"
    assert projection["scores"]["aligned_fraction"] == 1.0


def test_frame_alignment_stage_returns_skip_reasons(tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-query"
    _write_descriptor(capture_root, {"capture_id": "capture-query", "metadata": {}})

    assert run_frame_alignment_stage(capture_root=capture_root) == {
        "status": "skipped",
        "reason": "no_site_id",
        "capture_id": "capture-query",
    }

    _write_descriptor(capture_root, {"capture_id": "capture-query", "metadata": {"site_identity": {"site_id": "site-1"}}})
    assert run_frame_alignment_stage(capture_root=capture_root)["reason"] == "no_site_index"

    site_root = tmp_path / "bucket" / "sites" / "site-1" / "reference_memory"
    _write_site_index(site_root, [])
    assert run_frame_alignment_stage(capture_root=capture_root)["reason"] == "empty_site_index"

    _write_site_index(
        site_root,
        [
            _record(
                session_id="session-ref",
                capture_id="capture-ref",
                reference_id="ref-0",
                frame_index=0,
                tx=0.0,
                embedding_uri="",
                depth_uri="",
                confidence_uri="",
            )
        ],
    )
    result = run_frame_alignment_stage(capture_root=capture_root)
    assert result["reason"] == "single_session_no_alignment_needed"
    assert result["session_count"] == 1


def test_frame_alignment_stage_skips_already_aligned_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_expensive_alignment_writers(monkeypatch)
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-query"
    _write_descriptor(capture_root, {"capture_id": "capture-query", "metadata": {"site_identity": {"site_id": "site-1"}}})
    site_root = tmp_path / "bucket" / "sites" / "site-1" / "reference_memory"
    query = _record(
        session_id="session-query",
        capture_id="capture-query",
        reference_id="query-0",
        frame_index=0,
        tx=1.0,
        embedding_uri="",
        depth_uri="",
        confidence_uri="",
    )
    query["site_frame_transform"] = _pose(10.0)
    _write_site_index(
        site_root,
        [
            _record(
                session_id="session-ref",
                capture_id="capture-ref",
                reference_id="ref-0",
                frame_index=0,
                tx=0.0,
                embedding_uri="",
                depth_uri="",
                confidence_uri="",
            ),
            query,
        ],
    )

    result = run_frame_alignment_stage(capture_root=capture_root)

    assert result["status"] == "completed"
    assert result["sessions_aligned"] == 2
    assert result["session_results"] == [
        {"session_id": "session-query", "status": "skipped", "reason": "already_aligned"}
    ]


def test_frame_alignment_stage_keeps_insufficient_matches_pending_and_drops_hard_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_expensive_alignment_writers(monkeypatch)
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-query"
    _write_descriptor(capture_root, {"capture_id": "capture-query", "metadata": {"site_identity": {"site_id": "site-1"}}})
    site_root = tmp_path / "bucket" / "sites" / "site-1" / "reference_memory"
    rows = [
        _record(
            session_id="session-ref",
            capture_id="capture-ref",
            reference_id="ref-0",
            frame_index=0,
            tx=0.0,
            embedding_uri="",
            depth_uri="",
            confidence_uri="",
        ),
        _record(
            session_id="session-stall",
            capture_id="capture-stall",
            reference_id="stall-0",
            frame_index=0,
            tx=1.0,
            embedding_uri="",
            depth_uri="",
            confidence_uri="",
        ),
        _record(
            session_id="session-drop",
            capture_id="capture-drop",
            reference_id="drop-0",
            frame_index=0,
            tx=2.0,
            embedding_uri="",
            depth_uri="",
            confidence_uri="",
        ),
    ]
    _write_site_index(site_root, rows)

    def fake_align(**kwargs: object) -> dict[str, object]:
        session_id = str(kwargs["session_id"])
        if session_id == "session-stall":
            return {"session_id": session_id, "status": "failed", "reason": "insufficient_matches", "combined_score": 0.2}
        return {"session_id": session_id, "status": "failed", "reason": "ransac_failed", "combined_score": 0.1}

    monkeypatch.setattr(fas, "_align_session", fake_align)

    result = run_frame_alignment_stage(capture_root=capture_root)

    reasons = {item["session_id"]: item["reason"] for item in result["session_results"]}
    assert reasons == {"session-stall": "insufficient_matches", "session-drop": "ransac_failed"}
    assert result["sessions_aligned"] == 1


def test_alignment_session_failure_reasons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path)
    no_embeddings = [
        _record(
            session_id="session-query",
            capture_id="capture-query",
            reference_id="query-0",
            frame_index=0,
            tx=0.0,
            embedding_uri="",
            depth_uri="",
            confidence_uri="",
        )
    ]
    assert fas._align_session(
        session_id="session-query",
        session_records=no_embeddings,
        anchor_session_id="session-ref",
        anchor_records=no_embeddings,
        anchor_transform=None,
        ctx=ctx,
    )["reason"] == "no_embeddings"

    missing_embedding_rows = [
        dict(no_embeddings[0], embedding_uri=""),
        dict(no_embeddings[0], embedding_uri=str(tmp_path / "missing.bin")),
    ]
    embeddings, filtered = fas._load_session_embeddings(missing_embedding_rows, ctx)
    assert embeddings is None
    assert filtered == []

    query_embedding = tmp_path / "query.bin"
    anchor_embedding = tmp_path / "anchor.bin"
    _write_embedding(query_embedding, 0)
    _write_embedding(anchor_embedding, 0)
    query_records = [
        _record(
            session_id="session-query",
            capture_id="capture-query",
            reference_id="query-0",
            frame_index=0,
            tx=0.0,
            embedding_uri=str(query_embedding),
            depth_uri="",
            confidence_uri="",
        )
    ]
    anchor_records = [
        _record(
            session_id="session-ref",
            capture_id="capture-ref",
            reference_id="ref-0",
            frame_index=0,
            tx=1.0,
            embedding_uri="",
            depth_uri="",
            confidence_uri="",
        )
    ]
    assert fas._align_session(
        session_id="session-query",
        session_records=query_records,
        anchor_session_id="session-ref",
        anchor_records=anchor_records,
        anchor_transform=None,
        ctx=ctx,
    )["reason"] == "anchor_session_has_no_embeddings"

    anchor_records[0]["embedding_uri"] = str(anchor_embedding)
    insufficient = fas._align_session(
        session_id="session-query",
        session_records=query_records,
        anchor_session_id="session-ref",
        anchor_records=anchor_records,
        anchor_transform=None,
        ctx=ctx,
    )
    assert insufficient["reason"] == "insufficient_matches"
    assert insufficient["candidate_count"] == 1

    monkeypatch.setattr(fas, "_find_candidate_matches", lambda **_: [(query_records[0], anchor_records[0], 1.0)] * 8)
    monkeypatch.setattr(fas, "_ransac_se3", lambda **_: (np.eye(4), 1, 0.1))
    ransac_failed = fas._align_session(
        session_id="session-query",
        session_records=query_records,
        anchor_session_id="session-ref",
        anchor_records=anchor_records,
        anchor_transform=None,
        ctx=ctx,
    )
    assert ransac_failed["reason"] == "ransac_failed"
    assert ransac_failed["inlier_fraction"] == 0.1


def test_candidate_matching_can_add_reverse_unique_match() -> None:
    query_records = [
        _record(
            session_id="session-query",
            capture_id="capture-query",
            reference_id="query-0",
            frame_index=0,
            tx=0.0,
            embedding_uri="",
            depth_uri="",
            confidence_uri="",
        ),
        _record(
            session_id="session-query",
            capture_id="capture-query",
            reference_id="query-1",
            frame_index=1,
            tx=1.0,
            embedding_uri="",
            depth_uri="",
            confidence_uri="",
        ),
    ]
    ref_records = [
        _record(
            session_id="session-ref",
            capture_id="capture-ref",
            reference_id="ref-0",
            frame_index=0,
            tx=10.0,
            embedding_uri="",
            depth_uri="",
            confidence_uri="",
        ),
        _record(
            session_id="session-ref",
            capture_id="capture-ref",
            reference_id="ref-1",
            frame_index=1,
            tx=11.0,
            embedding_uri="",
            depth_uri="",
            confidence_uri="",
        ),
    ]
    ref_records[1]["T_world_camera"] = None
    candidates = fas._find_candidate_matches(
        query_embeddings=np.array([[0.1, 0.0], [0.8, 0.9]], dtype=np.float64),
        query_record_index=query_records,
        ref_embeddings=np.eye(2, dtype=np.float64),
        ref_record_index=ref_records,
        sim_threshold=0.75,
    )

    assert [(item[0]["reference_id"], item[1]["reference_id"]) for item in candidates] == [("query-1", "ref-0")]
    assert fas._has_valid_pose({"T_world_camera": None}) is False


def test_ransac_and_refinement_reject_outliers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    q0 = _record(
        session_id="session-query",
        capture_id="capture-query",
        reference_id="query-0",
        frame_index=0,
        tx=0.0,
        embedding_uri="",
        depth_uri="",
        confidence_uri="",
    )
    r0 = _record(
        session_id="session-ref",
        capture_id="capture-ref",
        reference_id="ref-0",
        frame_index=0,
        tx=0.0,
        embedding_uri="",
        depth_uri="",
        confidence_uri="",
    )
    q_trans = dict(q0, reference_id="query-trans")
    r_trans = dict(r0, reference_id="ref-trans", T_world_camera=_pose(10.0))
    q_rot = dict(q0, reference_id="query-rot")
    T_rot = np.eye(4, dtype=float)
    T_rot[0, 0] = -1.0
    T_rot[1, 1] = -1.0
    r_rot = dict(r0, reference_id="ref-rot", T_world_camera=T_rot.tolist())
    candidates = [(q0, r0, 1.0), (q_trans, r_trans, 1.0), (q_rot, r_rot, 1.0)]
    first_transform, inliers, fraction = fas._ransac_se3(
        candidates=candidates,
        n_iters=len(candidates),
        translation_threshold_m=0.1,
        rotation_threshold_deg=20.0,
    )

    assert inliers == 1
    assert fraction == pytest.approx(1 / 3)
    second_transform, second_inliers, second_fraction = fas._ransac_se3(
        candidates=candidates,
        n_iters=len(candidates),
        translation_threshold_m=0.1,
        rotation_threshold_deg=20.0,
    )
    np.testing.assert_array_equal(first_transform, second_transform)
    assert second_inliers == inliers
    assert second_fraction == fraction
    assert fas._transform_fingerprint(first_transform) == fas._transform_fingerprint(second_transform)
    refined = fas._refine_translation(
        T_site_from_session=np.eye(4),
        candidates=candidates[1:],
        translation_threshold_m=0.1,
        rotation_threshold_deg=20.0,
    )
    np.testing.assert_array_equal(refined, np.eye(4))

    monkeypatch.setattr(fas, "_mat_inv", lambda _T: (_ for _ in ()).throw(RuntimeError("bad inverse")))
    failed_T, failed_inliers, failed_fraction = fas._ransac_se3(
        candidates=[(q0, r0, 1.0)],
        n_iters=1,
        translation_threshold_m=0.1,
        rotation_threshold_deg=20.0,
    )
    assert failed_T is None
    assert failed_inliers == 0
    assert failed_fraction == 0.0

    reflected = dict(q0)
    bad_pose = np.eye(4, dtype=float)
    bad_pose[0, 0] = -1.0
    reflected["T_world_camera"] = bad_pose.tolist()
    with pytest.raises(ValueError, match="invalid T_world_camera"):
        fas._ransac_se3(
            candidates=[(reflected, r0, 1.0)],
            n_iters=1,
            translation_threshold_m=0.1,
            rotation_threshold_deg=20.0,
        )


def test_static_memory_artifact_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site_root = tmp_path / "reference_memory"
    unaligned_index = _write_site_index(
        site_root,
        [
            _record(
                session_id="session-ref",
                capture_id="capture-ref",
                reference_id="unaligned",
                frame_index=0,
                tx=0.0,
                embedding_uri="",
                depth_uri="",
                confidence_uri="",
            )
        ],
    )

    fas._write_static_memory_artifacts(
        site_root=site_root,
        site_index_path=unaligned_index,
        site_id="site-1",
        storage_root=tmp_path,
    )
    assert not (site_root / "site_static_map_manifest.json").exists()

    def aligned(reference_id: str, *, staticness: float, depth_uri: str = "valid-depth") -> dict[str, object]:
        record = _record(
            session_id="session-ref",
            capture_id="capture-ref",
            reference_id=reference_id,
            frame_index=0,
            tx=0.0,
            embedding_uri="",
            depth_uri=depth_uri,
            confidence_uri="valid-confidence",
        )
        record["site_frame_transform"] = _pose()
        record["T_site_camera"] = _pose()
        record["staticness_score"] = staticness
        return record

    mixed_index = _write_site_index(
        site_root,
        [
            aligned("dynamic", staticness=0.2),
            aligned("missing-depth", staticness=0.9, depth_uri="missing-depth"),
            aligned("no-pose", staticness=0.9),
            aligned("empty-points", staticness=0.9),
        ],
    )

    def fake_load_numeric_array(uri: object, *, storage_root: Path) -> np.ndarray | None:
        if uri == "missing-depth":
            return None
        return np.ones((2, 2), dtype=np.float32)

    def fake_effective_pose(record: dict[str, object]) -> np.ndarray | None:
        if record["reference_id"] == "no-pose":
            return None
        return np.eye(4)

    monkeypatch.setattr(fas, "_sm_load_numeric_array", fake_load_numeric_array)
    monkeypatch.setattr(fas, "effective_pose", fake_effective_pose)
    monkeypatch.setattr(fas, "backproject_depth_points", lambda **_: np.zeros((0, 4), dtype=np.float32))

    fas._write_static_memory_artifacts(
        site_root=site_root,
        site_index_path=mixed_index,
        site_id="site-1",
        storage_root=tmp_path,
    )

    manifest = json.loads((site_root / "site_static_map_manifest.json").read_text(encoding="utf-8"))
    assert manifest["point_count"] == 0
    assert manifest["dynamic_observation_count"] == 1
    dynamic_rows = (site_root / "static_memory" / "dynamic_observations.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(dynamic_rows[0])["reference_id"] == "dynamic"


def test_manifest_descriptor_pose_and_uri_edges(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    with pytest.raises(PipelineError):
        fas._load_descriptor(ctx)

    fas._update_site_manifest_alignment(
        site_root=tmp_path / "missing-site-root",
        site_index_path=tmp_path / "missing-site-root" / "site_reference_index.jsonl",
        site_id="site-1",
        aligned_sessions=[],
    )

    with pytest.raises(ValueError, match="no T_world_camera"):
        fas._pose_matrix({"frame_id": "missing-pose"})
    with pytest.raises(ValueError, match="not 4x4"):
        fas._pose_matrix({"T_world_camera": [[1.0]]})

    direct = tmp_path / "direct.txt"
    direct.write_text("ok", encoding="utf-8")
    assert fas._uri_to_local(str(direct), ctx) == direct

    flat = tmp_path / "flat" / "artifact.txt"
    flat.parent.mkdir(parents=True, exist_ok=True)
    flat.write_text("ok", encoding="utf-8")
    assert fas._uri_to_local("gs://bucket/flat/artifact.txt", ctx) == flat

    missing = tmp_path / "not-created.txt"
    assert fas._uri_to_local(str(missing), ctx) == missing
