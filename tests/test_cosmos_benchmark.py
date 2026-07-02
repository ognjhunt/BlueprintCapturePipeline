from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
pytest.importorskip("PIL")
from PIL import Image

from blueprint_pipeline.synthesis import cosmos_benchmark
from blueprint_pipeline.capture_orchestrator import PipelineConfig
from blueprint_pipeline.synthesis.cosmos_benchmark import (
    _video_bootstrap_reference_policy,
    run_cosmos_single_capture_smoke_lane,
    run_cosmos_zero_shot_validation_lane,
)


def _pose(tx: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_cosmos_benchmark_records_decoupled_reference_selection(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    pipeline_root = capture_root / "pipeline"
    (capture_root / "world_model_export").mkdir(parents=True)
    (pipeline_root / "scene_memory").mkdir(parents=True)
    (pipeline_root / "evaluation_prep").mkdir(parents=True)

    dense_rows = [
        {
            "reference_id": "target",
            "frame_id": "frame_0001",
            "frame_index": 1,
            "frame_uri": "gs://bucket/frames/frame_0001.jpg",
            "included_in_index": True,
            "t_capture_sec": 0.0,
            "T_world_camera": _pose(0.0),
            "anchor_observations": [],
            "retrieval_signals": {"anchor_observation_count": 0, "route_anchor_density": 0.0, "capture_confidence": 0.8},
        },
        {
            "reference_id": "near-duplicate",
            "frame_id": "frame_0002",
            "frame_index": 2,
            "frame_uri": "gs://bucket/frames/frame_0002.jpg",
            "included_in_index": True,
            "t_capture_sec": 0.05,
            "T_world_camera": _pose(0.01),
            "anchor_observations": [],
            "retrieval_signals": {"anchor_observation_count": 0, "route_anchor_density": 0.0, "capture_confidence": 0.85},
        },
        {
            "reference_id": "decoupled",
            "frame_id": "frame_0003",
            "frame_index": 8,
            "frame_uri": "gs://bucket/frames/frame_0003.jpg",
            "included_in_index": True,
            "t_capture_sec": 0.7,
            "T_world_camera": _pose(0.35),
            "anchor_observations": [],
            "retrieval_signals": {"anchor_observation_count": 0, "route_anchor_density": 0.3, "capture_confidence": 0.92},
        },
        {
            "reference_id": "decoupled-rich",
            "frame_id": "frame_0004",
            "frame_index": 12,
            "frame_uri": "gs://bucket/frames/frame_0004.jpg",
            "included_in_index": True,
            "t_capture_sec": 1.2,
            "T_world_camera": _pose(0.55),
            "anchor_observations": ["anchor_entry"],
            "retrieval_signals": {
                "anchor_observation_count": 1,
                "route_anchor_density": 0.75,
                "checkpoint_proximity_sec": 0.1,
                "capture_confidence": 0.95,
                "geometry_grounding_quality": 1.0,
            },
        },
    ]
    (capture_root / "world_model_export" / "dense_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in dense_rows),
        encoding="utf-8",
    )
    (pipeline_root / "scene_memory" / "conditioning_bundle.json").write_text("{}", encoding="utf-8")
    (pipeline_root / "evaluation_prep" / "task_anchor_manifest.json").write_text(
        json.dumps({"tasks": [{"task_id": "task-1", "target_object_ids": ["obj-1"]}]}),
        encoding="utf-8",
    )
    (pipeline_root / "evaluation_prep" / "protected_regions_manifest.json").write_text(
        json.dumps({"regions": []}),
        encoding="utf-8",
    )

    descriptor_path = capture_root / "capture_descriptor.json"
    descriptor_path.write_text(
        json.dumps({"capture_id": "capture-1", "scene_id": "scene-1", "quality": {}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "blueprint_pipeline.synthesis.cosmos_benchmark._probe_cosmos_runtime",
        lambda: {"status": "blocked", "blockers": ["missing_cosmos_runtime_package"], "packages": {}, "model_id": "test"},
    )

    manifest = run_cosmos_zero_shot_validation_lane(
        capture_root=capture_root,
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        cfg=PipelineConfig(gcs_root=tmp_path),
        max_examples=3,
    )

    reference_selection_manifest = json.loads(
        Path(manifest["reference_selection_manifest_path"]).read_text(encoding="utf-8")
    )
    reference_selection_comparison = json.loads(
        Path(manifest["reference_selection_comparison_path"]).read_text(encoding="utf-8")
    )
    synthetic_trajectory_manifest = json.loads(
        Path(manifest["synthetic_trajectory_manifest_path"]).read_text(encoding="utf-8")
    )
    future_anchor_manifest = json.loads(
        Path(manifest["future_anchor_regrounding_manifest_path"]).read_text(encoding="utf-8")
    )
    target_entry = next(
        entry for entry in reference_selection_manifest["entries"] if entry["target_frame_id"] == "frame_0001"
    )
    validation_entry = next(
        entry for entry in manifest["validation_set"] if entry["frame_id"] == "frame_0001"
    )

    assert manifest["status"] == "blocked"
    assert manifest["target_reference_decoupling_mode"] == "temporal_gap_with_pose_and_anchor_reranking"
    assert manifest["rejected_near_duplicate_count"] >= 1
    assert target_entry["selected_reference_frame_ids"][0] == "frame_0004"
    assert target_entry["rejected_near_duplicate_count"] >= 1
    assert validation_entry["selected_reference_frame_ids"][0] == "frame_0004"
    assert reference_selection_comparison["changed_primary_reference_count"] >= 1
    assert reference_selection_comparison["quality_metrics"]["primary_temporal_gap_sec"]["delta"] > 0
    assert synthetic_trajectory_manifest["augmented_target_count"] >= 1
    assert validation_entry["synthetic_waypoint_count"] >= 1
    assert validation_entry["synthetic_trajectory_status"] == "augmented"
    assert manifest["sparse_view_interpolation"]["interpolated_target_count"] == 0
    assert future_anchor_manifest["re_grounded_target_count"] >= 1
    assert validation_entry["future_anchor_status"] == "re_grounded"
    assert validation_entry["future_anchor_count"] >= 1


def test_cosmos_benchmark_surfaces_sparse_view_interpolation_when_context_is_sparse(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    pipeline_root = capture_root / "pipeline"
    (capture_root / "world_model_export").mkdir(parents=True)
    (pipeline_root / "scene_memory").mkdir(parents=True)
    (pipeline_root / "evaluation_prep").mkdir(parents=True)

    dense_rows = [
        {
            "reference_id": "target",
            "frame_id": "frame_0001",
            "frame_index": 1,
            "frame_uri": "gs://bucket/frames/frame_0001.jpg",
            "included_in_index": True,
            "t_capture_sec": 0.0,
            "T_world_camera": _pose(0.0),
            "anchor_observations": [],
            "retrieval_signals": {"anchor_observation_count": 0, "route_anchor_density": 0.0, "capture_confidence": 0.8},
        },
        {
            "reference_id": "decoupled-ref",
            "frame_id": "frame_0002",
            "frame_index": 10,
            "frame_uri": "gs://bucket/frames/frame_0002.jpg",
            "included_in_index": True,
            "t_capture_sec": 1.1,
            "T_world_camera": _pose(0.5),
            "anchor_observations": ["anchor_entry"],
            "retrieval_signals": {
                "anchor_observation_count": 1,
                "route_anchor_density": 0.8,
                "checkpoint_proximity_sec": 0.2,
                "capture_confidence": 0.94,
                "geometry_grounding_quality": 1.0,
            },
        },
    ]
    (capture_root / "world_model_export" / "dense_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in dense_rows),
        encoding="utf-8",
    )
    (pipeline_root / "scene_memory" / "conditioning_bundle.json").write_text("{}", encoding="utf-8")
    (pipeline_root / "evaluation_prep" / "task_anchor_manifest.json").write_text(
        json.dumps({"tasks": [{"task_id": "task-1", "target_object_ids": ["obj-1"]}]}),
        encoding="utf-8",
    )
    (pipeline_root / "evaluation_prep" / "protected_regions_manifest.json").write_text(
        json.dumps({"regions": []}),
        encoding="utf-8",
    )
    descriptor_path = capture_root / "capture_descriptor.json"
    descriptor_path.write_text(
        json.dumps({"capture_id": "capture-1", "scene_id": "scene-1", "quality": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "blueprint_pipeline.synthesis.cosmos_benchmark._probe_cosmos_runtime",
        lambda: {"status": "blocked", "blockers": ["missing_cosmos_runtime_package"], "packages": {}, "model_id": "test"},
    )

    manifest = run_cosmos_zero_shot_validation_lane(
        capture_root=capture_root,
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        cfg=PipelineConfig(gcs_root=tmp_path),
        max_examples=2,
    )
    interpolation_manifest = json.loads(
        Path(manifest["sparse_view_interpolation_manifest_path"]).read_text(encoding="utf-8")
    )
    future_anchor_manifest = json.loads(
        Path(manifest["future_anchor_regrounding_manifest_path"]).read_text(encoding="utf-8")
    )
    validation_entry = next(entry for entry in manifest["validation_set"] if entry["frame_id"] == "frame_0001")

    assert interpolation_manifest["interpolated_target_count"] >= 1
    assert validation_entry["synthetic_trajectory_status"] == "skipped"
    assert validation_entry["sparse_view_interpolation_status"] == "interpolated"
    assert validation_entry["interpolated_view_count"] >= 1
    assert future_anchor_manifest["re_grounded_target_count"] >= 1
    assert validation_entry["future_anchor_status"] == "re_grounded"
    assert validation_entry["future_anchor_count"] >= 1


def test_video_bootstrap_reference_policy_expands_temporal_window_for_sparse_clip() -> None:
    policy = _video_bootstrap_reference_policy(
        [
            {"t_capture_sec": 0.0},
            {"t_capture_sec": 21.0},
            {"t_capture_sec": 42.0},
            {"t_capture_sec": 63.0},
        ]
    )

    assert policy is not None
    assert policy["max_temporal_window_sec"] > 63.0
    assert policy["preferred_temporal_gap_sec"] >= 1.5


def _capture_fixture(tmp_path: Path) -> tuple[Path, Path, PipelineConfig, str]:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    pipeline_root = capture_root / "pipeline"
    (capture_root / "world_model_export").mkdir(parents=True)
    (pipeline_root / "scene_memory").mkdir(parents=True)
    (pipeline_root / "evaluation_prep").mkdir(parents=True)
    (pipeline_root / "scene_memory" / "conditioning_bundle.json").write_text("{}", encoding="utf-8")
    (pipeline_root / "evaluation_prep" / "task_anchor_manifest.json").write_text(
        json.dumps({"tasks": [{"task_id": "task-1", "target_object_ids": ["obj-1"]}]}),
        encoding="utf-8",
    )
    (pipeline_root / "evaluation_prep" / "protected_regions_manifest.json").write_text(
        json.dumps({"regions": [], "grounding_status": "grounded"}),
        encoding="utf-8",
    )
    descriptor_path = capture_root / "capture_descriptor.json"
    descriptor_path.write_text(
        json.dumps({"capture_id": "capture-1", "scene_id": "scene-1", "quality": {}}),
        encoding="utf-8",
    )
    descriptor_uri = "gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json"
    return capture_root, pipeline_root, PipelineConfig(gcs_root=tmp_path), descriptor_uri


def _write_dense_rows(capture_root: Path, rows: list[dict]) -> None:
    (capture_root / "world_model_export" / "dense_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_cosmos_benchmark_private_helpers_and_runtime_probe(monkeypatch, tmp_path: Path) -> None:
    missing_json = tmp_path / "missing.json"
    assert cosmos_benchmark._read_json(missing_json) == {}
    non_mapping = tmp_path / "list.json"
    non_mapping.write_text("[]", encoding="utf-8")
    assert cosmos_benchmark._read_json(non_mapping) == {}
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('\n{"a": 1}\n[]\n{"b": 2}\n', encoding="utf-8")
    assert cosmos_benchmark._read_jsonl(tmp_path / "missing.jsonl") == []
    assert cosmos_benchmark._read_jsonl(jsonl) == [{"a": 1}, {"b": 2}]

    monkeypatch.setattr(cosmos_benchmark.importlib.util, "find_spec", lambda name: object() if name == "torch" else None)
    blocked = cosmos_benchmark._probe_cosmos_runtime()
    assert blocked["status"] == "blocked"
    assert "missing_cosmos_runtime_package" in blocked["blockers"]
    monkeypatch.setattr(cosmos_benchmark.importlib.util, "find_spec", lambda _name: None)
    assert "missing_torch" in cosmos_benchmark._probe_cosmos_runtime()["blockers"]
    monkeypatch.setattr(cosmos_benchmark.importlib.util, "find_spec", lambda _name: object())
    assert cosmos_benchmark._probe_cosmos_runtime()["status"] == "ready"
    assert cosmos_benchmark._runtime_blocked_reason("No module named cosmos") is True
    assert cosmos_benchmark._runtime_blocked_reason("other failure") is False

    image_path = tmp_path / "frame.jpg"
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(image_path)
    assert cosmos_benchmark._load_frame_image(image_path).shape == (2, 3, 3)
    context = SimpleNamespace(storage_root=tmp_path)
    gs_frame = tmp_path / "bucket" / "frames" / "frame.jpg"
    gs_frame.parent.mkdir(parents=True)
    gs_frame.write_bytes(image_path.read_bytes())
    assert cosmos_benchmark._resolve_record_frame_path(context=context, record={}) is None
    assert cosmos_benchmark._resolve_record_frame_path(context=context, record={"frame_uri": str(image_path)}) == image_path.resolve()
    assert cosmos_benchmark._resolve_record_frame_path(context=context, record={"frame_uri": "gs://bucket/frames/frame.jpg"}) == gs_frame
    assert cosmos_benchmark._resolve_record_frame_path(context=context, record={"frame_uri": str(tmp_path / "absent.jpg")}) is None

    intrinsics = cosmos_benchmark._target_intrinsics({}, (2, 3, 3))
    assert intrinsics["width"] == 3
    assert cosmos_benchmark._target_plucker_map({"T_world_camera": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], "intrinsics": {"width": 3, "height": 2}}, (2, 3, 3)).shape[0] == 6
    assert cosmos_benchmark._target_plucker_map({"intrinsics": {"width": 3, "height": 2}}, (2, 3, 3)) is None
    assert cosmos_benchmark._target_plucker_map({"T_world_camera": [[1, 0], [0, 1]]}, (2, 3, 3)) is None
    assert cosmos_benchmark._smoke_manifest_base(
        benchmark_root=tmp_path,
        runtime_probe={"status": "ready"},
        context=SimpleNamespace(capture_id="cap", scene_id="scene"),
        bootstrap_origin=None,
        bootstrap_source_manifest_path=None,
        reference_selection_manifest_path=tmp_path / "ref.json",
        reference_selection_comparison_path=tmp_path / "cmp.json",
        validation_set=[],
    )["self_contained"] is True
    assert _video_bootstrap_reference_policy([{"t_capture_sec": 1.0}]) is None
    assert _video_bootstrap_reference_policy([{"t_capture_sec": 1.0}, {"t_capture_sec": 2.0}]) is None


def test_cosmos_single_capture_smoke_lane_edges(monkeypatch, tmp_path: Path) -> None:
    capture_root, _pipeline_root, cfg, descriptor_uri = _capture_fixture(tmp_path)
    frame = tmp_path / "reference.jpg"
    Image.fromarray(np.zeros((3, 3, 3), dtype=np.uint8)).save(frame)
    records = [
        {"reference_id": "target", "frame_id": "target", "frame_uri": str(frame), "included_in_index": True, "t_capture_sec": 0.0, "T_world_camera": _pose(0.0), "intrinsics": {"width": 3, "height": 3}},
        {"reference_id": "ref", "frame_id": "ref", "frame_uri": str(frame), "included_in_index": True, "t_capture_sec": 1.0, "T_world_camera": _pose(0.4), "intrinsics": {"width": 3, "height": 3}},
        {"reference_id": "missing-ref", "frame_id": "missing", "frame_uri": str(tmp_path / "missing.jpg"), "included_in_index": True, "t_capture_sec": 2.0, "T_world_camera": _pose(0.8), "intrinsics": {"width": 3, "height": 3}},
    ]
    _write_dense_rows(capture_root, records)

    entries = [
        {"target_frame_id": "invalid", "target_frame_uri": "x", "target_index": 99, "selected_references": [{"candidate_index": 1}], "selected_reference_ids": ["ref"], "selected_reference_frame_ids": ["ref"], "selected_reference_frame_uris": [str(frame)], "decoupling": {"mode": "test"}},
        {"target_frame_id": "no_ref", "target_frame_uri": "x", "target_index": 0, "selected_references": [], "selected_reference_ids": [], "selected_reference_frame_ids": [], "selected_reference_frame_uris": [], "decoupling": {"mode": "test"}},
        {"target_frame_id": "missing_ref", "target_frame_uri": "x", "target_index": 0, "selected_references": [{"candidate_index": 2, "pose_distance_m": 0.3, "temporal_gap_sec": 1.0}], "selected_reference_ids": ["missing-ref"], "selected_reference_frame_ids": ["missing"], "selected_reference_frame_uris": [str(tmp_path / "missing.jpg")], "decoupling": {"mode": "test"}},
        {"target_frame_id": "raises", "target_frame_uri": "x", "target_index": 0, "selected_references": [{"candidate_index": 1, "pose_distance_m": 0.3, "temporal_gap_sec": 1.0}], "selected_reference_ids": ["ref"], "selected_reference_frame_ids": ["ref"], "selected_reference_frame_uris": [str(frame)], "decoupling": {"mode": "test"}},
        {"target_frame_id": "no_output", "target_frame_uri": "x", "target_index": 0, "selected_references": [{"candidate_index": 1, "pose_distance_m": 0.3, "temporal_gap_sec": 1.0}], "selected_reference_ids": ["ref"], "selected_reference_frame_ids": ["ref"], "selected_reference_frame_uris": [str(frame)], "decoupling": {"mode": "test"}},
        {"target_frame_id": "success", "target_frame_uri": "x", "target_index": 0, "selected_references": [{"candidate_index": 1, "pose_distance_m": 0.3, "temporal_gap_sec": 1.0}], "selected_reference_ids": ["ref"], "selected_reference_frame_ids": ["ref"], "selected_reference_frame_uris": [str(frame)], "decoupling": {"mode": "test"}},
    ]

    monkeypatch.setattr(cosmos_benchmark, "_probe_cosmos_runtime", lambda: {"status": "ready", "blockers": [], "packages": {}, "model_id": "test"})
    monkeypatch.setattr(cosmos_benchmark, "build_reference_selection_manifest", lambda **_kwargs: {"entries": entries, "policy": {"target_reference_decoupling_mode": "test"}, "selected_target_count": len(entries), "skipped_target_count": 0, "rejected_near_duplicate_count": 0})
    monkeypatch.setattr(cosmos_benchmark, "build_legacy_reference_selection_manifest", lambda **_kwargs: {"entries": [], "policy": {}})
    monkeypatch.setattr(cosmos_benchmark, "build_reference_selection_comparison", lambda **_kwargs: {"changed_primary_reference_count": 1})
    monkeypatch.setattr(cosmos_benchmark, "load_cosmos_model", lambda: object())

    def fake_generate_view(*, output_path: Path, **_kwargs):
        if output_path.stem == "raises":
            raise RuntimeError("generation failed")
        if output_path.stem == "success":
            output_path.write_bytes(b"jpg")
            output_path.with_suffix(".mp4").write_bytes(b"mp4")
        return output_path

    monkeypatch.setattr(cosmos_benchmark, "generate_view", fake_generate_view)
    manifest = run_cosmos_single_capture_smoke_lane(
        capture_root=capture_root,
        descriptor_gcs_uri=descriptor_uri,
        cfg=cfg,
        max_examples=6,
    )
    assert manifest["status"] == "completed"
    assert manifest["render_success_count"] == 1
    assert {row["reason"] for row in manifest["smoke_examples"] if row["status"] == "skipped"} == {"no_selected_reference", "reference_frame_unavailable"}
    assert any(row.get("reason") == "generation failed" for row in manifest["smoke_examples"])

    monkeypatch.setattr(cosmos_benchmark, "load_cosmos_model", lambda: (_ for _ in ()).throw(ImportError("missing runtime")))
    blocked = run_cosmos_single_capture_smoke_lane(
        capture_root=capture_root,
        descriptor_gcs_uri=descriptor_uri,
        cfg=cfg,
        max_examples=1,
    )
    assert blocked["status"] == "blocked"
    monkeypatch.setattr(cosmos_benchmark, "_probe_cosmos_runtime", lambda: {"status": "blocked", "blockers": ["missing"], "packages": {}, "model_id": "test"})
    runtime_blocked = run_cosmos_single_capture_smoke_lane(
        capture_root=capture_root,
        descriptor_gcs_uri=descriptor_uri,
        cfg=cfg,
        max_examples=1,
    )
    assert runtime_blocked["reason"] == "cosmos_runtime_unavailable"


def test_cosmos_single_capture_smoke_missing_and_bootstrap(monkeypatch, tmp_path: Path) -> None:
    capture_root, _pipeline_root, cfg, descriptor_uri = _capture_fixture(tmp_path)
    _write_dense_rows(capture_root, [])
    monkeypatch.setattr(cosmos_benchmark, "_probe_cosmos_runtime", lambda: {"status": "ready", "blockers": [], "packages": {}, "model_id": "test"})
    monkeypatch.setattr(cosmos_benchmark, "resolve_video_bootstrap_sources", lambda **_kwargs: None)
    missing = run_cosmos_single_capture_smoke_lane(
        capture_root=capture_root,
        descriptor_gcs_uri=descriptor_uri,
        cfg=cfg,
        max_examples=1,
    )
    assert missing["status"] == "missing"

    frame = tmp_path / "bootstrap.jpg"
    Image.fromarray(np.zeros((3, 3, 3), dtype=np.uint8)).save(frame)
    bootstrap_records = [
        {"reference_id": "b0", "frame_id": "b0", "frame_uri": str(frame), "included_in_index": True, "t_capture_sec": 0.0, "T_world_camera": _pose(0.0)},
        {"reference_id": "b1", "frame_id": "b1", "frame_uri": str(frame), "included_in_index": True, "t_capture_sec": 20.0, "T_world_camera": _pose(0.5)},
    ]
    monkeypatch.setattr(cosmos_benchmark, "resolve_video_bootstrap_sources", lambda **_kwargs: {"origin": "raw_video", "video_path": "video.mp4"})
    monkeypatch.setattr(cosmos_benchmark, "extract_video_bootstrap_records", lambda **_kwargs: bootstrap_records)
    monkeypatch.setattr(cosmos_benchmark, "build_reference_selection_manifest", lambda **_kwargs: {"entries": [], "policy": {}, "selected_target_count": 0, "skipped_target_count": 0, "rejected_near_duplicate_count": 0})
    monkeypatch.setattr(cosmos_benchmark, "build_legacy_reference_selection_manifest", lambda **_kwargs: {"entries": [], "policy": {}})
    monkeypatch.setattr(cosmos_benchmark, "build_reference_selection_comparison", lambda **_kwargs: {})
    bootstrapped = run_cosmos_single_capture_smoke_lane(
        capture_root=capture_root,
        descriptor_gcs_uri=descriptor_uri,
        cfg=cfg,
        max_examples=1,
    )
    assert bootstrapped["bootstrap_origin"] == "raw_video"
    assert Path(bootstrapped["bootstrap_source_manifest_path"]).is_file()


def test_cosmos_zero_shot_validation_missing_blocked_and_completed(monkeypatch, tmp_path: Path) -> None:
    capture_root, pipeline_root, cfg, descriptor_uri = _capture_fixture(tmp_path)
    _write_dense_rows(capture_root, [])
    monkeypatch.setattr(cosmos_benchmark, "_probe_cosmos_runtime", lambda: {"status": "ready", "blockers": [], "packages": {}, "model_id": "test"})
    monkeypatch.setattr(cosmos_benchmark, "resolve_video_bootstrap_sources", lambda **_kwargs: None)
    missing = run_cosmos_zero_shot_validation_lane(
        capture_root=capture_root,
        descriptor_gcs_uri=descriptor_uri,
        cfg=cfg,
        max_examples=1,
    )
    assert missing["status"] == "missing"

    bootstrap_records = [
        {"reference_id": "b0", "frame_id": "b0", "frame_uri": "missing", "included_in_index": True, "t_capture_sec": 0.0, "T_world_camera": _pose(0.0)},
        {"reference_id": "b1", "frame_id": "b1", "frame_uri": "missing", "included_in_index": True, "t_capture_sec": 20.0, "T_world_camera": _pose(0.5)},
    ]
    monkeypatch.setattr(cosmos_benchmark, "resolve_video_bootstrap_sources", lambda **_kwargs: {"origin": "raw_video", "video_path": "video.mp4"})
    monkeypatch.setattr(cosmos_benchmark, "extract_video_bootstrap_records", lambda **_kwargs: bootstrap_records)
    monkeypatch.setattr(cosmos_benchmark, "build_reference_selection_manifest", lambda **_kwargs: {"entries": [], "policy": {}, "selected_target_count": 0, "skipped_target_count": 0, "rejected_near_duplicate_count": 0})
    monkeypatch.setattr(cosmos_benchmark, "build_legacy_reference_selection_manifest", lambda **_kwargs: {"entries": [], "policy": {}})
    monkeypatch.setattr(cosmos_benchmark, "build_reference_selection_comparison", lambda **_kwargs: {})
    bootstrapped_missing = run_cosmos_zero_shot_validation_lane(
        capture_root=capture_root,
        descriptor_gcs_uri=descriptor_uri,
        cfg=cfg,
        max_examples=1,
    )
    assert bootstrapped_missing["bootstrap_origin"] == "raw_video"
    assert Path(bootstrapped_missing["bootstrap_source_manifest_path"]).is_file()

    frame = tmp_path / "zero.jpg"
    Image.fromarray(np.zeros((3, 3, 3), dtype=np.uint8)).save(frame)
    rows = [
        {"reference_id": "target", "frame_id": "target", "frame_uri": str(frame), "included_in_index": True, "t_capture_sec": 0.0, "T_world_camera": _pose(0.0), "anchor_observations": ["anchor"]},
        {"reference_id": "ref", "frame_id": "ref", "frame_uri": str(frame), "included_in_index": True, "t_capture_sec": 1.0, "T_world_camera": _pose(0.5), "anchor_observations": ["anchor"]},
    ]
    _write_dense_rows(capture_root, rows)
    entry = {
        "target_frame_id": "target",
        "target_frame_uri": str(frame),
        "target_index": 0,
        "selected_references": [{"candidate_index": 1, "pose_distance_m": 0.5, "temporal_gap_sec": 1.0}],
        "selected_reference_ids": ["ref"],
        "selected_reference_frame_ids": ["ref"],
        "selected_reference_frame_uris": [str(frame)],
        "decoupling": {"mode": "test"},
    }
    monkeypatch.setattr(cosmos_benchmark, "build_reference_selection_manifest", lambda **_kwargs: {"entries": [entry], "policy": {"target_reference_decoupling_mode": "test"}, "selected_target_count": 1, "skipped_target_count": 0, "rejected_near_duplicate_count": 0})
    monkeypatch.setattr(cosmos_benchmark, "build_legacy_reference_selection_manifest", lambda **_kwargs: {"entries": [], "policy": {}})
    monkeypatch.setattr(cosmos_benchmark, "build_reference_selection_comparison", lambda **_kwargs: {"changed_primary_reference_count": 1})
    monkeypatch.setattr(cosmos_benchmark, "build_synthetic_trajectory_manifest", lambda **_kwargs: {"policy": {}, "entries": [{"target_frame_id": "target", "trajectory_context_id": "traj", "status": "augmented", "synthetic_waypoint_count": 1, "synthetic_waypoint_ids": ["w1"]}], "augmented_target_count": 1, "skipped_sparse_context_count": 0, "synthetic_waypoint_count": 1})
    monkeypatch.setattr(cosmos_benchmark, "build_sparse_view_interpolation_manifest", lambda **_kwargs: {"policy": {}, "entries": [{"target_frame_id": "target", "interpolation_context_id": "interp", "status": "interpolated", "interpolated_view_count": 1, "interpolated_view_ids": ["i1"]}], "interpolated_target_count": 1, "skipped_sparse_target_count": 0, "interpolated_view_count": 1})
    monkeypatch.setattr(cosmos_benchmark, "build_future_anchor_regrounding_manifest", lambda **_kwargs: {"policy": {}, "entries": [{"target_frame_id": "target", "future_anchor_context_id": "future", "status": "re_grounded", "future_anchor_count": 1, "future_anchor_reference_ids": ["ref"], "future_anchor_frame_ids": ["ref"]}], "re_grounded_target_count": 1})
    monkeypatch.setattr(cosmos_benchmark, "_probe_cosmos_runtime", lambda: {"status": "blocked", "blockers": ["missing"], "packages": {}, "model_id": "test"})
    runtime_blocked = run_cosmos_zero_shot_validation_lane(
        capture_root=capture_root,
        descriptor_gcs_uri=descriptor_uri,
        cfg=cfg,
        max_examples=1,
    )
    assert runtime_blocked["status"] == "blocked"

    monkeypatch.setattr(cosmos_benchmark, "_probe_cosmos_runtime", lambda: {"status": "ready", "blockers": [], "packages": {}, "model_id": "test"})
    monkeypatch.setattr(cosmos_benchmark, "run_capture_synthesis_validation", lambda **_kwargs: {"status": "failed", "reason": "No module named cosmos"})
    synthesis_blocked = run_cosmos_zero_shot_validation_lane(
        capture_root=capture_root,
        descriptor_gcs_uri=descriptor_uri,
        cfg=cfg,
        max_examples=1,
    )
    assert synthesis_blocked["reason"] == "No module named cosmos"

    monkeypatch.setattr(cosmos_benchmark, "run_capture_synthesis_validation", lambda **_kwargs: {"status": "completed", "coverage_frac": 0.8, "ref_frame_distance_m": 1.0, "output_video_uri": "gs://bucket/out.mp4"})
    completed = run_cosmos_zero_shot_validation_lane(
        capture_root=capture_root,
        descriptor_gcs_uri=descriptor_uri,
        cfg=cfg,
        max_examples=1,
    )
    assert completed["status"] == "completed"
    assert completed["checks"]["spatial_faithfulness"]["passed"] is True
