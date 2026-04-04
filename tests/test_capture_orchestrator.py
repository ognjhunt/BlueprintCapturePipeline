from __future__ import annotations

from pathlib import Path
import json

from blueprint_pipeline.capture_orchestrator import PipelineConfig, resolve_requested_lanes, run_capture_pipeline


def test_capture_orchestrator_keeps_supported_lanes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_requested_lanes",
        lambda **_kwargs: ["qualification", "scene_memory", "evaluation_prep"],
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        lambda **_kwargs: {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "evaluation_prep_manifest.json")},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json",
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        config=PipelineConfig(gcs_root=tmp_path),
    )
    assert result["lanes"] == ["qualification", "scene_memory", "evaluation_prep"]
    assert all(item["lane"] != "advanced_geometry" for item in result["results"])


def test_capture_orchestrator_runs_single_capture_smoke_lane(monkeypatch, tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_requested_lanes",
        lambda **_kwargs: ["cosmos_single_capture_smoke"],
    )
    monkeypatch.setattr(
        "blueprint_pipeline.synthesis.cosmos_benchmark.run_cosmos_single_capture_smoke_lane",
        lambda **_kwargs: {"status": "blocked", "reason": "runtime_unavailable"},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    assert result["lanes"] == ["cosmos_single_capture_smoke"]
    assert result["results"] == [
        {
            "lane": "cosmos_single_capture_smoke",
            "status": "blocked",
            "reason": "runtime_unavailable",
        }
    ]


def test_resolve_requested_lanes_defaults_to_native_stack_for_site_world_candidate(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "capture_mode": {"resolved_mode": "site_world_candidate"},
                "scene_memory_capture": {"world_model_candidate": True},
                "requested_outputs": ["preview_simulation"],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == [
        "qualification",
        "scene_memory",
        "retrieval_index",
        "frame_alignment",
        "evaluation_prep",
    ]


def test_resolve_requested_lanes_honors_explicit_descriptor_requested_lanes(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_lanes": [
                    "qualification",
                    "retrieval_index",
                    "synthesis_coverage_validation",
                ],
                "requested_outputs": [],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == [
        "qualification",
        "retrieval_index",
        "synthesis_coverage_validation",
    ]


def test_resolve_requested_lanes_demotes_bridge_default_scene_memory_pair(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_lanes": ["qualification", "scene_memory"],
                "requested_outputs": [],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == ["qualification"]


def test_resolve_requested_lanes_prefers_explicit_descriptor_lanes_over_output_inference(
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_lanes": [
                    "qualification",
                    "synthesis_coverage_validation",
                ],
                "requested_outputs": ["preview_simulation", "deeper_evaluation"],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == [
        "qualification",
        "synthesis_coverage_validation",
    ]


def test_resolve_requested_lanes_prefers_explicit_descriptor_lanes_over_native_candidate_default(
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "capture_mode": {"resolved_mode": "site_world_candidate"},
                "scene_memory_capture": {"world_model_candidate": True},
                "requested_lanes": [
                    "qualification",
                    "synthesis_coverage_validation",
                ],
                "requested_outputs": ["preview_simulation"],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == [
        "qualification",
        "synthesis_coverage_validation",
    ]


def test_resolve_requested_lanes_accepts_camel_case_descriptor_fields(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requestedLanes": [
                    "qualification",
                    "retrieval_index",
                ],
                "requestedOutputs": ["preview_simulation"],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == [
        "qualification",
        "retrieval_index",
    ]


def test_resolve_requested_lanes_accepts_scalar_descriptor_requested_lanes(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requestedLanes": "retrieval_index",
                "requestedOutputs": ["preview_simulation"],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == [
        "qualification",
        "retrieval_index",
    ]
