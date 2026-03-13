from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.capture_orchestrator import PipelineConfig, run_capture_pipeline


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
