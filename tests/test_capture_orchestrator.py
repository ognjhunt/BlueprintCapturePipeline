"""Tests for lane-aware capture orchestration helpers."""

from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.capture_orchestrator import resolve_requested_lanes, run_capture_pipeline
from blueprint_pipeline.swap_orchestrator import OrchestratorConfig


def _write_descriptor(tmp_path: Path, *, requested_lanes: list[str] | None = None) -> str:
    descriptor_path = tmp_path / "bucket/scenes/scene_a/captures/cap_a/capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = {
        "schema_version": "v1",
        "scene_id": "scene_a",
        "capture_id": "cap_a",
        "capture_source": "iphone",
        "capture_tier": "tier1_iphone",
        "raw_prefix_uri": "gs://bucket/scenes/scene_a/iphone/cap_a/raw",
        "frames_index_uri": "gs://bucket/scenes/scene_a/captures/cap_a/frames/index.jsonl",
    }
    if requested_lanes is not None:
        descriptor["requested_lanes"] = requested_lanes
    descriptor_path.write_text(json.dumps(descriptor, indent=2), encoding="utf-8")
    return "gs://bucket/scenes/scene_a/captures/cap_a/capture_descriptor.json"


def test_resolve_requested_lanes_defaults_to_qualification(tmp_path: Path) -> None:
    descriptor_uri = _write_descriptor(tmp_path)
    lanes = resolve_requested_lanes(descriptor_gcs_uri=descriptor_uri, gcs_root=tmp_path)
    assert lanes == ["qualification"]


def test_resolve_requested_lanes_uses_descriptor_request(tmp_path: Path) -> None:
    descriptor_uri = _write_descriptor(tmp_path, requested_lanes=["qualification", "advanced_geometry"])
    lanes = resolve_requested_lanes(descriptor_gcs_uri=descriptor_uri, gcs_root=tmp_path)
    assert lanes == ["qualification", "scene_memory", "advanced_geometry"]


def test_resolve_requested_lanes_descriptor_can_select_advanced_geometry_only(tmp_path: Path) -> None:
    descriptor_uri = _write_descriptor(tmp_path, requested_lanes=["advanced_geometry"])
    lanes = resolve_requested_lanes(descriptor_gcs_uri=descriptor_uri, gcs_root=tmp_path)
    assert lanes == ["scene_memory", "advanced_geometry"]


def test_resolve_requested_lanes_cli_override_wins(tmp_path: Path, monkeypatch) -> None:
    descriptor_uri = _write_descriptor(tmp_path, requested_lanes=["qualification"])
    monkeypatch.setenv("PIPELINE_LANE", "qualification")
    lanes = resolve_requested_lanes(
        descriptor_gcs_uri=descriptor_uri,
        gcs_root=tmp_path,
        lane="advanced_geometry",
    )
    assert lanes == ["scene_memory", "advanced_geometry"]


def test_resolve_requested_lanes_env_override_wins_over_descriptor(tmp_path: Path, monkeypatch) -> None:
    descriptor_uri = _write_descriptor(tmp_path, requested_lanes=["qualification"])
    monkeypatch.setenv("PIPELINE_LANE", "advanced_geometry")
    lanes = resolve_requested_lanes(descriptor_gcs_uri=descriptor_uri, gcs_root=tmp_path)
    assert lanes == ["scene_memory", "advanced_geometry"]


def test_resolve_requested_lanes_evaluation_prep_requires_qualification(tmp_path: Path) -> None:
    descriptor_uri = _write_descriptor(tmp_path, requested_lanes=["evaluation_prep"])
    lanes = resolve_requested_lanes(descriptor_gcs_uri=descriptor_uri, gcs_root=tmp_path)
    assert lanes == ["qualification", "evaluation_prep"]


def test_resolve_requested_lanes_all_includes_evaluation_prep(tmp_path: Path) -> None:
    descriptor_uri = _write_descriptor(tmp_path, requested_lanes=["all"])
    lanes = resolve_requested_lanes(descriptor_gcs_uri=descriptor_uri, gcs_root=tmp_path)
    assert lanes == ["qualification", "scene_memory", "advanced_geometry", "evaluation_prep"]


def test_run_capture_pipeline_evaluation_prep_runs_after_prerequisites(tmp_path: Path, monkeypatch) -> None:
    descriptor_uri = _write_descriptor(tmp_path, requested_lanes=["all"])
    calls: list[str] = []

    def fake_qualification(*, descriptor_gcs_uri: str, config) -> dict:
        calls.append(f"qualification:{descriptor_gcs_uri}")
        return {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene_a",
            "capture_id": "cap_a",
            "pipeline_prefix": "scenes/scene_a/captures/cap_a/pipeline",
        }

    def fake_advanced_geometry(*, descriptor_gcs_uri: str, config, nurec_client=None, blueprint_runner=None) -> dict:
        calls.append(f"advanced_geometry:{descriptor_gcs_uri}")
        return {
            "status": "completed",
            "lane": "advanced_geometry",
            "pipeline_prefix": "scenes/scene_a/captures/cap_a/pipeline",
        }

    def fake_evaluation_prep(*, capture_root, provider_name: str) -> dict:
        calls.append(f"evaluation_prep:{Path(str(capture_root)).name}:{provider_name}")
        return {"manifest_path": "/tmp/evaluation_prep_manifest.json"}

    monkeypatch.setattr("blueprint_pipeline.capture_orchestrator.run_qualification_pipeline", fake_qualification)
    monkeypatch.setattr("blueprint_pipeline.capture_orchestrator.run_swap_pipeline", fake_advanced_geometry)
    monkeypatch.setattr("blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage", fake_evaluation_prep)

    result = run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        requested_lanes=["all"],
        config=OrchestratorConfig(gcs_root=tmp_path),
    )

    assert result["lanes"] == ["qualification", "scene_memory", "advanced_geometry", "evaluation_prep"]
    assert calls == [
        f"qualification:{descriptor_uri}",
        f"advanced_geometry:{descriptor_uri}",
        "evaluation_prep:cap_a:manual",
    ]
    assert result["results"][-1]["lane"] == "evaluation_prep"
