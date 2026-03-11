"""Tests for lane-aware capture orchestration helpers."""

from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.capture_orchestrator import resolve_requested_lanes


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
