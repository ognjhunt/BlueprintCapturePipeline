from __future__ import annotations

from blueprint_pipeline.capture_bridge import CaptureDescriptor
from blueprint_pipeline.site_package_orchestrator import _worldlabs_source_candidate


def test_production_rejects_raw_worldlabs_bypass(monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_LAUNCH_PROOF_MODE", "production")
    monkeypatch.setenv("BLUEPRINT_ALLOW_RAW_WORLDLABS_BYPASS", "true")
    descriptor = CaptureDescriptor.from_dict(
        {
            "schema_version": "v1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "raw_prefix_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw",
            "raw_video_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw/walkthrough.mov",
            "frames_index_uri": "gs://bucket/scenes/scene-1/captures/capture-1/frames/index.jsonl",
            "capture_source": "iphone",
            "capture_tier": "tier1",
            "quality": {},
            "metadata": {},
        }
    )

    selection = _worldlabs_source_candidate(
        descriptor=descriptor,
        privacy_processing={"status": "not_run"},
    )

    raw_candidate = next(item for item in selection["candidates"] if item["source_id"] == "raw_video_uri")
    assert selection["raw_video_bypass_allowed"] is False
    assert raw_candidate["eligible"] is False
    assert selection["selected"] is None
