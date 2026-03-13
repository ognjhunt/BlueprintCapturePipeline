from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.webapp_sync import build_webapp_pipeline_attachment_payload


def test_build_webapp_pipeline_attachment_payload_matches_shared_fixture() -> None:
    fixture_path = (
        Path(__file__).resolve().parent / "fixtures" / "pipeline_attachment_payload.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    payload = build_webapp_pipeline_attachment_payload(
        site_submission_id="req-1",
        request_id="req-1",
        scene_id="scene-1",
        capture_id="cap-1",
        pipeline_prefix="scenes/scene-1/captures/cap-1/pipeline",
        qualification_state="qualified_ready",
        opportunity_state="handoff_ready",
        authoritative_state_update=True,
        artifacts=fixture["artifacts"],
        derived_assets=fixture["derived_assets"],
    )

    assert payload == fixture
