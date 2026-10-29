"""Actual media-stage shared pointers after source analysis loses its lease."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from blueprint_pipeline import clean_plate_stage as media
from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_capture_preparation as preparation
from blueprint_pipeline.clean_plate_removal_analysis_gemini import empty_removal_plan
from blueprint_pipeline.core.handoff_lease import HandoffLeaseError, bind_handoff_lease
from tests.test_clean_plate_stage import _make_capture
from tests.test_qualification_coverage_edges import _descriptor


def test_actual_media_stage_cannot_replace_b_current_pointer_after_reclaim(tmp_path, monkeypatch):
    root = _make_capture(tmp_path)
    pipeline = root / "pipeline"
    descriptor = _descriptor(scene_id="s1", capture_id="c1", requested_outputs=["preview_simulation"],
        metadata={"site_task_context": {"capture_id": "c1", "scene_id": "s1", "purpose": "scene_preparation"}})

    def claim(owner):
        status, value = listener._claim_job_lease(root, scene_id="s1", capture_id="c1",
            owner=owner, lease_seconds=60)
        assert status == "claimed"
        return value

    lease = claim("review-a")
    heartbeat = listener._JobLeaseHeartbeat(capture_root=root, owner="review-a",
        token=lease["lease_token"], lease_seconds=60)
    shared = pipeline / media.CLEAN_PLATE_DIRNAME
    pointer = shared / media.CURRENT_POINTER_FILENAME
    ledger_path = root / listener.JOB_LEDGER_FILENAME
    newer = {}

    def retained_source_analysis(**_kwargs):
        with listener._locked_job_ledger(root) as ledger:
            listener.write_json(ledger_path, {**ledger,
                "lease_expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()})
        claim("review-b")
        shared.mkdir(parents=True, exist_ok=True)
        pointer.write_bytes(b'{"owner":"review-b","current":true}\n')
        newer.update(pointer=pointer.read_bytes(), ledger=ledger_path.read_bytes())
        return empty_removal_plan(status="blocked", model="fixture", processing="agentic",
            blockers=["retained_analysis_pending"])

    monkeypatch.setattr(media, "analyze_removal_targets", retained_source_analysis)
    with bind_handoff_lease(heartbeat.checkpoint, heartbeat.effect):
        with pytest.raises(HandoffLeaseError, match="lost"):
            preparation.run_website_capture_preparation(
                descriptor=descriptor, capture_root=root, pipeline_dir=pipeline,
                source_video=root / "raw/walkthrough.mp4", derived_scene_generation_allowed=True,
                gates=[], prepare_media=media.run_clean_plate_stage,
                check_privacy=lambda **_kwargs: SimpleNamespace(passed=True),
                make_gate=lambda *_args: SimpleNamespace(passed=True),
            )
    assert ledger_path.read_bytes() == newer["ledger"]
    assert pointer.read_bytes() == newer["pointer"], "stale A replaced B's actual media current pointer"
