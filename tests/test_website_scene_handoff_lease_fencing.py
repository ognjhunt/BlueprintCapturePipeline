"""Actual native handoff writes after a real A-to-B handoff lease reclaim."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_atlas_scene_source as atlas
from blueprint_pipeline import website_capture_preparation as preparation
from blueprint_pipeline.core.handoff_lease import HandoffLeaseError, bind_handoff_lease
from tests.test_qualification_coverage_edges import _descriptor


def test_actual_native_handoff_cannot_replace_b_receipt_after_reclaim(tmp_path, monkeypatch):
    root = tmp_path / "capture"
    pipeline = root / "pipeline"
    pipeline.mkdir(parents=True)
    descriptor = _descriptor(metadata={"site_task_context": {
        "capture_id": "capture-1", "scene_id": "scene-1", "purpose": "scene_preparation",
    }})

    def claim(owner):
        status, value = listener._claim_job_lease(root, scene_id=descriptor.scene_id,
            capture_id=descriptor.capture_id, owner=owner, lease_seconds=60)
        assert status == "claimed"
        return value

    lease = claim("review-a")
    heartbeat = listener._JobLeaseHeartbeat(capture_root=root, owner="review-a",
        token=lease["lease_token"], lease_seconds=60)
    paths = [root / "capture_descriptor.json", pipeline / "provider_run_manifest.json",
             pipeline / "website_scene_preparation/handoff.json", root / listener.JOB_LEDGER_FILENAME]
    newer = {}

    def existing_atlas_reconstruction(**_kwargs):
        with listener._locked_job_ledger(root) as ledger:
            listener.write_json(paths[-1], {**ledger,
                "lease_expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()})
        claim("review-b")
        for path in paths[:-1]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'{"owner":"review-b","current":true}\n')
        newer.update({path: path.read_bytes() for path in paths})
        return {"status": "pending", "reason": "retained_reconstruction_running"}

    monkeypatch.setattr(atlas, "atlas_scene_source", existing_atlas_reconstruction)
    with bind_handoff_lease(heartbeat.checkpoint, heartbeat.effect):
        with pytest.raises(HandoffLeaseError, match="lost"):
            preparation.run_website_capture_preparation(
                descriptor=descriptor, capture_root=root, pipeline_dir=pipeline, source_video=None,
                derived_scene_generation_allowed=True, gates=[],
                prepare_media=lambda **_kwargs: {"status": "noop", "mode": "prepared",
                    "privacy_status": "website_capture_admitted", "privacy_verified": True},
                check_privacy=lambda **_kwargs: SimpleNamespace(passed=True),
                make_gate=lambda *_args: SimpleNamespace(passed=True),
            )
    assert newer
    for path in paths:
        assert path.read_bytes() == newer[path], f"stale A replaced B's shared artifact: {path.name}"
