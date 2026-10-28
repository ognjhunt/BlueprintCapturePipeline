"""The direct website path inherits the listener's existing lease fence."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_scene_handoff
from blueprint_pipeline import website_capture_preparation
from blueprint_pipeline.core.handoff_lease import HandoffLeaseError, bind_handoff_lease
from blueprint_pipeline.website_capture_preparation import run_website_capture_preparation
from tests.test_qualification_coverage_edges import _descriptor


@pytest.mark.parametrize("loss_at", ["before_media", "media", "privacy", "descriptor", "native"])
def test_direct_website_stale_worker_preserves_newer_outputs_and_stops_next_effect(
    tmp_path, monkeypatch, loss_at,
):
    capture = tmp_path / "capture"
    pipeline = capture / "pipeline"
    pipeline.mkdir(parents=True)
    descriptor = _descriptor(metadata={"site_task_context": {"purpose": "scene_preparation"}})

    def claim(owner):
        state, lease = listener._claim_job_lease(
            capture, scene_id=descriptor.scene_id, capture_id=descriptor.capture_id,
            owner=owner, lease_seconds=60,
        )
        assert state == "claimed"
        return lease

    lease = claim("worker-a")
    heartbeat = listener._JobLeaseHeartbeat(
        capture_root=capture, owner="worker-a", token=lease["lease_token"], lease_seconds=60,
    )
    paths = [capture / "capture_descriptor.json", pipeline / "provider_run_manifest.json",
             capture / listener.JOB_LEDGER_FILENAME]
    newer = {}

    def takeover():
        # Deterministically expire A, then use the real claim/lock path for B.
        with listener._locked_job_ledger(capture) as current:
            listener.write_json(paths[-1], {**current,
                "lease_expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()})
        claim("worker-b")
        for path in paths[:-1]:
            path.write_bytes(b'{"worker":"newer-b"}\n')
        newer.update({path: path.read_bytes() for path in paths})

    effects = []

    def prepare_media(**_):
        effects.append("media")
        if loss_at == "media":
            takeover()
        return {"status": "noop", "mode": "task_aware_prepared_images",
                "privacy_status": "website_capture_admitted", "privacy_verified": True}

    def check_privacy(**_):
        if loss_at == "privacy":
            takeover()
        return SimpleNamespace(passed=True)

    def prepare_native(**_):
        effects.append("native")
        if loss_at == "native":
            takeover()
        return {"website_intake_outbox": {"state": "forward_pending"}}

    monkeypatch.setattr(website_scene_handoff, "prepare_website_scene_handoff", prepare_native)
    if loss_at == "descriptor":
        write = website_capture_preparation.write_json

        def write_then_lose(path, value):
            write(path, value)
            if path == paths[0]:
                takeover()

        monkeypatch.setattr(website_capture_preparation, "write_json", write_then_lose)
    if loss_at == "before_media":
        takeover()

    # No new lease is bound by the helper. Carry the listener's actual checker
    # and atomic promotion scope just as the original invocation does.
    with bind_handoff_lease(heartbeat.checkpoint, heartbeat.effect):
        with pytest.raises(HandoffLeaseError):
            run_website_capture_preparation(
                descriptor=descriptor, capture_root=capture, pipeline_dir=pipeline,
                source_video=None, derived_scene_generation_allowed=True, gates=[],
                prepare_media=prepare_media, check_privacy=check_privacy,
                make_gate=lambda _name, passed, _detail: SimpleNamespace(passed=passed),
            )

    assert effects == ([] if loss_at == "before_media" else
                       ["media", "native"] if loss_at == "native" else ["media"])
    assert {path: path.read_bytes() for path in paths} == newer
