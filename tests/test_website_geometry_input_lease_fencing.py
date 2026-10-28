"""A resumed source-frame producer cannot replace B's selected input binding."""

from datetime import datetime, timedelta, timezone
import json

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_scene_geometry as geometry
from blueprint_pipeline.core.handoff_lease import HandoffLeaseError, bind_handoff_lease
from tests.test_website_scene_geometry import geometry_case as geometry_case


def test_source_frame_input_publication_preserves_newer_selected_binding(geometry_case, tmp_path, monkeypatch):
    args, _ = geometry_case
    capture = tmp_path / "capture"
    capture.mkdir()
    args["output_root"] = capture / "pipeline/clean_plate/source_geometry"
    # Retain a valid completed prefix for the new worker's selected bound. A
    # missing live input manifest models a crash before its publication; source
    # candidate bytes remain immutable and available for adoption.
    selected = geometry.prepare_website_geometry_inputs(**args, maximum_frames=4)
    manifest = args["output_root"] / "worker_inputs/geometry_inputs.json"
    selected_bytes = manifest.read_bytes()
    manifest.unlink()

    def claim(owner):
        state, lease = listener._claim_job_lease(capture, scene_id="s", capture_id=args["capture_id"],
            owner=owner, lease_seconds=60)
        assert state == "claimed"
        return lease

    a = claim("review-a")
    heartbeat = listener._JobLeaseHeartbeat(capture_root=capture, owner="review-a",
        token=a["lease_token"], lease_seconds=60)
    newer = {}

    def resume_decode(_self, **kwargs):
        assert kwargs["maximum_frames"] == 2
        with listener._locked_job_ledger(capture) as current:
            listener.write_json(capture / listener.JOB_LEDGER_FILENAME, {**current,
                "lease_expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()})
        claim("review-b")
        manifest.write_bytes(selected_bytes)
        assert geometry._bound_frames(json.loads(selected_bytes), manifest.parent, geometry=False)
        newer["ledger"] = (capture / listener.JOB_LEDGER_FILENAME).read_bytes()
        return {"asset_references": {"candidate_dataset_manifest": {"relative_path": "candidate.json"},
            "decoded_observation_index": {"relative_path": "index.json"}}}

    monkeypatch.setattr(geometry.LocalDecodedObservationAdapter, "execute", resume_decode)
    with bind_handoff_lease(heartbeat.checkpoint, heartbeat.effect):
        try:
            geometry.prepare_website_source_frames(**args, maximum_frames=2)
        except HandoffLeaseError:
            pass
    assert selected["binding"]["maximum_frames"] == 4
    assert (capture / listener.JOB_LEDGER_FILENAME).read_bytes() == newer["ledger"]
    assert manifest.read_bytes() == selected_bytes, "stale A replaced B's selected source-frame input binding"
