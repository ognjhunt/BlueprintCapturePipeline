"""Real retained Atlas source path, synthetic receipts and no provider calls."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from blueprint_pipeline import atlas_source_geometry as geometry
from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_atlas_scene_source as source
from blueprint_pipeline import website_scene_handoff as handoff
from blueprint_pipeline.core.handoff_lease import HandoffLeaseError, bind_handoff_lease
from tests.test_website_atlas_scene_source import digest, job_fixture, write


def test_real_handoff_source_geometry_keeps_b_current_scale_after_a_returns(tmp_path, monkeypatch):
    capture = tmp_path / "capture"
    job_dir = job_fixture(capture)
    output = capture / "pipeline/website_scene_preparation/atlas"

    def claim(owner):
        state, lease = listener._claim_job_lease(capture, scene_id="s", capture_id="capture",
                                                owner=owner, lease_seconds=60)
        assert state == "claimed"
        return listener._JobLeaseHeartbeat(capture_root=capture, owner=owner,
                                           token=lease["lease_token"], lease_seconds=60)

    a = claim("atlas-review-a")
    newer = {}
    original_verified = geometry._verified_pose

    def verified(**kwargs):
        pose = original_verified(**kwargs)
        if not newer:
            # A has already captured the previous named reference-scale record.
            # B claims normally and the retained fixture advances to a different
            # reference estimate. This is synthetic source data, not a measured
            # calibration or a provider operation.
            newer["reclaimed"] = True
            with listener._locked_job_ledger(capture) as current:
                listener.write_json(capture / listener.JOB_LEDGER_FILENAME, {**current,
                    "lease_expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()})
            b = claim("atlas-review-b")
            measurement_path = job_dir / "pose/scene_measurements.json"
            measurement = json.loads(measurement_path.read_text())
            measurement["meters_per_unit_estimate"] *= 2
            write(measurement_path, digest(measurement))
            result_path = job_dir / "result.json"
            result = json.loads(result_path.read_text())
            result["source_pose"]["scene_measurements_digest"] = measurement["digest"]
            write(result_path, digest(result))
            job_path = job_dir / "preview_job.json"
            job = json.loads(job_path.read_text())
            write(job_path, {**job, "result_digest": result["digest"]})
            with bind_handoff_lease(b.checkpoint, b.effect):
                current_source = source.atlas_scene_source(capture_root=capture, output_root=output)
            assert current_source["status"] == "ready"
            manifest = Path(current_source["source_geometry"]["manifest_path"])
            newer.update(manifest=manifest, manifest_bytes=manifest.read_bytes(),
                         tree={path: path.read_bytes() for path in manifest.parent.rglob("*")
                               if path.is_file()},
                         ledger=(capture / listener.JOB_LEDGER_FILENAME).read_bytes())
        return pose

    monkeypatch.setattr(geometry, "_verified_pose", verified)
    descriptor = {"capture_id": "capture", "scene_id": "s", "metadata": {
        "site_task_context": {"capture_id": "capture", "scene_id": "s"}}}
    with bind_handoff_lease(a.checkpoint, a.effect):
        with pytest.raises(HandoffLeaseError):
            handoff.prepare_website_scene_handoff(descriptor=descriptor,
                clean_plate={"status": "noop", "privacy_verified": True},
                capture_root=capture, now=1)
    assert (capture / listener.JOB_LEDGER_FILENAME).read_bytes() == newer["ledger"]
    assert newer["manifest"].read_bytes() == newer["manifest_bytes"], (
        "A replaced B's selected Atlas source geometry before the outer handoff refused A")
    assert all(path.read_bytes() == payload for path, payload in newer["tree"].items())
