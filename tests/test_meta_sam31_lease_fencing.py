"""The next provider request is a new effect even within one SAM batch."""

from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from pathlib import Path

import pytest

from blueprint_pipeline import meta_sam31 as sam
from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline.core.handoff_lease import HandoffLeaseError, bind_handoff_lease
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import build_paid_lane_admission, require_paid_resource_admission
from tests.test_meta_sam31 import inputs, response, PROMPT


def test_second_sam_prompt_cannot_start_after_real_lease_reclaim(tmp_path, monkeypatch):
    capture = tmp_path / "capture"
    capture.mkdir()

    def claim(owner):
        state, lease = listener._claim_job_lease(capture, scene_id="s", capture_id="c",
            owner=owner, lease_seconds=60)
        assert state == "claimed"
        return lease

    lease = claim("review-a")
    heartbeat = listener._JobLeaseHeartbeat(capture_root=capture, owner="review-a",
        token=lease["lease_token"], lease_seconds=60)
    monkeypatch.setenv("META_MODEL_API_KEY", "independent-fixture-token")
    # Encoding is unrelated to the per-request ownership boundary. The fake
    # provider below consumes no video; keep the exact returned bytes/hash.
    def fake_clip(*, root, artifacts, **_kwargs):
        root.mkdir(parents=True, exist_ok=True)
        (root / "frame-000000.png").write_bytes(Path(artifacts[0]["path"]).read_bytes())
        path = root / "fixture-clip.mp4"
        path.write_bytes(b"independent-encoding-double")
        return path

    monkeypatch.setattr(sam, "encode_clip", fake_clip)
    sent = []
    successor = {}

    def fake_provider(request, **_kwargs):
        sent.append(json.loads(request.data)["input"][0]["content"][0]["text"])
        if len(sent) == 1:
            with listener._locked_job_ledger(capture) as ledger:
                listener.write_json(capture / listener.JOB_LEDGER_FILENAME, {**ledger,
                    "lease_expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()})
            successor.update(claim("review-b"))
        return BytesIO(json.dumps(response()).encode())

    with bind_handoff_lease(heartbeat.checkpoint, heartbeat.effect):
        args = inputs(tmp_path)
        args["prompts"] = [{**PROMPT, "prompt_id": f"target-{i}", "text": f"object {i}"} for i in range(2)]
        digest = canonical_digest(sam.request_binding(frame_registry=args["frame_registry"],
            frame_artifacts=args["frame_artifacts"], prompts=args["prompts"]))
        args["admission"] = {**build_paid_lane_admission(resource_class="evaluator_api"),
            "allocation_binding_digest": digest, "maximum_cost_usd": 0.01,
            "external_disclosure_allowed": True}
        args["admission_grant"] = require_paid_resource_admission(args["admission"],
            resource_class="evaluator_api", expected_schema_version="paid_lane_admission.v1")
        stopped = None
        try:
            sam.run_meta_sam31(**args, opener=fake_provider)
        except HandoffLeaseError as error:
            stopped = error
    assert successor["lease_owner"] == "review-b"
    assert sent == ["object 0"], f"new provider prompt began after A lost its lease: {sent}"
    assert stopped is not None
    assert (args["output_root"] / digest[7:] / "response-0.json").is_file()


def test_cancelled_video_keeps_response_prefix_and_closes_only_owned_upload(tmp_path, monkeypatch):
    from tests.test_handoff_lease_fencing import claim

    capture = tmp_path / "capture"
    capture.mkdir()
    lease = claim(capture, "a")
    heartbeat = listener._JobLeaseHeartbeat(capture_root=capture, owner="a", token=lease["lease_token"], lease_seconds=60)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"offline-video-double")
    monkeypatch.setenv("META_MODEL_API_KEY", "offline-fixture-token")
    monkeypatch.setattr(sam, "_probe_video", lambda _path: {"frames": [{}, {}], "streams": [{"width": 4, "height": 4}]})
    rows = [{"source_frame_id": f"source-{i}", "width": 4, "height": 4} for i in range(2)]
    media = {"path": str(video), "sha256": sam._sha256_file(video),
             "source_video_digest": "sha256:" + "a" * 64, "encoding": "retained-test-encoding"}
    prompts = [{**PROMPT, "prompt_id": f"target-{i}", "text": f"object {i}"} for i in range(2)]
    digest = canonical_digest(sam.request_binding(frame_registry=rows, frame_artifacts=[], prompts=prompts, video_artifact=media))
    sent, deleted = [], []

    def provider(request, **_kwargs):
        if request.get_method() == "DELETE":
            deleted.append(request.full_url)
            return BytesIO(b'{"deleted":true}')
        if request.full_url == "https://api.meta.ai/v1/files":
            return BytesIO(b'{"id":"file-owned-fixture"}')
        assert request.full_url == sam.ENDPOINT
        sent.append(json.loads(request.data)["input"][0]["content"][0]["text"])
        listener._finish_job_lease(capture, owner="a", token=lease["lease_token"], update={"status": "failed_retryable"})
        claim(capture, "b")
        return BytesIO(json.dumps(response()).encode())

    admission = {**build_paid_lane_admission(resource_class="evaluator_api"), "allocation_binding_digest": digest,
                 "maximum_cost_usd": 0.1, "external_disclosure_allowed": True}
    output = tmp_path / "output"
    with bind_handoff_lease(heartbeat.checkpoint, heartbeat.effect):
        grant = require_paid_resource_admission(admission, resource_class="evaluator_api", expected_schema_version="paid_lane_admission.v1")
        with pytest.raises(HandoffLeaseError, match="lost"):
            sam.run_meta_sam31(frame_registry=rows, frame_artifacts=[], prompts=prompts, output_root=output,
                admission=admission, admission_grant=grant, video_artifact=media, opener=provider)
    retained = output / digest[7:]
    assert sent == ["object 0"]
    assert deleted == ["https://api.meta.ai/v1/files/file-owned-fixture"]
    assert (retained / "response-0.json").is_file()
    assert not (retained / "intent-1.json").exists()
    assert json.loads((retained / "video-cleanup.json").read_text()) == {"file_id": "file-owned-fixture", "deleted": True}
    assert listener._read_job_ledger(capture)["lease_owner"] == "b"
