"""Each new removal-model request stops; exact uploaded-file cleanup remains available."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from blueprint_pipeline import clean_plate_removal_analysis_gemini as removal
from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline.core.handoff_lease import HandoffLeaseError, bind_handoff_lease
from tests.test_handoff_lease_fencing import claim


def _takeover(root, lease):
    listener._finish_job_lease(root, owner="a", token=lease["lease_token"], update={"status": "failed_retryable"})
    return claim(root, "b")


def _types():
    return SimpleNamespace(HttpOptions=lambda **_: None, HttpRetryOptions=lambda **_: None)


def test_validation_repair_refuses_new_model_request_after_reclaim(tmp_path):
    lease = claim(tmp_path, "a")
    heartbeat = listener._JobLeaseHeartbeat(capture_root=tmp_path, owner="a", token=lease["lease_token"], lease_seconds=60)
    _takeover(tmp_path, lease)
    sent = []

    def create(**_kwargs):
        sent.append("repair")
        return SimpleNamespace(status="completed", steps=[])

    sdk = SimpleNamespace(Client=lambda **_: SimpleNamespace(interactions=SimpleNamespace(create=create)))
    with bind_handoff_lease(heartbeat.checkpoint, heartbeat.effect):
        with pytest.raises(HandoffLeaseError, match="lost"):
            removal._invoke_text_repair(api_key="offline-fixture", model="existing-model", genai=sdk,
                types=_types(), prompt="retained prompt", previous_text="retained response", error="schema")
    assert sent == []


def test_reclaim_during_video_upload_stops_model_but_cleans_only_owned_upload(tmp_path, monkeypatch):
    lease = claim(tmp_path, "a")
    heartbeat = listener._JobLeaseHeartbeat(capture_root=tmp_path, owner="a", token=lease["lease_token"], lease_seconds=60)
    sent, deleted = [], []
    video = tmp_path / "video.mp4"
    video.write_bytes(b"offline-media-encoding-double")

    @contextmanager
    def audio_copy(path):
        yield path, {"audio_present": False}

    def upload(**_kwargs):
        sent.append("upload")
        _takeover(tmp_path, lease)
        return SimpleNamespace(name="files/owned-by-this-call", uri="fixture://owned-video", state=SimpleNamespace(name="ACTIVE"))

    def create(**_kwargs):
        sent.append("model")
        return SimpleNamespace(status="completed", steps=[], usage=None)

    client = SimpleNamespace(files=SimpleNamespace(upload=upload, delete=lambda *, name: deleted.append(name)),
                             interactions=SimpleNamespace(create=create))
    monkeypatch.setattr(removal, "audio_free_upload_copy", audio_copy)
    with bind_handoff_lease(heartbeat.checkpoint, heartbeat.effect):
        with pytest.raises(HandoffLeaseError, match="lost"):
            removal._invoke_agentic_video(api_key="offline-fixture", model="existing-model", processing="static",
                video_path=video, genai=SimpleNamespace(Client=lambda **_: client), types=_types())
    assert sent == ["upload"]
    assert deleted == ["files/owned-by-this-call"]
    assert listener._read_job_ledger(tmp_path)["lease_owner"] == "b"
