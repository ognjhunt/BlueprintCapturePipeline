"""Terminal status contract: what the WebApp is allowed to be told."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.capture_reconstruction_status_sync import (
    RECONSTRUCTION_FIELD,
    CaptureReconstructionStatusError,
    build_terminal_status,
    status_from_campaign,
    sync_terminal_status,
)


def _sha(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


CAPTURE = _sha("capture")
PLY = _sha("ply")


def _artifact() -> dict:
    return {
        "artifact_id": "standard_3dgs_ply",
        "digest": PLY,
        "uri": "gs://blueprint-derived/capture-live-001/postshot-primary.ply",
    }


def _published(**overrides) -> dict:
    kwargs = dict(
        capture_id="capture-live-001",
        capture_digest=CAPTURE,
        state="published",
        arm="postshot-primary",
        artifacts=[_artifact()],
        completed_at="2026-08-17T12:00:00Z",
    )
    kwargs.update(overrides)
    return build_terminal_status(**kwargs)


class _Store:
    """Stand-in for the Firestore document."""

    def __init__(self, existing: dict | None = None) -> None:
        self.doc = existing
        self.writes: list[tuple[str, dict]] = []

    def write(self, capture_id: str, payload):
        self.writes.append((capture_id, dict(payload)))
        self.doc = dict(payload[RECONSTRUCTION_FIELD])

    def read(self, _capture_id: str):
        return self.doc


# --------------------------------------------------------------------------
# A finished run is never a quality claim
# --------------------------------------------------------------------------


def test_published_status_asserts_no_quality() -> None:
    status = _published()
    assert status["state"] == "published"
    for claim in (
        "appearance_fidelity_qualified",
        "metric_accuracy_qualified",
        "collision_suitability_qualified",
        "physical_task_success_proven",
    ):
        assert status[claim] is False, claim


def test_status_digest_binds_its_own_content() -> None:
    status = _published()
    tampered = dict(status)
    tampered["state"] = "abstained"
    with pytest.raises(CaptureReconstructionStatusError) as excinfo:
        sync_terminal_status(status=tampered, writer=lambda *_: None)
    assert "digest_mismatch" in str(excinfo.value)


# --------------------------------------------------------------------------
# Artifacts must be digest-bound
# --------------------------------------------------------------------------


def test_published_requires_at_least_one_artifact() -> None:
    with pytest.raises(CaptureReconstructionStatusError) as excinfo:
        _published(artifacts=[])
    assert "published_without_artifacts" in str(excinfo.value)


def test_artifact_without_a_digest_is_refused() -> None:
    with pytest.raises(CaptureReconstructionStatusError) as excinfo:
        _published(artifacts=[{"artifact_id": "ply", "uri": "gs://x/y.ply"}])
    assert "artifact_binding_incomplete" in str(excinfo.value)


def test_artifact_without_a_uri_is_refused() -> None:
    with pytest.raises(CaptureReconstructionStatusError) as excinfo:
        _published(artifacts=[{"artifact_id": "ply", "digest": PLY}])
    assert "artifact_binding_incomplete" in str(excinfo.value)


def test_abstention_must_name_a_blocker() -> None:
    with pytest.raises(CaptureReconstructionStatusError) as excinfo:
        build_terminal_status(
            capture_id="c",
            capture_digest=CAPTURE,
            state="abstained",
            artifacts=[],
            blockers=[],
            completed_at="2026-08-17T12:00:00Z",
        )
    assert "requires_blockers" in str(excinfo.value)


def test_abstention_is_a_first_class_terminal_state() -> None:
    status = build_terminal_status(
        capture_id="capture-live-001",
        capture_digest=CAPTURE,
        state="abstained",
        artifacts=[],
        blockers=["site_task_reconstruction_policy_absent"],
        completed_at="2026-08-17T12:00:00Z",
    )
    assert status["state"] == "abstained"
    assert status["blockers"] == ["site_task_reconstruction_policy_absent"]


def test_unknown_state_is_refused() -> None:
    with pytest.raises(CaptureReconstructionStatusError) as excinfo:
        _published(state="probably_fine")
    assert "state_invalid" in str(excinfo.value)


# --------------------------------------------------------------------------
# Exactly-once sync
# --------------------------------------------------------------------------


def test_status_is_written_under_the_server_owned_field() -> None:
    store = _Store()
    sync_terminal_status(status=_published(), writer=store.write, reader=store.read)
    capture_id, payload = store.writes[0]
    assert capture_id == "capture-live-001"
    assert set(payload) == {RECONSTRUCTION_FIELD}


def test_repeat_sync_of_identical_status_is_a_noop() -> None:
    store = _Store()
    status = _published()
    first = sync_terminal_status(status=status, writer=store.write, reader=store.read)
    second = sync_terminal_status(status=status, writer=store.write, reader=store.read)
    assert first["written"] is True
    assert second["written"] is False
    assert second["already_synced"] is True
    assert len(store.writes) == 1


def test_a_published_capture_cannot_be_quietly_restated() -> None:
    store = _Store()
    sync_terminal_status(status=_published(), writer=store.write, reader=store.read)
    different = _published(
        artifacts=[{**_artifact(), "digest": _sha("a-different-ply")}]
    )
    with pytest.raises(CaptureReconstructionStatusError) as excinfo:
        sync_terminal_status(status=different, writer=store.write, reader=store.read)
    assert "terminal_conflict" in str(excinfo.value)
    assert len(store.writes) == 1


def test_first_sync_without_a_reader_still_writes() -> None:
    store = _Store()
    receipt = sync_terminal_status(status=_published(), writer=store.write)
    assert receipt["written"] is True


# --------------------------------------------------------------------------
# Derivation from a finalized campaign
# --------------------------------------------------------------------------


def test_status_derives_from_a_finalized_campaign(tmp_path: Path) -> None:
    campaign = {
        "campaign_digest": _sha("campaign"),
        "primary_arm_id": "postshot-primary",
        "arms": [
            {
                "arm_id": "postshot-primary",
                "artifacts": [
                    {
                        "artifact_id": "standard_3dgs_ply",
                        "digest": PLY,
                        "uri": "gs://d/postshot-primary.ply",
                    }
                ],
            }
        ],
    }
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")

    status = status_from_campaign(
        capture_id="capture-live-001",
        capture_digest=CAPTURE,
        campaign_path=path,
        completed_at="2026-08-17T12:00:00Z",
    )
    assert status["state"] == "published"
    assert status["arm"] == "postshot-primary"
    assert status["artifacts"][0]["digest"] == PLY
    assert status["campaign_digest"] == _sha("campaign")
    assert status["appearance_fidelity_qualified"] is False


def test_campaign_with_no_artifacts_is_reported_failed(tmp_path: Path) -> None:
    path = tmp_path / "campaign.json"
    path.write_text(
        json.dumps({"campaign_digest": _sha("c"), "arms": []}), encoding="utf-8"
    )
    status = status_from_campaign(
        capture_id="c",
        capture_digest=CAPTURE,
        campaign_path=path,
        completed_at="2026-08-17T12:00:00Z",
    )
    assert status["state"] == "failed"
    assert status["blockers"] == ["canonical_3dgs_campaign_produced_no_artifacts"]


def test_unreadable_campaign_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(CaptureReconstructionStatusError) as excinfo:
        status_from_campaign(
            capture_id="c",
            capture_digest=CAPTURE,
            campaign_path=tmp_path / "missing.json",
            completed_at="2026-08-17T12:00:00Z",
        )
    assert "campaign_unreadable" in str(excinfo.value)
