"""The tail of the chain: publish -> WebApp status -> downstream -> terminal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.capture_reconstruction_checkpoints import read_checkpoints
from blueprint_pipeline.capture_reconstruction_launch_dispatcher import (
    complete_capture_reconstruction,
)
from blueprint_pipeline.capture_reconstruction_status_sync import (
    RECONSTRUCTION_FIELD,
    CaptureReconstructionStatusError,
)


def _sha(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


CAPTURE = _sha("capture")
PLY = _sha("ply")


class _Store:
    def __init__(self) -> None:
        self.doc: dict | None = None
        self.writes: list = []

    def write(self, capture_id, payload):
        self.writes.append((capture_id, dict(payload)))
        self.doc = dict(payload[RECONSTRUCTION_FIELD])

    def read(self, _capture_id):
        return self.doc


def _campaign(tmp_path: Path, *, with_artifacts: bool = True, name: str = "campaign") -> Path:
    arms = (
        [
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
        ]
        if with_artifacts
        else []
    )
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "campaign_digest": _sha("campaign"),
                "primary_arm_id": "postshot-primary",
                "arms": arms,
            }
        ),
        encoding="utf-8",
    )
    return path


def _complete(tmp_path: Path, store: _Store, **overrides):
    kwargs = dict(
        state_root=tmp_path / "checkpoints",
        capture_id="capture-live-001",
        capture_digest=CAPTURE,
        campaign_path=_campaign(tmp_path),
        completed_at="2026-08-17T12:00:00Z",
        status_writer=store.write,
        status_reader=store.read,
    )
    kwargs.update(overrides)
    return complete_capture_reconstruction(**kwargs)


def test_finished_campaign_reaches_terminal_through_every_stage(
    tmp_path: Path,
) -> None:
    store = _Store()
    dispatched: list = []
    result = _complete(
        tmp_path,
        store,
        downstream_dispatch=lambda *, status: dispatched.append(status) or {"ok": True},
    )

    assert result["terminal_state"] == "published"
    assert result["status_written"] is True
    assert result["downstream_dispatched"] is True
    assert len(dispatched) == 1

    ledger = read_checkpoints(state_root=tmp_path / "checkpoints", capture_digest=CAPTURE)
    assert ledger["recorded_stages"] == [
        "export",
        "publish",
        "downstream_dispatched",
        "terminal",
    ]
    assert ledger["terminal"] is True


def test_webapp_receives_the_digest_bound_artifacts(tmp_path: Path) -> None:
    store = _Store()
    _complete(tmp_path, store)
    _, payload = store.writes[0]
    published = payload[RECONSTRUCTION_FIELD]
    assert published["artifacts"][0]["digest"] == PLY
    assert published["capture_digest"] == CAPTURE
    assert published["appearance_fidelity_qualified"] is False


def test_rerunning_after_a_crash_does_not_restate_the_capture(tmp_path: Path) -> None:
    store = _Store()
    first = _complete(tmp_path, store)
    second = _complete(tmp_path, store)
    assert first["status_written"] is True
    assert second["status_written"] is False
    assert len(store.writes) == 1


def test_downstream_is_not_triggered_for_a_failed_campaign(tmp_path: Path) -> None:
    store = _Store()
    dispatched: list = []
    result = _complete(
        tmp_path,
        store,
        campaign_path=_campaign(tmp_path, with_artifacts=False, name="failed"),
        downstream_dispatch=lambda *, status: dispatched.append(status),
    )
    assert result["terminal_state"] == "failed"
    assert result["downstream_dispatched"] is False
    assert dispatched == []


def test_capture_still_reaches_terminal_without_a_downstream_consumer(
    tmp_path: Path,
) -> None:
    """Downstream analysis needs dynamics geometry reconstruction cannot supply."""
    store = _Store()
    result = _complete(tmp_path, store, downstream_dispatch=None)
    assert result["terminal_state"] == "published"
    assert result["downstream_dispatched"] is False
    ledger = read_checkpoints(state_root=tmp_path / "checkpoints", capture_digest=CAPTURE)
    assert ledger["terminal"] is True


def test_a_second_different_campaign_cannot_overwrite_a_published_capture(
    tmp_path: Path,
) -> None:
    store = _Store()
    _complete(tmp_path, store)
    other = tmp_path / "other.json"
    other.write_text(
        json.dumps(
            {
                "campaign_digest": _sha("different-campaign"),
                "primary_arm_id": "postshot-primary",
                "arms": [
                    {
                        "arm_id": "postshot-primary",
                        "artifacts": [
                            {
                                "artifact_id": "standard_3dgs_ply",
                                "digest": _sha("different-ply"),
                                "uri": "gs://d/other.ply",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    # The checkpoint ledger catches this before the status sync even runs:
    # a capture whose export stage already recorded different artifact digests
    # cannot be re-exported. That is the stronger of the two guards.
    with pytest.raises(Exception) as excinfo:
        _complete(tmp_path, store, campaign_path=other)
    message = str(excinfo.value)
    assert "checkpoint_conflict" in message or "terminal_conflict" in message
    assert len(store.writes) == 1
