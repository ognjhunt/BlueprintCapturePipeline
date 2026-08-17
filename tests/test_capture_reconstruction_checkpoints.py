"""Checkpoint/resume contract, with emphasis on never paying twice."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from blueprint_pipeline.capture_reconstruction_checkpoints import (
    CHECKPOINT_STAGES,
    PAID_STAGES,
    CaptureReconstructionCheckpointError,
    assert_paid_stage_not_repeated,
    next_stage,
    read_checkpoints,
    record_checkpoint,
)


def _sha(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


CAPTURE = _sha("capture")


def test_stage_order_matches_the_real_journey() -> None:
    assert CHECKPOINT_STAGES[0] == "upload_received"
    assert CHECKPOINT_STAGES[-1] == "terminal"
    assert "downstream_dispatched" in CHECKPOINT_STAGES
    assert PAID_STAGES <= set(CHECKPOINT_STAGES)


def test_recording_a_stage_twice_with_identical_evidence_is_a_noop(
    tmp_path: Path,
) -> None:
    first = record_checkpoint(
        state_root=tmp_path, capture_digest=CAPTURE, stage="queued", evidence={"n": 1}
    )
    second = record_checkpoint(
        state_root=tmp_path, capture_digest=CAPTURE, stage="queued", evidence={"n": 1}
    )
    assert first["already_recorded"] is False
    assert second["already_recorded"] is True
    assert second["checkpoint_digest"] == first["checkpoint_digest"]


def test_recording_different_evidence_for_a_passed_stage_conflicts(
    tmp_path: Path,
) -> None:
    """Inputs changing under a run that already spent something is a defect."""
    record_checkpoint(
        state_root=tmp_path, capture_digest=CAPTURE, stage="training", evidence={"n": 1}
    )
    with pytest.raises(CaptureReconstructionCheckpointError) as excinfo:
        record_checkpoint(
            state_root=tmp_path,
            capture_digest=CAPTURE,
            stage="training",
            evidence={"n": 2},
        )
    assert "checkpoint_conflict" in str(excinfo.value)


def test_unknown_stage_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CaptureReconstructionCheckpointError) as excinfo:
        record_checkpoint(
            state_root=tmp_path,
            capture_digest=CAPTURE,
            stage="vibes",
            evidence={},
        )
    assert "stage_unknown" in str(excinfo.value)


def test_resume_starts_at_the_first_missing_stage(tmp_path: Path) -> None:
    assert next_stage(state_root=tmp_path, capture_digest=CAPTURE) == "upload_received"
    record_checkpoint(
        state_root=tmp_path,
        capture_digest=CAPTURE,
        stage="upload_received",
        evidence={},
    )
    assert next_stage(state_root=tmp_path, capture_digest=CAPTURE) == "intake_validated"


def test_resume_does_not_skip_a_gap(tmp_path: Path) -> None:
    """A later stage recorded without its predecessor must surface, not pass."""
    record_checkpoint(
        state_root=tmp_path, capture_digest=CAPTURE, stage="training", evidence={}
    )
    assert next_stage(state_root=tmp_path, capture_digest=CAPTURE) == "upload_received"


def test_terminal_capture_has_no_next_stage(tmp_path: Path) -> None:
    for stage in CHECKPOINT_STAGES:
        record_checkpoint(
            state_root=tmp_path, capture_digest=CAPTURE, stage=stage, evidence={}
        )
    assert next_stage(state_root=tmp_path, capture_digest=CAPTURE) is None
    assert read_checkpoints(state_root=tmp_path, capture_digest=CAPTURE)["terminal"]


# --------------------------------------------------------------------------
# Never pay twice
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stage", sorted(PAID_STAGES))
def test_a_completed_paid_stage_cannot_be_repeated(tmp_path: Path, stage: str) -> None:
    record_checkpoint(
        state_root=tmp_path, capture_digest=CAPTURE, stage=stage, evidence={"arm": "postshot-primary"}
    )
    with pytest.raises(CaptureReconstructionCheckpointError) as excinfo:
        assert_paid_stage_not_repeated(
            state_root=tmp_path, capture_digest=CAPTURE, stage=stage
        )
    assert "paid_stage_already_completed" in str(excinfo.value)


def test_an_unstarted_paid_stage_is_allowed(tmp_path: Path) -> None:
    assert_paid_stage_not_repeated(
        state_root=tmp_path, capture_digest=CAPTURE, stage="training"
    )


def test_paid_guard_is_per_capture_not_global(tmp_path: Path) -> None:
    """A second capture must not be blocked by the first capture's spend."""
    record_checkpoint(
        state_root=tmp_path, capture_digest=CAPTURE, stage="training", evidence={}
    )
    assert_paid_stage_not_repeated(
        state_root=tmp_path, capture_digest=_sha("other-capture"), stage="training"
    )


def test_unpaid_stages_are_replayable(tmp_path: Path) -> None:
    record_checkpoint(
        state_root=tmp_path, capture_digest=CAPTURE, stage="publish", evidence={}
    )
    assert_paid_stage_not_repeated(
        state_root=tmp_path, capture_digest=CAPTURE, stage="publish"
    )


def test_ledger_reports_which_paid_stages_were_completed(tmp_path: Path) -> None:
    for stage in ("upload_received", "worker_allocated", "training"):
        record_checkpoint(
            state_root=tmp_path, capture_digest=CAPTURE, stage=stage, evidence={}
        )
    ledger = read_checkpoints(state_root=tmp_path, capture_digest=CAPTURE)
    assert ledger["paid_stages_completed"] == ["worker_allocated", "training"]
    assert ledger["terminal"] is False
