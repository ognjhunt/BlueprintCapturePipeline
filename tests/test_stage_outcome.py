from __future__ import annotations

import pytest

from blueprint_pipeline.core.stage_outcome import (
    OutcomeKind,
    StageOutcome,
    stage_ledger_outcome_kind,
)
from blueprint_pipeline.stage_outcome import StageOutcome as CompatibilityStageOutcome


def test_stage_outcome_compatibility_import_is_canonical_type() -> None:
    assert CompatibilityStageOutcome is StageOutcome


def test_stage_outcome_distinguishes_absence_from_failure() -> None:
    absent = StageOutcome(
        kind=OutcomeKind.NOT_REQUESTED,
        reason="optional trust layer not requested",
    )
    failed = StageOutcome(kind=OutcomeKind.FAILED, reason="provider timed out")

    assert absent.to_mapping()["kind"] == "not_requested"
    assert failed.to_mapping()["kind"] == "failed"


def test_produced_outcome_requires_artifact() -> None:
    with pytest.raises(ValueError, match="requires_artifact"):
        StageOutcome(kind=OutcomeKind.PRODUCED)


def test_run_ledger_status_mapping_is_explicit() -> None:
    assert (
        stage_ledger_outcome_kind(
            status="skipped",
            detail="optional_trust_layer_not_requested",
        )
        is OutcomeKind.NOT_REQUESTED
    )
    assert stage_ledger_outcome_kind(status="failed") is OutcomeKind.FAILED
    assert stage_ledger_outcome_kind(status="completed") is OutcomeKind.PRODUCED
