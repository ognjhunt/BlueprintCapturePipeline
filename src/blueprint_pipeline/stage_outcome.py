"""Compatibility imports for outcome semantics now canonical in :mod:`core`."""

from .core.stage_outcome import OutcomeKind, StageOutcome, stage_ledger_outcome_kind

__all__ = ["OutcomeKind", "StageOutcome", "stage_ledger_outcome_kind"]
