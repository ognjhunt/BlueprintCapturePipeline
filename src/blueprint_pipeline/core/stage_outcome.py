"""Shared result semantics for produced, absent, blocked, and failed artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class OutcomeKind(str, Enum):
    PRODUCED = "produced"
    NOT_REQUESTED = "not_requested"
    UNAVAILABLE = "unavailable"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True)
class StageOutcome:
    kind: OutcomeKind
    reason: str | None = None
    artifact: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.kind is OutcomeKind.PRODUCED and self.artifact is None:
            raise ValueError("produced_stage_outcome_requires_artifact")
        if self.kind is not OutcomeKind.PRODUCED and not (self.reason or "").strip():
            raise ValueError(f"{self.kind.value}_stage_outcome_requires_reason")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "stage_outcome.v1",
            "kind": self.kind.value,
            "reason": self.reason,
            "artifact": dict(self.artifact) if self.artifact is not None else None,
        }


def stage_ledger_outcome_kind(*, status: str, detail: str | None = None) -> OutcomeKind:
    if status == "completed":
        return OutcomeKind.PRODUCED
    if status == "failed":
        return OutcomeKind.FAILED
    if status == "blocked":
        return OutcomeKind.BLOCKED
    if status == "skipped" and detail in {
        "not_requested",
        "optional_trust_layer_not_requested",
    }:
        return OutcomeKind.NOT_REQUESTED
    return OutcomeKind.UNAVAILABLE
