"""Lightweight validation contract for adopted terminal native feedback."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "task_evaluation_native_construction_terminal_feedback_adoption.v1"
BASELINE_SCHEMA_VERSION = "task_evaluation_native_construction_adopted_baseline.v1"


def validate_terminal_feedback_adoption(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = json.loads(json.dumps(dict(value), allow_nan=False))
    feedback = checkpoint.get("initial_native_feedback")
    baseline = checkpoint.get("prior_attempted_baseline_binding")
    if (
        checkpoint.get("schema_version") != SCHEMA_VERSION
        or checkpoint.get("status") != "accepted_for_feedback_bootstrap"
        or checkpoint.get("feedback_bootstrap_required") is not True
        or checkpoint.get("baseline_physics_replay_required") is not False
        or checkpoint.get("native_gates_or_thresholds_modified") is not False
        or checkpoint.get("prior_attempted_candidate_digests") != []
        or not isinstance(feedback, Mapping)
        or feedback.get("passed") is not False
        or feedback.get("feedback_digest")
        != canonical_digest(feedback, digest_field="feedback_digest")
        or not isinstance(baseline, Mapping)
        or baseline.get("schema_version") != BASELINE_SCHEMA_VERSION
        or baseline.get("optuna_trial_recorded") is not False
        or baseline.get("candidate_digest") is not None
        or baseline.get("binding_digest")
        != canonical_digest(baseline, digest_field="binding_digest")
        or baseline.get("native_feedback_digest") != feedback.get("feedback_digest")
        or checkpoint.get("checkpoint_digest")
        != canonical_digest(checkpoint, digest_field="checkpoint_digest")
    ):
        raise ValueError("terminal_feedback_checkpoint_invalid")
    return checkpoint


__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "validate_terminal_feedback_adoption",
]
