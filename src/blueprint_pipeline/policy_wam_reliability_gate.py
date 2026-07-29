"""Adapters that bind the reusable rollout-reliability gate to WAM loops."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .policy_ranking_thesis import canonical_sha256
from .wam_rollout_reliability import (
    TIMING_SCOPE_SESSION,
    ReliabilityThresholds,
    RolloutReliabilityReport,
    assess_rollout_reliability,
)


@dataclass(frozen=True)
class Tier1VideoReliabilityGate:
    """Fail closed on static, corrupt, or semantically invalid WAM video.

    Thresholds are mandatory rather than silently defaulted because every
    scientific experiment must freeze its own values prospectively.  Timing is
    recorded per window but aggregated at the session level, matching the
    governed Phase-B interpretation of the gate.
    """

    thresholds: ReliabilityThresholds
    assessor: Callable[..., RolloutReliabilityReport] = assess_rollout_reliability
    gate_id: str = "tier1_video_rollout_reliability_session_timing_v1"

    @property
    def thresholds_sha256(self) -> str:
        return canonical_sha256(asdict(self.thresholds))

    def assess(
        self,
        *,
        previous_observation: Mapping[str, Any],
        prepared_transition: Mapping[str, Any],
        wam_prediction: Mapping[str, Any],
        query_index: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        del previous_observation, query_index, output_dir
        video_value = wam_prediction.get("generated_video_path")
        if not video_value:
            raise ValueError("wam_prediction_generated_video_path_missing")
        video_path = Path(str(video_value)).expanduser().resolve()
        if not video_path.is_file() or video_path.is_symlink():
            raise ValueError("wam_prediction_generated_video_missing_or_unsafe")
        actions_value = prepared_transition.get("reliability_actions_10d")
        if actions_value is None:
            raise ValueError("prepared_transition_reliability_actions_10d_missing")
        actions = np.asarray(actions_value, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 10 or not np.isfinite(actions).all():
            raise ValueError("prepared_transition_reliability_actions_10d_invalid")
        report = self.assessor(
            video_path,
            actions,
            self.thresholds,
            timing_flag_scope=TIMING_SCOPE_SESSION,
        )
        payload = report.as_dict()
        return {
            "status": "passed" if report.reliable else "failed",
            "abstain": not report.reliable,
            "reasons": list(report.flags),
            "report": payload,
            "thresholds": asdict(self.thresholds),
            "thresholds_sha256": self.thresholds_sha256,
            "timing_flag_scope": TIMING_SCOPE_SESSION,
            "session_timing_aggregation_still_required": True,
            "claim_boundary": (
                "necessary rollout reliability only; not sufficient proof of causal "
                "action following or policy rank fidelity"
            ),
        }


@dataclass(frozen=True)
class FrozenMaximumHorizonTerminalCriterion:
    """Use collapse/abstention or the preregistered horizon, not WAM self-grading."""

    criterion_id: str = "frozen_maximum_horizon_only_v1"

    def assess(
        self,
        *,
        observation: Mapping[str, Any],
        query_index: int,
    ) -> dict[str, Any]:
        del observation, query_index
        return {
            "terminal": False,
            "reason": None,
            "candidate_policy_graded_itself": False,
            "wam_graded_itself": False,
            "external_evaluator_in_control_loop": False,
        }


__all__ = ["FrozenMaximumHorizonTerminalCriterion", "Tier1VideoReliabilityGate"]
