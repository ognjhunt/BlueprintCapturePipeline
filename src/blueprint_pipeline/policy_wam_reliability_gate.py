"""Adapters that bind the reusable rollout-reliability gate to WAM loops."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
    assess_frame_sequence_reliability,
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


@dataclass(frozen=True)
class MultiViewFrameSequenceReliabilityGate:
    """Apply the same frozen gate independently to every registered WAM view."""

    thresholds: ReliabilityThresholds
    view_order: tuple[str, ...]
    assessor: Callable[..., RolloutReliabilityReport] = assess_frame_sequence_reliability
    gate_id: str = "multi_view_frame_sequence_rollout_reliability_v1"

    def __post_init__(self) -> None:
        if not self.view_order or len(set(self.view_order)) != len(self.view_order):
            raise ValueError("reliability_view_order_invalid")

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
        sequences = wam_prediction.get("generated_view_frame_sequences")
        current_views = (
            prepared_transition.get("wam_request", {}).get("current_views")
            if isinstance(prepared_transition.get("wam_request"), Mapping)
            else None
        )
        if not isinstance(sequences, Mapping) or set(sequences) != set(self.view_order):
            raise ValueError("wam_prediction_registered_view_sequences_missing")
        if not isinstance(current_views, Mapping) or set(current_views) != set(self.view_order):
            raise ValueError("prepared_transition_registered_current_views_missing")
        actions_value = prepared_transition.get("reliability_actions_10d")
        actions = np.asarray(actions_value, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 10 or not np.isfinite(actions).all():
            raise ValueError("prepared_transition_reliability_actions_10d_invalid")

        reports: dict[str, Any] = {}
        reasons: list[str] = []
        for view_id in self.view_order:
            generated = sequences[view_id]
            current = current_views[view_id]
            if not isinstance(generated, Sequence) or isinstance(
                generated, (str, bytes, bytearray)
            ):
                raise ValueError(f"wam_prediction_view_sequence_invalid:{view_id}")
            if not isinstance(current, Mapping) or not current.get("path"):
                raise ValueError(f"prepared_transition_current_view_invalid:{view_id}")
            frame_paths = [str(current["path"]), *(str(path) for path in generated)]
            if len(frame_paths) != actions.shape[0] + 1:
                raise ValueError(f"wam_prediction_view_action_frame_count_mismatch:{view_id}")
            report = self.assessor(
                frame_paths,
                actions,
                self.thresholds,
                timing_flag_scope=TIMING_SCOPE_SESSION,
            )
            reports[view_id] = report.as_dict()
            reasons.extend(f"{view_id}:{flag}" for flag in report.flags)
        return {
            "status": "passed" if not reasons else "failed",
            "abstain": bool(reasons),
            "reasons": reasons,
            "view_order": list(self.view_order),
            "per_view_reports": reports,
            "all_registered_views_scored_separately": True,
            "current_to_first_generated_transition_included": True,
            "thresholds": asdict(self.thresholds),
            "thresholds_sha256": self.thresholds_sha256,
            "timing_flag_scope": TIMING_SCOPE_SESSION,
            "session_timing_aggregation_still_required": True,
            "claim_boundary": (
                "necessary per-view rollout reliability only; not sufficient proof of "
                "causal action following or policy rank fidelity"
            ),
        }


__all__ = [
    "FrozenMaximumHorizonTerminalCriterion",
    "MultiViewFrameSequenceReliabilityGate",
    "Tier1VideoReliabilityGate",
]
