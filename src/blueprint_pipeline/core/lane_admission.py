"""Typed admission policy for deprecated capture-pipeline lanes."""

from __future__ import annotations

from collections.abc import Sequence

from .common import PipelineError


LEGACY_CAPTURE_LANES = frozenset(
    {
        "scene_memory",
        "retrieval_index",
        "frame_alignment",
        "synthesis_coverage_validation",
        "cosmos_single_capture_smoke",
    }
)


def require_legacy_lane_admission(
    lanes: Sequence[str], *, allow_legacy_lanes: bool
) -> None:
    selected = [lane for lane in lanes if lane in LEGACY_CAPTURE_LANES]
    if selected and not allow_legacy_lanes:
        raise PipelineError(
            "Legacy capture lanes require explicit allow_legacy_lanes=True: "
            + ",".join(selected)
        )
