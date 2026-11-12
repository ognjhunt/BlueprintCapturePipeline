"""When a near-tie between registration poses is still a safe guess, and the refusal that records it.

Moved from ``website_task_preparation`` (2026-11-12, file budget). On the unanchored search a close
runner-up keeps the best pose as a recorded, flagged guess (owner direction 2026-11-11); one that puts the
scene somewhere else is refused with its evidence.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class RegistrationRefusal(ValueError):
    """A registration refusal that carries the scores it was decided on."""

    def __init__(self, code: str, evidence: Mapping[str, Any]):
        super().__init__(code)
        self.evidence = dict(evidence)


#: A runner-up pose scoring within this ratio of the best makes the registration a recorded best guess.
REGISTRATION_DECISIVE_RATIO = 1.2
REGISTRATION_AMBIGUOUS_FLAG = "scene_estimate:registration_ambiguous"


#: A runner-up this close to the best pose puts the scene in nearly the same place: the guess is safe.
CLOSE_POSE_DEGREES, CLOSE_POSE_METRES, CLOSE_POSE_SCALE = 10.0, 0.10, 1.1


def _close_runner_up(best, runner_up, ratio, metres_per_runtime_unit, *, roll=None, runner_up_roll=None) -> None:
    """Refuse, with the evidence kept, a near-tie whose runner-up is materially another pose."""
    if ratio >= REGISTRATION_DECISIVE_RATIO:
        return
    from .website_task_preparation import _rotation_degrees

    degrees = _rotation_degrees(runner_up[2], best[2])
    metres = float(np.linalg.norm(np.asarray(runner_up[3]) - np.asarray(best[3]))) * metres_per_runtime_unit
    scale = float(runner_up[1] / best[1])
    if degrees <= CLOSE_POSE_DEGREES and metres <= CLOSE_POSE_METRES and 1 / CLOSE_POSE_SCALE <= scale <= CLOSE_POSE_SCALE:
        return
    raise RegistrationRefusal("website_registration_ambiguous", {
        "runner_up_ratio": float(ratio), "decisive_ratio": REGISTRATION_DECISIVE_RATIO,
        "best": {"score_m": float(best[0] * metres_per_runtime_unit), **({"roll_degrees": roll} if roll is not None else {})},
        "runner_up": {"score_m": float(runner_up[0] * metres_per_runtime_unit),
                      **({"roll_degrees": runner_up_roll} if runner_up_roll is not None else {}),
                      "rotation_from_best_degrees": degrees, "translation_from_best_m": metres,
                      "scale_ratio_to_best": scale},
        "basis": "runner_up_is_materially_another_pose"})
