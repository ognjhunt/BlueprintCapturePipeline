"""Which registration poses are admissible, and when a near-tie among them is still a safe guess.

Moved unchanged from ``website_task_preparation`` (2026-11-12, file budget). Owner direction 2026-11-12
(capture-39ecdc97): a pose that turns the scene over is not a candidate; ambiguity is judged only among
admissible poses, and a close runner-up keeps the best pose as a recorded, flagged guess.
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


#: An alternative pose under which the capture's cameras are clearly upside down is no rival to the chosen
#: one: the median camera's image-up direction then points down, its vertical component below -0.5. A sign
#: test, never a magnitude one: a camera pitched steeply down at a table has an image-up direction near the
#: horizontal (cos of its pitch, +0.09 at 85 degrees) without being upside down, and a phone held sideways
#: reads about 0. On capture-39ecdc97 and 447 the turned-over alternative read -0.81 (review 2026-11-12).
#: Used only to drop a refusal, never to choose a pose.
UPSIDE_DOWN_MARGIN = 0.5


def _camera_ups(source_geometry: Mapping[str, Any]) -> np.ndarray | None:
    """Each source camera's image-up direction in the source world (OpenCV cameras: image up is -y)."""
    ups = [-np.asarray(frame["world_from_camera"], dtype=np.float64)[:3, 1] for frame in source_geometry.get("frames", [])
           if isinstance(frame, Mapping) and frame.get("world_from_camera") is not None]
    return np.asarray(ups) if ups else None


def _upright(rotation: np.ndarray, ups: np.ndarray | None, up: int, up_sign: int) -> float | None:
    """The median camera's up direction along the scene's up after ``rotation``; None without cameras."""
    if ups is None:
        return None
    return float(np.median((ups @ np.asarray(rotation).T)[:, up] * up_sign))


def _upside_down(rotation: np.ndarray, ups: np.ndarray | None, up: int, up_sign: int) -> bool:
    """Whether ``rotation`` puts the capture's cameras clearly upside down (``UPSIDE_DOWN_MARGIN``)."""
    value = _upright(rotation, ups, up, up_sign)
    return value is not None and value < -UPSIDE_DOWN_MARGIN


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
