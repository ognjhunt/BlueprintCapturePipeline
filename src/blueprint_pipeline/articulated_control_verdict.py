"""The judged contract for an articulated control pair, shared by both workers.

This lives here rather than inside a worker because it took three corrections
to get right and there are now two runtimes that need it. A contract duplicated
across two scripts diverges the first time only one of them is edited, and the
divergence stays invisible until two runs disagree about the same trajectory.

Each part is deliberately not the obvious thing.

Reaching the window is about the angle the door attains, not the angle it was
released at: the schedule lets go early on purpose and lets the door coast in,
so testing the release angle contradicts the design.

Holding is neither "ended up inside" - a door still travelling when the clock
stops lands there by accident - nor "never left", since the settle window opens
at release, below the window, and the door legitimately spends its first
moments on the way in. It is: entered, stayed inside from that point, motion
shrinking, finished inside.

The gasket is a torque applied per step because USD cannot express it. A drive
is a spring or a damper and both grow with displacement; a seal is strongest at
closed and gone a few degrees later.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


CONTROL_VERDICT_SCHEMA_VERSION = "articulated_control_verdict.v1"


def downsample_trace(trace: Sequence[float], *, limit: int) -> list[float]:
    """Keep the shape of a motion without keeping every step of it.

    Start, release and settle are three numbers; they cannot say whether the
    door crept open, slammed and bounced, or oscillated through the seal, and
    that is what separates a good positive from a lucky one. The extremes are
    kept explicitly - a transient overshoot that decays away is the entire
    story of a bad release, and it is exactly what uniform sampling drops.
    """

    values = [float(value) for value in trace]
    if len(values) <= limit:
        return values
    step = (len(values) - 1) / float(limit - 1)
    kept = {0, len(values) - 1, values.index(max(values)), values.index(min(values))}
    kept.update(int(round(index * step)) for index in range(limit))
    return [values[index] for index in sorted(kept) if index < len(values)]


def seal_detent_torque(angle_degrees: float, peak: float, width: float) -> float:
    if peak <= 0.0 or width <= 0.0:
        return 0.0
    magnitude = abs(angle_degrees)
    if magnitude >= width:
        return 0.0
    taper = 0.5 * (1.0 + math.cos(math.pi * magnitude / width))
    resistance = peak * taper
    return resistance if angle_degrees >= 0.0 else -resistance


def evaluate_positive_control(
    *,
    positive: Mapping[str, Any],
    window: Sequence[float],
    hold_tolerance_degrees: float,
    tail_fraction: float = 0.25,
) -> dict[str, Any]:
    """Judge the positive on where the door got to and whether it stopped.

    Two corrections over the obvious reading. Reaching the window is about the
    angle the door attains, not the angle it was released at - the schedule
    lets go early on purpose and lets the door coast in, so testing the release
    angle contradicts the design and can only pass when the coast model is
    wrong.

    And holding is not the same as ending up somewhere. A door still swinging
    when the clock runs out lands in the window by accident; one that came to
    rest is holding. Only the tail of the settle window separates them, so a
    run with no trace reports that it cannot tell rather than assuming.
    """

    low, high = float(window[0]), float(window[1])
    maximum = float(positive.get("maximum_angle_degrees") or 0.0)
    settled = float(positive.get("settled_angle_degrees") or 0.0)
    settle = [float(v) for v in (positive.get("settle_trace_degrees") or [])]

    # Judged on the settle window alone. A tail taken from the whole episode
    # still contains the coast, where the door is supposed to be moving, so
    # measuring across it reads deceleration as a failure to hold.
    entered: bool | None = None
    stayed_after_entry: bool | None = None
    decaying: bool | None = None
    tail_motion: float | None = None
    if len(settle) >= 4:
        # The settle window opens at release, and release is deliberately below
        # the window - the door spends its first moments coasting in. Demanding
        # every sample be inside fails the design, not the door. What holding
        # means is that once it arrives, it stays.
        entry = next(
            (i for i, value in enumerate(settle) if low <= value <= high), None
        )
        entered = entry is not None
        if entry is not None:
            after = settle[entry:]
            stayed_after_entry = all(low <= value <= high for value in after)
            half = max(1, len(after) // 2)
            early = max(after[:half]) - min(after[:half])
            late = max(after[half:]) - min(after[half:]) if after[half:] else 0.0
            # Asymptotic settling never reaches exactly zero, so what matters is
            # that the motion is shrinking, not that it has stopped.
            decaying = late <= early
            tail_motion = late

    return {
        "reaches_success_window": {
            "maximum_angle_degrees": maximum,
            "window": [low, high],
            "passed": low <= maximum <= high,
        },
        "holds_after_release": {
            "settled_angle_degrees": settled,
            "entered_window": entered,
            "stayed_inside_after_entry": stayed_after_entry,
            "motion_is_decaying": decaying,
            "tail_motion_degrees": tail_motion,
            "hold_tolerance_degrees": float(hold_tolerance_degrees),
            "passed": bool(
                entered
                and stayed_after_entry
                and decaying
                and low <= settled <= high
            ),
        },
    }




__all__ = [
    "CONTROL_VERDICT_SCHEMA_VERSION",
    "downsample_trace",
    "evaluate_positive_control",
    "seal_detent_torque",
]
