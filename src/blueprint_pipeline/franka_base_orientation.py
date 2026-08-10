"""Resolve a task-neutral Franka base yaw from phase target geometry.

Base collision search produces an XY position, but a fixed arm also needs an
authored orientation.  Reusing the canned-beverage yaw for another task can
silently put the full manipulation sweep behind the robot.  This resolver
chooses the center of the smallest circular bearing interval containing every
preregistered phase target.  Native IK remains a separate required gate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "franka_base_orientation_resolution.v1"
PROGRAM_ID = "arm-decision-proof-v1"


class FrankaBaseOrientationError(ValueError):
    """Stable geometry-resolution failures before native IK."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _point(value: Any, *, error: str, errors: list[str]) -> list[float]:
    try:
        point = [float(item) for item in value]
    except (TypeError, ValueError):
        errors.append(error)
        return []
    if len(point) != 3 or not all(math.isfinite(item) for item in point):
        errors.append(error)
        return []
    return point


def _wrap(angle: float) -> float:
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if math.isclose(wrapped, -math.pi, abs_tol=1e-15) else wrapped


def _center_smallest_bearing_interval(bearings: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(angle % (2.0 * math.pi) for angle in bearings)
    candidates: list[tuple[float, float, float]] = []
    for index, angle in enumerate(ordered):
        following = ordered[(index + 1) % len(ordered)]
        if index == len(ordered) - 1:
            following += 2.0 * math.pi
        gap = following - angle
        span = 2.0 * math.pi - gap
        start = following % (2.0 * math.pi)
        center = _wrap(start + span / 2.0)
        candidates.append((span, abs(center), center))
    span, _, center = min(candidates)
    return center, span


def resolve_franka_base_orientation(
    *,
    base_position_world_m: Sequence[float],
    phase_targets: Sequence[Mapping[str, Any]],
    source_receipt_digest: str,
    maximum_allowed_deviation_rad: float = math.pi / 2.0,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Choose one deterministic yaw that centers all preregistered phase targets."""

    errors: list[str] = []
    base = _point(
        base_position_world_m,
        error="franka_base_orientation_base_position_invalid",
        errors=errors,
    )
    if not _digest(source_receipt_digest):
        errors.append("franka_base_orientation_source_receipt_digest_invalid")
    try:
        allowed = float(maximum_allowed_deviation_rad)
    except (TypeError, ValueError):
        allowed = math.nan
    if not math.isfinite(allowed) or not 0.0 < allowed <= math.pi:
        errors.append("franka_base_orientation_deviation_limit_invalid")
    if not isinstance(phase_targets, Sequence) or isinstance(
        phase_targets, (str, bytes)
    ) or not phase_targets:
        errors.append("franka_base_orientation_phase_targets_missing")
        raw_targets: Sequence[Mapping[str, Any]] = ()
    else:
        raw_targets = phase_targets
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    bearings: list[float] = []
    for index, raw in enumerate(raw_targets):
        if not isinstance(raw, Mapping):
            errors.append(f"franka_base_orientation_phase_target_invalid:{index}")
            continue
        phase_id = str(raw.get("phase_id") or "").strip()
        if not phase_id or phase_id in seen_ids:
            errors.append(f"franka_base_orientation_phase_id_invalid:{index}")
            continue
        seen_ids.add(phase_id)
        target = _point(
            raw.get("target_position_world_m"),
            error=f"franka_base_orientation_phase_target_invalid:{phase_id}",
            errors=errors,
        )
        if len(target) != 3 or len(base) != 3:
            continue
        dx = target[0] - base[0]
        dy = target[1] - base[1]
        distance_xy = math.hypot(dx, dy)
        if distance_xy <= 1e-9:
            errors.append(f"franka_base_orientation_phase_target_at_base:{phase_id}")
            continue
        bearing = math.atan2(dy, dx)
        bearings.append(bearing)
        rows.append(
            {
                "phase_id": phase_id,
                "target_position_world_m": target,
                "distance_xy_m": distance_xy,
                "bearing_world_rad": bearing,
            }
        )
    if errors:
        raise FrankaBaseOrientationError(errors)
    yaw, span = _center_smallest_bearing_interval(bearings)
    maximum_deviation = 0.0
    for row in rows:
        deviation = _wrap(float(row["bearing_world_rad"]) - yaw)
        row["deviation_from_resolved_yaw_rad"] = deviation
        maximum_deviation = max(maximum_deviation, abs(deviation))
    blockers = []
    if maximum_deviation > allowed + 1e-12:
        blockers.append("franka_base_orientation_phase_span_exceeds_limit")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "status": "resolved_candidate" if not blockers else "blocked",
        "method": "center_of_smallest_circular_phase_bearing_interval",
        "source_receipt_digest": source_receipt_digest,
        "base_position_world_m": base,
        "phase_targets": rows,
        "bearing_interval_span_rad": span,
        "maximum_allowed_deviation_rad": allowed,
        "maximum_observed_deviation_rad": maximum_deviation,
        "resolved_yaw_world_rad": yaw,
        "resolved_orientation_xyzw": [
            0.0,
            0.0,
            math.sin(yaw / 2.0),
            math.cos(yaw / 2.0),
        ],
        "blockers": blockers,
        "claim_boundary": {
            "geometry_based_orientation_candidate_only": True,
            "native_ik_qualified": False,
            "collision_or_contact_qualified": False,
            "physical_reachability_proven": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    if destination is not None:
        write_json(Path(destination), receipt)
    return json.loads(json.dumps(receipt))


__all__ = [
    "FrankaBaseOrientationError",
    "SCHEMA_VERSION",
    "resolve_franka_base_orientation",
]
