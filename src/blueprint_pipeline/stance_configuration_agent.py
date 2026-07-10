"""Bounded feedback-driven stance-configuration search for manipulation task stances.

This module is the deterministic "scene-configuration agent" that closes the loop
between stance sweeps and their structured rejection feedback:

1. The caller resolves the exact task object and affordance (unchanged).
2. The caller's placement solver (``plan_task_stance``) produces candidate poses and
   validates each with the live-stage collision probe, floor-support/facing geometry
   gate, and the approximate reach estimate.
3. This agent consumes the *quantitative* failure record of a blocked sweep — e.g.
   "seed effector 0.148 m beyond the reach limit at standoff 0.38 m" or "clearance
   0.02 m short of the required gap" — and derives the next search parameters:
   new standoff distances inside the evidence-derived feasible window, a refined
   approach-angle fan, or an approach-mode switch
   (``direct_manipulation`` -> ``navigation_staging`` -> ``final_manipulation_stance``).
4. It stops only when a sweep fully PASSES the caller's own gates, or emits a proved
   infeasibility (under the declared monotonic ray model) / an explicit
   ``budget_exhausted`` after bounded rounds. ``budget_exhausted`` is fail-closed and
   is never an infeasibility proof.

Design constraints (deliberate):

- The geometry feedback loop is ordinary, tested, deterministic code — reproducible
  and cheap to run locally. There is no randomness and no wall-clock dependence.
- The agent NEVER invents coordinates, alters acceptance thresholds, or declares
  success. Every proposal is re-validated by the caller's unchanged sweep + gates;
  the only authority this module has is over which *registered* search strategy to
  try next.
- An external chooser (an LLM, for example) may be plugged in through
  ``strategy_chooser``, but it can only pick among the strategy ids the geometry
  code has already computed as eligible. Any other output is recorded as a
  violation and overridden by the deterministic default choice.

The module is stdlib-only and importable both as ``blueprint_pipeline.
stance_configuration_agent`` (repo/tests) and as a flat ``stance_configuration_agent``
module from the GPU worker bundle directory.

Robot-agnostic by construction: the agent never encodes an embodiment. Reach
limits, margins, and measured distances are read from the sweep's own per-candidate
records (populated from the active ``RobotProfile`` — Unitree G1 is only the
default reference embodiment), and footprint/standoff hard bounds arrive through
``StanceSearchLimits`` from the caller. A humanoid, a mobile manipulator, or a
tabletop arm like a Panda differ only in the numbers those records carry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

STANCE_SEARCH_SCHEMA_VERSION = "stance_configuration_search.v1"

APPROACH_MODE_DIRECT = "direct_manipulation"
APPROACH_MODE_STAGING = "navigation_staging"
APPROACH_MODE_FINAL = "final_manipulation_stance"

STRATEGY_PROBE_FEASIBLE_WINDOW = "probe_feasible_window"
STRATEGY_DESCEND_TO_REACH = "descend_to_reach"
STRATEGY_BACK_OFF_TO_CLEARANCE = "back_off_to_clearance"
STRATEGY_REFINE_ANGLE_FAN = "refine_angle_fan"
STRATEGY_RESTAGE_APPROACH_ANCHOR = "restage_approach_anchor"

REGISTERED_STRATEGY_IDS = (
    STRATEGY_PROBE_FEASIBLE_WINDOW,
    STRATEGY_DESCEND_TO_REACH,
    STRATEGY_BACK_OFF_TO_CLEARANCE,
    STRATEGY_REFINE_ANGLE_FAN,
    STRATEGY_RESTAGE_APPROACH_ANCHOR,
)

# The interval model assumes reach shortfall shrinks and clearance grows monotonically
# with standoff along a fixed approach ray. Real kitchens violate this occasionally
# (an island behind the robot, say); model errors only cost search rounds, never
# correctness, because every proposal is re-validated by the caller's gates.
INTERVAL_MODEL_ASSUMPTION = (
    "Reach shortfall decreases and obstacle clearance increases monotonically with "
    "standoff along each fixed approach ray. Bounds below are derived from measured "
    "candidate rejections under that model; every proposed pose was still "
    "re-validated by the unchanged sweep gates."
)

_STANDOFF_EPSILON_M = 0.01
_STANDOFF_DEDUPE_TOLERANCE_M = 0.015
_COLLISION_BACKOFF_STEP_M = 0.05
_MAX_PROPOSED_DISTANCES_PER_ROUND = 4
_MAX_PROPOSED_ANGLES_PER_ROUND = 6


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_xyz(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        coords = [_finite(v) for v in list(value)[:3]]
        if all(v is not None for v in coords):
            return (coords[0], coords[1], coords[2])  # type: ignore[return-value]
    return None


@dataclass(frozen=True)
class StanceSweepParameters:
    """One bounded sweep request. The caller maps this to its own tested geometry."""

    approach_mode: str
    standoff_candidates_m: tuple[float, ...]
    angle_offsets_deg: tuple[float, ...] | None = None  # None -> caller default fan
    approach_anchor_xyz: tuple[float, float, float] | None = None
    approach_anchor_source: str | None = None

    def signature(self) -> tuple:
        anchor = (
            tuple(round(v, 3) for v in self.approach_anchor_xyz)
            if self.approach_anchor_xyz is not None
            else None
        )
        angles = (
            tuple(round(float(a), 1) for a in self.angle_offsets_deg)
            if self.angle_offsets_deg is not None
            else None
        )
        return (
            self.approach_mode,
            tuple(round(float(d), 3) for d in self.standoff_candidates_m),
            angles,
            anchor,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "approach_mode": self.approach_mode,
            "standoff_candidates_m": [round(float(d), 4) for d in self.standoff_candidates_m],
            "angle_offsets_deg": (
                [round(float(a), 2) for a in self.angle_offsets_deg]
                if self.angle_offsets_deg is not None
                else None
            ),
            "approach_anchor_xyz": (
                [round(float(v), 4) for v in self.approach_anchor_xyz]
                if self.approach_anchor_xyz is not None
                else None
            ),
            "approach_anchor_source": self.approach_anchor_source,
        }


@dataclass
class StanceSearchLimits:
    """Hard bounds for the search. The agent never proposes outside these."""

    min_standoff_m: float
    max_standoff_m: float
    max_rounds: int = 8
    max_staging_anchors: int = 2

    def clamp(self, standoff_m: float) -> float:
        return min(self.max_standoff_m, max(self.min_standoff_m, float(standoff_m)))


@dataclass
class DirectionEvidence:
    """Quantitative rejection evidence for one approach direction (angle offset)."""

    angle_offset_deg: float
    tried_standoffs_m: list[float] = field(default_factory=list)
    # Feasible-window bounds under the monotonic ray model.
    lower_bound_m: float = 0.0
    upper_bound_m: float = math.inf
    lower_bound_evidence: list[dict[str, Any]] = field(default_factory=list)
    upper_bound_evidence: list[dict[str, Any]] = field(default_factory=list)
    collision_failures: int = 0
    placement_failures: int = 0
    reach_failures: int = 0
    min_reach_shortfall_m: float | None = None
    min_clearance_shortfall_m: float | None = None

    def window(self) -> tuple[float, float]:
        return (self.lower_bound_m, self.upper_bound_m)

    def window_is_empty(self) -> bool:
        return self.lower_bound_m >= self.upper_bound_m - _STANDOFF_EPSILON_M

    def to_json(self) -> dict[str, Any]:
        return {
            "angle_offset_deg": round(float(self.angle_offset_deg), 2),
            "tried_standoffs_m": [round(v, 4) for v in sorted(self.tried_standoffs_m)],
            "lower_bound_m": round(float(self.lower_bound_m), 4),
            "upper_bound_m": (
                round(float(self.upper_bound_m), 4)
                if math.isfinite(self.upper_bound_m)
                else None
            ),
            "window_is_empty": self.window_is_empty(),
            "collision_failures": self.collision_failures,
            "placement_failures": self.placement_failures,
            "reach_failures": self.reach_failures,
            "min_reach_shortfall_m": (
                round(self.min_reach_shortfall_m, 4)
                if self.min_reach_shortfall_m is not None
                else None
            ),
            "min_clearance_shortfall_m": (
                round(self.min_clearance_shortfall_m, 4)
                if self.min_clearance_shortfall_m is not None
                else None
            ),
            "lower_bound_evidence": self.lower_bound_evidence[-4:],
            "upper_bound_evidence": self.upper_bound_evidence[-4:],
        }


def _reach_shortfall_m(estimate: Mapping[str, Any]) -> float | None:
    """Signed reach shortfall in meters (>0 means the affordance is out of reach).

    Uses the same fields the sweep's reachability gate records: nearest seed effector
    and shoulder distances against their limits plus the pre-selection margins.
    """
    shortfalls: list[float] = []
    effector = _finite(estimate.get("nearest_seed_effector_to_affordance_m"))
    effector_limit = _finite(estimate.get("required_max_seed_effector_to_affordance_m"))
    effector_margin = _finite(estimate.get("approx_preselection_effector_margin_m")) or 0.0
    if effector is not None and effector_limit is not None:
        shortfalls.append(effector - (effector_limit + effector_margin))
    shoulder = _finite(estimate.get("nearest_shoulder_to_affordance_m"))
    shoulder_limit = _finite(estimate.get("required_max_shoulder_to_affordance_m"))
    shoulder_margin = _finite(estimate.get("approx_preselection_shoulder_margin_m")) or 0.0
    if shoulder is not None and shoulder_limit is not None:
        shortfalls.append(shoulder - (shoulder_limit + shoulder_margin))
    if not shortfalls:
        return None
    return max(shortfalls)


def _placement_clearance_shortfall_m(validation: Mapping[str, Any]) -> float | None:
    """Best-effort quantitative gap shortfall from a placement-validation verdict."""
    required = _finite(validation.get("required_target_gap_m"))
    geometry = validation.get("deterministic_geometry")
    measured: float | None = None
    if isinstance(geometry, Mapping):
        measured = _finite(geometry.get("min_obstacle_clearance_m"))
        if required is None:
            required = _finite(geometry.get("required_clearance_m"))
    if required is not None and measured is not None and measured < required:
        return required - measured
    return None


def _placement_validation_failed(validation: Any) -> bool:
    if not isinstance(validation, Mapping):
        return False
    status = str(validation.get("status") or "").strip().upper()
    return status not in {"PASS", "PASSED", "OK"}


def analyze_stance_plan_feedback(
    plan: Mapping[str, Any],
    *,
    limits: StanceSearchLimits,
    previous: Mapping[float, DirectionEvidence] | None = None,
) -> dict[float, DirectionEvidence]:
    """Fold a sweep's per-candidate rejection records into per-direction evidence.

    Evidence accumulates across rounds so the feasible window only tightens; a
    later round never forgets an earlier measured rejection.
    """
    evidence: dict[float, DirectionEvidence] = dict(previous or {})
    candidates = plan.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return evidence
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        angle = _finite(candidate.get("angle_offset_deg"))
        standoff = _finite(candidate.get("standoff_from_target_surface_m"))
        if angle is None or standoff is None:
            continue
        record = evidence.setdefault(angle, DirectionEvidence(angle_offset_deg=angle))
        record.tried_standoffs_m.append(standoff)
        contact_count = int(_finite(candidate.get("scene_collision_contact_count")) or 0)
        validation = candidate.get("placement_validation")
        reach = candidate.get("reachability_estimate")
        if contact_count > 0:
            record.collision_failures += 1
            bound = standoff + _COLLISION_BACKOFF_STEP_M
            if bound > record.lower_bound_m:
                record.lower_bound_m = bound
                record.lower_bound_evidence.append(
                    {
                        "kind": "collision",
                        "standoff_m": round(standoff, 4),
                        "scene_collision_contact_count": contact_count,
                        "derived_lower_bound_m": round(bound, 4),
                    }
                )
            continue
        if _placement_validation_failed(validation):
            record.placement_failures += 1
            shortfall = _placement_clearance_shortfall_m(validation)
            if shortfall is not None:
                record.min_clearance_shortfall_m = (
                    shortfall
                    if record.min_clearance_shortfall_m is None
                    else min(record.min_clearance_shortfall_m, shortfall)
                )
                bound = standoff + shortfall + _STANDOFF_EPSILON_M
            else:
                bound = standoff + _COLLISION_BACKOFF_STEP_M
            if bound > record.lower_bound_m:
                record.lower_bound_m = bound
                record.lower_bound_evidence.append(
                    {
                        "kind": "placement",
                        "standoff_m": round(standoff, 4),
                        "blockers": [
                            str(b) for b in (validation.get("blockers") or []) if b
                        ][:4]
                        if isinstance(validation, Mapping)
                        else [],
                        "clearance_shortfall_m": (
                            round(shortfall, 4) if shortfall is not None else None
                        ),
                        "derived_lower_bound_m": round(bound, 4),
                    }
                )
            continue
        if isinstance(reach, Mapping) and str(reach.get("status") or "").upper() != "PASS":
            record.reach_failures += 1
            shortfall = _reach_shortfall_m(reach)
            if shortfall is not None and shortfall > 0.0:
                record.min_reach_shortfall_m = (
                    shortfall
                    if record.min_reach_shortfall_m is None
                    else min(record.min_reach_shortfall_m, shortfall)
                )
                bound = standoff - shortfall
                if bound < record.upper_bound_m:
                    record.upper_bound_m = bound
                    record.upper_bound_evidence.append(
                        {
                            "kind": "reach",
                            "standoff_m": round(standoff, 4),
                            "reach_shortfall_m": round(shortfall, 4),
                            "derived_upper_bound_m": round(bound, 4),
                        }
                    )
    # Clamp windows to the hard limits so the proof reflects the searchable space.
    for record in evidence.values():
        record.lower_bound_m = max(record.lower_bound_m, limits.min_standoff_m)
        record.upper_bound_m = min(record.upper_bound_m, limits.max_standoff_m)
    return evidence


def _untried_window_standoffs(
    record: DirectionEvidence, limits: StanceSearchLimits
) -> list[float]:
    """Standoffs inside a direction's feasible window not yet tried anywhere near."""
    lo, hi = record.window()
    if record.window_is_empty():
        return []
    lo = max(lo, limits.min_standoff_m)
    hi = min(hi, limits.max_standoff_m)
    if hi - lo < _STANDOFF_EPSILON_M:
        return []
    raw = [
        lo + _STANDOFF_EPSILON_M,
        0.5 * (lo + hi),
        hi - _STANDOFF_EPSILON_M,
        0.25 * lo + 0.75 * hi,
    ]
    tried = sorted(record.tried_standoffs_m)
    proposals: list[float] = []
    for value in raw:
        value = limits.clamp(value)
        if any(abs(value - t) <= _STANDOFF_DEDUPE_TOLERANCE_M for t in tried):
            continue
        if any(abs(value - p) <= _STANDOFF_DEDUPE_TOLERANCE_M for p in proposals):
            continue
        proposals.append(round(value, 4))
    return proposals[:_MAX_PROPOSED_DISTANCES_PER_ROUND]


def _direction_priority(record: DirectionEvidence) -> tuple[float, float]:
    """Search directions with the smallest known shortfalls first."""
    reach = record.min_reach_shortfall_m if record.min_reach_shortfall_m is not None else math.inf
    return (reach, abs(record.angle_offset_deg))


@dataclass(frozen=True)
class EligibleStrategy:
    strategy_id: str
    parameters: StanceSweepParameters
    rationale: str

    def to_json(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "rationale": self.rationale,
            "parameters": self.parameters.to_json(),
        }


def _refined_angle_fan(
    evidence: Mapping[float, DirectionEvidence],
    tried_fans: Sequence[tuple[float, ...]],
) -> tuple[float, ...] | None:
    """Midpoint angles between the two most promising adjacent tried directions."""
    if not evidence:
        return None
    ranked = sorted(evidence.values(), key=_direction_priority)
    tried_angles = sorted({float(r.angle_offset_deg) for r in evidence.values()})
    if len(tried_angles) < 2:
        return None
    already: set[float] = set()
    for fan in tried_fans:
        already.update(round(float(a), 1) for a in fan)
    proposals: list[float] = []
    for best in ranked[:3]:
        angle = float(best.angle_offset_deg)
        index = tried_angles.index(angle)
        for neighbor_index in (index - 1, index + 1):
            if 0 <= neighbor_index < len(tried_angles):
                midpoint = 0.5 * (angle + tried_angles[neighbor_index])
                key = round(midpoint, 1)
                if key in already or any(abs(key - p) < 1.0 for p in proposals):
                    continue
                proposals.append(key)
    if not proposals:
        return None
    return tuple(proposals[:_MAX_PROPOSED_ANGLES_PER_ROUND])


def compute_eligible_strategies(
    *,
    evidence: Mapping[float, DirectionEvidence],
    limits: StanceSearchLimits,
    approach_mode: str,
    tried_signatures: set[tuple],
    tried_angle_fans: Sequence[tuple[float, ...]],
    staging_anchors_remaining: Sequence[Mapping[str, Any]],
    default_standoffs_m: Sequence[float],
) -> list[EligibleStrategy]:
    """Deterministically derive every strategy the evidence currently supports.

    Order encodes the default priority: exploit a known feasible window first,
    then move toward reach, then back off from collisions, then widen the angle
    fan, and only then re-stage the approach from a navigable anchor.
    """
    eligible: list[EligibleStrategy] = []
    ranked = sorted(evidence.values(), key=_direction_priority)

    def _add(strategy_id: str, parameters: StanceSweepParameters, rationale: str) -> None:
        if parameters.signature() in tried_signatures:
            return
        if not parameters.standoff_candidates_m:
            return
        eligible.append(
            EligibleStrategy(
                strategy_id=strategy_id, parameters=parameters, rationale=rationale
            )
        )

    for record in ranked:
        window_standoffs = _untried_window_standoffs(record, limits)
        if not window_standoffs:
            continue
        lo, hi = record.window()
        has_reach_bound = math.isfinite(hi) and hi < limits.max_standoff_m
        has_collision_bound = lo > limits.min_standoff_m + _STANDOFF_EPSILON_M
        if has_reach_bound and has_collision_bound:
            strategy_id = STRATEGY_PROBE_FEASIBLE_WINDOW
            rationale = (
                f"direction {record.angle_offset_deg:+.1f} deg has a nonempty measured "
                f"window [{lo:.3f}, {hi:.3f}] m not yet sampled"
            )
        elif has_reach_bound:
            strategy_id = STRATEGY_DESCEND_TO_REACH
            rationale = (
                f"direction {record.angle_offset_deg:+.1f} deg reach-failed by "
                f"{record.min_reach_shortfall_m if record.min_reach_shortfall_m is not None else float('nan'):.3f} m; "
                f"descending below the tried ladder toward {hi:.3f} m"
            )
        elif has_collision_bound:
            strategy_id = STRATEGY_BACK_OFF_TO_CLEARANCE
            rationale = (
                f"direction {record.angle_offset_deg:+.1f} deg collided/failed placement up to "
                f"{lo:.3f} m; backing off to untried larger standoffs"
            )
        else:
            strategy_id = STRATEGY_PROBE_FEASIBLE_WINDOW
            rationale = (
                f"direction {record.angle_offset_deg:+.1f} deg has untried standoffs inside "
                "the hard search bounds"
            )
        _add(
            strategy_id,
            StanceSweepParameters(
                approach_mode=approach_mode,
                standoff_candidates_m=tuple(window_standoffs),
                angle_offsets_deg=(record.angle_offset_deg,),
            ),
            rationale,
        )

    fan = _refined_angle_fan(evidence, tried_angle_fans)
    if fan:
        base = [
            limits.clamp(d) for d in default_standoffs_m[:_MAX_PROPOSED_DISTANCES_PER_ROUND]
        ]
        deduped: list[float] = []
        for value in base:
            if not any(abs(value - p) <= _STANDOFF_DEDUPE_TOLERANCE_M for p in deduped):
                deduped.append(round(value, 4))
        _add(
            STRATEGY_REFINE_ANGLE_FAN,
            StanceSweepParameters(
                approach_mode=approach_mode,
                standoff_candidates_m=tuple(deduped),
                angle_offsets_deg=fan,
            ),
            "refining the approach-angle fan between the most promising tried directions",
        )

    for anchor in staging_anchors_remaining[: limits.max_staging_anchors]:
        anchor_xyz = _optional_xyz(anchor.get("xyz"))
        if anchor_xyz is None:
            continue
        base = [
            limits.clamp(d) for d in default_standoffs_m[:_MAX_PROPOSED_DISTANCES_PER_ROUND]
        ]
        deduped = []
        for value in base:
            if not any(abs(value - p) <= _STANDOFF_DEDUPE_TOLERANCE_M for p in deduped):
                deduped.append(round(value, 4))
        _add(
            STRATEGY_RESTAGE_APPROACH_ANCHOR,
            StanceSweepParameters(
                approach_mode=APPROACH_MODE_FINAL,
                standoff_candidates_m=tuple(deduped),
                angle_offsets_deg=None,
                approach_anchor_xyz=anchor_xyz,
                approach_anchor_source=str(anchor.get("source") or "staging_anchor"),
            ),
            "re-anchoring the approach from a navigable staging point and re-running "
            "the final manipulation stance search",
        )
    return eligible


def deterministic_strategy_chooser(
    eligible: Sequence[EligibleStrategy], context: Mapping[str, Any]
) -> str:
    """Default chooser: first eligible strategy in registered priority order."""
    del context
    return eligible[0].strategy_id


def _summarize_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    candidates = plan.get("candidates")
    candidate_count = (
        len(candidates)
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes))
        else 0
    )
    return {
        "status": str(plan.get("status") or ""),
        "blockers": [str(b) for b in (plan.get("blockers") or []) if b],
        "candidate_count": candidate_count,
        "accepted_candidate_count": plan.get("accepted_candidate_count"),
        "placement_validation_rejected_candidate_count": plan.get(
            "placement_validation_rejected_candidate_count"
        ),
        "reachability_rejected_candidate_count": plan.get(
            "reachability_rejected_candidate_count"
        ),
    }


def _plan_accepted(plan: Mapping[str, Any]) -> bool:
    return str(plan.get("status") or "").strip().lower() == "accepted"


def run_stance_configuration_search(
    *,
    attempt_sweep: Callable[[StanceSweepParameters], Mapping[str, Any]],
    initial_plan: Mapping[str, Any],
    initial_parameters: StanceSweepParameters,
    limits: StanceSearchLimits,
    staging_anchor_candidates: Sequence[Mapping[str, Any]] = (),
    strategy_chooser: Callable[[Sequence[EligibleStrategy], Mapping[str, Any]], str]
    | None = None,
    chooser_kind: str = "deterministic",
) -> dict[str, Any]:
    """Run the bounded feedback loop over a caller-supplied stance sweep.

    ``attempt_sweep`` must run the caller's full, unchanged candidate validation and
    return a stance-plan mapping (``status``, ``blockers``, ``candidates`` with the
    quantitative rejection fields). ``initial_plan`` is the already-run round-0
    sweep so evidence is never recomputed from scratch.
    """
    chooser = strategy_chooser or deterministic_strategy_chooser
    if strategy_chooser is None:
        chooser_kind = "deterministic"

    rounds: list[dict[str, Any]] = []
    chooser_violations: list[dict[str, Any]] = []
    tried_signatures: set[tuple] = {initial_parameters.signature()}
    tried_angle_fans: list[tuple[float, ...]] = []
    if initial_parameters.angle_offsets_deg is not None:
        tried_angle_fans.append(tuple(initial_parameters.angle_offsets_deg))
    used_anchor_keys: set[tuple] = set()
    approach_mode = initial_parameters.approach_mode
    evidence = analyze_stance_plan_feedback(initial_plan, limits=limits)
    rounds.append(
        {
            "round_index": 0,
            "approach_mode": approach_mode,
            "strategy_id": "initial_structured_sweep",
            "parameters": initial_parameters.to_json(),
            "plan_summary": _summarize_plan(initial_plan),
        }
    )
    if _plan_accepted(initial_plan):
        return _search_result(
            status="accepted",
            accepted_plan=dict(initial_plan),
            accepted_parameters=initial_parameters,
            rounds=rounds,
            evidence=evidence,
            limits=limits,
            chooser_kind=chooser_kind,
            chooser_violations=chooser_violations,
            staging_anchor_candidates=staging_anchor_candidates,
        )

    default_standoffs = tuple(initial_parameters.standoff_candidates_m) or (
        limits.clamp(0.5 * (limits.min_standoff_m + limits.max_standoff_m)),
    )

    for round_index in range(1, max(1, int(limits.max_rounds)) + 1):
        anchors_remaining = [
            anchor
            for anchor in staging_anchor_candidates
            if _anchor_key(anchor) not in used_anchor_keys
        ]
        eligible = compute_eligible_strategies(
            evidence=evidence,
            limits=limits,
            approach_mode=approach_mode,
            tried_signatures=tried_signatures,
            tried_angle_fans=tried_angle_fans,
            staging_anchors_remaining=anchors_remaining,
            default_standoffs_m=default_standoffs,
        )
        if not eligible:
            # Infeasibility may only be claimed when the measured evidence is
            # conclusive: every explored direction's feasible window is empty and
            # no staging anchor remains. Anything else (e.g. a sweep that stopped
            # returning candidate records) is a fail-closed stall, not a proof.
            windows_conclusive = bool(evidence) and all(
                record.window_is_empty() for record in evidence.values()
            )
            return _search_result(
                status="infeasible" if windows_conclusive else "search_stalled",
                accepted_plan=None,
                accepted_parameters=None,
                rounds=rounds,
                evidence=evidence,
                limits=limits,
                chooser_kind=chooser_kind,
                chooser_violations=chooser_violations,
                staging_anchor_candidates=staging_anchor_candidates,
            )
        context = {
            "round_index": round_index,
            "approach_mode": approach_mode,
            "eligible": [item.to_json() for item in eligible],
            "direction_evidence": [record.to_json() for record in evidence.values()],
        }
        try:
            chosen_id = str(chooser(eligible, context))
        except Exception as exc:  # noqa: BLE001 - fail closed to the deterministic choice
            chooser_violations.append(
                {
                    "round_index": round_index,
                    "kind": "chooser_error",
                    "error": repr(exc),
                }
            )
            chosen_id = eligible[0].strategy_id
        chosen = next((item for item in eligible if item.strategy_id == chosen_id), None)
        if chosen is None:
            chooser_violations.append(
                {
                    "round_index": round_index,
                    "kind": "chooser_returned_unregistered_or_ineligible_strategy",
                    "returned": chosen_id,
                    "eligible": [item.strategy_id for item in eligible],
                }
            )
            chosen = eligible[0]
        parameters = chosen.parameters
        tried_signatures.add(parameters.signature())
        if parameters.angle_offsets_deg is not None:
            tried_angle_fans.append(tuple(parameters.angle_offsets_deg))
        if chosen.strategy_id == STRATEGY_RESTAGE_APPROACH_ANCHOR:
            for anchor in anchors_remaining:
                if _optional_xyz(anchor.get("xyz")) == parameters.approach_anchor_xyz:
                    used_anchor_keys.add(_anchor_key(anchor))
                    break
            approach_mode = APPROACH_MODE_FINAL
            # A fresh approach ray invalidates per-direction windows measured on the
            # old ray; keep the old evidence only in the recorded rounds.
            evidence = {}
        try:
            plan = attempt_sweep(parameters)
        except Exception as exc:  # noqa: BLE001 - a broken sweep must not fake progress
            rounds.append(
                {
                    "round_index": round_index,
                    "approach_mode": approach_mode,
                    "strategy_id": chosen.strategy_id,
                    "rationale": chosen.rationale,
                    "parameters": parameters.to_json(),
                    "plan_summary": {"status": "error", "error": repr(exc)},
                }
            )
            return _search_result(
                status="error",
                accepted_plan=None,
                accepted_parameters=None,
                rounds=rounds,
                evidence=evidence,
                limits=limits,
                chooser_kind=chooser_kind,
                chooser_violations=chooser_violations,
                staging_anchor_candidates=staging_anchor_candidates,
                error=repr(exc),
            )
        if not isinstance(plan, Mapping):
            plan = {"status": "blocked", "blockers": ["attempt_sweep_returned_non_mapping"]}
        rounds.append(
            {
                "round_index": round_index,
                "approach_mode": approach_mode,
                "strategy_id": chosen.strategy_id,
                "rationale": chosen.rationale,
                "parameters": parameters.to_json(),
                "plan_summary": _summarize_plan(plan),
            }
        )
        if _plan_accepted(plan):
            return _search_result(
                status="accepted",
                accepted_plan=dict(plan),
                accepted_parameters=parameters,
                rounds=rounds,
                evidence=evidence,
                limits=limits,
                chooser_kind=chooser_kind,
                chooser_violations=chooser_violations,
                staging_anchor_candidates=staging_anchor_candidates,
            )
        evidence = analyze_stance_plan_feedback(plan, limits=limits, previous=evidence)

    return _search_result(
        status="budget_exhausted",
        accepted_plan=None,
        accepted_parameters=None,
        rounds=rounds,
        evidence=evidence,
        limits=limits,
        chooser_kind=chooser_kind,
        chooser_violations=chooser_violations,
        staging_anchor_candidates=staging_anchor_candidates,
    )


def _anchor_key(anchor: Mapping[str, Any]) -> tuple:
    xyz = _optional_xyz(anchor.get("xyz"))
    return (
        tuple(round(v, 3) for v in xyz) if xyz is not None else None,
        str(anchor.get("source") or ""),
    )


def _search_result(
    *,
    status: str,
    accepted_plan: dict[str, Any] | None,
    accepted_parameters: StanceSweepParameters | None,
    rounds: list[dict[str, Any]],
    evidence: Mapping[float, DirectionEvidence],
    limits: StanceSearchLimits,
    chooser_kind: str,
    chooser_violations: list[dict[str, Any]],
    staging_anchor_candidates: Sequence[Mapping[str, Any]],
    error: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": STANCE_SEARCH_SCHEMA_VERSION,
        "status": status,
        "round_count": len(rounds),
        "rounds": rounds,
        "limits": {
            "min_standoff_m": round(float(limits.min_standoff_m), 4),
            "max_standoff_m": round(float(limits.max_standoff_m), 4),
            "max_rounds": int(limits.max_rounds),
            "max_staging_anchors": int(limits.max_staging_anchors),
        },
        "chooser_kind": chooser_kind,
        "chooser_violations": chooser_violations,
        "registered_strategy_ids": list(REGISTERED_STRATEGY_IDS),
        "staging_anchor_candidate_count": len(list(staging_anchor_candidates)),
        "claim_boundary": (
            "The stance search only re-parameterizes the caller's unchanged sweep. "
            "Acceptance authority remains the sweep's collision, placement, and reach "
            "gates plus the downstream rendered gates. A budget_exhausted status is a "
            "fail-closed stop, not an infeasibility proof."
        ),
    }
    if error is not None:
        result["error"] = error
    if accepted_plan is not None:
        result["accepted_plan"] = accepted_plan
        result["accepted_plan_summary"] = _summarize_plan(accepted_plan)
    if accepted_parameters is not None:
        result["accepted_parameters"] = accepted_parameters.to_json()
    if status == "infeasible":
        result["infeasibility_proof"] = {
            "assumptions": INTERVAL_MODEL_ASSUMPTION,
            "per_direction": [record.to_json() for record in evidence.values()],
            "all_direction_windows_empty_or_exhausted": True,
            "staging_anchors_exhausted": True,
        }
    else:
        result["direction_evidence"] = [record.to_json() for record in evidence.values()]
    return result
