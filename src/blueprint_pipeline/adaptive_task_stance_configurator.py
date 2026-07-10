"""Bounded adaptive scene-configuration loop for task stance search.

This module wraps a deterministic Isaac measurement backend (spawn + settle +
measure, injected as ``measure_candidate``) in a bounded search loop.  An
optional agent hook may propose the next stance candidate, but it can never
waive, alter, or fabricate a gate: proposals carrying gate-override keys are
refused with ``stance_agent_gate_waiver_refused`` and discarded.  Acceptance is
decided exclusively by the fixed deterministic gate tuple, including fresh
robot-POV and third-person PNG render evidence from a real Isaac RGB render.
A watchdog or search-budget stop is ``blocked``, never success.  This module
performs no LLM or network calls.
"""
from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import ensure_dir, utc_now_iso, write_json


SCHEMA_VERSION = "adaptive_task_stance_configurator.v1"
RESULT_ARTIFACT_NAME = "adaptive_task_stance_configurator.json"
EVALUATIONS_ARTIFACT_NAME = "stance_candidate_evaluations.json"

STANCE_GATE_IDS: tuple[str, ...] = (
    "floor_support",
    "uprightness",
    "collision_clearance",
    "target_facing_yaw",
    "reach_envelope",
    "affordance_visibility",
    "robot_target_framing",
)
RENDER_EVIDENCE_GATE_ID = "render_evidence_fresh"
ALL_STANCE_GATE_IDS: tuple[str, ...] = STANCE_GATE_IDS + (RENDER_EVIDENCE_GATE_ID,)

REQUIRED_RENDER_SOURCE = "isaac_rtx_rgb"
MAX_CANDIDATE_RADIUS_M = 3.0
FLOOR_SUPPORT_TOLERANCE_M = 0.02
UPRIGHTNESS_TOLERANCE_DEG = 5.0
DEFAULT_YAW_TOLERANCE_RAD = 0.2
DEFAULT_MIN_AFFORDANCE_VISIBLE_FRACTION = 0.6
DEFAULT_MIN_ROBOT_IN_FRAME_FRACTION = 0.15
DEFAULT_MIN_TARGET_IN_FRAME_FRACTION = 0.15

_MIN_STANDOFF_M = 0.2
_MAX_STANDOFF_M = 2.5
_GOLDEN_ANGLE_RAD = math.pi * (3.0 - math.sqrt(5.0))

_ALLOWED_PROPOSAL_KEYS = frozenset({"pose_xyz", "yaw_rad", "standoff_m"})
_GATE_WAIVER_KEY_TOKENS = (
    "gate",
    "threshold",
    "force",
    "waive",
    "skip",
    "override",
    "accept",
    "pass",
    "status",
    "render",
    "evidence",
)

AGENT_WAIVER_BLOCKER = "stance_agent_gate_waiver_refused"
BUDGET_EXHAUSTED_BLOCKER = "stance_search_budget_exhausted"
MEASUREMENT_ERROR_BLOCKER = "stance_measurement_backend_error"

PROVIDER_ACCEPTANCE_NOTE = (
    "Local gate acceptance is scene-grounded reference evidence only. The provider must "
    "independently revalidate stance, placement, reach, and render evidence before any "
    "provider-acceptance or task-success claim (provider_revalidation_required=True)."
)

CLAIM_BOUNDARY = {
    "local_gate_acceptance_only": True,
    "provider_acceptance_proven": False,
    "task_success_proven": False,
    "agent_cannot_waive_gates": True,
}


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _pose3(value: Any) -> list[float] | None:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return None
    values = [_finite_float(item) for item in value]
    if len(values) != 3 or any(item is None for item in values):
        return None
    return [float(item) for item in values]  # type: ignore[arg-type]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _distance_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def normalize_stance_search_request(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one stance-search request. Fail-closed on any gap."""

    if not isinstance(raw, Mapping):
        raise ValueError("stance_request_must_be_a_mapping")

    def _text(name: str) -> str:
        value = str(raw.get(name) or "").strip()
        if not value:
            raise ValueError(f"stance_request_missing_{name}")
        return value

    kitchen_scene_digest = _text("kitchen_scene_digest")
    task_id = _text("task_id")
    completion_contract = raw.get("completion_contract")
    if not isinstance(completion_contract, Mapping):
        raise ValueError("stance_request_missing_completion_contract")
    target_prim_path = _text("target_prim_path")
    affordance_prim_path = _text("affordance_prim_path")
    if not affordance_prim_path.startswith(target_prim_path.rstrip("/") + "/"):
        raise ValueError("stance_request_affordance_not_descendant_of_target")

    robot_profile = raw.get("robot_profile")
    if not isinstance(robot_profile, Mapping):
        raise ValueError("stance_request_missing_robot_profile")
    collision_radius_m = _finite_float(robot_profile.get("collision_radius_m"))
    if collision_radius_m is None or collision_radius_m <= 0:
        raise ValueError("stance_request_invalid_robot_collision_radius_m")
    min_reach_m = _finite_float(robot_profile.get("min_reach_m"))
    max_reach_m = _finite_float(robot_profile.get("max_reach_m"))
    if min_reach_m is None or max_reach_m is None or min_reach_m < 0 or max_reach_m <= min_reach_m:
        raise ValueError("stance_request_invalid_robot_reach_envelope")

    reference_pose_xyz = _pose3(raw.get("reference_pose_xyz"))
    if reference_pose_xyz is None:
        raise ValueError("stance_request_invalid_reference_pose_xyz")
    reference_yaw_rad = _finite_float(raw.get("reference_yaw_rad"))
    if reference_yaw_rad is None:
        raise ValueError("stance_request_invalid_reference_yaw_rad")

    camera_limits = raw.get("camera_limits")
    if not isinstance(camera_limits, Mapping):
        raise ValueError("stance_request_missing_camera_limits")

    search_budget = raw.get("search_budget")
    if not isinstance(search_budget, Mapping):
        raise ValueError("stance_request_missing_search_budget")
    max_candidates = search_budget.get("max_candidates")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
        raise ValueError("stance_request_invalid_search_budget_max_candidates")
    if max_candidates < 1:
        raise ValueError("stance_request_invalid_search_budget_max_candidates")
    normalized_budget: dict[str, Any] = {"max_candidates": int(max_candidates)}
    if search_budget.get("max_seconds") is not None:
        max_seconds = _finite_float(search_budget.get("max_seconds"))
        if max_seconds is None or max_seconds <= 0:
            raise ValueError("stance_request_invalid_search_budget_max_seconds")
        normalized_budget["max_seconds"] = max_seconds

    seed_raw = raw.get("seed")
    if seed_raw is None:
        seed = 0
    elif isinstance(seed_raw, bool) or not isinstance(seed_raw, int):
        raise ValueError("stance_request_invalid_seed")
    else:
        seed = int(seed_raw)

    yaw_tolerance_rad = _finite_float(raw.get("yaw_tolerance_rad"))
    if yaw_tolerance_rad is None or yaw_tolerance_rad <= 0:
        yaw_tolerance_rad = DEFAULT_YAW_TOLERANCE_RAD

    return {
        "kitchen_scene_digest": kitchen_scene_digest,
        "task_id": task_id,
        "completion_contract": dict(completion_contract),
        "target_prim_path": target_prim_path,
        "affordance_prim_path": affordance_prim_path,
        "robot_profile": {
            **{str(key): _jsonable(value) for key, value in robot_profile.items()},
            "collision_radius_m": collision_radius_m,
            "min_reach_m": min_reach_m,
            "max_reach_m": max_reach_m,
        },
        "reference_pose_xyz": reference_pose_xyz,
        "reference_yaw_rad": reference_yaw_rad,
        "camera_limits": {str(key): _jsonable(value) for key, value in camera_limits.items()},
        "search_budget": normalized_budget,
        "seed": seed,
        "yaw_tolerance_rad": yaw_tolerance_rad,
    }


def _camera_limit(request: Mapping[str, Any], name: str, default: float) -> float:
    value = _finite_float(dict(request.get("camera_limits") or {}).get(name))
    if value is None:
        return default
    return value


def evaluate_stance_gates(
    measurement: Mapping[str, Any], *, request: Mapping[str, Any]
) -> dict[str, Any]:
    """Convert one candidate's measured metrics into per-gate PASS/FAIL rows.

    ALL gates must pass for acceptance, including ``render_evidence_fresh``:
    both PNG paths must exist on disk and ``render_source`` must equal
    ``isaac_rtx_rgb``.  Missing metrics fail closed.
    """

    metrics: Mapping[str, Any] = measurement if isinstance(measurement, Mapping) else {}
    robot_profile = dict(request.get("robot_profile") or {})
    collision_radius_m = float(robot_profile["collision_radius_m"])
    min_reach_m = float(robot_profile["min_reach_m"])
    max_reach_m = float(robot_profile["max_reach_m"])
    yaw_tolerance_rad = float(request.get("yaw_tolerance_rad") or DEFAULT_YAW_TOLERANCE_RAD)

    rows: list[dict[str, Any]] = []
    reasons: list[str] = []

    def _add(gate_id: str, passed: bool, measured: Any, threshold: Any, reason: str) -> None:
        row: dict[str, Any] = {
            "gate_id": gate_id,
            "status": "PASS" if passed else "FAIL",
            "measured": _jsonable(measured),
            "threshold": _jsonable(threshold),
        }
        if not passed:
            row["rejection_reason"] = reason
            reasons.append(reason)
        rows.append(row)

    floor_error = _finite_float(metrics.get("floor_support_error_m"))
    _add(
        "floor_support",
        floor_error is not None and abs(floor_error) <= FLOOR_SUPPORT_TOLERANCE_M,
        floor_error,
        {"max_abs_floor_support_error_m": FLOOR_SUPPORT_TOLERANCE_M},
        "floor_support_error_exceeds_tolerance",
    )

    uprightness_deg = _finite_float(metrics.get("uprightness_deg"))
    _add(
        "uprightness",
        uprightness_deg is not None and 0.0 <= uprightness_deg <= UPRIGHTNESS_TOLERANCE_DEG,
        uprightness_deg,
        {"max_uprightness_deg": UPRIGHTNESS_TOLERANCE_DEG},
        "uprightness_tilt_exceeds_tolerance",
    )

    min_clearance_m = _finite_float(metrics.get("min_clearance_m"))
    _add(
        "collision_clearance",
        min_clearance_m is not None and min_clearance_m >= collision_radius_m,
        min_clearance_m,
        {"min_clearance_m": collision_radius_m},
        "collision_clearance_below_robot_radius",
    )

    yaw_error_rad = _finite_float(metrics.get("yaw_error_rad"))
    _add(
        "target_facing_yaw",
        yaw_error_rad is not None and abs(yaw_error_rad) <= yaw_tolerance_rad,
        yaw_error_rad,
        {"max_abs_yaw_error_rad": yaw_tolerance_rad},
        "target_facing_yaw_error_exceeds_tolerance",
    )

    reach_distance_m = _finite_float(metrics.get("reach_distance_m"))
    _add(
        "reach_envelope",
        reach_distance_m is not None and min_reach_m <= reach_distance_m <= max_reach_m,
        reach_distance_m,
        {"min_reach_m": min_reach_m, "max_reach_m": max_reach_m},
        "reach_distance_outside_envelope",
    )

    min_visible = _camera_limit(
        request, "min_affordance_visible_fraction", DEFAULT_MIN_AFFORDANCE_VISIBLE_FRACTION
    )
    visible_fraction = _finite_float(metrics.get("affordance_visible_fraction"))
    _add(
        "affordance_visibility",
        visible_fraction is not None and visible_fraction >= min_visible,
        visible_fraction,
        {"min_affordance_visible_fraction": min_visible},
        "affordance_visibility_below_minimum",
    )

    min_robot_frame = _camera_limit(
        request, "min_robot_in_frame_fraction", DEFAULT_MIN_ROBOT_IN_FRAME_FRACTION
    )
    min_target_frame = _camera_limit(
        request, "min_target_in_frame_fraction", DEFAULT_MIN_TARGET_IN_FRAME_FRACTION
    )
    robot_fraction = _finite_float(metrics.get("robot_in_frame_fraction"))
    target_fraction = _finite_float(metrics.get("target_in_frame_fraction"))
    _add(
        "robot_target_framing",
        robot_fraction is not None
        and target_fraction is not None
        and robot_fraction >= min_robot_frame
        and target_fraction >= min_target_frame,
        {"robot_in_frame_fraction": robot_fraction, "target_in_frame_fraction": target_fraction},
        {
            "min_robot_in_frame_fraction": min_robot_frame,
            "min_target_in_frame_fraction": min_target_frame,
        },
        "robot_target_framing_insufficient",
    )

    robot_pov_png = str(metrics.get("robot_pov_png") or "")
    third_person_png = str(metrics.get("third_person_png") or "")
    render_source = str(metrics.get("render_source") or "")
    render_ok = (
        bool(robot_pov_png)
        and Path(robot_pov_png).is_file()
        and bool(third_person_png)
        and Path(third_person_png).is_file()
        and render_source == REQUIRED_RENDER_SOURCE
    )
    _add(
        RENDER_EVIDENCE_GATE_ID,
        render_ok,
        {
            "robot_pov_png": robot_pov_png or None,
            "third_person_png": third_person_png or None,
            "render_source": render_source or None,
        },
        {"required_render_source": REQUIRED_RENDER_SOURCE, "png_files_must_exist": True},
        "render_evidence_not_fresh_isaac_rgb",
    )

    return {
        "gates": rows,
        "all_gates_passed": all(row["status"] == "PASS" for row in rows),
        "rejection_reasons": reasons,
        "robot_pov_png": robot_pov_png or None,
        "third_person_png": third_person_png or None,
    }


def default_search_region(request: Mapping[str, Any]) -> dict[str, Any]:
    robot_profile = dict(request.get("robot_profile") or {})
    midpoint = (float(robot_profile["min_reach_m"]) + float(robot_profile["max_reach_m"])) / 2.0
    return {
        "standoff_m": min(max(midpoint, _MIN_STANDOFF_M), _MAX_STANDOFF_M),
        "yaw_correction_rad": 0.0,
    }


def refine_search_region(
    region: Mapping[str, Any],
    gate_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure deterministic search-region update from measured gate failures."""

    refined = dict(region)
    failed = {
        str(row.get("gate_id") or ""): dict(row)
        for row in gate_rows
        if isinstance(row, Mapping) and row.get("status") == "FAIL"
    }
    standoff = _finite_float(region.get("standoff_m"))
    if standoff is None:
        standoff = 0.75
    if "collision_clearance" in failed:
        standoff *= 1.2
    elif "reach_envelope" in failed:
        threshold = dict(failed["reach_envelope"].get("threshold") or {})
        reach = _finite_float(metrics.get("reach_distance_m"))
        max_reach = _finite_float(threshold.get("max_reach_m"))
        min_reach = _finite_float(threshold.get("min_reach_m"))
        if reach is not None and max_reach is not None and reach > max_reach:
            standoff *= 0.85
        elif reach is not None and min_reach is not None and reach < min_reach:
            standoff *= 1.15
    refined["standoff_m"] = min(max(standoff, _MIN_STANDOFF_M), _MAX_STANDOFF_M)
    if "target_facing_yaw" in failed:
        yaw_error = _finite_float(metrics.get("yaw_error_rad"))
        if yaw_error is not None:
            base = _finite_float(region.get("yaw_correction_rad")) or 0.0
            refined["yaw_correction_rad"] = base - yaw_error
    return refined


def generate_stance_candidate(
    *, request: Mapping[str, Any], candidate_index: int, region: Mapping[str, Any]
) -> dict[str, Any]:
    """Deterministic ring-of-standoffs generator around the reference stance pose.

    Seeded purely from (request seed, candidate_index) so the same seed always
    reproduces the same candidate for the same search region. Never emits a
    pose outside the sane radius of the reference/affordance stance pose.
    """

    seed = int(request.get("seed") or 0)
    rng = random.Random((seed * 1_000_003 + int(candidate_index)) % (2**63))
    reference = [float(value) for value in request["reference_pose_xyz"]]
    reference_yaw = float(request["reference_yaw_rad"])
    standoff_base = _finite_float(region.get("standoff_m"))
    if standoff_base is None:
        standoff_base = 0.75
    standoff = standoff_base * (1.0 + rng.uniform(-0.05, 0.05))
    standoff = min(max(standoff, _MIN_STANDOFF_M), _MAX_STANDOFF_M)
    theta = (
        reference_yaw
        + math.pi
        + candidate_index * _GOLDEN_ANGLE_RAD
        + rng.uniform(-0.05, 0.05)
    )
    pose = [
        reference[0] + standoff * math.cos(theta),
        reference[1] + standoff * math.sin(theta),
        reference[2],
    ]
    yaw = math.atan2(reference[1] - pose[1], reference[0] - pose[0])
    yaw += _finite_float(region.get("yaw_correction_rad")) or 0.0
    candidate = {
        "candidate_source": "deterministic_generator",
        "pose_xyz": [round(value, 6) for value in pose],
        "yaw_rad": round(yaw, 6),
        "standoff_m": round(standoff, 6),
    }
    if _distance_xy(candidate["pose_xyz"], reference) > MAX_CANDIDATE_RADIUS_M:
        raise ValueError("stance_candidate_outside_sane_radius")
    return candidate


def validate_agent_proposal(
    proposal: Any, *, request: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate an agent-proposed candidate. The agent can never waive a gate.

    Returns ``(candidate, [])`` for a valid proposal, else ``(None, blockers)``.
    Any gate-override key (gates, thresholds, force_pass, waive, skip_gates,
    render/evidence/status overrides, ...) yields ``stance_agent_gate_waiver_refused``.
    """

    if not isinstance(proposal, Mapping):
        return None, ["stance_agent_proposal_invalid"]
    keys = {str(key) for key in proposal}
    unknown = keys - _ALLOWED_PROPOSAL_KEYS
    waiver_keys = sorted(
        key
        for key in unknown
        if any(token in key.lower() for token in _GATE_WAIVER_KEY_TOKENS)
    )
    if waiver_keys:
        return None, [AGENT_WAIVER_BLOCKER]
    if unknown:
        return None, ["stance_agent_proposal_invalid"]
    pose = _pose3(proposal.get("pose_xyz"))
    yaw = _finite_float(proposal.get("yaw_rad"))
    if pose is None or yaw is None:
        return None, ["stance_agent_proposal_invalid"]
    reference = [float(value) for value in request["reference_pose_xyz"]]
    distance = _distance_xy(pose, reference)
    if distance > MAX_CANDIDATE_RADIUS_M:
        return None, ["stance_agent_proposal_outside_sane_radius"]
    standoff = _finite_float(proposal.get("standoff_m"))
    if "standoff_m" in proposal and (standoff is None or standoff <= 0):
        return None, ["stance_agent_proposal_invalid"]
    if standoff is None:
        standoff = distance
    return (
        {
            "candidate_source": "agent_proposal",
            "pose_xyz": [round(value, 6) for value in pose],
            "yaw_rad": round(yaw, 6),
            "standoff_m": round(standoff, 6),
        },
        [],
    )


def run_adaptive_stance_search(
    *,
    request: Mapping[str, Any],
    measure_candidate: Callable[[dict[str, Any]], Mapping[str, Any]],
    out_dir: str | Path,
    propose_next_candidate: Callable[[list[dict[str, Any]]], dict[str, Any] | None] | None = None,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Run the bounded stance search. Budget/watchdog stops are ``blocked``."""

    normalized = normalize_stance_search_request(request)
    tick = clock if clock is not None else time.monotonic
    out = Path(out_dir).expanduser().resolve()
    ensure_dir(out)
    evaluations_path = out / EVALUATIONS_ARTIFACT_NAME
    result_path = out / RESULT_ARTIFACT_NAME

    budget = dict(normalized["search_budget"])
    max_candidates = int(budget["max_candidates"])
    max_seconds = _finite_float(budget.get("max_seconds"))

    region = default_search_region(normalized)
    history: list[dict[str, Any]] = []
    waiver_refusals: list[dict[str, Any]] = []
    accepted_record: dict[str, Any] | None = None
    stop_reason = "max_candidates_exhausted"
    started = tick()

    def _persist_evaluations() -> None:
        write_json(
            evaluations_path,
            {
                "schema_version": SCHEMA_VERSION,
                "artifact": "stance_candidate_evaluations",
                "generated_at": utc_now_iso(),
                "task_id": normalized["task_id"],
                "candidate_evaluations": history,
            },
        )

    _persist_evaluations()
    for candidate_index in range(max_candidates):
        if max_seconds is not None and (tick() - started) >= max_seconds:
            stop_reason = "max_seconds_exhausted"
            break

        candidate: dict[str, Any] | None = None
        agent_blockers: list[str] = []
        if propose_next_candidate is not None:
            proposal = propose_next_candidate([dict(row) for row in history])
            if proposal is not None:
                candidate, agent_blockers = validate_agent_proposal(
                    proposal, request=normalized
                )
                if candidate is None:
                    event = {
                        "candidate_index": candidate_index,
                        "blockers": list(agent_blockers),
                        "proposal_keys": sorted(str(key) for key in proposal)
                        if isinstance(proposal, Mapping)
                        else [],
                    }
                    if AGENT_WAIVER_BLOCKER in agent_blockers:
                        waiver_refusals.append(event)
        if candidate is None:
            candidate = generate_stance_candidate(
                request=normalized, candidate_index=candidate_index, region=region
            )

        record: dict[str, Any] = {
            "candidate_index": candidate_index,
            "candidate": candidate,
            "metrics": {},
            "gates": [],
            "rejection_reasons": [],
            "blockers": list(agent_blockers),
            "robot_pov_png": None,
            "third_person_png": None,
        }
        try:
            measurement = measure_candidate(dict(candidate))
        except Exception as exc:  # bounded: record and continue to next candidate
            record["blockers"].append(MEASUREMENT_ERROR_BLOCKER)
            record["rejection_reasons"].append(
                f"{MEASUREMENT_ERROR_BLOCKER}: {type(exc).__name__}: {exc}"
            )
            history.append(record)
            _persist_evaluations()
            continue

        metrics = dict(measurement) if isinstance(measurement, Mapping) else {}
        evaluation = evaluate_stance_gates(metrics, request=normalized)
        record["metrics"] = _jsonable(metrics)
        record["gates"] = evaluation["gates"]
        record["rejection_reasons"] = list(evaluation["rejection_reasons"])
        record["robot_pov_png"] = evaluation["robot_pov_png"]
        record["third_person_png"] = evaluation["third_person_png"]
        history.append(record)
        _persist_evaluations()

        if evaluation["all_gates_passed"]:
            accepted_record = record
            break
        region = refine_search_region(region, evaluation["gates"], metrics)

    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "task_id": normalized["task_id"],
        "kitchen_scene_digest": normalized["kitchen_scene_digest"],
        "target_prim_path": normalized["target_prim_path"],
        "affordance_prim_path": normalized["affordance_prim_path"],
        "gate_ids": list(ALL_STANCE_GATE_IDS),
        "search_budget": budget,
        "seed": normalized["seed"],
        "candidates_evaluated": len(history),
        "evaluations_path": str(evaluations_path),
        "result_path": str(result_path),
        "agent_waiver_refusals": waiver_refusals,
        "provider_revalidation_required": True,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    if accepted_record is not None:
        result = {
            **base,
            "status": "accepted",
            "blockers": [],
            "accepted_candidate": accepted_record["candidate"],
            "accepted_metrics": accepted_record["metrics"],
            "accepted_gates": accepted_record["gates"],
            "evidence": {
                "robot_pov_png": accepted_record["robot_pov_png"],
                "third_person_png": accepted_record["third_person_png"],
                "render_source": REQUIRED_RENDER_SOURCE,
            },
            "provider_acceptance_note": PROVIDER_ACCEPTANCE_NOTE,
        }
    else:
        result = {
            **base,
            "status": "blocked",
            "blockers": [BUDGET_EXHAUSTED_BLOCKER],
            "budget_stop_reason": stop_reason,
            "accepted_candidate": None,
        }
    write_json(result_path, result)
    return result
