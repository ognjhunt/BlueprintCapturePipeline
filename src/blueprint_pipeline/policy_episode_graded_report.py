"""Graded episode reports: partial credit, failure events, smoothness, timing.

NVIDIA's RoboLab benchmark reports more about an episode than whether it
passed: how far a policy got through a task's subtasks, which failure events
happened on the way (drops, collisions, wrong-object contact), how smooth the
motion was (spectral arc length), and how long it took. A single pass/fail on
a site scene hides all of that, and it is exactly what a site and a robot team
need to see where a candidate falls short.

This module derives those measures for our own site-scene episodes. It reads
only what the canary already sealed: the deterministic score report (or its
published correction) and the episode's state trace. The score report stays
the only authority on success. Nothing here changes a score, ranks a
candidate, or promotes anything; the sidecar says so and the Website checks
it.

Backlog: ADP-009D (Day-28 ``public_data_rehearsal``), whose acceptance asks
for per-family success, contacts and task-state failures rather than a bare
success count. It adds a sidecar next to an existing publication and changes
no scorer source, so ``scoring_version_digest`` is untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest

EPISODE_SCHEMA_VERSION = "policy_episode_graded_report.v1"
CANDIDATE_SUMMARY_SCHEMA_VERSION = "policy_canary_graded_candidate_summary.v1"
SIDECAR_SCHEMA_VERSION = "task_evaluation_policy_canary_graded_report_sidecar.v1"

DEFAULT_CONTROL_FREQUENCY_HZ = 15.0
DEFAULT_MOVEMENT_EPSILON_M = 0.005
DEFAULT_SETTLE_POSITION_TOLERANCE_M = 0.005

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")

# The first position of each pose field is the one the scorer reads.
_OBJECT_POSE_FIELDS = ("task_scoring_pose_world", "task_object_pose_world", "can_pose_world")
_END_EFFECTOR_POSE_FIELD = "controlled_body_pose_world"

_COLLISION_EVENT_KINDS = {
    "forbidden_robot_object_contact_force_exceeded": "robot_body_hit_object",
    "robot_scene_contact_force_exceeded": "robot_hit_scene",
    "task_scene_contact_force_exceeded": "object_hit_scene",
}


class GradedReportError(ValueError):
    """Stable failure for evidence that cannot be graded or bound exactly."""


# --------------------------------------------------------------- smoothness


def sparc(
    speed: Sequence[float],
    sample_rate_hz: float,
    *,
    pad_level: int = 4,
    cutoff_hz: float = 10.0,
    amplitude_threshold: float = 0.05,
) -> float | None:
    """Spectral arc length of a speed profile (<= 0; closer to 0 is smoother).

    Balasubramanian et al., "On the analysis of movement smoothness", J.
    NeuroEngineering Rehabil. 12:112 (2015). A single minimum-jerk reach
    scores about -1.4; jerky motion and repeated sub-movements score lower.
    Returns None when the profile is too short, flat, or not finite.
    """

    import numpy as np

    values = np.asarray(list(speed), dtype=float)
    if values.size < 4 or not np.all(np.isfinite(values)) or sample_rate_hz <= 0:
        return None
    if float(np.max(np.abs(values))) == 0.0:
        return None
    nfft = int(2 ** (math.ceil(math.log2(values.size)) + pad_level))
    freqs = np.arange(0, sample_rate_hz, sample_rate_hz / nfft)
    magnitude = np.abs(np.fft.fft(values, nfft))
    magnitude = magnitude / np.max(magnitude)
    keep = freqs <= cutoff_hz
    freqs, magnitude = freqs[keep], magnitude[keep]
    above = np.nonzero(magnitude >= amplitude_threshold)[0]
    if above.size < 2:
        return None
    freqs = freqs[above[0] : above[-1] + 1]
    magnitude = magnitude[above[0] : above[-1] + 1]
    span = freqs[-1] - freqs[0]
    if span <= 0:
        return None
    arc = np.sqrt((np.diff(freqs) / span) ** 2 + np.diff(magnitude) ** 2)
    return round(float(-np.sum(arc)), 6)


# ------------------------------------------------------------------ helpers


def _finite_vector(value: Any, length: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < length:
        return None
    out: list[float] = []
    for item in value[:length]:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        out.append(number)
    return out


def _positions(samples: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[list[float]]:
    """The xyz series for the first field every sample carries, else empty."""

    for field in fields:
        series = [_finite_vector(sample.get(field), 3) for sample in samples]
        if series and all(point is not None for point in series):
            return [point for point in series if point is not None]
    return []


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _speeds(points: Sequence[Sequence[float]], hz: float) -> list[float]:
    return [_distance(points[i], points[i - 1]) * hz for i in range(1, len(points))]


def _step(samples: Sequence[Mapping[str, Any]], index: int) -> int:
    value = samples[index].get("step_index", index)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else index


def _seconds(step: int | None, hz: float) -> float | None:
    return None if step is None else round(step / hz, 4)


def _criteria(score: Mapping[str, Any]) -> dict[str, bool]:
    raw = score.get("criteria_satisfied")
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): value is True for key, value in raw.items()}


# -------------------------------------------------------- partial credit


def _subtasks(
    *,
    score: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    object_points: Sequence[Sequence[float]],
    movement_epsilon_m: float,
) -> list[dict[str, Any]]:
    """The task's stages, in order, each achieved or not, from sealed evidence.

    Mirrors RoboLab's subtask partial credit. Every stage is read from the
    deterministic score report's own criteria where it has one, so the grade
    can never disagree with the score about what happened.
    """

    criteria = _criteria(score)
    measurements = score.get("measurements") if isinstance(score.get("measurements"), Mapping) else {}
    strategy = str(score.get("manipulation_strategy") or "pick_and_place")

    contact_step = next(
        (_step(samples, i) for i, sample in enumerate(samples) if sample.get("task_contact_active") is True),
        None,
    )
    if contact_step is None:
        contact_step = next(
            (
                _step(samples, i)
                for i, sample in enumerate(samples)
                if isinstance(sample.get("finger_contact_forces_n"), list)
                and any(
                    isinstance(force, (int, float)) and not isinstance(force, bool) and force > 0
                    for force in sample["finger_contact_forces_n"]
                )
            ),
            None,
        )
    motion_step = None
    if object_points:
        origin = object_points[0]
        motion_step = next(
            (
                _step(samples, i)
                for i, point in enumerate(object_points)
                if _distance(point, origin) > movement_epsilon_m
            ),
            None,
        )
    maximum_translation = measurements.get("maximum_translation_m")
    moved = motion_step is not None or (
        isinstance(maximum_translation, (int, float)) and maximum_translation > movement_epsilon_m
    )

    def stage(stage_id: str, label: str, achieved: bool, step: int | None = None) -> dict[str, Any]:
        return {"id": stage_id, "label": label, "condition_met": bool(achieved),
                "first_step_index": step if achieved else None}

    reached = criteria.get("destination_containment", False) and criteria.get("surface_target", True)
    if strategy == "planar_push":
        stages = [
            stage("contact", "Touched the object", contact_step is not None, contact_step),
            stage("moved", "Moved the object", moved, motion_step),
            stage("translated", "Moved it the required distance", criteria.get("minimum_translation", False)),
            stage("reached_destination", "Reached the destination", reached),
            stage(
                "settled_and_cleared",
                "Left it settled and let go",
                criteria.get("settling", False)
                and criteria.get("terminal_task_contact", False)
                and criteria.get("retreat", True),
            ),
        ]
        return _ladder(stages)
    return _ladder([
        stage("contact", "Touched the object", contact_step is not None, contact_step),
        stage("moved", "Moved the object", moved, motion_step),
        stage("lifted", "Lifted it", criteria.get("minimum_lift", False)),
        stage("translated", "Carried it the required distance", criteria.get("minimum_translation", False)),
        stage(
            "placed",
            "Placed it at the destination",
            reached and criteria.get("support_height", True) and criteria.get("support_contact", True),
        ),
        stage(
            "released_and_settled",
            "Released it and it settled",
            criteria.get("gripper_state", False)
            and criteria.get("settling", False)
            and criteria.get("terminal_task_contact", True)
            and criteria.get("retreat", True),
        ),
    ])


def _ladder(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A stage counts only once every earlier stage has.

    "Released and settled" is trivially true of an object nobody touched, so
    a condition met out of order is reported (``condition_met``) but earns no
    credit (``achieved``). Partial credit then means how far along the task
    the policy actually got.
    """

    reached = True
    for item in stages:
        reached = reached and item["condition_met"]
        item["achieved"] = reached
        if not reached:
            item["first_step_index"] = None
    return stages


# ---------------------------------------------------------- failure events


def _failure_events(score: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ledger = score.get("event_ledger") if isinstance(score.get("event_ledger"), Mapping) else {}
    safety_events = [event for event in ledger.get("safety_events") or [] if isinstance(event, Mapping)]
    counts = {name: 0 for name in _COLLISION_EVENT_KINDS.values()}
    for event in safety_events:
        name = _COLLISION_EVENT_KINDS.get(str(event.get("event_type") or ""))
        if name:
            counts[name] += 1
    drops = [event for event in ledger.get("drop_events") or [] if isinstance(event, Mapping)]

    # One task object per scene today, so a wrong-object grasp cannot happen
    # and is not claimed as zero. A scene that reports contact with another
    # object is counted.
    wrong_object_steps = [
        _step(samples, i)
        for i, sample in enumerate(samples)
        if sample.get("non_task_object_contact_active") is True
    ]
    wrong_object = (
        {"status": "measured", "count": len(wrong_object_steps)}
        if any("non_task_object_contact_active" in sample for sample in samples)
        else {"status": "not_measurable", "reason": "single_task_object_scene"}
    )

    def count(value: Any) -> int:
        return len(value) if isinstance(value, list) else 0

    def optional_int(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    return {
        "drops": len(drops),
        "drop_steps": [
            int(event["drop_step_index"])
            for event in drops
            if isinstance(event.get("drop_step_index"), int)
        ],
        **counts,
        "containment_excursion_steps": count(ledger.get("containment_excursion_steps")),
        "workspace_excursion_steps": count(ledger.get("workspace_excursion_steps")),
        "retries": optional_int(ledger.get("maximum_retries_observed")),
        "regrasps": optional_int(ledger.get("maximum_regrasps_observed")),
        "wrong_object_interactions": wrong_object,
    }


# ------------------------------------------------------------------ timing


def _settled_at_step(
    samples: Sequence[Mapping[str, Any]],
    object_points: Sequence[Sequence[float]],
    tolerance_m: float,
) -> int | None:
    """The first step after which the object never leaves its final position."""

    if not object_points:
        return None
    final = object_points[-1]
    index = len(object_points) - 1
    while index > 0 and _distance(object_points[index - 1], final) <= tolerance_m:
        index -= 1
    return _step(samples, index)


# ------------------------------------------------------------------ episode


def grade_episode(
    *,
    score: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    joint_states: Sequence[Mapping[str, Any]] = (),
    control_frequency_hz: float = DEFAULT_CONTROL_FREQUENCY_HZ,
    movement_epsilon_m: float = DEFAULT_MOVEMENT_EPSILON_M,
    settle_position_tolerance_m: float = DEFAULT_SETTLE_POSITION_TOLERANCE_M,
    state_trace_digest: str | None = None,
) -> dict[str, Any]:
    """Grade one episode from its deterministic score and sealed state trace."""

    if not isinstance(score, Mapping):
        raise GradedReportError("graded_report_score_invalid")
    if control_frequency_hz <= 0 or not math.isfinite(control_frequency_hz):
        raise GradedReportError("graded_report_frequency_invalid")
    samples = [sample for sample in samples if isinstance(sample, Mapping)]
    hz = float(control_frequency_hz)
    object_points = _positions(samples, _OBJECT_POSE_FIELDS)
    status = str(score.get("status") or "")
    succeeded = score.get("task_succeeded")

    subtasks = _subtasks(
        score=score, samples=samples, object_points=object_points,
        movement_epsilon_m=movement_epsilon_m,
    )
    gradable = status == "scored" and bool(samples)
    graded_score = (
        round(sum(1 for item in subtasks if item["achieved"]) / len(subtasks), 4)
        if gradable else None
    )

    end_effector = _positions(samples, (_END_EFFECTOR_POSE_FIELD,))
    ee_speed = _speeds(end_effector, hz) if len(end_effector) >= 2 else []
    joints = [
        vector
        for vector in (
            _finite_vector(row.get("joint_positions_rad"), len(row.get("joint_positions_rad") or []))
            for row in joint_states
            if isinstance(row, Mapping)
        )
        if vector
    ]
    joint_speed = (
        [_distance(joints[i], joints[i - 1]) * hz for i in range(1, len(joints))]
        if len(joints) >= 2 and len({len(vector) for vector in joints}) == 1
        else []
    )

    first_contact = next((item["first_step_index"] for item in subtasks if item["id"] == "contact"), None)
    first_motion = next((item["first_step_index"] for item in subtasks if item["id"] == "moved"), None)
    first_step = _step(samples, 0) if samples else None
    last_step = _step(samples, len(samples) - 1) if samples else None

    report: dict[str, Any] = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "status": "graded" if gradable else "not_gradable",
        "not_gradable_reason": None if gradable else (
            "no_state_samples" if not samples else f"score_{status or 'missing'}"
        ),
        "task_succeeded": succeeded if isinstance(succeeded, bool) else None,
        "outcome": str(score.get("outcome") or "") or None,
        "manipulation_strategy": str(score.get("manipulation_strategy") or "pick_and_place"),
        "graded_score": graded_score,
        "subtasks": subtasks,
        "safety_ok": _criteria(score).get("safety") if "safety" in _criteria(score) else None,
        "failure_events": _failure_events(score, samples),
        "smoothness": {
            "end_effector_sparc": sparc(ee_speed, hz) if ee_speed else None,
            "joint_sparc": sparc(joint_speed, hz) if joint_speed else None,
            "end_effector_path_length_m": (
                round(sum(ee_speed) / hz, 4) if ee_speed else None
            ),
        },
        "timing": {
            "control_frequency_hz": hz,
            "episode_duration_s": (
                round((last_step - first_step + 1) / hz, 4)
                if first_step is not None and last_step is not None else None
            ),
            "first_task_contact_s": _seconds(first_contact, hz),
            "first_object_motion_s": _seconds(first_motion, hz),
            # Only a success has a meaningful "done" moment.
            "settled_at_s": _seconds(
                _settled_at_step(samples, object_points, settle_position_tolerance_m), hz
            ) if succeeded is True else None,
        },
        "parameters": {
            "movement_epsilon_m": movement_epsilon_m,
            "settle_position_tolerance_m": settle_position_tolerance_m,
            "sparc": {"pad_level": 4, "cutoff_hz": 10.0, "amplitude_threshold": 0.05},
        },
        "inputs": {
            "score_report_digest": (
                str(score.get("report_digest"))
                if _DIGEST.fullmatch(str(score.get("report_digest") or "")) else None
            ),
            "state_trace_digest": (
                state_trace_digest if _DIGEST.fullmatch(str(state_trace_digest or "")) else None
            ),
        },
        "authority": {
            "grader": "deterministic_derivation_from_sealed_evidence",
            "task_success_authority": "deterministic_score_report",
            "learned_judge_consulted": False,
            "ranking_or_promotion_effect": "none",
        },
        "report_digest": "",
    }
    report["report_digest"] = cross_runtime_canonical_digest(report, digest_field="report_digest")
    return report


# ---------------------------------------------------------------- summary


def _median(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if isinstance(value, (int, float))]
    return round(float(statistics.median(present)), 4) if present else None


def summarize_candidate(candidate_id: str, graded: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-candidate totals. A description of one run, never a ranking."""

    scored = [item for item in graded if item.get("status") == "graded"]
    stage_ids: list[str] = []
    for item in scored:
        for stage in item.get("subtasks") or []:
            if stage["id"] not in stage_ids:
                stage_ids.append(stage["id"])
    completion = {
        stage_id: round(
            sum(
                1 for item in scored
                for stage in item["subtasks"] if stage["id"] == stage_id and stage["achieved"]
            ) / len(scored),
            4,
        )
        for stage_id in stage_ids
    } if scored else {}
    events = [item.get("failure_events") or {} for item in graded]
    collisions = ("robot_body_hit_object", "robot_hit_scene", "object_hit_scene")
    return {
        "schema_version": CANDIDATE_SUMMARY_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "episode_count": len(graded),
        "graded_episode_count": len(scored),
        "success_count": sum(1 for item in graded if item.get("task_succeeded") is True),
        "mean_graded_score": (
            round(statistics.fmean(item["graded_score"] for item in scored), 4) if scored else None
        ),
        "subtask_completion_rate": completion,
        "episodes_with_drop": sum(1 for event in events if event.get("drops", 0) > 0),
        "episodes_with_collision": sum(
            1 for event in events if any(event.get(name, 0) > 0 for name in collisions)
        ),
        "total_drops": sum(int(event.get("drops", 0)) for event in events),
        "median_end_effector_sparc": _median(
            [(item.get("smoothness") or {}).get("end_effector_sparc") for item in scored]
        ),
        "median_episode_duration_s": _median(
            [(item.get("timing") or {}).get("episode_duration_s") for item in scored]
        ),
        "median_settled_at_s": _median(
            [(item.get("timing") or {}).get("settled_at_s") for item in scored]
        ),
        "ranking_permitted": False,
    }


# ----------------------------------------------------------------- sidecar


def _mapping(value: Any, *, code: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GradedReportError(code)
    return json.loads(json.dumps(dict(value), allow_nan=False))


def _episode_key(value: Mapping[str, Any]) -> tuple[str, str, str, int]:
    episode_id = str(value.get("episode_id") or "")
    candidate_id = str(value.get("candidate_id") or "")
    cell_id = str(value.get("cell_id") or "")
    seed = value.get("seed")
    if (
        not _IDENTIFIER.fullmatch(episode_id)
        or not _IDENTIFIER.fullmatch(candidate_id)
        or not _IDENTIFIER.fullmatch(cell_id)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2_147_483_647
    ):
        raise GradedReportError("graded_report_episode_identity_invalid")
    return episode_id, candidate_id, cell_id, seed


def _load_source_result(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise GradedReportError("graded_report_source_result_invalid")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GradedReportError("graded_report_source_result_invalid") from exc
    return _mapping(value, code="graded_report_source_result_invalid")


def _control_frequency(task_spec: Mapping[str, Any]) -> float:
    value = task_spec.get("control_frequency_hz")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and number > 0:
            return number
    return DEFAULT_CONTROL_FREQUENCY_HZ


def build_policy_canary_graded_report_sidecar(
    *,
    source_site_record: Mapping[str, Any],
    source_result_path: str | Path,
    evidence_root: str | Path,
    record_id: str,
    generated_at_iso: str,
) -> dict[str, Any]:
    """Bind graded reports to one published run without rewriting it.

    ``source_site_record`` is the Website's record for the run: its
    publication and, when one was applied, the score correction.
    ``source_result_path`` is the retained terminal session result the
    publication was projected from, and ``evidence_root`` holds the artifacts
    its inventory names. Every completed episode is verified exactly as the
    rescorer verifies it before a single measure is derived. A published
    correction's scores are used where they exist, because they are the
    published deterministic result.
    """

    # The rescorer owns retained-result verification; grading reuses it so the
    # two can never disagree about which evidence is genuine.
    from .native_task_arena_policy_canary_session import validate_session_result
    from .task_evaluation_policy_canary_rescore import (
        PolicyCanaryRescoreError,
        _verify_artifact_inventory,
        _verify_source_episode,
    )

    if not _IDENTIFIER.fullmatch(record_id):
        raise GradedReportError("graded_report_record_id_invalid")
    try:
        generated_at = datetime.fromisoformat(generated_at_iso.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GradedReportError("graded_report_generated_at_invalid") from exc
    if generated_at.tzinfo is None:
        raise GradedReportError("graded_report_generated_at_invalid")

    root = _mapping(source_site_record, code="graded_report_source_invalid")
    publication = _mapping(root.get("publication", root), code="graded_report_publication_invalid")
    if publication.get("schema_version") != "task_evaluation_run_publication.v4":
        raise GradedReportError("graded_report_publication_invalid")
    projection = _mapping(publication.get("policy_canary_result"), code="graded_report_projection_invalid")
    if (
        projection.get("schema_version") != "task_evaluation_policy_canary_result_projection.v1"
        or projection.get("projection_digest")
        != cross_runtime_canonical_digest(projection, digest_field="projection_digest")
    ):
        raise GradedReportError("graded_report_projection_invalid")
    delivery = _mapping(publication.get("result_delivery"), code="graded_report_delivery_invalid")
    delivery_digest = str(delivery.get("delivery_digest") or "")
    if (
        not _DIGEST.fullmatch(delivery_digest)
        or projection.get("result_delivery_digest") != delivery_digest
        or publication.get("run_id") != projection.get("run_id")
    ):
        raise GradedReportError("graded_report_delivery_invalid")
    report = projection.get("report") if isinstance(projection.get("report"), Mapping) else {}
    published_result_digest = str(report.get("result_digest") or "")

    correction = root.get("score_correction")
    correction_digest: str | None = None
    corrected: dict[tuple[str, str, int], dict[str, Any]] = {}
    if correction is not None:
        correction = _mapping(correction, code="graded_report_score_correction_invalid")
        correction_digest = str(correction.get("sidecar_digest") or "")
        if not _DIGEST.fullmatch(correction_digest):
            raise GradedReportError("graded_report_score_correction_invalid")
        # The Website stores the sidecar with the correction itself nested
        # under ``correction``; its score updates are the published scores.
        body = correction.get("correction")
        updates = body.get("score_updates") if isinstance(body, Mapping) else None
        if not isinstance(updates, list):
            raise GradedReportError("graded_report_score_correction_invalid")
        for update in updates:
            if isinstance(update, Mapping) and isinstance(update.get("new_score"), Mapping):
                corrected[(str(update.get("candidate_id")), str(update.get("cell_id")),
                           int(update.get("seed", -1)))] = dict(update["new_score"])

    source_episodes = projection.get("episodes")
    if not isinstance(source_episodes, list) or not source_episodes:
        raise GradedReportError("graded_report_episode_inventory_invalid")
    keys = sorted({_episode_key(row) for row in source_episodes if isinstance(row, Mapping)})
    if len(keys) != len(source_episodes):
        raise GradedReportError("graded_report_episode_inventory_invalid")

    source = _load_source_result(source_result_path)
    try:
        validate_session_result(source, allow_legacy_missing_task_success_contract=True)
    except ValueError as exc:
        raise GradedReportError("graded_report_source_result_invalid") from exc
    if (
        source.get("run_id") != projection.get("run_id")
        or not _DIGEST.fullmatch(published_result_digest)
        or source.get("result_digest") != published_result_digest
    ):
        raise GradedReportError("graded_report_source_result_unbound")
    evidence = Path(evidence_root).expanduser()
    if evidence.is_symlink() or not evidence.is_dir():
        raise GradedReportError("graded_report_evidence_root_invalid")
    evidence = evidence.resolve()
    try:
        inventory = _verify_artifact_inventory(source, evidence_root=evidence)
    except PolicyCanaryRescoreError as exc:
        raise GradedReportError("graded_report_artifact_inventory_invalid") from exc
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for raw in source.get("episodes") or []:
        row = _mapping(raw, code="graded_report_source_episode_invalid")
        key = (str(row.get("candidate_id")), str(row.get("cell_id")), row.get("seed"))
        if key in rows:
            raise GradedReportError("graded_report_source_episode_duplicate")
        rows[key] = row

    episodes: list[dict[str, Any]] = []
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for episode_id, candidate_id, cell_id, seed in keys:
        row = rows.get((candidate_id, cell_id, seed))
        if row is None:
            raise GradedReportError("graded_report_episode_evidence_missing")
        raw_episode = row.get("episode") if isinstance(row.get("episode"), Mapping) else {}
        if row.get("status") == "completed":
            try:
                task_spec, recorded_score, samples = _verify_source_episode(
                    row, inventory=inventory, evidence_root=evidence
                )
            except PolicyCanaryRescoreError as exc:
                raise GradedReportError("graded_report_episode_evidence_invalid") from exc
            state = raw_episode["state_trace"]
            graded = grade_episode(
                score=corrected.get((candidate_id, cell_id, seed)) or recorded_score,
                samples=samples,
                joint_states=state.get("joint_states") or [],
                control_frequency_hz=_control_frequency(task_spec),
                state_trace_digest=str(row.get("state_trace_digest") or "") or None,
            )
        else:
            # An episode that never completed has no trace to grade. It is
            # listed, not dropped, so the report covers the whole run.
            score = raw_episode.get("score") if isinstance(raw_episode.get("score"), Mapping) else {}
            graded = grade_episode(score=score or {"status": "not_completed"}, samples=[])
        episodes.append({
            "episode_id": episode_id,
            "candidate_id": candidate_id,
            "cell_id": cell_id,
            "seed": seed,
            "graded": graded,
        })
        by_candidate.setdefault(candidate_id, []).append(graded)

    sidecar: dict[str, Any] = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "source_binding": {
            "record_id": record_id,
            "source_run_id": str(projection["run_id"]),
            "source_projection_digest": str(projection["projection_digest"]),
            "source_delivery_digest": delivery_digest,
            "source_score_correction_sidecar_digest": correction_digest,
        },
        "candidates": [
            summarize_candidate(candidate_id, graded)
            for candidate_id, graded in sorted(by_candidate.items())
        ],
        "episodes": episodes,
        "audit": {
            "original_publication_preserved": True,
            "deterministic_scores_unchanged": True,
            "derived_only_from_sealed_episode_evidence": True,
            "ranking_or_promotion_effect": "none",
            "generated_at_iso": generated_at_iso,
        },
        "sidecar_digest": "",
    }
    sidecar["sidecar_digest"] = cross_runtime_canonical_digest(sidecar, digest_field="sidecar_digest")
    return sidecar


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site-record", required=True, help="The Website's record for the run (JSON).")
    parser.add_argument("--source-result", required=True, help="The run's retained terminal result.")
    parser.add_argument("--evidence-root", required=True, help="Directory holding its artifact inventory.")
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--generated-at", required=True, help="ISO 8601 with a timezone.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    site_record = json.loads(Path(args.site_record).read_text(encoding="utf-8"))
    sidecar = build_policy_canary_graded_report_sidecar(
        source_site_record=site_record,
        source_result_path=args.source_result,
        evidence_root=args.evidence_root,
        record_id=args.record_id,
        generated_at_iso=args.generated_at,
    )
    out = Path(args.out)
    if out.exists():
        raise GradedReportError("graded_report_output_exists")
    out.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"sidecar_digest": sidecar["sidecar_digest"], "episodes": len(sidecar["episodes"])}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_SUMMARY_SCHEMA_VERSION",
    "EPISODE_SCHEMA_VERSION",
    "GradedReportError",
    "SIDECAR_SCHEMA_VERSION",
    "build_policy_canary_graded_report_sidecar",
    "grade_episode",
    "sparc",
    "summarize_candidate",
]
