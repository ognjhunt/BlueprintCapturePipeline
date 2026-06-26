"""RL/post-training handoff packet builder.

The packet is deliberately model-agnostic. It turns Task Evaluation Run traces
and labels into post-training inputs without claiming that Blueprint trained,
validated, or safely deployed a robot policy.
"""

from __future__ import annotations

import json
import math
from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence


RL_POST_TRAINING_HANDOFF_SCHEMA_VERSION = "rl_post_training_handoff_packet.v1"

SAFETY_EVENT_TYPES = (
    "force_contact_stop",
    "timeout",
    "misgrasp",
    "false_success",
    "human_rescue",
    "reset_quality_issue",
)

BOTTLENECK_STAGE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "pick": ("pick", "reach", "approach", "select"),
    "grasp_close_timing": (
        "grasp_close",
        "grasp close",
        "close_timing",
        "finger_close",
        "grasp_timing",
    ),
    "transfer": ("transfer", "carry", "transport", "move"),
    "handoff": ("handoff", "hand_off", "exchange"),
    "tote_lift": ("tote_lift", "tote lift", "lift", "raise"),
    "place": ("place", "deposit", "dropoff", "bin"),
    "retry_loop": ("retry", "loop", "oscillation", "repeated"),
    "safety_stop": ("safety", "force_stop", "contact_stop", "e_stop", "collision"),
    "timeout": ("timeout", "deadline"),
}

SECRET_KEY_PARTS = ("token", "secret", "password", "credential", "authorization", "api_key")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _safe_id(value: Any, *, fallback: str = "unknown") -> str:
    text = _string(value).lower()
    if not text:
        return fallback
    cleaned = "".join(character if character.isalnum() else "_" for character in text)
    return "_".join(part for part in cleaned.split("_") if part) or fallback


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: Any) -> int | None:
    number = _float(value)
    if number is None:
        return None
    return int(number)


def _string_list(value: Any) -> list[str]:
    if value is None:
        values: Sequence[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = value
    else:
        values = [value]
    out: list[str] = []
    for item in values:
        text = _string(item)
        if text and text not in out:
            out.append(text)
    return out


def _rows(payload: Mapping[str, Any], *keys: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            out.extend(dict(item) for item in value if isinstance(item, Mapping))
    return out


def _first_string(*values: Any) -> str:
    for value in values:
        text = _string(value)
        if text:
            return text
    return ""


def _scrub_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        scrubbed: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(part in key_text.lower() for part in SECRET_KEY_PARTS):
                scrubbed[key_text] = "<redacted>"
            else:
                scrubbed[key_text] = _scrub_secrets(item)
        return scrubbed
    if isinstance(value, list):
        return [_scrub_secrets(item) for item in value]
    return value


def _dominant_counts(rows: Iterable[Mapping[str, Any]], *, key_options: Sequence[str]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        label = ""
        for key in key_options:
            label = _string(row.get(key))
            if label:
                break
        if not label:
            label = "unlabeled"
        label_id = _safe_id(label, fallback="unlabeled")
        counts[label_id] = counts.get(label_id, 0) + 1
    return [
        {"id": key, "count": count}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _success_definition(
    *,
    job_request: Mapping[str, Any],
    evaluation_result: Mapping[str, Any],
) -> dict[str, Any]:
    task = _mapping(job_request.get("task") or job_request.get("target_task"))
    thresholds = _mapping(
        job_request.get("thresholds")
        or job_request.get("success_thresholds")
        or job_request.get("task_thresholds")
    )
    scorecard = _mapping(evaluation_result.get("standard_policy_scorecard"))
    cycle_time = _mapping(scorecard.get("cycle_time"))
    criteria = _string_list(
        task.get("success_criteria")
        or task.get("successCriteria")
        or thresholds.get("success_criteria")
    )
    if not criteria:
        criteria = [
            "task outcome is labeled successful",
            "cycle time remains within the task threshold when a threshold is supplied",
            "operator or safety interventions remain within the task threshold",
        ]
    target_success_rate = _float(
        thresholds.get("target_success_rate")
        or thresholds.get("success_rate")
        or job_request.get("target_success_rate")
    )
    max_cycle_time = _float(
        thresholds.get("max_cycle_time_seconds")
        or thresholds.get("cycle_time_seconds")
        or job_request.get("cycle_time_threshold_seconds")
    )
    return {
        "task_id": _first_string(task.get("task_id"), job_request.get("task_id")),
        "criteria": criteria,
        "target_success_rate": target_success_rate,
        "max_cycle_time_seconds": max_cycle_time,
        "max_intervention_rate": _float(
            thresholds.get("max_intervention_rate")
            or thresholds.get("intervention_rate")
            or job_request.get("max_intervention_rate")
        ),
        "observed_success_rate": _float(scorecard.get("success_rate")),
        "observed_cycle_time_mean_seconds": _float(cycle_time.get("mean_seconds")),
        "source": "job_request.thresholds + evaluation_result.standard_policy_scorecard",
    }


def _sparse_reward_signal(success_definition: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "reward_family": "sparse_task_success_with_intervention_penalties",
        "success_reward": 1.0,
        "failure_reward": 0.0,
        "penalties": {
            "force_contact_stop": -1.0,
            "false_success": -1.0,
            "timeout": -0.75,
            "human_rescue": -0.5,
            "misgrasp": -0.25,
            "reset_quality_issue": -0.25,
        },
        "success_threshold": success_definition.get("target_success_rate"),
        "cycle_time_threshold_seconds": success_definition.get("max_cycle_time_seconds"),
        "dense_reward_required": False,
        "human_review_required_before_training_use": True,
        "source_signal": "attempt success labels, intervention labels, timing metrics, and safety ledger rows",
    }


def _failure_mode(row: Mapping[str, Any]) -> str:
    return _safe_id(
        _first_string(
            row.get("failure_mode_id"),
            row.get("failure_mode"),
            row.get("label"),
            row.get("reason"),
            row.get("event_type"),
        ),
        fallback="unlabeled_failure",
    )


def _recoverability(row: Mapping[str, Any]) -> tuple[bool | None, str]:
    explicit = row.get("recoverable")
    if isinstance(explicit, bool):
        return explicit, "explicit_label"
    mode = _failure_mode(row)
    non_recoverable_markers = ("unsafe", "collision", "e_stop", "force_contact_stop")
    recoverable_markers = (
        "misgrasp",
        "alignment",
        "occlusion",
        "handoff",
        "timeout",
        "retry",
        "grasp",
        "pick",
        "transfer",
    )
    if any(marker in mode for marker in non_recoverable_markers):
        return False, "keyword_heuristic"
    if any(marker in mode for marker in recoverable_markers):
        return True, "keyword_heuristic"
    return None, "review_required"


def _recoverable_failure_labels(label_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels: list[dict[str, Any]] = []
    for row in label_rows:
        recoverable, source = _recoverability(row)
        labels.append(
            {
                "attempt_id": row.get("attempt_id"),
                "scenario_id": row.get("scenario_id"),
                "failure_mode": _failure_mode(row),
                "bottleneck_stage": _classify_bottleneck_stage(row),
                "recoverable": recoverable,
                "recoverability_source": source,
            }
        )
    return {
        "label_count": len(labels),
        "labels": labels,
        "dominant_recoverable_failure_modes": _dominant_counts(
            [label for label in labels if label.get("recoverable") is True],
            key_options=("failure_mode",),
        ),
        "review_required_count": sum(1 for label in labels if label.get("recoverable") is None),
    }


def _classify_event_type(row: Mapping[str, Any]) -> str | None:
    text = " ".join(
        _string(row.get(key)).lower()
        for key in (
            "event_type",
            "failure_mode_id",
            "failure_mode",
            "label",
            "reason",
            "status",
        )
    )
    if "force" in text and ("stop" in text or "contact" in text):
        return "force_contact_stop"
    if "timeout" in text or "deadline" in text:
        return "timeout"
    if "misgrasp" in text or "grasp" in text:
        return "misgrasp"
    if "false_success" in text or "false success" in text:
        return "false_success"
    if "human" in text and ("rescue" in text or "intervention" in text):
        return "human_rescue"
    if "reset" in text:
        return "reset_quality_issue"
    return None


def _collect_event_rows(
    *,
    attempts: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    safety_events: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        attempt_id = attempt.get("attempt_id") or attempt.get("episode_id")
        for key in ("interventions", "safety_events", "events"):
            value = attempt.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, Mapping):
                        rows.append({"attempt_id": attempt_id, **dict(item), "source": f"attempt.{key}"})
    for row in label_rows:
        event_type = _classify_event_type(row)
        if event_type:
            rows.append({**dict(row), "event_type": event_type, "source": "failure_labels"})
    for key in ("events", "interventions", "safety_events", "ledger"):
        value = safety_events.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    rows.append({**dict(item), "source": f"safety_events.{key}"})
    return rows


def _intervention_labels(event_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels: list[dict[str, Any]] = []
    for row in event_rows:
        event_type = _classify_event_type(row) or _safe_id(row.get("event_type"), fallback="other")
        labels.append(
            {
                "attempt_id": row.get("attempt_id"),
                "scenario_id": row.get("scenario_id"),
                "event_type": event_type,
                "severity": row.get("severity") or "review_required",
                "source": row.get("source") or "unknown",
            }
        )
    return {
        "label_count": len(labels),
        "labels": labels,
        "event_type_counts": _dominant_counts(labels, key_options=("event_type",)),
        "supported_event_types": list(SAFETY_EVENT_TYPES),
    }


def _timing_throughput_metrics(
    *,
    attempts: Sequence[Mapping[str, Any]],
    evaluation_result: Mapping[str, Any],
) -> dict[str, Any]:
    scorecard = _mapping(evaluation_result.get("standard_policy_scorecard"))
    cycle_time = _mapping(scorecard.get("cycle_time"))
    mean_seconds = _float(cycle_time.get("mean_seconds"))
    if mean_seconds is None:
        attempt_times = [
            _float(_mapping(attempt.get("metrics")).get("cycle_time_seconds"))
            for attempt in attempts
        ]
        numeric = [value for value in attempt_times if value is not None]
        mean_seconds = round(sum(numeric) / len(numeric), 6) if numeric else None
    throughput = None
    if mean_seconds and mean_seconds > 0:
        throughput = round(3600.0 / mean_seconds, 6)
    return {
        "attempt_count": int(
            scorecard.get("attempt_count")
            or cycle_time.get("sample_count")
            or len(attempts)
            or 0
        ),
        "success_rate": _float(scorecard.get("success_rate")),
        "cycle_time_mean_seconds": mean_seconds,
        "throughput_attempts_per_hour": throughput,
        "intervention_rate": _float(scorecard.get("intervention_rate")),
        "source": "evaluation_result.standard_policy_scorecard with attempt metric fallback",
    }


def _policy_baseline_fingerprint(
    *,
    job_request: Mapping[str, Any],
    policy_package: Mapping[str, Any],
    policy_report: Mapping[str, Any],
    candidate_package: Mapping[str, Any],
    heldout_result: Mapping[str, Any],
) -> dict[str, Any]:
    request_policy = _mapping(job_request.get("policy_package") or job_request.get("policy"))
    baseline_policy_id = _first_string(
        policy_package.get("baseline_policy_id"),
        policy_package.get("policy_id"),
        request_policy.get("baseline_policy_id"),
        request_policy.get("policy_id"),
        request_policy.get("policy_uri"),
    )
    fingerprint_source = _scrub_secrets(
        {
            "baseline_policy_id": baseline_policy_id,
            "policy_package": policy_package,
            "request_policy": request_policy,
            "frozen_verifier_sha256": _first_string(
                policy_report.get("frozen_verifier_sha256"),
                candidate_package.get("frozen_verifier_sha256"),
                heldout_result.get("frozen_verifier_sha256"),
            ),
        }
    )
    digest = sha256(
        json.dumps(fingerprint_source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "baseline_policy_id": baseline_policy_id or None,
        "fingerprint_sha256": digest,
        "fingerprint_source": fingerprint_source,
        "secret_fields_redacted": True,
    }


def _policy_role(run: Mapping[str, Any]) -> str:
    explicit = _safe_id(
        _first_string(
            run.get("policy_role"),
            run.get("cohort"),
            run.get("evaluation_cohort"),
            run.get("candidate_role"),
            run.get("variant_role"),
        ),
        fallback="",
    )
    if explicit:
        return explicit
    policy_id = _string(run.get("policy_id")).lower()
    if "baseline" in policy_id or "frozen" in policy_id or "control" in policy_id:
        return "baseline"
    if "candidate" in policy_id or "treatment" in policy_id or "improved" in policy_id:
        return "candidate"
    return "unspecified"


def _condition_key(run: Mapping[str, Any]) -> str:
    parts = [
        _first_string(run.get("task_id"), run.get("taskId")),
        _first_string(run.get("scenario_id"), run.get("scenarioId")),
        _first_string(
            run.get("scenario_variation_instance_id"),
            run.get("scenarioVariationInstanceId"),
            run.get("variation_id"),
        ),
        _first_string(run.get("split"), run.get("scenario_split"), run.get("eval_split")),
        _string(run.get("speed_factor") or run.get("speed_multiplier") or "1.0"),
    ]
    return "|".join(parts)


def _concurrent_baseline_ab_plan(
    *,
    scenario_matrix: Mapping[str, Any],
    baseline_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    runs = [dict(row) for row in scenario_matrix.get("runs") or [] if isinstance(row, Mapping)]
    baseline_ids: list[str] = []
    candidate_ids: list[str] = []
    by_condition: dict[str, set[str]] = {}
    matched_conditions: list[str] = []
    for index, run in enumerate(runs, start=1):
        role = _policy_role(run)
        run_id = _first_string(
            run.get("scenario_eval_run_id"),
            run.get("run_id"),
            run.get("attempt_id"),
            f"matrix_row_{index}",
        )
        if role in {"baseline", "frozen_baseline", "control"}:
            baseline_ids.append(run_id)
        elif role in {"candidate", "treatment", "improved", "new_policy"}:
            candidate_ids.append(run_id)
        condition = _condition_key(run)
        by_condition.setdefault(condition, set()).add(role)
    for condition, roles in sorted(by_condition.items()):
        if roles.intersection({"baseline", "frozen_baseline", "control"}) and roles.intersection(
            {"candidate", "treatment", "improved", "new_policy"}
        ):
            matched_conditions.append(condition)
    present = bool(matched_conditions)
    blockers: list[str] = []
    if not baseline_ids:
        blockers.append("missing_frozen_baseline_reserved_episodes")
    if not candidate_ids:
        blockers.append("missing_candidate_policy_episodes")
    if baseline_ids and candidate_ids and not matched_conditions:
        blockers.append("missing_same_condition_baseline_candidate_pairs")
    return {
        "status": "ready_for_concurrent_ab_comparison" if present else "planned_missing_concurrent_ab_evidence",
        "frozen_baseline_required": True,
        "old_run_only_comparison_allowed": False,
        "baseline_policy_id": baseline_fingerprint.get("baseline_policy_id"),
        "baseline_run_ids": baseline_ids,
        "candidate_run_ids": candidate_ids,
        "matched_condition_keys": matched_conditions,
        "matched_condition_count": len(matched_conditions),
        "reservation_policy": {
            "reserve_baseline_episodes_in_same_conditions": True,
            "recommended_min_baseline_fraction": 0.2,
            "interleave_baseline_and_candidate_execution": True,
            "same_day_or_same_batch_preferred": True,
        },
        "blockers": blockers,
        "candidate_claim_allowed": present,
    }


def _classify_bottleneck_stage(row: Mapping[str, Any]) -> str:
    explicit = _safe_id(
        _first_string(row.get("bottleneck_stage"), row.get("stage"), row.get("phase")),
        fallback="",
    )
    if explicit:
        if explicit in BOTTLENECK_STAGE_KEYWORDS:
            return explicit
        return explicit
    text = " ".join(
        _string(row.get(key)).lower()
        for key in ("failure_mode_id", "failure_mode", "label", "reason", "event_type")
    )
    for stage, keywords in BOTTLENECK_STAGE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return stage
    return "unknown"


def _bottleneck_stage_detection(
    *,
    label_rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stage_rows: list[dict[str, Any]] = []
    for row in label_rows:
        stage_rows.append(
            {
                "attempt_id": row.get("attempt_id"),
                "scenario_id": row.get("scenario_id"),
                "stage": _classify_bottleneck_stage(row),
                "failure_mode": _failure_mode(row),
            }
        )
    for attempt in attempts:
        if attempt.get("success") is False or _string(attempt.get("status")).lower() in {
            "failed",
            "timeout",
            "blocked",
        }:
            metrics = _mapping(attempt.get("metrics"))
            row = {
                "attempt_id": attempt.get("attempt_id") or attempt.get("episode_id"),
                "scenario_id": attempt.get("scenario_id"),
                "stage": _classify_bottleneck_stage(
                    {
                        "stage": attempt.get("stage") or metrics.get("stage"),
                        "failure_mode": attempt.get("failure_mode") or metrics.get("failure_mode"),
                        "status": attempt.get("status"),
                    }
                ),
                "failure_mode": _safe_id(
                    _first_string(attempt.get("failure_mode"), metrics.get("failure_mode"), attempt.get("status")),
                    fallback="failed_attempt",
                ),
            }
            stage_rows.append(row)
    counts = _dominant_counts(stage_rows, key_options=("stage",))
    dominant = counts[0]["id"] if counts else None
    return {
        "status": "detected_from_failure_labels" if counts else "blocked_missing_failure_stage_labels",
        "dominant_stage": dominant,
        "stage_counts": counts,
        "stage_rows": stage_rows,
        "training_focus_recommendations": [
            {
                "stage": dominant,
                "recommendation": "prioritize post-training on the dominant failing substage before expanding the full task",
            }
        ]
        if dominant
        else [],
        "supported_stage_taxonomy": sorted(BOTTLENECK_STAGE_KEYWORDS),
    }


def _speed_failure_rows(label_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in label_rows:
        mode = _failure_mode(row)
        if any(marker in mode for marker in ("timing", "speed", "physics", "slip", "collision")):
            rows.append(
                {
                    "attempt_id": row.get("attempt_id"),
                    "scenario_id": row.get("scenario_id"),
                    "failure_mode": mode,
                    "speed_factor": _float(row.get("speed_factor") or row.get("speed_multiplier")),
                }
            )
    return rows


def _speed_curriculum_plan(
    *,
    job_request: Mapping[str, Any],
    policy_package: Mapping[str, Any],
    success_definition: Mapping[str, Any],
    timing_metrics: Mapping[str, Any],
    label_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    action_contract = _mapping(
        policy_package.get("action_chunk_contract")
        or policy_package.get("action_contract")
        or job_request.get("action_chunk_contract")
    )
    native_speed = _float(
        action_contract.get("native_demo_speed_factor")
        or action_contract.get("native_speed_factor")
        or job_request.get("native_demo_speed_factor")
    )
    if native_speed is None:
        native_speed = 1.0
    target_success = _float(success_definition.get("target_success_rate")) or 0.95
    max_cycle_time = _float(success_definition.get("max_cycle_time_seconds"))
    observed_success = _float(timing_metrics.get("success_rate"))
    speed_factors = [native_speed, round(native_speed * 1.1, 4), round(native_speed * 1.25, 4), round(native_speed * 1.5, 4)]
    milestones: list[dict[str, Any]] = []
    for index, factor in enumerate(speed_factors):
        milestone_cycle_time = None
        if max_cycle_time:
            milestone_cycle_time = round(max_cycle_time / max(factor, 0.0001), 6)
        milestones.append(
            {
                "stage": index,
                "speed_factor": factor,
                "gate": {
                    "min_success_rate": target_success,
                    "max_cycle_time_seconds": milestone_cycle_time,
                    "max_force_contact_stop_count": 0,
                    "max_false_success_count": 0,
                    "requires_recovery_success_at_this_speed": True,
                },
            }
        )
    blocked_native = observed_success is not None and observed_success < target_success
    return {
        "status": "blocked_until_native_speed_recovery_passes" if blocked_native else "ready_for_native_speed_gate",
        "native_demo_speed_factor": native_speed,
        "milestones": milestones,
        "physics_or_timing_failure_labels": _speed_failure_rows(label_rows),
        "advance_rule": "advance only when task success and safety/contact gates pass at the current speed",
        "candidate_training_use_requires_review": True,
    }


def _to_float_vector(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    vector: list[float] = []
    for item in value:
        number = _float(item)
        if number is None:
            return None
        vector.append(number)
    return vector if vector else None


def _chunk_sequence(value: Any) -> list[list[float]]:
    if isinstance(value, Mapping):
        for key in ("actions", "action_chunk", "chunk", "values", "trajectory"):
            sequence = _chunk_sequence(value.get(key))
            if sequence:
                return sequence
        return []
    vector = _to_float_vector(value)
    if vector is not None:
        return [vector]
    if isinstance(value, list):
        sequence: list[list[float]] = []
        for item in value:
            item_vector = _to_float_vector(item)
            if item_vector is not None:
                sequence.append(item_vector)
            else:
                nested = _chunk_sequence(item)
                if nested:
                    sequence.extend(nested)
        return sequence
    return []


def _extract_action_chunks(
    *,
    policy_execution_trace: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> list[list[list[float]]]:
    chunks: list[list[list[float]]] = []
    direct = policy_execution_trace.get("action_chunks") or policy_execution_trace.get("chunks")
    if isinstance(direct, list):
        for item in direct:
            sequence = _chunk_sequence(item)
            if sequence:
                chunks.append(sequence)
    for key in ("policy_outputs", "outputs", "steps"):
        value = policy_execution_trace.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    sequence = _chunk_sequence(
                        item.get("action_chunk")
                        or item.get("actions")
                        or item.get("chunk")
                        or item.get("policy_action")
                    )
                    if sequence:
                        chunks.append(sequence)
    for attempt in _rows(trace, "attempts"):
        sequence = _chunk_sequence(attempt.get("action_chunk") or attempt.get("actions") or attempt.get("action_trace"))
        if sequence:
            chunks.append(sequence)
    return chunks


def _mean_abs_delta(left: Sequence[float], right: Sequence[float]) -> float | None:
    width = min(len(left), len(right))
    if width <= 0:
        return None
    return sum(abs(float(left[index]) - float(right[index])) for index in range(width)) / width


def _score_from_delta(delta: float | None) -> float | None:
    if delta is None:
        return None
    return round(1.0 / (1.0 + max(delta, 0.0)), 6)


def _inactive_dimensions(policy_package: Mapping[str, Any], job_request: Mapping[str, Any]) -> list[int]:
    contract = _mapping(
        policy_package.get("action_chunk_contract")
        or policy_package.get("action_contract")
        or job_request.get("action_chunk_contract")
    )
    dims: list[int] = []
    for key in (
        "inactive_dimensions",
        "inactive_arm_dimensions",
        "inactive_torso_dimensions",
        "inactive_joint_indices",
    ):
        value = contract.get(key)
        if isinstance(value, list):
            for item in value:
                index = _int(item)
                if index is not None and index >= 0 and index not in dims:
                    dims.append(index)
    return sorted(dims)


def _action_chunk_continuity_qa(
    *,
    policy_execution_trace: Mapping[str, Any],
    trace: Mapping[str, Any],
    policy_package: Mapping[str, Any],
    job_request: Mapping[str, Any],
) -> dict[str, Any]:
    chunks = _extract_action_chunks(policy_execution_trace=policy_execution_trace, trace=trace)
    if not chunks:
        return {
            "status": "not_applicable_no_action_chunks",
            "chunk_count": 0,
            "prefix_consistency_score": None,
            "boundary_smoothness_score": None,
            "inactive_arm_torso_drift_score": None,
            "autoregressive_degradation_score": None,
            "claim_boundary": {
                "no_action_chunk_quality_claim_without_action_chunks": True,
            },
        }
    boundary_deltas: list[float] = []
    for previous, current in zip(chunks, chunks[1:]):
        if previous and current:
            delta = _mean_abs_delta(previous[-1], current[0])
            if delta is not None:
                boundary_deltas.append(delta)
    avg_boundary_delta = (
        sum(boundary_deltas) / len(boundary_deltas) if boundary_deltas else None
    )
    inactive_dims = _inactive_dimensions(policy_package, job_request)
    inactive_drift: float | None = None
    if inactive_dims:
        vectors = [vector for chunk in chunks for vector in chunk]
        if vectors:
            first = vectors[0]
            max_drift = 0.0
            for vector in vectors[1:]:
                for index in inactive_dims:
                    if index < len(first) and index < len(vector):
                        max_drift = max(max_drift, abs(vector[index] - first[index]))
            inactive_drift = max_drift
    degradation_delta: float | None = None
    if len(boundary_deltas) >= 2:
        degradation_delta = max(boundary_deltas[-1] - boundary_deltas[0], 0.0)
    status = "passed_continuity_smoke"
    prefix_score = _score_from_delta(avg_boundary_delta)
    inactive_score = _score_from_delta(inactive_drift)
    degradation_score = _score_from_delta(degradation_delta)
    if prefix_score is not None and prefix_score < 0.5:
        status = "review_required_action_chunk_boundary_jump"
    if inactive_score is not None and inactive_score < 0.5:
        status = "review_required_inactive_axis_drift"
    return {
        "status": status,
        "chunk_count": len(chunks),
        "action_dimensions": len(chunks[0][0]) if chunks and chunks[0] else 0,
        "prefix_consistency_score": prefix_score,
        "boundary_smoothness_score": _score_from_delta(avg_boundary_delta),
        "inactive_arm_torso_drift_score": inactive_score,
        "autoregressive_degradation_score": degradation_score,
        "boundary_mean_abs_delta": round(avg_boundary_delta, 6) if avg_boundary_delta is not None else None,
        "inactive_dimensions": inactive_dims,
        "claim_boundary": {
            "chunk_qa_is_trace_quality_not_task_success": True,
            "chunk_qa_is_not_safety_validation": True,
        },
    }


def _real_robot_evidence_present(row: Mapping[str, Any]) -> bool:
    return any(
        bool(row.get(key))
        for key in (
            "real_robot_sensor_evidence",
            "sensor_log_ref",
            "force_torque_log_ref",
            "owner_evidence_ref",
            "hardware_log_ref",
        )
    )


def _learned_to_avoid_summary(
    *,
    policy_report: Mapping[str, Any],
    heldout_result: Mapping[str, Any],
    concurrent_ab: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_rate = _float(
        policy_report.get("baseline_safety_event_rate")
        or policy_report.get("baseline_intervention_rate")
        or heldout_result.get("baseline_safety_event_rate")
    )
    candidate_rate = _float(
        policy_report.get("candidate_safety_event_rate")
        or policy_report.get("candidate_intervention_rate")
        or heldout_result.get("candidate_safety_event_rate")
    )
    comparable = concurrent_ab.get("candidate_claim_allowed") is True
    avoided = (
        baseline_rate is not None
        and candidate_rate is not None
        and candidate_rate < baseline_rate
        and comparable
    )
    return {
        "status": "supported_by_concurrent_ab_rates" if avoided else "not_proven",
        "baseline_event_rate": baseline_rate,
        "candidate_event_rate": candidate_rate,
        "rate_delta": round(candidate_rate - baseline_rate, 6)
        if baseline_rate is not None and candidate_rate is not None
        else None,
        "requires_concurrent_ab_evidence": True,
        "concurrent_ab_evidence_present": comparable,
    }


def _intervention_safety_ledger(
    *,
    event_rows: Sequence[Mapping[str, Any]],
    policy_report: Mapping[str, Any],
    heldout_result: Mapping[str, Any],
    concurrent_ab: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in event_rows:
        event_type = _classify_event_type(row) or _safe_id(row.get("event_type"), fallback="other")
        real_evidence = _real_robot_evidence_present(row)
        rows.append(
            {
                "attempt_id": row.get("attempt_id"),
                "scenario_id": row.get("scenario_id"),
                "event_type": event_type,
                "severity": row.get("severity") or "review_required",
                "real_robot_sensor_evidence": real_evidence,
                "source": row.get("source") or "unknown",
            }
        )
    real_evidence_present = any(row["real_robot_sensor_evidence"] for row in rows)
    return {
        "schema_version": "rl_intervention_safety_ledger.v1",
        "event_count": len(rows),
        "events": rows,
        "event_type_counts": _dominant_counts(rows, key_options=("event_type",)),
        "learned_to_avoid_events": _learned_to_avoid_summary(
            policy_report=policy_report,
            heldout_result=heldout_result,
            concurrent_ab=concurrent_ab,
        ),
        "real_robot_sensor_evidence_present": real_evidence_present,
        "claim_boundary": {
            "sim_or_label_events_are_training_signals_not_safety_validation": True,
            "safety_validation_proven": False,
            "physical_robot_readiness_proven": False,
            "requires_real_robot_sensor_evidence_for_safety_claim": True,
        },
    }


def build_rl_post_training_handoff_packet(
    *,
    scene_id: str,
    capture_id: str,
    job_id: str | None = None,
    generated_at: str,
    job_request: Mapping[str, Any] | None = None,
    scenario_matrix: Mapping[str, Any] | None = None,
    trace: Mapping[str, Any] | None = None,
    labels: Mapping[str, Any] | None = None,
    evaluation_result: Mapping[str, Any] | None = None,
    policy_package: Mapping[str, Any] | None = None,
    policy_report: Mapping[str, Any] | None = None,
    candidate_package: Mapping[str, Any] | None = None,
    heldout_result: Mapping[str, Any] | None = None,
    policy_execution_trace: Mapping[str, Any] | None = None,
    safety_events: Mapping[str, Any] | None = None,
    source_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    request = _mapping(job_request)
    matrix = _mapping(scenario_matrix)
    normalized_trace = _mapping(trace)
    failure_labels = _mapping(labels)
    eval_result = _mapping(evaluation_result)
    package = _mapping(policy_package)
    report = _mapping(policy_report)
    candidate = _mapping(candidate_package)
    heldout = _mapping(heldout_result)
    action_trace = _mapping(policy_execution_trace)
    safety = _mapping(safety_events)
    artifacts = _mapping(source_artifacts)

    attempts = _rows(normalized_trace, "attempts")
    label_rows = _rows(
        failure_labels,
        "labels",
        "failure_labels",
        "accepted_failure_labels",
        "failures",
    )
    success_definition = _success_definition(job_request=request, evaluation_result=eval_result)
    timing_metrics = _timing_throughput_metrics(attempts=attempts, evaluation_result=eval_result)
    fingerprint = _policy_baseline_fingerprint(
        job_request=request,
        policy_package=package,
        policy_report=report,
        candidate_package=candidate,
        heldout_result=heldout,
    )
    concurrent_ab = _concurrent_baseline_ab_plan(
        scenario_matrix=matrix,
        baseline_fingerprint=fingerprint,
    )
    event_rows = _collect_event_rows(
        attempts=attempts,
        label_rows=label_rows,
        safety_events=safety,
    )

    return {
        "schema_version": RL_POST_TRAINING_HANDOFF_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "job_id": job_id,
        "artifact_purpose": "rl_post_training_handoff_packet",
        "intended_package_surfaces": [
            "Task Evaluation Run",
            "Post-Training Data Package",
            "Policy Improvement Run",
        ],
        "success_definition": success_definition,
        "sparse_reward_signal": _sparse_reward_signal(success_definition),
        "recoverable_failure_labels": _recoverable_failure_labels(label_rows),
        "intervention_labels": _intervention_labels(event_rows),
        "timing_throughput_metrics": timing_metrics,
        "policy_baseline_fingerprint": fingerprint,
        "concurrent_baseline_ab": concurrent_ab,
        "bottleneck_stage_detection": _bottleneck_stage_detection(
            label_rows=label_rows,
            attempts=attempts,
        ),
        "speed_curriculum_plan": _speed_curriculum_plan(
            job_request=request,
            policy_package=package,
            success_definition=success_definition,
            timing_metrics=timing_metrics,
            label_rows=label_rows,
        ),
        "action_chunk_continuity_qa": _action_chunk_continuity_qa(
            policy_execution_trace=action_trace,
            trace=normalized_trace,
            policy_package=package,
            job_request=request,
        ),
        "intervention_safety_ledger": _intervention_safety_ledger(
            event_rows=event_rows,
            policy_report=report,
            heldout_result=heldout,
            concurrent_ab=concurrent_ab,
        ),
        "source_artifacts": artifacts,
        "claim_boundary": {
            "handoff_packet_is_training_support_not_training_completion": True,
            "sparse_reward_signal_requires_robot_team_review": True,
            "concurrent_ab_required_for_candidate_improvement_claim": True,
            "bottleneck_detection_is_label_derived": True,
            "speed_curriculum_is_plan_not_completed_training": True,
            "action_chunk_qa_is_trace_quality_not_task_success": True,
            "intervention_ledger_is_not_safety_validation_without_real_robot_sensor_evidence": True,
            "deployment_approval_proven": False,
            "physical_robot_readiness_proven": False,
            "safety_validation_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
