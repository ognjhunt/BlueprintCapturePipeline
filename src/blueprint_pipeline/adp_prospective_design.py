"""Fail-closed prospective design compiler for Arm Decision Proof v1.

The retrospective ADP-008 replay is intentionally immutable.  This module is
the forward-only seam used before a partner execution: an explicit candidate
baseline and one declared independent two-proportion design compile into the
exact condition/reset/seed/repetition schedule that execution must admit.

All arithmetic is deterministic and local.  The module does not admit a
partner, approve a protocol, launch a simulator, or access physical outcomes.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from statistics import NormalDist
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


POWER_METHOD = "independent_two_proportion_fixed_sample_normal_approximation.v1"
UNCERTAINTY_METHOD = "conservative_difference_of_wilson_marginal_intervals.v1"
INVALID_TRIAL_RULE = "retain_in_frozen_denominator_as_failure"
STOP_RULE = "fixed_schedule_all_trials_terminal_no_early_success_stop"
MULTIPLICITY_RULE = "single_primary_two_candidate_contrast_none"
SCHEDULE_SCHEMA_VERSION = "adp_prospective_trial_schedule.v1"
DECISION_SCHEMA_VERSION = "adp_prospective_decision.v1"
ADMISSION_SCHEMA_VERSION = "adp_execution_schedule_admission.v1"
EPISODE_ADMISSION_SCHEMA_VERSION = "adp_prospective_episode_admission.v1"
SHA256_PREFIX = "sha256:"


class ADPProspectiveDesignError(ValueError):
    """Stable, sorted blockers for a prospective design or schedule."""

    def __init__(self, blockers: Sequence[str]):
        self.blockers = tuple(sorted(set(str(item) for item in blockers if str(item))))
        super().__init__(";".join(self.blockers))


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _is_digest(value: Any) -> bool:
    text = _string(value)
    return (
        len(text) == 71
        and text.startswith(SHA256_PREFIX)
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _candidate_pair(value: Mapping[str, Any]) -> tuple[str, str]:
    baseline = _string(value.get("baseline_candidate_id"))
    alternative = _string(value.get("alternative_candidate_id"))
    blockers: list[str] = []
    if not baseline:
        blockers.append("baseline_candidate_id_missing")
    if not alternative:
        blockers.append("alternative_candidate_id_missing")
    if baseline and baseline == alternative:
        blockers.append("candidate_pair_not_distinct")
    if blockers:
        raise ADPProspectiveDesignError(blockers)
    return baseline, alternative


def compile_power_requirement(design: Mapping[str, Any]) -> dict[str, Any]:
    """Compile the one supported statistical design into a trial requirement.

    ``planning_variance_rate`` is frozen at 0.5 because it maximizes Bernoulli
    variance and avoids using outcome-informed pilot estimates.  The formula is
    the direct inverse of the method named in ``POWER_METHOD``.
    """

    blockers: list[str] = []
    method = _string(design.get("method"))
    alpha = _number(design.get("alpha"))
    power = _number(design.get("power"))
    difference = _number(design.get("minimum_decision_relevant_difference"))
    variance_rate = _number(design.get("planning_variance_rate"))
    if method != POWER_METHOD:
        blockers.append("statistical_design_method_unsupported")
    if alpha is None or not 0.0 < alpha < 0.5:
        blockers.append("statistical_design_alpha_invalid")
    if power is None or not 0.5 < power < 1.0:
        blockers.append("statistical_design_power_invalid")
    if difference is None or not 0.0 < difference <= 1.0:
        blockers.append("statistical_design_mdre_invalid")
    if variance_rate != 0.5:
        blockers.append("statistical_design_requires_conservative_variance_rate_0_5")
    if design.get("uncertainty_method") != UNCERTAINTY_METHOD:
        blockers.append("statistical_design_uncertainty_method_mismatch")
    if design.get("invalid_trial_handling") != INVALID_TRIAL_RULE:
        blockers.append("statistical_design_invalid_trial_rule_mismatch")
    if design.get("stop_rule") != STOP_RULE:
        blockers.append("statistical_design_stop_rule_mismatch")
    if design.get("multiplicity") != MULTIPLICITY_RULE:
        blockers.append("statistical_design_multiplicity_mismatch")
    if blockers:
        raise ADPProspectiveDesignError(blockers)

    assert alpha is not None
    assert power is not None
    assert difference is not None
    assert variance_rate is not None
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(power)
    numerator = 2.0 * variance_rate * (1.0 - variance_rate) * (z_alpha + z_power) ** 2
    minimum_trials = math.ceil(numerator / difference**2)
    result = {
        "schema_version": "adp_power_requirement.v1",
        "method": method,
        "formula": "ceil(2*p*(1-p)*(z_(1-alpha/2)+z_power)^2/mdrd^2)",
        "planning_variance_rate": variance_rate,
        "minimum_decision_relevant_difference": difference,
        "alpha": alpha,
        "power": power,
        "uncertainty_method": UNCERTAINTY_METHOD,
        "invalid_trial_handling": INVALID_TRIAL_RULE,
        "stop_rule": STOP_RULE,
        "multiplicity": MULTIPLICITY_RULE,
        "minimum_trials_per_candidate": minimum_trials,
    }
    result["power_requirement_digest"] = canonical_digest(
        result, digest_field="power_requirement_digest"
    )
    return result


def validate_secondary_metrics(metrics: Any) -> list[dict[str, Any]]:
    """Reject every secondary metric not evidenced as owner-preregistered."""

    if metrics is None:
        return []
    rows = _rows(metrics)
    blockers: list[str] = []
    if not isinstance(metrics, list) or len(rows) != len(metrics):
        raise ADPProspectiveDesignError(["secondary_metrics_invalid"])
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        metric_id = _string(row.get("metric_id"))
        label = metric_id or f"index_{index}"
        if not metric_id:
            blockers.append(f"secondary_metric_id_missing:{label}")
        elif metric_id in seen:
            blockers.append(f"secondary_metric_duplicate:{metric_id}")
        seen.add(metric_id)
        if row.get("preregistered_by_partner_task_owner") is not True:
            blockers.append(f"secondary_metric_not_owner_preregistered:{label}")
        if not _is_digest(row.get("owner_evidence_digest")):
            blockers.append(f"secondary_metric_owner_evidence_missing:{label}")
        normalized.append(row)
    if blockers:
        raise ADPProspectiveDesignError(blockers)
    return normalized


def _normalized_conditions(conditions: Any) -> list[dict[str, Any]]:
    rows = _rows(conditions)
    blockers: list[str] = []
    if not isinstance(conditions, list) or not rows or len(rows) != len(conditions):
        raise ADPProspectiveDesignError(["schedule_conditions_missing_or_invalid"])
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        condition_id = _string(row.get("condition_id"))
        if not condition_id:
            blockers.append(f"schedule_condition_id_missing:index_{index}")
        elif condition_id in seen:
            blockers.append(f"schedule_condition_duplicate:{condition_id}")
        seen.add(condition_id)
        if not _is_digest(row.get("reset_digest")):
            blockers.append(f"schedule_reset_digest_missing:{condition_id or index}")
        normalized.append(
            {
                "condition_id": condition_id,
                "reset_digest": row.get("reset_digest"),
            }
        )
    if blockers:
        raise ADPProspectiveDesignError(blockers)
    return normalized


def _trial_id(row: Mapping[str, Any]) -> str:
    return "adp-trial-" + canonical_digest(row)[7:31]


def _order_key(randomization_seed: int, trial_id: str) -> str:
    payload = f"{randomization_seed}:{trial_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _paired_interleaving_order(
    rows: Sequence[Mapping[str, Any]], *, randomization_seed: int
) -> list[dict[str, Any]]:
    """Randomize matched condition/seed pairs while alternating candidates."""

    pairs: dict[str, list[dict[str, Any]]] = {}
    for value in rows:
        row = dict(value)
        pair_id = canonical_digest(
            {
                "condition_id": row.get("condition_id"),
                "reset_digest": row.get("reset_digest"),
                "repetition": row.get("repetition"),
                "seed": row.get("seed"),
            }
        )
        pairs.setdefault(pair_id, []).append(row)
    ordered: list[dict[str, Any]] = []
    for pair_id in sorted(
        pairs,
        key=lambda value: (_order_key(randomization_seed, value), value),
    ):
        pair = pairs[pair_id]
        pair.sort(
            key=lambda row: (
                _order_key(randomization_seed, _string(row.get("trial_id"))),
                _string(row.get("trial_id")),
            )
        )
        ordered.extend(pair)
    return ordered


def compile_trial_schedule(
    *,
    candidate_pair: Mapping[str, Any],
    conditions: Any,
    statistical_design: Mapping[str, Any],
    randomization_seed: int,
    seed_start: int = 0,
) -> dict[str, Any]:
    """Compile power into an exact balanced condition/reset/seed schedule."""

    baseline, alternative = _candidate_pair(candidate_pair)
    normalized_conditions = _normalized_conditions(conditions)
    power_requirement = compile_power_requirement(statistical_design)
    if isinstance(randomization_seed, bool) or not isinstance(randomization_seed, int):
        raise ADPProspectiveDesignError(["schedule_randomization_seed_invalid"])
    if isinstance(seed_start, bool) or not isinstance(seed_start, int) or seed_start < 0:
        raise ADPProspectiveDesignError(["schedule_seed_start_invalid"])
    minimum_trials = power_requirement["minimum_trials_per_candidate"]
    repetitions = math.ceil(minimum_trials / len(normalized_conditions))
    rows: list[dict[str, Any]] = []
    for candidate_role, candidate_id in (
        ("baseline", baseline),
        ("alternative", alternative),
    ):
        for condition in normalized_conditions:
            for repetition in range(repetitions):
                identity = {
                    "candidate_id": candidate_id,
                    "candidate_role": candidate_role,
                    "condition_id": condition["condition_id"],
                    "reset_digest": condition["reset_digest"],
                    "repetition": repetition,
                    "seed": seed_start + repetition,
                }
                rows.append({"trial_id": _trial_id(identity), **identity})
    rows = _paired_interleaving_order(rows, randomization_seed=randomization_seed)
    for order_index, row in enumerate(rows):
        row["execution_order"] = order_index
    trials_per_candidate = repetitions * len(normalized_conditions)
    result = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "status": "qualified_for_execution",
        "candidate_pair": {
            "baseline_candidate_id": baseline,
            "alternative_candidate_id": alternative,
        },
        "conditions": normalized_conditions,
        "statistical_design": dict(statistical_design),
        "power_requirement": power_requirement,
        "randomization": {
            "method": "sha256_block_randomized_candidate_pairs.v1",
            "seed": randomization_seed,
            "interleaving": (
                "each adjacent two-trial block contains baseline and alternative "
                "under one matched condition reset and seed"
            ),
            "blinding": "candidate_identity_hidden_from_independent_outcome_grader",
        },
        "seed_start": seed_start,
        "repetitions_per_candidate_condition": repetitions,
        "trials_per_candidate": trials_per_candidate,
        "total_trial_budget": len(rows),
        "rows": rows,
    }
    result["schedule_digest"] = canonical_digest(result, digest_field="schedule_digest")
    return result


def validate_schedule_for_execution(schedule: Mapping[str, Any]) -> dict[str, Any]:
    """Reject an underpowered or altered schedule before any execution starts."""

    blockers: list[str] = []
    candidate_pair = _mapping(schedule.get("candidate_pair"))
    conditions = schedule.get("conditions")
    design = _mapping(schedule.get("statistical_design"))
    randomization = _mapping(schedule.get("randomization"))
    try:
        baseline, alternative = _candidate_pair(candidate_pair)
        normalized_conditions = _normalized_conditions(conditions)
        power_requirement = compile_power_requirement(design)
    except ADPProspectiveDesignError as exc:
        raise ADPProspectiveDesignError(exc.blockers) from exc
    supplied_digest = schedule.get("schedule_digest")
    if schedule.get("schema_version") != SCHEDULE_SCHEMA_VERSION:
        blockers.append("schedule_schema_invalid")
    if schedule.get("status") != "qualified_for_execution":
        blockers.append("schedule_status_invalid")
    if supplied_digest != canonical_digest(schedule, digest_field="schedule_digest"):
        blockers.append("schedule_digest_mismatch")
    if schedule.get("power_requirement") != power_requirement:
        blockers.append("schedule_power_requirement_mismatch")
    rows = _rows(schedule.get("rows"))
    if len(rows) != len(schedule.get("rows", [])):
        blockers.append("schedule_rows_invalid")
    trial_ids = [_string(row.get("trial_id")) for row in rows]
    if any(not trial_id for trial_id in trial_ids) or len(trial_ids) != len(set(trial_ids)):
        blockers.append("schedule_trial_ids_missing_or_duplicate")
    counts = Counter(_string(row.get("candidate_id")) for row in rows)
    minimum_trials = power_requirement["minimum_trials_per_candidate"]
    for candidate_id in (baseline, alternative):
        if counts[candidate_id] < minimum_trials:
            blockers.append(f"schedule_below_frozen_power_requirement:{candidate_id}")
    if set(counts) != {baseline, alternative}:
        blockers.append("schedule_candidate_set_mismatch")
    expected_condition_resets = {
        (row["condition_id"], row["reset_digest"]) for row in normalized_conditions
    }
    observed_condition_resets = {
        (_string(row.get("condition_id")), row.get("reset_digest")) for row in rows
    }
    if observed_condition_resets != expected_condition_resets:
        blockers.append("schedule_condition_reset_matrix_mismatch")
    repetitions = schedule.get("repetitions_per_candidate_condition")
    seed_start = schedule.get("seed_start")
    if isinstance(seed_start, bool) or not isinstance(seed_start, int) or seed_start < 0:
        blockers.append("schedule_seed_start_invalid")
        seed_start = 0
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        blockers.append("schedule_repetition_count_invalid")
    else:
        expected_cells = {
            (candidate_id, condition["condition_id"], repetition, seed_start + repetition)
            for candidate_id in (baseline, alternative)
            for condition in normalized_conditions
            for repetition in range(repetitions)
        }
        observed_cells = {
            (
                _string(row.get("candidate_id")),
                _string(row.get("condition_id")),
                row.get("repetition"),
                row.get("seed"),
            )
            for row in rows
        }
        if observed_cells != expected_cells:
            blockers.append("schedule_condition_seed_repetition_matrix_mismatch")
        expected_trials_per_candidate = repetitions * len(normalized_conditions)
        if schedule.get("trials_per_candidate") != expected_trials_per_candidate:
            blockers.append("schedule_trials_per_candidate_mismatch")
        if any(
            counts[candidate_id] != expected_trials_per_candidate
            for candidate_id in (baseline, alternative)
        ):
            blockers.append("schedule_candidate_trial_count_mismatch")
    if schedule.get("total_trial_budget") != len(rows):
        blockers.append("schedule_total_trial_budget_mismatch")
    if randomization.get("method") != "sha256_block_randomized_candidate_pairs.v1":
        blockers.append("schedule_randomization_method_invalid")
    if randomization.get("interleaving") != (
        "each adjacent two-trial block contains baseline and alternative "
        "under one matched condition reset and seed"
    ):
        blockers.append("schedule_interleaving_rule_invalid")
    if randomization.get("blinding") != (
        "candidate_identity_hidden_from_independent_outcome_grader"
    ):
        blockers.append("schedule_blinding_rule_invalid")
    randomization_seed = randomization.get("seed")
    if isinstance(randomization_seed, bool) or not isinstance(randomization_seed, int):
        blockers.append("schedule_randomization_seed_invalid")
    else:
        expected_rows = _paired_interleaving_order(
            rows,
            randomization_seed=randomization_seed,
        )
        if trial_ids != [row.get("trial_id") for row in expected_rows] or [
            row.get("execution_order") for row in rows
        ] != list(range(len(rows))):
            blockers.append("schedule_randomization_order_mismatch")
        for offset in range(0, len(rows), 2):
            pair = rows[offset : offset + 2]
            if len(pair) != 2:
                blockers.append("schedule_interleaving_pair_incomplete")
                continue
            if {row.get("candidate_id") for row in pair} != {baseline, alternative}:
                blockers.append(f"schedule_interleaving_candidate_pair_invalid:{offset // 2}")
            matched_fields = {
                (
                    row.get("condition_id"),
                    row.get("reset_digest"),
                    row.get("repetition"),
                    row.get("seed"),
                )
                for row in pair
            }
            if len(matched_fields) != 1:
                blockers.append(f"schedule_interleaving_reset_pair_invalid:{offset // 2}")
    for row in rows:
        identity = {
            "candidate_id": row.get("candidate_id"),
            "candidate_role": row.get("candidate_role"),
            "condition_id": row.get("condition_id"),
            "reset_digest": row.get("reset_digest"),
            "repetition": row.get("repetition"),
            "seed": row.get("seed"),
        }
        expected_role = "baseline" if row.get("candidate_id") == baseline else "alternative"
        if row.get("candidate_role") != expected_role or row.get("trial_id") != _trial_id(identity):
            blockers.append(f"schedule_trial_identity_mismatch:{row.get('trial_id')}")
    if blockers:
        raise ADPProspectiveDesignError(blockers)
    result = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "status": "admitted_for_execution",
        "schedule_digest": supplied_digest,
        "minimum_trials_per_candidate": minimum_trials,
        "scheduled_trials_per_candidate": {
            baseline: counts[baseline],
            alternative: counts[alternative],
        },
        "total_trial_budget": len(rows),
        "execution_may_start": True,
    }
    result["admission_digest"] = canonical_digest(result, digest_field="admission_digest")
    return result


def validate_episode_evidence_contract(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Admit future episode metadata only with complete review and grader evidence.

    File decoding and digest verification remain runtime responsibilities.  This
    pre-seal contract ensures those required bindings cannot be omitted from a
    prospective receipt.  A failure before the first observation is the sole
    media exception and must carry an explicit typed gap.
    """

    blockers: list[str] = []
    episode_id = _string(episode.get("episode_id"))
    if not episode_id:
        blockers.append("episode_id_missing")
    status = _string(episode.get("status"))
    policy_query_count = episode.get("policy_query_count")
    if (
        isinstance(policy_query_count, bool)
        or not isinstance(policy_query_count, int)
        or policy_query_count < 0
    ):
        blockers.append("episode_policy_query_count_invalid")
        policy_query_count = 0
    visual = _mapping(episode.get("visual_evidence"))
    artifacts = _rows(episode.get("artifacts"))
    role_counts = Counter(_string(row.get("role")) for row in artifacts)

    before_first_observation = (
        status
        in {
            "failed",
            "timed_out",
            "invalid",
            "interrupted",
        }
        and policy_query_count == 0
    )
    if before_first_observation:
        media_gap = _mapping(visual.get("media_gap"))
        if (
            visual.get("status") != "unavailable_before_first_observation"
            or media_gap.get("type") != "before_first_observation"
            or not _string(media_gap.get("reason"))
        ):
            blockers.append("episode_typed_media_gap_missing")
    else:
        if status not in {
            "completed",
            "failed",
            "timed_out",
            "invalid",
            "interrupted",
        }:
            blockers.append("episode_terminal_status_invalid")
        if policy_query_count < 1:
            blockers.append("episode_lossless_policy_inputs_missing")
        if visual.get("status") != "complete" or visual.get("human_review_available") is not True:
            blockers.append("episode_human_review_evidence_incomplete")
        if visual.get("terminal_observation_frame_present") is not True:
            blockers.append("episode_terminal_observation_required")
        if not _is_digest(visual.get("frame_manifest_digest")):
            blockers.append("episode_frame_manifest_digest_missing")
        video = _mapping(visual.get("video"))
        if not _is_digest(video.get("sha256")) or not _string(video.get("relative_path")).endswith(
            ".mp4"
        ):
            blockers.append("episode_review_video_binding_missing")
        required_single_roles = {
            "observation_frame_manifest",
            "terminal_observation_frame",
            "episode_video",
        }
        for role in required_single_roles:
            if role_counts[role] != 1:
                blockers.append(f"episode_artifact_role_count_invalid:{role}")
        if role_counts["policy_input_frame"] != policy_query_count:
            blockers.append("episode_policy_input_frame_count_mismatch")
        for artifact in artifacts:
            role = _string(artifact.get("role"))
            if role in {"policy_input_frame", "terminal_observation_frame"}:
                if (
                    not _string(artifact.get("relative_path")).endswith(".png")
                    or not _is_digest(artifact.get("sha256"))
                    or not _is_digest(artifact.get("raw_rgb_sha256"))
                ):
                    blockers.append(f"episode_lossless_image_binding_invalid:{role}")
            elif role in {"observation_frame_manifest", "episode_video"} and not _is_digest(
                artifact.get("sha256")
            ):
                blockers.append(f"episode_artifact_digest_missing:{role}")

    evaluator = _mapping(episode.get("evaluator"))
    success_evidence = _mapping(episode.get("success_evidence"))
    grader_type = _string(evaluator.get("grader_type"))
    provenance_present = bool(
        _string(evaluator.get("success_source"))
        or _is_digest(evaluator.get("provenance_digest"))
        or (
            len(_string(evaluator.get("source_git_blob_sha1"))) == 40
            and all(
                character in "0123456789abcdef"
                for character in _string(evaluator.get("source_git_blob_sha1"))
            )
        )
    )
    if (
        not grader_type
        or grader_type == "policy_self_report"
        or evaluator.get("owner") in {"policy", "candidate_policy"}
        or evaluator.get("policy_self_report_used") is not False
        or not provenance_present
    ):
        blockers.append("episode_independent_grader_provenance_invalid")
    if (
        success_evidence.get("grader_type") != grader_type
        or success_evidence.get("policy_self_report_used") is not False
    ):
        blockers.append("episode_success_evidence_grader_mismatch")
    if blockers:
        raise ADPProspectiveDesignError(blockers)
    result = {
        "schema_version": EPISODE_ADMISSION_SCHEMA_VERSION,
        "status": "admitted",
        "episode_id": episode_id,
        "completed_media_contract": not before_first_observation,
        "typed_pre_observation_media_gap": before_first_observation,
        "independent_grader_type": grader_type,
    }
    result["episode_admission_digest"] = canonical_digest(
        result, digest_field="episode_admission_digest"
    )
    return result


def _wilson(successes: int, trials: int, confidence: float) -> list[float]:
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4 * trials**2))
        / denominator
    )
    return [round(max(0.0, center - margin), 12), round(min(1.0, center + margin), 12)]


def compile_decision(
    *,
    schedule: Mapping[str, Any],
    trial_results: Any,
) -> dict[str, Any]:
    """Apply the frozen rule with every scheduled trial in the denominator."""

    admission = validate_schedule_for_execution(schedule)
    baseline, alternative = _candidate_pair(_mapping(schedule.get("candidate_pair")))
    result_rows = _rows(trial_results)
    blockers: list[str] = []
    if not isinstance(trial_results, list) or len(result_rows) != len(trial_results):
        raise ADPProspectiveDesignError(["trial_results_invalid"])
    by_trial: dict[str, dict[str, Any]] = {}
    scheduled_ids = {_string(row.get("trial_id")) for row in _rows(schedule.get("rows"))}
    for row in result_rows:
        trial_id = _string(row.get("trial_id"))
        if not trial_id:
            blockers.append("trial_result_id_missing")
        elif trial_id in by_trial:
            blockers.append(f"trial_result_duplicate:{trial_id}")
        elif trial_id not in scheduled_ids:
            blockers.append(f"trial_result_not_scheduled:{trial_id}")
        else:
            by_trial[trial_id] = row
    if blockers:
        raise ADPProspectiveDesignError(blockers)

    allowed_statuses = {"completed", "failed", "timed_out", "invalid", "interrupted"}
    frozen_rows: list[dict[str, Any]] = []
    for scheduled in _rows(schedule.get("rows")):
        trial_id = scheduled["trial_id"]
        supplied = by_trial.get(trial_id)
        if supplied is None:
            status = "missing"
            success = False
        else:
            status = _string(supplied.get("status"))
            if status not in allowed_statuses:
                status = "invalid"
            success = status == "completed" and supplied.get("success") is True
            if status == "completed" and not isinstance(supplied.get("success"), bool):
                status = "invalid"
                success = False
        frozen_rows.append(
            {
                "trial_id": trial_id,
                "candidate_id": scheduled["candidate_id"],
                "condition_id": scheduled["condition_id"],
                "status": status,
                "success": success,
            }
        )

    summaries: dict[str, dict[str, Any]] = {}
    confidence = 1.0 - float(_mapping(schedule.get("statistical_design"))["alpha"])
    for candidate_id in (baseline, alternative):
        rows = [row for row in frozen_rows if row["candidate_id"] == candidate_id]
        successes = sum(row["success"] is True for row in rows)
        statuses = Counter(row["status"] for row in rows)
        summaries[candidate_id] = {
            "frozen_denominator": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows),
            "confidence_interval": _wilson(successes, len(rows), confidence),
            "status_counts": dict(sorted(statuses.items())),
            "non_success_trials_in_denominator": len(rows) - successes,
        }
    baseline_summary = summaries[baseline]
    alternative_summary = summaries[alternative]
    observed_difference = alternative_summary["success_rate"] - baseline_summary["success_rate"]
    difference_interval = [
        round(
            alternative_summary["confidence_interval"][0]
            - baseline_summary["confidence_interval"][1],
            12,
        ),
        round(
            alternative_summary["confidence_interval"][1]
            - baseline_summary["confidence_interval"][0],
            12,
        ),
    ]
    mdre = float(
        _mapping(schedule.get("statistical_design"))["minimum_decision_relevant_difference"]
    )
    if difference_interval[0] >= mdre:
        decision = "select"
        selected = alternative
        eliminated = baseline
        reason = "alternative_conservative_difference_meets_positive_mdre"
    elif difference_interval[1] <= -mdre:
        decision = "eliminate"
        selected = baseline
        eliminated = alternative
        reason = "alternative_conservative_difference_meets_negative_mdre"
    elif difference_interval[0] >= -mdre and difference_interval[1] <= mdre:
        decision = "equivalent_inconclusive"
        selected = None
        eliminated = None
        reason = "difference_bounded_inside_preregistered_equivalence_region"
    else:
        decision = "abstain"
        selected = None
        eliminated = None
        reason = "uncertainty_crosses_preregistered_decision_boundaries"
    result = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision": decision,
        "selected_candidate_id": selected,
        "eliminated_candidate_id": eliminated,
        "reason": reason,
        "baseline_candidate_id": baseline,
        "alternative_candidate_id": alternative,
        "schedule_digest": schedule.get("schedule_digest"),
        "schedule_admission_digest": admission["admission_digest"],
        "rule": dict(_mapping(schedule.get("statistical_design"))),
        "candidate_summaries": summaries,
        "observed_difference_alternative_minus_baseline": round(observed_difference, 12),
        "difference_interval": difference_interval,
        "all_scheduled_trials_retained": len(frozen_rows) == len(_rows(schedule.get("rows"))),
        "frozen_denominator_statuses": [
            "completed",
            "failed",
            "timed_out",
            "invalid",
            "interrupted",
            "missing",
        ],
    }
    result["decision_digest"] = canonical_digest(result, digest_field="decision_digest")
    return result
