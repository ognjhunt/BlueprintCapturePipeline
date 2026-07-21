"""Decision-grade simulator episode scoring and connected policy ranking."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import Any

from .evaluator_evidence_profiles import required_evaluator_evidence_digest_fields
from .policy_evaluation_contracts import (
    MINIMUM_MATCHED_REPLICATES_PER_POLICY_CONDITION,
    validate_policy_evaluation_design,
)


SCHEMA_VERSION = "decision_grade_ranking_request.v2"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_METHOD = "matched_cell_policy_criterion_cluster_percentile.v1"
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _strict_rows(value: Any) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return [], False
    if any(not isinstance(row, Mapping) for row in value):
        return [], False
    return [dict(row) for row in value], True


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _digest(value: Any) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value or "").strip().lower()))


def _normalized_digest(value: Any) -> str:
    digest = str(value or "").strip().lower()
    return digest.removeprefix("sha256:") if _SHA256_RE.fullmatch(digest) else ""


def _canonical_payload_sha256(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return ""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _valid_interval(value: Any, *, minimum: float, maximum: float) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return False
    bounds = [_number(item) for item in value]
    return bool(
        len(bounds) == 2
        and all(item is not None for item in bounds)
        and minimum <= bounds[0] <= bounds[1] <= maximum
    )


def _connected(policy_ids: Sequence[str], preferences: Sequence[Mapping[str, Any]]) -> bool:
    if not policy_ids:
        return False
    graph: dict[str, set[str]] = {policy_id: set() for policy_id in policy_ids}
    for row in preferences:
        left = str(row.get("policy_a") or "")
        right = str(row.get("policy_b") or "")
        if (
            row.get("outcome") in {"policy_a", "policy_b", "tie"}
            and left in graph
            and right in graph
            and left != right
        ):
            graph[left].add(right)
            graph[right].add(left)
    visited = {policy_ids[0]}
    queue = deque([policy_ids[0]])
    while queue:
        current = queue.popleft()
        for neighbor in graph[current] - visited:
            visited.add(neighbor)
            queue.append(neighbor)
    return visited == set(policy_ids)


def _bradley_terry(
    policy_ids: Sequence[str], preferences: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    index = {policy_id: offset for offset, policy_id in enumerate(policy_ids)}
    wins = [0.0] * len(policy_ids)
    games = [[0.0] * len(policy_ids) for _ in policy_ids]
    for row in preferences:
        left = index.get(str(row.get("policy_a") or ""))
        right = index.get(str(row.get("policy_b") or ""))
        if left is None or right is None or left == right:
            continue
        outcome = str(row.get("outcome") or "")
        if outcome not in {"policy_a", "policy_b", "tie"}:
            continue
        games[left][right] += 1.0
        games[right][left] += 1.0
        if outcome == "policy_a":
            wins[left] += 1.0
        elif outcome == "policy_b":
            wins[right] += 1.0
        elif outcome == "tie":
            wins[left] += 0.5
            wins[right] += 0.5
    ability = [1.0] * len(policy_ids)
    for _ in range(1000):
        updated = []
        for left in range(len(policy_ids)):
            denominator = sum(
                games[left][right] / max(1e-12, ability[left] + ability[right])
                for right in range(len(policy_ids))
                if right != left
            )
            updated.append(wins[left] / denominator if denominator > 0 else ability[left])
        scale = sum(updated) / max(1, len(updated))
        updated = [max(1e-12, value / max(scale, 1e-12)) for value in updated]
        if max(abs(a - b) for a, b in zip(ability, updated)) < 1e-10:
            ability = updated
            break
        ability = updated
    ordered = sorted(zip(policy_ids, ability), key=lambda item: (-item[1], item[0]))
    return [
        {"policy_id": policy_id, "ability": round(score, 9), "rank": rank}
        for rank, (policy_id, score) in enumerate(ordered, 1)
    ]


def build_decision_grade_ranking(request: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if request.get("schema_version") != SCHEMA_VERSION:
        blockers.append("decision_grade_ranking_schema_missing_or_unsupported")
    design = request.get("evaluation_design")
    design_validation = validate_policy_evaluation_design(
        design if isinstance(design, Mapping) else {}
    )
    blockers.extend(f"evaluation_design:{item}" for item in design_validation["blockers"])
    policies = _rows((design if isinstance(design, Mapping) else {}).get("policies"))
    policy_ids = [str(row.get("policy_id") or "") for row in policies]

    minimum_confidence = _number(request.get("minimum_calibrated_judge_confidence"))
    if minimum_confidence is None or not 0.0 <= minimum_confidence <= 1.0:
        blockers.append("minimum_calibrated_judge_confidence_missing_or_invalid")
        minimum_confidence = 1.0
    if not _digest(request.get("judge_calibration_set_sha256")):
        blockers.append("judge_calibration_set_digest_missing")
    if request.get("judge_calibration_status") != "accepted":
        blockers.append("judge_calibration_not_independently_accepted")
    if request.get("label_authority_independent_of_policy_and_model") is not True:
        blockers.append("label_authority_not_independent")

    results = _rows(request.get("episode_results"))
    design_rows_by_key = {
        (
            str(row.get("policy_id") or ""),
            str(row.get("site_id") or ""),
            str(row.get("task_id") or ""),
            str(row.get("condition_id") or ""),
            row.get("seed"),
        ): row
        for row in _rows((design if isinstance(design, Mapping) else {}).get("rows"))
    }
    expected_result_keys = set(design_rows_by_key)
    observed_result_keys: set[tuple[Any, ...]] = set()
    outcomes_by_policy: dict[str, list[float | None]] = defaultdict(list)
    outcomes_by_policy_cell: dict[str, dict[tuple[Any, ...], float | None]] = defaultdict(dict)
    outcomes_by_policy_condition: dict[tuple[str, str, str, str], list[float | None]] = defaultdict(
        list
    )
    failures_by_policy: dict[str, list[str]] = defaultdict(list)
    for row_index, row in enumerate(results):
        policy_id = str(row.get("policy_id") or "")
        if policy_id not in policy_ids:
            blockers.append(f"episode_result_unknown_policy:{row_index}")
        result_key = (
            policy_id,
            str(row.get("site_id") or ""),
            str(row.get("task_id") or ""),
            str(row.get("condition_id") or ""),
            row.get("seed"),
        )
        if result_key in observed_result_keys:
            blockers.append(f"duplicate_episode_result_cell:{row_index}")
        observed_result_keys.add(result_key)
        if row.get("full_ordered_episode_evidence") is not True:
            blockers.append(f"episode_result_not_full_ordered_episode:{row_index}")
        if not _digest(row.get("episode_evidence_sha256")):
            blockers.append(f"episode_result_evidence_digest_missing:{row_index}")
        if row.get("artifact_freshness_status") != "current":
            blockers.append(f"episode_result_artifact_not_current:{row_index}")
        if row.get("fresh_evaluator_model_execution_proven") is not True:
            blockers.append(f"episode_result_fresh_evaluator_execution_not_proven:{row_index}")
        fresh_evaluator_model_run_steps = row.get("fresh_evaluator_model_run_steps")
        if (
            isinstance(fresh_evaluator_model_run_steps, bool)
            or not isinstance(fresh_evaluator_model_run_steps, int)
            or fresh_evaluator_model_run_steps <= 0
        ):
            blockers.append(f"episode_result_fresh_evaluator_steps_invalid:{row_index}")
        if row.get("fixture_or_proxy_model_output_used") is not False:
            blockers.append(f"episode_result_fixture_or_proxy_not_blocked:{row_index}")
        if row.get("fallback_policy_used") is not False:
            blockers.append(f"episode_result_fallback_policy_not_blocked:{row_index}")
        design_row = design_rows_by_key.get(result_key, {})
        if (
            isinstance(fresh_evaluator_model_run_steps, int)
            and not isinstance(fresh_evaluator_model_run_steps, bool)
            and fresh_evaluator_model_run_steps > 0
            and fresh_evaluator_model_run_steps != design_row.get("fresh_evaluator_model_run_steps")
        ):
            blockers.append(f"episode_result_fresh_evaluator_steps_mismatch:{row_index}")
        evaluator_profile_id = str(row.get("evaluator_profile_id") or "").strip()
        design_evaluator_profile_id = str(design_row.get("evaluator_profile_id") or "").strip()
        if not evaluator_profile_id:
            blockers.append(f"episode_result_evaluator_profile_missing:{row_index}")
        elif evaluator_profile_id != design_evaluator_profile_id:
            blockers.append(f"episode_result_evaluator_profile_mismatch:{row_index}")
        design_backend = (
            design_row.get("evaluator_backend")
            if isinstance(design_row.get("evaluator_backend"), Mapping)
            else {}
        )
        evaluator_backend_id = str(row.get("evaluator_backend_id") or "").strip()
        design_evaluator_backend_id = str(design_backend.get("backend_id") or "").strip()
        if not evaluator_backend_id:
            blockers.append(f"episode_result_evaluator_backend_missing:{row_index}")
        elif evaluator_backend_id != design_evaluator_backend_id:
            blockers.append(f"episode_result_evaluator_backend_mismatch:{row_index}")
        if row.get("authoritative_manifest_status") != "completed":
            blockers.append(f"episode_result_authoritative_manifest_not_completed:{row_index}")
        if row.get("infrastructure_status") != "succeeded":
            blockers.append(f"episode_result_infrastructure_not_succeeded:{row_index}")
        evaluator_outcome_status = row.get("evaluator_outcome_status")
        if evaluator_outcome_status not in {"valid", "abstained"}:
            blockers.append(f"episode_result_evaluator_outcome_invalid:{row_index}")
        elif evaluator_outcome_status != design_row.get("evaluator_outcome_status"):
            blockers.append(f"episode_result_evaluator_outcome_mismatch:{row_index}")
        for field in required_evaluator_evidence_digest_fields(evaluator_profile_id):
            if not _digest(row.get(field)):
                blockers.append(f"episode_result_chain_digest_missing:{row_index}:{field}")
            elif _normalized_digest(row.get(field)) != _normalized_digest(design_row.get(field)):
                blockers.append(f"episode_result_chain_digest_mismatch:{row_index}:{field}")
        criteria, criteria_payload_valid = _strict_rows(row.get("criterion_results"))
        if not criteria_payload_valid:
            blockers.append(f"criterion_results_payload_invalid:{row_index}")
        elif not criteria:
            blockers.append(f"episode_result_criteria_missing:{row_index}")
        elif _normalized_digest(row.get("criterion_result_sha256")) != (
            _canonical_payload_sha256(criteria)
        ):
            blockers.append(f"criterion_result_payload_digest_mismatch:{row_index}")
        episode_values: list[float] = []
        episode_abstained = evaluator_outcome_status == "abstained"
        for criterion_index, criterion in enumerate(criteria):
            outcome = str(criterion.get("outcome") or "")
            confidence = _number(criterion.get("confidence"))
            if outcome not in {"success", "failure", "abstain", "inconclusive"}:
                blockers.append(f"criterion_outcome_invalid:{row_index}:{criterion_index}")
                continue
            if confidence is None or not 0.0 <= confidence <= 1.0:
                blockers.append(
                    f"criterion_confidence_missing_or_invalid:{row_index}:{criterion_index}"
                )
            if confidence is not None and confidence < minimum_confidence and outcome != "abstain":
                blockers.append(
                    f"low_confidence_criterion_must_abstain:{row_index}:{criterion_index}"
                )
            if criterion.get("label_blinded_and_randomized") is not True:
                blockers.append(
                    f"criterion_label_not_blinded_randomized:{row_index}:{criterion_index}"
                )
            evidence_refs, evidence_payload_valid = _strict_rows(criterion.get("evidence_refs"))
            if not evidence_payload_valid:
                blockers.append(f"criterion_evidence_payload_invalid:{row_index}:{criterion_index}")
            elif not evidence_refs:
                blockers.append(f"criterion_evidence_missing:{row_index}:{criterion_index}")
            elif any(not _digest(ref.get("sha256")) for ref in evidence_refs):
                blockers.append(f"criterion_evidence_digest_invalid:{row_index}:{criterion_index}")
            if outcome in {"abstain", "inconclusive"}:
                episode_abstained = True
            elif outcome == "success":
                if evaluator_outcome_status == "abstained":
                    blockers.append(
                        f"abstained_evaluator_cannot_emit_decided_criterion:{row_index}:{criterion_index}"
                    )
                episode_values.append(1.0)
            else:
                if evaluator_outcome_status == "abstained":
                    blockers.append(
                        f"abstained_evaluator_cannot_emit_decided_criterion:{row_index}:{criterion_index}"
                    )
                episode_values.append(0.0)
                taxonomy = [str(item) for item in criterion.get("failure_taxonomy", []) or []]
                if not taxonomy:
                    blockers.append(
                        f"criterion_failure_taxonomy_missing:{row_index}:{criterion_index}"
                    )
                failures_by_policy[policy_id].extend(taxonomy)
        episode_outcome = (
            None
            if episode_abstained or not episode_values
            else float(all(value == 1.0 for value in episode_values))
        )
        outcomes_by_policy[policy_id].append(episode_outcome)
        matched_cell_key = result_key[1:]
        outcomes_by_policy_cell[policy_id][matched_cell_key] = episode_outcome
        outcomes_by_policy_condition[result_key[:4]].append(episode_outcome)
    if observed_result_keys != expected_result_keys:
        blockers.append("episode_results_do_not_exactly_cover_registered_matched_cells")
    expected_policy_conditions = {key[:4] for key in expected_result_keys}
    for policy_condition in sorted(expected_policy_conditions):
        decided_count = sum(
            outcome is not None
            for outcome in outcomes_by_policy_condition.get(policy_condition, [])
        )
        if decided_count < MINIMUM_MATCHED_REPLICATES_PER_POLICY_CONDITION:
            blockers.append(
                "decided_outcome_count_below_minimum:"
                + ":".join(policy_condition)
                + f":{decided_count}<{MINIMUM_MATCHED_REPLICATES_PER_POLICY_CONDITION}"
            )

    preferences = _rows(request.get("pairwise_preferences"))
    for index, row in enumerate(preferences):
        if row.get("label_blinded_and_randomized") is not True:
            blockers.append(f"pairwise_label_not_blinded_randomized:{index}")
        if row.get("outcome") not in {"policy_a", "policy_b", "tie", "inconclusive"}:
            blockers.append(f"pairwise_outcome_invalid:{index}")
        evidence_refs = _rows(row.get("evidence_refs"))
        if not evidence_refs:
            blockers.append(f"pairwise_evidence_missing:{index}")
        elif any(not _digest(ref.get("sha256")) for ref in evidence_refs):
            blockers.append(f"pairwise_evidence_digest_invalid:{index}")
    graph_connected = _connected(policy_ids, preferences)
    if not graph_connected:
        blockers.append("bradley_terry_preference_graph_not_connected")

    bootstrap = request.get("bootstrap") if isinstance(request.get("bootstrap"), Mapping) else {}
    seed = bootstrap.get("seed")
    replicate_count = bootstrap.get("replicate_count")
    if isinstance(seed, bool) or not isinstance(seed, int):
        blockers.append("bootstrap_seed_missing_or_invalid")
        seed = 0
    if replicate_count != BOOTSTRAP_REPLICATES:
        blockers.append("bootstrap_replicate_count_must_equal_10000")
        replicate_count = BOOTSTRAP_REPLICATES
    if bootstrap.get("method") != BOOTSTRAP_METHOD:
        blockers.append("bootstrap_method_missing_or_mismatch")

    rng = random.Random(seed)
    matched_cell_keys = sorted({key[1:] for key in expected_result_keys}, key=repr)
    bootstrap_samples_by_policy: dict[str, list[float]] = defaultdict(list)
    if matched_cell_keys:
        for _ in range(replicate_count):
            sampled_cells = [rng.choice(matched_cell_keys) for _ in matched_cell_keys]
            for policy_id in policy_ids:
                sampled_outcomes = [
                    outcomes_by_policy_cell.get(policy_id, {}).get(cell_key)
                    for cell_key in sampled_cells
                ]
                observed_sample = [value for value in sampled_outcomes if value is not None]
                if observed_sample:
                    bootstrap_samples_by_policy[policy_id].append(
                        sum(observed_sample) / len(observed_sample)
                    )
    score_rows: list[dict[str, Any]] = []
    for policy_id in policy_ids:
        outcomes = outcomes_by_policy.get(policy_id, [])
        observed = [value for value in outcomes if value is not None]
        samples = bootstrap_samples_by_policy.get(policy_id, [])
        score_rows.append(
            {
                "policy_id": policy_id,
                "success_rate": round(sum(observed) / len(observed), 6) if observed else None,
                "coverage": round(len(observed) / len(outcomes), 6) if outcomes else 0.0,
                "abstention_rate": round(1.0 - len(observed) / len(outcomes), 6)
                if outcomes
                else 1.0,
                "success_rate_95_ci": [
                    _percentile(samples, 0.025),
                    _percentile(samples, 0.975),
                ],
                "failure_taxonomy": sorted(set(failures_by_policy.get(policy_id, []))),
            }
        )

    required_ood_axes = {"site", "task", "embodiment", "viewpoint", "appearance"}
    ood_rows = _rows(request.get("ood_axis_results"))
    observed_ood_axes = {str(row.get("axis") or "") for row in ood_rows}
    if observed_ood_axes != required_ood_axes:
        blockers.append("ood_axis_results_missing_or_mismatched")
    if len(ood_rows) != len(observed_ood_axes):
        blockers.append("ood_axis_results_duplicate_axis")
    for index, row in enumerate(ood_rows):
        coverage = _number(row.get("coverage"))
        abstention = _number(row.get("abstention_rate"))
        sample_count = row.get("sample_count")
        if coverage is None or not 0.0 <= coverage <= 1.0:
            blockers.append(f"ood_axis_coverage_missing_or_invalid:{index}")
        if abstention is None or not 0.0 <= abstention <= 1.0:
            blockers.append(f"ood_axis_abstention_missing_or_invalid:{index}")
        if (
            coverage is not None
            and abstention is not None
            and abs(coverage + abstention - 1.0) > 1e-6
        ):
            blockers.append(f"ood_axis_coverage_abstention_mismatch:{index}")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
            blockers.append(f"ood_axis_sample_count_missing_or_invalid:{index}")
        if not _valid_interval(row.get("coverage_95_ci"), minimum=0.0, maximum=1.0):
            blockers.append(f"ood_axis_coverage_ci_missing_or_invalid:{index}")
        if not _valid_interval(row.get("abstention_95_ci"), minimum=0.0, maximum=1.0):
            blockers.append(f"ood_axis_abstention_ci_missing_or_invalid:{index}")
        if not isinstance(row.get("failure_taxonomy"), Mapping):
            blockers.append(f"ood_axis_failure_taxonomy_missing:{index}")
        if not _digest(row.get("split_manifest_sha256")):
            blockers.append(f"ood_axis_split_manifest_digest_missing:{index}")

    anchor_rows = _rows(request.get("accepted_external_anchor_rows"))
    anchor_status = "correlation_not_measured"
    if anchor_rows:
        blockers.append("accepted_anchor_rows_require_frozen_calibration_recomputation")
    blockers = sorted(set(blockers))
    return {
        "schema_version": "decision_grade_ranking.v2",
        "status": "decision_grade" if not blockers else "blocked",
        "decision_grade": not blockers,
        "policy_scorecards": score_rows,
        "bradley_terry": {
            "graph_connected": graph_connected,
            "ranking": _bradley_terry(policy_ids, preferences) if graph_connected else [],
            "ties_retained": True,
        },
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "seed": seed,
            "replicate_count": replicate_count,
            "matched_cells_resampled_jointly_across_policies": True,
        },
        "correlation_status": anchor_status,
        "pearson": None,
        "spearman": None,
        "mmrv": None,
        "blockers": blockers,
        "claim_boundary": {
            "simulator_ranking_is_not_real_world_ordering": True,
            "evaluator_profile_is_not_compute_provider_identity": True,
            "oscar_and_sc3_are_optional_versioned_evaluator_profiles": True,
            "paper_metrics_are_never_inherited": True,
            "correlation_requires_independently_accepted_anchor_rows": True,
        },
    }
