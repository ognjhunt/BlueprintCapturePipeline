"""Frozen calibration and captured-site contracts for the policy-ranking thesis.

This module is deliberately model-neutral.  It binds public benchmark and WAM
artifacts, keeps benchmark outcomes out of evaluator requests, freezes a split
before scoring, and computes ranking/abstention metrics only after judge outputs
exist.  It does not run a physical robot or promote prospective site rankings to
site-specific physical evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .common import write_json


PREREGISTRATION_SCHEMA = "policy_ranking_thesis_preregistration.v2"
ROLLOUT_INDEX_SCHEMA = "frozen_wam_rollout_index.v1"
JUDGE_RESULT_SCHEMA = "policy_ranking_episode_judgment.v1"
CALIBRATION_REPORT_SCHEMA = "policy_ranking_frozen_calibration_report.v1"
HYBRID_SCENE_SCHEMA = "hybrid_3dgs_policy_ranking_scene.v1"
CONTROLLED_SCENE_SCHEMA = "controlled_usd_policy_ranking_scene.v1"

DEFAULT_POLICIES = (
    "paligemma_binning_droid",
    "paligemma_diffusion_droid",
    "paligemma_fast_droid",
    "paligemma_fast_specialist_droid",
    "paligemma_vq_droid",
    "pi0_droid",
    "pi0_fast_droid",
)
PARTITION_COUNTS = {"pilot": 7, "calibration": 7, "heldout": 49}
SPLIT_SALT = "blueprint-policy-ranking-thesis-2026-07-26-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_sha(value: Any) -> str:
    text = str(value or "").lower().removeprefix("sha256:")
    return text if _SHA256.fullmatch(text) else ""


def parse_lfs_pointer(path: str | Path) -> dict[str, Any]:
    """Parse a Git-LFS pointer without downloading the represented object."""

    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    lines = dict(
        line.split(" ", 1) for line in text.splitlines() if " " in line
    )
    oid = str(lines.get("oid") or "").removeprefix("sha256:")
    size = int(lines.get("size") or 0)
    if lines.get("version") != "https://git-lfs.github.com/spec/v1":
        raise ValueError(f"not_git_lfs_pointer:{resolved}")
    if not _strict_sha(oid) or size <= 0:
        raise ValueError(f"invalid_git_lfs_pointer:{resolved}")
    return {"sha256": oid, "size_bytes": size}


def _session_order(session_id: str) -> str:
    return hashlib.sha256(f"{SPLIT_SALT}:{session_id}".encode()).hexdigest()


def build_preregistration(
    session_ids: Sequence[str],
    *,
    policy_ids: Sequence[str] = DEFAULT_POLICIES,
) -> dict[str, Any]:
    """Freeze a falsification-oriented protocol over the released OSCAR pool."""

    sessions = sorted({str(item) for item in session_ids}, key=_session_order)
    policies = sorted({str(item) for item in policy_ids})
    if len(sessions) != sum(PARTITION_COUNTS.values()):
        raise ValueError(
            f"expected_{sum(PARTITION_COUNTS.values())}_complete_sessions:got_{len(sessions)}"
        )
    if policies != sorted(DEFAULT_POLICIES):
        raise ValueError("policy_set_drift")

    cursor = 0
    partitions: dict[str, list[str]] = {}
    for name, count in PARTITION_COUNTS.items():
        partitions[name] = sessions[cursor : cursor + count]
        cursor += count

    protocol: dict[str, Any] = {
        "schema_version": PREREGISTRATION_SCHEMA,
        "protocol_id": "blueprint_franka_roboarena_oscar_v2",
        "frozen": True,
        "split_salt": SPLIT_SALT,
        "hypotheses": {
            "null": (
                "The frozen WAM-plus-independent-evaluator does not order the seven "
                "RoboArena policies better than chance or the strongest frozen cheap baseline, "
                "and abstention does not improve reliability."
            ),
            "alternative": (
                "The frozen stack orders the policies better than chance and the strongest "
                "cheap baseline, with improved selective reliability, then emits a prospective "
                "ranking on an unseen 3DGS site without policy-specific evaluator changes."
            ),
        },
        "benchmark": {
            "dataset": "RoboArena/DataDump_07-17-2026",
            "revision": "7931db81f3f6a48a3245427f7213a4c461f92ccc",
            "label_source": "independent_third_party_real_robot_policy_execution",
            "label_granularity": "per_session_per_policy_binary_and_partial_plus_session_preference",
            "pii_fields_forbidden": ["evaluator_name", "evaluator_email", "evaluation_location"],
        },
        "wam": {
            "family": "OSCAR-2B",
            "source_revision": "4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb",
            "checkpoint_revision": "c9781ffa7dd8556d862d7d9f338a2ea008a58ca6",
            "released_rollout_revision": "db5edfaef285c15d0a41d5115177a983c08b4f5f",
            "released_rollout_count": 441,
            "paper_pool_claim": 455,
            "unresolved_count_contradiction": 14,
            "released_subset_selection_process": "not_documented",
            "selection_bias_caveat": (
                "The 441 released rollouts are an author-provided subset of the paper's "
                "claimed 455; representativeness cannot be assumed."
            ),
            "generated_half": "left_half_only",
            "third_party_physical_half_forbidden_to_evaluator": True,
        },
        "policies": policies,
        "partitions": partitions,
        "partition_counts": dict(PARTITION_COUNTS),
        "evaluator": {
            "profile": "roboworld_progress_v1",
            "independent_of_wam_and_candidate_policies": True,
            "input": "task_instruction_plus_32_uniform_frames_from_generated_half_only",
            "output_schema": JUDGE_RESULT_SCHEMA,
            "full_temporal_method": "frozen_temporal_32_frame_progress_judge_v1",
            "cheap_baseline_method": "frozen_first_last_frame_progress_judge_v1",
            "no_benchmark_labels_in_request": True,
            "provider_configuration": {
                "model_snapshot": "gpt-5-2025-08-07",
                "reasoning_effort": "high",
                "max_output_tokens_including_reasoning": 8192,
                "temperature": "not_requested_model_default",
                "top_p": "not_requested_model_default",
                "seed": "not_supported_by_this_responses_configuration",
                "image_detail": "low",
                "strict_json_schema": True,
                "store": False,
            },
            "retry_rule": {
                "maximum_exact_attempts_per_request": 2,
                "configuration_change_between_attempts_forbidden": True,
                "accept_first_valid_response_only": True,
                "all_attempt_usage_preserved": True,
                "exhausted_request_result": "explicit_abstention_and_inconclusive_if_required_matrix_incomplete",
            },
        },
        "power_analysis": {
            "artifact": "power_analysis.json",
            "analysis_unit": "heldout_session_cluster",
            "heldout_session_count": 49,
            "within_session_pairs_treated_as_independent": False,
            "one_sided_alpha": 0.05,
            "target_power": 0.80,
            "null_accuracy": 0.50,
            "conservative_minimum_detectable_accuracy": 0.6776,
            "exact_binomial_reference_minimum_accuracy": 0.6783,
            "sample_size_basis": "all_released_complete_sessions_remaining_after_pilot_and_calibration",
            "small_effect_or_wide_interval_disposition": "inconclusive",
            "adjacent_pairs_exempt_from_decision_rule": False,
        },
        "experimental_lanes": {
            "lane_a_frozen_benchmark": {
                "rollout_mode": "author_generated_open_loop_action_replay",
                "answer_key": "independent_frozen_roboarena_real_policy_outcomes",
            },
            "lane_b_controlled_bridge": {
                "required_before_lane_c_claim": True,
                "scene": "nvidia_physicalai_simready_warehouse_01",
                "rollout_mode": "blueprint_operated_closed_loop_policy_in_the_loop",
                "evaluator": "deterministic_object_state_predicates_plus_frozen_visual_consistency_check",
                "independent_physical_answer_key": False,
                "claim": "closed_loop_execution_and_internal_simulator_outcome_bridge_only",
            },
            "lane_c_captured_site": {
                "rollout_mode": "blueprint_operated_closed_loop_policy_in_the_loop",
                "claim": "prospective_externally_calibrated_ranking_only",
            },
            "lane_a_does_not_validate_lane_b_or_c_closed_loop_behavior": True,
        },
        "thresholds": {
            "episode_judge_confidence_min": 0.65,
            "selective_judge_confidence_min": 0.80,
            "action_following_confidence_min": 0.70,
            "temporal_coherence_confidence_min": 0.70,
            "pair_score_margin_min": 0.20,
            "bootstrap_replicates": 10_000,
            "confidence_level": 0.95,
            "minimum_selective_coverage": 0.25,
            "minimum_selective_accuracy_gain": 0.05,
        },
        "exclusions": {
            "allowed": [
                "missing_released_rollout_object",
                "corrupt_video_decode",
                "metadata_missing_before_freeze",
                "prohibited_third_party_physical_pixels_not_removed",
            ],
            "all_exclusions_remain_in_ledger": True,
            "replacement_forbidden": True,
        },
        "abstention": {
            "required_if": [
                "judge_confidence_below_threshold",
                "action_following_below_threshold",
                "temporal_coherence_below_threshold",
                "critical_contradiction",
                "missing_or_nonfinite_score",
            ],
            "abstention_is_not_success": True,
        },
        "metrics": [
            "session_pairwise_accuracy",
            "kendall_tau_b",
            "spearman_rank_correlation",
            "top_policy_selection",
            "top_policy_regret",
            "brier_score",
            "selective_accuracy_and_coverage",
            "bootstrap_confidence_intervals",
            "false_success_rate",
            "false_failure_rate",
            "cost_and_wall_time",
        ],
        "decision_rule": {
            "thesis_supported_requires_all": [
                "heldout_pairwise_accuracy_bootstrap_lower_bound_gt_0_5",
                "heldout_kendall_tau_gt_0",
                "full_temporal_pairwise_accuracy_gt_strongest_frozen_cheap_baseline",
                "selective_accuracy_gain_gte_0_05_at_coverage_gte_0_25",
                "heldout_action_following_pass_rate_gte_0_80",
                "blueprint_operated_closed_loop_warehouse_bridge_passes_frozen_deterministic_predicates_without_policy_specific_scoring",
                "identical_evaluator_digest_used_for_unseen_3dgs_transfer",
                "hybrid_scene_keeps_3dgs_visual_source_and_local_interaction_assets_separate",
                "prospective_transfer_ranks_at_least_four_policies_without_policy_specific_scoring",
                "cost_and_wall_time_measured",
                "complete_provenance",
            ],
            "thesis_not_supported_if_any": [
                "heldout_ranking_does_not_beat_strongest_cheap_baseline",
                "abstention_fails_selective_gain_gate",
                "heldout_action_following_pass_rate_lt_0_80",
                "data_leakage_detected",
                "transfer_requires_policy_specific_evaluator_changes",
                "closed_loop_bridge_requires_policy_specific_tuning_or_cannot_execute",
                "cost_advantage_eliminated",
            ],
            "inconclusive_if_any_required_component_unmeasured": True,
        },
        "claim_boundary": {
            "benchmark_result_type": "externally_verified_real_policy_ranking_fidelity",
            "captured_site_result_type": "prospective_policy_ranking",
            "captured_site_physical_success_proven": False,
            "captured_site_policy_ordering_proven": False,
            "blueprint_operated_physical_robot": False,
            "open_loop_calibration_proves_blueprint_closed_loop_wam_behavior": False,
            "warehouse_deterministic_predicates_are_independent_physical_truth": False,
        },
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    return protocol


def discover_released_rollouts(root: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    resolved = Path(root).resolve()
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for path in sorted(resolved.glob("*/*/left/compare_overlay_vs_gt.mp4")):
        session_id = path.parts[-4]
        policy_id = path.parts[-3]
        try:
            pointer = parse_lfs_pointer(path)
            artifact_kind = "git_lfs_pointer"
        except (UnicodeDecodeError, ValueError):
            pointer = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
            artifact_kind = "materialized_video"
        rows.append(
            {
                "session_id": session_id,
                "policy_id": policy_id,
                "relative_path": path.relative_to(resolved).as_posix(),
                "artifact_kind": artifact_kind,
                **pointer,
                "contains_generated_and_third_party_physical_halves": True,
                "evaluator_crop": {"x_start_fraction": 0.0, "x_end_fraction": 0.5},
            }
        )
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[row["session_id"]].add(row["policy_id"])
    expected = set(DEFAULT_POLICIES)
    for session_id, policies in grouped.items():
        if policies != expected:
            blockers.append(f"incomplete_policy_coverage:{session_id}")
    if len(rows) != 441:
        blockers.append(f"released_rollout_count:{len(rows)}")
    if len(grouped) != 63:
        blockers.append(f"released_session_count:{len(grouped)}")
    return rows, sorted(set(blockers))


def _redacted_metadata(path: Path, policies: Sequence[str]) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"metadata_not_mapping:{path}")
    policy_rows = payload.get("policies")
    names = {
        str(row.get("policy_name"))
        for row in policy_rows
        if isinstance(row, Mapping) and row.get("policy_name")
    } if isinstance(policy_rows, list) else set()
    missing = sorted(set(policies) - names)
    return {
        "language_instruction": str(payload.get("language_instruction") or "").strip(),
        "metadata_sha256": file_sha256(path),
        "policy_coverage_complete": not missing,
        "missing_policies": missing,
    }


def build_rollout_index(
    rollout_root: str | Path,
    roboarena_root: str | Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    rows, blockers = discover_released_rollouts(rollout_root)
    benchmark_root = Path(roboarena_root).resolve()
    metadata_cache: dict[str, dict[str, Any]] = {}
    policies = [str(item) for item in protocol.get("policies", [])]
    for session_id in sorted({row["session_id"] for row in rows}):
        path = benchmark_root / "evaluation_sessions" / session_id / "metadata.yaml"
        if not path.is_file():
            blockers.append(f"metadata_missing:{session_id}")
            continue
        metadata_cache[session_id] = _redacted_metadata(path, policies)
    indexed = []
    for row in rows:
        metadata = metadata_cache.get(row["session_id"], {})
        indexed.append(
            {
                **row,
                **metadata,
                "benchmark_labels_included": False,
                "pii_included": False,
            }
        )
    result: dict[str, Any] = {
        "schema_version": ROLLOUT_INDEX_SCHEMA,
        "status": "ready" if not blockers else "blocked",
        "protocol_sha256": protocol.get("protocol_sha256"),
        "row_count": len(indexed),
        "session_count": len({row["session_id"] for row in indexed}),
        "policy_count": len({row["policy_id"] for row in indexed}),
        "rows": indexed,
        "blockers": sorted(set(blockers)),
        "privacy": {
            "source_metadata_contains_pii": True,
            "output_allowlist": ["language_instruction", "metadata_sha256", "policy_coverage"],
            "evaluator_name_omitted": True,
            "evaluator_email_omitted": True,
            "evaluation_location_omitted": True,
        },
    }
    result["index_sha256"] = canonical_sha256(result)
    return result


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate_judgment(row: Mapping[str, Any], protocol: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if row.get("schema_version") != JUDGE_RESULT_SCHEMA:
        blockers.append("schema_version")
    if str(row.get("method") or "") not in {
        protocol["evaluator"]["full_temporal_method"],
        protocol["evaluator"]["cheap_baseline_method"],
    }:
        blockers.append("method")
    if str(row.get("policy_id") or "") not in set(protocol["policies"]):
        blockers.append("policy_id")
    for field in (
        "success_probability",
        "judge_confidence",
        "action_following_confidence",
        "temporal_coherence_confidence",
    ):
        number = _number(row.get(field))
        if number is None or not 0.0 <= number <= 1.0:
            blockers.append(field)
    if not _strict_sha(row.get("evaluator_digest")):
        blockers.append("evaluator_digest")
    if row.get("benchmark_labels_seen") is not False:
        blockers.append("benchmark_label_leakage")
    if row.get("third_party_physical_pixels_seen") is not False:
        blockers.append("physical_half_leakage")
    if not isinstance(row.get("abstained"), bool):
        blockers.append("abstained")
    return blockers


def _rank(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda item: (values[item], item))
    ranks: dict[str, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for item in ordered[cursor:end]:
            ranks[item] = rank
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    lm, rm = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - lm) ** 2 for a in left) * sum((b - rm) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def _spearman(predicted: Mapping[str, float], actual: Mapping[str, float]) -> float | None:
    policies = sorted(set(predicted) & set(actual))
    pr, ar = _rank(predicted), _rank(actual)
    return _pearson([pr[p] for p in policies], [ar[p] for p in policies])


def _kendall_tau_b(predicted: Mapping[str, float], actual: Mapping[str, float]) -> float | None:
    policies = sorted(set(predicted) & set(actual))
    concordant = discordant = pred_ties = actual_ties = 0
    for index, left in enumerate(policies):
        for right in policies[index + 1 :]:
            pd = predicted[left] - predicted[right]
            ad = actual[left] - actual[right]
            if pd == 0 and ad == 0:
                continue
            if pd == 0:
                pred_ties += 1
            elif ad == 0:
                actual_ties += 1
            elif pd * ad > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + pred_ties)
        * (concordant + discordant + actual_ties)
    )
    return (concordant - discordant) / denominator if denominator else None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _benchmark_outcomes(metadata_path: Path) -> dict[str, dict[str, float]]:
    payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    outcomes: dict[str, dict[str, float]] = {}
    for row in payload.get("policies", []):
        if not isinstance(row, Mapping):
            continue
        policy_id = str(row.get("policy_name") or "")
        if policy_id in DEFAULT_POLICIES:
            outcomes[policy_id] = {
                "binary_success": float(bool(row.get("binary_success"))),
                "partial_success": float(row.get("partial_success") or 0.0),
            }
    return outcomes


def _episode_rows(
    judgments: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    partition: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    allowed_sessions = set(protocol["partitions"][partition])
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in judgments:
        row = dict(raw)
        if row.get("session_id") not in allowed_sessions:
            continue
        validation = validate_judgment(row, protocol)
        if validation:
            blockers.extend(
                f"judgment:{row.get('session_id')}:{row.get('policy_id')}:{item}"
                for item in validation
            )
            continue
        key = (str(row["session_id"]), str(row["policy_id"]), str(row["method"]))
        if key in seen:
            blockers.append("duplicate_judgment:" + ":".join(key))
            continue
        seen.add(key)
        rows.append(row)
    return rows, blockers


def evaluate_frozen_calibration(
    judgments: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    roboarena_root: str | Path,
    partition: str = "heldout",
    expected_evaluator_digest: str | None = None,
) -> dict[str, Any]:
    """Join frozen predictions to labels and compute preregistered metrics."""

    if partition not in {"pilot", "calibration", "heldout"}:
        raise ValueError("invalid_partition")
    rows, blockers = _episode_rows(judgments, protocol, partition)
    evaluator_digests = sorted({str(row["evaluator_digest"]) for row in rows})
    if len(evaluator_digests) != 1:
        blockers.append(f"evaluator_digest_count:{len(evaluator_digests)}")
    if expected_evaluator_digest is not None and evaluator_digests != [expected_evaluator_digest]:
        blockers.append("evaluator_digest_mismatch")
    methods = (
        protocol["evaluator"]["full_temporal_method"],
        protocol["evaluator"]["cheap_baseline_method"],
    )
    expected = len(protocol["partitions"][partition]) * len(protocol["policies"])
    root = Path(roboarena_root).resolve()
    labels: dict[str, dict[str, dict[str, float]]] = {}
    for session_id in protocol["partitions"][partition]:
        labels[session_id] = _benchmark_outcomes(
            root / "evaluation_sessions" / session_id / "metadata.yaml"
        )

    by_method: dict[str, dict[str, Any]] = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != expected:
            blockers.append(f"{method}:expected_{expected}:got_{len(selected)}")
        predicted_by_policy: dict[str, list[float]] = defaultdict(list)
        actual_by_policy: dict[str, list[float]] = defaultdict(list)
        brier_terms: list[float] = []
        correctness: list[float] = []
        selective_correctness: list[float] = []
        selective_predicted_by_policy: dict[str, list[float]] = defaultdict(list)
        action_passes: list[float] = []
        false_success = false_failure = actual_failure = actual_success = 0
        thresholds = protocol["thresholds"]
        for row in selected:
            outcome = labels.get(row["session_id"], {}).get(row["policy_id"])
            if outcome is None:
                blockers.append(f"label_missing:{row['session_id']}:{row['policy_id']}")
                continue
            probability = float(row["success_probability"])
            actual = outcome["binary_success"]
            predicted_by_policy[row["policy_id"]].append(probability)
            actual_by_policy[row["policy_id"]].append(actual)
            brier_terms.append((probability - actual) ** 2)
            predicted_class = float(probability >= 0.5)
            correctness.append(float(predicted_class == actual))
            action_ok = (
                float(row["action_following_confidence"])
                >= thresholds["action_following_confidence_min"]
                and float(row["temporal_coherence_confidence"])
                >= thresholds["temporal_coherence_confidence_min"]
            )
            action_passes.append(float(action_ok))
            selective = (
                action_ok
                and float(row["judge_confidence"])
                >= thresholds["selective_judge_confidence_min"]
                and abs(probability - 0.5) >= thresholds["pair_score_margin_min"] / 2
                and not bool(row.get("critical_contradiction"))
                and not bool(row["abstained"])
            )
            if selective:
                selective_correctness.append(float(predicted_class == actual))
                selective_predicted_by_policy[row["policy_id"]].append(probability)
            if actual == 0:
                actual_failure += 1
                false_success += int(predicted_class == 1)
            else:
                actual_success += 1
                false_failure += int(predicted_class == 0)

        predicted_means = {
            policy: sum(values) / len(values) for policy, values in predicted_by_policy.items()
        }
        actual_means = {
            policy: sum(values) / len(values) for policy, values in actual_by_policy.items()
        }
        session_ids = list(protocol["partitions"][partition])
        rng = random.Random(20260726)
        bootstrap_accuracy: list[float] = []
        lookup = {(row["session_id"], row["policy_id"]): row for row in selected}

        def pairwise_accuracy(sample: Sequence[str]) -> float | None:
            right = 0.0
            total = 0
            for session_id in sample:
                session_labels = labels.get(session_id, {})
                for index, left_policy in enumerate(protocol["policies"]):
                    for right_policy in protocol["policies"][index + 1 :]:
                        left_label = session_labels.get(left_policy, {}).get("binary_success")
                        right_label = session_labels.get(right_policy, {}).get("binary_success")
                        if left_label is None or right_label is None or left_label == right_label:
                            continue
                        left_row = lookup.get((session_id, left_policy))
                        right_row = lookup.get((session_id, right_policy))
                        if left_row is None or right_row is None:
                            continue
                        delta = float(left_row["success_probability"]) - float(
                            right_row["success_probability"]
                        )
                        total += 1
                        if delta == 0:
                            right += 0.5
                        else:
                            right += float(delta * (left_label - right_label) > 0)
            return right / total if total else None

        observed_pairwise_accuracy = pairwise_accuracy(session_ids)
        replicates = int(thresholds["bootstrap_replicates"])
        for _ in range(replicates):
            sample = [rng.choice(session_ids) for _ in session_ids]
            replicate_accuracy = pairwise_accuracy(sample)
            if replicate_accuracy is not None:
                bootstrap_accuracy.append(replicate_accuracy)
        accuracy = sum(correctness) / len(correctness) if correctness else None
        selective_accuracy = (
            sum(selective_correctness) / len(selective_correctness)
            if selective_correctness
            else None
        )
        selective_predicted_means = {
            policy: sum(values) / len(values)
            for policy, values in selective_predicted_by_policy.items()
        }
        predicted_top = max(predicted_means, key=predicted_means.get) if predicted_means else None
        benchmark_top = max(actual_means, key=actual_means.get) if actual_means else None
        top_policy_regret = (
            actual_means[benchmark_top] - actual_means[predicted_top]
            if predicted_top in actual_means and benchmark_top in actual_means
            else None
        )
        by_method[method] = {
            "episode_count": len(selected),
            "predicted_policy_success": predicted_means,
            "selective_predicted_policy_success": selective_predicted_means,
            "selective_policy_episode_count": {
                policy: len(selective_predicted_by_policy.get(policy, []))
                for policy in protocol["policies"]
            },
            "benchmark_policy_success": actual_means,
            "spearman": _spearman(predicted_means, actual_means),
            "kendall_tau_b": _kendall_tau_b(predicted_means, actual_means),
            "episode_accuracy": accuracy,
            "brier_score": sum(brier_terms) / len(brier_terms) if brier_terms else None,
            "selective_accuracy": selective_accuracy,
            "selective_coverage": len(selective_correctness) / len(correctness) if correctness else 0.0,
            "selective_accuracy_gain": (
                selective_accuracy - accuracy
                if selective_accuracy is not None and accuracy is not None
                else None
            ),
            "action_following_pass_rate": (
                sum(action_passes) / len(action_passes) if action_passes else None
            ),
            "abstention_rate": (
                1.0 - len(selective_correctness) / len(correctness) if correctness else None
            ),
            "false_success_rate": false_success / actual_failure if actual_failure else None,
            "false_failure_rate": false_failure / actual_success if actual_success else None,
            "session_pairwise_accuracy_bootstrap_ci95": [
                _percentile(bootstrap_accuracy, 0.025),
                _percentile(bootstrap_accuracy, 0.975),
            ],
            "session_pairwise_accuracy": observed_pairwise_accuracy,
            "top_policy": predicted_top,
            "benchmark_top_policy": benchmark_top,
            "top_policy_regret": top_policy_regret,
        }

    full = by_method.get(methods[0], {})
    baseline = by_method.get(methods[1], {})
    gates = {
        "better_than_chance": bool(
            full.get("session_pairwise_accuracy_bootstrap_ci95")
            and full["session_pairwise_accuracy_bootstrap_ci95"][0] is not None
            and full["session_pairwise_accuracy_bootstrap_ci95"][0] > 0.5
        ),
        "positive_kendall": bool((full.get("kendall_tau_b") or 0.0) > 0.0),
        "beats_cheap_baseline": bool(
            full.get("session_pairwise_accuracy") is not None
            and baseline.get("session_pairwise_accuracy") is not None
            and full["session_pairwise_accuracy"] > baseline["session_pairwise_accuracy"]
        ),
        "abstention_improves": bool(
            (full.get("selective_coverage") or 0.0)
            >= protocol["thresholds"]["minimum_selective_coverage"]
            and (full.get("selective_accuracy_gain") or -1.0)
            >= protocol["thresholds"]["minimum_selective_accuracy_gain"]
        ),
        "action_following": bool((full.get("action_following_pass_rate") or 0.0) >= 0.80),
    }
    report: dict[str, Any] = {
        "schema_version": CALIBRATION_REPORT_SCHEMA,
        "status": "blocked" if blockers else "completed",
        "partition": partition,
        "protocol_sha256": protocol.get("protocol_sha256"),
        "evaluator_digest": evaluator_digests[0] if len(evaluator_digests) == 1 else None,
        "benchmark_outcomes_unsealed_after_predictions": True,
        "methods": by_method,
        "gates": gates,
        "benchmark_component_passed": not blockers and all(gates.values()),
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "externally_calibrated_against_frozen_real_policy_outcomes": True,
            "captured_site_accuracy_proven": False,
            "blueprint_conducted_physical_experiments": False,
        },
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def build_hybrid_scene_bundle(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the portable five-layer 3DGS plus local interaction contract."""

    required = (
        "scene_id",
        "site_visual",
        "site_spatial",
        "interactive_assets",
        "robot_runtime",
        "task_semantics",
    )
    blockers = [f"missing:{field}" for field in required if not spec.get(field)]
    visual = dict(spec.get("site_visual") or {})
    spatial = dict(spec.get("site_spatial") or {})
    assets = list(spec.get("interactive_assets") or [])
    robot = dict(spec.get("robot_runtime") or {})
    task = dict(spec.get("task_semantics") or {})
    if visual.get("representation") not in {"3dgs_ply", "splat", "spz", "ksplat"}:
        blockers.append("site_visual_not_3dgs")
    if not _strict_sha(visual.get("sha256")):
        blockers.append("site_visual_sha256")
    if spatial.get("is_complete_usd_rebuild") is not False:
        blockers.append("complete_usd_rebuild_forbidden")
    if len(assets) < 2:
        blockers.append("interactive_object_and_target_required")
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping) or not _strict_sha(asset.get("sha256")):
            blockers.append(f"interactive_asset_sha256:{index}")
        if not isinstance(asset, Mapping) or not asset.get("transform_site_m"):
            blockers.append(f"interactive_asset_transform:{index}")
    if robot.get("profile_id") != "franka_droid_fixed_base_v1":
        blockers.append("robot_profile")
    if not task.get("success_predicates"):
        blockers.append("success_predicates")
    if not task.get("abstention_rules"):
        blockers.append("abstention_rules")
    result: dict[str, Any] = {
        "schema_version": HYBRID_SCENE_SCHEMA,
        "status": "ready" if not blockers else "blocked",
        **dict(spec),
        "layer_identity": {
            "site_visual": canonical_sha256(visual),
            "site_spatial": canonical_sha256(spatial),
            "interactive_assets": canonical_sha256(assets),
            "robot_runtime": canonical_sha256(robot),
            "task_semantics": canonical_sha256(task),
            "optional_physics_sidecar": canonical_sha256(
                spec.get("optional_physics_sidecar") or {"status": "not_provided"}
            ),
        },
        "full_site_usd_rebuild_required": False,
        "policy_specific_evaluator_changes_allowed": False,
        "result_type": "prospective_policy_ranking",
        "site_specific_physical_accuracy_proven": False,
        "blockers": sorted(set(blockers)),
    }
    result["bundle_sha256"] = canonical_sha256(result)
    return result


def build_controlled_scene_bundle(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the controlled USD diagnostic without promoting it to an answer key."""

    required = ("scene_id", "environment", "interactive_assets", "robot_runtime", "task_semantics")
    blockers = [f"missing:{field}" for field in required if not spec.get(field)]
    environment = dict(spec.get("environment") or {})
    assets = list(spec.get("interactive_assets") or [])
    robot = dict(spec.get("robot_runtime") or {})
    task = dict(spec.get("task_semantics") or {})
    if environment.get("representation") not in {"usd", "usda", "usdc"}:
        blockers.append("controlled_environment_not_usd")
    if not _strict_sha(environment.get("sha256")):
        blockers.append("controlled_environment_sha256")
    if len(assets) < 2:
        blockers.append("interactive_object_and_target_required")
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping) or not _strict_sha(asset.get("sha256")):
            blockers.append(f"interactive_asset_sha256:{index}")
    if robot.get("profile_id") != "franka_droid_fixed_base_v1":
        blockers.append("robot_profile")
    if not task.get("success_predicates"):
        blockers.append("success_predicates")
    if task.get("physical_answer_key") is not False:
        blockers.append("controlled_scene_cannot_be_physical_answer_key")
    result: dict[str, Any] = {
        "schema_version": CONTROLLED_SCENE_SCHEMA,
        "status": "ready" if not blockers else "blocked",
        **dict(spec),
        "result_type": "controlled_simulation_diagnostic",
        "independent_physical_ranking_answer_key": False,
        "site_specific_physical_accuracy_proven": False,
        "blockers": sorted(set(blockers)),
    }
    result["bundle_sha256"] = canonical_sha256(result)
    return result


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prereg = sub.add_parser("preregister")
    prereg.add_argument("--rollout-root", required=True)
    prereg.add_argument("--output", required=True)
    index = sub.add_parser("index")
    index.add_argument("--rollout-root", required=True)
    index.add_argument("--roboarena-root", required=True)
    index.add_argument("--protocol", required=True)
    index.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--judgments", required=True)
    evaluate.add_argument("--roboarena-root", required=True)
    evaluate.add_argument("--protocol", required=True)
    evaluate.add_argument("--partition", default="heldout")
    evaluate.add_argument("--expected-evaluator-digest")
    evaluate.add_argument("--output", required=True)
    hybrid = sub.add_parser("hybrid-bundle")
    hybrid.add_argument("--spec", required=True)
    hybrid.add_argument("--output", required=True)
    controlled = sub.add_parser("controlled-bundle")
    controlled.add_argument("--spec", required=True)
    controlled.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command == "preregister":
        rows, blockers = discover_released_rollouts(args.rollout_root)
        if blockers:
            raise SystemExit(";".join(blockers))
        result = build_preregistration(sorted({row["session_id"] for row in rows}))
    elif args.command == "index":
        result = build_rollout_index(
            args.rollout_root, args.roboarena_root, _read_json(args.protocol)
        )
    elif args.command == "evaluate":
        payload = _read_json(args.judgments)
        result = evaluate_frozen_calibration(
            payload["judgments"] if isinstance(payload, Mapping) else payload,
            protocol=_read_json(args.protocol),
            roboarena_root=args.roboarena_root,
            partition=args.partition,
            expected_evaluator_digest=args.expected_evaluator_digest,
        )
    elif args.command == "hybrid-bundle":
        result = build_hybrid_scene_bundle(_read_json(args.spec))
    else:
        result = build_controlled_scene_bundle(_read_json(args.spec))
    write_json(Path(args.output), result)
    return 0 if result.get("status") not in {"blocked", "invalid"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
