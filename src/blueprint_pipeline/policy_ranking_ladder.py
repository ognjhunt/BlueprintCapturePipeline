"""Known-ordering policy ladder for evaluation-ranker validation.

Builds a set of request-ready ``policy_candidates`` with a ground-truth
ordering hypothesis the evaluator did not choose: the clean inner policy,
then noise-degraded variants of the same immutable policy at increasing
amplitudes (via ``noise_degraded_policy_command_adapter``), plus an optional
scripted reference floor. Recovery is accepted only when matched, signed
runtime outcomes empirically establish the registered strict ordering; noise
amplitude alone is never treated as ground truth.

The reference floor's position relative to noised learned policies is NOT
provable a priori, so it is excluded from strict-ordering validation and only
serves as a behavior-distinctness probe.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .common import utc_now_iso, write_json
from .noise_degraded_policy_command_adapter import (
    AMPLITUDE_ENV,
    INNER_COMMAND_ENV,
    registered_action_bounds_sha256,
    noise_degraded_policy_id,
    validate_registered_action_bounds_contract,
)
from .rank_fidelity_statistics import (
    fisher_exact_greater,
    minimum_detectable_difference,
)
from .sc3_fidelity_contracts import (
    SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
    validate_trusted_ed25519_attestation,
)


LADDER_SCHEMA_VERSION = "policy_ranking_ladder.v1"
VALIDATION_SCHEMA_VERSION = "policy_ranking_ladder_validation.v1"
DEFAULT_AMPLITUDES = (0.1, 0.3, 0.6)
DEFAULT_SEED = 1337
# Structural floor only: it bounds the shape of the evidence artifact and says
# nothing about whether the observed ordering is distinguishable from chance.
# Statistical acceptance is decided by _ladder_separation_analysis.
MIN_LADDER_SEED_COUNT = 3
LADDER_SEPARATION_ALPHA = 0.05
MAX_LADDER_SEED_SEARCH = 100_000
# Success-rate gap between adjacent rungs the default ladder is built to
# resolve.  Rungs closer than this are not distinguishable by the ladder and
# should be registered as a single rung instead.
DEFAULT_TARGET_ADJACENT_SEPARATION = 0.25
# Replicate seeds per rung needed for the default separation to clear the
# two-proportion resolving-power threshold.  Derived rather than guessed; a
# ladder run at MIN_LADDER_SEED_COUNT cannot distinguish any ordering from
# chance, because at three Bernoulli trials the adjacent-rung exact one-sided
# p-value is 0.5.
DEFAULT_LADDER_SEED_COUNT = 63
_SEED_STRIDE = 104729


def replicate_seed_ids(seed: int, count: int = DEFAULT_LADDER_SEED_COUNT) -> List[int]:
    """Deterministic replicate seed ids for a ladder rung."""

    if count < 1:
        raise ValueError("replicate_seed_count must be positive")
    return [int(seed) + index * _SEED_STRIDE for index in range(int(count))]


def recommended_replicate_seed_count(
    target_separation: float = DEFAULT_TARGET_ADJACENT_SEPARATION,
) -> int | None:
    """Smallest per-rung replicate count that can resolve ``target_separation``."""

    if not 0.0 < target_separation <= 1.0:
        return None
    for candidate in range(2, MAX_LADDER_SEED_SEARCH + 1):
        detectable = minimum_detectable_difference(candidate)
        if detectable is not None and detectable <= target_separation:
            return candidate
    return None
REFERENCE_FLOOR_POLICY_ID = "blueprint_default_walk_to_target_smoke_policy"
POLICY_LADDER_VALIDATION_METHOD = "trusted_policy_ladder_validation_authority.v1"
POLICY_LADDER_VALIDATION_SIGNING_PRIVATE_KEY_FILE_ENV = (
    "BLUEPRINT_POLICY_LADDER_VALIDATION_SIGNING_PRIVATE_KEY_FILE"
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _is_sha256(value: Any) -> bool:
    text = _string(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _ladder_claim_boundary() -> dict[str, Any]:
    return {
        "ladder_validates_evaluator_ordering_sensitivity_only": True,
        "degraded_variants_are_synthetic_not_real_checkpoints": True,
        "reference_floor_ordering_not_provable_a_priori": True,
        "rank_fidelity_result_proven": False,
        "real_world_outcome_proven": False,
        "deployment_approval_proven": False,
        "public_claim_upgrade_allowed": False,
    }


def _noise_variant_command(
    *,
    inner_command: str,
    amplitude: float,
    seed: int,
    policy_id: str,
    python_executable: str,
    registered_action_bounds: Mapping[str, Any] | None = None,
    registered_action_bounds_sha256_value: str | None = None,
) -> str:
    parts = [
        shlex.quote(python_executable),
        "-m",
        "blueprint_pipeline.noise_degraded_policy_command_adapter",
        "--inner-command",
        shlex.quote(inner_command),
        "--noise-amplitude",
        f"{amplitude:g}",
        "--seed",
        str(int(seed)),
        "--policy-id",
        shlex.quote(policy_id),
    ]
    bounds = _mapping(registered_action_bounds)
    bounds_digest = _string(registered_action_bounds_sha256_value).lower()
    if bounds and bounds_digest:
        parts.extend(
            [
                "--registered-action-bounds-json",
                shlex.quote(json.dumps(bounds, sort_keys=True, separators=(",", ":"))),
                "--registered-action-bounds-sha256",
                bounds_digest,
            ]
        )
    return " ".join(parts)


def build_known_ordering_policy_ladder(
    *,
    inner_policy_id: str,
    inner_command: str | None = None,
    inner_checkpoint_sha256: str | None = None,
    amplitudes: Sequence[float] = DEFAULT_AMPLITUDES,
    seed: int = DEFAULT_SEED,
    replicate_seed_count: int = DEFAULT_LADDER_SEED_COUNT,
    include_reference_floor: bool = True,
    registered_task_id: str = "ladder-task",
    registered_condition_id: str = "ladder-condition",
    registered_criterion_id: str = "registered_task_success",
    registered_action_bounds: Mapping[str, Any] | None = None,
    python_executable: str | None = None,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    clean_policy_id = _string(inner_policy_id)
    if not clean_policy_id:
        raise ValueError("inner_policy_id is required")
    cleaned_amplitudes = sorted({float(a) for a in amplitudes})
    if not cleaned_amplitudes:
        raise ValueError("at least one noise amplitude is required")
    if any(not math.isfinite(a) or a <= 0.0 for a in cleaned_amplitudes):
        raise ValueError("noise amplitudes must be positive (the clean rung is amplitude zero)")
    executable = _string(python_executable) or sys.executable
    checkpoint_sha256 = _string(inner_checkpoint_sha256).lower()
    if checkpoint_sha256 and not _is_sha256(checkpoint_sha256):
        raise ValueError("inner_checkpoint_sha256 must be a SHA-256 digest")
    action_bounds_contract = _mapping(registered_action_bounds)
    action_bounds_digest = ""
    if action_bounds_contract:
        action_bounds_blockers = validate_registered_action_bounds_contract(action_bounds_contract)
        if action_bounds_blockers:
            raise ValueError(
                "registered_action_bounds invalid: " + ",".join(action_bounds_blockers)
            )
        action_bounds_digest = registered_action_bounds_sha256(action_bounds_contract)
    if int(replicate_seed_count) < MIN_LADDER_SEED_COUNT:
        raise ValueError(
            f"replicate_seed_count must be at least {MIN_LADDER_SEED_COUNT}"
        )
    noise_seeds = replicate_seed_ids(int(seed), int(replicate_seed_count))
    registered_condition_descriptor = {
        "schema_version": "policy_ladder_registered_condition.v1",
        "task_id": _string(registered_task_id),
        "condition_id": _string(registered_condition_id),
        "criterion_id": _string(registered_criterion_id),
    }
    if any(
        not value
        for key, value in registered_condition_descriptor.items()
        if key != "schema_version"
    ):
        raise ValueError("registered ladder task/condition/criterion are required")
    registered_condition_manifest_sha256 = hashlib.sha256(
        json.dumps(
            registered_condition_descriptor,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    candidates: List[Dict[str, Any]] = [
        {
            "policy_id": clean_policy_id,
            "display_name": f"{clean_policy_id} (clean)",
            "candidate_role": "ladder_clean_reference",
            "source": "policy_ranking_ladder",
            "expected_rank": 1,
            "expected_ordering_provable": True,
            "noise_amplitude": 0.0,
            "required_replicate_seeds": noise_seeds,
            "adapter_command": _string(inner_command) or None,
            "reference_only": False,
            "candidate_behavior_distinctness_proven": False,
            "robot_team_policy_execution_proven": False,
        }
    ]
    for index, amplitude in enumerate(cleaned_amplitudes, start=2):
        policy_id = noise_degraded_policy_id(clean_policy_id, amplitude)
        candidates.append(
            {
                "policy_id": policy_id,
                "display_name": f"{clean_policy_id} + noise amplitude {amplitude:g}",
                "candidate_role": "ladder_noise_degraded_variant",
                "source": "policy_ranking_ladder",
                "expected_rank": index,
                "expected_ordering_provable": True,
                "noise_amplitude": amplitude,
                "noise_seed": int(seed),
                "required_replicate_seeds": noise_seeds,
                "adapter_commands_by_seed": [
                    _noise_variant_command(
                        inner_command=_string(inner_command),
                        amplitude=amplitude,
                        seed=replicate_seed,
                        policy_id=policy_id,
                        python_executable=executable,
                        registered_action_bounds=action_bounds_contract,
                        registered_action_bounds_sha256_value=action_bounds_digest,
                    )
                    for replicate_seed in noise_seeds
                ]
                if _string(inner_command)
                else [],
                "adapter_command": _noise_variant_command(
                    inner_command=_string(inner_command),
                    amplitude=amplitude,
                    seed=seed,
                    policy_id=policy_id,
                    python_executable=executable,
                    registered_action_bounds=action_bounds_contract,
                    registered_action_bounds_sha256_value=action_bounds_digest,
                )
                if _string(inner_command)
                else None,
                "reference_only": False,
                "candidate_behavior_distinctness_proven": False,
                "robot_team_policy_execution_proven": False,
            }
        )
    if include_reference_floor:
        candidates.append(
            {
                "policy_id": REFERENCE_FLOOR_POLICY_ID,
                "display_name": "Blueprint default walk-to-target smoke policy (floor probe)",
                "candidate_role": "ladder_reference_floor_probe",
                "source": "policy_ranking_ladder",
                "expected_rank": None,
                "expected_ordering_provable": False,
                "noise_amplitude": None,
                "adapter_command": None,
                "reference_only": True,
                "candidate_behavior_distinctness_proven": False,
                "robot_team_policy_execution_proven": False,
            }
        )
    provable_policy_ids = [
        _string(candidate.get("policy_id"))
        for candidate in candidates
        if candidate.get("expected_ordering_provable")
    ]
    for candidate in candidates:
        command = _string(candidate.get("adapter_command"))
        candidate["adapter_command_sha256"] = (
            hashlib.sha256(command.encode("utf-8")).hexdigest() if command else None
        )
        candidate["policy_checkpoint_sha256"] = (
            checkpoint_sha256 or None if candidate.get("expected_ordering_provable") else None
        )
        candidate["registered_action_bounds_sha256"] = (
            action_bounds_digest or None if candidate.get("expected_ordering_provable") else None
        )
        commands_by_seed = candidate.get("adapter_commands_by_seed")
        if not isinstance(commands_by_seed, Sequence) or isinstance(
            commands_by_seed, (str, bytes, bytearray)
        ):
            commands_by_seed = [command for _ in noise_seeds] if command else []
        candidate["adapter_command_sha256_by_seed"] = {
            str(replicate_seed): hashlib.sha256(_string(seed_command).encode("utf-8")).hexdigest()
            for replicate_seed, seed_command in zip(noise_seeds, commands_by_seed)
            if _string(seed_command)
        }
    return {
        "schema_version": LADDER_SCHEMA_VERSION,
        "generated_at": _string(generated_at) or utc_now_iso(),
        "inner_policy_id": clean_policy_id,
        "inner_command_configured": bool(_string(inner_command)),
        "inner_policy_command_sha256": (
            hashlib.sha256(_string(inner_command).encode("utf-8")).hexdigest()
            if _string(inner_command)
            else None
        ),
        "inner_checkpoint_sha256": checkpoint_sha256 or None,
        "inner_command_env": INNER_COMMAND_ENV,
        "amplitude_env": AMPLITUDE_ENV,
        "noise_amplitudes": cleaned_amplitudes,
        "noise_seed": int(seed),
        "required_replicate_seeds": noise_seeds,
        "minimum_seed_count": MIN_LADDER_SEED_COUNT,
        "replicate_seed_count": len(noise_seeds),
        "recommended_seed_count_for_default_separation": recommended_replicate_seed_count(),
        "target_adjacent_separation": DEFAULT_TARGET_ADJACENT_SEPARATION,
        "registered_condition_descriptor": registered_condition_descriptor,
        "registered_condition_manifest_sha256": (registered_condition_manifest_sha256),
        "registered_action_bounds_contract": action_bounds_contract,
        "registered_action_bounds_sha256": action_bounds_digest or None,
        "empirical_ground_truth_acceptance_required": True,
        "expected_ranking": provable_policy_ids,
        "expected_ranking_basis": (
            "same_immutable_policy_with_registered_noise_and_empirically_accepted_matched_outcomes"
        ),
        "policy_candidates": candidates,
        "policy_comparison_mode": True,
        "claim_boundary": _ladder_claim_boundary(),
    }


def _average_ranks(scores: Mapping[str, float]) -> Dict[str, float]:
    """Rank policies by score descending with average-rank tie handling."""
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ranks: Dict[str, float] = {}
    index = 0
    while index < len(ordered):
        tie_end = index
        while (
            tie_end + 1 < len(ordered) and abs(ordered[tie_end + 1][1] - ordered[index][1]) <= 1e-12
        ):
            tie_end += 1
        average = (index + tie_end) / 2.0 + 1.0
        for position in range(index, tie_end + 1):
            ranks[ordered[position][0]] = average
        index = tie_end + 1
    return ranks


def _spearman(
    expected_ranks: Mapping[str, float], observed_ranks: Mapping[str, float]
) -> float | None:
    keys = sorted(set(expected_ranks) & set(observed_ranks))
    n = len(keys)
    if n < 2:
        return None
    expected_mean = sum(expected_ranks[key] for key in keys) / n
    observed_mean = sum(observed_ranks[key] for key in keys) / n
    covariance = sum(
        (expected_ranks[key] - expected_mean) * (observed_ranks[key] - observed_mean)
        for key in keys
    )
    expected_var = sum((expected_ranks[key] - expected_mean) ** 2 for key in keys)
    observed_var = sum((observed_ranks[key] - observed_mean) ** 2 for key in keys)
    if expected_var <= 0.0 or observed_var <= 0.0:
        return None
    return round(covariance / (expected_var**0.5 * observed_var**0.5), 6)


def _ladder_separation_analysis(
    expected_ranking: Sequence[str],
    empirical_success_counts: Mapping[str, tuple[int, int]],
    *,
    alpha: float = LADDER_SEPARATION_ALPHA,
) -> Dict[str, Any]:
    """Test whether the registered rung ordering is statistically resolvable.

    The ladder's acceptance decision used to rest entirely on a strict ordering
    of per-rung Bernoulli means.  At the structural floor of three replicate
    seeds the attainable rates are 0, 1/3, 2/3 and 1, so adjacent rungs differ
    by a single success and an exact one-sided test on that difference returns
    p = 0.5 -- the ordering is as likely to have arisen by chance as not.

    This computes the exact one-sided Fisher p-value for every adjacent pair
    *before* any pass/fail decision is made, and reports how many seeds per rung
    the registered separation would actually need.
    """

    pairs: list[Dict[str, Any]] = []
    separations: list[float] = []
    for index in range(len(expected_ranking) - 1):
        better_id = expected_ranking[index]
        worse_id = expected_ranking[index + 1]
        better = empirical_success_counts.get(better_id)
        worse = empirical_success_counts.get(worse_id)
        row: Dict[str, Any] = {
            "expected_better_policy_id": better_id,
            "expected_worse_policy_id": worse_id,
            "better_successes": better[0] if better else None,
            "better_trials": better[1] if better else None,
            "worse_successes": worse[0] if worse else None,
            "worse_trials": worse[1] if worse else None,
            "observed_separation": None,
            "one_sided_p_value": None,
            "resolvable_at_alpha": False,
        }
        if better and worse and better[1] and worse[1]:
            separation = better[0] / better[1] - worse[0] / worse[1]
            row["observed_separation"] = round(separation, 6)
            separations.append(separation)
            p_value = fisher_exact_greater(better[0], better[1], worse[0], worse[1])
            row["one_sided_p_value"] = p_value
            row["resolvable_at_alpha"] = p_value is not None and p_value <= alpha
        pairs.append(row)

    smallest_separation = min(separations) if separations else None
    required_seeds = None
    if smallest_separation is not None and smallest_separation > 0.0:
        # Smallest per-rung replicate count whose two-proportion resolving power
        # covers the tightest adjacent gap the ladder actually registered.
        for candidate in range(2, MAX_LADDER_SEED_SEARCH + 1):
            detectable = minimum_detectable_difference(candidate)
            if detectable is not None and detectable <= smallest_separation:
                required_seeds = candidate
                break
    return {
        "method": "exact_one_sided_adjacent_rung_separation.v1",
        "alpha": alpha,
        "adjacent_pairs": pairs,
        "adjacent_pair_count": len(pairs),
        "resolvable_adjacent_pair_count": sum(
            1 for row in pairs if row["resolvable_at_alpha"]
        ),
        "all_adjacent_pairs_resolvable": bool(pairs)
        and all(row["resolvable_at_alpha"] for row in pairs),
        "smallest_observed_separation": (
            round(smallest_separation, 6) if smallest_separation is not None else None
        ),
        "minimum_replicate_seed_count_for_statistical_separation": required_seeds,
        "structural_minimum_replicate_seed_count": MIN_LADDER_SEED_COUNT,
        "note": (
            "the structural minimum bounds artifact shape only; a strict ordering "
            "of Bernoulli means at that count is not evidence of ordering"
        ),
    }


def _single_command_option(parts: Sequence[str], option: str) -> str | None:
    if parts.count(option) != 1:
        return None
    index = parts.index(option)
    return parts[index + 1] if index + 1 < len(parts) else None


def _noise_command_matches_registration(
    command: str,
    *,
    inner_command: str,
    amplitude: float,
    seed: int,
    policy_id: str,
    registered_action_bounds: Mapping[str, Any],
    registered_action_bounds_sha256_value: str,
) -> bool:
    try:
        parts = shlex.split(command)
        observed_bounds = _mapping(
            json.loads(_single_command_option(parts, "--registered-action-bounds-json") or "")
        )
        observed_amplitude = float(_single_command_option(parts, "--noise-amplitude") or "nan")
        observed_seed = int(_single_command_option(parts, "--seed") or "")
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        len(parts) == 15
        and bool(_string(parts[0]))
        and not parts[0].startswith("-")
        and parts[1:3] == ["-m", "blueprint_pipeline.noise_degraded_policy_command_adapter"]
        and _single_command_option(parts, "--inner-command") == inner_command
        and observed_amplitude == amplitude
        and observed_seed == seed
        and _single_command_option(parts, "--policy-id") == policy_id
        and observed_bounds == _mapping(registered_action_bounds)
        and _single_command_option(parts, "--registered-action-bounds-sha256")
        == registered_action_bounds_sha256_value
    )


def validate_policy_ranking_scorecard(
    scorecard: Mapping[str, Any],
    ladder: Mapping[str, Any],
    *,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Check whether the ranking scorecard recovered the ladder's ordering.

    Only ladder rungs with ``expected_ordering_provable`` participate in the
    strict-ordering check; the reference floor is reported but never fails
    the validation.
    """
    expected_ranking = [
        _string(item) for item in ladder.get("expected_ranking", []) if _string(item)
    ]
    rankings = [
        _mapping(row)
        for row in scorecard.get("policy_rankings", []) or []
        if isinstance(row, Mapping)
    ]
    observed_scores: Dict[str, float] = {}
    seed_counts: Dict[str, int] = {}
    empirical_ground_truth: Dict[str, bool] = {}
    empirical_success_rates: Dict[str, float] = {}
    empirical_success_counts: Dict[str, tuple[int, int]] = {}
    row_contract_blockers: list[str] = []
    required_seed_values = ladder.get("required_replicate_seeds")
    required_seeds = (
        list(required_seed_values)
        if isinstance(required_seed_values, Sequence)
        and not isinstance(required_seed_values, (str, bytes, bytearray))
        else []
    )
    registered_condition = _mapping(ladder.get("registered_condition_descriptor"))
    registered_condition_sha256 = _string(
        ladder.get("registered_condition_manifest_sha256")
    ).lower()
    recomputed_condition_sha256 = hashlib.sha256(
        json.dumps(
            registered_condition,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if not (
        len(required_seeds) >= MIN_LADDER_SEED_COUNT
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in required_seeds)
        and len(set(required_seeds)) == len(required_seeds)
    ):
        row_contract_blockers.append("ladder_required_replicate_seeds_invalid")
    if not (
        registered_condition.get("schema_version") == "policy_ladder_registered_condition.v1"
        and _string(registered_condition.get("task_id"))
        and _string(registered_condition.get("condition_id"))
        and _string(registered_condition.get("criterion_id"))
        and registered_condition_sha256 == recomputed_condition_sha256
    ):
        row_contract_blockers.append("ladder_registered_condition_manifest_invalid")
    registered_action_bounds = _mapping(ladder.get("registered_action_bounds_contract"))
    registered_action_bounds_digest = _string(ladder.get("registered_action_bounds_sha256")).lower()
    bounds_blockers = validate_registered_action_bounds_contract(
        registered_action_bounds,
        expected_sha256=registered_action_bounds_digest,
    )
    row_contract_blockers.extend(f"ladder_{blocker}" for blocker in bounds_blockers)
    inner_checkpoint_sha256 = _string(ladder.get("inner_checkpoint_sha256")).lower()
    if not _is_sha256(inner_checkpoint_sha256):
        row_contract_blockers.append("ladder_inner_checkpoint_sha256_invalid")
    inner_policy_command_sha256 = _string(ladder.get("inner_policy_command_sha256")).lower()
    if not (
        ladder.get("inner_command_configured") is True and _is_sha256(inner_policy_command_sha256)
    ):
        row_contract_blockers.append("ladder_inner_policy_command_missing_or_invalid")
    ladder_candidates = {
        _string(candidate.get("policy_id")): candidate
        for candidate in ladder.get("policy_candidates", []) or []
        if isinstance(candidate, Mapping) and _string(candidate.get("policy_id"))
    }
    provable_candidates = [
        _mapping(candidate)
        for candidate in ladder.get("policy_candidates", []) or []
        if isinstance(candidate, Mapping) and candidate.get("expected_ordering_provable") is True
    ]
    provable_candidate_ids = [
        _string(candidate.get("policy_id")) for candidate in provable_candidates
    ]
    if (
        not provable_candidate_ids
        or len(set(provable_candidate_ids)) != len(provable_candidate_ids)
        or provable_candidate_ids != expected_ranking
    ):
        row_contract_blockers.append("ladder_provable_candidate_identity_invalid")
    declared_noise_amplitudes_raw = ladder.get("noise_amplitudes")
    declared_noise_amplitudes = (
        list(declared_noise_amplitudes_raw)
        if isinstance(declared_noise_amplitudes_raw, Sequence)
        and not isinstance(declared_noise_amplitudes_raw, (str, bytes, bytearray))
        else []
    )
    provable_amplitudes = [candidate.get("noise_amplitude") for candidate in provable_candidates]
    if not (
        declared_noise_amplitudes
        and all(
            isinstance(amplitude, (int, float))
            and not isinstance(amplitude, bool)
            and math.isfinite(float(amplitude))
            and float(amplitude) > 0.0
            for amplitude in declared_noise_amplitudes
        )
        and [float(amplitude) for amplitude in declared_noise_amplitudes]
        == sorted({float(amplitude) for amplitude in declared_noise_amplitudes})
        and provable_amplitudes
        == [0.0, *[float(amplitude) for amplitude in declared_noise_amplitudes]]
        and [candidate.get("expected_rank") for candidate in provable_candidates]
        == list(range(1, len(provable_candidates) + 1))
    ):
        row_contract_blockers.append("ladder_registered_noise_ordering_invalid")
    clean_candidates = [
        candidate for candidate in provable_candidates if candidate.get("noise_amplitude") == 0.0
    ]
    registered_inner_command = (
        _string(clean_candidates[0].get("adapter_command")) if len(clean_candidates) == 1 else ""
    )
    if len(clean_candidates) != 1:
        row_contract_blockers.append("ladder_clean_candidate_identity_invalid")
    for candidate in provable_candidates:
        policy_id = _string(candidate.get("policy_id")) or "missing-policy"
        if candidate.get("policy_checkpoint_sha256") != inner_checkpoint_sha256:
            row_contract_blockers.append(
                f"ladder_candidate_checkpoint_binding_mismatch:{policy_id}"
            )
        if candidate.get("registered_action_bounds_sha256") != registered_action_bounds_digest:
            row_contract_blockers.append(
                f"ladder_candidate_action_bounds_binding_mismatch:{policy_id}"
            )
        if candidate.get("required_replicate_seeds") != required_seeds:
            row_contract_blockers.append(f"ladder_candidate_seed_set_mismatch:{policy_id}")
        command = _string(candidate.get("adapter_command"))
        command_sha256 = _string(candidate.get("adapter_command_sha256")).lower()
        if not (
            command
            and _is_sha256(command_sha256)
            and hashlib.sha256(command.encode("utf-8")).hexdigest() == command_sha256
        ):
            row_contract_blockers.append(f"ladder_candidate_adapter_command_invalid:{policy_id}")
        if candidate.get("noise_amplitude") == 0.0:
            if command_sha256 != inner_policy_command_sha256:
                row_contract_blockers.append(f"ladder_clean_command_binding_mismatch:{policy_id}")
            expected_clean_seed_digests = {
                str(seed_value): inner_policy_command_sha256 for seed_value in required_seeds
            }
            if (
                _mapping(candidate.get("adapter_command_sha256_by_seed"))
                != expected_clean_seed_digests
            ):
                row_contract_blockers.append(
                    f"ladder_clean_seeded_command_binding_mismatch:{policy_id}"
                )
        else:
            candidate_amplitude = candidate.get("noise_amplitude")
            commands_by_seed = candidate.get("adapter_commands_by_seed")
            commands = (
                list(commands_by_seed)
                if isinstance(commands_by_seed, Sequence)
                and not isinstance(commands_by_seed, (str, bytes, bytearray))
                else []
            )
            command_digests = _mapping(candidate.get("adapter_command_sha256_by_seed"))
            if len(commands) != len(required_seeds):
                row_contract_blockers.append(
                    f"ladder_seeded_adapter_commands_incomplete:{policy_id}"
                )
            if not commands or command != _string(commands[0]):
                row_contract_blockers.append(
                    f"ladder_primary_seeded_adapter_command_mismatch:{policy_id}"
                )
            for seed_value, seeded_command in zip(required_seeds, commands):
                seeded_command_text = _string(seeded_command)
                seeded_digest = _string(command_digests.get(str(seed_value))).lower()
                if not (
                    seeded_command_text
                    and _is_sha256(seeded_digest)
                    and hashlib.sha256(seeded_command_text.encode("utf-8")).hexdigest()
                    == seeded_digest
                ):
                    row_contract_blockers.append(
                        f"ladder_seeded_adapter_command_invalid:{policy_id}:{seed_value}"
                    )
                if not (
                    isinstance(candidate_amplitude, (int, float))
                    and not isinstance(candidate_amplitude, bool)
                    and math.isfinite(float(candidate_amplitude))
                    and float(candidate_amplitude) > 0.0
                    and _noise_command_matches_registration(
                        seeded_command_text,
                        inner_command=registered_inner_command,
                        amplitude=float(candidate_amplitude),
                        seed=seed_value,
                        policy_id=policy_id,
                        registered_action_bounds=registered_action_bounds,
                        registered_action_bounds_sha256_value=(registered_action_bounds_digest),
                    )
                ):
                    row_contract_blockers.append(
                        f"ladder_seeded_adapter_command_contract_mismatch:{policy_id}:{seed_value}"
                    )
    for row in rankings:
        policy_id = _string(row.get("policy_id"))
        if policy_id:
            try:
                if "predicted_success_rate" not in row:
                    continue
                score_value = row["predicted_success_rate"]
                if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
                    continue
                score = float(score_value)
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    continue
                observed_scores[policy_id] = score
                candidate = _mapping(ladder_candidates.get(policy_id))
                if candidate.get("expected_ordering_provable") is not True:
                    continue
                seed_ids = row.get("replicate_seed_ids")
                seeds = (
                    list(seed_ids)
                    if isinstance(seed_ids, Sequence)
                    and not isinstance(seed_ids, (str, bytes, bytearray))
                    else []
                )
                seeds_valid = bool(
                    seeds
                    and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
                    and len(set(seeds)) == len(seeds)
                )
                seed_counts[policy_id] = len(seeds) if seeds_valid else 0
                if row.get("replicate_seed_count") != len(seeds):
                    row_contract_blockers.append(
                        f"ladder_replicate_seed_count_mismatch:{policy_id}"
                    )
                if seeds != required_seeds:
                    row_contract_blockers.append(f"ladder_replicate_seed_set_mismatch:{policy_id}")
                artifact = _mapping(row.get("empirical_ground_truth_artifact"))
                artifact_path = Path(_string(artifact.get("path"))).expanduser()
                artifact_digest = _string(artifact.get("sha256")).lower()
                evidence_valid = False
                empirical_successes: list[bool] = []
                if (
                    artifact_path.is_file()
                    and len(artifact_digest) == 64
                    and hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact_digest
                ):
                    try:
                        evidence = json.loads(artifact_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        evidence = None
                    if isinstance(evidence, Mapping):
                        evidence_seeds = evidence.get("replicate_seed_ids")
                        expected_amplitude = candidate.get("noise_amplitude")
                        outcome_records = [
                            dict(record)
                            for record in evidence.get("outcome_records", []) or []
                            if isinstance(record, Mapping)
                        ]
                        outcome_seeds: list[int] = []
                        outcome_records_valid = len(outcome_records) == len(seeds)
                        for record in outcome_records:
                            outcome_ref = _mapping(record.get("outcome_artifact"))
                            outcome_path = Path(_string(outcome_ref.get("path"))).expanduser()
                            outcome_digest = _string(outcome_ref.get("sha256")).lower()
                            trace_ref = _mapping(record.get("action_trace_artifact"))
                            trace_path = Path(_string(trace_ref.get("path"))).expanduser()
                            trace_digest = _string(trace_ref.get("sha256")).lower()
                            if not (
                                outcome_path.is_file()
                                and trace_path.is_file()
                                and hashlib.sha256(outcome_path.read_bytes()).hexdigest()
                                == outcome_digest
                                and hashlib.sha256(trace_path.read_bytes()).hexdigest()
                                == trace_digest
                            ):
                                outcome_records_valid = False
                                continue
                            try:
                                outcome_payload = json.loads(
                                    outcome_path.read_text(encoding="utf-8")
                                )
                                trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
                            except (OSError, json.JSONDecodeError):
                                outcome_records_valid = False
                                continue
                            seed_value = record.get("replicate_seed")
                            success_value = record.get("empirical_success")
                            expected_command_sha256 = _string(
                                _mapping(candidate.get("adapter_command_sha256_by_seed")).get(
                                    str(seed_value)
                                )
                            )
                            expected_checkpoint_sha256 = _string(
                                candidate.get("policy_checkpoint_sha256")
                            )
                            expected_action_bounds_sha256 = _string(
                                candidate.get("registered_action_bounds_sha256")
                            )
                            runtime_attestation = (
                                validate_trusted_ed25519_attestation(
                                    _mapping(outcome_payload.get("runtime_attestation")),
                                    signed_payload={
                                        key: value
                                        for key, value in outcome_payload.items()
                                        if key != "runtime_attestation"
                                    },
                                    prefix="policy_ladder_runtime_outcome_attestation",
                                    trusted_public_key_sha256_env=(
                                        SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV
                                    ),
                                )
                                if isinstance(outcome_payload, Mapping)
                                else {"status": "blocked"}
                            )
                            condition_binding_valid = bool(
                                isinstance(outcome_payload, Mapping)
                                and outcome_payload.get("task_id")
                                == registered_condition.get("task_id")
                                and outcome_payload.get("condition_id")
                                == registered_condition.get("condition_id")
                                and outcome_payload.get("criterion_id")
                                == registered_condition.get("criterion_id")
                                and outcome_payload.get("registered_condition_manifest_sha256")
                                == registered_condition_sha256
                                and isinstance(trace_payload, Mapping)
                                and trace_payload.get("task_id")
                                == registered_condition.get("task_id")
                                and trace_payload.get("condition_id")
                                == registered_condition.get("condition_id")
                                and trace_payload.get("criterion_id")
                                == registered_condition.get("criterion_id")
                                and trace_payload.get("registered_condition_manifest_sha256")
                                == registered_condition_sha256
                            )
                            if not condition_binding_valid:
                                row_contract_blockers.append(
                                    f"ladder_registered_condition_mismatch:{policy_id}:{seed_value}"
                                )
                            if not (
                                isinstance(seed_value, int)
                                and not isinstance(seed_value, bool)
                                and isinstance(success_value, bool)
                                and record.get("accepted") is True
                                and isinstance(outcome_payload, Mapping)
                                and outcome_payload.get("schema_version")
                                == "policy_ladder_runtime_outcome.v1"
                                and _string(outcome_payload.get("runtime_session_id"))
                                and _string(outcome_payload.get("runtime_executor_id"))
                                and _is_sha256(outcome_payload.get("runtime_executor_code_sha256"))
                                and runtime_attestation.get("status") == "validated"
                                and _string(outcome_payload.get("policy_id")) == policy_id
                                and outcome_payload.get("replicate_seed") == seed_value
                                and outcome_payload.get("empirical_success") is success_value
                                and outcome_payload.get("accepted") is True
                                and outcome_payload.get("noise_amplitude") == expected_amplitude
                                and outcome_payload.get("action_trace_sha256") == trace_digest
                                and outcome_payload.get("adapter_command_sha256")
                                == expected_command_sha256
                                and outcome_payload.get("policy_checkpoint_sha256")
                                == expected_checkpoint_sha256
                                and outcome_payload.get("registered_action_bounds_sha256")
                                == expected_action_bounds_sha256
                                and outcome_payload.get("action_bounds_enforced") is True
                                and condition_binding_valid
                                and isinstance(trace_payload, Mapping)
                                and trace_payload.get("schema_version")
                                == "policy_ladder_action_trace.v1"
                                and _string(trace_payload.get("policy_id")) == policy_id
                                and trace_payload.get("replicate_seed") == seed_value
                                and trace_payload.get("noise_amplitude") == expected_amplitude
                                and trace_payload.get("runtime_session_id")
                                == outcome_payload.get("runtime_session_id")
                                and trace_payload.get("adapter_command_sha256")
                                == expected_command_sha256
                                and trace_payload.get("policy_checkpoint_sha256")
                                == expected_checkpoint_sha256
                                and trace_payload.get("registered_action_bounds_sha256")
                                == expected_action_bounds_sha256
                                and trace_payload.get("action_bounds_enforced") is True
                                and isinstance(trace_payload.get("action_sequence"), list)
                                and _string(trace_payload.get("action_sequence_sha256"))
                                == hashlib.sha256(
                                    json.dumps(
                                        trace_payload.get("action_sequence"),
                                        sort_keys=True,
                                        separators=(",", ":"),
                                    ).encode("utf-8")
                                ).hexdigest()
                            ):
                                outcome_records_valid = False
                                continue
                            outcome_seeds.append(seed_value)
                            empirical_successes.append(success_value)
                        evidence_valid = bool(
                            evidence.get("schema_version")
                            == "policy_ladder_empirical_ground_truth.v1"
                            and _string(evidence.get("policy_id")) == policy_id
                            and evidence.get("accepted") is True
                            and evidence_seeds == seeds
                            and evidence_seeds == required_seeds
                            and evidence.get("noise_amplitude") == expected_amplitude
                            and evidence.get("registered_condition_manifest_sha256")
                            == registered_condition_sha256
                            and evidence.get("registered_action_bounds_sha256")
                            == registered_action_bounds_digest
                            and outcome_records_valid
                            and outcome_seeds == seeds
                        )
                if not seeds_valid:
                    row_contract_blockers.append(f"ladder_replicate_seed_ids_invalid:{policy_id}")
                if not evidence_valid:
                    row_contract_blockers.append(
                        f"ladder_empirical_ground_truth_artifact_invalid:{policy_id}"
                    )
                empirical_ground_truth[policy_id] = bool(
                    row.get("empirical_ground_truth_accepted") is True
                    and seeds_valid
                    and evidence_valid
                )
                if evidence_valid and empirical_successes:
                    success_total = sum(1 for success in empirical_successes if success)
                    empirical_success_rates[policy_id] = success_total / len(empirical_successes)
                    empirical_success_counts[policy_id] = (
                        success_total,
                        len(empirical_successes),
                    )
            except (TypeError, ValueError):
                continue
    scorecard_status = _string(scorecard.get("status"))
    comparison_blockers = [
        _string(item) for item in scorecard.get("comparison_blockers", []) or [] if _string(item)
    ]

    blockers: List[str] = []
    blockers.extend(row_contract_blockers)
    missing_policy_ids = [
        policy_id for policy_id in expected_ranking if policy_id not in observed_scores
    ]
    if scorecard_status.startswith("blocked") or comparison_blockers:
        blockers.append("scorecard_blocked_or_has_comparison_blockers")
    if missing_policy_ids:
        blockers.append("ladder_policies_missing_from_scorecard_rankings")
    if len(expected_ranking) < 2:
        blockers.append("ladder_requires_at_least_two_provable_rungs")
    if any(seed_counts.get(policy_id, 0) < MIN_LADDER_SEED_COUNT for policy_id in expected_ranking):
        blockers.append("ladder_requires_multiple_replicate_seeds_per_rung")
    if any(not empirical_ground_truth.get(policy_id, False) for policy_id in expected_ranking):
        blockers.append("ladder_empirical_ground_truth_not_accepted")
    empirical_rates = [empirical_success_rates.get(policy_id) for policy_id in expected_ranking]
    if any(rate is None for rate in empirical_rates) or any(
        empirical_rates[index] <= empirical_rates[index + 1]
        for index in range(len(empirical_rates) - 1)
        if empirical_rates[index] is not None and empirical_rates[index + 1] is not None
    ):
        blockers.append("ladder_empirical_outcome_order_not_strict")

    # Computed unconditionally and before the acceptance decision, so the report
    # carries the strength of the ordering evidence even when the run is
    # otherwise blocked.
    separation_analysis = _ladder_separation_analysis(
        expected_ranking, empirical_success_counts
    )
    if not separation_analysis["all_adjacent_pairs_resolvable"]:
        blockers.append("ladder_empirical_separation_not_statistically_resolvable")

    pairwise_violations: List[Dict[str, Any]] = []
    tied_pairs: List[Dict[str, Any]] = []
    if not blockers:
        for better_index in range(len(expected_ranking)):
            for worse_index in range(better_index + 1, len(expected_ranking)):
                better_id = expected_ranking[better_index]
                worse_id = expected_ranking[worse_index]
                delta = observed_scores[worse_id] - observed_scores[better_id]
                if delta > 1e-12:
                    pairwise_violations.append(
                        {
                            "expected_better_policy_id": better_id,
                            "expected_worse_policy_id": worse_id,
                            "score_violation": round(delta, 6),
                        }
                    )
                elif abs(delta) <= 1e-12:
                    tied_pairs.append(
                        {
                            "expected_better_policy_id": better_id,
                            "expected_worse_policy_id": worse_id,
                        }
                    )

    expected_ranks = {
        policy_id: float(index + 1) for index, policy_id in enumerate(expected_ranking)
    }
    observed_ladder_scores = {
        policy_id: observed_scores[policy_id]
        for policy_id in expected_ranking
        if policy_id in observed_scores
    }
    observed_ranks = _average_ranks(observed_ladder_scores) if observed_ladder_scores else {}
    spearman = _spearman(expected_ranks, observed_ranks) if not blockers else None
    max_violation = max(
        (float(item["score_violation"]) for item in pairwise_violations), default=0.0
    )
    mean_violation = (
        round(
            sum(float(item["score_violation"]) for item in pairwise_violations)
            / len(pairwise_violations),
            6,
        )
        if pairwise_violations
        else 0.0
    )

    if blockers:
        if missing_policy_ids:
            status = "inconclusive_missing_ladder_policies"
        elif blockers == ["ladder_empirical_separation_not_statistically_resolvable"]:
            # The ladder ran and the ordering came out as registered, but the
            # replicate count cannot distinguish it from chance.  That is a
            # distinct outcome from a blocked scorecard and reports as one.
            status = "inconclusive_underpowered_separation"
        else:
            status = "inconclusive_scorecard_blocked"
    elif pairwise_violations:
        status = "not_recovered"
    elif tied_pairs:
        status = "inconclusive_tied_scores"
    else:
        status = "recovered"

    floor_rows = [
        {
            "policy_id": _string(candidate.get("policy_id")),
            "observed_score": observed_scores.get(_string(candidate.get("policy_id"))),
            "present_in_scorecard": _string(candidate.get("policy_id")) in observed_scores,
        }
        for candidate in ladder.get("policy_candidates", []) or []
        if isinstance(candidate, Mapping) and not candidate.get("expected_ordering_provable")
    ]

    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "generated_at": _string(generated_at) or utc_now_iso(),
        "status": status,
        "ranker_ordering_recovered": status == "recovered",
        "score_field": "predicted_success_rate",
        "minimum_replicate_seed_count": MIN_LADDER_SEED_COUNT,
        "replicate_seed_count_by_policy": seed_counts,
        "empirical_ground_truth_accepted_by_policy": empirical_ground_truth,
        "expected_ranking": expected_ranking,
        "observed_ladder_scores": {
            policy_id: round(score, 6) for policy_id, score in observed_ladder_scores.items()
        },
        "observed_ladder_ranks": observed_ranks,
        "spearman_rank_correlation_vs_expected": spearman,
        "empirical_success_rates_by_policy": {
            policy_id: round(rate, 6) for policy_id, rate in empirical_success_rates.items()
        },
        "empirical_separation_analysis": separation_analysis,
        "pairwise_violations": pairwise_violations,
        "pairwise_violation_count": len(pairwise_violations),
        "maximum_score_violation": round(max_violation, 6),
        "mean_score_violation": mean_violation,
        "tied_pairs": tied_pairs,
        "reference_floor_probes": floor_rows,
        "scorecard_status": scorecard_status,
        "scorecard_comparison_blockers": comparison_blockers,
        "missing_policy_ids": missing_policy_ids,
        "blockers": blockers,
        "claim_boundary": {
            **_ladder_claim_boundary(),
            "validation_is_evaluator_discrimination_check_only": True,
            "recovered_ordering_is_not_rank_fidelity_vs_real_world": True,
        },
    }


def _load_ed25519_signing_private_key(path: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    encoded = path.expanduser().read_bytes()
    if len(encoded) == 32:
        return Ed25519PrivateKey.from_private_bytes(encoded)
    try:
        decoded = base64.b64decode(encoded.strip(), validate=True)
    except (ValueError, TypeError):
        decoded = b""
    if len(decoded) == 32:
        return Ed25519PrivateKey.from_private_bytes(decoded)
    try:
        private_key = serialization.load_pem_private_key(encoded, password=None)
    except (TypeError, ValueError) as exc:
        raise ValueError("policy_ladder_validation_private_key_invalid") from exc
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ValueError("policy_ladder_validation_private_key_not_ed25519")
    return private_key


def produce_signed_policy_ranking_ladder_validation(
    *,
    ladder_path: str | Path,
    scorecard_path: str | Path,
    output_path: str | Path,
    verification_report_path: str | Path,
    signing_private_key_file: str | Path,
    signer_key_id: str = "policy-ladder-validation-authority",
    verifier_id: str = "blueprint-policy-ladder-validator",
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Recompute, bind, and sign a consumable ladder validation artifact."""

    from cryptography.hazmat.primitives import serialization

    ladder_file = Path(ladder_path).expanduser().resolve()
    scorecard_file = Path(scorecard_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    report_file = Path(verification_report_path).expanduser().resolve()
    signing_key_file = Path(signing_private_key_file).expanduser().resolve()
    source_root = ladder_file.parent
    if not (
        scorecard_file.parent == source_root
        and output_file.parent == source_root
        and report_file.parent == source_root
    ):
        raise ValueError("policy_ladder_validation_sources_output_and_report_must_share_directory")
    if len({ladder_file, scorecard_file, output_file, report_file, signing_key_file}) != 5:
        raise ValueError("policy_ladder_validation_artifact_paths_must_be_distinct")
    ladder = _load_json(ladder_file)
    scorecard = _load_json(scorecard_file)
    validation = validate_policy_ranking_scorecard(
        scorecard,
        ladder,
        generated_at=generated_at,
    )
    executor_trusted_public_key_sha256 = _string(
        os.getenv(SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV)
    ).lower()
    if not _is_sha256(executor_trusted_public_key_sha256):
        raise ValueError("policy_ladder_validation_executor_trusted_public_key_not_configured")
    source_bindings = {
        "ladder": {
            "artifact_id": ladder_file.name,
            "sha256": hashlib.sha256(ladder_file.read_bytes()).hexdigest(),
        },
        "scorecard": {
            "artifact_id": scorecard_file.name,
            "sha256": hashlib.sha256(scorecard_file.read_bytes()).hexdigest(),
        },
    }
    validation.update(
        {
            "validation_method": POLICY_LADDER_VALIDATION_METHOD,
            "source_validation_recomputed": True,
            "executor_trusted_public_key_sha256": (executor_trusted_public_key_sha256),
            "source_artifact_bindings": source_bindings,
        }
    )
    signing_key = _load_ed25519_signing_private_key(signing_key_file)
    public_key = signing_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    message = json.dumps(validation, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = signing_key.sign(message)
    signing_key.public_key().verify(signature, message)
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    signed_payload_sha256 = hashlib.sha256(message).hexdigest()
    source_bindings_sha256 = hashlib.sha256(
        json.dumps(source_bindings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json(
        report_file,
        {
            "schema_version": "sc3_signature_verification_report.v1",
            "algorithm": "Ed25519",
            "verification_status": "verified",
            "public_key_sha256": public_key_sha256,
            "signed_payload_sha256": signed_payload_sha256,
            "source_artifact_bindings_sha256": source_bindings_sha256,
            "signer_key_id": _string(signer_key_id),
            "verifier_id": _string(verifier_id),
        },
    )
    validation["validation_attestation"] = {
        "algorithm": "Ed25519",
        "signature_verified": True,
        "authority_role": "policy_ladder_validation_authority",
        "signer_key_id": _string(signer_key_id),
        "verifier_id": _string(verifier_id),
        "public_key_base64": base64.b64encode(public_key).decode("ascii"),
        "public_key_sha256": public_key_sha256,
        "signature_base64": base64.b64encode(signature).decode("ascii"),
        "signed_payload_sha256": signed_payload_sha256,
        "verification_report_artifact": {
            "artifact_id": report_file.name,
            "sha256": hashlib.sha256(report_file.read_bytes()).hexdigest(),
        },
    }
    write_json(output_file, validation)
    return validation


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    build = subparsers.add_parser("build", help="Emit a known-ordering ladder JSON")
    build.add_argument("--inner-policy-id", required=True)
    build.add_argument("--inner-command")
    build.add_argument("--inner-checkpoint-sha256")
    build.add_argument("--registered-action-bounds-manifest", type=Path)
    build.add_argument(
        "--amplitude",
        dest="amplitudes",
        type=float,
        action="append",
        help="Noise amplitude rung (repeatable); defaults to 0.1 0.3 0.6",
    )
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument(
        "--replicate-seed-count",
        type=int,
        default=DEFAULT_LADDER_SEED_COUNT,
        help=(
            "replicate seeds per rung; the default is derived from the "
            "separation the ladder must resolve, and lowering it toward the "
            "structural minimum makes the ordering statistically meaningless"
        ),
    )
    build.add_argument("--no-reference-floor", action="store_true")
    build.add_argument("--out", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate", help="Validate a policy_ranking_scorecard.json against a ladder"
    )
    validate.add_argument("--scorecard", type=Path, required=True)
    validate.add_argument("--ladder", type=Path, required=True)
    validate.add_argument("--out", type=Path)
    validate.add_argument("--signing-private-key-file", type=Path)
    validate.add_argument("--verification-report-out", type=Path)
    validate.add_argument("--signer-key-id", default="policy-ladder-validation-authority")
    validate.add_argument("--verifier-id", default="blueprint-policy-ladder-validator")

    args = parser.parse_args(argv)
    if args.mode == "build":
        ladder = build_known_ordering_policy_ladder(
            inner_policy_id=args.inner_policy_id,
            inner_command=args.inner_command,
            inner_checkpoint_sha256=args.inner_checkpoint_sha256,
            registered_action_bounds=(
                _load_json(args.registered_action_bounds_manifest)
                if args.registered_action_bounds_manifest
                else None
            ),
            amplitudes=tuple(args.amplitudes) if args.amplitudes else DEFAULT_AMPLITUDES,
            seed=args.seed,
            replicate_seed_count=args.replicate_seed_count,
            include_reference_floor=not args.no_reference_floor,
        )
        write_json(args.out, ladder)
        print(json.dumps({"status": "written", "path": str(args.out)}, sort_keys=True))
        return 0

    signing_private_key_file = args.signing_private_key_file or (
        Path(_string(os.getenv(POLICY_LADDER_VALIDATION_SIGNING_PRIVATE_KEY_FILE_ENV)))
        if _string(os.getenv(POLICY_LADDER_VALIDATION_SIGNING_PRIVATE_KEY_FILE_ENV))
        else None
    )
    if signing_private_key_file:
        if not args.out:
            parser.error("--out is required for signed validation")
        report_out = args.verification_report_out or args.out.with_name(
            f"{args.out.stem}.signature-verification.json"
        )
        validation = produce_signed_policy_ranking_ladder_validation(
            ladder_path=args.ladder,
            scorecard_path=args.scorecard,
            output_path=args.out,
            verification_report_path=report_out,
            signing_private_key_file=signing_private_key_file,
            signer_key_id=args.signer_key_id,
            verifier_id=args.verifier_id,
        )
    else:
        validation = validate_policy_ranking_scorecard(
            _load_json(args.scorecard),
            _load_json(args.ladder),
        )
        if args.out:
            write_json(args.out, validation)
    print(json.dumps(validation, sort_keys=True))
    return 0 if validation.get("ranker_ordering_recovered") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
