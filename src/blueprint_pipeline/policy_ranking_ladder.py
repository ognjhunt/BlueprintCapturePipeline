"""Known-ordering policy ladder for evaluation-ranker validation.

Builds a set of request-ready ``policy_candidates`` with a ground-truth
quality ordering the evaluator did not choose: the clean inner policy, then
noise-degraded variants of the same policy at increasing amplitudes (via
``noise_degraded_policy_command_adapter``), plus an optional scripted
reference floor. Because each degraded rung is the same policy with strictly
more action noise, the expected ordering over the noise rungs is provable a
priori — if the ranking scorecard cannot recover it, the ranker (not the
policies) is the problem.

The reference floor's position relative to noised learned policies is NOT
provable a priori, so it is excluded from strict-ordering validation and only
serves as a behavior-distinctness probe.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .common import utc_now_iso, write_json
from .noise_degraded_policy_command_adapter import (
    AMPLITUDE_ENV,
    INNER_COMMAND_ENV,
    noise_degraded_policy_id,
)


LADDER_SCHEMA_VERSION = "policy_ranking_ladder.v1"
VALIDATION_SCHEMA_VERSION = "policy_ranking_ladder_validation.v1"
DEFAULT_AMPLITUDES = (0.1, 0.3, 0.6)
DEFAULT_SEED = 1337
REFERENCE_FLOOR_POLICY_ID = "blueprint_default_walk_to_target_smoke_policy"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


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
) -> str:
    return " ".join(
        [
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
    )


def build_known_ordering_policy_ladder(
    *,
    inner_policy_id: str,
    inner_command: str | None = None,
    amplitudes: Sequence[float] = DEFAULT_AMPLITUDES,
    seed: int = DEFAULT_SEED,
    include_reference_floor: bool = True,
    python_executable: str | None = None,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    clean_policy_id = _string(inner_policy_id)
    if not clean_policy_id:
        raise ValueError("inner_policy_id is required")
    cleaned_amplitudes = sorted({float(a) for a in amplitudes})
    if not cleaned_amplitudes:
        raise ValueError("at least one noise amplitude is required")
    if any(a <= 0.0 for a in cleaned_amplitudes):
        raise ValueError("noise amplitudes must be positive (the clean rung is amplitude zero)")
    executable = _string(python_executable) or sys.executable

    candidates: List[Dict[str, Any]] = [
        {
            "policy_id": clean_policy_id,
            "display_name": f"{clean_policy_id} (clean)",
            "candidate_role": "ladder_clean_reference",
            "source": "policy_ranking_ladder",
            "expected_rank": 1,
            "expected_ordering_provable": True,
            "noise_amplitude": 0.0,
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
                "adapter_command": _noise_variant_command(
                    inner_command=_string(inner_command),
                    amplitude=amplitude,
                    seed=seed,
                    policy_id=policy_id,
                    python_executable=executable,
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
    return {
        "schema_version": LADDER_SCHEMA_VERSION,
        "generated_at": _string(generated_at) or utc_now_iso(),
        "inner_policy_id": clean_policy_id,
        "inner_command_configured": bool(_string(inner_command)),
        "inner_command_env": INNER_COMMAND_ENV,
        "amplitude_env": AMPLITUDE_ENV,
        "noise_amplitudes": cleaned_amplitudes,
        "noise_seed": int(seed),
        "expected_ranking": provable_policy_ids,
        "expected_ranking_basis": "same_policy_with_strictly_increasing_action_noise",
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
            tie_end + 1 < len(ordered)
            and abs(ordered[tie_end + 1][1] - ordered[index][1]) <= 1e-12
        ):
            tie_end += 1
        average = (index + tie_end) / 2.0 + 1.0
        for position in range(index, tie_end + 1):
            ranks[ordered[position][0]] = average
        index = tie_end + 1
    return ranks


def _spearman(expected_ranks: Mapping[str, float], observed_ranks: Mapping[str, float]) -> float | None:
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
    for row in rankings:
        policy_id = _string(row.get("policy_id"))
        if policy_id:
            try:
                observed_scores[policy_id] = float(row.get("score", 0.0))
            except (TypeError, ValueError):
                continue
    scorecard_status = _string(scorecard.get("status"))
    comparison_blockers = [
        _string(item)
        for item in scorecard.get("comparison_blockers", []) or []
        if _string(item)
    ]

    blockers: List[str] = []
    missing_policy_ids = [
        policy_id for policy_id in expected_ranking if policy_id not in observed_scores
    ]
    if scorecard_status.startswith("blocked") or comparison_blockers:
        blockers.append("scorecard_blocked_or_has_comparison_blockers")
    if missing_policy_ids:
        blockers.append("ladder_policies_missing_from_scorecard_rankings")
    if len(expected_ranking) < 2:
        blockers.append("ladder_requires_at_least_two_provable_rungs")

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
        status = (
            "inconclusive_missing_ladder_policies"
            if missing_policy_ids
            else "inconclusive_scorecard_blocked"
        )
    elif pairwise_violations:
        status = "not_recovered"
    elif tied_pairs:
        status = "recovered_with_ties"
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
        "ranker_ordering_recovered": status in {"recovered", "recovered_with_ties"},
        "expected_ranking": expected_ranking,
        "observed_ladder_scores": {
            policy_id: round(score, 6) for policy_id, score in observed_ladder_scores.items()
        },
        "observed_ladder_ranks": observed_ranks,
        "spearman_rank_correlation_vs_expected": spearman,
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
    build.add_argument(
        "--amplitude",
        dest="amplitudes",
        type=float,
        action="append",
        help="Noise amplitude rung (repeatable); defaults to 0.1 0.3 0.6",
    )
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--no-reference-floor", action="store_true")
    build.add_argument("--out", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate", help="Validate a policy_ranking_scorecard.json against a ladder"
    )
    validate.add_argument("--scorecard", type=Path, required=True)
    validate.add_argument("--ladder", type=Path, required=True)
    validate.add_argument("--out", type=Path)

    args = parser.parse_args(argv)
    if args.mode == "build":
        ladder = build_known_ordering_policy_ladder(
            inner_policy_id=args.inner_policy_id,
            inner_command=args.inner_command,
            amplitudes=tuple(args.amplitudes) if args.amplitudes else DEFAULT_AMPLITUDES,
            seed=args.seed,
            include_reference_floor=not args.no_reference_floor,
        )
        write_json(args.out, ladder)
        print(json.dumps({"status": "written", "path": str(args.out)}, sort_keys=True))
        return 0

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
