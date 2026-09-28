"""Paired, cohort-explicit diagnostic summaries from retained producer receipts."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .policy_scientific_reset import compare_reset_readbacks


def _score(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return (row.get("episode") or {}).get("score") or {}


def _scorable(row: Mapping[str, Any]) -> bool:
    score = _score(row)
    return (row.get("status") == "completed" and row.get("policy_outcome_interpretable") is True
            and row.get("candidate_policy_queried") is True
            and score.get("status") == "scored" and isinstance(score.get("task_succeeded"), bool)
            and row.get("scoring_authority") == "deterministic_simulator_state")


def paired_summary(episodes: Sequence[Mapping[str, Any]], *, candidate_ids: Sequence[str],
                   planned_cells: Sequence[Mapping[str, Any]] | None = None,
                   source_digest: str | None = None) -> dict[str, Any]:
    if len(candidate_ids) != 2 or any(not isinstance(value, str) or not value for value in candidate_ids) or len(set(candidate_ids)) != 2:
        raise ValueError("paired_summary_candidate_pair_invalid")
    candidates = sorted(candidate_ids)
    rows = {}
    cells = {}
    for cell in planned_cells or []:
        key = (cell["cell_id"], cell["seed"])
        if key in cells:
            raise ValueError("paired_summary_duplicate_planned_cell")
        cells[key] = dict(cell)
    for row in episodes:
        candidate, cell, seed = row.get("candidate_id"), row.get("cell_id"), row.get("seed")
        if candidate not in candidates or not isinstance(cell, str) or not cell or type(seed) is not int:
            raise ValueError("paired_summary_episode_identity_invalid")
        key = (cell, seed)
        if planned_cells is not None and key not in cells:
            raise ValueError("paired_summary_unscheduled_episode")
        if (candidate, *key) in rows:
            raise ValueError("paired_summary_duplicate_episode")
        rows[(candidate, *key)] = row
        if key in cells:
            if any(field in cells[key] and field in row and cells[key][field] != row[field]
                   for field in ("family", "partition", "resolved_scenario_digest", "cell_spec_digest")):
                raise ValueError("paired_summary_cell_identity_mismatch")
        else:
            cells[key] = {field: row[field] for field in ("cell_id", "seed", "family", "partition", "resolved_scenario_digest", "cell_spec_digest") if field in row}

    def summarize(keys):
        summaries = {}
        for candidate in candidates:
            retained = [rows[(candidate, *key)] for key in keys if (candidate, *key) in rows]
            scored = [row for row in retained if _scorable(row)]
            successes = sum(_score(row)["task_succeeded"] is True for row in scored)
            summaries[candidate] = {"scheduled": len(keys), "retained": len(retained),
                "query_attempted_episodes": sum(row.get("candidate_policy_query_attempted") is True or row.get("candidate_policy_queried") is True for row in retained),
                "queried_episodes": sum(row.get("candidate_policy_queried") is True for row in retained),
                "completed": sum(row.get("status") == "completed" for row in retained),
                "interpretable": sum(row.get("policy_outcome_interpretable") is True for row in retained),
                "scorable": len(scored), "successful": successes, "missing": len(keys) - len(retained),
                "marginal_success_rate": successes / len(scored) if scored else None,
                "failed_or_unscorable": len(keys) - len(scored)}
        pairs = []
        a_only = b_only = both = neither = 0
        measured_pairs = 0
        reset_gaps = []
        for key in keys:
            left, right = (rows.get((candidate, *key)) for candidate in candidates)
            if left is None or right is None or not _scorable(left) or not _scorable(right):
                continue
            pairs.append({"cell_id": key[0], "seed": key[1]})
            a, b = _score(left)["task_succeeded"], _score(right)["task_succeeded"]
            a_only += int(a and not b)
            b_only += int(b and not a)
            both += int(a and b)
            neither += int(not a and not b)
            if left.get("scientific_reset") and right.get("scientific_reset"):
                parity = compare_reset_readbacks(left["scientific_reset"], right["scientific_reset"])
                measured_pairs += int(parity["comparison_eligible"])
                if not parity["comparison_eligible"]:
                    reset_gaps.append({"cell_id": key[0], "seed": key[1], "status": parity["status"],
                                       "gaps": parity["gaps"], "mismatches": parity["mismatches"]})
            else:
                reset_gaps.append({"cell_id": key[0], "seed": key[1], "status": "unverified", "gaps": ["scientific_reset_missing"]})
        discordant = a_only + b_only
        p = min(1.0, 2 * sum(math.comb(discordant, index) for index in range(min(a_only, b_only) + 1)) / 2**discordant) if discordant else None
        return {"candidate_counts": summaries, "mutually_scorable_pairs": len(pairs), "paired_cells": pairs,
            "a_only_successes": a_only, "b_only_successes": b_only, "both_successes": both, "both_failures": neither,
            "paired_delta_b_minus_a": (b_only - a_only) / len(pairs) if pairs else None,
            "two_sided_exact_sign_test_p": p, "measured_reset_pairs": measured_pairs, "reset_parity_gaps": reset_gaps}

    ordered = sorted(cells)
    groups = {}
    for key in ordered:
        cell = cells[key]
        group = str(cell.get("family") or "unlabeled") + "/" + str(cell.get("partition") or "unlabeled")
        groups.setdefault(group, []).append(key)
    by_family = {key: summarize(value) for key, value in sorted(groups.items())}
    canonical = [key for key in ordered if cells[key].get("family") == "canonical_anchor"]
    anchor_summary = summarize(canonical)
    for group in by_family.values():
        group["degradation_from_canonical"] = {}
        for candidate in candidates:
            anchor = anchor_summary["candidate_counts"][candidate]["marginal_success_rate"]
            rate = group["candidate_counts"][candidate]["marginal_success_rate"]
            group["degradation_from_canonical"][candidate] = anchor - rate if anchor is not None and rate is not None else None
    evidence = sorted((dict(row) for row in episodes), key=lambda row: (row["candidate_id"], row["cell_id"], row["seed"]))
    result = {"schema_version": "policy_paired_diagnostic_summary.v1", "candidate_ids": candidates,
        "source_result_digest": source_digest, "source_episode_set_digest": canonical_digest({"episodes": evidence}),
        "overall": summarize(ordered), "canonical": anchor_summary, "by_family_and_partition": by_family,
        "analysis_rule": "mutually_scorable_pairs_for_paired_delta;separate_marginal_denominators;no_missingness_imputation",
        "official_ranking_authorized": False, "qualification_authorized": False,
        "claim_ceiling": "diagnostic_policy_execution"}
    result["summary_digest"] = canonical_digest(result, digest_field="summary_digest")
    return result
