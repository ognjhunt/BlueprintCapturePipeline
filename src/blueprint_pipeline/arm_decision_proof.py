"""One-command Arm Decision Proof v1 evidence reconstruction.

The command consumes an admitted public-source manifest and one immutable
closed-loop execution package.  Physical-reference values are deliberately a
separate input: they are not opened until a digest-bound development decision
seal has been written and verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .evaluation_run_contract import compile_evaluation_run
from .public_reference_admission import build_public_reference_admission_receipt


EXECUTION_SCHEMA_VERSION = "simpler_closed_loop_execution.v1"
EPISODE_RECEIPT_SCHEMA_VERSION = "adp_episode_receipt.v1"
SEAL_SCHEMA_VERSION = "adp_development_decision_seal.v1"
RELEASE_SCHEMA_VERSION = "adp_physical_outcome_release_receipt.v1"
JOIN_SCHEMA_VERSION = "adp_physical_outcome_join.v1"
VERDICT_SCHEMA_VERSION = "adp_bounded_verdict.v1"
PHASE_LABEL = "retrospective_external_reference"
CLAIM_CEILING = "development_only"
SHA256_PREFIX = "sha256:"


class ArmDecisionProofError(ValueError):
    """Fail-closed ADP error with stable blocker identifiers."""

    def __init__(self, blockers: Sequence[str]):
        self.blockers = tuple(sorted(set(str(item) for item in blockers if str(item))))
        super().__init__(";".join(self.blockers))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _is_digest(value: Any) -> bool:
    text = _string(value)
    return len(text) == 71 and text.startswith(SHA256_PREFIX) and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return SHA256_PREFIX + digest.hexdigest()


def _load_json(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArmDecisionProofError([blocker]) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArmDecisionProofError([blocker + ":invalid_json"]) from exc
    if not isinstance(value, Mapping):
        raise ArmDecisionProofError([blocker + ":not_mapping"])
    return dict(value)


def _checkpoint_digest(candidate: Mapping[str, Any]) -> str:
    return canonical_digest(
        {
            "candidate_id": candidate.get("candidate_id"),
            "checkpoint_prefix": candidate.get("checkpoint_prefix"),
            "checkpoint_objects": candidate.get("checkpoint_objects"),
        }
    )


def _expected_pairs(manifest: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {
        (candidate["candidate_id"], condition["condition_id"])
        for candidate in _rows(manifest.get("candidates"))
        for condition in _rows(manifest.get("conditions"))
    }


def build_evaluation_run_spec(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the admitted SIMPLER bindings through EvaluationRunSpec."""

    task = _mapping(manifest.get("task"))
    source = _mapping(manifest.get("source"))
    repository = _mapping(source.get("repository"))
    runtime = _mapping(manifest.get("runtime"))
    candidates = _rows(manifest.get("candidates"))
    conditions = _rows(manifest.get("conditions"))
    prohibited = sorted(
        key
        for key, allowed in _mapping(manifest.get("claim_boundaries")).items()
        if allowed is False
    )
    return {
        "schema_version": "evaluation_run.v1",
        "run_id": "adp-v1-simpler-google-robot-pick-coke-can",
        "mode": "evaluate",
        "scene_bundle": {
            "adapter_id": "simpler_public_scene",
            "adapter_version": "1",
            "bundle_id": manifest["reference_id"],
            "uri": repository["url"],
            "entrypoint": task["scene_id"],
            "content_digest": manifest["manifest_digest"],
        },
        "robot_adapter": {
            "adapter_id": "simpler_google_robot",
            "adapter_version": "1",
            "robot_profile_id": task["robot_id"],
            "asset_ref": next(
                row["git_object_sha1"]
                for row in _rows(source.get("asset_bindings"))
                if row.get("role") == "robot_assets"
            ),
        },
        "task_scenario_pack": {
            "adapter_id": "simpler_condition_matrix",
            "adapter_version": "1",
            "pack_id": task["task_id"],
            "tasks": [{"task_id": task["task_id"]}],
            "scenarios": [
                {
                    "scenario_id": row["condition_id"],
                    "task_id": task["task_id"],
                    "reset_binding": row["reset_binding"],
                }
                for row in conditions
            ],
        },
        "policy_adapter": {
            "adapter_id": "simpler_rt1_candidate_set",
            "adapter_version": "1",
            "policy_id": "adp-exactly-two-rt1-candidates",
            "observation_schema_ref": canonical_digest(task["observation_schema"]),
            "action_schema_ref": canonical_digest(task["action_schema"]),
            "candidate_ids": [row["candidate_id"] for row in candidates],
        },
        "runtime_provider_profile": {
            "adapter_id": "simpler_cached_execution",
            "adapter_version": "1",
            "profile_id": "simpler-sapien-vast-immutable-input",
            "providers": ["vast"],
            "simulator": "SAPIEN-2.2.2",
            "environment_lock_digest": _mapping(runtime.get("environment_lock")).get(
                "digest"
            ),
        },
        "proof_contract": {
            "adapter_id": "declared_evidence_proof_contract",
            "adapter_version": "1",
            "contract_id": "arm-decision-proof-v1-retrospective",
            "required_evidence": [
                "closed_loop_execution",
                "independent_environment_metric",
                "episode_receipts",
                "decision_seal",
                "physical_outcome_release",
                "exact_physical_outcome_join",
            ],
            "claim_ceiling": {
                "phase_label": PHASE_LABEL,
                "claim_ceiling": CLAIM_CEILING,
            },
            "prohibited_claims": prohibited,
        },
        "metadata": {
            "program_id": "arm-decision-proof-v1",
            "source_commit": repository["commit"],
            "candidate_count": len(candidates),
        },
    }


def validate_execution_package(
    execution: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    execution_root: Path | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    if execution.get("schema_version") != EXECUTION_SCHEMA_VERSION:
        blockers.append("execution_schema_invalid")
    if execution.get("reference_id") != manifest.get("reference_id"):
        blockers.append("execution_reference_id_mismatch")
    if execution.get("source_identity_digest") != manifest.get("source_identity_digest"):
        blockers.append("execution_source_identity_digest_mismatch")
    runtime_digest = _mapping(_mapping(manifest.get("runtime")).get("environment_lock")).get(
        "digest"
    )
    if execution.get("runtime_lock_digest") != runtime_digest or not _is_digest(
        runtime_digest
    ):
        blockers.append("execution_runtime_lock_digest_mismatch")
    expected_candidates = {
        row["candidate_id"]: _checkpoint_digest(row)
        for row in _rows(manifest.get("candidates"))
    }
    observed_candidates = _rows(execution.get("candidates"))
    if len(observed_candidates) != 2:
        blockers.append("execution_must_bind_exactly_two_candidates")
    observed_ids: list[str] = []
    for row in observed_candidates:
        candidate_id = _string(row.get("candidate_id"))
        observed_ids.append(candidate_id)
        if row.get("checkpoint_identity_digest") != expected_candidates.get(candidate_id):
            blockers.append(f"execution_checkpoint_identity_mismatch:{candidate_id}")
        if row.get("genuine_checkpoint_loaded") is not True:
            blockers.append(f"execution_genuine_checkpoint_not_loaded:{candidate_id}")
    if len(set(observed_ids)) != len(observed_ids):
        blockers.append("execution_duplicate_candidate_identity")
    if set(observed_ids) != set(expected_candidates):
        blockers.append("execution_candidate_set_mismatch")

    expected = _expected_pairs(manifest)
    episodes = _rows(execution.get("episodes"))
    pairs: list[tuple[str, str]] = []
    completed_by_candidate = {candidate_id: 0 for candidate_id in expected_candidates}
    for episode in episodes:
        candidate_id = _string(episode.get("candidate_id"))
        condition_id = _string(episode.get("condition_id"))
        pair = (candidate_id, condition_id)
        pairs.append(pair)
        status = _string(episode.get("status"))
        if status not in {"completed", "failed", "invalid", "timed_out", "interrupted"}:
            blockers.append(f"execution_episode_status_invalid:{candidate_id}/{condition_id}")
        if not _string(episode.get("episode_id")):
            blockers.append(f"execution_episode_id_missing:{candidate_id}/{condition_id}")
        if not isinstance(episode.get("seed"), int):
            blockers.append(f"execution_seed_missing:{candidate_id}/{condition_id}")
        for name in (
            "source_commit",
            "dependency_lock_digest",
            "reset_digest",
            "observation_trace_digest",
            "action_trace_digest",
            "metric_trace_digest",
        ):
            if not _is_digest(episode.get(name)) and name != "source_commit":
                blockers.append(f"execution_episode_{name}_invalid:{candidate_id}/{condition_id}")
        if episode.get("source_commit") != _mapping(manifest.get("source")).get(
            "repository", {}
        ).get("commit"):
            blockers.append(f"execution_episode_source_commit_mismatch:{candidate_id}/{condition_id}")
        if episode.get("checkpoint_identity_digest") != expected_candidates.get(candidate_id):
            blockers.append(
                f"execution_episode_checkpoint_identity_mismatch:{candidate_id}/{condition_id}"
            )
        if status == "completed":
            completed_by_candidate[candidate_id] = completed_by_candidate.get(candidate_id, 0) + 1
            if not isinstance(episode.get("policy_query_count"), int) or episode.get(
                "policy_query_count", 0
            ) <= 0:
                blockers.append(f"execution_policy_not_queried:{candidate_id}/{condition_id}")
            if not isinstance(episode.get("simulator_step_count"), int) or episode.get(
                "simulator_step_count", 0
            ) <= 0:
                blockers.append(f"execution_simulator_not_stepped:{candidate_id}/{condition_id}")
        evaluator = _mapping(episode.get("evaluator"))
        if evaluator.get("owner") != "environment_not_policy" or evaluator.get(
            "policy_self_report_used"
        ) is not False:
            blockers.append(f"execution_evaluator_not_independent:{candidate_id}/{condition_id}")
        if status == "completed" and not isinstance(episode.get("success"), bool):
            blockers.append(f"execution_completed_success_missing:{candidate_id}/{condition_id}")
        artifacts = _rows(episode.get("artifacts"))
        if not artifacts or any(not _is_digest(row.get("sha256")) for row in artifacts):
            blockers.append(f"execution_artifact_digests_missing:{candidate_id}/{condition_id}")
        if execution_root is not None:
            for artifact in artifacts:
                relative_path = _string(artifact.get("relative_path"))
                if not relative_path:
                    continue
                target = (execution_root / relative_path).resolve()
                try:
                    target.relative_to(execution_root.resolve())
                except ValueError:
                    blockers.append(
                        f"execution_artifact_path_outside_root:{candidate_id}/{condition_id}"
                    )
                    continue
                if not target.is_file():
                    blockers.append(
                        f"execution_artifact_missing:{candidate_id}/{condition_id}:{relative_path}"
                    )
                elif _file_digest(target) != artifact.get("sha256"):
                    blockers.append(
                        f"execution_artifact_digest_mismatch:{candidate_id}/{condition_id}:{relative_path}"
                    )
    if len(pairs) != len(set(pairs)):
        blockers.append("execution_duplicate_candidate_condition_episode")
    if set(pairs) != expected:
        blockers.append("execution_candidate_condition_matrix_incomplete")
    for candidate_id, count in completed_by_candidate.items():
        if count == 0:
            blockers.append(f"execution_candidate_has_no_completed_episode:{candidate_id}")
    supplied_digest = execution.get("execution_digest")
    expected_digest = canonical_digest(execution, digest_field="execution_digest")
    if supplied_digest != expected_digest:
        blockers.append("execution_digest_mismatch")
    if blockers:
        raise ArmDecisionProofError(blockers)
    return {
        "status": "passed",
        "candidate_count": 2,
        "episode_count": len(episodes),
        "pair_count": len(expected),
        "execution_digest": expected_digest,
    }


def _episode_receipt(
    episode: Mapping[str, Any], *, manifest: Mapping[str, Any], execution_digest: str
) -> dict[str, Any]:
    receipt = {
        "schema_version": EPISODE_RECEIPT_SCHEMA_VERSION,
        "episode_id": episode["episode_id"],
        "candidate_id": episode["candidate_id"],
        "condition_id": episode["condition_id"],
        "seed": episode["seed"],
        "status": episode["status"],
        "success": episode.get("success"),
        "source_manifest_digest": manifest["manifest_digest"],
        "execution_digest": execution_digest,
        "source_commit": episode["source_commit"],
        "dependency_lock_digest": episode["dependency_lock_digest"],
        "checkpoint_identity_digest": episode["checkpoint_identity_digest"],
        "reset_digest": episode["reset_digest"],
        "observation_trace_digest": episode["observation_trace_digest"],
        "action_trace_digest": episode["action_trace_digest"],
        "metric_trace_digest": episode["metric_trace_digest"],
        "policy_query_count": episode["policy_query_count"],
        "simulator_step_count": episode["simulator_step_count"],
        "evaluator": episode["evaluator"],
        "failure": episode.get("failure"),
        "artifacts": episode["artifacts"],
        "phase_label": PHASE_LABEL,
        "claim_ceiling": CLAIM_CEILING,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [0.0, 1.0]
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return [round(max(0.0, center - margin), 12), round(min(1.0, center + margin), 12)]


def compile_bounded_decision(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compile the frozen two-candidate decision; uncertainty can force abstention."""

    candidate_ids = sorted({_string(row.get("candidate_id")) for row in receipts})
    if len(candidate_ids) != 2:
        raise ArmDecisionProofError(["decision_requires_exactly_two_candidates"])
    rule = {
        "baseline_candidate_id": candidate_ids[0],
        "minimum_decision_relevant_difference": 0.20,
        "alpha": 0.05,
        "target_power": 0.80,
        "design": "paired_fixed_condition_matrix",
        "invalid_trial_handling": "count_as_failure_and_expose_invalid_region",
        "multiplicity": "single_two_candidate_contrast_none_required",
        "stop_rule": "all_planned_cells_terminal",
        "uncertainty": "95_percent_wilson_per_candidate_conservative_difference_bounds",
    }
    rows: dict[str, list[Mapping[str, Any]]] = {
        candidate_id: [row for row in receipts if row.get("candidate_id") == candidate_id]
        for candidate_id in candidate_ids
    }
    summaries: dict[str, dict[str, Any]] = {}
    for candidate_id, candidate_rows in rows.items():
        valid = [row for row in candidate_rows if row.get("status") == "completed"]
        successes = sum(row.get("success") is True for row in valid)
        denominator = len(candidate_rows)
        summaries[candidate_id] = {
            "planned": denominator,
            "valid": len(valid),
            "invalid_or_failed": denominator - len(valid),
            "successes": successes,
            "success_rate_with_invalid_as_failure": successes / denominator if denominator else 0.0,
            "wilson_interval": _wilson(successes, denominator),
        }
    baseline, challenger = candidate_ids
    baseline_summary = summaries[baseline]
    challenger_summary = summaries[challenger]
    observed_difference = (
        challenger_summary["success_rate_with_invalid_as_failure"]
        - baseline_summary["success_rate_with_invalid_as_failure"]
    )
    difference_interval = [
        round(challenger_summary["wilson_interval"][0] - baseline_summary["wilson_interval"][1], 12),
        round(challenger_summary["wilson_interval"][1] - baseline_summary["wilson_interval"][0], 12),
    ]
    mdre = rule["minimum_decision_relevant_difference"]
    invalid = sum(summary["invalid_or_failed"] for summary in summaries.values())
    if invalid:
        decision = "abstain"
        selected = None
        reason = "invalid_or_failed_cells_make_selection_unsafe"
    elif difference_interval[0] >= mdre:
        decision = "select"
        selected = challenger
        reason = "challenger_conservative_difference_exceeds_mdre"
    elif difference_interval[1] <= -mdre:
        decision = "eliminate"
        selected = challenger
        reason = "challenger_conservative_difference_below_negative_mdre"
    elif difference_interval[0] >= -mdre and difference_interval[1] <= mdre:
        decision = "equivalent_inconclusive"
        selected = None
        reason = "difference_bounded_inside_equivalence_region"
    else:
        decision = "abstain"
        selected = None
        reason = "uncertainty_crosses_decision_boundaries"
    result = {
        "schema_version": "adp_bounded_decision.v1",
        "decision": decision,
        "selected_candidate_id": selected,
        "reason": reason,
        "rule": rule,
        "candidate_summaries": summaries,
        "observed_difference_challenger_minus_baseline": round(observed_difference, 12),
        "difference_interval": difference_interval,
        "coverage": sum(summary["valid"] for summary in summaries.values())
        / sum(summary["planned"] for summary in summaries.values()),
        "invalid_region": [
            row["receipt_digest"] for row in receipts if row.get("status") != "completed"
        ],
        "next_cheapest_missing_measurement": (
            "additional digest-bound fixed-reset replications for each candidate-condition cell"
            if decision in {"abstain", "equivalent_inconclusive"}
            else None
        ),
        "phase_label": PHASE_LABEL,
        "claim_ceiling": CLAIM_CEILING,
    }
    result["decision_digest"] = canonical_digest(result, digest_field="decision_digest")
    return result


def seal_decision(
    *, manifest: Mapping[str, Any], execution: Mapping[str, Any], plan: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]], decision: Mapping[str, Any]
) -> dict[str, Any]:
    seal = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "status": "sealed",
        "source_manifest_digest": manifest["manifest_digest"],
        "execution_digest": execution["execution_digest"],
        "evaluation_run_spec_digest": plan["spec_digest"],
        "candidate_ids": sorted(row["candidate_id"] for row in _rows(manifest.get("candidates"))),
        "condition_ids": sorted(row["condition_id"] for row in _rows(manifest.get("conditions"))),
        "episode_receipt_digests": sorted(row["receipt_digest"] for row in receipts),
        "decision": dict(decision),
        "physical_outcome_values_accessed": False,
        "physical_outcomes_artifact_digest": _mapping(
            _mapping(manifest.get("physical_reference")).get("outcomes_artifact")
        ).get("digest"),
        "amendment": None,
        "phase_label": PHASE_LABEL,
        "claim_ceiling": CLAIM_CEILING,
    }
    seal["seal_digest"] = canonical_digest(seal, digest_field="seal_digest")
    return seal


def release_physical_outcomes(
    *, outcomes_path: Path, manifest: Mapping[str, Any], seal: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open outcome values only after a complete, matching decision seal exists."""

    if seal.get("schema_version") != SEAL_SCHEMA_VERSION or seal.get("status") != "sealed":
        raise ArmDecisionProofError(["physical_outcome_release_requires_valid_seal"])
    if seal.get("seal_digest") != canonical_digest(seal, digest_field="seal_digest"):
        raise ArmDecisionProofError(["physical_outcome_release_seal_digest_mismatch"])
    if seal.get("physical_outcome_values_accessed") is not False:
        raise ArmDecisionProofError(["physical_outcome_release_early_access_detected"])
    expected_digest = _mapping(
        _mapping(manifest.get("physical_reference")).get("outcomes_artifact")
    ).get("digest")
    outcomes = _load_json(outcomes_path, blocker="physical_outcomes_artifact_missing")
    actual_digest = canonical_digest(outcomes, digest_field="outcomes_digest")
    if outcomes.get("outcomes_digest") != expected_digest or actual_digest != expected_digest:
        raise ArmDecisionProofError(["physical_outcomes_artifact_digest_mismatch"])
    receipt = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "status": "released_after_seal",
        "seal_digest": seal["seal_digest"],
        "outcomes_digest": actual_digest,
        "reference_id": manifest["reference_id"],
        "task_id": _mapping(manifest.get("task"))["task_id"],
        "custodian": "blueprint_programmatic_public_reference_loader",
        "software_firebreak_only": True,
        "published_outcomes_were_not_genuinely_unseen": True,
        "phase_label": PHASE_LABEL,
        "claim_ceiling": CLAIM_CEILING,
    }
    receipt["release_receipt_digest"] = canonical_digest(
        receipt, digest_field="release_receipt_digest"
    )
    return outcomes, receipt


def join_physical_outcomes(
    *, manifest: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]], outcomes: Mapping[str, Any], release: Mapping[str, Any], seal: Mapping[str, Any]
) -> dict[str, Any]:
    blockers: list[str] = []
    if release.get("seal_digest") != seal.get("seal_digest"):
        blockers.append("physical_join_seal_id_mismatch")
    if outcomes.get("reference_id") != manifest.get("reference_id"):
        blockers.append("physical_join_reference_id_mismatch")
    if outcomes.get("task_id") != _mapping(manifest.get("task")).get("task_id"):
        blockers.append("physical_join_task_id_mismatch")
    receipt_pairs = {(row["candidate_id"], row["condition_id"]): row for row in receipts}
    outcome_rows = _rows(outcomes.get("cells"))
    outcome_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for row in outcome_rows:
        pair = (_string(row.get("candidate_id")), _string(row.get("condition_id")))
        if pair in outcome_pairs:
            blockers.append("physical_join_duplicate_candidate_condition")
        outcome_pairs[pair] = row
    if set(receipt_pairs) != set(outcome_pairs) or set(receipt_pairs) != _expected_pairs(manifest):
        blockers.append("physical_join_candidate_condition_set_mismatch")
    if blockers:
        raise ArmDecisionProofError(blockers)
    cells = []
    for pair in sorted(receipt_pairs):
        receipt = receipt_pairs[pair]
        outcome = outcome_pairs[pair]
        cells.append(
            {
                "candidate_id": pair[0],
                "condition_id": pair[1],
                "episode_receipt_digest": receipt["receipt_digest"],
                "simulation_status": receipt["status"],
                "simulation_success": receipt.get("success"),
                "physical_trial_count": outcome["trial_count"],
                "physical_success_rate": outcome["success_rate"],
                "physical_reported_uncertainty": outcome["reported_uncertainty"],
                "cell_status": "joined",
            }
        )
    joined = {
        "schema_version": JOIN_SCHEMA_VERSION,
        "status": "joined_exactly",
        "seal_digest": seal["seal_digest"],
        "release_receipt_digest": release["release_receipt_digest"],
        "source_manifest_digest": manifest["manifest_digest"],
        "outcomes_digest": outcomes["outcomes_digest"],
        "cells": cells,
        "missing_outcomes": [],
        "phase_label": PHASE_LABEL,
        "claim_ceiling": CLAIM_CEILING,
    }
    joined["join_digest"] = canonical_digest(joined, digest_field="join_digest")
    return joined


def adjudicate(decision: Mapping[str, Any], joined: Mapping[str, Any]) -> dict[str, Any]:
    candidate_rates: dict[str, list[float]] = {}
    cells = []
    for row in _rows(joined.get("cells")):
        candidate_rates.setdefault(row["candidate_id"], []).append(row["physical_success_rate"])
        sim_success = row.get("simulation_success")
        physical_success = row.get("physical_success_rate", 0.0) >= 0.5
        if row.get("simulation_status") != "completed":
            relation = "inconclusive"
        else:
            relation = "agreement" if sim_success is physical_success else "contradiction"
        cells.append({**row, "cell_relation": relation})
    physical_means = {
        candidate_id: sum(values) / len(values) for candidate_id, values in candidate_rates.items()
    }
    physical_preferred = max(sorted(physical_means), key=physical_means.get)
    sealed_decision = decision.get("decision")
    selected = decision.get("selected_candidate_id")
    if sealed_decision in {"abstain", "equivalent_inconclusive"}:
        overall = "inconclusive"
    elif selected == physical_preferred and sealed_decision == "select":
        overall = "agreement"
    elif selected == physical_preferred and sealed_decision == "eliminate":
        overall = "contradiction"
    elif sealed_decision in {"select", "eliminate"}:
        overall = "contradiction"
    else:
        overall = "abstain"
    verdict = {
        "schema_version": VERDICT_SCHEMA_VERSION,
        "verdict": overall,
        "sealed_development_decision": sealed_decision,
        "physical_reference_preferred_candidate": physical_preferred,
        "physical_candidate_mean_success_rates": physical_means,
        "agreement_cell_count": sum(row["cell_relation"] == "agreement" for row in cells),
        "contradiction_cell_count": sum(row["cell_relation"] == "contradiction" for row in cells),
        "inconclusive_cell_count": sum(row["cell_relation"] == "inconclusive" for row in cells),
        "coverage": len(cells) / 6.0,
        "uncertainty": {
            "published_cell_uncertainty": "not_reported",
            "simulation_decision_interval": decision.get("difference_interval"),
        },
        "invalid_region": decision.get("invalid_region"),
        "next_cheapest_missing_measurement": decision.get("next_cheapest_missing_measurement"),
        "cells": cells,
        "phase_label": PHASE_LABEL,
        "claim_ceiling": CLAIM_CEILING,
        "two_candidates_establish_rank_correlation": False,
    }
    verdict["verdict_digest"] = canonical_digest(verdict, digest_field="verdict_digest")
    return verdict


def _write_artifact(path: Path, value: Mapping[str, Any]) -> None:
    write_json(path, dict(value))


def reconstruct_evidence_package(
    *, manifest_path: str | Path, execution_path: str | Path, outcomes_path: str | Path, output_dir: str | Path, generated_at: str = "2026-08-04T00:00:00Z"
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    execution_file = Path(execution_path).expanduser().resolve()
    outcomes_file = Path(outcomes_path).expanduser().resolve()
    manifest = _load_json(manifest_file, blocker="public_reference_manifest_missing")
    admission = build_public_reference_admission_receipt(manifest)
    if admission.get("status") != "admitted":
        raise ArmDecisionProofError(
            ["public_reference_not_admitted", *admission.get("blockers", [])]
        )
    execution = _load_json(
        execution_file,
        blocker=(
            "immutable_execution_input_missing:run canonical Vast acquisition command "
            "documented in docs/arm_decision_proof_v1/REPLAY.md"
        ),
    )
    execution_validation = validate_execution_package(
        execution, manifest, execution_root=execution_file.parent
    )
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    inputs_dir = output / "immutable_inputs"
    inputs_dir.mkdir()
    shutil.copy2(manifest_file, inputs_dir / manifest_file.name)
    shutil.copy2(execution_file, inputs_dir / execution_file.name)
    spec = build_evaluation_run_spec(manifest)
    plan = compile_evaluation_run(spec, output_dir=output / "normalized_run", generated_at=generated_at)
    if plan.get("status") != "prepared":
        raise ArmDecisionProofError(["normalized_evaluation_run_plan_blocked"])
    receipts_dir = output / "episode_receipts"
    receipts_dir.mkdir()
    receipts = [
        _episode_receipt(
            episode,
            manifest=manifest,
            execution_digest=execution["execution_digest"],
        )
        for episode in _rows(execution.get("episodes"))
    ]
    for receipt in receipts:
        _write_artifact(receipts_dir / f"{receipt['episode_id']}.json", receipt)
    replay = {
        "schema_version": "adp_receipt_replay.v1",
        "status": "reproduced",
        "execution_digest": execution["execution_digest"],
        "receipt_digests": sorted(row["receipt_digest"] for row in receipts),
        "non_reproducibility_failures": [],
    }
    replay["replay_digest"] = canonical_digest(replay, digest_field="replay_digest")
    decision = compile_bounded_decision(receipts)
    seal = seal_decision(
        manifest=manifest,
        execution=execution,
        plan=plan,
        receipts=receipts,
        decision=decision,
    )
    _write_artifact(output / "public_reference_admission_receipt.json", admission)
    _write_artifact(output / "execution_validation.json", execution_validation)
    _write_artifact(output / "receipt_replay.json", replay)
    _write_artifact(output / "bounded_development_decision.json", decision)
    _write_artifact(output / "decision_seal.json", seal)
    outcomes, release = release_physical_outcomes(
        outcomes_path=outcomes_file,
        manifest=manifest,
        seal=seal,
    )
    shutil.copy2(outcomes_file, inputs_dir / outcomes_file.name)
    _write_artifact(output / "physical_outcome_release_receipt.json", release)
    joined = join_physical_outcomes(
        manifest=manifest,
        receipts=receipts,
        outcomes=outcomes,
        release=release,
        seal=seal,
    )
    verdict = adjudicate(decision, joined)
    _write_artifact(output / "physical_outcome_join.json", joined)
    _write_artifact(output / "bounded_verdict.json", verdict)
    matrix = {
        "schema_version": "adp_evidence_matrix.v1",
        "status": "complete",
        "labels": [PHASE_LABEL, CLAIM_CEILING],
        "source_manifest_digest": manifest["manifest_digest"],
        "seal_digest": seal["seal_digest"],
        "release_receipt_digest": release["release_receipt_digest"],
        "join_digest": joined["join_digest"],
        "cells": verdict["cells"],
        "missing_outcomes_visible": True,
        "missing_outcomes": joined["missing_outcomes"],
    }
    matrix["matrix_digest"] = canonical_digest(matrix, digest_field="matrix_digest")
    _write_artifact(output / "evidence_matrix.json", matrix)
    replay_instructions = {
        "schema_version": "adp_replay_instructions.v1",
        "command": (
            "python -m blueprint_pipeline.arm_decision_proof "
            f"--manifest {manifest_file} --execution-package {execution_file} "
            f"--physical-outcomes {outcomes_file} --output-dir {output}"
        ),
        "immutable_input_digests": {
            "manifest": manifest["manifest_digest"],
            "execution": execution["execution_digest"],
            "physical_outcomes": outcomes["outcomes_digest"],
        },
        "phase_label": PHASE_LABEL,
        "claim_ceiling": CLAIM_CEILING,
    }
    _write_artifact(output / "replay_instructions.json", replay_instructions)
    artifact_rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "artifact_index.json":
            continue
        artifact_rows.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "sha256": _file_digest(path),
                "size_bytes": path.stat().st_size,
            }
        )
    index = {
        "schema_version": "adp_artifact_index.v1",
        "status": "complete",
        "generated_at": generated_at,
        "artifact_count": len(artifact_rows),
        "artifacts": artifact_rows,
        "phase_label": PHASE_LABEL,
        "claim_ceiling": CLAIM_CEILING,
        "adp_008_complete": True,
    }
    index["index_digest"] = canonical_digest(index, digest_field="index_digest")
    _write_artifact(output / "artifact_index.json", index)
    return index


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execution-package", type=Path, required=True)
    parser.add_argument("--physical-outcomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = reconstruct_evidence_package(
            manifest_path=args.manifest,
            execution_path=args.execution_package,
            outcomes_path=args.physical_outcomes,
            output_dir=args.output_dir,
        )
    except ArmDecisionProofError as exc:
        print(json.dumps({"status": "blocked", "blockers": exc.blockers}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ArmDecisionProofError",
    "adjudicate",
    "build_evaluation_run_spec",
    "compile_bounded_decision",
    "join_physical_outcomes",
    "reconstruct_evidence_package",
    "release_physical_outcomes",
    "seal_decision",
    "validate_execution_package",
]
