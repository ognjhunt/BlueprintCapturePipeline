"""Freeze Experiment 2 protocol and source/data integrity manifests."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import write_json
from .policy_ranking_thesis import canonical_sha256, file_sha256


EXPERIMENT_ID = "policy_ranking_thesis_experiment_2_20260727"
PREVIOUS_DIR = Path("docs/experiments/policy_ranking_thesis_20260726")


def _git(path: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *arguments], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _identity_hash(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result.pop(field, None)
    result[field] = canonical_sha256(result)
    return result


def build_protocol(previous: dict[str, Any], *, source_commit: str) -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "schema_version": "policy_ranking_thesis_experiment_2_preregistration.v1",
        "experiment_id": EXPERIMENT_ID,
        "frozen_at_local_date": "2026-07-27",
        "source_commit": source_commit,
        "historical_experiment": {
            "path": PREVIOUS_DIR.as_posix(),
            "verdict": "inconclusive",
            "immutable": True,
            "protocol_sha256": previous["protocol_sha256"],
            "heldout_labels_opened": False,
            "frozen_attempts_exhausted": True,
            "experiment_2_is_not_a_retroactive_completion": True,
        },
        "thesis": (
            "Given a captured real-site representation, a real robot task, and multiple "
            "candidate robot policies, Blueprint can predict a useful ordering of those "
            "policies substantially faster and more cheaply than exhaustively evaluating "
            "every policy physically, while abstaining when its prediction is not trustworthy."
        ),
        "embodiment": "DROID-compatible Franka Panda",
        "policies": previous["policies"],
        "evaluator": previous["evaluator"],
        "thresholds": previous["thresholds"],
        "benchmark": {
            "dataset": "RoboArena/DataDump_07-17-2026",
            "revision": previous["benchmark"]["revision"],
            "wam_rollouts": "zywu2115/OSCAR_policy_rollout",
            "wam_rollout_revision": previous["wam"]["released_rollout_revision"],
            "independent_session_cluster_count": 49,
            "policies": previous["policies"],
            "label_basis": "binary_then_partial",
            "labels_sealed_until_complete_prediction_freeze": True,
            "physical_pixels_excluded_from_provider": True,
        },
        "partitions": previous["partitions"],
        "arm_freeze": {
            "heldout_ranking_arms": [
                previous["evaluator"]["full_temporal_method"],
                previous["evaluator"]["cheap_baseline_method"],
            ],
            "evaluator_digest": "3c282af98cd968a32fa130fc0d717b7aa4d471f50ff5b7b204e2cff508671314",
            "model_snapshot": "gpt-5-2025-08-07",
            "prompt_sha256": "095f2cf31b42eb9aef165c7d1665063807abe0e7023faf7bfe198f1936497ae4",
            "sampling_contract": previous["evaluator"]["provider_configuration"],
            "new_heldout_ranking_arms_forbidden": True,
            "label_free_causal_diagnostics_are_not_ranking_arms": True,
        },
        "causal_conditioning": {
            "analysis_unit": "session_cluster",
            "channels": ["full_generated", "overlay_region", "overlay_masked_residual"],
            "placebos": [
                "zero_actions",
                "shuffled_action_order",
                "temporally_reversed_actions",
                "circularly_shifted_actions",
                "within_session_swapped_policy_actions",
            ],
            "policy_name_in_provider_prompt": False,
            "session_identity_in_provider_prompt": False,
            "identical_imagery_policy_label_leakage_prevented_by_request_deduplication": True,
            "meaningful_correlation_margin": 0.05,
            "minimum_original_correlation": 0.10,
            "minimum_validity_pass_rate": 0.80,
            "bootstrap_replicates": 10_000,
            "claim_boundary": (
                "Temporal alignment can falsify action following, but cannot by itself prove "
                "counterfactual WAM causality because alternate actions are not regenerated."
            ),
        },
        "ranking_gates": {
            "primary_pairwise_accuracy_clustered_bootstrap_lower95_gt": 0.50,
            "minimum_detectable_pairwise_accuracy_at_80pct_power": 0.6783,
            "kendall_tau_b_gt": 0.0,
            "kendall_permutation_one_sided_alpha": 0.05,
            "correct_top_policy_required": True,
            "selective_pairwise_coverage_min": 0.25,
            "selective_pairwise_accuracy_lower95_min": 0.60,
            "temporal_minus_endpoint_pairwise_accuracy_min": 0.05,
            "overlay_masked_action_alignment_lower95_gt": 0.05,
            "action_following_validity_pass_rate_lower95_min": 0.80,
            "uncertainty_error_association_required": True,
            "complete_risk_coverage_curve_required": True,
        },
        "retry_semantics": {
            "valid_response_accepted_exactly_once": True,
            "completed_request_never_resampled": True,
            "infrastructure_retries_per_request": 1,
            "invalid_structured_response_consumes_scientific_response": True,
            "provider_failure_without_usable_output_consumes_infrastructure_retry": True,
            "systemic_rejection_stop_after_consecutive_failures": 5,
            "resume_idempotent": True,
        },
        "spend": {
            "total_usd_max": 50.0,
            "model_vision_api_usd_max": 30.0,
            "gpu_build_serving_usd_max": 15.0,
            "storage_contingency_usd_max": 5.0,
            "physical_robotics_usd_max": 0.0,
            "openai_replication_projected_usd": 22.0,
            "openai_replication_hard_stop_usd": 29.0,
        },
        "stop_conditions": {
            "futility": (
                "If the complete heldout benchmark fails either ranking utility, causal "
                "conditioning, or useful abstention gates, overall support is impossible and "
                "no new GPU transfer campaign is admitted."
            ),
            "provider": "Stop on five consecutive systemic rejections before any accepted response.",
            "rights": "Fail closed on unattributable action semantics or data rights.",
            "spend": "Stop before any category or total ceiling can be exceeded.",
        },
        "captured_site_transfer": {
            "historical_input_only": True,
            "historical_experiment_outcomes_known_before_experiment_2": True,
            "historical_result_cannot_be_reclassified_as_new_blinded_evidence": True,
            "scene": "InteriorGS 0787_841244",
            "room_remained_3dgs": True,
            "local_interaction_layers_only": True,
            "new_gpu_campaign_admitted_only_if_frozen_benchmark_all_gates_pass": True,
            "site_specific_physical_accuracy_claimed": False,
        },
        "economic_rule": {
            "physical_control_time_lower_bound_is_not_full_physical_cost": True,
            "physical_monetary_cost_unmeasured": True,
            "thesis_supported_forbidden_without_attributable_time_and_cost_comparison": True,
        },
        "label_access": {
            "pilot_opened_historically": True,
            "calibration_opened_historically": True,
            "heldout_opened": False,
            "automated_file_hashing_is_not_outcome_inspection": True,
            "unseal_once_after_prediction_and_cost_freeze": True,
        },
        "component_verdict_values": ["supported", "not_supported", "inconclusive"],
        "overall_verdict_values": ["thesis_supported", "thesis_not_supported", "inconclusive"],
        "overall_verdict_rules": {
            "thesis_supported": (
                "Both components pass, causal conditioning survives controls, abstention is "
                "useful, and attributable time and cost advantages are measured."
            ),
            "thesis_not_supported": (
                "A sufficiently powered heldout result fails ranking versus chance/endpoint, "
                "causal action signal, captured transfer, or useful risk-coverage."
            ),
            "inconclusive": (
                "Required evidence is incomplete, underpowered, provenance-limited, or "
                "contradictory within authorized scope."
            ),
        },
        "claim_boundaries": {
            "blueprint_physical_robot_operation": False,
            "simulator_is_physical": False,
            "generated_video_is_ranking_fidelity": False,
            "wam_is_answer_key": False,
            "captured_site_physical_success_proven": False,
            "deployment_or_safety_readiness_proven": False,
        },
    }
    return _identity_hash(protocol, "protocol_sha256")


def historical_inventory(repo: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted((repo / PREVIOUS_DIR).rglob("*")):
        if not path.is_file():
            continue
        row: dict[str, Any] = {
            "path": path.relative_to(repo).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                row.update(
                    {
                        "json_valid": True,
                        "schema_version": payload.get("schema_version")
                        if isinstance(payload, dict)
                        else None,
                        "status": payload.get("status") if isinstance(payload, dict) else None,
                        "top_level_keys": sorted(payload) if isinstance(payload, dict) else [],
                    }
                )
            except json.JSONDecodeError:
                row["json_valid"] = False
        rows.append(row)
    result = {
        "schema_version": "policy_ranking_historical_evidence_inventory.v1",
        "historical_experiment_immutable": True,
        "artifact_count": len(rows),
        "artifacts": rows,
    }
    return _identity_hash(result, "inventory_sha256")


def environment_manifest(repo: Path, *, source_commit: str, worktree_path: Path) -> dict[str, Any]:
    dependency_files = [
        path for path in (repo / "pyproject.toml", repo / "uv.lock") if path.is_file()
    ]
    pip_freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL
    ).splitlines()
    result = {
        "schema_version": "policy_ranking_experiment_environment.v1",
        "experiment_id": EXPERIMENT_ID,
        "starting_origin_main_sha": source_commit,
        "worktree_path": str(worktree_path.resolve()),
        "experiment_branch": _git(repo, "branch", "--show-current"),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependency_lock_state": [
            {
                "path": path.relative_to(repo).as_posix(),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in dependency_files
        ],
        "installed_python_distributions": sorted(pip_freeze),
        "evaluator_models": [
            {
                "provider": "openai",
                "model_snapshot": "gpt-5-2025-08-07",
                "live_discovery_pending": True,
            }
        ],
        "policy_checkpoint_manifest": {
            "historical_openpi_inventory_path": (
                PREVIOUS_DIR / "openpi_polaris_checkpoint_inventory.json"
            ).as_posix(),
            "sha256": file_sha256(repo / PREVIOUS_DIR / "openpi_polaris_checkpoint_inventory.json"),
        },
    }
    return _identity_hash(result, "environment_sha256")


def freeze(repo: Path, output: Path, *, oscar_root: Path, roboarena_root: Path) -> None:
    previous = json.loads((repo / PREVIOUS_DIR / "preregistered_protocol.json").read_text())
    source_commit = _git(repo, "rev-parse", "HEAD")
    protocol = build_protocol(previous, source_commit=source_commit)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preregistered_protocol.json", protocol)
    write_json(output / "historical_evidence_inventory.json", historical_inventory(repo))
    write_json(
        output / "environment_and_source_manifest.json",
        environment_manifest(repo, source_commit=source_commit, worktree_path=repo),
    )
    dataset = {
        "schema_version": "policy_ranking_experiment_2_dataset_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "oscar_code": {
            "path": str(oscar_root.resolve()),
            "revision": _git(oscar_root, "rev-parse", "HEAD"),
            "license_file_sha256": file_sha256(oscar_root / "LICENSE"),
        },
        "oscar_rollouts": {
            "revision": _git(oscar_root.parent / "OSCAR_policy_rollout", "rev-parse", "HEAD"),
            "frozen_index_sha256": file_sha256(repo / PREVIOUS_DIR / "frozen_rollout_index.json"),
            "media_rights": "internal_only_no_explicit_dataset_license_found",
        },
        "roboarena": {
            "path": str(roboarena_root.resolve()),
            "revision": _git(roboarena_root, "rev-parse", "HEAD"),
            "license": "MIT",
            "heldout_metadata_parsed_or_displayed": False,
        },
        "split_manifest": previous["partitions"],
        "split_manifest_sha256": canonical_sha256(previous["partitions"]),
    }
    write_json(
        output / "dataset_and_split_manifest.json", _identity_hash(dataset, "manifest_sha256")
    )
    signature = {
        "schema_version": "policy_ranking_protocol_signature.v1",
        "experiment_id": EXPERIMENT_ID,
        "source_commit": source_commit,
        "protocol_canonical_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": file_sha256(output / "preregistered_protocol.json"),
        "signed_by": "deterministic_sha256_freeze",
        "heldout_predictions_started": False,
        "heldout_labels_opened": False,
    }
    write_json(output / "protocol_signature.json", _identity_hash(signature, "signature_sha256"))
    label_ledger = {
        "schema_version": "policy_ranking_experiment_2_label_access_ledger.v1",
        "experiment_id": EXPERIMENT_ID,
        "historical_pilot_labels_opened": True,
        "historical_calibration_labels_opened": True,
        "heldout_labels_opened": False,
        "heldout_metadata_parsed_or_displayed": False,
        "heldout_files_touched_only_by_git_checkout_lfs_and_sha256": True,
        "heldout_outcome_join_count": 0,
        "unseal_allowed_only_after_complete_prediction_cost_and_manifest_freeze": True,
    }
    write_json(output / "label_access_ledger.json", _identity_hash(label_ledger, "ledger_sha256"))
    arm_freeze = {
        "schema_version": "policy_ranking_experiment_2_arm_freeze.v1",
        "experiment_id": EXPERIMENT_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        **protocol["arm_freeze"],
        "heldout_predictions_started": False,
        "heldout_labels_opened": False,
    }
    write_json(output / "arm_freeze_manifest.json", _identity_hash(arm_freeze, "freeze_sha256"))
    previous_power = json.loads((repo / PREVIOUS_DIR / "power_analysis.json").read_text())
    power = {
        "schema_version": "policy_ranking_experiment_2_power_analysis.v1",
        "experiment_id": EXPERIMENT_ID,
        "ranking": previous_power,
        "causal_alignment": {
            "analysis_unit": "heldout_session_cluster",
            "session_cluster_count": 49,
            "target_power": 0.8,
            "one_sided_alpha": 0.05,
            "meaningful_margin": 0.05,
            "power_reestimate_from_label_free_development_alignment_allowed": True,
            "heldout_outcome_labels_used": False,
            "decision": (
                "If the development-estimated session SD makes 49 clusters unable to detect "
                "the 0.05 alignment margin at 80% power, the causal component is inconclusive "
                "and bulk provider spend is stopped before unseal."
            ),
        },
    }
    write_json(output / "power_analysis.json", _identity_hash(power, "analysis_sha256"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("freeze", nargs="?")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--oscar-root", required=True)
    parser.add_argument("--roboarena-root", required=True)
    args = parser.parse_args(argv)
    freeze(
        Path(args.repo).resolve(),
        Path(args.output).resolve(),
        oscar_root=Path(args.oscar_root).resolve(),
        roboarena_root=Path(args.roboarena_root).resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
