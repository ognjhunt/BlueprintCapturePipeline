"""Serialization and compatibility helpers for robot-evaluation datasets."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifact_contracts import validate_sellable_artifact
from .core.common import read_json_any, write_json


ROBOT_EVAL_DATASET_MANIFEST_COMPATIBILITY_ALIAS = {
    "legacy_filename": "real_site_robot_eval_dataset_manifest.json",
    "canonical_filename": "robot_eval_dataset_manifest.json",
    "sunset_not_before": "2026-08-21",
    "removal_condition": "all_consumers_confirm_robot_eval_dataset_manifest_json",
}


def validate_and_write_robot_eval_cards(
    output_dir: Path,
    *,
    site_card: Mapping[str, Any],
    task_cards: Mapping[str, Any],
    scenario_cards: Mapping[str, Any],
    eval_cards: Mapping[str, Any],
) -> None:
    """Validate the sellable card boundary before committing any card file."""

    cards = {
        "site_card": site_card,
        "task_cards": task_cards,
        "scenario_cards": scenario_cards,
        "eval_cards": eval_cards,
    }
    for artifact_type, payload in cards.items():
        validate_sellable_artifact(artifact_type, payload)
    for artifact_type, payload in cards.items():
        write_json(output_dir / f"{artifact_type}.json", payload)


def robot_eval_result_artifact_paths(
    output_dir: Path,
    *,
    manifest_path: Path,
    legacy_manifest_path: Path,
    evaluate_recorded_evidence: bool = True,
) -> dict[str, str]:
    """Return the stable path projection exposed by the dataset builder."""

    names = {
        "site_card_path": "site_card.json",
        "task_cards_path": "task_cards.json",
        "scenario_cards_path": "scenario_cards.json",
        "eval_cards_path": "eval_cards.json",
        "annotation_backlog_path": "annotation_backlog.json",
        "proof_boundaries_path": "proof_boundaries.json",
        "methodology_path": "eval_methodology_summary.md",
        "prediction_outcome_ledger_path": "prediction_outcome_ledger.json",
        "prediction_vs_actual_summary_path": "prediction_vs_actual_summary.json",
        "recorded_trace_eval_report_path": "recorded_trace_eval_report.json",
        "task_thresholds_path": "task_thresholds.json",
        "publication_readiness_path": "publication_readiness.json",
        "rights_packet_path": "rights_packet.json",
        "rights_ledger_path": "rights_ledger.json",
        "robot_team_test_submission_modalities_path": (
            "robot_team_test_submission_modalities.json"
        ),
    }
    if not evaluate_recorded_evidence:
        for key in ("prediction_vs_actual_summary_path", "recorded_trace_eval_report_path"):
            names.pop(key)
    paths = {key: str((output_dir / name).resolve()) for key, name in names.items()}
    paths.update(
        manifest_path=str(manifest_path.resolve()),
        legacy_manifest_path=str(legacy_manifest_path.resolve()),
    )
    return paths


def robot_eval_dataset_output_paths(
    output_dir: Path, *, evaluate_recorded_evidence: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    """Project preparation artifacts separately from retained evaluation evidence."""
    output_paths = {
        "robot_eval_dataset_manifest": "robot_eval_dataset_manifest.json",
        "site_card": "site_card.json",
        "task_cards": "task_cards.json",
        "scenario_cards": "scenario_cards.json",
        "eval_cards": "eval_cards.json",
        "annotation_backlog": "annotation_backlog.json",
        "proof_boundaries": "proof_boundaries.json",
        "legacy_real_site_robot_eval_dataset_manifest": "real_site_robot_eval_dataset_manifest.json",
        "robot_task_library": "robot_task_library.json",
        "task_ontology_v1": "task_ontology_v1.json",
        "scenario_library": "scenario_library.json",
        "scenario_family_library": "scenario_family_library.json",
        "robot_pov_evidence_requirements": "robot_pov_evidence_requirements.json",
        "human_demo_evidence_requirements": "human_demo_evidence_requirements.json",
        "robot_eval_inputs_evidence_contract": "robot_eval_inputs_evidence_contract.json",
        "robot_team_test_submission_modalities": "robot_team_test_submission_modalities.json",
        "failure_taxonomy": "failure_taxonomy.json",
        "prediction_outcome_ledger": "prediction_outcome_ledger.json",
        "prediction_vs_actual_summary": "prediction_vs_actual_summary.json",
        "scoring_methodology": "scoring_methodology.json",
        "task_thresholds": "task_thresholds.json",
        "publication_readiness": "publication_readiness.json",
        "recorded_trace_eval_report": "recorded_trace_eval_report.json",
        "policy_eval_report": "policy_eval_report.json",
        "rights_packet": "rights_packet.json",
        "rights_ledger": "rights_ledger.json",
        "eval_methodology_summary": "eval_methodology_summary.md",
        "cpu_preflight_scorecard": "../simulation_automation/cpu_preflight_scorecard.json",
        "episode_spec_manifest": "../simulation_automation/episode_spec_manifest.json",
        "cpu_simulator_preflight_manifest": (
            "../simulation_automation/cpu_simulator_preflight_manifest.json"
        ),
    }
    retained_evaluation_evidence = {}
    if not evaluate_recorded_evidence:
        for name in ("recorded_trace_eval_report", "policy_eval_report", "prediction_vs_actual_summary"):
            filename = output_paths.pop(name)
            path = output_dir / filename
            previous = read_json_any(path) if path.is_file() else None
            if isinstance(previous, Mapping) and previous and previous.get("artifact_purpose") != "evaluation_preparation":
                retained_evaluation_evidence[name] = filename
    return output_paths, retained_evaluation_evidence


def write_robot_eval_reports(
    output_dir: Path, *, prediction_vs_actual_summary: Mapping[str, Any],
    recorded_trace_eval_report: Mapping[str, Any], evaluate_recorded_evidence: bool,
) -> None:
    """Preparation cannot replace performance evidence from an earlier run."""
    reports = {
        "prediction_vs_actual_summary.json": prediction_vs_actual_summary,
        "recorded_trace_eval_report.json": recorded_trace_eval_report,
        "policy_eval_report.json": recorded_trace_eval_report,
    }
    for filename, payload in reports.items():
        path = output_dir / filename
        if evaluate_recorded_evidence or not path.exists():
            write_json(path, payload)


def robot_eval_methodology_summary(
    *,
    dataset_statuses: list[str],
    task_count: int,
    scenario_count: int,
    record_count: int,
) -> str:
    statuses = ", ".join(dataset_statuses)
    return "\n".join(
        [
            "# Real-Site Robot Evaluation Dataset Methodology",
            "",
            "Status: repo-local deterministic contract. No live provider jobs, simulator runs, "
            "model downloads, sends, payments, deployments, or public-claim upgrades were performed.",
            "",
            "## Scope",
            "",
            "This dataset layer defines robot tasks, scenario records, evidence requirements, "
            "failure labels, and prediction-vs-actual ledger fields for one capture-backed site "
            "package. It is advisory until actual robot POV, human demo, action-log, rights/privacy, "
            "and outcome evidence exists.",
            "",
            "## Current Counts",
            "",
            f"- Tasks: {task_count}",
            f"- Scenarios: {scenario_count}",
            f"- Prediction/outcome records: {record_count}",
            f"- Dataset statuses: {statuses}",
            "",
            "## Evaluation Method",
            "",
            "1. Define task records from `evaluation_prep/task_anchor_manifest.json`.",
            "2. Pair each task with available robot profiles to form scenario records.",
            "3. Attach local review sources such as simready, Marble, or Cosmos preflight as "
            "prediction inputs only.",
            "4. Require robot POV, human-demo, action-log, and actual-outcome records before "
            "calibration or operational conclusions.",
            "5. Use `failure_taxonomy.json` IDs for every failed or ambiguous attempt.",
            "6. Keep WebApp display advisory-only unless owner-system proof supports a stronger "
            "request-scoped claim.",
            "",
            "## Blocked Claim Boundary",
            "",
            "This artifact does not prove simulator execution, generated-world rank fidelity, off-scope validation, "
            "provider execution, or deployment outcomes.",
            "",
        ]
    )
