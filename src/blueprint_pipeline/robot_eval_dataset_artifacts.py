"""Serialization and compatibility helpers for robot-evaluation datasets."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifact_contracts import validate_sellable_artifact
from .core.common import write_json


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
    paths = {key: str((output_dir / name).resolve()) for key, name in names.items()}
    paths.update(
        manifest_path=str(manifest_path.resolve()),
        legacy_manifest_path=str(legacy_manifest_path.resolve()),
    )
    return paths
