from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.second_scene_freeze import (
    EVIDENCE_ROLES,
    SOURCE_ROLES,
    SecondSceneFreezeError,
    validate_second_scene_freeze,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _record(role: str) -> dict:
    return {
        "role": role,
        "relative_path": f"evidence/{role}.json",
        "size_bytes": 10,
        "sha256": "sha256:" + "a" * 64,
    }


def _freeze() -> dict:
    value = {
        "schema_version": "second_scene_scene_task_freeze.v1",
        "program_id": "arm-decision-proof-v1",
        "status": "frozen_for_construction_before_learned_outcomes",
        "scene": {"publisher_scene_id": "840796"},
        "rights": {
            "declared_use_scope": "noncommercial_internal_research",
            "interiorgs_revision": "revision",
            "raw_dataset_redistribution_allowed": False,
            "external_provider_upload_authorized": False,
            "commercial_use_allowed": False,
        },
        "rights_authority_record": {
            "relative_path": "rights.json",
            "size_bytes": 10,
            "sha256": "sha256:" + "b" * 64,
        },
        "source_artifacts": [_record(role) for role in sorted(SOURCE_ROLES)],
        "evidence_artifacts": [_record(role) for role in sorted(EVIDENCE_ROLES)],
        "task_spec": {
            "schema_version": "adp_task_spec.v1",
            "task_kind": "articulated_open_close",
            "target_joint_id": "refrigerator_upper_door_hinge",
            "joint_reset_positions_rad": {
                "refrigerator_upper_door_hinge": 0.0,
                "refrigerator_lower_door_hinge": 0.0,
            },
            "target_success_interval_rad": [0.785398163, 0.959931089],
            "joint_hard_limits_rad": {
                "refrigerator_upper_door_hinge": [0.0, 1.570796327],
                "refrigerator_lower_door_hinge": [0.0, 1.570796327],
            },
            "settle_window_samples": 40,
            "maximum_settled_target_speed_rad_s": 0.05,
            "non_task_joint_motion_tolerance_rad": 0.001,
            "movement_epsilon_rad": 0.0001,
            "reset_tolerance_rad": 0.0001,
        },
        "seeds": [2026080800, 2026080801],
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "learned_policy_outcomes_consulted": False,
        "new_inpainting_outcomes_consulted": False,
        "scenario_materialization_authorized": False,
        "freeze_digest": "",
    }
    value["freeze_digest"] = canonical_digest(value, digest_field="freeze_digest")
    return value


def test_freeze_binds_new_scene_one_target_joint_and_exact_candidates(tmp_path) -> None:
    freeze = _freeze()

    assert validate_second_scene_freeze(
        freeze, repo_root=tmp_path, data_root=tmp_path, verify_files=False
    ) == freeze


def test_checked_freeze_is_outcome_blind_and_not_evaluation_authority() -> None:
    path = (
        REPO_ROOT
        / "docs/arm_decision_proof_v1/manifests"
        / "second_scene_840796_scene_task_freeze.v1.json"
    )
    freeze = json.loads(path.read_text(encoding="utf-8"))

    validated = validate_second_scene_freeze(
        freeze, repo_root=REPO_ROOT, data_root=REPO_ROOT, verify_files=False
    )

    assert validated["scene"]["publisher_scene_id"] == "840796"
    assert validated["task_spec"]["target_joint_id"] == (
        "refrigerator_upper_door_hinge"
    )
    assert validated["construction_authorized"] is True
    assert validated["scenario_materialization_authorized"] is False
    assert validated["evaluation_authorized"] is False
    assert validated["learned_policy_outcomes_consulted"] is False


def test_freeze_rejects_empty_scene_or_policy_outcome_leakage(tmp_path) -> None:
    freeze = _freeze()
    freeze["scene"]["publisher_scene_id"] = ""
    freeze["learned_policy_outcomes_consulted"] = True
    freeze["freeze_digest"] = canonical_digest(freeze, digest_field="freeze_digest")

    with pytest.raises(SecondSceneFreezeError) as caught:
        validate_second_scene_freeze(
            freeze, repo_root=tmp_path, data_root=tmp_path, verify_files=False
        )

    assert "freeze_scene_identity_invalid" in caught.value.errors
    assert "freeze_policy_outcome_leakage" in caught.value.errors


def test_freeze_accepts_non_refrigerator_articulated_joint_names(tmp_path) -> None:
    freeze = _freeze()
    task = freeze["task_spec"]
    task["target_joint_id"] = "drawer_slide"
    task["joint_reset_positions_rad"] = {"drawer_slide": 0.0, "safety_latch": 0.0}
    task["joint_hard_limits_rad"] = {
        "drawer_slide": [0.0, 1.570796327],
        "safety_latch": [0.0, 1.570796327],
    }
    freeze["freeze_digest"] = canonical_digest(freeze, digest_field="freeze_digest")

    validated = validate_second_scene_freeze(
        freeze, repo_root=tmp_path, data_root=tmp_path, verify_files=False
    )
    assert validated["task_spec"]["target_joint_id"] == "drawer_slide"


def test_freeze_rejects_candidate_substitution(tmp_path) -> None:
    freeze = copy.deepcopy(_freeze())
    freeze["candidate_ids"][1] = "replacement_policy"
    freeze["freeze_digest"] = canonical_digest(freeze, digest_field="freeze_digest")

    with pytest.raises(SecondSceneFreezeError, match="freeze_candidate_pair_invalid"):
        validate_second_scene_freeze(
            freeze, repo_root=tmp_path, data_root=tmp_path, verify_files=False
        )
