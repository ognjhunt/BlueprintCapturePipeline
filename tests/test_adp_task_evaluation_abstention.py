from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp_task_evaluation_abstention import (
    TaskEvaluationAbstentionError,
    materialize_task_evaluation_abstention,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _freeze() -> dict:
    return json.loads(
        (
            REPO_ROOT
            / "docs/arm_decision_proof_v1/manifests"
            / "second_scene_840796_scene_task_freeze.v1.json"
        ).read_text(encoding="utf-8")
    )


def _run(freeze: dict) -> dict:
    value = {
        "schema_version": "articulated_public_scene_construction_run.v2",
        "status": "typed_abstention_before_simready_build",
        "scene": {
            "publisher_scene_id": "840796",
            "target_instance_id": "123",
            "task_kind": "articulated_open_close",
            "freeze_digest": freeze["freeze_digest"],
        },
        "frozen_candidates": ["pi05_droid", "groot_n17_droid"],
        "stage_receipts": {
            "aura_execution": "sha256:" + "a" * 64,
            "joint_agent_execution": "sha256:" + "b" * 64,
        },
        "stage_status": {
            "released_code_inpainting_executed": True,
            "joint_agent_execution_attempted": True,
            "joint_agent_execution_abstained": True,
            "joint_agent_topology_executed": False,
            "simready_replacement_materialized": False,
            "native_simulator_qualified": False,
            "controls_executed": False,
            "learned_candidates_executed": False,
        },
        "blockers": [
            "released_code_inpainting_quality_admission_missing",
            "joint_agent_topology_execution_abstained:joint_agent_local_ovrtx_renderer_not_ready",
        ],
        "smallest_blocker": "released_code_inpainting_quality_admission_missing",
        "next_action": "qualify the retained Aura candidate",
        "run_digest": "",
    }
    value["run_digest"] = canonical_digest(value, digest_field="run_digest")
    return value


def test_seals_pre_episode_completion_path_b_with_exact_candidates() -> None:
    freeze = _freeze()
    receipt = materialize_task_evaluation_abstention(
        construction_run=_run(freeze), scene_task_freeze=freeze
    )

    assert receipt["status"] == "typed_evidence_backed_abstention"
    assert receipt["candidate_ids"] == ["pi05_droid", "groot_n17_droid"]
    assert receipt["controls_executed"] is False
    assert receipt["learned_candidate_episodes_executed"] is False
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda run: run["stage_receipts"].update({"joint_agent_execution": None}),
        lambda run: run["stage_status"].update({"controls_executed": True}),
        lambda run: run.update({"blockers": ["joint_agent_topology_execution_missing"]}),
    ],
)
def test_refuses_unobserved_or_post_episode_abstention(mutation) -> None:
    freeze = _freeze()
    run = _run(freeze)
    mutation(run)
    run["smallest_blocker"] = run["blockers"][0]
    run["run_digest"] = canonical_digest(run, digest_field="run_digest")

    with pytest.raises(TaskEvaluationAbstentionError):
        materialize_task_evaluation_abstention(
            construction_run=copy.deepcopy(run), scene_task_freeze=freeze
        )
