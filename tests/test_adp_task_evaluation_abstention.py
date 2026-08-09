from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.adp_task_evaluation_abstention import (
    TaskEvaluationAbstentionError,
    collect_vast_provider_zero_receipt,
    materialize_native_gate_task_evaluation_abstention,
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


def _write_native_gate_files(root: Path) -> tuple[str, str, str]:
    construction = {
        "schema_version": "articulated_excision_join.v1",
        "status": "join_admitted",
        "claim_boundary": {"native_simulator_qualified": False},
        "receipt_digest": "",
    }
    construction["receipt_digest"] = canonical_digest(
        construction, digest_field="receipt_digest"
    )
    construction_path = root / "construction.json"
    construction_path.write_text(json.dumps(construction), encoding="utf-8")

    teardown_path = root / "teardown.json"
    teardown = {
        "schema_version": "vast_teardown_manifest.v1",
        "status": "completed",
        "vast_instance_ids": [123],
        "teardown_actions_performed": [
            {
                "instance_id": 123,
                "action": "destroy_instance",
                "http_status_code": 200,
                "status": "completed",
            }
        ],
        "runner_gpu_teardown_completed": True,
        "continuing_spend_from_this_run": False,
    }
    teardown_path.write_text(json.dumps(teardown), encoding="utf-8")

    adapter_path = root / "adapter.json"
    adapter = {
        "schema_version": "adp009d_franka_vast_run.v1",
        "status": "blocked",
        "attempt_number": 1,
        "native_control_result_path": None,
        "teardown_manifest_path": str(teardown_path.resolve()),
        "estimated_cost_usd": 0.007044,
        "hard_cap_usd": 0.8,
        "hard_ttl_seconds": 3600,
        "retry_cap": 0,
        "candidate_policy_query_expected": False,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "blockers": [
            "adp009d_runtime_not_completed",
            "vast_heartbeat_container_missing",
        ],
    }
    adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
    return construction_path.name, adapter_path.name, teardown_path.name


def test_native_gate_abstention_is_derived_from_null_teardown_and_api_zero(
    tmp_path: Path,
) -> None:
    construction, adapter, teardown = _write_native_gate_files(tmp_path)
    provider_zero = collect_vast_provider_zero_receipt(
        command_runner=lambda argv: subprocess.CompletedProcess(
            argv, returncode=0, stdout="[]\n", stderr=""
        )
    )

    receipt = materialize_native_gate_task_evaluation_abstention(
        scene_task_freeze=_freeze(),
        evidence_root=tmp_path,
        construction_join_relative_path=construction,
        native_adapter_relative_path=adapter,
        teardown_relative_path=teardown,
        provider_zero_receipt=provider_zero,
    )

    assert receipt["smallest_missing_capability"] == (
        "native_articulated_asset_diagnostic_unobserved:"
        "vast_heartbeat_container_missing"
    )
    assert receipt["native_asset_opened"] is False
    assert receipt["controls_executed"] is False
    assert receipt["paid_attempt"]["estimated_cost_usd"] == pytest.approx(0.007044)
    assert receipt["provider_zero"]["provider_zero"] is True
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_native_gate_abstention_refuses_policy_or_control_result(tmp_path: Path) -> None:
    construction, adapter, teardown = _write_native_gate_files(tmp_path)
    adapter_path = tmp_path / adapter
    value = json.loads(adapter_path.read_text(encoding="utf-8"))
    value["native_control_result_path"] = "results/native_control.json"
    adapter_path.write_text(json.dumps(value), encoding="utf-8")
    provider_zero = collect_vast_provider_zero_receipt(
        command_runner=lambda argv: subprocess.CompletedProcess(
            argv, returncode=0, stdout="[]", stderr=""
        )
    )

    with pytest.raises(
        TaskEvaluationAbstentionError,
        match="native_gate_adapter_not_infrastructure_null",
    ):
        materialize_native_gate_task_evaluation_abstention(
            scene_task_freeze=_freeze(),
            evidence_root=tmp_path,
            construction_join_relative_path=construction,
            native_adapter_relative_path=adapter,
            teardown_relative_path=teardown,
            provider_zero_receipt=provider_zero,
        )
