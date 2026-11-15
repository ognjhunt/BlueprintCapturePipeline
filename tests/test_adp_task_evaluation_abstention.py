from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import blueprint_pipeline.adp_task_evaluation_abstention as abstention

from blueprint_pipeline.adp_task_evaluation_abstention import (
    TaskEvaluationAbstentionError,
    VAST_PROVIDER_ZERO_API_CALL,
    collect_vast_provider_zero_receipt,
    materialize_gaussian_heldout_ownership_abstention,
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


def _gaussian_authority_inputs() -> tuple[dict, dict, dict, dict]:
    evidence = REPO_ROOT / "docs/arm_decision_proof_v1/third_scene_dual_task_evidence"
    manifests = REPO_ROOT / "docs/arm_decision_proof_v1/manifests"
    attempt = json.loads(
        (evidence / "task_a/gaussian_excision_attempt.v1.json").read_text(
            encoding="utf-8"
        )
    )
    recovery = json.loads(
        (evidence / "task_a/gaussian_excision_recovery_readiness.v1.json").read_text(
            encoding="utf-8"
        )
    )
    task_freeze = json.loads(
        (manifests / "third_scene_840920_task_a_freeze.v1.json").read_text(
            encoding="utf-8"
        )
    )
    removal_binding = json.loads(
        (manifests / "third_scene_840920_task_a_removal_local_binding.v1.json").read_text(
            encoding="utf-8"
        )
    )
    return attempt, recovery, task_freeze, removal_binding


def _gaussian_heldout_inputs() -> tuple[dict, dict, dict, dict, dict, dict]:
    excision_freeze = {
        "schema_version": "adp009b_gaussian_excision_audit_freeze.v1",
        "status": "frozen_before_excision_execution",
        "scene": {"publisher_scene_id": "fixture_scene", "task_id": "fixture_task"},
        "learned_policy_outcomes_observed": False,
        "freeze_digest": "",
    }
    excision_freeze["freeze_digest"] = canonical_digest(
        excision_freeze, digest_field="freeze_digest"
    )
    task_freeze = {
        "schema_version": "dual_task_task_freeze.v1",
        "task_id": "fixture_task",
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "learned_policy_outcomes_accessed": False,
        "task_freeze_digest": "",
    }
    task_freeze["task_freeze_digest"] = canonical_digest(
        task_freeze, digest_field="task_freeze_digest"
    )
    attempt = {
        "schema_version": "adp_gaussian_excision_attempt_receipt.v1",
        "status": "sealed_completed_attempt",
        "execution_status": "completed",
        "freeze_digest": excision_freeze["freeze_digest"],
        "released_code_executed": True,
        "heldout_cameras_accessed_for_classification": False,
        "continuing_spend": False,
        "provider_absence_confirmed": True,
        "execution_blockers": [],
        "proof_boundaries": {
            "gaussian_contribution_evidence_completed": True,
            "gaussian_ownership_qualified": False,
            "source_removal_qualified": False,
        },
        "receipt_digest": "",
    }
    attempt["receipt_digest"] = canonical_digest(attempt, digest_field="receipt_digest")
    ownership = {
        "schema_version": "adp009b_gaussian_excision_ownership_receipt.v1",
        "freeze_digest": excision_freeze["freeze_digest"],
        "heldout_cameras_accessed_for_classification": False,
        "replacement_usd_inserted": False,
        "ownership": {"exhaustive": True, "pairwise_disjoint": True},
        "receipt_digest": "",
    }
    ownership["receipt_digest"] = canonical_digest(
        ownership, digest_field="receipt_digest"
    )
    replay = {
        "schema_version": "adp009b_gaussian_excision_ownership_replay.v1",
        "freeze_digest": excision_freeze["freeze_digest"],
        "ownership_receipt_digest": ownership["receipt_digest"],
        "execution_count": 2,
        "canonical_manifests_identical": True,
        "receipt_files_byte_identical": True,
        "output_digests_identical": True,
        "index_sets_identical": True,
        "protected_source_records_byte_identical": True,
        "gate_passed": True,
        "replay_digest": "",
    }
    replay["replay_digest"] = canonical_digest(replay, digest_field="replay_digest")
    heldout = {
        "schema_version": "adp009b_gaussian_excision_heldout_audit.v1",
        "status": "abstained_calibrated_gaussian_ownership_separation_insufficient",
        "freeze_digest": excision_freeze["freeze_digest"],
        "ownership_receipt_digest": ownership["receipt_digest"],
        "ownership_replay_digest": replay["replay_digest"],
        "heldout_gate_passed": False,
        "replacement_coverage_sweep_authorized": False,
        "smallest_missing_capability": (
            "calibrated_gaussian_ownership_separation_without_protected_scene_deletion"
        ),
        "receipt_digest": "",
    }
    heldout["receipt_digest"] = canonical_digest(heldout, digest_field="receipt_digest")
    return attempt, excision_freeze, ownership, replay, heldout, task_freeze


def test_gaussian_heldout_abstention_is_a_terminal_precontrol_receipt(
    tmp_path: Path,
) -> None:
    attempt, excision, ownership, replay, heldout, task = _gaussian_heldout_inputs()

    receipt = materialize_gaussian_heldout_ownership_abstention(
        gaussian_excision_attempt=attempt,
        excision_freeze=excision,
        gaussian_ownership=ownership,
        ownership_replay=replay,
        heldout_audit=heldout,
        task_freeze=task,
        scene_id="fixture_scene",
        output_path=tmp_path / "typed_abstention.json",
        repo_root=tmp_path,
    )

    assert receipt["status"] == "typed_evidence_backed_abstention"
    assert receipt["gaussian_contribution_evidence_completed"] is True
    assert receipt["gaussian_ownership_materialized"] is True
    assert receipt["heldout_ownership_gate_passed"] is False
    assert receipt["replacement_coverage_sweep_authorized"] is False
    assert receipt["source_removal_qualified"] is False
    assert receipt["controls_executed"] is False
    assert receipt["candidate_ids"] == ["pi05_droid", "groot_n17_droid"]
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    assert json.loads((tmp_path / "typed_abstention.json").read_text()) == receipt


@pytest.mark.parametrize(
    "mutation",
    [
        lambda attempt, excision, ownership, replay, heldout, task: heldout.update(
            {"replacement_coverage_sweep_authorized": True}
        ),
        lambda attempt, excision, ownership, replay, heldout, task: attempt[
            "proof_boundaries"
        ].update({"source_removal_qualified": True}),
        lambda attempt, excision, ownership, replay, heldout, task: task.update(
            {"task_id": "other_task"}
        ),
    ],
)
def test_gaussian_heldout_abstention_rejects_nonterminal_or_mismatched_inputs(
    mutation,
) -> None:
    attempt, excision, ownership, replay, heldout, task = _gaussian_heldout_inputs()
    mutation(attempt, excision, ownership, replay, heldout, task)
    for value, field in (
        (attempt, "receipt_digest"),
        (excision, "freeze_digest"),
        (ownership, "receipt_digest"),
        (replay, "replay_digest"),
        (heldout, "receipt_digest"),
        (task, "task_freeze_digest"),
    ):
        value[field] = canonical_digest(value, digest_field=field)

    with pytest.raises(TaskEvaluationAbstentionError):
        materialize_gaussian_heldout_ownership_abstention(
            gaussian_excision_attempt=attempt,
            excision_freeze=excision,
            gaussian_ownership=ownership,
            ownership_replay=replay,
            heldout_audit=heldout,
            task_freeze=task,
            scene_id="fixture_scene",
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
        ],
        "smallest_blocker": "released_code_inpainting_quality_admission_missing",
        "next_action": "qualify the retained Aura candidate",
        "run_digest": "",
    }
    value["run_digest"] = canonical_digest(value, digest_field="run_digest")
    return value


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


def test_provider_zero_uses_hardened_vast_api_without_cli() -> None:
    provider_zero = collect_vast_provider_zero_receipt(
        inventory_collector=lambda: {
            "provider": "vast",
            "status": "observed",
            "api_confirmed": True,
            "resources": [],
            "blockers": None,
            "live_resource_count": 0,
            "raw_provider_response_recorded": False,
        }
    )

    assert provider_zero["api_command"] == VAST_PROVIDER_ZERO_API_CALL
    assert provider_zero["provider_zero"] is True
    assert provider_zero["global_live_resource_count"] == 0


def test_provider_zero_refuses_ambiguous_or_failed_api_collectors() -> None:
    with pytest.raises(TaskEvaluationAbstentionError, match="collector_ambiguous"):
        collect_vast_provider_zero_receipt(
            command_runner=lambda argv: subprocess.CompletedProcess(
                argv, returncode=0, stdout="[]", stderr=""
            ),
            inventory_collector=lambda: {},
        )
    with pytest.raises(TaskEvaluationAbstentionError, match="api_query_failed"):
        collect_vast_provider_zero_receipt(
            inventory_collector=lambda: {
                "provider": "vast",
                "status": "blocked",
                "api_confirmed": False,
                "resources": [],
                "blockers": ["vast_api_key_missing"],
                "live_resource_count": None,
                "raw_provider_response_recorded": False,
            }
        )


def test_provider_zero_module_cli_writes_exclusive_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = collect_vast_provider_zero_receipt(
        inventory_collector=lambda: {
            "provider": "vast",
            "status": "observed",
            "api_confirmed": True,
            "resources": [],
            "blockers": None,
            "live_resource_count": 0,
            "raw_provider_response_recorded": False,
        }
    )
    monkeypatch.setattr(abstention, "collect_vast_provider_zero_receipt", lambda: receipt)
    output = tmp_path / "provider-zero.json"

    assert abstention.main(["--collect-vast-provider-zero", str(output)]) == 0
    assert json.loads(output.read_text()) == receipt
    with pytest.raises(FileExistsError):
        abstention.main(["--collect-vast-provider-zero", str(output)])
