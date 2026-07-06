from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.lerobot_policy_family import (
    create_scripted_baseline_checkpoint,
)
from blueprint_pipeline.real_policy_closed_loop_rollout import (
    SIMULATOR_OUTPUT_SCHEMA_VERSION,
    SUBSTRATE,
    run_real_policy_closed_loop_rollout,
)

pytest.importorskip("mujoco")


def _matrix(tmp_path: Path, variations: list[str]) -> Path:
    payload = {
        "runs": [
            {
                "scenario_eval_run_id": f"run-{index}",
                "task_id": "place_return_in_bin",
                "scenario_id": "scenario_tabletop",
                "variation_name": variation,
            }
            for index, variation in enumerate(variations)
        ]
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_rollout_produces_measured_attempts_with_policy_in_the_loop(
    tmp_path: Path,
) -> None:
    checkpoint = create_scripted_baseline_checkpoint(tmp_path / "ckpt")
    matrix = _matrix(tmp_path, ["normal", "object_rotation"])
    payload = run_real_policy_closed_loop_rollout(
        checkpoint_dir=checkpoint,
        scenario_eval_matrix_path=matrix,
        output_path=tmp_path / "out.json",
        max_seconds=10.0,
    )

    assert payload["schema_version"] == SIMULATOR_OUTPUT_SCHEMA_VERSION
    assert payload["status"] == "completed"
    assert payload["substrate"] == SUBSTRATE
    assert payload["policy_in_the_loop"] is True
    assert payload["policy_transport"] == "in_process_checkpoint_policy"
    assert payload["simulator_execution_proven"] is True
    assert payload["physics_step_count"] > 0

    attempts = payload["attempts"]
    assert [a["scenario_eval_run_id"] for a in attempts] == ["run-0", "run-1"]
    for attempt in attempts:
        outcome = attempt["task_outcome"]
        assert outcome["success_criteria_source"] == "measured_simulator_state"
        criteria = outcome["success_criteria"]
        assert set(criteria) == {
            "language_following",
            "object_lifting",
            "object_placing",
        }
        for detail in criteria.values():
            assert isinstance(detail["passed"], bool)
        assert attempt["metrics"]["policy_query_count"] > 0
        assert attempt["metrics"]["physics_step_count"] > 0
        assert Path(attempt["artifact_paths"]["control_stream"]).is_file()
        # task_success must equal the conjunction of measured criteria — the
        # policy has no way to declare success.
        assert attempt["task_success"] == all(
            detail["passed"] for detail in criteria.values()
        )

    requery_path = Path(payload["policy_requery_trace_path"])
    assert requery_path.is_file()
    rows = [
        json.loads(line)
        for line in requery_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows and all(row["transport"] == "in_process_checkpoint_policy" for row in rows)
    assert payload["sc3_alignment"]["success_criteria_source"] == (
        "measured_simulator_state_not_generated_video"
    )


def test_rollout_is_deterministic(tmp_path: Path) -> None:
    checkpoint = create_scripted_baseline_checkpoint(tmp_path / "ckpt")
    matrix = _matrix(tmp_path, ["normal"])
    first = run_real_policy_closed_loop_rollout(
        checkpoint_dir=checkpoint,
        scenario_eval_matrix_path=matrix,
        output_path=tmp_path / "a.json",
        max_seconds=10.0,
    )
    second = run_real_policy_closed_loop_rollout(
        checkpoint_dir=checkpoint,
        scenario_eval_matrix_path=matrix,
        output_path=tmp_path / "b.json",
        max_seconds=10.0,
    )
    for key in ("task_success", "deterministic_seed"):
        assert first["attempts"][0][key] == second["attempts"][0][key]
    assert (
        first["attempts"][0]["task_outcome"]["final_target_error_m"]
        == second["attempts"][0]["task_outcome"]["final_target_error_m"]
    )


def test_rollout_fails_closed_without_loadable_policy(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path, ["normal"])
    bogus = tmp_path / "bogus_ckpt"
    bogus.mkdir()
    (bogus / "config.json").write_text(json.dumps({"type": "act"}), encoding="utf-8")
    payload = run_real_policy_closed_loop_rollout(
        checkpoint_dir=bogus,
        scenario_eval_matrix_path=matrix,
        output_path=tmp_path / "out.json",
    )
    assert payload["status"] == "blocked"
    assert payload["simulator_execution_proven"] is False
    assert payload["attempts"] == []
    assert "policy_type_requires_torch_inference_runtime" in payload["blockers"]
    written = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert written["status"] == "blocked"


def test_variation_honesty_labels(tmp_path: Path) -> None:
    checkpoint = create_scripted_baseline_checkpoint(tmp_path / "ckpt")
    matrix = _matrix(tmp_path, ["lighting_variation", "blocked_path"])
    payload = run_real_policy_closed_loop_rollout(
        checkpoint_dir=checkpoint,
        scenario_eval_matrix_path=matrix,
        output_path=tmp_path / "out.json",
        max_seconds=6.0,
    )
    by_variation = {a["variation_name"]: a for a in payload["attempts"]}
    assert by_variation["lighting_variation"]["variation_physically_modeled"] is False
    assert by_variation["blocked_path"]["variation_physically_modeled"] is True
    boundary = payload["claim_boundary"]
    assert boundary["classical_sim_rollout_is_not_physical_robot_proof"] is True
    assert boundary["task_success_labels_are_simulator_measured_only"] is True
    assert boundary["public_claim_upgrade_allowed"] is False
    embodiment = payload["embodiment_contract"]
    assert embodiment["not_a_humanoid_or_unitree_g1_claim"] is True
