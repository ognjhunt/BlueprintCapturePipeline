from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.lerobot_policy_family import (
    SCRIPTED_PICK_PLACE_FAMILY_ID,
    create_scripted_baseline_checkpoint,
)
from blueprint_pipeline.real_policy_family_eval_harness import (
    default_groot_libero_family_config,
    default_family_config,
    family_adapter_command,
    register_family_in_validation_ladder,
    rollout_simulator_command,
    run_real_policy_family_eval,
    verify_real_substrate_task_eval_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _real_job_dir(tmp_path: Path) -> Path:
    job_dir = tmp_path / "job"
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "backend": "mujoco",
            "simulator_execution_proven": True,
            "attempts": [
                {"runner": "command_adapter", "task_success": True},
                {"runner": "command_adapter", "task_success": False},
            ],
            "task_success_summary": {"task_success_rate": 0.5},
        },
    )
    _write_json(job_dir / "simulator_service_result.json", {"framework": "mujoco"})
    _write_json(
        job_dir / "mujoco_simulator_output.json",
        {
            "policy_in_the_loop": True,
            "substrate": "classical_sim_mujoco",
            "attempts": [
                {
                    "task_outcome": {
                        "success_criteria_source": "measured_simulator_state"
                    }
                }
            ],
        },
    )
    _write_json(
        job_dir / "task_eval_run_report.json",
        {
            "status": "ready_review_required",
            "evidence_level": "no_claim",
            "scorecard": {
                "conditions": [{"trials": 2, "successes": 1}],
            },
        },
    )
    _write_json(
        job_dir / "sc3_eval_protocol.json",
        {"data_requirements": {"policy_requery_trace": {"status": "completed"}}},
    )
    return job_dir


def test_verify_accepts_real_substrate_job_dir(tmp_path: Path) -> None:
    verification = verify_real_substrate_task_eval_report(_real_job_dir(tmp_path))
    assert verification["real_rollout_report"] is True
    assert verification["blockers"] == []
    assert verification["attempt_count"] == 2
    assert verification["scorecard_trials"] == 2


def test_verify_rejects_fixture_and_empty_reports(tmp_path: Path) -> None:
    job_dir = _real_job_dir(tmp_path)
    _write_json(job_dir / "simulator_service_result.json", {"framework": "fixture"})
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "backend": "fixture",
            "simulator_execution_proven": False,
            "attempts": [{"runner": "fixture"}],
        },
    )
    verification = verify_real_substrate_task_eval_report(job_dir)
    assert verification["real_rollout_report"] is False
    blockers = set(verification["blockers"])
    assert "simulator_framework_is_fixture" in blockers
    assert "attempt_runner_not_command_adapter:fixture" in blockers
    assert "attempt_trace_backend_not_mujoco" in blockers
    assert "simulator_execution_not_proven" in blockers

    empty = verify_real_substrate_task_eval_report(tmp_path / "missing")
    assert empty["real_rollout_report"] is False
    assert "normalized_attempt_trace_empty" in empty["blockers"]


def test_gpu_policy_swap_is_config_only(tmp_path: Path) -> None:
    cpu_family = default_family_config(
        family_id="blueprint_scripted_pick_place_v1",
        checkpoint_dir=str(tmp_path / "cpu_ckpt"),
    )
    gpu_family = default_family_config(
        family_id="lerobot_act_family",
        checkpoint_dir=str(tmp_path / "act_ckpt"),
        adapter_command="python -m robot_team.act_inference --device cuda",
        requires_gpu=True,
    )
    # Same schema, same harness entrypoints — only data differs.
    assert set(cpu_family) == set(gpu_family)

    cpu_command = rollout_simulator_command(family=cpu_family)
    gpu_command = rollout_simulator_command(family=gpu_family)
    assert "real_policy_closed_loop_rollout" in cpu_command
    assert "real_policy_closed_loop_rollout" in gpu_command
    assert "--adapter-command" in gpu_command
    assert "--adapter-command" not in cpu_command
    assert family_adapter_command(family=gpu_family) == (
        "python -m robot_team.act_inference --device cuda"
    )


def test_groot_libero_family_config_is_gpu_integration_proof() -> None:
    family = default_groot_libero_family_config(python_executable="/usr/bin/python3")

    assert family["family_id"] == "nvidia_groot_n17_lerobot_libero_10_640"
    assert family["checkpoint_dir"] == "nvidia/gr00t17-lerobot-libero_10-640"
    assert family["requires_gpu"] is True
    assert family["adapter_mode"] == "persistent"
    assert family["render_obs_frames"] is True
    assert family["chunk_size"] == 16
    assert family["executed_horizon"] == 16
    assert family["integration_proof_scope"] == "libero_panda_groot_integration_proof"
    assert "--serve" in family["adapter_command"]
    assert "blueprint_pipeline.lerobot_torch_policy_adapter" in family["adapter_command"]
    assert family["gpu_runtime"]["python_package_extra"] == "lerobot[groot]"
    assert family["gpu_runtime"]["not_cpu_baseline_lane"] is True
    assert family["source_policy"]["embodiment_tag"] == "libero_sim"
    assert family["source_policy"]["action_decode_transform"] == "libero"
    scoring = family["meaningful_manipulator_scoring"]
    assert scoring["workflow_projection_status"] == "available_for_integration_plumbing"
    assert scoring["required_for_quality_claim"] == (
        "libero_panda_simulator_bridge_or_panda_task_evaluator"
    )
    assert scoring["blueprint_tabletop_proxy_is_panda_evaluator"] is False
    boundary = family["claim_boundary"]
    assert boundary["learned_policy_invocation_proof_target"] is True
    assert boundary["panda_or_libero_task_success_proven"] is False
    assert boundary["meaningful_manipulator_scoring_proven"] is False
    assert boundary["blueprint_site_task_success_proven"] is False
    assert boundary["physical_robot_readiness_proven"] is False
    assert boundary["buyer_facing_deployment_claim_allowed"] is False


def test_ladder_registration_is_validation_only_and_never_production(
    tmp_path: Path,
) -> None:
    checkpoint = create_scripted_baseline_checkpoint(tmp_path / "ckpt")
    family = default_family_config(
        family_id=SCRIPTED_PICK_PLACE_FAMILY_ID, checkpoint_dir=str(checkpoint)
    )
    registry = register_family_in_validation_ladder(
        job_dir=tmp_path,
        family=family,
        checkpoint_sha256="a" * 64,
        verification={"real_rollout_report": True},
        generated_at="2026-07-05T00:00:00+00:00",
    )
    assert registry["registered_in"] == "validation_ladder_only"
    production = registry["production_candidate_registration"]
    assert production["registered"] is False
    assert production["registry"] == "UNITREE_ACTION_COMMAND_CANDIDATES"
    assert registry["gpu_swap_contract"]["config_only"] is True
    assert registry["claim_boundary"]["public_claim_upgrade_allowed"] is False

    ladder = json.loads(
        (tmp_path / "real_policy_family_ranking_ladder.json").read_text(
            encoding="utf-8"
        )
    )
    assert ladder["expected_ranking"][0] == SCRIPTED_PICK_PLACE_FAMILY_ID
    assert ladder["inner_command_configured"] is True

    # The production registry itself must remain untouched by registration.
    from blueprint_pipeline.unitree_lerobot_policy_runtime import (
        UNITREE_ACTION_COMMAND_CANDIDATES,
    )

    candidate_ids = {
        candidate.get("candidate_id") for candidate in UNITREE_ACTION_COMMAND_CANDIDATES
    }
    assert SCRIPTED_PICK_PLACE_FAMILY_ID not in candidate_ids


def test_harness_fails_closed_without_simulator_execution_optin(
    tmp_path: Path,
) -> None:
    checkpoint = create_scripted_baseline_checkpoint(tmp_path / "ckpt")
    family = default_family_config(
        family_id=SCRIPTED_PICK_PLACE_FAMILY_ID, checkpoint_dir=str(checkpoint)
    )
    manifest = run_real_policy_family_eval(
        capture_root=tmp_path / "scenes" / "s" / "captures" / "c",
        job_id="job-blocked",
        family_config=family,
        allow_simulator_execution=False,
    )
    assert manifest["status"] == "blocked"
    assert "simulator_execution_not_allowed_by_caller" in manifest["blockers"]


@pytest.mark.slow
@pytest.mark.integration
def test_full_real_policy_family_eval_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    checkpoint = create_scripted_baseline_checkpoint(tmp_path / "ckpt")
    family = default_family_config(
        family_id=SCRIPTED_PICK_PLACE_FAMILY_ID, checkpoint_dir=str(checkpoint)
    )
    capture_root = (
        tmp_path / "scenes" / "demo-scene-1" / "captures" / "demo-capture-1"
    )
    manifest = run_real_policy_family_eval(
        capture_root=capture_root,
        job_id="job-e2e",
        family_config=family,
        allow_simulator_execution=True,
        bootstrap_demo_capture_root=True,
    )
    assert manifest["status"] == "real_closed_loop_eval_completed"
    verification = manifest["verification"]
    assert verification["real_rollout_report"] is True
    assert verification["blockers"] == []
    assert verification["scorecard_trials"] > 0

    job_dir = Path(manifest["job_dir"])
    report = json.loads(
        (job_dir / "task_eval_run_report.json").read_text(encoding="utf-8")
    )
    assert report["provider_execution"]["evaluation_substrate"] == (
        "classical_sim_mujoco"
    )
    assert report["provider_execution"]["simulator_framework"] == "mujoco"
    conditions = report["scorecard"]["conditions"]
    assert conditions and conditions[0]["trials"] > 0
    # Rates always carry the trial count and interval, never a bare percentage.
    assert set(conditions[0]["success_rate"]) == {"point", "lower_95", "upper_95"}

    registry = json.loads(
        (job_dir / "real_policy_family_registry.json").read_text(encoding="utf-8")
    )
    assert registry["registered_in"] == "validation_ladder_only"
    assert registry["production_candidate_registration"]["registered"] is False


def test_ladder_registration_carries_groot_libero_boundaries(tmp_path: Path) -> None:
    family = default_groot_libero_family_config(python_executable="/usr/bin/python3")
    registry = register_family_in_validation_ladder(
        job_dir=tmp_path,
        family=family,
        checkpoint_sha256="",
        verification={"real_rollout_report": False},
        generated_at="2026-07-06T00:00:00+00:00",
    )

    assert registry["registered_in"] == "validation_ladder_only"
    assert registry["integration_proof_scope"] == "libero_panda_groot_integration_proof"
    assert registry["source_policy"]["repo_id"] == "nvidia/gr00t17-lerobot-libero_10-640"
    assert registry["gpu_runtime"]["requires_gpu_runtime"] is True
    assert registry["meaningful_manipulator_scoring"]["required_for_quality_claim"] == (
        "libero_panda_simulator_bridge_or_panda_task_evaluator"
    )
    assert registry["production_candidate_registration"]["registered"] is False
    boundary = registry["claim_boundary"]
    assert boundary["libero_panda_action_projection_to_blueprint_delta_ee"] is True
    assert boundary["panda_or_libero_task_success_proven"] is False
    assert boundary["physical_robot_readiness_proven"] is False
    assert boundary["public_claim_upgrade_allowed"] is False
