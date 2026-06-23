from __future__ import annotations

import json
import sys
from pathlib import Path

from blueprint_pipeline import unitree_lerobot_policy_runtime as runtime


def _clear_unitree_lerobot_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for name in runtime.ENV_VAR_NAMES + (
        "BLUEPRINT_UNITREE_LEROBOT_MODE",
        "BLUEPRINT_UNITREE_LEROBOT_TIMEOUT_SECONDS",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT",
        "BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_CHECKPOINT",
        "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
        "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_G1_POLICY_CHECKPOINT",
        "BLUEPRINT_UNITREE_RL_GYM_ROOT",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT",
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_SOURCE_ROOT",
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_SOURCE_ROOT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT",
        "BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT",
        "BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIM2SIM_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)


def _write_fake_lerobot_root(tmp_path: Path, script_body: str) -> Path:
    root = tmp_path / "unitree_lerobot"
    script = root / "unitree_lerobot" / "eval_robot" / "eval_g1_sim.py"
    script.parent.mkdir(parents=True)
    script.write_text(script_body, encoding="utf-8")
    return root


def _policy_dir(tmp_path: Path) -> Path:
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "config.json").write_text("{}", encoding="utf-8")
    return policy


def _write_fake_groot_sonic_roots(tmp_path: Path) -> tuple[Path, Path]:
    groot_root = tmp_path / "Isaac-GR00T"
    for relative in (
        "gr00t/eval/run_gr00t_server.py",
        "gr00t/eval/open_loop_eval.py",
        "scripts/deployment/standalone_inference_script.py",
    ):
        path = groot_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("print('ok')\n", encoding="utf-8")
    wbc_root = tmp_path / "GR00T-WholeBodyControl"
    for relative in (
        "gear_sonic/scripts/launch_inference.py",
        "gear_sonic/scripts/launch_data_collection.py",
        "download_from_hf.py",
    ):
        path = wbc_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("print('ok')\n", encoding="utf-8")
    return groot_root, wbc_root


def test_unitree_lerobot_probe_not_configured_reports_missing_requirements(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _clear_unitree_lerobot_env(monkeypatch)

    summary = runtime.run_unitree_lerobot_g1_policy_eval(
        job_dir=tmp_path / "job",
        config=runtime.UnitreeLeRobotPolicyRuntimeConfig.from_env(
            job_dir=tmp_path / "job",
            mode="probe",
        ),
        generated_at="now",
    )

    assert summary["status"] == "not_configured"
    assert summary["unitree_lerobot_sim_inference_attempted"] is False
    assert "BLUEPRINT_UNITREE_LEROBOT_ROOT" in summary["missing_requirements"]
    assert "unitree_lerobot_eval_g1_sim_script_missing" in summary["missing_requirements"]
    assert "BLUEPRINT_UNITREE_LEROBOT_POLICY_PATH" in summary["missing_requirements"]
    truth = json.loads(
        (tmp_path / "job" / "unitree_lerobot_g1_policy_runtime_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    assert truth["unitree_lerobot_runtime_configured"] is False
    assert truth["unitree_lerobot_sim_inference_proven"] is False
    assert truth["physical_robot_readiness_proven"] is False
    assert truth["deployment_readiness_proven"] is False


def test_unitree_lerobot_root_configured_but_script_missing_fails_cleanly(
    tmp_path: Path,
) -> None:
    root = tmp_path / "unitree_lerobot"
    root.mkdir()
    policy = _policy_dir(tmp_path)
    config = runtime.UnitreeLeRobotPolicyRuntimeConfig(
        root=root,
        policy_path=str(policy),
        mode="dry_run",
        task_dir=tmp_path / "job" / "task_data",
    )

    summary = runtime.run_unitree_lerobot_g1_policy_eval(
        job_dir=tmp_path / "job",
        config=config,
        generated_at="now",
    )
    command = json.loads(
        (tmp_path / "job" / "unitree_lerobot_g1_policy_handoff" / "command.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["status"] == "not_configured"
    assert command["command_built"] is False
    assert "unitree_lerobot_eval_g1_sim_script_missing" in command["missing_requirements"]
    assert summary["unitree_lerobot_sim_inference_attempted"] is False


def test_unitree_lerobot_command_builder_uses_expected_sim_eval_flags(
    tmp_path: Path,
) -> None:
    root = _write_fake_lerobot_root(tmp_path, "print('ok')\n")
    policy = _policy_dir(tmp_path)
    config = runtime.UnitreeLeRobotPolicyRuntimeConfig(
        root=root,
        policy_path=str(policy),
        dataset_repo_id="unitreerobotics/G1_Dex3_ToastedBread_Dataset",
        dataset_root="",
        policy_family="pi05",
        arm="G1_29",
        ee="dex3",
        frequency=30,
        episodes=0,
        max_episodes=1200,
        visualization=True,
        save_data=False,
        task_dir=tmp_path / "job" / "task_data",
        mode="dry_run",
    )

    command = runtime.build_unitree_lerobot_g1_sim_command(config, job_dir=tmp_path / "job")
    argv = command["command"]

    assert command["command_built"] is True
    assert argv[0] == sys.executable
    assert any(item.endswith("eval_g1_sim.py") for item in argv)
    assert f"--policy.path={policy}" in argv
    assert "--repo_id=unitreerobotics/G1_Dex3_ToastedBread_Dataset" in argv
    assert "--root=" in argv
    assert "--episodes=0" in argv
    assert "--frequency=30" in argv
    assert "--arm=G1_29" in argv
    assert "--ee=dex3" in argv
    assert "--visualization=true" in argv
    assert "--save_data=false" in argv
    assert f"--task_dir={tmp_path / 'job' / 'task_data'}" in argv
    assert "--max_episodes=1200" in argv
    assert "--send_real_robot=false" in argv


def test_unitree_lerobot_command_builder_uses_configured_python(
    tmp_path: Path,
) -> None:
    root = _write_fake_lerobot_root(tmp_path, "print('ok')\n")
    policy = _policy_dir(tmp_path)
    config = runtime.UnitreeLeRobotPolicyRuntimeConfig(
        root=root,
        python_executable="/opt/unitree/bin/python",
        policy_path=str(policy),
        mode="dry_run",
    )

    command = runtime.build_unitree_lerobot_g1_sim_command(config, job_dir=tmp_path / "job")

    assert command["command"][0] == "/opt/unitree/bin/python"
    assert command["python_executable"] == "/opt/unitree/bin/python"
    assert command["python_executable_available"] is False


def test_unitree_lerobot_smoke_uses_local_source_pythonpath(
    tmp_path: Path,
) -> None:
    root = _write_fake_lerobot_root(
        tmp_path,
        "import smoke_marker\nprint(smoke_marker.VALUE)\n",
    )
    marker_src = root / "unitree_lerobot" / "lerobot" / "src"
    marker_src.mkdir(parents=True)
    (marker_src / "smoke_marker.py").write_text("VALUE = 'from-submodule-src'\n", encoding="utf-8")
    config = runtime.UnitreeLeRobotPolicyRuntimeConfig(
        root=root,
        python_executable=sys.executable,
        mode="probe",
    )

    smoke = runtime._unitree_lerobot_eval_script_smoke_probe(config=config)

    assert smoke["passed"] is True
    assert smoke["python_executable"] == sys.executable
    assert smoke["pythonpath_local_source_enabled"] is True
    assert str(root) in smoke["pythonpath_entries"]
    assert str(marker_src) in smoke["pythonpath_entries"]


def test_unitree_lerobot_fake_sim_eval_success_writes_truth_and_handoff(
    tmp_path: Path,
) -> None:
    root = _write_fake_lerobot_root(
        tmp_path,
        "\n".join(
            [
                "import json, sys",
                "from pathlib import Path",
                "task_dir = Path(next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--task_dir=')))",
                "task_dir.mkdir(parents=True, exist_ok=True)",
                "(task_dir / 'traces').mkdir(exist_ok=True)",
                "(task_dir / 'traces' / 'trace.jsonl').write_text('{\"ok\": true}\\n')",
                "(task_dir / 'metrics.json').write_text(json.dumps({'success': True}))",
                "(task_dir / 'rendered_policy_motion').mkdir(exist_ok=True)",
                "(task_dir / 'rendered_policy_motion' / 'placeholder.mp4').write_bytes(b'not-real-video')",
                "print('fake unitree lerobot sim eval completed')",
            ]
        )
        + "\n",
    )
    policy = _policy_dir(tmp_path)
    config = runtime.UnitreeLeRobotPolicyRuntimeConfig(
        root=root,
        policy_path=str(policy),
        policy_family="pi05",
        mode="sim_eval",
        save_data=True,
        task_dir=tmp_path / "job" / "task_data",
        timeout_seconds=5,
    )

    summary = runtime.run_unitree_lerobot_g1_policy_eval(
        job_dir=tmp_path / "job",
        config=config,
        generated_at="now",
    )
    truth = json.loads(
        (tmp_path / "job" / "unitree_lerobot_g1_policy_runtime_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    handoff = json.loads(
        (
            tmp_path
            / "job"
            / "unitree_lerobot_g1_policy_handoff"
            / "robot_team_handoff_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert summary["status"] == "completed"
    assert truth["unitree_lerobot_sim_inference_attempted"] is True
    assert truth["unitree_lerobot_sim_inference_proven"] is True
    assert truth["unitree_lerobot_policy_loaded"] is True
    assert truth["vla_policy_family"] == "pi05"
    assert truth["vla_policy_used"] is True
    assert truth["unitree_hand_manipulation_policy_used"] is True
    assert truth["physical_robot_command_attempted"] is False
    assert truth["physical_robot_readiness_proven"] is False
    assert handoff["return_code"] == 0
    assert handoff["command"]
    assert handoff["videos"]
    assert handoff["traces"]
    assert handoff["metrics"]


def test_unitree_lerobot_fake_sim_eval_failure_captures_stderr(
    tmp_path: Path,
) -> None:
    root = _write_fake_lerobot_root(
        tmp_path,
        "import sys\nprint('sim failed', file=sys.stderr)\nsys.exit(3)\n",
    )
    policy = _policy_dir(tmp_path)
    config = runtime.UnitreeLeRobotPolicyRuntimeConfig(
        root=root,
        policy_path=str(policy),
        mode="sim_eval",
        task_dir=tmp_path / "job" / "task_data",
        timeout_seconds=5,
    )

    summary = runtime.run_unitree_lerobot_g1_policy_eval(
        job_dir=tmp_path / "job",
        config=config,
        generated_at="now",
    )
    truth = json.loads(
        (tmp_path / "job" / "unitree_lerobot_g1_policy_runtime_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    stderr = (tmp_path / "job" / "unitree_lerobot_g1_policy_handoff" / "stderr.log").read_text(
        encoding="utf-8"
    )

    assert summary["status"] == "failed"
    assert truth["unitree_lerobot_sim_inference_attempted"] is True
    assert truth["unitree_lerobot_sim_inference_proven"] is False
    assert truth["runtime_error_summary"] == "unitree_lerobot_sim_eval_exited_3"
    assert "sim failed" in stderr


def test_unitree_lerobot_real_robot_safety_block_prevents_subprocess(
    tmp_path: Path,
) -> None:
    root = _write_fake_lerobot_root(tmp_path, "raise SystemExit('should not run')\n")
    policy = _policy_dir(tmp_path)
    config = runtime.UnitreeLeRobotPolicyRuntimeConfig(
        root=root,
        policy_path=str(policy),
        mode="sim_eval",
        task_dir=tmp_path / "job" / "task_data",
        send_real_robot=True,
        allow_real_robot_commands=False,
    )

    summary = runtime.run_unitree_lerobot_g1_policy_eval(
        job_dir=tmp_path / "job",
        config=config,
        generated_at="now",
    )
    truth = json.loads(
        (tmp_path / "job" / "unitree_lerobot_g1_policy_runtime_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["status"] == "blocked"
    assert summary["unitree_lerobot_sim_inference_attempted"] is False
    assert truth["physical_robot_command_attempted"] is False
    assert truth["unitree_lerobot_sim_inference_proven"] is False
    assert (
        "blocked_real_robot_command_requires_BLUEPRINT_UNITREE_ALLOW_REAL_ROBOT_COMMANDS"
        in truth["safety_blocks"]
    )


def test_policy_family_registry_keeps_openvla_and_unifolm_truth_boundaries(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _clear_unitree_lerobot_env(monkeypatch)
    monkeypatch.setenv("BLUEPRINT_UNITREE_OPENVLA_ENDPOINT_URL", "http://127.0.0.1:9999")
    config = runtime.UnitreeLeRobotPolicyRuntimeConfig.from_env(
        job_dir=tmp_path / "job",
        mode="probe",
    )

    registry = runtime.build_policy_provider_registry_probe(
        job_dir=tmp_path / "job",
        generated_at="now",
        config=config,
    )

    openvla = next(row for row in registry["providers"] if row["lane_name"] == "openvla_endpoint")
    unifolm_wma = next(row for row in registry["providers"] if row["lane_name"] == "unifolm_wma")
    assert openvla["status"] == "configured"
    assert openvla["g1_action_adapter_configured"] is False
    assert openvla["openvla_endpoint_used"] is False
    assert unifolm_wma["wam_world_model_used"] is False
    assert registry["openvla_selected_for_g1_policy"] is False
    assert registry["wam_selected_for_g1_policy"] is False
    assert registry["unitree_hand_manipulation_policy_in_place"] is False
    assert registry["g1_robot_policy_family_decision"]["openvla_is_comparison_only_for_g1"] is True
    assert registry["g1_robot_policy_family_decision"]["wam_is_evaluator_not_robot_policy"] is True


def test_unitree_stack_audit_does_not_call_rl_gym_only_whole_stack_installed(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _clear_unitree_lerobot_env(monkeypatch)
    rl_gym_root = tmp_path / "unitree_rl_gym"
    checkpoint = rl_gym_root / "deploy" / "pre_train" / "g1" / "motion.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("weights", encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_UNITREE_RL_GYM_ROOT", str(rl_gym_root))

    registry = runtime.build_policy_provider_registry_probe(
        job_dir=tmp_path / "job",
        generated_at="now",
        config=runtime.UnitreeLeRobotPolicyRuntimeConfig.from_env(
            job_dir=tmp_path / "job",
            mode="probe",
        ),
    )
    audit = registry["installation_audit"]

    assert registry["selected_provider"] == "official_unitree_rl_gym"
    assert registry["selected_provider_legacy_first_configured"] == "official_unitree_rl_gym"
    assert registry["selected_locomotion_provider"] == "official_unitree_rl_gym"
    assert registry["selected_unitree_manipulation_runtime"] is None
    assert registry["selected_unitree_action_command"] is None
    assert registry["selected_unitree_hand_policy"] is None
    assert registry["unitree_hand_manipulation_policy_in_place"] is False
    assert registry["unitree_hand_manipulation_policy_used"] is False
    assert registry["openvla_selected_for_g1_policy"] is False
    assert registry["wam_selected_for_g1_policy"] is False
    assert registry["claim_boundary"]["selected_provider_legacy_field_may_be_locomotion_only"] is True
    assert registry["claim_boundary"]["unitree_hand_policy_requires_manipulation_runtime_and_action_command"] is True
    assert registry["whole_unitree_policy_stack_installed"] is False
    assert registry["installation_status"] == "not_installed"
    assert audit["component_checks"]["official_rl_gym_locomotion"]["configured"] is True
    assert audit["component_checks"]["unitree_manipulation_runtime"]["configured"] is False
    assert audit["component_checks"]["unitree_action_command"]["configured"] is False
    assert "unitree_manipulation_runtime_not_configured" in audit["blockers"]
    assert "unitree_specific_action_command_not_configured" in audit["blockers"]


def test_unitree_stack_audit_reports_lerobot_source_runtime_as_partial(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _clear_unitree_lerobot_env(monkeypatch)
    rl_gym_root = tmp_path / "unitree_rl_gym"
    checkpoint = rl_gym_root / "deploy" / "pre_train" / "g1" / "motion.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("weights", encoding="utf-8")
    lerobot_root = _write_fake_lerobot_root(tmp_path, "print('ok')\n")
    monkeypatch.setenv("BLUEPRINT_UNITREE_RL_GYM_ROOT", str(rl_gym_root))
    monkeypatch.setenv("BLUEPRINT_UNITREE_LEROBOT_ROOT", str(lerobot_root))

    config = runtime.UnitreeLeRobotPolicyRuntimeConfig.from_env(
        job_dir=tmp_path / "job",
        mode="probe",
    )
    audit = runtime.build_unitree_policy_stack_installation_audit(
        job_dir=tmp_path / "job",
        generated_at="now",
        config=config,
    )
    manipulation = audit["component_checks"]["unitree_manipulation_runtime"]
    lerobot = next(
        row for row in manipulation["candidates"] if row["candidate_id"] == "unitree_lerobot_g1"
    )
    action_lerobot = next(
        row
        for row in audit["component_checks"]["unitree_action_command"]["candidates"]
        if row["candidate_id"] == "unitree_lerobot_policy"
    )

    assert audit["whole_unitree_policy_stack_installed"] is False
    assert audit["status"] == "not_installed"
    assert audit["partial_component_ids"] == ["unitree_manipulation_runtime"]
    assert manipulation["partial_candidate_ids"] == ["unitree_lerobot_g1"]
    assert lerobot["configuration_stage"] == "source_runtime_ready_policy_missing"
    assert lerobot["source_runtime_configured"] is True
    assert lerobot["runtime_configured"] is True
    assert lerobot["policy_path_configured"] is False
    assert lerobot["policy_or_endpoint_configured"] is False
    assert lerobot["source_runtime_ready_without_policy"] is True
    assert lerobot["partial_configuration"] is True
    assert "blocked_unitree_lerobot_policy_or_endpoint_not_configured" in lerobot["blockers"]
    assert action_lerobot["source_root_exists"] is True
    assert action_lerobot["command_configured"] is False
    assert action_lerobot["checkpoint_configured"] is False
    assert "unitree_manipulation_runtime_not_configured" in audit["blockers"]
    assert "unitree_specific_action_command_not_configured" in audit["blockers"]


def test_unitree_stack_audit_blocks_lerobot_source_runtime_when_smoke_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _clear_unitree_lerobot_env(monkeypatch)
    rl_gym_root = tmp_path / "unitree_rl_gym"
    checkpoint = rl_gym_root / "deploy" / "pre_train" / "g1" / "motion.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("weights", encoding="utf-8")
    lerobot_root = _write_fake_lerobot_root(
        tmp_path,
        "import sys\nprint('missing dependency', file=sys.stderr)\nsys.exit(7)\n",
    )
    monkeypatch.setenv("BLUEPRINT_UNITREE_RL_GYM_ROOT", str(rl_gym_root))
    monkeypatch.setenv("BLUEPRINT_UNITREE_LEROBOT_ROOT", str(lerobot_root))

    config = runtime.UnitreeLeRobotPolicyRuntimeConfig.from_env(
        job_dir=tmp_path / "job",
        mode="probe",
    )
    audit = runtime.build_unitree_policy_stack_installation_audit(
        job_dir=tmp_path / "job",
        generated_at="now",
        config=config,
    )
    manipulation = audit["component_checks"]["unitree_manipulation_runtime"]
    lerobot = next(
        row for row in manipulation["candidates"] if row["candidate_id"] == "unitree_lerobot_g1"
    )

    assert audit["whole_unitree_policy_stack_installed"] is False
    assert lerobot["configuration_stage"] == "source_runtime_files_ready_dependency_smoke_failed"
    assert lerobot["source_runtime_files_configured"] is True
    assert lerobot["source_runtime_execution_ready"] is False
    assert lerobot["source_runtime_dependency_smoke_passed"] is False
    assert lerobot["source_runtime_dependency_smoke"]["return_code"] == 7
    assert "missing dependency" in lerobot["source_runtime_dependency_smoke"]["stderr_tail"]
    assert "blocked_unitree_lerobot_eval_script_smoke_failed" in lerobot["blockers"]
    assert "blocked_unitree_lerobot_policy_or_endpoint_not_configured" in lerobot["blockers"]

    summary = runtime.run_unitree_lerobot_g1_policy_eval(
        job_dir=tmp_path / "job_run",
        config=config,
        generated_at="now",
    )
    assert summary["unitree_lerobot_configuration_stage"] == (
        "source_runtime_files_ready_dependency_smoke_failed"
    )
    assert summary["unitree_lerobot_source_runtime_files_configured"] is True
    assert summary["unitree_lerobot_source_runtime_execution_ready"] is False
    assert summary["unitree_lerobot_source_runtime_dependency_smoke_passed"] is False
    assert "blocked_unitree_lerobot_eval_script_smoke_failed" in summary[
        "unitree_lerobot_source_runtime_blockers"
    ]


def test_unitree_stack_audit_names_groot_n17_sonic_candidate_blockers(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _clear_unitree_lerobot_env(monkeypatch)
    rl_gym_root = tmp_path / "unitree_rl_gym"
    checkpoint = rl_gym_root / "deploy" / "pre_train" / "g1" / "motion.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("weights", encoding="utf-8")
    command = tmp_path / "groot_policy_command.py"
    command.write_text("print('{}')\n", encoding="utf-8")
    groot_root = tmp_path / "Isaac-GR00T"
    groot_root.mkdir()
    monkeypatch.setenv("BLUEPRINT_UNITREE_RL_GYM_ROOT", str(rl_gym_root))
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
        f"{sys.executable} {command}",
    )
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT", "nvidia/GR00T-N1.7-3B")
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT", str(groot_root))

    config = runtime.UnitreeLeRobotPolicyRuntimeConfig.from_env(
        job_dir=tmp_path / "job",
        mode="probe",
    )
    audit = runtime.build_unitree_policy_stack_installation_audit(
        job_dir=tmp_path / "job",
        generated_at="now",
        config=config,
    )
    manipulation = audit["component_checks"]["unitree_manipulation_runtime"]
    groot = next(
        row
        for row in manipulation["candidates"]
        if row["candidate_id"] == "unitree_groot_n17_sonic_policy"
    )
    action_groot = next(
        row
        for row in audit["component_checks"]["unitree_action_command"]["candidates"]
        if row["candidate_id"] == "unitree_groot_n17_sonic_policy"
    )

    assert audit["whole_unitree_policy_stack_installed"] is False
    assert groot["configured"] is False
    assert groot["partial_configuration"] is True
    assert groot["groot_checkpoint_original_reference"] == "nvidia/GR00T-N1.7-3B"
    assert groot["groot_checkpoint_path"] == "LucaFrat/groot-bs16"
    assert groot["groot_checkpoint_effective_reference"] == "LucaFrat/groot-bs16"
    assert groot["groot_default_experimental_checkpoint_applied"] is True
    assert "blocked_missing_BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT" in groot["blockers"]
    assert "blocked_missing_BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT" in groot["blockers"]
    assert action_groot["command_configured"] is True
    assert action_groot["checkpoint_env"] == "BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT"
    assert action_groot["checkpoint_configured"] is False
    assert action_groot["extra_required_checkpoints"][0]["checkpoint_env"] == (
        "BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT"
    )
    assert action_groot["extra_required_checkpoints"][0]["checkpoint_configured"] is True
    assert "unitree_manipulation_runtime_not_configured" in audit["blockers"]


def test_unitree_stack_audit_accepts_configured_groot_runtime_without_action_command(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _clear_unitree_lerobot_env(monkeypatch)
    rl_gym_root = tmp_path / "unitree_rl_gym"
    checkpoint = rl_gym_root / "deploy" / "pre_train" / "g1" / "motion.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("weights", encoding="utf-8")
    groot_root, wbc_root = _write_fake_groot_sonic_roots(tmp_path)
    n17_checkpoint = tmp_path / "finetuned_n17_unitree_g1_sonic"
    n17_checkpoint.mkdir()
    sonic_checkpoint = tmp_path / "gear_sonic_deploy"
    sonic_checkpoint.mkdir()
    monkeypatch.setenv("BLUEPRINT_UNITREE_RL_GYM_ROOT", str(rl_gym_root))
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT", str(groot_root))
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT", str(wbc_root))
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT", str(n17_checkpoint))
    monkeypatch.setenv("BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT", str(sonic_checkpoint))

    config = runtime.UnitreeLeRobotPolicyRuntimeConfig.from_env(
        job_dir=tmp_path / "job",
        mode="probe",
    )
    audit = runtime.build_unitree_policy_stack_installation_audit(
        job_dir=tmp_path / "job",
        generated_at="now",
        config=config,
    )
    manipulation = audit["component_checks"]["unitree_manipulation_runtime"]
    groot = next(
        row
        for row in manipulation["candidates"]
        if row["candidate_id"] == "unitree_groot_n17_sonic_policy"
    )

    assert manipulation["configured"] is False
    assert manipulation["selected_candidate_id"] is None
    assert "unitree_manipulation_runtime_not_configured" in manipulation["blockers"]
    assert "unitree_manipulation_runtime" in audit["partial_component_ids"]
    assert groot["configured"] is True
    assert groot["partial_configuration"] is True
    assert groot["ready_for_policy_action_command"] is False
    assert groot["action_command_blockers"] == [
        "blocked_missing_BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"
    ]
    assert audit["whole_unitree_policy_stack_installed"] is False
    assert "unitree_manipulation_runtime_not_configured" in audit["blockers"]
    assert "unitree_specific_action_command_not_configured" in audit["blockers"]


def test_unitree_stack_audit_requires_manipulation_runtime_and_action_command(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _clear_unitree_lerobot_env(monkeypatch)
    rl_gym_root = tmp_path / "unitree_rl_gym"
    checkpoint = rl_gym_root / "deploy" / "pre_train" / "g1" / "motion.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("weights", encoding="utf-8")
    lerobot_root = _write_fake_lerobot_root(tmp_path, "print('ok')\n")
    policy = _policy_dir(tmp_path)
    command = tmp_path / "unitree_lerobot_action_command.py"
    command.write_text("print('{}')\n", encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_UNITREE_RL_GYM_ROOT", str(rl_gym_root))
    monkeypatch.setenv("BLUEPRINT_UNITREE_LEROBOT_ROOT", str(lerobot_root))
    monkeypatch.setenv("BLUEPRINT_UNITREE_LEROBOT_POLICY_PATH", str(policy))
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        f"{sys.executable} {command}",
    )
    monkeypatch.setenv("BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT", str(policy))

    config = runtime.UnitreeLeRobotPolicyRuntimeConfig.from_env(
        job_dir=tmp_path / "job",
        mode="probe",
    )
    audit = runtime.build_unitree_policy_stack_installation_audit(
        job_dir=tmp_path / "job",
        generated_at="now",
        config=config,
    )

    assert audit["status"] == "installed"
    assert audit["whole_unitree_policy_stack_installed"] is True
    assert audit["blockers"] == []
    assert audit["component_checks"]["official_rl_gym_locomotion"]["configured"] is True
    assert audit["component_checks"]["unitree_manipulation_runtime"]["configured"] is True
    assert audit["component_checks"]["unitree_action_command"]["configured"] is True
    assert (
        audit["component_checks"]["unitree_action_command"]["selected_candidate_id"]
        == "unitree_lerobot_policy"
    )
