from __future__ import annotations

import json
import sys
from pathlib import Path

from blueprint_pipeline import unitree_groot_n17_sonic_policy_runtime as runtime


def _clear_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for name in runtime.ENV_VAR_NAMES:
        monkeypatch.delenv(name, raising=False)


def _fake_groot_root(tmp_path: Path) -> Path:
    root = tmp_path / "Isaac-GR00T"
    for relative in runtime.EXPECTED_GROOT_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fake\n", encoding="utf-8")
    return root


def _fake_wbc_root(tmp_path: Path) -> Path:
    root = tmp_path / "GR00T-WholeBodyControl"
    for relative in runtime.EXPECTED_WBC_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fake\n", encoding="utf-8")
    return root


def _fake_official_sonic_wbc_root(tmp_path: Path) -> Path:
    root = tmp_path / "GR00T-WholeBodyControl-official"
    for relative in runtime.EXPECTED_WBC_FILES + runtime.EXPECTED_OFFICIAL_SONIC_SIM_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fake\n", encoding="utf-8")
    (root / runtime.OFFICIAL_SONIC_DEFAULT_MOTION_DATA_RELATIVE_PATH).mkdir(
        parents=True,
        exist_ok=True,
    )
    return root


def _fake_sonic_checkpoint(tmp_path: Path) -> Path:
    root = tmp_path / "sonic_checkpoint"
    for relative in runtime.OFFICIAL_SONIC_DEPLOY_ASSET_RELATIVE_PATHS.values():
        if relative == "policy/release/model":
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake\n", encoding="utf-8")
    return root


def _fake_venv(root: Path, name: str) -> None:
    bin_dir = root / name / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "python").write_text("# fake python\n", encoding="utf-8")
    (bin_dir / "activate").write_text("# fake activate\n", encoding="utf-8")


def test_groot_n17_sonic_runtime_blocks_without_local_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _clear_env(monkeypatch)

    summary = runtime.run_unitree_groot_n17_sonic_policy_runtime(
        job_dir=tmp_path / "job",
        generated_at="now",
    )
    audit = json.loads(
        (tmp_path / "job" / "unitree_groot_n17_sonic_installation_audit.json").read_text(
            encoding="utf-8"
        )
    )
    truth = json.loads(
        (
            tmp_path
            / "job"
            / "unitree_groot_n17_sonic_policy_runtime_truth_boundary.json"
        ).read_text(encoding="utf-8")
    )

    assert summary["status"] == "not_configured"
    assert summary["unitree_groot_n17_sonic_policy_configured"] is False
    assert summary["unitree_groot_n17_sonic_policy_action_command_ran"] is False
    assert f"blocked_missing_{runtime.GROOT_ROOT_ENV}" in audit["blockers"]
    assert f"blocked_missing_{runtime.WBC_ROOT_ENV}" in audit["blockers"]
    assert f"blocked_missing_{runtime.N17_CHECKPOINT_ENV}" not in audit["blockers"]
    assert audit["default_experimental_checkpoint_applied"] is True
    assert (
        audit["n17_checkpoint_effective_reference"]
        == runtime.DEFAULT_EXPERIMENTAL_UNITREE_G1_SONIC_POLICY_CHECKPOINT
    )
    assert f"blocked_missing_{runtime.SONIC_CHECKPOINT_ENV}" in audit["blockers"]
    assert Path(summary["official_launcher_preflight_path"]).is_file()
    assert truth["physical_robot_readiness_proven"] is False
    assert truth["deployment_readiness_proven"] is False
    assert truth["safety_validation_proven"] is False


def test_groot_n17_sonic_runtime_configured_with_roots_and_checkpoints(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _clear_env(monkeypatch)
    command = tmp_path / "policy_command.py"
    command.write_text("print('{}')\n", encoding="utf-8")
    n17_checkpoint = tmp_path / "n17"
    n17_checkpoint.mkdir()
    sonic_checkpoint = tmp_path / "sonic"
    sonic_checkpoint.mkdir()
    hf_token_file = tmp_path / "hf_token"
    hf_token_file.write_text("fake-token", encoding="utf-8")
    monkeypatch.setenv(runtime.GROOT_ROOT_ENV, str(_fake_groot_root(tmp_path)))
    monkeypatch.setenv(runtime.WBC_ROOT_ENV, str(_fake_wbc_root(tmp_path)))
    monkeypatch.setenv(runtime.N17_CHECKPOINT_ENV, str(n17_checkpoint))
    monkeypatch.setenv(runtime.SONIC_CHECKPOINT_ENV, str(sonic_checkpoint))
    monkeypatch.setenv(runtime.POLICY_COMMAND_ENV, f"{sys.executable} {command}")
    monkeypatch.setenv(runtime.HF_TOKEN_FILE_ENV, str(hf_token_file))

    audit = runtime.probe_unitree_groot_n17_sonic_runtime(generated_at="now")

    assert audit["status"] == "configured"
    assert audit["unitree_groot_n17_sonic_policy_configured"] is True
    assert audit["ready_for_policy_action_command"] is True
    assert audit["n17_checkpoint_exists"] is True
    assert audit["g1_sonic_checkpoint_exists"] is True
    assert audit["hf_token_file_configured"] is True
    assert audit["hf_token_file_exists"] is True
    assert audit["hf_token_value_written_to_artifacts"] is False
    assert audit["raw_credentials_written_to_artifacts"] is False


def test_groot_n17_sonic_runtime_uses_experimental_default_when_env_points_to_base(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _clear_env(monkeypatch)
    command = tmp_path / "policy_command.py"
    command.write_text("print('{}')\n", encoding="utf-8")
    sonic_checkpoint = tmp_path / "sonic"
    sonic_checkpoint.mkdir()
    monkeypatch.setenv(runtime.GROOT_ROOT_ENV, str(_fake_groot_root(tmp_path)))
    monkeypatch.setenv(runtime.WBC_ROOT_ENV, str(_fake_wbc_root(tmp_path)))
    monkeypatch.setenv(runtime.N17_CHECKPOINT_ENV, "nvidia/GR00T-N1.7-3B")
    monkeypatch.setenv(runtime.SONIC_CHECKPOINT_ENV, str(sonic_checkpoint))
    monkeypatch.setenv(runtime.POLICY_COMMAND_ENV, f"{sys.executable} {command}")
    monkeypatch.setenv(runtime.POLICY_SERVER_URL_ENV, "tcp://127.0.0.1:5550")

    audit = runtime.probe_unitree_groot_n17_sonic_runtime(generated_at="now")

    assert audit["status"] == "configured"
    assert audit["unitree_groot_n17_sonic_policy_configured"] is True
    assert audit["n17_checkpoint_known_base_model_without_unitree_g1_sonic_support"] is True
    assert audit["default_experimental_checkpoint_applied"] is True
    assert (
        audit["n17_checkpoint_effective_reference"]
        == runtime.DEFAULT_EXPERIMENTAL_UNITREE_G1_SONIC_POLICY_CHECKPOINT
    )
    assert audit["task_specific_finetuning_required_for_admission"] is False
    assert audit["unitree_g1_sonic_policy_checkpoint_provenance"][
        "trusted_for_production"
    ] is False
    assert runtime.BASE_N17_WITHOUT_SONIC_SUPPORT_BLOCKER not in audit["blockers"]
    assert runtime.DEFAULT_EXPERIMENTAL_UNITREE_G1_SONIC_POLICY_CHECKPOINT in audit[
        "retry_commands_once_access_is_supplied"
    ]["start_groot_sonic_policy_server"]
    assert "$BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT" not in audit[
        "retry_commands_once_access_is_supplied"
    ]["start_groot_sonic_policy_server"]


def test_groot_n17_sonic_policy_server_command_requires_server_url_for_readiness(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _clear_env(monkeypatch)
    command = tmp_path / "blueprint-unitree-groot-n17-sonic-policy-server-command"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    n17_checkpoint = tmp_path / "n17"
    n17_checkpoint.mkdir()
    sonic_checkpoint = tmp_path / "sonic"
    sonic_checkpoint.mkdir()
    monkeypatch.setenv(runtime.GROOT_ROOT_ENV, str(_fake_groot_root(tmp_path)))
    monkeypatch.setenv(runtime.WBC_ROOT_ENV, str(_fake_wbc_root(tmp_path)))
    monkeypatch.setenv(runtime.N17_CHECKPOINT_ENV, str(n17_checkpoint))
    monkeypatch.setenv(runtime.SONIC_CHECKPOINT_ENV, str(sonic_checkpoint))
    monkeypatch.setenv(runtime.POLICY_COMMAND_ENV, str(command))

    audit = runtime.probe_unitree_groot_n17_sonic_runtime(generated_at="now")

    assert audit["status"] == "configured"
    assert audit["policy_server_command_selected"] is True
    assert audit["ready_for_policy_action_command"] is False
    assert audit["policy_command_readiness_blockers"] == [
        f"blocked_missing_{runtime.POLICY_SERVER_URL_ENV}"
    ]
    assert f"blocked_missing_{runtime.POLICY_SERVER_URL_ENV}" in audit["blockers"]
    assert f"blocked_missing_{runtime.SIM2SIM_COMMAND_ENV}" in audit["blockers"]


def test_groot_n17_sonic_runtime_preserves_existing_action_and_sim2sim_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _clear_env(monkeypatch)
    job = tmp_path / "job"
    job.mkdir()
    (job / "policy_action_model_command_output.json").write_text(
        json.dumps(
            {
                "unitree_groot_n17_sonic_policy_action_command_ran": True,
                "unitree_policy_action_command_ran": True,
                "unitree_specific_manipulation_candidate_ran": True,
            }
        ),
        encoding="utf-8",
    )
    (job / "unitree_groot_n17_sonic_sim2sim_execution.json").write_text(
        json.dumps(
            {
                "unitree_groot_n17_sonic_sim2sim_command_ran": True,
                "unitree_groot_n17_sonic_action_chunk_consumed": True,
            }
        ),
        encoding="utf-8",
    )

    summary = runtime.run_unitree_groot_n17_sonic_policy_runtime(
        job_dir=job,
        generated_at="now",
    )
    truth = json.loads(
        (job / "unitree_groot_n17_sonic_policy_runtime_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["status"] == "action_command_evidence_present"
    assert summary["selected_candidate_id"] == runtime.POLICY_ID
    assert summary["unitree_groot_n17_sonic_policy_action_command_ran"] is True
    assert summary["unitree_groot_n17_sonic_sim2sim_command_ran"] is True
    assert summary["unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim"] is True
    assert truth["unitree_groot_n17_sonic_policy_action_command_ran"] is True
    assert truth["unitree_groot_n17_sonic_sim2sim_command_ran"] is True
    assert truth["unitree_groot_n17_sonic_action_chunk_consumed_by_sim2sim"] is True


def test_official_sonic_sim2sim_probe_blocks_missing_inference_venv(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(runtime.shutil, "which", lambda command: "/usr/bin/tmux")
    root = _fake_official_sonic_wbc_root(tmp_path)
    _fake_venv(root, ".venv_sim")

    audit = runtime.probe_unitree_groot_n17_sonic_official_sim2sim_runtime(
        generated_at="now",
        wbc_root=root,
    )

    assert audit["status"] == "not_configured"
    assert audit["official_groot_wholebodycontrol_sim2sim_configured"] is False
    assert "blocked_missing_groot_wholebodycontrol_venv_inference" in audit["blockers"]
    assert "--sim --no-data-exporter" in audit["safe_sim_only_launcher_command"]
    assert audit["official_groot_wholebodycontrol_sim2sim_used"] is False
    assert audit["official_sonic_wbc_mapping_proven"] is False
    assert audit["physical_robot_launcher_intentionally_not_run"] is True
    assert audit["claim_boundary"]["physical_robot_readiness_proven"] is False


def test_official_sonic_sim2sim_probe_configures_with_required_venvs(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(runtime.shutil, "which", lambda command: "/usr/bin/tmux")
    monkeypatch.setattr(
        runtime,
        "_venv_import_status",
        lambda _python, modules: [
            {"module": module, "importable": True}
            for module in modules
        ],
    )
    root = _fake_official_sonic_wbc_root(tmp_path)
    _fake_venv(root, ".venv_sim")
    _fake_venv(root, ".venv_inference")

    audit = runtime.probe_unitree_groot_n17_sonic_official_sim2sim_runtime(
        generated_at="now",
        wbc_root=root,
    )

    assert audit["status"] == "configured"
    assert audit["official_groot_wholebodycontrol_sim2sim_configured"] is True
    assert audit["blockers"] == []
    assert audit["current_blueprint_direct_action_bridge_is_official_sonic_wbc"] is False


def test_official_sonic_sim2sim_probe_blocks_failed_inference_venv_imports(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(runtime.shutil, "which", lambda command: "/usr/bin/tmux")
    monkeypatch.setattr(
        runtime,
        "_venv_import_status",
        lambda _python, modules: [
            {"module": module, "importable": module != "gr00t"}
            for module in modules
        ],
    )
    root = _fake_official_sonic_wbc_root(tmp_path)
    _fake_venv(root, ".venv_sim")
    _fake_venv(root, ".venv_inference")

    audit = runtime.probe_unitree_groot_n17_sonic_official_sim2sim_runtime(
        generated_at="now",
        wbc_root=root,
    )

    assert audit["status"] == "not_configured"
    assert ".venv_inference:gr00t" in audit["missing_required_imports"]
    assert (
        "blocked_groot_wholebodycontrol_venv_inference_missing_import_gr00t"
        in audit["blockers"]
    )


def test_official_sonic_launcher_preflight_records_assets_and_runtime_blockers(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _clear_env(monkeypatch)
    monkeypatch.setattr(runtime.sys, "platform", "darwin")
    monkeypatch.setattr(
        runtime,
        "_toolchain_status",
        lambda _commands: [
            {"command": "cmake", "available": True, "path": "/usr/bin/cmake"},
            {"command": "clang", "available": True, "path": "/usr/bin/clang"},
            {"command": "clang++", "available": True, "path": "/usr/bin/clang++"},
            {"command": "just", "available": True, "path": "/usr/bin/just"},
            {"command": "git-lfs", "available": True, "path": "/usr/bin/git-lfs"},
            {"command": "pkg-config", "available": False, "path": None},
            {"command": "nvcc", "available": False, "path": None},
            {"command": "ros2", "available": False, "path": None},
        ],
    )
    monkeypatch.setattr(
        runtime,
        "_venv_import_status",
        lambda _python, modules: [{"module": module, "importable": True} for module in modules],
    )
    root = _fake_official_sonic_wbc_root(tmp_path)
    _fake_venv(root, ".venv_sim")
    _fake_venv(root, ".venv_inference")
    checkpoint = _fake_sonic_checkpoint(tmp_path)
    monkeypatch.setenv(runtime.N17_CHECKPOINT_ENV, runtime.KNOWN_BASE_N17_MODEL_REPO)

    preflight = runtime.probe_unitree_groot_n17_sonic_official_launcher_preflight(
        generated_at="now",
        wbc_root=root,
        sonic_checkpoint=checkpoint,
        run_help_commands=False,
    )

    assert preflight["status"] == "preflight_complete"
    assert preflight["official_sonic_wbc_launcher_preflight_completed"] is True
    assert preflight["official_sonic_wbc_launcher_executed"] is False
    assert preflight["official_sonic_wbc_deploy_loop_executed"] is False
    assert preflight["sonic_checkpoint_local_path_exists"] is True
    assert preflight["missing_official_sonic_deploy_assets"] == []
    assert preflight["official_launcher_help_probes"]["skipped"] is True
    assert preflight["default_experimental_checkpoint_applied"] is True
    assert (
        preflight["n17_checkpoint_effective_reference"]
        == runtime.DEFAULT_EXPERIMENTAL_UNITREE_G1_SONIC_POLICY_CHECKPOINT
    )
    assert preflight["task_specific_finetuning_required_for_admission"] is False
    assert preflight["unitree_g1_sonic_policy_checkpoint_provenance"][
        "trusted_for_production"
    ] is False
    assert (
        "blocked_local_official_sonic_deploy_runtime_requires_linux_gpu_or_jetson_not_macos"
        in preflight["blockers"]
    )
    assert "blocked_local_official_sonic_deploy_missing_nvcc" in preflight["blockers"]
    assert (
        "--deploy-checkpoint"
        in preflight["exact_retry_commands_once_checkpoint_and_linux_gpu_runtime_are_available"][
            "run_official_sonic_wbc_launcher_sim_only"
        ]
    )
    assert preflight["claim_boundary"]["physical_robot_readiness_proven"] is False
    assert preflight["raw_credentials_written_to_artifacts"] is False
