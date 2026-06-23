from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from blueprint_pipeline import unitree_groot_n17_sonic_policy_runtime as runtime
from blueprint_pipeline import unitree_groot_n17_sonic_policy_server_preflight as preflight


def _clear_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for name in runtime.ENV_VAR_NAMES:
        monkeypatch.delenv(name, raising=False)


def _fake_root(tmp_path: Path, name: str, expected_files: tuple[str, ...]) -> Path:
    root = tmp_path / name
    for relative in expected_files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fake\n", encoding="utf-8")
    return root


def _fake_groot_root(tmp_path: Path) -> Path:
    root = _fake_root(tmp_path, "Isaac-GR00T", runtime.EXPECTED_GROOT_FILES)
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    return root


def _fake_wbc_root(tmp_path: Path) -> Path:
    return _fake_root(tmp_path, "GR00T-WholeBodyControl", runtime.EXPECTED_WBC_FILES)


def _fake_runtime_setup_audit(tmp_path: Path, *, n17_size_gib: float = 6.452) -> Path:
    path = tmp_path / "unitree_groot_n17_sonic_runtime_setup_audit.json"
    path.write_text(
        json.dumps(
            {
                "huggingface_models": {
                    "nvidia/GR00T-N1.7-3B": {
                        "file_count": 27,
                        "gated": False,
                        "private": False,
                        "sha": "2fc962b973bccdd5d8ce4f67cc63b264d6886495",
                        "total_size_gib": n17_size_gib,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _fake_runner(*, cuda_available: bool = True):  # type: ignore[no-untyped-def]
    def run(probe: preflight.CommandProbe) -> dict[str, object]:
        args = tuple(probe.args)
        if args[:4] == ("git", "-C", str(probe.cwd), "rev-parse"):
            return {
                "ran": True,
                "timed_out": False,
                "returncode": 0,
                "stdout_tail": "abc123\n",
                "stderr_tail": "",
            }
        if len(args) >= 2 and args[1] == "--version":
            return {
                "ran": True,
                "timed_out": False,
                "returncode": 0,
                "stdout_tail": "Python 3.10.19\n",
                "stderr_tail": "",
            }
        if "-c" in args:
            script = str(args[args.index("-c") + 1])
            if "torch_cuda_available" in script:
                modules = {
                    name: {"imported": True, "version": "test"}
                    for name in [
                        "torch",
                        "transformers",
                        "huggingface_hub",
                        "gr00t",
                        "zmq",
                        "msgpack",
                        "msgpack_numpy",
                        "tyro",
                    ]
                }
                return {
                    "ran": True,
                    "timed_out": False,
                    "returncode": 0,
                    "stdout_tail": json.dumps(
                        {
                            "modules": modules,
                            "torch_cuda_available": cuda_available,
                        }
                    ),
                    "stderr_tail": "",
                }
            if "EmbodimentTag.resolve" in script:
                return {
                    "ran": True,
                    "timed_out": False,
                    "returncode": 0,
                    "stdout_tail": json.dumps(
                        {
                            "tag_name": "UNITREE_G1_SONIC",
                            "tag_value": "unitree_g1_sonic",
                            "is_pretrain_tag": False,
                            "is_posttrain_tag": True,
                            "modality_config_present": True,
                            "video_keys": ["ego_view"],
                            "state_keys": [
                                "left_leg",
                                "right_leg",
                                "waist",
                                "left_arm",
                                "right_arm",
                                "left_hand",
                                "right_hand",
                                "projected_gravity",
                            ],
                            "action_keys": [
                                "motion_token",
                                "left_hand_joints",
                                "right_hand_joints",
                            ],
                            "action_delta_indices": list(range(40)),
                        }
                    ),
                    "stderr_tail": "",
                }
        if args[-1] == "--help":
            return {
                "ran": True,
                "timed_out": False,
                "returncode": 0,
                "stdout_tail": (
                    "--model-path --dataset-path --embodiment-tag "
                    "--device --port --no-strict"
                ),
                "stderr_tail": "",
            }
        return {
            "ran": True,
            "timed_out": False,
            "returncode": 1,
            "stdout_tail": "",
            "stderr_tail": "unexpected command",
        }

    return run


def test_policy_server_preflight_blocks_repo_checkpoint_when_disk_is_short(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _clear_env(monkeypatch)
    sonic_checkpoint = tmp_path / "gear_sonic_deploy"
    sonic_checkpoint.mkdir()
    monkeypatch.setenv(runtime.GROOT_ROOT_ENV, str(_fake_groot_root(tmp_path)))
    monkeypatch.setenv(runtime.WBC_ROOT_ENV, str(_fake_wbc_root(tmp_path)))
    monkeypatch.setenv(runtime.N17_CHECKPOINT_ENV, "nvidia/GR00T-N1.7-3B")
    monkeypatch.setenv(runtime.SONIC_CHECKPOINT_ENV, str(sonic_checkpoint))
    monkeypatch.setenv(runtime.SIM2SIM_COMMAND_ENV, "python")

    report = preflight.run_unitree_groot_n17_sonic_policy_server_preflight(
        job_dir=tmp_path / "job",
        generated_at="now",
        runtime_setup_audit_path=_fake_runtime_setup_audit(tmp_path),
        run_command=_fake_runner(cuda_available=True),
        disk_usage_provider=lambda _path: SimpleNamespace(free=4 * 1024**3),
    )

    assert report["status"] == "blocked"
    assert report["policy_server_start_ready"] is False
    assert (
        "blocked_insufficient_local_disk_for_gr00t_n17_checkpoint_download"
        in report["blockers"]
    )
    assert report["n17_checkpoint"]["reference_kind"] == "repo_id"
    assert report["n17_checkpoint"]["exists_locally"] is False
    assert report["unitree_g1_sonic_embodiment_contract"]["status"] == "ok"
    assert report["local_disk"]["sufficient_for_minimum_download"] is False
    assert "UNITREE_G1_SONIC" in report["exact_commands_once_blockers_are_resolved"][
        "start_gr00t_n17_sonic_policy_server"
    ]
    assert report["claim_boundary"]["preflight_is_not_policy_execution"] is True
    assert report["raw_credentials_written_to_artifacts"] is False


def test_policy_server_preflight_ready_with_local_checkpoint_cuda_and_sim2sim(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _clear_env(monkeypatch)
    n17_checkpoint = tmp_path / "n17"
    sonic_checkpoint = tmp_path / "gear_sonic_deploy"
    n17_checkpoint.mkdir()
    sonic_checkpoint.mkdir()
    monkeypatch.setenv(runtime.GROOT_ROOT_ENV, str(_fake_groot_root(tmp_path)))
    monkeypatch.setenv(runtime.WBC_ROOT_ENV, str(_fake_wbc_root(tmp_path)))
    monkeypatch.setenv(runtime.N17_CHECKPOINT_ENV, str(n17_checkpoint))
    monkeypatch.setenv(runtime.SONIC_CHECKPOINT_ENV, str(sonic_checkpoint))
    monkeypatch.setenv(runtime.SIM2SIM_COMMAND_ENV, "python")

    report = preflight.run_unitree_groot_n17_sonic_policy_server_preflight(
        job_dir=tmp_path / "job",
        generated_at="now",
        run_command=_fake_runner(cuda_available=True),
        disk_usage_provider=lambda _path: SimpleNamespace(free=32 * 1024**3),
    )

    assert report["status"] == "ready_to_start"
    assert report["policy_server_start_ready"] is True
    assert report["blockers"] == []
    assert report["n17_checkpoint"]["reference_kind"] == "local_path"
    assert report["server_imports"]["torch_cuda_available"] is True
    assert report["unitree_policy_action_command_ran"] is False
    assert report["claim_boundary"]["physical_robot_readiness_proven"] is False
