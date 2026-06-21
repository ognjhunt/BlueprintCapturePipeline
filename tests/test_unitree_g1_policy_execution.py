from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from blueprint_pipeline import unitree_g1_policy_execution as unitree


def _write_unitree_repo(repo_root: Path, *, control_decimation: int = 1) -> None:
    config_dir = repo_root / "deploy" / "deploy_mujoco" / "configs"
    config_dir.mkdir(parents=True)
    policy_path = repo_root / "policy.pt"
    xml_path = repo_root / "robot.xml"
    policy_path.write_bytes(b"policy")
    xml_path.write_text("<mujoco />", encoding="utf-8")
    config = {
        "policy_path": "{LEGGED_GYM_ROOT_DIR}/policy.pt",
        "xml_path": "{LEGGED_GYM_ROOT_DIR}/robot.xml",
        "simulation_dt": 0.1,
        "control_decimation": control_decimation,
        "kps": [2.0, 2.0],
        "kds": [0.5, 0.5],
        "default_angles": [0.1, -0.1],
        "cmd_init": [0.2, 0.0, 0.0],
        "cmd_scale": [1.0, 1.0, 1.0],
        "num_actions": 2,
        "num_obs": 17,
        "action_scale": 0.25,
        "dof_pos_scale": 1.5,
        "dof_vel_scale": 2.0,
        "ang_vel_scale": 0.5,
    }
    (config_dir / "g1.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _install_fake_mujoco_and_torch(monkeypatch, *, action_values: list[float]) -> None:  # type: ignore[no-untyped-def]
    class FakeModel:
        def __init__(self) -> None:
            self.opt = types.SimpleNamespace(timestep=0.0)

        @classmethod
        def from_xml_path(cls, _path: str):
            return cls()

    class FakeData:
        def __init__(self, _model) -> None:  # type: ignore[no-untyped-def]
            self.qpos = np.zeros(9, dtype=np.float32)
            self.qpos[3] = 1.0
            self.qpos[7:] = np.asarray([0.1, -0.1], dtype=np.float32)
            self.qvel = np.zeros(8, dtype=np.float32)
            self.ctrl = np.zeros(2, dtype=np.float32)

    def fake_mj_step(_model, data) -> None:  # type: ignore[no-untyped-def]
        data.qpos[:3] += 0.01

    fake_mujoco = types.SimpleNamespace(MjModel=FakeModel, MjData=FakeData, mj_step=fake_mj_step)

    class FakeTensor:
        def __init__(self, array) -> None:  # type: ignore[no-untyped-def]
            self._array = np.asarray(array, dtype=np.float32)

        def unsqueeze(self, _dim: int):
            return self

        def detach(self):
            return self

        def numpy(self):
            return self._array

    class FakePolicy:
        def __call__(self, _obs) -> FakeTensor:  # type: ignore[no-untyped-def]
            return FakeTensor([action_values])

    fake_torch = types.SimpleNamespace(
        jit=types.SimpleNamespace(load=lambda *_args, **_kwargs: FakePolicy()),
        from_numpy=lambda array: FakeTensor(array),
    )

    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_unitree_helpers_hash_yaml_math_and_repo_head(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abc")
    assert unitree._sha256(payload) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    yaml_path = tmp_path / "payload.yaml"
    yaml_path.write_text("a: 1\n", encoding="utf-8")
    assert unitree._read_yaml(yaml_path) == {"a": 1}
    yaml_path.write_text("- not\n- object\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML payload is not an object"):
        unitree._read_yaml(yaml_path)

    gravity = unitree._gravity_orientation(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert gravity.tolist() == [0.0, -0.0, -1.0]
    tau = unitree._pd_control(
        np.asarray([1.0]),
        np.asarray([0.25]),
        np.asarray([2.0]),
        np.asarray([0.0]),
        np.asarray([0.5]),
        np.asarray([0.1]),
    )
    assert tau.tolist() == [1.45]

    assert unitree._repo_head(tmp_path) is None
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("abc123", encoding="utf-8")
    assert unitree._repo_head(tmp_path) == "abc123"
    (git / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")
    assert unitree._repo_head(tmp_path) == "ref: refs/heads/main"
    ref = git / "refs" / "heads" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("def456", encoding="utf-8")
    assert unitree._repo_head(tmp_path) == "def456"


def test_build_unitree_g1_policy_execution_with_fake_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _install_fake_mujoco_and_torch(monkeypatch, action_values=[0.2, -0.2])
    capture_root = tmp_path / "capture"
    repo_root = tmp_path / "unitree_rl_gym"
    _write_unitree_repo(repo_root)
    output_dir = tmp_path / "out"

    manifest = unitree.build_unitree_g1_policy_execution(
        capture_root=capture_root,
        unitree_rl_gym_root=repo_root,
        job_id="job-g1",
        duration_seconds=0.2,
        max_steps=2,
        output_dir=output_dir,
        command_xyz=[0.3, 0.1, -0.2],
    )

    assert manifest["status"] == "completed"
    assert manifest["job_id"] == "job-g1"
    assert manifest["metrics"]["steps"] == 2
    assert manifest["metrics"]["control_updates"] == 2
    assert manifest["metrics"]["command_xyz"] == pytest.approx([0.3, 0.1, -0.2])
    assert manifest["metrics"]["command_source"] == "caller_override"
    assert manifest["execution"]["command_source"] == "caller_override"
    assert manifest["metrics"]["mean_abs_action"] == pytest.approx(0.2)
    assert manifest["proof_boundary"]["non_default_policy_execution_trace_proven"] is True
    assert manifest["blockers"] == []
    output_path = Path(manifest["output_path"])
    assert output_path.is_file()
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["output_path"] == str(output_path)
    assert Path(manifest["execution"]["trace_path"]).read_text(encoding="utf-8").count("\n") == 2


def test_build_unitree_g1_policy_execution_blocks_without_control_updates(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _install_fake_mujoco_and_torch(monkeypatch, action_values=[float("nan"), 0.0])
    capture_root = tmp_path / "capture"
    repo_root = tmp_path / "unitree_rl_gym"
    _write_unitree_repo(repo_root, control_decimation=99)

    manifest = unitree.build_unitree_g1_policy_execution(
        capture_root=capture_root,
        unitree_rl_gym_root=repo_root,
        job_id="job-blocked",
        duration_seconds=0.1,
        max_steps=1,
        output_dir=tmp_path / "blocked",
    )

    assert manifest["status"] == "blocked"
    assert manifest["metrics"]["control_updates"] == 1
    assert manifest["metrics"]["finite_actions"] is False
    assert manifest["blockers"] == ["official_unitree_g1_policy_execution_failed"]


def test_build_unitree_g1_policy_execution_rejects_invalid_command_shape(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _install_fake_mujoco_and_torch(monkeypatch, action_values=[0.0, 0.0])
    repo_root = tmp_path / "unitree_rl_gym"
    _write_unitree_repo(repo_root)

    with pytest.raises(ValueError, match="command_xyz must contain"):
        unitree.build_unitree_g1_policy_execution(
            capture_root=tmp_path / "capture",
            unitree_rl_gym_root=repo_root,
            job_id="job-invalid-command",
            output_dir=tmp_path / "invalid",
            command_xyz=[0.1, 0.2],
        )


def test_unitree_g1_policy_execution_main_prints_and_returns_status(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    calls: list[dict[str, object]] = []

    def fake_build(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return {
            "status": "completed" if len(calls) == 1 else "blocked",
            "output_path": str(tmp_path / f"manifest-{len(calls)}.json"),
        }

    monkeypatch.setattr(unitree, "build_unitree_g1_policy_execution", fake_build)
    argv = [
        "--capture-root",
        str(tmp_path / "capture"),
        "--unitree-rl-gym-root",
        str(tmp_path / "repo"),
        "--job-id",
        "job-main",
        "--duration-seconds",
        "0.5",
        "--max-steps",
        "3",
        "--output-dir",
        str(tmp_path / "out"),
        "--command-xyz",
        "0.4",
        "0.1",
        "-0.2",
    ]

    assert unitree.main(argv) == 0
    assert str(tmp_path / "manifest-1.json") in capsys.readouterr().out
    assert calls[0]["job_id"] == "job-main"
    assert calls[0]["duration_seconds"] == 0.5
    assert calls[0]["max_steps"] == 3
    assert calls[0]["command_xyz"] == [0.4, 0.1, -0.2]
    assert unitree.main(argv) == 1
