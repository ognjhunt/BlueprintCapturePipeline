from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from blueprint_pipeline import mujoco_g1_wam_vla_policy_endpoint_eval as lane


@pytest.fixture(autouse=True)
def _bound_lerobot_smoke_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUEPRINT_UNITREE_LEROBOT_SMOKE_TIMEOUT_SECONDS", "0.5")


def _write_required_unitree_root(root: Path) -> Path:
    for path in lane._unitree_rl_gym_required_files(root).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    return root


def _clear_external_unitree_policy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "BLUEPRINT_UNITREE_LEROBOT_ROOT",
        "BLUEPRINT_UNITREE_LEROBOT_PYTHON",
        "BLUEPRINT_UNITREE_LEROBOT_POLICY_PATH",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND",
        "BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT",
        "BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT",
        "BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL",
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SIM2SIM_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)


def _fake_torch_module(num_actions: int) -> ModuleType:
    import numpy as np

    class FakeTensor:
        def __init__(self, values: Any) -> None:
            self._values = np.asarray(values, dtype=np.float32)

        def unsqueeze(self, _axis: int) -> "FakeTensor":
            return self

        def detach(self) -> "FakeTensor":
            return self

        def cpu(self) -> "FakeTensor":
            return self

        def numpy(self) -> Any:
            return self._values

    class FakeNoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    class FakePolicy:
        def eval(self) -> None:
            return None

        def __call__(self, _obs: FakeTensor) -> FakeTensor:
            return FakeTensor(np.full(num_actions, 0.5, dtype=np.float32))

    fake_torch = ModuleType("torch")
    fake_torch.no_grad = lambda: FakeNoGrad()  # type: ignore[attr-defined]
    fake_torch.from_numpy = lambda values: FakeTensor(values)  # type: ignore[attr-defined]
    fake_torch.jit = SimpleNamespace(load=lambda *_args, **_kwargs: FakePolicy())
    return fake_torch


def _fake_model_and_mujoco(*, position_target: bool = True) -> tuple[Any, Any]:
    leg_count = len(lane.UNITREE_RL_GYM_LEG_JOINT_NAMES)
    total_actuators = leg_count + 2
    joint_ids = {
        name: index for index, name in enumerate(lane.UNITREE_RL_GYM_LEG_JOINT_NAMES)
    }

    class FakeModel:
        nu = total_actuators
        actuator_trnid = [[index] for index in range(total_actuators)]
        jnt_qposadr = [7 + index for index in range(total_actuators)]
        jnt_dofadr = [6 + index for index in range(total_actuators)]
        actuator_ctrlrange = [[-0.2, 0.2] for _ in range(total_actuators)]
        actuator_gainprm = [[20.0] for _ in range(total_actuators)]
        actuator_biasprm = (
            [[0.0, -20.0] for _ in range(total_actuators)]
            if position_target
            else [[0.0, 0.0] for _ in range(total_actuators)]
        )

    fake_mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_ACTUATOR=1, mjOBJ_JOINT=2),
        mj_name2id=lambda _model, _obj, name: joint_ids.get(name, -1),
        mj_id2name=lambda _model, _obj, index: f"name_{index}",
    )
    return FakeModel(), fake_mujoco


def _fake_controller_config(num_actions: int) -> dict[str, Any]:
    return {
        "simulation_dt": 0.002,
        "control_decimation": 2,
        "kps": [2.0] * num_actions,
        "kds": [0.5] * num_actions,
        "default_angles": [0.0] * num_actions,
        "cmd_scale": [1.0, 1.0, 1.0],
        "num_actions": num_actions,
        "num_obs": 9 + 3 * num_actions + 2,
        "action_scale": 0.25,
        "dof_pos_scale": 1.0,
        "dof_vel_scale": 1.0,
        "ang_vel_scale": 1.0,
        "policy_path": "{LEGGED_GYM_ROOT_DIR}/exported/policy.pt",
    }


def _fake_data(model: Any) -> Any:
    qpos = [0.0] * 64
    qvel = [0.0] * 64
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    return SimpleNamespace(qpos=qpos, qvel=qvel, ctrl=[0.0] * int(model.nu), time=0.25)


def _install_fake_mujoco_for_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    class FakeVec(list[float]):
        def __setitem__(self, key: object, value: object) -> None:
            if isinstance(key, slice) and isinstance(value, (int, float)):
                for index in range(*key.indices(len(self))):
                    super().__setitem__(index, float(value))
                return
            super().__setitem__(key, value)  # type: ignore[index]

    class FakeModel:
        nq = 14
        nv = 12
        nu = 1
        opt = SimpleNamespace(timestep=0.05)
        actuator_trnid = [[0]]
        jnt_qposadr = [0, 7]
        jnt_dofadr = [0, 6]
        geom_bodyid = [0, 1]

        def __init__(self) -> None:
            self.qpos0 = FakeVec(
                [0.0, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0, 0.36, -0.65, 0.27, 1.0, 0.0, 0.0, 0.0]
            )
            self.key_qpos = [self.qpos0.copy()]

        @classmethod
        def from_xml_path(cls, _path: str) -> "FakeModel":
            return cls()

    class FakeData:
        def __init__(self, model: FakeModel) -> None:
            self.qpos = FakeVec(model.qpos0.copy())
            self.qvel = FakeVec([0.0] * model.nv)
            self.ctrl = FakeVec([0.0] * model.nu)
            self.time = 0.0
            self.ncon = 0
            self.contact: list[object] = []

    def fake_mj_name2id(_model: FakeModel, _obj: int, name: str) -> int:
        return {
            "floating_base_joint": 0,
            "blueprint_light_object_freejoint": 1,
            "stand": 0,
        }.get(name, -1)

    def fake_step(model: FakeModel, data: FakeData) -> None:
        data.qpos[0] += data.qvel[0] * model.opt.timestep
        data.qpos[1] += data.qvel[1] * model.opt.timestep
        data.time += model.opt.timestep

    fake_mujoco = ModuleType("mujoco")
    fake_mujoco.__version__ = "fake-1.0"
    fake_mujoco.__file__ = "fake_mujoco.py"
    fake_mujoco.MjModel = FakeModel
    fake_mujoco.MjData = FakeData
    fake_mujoco.mjtObj = SimpleNamespace(mjOBJ_JOINT=1, mjOBJ_KEY=2, mjOBJ_GEOM=3, mjOBJ_BODY=4)
    fake_mujoco.mjtCamera = SimpleNamespace(mjCAMERA_FREE=5)
    fake_mujoco.MjvCamera = lambda: SimpleNamespace(
        type=None,
        lookat=[0.0, 0.0, 0.0],
        distance=0.0,
        azimuth=0.0,
        elevation=0.0,
    )
    fake_mujoco.mj_name2id = fake_mj_name2id
    fake_mujoco.mj_forward = lambda _model, _data: None
    fake_mujoco.mj_step = fake_step
    fake_mujoco.mj_id2name = lambda _model, obj, index: {
        (3, 0): "blueprint_reference_floor",
        (3, 1): "blueprint_light_object_geom",
        (4, 0): "floor_body",
        (4, 1): "blueprint_light_object",
    }.get((obj, index))
    fake_mujoco.mj_contactForce = lambda _model, _data, _index, force: force.__setitem__(0, 0.5)
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)

    g1_root = tmp_path / "g1"
    g1_root.mkdir(exist_ok=True)
    (g1_root / "g1.xml").write_text("<mujoco/>", encoding="utf-8")
    monkeypatch.setattr(lane, "_resolve_g1_model_root", lambda **_kwargs: g1_root)

    def fake_write_g1_xml(_src: Path, dst: Path) -> None:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text("<mujoco/>", encoding="utf-8")

    monkeypatch.setattr(lane, "_write_g1_xml_with_absolute_meshes", fake_write_g1_xml)
    monkeypatch.setattr(
        lane,
        "_asset_source_manifest",
        lambda _root: {"asset_source": "fake", "asset_source_sha256": "abc"},
    )
    return fake_mujoco


def test_unitree_root_discovery_and_navigation_policy_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing_unitree"
    present = _write_required_unitree_root(tmp_path / "present_unitree")
    monkeypatch.setattr(
        lane,
        "_default_unitree_rl_gym_root_candidates",
        lambda: [("missing", missing), ("present", present)],
    )

    assert lane._select_unitree_rl_gym_root(explicit_root=None, discovery={}) == present

    for env_name in (
        "BLUEPRINT_REALISTIC_G1_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
        "UNITREE_G1_POLICY_COMMAND",
        "BLUEPRINT_UNITREE_G1_POLICY_ROOT",
        "BLUEPRINT_UNITREE_G1_POLICY_SOURCE_ROOT",
        "BLUEPRINT_UNITREE_RL_GYM_ROOT",
        "UNITREE_G1_POLICY_ROOT",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(lane, "_default_unitree_rl_gym_root_candidates", lambda: [])
    monkeypatch.setattr(lane, "OFFICIAL_UNITREE_G1_POLICY_SOURCES", [{"name": ""}])
    monkeypatch.setattr(lane, "EXTRA_G1_POLICY_ONLINE_CANDIDATES", [])

    discovery = lane.discover_realistic_navigation_policy(generated_at="now")

    assert discovery["status"] == "blocked_missing_controller_command"
    assert discovery["blockers"] == ["blocked_missing_realistic_g1_navigation_policy"]

    command = tmp_path / "unitree-policy-command"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_REALISTIC_G1_POLICY_COMMAND", str(command))
    command_only = lane.discover_realistic_navigation_policy(generated_at="now")
    assert command_only["status"] == "candidate_available_for_endpoint_controller_selection"
    assert command_only["blockers"] == [
        "blocked_controller_command_not_integrated_into_same_scene_endpoint_rollouts"
    ]


def test_sidecar_replay_and_bridge_blocked_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import unitree_g1_policy_execution as unitree

    unitree_root = _write_required_unitree_root(tmp_path / "unitree_rl_gym")
    monkeypatch.setattr(lane, "_default_unitree_rl_gym_root_candidates", lambda: [])
    blocked_sidecar = lane._run_official_unitree_controller_sidecar(
        job_dir=tmp_path / "job",
        job_id="job",
        generated_at="now",
        unitree_rl_gym_root=None,
        navigation_discovery={},
        enabled=True,
        max_steps=1,
    )
    assert blocked_sidecar["status"] == "blocked"
    assert blocked_sidecar["blockers"] == ["blocked_missing_unitree_rl_gym_root_or_required_files"]

    monkeypatch.setattr(
        unitree,
        "build_unitree_g1_policy_execution",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("sidecar failed")),
    )
    failed_sidecar = lane._run_official_unitree_controller_sidecar(
        job_dir=tmp_path / "job",
        job_id="job",
        generated_at="now",
        unitree_rl_gym_root=unitree_root,
        navigation_discovery={},
        enabled=True,
        max_steps=1,
    )
    assert failed_sidecar["error_type"] == "RuntimeError"

    blocked_replay = lane._run_unitree_controller_replay_from_endpoint_actions(
        job_dir=tmp_path / "job",
        job_id="job",
        generated_at="now",
        unitree_rl_gym_root=None,
        navigation_discovery={},
        enabled=True,
        max_steps=1,
        command_rows=[{"command_xyz": [0.1, 0.0, 0.0]}],
    )
    assert blocked_replay["blockers"] == ["blocked_missing_unitree_rl_gym_root_or_required_files"]

    missing_command = lane._run_unitree_controller_replay_from_endpoint_actions(
        job_dir=tmp_path / "job",
        job_id="job",
        generated_at="now",
        unitree_rl_gym_root=unitree_root,
        navigation_discovery={},
        enabled=True,
        max_steps=1,
        command_rows=[],
    )
    assert missing_command["blockers"] == ["blocked_missing_endpoint_command_vector"]

    failed_replay = lane._run_unitree_controller_replay_from_endpoint_actions(
        job_dir=tmp_path / "job",
        job_id="job",
        generated_at="now",
        unitree_rl_gym_root=unitree_root,
        navigation_discovery={},
        enabled=True,
        max_steps=1,
        command_rows=[{"command_xyz": [0.1, 0.0, 0.0]}],
    )
    assert failed_replay["blockers"] == ["blocked_unitree_endpoint_action_controller_replay_failed"]

    bridge = lane.build_unitree_controller_bridge_manifest(
        generated_at="now",
        command_rows=[{"command_xyz": [0.1, 0.0, 0.0]}],
        official_controller_sidecar={},
        endpoint_replay={},
        same_scene_controller={
            "same_scene_controller_backend_integrated": True,
            "balanced_walking_controller_proven": False,
            "blockers": ["blocked_same_scene_unitree_controller_rollout_fell"],
        },
    )
    assert "blocked_same_scene_unitree_controller_rollout_fell" in bridge["blockers"]


def test_mujoco_name_helpers_return_ids_and_swallow_lookup_errors() -> None:
    good = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_ACTUATOR=1, mjOBJ_JOINT=2),
        mj_id2name=lambda _model, _obj, index: f"name-{index}",
        mj_name2id=lambda _model, _obj, name: {"known": 7}[name],
    )
    bad = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_ACTUATOR=1, mjOBJ_JOINT=2),
        mj_id2name=lambda *_args: (_ for _ in ()).throw(RuntimeError("bad")),
        mj_name2id=lambda *_args: (_ for _ in ()).throw(RuntimeError("bad")),
    )

    assert lane._mujoco_name(good, object(), good.mjtObj.mjOBJ_JOINT, 3) == "name-3"
    assert lane._mujoco_name(bad, object(), bad.mjtObj.mjOBJ_JOINT, 3) is None
    assert lane._actuator_id_by_name(good, object(), "known") == 7
    assert lane._actuator_id_by_name(bad, object(), "known") == -1
    assert lane._joint_id_by_name(good, object(), "known") == 7
    assert lane._joint_id_by_name(bad, object(), "known") == -1


def test_same_scene_controller_creation_failures_and_successful_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import unitree_g1_policy_execution as unitree

    num_actions = len(lane.UNITREE_RL_GYM_LEG_JOINT_NAMES)
    unitree_root = _write_required_unitree_root(tmp_path / "unitree_rl_gym")
    model, fake_mujoco = _fake_model_and_mujoco(position_target=True)
    data = _fake_data(model)
    monkeypatch.setattr(lane, "_default_unitree_rl_gym_root_candidates", lambda: [])
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module(num_actions))

    blocked_controller, blocked_manifest = lane._create_same_scene_unitree_rl_gym_controller(
        model=model,
        data=data,
        mujoco_module=fake_mujoco,
        root_qpos=0,
        root_dof=0,
        generated_at="now",
        unitree_rl_gym_root=None,
        navigation_discovery={},
        enabled=True,
    )
    assert blocked_controller is None
    assert blocked_manifest["blockers"] == ["blocked_missing_unitree_rl_gym_root_or_required_files"]

    monkeypatch.setattr(unitree, "_read_yaml", lambda _path: (_ for _ in ()).throw(ValueError("bad yaml")))
    failed_controller, failed_manifest = lane._create_same_scene_unitree_rl_gym_controller(
        model=model,
        data=data,
        mujoco_module=fake_mujoco,
        root_qpos=0,
        root_dof=0,
        generated_at="now",
        unitree_rl_gym_root=unitree_root,
        navigation_discovery={},
        enabled=True,
    )
    assert failed_controller is None
    assert failed_manifest["blockers"] == ["blocked_unitree_rl_gym_policy_or_config_load_failed"]

    config = _fake_controller_config(num_actions)
    monkeypatch.setattr(unitree, "_read_yaml", lambda _path: dict(config, num_actions=1))
    monkeypatch.setattr(unitree, "_sha256", lambda _path: "sha")
    mismatch_controller, mismatch_manifest = lane._create_same_scene_unitree_rl_gym_controller(
        model=model,
        data=data,
        mujoco_module=fake_mujoco,
        root_qpos=0,
        root_dof=0,
        generated_at="now",
        unitree_rl_gym_root=unitree_root,
        navigation_discovery={},
        enabled=True,
    )
    assert mismatch_controller is None
    assert mismatch_manifest["blockers"] == ["blocked_unitree_policy_action_dimension_mismatch"]

    missing_mujoco = SimpleNamespace(
        mjtObj=fake_mujoco.mjtObj,
        mj_name2id=lambda *_args: -1,
    )
    monkeypatch.setattr(unitree, "_read_yaml", lambda _path: dict(config))
    missing_controller, missing_manifest = lane._create_same_scene_unitree_rl_gym_controller(
        model=model,
        data=data,
        mujoco_module=missing_mujoco,
        root_qpos=0,
        root_dof=0,
        generated_at="now",
        unitree_rl_gym_root=unitree_root,
        navigation_discovery={},
        enabled=True,
    )
    assert missing_controller is None
    assert missing_manifest["blockers"] == ["blocked_same_scene_missing_required_leg_actuators_or_joints"]

    controller, ready_manifest = lane._create_same_scene_unitree_rl_gym_controller(
        model=model,
        data=data,
        mujoco_module=fake_mujoco,
        root_qpos=0,
        root_dof=0,
        generated_at="now",
        unitree_rl_gym_root=unitree_root,
        navigation_discovery={},
        enabled=True,
    )
    assert controller is not None
    assert ready_manifest["status"] == "ready"
    assert ready_manifest["position_target_action_clip_abs"] == 0.5
    assert ready_manifest["position_target_action_clip_env"] == (
        "BLUEPRINT_UNITREE_RL_GYM_POSITION_TARGET_ACTION_CLIP_ABS"
    )
    assert ready_manifest["position_target_action_clip_default_abs"] == 0.5

    controller.reset(data)
    update = controller.step(data=data, step=0, command_xyz=[0.2, -0.1, 0.05])
    assert update is not None
    assert update["command_xyz"] == [0.2, -0.1, 0.05]
    assert update["policy_action_clipped"] is False
    assert update["raw_policy_action"] == update["action"]
    assert controller.step(data=data, step=1, command_xyz=[0.0, 0.0, 0.0]) is None

    class SpikeTensor:
        def detach(self) -> "SpikeTensor":
            return self

        def cpu(self) -> "SpikeTensor":
            return self

        def numpy(self) -> Any:
            import numpy as np

            return np.full(num_actions, 8.0, dtype=np.float32)

    class SpikePolicy:
        def __call__(self, _obs: Any) -> SpikeTensor:
            return SpikeTensor()

    controller.policy = SpikePolicy()
    clipped_update = controller.step(data=data, step=2, command_xyz=[0.0, 0.0, 0.0])
    assert clipped_update is not None
    assert clipped_update["policy_action_clipped"] is True
    assert clipped_update["policy_action_clip_abs"] == 0.5
    assert set(clipped_update["raw_policy_action"]) == {8.0}
    assert set(clipped_update["action"]) == {0.5}
    assert controller.policy_action_clipped_count == 1
    assert controller.max_raw_policy_action_abs == pytest.approx(8.0)
    assert controller.max_applied_policy_action_abs == pytest.approx(0.5)
    monkeypatch.setenv("BLUEPRINT_UNITREE_RL_GYM_POSITION_TARGET_ACTION_CLIP_ABS", "0.25")
    env_clipped_update = controller.step(data=data, step=4, command_xyz=[0.0, 0.0, 0.0])
    assert env_clipped_update is not None
    assert env_clipped_update["policy_action_clip_abs"] == 0.25
    assert set(env_clipped_update["action"]) == {0.25}
    controller.model.actuator_ctrlrange = object()
    controller.apply(data=data)

    fallback_model, fallback_mujoco = _fake_model_and_mujoco(position_target=True)
    fallback_model.actuator_gainprm = object()
    fallback_model.actuator_trnid[-1] = [-1]
    fallback_controller, fallback_manifest = lane._create_same_scene_unitree_rl_gym_controller(
        model=fallback_model,
        data=_fake_data(fallback_model),
        mujoco_module=fallback_mujoco,
        root_qpos=0,
        root_dof=0,
        generated_at="now",
        unitree_rl_gym_root=unitree_root,
        navigation_discovery={},
        enabled=True,
    )
    assert fallback_controller is not None
    assert fallback_controller.actuator_output_mode == "torque"
    assert fallback_manifest["status"] == "ready"

    torque_model, torque_mujoco = _fake_model_and_mujoco(position_target=False)
    torque_controller = lane._SameSceneUnitreeRLGymController(
        model=torque_model,
        mujoco_module=torque_mujoco,
        root_qpos=0,
        root_dof=0,
        selected_root=unitree_root,
        config=config,
        policy=controller.policy,
        policy_path=controller.policy_path,
        config_path=controller.config_path,
        leg_actuator_ids=list(range(num_actions)),
        leg_qpos_addrs=[7 + index for index in range(num_actions)],
        leg_qvel_addrs=[6 + index for index in range(num_actions)],
        upper_hold_actuator_ids=[num_actions],
        upper_hold_qpos_addrs=[7 + num_actions],
        actuator_output_mode="torque",
    )
    torque_data = _fake_data(torque_model)
    torque_controller.upper_hold_targets = [1.25]
    torque_controller.target_dof_pos = torque_controller.default_angles + 0.5
    torque_controller.apply(data=torque_data)
    assert torque_data.ctrl[num_actions] == 1.25


def test_run_rejects_unknown_controller_backend() -> None:
    with pytest.raises(ValueError, match="controller_backend must be one of"):
        lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(controller_backend="bad")


def test_contact_metadata_exception_fallbacks_and_auto_controller_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_GEOM=1, mjOBJ_BODY=2),
        mj_id2name=lambda _model, _obj, index: f"name_{index}",
    )

    class MissingGeomBodyIds:
        ngeom = 0

        @property
        def geom_bodyid(self) -> object:
            raise RuntimeError("missing geom body ids")

    assert lane._build_contact_metadata(MissingGeomBodyIds(), fake_mujoco) == {}

    class RaisingGeomBodyIndex:
        ngeom = 1

        class GeomBodyIds:
            def __getitem__(self, _index: int) -> int:
                raise RuntimeError("bad body index")

        geom_bodyid = GeomBodyIds()

    metadata = lane._build_contact_metadata(RaisingGeomBodyIndex(), fake_mujoco)
    assert metadata[0]["body_id"] == -1
    fallback = lane._contact_metadata_for_geom(
        model=RaisingGeomBodyIndex(),
        mujoco_module=fake_mujoco,
        contact_metadata=None,
        geom_id=0,
    )
    assert fallback["body_id"] == -1

    _clear_external_unitree_policy_env(monkeypatch)
    _install_fake_mujoco_for_run(monkeypatch, tmp_path)
    monkeypatch.setattr(lane, "_select_unitree_rl_gym_root", lambda **_kwargs: None)
    monkeypatch.setenv("BLUEPRINT_UNITREE_LEROBOT_SMOKE_TIMEOUT_SECONDS", "0.5")
    summary = lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
        job_dir=tmp_path / "auto-freejoint",
        max_tasks=1,
        max_spawns=1,
        steps_per_episode=1,
        controller_backend="auto",
        render=False,
        generated_at="now",
    )
    assert summary["controller_backend"] == "freejoint_proxy"


def test_unitree_backend_run_blocks_when_same_scene_controller_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mujoco_for_run(monkeypatch, tmp_path)
    monkeypatch.setattr(
        lane,
        "_create_same_scene_unitree_rl_gym_controller",
        lambda **_kwargs: (
            None,
            {
                "status": "blocked",
                "same_scene_controller_backend_integrated": False,
                "blockers": ["blocked_missing_unitree_rl_gym_root_or_required_files"],
            },
        ),
    )

    summary = lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
        job_dir=tmp_path / "blocked-unitree-backend",
        max_tasks=1,
        max_spawns=1,
        steps_per_episode=1,
        render=False,
        controller_backend="unitree_rl_gym",
        generated_at="now",
    )

    assert summary["status"] == "blocked"
    assert summary["same_scene_unitree_controller_backend_integrated"] is False
    assert summary["blockers"] == ["blocked_missing_unitree_rl_gym_root_or_required_files"]
    assert (tmp_path / "blocked-unitree-backend" / "policy_evaluation_summary.json").is_file()


def test_unitree_backend_run_records_controller_updates_and_full_episode_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_external_unitree_policy_env(monkeypatch)
    fake_mujoco = _install_fake_mujoco_for_run(monkeypatch, tmp_path)

    class FakeSameSceneController:
        def reset(self, data: object) -> None:
            return None

        def step(self, *, data: object, step: int, command_xyz: list[float]) -> dict[str, object]:
            return {
                "schema_version": "unitree_rl_gym_same_scene_controller_update.v1",
                "step": step,
                "sim_time_s": 0.0,
                "command_xyz": command_xyz,
                "target_dof_pos": [0.0],
                "action": [0.0],
            }

    monkeypatch.setattr(
        lane,
        "_create_same_scene_unitree_rl_gym_controller",
        lambda **_kwargs: (
            FakeSameSceneController(),
            {
                "status": "ready",
                "same_scene_controller_backend_integrated": True,
                "official_unitree_controller_used": True,
                "balanced_walking_controller_proven": False,
                "blockers": [],
            },
        ),
    )

    class FakeRenderer:
        def update_scene(self, *_args: object, **_kwargs: object) -> None:
            return None

        def render(self) -> list[list[list[int]]]:
            return [[[0, 0, 0]]]

        def close(self) -> None:
            return None

    class FakeImage:
        @staticmethod
        def fromarray(_frame: object) -> "FakeImage":
            return FakeImage()

        def save(self, path: Path) -> None:
            Path(path).write_bytes(b"png")

    fake_mujoco.Renderer = lambda *_args, **_kwargs: FakeRenderer()
    fake_pil = ModuleType("PIL")
    fake_pil.Image = FakeImage
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setattr(
        lane,
        "_write_video_from_frames",
        lambda **kwargs: {"path": str(kwargs["output_path"]), "status": "complete", "size_bytes": 3},
    )
    monkeypatch.setattr(
        lane,
        "_ffprobe_video",
        lambda path: {"path": str(path), "status": "complete", "duration_s": 0.05, "frame_count": 1},
    )

    summary = lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
        job_dir=tmp_path / "ready-unitree-backend",
        max_tasks=1,
        max_spawns=1,
        steps_per_episode=1,
        policy_interval_steps=1,
        render=True,
        render_frame_count=0,
        video_frame_stride_steps=1,
        controller_backend="unitree_rl_gym",
        generated_at="now",
    )

    assert summary["status"] == "completed"
    assert summary["same_scene_unitree_controller_backend_integrated"] is True
    same_scene_manifest = json.loads(
        (tmp_path / "ready-unitree-backend" / "same_scene_unitree_controller_backend_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert same_scene_manifest["controller_update_count"] == 1
    assert same_scene_manifest["same_scene_controller_clamped_update_count"] == 1
    assert summary["same_scene_controller_clamped_update_count"] == 1
    video_status = json.loads(
        (tmp_path / "ready-unitree-backend" / "video_generation_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert video_status["videos"][0]["full_episode_video"] is True
    attempts = json.loads(
        (tmp_path / "ready-unitree-backend" / "normalized_attempt_trace.json").read_text(
            encoding="utf-8"
        )
    )["attempts"]
    assert attempts[0]["video_analysis_binding"]["human_review_media_source"] == (
        "full_episode_mujoco_video"
    )

    def fake_falling_step(model: object, data: object) -> None:
        data.qpos[2] = 0.1
        data.time += 0.05

    fake_mujoco.mj_step = fake_falling_step
    falling_summary = lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
        job_dir=tmp_path / "falling-unitree-backend",
        max_tasks=1,
        max_spawns=1,
        steps_per_episode=1,
        policy_interval_steps=1,
        render=False,
        controller_backend="unitree_rl_gym",
        generated_at="now",
    )

    assert falling_summary["failed_episode_count"] == 1
    falling_manifest = json.loads(
        (
            tmp_path / "falling-unitree-backend" / "same_scene_unitree_controller_backend_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert "blocked_same_scene_unitree_controller_rollout_fell" in falling_manifest["blockers"]
