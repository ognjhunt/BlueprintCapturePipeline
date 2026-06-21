from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from blueprint_pipeline import mujoco_g1_wam_vla_policy_endpoint_eval as lane


def _load_mujoco_edge_helpers() -> Any:
    helper_path = Path(__file__).with_name(
        "test_mujoco_g1_wam_vla_policy_endpoint_eval_coverage_edges.py"
    )
    spec = importlib.util.spec_from_file_location("_mujoco_edge_helpers_for_zz", helper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_late_mujoco_state_sensitive_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helpers = _load_mujoco_edge_helpers()
    num_actions = len(lane.UNITREE_RL_GYM_LEG_JOINT_NAMES)
    unitree_root = helpers._write_required_unitree_root(tmp_path / "unitree")
    config = helpers._fake_controller_config(num_actions)
    fake_torch = helpers._fake_torch_module(num_actions)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    from blueprint_pipeline import unitree_g1_policy_execution as unitree

    monkeypatch.setattr(unitree, "_read_yaml", lambda _path: dict(config))
    monkeypatch.setattr(unitree, "_sha256", lambda _path: "sha256")
    monkeypatch.setattr(
        unitree,
        "build_unitree_g1_policy_execution",
        lambda **_kwargs: {
            "status": "completed",
            "policy_id": "unitree-rl-gym",
            "proof_boundary": {"non_default_policy_execution_trace_proven": True},
            "metrics": {"command_xyz": [0.1, 0.0, 0.0]},
        },
    )

    model, fake_mujoco = helpers._fake_model_and_mujoco(position_target=True)
    data = helpers._fake_data(model)
    sidecar = lane._run_official_unitree_controller_sidecar(
        job_dir=tmp_path / "sidecar-job",
        job_id="job",
        generated_at="now",
        unitree_rl_gym_root=unitree_root,
        navigation_discovery={},
        enabled=True,
        max_steps=2,
        command_xyz=[0.1, 0.0, 0.0],
    )
    assert sidecar["status"] == "completed"

    action_rows = [
        {
            "episode_id": "ep1",
            "scenario_eval_run_id": "run1",
            "task_id": "approach_target",
            "spawn_id": "nominal",
            "step": 1,
            "sim_time_s": 0.1,
            "source": "endpoint",
            "normalized_action": {
                "normalization_status": "accepted",
                "action_type": "base_velocity",
                "vx_mps": 3.0,
                "vy_mps": -3.0,
                "yaw_rate_rad_s": 3.0,
            },
        },
        {
            "normalized_action": {"normalization_status": "rejected"},
            "rejected": True,
        },
    ]
    command_rows = lane._unitree_command_rows_from_endpoint_actions(action_rows)
    assert command_rows[0]["controller_command_clamped"] is True
    assert lane._bounded_float(5.0, -1.0, 1.0) == 1.0
    assert lane._representative_unitree_command(command_rows) is not None
    assert lane._representative_unitree_command([{"command_xyz": [0.0, 0.0, 0.0]}]) == [
        0.0,
        0.0,
        0.0,
    ]
    assert lane._representative_unitree_command([]) is None

    replay = lane._run_unitree_controller_replay_from_endpoint_actions(
        job_dir=tmp_path / "replay-job",
        job_id="job",
        generated_at="now",
        unitree_rl_gym_root=unitree_root,
        navigation_discovery={},
        enabled=True,
        max_steps=2,
        command_rows=command_rows,
    )
    assert replay["status"] == "completed"
    bridge = lane.build_unitree_controller_bridge_manifest(
        generated_at="now",
        command_rows=command_rows,
        endpoint_replay={"official_unitree_controller_used": True},
        official_controller_sidecar={"official_unitree_controller_used": False},
        same_scene_controller={"same_scene_controller_backend_integrated": False},
    )
    assert bridge["status"] == "bridge_ready_for_implementation"

    lane._set_joint_position_holds(model, data)
    bad_mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_ACTUATOR=1, mjOBJ_JOINT=2),
        mj_name2id=lambda *_args: (_ for _ in ()).throw(RuntimeError("lookup")),
        mj_id2name=lambda *_args: (_ for _ in ()).throw(RuntimeError("lookup")),
    )
    assert lane._mujoco_name(bad_mujoco, model, 1, 0) is None
    assert lane._actuator_id_by_name(bad_mujoco, model, "missing") == -1
    assert lane._joint_id_by_name(bad_mujoco, model, "missing") == -1

    contact_model = SimpleNamespace(geom_bodyid=[0, 1])
    contact_data = SimpleNamespace(
        ncon=1,
        contact=[
            SimpleNamespace(
                geom1=0,
                geom2=1,
                dist=0.01,
                pos=[0.1, 0.2, 0.3],
            )
        ],
    )
    contact_mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_GEOM=1, mjOBJ_BODY=2),
        mj_id2name=lambda _model, obj_type, index: {
            (1, 0): "blueprint_reference_floor",
            (1, 1): "blueprint_light_object_geom",
            (2, 0): "floor_body",
            (2, 1): "blueprint_light_object",
        }.get((obj_type, index)),
        mj_contactForce=lambda *_args: (_ for _ in ()).throw(RuntimeError("force")),
    )
    contacts = lane._contact_records(contact_model, contact_data, contact_mujoco)
    assert contacts[0]["object_contact"] is True
    assert lane._object_pose(SimpleNamespace(qpos=[0, 0, 0, 1, 0, 0, 0]), None) == {
        "available": False
    }

    assert lane._extract_action({"no_action": True}) == {"no_action": True}
    observation = {
        "episode_id": "ep1",
        "base_pose": {"position": [0.0, 0.0, 0.79], "yaw_rad": 0.0},
        "object_state": {"position": [0.2, -0.1, 0.3]},
    }
    for raw_payload, reason in [
        (["bad"], "policy_action_not_mapping"),
        ({"action_type": "base_velocity"}, "base_velocity_missing_numeric_linear_velocity"),
        ({"action_type": "heading_yaw"}, "heading_yaw_missing_numeric_target_yaw"),
        ({"action_type": "waypoint", "waypoint": "bad"}, "waypoint_missing_xy"),
    ]:
        normalized, rejection = lane.normalize_policy_action(
            raw_payload=raw_payload,
            observation=observation,
            source="late-test",
        )
        assert normalized["normalization_status"] == "rejected"
        assert rejection is not None
        assert rejection["reason"] == reason
    manipulation, rejection = lane.normalize_policy_action(
        raw_payload={"action_type": "manipulation_contact", "waypoint": [0.4, -0.1, 0.8]},
        observation=observation,
        source="late-test",
    )
    assert rejection is None
    assert manipulation["action_type"] == "manipulation_contact"

    torque_model, torque_mujoco = helpers._fake_model_and_mujoco(position_target=False)
    torque_controller = lane._SameSceneUnitreeRLGymController(
        model=torque_model,
        mujoco_module=torque_mujoco,
        root_qpos=0,
        root_dof=0,
        selected_root=unitree_root,
        config=config,
        policy=fake_torch.jit.load("unused"),
        policy_path=unitree_root / "policy.pt",
        config_path=unitree_root / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml",
        leg_actuator_ids=list(range(num_actions)),
        leg_qpos_addrs=[7 + index for index in range(num_actions)],
        leg_qvel_addrs=[6 + index for index in range(num_actions)],
        upper_hold_actuator_ids=[num_actions],
        upper_hold_qpos_addrs=[7 + num_actions],
        actuator_output_mode="torque",
    )
    torque_data = helpers._fake_data(torque_model)
    torque_controller.reset(torque_data)
    update = torque_controller.step(data=torque_data, step=0, command_xyz=[0.2, -0.1, 0.05])
    assert update is not None
    assert torque_controller.step(data=torque_data, step=1, command_xyz=[0.0, 0.0, 0.0]) is None

    fallback_model, fallback_mujoco = helpers._fake_model_and_mujoco(position_target=True)
    fallback_model.actuator_gainprm = object()
    controller, manifest = lane._create_same_scene_unitree_rl_gym_controller(
        model=fallback_model,
        data=helpers._fake_data(fallback_model),
        mujoco_module=fallback_mujoco,
        root_qpos=0,
        root_dof=0,
        generated_at="now",
        unitree_rl_gym_root=unitree_root,
        navigation_discovery={},
        enabled=True,
    )
    assert controller is not None
    assert manifest["actuator_output_mode"] == "torque"

    camera_mujoco = SimpleNamespace(
        MjvCamera=lambda: SimpleNamespace(lookat=[0.0, 0.0, 0.0]),
        mjtCamera=SimpleNamespace(mjCAMERA_FREE=1),
    )
    camera = lane._camera_for(
        camera_mujoco,
        "robot_follow",
        [1.0, 2.0, 0.8],
        0.5,
    )
    assert camera.elevation == -14.0
    assert lane._camera_for(
        camera_mujoco,
        "overhead",
        [1.0, 2.0, 0.8],
        0.5,
    ).distance == 4.8
    assert lane._episode_frame_steps(
        steps_per_episode=5,
        render_frame_count=0,
        video_frame_stride_steps=2,
    )[0][-1] == 4
    assert lane._video_output_fps(requested_fps=0, timestep=0.1, stride_steps=2) == 5
    assert lane._counts_by_key(
        [
            {"task_id": "a", "success": True},
            {"task_id": "a", "success": False},
            {"task_id": "b", "status": "blocked"},
        ],
        "task_id",
    )[0]["passed"] == 1
    status, success, labels, metrics = lane._score_attempt(
        run={"task_id": "stop_at_goal_and_report"},
        final_error_m=0.2,
        final_speed_mps=0.01,
        fall_count=0,
        unsafe_collision_count=0,
        object_contact_count=0,
        object_displacement_m=0.0,
        rejected_action_count=0,
        action_types=["stop"],
    )
    assert status == "completed"
    assert success is True
    assert labels == []
    assert metrics["stopped_at_goal"] is True


def test_late_mujoco_fake_run_and_cli_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    helpers = _load_mujoco_edge_helpers()
    fake_mujoco = helpers._install_fake_mujoco_for_run(monkeypatch, tmp_path)

    class FakeSameSceneController:
        def reset(self, _data: object) -> None:
            return None

        def step(self, *, data: object, step: int, command_xyz: list[float]) -> dict[str, object]:
            return {
                "schema_version": "unitree_rl_gym_same_scene_controller_update.v1",
                "step": step,
                "sim_time_s": getattr(data, "time", 0.0),
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
        lambda **kwargs: {
            "path": str(kwargs["output_path"]),
            "status": "complete",
            "size_bytes": 3,
        },
    )
    monkeypatch.setattr(
        lane,
        "_ffprobe_video",
        lambda path: {
            "path": str(path),
            "status": "complete",
            "duration_s": 0.05,
            "frame_count": 1,
        },
    )

    summary = lane.run_mujoco_g1_wam_vla_policy_endpoint_eval(
        job_dir=tmp_path / "late-ready-unitree-backend",
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

    monkeypatch.setattr(
        lane,
        "run_mujoco_g1_wam_vla_policy_endpoint_eval",
        lambda **_kwargs: {
            "status": "completed",
            "attempted_episode_count": 1,
            "successful_episode_count": 1,
            "failed_episode_count": 0,
            "blocked_episode_count": 0,
            "fixture_policy_used": False,
            "endpoint_policy_used": True,
            "rejected_policy_action_count": 0,
        },
    )
    assert lane.main(["--job-dir", str(tmp_path / "cli-job")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
