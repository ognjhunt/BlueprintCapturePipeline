"""Drive the scene worker's whole path against a stubbed Arena.

Every failure this worker has produced so far cost a launch: a missing module,
an asset under a different name, a constructor missing two required arguments.
None of them were physics. They were plumbing, and plumbing is checkable on a
laptop if something stands in for Arena.

So this installs fake ``isaaclab`` and ``isaaclab_arena`` modules and runs
``main`` end to end. The episode adapter, the control episode and the task
sampler are all the **real** modules - only the simulator is fake.

What this cannot do is decide whether the door opens. A stub has no contact,
so the fake articulation moves its hinge on a schedule and the control outcome
here means nothing about the task. Every assertion below is about plumbing:
which phases were reached, what the adapter was handed, which asset resolved.
Task success is decided on hardware and nowhere else.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from enum import Enum
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = REPO_ROOT / "scripts/run_adp009d_articulated_scene_worker.py"
DOF_NAMES = ["upper_door_hinge", "lower_door_hinge"]


class _FakeObjectType(Enum):
    BASE = "BASE"
    ARTICULATION = "ARTICULATION"
    SPAWNER = "SPAWNER"


class _FakeObject:
    def __init__(self, *, name="", object_type=None, usd_path="", initial_pose=None,
                 spawn_cfg_addon=None, prim_path="", **_extra):
        self.name = name
        self.object_type = object_type
        self.usd_path = usd_path
        self.initial_pose = initial_pose
        self.spawn_cfg_addon = spawn_cfg_addon or {}
        self.prim_path = prim_path


class _FakePose:
    def __init__(self, position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0, 0, 0, 1)):
        self.position_xyz = tuple(position_xyz)
        self.rotation_xyzw = tuple(rotation_xyzw)


class _FakeArticulationData:
    """A hinge that moves on a schedule, because a stub has no contact."""

    def __init__(self, joint_names=None, body_names=None):
        import torch

        self.joint_names = list(joint_names if joint_names is not None else DOF_NAMES)
        self.body_names = list(
            body_names
            if body_names is not None
            # The adapter resolves the wrist from END_EFFECTOR_BODY_CANDIDATES
            # and the fingers from FINGER_BODIES; a fake that omits them tests
            # the fake, not the worker.
            else ["left_inner_finger", "right_inner_finger", "panda_hand"]
        )
        self._torch = torch
        self.joint_pos = torch.zeros((1, len(self.joint_names)))
        self.joint_vel = torch.zeros((1, len(self.joint_names)))
        self.body_pose_w = torch.zeros((1, max(len(self.body_names), 3), 7))
        self._gripper_command = 0.0
        self._refresh_fingers()

    def _refresh_fingers(self):
        # Wide at 0.0, closed at 1.0 - the DROID convention, so the probe has
        # something unambiguous to measure.
        gap = 0.08 - 0.07 * float(self._gripper_command)
        self.body_pose_w[0, 0, :3] = self._torch.tensor([0.0, -gap / 2.0, 0.0])
        self.body_pose_w[0, 1, :3] = self._torch.tensor([0.0, gap / 2.0, 0.0])


class _FakeScene(dict):
    pass


class _FakeActionManager:
    total_action_dim = 8


class _FakeUnwrapped:
    def __init__(self, scene):
        self.scene = scene
        self.action_manager = _FakeActionManager()
        self.device = "cpu"
        self.num_envs = 1


class _FakeEnv:
    def __init__(self, scene):
        self.unwrapped = _FakeUnwrapped(scene)
        self.steps = 0
        self.resets = 0
        self._articulation = scene["task_object"]
        self._robot = scene["robot"]

    def reset(self, seed=None, **_kw):
        self.resets += 1
        self._articulation.data.joint_pos[:] = 0.0
        self._articulation.data.joint_vel[:] = 0.0
        return {}, {}

    def step(self, action):
        self.steps += 1
        try:
            self._robot.data._gripper_command = float(action[0, 7])
            self._robot.data._refresh_fingers()
        except (IndexError, TypeError):
            pass
        return {}, 0.0, False, False, {}

    def close(self):
        return None


# A 7-axis arm with two fingers, which is what the gripper probe indexes into.
ROBOT_JOINT_NAMES = [f"panda_joint{index}" for index in range(1, 8)] + [
    "panda_finger_joint1",
    "panda_finger_joint2",
]


class _FakeContactSensorData:
    def __init__(self):
        import torch

        # Resting contact, below the force threshold: present but not a grasp.
        self.net_forces_w = torch.zeros((1, 3, 3))
        self.force_matrix_w = torch.zeros((1, 3, 1, 3))


class _FakeContactSensor:
    def __init__(self):
        self.data = _FakeContactSensorData()
        # A ContactSensor publishes the bodies its regex matched; the worker
        # must take its row indices from here and nowhere else.
        self.body_names = ["panda_link5", "left_inner_finger", "right_inner_finger"]


class _FakeCameraData:
    def __init__(self):
        import torch

        self.output = {"rgb": torch.zeros((1, 180, 320, 3), dtype=torch.uint8)}
        self.pos_w = torch.tensor([[1.2, 0.9, 1.4]])
        # An all-zero quaternion is not a rotation, and the adapter is right to
        # reject one; identity keeps the fake honest without faking a pose.
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.quat_w_world = identity.clone()
        self.quat_w_ros = identity.clone()
        self.quat_w_opengl = identity.clone()
        self.intrinsic_matrices = torch.tensor(
            [[[200.0, 0.0, 160.0], [0.0, 200.0, 90.0], [0.0, 0.0, 1.0]]]
        )


class _FakeCameraSpawn:
    # Arena's own camera cfg carries this; the adapter reads it for the
    # calibration record and is right to refuse a camera without one.
    clipping_range = (0.01, 1000.0)


class _FakeCameraRuntimeCfg:
    def __init__(self):
        self.spawn = _FakeCameraSpawn()


class _FakeCamera:
    def __init__(self):
        self.data = _FakeCameraData()
        self.cfg = _FakeCameraRuntimeCfg()


class _FakeArticulation:
    def __init__(self, joint_names=None, body_names=None):
        self.data = _FakeArticulationData(joint_names, body_names)


class _FakeArenaEnvironment:
    """Signature mirrors the rigid lane's call site, not my first guess.

    The first version accepted embodiments=[...] and env_config_modifier=,
    which are not Arena's API - so the stub passed while hardware raised
    TypeError. test_arena_api_parity pins the keywords independently of this
    file precisely because a fake cannot be trusted to police itself.
    """

    def __init__(self, *, name=None, scene=None, embodiment=None, task=None,
                 env_cfg_callback=None):
        self.name = name
        self.scene_spec = scene
        self.embodiments = [embodiment] if embodiment is not None else []
        if env_cfg_callback is not None:
            env_cfg_callback(_FakeEnvCfg())
        runtime_scene = _FakeScene()
        for sensor_name in (
            "robot_contact_sensor",
            "task_object_contact_sensor",
            "scene_collision_contact_sensor",
        ):
            runtime_scene[sensor_name] = _FakeContactSensor()
        runtime_scene["task_object"] = _FakeArticulation(
            body_names=["cabinet", "upper_door", "lower_door"]
        )
        runtime_scene["robot"] = _FakeArticulation(joint_names=ROBOT_JOINT_NAMES)
        # The adapter binds these three by name via EVALUATION_CAMERA_BINDING.
        for camera_name in ("external_camera", "wrist_camera", "external_camera_2"):
            runtime_scene[camera_name] = _FakeCamera()
        for embodiment in self.embodiments:
            runtime_scene[getattr(embodiment, "name", "robot")] = runtime_scene["robot"]
        self._env = _FakeEnv(runtime_scene)

    def get_env(self):  # pragma: no cover - not Arena's API, kept out of use
        raise AssertionError("Arena builds through ArenaEnvBuilder, not get_env")


class _FakeArenaEnvBuilder:
    def __init__(self, arena_env, builder_args):
        self._arena_env = arena_env
        self.builder_args = builder_args

    def make_registered_and_return_cfg(self, render_mode=None):
        return self._arena_env._env, _FakeEnvCfg()


class _FakeSimCfg:
    dt = 1.0 / 60.0
    render_interval = 1
    physics = None


class _FakeSceneCfg:
    """Where declared sensors land; the worker sets attributes on this."""


class _FakeEnvCfg:
    def __init__(self):
        self.sim = _FakeSimCfg()
        self.scene = _FakeSceneCfg()
        self.seed = 0
        self.decimation = 1
        self.episode_length_s = 0.0


class _FakeCameraCfg:
    """Arena hands these out on the embodiment; the worker configures them."""

    def __init__(self):
        self.data_types = []
        self.colorize_semantic_segmentation = True
        self.update_period = 1.0
        self.update_latest_camera_pose = False
        self.width = 0
        self.height = 0


class _FakeCameraConfig:
    def __init__(self):
        self.external_camera = _FakeCameraCfg()
        self.wrist_camera = _FakeCameraCfg()
        self.external_camera_2 = _FakeCameraCfg()


class _FakeEmbodiment:
    name = "robot"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.camera_config = _FakeCameraConfig()


class _FakeAppLauncher:
    def __init__(self, _args):
        self.app = types.SimpleNamespace(close=lambda: None, update=lambda: None)

    @staticmethod
    def add_app_launcher_args(parser):
        return parser


def _module(name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture()
def stubbed_arena(monkeypatch):
    """Install a fake Arena surface for the duration of one test."""

    modules = {
        "isaaclab": _module("isaaclab"),
        "isaaclab.app": _module("isaaclab.app", AppLauncher=_FakeAppLauncher),
        "isaaclab.sensors": _module(
            "isaaclab.sensors",
            # Declared before the env is built; the worker cannot attach a
            # sensor to a constructed scene, so the fake has to exist at cfg
            # time too or the contact path is never exercised here.
            ContactSensorCfg=lambda **kw: types.SimpleNamespace(**kw),
        ),
        "isaaclab.sim": _module(
            "isaaclab.sim",
            DomeLightCfg=lambda **kw: types.SimpleNamespace(**kw),
        ),
        "isaaclab_physx": _module("isaaclab_physx"),
        "isaaclab_physx.physics": _module(
            "isaaclab_physx.physics",
            PhysxCfg=lambda **kw: types.SimpleNamespace(**kw),
        ),
        "isaaclab_arena": _module("isaaclab_arena"),
        "isaaclab_arena.assets": _module("isaaclab_arena.assets"),
        "isaaclab_arena.assets.object": _module(
            "isaaclab_arena.assets.object", Object=_FakeObject
        ),
        "isaaclab_arena.assets.asset_registry": _module(
            "isaaclab_arena.assets.asset_registry",
            # Arena's own scenes get their light from here rather than
            # hand-rolling an Object subclass, so the fake mirrors that.
            AssetRegistry=lambda: types.SimpleNamespace(
                get_asset_by_name=lambda name: (
                    lambda **kw: _FakeObject(name=name, object_type=_FakeObjectType.SPAWNER)
                )
            ),
        ),
        "isaaclab_arena.assets.object_base": _module(
            "isaaclab_arena.assets.object_base", ObjectType=_FakeObjectType
        ),
        "isaaclab_arena.embodiments": _module("isaaclab_arena.embodiments"),
        "isaaclab_arena.embodiments.droid": _module("isaaclab_arena.embodiments.droid"),
        "isaaclab_arena.embodiments.droid.droid": _module(
            "isaaclab_arena.embodiments.droid.droid",
            DroidAbsoluteJointPositionEmbodiment=_FakeEmbodiment,
        ),
        "isaaclab_arena.environments": _module("isaaclab_arena.environments"),
        "isaaclab_arena.environments.isaaclab_arena_environment": _module(
            "isaaclab_arena.environments.isaaclab_arena_environment",
            IsaacLabArenaEnvironment=_FakeArenaEnvironment,
        ),
        "isaaclab_arena.environments.arena_env_builder": _module(
            "isaaclab_arena.environments.arena_env_builder",
            ArenaEnvBuilder=_FakeArenaEnvBuilder,
        ),
        "isaaclab_arena.scene": _module("isaaclab_arena.scene"),
        "isaaclab_arena.scene.scene": _module(
            "isaaclab_arena.scene.scene", Scene=lambda assets=(): list(assets)
        ),
        "isaaclab_arena.tasks": _module("isaaclab_arena.tasks"),
        "isaaclab_arena.tasks.no_task": _module(
            "isaaclab_arena.tasks.no_task", NoTask=lambda: object()
        ),
        "isaaclab_arena.utils": _module("isaaclab_arena.utils"),
        "isaaclab_arena.utils.pose": _module("isaaclab_arena.utils.pose", Pose=_FakePose),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return modules


def _load_worker():
    spec = importlib.util.spec_from_file_location("scene_worker_e2e", WORKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_runtime(tmp_path: Path) -> Path:
    """A runtime directory shaped like the bundle, with renamed assets."""

    runtime = tmp_path / "runtime"
    (runtime / "assets").mkdir(parents=True)
    # The worker imports these flat out of the runtime directory, exactly as
    # the bundle stages them. Staging with the real tool means this test also
    # covers the case where a module the worker needs was never carried.
    from blueprint_pipeline.flat_module_closure import stage_flat_module_closure

    stage_flat_module_closure(
        package_root=REPO_ROOT / "src/blueprint_pipeline",
        entry_modules=[
            "articulated_task_sample",
            "adp009d_isaac_episode_adapter",
            "adp009d_control_episode",
            "gripper_convention_probe",
            "runtime_asset_resolution",
        ],
        destination=runtime,
        verify_import=False,
    )
    for name in ("approved_can.usda", "sage_collision.usd"):
        (runtime / "assets" / name).write_text("#usda 1.0\n", encoding="utf-8")

    spec = {
        "schema_version": "adp009d_articulated_scene_spec.v1",
        "seed": 20260810,
        "episode_length_s": 12.0,
        "gripper_open_command": 1.0,
        "support_link_body": "cabinet",
        "robot_base": {
            "position_xyz": [1.75, 1.99, 0.0],
            "rotation_xyzw": [0, 0, 0, 1],
            "reset_joints": [0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.785],
        },
        "composition": {
            "schema_version": "articulated_runtime_composition.v1",
            "task_kind": "articulated_open_close",
            "objects": [
                {
                    "name": "scene_collision",
                    "semantic_role": "scene_collision",
                    "object_type": "BASE",
                    "usd_filename": "840796_collision_without_refrigerator.usda",
                    "usd_filename_aliases": ["sage_collision.usd"],
                    "visible": False,
                    "initial_position_world_m": [0.0, 0.0, 0.0],
                },
                {
                    "name": "task_object",
                    "semantic_role": "task_object",
                    "object_type": "ARTICULATION",
                    "usd_filename": "simready_refrigerator_840796_handle.usda",
                    "usd_filename_aliases": ["approved_can.usda"],
                    "visible": True,
                    "initial_position_world_m": [1.9742142, 1.4792181, 0.0],
                },
            ],
            "task_sample_binding": {
                "joint_ids": DOF_NAMES,
                "joint_prim_paths": {
                    name: f"/Asset/joints/{name}" for name in DOF_NAMES
                },
                "joint_roles": {
                    "upper_door_hinge": "task_joint",
                    "lower_door_hinge": "locked_joint",
                },
            },
        },
        "task_spec": {
            "schema_version": "adp_task_spec.v1",
            "task_kind": "articulated_open_close",
            "target_joint_id": "upper_door_hinge",
            "target_success_interval_rad": [0.7853981633974483, 0.9599310885968813],
            "joint_reset_positions_rad": {name: 0.0 for name in DOF_NAMES},
            "joint_hard_limits_rad": {name: [0.0, 1.5707963267948966] for name in DOF_NAMES},
            "reset_tolerance_rad": 0.005,
            "movement_epsilon_rad": 0.01,
            "non_task_joint_motion_tolerance_rad": 0.01,
            "settle_window_samples": 40,
            "maximum_settled_target_speed_rad_s": 0.05,
            "maximum_action_steps": 40,
        },
        "control_plan": {
            "schema_version": "adp_task_control_plan.v1",
            "cell_id": "stub_cell",
            "plan_digest": "sha256:" + "0" * 64,
            "planner_receipt_digest": "sha256:" + "0" * 64,
            "scripted_positive_actions": [
                {"phase_id": "approach", "isaac_action": [0.0] * 7 + [1.0]}
                for _ in range(4)
            ],
        },
    }
    (runtime / "adp009d_articulated_scene_spec.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8"
    )
    return runtime


def test_worker_reaches_the_adapter_with_measured_gripper_widths(
    stubbed_arena, tmp_path
):
    """The path from spec to wired adapter runs without a simulator.

    The adapter requires gripper_closed_width_m and gripper_open_width_m and
    the worker did not pass them - a TypeError that only shows up after Isaac
    boots and Arena provisions, which is roughly six minutes and a GPU hour
    rate to discover.
    """

    runtime = _write_runtime(tmp_path)
    output = tmp_path / "out" / "result.json"
    worker = _load_worker()

    worker.main(
        [
            "--runtime-dir", str(runtime),
            "--output-dir", str(output.parent),
            "--spec", str(runtime / "adp009d_articulated_scene_spec.json"),
            "--output", str(output),
        ]
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    # adapter_wired is the end of the plumbing this stub can speak to. The
    # control plan's own contract is validated by the control-episode suite
    # against real plans; a synthetic one here would only test the synthetic.
    reached = result["phase_reached"]
    assert reached == "adapter_wired", result.get("blockers")

    probe = result.get("gripper_convention_probe") or {}
    assert probe, result.get("blockers")
    assert probe.get("closed_command") == 1.0
    assert probe.get("gripper_open_width_m") == pytest.approx(0.08)


def test_worker_resolves_both_assets_through_their_bundle_aliases(
    stubbed_arena, tmp_path
):
    runtime = _write_runtime(tmp_path)
    output = tmp_path / "out" / "result.json"

    _load_worker().main(
        [
            "--runtime-dir", str(runtime),
            "--output-dir", str(output.parent),
            "--spec", str(runtime / "adp009d_articulated_scene_spec.json"),
            "--output", str(output),
        ]
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    by_role = {row["role"]: row for row in result.get("asset_resolution") or []}
    assert by_role["scene_collision"]["matched_on"] == "alias"
    assert by_role["task_object"]["matched_on"] == "alias"


def test_a_joint_absent_from_the_runtime_names_both_sides(stubbed_arena, tmp_path):
    """Naming observed and expected turns a launch into a one-line fix."""

    runtime = _write_runtime(tmp_path)
    spec_path = runtime / "adp009d_articulated_scene_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    binding = spec["composition"]["task_sample_binding"]
    binding["joint_ids"] = ["a_joint_the_runtime_does_not_have"]
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "out" / "result.json"

    _load_worker().main(
        [
            "--runtime-dir", str(runtime),
            "--output-dir", str(output.parent),
            "--spec", str(spec_path),
            "--output", str(output),
        ]
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    joined = ";".join(result.get("blockers") or [])
    assert "articulated_scene_joints_absent" in joined
    assert "a_joint_the_runtime_does_not_have" in joined
    assert "upper_door_hinge" in joined  # what the runtime does offer


def test_the_lane_result_filename_is_always_written(stubbed_arena, tmp_path):
    """The collector looks for its own filename, not for --output."""

    runtime = _write_runtime(tmp_path)
    output = tmp_path / "out" / "some_other_name.json"

    _load_worker().main(
        [
            "--runtime-dir", str(runtime),
            "--output-dir", str(output.parent),
            "--spec", str(runtime / "adp009d_articulated_scene_spec.json"),
            "--output", str(output),
        ]
    )

    assert (output.parent / "adp009d_native_microcheck.json").is_file()


def test_worker_finds_its_payload_modules_in_the_bundles_native_dir(tmp_path):
    """The bundle splits the runtime module from the modules it imports.

    The runtime module lands at provider_runtime/adp009d_isaac_runtime.py while
    extra natives land in provider_runtime/native/. A plain flat import misses
    by exactly one directory, and the repository fallback points at a src tree
    that does not exist on a provider - so the worker died at its own import
    block after Arena had finished installing.
    """

    bundle = tmp_path / "provider_runtime"
    (bundle / "native").mkdir(parents=True)
    from blueprint_pipeline.flat_module_closure import stage_flat_module_closure

    stage_flat_module_closure(
        package_root=REPO_ROOT / "src/blueprint_pipeline",
        entry_modules=["runtime_asset_resolution", "gripper_convention_probe"],
        destination=bundle / "native",
        verify_import=False,
    )
    worker_copy = bundle / "adp009d_isaac_runtime.py"
    worker_copy.write_text(WORKER_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    import subprocess

    probe = subprocess.run(
        [sys.executable, "-c", f"import runpy; runpy.run_path({str(worker_copy)!r})"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(bundle),
    )

    combined = (probe.stderr or "") + (probe.stdout or "")
    assert "No module named 'runtime_asset_resolution'" not in combined, combined[-600:]
    assert "No module named 'gripper_convention_probe'" not in combined, combined[-600:]


def test_worker_composes_from_the_real_bundle_layout(stubbed_arena, tmp_path):
    """Spec in provider_runtime/native/, assets in provider_runtime/assets/.

    The other composition tests use a flat runtime directory, which is not what
    a bundle looks like - so they could not have caught rt13, where resolution
    rooted at the spec's own directory never climbed to the sibling assets dir.
    This one is shaped like the real thing.
    """

    bundle = tmp_path / "provider_runtime"
    native = bundle / "native"
    assets = bundle / "assets"
    native.mkdir(parents=True)
    assets.mkdir()
    for name in ("approved_can.usda", "sage_collision.usd"):
        (assets / name).write_text("#usda 1.0\n", encoding="utf-8")

    # _write_runtime builds the spec and stages the payload modules; take its
    # spec and move it into native/ where the bundle actually puts it.
    staged = _write_runtime(tmp_path)
    for entry in staged.iterdir():
        if entry.is_file():
            entry.replace(native / entry.name)

    output = tmp_path / "out" / "result.json"
    _load_worker().main(
        [
            "--runtime-dir", str(native),
            "--output-dir", str(output.parent),
            "--spec", str(native / "adp009d_articulated_scene_spec.json"),
            "--output", str(output),
        ]
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    by_role = {row["role"]: row for row in result.get("asset_resolution") or []}
    assert set(by_role) == {"scene_collision", "task_object"}, result.get("blockers")
    for row in by_role.values():
        assert Path(row["resolved_path"]).parent == assets
    assert result["phase_reached"] == "adapter_wired", result.get("blockers")


def test_the_app_launcher_is_told_to_enable_cameras(stubbed_arena, tmp_path):
    """Configuring a camera is not the same as enabling rendering.

    rt15 configured all three cameras, built the embodiment, and then Isaac
    refused: "A camera was spawned without the --enable_cameras flag". The
    launcher args are parsed from an empty list - the worker's own command line
    is not consulted for them - so headless and enable_cameras have to be set
    explicitly, and only headless was.
    """

    seen = {}

    class _RecordingLauncher(_FakeAppLauncher):
        def __init__(self, args):
            seen["headless"] = getattr(args, "headless", None)
            seen["enable_cameras"] = getattr(args, "enable_cameras", None)
            super().__init__(args)

    stubbed_arena["isaaclab.app"].AppLauncher = _RecordingLauncher

    runtime = _write_runtime(tmp_path)
    output = tmp_path / "out" / "result.json"
    _load_worker().main(
        [
            "--runtime-dir", str(runtime),
            "--output-dir", str(output.parent),
            "--spec", str(runtime / "adp009d_articulated_scene_spec.json"),
            "--output", str(output),
        ]
    )

    assert seen["headless"] is True
    assert seen["enable_cameras"] is True
