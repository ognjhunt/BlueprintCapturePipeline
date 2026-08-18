from __future__ import annotations

import enum
import hashlib
import math
import sys
import types
from types import SimpleNamespace

import pytest

from blueprint_pipeline.native_task_arena_runtime import (
    NativeTaskArenaRuntimeError,
    build_native_task_arena_environment,
    camera_runtime_parameters,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _camera(role: str, matrix: list[float] | None = None) -> dict:
    wrist = role == "wrist"
    return {
        "role": role,
        "policy_input": role in {"external", "wrist"},
        "review_only": role == "overview",
        "pose_frame": "robot_body" if wrist else "world",
        "parent_prim_path": (
            "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
            if wrist
            else "{ENV_REGEX_NS}"
        ),
        "optical_convention": "opencv",
        "frame_from_camera_matrix": matrix
        or [
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            2.0,
            0.0,
            0.0,
            1.0,
            3.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "intrinsics": {
            "fx": 300.0,
            "fy": 300.0,
            "cx": 159.5,
            "cy": 89.5,
            "width": 320,
            "height": 180,
        },
    }


@pytest.mark.parametrize(
    ("role", "runtime_name", "prim_path"),
    [
        ("external", "external_camera", "{ENV_REGEX_NS}/external_camera"),
        (
            "wrist",
            "wrist_camera",
            (
                "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/"
                "base_link/wrist_camera"
            ),
        ),
        ("overview", "external_camera_2", "{ENV_REGEX_NS}/external_camera_2"),
    ],
)
def test_calibrated_cameras_map_to_role_neutral_native_cfg(
    role: str, runtime_name: str, prim_path: str
) -> None:
    parameters = camera_runtime_parameters(_camera(role))

    assert parameters["runtime_name"] == runtime_name
    assert parameters["prim_path"] == prim_path
    assert parameters["offset_position_m"] == [1.0, 2.0, 3.0]
    assert parameters["offset_rotation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert parameters["isaac_offset_convention"] == "ros"
    assert parameters["focal_length_mm"] == pytest.approx(
        300.0 * 20.955 / 320.0
    )
    assert parameters["vertical_aperture_mm"] == pytest.approx(
        20.955 * 180.0 / 320.0
    )


def test_rotation_is_converted_to_xyzw_not_legacy_wxyz() -> None:
    angle = math.pi / 2.0
    cosine = math.cos(angle)
    sine = math.sin(angle)
    matrix = [
        cosine,
        -sine,
        0.0,
        0.0,
        sine,
        cosine,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]

    parameters = camera_runtime_parameters(_camera("external", matrix))

    assert parameters["offset_rotation_xyzw"] == pytest.approx(
        [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]
    )


def test_off_center_or_non_square_intrinsics_fail_before_gpu() -> None:
    camera = _camera("external")
    camera["intrinsics"]["cx"] = 160.0

    with pytest.raises(NativeTaskArenaRuntimeError) as excinfo:
        camera_runtime_parameters(camera)

    assert excinfo.value.errors == (
        "native_task_arena_camera_intrinsics_not_representable:external",
    )


class _Replaceable(SimpleNamespace):
    def replace(self, **kwargs):
        values = dict(vars(self))
        values.update(kwargs)
        return _Replaceable(**values)


class _ObjectType(enum.Enum):
    BASE = "BASE"
    RIGID = "RIGID"
    ARTICULATION = "ARTICULATION"
    SPAWNER = "SPAWNER"


class _Asset:
    def __init__(self, name: str, **_kwargs):
        self.name = name


class _Object(_Asset):
    def __init__(
        self,
        *,
        name,
        prim_path=None,
        object_type=None,
        usd_path=None,
        initial_pose=None,
        spawn_cfg_addon=None,
        **_kwargs,
    ):
        super().__init__(name)
        self.prim_path = prim_path
        self.object_type = object_type
        self.usd_path = usd_path
        self.initial_pose = initial_pose
        self.spawn_cfg_addon = spawn_cfg_addon or {}
        self.object_cfg = SimpleNamespace(init_state=_Replaceable(joint_pos={}))


class _Pose:
    def __init__(self, *, position_xyz, rotation_xyzw):
        self.position_xyz = position_xyz
        self.rotation_xyzw = rotation_xyzw


class _CameraCfg:
    def __init__(self):
        self.prim_path = ""
        self.offset = SimpleNamespace(pos=(), rot=(), convention="")
        self.spawn = SimpleNamespace(
            focal_length=0.0, horizontal_aperture=0.0, vertical_aperture=0.0
        )


class _Embodiment:
    def __init__(self, *, enable_cameras, initial_pose, initial_joint_pose):
        assert enable_cameras is True
        self.initial_pose = initial_pose
        self.initial_joint_pose = initial_joint_pose
        self.event_config = SimpleNamespace(
            init_franka_arm_pose=SimpleNamespace(params={}),
            randomize_franka_joint_state=SimpleNamespace(params={}),
        )
        self.scene_config = SimpleNamespace(
            stand=object(),
            robot=SimpleNamespace(
                init_state=_Replaceable(joint_pos={}),
                spawn=SimpleNamespace(semantic_tags=[]),
            ),
        )
        self.camera_config = SimpleNamespace(
            external_camera=_CameraCfg(),
            wrist_camera=_CameraCfg(),
            external_camera_2=_CameraCfg(),
        )

    def get_scene_cfg(self):
        return self.scene_config


class _Scene:
    def __init__(self, *, assets):
        self.assets = assets


class _ArenaEnvironment:
    def __init__(self, **kwargs):
        vars(self).update(kwargs)


class _ArenaBuilder:
    last = None

    def __init__(self, arena_env, args):
        self.arena_env = arena_env
        self.args = args
        type(self).last = self

    def make_registered_and_return_cfg(self, *, render_mode):
        assert render_mode == "rgb_array"
        cfg = SimpleNamespace(
            sim=SimpleNamespace(), seed=None, decimation=None, episode_length_s=None
        )
        self.arena_env.env_cfg_callback(cfg)
        return "native-env", cfg


def _install_fake_native_runtime(monkeypatch) -> None:
    from blueprint_pipeline import native_task_arena_preconstruction

    preconstruction = {
        "schema_version": "native_task_arena_preconstruction.v1",
        "upstream_contract": {
            "simulation_lifecycle_ownership_fix": (
                "03904ab49152d1bae929513529913b9be2e06808"
            ),
            "warp_extension_exclusion_fix": (
                "c4169b2f1c41117b67154c569668b8834519a5ee"
            ),
        },
        "expected_device": "cuda:0",
        "observed": {},
        "passed": True,
        "blockers": [],
        "postconstruction_native_view_readback_still_required": True,
        "receipt_digest": "",
    }
    preconstruction["receipt_digest"] = canonical_digest(
        preconstruction, digest_field="receipt_digest"
    )
    monkeypatch.setattr(
        native_task_arena_preconstruction,
        "prepare_native_task_arena_preconstruction",
        lambda *, expected_device: preconstruction,
    )
    modules = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.envs": types.ModuleType("isaaclab.envs"),
        "isaaclab.envs.mdp": types.ModuleType("isaaclab.envs.mdp"),
        "isaaclab.sim": types.ModuleType("isaaclab.sim"),
        "isaaclab.managers": types.ModuleType("isaaclab.managers"),
        "isaaclab.sensors": types.ModuleType("isaaclab.sensors"),
        "isaaclab_arena": types.ModuleType("isaaclab_arena"),
        "isaaclab_arena.assets": types.ModuleType("isaaclab_arena.assets"),
        "isaaclab_arena.assets.asset": types.ModuleType(
            "isaaclab_arena.assets.asset"
        ),
        "isaaclab_arena.assets.object": types.ModuleType(
            "isaaclab_arena.assets.object"
        ),
        "isaaclab_arena.assets.object_base": types.ModuleType(
            "isaaclab_arena.assets.object_base"
        ),
        "isaaclab_arena.embodiments": types.ModuleType(
            "isaaclab_arena.embodiments"
        ),
        "isaaclab_arena.embodiments.droid": types.ModuleType(
            "isaaclab_arena.embodiments.droid"
        ),
        "isaaclab_arena.embodiments.droid.droid": types.ModuleType(
            "isaaclab_arena.embodiments.droid.droid"
        ),
        "isaaclab_arena.environments": types.ModuleType(
            "isaaclab_arena.environments"
        ),
        "isaaclab_arena.environments.arena_env_builder": types.ModuleType(
            "isaaclab_arena.environments.arena_env_builder"
        ),
        "isaaclab_arena.environments.isaaclab_arena_environment": types.ModuleType(
            "isaaclab_arena.environments.isaaclab_arena_environment"
        ),
        "isaaclab_arena.scene": types.ModuleType("isaaclab_arena.scene"),
        "isaaclab_arena.scene.scene": types.ModuleType("isaaclab_arena.scene.scene"),
        "isaaclab_arena.tasks": types.ModuleType("isaaclab_arena.tasks"),
        "isaaclab_arena.tasks.no_task": types.ModuleType(
            "isaaclab_arena.tasks.no_task"
        ),
        "isaaclab_arena.utils": types.ModuleType("isaaclab_arena.utils"),
        "isaaclab_arena.utils.pose": types.ModuleType("isaaclab_arena.utils.pose"),
        "isaaclab_physx": types.ModuleType("isaaclab_physx"),
        "isaaclab_physx.physics": types.ModuleType("isaaclab_physx.physics"),
    }
    modules["isaaclab.envs.mdp"].reset_joints_by_offset = object()
    modules["isaaclab.sim"].DomeLightCfg = lambda **kwargs: SimpleNamespace(**kwargs)
    modules["isaaclab.managers"].EventTermCfg = (
        lambda **kwargs: SimpleNamespace(**kwargs)
    )
    modules["isaaclab.managers"].SceneEntityCfg = (
        lambda name: SimpleNamespace(name=name)
    )
    modules["isaaclab.sensors"].ContactSensorCfg = (
        lambda **kwargs: SimpleNamespace(**kwargs)
    )
    modules["isaaclab_arena.assets.asset"].Asset = _Asset
    modules["isaaclab_arena.assets.object"].Object = _Object
    modules["isaaclab_arena.assets.object_base"].ObjectType = _ObjectType
    modules[
        "isaaclab_arena.embodiments.droid.droid"
    ].DroidAbsoluteJointPositionEmbodiment = _Embodiment
    modules["isaaclab_arena.embodiments"].__blueprint_scoped__ = True
    modules[
        "isaaclab_arena.environments.arena_env_builder"
    ].ArenaEnvBuilder = _ArenaBuilder
    modules[
        "isaaclab_arena.environments.isaaclab_arena_environment"
    ].IsaacLabArenaEnvironment = _ArenaEnvironment
    modules["isaaclab_arena.scene.scene"].Scene = _Scene
    modules["isaaclab_arena.tasks.no_task"].NoTask = object
    modules["isaaclab_arena.utils.pose"].Pose = _Pose
    modules["isaaclab_physx.physics"].PhysxCfg = (
        lambda **kwargs: SimpleNamespace(**kwargs)
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _sealed_scene_plan() -> dict:
    cameras = [_camera(role) for role in ("external", "wrist", "overview")]
    plan = {
        "schema_version": "native_task_arena_scene_plan.v1",
        "scene_id": "fixture_scene",
        "task_id": "fixture_task",
        "task_kind": "articulated_open_close",
        "runtime_contract_digest": "sha256:" + "a" * 64,
        "scenario": {"seed": 17},
        "asset_directory": "/provider/assets",
        "objects": [
            {
                "semantic_role": "scene_collision",
                "prim_path": "{ENV_REGEX_NS}/scene_collision",
                "object_type": "BASE",
                "usd_path": "/provider/assets/collision.usd",
                "visible": False,
                "pose_world": {
                    "position_world_m": [0.0, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            {
                "semantic_role": "task_object",
                "prim_path": "{ENV_REGEX_NS}/task_object",
                "object_type": "ARTICULATION",
                "usd_path": "/provider/assets/task.usda",
                "visible": True,
                "pose_world": {
                    "position_world_m": [1.0, 2.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        ],
        "robot": {
            "robot_id": "franka_panda",
            "base_pose_world": {
                "position_world_m": [1.75, 1.99, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "joint_reset_positions_rad": {
                "panda_joint1": 0.0,
                "panda_joint2": -0.6,
            },
        },
        "cameras": cameras,
        "cadence": {
            "physics_dt_seconds": 1.0 / 120.0,
            "control_decimation": 8,
            "episode_length_seconds": 32.0,
        },
        "articulation": {
            "contact_sensors": [
                {
                    "sensor_instance_id": "task_robot_contact__moving_link",
                    "logical_sensor_id": "task_robot_contact",
                    "prim_path": "{ENV_REGEX_NS}/task_object/upper_door",
                    "filter_prim_paths_expr": [
                        "{ENV_REGEX_NS}/Robot/gripper/left_finger",
                        "{ENV_REGEX_NS}/Robot/gripper/right_finger",
                    ],
                },
                {
                    "sensor_instance_id": "task_scene_contact__moving_link",
                    "logical_sensor_id": "task_scene_contact",
                    "prim_path": "{ENV_REGEX_NS}/task_object/upper_door",
                    "filter_prim_paths_expr": [
                        "{ENV_REGEX_NS}/scene_collision/floor"
                    ],
                },
                {
                    "sensor_instance_id": "robot_scene_contact__00",
                    "logical_sensor_id": "robot_scene_contact",
                    "prim_path": "{ENV_REGEX_NS}/Robot/panda_link0",
                    "filter_prim_paths_expr": [
                        "{ENV_REGEX_NS}/scene_collision/floor"
                    ],
                },
                {
                    "sensor_instance_id": "robot_scene_contact__01",
                    "logical_sensor_id": "robot_scene_contact",
                    "prim_path": "{ENV_REGEX_NS}/Robot/panda_link1",
                    "filter_prim_paths_expr": [
                        "{ENV_REGEX_NS}/scene_collision/floor"
                    ],
                },
            ]
        },
        "reset": {
            "task_joint_positions_rad": {
                "upper_door_hinge": 0.0,
                "lower_door_hinge": 0.0,
            }
        },
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def test_builder_wires_articulation_contacts_resets_and_cameras(monkeypatch) -> None:
    _install_fake_native_runtime(monkeypatch)

    built = build_native_task_arena_environment(_sealed_scene_plan())

    assert built.env == "native-env"
    assert built.cfg.sim.dt == pytest.approx(1.0 / 120.0)
    assert built.cfg.sim.device == "cuda:0"
    assert built.cfg.decimation == 8
    assert built.cfg.episode_length_s == 32.0
    assert built.scene_asset_names == {
        "scene_collision": "scene_collision",
        "task_object": "task_object",
    }
    assert built.contact_sensor_names == {
        "robot_scene_contact": (
            "robot_scene_contact__00",
            "robot_scene_contact__01",
        ),
        "task_robot_contact": ("task_robot_contact__moving_link",),
        "task_scene_contact": ("task_scene_contact__moving_link",),
    }
    assert built.camera_scene_names == {
        "external": "external_camera",
        "wrist": "wrist_camera",
        "overview": "external_camera_2",
    }
    arena_env = _ArenaBuilder.last.arena_env
    task_object = next(
        asset for asset in arena_env.scene.assets if asset.name == "task_object"
    )
    assert task_object.object_type is _ObjectType.ARTICULATION
    assert task_object.object_cfg.init_state.joint_pos == {
        "upper_door_hinge": 0.0,
        "lower_door_hinge": 0.0,
    }
    reset_owner = task_object
    assert reset_owner.reset_event_name == "reset_task_object_state"
    assert reset_owner.reset_event_cfg.params["asset_cfg"].name == "task_object"
    assert reset_owner.reset_event_cfg.params["reset_joints"] is True
    assert _ArenaBuilder.last.args.device == "cuda:0"
    assert built.preconstruction_device_binding["passed"] is True
    assert built.native_configuration_readback["cameras"]["external"][
        "offset_position_m"
    ] == [1.0, 2.0, 3.0]


def test_builder_keeps_inactive_articulated_replacement_and_its_reset(
    monkeypatch,
) -> None:
    _install_fake_native_runtime(monkeypatch)
    plan = _sealed_scene_plan()
    plan["objects"].append(
        {
            "name": "replacement__inactive_articulation",
            "semantic_role": "replacement",
            "asset_id": "inactive_articulation",
            "task_subject": False,
            "prim_path": "{ENV_REGEX_NS}/replacement__inactive_articulation",
            "object_type": "ARTICULATION",
            "usd_path": "/provider/assets/inactive.usda",
            "visible": True,
            "pose_world": {
                "position_world_m": [2.0, 3.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "reset_state": {
                "joint_positions": {"inactive_hinge": 0.25},
            },
        }
    )
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    built = build_native_task_arena_environment(plan)

    assert built.scene_asset_names["replacement__inactive_articulation"] == (
        "replacement__inactive_articulation"
    )
    inactive = next(
        asset
        for asset in _ArenaBuilder.last.arena_env.scene.assets
        if asset.name == "replacement__inactive_articulation"
    )
    assert inactive.object_type is _ObjectType.ARTICULATION
    assert inactive.object_cfg.init_state.joint_pos == {"inactive_hinge": 0.25}
    assert inactive.spawn_cfg_addon["semantic_tags"] == [
        ("class", "inactive_replacement")
    ]
    assert inactive.reset_event_name == "reset_replacement__inactive_articulation_state"
    assert inactive.reset_event_cfg.params["asset_cfg"].name == (
        "replacement__inactive_articulation"
    )
    assert inactive.reset_event_cfg.params["reset_joints"] is True


def test_rigid_task_keeps_locked_articulation_and_separate_support_collision_channels(
    monkeypatch,
) -> None:
    _install_fake_native_runtime(monkeypatch)
    plan = _sealed_scene_plan()
    plan["task_kind"] = "rigid_pick_place"
    plan["articulation"]["contact_sensors"][1].update(
        sensor_instance_id="task_support_contact__rigid_00",
        logical_sensor_id="task_support_contact",
    )
    plan["articulation"]["contact_sensors"].insert(
        2,
        {
            "sensor_instance_id": "task_scene_collision__rigid_00",
            "logical_sensor_id": "task_scene_collision",
            "prim_path": "{ENV_REGEX_NS}/task_object/upper_door",
            "filter_prim_paths_expr": ["{ENV_REGEX_NS}/scene_collision/wall"],
        },
    )
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    built = build_native_task_arena_environment(plan)

    task_object = next(
        asset for asset in _ArenaBuilder.last.arena_env.scene.assets
        if asset.name == "task_object"
    )
    assert task_object.object_type is _ObjectType.ARTICULATION
    assert task_object.object_cfg.init_state.joint_pos == {
        "upper_door_hinge": 0.0,
        "lower_door_hinge": 0.0,
    }
    assert built.contact_sensor_names["task_support_contact"] == (
        "task_support_contact__rigid_00",
    )
    assert built.contact_sensor_names["task_scene_collision"] == (
        "task_scene_collision__rigid_00",
    )


def test_many_to_many_contact_patterns_fail_before_native_build(monkeypatch) -> None:
    _install_fake_native_runtime(monkeypatch)
    plan = _sealed_scene_plan()
    plan["articulation"]["contact_sensors"][0]["filter_prim_paths_expr"] = [
        "{ENV_REGEX_NS}/Robot/gripper/.*"
    ]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    with pytest.raises(NativeTaskArenaRuntimeError) as excinfo:
        build_native_task_arena_environment(plan)

    assert excinfo.value.errors == (
        "native_task_arena_contact_sensor_contract_invalid:0",
    )


def test_portable_packet_assets_require_root_and_are_reverified(
    monkeypatch, tmp_path
) -> None:
    _install_fake_native_runtime(monkeypatch)
    root = tmp_path / "packet"
    assets = root / "assets"
    assets.mkdir(parents=True)
    collision = assets / "collision.usd"
    task = assets / "task.usda"
    collision.write_bytes(b"collision")
    task.write_bytes(b"task")
    plan = _sealed_scene_plan()
    plan["asset_directory"] = "assets"
    for row, path in zip(plan["objects"], (collision, task), strict=True):
        row["usd_path"] = f"assets/{path.name}"
        row["size_bytes"] = path.stat().st_size
        row["sha256"] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    with pytest.raises(
        NativeTaskArenaRuntimeError, match="bundle_root_required"
    ):
        build_native_task_arena_environment(plan)

    built = build_native_task_arena_environment(plan, bundle_root=root)

    assert built.plan == plan
    staged = {
        asset.name: asset.usd_path
        for asset in _ArenaBuilder.last.arena_env.scene.assets
        if isinstance(asset, _Object) and asset.usd_path is not None
    }
    assert staged == {
        "scene_collision": str(collision),
        "task_object": str(task),
    }

    task.write_bytes(b"tampered")
    with pytest.raises(
        NativeTaskArenaRuntimeError, match="asset_identity_mismatch:task_object"
    ):
        build_native_task_arena_environment(plan, bundle_root=root)


def test_the_runtime_admits_every_sensor_the_scene_plan_can_emit() -> None:
    """Both halves of the contact-sensor vocabulary must agree.

    The scene plan writes ``logical_sensor_id`` and the runtime admits it.
    Nothing local couples them: the plan validates, the packet digests, the
    bundle builds, the terminal-contract rehearsal passes, and the allocator
    admits the launch. The divergence surfaces only inside Isaac, on a GPU
    that has already been rented.

    That is how ``robot_task_forbidden_collision`` was found -- the signal for
    non-fingertip robot links striking the task object -- on the second paid
    Arena attempt, at sensor index 2 of 31.
    """

    import ast
    from pathlib import Path

    from blueprint_pipeline.native_task_arena_runtime import (
        LOGICAL_CONTACT_SENSOR_IDS,
    )

    plan_source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "blueprint_pipeline"
        / "native_task_arena_scene_plan.py"
    ).read_text(encoding="utf-8")

    emitted: set[str] = set()
    for node in ast.walk(ast.parse(plan_source)):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "logical_sensor_id"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                emitted.add(value.value)

    # the scan itself must not silently find nothing
    assert len(emitted) >= 6, emitted

    unadmitted = sorted(emitted - LOGICAL_CONTACT_SENSOR_IDS)
    assert unadmitted == [], (
        "the scene plan emits contact sensors the runtime refuses, which is "
        f"only discoverable on a rented GPU: {unadmitted}"
    )

    # and the runtime must not carry an id no producer can emit
    assert sorted(LOGICAL_CONTACT_SENSOR_IDS - emitted) == []
