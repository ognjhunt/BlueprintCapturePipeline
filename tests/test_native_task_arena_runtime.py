from __future__ import annotations

import enum
import hashlib
import math
import sys
import types
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from blueprint_pipeline.native_task_arena_runtime import (
    NativeTaskArenaRuntimeError,
    build_native_task_arena_environment,
    camera_runtime_parameters,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_entity_asset_authoring_bundle import (
    DEFORMABLE_RUNTIME_CLASS,
    INPUT_SCHEMA_VERSION as AUTHORING_INPUT_SCHEMA,
    RIGID_RUNTIME_CLASS,
)
from blueprint_pipeline.native_task_entity_spawn_plan import (
    ADAPTER_ARENA_OBJECT,
    ADAPTER_ISAAC_DEFORMABLE_OBJECT,
    NativeTaskEntitySpawnPlanError,
    materialize_native_task_entity_spawn_plan,
)
from blueprint_pipeline.task_entity_asset_candidate import (
    SCHEMA_VERSION as ASSET_CANDIDATE_SCHEMA,
)


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
                    "sensor_id": "task_robot_contact",
                    "prim_path": "{ENV_REGEX_NS}/task_object/upper_door",
                    "filter_prim_paths_expr": ["{ENV_REGEX_NS}/Robot/gripper/.*"],
                },
                {
                    "sensor_id": "task_scene_contact",
                    "prim_path": "{ENV_REGEX_NS}/task_object/upper_door",
                    "filter_prim_paths_expr": [
                        "{ENV_REGEX_NS}/scene_collision/.*"
                    ],
                },
                {
                    "sensor_id": "robot_scene_contact",
                    "prim_path": "{ENV_REGEX_NS}/Robot/.*",
                    "filter_prim_paths_expr": [
                        "{ENV_REGEX_NS}/scene_collision/.*"
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


def _fixed_sha(character: str) -> str:
    return "sha256:" + character * 64


def _spawn_entity(
    entity_id: str,
    role: str,
    physics_type: str,
    *,
    pose: dict,
    digest: str,
    configuration: dict | None = None,
    binding_kind: str = "usd_asset",
    inserted: bool = True,
) -> dict:
    reset_kind = {
        "deformable_volume": "native_deformable_state",
        "rigid_body": "native_rigid_state",
        "articulation": "native_articulation_state",
        "static_collider": "immutable_scene_state",
        "robot_articulation": "native_robot_state",
    }[physics_type]
    return {
        "entity_id": entity_id,
        "semantic_role": role,
        "physics_type": physics_type,
        "runtime_asset": {
            "asset_id": f"asset:{entity_id}",
            "binding_kind": binding_kind,
            "source_reference": (
                f"assets/{entity_id}.usd"
                if binding_kind == "usd_asset"
                else f"runtime/{entity_id}"
            ),
            "sha256": digest,
        },
        "initial_state": {"pose_world": pose},
        "reset_method": {
            "kind": reset_kind,
            "state_id": f"reset:{entity_id}",
            "native_readback_required": True,
            "direct_state_write_after_episode_start_allowed": False,
        },
        "replacement_policy": {
            "action": "insert_runtime_asset" if inserted else "retain_registered_source"
        },
        "digests": {
            "configuration_sha256": canonical_digest(configuration or {})
        },
    }


def _spawn_candidate(
    entity: dict,
    *,
    asset_class: str,
    configuration: dict,
) -> tuple[dict, dict]:
    entity_id = entity["entity_id"]
    file_row = {
        "role": "runtime_usd",
        "path": f"{entity_id}.usd",
        "size_bytes": 100 + len(entity_id),
        "sha256": entity["runtime_asset"]["sha256"],
    }
    transform = {"authored_origin": "asset_origin", "scale_xyz": [1.0, 1.0, 1.0]}
    candidate = {
        "schema_version": ASSET_CANDIDATE_SCHEMA,
        "entity_id": entity_id,
        "asset_id": entity["runtime_asset"]["asset_id"],
        "asset_class": asset_class,
        "files": [file_row],
        "transform": transform,
        (
            "deformable_configuration"
            if asset_class == "deformable_volume"
            else "receptacle_configuration"
        ): configuration,
        "candidate_digest": "",
    }
    candidate["candidate_digest"] = canonical_digest(
        candidate, digest_field="candidate_digest"
    )
    operation = {
        "operation_kind": (
            "compose_closed_volumetric_fem_candidate"
            if asset_class == "deformable_volume"
            else "compose_open_rigid_receptacle_candidate"
        ),
        "runtime_class": (
            DEFORMABLE_RUNTIME_CLASS
            if asset_class == "deformable_volume"
            else RIGID_RUNTIME_CLASS
        ),
        "configuration": configuration,
        "candidate_authored_transform": transform,
        "initial_pose_world": entity["initial_state"]["pose_world"],
    }
    authoring_row = {
        "entity_id": entity_id,
        "semantic_role": entity["semantic_role"],
        "physics_type": entity["physics_type"],
        "asset_id": entity["runtime_asset"]["asset_id"],
        "candidate_digest": candidate["candidate_digest"],
        "candidate_record": candidate,
        "staged_files": [
            {
                "role": "runtime_usd",
                "archive_relative_path": (
                    f"candidate_assets/{entity_id}/{entity_id}.usd"
                ),
                "size_bytes": file_row["size_bytes"],
                "sha256": file_row["sha256"],
            }
        ],
        "operation": operation,
    }
    return candidate, authoring_row


def _sealed_deformable_spawn_inputs() -> tuple[dict, dict]:
    plan = _sealed_scene_plan()
    plan["scene_id"] = "deformable_fixture_scene"
    plan["task_id"] = "cloth_into_basket"
    plan["task_kind"] = "deformable_transfer"
    plan["articulation"] = {"contact_sensors": []}
    plan["reset"] = {"task_joint_positions_rad": {}}
    pose = {
        "position_world_m": [0.8, 1.8, 0.7],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    cloth_configuration = {
        "reset": {
            "reset_kind": "native_default_nodal_state",
            "write_default_nodal_state_before_episode": True,
            "zero_nodal_velocities": True,
            "free_kinematic_flag_value": 1.0,
            "native_readback_required": True,
            "direct_state_write_after_episode_start_allowed": False,
        }
    }
    basket_configuration = {
        "geometry": {"open_interior": True, "top_cap_present": False}
    }
    entities = [
        _spawn_entity(
            "cloth",
            "movable_deformable",
            "deformable_volume",
            pose=pose,
            digest=_fixed_sha("1"),
            configuration=cloth_configuration,
        ),
        _spawn_entity(
            "basket",
            "destination_receptacle",
            "static_collider",
            pose={**pose, "position_world_m": [1.0, 2.0, 0.3]},
            digest=_fixed_sha("2"),
            configuration=basket_configuration,
        ),
        _spawn_entity(
            "chair",
            "obstacle",
            "static_collider",
            pose={**pose, "position_world_m": [0.4, 2.2, 0.4]},
            digest=_fixed_sha("3"),
        ),
        _spawn_entity(
            "wall",
            "obstacle",
            "static_collider",
            pose={**pose, "position_world_m": [1.5, 2.5, 0.5]},
            digest=_fixed_sha("4"),
        ),
        _spawn_entity(
            "counter",
            "support_surface",
            "static_collider",
            pose={**pose, "position_world_m": [0.0, 0.0, 0.0]},
            digest=_fixed_sha("5"),
            binding_kind="registered_scene_geometry",
            inserted=False,
        ),
        _spawn_entity(
            "franka",
            "robot",
            "robot_articulation",
            pose=plan["robot"]["base_pose_world"],
            digest=_fixed_sha("6"),
            binding_kind="runtime_embodiment",
            inserted=False,
        ),
    ]
    by_id = {entity["entity_id"]: entity for entity in entities}
    plan["task_entities"] = entities
    plan["task_entity_role_index"] = {
        role: sorted(
            entity["entity_id"]
            for entity in entities
            if entity["semantic_role"] == role
        )
        for role in sorted({entity["semantic_role"] for entity in entities})
    }
    plan["task_entity_contract_digest"] = _fixed_sha("a")
    plan["objects"] = [plan["objects"][0]]
    object_types = {
        "cloth": "DEFORMABLE",
        "basket": "BASE",
        "chair": "BASE",
        "wall": "BASE",
    }
    for entity_id in ("cloth", "basket", "chair", "wall"):
        entity = by_id[entity_id]
        plan["objects"].append(
            {
                "name": f"{entity_id}_runtime",
                "entity_id": entity_id,
                "semantic_role": entity["semantic_role"],
                "prim_path": f"{{ENV_REGEX_NS}}/task_entities/{entity_id}_runtime",
                "object_type": object_types[entity_id],
                "usd_path": f"/provider/assets/{entity_id}.usd",
                "sha256": entity["runtime_asset"]["sha256"],
                "size_bytes": 100 + len(entity_id),
                "visible": True,
                "pose_world": entity["initial_state"]["pose_world"],
                "activate_contact_sensors": entity_id != "cloth",
            }
        )

    _, cloth_authoring = _spawn_candidate(
        by_id["cloth"],
        asset_class="deformable_volume",
        configuration=cloth_configuration,
    )
    _, basket_authoring = _spawn_candidate(
        by_id["basket"],
        asset_class="rigid_receptacle",
        configuration=basket_configuration,
    )
    manifest = {
        "schema_version": AUTHORING_INPUT_SCHEMA,
        "task_kind": "deformable_transfer",
        "task_entity_contract_digest": plan["task_entity_contract_digest"],
        "asset_entity_ids": ["basket", "cloth"],
        "entity_authoring_plans": [basket_authoring, cloth_authoring],
        "input_digest": "",
    }
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan, manifest


def _sealed_entity_articulated_plan() -> dict:
    plan = _sealed_scene_plan()
    target = _spawn_entity(
        "refrigerator",
        "articulated_fixture",
        "articulation",
        pose=plan["objects"][1]["pose_world"],
        digest=_fixed_sha("7"),
    )
    target["runtime_asset"]["source_reference"] = "assets/task.usda"
    robot = _spawn_entity(
        "franka",
        "robot",
        "robot_articulation",
        pose=plan["robot"]["base_pose_world"],
        digest=_fixed_sha("8"),
        binding_kind="runtime_embodiment",
        inserted=False,
    )
    plan["task_entities"] = [target, robot]
    plan["task_entity_role_index"] = {
        "articulated_fixture": ["refrigerator"],
        "robot": ["franka"],
    }
    plan["task_entity_contract_digest"] = _fixed_sha("b")
    task_object = plan["objects"][1]
    task_object.update(
        {
            "name": "refrigerator_runtime",
            "entity_id": "refrigerator",
            "semantic_role": "articulated_fixture",
            "prim_path": "{ENV_REGEX_NS}/task_entities/refrigerator_runtime",
            "sha256": target["runtime_asset"]["sha256"],
            "size_bytes": 123,
        }
    )
    for sensor in plan["articulation"]["contact_sensors"]:
        sensor["prim_path"] = sensor["prim_path"].replace(
            "{ENV_REGEX_NS}/task_object",
            task_object["prim_path"],
        )
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def test_builder_wires_articulation_contacts_resets_and_cameras(monkeypatch) -> None:
    _install_fake_native_runtime(monkeypatch)

    built = build_native_task_arena_environment(_sealed_scene_plan())

    assert built.env == "native-env"
    assert built.cfg.sim.dt == pytest.approx(1.0 / 120.0)
    assert built.cfg.decimation == 8
    assert built.cfg.episode_length_s == 32.0
    assert built.scene_asset_names == {
        "scene_collision": "scene_collision",
        "task_object": "task_object",
    }
    assert set(built.contact_sensor_names) == {
        "task_robot_contact",
        "task_scene_contact",
        "robot_scene_contact",
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
    reset_owner = next(
        asset for asset in arena_env.scene.assets if asset.name == "task_robot_contact"
    )
    assert reset_owner.event_name == "reset_task_object_joints"
    assert reset_owner.event_cfg.params["asset_cfg"].name == "task_object"
    assert reset_owner.event_cfg.params["position_range"] == (0.0, 0.0)
    assert _ArenaBuilder.last.args.device == "cuda:0"


def test_entity_keyed_articulation_keeps_legacy_task_object_alias_and_reset(
    monkeypatch,
) -> None:
    _install_fake_native_runtime(monkeypatch)

    built = build_native_task_arena_environment(_sealed_entity_articulated_plan())

    assert built.scene_asset_names["articulated_fixture"] == "refrigerator_runtime"
    assert built.scene_asset_names["task_object"] == "refrigerator_runtime"
    assert built.scene_asset_names_by_entity_id == {
        "refrigerator": "refrigerator_runtime"
    }
    refrigerator = next(
        asset
        for asset in _ArenaBuilder.last.arena_env.scene.assets
        if asset.name == "refrigerator_runtime"
    )
    assert refrigerator.object_type is _ObjectType.ARTICULATION
    assert refrigerator.spawn_cfg_addon["semantic_tags"] == [
        ("class", "articulated_fixture"),
        ("entity_id", "refrigerator"),
    ]
    reset_owner = next(
        asset
        for asset in _ArenaBuilder.last.arena_env.scene.assets
        if asset.name == "task_robot_contact"
    )
    assert reset_owner.event_cfg.params["asset_cfg"].name == "refrigerator_runtime"


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


@pytest.mark.parametrize("object_type", ["RIGID", "ARTICULATION"])
def test_entity_spawn_plan_preserves_legacy_arena_object_adapter(
    object_type: str,
) -> None:
    plan = _sealed_scene_plan()
    plan["objects"][1]["object_type"] = object_type
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    spawn_plan = materialize_native_task_entity_spawn_plan(scene_plan=plan)

    assert [row["adapter_kind"] for row in spawn_plan["assets"]] == [
        ADAPTER_ARENA_OBJECT,
        ADAPTER_ARENA_OBJECT,
    ]
    assert spawn_plan["role_aliases"] == {
        "scene_collision": "scene_collision",
        "task_object": "task_object",
    }
    assert spawn_plan["entity_asset_names"] == {}


def test_entity_spawn_plan_uses_unique_entity_handles_and_role_safe_aliases() -> None:
    plan, manifest = _sealed_deformable_spawn_inputs()

    spawn_plan = materialize_native_task_entity_spawn_plan(
        scene_plan=plan,
        authoring_manifest=manifest,
    )

    assert spawn_plan["entity_asset_names"] == {
        "basket": "basket_runtime",
        "chair": "chair_runtime",
        "cloth": "cloth_runtime",
        "wall": "wall_runtime",
    }
    assert len(set(spawn_plan["entity_asset_names"].values())) == 4
    assert len(set(spawn_plan["entity_prim_paths"].values())) == 4
    assert "obstacle" not in spawn_plan["role_aliases"]
    assert spawn_plan["role_aliases"]["movable_deformable"] == "cloth_runtime"
    chair = next(row for row in spawn_plan["assets"] if row.get("entity_id") == "chair")
    assert chair["semantic_tags"] == [
        ["class", "obstacle"],
        ["entity_id", "chair"],
    ]


def test_deformable_runtime_uses_injected_cfg_instead_of_arena_object_enum(
    monkeypatch,
) -> None:
    _install_fake_native_runtime(monkeypatch)
    plan, manifest = _sealed_deformable_spawn_inputs()
    captured: list[dict] = []

    def deformable_cfg_factory(spawn: Mapping[str, Any]) -> SimpleNamespace:
        captured.append(dict(spawn))
        return SimpleNamespace(config_class="DeformableObjectCfg")

    built = build_native_task_arena_environment(
        plan,
        entity_authoring_manifest=manifest,
        deformable_object_cfg_factory=deformable_cfg_factory,
    )

    assert [row["entity_id"] for row in captured] == ["cloth"]
    assert captured[0]["adapter_kind"] == ADAPTER_ISAAC_DEFORMABLE_OBJECT
    assert captured[0]["semantic_tags"] == [
        ["class", "movable_deformable"],
        ["entity_id", "cloth"],
    ]
    assert built.scene_asset_names_by_entity_id == {
        "basket": "basket_runtime",
        "chair": "chair_runtime",
        "cloth": "cloth_runtime",
        "wall": "wall_runtime",
    }
    assert built.entity_reset_recipes_by_entity_id["cloth"] == captured[0][
        "reset_recipe"
    ]
    cloth_asset = next(
        asset
        for asset in _ArenaBuilder.last.arena_env.scene.assets
        if asset.name == "cloth_runtime"
    )
    assert not isinstance(cloth_asset, _Object)
    assert cloth_asset.object_cfg.config_class == "DeformableObjectCfg"
    basket_asset = next(
        asset
        for asset in _ArenaBuilder.last.arena_env.scene.assets
        if asset.name == "basket_runtime"
    )
    assert basket_asset.object_type is _ObjectType.BASE
    assert basket_asset.spawn_cfg_addon["semantic_tags"] == [
        ("class", "destination_receptacle"),
        ("entity_id", "basket"),
    ]


def test_entity_spawn_plan_binds_candidate_operation_digest_and_pose() -> None:
    plan, manifest = _sealed_deformable_spawn_inputs()

    spawn_plan = materialize_native_task_entity_spawn_plan(
        scene_plan=plan,
        authoring_manifest=manifest,
    )

    cloth = next(row for row in spawn_plan["assets"] if row.get("entity_id") == "cloth")
    source = next(
        row for row in manifest["entity_authoring_plans"] if row["entity_id"] == "cloth"
    )
    binding = cloth["authoring_binding"]
    assert "candidate_record" not in binding
    assert binding["candidate_digest"] == source["candidate_digest"]
    assert binding["runtime_usd_sha256"] == cloth["sha256"]
    assert binding["operation"] == source["operation"]
    assert binding["operation"]["initial_pose_world"] == cloth["pose_world"]
    assert spawn_plan["authoring_input_digest"] == manifest["input_digest"]


def test_deformable_reset_recipe_freezes_write_and_readback_order() -> None:
    plan, manifest = _sealed_deformable_spawn_inputs()

    spawn_plan = materialize_native_task_entity_spawn_plan(
        scene_plan=plan,
        authoring_manifest=manifest,
    )

    cloth = next(row for row in spawn_plan["assets"] if row.get("entity_id") == "cloth")
    recipe = cloth["reset_recipe"]
    assert [step["operation"] for step in recipe["steps"]] == [
        "load_default_nodal_state",
        "zero_nodal_velocities",
        "write_nodal_state_to_sim_index",
        "write_nodal_kinematic_target_to_sim_index",
        "readback_data_nodal_state_and_kinematic_target",
    ]
    assert [step["order"] for step in recipe["steps"]] == [1, 2, 3, 4, 5]
    assert recipe["steps"][3]["free_flag_value"] == 1.0
    assert recipe["write_scope"] == "before_episode_start_only"
    assert recipe["direct_state_write_after_episode_start_allowed"] is False
    assert recipe["native_readback_required"] is True


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("missing_scene_binding", "native_task_entity_spawn_scene_object_missing:basket"),
        ("duplicate_entity", "native_task_entity_spawn_entity_duplicate:cloth"),
        ("missing_authoring", "native_task_entity_spawn_authoring_plan_missing:cloth"),
        ("unbound_authoring", "native_task_entity_spawn_authoring_plan_unbound:ghost"),
        ("missing_runtime_output", "native_task_entity_spawn_runtime_usd_missing:cloth"),
        ("digest_mismatch", "native_task_entity_spawn_asset_digest_mismatch:cloth"),
        (
            "duplicate_runtime_name",
            "native_task_entity_spawn_runtime_name_duplicate:wall",
        ),
        (
            "authoring_pose_mismatch",
            "native_task_entity_spawn_authoring_join_invalid:cloth",
        ),
        ("reset_write_after_start", "native_task_entity_spawn_reset_invalid:cloth"),
        ("robot_pose_mismatch", "native_task_entity_spawn_robot_join_invalid:franka"),
        (
            "malformed_runtime_asset",
            "native_task_entity_spawn_entity_field_invalid:cloth:runtime_asset",
        ),
    ],
)
def test_entity_spawn_plan_fails_closed_on_incomplete_or_ambiguous_joins(
    failure: str,
    expected_error: str,
) -> None:
    plan, manifest = _sealed_deformable_spawn_inputs()
    if failure == "missing_scene_binding":
        plan["objects"] = [
            row for row in plan["objects"] if row.get("entity_id") != "basket"
        ]
    elif failure == "duplicate_entity":
        cloth = next(row for row in plan["task_entities"] if row["entity_id"] == "cloth")
        plan["task_entities"].append(dict(cloth))
    elif failure == "missing_authoring":
        manifest["entity_authoring_plans"] = [
            row
            for row in manifest["entity_authoring_plans"]
            if row["entity_id"] != "cloth"
        ]
    elif failure == "unbound_authoring":
        ghost = dict(manifest["entity_authoring_plans"][0])
        ghost["entity_id"] = "ghost"
        manifest["entity_authoring_plans"].append(ghost)
    elif failure == "missing_runtime_output":
        cloth = next(
            row
            for row in manifest["entity_authoring_plans"]
            if row["entity_id"] == "cloth"
        )
        cloth["staged_files"] = []
    elif failure == "digest_mismatch":
        cloth = next(row for row in plan["objects"] if row.get("entity_id") == "cloth")
        cloth["sha256"] = _fixed_sha("f")
    elif failure == "duplicate_runtime_name":
        cloth = next(row for row in plan["objects"] if row.get("entity_id") == "cloth")
        wall = next(row for row in plan["objects"] if row.get("entity_id") == "wall")
        wall["name"] = cloth["name"]
    elif failure == "authoring_pose_mismatch":
        cloth = next(
            row
            for row in manifest["entity_authoring_plans"]
            if row["entity_id"] == "cloth"
        )
        cloth["operation"]["initial_pose_world"] = {
            "position_world_m": [9.0, 9.0, 9.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    elif failure == "reset_write_after_start":
        cloth = next(row for row in plan["task_entities"] if row["entity_id"] == "cloth")
        cloth["reset_method"][
            "direct_state_write_after_episode_start_allowed"
        ] = True
    elif failure == "robot_pose_mismatch":
        robot = next(row for row in plan["task_entities"] if row["entity_id"] == "franka")
        robot["initial_state"]["pose_world"] = {
            "position_world_m": [9.0, 9.0, 9.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    elif failure == "malformed_runtime_asset":
        cloth = next(row for row in plan["task_entities"] if row["entity_id"] == "cloth")
        cloth["runtime_asset"] = "not-a-mapping"
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")

    with pytest.raises(NativeTaskEntitySpawnPlanError) as exc_info:
        materialize_native_task_entity_spawn_plan(
            scene_plan=plan,
            authoring_manifest=manifest,
        )

    assert expected_error in exc_info.value.errors
