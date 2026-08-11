from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline.native_deformable_task_arena_readback import (
    CLOCK_SCHEMA_VERSION,
    CONTACT_CAPABILITY_BLOCKER,
    DESTINATION_SCHEMA_VERSION,
    DIAGNOSTIC_SCHEMA_VERSION,
    INTEGRITY_AUDIT_SCHEMA_VERSION,
    POST_RESET_SYNC_SCHEMA_VERSION,
    NativeDeformableTaskArenaReadback,
    NativeDeformableTaskArenaReadbackError,
)
from blueprint_pipeline.native_task_arena_runtime import NativeTaskArenaEnvironment


class _RootView:
    def __init__(self, positions, velocities, targets) -> None:
        self.positions = np.asarray(positions, dtype=np.float64).copy()
        self.velocities = np.asarray(velocities, dtype=np.float64).copy()
        self.targets = np.asarray(targets, dtype=np.float64).copy()

    def get_sim_nodal_positions(self):
        return self.positions.copy()

    def get_sim_nodal_velocities(self):
        return self.velocities.copy()

    def get_sim_kinematic_targets(self):
        return self.targets.copy()


class _Deformable:
    def __init__(self) -> None:
        positions = np.asarray(
            [[[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.0, 0.1, 0.5]]],
            dtype=np.float64,
        )
        velocities = np.full((1, 3, 3), 4.0, dtype=np.float64)
        state = np.concatenate((positions, velocities), axis=2)
        target = np.concatenate((positions, np.zeros((1, 3, 1))), axis=2)
        self.data = SimpleNamespace(
            default_nodal_state_w=state.copy(),
            nodal_state_w=state.copy(),
            nodal_pos_w=positions.copy(),
            nodal_vel_w=velocities.copy(),
            nodal_kinematic_target=target.copy(),
            sim_element_deform_gradient_w=np.asarray([[np.eye(3), np.eye(3)]], dtype=np.float64),
        )
        self.root_view = _RootView(positions, velocities, target)
        self.pending_state = None
        self.pending_target = None
        self.state_writes = 0
        self.target_writes = 0

    def write_nodal_state_to_sim_index(self, state, *, env_ids):
        assert env_ids == [0]
        self.state_writes += 1
        self.pending_state = np.asarray(state, dtype=np.float64).copy()
        # Isaac Lab updates these cached tensors before making the PhysX setter.
        # A test must therefore use root_view, not these convenient cache fields,
        # to prove the native write actually landed.
        self.data.nodal_state_w = self.pending_state.copy()
        self.data.nodal_pos_w = self.pending_state[..., :3].copy()
        self.data.nodal_vel_w = self.pending_state[..., 3:].copy()

    def write_nodal_kinematic_target_to_sim_index(self, target, *, env_ids):
        assert env_ids == [0]
        self.target_writes += 1
        self.pending_target = np.asarray(target, dtype=np.float64).copy()
        self.data.nodal_kinematic_target = self.pending_target.copy()

    def commit_pending_writes_to_physx(self) -> None:
        assert self.pending_state is not None
        assert self.pending_target is not None
        self.root_view.positions = self.pending_state[..., :3].copy()
        self.root_view.velocities = self.pending_state[..., 3:].copy()
        self.root_view.targets = self.pending_target.copy()


def _recipe() -> dict:
    operations = [
        "load_default_nodal_state",
        "zero_nodal_velocities",
        "write_nodal_state_to_sim_index",
        "write_nodal_kinematic_target_to_sim_index",
        "readback_physx_root_view_state_and_kinematic_target",
    ]
    return {
        "reset_kind": "native_deformable_state",
        "state_id": "cloth-reset",
        "write_scope": "before_episode_start_only",
        "direct_state_write_after_episode_start_allowed": False,
        "native_readback_required": True,
        "steps": [
            {
                "order": index,
                "operation": operation,
                **({"free_flag_value": 1.0} if index == 4 else {}),
            }
            for index, operation in enumerate(operations, start=1)
        ],
    }


def _built() -> NativeTaskArenaEnvironment:
    cloth = _Deformable()
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_names=["left_finger", "right_finger"],
            body_pose_w=np.asarray(
                [
                    [
                        [0.0, -0.02, 0.6, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.02, 0.6, 0.0, 0.0, 0.0, 1.0],
                    ]
                ],
                dtype=np.float64,
            ),
        )
    )
    task_entities = [
        {
            "entity_id": "cloth",
            "semantic_role": "movable_deformable",
            "physics_type": "deformable_volume",
        },
        {
            "entity_id": "basket",
            "semantic_role": "destination_receptacle",
            "physics_type": "static_collider",
        },
        {
            "entity_id": "table",
            "semantic_role": "support_surface",
            "physics_type": "static_collider",
        },
        {
            "entity_id": "wall",
            "semantic_role": "obstacle",
            "physics_type": "static_collider",
        },
        {
            "entity_id": "franka",
            "semantic_role": "robot",
            "physics_type": "robot_articulation",
        },
    ]
    plan = {
        "task_kind": "deformable_transfer",
        "task_spec": {
            "deformable_entity_id": "cloth",
            "destination_entity_id": "basket",
            "robot_entity_id": "franka",
        },
        "task_entities": task_entities,
        "task_entity_role_index": {"robot": ["franka"]},
        "robot": {
            "grasp_frame": {
                "kind": "body_midpoint",
                "body_names": ["left_finger", "right_finger"],
            }
        },
        "cadence": {"control_frequency_hz": 20.0},
    }
    return NativeTaskArenaEnvironment(
        env=SimpleNamespace(
            unwrapped=SimpleNamespace(
                scene={
                    "cloth_runtime": cloth,
                    "basket_runtime": object(),
                    "robot": robot,
                }
            )
        ),
        cfg=None,
        plan=plan,
        scene_asset_names={
            "movable_deformable": "cloth_runtime",
            "destination_receptacle": "basket_runtime",
        },
        contact_sensor_names={},
        camera_scene_names={},
        scene_asset_names_by_entity_id={
            "cloth": "cloth_runtime",
            "basket": "basket_runtime",
        },
        scene_asset_prim_paths_by_entity_id={
            "cloth": "/World/cloth",
            "basket": "/World/basket",
        },
        entity_reset_recipes_by_entity_id={"cloth": _recipe()},
    )


def _cloth(built: NativeTaskArenaEnvironment) -> _Deformable:
    return built.env.unwrapped.scene["cloth_runtime"]


class _Observers:
    def __init__(self, built: NativeTaskArenaEnvironment) -> None:
        self.built = built
        self.sample_index = 0
        self.time_seconds = 0.0
        self.commit_physx_writes = True
        self.destination_changes: dict = {}
        self.diagnostic_changes: dict = {}
        self.audit_changes: dict = {}
        self.clock_changes: dict = {}
        self.sync_changes: dict = {}

    def destination(self, **kwargs):
        row = {
            "schema_version": DESTINATION_SCHEMA_VERSION,
            "destination_entity_id": kwargs["destination_entity_id"],
            "pose_world": {
                "position_m": [0.5, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "linear_velocity_world_mps": [0.0, 0.0, 0.0],
            "angular_velocity_world_radps": [0.0, 0.0, 0.0],
            "source": "native_stage_transform_and_static_anchor",
            "native_pose_readback": True,
        }
        row.update(copy.deepcopy(self.destination_changes))
        return row

    def diagnostic(self, **kwargs):
        row = {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "deformable_entity_id": kwargs["deformable_entity_id"],
            "solver_divergence_count": 0,
            "source": "native_solver_diagnostic_buffer",
        }
        row.update(copy.deepcopy(self.diagnostic_changes))
        return row

    def audit(self, **kwargs):
        entity_ids = sorted(kwargs["entity_ids"])
        row = {
            "schema_version": INTEGRITY_AUDIT_SCHEMA_VERSION,
            "audited_entity_ids": entity_ids,
            "entity_write_count_after_episode_start": {entity_id: 0 for entity_id in entity_ids},
            "hidden_attachment_constraint_count_by_entity_id": {
                entity_id: 0 for entity_id in entity_ids
            },
            "hidden_attachment_constraint_count": 0,
            "source": "pinned_worker_entity_write_and_constraint_audit",
        }
        row.update(copy.deepcopy(self.audit_changes))
        return row

    def clock(self):
        row = {
            "schema_version": CLOCK_SCHEMA_VERSION,
            "sample_index": self.sample_index,
            "time_seconds": self.time_seconds,
            "source": "native_environment_step_clock",
        }
        row.update(copy.deepcopy(self.clock_changes))
        return row

    def synchronize(self):
        if self.commit_physx_writes:
            _cloth(self.built).commit_pending_writes_to_physx()
        row = {
            "schema_version": POST_RESET_SYNC_SCHEMA_VERSION,
            "physics_fetch_completed": True,
            "camera_refresh_completed": True,
            "synchronization_step_index": 0,
            "source": "native_physics_fetch_and_renderer_refresh",
        }
        row.update(copy.deepcopy(self.sync_changes))
        return row


def _readback(
    built: NativeTaskArenaEnvironment | None = None,
) -> tuple[NativeDeformableTaskArenaReadback, _Observers]:
    built = built or _built()
    observers = _Observers(built)
    return (
        NativeDeformableTaskArenaReadback(
            built,
            destination_state_observer=observers.destination,
            solver_diagnostic_observer=observers.diagnostic,
            entity_integrity_audit_observer=observers.audit,
            native_clock_observer=observers.clock,
            post_reset_synchronizer=observers.synchronize,
        ),
        observers,
    )


def test_reset_uses_fresh_physx_root_views_and_diagnostic_never_admits_scoring() -> None:
    built = _built()
    readback, _ = _readback(built)

    reset = readback.reset_after_native_reset()
    sample = readback.read_noncontact_diagnostic_sample()

    cloth = _cloth(built)
    assert cloth.state_writes == 1
    assert cloth.target_writes == 1
    assert np.allclose(cloth.root_view.velocities, 0.0)
    assert np.allclose(cloth.root_view.targets[..., 3], 1.0)
    assert reset["post_reset_episode_state_writes_by_adapter"] == 0
    assert reset["evaluation_or_scoring_admitted"] is False
    assert reset["blockers"] == [CONTACT_CAPABILITY_BLOCKER]
    assert reset["receipt_digest"].startswith("sha256:")
    assert sample["sample_index"] == 0
    assert sample["time_seconds"] == 0.0
    assert set(sample["entities"]) == {"cloth", "basket", "franka"}
    assert sample["entities"]["cloth"]["nodal_kinematic_flags"] == [1.0] * 3
    assert "gripper_contact_pair_count_by_entity_id" not in sample["entities"]["franka"]
    assert sample["native_readback"]["evaluation_or_scoring_admitted"] is False
    assert sample["native_readback"]["contact_capability"] == "unavailable"
    assert sample["native_readback"]["caller_asserted_success_used"] is False
    assert sample["native_readback"]["entity_integrity_audit"]["audited_entity_ids"] == [
        "basket",
        "cloth",
        "franka",
        "table",
        "wall",
    ]


def test_reads_do_not_advance_the_native_environment_clock() -> None:
    readback, observers = _readback()
    readback.reset_after_native_reset()

    first = readback.read_noncontact_diagnostic_sample()
    second = readback.read_noncontact_diagnostic_sample()
    assert (first["sample_index"], first["time_seconds"]) == (0, 0.0)
    assert (second["sample_index"], second["time_seconds"]) == (0, 0.0)

    observers.sample_index = 1
    observers.time_seconds = 0.05
    advanced = readback.read_noncontact_diagnostic_sample()
    assert (advanced["sample_index"], advanced["time_seconds"]) == (1, 0.05)


def test_evaluation_sample_and_capability_gate_raise_typed_contact_blocker() -> None:
    readback, _ = _readback()

    for operation in (readback.ensure_evaluation_capable, readback.read_task_sample):
        with pytest.raises(
            NativeDeformableTaskArenaReadbackError,
            match=CONTACT_CAPABILITY_BLOCKER,
        ):
            operation()


def test_diagnostic_sample_before_reset_fails_closed() -> None:
    readback, _ = _readback()
    with pytest.raises(
        NativeDeformableTaskArenaReadbackError,
        match="native_deformable_reset_not_observed",
    ):
        readback.read_noncontact_diagnostic_sample()


def test_default_buffers_are_copied_before_reset_writes() -> None:
    built = _built()
    cloth = _cloth(built)
    original_default = cloth.data.default_nodal_state_w.copy()
    original_target = cloth.data.nodal_kinematic_target.copy()

    readback, _ = _readback(built)
    readback.reset_after_native_reset()

    assert np.array_equal(cloth.data.default_nodal_state_w, original_default)
    assert np.allclose(original_target[..., 3], 0.0)


def test_cached_write_without_physx_root_view_change_is_not_reset_proof() -> None:
    built = _built()
    readback, observers = _readback(built)
    observers.commit_physx_writes = False

    with pytest.raises(
        NativeDeformableTaskArenaReadbackError,
        match="native_deformable_reset_physx_readback_mismatch",
    ):
        readback.reset_after_native_reset()

    assert np.allclose(_cloth(built).data.nodal_vel_w, 0.0)
    assert np.allclose(_cloth(built).root_view.velocities, 4.0)


def test_diagnostic_reads_root_view_instead_of_mutable_isaac_cache() -> None:
    built = _built()
    readback, _ = _readback(built)
    readback.reset_after_native_reset()
    _cloth(built).data.nodal_vel_w[...] = 99.0

    sample = readback.read_noncontact_diagnostic_sample()

    assert sample["entities"]["cloth"]["nodal_velocities_world_mps"] == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("task_kind", "native_deformable_readback_task_kind_invalid"),
        ("entity_binding", "native_deformable_scene_entity_binding_missing"),
        ("robot_binding", "native_deformable_robot_entity_binding_invalid"),
        (
            "destination_physics",
            "native_deformable_destination_physics_type_invalid",
        ),
        ("wrong_operation", "native_deformable_reset_recipe_invalid:cloth"),
        ("extra_step", "native_deformable_reset_recipe_invalid:cloth"),
        ("empty_state_id", "native_deformable_reset_recipe_invalid:cloth"),
        ("boolean_free_flag", "native_deformable_reset_recipe_invalid:cloth"),
        ("integer_free_flag", "native_deformable_reset_recipe_invalid:cloth"),
        ("boolean_order", "native_deformable_reset_recipe_invalid:cloth"),
        ("duplicate_gripper", "native_deformable_grasp_frame_invalid"),
        ("unhashable_gripper", "native_deformable_grasp_frame_invalid"),
    ],
)
def test_frozen_entity_gripper_and_exact_reset_contracts_fail_closed(
    mutation: str, expected: str
) -> None:
    built = _built()
    if mutation == "task_kind":
        built.plan["task_kind"] = "articulated_open_close"
    elif mutation == "entity_binding":
        del built.scene_asset_names_by_entity_id["cloth"]
    elif mutation == "robot_binding":
        built.plan["task_entity_role_index"]["robot"] = ["other"]
    elif mutation == "destination_physics":
        destination = next(
            row for row in built.plan["task_entities"] if row["entity_id"] == "basket"
        )
        destination["physics_type"] = "articulation"
    elif mutation == "wrong_operation":
        built.entity_reset_recipes_by_entity_id["cloth"]["steps"][4]["operation"] = (
            "readback_data_nodal_state_and_kinematic_target"
        )
    elif mutation == "extra_step":
        built.entity_reset_recipes_by_entity_id["cloth"]["steps"].append(
            {"order": 6, "operation": "extra"}
        )
    elif mutation == "empty_state_id":
        built.entity_reset_recipes_by_entity_id["cloth"]["state_id"] = ""
    elif mutation == "boolean_free_flag":
        built.entity_reset_recipes_by_entity_id["cloth"]["steps"][3]["free_flag_value"] = True
    elif mutation == "integer_free_flag":
        built.entity_reset_recipes_by_entity_id["cloth"]["steps"][3]["free_flag_value"] = 1
    elif mutation == "boolean_order":
        built.entity_reset_recipes_by_entity_id["cloth"]["steps"][0]["order"] = True
    elif mutation == "duplicate_gripper":
        built.plan["robot"]["grasp_frame"]["body_names"] = [
            "left_finger",
            "left_finger",
        ]
    else:
        built.plan["robot"]["grasp_frame"]["body_names"] = [{}, {}]
    with pytest.raises(NativeDeformableTaskArenaReadbackError, match=expected):
        _readback(built)


def test_post_reset_requires_physics_fetch_and_camera_refresh() -> None:
    readback, observers = _readback()
    observers.sync_changes = {"camera_refresh_completed": False}

    with pytest.raises(
        NativeDeformableTaskArenaReadbackError,
        match="native_deformable_post_reset_synchronization_invalid",
    ):
        readback.reset_after_native_reset()


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        (
            {"native_pose_readback": False},
            "native_deformable_destination_observation_invalid",
        ),
        (
            {"source": "caller_asserted_pose"},
            "native_deformable_destination_observation_invalid",
        ),
    ],
)
def test_destination_observation_rejects_unqualified_sources(changes: dict, expected: str) -> None:
    readback, observers = _readback()
    readback.reset_after_native_reset()
    observers.destination_changes = changes
    with pytest.raises(NativeDeformableTaskArenaReadbackError, match=expected):
        readback.read_noncontact_diagnostic_sample()


def test_dynamic_rigid_destination_requires_native_rigid_root_state_source() -> None:
    built = _built()
    destination = next(row for row in built.plan["task_entities"] if row["entity_id"] == "basket")
    destination["physics_type"] = "rigid_body"
    readback, observers = _readback(built)
    observers.destination_changes = {"source": "native_rigid_body_root_state"}
    readback.reset_after_native_reset()

    sample = readback.read_noncontact_diagnostic_sample()

    assert sample["native_readback"]["destination_pose_source"] == ("native_rigid_body_root_state")


def test_solver_diagnostic_rejects_boolean_divergence_count() -> None:
    readback, observers = _readback()
    readback.reset_after_native_reset()
    observers.diagnostic_changes = {"solver_divergence_count": True}
    with pytest.raises(
        NativeDeformableTaskArenaReadbackError,
        match="native_deformable_solver_divergence_count_invalid",
    ):
        readback.read_noncontact_diagnostic_sample()


def test_integrity_audit_requires_every_entity_and_zero_post_start_writes() -> None:
    readback, observers = _readback()
    readback.reset_after_native_reset()
    observers.audit_changes = {
        "entity_write_count_after_episode_start": {
            "basket": 0,
            "cloth": 0,
            "franka": 0,
            "table": 1,
            "wall": 0,
        }
    }
    with pytest.raises(
        NativeDeformableTaskArenaReadbackError,
        match="native_deformable_entity_write_after_episode_start_observed",
    ):
        readback.read_noncontact_diagnostic_sample()

    observers.audit_changes = {
        "entity_write_count_after_episode_start": {
            "basket": 0,
            "cloth": 0,
            "franka": 0,
            "table": 0,
        }
    }
    with pytest.raises(
        NativeDeformableTaskArenaReadbackError,
        match="native_deformable_entity_integrity_audit_invalid",
    ):
        readback.read_noncontact_diagnostic_sample()


def test_integrity_audit_rejects_hidden_attachment_constraint() -> None:
    readback, observers = _readback()
    readback.reset_after_native_reset()
    observers.audit_changes = {"hidden_attachment_constraint_count": 1}
    with pytest.raises(
        NativeDeformableTaskArenaReadbackError,
        match="native_deformable_entity_integrity_audit_invalid",
    ):
        readback.read_noncontact_diagnostic_sample()

    observers.audit_changes = {
        "hidden_attachment_constraint_count_by_entity_id": {
            "basket": 0,
            "cloth": 1,
            "franka": 0,
            "table": 0,
            "wall": 0,
        },
        "hidden_attachment_constraint_count": 1,
    }
    with pytest.raises(
        NativeDeformableTaskArenaReadbackError,
        match="native_deformable_hidden_attachment_constraint_observed",
    ):
        readback.read_noncontact_diagnostic_sample()


def test_clock_rejects_boolean_and_regression_but_not_an_unchanged_read() -> None:
    readback, observers = _readback()
    readback.reset_after_native_reset()
    readback.read_noncontact_diagnostic_sample()
    readback.read_noncontact_diagnostic_sample()

    observers.clock_changes = {"sample_index": True}
    with pytest.raises(
        NativeDeformableTaskArenaReadbackError,
        match="native_deformable_clock_sample_index_invalid",
    ):
        readback.read_noncontact_diagnostic_sample()

    observers.clock_changes = {}
    observers.sample_index = 2
    observers.time_seconds = 0.1
    readback.read_noncontact_diagnostic_sample()
    observers.sample_index = 1
    with pytest.raises(
        NativeDeformableTaskArenaReadbackError,
        match="native_deformable_clock_regressed",
    ):
        readback.read_noncontact_diagnostic_sample()


def test_native_robot_body_names_must_be_unique() -> None:
    built = _built()
    built.env.unwrapped.scene["robot"].data.body_names = [
        "left_finger",
        "left_finger",
    ]
    readback, _ = _readback(built)
    readback.reset_after_native_reset()
    with pytest.raises(
        NativeDeformableTaskArenaReadbackError,
        match="native_deformable_robot_body_names_missing",
    ):
        readback.read_noncontact_diagnostic_sample()


def test_root_view_rejects_multiple_environments_and_non_numeric_state() -> None:
    built = _built()
    readback, _ = _readback(built)
    readback.reset_after_native_reset()
    cloth = _cloth(built)
    cloth.root_view.positions = np.concatenate(
        (cloth.root_view.positions, cloth.root_view.positions), axis=0
    )
    cloth.root_view.velocities = np.concatenate(
        (cloth.root_view.velocities, cloth.root_view.velocities), axis=0
    )
    cloth.root_view.targets = np.concatenate(
        (cloth.root_view.targets, cloth.root_view.targets), axis=0
    )
    with pytest.raises(
        NativeDeformableTaskArenaReadbackError,
        match="native_deformable_physx_nodal_positions_missing",
    ):
        readback.read_noncontact_diagnostic_sample()

    built = _built()
    readback, _ = _readback(built)
    readback.reset_after_native_reset()
    _cloth(built).root_view.positions = np.asarray(
        [[[True, 0.0, 0.5], [0.1, 0.0, 0.5], [0.0, 0.1, 0.5]]],
        dtype=object,
    )
    with pytest.raises(
        NativeDeformableTaskArenaReadbackError,
        match="native_deformable_physx_nodal_positions_invalid",
    ):
        readback.read_noncontact_diagnostic_sample()
