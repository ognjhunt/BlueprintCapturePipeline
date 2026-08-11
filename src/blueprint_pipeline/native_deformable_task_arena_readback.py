"""Fail-closed native Arena readback for a deformable transfer.

The pinned Isaac Lab deformable object exposes PhysX nodal state, kinematic
targets, element deformation gradients, and reset writes.  It does *not*
expose qualified rigid--deformable contact-pair attribution or normal force.
This module therefore supports a diagnostic non-contact snapshot and an exact
native reset proof, but refuses to present an Arena sample as evaluation-ready.

The typed contact blocker is deliberate.  A future backend may implement the
same entity-keyed sample contract only after a released contact API is bound by
the trusted native worker; a callback that merely claims a source name cannot
upgrade this Arena capability boundary.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_runtime import NativeTaskArenaEnvironment


SCHEMA_VERSION = "native_deformable_task_arena_readback.v2"
DESTINATION_SCHEMA_VERSION = "native_destination_state_observation.v1"
DIAGNOSTIC_SCHEMA_VERSION = "native_deformable_solver_diagnostic.v1"
INTEGRITY_AUDIT_SCHEMA_VERSION = "native_task_entity_integrity_audit.v1"
CLOCK_SCHEMA_VERSION = "native_environment_step_clock.v1"
POST_RESET_SYNC_SCHEMA_VERSION = "native_deformable_post_reset_sync.v1"

# Kept as an import-compatible name for code that inventories the unavailable
# requirement.  No Arena observer is accepted for this schema.
CONTACT_SCHEMA_VERSION = "native_deformable_contact_observation.v1"
CONTACT_CAPABILITY_BLOCKER = "native_rigid_deformable_contact_attribution_unavailable"


class NativeDeformableTaskArenaReadbackError(ValueError):
    """Stable failures for missing or malformed native deformable evidence."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


Observer = Callable[..., Mapping[str, Any]]


def _raise(error: str) -> None:
    raise NativeDeformableTaskArenaReadbackError([error])


def _native_copy(value: Any, *, error: str) -> Any:
    if value is None:
        _raise(error)
    module = type(value).__module__
    if module == "warp" or module.startswith("warp."):
        import warp as wp

        value = wp.to_torch(value)
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    try:
        import numpy as np

        return np.asarray(value).copy()
    except (TypeError, ValueError) as exc:
        raise NativeDeformableTaskArenaReadbackError([error]) from exc


def _native_list(value: Any, *, error: str) -> Any:
    if value is None:
        _raise(error)
    module = type(value).__module__
    if module == "warp" or module.startswith("warp."):
        import warp as wp

        value = wp.to_torch(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeDeformableTaskArenaReadbackError([error]) from exc


def _first_environment(
    value: Any,
    *,
    trailing_rank: int,
    error: str,
) -> list[Any]:
    rows = _native_list(value, error=error)
    if not isinstance(rows, list) or len(rows) != 1:
        _raise(error)
    first = rows[0]
    if not isinstance(first, list) or not first:
        _raise(error)

    def rank(node: Any) -> int:
        depth = 0
        while isinstance(node, list) and node:
            depth += 1
            node = node[0]
        return depth

    if rank(first) != trailing_rank:
        _raise(error)
    return first


def _finite_vector(value: Any, *, size: int, error: str) -> list[float]:
    if (
        isinstance(value, (str, bytes, Mapping))
        or not isinstance(value, Sequence)
        or len(value) != size
    ):
        _raise(error)
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        _raise(error)
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise NativeDeformableTaskArenaReadbackError([error]) from exc
    if not all(math.isfinite(item) for item in result):
        _raise(error)
    return result


def _nonnegative_integer(value: Any, *, error: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _raise(error)
    return value


def _body_positions(robot: Any, body_names: Sequence[str]) -> list[list[float]]:
    if len(body_names) != 2 or len(set(body_names)) != 2:
        _raise("native_deformable_gripper_body_identity_invalid")
    data = getattr(robot, "data", None)
    native_names = getattr(data, "body_names", None) or getattr(robot, "body_names", None)
    if (
        isinstance(native_names, (str, bytes))
        or not isinstance(native_names, Sequence)
        or any(
            not isinstance(name, str) or not name.strip() or name != name.strip()
            for name in native_names
        )
        or len(set(native_names)) != len(native_names)
    ):
        _raise("native_deformable_robot_body_names_missing")
    poses = _first_environment(
        getattr(data, "body_pose_w", None),
        trailing_rank=2,
        error="native_deformable_robot_body_pose_missing",
    )
    if len(poses) != len(native_names):
        _raise("native_deformable_robot_body_pose_missing")
    positions: list[list[float]] = []
    for body_name in body_names:
        if body_name not in native_names:
            _raise(f"native_deformable_gripper_body_missing:{body_name}")
        pose = poses[list(native_names).index(body_name)]
        if not isinstance(pose, list) or len(pose) < 7:
            _raise("native_deformable_robot_body_pose_missing")
        positions.append(
            _finite_vector(
                pose[:3],
                size=3,
                error="native_deformable_robot_body_pose_invalid",
            )
        )
    return positions


def _validate_reset_recipe(recipe: Any, *, entity_id: str) -> dict[str, Any]:
    expected_operations = [
        "load_default_nodal_state",
        "zero_nodal_velocities",
        "write_nodal_state_to_sim_index",
        "write_nodal_kinematic_target_to_sim_index",
        "readback_physx_root_view_state_and_kinematic_target",
    ]
    if not isinstance(recipe, Mapping):
        _raise(f"native_deformable_reset_recipe_invalid:{entity_id}")
    steps = recipe.get("steps")
    if (
        recipe.get("reset_kind") != "native_deformable_state"
        or not isinstance(recipe.get("state_id"), str)
        or not recipe["state_id"].strip()
        or recipe["state_id"] != recipe["state_id"].strip()
        or recipe.get("write_scope") != "before_episode_start_only"
        or recipe.get("direct_state_write_after_episode_start_allowed") is not False
        or recipe.get("native_readback_required") is not True
        or not isinstance(steps, list)
        or len(steps) != 5
        or any(not isinstance(row, Mapping) for row in steps)
        or [row.get("operation") for row in steps] != expected_operations
        or any(
            isinstance(row.get("order"), bool) or not isinstance(row.get("order"), int)
            for row in steps
        )
        or [row.get("order") for row in steps] != [1, 2, 3, 4, 5]
        or not isinstance(steps[3].get("free_flag_value"), float)
        or steps[3].get("free_flag_value") != 1.0
    ):
        _raise(f"native_deformable_reset_recipe_invalid:{entity_id}")
    try:
        return json.loads(json.dumps(dict(recipe), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeDeformableTaskArenaReadbackError(
            [f"native_deformable_reset_recipe_invalid:{entity_id}"]
        ) from exc


def _physx_root_state(asset: Any) -> tuple[list[Any], list[Any], list[Any]]:
    view = getattr(asset, "root_view", None)
    getters = (
        getattr(view, "get_sim_nodal_positions", None),
        getattr(view, "get_sim_nodal_velocities", None),
        getattr(view, "get_sim_kinematic_targets", None),
    )
    if not all(callable(getter) for getter in getters):
        _raise("native_deformable_physx_root_readback_api_missing")
    positions = _first_environment(
        getters[0](),
        trailing_rank=2,
        error="native_deformable_physx_nodal_positions_missing",
    )
    velocities = _first_environment(
        getters[1](),
        trailing_rank=2,
        error="native_deformable_physx_nodal_velocities_missing",
    )
    targets = _first_environment(
        getters[2](),
        trailing_rank=2,
        error="native_deformable_physx_kinematic_targets_missing",
    )
    if len(positions) != len(velocities) or len(positions) != len(targets):
        _raise("native_deformable_physx_root_readback_shape_invalid")
    return (
        [
            _finite_vector(
                row,
                size=3,
                error="native_deformable_physx_nodal_positions_invalid",
            )
            for row in positions
        ],
        [
            _finite_vector(
                row,
                size=3,
                error="native_deformable_physx_nodal_velocities_invalid",
            )
            for row in velocities
        ],
        [
            _finite_vector(
                row,
                size=4,
                error="native_deformable_physx_kinematic_targets_invalid",
            )
            for row in targets
        ],
    )


def _midpoint(points: Sequence[Sequence[float]]) -> list[float]:
    return [float(points[0][axis] + points[1][axis]) / 2.0 for axis in range(3)]


class NativeDeformableTaskArenaReadback:
    """Reset and diagnose one entity-keyed Arena deformable transfer."""

    def __init__(
        self,
        built: NativeTaskArenaEnvironment,
        *,
        destination_state_observer: Observer,
        solver_diagnostic_observer: Observer,
        entity_integrity_audit_observer: Observer,
        native_clock_observer: Observer,
        post_reset_synchronizer: Observer,
    ) -> None:
        if built.plan.get("task_kind") != "deformable_transfer":
            _raise("native_deformable_readback_task_kind_invalid")
        task_spec = built.plan.get("task_spec")
        if not isinstance(task_spec, Mapping):
            _raise("native_deformable_task_spec_missing")
        identifiers = {
            "deformable": task_spec.get("deformable_entity_id"),
            "destination": task_spec.get("destination_entity_id"),
            "robot": task_spec.get("robot_entity_id"),
        }
        if (
            any(
                not isinstance(value, str) or not value.strip() or value != value.strip()
                for value in identifiers.values()
            )
            or len(set(identifiers.values())) != 3
        ):
            _raise("native_deformable_task_entity_ids_invalid")
        self._deformable_id = str(identifiers["deformable"])
        self._destination_id = str(identifiers["destination"])
        self._robot_id = str(identifiers["robot"])
        entity_names = built.scene_asset_names_by_entity_id
        if self._deformable_id not in entity_names or self._destination_id not in entity_names:
            _raise("native_deformable_scene_entity_binding_missing")
        role_index = built.plan.get("task_entity_role_index")
        robot_ids = role_index.get("robot") if isinstance(role_index, Mapping) else None
        if robot_ids != [self._robot_id]:
            _raise("native_deformable_robot_entity_binding_invalid")
        task_entities = built.plan.get("task_entities")
        if (
            not isinstance(task_entities, list)
            or not task_entities
            or any(not isinstance(row, Mapping) for row in task_entities)
        ):
            _raise("native_deformable_task_entities_missing")
        entity_ids = [row.get("entity_id") for row in task_entities]
        if (
            any(
                not isinstance(value, str) or not value.strip() or value != value.strip()
                for value in entity_ids
            )
            or len(set(entity_ids)) != len(entity_ids)
            or not set(identifiers.values()).issubset(entity_ids)
        ):
            _raise("native_deformable_task_entities_invalid")
        entity_rows = {str(row["entity_id"]): row for row in task_entities}
        destination_physics_type = entity_rows[self._destination_id].get("physics_type")
        destination_state_sources = {
            "static_collider": "native_stage_transform_and_static_anchor",
            "rigid_body": "native_rigid_body_root_state",
        }
        if destination_physics_type not in destination_state_sources:
            _raise("native_deformable_destination_physics_type_invalid")
        observers = (
            destination_state_observer,
            solver_diagnostic_observer,
            entity_integrity_audit_observer,
            native_clock_observer,
            post_reset_synchronizer,
        )
        if not all(callable(observer) for observer in observers):
            _raise("native_deformable_observer_missing")
        robot_plan = built.plan.get("robot")
        grasp_frame = robot_plan.get("grasp_frame") if isinstance(robot_plan, Mapping) else None
        if (
            not isinstance(grasp_frame, Mapping)
            or grasp_frame.get("kind") != "body_midpoint"
            or not isinstance(grasp_frame.get("body_names"), list)
            or len(grasp_frame["body_names"]) != 2
            or any(
                not isinstance(name, str) or not name.strip() or name != name.strip()
                for name in grasp_frame["body_names"]
            )
            or len(set(grasp_frame["body_names"])) != 2
        ):
            _raise("native_deformable_grasp_frame_invalid")

        recipe = built.entity_reset_recipes_by_entity_id.get(self._deformable_id)
        self._recipe = _validate_reset_recipe(recipe, entity_id=self._deformable_id)
        self._built = built
        self._entity_ids = sorted(str(value) for value in entity_ids)
        self._gripper_body_names = list(grasp_frame["body_names"])
        self._destination_state_source = destination_state_sources[destination_physics_type]
        self._destination_state_observer = destination_state_observer
        self._solver_diagnostic_observer = solver_diagnostic_observer
        self._entity_integrity_audit_observer = entity_integrity_audit_observer
        self._native_clock_observer = native_clock_observer
        self._post_reset_synchronizer = post_reset_synchronizer
        self._last_clock: tuple[int, float] | None = None
        self._reset_receipt: dict[str, Any] | None = None

    def _scene(self) -> Any:
        env = getattr(self._built.env, "unwrapped", self._built.env)
        scene = getattr(env, "scene", None)
        if scene is None:
            _raise("native_deformable_scene_readback_missing")
        return scene

    def _deformable_asset(self) -> Any:
        try:
            runtime_name = self._built.scene_asset_names_by_entity_id[self._deformable_id]
            return self._scene()[runtime_name]
        except (KeyError, TypeError) as exc:
            raise NativeDeformableTaskArenaReadbackError(
                ["native_deformable_asset_readback_missing"]
            ) from exc

    def _clock(self) -> tuple[int, float, Mapping[str, Any]]:
        value = self._native_clock_observer()
        if (
            not isinstance(value, Mapping)
            or value.get("schema_version") != CLOCK_SCHEMA_VERSION
            or value.get("source") != "native_environment_step_clock"
        ):
            _raise("native_deformable_clock_observation_invalid")
        sample_index = _nonnegative_integer(
            value.get("sample_index"),
            error="native_deformable_clock_sample_index_invalid",
        )
        raw_time = value.get("time_seconds")
        if isinstance(raw_time, bool) or not isinstance(raw_time, Real):
            _raise("native_deformable_clock_time_invalid")
        try:
            time_seconds = float(raw_time)
        except (TypeError, ValueError) as exc:
            raise NativeDeformableTaskArenaReadbackError(
                ["native_deformable_clock_time_invalid"]
            ) from exc
        if not math.isfinite(time_seconds) or time_seconds < 0.0:
            _raise("native_deformable_clock_time_invalid")
        if self._last_clock is not None and (
            sample_index < self._last_clock[0] or time_seconds < self._last_clock[1]
        ):
            _raise("native_deformable_clock_regressed")
        self._last_clock = (sample_index, time_seconds)
        return sample_index, time_seconds, value

    def reset_after_native_reset(self) -> dict[str, Any]:
        """Write the pre-episode reset and re-read PhysX root-view tensors."""

        asset = self._deformable_asset()
        data = getattr(asset, "data", None)
        state = _native_copy(
            getattr(data, "default_nodal_state_w", None),
            error="native_deformable_default_nodal_state_missing",
        )
        target = _native_copy(
            getattr(data, "nodal_kinematic_target", None),
            error="native_deformable_kinematic_target_missing",
        )
        state_list = _first_environment(
            state,
            trailing_rank=2,
            error="native_deformable_default_nodal_state_invalid",
        )
        target_list = _first_environment(
            target,
            trailing_rank=2,
            error="native_deformable_kinematic_target_invalid",
        )
        if len(state_list) != len(target_list):
            _raise("native_deformable_reset_tensor_shape_invalid")
        for row in state_list:
            _finite_vector(
                row,
                size=6,
                error="native_deformable_default_nodal_state_invalid",
            )
        for row in target_list:
            _finite_vector(
                row,
                size=4,
                error="native_deformable_kinematic_target_invalid",
            )

        state[..., 3:6] = 0.0
        target[..., 0:3] = state[..., 0:3]
        target[..., 3] = 1.0
        write_state = getattr(asset, "write_nodal_state_to_sim_index", None)
        write_target = getattr(asset, "write_nodal_kinematic_target_to_sim_index", None)
        if not callable(write_state) or not callable(write_target):
            _raise("native_deformable_reset_api_missing")
        write_state(state, env_ids=[0])
        write_target(target, env_ids=[0])

        synchronization = self._post_reset_synchronizer()
        if (
            not isinstance(synchronization, Mapping)
            or synchronization.get("schema_version") != POST_RESET_SYNC_SCHEMA_VERSION
            or synchronization.get("source") != "native_physics_fetch_and_renderer_refresh"
            or synchronization.get("physics_fetch_completed") is not True
            or synchronization.get("camera_refresh_completed") is not True
            or isinstance(synchronization.get("synchronization_step_index"), bool)
            or not isinstance(synchronization.get("synchronization_step_index"), int)
            or synchronization.get("synchronization_step_index") < 0
        ):
            _raise("native_deformable_post_reset_synchronization_invalid")
        normalized_synchronization = {
            "schema_version": POST_RESET_SYNC_SCHEMA_VERSION,
            "source": "native_physics_fetch_and_renderer_refresh",
            "physics_fetch_completed": True,
            "camera_refresh_completed": True,
            "synchronization_step_index": synchronization["synchronization_step_index"],
        }

        positions, velocities, targets = _physx_root_state(asset)
        expected_state = _first_environment(
            state, trailing_rank=2, error="native_deformable_reset_state_invalid"
        )
        expected_target = _first_environment(
            target,
            trailing_rank=2,
            error="native_deformable_reset_target_invalid",
        )
        expected_positions = [row[:3] for row in expected_state]
        expected_velocities = [row[3:6] for row in expected_state]
        if (
            positions != expected_positions
            or velocities != expected_velocities
            or targets != expected_target
        ):
            _raise("native_deformable_reset_physx_readback_mismatch")

        self._last_clock = None
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "deformable_entity_id": self._deformable_id,
            "reset_state_digest": canonical_digest(
                {
                    "nodal_positions_world_m": positions,
                    "nodal_velocities_world_mps": velocities,
                }
            ),
            "reset_kinematic_target_digest": canonical_digest({"nodal_kinematic_target": targets}),
            "native_readback_api": [
                "root_view.get_sim_nodal_positions",
                "root_view.get_sim_nodal_velocities",
                "root_view.get_sim_kinematic_targets",
            ],
            "free_kinematic_flag_value": 1.0,
            "post_reset_episode_state_writes_by_adapter": 0,
            "post_reset_synchronization": normalized_synchronization,
            "contact_capability": "unavailable",
            "blockers": [CONTACT_CAPABILITY_BLOCKER],
            "evaluation_or_scoring_admitted": False,
            "recipe": self._recipe,
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        self._reset_receipt = receipt
        return json.loads(json.dumps(receipt))

    @property
    def reset_receipt(self) -> dict[str, Any] | None:
        return None if self._reset_receipt is None else json.loads(json.dumps(self._reset_receipt))

    def ensure_evaluation_capable(self) -> None:
        """Fail before an Arena control/policy episode can claim contact."""

        _raise(CONTACT_CAPABILITY_BLOCKER)

    def read_task_sample(self) -> dict[str, Any]:
        """Refuse an evaluation sample until native contact attribution exists."""

        self.ensure_evaluation_capable()
        raise AssertionError("unreachable")

    def read_noncontact_diagnostic_sample(self) -> dict[str, Any]:
        """Return native state for loading/reset diagnostics, never scoring."""

        if self._reset_receipt is None:
            _raise("native_deformable_reset_not_observed")
        sample_index, time_seconds, clock = self._clock()
        asset = self._deformable_asset()
        positions, velocities, targets = _physx_root_state(asset)
        data = getattr(asset, "data", None)
        raw_gradients = _first_environment(
            getattr(data, "sim_element_deform_gradient_w", None),
            trailing_rank=3,
            error="native_deformable_gradient_readback_missing",
        )
        gradients: list[list[list[float]]] = []
        for matrix in raw_gradients:
            if not isinstance(matrix, list) or len(matrix) != 3:
                _raise("native_deformable_gradient_readback_invalid")
            gradients.append(
                [
                    _finite_vector(
                        row,
                        size=3,
                        error="native_deformable_gradient_readback_invalid",
                    )
                    for row in matrix
                ]
            )
        flags = [float(row[3]) for row in targets]
        if any(not math.isfinite(value) for value in flags):
            _raise("native_deformable_kinematic_flag_invalid")

        scene = self._scene()
        try:
            robot = scene["robot"]
        except (KeyError, TypeError) as exc:
            raise NativeDeformableTaskArenaReadbackError(
                ["native_deformable_robot_readback_missing"]
            ) from exc
        body_points = _body_positions(robot, self._gripper_body_names)

        destination = self._destination_state_observer(
            destination_entity_id=self._destination_id,
            runtime_name=self._built.scene_asset_names_by_entity_id[self._destination_id],
            prim_path=self._built.scene_asset_prim_paths_by_entity_id.get(self._destination_id),
        )
        if (
            not isinstance(destination, Mapping)
            or destination.get("schema_version") != DESTINATION_SCHEMA_VERSION
            or destination.get("destination_entity_id") != self._destination_id
            or destination.get("source") != self._destination_state_source
            or destination.get("native_pose_readback") is not True
        ):
            _raise("native_deformable_destination_observation_invalid")
        pose = destination.get("pose_world")
        if not isinstance(pose, Mapping):
            _raise("native_deformable_destination_pose_invalid")
        destination_position = _finite_vector(
            pose.get("position_m"),
            size=3,
            error="native_deformable_destination_pose_invalid",
        )
        destination_orientation = _finite_vector(
            pose.get("orientation_xyzw"),
            size=4,
            error="native_deformable_destination_pose_invalid",
        )
        if abs(math.sqrt(sum(value * value for value in destination_orientation)) - 1.0) > 1.0e-5:
            _raise("native_deformable_destination_pose_invalid")
        destination_linear_velocity = _finite_vector(
            destination.get("linear_velocity_world_mps"),
            size=3,
            error="native_deformable_destination_velocity_invalid",
        )
        destination_angular_velocity = _finite_vector(
            destination.get("angular_velocity_world_radps"),
            size=3,
            error="native_deformable_destination_velocity_invalid",
        )

        diagnostic = self._solver_diagnostic_observer(deformable_entity_id=self._deformable_id)
        if (
            not isinstance(diagnostic, Mapping)
            or diagnostic.get("schema_version") != DIAGNOSTIC_SCHEMA_VERSION
            or diagnostic.get("deformable_entity_id") != self._deformable_id
            or diagnostic.get("source") != "native_solver_diagnostic_buffer"
        ):
            _raise("native_deformable_solver_diagnostic_invalid")
        divergence_count = _nonnegative_integer(
            diagnostic.get("solver_divergence_count"),
            error="native_deformable_solver_divergence_count_invalid",
        )

        audit = self._entity_integrity_audit_observer(entity_ids=list(self._entity_ids))
        writes = (
            audit.get("entity_write_count_after_episode_start", {})
            if isinstance(audit, Mapping)
            else {}
        )
        hidden_constraints = (
            audit.get("hidden_attachment_constraint_count_by_entity_id", {})
            if isinstance(audit, Mapping)
            else {}
        )
        hidden_constraint_total = (
            audit.get("hidden_attachment_constraint_count") if isinstance(audit, Mapping) else None
        )
        if (
            not isinstance(audit, Mapping)
            or audit.get("schema_version") != INTEGRITY_AUDIT_SCHEMA_VERSION
            or audit.get("source") != "pinned_worker_entity_write_and_constraint_audit"
            or audit.get("audited_entity_ids") != self._entity_ids
            or not isinstance(writes, Mapping)
            or set(writes) != set(self._entity_ids)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in writes.values()
            )
            or not isinstance(hidden_constraints, Mapping)
            or set(hidden_constraints) != set(self._entity_ids)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in hidden_constraints.values()
            )
            or isinstance(hidden_constraint_total, bool)
            or not isinstance(hidden_constraint_total, int)
            or hidden_constraint_total < 0
            or hidden_constraint_total != sum(hidden_constraints.values())
        ):
            _raise("native_deformable_entity_integrity_audit_invalid")
        if any(value != 0 for value in writes.values()):
            _raise("native_deformable_entity_write_after_episode_start_observed")
        if hidden_constraint_total != 0:
            _raise("native_deformable_hidden_attachment_constraint_observed")
        normalized_audit = {
            "schema_version": INTEGRITY_AUDIT_SCHEMA_VERSION,
            "source": "pinned_worker_entity_write_and_constraint_audit",
            "audited_entity_ids": list(self._entity_ids),
            "entity_write_count_after_episode_start": {
                entity_id: writes[entity_id] for entity_id in self._entity_ids
            },
            "hidden_attachment_constraint_count_by_entity_id": {
                entity_id: hidden_constraints[entity_id] for entity_id in self._entity_ids
            },
            "hidden_attachment_constraint_count": 0,
        }

        return {
            "schema_version": SCHEMA_VERSION,
            "sample_index": sample_index,
            "time_seconds": time_seconds,
            "entities": {
                self._deformable_id: {
                    "nodal_positions_world_m": positions,
                    "nodal_velocities_world_mps": velocities,
                    "deformation_gradients": gradients,
                    "nodal_kinematic_flags": flags,
                    "state_write_count_after_episode_start": writes[self._deformable_id],
                    "solver_divergence_count": divergence_count,
                },
                self._destination_id: {
                    "pose_world": {
                        "position_m": destination_position,
                        "orientation_xyzw": destination_orientation,
                    },
                    "linear_velocity_world_mps": destination_linear_velocity,
                    "angular_velocity_world_radps": destination_angular_velocity,
                    "state_write_count_after_episode_start": writes[self._destination_id],
                },
                self._robot_id: {
                    "gripper_body_origins_world_m_diagnostic_only": body_points,
                    "grasp_frame_position_world_m_diagnostic_only": _midpoint(body_points),
                    "state_write_count_after_episode_start": writes[self._robot_id],
                },
            },
            "native_readback": {
                "reset_receipt_digest": self._reset_receipt["receipt_digest"],
                "clock_source": clock["source"],
                "destination_pose_source": destination.get("source"),
                "solver_diagnostic_source": diagnostic["source"],
                "entity_integrity_audit": normalized_audit,
                "contact_capability": "unavailable",
                "blockers": [CONTACT_CAPABILITY_BLOCKER],
                "evaluation_or_scoring_admitted": False,
                "caller_asserted_success_used": False,
            },
        }


__all__ = [
    "CLOCK_SCHEMA_VERSION",
    "CONTACT_CAPABILITY_BLOCKER",
    "CONTACT_SCHEMA_VERSION",
    "DESTINATION_SCHEMA_VERSION",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "INTEGRITY_AUDIT_SCHEMA_VERSION",
    "NativeDeformableTaskArenaReadback",
    "NativeDeformableTaskArenaReadbackError",
    "POST_RESET_SYNC_SCHEMA_VERSION",
    "SCHEMA_VERSION",
]
