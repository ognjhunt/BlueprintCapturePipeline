"""Decide what an articulated task's Isaac scene contains, off the GPU.

The runtime that builds the scene can only run inside the container, which
makes every mistake in it cost a launch. Most of those mistakes are not about
physics at all - they are about which asset gets which spawn type, which one
the cameras may see, and whether the joint names the scorer will demand match
the ones the runtime will read. All of that is decidable from the task spec
alone, so it is decided here and the runtime just executes the result.

The spawn type is the one worth being careful about. A refrigerator spawned as
a rigid body has no door: its joints are simply frozen, nothing errors, and the
task reads as impossible rather than misconfigured. Nothing downstream inspects
the spawn type, so if it is wrong here it is wrong everywhere and looks like a
policy failure.

The joint binding is the other. The scorer rejects a sample whose joint set
differs at all from the spec's, while simulator prim paths are an asset/runtime
property and do not belong in the scorer spec.  The planner therefore joins the
exact scorer joint ids to an explicit runtime binding and rejects either a
missing id or an extra prim before any paid run.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from .articulation_graph_contract import (
    ArticulationGraphContractError,
    validate_articulation_graph,
)


RUNTIME_COMPOSITION_SCHEMA_VERSION = "articulated_runtime_composition.v1"
TASK_KIND_ARTICULATED = "articulated_open_close"


class ArticulatedRuntimeCompositionError(ValueError):
    """Stable, sorted runtime-composition failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def plan_articulated_runtime_composition(
    *,
    task_spec: Mapping[str, Any],
    task_joint_bindings: Sequence[Mapping[str, Any]] | None = None,
    twin_usd_filename: str,
    scene_collision_filename: str,
    appearance_filename: str | None = None,
    twin_position_world_m: Sequence[float] | None = None,
    twin_object_type: str | None = None,
) -> dict[str, Any]:
    """Resolve spawn types, visibility, and the joint binding for one task."""

    errors: list[str] = []
    if not isinstance(task_spec, Mapping):
        raise ArticulatedRuntimeCompositionError(
            ["articulated_runtime_composition_task_spec_invalid"]
        )
    task_kind = str(task_spec.get("task_kind") or "")
    if not str(twin_usd_filename or "").strip():
        errors.append("articulated_runtime_composition_twin_missing")
    if not str(scene_collision_filename or "").strip():
        errors.append("articulated_runtime_composition_scene_collision_missing")

    position = [0.0, 0.0, 0.0]
    if twin_position_world_m is not None:
        try:
            position = [float(value) for value in twin_position_world_m]
        except (TypeError, ValueError):
            errors.append("articulated_runtime_composition_twin_position_invalid")
        else:
            if len(position) != 3:
                errors.append("articulated_runtime_composition_twin_position_invalid")

    articulated = task_kind == TASK_KIND_ARTICULATED
    resolved_object_type = twin_object_type or (
        "ARTICULATION" if articulated else "RIGID"
    )
    if resolved_object_type not in {"RIGID", "ARTICULATION"}:
        errors.append("articulated_runtime_composition_twin_object_type_invalid")
    if articulated and resolved_object_type != "ARTICULATION":
        errors.append("articulated_runtime_composition_articulated_spawn_required")
    reset_positions = task_spec.get("joint_reset_positions_rad")
    scorer_joint_ids: set[str] = set()
    scorer_joint_roles: dict[str, str] = {}
    if task_spec.get("schema_version") == "adp_task_spec.v2":
        graph = task_spec.get("articulation_graph")
        if not isinstance(graph, Mapping):
            errors.append("articulated_runtime_composition_graph_missing")
        else:
            try:
                normalized_graph = validate_articulation_graph(
                    graph,
                    require_target_joint=task_kind != "rigid_pick_place",
                )
            except ArticulationGraphContractError as exc:
                errors.extend(exc.errors)
            else:
                scorer_joint_ids = {
                    str(row["joint_id"]) for row in normalized_graph["joints"]
                }
                scorer_joint_roles = {
                    str(row["joint_id"]): str(row["role"])
                    for row in normalized_graph["joints"]
                }
    if reset_positions is not None:
        if not isinstance(reset_positions, Mapping):
            errors.append("articulated_runtime_composition_scorer_joints_invalid")
        else:
            scorer_joint_ids = {
                str(joint_id).strip()
                for joint_id in reset_positions
                if str(joint_id).strip()
            }
            if len(scorer_joint_ids) != len(reset_positions):
                errors.append("articulated_runtime_composition_scorer_joints_invalid")

    # Compatibility is retained for the first fixture, whose early development
    # spec carried runtime paths inline.  New contracts always pass the binding
    # separately and record that source in the resulting plan.
    binding_source = "runtime_contract"
    if task_joint_bindings is None:
        raw_joints = task_spec.get("articulated_joints") or []
        binding_source = "legacy_task_spec"
        if not scorer_joint_ids:
            scorer_joint_ids = {
                str(row.get("joint_id") or "").strip()
                for row in raw_joints
                if isinstance(row, Mapping) and str(row.get("joint_id") or "").strip()
            }
    else:
        raw_joints = task_joint_bindings

    joints: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_native_names: set[str] = set()
    for index, row in enumerate(raw_joints):
        if not isinstance(row, Mapping):
            errors.append(f"articulated_runtime_composition_joint_invalid:{index}")
            continue
        joint_id = str(row.get("joint_id") or "")
        prim_path = str(row.get("joint_prim_path") or "")
        native_joint_name = str(row.get("native_joint_name") or "")
        if binding_source == "legacy_task_spec" and not native_joint_name:
            native_joint_name = PurePosixPath(prim_path).name
        if (
            not joint_id
            or not PurePosixPath(prim_path).is_absolute()
            or len(PurePosixPath(prim_path).parts) < 3
            or ".." in PurePosixPath(prim_path).parts
            or not native_joint_name
            or PurePosixPath(native_joint_name).name != native_joint_name
        ):
            errors.append(f"articulated_runtime_composition_joint_invalid:{index}")
            continue
        if joint_id in seen:
            # Two joints sharing an id collapse into one sample entry, and the
            # scorer would then compare a shorter set than the spec declares.
            errors.append(
                f"articulated_runtime_composition_joint_id_duplicated:{joint_id}"
            )
            continue
        seen.add(joint_id)
        if native_joint_name in seen_native_names:
            errors.append(
                "articulated_runtime_composition_native_joint_name_duplicated:"
                + native_joint_name
            )
            continue
        seen_native_names.add(native_joint_name)
        joints.append(
            {
                "joint_id": joint_id,
                "joint_prim_path": prim_path,
                "native_joint_name": native_joint_name,
                "role": str(
                    row.get("role")
                    or scorer_joint_roles.get(joint_id)
                    or "unspecified"
                ),
            }
        )

    bound_joint_ids = {row["joint_id"] for row in joints}
    if articulated and not scorer_joint_ids:
        errors.append("articulated_runtime_composition_scorer_joints_missing")
    if articulated and not joints:
        errors.append("articulated_runtime_composition_joints_missing")
    for joint_id in sorted(scorer_joint_ids - bound_joint_ids):
        errors.append(
            f"articulated_runtime_composition_joint_binding_missing:{joint_id}"
        )
    for joint_id in sorted(bound_joint_ids - scorer_joint_ids):
        errors.append(
            f"articulated_runtime_composition_joint_binding_unexpected:{joint_id}"
        )
    for row in joints:
        expected_role = scorer_joint_roles.get(row["joint_id"])
        if expected_role is not None and row["role"] != expected_role:
            errors.append(
                "articulated_runtime_composition_joint_role_mismatch:"
                + row["joint_id"]
            )
    if task_kind == "rigid_pick_place" and any(
        role != "locked" for role in scorer_joint_roles.values()
    ):
        errors.append("articulated_runtime_composition_rigid_joint_not_locked")
    if errors:
        raise ArticulatedRuntimeCompositionError(errors)

    objects: list[dict[str, Any]] = [
        {
            "name": "scene_collision",
            "semantic_role": "scene_collision",
            "object_type": "BASE",
            "usd_filename": str(scene_collision_filename),
            "visible": False,
            "initial_position_world_m": [0.0, 0.0, 0.0],
        }
    ]
    if appearance_filename:
        objects.append(
            {
                "name": "scene_appearance",
                "semantic_role": "scene_appearance",
                "object_type": "BASE",
                "usd_filename": str(appearance_filename),
                "visible": True,
                "initial_position_world_m": [0.0, 0.0, 0.0],
            }
        )
    objects.append(
        {
            "name": "task_object",
            "semantic_role": "task_object",
            # An articulated twin spawned rigid has frozen joints and no error.
            "object_type": resolved_object_type,
            "usd_filename": str(twin_usd_filename),
            "visible": True,
            "initial_position_world_m": position,
        }
    )

    return {
        "schema_version": RUNTIME_COMPOSITION_SCHEMA_VERSION,
        "task_kind": task_kind,
        "objects": objects,
        "task_sample_binding": {
            "binding_source": binding_source,
            "joint_ids": sorted(row["joint_id"] for row in joints),
            "joint_prim_paths": {
                row["joint_id"]: row["joint_prim_path"] for row in joints
            },
            "native_joint_names": {
                row["joint_id"]: row["native_joint_name"] for row in joints
            },
            "joint_roles": {row["joint_id"]: row["role"] for row in joints},
        },
        "claim_boundary": {
            "spawn_types_are_planned_not_verified_against_a_runtime": True,
            "cameras_see_no_scene_background": not appearance_filename,
            "joint_indices_are_resolved_by_the_runtime_not_here": True,
        },
    }


__all__ = [
    "ArticulatedRuntimeCompositionError",
    "RUNTIME_COMPOSITION_SCHEMA_VERSION",
    "TASK_KIND_ARTICULATED",
    "plan_articulated_runtime_composition",
]
