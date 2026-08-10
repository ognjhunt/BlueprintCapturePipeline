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
differs at all from the spec's, and discovering that inside a paid run costs
the run. Both sides are therefore derived from the same spec.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


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
    twin_usd_filename: str,
    scene_collision_filename: str,
    appearance_filename: str | None = None,
    twin_position_world_m: Sequence[float] | None = None,
    asset_filename_aliases: Mapping[str, Sequence[str]] | None = None,
    intended_for_policy_execution: bool = False,
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

    raw_joints = task_spec.get("articulated_joints") or []
    joints: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(raw_joints):
        if not isinstance(row, Mapping):
            errors.append(f"articulated_runtime_composition_joint_invalid:{index}")
            continue
        joint_id = str(row.get("joint_id") or "")
        prim_path = str(row.get("joint_prim_path") or "")
        if not joint_id or not prim_path:
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
        joints.append(
            {
                "joint_id": joint_id,
                "joint_prim_path": prim_path,
                "role": str(row.get("role") or "unspecified"),
            }
        )

    articulated = task_kind == TASK_KIND_ARTICULATED
    if articulated and not joints:
        errors.append("articulated_runtime_composition_joints_missing")

    # A bundle renames assets, so the authoring filename recorded here is often
    # not the filename on the provider. Aliases travel with the object rather
    # than being guessed at load time, and an alias for a role that does not
    # exist is an error - a typo that silently left an asset unaliased would
    # surface as a missing file on paid hardware.
    # A vision policy cannot be evaluated against an invisible room. Scripted
    # controls are pure physics and need no appearance; a learned policy reads
    # the cameras, and a collision-only background gives it a floating
    # appliance in a void. Refusing here is the only place that distinction
    # gets checked.
    if intended_for_policy_execution and not appearance_filename:
        errors.append("articulated_runtime_composition_appearance_missing_for_policy")

    known_roles = {"scene_collision", "task_object"}
    if appearance_filename:
        known_roles.add("scene_appearance")
    aliases_by_role: dict[str, list[str]] = {}
    for role, values in dict(asset_filename_aliases or {}).items():
        if str(role) not in known_roles:
            errors.append(
                "articulated_runtime_composition_alias_role_unknown:"
                f"{role}:{sorted(known_roles)}"
            )
            continue
        aliases_by_role[str(role)] = [str(value) for value in values if str(value)]

    if errors:
        raise ArticulatedRuntimeCompositionError(errors)

    objects: list[dict[str, Any]] = [
        {
            "name": "scene_collision",
            "semantic_role": "scene_collision",
            "object_type": "BASE",
            "usd_filename": str(scene_collision_filename),
            "usd_filename_aliases": list(aliases_by_role.get("scene_collision", [])),
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
                "usd_filename_aliases": list(
                    aliases_by_role.get("scene_appearance", [])
                ),
                "visible": True,
                "initial_position_world_m": [0.0, 0.0, 0.0],
            }
        )
    objects.append(
        {
            "name": "task_object",
            "semantic_role": "task_object",
            # An articulated twin spawned rigid has frozen joints and no error.
            "object_type": "ARTICULATION" if articulated else "RIGID",
            "usd_filename": str(twin_usd_filename),
            "usd_filename_aliases": list(aliases_by_role.get("task_object", [])),
            "visible": True,
            "initial_position_world_m": position,
        }
    )

    return {
        "schema_version": RUNTIME_COMPOSITION_SCHEMA_VERSION,
        "task_kind": task_kind,
        "objects": objects,
        # Recorded, not inferred by a reader: a scene with no appearance is
        # valid for scripted controls and invalid for a vision policy, and the
        # difference has to survive into the receipt.
        "appearance_present": bool(appearance_filename),
        "intended_for_policy_execution": bool(intended_for_policy_execution),
        "task_sample_binding": {
            "joint_ids": sorted(row["joint_id"] for row in joints),
            "joint_prim_paths": {
                row["joint_id"]: row["joint_prim_path"] for row in joints
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
