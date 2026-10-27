"""Pure native task-profile metadata; no robot or policy inventory imports.

Profiles describe existing adapter/scorer contracts, never an intake allowlist.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskProfile:
    profile_id: str
    native_task_kind: str
    catalog_task_family_id: str
    runtime_adapter: str
    success_predicate: str
    aliases: tuple[str, ...]
    required_inputs: tuple[str, ...]


# First supported profiles. Each names the native adapter that compiles it and
# the success predicate the independent scorer reads.
TASK_PROFILES: tuple[TaskProfile, ...] = (
    TaskProfile(
        profile_id="rigid_pick_place",
        native_task_kind="rigid_pick_place",
        catalog_task_family_id="rigid_relocation",
        runtime_adapter="task_evaluation_rigid_relocation_native_adapter",
        success_predicate="object_pose_within_target_region",
        aliases=("rigid_pick_place", "rigid_pick_and_place", "rigid_object_pick_place", "rigid_object_relocation",
                 "rigid_relocation"),
        required_inputs=("metric_object_geometry", "target_region"),
    ),
    TaskProfile(
        profile_id="articulated_open_close",
        native_task_kind="articulated_open_close",
        catalog_task_family_id="articulated_open_close",
        runtime_adapter="task_evaluation_articulated_open_close_native_adapter",
        success_predicate="target_joint_within_qualified_interval",
        aliases=("articulated_open_close",),
        required_inputs=("qualified_articulation_graph",),
    ),
)


def task_profile_for(task_family: str | None) -> TaskProfile | None:
    value = (task_family or "").strip().lower()
    return next((row for row in TASK_PROFILES if value in row.aliases), None)

