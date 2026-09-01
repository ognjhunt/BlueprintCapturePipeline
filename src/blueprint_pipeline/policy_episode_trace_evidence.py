"""Digest-bound state, contact/force, and task-object episode traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from .decision_evidence_contracts import canonical_digest


def episode_trace_evidence(
    *,
    joint_trace: Sequence[Sequence[float]],
    task_samples: Sequence[Mapping[str, Any]],
    task_pose_field: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Retain the exact state, contact/force, and task-object traces used to score."""

    indexed_joints = [
        {
            "step_index": int(
                task_samples[index].get("step_index", index)
                if index < len(task_samples)
                else index
            ),
            "joint_positions_rad": [float(value) for value in positions],
        }
        for index, positions in enumerate(joint_trace)
    ]
    samples = [json.loads(json.dumps(dict(sample), allow_nan=False)) for sample in task_samples]
    state: dict[str, Any] = {
        "schema_version": "policy_episode_state_trace.v1",
        "joint_states": indexed_joints,
        "task_state_samples": samples,
        "trace_digest": "",
    }
    state["trace_digest"] = canonical_digest(state, digest_field="trace_digest")

    contact_keys = (
        "finger_contact_forces_n",
        "contact_force_n",
        "contact_forces_n",
        "contact_active",
        "task_contact_active",
        "gripper_width_m",
        "robot_collision_failure",
        "scene_collision_failure",
    )
    contact_rows = [
        {
            "step_index": int(sample.get("step_index", index)),
            **{key: sample[key] for key in contact_keys if key in sample},
        }
        for index, sample in enumerate(samples)
        if any(key in sample for key in contact_keys)
    ]
    contacts: dict[str, Any] = {
        "schema_version": "policy_episode_contact_force_trace.v1",
        "samples": contact_rows,
        "typed_gap": (
            None if contact_rows else "contact_force_channels_unavailable_in_task_samples"
        ),
        "trace_digest": "",
    }
    contacts["trace_digest"] = canonical_digest(
        contacts, digest_field="trace_digest"
    )

    trajectory_rows = [
        {
            "step_index": int(sample.get("step_index", index)),
            "task_object_pose_world": sample.get(task_pose_field),
        }
        for index, sample in enumerate(samples)
        if task_pose_field in sample
    ]
    trajectory: dict[str, Any] = {
        "schema_version": "policy_episode_task_object_trajectory.v1",
        "source_field": task_pose_field,
        "samples": trajectory_rows,
        "typed_gap": (
            None
            if trajectory_rows
            else "task_object_pose_unavailable_in_task_samples"
        ),
        "trace_digest": "",
    }
    trajectory["trace_digest"] = canonical_digest(
        trajectory, digest_field="trace_digest"
    )
    return state, contacts, trajectory


__all__ = ["episode_trace_evidence"]
