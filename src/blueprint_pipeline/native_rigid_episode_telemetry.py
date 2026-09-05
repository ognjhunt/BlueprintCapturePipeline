"""Episode-owned counters and native rigid safety channels for strict scoring."""
from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import product
from typing import Any

from .adp_rigid_retreat_scoring import _rotate, _vector

CONTACT_CHANNELS = {
    "robot_background": "robot_scene_contact_peak_force_n",
    "robot_object": "robot_task_forbidden_collision_peak_force_n",
    "object_background": "task_scene_collision_peak_force_n",
    "destination_background": "destination_scene_forbidden_contact_peak_force_n",
}


class NativeRigidEpisodeTelemetry:
    """Track real resets and contact acquisitions only within an episode."""

    def __init__(self, task_spec: Mapping[str, Any]):
        self.spec = task_spec
        self.active = False
        self.retries = 0
        self.acquisitions = 0
        self.previous_acquired = False
        self.trace_complete = True

    def begin_episode(self) -> None:
        # The runner calls this after its initial reset, before its first sample.
        self.active = True
        self.retries = 0
        self.acquisitions = 0
        self.previous_acquired = False
        self.trace_complete = True

    def reset_executed(self) -> None:
        if self.active:
            self.retries += 1
            self.previous_acquired = False

    @staticmethod
    def _inside(point: list[float], bounds: Any) -> bool | None:
        if not isinstance(bounds, Mapping):
            return None
        lower, upper = _vector(bounds.get("minimum"), 3), _vector(bounds.get("maximum"), 3)
        if lower is None or upper is None or any(a >= b for a, b in zip(lower, upper, strict=True)):
            return None
        return all(a <= x <= b for a, x, b in zip(lower, point, upper, strict=True))

    def observe(self, sample: dict[str, Any]) -> None:
        native_grasp = sample.get("grasp_frame_position_world_m")
        grasp = _vector(native_grasp, 3)
        pose = _vector(sample.get("task_object_pose_world"), 7)
        threshold = self.spec["collision_failure_minimum_force_n"]
        configured = self.spec.get("task_success_contract", {}).get("criteria", {}).get("temporal_invariants", {})
        requested_classes = configured.get("forbidden_contact_classes", [])
        active_classes: list[str] = []
        complete = all(name in CONTACT_CHANNELS for name in requested_classes)
        for name, channel in CONTACT_CHANNELS.items():
            if name == "destination_background" and not self.spec.get("destination_support_asset_id"):
                continue
            force = sample.get(channel)
            if (isinstance(force, bool) or not isinstance(force, (int, float))
                    or not math.isfinite(force) or force < 0):
                complete = False
            elif force >= threshold:
                active_classes.append(name)
        sample["contact_classes_active"] = active_classes if complete else None
        sample["contact_class_measurement_channels"] = dict(CONTACT_CHANNELS)
        bounds = self.spec.get("subject_collision_bounds_scoring_frame_m")
        subject_points = None
        if pose is not None and isinstance(bounds, Mapping):
            low, high = _vector(bounds.get("minimum"), 3), _vector(bounds.get("maximum"), 3)
            if low is not None and high is not None:
                subject_points = [[pose[i] + point[i] for i in range(3)]
                                  for point in (_rotate(corner, pose[3:])
                                                for corner in product(*zip(low, high, strict=True)))]
        object_inside = [self._inside(point, self.spec.get("workspace_position_bounds_world_m"))
                         for point in subject_points] if subject_points else [None]
        robot_inside = self._inside(grasp, self.spec.get("robot_workspace_position_bounds_world_m")) if grasp else None
        all_inside = [*object_inside, robot_inside]
        sample["workspace_excursion"] = (any(value is False for value in all_inside)
                                          if all(value is not None for value in all_inside) else None)
        sample["workspace_measurement_source"] = "oriented_subject_collision_corners_and_measured_grasp_frame"
        width = sample.get("gripper_width_m")
        release = self.spec.get("release_gripper_width_min_m")
        contact = sample.get("task_contact_active")
        if (not isinstance(contact, bool) or isinstance(width, bool)
                or not isinstance(width, (int, float)) or not math.isfinite(width)
                or isinstance(release, bool) or not isinstance(release, (int, float))):
            self.trace_complete = False
        else:
            acquired = contact and width < release
            if self.active and acquired and not self.previous_acquired:
                self.acquisitions += 1
            self.previous_acquired = acquired
        sample["retry_count"] = self.retries if self.active else None
        sample["regrasp_count"] = max(0, self.acquisitions - 1) if self.active and self.trace_complete else None
        sample["episode_event_measurement_source"] = "runner_initial_reset_boundary_and_filtered_task_contact_acquisitions"
