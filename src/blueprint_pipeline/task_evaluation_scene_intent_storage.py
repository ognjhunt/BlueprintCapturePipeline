"""Existing scene-intent publication checkpoint and exclusive writer."""
from __future__ import annotations

from .control_plane_registered_reference_gate import _publisher_checkpoint
from .task_evaluation_launch_preparation_storage import (
    _write_launch_preparation_record_exclusive_locked as _write_exclusive_native,
)
from .task_evaluation_scene_intent_contracts import SceneIntakeError


def _scene_publisher_checkpoint():
    from .control_plane_lane_experiment_errors import OwnerTargetVersionError
    try:
        _publisher_checkpoint()
    except OwnerTargetVersionError as exc:
        raise SceneIntakeError(exc.code) from None


def write_exclusive(path, value):
    from .control_plane_lane_experiment_errors import OwnerTargetVersionError
    _scene_publisher_checkpoint()
    try:
        return _write_exclusive_native(path, value)
    except OwnerTargetVersionError as exc:
        raise SceneIntakeError(exc.code) from None

