"""Derive a variation request for one site task from what the runtime can vary.

RoboLab-style task variations (where the object starts, how it is turned, the
light, the camera, the friction) tell a site whether a policy's success holds
up or only works in the one pose it was shown. Writing those matrices by hand
for each site is slow and easy to get wrong, so this derives the
``exact_workcell_variation_request.v1`` from the site task itself:

- only runtime targets the native runtime can apply and read back
  (``SUPPORTED_SCENARIO_RUNTIME_TARGETS``) are ever proposed;
- bounds come from the task: the object's start may move by at most a
  quarter of the way to its destination, so a variation never pre-solves or
  reshapes the task;
- a dimension whose nominal value is unknown (friction without a material
  readback) is left out rather than given a plausible number.

The result is admitted only through ``validate_variation_request``, which
still owns every rule about exact-workcell invariance and identity.

Backlog: ADP-009D (Day-28 ``public_data_rehearsal``: per-family success
across the preregistered variation families). Existing scenes may exercise it
only as ``development_only``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .exact_workcell_variation_matrix import (
    DEFAULT_CELL_COUNT,
    REQUEST_SCHEMA_VERSION,
    REQUIRED_CONTROLS,
    ExactWorkcellVariationError,
    validate_variation_request,
)
from .native_task_runtime_contract import (
    SCENARIO_RUNTIME_TARGET_UNITS,
    SUPPORTED_SCENARIO_RUNTIME_TARGETS,
)

# Upper bounds for each family. The task can only narrow them.
MAXIMUM_START_SHIFT_M = 0.03
MAXIMUM_YAW_DEGREES = 10.0
LIGHT_SCALE_RANGE = (0.8, 1.2)
MAXIMUM_CAMERA_SHIFT_M = 0.02
FRICTION_FRACTION = 0.2
# The object may move at most this share of the way to its destination.
START_SHIFT_SHARE_OF_REACH = 0.25


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _reach_m(task_spec: Mapping[str, Any]) -> float | None:
    """Distance from the object's start to the middle of its destination."""

    start = task_spec.get("start_pose_world")
    bounds = task_spec.get("destination_position_bounds_world_m")
    if not isinstance(start, list) or len(start) < 3 or not isinstance(bounds, Mapping):
        return None
    low, high = bounds.get("minimum"), bounds.get("maximum")
    if not isinstance(low, list) or not isinstance(high, list) or len(low) < 3 or len(high) < 3:
        return None
    points = [_finite(value) for value in [*start[:3], *low[:3], *high[:3]]]
    if any(value is None for value in points):
        return None
    begin = points[0:3]
    middle = [(points[3 + axis] + points[6 + axis]) / 2 for axis in range(3)]  # type: ignore[operator]
    return math.dist(begin, middle)  # type: ignore[arg-type]


def _dimension(
    *,
    dimension_id: str,
    family: str,
    target: str,
    nominal: float,
    minimum: float,
    maximum: float,
    decimals: int,
    tolerance: float,
    source_contract: str,
    authority_digest: str,
) -> dict[str, Any]:
    if target not in SUPPORTED_SCENARIO_RUNTIME_TARGETS:  # pragma: no cover - a coding error
        raise ExactWorkcellVariationError([f"variation_target_unsupported:{target}"])
    return {
        "dimension_id": dimension_id,
        "family": family,
        "value_type": "continuous",
        "nominal": round(nominal, decimals),
        "minimum": round(minimum, decimals),
        "maximum": round(maximum, decimals),
        "decimals": decimals,
        "unit": SCENARIO_RUNTIME_TARGET_UNITS[target],
        "application_target": target,
        "application_tolerance": tolerance,
        "parameter_path": target.removeprefix("EventManager.reset."),
        "source_contract": source_contract,
        "authority_digest": authority_digest,
        "exact_workcell_invariant": True,
        "changes_object_or_task_identity": False,
    }


def derive_variation_dimensions(
    *,
    task_spec: Mapping[str, Any],
    scene_binding: Mapping[str, Any],
    task_binding: Mapping[str, Any],
    embodiment_binding: Mapping[str, Any],
    nominal_dynamic_friction: float | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """The dimensions this task supports, and the ones left out with why."""

    omitted: list[str] = []
    dimensions: list[dict[str, Any]] = []
    task_authority = str(task_binding.get("task_digest") or "")
    scene_authority = str(scene_binding.get("scene_digest") or "")
    camera_authority = str(embodiment_binding.get("camera_calibration_digest") or "")

    reach = _reach_m(task_spec)
    if reach is None:
        omitted.append("object_start_y:task_start_or_destination_missing")
    else:
        shift = min(MAXIMUM_START_SHIFT_M, START_SHIFT_SHARE_OF_REACH * reach)
        if shift < 0.005:
            omitted.append("object_start_y:destination_too_close_to_vary_start")
        else:
            dimensions.append(_dimension(
                dimension_id="object_start_y_delta", family="placement_approach",
                target="EventManager.reset.object_start_position_m.y",
                nominal=0.0, minimum=-shift, maximum=shift, decimals=4, tolerance=0.001,
                source_contract="task", authority_digest=task_authority,
            ))

    # A task that constrains the final orientation tightly keeps the start
    # rotation well inside that tolerance, so the policy is not asked to fix
    # more than the task allows.
    tolerance = _finite(task_spec.get("destination_orientation_tolerance_rad"))
    yaw = MAXIMUM_YAW_DEGREES
    if tolerance is not None and tolerance > 0:
        yaw = min(yaw, math.degrees(tolerance) * 0.5)
    if yaw < 1.0:
        omitted.append("object_yaw:orientation_tolerance_too_tight_to_vary")
    else:
        dimensions.append(_dimension(
            dimension_id="object_yaw_delta", family="placement_approach",
            target="EventManager.reset.object_orientation.yaw",
            nominal=0.0, minimum=-yaw, maximum=yaw, decimals=2, tolerance=0.1,
            source_contract="task", authority_digest=task_authority,
        ))

    dimensions.append(_dimension(
        dimension_id="task_light_intensity_scale", family="illumination",
        target="EventManager.reset.task_light.intensity_scale",
        nominal=1.0, minimum=LIGHT_SCALE_RANGE[0], maximum=LIGHT_SCALE_RANGE[1],
        decimals=3, tolerance=0.01, source_contract="scene", authority_digest=scene_authority,
    ))
    dimensions.append(_dimension(
        dimension_id="external_camera_x_delta", family="camera_sensor",
        target="EventManager.reset.external_camera.pose.position.x",
        nominal=0.0, minimum=-MAXIMUM_CAMERA_SHIFT_M, maximum=MAXIMUM_CAMERA_SHIFT_M,
        decimals=4, tolerance=0.001, source_contract="embodiment", authority_digest=camera_authority,
    ))

    friction = _finite(nominal_dynamic_friction)
    if friction is None or friction <= 0:
        omitted.append("dynamic_friction:no_measured_nominal")
    else:
        dimensions.append(_dimension(
            dimension_id="task_subject_dynamic_friction", family="bounded_physics",
            target="EventManager.reset.task_subject_material.dynamic_friction",
            nominal=friction, minimum=friction * (1 - FRICTION_FRACTION),
            maximum=friction * (1 + FRICTION_FRACTION), decimals=3, tolerance=0.01,
            source_contract="scene", authority_digest=scene_authority,
        ))
    return dimensions, omitted


def build_site_task_variation_request(
    *,
    matrix_id: str,
    implementation_commit: str,
    seed_root: int,
    task_spec: Mapping[str, Any],
    scene_binding: Mapping[str, Any],
    task_binding: Mapping[str, Any],
    embodiment_binding: Mapping[str, Any],
    nominal_dynamic_friction: float | None = None,
    cell_count: int = DEFAULT_CELL_COUNT,
) -> tuple[dict[str, Any], list[str]]:
    """A validated variation request for one site task, and what it left out."""

    dimensions, omitted = derive_variation_dimensions(
        task_spec=task_spec,
        scene_binding=scene_binding,
        task_binding=task_binding,
        embodiment_binding=embodiment_binding,
        nominal_dynamic_friction=nominal_dynamic_friction,
    )
    request: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "matrix_id": matrix_id,
        "matrix_kind": "exact_workcell_primary",
        "implementation_commit": implementation_commit,
        "cell_count": cell_count,
        "seed_root": seed_root,
        "scene_binding": dict(scene_binding),
        "task_binding": dict(task_binding),
        "embodiment_binding": dict(embodiment_binding),
        "controls": {
            "control_ids": list(REQUIRED_CONTROLS),
            "run_on_every_cell": True,
            "same_resolved_cell_required": True,
        },
        "variation_dimensions": dimensions,
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return validate_variation_request(request), omitted


__all__ = [
    "build_site_task_variation_request",
    "derive_variation_dimensions",
]
