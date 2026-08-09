"""Fail-closed input-scope admission for released reconstruction methods.

Being able to serialize an input into a method's file format is not evidence
that the input is within the bounded operating region Blueprint has qualified.
This module keeps that distinction explicit and digestable before paid work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


METHOD_SCOPE_SCHEMA_VERSION = "method_input_scope_admission.v1"


class MethodInputScopeError(ValueError):
    """The proposed method input could not be measured deterministically."""


def evaluate_multiview_mask_fraction_scope(
    *,
    method_id: str,
    profile_id: str,
    cameras: Sequence[Mapping[str, Any]],
    mask_records: Sequence[Mapping[str, Any]],
    maximum_mask_fraction: float,
    profile_basis: str,
) -> dict[str, Any]:
    """Measure target-mask scale and return a typed paid-admission decision.

    The ceiling is a Blueprint qualification boundary, not a claim about the
    upstream authors' training distribution. Every camera must have an exact
    pixel count and dimensions; partial evidence is rejected.
    """

    if (
        not method_id
        or not profile_id
        or not profile_basis
        or isinstance(maximum_mask_fraction, bool)
        or not 0.0 < float(maximum_mask_fraction) < 1.0
    ):
        raise MethodInputScopeError("method_scope_profile_invalid")
    camera_dimensions: dict[str, tuple[int, int]] = {}
    for camera in cameras:
        camera_id = str(camera.get("camera_id") or "")
        intrinsics = camera.get("intrinsics")
        if not camera_id or camera_id in camera_dimensions or not isinstance(intrinsics, Mapping):
            raise MethodInputScopeError("method_scope_camera_contract_invalid")
        width = intrinsics.get("width")
        height = intrinsics.get("height")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
        ):
            raise MethodInputScopeError("method_scope_camera_dimensions_invalid")
        camera_dimensions[camera_id] = (width, height)

    observed: dict[str, int] = {}
    for record in mask_records:
        camera_id = str(record.get("camera_id") or "")
        count = record.get("masked_pixel_count")
        if (
            camera_id not in camera_dimensions
            or camera_id in observed
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise MethodInputScopeError("method_scope_mask_contract_invalid")
        width, height = camera_dimensions[camera_id]
        if count > width * height:
            raise MethodInputScopeError("method_scope_mask_pixel_count_invalid")
        observed[camera_id] = count
    if set(observed) != set(camera_dimensions):
        raise MethodInputScopeError("method_scope_camera_mask_join_incomplete")

    rows = []
    for camera_id in sorted(camera_dimensions):
        width, height = camera_dimensions[camera_id]
        count = observed[camera_id]
        rows.append(
            {
                "camera_id": camera_id,
                "width": width,
                "height": height,
                "masked_pixel_count": count,
                "mask_fraction": count / (width * height),
            }
        )
    maximum_observed = max(row["mask_fraction"] for row in rows)
    admitted = maximum_observed <= float(maximum_mask_fraction)
    blocker = f"{method_id}_input_exceeds_qualified_mask_scale"
    return {
        "schema_version": METHOD_SCOPE_SCHEMA_VERSION,
        "method_id": method_id,
        "profile_id": profile_id,
        "profile_basis": profile_basis,
        "profile_is_blueprint_admission_policy_not_author_claim": True,
        "maximum_allowed_mask_fraction": float(maximum_mask_fraction),
        "maximum_observed_mask_fraction": maximum_observed,
        "per_camera": rows,
        "status": "admitted" if admitted else "blocked",
        "paid_execution_admitted": admitted,
        "blockers": [] if admitted else [blocker],
    }


__all__ = [
    "METHOD_SCOPE_SCHEMA_VERSION",
    "MethodInputScopeError",
    "evaluate_multiview_mask_fraction_scope",
]
