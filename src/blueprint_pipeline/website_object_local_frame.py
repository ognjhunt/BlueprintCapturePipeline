"""Estimated object frame for authorized component tests, never room registration."""
from __future__ import annotations

import numpy as np

from .website_task_masks import masked_source_points

REGISTRATION_REFUSALS = frozenset({
    "website_registration_ambiguous", "website_registration_conflicts_provider_anchor",
    "website_registration_anchor_deviation", "website_registration_ground_plane_inconsistent",
    "website_registration_poor_fit",
})


def estimated_object_frame(*, track, source_geometry, registration_blocker):
    if registration_blocker not in REGISTRATION_REFUSALS:
        raise ValueError("website_object_frame_refusal_not_supported")
    if source_geometry.get("unit") != "estimated_meters":
        raise ValueError("website_object_frame_scale_invalid")
    points = masked_source_points(track, source_geometry["frames"])
    if len(points) < 8 or not np.isfinite(points).all():
        raise ValueError("website_object_frame_geometry_invalid")
    center = np.median(points, axis=0)
    # Source-view up is an orientation hint, not measured gravity. Principal
    # axes enclose the observed object itself, independently of Marble's frame.
    _, axes = np.linalg.eigh(np.cov((points - center).T))
    frame = next(row for row in source_geometry["frames"]
                 if row["frame_id"] == track["observations"][0]["source_frame_id"])
    camera = np.asarray(frame["world_from_camera"], dtype=float)[:3, :3]
    up = -camera[:, 1]
    z_index = int(np.argmax(np.abs(axes.T @ up)))
    z = axes[:, z_index]
    z *= 1 if np.dot(z, up) >= 0 else -1
    x = axes[:, max(i for i in range(3) if i != z_index)]
    x *= 1 if np.dot(x, camera[:, 0]) >= 0 else -1
    y = np.cross(z, x)
    x = np.cross(y, z)
    transform = np.eye(4)
    transform[:3, :3] = np.stack([x, y, z])
    transform[:3, 3] = -transform[:3, :3] @ center
    return {"schema_version": "website_object_local_frame.v1",
        "source_to_runtime": transform.tolist(), "scale": 1.0,
        "basis": "estimated_masked_object_principal_axes",
        "up_axis_basis": "principal_axis_nearest_source_camera_up_estimate",
        "source_geometry_digest": source_geometry["digest"],
        "captured_scene_integration": "pending", "registration_blocker": registration_blocker,
        "physical_scale_measured": False, "physical_registration_proven": False}
