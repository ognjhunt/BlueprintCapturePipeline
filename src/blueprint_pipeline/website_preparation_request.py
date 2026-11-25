"""What a website job's scene preparation hands on: the scene intake's request and the pose its object goes back at.

``website_task_preparation.compile_website_scene_preparation`` works out where the job's object is, what it rests on,
where it goes or how it opens, what builds it and what the site's authority allows; these records are assembled from
what it found, and nothing here estimates, places or admits anything. The request
(``task_evaluation_scene_intake.REQUEST_SCHEMA``) names the scene, the job's object (``task.subject``), its support,
its destination or mechanism with the success rule, and the execution the authority allows; compile validates it. The
compose-back record is the pose the rebuilt object is set back at in the room, on the removal manifest's row for it.

One object per job today. The records moved here from compile unchanged, byte for byte
(``tests/test_website_preparation_request.py``), so several objects per job have one place to grow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from . import task_evaluation_scene_intake as intake
from .website_preparation_contracts import scene_configuration_max_retries

#: What a website preparation may claim (its ``claim_ceiling``), and so the intake request's ``claim_scope``.
CLAIM_CEILING = "development_only"


def geometry_origin(*, designed: bool, atlas_route: str) -> str:
    """Where the object's geometry comes from: designed, kept in the scan as filmed (the Atlas job's geometry-only
    route), or taken out of the scan before it was reconstructed."""
    return ("synthetic_design" if designed
            else "retained_in_captured_reconstruction" if atlas_route == "geometry_only"
            else "removed_before_reconstruction")


def request_subject(entry: Mapping[str, Any], *, aabb_min: Sequence[float], aabb_max: Sequence[float], origin: str,
                    complete: bool) -> dict[str, Any]:
    """The request's ``task.subject``: the object by its label, in its estimated simulator-frame bounds."""
    return {"description": entry.get("semantic_label") or entry["target_id"],
            "aabb_min_xyz": aabb_min, "aabb_max_xyz": aabb_max,
            "coordinate_frame": "Z_up_estimated_meters",
            "geometry_origin": origin,
            "complete_object_dimensions": complete}


def intake_request(*, capture_id: str, task_context: Mapping[str, Any], base_scene: Mapping[str, Any],
                   spend: Mapping[str, Any], articulated: bool, subject: Mapping[str, Any], support_label: Any,
                   support_aabb: Sequence[Any], mechanism: Any, destination: Any, success: Any,
                   anthropic: bool) -> dict[str, Any]:
    """The scene intake's request for the job's object. ``support_aabb``: the support's simulator-frame bounds (each
    None when no support was found); ``anthropic``: whether the authoring writes its text with Anthropic."""
    # Capture consent permits scene preparation; it does not manufacture a
    # paid simulation authorization or accept provider terms on the owner's behalf.
    owner = dict(spend.get("owner") or {})
    consent = dict(spend.get("consent") or {})
    return {
        "schema_version": intake.REQUEST_SCHEMA, "submission_id": capture_id, "owner": owner,
        "source": {"kind": "gaussian_splat", "binding_id": base_scene["splat_binding_id"],
                   "content_digest": base_scene["splat_digest"]},
        # The preparation binds the reconstructed collider and estimated
        # registration. This is not an owner-uploaded companion mesh or an
        # owner declaration of its coordinate frame.
        "task": {"task_id": "website-" + task_context["context_digest"][7:27],
                 "strategy": "articulated_open_close" if articulated else "pick_and_place",
                 "subject": subject,
                 "support": {"description": support_label or "support surface under subject",
                             "aabb_min_xyz": support_aabb[0], "aabb_max_xyz": support_aabb[1]},
                 **({"articulation": mechanism, "success": success} if articulated else
                    {"destination": destination or {"needs_input": "task_destination_pose_required"},
                     # The object's centre ends within the destination's area: its radius is the tolerance
                     # (``website_destination_area``) unless the site's words ask for less.
                     "success": success})},
        "execution": {"purpose": "scene_preparation", "max_total_spend_usd": spend["max_total_spend_usd"],
                      "max_paid_attempts": spend["max_paid_attempts"],
                      "max_retries": scene_configuration_max_retries(spend),
                      "expires_at_epoch": spend["expires_at_epoch"],
                      "allowed_providers": ["vast", "openai", "anthropic"] if anthropic else ["vast", "openai"],
                      "policy_candidates": [],
                      "claim_scope": CLAIM_CEILING},
        "consent": consent,
    }


def compose_back_record(*, task_context: Mapping[str, Any], subject_min: Sequence[float],
                        subject_max: Sequence[float], base_scene: Mapping[str, Any], meters_per_unit: float,
                        snap: float, registration_path: Path) -> dict[str, Any]:
    """Where the rebuilt object goes back in the room: its bounds' centre in runtime units, upright, with the
    registration that put it there."""
    return {"replacement_asset_id": "website-subject-" + task_context["context_digest"][7:27],
            "pose_world": {"position": [(subject_min[i] + subject_max[i]) / 2 for i in range(3)],
                           "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                           "unit": base_scene["up_axis"] + "-up runtime units",
                           "meters_per_unit": meters_per_unit, "support_snap_runtime_units": snap},
            "replacement_asset_frame_registration_uri": str(registration_path)}


def composed_removal_manifest(removal_manifest: Mapping[str, Any],
                              compose_backs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The removal manifest with each rebuilt object's compose-back record on its row (``compose_backs``, by target);
    every other row as it was."""
    composed = dict(removal_manifest)
    composed["entries"] = [dict(row, compose_back=compose_backs[row["target_id"]]) if row["target_id"] in compose_backs
                           else dict(row) for row in removal_manifest["entries"]]
    return composed


__all__ = ["CLAIM_CEILING", "compose_back_record", "composed_removal_manifest", "geometry_origin", "intake_request",
           "request_subject"]
