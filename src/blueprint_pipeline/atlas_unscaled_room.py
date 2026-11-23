"""A built room whose scale was never measured ships flagged, in its own units (owner, 2026-11-23).

The scale comes from the original views' own Atlas pose (``website_atlas_pipeline._source_pose``). When that pose
fails, World Labs answering ``EXECUTION_FAILED`` on both attempts for instance, the object-removed background is
still built: its splat and mesh are complete, posed on the edited views' own pose. Until 2026-11-23 the scene source
answered ``website_atlas_source_pose_unavailable`` for it on every retry, and the job never shipped (internal test
job 2, ``itest-917ca852…``, from 20 Nov).

Now the scene source hands on that background as the room (``website_atlas_scene_source``), flagged
``scale_not_measured``. It is levelled on what it observed with its own units read as metres for the levelling's
thresholds only (``LEVELLING_UNITS``); the base scene's ``meters_per_unit`` is None, and so is its source geometry:
nothing is lifted onto it, so the job's object is not built. The room package (``website_scene_handoff.
write_room_package``) records relative units and writes no metres value; the company package's README says
``SCALE_NOT_MEASURED_NOTE``.

Walls stay: a pose stopped by privacy, rights or money hands on nothing (``ships_unscaled``), the spending cap
answers as before, and a withdrawal, a takedown or revoked rights stop the company package as any other.
Standard library only; no I/O.
"""
from __future__ import annotations

from typing import Any, Mapping

#: On the scene source's answer and the room package's record: the room's physical scale was not measured.
SCALE_NOT_MEASURED = "scale_not_measured"
#: The README's one sentence for it (``website_company_simulation``).
SCALE_NOT_MEASURED_NOTE = "Sizes in this room are relative; its scale was not measured."
#: The README's units line for it, in place of "scene.usda is in metres (1 unit is 1 metre), radians and seconds.".
UNITS_LINE = ("- scene.usda states 1 unit as 1 metre only so that tools can open it; the room's scale was not "
              "measured. Angles are in radians, times in seconds.")
#: The room's own units, read as metres by the levelling's thresholds and carried one to one into the package.
LEVELLING_UNITS = 1.0
#: The room package's coordinate frame for it: no metres per unit, its units relative.
RELATIVE_FRAME = {"declared_meters_per_unit": None, "units": "relative", "units_per_reconstruction_unit": 1.0}


def ships_unscaled(source_pose: Mapping[str, Any]) -> bool:
    """Whether a background whose source pose did not complete ships unscaled: never after a wall.

    A wall is recorded on the pose by its type (``wall``, ``website_atlas_pipeline._source_pose``); a pose stored
    before that is read by its code's words, as the retry policy reads them (``job_retry_policy``). Kept free of
    the retry module itself: the company builder imports this module and must not reach the GPU lane."""
    from .job_retry_policy import hard_stop_kind, owner_closed_refusal
    reason = str(source_pose.get("reason") or "")
    return (bool(reason) and not source_pose.get("wall") and hard_stop_kind(reason) is None
            and not owner_closed_refusal(reason))


def is_unscaled(base_scene: Mapping[str, Any]) -> bool:
    return base_scene.get("scale_status") == SCALE_NOT_MEASURED and base_scene.get("meters_per_unit") is None


def relative_frame(base_scene: Mapping[str, Any]) -> tuple[float, Any, None]:
    """``website_scene_handoff._sim_frame`` for an unscaled room: its units one to one, Z up, no registration."""
    import numpy as np

    up_sign = -1 if base_scene["up_axis"] == "-Y" else 1
    runtime_to_sim = np.eye(4)
    # Y up turns to Z up as ``_sim_frame`` turns it; a Z-up room keeps its axes.
    runtime_to_sim[:3, :3] = LEVELLING_UNITS * (np.array([[1, 0, 0], [0, 0, -up_sign], [0, up_sign, 0]])
                                                if base_scene["up_axis"] in {"Y", "-Y"} else np.eye(3))
    return LEVELLING_UNITS, runtime_to_sim, None


def runtime_scale(coordinate_frame: Mapping[str, Any]) -> Any:
    """The scale a room's files are written at: its declared metres per unit, or one to one for relative units."""
    declared = coordinate_frame.get("declared_meters_per_unit")
    if declared is None and coordinate_frame.get("units") == RELATIVE_FRAME["units"]:
        return RELATIVE_FRAME["units_per_reconstruction_unit"]
    return declared
