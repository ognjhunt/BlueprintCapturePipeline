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

Walls stay: only a pose attempt the provider itself failed hands on its room (``ships_unscaled``); a pose stopped
by privacy, rights or money, or for any reason not on that list, hands on nothing; the spending cap answers as before; and a withdrawal, a takedown or revoked rights stop the company package as any other.
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


#: The scale review could not be asked now: its provider is out (a 429 or 5xx) or refused before any work (no credit,
#: a refused key: ``job_retry_policy.provider_error_waits``). ``website_atlas_pipeline._source_pose`` records it for a
#: built background, so the room ships unscaled at once, and the Atlas job asks the review again on the outage
#: schedule (``website_atlas_preview``): every ask goes through the same admissions, and a measured room replaces
#: this one. It is no wall: a provider being out says nothing about privacy, rights or money; a refusal that does
#: (the job's spending cap, a withdrawal, revoked consent) keeps its own code and ships nothing.
SCALE_REVIEW_WAITING = "atlas_scale_review_waiting_on_provider"
#: The scale review's own ordinary failures (the views were posed; no scale could be established from them):
#: ``website_atlas_pipeline._source_pose`` returns the first when the review leaves the measurements unestimated,
#: and ``atlas_scene_planner`` raises the second when the review's planner call ends with no valid answer; the
#: third is the review waiting on its provider (``SCALE_REVIEW_WAITING``). None is a wall: the planner's money and
#: release stops carry their own codes.
ORDINARY_SCALE_FAILURES = frozenset({"atlas_scale_review_unresolved", "atlas_scene_planner_output_invalid",
                                     SCALE_REVIEW_WAITING})


def ships_unscaled(source_pose: Mapping[str, Any]) -> bool:
    """Whether a background whose source pose did not complete ships unscaled: only after an ordinary failure.

    Fails closed: the stored reason must be on ``website_atlas_pose.ORDINARY_POSE_FAILURES`` (the provider failing
    to pose the views) or on ``ORDINARY_SCALE_FAILURES`` (the scale review ending without a scale), and no wall
    may be recorded on the pose (``wall``, ``website_atlas_pipeline._source_pose``). Any other code, a pose stored
    before walls were recorded included, ships nothing, whatever its words say.
    Imported when called: the company builder imports this module and must not reach the provider client."""
    from .website_atlas_pose import ORDINARY_POSE_FAILURES
    reason = source_pose.get("reason")
    return (reason in ORDINARY_POSE_FAILURES or reason in ORDINARY_SCALE_FAILURES) and not source_pose.get("wall")


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
