"""The estimates a website scene was built on, named once, so they reach the site's result as flags.

Reconstruction steps that used to refuse now continue on a labelled estimate (2026-11-10). Each
estimate stays in its own record (levelling, mesh registration, the scene review, the view
selection); this module reads those records, gives each estimate one typed code, and keeps the list
on the base scene (``scene_estimates``). Screening planning copies the codes into its plan, and
publication v3 lists them in ``scene_feasibility.flags`` (``beta_policy_screening_publication_flags``),
where the website names each in its own sentence and withholds any derived headline band. Nothing
measured is adjusted: an estimate only adds a flag.

Codes, each ``scene_estimate:<kind>``:

- ``level``: nothing observed gave the up direction; the provider's +Y stands in, unverified.
- ``splat_view_dependence_dropped``: the levelled splat keeps its base colour only.
- ``floor``: a flat floor stands in for an uneven or missing scanned one (the levelling's
  ``fallback_floor`` or ``walked_floor``).
- ``collider_registration``: the collider's frame, or the faces kept, are an estimate.
- ``review_citations_dropped``: the scene review cited frames that could not be inputs; those
  citations were left out.
- ``surrounding_views_dropped``: surrounding views were left out to keep the frame budget.
- ``task_object_guessed``: the analysis of the video could not tell which object the robot moves;
  the job went on with its best supported candidate, which the site may correct.
- ``target_role_kept``: the analysis could not decide a target's part in the task; it kept the
  role the analysis gave it.
- ``destination_guessed``: the job's sentence named no place to put the object; the destination
  is the harness's choice on the support (``website_destination_area``).
- ``destination_near_start``: the support left no spot at least the object's longest side away
  from its start; the destination is nearer than that.
- ``registration_ambiguous``: the object's observed shape fitted the scene almost as well in a second
  pose; the best one is used and its runner-up ratio recorded (``registration_ambiguity``).
- ``task_review_estimated``: the scene review proposed its selection while naming what it could
  not see (a support or destination only partly filmed, the object's origin, the scene's state);
  those parts are estimates.

Pure apart from reading the retained JSON records named. Records written before these estimates
existed carry none of their fields and give no code.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

__all__ = ["CODES", "FIELD", "analysis_guess_flags", "base_scene_flags", "capture_scene_flags", "scene_estimates"]

FIELD = "scene_estimates"
LEVEL, APPEARANCE, FLOOR = "scene_estimate:level", "scene_estimate:splat_view_dependence_dropped", "scene_estimate:floor"
COLLIDER = "scene_estimate:collider_registration"
CITATIONS, SURROUNDING = "scene_estimate:review_citations_dropped", "scene_estimate:surrounding_views_dropped"
TASK_OBJECT_GUESSED, TARGET_ROLE_KEPT = "scene_estimate:task_object_guessed", "scene_estimate:target_role_kept"
TASK_REVIEW_ESTIMATED = "scene_estimate:task_review_estimated"
DESTINATION_GUESSED, DESTINATION_NEAR_START = "scene_estimate:destination_guessed", "scene_estimate:destination_near_start"
REGISTRATION_AMBIGUOUS = "scene_estimate:registration_ambiguous"
#: The registration's own residual near the task, above ``PLACEMENT_UNCERTAINTY_FLOOR_M``: where the
#: object sits in the room is known only to within that (7.2 cm on capture-447ae4f4).
PLACEMENT_UNCERTAINTY = "scene_estimate:placement_uncertainty"
#: The placed collision box fits the object's own masks poorly (``website_task_placement_check``).
TASK_OBJECT_BOX_FIT = "scene_estimate:task_object_box_fit"
#: The support the object rests on is at another height in the picture than in the physics
#: (``website_support_height_agreement``), and it was not corrected.
SUPPORT_HEIGHT = "scene_estimate:support_height_differs_from_picture"
#: No surface lay within the search window under the object's estimated bottom; it rests on the nearest
#: one under its footprint (``website_support_geometry.nearest_support_under``), moved by the recorded drop.
SUPPORT_GUESSED = "scene_estimate:support_guessed"
#: No catalog robot fits the job on paper (``website_robot_fit_precheck``): a note on the job, never a stop.
#: Carried as ``scene_estimate:robot_fit_none:<cause>``, the cause that refused the most candidates.
ROBOT_FIT_NONE = "scene_estimate:robot_fit_none"
#: The capability verdict abstained (``website_rigid_feasibility``: no enabled arm, no clear stance found,
#: or the object beyond reach from every stance found): the job is built and tested anyway.
CAPABILITY_UNASSESSED = "scene_estimate:capability_unassessed"
#: Below this the residual is inside what an 85 mm hand closing on the placed object still takes:
#: half its opening plus a centimetre of finger pad, rounded to 5 cm.
PLACEMENT_UNCERTAINTY_FLOOR_M = 0.05
CODES = frozenset({LEVEL, APPEARANCE, FLOOR, COLLIDER, CITATIONS, SURROUNDING, TASK_OBJECT_GUESSED,
                   TARGET_ROLE_KEPT, TASK_REVIEW_ESTIMATED, DESTINATION_GUESSED, DESTINATION_NEAR_START,
                   REGISTRATION_AMBIGUOUS, PLACEMENT_UNCERTAINTY, TASK_OBJECT_BOX_FIT, SUPPORT_HEIGHT, SUPPORT_GUESSED})


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def scene_estimates(*, levelling: Mapping[str, Any], registration: Mapping[str, Any],
                    review: Mapping[str, Any] | None = None,
                    selection: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """One row per estimate the scene rests on: its code, the record it is read from, and its evidence."""
    rows: list[dict[str, Any]] = []
    for key, code in (("level_estimate", LEVEL), ("appearance_estimate", APPEARANCE)):
        if isinstance(levelling.get(key), Mapping):
            rows.append({"code": code, "source": "levelling", "evidence": dict(levelling[key])})
    floors = {key: dict(levelling[key]) for key in ("fallback_floor", "walked_floor")
              if isinstance(levelling.get(key), Mapping)}
    if floors:
        rows.append({"code": FLOOR, "source": "levelling", "evidence": floors})
    for key in ("registration_estimate", "support_estimate"):
        if isinstance(registration.get(key), Mapping):
            rows.append({"code": COLLIDER, "source": "mesh_registration", "evidence": dict(registration[key])})
    estimated = _mapping(review).get("estimated_aspects")
    if isinstance(estimated, list) and estimated:
        rows.append({"code": TASK_REVIEW_ESTIMATED, "source": "scene_review",
                     "evidence": {"estimated_aspects": [str(item) for item in estimated]}})
    dropped = _mapping(review).get("automation_citations_dropped")
    if isinstance(dropped, list) and dropped:
        rows.append({"code": CITATIONS, "source": "scene_review", "evidence": {"frame_ids": list(dropped)}})
    from .atlas_video_inputs import OVER_FRAME_BUDGET
    left_out = [row.get("frame_id") for row in _mapping(selection).get("diagnostics") or []
                if isinstance(row, Mapping) and row.get("decision") == OVER_FRAME_BUDGET]
    if left_out:
        rows.append({"code": SURROUNDING, "source": "view_selection", "evidence": {"frame_ids": left_out}})
    return rows


def _codes(rows: Iterable[Any]) -> set[str]:
    return {row["code"] for row in rows if isinstance(row, Mapping) and row.get("code") in CODES}


def base_scene_flags(base_scene: Mapping[str, Any]) -> list[str]:
    """The estimate codes of a base scene: its ``scene_estimates``, and the floor stand-ins of one written before them."""
    rows = base_scene.get(FIELD)
    codes = _codes(rows if isinstance(rows, list) else [])
    levelling = _mapping(base_scene.get("levelling"))
    if isinstance(levelling.get("fallback_floor"), Mapping) or isinstance(levelling.get("walked_floor"), Mapping):
        codes.add(FLOOR)
    return sorted(codes)


def capture_scene_flags(capture_root: Path) -> list[str]:
    """The estimate codes of the base scene a website capture's preparation built; none when it has none."""
    from .website_preparation_revision import workspace
    path = workspace(Path(capture_root)) / "base_scene.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not path.is_symlink() else {}
    except (OSError, ValueError):
        return []
    codes = set(base_scene_flags(value) if isinstance(value, Mapping) else [])
    return sorted(codes | analysis_guess_flags(Path(capture_root)) | preparation_flags(path.parent / "preparation.json"))


def preparation_flags(path: Path) -> set[str]:
    """The estimate codes the preparation's own choices recorded (``destination_choice``)."""
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not path.is_symlink() else {}
    except (OSError, ValueError):
        return set()
    if not isinstance(value, Mapping):
        return set()
    choice = value.get("destination_choice")
    flags = list(choice.get("flags") or []) if isinstance(choice, Mapping) else []
    ambiguity = value.get("registration_ambiguity")
    if isinstance(ambiguity, Mapping):
        flags.append(REGISTRATION_AMBIGUOUS)
    residual = _mapping(value.get("coordinate_frame")).get("placement_uncertainty_m")
    if (isinstance(residual, (int, float)) and not isinstance(residual, bool)
            and residual > PLACEMENT_UNCERTAINTY_FLOOR_M):
        flags.append(PLACEMENT_UNCERTAINTY)
    if _mapping(value.get("video_box_fit_check")).get("flags"):
        flags.append(TASK_OBJECT_BOX_FIT)
    if SUPPORT_HEIGHT in (_mapping(value.get("support_height_agreement")).get("flags") or []):
        flags.append(SUPPORT_HEIGHT)
    if _mapping(_mapping(value.get("support")).get("guess")).get("flag") == SUPPORT_GUESSED:
        flags.append(SUPPORT_GUESSED)
    codes = {code for code in flags if code in CODES}
    fit = _robot_fit_none(path.parent / "native" / "robot_fit_precheck.json")
    codes |= {fit} if fit else set()
    try:
        verdict = json.loads((path.parent / "native" / "capability_verdict.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        verdict = {}
    if isinstance(verdict, Mapping) and verdict.get("status") == "abstained":
        codes.add(CAPABILITY_UNASSESSED)
    return codes


def _robot_fit_none(path: Path) -> str | None:
    """``scene_estimate:robot_fit_none:<cause>`` when the job's paper robot fit found no robot, else None."""
    import re

    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not path.is_symlink() else {}
    except (OSError, ValueError):
        return None
    hold = value.get("hold_reason") if isinstance(value, Mapping) and value.get("status") == "no_fit" else None
    match = re.fullmatch(r"website_robot_fit_none:([a-z][a-z0-9_]{0,95})", hold) if isinstance(hold, str) else None
    return f"{ROBOT_FIT_NONE}:{match.group(1)}" if match else None


def analysis_guess_flags(capture_root: Path) -> set[str]:
    """The guesses the removal analysis's uncertain targets were resolved by (``resolve_uncertain_targets``)."""
    path = Path(capture_root) / "pipeline" / "clean_plate" / "removal_plan.json"
    try:
        plan = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not path.is_symlink() else {}
    except (OSError, ValueError):
        return set()
    rows = plan.get("uncertain_target_guesses") if isinstance(plan, Mapping) else None
    known = {"task_object_guessed": TASK_OBJECT_GUESSED, "target_role_kept": TARGET_ROLE_KEPT}
    return {known[row["code"]] for row in rows or [] if isinstance(row, Mapping) and row.get("code") in known}
