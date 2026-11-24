"""What a deployment company reads about the job's object on the company's job page (preparation status v9).

There is one delivery path (owner, 2026-11-19): the website shows every job to the deployment companies Blueprint
has approved, on the company's job page. This is the part of that page the Pipeline knows and the website does not:
the object and where it goes, with its heights above the floor; how sure each figure is (the scale band, the
registration residual, every estimate the scene rests on); and Blueprint's own estimates of what the site did not
say, the object's weight and how often the job runs (``website_job_estimates``).

It is read-only and pays for nothing. It reads what the scene handoff names: the handoff (sealed by its digest),
the preparation it names (sealed, and bound to the current task context) or, for a room package shipped on compile's
own stops, the one its sealed record names by digest (job move-the-lamp-2: the company page said nothing about the
object was estimated while telling the company to size a stand-in from those facts), the base scene it names, and the
robot fit table when one is verified (``website_robot_fit_projection``). Anything it cannot read or verify is left out (None),
so it never fails a status read.

Each line is plain words, never a price, an id, a path or a link. A line that is Blueprint's estimate of something
the site's job page asks names that question (``answers``: ``item_weight``, ``item_size_longest_side``,
``items_per_hour`` or ``hours_per_day``): the website holds the site's quick answers and shows the site's answer
instead of Blueprint's estimate of the same thing. The object's name and the place's label are the video analysis's
words, which may echo the site's sentence, so the website removes any contact detail or street address from every
line before a company sees it.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .atlas_automation_plan import SCENE_COMPLETION_NOT_ADMITTED
from .atlas_scene_estimates import (DESTINATION_OVERHANG, OBJECT_SIZE_DISAGREES_WITH_PUBLISHED,
                                    OBJECT_SIZE_FROM_ITS_POINTS, SUPPORT_GUESSED, SUPPORT_HEIGHT,
                                    preparation_record_flags)
from . import website_destination_unheld as unheld

SCHEMA_VERSION = "website_packet_facts.v1"
#: The record's bound inside the status receipt; lines are cut from the end of the estimates past it.
MAX_BYTES = 8192
MAX_LINE = 700
NOT_RECORDED = "not recorded"
INVALID = "not known"
#: Plain sentences for the scene estimate flags (``atlas_scene_estimates``).
ESTIMATE_SENTENCES = {
    "scene_estimate:level": "Which way is up was not seen in the video; it was estimated.",
    "scene_estimate:splat_view_dependence_dropped": "The room's look keeps base colours only; shine that changes with the view was dropped.",
    "scene_estimate:floor": "A flat floor stands in for an uneven or missing scanned floor.",
    "scene_estimate:collider_registration": "Where the collision surfaces sit, or which faces were kept, is an estimate.",
    "scene_estimate:surrounding_views_dropped": "Some views of the surroundings were left out.",
    "scene_estimate:task_views_left_out": "Some views of the job's area were too blurred, did not connect to the others, or were more views than the rebuild takes, and were left out; the object's start position comes from the views that were kept.",
    "scene_estimate:surroundings_not_connected": "Only the job's area was filmed in views that connect. Any of the room around it in the simulation was not filmed; it is generated from the surrounding footage.",
    "scene_estimate:surroundings_past_frame_budget": "The views of the room around the job's area were more views than the rebuild takes beside the job's area, and were left out. Any of the room around it in the simulation is generated from the footage of the job's area.",
    SUPPORT_GUESSED: "No surface lay at the object's estimated bottom: Blueprint chose the surface it rests on from the video (the one its frames show it touching, else the nearest under it) and corrected where it rests onto that surface, as the object size line says.",
    DESTINATION_OVERHANG: "No surface at the destination holds the object's whole footprint: it is placed with its centre over the surface and part of its footprint over the edge, as the overhang line says.",
    OBJECT_SIZE_FROM_ITS_POINTS: "The object's masks also took in what lies under or behind it in some frames: Blueprint estimated its size from the video's points of the object that the frames agree on, as the object size line says.",
    OBJECT_SIZE_DISAGREES_WITH_PUBLISHED: "The published size of the object's exact model disagrees with the size estimated from the video; the simulation keeps the video's estimate.",
    "scene_estimate:destination_surface_estimated": "The destination shelf or surface did not show in the scan (glass and other see-through or shiny surfaces often do not); it was estimated from its visible edges in the video and is in the collision file, not in the look.",
    "scene_estimate:task_object_guessed": "The analysis of the video could not tell for certain which object is moved; Blueprint went on with the most likely one, which the site may correct.",
    "scene_estimate:target_role_kept": "The analysis could not decide what part one object plays in the job; the part it first gave was kept.",
    "scene_estimate:task_review_estimated": "Some parts of the job were only partly filmed, such as where the object comes from or where it goes; those parts are estimates.",
    "scene_estimate:destination_guessed": "The site's sentence named no place to put the object: Blueprint chose the destination on the surface.",
    "scene_estimate:destination_near_start": "The surface left no spot at least the object's longest side away from where it starts: the destination is nearer than that.",
    "scene_estimate:registration_ambiguous": "The object's filmed shape fitted the scene almost as well in a second position; the best fit is used.",
    "scene_estimate:placement_uncertainty": "Where the object sits in the room is known only to within the registration residual near the job, given under Measured.",
    SUPPORT_HEIGHT: "The surface under the object is at a different height in the collision geometry than in the reconstruction's picture; it was not corrected, as the surface height line says.",
    "task_object_box_fit_offset": "In the frames, the centre of the box Blueprint fitted to the object is off the centre of the object's outline by more than the placement uncertainty. This is how well the box fits the object, not where it sits in the room.",
    "task_object_box_fit_size": "In the frames, the box Blueprint fitted to the object and the object's outline differ in size by more than 15 % along a side, as the object size line says: the size is less certain than the scale band alone suggests.",
    # The scene's generated gap views were not admitted (``atlas_scene_automation``, ``website_atlas_pipeline``):
    # the base scene's ``scene_completion``. The company package's README says the same words (``NOT_FILLED_IN``).
    SCENE_COMPLETION_NOT_ADMITTED: "Parts of the room the video did not show were not filled in.",
}
#: Where the gap views were not admitted nothing was generated, so the surroundings lines say only what was left out.
NOT_FILLED_IN_SENTENCES = {
    "scene_estimate:surroundings_not_connected": "Only the job's area was filmed in views that connect; the views of the room around it were left out.",
    "scene_estimate:surroundings_past_frame_budget": "The views of the room around the job's area were more views than the rebuild takes beside the job's area, and were left out.",
}
#: Estimates recorded for Blueprint's own checks that say nothing a company can act on: never shown on its page.
UNSAID = frozenset({"scene_estimate:review_citations_dropped"})
#: Whether the job's object keeps its shape when handled (``task_profile_contracts.task_object_rigidity``, on the
#: handoff as ``task_object_rigidity``), and which source said so: a model's reading, never a measurement.
RIGIDITY_WORDS = {"rigid": "rigid: it keeps its shape when handled",
                  "deformable": "soft: it does not keep its shape when handled",
                  "unknown": "not known whether it keeps its shape when handled"}
RIGIDITY_SOURCES = {"object_analysis": "Blueprint's reading of the video",
                    "label_word_backstop": "read from the object's name only"}
#: What set the scene's scale (the preparation's ``coordinate_frame.scale_authority``), in plain words.
SCALE_WORDS = {"atlas_reference_estimate": "the 3D reconstruction's own estimate",
               "registration_estimate": "fitting the video's geometry to the 3D reconstruction",
               "model_estimated_object_frame": "a model's estimate in the object's own frame",
               "model_estimated": "a model's estimate"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _cell(value: Any) -> str:
    return str(value if value not in (None, "") else NOT_RECORDED).replace("\n", " ")


def _length(value: Any) -> str:
    number = _number(value)
    if value is None:
        return NOT_RECORDED
    if number is None or number <= 0:
        return INVALID
    return f"{number * 1000:.0f} mm" if number < 0.01 else f"{number * 100:.0f} cm"


def _point(value: Any) -> str:
    if value is None:
        return NOT_RECORDED
    if isinstance(value, (list, tuple)) and len(value) in (2, 3) and all(_number(v) is not None for v in value):
        return "(" + ", ".join(f"{float(v):.3f}" for v in value) + ") m"
    return INVALID


def _range(value: Any, unit: str) -> str:
    if value is None:
        return NOT_RECORDED
    if (isinstance(value, (list, tuple)) and len(value) == 2 and all(_number(v) is not None for v in value)
            and float(value[0]) <= float(value[1])):
        return f"{float(value[0]):.3g}–{float(value[1]):.3g} {unit}".strip()
    return INVALID


def _metres(value: float) -> str:
    return f"{value:.2f} m"


def line(text: str, answers: str | None = None) -> dict[str, Any]:
    """One line of the record; ``answers`` names the site's quick question an estimate answers."""
    return {"text": text, "answers": answers}


def _size_words(prep: Mapping[str, Any]) -> str:
    """Whose points the object's size was read from, when its masks spilled (``website_object_extent``)."""
    extent = _mapping(prep.get("subject_extent"))
    before = _mapping(extent.get("before")).get("dimensions_m")
    if extent.get("status") != "from_its_own_points" or not isinstance(before, list) or len(before) != 3:
        return ""
    return (" (from its own points that the frames agree on; points its masks also took in from what lies under or "
            "behind it were left out, which made it " + " × ".join(_length(value) for value in before) + ")")


def _resting_words(prep: Mapping[str, Any]) -> str:
    """How Blueprint set the object down when no surface lay at its estimated bottom; empty when one did."""
    guess = _mapping(_mapping(prep.get("support")).get("guess"))
    heights = _mapping(_mapping(prep.get("support_search")).get("correction"))
    moved = _number(guess.get("bottom_moved_m"))
    if guess.get("correction") == "bottom_raised_to_the_surface_observed_top_kept" and moved is not None:
        return (f"; its box reached {_length(moved)} below the surface its frames show it resting on, so Blueprint "
                "raised its bottom onto that surface and kept its observed top (height "
                f"{_length(heights.get('height_before_m'))} before, {_length(heights.get('height_after_m'))} after)")
    if guess.get("correction") in ("moved_up_onto_the_surface", "moved_down_onto_the_surface") and moved:
        return (f"; Blueprint moved it {'up' if moved > 0 else 'down'} {_length(abs(moved))} onto the surface its "
                "frames show it resting on, its size unchanged")
    drop = _number(guess.get("drop_m"))
    if guess.get("basis") == "nearest_surface_under_footprint" and drop:
        return (f"; no surface lay at its estimated bottom, so Blueprint moved it {'down' if drop > 0 else 'up'} "
                f"{_length(abs(drop))} onto the nearest surface under it, its size unchanged")
    return ""


def scale_band(preparation: Mapping[str, Any] | None, fit_table: Mapping[str, Any] | None) -> tuple[float, float] | None:
    """How far the scene's sizes and distances may be off, as factors: the fit table's ``scale_band``, else the band
    ``website_robot_fit`` takes from the preparation's dimension prior f, [1 - f, 1 / (1 - f)]; None without either."""
    band = _mapping(_mapping(fit_table).get("inputs")).get("scale_band")
    if (isinstance(band, list) and len(band) == 2 and all(_number(v) is not None and v > 0 for v in band)
            and band[0] <= band[1]):
        return float(band[0]), float(band[1])
    error = _number(_mapping(_mapping(_mapping(preparation).get("authoring_inputs")).get("metric_envelope")).get(
        "maximum_dimension_relative_error"))
    return (1.0 - error, 1.0 / (1.0 - error)) if error is not None and 0 <= error < 1 else None


def _box(subject: Mapping[str, Any]) -> tuple[list[float], list[float]] | None:
    """The object's box corners in the scene's metres, or None when they are not three ordered numbers each."""
    low, high = subject.get("aabb_min_xyz"), subject.get("aabb_max_xyz")
    if (isinstance(low, list) and isinstance(high, list) and len(low) == len(high) == 3
            and all(_number(v) is not None for v in (*low, *high)) and all(a <= b for a, b in zip(low, high, strict=True))):
        return [float(v) for v in low], [float(v) for v in high]
    return None


def _across_band(sizes: list[float], band: tuple[float, float] | None) -> str:
    """`` (a × b to c × d across the scale band)`` for sizes in metres, or empty without a band."""
    if band is None or not sizes:
        return ""
    return (" (" + " × ".join(_length(size * band[0]) for size in sizes) + " to "
            + " × ".join(_length(size * band[1]) for size in sizes) + " across the scale band)")


def _support_height_words(prep: Mapping[str, Any]) -> str | None:
    """How far the object's seat in the collision geometry sits from the surface the picture shows under it where it
    starts (``website_support_height_agreement``), when that check flagged it; else None. The check's goal figure
    compares the same seat with the picture at the goal, which says nothing when the goal is another surface, so it
    is not printed."""
    agreement = _mapping(prep.get("support_height_agreement"))
    gap = _number(agreement.get("start_task_plane_minus_picture_m"))
    if SUPPORT_HEIGHT not in (agreement.get("flags") or []) or gap is None:
        return None
    worded = f"{_length(abs(gap))} {'above' if gap > 0 else 'below'}" if gap else "level with"
    return (f"Surface height under the object: where it starts, the simulation seats it {worded} the surface the "
            "reconstruction's picture shows there; not corrected.")


def _box_fit_words(prep: Mapping[str, Any]) -> str:
    """The box-fit check's median outline-to-box size ratios, when its size flag is up (``website_task_placement_check``)."""
    check = _mapping(prep.get("video_box_fit_check"))
    ratios = _mapping(_mapping(check.get("summary")).get("median_size_ratio"))
    shown = [f"{float(ratios[axis]) * 100:.0f} %" for axis in ("x", "y") if _number(ratios.get(axis)) is not None]
    if "task_object_box_fit_size" not in (check.get("flags") or []) or not shown:
        return ""
    return ("; in the frames its outline measures " + " and ".join(shown) + " of the box along its sides (median), "
            "so the box may be larger than the object")


def floor_height(base_scene: Mapping[str, Any] | None, fit_table: Mapping[str, Any] | None) -> tuple[float, str] | None:
    """The floor's height in the scene's coordinates, and where it comes from; None when nothing recorded it.

    The robot fit table's floor (the capability verdict's station, else its floor search) first; else 0 when the
    base scene was levelled on its observed floor (``atlas_world_levelling``: ``ground_plane_offset_m`` is 0 only
    then, and the preparation's coordinates keep that floor at height 0)."""
    inputs = _mapping(_mapping(fit_table).get("inputs"))
    for value in (_mapping(inputs.get("floor_height")).get("floor_z_m"),
                  _mapping(_mapping(inputs.get("floor_search")).get("result")).get("floor_z_m")):
        if _number(value) is not None:
            return float(value), "the floor next to the task, as Blueprint's robot fit check found it"
    if _number(_mapping(base_scene).get("ground_plane_offset_m")) == 0.0:
        return 0.0, "the scene's floor, which Blueprint levelled to height 0"
    return None


def _height(z: float, floor: tuple[float, str] | None, band: tuple[float, float] | None) -> str:
    """A point's height above the floor, across the scale band; its bare coordinate when the floor is unknown."""
    if floor is None:
        return f"at height {z:.2f} in the scene's coordinates (where the floor is: {NOT_RECORDED})"
    above = z - floor[0]
    if above <= 0:
        return "at or below the floor's height"
    return (f"about {_length(above)} above the floor"
            + (f" ({_length(above * band[0])} to {_length(above * band[1])} across the scale band)" if band else ""))


def _station_base(fit_table: Mapping[str, Any] | None) -> str:
    """``"30 cm square "``: the robot base the fit table's stance check placed (its recorded footprint), else empty."""
    for row in _mapping(fit_table).get("rows") or []:
        for check in _mapping(row).get("checks") or []:
            half = _mapping(_mapping(check).get("values")).get("station_footprint_half_extent_m")
            if (_mapping(check).get("check") == "stance" and isinstance(half, list) and half
                    and all(_number(v) is not None and v > 0 for v in half)):
                return f"{_length(2 * max(half))} square "
    return ""


def _holding_lines(box: tuple[list[float], list[float]], band: tuple[float, float] | None) -> list[dict[str, Any]]:
    """The object's sizes that matter for holding it, and nothing about any particular gripper: which hand or tool
    can hold it is the deployment company's to judge (owner decision 2026-11-19). Its longest side is Blueprint's
    estimate of the site's size question, on its own line."""
    from .website_job_estimates import ESTIMATE_LABEL
    plan = sorted(b - a for a, b in zip(box[0][:2], box[1][:2], strict=True))
    tall = box[1][2] - box[0][2]
    return [line(f"Sizes that matter for holding it, from the video: {_length(plan[0])} across the narrower side of "
                 f"its footprint, {_length(plan[1])} across the wider, {_length(tall)} tall."
                 + (f" Each may be {band[0]:.3g} to {band[1]:.3g} times that (the scale band)." if band else "")
                 + " No particular gripper or hand was checked against them."),
            line(f"Longest side: about {_length(max(*plan, tall))} ({ESTIMATE_LABEL}).", "item_size_longest_side")]


def object_lines(prep: Mapping[str, Any], base_scene: Mapping[str, Any] | None, fit_table: Mapping[str, Any] | None,
                 handoff: Mapping[str, Any] | None, criteria: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """The object, where it starts and where it goes, with heights above the floor and the room next to the task:
    every figure the preparation, the base scene and the fit table already hold, each with how sure it is."""
    subject, destination = _mapping(prep.get("subject")), _mapping(prep.get("destination"))
    target = _mapping(_mapping(criteria).get("surface_target"))
    place = target.get("visible_label") or destination.get("visible_label")
    centre = target.get("surface_position_world_m") or destination.get("position_world_m")
    band, floor, box = scale_band(prep, fit_table), floor_height(base_scene, fit_table), _box(subject)
    lines = [line(f"Object: {_cell(subject.get('description'))}.")] if subject.get("description") else []
    rigidity = _mapping(_mapping(handoff).get("task_object_rigidity"))
    if rigidity.get("rigidity") in RIGIDITY_WORDS:
        lines.append(line(f"Rigid or soft: {RIGIDITY_WORDS[rigidity['rigidity']]} "
                          f"({RIGIDITY_SOURCES.get(rigidity.get('source'), 'its source is not recorded')}; not checked)."))
    material = _mapping(prep.get("weight_estimate")).get("material")
    lines.append(line(f"Material: mostly {_cell(material)}, as Blueprint's video analysis read it (not checked); how "
                      "fragile it is and how it should be held: not recorded."
                      if isinstance(material, str) and material != "unknown"
                      else "Material, how fragile it is and how it should be held: not recorded."))
    if box is not None:
        lines += _holding_lines(box, band)
    start = None
    if box is not None:
        start = [(box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2, box[0][2]]
        lines.append(line(f"Starts: the middle of its base at {_point(start)}, {_height(start[2], floor, band)}."))
    goal = ([float(v) for v in centre] if isinstance(centre, (list, tuple)) and len(centre) == 3
            and all(_number(v) is not None for v in centre) else None)
    if place or centre is not None:
        lines.append(line("Goes to: " + "; ".join(item for item in (
            _cell(place) if place else "", f"centre {_point(centre)}" if centre is not None else "",
            f"within {_length(target.get('radius_m') or destination.get('radius_m'))}"
            if (target.get("radius_m") or destination.get("radius_m")) is not None else "",
            _height(goal[2], floor, band) if goal is not None else "") if item) + "."))
    inputs = _mapping(_mapping(fit_table).get("inputs"))
    if start is not None and goal is not None and inputs.get("straight_line_m") is None:
        distance = math.dist(start, goal)
        lines.append(line(f"Start to destination: about {_metres(distance)} in a straight line"
                          + (f" ({_metres(distance * band[0])} to {_metres(distance * band[1])} across the scale band)"
                             if band else "") + ", from the middle of its base to the place point."))
    residual = _number(_mapping(prep.get("coordinate_frame")).get("placement_uncertainty_m"))
    lines.append(line("Positions are in the simulation's coordinates, in metres with z up"
                      + (f"; heights are above {floor[1]}" if floor else "") + "."
                      + (f" Each may also be off by about {_length(residual)}, the registration residual near the job."
                         if residual is not None and residual > 0 else "")))
    stations = _mapping(inputs.get("stations")).get("positions_world_m")
    if isinstance(stations, list):
        base = _station_base(fit_table)
        lines.append(line(f"Room to stand: {len(stations)} clear spot(s) for a {base}robot base were found "
                          "next to the task." if stations else f"Room to stand: no clear spot for a {base}robot "
                          "base was found next to the task; the scan may have gaps there."))
    found = _mapping(_mapping(inputs.get("floor_search")).get("result"))
    lower, upper = _number(found.get("lower_half_m")), _number(found.get("upper_half_m"))
    # A wider square was "not clear" only when a search at that width ran and failed; a search the time budget
    # stopped first established nothing about it.
    refused = upper is not None and any(
        isinstance(row, Mapping) and row.get("clear") is False and _number(row.get("half_extent_m")) == upper
        for row in found.get("searches") or [])
    if band is not None and lower is not None and lower > 0:
        lines.append(line(f"Widest clear floor found next to the task: a square at least {_length(2 * lower * band[0])} "
                          "wide" + ((f"; one wider than {_length(2 * upper * band[1])} was not clear"
                                     + (" (or the scan has a gap there)" if found.get("upper_hole_possible") else ""))
                                    if refused else ", the widest searched" if upper is None
                                    else "; wider floor was not assessed") + "."))
    return lines


def measured_and_estimated(preparation: Mapping[str, Any], base_scene: Mapping[str, Any] | None,
                           fit_table: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """What was measured, and every estimate with how sure it is, Blueprint's own estimates of what the site's job
    page asks included (each naming the question it answers)."""
    from .website_job_estimates import operations_lines, weight_line
    prep = preparation
    residual = _length(_mapping(_mapping(prep.get("registration")).get("task_region")).get("trimmed_rmse_m"))
    measured = [line(f"How closely the video's geometry and the reconstruction line up near the job (registration "
                     f"residual, trimmed RMS): {residual}, in the scene's estimated metres.")]
    frame = _mapping(prep.get("coordinate_frame"))
    band = scale_band(prep, fit_table)
    authority = frame.get("scale_authority")
    estimated = [line("Scale: set by " + SCALE_WORDS.get(authority, "a method not recorded in words")
                      + (", and measured at the site" if frame.get("physical_scale_measured") is True
                         else ", not measured at the site") + ". "
                      + (f"Sizes, heights and distances may be {band[0]:.3g} to {band[1]:.3g} times those given here "
                         "(the scale band)." if band is not None else f"How far sizes may be off: {NOT_RECORDED}."))]
    inputs = _mapping(_mapping(fit_table).get("inputs"))
    subject = _mapping(prep.get("subject"))
    box = _box(subject)
    if box is not None:
        sizes = [b - a for a, b in zip(*box, strict=True)]
        estimated.append(line("Object size: " + " × ".join(_length(size) for size in sizes) + _across_band(sizes, band)
                              + ", from the video" + _size_words(prep) + _box_fit_words(prep) + _resting_words(prep)
                              + "."))
        if subject.get("complete_object_dimensions") is False:
            estimated.append(line("The object's box is fitted to the parts of it the frames show; Blueprint did not "
                                  "check that every side of it was filmed."))
    height = _support_height_words(prep)
    if height:
        estimated.append(line(height))
    overhang = _mapping(prep.get("destination_overhang"))
    reach = _number(overhang.get("largest_overhang_m"))
    if overhang.get("flag") == DESTINATION_OVERHANG and reach is not None and reach > 0:
        estimated.append(line("The object's footprint is larger than the supported part of the surface at that spot: "
                              f"about {reach * 100:.1f} cm hangs over its edge; its centre is over the surface."))
    # No spot on the destination's surface held the object: the room ships without it (``website_destination_unheld``).
    not_placed = unheld.sentence(_mapping(prep.get("destination_unheld")).get("cause"), "the surface it goes to")
    if not_placed:
        estimated.append(line(not_placed))
    weight = weight_line(prep)  # Blueprint's best estimate, labelled, never as weighed
    if weight is not None:
        estimated.append(line(weight, "item_weight"))
    bounds = _mapping(_mapping(prep.get("physics")).get("bounds"))
    if "static_friction" in bounds or "dynamic_friction" in bounds:
        estimated.append(line(f"Friction: static {_range(bounds.get('static_friction'), '')}, dynamic "
                              f"{_range(bounds.get('dynamic_friction'), '')}; prior ranges, not measured."))
    straight, path = _range(inputs.get("straight_line_m"), "m"), _range(inputs.get("carry_path_m"), "m")
    if straight not in (NOT_RECORDED, INVALID) and path not in (NOT_RECORDED, INVALID):
        estimated.append(line(f"Carry distance: {straight} in a straight line, {path} along the carry path, from the "
                              "scene's estimated geometry."))
    estimated += [line(text, answers) for text, answers in operations_lines(prep)]
    codes = {row["code"] for row in _mapping(base_scene).get("scene_estimates") or []
             if isinstance(row, Mapping) and isinstance(row.get("code"), str)}
    # Every estimate the preparation record itself carries, read as the publication reads it
    # (``atlas_scene_estimates.preparation_record_flags``): its choices, its checks and its corrections.
    codes |= preparation_record_flags(prep)
    if _mapping(_mapping(base_scene).get("scene_completion")).get("generated_geometry_rejected") \
            == SCENE_COMPLETION_NOT_ADMITTED:
        codes.add(SCENE_COMPLETION_NOT_ADMITTED)
    words = {**ESTIMATE_SENTENCES, **(NOT_FILLED_IN_SENTENCES if SCENE_COMPLETION_NOT_ADMITTED in codes else {})}
    estimated += [line(words.get(code, "Another estimate the scene rests on, not yet described in words."))
                  for code in sorted(codes - UNSAID)]
    return measured, estimated


def _inside(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str):
        return None
    path = Path(value) if Path(value).is_absolute() else root / value
    absolute = path.absolute()
    if (".." in absolute.parts or not absolute.is_relative_to(root) or absolute.is_symlink()
            or not absolute.is_file()):
        return None
    return absolute


def _bounded(value: dict[str, Any]) -> dict[str, Any]:
    for section in ("object", "measured", "estimated"):
        for row in value[section]:
            row["text"] = row["text"][:MAX_LINE]
    while len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()) > MAX_BYTES and value["estimated"]:
        value["estimated"].pop()
        value["truncated"] = True
    return value


def read_packet_facts(*, capture_root: Path, task_context_digest: str) -> dict[str, Any] | None:
    """The job's object facts for the company's job page, or None when there are none to show (no handoff yet, or
    one not bound to the current task context). Never raises."""
    from .decision_evidence_contracts import canonical_digest
    from .website_preparation_outcome import _load
    from .website_robot_fit_projection import _verified_table

    root = Path(capture_root).absolute()
    try:
        path = root / "pipeline" / "website_scene_preparation" / "handoff.json"
        if not path.exists():
            return None
        handoff = _load(path, root, "digest")
        if (handoff.get("schema_version") != "website_scene_handoff.v1" or handoff.get("capture_id") != root.name
                or handoff.get("scene_id") != root.parent.parent.name):
            return None
        if handoff.get("status") == "room_package_written" and "preparation_path" not in handoff:
            # A room package shipped for compile's own stops names the preparation compile wrote beside it, by
            # digest, in its sealed record (``website_scene_handoff.write_room_package``): the object's estimates.
            reference = _mapping(handoff.get("room_package"))
            room = _load(Path(str(reference.get("path"))), root, "inputs_digest")
            handoff = {**handoff, "preparation_path": str(path.parent / "preparation.json"),
                       "preparation_digest": room.get("preparation_digest")
                       if room["inputs_digest"] == reference.get("inputs_digest") else None}
        preparation = _load(Path(handoff["preparation_path"]), root, "digest")
        if (handoff.get("preparation_digest") != preparation["digest"]
                or _mapping(preparation.get("binding")).get("task_context_digest") != task_context_digest
                or preparation["digest"] != canonical_digest(preparation, digest_field="digest")):
            return None
        base_path = _inside(root, handoff.get("base_scene_path"))
        base_scene = json.loads(base_path.read_text()) if base_path is not None else None
        found = _verified_table(root, task_context_digest)
        table = found[1] if found is not None and isinstance(found[1], Mapping) else None
        measured, estimated = measured_and_estimated(preparation, _mapping(base_scene) or None, table)
        return _bounded({"schema_version": SCHEMA_VERSION,
                         "object": object_lines(preparation, _mapping(base_scene) or None, table, handoff),
                         "measured": measured, "estimated": estimated, "truncated": False})
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


__all__ = ["ESTIMATE_SENTENCES", "NOT_FILLED_IN_SENTENCES", "SCHEMA_VERSION", "UNSAID", "measured_and_estimated", "object_lines", "read_packet_facts"]
