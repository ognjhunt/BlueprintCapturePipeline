"""The company package's ``README.txt``, assembled from what is true of the package in front of it.

Kept apart from ``website_company_simulation``, which builds the package and calls ``readme`` once per build. The
wording itself stays there, where every other module and test reads it; this module only puts it together, in the
order the README has always had. It imports that module; that module imports this one only when it writes a README.
Reads only: never decides, never fetches, never buys.

The words for the further objects of a several-object job (``website_further_objects``) are new and live here
(``further_notes``): each one the package holds as its own part, by its prim, with how it was built and that it stands
where the video shows it, an estimate; and each its preparation built that this package does not hold. What the
preparation did with each of the plan's further objects, built or left in the scan and why, is the plan's note
(``website_build_plan_notes.plan_notes``); that the rule scores only the job's object is the object's note
(``website_half_preparation.object_note``). A package without further objects says none of it.
"""
from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

from . import website_destination_unheld as unheld
from . import website_half_preparation as half_preparation
from . import website_library_notes as library_notes
from . import website_object_variations as variations
from .atlas_unscaled_room import SCALE_NOT_MEASURED_NOTE, UNITS_LINE
from .website_company_simulation import (
    APPEARANCE_QUALITY_NOT_PASSED, DESTINATION_CONTACT_CODE, DESTINATION_OVERHANG_NOTE, FIELDS_LEFT_OUT_LINES,
    FURNITURE_REMOVED, FURNITURE_REMOVED_NO_LOOK, FURNITURE_REMOVED_NO_LOOK_ONE, FURNITURE_REMOVED_ONE,
    GENERATED_COLLISION_ONLY_NOTE, GENERATED_LOOK_AND_COLLISION_NOTE, GENERATED_NO_LOOK_NOTE, GENERATED_VIEWS_LOOK_NOTE,
    GOAL_ESTIMATED, LOOK_AS_FIRST_WRITTEN, LOOK_LEFT_OUT, NO_RULE, NOT_FILLED_IN, NOT_FOUND, NOT_LOCATED,
    NOT_PLACED_ON_ESTIMATED_SURFACE, NOT_PREPARED, NOT_PREPARED_AT_CAP, NOT_PREPARED_PIECES, NOTHING_TAKEN_OUT,
    OBJECT_NOT_PLACED, OBJECT_NOT_PREPARED, OBJECTS_IN_SCAN, OBJECTS_TAKEN_OUT, OBJECTS_TAKEN_OUT_ONE,
    PHYSICS_CHECK_WORDS, PLY_LEFT_OUT, ROOM_NOT_RUN, ROTATIONS_STORED_WRONG, SIZE_DISAGREES_WITH_PUBLISHED_NOTE,
    SIZE_FROM_ITS_POINTS_NOTE, SPENDING_CAP_REACHED, SUPPORT_NOT_HELD, SUPPORT_REGION_NOT_HELD, SURFACE_ESTIMATED_NOTE,
    SURFACE_ESTIMATED_NO_LOOK_NOTE, SURFACE_ESTIMATED_ROOM_NO_LOOK_NOTE, SURFACE_ESTIMATED_ROOM_NOTE)
from .website_half_preparation import COLLISION_ONLY, THRESHOLDS_ASSUMED, THRESHOLDS_CONFIRMED_LATER

#: A further object the package holds as its own part, by its final prim, in plain object words, and how it was built
#: (``FURTHER_HOW``, by its ``source``). Its pose is the video's, estimated: never said to be placed exactly.
FURTHER_HELD = ("The package also holds /World/{prim} ({name}): {how}. It stands where the video shows it; that "
                "position is an estimate from the video, not a measurement.")
FURTHER_HOW = {"library": ("a model of the same kind of object from a public library, picked and sized from the video "
                           "and credited under Library models; it is not the site's own object"),
               "parametric": ("a plain {shape} shape Blueprint made for it, sized from the video, with none of the "
                              "real object's detail")}
#: A further object its preparation built that this package does not hold: the half left it out (its file changed
#: since, or the scene could not take it). Nothing is said of where else it is.
FURTHER_NOT_HELD = "Built as {its} own object{s} but not in this package: {names}."


def _joined(names: Sequence[str]) -> str:
    return ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1]


def further_notes(source: Mapping[str, Any]) -> list[str]:
    """The README's notes on the further objects of the job its half's preparation built (``further_objects``, read by
    ``website_half_preparation``): one for each part the package holds (matched by its identity, at its final prim,
    in scene.usda's order), then one naming those it does not hold. None without any."""
    rows = [row for row in (source.get("further_objects") or {}).get("composed") or [] if isinstance(row, Mapping)]
    if not rows:
        return []
    notes, held = [], []
    for prim, shipped_as in library_notes.further_prims(source).items():
        row = next((row for row in rows if library_notes.matches(row, shipped_as)), None)
        if row is None or row.get("source") not in FURTHER_HOW:
            continue  # a part no record accounts for never ships (``website_library_notes.refusal``)
        held.append(row)
        how = FURTHER_HOW[str(row["source"])].format(shape=row.get("shape") or "simple")
        notes.append(FURTHER_HELD.format(prim=prim, name=row.get("name"), how=how))
    missing = [str(row.get("name")) for row in rows if not any(row is kept for kept in held)]
    if missing:
        one = len(missing) == 1
        notes.append(FURTHER_NOT_HELD.format(its="its" if one else "their", s="" if one else "s",
                                             names=_joined(missing)))
    return notes

_SAYS = {"scene_appearance": "the look file, which shows how the site looks",
         "scene_collision": "the site's surfaces for contact",
         "task_object": "the job's object, at its start position", "task_support": "the surface the object goes on",
         "replacement": "a stand-in part of the site", "job_object": "another of the job's objects, where the video shows it"}


def _counted(many: str, one: str, count: int) -> str:
    """A note with a count, in the singular for one."""
    return one if count == 1 else many.format(count=count)


def _not_prepared(piece: str, codes: Sequence[str], *, surface_estimated: bool, cause: str | None = None) -> str:
    """A room package's sentence for one piece it left out, for the codes it was left out for (and ``cause``)."""
    if all(code == SPENDING_CAP_REACHED for code in codes):
        return NOT_PREPARED_AT_CAP.format(piece=piece)
    if piece == "destination" and surface_estimated and all(code == DESTINATION_CONTACT_CODE for code in codes):
        return unheld.sentence(cause, "that estimated surface") or NOT_PLACED_ON_ESTIMATED_SURFACE
    return NOT_PREPARED.format(piece=piece)


def _stated(up: str | None, metres: float | None) -> str:
    unit = None if metres is None else f"{metres:g} {'metre' if metres == 1 else 'metres'} per unit"
    stated = [f"up axis {up}" if up else None, unit]
    return " and ".join(item for item in stated if item) or "no up axis or unit"


def readme(source: Mapping[str, Any], placed: Mapping[str, str], names: list[str],
           units: Mapping[str, tuple[str | None, float | None]],
           offsets: Mapping[str, Any] | None = None, plys: Mapping[str, str] | None = None,
           first_written: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """Only what is true of the package in front of it. ``first_written``: the look sources that could not be
    written again, so their looks ship as first written."""
    plys, first_written = plys or {}, first_written or {}
    room, withheld = source.get("room"), bool(source.get("rule_withheld"))
    what = (["A simulation of the site for this job, with the job's object in it at its",
             "start position as estimated from the video. Test in your own tools."] if withheld else
            ["A simulation of the site for this job, with the job's object in it, and the rule for what counts",
             "as the job done. Test in your own tools."] if room is None else
            ["A simulation of the site for this job, with the job's goal region and an assumed rule for reaching",
             "it. Test in your own tools."] if room["success_rule_defined"] else
            ["A simulation of the site for this job. The job, as the site described it, is on the job's page.",
             "Test in your own tools."])
    lines = ["Simulation of the site", "", "What this is", "", *what, "",
             "How the pieces fit", "",
             "- scene.usda: open this one. It places every part below at its position, in /World."]
    hidden = []
    for prim, relative in placed.items():
        row = source["files"][prim]
        says = _SAYS.get(row["role"], "a part of the site")
        visible = row["placement"]["visible"]
        lines.append(f"- {relative} (/World/{prim}): {says}." + ("" if visible else " Hidden in scene.usda."))
        if prim in plys:
            lines += [f"- {plys[prim]}: the same look as a standard 3D Gaussian splat PLY, for splat viewers that do",
                      "  not open USD. scene.usda does not use it."]
        if not visible:
            hidden.append(prim)
    if any(name.startswith("scene/files/") and name != library_notes.MODEL_FILE for name in names):
        lines.append("- scene/files/: files the parts above use.")
    if any(name.lower().endswith((".png", ".jpg", ".jpeg")) for name in names):
        lines.append("- Textures were rewritten without their colour profile and EXIF data, so a photo that relied on "
                     "an EXIF rotation shows unrotated.")
    if room is None and not withheld:
        lines += ["- success_rule.json: what counts as done, as numbers, with the run's timing and the object's",
                  "  start position and size where known."]
        if isinstance(source.get("rule"), Mapping) and source["rule"].get("criteria_basis"):
            lines += ["  criteria_basis says, for each number Blueprint set, whether the site's own words state it",
                      "  (site_stated), it is the goal area's radius (destination_area) or Blueprint's default (assumed)."]
            if half_preparation.SITE_CONFIRMED in source["rule"]["criteria_basis"].values():
                lines += ["  site_confirmed: Blueprint's default, which the site confirmed after this was first built."]
    elif room is not None and room["success_rule_defined"]:
        lines += ["- success_rule.json: the goal region (surface_target: radius_m around surface_position_world_m,",
                  "  the middle of the goal's footprint at the lowest point of it the video shows) and how long",
                  "  the robot must stay within it (settling.stable_seconds)."]
    if "success_rule.json" in names and "criteria_fields_left_out" in (source.get("rule") or {}):
        lines += FIELDS_LEFT_OUT_LINES
    lines += ["- README.txt: this file.", ""]
    lines += ["Each part's file is referenced on a child prim, /World/<part>/Asset, under the part's position and",
              "orientation in the simulation, which are on /World/<part>. The file keeps its own root transform."]
    for prim, note in sorted((offsets or {}).items()):
        lines += [f"/World/{prim}: this part's file carries its own offset (its root prim is moved or turned",
                  "inside the file" + ("; see part_notes in success_rule.json" if "success_rule.json" in names
                                       else "")
                  + "). Where a tool places a file's root at the",
                  "part's position instead of under it, this part lands somewhere else: check it."]
    for prim in hidden:
        lines += [f"/World/{prim} is hidden: it is for contact, not for display.",
                  "To see it, set its visibility to inherited (in usdview: select it and press Make Visible)."]
    unscaled = bool((room or {}).get("scale_not_measured"))  # its units relative (``atlas_unscaled_room``)
    lines += ["", "Units and axes", "", UNITS_LINE if unscaled else "- scene.usda is in metres (1 unit is 1 metre), "
              "radians and seconds.", f"- Up is +{source['axis']} in scene.usda."]
    for prim, relative in placed.items():
        lines.append(f"- {relative} states {_stated(*units[prim])}.")
    for prim, relative in plys.items():
        lines += [f"- {relative} holds the Gaussians where {placed[prim]} places them (that file's own transform",
                  f"  applied), in its frame: {_stated(*units[prim])}. A PLY states no axis, so a viewer that",
                  "  assumes another up axis shows it turned."]
    if source.get("appearance_representation") == "particlefield_3d_gaussian_splat" and not source.get("look_left_out"):
        lines += ["- The look file is a 3D Gaussian splat stored as USD particle fields. Some tools cannot show",
                  "  it; the hidden collision part holds the room's geometry."]
    lines += ["", "How it was made", "",
              "Built from one phone video of the site. Sizes, distances and positions are estimates from that",
              "video, not measurements. A result in this simulation is not a promise of what a robot does at the",
              "site, and says nothing about safety. No robot is included: bring your own."]
    notes = [*(source.get("plan_notes") or []), *([LOOK_LEFT_OUT] if source.get("look_left_out") else []),
             *([PLY_LEFT_OUT] if source.get("ply_left_out") and not source.get("look_left_out") else []),
             *([LOOK_AS_FIRST_WRITTEN, *([ROTATIONS_STORED_WRONG] if any(row["first_rotations_scrambled"]
                                                                        for row in first_written.values()) else [])]
               if first_written and not source.get("look_left_out") else []),
             *(PHYSICS_CHECK_WORDS.get(source.get("physics_check")) or ()),
             *([_counted(*((FURNITURE_REMOVED_NO_LOOK, FURNITURE_REMOVED_NO_LOOK_ONE) if source.get("look_left_out")
                           else (FURNITURE_REMOVED, FURNITURE_REMOVED_ONE)), source["removed_items"])]
               if source.get("removed_items") else []),
             *([APPEARANCE_QUALITY_NOT_PASSED] if source.get("appearance_quality") else []),
             *([GENERATED_VIEWS_LOOK_NOTE] if source.get("look_views") and not source.get("look_left_out") else []),
             *([GENERATED_NO_LOOK_NOTE if source.get("look_left_out")
                else GENERATED_COLLISION_ONLY_NOTE if source["generated_patch"] == COLLISION_ONLY
                else GENERATED_LOOK_AND_COLLISION_NOTE] if source.get("generated_patch") else []),
             # A package with the job's object in it, from its base scene (a room package says it with its room notes).
             *([NOT_FILLED_IN] if source.get("not_filled_in") and room is None else []),
             *([(SURFACE_ESTIMATED_ROOM_NO_LOOK_NOTE if source.get("look_left_out") else SURFACE_ESTIMATED_ROOM_NOTE)
                if room is not None else
                (SURFACE_ESTIMATED_NO_LOOK_NOTE if source.get("look_left_out") else SURFACE_ESTIMATED_NOTE)]
               if source.get("surface_estimated") else []),
             *([SIZE_FROM_ITS_POINTS_NOTE] if source.get("size_from_its_points") else []),
             *([SIZE_DISAGREES_WITH_PUBLISHED_NOTE] if source.get("size_disagrees_with_published") else []),
             *([DESTINATION_OVERHANG_NOTE] if source.get("destination_overhang") else []),
             *([unheld.BOX_MEETS_ROOM_NOTE] if source.get("destination_box_meets_room") else []),
             *([SUPPORT_REGION_NOT_HELD if source["support_not_held"] == "region" else SUPPORT_NOT_HELD]
               if source.get("support_not_held") and not withheld else []),
             *([ROOM_NOT_RUN, GOAL_ESTIMATED if room["success_rule_assumed"] else NO_RULE,
                *([NOT_FOUND] if room.get("not_found") else []),
                *([NOT_LOCATED] if room.get("not_located") else []),
                *([OBJECT_NOT_PREPARED] if room.get("object_not_prepared") else []),
                *([OBJECT_NOT_PLACED] if room.get("object_not_placed") else []),
                *[_not_prepared(piece, [code for named, code in room["not_prepared"] if named == piece],
                                surface_estimated=bool(source.get("surface_estimated")), cause=room.get("destination_unheld"))
                  for piece in NOT_PREPARED_PIECES if any(named == piece for named, _code in room.get("not_prepared") or [])],
                *([OBJECTS_IN_SCAN if room["objects_in_scan"]
                   else _counted(OBJECTS_TAKEN_OUT, OBJECTS_TAKEN_OUT_ONE, room["objects"])]
                  if room.get("objects") else []),
                *([NOTHING_TAKEN_OUT] if room.get("nothing_taken_out") else []),
                *([NOT_FILLED_IN] if room.get("not_filled_in") else []), *([SCALE_NOT_MEASURED_NOTE] if unscaled else [])]
               if room is not None else []), *([NO_RULE] if withheld else []),
             *half_preparation.object_note(source), *further_notes(source),  # and the job's further objects
             *half_preparation.estimates_note(source),  # where the mass range comes from; friction's default
             *(([THRESHOLDS_CONFIRMED_LATER] if source.get("thresholds_confirmed_later") else [THRESHOLDS_ASSUMED])
               if source.get("thresholds_assumed") and source.get("rule") and room is None else [])]
    # One bullet per note, under its own heading, so a reader can pick out the ones that matter.
    lines += ["", "Notes on this simulation", ""] if notes else []
    for note in notes:
        lines += textwrap.wrap(note, width=100, initial_indent="- ", subsequent_indent="  ",
                               break_long_words=False, break_on_hyphens=False)
    lines += [*variations.readme_lines(source), *library_notes.readme_lines(source), "", "Terms", "",
              "Use this package only to decide whether you can do this job. Do not pass it on and do not use it",
              "to train software. Blueprint may ask you to delete it; then delete every copy.", ""]
    return "\n".join(lines)


__all__ = ["FURTHER_HELD", "FURTHER_HOW", "FURTHER_NOT_HELD", "further_notes", "readme"]
