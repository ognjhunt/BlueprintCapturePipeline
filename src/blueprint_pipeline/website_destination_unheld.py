"""Why no spot on the job's destination held its object, in one vocabulary for every reader.

Beta journey step: compile's destination check (``website_destination_overhang``) searched the destination's surface
and no spot held the object, so compile stops on ``task_destination_surface_contact_required`` and the room package
ships. The search names one of these causes, compile keeps it as ``destination_unheld``, the room package's record
carries it (``website_scene_handoff``), and the package's README (``website_company_simulation``) and the company's
job page (``website_packet_facts``) say it in these words. Job move-the-lamp-2 (24 Nov): the README said no spot
"held the object's estimated footprint", but the lamp's foot, about 6.5 cm across, would have fitted on the 16 cm
glass shelf; what did not fit was its whole box, shade and all, clear of the wall behind the shelf with its centre
5 cm inside the shelf's edges.

Where the only cause is other geometry in the object's box (``IN_THE_WAY``), the object is placed all the same at the
nearest spot that holds it but for that (``website_destination_overhang._through``), flagged
``scene_estimate:destination_box_meets_room``: the package (``BOX_MEETS_ROOM_NOTE``) and the company's job page
(``BOX_MEETS_ROOM_SENTENCE``) say so in these words, and nothing says the object could not be placed.

Pure and import-free, so the readers' lanes import nothing of the geometry.
"""
from __future__ import annotations

#: Why no spot held it: no surface under its centre; no spot tested with the scorer's disc on the surface and half the
#: footprint over it; or, at every such spot tested, other geometry in its box.
NO_SURFACE = "no_surface_under_its_centre"
TOO_SMALL = "no_spot_with_the_disc_and_half_of_it_on_the_surface"
IN_THE_WAY = "other_geometry_in_its_box_wherever_it_would_rest"
CAUSES = frozenset({NO_SURFACE, TOO_SMALL, IN_THE_WAY})
#: How far inside the surface's edges the search keeps the object's centre: the scorer's disc radius
#: (``website_surface_target_admission.WEBSITE_TARGET_TOLERANCE_M``; a test holds them equal).
CENTRE_INSIDE_M = 0.05
_INSIDE = f"{CENTRE_INSIDE_M * 100:g} cm"
#: What follows "Blueprint could not place the job's object on <the surface>: ".
_WHY = {
    IN_THE_WAY: (f"at every spot it tried with the object's centre at least {_INSIDE} inside the surface's edges and at "
                 "least half of the object over the surface, the object's estimated box, which takes in the whole "
                 "object and not only its base, reached into another surface of the room."),
    TOO_SMALL: (f"no spot it tried had room for the object's centre at least {_INSIDE} inside the surface's edges with "
                "at least half of the object's estimated box over the surface."),
    NO_SURFACE: "no surface was found under the place the object was to go."}
#: Placed where its box meets the room (``IN_THE_WAY`` alone): the README of a package with the object and its rule.
BOX_MEETS_ROOM_NOTE = ("Where the success rule's goal is, the job's object's estimated box, which takes in the whole "
                       "object and not only its base, reaches into another surface of the room, as it did at every "
                       f"spot Blueprint tried with the object's centre at least {_INSIDE} inside the surface's edges "
                       "and at least half of the object over the surface. The goal is the nearest such spot all the "
                       "same; Blueprint did not test whether the object fits there.")
#: The same on the company's job page (``website_packet_facts.ESTIMATE_SENTENCES``).
BOX_MEETS_ROOM_SENTENCE = ("Wherever the object would rest on the surface it goes to, its estimated box, which takes in "
                           "the whole object and not only its base, reached into another surface of the room: the "
                           "goal is at the nearest such spot all the same, and Blueprint did not test whether the "
                           "object fits there.")


def sentence(cause: object, surface: str) -> str | None:
    """``"Blueprint could not place the job's object on <surface>: <why>."``, or None for no known cause."""
    if not isinstance(cause, str) or cause not in CAUSES:
        return None
    return f"Blueprint could not place the job's object on {surface}: {_WHY[cause]}"
