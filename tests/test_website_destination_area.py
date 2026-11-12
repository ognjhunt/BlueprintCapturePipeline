"""The place the sentence names is the destination and the scored, marked area (capture-447ae4f4).

Numbers are 447's retained simulator-frame geometry: the tray's start box, the table's support box,
and the region "the other end of the same table" registered from the same footage's retained
answer (capture-39ecdc97). Before this change the sealed destination lay 8 cm from the tray's centre,
inside its start footprint, and the scored disc was 5 cm.
"""
import math

import pytest

from blueprint_pipeline.task_evaluation_surface_target import marker_for_surface_target, region_clear_of_start
from blueprint_pipeline.website_destination_area import (
    DESTINATION_GUESSED, DESTINATION_NEAR_START, FARTHEST_SPOT, GROUNDED_CENTRE, NAMED_REGION, area_for_region,
    choose_destination)
from blueprint_pipeline.website_surface_target_admission import admit_surface_target, submission_relocates_destination

SUBJECT_MIN = [-0.5893061592060521, -0.5636603226565607, 0.8018946653229508]
SUBJECT_MAX = [-0.2759361106214084, -0.04253135052284709, 0.8468228006697404]
SUPPORT_MIN = [-1.0020434187791705, -1.0564821977505914, 0.7657441024069072]
SUPPORT_MAX = [0.2905184420020341, 0.5146232822199573, 0.8166057543686343]
SEALED_BEFORE = [-0.355753462504243, -0.2751595589878311, 0.8049334496059336]
REGION = {"destination_region_status": "placed_at_region_centre", "estimated_bounds_world_m": {
    "minimum": [-0.5983685821891165, -0.04074108942348598, 0.7557504422380983],
    "maximum": [0.19072455738978553, 0.4545824409036417, 0.7892787374704473]}}
DESTINATION = {"relation": "on", "visible_label": "wooden dining table", "mode": "existing_support_surface",
               "position_world_m": SEALED_BEFORE, "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
               "basis": "registered_estimated_visible_bounds_and_collider_contact"}


def _choose(region=REGION, relocation=None, **extra):
    return choose_destination(destination=DESTINATION, subject_min=SUBJECT_MIN, subject_max=SUBJECT_MAX,
                              support_min=SUPPORT_MIN, support_max=SUPPORT_MAX, region=region,
                              relocation=relocation, **extra)


def test_the_named_place_is_the_destination_and_its_area_is_scored():
    start = [(a + b) / 2 for a, b in zip(SUBJECT_MIN, SUBJECT_MAX)]
    assert math.dist(start[:2], SEALED_BEFORE[:2]) == pytest.approx(0.0818, abs=1e-3)  # what 447 sealed
    sealed, choice = _choose()
    assert choice["basis"] == NAMED_REGION and choice["flags"] == []
    goal = sealed["position_world_m"]
    bounds = REGION["estimated_bounds_world_m"]
    assert all(bounds["minimum"][i] <= goal[i] <= bounds["maximum"][i] for i in (0, 1))
    assert choice["distance_from_start_m"] >= choice["object_longest_side_m"] > 0.5
    assert not choice["inside_start_footprint"]
    # The area: larger than the 5 cm disc, within the contracts' 0.5 m, wholly off the start footprint.
    assert 0.05 < sealed["radius_m"] <= 0.5 and region_clear_of_start(SUBJECT_MIN, SUBJECT_MAX, goal, sealed["radius_m"])


def test_the_sealed_target_and_mark_are_the_same_area():
    sealed, _ = _choose()
    target = admit_surface_target(destination=sealed, support={"aabb_min_xyz": SUPPORT_MIN, "aabb_max_xyz": SUPPORT_MAX},
                                  subject_min=SUBJECT_MIN, subject_max=SUBJECT_MAX, placement_uncertainty_m=0.072,
                                  development_support=False)
    assert target["radius_m"] == pytest.approx(sealed["radius_m"])
    assert target["surface_position_world_m"][:2] == pytest.approx(sealed["position_world_m"][:2])
    marker = marker_for_surface_target(target)
    assert marker["radius_m"] == target["radius_m"] and marker["shape"] == "flat_green_disc"


def test_a_named_place_is_never_moved_away_at_submission():
    sealed, choice = _choose(relocation={"rule": "farthest_supported_footprint_on_start_surface"})
    assert choice["basis"] == NAMED_REGION and "replaced_harness_relocation" in choice
    assert not submission_relocates_destination({"destination_choice": choice}, sealed,
                                                subject_min=SUBJECT_MIN, subject_max=SUBJECT_MAX)


def test_without_place_words_the_harness_choice_keeps_the_disc_and_is_flagged():
    sealed, choice = _choose(region=None)
    assert choice["basis"] == GROUNDED_CENTRE and "radius_m" not in sealed and choice["area_radius_m"] == 0.05
    assert DESTINATION_GUESSED in choice["flags"] and DESTINATION_NEAR_START in choice["flags"]
    assert choice["inside_start_footprint"] is True
    _, moved = _choose(region=None, relocation={"rule": "farthest_supported_footprint_on_start_surface"})
    assert moved["basis"] == FARTHEST_SPOT


def test_a_place_off_the_support_falls_back_to_the_disc():
    away = {"estimated_bounds_world_m": {"minimum": [2.0, 2.0, 0.7], "maximum": [2.5, 2.5, 0.8]}}
    sealed, choice = _choose(region=away)
    assert choice["basis"] == GROUNDED_CENTRE and "radius_m" not in sealed


def test_the_surface_check_shrinks_the_area():
    sealed, _ = _choose(holds=lambda centre, radius: radius <= 0.12)
    assert 0.05 <= sealed["radius_m"] <= 0.12


def test_destination_flags_reach_the_scene_estimate_channel(tmp_path):
    import json
    from blueprint_pipeline.atlas_scene_estimates import CODES, preparation_flags
    _, choice = _choose(region=None)
    path = tmp_path / "preparation.json"
    path.write_text(json.dumps({"destination_choice": choice}))
    assert preparation_flags(path) == {DESTINATION_GUESSED, DESTINATION_NEAR_START} <= CODES


def test_an_ambiguous_registration_flag_reaches_the_scene_estimate_channel(tmp_path):
    import json
    from blueprint_pipeline.atlas_scene_estimates import CODES, preparation_flags
    path = tmp_path / "preparation.json"
    path.write_text(json.dumps({"registration_ambiguity": {"runner_up_ratio": 1.12, "decisive_ratio": 1.2}}))
    assert preparation_flags(path) == {"scene_estimate:registration_ambiguous"} <= CODES


def test_a_named_place_wholly_under_the_start_keeps_the_harness_relocation():
    """The guarantee the same-surface rule gave: a destination is never left inside the start footprint.
    A named place that is only the start's own footprint cannot hold the area; the harness's move stands."""
    under = {"estimated_bounds_world_m": {"minimum": SUBJECT_MIN, "maximum": SUBJECT_MAX}}
    relocation = {"rule": "farthest_supported_footprint_on_start_surface"}
    sealed, choice = _choose(region=under, relocation=relocation)
    assert choice["basis"] == FARTHEST_SPOT and "replaced_harness_relocation" not in choice
    assert sealed["position_world_m"] == SEALED_BEFORE and "radius_m" not in sealed


def test_the_record_says_how_the_radius_came_from_the_named_place():
    """The 0.26 m disc is the named place's size (its room plus half the tray), not a cap: the record says so."""
    sealed, choice = _choose()
    derivation = choice["area"]["radius_derivation"]
    assert derivation["limited_by"] == "named_place"
    assert derivation["named_place_size_m"] == pytest.approx([0.789, 0.495], abs=1e-3)
    assert derivation["object_longest_side_m"] == pytest.approx(0.522, abs=1e-3)
    assert derivation["largest_radius_m"] == 0.5 and derivation["smallest_radius_m"] == 0.05
    assert sealed["radius_m"] < derivation["largest_radius_m"] and "named place" in derivation["rule"]


def _sealer(subject_min, subject_max, support_min, support_max, relative_error=0.25):
    """The sealer's own admission (``admit_surface_target``), as preparation runs it ($0)."""
    def admit(destination):
        try:
            admit_surface_target(destination=destination, support={"aabb_min_xyz": support_min, "aabb_max_xyz": support_max},
                                 subject_min=subject_min, subject_max=subject_max, placement_uncertainty_m=0.072,
                                 development_support=False, relative_error=relative_error)
        except ValueError:
            return False
        return True
    return admit


@pytest.mark.parametrize("name, region_min, region_max", [
    ("far end", [0.0, -0.6], [0.4, 0.1]), ("adjacent", [-0.25, -0.6], [0.2, 0.1]),
    ("tiny-near", [-0.2, -0.3], [-0.15, -0.25])])
def test_the_sealed_area_is_always_one_the_sealer_admits(name, region_min, region_max):
    """Review 2026-11-12 (probe_area): the radius reached the start box exactly, while the sealer needs it
    strictly clear of a start enlarged by the size tolerance, so a paid build could not be scored."""
    sub_min, sub_max = [-0.589, -0.564, 0.805], [-0.276, -0.043, 0.85]
    sup_min, sup_max = [-0.9, -0.7, 0.0], [0.4, 0.1, 0.805]
    admit = _sealer(sub_min, sub_max, sup_min, sup_max)
    region = {"estimated_bounds_world_m": {"minimum": [*region_min, 0.8], "maximum": [*region_max, 0.81]}}
    destination = {"relation": "on", "visible_label": "table", "mode": "existing_support_surface",
                   "position_world_m": [0.3, 0.0, 0.805], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    without = area_for_region(region_min=region_min, region_max=region_max, support_min=sup_min,
                              support_max=sup_max, subject_min=sub_min, subject_max=sub_max)
    sealed, choice = choose_destination(destination=destination, subject_min=sub_min, subject_max=sub_max,
                                        support_min=sup_min, support_max=sup_max, region=region,
                                        relocation=None, admit=admit)
    assert admit(sealed), name  # never an area the sealer refuses
    if choice["basis"] == NAMED_REGION:
        assert sealed["radius_m"] >= 0.05
    else:
        assert choice.get("named_place_not_admitted") is True and DESTINATION_GUESSED in choice["flags"]
        assert "radius_m" not in sealed
    if without is not None and not admit({**destination, "position_world_m": [*without["centre_xy"], 0.805],
                                         "radius_m": without["radius_m"]}):
        print(name, "before: refused", without["radius_m"], "after:", choice["basis"], sealed.get("radius_m"))


def test_447s_area_passes_the_sealer_with_the_size_tolerance():
    admit = _sealer(SUBJECT_MIN, SUBJECT_MAX, SUPPORT_MIN, SUPPORT_MAX)
    sealed, choice = _choose(admit=admit)
    assert choice["basis"] == NAMED_REGION and admit(sealed) and sealed["radius_m"] >= 0.05
