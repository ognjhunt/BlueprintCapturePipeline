"""One centre tolerance for the region, one reach for every start clearance (G3, 2026-11-06).

capture-447ae4f4: the v1 region held the estimated 31x52 cm tray plus 1 cm, so a
replacement static qualification admitted 3.3% larger was refused at native bind,
after the scene configuration was paid for; widening it to hold the largest
replacement made a 76 cm disc on a 1.3 m table. The owner switched the rule: the
tray's centre must end within the disc, so the disc is a 5 cm tolerance kept off
the start footprint, and any qualified replacement binds.
"""
import math

import pytest

from blueprint_pipeline.task_evaluation_surface_target import (
    CENTRE_RULE, SCHEMA_V2, bind_native_surface_target, region_clear_of_start,
)
from blueprint_pipeline.website_surface_target_admission import (
    WEBSITE_TARGET_TOLERANCE_M, admit_surface_target, preparation_size_tolerance, replacement_size_tolerance,
    website_target_radius, website_target_reach,
)

TRAY_MIN, TRAY_MAX = [-0.155, -0.26, 0.80], [0.155, 0.26, 0.845]
TABLE = {"description": "dining table", "aabb_min_xyz": [-0.65, -0.8, 0.76], "aabb_max_xyz": [0.65, 0.8, 0.80]}


def _admit(position=(0.2, 0.45), **kwargs):
    destination = {"relation": "on", "visible_label": "the other end of the table", "mode": "existing_support_surface",
                   "position_world_m": [*position, 0.80], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                   "basis": "registered_estimated_visible_bounds_and_collider_contact"}
    return admit_surface_target(destination=destination, support=TABLE, subject_min=TRAY_MIN, subject_max=TRAY_MAX,
                                placement_uncertainty_m=0.072, development_support=False, **kwargs)


def _bind(target, scale):
    """Bind a qualified replacement collider ``scale`` times the estimate on every axis."""
    half = [(TRAY_MAX[i] - TRAY_MIN[i]) / 2 * scale for i in range(2)]
    height = (TRAY_MAX[2] - TRAY_MIN[2]) * scale
    spec = {"interaction_affordance": {"asset_root_from_scoring_frame": {
        "position_m": [0, 0, height / 2], "orientation_xyzw": [0, 0, 0, 1]}},
        "destination_position_tolerance_m": .02, "control_frequency_hz": 15}
    static = {"observed_structure": {"collision_bounds_body_frame_m": {
        "minimum": [-half[0], -half[1], 0], "maximum": [half[0], half[1], height]}}}
    return bind_native_surface_target(task_spec=spec, target=target, static=static,
                                      support={"sage_prim_path": "/Root", "top_z_m": 0.80})


@pytest.mark.parametrize("scale", [1.0, 1.05, 1.2, 1.25, 1.3])
def test_any_qualified_replacement_binds_because_only_its_centre_is_scored(scale):
    target = _admit(relative_error=0.25)
    assert _bind(target, scale)["surface_target"] == target


@pytest.mark.parametrize("relative_error", [0.0, 0.25])
def test_the_region_is_the_centre_tolerance_whatever_the_subject(relative_error):
    target = _admit(relative_error=relative_error)
    assert target["schema_version"] == SCHEMA_V2 and target["success_rule"] == CENTRE_RULE
    assert target["radius_m"] == website_target_radius() == WEBSITE_TARGET_TOLERANCE_M == 0.05
    # The v1 disc that held the largest replacement was 76 cm across on this 1.3 m table.
    v1_radius = math.hypot(*[(TRAY_MAX[i] - TRAY_MIN[i]) / 2 * 1.25 for i in (0, 1)]) + 0.01
    assert 2 * v1_radius > 0.75 and 2 * target["radius_m"] == pytest.approx(0.10)


def test_the_start_rule_keeps_the_disc_off_the_start_footprint():
    # The tray spans y up to 0.26: a disc at 0.30 reaches under it, one at 0.32 does not.
    with pytest.raises(ValueError, match="surface_target_initial_overlap"):
        _admit(position=(0.0, 0.30))
    assert _admit(position=(0.0, 0.32))["radius_m"] == 0.05
    # A 25% larger replacement reaches 0.325: the same disc now lies under it.
    with pytest.raises(ValueError, match="surface_target_initial_overlap"):
        _admit(position=(0.0, 0.32), relative_error=0.25)
    assert _admit(position=(0.0, 0.38), relative_error=0.25)["radius_m"] == 0.05


def test_the_reach_keeps_a_relocated_disc_off_any_replacement_at_any_yaw():
    reach = website_target_reach(TRAY_MIN, TRAY_MAX, relative_error=0.25)
    assert reach == pytest.approx(math.hypot(0.155 * 1.25, 0.26 * 1.25) + 0.05)
    for angle in (0.0, 0.7, 1.9, 3.0):
        centre = [reach * 1.0001 * math.cos(angle), reach * 1.0001 * math.sin(angle), 0.80]
        scaled_min = [TRAY_MIN[i] * 1.25 for i in (0, 1)]
        scaled_max = [TRAY_MAX[i] * 1.25 for i in (0, 1)]
        assert region_clear_of_start(scaled_min, scaled_max, centre, website_target_radius())


@pytest.mark.parametrize("lower, upper", [(TRAY_MIN, TRAY_MAX), ([0.1, -2.3, 0.7], [0.4, -2.21, 1.1])])
def test_the_reach_in_a_runtime_frame_is_in_its_own_units(lower, upper):
    mpu = 1.103344777840341
    assert website_target_radius(meters_per_unit=mpu) == 0.05 / mpu
    assert website_target_reach(lower, upper, relative_error=0.0, up=1, meters_per_unit=mpu) == (
        math.hypot(*[(upper[axis] - lower[axis]) / 2 for axis in range(3) if axis != 1]) + 0.05 / mpu)


ESTIMATED = {"dimension_authority": "estimated", "metric_envelope": {"maximum_dimension_relative_error": 0.25}}


@pytest.mark.parametrize("preparation, tolerance", [
    ({"authoring_inputs": ESTIMATED}, 0.25),
    ({"authoring_inputs": ESTIMATED, "development_test": {"kind": "authored_surface_component_test"}}, 0.0),
    ({"authoring_inputs": {**ESTIMATED, "dimension_authority": "synthetic_design"}}, 0.0),
    ({"authoring_inputs": {**ESTIMATED, "dimension_authority": "published_product_specification"}}, 0.0),
    ({}, 0.0),  # recorded before the authority was carried: the region as before
])
def test_only_an_estimated_subject_widens_its_region(preparation, tolerance):
    assert preparation_size_tolerance(preparation) == tolerance


@pytest.mark.parametrize("value", [None, True, -0.1, 1.0, float("nan"), "0.25"])
def test_an_unreadable_size_tolerance_is_refused_not_guessed(value):
    with pytest.raises(ValueError, match="website_target_size_tolerance_invalid"):
        replacement_size_tolerance(dimension_authority="estimated", relative_error=value, development=False)
    with pytest.raises(ValueError, match="website_target_size_tolerance_invalid"):
        website_target_reach(TRAY_MIN, TRAY_MAX, relative_error=value)
    with pytest.raises(ValueError, match="website_target_size_tolerance_invalid"):
        _admit(relative_error=value)
