"""One target-disc radius for admission, relocation, the start rule and native bind.

capture-447ae4f4: the region held the estimated 31x52 cm tray plus 1 cm, so a
replacement static qualification admitted 3.3% larger was refused at native
bind, after the scene configuration was paid for.
"""
import math

import pytest

from blueprint_pipeline.task_evaluation_surface_target import bind_native_surface_target
from blueprint_pipeline.website_support_geometry import DESTINATION_MARGIN_M
from blueprint_pipeline.website_surface_target_admission import (
    admit_surface_target, preparation_size_tolerance, replacement_size_tolerance, website_target_radius,
)

TRAY_MIN, TRAY_MAX = [-0.155, -0.26, 0.80], [0.155, 0.26, 0.845]
TABLE = {"description": "dining table", "aabb_min_xyz": [-0.65, -0.8, 0.76], "aabb_max_xyz": [0.65, 0.8, 0.80]}


def _admit(position=(0.2, 0.35), **kwargs):
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


@pytest.mark.parametrize("scale", [1.05, 1.2])
def test_a_replacement_qualification_admits_larger_than_the_estimate_now_binds(scale):
    before = _admit()
    with pytest.raises(ValueError, match="surface_target_native_subject_does_not_fit"):
        _bind(before, scale)
    target = _admit(relative_error=0.25)
    assert target["radius_m"] == website_target_radius(TRAY_MIN, TRAY_MAX, relative_error=0.25)
    assert target["radius_m"] > before["radius_m"]
    assert _bind(target, scale)["surface_target"] == target


def test_a_replacement_past_the_qualification_tolerance_is_still_refused():
    target = _admit(relative_error=0.25)
    assert _bind(target, 1.25)["surface_target"] == target  # the largest one qualification admits
    with pytest.raises(ValueError, match="surface_target_native_subject_does_not_fit"):
        _bind(target, 1.3)


def test_the_start_rule_reads_the_larger_radius():
    # 35 cm from the tray's centre clears the old disc, not one that holds a 25% larger replacement.
    assert _admit(position=(0.0, 0.35))["radius_m"] < 0.35
    with pytest.raises(ValueError, match="surface_target_initial_overlap"):
        _admit(position=(0.0, 0.35), relative_error=0.25)


@pytest.mark.parametrize("lower, upper", [(TRAY_MIN, TRAY_MAX), ([0.1, -2.3, 0.7], [0.4, -2.21, 1.1]),
                                          ([-0.5893061592060521, -0.5636603226565607, 0.8018946653229508],
                                           [-0.2759361106214084, -0.04253135052284709, 0.8468228006697404])])
def test_a_subject_built_to_size_keeps_its_radius_and_region_bit_for_bit(lower, upper):
    # Simulator frame (Z up) and a Y-up runtime frame in its own units, exactly as before.
    assert website_target_radius(lower, upper, relative_error=0) == (
        math.hypot(*[(upper[i] - lower[i]) / 2 for i in (0, 1)]) + DESTINATION_MARGIN_M)
    mpu = 1.103344777840341
    assert website_target_radius(lower, upper, relative_error=0.0, up=1, meters_per_unit=mpu) == (
        math.hypot(*[(upper[axis] - lower[axis]) / 2 for axis in range(3) if axis != 1]) + DESTINATION_MARGIN_M / mpu)
    assert _admit(relative_error=0.0) == _admit()


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
        website_target_radius(TRAY_MIN, TRAY_MAX, relative_error=value)
