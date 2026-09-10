from types import SimpleNamespace as NS
import math

import numpy as np
import pytest

from blueprint_pipeline.native_task_arena_readback import (
    _quaternion_angle_xyzw, read_native_task_arena_scenario_parameters,
)


def read_y(expected, observed, dtype, tolerance=0.0):
    pose = np.array([[0., observed, 1., 0., 0., 0., 1.]], dtype=dtype)
    application = {
        "parameter_id": "object_start_y_delta_m", "readback_kind": "task_subject_root_position_y_m",
        "application_tolerance": tolerance, "expected_native_value": expected,
        "runtime_name": "book", "runtime_target": "book.root_pose_w.y", "unit": "m",
        "resolved_value": -.02,
    }
    built = NS(plan={"scenario": {"parameter_applications": [application]}},
               env=NS(unwrapped=NS(scene={"book": NS(data=NS(root_pose_w=pose))})),
               scene_asset_names={"book": "book"}, native_configuration_readback={})
    return read_native_task_arena_scenario_parameters(built)


@pytest.mark.parametrize("expected", [-3.461138, -3.426138])
def test_exact_float32_reset_preserves_original_request_and_reports_storage_precision(expected):
    result = read_y(expected, expected, np.float32)
    assert result["passed"] is True
    row = result["parameters"][0]
    assert row["expected_native_value"] == expected
    assert row["observed_native_value"] == float(np.float32(expected))
    assert row["absolute_error_native_unit"] == abs(float(np.float32(expected)) - expected)
    assert row["absolute_error_native_unit"] > 0
    assert row["application_tolerance_native_unit"] == 0
    comparison = row["native_storage_comparison"]
    assert comparison["expected_stored_value"] == float(np.float32(expected))
    assert comparison["absolute_error_stored_value"] == 0
    assert comparison["physical_tolerance_changed"] is False


@pytest.mark.parametrize("direction", [-np.inf, np.inf])
def test_one_float32_ulp_of_actual_drift_still_refuses(direction):
    expected = -3.461138
    drifted = np.nextafter(np.float32(expected), np.float32(direction))
    report = read_y(expected, drifted, np.float32)
    assert report["passed"] is False
    assert report["parameters"][0]["native_storage_comparison"]["absolute_error_stored_value"] > 0


def test_float64_comparison_retains_its_own_precision():
    expected = -3.461138
    assert read_y(expected, expected, np.float64)["passed"] is True
    report = read_y(expected, np.nextafter(expected, np.inf), np.float64)
    assert report["passed"] is False
    assert "native_storage_comparison" not in report["parameters"][0]


def test_existing_physical_tolerance_keeps_original_error_comparison():
    expected = -3.461138
    report = read_y(expected, expected, np.float32, tolerance=1e-9)
    row = report["parameters"][0]
    assert report["passed"] is False
    assert row["absolute_error_native_unit"] > row["application_tolerance_native_unit"]
    assert "native_storage_comparison" not in row


def test_unapplied_two_centimeter_translation_remains_a_failure():
    assert read_y(-3.461138, -3.441138, np.float32)["passed"] is False


def read_yaw(degrees, *, perturb=False, negate=False):
    half = math.radians(degrees) / 2
    expected = [0., 0., math.sin(half), math.cos(half)]
    observed = np.asarray(expected, dtype=np.float32)
    if perturb:
        observed[2] = np.nextafter(observed[2], np.float32(np.inf))
    if negate:
        observed = -observed
    pose = np.array([[0., 0., 1., *observed]], dtype=np.float32)
    application = {
        "parameter_id": "object_yaw_delta_degrees", "readback_kind": "task_subject_root_orientation_xyzw",
        "application_tolerance": 0., "expected_native_value": expected,
        "runtime_name": "book", "runtime_target": "book.root_pose_w.orientation", "unit": "degrees",
        "resolved_value": degrees,
    }
    built = NS(plan={"scenario": {"parameter_applications": [application]}},
               env=NS(unwrapped=NS(scene={"book": NS(data=NS(root_pose_w=pose))})),
               scene_asset_names={"book": "book"}, native_configuration_readback={})
    return read_native_task_arena_scenario_parameters(built)


@pytest.mark.parametrize("degrees", [7.5, -5.])
@pytest.mark.parametrize("negate", [False, True])
def test_encoded_native_orientation_and_antipode_match_exactly(degrees, negate):
    report = read_yaw(degrees, negate=negate)
    assert report["passed"] is True
    comparison = report["parameters"][0]["native_storage_comparison"]
    assert comparison["absolute_error_stored_value"] == 0
    assert comparison["physical_tolerance_changed"] is False


@pytest.mark.parametrize("degrees", [7.5, -5.])
def test_one_float32_ulp_of_orientation_drift_remains_nonzero(degrees):
    report = read_yaw(degrees, perturb=True)
    assert report["passed"] is False
    assert report["parameters"][0]["native_storage_comparison"]["absolute_error_stored_value"] > 0


def test_stable_quaternion_angle_matches_known_rotation_and_scale_invariance():
    half = math.radians(43) / 2
    q = [0., 0., math.sin(half), math.cos(half)]
    assert _quaternion_angle_xyzw([0., 0., 0., 1.], q) == pytest.approx(math.radians(43))
    assert _quaternion_angle_xyzw(q, [-2 * value for value in q]) == 0
