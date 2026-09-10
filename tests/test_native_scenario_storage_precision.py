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


class _TorchStorageView:
    """CPU tensor-shaped readback with dtype derived from its stored array."""
    def __init__(self, values):
        self.values = values
        self.dtype = 'torch.' + str(values.dtype)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values.tolist()


class _IsaacProxyArray:
    """Pinned Isaac ProxyArray: dtype belongs to Warp; methods delegate to Torch."""
    __module__ = 'isaaclab.utils.warp.proxy_array'
    dtype = type('transformf', (), {'__module__': 'warp._src.types'})

    def __init__(self, values):
        self.torch = _TorchStorageView(values)

    def __getattr__(self, name):
        return getattr(self.torch, name)


class _WarpTransformArray:
    __module__ = 'warp._src.types'
    dtype = _IsaacProxyArray.dtype

    def __init__(self, values):
        self.values = values


def _read_backend_pose(pose, *, kind, expected, degrees=7.5):
    application = {
        'parameter_id': 'object_start_y_delta_m' if kind.endswith('position_y_m') else 'object_yaw_delta_degrees',
        'readback_kind': kind, 'application_tolerance': 0., 'expected_native_value': expected,
        'runtime_name': 'book', 'runtime_target': kind,
        'unit': 'm' if kind.endswith('position_y_m') else 'degrees',
        'resolved_value': -.02 if kind.endswith('position_y_m') else degrees,
    }
    built = NS(plan={'scenario': {'parameter_applications': [application]}},
        env=NS(unwrapped=NS(scene={'book': NS(data=NS(root_pose_w=pose))})),
        scene_asset_names={'book': 'book'}, native_configuration_readback={})
    return read_native_task_arena_scenario_parameters(built)


@pytest.mark.parametrize('backend', [_IsaacProxyArray, _WarpTransformArray])
@pytest.mark.parametrize('kind', ['task_subject_root_position_y_m', 'task_subject_root_orientation_xyzw'])
def test_retained_cells_02_03_use_converted_storage_dtype_without_relaxing_tolerance(monkeypatch, backend, kind):
    import sys
    monkeypatch.setitem(sys.modules, 'warp', NS(to_torch=lambda value: _TorchStorageView(value.values)))
    orientation = [0., 0., math.sin(math.radians(7.5)/2), math.cos(math.radians(7.5)/2)]
    values = np.asarray([[0., -3.461138, 1., *orientation]], dtype=np.float32)
    expected = -3.461138 if kind.endswith('position_y_m') else orientation
    report = _read_backend_pose(backend(values), kind=kind, expected=expected)
    row = report['parameters'][0]
    assert report['passed'] is True
    assert row['application_tolerance_native_unit'] == 0
    assert row['native_storage_comparison']['absolute_error_stored_value'] == 0
    assert row['native_storage_comparison']['physical_tolerance_changed'] is False
    assert row['native_storage_provenance']['source_backend'] == backend.__module__
    assert 'transformf' in row['native_storage_provenance']['source_dtype']
    assert row['native_storage_provenance']['storage_dtype'] == 'torch.float32'
    assert row['native_storage_provenance']['canonical_scalar_dtype'] == 'float32'
    observed_error = 1.0025024366200341e-08 if kind.endswith('position_y_m') else 8.482783936408417e-09
    assert row['absolute_error_native_unit'] == pytest.approx(observed_error, abs=1e-18)
    coordinate = 1 if kind.endswith('position_y_m') else 5
    values[0, coordinate] = np.nextafter(values[0, coordinate], np.float32(np.inf))
    changed = _read_backend_pose(backend(values), kind=kind, expected=expected)
    assert changed['passed'] is False
    assert changed['parameters'][0]['native_storage_comparison']['absolute_error_stored_value'] > 0


def test_proxy_float64_storage_is_not_guessed_float32_from_wrapper_name():
    expected = -3.461138
    values = np.asarray([[0., float(np.float32(expected)), 1., 0., 0., 0., 1.]], dtype=np.float64)
    report = _read_backend_pose(_IsaacProxyArray(values), kind='task_subject_root_position_y_m', expected=expected)
    assert report['passed'] is False
    row = report['parameters'][0]
    assert row['native_storage_provenance']['canonical_scalar_dtype'] == 'float64'
    assert 'native_storage_comparison' not in row


@pytest.mark.parametrize('cell', ['08', '09'])
def test_retained_composed_cells_preserve_precision_and_independent_parameter_checks(cell):
    """Native cell 08 combines y/light; 09 combines yaw/external camera offset."""
    yaw = -5.0
    orientation = [0., 0., math.sin(math.radians(yaw) / 2), math.cos(math.radians(yaw) / 2)]
    values = np.asarray([[0., -3.426138, 1., *orientation]], dtype=np.float32)
    translation = cell == '08'
    primary = {
        'parameter_id': 'object_start_y_delta_m' if translation else 'object_yaw_delta_degrees',
        'readback_kind': 'task_subject_root_position_y_m' if translation else 'task_subject_root_orientation_xyzw',
        'application_tolerance': 0., 'expected_native_value': -3.426138 if translation else orientation,
        'runtime_name': 'task_object', 'runtime_target': 'root_pose_w',
        'unit': 'm' if translation else 'degrees', 'resolved_value': .015 if translation else yaw,
    }
    companion = {
        'parameter_id': 'task_light_intensity_scale' if translation else 'external_camera_x_delta_m',
        'readback_kind': 'task_light_intensity_scale' if translation else 'camera_offset_position_x_m',
        'application_tolerance': 1e-6 if translation else 0.,
        'expected_native_value': 1.1 if translation else .065,
        'runtime_target': 'task_light_intensity_scale' if translation else 'camera_offset_position_x_m',
        'unit': 'ratio' if translation else 'm', 'resolved_value': 1.1 if translation else -.015,
        'camera_role': 'external',
    }
    config = {'scenario_parameters': {'task_light_intensity_scale': {'observed_intensity_scale': 1.1}},
              'cameras': {'external': {'offset_position_m': [.065, 0., 0.]}}}
    built = NS(plan={'scenario': {'parameter_applications': [primary, companion]}},
        env=NS(unwrapped=NS(scene={'task_object': NS(data=NS(root_pose_w=_IsaacProxyArray(values)))})),
        scene_asset_names={'task_object': 'task_object'}, native_configuration_readback=config)
    result = read_native_task_arena_scenario_parameters(built)
    assert result['passed'] is True
    assert result['requested_parameter_count'] == 2
    assert result['parameters'][0]['absolute_error_native_unit'] > 0
    assert result['parameters'][0]['application_tolerance_native_unit'] == 0
    assert result['parameters'][0]['native_storage_comparison']['absolute_error_stored_value'] == 0
    assert 'native_storage_comparison' not in result['parameters'][1]
    coordinate = 1 if translation else 5
    original = values[0, coordinate]
    values[0, coordinate] = np.nextafter(original, np.float32(np.inf))
    drifted = read_native_task_arena_scenario_parameters(built)
    assert not drifted['passed'] and not drifted['parameters'][0]['passed']
    assert drifted['parameters'][1]['passed']
    values[0, coordinate] = original
    if translation:
        config['scenario_parameters']['task_light_intensity_scale']['observed_intensity_scale'] += .01
    else:
        config['cameras']['external']['offset_position_m'][0] += .01
    drifted = read_native_task_arena_scenario_parameters(built)
    assert not drifted['passed'] and not drifted['parameters'][1]['passed']
    assert drifted['parameters'][0]['passed']
