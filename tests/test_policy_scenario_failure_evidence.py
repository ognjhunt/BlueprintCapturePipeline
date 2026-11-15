from types import SimpleNamespace as NS

import pytest

from blueprint_pipeline import policy_scientific_reset as reset


def mismatch_readback():
    return {"passed": False, "requested_parameter_count": 1, "parameters": [{
        "parameter_id": "object_start_y_delta_m", "expected_native_value": -2.97,
        "observed_native_value": -2.95, "absolute_error_native_unit": .02,
        "application_tolerance_native_unit": 1e-5, "passed": False,
    }], "native_readback_required": True}


def test_native_scenario_refusal_retains_expected_and_observed_measurements(monkeypatch):
    from blueprint_pipeline import native_task_arena_readback
    observed = mismatch_readback()
    monkeypatch.setattr(native_task_arena_readback, "read_native_task_arena_scenario_parameters", lambda _: observed)
    built = NS(env=NS(unwrapped=NS(scene={})), scene_asset_names={}, contact_sensor_names={},
               plan={"objects": [], "scenario": {"parameter_applications": [{}]}})
    episode = NS(read_control_observation_metadata=lambda: {"calibrations": {"external": {"resolution": [64, 32]}}})
    with pytest.raises(reset.ScientificResetScenarioMismatch) as caught:
        reset.read_native_reset_channels(built, episode)
    assert str(caught.value) == "scientific_reset_scenario_application_mismatch"
    assert caught.value.channels["observed"]["scenario_parameters"] == observed
    assert "scenario_application_mismatch" in caught.value.channels["gaps"]
    # The retained evidence is detached from subsequently mutable SDK readbacks.
    observed["parameters"][0]["observed_native_value"] = 999
    assert caught.value.channels["observed"]["scenario_parameters"]["parameters"][0]["observed_native_value"] == -2.95
