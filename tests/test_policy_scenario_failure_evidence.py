from types import SimpleNamespace as NS
import json

import pytest

from blueprint_pipeline import policy_scientific_reset as reset
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


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


@pytest.mark.parametrize("standalone", [False, True])
def test_real_worker_seals_wrong_reset_before_first_policy_query(tmp_path, monkeypatch, standalone):
    from tests.test_native_task_arena_policy_canary_lifecycle_rehearsal import (
        FakeIsaac, PROVIDER_RESULT_FILENAME, _rehearsal_runtime, _sealed_result, _stage_runtime_root, worker,
    )
    from tests.test_policy_scientific_reset import snapshot
    runner = worker
    if standalone:
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("policy_scenario_worker_standalone", worker.__file__)
        runner = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, runner)
        spec.loader.exec_module(runner)
        assert not runner.__package__
    full = snapshot()
    full["observed"]["scenario_parameters"] = mismatch_readback()
    full["sources"]["scenario_parameters"] = "live_native_scenario_parameter_readback"

    def refuse(*_):
        raise reset.ScientificResetScenarioMismatch(observed=full["observed"], sources=full["sources"], gaps=[])

    monkeypatch.setattr(reset, "read_native_reset_channels", refuse)
    runtime_root, provider_output = _stage_runtime_root(tmp_path)
    child_root = provider_output / "cell_runs" / "00"
    child_root.mkdir(parents=True)
    isaac = FakeIsaac(child_root / PROVIDER_RESULT_FILENAME)
    with pytest.raises(SystemExit) as exited:
        runner._run_selected_cell(0, runtime_root=runtime_root, output_root=child_root,
            provider_output_root=provider_output, cell_runtime=_rehearsal_runtime(isaac))
    assert exited.value.code == 0 and isaac.result_sealed_at_close is True
    result = _sealed_result(child_root / PROVIDER_RESULT_FILENAME)
    assert len(result["episodes"]) == 2
    for episode in result["episodes"]:
        assert episode["candidate_policy_queried"] is False
        assert episode["typed_harness_failure"] == "ScientificResetScenarioMismatch"
    gaps = sorted((child_root / "episodes").glob("*.failure_gap.json"))
    assert len(gaps) == 2
    for path in gaps:
        evidence = json.loads(path.read_text())
        measured = reset.validate_reset_readback(evidence["scientific_reset"])
        assert measured["complete"] is False
        assert measured["observed"]["scenario_parameters"] == mismatch_readback()
        assert measured["binding"]["candidate_id"] == evidence["candidate_id"]
        assert measured["receipt_digest"] == canonical_digest(measured, digest_field="receipt_digest")
        assert evidence["first_observation_retained"] is False
        assert evidence["visual_evidence"]["media_gap"]["type"] == "before_first_observation"
