from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.exact_workcell_variation_matrix import compile_variation_matrix
from blueprint_pipeline.policy_object_acquisition import (
    AcquisitionSample,
    ObservationStagingRequest,
    ObjectAcquisitionProtocol,
    assess_object_acquisition,
    compile_observation_protocol_plan,
    object_acquisition_dimension,
    stage_observation_protocol,
    main,
)
from blueprint_pipeline.policy_observation_information import DROID_INFORMATION_CONTRACTS
from tests.test_exact_workcell_variation_matrix import _request, _schedule_request
from tests.test_policy_observation_information import coordinates, information

_DIGEST = "sha256:" + "a" * 64
_CAMERAS = {"external": _DIGEST, "wrist": _DIGEST}
_BINDING = {
    "reset_digest": _DIGEST,
    "episode_id": "test.cell0.pi05",
    "target_object_id": "fixture_object",
    "episode_started_at_sim_time_s": 17.0,
}


def protocol(visibility="visible", **kwargs):
    return ObjectAcquisitionProtocol.model_validate(
        {
            "mode": "baseline_visible" if visibility == "visible" else "visual_search",
            "initial_visibility": visibility,
            "maximum_search_seconds": None if visibility == "visible" else 5.0,
            "camera_calibration_digests": _CAMERAS,
            **kwargs,
        }
    )


def sample(t=0.0, visibility="visible", **kwargs):
    return AcquisitionSample.model_validate(
        {
            "episode_id": _BINDING["episode_id"],
            "target_object_id": _BINDING["target_object_id"],
            "episode_elapsed_s": t,
            "episode_started_at_sim_time_s": _BINDING["episode_started_at_sim_time_s"],
            "observation_sim_time_s": _BINDING["episode_started_at_sim_time_s"] + t,
            "physics_step": int(t * 60),
            "reset_digest": _DIGEST,
            "visibility_source": "deterministic_simulator_segmentation",
            "cameras": {
                name: {
                    "frame_digest": canonical_digest({"camera": name, "t": t}),
                    "calibration_digest": digest,
                    "renderer_frame": int(t * 60) + 1,
                    "rendered": True,
                    "fresh": True,
                    "target_pixels": 0 if visibility == "initially_out_of_view" else 20,
                    "target_visibility": visibility,
                }
                for name, digest in _CAMERAS.items()
            },
            **kwargs,
        }
    )


def test_out_of_view_valid_only_for_explicit_search_and_acquisition_timing_recorded():
    samples = [
        sample(visibility="initially_out_of_view"),
        sample(1.0, "initially_out_of_view"),
        sample(2.0),
    ]
    with pytest.raises(ValueError, match="initial_visibility_protocol_mismatch"):
        assess_object_acquisition(protocol(), samples, **_BINDING)
    result = assess_object_acquisition(protocol("initially_out_of_view"), samples, **_BINDING)
    assert result["acquisition_status"] == "acquired"
    assert result["first_observed_acquisition_seconds"] == 2.0
    assert result["previous_observation_seconds"] == 1.0
    assert result["acquisition_is_task_success"] is False


@pytest.mark.parametrize("visibility", ["visible", "partially_occluded"])
def test_visible_and_partial_observations_record_initial_acquisition(visibility):
    result = assess_object_acquisition(
        protocol(visibility), [sample(visibility=visibility)], **_BINDING
    )
    assert result["first_observed_acquisition_seconds"] == 0.0


def test_search_timeout_preserves_unacquired_outcome_and_late_acquisition():
    initial = sample(visibility="initially_out_of_view")
    for last, expected_time in ((sample(6.0, "initially_out_of_view"), None), (sample(6.0), 6.0)):
        result = assess_object_acquisition(
            protocol("initially_out_of_view"), [initial, last], **_BINDING
        )
        assert result["acquisition_status"] == "budget_exceeded"
        assert result["first_observed_acquisition_seconds"] == expected_time


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("fresh", False, "invalid_or_stale"),
        ("rendered", False, "invalid_or_stale"),
        ("calibration_digest", "sha256:" + "b" * 64, "calibration_mismatch"),
        ("renderer_frame", 1, "renderer_frame_not_increasing"),
    ],
)
def test_search_never_relaxes_sensor_validity_even_after_acquisition(field, value, error):
    last = sample(1.0).model_dump(mode="json")
    last["cameras"]["wrist"][field] = value
    with pytest.raises(ValueError, match=error):
        assess_object_acquisition(
            protocol("initially_out_of_view"),
            [sample(visibility="initially_out_of_view"), AcquisitionSample.model_validate(last)],
            **_BINDING,
        )


def test_missing_camera_reset_mismatch_and_nonmonotonic_readback_rejected():
    for changed, error in [
        ({"cameras": {}}, "camera_set_mismatch"),
        ({"reset_digest": "sha256:" + "b" * 64}, "reset_binding_mismatch"),
        ({"physics_step": 0}, "time_not_increasing"),
        ({"episode_id": "another_episode"}, "episode_or_target_mismatch"),
        ({"target_object_id": "another_object"}, "episode_or_target_mismatch"),
    ]:
        with pytest.raises(ValueError, match=error):
            assess_object_acquisition(protocol(), [sample(), sample(1.0, **changed)], **_BINDING)


def test_search_requires_budget_and_baseline_cannot_opt_out_of_visibility():
    with pytest.raises(ValidationError, match="time_budget_required"):
        protocol("initially_out_of_view", maximum_search_seconds=None)
    with pytest.raises(ValidationError, match="baseline_requires_initial_visibility"):
        protocol("initially_out_of_view", mode="baseline_visible")


def test_small_visible_object_is_not_mislabeled_as_partial_occlusion():
    data = sample().model_dump(mode="json")
    for camera in data["cameras"].values():
        camera["target_pixels"] = 1
    with pytest.raises(ValueError, match="initial_visibility_protocol_mismatch"):
        assess_object_acquisition(
            protocol("partially_occluded"), [AcquisitionSample.model_validate(data)], **_BINDING
        )


def plan_arguments():
    request = _request()
    request["variation_dimensions"].append(
        object_acquisition_dimension(authority_digest="sha256:" + "d" * 64)
    )
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    matrix = compile_variation_matrix(request)
    schedule_request = _schedule_request(matrix)
    return {
        "matrix": matrix,
        "request": request,
        "schedule_request": schedule_request,
        "information_by_cell": {
            c["cell_id"]: information(
                setup_object_coordinates=coordinates(measurement_context_digest=c["reset_digest"])
            )
            for c in matrix["cells"]
        },
        "acquisition_by_cell": {
            c["cell_id"]: protocol(c["resolved_values"]["initial_object_visibility"])
            for c in matrix["cells"]
        },
        "adapters_by_candidate": DROID_INFORMATION_CONTRACTS,
        "preregistration_digest": schedule_request["decision_design"][
            "preregistered_experiment_digest"
        ],
    }


def test_plan_reuses_exact_harness_schedule_preserves_anchor_and_pairs_all_subjects():
    args = plan_arguments()
    original = copy.deepcopy(args["matrix"])
    plan = compile_observation_protocol_plan(**args)
    assert args["matrix"] == original
    assert plan == compile_observation_protocol_plan(**args)
    assert plan["execution_authorized"] is False
    assert plan["status"] == "staged_runtime_integration_required"
    assert len(plan["rows"]) == 400
    assert plan["cells"][0]["acquisition"]["mode"] == "baseline_visible"
    assert {c["acquisition"]["initial_visibility"] for c in plan["cells"]} == {
        "visible",
        "partially_occluded",
        "initially_out_of_view",
    }
    for cell in plan["cells"]:
        rows = [r for r in plan["rows"] if r["cell_id"] == cell["cell_id"]]
        assert len(rows) == 4
        assert (
            len(
                {
                    (r["reset_digest"], r["seed"], r["observation_configuration_digest"])
                    for r in rows
                }
            )
            == 1
        )


def test_plan_rejects_missing_cells_and_unregistered_visibility_changes():
    args = plan_arguments()
    first = args["matrix"]["cells"][0]["cell_id"]
    del args["information_by_cell"][first]
    with pytest.raises(ValueError, match="cell_coverage_mismatch"):
        compile_observation_protocol_plan(**args)
    args = plan_arguments()
    canonical = args["matrix"]["cells"][0]["cell_id"]
    args["acquisition_by_cell"][canonical] = protocol("initially_out_of_view")
    with pytest.raises(ValueError, match="canonical_anchor_must_remain_visible"):
        compile_observation_protocol_plan(**args)
    args = plan_arguments()
    cell = next(
        c
        for c in args["matrix"]["cells"]
        if c["phase"] != "canonical_anchor"
        and c["resolved_values"]["initial_object_visibility"] == "visible"
    )
    args["acquisition_by_cell"][cell["cell_id"]] = protocol("initially_out_of_view")
    with pytest.raises(ValueError, match="preregistered_resolved_dimension"):
        compile_observation_protocol_plan(**args)
    args = plan_arguments()
    cell = next(
        c
        for c in args["matrix"]["cells"]
        if c["resolved_values"]["initial_object_visibility"] == "initially_out_of_view"
    )
    args["acquisition_by_cell"][cell["cell_id"]] = protocol()
    with pytest.raises(ValueError, match="preregistered_resolved_dimension"):
        compile_observation_protocol_plan(**args)


def test_setup_measurement_cannot_be_reused_for_another_resolved_placement():
    args = plan_arguments()
    first, second = [c["cell_id"] for c in args["matrix"]["cells"][:2]]
    args["information_by_cell"][second] = args["information_by_cell"][first]
    with pytest.raises(ValueError, match="setup_resolved_reset_mismatch"):
        compile_observation_protocol_plan(**args)


def test_acquisition_clock_cannot_restart_or_change_origin():
    with pytest.raises(ValidationError, match="elapsed_time_origin_mismatch"):
        sample(2.0, observation_sim_time_s=1.0)
    with pytest.raises(ValueError, match="episode_time_origin_mismatch"):
        assess_object_acquisition(
            protocol(),
            [
                sample(),
                sample(2.0, observation_sim_time_s=102.0, episode_started_at_sim_time_s=100.0),
            ],
            **_BINDING,
        )


_EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "docs/arm_decision_proof_v1/examples/policy_object_acquisition"
)


def test_documented_staging_entrypoint_reproduces_plan_without_overwrite(tmp_path):
    request_path = _EXAMPLE / "staging_request.v1.json"
    output = tmp_path / "plan.json"
    assert main(["--request", str(request_path), "--output", str(output)]) == 0
    expected = json.loads((_EXAMPLE / "staged_plan.v1.json").read_text())
    assert json.loads(output.read_text()) == expected
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(["--request", str(request_path), "--output", str(output)])
    assert output.read_bytes() == before


def test_wrong_position_cannot_pass_even_with_rebound_reset_context():
    data = json.loads((_EXAMPLE / "staging_request.v1.json").read_text())
    config = next(iter(data["information_by_cell"].values()))
    config["setup_object_coordinates"]["position_m"][0] += 0.01
    config["setup_object_coordinates"]["measurement_digest"] = canonical_digest(
        config["setup_object_coordinates"], digest_field="measurement_digest"
    )
    with pytest.raises(ValueError, match="setup_position_readback_mismatch"):
        stage_observation_protocol(ObservationStagingRequest.model_validate(data))
