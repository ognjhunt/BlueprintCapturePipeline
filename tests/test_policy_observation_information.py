from __future__ import annotations

import copy

import numpy as np
import pytest
from pydantic import ValidationError

from blueprint_pipeline.adp009d_droid_observation import build_droid_observation_from_inputs
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.policy_observation_information import (
    AdapterInformationContract,
    DROID_INFORMATION_CONTRACTS,
    ObjectCoordinates,
    PolicyObservationInformation,
    build_configured_droid_observation,
    policy_coordinate_fields,
)


def coordinates(**updates):
    value = {
        "object_id": "articulated_washer",
        "coordinate_frame_digest": "sha256:" + "2" * 64,
        "source_kind": "simulator_ground_truth",
        "source_id": "scene_setup",
        "source_contract_digest": "sha256:" + "a" * 64,
        "measurement_context_digest": "sha256:" + "a" * 64,
        "position_m": [1.0, 2.0, 3.0],
        "measured_at_s": 100.0,
        **updates,
    }
    value["measurement_digest"] = canonical_digest(value)
    return value


def information(**updates):
    return PolicyObservationInformation.model_validate(
        {
            "scene_id": "scene_840920",
            "task_id": "open_washer",
            "setup_object_coordinates": coordinates(),
            **updates,
        }
    )


def selected_coordinates(**updates):
    return {
        "source_kind": "deployment_estimated",
        "source_id": "registered_rgbd_estimator",
        "source_contract_digest": "sha256:" + "b" * 64,
        "deployment_availability_evidence_digest": "sha256:" + "c" * 64,
        "coordinate_frame_digest": "sha256:" + "d" * 64,
        "maximum_age_s": 0.5,
        **updates,
    }


def coordinate_adapter():
    return AdapterInformationContract(
        interface_id="test_coordinate_policy.v1",
        observation_keys=("object/position_m",),
        coordinate_schema="object_position_m.v1",
        coordinate_input_key="object/position_m",
    )


@pytest.mark.parametrize("candidate", list(DROID_INFORMATION_CONTRACTS))
def test_setup_coordinates_never_change_production_droid_input(candidate):
    cameras = {
        key: np.full((32, 48, 3), i, dtype=np.uint8)
        for i, key in enumerate(
            ("observation/exterior_image_1_left", "observation/wrist_image_left"), start=12
        )
    }
    inputs = {
        "joint_position": [0.0] * 7,
        "gripper_position": 0.04,
        "eef_9d": list(range(9)),
        "eef_9d_frame_provenance": {"frame": "robot_base"},
        "object_position_m": [999.0, 999.0, 999.0],
    }
    expected = build_droid_observation_from_inputs(candidate, cameras, inputs, "move the object")
    for position in ([1.0, 2.0, 3.0], [30.0, 40.0, 50.0]):
        actual = build_configured_droid_observation(
            information(setup_object_coordinates=coordinates(position_m=position)),
            candidate_id=candidate,
            camera_rgb=cameras,
            inputs=inputs,
            prompt="move the object",
        )
        assert set(actual) == set(expected)
        for key in expected:
            if isinstance(expected[key], np.ndarray):
                np.testing.assert_array_equal(actual[key], expected[key])
            else:
                assert actual[key] == expected[key]
    assert "object_position_m" not in actual


@pytest.mark.parametrize("candidate", list(DROID_INFORMATION_CONTRACTS))
def test_droid_coordinate_request_refused_before_observation_build(candidate):
    with pytest.raises(ValueError, match="object_coordinates_not_supported"):
        build_configured_droid_observation(
            information(policy_object_coordinates=selected_coordinates()),
            candidate_id=candidate,
            camera_rgb={},
            inputs={},
            prompt="move the object",
        )


def test_simulator_truth_cannot_be_declared_deployment_information():
    with pytest.raises(ValidationError):
        information(
            policy_object_coordinates=selected_coordinates(source_kind="simulator_ground_truth")
        )


@pytest.mark.parametrize("source", ["deployment_known", "deployment_estimated"])
def test_compatible_adapter_gets_only_explicit_deployment_measurement(source):
    selected = selected_coordinates(source_kind=source)
    config = information(policy_object_coordinates=selected)
    measurement = ObjectCoordinates.model_validate(
        coordinates(
            **{
                k: selected[k]
                for k in (
                    "source_kind",
                    "source_id",
                    "source_contract_digest",
                    "coordinate_frame_digest",
                )
            },
            position_m=[0.4, -0.2, 0.8],
        )
    )
    fields, receipt = policy_coordinate_fields(
        config, adapter=coordinate_adapter(), measurement=measurement, query_time_s=100.2
    )
    assert fields == {"object/position_m": [0.4, -0.2, 0.8]}
    assert receipt["policy_coordinate_measurement_digest"] == measurement.measurement_digest
    assert receipt["setup_coordinates_used_as_policy_input"] is False
    assert receipt["deployment_source_qualification_proven"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        {"source_kind": "simulator_ground_truth"},
        {"source_id": "another_source"},
        {"coordinate_frame_digest": "sha256:" + "e" * 64},
        {"object_id": "wrong_object"},
        {"measured_at_s": 90.0},
        {"measured_at_s": 101.0},
    ],
)
def test_coordinate_source_identity_frame_and_freshness_fail_closed(mutation):
    selected = selected_coordinates()
    data = coordinates(
        **{
            k: selected[k]
            for k in (
                "source_kind",
                "source_id",
                "source_contract_digest",
                "coordinate_frame_digest",
            )
        },
        **{},
    )
    data.update(mutation)
    data["measurement_digest"] = canonical_digest(data, digest_field="measurement_digest")
    with pytest.raises(ValueError, match="source_or_frame_mismatch|stale_or_future"):
        policy_coordinate_fields(
            information(policy_object_coordinates=selected),
            adapter=coordinate_adapter(),
            measurement=ObjectCoordinates.model_validate(data),
            query_time_s=100.2,
        )


def test_missing_measurement_never_falls_back_to_setup():
    with pytest.raises(ValueError, match="measurement_required"):
        policy_coordinate_fields(
            information(policy_object_coordinates=selected_coordinates()),
            adapter=coordinate_adapter(),
            measurement=None,
            query_time_s=100.2,
        )
    fields, _ = policy_coordinate_fields(
        information(), adapter=coordinate_adapter(), measurement=None, query_time_s=100.2
    )
    assert fields == {}


def test_tampering_unknown_fields_and_forged_frozen_adapter_rejected():
    data = coordinates()
    data["position_m"][0] = 123.0
    with pytest.raises(ValidationError, match="measurement_digest_mismatch"):
        ObjectCoordinates.model_validate(data)
    data = copy.deepcopy(selected_coordinates())
    data["allow_privileged_ground_truth"] = True
    with pytest.raises(ValidationError):
        information(policy_object_coordinates=data)
    forged = coordinate_adapter().model_copy(update={"interface_id": "openpi_droid.v1"})
    with pytest.raises(ValueError, match="frozen_droid_adapter_information_contract_mismatch"):
        policy_coordinate_fields(
            information(), adapter=forged, measurement=None, query_time_s=100.2
        )
