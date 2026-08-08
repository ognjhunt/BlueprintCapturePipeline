from __future__ import annotations

import copy
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_scenario_application import (
    APPLIED_PARAMETER_IDS,
    PARAMETER_RUNTIME_TARGETS,
    ScenarioApplicationError,
    apply_scenario_instance,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


SUITE_DIGEST = "sha256:" + "1" * 64
HARNESS_DIGEST = "sha256:" + "2" * 64
INSTANCE_DIGEST_FIELD = "instance_digest"


def _instance(*, factor_id: str | None = None, cousin_id: str = "approved_can") -> dict:
    parameters = {
        "camera_blur_sigma": 0.0,
        "camera_exposure_scale": 1.0,
        "camera_latency_frames": 0,
        "camera_noise_std": 0.0,
        "external_camera_extrinsic_dx_m": 0.0,
        "external_camera_extrinsic_dy_m": 0.0,
        "external_camera_extrinsic_dz_m": 0.0,
        "external_camera_focal_scale": 1.0,
        "light_color_temperature_kelvin": 5000.0,
        "light_direction_yaw_degrees": 0.0,
        "light_intensity_scale": 1.0,
        "object_contact_offset_m": 0.005,
        "object_dynamic_friction": 0.4,
        "object_height_m": 0.1694279937744141,
        "object_mass_kg": 0.355,
        "object_radius_m": 0.031094726014345042,
        "object_rest_offset_m": 0.0,
        "object_restitution": 0.1,
        "object_start_x_m": 3.4681748,
        "object_start_y_m": -3.3100837,
        "object_start_z_m": 0.5264650138348479,
        "object_static_friction": 0.5,
        "object_yaw_degrees": 0.0,
        "robot_base_x_m": 3.4681748,
        "robot_base_y_m": -2.8100837,
        "robot_base_yaw_degrees": -90.0,
        "target_x_m": 3.750152333333333,
        "target_y_m": -3.4074919,
        "target_z_m": 0.5264650138348479,
        "wrist_camera_extrinsic_dx_m": 0.0,
        "wrist_camera_extrinsic_dy_m": 0.0,
        "wrist_camera_extrinsic_dz_m": 0.0,
    }
    values = {
        "object_start_y_m": -3.3000837,
        "object_yaw_degrees": 10.0,
        "light_intensity_scale": 1.25,
        "light_color_temperature_kelvin": 6000.0,
        "external_camera_extrinsic_dx_m": 0.005,
        "wrist_camera_extrinsic_dx_m": -0.005,
        "object_mass_kg": 0.39,
        "object_dynamic_friction": 0.5,
    }
    records = []
    if factor_id is not None:
        parameters[factor_id] = values.get(factor_id, 7.0)
        records.append(
            {
                "parameter_id": factor_id,
                "resolved_value": parameters[factor_id],
                "runtime_target": PARAMETER_RUNTIME_TARGETS.get(
                    factor_id, "unsupported.native.target"
                ),
                "validity": {
                    "invalid_behavior": "reject_instance_fail_closed",
                    "native_probe_required": True,
                },
            }
        )
    cousin = _bindings()[cousin_id]
    value = {
        "schema_version": "adp009d_scenario_instance.v1",
        "program_id": "arm-decision-proof-v1",
        "suite_digest": SUITE_DIGEST,
        "harness_digest": HARNESS_DIGEST,
        "cell_id": "cell_1",
        "template_id": "template_1",
        "family": "canonical" if factor_id is None else "diagnostic",
        "partition": "qualification",
        "scored": True,
        "seed": 2026080600,
        "cell_seed_digest": "sha256:" + "3" * 64,
        "cousin_id": cousin_id,
        "cousin_digest": cousin["asset_digest"],
        "cousin_static_validation_receipt_digest": cousin.get(
            "static_validation_receipt_digest"
        ),
        "resolved_parameters": parameters,
        "factor_records": records,
        "required_controls": [
            "deterministic_scripted_positive",
            "zero_action_negative",
        ],
        "policy_neutral": True,
        "caller_asserted_success": False,
        INSTANCE_DIGEST_FIELD: "",
    }
    value[INSTANCE_DIGEST_FIELD] = canonical_digest(
        value, digest_field=INSTANCE_DIGEST_FIELD
    )
    return value


def _bindings() -> dict[str, dict]:
    return {
        "approved_can": {
            "cousin_id": "approved_can",
            "cousin_type": "canonical",
            "admission_status": "canonical_anchor",
            "asset_digest": "sha256:" + "a" * 64,
            "native_asset_path": "/runtime/assets/approved_can.usda",
            "static_validation_receipt_digest": None,
        },
        "adp009d_visual_material_cousin": {
            "cousin_id": "adp009d_visual_material_cousin",
            "cousin_type": "visual_material",
            "admission_status": "admitted_for_control_execution",
            "asset_digest": "sha256:" + "b" * 64,
            "native_asset_path": "/runtime/assets/visual.usda",
            "static_validation_receipt_digest": "sha256:" + "c" * 64,
        },
        "adp009d_geometric_cousin": {
            "cousin_id": "adp009d_geometric_cousin",
            "cousin_type": "geometric",
            "admission_status": "admitted_for_control_execution",
            "asset_digest": "sha256:" + "d" * 64,
            "native_asset_path": "/runtime/assets/geometric.usda",
            "static_validation_receipt_digest": "sha256:" + "e" * 64,
        },
    }


class FakeNativeBackend:
    def __init__(self) -> None:
        self.parameters: dict[str, float] = {}
        self.cousin: dict | None = None
        self.readback_overrides: dict[str, float] = {}
        self.supported = set(APPLIED_PARAMETER_IDS)

    def runtime_identity(self) -> dict:
        return {
            "backend": "hermetic_native_simulator_fake",
            "version": "1",
            "configuration_digest": "sha256:" + "f" * 64,
        }

    def supported_parameter_ids(self) -> set[str]:
        return set(self.supported)

    def apply_parameter(self, parameter_id: str, value: float) -> None:
        self.parameters[parameter_id] = value

    def apply_cousin(self, binding: dict) -> None:
        self.cousin = copy.deepcopy(binding)

    def read_parameter(self, parameter_id: str) -> dict:
        return {
            "value": self.readback_overrides.get(
                parameter_id, self.parameters[parameter_id]
            ),
            "source": (
                "native_simulator_state"
                if parameter_id.startswith("object_start_")
                or parameter_id == "object_yaw_degrees"
                else "native_simulator_configuration"
            ),
            "native_path": f"native/{parameter_id}",
        }

    def read_cousin(self) -> dict:
        assert self.cousin is not None
        return {
            "cousin_id": self.cousin["cousin_id"],
            "asset_digest": self.cousin["asset_digest"],
            "source": "native_simulator_configuration",
            "native_path": "native/object_asset",
        }


@pytest.mark.parametrize(
    "factor_id",
    [
        "object_start_y_m",
        "object_yaw_degrees",
        "light_intensity_scale",
        "light_color_temperature_kelvin",
        "external_camera_extrinsic_dx_m",
        "wrist_camera_extrinsic_dx_m",
        "object_mass_kg",
        "object_dynamic_friction",
    ],
)
def test_applies_each_frozen_factor_and_receipts_native_readback(
    tmp_path: Path, factor_id: str
) -> None:
    instance = _instance(factor_id=factor_id)
    backend = FakeNativeBackend()

    receipt = apply_scenario_instance(
        instance,
        backend=backend,
        admitted_cousins=_bindings(),
        expected_suite_digest=SUITE_DIGEST,
        expected_harness_digest=HARNESS_DIGEST,
        receipt_path=tmp_path / "receipt.json",
    )

    rows = {row["parameter_id"]: row for row in receipt["applied_parameters"]}
    assert rows[factor_id]["requested_value"] == instance["resolved_parameters"][factor_id]
    assert rows[factor_id]["native_readback"]["value"] == rows[factor_id]["requested_value"]
    assert rows[factor_id]["runtime_target"] == PARAMETER_RUNTIME_TARGETS[factor_id]
    assert receipt["status"] == "applied_and_native_readback_verified"
    assert receipt["instance_digest"] == instance["instance_digest"]
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    assert (tmp_path / "receipt.json").is_file()


@pytest.mark.parametrize(
    "cousin_id,cousin_type",
    [
        ("adp009d_visual_material_cousin", "visual_material"),
        ("adp009d_geometric_cousin", "geometric"),
    ],
)
def test_applies_each_admitted_cousin_and_reads_native_asset_identity(
    cousin_id: str, cousin_type: str
) -> None:
    instance = _instance(cousin_id=cousin_id)

    receipt = apply_scenario_instance(
        instance,
        backend=FakeNativeBackend(),
        admitted_cousins=_bindings(),
        expected_suite_digest=SUITE_DIGEST,
        expected_harness_digest=HARNESS_DIGEST,
    )

    assert receipt["object_asset"]["requested_binding"]["cousin_type"] == cousin_type
    assert receipt["object_asset"]["native_readback"]["cousin_id"] == cousin_id
    assert receipt["object_asset"]["native_readback"]["asset_digest"] == instance["cousin_digest"]


def test_fails_closed_when_instance_requests_unsupported_parameter() -> None:
    instance = _instance(factor_id="unregistered_parameter")

    with pytest.raises(ScenarioApplicationError) as exc:
        apply_scenario_instance(
            instance,
            backend=FakeNativeBackend(),
            admitted_cousins=_bindings(),
            expected_suite_digest=SUITE_DIGEST,
            expected_harness_digest=HARNESS_DIGEST,
        )

    assert "scenario_application_factor_unsupported:unregistered_parameter" in exc.value.errors


def test_fails_closed_when_native_backend_lacks_a_required_parameter() -> None:
    backend = FakeNativeBackend()
    backend.supported.remove("object_mass_kg")

    with pytest.raises(ScenarioApplicationError) as exc:
        apply_scenario_instance(
            _instance(),
            backend=backend,
            admitted_cousins=_bindings(),
            expected_suite_digest=SUITE_DIGEST,
            expected_harness_digest=HARNESS_DIGEST,
        )

    assert "scenario_application_native_parameter_unsupported:object_mass_kg" in exc.value.errors


def test_fails_closed_when_native_readback_differs_from_requested_value() -> None:
    backend = FakeNativeBackend()
    backend.readback_overrides["object_dynamic_friction"] = 0.401

    with pytest.raises(ScenarioApplicationError) as exc:
        apply_scenario_instance(
            _instance(),
            backend=backend,
            admitted_cousins=_bindings(),
            expected_suite_digest=SUITE_DIGEST,
            expected_harness_digest=HARNESS_DIGEST,
        )

    assert "scenario_application_native_readback_mismatch:object_dynamic_friction" in exc.value.errors


def test_fails_closed_on_unadmitted_or_digest_mismatched_cousin() -> None:
    bindings = _bindings()
    bindings["adp009d_visual_material_cousin"]["admission_status"] = "static_candidate"

    with pytest.raises(ScenarioApplicationError) as exc:
        apply_scenario_instance(
            _instance(cousin_id="adp009d_visual_material_cousin"),
            backend=FakeNativeBackend(),
            admitted_cousins=bindings,
            expected_suite_digest=SUITE_DIGEST,
            expected_harness_digest=HARNESS_DIGEST,
        )

    assert "scenario_application_cousin_not_admitted" in exc.value.errors

    instance = _instance(cousin_id="adp009d_geometric_cousin")
    instance["cousin_digest"] = "sha256:" + "0" * 64
    instance[INSTANCE_DIGEST_FIELD] = canonical_digest(
        instance, digest_field=INSTANCE_DIGEST_FIELD
    )
    with pytest.raises(ScenarioApplicationError) as exc:
        apply_scenario_instance(
            instance,
            backend=FakeNativeBackend(),
            admitted_cousins=_bindings(),
            expected_suite_digest=SUITE_DIGEST,
            expected_harness_digest=HARNESS_DIGEST,
        )
    assert "scenario_application_cousin_digest_mismatch" in exc.value.errors


def test_fails_closed_on_instance_or_frozen_identity_mismatch() -> None:
    instance = _instance()
    instance["resolved_parameters"]["object_mass_kg"] = 0.39

    with pytest.raises(ScenarioApplicationError) as exc:
        apply_scenario_instance(
            instance,
            backend=FakeNativeBackend(),
            admitted_cousins=_bindings(),
            expected_suite_digest=SUITE_DIGEST,
            expected_harness_digest=HARNESS_DIGEST,
        )
    assert "scenario_application_instance_digest_mismatch" in exc.value.errors

    with pytest.raises(ScenarioApplicationError) as exc:
        apply_scenario_instance(
            _instance(),
            backend=FakeNativeBackend(),
            admitted_cousins=_bindings(),
            expected_suite_digest="sha256:" + "9" * 64,
            expected_harness_digest=HARNESS_DIGEST,
        )
    assert "scenario_application_suite_digest_mismatch" in exc.value.errors
