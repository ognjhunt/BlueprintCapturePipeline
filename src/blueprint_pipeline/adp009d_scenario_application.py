"""Fail-closed native application of one resolved ADP-009D scenario instance.

Scenario materialization selects values.  This module is the distinct boundary
that applies those values through a simulator-owned adapter and accepts them
only after reading the resulting native configuration or state back.  An
adapter may not use its request payload as readback evidence.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Set
from pathlib import Path
from typing import Any, Protocol

try:  # flat provider-bundle layout
    from common import write_json
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .common import write_json
    from .decision_evidence_contracts import canonical_digest


APPLICATION_PLAN_SCHEMA_VERSION = "adp009d_scenario_application_plan.v1"
APPLICATION_RECEIPT_SCHEMA_VERSION = "adp009d_scenario_application_receipt.v1"
SCENARIO_INSTANCE_SCHEMA_VERSION = "adp009d_scenario_instance.v1"
PROGRAM_ID = "arm-decision-proof-v1"

# These values are authored into the simulator, rather than merely carried to
# the task plan or scorer.  Camera deltas retain their frozen frame semantics:
# external is world-frame x and wrist is mount-local x.
APPLIED_PARAMETER_IDS = (
    "object_start_x_m",
    "object_start_y_m",
    "object_start_z_m",
    "object_yaw_degrees",
    "light_intensity_scale",
    "light_color_temperature_kelvin",
    "external_camera_extrinsic_dx_m",
    "external_camera_extrinsic_dy_m",
    "external_camera_extrinsic_dz_m",
    "wrist_camera_extrinsic_dx_m",
    "wrist_camera_extrinsic_dy_m",
    "wrist_camera_extrinsic_dz_m",
    "object_mass_kg",
    "object_dynamic_friction",
)

PARAMETER_RUNTIME_TARGETS = {
    "object_start_x_m": "EventManager.reset.object_start_position_m.x",
    "object_start_y_m": "EventManager.reset.object_start_position_m.y",
    "object_start_z_m": "EventManager.reset.object_start_position_m.z",
    "object_yaw_degrees": "EventManager.reset.object_orientation.yaw",
    "light_intensity_scale": "EventManager.reset.task_light.intensity_scale",
    "light_color_temperature_kelvin": (
        "EventManager.reset.task_light.color_temperature_kelvin"
    ),
    "external_camera_extrinsic_dx_m": (
        "EventManager.reset.external_camera.pose.position.x"
    ),
    "external_camera_extrinsic_dy_m": (
        "EventManager.reset.external_camera.pose.position.y"
    ),
    "external_camera_extrinsic_dz_m": (
        "EventManager.reset.external_camera.pose.position.z"
    ),
    "wrist_camera_extrinsic_dx_m": (
        "EventManager.reset.wrist_camera.pose.position.x"
    ),
    "wrist_camera_extrinsic_dy_m": (
        "EventManager.reset.wrist_camera.pose.position.y"
    ),
    "wrist_camera_extrinsic_dz_m": (
        "EventManager.reset.wrist_camera.pose.position.z"
    ),
    "object_mass_kg": "EventManager.reset.object_rigid_body.mass_kg",
    "object_dynamic_friction": (
        "EventManager.reset.object_material.dynamic_friction"
    ),
}

# These resolved fields have an existing owner (task planning, policy sensor
# processing, scoring, or canonical robot configuration).  They may be carried
# through this seam but are deliberately not claimed as applied here.  Any
# field outside the applied and pass-through sets is unsupported and rejected.
PASSTHROUGH_PARAMETER_IDS = {
    "camera_blur_sigma",
    "camera_exposure_scale",
    "camera_latency_frames",
    "camera_noise_std",
    "external_camera_focal_scale",
    "light_direction_yaw_degrees",
    "object_contact_offset_m",
    "object_height_m",
    "object_radius_m",
    "object_rest_offset_m",
    "object_restitution",
    "object_static_friction",
    "robot_base_x_m",
    "robot_base_y_m",
    "robot_base_yaw_degrees",
    "target_x_m",
    "target_y_m",
    "target_z_m",
}

FROZEN_FACTOR_IDS = {
    "object_start_y_m",
    "object_yaw_degrees",
    "light_intensity_scale",
    "light_color_temperature_kelvin",
    "external_camera_extrinsic_dx_m",
    "wrist_camera_extrinsic_dx_m",
    "object_mass_kg",
    "object_dynamic_friction",
}

NATIVE_READBACK_SOURCES = {
    "native_simulator_configuration",
    "native_simulator_state",
}
ALLOWED_COUSIN_TYPES = {"canonical", "visual_material", "geometric"}


class ScenarioApplicationError(ValueError):
    """Typed fail-closed scenario-application failure."""

    def __init__(self, errors: list[str] | tuple[str, ...]):
        self.errors = tuple(dict.fromkeys(str(error) for error in errors))
        super().__init__("; ".join(self.errors))


class NativeScenarioBackend(Protocol):
    """Small simulator-owned boundary used by the reusable application seam."""

    def runtime_identity(self) -> Mapping[str, Any]: ...

    def supported_parameter_ids(self) -> Set[str]: ...

    def apply_parameter(self, parameter_id: str, value: float) -> None: ...

    def apply_cousin(self, binding: Mapping[str, Any]) -> None: ...

    def read_parameter(self, parameter_id: str) -> Mapping[str, Any]: ...

    def read_cousin(self) -> Mapping[str, Any]: ...


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _digest(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    suffix = value.removeprefix("sha256:")
    return len(suffix) == 64 and all(char in "0123456789abcdef" for char in suffix)


def _validate_parameter_value(parameter_id: str, value: Any) -> str | None:
    number = _number(value)
    if number is None:
        return f"scenario_application_parameter_nonfinite:{parameter_id}"
    if parameter_id == "light_intensity_scale" and number <= 0:
        return "scenario_application_light_intensity_nonpositive"
    if parameter_id == "light_color_temperature_kelvin" and not 1000 <= number <= 20000:
        return "scenario_application_light_temperature_invalid"
    if parameter_id == "object_mass_kg" and number <= 0:
        return "scenario_application_mass_nonpositive"
    if parameter_id == "object_dynamic_friction" and not 0 <= number <= 2:
        return "scenario_application_dynamic_friction_invalid"
    if "camera_extrinsic_d" in parameter_id and abs(number) > 0.05:
        return f"scenario_application_camera_offset_invalid:{parameter_id}"
    return None


def build_scenario_application_plan(
    scenario_instance: Mapping[str, Any],
    *,
    admitted_cousins: Mapping[str, Mapping[str, Any]],
    expected_suite_digest: str,
    expected_harness_digest: str,
) -> dict[str, Any]:
    """Validate one resolved instance and bind its exact native application."""

    try:
        instance = json.loads(json.dumps(scenario_instance))
    except (TypeError, ValueError) as exc:
        raise ScenarioApplicationError(["scenario_application_instance_not_json"]) from exc
    if not isinstance(instance, dict):
        raise ScenarioApplicationError(["scenario_application_instance_not_mapping"])

    errors: list[str] = []
    if instance.get("schema_version") != SCENARIO_INSTANCE_SCHEMA_VERSION:
        errors.append("scenario_application_instance_schema_invalid")
    if instance.get("program_id") != PROGRAM_ID:
        errors.append("scenario_application_program_invalid")
    if instance.get("suite_digest") != expected_suite_digest:
        errors.append("scenario_application_suite_digest_mismatch")
    if instance.get("harness_digest") != expected_harness_digest:
        errors.append("scenario_application_harness_digest_mismatch")
    if instance.get("instance_digest") != canonical_digest(
        instance, digest_field="instance_digest"
    ):
        errors.append("scenario_application_instance_digest_mismatch")
    if instance.get("policy_neutral") is not True:
        errors.append("scenario_application_instance_not_policy_neutral")
    if instance.get("caller_asserted_success") is not False:
        errors.append("scenario_application_caller_success_forbidden")
    if not isinstance(instance.get("seed"), int) or isinstance(instance.get("seed"), bool):
        errors.append("scenario_application_seed_invalid")

    parameters = _mapping(instance.get("resolved_parameters"))
    missing = [parameter_id for parameter_id in APPLIED_PARAMETER_IDS if parameter_id not in parameters]
    errors.extend(
        f"scenario_application_parameter_missing:{parameter_id}" for parameter_id in missing
    )
    unknown = set(parameters) - set(APPLIED_PARAMETER_IDS) - PASSTHROUGH_PARAMETER_IDS
    errors.extend(
        f"scenario_application_parameter_unsupported:{parameter_id}"
        for parameter_id in sorted(unknown)
    )
    for parameter_id in APPLIED_PARAMETER_IDS:
        if parameter_id in parameters:
            error = _validate_parameter_value(parameter_id, parameters[parameter_id])
            if error:
                errors.append(error)

    factor_records = _rows(instance.get("factor_records"))
    if len(factor_records) != len(instance.get("factor_records") or []):
        errors.append("scenario_application_factor_records_invalid")
    factor_ids: list[str] = []
    for record in factor_records:
        factor_id = str(record.get("parameter_id") or "")
        factor_ids.append(factor_id)
        if factor_id not in FROZEN_FACTOR_IDS:
            errors.append(f"scenario_application_factor_unsupported:{factor_id}")
            continue
        if record.get("runtime_target") != PARAMETER_RUNTIME_TARGETS[factor_id]:
            errors.append(f"scenario_application_runtime_target_mismatch:{factor_id}")
        if record.get("resolved_value") != parameters.get(factor_id):
            errors.append(f"scenario_application_factor_value_mismatch:{factor_id}")
        validity = _mapping(record.get("validity"))
        if validity.get("invalid_behavior") != "reject_instance_fail_closed":
            errors.append(f"scenario_application_factor_not_fail_closed:{factor_id}")
        if validity.get("native_probe_required") is not True:
            errors.append(f"scenario_application_factor_native_probe_missing:{factor_id}")
    if len(factor_ids) != len(set(factor_ids)):
        errors.append("scenario_application_factor_duplicate")

    cousin_id = str(instance.get("cousin_id") or "")
    binding = _mapping(admitted_cousins.get(cousin_id))
    if not binding:
        errors.append("scenario_application_cousin_unrecognized")
    else:
        if binding.get("cousin_id") != cousin_id:
            errors.append("scenario_application_cousin_binding_id_mismatch")
        cousin_type = binding.get("cousin_type")
        if cousin_type not in ALLOWED_COUSIN_TYPES:
            errors.append("scenario_application_cousin_type_invalid")
        expected_admission = (
            "canonical_anchor" if cousin_type == "canonical" else "admitted_for_control_execution"
        )
        if binding.get("admission_status") != expected_admission:
            errors.append("scenario_application_cousin_not_admitted")
        if not _digest(binding.get("asset_digest")):
            errors.append("scenario_application_cousin_asset_digest_invalid")
        if instance.get("cousin_digest") != binding.get("asset_digest"):
            errors.append("scenario_application_cousin_digest_mismatch")
        expected_static_receipt = binding.get("static_validation_receipt_digest")
        if cousin_type == "canonical":
            if expected_static_receipt is not None:
                errors.append("scenario_application_canonical_static_receipt_unexpected")
        else:
            if not _digest(expected_static_receipt):
                errors.append("scenario_application_cousin_static_receipt_invalid")
            if instance.get("cousin_static_validation_receipt_digest") != expected_static_receipt:
                errors.append("scenario_application_cousin_static_receipt_mismatch")
        if not str(binding.get("native_asset_path") or ""):
            errors.append("scenario_application_cousin_native_asset_path_missing")

    if errors:
        raise ScenarioApplicationError(errors)

    plan = {
        "schema_version": APPLICATION_PLAN_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "suite_digest": instance["suite_digest"],
        "harness_digest": instance["harness_digest"],
        "instance_digest": instance["instance_digest"],
        "cell_id": instance.get("cell_id"),
        "seed": instance["seed"],
        "factor_parameter_ids": factor_ids,
        "parameters": {
            parameter_id: float(parameters[parameter_id])
            for parameter_id in APPLIED_PARAMETER_IDS
        },
        "runtime_targets": dict(PARAMETER_RUNTIME_TARGETS),
        "camera_offset_frames": {
            "external_camera": "world",
            "wrist_camera": "mount_local",
        },
        "object_asset": binding,
        "native_readback_required": True,
        "application_plan_digest": "",
    }
    plan["application_plan_digest"] = canonical_digest(
        plan, digest_field="application_plan_digest"
    )
    return plan


def _readback_row(
    *, parameter_id: str, requested_value: float, runtime_target: str, backend: NativeScenarioBackend
) -> dict[str, Any]:
    try:
        readback = _mapping(backend.read_parameter(parameter_id))
    except (AttributeError, KeyError, NotImplementedError) as exc:
        raise ScenarioApplicationError(
            [f"scenario_application_native_readback_unsupported:{parameter_id}"]
        ) from exc
    source = readback.get("source")
    native_path = readback.get("native_path")
    observed = _number(readback.get("value"))
    errors: list[str] = []
    if source not in NATIVE_READBACK_SOURCES:
        errors.append(f"scenario_application_native_readback_source_invalid:{parameter_id}")
    if not isinstance(native_path, str) or not native_path:
        errors.append(f"scenario_application_native_readback_path_missing:{parameter_id}")
    tolerance = 1e-8 if parameter_id == "object_yaw_degrees" else 1e-9
    if observed is None or not math.isclose(
        observed, requested_value, rel_tol=0.0, abs_tol=tolerance
    ):
        errors.append(f"scenario_application_native_readback_mismatch:{parameter_id}")
    if errors:
        raise ScenarioApplicationError(errors)
    return {
        "parameter_id": parameter_id,
        "runtime_target": runtime_target,
        "requested_value": requested_value,
        "native_readback": readback,
    }


def apply_scenario_instance(
    scenario_instance: Mapping[str, Any],
    *,
    backend: NativeScenarioBackend,
    admitted_cousins: Mapping[str, Mapping[str, Any]],
    expected_suite_digest: str,
    expected_harness_digest: str,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Apply one instance and retain only independently read-back native values."""

    plan = build_scenario_application_plan(
        scenario_instance,
        admitted_cousins=admitted_cousins,
        expected_suite_digest=expected_suite_digest,
        expected_harness_digest=expected_harness_digest,
    )
    try:
        supported = set(backend.supported_parameter_ids())
    except (AttributeError, NotImplementedError) as exc:
        raise ScenarioApplicationError(
            ["scenario_application_native_capability_query_unsupported"]
        ) from exc
    missing = sorted(set(APPLIED_PARAMETER_IDS) - supported)
    if missing:
        raise ScenarioApplicationError(
            [f"scenario_application_native_parameter_unsupported:{item}" for item in missing]
        )

    try:
        backend.apply_cousin(plan["object_asset"])
    except (AttributeError, NotImplementedError) as exc:
        raise ScenarioApplicationError(
            ["scenario_application_native_cousin_unsupported"]
        ) from exc
    for parameter_id in APPLIED_PARAMETER_IDS:
        try:
            backend.apply_parameter(parameter_id, plan["parameters"][parameter_id])
        except (AttributeError, KeyError, NotImplementedError) as exc:
            raise ScenarioApplicationError(
                [f"scenario_application_native_parameter_unsupported:{parameter_id}"]
            ) from exc

    rows = [
        _readback_row(
            parameter_id=parameter_id,
            requested_value=plan["parameters"][parameter_id],
            runtime_target=plan["runtime_targets"][parameter_id],
            backend=backend,
        )
        for parameter_id in APPLIED_PARAMETER_IDS
    ]
    try:
        cousin_readback = _mapping(backend.read_cousin())
    except (AttributeError, NotImplementedError) as exc:
        raise ScenarioApplicationError(
            ["scenario_application_native_cousin_readback_unsupported"]
        ) from exc
    cousin_errors: list[str] = []
    if cousin_readback.get("source") not in NATIVE_READBACK_SOURCES:
        cousin_errors.append("scenario_application_cousin_readback_source_invalid")
    if not str(cousin_readback.get("native_path") or ""):
        cousin_errors.append("scenario_application_cousin_readback_path_missing")
    if cousin_readback.get("cousin_id") != plan["object_asset"]["cousin_id"]:
        cousin_errors.append("scenario_application_cousin_readback_id_mismatch")
    if cousin_readback.get("asset_digest") != plan["object_asset"]["asset_digest"]:
        cousin_errors.append("scenario_application_cousin_readback_digest_mismatch")
    if cousin_errors:
        raise ScenarioApplicationError(cousin_errors)

    identity = _mapping(backend.runtime_identity())
    if not identity or not _digest(identity.get("configuration_digest")):
        raise ScenarioApplicationError(
            ["scenario_application_native_runtime_identity_invalid"]
        )
    receipt = {
        "schema_version": APPLICATION_RECEIPT_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "status": "applied_and_native_readback_verified",
        "suite_digest": plan["suite_digest"],
        "harness_digest": plan["harness_digest"],
        "instance_digest": plan["instance_digest"],
        "cell_id": plan["cell_id"],
        "seed": plan["seed"],
        "application_plan_digest": plan["application_plan_digest"],
        "native_runtime_identity": identity,
        "applied_parameters": rows,
        "object_asset": {
            "requested_binding": plan["object_asset"],
            "native_readback": cousin_readback,
        },
        "caller_asserted_application_accepted": False,
        "native_readback_required": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    if receipt_path is not None:
        path = Path(receipt_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, receipt)
    return receipt


__all__ = [
    "APPLICATION_PLAN_SCHEMA_VERSION",
    "APPLICATION_RECEIPT_SCHEMA_VERSION",
    "APPLIED_PARAMETER_IDS",
    "FROZEN_FACTOR_IDS",
    "NativeScenarioBackend",
    "PARAMETER_RUNTIME_TARGETS",
    "ScenarioApplicationError",
    "apply_scenario_instance",
    "build_scenario_application_plan",
]
