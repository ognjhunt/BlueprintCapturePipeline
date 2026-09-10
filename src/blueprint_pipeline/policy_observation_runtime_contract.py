"""Explicit native-cell binding for setup knowledge and acquisition protocols.

This is configuration admission, not paid execution authority. The worker
accepts it only in an independently authorized internal-policy canary manifest.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Annotated, Any, Literal

from pydantic import Field, StrictInt, model_validator

from .decision_evidence_contracts import canonical_digest
from .policy_object_acquisition_contract import ObjectAcquisitionProtocol
from .policy_observation_information import (
    Digest,
    DROID_INFORMATION_CONTRACTS,
    Finite,
    Identifier,
    InformationContract,
    PolicyObservationInformation,
    validate_adapter_information,
)

Matrix16 = Annotated[tuple[Finite, ...], Field(min_length=16, max_length=16)]


def validate_rigid_matrix(matrix: Any) -> None:
    import numpy as np

    value = np.asarray(matrix, dtype=float).reshape(4, 4)
    if (
        not np.isfinite(value).all()
        or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6, rtol=0)
        or not abs(np.linalg.det(value[:3, :3]) - 1) <= 1e-6
    ):
        raise ValueError("observation_geometry_rigid_matrix_invalid")


class NativeCameraSetup(InformationContract):
    frame_from_camera_matrix: Matrix16
    reset_world_from_camera_opencv_matrix: Matrix16
    source_intrinsics_digest: Digest
    pose_tolerance_m: Annotated[float, Field(gt=0, le=0.001, allow_inf_nan=False)] = 0.0001
    rotation_tolerance: Annotated[float, Field(gt=0, le=0.001, allow_inf_nan=False)] = 0.0001

    @model_validator(mode="after")
    def rigid_poses(self) -> NativeCameraSetup:
        validate_rigid_matrix(self.frame_from_camera_matrix)
        validate_rigid_matrix(self.reset_world_from_camera_opencv_matrix)
        return self


class NativeObservationProtocol(InformationContract):
    schema_version: Literal["native_policy_observation_protocol.v1"] = (
        "native_policy_observation_protocol.v1"
    )
    run_kind: Literal["internal_policy_canary"] = "internal_policy_canary"
    claim_ceiling: Literal["diagnostic_policy_execution"] = "diagnostic_policy_execution"
    cell_id: Identifier
    seed: StrictInt
    cell_spec_digest: Digest
    resolved_scenario_digest: Digest
    reset_digest: Digest
    task_success_contract_digest: Digest
    preregistration_digest: Digest
    source_plan_digest: Digest
    native_scene_plan_digest: Digest
    source_configuration_digest: Digest
    information: PolicyObservationInformation
    acquisition: ObjectAcquisitionProtocol
    camera_setups: dict[Literal["external", "wrist"], NativeCameraSetup]
    binding_digest: Digest

    @model_validator(mode="after")
    def complete_binding(self) -> NativeObservationProtocol:
        if set(self.camera_setups) != {"external", "wrist"} or set(
            self.acquisition.camera_calibration_digests
        ) != set(self.camera_setups):
            raise ValueError("observation_protocol_native_camera_set_invalid")
        if (
            self.information.setup_object_coordinates.measurement_context_digest
            != self.reset_digest
        ):
            raise ValueError("observation_protocol_native_setup_reset_mismatch")
        for adapter in DROID_INFORMATION_CONTRACTS.values():
            validate_adapter_information(self.information, adapter)
        if self.binding_digest != canonical_digest(
            self.model_dump(mode="json"), digest_field="binding_digest"
        ):
            raise ValueError("observation_protocol_native_binding_digest_mismatch")
        return self


def seal_native_observation_protocol(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a complete new proposal before sealing its runtime binding."""

    payload = {
        "schema_version": "native_policy_observation_protocol.v1",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        **dict(value),
    }
    payload["information"] = PolicyObservationInformation.model_validate(
        payload["information"]
    ).model_dump(mode="json")
    payload["acquisition"] = ObjectAcquisitionProtocol.model_validate(
        payload["acquisition"]
    ).model_dump(mode="json")
    payload["camera_setups"] = {
        role: NativeCameraSetup.model_validate(setup).model_dump(mode="json")
        for role, setup in payload["camera_setups"].items()
    }
    payload["binding_digest"] = canonical_digest(payload, digest_field="binding_digest")
    return NativeObservationProtocol.model_validate(payload).model_dump(mode="json")


def validate_native_cell_protocol(
    cell: Mapping[str, Any], *, task_success_contract_digest: str
) -> NativeObservationProtocol | None:
    raw = cell.get("observation_protocol")
    if raw is None:
        return None
    if cell.get("operator_wrist_camera_aim") is not None:
        raise ValueError("observation_protocol_operator_camera_aim_conflict")
    binding = NativeObservationProtocol.model_validate(raw)
    if any(
        getattr(binding, name) != cell.get(name)
        for name in ("cell_id", "seed", "cell_spec_digest", "resolved_scenario_digest")
    ):
        raise ValueError("observation_protocol_native_cell_binding_mismatch")
    if binding.task_success_contract_digest != task_success_contract_digest:
        raise ValueError("observation_protocol_native_success_contract_mismatch")
    if (
        str(cell.get("family")) in {"canonical", "canonical_anchor"}
        and binding.acquisition.mode != "baseline_visible"
    ):
        raise ValueError("observation_protocol_canonical_search_forbidden")
    return binding


def camera_calibration_digest(camera: Mapping[str, Any]) -> str:
    return canonical_digest(
        {
            key: camera.get(key)
            for key in (
                "role",
                "frame_from_camera_matrix",
                "intrinsics",
                "optical_convention",
                "pose_frame",
                "parent_prim_path",
            )
        }
    )


def apply_native_cell_protocol(plan: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    """Apply only explicitly bound camera mounts to a new resolved plan."""

    binding = validate_native_cell_protocol(
        cell,
        task_success_contract_digest=plan["task_spec"]["task_success_contract"]["contract_digest"],
    )
    if binding is None:
        return dict(plan)
    from .native_task_arena_runtime import camera_runtime_parameters

    value = deepcopy(dict(plan))
    if value.get("operator_wrist_camera_aim") is not None:
        raise ValueError("observation_protocol_operator_camera_aim_conflict")
    if binding.native_scene_plan_digest != value.get("plan_digest"):
        raise ValueError("observation_protocol_native_scene_plan_digest_mismatch")
    subject = next(row for row in value["objects"] if row.get("task_subject") is True)
    if binding.information.scene_id != value.get(
        "scene_id"
    ) or binding.information.task_id != value.get("task_id"):
        raise ValueError("observation_protocol_native_scene_task_mismatch")
    if binding.information.setup_object_coordinates.object_id != subject.get("asset_id"):
        raise ValueError("observation_protocol_native_object_identity_mismatch")
    position = subject["pose_world"]["position_world_m"]
    if any(
        abs(float(a) - b) > 1e-6
        for a, b in zip(
            position, binding.information.setup_object_coordinates.position_m, strict=True
        )
    ):
        raise ValueError("observation_protocol_native_object_position_mismatch")
    for camera in value["cameras"]:
        setup = binding.camera_setups.get(camera["role"])
        if setup is None:
            continue
        if canonical_digest(camera["intrinsics"]) != setup.source_intrinsics_digest:
            raise ValueError("observation_protocol_native_intrinsics_mismatch")
        camera["frame_from_camera_matrix"] = list(setup.frame_from_camera_matrix)
        camera_runtime_parameters(camera)
        if (
            camera_calibration_digest(camera)
            != binding.acquisition.camera_calibration_digests[camera["role"]]
        ):
            raise ValueError("observation_protocol_native_calibration_digest_mismatch")
    profile = value["policy_canary_embodiment_profile"]
    profile["preserve_official_policy_camera_calibration"] = False
    profile["preserve_official_policy_camera_intrinsics"] = False
    profile["observation_protocol_binding_digest"] = binding.binding_digest
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    value["observation_protocol"] = binding.model_dump(mode="json")
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return value


def validate_search_gate(gate: Mapping[str, Any], *, expected_binding_digest: str | None) -> bool:
    """A search condition never impersonates a baseline task-visible verdict."""

    if expected_binding_digest is None:
        return False
    raw = gate.get("observation_protocol")
    if not isinstance(raw, Mapping):
        return False
    try:
        binding = NativeObservationProtocol.model_validate(raw)
    except ValueError:
        return False
    setup = gate.get("observation_protocol_setup")
    return bool(
        binding.binding_digest == expected_binding_digest
        and binding.acquisition.mode == "visual_search"
        and isinstance(setup, Mapping)
        and setup.get("schema_version") == "native_observation_protocol_setup.v1"
        and setup.get("binding_digest") == binding.binding_digest
        and setup.get("initial_visibility") == binding.acquisition.initial_visibility
        and setup.get("geometry_passed") is True
        and setup.get("sensor_freshness_passed") is True
        and setup.get("initial_condition_passed") is True
        and setup.get("receipt_digest") == canonical_digest(setup, digest_field="receipt_digest")
        and gate.get("run_kind") == "internal_policy_canary"
        and gate.get("claim_ceiling") == "diagnostic_policy_execution"
        and gate.get("status") == "passed"
        and gate.get("frame_structure_passed") is True
        and gate.get("policy_observation_integrity_passed") is True
        and gate.get("blockers") == []
        and gate.get("candidate_policy_loaded") is False
        and gate.get("candidate_policy_queried") is False
        and gate.get("official_ranking_permitted") is False
        and gate.get("scene_promotion_permitted") is False
        and gate.get("gate_digest") == canonical_digest(gate, digest_field="gate_digest")
    )
