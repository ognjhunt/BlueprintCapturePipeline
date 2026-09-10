"""ADP-009D/day-14 setup knowledge and policy-visible information contracts.

These declarations configure a test; they do not qualify a deployment source.
The DROID path delegates to the production observation builder unchanged.
Coordinate-aware adapters must explicitly declare their own compatible input.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .decision_evidence_contracts import canonical_digest

Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(min_length=1, max_length=192, pattern=r"^\S+$")]
Finite = Annotated[float, Field(allow_inf_nan=False)]
SourceKind = Literal["simulator_ground_truth", "deployment_known", "deployment_estimated"]


class InformationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ObjectCoordinates(InformationContract):
    """A labeled measurement, not an implicit model input or physical truth."""

    object_id: Identifier
    coordinate_frame_digest: Digest
    source_kind: SourceKind
    source_id: Identifier
    source_contract_digest: Digest
    measurement_digest: Digest
    measurement_context_digest: Digest
    position_m: tuple[Finite, Finite, Finite]
    measured_at_s: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    @model_validator(mode="after")
    def measurement_binding(self) -> ObjectCoordinates:
        if self.measurement_digest != canonical_digest(
            self.model_dump(mode="json"), digest_field="measurement_digest"
        ):
            raise ValueError("object_coordinate_measurement_digest_mismatch")
        return self


class PolicyCoordinateInput(InformationContract):
    schema_version: Literal["object_position_m.v1"] = "object_position_m.v1"
    source_kind: Literal["deployment_known", "deployment_estimated"]
    source_id: Identifier
    source_contract_digest: Digest
    deployment_availability_evidence_digest: Digest
    coordinate_frame_digest: Digest
    maximum_age_s: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class PolicyObservationInformation(InformationContract):
    schema_version: Literal["policy_observation_information.v1"] = (
        "policy_observation_information.v1"
    )
    scene_id: Identifier
    task_id: Identifier
    setup_object_coordinates: ObjectCoordinates
    setup_position_dimensions: dict[Literal["x", "y", "z"], Identifier] = Field(
        default_factory=dict
    )
    policy_object_coordinates: PolicyCoordinateInput | None = None


class AdapterInformationContract(InformationContract):
    """Selected by adapter code, never inferred from a caller's support flag."""

    interface_id: Identifier
    observation_keys: tuple[Identifier, ...]
    required_camera_ids: tuple[Identifier, ...] = ()
    coordinate_schema: Literal["object_position_m.v1"] | None = None
    coordinate_input_key: Identifier | None = None

    @model_validator(mode="after")
    def coherent_coordinate_declaration(self) -> AdapterInformationContract:
        if len(set(self.observation_keys)) != len(self.observation_keys):
            raise ValueError("adapter_observation_keys_duplicate")
        if (self.coordinate_schema is None) != (self.coordinate_input_key is None):
            raise ValueError("adapter_coordinate_schema_and_key_required_together")
        if (
            self.coordinate_input_key is not None
            and self.coordinate_input_key not in self.observation_keys
        ):
            raise ValueError("adapter_coordinate_key_not_declared")
        return self


_DROID_KEYS = (
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
    "observation/joint_position",
    "observation/gripper_position",
    "prompt",
)
DROID_INFORMATION_CONTRACTS = {
    "pi05_droid": AdapterInformationContract(
        interface_id="openpi_droid.v1",
        observation_keys=_DROID_KEYS,
        required_camera_ids=("external", "wrist"),
    ),
    "groot_n17_droid": AdapterInformationContract(
        interface_id="groot_n17_droid.v1",
        required_camera_ids=("external", "wrist"),
        observation_keys=(
            *_DROID_KEYS,
            "observation/eef_9d",
            "observation/eef_9d_frame_provenance",
        ),
    ),
}


def validate_adapter_information(
    config: PolicyObservationInformation,
    adapter: AdapterInformationContract,
) -> None:
    for declared in DROID_INFORMATION_CONTRACTS.values():
        if adapter.interface_id == declared.interface_id and adapter != declared:
            raise ValueError("frozen_droid_adapter_information_contract_mismatch")
    selected = config.policy_object_coordinates
    if selected is not None and adapter.coordinate_schema != selected.schema_version:
        raise ValueError("policy_adapter_object_coordinates_not_supported")


def policy_coordinate_fields(
    config: PolicyObservationInformation,
    *,
    adapter: AdapterInformationContract,
    measurement: ObjectCoordinates | None,
    query_time_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project only an explicitly selected deployment measurement into input.

    Setup coordinates are never a fallback, including when both sources use
    the same object ID. The returned receipt stays outside the policy payload.
    """

    import math

    validate_adapter_information(config, adapter)
    selected = config.policy_object_coordinates
    fields: dict[str, Any] = {}
    if selected is not None:
        if measurement is None:
            raise ValueError("policy_coordinate_measurement_required")
        if (
            measurement.source_kind != selected.source_kind
            or measurement.source_id != selected.source_id
            or measurement.source_contract_digest != selected.source_contract_digest
            or measurement.coordinate_frame_digest != selected.coordinate_frame_digest
            or measurement.object_id != config.setup_object_coordinates.object_id
        ):
            raise ValueError("policy_coordinate_source_or_frame_mismatch")
        if (
            not math.isfinite(query_time_s)
            or not 0 <= query_time_s - measurement.measured_at_s <= selected.maximum_age_s
        ):
            raise ValueError("policy_coordinate_measurement_stale_or_future")
        assert adapter.coordinate_input_key is not None
        fields[adapter.coordinate_input_key] = list(measurement.position_m)
    receipt = {
        "schema_version": "policy_information_projection.v1",
        "configuration_digest": canonical_digest(config.model_dump(mode="json")),
        "adapter_contract_digest": canonical_digest(adapter.model_dump(mode="json")),
        "policy_coordinate_keys": sorted(fields),
        "policy_coordinate_fields_digest": canonical_digest(fields),
        "policy_coordinate_measurement_digest": measurement.measurement_digest
        if fields and measurement
        else None,
        "query_time_s": query_time_s if fields else None,
        "setup_coordinates_used_as_policy_input": False,
        "deployment_source_qualification_proven": False,
    }
    receipt["projection_digest"] = canonical_digest(receipt)
    return fields, receipt


def build_configured_droid_observation(
    config: PolicyObservationInformation,
    *,
    candidate_id: str,
    camera_rgb: Mapping[str, Any],
    inputs: Mapping[str, Any],
    prompt: str,
) -> dict[str, Any]:
    """Use the existing production DROID builder; never append setup XYZ."""

    from .adp009d_droid_observation import build_droid_observation_from_inputs

    adapter = DROID_INFORMATION_CONTRACTS.get(candidate_id)
    if adapter is None:
        raise ValueError("configured_droid_candidate_not_admitted")
    validate_adapter_information(config, adapter)
    observation = build_droid_observation_from_inputs(candidate_id, camera_rgb, inputs, prompt)
    if set(observation) - set(adapter.observation_keys):
        raise ValueError("policy_observation_undeclared_key")
    return observation
