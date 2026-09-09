"""Staged object-acquisition configuration on the existing paired harness.

This module neither changes a frozen matrix nor admits a GPU launch. Search
may relax initial object visibility only; renderer, calibration, reset, and
sensor freshness remain mandatory. Acquisition is separate from task success.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import Field, StrictBool, StrictInt, model_validator

from .decision_evidence_contracts import canonical_digest
from .policy_observation_information import (
    AdapterInformationContract,
    DROID_INFORMATION_CONTRACTS,
    Digest,
    Identifier,
    InformationContract,
    PolicyObservationInformation,
    validate_adapter_information,
)

Visibility = Literal["visible", "partially_occluded", "initially_out_of_view"]
_VISIBILITY_TARGET = "ObservationManager.object_acquisition.initial_visibility"


class ObjectAcquisitionProtocol(InformationContract):
    mode: Literal["baseline_visible", "visual_search"] = "baseline_visible"
    initial_visibility: Visibility = "visible"
    minimum_target_pixels: Annotated[StrictInt, Field(ge=1)] = 8
    maximum_search_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    camera_calibration_digests: dict[Identifier, Digest]

    @model_validator(mode="after")
    def explicit_search(self) -> ObjectAcquisitionProtocol:
        if not self.camera_calibration_digests:
            raise ValueError("acquisition_policy_cameras_required")
        if self.mode == "baseline_visible" and (
            self.initial_visibility != "visible" or self.maximum_search_seconds is not None
        ):
            raise ValueError("baseline_requires_initial_visibility")
        if self.mode == "visual_search" and self.maximum_search_seconds is None:
            raise ValueError("visual_search_time_budget_required")
        return self


class AcquisitionCameraSample(InformationContract):
    frame_digest: Digest
    calibration_digest: Digest
    renderer_frame: Annotated[StrictInt, Field(ge=0)]
    rendered: StrictBool
    fresh: StrictBool
    target_pixels: Annotated[StrictInt, Field(ge=0)]
    target_visibility: Visibility

    @model_validator(mode="after")
    def visibility_matches_pixels(self) -> AcquisitionCameraSample:
        if (self.target_visibility == "initially_out_of_view") != (self.target_pixels == 0):
            raise ValueError("acquisition_visibility_pixel_count_inconsistent")
        return self


class AcquisitionSample(InformationContract):
    """Independent, frame-bound visibility readback, never a policy self-grade."""

    episode_id: Identifier
    target_object_id: Identifier
    episode_elapsed_s: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    episode_started_at_sim_time_s: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    observation_sim_time_s: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    physics_step: Annotated[StrictInt, Field(ge=0)]
    reset_digest: Digest
    visibility_source: Literal["deterministic_simulator_segmentation"]
    cameras: dict[Identifier, AcquisitionCameraSample]

    @model_validator(mode="after")
    def episode_time_origin(self) -> AcquisitionSample:
        if (
            abs(
                self.observation_sim_time_s
                - self.episode_started_at_sim_time_s
                - self.episode_elapsed_s
            )
            > 1e-9
        ):
            raise ValueError("acquisition_elapsed_time_origin_mismatch")
        return self


def assess_object_acquisition(
    protocol: ObjectAcquisitionProtocol,
    samples: Sequence[AcquisitionSample],
    *,
    reset_digest: str,
    episode_id: str,
    target_object_id: str,
    episode_started_at_sim_time_s: float,
) -> dict[str, Any]:
    """Assess sampled acquisition timing while rejecting invalid camera input.

    Initially out of view is valid only in an explicit search protocol. Absence
    of acquisition is an outcome; absence of a valid observation is an error.
    This staged assessor does not replace any current runtime admission gate.
    """

    if not samples or samples[0].episode_elapsed_s != 0:
        raise ValueError("acquisition_initial_observation_required")
    previous = None
    first_acquired = None
    acquisition_lower_bound = None
    initial_visibility = None
    for sample in samples:
        if sample.episode_id != episode_id or sample.target_object_id != target_object_id:
            raise ValueError("acquisition_episode_or_target_mismatch")
        if sample.reset_digest != reset_digest:
            raise ValueError("acquisition_reset_binding_mismatch")
        if sample.episode_started_at_sim_time_s != episode_started_at_sim_time_s:
            raise ValueError("acquisition_episode_time_origin_mismatch")
        if set(sample.cameras) != set(protocol.camera_calibration_digests):
            raise ValueError("acquisition_policy_camera_set_mismatch")
        if previous and (
            sample.episode_elapsed_s <= previous.episode_elapsed_s
            or sample.physics_step <= previous.physics_step
        ):
            raise ValueError("acquisition_observation_time_not_increasing")
        for name, camera in sample.cameras.items():
            if not camera.rendered or not camera.fresh:
                raise ValueError("acquisition_camera_invalid_or_stale")
            if camera.calibration_digest != protocol.camera_calibration_digests[name]:
                raise ValueError("acquisition_camera_calibration_mismatch")
            if previous and camera.renderer_frame <= previous.cameras[name].renderer_frame:
                raise ValueError("acquisition_renderer_frame_not_increasing")
        visible = any(
            c.target_pixels >= protocol.minimum_target_pixels for c in sample.cameras.values()
        )
        if previous is None:
            if all(
                c.target_visibility == "visible"
                and c.target_pixels >= protocol.minimum_target_pixels
                for c in sample.cameras.values()
            ):
                initial_visibility = "visible"
            elif all(
                c.target_visibility == "initially_out_of_view" for c in sample.cameras.values()
            ):
                initial_visibility = "initially_out_of_view"
            elif any(c.target_visibility == "partially_occluded" for c in sample.cameras.values()):
                initial_visibility = "partially_occluded"
            else:
                raise ValueError("acquisition_initial_visibility_protocol_mismatch")
            if initial_visibility != protocol.initial_visibility:
                raise ValueError("acquisition_initial_visibility_protocol_mismatch")
        if visible and first_acquired is None:
            first_acquired = sample.episode_elapsed_s
            acquisition_lower_bound = previous.episode_elapsed_s if previous else 0.0
        previous = sample
    deadline = protocol.maximum_search_seconds
    within_budget = first_acquired is not None and (deadline is None or first_acquired <= deadline)
    result = {
        "schema_version": "policy_object_acquisition_assessment.v1",
        "protocol_digest": canonical_digest(protocol.model_dump(mode="json")),
        "reset_digest": reset_digest,
        "episode_id": episode_id,
        "target_object_id": target_object_id,
        "episode_started_at_sim_time_s": episode_started_at_sim_time_s,
        "observations_digest": canonical_digest(
            {"samples": [s.model_dump(mode="json") for s in samples]}
        ),
        "initial_visibility": initial_visibility,
        "acquisition_status": "acquired"
        if within_budget
        else "budget_exceeded"
        if deadline is not None and samples[-1].episode_elapsed_s >= deadline
        else "pending",
        "first_observed_acquisition_seconds": first_acquired,
        "previous_observation_seconds": acquisition_lower_bound,
        "last_observation_seconds": samples[-1].episode_elapsed_s,
        "acquisition_is_task_success": False,
        "physical_outcome_proven": False,
    }
    result["assessment_digest"] = canonical_digest(result)
    return result


def object_acquisition_dimension(*, authority_digest: str) -> dict[str, Any]:
    """An optional camera/sensor dimension for a newly preregistered matrix."""

    return {
        "dimension_id": "initial_object_visibility",
        "family": "camera_sensor",
        "value_type": "categorical",
        "nominal": "visible",
        "values": ["visible", "partially_occluded", "initially_out_of_view"],
        "unit": "category",
        "application_target": _VISIBILITY_TARGET,
        "application_tolerance": 0.0,
        "source_contract": "embodiment",
        "authority_digest": authority_digest,
        "exact_workcell_invariant": True,
        "changes_object_or_task_identity": False,
    }


def compile_observation_protocol_plan(
    *,
    matrix: Mapping[str, Any],
    request: Mapping[str, Any],
    schedule_request: Mapping[str, Any],
    information_by_cell: Mapping[str, PolicyObservationInformation],
    acquisition_by_cell: Mapping[str, ObjectAcquisitionProtocol],
    adapters_by_candidate: Mapping[str, AdapterInformationContract],
    preregistration_digest: str,
) -> dict[str, Any]:
    """Bind staged configurations to the harness's exact two-policy schedule.

    Every cell, including controls, shares one setup and acquisition protocol.
    Runtime search integration and its geometric setup/readback are deliberately
    staged: this artifact is not an execution spec and cannot authorize a run.
    """

    from .exact_workcell_variation_runtime import compile_evaluation_schedule

    schedule = compile_evaluation_schedule(
        matrix, request=request, schedule_request=schedule_request
    )
    if (
        preregistration_digest
        != schedule_request["decision_design"]["preregistered_experiment_digest"]
    ):
        raise ValueError("observation_protocol_preregistration_mismatch")
    ids = {c["cell_id"] for c in matrix["cells"]}
    if set(information_by_cell) != ids or set(acquisition_by_cell) != ids:
        raise ValueError("observation_protocol_cell_coverage_mismatch")
    if set(adapters_by_candidate) != set(schedule["candidate_ids"]):
        raise ValueError("observation_protocol_candidate_coverage_mismatch")
    for candidate, adapter in adapters_by_candidate.items():
        if (
            candidate in DROID_INFORMATION_CONTRACTS
            and adapter != DROID_INFORMATION_CONTRACTS[candidate]
        ):
            raise ValueError("observation_protocol_frozen_adapter_mismatch")
    bindings = {}
    for cell in matrix["cells"]:
        cell_id = cell["cell_id"]
        information, acquisition = information_by_cell[cell_id], acquisition_by_cell[cell_id]
        identity = cell["exact_workcell_identity"]
        if (
            information.scene_id != identity["scene_id"]
            or information.task_id != identity["task_id"]
            or information.setup_object_coordinates.object_id
            != identity["canonical_object_asset_id"]
            or information.setup_object_coordinates.coordinate_frame_digest
            != identity["coordinate_frame_digest"]
        ):
            raise ValueError("observation_protocol_setup_identity_mismatch")
        for adapter in adapters_by_candidate.values():
            validate_adapter_information(information, adapter)
            if set(adapter.required_camera_ids) != set(acquisition.camera_calibration_digests):
                raise ValueError("observation_protocol_adapter_camera_set_mismatch")
        if information.setup_object_coordinates.measurement_context_digest != cell["reset_digest"]:
            raise ValueError("observation_protocol_setup_resolved_reset_mismatch")
        for axis, dimension_id in information.setup_position_dimensions.items():
            record = next(
                (r for r in cell["application_records"] if r["dimension_id"] == dimension_id), None
            )
            resolved = cell["resolved_values"].get(dimension_id)
            if (
                record is None
                or isinstance(resolved, bool)
                or not isinstance(resolved, (float, int))
            ):
                raise ValueError("observation_protocol_setup_position_dimension_invalid")
            position = information.setup_object_coordinates.position_m[("x", "y", "z").index(axis)]
            if abs(position - resolved) > record["application_tolerance"]:
                raise ValueError("observation_protocol_setup_position_readback_mismatch")
        if cell["phase"] == "canonical_anchor" and acquisition.mode != "baseline_visible":
            raise ValueError("observation_protocol_canonical_anchor_must_remain_visible")
        visibility_records = [
            r for r in cell["application_records"] if r["application_target"] == _VISIBILITY_TARGET
        ]
        if any(r["resolved_value"] != acquisition.initial_visibility for r in visibility_records):
            raise ValueError("visual_search_requires_preregistered_resolved_dimension")
        if acquisition.mode == "visual_search" and not any(
            r["family"] == "camera_sensor"
            and r["application_target"] == _VISIBILITY_TARGET
            and r["resolved_value"] == acquisition.initial_visibility
            for r in cell["application_records"]
        ):
            raise ValueError("visual_search_requires_preregistered_resolved_dimension")
        binding = {
            "cell_id": cell_id,
            "cell_digest": cell["cell_digest"],
            "reset_digest": cell["reset_digest"],
            "seed": cell["seed"],
            "information": information.model_dump(mode="json"),
            "acquisition": acquisition.model_dump(mode="json"),
        }
        binding["configuration_digest"] = canonical_digest(binding)
        bindings[cell_id] = binding
    rows = [
        {
            **row,
            "observation_configuration_digest": bindings[row["cell_id"]]["configuration_digest"],
        }
        for row in schedule["rows"]
    ]
    plan = {
        "schema_version": "policy_observation_protocol_plan.v1",
        "status": "staged_runtime_integration_required",
        "development_only": True,
        "execution_authorized": False,
        "existing_matrix_modified": False,
        "matrix_digest": matrix["matrix_digest"],
        "schedule_digest": schedule["schedule_digest"],
        "preregistration_digest": preregistration_digest,
        "adapter_contracts": {
            k: v.model_dump(mode="json") for k, v in adapters_by_candidate.items()
        },
        "cells": list(bindings.values()),
        "rows": rows,
        "required_runtime_integration": [
            "apply_and_read_back_preregistered_visibility_setup",
            "bind_visibility_measurements_to_exact_policy_frames",
            "record_acquisition_assessment_in_episode_receipt",
            "retain_baseline_and_qualified_admission_gates",
        ],
    }
    plan["plan_digest"] = canonical_digest(plan)
    return plan


class ObservationStagingRequest(InformationContract):
    schema_version: Literal["policy_observation_staging_request.v1"] = (
        "policy_observation_staging_request.v1"
    )
    development_only: Literal[True] = True
    matrix_request: dict[str, Any]
    schedule_request: dict[str, Any]
    information_by_cell: dict[str, PolicyObservationInformation]
    acquisition_by_cell: dict[str, ObjectAcquisitionProtocol]


def stage_observation_protocol(request: ObservationStagingRequest) -> dict[str, Any]:
    """Configure the current adapters through a local, non-launching entrypoint."""

    from .exact_workcell_variation_matrix import compile_variation_matrix

    matrix = compile_variation_matrix(request.matrix_request)
    return compile_observation_protocol_plan(
        matrix=matrix,
        request=request.matrix_request,
        schedule_request=request.schedule_request,
        information_by_cell=request.information_by_cell,
        acquisition_by_cell=request.acquisition_by_cell,
        adapters_by_candidate=DROID_INFORMATION_CONTRACTS,
        preregistration_digest=request.schedule_request["decision_design"][
            "preregistered_experiment_digest"
        ],
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write a create-only staging artifact. There is deliberately no launch flag."""

    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    request = ObservationStagingRequest.model_validate_json(args.request.read_text())
    plan = stage_observation_protocol(request)
    with args.output.open("x") as stream:
        stream.write(json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
