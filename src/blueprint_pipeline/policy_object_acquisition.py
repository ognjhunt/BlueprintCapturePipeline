"""Configure object acquisition on the existing paired exact-workcell harness."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .decision_evidence_contracts import canonical_digest
from .policy_observation_information import (
    AdapterInformationContract,
    DROID_INFORMATION_CONTRACTS,
    InformationContract,
    PolicyObservationInformation,
    validate_adapter_information,
)
from .policy_object_acquisition_contract import (
    AcquisitionCameraSample,
    AcquisitionSample,
    ObjectAcquisitionProtocol,
    assess_object_acquisition,
    _VISIBILITY_TARGET,
)

__all__ = [
    "AcquisitionCameraSample",
    "AcquisitionSample",
    "ObjectAcquisitionProtocol",
    "assess_object_acquisition",
    "object_acquisition_dimension",
    "compile_observation_protocol_plan",
    "ObservationStagingRequest",
    "stage_observation_protocol",
    "main",
    "bind_native_cell_observation_protocol",
]


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
        "status": "configured_pending_native_binding",
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
        "required_native_validation": [
            "camera_mount_and_reset_pose_readback",
            "same_render_product_reference_time_and_bbox_annotations",
            "exact_policy_frame_visibility_capture",
            "paired_episode_acquisition_receipts",
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


def bind_native_cell_observation_protocol(
    *,
    plan: Mapping[str, Any],
    native_cell: Mapping[str, Any],
    camera_setups: Mapping[str, Any],
    task_success_contract_digest: str,
    native_scene_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce an explicit native-cell attachment without editing a manifest."""

    from .policy_observation_runtime_contract import (
        apply_native_cell_protocol,
        seal_native_observation_protocol,
        validate_native_cell_protocol,
    )

    if plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest"):
        raise ValueError("observation_protocol_source_plan_digest_mismatch")
    if native_scene_plan.get("plan_digest") != canonical_digest(
        native_scene_plan, digest_field="plan_digest"
    ):
        raise ValueError("observation_protocol_native_scene_plan_digest_mismatch")
    source = next((c for c in plan["cells"] if c["cell_id"] == native_cell.get("cell_id")), None)
    if source is None or source["seed"] != native_cell.get("seed"):
        raise ValueError("observation_protocol_source_native_cell_mismatch")
    binding = seal_native_observation_protocol(
        {
            **{
                name: native_cell[name]
                for name in ("cell_id", "seed", "cell_spec_digest", "resolved_scenario_digest")
            },
            "reset_digest": source["reset_digest"],
            "task_success_contract_digest": task_success_contract_digest,
            "preregistration_digest": plan["preregistration_digest"],
            "source_plan_digest": plan["plan_digest"],
            "native_scene_plan_digest": native_scene_plan["plan_digest"],
            "source_configuration_digest": source["configuration_digest"],
            "information": source["information"],
            "acquisition": source["acquisition"],
            "camera_setups": camera_setups,
        }
    )
    validate_native_cell_protocol(
        {**dict(native_cell), "observation_protocol": binding},
        task_success_contract_digest=task_success_contract_digest,
    )
    apply_native_cell_protocol(
        native_scene_plan, {**dict(native_cell), "observation_protocol": binding}
    )
    return binding


def main(argv: Sequence[str] | None = None) -> int:
    """Write a create-only staging artifact. There is deliberately no launch flag."""

    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-cell-request", type=Path)
    args = parser.parse_args(argv)
    request = ObservationStagingRequest.model_validate_json(args.request.read_text())
    plan = stage_observation_protocol(request)
    if args.native_cell_request is not None:
        native = json.loads(args.native_cell_request.read_text())
        plan = bind_native_cell_observation_protocol(plan=plan, **native)
    with args.output.open("x") as stream:
        stream.write(json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
