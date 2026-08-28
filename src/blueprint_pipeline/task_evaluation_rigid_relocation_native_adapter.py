"""Translate a configured rigid-relocation template into native-Arena inputs.

Scene configuration publishes robot-neutral task truth.  The native episode
compiler consumes a simulator-specific packet.  This module is the one
production-owned boundary between those contracts: it reopens the exact three
digest-bound source documents, cross-compares every duplicated task fact, and
emits a sealed native representation without changing the external
``planar_push`` manipulation strategy.  ``rigid_pick_place`` remains only the
legacy native runtime's umbrella task kind.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_configured_scene_revision import (
    TaskEvaluationConfiguredSceneRevisionError,
    validate_configured_scene_revision,
)
from .task_evaluation_launch_preparation_contract import (
    TaskEvaluationLaunchPreparationContractError,
    validate_launch_preparation_request,
)


SCHEMA_VERSION = "task_evaluation_rigid_relocation_native_adapter.v1"
DEFINITION_CONTRACT_PATH = "scene.configured_revision.task_template.definition"
SUCCESS_CONTRACT_PATH = (
    "scene.configured_revision.task_template.success_criteria"
)
EXECUTION_CONTRACT_PATH = "scene.configured_revision.task_template.execution"
SOURCE_SCHEMAS = {
    DEFINITION_CONTRACT_PATH: "task_evaluation_rigid_relocation_template.v1",
    SUCCESS_CONTRACT_PATH: (
        "task_evaluation_rigid_relocation_success_criteria.v1"
    ),
    EXECUTION_CONTRACT_PATH: (
        "task_evaluation_rigid_relocation_execution_spec.v1"
    ),
}
NATIVE_PHYSICS_FREQUENCY_HZ = 120


class TaskEvaluationRigidRelocationNativeAdapterError(ValueError):
    """Configured task truth cannot be translated without changing meaning."""


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _source_document(
    references: Mapping[str, Mapping[str, Any]],
    *,
    contract_path: str,
    expected_reference: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = references.get(contract_path)
    unresolved_path = Path(
        str((row or {}).get("materialized_path") or "")
    ).expanduser()
    path = unresolved_path.resolve()
    if (
        row is None
        or row.get("contract_path") != contract_path
        or row.get("uri") != expected_reference.get("uri")
        or row.get("digest") != expected_reference.get("digest")
        or row.get("size_bytes") != expected_reference.get("size_bytes")
        or row.get("full_byte_service_account_readback_passed") is not True
        or unresolved_path.is_symlink()
        or not path.is_file()
        or _sha256_and_size(path) != (row.get("digest"), row.get("size_bytes"))
    ):
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            f"rigid_relocation_native_adapter_source_invalid:{contract_path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            f"rigid_relocation_native_adapter_source_json_invalid:{contract_path}"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != SOURCE_SCHEMAS[contract_path]
    ):
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            f"rigid_relocation_native_adapter_source_contract_invalid:{contract_path}"
        )
    document = json.loads(json.dumps(dict(value), sort_keys=True))
    binding = {
        "contract_path": contract_path,
        "uri": row["uri"],
        "digest": row["digest"],
        "size_bytes": row["size_bytes"],
        "schema_version": document["schema_version"],
        "canonical_document_digest": canonical_digest(document),
        "full_byte_service_account_readback_passed": True,
    }
    return document, binding


def _vector(value: Any, *, field: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            f"rigid_relocation_native_adapter_vector_invalid:{field}"
        )
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            f"rigid_relocation_native_adapter_vector_invalid:{field}"
        ) from exc
    if not all(math.isfinite(item) for item in result):
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            f"rigid_relocation_native_adapter_vector_invalid:{field}"
        )
    return result


def _positive_number(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            f"rigid_relocation_native_adapter_number_invalid:{field}"
        ) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            f"rigid_relocation_native_adapter_number_invalid:{field}"
        )
    return number


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            f"rigid_relocation_native_adapter_integer_invalid:{field}"
        )
    return value


def _success_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: json.loads(json.dumps(item))
        for key, item in value.items()
        if key not in {"schema_version", "status"}
    }


def _success_bounds(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in _success_payload(value).items()
        if key != "target_center_xyz_m"
    }


def adapt_rigid_relocation_task_template(
    *,
    request: Mapping[str, Any],
    configured_revision: Mapping[str, Any],
    materialized_references: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return one digest-bound native view of exact configured task bytes."""

    try:
        validated_request = validate_launch_preparation_request(request)
        revision = validate_configured_scene_revision(configured_revision)
    except (
        TaskEvaluationLaunchPreparationContractError,
        TaskEvaluationConfiguredSceneRevisionError,
    ) as exc:
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_authority_invalid"
        ) from exc
    task = validated_request["task"]
    if (
        validated_request["run_mode"] != "episode_evaluation"
        or task["binding_mode"] != "reuse_configured_template"
        or task["kind"] != "rigid_relocation"
        or task["strategy"] != "planar_push"
        or task["identity"] != revision["task_template"]["identity"]
        or task["subject"]["identity"] != revision["replacement"]["identity"]
        or task["configured_scene_revision_digest"] != revision["revision_digest"]
    ):
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_request_binding_mismatch"
        )

    documents: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    for contract_path, revision_field in (
        (DEFINITION_CONTRACT_PATH, "definition"),
        (SUCCESS_CONTRACT_PATH, "success_criteria"),
        (EXECUTION_CONTRACT_PATH, "execution"),
    ):
        document, binding = _source_document(
            materialized_references,
            contract_path=contract_path,
            expected_reference=revision["task_template"][revision_field],
        )
        documents[contract_path] = document
        bindings.append(binding)

    template = documents[DEFINITION_CONTRACT_PATH]
    success = documents[SUCCESS_CONTRACT_PATH]
    execution = documents[EXECUTION_CONTRACT_PATH]
    if (
        template.get("status")
        != "preregistered_candidate_pending_configured_scene_revision"
        or success.get("status") != "preregistered_before_any_episode"
        or execution.get("status") != "preregistered_before_any_episode"
        or template.get("task_identity") != task["identity"]
        or template.get("object_identity") != task["subject"]["identity"]
        or template.get("strategy") != "planar_push"
        or execution.get("strategy") != "planar_push"
    ):
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_identity_or_strategy_mismatch"
        )

    start = _vector(template.get("start_center_xyz_m"), field="template.start")
    target = _vector(template.get("target_center_xyz_m"), field="template.target")
    execution_start = _vector(
        execution.get("start_center_xyz_m"), field="execution.start"
    )
    execution_target = _vector(
        execution.get("target_center_xyz_m"), field="execution.target"
    )
    success_target = _vector(
        success.get("target_center_xyz_m"), field="success.target"
    )
    if start != execution_start or target != execution_target or target != success_target:
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_task_pose_mismatch"
        )
    if not math.isclose(start[2], target[2], rel_tol=0.0, abs_tol=1.0e-9):
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_planar_height_mismatch"
        )

    template_success = template.get("success")
    if (
        not isinstance(template_success, Mapping)
        or dict(template_success) != _success_bounds(success)
    ):
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_success_bounds_mismatch"
        )
    control_frequency = _positive_number(
        execution.get("control_frequency_hz"), field="control_frequency_hz"
    )
    maximum_steps = _positive_integer(
        execution.get("maximum_step_count"), field="maximum_step_count"
    )
    maximum_seconds = _positive_number(
        execution.get("maximum_episode_seconds"), field="maximum_episode_seconds"
    )
    seed = _positive_integer(execution.get("resolved_seed"), field="resolved_seed")
    if (
        template.get("control_frequency_hz") != execution.get("control_frequency_hz")
        or template.get("maximum_step_count") != execution.get("maximum_step_count")
        or template.get("maximum_episode_seconds")
        != execution.get("maximum_episode_seconds")
        or template.get("resolved_seed") != execution.get("resolved_seed")
        or not math.isclose(
            maximum_steps / control_frequency,
            maximum_seconds,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_execution_timing_mismatch"
        )
    decimation = NATIVE_PHYSICS_FREQUENCY_HZ / control_frequency
    if not decimation.is_integer() or decimation < 1:
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_physics_cadence_invalid"
        )
    action_bounds = execution.get("action_bounds_m_per_step")
    try:
        action_minimum = float(action_bounds["minimum"])
        action_maximum = float(action_bounds["maximum"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_action_bounds_invalid"
        ) from exc
    if (
        not all(math.isfinite(item) for item in (action_minimum, action_maximum))
        or not action_minimum < 0.0 < action_maximum
    ):
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_action_bounds_invalid"
        )

    minimum_displacement = _positive_number(
        success.get("minimum_planar_displacement_m"),
        field="minimum_planar_displacement_m",
    )
    target_tolerance = _positive_number(
        success.get("maximum_final_planar_target_error_m"),
        field="maximum_final_planar_target_error_m",
    )
    source_documents = {
        "bindings": bindings,
        "documents": {
            "definition": template,
            "success_criteria": success,
            "execution": execution,
        },
    }
    source_documents["source_documents_digest"] = canonical_digest(
        source_documents
    )
    cell_id = f"configured_scene_canonical.seed_{seed}"
    scenario_document: dict[str, Any] = {
        "schema_version": "adp009d_scenario_instance.v1",
        "program_id": "arm-decision-proof-v1",
        "cell_id": cell_id,
        "template_id": "configured_scene_canonical",
        "family": "canonical",
        "partition": "qualification",
        "scored": True,
        "seed": seed,
        "resolved_parameters": {
            "object_start_x_m": start[0],
            "object_start_y_m": start[1],
            "object_start_z_m": start[2],
            "target_x_m": target[0],
            "target_y_m": target[1],
            "target_z_m": target[2],
        },
        "factor_records": [],
        "required_controls": [
            "zero_action_negative",
            "deterministic_scripted_positive",
        ],
        "policy_neutral": True,
        "caller_asserted_success": False,
        "learned_policy_outcomes_consulted": False,
        "configured_task_source_documents_digest": source_documents[
            "source_documents_digest"
        ],
        "instance_digest": "",
    }
    scenario_document["instance_digest"] = canonical_digest(
        scenario_document, digest_field="instance_digest"
    )
    native_task_spec = {
        "schema_version": "adp_task_spec.v1",
        "task_kind": "rigid_pick_place",
        "manipulation_strategy": "planar_push",
        "subject_asset_id": task["subject"]["identity"]["id"],
        "prompt": "Move the configured rigid object to the registered target by planar push.",
        "start_pose_world": [*start, 0.0, 0.0, 0.0, 1.0],
        "target_position_world_m": target,
        "destination_position_tolerance_m": target_tolerance,
        "minimum_translation_m": minimum_displacement,
        "minimum_lift_m": 0.0,
        "control_frequency_hz": control_frequency,
        "maximum_action_steps": maximum_steps,
        "settle_window_samples": 1,
        "maximum_episode_seconds": maximum_seconds,
        "action_bounds_m_per_step": {
            "minimum": action_minimum,
            "maximum": action_maximum,
        },
        "configured_success_criteria": _success_payload(success),
        "configured_task_source_documents_digest": source_documents[
            "source_documents_digest"
        ],
    }
    native_definition = {
        "schema_version": "task_evaluation_native_task_definition.v1",
        "identity": dict(task["identity"]),
        "task_spec": native_task_spec,
        "task_object_pose_world": {
            "position_world_m": start,
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "task_joint_bindings": [],
        "task_state_binding": None,
    }
    native_success = {
        "schema_version": "task_evaluation_native_success_criteria.v1",
        "identity": dict(task["identity"]),
        "criteria": _success_payload(success),
    }
    native_execution = {
        "schema_version": "task_evaluation_native_episode_execution.v1",
        "identity": dict(task["identity"]),
        "physics_frequency_hz": NATIVE_PHYSICS_FREQUENCY_HZ,
        "control_frequency_hz": control_frequency,
        "control_decimation": int(decimation),
        "maximum_step_count": maximum_steps,
        "maximum_episode_seconds": maximum_seconds,
        "scenario": {
            "context_kind": "evaluation_cell",
            "cell_id": cell_id,
            "instance_digest": scenario_document["instance_digest"],
            "seed": seed,
            "context_document": scenario_document,
        },
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "adapted",
        "external_task_kind": "rigid_relocation",
        "native_task_kind": "rigid_pick_place",
        "manipulation_strategy": "planar_push",
        "configured_scene_revision_digest": revision["revision_digest"],
        "source_documents": source_documents,
        "native_task_definition": native_definition,
        "native_success_criteria": native_success,
        "native_episode_execution": native_execution,
        "adapter_digest": "",
    }
    result["adapter_digest"] = canonical_digest(
        result, digest_field="adapter_digest"
    )
    return result


__all__ = [
    "EXECUTION_CONTRACT_PATH",
    "NATIVE_PHYSICS_FREQUENCY_HZ",
    "SCHEMA_VERSION",
    "SUCCESS_CONTRACT_PATH",
    "TaskEvaluationRigidRelocationNativeAdapterError",
    "adapt_rigid_relocation_task_template",
]
