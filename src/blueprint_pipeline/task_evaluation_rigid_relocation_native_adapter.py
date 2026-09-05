"""Translate a configured rigid-relocation template into native-Arena inputs.

Scene configuration publishes robot-neutral task truth.  The native episode
compiler consumes a simulator-specific packet.  This module is the one
production-owned boundary between those contracts: it reopens the exact three
digest-bound source documents, cross-compares every duplicated task fact, and
emits a sealed native representation without changing the external strategy.
Both planar push and explicitly authored rigid pick-and-place compile into the
existing native ``rigid_pick_place`` runtime task kind.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_franka_action_math import (
    NativeFrankaActionMathError,
    grasp_orientation_contact_xyzw,
)
from .task_evaluation_configured_scene_revision import (
    TaskEvaluationConfiguredSceneRevisionError,
    validate_configured_scene_revision,
)
from .task_evaluation_launch_preparation_contract import (
    TaskEvaluationLaunchPreparationContractError,
    validate_launch_preparation_request,
)


SCHEMA_VERSION = "task_evaluation_rigid_relocation_native_adapter.v1"
DIAGNOSTIC_SCHEMA_VERSION = (
    "task_evaluation_rigid_relocation_diagnostic_native_adapter.v1"
)
DEFINITION_CONTRACT_PATH = "scene.configured_revision.task_template.definition"
SUCCESS_CONTRACT_PATH = (
    "scene.configured_revision.task_template.success_criteria"
)
EXECUTION_CONTRACT_PATH = "scene.configured_revision.task_template.execution"
SUPPORT_PLANE_CONTRACT_PATH = (
    "scene.configured_revision.registration.support_plane"
)
SOURCE_OBJECT_CONTRACT_PATH = (
    "scene.configured_revision.replacement.source_object"
)
STATIC_QUALIFICATION_CONTRACT_PATH = (
    "scene.configured_revision.replacement.static_qualification"
)
NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH = (
    "scene.configured_revision.replacement.native_import_qualification"
)
SOURCE_SCHEMAS = {
    DEFINITION_CONTRACT_PATH: "task_evaluation_rigid_relocation_template.v1",
    SUCCESS_CONTRACT_PATH: (
        "task_evaluation_rigid_relocation_success_criteria.v1"
    ),
    EXECUTION_CONTRACT_PATH: (
        "task_evaluation_rigid_relocation_execution_spec.v1"
    ),
    SUPPORT_PLANE_CONTRACT_PATH: "task_evaluation_support_plane_input.v1",
    SOURCE_OBJECT_CONTRACT_PATH: "task_evaluation_source_object_selection.v1",
    STATIC_QUALIFICATION_CONTRACT_PATH: (
        "task_evaluation_rigid_replacement_static_qualification.v1"
    ),
    NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH: (
        "task_evaluation_replacement_native_import_result.v1"
    ),
}
NATIVE_PHYSICS_FREQUENCY_HZ = 120
# Closed-hand geometry of the fixed DROID Robotiq 2F-85 embodiment, measured
# from paid-run readbacks rather than assumed from the vendor model.  Scene
# 839873 franka-controls attempt 001 (instance 49506537) established both
# numbers: object contact began when the commanded pinch centre was still
# 39 mm short of the pushed face (the closed fingertips protrude that far
# along the approach), and the 59.6 N robot-scene graze at the bottom of the
# precontact descent puts the hand's collision envelope ~55 mm below the
# pinch centre.  Both stay bracketed by measurements on every subsequent run:
# ``push_contact_standoff`` refuses an understated forward offset,
# ``push_contact`` refuses an overstated one, and ``base_collision_clearance``
# refuses an understated support envelope.
ROBOTIQ_2F85_CLOSED_FINGERTIP_FORWARD_OFFSET_M = 0.040
CLOSED_HAND_SUPPORT_ENVELOPE_BELOW_GRASP_FRAME_M = 0.055
# Commanded interference held through the push so the contact normal stays
# measurable at equilibrium; position control cannot hold force at exact
# tangency, and ``push_contact_maintained`` requires force on every sample.
PUSH_CONTACT_INTERFERENCE_M = 0.005
# Margin over the authored support clearance covering descent-tolerance sag
# (the 2 cm arrival tolerance let attempt 001 settle 8 mm low).
PUSH_SUPPORT_CLEARANCE_MARGIN_M = 0.02
# Object displacement allowed while push_contact establishes the commanded
# interference; attempt 001 measured 31.5 mm from the uncorrected frame.
PUSH_CONTACT_MAX_DISPLACEMENT_M = 0.02
DIAGNOSTIC_SOURCE_ALIASES = {
    DEFINITION_CONTRACT_PATH: "task.definition",
    SUCCESS_CONTRACT_PATH: "task.success_criteria",
    EXECUTION_CONTRACT_PATH: "task.execution",
    SUPPORT_PLANE_CONTRACT_PATH: "scene.registration.support_plane",
    SOURCE_OBJECT_CONTRACT_PATH: "task.subject.source_object",
    STATIC_QUALIFICATION_CONTRACT_PATH: (
        "diagnostic_output.static_qualification_receipt.v1.json"
    ),
    NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH: (
        "diagnostic_output.native_import_qualification_receipt.v1.json"
    ),
}


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


def _runtime_geometry(
    *,
    replacement_identity: Mapping[str, Any],
    support: Mapping[str, Any],
    source_object: Mapping[str, Any],
    static: Mapping[str, Any],
    native_import: Mapping[str, Any],
    start: Sequence[float],
    target: Sequence[float],
    strategy: str,
) -> dict[str, Any]:
    observed = static.get("observed_structure")
    if (
        static.get("status") != "authored_structure_statically_qualified"
        or static.get("replacement_identity") != replacement_identity
        or static.get("result_digest")
        != canonical_digest(static, digest_field="result_digest")
        or not isinstance(observed, Mapping)
        or source_object.get("status")
        != "frozen_before_scene_configuration_run"
        or source_object.get("center_xyz_m") != list(start)
        or native_import.get("status") != "qualified"
        or native_import.get("replacement_identity") != replacement_identity
        or native_import.get("native_simulator_import_qualified") is not True
        or native_import.get("blockers") not in ([], ())
        or native_import.get("result_digest")
        != canonical_digest(native_import, digest_field="result_digest")
        or support.get("status")
        != "frozen_candidate_pending_production_validation"
        or not str(support.get("sage_prim_path") or "").startswith("/")
    ):
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_executable_geometry_missing"
        )
    source_lower = _vector(
        source_object.get("aabb_min_xyz_m"), field="source_object.aabb_min"
    )
    source_upper = _vector(
        source_object.get("aabb_max_xyz_m"), field="source_object.aabb_max"
    )
    support_upper = _vector(
        support.get("bounds_max_xyz_m"), field="support.bounds_max"
    )
    try:
        support_top = float(support.get("top_z_m"))
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_support_top_invalid"
        ) from exc
    if not math.isfinite(support_top) or not math.isclose(
        support_top, support_upper[2], rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_support_top_invalid"
        )
    lower = [source_lower[index] - float(start[index]) for index in range(3)]
    upper = [source_upper[index] - float(start[index]) for index in range(3)]
    if any(low >= high for low, high in zip(lower, upper, strict=True)):
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_executable_geometry_missing"
        )
    rigid_paths = observed.get("rigid_body_paths")
    if (
        not isinstance(rigid_paths, list)
        or len(rigid_paths) != 1
        or not str(rigid_paths[0]).startswith("/")
    ):
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_executable_geometry_missing"
        )
    delta = [float(target[index]) - float(start[index]) for index in range(3)]
    if strategy == "planar_push" and abs(delta[2]) > 1.0e-9:
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_planar_height_mismatch"
        )
    horizontal_norm = math.hypot(delta[0], delta[1])
    if horizontal_norm <= 0.0:
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_push_direction_invalid"
        )
    push = [delta[0] / horizontal_norm, delta[1] / horizontal_norm, 0.0]
    try:
        center = [float(value) for value in observed["center_of_mass_m"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_executable_geometry_missing"
        ) from exc
    if len(center) != 3 or not all(math.isfinite(value) for value in center):
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_executable_geometry_missing"
        )
    # The contact point is the support-function minimum of the qualified
    # collision AABB along the push direction. The gripper approaches farther
    # from the object along the opposite direction, then follows +push.
    contact = [
        lower[index]
        if push[index] > 0.0
        else upper[index]
        if push[index] < 0.0
        else 0.0
        for index in range(3)
    ]
    # Author the push height so the closed hand's collision envelope clears
    # the support.  ``contact`` is in the scoring frame, so the pinch centre's
    # world height is ``start_z + contact[2]``; at the object's centre height
    # attempt 001 measured a 59.6 N robot-scene graze at the bottom of the
    # precontact descent.  Raise the contact point up the pushed face until
    # the envelope clears, and refuse the task when the face is too short to
    # push at a clearing height.
    if strategy == "planar_push":
        required_contact_z_world = (
            support_top
            + CLOSED_HAND_SUPPORT_ENVELOPE_BELOW_GRASP_FRAME_M
            + PUSH_SUPPORT_CLEARANCE_MARGIN_M
        )
        contact[2] = max(contact[2], required_contact_z_world - float(start[2]))
        if contact[2] > float(upper[2]) - 0.005:
            raise TaskEvaluationRigidRelocationNativeAdapterError(
                "rigid_relocation_native_adapter_push_support_clearance_unauthorable"
            )
    root_position = [float(start[index]) - center[index] for index in range(3)]
    # The task start is a scoring-frame center, while Isaac spawns the rigid
    # asset at its authored root.  Those frames are normally related by the
    # qualified center of mass, but observed source bounds and generated
    # replacement geometry can differ by sub-millimetres.  Scene 839873 paid
    # for that distinction: the old center-only conversion placed the
    # replacement 0.18 mm through the support, and PhysX tipped it before the
    # robot made contact.  Align the replacement's inferred local minimum to
    # the registered support top instead of allowing any initial penetration.
    source_minimum_scoring_z = source_lower[2] - float(start[2])
    replacement_minimum_root_z = center[2] + source_minimum_scoring_z
    support_aligned_root_z = support_top - replacement_minimum_root_z
    root_position[2] = max(root_position[2], support_aligned_root_z)
    return {
        "center_body_frame_m": center,
        "root_position_world_m": root_position,
        "support_alignment": {
            "support_top_z_m": support_top,
            "source_minimum_scoring_z_m": source_minimum_scoring_z,
            "replacement_minimum_root_z_m": replacement_minimum_root_z,
            "support_aligned_root_z_m": support_aligned_root_z,
            "initial_support_penetration_permitted": False,
        },
        "contact_point_scoring_frame_m": contact,
        "approach_unit_scoring_frame": [-push[0], -push[1], 0.0],
        "allowed_contact_prim_paths": [str(rigid_paths[0])],
        "intended_support_prim_paths": [str(support["sage_prim_path"])],
    }


def adapt_rigid_relocation_task_template(
    *,
    request: Mapping[str, Any] | None = None,
    configured_revision: Mapping[str, Any] | None = None,
    materialized_references: Mapping[str, Mapping[str, Any]],
    diagnostic_controls_input: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one digest-bound native view of exact configured task bytes.

    The diagnostic authority is deliberately separate from a configured scene
    revision. It can compile construction/controls inputs but cannot be passed
    through the production qualification compiler or acquire a revision digest.
    """

    diagnostic = diagnostic_controls_input is not None
    documents: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    if diagnostic:
        authority = dict(diagnostic_controls_input or {})
        if (
            request is not None
            or configured_revision is not None
            or authority.get("schema_version")
            != "task_evaluation_configured_scene_diagnostic_controls_input.v1"
            or authority.get("status") != "materialized"
            or authority.get("qualification_eligible") is not False
            or authority.get("configured_revision_publication_permitted") is not False
            or authority.get("evaluation_ready_promotion_permitted") is not False
            or authority.get("claim_ceiling")
            != "development_only_downstream_construction_and_controls_diagnostic"
            or authority.get("receipt_digest")
            != canonical_digest(authority, digest_field="receipt_digest")
        ):
            raise TaskEvaluationRigidRelocationNativeAdapterError(
                "rigid_relocation_native_adapter_diagnostic_authority_invalid"
            )
        rows = authority.get("materialized_inputs")
        by_contract = {
            str(row.get("contract_path") or ""): row
            for row in rows or []
            if isinstance(row, Mapping)
        }
        provider_row = by_contract.get(
            "diagnostic_output.task_evaluation_scene_configuration_provider_result.v1.json"
        )
        if not isinstance(provider_row, Mapping):
            raise TaskEvaluationRigidRelocationNativeAdapterError(
                "rigid_relocation_native_adapter_diagnostic_authority_invalid"
            )
        provider_path = Path(str(provider_row.get("path") or "")).expanduser()
        try:
            provider_result = json.loads(provider_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TaskEvaluationRigidRelocationNativeAdapterError(
                "rigid_relocation_native_adapter_diagnostic_provider_result_invalid"
            ) from exc
        if (
            provider_path.is_symlink()
            or not provider_path.is_file()
            or _sha256_and_size(provider_path)
            != (provider_row.get("digest"), provider_row.get("size_bytes"))
            or provider_row.get("full_byte_readback_passed") is not True
            or provider_result.get("schema_version")
            != "task_evaluation_scene_configuration_diagnostic_provider_result.v1"
            or provider_result.get("status")
            != "completed_diagnostic_only_not_qualification_eligible"
            or provider_result.get("diagnostic_only") is not True
            or provider_result.get("qualification_eligible") is not False
            or provider_result.get("executed_inside_one_parent_provider_run") is not False
            or provider_result.get("configured_revision_publication_permitted") is not False
            or provider_result.get("result_digest")
            != canonical_digest(provider_result, digest_field="result_digest")
        ):
            raise TaskEvaluationRigidRelocationNativeAdapterError(
                "rigid_relocation_native_adapter_diagnostic_provider_result_invalid"
            )
        diagnostic_references: dict[str, dict[str, Any]] = {}
        for contract_path, source_alias in DIAGNOSTIC_SOURCE_ALIASES.items():
            row = by_contract.get(source_alias)
            if not isinstance(row, Mapping):
                raise TaskEvaluationRigidRelocationNativeAdapterError(
                    f"rigid_relocation_native_adapter_source_invalid:{contract_path}"
                )
            diagnostic_references[contract_path] = {
                "contract_path": contract_path,
                "uri": row.get("uri") or f"diagnostic-input://{source_alias}",
                "digest": row.get("digest"),
                "size_bytes": row.get("size_bytes"),
                "materialized_path": row.get("path"),
                "full_byte_service_account_readback_passed": row.get(
                    "full_byte_readback_passed"
                ),
            }
            document, binding = _source_document(
                diagnostic_references,
                contract_path=contract_path,
                expected_reference=diagnostic_references[contract_path],
            )
            documents[contract_path] = document
            bindings.append(binding)
        template = documents[DEFINITION_CONTRACT_PATH]
        task = {
            "identity": template.get("task_identity"),
            "subject": {"identity": template.get("object_identity")},
            "strategy": template.get("strategy"),
        }
        replacement_identity = template.get("object_identity")
        authority_digest = authority["receipt_digest"]
    else:
        try:
            validated_request = validate_launch_preparation_request(request or {})
            revision = validate_configured_scene_revision(configured_revision or {})
        except (
            TaskEvaluationLaunchPreparationContractError,
            TaskEvaluationConfiguredSceneRevisionError,
        ) as exc:
            raise TaskEvaluationRigidRelocationNativeAdapterError(
                "rigid_relocation_native_adapter_authority_invalid"
            ) from exc
        task = validated_request["task"]
        if (
            validated_request["run_mode"]
            not in {"episode_evaluation", "destination_qualification"}
            or task["binding_mode"] != "reuse_configured_template"
            or task["kind"] != "rigid_relocation"
            or task["strategy"] not in {"planar_push", "pick_and_place"}
            or task["identity"] != revision["task_template"]["identity"]
            or task["subject"]["identity"] != revision["replacement"]["identity"]
            or task["configured_scene_revision_digest"] != revision["revision_digest"]
        ):
            raise TaskEvaluationRigidRelocationNativeAdapterError(
                "rigid_relocation_native_adapter_request_binding_mismatch"
            )
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
        for contract_path, expected_reference in (
            (SUPPORT_PLANE_CONTRACT_PATH, revision["registration"]["support_plane"]),
            (SOURCE_OBJECT_CONTRACT_PATH, revision["replacement"]["source_object"]),
            (
                STATIC_QUALIFICATION_CONTRACT_PATH,
                revision["replacement"]["static_qualification"],
            ),
            (
                NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH,
                revision["replacement"]["native_import_qualification"],
            ),
        ):
            document, binding = _source_document(
                materialized_references,
                contract_path=contract_path,
                expected_reference=expected_reference,
            )
            documents[contract_path] = document
            bindings.append(binding)
        replacement_identity = revision["replacement"]["identity"]
        authority_digest = revision["revision_digest"]

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
        or template.get("strategy") != task["strategy"]
        or execution.get("strategy") != task["strategy"]
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
    strategy = str(task["strategy"])
    if strategy == "planar_push" and not math.isclose(
        start[2], target[2], rel_tol=0.0, abs_tol=1.0e-9
    ):
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
    geometry = _runtime_geometry(
        replacement_identity=replacement_identity,
        support=documents[SUPPORT_PLANE_CONTRACT_PATH],
        source_object=documents[SOURCE_OBJECT_CONTRACT_PATH],
        static=documents[STATIC_QUALIFICATION_CONTRACT_PATH],
        native_import=documents[NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH],
        start=start,
        target=target,
        strategy=strategy,
    )
    source_documents = {
        "bindings": bindings,
        "documents": {
            "definition": template,
            "success_criteria": success,
            "execution": execution,
            "support_plane": documents[SUPPORT_PLANE_CONTRACT_PATH],
            "source_object": documents[SOURCE_OBJECT_CONTRACT_PATH],
            "static_qualification": documents[STATIC_QUALIFICATION_CONTRACT_PATH],
            "native_import_qualification": documents[
                NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH
            ],
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
    support = documents[SUPPORT_PLANE_CONTRACT_PATH]
    support_minimum = _vector(
        support.get("bounds_min_xyz_m"), field="support.bounds_min"
    )
    support_maximum = _vector(
        support.get("bounds_max_xyz_m"), field="support.bounds_max"
    )
    if any(
        low >= high
        for low, high in zip(support_minimum, support_maximum, strict=True)
    ):
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_support_bounds_invalid"
        )
    configured_affordance = template.get("interaction_affordance")
    if strategy == "pick_and_place":
        if not isinstance(configured_affordance, Mapping):
            raise TaskEvaluationRigidRelocationNativeAdapterError(
                "rigid_relocation_native_adapter_pick_affordance_missing"
            )
        contact_point = _vector(
            configured_affordance.get("contact_point_scoring_frame_m"),
            field="interaction_affordance.contact_point",
        )
        outward = _vector(
            configured_affordance.get("approach_unit_scoring_frame"),
            field="interaction_affordance.approach_unit",
        )
        jaw_axis = _vector(
            configured_affordance.get("jaw_unit_scoring_frame"),
            field="interaction_affordance.jaw_unit",
        )
        lift_unit = _vector(
            configured_affordance.get("lift_unit_world"),
            field="interaction_affordance.lift_unit",
        )
        pregrasp_clearance = _positive_number(
            configured_affordance.get("pregrasp_clearance_m"),
            field="interaction_affordance.pregrasp_clearance_m",
        )
        minimum_lift = _positive_number(
            configured_affordance.get("minimum_lift_m"),
            field="interaction_affordance.minimum_lift_m",
        )
        for field, vector in (
            ("approach_unit", outward),
            ("jaw_unit", jaw_axis),
            ("lift_unit", lift_unit),
        ):
            if not math.isclose(
                math.sqrt(sum(value * value for value in vector)),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise TaskEvaluationRigidRelocationNativeAdapterError(
                    "rigid_relocation_native_adapter_pick_affordance_invalid:"
                    + field
                )
        contact_approach = [-float(value) for value in outward]
        if abs(sum(a * b for a, b in zip(contact_approach, jaw_axis, strict=True))) > 1e-6:
            raise TaskEvaluationRigidRelocationNativeAdapterError(
                "rigid_relocation_native_adapter_pick_affordance_invalid:jaw_axis"
            )
    else:
        contact_point = geometry["contact_point_scoring_frame_m"]
        outward = geometry["approach_unit_scoring_frame"]
        contact_approach = [-float(value) for value in outward]
        jaw_axis = [-contact_approach[1], contact_approach[0], 0.0]
        lift_unit = [0.0, 0.0, 1.0]
        pregrasp_clearance = 0.12
        minimum_lift = 0.0
    interaction_affordance: dict[str, Any] = {
        "schema_version": "native_rigid_interaction_affordance.v1",
        "subject_asset_id": task["subject"]["identity"]["id"],
        "scoring_frame_id": "task_scoring_frame",
        "asset_root_from_scoring_frame": {
            "position_m": geometry["center_body_frame_m"],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "contact_point_scoring_frame_m": contact_point,
        "approach_unit_scoring_frame": outward,
        "lift_unit_world": lift_unit,
        "gripper_orientation_scoring_frame_xyzw": [],
        "pregrasp_clearance_m": pregrasp_clearance,
        "arrival_orientation_tolerance_rad": 0.08,
        "allowed_contact_prim_paths": geometry["allowed_contact_prim_paths"],
        "intended_support_prim_paths": geometry[
            "intended_support_prim_paths"
        ],
        "support_alignment": geometry["support_alignment"],
        "affordance_digest": "",
    }
    if strategy == "planar_push":
        interaction_affordance.update(
            closed_fingertip_forward_offset_m=(
                ROBOTIQ_2F85_CLOSED_FINGERTIP_FORWARD_OFFSET_M
            ),
            push_contact_interference_m=PUSH_CONTACT_INTERFERENCE_M,
        )
    # ``approach_unit_scoring_frame`` points outward from the contact face;
    # the gripper's +Z approach axis must point the other way, into the object.
    # Keep the parallel-jaw axis horizontal and tangent to the planar push so
    # neither finger is authored through the support.  Identity here is not a
    # harmless default: it points +Z upward and made native Isaac spend every
    # phase rotating toward an unrelated frame instead of reaching the mug.
    try:
        interaction_affordance[
            "gripper_orientation_scoring_frame_xyzw"
        ] = grasp_orientation_contact_xyzw(
            approach_axis=contact_approach,
            jaw_axis=jaw_axis,
        )
    except NativeFrankaActionMathError as exc:
        raise TaskEvaluationRigidRelocationNativeAdapterError(
            "rigid_relocation_native_adapter_gripper_orientation_unauthorable:"
            + ";".join(exc.errors)
        ) from exc
    interaction_affordance["affordance_digest"] = canonical_digest(
        interaction_affordance, digest_field="affordance_digest"
    )
    native_task_spec = {
        "schema_version": "adp_task_spec.v2",
        "task_kind": "rigid_pick_place",
        "manipulation_strategy": strategy,
        "subject_asset_id": task["subject"]["identity"]["id"],
        "prompt": (
            "Pick up the configured rigid object and place it at the registered target."
            if strategy == "pick_and_place"
            else "Move the configured rigid object to the registered target by planar push."
        ),
        "start_pose_world": [*start, 0.0, 0.0, 0.0, 1.0],
        "target_position_world_m": target,
        "destination_position_tolerance_m": target_tolerance,
        "destination_position_bounds_world_m": {
            "minimum": [
                target[0] - target_tolerance,
                target[1] - target_tolerance,
                target[2] - 0.01,
            ],
            "maximum": [
                target[0] + target_tolerance,
                target[1] + target_tolerance,
                target[2] + 0.01,
            ],
        },
        "support_height_interval_m": [target[2] - 0.01, target[2] + 0.01],
        "destination_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "destination_orientation_tolerance_rad": 0.08,
        "minimum_translation_m": minimum_displacement,
        "minimum_lift_m": minimum_lift,
        "movement_epsilon_m": min(0.005, minimum_displacement / 10.0),
        "control_frequency_hz": control_frequency,
        "maximum_action_steps": maximum_steps,
        "settle_window_samples": 20,
        "maximum_episode_seconds": maximum_seconds,
        "release_required": True,
        "release_gripper_width_min_m": 0.06,
        "task_contact_minimum_force_n": 0.5,
        "collision_failure_minimum_force_n": 1.0,
        "reset_translation_tolerance_m": 0.002,
        "reset_orientation_tolerance_rad": 0.01,
        "settle_position_tolerance_m": 0.005,
        "settle_orientation_tolerance_rad": 0.03,
        "relocation_tracking_tolerance_m": target_tolerance,
        "workspace_position_bounds_world_m": {
            "minimum": [support_minimum[0], support_minimum[1], min(start[2], target[2]) - 0.25],
            "maximum": [
                support_maximum[0],
                support_maximum[1],
                max(start[2], target[2]) + minimum_lift + 0.25,
            ],
        },
        "interaction_affordance": interaction_affordance,
        "action_bounds_m_per_step": {
            "minimum": action_minimum,
            "maximum": action_maximum,
        },
        "configured_success_criteria": _success_payload(success),
        "configured_task_source_documents_digest": source_documents[
            "source_documents_digest"
        ],
    }
    if "instruction" in template:
        instruction = template["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise TaskEvaluationRigidRelocationNativeAdapterError(
                "rigid_relocation_native_adapter_instruction_invalid")
        native_task_spec["prompt"] = instruction
        for label in ("instruction_subject_label", "visible_target_label"):
            value = template.get(label)
            if not isinstance(value, str) or not value.strip():
                raise TaskEvaluationRigidRelocationNativeAdapterError(
                    "rigid_relocation_native_adapter_instruction_grounding_missing")
            native_task_spec[label] = value
    if "retreat_clearance_m" in success:
        native_task_spec["retreat_clearance_m"] = _positive_number(
            success["retreat_clearance_m"], field="success.retreat_clearance_m"
        )
    if "owner_success_contract_authority" in template:
        native_task_spec["configured_owner_authority"] = dict(
            template["owner_success_contract_authority"]
        )
    if strategy == "planar_push":
        native_task_spec["push_contact_max_displacement_m"] = (
            PUSH_CONTACT_MAX_DISPLACEMENT_M
        )
    native_definition = {
        "schema_version": "task_evaluation_native_task_definition.v1",
        "identity": dict(task["identity"]),
        "task_spec": native_task_spec,
        "task_object_pose_world": {
            "position_world_m": geometry["root_position_world_m"],
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
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION if diagnostic else SCHEMA_VERSION,
        "status": "adapted",
        "external_task_kind": "rigid_relocation",
        "native_task_kind": "rigid_pick_place",
        "manipulation_strategy": strategy,
        "source_documents": source_documents,
        "native_task_definition": native_definition,
        "native_success_criteria": native_success,
        "native_episode_execution": native_execution,
        "adapter_digest": "",
    }
    if diagnostic:
        result.update(
            diagnostic_controls_input_receipt_digest=authority_digest,
            claim_ceiling=(
                "development_only_downstream_construction_and_controls_diagnostic"
            ),
            qualification_eligible=False,
            configured_revision_publication_permitted=False,
        )
    else:
        result["configured_scene_revision_digest"] = authority_digest
    result["adapter_digest"] = canonical_digest(
        result, digest_field="adapter_digest"
    )
    return result


__all__ = [
    "DIAGNOSTIC_SCHEMA_VERSION",
    "EXECUTION_CONTRACT_PATH",
    "NATIVE_PHYSICS_FREQUENCY_HZ",
    "SCHEMA_VERSION",
    "SUCCESS_CONTRACT_PATH",
    "TaskEvaluationRigidRelocationNativeAdapterError",
    "adapt_rigid_relocation_task_template",
]
