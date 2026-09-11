"""Small pure validators extracted from the configured-controls autostart spine."""

from __future__ import annotations

import json
import math
import re
import stat
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_configured_controls_autostart_support import (
    TaskEvaluationConfiguredControlsAutostartError, _read, _sha256,
)
from .task_evaluation_robot_placement_agent import ROBOT_PLACEMENT_AGENT_MODEL, ROBOT_PLACEMENT_AGENT_REASONING_EFFORT
from .task_evaluation_shared_mutation_window import TaskEvaluationSharedMutationWindowError, validate_shared_mutation_window_template
from .task_evaluation_configured_controls_openai_placement import (
    PAID_RESOURCE_CLASS as OPENAI_PLACEMENT_PAID_RESOURCE_CLASS, VISUAL_REVIEW_CREDENTIAL_ROLE,
)
from . import task_evaluation_configured_controls_destination_phases as destination_phases
from . import task_evaluation_deferred_controls_contract as deferred_inputs
from . import task_evaluation_openai_usage_validation as inference_usage


from collections.abc import Callable, Mapping
from typing import Any


def configuration_adoption_valid(
    *,
    adoption: Mapping[str, Any],
    source_launch_id: str,
    terminal: Mapping[str, Any],
    receipt: Mapping[str, Any],
    revision: Mapping[str, Any],
    publication: Mapping[str, Any],
    sync: Mapping[str, Any],
    zero: Mapping[str, Any],
) -> bool:
    return bool(
        adoption.get("mode") == "explicit_terminal_adoption"
        and adoption.get("source_launch_id") == source_launch_id
        and adoption.get("source_launch_receipt_digest")
        == receipt.get("receipt_digest")
        and adoption.get("terminal_result_digest") == terminal.get("result_digest")
        and adoption.get("configured_scene_revision_digest")
        == revision.get("revision_digest")
        and adoption.get("publication_result_digest")
        == publication.get("result_digest")
        and adoption.get("webapp_sync_result_digest")
        == sync.get("sync_result_digest")
        and adoption.get("provider_zero_receipt_digest")
        == zero.get("provider_zero_receipt_digest")
    )


def configuration_adoption_validator(
    error_type: type[Exception],
) -> Callable[..., None]:
    def require(**kwargs: Any) -> None:
        if not configuration_adoption_valid(**kwargs):
            raise error_type("configured_controls_autostart_adoption_evidence_invalid")

    return require


__all__ = ["configuration_adoption_valid", "configuration_adoption_validator"]


INTENT_SCHEMA_VERSION = destination_phases.LEGACY_INTENT_SCHEMA_VERSION


RESULT_SCHEMA_VERSION = "task_evaluation_configured_controls_autostart.v3"


DEFAULT_MAX_PLACEMENT_INFERENCE_COST_USD = 2.56


_COMMIT = re.compile(r"[0-9a-f]{40}")


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


_FIXED_PATHS = {
    "robot_asset_usd_path",
    "robot_mount_interface_path",
    "scene_camera_calibration_path",
    "native_trajectory_plan_path",
    "cameras_path",
    "runtime_binding_path",
}


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_artifact_invalid"
        )
    metadata = path.stat()
    return {
        "path": str(path),
        "digest": _sha256(path),
        "size_bytes": metadata.st_size,
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
    }


def _intent_paths(value: Mapping[str, Any]) -> dict[str, Path]:
    paths = value.get("paths")
    phases = value.get("phases")
    expected_phases = destination_phases.phase_paths(phases)
    try:
        declared_deferred = deferred_inputs.deferred_declarations(paths)
    except deferred_inputs.ConfiguredControlsDeferredInputError as exc:
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_intent_paths_invalid"
        ) from exc
    overview_deferred = "overview_image_paths" in declared_deferred
    if (
        not isinstance(paths, Mapping)
        or set(paths) != _FIXED_PATHS | {"overview_image_paths"}
        or (
            not overview_deferred
            and (
                not isinstance(paths.get("overview_image_paths"), list)
                or not paths["overview_image_paths"]
            )
        )
        or expected_phases is None
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_intent_paths_invalid"
        )
    flattened = {
        name: Path(str(paths[name])).expanduser()
        for name in _FIXED_PATHS
        if name not in declared_deferred
    }
    if not overview_deferred:
        for index, item in enumerate(paths["overview_image_paths"]):
            flattened[f"overview_image_paths.{index}"] = Path(str(item)).expanduser()
    for phase, expected in expected_phases.items():
        row = phases.get(phase)
        if not isinstance(row, Mapping) or set(row) != expected:
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_intent_paths_invalid"
            )
        for name in expected:
            flattened[f"phases.{phase}.{name}"] = Path(str(row[name])).expanduser()
    if any(not path.is_absolute() for path in flattened.values()):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_intent_paths_invalid"
        )
    return flattened


def validate_configured_controls_autostart_intent(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    intent = json.loads(json.dumps(dict(value), allow_nan=False))
    target = intent.get("target_position_world_m")
    placement = intent.get("placement")
    placement_authority = (
        placement.get("official_cost_authority")
        if isinstance(placement, Mapping)
        else None
    )
    adoption = intent.get("configuration_adoption")
    review_continuation = intent.get('visual_review_continuation')
    completed_adoption = intent.get('completed_placement_adoption')
    expected_inference_cap = DEFAULT_MAX_PLACEMENT_INFERENCE_COST_USD
    if completed_adoption is not None:
        from .task_evaluation_retained_controls_evidence import validate_placement_adoption as validate_adoption
        validate_adoption(completed_adoption)
        if completed_adoption.get('execution_commit') != intent.get('expected_production_commit') or review_continuation is not None:
            raise TaskEvaluationConfiguredControlsAutostartError('configured_controls_completed_adoption_invalid')
        expected_inference_cap = 0.0
    if review_continuation is not None:
        from . import task_evaluation_retained_controls_evidence as visual
        visual.validate_visual_continuation(review_continuation,expected_commit=intent.get('expected_production_commit'))
        expected_inference_cap = visual.REVIEW_CAP
    if (
        intent.get("schema_version") not in destination_phases.INTENT_SCHEMA_VERSIONS
        or intent.get("enabled") is not True
        or _COMMIT.fullmatch(str(intent.get("expected_production_commit") or "")) is None
        or _COMMIT.fullmatch(str(intent.get("configuration_source_commit") or "")) is None
        or not str(intent.get("submitted_by") or "").strip()
        or not str(intent.get("team_namespace") or "").strip()
        or not str(intent.get("scene_id") or "").strip()
        or not str(intent.get("task_id") or "").strip()
        or not isinstance(target, list)
        or len(target) != 3
        or not all(math.isfinite(float(item)) for item in target)
        or not isinstance(placement, Mapping)
        or set(placement)
        != {
            "robot_id",
            "max_rounds",
            "candidate_inventory_cap",
            "max_input_tokens",
            "max_inference_cost_usd",
            *inference_usage.PROMPT_CACHE_INTENT_FIELDS,
            "agent_selection_required",
            "agent_model",
            "reasoning_effort",
            "official_cost_authority",
        }
        or placement.get("robot_id") != "franka_panda"
        or not (int(placement.get('max_rounds', -1)) == 0 if completed_adoption is not None
                else 1 <= int(placement.get('max_rounds', 0)) <= 8)
        or not 1 <= int(placement.get("candidate_inventory_cap", 0)) <= 128
        or not 1 <= int(placement.get("max_input_tokens", 0)) <= 1_000_000
        or float(placement.get("max_inference_cost_usd", -1.0))
        != expected_inference_cap
        or (review_continuation is not None and (placement.get('max_rounds') != 1
            or placement.get('max_input_tokens') != visual.MAX_INPUT_TOKENS))
        or not inference_usage.prompt_cache_placement_intent_valid(placement)
        or placement.get("agent_selection_required") is not True
        or placement.get("agent_model") != ROBOT_PLACEMENT_AGENT_MODEL
        or placement.get("reasoning_effort")
        != ROBOT_PLACEMENT_AGENT_REASONING_EFFORT
        or not isinstance(placement_authority, Mapping)
        or set(placement_authority)
        != {
            "provider_id",
            "credential_role",
            "project_id",
            "api_key_id",
            "paid_resource_class",
            "maximum_cost_usd",
        }
        or placement_authority.get("provider_id") != "openai"
        or placement_authority.get("credential_role")
        != VISUAL_REVIEW_CREDENTIAL_ROLE
        or not str(placement_authority.get("project_id") or "").strip()
        or not str(placement_authority.get("api_key_id") or "").strip()
        or placement_authority.get("paid_resource_class")
        != OPENAI_PLACEMENT_PAID_RESOURCE_CLASS
        or float(placement_authority.get("maximum_cost_usd", -1.0))
        != float(placement["max_inference_cost_usd"])
        or not Path(str(intent.get("profile_dir") or "")).is_absolute()
        or intent.get("provider_mutation_performed") is not False
        or intent.get("paid_execution_requested") is not True
        or not isinstance(adoption, Mapping)
        or intent.get("intent_digest")
        != canonical_digest(intent, digest_field="intent_digest")
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_intent_invalid"
        )
    expected_schema = destination_phases.schema_for_phases(intent.get("phases"))
    if intent.get("schema_version") != expected_schema:
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_intent_invalid"
        )
    if adoption.get("mode") == "same_commit_automatic":
        if (
            set(adoption) != {"mode"}
            or intent["configuration_source_commit"]
            != intent["expected_production_commit"]
        ):
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_adoption_invalid"
            )
    elif adoption.get("mode") == "explicit_terminal_adoption":
        if (
            set(adoption)
            != {
                "mode",
                "source_launch_id",
                "source_launch_receipt_digest",
                "terminal_result_digest",
                "configured_scene_revision_digest",
                "publication_result_digest",
                "webapp_sync_result_digest",
                "provider_zero_receipt_digest",
            }
            or not str(adoption.get("source_launch_id") or "").strip()
            or any(
                _DIGEST.fullmatch(str(adoption.get(field) or "")) is None
                for field in (
                    "source_launch_receipt_digest",
                    "terminal_result_digest",
                    "configured_scene_revision_digest",
                    "publication_result_digest",
                    "webapp_sync_result_digest",
                    "provider_zero_receipt_digest",
                )
            )
        ):
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_adoption_invalid"
            )
    else:
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_adoption_invalid"
        )
    flattened = _intent_paths(intent)
    inventory = intent.get("artifact_inventory")
    if not isinstance(inventory, Mapping) or set(inventory) != set(flattened):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_inventory_invalid"
        )
    for name, path in flattened.items():
        row = inventory.get(name)
        if not isinstance(row, Mapping) or dict(row) != _artifact(path):
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_inventory_invalid"
            )
    for phase in intent["phases"]:
        try:
            validate_shared_mutation_window_template(
                _read(
                    flattened[f"phases.{phase}.release_window_template_path"],
                    blocker="configured_controls_autostart_release_window_template_invalid",
                ),
                team_namespace=str(intent["team_namespace"]),
                expected_production_commit=str(intent["expected_production_commit"]),
            )
        except TaskEvaluationSharedMutationWindowError as exc:
            raise TaskEvaluationConfiguredControlsAutostartError(
                "configured_controls_autostart_release_window_template_invalid"
            ) from exc
    return intent


def _validate_result(
    value: Mapping[str, Any],
    *,
    expected_intent_digest: str,
    expected_scene_binding_digest: str,
    expected_task_binding_digest: str,
    expected_cpu_checkpoint_binding_digest: str,
) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(value), allow_nan=False))
    if result.get('completed_placement_adoption') is not None:
        from .task_evaluation_retained_controls_evidence import validate_placement_adoption as validate_adoption
        validate_adoption(result['completed_placement_adoption'])
        if result.get('placement_calls_reexecuted') is not False:
            raise TaskEvaluationConfiguredControlsAutostartError('configured_controls_completed_adoption_invalid')
    openai_evidence = result.get("official_openai_cost_evidence")
    native_universe = result.get("native_construction_candidate_universe")
    native_universe_path = (
        Path(str(native_universe.get("path") or ""))
        if isinstance(native_universe, Mapping)
        else Path()
    )
    if (
        result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("status") != "agent_binding_accepted_plan_materialized"
        or not str(result.get("source_launch_id") or "")
        or result.get("intent_digest") != expected_intent_digest
        or result.get("scene_binding_digest")
        != expected_scene_binding_digest
        or result.get("task_binding_digest") != expected_task_binding_digest
        or result.get("cpu_placement_checkpoint_binding_digest")
        != expected_cpu_checkpoint_binding_digest
        or _DIGEST.fullmatch(
            str(result.get("configured_scene_revision_digest") or "")
        )
        is None
        or _DIGEST.fullmatch(str(result.get("trajectory_digest") or "")) is None
        or _DIGEST.fullmatch(
            str(result.get("candidate_inventory_digest") or "")
        )
        is None
        or not str(result.get("selected_candidate_id") or "")
        or _DIGEST.fullmatch(
            str(result.get("cpu_inventory_ranker_receipt_digest") or "")
        )
        is None
        or _DIGEST.fullmatch(
            str(result.get("placement_agent_receipt_digest") or "")
        )
        is None
        or result.get("placement_agent_model") != ROBOT_PLACEMENT_AGENT_MODEL
        or result.get("placement_agent_reasoning_effort")
        != ROBOT_PLACEMENT_AGENT_REASONING_EFFORT
        or result.get("placement_agent_selected_exact_inventory_member") is not True
        or result.get("placement_agent_visual_review_completed") is not True
        or not isinstance(openai_evidence, Mapping)
        or set(openai_evidence)
        != {
            "reservation",
            "completion",
            "exclusive_lock",
            "exclusive_lock_release",
            "inference_reservations",
        }
        or not all(inference_usage.artifact_record_valid(row) for row in openai_evidence.values())
        or not inference_usage.result_projection_valid(result)
        or not Path(str(result.get("base_pose_candidate_path") or "")).is_absolute()
        or not isinstance(native_universe, Mapping)
        or not Path(str(native_universe.get("path") or "")).is_absolute()
        or native_universe_path.is_symlink()
        or not native_universe_path.is_file()
        or _sha256(native_universe_path) != native_universe.get("file_sha256")
        or _DIGEST.fullmatch(str(native_universe.get("file_sha256") or ""))
        is None
        or _DIGEST.fullmatch(str(native_universe.get("inventory_digest") or ""))
        is None
        or not 1 <= int(native_universe.get("candidate_count") or 0) <= 64
        or not Path(str(result.get("plan_path") or "")).is_absolute()
        or _DIGEST.fullmatch(str(result.get("plan_digest") or "")) is None
        or result.get("cpu_position_ik_qualified") is not True
        or result.get(
            "native_orientation_collision_contact_camera_and_execution_required"
        )
        is not True
        or result.get("provider_mutation_performed") is not False
        or result.get("paid_execution_requested") is not True
        or result.get("result_digest")
        != canonical_digest(result, digest_field="result_digest")
    ):
        raise TaskEvaluationConfiguredControlsAutostartError(
            "configured_controls_autostart_result_invalid"
        )
    return result
