"""Production state machine for one scene-configuration Task Evaluation Run."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_configured_scene_revision import (
    TaskEvaluationConfiguredSceneRevisionError,
    validate_configured_scene_revision,
)
from .task_evaluation_launch_preparation_queue import (
    write_launch_preparation_record_exclusive,
)
from .task_evaluation_scene_construction_queue import (
    ENVELOPE_SCHEMA_VERSION,
    ensure_scene_construction_queue_root,
)
from .task_evaluation_scene_construction_recipe import (
    TaskEvaluationSceneConstructionRecipeError,
    validate_scene_construction_recipe,
)


RESULT_SCHEMA_VERSION = "task_evaluation_scene_configuration_result.v1"
STAGE_RESULT_SCHEMA_VERSION = "task_evaluation_scene_configuration_stage_result.v1"
CANONICAL_ALLOCATOR = "python -m blueprint_pipeline.paid_resource_allocator gpu-canary"
ConfigurationRunExecutor = Callable[..., Mapping[str, Any]]
PROVIDER_EXECUTION_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_provider_execution.v1"
)


class TaskEvaluationSceneConfigurationOrchestratorError(RuntimeError):
    """A scene-configuration run could not advance without weakening evidence."""


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _load_envelope(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationOrchestratorError(
            "scene_configuration_envelope_invalid"
        ) from exc
    if (
        path.is_symlink()
        or not isinstance(value, Mapping)
        or value.get("schema_version") != ENVELOPE_SCHEMA_VERSION
        or value.get("envelope_digest")
        != canonical_digest(value, digest_field="envelope_digest")
    ):
        raise TaskEvaluationSceneConfigurationOrchestratorError(
            "scene_configuration_envelope_invalid"
        )
    try:
        recipe = validate_scene_construction_recipe(value["recipe"])
    except (KeyError, TaskEvaluationSceneConstructionRecipeError) as exc:
        raise TaskEvaluationSceneConfigurationOrchestratorError(
            "scene_configuration_recipe_invalid"
        ) from exc
    if recipe["recipe_digest"] != value.get("recipe_digest"):
        raise TaskEvaluationSceneConfigurationOrchestratorError(
            "scene_configuration_recipe_binding_mismatch"
        )
    return dict(value)


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _verified_stage_configurations(
    *, envelope: Mapping[str, Any], input_root: Path
) -> dict[str, tuple[dict[str, Any], Path]]:
    rows = envelope.get("stage_configuration_references")
    recipe = envelope["recipe"]
    if not isinstance(rows, list) or len(rows) != len(recipe["stage_sequence"]):
        raise TaskEvaluationSceneConfigurationOrchestratorError(
            "scene_configuration_stage_configuration_set_invalid"
        )
    by_contract_path: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TaskEvaluationSceneConfigurationOrchestratorError(
                "scene_configuration_stage_configuration_set_invalid"
            )
        contract_path = str(row.get("contract_path") or "")
        if contract_path in by_contract_path:
            raise TaskEvaluationSceneConfigurationOrchestratorError(
                "scene_configuration_stage_configuration_set_invalid"
            )
        by_contract_path[contract_path] = row
    verified: dict[str, tuple[dict[str, Any], Path]] = {}
    for index, stage in enumerate(recipe["stage_sequence"]):
        contract_path = (
            f"construction.recipe.stage_sequence.{index}.configuration"
        )
        row = by_contract_path.get(contract_path)
        path = Path(str((row or {}).get("materialized_path") or "")).resolve()
        expected = stage["configuration"]
        if (
            row is None
            or row.get("uri") != expected["uri"]
            or row.get("digest") != expected["digest"]
            or row.get("size_bytes") != expected["size_bytes"]
            or row.get("full_byte_service_account_readback_passed") is not True
            or not _under(path, input_root)
            or path.is_symlink()
            or not path.is_file()
            or _sha256_and_size(path)
            != (expected["digest"], expected["size_bytes"])
        ):
            raise TaskEvaluationSceneConfigurationOrchestratorError(
                "scene_configuration_stage_configuration_readback_invalid"
            )
        try:
            configuration = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskEvaluationSceneConfigurationOrchestratorError(
                "scene_configuration_stage_configuration_json_invalid"
            ) from exc
        if not isinstance(configuration, Mapping):
            raise TaskEvaluationSceneConfigurationOrchestratorError(
                "scene_configuration_stage_configuration_json_invalid"
            )
        verified[stage["stage_id"]] = (dict(configuration), path)
    return verified


def _validated_stage_result(
    *,
    value: Mapping[str, Any],
    stage: Mapping[str, Any],
    configuration_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    result = dict(value)
    if (
        result.get("schema_version") != STAGE_RESULT_SCHEMA_VERSION
        or result.get("status") != "completed"
        or result.get("stage_id") != stage["stage_id"]
        or result.get("capability") != stage["capability"]
        or result.get("execution_class") != stage["execution_class"]
        or result.get("configuration_digest")
        != "sha256:" + hashlib.sha256(configuration_path.read_bytes()).hexdigest()
        or result.get("retry_cap") != 0
        or result.get("stage_result_digest")
        != canonical_digest(result, digest_field="stage_result_digest")
        or result.get("raw_secret_values_recorded") is not False
    ):
        raise TaskEvaluationSceneConfigurationOrchestratorError(
            f"scene_configuration_stage_result_invalid:{stage['stage_id']}"
        )
    if (
        result.get("canonical_allocator") is not None
        or result.get("provider_mutations_performed") != 0
        or result.get("paid_execution_requested") is not False
        or result.get("executed_inside_parent_configuration_run") is not True
    ):
        raise TaskEvaluationSceneConfigurationOrchestratorError(
            f"scene_configuration_stage_governance_invalid:{stage['stage_id']}"
        )
    artifacts = result.get("output_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise TaskEvaluationSceneConfigurationOrchestratorError(
            f"scene_configuration_stage_outputs_missing:{stage['stage_id']}"
        )
    roles: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise TaskEvaluationSceneConfigurationOrchestratorError(
                f"scene_configuration_stage_output_invalid:{stage['stage_id']}"
            )
        role = str(artifact.get("role") or "")
        path = Path(str(artifact.get("path") or "")).resolve()
        if (
            not role
            or role in roles
            or not _under(path, output_root)
            or path.is_symlink()
            or not path.is_file()
            or _sha256_and_size(path)
            != (artifact.get("digest"), artifact.get("size_bytes"))
        ):
            raise TaskEvaluationSceneConfigurationOrchestratorError(
                f"scene_configuration_stage_output_invalid:{stage['stage_id']}"
            )
        roles.add(role)
    return result


def _validated_parent_execution(
    value: Mapping[str, Any], *, output_root: Path
) -> dict[str, Any]:
    execution = dict(value)
    revision_artifact = execution.get("configured_scene_revision")
    revision_path = Path(
        str((revision_artifact or {}).get("path") or "")
    ).resolve()
    if (
        execution.get("schema_version") != PROVIDER_EXECUTION_SCHEMA_VERSION
        or execution.get("status") != "completed"
        or execution.get("canonical_allocator") != CANONICAL_ALLOCATOR
        or execution.get("provider_mutations_performed") != 1
        or execution.get("paid_execution_requested") is not True
        or execution.get("retry_cap") != 0
        or execution.get("evaluation_episode_executed") is not False
        or execution.get("raw_secret_values_recorded") is not False
        or execution.get("execution_digest")
        != canonical_digest(execution, digest_field="execution_digest")
        or not isinstance(execution.get("stage_results"), list)
        or len(execution["stage_results"]) != 6
        or not isinstance(revision_artifact, Mapping)
        or revision_artifact.get("role") != "configured_scene_revision"
        or not _under(revision_path, output_root)
        or revision_path.is_symlink()
        or not revision_path.is_file()
        or _sha256_and_size(revision_path)
        != (
            revision_artifact.get("digest"),
            revision_artifact.get("size_bytes"),
        )
    ):
        raise TaskEvaluationSceneConfigurationOrchestratorError(
            "scene_configuration_parent_execution_invalid"
        )
    for field in (
        "paid_authority_digest",
        "billing_reconciliation_digest",
        "teardown_digest",
        "provider_zero_digest",
        "launch_receipt_digest",
    ):
        if not re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(execution.get(field) or "")
        ):
            raise TaskEvaluationSceneConfigurationOrchestratorError(
                f"scene_configuration_parent_execution_governance_invalid:{field}"
            )
    return execution


def process_scene_configuration_queue(
    *,
    queue_root: str | Path,
    input_root: str | Path,
    output_root: str | Path,
    source_commit: str,
    configuration_run_executor: ConfigurationRunExecutor,
    max_messages: int = 1,
) -> dict[str, Any]:
    """Execute every ordered configuration stage for one claimed website run."""

    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise TaskEvaluationSceneConfigurationOrchestratorError(
            "scene_configuration_source_commit_unproven"
        )
    if not isinstance(max_messages, int) or isinstance(max_messages, bool) or not 1 <= max_messages <= 4:
        raise TaskEvaluationSceneConfigurationOrchestratorError(
            "scene_configuration_max_messages_invalid"
        )
    queue = ensure_scene_construction_queue_root(queue_root)
    inputs = Path(input_root).resolve(strict=True)
    outputs = Path(output_root)
    outputs.mkdir(parents=True, exist_ok=True, mode=0o750)
    outputs = outputs.resolve(strict=True)
    results_root = queue / "results"
    results_root.mkdir(mode=0o750, exist_ok=True)
    processed: list[dict[str, Any]] = []
    for source in sorted((queue / "pending").glob("*.json"))[:max_messages]:
        claimed = queue / "processing" / source.name
        try:
            descriptor = os.open(
                claimed, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError:
            continue
        else:
            os.close(descriptor)
        try:
            os.replace(source, claimed)
        except FileNotFoundError:
            claimed.unlink(missing_ok=True)
            continue
        terminal_state = "completed"
        try:
            envelope = _load_envelope(claimed)
            if envelope.get("expected_production_commit") != source_commit:
                raise TaskEvaluationSceneConfigurationOrchestratorError(
                    "scene_configuration_source_commit_mismatch"
                )
            configurations = _verified_stage_configurations(
                envelope=envelope, input_root=inputs
            )
            owned_output = outputs / envelope["orchestration_id"]
            owned_output.mkdir(mode=0o750, exist_ok=False)
            parent_execution = _validated_parent_execution(
                configuration_run_executor(
                    envelope=envelope,
                    configurations=configurations,
                    output_root=owned_output,
                ),
                output_root=owned_output,
            )
            stage_results: list[dict[str, Any]] = []
            for stage, candidate in zip(
                envelope["recipe"]["stage_sequence"],
                parent_execution["stage_results"],
                strict=True,
            ):
                configuration, configuration_path = configurations[stage["stage_id"]]
                stage_output = owned_output / stage["stage_id"]
                stage_result = _validated_stage_result(
                    value=candidate,
                    stage=stage,
                    configuration_path=configuration_path,
                    output_root=stage_output,
                )
                stage_results.append(stage_result)
            revision_path = Path(
                parent_execution["configured_scene_revision"]["path"]
            ).resolve()
            try:
                revision = validate_configured_scene_revision(
                    json.loads(revision_path.read_text(encoding="utf-8"))
                )
            except (
                OSError,
                json.JSONDecodeError,
                TaskEvaluationConfiguredSceneRevisionError,
            ) as exc:
                raise TaskEvaluationSceneConfigurationOrchestratorError(
                    "scene_configuration_revision_invalid"
                ) from exc
            if (
                revision["configuration_run_id"] != envelope["run_id"]
                or revision["team_namespace"] != envelope["team_namespace"]
                or revision["scene_identity"]
                != envelope["recipe"]["scene_identity"]
                or revision["source_commit"] != source_commit
            ):
                raise TaskEvaluationSceneConfigurationOrchestratorError(
                    "scene_configuration_revision_binding_mismatch"
                )
            result: dict[str, Any] = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "configured",
                "orchestration_id": envelope["orchestration_id"],
                "run_id": envelope["run_id"],
                "team_namespace": envelope["team_namespace"],
                "source_commit": source_commit,
                "recipe_digest": envelope["recipe_digest"],
                "stage_result_digests": [
                    row["stage_result_digest"] for row in stage_results
                ],
                "configured_scene_revision_digest": revision["revision_digest"],
                "configured_scene_revision_file_digest": _sha256_and_size(
                    revision_path
                )[0],
                "stage_count": len(stage_results),
                "automatic_progression_performed": True,
                "evaluation_episode_executed": False,
                "retry_cap": 0,
                "parent_launch_receipt_digest": parent_execution[
                    "launch_receipt_digest"
                ],
                "billing_reconciliation_digest": parent_execution[
                    "billing_reconciliation_digest"
                ],
                "teardown_digest": parent_execution["teardown_digest"],
                "provider_zero_digest": parent_execution["provider_zero_digest"],
                "blockers": [],
                "result_digest": "",
            }
        except Exception as exc:
            terminal_state = "blocked"
            result = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "blocked",
                "orchestration_id": re.sub(r"-[0-9a-f]{64}\.json$", "", source.name),
                "source_commit": source_commit,
                "blockers": [
                    str(exc)
                    if isinstance(
                        exc,
                        TaskEvaluationSceneConfigurationOrchestratorError,
                    )
                    else f"scene_configuration_orchestrator_failed:{type(exc).__name__}"
                ],
                "automatic_retry_performed": False,
                "result_digest": "",
            }
        result["result_digest"] = canonical_digest(
            result, digest_field="result_digest"
        )
        try:
            write_launch_preparation_record_exclusive(
                results_root / source.name, result
            )
        except FileExistsError:
            terminal_state = "blocked"
            result = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "blocked",
                "orchestration_id": result.get("orchestration_id"),
                "source_commit": source_commit,
                "blockers": ["scene_configuration_immutable_result_conflict"],
                "automatic_retry_performed": False,
                "result_digest": "",
            }
            result["result_digest"] = canonical_digest(
                result, digest_field="result_digest"
            )
        os.replace(claimed, queue / terminal_state / source.name)
        processed.append(result)
    return {
        "schema_version": "task_evaluation_scene_configuration_queue_run.v1",
        "status": "processed" if processed else "idle",
        "processed_count": len(processed),
        "results": processed,
        "automatic_retry_performed": False,
    }


__all__ = [
    "CANONICAL_ALLOCATOR",
    "PROVIDER_EXECUTION_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "STAGE_RESULT_SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationOrchestratorError",
    "process_scene_configuration_queue",
]
