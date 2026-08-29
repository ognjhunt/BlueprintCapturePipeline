"""Recover a completed diagnostic Stage 1 after a control-plane adapter fix.

The paid producer may complete and upload immutable artifacts before the
control-plane adapter rejects their handoff.  This module reopens those exact
bytes, reruns only the repository-owned no-spend adapter, and advances the
existing diagnostic checkpoint.  It never invokes a provider, publishes an
offering, or upgrades the result to qualifying evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_scene_configuration_adapters import (
    SceneConfigurationAdapterRegistry,
)
from .task_evaluation_scene_configuration_builtin_adapters import (
    builtin_scene_configuration_diagnostic_adapter_handlers,
)
from .task_evaluation_scene_configuration_diagnostic_checkpoint import (
    SCHEMA_VERSION as CHECKPOINT_SCHEMA_VERSION,
    advance_scene_configuration_diagnostic_checkpoint,
    diagnostic_checkpoint_scientific_binding_digest,
    validate_scene_configuration_diagnostic_checkpoint,
)
from .task_evaluation_scene_configuration_orchestrator import (
    STAGE_RESULT_SCHEMA_VERSION,
)
from .task_evaluation_scene_configuration_stage_producers import (
    PRODUCTION_RESULT_SCHEMA_VERSION,
)


REFERENCE_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_advanced_checkpoint_reference.v1"
)
REFERENCE_STATUS = "validated_diagnostic_checkpoint_ready_for_next_retry"
_STAGE_INPUT_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_stage_production_input.v1"
)
_EXPECTED_STAGE_ID = "stage-1"
_EXPECTED_CAPABILITY = "observed_appearance_object_removal"
_EXPECTED_ADAPTER_ID = "artifixer3d_observed_object_removal"
_EXPECTED_ARTIFACT_ROLES = frozenset(
    {
        "configured_appearance_without_source_object",
        "appearance_removal_receipt",
        "appearance_visual_review_receipt",
        "configured_task_thumbnail",
        "provider_render_reference_manifest",
    }
)


class TaskEvaluationSceneConfigurationDiagnosticRecoveryError(RuntimeError):
    """Retained provider bytes could not safely advance the checkpoint."""


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationDiagnosticRecoveryError(code) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationDiagnosticRecoveryError(code)
    return dict(value)


def _validated_stage_input(
    *, stage_input_path: Path, configuration_path: Path
) -> dict[str, Any]:
    value = _read(
        stage_input_path,
        code="scene_configuration_diagnostic_stage_one_input_invalid",
    )
    stage = value.get("stage")
    configuration = value.get("configuration")
    envelope = value.get("construction_envelope")
    stages = (
        (envelope.get("recipe") or {}).get("stage_sequence")
        if isinstance(envelope, Mapping)
        else None
    )
    adapter = stage.get("adapter") if isinstance(stage, Mapping) else None
    try:
        materialized_configuration = _read(
            configuration_path,
            code="scene_configuration_diagnostic_stage_one_configuration_invalid",
        )
    except TaskEvaluationSceneConfigurationDiagnosticRecoveryError:
        raise
    if (
        value.get("schema_version") != _STAGE_INPUT_SCHEMA_VERSION
        or value.get("execution_mode") != "diagnostic_only"
        or not isinstance(stage, Mapping)
        or stage.get("stage_id") != _EXPECTED_STAGE_ID
        or stage.get("capability") != _EXPECTED_CAPABILITY
        or stage.get("execution_class") != "gpu_canary"
        or stage.get("depends_on") != []
        or not isinstance(adapter, Mapping)
        or adapter.get("id") != _EXPECTED_ADAPTER_ID
        or adapter.get("version") != "v1"
        or not isinstance(configuration, Mapping)
        or materialized_configuration != dict(configuration)
        or configuration_path.is_symlink()
        or _sha256_and_size(configuration_path)[0]
        != value.get("configuration_sha256")
        or not isinstance(envelope, Mapping)
        or envelope.get("envelope_digest")
        != canonical_digest(envelope, digest_field="envelope_digest")
        or not isinstance(stages, list)
        or len(stages) != 6
        or stages[0] != stage
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticRecoveryError(
            "scene_configuration_diagnostic_stage_one_input_invalid"
        )
    return value


def _rehydrated_stage_input(
    *, stage_input: Mapping[str, Any], configuration_path: Path
) -> dict[str, Any]:
    """Replace only the provider-local Stage-1 config path with verified host bytes."""

    value = json.loads(json.dumps(dict(stage_input)))
    envelope = value["construction_envelope"]
    rows = envelope.get("stage_configuration_references")
    if not isinstance(rows, list) or len(rows) != 6:
        raise TaskEvaluationSceneConfigurationDiagnosticRecoveryError(
            "scene_configuration_diagnostic_stage_one_configuration_invalid"
        )
    first = rows[0]
    digest, size = _sha256_and_size(configuration_path)
    if (
        not isinstance(first, dict)
        or first.get("stage_id") != _EXPECTED_STAGE_ID
        or first.get("digest") != digest
        or first.get("size_bytes") != size
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticRecoveryError(
            "scene_configuration_diagnostic_stage_one_configuration_invalid"
        )
    first["materialized_path"] = str(configuration_path)
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    return value


def _portable_provider_artifacts(
    *, producer_root: Path, production: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    rows = production.get("artifacts")
    if not isinstance(rows, list) or {
        str(row.get("role") or "")
        for row in rows
        if isinstance(row, Mapping)
    } != _EXPECTED_ARTIFACT_ROLES:
        raise TaskEvaluationSceneConfigurationDiagnosticRecoveryError(
            "scene_configuration_diagnostic_stage_one_artifacts_invalid"
        )
    resolved: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TaskEvaluationSceneConfigurationDiagnosticRecoveryError(
                "scene_configuration_diagnostic_stage_one_artifacts_invalid"
            )
        provider_path = PurePosixPath(str(row.get("path") or ""))
        expected_parent = PurePosixPath("stages/stage-1/producer")
        candidate = (producer_root / provider_path.name).resolve()
        try:
            candidate.relative_to(producer_root)
        except ValueError as exc:
            raise TaskEvaluationSceneConfigurationDiagnosticRecoveryError(
                "scene_configuration_diagnostic_stage_one_artifacts_invalid"
            ) from exc
        if (
            not provider_path.is_absolute()
            or PurePosixPath(*provider_path.parent.parts[-3:]) != expected_parent
            or candidate.is_symlink()
            or not candidate.is_file()
            or _sha256_and_size(candidate)
            != (row.get("digest"), row.get("size_bytes"))
        ):
            raise TaskEvaluationSceneConfigurationDiagnosticRecoveryError(
                "scene_configuration_diagnostic_stage_one_artifacts_invalid"
            )
        resolved.append({**dict(row), "path": str(candidate)})
    return tuple(resolved)


def _validated_production_result(
    *, path: Path, stage_input: Mapping[str, Any]
) -> dict[str, Any]:
    value = _read(
        path,
        code="scene_configuration_diagnostic_stage_one_production_result_invalid",
    )
    stage = stage_input["stage"]
    if (
        value.get("schema_version") != PRODUCTION_RESULT_SCHEMA_VERSION
        or value.get("status") != "completed"
        or value.get("stage_id") != stage.get("stage_id")
        or value.get("capability") != stage.get("capability")
        or value.get("adapter_id") != _EXPECTED_ADAPTER_ID
        or value.get("source_commit") != stage_input.get("source_commit")
        or value.get("toolchain_digest") != stage_input.get("toolchain_digest")
        or value.get("provider_mutations_performed") != 0
        or value.get("paid_execution_requested") is not False
        or value.get("executed_inside_parent_configuration_run") is not True
        or value.get("production_result_digest")
        != canonical_digest(value, digest_field="production_result_digest")
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticRecoveryError(
            "scene_configuration_diagnostic_stage_one_production_result_invalid"
        )
    return value


def _diagnostic_result(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.update(
        {
            "diagnostic_only": True,
            "qualification_eligible": False,
            "executed_inside_one_parent_provider_run": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
        }
    )
    result["stage_result_digest"] = canonical_digest(
        result, digest_field="stage_result_digest"
    )
    if (
        result.get("schema_version") != STAGE_RESULT_SCHEMA_VERSION
        or result.get("status") != "completed"
        or result.get("provider_mutations_performed") != 0
        or result.get("paid_execution_requested") is not False
        or result.get("executed_inside_parent_configuration_run") is not True
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticRecoveryError(
            "scene_configuration_diagnostic_stage_one_adapter_result_invalid"
        )
    return result


def _checkpoint_reference(
    *, root: Path, checkpoint: Mapping[str, Any], production_result_digest: str
) -> dict[str, Any]:
    manifest = root / f"{CHECKPOINT_SCHEMA_VERSION}.json"
    files = [path for path in root.rglob("*") if path.is_file()]
    value: dict[str, Any] = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "status": REFERENCE_STATUS,
        "checkpoint_root": str(root),
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256_and_size(manifest)[0],
        "checkpoint_digest": checkpoint["checkpoint_digest"],
        "completed_stage_prefix_count": checkpoint[
            "completed_stage_prefix_count"
        ],
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "source_provider_result_digest": production_result_digest,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "reference_digest": "",
    }
    value["reference_digest"] = canonical_digest(
        value, digest_field="reference_digest"
    )
    return value


def recover_scene_configuration_diagnostic_stage_one_checkpoint(
    *,
    source_checkpoint_root: str | Path,
    stage_production_input_path: str | Path,
    stage_production_result_path: str | Path,
    stage_configuration_path: str | Path,
    producer_root: str | Path,
    adapter_output_root: str | Path,
    checkpoint_output_root: str | Path,
    reference_output_path: str | Path,
) -> dict[str, Any]:
    """Replay the fixed Stage-1 adapter and seal one completed-stage checkpoint."""

    source_root = Path(source_checkpoint_root).expanduser().resolve()
    stage_input_path = Path(stage_production_input_path).expanduser().resolve()
    production_path = Path(stage_production_result_path).expanduser().resolve()
    configuration_path = Path(stage_configuration_path).expanduser().resolve()
    producer = Path(producer_root).expanduser().resolve()
    adapter_output = Path(adapter_output_root).expanduser().resolve()
    checkpoint_output = Path(checkpoint_output_root).expanduser().resolve()
    reference_output = Path(reference_output_path).expanduser().resolve()
    if (
        producer.is_symlink()
        or not producer.is_dir()
        or adapter_output.exists()
        or adapter_output.is_symlink()
        or checkpoint_output.exists()
        or checkpoint_output.is_symlink()
        or reference_output.exists()
        or reference_output.is_symlink()
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticRecoveryError(
            "scene_configuration_diagnostic_stage_one_recovery_output_invalid"
        )
    stage_input = _validated_stage_input(
        stage_input_path=stage_input_path,
        configuration_path=configuration_path,
    )
    production = _validated_production_result(
        path=production_path, stage_input=stage_input
    )
    stage_input = _rehydrated_stage_input(
        stage_input=stage_input, configuration_path=configuration_path
    )
    envelope = stage_input["construction_envelope"]
    expected_binding = diagnostic_checkpoint_scientific_binding_digest(
        stage_input=stage_input,
        render_inputs=envelope["render_inputs_result"],
    )
    source_checkpoint = validate_scene_configuration_diagnostic_checkpoint(
        checkpoint_root=source_root,
        expected_scientific_binding_digest=expected_binding,
    )
    if source_checkpoint.get("completed_stage_prefix_count") != 0:
        raise TaskEvaluationSceneConfigurationDiagnosticRecoveryError(
            "scene_configuration_diagnostic_stage_one_source_checkpoint_invalid"
        )
    artifacts = _portable_provider_artifacts(
        producer_root=producer,
        production=production,
    )
    adapter_output.mkdir(parents=True, mode=0o750)
    try:
        registry = SceneConfigurationAdapterRegistry(
            builtin_scene_configuration_diagnostic_adapter_handlers()
        )
        stage_result = _diagnostic_result(
            registry.execute(
                stage=stage_input["stage"],
                envelope=envelope,
                configuration=stage_input["configuration"],
                configuration_path=configuration_path,
                dependency_results=(),
                output_root=adapter_output,
                provider_runtime_artifacts=artifacts,
            )
        )
        stages = envelope["recipe"]["stage_sequence"]
        checkpoint = advance_scene_configuration_diagnostic_checkpoint(
            checkpoint_root=source_root,
            stage_results=[stage_result],
            stage_sequence=stages,
            configurations={
                _EXPECTED_STAGE_ID: (stage_input["configuration"], configuration_path)
            },
            output_root=checkpoint_output,
        )
        checkpoint = validate_scene_configuration_diagnostic_checkpoint(
            checkpoint_root=checkpoint_output,
            expected_scientific_binding_digest=expected_binding,
        )
        if checkpoint.get("completed_stage_prefix_count") != 1:
            raise TaskEvaluationSceneConfigurationDiagnosticRecoveryError(
                "scene_configuration_diagnostic_stage_one_checkpoint_invalid"
            )
        reference = _checkpoint_reference(
            root=checkpoint_output,
            checkpoint=checkpoint,
            production_result_digest=str(production["production_result_digest"]),
        )
        reference_output.parent.mkdir(parents=True, mode=0o750, exist_ok=True)
        reference_output.write_text(canonical_json(reference) + "\n", encoding="utf-8")
        return reference
    except Exception:
        shutil.rmtree(adapter_output, ignore_errors=True)
        shutil.rmtree(checkpoint_output, ignore_errors=True)
        reference_output.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint-root", required=True)
    parser.add_argument("--stage-production-input", required=True)
    parser.add_argument("--stage-production-result", required=True)
    parser.add_argument("--stage-configuration", required=True)
    parser.add_argument("--producer-root", required=True)
    parser.add_argument("--adapter-output-root", required=True)
    parser.add_argument("--checkpoint-output-root", required=True)
    parser.add_argument("--reference-output", required=True)
    args = parser.parse_args(argv)
    result = recover_scene_configuration_diagnostic_stage_one_checkpoint(
        source_checkpoint_root=args.source_checkpoint_root,
        stage_production_input_path=args.stage_production_input,
        stage_production_result_path=args.stage_production_result,
        stage_configuration_path=args.stage_configuration,
        producer_root=args.producer_root,
        adapter_output_root=args.adapter_output_root,
        checkpoint_output_root=args.checkpoint_output_root,
        reference_output_path=args.reference_output,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
