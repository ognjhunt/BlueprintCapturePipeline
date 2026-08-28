"""Execute one admitted scene-configuration component inside its parent GPU run.

The Website recipe selects a capability identity, never a command.  The
production-published toolchain binds that identity to an immutable component
entrypoint.  This module verifies the exact stage input and dependency receipts,
invokes only that published entrypoint, and seals the artifacts returned to the
stage producer.  It performs no provider mutation and cannot allocate another
resource.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess  # nosec B404 - command is toolchain-manifest-bound
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .core.common import redacted_failure_text
from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_scene_configuration_builtin_producers import (
    TOOLCHAIN_ROOT_ENV,
    _secret_values,
    _validate_toolchain,
)
from .task_evaluation_scene_configuration_stage_producers import (
    ADMITTED_PRODUCER_IDENTITIES,
    PRODUCTION_RESULT_SCHEMA_VERSION,
    TaskEvaluationSceneConfigurationStageProducerError,
)


COMPONENT_RESULT_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_component_result.v1"
)
_INPUT_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_stage_production_input.v1"
)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_EXPECTED_ROLES = {
    "artifixer3d_observed_object_removal": frozenset(
        {
            "configured_appearance_without_source_object",
            "appearance_removal_receipt",
            "appearance_visual_review_receipt",
            "configured_task_thumbnail",
            "provider_render_reference_manifest",
        }
    ),
    "content_agents_rigid_replacement": frozenset(
        {
            "replacement_asset",
            "replacement_authoring_receipt",
            "replacement_graph_spec",
        }
    ),
    "simready_native_import_qualification": frozenset(
        {"native_import_runtime_result"}
    ),
}
_INPUT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"
_DEPENDENCIES_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES"
_OUTPUT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"
_RESULT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_RESULT"
_COMPONENT_RESULT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_RESULT"
_DIAGNOSTIC_ONLY_ENV = "BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_ONLY"


class TaskEvaluationSceneConfigurationStageToolError(RuntimeError):
    """One published stage component violated its immutable execution contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationStageToolError(code) from exc
    if path.is_symlink():
        raise TaskEvaluationSceneConfigurationStageToolError(code)
    return value


def _required_path(environment: Mapping[str, str], name: str) -> Path:
    unresolved = str(environment.get(name) or "").strip()
    if not unresolved:
        raise TaskEvaluationSceneConfigurationStageToolError(
            f"scene_configuration_stage_tool_environment_missing:{name}"
        )
    return Path(unresolved).expanduser().resolve()


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_dependencies(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TaskEvaluationSceneConfigurationStageToolError(
            "scene_configuration_stage_tool_dependencies_invalid"
        )
    results: list[dict[str, Any]] = []
    for row in value:
        if (
            not isinstance(row, Mapping)
            or row.get("schema_version")
            != "task_evaluation_scene_configuration_stage_result.v1"
            or row.get("status") != "completed"
            or row.get("stage_result_digest")
            != canonical_digest(row, digest_field="stage_result_digest")
        ):
            raise TaskEvaluationSceneConfigurationStageToolError(
                "scene_configuration_stage_tool_dependencies_invalid"
            )
        results.append(dict(row))
    return results


def _validate_input(
    value: Any, *, adapter_id: str, diagnostic_only: bool
) -> dict[str, Any]:
    admitted = {identity.adapter_id for identity in ADMITTED_PRODUCER_IDENTITIES}
    if not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationStageToolError(
            "scene_configuration_stage_tool_input_invalid"
        )
    stage = value.get("stage")
    adapter = stage.get("adapter") if isinstance(stage, Mapping) else None
    envelope = value.get("construction_envelope")
    source_commit = str(value.get("source_commit") or "")
    construction_source_commit = str(
        value.get("construction_source_commit") or source_commit
    )
    execution_mode = str(value.get("execution_mode") or "production")
    if (
        adapter_id not in admitted
        or value.get("schema_version") != _INPUT_SCHEMA_VERSION
        or not isinstance(stage, Mapping)
        or not isinstance(adapter, Mapping)
        or adapter.get("id") != adapter_id
        or stage.get("execution_class") != "gpu_canary"
        or value.get("run_id") != (envelope or {}).get("run_id")
        or not isinstance(value.get("configuration"), Mapping)
        or not _DIGEST.fullmatch(str(value.get("configuration_sha256") or ""))
        or not _COMMIT.fullmatch(source_commit)
        or not _COMMIT.fullmatch(construction_source_commit)
        or not _DIGEST.fullmatch(str(value.get("toolchain_digest") or ""))
        or not isinstance(envelope, Mapping)
        or envelope.get("expected_production_commit")
        != construction_source_commit
        or execution_mode not in {"production", "diagnostic_only"}
        or (execution_mode == "diagnostic_only") != diagnostic_only
        or (
            execution_mode == "production"
            and source_commit != construction_source_commit
        )
        or envelope.get("envelope_digest")
        != canonical_digest(envelope, digest_field="envelope_digest")
    ):
        raise TaskEvaluationSceneConfigurationStageToolError(
            "scene_configuration_stage_tool_input_invalid"
        )
    return dict(value)


def _validate_component_result(
    value: Any,
    *,
    adapter_id: str,
    stage_id: str,
    output_root: Path,
) -> list[dict[str, Any]]:
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != COMPONENT_RESULT_SCHEMA_VERSION
        or value.get("status") != "completed"
        or value.get("adapter_id") != adapter_id
        or value.get("stage_id") != stage_id
        or value.get("provider_mutations_performed") != 0
        or value.get("nested_paid_execution_requested") is not False
        or value.get("result_digest")
        != canonical_digest(value, digest_field="result_digest")
        or not isinstance(value.get("artifacts"), list)
    ):
        raise TaskEvaluationSceneConfigurationStageToolError(
            "scene_configuration_component_result_invalid"
        )
    artifacts: list[dict[str, Any]] = []
    roles: set[str] = set()
    for row in value["artifacts"]:
        if not isinstance(row, Mapping):
            raise TaskEvaluationSceneConfigurationStageToolError(
                "scene_configuration_component_artifact_invalid"
            )
        role = str(row.get("role") or "")
        path = Path(str(row.get("path") or "")).expanduser().resolve()
        if (
            not role
            or role in roles
            or not _under(path, output_root)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("digest")
        ):
            raise TaskEvaluationSceneConfigurationStageToolError(
                "scene_configuration_component_artifact_invalid"
            )
        roles.add(role)
        artifacts.append(dict(row))
    if roles != _EXPECTED_ROLES[adapter_id]:
        raise TaskEvaluationSceneConfigurationStageToolError(
            "scene_configuration_component_artifact_roles_invalid"
        )
    return artifacts


#: Retained tail per captured stream. Enough to carry a traceback and the
#: lines above it; small enough that a component looping on output cannot
#: bury the stage producer log that has to survive the run.
_COMPONENT_FAILURE_STREAM_TAIL_BYTES = 20_000


def _failure_stream_tail(value: Any, *, secrets: Sequence[str] = ()) -> str:
    """Redact one captured stream and keep its tail, saying what was dropped."""

    text = "" if value is None else str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "REDACTED_SECRET")
    text = redacted_failure_text(text)
    if len(text) <= _COMPONENT_FAILURE_STREAM_TAIL_BYTES:
        return text
    dropped = len(text) - _COMPONENT_FAILURE_STREAM_TAIL_BYTES
    tail = text[-_COMPONENT_FAILURE_STREAM_TAIL_BYTES:]
    return f"<{dropped} earlier bytes dropped>\n{tail}"


def _emit_component_failure_diagnostics(
    *,
    adapter_id: str,
    completed: Any,
    component_result_written: bool,
    secret_values: Sequence[str] = (),
) -> None:
    """Print the component's own redacted output before refusing.

    ``capture_output=True`` pulls the component's stdout and stderr into this
    process, and the refusal carried only its exit code. So a stage that died
    on a rented GPU left ``scene_configuration_component_failed:<id>:1`` and
    nothing else: the component's traceback existed, was captured, and was
    dropped on the floor. The only way to learn why was to rent the GPU and
    run the whole parent allocation again.

    The stage producer retains this process's stderr, so writing the redacted
    tail here is what makes a paid failure diagnosable from the run that paid
    for it. Redaction is the same one every other failure path uses, and the
    exception message is unchanged so no contract keyed on it moves.
    """

    lines = [
        f"scene_configuration_component_failed:{adapter_id}",
        f"returncode={getattr(completed, 'returncode', None)}",
        f"component_result_written={component_result_written}",
    ]
    for name in ("stdout", "stderr"):
        tail = _failure_stream_tail(
            getattr(completed, name, None), secrets=secret_values
        )
        lines.append(f"--- component {name} ---")
        lines.append(tail if tail.strip() else "<empty>")
    print("\n".join(lines), file=sys.stderr, flush=True)


def execute_stage_tool(
    *,
    adapter_id: str,
    environment: Mapping[str, str] | None = None,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Execute the manifest-bound component and seal its exact output bytes."""

    values = dict(os.environ if environment is None else environment)
    output_root = _required_path(values, _OUTPUT_ENV)
    input_path = _required_path(values, _INPUT_ENV)
    dependencies_path = _required_path(values, _DEPENDENCIES_ENV)
    result_path = _required_path(values, _RESULT_ENV)
    if (
        output_root.is_symlink()
        or not output_root.is_dir()
        or not all(
            _under(path, output_root)
            for path in (input_path, dependencies_path, result_path)
        )
        or result_path.exists()
    ):
        raise TaskEvaluationSceneConfigurationStageToolError(
            "scene_configuration_stage_tool_path_invalid"
        )
    production_input = _validate_input(
        _read(input_path, code="scene_configuration_stage_tool_input_invalid"),
        adapter_id=adapter_id,
        diagnostic_only=values.get(_DIAGNOSTIC_ONLY_ENV) == "1",
    )
    _validate_dependencies(
        _read(
            dependencies_path,
            code="scene_configuration_stage_tool_dependencies_invalid",
        )
    )
    toolchain_root = _required_path(values, TOOLCHAIN_ROOT_ENV)
    manifest, _entrypoints = _validate_toolchain(
        root=toolchain_root,
        expected_source_commit=str(production_input["source_commit"]),
    )
    stage_binding = manifest["stages"].get(adapter_id)
    component_relative = (
        stage_binding.get("component_entrypoint")
        if isinstance(stage_binding, Mapping)
        else None
    )
    component = (toolchain_root / str(component_relative or "")).resolve()
    if (
        not component_relative
        or not _under(component, toolchain_root)
        or component.is_symlink()
        or not component.is_file()
        or not component.stat().st_mode & 0o111
    ):
        raise TaskEvaluationSceneConfigurationStageToolError(
            "scene_configuration_component_entrypoint_invalid"
        )
    component_result_path = output_root / COMPONENT_RESULT_SCHEMA_VERSION
    component_environment = {
        **values,
        _COMPONENT_RESULT_ENV: str(component_result_path),
    }
    try:
        secret_values = tuple(_secret_values(component_environment))
    except (
        OSError,
        UnicodeError,
        ValueError,
        TaskEvaluationSceneConfigurationStageProducerError,
    ) as exc:
        raise TaskEvaluationSceneConfigurationStageToolError(
            "scene_configuration_stage_tool_secret_file_invalid"
        ) from exc
    completed = runner(
        [str(component)],
        cwd=toolchain_root,
        env=component_environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=7_200,
    )
    if completed.returncode != 0 or not component_result_path.is_file():
        _emit_component_failure_diagnostics(
            adapter_id=adapter_id,
            completed=completed,
            component_result_written=component_result_path.is_file(),
            secret_values=secret_values,
        )
        raise TaskEvaluationSceneConfigurationStageToolError(
            f"scene_configuration_component_failed:{adapter_id}:"
            f"{completed.returncode}"
        )
    artifacts = _validate_component_result(
        _read(
            component_result_path,
            code="scene_configuration_component_result_invalid",
        ),
        adapter_id=adapter_id,
        stage_id=str(production_input["stage"]["stage_id"]),
        output_root=output_root,
    )
    result: dict[str, Any] = {
        "schema_version": PRODUCTION_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "stage_id": production_input["stage"]["stage_id"],
        "capability": production_input["stage"]["capability"],
        "adapter_id": adapter_id,
        "source_commit": production_input["source_commit"],
        "toolchain_digest": production_input["toolchain_digest"],
        "artifacts": artifacts,
        "provider_mutations_performed": 0,
        "paid_execution_requested": False,
        "executed_inside_parent_configuration_run": True,
        "production_result_digest": "",
    }
    result["production_result_digest"] = canonical_digest(
        result, digest_field="production_result_digest"
    )
    result_path.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-id", required=True)
    args = parser.parse_args(argv)
    result = execute_stage_tool(adapter_id=args.adapter_id)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COMPONENT_RESULT_SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationStageToolError",
    "execute_stage_tool",
    "main",
]
