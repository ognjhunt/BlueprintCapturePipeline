"""Execute the three admitted GPU stage tools from one sealed toolchain.

The scene recipe cannot name commands.  A production-published toolchain binds
one executable to each admitted adapter identity, and this module invokes those
fixed bytes with file-path environment variables.  Secret values are read only
for log redaction and never placed in argv, JSON inputs, receipts, or outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess  # nosec B404 - executable is full-byte toolchain-bound
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .core.common import redacted_failure_text
from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_scene_configuration_component_package import (
    TaskEvaluationSceneConfigurationComponentPackageError,
    validate_scene_configuration_component_package,
)
from .task_evaluation_scene_configuration_runtime_budget import (
    GPU_STAGE_TIMEOUT_SECONDS,
)
from .task_evaluation_scene_configuration_stage_producers import (
    ADMITTED_PRODUCER_IDENTITIES,
    PRODUCTION_RESULT_SCHEMA_VERSION,
    SceneConfigurationStageProducerIdentity,
    SceneConfigurationStageProducerRegistry,
    TaskEvaluationSceneConfigurationStageProducerError,
)


TOOLCHAIN_SCHEMA_VERSION = "task_evaluation_scene_configuration_toolchain.v1"
TOOLCHAIN_ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_SCENE_CONFIGURATION_TOOLCHAIN_ROOT"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
# Compatibility alias retained for focused tests and existing callers.  The
# canonical mapping lives with the serialized parent-runtime contract so the
# parent budget cannot drift from the child allowances it must cover.
_STAGE_TIMEOUTS_SECONDS = GPU_STAGE_TIMEOUT_SECONDS
_EXPECTED_ARTIFACT_ROLES = {
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
_DIAGNOSTIC_REJECTED_ARTIFACT_ROLES = frozenset(
    {
        "diagnostic_rejected_appearance_candidate",
        "appearance_rejection_receipt",
        "appearance_visual_review_execution",
        "provider_render_reference_manifest",
    }
)
_SUPPLEMENTAL_DESTINATION_ARTIFACT_ROLES = {
    "simready_native_import_qualification": frozenset(
        {"destination_native_import_runtime_result"}
    ),
}


def expected_stage_artifact_roles(
    adapter_id: str, *, supplemental_destination: bool
) -> frozenset[str]:
    """Exact artifact roles one GPU stage component must emit.

    A recipe that binds a supplemental passive destination makes the native
    import component emit the destination's runtime result too; no other
    component grows an extra role, and an undeclared destination is refused.
    """

    roles = _EXPECTED_ARTIFACT_ROLES[adapter_id]
    if supplemental_destination:
        roles = roles | _SUPPLEMENTAL_DESTINATION_ARTIFACT_ROLES.get(
            adapter_id, frozenset()
        )
    return roles


def envelope_declares_supplemental_destination(envelope: Any) -> bool:
    recipe = envelope.get("recipe") if isinstance(envelope, Mapping) else None
    return isinstance(recipe, Mapping) and isinstance(
        recipe.get("supplemental_destination"), Mapping
    )


_SECRET_ENVIRONMENT_FILES = (
    "OPENAI_API_KEY_FILE",
    "OPENAI_ADMIN_API_KEY_FILE",
    "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE",
    "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE",
    "OPENAI_CONTENT_AGENTS_API_KEY_FILE",
    "BLUEPRINT_OPENAI_COST_SCOPE_ATTESTATION_FILE",
)
_RAW_SECRET_ENVIRONMENT_NAMES = (
    "OPENAI_API_KEY",
    "BLUEPRINT_OPENAI_ADMIN_KEY",
)
Runner = Callable[..., subprocess.CompletedProcess[str]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationStageProducerError(code) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationStageProducerError(code)
    return dict(value)


def _relative_file(root: Path, value: Any, *, code: str) -> Path:
    relative = str(value or "")
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise TaskEvaluationSceneConfigurationStageProducerError(code)
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TaskEvaluationSceneConfigurationStageProducerError(code) from exc
    if path.is_symlink() or not path.is_file():
        raise TaskEvaluationSceneConfigurationStageProducerError(code)
    return path


def _validate_toolchain(
    *, root: Path, expected_source_commit: str
) -> tuple[dict[str, Any], dict[str, Path]]:
    if root.is_symlink() or not root.is_dir() or root.stat().st_mode & 0o222:
        raise TaskEvaluationSceneConfigurationStageProducerError(
            "scene_configuration_toolchain_root_invalid"
        )
    manifest_path = root / f"{TOOLCHAIN_SCHEMA_VERSION}.json"
    manifest = _read(
        manifest_path, code="scene_configuration_toolchain_manifest_invalid"
    )
    stages = manifest.get("stages")
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != TOOLCHAIN_SCHEMA_VERSION
        or manifest.get("status") != "published_full_byte_readback_passed"
        or manifest.get("source_commit") != expected_source_commit
        or _COMMIT.fullmatch(str(expected_source_commit or "")) is None
        or manifest.get("full_byte_service_account_readback_passed") is not True
        or manifest.get("toolchain_digest")
        != canonical_digest(manifest, digest_field="toolchain_digest")
        or not isinstance(stages, Mapping)
        or not isinstance(files, list)
        or not files
    ):
        raise TaskEvaluationSceneConfigurationStageProducerError(
            "scene_configuration_toolchain_manifest_invalid"
        )
    inventory: dict[str, tuple[str, int, bool]] = {}
    for row in files:
        if not isinstance(row, Mapping):
            raise TaskEvaluationSceneConfigurationStageProducerError(
                "scene_configuration_toolchain_inventory_invalid"
            )
        relative = str(row.get("relative_path") or "")
        digest = str(row.get("sha256") or "")
        size = row.get("size_bytes")
        executable = row.get("executable")
        if (
            not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in inventory
            or _DIGEST.fullmatch(digest) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not isinstance(executable, bool)
        ):
            raise TaskEvaluationSceneConfigurationStageProducerError(
                "scene_configuration_toolchain_inventory_invalid"
            )
        inventory[relative] = (digest, size, executable)
    observed: set[str] = set()
    for path in root.rglob("*"):
        if path == manifest_path:
            continue
        if path.is_symlink():
            raise TaskEvaluationSceneConfigurationStageProducerError(
                "scene_configuration_toolchain_symlink_forbidden"
            )
        if path.is_dir():
            if path.stat().st_mode & 0o222:
                raise TaskEvaluationSceneConfigurationStageProducerError(
                    "scene_configuration_toolchain_not_read_only"
                )
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        expected = inventory.get(relative)
        if (
            expected is None
            or path.stat().st_size != expected[1]
            or _sha256(path) != expected[0]
            or bool(path.stat().st_mode & 0o111) != expected[2]
            or path.stat().st_mode & 0o222
        ):
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_toolchain_file_invalid:{relative}"
            )
        observed.add(relative)
    if observed != set(inventory):
        raise TaskEvaluationSceneConfigurationStageProducerError(
            "scene_configuration_toolchain_inventory_incomplete"
        )
    entrypoints: dict[str, Path] = {}
    expected_ids = {identity.adapter_id for identity in ADMITTED_PRODUCER_IDENTITIES}
    if set(stages) != expected_ids:
        raise TaskEvaluationSceneConfigurationStageProducerError(
            "scene_configuration_toolchain_stage_set_invalid"
        )
    for adapter_id, row in stages.items():
        if (
            not isinstance(row, Mapping)
            or row.get("network_policy")
            not in {"openai_api_only", "provider_and_openai_api", "disabled"}
            or row.get("secrets_via_files_only") is not True
            or row.get("raw_secret_values_in_argv_or_logs") is not False
        ):
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_toolchain_stage_invalid:{adapter_id}"
            )
        executable = _relative_file(
            root,
            row.get("entrypoint"),
            code=f"scene_configuration_toolchain_stage_invalid:{adapter_id}",
        )
        relative = executable.relative_to(root).as_posix()
        if relative not in inventory or inventory[relative][2] is not True:
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_toolchain_stage_invalid:{adapter_id}"
            )
        entrypoints[str(adapter_id)] = executable
        component = _relative_file(
            root,
            row.get("component_entrypoint"),
            code=f"scene_configuration_toolchain_stage_invalid:{adapter_id}",
        )
        component_relative = component.relative_to(root).as_posix()
        package_manifest = _relative_file(
            root,
            row.get("component_package_manifest"),
            code=f"scene_configuration_toolchain_stage_invalid:{adapter_id}",
        )
        package_manifest_relative = package_manifest.relative_to(root).as_posix()
        try:
            package = validate_scene_configuration_component_package(
                root=package_manifest.parent,
                expected_adapter_id=str(adapter_id),
            )
        except TaskEvaluationSceneConfigurationComponentPackageError as exc:
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_toolchain_stage_invalid:{adapter_id}"
            ) from exc
        if (
            component_relative not in inventory
            or inventory[component_relative][2] is not True
            or package_manifest_relative not in inventory
            or not _DIGEST.fullmatch(
                str(row.get("component_package_digest") or "")
            )
            or row.get("component_package_digest") != package["package_digest"]
            or component
            != (package_manifest.parent / package["driver_entrypoint"]).resolve()
            or row.get("network_policy") != package["network_policy"]
        ):
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_toolchain_stage_invalid:{adapter_id}"
            )
    return manifest, entrypoints


def validate_scene_configuration_toolchain(
    *, root: str | Path, expected_source_commit: str
) -> dict[str, Any]:
    """Validate every toolchain byte without constructing executable handlers."""

    manifest, _entrypoints = _validate_toolchain(
        root=Path(root).expanduser().resolve(),
        expected_source_commit=expected_source_commit,
    )
    return manifest


def _secret_values(environment: Mapping[str, str]) -> list[str]:
    values: list[str] = []
    for name in _SECRET_ENVIRONMENT_FILES:
        unresolved = str(environment.get(name) or "").strip()
        if not unresolved:
            continue
        path = Path(unresolved).expanduser()
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_secret_file_invalid:{name}"
            )
        value = path.read_text(encoding="utf-8").strip()
        if not value:
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_secret_file_invalid:{name}"
            )
        values.append(value)
    return values


def _redact(value: str, secrets: Sequence[str]) -> str:
    result = str(value or "")
    for secret in secrets:
        result = result.replace(secret, "REDACTED_SECRET")
    return redacted_failure_text(result)


def _captured_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _handler(
    *,
    identity: SceneConfigurationStageProducerIdentity,
    toolchain_root: Path,
    expected_source_commit: str,
    diagnostic_only: bool,
    runner: Runner,
    environment: Mapping[str, str],
) -> Callable[..., Mapping[str, Any]]:
    manifest, entrypoints = _validate_toolchain(
        root=toolchain_root, expected_source_commit=expected_source_commit
    )
    executable = entrypoints[identity.adapter_id]
    toolchain_digest = str(manifest["toolchain_digest"])

    def execute(
        *,
        stage: Mapping[str, Any],
        envelope: Mapping[str, Any],
        configuration: Mapping[str, Any],
        configuration_path: Path,
        dependency_results: tuple[Mapping[str, Any], ...],
        output_root: Path,
    ) -> Mapping[str, Any]:
        input_path = output_root / "stage_production_input.v1.json"
        dependency_path = output_root / "dependency_results.v1.json"
        result_path = output_root / f"{PRODUCTION_RESULT_SCHEMA_VERSION}.json"
        input_value = {
            "schema_version": "task_evaluation_scene_configuration_stage_production_input.v1",
            "run_id": envelope["run_id"],
            "stage": dict(stage),
            "configuration": dict(configuration),
            "configuration_sha256": _sha256(configuration_path),
            "source_commit": expected_source_commit,
            "construction_source_commit": str(
                envelope.get("expected_production_commit")
                or expected_source_commit
            ),
            "execution_mode": (
                "diagnostic_only" if diagnostic_only else "production"
            ),
            "toolchain_digest": toolchain_digest,
            "construction_envelope": dict(envelope),
        }
        input_path.write_text(canonical_json(input_value) + "\n", encoding="utf-8")
        dependency_path.write_text(
            canonical_json(list(dependency_results)) + "\n", encoding="utf-8"
        )
        run_environment = {
            **dict(environment),
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT": str(input_path),
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES": str(
                dependency_path
            ),
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT": str(output_root),
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_RESULT": str(result_path),
        }
        if diagnostic_only:
            run_environment[
                "BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_ONLY"
            ] = "1"
        else:
            run_environment.pop(
                "BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_ONLY", None
            )
        if any(str(run_environment.get(name) or "").strip() for name in _RAW_SECRET_ENVIRONMENT_NAMES):
            raise TaskEvaluationSceneConfigurationStageProducerError(
                "scene_configuration_raw_secret_environment_forbidden"
            )
        secrets = _secret_values(run_environment)
        log_path = output_root / "stage_producer.log"
        timeout_seconds = _STAGE_TIMEOUTS_SECONDS[identity.adapter_id]
        try:
            completed = runner(
                [str(executable)],
                cwd=toolchain_root,
                env=run_environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            log_path.write_text(
                _redact(
                    f"{_captured_text(exc.stdout)}\n{_captured_text(exc.stderr)}",
                    secrets,
                ),
                encoding="utf-8",
            )
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_stage_producer_timeout:{identity.adapter_id}:"
                f"{timeout_seconds}"
            ) from exc
        log_path.write_text(
            _redact(f"{completed.stdout}\n{completed.stderr}", secrets),
            encoding="utf-8",
        )
        if completed.returncode != 0 or not result_path.is_file():
            raise TaskEvaluationSceneConfigurationStageProducerError(
                f"scene_configuration_stage_producer_failed:{identity.adapter_id}:"
                f"{completed.returncode}"
            )
        result = _read(
            result_path, code="scene_configuration_stage_production_result_invalid"
        )
        observed_roles = {
            str(row.get("role") or "")
            for row in result.get("artifacts") or []
            if isinstance(row, Mapping)
        }
        diagnostic_rejection = (
            diagnostic_only
            and identity.adapter_id == "artifixer3d_observed_object_removal"
            and observed_roles == _DIAGNOSTIC_REJECTED_ARTIFACT_ROLES
        )
        if diagnostic_rejection:
            if (
                result.get("diagnostic_only") is not True
                or result.get("qualification_eligible") is not False
                or result.get("configured_revision_publication_permitted") is not False
                or result.get("offering_publication_permitted") is not False
                or result.get("terminal_e2e_completion_permitted") is not False
            ):
                raise TaskEvaluationSceneConfigurationStageProducerError(
                    "scene_configuration_stage_production_diagnostic_claim_invalid"
                )
        elif observed_roles != expected_stage_artifact_roles(
            identity.adapter_id,
            supplemental_destination=envelope_declares_supplemental_destination(
                envelope
            ),
        ):
            raise TaskEvaluationSceneConfigurationStageProducerError(
                "scene_configuration_stage_production_artifact_roles_invalid"
            )
        return result

    return execute


def builtin_scene_configuration_stage_producer_registry(
    *,
    expected_source_commit: str,
    toolchain_root: str | Path | None = None,
    runner: Runner = subprocess.run,
    environment: Mapping[str, str] | None = None,
    diagnostic_only: bool = False,
) -> SceneConfigurationStageProducerRegistry:
    """Resolve all GPU stage producers from one exact published toolchain."""

    values = os.environ if environment is None else environment
    unresolved = str(
        toolchain_root or values.get(TOOLCHAIN_ROOT_ENV) or ""
    ).strip()
    if not unresolved:
        raise TaskEvaluationSceneConfigurationStageProducerError(
            "scene_configuration_toolchain_environment_missing"
        )
    root = Path(unresolved).expanduser().resolve()
    handlers = {
        identity: _handler(
            identity=identity,
            toolchain_root=root,
            expected_source_commit=expected_source_commit,
            diagnostic_only=diagnostic_only,
            runner=runner,
            environment=values,
        )
        for identity in ADMITTED_PRODUCER_IDENTITIES
    }
    return SceneConfigurationStageProducerRegistry(handlers)


__all__ = [
    "TOOLCHAIN_ROOT_ENV",
    "TOOLCHAIN_SCHEMA_VERSION",
    "builtin_scene_configuration_stage_producer_registry",
    "validate_scene_configuration_toolchain",
]
