#!/usr/bin/env python3
"""Resume a sealed scene configuration checkpoint on one diagnostic GPU."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from blueprint_pipeline.core.common import redacted_failure_detail
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.task_evaluation_scene_configuration_adapters import (
    SceneConfigurationAdapterRegistry,
)
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_adapters import (
    builtin_scene_configuration_adapter_handlers,
)
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
    TOOLCHAIN_SCHEMA_VERSION,
    builtin_scene_configuration_stage_producer_registry,
)
from blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_checkpoint import (
    SCHEMA_VERSION as CHECKPOINT_SCHEMA_VERSION,
    advance_scene_configuration_diagnostic_checkpoint,
    validate_scene_configuration_diagnostic_checkpoint,
)
from blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_runtime import (
    STATUS,
    execute_scene_configuration_diagnostic_stage_chain,
)
from blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_mode import (
    CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE,
    FRESH_DIAGNOSTIC_BOOTSTRAP_MODE,
    validate_diagnostic_bootstrap_mode,
)
from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
    PARENT_DEADLINE_EPOCH_ENV,
)


RESULT_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_diagnostic_provider_result.v1"
)
BUNDLE_SCHEMA_VERSION = "task_evaluation_scene_configuration_provider_bundle.v1"
RESULT_FILENAME = "task_evaluation_scene_configuration_provider_result.v1.json"
WARM_READINESS_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_warm_readiness.v1"
)
WARM_READINESS_FILENAME = WARM_READINESS_SCHEMA_VERSION + ".json"
WARM_SOURCE_COMMIT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_SOURCE_COMMIT"
WARM_OVERLAY_MANIFEST_ENV = (
    "BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_SOURCE_OVERLAY_MANIFEST"
)
WARM_OVERLAY_MANIFEST_DIGEST_ENV = (
    "BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_SOURCE_OVERLAY_MANIFEST_DIGEST"
)
WARM_SESSION_DIGEST_ENV = "BLUEPRINT_SCENE_CONFIGURATION_WARM_SESSION_DIGEST"
WARM_PROVIDER_INSTANCE_ID_ENV = (
    "BLUEPRINT_SCENE_CONFIGURATION_WARM_PROVIDER_INSTANCE_ID"
)
WARM_BOOTSTRAP_ALLOCATION_BINDING_DIGEST_ENV = (
    "BLUEPRINT_SCENE_CONFIGURATION_WARM_BOOTSTRAP_ALLOCATION_BINDING_DIGEST"
)
WARM_OVERLAY_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_warm_source_overlay.v1"
)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if path.is_symlink() or not isinstance(value, dict):
        raise ValueError("scene_configuration_diagnostic_provider_input_invalid")
    return value


def _effective_diagnostic_bootstrap_mode(
    *, bundle_bootstrap_mode: object, warm_source_commit: str
) -> str:
    bundle_mode = validate_diagnostic_bootstrap_mode(bundle_bootstrap_mode)
    return (
        CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE
        if warm_source_commit
        else bundle_mode
    )


def _diagnostic_implementation_identity(
    *, runtime: Path, checkpoint: dict, base_source_commit: str
) -> tuple[str, dict]:
    """Separate a warm implementation overlay from immutable carried inputs."""

    source_commit = str(os.environ.get(WARM_SOURCE_COMMIT_ENV) or "")
    manifest_value = str(os.environ.get(WARM_OVERLAY_MANIFEST_ENV) or "")
    expected_digest = str(
        os.environ.get(WARM_OVERLAY_MANIFEST_DIGEST_ENV) or ""
    )
    session_digest = str(os.environ.get(WARM_SESSION_DIGEST_ENV) or "")
    provider_instance_id = str(
        os.environ.get(WARM_PROVIDER_INSTANCE_ID_ENV) or ""
    )
    bootstrap_binding_digest = str(
        os.environ.get(WARM_BOOTSTRAP_ALLOCATION_BINDING_DIGEST_ENV) or ""
    )
    configured = [
        bool(source_commit),
        bool(manifest_value),
        bool(expected_digest),
        bool(session_digest),
        bool(provider_instance_id),
        bool(bootstrap_binding_digest),
    ]
    if not any(configured):
        return base_source_commit, {}
    if (
        not all(configured)
        or _COMMIT.fullmatch(source_commit) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", session_digest) is None
        or re.fullmatch(r"[1-9][0-9]*", provider_instance_id) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", bootstrap_binding_digest) is None
    ):
        raise ValueError("scene_configuration_diagnostic_warm_overlay_identity_invalid")
    manifest_path = Path(manifest_value).resolve()
    expected_path = runtime / f"{WARM_OVERLAY_SCHEMA_VERSION}.json"
    if manifest_path != expected_path or manifest_path.is_symlink():
        raise ValueError("scene_configuration_diagnostic_warm_overlay_identity_invalid")
    manifest = _read(manifest_path)
    if (
        manifest.get("schema_version") != WARM_OVERLAY_SCHEMA_VERSION
        or manifest.get("status") != "ready"
        or manifest.get("source_commit") != source_commit
        or manifest.get("source_checkpoint_digest")
        != checkpoint.get("checkpoint_digest")
        or manifest.get("scientific_binding_digest")
        != (checkpoint.get("scientific_bindings") or {}).get("binding_digest")
        or manifest.get("diagnostic_only") is not True
        or manifest.get("development_only") is not True
        or manifest.get("qualification_eligible") is not False
        or manifest.get("configured_revision_publication_permitted") is not False
        or manifest.get("offering_publication_permitted") is not False
        or manifest.get("terminal_e2e_completion_permitted") is not False
        or manifest.get("arbitrary_command_permitted") is not False
        or manifest.get("raw_secret_values_recorded") is not False
        or manifest.get("manifest_digest") != expected_digest
        or manifest.get("manifest_digest")
        != canonical_digest(manifest, digest_field="manifest_digest")
    ):
        raise ValueError("scene_configuration_diagnostic_warm_overlay_identity_invalid")
    return source_commit, {
        "warm_source_overlay_used": True,
        "base_bundle_source_commit": base_source_commit,
        "diagnostic_source_overlay_manifest_digest": expected_digest,
        "warm_session_digest": session_digest,
        "warm_provider_instance_id": provider_instance_id,
        "warm_bootstrap_allocation_binding_digest": bootstrap_binding_digest,
    }


def _runtime_file(
    runtime: Path, relative: object, *, digest: object, size_bytes: object
) -> Path:
    value = str(relative or "")
    if not value or value.startswith("/") or ".." in Path(value).parts:
        raise ValueError("scene_configuration_diagnostic_provider_relative_path_invalid")
    path = (runtime / value).resolve()
    try:
        path.relative_to(runtime)
    except ValueError as exc:
        raise ValueError(
            "scene_configuration_diagnostic_provider_relative_path_invalid"
        ) from exc
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != size_bytes
        or _sha256(path) != digest
    ):
        raise ValueError("scene_configuration_diagnostic_provider_bound_file_invalid")
    return path


def _hydrate_envelope(runtime: Path, portable: dict) -> dict:
    if portable.get("envelope_digest") != canonical_digest(
        portable, digest_field="envelope_digest"
    ):
        raise ValueError("scene_configuration_diagnostic_provider_envelope_invalid")
    envelope = json.loads(json.dumps(portable))
    for row in envelope.get("materialized_references") or []:
        row["materialized_path"] = str(
            _runtime_file(
                runtime,
                row.get("provider_relative_path"),
                digest=row.get("digest"),
                size_bytes=row.get("size_bytes"),
            )
        )
    render = envelope.get("render_inputs_result") or {}
    for key in ("camera_calibration", "render_manifest"):
        row = render.get(key)
        if row is None:
            continue
        row["path"] = str(
            _runtime_file(
                runtime,
                row.get("path"),
                digest=row.get("digest"),
                size_bytes=row.get("size_bytes"),
            )
        )
    for row in render.get("derived_frames") or []:
        row["path"] = str(
            _runtime_file(
                runtime,
                row.get("path"),
                digest=row.get("digest"),
                size_bytes=row.get("size_bytes"),
            )
        )
        mask = row.get("source_object_mask") or {}
        mask["path"] = str(
            _runtime_file(
                runtime,
                mask.get("path"),
                digest=mask.get("digest"),
                size_bytes=mask.get("size_bytes"),
            )
        )
    cutout = render.get("derived_gaussian_cutout") or {}
    for key in ("source_object_candidate", "retained_scene_without_source_object"):
        row = cutout.get(key)
        if not isinstance(row, dict):
            continue
        row["path"] = str(
            _runtime_file(
                runtime,
                row.get("path"),
                digest=row.get("digest"),
                size_bytes=row.get("size_bytes"),
            )
        )
    for row in envelope.get("stage_configuration_references") or []:
        row["materialized_path"] = str(
            _runtime_file(
                runtime,
                row.get("relative_path"),
                digest=row.get("digest"),
                size_bytes=row.get("size_bytes"),
            )
        )
    envelope["portable_envelope_digest"] = portable["envelope_digest"]
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    return envelope


def _restore_checkpoint_modes(checkpoint_root: Path) -> None:
    manifest = _read(checkpoint_root / f"{CHECKPOINT_SCHEMA_VERSION}.json")
    for row in manifest.get("inventory") or []:
        relative = str(row.get("relative_path") or "")
        path = (checkpoint_root / relative).resolve()
        try:
            path.relative_to(checkpoint_root)
        except ValueError as exc:
            raise ValueError(
                "scene_configuration_diagnostic_checkpoint_inventory_invalid"
            ) from exc
        mode = row.get("mode")
        if (
            not relative
            or ".." in Path(relative).parts
            or path.is_symlink()
            or not path.is_file()
            or not isinstance(mode, int)
            or isinstance(mode, bool)
            or mode < 0
            or mode > 0o777
        ):
            raise ValueError(
                "scene_configuration_diagnostic_checkpoint_inventory_invalid"
            )
        os.chmod(path, mode)


def _portable_stage_chain(chain: dict, *, output_root: Path) -> dict:
    root = output_root.resolve()
    portable = json.loads(json.dumps(chain))
    for result in portable.get("stage_results") or []:
        for artifact in result.get("output_artifacts") or []:
            source = Path(str(artifact.get("path") or "")).resolve()
            try:
                relative = source.relative_to(root)
            except ValueError:
                stage_id = str(result.get("stage_id") or "")
                role = str(artifact.get("role") or "")
                if not stage_id or not role or "/" in role or ".." in role:
                    raise ValueError(
                        "scene_configuration_diagnostic_provider_artifact_invalid"
                    )
                suffix = source.suffix if len(source.suffix) <= 12 else ""
                destination = (
                    root
                    / "carried_stage_artifacts"
                    / stage_id
                    / f"{role}{suffix}"
                )
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
                if destination.exists() or destination.is_symlink():
                    raise ValueError(
                        "scene_configuration_diagnostic_provider_artifact_invalid"
                    )
                shutil.copyfile(source, destination)
                os.chmod(destination, int(artifact.get("mode") or 0o440))
                source = destination.resolve()
                relative = source.relative_to(root)
                artifact["path"] = str(source)
            if (
                source.is_symlink()
                or not source.is_file()
                or source.stat().st_size != artifact.get("size_bytes")
                or _sha256(source) != artifact.get("digest")
            ):
                raise ValueError(
                    "scene_configuration_diagnostic_provider_artifact_invalid"
                )
            artifact["provider_output_relative_path"] = relative.as_posix()
        result["stage_result_digest"] = canonical_digest(
            result, digest_field="stage_result_digest"
        )
    portable["stage_result_digests"] = [
        result["stage_result_digest"] for result in portable["stage_results"]
    ]
    portable["result_digest"] = canonical_digest(
        portable, digest_field="result_digest"
    )
    return portable


def _advanced_checkpoint_reference(
    *, output: Path, advanced_root: Path, advanced: dict
) -> dict:
    manifest = advanced_root / f"{CHECKPOINT_SCHEMA_VERSION}.json"
    return {
        "provider_output_relative_root": advanced_root.relative_to(output).as_posix(),
        "manifest_relative_path": manifest.relative_to(output).as_posix(),
        "manifest_sha256": _sha256(manifest),
        "checkpoint_digest": advanced["checkpoint_digest"],
        "completed_stage_prefix_count": advanced["completed_stage_prefix_count"],
        "file_count": 1 + len(advanced["inventory"]),
        "total_bytes": sum(
            path.stat().st_size for path in advanced_root.rglob("*") if path.is_file()
        ),
    }


def _retained_checkpoint_after_failure(
    *,
    output: Path,
    checkpoint_root: Path,
    checkpoint: dict | None,
    advanced: dict | None,
    advanced_root: Path | None,
) -> dict | None:
    """Retain only a validated prefix that already carries every paid stage."""

    if (
        not isinstance(checkpoint, dict)
        or not isinstance(advanced, dict)
        or int(advanced.get("completed_stage_prefix_count") or 0) < 3
    ):
        return None
    retained_root = advanced_root
    if retained_root is None:
        retained_root = (
            output
            / "diagnostic_checkpoints"
            / f"carried-source-prefix-{checkpoint['completed_stage_prefix_count']}"
        )
        shutil.copytree(
            checkpoint_root,
            retained_root,
            copy_function=shutil.copy2,
            symlinks=False,
        )
    return _advanced_checkpoint_reference(
        output=output, advanced_root=retained_root, advanced=advanced
    )


def _install_warm_ready_checkpoint(
    *,
    runtime: Path,
    checkpoint_root: Path,
    checkpoint: dict,
    diagnostic_bootstrap_mode: str,
    expected_scientific_binding_digest: str,
) -> dict:
    """Install only a validated Stage-3+ prefix as the warm runtime source."""

    completed_count = checkpoint.get("completed_stage_prefix_count")
    completed_results = checkpoint.get("completed_stage_results")
    completed_ids = (
        [str(row.get("stage_id") or "") for row in completed_results]
        if isinstance(completed_results, list)
        and all(isinstance(row, dict) for row in completed_results)
        else []
    )
    if (
        not isinstance(completed_count, int)
        or isinstance(completed_count, bool)
        or completed_count < 3
        or len(completed_ids) != completed_count
        or any(not stage_id for stage_id in completed_ids)
        or (checkpoint.get("scientific_bindings") or {}).get("binding_digest")
        != expected_scientific_binding_digest
    ):
        raise ValueError("scene_configuration_warm_checkpoint_prefix_invalid")
    validate_scene_configuration_diagnostic_checkpoint(
        checkpoint_root=checkpoint_root,
        expected_scientific_binding_digest=expected_scientific_binding_digest,
    )
    checkpoint_target = runtime / "input/diagnostic_checkpoint"
    checkpoint_staging = runtime / "input/.diagnostic_checkpoint.next"
    shutil.rmtree(checkpoint_staging, ignore_errors=True)
    shutil.copytree(
        checkpoint_root,
        checkpoint_staging,
        copy_function=shutil.copy2,
        symlinks=False,
    )
    validate_scene_configuration_diagnostic_checkpoint(
        checkpoint_root=checkpoint_staging,
        expected_scientific_binding_digest=expected_scientific_binding_digest,
    )
    shutil.rmtree(checkpoint_target, ignore_errors=True)
    os.replace(checkpoint_staging, checkpoint_target)
    readiness = {
        "schema_version": WARM_READINESS_SCHEMA_VERSION,
        "status": "ready_after_validated_stage_three_prefix",
        "diagnostic_bootstrap_mode": diagnostic_bootstrap_mode,
        "advanced_checkpoint_digest": checkpoint["checkpoint_digest"],
        "completed_stage_prefix_count": completed_count,
        "completed_stage_ids": completed_ids,
        "scientific_binding_digest": expected_scientific_binding_digest,
        "raw_secret_values_recorded": False,
        "readiness_digest": "",
    }
    readiness["readiness_digest"] = canonical_digest(
        readiness, digest_field="readiness_digest"
    )
    (runtime / WARM_READINESS_FILENAME).write_text(
        canonical_json(readiness) + "\n", encoding="utf-8"
    )
    return readiness


def main() -> int:
    runtime = Path(
        os.environ.get(
            "BLUEPRINT_SCENE_CONFIGURATION_RUNTIME_ROOT",
            "/workspace/task_evaluation_scene_configuration_provider_bundle/provider_runtime",
        )
    ).resolve()
    output = Path(
        os.environ.get(
            "BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_ROOT",
            str(runtime.parent / "runtime_output"),
        )
    ).resolve()
    result_path = Path(
        os.environ.get(
            "BLUEPRINT_SCENE_CONFIGURATION_PROVIDER_RESULT",
            str(output / RESULT_FILENAME),
        )
    ).resolve()
    checkpoint_root = runtime / "input/diagnostic_checkpoint"
    output.mkdir(parents=True, exist_ok=True)
    stages_root = output / "stages"
    stages_root.mkdir(mode=0o750)
    checkpoint: dict | None = None
    advanced: dict | None = None
    advanced_root: Path | None = None
    diagnostic_source_commit: str | None = None
    diagnostic_run_id: str | None = None
    warm_identity: dict = {}
    toolchain: dict | None = None
    active_checkpoint_root: Path | None = None
    source_checkpoint_digest: str | None = None
    diagnostic_bootstrap_mode: str | None = None
    try:
        parent_deadline_epoch = float(os.environ[PARENT_DEADLINE_EPOCH_ENV])
        bundle_manifest = _read(runtime / f"{BUNDLE_SCHEMA_VERSION}.json")
        diagnostic_bootstrap_mode = _effective_diagnostic_bootstrap_mode(
            bundle_bootstrap_mode=bundle_manifest.get(
                "diagnostic_bootstrap_mode"
            ),
            warm_source_commit=str(os.environ.get(WARM_SOURCE_COMMIT_ENV) or ""),
        )
        if diagnostic_bootstrap_mode == FRESH_DIAGNOSTIC_BOOTSTRAP_MODE:
            if checkpoint_root.exists():
                raise ValueError(
                    "scene_configuration_fresh_diagnostic_checkpoint_unexpected"
                )
        else:
            if (
                diagnostic_bootstrap_mode
                != CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE
            ):
                raise ValueError(
                    "scene_configuration_diagnostic_bootstrap_mode_invalid"
                )
            _restore_checkpoint_modes(checkpoint_root)
            checkpoint = validate_scene_configuration_diagnostic_checkpoint(
                checkpoint_root=checkpoint_root
            )
            active_checkpoint_root = checkpoint_root
            source_checkpoint_digest = checkpoint["checkpoint_digest"]
        portable = _read(runtime / "input/portable_construction_envelope.v1.json")
        envelope = _hydrate_envelope(runtime, portable)
        diagnostic_run_id = str(envelope["run_id"])
        diagnostic_source_commit, warm_identity = (
            _diagnostic_implementation_identity(
                runtime=runtime,
                checkpoint=checkpoint or {},
                base_source_commit=str(bundle_manifest["source_commit"]),
            )
        )
        configurations = {}
        for row in envelope["stage_configuration_references"]:
            stage_id = str(row["stage_id"])
            path = Path(row["materialized_path"]).resolve()
            configurations[stage_id] = (_read(path), path)
        registry = SceneConfigurationAdapterRegistry(
            builtin_scene_configuration_adapter_handlers()
        )
        if checkpoint is not None:
            os.environ[
                "BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_CHECKPOINT_ROOT"
            ] = str(checkpoint_root)
        else:
            os.environ.pop(
                "BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_CHECKPOINT_ROOT", None
            )
        producers = builtin_scene_configuration_stage_producer_registry(
            expected_source_commit=str(envelope["expected_production_commit"])
        )
        advanced = checkpoint

        def write_checkpoint(results, source_checkpoint_root: Path) -> None:
            nonlocal checkpoint, active_checkpoint_root, advanced, advanced_root
            if checkpoint is None:
                checkpoint = validate_scene_configuration_diagnostic_checkpoint(
                    checkpoint_root=source_checkpoint_root
                )
                if (
                    (checkpoint.get("scientific_bindings") or {}).get(
                        "binding_digest"
                    )
                    != bundle_manifest.get(
                        "diagnostic_scientific_binding_digest"
                    )
                ):
                    raise ValueError(
                        "scene_configuration_fresh_diagnostic_binding_mismatch"
                    )
            previous_checkpoint_root = source_checkpoint_root.resolve()
            destination = output / "diagnostic_checkpoints" / f"after-stage-{len(results)}"
            advanced = advance_scene_configuration_diagnostic_checkpoint(
                checkpoint_root=previous_checkpoint_root,
                stage_results=list(results),
                stage_sequence=envelope["recipe"]["stage_sequence"],
                configurations=configurations,
                output_root=destination,
            )
            advanced_root = destination
            checkpoint = advanced
            active_checkpoint_root = destination.resolve()
            if previous_checkpoint_root.is_relative_to(output.resolve()):
                shutil.rmtree(previous_checkpoint_root)
            if int(advanced["completed_stage_prefix_count"]) >= 3:
                _install_warm_ready_checkpoint(
                    runtime=runtime,
                    checkpoint_root=active_checkpoint_root,
                    checkpoint=advanced,
                    diagnostic_bootstrap_mode=diagnostic_bootstrap_mode,
                    expected_scientific_binding_digest=bundle_manifest[
                        "diagnostic_scientific_binding_digest"
                    ],
                )

        def resume_stage_one(**kwargs):
            return producers.execute(**{key: value for key, value in kwargs.items() if key not in {"checkpoint", "checkpoint_root"}})

        toolchain = _read(
            runtime / "toolchain" / f"{TOOLCHAIN_SCHEMA_VERSION}.json"
        )
        chain = execute_scene_configuration_diagnostic_stage_chain(
            diagnostic_bootstrap_mode=diagnostic_bootstrap_mode,
            checkpoint_root=(
                None
                if diagnostic_bootstrap_mode
                == FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
                else checkpoint_root
            ),
            envelope=envelope,
            configurations=configurations,
            output_root=stages_root,
            registry=registry,
            producer_registry=producers,
            stage_one_resume_producer=resume_stage_one,
            stage_checkpoint_writer=write_checkpoint,
            parent_deadline_epoch=parent_deadline_epoch,
        )
        if advanced_root is None:
            if active_checkpoint_root is None:
                raise ValueError(
                    "scene_configuration_fresh_diagnostic_checkpoint_missing"
                )
            advanced_root = output / "diagnostic_checkpoints" / "after-stage-6"
            advanced = advance_scene_configuration_diagnostic_checkpoint(
                checkpoint_root=active_checkpoint_root,
                stage_results=chain["stage_results"],
                stage_sequence=envelope["recipe"]["stage_sequence"],
                configurations=configurations,
                output_root=advanced_root,
            )
        chain = _portable_stage_chain(chain, output_root=output)
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": STATUS,
            "source_checkpoint_digest": source_checkpoint_digest,
            "advanced_checkpoint_digest": advanced["checkpoint_digest"],
            "advanced_checkpoint": _advanced_checkpoint_reference(
                output=output, advanced_root=advanced_root, advanced=advanced
            ),
            "diagnostic_source_commit": diagnostic_source_commit,
            "diagnostic_run_id": diagnostic_run_id,
            "diagnostic_toolchain_digest": toolchain["toolchain_digest"],
            "diagnostic_construction_envelope_digest": portable[
                "envelope_digest"
            ],
            "diagnostic_stage_chain": chain,
            "diagnostic_bootstrap_mode": diagnostic_bootstrap_mode,
            "diagnostic_scientific_binding_digest": bundle_manifest[
                "diagnostic_scientific_binding_digest"
            ],
            "diagnostic_only": True,
            "qualification_eligible": False,
            "executed_inside_one_parent_provider_run": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
            "provider_zero_required_after_return": True,
            "raw_secret_values_recorded": False,
            "blockers": [],
            "result_digest": "",
            **warm_identity,
        }
    except Exception as exc:
        retained_checkpoint_reference = _retained_checkpoint_after_failure(
            output=output,
            checkpoint_root=(active_checkpoint_root or checkpoint_root),
            checkpoint=checkpoint,
            advanced=advanced,
            advanced_root=advanced_root,
        )
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked_diagnostic_only",
            "diagnostic_only": True,
            "qualification_eligible": False,
            "executed_inside_one_parent_provider_run": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
            "provider_zero_required_after_return": True,
            "raw_secret_values_recorded": False,
            "blockers": [
                "scene_configuration_diagnostic_provider_failed:"
                + redacted_failure_detail(exc)
            ],
            "result_digest": "",
        }
        if (
            retained_checkpoint_reference is not None
            and diagnostic_source_commit is not None
            and isinstance(toolchain, dict)
        ):
            result.update(
                {
                    "source_checkpoint_digest": source_checkpoint_digest,
                    "advanced_checkpoint_digest": advanced["checkpoint_digest"],
                    "advanced_checkpoint": retained_checkpoint_reference,
                    "diagnostic_source_commit": diagnostic_source_commit,
                    "diagnostic_run_id": diagnostic_run_id,
                    "diagnostic_toolchain_digest": toolchain["toolchain_digest"],
                    "diagnostic_construction_envelope_digest": portable[
                        "envelope_digest"
                    ],
                    "diagnostic_bootstrap_mode": diagnostic_bootstrap_mode,
                    "diagnostic_scientific_binding_digest": bundle_manifest[
                        "diagnostic_scientific_binding_digest"
                    ],
                    **warm_identity,
                }
            )
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    result_path.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return 0 if result["status"] == STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
