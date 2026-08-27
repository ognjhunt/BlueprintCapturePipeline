#!/usr/bin/env python3
"""Resume a sealed scene configuration checkpoint on one diagnostic GPU."""

from __future__ import annotations

import hashlib
import json
import os
import stat
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
from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
    PARENT_DEADLINE_EPOCH_ENV,
)


RESULT_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_diagnostic_provider_result.v1"
)
RESULT_FILENAME = "task_evaluation_scene_configuration_provider_result.v1.json"


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
            except ValueError as exc:
                raise ValueError(
                    "scene_configuration_diagnostic_provider_artifact_invalid"
                ) from exc
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
    try:
        parent_deadline_epoch = float(os.environ[PARENT_DEADLINE_EPOCH_ENV])
        _restore_checkpoint_modes(checkpoint_root)
        checkpoint = validate_scene_configuration_diagnostic_checkpoint(
            checkpoint_root=checkpoint_root
        )
        portable = _read(runtime / "input/portable_construction_envelope.v1.json")
        envelope = _hydrate_envelope(runtime, portable)
        configurations = {}
        for row in envelope["stage_configuration_references"]:
            stage_id = str(row["stage_id"])
            path = Path(row["materialized_path"]).resolve()
            configurations[stage_id] = (_read(path), path)
        registry = SceneConfigurationAdapterRegistry(
            builtin_scene_configuration_adapter_handlers()
        )
        os.environ[
            "BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_CHECKPOINT_ROOT"
        ] = str(checkpoint_root)
        producers = builtin_scene_configuration_stage_producer_registry(
            expected_source_commit=str(envelope["expected_production_commit"])
        )
        advanced: dict = checkpoint

        def write_checkpoint(results) -> None:
            nonlocal advanced
            destination = output / "diagnostic_checkpoints" / f"after-stage-{len(results)}"
            advanced = advance_scene_configuration_diagnostic_checkpoint(
                checkpoint_root=checkpoint_root,
                stage_results=list(results),
                stage_sequence=envelope["recipe"]["stage_sequence"],
                configurations=configurations,
                output_root=destination,
            )

        def resume_stage_one(**kwargs):
            return producers.execute(**{key: value for key, value in kwargs.items() if key not in {"checkpoint", "checkpoint_root"}})

        chain = execute_scene_configuration_diagnostic_stage_chain(
            checkpoint_root=checkpoint_root,
            envelope=envelope,
            configurations=configurations,
            output_root=stages_root,
            registry=registry,
            producer_registry=producers,
            stage_one_resume_producer=resume_stage_one,
            stage_checkpoint_writer=write_checkpoint,
            parent_deadline_epoch=parent_deadline_epoch,
        )
        chain = _portable_stage_chain(chain, output_root=output)
        toolchain = _read(
            runtime / "toolchain" / f"{TOOLCHAIN_SCHEMA_VERSION}.json"
        )
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": STATUS,
            "source_checkpoint_digest": checkpoint["checkpoint_digest"],
            "advanced_checkpoint_digest": advanced["checkpoint_digest"],
            "diagnostic_source_commit": envelope["expected_production_commit"],
            "diagnostic_toolchain_digest": toolchain["toolchain_digest"],
            "diagnostic_stage_chain": chain,
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
        }
    except Exception as exc:
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
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    result_path.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return 0 if result["status"] == STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
