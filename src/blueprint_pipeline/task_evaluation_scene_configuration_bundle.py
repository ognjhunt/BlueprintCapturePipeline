"""Build the portable single-allocation scene-configuration provider bundle."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_scene_configuration_builtin_producers import (
    TOOLCHAIN_SCHEMA_VERSION,
    validate_scene_configuration_toolchain,
)
from .task_evaluation_scene_construction_queue import ENVELOPE_SCHEMA_VERSION


PROBE_KIND = "task-evaluation-scene-configuration"
PROVIDER_BUNDLE_KIND = "task_evaluation_scene_configuration"
BUNDLE_SCHEMA_VERSION = "task_evaluation_scene_configuration_provider_bundle.v1"
ENTRYPOINT = "provider_runtime/run_task_evaluation_scene_configuration_provider.sh"
RUNNER = "provider_runtime/task_evaluation_scene_configuration_provider_runner.py"
RESULT_FILENAME = "task_evaluation_scene_configuration_provider_result.v1.json"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_MAX_MEMBER_BYTES = 2 * 1024**3
_MAX_TOTAL_BYTES = 8 * 1024**3


class TaskEvaluationSceneConfigurationBundleError(ValueError):
    """The Website-started construction could not become a portable job."""


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
        raise TaskEvaluationSceneConfigurationBundleError(code) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationBundleError(code)
    return dict(value)


def _copy_file(source: Path, destination: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_file():
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_source_file_invalid"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _sha256(destination) != _sha256(source):
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_copy_mismatch"
        )
    return {
        "relative_path": destination.as_posix(),
        "digest": _sha256(destination),
        "size_bytes": destination.stat().st_size,
    }


def _copy_tree(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_source_tree_invalid"
        )
    for path in sorted(source.rglob("*")):
        if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise TaskEvaluationSceneConfigurationBundleError(
                "scene_configuration_bundle_source_tree_symlink"
            )
        if path.is_file():
            target = destination / path.relative_to(source)
            _copy_file(path, target)
            target.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
    for directory in sorted(
        (path for path in destination.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    destination.chmod(0o555)


def _bound_file(row: Mapping[str, Any], *, code: str) -> Path:
    path = Path(str(row.get("materialized_path") or row.get("path") or "")).resolve()
    digest = str(row.get("digest") or row.get("sha256") or "")
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != row.get("size_bytes")
        or _sha256(path) != digest
    ):
        raise TaskEvaluationSceneConfigurationBundleError(code)
    return path


def _portable_render_inputs(
    *, runtime: Path, render: Mapping[str, Any]
) -> dict[str, Any]:
    portable = json.loads(json.dumps(dict(render)))
    source_result_digest = str(portable.get("result_digest") or "")
    calibration = _bound_file(render["camera_calibration"], code="scene_configuration_render_input_invalid")
    calibration_target = runtime / "input/render/cameras.json"
    _copy_file(calibration, calibration_target)
    portable["camera_calibration"].pop("materialized_path", None)
    portable["camera_calibration"]["path"] = calibration_target.relative_to(runtime).as_posix()
    manifest = _bound_file(render["render_manifest"], code="scene_configuration_render_input_invalid")
    manifest_target = runtime / "input/render/render_manifest.json"
    _copy_file(manifest, manifest_target)
    portable["render_manifest"].pop("materialized_path", None)
    portable["render_manifest"]["path"] = manifest_target.relative_to(runtime).as_posix()
    for index, row in enumerate(render.get("derived_frames") or []):
        frame = _bound_file(row, code="scene_configuration_render_input_invalid")
        target = runtime / "input/render/frames" / f"{index:04d}.png"
        _copy_file(frame, target)
        portable["derived_frames"][index].pop("materialized_path", None)
        portable["derived_frames"][index]["path"] = target.relative_to(runtime).as_posix()
    portable["control_plane_result_digest"] = source_result_digest
    portable["result_digest"] = canonical_digest(
        portable, digest_field="result_digest"
    )
    return portable


def _zip_tree(source: Path, destination: Path) -> None:
    total = 0
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_dir()):
            info = zipfile.ZipInfo(
                path.relative_to(source).as_posix() + "/",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.create_system = 3
            info.external_attr = (stat.S_IFDIR | 0o555) << 16
            archive.writestr(info, b"")
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            size = path.stat().st_size
            total += size
            if size <= 0 or size > _MAX_MEMBER_BYTES or total > _MAX_TOTAL_BYTES:
                raise TaskEvaluationSceneConfigurationBundleError(
                    "scene_configuration_bundle_archive_limit_exceeded"
                )
            info = zipfile.ZipInfo(
                path.relative_to(source).as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | (0o755 if path.stat().st_mode & 0o111 else 0o444)) << 16
            with path.open("rb") as input_stream, archive.open(info, "w") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)


def build_scene_configuration_provider_bundle(
    *,
    construction_envelope_path: str | Path,
    toolchain_root: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
    expected_source_commit: str,
) -> dict[str, Any]:
    """Package provider-authorized derived inputs; raw InteriorGS stays local."""

    if _COMMIT.fullmatch(expected_source_commit) is None:
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_source_commit_invalid"
        )
    envelope_path = Path(construction_envelope_path).resolve()
    envelope = _read(envelope_path, code="scene_configuration_bundle_envelope_invalid")
    render_inputs = envelope.get("render_inputs_result")
    if (
        envelope.get("schema_version") != ENVELOPE_SCHEMA_VERSION
        or envelope.get("expected_production_commit") != expected_source_commit
        or envelope.get("envelope_digest")
        != canonical_digest(envelope, digest_field="envelope_digest")
        or not isinstance(render_inputs, Mapping)
        or render_inputs.get("status") != "derived_method_inputs_materialized"
        or render_inputs.get("raw_interiorgs_bytes_in_provider_packet") is not False
        or render_inputs.get("result_digest")
        != canonical_digest(render_inputs, digest_field="result_digest")
    ):
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_envelope_invalid"
        )
    repo = Path(repository_root).resolve()
    toolchain = Path(toolchain_root).resolve()
    toolchain_manifest = validate_scene_configuration_toolchain(
        root=toolchain, expected_source_commit=expected_source_commit
    )
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_output_not_empty"
        )
    output.mkdir(parents=True, exist_ok=True)
    stage = output / "stage"
    runtime = stage / "provider_runtime"
    runtime.mkdir(parents=True)
    portable = json.loads(json.dumps(envelope))
    portable["control_plane_envelope_digest"] = envelope["envelope_digest"]
    portable_refs = []
    for index, row in enumerate(envelope.get("materialized_references") or []):
        contract_path = str(row.get("contract_path") or "")
        if contract_path == "scene.appearance.representation":
            continue
        source = _bound_file(row, code="scene_configuration_bundle_reference_invalid")
        target = runtime / "input/references" / f"{index:04d}{source.suffix}"
        _copy_file(source, target)
        copied = dict(row)
        copied.pop("materialized_path", None)
        copied.pop("path", None)
        copied["provider_relative_path"] = target.relative_to(runtime).as_posix()
        portable_refs.append(copied)
    portable["materialized_references"] = portable_refs
    portable_configs = []
    for stage_row, row in zip(
        envelope["recipe"]["stage_sequence"],
        envelope["stage_configuration_references"],
        strict=True,
    ):
        source = _bound_file(row, code="scene_configuration_bundle_configuration_invalid")
        target = runtime / "input/configurations" / f"{stage_row['stage_id']}.json"
        _copy_file(source, target)
        portable_configs.append(
            {
                "stage_id": stage_row["stage_id"],
                "relative_path": target.relative_to(runtime).as_posix(),
                "digest": _sha256(target),
                "size_bytes": target.stat().st_size,
            }
        )
    portable["stage_configuration_references"] = portable_configs
    portable["render_inputs_result"] = _portable_render_inputs(
        runtime=runtime, render=render_inputs
    )
    raw_paths = [
        row for row in envelope.get("materialized_references") or []
        if row.get("contract_path") == "scene.appearance.representation"
    ]
    portable["provider_disclosure_receipt"] = {
        "raw_interiorgs_reference_count_omitted": len(raw_paths),
        "raw_interiorgs_bytes_in_provider_bundle": False,
        "derived_rendered_views_in_provider_bundle": True,
    }
    portable["envelope_digest"] = canonical_digest(
        portable, digest_field="envelope_digest"
    )
    (runtime / "input/portable_construction_envelope.v1.json").write_text(
        canonical_json(portable) + "\n", encoding="utf-8"
    )
    portable_toolchain = runtime / "toolchain"
    _copy_tree(toolchain, portable_toolchain)
    copied_toolchain_manifest = validate_scene_configuration_toolchain(
        root=portable_toolchain, expected_source_commit=expected_source_commit
    )
    if copied_toolchain_manifest["toolchain_digest"] != toolchain_manifest["toolchain_digest"]:
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_toolchain_copy_mismatch"
        )
    _copy_tree(repo / "src/blueprint_pipeline", runtime / "blueprint_pipeline")
    _copy_file(repo / "scripts/run_task_evaluation_scene_configuration_provider.sh", stage / ENTRYPOINT)
    _copy_file(repo / "scripts/task_evaluation_scene_configuration_provider_runner.py", stage / RUNNER)
    (stage / ENTRYPOINT).chmod(0o555)
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "status": "ready",
        "provider_bundle_kind": PROVIDER_BUNDLE_KIND,
        "probe_kind": PROBE_KIND,
        "run_id": envelope["run_id"],
        "source_commit": expected_source_commit,
        "construction_envelope_source_digest": envelope["envelope_digest"],
        "portable_construction_envelope_digest": portable["envelope_digest"],
        "toolchain_schema_version": TOOLCHAIN_SCHEMA_VERSION,
        "toolchain_digest": toolchain_manifest["toolchain_digest"],
        "raw_interiorgs_bytes_in_provider_bundle": False,
        "derived_rendered_view_count": len(portable["render_inputs_result"]["derived_frames"]),
        "single_parent_allocation": True,
        "nested_provider_mutations_performed": 0,
        "evaluation_episode_executed": False,
        "expected_result_filename": RESULT_FILENAME,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    (runtime / f"{BUNDLE_SCHEMA_VERSION}.json").write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )
    bundle = output / "task_evaluation_scene_configuration_provider_bundle.zip"
    _zip_tree(stage, bundle)
    receipt = {
        **manifest,
        "bundle_path": str(bundle),
        "bundle_sha256": _sha256(bundle),
        "bundle_size_bytes": bundle.stat().st_size,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    (output / f"{BUNDLE_SCHEMA_VERSION}.receipt.json").write_text(
        canonical_json(receipt) + "\n", encoding="utf-8"
    )
    return receipt


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "ENTRYPOINT",
    "PROBE_KIND",
    "PROVIDER_BUNDLE_KIND",
    "RESULT_FILENAME",
    "TaskEvaluationSceneConfigurationBundleError",
    "build_scene_configuration_provider_bundle",
]
