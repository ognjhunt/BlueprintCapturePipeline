"""Build the portable single-allocation scene-configuration provider bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_scene_configuration_disclosure import (
    PENDING_PROVIDER_RENDER_STATUS,
    RENDER_INPUT_STATUSES,
    render_inputs_disclosure_is_coherent,
    renders_on_provider,
)
from .task_evaluation_scene_configuration_diagnostic_checkpoint import (
    SCHEMA_VERSION as DIAGNOSTIC_CHECKPOINT_SCHEMA_VERSION,
    diagnostic_checkpoint_scientific_binding_digest,
    validate_scene_configuration_diagnostic_checkpoint,
)
from .task_evaluation_scene_configuration_diagnostic_mode import (
    CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE,
    FRESH_DIAGNOSTIC_BOOTSTRAP_MODE,
)
from .task_evaluation_scene_configuration_python_wheelhouse import (
    MANIFEST_NAME as PYTHON_WHEELHOUSE_MANIFEST_NAME,
    validate_scene_configuration_python_wheelhouse,
)
from .task_evaluation_scene_configuration_stage_configuration import (
    TaskEvaluationSceneConfigurationStageConfigurationError,
    validate_immutable_stage_configurations,
)
from .task_evaluation_scene_configuration_source_preflight import (
    TaskEvaluationSceneConfigurationSourcePreflightError,
    validate_scene_configuration_source_preflight,
)
from .task_evaluation_scene_configuration_builtin_producers import (
    TOOLCHAIN_SCHEMA_VERSION,
    validate_scene_configuration_toolchain,
)
from .task_evaluation_scene_construction_queue import ENVELOPE_SCHEMA_VERSION
from .task_evaluation_splat_render_runtime import (
    DEFAULT_ENVIRONMENT_VARIABLE as SPLAT_RENDER_RUNTIME_ROOT_ENV,
    PROVIDER_RENDERER_REQUIRED_PACKAGES,
    PROVIDER_RENDERER_SCHEMA_VERSION,
    TaskEvaluationSplatRenderRuntimeError,
    validate_diagnostic_splat_render_runtime,
    validate_splat_render_runtime,
)


PROBE_KIND = "task-evaluation-scene-configuration"
PROVIDER_BUNDLE_KIND = "task_evaluation_scene_configuration"
BUNDLE_SCHEMA_VERSION = "task_evaluation_scene_configuration_provider_bundle.v1"
ENTRYPOINT = "provider_runtime/run_task_evaluation_scene_configuration_provider.sh"
RUNNER = "provider_runtime/task_evaluation_scene_configuration_provider_runner.py"
RESULT_FILENAME = "task_evaluation_scene_configuration_provider_result.v1.json"
DIAGNOSTIC_RUNNER = (
    "scripts/task_evaluation_scene_configuration_diagnostic_provider_runner.py"
)
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_MEMBER_BYTES = 2 * 1024**3
_MAX_TOTAL_BYTES = 8 * 1024**3
_PROVIDER_RENDERER_FILES = (
    "tools/splat_render/render_splat.mjs",
    "tools/splat_render/src/render_entry.mjs",
    "tools/splat_render/harness.html",
    "tools/splat_render/package.json",
    "tools/splat_render/package-lock.json",
)
_PROVIDER_PYTHON_WHEELHOUSE_RELATIVE = Path(
    "components/artifixer3d_observed_object_removal/package/python_wheelhouse"
)
_COLLISION_REFERENCE_CONTRACT_PATH = "scene.geometry.collision"
_OPENUSD_REFERENCE_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})


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


def _provider_reference_suffix(row: Mapping[str, Any], *, source: Path) -> str:
    """Preserve the declared OpenUSD format when CAS materialization hid it.

    Prepared references are commonly materialized under a digest-only filename.
    OpenUSD selects its crate/text file format from the filename extension, so
    copying a valid collision layer with ``source.suffix == ""`` makes the exact
    same bytes unreadable on the provider.  The immutable reference URI already
    declares the publisher's format; use that bound declaration rather than
    guessing from bytes or from a mutable local path.
    """

    if row.get("contract_path") != _COLLISION_REFERENCE_CONTRACT_PATH:
        return source.suffix
    try:
        suffix = Path(urlsplit(str(row.get("uri") or "")).path).suffix.lower()
    except ValueError as exc:
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_collision_reference_format_invalid"
        ) from exc
    if suffix not in _OPENUSD_REFERENCE_SUFFIXES:
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_collision_reference_format_invalid"
        )
    return suffix


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


def _stage_provider_renderer(
    *,
    runtime: Path,
    repo: Path,
    source_runtime_root: Path,
    validated_runtime: Mapping[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    """Stage only the renderer, lockfile vendors, and Chromium needed remotely."""

    destination = runtime / "renderer"
    for relative in _PROVIDER_RENDERER_FILES:
        target = destination / relative
        _copy_file(repo / relative, target)
        target.chmod(0o444)
    source_renderer = Path(str(validated_runtime["renderer_root"])).resolve()
    for package in PROVIDER_RENDERER_REQUIRED_PACKAGES:
        _copy_tree(
            source_renderer / "tools/splat_render/node_modules" / package,
            destination / "tools/splat_render/node_modules" / package,
        )
    source_browser = Path(str(validated_runtime["browser_executable"])).resolve()
    source_node = Path(str(validated_runtime["node"])).resolve()
    try:
        source_node.relative_to(source_runtime_root)
    except ValueError as exc:
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_provider_renderer_node_invalid"
        ) from exc
    bundled_node = destination / "node/bin/node"
    _copy_file(source_node, bundled_node)
    bundled_node.chmod(0o555)
    browser_root = source_runtime_root / "browser"
    try:
        browser_relative = source_browser.relative_to(browser_root)
    except ValueError as exc:
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_provider_renderer_browser_invalid"
        ) from exc
    _copy_tree(browser_root, destination / "browser")
    files = [
        {
            "relative_path": path.relative_to(destination).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "executable": bool(path.stat().st_mode & 0o111),
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    ]
    manifest: dict[str, Any] = {
        "schema_version": PROVIDER_RENDERER_SCHEMA_VERSION,
        "status": "ready_for_provider_render",
        "platform": "linux-x86_64",
        "source_commit": source_commit,
        "source_runtime_digest": validated_runtime["identity"]["runtime_digest"],
        "entrypoints": {
            "node": "node/bin/node",
            "browser": (Path("browser") / browser_relative).as_posix(),
            "renderer_root": ".",
        },
        "files": files,
        "renderer_digest": "",
    }
    manifest["renderer_digest"] = canonical_digest(
        manifest, digest_field="renderer_digest"
    )
    manifest_path = destination / f"{PROVIDER_RENDERER_SCHEMA_VERSION}.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    manifest_path.chmod(0o444)
    for directory in sorted(
        (path for path in destination.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    destination.chmod(0o555)
    return manifest


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
    *, runtime: Path, render: Mapping[str, Any], appearance: Path | None = None
) -> dict[str, Any]:
    portable = json.loads(json.dumps(dict(render)))
    source_result_digest = str(portable.get("result_digest") or "")
    calibration = _bound_file(render["camera_calibration"], code="scene_configuration_render_input_invalid")
    calibration_target = runtime / "input/render/cameras.json"
    _copy_file(calibration, calibration_target)
    portable["camera_calibration"].pop("materialized_path", None)
    portable["camera_calibration"]["path"] = calibration_target.relative_to(runtime).as_posix()
    # A packet whose render is still owed by the provider has no manifest yet.
    if isinstance(render.get("render_manifest"), Mapping):
        manifest = _bound_file(
            render["render_manifest"], code="scene_configuration_render_input_invalid"
        )
        manifest_target = runtime / "input/render/render_manifest.json"
        _copy_file(manifest, manifest_target)
        portable["render_manifest"].pop("materialized_path", None)
        portable["render_manifest"]["path"] = manifest_target.relative_to(
            runtime
        ).as_posix()
    if appearance is not None:
        appearance_target = (
            runtime / "input/render" / f"source_appearance{appearance.suffix}"
        )
        _copy_file(appearance, appearance_target)
        portable["source_appearance"] = {
            **dict(render.get("source_appearance") or {}),
            "path": appearance_target.relative_to(runtime).as_posix(),
        }
    for index, row in enumerate(render.get("derived_frames") or []):
        frame = _bound_file(row, code="scene_configuration_render_input_invalid")
        target = runtime / "input/render/frames" / f"{index:04d}.png"
        _copy_file(frame, target)
        portable["derived_frames"][index].pop("materialized_path", None)
        portable["derived_frames"][index]["path"] = target.relative_to(runtime).as_posix()
        mask = _bound_file(
            row["source_object_mask"],
            code="scene_configuration_render_mask_input_invalid",
        )
        mask_target = runtime / "input/render/masks" / f"{index:04d}.png"
        _copy_file(mask, mask_target)
        portable_mask = portable["derived_frames"][index]["source_object_mask"]
        portable_mask.pop("materialized_path", None)
        portable_mask["path"] = mask_target.relative_to(runtime).as_posix()
    cutout = render.get("derived_gaussian_cutout")
    if not isinstance(cutout, Mapping):
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_render_gaussian_cutout_invalid"
        )
    for key, filename in (
        ("source_object_candidate", "source_object_candidate_gaussians.ply"),
        (
            "retained_scene_without_source_object",
            "retained_scene_gaussians_without_source_object.ply",
        ),
    ):
        source = _bound_file(
            cutout.get(key) or {},
            code="scene_configuration_render_gaussian_cutout_invalid",
        )
        target = runtime / "input/render/gaussians" / filename
        _copy_file(source, target)
        portable_row = portable["derived_gaussian_cutout"][key]
        portable_row.pop("materialized_path", None)
        portable_row["path"] = target.relative_to(runtime).as_posix()
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
            if size > _MAX_MEMBER_BYTES or total > _MAX_TOTAL_BYTES:
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
    destination.chmod(0o644)


def _remove_self_created_stage_tree(stage: Path) -> None:
    """Remove only the builder-owned tree, including its sealed read-only files."""

    for root, directories, files in os.walk(stage, topdown=True, followlinks=False):
        root_path = Path(root)
        if root_path.is_symlink():
            raise OSError("scene bundle stage symlink")
        root_path.chmod(0o700)
        for name in directories:
            child = root_path / name
            if child.is_symlink():
                raise OSError("scene bundle stage symlink")
            child.chmod(0o700)
        for name in files:
            child = root_path / name
            if child.is_symlink():
                raise OSError("scene bundle stage symlink")
            child.chmod(0o600)
    shutil.rmtree(stage)


def _diagnostic_portable_render_inputs(
    *, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    """Point the slim envelope at its sole checkpoint inventory copy."""

    render = json.loads(json.dumps(checkpoint["render_inputs_template"]))
    inventory = {
        str(row["role"]): row for row in checkpoint["inventory"]
    }

    def bind(row: dict[str, Any], *, required: bool = True) -> None:
        role = row.pop("checkpoint_role", None)
        if role is None and not required:
            row.pop("path", None)
            return
        source = inventory.get(str(role or ""))
        if source is None:
            raise TaskEvaluationSceneConfigurationBundleError(
                "scene_configuration_bundle_diagnostic_checkpoint_role_missing"
            )
        row["path"] = (
            "input/diagnostic_checkpoint/" + str(source["relative_path"])
        )
        row["digest"] = source["digest"]
        row["size_bytes"] = source["size_bytes"]

    bind(render["source_appearance"], required=False)
    bind(render["camera_calibration"])
    bind(render["render_manifest"])
    for frame in render["derived_frames"]:
        bind(frame)
        bind(frame["source_object_mask"])
    cutout = render["derived_gaussian_cutout"]
    bind(cutout["retained_scene_without_source_object"])
    candidate = cutout.get("source_object_candidate")
    if isinstance(candidate, dict):
        bind(candidate, required=False)
        if "path" not in candidate:
            cutout.pop("source_object_candidate", None)
    render["diagnostic_checkpoint_reused"] = True
    render["provider_render_skipped"] = True
    # This field describes bytes in the *current* provider packet, not the
    # historical checkpoint's original render site. Diagnostic retries carry
    # only digest-bound derived frames, so retaining True would make the slim
    # envelope contradict its own inventory and fail the disclosure gate.
    render["raw_interiorgs_bytes_in_provider_packet"] = False
    render["result_digest"] = canonical_digest(
        render, digest_field="result_digest"
    )
    return render


def _resolve_diagnostic_checkpoint_reference(path: str | Path) -> Path:
    reference = _read(
        Path(path).expanduser().resolve(),
        code="scene_configuration_bundle_diagnostic_checkpoint_reference_invalid",
    )
    root = Path(str(reference.get("checkpoint_root") or "")).expanduser().resolve()
    manifest = Path(str(reference.get("manifest_path") or "")).expanduser().resolve()
    if (
        reference.get("schema_version")
        != "task_evaluation_scene_configuration_advanced_checkpoint_reference.v1"
        or reference.get("status")
        != "validated_diagnostic_checkpoint_ready_for_next_retry"
        or reference.get("diagnostic_only") is not True
        or reference.get("qualification_eligible") is not False
        or reference.get("reference_digest")
        != canonical_digest(reference, digest_field="reference_digest")
        or manifest != root / f"{DIAGNOSTIC_CHECKPOINT_SCHEMA_VERSION}.json"
        or not root.is_dir()
        or manifest.is_symlink()
        or not manifest.is_file()
        or _sha256(manifest) != reference.get("manifest_sha256")
    ):
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_diagnostic_checkpoint_reference_invalid"
        )
    checkpoint = validate_scene_configuration_diagnostic_checkpoint(
        checkpoint_root=root
    )
    files = [item for item in root.rglob("*") if item.is_file()]
    if (
        checkpoint.get("checkpoint_digest") != reference.get("checkpoint_digest")
        or checkpoint.get("completed_stage_prefix_count")
        != reference.get("completed_stage_prefix_count")
        or len(files) != reference.get("file_count")
        or sum(item.stat().st_size for item in files) != reference.get("total_bytes")
    ):
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_diagnostic_checkpoint_reference_invalid"
        )
    return root


def build_scene_configuration_provider_bundle(
    *,
    construction_envelope_path: str | Path,
    toolchain_root: str | Path,
    repository_root: str | Path,
    splat_render_runtime_root: str | Path | None = None,
    output_root: str | Path,
    expected_source_commit: str,
    diagnostic_checkpoint_root: str | Path | None = None,
    diagnostic_checkpoint_reference_path: str | Path | None = None,
    fresh_diagnostic_bootstrap: bool = False,
) -> dict[str, Any]:
    """Package provider-authorized derived inputs; raw InteriorGS stays local."""

    diagnostic_mode_requested = bool(
        fresh_diagnostic_bootstrap
        or diagnostic_checkpoint_root is not None
        or diagnostic_checkpoint_reference_path is not None
    )
    if (
        diagnostic_checkpoint_root is not None
        and diagnostic_checkpoint_reference_path is not None
    ):
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_diagnostic_checkpoint_source_ambiguous"
        )
    if fresh_diagnostic_bootstrap and (
        diagnostic_checkpoint_root is not None
        or diagnostic_checkpoint_reference_path is not None
    ):
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_diagnostic_bootstrap_source_ambiguous"
        )
    if diagnostic_checkpoint_reference_path is not None:
        diagnostic_checkpoint_root = _resolve_diagnostic_checkpoint_reference(
            diagnostic_checkpoint_reference_path
        )
    if _COMMIT.fullmatch(expected_source_commit) is None:
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_source_commit_invalid"
        )
    envelope_path = Path(construction_envelope_path).resolve()
    envelope = _read(envelope_path, code="scene_configuration_bundle_envelope_invalid")
    render_inputs = envelope.get("render_inputs_result")
    construction_source_commit = str(
        envelope.get("expected_production_commit") or ""
    )
    if (
        envelope.get("schema_version") != ENVELOPE_SCHEMA_VERSION
        or _COMMIT.fullmatch(construction_source_commit) is None
        or (
            not diagnostic_mode_requested
            and construction_source_commit != expected_source_commit
        )
        or envelope.get("envelope_digest")
        != canonical_digest(envelope, digest_field="envelope_digest")
        or not isinstance(render_inputs, Mapping)
        or render_inputs.get("status") not in RENDER_INPUT_STATUSES
        or not render_inputs_disclosure_is_coherent(render_inputs)
        or render_inputs.get("result_digest")
        != canonical_digest(render_inputs, digest_field="result_digest")
    ):
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_envelope_invalid"
        )
    configuration_sources: dict[str, Path] = {}
    configuration_values: dict[str, dict[str, Any]] = {}
    for stage_row, row in zip(
        envelope["recipe"]["stage_sequence"],
        envelope["stage_configuration_references"],
        strict=True,
    ):
        stage_id = str(stage_row["stage_id"])
        source = _bound_file(
            row, code="scene_configuration_bundle_configuration_invalid"
        )
        configuration_sources[stage_id] = source
        configuration_values[stage_id] = _read(
            source, code="scene_configuration_bundle_configuration_invalid"
        )
    try:
        validate_immutable_stage_configurations(
            envelope=envelope, configurations=configuration_values
        )
    except TaskEvaluationSceneConfigurationStageConfigurationError as exc:
        raise TaskEvaluationSceneConfigurationBundleError(str(exc)) from exc
    try:
        validate_scene_configuration_source_preflight(
            envelope=envelope, configurations=configuration_values
        )
    except TaskEvaluationSceneConfigurationSourcePreflightError as exc:
        raise TaskEvaluationSceneConfigurationBundleError(str(exc)) from exc
    repo = Path(repository_root).resolve()
    toolchain = Path(toolchain_root).resolve()
    toolchain_source_commit = (
        expected_source_commit
        if diagnostic_mode_requested
        else construction_source_commit
    )
    toolchain_manifest = validate_scene_configuration_toolchain(
        root=toolchain, expected_source_commit=toolchain_source_commit
    )
    try:
        provider_python_runtime = validate_scene_configuration_python_wheelhouse(
            root=toolchain / _PROVIDER_PYTHON_WHEELHOUSE_RELATIVE
        )
    except (OSError, ValueError) as exc:
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_provider_python_runtime_invalid"
        ) from exc
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_output_not_empty"
        )
    output.mkdir(parents=True, exist_ok=True)
    stage = output / "stage"
    runtime = stage / "provider_runtime"
    runtime.mkdir(parents=True)
    diagnostic_checkpoint: dict[str, Any] | None = None
    diagnostic_first_configuration: dict[str, Any] | None = None
    diagnostic_first_configuration_path: Path | None = None
    if diagnostic_checkpoint_root is not None:
        first_stage_id = str(envelope["recipe"]["stage_sequence"][0]["stage_id"])
        first_configuration_path = configuration_sources[first_stage_id]
        diagnostic_first_configuration = configuration_values[first_stage_id]
        diagnostic_first_configuration_path = first_configuration_path
        diagnostic_checkpoint = validate_scene_configuration_diagnostic_checkpoint(
            checkpoint_root=diagnostic_checkpoint_root
        )
    diagnostic_mode = diagnostic_checkpoint is not None or fresh_diagnostic_bootstrap
    diagnostic_bootstrap_mode = (
        FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
        if fresh_diagnostic_bootstrap
        else (
            CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE
            if diagnostic_checkpoint is not None
            else None
        )
    )
    # The render-inputs result carries the digest-bound decision that says
    # whether source appearance bytes may cross to the provider. The bundle
    # honours that decision rather than making a second, divergent one.
    source_renders_on_provider = renders_on_provider(
        render_inputs.get("disclosure_decision") or {}
    )
    provider_render = diagnostic_checkpoint is None and source_renders_on_provider
    provider_render_runtime: dict[str, Any] | None = None
    provider_render_runtime_source: Path | None = None
    if source_renders_on_provider:
        unresolved_runtime = str(splat_render_runtime_root or "").strip()
        if not unresolved_runtime:
            raise TaskEvaluationSceneConfigurationBundleError(
                "scene_configuration_bundle_provider_render_runtime_missing"
            )
        provider_render_runtime_source = Path(unresolved_runtime).expanduser()
        try:
            if diagnostic_mode_requested and (
                construction_source_commit != expected_source_commit
            ):
                provider_render_runtime = (
                    validate_diagnostic_splat_render_runtime(
                        runtime_root=provider_render_runtime_source,
                        repo_root=repo,
                        expected_runtime_source_commit=construction_source_commit,
                    )
                )
            else:
                provider_render_runtime = validate_splat_render_runtime(
                    runtime_root=provider_render_runtime_source,
                    repo_root=repo,
                )
        except (OSError, TaskEvaluationSplatRenderRuntimeError) as exc:
            raise TaskEvaluationSceneConfigurationBundleError(
                "scene_configuration_bundle_provider_render_runtime_invalid"
            ) from exc
    diagnostic_scientific_binding_digest: str | None = None
    if diagnostic_checkpoint is not None:
        if (
            provider_render_runtime is None
            or diagnostic_first_configuration is None
            or diagnostic_first_configuration_path is None
        ):
            raise TaskEvaluationSceneConfigurationBundleError(
                "scene_configuration_bundle_diagnostic_renderer_identity_missing"
            )
        current_binding_render = json.loads(
            json.dumps(diagnostic_checkpoint["render_inputs_template"])
        )
        for key in (
            "source_splat_digest",
            "source_appearance",
            "camera_calibration",
            "disclosure_decision",
        ):
            current_binding_render[key] = json.loads(json.dumps(render_inputs[key]))
        checkpoint_renderer = current_binding_render.get("renderer_runtime")
        retained_host_runtime = provider_render_runtime.get("identity")
        if (
            not isinstance(checkpoint_renderer, Mapping)
            or not isinstance(retained_host_runtime, Mapping)
            or checkpoint_renderer.get("mode")
            != "digest_bound_provider_bundle_renderer"
            or checkpoint_renderer.get("schema_version")
            != PROVIDER_RENDERER_SCHEMA_VERSION
            or checkpoint_renderer.get("provider_full_byte_inventory_reopened")
            is not True
            or checkpoint_renderer.get("source_runtime_digest")
            != retained_host_runtime.get("runtime_digest")
            or checkpoint_renderer.get("platform")
            != retained_host_runtime.get("platform")
            or not isinstance(checkpoint_renderer.get("file_count"), int)
            or isinstance(checkpoint_renderer.get("file_count"), bool)
            or checkpoint_renderer.get("file_count", 0) <= 0
            or not isinstance(retained_host_runtime.get("file_count"), int)
            or isinstance(retained_host_runtime.get("file_count"), bool)
            or retained_host_runtime.get("file_count", 0) <= 0
        ):
            raise TaskEvaluationSceneConfigurationBundleError(
                "scene_configuration_bundle_diagnostic_checkpoint_binding_mismatch"
            )
        # The checkpoint binds the provider-reopened renderer receipt, while a
        # retry can only reopen its immutable host source runtime.  The runtime
        # digest, platform, valid scoped inventories, and the validator's
        # full-byte comparison
        # prove those are the same renderer.  Preserve the checkpoint receipt
        # here so an executable-only source commit does not rewrite scientific
        # history into a different receipt schema.
        expected_binding = diagnostic_checkpoint_scientific_binding_digest(
            stage_input={
                "stage": dict(envelope["recipe"]["stage_sequence"][0]),
                "configuration": diagnostic_first_configuration,
                "configuration_sha256": _sha256(
                    diagnostic_first_configuration_path
                ),
                "construction_envelope": envelope,
            },
            render_inputs=current_binding_render,
        )
        if expected_binding != diagnostic_checkpoint["scientific_bindings"][
            "binding_digest"
        ]:
            raise TaskEvaluationSceneConfigurationBundleError(
                "scene_configuration_bundle_diagnostic_checkpoint_binding_mismatch"
            )
        diagnostic_scientific_binding_digest = expected_binding
    if fresh_diagnostic_bootstrap:
        if (
            provider_render_runtime is None
            or not envelope["recipe"]["stage_sequence"]
        ):
            raise TaskEvaluationSceneConfigurationBundleError(
                "scene_configuration_bundle_fresh_diagnostic_identity_missing"
            )
        first_stage = dict(envelope["recipe"]["stage_sequence"][0])
        first_stage_id = str(first_stage["stage_id"])
        current_binding_render = json.loads(json.dumps(render_inputs))
        current_binding_render["renderer_runtime"] = dict(
            provider_render_runtime["identity"]
        )
        diagnostic_scientific_binding_digest = (
            diagnostic_checkpoint_scientific_binding_digest(
                stage_input={
                    "stage": first_stage,
                    "configuration": configuration_values[first_stage_id],
                    "configuration_sha256": _sha256(
                        configuration_sources[first_stage_id]
                    ),
                    "construction_envelope": envelope,
                },
                render_inputs=current_binding_render,
            )
        )
    portable = json.loads(json.dumps(envelope))
    portable["control_plane_envelope_digest"] = envelope["envelope_digest"]
    portable_refs = []
    for index, row in enumerate(envelope.get("materialized_references") or []):
        contract_path = str(row.get("contract_path") or "")
        if contract_path == "scene.appearance.representation":
            # Staged once under input/render when the decision admits it.
            continue
        source = _bound_file(row, code="scene_configuration_bundle_reference_invalid")
        suffix = _provider_reference_suffix(row, source=source)
        target = runtime / "input/references" / f"{index:04d}{suffix}"
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
        stage_id = str(stage_row["stage_id"])
        source = configuration_sources[stage_id]
        target = runtime / "input/configurations" / f"{stage_id}.json"
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
    appearance_source: Path | None = None
    if provider_render:
        appearance_row = next(
            (
                row
                for row in envelope.get("materialized_references") or []
                if row.get("contract_path") == "scene.appearance.representation"
            ),
            None,
        )
        if appearance_row is None:
            raise TaskEvaluationSceneConfigurationBundleError(
                "scene_configuration_bundle_source_appearance_missing"
            )
        appearance_source = _bound_file(
            appearance_row, code="scene_configuration_bundle_reference_invalid"
        )
    if diagnostic_checkpoint is not None:
        _copy_tree(
            Path(diagnostic_checkpoint_root).expanduser().resolve(),
            runtime / "input/diagnostic_checkpoint",
        )
        portable["render_inputs_result"] = _diagnostic_portable_render_inputs(
            checkpoint=diagnostic_checkpoint
        )
    else:
        portable["render_inputs_result"] = _portable_render_inputs(
            runtime=runtime, render=render_inputs, appearance=appearance_source
        )
    raw_paths = [
        row for row in envelope.get("materialized_references") or []
        if row.get("contract_path") == "scene.appearance.representation"
    ]
    portable["provider_disclosure_receipt"] = {
        "raw_interiorgs_reference_count_omitted": 0 if provider_render else len(raw_paths),
        "raw_interiorgs_bytes_in_provider_bundle": provider_render,
        "derived_rendered_views_in_provider_bundle": not provider_render,
        "render_execution_site": render_inputs.get("render_execution_site")
        or "control_plane",
        "disclosure_decision_digest": (
            (render_inputs.get("disclosure_decision") or {}).get("decision_digest")
        ),
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
        root=portable_toolchain, expected_source_commit=toolchain_source_commit
    )
    if copied_toolchain_manifest["toolchain_digest"] != toolchain_manifest["toolchain_digest"]:
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_toolchain_copy_mismatch"
        )
    copied_python_runtime = validate_scene_configuration_python_wheelhouse(
        root=portable_toolchain / _PROVIDER_PYTHON_WHEELHOUSE_RELATIVE
    )
    if (
        copied_python_runtime["manifest_digest"]
        != provider_python_runtime["manifest_digest"]
    ):
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_provider_python_runtime_copy_mismatch"
        )
    provider_renderer: dict[str, Any] | None = None
    if provider_render and provider_render_runtime_source is not None:
        provider_renderer = _stage_provider_renderer(
            runtime=runtime,
            repo=repo,
            source_runtime_root=provider_render_runtime_source.resolve(),
            validated_runtime=provider_render_runtime,
            source_commit=expected_source_commit,
        )
    _copy_tree(repo / "src/blueprint_pipeline", runtime / "blueprint_pipeline")
    _copy_file(repo / "scripts/run_task_evaluation_scene_configuration_provider.sh", stage / ENTRYPOINT)
    if diagnostic_mode:
        runner_source = repo / DIAGNOSTIC_RUNNER
    else:
        runner_source = (
            repo / "scripts/task_evaluation_scene_configuration_provider_runner.py"
        )
    _copy_file(runner_source, stage / RUNNER)
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
        "toolchain_source_commit": toolchain_source_commit,
        "toolchain_digest": toolchain_manifest["toolchain_digest"],
        "provider_python_runtime_required": True,
        "provider_python_runtime_manifest": (
            "toolchain/"
            + _PROVIDER_PYTHON_WHEELHOUSE_RELATIVE.as_posix()
            + "/"
            + PYTHON_WHEELHOUSE_MANIFEST_NAME
        ),
        "provider_python_runtime_digest": provider_python_runtime[
            "manifest_digest"
        ],
        "provider_python_runtime_python_version": provider_python_runtime[
            "python_version"
        ],
        # The disclosure receipt beside this manifest already reports the
        # truth. A manifest that hardcodes False asserts no source bytes
        # crossed on a run where they did, which is a provenance falsehood,
        # not a safety guarantee -- the guarantee is the digest-bound
        # decision that authorized the crossing.
        "raw_interiorgs_bytes_in_provider_bundle": provider_render,
        # A receipt claiming the source bytes crossed must carry the
        # digest-bound decision that permitted it, or the claim cannot be
        # checked anywhere the receipt travels. The receipt is built from
        # this manifest, so recording it once covers both.
        "disclosure_decision": (
            portable["render_inputs_result"].get("disclosure_decision") or {}
        ),
        "derived_rendered_view_count": len(portable["render_inputs_result"]["derived_frames"]),
        "single_parent_allocation": True,
        "nested_provider_mutations_performed": 0,
        "evaluation_episode_executed": False,
        "expected_result_filename": RESULT_FILENAME,
        "manifest_digest": "",
    }
    if diagnostic_mode:
        manifest.update(
            {
                "construction_source_commit": construction_source_commit,
                "source_diagnostic_checkpoint_digest": (
                    diagnostic_checkpoint["checkpoint_digest"]
                    if diagnostic_checkpoint is not None
                    else None
                ),
                "carried_completed_stage_count": (
                    diagnostic_checkpoint["completed_stage_prefix_count"]
                    if diagnostic_checkpoint is not None
                    else 0
                ),
                "diagnostic_bootstrap_mode": diagnostic_bootstrap_mode,
                "diagnostic_scientific_binding_digest": (
                    diagnostic_scientific_binding_digest
                ),
                "diagnostic_stage_sequence_ids": [
                    str(row["stage_id"])
                    for row in envelope["recipe"]["stage_sequence"]
                ],
                "normal_production_lane_used": False,
                "diagnostic_only": True,
                "qualification_eligible": False,
                "executed_inside_one_parent_provider_run": False,
                "configured_revision_publication_permitted": False,
                "offering_publication_permitted": False,
                "terminal_e2e_completion_permitted": False,
            }
        )
    if provider_renderer is not None:
        manifest.update(
            {
                "provider_renderer_required": True,
                "provider_renderer_digest": provider_renderer["renderer_digest"],
                "provider_renderer_source_runtime_digest": provider_renderer[
                    "source_runtime_digest"
                ],
            }
        )
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    (runtime / f"{BUNDLE_SCHEMA_VERSION}.json").write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )
    bundle = output / "task_evaluation_scene_configuration_provider_bundle.zip"
    _zip_tree(stage, bundle)
    # ``stage`` is a self-created expansion tree used only to construct the
    # immutable ZIP. Keeping a second 1.5 GB copy per activation filled the
    # production host even though every allocator consumes ``bundle_path``.
    # Remove it before sealing the receipt; the ZIP's digest and internal
    # manifest remain the sole portable provider input.
    try:
        _remove_self_created_stage_tree(stage)
    except OSError as exc:
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_stage_cleanup_failed"
        ) from exc
    if stage.exists() or stage.is_symlink():
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_stage_cleanup_failed"
        )
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


def _receipt_disclosure_is_coherent(receipt: Mapping[str, Any]) -> bool:
    """True when the receipt's raw-bytes claim matches its authorizing decision.

    The old literal ``raw_interiorgs_bytes_in_provider_bundle is False`` is
    preserved for every scene whose rights do not admit the upload. A receipt
    that claims the bytes crossed must additionally carry the digest-bound
    decision that permitted it, so this narrows what a bundle may claim
    rather than widening it.
    """

    crossed = receipt.get("raw_interiorgs_bytes_in_provider_bundle")
    decision = (receipt.get("provider_disclosure_receipt") or {}).get(
        "disclosure_decision"
    ) or receipt.get("disclosure_decision")
    if crossed is False:
        return True
    return crossed is True and renders_on_provider(decision or {})


def _provider_renderer_archive_is_valid(
    archive: zipfile.ZipFile, bundle_manifest: Mapping[str, Any]
) -> bool:
    prefix = "provider_runtime/renderer/"
    manifest_member = prefix + f"{PROVIDER_RENDERER_SCHEMA_VERSION}.json"
    members = {
        info.filename: info
        for info in archive.infolist()
        if info.filename.startswith(prefix) and not info.is_dir()
    }
    required = bundle_manifest.get("provider_renderer_required") is True
    if not required:
        return not members and not any(
            key in bundle_manifest
            for key in (
                "provider_renderer_required",
                "provider_renderer_digest",
                "provider_renderer_source_runtime_digest",
            )
        )
    try:
        renderer_value = json.loads(archive.read(manifest_member).decode("utf-8"))
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(renderer_value, Mapping):
        return False
    renderer = dict(renderer_value)
    rows = renderer.get("files")
    if (
        renderer.get("schema_version") != PROVIDER_RENDERER_SCHEMA_VERSION
        or renderer.get("status") != "ready_for_provider_render"
        or renderer.get("platform") != "linux-x86_64"
        or renderer.get("source_commit") != bundle_manifest.get("source_commit")
        or renderer.get("source_runtime_digest")
        != bundle_manifest.get("provider_renderer_source_runtime_digest")
        or renderer.get("renderer_digest")
        != bundle_manifest.get("provider_renderer_digest")
        or renderer.get("renderer_digest")
        != canonical_digest(renderer, digest_field="renderer_digest")
        or not isinstance(rows, list)
        or not rows
    ):
        return False
    expected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        relative = str(row.get("relative_path") or "")
        if (
            not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in expected
            or not isinstance(row.get("size_bytes"), int)
            or isinstance(row.get("size_bytes"), bool)
            or row.get("size_bytes") < 0
            or not isinstance(row.get("executable"), bool)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(row.get("sha256") or ""))
            is None
        ):
            return False
        expected[relative] = row
    if set(members) != {
        manifest_member,
        *(prefix + relative for relative in expected),
    }:
        return False
    for relative, row in expected.items():
        member = prefix + relative
        info = members[member]
        body = archive.read(member)
        archived_mode = info.external_attr >> 16
        if (
            len(body) != row["size_bytes"]
            or "sha256:" + hashlib.sha256(body).hexdigest() != row["sha256"]
            or not stat.S_ISREG(archived_mode)
            or bool(archived_mode & 0o111) != row["executable"]
        ):
            return False
    entrypoints = renderer.get("entrypoints")
    node = str(
        entrypoints.get("node") if isinstance(entrypoints, Mapping) else ""
    )
    browser = str(
        entrypoints.get("browser") if isinstance(entrypoints, Mapping) else ""
    )
    if any(
        relative not in expected or expected[relative].get("executable") is not True
        for relative in (node, browser)
    ):
        return False
    if any(relative not in expected for relative in _PROVIDER_RENDERER_FILES):
        return False
    return all(
        any(
            relative.startswith(
                f"tools/splat_render/node_modules/{package}/"
            )
            for relative in expected
        )
        for package in PROVIDER_RENDERER_REQUIRED_PACKAGES
    )


def load_scene_configuration_provider_bundle_receipt(
    path: str | Path,
    *,
    expected_source_commit: str | None = None,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    """Reopen the exact portable bundle and its immutable internal manifest."""

    receipt_path = Path(path).expanduser().resolve()
    receipt = _read(
        receipt_path, code="scene_configuration_bundle_receipt_invalid"
    )
    bundle = Path(str(receipt.get("bundle_path") or "")).expanduser().resolve()
    errors: list[str] = []
    if (
        receipt.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or receipt.get("status") != "ready"
        or receipt.get("provider_bundle_kind") != PROVIDER_BUNDLE_KIND
        or receipt.get("probe_kind") != PROBE_KIND
        or not _receipt_disclosure_is_coherent(receipt)
        or receipt.get("single_parent_allocation") is not True
        or receipt.get("nested_provider_mutations_performed") != 0
        or receipt.get("evaluation_episode_executed") is not False
        or (
            diagnostic_only
            and (
                receipt.get("diagnostic_only") is not True
                or receipt.get("qualification_eligible") is not False
                or receipt.get("executed_inside_one_parent_provider_run") is not False
                or receipt.get("configured_revision_publication_permitted") is not False
                or receipt.get("offering_publication_permitted") is not False
                or receipt.get("terminal_e2e_completion_permitted") is not False
                or _COMMIT.fullmatch(
                    str(receipt.get("construction_source_commit") or "")
                )
                is None
                or _DIGEST.fullmatch(
                    str(
                        receipt.get("diagnostic_scientific_binding_digest")
                        or ""
                    )
                )
                is None
                or not isinstance(
                    receipt.get("diagnostic_stage_sequence_ids"), list
                )
                or len(receipt.get("diagnostic_stage_sequence_ids") or []) != 6
                or len(set(receipt.get("diagnostic_stage_sequence_ids") or []))
                != 6
                or any(
                    not isinstance(stage_id, str) or not stage_id
                    for stage_id in receipt.get("diagnostic_stage_sequence_ids")
                    or []
                )
                or (
                    receipt.get("diagnostic_bootstrap_mode")
                    == FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
                    and (
                        receipt.get("source_diagnostic_checkpoint_digest")
                        is not None
                        or receipt.get("carried_completed_stage_count") != 0
                    )
                )
                or (
                    receipt.get("diagnostic_bootstrap_mode")
                    != FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
                    and (
                        receipt.get("diagnostic_bootstrap_mode")
                        != CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE
                        or _DIGEST.fullmatch(
                            str(
                                receipt.get(
                                    "source_diagnostic_checkpoint_digest"
                                )
                                or ""
                            )
                        )
                        is None
                    )
                )
            )
        )
        or (
            not diagnostic_only
            and any(
                key in receipt
                for key in (
                    "diagnostic_only",
                    "qualification_eligible",
                    "configured_revision_publication_permitted",
                    "offering_publication_permitted",
                    "terminal_e2e_completion_permitted",
                    "diagnostic_bootstrap_mode",
                    "diagnostic_scientific_binding_digest",
                    "diagnostic_stage_sequence_ids",
                    "construction_source_commit",
                )
            )
        )
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
        or (
            expected_source_commit is not None
            and receipt.get("source_commit") != expected_source_commit
        )
    ):
        errors.append("receipt_contract_invalid")
    internal: dict[str, Any] = {}
    if (
        bundle.is_symlink()
        or not bundle.is_file()
        or bundle.stat().st_size != receipt.get("bundle_size_bytes")
        or _sha256(bundle) != receipt.get("bundle_sha256")
    ):
        errors.append("bundle_bytes_invalid")
    else:
        try:
            with zipfile.ZipFile(bundle) as archive:
                internal_value = json.loads(
                    archive.read(
                        f"provider_runtime/{BUNDLE_SCHEMA_VERSION}.json"
                    ).decode("utf-8")
                )
                provider_renderer_valid = _provider_renderer_archive_is_valid(
                    archive,
                    internal_value if isinstance(internal_value, Mapping) else {},
                )
                if diagnostic_only:
                    names = {
                        row.filename
                        for row in archive.infolist()
                        if not row.is_dir()
                    }
                    portable_value = json.loads(
                        archive.read(
                            "provider_runtime/input/portable_construction_envelope.v1.json"
                        ).decode("utf-8")
                    )
                    diagnostic_render = (
                        portable_value.get("render_inputs_result")
                        if isinstance(portable_value, Mapping)
                        else None
                    )
                    source_appearance = (
                        diagnostic_render.get("source_appearance")
                        if isinstance(diagnostic_render, Mapping)
                        else None
                    )
                    if (
                        receipt.get("diagnostic_bootstrap_mode")
                        == FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
                    ):
                        diagnostic_archive_valid = (
                            isinstance(diagnostic_render, Mapping)
                            and diagnostic_render.get("status")
                            == PENDING_PROVIDER_RENDER_STATUS
                            and diagnostic_render.get("provider_render_required")
                            is True
                            and isinstance(source_appearance, Mapping)
                            and str(source_appearance.get("path") or "").startswith(
                                "input/render/"
                            )
                            and any(
                                name.startswith("provider_runtime/renderer/")
                                for name in names
                            )
                            and any(
                                name.startswith(
                                    "provider_runtime/input/render/source_appearance"
                                )
                                for name in names
                            )
                            and not any(
                                name.startswith(
                                    "provider_runtime/input/diagnostic_checkpoint/"
                                )
                                for name in names
                            )
                        )
                    else:
                        diagnostic_archive_valid = (
                            isinstance(diagnostic_render, Mapping)
                            and diagnostic_render.get(
                                "diagnostic_checkpoint_reused"
                            )
                            is True
                            and diagnostic_render.get("provider_render_skipped")
                            is True
                            and isinstance(source_appearance, Mapping)
                            and "path" not in source_appearance
                            and not any(
                                name.startswith("provider_runtime/renderer/")
                                or name.startswith(
                                    "provider_runtime/input/render/source_appearance"
                                )
                                for name in names
                            )
                            and any(
                                name.startswith(
                                    "provider_runtime/input/diagnostic_checkpoint/semantic/"
                                )
                                for name in names
                            )
                            and all(
                                str(row.get("path") or "").startswith(
                                    "input/diagnostic_checkpoint/"
                                )
                                for row in diagnostic_render.get("derived_frames")
                                or []
                                if isinstance(row, Mapping)
                            )
                        )
                    if not diagnostic_archive_valid:
                        errors.append("bundle_diagnostic_archive_invalid")
            internal = (
                dict(internal_value)
                if isinstance(internal_value, Mapping)
                else {}
            )
            if not provider_renderer_valid:
                errors.append("bundle_provider_renderer_invalid")
        except (
            KeyError,
            OSError,
            UnicodeError,
            ValueError,
            zipfile.BadZipFile,
            json.JSONDecodeError,
        ):
            errors.append("bundle_internal_manifest_invalid")
    compared_fields = (
        "schema_version",
        "status",
        "provider_bundle_kind",
        "probe_kind",
        "run_id",
        "source_commit",
        "construction_envelope_source_digest",
        "portable_construction_envelope_digest",
        "toolchain_digest",
        "provider_python_runtime_required",
        "provider_python_runtime_manifest",
        "provider_python_runtime_digest",
        "provider_python_runtime_python_version",
        "raw_interiorgs_bytes_in_provider_bundle",
        # Cross-compared with everything else it authorizes: without this, a
        # receipt's decision could drift from the one sealed in the bundle,
        # and the byte-crossing claim would be checkable only against itself.
        "disclosure_decision",
        "single_parent_allocation",
        "nested_provider_mutations_performed",
        "evaluation_episode_executed",
        "expected_result_filename",
        "manifest_digest",
    )
    if diagnostic_only:
        compared_fields += (
            "construction_source_commit",
            "source_diagnostic_checkpoint_digest",
            "carried_completed_stage_count",
            "diagnostic_bootstrap_mode",
            "diagnostic_scientific_binding_digest",
            "diagnostic_stage_sequence_ids",
            "normal_production_lane_used",
            "diagnostic_only",
            "qualification_eligible",
            "executed_inside_one_parent_provider_run",
            "configured_revision_publication_permitted",
            "offering_publication_permitted",
            "terminal_e2e_completion_permitted",
        )
    if receipt.get("provider_renderer_required") is True:
        compared_fields += (
            "provider_renderer_required",
            "provider_renderer_digest",
            "provider_renderer_source_runtime_digest",
        )
    if (
        not internal
        or internal.get("manifest_digest")
        != canonical_digest(internal, digest_field="manifest_digest")
        or any(internal.get(field) != receipt.get(field) for field in compared_fields)
    ):
        errors.append("bundle_internal_manifest_invalid")
    if errors:
        raise TaskEvaluationSceneConfigurationBundleError(
            "scene_configuration_bundle_receipt_invalid:"
            + ",".join(sorted(set(errors)))
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
    "load_scene_configuration_provider_bundle_receipt",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-envelope", required=True)
    parser.add_argument("--toolchain-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument(
        "--splat-render-runtime-root",
        default=os.getenv(SPLAT_RENDER_RUNTIME_ROOT_ENV, ""),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--diagnostic-checkpoint-root")
    parser.add_argument("--diagnostic-checkpoint-reference")
    parser.add_argument("--fresh-diagnostic-bootstrap", action="store_true")
    args = parser.parse_args(argv)
    receipt = build_scene_configuration_provider_bundle(
        construction_envelope_path=args.construction_envelope,
        toolchain_root=args.toolchain_root,
        repository_root=args.repository_root,
        splat_render_runtime_root=args.splat_render_runtime_root,
        output_root=args.output_root,
        expected_source_commit=args.expected_source_commit,
        diagnostic_checkpoint_root=args.diagnostic_checkpoint_root,
        diagnostic_checkpoint_reference_path=args.diagnostic_checkpoint_reference,
        fresh_diagnostic_bootstrap=args.fresh_diagnostic_bootstrap,
    )
    print(canonical_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
