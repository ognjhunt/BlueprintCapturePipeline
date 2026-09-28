#!/usr/bin/env python3
"""Stage and run one pushed-branch scene diagnostic retry without a deploy.

The command surface is fixed-purpose: it accepts paths and bounded authority
fields, never an arbitrary command or argv.  It stages source only, reuses the
operator-selected venv, toolchain, checkpoint, and splat-runtime identity by
reference, then invokes the existing bundle, authority, and canonical allocator
entrypoints from the detached release.  ``--execute`` remains the only switch
that can reach paid allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess  # nosec B404 - fixed Python module commands only
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


# systemd EnvironmentFile entries can replace an earlier PYTHONPATH setting.
# Re-establish this script's exact staged source root before importing any
# Blueprint module, then prove below that every loaded Blueprint module came
# from the same release.
_SCRIPT_SOURCE_ROOT = (Path(__file__).resolve().parents[1] / "src").resolve()
_script_source_text = str(_SCRIPT_SOURCE_ROOT)
sys.path[:] = [entry for entry in sys.path if entry != _script_source_text]
sys.path.insert(0, _script_source_text)

from blueprint_pipeline.core.common import redacted_failure_text  # noqa: E402
from blueprint_pipeline.decision_evidence_contracts import canonical_digest  # noqa: E402
from blueprint_pipeline.task_evaluation_scene_configuration_bundle import (  # noqa: E402
    BUNDLE_SCHEMA_VERSION,
    PROBE_KIND,
    TaskEvaluationSceneConfigurationBundleError,
    load_scene_configuration_provider_bundle_receipt,
)
from blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_release import (  # noqa: E402
    SceneConfigurationDiagnosticReleaseError,
    stage_scene_configuration_diagnostic_release,
    validate_scene_configuration_diagnostic_release_receipt,
)
from blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_mode import (  # noqa: E402
    CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE,
    FRESH_DIAGNOSTIC_BOOTSTRAP_MODE,
)
from blueprint_pipeline.task_evaluation_scene_configuration_paid_authority import (  # noqa: E402
    AUTHORITY_SCHEMA_VERSION,
    SCENE_CONFIGURATION_PROVIDER_IMAGE,
)
from blueprint_pipeline.spend_authority_consumption_root import (  # noqa: E402
    SpendAuthorityRootError,
    prepare_consumption_root,
)
from blueprint_pipeline.task_evaluation_scene_configuration_warm_diagnostic import (  # noqa: E402
    materialize_scene_configuration_warm_session_authority,
)


PREPARATION_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_diagnostic_iteration_preparation.v1"
)
CLEANUP_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_diagnostic_bundle_staging_cleanup.v1"
)
CLEANUP_COMMAND = "cleanup-sealed-bundle-staging"
CONTENT_ADDRESS_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_diagnostic_bundle_content_address.v1"
)
CONTENT_ADDRESS_COMMAND = "content-address-sealed-provider-bundle"
CONTENT_ADDRESS_DIRECTORY = "provider-bundle-content-addressed"
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
BundleContentAddresser = Callable[..., Mapping[str, Any]]

_OPENAI_RUNTIME_FILE_ENV_NAMES = (
    "OPENAI_ADMIN_API_KEY_FILE",
    "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE",
    "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE",
    "OPENAI_CONTENT_AGENTS_API_KEY_FILE",
    "BLUEPRINT_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_COST_SCOPE_ATTESTATION_FILE",
    "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE",
    "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE",
)
_OPENAI_RUNTIME_VALUE_ENV_NAMES = (
    "OPENAI_PROJECT_ID",
    "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_ID",
    "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID",
    "OPENAI_CONTENT_AGENTS_API_KEY_ID",
)
_CHILD_FAILURE_DETAIL_MAX_CHARS = 300
_REQUIRED_BLUEPRINT_IMPORTS = frozenset(
    {
        "blueprint_pipeline.core.common",
        "blueprint_pipeline.decision_evidence_contracts",
        "blueprint_pipeline.spend_authority_consumption_root",
        "blueprint_pipeline.task_evaluation_scene_configuration_bundle",
        "blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_mode",
        "blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_release",
        "blueprint_pipeline.task_evaluation_scene_configuration_paid_authority",
        "blueprint_pipeline.task_evaluation_scene_configuration_warm_diagnostic",
    }
)


class SceneConfigurationDiagnosticIterationError(ValueError):
    """The fixed diagnostic iteration command could not be prepared safely."""


def _blueprint_import_provenance_blockers(
    *,
    script_path: Path = Path(__file__),
    loaded_modules: Mapping[str, object] | None = None,
) -> list[str]:
    """Name Blueprint imports that do not come from this script's release."""

    expected_package_root = (
        script_path.resolve().parents[1] / "src" / "blueprint_pipeline"
    ).resolve()
    modules = sys.modules if loaded_modules is None else loaded_modules
    blockers: list[str] = []
    for name in sorted(_REQUIRED_BLUEPRINT_IMPORTS):
        module = modules.get(name)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not module_file:
            blockers.append(name)
            continue
        try:
            resolved = Path(module_file).resolve(strict=True)
        except OSError:
            blockers.append(name)
            continue
        if not resolved.is_relative_to(expected_package_root):
            blockers.append(name)
    return blockers


def _assert_blueprint_import_provenance() -> None:
    blockers = _blueprint_import_provenance_blockers()
    if blockers:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_import_provenance_invalid:"
            + ",".join(blockers)
        )


def _absolute(value: str | Path, *, field: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SceneConfigurationDiagnosticIterationError(
            f"scene_configuration_diagnostic_iteration_{field}_must_be_absolute"
        )
    return path


def _input_file(value: str | Path, *, field: str) -> Path:
    path = _absolute(value, field=field)
    if path.is_symlink() or not path.is_file():
        raise SceneConfigurationDiagnosticIterationError(
            f"scene_configuration_diagnostic_iteration_{field}_invalid"
        )
    return path.resolve()


def _input_directory(value: str | Path, *, field: str) -> Path:
    path = _absolute(value, field=field)
    if path.is_symlink() or not path.is_dir():
        raise SceneConfigurationDiagnosticIterationError(
            f"scene_configuration_diagnostic_iteration_{field}_invalid"
        )
    return path.resolve()


def _python_executable(value: str | Path) -> Path:
    """Validate a Python entrypoint without discarding its venv identity."""

    path = _absolute(value, field="python_executable")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_python_executable_invalid"
        ) from exc
    if not resolved.is_file() or not os.access(path, os.X_OK):
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_python_executable_invalid"
        )
    return path


def _output_path(value: str | Path, *, field: str) -> Path:
    path = _absolute(value, field=field)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise SceneConfigurationDiagnosticIterationError(
            f"scene_configuration_diagnostic_iteration_{field}_invalid"
        )
    if path.parent.exists() and path.parent.is_symlink():
        raise SceneConfigurationDiagnosticIterationError(
            f"scene_configuration_diagnostic_iteration_{field}_parent_invalid"
        )
    return path


def _output_directory(value: str | Path, *, field: str, empty: bool) -> Path:
    path = _absolute(value, field=field)
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise SceneConfigurationDiagnosticIterationError(
            f"scene_configuration_diagnostic_iteration_{field}_invalid"
        )
    if empty and path.exists() and any(path.iterdir()):
        raise SceneConfigurationDiagnosticIterationError(
            f"scene_configuration_diagnostic_iteration_{field}_not_empty"
        )
    return path


def _read_json(path: Path, *, schema: str, code: str) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("unsafe JSON")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SceneConfigurationDiagnosticIterationError(code) from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != schema:
        raise SceneConfigurationDiagnosticIterationError(code)
    return dict(value)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o440,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short diagnostic iteration receipt write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o440)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_content_unreadable"
        ) from exc
    return "sha256:" + digest.hexdigest()


def _validated_diagnostic_bundle(
    bundle_output_root: str | Path,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    bundle_output = _input_directory(
        bundle_output_root, field="sealed_bundle_output_root"
    )
    receipt_path = bundle_output / f"{BUNDLE_SCHEMA_VERSION}.receipt.json"
    expected_bundle = (
        bundle_output / "task_evaluation_scene_configuration_provider_bundle.zip"
    )
    if receipt_path.is_symlink():
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_sealed_bundle_invalid"
        )
    try:
        bundle_receipt = load_scene_configuration_provider_bundle_receipt(
            receipt_path,
            diagnostic_only=True,
        )
    except (OSError, TaskEvaluationSceneConfigurationBundleError) as exc:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_sealed_bundle_invalid"
        ) from exc
    received_bundle = Path(str(bundle_receipt.get("bundle_path") or "")).expanduser()
    try:
        expected_bundle_resolved = expected_bundle.resolve(strict=True)
        received_bundle_resolved = received_bundle.resolve(strict=True)
    except OSError as exc:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_sealed_bundle_invalid"
        ) from exc
    if (
        expected_bundle.is_symlink()
        or received_bundle.is_symlink()
        or not received_bundle.is_absolute()
        or expected_bundle_resolved != received_bundle_resolved
        or expected_bundle_resolved.parent != bundle_output
    ):
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_sealed_bundle_invalid"
        )
    return bundle_output, receipt_path, expected_bundle_resolved, bundle_receipt


def _validate_content_addressed_bundle_file(
    path: Path,
    *,
    expected_digest: str,
    expected_size: int,
    expected_device: int,
    expected_uid: int,
    expected_gid: int,
    expected_mode: int,
) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_content_object_invalid"
        ) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_dev != expected_device
        or info.st_uid != expected_uid
        or info.st_gid != expected_gid
        or stat.S_IMODE(info.st_mode) != expected_mode
        or info.st_size != expected_size
        or _sha256_file(path) != expected_digest
    ):
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_content_object_invalid"
        )
    return info


def content_address_sealed_provider_bundle(
    *, bundle_output_root: str | Path, content_address_root: str | Path
) -> dict[str, Any]:
    """Deduplicate one validated bundle ZIP while preserving its receipt path."""

    bundle_output, receipt_path, bundle, receipt = _validated_diagnostic_bundle(
        bundle_output_root
    )
    before = bundle.lstat()
    mode = stat.S_IMODE(before.st_mode)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_content_source_invalid"
        )
    expected_digest = str(receipt["bundle_sha256"])
    hexadecimal = expected_digest.removeprefix("sha256:")
    if (
        len(hexadecimal) != 64
        or any(character not in "0123456789abcdef" for character in hexadecimal)
        or before.st_size != receipt["bundle_size_bytes"]
        or _sha256_file(bundle) != expected_digest
    ):
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_content_source_invalid"
        )

    root = _absolute(content_address_root, field="bundle_content_address_root")
    try:
        root_parent = root.parent.resolve(strict=True)
    except OSError as exc:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_content_root_invalid"
        ) from exc
    unresolved_root = root_parent / root.name
    if (
        unresolved_root.is_relative_to(bundle_output)
        or bundle_output.is_relative_to(unresolved_root)
        or unresolved_root.is_symlink()
        or (unresolved_root.exists() and not unresolved_root.is_dir())
    ):
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_content_root_invalid"
        )
    try:
        unresolved_root.mkdir(mode=0o750, exist_ok=True)
        root_info = unresolved_root.lstat()
        root_resolved = unresolved_root.resolve(strict=True)
    except OSError as exc:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_content_root_invalid"
        ) from exc
    root_mode = stat.S_IMODE(root_info.st_mode)
    if (
        stat.S_ISLNK(root_info.st_mode)
        or not stat.S_ISDIR(root_info.st_mode)
        or root_info.st_dev != before.st_dev
        or root_info.st_uid != before.st_uid
        or root_info.st_gid != before.st_gid
        or root_mode & (stat.S_IWGRP | stat.S_IWOTH)
        or root_mode & stat.S_IRWXU != stat.S_IRWXU
    ):
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_content_root_invalid"
        )

    object_path = root_resolved / f"{hexadecimal}.zip"
    temporary = bundle.parent / f".{bundle.name}.content-address.tmp"
    object_created = False
    try:
        try:
            os.link(bundle, object_path, follow_symlinks=False)
            object_created = True
        except FileExistsError:
            pass
        object_info = _validate_content_addressed_bundle_file(
            object_path,
            expected_digest=expected_digest,
            expected_size=before.st_size,
            expected_device=before.st_dev,
            expected_uid=before.st_uid,
            expected_gid=before.st_gid,
            expected_mode=mode,
        )
        reused = (before.st_dev, before.st_ino) != (
            object_info.st_dev,
            object_info.st_ino,
        )
        if reused:
            if os.path.lexists(temporary):
                raise SceneConfigurationDiagnosticIterationError(
                    "scene_configuration_diagnostic_iteration_bundle_content_temporary_exists"
                )
            os.link(object_path, temporary, follow_symlinks=False)
            linked = temporary.lstat()
            if (linked.st_dev, linked.st_ino) != (
                object_info.st_dev,
                object_info.st_ino,
            ):
                raise SceneConfigurationDiagnosticIterationError(
                    "scene_configuration_diagnostic_iteration_bundle_content_link_invalid"
                )
            os.replace(temporary, bundle)
    except SceneConfigurationDiagnosticIterationError:
        raise
    except OSError as exc:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_content_link_failed"
        ) from exc
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()

    after = _validate_content_addressed_bundle_file(
        bundle,
        expected_digest=expected_digest,
        expected_size=before.st_size,
        expected_device=before.st_dev,
        expected_uid=before.st_uid,
        expected_gid=before.st_gid,
        expected_mode=mode,
    )
    if (after.st_dev, after.st_ino) != (object_info.st_dev, object_info.st_ino):
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_content_link_invalid"
        )
    _, reopened_receipt_path, reopened_bundle, reopened = (
        _validated_diagnostic_bundle(bundle_output)
    )
    if (
        reopened_receipt_path != receipt_path
        or reopened_bundle != bundle
        or reopened.get("receipt_digest") != receipt.get("receipt_digest")
        or reopened.get("bundle_sha256") != expected_digest
    ):
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_content_readback_invalid"
        )
    return {
        "bundle_path": str(bundle),
        "bundle_sha256": expected_digest,
        "bundle_size_bytes": before.st_size,
        "bundle_mode": mode,
        "bundle_uid": before.st_uid,
        "bundle_gid": before.st_gid,
        "content_object_path": str(object_path),
        "content_object_created": object_created,
        "existing_content_object_reused": reused,
        "path_preserved": True,
        "bytes_preserved": True,
        "mode_preserved": stat.S_IMODE(after.st_mode) == mode,
        "before_inode": before.st_ino,
        "after_inode": after.st_ino,
        "content_object_inode": object_info.st_ino,
        "allocated_bytes_reclaimed": (
            before.st_blocks * 512 if reused and before.st_nlink == 1 else 0
        ),
        "provider_mutations_performed": 0,
    }


def _discard_sealed_bundle_staging_tree_with_evidence(
    bundle_output: Path,
) -> dict[str, Any]:
    """Remove only a contained, self-created expanded tree after sealing."""

    invalid_code = "scene_configuration_diagnostic_iteration_bundle_staging_invalid"
    cleanup_code = (
        "scene_configuration_diagnostic_iteration_bundle_staging_cleanup_failed"
    )
    staging = bundle_output / "stage"
    try:
        bundle_stat = bundle_output.lstat()
        bundle_resolved = bundle_output.resolve(strict=True)
    except OSError as exc:
        raise SceneConfigurationDiagnosticIterationError(invalid_code) from exc
    if stat.S_ISLNK(bundle_stat.st_mode) or not stat.S_ISDIR(bundle_stat.st_mode):
        raise SceneConfigurationDiagnosticIterationError(invalid_code)
    try:
        staging_stat = staging.lstat()
    except FileNotFoundError:
        return {
            "removed": False,
            "staging_path": str(staging),
            "directory_count": 0,
            "file_count": 0,
            "file_bytes": 0,
            "allocated_bytes": 0,
            "root_stat": None,
            "post_cleanup_absent": True,
        }
    except OSError as exc:
        raise SceneConfigurationDiagnosticIterationError(invalid_code) from exc
    if stat.S_ISLNK(staging_stat.st_mode) or not stat.S_ISDIR(staging_stat.st_mode):
        raise SceneConfigurationDiagnosticIterationError(invalid_code)
    try:
        staging_resolved = staging.resolve(strict=True)
    except OSError as exc:
        raise SceneConfigurationDiagnosticIterationError(invalid_code) from exc
    if staging_resolved.parent != bundle_resolved:
        raise SceneConfigurationDiagnosticIterationError(invalid_code)

    directories: list[tuple[Path, os.stat_result]] = []
    file_count = 0
    file_bytes = 0
    allocated_bytes = 0
    pending = [staging]
    try:
        while pending:
            directory = pending.pop()
            directory_stat = directory.lstat()
            if (
                stat.S_ISLNK(directory_stat.st_mode)
                or not stat.S_ISDIR(directory_stat.st_mode)
                or not directory.resolve(strict=True).is_relative_to(staging_resolved)
                or directory_stat.st_dev != staging_stat.st_dev
                or directory_stat.st_uid != os.geteuid()
            ):
                raise SceneConfigurationDiagnosticIterationError(invalid_code)
            directories.append((directory, directory_stat))
            allocated_bytes += directory_stat.st_blocks * 512
            with os.scandir(directory) as entries:
                for entry in entries:
                    entry_path = directory / entry.name
                    entry_stat = entry.stat(follow_symlinks=False)
                    if stat.S_ISLNK(entry_stat.st_mode):
                        raise SceneConfigurationDiagnosticIterationError(invalid_code)
                    if stat.S_ISDIR(entry_stat.st_mode):
                        pending.append(entry_path)
                    elif stat.S_ISREG(entry_stat.st_mode):
                        file_count += 1
                        file_bytes += entry_stat.st_size
                        allocated_bytes += entry_stat.st_blocks * 512
                    else:
                        raise SceneConfigurationDiagnosticIterationError(invalid_code)
    except SceneConfigurationDiagnosticIterationError:
        raise
    except OSError as exc:
        raise SceneConfigurationDiagnosticIterationError(invalid_code) from exc

    try:
        for directory, before in directories:
            mode = stat.S_IMODE(before.st_mode)
            if mode & stat.S_IWUSR:
                continue
            os.chmod(directory, mode | stat.S_IWUSR, follow_symlinks=False)
            after = directory.lstat()
            if (
                stat.S_ISLNK(after.st_mode)
                or not stat.S_ISDIR(after.st_mode)
                or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
                or not stat.S_IMODE(after.st_mode) & stat.S_IWUSR
            ):
                raise SceneConfigurationDiagnosticIterationError(cleanup_code)
        shutil.rmtree(staging)
    except SceneConfigurationDiagnosticIterationError:
        raise
    except OSError as exc:
        raise SceneConfigurationDiagnosticIterationError(cleanup_code) from exc
    if staging.exists() or staging.is_symlink():
        raise SceneConfigurationDiagnosticIterationError(cleanup_code)
    return {
        "removed": True,
        "staging_path": str(staging),
        "directory_count": len(directories),
        "file_count": file_count,
        "file_bytes": file_bytes,
        "allocated_bytes": allocated_bytes,
        "root_stat": {
            "device": staging_stat.st_dev,
            "inode": staging_stat.st_ino,
            "uid": staging_stat.st_uid,
            "gid": staging_stat.st_gid,
            "mode": stat.S_IMODE(staging_stat.st_mode),
        },
        "post_cleanup_absent": True,
    }


def _discard_sealed_bundle_staging_tree(bundle_output: Path) -> bool:
    """Remove the builder staging tree and preserve the boolean run contract."""

    return bool(
        _discard_sealed_bundle_staging_tree_with_evidence(bundle_output)["removed"]
    )


def reconcile_sealed_bundle_staging_tree(
    *, bundle_output_root: str | Path, cleanup_receipt: str | Path
) -> dict[str, Any]:
    """Validate a sealed diagnostic bundle, reclaim staging, and receipt it."""

    try:
        bundle_output, receipt_path, expected_bundle, bundle_receipt = (
            _validated_diagnostic_bundle(bundle_output_root)
        )
    except SceneConfigurationDiagnosticIterationError as exc:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_cleanup_bundle_invalid"
        ) from exc
    cleanup_receipt_path = _output_path(
        cleanup_receipt, field="cleanup_receipt"
    )
    try:
        cleanup_receipt_parent = cleanup_receipt_path.parent.resolve(strict=True)
    except OSError as exc:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_cleanup_receipt_parent_invalid"
        ) from exc
    if cleanup_receipt_parent != bundle_output:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_cleanup_receipt_parent_invalid"
        )
    cleanup = _discard_sealed_bundle_staging_tree_with_evidence(bundle_output)
    result = {
        "schema_version": CLEANUP_SCHEMA_VERSION,
        "status": "completed",
        "bundle_output_root": str(bundle_output),
        "bundle_receipt_path": str(receipt_path),
        "bundle_receipt_digest": bundle_receipt["receipt_digest"],
        "bundle_path": str(expected_bundle),
        "bundle_sha256": bundle_receipt["bundle_sha256"],
        **cleanup,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    _write_exclusive(cleanup_receipt_path, result)
    return result


def reconcile_content_addressed_provider_bundle(
    *,
    bundle_output_root: str | Path,
    content_address_root: str | Path,
    deduplication_receipt: str | Path,
) -> dict[str, Any]:
    """Apply path-preserving bundle deduplication and write its evidence."""

    bundle_output = _input_directory(
        bundle_output_root, field="content_address_bundle_output_root"
    )
    receipt_path = _output_path(
        deduplication_receipt, field="bundle_content_address_receipt"
    )
    try:
        receipt_parent = receipt_path.parent.resolve(strict=True)
    except OSError as exc:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_content_receipt_parent_invalid"
        ) from exc
    if receipt_parent != bundle_output or os.path.lexists(receipt_path):
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_content_receipt_invalid"
        )
    storage = content_address_sealed_provider_bundle(
        bundle_output_root=bundle_output,
        content_address_root=content_address_root,
    )
    result = {
        "schema_version": CONTENT_ADDRESS_SCHEMA_VERSION,
        "status": "completed",
        "bundle_output_root": str(bundle_output),
        **storage,
        "raw_secret_values_recorded": False,
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    _write_exclusive(receipt_path, result)
    return result


def _run_fixed(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    runner: CommandRunner,
    code: str,
) -> None:
    try:
        completed = runner(
            list(argv),
            cwd=str(cwd),
            env=dict(environment),
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SceneConfigurationDiagnosticIterationError(code) from exc
    if completed.returncode != 0:
        # Keep the typed child refusal so the operator does not need to rerun a
        # paid path just to learn its cause.  Credential-shaped text and signed
        # URL query values are removed before the bounded detail is surfaced.
        detail = redacted_failure_text(completed.stderr or completed.stdout)
        detail = " ".join(detail.split())
        if len(detail) > _CHILD_FAILURE_DETAIL_MAX_CHARS:
            detail = detail[:_CHILD_FAILURE_DETAIL_MAX_CHARS] + "..."
        suffix = f":{detail}" if detail else f":exit_{completed.returncode}"
        raise SceneConfigurationDiagnosticIterationError(code + suffix)


def _preflight_paid_runtime_environment(
    environment: Mapping[str, str],
    *,
    execute: bool,
    openai_max_cost_usd: float,
) -> None:
    """Reject a malformed paid launch before bundle work or provider mutation."""

    if not execute or openai_max_cost_usd <= 0:
        return
    missing = [
        name
        for name in (*_OPENAI_RUNTIME_FILE_ENV_NAMES, *_OPENAI_RUNTIME_VALUE_ENV_NAMES)
        if not str(environment.get(name) or "").strip()
    ]
    if missing:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_openai_runtime_environment_missing:"
            + ",".join(missing)
        )
    invalid_files: list[str] = []
    for name in _OPENAI_RUNTIME_FILE_ENV_NAMES:
        path = Path(str(environment[name])).expanduser()
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or not os.access(path, os.R_OK)
        ):
            invalid_files.append(name)
    if invalid_files:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_openai_runtime_file_invalid:"
            + ",".join(invalid_files)
        )


def _preflight_paid_service_identity(*, execute: bool) -> None:
    """Reach the canonical single-use-ledger gate before expensive bundle work."""

    if not execute:
        return
    try:
        prepare_consumption_root()
    except SpendAuthorityRootError as exc:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_spend_identity_invalid:"
            + str(exc)
        ) from exc


def run_scene_configuration_diagnostic_iteration(
    args: argparse.Namespace,
    *,
    runner: CommandRunner = subprocess.run,
    clock: Callable[[], float] = time.monotonic,
    bundle_content_addresser: BundleContentAddresser = (
        content_address_sealed_provider_bundle
    ),
) -> dict[str, Any]:
    """Prepare the source overlay and invoke the fixed diagnostic chain."""

    _assert_blueprint_import_provenance()
    _preflight_paid_runtime_environment(
        os.environ,
        execute=bool(args.execute),
        openai_max_cost_usd=float(args.openai_max_cost_usd),
    )
    _preflight_paid_service_identity(execute=bool(args.execute))

    source_repo = _input_directory(args.source_repo, field="source_repo")
    release_root = _output_directory(
        args.release_root, field="release_root", empty=False
    )
    state_root = _output_directory(args.state_root, field="state_root", empty=False)
    configured_content_root = getattr(args, "bundle_content_address_root", None)
    bundle_content_address_root = (
        _absolute(
            configured_content_root,
            field="bundle_content_address_root",
        )
        if configured_content_root
        else state_root.parent / CONTENT_ADDRESS_DIRECTORY
    )
    python = _python_executable(args.python_executable)
    construction_envelope = _input_file(
        args.construction_envelope, field="construction_envelope"
    )
    toolchain_root = _input_directory(args.toolchain_root, field="toolchain_root")
    splat_runtime = _input_directory(
        args.splat_render_runtime_root, field="splat_render_runtime_root"
    )
    fresh_diagnostic_bootstrap = bool(args.fresh_diagnostic_bootstrap)
    diagnostic_bootstrap_mode = (
        FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
        if fresh_diagnostic_bootstrap
        else CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE
    )
    if fresh_diagnostic_bootstrap:
        if args.diagnostic_checkpoint_reference:
            raise SceneConfigurationDiagnosticIterationError(
                "scene_configuration_diagnostic_iteration_checkpoint_source_ambiguous"
            )
        checkpoint_reference = None
    else:
        checkpoint_reference = _input_file(
            args.diagnostic_checkpoint_reference,
            field="diagnostic_checkpoint_reference",
        )
    project_spend = _input_file(
        args.project_spend_reconciliation,
        field="project_spend_reconciliation",
    )
    provider_zero = _input_file(
        args.initial_provider_zero, field="initial_provider_zero"
    )
    bundle_output = _output_directory(
        args.bundle_output_root, field="bundle_output_root", empty=True
    )
    authority_output = _output_path(
        args.scene_configuration_attempt_authority,
        field="scene_configuration_attempt_authority",
    )
    job_dir = _output_directory(
        args.scene_configuration_job_dir,
        field="scene_configuration_job_dir",
        empty=False,
    )
    admission_output = _output_path(args.admission_out, field="admission_out")
    adapter_output = _output_path(args.adapter_output, field="adapter_output")
    preparation_output = _output_path(
        args.iteration_preparation_receipt,
        field="iteration_preparation_receipt",
    )
    retain_warm_session = bool(getattr(args, "retain_warm_session", False))
    allowed_machine_values = getattr(args, "allowed_vast_machine_id", ()) or ()
    try:
        if any(isinstance(value, bool) for value in allowed_machine_values):
            raise ValueError("boolean machine id")
        allowed_machine_ids = tuple(
            sorted({int(value) for value in allowed_machine_values})
        )
        if any(value <= 0 for value in allowed_machine_ids):
            raise ValueError("non-positive machine id")
    except (TypeError, ValueError) as exc:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_allowed_vast_machine_ids_invalid"
        ) from exc
    if retain_warm_session and not args.execute:
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_warm_retention_requires_execute"
        )
    warm_authority_output: Path | None = None
    warm_session_output_root: Path | None = None
    if retain_warm_session:
        if not args.warm_session_authority or not args.warm_session_output_root:
            raise SceneConfigurationDiagnosticIterationError(
                "scene_configuration_diagnostic_iteration_warm_outputs_missing"
            )
        warm_authority_output = _output_path(
            args.warm_session_authority,
            field="warm_session_authority",
        )
        warm_session_output_root = _output_directory(
            args.warm_session_output_root,
            field="warm_session_output_root",
            empty=True,
        )
        if warm_session_output_root.exists():
            raise SceneConfigurationDiagnosticIterationError(
                "scene_configuration_diagnostic_iteration_warm_session_output_root_exists"
            )

    preparation_started = clock()
    stage_started = preparation_started
    staged = stage_scene_configuration_diagnostic_release(
        source_repo=source_repo,
        source_commit=args.source_commit,
        remote_branch=args.remote_branch,
        release_root=release_root,
        state_root=state_root,
    )
    stage_elapsed_ms = int((clock() - stage_started) * 1000)
    release_path = Path(staged["release_path"])
    release_receipt = Path(staged["receipt_path"])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(release_path / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    bundle_started = clock()
    bundle_command = [
        str(python),
        "-m",
        "blueprint_pipeline.task_evaluation_scene_configuration_bundle",
        "--construction-envelope",
        str(construction_envelope),
        "--toolchain-root",
        str(toolchain_root),
        "--repository-root",
        str(release_path),
        "--splat-render-runtime-root",
        str(splat_runtime),
        "--output-root",
        str(bundle_output),
        "--expected-source-commit",
        args.source_commit,
    ]
    if fresh_diagnostic_bootstrap:
        bundle_command.append("--fresh-diagnostic-bootstrap")
    else:
        bundle_command.extend(
            ["--diagnostic-checkpoint-reference", str(checkpoint_reference)]
        )
    try:
        _run_fixed(
            bundle_command,
            cwd=release_path,
            environment=environment,
            runner=runner,
            code="scene_configuration_diagnostic_iteration_bundle_failed",
        )
    except SceneConfigurationDiagnosticIterationError:
        # The builder creates ``stage`` itself after first proving that the
        # output root is empty. A validation failure may leave that expanded,
        # unsealed tree behind; remove only that known self-created directory
        # so a corrected no-spend retry does not strand itself. Any unexpected
        # sibling remains and the existing non-empty gate still fails closed.
        _discard_sealed_bundle_staging_tree(bundle_output)
        raise
    bundle_elapsed_ms = int((clock() - bundle_started) * 1000)
    bundle_receipt_path = (
        bundle_output / f"{BUNDLE_SCHEMA_VERSION}.receipt.json"
    )
    bundle_receipt = _read_json(
        bundle_receipt_path,
        schema=BUNDLE_SCHEMA_VERSION,
        code="scene_configuration_diagnostic_iteration_bundle_receipt_invalid",
    )
    if (
        bundle_receipt.get("source_commit") != args.source_commit
        or bundle_receipt.get("diagnostic_only") is not True
        or bundle_receipt.get("qualification_eligible") is not False
        or bundle_receipt.get("configured_revision_publication_permitted") is not False
        or bundle_receipt.get("offering_publication_permitted") is not False
        or bundle_receipt.get("terminal_e2e_completion_permitted") is not False
        or bundle_receipt.get("diagnostic_bootstrap_mode")
        != diagnostic_bootstrap_mode
        or (
            fresh_diagnostic_bootstrap
            and (
                bundle_receipt.get("source_diagnostic_checkpoint_digest")
                is not None
                or bundle_receipt.get("carried_completed_stage_count") != 0
            )
        )
    ):
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_receipt_invalid"
        )
    bundle_storage = dict(
        bundle_content_addresser(
            bundle_output_root=bundle_output,
            content_address_root=bundle_content_address_root,
        )
    )
    if (
        bundle_storage.get("bundle_path")
        != str(
            bundle_output
            / "task_evaluation_scene_configuration_provider_bundle.zip"
        )
        or bundle_storage.get("bundle_sha256")
        != bundle_receipt.get("bundle_sha256")
        or bundle_storage.get("path_preserved") is not True
        or bundle_storage.get("bytes_preserved") is not True
        or bundle_storage.get("mode_preserved") is not True
        or bundle_storage.get("provider_mutations_performed") != 0
    ):
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_bundle_content_address_invalid"
        )
    bundle_staging_tree_removed = _discard_sealed_bundle_staging_tree(bundle_output)
    authority_started = clock()
    authority_command = [
        str(python),
        "-m",
        "blueprint_pipeline.task_evaluation_scene_configuration_paid_authority",
        "--bundle-receipt",
        str(bundle_receipt_path),
        "--project-spend-reconciliation",
        str(project_spend),
        "--initial-provider-zero",
        str(provider_zero),
        "--authorization-reference",
        args.authorization_reference,
        "--authorized-by",
        args.authorized_by,
        "--authorized-on",
        args.authorized_on,
        "--source-commit",
        args.source_commit,
        "--container-image",
        SCENE_CONFIGURATION_PROVIDER_IMAGE,
        "--resource-name",
        args.pod_name,
        "--max-hourly-rate-usd",
        str(args.max_hourly_rate_usd),
        "--hard-cap-usd",
        str(args.hard_cap_usd),
        "--hard-ttl-seconds",
        str(args.hard_ttl_seconds),
        "--provider-compute-spend-cap-usd",
        str(args.provider_compute_spend_cap_usd),
        "--openai-max-cost-usd",
        str(args.openai_max_cost_usd),
        "--openai-max-requests",
        str(args.openai_max_requests),
        "--openai-artifixer-semantic-teacher-max-cost-usd",
        str(args.openai_artifixer_semantic_teacher_max_cost_usd),
        "--openai-artifixer-visual-review-max-cost-usd",
        str(args.openai_artifixer_visual_review_max_cost_usd),
        "--openai-content-agents-max-cost-usd",
        str(args.openai_content_agents_max_cost_usd),
        "--output",
        str(authority_output),
    ]
    _run_fixed(
        authority_command,
        cwd=release_path,
        environment=environment,
        runner=runner,
        code="scene_configuration_diagnostic_iteration_authority_failed",
    )
    authority_finished = clock()
    authority_elapsed_ms = int((authority_finished - authority_started) * 1000)
    total_preparation_elapsed_ms = int(
        (authority_finished - preparation_started) * 1000
    )
    authority = _read_json(
        authority_output,
        schema=AUTHORITY_SCHEMA_VERSION,
        code="scene_configuration_diagnostic_iteration_authority_invalid",
    )
    if (
        authority.get("source_commit") != args.source_commit
        or authority.get("bundle_sha256") != bundle_receipt.get("bundle_sha256")
        or authority.get("diagnostic_only") is not True
        or authority.get("qualification_eligible") is not False
        or authority.get("configured_revision_publication_permitted") is not False
        or authority.get("offering_publication_permitted") is not False
        or authority.get("terminal_e2e_completion_permitted") is not False
        or authority.get("diagnostic_bootstrap_mode")
        != diagnostic_bootstrap_mode
    ):
        raise SceneConfigurationDiagnosticIterationError(
            "scene_configuration_diagnostic_iteration_authority_invalid"
        )
    if retain_warm_session:
        checkpoint_root: Path | None = None
        if not fresh_diagnostic_bootstrap:
            try:
                checkpoint_reference_value = json.loads(
                    checkpoint_reference.read_text(encoding="utf-8")
                )
                checkpoint_root = _input_directory(
                    str(checkpoint_reference_value.get("checkpoint_root") or ""),
                    field="diagnostic_checkpoint_root",
                )
            except (
                AttributeError,
                json.JSONDecodeError,
                OSError,
                UnicodeError,
            ) as exc:
                raise SceneConfigurationDiagnosticIterationError(
                    "scene_configuration_diagnostic_iteration_checkpoint_reference_invalid"
                ) from exc
        materialize_scene_configuration_warm_session_authority(
            bundle_receipt_path=bundle_receipt_path,
            paid_attempt_authority_path=authority_output,
            diagnostic_release_receipt_path=release_receipt,
            checkpoint_root=checkpoint_root,
            maximum_warm_iterations=args.maximum_warm_iterations,
            output_path=warm_authority_output,
        )

    preparation: dict[str, Any] = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "status": "ready_for_diagnostic_allocator",
        "program_id": "arm-decision-proof-v1",
        "day_gate": "day-28",
        "probe_kind": PROBE_KIND,
        "source_commit": args.source_commit,
        "remote_ref": staged["remote_ref"],
        "diagnostic_release_receipt": {
            "path": str(release_receipt),
            "receipt_digest": staged["receipt_digest"],
        },
        "release_path": str(release_path),
        "source_only_release": True,
        "source_checkout_reused": staged["reused_existing_checkout"],
        "source_materialization_elapsed_ms": stage_elapsed_ms,
        "source_materialization_target_ms": 5_000,
        "source_materialization_target_met": stage_elapsed_ms < 5_000,
        "total_preparation_elapsed_ms": total_preparation_elapsed_ms,
        "total_preparation_seconds_claimed": False,
        "bundle_build_elapsed_ms": bundle_elapsed_ms,
        "bundle_content_address_storage": bundle_storage,
        "bundle_staging_tree_removed_after_seal": bundle_staging_tree_removed,
        "authority_build_elapsed_ms": authority_elapsed_ms,
        "bundle_receipt_digest": bundle_receipt.get("receipt_digest"),
        "authority_digest": authority.get("authority_digest"),
        "splat_runtime_reused_by_reference": True,
        "splat_runtime_copied": False,
        "remaining_preparation_bottleneck": {
            "diagnostic_bundle_rebuilt_for_exact_source_commit": True,
            "toolchain_tree_copied_and_provider_zip_rebuilt": True,
            "unsafe_hardlink_optimization_used": False,
            "sealed_bundle_zip_content_addressed": True,
            "reason": (
                "the existing bundle builder seals modes and byte inventories; "
                "the mutable staging tree stays private, while only a fully "
                "validated immutable ZIP may share a digest-bound inode"
            ),
        },
        "active_release_link_updated": False,
        "systemd_units_reinstalled": False,
        "systemd_services_restarted": False,
        "diagnostic_only": True,
        "development_only": True,
        "qualification_eligible": False,
        "configured_revision_publication_permitted": False,
        "offering_publication_permitted": False,
        "terminal_e2e_completion_permitted": False,
        "paid_execution_requested": bool(args.execute),
        "warm_session_retention_requested": retain_warm_session,
        "allowed_vast_machine_ids": list(allowed_machine_ids),
        "diagnostic_bootstrap_mode": diagnostic_bootstrap_mode,
        "provider_mutation_performed_during_preparation": False,
        "raw_secret_values_recorded": False,
        "preparation_digest": "",
    }
    preparation["preparation_digest"] = canonical_digest(
        preparation, digest_field="preparation_digest"
    )
    _write_exclusive(preparation_output, preparation)

    # This is intentionally the last observation before the canonical
    # allocator subprocess.  A force-push or checkout edit during bundle or
    # authority preparation therefore fails before paid admission.
    validate_scene_configuration_diagnostic_release_receipt(
        release_receipt,
        expected_source_commit=args.source_commit,
        expected_release_path=release_path,
    )
    allocator_command = [
        str(python),
        "-m",
        "blueprint_pipeline.paid_resource_allocator",
        "gpu-canary",
        "--provider",
        "vast",
        "--probe-kind",
        PROBE_KIND,
        "--expected-source-commit",
        args.source_commit,
        "--experimental-branch-diagnostic",
        "--scene-configuration-diagnostic-only",
        "--release-evidence",
        str(release_receipt),
        "--scene-configuration-bundle-receipt",
        str(bundle_receipt_path),
        "--scene-configuration-attempt-authority",
        str(authority_output),
        "--scene-configuration-job-dir",
        str(job_dir),
        "--pod-name",
        args.pod_name,
        "--admission-out",
        str(admission_output),
        "--adapter-output",
        str(adapter_output),
    ]
    if args.execute:
        allocator_command.append("--execute")
    for machine_id in allowed_machine_ids:
        allocator_command.extend(
            ["--scene-configuration-allowed-vast-machine-id", str(machine_id)]
        )
    if retain_warm_session:
        allocator_command.extend(
            [
                "--scene-configuration-retain-warm-session",
                "--scene-configuration-warm-session-authority",
                str(warm_authority_output),
                "--scene-configuration-warm-session-output-root",
                str(warm_session_output_root),
            ]
        )
    _run_fixed(
        allocator_command,
        cwd=release_path,
        environment=environment,
        runner=runner,
        code="scene_configuration_diagnostic_iteration_allocator_failed",
    )
    return preparation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--remote-branch", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--construction-envelope", required=True)
    parser.add_argument("--toolchain-root", required=True)
    parser.add_argument("--splat-render-runtime-root", required=True)
    parser.add_argument("--diagnostic-checkpoint-reference")
    parser.add_argument("--fresh-diagnostic-bootstrap", action="store_true")
    parser.add_argument("--bundle-output-root", required=True)
    parser.add_argument("--bundle-content-address-root")
    parser.add_argument("--project-spend-reconciliation", required=True)
    parser.add_argument("--initial-provider-zero", required=True)
    parser.add_argument("--authorization-reference", required=True)
    parser.add_argument("--authorized-by", required=True)
    parser.add_argument("--authorized-on", required=True)
    parser.add_argument("--pod-name", required=True)
    parser.add_argument("--max-hourly-rate-usd", required=True, type=float)
    parser.add_argument("--hard-cap-usd", required=True, type=float)
    parser.add_argument("--hard-ttl-seconds", required=True, type=int)
    parser.add_argument("--provider-compute-spend-cap-usd", required=True, type=float)
    parser.add_argument("--openai-max-cost-usd", type=float, default=0.0)
    parser.add_argument("--openai-max-requests", type=int, default=0)
    parser.add_argument(
        "--openai-artifixer-semantic-teacher-max-cost-usd", type=float, default=0.0
    )
    parser.add_argument(
        "--openai-artifixer-visual-review-max-cost-usd", type=float, default=0.0
    )
    parser.add_argument("--openai-content-agents-max-cost-usd", type=float, default=0.0)
    parser.add_argument("--scene-configuration-attempt-authority", required=True)
    parser.add_argument("--scene-configuration-job-dir", required=True)
    parser.add_argument("--admission-out", required=True)
    parser.add_argument("--adapter-output", required=True)
    parser.add_argument("--iteration-preparation-receipt", required=True)
    parser.add_argument("--retain-warm-session", action="store_true")
    parser.add_argument("--allowed-vast-machine-id", action="append", type=int, default=[])
    parser.add_argument("--warm-session-authority")
    parser.add_argument("--warm-session-output-root")
    parser.add_argument("--maximum-warm-iterations", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    return parser


def _cleanup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a sealed diagnostic provider bundle and reclaim only its "
            "self-created expanded staging tree."
        )
    )
    parser.add_argument("--bundle-output-root", required=True)
    parser.add_argument("--cleanup-receipt", required=True)
    return parser


def _content_address_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one sealed diagnostic provider bundle and replace only "
            "its ZIP inode with a digest-addressed hardlink when bytes and "
            "metadata match exactly."
        )
    )
    parser.add_argument("--bundle-output-root", required=True)
    parser.add_argument("--content-address-root", required=True)
    parser.add_argument("--deduplication-receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == [CONTENT_ADDRESS_COMMAND]:
        content_args = _content_address_parser().parse_args(arguments[1:])
        try:
            result = reconcile_content_addressed_provider_bundle(
                bundle_output_root=content_args.bundle_output_root,
                content_address_root=content_args.content_address_root,
                deduplication_receipt=content_args.deduplication_receipt,
            )
        except (OSError, SceneConfigurationDiagnosticIterationError) as exc:
            print(
                json.dumps(
                    {
                        "schema_version": CONTENT_ADDRESS_SCHEMA_VERSION,
                        "status": "blocked",
                        "blockers": [str(exc)],
                        "provider_mutations_performed": 0,
                        "raw_secret_values_recorded": False,
                    },
                    sort_keys=True,
                )
            )
            return 2
        print(
            json.dumps(
                {
                    "schema_version": CONTENT_ADDRESS_SCHEMA_VERSION,
                    "status": "completed",
                    "deduplication_receipt": str(
                        content_args.deduplication_receipt
                    ),
                    "receipt_digest": result["receipt_digest"],
                    "existing_content_object_reused": result[
                        "existing_content_object_reused"
                    ],
                    "allocated_bytes_reclaimed": result[
                        "allocated_bytes_reclaimed"
                    ],
                    "provider_mutations_performed": 0,
                    "raw_secret_values_recorded": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments[:1] == [CLEANUP_COMMAND]:
        cleanup_args = _cleanup_parser().parse_args(arguments[1:])
        try:
            result = reconcile_sealed_bundle_staging_tree(
                bundle_output_root=cleanup_args.bundle_output_root,
                cleanup_receipt=cleanup_args.cleanup_receipt,
            )
        except (OSError, SceneConfigurationDiagnosticIterationError) as exc:
            print(
                json.dumps(
                    {
                        "schema_version": CLEANUP_SCHEMA_VERSION,
                        "status": "blocked",
                        "blockers": [str(exc)],
                        "provider_mutations_performed": 0,
                        "raw_secret_values_recorded": False,
                    },
                    sort_keys=True,
                )
            )
            return 2
        print(
            json.dumps(
                {
                    "schema_version": CLEANUP_SCHEMA_VERSION,
                    "status": "completed",
                    "cleanup_receipt": str(cleanup_args.cleanup_receipt),
                    "receipt_digest": result["receipt_digest"],
                    "removed": result["removed"],
                    "file_bytes": result["file_bytes"],
                    "provider_mutations_performed": 0,
                    "raw_secret_values_recorded": False,
                },
                sort_keys=True,
            )
        )
        return 0

    args = _parser().parse_args(arguments)
    try:
        result = run_scene_configuration_diagnostic_iteration(args)
    except (
        OSError,
        SceneConfigurationDiagnosticIterationError,
        SceneConfigurationDiagnosticReleaseError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema_version": PREPARATION_SCHEMA_VERSION,
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutations_performed": 0,
                    "diagnostic_only": True,
                    "qualification_eligible": False,
                    "configured_revision_publication_permitted": False,
                    "offering_publication_permitted": False,
                    "terminal_e2e_completion_permitted": False,
                    "raw_secret_values_recorded": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "schema_version": PREPARATION_SCHEMA_VERSION,
                "status": "allocator_invoked",
                "source_commit": result["source_commit"],
                "source_materialization_elapsed_ms": result[
                    "source_materialization_elapsed_ms"
                ],
                "source_checkout_reused": result["source_checkout_reused"],
                "diagnostic_only": True,
                "qualification_eligible": False,
                "raw_secret_values_recorded": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
