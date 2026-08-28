#!/usr/bin/env python3
"""Provision reusable scene-configuration runtimes for one exact release.

This is release infrastructure, not a scene harness.  It consumes governed
platform prerequisites and pinned public source mirrors, publishes exact-SHA
renderer/toolchain trees, and returns the non-secret environment bindings used
by the Website preparation and activation workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
    validate_scene_configuration_toolchain,
)
from blueprint_pipeline.public_scene_artifixer3d_bundle import (
    VGG16_WEIGHTS_FILENAME,
    VGG16_WEIGHTS_SHA256,
    VGG16_WEIGHTS_SIZE_BYTES,
    VGG16_WEIGHTS_SOURCE_URL,
)
from blueprint_pipeline.task_evaluation_splat_render_runtime import (
    validate_splat_render_runtime,
)
from scripts.build_task_evaluation_scene_configuration_artifixer_component import (
    build_artifixer_scene_configuration_component,
)
from scripts.build_task_evaluation_scene_configuration_content_agents_component import (
    build_content_agents_scene_configuration_component,
)
from scripts.build_task_evaluation_scene_configuration_native_import_component import (
    build_native_import_scene_configuration_component,
)
from scripts.build_task_evaluation_scene_configuration_toolchain import (
    build_published_scene_configuration_toolchain,
)
from scripts.build_task_evaluation_splat_render_runtime import (
    build_published_splat_render_runtime,
)


SCHEMA_VERSION = "task_evaluation_scene_configuration_release_runtime.v1"
DEFAULT_RUNTIME_ROOT = Path("/var/lib/blueprint/task-evaluation-inputs/system-runtimes")
DEFAULT_RELEASE_WINDOW_PREFIX = (
    "s3://blueprint/task-evaluation/production-inputs/coordinator-release-windows/"
)
DEFAULT_ACTIVATION_DESTINATION_PREFIX = (
    "s3://blueprint/task-evaluation/production-inputs/task-evaluation-activations"
)
Readback = Callable[[Path], bytes]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def materialize_vgg16_weights(*, cache_root: str | Path) -> Path:
    """Cache the exact public LPIPS backbone before building an immutable release."""

    root = Path(cache_root).expanduser().absolute()
    destination = root / VGG16_WEIGHTS_FILENAME
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.stat().st_size != VGG16_WEIGHTS_SIZE_BYTES
            or _sha256(destination) != VGG16_WEIGHTS_SHA256
        ):
            raise ValueError("scene_configuration_vgg16_cache_invalid")
        return destination
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{VGG16_WEIGHTS_FILENAME}.download"
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("scene_configuration_vgg16_cache_invalid")
    request = urllib.request.Request(
        VGG16_WEIGHTS_SOURCE_URL,
        headers={"User-Agent": "BlueprintCapturePipeline/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:  # nosec B310
            with temporary.open("xb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        if (
            temporary.stat().st_size != VGG16_WEIGHTS_SIZE_BYTES
            or _sha256(temporary) != VGG16_WEIGHTS_SHA256
        ):
            raise ValueError("scene_configuration_vgg16_download_mismatch")
        temporary.chmod(0o444)
        temporary.replace(destination)
    except Exception:
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
        raise
    return destination


def reconcile_legacy_vgg16_runtime_storage(
    *,
    runtime_root: str | Path,
) -> dict[str, Any]:
    """Deduplicate validated legacy toolchain weights against the sealed cache.

    Older releases copied the immutable VGG16 object into every toolchain.  The
    current publisher hardlinks it, but that does not repair already-published
    releases.  Validate the complete candidate set before replacing any inode;
    then atomically install hardlinks to the exact digest-bound cache object.
    """

    root = Path(runtime_root).expanduser().absolute()
    canonical = root / "public-model-cache" / VGG16_WEIGHTS_FILENAME
    if (
        canonical.is_symlink()
        or not canonical.is_file()
        or canonical.stat().st_size != VGG16_WEIGHTS_SIZE_BYTES
        or stat.S_IMODE(canonical.stat().st_mode) != 0o444
        or _sha256(canonical) != VGG16_WEIGHTS_SHA256
    ):
        raise ValueError("scene_configuration_vgg16_cache_invalid")
    candidates = sorted(
        (root / "scene-configuration").glob(
            "*/components/artifixer3d_observed_object_removal/package/"
            "blueprint_runtime/torch_home/hub/checkpoints/"
            f"{VGG16_WEIGHTS_FILENAME}"
        )
    )
    validated: list[tuple[Path, os.stat_result]] = []
    for candidate in candidates:
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("scene_configuration_legacy_vgg16_runtime_invalid")
        metadata = candidate.stat()
        if (
            metadata.st_size != VGG16_WEIGHTS_SIZE_BYTES
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or _sha256(candidate) != VGG16_WEIGHTS_SHA256
        ):
            raise ValueError("scene_configuration_legacy_vgg16_runtime_invalid")
        validated.append((candidate, metadata))

    canonical_metadata = canonical.stat()
    reconciled: list[str] = []
    estimated_reclaimed_bytes = 0
    for candidate, previous_metadata in validated:
        if (
            previous_metadata.st_dev == canonical_metadata.st_dev
            and previous_metadata.st_ino == canonical_metadata.st_ino
        ):
            continue
        temporary = candidate.with_name(f".{candidate.name}.content-address-migration")
        if temporary.exists() or temporary.is_symlink():
            raise ValueError("scene_configuration_vgg16_migration_temporary_exists")
        try:
            os.link(canonical, temporary, follow_symlinks=False)
            os.replace(temporary, candidate)
        except Exception:
            if temporary.is_file() and not temporary.is_symlink():
                temporary.unlink()
            raise
        installed = candidate.stat()
        if (
            installed.st_dev != canonical_metadata.st_dev
            or installed.st_ino != canonical_metadata.st_ino
            or stat.S_IMODE(installed.st_mode) != 0o444
            or _sha256(candidate) != VGG16_WEIGHTS_SHA256
        ):
            raise ValueError("scene_configuration_vgg16_migration_readback_failed")
        reconciled.append(str(candidate))
        if previous_metadata.st_nlink == 1:
            estimated_reclaimed_bytes += previous_metadata.st_blocks * 512

    return {
        "schema_version": "task_evaluation_scene_configuration_vgg16_storage_reconciliation.v1",
        "status": "reconciled",
        "canonical_path": str(canonical),
        "validated_candidate_count": len(validated),
        "reconciled_candidate_count": len(reconciled),
        "reconciled_paths": reconciled,
        "estimated_reclaimed_bytes": estimated_reclaimed_bytes,
        "digest": VGG16_WEIGHTS_SHA256,
        "mode": "0444",
    }


def _remove_tree(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            path.chmod(0o700)
    root.chmod(0o700)
    shutil.rmtree(root)


def provision_scene_configuration_release(
    *,
    repository_root: str | Path,
    source_commit: str,
    runtime_root: str | Path,
    node_executable: str | Path,
    browser_root: str | Path,
    browser_executable: str | Path,
    node_modules_root: str | Path,
    artifixer_root: str | Path,
    content_agents_root: str | Path,
    readback: Readback,
    readback_actor: str,
) -> dict[str, Any]:
    """Publish or reopen the two immutable runtime trees for one release."""

    repository = Path(repository_root).expanduser().resolve()
    runtimes = Path(runtime_root).expanduser().absolute()
    splat_root = runtimes / "splat-render" / source_commit
    toolchain_root = runtimes / "scene-configuration" / source_commit
    if splat_root.exists():
        splat = validate_splat_render_runtime(
            runtime_root=splat_root,
            repo_root=repository,
            allowed_roots=(runtimes,),
        )
        splat_receipt: dict[str, Any] = {
            "status": "reused_validated_immutable_runtime",
            "runtime_root": str(splat_root),
            "runtime_digest": splat["identity"]["runtime_digest"],
        }
    else:
        splat_receipt = build_published_splat_render_runtime(
            repository_root=repository,
            source_commit=source_commit,
            node_executable=node_executable,
            browser_root=browser_root,
            browser_executable=browser_executable,
            node_modules_root=node_modules_root,
            output_root=splat_root,
            readback=readback,
            readback_actor=readback_actor,
        )
    if toolchain_root.exists():
        toolchain = validate_scene_configuration_toolchain(
            root=toolchain_root,
            expected_source_commit=source_commit,
        )
        toolchain_receipt: dict[str, Any] = {
            "status": "reused_validated_immutable_toolchain",
            "toolchain_root": str(toolchain_root),
            "toolchain_digest": toolchain["toolchain_digest"],
        }
    else:
        vgg16_weights = materialize_vgg16_weights(
            cache_root=runtimes / "public-model-cache"
        )
        component_parent = Path(
            tempfile.mkdtemp(prefix="scene-configuration-components-")
        )
        try:
            component_packages = {
                "artifixer3d_observed_object_removal": component_parent / "artifixer",
                "content_agents_rigid_replacement": component_parent / "content-agents",
                "simready_native_import_qualification": component_parent / "native-import",
            }
            build_artifixer_scene_configuration_component(
                repository_root=repository,
                expected_blueprint_commit=source_commit,
                artifixer_root=artifixer_root,
                vgg16_weights_path=vgg16_weights,
                output_root=component_packages[
                    "artifixer3d_observed_object_removal"
                ],
            )
            build_content_agents_scene_configuration_component(
                repository_root=repository,
                expected_blueprint_commit=source_commit,
                content_agents_root=content_agents_root,
                output_root=component_packages[
                    "content_agents_rigid_replacement"
                ],
            )
            build_native_import_scene_configuration_component(
                repository_root=repository,
                expected_blueprint_commit=source_commit,
                output_root=component_packages[
                    "simready_native_import_qualification"
                ],
            )
            toolchain_receipt = build_published_scene_configuration_toolchain(
                source_commit=source_commit,
                output_root=toolchain_root,
                readback=readback,
                readback_actor=readback_actor,
                component_packages=component_packages,
            )
        finally:
            _remove_tree(component_parent)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "source_commit": source_commit,
        "splat_render_runtime": splat_receipt,
        "scene_configuration_toolchain": toolchain_receipt,
        "environment": {
            "BLUEPRINT_TASK_EVALUATION_SPLAT_RENDER_RUNTIME_ROOT": str(splat_root),
            "BLUEPRINT_TASK_EVALUATION_SCENE_CONFIGURATION_TOOLCHAIN_ROOT": str(
                toolchain_root
            ),
            "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_RELEASE_WINDOW_PREFIX": (
                DEFAULT_RELEASE_WINDOW_PREFIX
            ),
            "BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_DESTINATION_PREFIX": (
                DEFAULT_ACTIVATION_DESTINATION_PREFIX
            ),
        },
        "scene_specific_artifacts_built": False,
        "provider_mutation_performed": False,
        "paid_resource_allocated": False,
    }


def service_account_readback(user: str) -> Readback:
    def read(path: Path) -> bytes:
        return subprocess.run(
            ["sudo", "-n", "-u", user, "--", "dd", f"if={path}", "status=none"],
            check=True,
            capture_output=True,
        ).stdout

    return read


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--node-executable", required=True)
    parser.add_argument("--browser-root", required=True)
    parser.add_argument("--browser-executable", required=True)
    parser.add_argument("--node-modules-root", required=True)
    parser.add_argument("--artifixer-root", required=True)
    parser.add_argument("--content-agents-root", required=True)
    parser.add_argument("--readback-user", required=True)
    args = parser.parse_args()
    value = provision_scene_configuration_release(
        repository_root=args.repository_root,
        source_commit=args.source_commit,
        runtime_root=args.runtime_root,
        node_executable=args.node_executable,
        browser_root=args.browser_root,
        browser_executable=args.browser_executable,
        node_modules_root=args.node_modules_root,
        artifixer_root=args.artifixer_root,
        content_agents_root=args.content_agents_root,
        readback=service_account_readback(args.readback_user),
        readback_actor=f"service-account:{args.readback_user}",
    )
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
