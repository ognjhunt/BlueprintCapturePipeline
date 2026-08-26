#!/usr/bin/env python3
"""Provision reusable scene-configuration runtimes for one exact release.

This is release infrastructure, not a scene harness.  It consumes governed
platform prerequisites and pinned public source mirrors, publishes exact-SHA
renderer/toolchain trees, and returns the non-secret environment bindings used
by the Website preparation and activation workers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
    validate_scene_configuration_toolchain,
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
    "s3://blueprint-production-inputs/coordinator-release-windows/"
)
DEFAULT_ACTIVATION_DESTINATION_PREFIX = (
    "s3://blueprint-production-inputs/task-evaluation-activations"
)
Readback = Callable[[Path], bytes]


def _remove_tree(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o700 if path.is_dir() else 0o600)
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
