#!/usr/bin/env python3
"""Build the release-bound native Isaac import qualification component."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from scripts.build_task_evaluation_scene_configuration_component_package import (
    build_scene_configuration_component_package,
)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(  # nosec B603 B607 - fixed git argv
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode:
        raise ValueError("scene_configuration_native_import_source_invalid")
    return completed.stdout.strip()


def build_native_import_scene_configuration_component(
    *,
    repository_root: str | Path,
    expected_blueprint_commit: str,
    output_root: str | Path,
) -> dict:
    """Seal the native driver identity used inside the parent Isaac runtime."""

    repository = Path(repository_root).expanduser().resolve()
    if (
        _git(repository, "rev-parse", "HEAD") != expected_blueprint_commit
        or _git(repository, "status", "--porcelain=v1")
    ):
        raise ValueError("scene_configuration_native_import_source_invalid")
    staging = Path(tempfile.mkdtemp(prefix="native-import-scene-configuration-"))
    try:
        run = staging / "run"
        run.write_text(
            "#!/bin/sh\n"
            "set -eu\n"
            "PYTHON_BIN=/isaac-sim/python.sh\n"
            "if [ ! -x \"$PYTHON_BIN\" ]; then PYTHON_BIN=$(command -v python3); fi\n"
            "RUNTIME_ROOT=${BLUEPRINT_SCENE_CONFIGURATION_RUNTIME_ROOT:-/workspace/task_evaluation_scene_configuration_provider_bundle/provider_runtime}\n"
            "ISAAC_PYTHONPATH=${BLUEPRINT_SCENE_CONFIGURATION_ISAAC_PYTHONPATH-}\n"
            "export PYTHONPATH=\"$RUNTIME_ROOT${ISAAC_PYTHONPATH:+:$ISAAC_PYTHONPATH}\"\n"
            "exec \"$PYTHON_BIN\" -m "
            "blueprint_pipeline.task_evaluation_scene_configuration_native_import_driver\n",
            encoding="utf-8",
        )
        run.chmod(0o755)
        driver = (
            repository
            / "src/blueprint_pipeline/task_evaluation_scene_configuration_native_import_driver.py"
        )
        if driver.is_symlink() or not driver.is_file():
            raise ValueError("scene_configuration_native_import_source_invalid")
        shutil.copyfile(driver, staging / "native_import_driver.py")
        source_receipt = {
            "schema_version": "task_evaluation_native_import_component_source.v1",
            "repository": "https://github.com/ognjhunt/BlueprintCapturePipeline",
            "commit": expected_blueprint_commit,
            "tree": _git(repository, "rev-parse", "HEAD^{tree}"),
            "license": "Blueprint production adapter",
            "scene_specific_source": False,
            "receipt_digest": "",
        }
        source_receipt["receipt_digest"] = canonical_digest(
            source_receipt, digest_field="receipt_digest"
        )
        (staging / "source_receipt.json").write_text(
            json.dumps(source_receipt, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return build_scene_configuration_component_package(
            adapter_id="simready_native_import_qualification",
            source_root=staging,
            driver_entrypoint="run",
            source_repository="https://github.com/ognjhunt/BlueprintCapturePipeline",
            source_commit=expected_blueprint_commit,
            source_license="Blueprint production adapter",
            output_root=output_root,
        )
    finally:
        shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-blueprint-commit", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    value = build_native_import_scene_configuration_component(
        repository_root=args.repository_root,
        expected_blueprint_commit=args.expected_blueprint_commit,
        output_root=args.output_root,
    )
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
