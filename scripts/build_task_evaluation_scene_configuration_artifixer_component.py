#!/usr/bin/env python3
"""Build the released ArtiFixer adapter used by every configured scene."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_artifixer3d_bundle import (
    ARTIFIXER_COMMIT,
    ARTIFIXER_REPOSITORY,
    ARTIFIXER_TREE,
    COMPONENT_SOURCE_SCHEMA_VERSION,
    RUNTIME_BLUEPRINT_MODULES,
    RUNTIME_VGG16_WEIGHTS,
    VGG16_WEIGHTS_SHA256,
    VGG16_WEIGHTS_SIZE_BYTES,
)
from blueprint_pipeline.task_evaluation_scene_configuration_python_wheelhouse import (
    build_scene_configuration_python_wheelhouse,
)
from scripts.build_task_evaluation_scene_configuration_component_package import (
    build_scene_configuration_component_package,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
                *args,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("scene_configuration_artifixer_source_invalid") from exc


def _copy(
    source: Path,
    destination: Path,
    *,
    executable: bool = False,
    immutable: bool = False,
) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError("scene_configuration_artifixer_source_invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination_mode = 0o755 if executable else (0o444 if immutable else 0o644)
    linked = False
    if immutable and stat.S_IMODE(source.stat().st_mode) == destination_mode:
        try:
            os.link(source, destination, follow_symlinks=False)
            linked = True
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
    if not linked:
        shutil.copyfile(source, destination)
        destination.chmod(destination_mode)


def build_artifixer_scene_configuration_component(
    *,
    repository_root: str | Path,
    expected_blueprint_commit: str,
    artifixer_root: str | Path,
    vgg16_weights_path: str | Path,
    output_root: str | Path,
) -> dict:
    """Seal exact released source plus the existing Blueprint runtime adapter."""

    repository = Path(repository_root).expanduser().resolve()
    upstream = Path(artifixer_root).expanduser().resolve()
    vgg16_weights = Path(vgg16_weights_path).expanduser().resolve()
    if (
        _git(repository, "rev-parse", "HEAD") != expected_blueprint_commit
        or _git(repository, "status", "--porcelain=v1")
        or _git(upstream, "rev-parse", "HEAD") != ARTIFIXER_COMMIT
        or _git(upstream, "rev-parse", "HEAD^{tree}") != ARTIFIXER_TREE
        or _git(upstream, "status", "--porcelain=v1")
        or vgg16_weights.is_symlink()
        or not vgg16_weights.is_file()
        or vgg16_weights.stat().st_size != VGG16_WEIGHTS_SIZE_BYTES
        or _sha256(vgg16_weights) != VGG16_WEIGHTS_SHA256
    ):
        raise ValueError("scene_configuration_artifixer_source_invalid")
    staging = Path(tempfile.mkdtemp(prefix="artifixer-scene-configuration-"))
    try:
        tracked = _git(upstream, "ls-files").splitlines()
        source_rows = []
        for name in sorted(tracked):
            if name == "thirdparty/3DGRUT-ArtiFixer":
                continue
            source = upstream / name
            destination = staging / "artifixer_source" / name
            _copy(source, destination, executable=bool(source.stat().st_mode & 0o111))
            source_rows.append(
                {
                    "relative_path": name,
                    "size_bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                }
            )
        source_receipt = {
            "schema_version": COMPONENT_SOURCE_SCHEMA_VERSION,
            "repository": ARTIFIXER_REPOSITORY,
            "commit": ARTIFIXER_COMMIT,
            "tree": ARTIFIXER_TREE,
            "license": "Apache-2.0",
            "files": source_rows,
            "receipt_digest": "",
        }
        source_receipt["receipt_digest"] = canonical_digest(
            source_receipt, digest_field="receipt_digest"
        )
        (staging / "artifixer_source_receipt.json").write_text(
            json.dumps(source_receipt, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        blueprint_receipt = {
            "schema_version": "task_evaluation_artifixer_blueprint_source.v1",
            "commit": expected_blueprint_commit,
            "tree": _git(repository, "rev-parse", "HEAD^{tree}"),
            "tracked_files_clean": True,
            "receipt_digest": "",
        }
        blueprint_receipt["receipt_digest"] = canonical_digest(
            blueprint_receipt, digest_field="receipt_digest"
        )
        (staging / "blueprint_source_receipt.json").write_text(
            json.dumps(blueprint_receipt, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        copies = {
            "scripts/run_task_evaluation_scene_configuration_artifixer_component.sh": ("run"),
            "scripts/run_public_scene_artifixer3d.sh": (
                "blueprint_runtime/scripts/run_public_scene_artifixer3d.sh"
            ),
            "scripts/public_scene_artifixer3d_runner.py": (
                "blueprint_runtime/scripts/public_scene_artifixer3d_runner.py"
            ),
            "docs/arm_decision_proof_v1/manifests/image_editor_backends.v1.json": (
                "blueprint_runtime/docs/arm_decision_proof_v1/manifests/"
                "image_editor_backends.v1.json"
            ),
        }
        copies.update(
            {
                f"src/blueprint_pipeline/{name}": (
                    f"blueprint_runtime/src/blueprint_pipeline/{name}"
                )
                for name in RUNTIME_BLUEPRINT_MODULES
            }
        )
        for source_name, destination_name in copies.items():
            source = repository / source_name
            _copy(
                source,
                staging / destination_name,
                executable=(destination_name == "run" or bool(source.stat().st_mode & 0o111)),
            )
        _copy(
            vgg16_weights,
            staging / "blueprint_runtime" / RUNTIME_VGG16_WEIGHTS,
            immutable=True,
        )
        # The Isaac image supplies the scientific stack, but not the OpenAI
        # Agents SDK/Pydantic closure used by the independent visual review.
        # Materialize exact lockfile wheels while still on the control plane;
        # the immutable component/package inventories bind every downloaded
        # byte and the rented provider performs no dependency resolution.
        build_scene_configuration_python_wheelhouse(
            lockfile_path=repository / "uv.lock",
            output_root=staging / "python_wheelhouse",
        )
        return build_scene_configuration_component_package(
            adapter_id="artifixer3d_observed_object_removal",
            source_root=staging,
            driver_entrypoint="run",
            source_repository="https://github.com/ognjhunt/BlueprintCapturePipeline",
            source_commit=expected_blueprint_commit,
            source_license="Blueprint adapter; bundled upstream Apache-2.0",
            output_root=output_root,
        )
    finally:
        shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-blueprint-commit", required=True)
    parser.add_argument("--artifixer-root", required=True)
    parser.add_argument("--vgg16-weights-path", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    value = build_artifixer_scene_configuration_component(
        repository_root=args.repository_root,
        expected_blueprint_commit=args.expected_blueprint_commit,
        artifixer_root=args.artifixer_root,
        vgg16_weights_path=args.vgg16_weights_path,
        output_root=args.output_root,
    )
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
