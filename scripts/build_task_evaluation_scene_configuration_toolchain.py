#!/usr/bin/env python3
"""Publish the exact scene-configuration stage toolchain for one release.

This command performs no network access and allocates no paid resource.  It is
intended to run from the canonical deployer as root, then read every installed
byte back as the unprivileged production service account before the immutable
tree becomes eligible for a Website-started configuration run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.immutable_directory_publication import (
    publish_staged_immutable_directory,
)
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
    TOOLCHAIN_SCHEMA_VERSION,
    validate_scene_configuration_toolchain,
)
from blueprint_pipeline.task_evaluation_scene_configuration_component_package import (
    SCHEMA_VERSION as COMPONENT_PACKAGE_SCHEMA_VERSION,
    validate_scene_configuration_component_package,
)
from blueprint_pipeline.task_evaluation_scene_configuration_stage_producers import (
    ADMITTED_PRODUCER_IDENTITIES,
)


RECEIPT_SCHEMA_VERSION = "task_evaluation_scene_configuration_toolchain_publication.v1"
_COMMIT = re.compile(r"[0-9a-f]{40}")
Readback = Callable[[Path], bytes]


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _entrypoint(adapter_id: str) -> bytes:
    return (
        "#!/bin/sh\n"
        "set -eu\n"
        "PYTHON_BIN=/isaac-sim/python.sh\n"
        "if [ ! -x \"$PYTHON_BIN\" ]; then PYTHON_BIN=$(command -v python3); fi\n"
        "exec \"$PYTHON_BIN\" -m "
        "blueprint_pipeline.task_evaluation_scene_configuration_stage_tool "
        f"--adapter-id {adapter_id}\n"
    ).encode("utf-8")


def build_published_scene_configuration_toolchain(
    *,
    source_commit: str,
    output_root: str | Path,
    readback: Readback,
    readback_actor: str,
    component_packages: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Install one exclusive read-only toolchain and prove full-byte readback."""

    if _COMMIT.fullmatch(source_commit) is None or not readback_actor.strip():
        raise ValueError("scene_configuration_toolchain_publication_input_invalid")
    admitted_ids = {identity.adapter_id for identity in ADMITTED_PRODUCER_IDENTITIES}
    if set(component_packages) != admitted_ids:
        raise ValueError("scene_configuration_toolchain_component_set_invalid")
    validated_packages = {
        adapter_id: (
            Path(root).expanduser().resolve(),
            validate_scene_configuration_component_package(
                root=root,
                expected_adapter_id=adapter_id,
            ),
        )
        for adapter_id, root in component_packages.items()
    }
    destination = Path(output_root).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    installed = False
    try:
        stages: dict[str, dict[str, Any]] = {}
        files: list[dict[str, Any]] = []
        for identity in ADMITTED_PRODUCER_IDENTITIES:
            relative = Path("stages") / identity.adapter_id
            executable = staging / relative
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_bytes(_entrypoint(identity.adapter_id))
            executable.chmod(0o555)
            files.append(
                {
                    "relative_path": relative.as_posix(),
                    "sha256": _sha256(executable),
                    "size_bytes": executable.stat().st_size,
                    "executable": True,
                }
            )
            package_source, package_manifest = validated_packages[
                identity.adapter_id
            ]
            package_relative = Path("components") / identity.adapter_id / "package"
            package_destination = staging / package_relative
            shutil.copytree(package_source, package_destination, symlinks=False)
            for component_file in sorted(
                path for path in package_destination.rglob("*") if path.is_file()
            ):
                relative_component_file = component_file.relative_to(staging)
                files.append(
                    {
                        "relative_path": relative_component_file.as_posix(),
                        "sha256": _sha256(component_file),
                        "size_bytes": component_file.stat().st_size,
                        "executable": bool(component_file.stat().st_mode & 0o111),
                    }
                )
            component_relative = package_relative / package_manifest["driver_entrypoint"]
            stages[identity.adapter_id] = {
                "entrypoint": relative.as_posix(),
                "component_entrypoint": component_relative.as_posix(),
                "component_package_manifest": (
                    package_relative
                    / f"{COMPONENT_PACKAGE_SCHEMA_VERSION}.json"
                ).as_posix(),
                "component_package_digest": package_manifest["package_digest"],
                "network_policy": package_manifest["network_policy"],
                "secrets_via_files_only": True,
                "raw_secret_values_in_argv_or_logs": False,
            }
        (staging / "stages").chmod(0o555)
        for path in sorted(
            (item for item in (staging / "components").rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            path.chmod(0o555)
        (staging / "components").chmod(0o555)
        for row in files:
            path = staging / row["relative_path"]
            observed = readback(path)
            if (
                len(observed) != row["size_bytes"]
                or _sha256_bytes(observed) != row["sha256"]
            ):
                raise ValueError("scene_configuration_toolchain_service_readback_failed")
        manifest: dict[str, Any] = {
            "schema_version": TOOLCHAIN_SCHEMA_VERSION,
            "status": "published_full_byte_readback_passed",
            "source_commit": source_commit,
            "full_byte_service_account_readback_passed": True,
            "readback_actor": readback_actor,
            "stages": stages,
            "files": files,
            "toolchain_digest": "",
        }
        manifest["toolchain_digest"] = canonical_digest(
            manifest, digest_field="toolchain_digest"
        )
        manifest_path = staging / f"{TOOLCHAIN_SCHEMA_VERSION}.json"
        manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        manifest_path.chmod(0o444)
        publish_staged_immutable_directory(
            staging=staging,
            destination=destination,
            manifest_name=manifest_path.name,
            output_exists_code="scene_configuration_toolchain_publication_output_exists",
        )
        installed = True
        installed_manifest = destination / manifest_path.name
        manifest_bytes = readback(installed_manifest)
        if manifest_bytes != installed_manifest.read_bytes():
            raise ValueError("scene_configuration_toolchain_service_readback_failed")
        validate_scene_configuration_toolchain(
            root=destination,
            expected_source_commit=source_commit,
        )
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "published_and_read_back",
            "source_commit": source_commit,
            "toolchain_root": str(destination),
            "toolchain_digest": manifest["toolchain_digest"],
            "file_count": len(files),
            "readback_actor": readback_actor,
            "full_byte_service_account_readback_passed": True,
            "provider_mutation_performed": False,
            "paid_resource_allocated": False,
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        receipt_path = destination.parent / f"{destination.name}.publication.v1.json"
        descriptor = os.open(
            receipt_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((canonical_json(receipt) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        return receipt
    except Exception:
        owned = destination if installed else staging
        if owned.exists() and not owned.is_symlink():
            for path in sorted(owned.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                path.chmod(0o700 if path.is_dir() else 0o600)
            owned.chmod(0o700)
            shutil.rmtree(owned)
        raise


def _service_account_readback(user: str) -> Readback:
    def read(path: Path) -> bytes:
        completed = subprocess.run(
            ["sudo", "-n", "-u", user, "--", "dd", f"if={path}", "status=none"],
            check=True,
            capture_output=True,
        )
        return completed.stdout

    return read


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--readback-user", required=True)
    parser.add_argument(
        "--component-package",
        action="append",
        default=[],
        metavar="ADAPTER_ID=PATH",
        help="Exact immutable component package; repeat once per admitted GPU adapter.",
    )
    args = parser.parse_args(argv)
    component_packages: dict[str, str] = {}
    for raw in args.component_package:
        adapter_id, separator, path = raw.partition("=")
        if not separator or not adapter_id or not path or adapter_id in component_packages:
            parser.error("--component-package must be unique ADAPTER_ID=PATH")
        component_packages[adapter_id] = path
    receipt = build_published_scene_configuration_toolchain(
        source_commit=args.source_commit,
        output_root=args.output_root,
        readback=_service_account_readback(args.readback_user),
        readback_actor=f"service-account:{args.readback_user}",
        component_packages=component_packages,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
