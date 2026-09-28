#!/usr/bin/env python3
"""Build one scene-neutral immutable production component package.

This is a release operation, not a scene operation.  A package is built once
from an admitted public tool/runtime checkout and reused by every Website scene
configuration bound to the same exact Blueprint release.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.immutable_directory_publication import (
    publish_staged_immutable_directory,
)
from blueprint_pipeline.task_evaluation_scene_configuration_component_package import (
    SCHEMA_VERSION,
    validate_scene_configuration_component_package,
)
from blueprint_pipeline.task_evaluation_scene_configuration_stage_producers import (
    ADMITTED_PRODUCER_IDENTITIES,
)


_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_source_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("scene_configuration_component_source_symlink_forbidden")
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target_mode = 0o555 if path.stat().st_mode & 0o111 else 0o444
            linked = False
            if stat.S_IMODE(path.stat().st_mode) == target_mode:
                try:
                    os.link(path, target, follow_symlinks=False)
                    linked = True
                except OSError as exc:
                    if exc.errno != errno.EXDEV:
                        raise
            if not linked:
                shutil.copyfile(path, target)
                target.chmod(target_mode)


def build_scene_configuration_component_package(
    *,
    adapter_id: str,
    source_root: str | Path,
    driver_entrypoint: str,
    source_repository: str,
    source_commit: str,
    source_license: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Seal one exhaustive scene-neutral package without network or execution."""

    identities = {
        identity.adapter_id: identity for identity in ADMITTED_PRODUCER_IDENTITIES
    }
    identity = identities.get(adapter_id)
    source = Path(source_root).expanduser().resolve()
    relative_driver = Path(driver_entrypoint)
    destination = Path(output_root).expanduser().absolute()
    if (
        identity is None
        or not source.is_dir()
        or source.is_symlink()
        or not relative_driver.parts
        or relative_driver.is_absolute()
        or ".." in relative_driver.parts
        or not source_repository.strip()
        or _COMMIT.fullmatch(source_commit) is None
        or not source_license.strip()
    ):
        raise ValueError("scene_configuration_component_package_input_invalid")
    driver_source = source / relative_driver
    if (
        driver_source.is_symlink()
        or not driver_source.is_file()
        or not driver_source.stat().st_mode & 0o111
    ):
        raise ValueError("scene_configuration_component_package_driver_invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    installed = False
    try:
        _copy_source_tree(source, staging)
        files = [
            {
                "relative_path": path.relative_to(staging).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "executable": bool(path.stat().st_mode & 0o111),
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        value: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "immutable_component_ready",
            "adapter_id": identity.adapter_id,
            "adapter_version": identity.version,
            "capability": identity.capability,
            "source_identity": {
                "repository": source_repository.strip(),
                "commit": source_commit,
                "license": source_license.strip(),
                "scene_specific_source": False,
            },
            "driver_protocol": (
                "task_evaluation_scene_configuration_component_driver.v1"
            ),
            "driver_entrypoint": relative_driver.as_posix(),
            "network_policy": (
                "disabled"
                if identity.adapter_id == "simready_native_import_qualification"
                else "provider_and_openai_api"
            ),
            "secrets_via_files_only": True,
            "raw_secret_values_in_argv_or_logs": False,
            "files": files,
            "package_digest": "",
        }
        value["package_digest"] = canonical_digest(
            value, digest_field="package_digest"
        )
        manifest = staging / f"{SCHEMA_VERSION}.json"
        manifest.write_text(canonical_json(value) + "\n", encoding="utf-8")
        manifest.chmod(0o444)
        for path in sorted(
            (item for item in staging.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            path.chmod(0o555)
        publish_staged_immutable_directory(
            staging=staging,
            destination=destination,
            manifest_name=manifest.name,
            output_exists_code="scene_configuration_component_package_output_exists",
        )
        installed = True
        validate_scene_configuration_component_package(
            root=destination,
            expected_adapter_id=adapter_id,
        )
        return value
    except Exception:
        owned = destination if installed else staging
        if owned.exists() and not owned.is_symlink():
            for path in sorted(
                owned.rglob("*"), key=lambda item: len(item.parts), reverse=True
            ):
                if path.is_dir():
                    path.chmod(0o700)
            owned.chmod(0o700)
            shutil.rmtree(owned)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--driver-entrypoint", required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    value = build_scene_configuration_component_package(
        adapter_id=args.adapter_id,
        source_root=args.source_root,
        driver_entrypoint=args.driver_entrypoint,
        source_repository=args.source_repository,
        source_commit=args.source_commit,
        source_license=args.source_license,
        output_root=args.output_root,
    )
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
