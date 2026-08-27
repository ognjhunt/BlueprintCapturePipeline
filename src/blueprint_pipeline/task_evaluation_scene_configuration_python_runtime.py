"""Materialize the sealed scene-configuration wheelhouse without network."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "task_evaluation_scene_configuration_python_wheelhouse.v1"
MANIFEST_NAME = f"{SCHEMA_VERSION}.json"
#: Deliberately a second copy of the builder's ``ROOT_DISTRIBUTIONS``. This
#: module runs on the provider and imports nothing from ``blueprint_pipeline``
#: so it can verify the sealed wheelhouse independently of whatever produced
#: it. The cost of that independence is drift, so the two are pinned equal by
#: ``test_provider_runtime_expects_the_same_roots_the_builder_ships``: adding
#: ``usd-core`` to the builder alone left this list refusing every wheelhouse
#: with ``scene_configuration_python_wheelhouse_manifest_invalid``.
EXPECTED_ROOT_DISTRIBUTIONS = ("openai-agents", "usd-core")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_MEMBER_BYTES = 256 * 1024**2
_MAX_TOTAL_BYTES = 768 * 1024**2


class TaskEvaluationSceneConfigurationPythonRuntimeError(ValueError):
    """The shipped provider dependency closure was unsafe or incompatible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any], *, digest_field: str) -> str:
    normalized = dict(value)
    normalized.pop(digest_field, None)
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationPythonRuntimeError(
            "scene_configuration_python_wheelhouse_manifest_invalid"
        ) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationPythonRuntimeError(
            "scene_configuration_python_wheelhouse_manifest_invalid"
        )
    return dict(value)


def _install_relative_path(member: zipfile.ZipInfo) -> PurePosixPath | None:
    raw = member.filename
    relative = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or relative.is_absolute()
        or ".." in relative.parts
        or stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK
    ):
        raise TaskEvaluationSceneConfigurationPythonRuntimeError(
            "scene_configuration_python_wheel_member_invalid"
        )
    parts = relative.parts
    if parts and parts[0].endswith(".data"):
        if len(parts) < 3 or parts[1] not in {"purelib", "platlib"}:
            return None
        relative = PurePosixPath(*parts[2:])
    return relative if relative.parts else None


def materialize_scene_configuration_python_runtime(
    *,
    wheelhouse_root: str | Path,
    output_root: str | Path,
    runtime_python: tuple[int, int] | None = None,
    runtime_platform: str | None = None,
    runtime_machine: str | None = None,
) -> Path:
    """Verify every wheel and extract it into one read-only import root."""

    observed_python = runtime_python or sys.version_info[:2]
    observed_platform = runtime_platform or sys.platform
    observed_machine = (runtime_machine or platform.machine()).lower()
    if (
        tuple(observed_python) != (3, 12)
        or observed_platform != "linux"
        or observed_machine not in {"x86_64", "amd64"}
    ):
        raise TaskEvaluationSceneConfigurationPythonRuntimeError(
            "scene_configuration_python_runtime_platform_mismatch"
        )
    root = Path(wheelhouse_root).resolve()
    destination = Path(output_root).resolve()
    if (
        root.is_symlink()
        or not root.is_dir()
        or destination.exists()
        or destination.parent.is_symlink()
        or not destination.parent.is_dir()
    ):
        raise TaskEvaluationSceneConfigurationPythonRuntimeError(
            "scene_configuration_python_runtime_path_invalid"
        )
    manifest = _read_manifest(root / MANIFEST_NAME)
    rows = manifest.get("wheels")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "ready"
        or manifest.get("python_version") != "3.12"
        or manifest.get("implementation") != "cpython"
        or manifest.get("platform") != "linux-x86_64"
        or manifest.get("root_distributions") != list(EXPECTED_ROOT_DISTRIBUTIONS)
        or manifest.get("sdists_allowed") is not False
        or manifest.get("provider_network_install_required") is not False
        or manifest.get("manifest_digest")
        != _canonical_digest(manifest, digest_field="manifest_digest")
        or not isinstance(rows, list)
        or not rows
    ):
        raise TaskEvaluationSceneConfigurationPythonRuntimeError(
            "scene_configuration_python_wheelhouse_manifest_invalid"
        )
    wheels_root = root / "wheels"
    expected: dict[str, tuple[str, int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TaskEvaluationSceneConfigurationPythonRuntimeError(
                "scene_configuration_python_wheelhouse_inventory_invalid"
            )
        filename = str(row.get("filename") or "")
        digest = str(row.get("sha256") or "")
        size = row.get("size_bytes")
        if (
            not filename
            or Path(filename).name != filename
            or not filename.endswith(".whl")
            or filename in expected
            or _DIGEST.fullmatch(digest) is None
            or type(size) is not int
            or size <= 0
        ):
            raise TaskEvaluationSceneConfigurationPythonRuntimeError(
                "scene_configuration_python_wheelhouse_inventory_invalid"
            )
        expected[filename] = (digest, size)
    observed = {
        path.name
        for path in wheels_root.iterdir()
        if path.is_file() and not path.is_symlink()
    } if wheels_root.is_dir() and not wheels_root.is_symlink() else set()
    if observed != set(expected):
        raise TaskEvaluationSceneConfigurationPythonRuntimeError(
            "scene_configuration_python_wheelhouse_inventory_incomplete"
        )
    staging = destination.parent / f".{destination.name}.staging"
    if staging.exists():
        raise TaskEvaluationSceneConfigurationPythonRuntimeError(
            "scene_configuration_python_runtime_path_invalid"
        )
    staging.mkdir(mode=0o700)
    total = 0
    try:
        for filename, (digest, size) in sorted(expected.items()):
            wheel = wheels_root / filename
            if (
                wheel.is_symlink()
                or wheel.stat().st_size != size
                or _sha256(wheel) != digest
            ):
                raise TaskEvaluationSceneConfigurationPythonRuntimeError(
                    "scene_configuration_python_wheel_invalid:" + filename
                )
            try:
                archive = zipfile.ZipFile(wheel)
            except (OSError, zipfile.BadZipFile) as exc:
                raise TaskEvaluationSceneConfigurationPythonRuntimeError(
                    "scene_configuration_python_wheel_invalid:" + filename
                ) from exc
            with archive:
                for member in archive.infolist():
                    relative = _install_relative_path(member)
                    if relative is None:
                        continue
                    if member.file_size > _MAX_MEMBER_BYTES:
                        raise TaskEvaluationSceneConfigurationPythonRuntimeError(
                            "scene_configuration_python_wheel_expansion_limit_exceeded"
                        )
                    total += member.file_size
                    if total > _MAX_TOTAL_BYTES:
                        raise TaskEvaluationSceneConfigurationPythonRuntimeError(
                            "scene_configuration_python_wheel_expansion_limit_exceeded"
                        )
                    target = staging.joinpath(*relative.parts)
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    body = archive.read(member)
                    if target.exists():
                        if target.is_symlink() or target.read_bytes() != body:
                            raise TaskEvaluationSceneConfigurationPythonRuntimeError(
                                "scene_configuration_python_wheel_member_collision"
                            )
                    else:
                        target.write_bytes(body)
        os.replace(staging, destination)
        for path in sorted(destination.rglob("*")):
            if path.is_symlink():
                raise TaskEvaluationSceneConfigurationPythonRuntimeError(
                    "scene_configuration_python_runtime_symlink_forbidden"
                )
            path.chmod(0o555 if path.is_dir() else 0o444)
        destination.chmod(0o555)
        return destination
    except Exception:
        if staging.exists() and not staging.is_symlink():
            for path in sorted(
                staging.rglob("*"), key=lambda value: len(value.parts), reverse=True
            ):
                path.chmod(0o700 if path.is_dir() else 0o600)
            staging.chmod(0o700)
            shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    try:
        path = materialize_scene_configuration_python_runtime(
            wheelhouse_root=args.wheelhouse_root,
            output_root=args.output_root,
        )
    except (OSError, TaskEvaluationSceneConfigurationPythonRuntimeError) as exc:
        print(f"BLUEPRINT_SCENE_CONFIGURATION_BLOCKED:{exc}", file=sys.stderr)
        return 86
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MANIFEST_NAME",
    "SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationPythonRuntimeError",
    "materialize_scene_configuration_python_runtime",
]
