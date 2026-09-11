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
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "task_evaluation_scene_configuration_python_wheelhouse.v1"
MANIFEST_NAME = f"{SCHEMA_VERSION}.json"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_MEMBER_BYTES = 256 * 1024**2
_MAX_TOTAL_BYTES = 768 * 1024**2
DEFAULT_RUNTIME_PROFILE = "base"
RUNTIME_PROFILE_ROOTS = {
    "base": ("openai-agents", "usd-core"),
    "astra_asset_authoring": ("openai-agents", "usd-core", "build123d", "langgraph", "trimesh", "pillow"),
}
RUNTIME_PROFILE_IMPORTS = {
    "base": (),
    "astra_asset_authoring": ("agents", "pxr.Usd", "build123d", "OCP", "langgraph.graph", "trimesh", "PIL.Image"),
}
_ASTRA_SOURCE_ROOT = Path(__file__).resolve().parent.parent
_ASTRA_STAGE_MODULES = ("blueprint_pipeline.astra_cad_skill_runtime",
                        "blueprint_pipeline.task_evaluation_scene_configuration_astra_driver",
                        "blueprint_pipeline.task_object_astra_authoring",
                        "blueprint_pipeline.task_object_simready_packaging")
RUNTIME_PROFILE_PLATFORM_TAGS = {
    "base": ("manylinux_2_17_x86_64", "manylinux2014_x86_64", "manylinux_2_28_x86_64", "manylinux_2_35_x86_64"),
    "astra_asset_authoring": tuple(f"manylinux_2_{minor}_x86_64" for minor in range(17, 36))
    + ("manylinux2014_x86_64",),
}


class TaskEvaluationSceneConfigurationPythonRuntimeError(ValueError):
    """The shipped provider dependency closure was unsafe or incompatible."""



def runtime_profile_roots(profile: str) -> tuple[str, ...]:
    if profile not in RUNTIME_PROFILE_ROOTS:
        raise TaskEvaluationSceneConfigurationPythonRuntimeError("scene_configuration_python_runtime_profile_invalid")
    return RUNTIME_PROFILE_ROOTS[profile]


def _compatible_astra_wheel(filename: str) -> bool:
    parts = filename.removesuffix(".whl").rsplit("-", 3)
    if len(parts) != 4:
        return False
    _, python_tags, abi_tags, platform_tags = parts
    for platform_tag in platform_tags.split("."):
        if platform_tag not in {"any", *RUNTIME_PROFILE_PLATFORM_TAGS["astra_asset_authoring"]}:
            continue
        for python_tag in python_tags.split("."):
            if "none" in abi_tags.split(".") and python_tag in {"py3", "py312", "cp312"}:
                return True
            if platform_tag != "any" and python_tag == "cp312" and "cp312" in abi_tags.split("."):
                return True
            stable = re.fullmatch(r"cp3([0-9]+)", python_tag)
            if platform_tag != "any" and stable and 2 <= int(stable.group(1)) <= 12 and "abi3" in abi_tags.split("."):
                return True
    return False


def validate_runtime_profile_inventory(manifest: Mapping[str, Any], profile: str) -> None:
    roots = runtime_profile_roots(profile)
    if manifest.get("runtime_profile", DEFAULT_RUNTIME_PROFILE) != profile or manifest.get("root_distributions") != list(roots):
        raise TaskEvaluationSceneConfigurationPythonRuntimeError("scene_configuration_python_runtime_profile_mismatch")
    if profile == DEFAULT_RUNTIME_PROFILE:
        return  # Preserve the existing base manifest/validation contract.
    rows, requirements = manifest.get("wheels"), manifest.get("requirements")
    if (manifest.get("required_imports") != list(RUNTIME_PROFILE_IMPORTS[profile])
            or manifest.get("platform_tags") != list(RUNTIME_PROFILE_PLATFORM_TAGS[profile])
            or not isinstance(rows, list) or not isinstance(requirements, list)):
        raise TaskEvaluationSceneConfigurationPythonRuntimeError("scene_configuration_python_profile_inventory_invalid")
    inventory = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TaskEvaluationSceneConfigurationPythonRuntimeError("scene_configuration_python_profile_inventory_invalid")
        name, version, filename = row.get("distribution"), row.get("version"), str(row.get("filename") or "")
        if (not isinstance(name, str) or not isinstance(version, str) or name in inventory
                or not filename.startswith(name.replace("-", "_") + "-" + version + "-")
                or not filename.endswith(".whl") or not _compatible_astra_wheel(filename)):
            raise TaskEvaluationSceneConfigurationPythonRuntimeError("scene_configuration_python_profile_wheel_abi_or_identity_invalid")
        inventory[name] = version
    if (not set(roots) <= inventory.keys()
            or requirements != [{"name": name, "version": version} for name, version in sorted(inventory.items())]):
        raise TaskEvaluationSceneConfigurationPythonRuntimeError("scene_configuration_python_profile_inventory_incomplete")


def _validate_astra_imports(root: Path) -> None:
    # -I/-S prevents accidental satisfaction by the host's installed site packages.
    code = """import importlib, json, pathlib, sys, sysconfig
root = pathlib.Path(sys.argv[1]).resolve()
source = pathlib.Path(sys.argv[3]).resolve()
sys.path[:0] = [str(root), str(source)]
for name in json.loads(sys.argv[2]):
    module = importlib.import_module(name)
    origin = getattr(module, '__file__', None)
    if not origin or not pathlib.Path(origin).resolve().is_relative_to(root):
        raise ImportError('sealed_import_origin_mismatch:' + name)
for name in json.loads(sys.argv[4]):
    module = importlib.import_module(name)
    origin = getattr(module, '__file__', None)
    if not origin or not pathlib.Path(origin).resolve().is_relative_to(source):
        raise ImportError('shipped_stage_import_origin_mismatch:' + name)
stdlib = pathlib.Path(sysconfig.get_path('stdlib')).resolve()
for name, module in list(sys.modules.items()):
    origin = getattr(module, '__file__', None)
    if not origin:
        continue
    path = pathlib.Path(origin).resolve()
    if path.is_relative_to(root) or path.is_relative_to(source):
        continue
    if path.is_relative_to(stdlib) and not {'site-packages', 'dist-packages'} & set(path.parts):
        continue
    raise ImportError('global_python_module_not_admitted:' + name)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", code, str(root), json.dumps(RUNTIME_PROFILE_IMPORTS["astra_asset_authoring"]),
             str(_ASTRA_SOURCE_ROOT), json.dumps(_ASTRA_STAGE_MODULES)],
            cwd=root, env={"PATH": os.defpath}, capture_output=True, text=True, timeout=90, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TaskEvaluationSceneConfigurationPythonRuntimeError("scene_configuration_python_import_preflight_failed") from exc
    if result.returncode:
        raise TaskEvaluationSceneConfigurationPythonRuntimeError(
            "scene_configuration_python_import_preflight_failed:" + result.stderr[-1500:]
        )


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
    profile: str = DEFAULT_RUNTIME_PROFILE,
) -> Path:
    """Verify every wheel and extract it into one read-only import root."""

    roots = runtime_profile_roots(profile)
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
        or manifest.get("root_distributions") != list(roots)
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
    validate_runtime_profile_inventory(manifest, profile)
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
                    if total > (_MAX_TOTAL_BYTES if profile == DEFAULT_RUNTIME_PROFILE else 2 * 1024**3):
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
        if profile == "astra_asset_authoring":
            _validate_astra_imports(staging)
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
    parser.add_argument("--profile", choices=tuple(RUNTIME_PROFILE_ROOTS), default=DEFAULT_RUNTIME_PROFILE)
    args = parser.parse_args(argv)
    try:
        path = materialize_scene_configuration_python_runtime(
            wheelhouse_root=args.wheelhouse_root,
            output_root=args.output_root,
            profile=args.profile,
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
    "DEFAULT_RUNTIME_PROFILE",
    "RUNTIME_PROFILE_ROOTS",
    "RUNTIME_PROFILE_IMPORTS",
    "RUNTIME_PROFILE_PLATFORM_TAGS",
    "runtime_profile_roots",
    "validate_runtime_profile_inventory",
    "SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationPythonRuntimeError",
    "materialize_scene_configuration_python_runtime",
]
