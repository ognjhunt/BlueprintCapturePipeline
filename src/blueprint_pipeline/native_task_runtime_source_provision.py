"""Install one verified native-task released-source packet without dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import sysconfig
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_runtime_source_packet import (
    NativeTaskRuntimeSourcePacketError,
    verify_native_task_runtime_source_packet,
)


SCHEMA_VERSION = "native_task_runtime_source_provisioning.v1"
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
TOP_LEVEL_PACKAGES = (
    "cloudpickle",
    "farama_notifications",
    "gymnasium",
    "lazy_loader",
    "packaging",
    "prettytable",
    "typing_extensions",
    "wcwidth",
    "h5py",
    "isaaclab",
    "isaaclab_physx",
    "isaaclab_assets",
    "isaaclab_tasks",
    "isaaclab_teleop",
    "isaaclab_arena",
)


def _extract_runtime_dependency_wheels(
    *,
    extraction_root: Path,
    wheel_rows: Sequence[dict[str, Any]],
    destination: Path,
    runtime_python_tag: str,
    runtime_platform_tags: Sequence[str],
) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    installed: list[dict[str, Any]] = []
    for row in wheel_rows:
        wheel = extraction_root / str(row["archive_path"])
        if not wheel.is_file() or _sha256(wheel) != row.get("sha256"):
            raise RuntimeError("native_task_runtime_dependency_wheel_identity_mismatch")
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            if any(
                Path(name).is_absolute()
                or ".." in Path(name).parts
                or ".data" in Path(name).parts
                for name in names
            ):
                raise RuntimeError("native_task_runtime_dependency_wheel_layout_invalid")
            wheel_metadata_names = [
                name for name in names if name.endswith(".dist-info/WHEEL")
            ]
            if len(wheel_metadata_names) != 1:
                raise RuntimeError("native_task_runtime_dependency_wheel_metadata_invalid")
            wheel_metadata = archive.read(wheel_metadata_names[0]).decode("utf-8")
            pure_python = bool(row.get("pure_python"))
            expected_root = f"Root-Is-Purelib: {str(pure_python).lower()}"
            wheel_tag = str(row.get("wheel_tag") or "")
            if expected_root not in wheel_metadata or f"Tag: {wheel_tag}" not in wheel_metadata:
                raise RuntimeError("native_task_runtime_dependency_wheel_platform_contract_invalid")
            if not pure_python:
                interpreter, abi, platform_tag = wheel_tag.split("-", 2)
                if (
                    interpreter != runtime_python_tag
                    or abi != runtime_python_tag
                    or platform_tag not in runtime_platform_tags
                ):
                    raise RuntimeError("native_task_runtime_dependency_binary_wheel_incompatible")
            for name in names:
                if name.endswith("/"):
                    continue
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                data = archive.read(name)
                if target.exists() and target.read_bytes() != data:
                    raise RuntimeError(
                        "native_task_runtime_dependency_wheel_member_conflict"
                    )
                target.write_bytes(data)
        installed.append(
            {
                "package": row["package"],
                "version": row["version"],
                "wheel_sha256": row["sha256"],
                "license_spdx": row["license_spdx"],
                "pure_python": row["pure_python"],
                "wheel_tag": row["wheel_tag"],
            }
        )
    return installed


def _runtime_wheel_compatibility() -> tuple[str, tuple[str, ...]]:
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        return python_tag, ()
    libc_name, libc_version = platform.libc_ver()
    tags: list[str] = []
    if libc_name.lower() == "glibc":
        try:
            major, minor = (int(value) for value in libc_version.split(".")[:2])
        except (TypeError, ValueError):
            major, minor = (0, 0)
        if (major, minor) >= (2, 28):
            tags.append("manylinux_2_28_x86_64")
    return python_tag, tuple(tags)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _persist(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    result["receipt_digest"] = canonical_digest(
        result, digest_field="receipt_digest"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def provision_native_task_runtime_sources(
    *,
    source_receipt_path: str | Path,
    source_packet_path: str | Path,
    extraction_dir: str | Path,
    output_path: str | Path,
    simulator_root: str | Path = "/isaac-sim",
    site_packages_dir: str | Path | None = None,
    python_executable: str | Path | None = None,
    runtime_python_tag: str | None = None,
    runtime_platform_tags: Sequence[str] | None = None,
    run_command: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    """Verify, extract, and install every pinned package in one pip operation."""

    receipt_path = Path(source_receipt_path).expanduser().resolve()
    packet_path = Path(source_packet_path).expanduser().resolve()
    destination = Path(extraction_dir).expanduser().resolve()
    result_path = Path(output_path).expanduser().resolve()
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "source_receipt_path": str(receipt_path),
        "source_packet_path": str(packet_path),
        "source_packet_sha256": _sha256(packet_path) if packet_path.is_file() else None,
        "extraction_dir": str(destination),
        "python_executable": str(python_executable or sys.executable),
        "install_roots": [],
        "installation_method": "verified_source_roots_pth",
        "path_file": None,
        "path_file_sha256": None,
        "package_path_probe_command": [],
        "package_path_probe_returncode": None,
        "package_path_probe_stdout": "",
        "package_path_probe_stderr": "",
        "isaac_sim_link": None,
        "runtime_dependency_target": None,
        "runtime_dependencies_installed": [],
        "all_sources_verified_before_install": False,
        "source_packages_made_importable": False,
        "dependencies_installed": False,
        "candidate_policy_queried": False,
        "scene_bytes_processed": False,
        "blockers": [],
        "receipt_digest": "",
    }
    try:
        verified = verify_native_task_runtime_source_packet(
            receipt_path, packet_path_override=packet_path
        )
        result["source_receipt_digest"] = verified["receipt_digest"]
        result["all_sources_verified_before_install"] = True
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(packet_path) as archive:
            archive.extractall(destination)
        install_roots = [destination / relative for relative in verified["install_roots"]]
        missing = [str(path) for path in install_roots if not path.is_dir()]
        if missing:
            result["blockers"].append(
                "native_task_runtime_source_install_roots_missing:" + ",".join(missing)
            )
            return _persist(result_path, result)
        isaaclab_root = destination / "runtime_sources/isaaclab"
        link = isaaclab_root / "_isaac_sim"
        simulator = Path(simulator_root).expanduser().resolve()
        if os.path.lexists(link):
            if not link.is_symlink() or Path(os.readlink(link)).resolve() != simulator:
                result["blockers"].append(
                    "native_task_runtime_source_isaac_sim_link_conflict"
                )
                return _persist(result_path, result)
        else:
            link.symlink_to(simulator, target_is_directory=True)
        result["isaac_sim_link"] = {"path": str(link), "target": str(simulator)}
        dependency_target = destination / "runtime_python_dependencies"
        detected_python_tag, detected_platform_tags = _runtime_wheel_compatibility()
        installed_dependencies = _extract_runtime_dependency_wheels(
            extraction_root=destination,
            wheel_rows=verified["runtime_dependency_wheels"],
            destination=dependency_target,
            runtime_python_tag=runtime_python_tag or detected_python_tag,
            runtime_platform_tags=(
                tuple(runtime_platform_tags)
                if runtime_platform_tags is not None
                else detected_platform_tags
            ),
        )
        result["runtime_dependency_target"] = str(dependency_target)
        result["runtime_dependencies_installed"] = installed_dependencies
        site_packages = Path(
            site_packages_dir or sysconfig.get_paths()["purelib"]
        ).expanduser().resolve()
        site_packages.mkdir(parents=True, exist_ok=True)
        path_file = site_packages / "blueprint_native_task_runtime_sources.pth"
        path_file.write_text(
            "".join(
                f"{path}\n" for path in (dependency_target, *install_roots)
            ),
            encoding="utf-8",
        )
        result["path_file"] = str(path_file)
        result["path_file_sha256"] = _sha256(path_file)
        package_probe = (
            "import importlib.util,json;"
            f"names={list(TOP_LEVEL_PACKAGES)!r};"
            "found={name:importlib.util.find_spec(name) is not None for name in names};"
            "print(json.dumps(found,sort_keys=True));"
            "raise SystemExit(0 if all(found.values()) else 3)"
        )
        command = [
            str(python_executable or sys.executable),
            "-I",
            "-c",
            package_probe,
        ]
        result["install_roots"] = [str(path) for path in install_roots]
        result["package_path_probe_command"] = command
        completed = run_command(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        result["package_path_probe_returncode"] = completed.returncode
        result["package_path_probe_stdout"] = completed.stdout[-16_000:]
        result["package_path_probe_stderr"] = completed.stderr[-16_000:]
        if completed.returncode != 0:
            result["blockers"].append(
                "native_task_runtime_source_package_path_probe_failed"
            )
            return _persist(result_path, result)
        result["status"] = "completed"
        result["source_packages_made_importable"] = True
        result["dependencies_installed"] = True
        return _persist(result_path, result)
    except NativeTaskRuntimeSourcePacketError as exc:
        result["blockers"].extend(exc.errors)
        return _persist(result_path, result)
    except Exception as exc:  # noqa: BLE001 - retained typed provisioning failure
        result["blockers"].append(
            f"native_task_runtime_source_provisioning_exception:{type(exc).__name__}"
        )
        result["exception"] = str(exc)
        return _persist(result_path, result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-receipt", required=True)
    parser.add_argument("--source-packet", required=True)
    parser.add_argument("--extraction-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--simulator-root", default="/isaac-sim")
    args = parser.parse_args(argv)
    result = provision_native_task_runtime_sources(
        source_receipt_path=args.source_receipt,
        source_packet_path=args.source_packet,
        extraction_dir=args.extraction_dir,
        output_path=args.output,
        simulator_root=args.simulator_root,
    )
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":  # pragma: no cover - CLI seam
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "main", "provision_native_task_runtime_sources"]
