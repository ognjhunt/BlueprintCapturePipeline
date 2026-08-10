"""Install one verified native-task released-source packet without dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
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
    python_executable: str | Path | None = None,
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
        "pip_install_command": [],
        "pip_returncode": None,
        "pip_stdout_tail": "",
        "pip_stderr_tail": "",
        "isaac_sim_link": None,
        "all_sources_verified_before_install": False,
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
        command = [
            str(python_executable or sys.executable),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--no-build-isolation",
            *[str(path) for path in install_roots],
        ]
        result["install_roots"] = [str(path) for path in install_roots]
        result["pip_install_command"] = command
        completed = run_command(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        result["pip_returncode"] = completed.returncode
        result["pip_stdout_tail"] = completed.stdout[-16_000:]
        result["pip_stderr_tail"] = completed.stderr[-16_000:]
        if completed.returncode != 0:
            result["blockers"].append("native_task_runtime_source_pip_install_failed")
            return _persist(result_path, result)
        result["status"] = "completed"
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
