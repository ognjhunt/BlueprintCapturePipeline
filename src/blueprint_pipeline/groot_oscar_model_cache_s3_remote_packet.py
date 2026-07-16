"""Build the closed typed packet for CPU model-cache preparation and S3 copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, write_json
from .groot_oscar_model_cache_s3_remote_executor import (
    CANARY_VERIFICATION_NAME,
    EXECUTION_RESULT_NAME,
    PACKET_SCHEMA_VERSION,
    RUNTIME_CACHE_ROOT,
    TRANSPORT_RESULT_NAME,
    VERIFICATION_ROOT,
)
from .groot_oscar_model_cache_wheelhouse import plan_model_cache_wheelhouse
from .groot_oscar_runpod_s3_model_cache import (
    DEFAULT_REMOTE_PREFIX,
    RUNPOD_S3_VOLUME_DATA_CENTER_IDS,
)
from .groot_oscar_runpod_carrier_volume import MIN_CARRIER_VOLUME_GIB


SCHEMA_VERSION = "groot_oscar_model_cache_s3_remote_packet_manifest.v1"
PACKET_KIND = "model_cache_s3"
PACKET_DIRNAME = "groot_oscar_model_cache_s3_remote"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SAFE_NONCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{7,127}")
_SAFE_WHEEL = re.compile(r"[A-Za-z0-9_.+-]+[.]whl")
_CONTEXT_MODULES = (
    "__init__.py",
    "common.py",
    "groot_oscar_infrastructure_admission.py",
    "groot_oscar_model_cache.py",
    "groot_oscar_model_cache_s3_remote_executor.py",
    "groot_oscar_runpod_carrier_volume.py",
    "groot_oscar_runpod_s3_model_cache.py",
    "paid_resource_admission.py",
)
_ENTRYPOINT = """from blueprint_pipeline.groot_oscar_model_cache_s3_remote_executor import execute_remote_packet
raise SystemExit(0 if execute_remote_packet()["status"] == "completed" else 2)
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("model_cache_dependency_manifest_not_object")
    return payload


def _source_identity(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _copy_regular(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"model_cache_packet_source_not_regular:{source}")
    ensure_dir(destination.parent)
    shutil.copyfile(source, destination)
    return {
        "path": destination.as_posix(),
        "sha256": _sha256(destination),
        "bytes": destination.stat().st_size,
    }


def prepare_remote_model_cache_packet(
    *,
    output_dir: str | Path,
    repo_root: str | Path,
    source_commit: str,
    source_patch_sha256: str,
    source_worktree_dirty: bool,
    volume_evidence: Mapping[str, Any],
    volume_watchdog_handoff: Mapping[str, Any],
    allocation_nonce: str,
    data_center_id: str,
    dependency_wheelhouse: str | Path,
    dependency_manifest_path: str | Path,
    runtime_source_release_image_ref: str = "",
    carrier_image_ref: str = "",
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Create a fresh archive from an explicit allowlist and locked wheels."""

    root = Path(repo_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    packet_dir = output / PACKET_DIRNAME
    tarball = output / "groot_oscar_model_cache_s3_remote_packet.tar.gz"
    manifest_path = output / "groot_oscar_model_cache_s3_remote_packet_manifest.json"
    ensure_dir(output)
    if packet_dir.exists() or tarball.exists() or manifest_path.exists():
        raise ValueError("model_cache_packet_output_already_exists")
    packet_dir.mkdir(mode=0o700)
    context_package = packet_dir / "context/src/blueprint_pipeline"
    wheelhouse_target = packet_dir / "wheelhouse"
    ensure_dir(context_package)
    ensure_dir(wheelhouse_target)
    blockers: list[str] = []
    if _HEX40.fullmatch(source_commit) is None:
        blockers.append("model_cache_packet_source_commit_invalid")
    if source_patch_sha256 != hashlib.sha256(b"").hexdigest():
        blockers.append("model_cache_packet_clean_patch_digest_invalid")
    if _SAFE_NONCE.fullmatch(allocation_nonce) is None:
        blockers.append("model_cache_packet_allocation_nonce_invalid")
    if data_center_id not in RUNPOD_S3_VOLUME_DATA_CENTER_IDS:
        blockers.append("model_cache_packet_data_center_invalid")
    try:
        actual_commit, actual_dirty = _source_identity(root)
    except (OSError, subprocess.CalledProcessError):
        actual_commit, actual_dirty = "", True
        blockers.append("model_cache_packet_source_git_identity_unavailable")
    if source_commit != actual_commit:
        blockers.append("model_cache_packet_source_commit_not_exact_head")
    if source_worktree_dirty != actual_dirty:
        blockers.append("model_cache_packet_source_dirty_claim_mismatch")
    if source_worktree_dirty or actual_dirty:
        blockers.append("model_cache_packet_requires_clean_source_worktree")
    if volume_evidence.get("schema_version") != (
        "groot_oscar_runpod_network_volume_evidence.v1"
    ) or volume_evidence.get("status") != "verified":
        blockers.append("model_cache_packet_volume_evidence_unverified")
    if (
        volume_evidence.get("data_center_id") != data_center_id
        or volume_evidence.get("allocation_nonce") != allocation_nonce
        or volume_evidence.get("allocation_name_verified") is not True
    ):
        blockers.append("model_cache_packet_volume_identity_unverified")
    if (
        volume_watchdog_handoff.get("schema_version")
        != "groot_oscar_model_volume_watchdog_handoff.v1"
        or volume_watchdog_handoff.get("status")
        != "storage_preparation_watchdog_armed"
        or volume_watchdog_handoff.get("volume_id") != volume_evidence.get("id")
    ):
        blockers.append("model_cache_packet_volume_watchdog_unverified")

    context_rows: list[dict[str, Any]] = []
    source_package = root / "src/blueprint_pipeline"
    for name in _CONTEXT_MODULES:
        source = source_package / name
        destination = context_package / name
        try:
            row = _copy_regular(source, destination)
        except ValueError:
            blockers.append(f"model_cache_packet_context_source_invalid:{name}")
            continue
        row["path"] = destination.relative_to(packet_dir).as_posix()
        context_rows.append(row)
    entrypoint = packet_dir / "remote_entrypoint.py"
    entrypoint.write_text(_ENTRYPOINT, encoding="utf-8")

    dependency_manifest = _load_object(Path(dependency_manifest_path).expanduser().resolve())
    dependency_rows = dependency_manifest.get("wheels")
    dependency_rows = dependency_rows if isinstance(dependency_rows, list) else []
    if dependency_manifest.get("schema_version") != "blueprint_python_wheelhouse.v1":
        blockers.append("model_cache_packet_dependency_manifest_schema_invalid")
    if (
        dependency_manifest.get("python_version") != "3.12"
        or dependency_manifest.get("implementation") != "cpython"
        or not isinstance(dependency_manifest.get("platform_tags"), list)
        or "manylinux_2_17_x86_64" not in dependency_manifest.get("platform_tags", [])
        or _HEX64.fullmatch(str(dependency_manifest.get("lockfile_sha256") or ""))
        is None
        or _HEX64.fullmatch(
            str(dependency_manifest.get("requirements_closure_sha256") or "")
        )
        is None
    ):
        blockers.append("model_cache_packet_dependency_runtime_binding_invalid")
    lock_source = root / "uv.lock"
    lock_target = packet_dir / "uv.lock"
    try:
        lock_row = _copy_regular(lock_source, lock_target)
    except ValueError:
        lock_row = {"sha256": ""}
        blockers.append("model_cache_packet_dependency_lock_missing")
    if dependency_manifest.get("lockfile_sha256") != lock_row["sha256"]:
        blockers.append("model_cache_packet_dependency_lock_digest_mismatch")
    try:
        locked_plan = plan_model_cache_wheelhouse(lock_target.read_bytes())
    except (OSError, ValueError, UnicodeDecodeError):
        locked_plan = {"requirements": [], "wheels": []}
        blockers.append("model_cache_packet_dependency_lock_plan_invalid")
    requirements = dependency_manifest.get("requirements")
    requirements = requirements if isinstance(requirements, list) else []
    if not requirements or any(
        not isinstance(row, Mapping)
        or set(row) != {"name", "version"}
        or not str(row.get("name") or "")
        or not str(row.get("version") or "")
        for row in requirements
    ):
        blockers.append("model_cache_packet_dependency_closure_invalid")
    closure_bytes = (
        json.dumps(requirements, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    closure_digest = hashlib.sha256(closure_bytes).hexdigest()
    if dependency_manifest.get("requirements_closure_sha256") != closure_digest:
        blockers.append("model_cache_packet_dependency_closure_digest_mismatch")
    if requirements != locked_plan["requirements"]:
        blockers.append("model_cache_packet_dependency_closure_not_locked")
    locked_wheels = [
        {key: value for key, value in row.items() if key != "url"}
        for row in locked_plan["wheels"]
    ]
    if dependency_rows != locked_wheels:
        blockers.append("model_cache_packet_dependency_wheels_not_locked")
    closure_path = packet_dir / "requirements_closure.json"
    closure_path.write_bytes(closure_bytes)
    wheel_source = Path(dependency_wheelhouse).expanduser().resolve()
    expected_wheels: set[str] = set()
    copied_wheels: list[dict[str, Any]] = []
    for row in dependency_rows:
        if not isinstance(row, Mapping):
            blockers.append("model_cache_packet_dependency_row_invalid")
            continue
        filename = str(row.get("filename") or "")
        digest = str(row.get("sha256") or "")
        if (
            set(row) != {"bytes", "distribution", "filename", "sha256", "version"}
            or _SAFE_WHEEL.fullmatch(filename) is None
            or _HEX64.fullmatch(digest) is None
            or type(row.get("bytes")) is not int
            or int(row.get("bytes") or 0) <= 0
            or not str(row.get("distribution") or "")
            or not str(row.get("version") or "")
        ):
            blockers.append("model_cache_packet_dependency_row_invalid")
            continue
        expected_wheels.add(filename)
        source = wheel_source / filename
        destination = wheelhouse_target / filename
        try:
            copied = _copy_regular(source, destination)
        except ValueError:
            blockers.append(f"model_cache_packet_dependency_missing:{filename}")
            continue
        if copied["sha256"] != digest:
            blockers.append(f"model_cache_packet_dependency_digest_mismatch:{filename}")
        if copied["bytes"] != row.get("bytes"):
            blockers.append(f"model_cache_packet_dependency_size_mismatch:{filename}")
        copied["path"] = destination.relative_to(packet_dir).as_posix()
        copied_wheels.append(copied)
    actual_wheels = {
        path.name
        for path in wheel_source.iterdir()
        if path.is_file() and not path.is_symlink()
    } if wheel_source.is_dir() else set()
    if actual_wheels != expected_wheels or not expected_wheels:
        blockers.append("model_cache_packet_dependency_inventory_mismatch")
    context_manifest_path = packet_dir / "context_manifest.json"
    write_json(
        context_manifest_path,
        {
            "schema_version": "groot_oscar_model_cache_s3_context.v1",
            "files": sorted(context_rows, key=lambda row: row["path"]),
        },
    )
    dependency_manifest_target = packet_dir / "dependency_manifest.json"
    write_json(dependency_manifest_target, dependency_manifest)
    packet = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "packet_kind": PACKET_KIND,
        "source_commit": source_commit,
        "source_patch_sha256": source_patch_sha256,
        "runtime_cache_root": str(RUNTIME_CACHE_ROOT),
        "verification_root": str(VERIFICATION_ROOT),
        "remote_prefix": DEFAULT_REMOTE_PREFIX,
        "data_center_id": data_center_id,
        "allocation_nonce": allocation_nonce,
        "volume_evidence": dict(volume_evidence),
        "volume_watchdog_handoff": dict(volume_watchdog_handoff),
        "runtime_bundle_request": (
            {
                "enabled": True,
                "source_release_image_ref": runtime_source_release_image_ref,
                "carrier_image_ref": carrier_image_ref,
            }
            if runtime_source_release_image_ref or carrier_image_ref
            else {"enabled": False}
        ),
        "context_manifest_sha256": _sha256(context_manifest_path),
        "dependency_manifest_sha256": _sha256(dependency_manifest_target),
        "dependency_lock_sha256": _sha256(lock_target) if lock_target.is_file() else None,
        "requirements_closure_sha256": closure_digest,
        "result_files": [
            TRANSPORT_RESULT_NAME,
            CANARY_VERIFICATION_NAME,
            EXECUTION_RESULT_NAME,
        ],
        "raw_secret_values_recorded": False,
    }
    if bool(runtime_source_release_image_ref) != bool(carrier_image_ref):
        blockers.append("model_cache_packet_runtime_image_refs_incomplete")
    if (
        (runtime_source_release_image_ref or carrier_image_ref)
        and (
            type(volume_evidence.get("size_bytes")) is not int
            or int(volume_evidence.get("size_bytes") or 0)
            < MIN_CARRIER_VOLUME_GIB * 1024**3
        )
    ):
        blockers.append("model_cache_packet_runtime_volume_below_120_gib")
    for label, value in (
        ("runtime_source_release", runtime_source_release_image_ref),
        ("carrier", carrier_image_ref),
    ):
        if value:
            _name, marker, digest = value.rpartition("@sha256:")
            if not marker or _HEX64.fullmatch(digest) is None:
                blockers.append(f"model_cache_packet_{label}_image_not_digest_pinned")
    write_json(packet_dir / "packet.json", packet)
    members = [
        packet_dir / "packet.json",
        context_manifest_path,
        packet_dir / "dependency_manifest.json",
        lock_target,
        closure_path,
        entrypoint,
        *[packet_dir / row["path"] for row in context_rows],
        *[packet_dir / row["path"] for row in copied_wheels],
    ]
    relative_members = sorted(
        {path.relative_to(output).as_posix(): path for path in members}.items()
    )
    archive_member_map = {
        name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for name, path in relative_members
    }
    archive_member_manifest_sha256 = hashlib.sha256(
        json.dumps(archive_member_map, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with tarfile.open(tarball, "x:gz") as archive:
        for arcname, path in relative_members:
            archive.add(path, arcname=arcname, recursive=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "packet_kind": PACKET_KIND,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "status": "blocked" if blockers else "ready",
        "blockers": sorted(set(blockers)),
        "packet_dir": str(packet_dir),
        "tarball_path": str(tarball),
        "tarball_sha256": _sha256(tarball),
        "archive_members": [name for name, _path in relative_members],
        "archive_member_manifest_sha256": archive_member_manifest_sha256,
        "dependency_manifest_sha256": packet["dependency_manifest_sha256"],
        "source_commit": source_commit,
        "source_patch_sha256": source_patch_sha256,
        "source_worktree_dirty": source_worktree_dirty,
        "provider_launch_performed_by_packet": False,
        "provider_volume_id": volume_evidence.get("id"),
        "allocation_nonce": allocation_nonce,
        "data_center_id": data_center_id,
        "fixed_remote_directory": PACKET_DIRNAME,
        "fixed_result_files": packet["result_files"],
        "dependency_wheel_count": len(copied_wheels),
        "runtime_bundle_requested": packet["runtime_bundle_request"]["enabled"],
        "runtime_source_release_image_ref": runtime_source_release_image_ref or None,
        "carrier_image_ref": carrier_image_ref or None,
        "arbitrary_entrypoint_supported": False,
        "raw_secret_values_recorded": False,
    }
    write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-patch-sha256", required=True)
    parser.add_argument("--source-worktree-dirty", action="store_true")
    parser.add_argument("--volume-evidence", required=True)
    parser.add_argument("--volume-watchdog-handoff", required=True)
    parser.add_argument("--allocation-nonce", required=True)
    parser.add_argument("--data-center-id", required=True)
    parser.add_argument("--dependency-wheelhouse", required=True)
    parser.add_argument("--dependency-manifest", required=True)
    parser.add_argument("--runtime-source-release-image-ref", default="")
    parser.add_argument("--carrier-image-ref", default="")
    args = parser.parse_args(argv)
    result = prepare_remote_model_cache_packet(
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        source_commit=args.source_commit,
        source_patch_sha256=args.source_patch_sha256,
        source_worktree_dirty=args.source_worktree_dirty,
        volume_evidence=_load_object(Path(args.volume_evidence)),
        volume_watchdog_handoff=_load_object(Path(args.volume_watchdog_handoff)),
        allocation_nonce=args.allocation_nonce,
        data_center_id=args.data_center_id,
        dependency_wheelhouse=args.dependency_wheelhouse,
        dependency_manifest_path=args.dependency_manifest,
        runtime_source_release_image_ref=args.runtime_source_release_image_ref,
        carrier_image_ref=args.carrier_image_ref,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
