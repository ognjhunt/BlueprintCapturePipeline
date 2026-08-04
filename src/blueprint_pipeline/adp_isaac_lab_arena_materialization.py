"""Byte-bound native-worker materialization for the founder Arena protocol.

The prospective protocol intentionally does not download or execute anything.
This module is the next fail-closed boundary: a future Linux GPU worker must
prove that every source checkout is clean and pinned and that every runtime,
asset, and checkpoint file group was inventoried from local bytes.  Passing
this boundary still does not authorize a policy episode or paid compute.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .adp_founder_sim_protocol import (
    ALTERNATIVE_ID,
    BASELINE_ID,
    PROTOCOL_ID,
    FounderSimProtocolError,
    admit_founder_sim_execution,
    build_founder_sim_protocol,
)
from .adp_isaac_lab_arena_request import build_arena_worker_request
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .groot_n16_arena_policy_runtime import (
    CHECKPOINT_REVISION,
    GROOT_SOURCE_REVISION,
)


SCHEMA_VERSION = "adp_isaac_lab_arena_materialization.v1"
ADMISSION_SCHEMA_VERSION = "adp_isaac_lab_arena_materialization_admission.v1"

SOURCE_REVISIONS = {
    "arena_source": "3c19a3a9e45fc2cc1b64ab8a43047ecac9c0ad4d",
    "isaac_lab_source": "af1bab4dc173ba69b08fab779c14ead61d13fd33",
    "openpi_source": "c23745b5ad24e98f66967ea795a07b2588ed6c79",
    "groot_source": GROOT_SOURCE_REVISION,
}

REQUIRED_BYTE_GROUPS = (
    "isaac_runtime_lock",
    "arena_registry_background",
    "arena_registry_pick_up_object",
    "arena_registry_destination",
    "arena_registry_hdr",
    "arena_droid_embodiment",
    "pi05_checkpoint",
    "groot_checkpoint",
)


class ArenaMaterializationError(ValueError):
    def __init__(self, blockers: Sequence[str]):
        self.blockers = tuple(sorted(set(str(item) for item in blockers if str(item))))
        super().__init__(";".join(self.blockers))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or not relative or ".." in parsed.parts:
        raise ArenaMaterializationError(["materialization_path_unsafe"])
    return relative


def inventory_byte_group(root: Path) -> dict[str, Any]:
    """Hash every file under ``root`` using stable logical paths.

    File symlinks are dereferenced and their target bytes are hashed, which
    supports Hugging Face snapshot layouts.  Directory symlinks are rejected
    because a non-recursive inventory could otherwise silently omit bytes.
    """

    resolved = root.expanduser().resolve()
    if not resolved.exists():
        raise ArenaMaterializationError(["materialization_group_path_missing"])
    files: list[dict[str, Any]] = []
    if resolved.is_file():
        candidates = [(resolved, resolved.parent)]
    elif resolved.is_dir():
        candidates = []
        for directory, directory_names, file_names in os.walk(resolved, followlinks=False):
            directory_path = Path(directory)
            for name in tuple(directory_names):
                candidate = directory_path / name
                if candidate.is_symlink():
                    raise ArenaMaterializationError(["materialization_directory_symlink_forbidden"])
            for name in file_names:
                candidates.append((directory_path / name, resolved))
    else:
        raise ArenaMaterializationError(["materialization_group_path_not_file_or_directory"])

    for path, relative_root in sorted(candidates, key=lambda row: str(row[0])):
        if path.is_symlink() and not path.exists():
            raise ArenaMaterializationError(["materialization_file_symlink_broken"])
        if not path.is_file():
            raise ArenaMaterializationError(["materialization_non_regular_file"])
        size = path.stat().st_size
        logical_path = path.name if path == resolved else _safe_relative_path(path, relative_root)
        files.append(
            {
                "path": logical_path,
                "size_bytes": size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise ArenaMaterializationError(["materialization_group_empty"])
    result: dict[str, Any] = {
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(row["size_bytes"]) for row in files),
    }
    result["inventory_digest"] = canonical_digest(result, digest_field="inventory_digest")
    return result


def _git_output(checkout: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ArenaMaterializationError(["materialization_git_probe_failed"]) from exc
    return completed.stdout.strip()


def inventory_source_checkout(
    checkout: Path, *, expected_revision: str, require_uv_lock_sha256: str | None = None
) -> dict[str, Any]:
    resolved = checkout.expanduser().resolve()
    if not resolved.is_dir():
        raise ArenaMaterializationError(["materialization_source_checkout_missing"])
    revision = _git_output(resolved, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise ArenaMaterializationError(["materialization_source_revision_mismatch"])
    if _git_output(resolved, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ArenaMaterializationError(["materialization_source_checkout_dirty"])
    tree = _git_output(resolved, "rev-parse", "HEAD^{tree}")
    submodules = _git_output(resolved, "submodule", "status", "--recursive")
    if any(row.startswith(("-", "+", "U")) for row in submodules.splitlines() if row):
        raise ArenaMaterializationError(["materialization_source_submodule_unresolved"])
    result: dict[str, Any] = {
        "revision": revision,
        "git_tree": tree,
        "clean": True,
        "recursive_submodule_status": submodules.splitlines(),
    }
    if require_uv_lock_sha256 is not None:
        lock_path = resolved / "uv.lock"
        if not lock_path.is_file() or _sha256_file(lock_path) != require_uv_lock_sha256:
            raise ArenaMaterializationError(["materialization_arena_uv_lock_mismatch"])
        result["uv_lock_sha256"] = require_uv_lock_sha256
    result["source_digest"] = canonical_digest(result, digest_field="source_digest")
    return result


def build_materialization_receipt(
    *,
    source_checkouts: Mapping[str, Path],
    byte_group_roots: Mapping[str, Path],
    isaac_sim_version: str,
) -> dict[str, Any]:
    protocol = build_founder_sim_protocol()
    worker_request = build_arena_worker_request(protocol)
    blockers: list[str] = []
    if set(source_checkouts) != set(SOURCE_REVISIONS):
        blockers.append("materialization_source_groups_not_exact")
    if set(byte_group_roots) != set(REQUIRED_BYTE_GROUPS):
        blockers.append("materialization_byte_groups_not_exact")
    expected_sim_version = protocol["scene"]["simulator_stack"]["isaac_sim_version"]
    if isaac_sim_version != expected_sim_version:
        blockers.append("materialization_isaac_sim_version_mismatch")
    if blockers:
        raise ArenaMaterializationError(blockers)

    arena_lock_sha256 = protocol["scene"]["simulator_stack"]["isaac_lab_arena"]["uv_lock_sha256"]
    sources = {
        name: inventory_source_checkout(
            Path(source_checkouts[name]),
            expected_revision=revision,
            require_uv_lock_sha256=arena_lock_sha256 if name == "arena_source" else None,
        )
        for name, revision in SOURCE_REVISIONS.items()
    }
    byte_groups = {
        name: inventory_byte_group(Path(byte_group_roots[name])) for name in REQUIRED_BYTE_GROUPS
    }
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_from_local_worker_bytes",
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": protocol["protocol_digest"],
        "schedule_digest": protocol["schedule"]["schedule_digest"],
        "worker_request_digest": worker_request["worker_request_digest"],
        "runtime": {
            "isaac_sim_version": isaac_sim_version,
            "physics_backend": "PhysX",
            "renderer": "Isaac RTX",
            "candidate_execution_started": False,
        },
        "sources": sources,
        "byte_groups": byte_groups,
        "candidate_bindings": {
            "baseline": {
                "candidate_id": BASELINE_ID,
                "openpi_revision": SOURCE_REVISIONS["openpi_source"],
                "checkpoint_inventory_digest": byte_groups["pi05_checkpoint"]["inventory_digest"],
            },
            "alternative": {
                "candidate_id": ALTERNATIVE_ID,
                "groot_source_revision": GROOT_SOURCE_REVISION,
                "checkpoint_revision": CHECKPOINT_REVISION,
                "checkpoint_inventory_digest": byte_groups["groot_checkpoint"]["inventory_digest"],
            },
        },
        "candidate_jobs_authorized": False,
        "paid_compute_authorized": False,
        "physical_execution_authorized": False,
    }
    receipt["materialization_digest"] = canonical_digest(
        receipt, digest_field="materialization_digest"
    )
    return receipt


def verify_materialization_receipt(
    receipt: Mapping[str, Any],
    *,
    source_checkouts: Mapping[str, Path],
    byte_group_roots: Mapping[str, Path],
) -> dict[str, Any]:
    """Re-inventory local bytes and reject a stale or edited receipt."""

    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ArenaMaterializationError(["materialization_receipt_runtime_invalid"])
    expected = build_materialization_receipt(
        source_checkouts=source_checkouts,
        byte_group_roots=byte_group_roots,
        isaac_sim_version=str(runtime.get("isaac_sim_version") or ""),
    )
    if dict(receipt) != expected:
        raise ArenaMaterializationError(["materialization_receipt_not_current_for_worker"])
    return expected


def admit_materialized_worker(
    *,
    founder_execution_admission: Mapping[str, Any],
    receipt: Mapping[str, Any],
    source_checkouts: Mapping[str, Path],
    byte_group_roots: Mapping[str, Path],
) -> dict[str, Any]:
    """Admit native control canaries, never candidate episodes or spend."""

    protocol = build_founder_sim_protocol()
    approval = founder_execution_admission.get("approval")
    if not isinstance(approval, Mapping):
        raise ArenaMaterializationError(["materialization_founder_admission_missing"])
    try:
        expected_founder_admission = admit_founder_sim_execution(protocol, approval)
    except FounderSimProtocolError as exc:
        raise ArenaMaterializationError(
            [f"materialization_{blocker}" for blocker in exc.blockers]
        ) from exc
    if dict(founder_execution_admission) != expected_founder_admission:
        raise ArenaMaterializationError(["materialization_founder_admission_not_canonical"])
    verified = verify_materialization_receipt(
        receipt,
        source_checkouts=source_checkouts,
        byte_group_roots=byte_group_roots,
    )
    admission: dict[str, Any] = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "status": "materialized_pending_native_controls",
        "protocol_digest": protocol["protocol_digest"],
        "founder_execution_admission_digest": founder_execution_admission.get(
            "execution_admission_digest"
        ),
        "materialization_digest": verified["materialization_digest"],
        "native_control_canaries_authorized": True,
        "candidate_jobs_authorized": False,
        "paid_compute_authorized": False,
        "physical_execution_authorized": False,
    }
    admission["admission_digest"] = canonical_digest(admission, digest_field="admission_digest")
    return admission


def _parse_paths(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in parsed:
            raise ArenaMaterializationError(["materialization_cli_path_binding_invalid"])
        parsed[name] = Path(raw_path)
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--byte-group", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--isaac-sim-version", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = build_materialization_receipt(
        source_checkouts=_parse_paths(args.source),
        byte_group_roots=_parse_paths(args.byte_group),
        isaac_sim_version=args.isaac_sim_version,
    )
    write_json(Path(args.output).expanduser().resolve(), receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADMISSION_SCHEMA_VERSION",
    "REQUIRED_BYTE_GROUPS",
    "SCHEMA_VERSION",
    "SOURCE_REVISIONS",
    "ArenaMaterializationError",
    "admit_materialized_worker",
    "build_materialization_receipt",
    "inventory_byte_group",
    "inventory_source_checkout",
    "verify_materialization_receipt",
]
