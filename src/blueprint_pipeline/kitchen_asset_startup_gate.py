"""Content-addressed kitchen asset startup gate (P1-4, Isaac worker startup reliability).

The Isaac worker startup canary ships no kitchen assets, so a canary pass does not
prove the ~1.24 GB ``Collected_KitchenRoom`` tree (185 files, main USD
``Collected_KitchenRoom/KitchenRoom.usd``) is present and extractable on a worker.
This module is a separate content-addressed asset-readiness stage that must fail
BEFORE simulator or policy startup when the bundle is incomplete.

Claim boundary: a passing gate proves provider-side asset presence ONLY. It does
not prove Isaac can load the scene, nor robot placement, nor task success.

Fail-closed rules:
- an invalid or wrong-schema inventory is a blocker, never a soft pass;
- archive safety screening (traversal, unsafe links, device/fifo members) happens
  BEFORE any extraction;
- reuse of a previously extracted tree is allowed only against a prior PASSING
  gate manifest that recorded the exact same archive digest — and the reused tree
  is still fully re-verified file-by-file;
- the ``kitchen_asset_startup_gate.json`` artifact is written on PASS and FAIL.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from .common import ensure_dir, read_json, utc_now_iso, write_json

GATE_SCHEMA_VERSION = "kitchen_asset_startup_gate.v1"
INVENTORY_SCHEMA_VERSION = "kitchen_asset_inventory_checksums.v1"
GATE_ARTIFACT_NAME = "kitchen_asset_startup_gate.json"
DEFAULT_MAIN_USD_RELPATH = "Collected_KitchenRoom/KitchenRoom.usd"
MATERIALIZED_DIR_NAME = "materialized"
# Extraction needs the tree plus tar bookkeeping/filesystem slack; refuse below this.
DISK_HEADROOM_FACTOR = 1.1
# Append an extraction progress record every N regular files (plus a final record).
PROGRESS_RECORD_EVERY_FILES = 25
# Cap detail lists in the artifact so a mangled bundle cannot bloat the manifest.
MAX_RECORDED_FAILURES = 20

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_CLAIM_BOUNDARY = {
    "asset_presence_proven_only": True,
    "scene_load_proven": False,
    "placement_proven": False,
    "task_success_proven": False,
    "note": (
        "A passing gate proves only that the expected kitchen asset bundle is "
        "present and content-address-intact on this worker before simulator "
        "startup. It does not prove Isaac can load the scene, nor robot "
        "placement, nor task success."
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_unsafe_relpath(name: str) -> bool:
    """True for absolute paths, backslash tricks, or any ``..`` traversal component."""
    if not name or name.startswith("/") or "\\" in name:
        return True
    return any(part == ".." for part in PurePosixPath(name).parts)


def _link_escapes(member: tarfile.TarInfo) -> bool:
    """True when a symlink/hardlink member points outside the extraction root."""
    target = str(member.linkname or "")
    if not target or target.startswith("/") or "\\" in target:
        return True
    if member.issym():
        base = PurePosixPath(member.name).parent
        joined = (base / target).as_posix()
    else:
        # Hardlink targets are archive-root relative.
        joined = target
    depth = 0
    for part in PurePosixPath(joined).parts:
        if part == "..":
            depth -= 1
            if depth < 0:
                return True
        elif part != ".":
            depth += 1
    return False


# ---------------------------------------------------------------------------
# Expected-inventory contract (kitchen_asset_inventory_checksums.v1).
# ---------------------------------------------------------------------------


def build_asset_inventory(
    tree_root: Path,
    *,
    main_usd_relpath: str = DEFAULT_MAIN_USD_RELPATH,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    """Build the content-addressed inventory of a materialized kitchen asset tree.

    Walks regular files deterministically (sorted POSIX relpaths, symlinks
    excluded), sha256-hashing each. Raises ``ValueError`` when the tree is
    missing or the main USD is not present — a producer must never emit an
    inventory for an incomplete bundle.
    """
    root = Path(tree_root)
    if not root.is_dir():
        raise ValueError(f"tree root is not a directory: {root}")
    files: list[dict[str, Any]] = []
    total_bytes = 0
    entries = sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix())
    for path in entries:
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        files.append({"path": rel, "sha256": _sha256_file(path), "bytes": size})
        total_bytes += size
    if main_usd_relpath not in {item["path"] for item in files}:
        raise ValueError(f"main USD missing from tree: {main_usd_relpath}")
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "main_usd": main_usd_relpath,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "archive_sha256": _sha256_file(Path(archive_path)) if archive_path else None,
        "files": files,
    }


def load_asset_inventory(path: Path) -> dict[str, Any]:
    """Load and strictly validate an inventory; raise ``ValueError`` on any defect."""
    payload = read_json(Path(path))
    if payload.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        raise ValueError(
            f"inventory schema_version must be {INVENTORY_SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    main_usd = payload.get("main_usd")
    if not isinstance(main_usd, str) or _is_unsafe_relpath(main_usd):
        raise ValueError(f"inventory main_usd invalid: {main_usd!r}")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("inventory files must be a non-empty list")
    seen_paths: set[str] = set()
    summed_bytes = 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("inventory file entry must be an object")
        rel = item.get("path")
        sha = item.get("sha256")
        size = item.get("bytes")
        if not isinstance(rel, str) or _is_unsafe_relpath(rel):
            raise ValueError(f"inventory file path invalid: {rel!r}")
        if rel in seen_paths:
            raise ValueError(f"inventory duplicate path: {rel}")
        seen_paths.add(rel)
        if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
            raise ValueError(f"inventory sha256 invalid for {rel}")
        if not _is_nonneg_int(size):
            raise ValueError(f"inventory bytes invalid for {rel}")
        summed_bytes += size
    if payload.get("file_count") != len(files) or not _is_nonneg_int(payload.get("file_count")):
        raise ValueError("inventory file_count does not match files list")
    total_bytes = payload.get("total_bytes")
    if total_bytes != summed_bytes or not _is_nonneg_int(total_bytes):
        raise ValueError("inventory total_bytes does not match files list")
    if main_usd not in seen_paths:
        raise ValueError("inventory main_usd not present in files list")
    digest = payload.get("archive_sha256")
    if digest is not None and (not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest)):
        raise ValueError("inventory archive_sha256 invalid")
    return payload


# ---------------------------------------------------------------------------
# Archive safety screening (runs BEFORE any extraction).
# ---------------------------------------------------------------------------


def verify_archive_safety(archive_path: Path) -> dict[str, Any]:
    """Screen a tar/tar.gz for unsafe members and compute the archive sha256.

    Rejects absolute-path and ``..`` traversal names, symlinks/hardlinks pointing
    outside the extraction root, and device/fifo/other non-file members. Nothing
    is extracted here.
    """
    archive = Path(archive_path)
    blockers: set[str] = set()
    flagged: list[dict[str, Any]] = []
    member_count = 0
    archive_sha256: str | None = None

    def _flag(blocker: str, member: tarfile.TarInfo, reason: str) -> None:
        blockers.add(blocker)
        if len(flagged) < MAX_RECORDED_FAILURES:
            flagged.append({"member": member.name, "reason": reason})

    if not archive.is_file():
        blockers.add("kitchen_archive_missing")
    else:
        archive_sha256 = _sha256_file(archive)
        try:
            with tarfile.open(archive, mode="r:*") as tar:
                for member in tar:
                    member_count += 1
                    if _is_unsafe_relpath(member.name):
                        _flag(
                            "kitchen_archive_path_traversal",
                            member,
                            "absolute_or_parent_traversal_name",
                        )
                    elif member.issym() or member.islnk():
                        if _link_escapes(member):
                            _flag(
                                "kitchen_archive_unsafe_link",
                                member,
                                "link_target_escapes_extraction_root",
                            )
                    elif not (member.isfile() or member.isdir()):
                        _flag(
                            "kitchen_archive_unsupported_member_type",
                            member,
                            "device_fifo_or_unknown_member",
                        )
        except (tarfile.TarError, OSError, EOFError):
            blockers.add("kitchen_archive_unreadable")
    return {
        "archive_path": str(archive),
        "archive_sha256": archive_sha256,
        "member_count": member_count,
        "flagged_members": flagged,
        "blockers": sorted(blockers),
        "safe": not blockers,
    }


# ---------------------------------------------------------------------------
# Tree verification against the inventory.
# ---------------------------------------------------------------------------


def _verify_tree(tree_root: Path, inventory: Mapping[str, Any]) -> dict[str, Any]:
    blockers: set[str] = set()
    failures: list[dict[str, Any]] = []

    def _fail(blocker: str, rel: str, reason: str) -> None:
        blockers.add(blocker)
        if len(failures) < MAX_RECORDED_FAILURES:
            failures.append({"path": rel, "reason": reason})

    expected = {str(item["path"]): item for item in inventory["files"]}
    disk_sizes: dict[str, int] = {}
    if not tree_root.is_dir():
        blockers.add("kitchen_asset_tree_missing")
    else:
        for path in sorted(tree_root.rglob("*"), key=lambda p: p.as_posix()):
            rel = path.relative_to(tree_root).as_posix()
            if path.is_symlink():
                _fail("kitchen_asset_unexpected_extra_files", rel, "symlink_not_in_contract")
            elif path.is_file():
                disk_sizes[rel] = path.stat().st_size

    verified_file_count = 0
    verified_total_bytes = 0
    for rel in sorted(expected):
        entry = expected[rel]
        if rel not in disk_sizes:
            _fail("kitchen_asset_file_missing", rel, "missing_from_tree")
            continue
        if disk_sizes[rel] != int(entry["bytes"]):
            _fail("kitchen_asset_checksum_mismatch", rel, "size_mismatch")
            continue
        if _sha256_file(tree_root / rel) != entry["sha256"]:
            _fail("kitchen_asset_checksum_mismatch", rel, "sha256_mismatch")
            continue
        verified_file_count += 1
        verified_total_bytes += disk_sizes[rel]

    for rel in sorted(set(disk_sizes) - set(expected)):
        _fail("kitchen_asset_unexpected_extra_files", rel, "not_in_inventory")
    if len(disk_sizes) != int(inventory["file_count"]):
        blockers.add("kitchen_asset_file_count_mismatch")
    if sum(disk_sizes.values()) != int(inventory["total_bytes"]):
        blockers.add("kitchen_asset_total_bytes_mismatch")

    main_rel = str(inventory["main_usd"])
    main_usd_present = main_rel in disk_sizes and (tree_root / main_rel).is_file()
    if not main_usd_present:
        blockers.add("kitchen_main_usd_missing")
    return {
        "blockers": blockers,
        "failures": failures,
        "verified_file_count": verified_file_count,
        "verified_total_bytes": verified_total_bytes,
        "main_usd_present": main_usd_present,
    }


# ---------------------------------------------------------------------------
# The gate.
# ---------------------------------------------------------------------------


def _free_bytes(probe: Callable[[Path], Any], path: Path) -> int:
    value = probe(path)
    return int(getattr(value, "free", value))


def _default_free_disk_probe(path: Path) -> int:
    return shutil.disk_usage(path).free


def run_kitchen_asset_startup_gate(
    *,
    expected_inventory_path: str | Path,
    out_dir: str | Path,
    tree_root: str | Path | None = None,
    archive_path: str | Path | None = None,
    reuse_manifest_path: str | Path | None = None,
    free_disk_probe: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Fail-closed asset-presence gate; must run before simulator/policy startup.

    Exactly one of ``tree_root`` / ``archive_path`` is required. The
    ``kitchen_asset_startup_gate.json`` artifact is always written into
    ``out_dir`` — pass or fail.
    """
    out = Path(out_dir)
    ensure_dir(out)
    probe = free_disk_probe or _default_free_disk_probe
    blockers: set[str] = set()
    result: dict[str, Any] = {
        "schema_version": GATE_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "blocked",
        "blockers": [],
        "expected_inventory_path": str(expected_inventory_path),
        "input_mode": None,
        "materialized_tree": None,
        "verified_file_count": 0,
        "verified_total_bytes": 0,
        "main_usd_present": False,
        "main_usd_relpath": None,
        "archive_sha256": None,
        "archive_safety": None,
        "reuse": {"reused": False, "reuse_reason": None},
        "disk": None,
        "progress": [],
        "verification_failures": [],
        "claim_boundary": dict(_CLAIM_BOUNDARY),
    }

    def _finish() -> dict[str, Any]:
        result["blockers"] = sorted(blockers)
        result["status"] = "completed" if not blockers else "blocked"
        write_json(out / GATE_ARTIFACT_NAME, result)
        return result

    if (tree_root is None) == (archive_path is None):
        blockers.add("kitchen_gate_input_invalid")
        return _finish()

    try:
        inventory = load_asset_inventory(Path(expected_inventory_path))
    except (OSError, ValueError):
        blockers.add("kitchen_inventory_invalid")
        return _finish()
    result["main_usd_relpath"] = inventory["main_usd"]

    tree: Path | None = None
    if archive_path is not None:
        result["input_mode"] = "archive"
        archive = Path(archive_path)
        safety = verify_archive_safety(archive)
        result["archive_safety"] = safety
        result["archive_sha256"] = safety["archive_sha256"]
        blockers.update(safety["blockers"])
        if blockers:
            # Unsafe or unreadable archive: fail BEFORE any extraction.
            return _finish()
        expected_digest = inventory.get("archive_sha256")
        if expected_digest and safety["archive_sha256"] != expected_digest:
            blockers.add("kitchen_archive_digest_mismatch")
            return _finish()

        reuse_reason: str | None = None
        if reuse_manifest_path is not None:
            try:
                prior = read_json(Path(reuse_manifest_path))
            except (OSError, ValueError):
                prior = None
            if prior is None:
                reuse_reason = "prior_manifest_unreadable"
            elif (
                prior.get("schema_version") != GATE_SCHEMA_VERSION
                or prior.get("status") != "completed"
            ):
                reuse_reason = "prior_manifest_not_passed"
            elif prior.get("archive_sha256") != safety["archive_sha256"]:
                # The worker holds a tree from a DIFFERENT bundle; refusing and
                # extracting fresh silently would hide a provisioning bug.
                result["reuse"] = {
                    "reused": False,
                    "reuse_reason": "prior_archive_digest_mismatch",
                }
                blockers.add("kitchen_asset_reuse_digest_mismatch")
                return _finish()
            else:
                prior_tree_raw = str(prior.get("materialized_tree") or "")
                if prior_tree_raw and Path(prior_tree_raw).is_dir():
                    tree = Path(prior_tree_raw)
                    reuse_reason = "prior_gate_passed_same_archive_digest"
                else:
                    reuse_reason = "prior_tree_missing"
        result["reuse"] = {"reused": tree is not None, "reuse_reason": reuse_reason}

        if tree is None:
            dest = out / MATERIALIZED_DIR_NAME
            free_before = _free_bytes(probe, out)
            required = int(int(inventory["total_bytes"]) * DISK_HEADROOM_FACTOR)
            disk: dict[str, Any] = {
                "free_bytes_before": free_before,
                "required_bytes": required,
                "free_bytes_after": None,
            }
            result["disk"] = disk
            if free_before < required:
                blockers.add("kitchen_asset_insufficient_disk")
                return _finish()
            ensure_dir(dest)
            files_extracted = 0
            bytes_extracted = 0

            def _progress_record() -> dict[str, Any]:
                return {
                    "at": utc_now_iso(),
                    "files_extracted": files_extracted,
                    "bytes_extracted": bytes_extracted,
                }

            try:
                with tarfile.open(archive, mode="r:*") as tar:
                    for member in tar:
                        tar.extract(member, path=dest, filter="data")
                        if member.isfile():
                            files_extracted += 1
                            bytes_extracted += member.size
                            if files_extracted % PROGRESS_RECORD_EVERY_FILES == 0:
                                result["progress"].append(_progress_record())
            except (tarfile.TarError, OSError) as exc:
                result["progress"].append(_progress_record())
                result["extract_error"] = str(exc)
                disk["free_bytes_after"] = _free_bytes(probe, out)
                blockers.add("kitchen_asset_extract_failed")
                return _finish()
            result["progress"].append(_progress_record())
            disk["free_bytes_after"] = _free_bytes(probe, out)
            tree = dest
    else:
        result["input_mode"] = "tree"
        tree = Path(tree_root)  # type: ignore[arg-type]
        if reuse_manifest_path is not None:
            result["reuse"] = {
                "reused": False,
                "reuse_reason": "not_applicable_for_tree_input",
            }

    result["materialized_tree"] = str(tree)
    verification = _verify_tree(tree, inventory)
    blockers.update(verification["blockers"])
    result["verified_file_count"] = verification["verified_file_count"]
    result["verified_total_bytes"] = verification["verified_total_bytes"]
    result["main_usd_present"] = verification["main_usd_present"]
    result["verification_failures"] = verification["failures"]
    return _finish()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Content-addressed kitchen asset startup gate: verify the "
            "Collected_KitchenRoom bundle on a worker before simulator startup."
        )
    )
    parser.add_argument("--expected-inventory", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tree-root", default=None)
    parser.add_argument("--archive", default=None)
    parser.add_argument("--reuse-manifest", default=None)
    args = parser.parse_args(argv)

    result = run_kitchen_asset_startup_gate(
        expected_inventory_path=Path(args.expected_inventory),
        out_dir=Path(args.out_dir),
        tree_root=Path(args.tree_root) if args.tree_root else None,
        archive_path=Path(args.archive) if args.archive else None,
        reuse_manifest_path=Path(args.reuse_manifest) if args.reuse_manifest else None,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "blockers": result["blockers"],
                "verified_file_count": result["verified_file_count"],
                "main_usd_present": result["main_usd_present"],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
