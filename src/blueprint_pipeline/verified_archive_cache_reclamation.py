"""Reclaim redundant extraction bytes while retaining their verified archive.

Callers establish terminal ownership and protect live references before planning.
This utility never removes an archive, receipt, log, unmatched file or hardlink.
Plans are dry-run first; apply repeats byte and inode checks and refuses open files.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import zipfile

from .decision_evidence_contracts import canonical_digest


def _sha(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _member_sha(archive, name):
    with archive.open(name) as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _process_root():
    return Path("/proc")


def plan_archive_cache_reclamation(
    *, archive_path: Path, extraction_root: Path, protected_paths=(), minimum_size_bytes=1024 * 1024
):
    archive_path, extraction_root = Path(archive_path), Path(extraction_root)
    if (
        not archive_path.is_absolute()
        or not extraction_root.is_absolute()
        or archive_path.is_symlink()
        or extraction_root.is_symlink()
        or not archive_path.is_file()
        or not extraction_root.is_dir()
        or archive_path.is_relative_to(extraction_root)
    ):
        raise ValueError("archive_cache_reclamation_roots_invalid")
    protected = [Path(p).absolute() for p in protected_paths]
    rows, skipped = [], {}
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = {i.filename: i for i in infos}
        if len(names) != len(infos):
            raise ValueError("archive_cache_reclamation_duplicate_members")
        for path in sorted(extraction_root.rglob("*")):
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                continue
            if (
                metadata.st_size < minimum_size_bytes
                or path.suffix.lower() in {".json", ".jsonl", ".log"}
                or any(path == p or path.is_relative_to(p) for p in protected)
            ):
                continue
            relative = path.relative_to(extraction_root).as_posix()
            info = names.get(relative)
            if (
                info is None
                or info.file_size != metadata.st_size
                or stat.S_ISLNK(info.external_attr >> 16)
            ):
                skipped[relative] = "not_an_exact_archive_member"
                continue
            digest = _sha(path)
            if digest != _member_sha(archive, relative):
                skipped[relative] = "archive_member_bytes_differ"
                continue
            rows.append(
                {
                    "relative_path": relative,
                    "size_bytes": metadata.st_size,
                    "inode": metadata.st_ino,
                    "mtime_ns": metadata.st_mtime_ns,
                    "sha256": digest,
                }
            )
    plan = {
        "schema_version": "verified_archive_cache_reclamation.v1",
        "status": "dry_run",
        "archive_path": str(archive_path),
        "archive_sha256": _sha(archive_path),
        "extraction_root": str(extraction_root),
        "candidates": rows,
        "candidate_bytes": sum(r["size_bytes"] for r in rows),
        "skipped": skipped,
        "archive_removed": False,
        "protected_paths": [str(p) for p in protected],
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def apply_archive_cache_reclamation(plan, *, ack):
    if (
        ack != "reclaim-byte-verified-extraction-cache"
        or plan.get("schema_version") != "verified_archive_cache_reclamation.v1"
        or plan.get("status") != "dry_run"
        or plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")
    ):
        raise ValueError("archive_cache_reclamation_plan_invalid")
    archive_path, root = Path(plan["archive_path"]), Path(plan["extraction_root"])
    if (
        archive_path.is_symlink()
        or root.is_symlink()
        or _sha(archive_path) != plan["archive_sha256"]
    ):
        raise ValueError("archive_cache_reclamation_archive_changed")
    targets = []
    for row in plan["candidates"]:
        relative = Path(row["relative_path"])
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or any(p.is_symlink() for p in (path, *path.parents))
        ):
            raise ValueError("archive_cache_reclamation_path_invalid")
        metadata = path.stat()
        if (
            not path.is_file()
            or metadata.st_nlink != 1
            or metadata.st_ino != row["inode"]
            or metadata.st_mtime_ns != row["mtime_ns"]
            or metadata.st_size != row["size_bytes"]
            or _sha(path) != row["sha256"]
        ):
            raise ValueError("archive_cache_reclamation_candidate_changed")
        targets.append(path)
    # On the production Linux host, a descriptor or cwd under the extraction
    # root refuses the entire mutation. No partial deletion precedes this check.
    proc = _process_root()
    if proc.is_dir():
        for process in proc.iterdir():
            if not process.name.isdecimal():
                continue
            links = [process / "cwd"]
            try:
                links.extend((process / "fd").iterdir())
            except FileNotFoundError:
                continue
            except PermissionError as exc:
                raise ValueError("archive_cache_reclamation_reader_visibility_missing") from exc
            for link in links:
                try:
                    value = Path(os.readlink(link))
                except OSError:
                    continue
                if value == root or value.is_relative_to(root):
                    raise ValueError("archive_cache_reclamation_active_reader")
    for path in targets:
        path.unlink()
    result = {
        "schema_version": "verified_archive_cache_reclamation_result.v1",
        "status": "applied",
        "plan_digest": plan["plan_digest"],
        "removed_count": len(targets),
        "removed_bytes": plan["candidate_bytes"],
        "archive_removed": False,
        "archive_sha256": plan["archive_sha256"],
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result
