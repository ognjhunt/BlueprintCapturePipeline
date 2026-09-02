"""Conservative reclamation for unreferenced control-plane CAS blobs.

Only direct children of an explicitly supplied ``sha256`` directory are ever
eligible.  A blob is reclaimable when its name is its SHA-256 digest, it is an
ordinary non-symlink file, its link count is exactly one, its bytes still match
its name, and it is older than the grace period.  The link-count rule makes the
prepared-reference and compiled-episode hardlinks implicit pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "control_plane_storage_gc.v1"
EXECUTE_ACK = "reap-unreferenced-content"
DEFAULT_MINIMUM_AGE_SECONDS = 24 * 60 * 60
_DIGEST_NAME = re.compile(r"[0-9a-f]{64}\Z")


class ControlPlaneStorageGCError(RuntimeError):
    """A storage root or requested mutation was unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_gc_manifest(
    *,
    content_store_roots: Sequence[str | Path],
    minimum_age_seconds: int = DEFAULT_MINIMUM_AGE_SECONDS,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    if (
        not content_store_roots
        or not isinstance(minimum_age_seconds, int)
        or isinstance(minimum_age_seconds, bool)
        or minimum_age_seconds < 0
    ):
        raise ControlPlaneStorageGCError("control_plane_storage_gc_input_invalid")
    roots: list[Path] = []
    candidates: list[dict[str, Any]] = []
    retained: dict[str, int] = {
        "linked": 0,
        "young": 0,
        "unsafe_or_unverified": 0,
    }
    observed_at = now()
    for raw_root in content_store_roots:
        raw = Path(raw_root).expanduser()
        if raw.is_symlink():
            raise ControlPlaneStorageGCError(
                "control_plane_storage_gc_root_unsafe"
            )
        root = raw.resolve(strict=True)
        if not root.is_dir() or root.name != "sha256" or root in roots:
            raise ControlPlaneStorageGCError(
                "control_plane_storage_gc_root_unsafe"
            )
        roots.append(root)
        for path in sorted(root.iterdir()):
            try:
                stat = path.lstat()
            except OSError:
                retained["unsafe_or_unverified"] += 1
                continue
            if (
                path.is_symlink()
                or not path.is_file()
                or _DIGEST_NAME.fullmatch(path.name) is None
            ):
                retained["unsafe_or_unverified"] += 1
                continue
            if stat.st_nlink != 1:
                retained["linked"] += 1
                continue
            age = max(0.0, observed_at - stat.st_mtime)
            if age < minimum_age_seconds:
                retained["young"] += 1
                continue
            if _sha256(path) != path.name:
                retained["unsafe_or_unverified"] += 1
                continue
            candidates.append(
                {
                    "root": str(root),
                    "digest": "sha256:" + path.name,
                    "size_bytes": stat.st_size,
                    "age_seconds": int(age),
                    "observed_link_count": stat.st_nlink,
                }
            )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run",
        "minimum_age_seconds": minimum_age_seconds,
        "root_count": len(roots),
        "candidate_count": len(candidates),
        "candidate_bytes": sum(row["size_bytes"] for row in candidates),
        "candidates": candidates,
        "retained_counts": retained,
        "evidence_roots_scanned": False,
        "release_or_worktree_roots_scanned": False,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    return manifest


def apply_gc_manifest(
    manifest: dict[str, Any], *, ack: str
) -> dict[str, Any]:
    if (
        ack != EXECUTE_ACK
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("manifest_digest")
        != canonical_digest(manifest, digest_field="manifest_digest")
    ):
        raise ControlPlaneStorageGCError(
            "control_plane_storage_gc_apply_not_authorized"
        )
    removed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in manifest.get("candidates") or []:
        root = Path(str(row.get("root") or ""))
        digest = str(row.get("digest") or "").removeprefix("sha256:")
        path = root / digest
        try:
            stat = path.lstat()
            safe = (
                root.name == "sha256"
                and path.parent == root
                and _DIGEST_NAME.fullmatch(path.name) is not None
                and not path.is_symlink()
                and path.is_file()
                and stat.st_nlink == 1
                and stat.st_size == row.get("size_bytes")
                and _sha256(path) == digest
            )
            if not safe:
                raise OSError("candidate changed after dry run")
            path.unlink()
        except OSError:
            skipped.append(
                {"digest": "sha256:" + digest, "reason": "candidate_changed"}
            )
        else:
            removed.append(
                {"digest": "sha256:" + digest, "size_bytes": stat.st_size}
            )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "applied",
        "source_manifest_digest": manifest["manifest_digest"],
        "removed_count": len(removed),
        "removed_bytes": sum(row["size_bytes"] for row in removed),
        "removed": removed,
        "skipped": skipped,
        "evidence_removed": False,
        "release_or_worktree_removed": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(
        result, digest_field="result_digest"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-store-root", action="append", required=True)
    parser.add_argument(
        "--minimum-age-seconds",
        type=int,
        default=DEFAULT_MINIMUM_AGE_SECONDS,
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--ack", default="")
    args = parser.parse_args(argv)
    manifest = build_gc_manifest(
        content_store_roots=args.content_store_root,
        minimum_age_seconds=args.minimum_age_seconds,
    )
    result = (
        apply_gc_manifest(manifest, ack=args.ack) if args.apply else manifest
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ControlPlaneStorageGCError",
    "EXECUTE_ACK",
    "SCHEMA_VERSION",
    "apply_gc_manifest",
    "build_gc_manifest",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
