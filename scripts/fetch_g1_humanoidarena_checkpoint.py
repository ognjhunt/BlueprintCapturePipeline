#!/usr/bin/env python3
"""Fetch or verify exact HumanoidArena G1 candidate bytes from ModelScope.

This writes only local checkpoint artifacts. It does not launch a policy
server, simulator, or paid provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any


MODEL_BASE = "https://modelscope.cn/models/Twang2026/HumanoidArena_models/resolve/master/"
DEFAULT_INVENTORY = (
    Path(__file__).resolve().parents[1]
    / "configs/g1_humanoidarena_checkpoint_inventory.v1.json"
)


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return digest.hexdigest(), size


def _candidate(inventory: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    rows = inventory.get("candidates")
    if (
        inventory.get("schema_version") != "g1_humanoidarena_checkpoint_inventory.v1"
        or not isinstance(rows, list)
        or sum(row.get("candidate_id") == candidate_id for row in rows if isinstance(row, dict)) != 1
    ):
        raise ValueError("g1_checkpoint_candidate_or_inventory_invalid")
    candidate = next(row for row in rows if row.get("candidate_id") == candidate_id)
    folder = PurePosixPath(str(candidate.get("subdirectory") or ""))
    files = candidate.get("files")
    if (
        not folder.parts or folder.is_absolute() or ".." in folder.parts
        or not isinstance(files, list) or not files
    ):
        raise ValueError("g1_checkpoint_candidate_or_inventory_invalid")
    seen: set[str] = set()
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("g1_checkpoint_file_inventory_invalid")
        relative = PurePosixPath(str(row.get("path") or ""))
        digest = row.get("sha256")
        size = row.get("size_bytes")
        if (
            not relative.parts or relative.is_absolute() or ".." in relative.parts
            or relative.as_posix() in seen
            or not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or isinstance(size, bool) or not isinstance(size, int) or size <= 0
        ):
            raise ValueError("g1_checkpoint_file_inventory_invalid")
        seen.add(relative.as_posix())
    inventory_digest = "sha256:" + hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    if candidate.get("inventory_digest") != inventory_digest:
        raise ValueError("g1_checkpoint_candidate_inventory_digest_invalid")
    return candidate


def materialize_candidate(
    *, inventory_path: Path, candidate_id: str, output_dir: Path, verify_only: bool = False
) -> dict[str, Any]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    candidate = _candidate(inventory, candidate_id)
    folder = PurePosixPath(candidate["subdirectory"])
    if output_dir.is_symlink():
        raise ValueError("g1_checkpoint_output_symlink_forbidden")
    verified = []
    for row in candidate["files"]:
        relative = PurePosixPath(row["path"])
        destination = output_dir.joinpath(*folder.parts, *relative.parts)
        if any(parent.is_symlink() for parent in (destination, *destination.parents) if parent != Path("/")):
            raise ValueError("g1_checkpoint_output_symlink_forbidden")
        expected = (row["sha256"], row["size_bytes"])
        if destination.exists():
            if not destination.is_file() or _sha256_and_size(destination) != expected:
                raise ValueError(f"g1_checkpoint_existing_file_identity_mismatch:{relative}")
        else:
            if verify_only:
                raise ValueError(f"g1_checkpoint_file_missing:{relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            path = folder / relative
            url = MODEL_BASE + urllib.parse.quote(path.as_posix(), safe="/")
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=".g1-checkpoint-", dir=destination.parent, delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    with urllib.request.urlopen(url, timeout=180) as response:
                        if urllib.parse.urlparse(response.geturl()).scheme != "https":
                            raise ValueError("g1_checkpoint_insecure_redirect")
                        digest = hashlib.sha256()
                        size = 0
                        while block := response.read(1024 * 1024):
                            size += len(block)
                            if size > expected[1]:
                                raise ValueError("g1_checkpoint_download_exceeds_pinned_size")
                            digest.update(block)
                            stream.write(block)
                if (digest.hexdigest(), size) != expected:
                    raise ValueError("g1_checkpoint_download_identity_mismatch")
                os.link(temporary, destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        verified.append({
            "relative_path": (folder / relative).as_posix(),
            "sha256": "sha256:" + expected[0],
            "size_bytes": expected[1],
        })
    return {
        "status": "checkpoint_bytes_verified",
        "candidate_id": candidate_id,
        "candidate_inventory_digest": candidate["inventory_digest"],
        "policy_role": candidate.get("policy_role"),
        "inventory_file_sha256": "sha256:" + _sha256_and_size(inventory_path)[0],
        "files": verified,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    receipt = materialize_candidate(
        inventory_path=args.inventory,
        candidate_id=args.candidate,
        output_dir=args.output_dir,
        verify_only=args.verify_only,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
