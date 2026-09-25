#!/usr/bin/env python3
"""Prefetch one pinned G1 model file over verified HTTPS byte ranges.

This never writes into a running campaign's checkpoint directory. Operators
may hardlink the completed digest-verified file before that candidate starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath


MODEL_BASE = "https://modelscope.cn/models/Twang2026/HumanoidArena_models/resolve/master/"


class _HTTPSRedirectsOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        if urllib.parse.urlsplit(new_url).scheme.lower() != "https":
            raise ValueError("g1_prefetch_insecure_redirect")
        return super().redirect_request(request, response, code, message, headers, new_url)


def _pinned_model(inventory_path: Path, candidate_id: str) -> tuple[str, int, str]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    rows = inventory.get("candidates")
    if inventory.get("schema_version") != "g1_humanoidarena_checkpoint_inventory.v1" or not isinstance(rows, list):
        raise ValueError("g1_prefetch_inventory_invalid")
    matches = [row for row in rows if row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError("g1_prefetch_candidate_invalid")
    candidate = matches[0]
    files = [row for row in candidate.get("files", []) if row.get("path") == "model.safetensors"]
    if len(files) != 1:
        raise ValueError("g1_prefetch_model_inventory_invalid")
    row = files[0]
    relative = PurePosixPath(candidate["subdirectory"]) / row["path"]
    if (relative.is_absolute() or ".." in relative.parts
            or not isinstance(row.get("size_bytes"), int) or row["size_bytes"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", ""))):
        raise ValueError("g1_prefetch_model_inventory_invalid")
    return relative.as_posix(), row["size_bytes"], row["sha256"]


def prefetch(*, inventory_path: Path, candidate_id: str, output_dir: Path,
             workers: int = 12, chunk_mib: int = 128) -> dict:
    if not (1 <= workers <= 24 and 8 <= chunk_mib <= 512):
        raise ValueError("g1_prefetch_bounds_invalid")
    relative, expected_size, expected_sha = _pinned_model(inventory_path, candidate_id)
    if (not output_dir.is_absolute() or output_dir.is_symlink()
            or output_dir.resolve() != output_dir):
        raise ValueError("g1_prefetch_output_root_invalid")
    destination = output_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if any(path.is_symlink() for path in (destination, *destination.parents) if path != Path("/")):
        raise ValueError("g1_prefetch_symlink_forbidden")
    if destination.exists():
        if destination.stat().st_size != expected_size or _sha256(destination) != expected_sha:
            raise ValueError("g1_prefetch_existing_identity_mismatch")
        return {"status": "verified", "candidate_id": candidate_id, "path": str(destination)}
    temporary = destination.with_name("." + destination.name + ".parallel-partial")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    url = MODEL_BASE + urllib.parse.quote(relative, safe="/")
    chunk_size = chunk_mib * 1024 * 1024
    intervals = [(start, min(start + chunk_size, expected_size) - 1)
                 for start in range(0, expected_size, chunk_size)]
    opener = urllib.request.build_opener(_HTTPSRedirectsOnly())

    def fetch(start: int, end: int) -> int:
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with opener.open(request, timeout=180) as response:
                    if (response.status != 206
                            or response.headers.get("Content-Range") != f"bytes {start}-{end}/{expected_size}"
                            or urllib.parse.urlsplit(response.geturl()).scheme.lower() != "https"):
                        raise ValueError("g1_prefetch_range_response_invalid")
                    written = 0
                    while written <= end - start:
                        block = response.read(min(1024 * 1024, end - start + 1 - written))
                        if not block:
                            raise ValueError("g1_prefetch_range_truncated")
                        if os.pwrite(descriptor, block, start + written) != len(block):
                            raise ValueError("g1_prefetch_short_write")
                        written += len(block)
                    if response.read(1):
                        raise ValueError("g1_prefetch_range_extra_bytes")
                    return written
            except (OSError, ValueError):
                if attempt == 2:
                    raise
        raise ValueError("g1_prefetch_unreachable")

    try:
        os.ftruncate(descriptor, expected_size)
        total = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, start, end) for start, end in intervals]
            for completed in as_completed(futures):
                total += completed.result()
                print(f"G1_PREFETCH_PROGRESS:{candidate_id}:{total}/{expected_size}", flush=True)
        if total != expected_size:
            raise ValueError("g1_prefetch_size_mismatch")
        os.fsync(descriptor)
        if _sha256(temporary) != expected_sha:
            raise ValueError("g1_prefetch_digest_mismatch")
        os.link(temporary, destination)
        return {"status": "verified", "candidate_id": candidate_id,
                "path": str(destination), "sha256": "sha256:" + expected_sha,
                "size_bytes": expected_size}
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-mib", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(prefetch(inventory_path=args.inventory, candidate_id=args.candidate,
                              output_dir=args.output_dir, workers=args.workers,
                              chunk_mib=args.chunk_mib), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
