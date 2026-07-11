#!/usr/bin/env python3
"""Fetch an exact, digest-pinned Isaac asset directory for a sealed image."""

from __future__ import annotations

import argparse
import hashlib
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


def _rows(path: Path) -> list[tuple[str, int, str]]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digest, size, relative = line.split(maxsplit=2)
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or len(digest) != 64:
            raise ValueError(f"invalid pinned asset row: {relative}")
        rows.append((digest, int(size), relative))
    if not rows:
        raise ValueError("pinned asset manifest is empty")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/") + "/"
    if not base_url.startswith("https://"):
        raise ValueError("asset base URL must use HTTPS")
    rows = _rows(args.manifest)
    for expected_digest, expected_size, relative in rows:
        destination = args.output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = base_url + urllib.parse.quote(relative, safe="/")
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(url, timeout=180) as response, destination.open(
            "wb"
        ) as output:
            if response.geturl() != url:
                raise ValueError(f"asset redirect rejected: {relative}")
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > expected_size:
                    raise ValueError(f"asset exceeds pinned size: {relative}")
                digest.update(chunk)
                output.write(chunk)
        if size != expected_size or digest.hexdigest() != expected_digest:
            destination.unlink(missing_ok=True)
            raise ValueError(f"asset digest or size mismatch: {relative}")
    print(f"BLUEPRINT_PINNED_ISAAC_ASSETS_READY files={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
