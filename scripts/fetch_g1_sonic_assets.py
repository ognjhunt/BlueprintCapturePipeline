#!/usr/bin/env python3
"""Fetch or verify the pinned default GEAR-SONIC ONNX pair for G1 rehearsal.

This stages model bytes only. It does not approve model rights, start a policy,
launch a simulator, or allocate a GPU. Run downloads only under the applicable
external-model acquisition authorization; --verify-only stays offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_INVENTORY = (
    Path(__file__).resolve().parents[1]
    / "configs/g1_sonic_default_asset_inventory.v1.json"
)
SOURCE_REPOSITORY = "https://huggingface.co/nvidia/GEAR-SONIC"
EXPECTED_FILES = {
    "encoder": "model_encoder.onnx",
    "decoder": "model_decoder.onnx",
}


class _HTTPSRedirectsOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        if urllib.parse.urlsplit(new_url).scheme.lower() != "https":
            raise ValueError("g1_sonic_insecure_redirect")
        return super().redirect_request(request, response, code, message, headers, new_url)


def _open_https(url: str):
    if urllib.parse.urlsplit(url).scheme.lower() != "https":
        raise ValueError("g1_sonic_insecure_source")
    return urllib.request.build_opener(_HTTPSRedirectsOnly()).open(url, timeout=180)


def _identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _inventory(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("g1_sonic_inventory_missing_or_symlink")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("g1_sonic_inventory_invalid")
    files = value.get("files")
    if (
        value.get("schema_version") != "g1_sonic_asset_inventory.v1"
        or value.get("source_repository") != SOURCE_REPOSITORY
        or re.fullmatch(r"[0-9a-f]{40}", str(value.get("source_revision"))) is None
        or value.get("variant_id") != "default"
        or value.get("model_license") != "NVIDIA Open Model License"
        or value.get("rights_review_required") is not True
        or not isinstance(files, list)
        or len(files) != 2
        or {row.get("role") for row in files if isinstance(row, dict)} != set(EXPECTED_FILES)
    ):
        raise ValueError("g1_sonic_inventory_invalid")
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("g1_sonic_inventory_file_invalid")
        relative = PurePosixPath(str(row.get("path") or ""))
        digest = row.get("sha256")
        size = row.get("size_bytes")
        if (
            row.get("path") != EXPECTED_FILES[row["role"]]
            or relative.is_absolute()
            or ".." in relative.parts
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise ValueError("g1_sonic_inventory_file_invalid")
    return value


def stage_sonic_assets(
    *, inventory_path: Path, output_dir: Path, verify_only: bool = False
) -> dict[str, Any]:
    inventory = _inventory(inventory_path)
    if not output_dir.is_absolute() or output_dir.is_symlink():
        raise ValueError("g1_sonic_output_directory_invalid")
    verified = []
    for row in inventory["files"]:
        destination = output_dir / row["path"]
        if any(parent.is_symlink() for parent in (destination, *destination.parents) if parent != Path("/")):
            raise ValueError("g1_sonic_output_symlink_forbidden")
        expected = (row["sha256"], row["size_bytes"])
        if destination.exists():
            if not destination.is_file() or _identity(destination) != expected:
                raise ValueError("g1_sonic_existing_file_identity_mismatch:" + row["role"])
        else:
            if verify_only:
                raise ValueError("g1_sonic_file_missing:" + row["role"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = (
                SOURCE_REPOSITORY
                + "/resolve/"
                + inventory["source_revision"]
                + "/"
                + urllib.parse.quote(row["path"], safe="")
            )
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=".g1-sonic-", dir=destination.parent, delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    with _open_https(url) as response:
                        if urllib.parse.urlparse(response.geturl()).scheme != "https":
                            raise ValueError("g1_sonic_insecure_redirect")
                        digest = hashlib.sha256()
                        size = 0
                        while block := response.read(1024 * 1024):
                            size += len(block)
                            if size > expected[1]:
                                raise ValueError("g1_sonic_download_exceeds_pinned_size")
                            digest.update(block)
                            stream.write(block)
                if (digest.hexdigest(), size) != expected:
                    raise ValueError("g1_sonic_download_identity_mismatch")
                os.link(temporary, destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        verified.append({
            "role": row["role"],
            "path": str(destination),
            "sha256": "sha256:" + expected[0],
            "size_bytes": expected[1],
        })
    return {
        "status": "sonic_asset_bytes_verified",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": inventory["source_revision"],
        "variant_id": "default",
        "inventory_file_sha256": "sha256:" + _identity(inventory_path)[0],
        "rights_review_required": True,
        "inference_executed": False,
        "files": verified,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    receipt = stage_sonic_assets(
        inventory_path=args.inventory,
        output_dir=args.output_dir,
        verify_only=args.verify_only,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
