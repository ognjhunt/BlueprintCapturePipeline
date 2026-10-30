"""Exact public TRLC checkpoint and legitimately retained offline tokenizer.

Public byte availability is not license authority. The execution registration
separately binds Apache additions, inherited Gemma terms and task permissions.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import time
import math

MODEL_ID = "djkesu/pi05_trlc_rabc_3cam_alpha025_20260319_v2_step10000_merged"
MODEL_REVISION = "10466a133c74018465c6839b7003b700775e4f69"
SOURCE_REVISION = "68479287a784a69be9ce6ad739311d2f11f75ef9"
INTERFACE_ID = "pi05_trlc_joint14.v1"
NORM_SHA256 = "be756f317fa7659062d7745ea831097ebf15c7689efd89a8d75fd8dd0ee4b740"
RELEASE_INVENTORY_PATH = "_catalog_data/trlc_pi05_release.v1.json"
RELEASE_INVENTORY_SHA256 = "c92ec43b64d57ae66a913eef95a9e771be44bd6e33292559a2b94796b1788562"
TOKENIZER_INVENTORY_PATH = "_catalog_data/trlc_pi05_tokenizer.v1.json"
TOKENIZER_INVENTORY_SHA256 = "5020414c0d782bb33cdf1f53b220569c83aff55356d2d920f7741657eb0fe830"
RETAINED_TOKENIZER_DIR = Path("/var/lib/blueprint/task-evaluation-inputs/g1-841757-development/pi-tokenizer-f1eaa13")
TOKENIZER_DIR_ENV = "BLUEPRINT_G1_PI_TOKENIZER_DIR"


def _inventory(relative: str, expected: str) -> dict:
    path = Path(__file__).parent / relative
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
        raise ValueError("trlc_pi05_inventory_invalid")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("trlc_pi05_inventory_changed")
    return json.loads(raw)


def release_inventory() -> dict:
    return _inventory(RELEASE_INVENTORY_PATH, RELEASE_INVENTORY_SHA256)


def tokenizer_inventory() -> dict:
    return _inventory(TOKENIZER_INVENTORY_PATH, TOKENIZER_INVENTORY_SHA256)


def _deadline(value):
    if value is not None and (not math.isfinite(value) or value <= time.time()):
        raise ValueError("trlc_pi05_asset_deadline_exceeded")


def _verify(directory: Path, rows: list[dict], *, deadline_epoch=None, exact=True) -> list[dict]:
    _deadline(deadline_epoch)
    root = Path(directory)
    if (not root.is_absolute() or root.is_symlink() or not root.is_dir()
            or any(parent.is_symlink() for parent in root.parents)):
        raise ValueError("trlc_pi05_asset_directory_unavailable")
    expected = {row["path"] for row in rows}
    def closure():
        if exact and {p.name for p in root.iterdir()} != expected:
            raise ValueError("trlc_pi05_asset_file_closure_changed")
    closure()
    verified = []
    for row in rows:
        path = root / row["path"]
        if (path.is_symlink() or not path.exists() or not stat.S_ISREG(path.stat().st_mode)
                or path.stat().st_size != row["size_bytes"]):
            raise ValueError("trlc_pi05_asset_file_invalid")
        sha = hashlib.sha256()
        blob = hashlib.sha1(usedforsecurity=False)
        blob.update(f"blob {row['size_bytes']}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024**2), b""):
                _deadline(deadline_epoch)
                sha.update(block)
                blob.update(block)
        if (row.get("sha256") and sha.hexdigest() != row["sha256"]
                or row.get("git_blob_sha1") and blob.hexdigest() != row["git_blob_sha1"]):
            raise ValueError("trlc_pi05_asset_bytes_changed")
        verified.append({"path": row["path"], "size_bytes": row["size_bytes"], "sha256": sha.hexdigest()})
    closure()
    return verified


def verify_checkpoint_directory(directory: Path, *, deadline_epoch=None) -> dict:
    return {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "checkpoint_bytes_verified": True,
            "files": _verify(directory, release_inventory()["files"], deadline_epoch=deadline_epoch),
            "policy_inference_executed": False}


def verify_tokenizer_directory(directory: Path, *, deadline_epoch=None, exact=True) -> dict:
    return {"source_revision": tokenizer_inventory()["source_revision"],
            "inventory_sha256": "sha256:" + TOKENIZER_INVENTORY_SHA256,
            "files": _verify(directory, tokenizer_inventory()["files"], deadline_epoch=deadline_epoch, exact=exact),
            "tokenizer_bytes_verified": True, "inference_executed": False}


def resolve_tokenizer(*, explicit_dir: Path | None = None):
    directory = (Path(explicit_dir) if explicit_dir is not None else
                 Path(os.environ[TOKENIZER_DIR_ENV]) if TOKENIZER_DIR_ENV in os.environ else RETAINED_TOKENIZER_DIR)
    try:
        # Only the six verified files are copied. Extra local receipts/caches
        # cannot become an executable tokenizer sidecar in the sealed bundle.
        return directory, verify_tokenizer_directory(directory, exact=False)
    except OSError as exc:
        raise ValueError("trlc_pi05_tokenizer_unavailable") from exc
    except ValueError as exc:
        raise ValueError("trlc_pi05_tokenizer_invalid_or_unavailable") from exc
