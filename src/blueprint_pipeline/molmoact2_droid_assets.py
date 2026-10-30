"""Pinned public MolmoAct2 DROID inputs; no allocation, download or activation."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

MODEL_ID = "allenai/MolmoAct2-DROID"
MODEL_REVISION = "d8c1abd8a27d8e859455bbe514df2bcc617db0fb"
SOURCE_REVISION = "6070080a20321b4f498ab30f28e1d09ac465edb7"
NORM_TAG = "franka_droid"
NORM_SHA256 = "9cc29b5018604be4b785dd6519be3c86d0420675da7364755c58155afccb131d"
INTERFACE_ID = "molmoact2_droid_joint8.v1"


RELEASE_INVENTORY_PATH = "_catalog_data/molmoact2_droid_release.v1.json"
RELEASE_INVENTORY_SHA256 = "sha256:a2b14535403baa3366ae3b232d3496e2c19820260f154cf3b39542229ef3abcb"


def release_inventory() -> dict[str, Any]:
    path = Path(__file__).parent / RELEASE_INVENTORY_PATH
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("molmoact2_release_inventory_invalid")
    raw = path.read_bytes()
    if "sha256:" + hashlib.sha256(raw).hexdigest() != RELEASE_INVENTORY_SHA256:
        raise ValueError("molmoact2_release_inventory_changed")
    return json.loads(raw)


def _checkpoint_files(root: Path, allowed: dict) -> set[str]:
    present = set()
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"molmoact2_checkpoint_file_invalid:{name}")
        if path.is_dir():
            continue
        if name not in allowed or not path.is_file():
            raise ValueError(f"molmoact2_checkpoint_unlisted_file:{name}")
        present.add(name)
    return present


def verify_checkpoint_directory(directory: str | Path, *, deadline_epoch: float | None = None) -> dict[str, Any]:
    """Hash all local inference inputs before executing checkpoint Python code.

    Large weight hashes are publisher LFS declarations until this function has
    read their full local bytes. A model card alone is never an execution receipt.
    The staged directory must contain only declared publisher files, not mutable
    cache symlinks or extra tokenizer sidecars. Runtime caches live outside it.
    """
    def check_deadline():
        if deadline_epoch is not None and (not math.isfinite(deadline_epoch) or time.time() >= deadline_epoch):
            raise ValueError("molmoact2_checkpoint_verification_deadline_exceeded")
    check_deadline()
    root = Path(directory).resolve(strict=True)
    manifest = release_inventory()
    allowed = {row["path"]: row for row in manifest["files"]}
    required = {name: row for name, row in allowed.items() if row["required_for_inference"]}
    present = _checkpoint_files(root, allowed)
    # Transformers can consume unlisted tokenizer/config sidecars even when all
    # Python and weight files match. Verify the entire permitted file closure,
    # including optional publisher assets when the stager supplies them.
    for name in sorted(set(required) | present):
        row = allowed[name]
        path = root / name
        if (not path.is_file() or path.is_symlink() or root not in path.resolve().parents
                or path.stat().st_size != row["bytes"]):
            raise ValueError(f"molmoact2_checkpoint_file_invalid:{name}")
        expected = row["publisher_lfs_sha256"] or row["locally_verified_sha256"]
        digest = hashlib.sha256()
        if not expected:
            expected = row.get("git_blob_id")
            digest = hashlib.sha1(f"blob {row['bytes']}\0".encode())
        if not expected:
            raise ValueError(f"molmoact2_checkpoint_digest_missing:{name}")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                check_deadline()
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"molmoact2_checkpoint_digest_mismatch:{name}")
    expected_shards = {name for name in required if name.endswith(".safetensors")}
    if ({p.name for p in root.glob("*.safetensors")} != expected_shards
            or set(json.loads((root / "model.safetensors.index.json").read_text())["weight_map"].values())
            != expected_shards):
        raise ValueError("molmoact2_checkpoint_selection_ambiguous")
    if _checkpoint_files(root, allowed) != present:
        raise ValueError("molmoact2_checkpoint_closure_changed")
    check_deadline()
    return {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "source_revision": SOURCE_REVISION,
            "norm_stats_sha256": NORM_SHA256, "checkpoint_bytes_verified": True,
            "files_verified": len(present), "license": manifest["license"],
            "license_evidence_url": manifest["license_evidence_url"],
            "inference_performed": False}

