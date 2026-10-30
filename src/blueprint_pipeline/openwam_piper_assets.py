"""Exact public PiPER release identity, without downloading or admitting a model.

Small files were fetched and SHA256-checked on 2026-10-29. The weight digest is
the publisher's Hugging Face LFS digest; only its safetensors header was read.
``verify_checkpoint_directory`` hashes full local weight bytes before loading.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import stat
import subprocess
import time
from typing import Any

from .openwam_piper_policy import MODEL_ID, MODEL_REVISION, SOURCE_REVISION, WEIGHT_SHA256

EXPECTED_FILES = {
    "README.md": (1397, "d044355de896208f4e804c8b529cc5ce1efbe4c767857e2f26aac1926915e448"),
    "config.yaml": (3394, "8c791a849543b325b121b0ac6e10c8b238f55d6954cddf3bce2962286a7d7f4a"),
    "normalization_stats.npy": (1884, "5412fe4539cedacef4c96d3518d6c738b54e704d6e12b910af4135ab9068e35e"),
    "tokenizer/google/umt5-xxl/special_tokens_map.json": (6623, "7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d"),
    "tokenizer/google/umt5-xxl/spiece.model": (4548313, "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"),
    "tokenizer/google/umt5-xxl/tokenizer.json": (16837417, "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"),
    "tokenizer/google/umt5-xxl/tokenizer_config.json": (61728, "ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05"),
    "checkpoint_step_30000.safetensors": (24813767464, WEIGHT_SHA256),
}


def _verify_checkpoint_file_closure(root: Path, expected_files: dict | None = None) -> None:
    """Bound all loader-visible files, including optional tokenizer sidecars.

    Only parent directories of the fixed inventory are traversed. Symlinks,
    special files and extra directories cannot hide unverified model inputs.
    Runtime/download receipts belong outside this checkpoint directory.
    """
    expected = set(EXPECTED_FILES if expected_files is None else expected_files)
    directories = {str(parent) for name in expected for parent in Path(name).parents
                   if parent != Path(".")}
    if not root.is_dir():
        raise ValueError("openwam_checkpoint_inventory_changed")
    pending, observed = [root], set()
    while pending:
        for path in pending.pop().iterdir():
            name = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode) and name in directories:
                pending.append(path)
            elif stat.S_ISREG(mode) and name in expected:
                observed.add(name)
            else:
                if path.parent == root and path.match("checkpoint_step_*.safetensors") and name not in expected:
                    raise ValueError("openwam_checkpoint_selection_ambiguous")
                raise ValueError("openwam_checkpoint_inventory_changed")
    if observed != expected:
        raise ValueError("openwam_checkpoint_inventory_changed")


def verify_checkpoint_directory(directory: str | Path, *, deadline_epoch: float | None = None) -> dict[str, Any]:
    """Verify exactly the deployed inputs before upstream's NumPy/checkpoint load."""
    _verify_checkpoint_inventory(directory, EXPECTED_FILES, deadline_epoch=deadline_epoch)
    return {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256,
            "checkpoint_bytes_verified": True, "files_verified": len(EXPECTED_FILES),
            "license_basis": "pinned_model_card_apache_2.0", "inference_performed": False}


def _verify_checkpoint_inventory(directory: str | Path, expected_files: dict, *, deadline_epoch: float | None = None) -> None:
    """Shared complete byte closure; callers supply only their fixed release inventory."""
    root = Path(directory).resolve(strict=True)
    if deadline_epoch is not None and (not math.isfinite(deadline_epoch) or deadline_epoch <= time.time()):
        raise ValueError("openwam_checkpoint_verification_deadline_exceeded")
    _verify_checkpoint_file_closure(root, expected_files)
    for name, (size, expected) in expected_files.items():
        path = root / name
        if (not path.is_file() or path.is_symlink() or root not in path.resolve().parents
                or path.stat().st_size != size):
            raise ValueError(f"openwam_checkpoint_file_invalid:{name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                if deadline_epoch is not None and time.time() >= deadline_epoch:
                    raise ValueError("openwam_checkpoint_verification_deadline_exceeded")
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"openwam_checkpoint_digest_mismatch:{name}")
    _verify_checkpoint_file_closure(root, expected_files)


def verify_source_checkout(directory: str | Path) -> Path:
    """Require the pinned clean upstream checkout; this does not install or fetch."""
    root = Path(directory).resolve(strict=True)
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(root), *args], check=False, capture_output=True, text=True)
        if result.returncode:
            raise ValueError("openwam_source_checkout_invalid")
        return result.stdout.strip()
    if git("rev-parse", "HEAD") != SOURCE_REVISION or Path(git("rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("openwam_source_revision_mismatch")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("openwam_source_checkout_modified")
    if not (root / "openwam/deploy/server.py").is_file():
        raise ValueError("openwam_source_checkout_invalid")
    return root
