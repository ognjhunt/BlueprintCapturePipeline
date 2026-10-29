"""Exact public PiPER release identity, without downloading or admitting a model.

Small files were fetched and SHA256-checked on 2026-10-29. The weight digest is
the publisher's Hugging Face LFS digest; only its safetensors header was read.
``verify_checkpoint_directory`` hashes full local weight bytes before loading.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
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


def verify_checkpoint_directory(directory: str | Path) -> dict[str, Any]:
    """Verify exactly the deployed inputs before upstream's NumPy/checkpoint load."""
    root = Path(directory).resolve(strict=True)
    for name, (size, expected) in EXPECTED_FILES.items():
        path = root / name
        if (not path.is_file() or path.is_symlink() or root not in path.resolve().parents
                or path.stat().st_size != size):
            raise ValueError(f"openwam_checkpoint_file_invalid:{name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"openwam_checkpoint_digest_mismatch:{name}")
    # Other numbered checkpoints must not be picked by an accidental 'latest'.
    if set(p.name for p in root.glob("checkpoint_step_*.safetensors")) != {"checkpoint_step_30000.safetensors"}:
        raise ValueError("openwam_checkpoint_selection_ambiguous")
    return {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256,
            "checkpoint_bytes_verified": True, "files_verified": len(EXPECTED_FILES),
            "license_basis": "pinned_model_card_apache_2.0", "inference_performed": False}


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
