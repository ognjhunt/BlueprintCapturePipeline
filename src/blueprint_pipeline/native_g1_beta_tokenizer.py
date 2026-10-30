"""Resolve exact retained Pi tokenizer assets without credentials or network.

The fixed directory is an existing model-only cache, not authority for its old
scene. Every use re-verifies the reviewed six-file inventory. Explicit caller
and existing environment configuration take precedence; an invalid override
never silently falls back to another directory.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .native_g1_beta_assets import needs_pi_tokenizer
from .native_g1_beta_authority import PI_AUXILIARY
from .native_g1_pi_tokenizer_assets import verify_tokenizer_assets

TOKENIZER_DIR_ENV = "BLUEPRINT_G1_PI_TOKENIZER_DIR"
RETAINED_TOKENIZER_DIR = Path("/var/lib/blueprint/task-evaluation-inputs/g1-841757-development/pi-tokenizer-f1eaa13")
INVENTORY = Path(__file__).resolve().parents[2] / "configs/g1_paligemma_tokenizer_inventory.v1.json"


def resolve_selected_tokenizer(candidate_ids, *, explicit_dir: Path | None = None):
    """Return verified model bytes only when the exact selection contains Pi."""
    if not needs_pi_tokenizer(candidate_ids):
        return None, None
    if explicit_dir is not None:
        directory = Path(explicit_dir)
    elif TOKENIZER_DIR_ENV in os.environ:
        directory = Path(os.environ[TOKENIZER_DIR_ENV])
    else:
        directory = RETAINED_TOKENIZER_DIR
    try:
        if (not directory.is_absolute() or directory.is_symlink() or not directory.is_dir()
                or any(parent.is_symlink() for parent in directory.parents)):
            raise ValueError("g1_beta_selected_pi_tokenizer_unavailable")
        if INVENTORY.is_symlink() or "sha256:" + hashlib.sha256(INVENTORY.read_bytes()).hexdigest() != PI_AUXILIARY["inventory_file_sha256"]:
            raise ValueError("g1_beta_selected_pi_tokenizer_invalid")
        receipt = verify_tokenizer_assets(inventory_path=INVENTORY, asset_dir=directory)
    except OSError as exc:
        raise ValueError("g1_beta_selected_pi_tokenizer_unavailable") from exc
    except ValueError as exc:
        if str(exc).startswith("g1_beta_selected_pi_tokenizer_"):
            raise
        raise ValueError("g1_beta_selected_pi_tokenizer_invalid") from exc
    return directory, receipt
