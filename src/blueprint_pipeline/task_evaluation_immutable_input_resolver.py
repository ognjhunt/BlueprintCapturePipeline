"""Fail-closed resolver for dispatcher-staged immutable launch inputs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


STAGING_RECEIPT_ENV = "BLUEPRINT_TASK_EVALUATION_IMMUTABLE_INPUT_STAGING_RECEIPT"
STAGING_SCHEMA_VERSION = "task_evaluation_immutable_input_staging.v1"


class ImmutableInputResolutionError(ValueError):
    """Raised when a dispatch child cannot prove a staged input identity."""


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _lexical_absolute(path: str | Path) -> str:
    return os.path.abspath(str(Path(path).expanduser()))


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise ImmutableInputResolutionError(
            "immutable_input_staging_receipt_invalid"
        ) from exc
    if (
        path.is_symlink()
        or not isinstance(value, Mapping)
        or value.get("schema_version") != STAGING_SCHEMA_VERSION
        or value.get("status") != "staged"
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
        or value.get("raw_secret_values_recorded") is not False
    ):
        raise ImmutableInputResolutionError(
            "immutable_input_staging_receipt_invalid"
        )
    return dict(value)


def resolve_immutable_input(
    original_path: str | Path,
    *,
    expected_digest: str,
    expected_size_bytes: int,
) -> Path:
    """Resolve one exact original path to its byte-verified staged snapshot.

    Outside a dispatcher child there is no resolver environment and the
    existing direct-file behavior is retained.  Once the environment is set,
    every call is fail-closed: there is no fallback to the original path.
    """

    receipt_value = os.getenv(STAGING_RECEIPT_ENV, "").strip()
    if not receipt_value:
        return Path(original_path).expanduser().resolve()
    receipt_path = Path(_lexical_absolute(receipt_value))
    receipt = _load_receipt(receipt_path)
    original = _lexical_absolute(original_path)
    matches = [
        dict(row)
        for row in receipt.get("inputs") or []
        if isinstance(row, Mapping) and row.get("source_path") == original
    ]
    if len(matches) != 1:
        raise ImmutableInputResolutionError(
            "immutable_input_staging_mapping_missing"
        )
    row = matches[0]
    if (
        not isinstance(expected_size_bytes, int)
        or isinstance(expected_size_bytes, bool)
        or expected_size_bytes < 0
        or row.get("expected_digest") != expected_digest
        or row.get("staged_digest") != expected_digest
        or row.get("staged_size_bytes") != expected_size_bytes
    ):
        raise ImmutableInputResolutionError(
            "immutable_input_staging_mapping_identity_mismatch"
        )
    staged = Path(str(row.get("staged_path") or "")).expanduser()
    stage_root = receipt_path.parent / "immutable_inputs"
    try:
        staged_resolved = staged.resolve(strict=True)
        stage_root_resolved = stage_root.resolve(strict=True)
    except OSError as exc:
        raise ImmutableInputResolutionError(
            "immutable_input_staging_target_missing"
        ) from exc
    if (
        staged.is_symlink()
        or not staged_resolved.is_relative_to(stage_root_resolved)
        or not staged_resolved.is_file()
    ):
        raise ImmutableInputResolutionError(
            "immutable_input_staging_target_invalid"
        )
    payload = staged_resolved.read_bytes()
    if len(payload) != expected_size_bytes or _sha256(payload) != expected_digest:
        raise ImmutableInputResolutionError(
            "immutable_input_staging_target_identity_mismatch"
        )
    return staged_resolved


__all__ = [
    "ImmutableInputResolutionError",
    "STAGING_RECEIPT_ENV",
    "STAGING_SCHEMA_VERSION",
    "resolve_immutable_input",
]
