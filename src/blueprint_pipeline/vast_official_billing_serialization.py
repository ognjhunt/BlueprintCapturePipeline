"""Stable official billing metadata and byte identities without provider clients."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


BILLING_SOURCE_SCHEMA_VERSION = "blueprint.provider_billing_source_receipt.v1"
VAST_CHARGES_URL = "https://console.vast.ai/api/v0/charges/"
MAX_RESPONSE_BYTES = 32 * 1024 * 1024


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _record(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }
