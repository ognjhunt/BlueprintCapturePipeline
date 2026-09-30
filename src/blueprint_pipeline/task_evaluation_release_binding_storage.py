"""Existing exclusive evidence-binding storage without release-retirement imports."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EVIDENCE_BINDING_SCHEMA_VERSION = "task_evaluation_release_retention_binding.v1"


DEFAULT_EVIDENCE_BINDING_ROOT = Path(
    "/var/lib/blueprint/pipeline-control-plane/"
    "task-evaluation-release-retention-bindings"
)


class ReleaseRetentionError(ValueError):
    """The retirement boundary could not be proven safe."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ReleaseRetentionError(
            f"release_retention_receipt_conflict:{path.name}"
        ) from exc
