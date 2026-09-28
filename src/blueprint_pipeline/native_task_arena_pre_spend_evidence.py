"""Fail-closed evidence checks for native pre-spend allocator exits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _zero_cost(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def validate_pre_spend_original_evidence(
    *, original: Mapping[str, Any], provider_run: Path
) -> dict[str, Any] | None:
    """Accept legacy omissions or the allocator's explicit zero-cost shape."""

    cost = original.get("estimated_cost_usd")
    if (cost is not None and not _zero_cost(cost)) or original.get(
        "continuing_spend_from_this_run"
    ) not in (None, False):
        raise ValueError("native_task_arena_pre_spend_result_invalid")
    path = provider_run / "vast_teardown_manifest.json"
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("native_task_arena_pre_spend_teardown_invalid") from exc
    if (
        path.is_symlink()
        or not isinstance(value, dict)
        or value.get("schema_version") != "vast_teardown_manifest.v1"
        or value.get("status") != "not_required_provider_adapter_never_invoked"
        or value.get("vast_instance_ids") != []
        or value.get("teardown_actions_performed") != []
        or value.get("continuing_spend_from_this_run") is not False
    ):
        raise ValueError("native_task_arena_pre_spend_teardown_invalid")
    return {
        "path": str(path),
        "size_bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


__all__ = ["validate_pre_spend_original_evidence"]
