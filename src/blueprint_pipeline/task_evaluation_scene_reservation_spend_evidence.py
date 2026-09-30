"""Existing reservation exposure reader without spend-publication or lock imports."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from .task_evaluation_scene_intent_contracts import _read as read_scene


def _record(path: Path) -> dict[str, Any]:
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("scene_spend_source_unsafe")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": "sha256:" + digest, "size_bytes": path.stat().st_size}


def scene_reservation_spend_record(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize a reservation for accounting, never into a provider launch grant."""
    record = _record(path)
    attempt = read_scene(path, "attempt_digest")
    intent_path = path.parent.parent / "intent.json"
    intent_record = _record(intent_path)
    intent = read_scene(intent_path, "intent_digest")
    cap = attempt.get("maximum_spend_usd")
    if (attempt.get("schema_version") != "task_evaluation_scene_attempt.v1"
            or attempt.get("intent_digest") != intent.get("intent_digest")
            or attempt.get("intent_id") != intent.get("intent_id")
            or attempt.get("provider") not in intent["request"]["execution"]["allowed_providers"]
            or isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0):
        raise ValueError("scene_spend_reservation_invalid")
    return {**attempt, "authorization_digest": attempt["attempt_digest"], "hard_attempt_spend_cap_usd": cap}, {
        **record, "authorization_digest": attempt["attempt_digest"], "hard_attempt_spend_cap_usd": cap,
        "accounting_kind": "persistent_scene_reservation", "owner_intent": intent_record,
    }
