"""Authority binding for generated closed-loop episode review artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [_string(item) for item in value if _string(item)]


def bind_generated_episode_to_authoritative_loop_status(
    episode_artifacts: dict[str, Any], *, authoritative_status: str
) -> dict[str, Any]:
    """Prevent review-media completion from overriding the loop manifest."""

    media_assembly_status = _string(episode_artifacts.get("status")) or "blocked"
    authoritative_completed = authoritative_status == "completed"
    effective_status = (
        media_assembly_status
        if authoritative_completed
        else "blocked_by_authoritative_closed_loop_manifest"
    )
    authoritative_blocker = f"authoritative_closed_loop_manifest_status:{authoritative_status}"
    blockers = sorted(
        {
            *_string_list(episode_artifacts.get("blockers")),
            *([] if authoritative_completed else [authoritative_blocker]),
        }
    )
    episode_artifacts.update(
        {
            "status": effective_status,
            "media_assembly_status": media_assembly_status,
            "authoritative_closed_loop_manifest_status": authoritative_status,
            "decision_grade_eligible": bool(
                authoritative_completed and media_assembly_status == "completed"
            ),
            "blockers": blockers,
        }
    )
    for path_key in ("manifest_path", "results_path"):
        path = Path(_string(episode_artifacts.get(path_key))).expanduser()
        if not path.is_file():
            continue
        payload = _mapping(json.loads(path.read_text(encoding="utf-8")))
        payload.update(
            {
                "status": effective_status,
                "media_assembly_status": media_assembly_status,
                "authoritative_closed_loop_manifest_status": authoritative_status,
                "decision_grade_eligible": episode_artifacts["decision_grade_eligible"],
                "blockers": blockers,
            }
        )
        claim_boundary = _mapping(payload.get("claim_boundary"))
        claim_boundary["generated_episode_status_cannot_override_authoritative_manifest"] = True
        payload["claim_boundary"] = claim_boundary
        write_json(path, payload)
    return episode_artifacts
