"""ADP-009D/day-21: reuse scene bytes while isolating each requested evaluation.

These identities grant no execution authority and do not change the source scene.
An omitted evaluation id preserves existing controller receipts byte for byte.
"""
from __future__ import annotations

import hashlib
import re

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}")


def evaluation_scope(evaluation_run_id: str | None) -> dict[str, str]:
    if evaluation_run_id is None:
        return {}
    if not isinstance(evaluation_run_id, str) or not _IDENTIFIER.fullmatch(evaluation_run_id):
        raise ValueError("configured_controls_evaluation_run_id_invalid")
    return {"evaluation_run_id": evaluation_run_id}


def scoped_identity(source_id: str, evaluation_run_id: str | None) -> str:
    evaluation_scope(evaluation_run_id)
    if evaluation_run_id is None:
        return source_id
    token = hashlib.sha256(f"{source_id}\0{evaluation_run_id}".encode()).hexdigest()[:32]
    return f"team-eval-{token}"


def episode_namespace(configuration_run_id: str, commit: str, *,
                      evaluation_run_id: str | None = None, destination: bool = False) -> str:
    source = scoped_identity(configuration_run_id, evaluation_run_id)
    phase = "destination-qualification" if destination else "controls"
    return f"{source}-franka-{phase}-{commit[:12]}"


def progression_directory(commit: str, evaluation_run_id: str | None = None) -> str:
    base = f"franka-controls-{commit[:12]}"
    return scoped_identity(base, evaluation_run_id)
