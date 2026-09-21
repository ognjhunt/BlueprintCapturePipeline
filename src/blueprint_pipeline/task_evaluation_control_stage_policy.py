"""Platform owner has paused control trials; simulation checks remain required."""
from __future__ import annotations

CONTROLS_PAUSED = True
AUTHORIZATION_REFERENCE = "docs/website-controls-pause-2026-09-21.md"
AUTHORIZED_BY = "blueprint-platform-owner"
USER_REQUEST = "Skip/pause the controls stage for all future runs."


def require_controls_enabled() -> None:
    if CONTROLS_PAUSED:
        raise ValueError("task_evaluation_controls_paused_by_owner")
