"""A website job's scene run that was moved to cold storage, as one typed state.

Beta journey step: L5 team policy evaluation. The storage GC moves a sealed
launch run to the artifact store and leaves a digest-bound pointer beside where
the directory stood (``control_plane_evidence_offload``). Stage 8 then reads the
run directly. Three readers each failed in their own way:

- the screening re-plan: ``configured_controls_worker_launch_receipt_invalid``,
  held after six hours under the catch-all stall reason;
- the team-half recompile: ``team_scene_source_launch_unavailable``;
- the robot-free probe: ``probe_root_unavailable``, which the screening pass read
  as "still in progress" for ever.

They now all name the same code, ``scene_run_in_cold_storage``. The job is told
it is held at once, under the accepted hold reason ``team_scene_packet_blocked``
(the website says it is a problem on our side) with this code as its detail.

What releases it today: an operator restores the run in place from its pointer
(``control_plane_evidence_offload.restore_offloaded_evidence(pointer_path=...,
destination=<the run's own path>)``). The next screening pass reads it again and
publishes the hold as released. Nothing here restores, writes or spends.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

COLD_STORAGE = "scene_run_in_cold_storage"
POINTER_SUFFIX = ".offloaded.v1.json"
POINTER_SCHEMA = "control_plane_evidence_offload_pointer.v1"
#: What releases the hold today, as the status says it.
RELEASED_BY = "operator_restore:control_plane_evidence_offload.restore_offloaded_evidence"


def offload_pointer(run_root: str | Path) -> dict[str, Any] | None:
    """The sealed offload pointer standing where ``run_root`` was, or None.

    None while the directory is on disk, and for anything that is not a valid,
    self-digest-intact pointer of exactly this directory: a missing run without
    a pointer stays the reader's own "unavailable" refusal.
    """

    run = Path(run_root)
    if not run.is_absolute() or run.is_dir():
        return None
    pointer = run.parent / (run.name + POINTER_SUFFIX)
    try:
        if pointer.is_symlink() or not pointer.is_file():
            return None
        value = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(value, dict) or value.get("schema_version") != POINTER_SCHEMA
            or value.get("status") != "offloaded" or value.get("directory") != run.name
            or value.get("pointer_digest") != canonical_digest(value, digest_field="pointer_digest")):
        return None
    return value


def in_cold_storage(run_root: str | Path) -> bool:
    """Whether the job's scene run is in cold storage rather than on this host."""

    return offload_pointer(run_root) is not None


def require_on_disk(run_root: str | Path) -> None:
    """Raise ``ValueError(COLD_STORAGE)`` when the run was moved to cold storage; otherwise nothing."""

    if in_cold_storage(run_root):
        raise ValueError(COLD_STORAGE)


def cold_storage_hold() -> tuple[str, str]:
    """The hold this state publishes: the accepted reason, with the code as its detail."""

    from .beta_policy_screening_holds import STALL_HOLD_REASON

    return STALL_HOLD_REASON, COLD_STORAGE


__all__ = ["COLD_STORAGE", "RELEASED_BY", "cold_storage_hold", "in_cold_storage", "offload_pointer",
           "require_on_disk"]
