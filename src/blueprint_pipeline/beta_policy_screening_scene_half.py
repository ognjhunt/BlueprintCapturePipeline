"""Which compiled scene half a screening plan was built from, read without rebuilding its inputs.

A website job's team half can be compiled again after its screening plan froze: a newer stance
search finds another station, or the site owner approves moving chairs out of the Franka's way.
The source (intent, capture, task, scene revision) stays the same, so the campaign id does too,
and without this the source would stay planned against the old half forever (capture-447ae4f4:
every learned policy planned ``infeasible`` from a station the recompiled half no longer uses).

A half's identity is the digest of the fields planning copies unchanged from the half record into
its scene inputs (``beta_policy_screening_capabilities.build_planning_inputs``): the scene
revision, the configuration digest (the revision, the packet bytes and the station's stance), the
native scene plan's digest and the packet receipt's bytes. The packet path is left out: the same
bytes compiled into another directory are the same half. Reading it costs one small record and the
two receipts ``read_team_scene_inputs`` already checks on every pass, never a planning rebuild.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

HALF_FIELDS = ("scene_revision_digest", "configuration_digest", "plan_digest")


def half_identity(inputs: Mapping[str, Any]) -> str | None:
    """The identity of the half ``inputs`` came from, or None when any field it rests on is missing."""

    packet = inputs.get("base_native_packet")
    values = {field: inputs.get(field) for field in HALF_FIELDS}
    if (not isinstance(packet, Mapping) or not isinstance(packet.get("sha256"), str)
            or any(not isinstance(value, str) or not value for value in values.values())):
        return None
    return canonical_digest({**values, "base_native_packet_sha256": packet["sha256"]})


def planned_half(plan: Mapping[str, Any]) -> str | None:
    """The half a plan froze, from its sealed scene inputs; None when they no longer read."""

    from .beta_policy_screening_files import _bound_path, _read

    try:
        return half_identity(_read(_bound_path(plan["scene_inputs"])))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def current_half(profile: Mapping[str, Any] | None, *, scene_revision_digest: str) -> str | None:
    """The team half the capture's profile names now, at this scene revision; None when there is none.

    Only a compiled team half (``team_scene_inputs_path``) is read: an operator's frozen pair and a
    task-family half are never compiled again by the packet worker. A half of another revision belongs
    to that revision's own source.
    """

    from .team_scene_packet import read_team_scene_inputs

    path = str((profile or {}).get("team_scene_inputs_path") or "")
    if not path or (profile or {}).get("scene_revision_digest") != scene_revision_digest:
        return None
    try:
        inputs = read_team_scene_inputs(Path(path))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return half_identity(inputs) if inputs.get("scene_revision_digest") == scene_revision_digest else None


__all__ = ["HALF_FIELDS", "current_half", "half_identity", "planned_half"]
