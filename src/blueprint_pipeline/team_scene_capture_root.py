"""Find a team scene request's website capture and check it is still this scene's task.

Split out of ``team_scene_packet_worker`` so a CPU-only reader can resolve a capture
without importing the worker, which reaches the paid launch path. The intake's review
video route is such a reader, and ``tests/test_live_pipeline_import_isolation.py`` keeps
the GPU and provider modules out of the intake's import graph. The worker re-exports
every name here, so its callers are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

# The root the agent-run executor admits captures under (agent_run_executor._row_capture_root).
CAPTURE_PARTITION_ENV = "BLUEPRINT_AGENT_RUN_CAPTURE_PARTITION_ROOT"
# A website scene's task is "website-" plus its task context digest (website_task_preparation).
WEBSITE_TASK_PREFIX = "website-"


def capture_key(capture_root: Path) -> str:
    """The name the team registry files this capture's derived row under (and its half's directory)."""

    return hashlib.sha256(str(capture_root).encode()).hexdigest()[:32]


class TeamScenePacketRequestRefused(ValueError):
    """This request cannot become a team scene half; the message names why."""


def _read_json(path: Path) -> dict[str, Any] | None:
    if path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def resolve_capture_root(request: Mapping[str, Any], *, partition: str | Path | None = None) -> Path:
    """The request's website capture inside the executor's partition, bound to the scene's task."""

    from .website_atlas_preview_store import capture_withdrawn

    raw = str(partition or os.environ.get(CAPTURE_PARTITION_ENV, "")).strip()
    if not raw:
        raise TeamScenePacketRequestRefused("team_scene_capture_partition_not_configured")
    root = Path(raw)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise TeamScenePacketRequestRefused("team_scene_capture_partition_invalid")
    capture_id = str(request["capture_id"])
    matches = sorted(path for path in (root / "scenes").glob(f"*/captures/{capture_id}")
                     if path.is_dir() and not path.is_symlink())
    if len(matches) != 1:
        raise TeamScenePacketRequestRefused(
            "team_scene_capture_root_" + ("missing" if not matches else "ambiguous"))
    capture = matches[0].resolve()
    scene_id = capture.parent.parent.name
    descriptor = _read_json(capture / "capture_descriptor.json") or {}
    if (not capture.is_relative_to(root.resolve()) or descriptor.get("capture_id") != capture_id
            or descriptor.get("scene_id") != scene_id):
        raise TeamScenePacketRequestRefused("team_scene_capture_root_identity_mismatch")
    if capture_withdrawn(capture):
        raise TeamScenePacketRequestRefused("team_scene_capture_withdrawn")
    context = _read_json(capture / "pipeline" / "website_task_context.json") or {}
    digest = str(context.get("context_digest") or "")
    if (context.get("capture_id") != capture_id or context.get("scene_id") != scene_id
            or digest != canonical_digest(context, digest_field="context_digest")
            or request["task_id"] != WEBSITE_TASK_PREFIX + digest[7:27]):
        raise TeamScenePacketRequestRefused("team_scene_website_task_context_mismatch")
    return capture
