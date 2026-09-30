"""Existing file evidence records and release schema without factory execution imports."""
from __future__ import annotations

from pathlib import Path

from .task_evaluation_scene_configuration_submission_inputs import sha

RELEASE_SCHEMA = "task_evaluation_public_scene_release_binding.v1"


def record(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}
