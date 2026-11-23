"""Readers of the time-only extensions of a scene intent's window that already exist.

No person approves a window extension any more: the operator door's
``extend-scene-execution-window`` kind and the code that applied one are gone,
so a job whose work window ends, ends its paid work with what it has. Extension
records written before stay valid for every reader, so past jobs read as they did.
"""
from __future__ import annotations

from .task_evaluation_scene_execution_window_evidence import (
    ACK,
    DIRECTORY,
    SCHEMA,
    _bounds,
    effective_execution_expiry,
)

# Preserve the original reader import surface.
__all__ = [
    'ACK',
    'DIRECTORY',
    'SCHEMA',
    '_bounds',
    'effective_execution_expiry',
]
