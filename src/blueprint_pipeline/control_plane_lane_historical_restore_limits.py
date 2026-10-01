"""Finite parsing allowances for a selected original restore partition.

This is no owner decision, birth proof or execution authority. Callers separately
authenticate the original manifest, action and exact observed journal partition.
Public generation admission keeps its existing bounds.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType

from . import control_plane_lane_historical_generation as generation

STAGE_PREFIX_BYTES = 52  # 51 ASCII bytes for stage name, then one slash.
# Each new kernel version has ten unsigned 64-bit values. Allow their complete
# decimal representation, rather than assuming original and restored widths match.
VERSION_BYTES = 10 * 21
MAX_OBSERVATION_BYTES = generation.MAX_MANIFEST_BYTES + generation.MAX_MEMBERS * (
    STAGE_PREFIX_BYTES + VERSION_BYTES) + 65536 + 4096


@dataclass(frozen=True, init=False)
class RestoreObservationBounds:
    """Exact original names/kinds plus the single action-specific stage root."""
    target_path: str
    stage: str
    kinds: Mapping[str, str]
    member_count: int
    payload_bytes: int
    encoded_bytes: int

    def __init__(self, original, action_id):
        from .control_plane_lane_historical_fence import _members
        generation._require(type(action_id) is str and re.fullmatch('[0-9a-f]{32}', action_id),
                            'restore_bounds_invalid')
        rows = _members(original)
        raw = json.dumps(original, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        generation._require(len(raw) <= generation.MAX_MANIFEST_BYTES
            and type(original.get('logical_payload_bytes')) is int
            and 0 <= original['logical_payload_bytes'] <= generation.MAX_PAYLOAD_BYTES,
            'restore_bounds_invalid')
        stage = '.historical-restore-' + action_id
        generation._require(not any(path.split('/')[0] == stage for path in rows), 'restore_stage_collision')
        kinds = {path: row['kind'] for path, row in rows.items()}
        kinds.update({stage + ('/' + path if path else ''): row['kind']
                      for path, row in rows.items()})
        for key, value in dict(target_path=original['target_path'], stage=stage, kinds=MappingProxyType(kinds),
                member_count=len(rows) + 1, payload_bytes=original['logical_payload_bytes'],
                encoded_bytes=len(raw) + len(rows) * (STAGE_PREFIX_BYTES + VERSION_BYTES) + 65536 + 4096).items():
            object.__setattr__(self, key, value)

    def accepts(self, relative, kind):
        return self.kinds.get(relative) == kind


def selected_restore_bounds(worker):
    """The selected original is already authenticated by the worker checkpoint."""
    cached = getattr(worker, 'restore_bounds', None)
    return cached if cached is not None else RestoreObservationBounds(worker.selected[2], worker.action_id)


def restore_members(original, observed, action_id, refusal):
    """Preserve each caller's typed refusal while bounding its observation."""
    from .control_plane_lane_historical_fence import HistoricalFenceError, _members
    try:
        return _members(observed, restore_bounds=RestoreObservationBounds(original, action_id))
    except HistoricalFenceError:
        generation._require(False, refusal)
