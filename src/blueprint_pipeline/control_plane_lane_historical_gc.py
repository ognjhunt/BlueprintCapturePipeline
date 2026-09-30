"""ADP-009D/day28: select at most one protected historical action per GC tick.

GC does not acquire target write mounts or count a submitted unit as deletion.
Current opt-in and owner decisions are reauthenticated by the fixed dispatcher,
and the separately sandboxed worker owns actual mutation and durable receipts.
"""
from __future__ import annotations

import os
import re
import time

from . import control_plane_lane_historical_authority as authority
from . import control_plane_lane_historical_generation as generation
from .control_plane_lane_historical_dispatch import dispatch_historical_action


def gc_historical_actions(*, installed_config_path, apply, now, monotonic=time.monotonic):
    result = dict(enabled=False, units_started=0, mutations=0, removed_bytes=0, outcomes=[])
    if apply is not True:
        return result
    operation = authority._Operation(now(), monotonic)
    with authority._session(installed_config_path, operation) as (files, config, store):
        if config.historical_generation_actions_enabled is not True:
            return result
        result['enabled'] = True
        names = sorted(os.listdir(store.parent))
        candidates = []
        for name in names:
            files.budget.tick()
            if not name.endswith('.json') or name.endswith('.manifest.json'):
                continue
            value, _ = store.read(name[:-5])
            if value.get('schema_version') not in ('control_plane_historical_decommission.v1',
                                                  'control_plane_historical_restore_decision.v1'):
                continue
            if value.get('action') == 'owner_review':
                result['outcomes'].append(dict(action_id=name[:-5], status='kept', reason='owner_review'))
                continue
            candidates.append(name[:-5])
    for action_id in candidates:
        try:
            submitted = dispatch_historical_action(installed_config_path=installed_config_path,
                action_id=action_id, now=operation.moment(), monotonic=monotonic)
        except (OSError, ValueError) as error:
            reason = getattr(error, 'code', None)
            if reason is None and isinstance(error, generation.HistoricalGenerationError):
                reason = str(error)
            if type(reason) is not str or not re.fullmatch(r'[a-z_]{1,160}', reason):
                reason = 'historical_generation_dispatch_unavailable'
            result['outcomes'].append(dict(action_id=action_id, status='kept',
                                          reason=reason))
            # An ambiguous admission might have started this exact operation.
            # It consumes the tick even though there is no completion proof.
            if reason in ('historical_generation_dispatch_start_unknown',
                          'historical_generation_dispatch_start_refused'):
                break
        else:
            result['outcomes'].append(submitted)
            result['units_started'] = 1
            break
    return result
