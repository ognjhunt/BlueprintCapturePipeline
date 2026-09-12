"""Private exact-owner opt-in to existing unqualified policy control omission."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

SCHEMA = 'task_evaluation_scene_control_omission_directive.v1'
ROOT_ENV = 'BLUEPRINT_TASK_EVALUATION_DIAGNOSTIC_CONTROL_OMISSION_ROOT'
DEFAULT_ROOT = '/etc/blueprint/task-evaluation-diagnostic-control-omissions'
OMITTED = ['zero_action_negative', 'deterministic_scripted_positive']


def _require(condition: bool, code: str):
    if not condition:
        raise ValueError('scene_control_omission_' + code)


def _safe(path: Path):
    _require(path.is_absolute() and '..' not in path.parts
             and not any(p.is_symlink() for p in (path, *path.parents)), 'path_unsafe')


def load_for_run(*, launch_state_root: str | Path, source_launch_id: str,
                 now: float | None = None) -> dict[str, Any] | None:
    """Missing opt-in preserves strict progression; malformed opt-in refuses."""
    from . import task_evaluation_scene_policy_binding as policy
    from . import task_evaluation_scene_intake as intake
    _require(intake._identifier(source_launch_id), 'source_id_invalid')
    profile_path = Path(launch_state_root) / source_launch_id / 'launch_profile.json'
    _safe(profile_path)
    if not profile_path.exists():
        return None
    profile = json.loads(profile_path.read_bytes())
    if profile.get('scene_intent_digest') is None:
        return None
    intent_id = (profile.get('scene_attempt_binding') or {}).get('intent_id')
    _require(intake._identifier(intent_id), 'owner_binding_missing')
    root = Path(os.getenv(ROOT_ENV, DEFAULT_ROOT))
    _safe(root)
    path = root / (intent_id + '.json')
    _safe(path)
    if not path.exists():
        return None
    _require(path.is_file() and not path.stat().st_mode & 0o027, 'directive_file_unsafe')
    _require(profile.get('profile_digest') == canonical_digest(profile, digest_field='profile_digest'), 'profile_changed')
    moment = time.time() if now is None else now
    _require(intake._number(moment), 'clock_invalid')
    owner = policy.owner_for_profile(profile, now=moment)
    _require(owner is not None and owner['intent_id'] == intent_id, 'owner_mismatch')
    directive = json.loads(path.read_bytes())
    request = owner['request']
    _require(isinstance(directive, dict) and directive.get('schema_version') == SCHEMA
        and directive.get('directive_digest') == canonical_digest(directive, digest_field='directive_digest')
        and directive.get('intent_id') == intent_id and directive.get('intent_digest') == owner['intent_digest']
        and directive.get('owner') == request['owner']
        and directive.get('authenticated_issuer') == owner['authenticated_issuer']
        and directive.get('authorized_by') == request['owner']['user_id']
        and isinstance(directive.get('authorization_reference'), str) and bool(directive['authorization_reference'].strip())
        and isinstance(directive.get('user_request'), str) and bool(directive['user_request'].strip())
        and directive.get('original_task_digest') == cross_runtime_canonical_digest(request['task'])
        and directive.get('policy_candidates') == request['execution']['policy_candidates']
        and intake._number(directive.get('expires_at_epoch'))
        and moment < directive['expires_at_epoch'] <= request['execution']['expires_at_epoch']
        and directive.get('omitted_controls') == OMITTED
        and directive.get('run_kind') == 'internal_policy_canary'
        and directive.get('claim_ceiling') == 'diagnostic_policy_execution'
        and type(directive.get('maximum_policy_episodes')) is int and directive['maximum_policy_episodes'] == 20
        and directive.get('task_scoring_criteria_changed') is False
        and directive.get('qualified_comparison_permitted') is False, 'directive_invalid')
    return directive


def derived_contract(*, packet_request_path: Path, directive: Mapping[str, Any]):
    """Use the existing typed omission producer; retain original request bytes."""
    from .native_marked_area_rehearsal import direct_policy_request
    _safe(packet_request_path)
    source = json.loads(packet_request_path.read_bytes())
    derived = direct_policy_request(source_request=source, authorized_by=directive['authorized_by'],
        authorization_reference=directive['authorization_reference'])
    return derived['task_spec']['task_success_contract'], derived['diagnostic_control_omission_authority']
