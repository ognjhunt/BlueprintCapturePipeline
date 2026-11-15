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
                 now: float | None = None, evaluation_authority: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Bind the platform pause or an exact directive to the current owner intent."""
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
    if evaluation_authority is None:
        _require(intake._identifier(intent_id), 'owner_binding_missing')
    from . import task_evaluation_control_stage_policy as stage_policy
    _require(profile.get('profile_digest') == canonical_digest(profile, digest_field='profile_digest'), 'profile_changed')
    moment = time.time() if now is None else now
    _require(intake._number(moment), 'clock_invalid')
    if evaluation_authority is None:
        owner = policy.owner_for_profile(profile, now=moment)
    else:
        from .task_evaluation_team_run_authority import evaluation_owner
        owner = evaluation_owner(source_profile=profile, authority=evaluation_authority,
            source_launch_id=source_launch_id,
            configured_scene_revision_digest=evaluation_authority.get('configured_scene_revision_digest'),
            evaluation_run_id=evaluation_authority.get('evaluation_run_id'), now=moment)
        intent_id = owner['intent_id']
    _require(owner is not None and owner['intent_id'] == intent_id, 'owner_mismatch')
    configured_root = os.getenv(ROOT_ENV)
    root = Path(configured_root or DEFAULT_ROOT)
    path = root / (intent_id + '.json')
    if root.exists() or root.is_symlink() or configured_root:
        _safe(root)
        _safe(path)
    if not path.exists():
        if not stage_policy.CONTROLS_PAUSED:
            return None
        request = owner['request']
        directive = {
            'schema_version': SCHEMA, 'intent_id': intent_id, 'intent_digest': owner['intent_digest'],
            'owner': request['owner'], 'authenticated_issuer': owner['authenticated_issuer'],
            'authorized_by': stage_policy.AUTHORIZED_BY,
            'authorization_reference': stage_policy.AUTHORIZATION_REFERENCE,
            'user_request': stage_policy.USER_REQUEST,
            'original_task_digest': cross_runtime_canonical_digest(request['task']),
            'policy_candidates': request['execution']['policy_candidates'],
            'expires_at_epoch': request['execution']['expires_at_epoch'],
            'omitted_controls': OMITTED, 'run_kind': 'internal_policy_canary',
            'claim_ceiling': 'diagnostic_policy_execution', 'maximum_policy_episodes': 20,
            'task_scoring_criteria_changed': False, 'qualified_comparison_permitted': False,
        }
        directive['directive_digest'] = canonical_digest(directive, digest_field='directive_digest')
        return directive
    _require(path.is_file() and not path.stat().st_mode & 0o027, 'directive_file_unsafe')
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


def confirmed_articulated_contract(*, task_spec: Mapping[str, Any], site_id: str,
                                   task_id: str) -> dict[str, Any]:
    """Translate the owner's frozen drawer criteria without changing scoring."""
    from .adp_articulated_task_success_contract import seal_task_success_contract
    owner = task_spec.get('configured_owner_authority') or {}
    criteria = task_spec.get('configured_success_criteria') or {}
    _require(task_spec.get('task_kind') == 'articulated_open_close'
             and criteria.get('owner_success_contract_required') is True
             and criteria.get('task_joint_drive_forbidden') is True
             and owner.get('confirmation_status') == 'confirmed'
             and bool(str(owner.get('accepted_by') or '').strip())
             and bool(str(owner.get('authority_reference') or '').strip()),
             'articulated_owner_contract_unconfirmed')
    contract = seal_task_success_contract(
        task_kind='articulated_open_close', task_spec=task_spec,
        site_id=site_id, task_id=task_id, author_source='task_owner',
        author_id=owner['accepted_by'], confirmation_status='confirmed',
        confirmed_by_team_id=owner['accepted_by'])
    opening = contract['criteria']['opening']['success_interval']
    _require(opening == task_spec['executable_opening_threshold']['success_interval'],
             'articulated_threshold_changed')
    return contract


# The CPU camera check has nothing to measure without a splat or a single task subject.
_CAMERA_VIEW_NOT_MEASURED = ('scene_camera_appearance_not_a_gaussian_splat', 'scene_camera_task_subject_unresolved')


def _scene_view_evaluator(packet_dir: Path):
    """The packet's world-camera view check, or None when the CPU has nothing to measure."""
    from .scene_camera_visibility import SceneCameraVisibilityError, packet_view_evaluator

    try:
        evaluate = packet_view_evaluator(packet_dir)
    except SceneCameraVisibilityError as exc:
        if str(exc) in _CAMERA_VIEW_NOT_MEASURED:
            return None
        raise ValueError('scene_control_omission_camera_external_view_unmeasurable:' + str(exc)) from exc

    def problems(row, plan=None):
        try:
            return evaluate(row, plan)
        except SceneCameraVisibilityError as exc:
            raise ValueError('scene_control_omission_camera_external_view_unmeasurable:' + str(exc)) from exc

    return problems


def _require_rendered_external_view(packet_dir: Path, resolved_cells, view=None) -> None:
    """Measure the external camera each cell renders through the captured scene, before any spend.

    The packet's camera check measured the plan's world ``external`` row; the policy runtime
    renders Arena's external camera at this binding's offset instead.
    """
    from .native_task_camera_start_configuration import require_external_camera_scene_view

    view = view or _scene_view_evaluator(packet_dir)
    if view is not None:
        require_external_camera_scene_view(resolved_cells, view)
