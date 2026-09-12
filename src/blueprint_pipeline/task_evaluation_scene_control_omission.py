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
    configured_root = os.getenv(ROOT_ENV)
    root = Path(configured_root or DEFAULT_ROOT)
    # An absent optional default does not activate this feature (including on
    # developer systems where /etc itself is a platform symlink).
    if not configured_root and not root.exists() and not root.is_symlink():
        return None
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


def bind_camera_start(*, directive, plan, construction, contract, cells):
    """Reopen exact owner robot calibration, then check every current cell's framing."""
    import hashlib
    from copy import deepcopy
    from . import task_evaluation_controls_autoprovision as controls
    from .task_evaluation_scene_robot_assignment import resolve_controls_robot_binding
    from .native_task_camera_start_configuration import materialize_camera_start_from_construction, validate_camera_start_configuration
    from .native_task_arena_policy_canary_worker import _resolved_scene_plan
    from . import task_evaluation_scene_intake as intake
    config = controls._json(Path(os.getenv(controls.CONFIG_ENV, '/etc/blueprint/task-evaluation-controls-autoprovision.json')))
    directory = Path(config['scene_root']) / directive['intent_id']
    intent = intake._read(directory / 'intent.json', 'intent_digest')
    _require(intent['intent_digest'] == directive['intent_digest'] and intent['authenticated_issuer'] in config['trusted_clients'],
             'camera_owner_changed')
    catalog = controls._sealed(Path(config['robot_catalog_path']), 'catalog_digest')
    robot, _ = resolve_controls_robot_binding(directory=directory, intent=intent, catalog=catalog)
    controls._asset(robot['robot_asset_usd'])
    robot_sha = robot['robot_asset_usd']['digest']
    root = Path(os.getenv('BLUEPRINT_TASK_EVALUATION_POLICY_CAMERA_CALIBRATION_ROOT',
                          '/etc/blueprint/task-evaluation-policy-camera-calibrations'))
    path = root / (robot_sha.removeprefix('sha256:') + '.json')
    _safe(path)
    _require(path.is_file() and not path.stat().st_mode & 0o027, 'camera_calibration_file_unsafe')
    calibration = json.loads(path.read_bytes())
    _require(calibration.get('schema_version') == 'policy_canary_robot_camera_kinematic_calibration.v1'
        and calibration.get('calibration_digest') == canonical_digest(calibration, digest_field='calibration_digest')
        and calibration.get('source_robot_asset_sha256') == robot_sha, 'camera_calibration_invalid')
    values = []
    for key in ('camera_start_binding', 'native_reference_gate'):
        ref = calibration[key]
        ref_path = Path(ref['path'])
        _safe(ref_path)
        _require(ref_path.is_file() and not ref_path.stat().st_mode & 0o027, 'camera_calibration_reference_unsafe')
        raw = ref_path.read_bytes()
        _require(ref == {'path':str(ref_path), 'sha256':'sha256:'+hashlib.sha256(raw).hexdigest(), 'size_bytes':len(raw)},
                 'camera_calibration_reference_changed')
        values.append(json.loads(raw))
    current = deepcopy(plan)
    current['task_spec']['task_success_contract'] = deepcopy(contract)
    binding = materialize_camera_start_from_construction(plan=current, construction=construction,
        source_binding=values[0], native_reference_gate=values[1], robot_asset_sha256=robot_sha,
        runtime_digest=robot['runtime_digest'], calibration_digest=calibration['calibration_digest'])
    current['policy_canary_camera_start_configuration'] = binding
    _require(len(cells) == 10, 'camera_cell_count_invalid')
    for cell in cells:
        resolved = _resolved_scene_plan(current, cell, task_success_contract=contract)
        validate_camera_start_configuration(resolved, binding)
    return binding
