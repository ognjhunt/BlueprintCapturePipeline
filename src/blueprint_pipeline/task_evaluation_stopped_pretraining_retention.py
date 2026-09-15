"""Reclaim only an exact extracted bundle after a stopped launch is closed."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path

from .control_plane_workspace_lock import workspace_lock
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_artifixer_pretraining import LOGICAL_ROOT
from .task_evaluation_blocked_activation_retention import (
    _file_record, _read_sealed, _remove_stage_tree, _tree_and_archive_snapshot,
)
from .task_evaluation_launch_dispatcher import validate_launch_profile_structure, verify_profile_immutable_inputs
from .task_evaluation_launch_reconciler import _guard_provider_zero, _required_provider_scope
from .task_evaluation_release_retention import _write_exclusive

LAUNCH_ROOT = Path('/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-runs')
SCHEMA = 'stopped_pretraining_bundle_retention.v1'
ACK = 'reclaim-stopped-pretraining-bundle'


def _require(condition, reason):
    if not condition:
        raise ValueError('stopped_pretraining_retention_' + reason)


def _safe(path):
    path = Path(path)
    _require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'path_unsafe')
    return path


def _sealed(path, schema, field):
    return _read_sealed(_safe(path), schema_version=schema, digest_field=field,
                        blocker='stopped_pretraining_retention_' + path.name + '_invalid')


def _binding(launch_root):
    launch = _safe(launch_root)
    _require(launch.parent == LAUNCH_ROOT and launch.is_dir(), 'launch_root_invalid')
    profile, profile_record = _sealed(launch/'launch_profile.json', 'task_evaluation_launch_profile.v1', 'profile_digest')
    request, request_record = _sealed(launch/'launch_request.json', 'task_evaluation_launch_request.v1', 'request_digest')
    started, started_record = _sealed(launch/'launch_started.json', 'task_evaluation_launch_started.v1', 'started_digest')
    recovery, recovery_record = _sealed(launch/'orphan_recovery_receipt.json', 'task_evaluation_launch_orphan_recovery.v1', 'recovery_digest')
    _require(not validate_launch_profile_structure(profile) and not verify_profile_immutable_inputs(profile), 'profile_inputs_invalid')
    state = recovery.get('dispatcher_state') or {}
    _require(recovery.get('status') == 'provider_zero_confirmed' and recovery.get('provider_zero_confirmed') is True
             and recovery.get('recovery_basis') == 'stopped_dispatcher_and_fresh_provider_zero'
             and recovery.get('automatic_retry_performed') is False and recovery.get('allocator_invoked') is False
             and recovery.get('blockers') == [] and state.get('ActiveState') in {'inactive', 'failed'}
             and str(state.get('MainPID')) == '0' and state.get('ControlGroup') == '', 'launch_not_stopped')
    _require(all(row.get('launch_id') == launch.name for row in (request, started, recovery))
             and request.get('launch_profile_digest') == recovery.get('launch_profile_digest') == profile['profile_digest']
             and request['request_digest'] == started.get('request_digest') == recovery.get('request_digest')
             and started['started_digest'] == recovery.get('started_digest')
             and started.get('automatic_retry_authorized') is False, 'launch_binding_changed')
    pid = started.get('process_id')
    _require(type(pid) is int and pid > 0, 'writer_identity_missing')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('stopped_pretraining_retention_writer_still_exists')
    guard_path = _safe(recovery['guard_report_path'])
    _require(guard_path.parent == launch/'reconciliations', 'guard_path_invalid')
    guard_record = _file_record(guard_path, blocker='stopped_pretraining_retention_guard_invalid')
    _require(guard_record['sha256'] == recovery.get('guard_report_sha256'), 'guard_changed')
    providers, _ = _required_provider_scope(profile, expected_profile_digest=profile['profile_digest'])
    zero, blockers = _guard_provider_zero(guard=json.loads(guard_path.read_text()), required_providers=providers,
        max_age_seconds=int(profile.get('reconciliation', {}).get('max_guard_age_seconds', 300)),
        now=datetime.fromisoformat(recovery['observed_at']), not_before=datetime.fromisoformat(started['started_at']))
    _require(zero and not blockers and recovery.get('required_providers') == providers, 'recorded_provider_zero_invalid')
    inputs = {row['name']: row for row in profile['immutable_inputs']}
    bundle_path = _safe(inputs['source_bundle_manifest']['path'])
    authority_path = _safe(inputs['scene_configuration_attempt_authority']['path'])
    argv = profile['allocator']['argv']
    for flag, expected in (('--scene-configuration-bundle-receipt', bundle_path),
                           ('--scene-configuration-attempt-authority', authority_path)):
        _require(argv.count(flag) == 1 and argv.index(flag)+1 < len(argv)
                 and argv[argv.index(flag)+1] == str(expected), 'allocator_input_changed')
    bundle, bundle_record = _sealed(bundle_path, 'task_evaluation_scene_configuration_provider_bundle.v1', 'receipt_digest')
    authority, authority_record = _sealed(authority_path, 'task_evaluation_scene_configuration_paid_authority.v1', 'authority_digest')
    _require(authority.get('provider') == 'vast' and authority.get('maximum_paid_attempts') == 1
             and authority.get('maximum_provider_allocations') == 1 and authority.get('maximum_automatic_retries') == 0
             and authority.get('automatic_paid_retry_authorized') is False and authority.get('retry_cap') == 0
             and all(authority.get(k) == bundle.get(k) for k in ('bundle_sha256', 'run_id', 'source_commit')),
             'paid_scope_changed')
    archive = _safe(bundle['bundle_path'])
    archive_record = _file_record(archive, blocker='stopped_pretraining_retention_archive_invalid')
    _require(archive_record['sha256'] == bundle['bundle_sha256'], 'archive_changed')
    key = canonical_digest({'bundle': bundle['bundle_sha256'], 'authority': authority['authority_digest']})[7:]
    workspace = _safe(LOGICAL_ROOT/key)
    _require(workspace.parent == LOGICAL_ROOT and workspace.is_dir(), 'derived_root_invalid')
    _require(not archive.is_relative_to(workspace), 'archive_inside_removable_workspace')
    return workspace, archive, {'profile': profile_record, 'request': request_record, 'started': started_record,
        'recovery': recovery_record, 'guard': guard_record, 'bundle_receipt': bundle_record,
        'authority': authority_record, 'archive': archive_record}


def _plan(launch_root):
    workspace, archive, bindings = _binding(launch_root)
    tree = _tree_and_archive_snapshot(workspace/'bundle', archive)
    return {'schema_version': SCHEMA, 'status': 'dry_run', 'launch_root': str(launch_root),
            'workspace': str(workspace), 'bindings': bindings, 'removable_bundle': tree,
            'preserved_paths': [str(p) for p in sorted(workspace.iterdir()) if p.name != 'bundle'],
            'original_archive_preserved': True, 'new_paid_execution_authorized': False,
            'historical_spend_settled': False, 'proof_effect': 'none'}


def retain_stopped_bundle(*, launch_root=None, plan_out=None, plan_path=None, apply=False, acknowledgement=None):
    """Dry-run first; fresh binding and byte equality are required under the writer lock."""
    if apply:
        _require(acknowledgement == ACK and plan_path is not None and launch_root is None and plan_out is None, 'apply_arguments_invalid')
        saved, _ = _sealed(_safe(plan_path), SCHEMA, 'plan_digest')
        _require(saved.get('status') == 'dry_run', 'plan_not_dry_run')
        launch_root = Path(saved['launch_root'])
    else:
        _require(launch_root is not None and plan_out is not None and plan_path is None, 'dry_run_arguments_invalid')
        launch_root = Path(launch_root)
    workspace, _, _ = _binding(launch_root)
    with workspace_lock(workspace, reclaim=True) as locked:
        _require(locked, 'workspace_writer_active_or_legacy_lock_missing')
        current = _plan(launch_root)
        current['plan_digest'] = canonical_digest(current, digest_field='plan_digest')
        target = _safe(plan_path if apply else plan_out)
        _require(not target.is_relative_to(workspace), 'receipt_inside_workspace')
        if not apply:
            _write_exclusive(target, current)
            return current
        _require(current == saved, 'plan_changed')
        intent = target.with_name(target.name + '.apply-intent.json')
        applied = target.with_name(target.name + '.applied.json')
        _write_exclusive(intent, {'schema_version': SCHEMA, 'status': 'applying', 'plan_digest': saved['plan_digest']})
        _remove_stage_tree(workspace/'bundle', saved['removable_bundle'])
        result = {'schema_version': SCHEMA, 'status': 'applied', 'plan_digest': saved['plan_digest'],
                  'removed_bundle': saved['removable_bundle'], 'original_archive_preserved': saved['bindings']['archive'],
                  'preserved_paths': saved['preserved_paths'], 'new_paid_execution_authorized': False,
                  'historical_spend_settled': False, 'proof_effect': 'none'}
        result['retention_digest'] = canonical_digest(result, digest_field='retention_digest')
        _write_exclusive(applied, result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-root', type=Path)
    parser.add_argument('--plan-out', type=Path)
    parser.add_argument('--plan', type=Path, dest='plan_path')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--ack', dest='acknowledgement')
    args = parser.parse_args(argv)
    print(json.dumps(retain_stopped_bundle(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
