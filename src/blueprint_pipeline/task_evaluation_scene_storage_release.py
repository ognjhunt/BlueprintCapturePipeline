"""Release a terminal automatic scene launch's reproducible activation cache."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re

from .control_plane_storage_pins import (
    DEFAULT_PINS_ROOT, SCHEMA_VERSION as PIN_SCHEMA, pin_path, pins_root_from_environment, release_storage_pin,
)
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest


def _read(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlink')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('not_object')
    return value


def release_terminal_scene_activation_pin(*, run_root, receipt, pins_root=None, queue_root=None):
    """Best effort, metadata only: never remove inputs or change a launch result.

    Bind the pin to the exact staged source and the sealed terminal receipt.
    A warm/uncertain resource or another queued user of the profile retains it.
    """
    retained = {'status': 'retained', 'evidence_removed': False}
    try:
        run_root = Path(run_root)
        launch_id = str(receipt.get('launch_id') or '')
        if (receipt.get('schema_version') != 'task_evaluation_launch_receipt.v1'
                or not launch_id.endswith('-scene-configuration-activation-auto-launch')
                or run_root.name != launch_id or receipt.get('run_id') != launch_id
                or receipt.get('execute_requested') is not True
                or receipt.get('status') not in {'completed', 'blocked'}
                or receipt.get('receipt_digest') != cross_runtime_canonical_digest(receipt, digest_field='receipt_digest')):
            return {**retained, 'reason': 'not_bound_terminal_scene_launch'}
        pins_root = Path(pins_root or pins_root_from_environment() or DEFAULT_PINS_ROOT)
        if (not run_root.is_absolute() or not pins_root.is_absolute()
                or any(p.is_symlink() for root in (run_root, pins_root) for p in (root, *root.parents))):
            return {**retained, 'reason': 'unsafe_root'}
        owner = launch_id.removesuffix('-launch')
        pin_file = pin_path(pins_root, 'activation', owner)
        if not pin_file.is_file():
            return {**retained, 'reason': 'no_pin'}
        pin = _read(pin_file)
        if pin.get('schema_version') != PIN_SCHEMA or pin.get('owner_id') != owner or pin.get('kind') != 'activation':
            return {**retained, 'reason': 'pin_identity_mismatch'}
        if pin.get('released_at_epoch') is not None:
            return {'status': 'already_released', 'evidence_removed': False}
        staging = _read(run_root / 'immutable_input_staging_receipt.json')
        digest = canonical_digest(staging, digest_field='receipt_digest')
        if (staging.get('schema_version') != 'task_evaluation_immutable_input_staging.v1'
                or staging.get('status') != 'staged' or staging.get('receipt_digest') != digest
                or not isinstance(staging.get('profile_id'), str) or not staging['profile_id']
                or receipt.get('immutable_input_staging', {}).get('receipt_digest') != digest
                or staging.get('profile_digest') != receipt.get('launch_profile_digest')):
            return {**retained, 'reason': 'staging_identity_mismatch'}
        paths = [Path(p) for p in pin.get('paths', [])]
        primary = [p for p in paths if p.name == owner and p.parent.name == 'launch-activations']
        if (len(primary) != 1 or len(set(paths)) != len(paths)
                or any(not p.is_absolute() or '..' in p.parts or p.parent != primary[0].parent for p in paths)):
            return {**retained, 'reason': 'pin_path_invalid'}
        if (any(not p.is_dir() or any(part.is_symlink() for part in (p, *p.parents)) for p in paths)):
            return {**retained, 'reason': 'pin_target_unsafe'}
        activation_path = primary[0]
        matched = [row for row in staging.get('inputs', [])
                   if Path(row.get('source_path', '')).is_absolute()
                   and '..' not in Path(row['source_path']).parts
                   and Path(row['source_path']).is_relative_to(activation_path)]
        if not matched:
            return {**retained, 'reason': 'staged_input_not_bound_to_pin'}
        verified = {}
        def identity(path):
            st = path.stat()
            return (st.st_dev, st.st_ino, st.st_size, st.st_mode, st.st_uid, st.st_gid,
                    st.st_mtime_ns, st.st_ctime_ns)
        for row in matched:
            staged = Path(row.get('staged_path', ''))
            digest = row.get('staged_digest')
            if (not isinstance(digest, str) or re.fullmatch(r'sha256:[0-9a-f]{64}', digest) is None
                    or row.get('expected_digest') != digest or '..' in staged.parts
                    or not staged.is_relative_to(run_root / 'immutable_inputs')
                    or any(p.is_symlink() for p in (staged, *staged.parents)) or not staged.is_file()
                    or staged.stat().st_size != row.get('staged_size_bytes')):
                return {**retained, 'reason': 'staged_copy_unavailable'}
            before = identity(staged)
            if before not in verified:
                with staged.open('rb') as stream:
                    verified[before] = 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()
            if verified[before] != digest or identity(staged) != before:
                return {**retained, 'reason': 'staged_copy_digest_mismatch'}
        queue_root = Path(queue_root or run_root.parent.parent / 'task-evaluation-launches')
        if (not queue_root.is_absolute() or not queue_root.is_dir()
                or any(p.is_symlink() for p in (queue_root, *queue_root.parents))):
            return {**retained, 'reason': 'queue_unavailable'}
        for state in ('pending', 'processing'):
            for path in (queue_root / state).glob('*.json'):
                other = _read(path)
                if other.get('launch_id') == launch_id:
                    continue
                if (owner in json.dumps(other)
                        or staging.get('profile_id') and staging['profile_id'] in {
                            other.get('profile_id'), other.get('launch_profile_id')}):
                    return {**retained, 'reason': 'other_launch_uses_activation'}
        absent = (receipt.get('provider_mutation_attempted') is False
                  and receipt.get('provider_mutation_evidence', {}).get('status') == 'absent_before_paid_admission')
        provider = run_root / 'allocator/scene-configuration-job/vast_provider_run'
        if absent and provider.exists() and any(provider.iterdir()):
            return {**retained, 'reason': 'provider_evidence_conflicts_with_absence'}
        never_allocated_closed = False
        teardown_path = provider / 'vast_teardown_manifest.json'
        if not absent and teardown_path.exists():
            teardown = _read(teardown_path)
            if teardown.get('status') == 'not_required_provider_adapter_never_invoked':
                from .task_evaluation_launch_reconciler import (
                    _terminal_teardown_evidence, _validated_post_teardown_provider_zero_receipt,
                )
                zero_path = run_root / 'post_teardown_provider_zero_receipt.json'
                _read(zero_path)  # Reject symlinks throughout the retained path.
                zero = _validated_post_teardown_provider_zero_receipt(path=zero_path, receipt=receipt)
                proof, blockers = _terminal_teardown_evidence(receipt=receipt)
                if (teardown.get('schema_version') != 'vast_teardown_manifest.v1'
                        or blockers or not proof or proof.get('provider_resource_allocated') is not False
                        or proof.get('path') != str(teardown_path)
                        or zero.get('teardown_manifest') != proof
                        or zero.get('required_providers') != ['vast']
                        or (provider / 'vast_budget_ledger.json').exists()
                        or (provider / 'vast_budget_ledger.json').is_symlink()):
                    return {**retained, 'reason': 'never_allocated_closure_unconfirmed'}
                never_allocated_closed = True
        if not absent and not never_allocated_closed:
            teardown = _read(provider / 'vast_teardown_manifest.json')
            budget = _read(provider / 'vast_budget_ledger.json')
            ids = set(budget.get('vast_instance_ids') or [])
            closed = {r.get('instance_id') for r in teardown.get('teardown_actions_performed', [])
                      if r.get('action') == 'destroy_instance' and r.get('status') == 'completed'
                      and r.get('http_status_code') in {200, 204, 404}}
            if (teardown.get('schema_version') != 'vast_teardown_manifest.v1'
                    or budget.get('schema_version') != 'vast_budget_ledger.v1'
                    or not ids or any(type(i) is not int or i <= 0 for i in ids)
                    or ids != set(teardown.get('vast_instance_ids') or []) or not ids <= closed
                    or teardown.get('status') != 'completed'
                    or teardown.get('runner_gpu_teardown_completed') is not True
                    or teardown.get('continuing_spend_from_this_run') is not False
                    or budget.get('continuing_spend_from_this_run') is not False):
                return {**retained, 'reason': 'provider_not_closed'}
        result = release_storage_pin(pins_root=pins_root, kind='activation', owner_id=owner)
        return {'status': 'released', 'launch_receipt_digest': receipt['receipt_digest'],
                'release': result, 'evidence_removed': False}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        # A cleanup failure cannot turn a completed launch into a failed one.
        return {'status': 'release_unconfirmed', 'reason': 'release_evidence_unavailable', 'evidence_removed': False}
