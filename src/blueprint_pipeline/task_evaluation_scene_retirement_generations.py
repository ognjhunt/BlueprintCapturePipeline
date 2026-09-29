"""Authenticated producer births; generation evidence never grants cleanup authority."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import stat
import sys
from contextlib import contextmanager
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_retirement_access import (
    SceneRetirementAccessError, _bytes, _canonical, _close_owned, _document, _identity,
    _opened, _open_owned, _policy, _read, _require, scene_access,
)


def _raw_reference(value):
    _require(type(value) is dict and set(value) == {'path', 'sha256', 'size_bytes'})
    _require(type(value['size_bytes']) is int and 0 < value['size_bytes'] <= 65536)
    raw = _bytes(_canonical(value['path']))
    _require(len(raw) == value['size_bytes']
             and 'sha256:' + hashlib.sha256(raw).hexdigest() == value['sha256'])
    return _document(raw)


def _guard(fd, expected):
    _require(_identity(os.fstat(fd)) == expected,
             'scene_retirement_descriptor_ownership_lost')


def _named(parent, parent_identity, name, fd, identity):
    _guard(parent, parent_identity)
    _guard(fd, identity)
    _require(_identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) == identity,
             'scene_retirement_descriptor_ownership_lost')


def _new_file(parent, name, *, parent_identity):
    _guard(parent, parent_identity)
    fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                 0o600, dir_fd=parent)
    # Independent named expectation precedes first descriptor adoption. A lost
    # parent or initial unknown token cannot authorize cleanup of that token.
    _guard(parent, parent_identity)
    expected = os.stat(name, dir_fd=parent, follow_symlinks=False)
    observed = os.fstat(fd)
    _require(stat.S_ISREG(expected.st_mode) and expected.st_nlink == 1
             and _identity(expected) == _identity(observed))
    return fd, _identity(observed)


@contextmanager
def _birth_gate(parent, key, *, parent_identity):
    name = key + '.lock'
    try:
        fd, identity = _new_file(parent, name, parent_identity=parent_identity)
    except FileExistsError:
        _guard(parent, parent_identity)
        fd, info = _open_owned(name, os.O_RDONLY, dir_fd=parent)
        identity = _identity(info)
    try:
        _named(parent, parent_identity, name, fd, identity)
        os.fsync(parent)
        _named(parent, parent_identity, name, fd, identity)
        info = os.fstat(fd)
        _require(info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o600
                 and info.st_nlink == 1)
        try:
            _named(parent, parent_identity, name, fd, identity)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SceneRetirementAccessError('scene_retirement_birth_active') from exc
        yield
    finally:
        incoming = sys.exc_info()[1]
        failure = _close_owned(fd, identity)
        if failure and incoming is None:
            raise SceneRetirementAccessError('scene_retirement_descriptor_cleanup_failed')
        if failure and incoming is not None:
            incoming.add_note('scene_retirement_descriptor_cleanup_failed')


def _write(parent, name, value, *, parent_identity, replace=False, raw_bytes=None):
    raw = (raw_bytes if raw_bytes is not None else
           json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    _require(type(raw) is bytes)
    _require(len(raw) <= 65536)
    _guard(parent,parent_identity)
    store_info=os.fstat(parent)
    owner=(store_info.st_uid,store_info.st_gid)
    _require(all(type(value) is int and 0 <= value < 2**32-1 for value in owner))
    def named():
        _guard(parent,parent_identity)
        current=os.fstat(parent)
        _require((current.st_uid,current.st_gid)==owner,'scene_retirement_generation_changed')
        _named(parent,parent_identity,temporary,fd,identity)
    temporary = '.' + secrets.token_hex(16) + '.pending'
    fd, identity = _new_file(parent, temporary, parent_identity=parent_identity)
    placed = False
    try:
        view = memoryview(raw)
        while view:
            named()
            written = os.write(fd, view)
            _require(written > 0)
            view = view[written:]
        named()
        os.fchown(fd,*owner)
        named()
        _require((os.fstat(fd).st_uid,os.fstat(fd).st_gid)==owner)
        named()
        os.fsync(fd)
        named()
        if replace:
            # Only producer metadata under the exact birth gate; never payload
            # restoration. Recheck the prior destination if present as well.
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
        else:
            os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            named()
            os.unlink(temporary, dir_fd=parent)
        placed = True
        _guard(parent, parent_identity)
        _guard(fd, identity)
        _require(_identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) == identity)
        _guard(parent, parent_identity)
        os.fsync(parent)
    finally:
        incoming = sys.exc_info()[1]
        cleanup_failure = None
        if not placed:
            try:
                named()
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            except (OSError, SceneRetirementAccessError):
                cleanup_failure = 'scene_retirement_descriptor_cleanup_failed'
        failure = _close_owned(fd, identity) or cleanup_failure
        if failure and incoming is None:
            raise SceneRetirementAccessError('scene_retirement_descriptor_cleanup_failed')
        if failure and incoming is not None:
            incoming.add_note('scene_retirement_descriptor_cleanup_failed')


def _sealed(value):
    value['state_digest'] = canonical_digest(value, digest_field='state_digest')
    return value


def birth_member(path, *, owner_intent_id, owner_raw_ref, birth_request_raw_ref, now=None):
    policy = _policy()
    if policy is None:
        return None  # The old producer owns creation in the disabled path.
    path = _canonical(str(path))
    rows = [row for row in policy['roots'] if path.is_relative_to(Path(row['root']))]
    _require(rows, 'scene_retirement_birth_outside_roots')
    with scene_access():
        from .task_evaluation_scene_owner_authority import reopen_scene_intent
        from .task_evaluation_scene_intake import ATTEMPT_SCHEMA
        owner = _raw_reference(owner_raw_ref)
        authenticated = reopen_scene_intent(owner_raw_ref, now=now)
        _require(owner == authenticated and owner['intent_id'] == owner_intent_id)
        request = _raw_reference(birth_request_raw_ref)
        preparation_only = request.get('schema_version') == 'task_evaluation_scene_preparation_attempt.v1'
        if preparation_only:
            _require(request.get('status') == 'preparation_only'
                     and type(request.get('maximum_spend_usd')) is int and request['maximum_spend_usd'] == 0
                     and request.get('provider') == 'control_plane'
                     and request.get('provider_allocation_permitted') is False
                     and request.get('paid_authority_granted') is False)
        _require((preparation_only or request.get('schema_version') == ATTEMPT_SCHEMA)
                 and request.get('intent_id') == owner_intent_id
                 and request.get('intent_digest') == owner['intent_digest']
                 and request.get('attempt_digest') == canonical_digest(request, digest_field='attempt_digest'))
        request_path = _canonical(birth_request_raw_ref['path'])
        _require(request_path.parent == Path(owner_raw_ref['path']).parent / ('preparation-attempts' if preparation_only else 'attempts')
                 and request_path.name == request.get('attempt_id', '') + '.json')
        store = Path(policy['generation_store'])
        key = hashlib.sha256(str(path).encode()).hexdigest()
        with _opened(store, directory=True) as (parent, store_info):
            _require(store_info.st_uid == os.geteuid() and stat.S_IMODE(store_info.st_mode) == 0o700)
            with _birth_gate(parent, key, parent_identity=_identity(store_info)):
                try:
                    prior = _read(store / (key + '.json'))
                except FileNotFoundError:
                    prior = None
                if prior:
                    _require(prior.get('canonical_path') == str(path)
                             and prior.get('state_digest') == canonical_digest(prior, digest_field='state_digest'))
                    if prior.get('state') in {'active', 'restored-active'}:
                        with _opened(path, directory=True) as (_, info):
                            _require(_identity(info) == (prior.get('dev'), prior.get('ino'), prior.get('mode')))
                        _require(prior.get('owner_raw_ref') == owner_raw_ref
                                 and prior.get('birth_request_raw_ref') == birth_request_raw_ref)
                        return prior
                    _require(prior.get('state') == 'retired'
                             and prior.get('birth_request_raw_ref') != birth_request_raw_ref,
                             'scene_retirement_generation_unavailable')
                elif path.exists() or path.is_symlink():
                    return None  # Legacy target is not adopted or qualified for cleanup.
                with _opened(path.parent, directory=True) as (target_parent, info):
                    _require(any(row['device'] == info.st_dev for row in rows))
                    value = _sealed(dict(schema_version='scene_member_generation.v1', canonical_path=str(path),
                        generation_id=secrets.token_hex(16), previous_generation_id=prior['generation_id'] if prior else None,
                        owner_intent_id=owner_intent_id, owner_raw_ref=owner_raw_ref,
                        birth_request_raw_ref=birth_request_raw_ref, state='birth', dev=None, ino=None, mode=None,
                        inventory_sha256=None, retirement_token=None, journal_sha256=None,
                        state_sequence=prior['state_sequence'] + 1 if prior else 0))
                    _write(parent, key + '.' + value['generation_id'] + '.birth.json', value, parent_identity=_identity(store_info))
                    _write(parent, key + '.json', value, parent_identity=_identity(store_info), replace=prior is not None)
                    _guard(target_parent, _identity(info))
                    os.mkdir(path.name, 0o750, dir_fd=target_parent)
                    _guard(target_parent, _identity(info))
                    os.fsync(target_parent)
                    with _opened(path, directory=True) as (_, born):
                        value = _sealed(dict(value, state='active', dev=born.st_dev, ino=born.st_ino,
                                             mode=born.st_mode, state_sequence=value['state_sequence'] + 1))
                    _write(parent, key + '.' + value['generation_id'] + '.active.json', value, parent_identity=_identity(store_info))
                    _write(parent, key + '.json', value, parent_identity=_identity(store_info), replace=True)
                    return value


def _retain_capture_proof(parent, store, value, *, label, parent_identity):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    _require(0 < len(raw) <= 65536, 'scene_capture_proof_bounds_invalid')
    digest = hashlib.sha256(raw).hexdigest()
    name = f'capture-{label}-{digest}.json'
    reference = {'path': str(store / name), 'sha256': 'sha256:' + digest,
                 'size_bytes': len(raw)}
    try:
        _write(parent, name, value, parent_identity=parent_identity)
    except FileExistsError:
        _require(_raw_reference(reference) == value, 'scene_capture_proof_changed')
    _require(_raw_reference(reference) == value, 'scene_capture_proof_changed')
    return reference


def _prepare_capture_parent(path, rows):
    """Create only structural ancestors below an existing enrolled owned root."""
    anchors = sorted((Path(row['root']), row['device']) for row in rows
                     if path.parent.is_relative_to(Path(row['root'])))
    _require(anchors, 'scene_capture_parent_root_unavailable')
    root, device = max(anchors, key=lambda item: len(item[0].parts))
    with _opened(root, directory=True) as (_, root_info):
        _require(root_info.st_dev == device and root_info.st_uid == os.geteuid()
                 and not root_info.st_mode & 0o022,
                 'scene_capture_parent_root_unsafe')
    current = root
    for part in path.parent.relative_to(root).parts:
        with _opened(current, directory=True) as (parent, info):
            parent_identity = _identity(info)
            _require(info.st_dev == device and info.st_uid == os.geteuid()
                     and not info.st_mode & 0o022,
                     'scene_capture_parent_unsafe')
            _guard(parent, parent_identity)
            try:
                os.mkdir(part, 0o750, dir_fd=parent)
            except FileExistsError:
                pass
            _guard(parent, parent_identity)
            with _opened(current / part, directory=True) as (child, child_info):
                _require(child_info.st_dev == device and child_info.st_uid == os.geteuid()
                         and not child_info.st_mode & 0o022,
                         'scene_capture_parent_unsafe')
                os.fsync(child)
            _guard(parent, parent_identity)
            os.fsync(parent)
        current /= part


def birth_capture_member(path, *, observation, membership_selector, membership_raw, now=None):
    """Birth a selected website capture before its first ledger or payload write.

    The caller acquires ``observation`` through the signed original-owner read
    and ``membership_raw`` through a generation-pinned GCS read. No local sidecar
    is an authority for either acquisition.
    """
    policy = _policy()
    if policy is None:
        return None
    path = _canonical(str(path))
    rows = [row for row in policy['roots'] if path.is_relative_to(Path(row['root']))]
    _require(rows, 'scene_retirement_birth_outside_roots')
    with scene_access():
        from .capture_original_owner_observer import validate_observation
        from .capture_delivery_membership import validate_capture_delivery_membership

        _require(type(observation) is dict and type(membership_selector) is dict,
                 'scene_capture_source_invalid')
        owner = validate_observation(
            observation, bucket=observation.get('bucket'),
            scene_id=observation.get('scene_id'), capture_id=observation.get('capture_id'),
            marker_generation=observation.get('completion_marker', {}).get('generation'),
            now_epoch=now,
        )
        _require(tuple(path.parts[-5:]) == (
            owner['bucket'], 'scenes', owner['scene_id'], 'captures', owner['capture_id']),
            'scene_capture_target_identity_invalid')
        membership = validate_capture_delivery_membership(
            membership_raw, selector=membership_selector, observation=owner)
        marker = owner['completion_marker']
        delivery = owner['producer_delivery']
        store = Path(policy['generation_store'])
        membership_digest = hashlib.sha256(membership_raw).hexdigest()
        membership_name = 'capture-membership-' + membership_digest + '.json'
        membership_ref = {'path': str(store / membership_name),
                          'sha256': 'sha256:' + membership_digest,
                          'size_bytes': len(membership_raw)}
        birth_delivery = {
            'schema_version': 'capture_birth_delivery.v1',
            'delivery_key': membership['delivery_key'],
            'source_finalize': membership['source_finalize'],
            'source_membership_selector': membership_selector,
            'source_membership_raw_ref': membership_ref,
            'producer_delivery': delivery,
        }
        key = hashlib.sha256(str(path).encode()).hexdigest()
        with _opened(store, directory=True) as (parent, store_info):
            store_identity = _identity(store_info)
            _require(store_info.st_uid == os.geteuid()
                     and stat.S_IMODE(store_info.st_mode) == 0o700,
                     'scene_capture_generation_store_unsafe')
            with _birth_gate(parent, key, parent_identity=store_identity):
                try:
                    prior = _read(store / (key + '.json'))
                except FileNotFoundError:
                    prior = None
                if prior is not None:
                    _require(prior.get('schema_version') == 'scene_capture_generation.v1'
                             and prior.get('canonical_path') == str(path)
                             and prior.get('state_digest') == canonical_digest(
                                 prior, digest_field='state_digest'),
                             'scene_capture_generation_invalid')
                    prior_owner = _raw_reference(prior['owner_observation_raw_ref'])
                    prior_delivery = _raw_reference(prior['birth_delivery_raw_ref'])
                    if prior['state'] in {'active', 'restored-active'}:
                        with _opened(path, directory=True) as (_, info):
                            _require(_identity(info) == (prior['dev'], prior['ino'], prior['mode']),
                                     'scene_capture_generation_changed')
                        _require(prior_owner['source_projection_digest'] == owner['source_projection_digest']
                                 and prior_delivery == birth_delivery
                                 and _raw_reference(membership_ref) == membership
                                 and prior['capture_owner_user_id'] == owner['capture_owner']['user_id']
                                 and prior['pinned_marker'] == marker,
                                 'scene_capture_active_delivery_conflict')
                        return prior
                    _require(prior['state'] == 'retired'
                             and prior_owner['capture_owner']['user_id'] == owner['capture_owner']['user_id']
                             and prior_delivery['producer_delivery']['delivery_key'] != delivery['delivery_key']
                             and prior_delivery['source_finalize']['generation'] != marker['generation']
                             and prior_delivery['producer_delivery']['raw_video'] != delivery['raw_video'],
                             'scene_capture_generation_unavailable')
                elif path.exists() or path.is_symlink():
                    return None  # Prebirth/legacy target is never adopted.
                _require(not path.exists() and not path.is_symlink(),
                         'scene_capture_target_occupied')
                _prepare_capture_parent(path, rows)
                with _opened(path.parent, directory=True) as (target_parent, info):
                    _require(any(row['device'] == info.st_dev for row in rows),
                             'scene_capture_parent_device_invalid')
                    try:
                        _write(parent, membership_name, None,
                               raw_bytes=membership_raw, parent_identity=store_identity)
                    except FileExistsError:
                        _require(_bytes(_canonical(membership_ref['path'])) == membership_raw,
                                 'scene_capture_membership_changed')
                    _require(_bytes(_canonical(membership_ref['path'])) == membership_raw,
                             'scene_capture_membership_changed')
                    owner_ref = _retain_capture_proof(
                        parent, store, owner, label='owner', parent_identity=store_identity)
                    delivery_ref = _retain_capture_proof(
                        parent, store, birth_delivery, label='delivery', parent_identity=store_identity)
                    value = _sealed(dict(
                        schema_version='scene_capture_generation.v1', canonical_path=str(path),
                        generation_id=secrets.token_hex(16),
                        previous_generation_id=prior['generation_id'] if prior else None,
                        capture_owner_user_id=owner['capture_owner']['user_id'],
                        owner_observation_raw_ref=owner_ref,
                        pinned_marker=marker,
                        birth_delivery_raw_ref=delivery_ref,
                        state='birth', dev=None, ino=None, mode=None,
                        inventory_sha256=None, retirement_token=None, journal_sha256=None,
                        state_sequence=prior['state_sequence'] + 1 if prior else 0))
                    _write(parent, key + '.' + value['generation_id'] + '.birth.json',
                           value, parent_identity=store_identity)
                    _write(parent, key + '.json', value, parent_identity=store_identity,
                           replace=prior is not None)
                    _guard(target_parent, _identity(info))
                    os.mkdir(path.name, 0o750, dir_fd=target_parent)
                    _guard(target_parent, _identity(info))
                    os.fsync(target_parent)
                    with _opened(path, directory=True) as (_, born):
                        value = _sealed(dict(value, state='active', dev=born.st_dev,
                                             ino=born.st_ino, mode=born.st_mode,
                                             state_sequence=value['state_sequence'] + 1))
                    _write(parent, key + '.' + value['generation_id'] + '.active.json',
                           value, parent_identity=store_identity)
                    _write(parent, key + '.json', value, parent_identity=store_identity,
                           replace=True)
                    return value


def capture_birth_source_projection(path):
    """Reopen one native capture's retained original source for registration.

    This is historical producer evidence, not execution or removal consent.
    An absent native generation returns None for the existing legacy caller.
    An incomplete enrolled generation refuses instead of becoming legacy.
    """
    policy = _policy()
    if policy is None:
        return None
    path = _canonical(str(path))
    store = Path(policy['generation_store'])
    key = hashlib.sha256(str(path).encode()).hexdigest()
    with scene_access():
        try:
            state = _read(store / (key + '.json'))
        except FileNotFoundError:
            return None
        if state.get('schema_version') != 'scene_capture_generation.v1':
            return None
        _require(state.get('canonical_path') == str(path)
                 and state.get('state_digest') == canonical_digest(
                     state, digest_field='state_digest')
                 and state.get('state') in {'active', 'restored-active'},
                 'scene_capture_generation_invalid')
        with _opened(path, directory=True) as (_, info):
            _require(_identity(info) == (state.get('dev'), state.get('ino'),
                                         state.get('mode')),
                     'scene_capture_generation_changed')
        owner = _raw_reference(state['owner_observation_raw_ref'])
        from .capture_original_owner_observer import validate_observation
        from .capture_delivery_membership import validate_capture_delivery_membership

        owner = validate_observation(
            owner, bucket=owner['bucket'], scene_id=owner['scene_id'],
            capture_id=owner['capture_id'],
            marker_generation=state['pinned_marker']['generation'],
            now_epoch=owner['observed_at_epoch'])
        delivery = _raw_reference(state['birth_delivery_raw_ref'])
        _require(delivery.get('schema_version') == 'capture_birth_delivery.v1'
                 and delivery.get('producer_delivery') == owner['producer_delivery']
                 and delivery.get('source_finalize') == {
                     'bucket': owner['bucket'],
                     'object_name': state['pinned_marker']['object_name'],
                     'generation': state['pinned_marker']['generation']}
                 and state['pinned_marker'] == owner['completion_marker'],
                 'scene_capture_birth_delivery_changed')
        member_ref = delivery.get('source_membership_raw_ref')
        _require(type(member_ref) is dict and set(member_ref) == {
            'path', 'sha256', 'size_bytes'}
            and type(member_ref['sha256']) is str
            and member_ref['path'] == str(store / (
                'capture-membership-' + member_ref['sha256'].removeprefix('sha256:') + '.json')),
            'scene_capture_membership_ref_invalid')
        try:
            member_raw = _bytes(_canonical(member_ref['path']))
        except FileNotFoundError as exc:
            raise SceneRetirementAccessError('scene_capture_membership_missing') from exc
        membership = validate_capture_delivery_membership(
            member_raw, selector=delivery['source_membership_selector'],
            observation=owner)
        _require(len(member_raw) == member_ref['size_bytes']
                 and 'sha256:' + hashlib.sha256(member_raw).hexdigest() == member_ref['sha256']
                 and delivery['delivery_key'] == membership['delivery_key'],
                 'scene_capture_membership_changed')
        video = next(row for row in membership['raw'] if
                     row['object_name'] == owner['producer_delivery']['raw_video']['object_name'])
        return {
            'schema_version': 'scene_capture_source_projection.v1',
            'canonical_path': str(path), 'generation_id': state['generation_id'],
            'request_id': owner['request_id'], 'scene_id': owner['scene_id'],
            'capture_id': owner['capture_id'],
            'capture_owner_user_id': owner['capture_owner']['user_id'],
            'capture_rights': owner['capture_rights'],
            'owner_observation_raw_ref': state['owner_observation_raw_ref'],
            'birth_delivery_raw_ref': state['birth_delivery_raw_ref'],
            'source_membership_raw_ref': member_ref,
            'source_membership_selector': delivery['source_membership_selector'],
            'delivery_key': owner['producer_delivery']['delivery_key'],
            'raw_video': video,
        }
