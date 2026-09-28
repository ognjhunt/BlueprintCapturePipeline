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


def _write(parent, name, value, *, parent_identity, replace=False):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    _require(len(raw) <= 65536)
    temporary = '.' + secrets.token_hex(16) + '.pending'
    fd, identity = _new_file(parent, temporary, parent_identity=parent_identity)
    placed = False
    try:
        view = memoryview(raw)
        while view:
            _named(parent, parent_identity, temporary, fd, identity)
            written = os.write(fd, view)
            _require(written > 0)
            view = view[written:]
        _named(parent, parent_identity, temporary, fd, identity)
        os.fsync(fd)
        _named(parent, parent_identity, temporary, fd, identity)
        if replace:
            # Only producer metadata under the exact birth gate; never payload
            # restoration. Recheck the prior destination if present as well.
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
        else:
            os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            _named(parent, parent_identity, temporary, fd, identity)
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
                _named(parent, parent_identity, temporary, fd, identity)
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
        _require(request.get('schema_version') == ATTEMPT_SCHEMA
                 and request.get('intent_id') == owner_intent_id
                 and request.get('intent_digest') == owner['intent_digest']
                 and request.get('attempt_digest') == canonical_digest(request, digest_field='attempt_digest'))
        request_path = _canonical(birth_request_raw_ref['path'])
        _require(request_path.parent == Path(owner_raw_ref['path']).parent / 'attempts'
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
