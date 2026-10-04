"""Actual scene reader/publisher admission; retirement alone takes exclusive access.

Absent policy preserves existing calls. Generation metadata is producer evidence,
never owner consent or permission to delete. Unknown lifetimes remain blockers.
"""
from __future__ import annotations

import fcntl
import functools
import hashlib
import inspect
import json
import os
import stat
import sys
from contextlib import contextmanager
from pathlib import Path

from .decision_evidence_contracts import canonical_digest

_INSTALLED_POLICY = Path('/etc/blueprint/scene-retirement-policy.json')
_POLICY_UID = 0
_SERVICE_IDENTITY = None
_MAX_JSON_BYTES = 64 * 1024
_POLICY_KEYS = {'schema_version', 'enabled', 'policy_id', 'roots', 'coordinator_path',
                'generation_store', 'journal_store', 'consumer_cohort', 'principals',
                'private_archive_allowed_classes', 'limits', 'policy_digest'}
_ADMISSIBLE = {'active', 'restored-active'}


class SceneRetirementAccessError(ValueError):
    """A fixed refusal at the real reader/publisher boundary."""


def _require(value, code='scene_retirement_access_unsafe'):
    if not value:
        raise SceneRetirementAccessError(code)


def _identity(info):
    return info.st_dev, info.st_ino, info.st_mode


def _close_owned(fd, expected):
    try:
        if _identity(os.fstat(fd)) != expected:
            return 'scene_retirement_descriptor_ownership_lost'
        os.close(fd)
    except OSError:
        return 'scene_retirement_descriptor_cleanup_failed'
    return None


def _open_owned(name, flags, *, dir_fd=None):
    before = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    _require(stat.S_ISDIR(before.st_mode) if flags & os.O_DIRECTORY else stat.S_ISREG(before.st_mode))
    fd = os.open(name, flags | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd)
    # Until this succeeds AND matches the independently named expectation the
    # numeric token is unproven. A failing/foreign first fstat is never closed.
    observed = os.fstat(fd)
    expected = _identity(before)
    _require(_identity(observed) == expected)
    try:
        _require(_identity(os.stat(name, dir_fd=dir_fd, follow_symlinks=False)) == expected)
    except BaseException:
        _close_owned(fd, expected)
        raise
    return fd, observed


def _service_identity():
    if _SERVICE_IDENTITY is not None:
        identity=_SERVICE_IDENTITY
    else:
        import pwd
        import grp
        identity=(pwd.getpwnam('blueprint').pw_uid,grp.getgrnam('blueprint').gr_gid)
    _require(type(identity) is tuple and len(identity)==2 and all(
        type(value) is int and 0<=value<2**32-1 for value in identity),
        'scene_retirement_service_identity_unproven')
    return identity


def _canonical(path):
    _require(type(path) is str and 0 < len(path) <= 4096)
    _require(len(path.encode('utf-8')) <= 4096)
    value = Path(path)
    _require(value.is_absolute() and str(value) == path
             and '..' not in value.parts and '\x00' not in path)
    return value


@contextmanager
def _opened(path, *, directory=False, protected=False):
    """No-follow component acquisition with independently proved owned cleanup."""
    path = _canonical(str(path))
    owners = []
    try:
        fd, info = _open_owned('/', os.O_RDONLY | os.O_DIRECTORY)
        owners.append((fd, _identity(info)))
        parts = path.parts[1:]
        _require(len(parts) <= 64)
        for index, part in enumerate(parts):
            _require(_identity(os.fstat(fd)) == owners[-1][1],
                     'scene_retirement_descriptor_ownership_lost')
            is_directory = directory or index < len(parts) - 1
            if protected:
                _require(info.st_uid in {0, _POLICY_UID})
                _require(not info.st_mode & 0o022
                         or (info.st_uid == 0 and info.st_mode & stat.S_ISVTX))
            fd, info = _open_owned(part, os.O_RDONLY | (os.O_DIRECTORY if is_directory else 0), dir_fd=fd)
            owners.append((fd, _identity(info)))
            _require(stat.S_ISDIR(info.st_mode) if is_directory else stat.S_ISREG(info.st_mode))
        if protected:
            _require(info.st_uid == _POLICY_UID)
            _require(not info.st_mode & 0o022)
        # Verify every retained token before handing out a leaf, even when the
        # root-only path had no child lookup. Never use a replaced parent token.
        for retained, identity in owners:
            _require(_identity(os.fstat(retained)) == identity,
                     'scene_retirement_descriptor_ownership_lost')
        yield fd, info
    finally:
        incoming = sys.exc_info()[1]
        failures = [_close_owned(fd, identity) for fd, identity in reversed(owners)]
        if any(failures):
            if incoming is not None:
                incoming.add_note('scene_retirement_descriptor_cleanup_failed')
            else:
                raise SceneRetirementAccessError('scene_retirement_descriptor_cleanup_failed')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result)
        result[key] = value
    return result


def _bytes(path, *, protected=False, required_mode=None):
    with _opened(path, protected=protected) as (fd, before):
        _require(0 < before.st_size <= _MAX_JSON_BYTES)
        if required_mode is not None:
            _require(stat.S_IMODE(before.st_mode) == required_mode,
                     'scene_retirement_policy_binding_unproven')
        raw = os.read(fd, _MAX_JSON_BYTES + 1)
        _require(len(raw) == before.st_size and len(raw) <= _MAX_JSON_BYTES)
        after = os.fstat(fd)
        _require((after.st_size, after.st_mtime_ns, after.st_ctime_ns, _identity(after))
                 == (before.st_size, before.st_mtime_ns, before.st_ctime_ns, _identity(before)))
    return raw


def _document(raw):
    # Lexical nesting is checked before allocating a decoded JSON graph.
    depth = 0
    quoted = escaped = False
    for character in raw:
        if quoted:
            if escaped:
                escaped = False
            elif character == 92:
                escaped = True
            elif character == 34:
                quoted = False
        elif character == 34:
            quoted = True
        elif character in (91, 123):
            depth += 1
            _require(depth <= 32)
        elif character in (93, 125):
            depth -= 1
            _require(depth >= 0)
    try:
        value = json.loads(raw, object_pairs_hook=_pairs,
                           parse_constant=lambda _: _require(False))
    except (ValueError, UnicodeError) as exc:
        raise SceneRetirementAccessError('scene_retirement_access_unsafe') from exc
    _require(type(value) is dict)
    return value


def _read(path, *, protected=False):
    return _document(_bytes(path, protected=protected))


def _policy_path():
    # Reader, publisher and root action use the same installation. An omitted
    # EnvironmentFile must not bypass it; a foreign path cannot choose a fence.
    selected = os.environ.get('BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE')
    fixed = str(_INSTALLED_POLICY)
    _require(selected in (None, '', fixed), 'scene_retirement_policy_binding_unproven')
    try:
        os.stat(fixed, follow_symlinks=False)
    except FileNotFoundError:
        _require(not selected, 'scene_retirement_policy_binding_unproven')
        return None
    return fixed


def _policy():
    selected = _policy_path()
    if selected is None:
        return None
    value = _document(_bytes(selected, protected=True, required_mode=0o644))
    _require(set(value) in (_POLICY_KEYS, _POLICY_KEYS | {'reference_context'})
             and value['schema_version'] == 'scene_retirement_policy.v1'
             and type(value['enabled']) is bool)
    _require(value['policy_digest'] == canonical_digest(value, digest_field='policy_digest'))
    _require(type(value['roots']) is list and len(value['roots']) <= 64)
    for row in value['roots']:
        _require(type(row) is dict and set(row) == {'root', 'storage_class', 'device'})
        _canonical(row['root'])
        _require(type(row['device']) is int and row['device'] >= 0)
    for field in ('coordinator_path', 'generation_store', 'journal_store'):
        _canonical(value[field])
    return value if value['enabled'] else None


def _admit(policy, paths):
    for raw_path in paths:
        path = _canonical(str(raw_path))
        _require(len(path.parts) <= 64)
        enrolled = [Path(row['root']) for row in policy['roots'] if path.is_relative_to(Path(row['root']))]
        # Roots are containers, not member births. Check exact logical path and
        # every bounded ancestor within an enrolled container; any closed parent
        # denies admission, including after the exclusive operation has ended.
        selected = [str(candidate) for candidate in (path, *path.parents)
                    if any(candidate.is_relative_to(root) for root in enrolled)]
        # A nonenrolled path may still be read under the coarse fence, but is
        # not promoted to owned/clear or eligible for cleanup by that read.
        for root in selected:
            key = hashlib.sha256(root.encode()).hexdigest() + '.json'
            generation_path = Path(policy['generation_store']) / key
            try:
                value = _read(generation_path)
            except FileNotFoundError:
                continue  # Legacy reads remain protected; absent birth keeps retirement.
            content = value.get('schema_version') == 'scene_content_generation.v1'
            _require(value.get('schema_version') in {'scene_member_generation.v1', 'scene_content_generation.v1',
                                                     'scene_capture_generation.v1'}
                     and value.get('canonical_path') == root
                     and value.get('state_digest') == canonical_digest(value, digest_field='state_digest'))
            _require(value.get('state') in _ADMISSIBLE, 'scene_retirement_generation_unavailable')
            with _opened(root, directory=not content) as (_, info):
                _require((value.get('dev'), value.get('ino'), value.get('mode')) == _identity(info),
                         'scene_retirement_generation_unavailable')
                if content:
                    _require(type(value.get('size_bytes')) is int and value['size_bytes'] == info.st_size
                             and type(value.get('digest')) is str and value['digest'] == 'sha256:' + path.name
                             and (value.get('uid'), value.get('gid')) == (info.st_uid, info.st_gid),
                             'scene_retirement_generation_unavailable')


@contextmanager
def scene_access(*paths):
    policy = _policy()
    if policy is None:
        yield
        return
    with _opened(policy['coordinator_path'], directory=True, protected=True) as (fd, _):
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SceneRetirementAccessError('scene_retirement_generation_unavailable') from exc
        _admit(policy, paths)
        yield
        # Closing our proved descriptor releases this lifetime's lock.


@contextmanager
def exclusive_scene_access():
    policy = _policy()
    _require(policy is not None, 'scene_retirement_disabled')
    with _opened(policy['coordinator_path'], directory=True, protected=True) as (fd, _):
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SceneRetirementAccessError('scene_retirement_reader_active') from exc
        yield policy


def scene_participant(*path_arguments):
    """Wrap a real entrypoint before it enters any preexisting inner lock."""
    def decorate(function):
        signature = inspect.signature(function)
        @functools.wraps(function)
        def admitted(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            paths = [values[key] for key in path_arguments if values.get(key) is not None]
            with scene_access(*paths):
                return function(*args, **kwargs)
        admitted.__scene_retirement_lifetime__ = 'scene_retirement_lifetime.v1'
        return admitted
    return decorate
