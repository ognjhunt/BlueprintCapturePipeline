"""Fence a fixed installed worker before importing or running its original module.

This conservative launch boundary holds SH for the entire process entrypoint.
A daemon therefore keeps retirement while alive. It does not attest older loaded
code, clear unknown/manual/external readers, or substitute for reference checks.
Absent/disabled policy preserves the original module arguments and exit status.
"""
from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import re
import runpy
import select
import shlex
import stat
import sys
import subprocess
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import MappingProxyType

from .task_evaluation_scene_retirement_access import _close_owned, _identity, _opened, _open_owned, _require, scene_access
from . import task_evaluation_scene_retirement_access as access
from .decision_evidence_contracts import canonical_digest


_INSTALLED_POLICY = Path('/etc/blueprint/scene-retirement-policy.json')
_POLICY_ENV = 'BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE'


_SYSTEMD_DIR = Path('/etc/systemd/system')
_KNOWN_UNITS = {
    'blueprint-pubsub-handoff-listener.service': 'pubsub_handoff_listener',
    'blueprint-task-evaluation-scene-progression.service': 'task_evaluation_scene_progression',
    'blueprint-task-evaluation-launch-preparation.service': 'task_evaluation_launch_preparation_worker',
    'blueprint-task-evaluation-launch-activation.service': 'task_evaluation_launch_activation_worker',
    'blueprint-task-evaluation-episode-compilation.service': 'task_evaluation_episode_compilation_worker',
    'blueprint-task-evaluation-sam31-preparation-execution.service': 'task_evaluation_sam31_preparation_execution',
    'blueprint-task-evaluation-launch-dispatcher.service': 'task_evaluation_launch_dispatcher',
    'blueprint-task-evaluation-launch-reconciler.service': 'task_evaluation_launch_reconciler',
    'blueprint-task-evaluation-launch-supervisor.service': 'task_evaluation_launch_supervisor',
    'blueprint-task-evaluation-policy-canary-dispatcher.service': 'task_evaluation_policy_canary_dispatcher',
    'blueprint-task-evaluation-terminal-resource-release.service': 'task_evaluation_terminal_resource_release',
    'blueprint-existing-policy-canary-watchdog.service': 'operator_policy_canary_continuation',
    'blueprint-existing-policy-canary-continuation.service': 'operator_policy_canary_continuation',
    'blueprint-task-evaluation-configured-controls-progression.service': 'task_evaluation_configured_controls_progression_worker',
}
_UNIT_PROPERTIES = frozenset(('Id', 'LoadState', 'ActiveState', 'SubState', 'MainPID',
                             'ControlPID', 'Job', 'NeedDaemonReload', 'FragmentPath',
                             'DropInPaths', 'ExecStart'))
_COHORT_ERROR = 'scene_retirement_worker_cohort_unproven'
_QUERY_BYTES = 128 * 1024
_READER_ERROR = 'scene_retirement_reader_closure_unproven'
_CONTINUOUS = {
    'blueprint_pipeline.live_pipeline_intake_service': 'blueprint-pipeline-intake.service',
    'blueprint_pipeline.agent_execution.production': 'blueprint-agent-execution.service',
}
_TRUSTED_SOURCE_ROOT = Path('/mnt/blueprint-work/scene-retirement-runtime/src')
_BOOTSTRAP = Path('/usr/lib/blueprint/scene-retirement-runtime/continuous_bootstrap.py')
_TRUSTED_DEPENDENCIES = Path('/mnt/blueprint-work/scene-retirement-runtime/dependencies')
_PRELOADED_CORE = frozenset(('blueprint_pipeline',
    'blueprint_pipeline.task_evaluation_scene_retirement_supervisor',
    'blueprint_pipeline.task_evaluation_scene_retirement_access',
    'blueprint_pipeline.decision_evidence_contracts',
    'blueprint_pipeline.task_evaluation_scene_retirement_preservation',
    'blueprint_pipeline.task_evaluation_scene_retirement_generations'))
_PROPERTIES_CONTINUOUS = frozenset(('Id', 'LoadState', 'ActiveState', 'SubState', 'MainPID',
    'ControlPID', 'Job', 'NeedDaemonReload', 'FragmentPath', 'DropInPaths', 'ExecStart',
    'ControlGroup', 'InvocationID', 'User', 'Group', 'NoNewPrivileges', 'AmbientCapabilities',
    'CapabilityBoundingSet'))
_OS_UNITS = {
    'systemd-journald.service': '/usr/lib/systemd/systemd-journald',
    'systemd-udevd.service': '/usr/lib/systemd/systemd-udevd',
    'systemd-logind.service': '/usr/lib/systemd/systemd-logind',
    'systemd-networkd.service': '/usr/lib/systemd/systemd-networkd',
    'systemd-resolved.service': '/usr/lib/systemd/systemd-resolved',
    'systemd-timesyncd.service': '/usr/lib/systemd/systemd-timesyncd',
    'ssh.service': '/usr/sbin/sshd',
}
_OS_PROPERTIES = frozenset(('Id', 'LoadState', 'ActiveState', 'SubState', 'MainPID',
                          'ControlPID', 'ControlGroup', 'FragmentPath', 'DropInPaths', 'ExecStart'))


def _query_native(properties, units, allowance):
    """Read only fixed unit state, with bounded output and an owned reaped child."""
    allowance.tick()
    command = ['/usr/bin/systemctl', 'show', '--all', '--no-pager',
               '--property=' + ','.join(sorted(properties)), '--', *sorted(units)]
    deadline = time.monotonic() + 5
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, close_fds=True,
                             env={'LC_ALL': 'C', 'PATH': '/usr/bin:/bin'})
    raw = bytearray()
    try:
        while True:
            allowance.tick()
            remaining = deadline - time.monotonic()
            _require(remaining > 0, _COHORT_ERROR)
            ready, _, _ = select.select([child.stdout], [], [], min(remaining, 0.1))
            if not ready:
                continue
            chunk = os.read(child.stdout.fileno(), min(4096, _QUERY_BYTES + 1 - len(raw)))
            allowance.tick()
            if not chunk:
                break
            raw.extend(chunk)
            _require(len(raw) <= _QUERY_BYTES, _COHORT_ERROR)
        allowance.tick()
        remaining = deadline - time.monotonic()
        _require(remaining > 0, _COHORT_ERROR)
        _require(child.wait(timeout=remaining) == 0, _COHORT_ERROR)
        allowance.tick()
        return bytes(raw)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdout.close()


def _query_systemd(allowance):
    return _query_native(_UNIT_PROPERTIES, _KNOWN_UNITS, allowance)


def _continuous_rows(allowance):
    raw = _query_native(_PROPERTIES_CONTINUOUS, _CONTINUOUS.values(), allowance)
    rows = {}
    for block in raw.decode('utf-8', errors='strict').strip().split('\n\n'):
        row = {}
        for line in block.splitlines():
            key, separator, value = line.partition('=')
            _require(separator and key in _PROPERTIES_CONTINUOUS and key not in row, _READER_ERROR)
            row[key] = value
        _require(set(row) == _PROPERTIES_CONTINUOUS and row['Id'] not in rows, _READER_ERROR)
        rows[row['Id']] = row
    _require(set(rows) == set(_CONTINUOUS.values()), _READER_ERROR)
    return rows


def _platform_rows(native):
    raw = _query_native(_OS_PROPERTIES, _OS_UNITS, native)
    rows = {}
    for block in raw.decode('utf-8', errors='strict').strip().split('\n\n'):
        row = {}
        for line in block.splitlines():
            key, separator, value = line.partition('=')
            _require(separator and key in _OS_PROPERTIES and key not in row, _READER_ERROR)
            row[key] = value
        if set(row) == _OS_PROPERTIES - {'ExecStart'}:
            _require(row['Id'] in _OS_UNITS and row['LoadState'] == 'not-found'
                     and row['ActiveState'] == 'inactive' and row['SubState'] == 'dead'
                     and row['MainPID'] == row['ControlPID'] == '0'
                     and row['ControlGroup'] == row['FragmentPath'] == row['DropInPaths'] == '',
                     _READER_ERROR)
            row['ExecStart'] = ''
        _require(set(row) == _OS_PROPERTIES and row['Id'] not in rows, _READER_ERROR)
        rows[row['Id']] = row
    _require(set(rows) == set(_OS_UNITS), _READER_ERROR)
    return rows


def _loaded_rows(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= _QUERY_BYTES, _COHORT_ERROR)
    try:
        text = raw.decode('utf-8', errors='strict')
    except UnicodeError:
        _require(False, _COHORT_ERROR)
    rows = {}
    for block in text.strip().split('\n\n'):
        fields = {}
        for line in block.splitlines():
            key, separator, value = line.partition('=')
            _require(separator and key in _UNIT_PROPERTIES and key not in fields,
                     _COHORT_ERROR)
            fields[key] = value
        _require(set(fields) == _UNIT_PROPERTIES and fields['Id'] not in rows,
                 _COHORT_ERROR)
        rows[fields['Id']] = fields
    _require(set(rows) == set(_KNOWN_UNITS), _COHORT_ERROR)
    return rows


def _unit_version(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _unit_bytes(path, allowance, *, protected):
    allowance.tick()
    with _opened(path, protected=protected) as (fd, before):
        _require(stat.S_IMODE(before.st_mode) == 0o644 and 0 < before.st_size <= 16384,
                 _COHORT_ERROR)
        raw = os.read(fd, 16385)
        allowance.tick()
        _require(len(raw) == before.st_size and _unit_version(os.fstat(fd)) == _unit_version(before),
                 _COHORT_ERROR)
        return raw, _unit_version(before)


def _idle_rows(rows):
    for unit, worker in _KNOWN_UNITS.items():
        row = rows[unit]
        expected = ('-m blueprint_pipeline.task_evaluation_scene_retirement_supervisor '
                    '--worker blueprint_pipeline.' + worker + ' --')
        _require(row['LoadState'] == 'loaded'
                 and (row['ActiveState'], row['SubState']) in {('inactive', 'dead'), ('failed', 'failed')}
                 and row['MainPID'] == row['ControlPID'] == '0'
                 and row['Job'] in {'', '0'} and row['NeedDaemonReload'] == 'no'
                 and row['FragmentPath'] == str(_SYSTEMD_DIR / unit)
                 and row['DropInPaths'] == '' and expected in row['ExecStart'], _COHORT_ERROR)


def _require_loaded_exec(row, source):
    """Bind loaded command argv to the actual fixed fragment, not a token in it.

    Unknown native show representations keep the unit. The deployed Linux
    representation must also be exercised by the final disposable native test.
    """
    try:
        commands = [line.partition('=')[2] for line in source.decode('utf-8').splitlines()
                    if line.startswith('ExecStart=')]
        _require(len(commands) == 1, _COHORT_ERROR)
        arguments = shlex.split(commands[0])
    except (ValueError, UnicodeError):
        _require(False, _COHORT_ERROR)
    _require(len(arguments) == 3 and arguments[:2] == ['/bin/bash', '-lc'], _COHORT_ERROR)
    script = arguments[2].replace('$$', '$')
    prefix = '{ path=/bin/bash ; argv[]=/bin/bash -lc ' + script + ' ; ignore_errors=no ; '
    value = row['ExecStart']
    _require(value.startswith(prefix) and value.endswith(' }'), _COHORT_ERROR)
    # Only status fields may follow the exact argv. Another command object or
    # duplicate argv/path token cannot be hidden in the remainder.
    suffix = value[len(prefix):-2]
    fields = {}
    for field in suffix.split(' ; '):
        key, separator, item = field.partition('=')
        _require(separator and key in {'start_time', 'stop_time', 'pid', 'code', 'status'}
                 and key not in fields, _COHORT_ERROR)
        fields[key] = item
    _require(set(fields) == {'start_time', 'stop_time', 'pid', 'code', 'status'}
             and fields['pid'] == '0', _COHORT_ERROR)


def require_inactive_known_workers(allowance):
    """Observe only known idle installed units. Caller must hold the actual EX fence.

    This never clears manual/unknown readers, external lifetimes, reference
    obligations, or the binding of a future interpreter to this exact release.
    Those remain independent required action evidence.
    """
    allowance.tick()
    before_rows = _loaded_rows(_query_systemd(allowance))
    _idle_rows(before_rows)
    proof = {}
    source_root = Path(__file__).resolve().parents[2] / 'deploy' / 'systemd'
    for unit in sorted(_KNOWN_UNITS):
        source, source_version = _unit_bytes(source_root / unit, allowance, protected=False)
        installed, installed_version = _unit_bytes(_SYSTEMD_DIR / unit, allowance, protected=True)
        _require(source == installed, _COHORT_ERROR)
        _require_loaded_exec(before_rows[unit], source)
        proof[unit] = (source, source_version, installed_version)
    after_rows = _loaded_rows(_query_systemd(allowance))
    _idle_rows(after_rows)
    _require(after_rows == before_rows, _COHORT_ERROR)
    for unit, (raw, source_version, installed_version) in proof.items():
        source_after, source_identity = _unit_bytes(source_root / unit, allowance, protected=False)
        installed_after, installed_identity = _unit_bytes(_SYSTEMD_DIR / unit, allowance, protected=True)
        _require(source_after == installed_after == raw
                 and source_identity == source_version and installed_identity == installed_version,
                 _COHORT_ERROR)
    allowance.tick()
    return {'scope': 'installed_known_worker_quiescence', 'worker_units': sorted(_KNOWN_UNITS),
            'unit_sha256': {unit: hashlib.sha256(value[0]).hexdigest() for unit, value in proof.items()},
            'unknown_readers_cleared': False, 'external_lifetimes_cleared': False}


class _NativeObservation:
    """Finite kernel metadata shares the action's original sticky deadline."""
    def __init__(self, allowance):
        self.allowance, self.bytes, self.entries = allowance, 0, 0

    def tick(self):
        self.allowance.tick()

    def consume(self, *, size=0, entries=0):
        self.tick()
        _require(type(size) is int and size >= 0 and type(entries) is int and entries >= 0,
                 _READER_ERROR)
        _require(self.bytes + size <= 64 * 1024 * 1024 and self.entries + entries <= 32768,
                 _READER_ERROR)
        self.bytes += size
        self.entries += entries


def _native_bytes(path, native, *, cap, protected=False):
    native.tick()
    with _opened(path, protected=protected) as (fd, before):
        raw = bytearray()
        while True:
            native.tick()
            block = os.read(fd, min(65536, cap + 1 - len(raw)))
            native.consume(size=len(block))
            _require(_unit_version(os.fstat(fd)) == _unit_version(before), _READER_ERROR)
            if not block:
                break
            _require(len(raw) + len(block) <= cap, _READER_ERROR)
            raw.extend(block)
        return bytes(raw), _unit_version(before)


@contextmanager
def _actual_proc(native):
    """No path override: only the actual Linux procfs grants kernel identities."""
    _require(sys.platform == 'linux', _READER_ERROR)
    with _opened('/proc', directory=True, protected=True) as (fd, _):
        native.tick()
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        storage = ctypes.create_string_buffer(256)
        _require(libc.fstatfs(fd, ctypes.byref(storage)) == 0
                 and ctypes.c_long.from_buffer(storage).value == 0x9FA0, _READER_ERROR)
        yield fd
        native.tick()


def _proc_fields(pid, native):
    root = Path('/proc') / str(pid)
    with _opened(root, directory=True) as (parent, before):
        raw, _ = _native_bytes(root / 'stat', native, cap=8192)
        end = raw.rfind(b') ')
        _require(end > 0 and raw[:raw.find(b' ')].decode() == str(pid), _READER_ERROR)
        fields = raw[end + 2:].split()
        _require(len(fields) >= 20 and fields[1].isdigit() and fields[19].isdigit(), _READER_ERROR)
        status_raw, _ = _native_bytes(root / 'status', native, cap=16384)
        status = {}
        for line in status_raw.decode('utf-8', errors='strict').splitlines():
            key, separator, value = line.partition(':')
            _require(separator and key not in status, _READER_ERROR)
            status[key] = value.strip()
        required = {'Uid', 'Gid', 'Threads', 'CapEff', 'CapPrm', 'CapInh', 'CapAmb', 'CapBnd', 'NoNewPrivs'}
        _require(required <= set(status), _READER_ERROR)
        uid, gid = status['Uid'].split(), status['Gid'].split()
        _require(len(uid) == len(gid) == 4 and all(item.isdigit() for item in uid + gid), _READER_ERROR)
        cgroup, _ = _native_bytes(root / 'cgroup', native, cap=16384)
        groups = cgroup.decode('utf-8', errors='strict').splitlines()
        _require(len(groups) == 1 and groups[0].startswith('0::/'), _READER_ERROR)
        command, _ = _native_bytes(root / 'cmdline', native, cap=16384)
        kernel = status.get('Kthread') == '1'
        links = {}
        if not kernel:
            for name in ('exe', 'cwd'):
                native.tick()
                value = os.readlink(name, dir_fd=parent)
                _require(len(value.encode()) <= 4096 and '\x00' not in value, _READER_ERROR)
                links[name] = value
        again, _ = _native_bytes(root / 'stat', native, cap=8192)
        _require(again.rfind(b') ') > 0 and again[again.rfind(b') ') + 2:].split()[19] == fields[19]
                 and _unit_version(os.fstat(parent)) == _unit_version(before), _READER_ERROR)
        return dict(pid=pid, ppid=int(fields[1]), start_ticks=int(fields[19]),
            uid=list(map(int, uid)), gid=list(map(int, gid)), status={key:status[key] for key in required},
            cgroup=groups[0][3:], cmdline=command.decode('utf-8', errors='strict').rstrip('\0').split('\0'),
            kernel_thread=kernel, **links)


def _under_roots(value, roots):
    value = value.removesuffix(' (deleted)')
    return value.startswith('/') and any(value == str(root) or value.startswith(str(root) + '/') for root in roots)


def _reader_inodes(roots, native):
    """Bounded actual no-follow inventory; path aliases cannot clear an inode."""
    rows, directories = {}, set()
    def visit(path):
        native.tick()
        with _opened(path, directory=True) as (parent, before):
            identity = (before.st_dev, before.st_ino)
            _require(identity not in directories, _READER_ERROR)
            directories.add(identity)
            rows[str(path)] = _unit_version(before)
            def names():
                result = []
                with os.scandir(parent) as stream:
                    for item in stream:
                        native.consume(entries=1)
                        _require(len(result) < 10000 and item.name not in {'.', '..'}, _READER_ERROR)
                        result.append(item.name)
                return sorted(result)
            selected = names()
            for name in selected:
                native.tick()
                info = os.stat(name, dir_fd=parent, follow_symlinks=False)
                child = path / name
                _require(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode), _READER_ERROR)
                if stat.S_ISDIR(info.st_mode):
                    visit(child)
                    _require(rows[str(child)] == _unit_version(info), _READER_ERROR)
                else:
                    rows[str(child)] = _unit_version(info)
            _require(names() == selected and _unit_version(os.fstat(parent)) == _unit_version(before),
                     _READER_ERROR)
    for root in roots:
        visit(root)
    _require(rows, _READER_ERROR)
    return rows


def _mount_path(raw):
    raw = re.sub(rb'\\(040|011|012|134)', lambda match: bytes([int(match[1], 8)]), raw)
    _require(0 < len(raw) <= 4096 and b'\\' not in raw, _READER_ERROR)
    text = os.fsdecode(raw)
    path = Path(text)
    _require(path.is_absolute() and str(path) == text and '..' not in path.parts
             and '\x00' not in text and len(path.parts) <= 64, _READER_ERROR)
    return path


def _mount_rows(raw, native):
    result, identities = [], set()
    for line in raw.splitlines():
        native.consume(entries=1)
        left, separator, right = line.partition(b' - ')
        fields, tail = left.split(), right.split()
        _require(separator and len(fields) >= 6 and len(tail) == 3 and len(result) < 4096
                 and fields[0].isdigit() and fields[1].isdigit() and fields[0] not in identities
                 and re.fullmatch(rb'[0-9]{1,10}:[0-9]{1,10}', fields[2]), _READER_ERROR)
        identities.add(fields[0])
        major, minor = fields[2].split(b':')
        result.append((os.makedev(int(major), int(minor)), _mount_path(fields[3]), _mount_path(fields[4])))
    _require(result, _READER_ERROR)
    return result


def _reader_namespaces(parent, native):
    rows = []
    for kind in ('pid', 'user', 'mnt'):
        native.tick()
        value = os.readlink('ns/' + kind, dir_fd=parent)
        _require(re.fullmatch(kind + r':\[[0-9]+\]', value), _READER_ERROR)
        info = os.stat('ns/' + kind, dir_fd=parent)
        rows.append((value, info.st_dev, info.st_ino))
    return tuple(rows)


@contextmanager
def _reader_root(parent, native):
    """Only an independently proved actual proc magic link may be followed."""
    native.tick()
    before = os.stat('root', dir_fd=parent)
    _require(stat.S_ISDIR(before.st_mode), _READER_ERROR)
    fd = os.open('root', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC, dir_fd=parent)
    observed = os.fstat(fd)
    expected = _identity(before)
    _require(_identity(observed) == expected, _READER_ERROR)
    try:
        _require(_unit_version(os.stat('root', dir_fd=parent)) == _unit_version(observed), _READER_ERROR)
        yield fd, observed
        native.tick()
        _require(_identity(os.fstat(fd)) == expected
                 and _unit_version(os.stat('root', dir_fd=parent)) == _unit_version(observed), _READER_ERROR)
    finally:
        _require(_close_owned(fd, expected) is None, _READER_ERROR)


def _reader_view(pid, roots, native):
    """Authenticate every current mount route; unknown/subtree aliases keep."""
    own = Path('/proc') / str(os.getpid())
    if not hasattr(native, 'own_view'):
        with _opened(own, directory=True) as (parent, _), _reader_root(parent, native) as (_, root):
            namespaces = _reader_namespaces(parent, native)
            raw, _ = _native_bytes(own / 'mountinfo', native, cap=1024 * 1024)
            mounts = _mount_rows(raw, native)
            physical = {}
            for selected in roots:
                candidates = [row for row in mounts if selected.is_relative_to(row[2])]
                _require(candidates, _READER_ERROR)
                device, prefix, point = max(candidates, key=lambda row: len(row[2].parts))
                _require(device == native.inventory[str(selected)][0], _READER_ERROR)
                physical[str(selected)] = prefix / selected.relative_to(point)
            _require(_native_bytes(own / 'mountinfo', native, cap=1024 * 1024)[0] == raw, _READER_ERROR)
            native.own_view = (namespaces, _identity(root), physical)
            native.mount_views = {}
    root = Path('/proc') / str(pid)
    with _opened(root, directory=True) as (parent, _), _reader_root(parent, native) as (root_fd, root_info):
        namespaces = _reader_namespaces(parent, native)
        own_namespaces, own_root, physical = native.own_view
        _require(namespaces[:2] == own_namespaces[:2] and _identity(root_info) == own_root, _READER_ERROR)
        raw, _ = _native_bytes(root / 'mountinfo', native, cap=1024 * 1024)
        key = namespaces[2]
        if key not in native.mount_views:
            _require(len(native.mount_views) < 16, _READER_ERROR)
            native.mount_views[key] = (raw, _mount_rows(raw, native))
        selected_raw, mounts = native.mount_views[key]
        _require(raw == selected_raw, _READER_ERROR)
        for selected in roots:
            version = native.inventory[str(selected)]
            target = physical[str(selected)]
            routes = []
            for device, prefix, point in mounts:
                if device != version[0]:
                    continue
                # A mount rooted below an enrolled root hides an unaccounted
                # subtree alias. Whole-root/ancestor aliases need exact route proof.
                _require(not prefix.is_relative_to(target) or prefix == target, _READER_ERROR)
                if target.is_relative_to(prefix):
                    route = point / target.relative_to(prefix)
                    _require(len(routes) < 16, _READER_ERROR)
                    routes.append(route)
            _require(routes, _READER_ERROR)
            for route in set(routes):
                with ExitStack() as stack:
                    current = root_fd
                    for part in route.parts[1:]:
                        native.tick()
                        current, info = _open_owned(part, os.O_RDONLY | os.O_DIRECTORY, dir_fd=current)
                        stack.callback(_reader_close, current, _identity(info))
                    info = os.fstat(current)
                    _require((info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid)
                             == tuple(version[:5]), _READER_ERROR)
            _require(_native_bytes(root / 'mountinfo', native, cap=1024 * 1024)[0] == raw, _READER_ERROR)
        _require(_reader_namespaces(parent, native) == namespaces, _READER_ERROR)
        return namespaces, own_root, hashlib.sha256(raw).hexdigest()


def _reader_close(fd, identity):
    _require(_close_owned(fd, identity) is None, _READER_ERROR)


def _exposure(pid, row, roots, native):
    """Observe physical cwd/FD/maps plus complete current mount/root views."""
    if row['kernel_thread']:
        return
    _require(type(getattr(native, 'inodes', None)) is set and native.inodes, _READER_ERROR)
    view = _reader_view(pid, roots, native)
    _require(not _under_roots(row['cwd'], roots), _READER_ERROR)
    root = Path('/proc') / str(pid)
    with _opened(root, directory=True) as (parent, _):
        native.tick()
        cwd = os.stat('cwd', dir_fd=parent)
        _require((cwd.st_dev, cwd.st_ino) not in native.inodes, _READER_ERROR)
    with _opened(root / 'fd', directory=True) as (parent, before):
        def names():
            native.tick()
            result = []
            with os.scandir(parent) as stream:
                for item in stream:
                    native.consume(entries=1)
                    _require(item.name.isdecimal() and len(result) < 1024, _READER_ERROR)
                    result.append(item.name)
            return sorted(result)
        selected = names()
        descriptors = []
        for name in selected:
            native.tick()
            info = os.stat(name, dir_fd=parent)
            value = os.readlink(name, dir_fd=parent)
            _require((info.st_dev, info.st_ino) not in native.inodes and len(value.encode()) <= 4096
                     and not _under_roots(value, roots)
                     and _unit_version(os.stat(name, dir_fd=parent)) == _unit_version(info), _READER_ERROR)
            descriptors.append((name, value, _unit_version(info)))
        _require(names() == selected and _unit_version(os.fstat(parent)) == _unit_version(before), _READER_ERROR)
    raw, _ = _native_bytes(root / 'maps', native, cap=512 * 1024)
    for line in raw.splitlines():
        native.consume(entries=1)
        fields = line.split(None, 5)
        _require(len(fields) >= 5 and re.fullmatch(rb'[0-9a-f]+-[0-9a-f]+', fields[0])
                 and re.fullmatch(rb'[r-][w-][x-][ps]', fields[1])
                 and re.fullmatch(rb'[0-9a-f]+', fields[2])
                 and re.fullmatch(rb'[0-9a-f]+:[0-9a-f]+', fields[3]) and fields[4].isdigit(), _READER_ERROR)
        major, minor = fields[3].split(b':')
        _require(not int(fields[4]) or (os.makedev(int(major, 16), int(minor, 16)), int(fields[4]))
                 not in native.inodes, _READER_ERROR)
        if len(fields) == 6:
            _require(not _under_roots(os.fsdecode(fields[5]), roots), _READER_ERROR)
    _require(_native_bytes(root / 'maps', native, cap=512 * 1024)[0] == raw
             and _reader_view(pid, roots, native) == view, _READER_ERROR)
    with _opened(root, directory=True) as (parent, _):
        _require(_unit_version(os.stat('cwd', dir_fd=parent)) == _unit_version(cwd), _READER_ERROR)
    return dict(view=view, cwd_identity=_unit_version(cwd), descriptors=tuple(descriptors),
                maps_sha256=hashlib.sha256(raw).hexdigest())


def _snapshot(proc, roots, native):
    native.tick()
    with os.scandir(proc) as stream:
        names = []
        for item in stream:
            native.consume(entries=1)
            if item.name.isdecimal():
                _require(len(item.name) <= 10 and len(names) < 512, _READER_ERROR)
                names.append(int(item.name))
    result = {}
    for pid in sorted(names):
        row = _proc_fields(pid, native)
        # The consent-bound root operation itself necessarily holds payload
        # descriptors. It cannot grant another process this exception.
        if pid != os.getpid():
            row['reader_observation'] = _exposure(pid, row, roots, native)
        result[pid] = row
    native.tick()
    return result


def _source_bundle(policy, native):
    """Bind the exact fixed installed catalogue without importing daemon code."""
    package = _TRUSTED_SOURCE_ROOT / 'blueprint_pipeline'
    _require(Path(__file__).absolute() == package / 'task_evaluation_scene_retirement_supervisor.py', _READER_ERROR)
    engine_raw, _ = _native_bytes(package / 'task_evaluation_scene_retirement.py', native,
                                 cap=1024 * 1024, protected=True)
    tree = ast.parse(engine_raw)
    assignments = [node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == '_COHORT_CALLS' for target in node.targets)]
    _require(len(assignments) == 1, _READER_ERROR)
    catalogue = ast.literal_eval(assignments[0])
    _require(type(catalogue) is dict and 0 < len(catalogue) <= 64
             and all(type(key) is str and type(value) is str for key, value in catalogue.items()), _READER_ERROR)
    expected = {'blueprint_pipeline.' + module + ':' + name for module, names in catalogue.items()
                for name in names.split()}
    rows = policy['consumer_cohort']
    _require(type(rows) is list and len(rows) == len(expected) <= 256
             and {row['entrypoint'] for row in rows} == expected, _READER_ERROR)
    hashes = {}
    for row in rows:
        _require(set(row) == {'entrypoint', 'installed_source_sha', 'lifetime_contract_version'}
                 and row['lifetime_contract_version'] == 'scene_retirement_lifetime.v1', _READER_ERROR)
        module = row['entrypoint'].split(':')[0]
        _require(module not in hashes or hashes[module] == row['installed_source_sha'], _READER_ERROR)
        hashes[module] = row['installed_source_sha']
    names = set(hashes) | set(_CONTINUOUS) | _PRELOADED_CORE
    result, total = {}, 0
    for module in sorted(names):
        _require(module == 'blueprint_pipeline' or
                 re.fullmatch(r'blueprint_pipeline\.[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*', module),
                 _READER_ERROR)
        path = (package / '__init__.py' if module == 'blueprint_pipeline'
                else _TRUSTED_SOURCE_ROOT / (module.replace('.', '/') + '.py'))
        raw, version = _native_bytes(path, native, cap=1024 * 1024, protected=True)
        digest = 'sha256:' + hashlib.sha256(raw).hexdigest()
        _require(module not in hashes or hashes[module] == digest, _READER_ERROR)
        total += len(raw)
        _require(total <= 16 * 1024 * 1024, _READER_ERROR)
        result[module] = dict(path=str(path), sha256=digest, identity=list(version), raw=raw)
    return result


def _source_records(bundle):
    return {name:{key:value for key, value in row.items() if key != 'raw'} for name, row in bundle.items()}


def _require_preloaded_core(bundle):
    """Bind current core claims to the actual isolated root loader's first read."""
    bootstrap = sys.modules.get('__main__')
    _require(bootstrap is not None and bootstrap.__dict__.get('__file__') == str(_BOOTSTRAP), _READER_ERROR)
    loader_type = bootstrap.__dict__.get('_SourceOnly')
    _require(type(loader_type) is type(importlib.abc.MetaPathFinder)
             and _PRELOADED_CORE <= bundle.keys(), _READER_ERROR)
    for name in sorted(_PRELOADED_CORE):
        loaded = sys.modules.get(name)
        _require(loaded is not None, _READER_ERROR)
        spec = loaded.__dict__.get('__spec__')
        _require(spec is not None, _READER_ERROR)
        loader = spec.loader
        # No evidence callback from a different loader may run before this
        # exact class proof. The kernel command/unit proof precedes this seam.
        _require(type(loader) is loader_type, _READER_ERROR)
        values, identities = vars(loader).get('values'), vars(loader).get('source_identities')
        _require(type(values) is MappingProxyType and type(identities) is MappingProxyType
                 and set(values) == set(identities) == _PRELOADED_CORE, _READER_ERROR)
        retained, identity = values[name], identities[name]
        current = bundle[name]
        _require(type(retained) is tuple and len(retained) == 2 and type(retained[1]) is bytes
                 and type(identity) is tuple and len(identity) == 9
                 and current['path'] == str(retained[0]) and current['raw'] == retained[1]
                 and current['identity'] == list(identity), _READER_ERROR)


def _deployment_identity_dropin(row, module, native):
    """One fixed deployment identity file; it cannot alter the native command.

    This is descriptive deployment metadata. The protected snapshot and exact
    policy cohort hashes remain the code authority, including when the legacy
    environment names a service-owned checkout or virtualenv.
    """
    if not row['DropInPaths']:
        return None
    _require(module == 'blueprint_pipeline.live_pipeline_intake_service', _READER_ERROR)
    selected = _SYSTEMD_DIR / 'blueprint-pipeline-intake.service.d/90-blueprint-deploy-identity.conf'
    environment = selected.with_suffix('.env')
    _require(row['DropInPaths'] == str(selected), _READER_ERROR)
    raw, version = _unit_bytes(selected, native, protected=True)
    expected = ('# Managed by scripts/deploy_control_plane_commit.py.\n'
                '# Loaded after the base unit credential EnvironmentFile.\n'
                '[Service]\nEnvironmentFile=' + str(environment) + '\nTimeoutStartSec=300s\n').encode()
    _require(raw == expected, _READER_ERROR)
    identity_raw, identity_version = _unit_bytes(environment, native, protected=True)
    lines = identity_raw.decode('utf-8').splitlines()
    _require(len(lines) == 7 and lines[:2] == [
        '# Managed by scripts/deploy_control_plane_commit.py.',
        '# Contains deployment identity only; no credentials.'], _READER_ERROR)
    keys = ('BLUEPRINT_PIPELINE_REPO', 'BLUEPRINT_SOURCE_COMMIT', 'BLUEPRINT_PIPELINE_PYTHON',
            'PYTHONPATH', 'BLUEPRINT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT')
    values = {}
    for key, line in zip(keys, lines[2:], strict=True):
        actual, separator, value = line.partition('=')
        _require(actual == key and separator and value and not any(character.isspace() for character in value)
                 and '\x00' not in value, _READER_ERROR)
        values[key] = value
    for key in ('BLUEPRINT_PIPELINE_REPO', 'BLUEPRINT_PIPELINE_PYTHON', 'PYTHONPATH'):
        path = Path(values[key])
        _require(path.is_absolute() and str(path) == values[key] and '..' not in path.parts, _READER_ERROR)
    _require(re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', values['BLUEPRINT_SOURCE_COMMIT'])
             and values['PYTHONPATH'] == values['BLUEPRINT_PIPELINE_REPO'] + '/src'
             and values['BLUEPRINT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT'] ==
             '/var/lib/blueprint/pipeline-control-plane/scene-object-discoveries', _READER_ERROR)
    return dict(path=str(selected), identity=list(version), sha256=hashlib.sha256(raw).hexdigest(),
                environment_path=str(environment), environment_identity=list(identity_version),
                environment_sha256=hashlib.sha256(identity_raw).hexdigest())


def _continuous_loaded_command(value, command):
    expected = '{ path=/usr/bin/python3 ; argv[]=' + ' '.join(command).removeprefix('!') + ' ; ignore_errors=no ; '
    _require(value.startswith(expected) and value.endswith(' }'), _READER_ERROR)
    fields = {}
    for field in value[len(expected):-2].split(' ; '):
        key, separator, item = field.partition('=')
        _require(separator and key in {'start_time', 'stop_time', 'pid', 'code', 'status'}
                 and key not in fields, _READER_ERROR)
        fields[key] = item
    _require(set(fields) == {'start_time', 'stop_time', 'pid', 'code', 'status'}
             and fields['pid'].isdecimal()
             and (fields['status'].lstrip('-').isdecimal()
                  or (fields['status'] == '0/0' and fields['code'] == '(null)')), _READER_ERROR)


def _exact_continuous_unit(row, module, native):
    unit = _CONTINUOUS[module]
    source_root = Path(__file__).parents[2] / 'deploy/systemd'
    source, _ = _unit_bytes(source_root / unit, native, protected=True)
    installed, version = _unit_bytes(_SYSTEMD_DIR / unit, native, protected=True)
    _require(source == installed and row['Id'] == unit and row['LoadState'] == 'loaded'
             and row['FragmentPath'] == str(_SYSTEMD_DIR / unit)
             and row['NeedDaemonReload'] == 'no' and row['ControlPID'] == '0'
             and row['Job'] in {'', '0'} and row['User'] == row['Group'] == 'blueprint'
             and row['NoNewPrivileges'] == 'yes' and row['AmbientCapabilities'] == ''
             and set(row['CapabilityBoundingSet'].split()) == {'cap_setuid', 'cap_setgid'}, _READER_ERROR)
    starts = [line.partition('=')[2] for line in source.decode().splitlines() if line.startswith('ExecStart=')]
    _require(len(starts) == 1, _READER_ERROR)
    command = shlex.split(starts[0])
    _require(command == ['!/usr/bin/python3', '-I', '-S', str(_BOOTSTRAP), '--continuous-module', module]
             or command == ['!/usr/bin/python3', '-I', '-S', str(_BOOTSTRAP), '--continuous-module', module, '--'],
             _READER_ERROR)
    _continuous_loaded_command(row['ExecStart'], command)
    drop_in = _deployment_identity_dropin(row, module, native)
    return dict(unit=unit, fragment_identity=list(version), fragment_sha256=hashlib.sha256(source).hexdigest(),
                command=command[1:], loaded_exec_start=row['ExecStart'], drop_in=drop_in)


def _boot_path(policy, row):
    return Path(policy['journal_store']) / 'processes' / (str(row['pid']) + '-' + str(row['start_ticks']) + '.json')


def _boot_record(policy, row, module, unit, sources, boot_id, native):
    raw, version = _native_bytes(_boot_path(policy, row), native, cap=65536, protected=True)
    _require(stat.S_IMODE(version[2]) == 0o600, _READER_ERROR)
    value = access._document(raw)
    fields = {'schema_version', 'module', 'pid', 'start_ticks', 'boot_id', 'cgroup', 'invocation_id',
              'interpreter', 'sources', 'policy_digest', 'coordinator_identity', 'target_uid', 'target_gid',
              'unit', 'boot_digest'}
    uid, gid = access._service_identity()
    _require(set(value) == fields and value['schema_version'] == 'scene_retirement_current_boot.v1'
             and value['boot_digest'] == canonical_digest(value, digest_field='boot_digest')
             and value['module'] == module and value['pid'] == row['pid']
             and value['start_ticks'] == row['start_ticks'] and value['boot_id'] == boot_id
             and value['cgroup'] == row['cgroup'] == unit['ControlGroup']
             and value['invocation_id'] == unit['InvocationID']
             and value['sources'] == sources and value['policy_digest'] == policy['policy_digest']
             and value['target_uid'] == uid and value['target_gid'] == gid
             and row['uid'] == [uid] * 4 and row['gid'] == [gid] * 4
             and row['status']['NoNewPrivs'] == '1'
             and all(int(row['status'][key], 16) == 0 for key in ('CapEff', 'CapPrm', 'CapInh', 'CapAmb')),
             _READER_ERROR)
    info = _exact_continuous_unit(unit, module, native)
    _require(value['unit'] == info and row['cmdline'] == ['/usr/bin/python3', *info['command']], _READER_ERROR)
    interpreter, identity = _native_bytes(row['exe'], native, cap=64 * 1024 * 1024, protected=True)
    _require(value['interpreter'] == dict(path=row['exe'], identity=list(identity),
             sha256='sha256:' + hashlib.sha256(interpreter).hexdigest()), _READER_ERROR)
    with _opened(policy['coordinator_path'], directory=True, protected=True) as (_, coordinator):
        _require(value['coordinator_identity'] == list(access._identity(coordinator)), _READER_ERROR)
    return raw, version


def _platform_processes(rows, snapshot, native):
    """Only fixed protected OS service MainPIDs; sessions/children are excluded."""
    result = set()
    for name, executable in _OS_UNITS.items():
        unit = rows[name]
        if unit['LoadState'] == 'not-found' or unit['ActiveState'] in {'inactive', 'failed'}:
            _require(unit['MainPID'] == unit['ControlPID'] == '0', _READER_ERROR)
            continue
        _require(unit['LoadState'] == 'loaded' and unit['ActiveState'] == 'active'
                 and unit['SubState'] == 'running' and unit['ControlPID'] == '0'
                 and unit['MainPID'].isdecimal() and int(unit['MainPID']) in snapshot
                 and unit['DropInPaths'] == '', _READER_ERROR)
        pid = int(unit['MainPID'])
        row = snapshot[pid]
        selected = Path(executable).resolve(strict=True)
        _require(row.get('exe') == str(selected) and row['cgroup'] == unit['ControlGroup']
                 and row['cgroup'] == '/system.slice/' + name, _READER_ERROR)
        binary, identity = _native_bytes(selected, native, cap=64 * 1024 * 1024, protected=True)
        kernel = os.stat(Path('/proc') / str(pid) / 'exe')
        _require(binary.startswith(b'\x7fELF') and _unit_version(kernel) == identity, _READER_ERROR)
        fragment = Path(unit['FragmentPath']).resolve(strict=True)
        _require(fragment.parent == Path('/usr/lib/systemd/system') and fragment.name == name,
                 _READER_ERROR)
        source, _ = _native_bytes(fragment, native, cap=65536, protected=True)
        starts = [line.partition('=')[2] for line in source.decode().splitlines() if line.startswith('ExecStart=')]
        _require(len(starts) == 1, _READER_ERROR)
        source_exe = shlex.split(starts[0])[0].lstrip('-+!:@')
        _require(Path(source_exe).resolve(strict=True) == selected
                 and unit['ExecStart'].startswith('{ path=' + source_exe + ' ; argv[]='), _READER_ERROR)
        result.add(pid)
    first = snapshot.get(1)
    if first is not None:
        _require(first['ppid'] == 0 and first.get('exe') == '/usr/lib/systemd/systemd'
                 and first['uid'] == [0] * 4 and first['cgroup'] == '/init.scope', _READER_ERROR)
        binary, identity = _native_bytes(first['exe'], native, cap=64 * 1024 * 1024, protected=True)
        _require(binary.startswith(b'\x7fELF')
                 and _unit_version(os.stat('/proc/1/exe')) == identity, _READER_ERROR)
        result.add(1)
    return result


def _require_current_reader_closure(policy, allowance):
    """Enforce actual boot/current-process authority under the caller's EX fence.

    No observed absence clears a legacy/manual/external lifetime. Known installed
    workers must be idle; only the two freshly root-bootstrapped fixed continuous
    processes may be alive. Unknown processes with the service/root identity or
    actual member exposure, and every unowned descendant, refuse retirement.
    """
    allowance.tick()

    _require(sys.platform == 'linux' and os.geteuid() == 0 and type(policy) is dict
             and policy.get('consumer_cohort'), _READER_ERROR)
    native = _NativeObservation(allowance)
    _require(access._policy() == policy, _READER_ERROR)
    sources = _source_records(_source_bundle(policy, native))
    require_inactive_known_workers(allowance)
    units = _continuous_rows(allowance)
    platforms = _platform_rows(native)
    roots = [Path(row['root']) for row in policy['roots']]
    boot_raw, _ = _native_bytes('/proc/sys/kernel/random/boot_id', native, cap=128)
    boot_id = boot_raw.decode('ascii').strip()
    _require(re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', boot_id), _READER_ERROR)
    with _actual_proc(native) as proc:
        native.inventory = _reader_inodes(roots, native)
        native.inodes = {tuple(version[:2]) for version in native.inventory.values()}
        before = _snapshot(proc, roots, native)
        platform = _platform_processes(platforms, before, native)
        active, receipts = {}, {}
        for module, name in _CONTINUOUS.items():
            unit = units[name]
            if unit['ActiveState'] in {'inactive', 'failed'}:
                _require(unit['SubState'] in {'dead', 'failed'} and unit['MainPID'] == unit['ControlPID'] == '0',
                         _READER_ERROR)
                # Even idle declarations require the actual current installed
                # root fragment and command; a stopped legacy unit isn't a grant.
                _exact_continuous_unit(unit, module, native)
                continue
            _require(unit['ActiveState'] == 'active' and unit['SubState'] == 'running'
                     and unit['MainPID'].isdigit() and int(unit['MainPID']) in before, _READER_ERROR)
            pid = int(unit['MainPID'])
            active[pid] = unit
            receipts[pid] = _boot_record(policy, before[pid], module, unit, sources, boot_id, native)
        service_uid, _ = access._service_identity()
        for pid, row in before.items():
            if pid == os.getpid() or row['kernel_thread']:
                continue
            if pid in active:
                continue
            # Native OS MainPIDs were independently bound above; no child,
            # session, root Python or manual process inherits that classification.
            _require(pid in platform or (0 not in row['uid'] and service_uid not in row['uid']
                                  and row['ppid'] not in active), _READER_ERROR)
        after = _snapshot(proc, roots, native)
        _require(_reader_inodes(roots, native) == native.inventory, _READER_ERROR)
        _require(after == before and _continuous_rows(allowance) == units
                 and _platform_rows(native) == platforms, _READER_ERROR)
        for pid, (raw, version) in receipts.items():
            observed, current = _native_bytes(_boot_path(policy, before[pid]), native, cap=65536, protected=True)
            _require(observed == raw and current == version, _READER_ERROR)
    _require(_source_records(_source_bundle(policy, native)) == sources and access._policy() == policy, _READER_ERROR)
    require_inactive_known_workers(allowance)
    allowance.tick()


def require_current_reader_closure(policy, allowance):
    allowance.tick()
    try:
        return _require_current_reader_closure(policy, allowance)
    except access.SceneRetirementAccessError:
        raise
    except Exception as error:
        raise access.SceneRetirementAccessError(_READER_ERROR) from error


class _BoundSourceLoader(importlib.abc.Loader):
    """Execute exactly the root-verified source bytes; never adopt stale pyc."""
    def __init__(self, name, row):
        self.name, self.row = name, row
        self.path = row['path']

    def create_module(self, spec):
        return None

    def get_filename(self, fullname):
        _require(fullname == self.name, _READER_ERROR)
        return self.row['path']

    def get_resource_reader(self, fullname):
        if Path(self.get_filename(fullname)).name != '__init__.py':
            return None
        from importlib.readers import FileReader
        return FileReader(self)

    def exec_module(self, module):
        module.__file__ = self.row['path']
        module.__loader__ = self
        exec(compile(self.row['raw'], self.row['path'], 'exec', dont_inherit=True), module.__dict__)


class _BoundSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, bundle, native=None):
        self.bundle, self.native = bundle, native

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.bundle:
            if self.native is None:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None:
                return None  # Actual built-in/frozen interpreter modules.
            if spec.loader is None:
                for candidate in spec.submodule_search_locations or ():
                    with _opened(candidate, directory=True, protected=True):
                        self.native.tick()
                return spec
            filename = spec.origin
            _require(type(filename) is str and Path(filename).is_absolute(), _READER_ERROR)
            # Validate the actual leaf and every ancestor, not just the nominal
            # dependencies directory. A service-writable nested module cannot
            # replace the real reader guard after the privilege drop.
            if type(spec.loader) is importlib.machinery.SourceFileLoader:
                raw, _ = _native_bytes(filename, self.native, cap=1024 * 1024, protected=True)
                spec.loader = _BoundSourceLoader(fullname, dict(path=filename, raw=raw))
            elif type(spec.loader) is importlib.machinery.ExtensionFileLoader:
                spec.loader = _ProtectedExtension(spec.loader, filename, self.native)
            else:
                _require(False, _READER_ERROR)
            return spec
        return importlib.util.spec_from_loader(fullname, _BoundSourceLoader(fullname, self.bundle[fullname]),
                                              origin=self.bundle[fullname]['path'])


class _ProtectedExtension(importlib.abc.Loader):
    def __init__(self, original, path, native):
        self.original, self.path, self.native = original, path, native

    def create_module(self, spec):
        with _opened(self.path, protected=True) as (fd, before):
            self.native.tick()
            value = self.original.create_module(spec)
            _require(_unit_version(os.fstat(fd)) == _unit_version(before), _READER_ERROR)
            self.native.tick()
            return value

    def exec_module(self, module):
        with _opened(self.path, protected=True) as (fd, before):
            self.native.tick()
            self.original.exec_module(module)
            _require(_unit_version(os.fstat(fd)) == _unit_version(before), _READER_ERROR)
            self.native.tick()


def _drop_service_credentials(native):
    _require(os.geteuid() == 0 and sys.platform == 'linux', _READER_ERROR)
    before = _proc_fields(os.getpid(), native)
    _require(before['uid'] == [0] * 4 and before['status']['NoNewPrivs'] == '1'
             and int(before['status']['CapEff'], 16) == 192
             and int(before['status']['CapBnd'], 16) == 192, _READER_ERROR)
    uid, gid = access._service_identity()
    _require(uid != 0 and gid != 0, _READER_ERROR)
    native.tick()
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)
    observed = _proc_fields(os.getpid(), native)
    _require(observed['uid'] == [uid] * 4 and observed['gid'] == [gid] * 4
             and observed['status']['NoNewPrivs'] == '1' and os.getgroups() == []
             and all(int(observed['status'][key], 16) == 0 for key in ('CapEff', 'CapPrm', 'CapInh', 'CapAmb')),
             _READER_ERROR)


def _runtime_import_paths():
    """Resolve installed package, dependencies and sibling first-party scripts."""
    return [str(_TRUSTED_SOURCE_ROOT), str(_TRUSTED_DEPENDENCIES),
            str(_TRUSTED_SOURCE_ROOT.parent)]


def _continuous_main(module, arguments):
    """Fixed root startup binds source/kernel evidence, then executes as blueprint.

    The installed isolated stdlib bootstrap is the only root entrypoint. Enabled
    startup reads root-owned source/dependencies and publishes a private native
    boot receipt. Disabled startup grants no receipt and resumes legacy service
    execution only after the exact credential drop.
    """
    from .task_evaluation_scene_retirement_preservation import ActionAllowance
    from .task_evaluation_scene_retirement_generations import _write
    _require(module in _CONTINUOUS and type(arguments) is list and len(arguments) <= 64
             and all(type(value) is str and len(value.encode()) <= 4096 for value in arguments), _READER_ERROR)
    allowance = ActionAllowance(expires_at=int(time.time()) + 60, elapsed_seconds=60,
                                local_bytes=0, archive_bytes=0, remote_bytes=0)
    native = _NativeObservation(allowance)
    policy = access._policy()
    if policy is None:
        # No service-writable interpreter/module is entered while root.
        _drop_service_credentials(native)
        repo = Path(os.environ.get('BLUEPRINT_PIPELINE_REPO', '/opt/blueprint/BlueprintCapturePipeline'))
        original = os.environ.get('BLUEPRINT_PIPELINE_PYTHON') or str(repo / '.venv/bin/python')
        _require(Path(original).is_absolute(), _READER_ERROR)
        os.chdir(repo)
        os.execve(original, [original, '-m', module, *arguments], os.environ | {'PYTHONDONTWRITEBYTECODE': '1'})
        raise AssertionError('execve returned')
    with _actual_proc(native):
        row = _proc_fields(os.getpid(), native)
        unit = _continuous_rows(allowance)[_CONTINUOUS[module]]
        _require(unit['ActiveState'] in {'activating', 'active'}
                 and unit['MainPID'] == str(os.getpid()) and row['cgroup'] == unit['ControlGroup']
                 and re.fullmatch(r'[0-9a-f]{32}', unit['InvocationID']), _READER_ERROR)
        installed_unit = _exact_continuous_unit(unit, module, native)
        _require(row['cmdline'] == ['/usr/bin/python3', *installed_unit['command']], _READER_ERROR)
        boot_raw, _ = _native_bytes('/proc/sys/kernel/random/boot_id', native, cap=128)
        boot_id = boot_raw.decode('ascii').strip()
        interpreter, version = _native_bytes(row['exe'], native, cap=64 * 1024 * 1024, protected=True)
        with _opened(_TRUSTED_DEPENDENCIES, directory=True, protected=True):
            pass
        with _opened(_TRUSTED_SOURCE_ROOT.parent / 'scripts', directory=True, protected=True):
            pass
        with scene_access(), _opened(policy['coordinator_path'], directory=True, protected=True) as (_, coordinator):
            bundle = _source_bundle(policy, native)
            _require_preloaded_core(bundle)
            _require(all(name not in sys.modules for name in bundle
                         if name not in _PRELOADED_CORE), _READER_ERROR)
            uid, gid = access._service_identity()
            receipt = dict(schema_version='scene_retirement_current_boot.v1', module=module,
                pid=row['pid'], start_ticks=row['start_ticks'], boot_id=boot_id, cgroup=row['cgroup'],
                invocation_id=unit['InvocationID'],
                interpreter=dict(path=row['exe'], identity=list(version),
                                 sha256='sha256:' + hashlib.sha256(interpreter).hexdigest()),
                sources=_source_records(bundle), policy_digest=policy['policy_digest'],
                coordinator_identity=list(access._identity(coordinator)), target_uid=uid, target_gid=gid,
                unit=installed_unit)
            receipt['boot_digest'] = canonical_digest(receipt, digest_field='boot_digest')
            path = _boot_path(policy, row)
            with _opened(path.parent, directory=True, protected=True) as (store, store_info):
                _require(stat.S_IMODE(store_info.st_mode) == 0o700, _READER_ERROR)
                native.tick()
                _write(store, path.name, receipt, parent_identity=access._identity(store_info))
                native.tick()
            # A live boot receipt can only become reader authority after the
            # root-compiled startup has dropped privileges, imported the entire
            # fixed cohort under this SH, and released it. A failed startup never
            # passes the action's EX/native live-process checks.
            _drop_service_credentials(native)
            stdlib = [value for value in sys.path if value and Path(value).is_absolute()
                      and value.startswith('/usr/lib/python')]
            _require(stdlib, _READER_ERROR)
            for path in stdlib:
                if Path(path).exists():
                    with _opened(path, directory=Path(path).is_dir(), protected=True):
                        pass
            sys.path[:] = [*_runtime_import_paths(), *stdlib]
            finder = _BoundSourceFinder(bundle, native)
            sys.meta_path.insert(0, finder)
            try:
                for name in sorted(bundle):
                    if name in _PRELOADED_CORE:
                        continue
                    native.tick()
                    loaded = importlib.import_module(name)
                    _require(type(loaded.__loader__) is _BoundSourceLoader
                             and loaded.__loader__.row is bundle[name], _READER_ERROR)
                loaded = sys.modules[module]
                _require(callable(getattr(loaded, 'main', None)), _READER_ERROR)
                native.tick()
            finally:
                sys.meta_path.remove(finder)
        # Continuous processes release the coarse startup admission here.
        # Every supported operation subsequently takes its actual per-reader SH.
        return loaded.main(arguments)


_WORKERS = frozenset('blueprint_pipeline.' + name for name in (
    'pubsub_handoff_listener',
    'task_evaluation_scene_progression',
    'task_evaluation_launch_preparation_worker',
    'task_evaluation_launch_activation_worker',
    'task_evaluation_episode_compilation_worker',
    'task_evaluation_sam31_preparation_execution',
    'task_evaluation_launch_dispatcher',
    'task_evaluation_launch_reconciler',
    'task_evaluation_launch_supervisor',
    'task_evaluation_policy_canary_dispatcher',
    'task_evaluation_terminal_resource_release',
    'operator_policy_canary_continuation',
    'task_evaluation_configured_controls_progression_worker',
))


@contextmanager
def _installed_policy_binding():
    """One installed public policy; an EnvironmentFile cannot redirect its fence."""
    selected = os.environ.get(_POLICY_ENV)
    fixed = str(_INSTALLED_POLICY)
    _require(selected in (None, '', fixed), 'scene_retirement_policy_binding_unproven')
    with ExitStack() as stack:
        try:
            _, info = stack.enter_context(_opened(fixed, protected=True))
        except FileNotFoundError:
            # Only a genuinely absent installation retains legacy startup. An
            # explicitly configured missing policy remains a refusal.
            _require(not selected, 'scene_retirement_policy_binding_unproven')
            yield
            return
        _require(stat.S_IMODE(info.st_mode) == 0o644,
                 'scene_retirement_policy_binding_unproven')
        os.environ[_POLICY_ENV] = fixed
        try:
            yield
        finally:
            if selected is None:
                os.environ.pop(_POLICY_ENV, None)
            else:
                os.environ[_POLICY_ENV] = selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selected_module = parser.add_mutually_exclusive_group(required=True)
    selected_module.add_argument('--worker', choices=sorted(_WORKERS))
    selected_module.add_argument('--continuous-module', choices=sorted(_CONTINUOUS))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    selected = parser.parse_args(argv)
    arguments = selected.arguments
    if arguments[:1] == ['--']:
        arguments = arguments[1:]
    if selected.continuous_module is not None:
        return _continuous_main(selected.continuous_module, arguments)
    original_argv = sys.argv
    # Acquire before target import, not after it has opened local artifacts or
    # loaded an unfenced worker. Existing SH reader/publisher locks nest safely.
    with _installed_policy_binding(), scene_access():
        try:
            sys.argv = [selected.worker, *arguments]
            runpy.run_module(selected.worker, run_name='__main__', alter_sys=True)
        finally:
            sys.argv = original_argv
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
