"""Fence a fixed installed worker before importing or running its original module.

This conservative launch boundary holds SH for the entire process entrypoint.
A daemon therefore keeps retirement while alive. It does not attest older loaded
code, clear unknown/manual/external readers, or substitute for reference checks.
Absent/disabled policy preserves the original module arguments and exit status.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import runpy
import select
import shlex
import stat
import sys
import subprocess
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .task_evaluation_scene_retirement_access import _opened, _require, scene_access


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


def _query_systemd(allowance):
    """Read only fixed unit state, with bounded output and an owned reaped child."""
    allowance.tick()
    command = ['/usr/bin/systemctl', 'show', '--all', '--no-pager',
               '--property=' + ','.join(sorted(_UNIT_PROPERTIES)), '--', *sorted(_KNOWN_UNITS)]
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
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


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
    parser.add_argument('--worker', required=True, choices=sorted(_WORKERS))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    selected = parser.parse_args(argv)
    arguments = selected.arguments
    if arguments[:1] == ['--']:
        arguments = arguments[1:]
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
