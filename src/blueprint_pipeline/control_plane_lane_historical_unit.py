"""Observe this actual fixed historical action unit, never caller-supplied rights.

Only the hardened root entrypoint uses this observer. Its result neither clears
readers nor replaces reopening the protected owner decision before mutation.
"""
from __future__ import annotations

import os
import re
import selectors
import stat
import subprocess
import sys
import time

_EXECUTABLE = '/opt/blueprint/operator-door/bin/blueprint-historical-generation-action'
_CAPS = frozenset({'cap_chown', 'cap_dac_override', 'cap_dac_read_search', 'cap_fowner', 'cap_sys_ptrace'})
_CAP_MASK = 0x8000f
_REQUIREMENTS = dict(Type='oneshot', User='root', Group='root', UMask='0077',
    NoNewPrivileges='yes', PrivateTmp='yes', PrivateDevices='yes', PrivateUsers='no',
    ProtectSystem='strict', ProtectHome='yes', ProtectHostname='yes', ProtectClock='yes',
    ProtectKernelTunables='yes', ProtectKernelModules='yes', ProtectKernelLogs='yes',
    ProtectControlGroups='yes', RestrictSUIDSGID='yes', LockPersonality='yes',
    ProtectProc='default', ProcSubset='all', AmbientCapabilities='', TasksMax='64',
    LimitNOFILE='512', MemoryMax=str(512 * 1024**2), Restart='no', WorkingDirectory='/')
_FIELDS = ('Id', 'MainPID', 'ControlGroup', 'ReadWritePaths', 'CapabilityBoundingSet',
    'ExecStart', 'TimeoutStartUSec', 'SystemCallFilter', *_REQUIREMENTS)


class HistoricalUnitError(ValueError):
    """Fixed refusal, without process output or protected paths."""


def _require(value, code):
    if not value:
        raise HistoricalUnitError('historical_generation_' + code)


def _validate_unit_observation(action_id, target, private_store, fields, status, cgroup, *, pid):
    """Private parser; production supplies freshly observed kernel/manager bytes."""
    code = 'unit_rights_unknown'
    unit = 'blueprint-historical-generation-' + action_id + '.service'
    _require(fields.get('Id') == unit and fields.get('MainPID') == str(pid)
        and fields.get('ControlGroup') == '/system.slice/' + unit
        and cgroup == '0::/system.slice/' + unit + '\n', code)
    _require(all(fields.get(key) == value for key, value in _REQUIREMENTS.items()), code)
    _require(fields.get('TimeoutStartUSec') == '4h', code)
    syscall_filter = fields.get('SystemCallFilter', '')
    _require(syscall_filter and not syscall_filter.startswith('~')
        and {'landlock_create_ruleset', 'landlock_add_rule', 'landlock_restrict_self'}
            .issubset(syscall_filter.split())
        and not {'ptrace', 'process_vm_readv', 'process_vm_writev'}.intersection(syscall_filter.split()), code)
    _require(set(str(fields.get('CapabilityBoundingSet', '')).lower().split()) == _CAPS
        and fields.get('ReadWritePaths', '').split() == [str(target), str(private_store)], code)
    command = fields.get('ExecStart', '')
    _require(isinstance(command, str) and command.count('{') == command.count('}') == 1
        and command.startswith('{ path=' + _EXECUTABLE + ' ; argv[]=' + _EXECUTABLE
                               + ' ' + action_id + ' ; ignore_errors=no ;'), code)
    _require(all(status.get(key, '').split() == ['0'] * 4 for key in ('Uid', 'Gid'))
        and status.get('NoNewPrivs') == '1' and status.get('Seccomp') == '2', code)
    for key in ('CapEff', 'CapBnd'):
        value = status.get(key)
        _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{16}', value)
                 and int(value, 16) == _CAP_MASK, code)


def _manager_fields(unit):
    """One fixed system-bus query, <=32KiB and five real elapsed seconds."""
    argv = ['/usr/bin/systemctl', '--system', '--no-pager', 'show', unit,
            '--property=' + ','.join(_FIELDS)]
    environment = dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C',
        DBUS_SYSTEM_BUS_ADDRESS='unix:path=/run/dbus/system_bus_socket', SYSTEMD_BUS_TIMEOUT='5s')
    started = time.monotonic()
    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env=environment, close_fds=True)
    data = bytearray()
    try:
        os.set_blocking(process.stdout.fileno(), False)
        with selectors.DefaultSelector() as selected:
            selected.register(process.stdout, selectors.EVENT_READ)
            ended = False
            while not ended:
                remaining = 5 - (time.monotonic() - started)
                _require(remaining > 0, 'unit_rights_unknown')
                for key, _ in selected.select(min(0.1, remaining)):
                    block = os.read(key.fd, min(4096, 32769 - len(data)))
                    _require(len(data) + len(block) <= 32768, 'unit_rights_unknown')
                    if not block:
                        ended = True
                        break
                    data.extend(block)
            _require(process.wait(timeout=max(0.001, 5 - (time.monotonic() - started))) == 0,
                     'unit_rights_unknown')
        values = {}
        for line in data.decode('ascii').splitlines():
            name, equal, value = line.partition('=')
            _require(equal and name in _FIELDS and name not in values, 'unit_rights_unknown')
            values[name] = value
        _require(set(values) == set(_FIELDS), 'unit_rights_unknown')
        return values
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()


def _proc_record(pid, name, cap):
    # PID and the two fixed names come from this process, never an input path.
    path = '/proc/' + str(pid) + '/' + name
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode), 'unit_rights_unknown')
        data = bytearray()
        while True:
            block = os.read(fd, min(4096, cap + 1 - len(data)))
            _require(len(data) + len(block) <= cap, 'unit_rights_unknown')
            if not block:
                return data.decode('ascii')
            data.extend(block)
    finally:
        os.close(fd)


def prove_historical_unit(action_id, target, private_store):
    """Current main PID, real cgroup, real root/cap/seccomp status and manager mounts."""
    _require(sys.platform == 'linux' and os.geteuid() == 0, 'native_unavailable')
    _require(isinstance(action_id, str) and re.fullmatch('[0-9a-f]{32}', action_id), 'unit_rights_unknown')
    pid = os.getpid()
    unit = 'blueprint-historical-generation-' + action_id + '.service'
    try:
        fields = _manager_fields(unit)
        status = {}
        for line in _proc_record(pid, 'status', 32768).splitlines():
            name, colon, value = line.partition(':')
            _require(colon and name not in status, 'unit_rights_unknown')
            status[name] = value.strip()
        cgroup = _proc_record(pid, 'cgroup', 4096)
        _validate_unit_observation(action_id, target, private_store, fields, status, cgroup, pid=pid)
        return dict(unit=unit, pid=pid, actual_kernel_and_manager_observed=True,
                    references_clear=False, execution_authorized=False)
    except (OSError, UnicodeError, subprocess.SubprocessError):
        raise HistoricalUnitError('historical_generation_unit_rights_unknown') from None
