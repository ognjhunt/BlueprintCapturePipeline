# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_retirement_supervisor.py
#   scripts/scene_retirement_continuous_bootstrap.py
#   scripts/install_scene_retirement_runtime.py
"""Actual disposable Linux privilege boundary; a Mac skip is not proof.

This first case proves only default-off startup and the real UID/capability
drop. It issues no retirement grant and does not clear current readers.
"""
from __future__ import annotations

import grp
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import sys
import tempfile
import uuid

import pytest


_RUNTIME = Path('/mnt/blueprint-work/scene-retirement-runtime')
_BOOT = Path('/usr/lib/blueprint/scene-retirement-runtime')
_POLICY = Path('/etc/blueprint/scene-retirement-policy.json')


def _native(command, *, timeout=30):
    value = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
        text=True, timeout=timeout, env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'})
    assert len(value.stdout.encode()) <= 65536 and len(value.stderr.encode()) <= 4096
    assert value.returncode == 0, value.stderr
    return value.stdout


def _copy_protected(source, target):
    shutil.copytree(source, target, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    for path in [target, *target.rglob('*')]:
        assert not path.is_symlink(), 'a source alias is not a protected runtime input'
        os.chown(path, 0, 0)
        path.chmod(0o755 if path.is_dir() or path.stat().st_mode & 0o111 else 0o644)


def _default_off_native_phase():
    assert sys.platform == 'linux' and os.getuid() == os.geteuid() == 0
    assert os.environ.get('BLUEPRINT_DISPOSABLE_LINUX_TEST') == '1'
    assert Path('/run/systemd/system').is_dir()
    # Preserve every pre-existing installation. This case requires a clean
    # disposable VM and must not weaken the installed policy to gain admission.
    assert not _POLICY.exists() and not _POLICY.is_symlink()
    assert not _RUNTIME.exists() and not _RUNTIME.is_symlink()
    assert not _BOOT.exists() and not _BOOT.is_symlink()
    created_account = False
    try:
        account = pwd.getpwnam('blueprint')
    except KeyError:
        _native(['/usr/sbin/useradd', '--system', '--user-group', '--no-create-home',
                 '--shell', '/usr/sbin/nologin', 'blueprint'])
        created_account = True
        account = pwd.getpwnam('blueprint')
    assert account.pw_uid != 0 and account.pw_gid == grp.getgrnam('blueprint').gr_gid
    root = Path(tempfile.mkdtemp(prefix='blueprint-scene-native-', dir='/var/lib'))
    root.chmod(0o755)
    identity = (root.stat().st_dev, root.stat().st_ino)
    unit = 'blueprint-scene-native-default-' + uuid.uuid4().hex + '.service'
    unit_path = Path('/etc/systemd/system') / unit
    installed = {}
    try:
        source = Path(__file__).resolve().parents[1]
        sealed = root / 'source'
        sealed.mkdir(mode=0o755)
        for folder in ('src/blueprint_pipeline', 'scripts', 'deploy/systemd'):
            _copy_protected(source / folder, sealed / folder)
        dependencies = root / 'dependencies'
        dependencies.mkdir(mode=0o755)
        # Default-off startup never imports any dependency or daemon as root.
        # An empty real protected dependency namespace is sufficient here.
        receipt = json.loads(_native([
            '/usr/bin/python3', '-I', '-S', str(sealed / 'scripts/install_scene_retirement_runtime.py'),
            '--source', str(sealed), '--dependencies', str(dependencies),
        ], timeout=300))
        assert receipt['status'] == 'prepared'
        assert receipt['authority_issued'] is False and receipt['cleanup_enabled'] is False
        for path in (_RUNTIME, _BOOT):
            info = path.stat()
            installed[path] = (info.st_dev, info.st_ino)
            assert info.st_uid == 0 and not info.st_mode & 0o022
        assert not _POLICY.exists()
        probe = root / 'probe.py'
        probe.write_text('''import json, os, sys
status = dict(line.split(':', 1) for line in open('/proc/self/status') if ':' in line)
print(json.dumps({'uid': os.getuid(), 'euid': os.geteuid(), 'gid': os.getgid(),
 'groups': os.getgroups(), 'argv': sys.argv[1:],
 'kernel_uid': status['Uid'].split(), 'kernel_gid': status['Gid'].split(),
 'caps': {k: status[k].strip() for k in ('CapEff','CapPrm','CapInh','CapAmb')},
 'no_new_privs': status['NoNewPrivs'].strip()}, sort_keys=True), flush=True)
''')
        probe.chmod(0o644)
        executable = root / 'legacy-python'
        executable.write_text('#!/bin/sh\nexec /usr/bin/python3 -I -S ' + str(probe) + ' "$@"\n')
        executable.chmod(0o755)
        unit_path.write_text('[Unit]\nDescription=Disposable scene retirement default boundary\n'
            '[Service]\nType=oneshot\nUser=blueprint\nGroup=blueprint\n'
            'NoNewPrivileges=yes\nCapabilityBoundingSet=CAP_SETUID CAP_SETGID\n'
            'AmbientCapabilities=\nProtectHome=yes\nProtectSystem=strict\n'
            'Environment=BLUEPRINT_PIPELINE_REPO=' + str(root) + '\n'
            'Environment=BLUEPRINT_PIPELINE_PYTHON=' + str(executable) + '\n'
            'ExecStart=!/usr/bin/python3 -I -S ' + str(_BOOT / 'continuous_bootstrap.py')
            + ' --continuous-module blueprint_pipeline.live_pipeline_intake_service -- --native-probe\n'
            'StandardOutput=journal\nStandardError=journal\nTimeoutStartSec=60\n')
        unit_path.chmod(0o644)
        _native(['/usr/bin/systemctl', 'daemon-reload'])
        _native(['/usr/bin/systemctl', 'start', unit], timeout=90)
        facts = dict(line.split('=', 1) for line in _native([
            '/usr/bin/systemctl', 'show', unit, '--no-pager', '--all',
            '--property=Id,LoadState,ActiveState,SubState,Result,ExecMainStatus',
        ]).splitlines())
        assert facts == dict(Id=unit, LoadState='loaded', ActiveState='inactive',
                             SubState='dead', Result='success', ExecMainStatus='0')
        journal = _native(['/usr/bin/journalctl', '--unit=' + unit, '--no-pager',
                           '--lines=40', '--output=cat', '--quiet'])
        rows = [json.loads(line) for line in journal.splitlines() if line.startswith('{')]
        assert len(rows) == 1, journal
        observed = rows[0]
        assert observed['uid'] == observed['euid'] == account.pw_uid
        assert observed['gid'] == account.pw_gid and observed['groups'] == []
        assert observed['kernel_uid'] == [str(account.pw_uid)] * 4
        assert observed['kernel_gid'] == [str(account.pw_gid)] * 4
        assert all(int(value, 16) == 0 for value in observed['caps'].values())
        assert observed['no_new_privs'] == '1'
        assert observed['argv'] == ['-m', 'blueprint_pipeline.live_pipeline_intake_service', '--native-probe']
        assert not _POLICY.exists()
        print(json.dumps(dict(status='passed', actual_uid=account.pw_uid,
             actual_systemd=True, default_off=True, authority_issued=False,
             kernel_caps_zero=True, args_preserved=True), sort_keys=True), flush=True)
    finally:
        if unit_path.exists():
            _native(['/usr/bin/systemctl', 'stop', unit])
            unit_path.unlink()
            _native(['/usr/bin/systemctl', 'daemon-reload'])
        for path, original in installed.items():
            assert (path.stat().st_dev, path.stat().st_ino) == original
            shutil.rmtree(path)
        assert (root.stat().st_dev, root.stat().st_ino) == identity
        shutil.rmtree(root)
        if created_account:
            _native(['/usr/sbin/userdel', 'blueprint'])


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != 'linux' or os.environ.get('BLUEPRINT_DISPOSABLE_LINUX_TEST') != '1',
                   reason='mandatory actual disposable Linux/systemd UID proof; Mac skip is unmet')
def test_actual_default_off_root_bootstrap_drops_blueprint_uid_and_all_caps():
    command = [sys.executable, str(Path(__file__).resolve()), '--native-default-off']
    if os.geteuid() != 0:
        command = ['sudo', '-n', 'env', 'BLUEPRINT_DISPOSABLE_LINUX_TEST=1',
                   'PYTHONDONTWRITEBYTECODE=1', *command]
    value = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           timeout=480, cwd=Path(__file__).parents[1])
    assert len(value.stdout.encode()) <= 65536 and len(value.stderr.encode()) <= 65536
    assert value.returncode == 0, value.stdout + value.stderr
    receipt = json.loads(value.stdout.strip().splitlines()[-1])
    assert receipt['status'] == 'passed' and receipt['actual_uid'] != 0
    assert receipt['actual_systemd'] and receipt['kernel_caps_zero'] and receipt['args_preserved']
    assert receipt['default_off'] and receipt['authority_issued'] is False


if __name__ == '__main__':
    assert sys.argv[1:] == ['--native-default-off']
    _default_off_native_phase()
