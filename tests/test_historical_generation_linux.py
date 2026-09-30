"""ADP-009D/day28: actual target-only historical action sandbox rights.

Disposable Linux only. This proves permissions and retained foreign references,
not a historical owner decision, a deletion receipt or a cleared reference set.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_dispatch.py
import json
import os
import pwd
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _root_fixture():
    assert sys.platform == 'linux' and os.geteuid() == 0
    assert os.environ.get('BLUEPRINT_DISPOSABLE_LINUX_TEST') == '1'
    assert Path('/proc/1/exe').resolve() == Path('/usr/lib/systemd/systemd')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from blueprint_pipeline.control_plane_lane_historical_dispatch import _unit_property_assignments
    foreign = pwd.getpwnam('nobody')
    root = Path(tempfile.mkdtemp(prefix='blueprint-historical-unit-', dir='/var/lib'))
    child = None
    try:
        root.chmod(0o755)
        parent = root / 'work'
        parent.mkdir(mode=0o755)
        target, adjacent, private = parent / 'selected', parent / 'adjacent', root / 'private'
        target.mkdir(mode=0o700)
        os.chown(target, foreign.pw_uid, foreign.pw_gid)
        adjacent.mkdir()
        original, neighbor = target / 'original.bin', adjacent / 'keep.bin'
        original.write_bytes(b'original foreign writer bytes')
        original.chmod(0o600)
        os.chown(original, foreign.pw_uid, foreign.pw_gid)
        neighbor.write_bytes(b'adjacent must remain unchanged')
        private.mkdir(mode=0o700)
        secret = private / 'protected.json'
        secret.write_bytes(b'private authority canary')
        secret.chmod(0o600)
        holder = (
            'import json,mmap,os,sys\n'
            'os.chdir(sys.argv[1])\n'
            'fd=os.open(sys.argv[2],os.O_RDWR)\n'
            'mapping=mmap.mmap(fd,0)\n'
            'print(json.dumps({"fd":fd}),flush=True)\n'
            'assert sys.stdin.readline().strip()=="check"\n'
            'denied=0\n'
            'for path in (sys.argv[2],sys.argv[1]+"/new-writer",sys.argv[3]):\n'
            ' try: os.close(os.open(path,os.O_WRONLY|os.O_CREAT,0o600))\n'
            ' except PermissionError: denied+=1\n'
            'assert denied==3\n'
            'assert os.read(fd,4096)==b"original foreign writer bytes"\n'
            'print(json.dumps({"future_writes_denied":denied,"old_fd_retained":True}),flush=True)\n'
        )
        child = subprocess.Popen(['/usr/bin/python3', '-c', holder, str(target), str(original), str(secret)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            preexec_fn=lambda: (os.setgroups([]), os.setgid(foreign.pw_gid), os.setuid(foreign.pw_uid)))
        fd = json.loads(child.stdout.readline())['fd']
        action_id = secrets.token_hex(16)
        entry = root / 'action-entry'
        guard_source = (Path(__file__).parents[1] / 'src/blueprint_pipeline/control_plane_lane_historical_unit.py').read_text()
        # This disposable installed namespace is sealed before startup. Only
        # the compiled fixed executable pathname changes; no kernel/manager
        # observation, expected property or authority condition is mocked.
        guard_source = guard_source.replace(
            "'/opt/blueprint/operator-door/bin/blueprint-historical-generation-action'", repr(str(entry)))
        guard = root / 'unit_guard.py'
        guard.write_text(guard_source)
        guard.chmod(0o644)
        probe = root / 'probe.py'
        probe.write_text(
            'import importlib.util,json,os,sys\nfrom pathlib import Path\n'
            'action_id=sys.argv[1]\n'
            'target,original,neighbor,private,pid,foreign_fd=' + repr(tuple(map(str,
                (target, original, neighbor, private, child.pid, fd)))) + '\n'
            'spec=importlib.util.spec_from_file_location("installed_historical_unit",'
                + repr(str(guard)) + ')\n'
            'unit_guard=importlib.util.module_from_spec(spec)\n'
            'spec.loader.exec_module(unit_guard)\n'
            'rights=unit_guard.prove_historical_unit(action_id,target,private)\n'
            'assert rights["actual_kernel_and_manager_observed"] is True\n'
            'assert rights["execution_authorized"] is False\n'
            'assert os.geteuid()==0\n'
            'assert "NoNewPrivs:\\t1" in Path("/proc/self/status").read_text()\n'
            'assert Path("/proc/"+pid+"/cwd").resolve()==Path(target)\n'
            'assert os.readlink("/proc/"+pid+"/fd/"+foreign_fd)==original\n'
            'assert original in Path("/proc/"+pid+"/maps").read_text()\n'
            'Path("/proc/"+pid+"/cmdline").read_bytes()\n'
            'Path("/proc/"+pid+"/environ").read_bytes()\n'
            'assert not any("acl" in name for name in os.listxattr(target))\n'
            'assert not any("acl" in name for name in os.listxattr(original))\n'
            'for path in (target,original):\n'
            ' os.chown(path,0,0)\n'
            ' os.chmod(path,0o700 if path==target else 0o600)\n'
            'created=Path(target)/"exact-write-probe"\n'
            'created.write_bytes(b"only selected target writable")\n'
            'created.unlink()\n'
            'denied=0\n'
            'for path in (neighbor,str(Path(target).parent/"forbidden-parent-entry")):\n'
            ' try: Path(path).write_bytes(b"must be refused")\n'
            ' except OSError: denied+=1\n'
            'assert denied==2\n'
            'assert os.readlink("/proc/"+pid+"/fd/"+foreign_fd)==original\n'
            'assert original in Path("/proc/"+pid+"/maps").read_text()\n'
            'Path(private,"receipt.json").write_text(json.dumps({"target_writable":True,'
            '"adjacent_and_parent_denied":2,"foreign_fd_cwd_mapping_visible":True,'
            '"references_clear":False,"actual_unit_guard_passed":True}))\n'
        )
        probe.chmod(0o644)
        entry.write_text('#!/bin/sh\nset -eu\ntest "$#" = 1\nexec /usr/bin/python3 '
                         + str(probe) + ' "$1"\n')
        entry.chmod(0o755)
        assignments = _unit_property_assignments(target, private)
        unit = 'blueprint-historical-generation-' + action_id
        argv = ['/usr/bin/systemd-run', '--unit=' + unit, '--wait', '--collect',
                *('--property=' + value for value in assignments), '--', str(entry), action_id]
        done = subprocess.run(argv, capture_output=True, text=True, timeout=45)
        assert done.returncode == 0, done.stdout + done.stderr
        receipt = json.loads((private / 'receipt.json').read_bytes())
        checked, errors = child.communicate('check\n', timeout=5)
        assert child.returncode == 0, errors
        receipt.update(json.loads(checked))
        assert neighbor.read_bytes() == b'adjacent must remain unchanged'
        assert original.read_bytes() == b'original foreign writer bytes'
        assert not (parent / 'forbidden-parent-entry').exists()
        assert original.stat().st_uid == target.stat().st_uid == 0
        return receipt
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate(timeout=5)
        shutil.rmtree(root)


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != 'linux' or os.environ.get('BLUEPRINT_DISPOSABLE_LINUX_TEST') != '1',
                   reason='actual disposable Linux action sandbox required; Mac skip is unmet')
def test_actual_historical_target_only_write_and_foreign_reference_visibility():
    command = [sys.executable, str(Path(__file__).resolve()), '--root-fixture']
    if os.geteuid() != 0:
        command = ['sudo', '-n', 'env', 'BLUEPRINT_DISPOSABLE_LINUX_TEST=1',
                   'PYTHONDONTWRITEBYTECODE=1', *command]
    done = subprocess.run(command, capture_output=True, text=True, timeout=70,
                          cwd=Path(__file__).parents[1],
                          env=os.environ | {'PYTHONDONTWRITEBYTECODE': '1'})
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == dict(target_writable=True,
        adjacent_and_parent_denied=2, foreign_fd_cwd_mapping_visible=True,
        references_clear=False, future_writes_denied=3, old_fd_retained=True,
        actual_unit_guard_passed=True)


if __name__ == '__main__' and sys.argv[1:] == ['--root-fixture']:
    print(json.dumps(_root_fixture(), sort_keys=True))
