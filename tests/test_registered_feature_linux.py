"""ADP-009D real root/blueprint UID acceptance on a disposable hosted Linux runner.

Requires explicit BLUEPRINT_DISPOSABLE_LINUX_TEST=1. A Mac skip is NOT proof.
No production host, remote provider, cloud SDK, or actual network is contacted.
Root-owned installation lives under /var/lib, with actual blueprint group modes.
"""
from __future__ import annotations

import grp
import hashlib
import io
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()+b'\n'


def install_protected_feature(root):
    """Exact real protected installation reusable by the experiment roundtrip."""
    from blueprint_pipeline import control_plane_lane_owner_consents as owners
    from tests.test_registered_checkpoint_cache import tiny_inventory
    source = Path(__file__).parents[1]
    package = root / 'installed/operator_door'
    package.mkdir(parents=True, mode=0o700)
    for name in ('__init__.py', 'config.py'):
        target = package / name
        target.write_bytes((source / 'deploy/operator-door/operator_door' / name).read_bytes())
        target.chmod(0o600)
    owners.INSTALLED_PACKAGE_ROOT = package.parent
    state = root / 'state'
    state.mkdir(mode=0o755)
    private = state / 'requests/needed-checkpoint-cache-records'
    private.mkdir(parents=True, mode=0o700)
    private.parent.chmod(0o700)
    public = state / 'needed-checkpoint-cache-registration'
    public.mkdir(mode=0o755)
    authority = public / 'authority'
    authority.mkdir(mode=0o750)
    gid = grp.getgrnam('blueprint').gr_gid
    os.chown(authority, 0, gid)
    for parent, name, mode, selected_gid in ((private, '.cache-store.lock', 0o600, 0),
            (authority, '.authority.lock', 0o640, gid)):
        path = parent/name
        path.write_bytes(b'')
        os.chown(path, 0, selected_gid)
        path.chmod(mode)
    work = root / 'work/lanes'
    work.mkdir(parents=True, mode=0o750)
    work.parent.chmod(0o755)
    os.chown(work, 0, gid)
    lock = work / '.lane-scratch.lock'
    lock.write_bytes(b'')
    lock.chmod(0o600)
    lane = work / 'g1-checkpoint'
    lane.mkdir(mode=0o750)
    os.chown(lane, 0, gid)
    policy = root / 'policy.json'
    policy.write_bytes(encoded(dict(schema_version=owners.POLICY_SCHEMA, enabled=True,
        principals=[dict(principal='operator', owners=['owner'], allowed_actions=['register'],
                         max_consent_seconds=3600)])))
    policy.chmod(0o600)
    inventory, payloads = tiny_inventory()
    inventory_path = root / 'installed-inventory.json'
    inventory_path.write_bytes(encoded(inventory))
    inventory_path.chmod(0o644)
    config = root / 'door.json'
    settings = dict(state_root=str(state), needed_checkpoint_cache_creation_enabled=True,
        needed_checkpoint_cache_inventory_file=str(inventory_path), lane_owner_policy_file=str(policy),
        lane_scratch_work_root=str(work), lane_scratch_inputs_root=str(root / 'inputs/lanes'))
    config.write_bytes(encoded(settings))
    config.chmod(0o600)
    return dict(config=config, settings=settings, private=private, public=public, authority=authority,
                inventory_path=inventory_path, payloads=payloads, policy=policy, work=work)


def _linux_cache_roundtrip():
    """Executed as actual root, with an actual different-UID forked reader."""
    import fcntl
    import socket
    from urllib.parse import unquote
    from blueprint_pipeline import control_plane_registered_checkpoint_cache as cache
    from blueprint_pipeline import native_g1_checkpoint_cache as native
    assert sys.platform == 'linux' and os.geteuid() == 0
    created_account, child = False, None
    try:
        account = pwd.getpwnam('blueprint')
    except KeyError:
        subprocess.run(['useradd', '--system', '--user-group', '--no-create-home',
                        '--shell', '/usr/sbin/nologin', 'blueprint'], check=True)
        created_account = True
        account = pwd.getpwnam('blueprint')
    root = Path(tempfile.mkdtemp(prefix='blueprint-adp-disk-test-', dir='/var/lib'))
    root.chmod(0o755)
    parent_sock = child_sock = None
    try:
        assert account.pw_uid != 0 and account.pw_gid == grp.getgrnam('blueprint').gr_gid
        value = install_protected_feature(root)
        inventory_raw = value['inventory_path'].read_bytes()
        grant = cache.issue_needed_checkpoint_cache_intent(principal='operator', owner='owner',
            name='needed-models', reference_kind='run_ref', reference_value='run1', lease_ttl_seconds=1800,
            size_budget_bytes=8*1024*1024, inventory_raw_sha256='sha256:'+hashlib.sha256(inventory_raw).hexdigest(),
            inventory_raw_size_bytes=len(inventory_raw), installed_config_path=value['config'], now=lambda: 1000)
        fetcher = native._fetcher()
        class Response(io.BytesIO):
            def __init__(self, url):
                super().__init__(value['payloads'][unquote(url.removeprefix(fetcher.MODEL_BASE))])
                self.url = url
            def geturl(self):
                return self.url
        fetcher._open_https = lambda url, **kw: Response(url)
        native._fetcher = lambda: fetcher
        filled = cache.fill_needed_checkpoint_cache(grant['intent_id'],
            expected_sha256=grant['intent']['sha256'], expected_size_bytes=grant['intent']['size_bytes'],
            installed_config_path=value['config'], now=lambda: 1100)
        target = Path(filled['path'])
        # Test-only fixed installation repin. Production code has no path/env/body grant fallback.
        cache._PUBLIC_REGISTRATION = value['public']
        cache._PUBLIC_INVENTORY = value['inventory_path']
        cache._REGISTERED_ROOTS = (value['work'],)
        parent_sock, child_sock = socket.socketpair()
        parent_sock.settimeout(60)
        child_sock.settimeout(60)
        child = os.fork()
        if child == 0:
            parent_sock.close()
            try:
                os.setgroups([])
                os.setgid(account.pw_gid)
                os.setuid(account.pw_uid)
                assert os.geteuid() == account.pw_uid
                denied = 0
                for path in (value['config'], value['policy'], value['private'] / (grant['intent_id']+'.json')):
                    try:
                        fd = os.open(path, os.O_RDONLY)
                    except PermissionError:
                        denied += 1
                    else:
                        os.close(fd)
                        raise AssertionError('private authority readable by blueprint')
                assert denied == 3
                # Default API must never open its private door config at this UID.
                with cache.NeededCheckpointCacheUse.open_registered(target, now=lambda: 1200) as use:
                    rows = native.verify_local_g1_checkpoint_cache(target, _cache_use=use)
                    assert len(rows) == 24
                    child_sock.sendall(b'held')
                    assert child_sock.recv(32) == b'revoked'
                    try:
                        use.hash_file(target / next(iter(value['payloads'])), role='wam_hash')
                    except cache.NeededCheckpointCacheError:
                        assert use.failure is not None
                    else:
                        raise AssertionError('revoked reader performed payload work')
                child_sock.sendall(b'closed')
                os._exit(0)
            except BaseException as exc:
                child_sock.sendall(('failure:'+type(exc).__name__+':'+str(exc)).encode()[:512])
                os._exit(1)
        child_sock.close()
        assert parent_sock.recv(512) == b'held'
        independent = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(independent)
        cache.update_needed_checkpoint_cache_authority(operation='revoke', intent_id=grant['intent_id'],
            installed_config_path=value['config'], now=lambda: 1201)
        parent_sock.sendall(b'revoked')
        assert parent_sock.recv(512) == b'closed'
        _, status = os.waitpid(child, 0)
        child = None
        assert os.waitstatus_to_exitcode(status) == 0
        assert all((target/path).read_bytes() == data for path, data in value['payloads'].items())
        return dict(status='passed', actual_uid=account.pw_uid, actual_gid=account.pw_gid,
                    private_denials=3, verified_files=24, retained_target_lock=True,
                    root_revoke_observed=True, payload_retained=True)
    finally:
        if child:
            os.kill(child, 9)
            os.waitpid(child, 0)
        if parent_sock:
            parent_sock.close()
        if child_sock:
            child_sock.close()
        assert root.parent == Path('/var/lib') and root.name.startswith('blueprint-adp-disk-test-')
        shutil.rmtree(root)
        if created_account:
            subprocess.run(['userdel', 'blueprint'], check=True)
            subprocess.run(['groupdel', 'blueprint'], check=False)


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != 'linux' or os.environ.get('BLUEPRINT_DISPOSABLE_LINUX_TEST') != '1',
                    reason='requires explicitly authorized disposable Linux root/UID fixture; Mac skip is unmet')
def test_actual_root_blueprint_uid_cache_lifetime_and_revoke(tmp_path):
    """A required successful Linux result, not a skipped-test acceptance claim."""
    command = [sys.executable, str(Path(__file__).resolve()), '--cache-root-fixture']
    if os.geteuid() != 0:
        command = ['sudo', '-n', 'env', 'BLUEPRINT_DISPOSABLE_LINUX_TEST=1',
                   'PYTHONDONTWRITEBYTECODE=1', *command]
    result = subprocess.run(command, cwd=Path(__file__).parents[1], capture_output=True, text=True,
                            timeout=180, env=os.environ | {'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode == 0, result.stdout+result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt['status'] == 'passed' and receipt['actual_uid'] != 0 and receipt['verified_files'] == 24


if __name__ == '__main__':
    assert sys.argv[1:] == ['--cache-root-fixture']
    assert os.environ.get('BLUEPRINT_DISPOSABLE_LINUX_TEST') == '1', 'disposable fixture opt-in required'
    sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
    sys.path.insert(0, str(Path(__file__).parents[1]))
    print(json.dumps(_linux_cache_roundtrip(), sort_keys=True))
