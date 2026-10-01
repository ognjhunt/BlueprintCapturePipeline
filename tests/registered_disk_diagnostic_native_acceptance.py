"""Installed Linux ordinary diagnostic lifecycle; no reference or clock doubles.

Only the tiny object service is in-process. Actual root/ordinary UIDs, producer
syscalls, lease expiry, protected shipped GC and restore code execute unchanged.
"""
from __future__ import annotations

import base64
import functools
import grp
import hashlib
import json
import os
import pwd
import select
import time
from pathlib import Path

from tests.test_registered_feature_linux import encoded, install_protected_feature, _run_shipped_gc_sandbox


def _ordinary_denied(target, value, account):
    read, write = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read)
        try:
            os.initgroups('blueprint', account.pw_gid)
            os.setgid(account.pw_gid)
            os.setuid(account.pw_uid)
            for path in (target, value['config'], value['policy'], target / 'disk-capacity-report.v1.json'):
                try:
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                except PermissionError:
                    continue
                os.close(fd)
                raise AssertionError('ordinary UID accessed root-only diagnostic')
            os.write(write, b'root-only')
        finally:
            os.close(write)
            os._exit(0)
    os.close(write)
    try:
        assert select.select([read], [], [], 10)[0]
        assert os.read(read, 128) == b'root-only'
    finally:
        os.close(read)
        os.waitpid(child, 0)


def _held_fd_keeps(value, action, pins, report):
    ready_read, ready_write = os.pipe()
    finish_read, finish_write = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(ready_read)
        os.close(finish_write)
        fd = os.open(report, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.write(ready_write, b'held')
            assert os.read(finish_read, 1) == b'x'
        finally:
            os.close(fd)
            os.close(ready_write)
            os.close(finish_read)
            os._exit(0)
    os.close(ready_write)
    os.close(finish_read)
    try:
        assert select.select([ready_read], [], [], 10)[0]
        assert os.read(ready_read, 4) == b'held'
        original = report.read_bytes()
        result = _run_shipped_gc_sandbox(value, action, time.time(), pins, realtime=True)
        row = next(row for row in result['report']['registered_experiments']['outcomes']
                   if row['action_id'] == action['action_id'])
        assert row['decision'] == 'kept' and row['removed_logical_bytes'] == row['removed_allocated_bytes'] == 0, row
        assert report.read_bytes() == original
    finally:
        os.write(finish_write, b'x')
        os.close(finish_write)
        os.close(ready_read)
        os.waitpid(child, 0)


def run(root):
    from blueprint_pipeline import control_plane_lane_owner_consents as owners
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    from blueprint_pipeline import control_plane_lane_experiment_birth as birth
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as diagnostic
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer
    from blueprint_pipeline import control_plane_lane_experiment_archive as archive
    from blueprint_pipeline import control_plane_lane_experiment_restore as restoration
    from blueprint_pipeline import control_plane_disk_budget as disk
    from blueprint_pipeline.control_plane_storage_gc import run_storage_gc, RUN_ACK
    from tests.test_registered_experiment_offload import Cloud

    assert os.geteuid() == 0 and root.parent == Path('/var/lib') and root.name.startswith('blueprint-adp-contained-')
    value = install_protected_feature(root)
    owners.INSTALLED_PACKAGE_ROOT = root / 'installed'
    account = pwd.getpwnam('blueprint')
    gid = grp.getgrnam('blueprint').gr_gid
    policy = json.loads(value['policy'].read_bytes())
    policy['principals'][0]['allowed_actions'] = ['register', 'delete', 'offload', 'keep']
    value['policy'].write_bytes(encoded(policy))
    pins = root / 'pins'
    pins.mkdir(mode=0o700)
    gc_env = root / 'gc.env'
    gc_env.write_text('BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT=' + str(pins) + '\n')
    gc_env.chmod(0o600)
    roots = (value['work'], root / 'inputs/lanes')
    for lane_root in roots:
        if not lane_root.exists():
            lane_root.mkdir(parents=True, mode=0o750)
            lane_root.parent.chmod(0o755)
            (lane_root / '.lane-scratch.lock').write_bytes(b'')
            (lane_root / '.lane-scratch.lock').chmod(0o600)
        os.chown(lane_root, 0, gid)
        (lane_root / 'diagnostics').mkdir(mode=0o750)
        os.chown(lane_root / 'diagnostics', 0, gid)
    value['config'].write_bytes(encoded(value['settings'] | {
        'experiment_creation_enabled': True, 'experiment_retirement_enabled': True,
        'experiment_gc_environment_file': str(gc_env)}))
    receipts = []
    for selected_root, profile, method in (('work', 'root_disk_diagnostic_disposable.v1', 'delete'),
                                           ('inputs', 'root_disk_diagnostic_evidence.v1', 'offload')):
        request = root / ('request-' + method + '.json')
        request.write_bytes(encoded(diagnostic.build_request(installed_config_path=value['config'], run_ref='native-' + method)))
        request.chmod(0o600)
        selector = dict(sha256='sha256:' + hashlib.sha256(request.read_bytes()).hexdigest(), size_bytes=request.stat().st_size)
        grant = issuer.issue_experiment_creation_intent(installed_config_path=value['config'],
            principal='operator', owner='owner', root=selected_root, reference_value='native-' + method,
            lease_ttl_seconds=8, participant_profile=profile, request_records=((request, selector),), now=time.time)
        born = birth.create_registered_experiment(grant['intent_id'], expected_intent=grant['intent'],
            installed_config_path=value['config'], now=time.time)
        target = Path(born['path'])
        _ordinary_denied(target, value, account)
        completed = diagnostic.run_registered_disk_diagnostic(grant['intent_id'], expected_intent=grant['intent'],
            request_path=request, installed_config_path=value['config'], now=time.time)
        report = target / diagnostic.REPORT_NAME
        before = report.read_bytes()
        original_inode = target.stat().st_ino
        assert completed['completion'] and all(row['statvfs']['f_blocks'] > 0 for row in completed['report']['observations'])
        try:
            consumer.RegisteredExperimentUse.admit(target, now=time.time)
        except ValueError:
            pass
        else:
            raise AssertionError('sealed diagnostic acquired writer/reader authority')
        _ordinary_denied(target, value, account)
        lease = json.loads((target / '.lane-scratch.v1.json').read_bytes())
        # Wait for the actual original lease; no replacement caller clock.
        time.sleep(max(0, lease['expires_at_epoch'] - time.time()) + .02)
        assert time.time() >= lease['expires_at_epoch']
        action = issuer.issue_experiment_action_intent(grant['intent_id'], principal='operator', owner='owner',
            action=method, expires_at_epoch=time.time() + 120, installed_config_path=value['config'], now=time.time)
        disabled = run_storage_gc(content_store_roots=(), derived_roots=(), queue_roots=(), pins_root=pins,
            apply=True, ack=RUN_ACK, lane_scratch_roots=roots, lane_scratch_enabled=False,
            _experiment_config_path=value['config'], now=time.time)
        assert disabled['registered_experiments']['enabled'] is False and report.read_bytes() == before
        _held_fd_keeps(value, action, pins, report)
        result = _run_shipped_gc_sandbox(value, action, time.time(), pins, realtime=True, invocation=1)
        row = next(row for row in result['report']['registered_experiments']['outcomes'] if row['action_id'] == action['action_id'])
        assert row['decision'] == 'retired' and row['receipt'] and row['removed_logical_bytes'] == len(before), row
        assert target.stat().st_ino == original_inode
        assert {p.name for p in target.iterdir()} == {'.lane-scratch.v1.json', '.registered-experiment.v1.json'}
        repeated = _run_shipped_gc_sandbox(value, action, time.time(), pins, realtime=True, invocation=2)
        assert not repeated['report']['registered_experiments']['outcomes']
        if method == 'offload':
            cloud = Cloud()
            cloud.objects = {item['key']: base64.b64decode(item['payload'], validate=True) for item in result['objects']}
            cloud.metadata = result['object_metadata']
            archive._client = lambda *args: (cloud, 'development-only')
            events = [json.loads(p.read_bytes()) for p in (root / 'state/requests/experiment-records/operations' / action['action_id']).glob('e-*.json')]
            ready = next(event for event in events if event['event_kind'] == 'preservation_ready')
            assert ready['body']['archive']['full_byte_service_account_readback_passed'] is True
            assert ready['sequence'] < min(event['sequence'] for event in events if event['event_kind'] == 'member_removed')
            restore = issuer.issue_experiment_restore_intent(grant['intent_id'], principal='operator', owner='owner',
                lease_ttl_seconds=60, expires_at_epoch=time.time() + 120, installed_config_path=value['config'], now=time.time)
            ledger = root / 'actual-diagnostic-reservation-ledger'
            ledger.mkdir(mode=0o700)
            restoration.reserve_control_plane_disk = functools.partial(disk.reserve_control_plane_disk, reservation_root=ledger)
            # This is an owned temporary foreign file. Refusal must preserve its
            # bytes/inode; remove that same fixture inode only before replay.
            fd = os.open(report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(b'owned-foreign-fixture')
            foreign_inode = report.stat().st_ino
            try:
                refused = issuer.restore_registered_experiment(restore['action_id'], expected_restore_intent=restore['restore_intent'],
                    installed_config_path=value['config'], now=time.time, _pins_root=pins)
            except ValueError:
                pass
            else:
                assert refused['decision'] in {'kept', 'refused'}, refused
            assert report.stat().st_ino == foreign_inode and report.read_bytes() == b'owned-foreign-fixture'
            report.unlink()
            outcome = issuer.restore_registered_experiment(restore['action_id'], expected_restore_intent=restore['restore_intent'],
                installed_config_path=value['config'], now=time.time, _pins_root=pins)
            assert outcome['decision'] == 'restored', outcome
            assert report.read_bytes() == before and target.stat().st_ino == original_inode
            _ordinary_denied(target, value, account)
            try:
                diagnostic.run_registered_disk_diagnostic(grant['intent_id'], expected_intent=grant['intent'],
                    request_path=request, installed_config_path=value['config'], now=time.time)
            except ValueError:
                pass
            else:
                raise AssertionError('historical completion reopened expired producer')
            assert report.read_bytes() == before
        receipts.append(method)
    assert receipts == ['delete', 'offload']
    return dict(status='passed', actual_ordinary_uid=account.pw_uid, real_lease_expiry=True,
        default_off_kept=True, actual_delete=True, actual_offload=True, full_readback_before_remove=True,
        actual_restore=True, restore_no_overwrite=True, sealed_writer_denied=True, actual_open_fd_kept=True,
        zero_repeat_credit=True, shipped_gc_sandbox=True)
