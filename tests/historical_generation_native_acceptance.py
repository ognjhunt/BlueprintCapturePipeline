"""Real owner-approved historical worker on a tiny disposable Linux install.

Only compiled installation paths are repinned. Owner records, tree hashes,
kernel rights, current readers and the journal are produced by their real code.
No live host or provider is used.
"""
from __future__ import annotations

import ast
import base64
import errno
import hashlib
import json
import os
import pwd
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode() + b'\n'


def _write(path, value, mode=0o600):
    path.write_bytes(value)
    path.chmod(mode)


def _stage(root):
    """Bounded local import closure, rather than copying payloads or a checkout."""
    source = Path(__file__).parents[1]
    origin = source / 'src/blueprint_pipeline'
    destination = root / 'python/blueprint_pipeline'
    destination.mkdir(parents=True, mode=0o755)
    todo = ['__init__', 'control_plane_lane_historical_action', 'control_plane_lane_scratch_census',
            'control_plane_lane_experiment_archive']
    seen, size = set(), 0
    while todo:
        name = todo.pop()
        if name in seen or not (origin / (name + '.py')).is_file():
            continue
        seen.add(name)
        raw = (origin / (name + '.py')).read_bytes()
        size += len(raw)
        assert len(seen) <= 256 and size <= 8 * 1024**2
        _write(destination / (name + '.py'), raw, 0o644)
        for node in ast.walk(ast.parse(raw)):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 1:
                names = [node.module.split('.')[0]] if node.module else [alias.name for alias in node.names]
            elif node.module == 'blueprint_pipeline':
                names = [alias.name for alias in node.names]
            elif node.module and node.module.startswith('blueprint_pipeline.'):
                names = [node.module.split('.')[1]]
            else:
                names = []
            todo.extend(name for name in names if (origin / (name + '.py')).is_file())
    _write(root / 'python/fixture_acceptance.py', Path(__file__).read_bytes(), 0o644)
    _write(root / 'python/historical_generation_fake_cloud.py',
           (source / 'tests/historical_generation_fake_cloud.py').read_bytes(), 0o644)
    package = root / 'operator/operator_door'
    package.mkdir(parents=True, mode=0o700)
    for name in ('__init__.py', 'config.py'):
        _write(package / name, (source / 'deploy/operator-door/operator_door' / name).read_bytes())


def _namespace(root):
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer
    from blueprint_pipeline import control_plane_lane_owner_consents as owners
    from blueprint_pipeline import control_plane_lane_legacy_owner as legacy
    from blueprint_pipeline import control_plane_lane_historical_unit as unit
    # A fixed, protected disposable installation. This changes no authority,
    # time, namespace, process or service-property observation.
    consumer.LANE_ROOTS = (root / 'work/lanes', root / 'inputs/lanes')
    owners.INSTALLED_PACKAGE_ROOT = root / 'operator'
    legacy._GC_UNIT = root / 'gc.service'
    unit._EXECUTABLE = str(root / 'action-entry')


def worker_main(root, action_id):
    root = Path(root)
    _namespace(root)
    from blueprint_pipeline.control_plane_lane_historical_action import run_historical_action
    cloud = None
    cloud_fixture = root / 'cloud-fixture.json'
    if cloud_fixture.exists():
        from historical_generation_fake_cloud import Cloud
        from blueprint_pipeline import control_plane_lane_experiment_archive as transport
        seed = json.loads(cloud_fixture.read_bytes())
        cloud = Cloud(corrupt=seed['corrupt'])
        cloud.objects = {key: base64.b64decode(raw) for key, raw in seed['objects'].items()}
        cloud.metadata = seed['metadata']
        transport._client = lambda _files, _config: (cloud, 'development-only')
    def emit(receipt):
        if cloud is not None:
            receipt = dict(receipt, _fixture_remote=dict(corrupt=cloud.corrupt,
                objects={key: base64.b64encode(raw).decode() for key, raw in cloud.objects.items()},
                metadata=cloud.metadata, calls=cloud.calls))
        print(json.dumps(receipt), flush=True)
    # Fault injection interrupts actual completed syscalls/publications. It
    # never supplies a kernel observation, authority record or success result.
    interruption = root / 'interrupt-once'
    if interruption.exists():
        phase = interruption.read_text()
        if phase in ('fenced', 'removed'):
            from blueprint_pipeline.control_plane_lane_historical_action import _Worker
            original_record = _Worker.record
            def record(self, kind, body):
                original_record(self, kind, body)
                if kind == phase:
                    raise RuntimeError('fixture_interrupted_after_' + phase)
            _Worker.record = record
        elif phase == 'chown':
            original_chown = os.fchown
            target = (root / 'work/old-owner-diagnostics').stat()
            def chown(fd, uid, gid):
                original_chown(fd, uid, gid)
                observed = os.fstat(fd)
                if (observed.st_dev, observed.st_ino) == (target.st_dev, target.st_ino):
                    raise RuntimeError('fixture_interrupted_after_chown')
            os.fchown = chown
        elif phase == 'unlink':
            original_unlink = os.unlink
            def unlink(name, *, dir_fd=None):
                original_unlink(name, dir_fd=dir_fd)
                if name == 'two.log':
                    raise RuntimeError('fixture_interrupted_after_unlink')
            os.unlink = unlink
        else:
            raise AssertionError('unknown fixture interruption')
    try:
        receipt = run_historical_action(installed_config_path=root / 'door.json',
                                       action_id=action_id, now=time.time())
    except BaseException as error:
        emit(dict(status='failed', error_type=type(error).__name__, code=str(error)))
        raise
    emit(receipt)


def _launch_worker(entry, action_id, target, journals, *, expected='completed'):
    from blueprint_pipeline.control_plane_lane_historical_dispatch import _unit_property_assignments
    unit = 'blueprint-historical-generation-' + action_id
    def observations():
        log = subprocess.run(['/usr/bin/journalctl', '--unit=' + unit, '--output=cat',
            '--no-pager', '--lines=64'], capture_output=True, text=True, timeout=5)
        assert log.returncode == 0 and len(log.stdout.encode()) <= 32768
        return [json.loads(line) for line in log.stdout.splitlines() if line.startswith('{')]
    previous = observations()
    done = subprocess.run(['/usr/bin/systemd-run', '--unit=' + unit, '--no-block', '--collect',
        *('--property=' + value for value in _unit_property_assignments(target, journals)),
        '--', str(entry), action_id], capture_output=True, text=True, timeout=10)
    assert done.returncode == 0, done.stdout + done.stderr
    # The launcher must exit: its argv contains the selected write mount and
    # therefore is a real reference. Wait on one retained kernel cgroup FD in
    # this existing parent, without spawning pollers or changing its FD set.
    events = Path('/sys/fs/cgroup/system.slice') / (unit + '.service') / 'cgroup.events'
    deadline, fd = time.monotonic() + 60, None
    while time.monotonic() < deadline:
        try:
            fd = os.open(events, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            break
        except FileNotFoundError:
            if time.monotonic() >= deadline - 55:
                # A short completed cached-result unit may already be collected.
                # Its sole current journal receipt is still required below.
                break
            time.sleep(0.01)
    if fd is not None:
        try:
            populated = False
            while time.monotonic() < deadline:
                try:
                    raw = os.pread(fd, 1025, 0)
                except OSError as error:
                    assert error.errno == errno.ENODEV, error
                    break
                assert 0 < len(raw) <= 1024
                fields = dict(line.split() for line in raw.splitlines())
                assert fields.get(b'populated') in (b'0', b'1')
                if fields[b'populated'] == b'1':
                    populated = True
                if populated and fields[b'populated'] == b'0':
                    break
                time.sleep(0.05)
            else:
                raise AssertionError('actual worker did not reach terminal cgroup state')
        finally:
            os.close(fd)
    receipt_deadline = time.monotonic() + 2
    current = observations()
    while current == previous and time.monotonic() < receipt_deadline:
        # The worker is terminal before these journal queries. Allow its final
        # stdout record to reach journald; no query races an active scan.
        time.sleep(0.05)
        current = observations()
    assert current[:-1] == previous and len(current) == len(previous) + 1, current
    receipt = current[-1]
    remote = receipt.pop('_fixture_remote', None)
    if remote is not None:
        assert len(_encoded(remote)) <= 16384
        _write(entry.parent / 'cloud-fixture.json', _encoded(remote))
    assert receipt['status'] == expected, receipt
    return receipt


def connected_delete(interruption=None, *, action='delete', corrupt=False):
    assert sys.platform == 'linux' and os.geteuid() == 0
    assert os.environ.get('BLUEPRINT_DISPOSABLE_LINUX_TEST') == '1'
    assert Path('/proc/1/exe').resolve() == Path('/usr/lib/systemd/systemd')
    with tempfile.TemporaryDirectory(prefix='blueprint-historical-connected-', dir='/var/lib') as temporary:
        root = Path(temporary)
        root.chmod(0o755)
        _stage(root)
        _namespace(root)
        if action == 'offload':
            _write(root / 'cloud-fixture.json', _encoded(dict(corrupt=corrupt, objects={}, metadata={})))
        from blueprint_pipeline import control_plane_lane_owner_consents as owners
        from blueprint_pipeline import control_plane_lane_historical_authority as authority
        from blueprint_pipeline.control_plane_lane_historical_action import run_historical_action
        from blueprint_pipeline.control_plane_lane_scratch_census import build_census
        from blueprint_pipeline.control_plane_storage_pins import PIN_KINDS
        for name in ('work', 'inputs', 'state', 'state/requests', 'control-state',
                     'queues', 'evidence', 'settlement', 'pins', 'release'):
            (root / name).mkdir(mode=0o755)
        for kind in PIN_KINDS:
            (root / 'pins' / kind).mkdir(mode=0o755)
        for name, lock in [('owner-consents', '.owner-consents.lock'),
                           ('historical-generation-actions', '.historical-generation.lock')]:
            directory = root / 'state/requests' / name
            directory.mkdir(mode=0o700)
            _write(directory / lock, b'')
        journals = root / 'state/requests/historical-generation-journals'
        journals.mkdir(mode=0o700)
        (root / 'active').symlink_to(root / 'release')
        _write(root / 'gc.service', b'[Service]\n')
        environment = '\n'.join(key + '=' + str(root / value) for key, value in (
            ('BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS', 'queues'),
            ('BLUEPRINT_CONTROL_PLANE_GC_EVIDENCE_ROOTS', 'evidence'),
            ('BLUEPRINT_CONTROL_PLANE_GC_SETTLEMENT_ROOTS', 'settlement'),
            ('BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT', 'pins')))
        _write(root / 'gc.env', environment.encode() + b'\n')
        policy = dict(schema_version=owners.POLICY_SCHEMA, enabled=True, principals=[dict(
            principal='operator', owners=['owner'], allowed_actions=['register', 'delete', 'offload'],
            max_consent_seconds=3600)])
        _write(root / 'policy.json', _encoded(policy))
        settings = dict(state_root=str(root / 'state'), control_plane_state=str(root / 'control-state'),
            lane_scratch_work_root=str(root / 'work/lanes'), lane_scratch_inputs_root=str(root / 'inputs/lanes'),
            lane_owner_policy_file=str(root / 'policy.json'), owner_census_decisions_enabled=1,
            experiment_gc_environment_file=str(root / 'gc.env'), active_release_link=str(root / 'active'))
        config = root / 'door.json'
        _write(config, _encoded(settings))
        target = root / 'work/old-owner-diagnostics'
        (target / 'nested').mkdir(parents=True, mode=0o700)
        original = {'one.log': b'original diagnostics\n', 'nested/two.log': b'nested owner bytes\n'}
        foreign = pwd.getpwnam('nobody')
        for relative, raw in original.items():
            _write(target / relative, raw)
        for path in (target, target / 'nested', *(target / name for name in original)):
            os.chown(path, foreign.pw_uid, foreign.pw_gid)
        # Default-off refuses before an ID can confer authority or create state.
        try:
            run_historical_action(installed_config_path=config, action_id='a' * 32, now=time.time())
        except ValueError as error:
            assert str(error).endswith('action_disabled')
        else:
            raise AssertionError('default-off worker executed')
        assert not list(journals.iterdir())
        settings['historical_generation_actions_enabled'] = True
        _write(config, _encoded(settings))
        clock = time.time()
        census = build_census(work_root=root / 'work', inputs_root=root / 'inputs',
            process_root=Path('/proc'), pins_root=root / 'pins', queue_roots=[root / 'queues'],
            active_run_roots=[root / 'evidence', root / 'settlement'], release_link=root / 'active', now=clock)
        assert census['status'] == 'complete', census['scan_errors']
        raw = _encoded(census)
        annotations = _encoded(dict(schema_version='control_plane_lane_scratch_annotations.v1',
            census_digest='sha256:' + hashlib.sha256(raw).hexdigest(), decisions=[dict(path=str(target),
            action='register', owner='owner', lane='diagnostics', name='old-owner-diagnostics',
            reason='finished-diagnostic-run', class_intent='scratch', cleanup='owner_review',
            ttl_seconds=900, run_ref='real-tiny-completed-fixture')]))
        _write(root / 'census.json', raw)
        _write(root / 'annotations.json', annotations)
        owners.issue_owner_consent(root / 'census.json', root / 'annotations.json',
            census_sha256='sha256:' + hashlib.sha256(raw).hexdigest(), census_size_bytes=len(raw),
            annotations_sha256='sha256:' + hashlib.sha256(annotations).hexdigest(),
            annotations_size_bytes=len(annotations), principal='operator', selected_paths=(str(target),),
            expires_at_epoch=clock + 800, installed_config_path=config, now=clock, monotonic=time.monotonic)
        consent = next((root / 'state/requests/owner-consents').glob('*.json'))
        selected = authority.issue_historical_packet(installed_config_path=config, selected_path=str(target),
            consent_id=consent.stem, consent_sha256='sha256:' + hashlib.sha256(consent.read_bytes()).hexdigest(),
            consent_size_bytes=consent.stat().st_size, now=time.time())
        decision = authority.approve_historical_decommission(installed_config_path=config,
            packet_id=selected['packet_id'], ack_packet_digest=selected['packet_digest'], principal='operator',
            owner='owner', action=action, finished_run_ref='real-tiny-completed-fixture',
            no_future_writers=True, no_future_readers=True, expires_at_epoch=clock + 700, now=time.time())
        action_id = decision['action_id']
        entry = root / 'action-entry'
        _write(entry, ('#!/opt/blueprint-native-test-venv/bin/python\nimport sys\n'
            + 'sys.path.insert(0,' + repr(str(root / 'python')) + ')\n'
            + 'from fixture_acceptance import worker_main\nassert len(sys.argv)==2\n'
            + 'worker_main(' + repr(str(root)) + ',sys.argv[1])\n').encode(), 0o755)
        if interruption:
            _write(root / 'interrupt-once', interruption.encode())
            interrupted = _launch_worker(entry, action_id, target, journals, expected='failed')
            assert interrupted['code'] == 'fixture_interrupted_after_' + interruption
            expected_paths = set(original) - ({'nested/two.log'} if interruption in ('removed', 'unlink') else set())
            assert all((target / path).read_bytes() == original[path] for path in expected_paths)
            if interruption in ('removed', 'unlink'):
                assert not (target / 'nested/two.log').exists()
            assert target.stat().st_uid == 0
            initial = [path.read_bytes() for path in sorted((journals / action_id).glob('e-*.json'))]
            head = json.loads(initial[-1])
            assert head['kind'] == {'fenced': 'fenced', 'chown': 'fence_intent',
                                   'removed': 'removed', 'unlink': 'removal_intent'}[interruption]
            (root / 'interrupt-once').unlink()
        if corrupt:
            refused = _launch_worker(entry, action_id, target, journals, expected='failed')
            assert refused['code'].endswith('archive_preservation_failed'), refused
            assert all((target / path).read_bytes() == raw for path, raw in original.items())
            events = [json.loads(path.read_bytes()) for path in (journals / action_id).glob('e-*.json')]
            assert not any(event['kind'] in ('preservation', 'removal_intent', 'removed', 'final') for event in events)
            remote = json.loads((root / 'cloud-fixture.json').read_bytes())
            assert remote['calls'].count('readback') == 1 and remote['calls'][-1] == 'client_closed'
            return dict(historical_corrupt_offload_keeps_bytes=True)
        receipt = _launch_worker(entry, action_id, target, journals)
        assert receipt['action'] == action
        assert receipt['removed_files'] == (1 if interruption == 'unlink' else 2) and receipt['removed_directories'] == 1
        assert receipt['logical_bytes'] == sum(map(len, original.values())) - (len(original['nested/two.log']) if interruption == 'unlink' else 0)
        assert receipt['uncertain_removed_allocated_bytes'] == 0
        if interruption:
            assert receipt['uncertain_removed_members'] == int(interruption == 'unlink')
        assert receipt['root_directory_retained'] is True and not list(target.iterdir())
        assert target.stat().st_uid == 0 and target.stat().st_mode & 0o777 == 0o700
        events = [json.loads(path.read_bytes()) for path in sorted((journals / action_id).glob('e-*.json'))]
        assert events[-1]['kind'] == 'final' and events[-1]['body'] == receipt
        assert len([event for event in events if event['kind'] == 'removed']) == (2 if interruption == 'unlink' else 3)
        uncertain = [event for event in events if event['kind'] == 'removal_uncertain']
        assert len(uncertain) == int(interruption == 'unlink')
        assert all(event['body']['observed_removed_allocated_bytes'] == 0 for event in uncertain)
        assert receipt['observed_removed_allocated_bytes'] == sum(event['body']['observed_removed_allocated_bytes'] for event in events if event['kind'] == 'removed')
        if action == 'offload':
            preservation = [event for event in events if event['kind'] == 'preservation']
            assert len(preservation) == 1
            assert receipt['preservation_event_digest'] == preservation[0]['event_digest']
            assert preservation[0]['sequence'] < min(event['sequence'] for event in events if event['kind'] == 'removal_intent')
            assert receipt['preservation'] == preservation[0]['body']['archive']
            assert receipt['preservation']['full_byte_service_account_readback_passed'] is True
            remote = json.loads((root / 'cloud-fixture.json').read_bytes())
            assert remote['calls'].count('readback') == 1
        assert all(events[index]['previous_event_digest'] == events[index - 1]['event_digest']
                   for index in range(1, len(events)))
        if interruption:
            assert [path.read_bytes() for path in sorted((journals / action_id).glob('e-*.json'))][:len(initial)] == initial
            assert sum(event['kind'] == 'intent' for event in events) == 1
            assert sum(event['kind'] == 'fence_intent' and event['body']['path'] == '' for event in events) == 1
        before = {path.name: path.read_bytes() for path in (journals / action_id).iterdir()}
        repeated = _launch_worker(entry, action_id, target, journals)
        assert repeated['idempotent'] is True
        assert repeated['observed_removed_allocated_bytes'] == 0
        assert repeated['removed_files'] == repeated['removed_directories'] == 0
        assert repeated['original_final_event_digest'] == events[-1]['event_digest']
        if action == 'offload':
            remote = json.loads((root / 'cloud-fixture.json').read_bytes())
            assert remote['calls'].count('readback') == 1
        assert {path.name: path.read_bytes() for path in (journals / action_id).iterdir()} == before
        assert not list(target.iterdir())
        # A completed receipt never adopts a rewritten or repopulated tombstone.
        _write(target / 'changed-after-final', b'keep changed bytes')
        changed = _launch_worker(entry, action_id, target, journals, expected='failed')
        assert changed['code'].endswith('action_tombstone_changed'), changed
        assert (target / 'changed-after-final').read_bytes() == b'keep changed bytes'
        assert {path.name: path.read_bytes() for path in (journals / action_id).iterdir()} == before
        if action == 'offload':
            return dict(historical_offload_full_readback=True)
        return dict(actual_owner_approved_delete=True, original_member_journal=True,
                    historical_delete_idempotent=True)


def connected_delete_recovery():
    for phase in (None, 'fenced', 'chown', 'removed', 'unlink'):
        try:
            connected_delete(phase)
        except Exception as error:
            raise AssertionError('connected phase=' + str(phase) + ':' + str(error)) from error
    connected_delete(action='offload')
    connected_delete(action='offload', corrupt=True)
    return dict(actual_owner_approved_delete=True, original_member_journal=True,
                historical_delete_idempotent=True, original_fence_recovered=True,
                interrupted_removal_recovered=True, uncertain_removal_credit_zero=True,
                historical_offload_full_readback=True, historical_corrupt_offload_keeps_bytes=True)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
    print(json.dumps(connected_delete(), sort_keys=True))
