"""Real owner-approved historical worker on a tiny disposable Linux install.

Only compiled installation paths are repinned. Owner records, tree hashes,
kernel rights, current readers and the journal are produced by their real code.
No live host or provider is used.
"""
from __future__ import annotations

import ast
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
    todo = ['__init__', 'control_plane_lane_historical_action', 'control_plane_lane_scratch_census']
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


def _unreadable_process_channels():
    """Bounded metadata diagnostics; never expose command/environment bytes."""
    blocked = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            fields = (process / 'stat').read_bytes().rpartition(b') ')[2].split()
            kernel_thread = bool(int(fields[6]) & 0x00200000)
        except (OSError, ValueError, IndexError):
            kernel_thread = None
        for channel in ('cmdline', 'environ', 'cwd', 'fd'):
            try:
                if channel in ('cmdline', 'environ'):
                    with (process / channel).open('rb') as stream:
                        stream.read(1)
                elif channel == 'cwd':
                    os.readlink(process / channel)
                else:
                    for descriptor in (process / 'fd').iterdir():
                        os.readlink(descriptor)
            except FileNotFoundError:
                continue
            except OSError as error:
                blocked.append(dict(pid=int(process.name), channel=channel,
                                    errno=error.errno, kernel_thread=kernel_thread))
                if len(blocked) == 16:
                    return blocked
    return blocked


def worker_main(root, action_id):
    root = Path(root)
    _namespace(root)
    from blueprint_pipeline.control_plane_lane_historical_action import run_historical_action
    from blueprint_pipeline import control_plane_lane_historical_processes as processes
    from blueprint_pipeline.control_plane_kernel_process import kernel_has_no_user_memory
    inspect_process, read_channel = processes._inspect_process, processes._Scan.read
    def diagnosed_read(self, directory, name, cap=1024**2):
        try:
            return read_channel(self, directory, name, cap)
        except OSError as error:
            if not isinstance(error, ProcessLookupError):
                print('PROCESS_CHANNEL:' + json.dumps(dict(channel=name, errno=error.errno)), flush=True)
            raise
    def diagnosed_inspection(scan, directory, pid, target, identities, namespaces, host_mount, root_identity):
        try:
            return inspect_process(scan, directory, pid, target, identities, namespaces, host_mount, root_identity)
        except BaseException:
            facts = dict(pid=int(pid))
            def raw(name, cap):
                descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
                try:
                    result = os.read(descriptor, cap + 1)
                    if len(result) > cap:
                        raise ValueError('diagnostic bounded')
                    return result
                finally:
                    os.close(descriptor)
            facts['kernel_no_mm_observed'] = kernel_has_no_user_memory(raw, pid)
            try:
                view = processes._namespace(directory, kernel=facts['kernel_no_mm_observed'])
                facts.update(pid_namespace_matches=view[0] == namespaces[0],
                    user_namespace_matches=view[1] == namespaces[1],
                    mount_matches_worker=view[2] == namespaces[2], mount_matches_host=view[2] == host_mount,
                    mount_absent=view[2] is None)
            except (OSError, ValueError) as error:
                facts['namespace_error'] = getattr(error, 'errno', None)
            try:
                current = os.stat('root', dir_fd=directory)
                facts['root_identity_matches'] = (current.st_dev, current.st_ino) == root_identity
            except OSError as error:
                facts['root_errno'] = error.errno
            print('PROCESS_GUARD:' + json.dumps(facts), flush=True)
            raise
    processes._inspect_process = diagnosed_inspection
    processes._Scan.read = diagnosed_read
    try:
        receipt = run_historical_action(installed_config_path=root / 'door.json',
                                       action_id=action_id, now=time.time())
    except BaseException as error:
        print(json.dumps(dict(status='failed', error_type=type(error).__name__, code=str(error))), flush=True)
        raise
    print(json.dumps(receipt), flush=True)


def connected_delete():
    assert sys.platform == 'linux' and os.geteuid() == 0
    assert os.environ.get('BLUEPRINT_DISPOSABLE_LINUX_TEST') == '1'
    assert Path('/proc/1/exe').resolve() == Path('/usr/lib/systemd/systemd')
    with tempfile.TemporaryDirectory(prefix='blueprint-historical-connected-', dir='/var/lib') as temporary:
        root = Path(temporary)
        root.chmod(0o755)
        _stage(root)
        _namespace(root)
        from blueprint_pipeline import control_plane_lane_owner_consents as owners
        from blueprint_pipeline import control_plane_lane_historical_authority as authority
        from blueprint_pipeline.control_plane_lane_historical_dispatch import _unit_property_assignments
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
        assert census['status'] == 'complete', dict(errors=census['scan_errors'],
            unreadable=_unreadable_process_channels())
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
            owner='owner', action='delete', finished_run_ref='real-tiny-completed-fixture',
            no_future_writers=True, no_future_readers=True, expires_at_epoch=clock + 700, now=time.time())
        action_id = decision['action_id']
        entry = root / 'action-entry'
        _write(entry, ('#!/opt/blueprint-native-test-venv/bin/python\nimport sys\n'
            + 'sys.path.insert(0,' + repr(str(root / 'python')) + ')\n'
            + 'from fixture_acceptance import worker_main\nassert len(sys.argv)==2\n'
            + 'worker_main(' + repr(str(root)) + ',sys.argv[1])\n').encode(), 0o755)
        unit = 'blueprint-historical-generation-' + action_id
        done = subprocess.run(['/usr/bin/systemd-run', '--unit=' + unit, '--no-block', '--collect',
            *('--property=' + value for value in _unit_property_assignments(target, journals)),
            '--', str(entry), action_id], capture_output=True, text=True, timeout=10)
        assert done.returncode == 0, done.stdout + done.stderr
        deadline = time.monotonic() + 60
        # Wait through the real manager, not a file that would contaminate the
        # protected immutable journal namespace. Terminal observations are in
        # the actual service journal, outside its target-only write mount.
        time.sleep(0.5)
        while time.monotonic() < deadline:
            observed = subprocess.run(['/usr/bin/systemctl', 'show', unit,
                '--property=ActiveState', '--value'], capture_output=True, text=True, timeout=5)
            if observed.stdout.strip() in ('inactive', 'failed'):
                break
            time.sleep(0.5)
        else:
            raise AssertionError('actual worker did not reach terminal state')
        log = subprocess.run(['/usr/bin/journalctl', '--unit=' + unit, '--output=cat',
            '--no-pager', '--lines=64'], capture_output=True, text=True, timeout=5)
        assert log.returncode == 0 and len(log.stdout.encode()) <= 32768
        observations = [json.loads(line) for line in log.stdout.splitlines() if line.startswith('{')]
        assert len(observations) == 1, log.stdout
        receipt = observations[0]
        diagnostics = [line for line in log.stdout.splitlines() if line.startswith('PROCESS_')]
        assert receipt['status'] == 'completed', dict(receipt=receipt, diagnostics=diagnostics)
        assert receipt['removed_files'] == 2 and receipt['removed_directories'] == 1
        assert receipt['logical_bytes'] == sum(map(len, original.values()))
        assert receipt['root_directory_retained'] is True and not list(target.iterdir())
        assert target.stat().st_uid == 0 and target.stat().st_mode & 0o777 == 0o700
        events = [json.loads(path.read_bytes()) for path in sorted((journals / action_id).glob('e-*.json'))]
        assert events[-1]['kind'] == 'final' and events[-1]['body'] == receipt
        assert len([event for event in events if event['kind'] == 'removed']) == 3
        assert all(events[index]['previous_event_digest'] == events[index - 1]['event_digest']
                   for index in range(1, len(events)))
        return dict(actual_owner_approved_delete=True, original_member_journal=True)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
    print(json.dumps(connected_delete(), sort_keys=True))
