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
import signal
import stat
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
            'control_plane_lane_experiment_archive', 'control_plane_lane_historical_restore_authority']
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
    from blueprint_pipeline import control_plane_lane_historical_dispatch as dispatch
    from blueprint_pipeline import control_plane_disk_ledger as ledger
    # A fixed, protected disposable installation. This changes no authority,
    # time, namespace, process or service-property observation.
    consumer.LANE_ROOTS = (root / 'work/lanes', root / 'inputs/lanes')
    owners.INSTALLED_PACKAGE_ROOT = root / 'operator'
    legacy._GC_UNIT = root / 'gc.service'
    unit._EXECUTABLE = str(root / 'action-entry')
    ledger.DEFAULT_RESERVATION_ROOT = dispatch.DEFAULT_RESERVATION_ROOT = root / 'disk-reservations'
    if not ledger.DEFAULT_RESERVATION_ROOT.exists():
        ledger.DEFAULT_RESERVATION_ROOT.mkdir(mode=0o2770)
        ledger.DEFAULT_RESERVATION_ROOT.chmod(0o2770)


def worker_main(root, action_id):
    root = Path(root)
    # The controller retains its startup and cgroup descriptors before allowing
    # real reference scans. Opening a new poll descriptor during the scan was
    # itself a genuine changing-FD observation. This supplies no reader proof.
    ready = root / ('.fixture-unit-ready-' + action_id)
    deadline = time.monotonic() + 10
    while ready.read_bytes() != b'ready\n':
        assert time.monotonic() < deadline, 'fixture_controller_startup_not_complete'
        time.sleep(0.01)
    _namespace(root)
    from blueprint_pipeline.control_plane_lane_historical_action import run_historical_action
    from blueprint_pipeline import control_plane_lane_historical_processes as processes
    original_inspect = processes._inspect_process
    original_require = processes._require
    scan_failure = {}
    def require(value, code='process_unknown'):
        if not value:
            frame, frames = sys._getframe(1), []
            while frame is not None:
                if frame.f_code.co_filename == processes.__file__:
                    frames.append(dict(function=frame.f_code.co_name, line=frame.f_lineno))
                    if frame.f_code.co_name == 'refuse_historical_process_references':
                        before, after = frame.f_locals.get('names'), frame.f_locals.get('after')
                        if type(before) is list and type(after) is list:
                            scan_failure['pid_census'] = dict(before_count=len(before), after_count=len(after),
                                added=[int(pid) for pid in sorted(set(after)-set(before))[:8]],
                                removed=[int(pid) for pid in sorted(set(before)-set(after))[:8]])
                frame = frame.f_back
            scan_failure.update(error_type='HistoricalProcessError', errno=None, frames=frames)
        return original_require(value, code)
    processes._require = require
    def inspect(*args, **kwargs):
        try:
            return original_inspect(*args, **kwargs)
        except Exception as error:
            # Keep only code locations/class/errno, never foreign process data.
            trace = error.__traceback__
            frames = []
            while trace is not None:
                if trace.tb_frame.f_code.co_filename == processes.__file__:
                    frames.append(dict(function=trace.tb_frame.f_code.co_name,
                                       line=trace.tb_lineno))
                    if trace.tb_frame.f_code.co_name == '_inspect_process':
                        local = trace.tb_frame.f_locals
                        scan_failure['process_id'] = int(args[2])
                        scan_failure['process_start_tick'] = local.get('started')
                        slot = local.get('name')
                        if type(slot) is str and slot.isdecimal():
                            scan_failure['fd_slot'] = int(slot)
                trace = trace.tb_next
            scan_failure.update(error_type=type(error).__name__,
                                errno=getattr(error, 'errno', None), frames=frames,
                                process_is_init=args[2] == '1')
            raise
    processes._inspect_process = inspect
    cloud = None
    cloud_fixture = root / 'cloud-fixture.json'
    if cloud_fixture.exists():
        from historical_generation_fake_cloud import Cloud
        import boto3
        seed = json.loads(cloud_fixture.read_bytes())
        cloud = Cloud(corrupt=seed['corrupt'])
        cloud.objects = {key: base64.b64decode(raw) for key, raw in seed['objects'].items()}
        cloud.metadata = seed['metadata']
        def object_client(service, **options):
            assert service == 's3'
            assert options['endpoint_url'] == 'https://development-only.invalid'
            assert options['aws_access_key_id'] == 'development-only-access'
            assert options['aws_secret_access_key'] == 'development-only-secret'
            assert options['region_name'] == 'us-east-1'
            return cloud
        # Keep the real protected environment/credential acquisition and SDK
        # selection. Replace only the object transport; never contact a provider.
        boto3.client = object_client
    metadata_proof = None
    if (root / 'metadata-boundary-probe').exists():
        from blueprint_pipeline.control_plane_lane_historical_action import _Worker
        from blueprint_pipeline import control_plane_lane_historical_publication as publication
        original_probe_record = _Worker.record
        def probe_record(self, kind, body):
            nonlocal metadata_proof
            result = original_probe_record(self, kind, body)
            if kind == 'fenced' and metadata_proof is None:
                assert self.sandbox is not None and not self.sandbox.closed
                with self.checkpoint(journal=True) as (files, _, journal):
                    # Explicit non-authorizing metadata projections. This
                    # exercises full-size byte publication under real installed
                    # systemd/Landlock rights, not a fabricated generation/owner.
                    leaf = 'metadata-publication-probe'
                    files.location(journal.directory)
                    os.mkdir(leaf, 0o700, dir_fd=journal.directory)
                    child = files.open(leaf, os.O_RDONLY | os.O_DIRECTORY, parent=journal.directory)
                    child_birth = os.fstat(child)
                    prefix = b'{"schema_version":"metadata_publication_probe.v1","execution_authorized":false,"data":"'
                    suffix = b'"}\n'
                    cap = publication._CAPS['manifest']
                    payload = prefix + b'x' * (cap - len(prefix) - len(suffix)) + suffix
                    assert len(payload) == cap == publication._CAPS['historical_restore_snapshot']
                    for output, category in [('f' * 32 + '.manifest.json', 'manifest'),
                                             ('restore.snapshot.json', 'historical_restore_snapshot')]:
                        published = publication._publish(files, child, output, payload, kind=category)
                        assert published == dict(sha256='sha256:' + hashlib.sha256(payload).hexdigest(), size_bytes=cap)
                        check = files.open(output, os.O_RDONLY, parent=child)
                        birth = os.fstat(check)
                        assert birth.st_uid == birth.st_gid == 0 and birth.st_nlink == 1
                        assert stat.S_IMODE(birth.st_mode) == 0o600
                        assert files.read_bytes(check, cap) == payload  # third full comparison
                        try:
                            publication._publish(files, child, output, b'conflict', kind=category)
                        except ValueError as error:
                            assert str(error) == 'experiment_publication_destination_exists'
                        else:
                            raise AssertionError('metadata destination was overwritten')
                        assert os.stat(output, dir_fd=child, follow_symlinks=False) == os.fstat(check)
                        assert os.fstat(check).st_ino == birth.st_ino and os.fstat(check).st_size == cap
                        # Remove only this fixture's fully verified actual new
                        # inode while its original descriptor remains retained.
                        os.unlink(output, dir_fd=child)
                        files.close(check)
                    files.location(child)
                    named_child = os.stat(leaf, dir_fd=journal.directory, follow_symlinks=False)
                    assert (named_child.st_dev, named_child.st_ino) == (child_birth.st_dev, child_birth.st_ino)
                    assert not os.listdir(child)
                    os.rmdir(leaf, dir_fd=journal.directory)
                    files.close(child)
                    files.location(journal.directory)
                    os.fsync(journal.directory)
                    metadata_proof = dict(records=2, bytes_per_record=cap, full_comparisons=6,
                                          no_replace_conflicts=2, actual_landlock=True)
            return result
        _Worker.record = probe_record
    def emit(receipt):
        if metadata_proof is not None:
            receipt = dict(receipt, _fixture_metadata_proof=metadata_proof)
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
        if phase.startswith('metadata_'):
            from blueprint_pipeline import control_plane_lane_historical_journal as journal_code
            original_publish = journal_code._publish
            def publish(files, parent, name, payload, **options):
                if name != 'e-00001.json':
                    return original_publish(files, parent, name, payload, **options)
                original_link, original_sync = os.link, os.fsync
                linked = False
                def link(*args, **kwargs):
                    nonlocal linked
                    if phase == 'metadata_before_link':
                        os.kill(os.getpid(), signal.SIGKILL)
                    result = original_link(*args, **kwargs)
                    linked = True
                    if phase == 'metadata_after_link':
                        os.kill(os.getpid(), signal.SIGKILL)
                    return result
                def sync(fd):
                    if phase == 'metadata_before_parent_fsync' and linked and fd == parent:
                        os.kill(os.getpid(), signal.SIGKILL)
                    return original_sync(fd)
                os.link, os.fsync = link, sync
                try:
                    return original_publish(files, parent, name, payload, **options)
                finally:
                    os.link, os.fsync = original_link, original_sync
            journal_code._publish = publish
        elif phase in ('unlogged_directory', 'unlogged_member', 'reconcile_remove', 'reconcile_sync'):
            if phase == 'unlogged_directory':
                original_mkdir = os.mkdir
                def mkdir(name, *args, **kwargs):
                    result = original_mkdir(name, *args, **kwargs)
                    if name == '.historical-restore-' + action_id:
                        os.kill(os.getpid(), signal.SIGKILL)
                    return result
                os.mkdir = mkdir
            elif phase == 'unlogged_member':
                original_write = os.write
                def write(fd, raw):
                    path = os.readlink('/proc/self/fd/' + str(fd))
                    if '/.historical-restore-' + action_id + '/' in path:
                        assert len(raw) > 1
                        original_write(fd, memoryview(raw)[:1])
                        os.kill(os.getpid(), signal.SIGKILL)
                    return original_write(fd, raw)
                os.write = write
            else:
                original_unlink, original_sync = os.unlink, os.fsync
                from blueprint_pipeline import control_plane_lane_historical_restore_reconciliation_worker as reconciliation_code
                from blueprint_pipeline.control_plane_lane_historical_fence import _version
                original_pin = reconciliation_code._effect_pin
                selected, removed_parent = None, None
                def reconciliation_pin(worker, binding):
                    nonlocal selected
                    selected = _observe_reconciliation_pin(worker, binding, original_pin)
                reconciliation_code._effect_pin = reconciliation_pin
                def unlink(name, *args, **kwargs):
                    nonlocal removed_parent
                    parent = kwargs.get('dir_fd')
                    matched = selected is not None and name == selected['remove_member']['path'].rpartition('/')[2] \
                        and parent is not None and _version(os.fstat(parent)) == selected['parent_after'] \
                        and _version(os.stat(name, dir_fd=parent, follow_symlinks=False)) == selected['remove_member']['version']
                    result = original_unlink(name, *args, **kwargs)
                    if matched:
                        removed_parent = parent
                    if matched and phase == 'reconcile_remove':
                        os.kill(os.getpid(), signal.SIGKILL)
                    return result
                def sync(fd):
                    result = original_sync(fd)
                    if removed_parent == fd and phase == 'reconcile_sync':
                        assert _version(os.fstat(fd))[:6] == selected['parent_after'][:6]
                        os.kill(os.getpid(), signal.SIGKILL)
                    return result
                os.unlink, os.fsync = unlink, sync
        elif phase in ('fenced', 'removed', 'restore_final', 'before_restore_final', 'stage_removed',
                     'unwritten_stage', 'access_intent', 'stage_complete', 'restore_directory', 'restore_member',
                     'publish_intent', 'publish_rename', 'publish_observed', 'stage_remove_intent', 'stage_remove_effect',
                     'reconcile_intent', 'reconcile_delete_resume', 'reconcile_consumed'):
            from blueprint_pipeline.control_plane_lane_historical_action import _Worker
            original_record = _Worker.record
            def record(self, kind, body):
                if phase == 'publish_rename' and kind == 'restore_intent' and body.get('phase') == 'publish_observed':
                    raise RuntimeError('fixture_interrupted_after_' + phase)
                if phase == 'stage_remove_effect' and kind == 'restore_intent' and body.get('phase') == 'stage_removed':
                    raise RuntimeError('fixture_interrupted_after_' + phase)
                if phase == 'before_restore_final' and kind == 'restore_final':
                    raise RuntimeError('fixture_interrupted_after_' + phase)
                actual_event = original_record(self, kind, body)
                if phase in ('reconcile_intent', 'reconcile_delete_resume') and kind == 'restore_intent' and body.get('phase') == phase:
                    os.kill(os.getpid(), signal.SIGKILL)
                if phase == 'reconcile_consumed' and kind == 'restore_intent' and body.get('phase') == 'reconciled':
                    os.kill(os.getpid(), signal.SIGKILL)
                if phase == 'publish_intent' and kind == 'restore_intent' and body.get('phase') == 'publish':
                    raise RuntimeError('fixture_interrupted_after_' + phase)
                if phase == 'stage_remove_intent' and kind == 'restore_intent' and body.get('phase') == 'stage_remove':
                    raise RuntimeError('fixture_interrupted_after_' + phase)
                if phase == 'unwritten_stage' and kind == 'restore_intent' \
                        and body.get('phase') == 'directory' and body.get('path') == '':
                    from blueprint_pipeline.control_plane_lane_historical_generation import HistoricalGenerationError
                    raise HistoricalGenerationError('fixture_interrupted_after_' + phase)
                if phase == 'access_intent' and kind == 'restore_intent' \
                        and body.get('phase') == 'owner_rights' and body.get('path') == '':
                    raise RuntimeError('fixture_interrupted_after_' + phase)
                if kind == phase and phase in ('restore_directory', 'restore_member'):
                    from blueprint_pipeline.control_plane_lane_historical_generation import HistoricalGenerationError
                    raise HistoricalGenerationError('fixture_interrupted_after_' + phase)
                if kind == phase or (kind == 'restore_intent' and body.get('phase') == phase):
                    raise RuntimeError('fixture_interrupted_after_' + phase)
                return actual_event
            _Worker.record = record
        elif phase == 'restore_member_chown':
            original_chown = os.fchown
            def restore_member_chown(fd, uid, gid):
                original_chown(fd, uid, gid)
                if uid != 0 and stat.S_ISREG(os.fstat(fd).st_mode):
                    raise RuntimeError('fixture_interrupted_after_' + phase)
            os.fchown = restore_member_chown
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
        emit(dict(status='failed', error_type=type(error).__name__, code=str(error),
                  **({'_fixture_scan_failure': scan_failure} if scan_failure else {})))
        # The original class/code and safe observation location are retained.
        # An unsuccessful actual unit exits nonzero without an unbounded stack
        # pushing earlier same-ID receipts out of the bounded journal window.
        raise SystemExit(1) from None
    emit(receipt)


def _launch_worker(entry, action_id, target, journals, *, expected='completed', restore=False, launch=None):
    # Each invocation reaches a real terminal unit before another GC tick is
    # considered. Never overlap handles, change an ID or mint a new deadline.
    observations = []
    births = []
    def invoke():
        nonlocal births
        births = _capture_restore_births(journals, action_id) if restore else []
        return _launch_worker_once(entry, action_id, target, journals, restore=restore, launch=launch)
    receipt = _later_reference_attempts(invoke, journals, action_id, observations=observations)
    assert receipt['status'] == expected, receipt
    result = dict(receipt, _fixture_reference_refusals=observations) if observations else receipt
    return dict(result, _fixture_prior_member_births=births) if receipt.get('recovered_prefix') is True else result


def _observe_reconciliation_pin(worker, binding, verify):
    """Observe the actual held grant only after its syscall pin succeeds.

    A resumed unit may already have a durable original resume event; the fault
    must not require another event or replace the worker's authorization.
    """
    verify(worker, binding)
    return worker.effect_selected[-1][0]['packet']['scope']


def _launch_death_worker(entry, action_id, target, journals):
    """Same three-unit cadence; each None must come from fresh real SIGKILL."""
    result = _later_reference_attempts(lambda: _launch_worker_once(entry, action_id, target, journals,
        restore=True, process_death=True), journals, action_id)
    assert result is None, result


def _capture_restore_births(journals, action_id):
    """Actual protected whole-chain births immediately before the next unit."""
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.control_plane_lane_historical_journal import MAX_SUPPORTED_EVENT_BYTES
    directory = journals / action_id
    if not directory.exists():
        return []
    names = sorted(path.name for path in directory.glob('e-*.json'))
    assert 0 < len(names) <= 256
    events, previous, size = [], None, 0
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        scope = None
        for index, name in enumerate(names):
            assert name == f'e-{index:05d}.json'
            record = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            try:
                info = os.fstat(record)
                assert stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600
                assert info.st_uid == info.st_gid == 0 and info.st_nlink == 1 and 0 < info.st_size <= MAX_SUPPORTED_EVENT_BYTES
                raw = os.read(record, MAX_SUPPORTED_EVENT_BYTES + 1)
                assert len(raw) == info.st_size and not os.read(record, 1)
                assert os.fstat(record) == os.stat(name, dir_fd=fd, follow_symlinks=False)
            finally:
                os.close(record)
            size += len(raw)
            assert size <= 1024**2
            event = json.loads(raw)
            assert event['action_id'] == action_id and event['sequence'] == index
            assert event['event_digest'] == canonical_digest(event, digest_field='event_digest')
            assert event['previous_event_digest'] == previous and event['execution_authorized'] is False
            assert (event['kind'] == 'intent') == (index == 0)
            scope = event['scope_digest'] if scope is None else scope
            assert event['scope_digest'] == scope
            previous = event['event_digest']
            if event['kind'] == 'restore_member':
                events.append(event)
    finally:
        os.close(fd)
    assert len({event['body']['path'] for event in events}) == len(events)
    return events


def _durable_receipt(receipt):
    # Controller observations describe earlier real units. They are retained
    # separately and never become part of the worker's immutable final event.
    return {key: value for key, value in receipt.items()
            if key not in ('_fixture_reference_refusals', '_fixture_prior_member_births')}


def _later_reference_attempts(invoke, journals, action_id, *, observations=None):
    """Keep genuine unknowns; at most three later SAME-operation attempts.

    This is fixture cadence, not reader clearance or worker recovery. Every
    later real unit reauthenticates the original clock, authority and all native
    facts. An unsupported partial state remains KEEP rather than being retried.
    """
    directory = journals / action_id
    def prefix():
        if not directory.exists():
            return {}
        rows = {path.name: path.read_bytes() for path in directory.iterdir()}
        assert len(rows) <= 256 and sum(map(len, rows.values())) <= 1024**2
        return rows
    original = prefix()
    for attempt in range(3):
        receipt = invoke()
        current = prefix()
        assert all(current.get(name) == raw for name, raw in original.items()), 'original_journal_changed'
        if receipt is None:
            # This is an observed fixture death, never a worker success receipt.
            # Ordinary callers still require an actual completed worker result.
            return None
        if receipt.get('status') not in ('kept', 'failed') or receipt.get('code') != \
                'historical_generation_process_unknown':
            return receipt
        raw = current.get('e-00000.json')
        assert raw is not None, 'refused_unit_has_no_original_intent'
        intent = json.loads(raw)
        assert intent['kind'] == 'intent' and intent['action_id'] == action_id
        # Evidence of the real refusal is retained, not relabeled completion.
        print(json.dumps(dict(fixture_reference_refusal=receipt, action_id=action_id,
            attempt=attempt + 1, original_intent_sha256=hashlib.sha256(raw).hexdigest())), flush=True)
        if observations is not None:
            observations.append(dict(action_id=action_id, code=receipt['code'],
                original_intent_sha256=hashlib.sha256(raw).hexdigest(), attempt=attempt + 1))
        if attempt == 2:
            return receipt
        original = current
        time.sleep(0.05)
    raise AssertionError('bounded_reference_attempts_exhausted')


def _assert_restore_increment(receipt, original):
    # A genuine refused earlier unit may already have written every byte.
    # Its later recovered receipt credits zero, while the original durable
    # final below still has to account for the full manifest exactly once.
    fields = ('recovered_publication', 'recovered_before_final', 'recovered_access',
              'restarted_unwritten', 'recovered_stage', 'recovered_prefix', 'recovered_split', 'idempotent')
    assert all(field not in receipt or type(receipt[field]) is bool for field in fields), receipt
    phases = [field for field in fields if receipt.get(field) is True]
    assert len(phases) <= 1, receipt
    recovered = bool(phases and phases != ['restarted_unwritten'])
    expected = (0, 0) if recovered else (len(original), sum(map(len, original.values())))
    if phases == ['recovered_prefix']:
        from itertools import combinations
        reuse = (receipt.get('reused_files'), receipt.get('reused_logical_bytes'))
        assert all(type(value) is int and value >= 0 for value in reuse), receipt
        assert len(original) <= 8  # finite tiny fixture, not production accounting
        if '_fixture_prior_member_births' in receipt:
            births = receipt['_fixture_prior_member_births']
            assert type(births) is list
            rows = [event['body'] for event in births]
            assert len({row['path'] for row in rows}) == len(rows)
            assert all(row['path'] in original and row['size_bytes'] == len(original[row['path']])
                and row['sha256'] == 'sha256:' + hashlib.sha256(original[row['path']]).hexdigest() for row in rows)
            assert reuse == (len(rows), sum(row['size_bytes'] for row in rows)), receipt
        else:
            # Parser-only arithmetic projections. Native execution always
            # supplies the exact protected pre-unit birth set above.
            allowed = {(len(rows), sum(map(len, rows))) for count in range(len(original) + 1)
                       for rows in combinations(original.values(), count)}
            assert reuse in allowed, receipt
        expected = (len(original)-reuse[0], sum(map(len, original.values()))-reuse[1])
    assert (receipt['restored_files'], receipt['restored_logical_bytes']) == expected, receipt


def _assert_boundary_recovery(receipt, expected, observations, action_id, original_intent):
    """A later refused unit can advance the same already interrupted operation.

    This only checks fixture cadence after the genuine boundary/state assertion.
    Full original journal, birth inodes, final bytes and rights are checked below.
    It creates no success receipt, clock, birth or native clearance observation.
    """
    assert receipt.get('status') == 'completed'
    if receipt.get(expected) is True:
        if expected != 'recovered_prefix':
            assert receipt['restored_files'] == receipt['restored_logical_bytes'] == 0
        return
    assert receipt['restored_files'] == receipt['restored_logical_bytes'] == 0
    assert expected in ('recovered_stage', 'recovered_split', 'recovered_prefix')
    later = ('recovered_publication', 'recovered_before_final', 'recovered_access', 'idempotent')
    if expected == 'recovered_stage':
        later += ('recovered_split',)
    elif expected == 'recovered_prefix':
        later += ('recovered_split', 'recovered_stage')
    assert any(receipt.get(field) is True for field in later)
    assert 0 < len(observations) <= 3
    assert all(row == dict(action_id=action_id, code='historical_generation_process_unknown',
        original_intent_sha256=hashlib.sha256(original_intent).hexdigest(), attempt=index + 1)
        for index, row in enumerate(observations))


def _unit_cursor(unit):
    """Actual journald position before a new same-ID unit, never a receipt."""
    log = subprocess.run(['/usr/bin/journalctl', '--unit=' + unit, '--output=json',
        '--no-pager', '--lines=1'], capture_output=True, text=True, timeout=5)
    assert log.returncode == 0 and len(log.stdout.encode()) <= 32768
    lines = log.stdout.splitlines()
    if not lines or lines == ['-- No entries --']:
        return None
    assert len(lines) == 1
    cursor = json.loads(lines[0]).get('__CURSOR')
    assert type(cursor) is str and 0 < len(cursor) <= 4096
    assert all(char.isascii() and (char.isalnum() or char in ';=_-') for char in cursor)
    return cursor


def _unit_output(unit, cursor):
    # JSON otherwise replaces fields above4096bytes with null. Retain actual
    # messages in full inside the same finite output/time/record limits.
    argv = ['/usr/bin/journalctl', '--unit=' + unit, '--output=json', '--all',
        '--output-fields=MESSAGE', '--no-pager', '--lines=65']
    if cursor is not None:
        argv.append('--after-cursor=' + cursor)
    log = subprocess.run(argv, capture_output=True, text=True, timeout=5)
    assert log.returncode == 0 and len(log.stdout.encode()) <= 32768
    lines = log.stdout.splitlines()
    if lines == ['-- No entries --']:
        return ''
    assert len(lines) <= 64, 'journal_delta_overflow'
    messages = []
    for line in lines:
        record = json.loads(line)
        message = record.get('MESSAGE')
        assert type(message) is str, ('journal_delta_message_unknown', type(message).__name__)
        messages.append(message)
    return '\n'.join(messages)


def _launch_worker_once(entry, action_id, target, journals, *, restore=False, launch=None, process_death=False):
    from blueprint_pipeline.control_plane_lane_historical_dispatch import _unit_property_assignments
    unit = 'blueprint-historical-generation-' + action_id
    cursor = None
    def observations():
        return [json.loads(line) for line in _unit_output(unit, cursor).splitlines() if line.startswith('{')]
    history = observations()
    cursor = _unit_cursor(unit)
    previous = []
    ready = entry.parent / ('.fixture-unit-ready-' + action_id)
    _write(ready, b'')
    startup = os.open(ready, os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        if launch is None:
            done = subprocess.run(['/usr/bin/systemd-run', '--unit=' + unit, '--no-block', '--collect',
                *('--property=' + value for value in _unit_property_assignments(target, journals, restore=restore)),
                '--', str(entry), action_id], capture_output=True, text=True, timeout=10)
            assert done.returncode == 0, done.stdout + done.stderr
        else:
            phase = launch()
            assert phase['removed_bytes'] == phase['mutations'] == 0, phase
            if phase['units_started'] == 0:
                receipt = next(row for row in phase['outcomes'] if row['action_id'] == action_id)
                assert receipt['status'] == 'completed' and receipt['observation_only'] is True, receipt
                assert receipt['action_unit_started'] is receipt['execution_authorized'] is False
                assert observations() == previous
                return receipt
            assert phase['units_started'] == 1, phase
            assert phase['outcomes'][-1]['status'] == 'submitted', phase
        # The worker waits before scanning. Reap the launcher (its argv is a
        # real target reference), retain the actual cgroup FD, then signal using
        # the already held startup FD. No controller FD changes race the scan.
        events = Path('/sys/fs/cgroup/system.slice') / (unit + '.service') / 'cgroup.events'
        deadline, fd = time.monotonic() + 60, None
        while time.monotonic() < deadline - 55:
            try:
                fd = os.open(events, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
                break
            except FileNotFoundError:
                time.sleep(0.01)
        assert fd is not None, 'actual_worker_cgroup_missing_before_startup'
        try:
            assert os.write(startup, b'ready\n') == 6
            os.fsync(startup)
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
    finally:
        os.close(startup)
    if process_death:
        # No worker receipt is manufactured after SIGKILL. The actual emptied
        # cgroup above and service-manager signal record prove this boundary.
        output = _unit_output(unit, cursor)
        current = observations()
        if len(current) == 1 and current[0].get('status') in ('kept', 'failed') \
                and current[0].get('code') == 'historical_generation_process_unknown':
            # The real fault was not reached. Preserve its actual refusal for
            # the existing bounded SAME-e0 cadence, with no clock/authority edit.
            assert 'code=killed, status=9/KILL' not in output, dict(cursor=cursor, records=current)
        else:
            assert 'code=killed, status=9/KILL' in output, dict(cursor=cursor, output=output, records=current)
            assert current == previous, dict(failure='killed worker must not emit a terminal receipt',
                cursor=cursor, history=history, current=current)
            return None
    receipt_deadline = time.monotonic() + 2
    current = observations()
    while current == previous and time.monotonic() < receipt_deadline:
        # The worker is terminal before these journal queries. Allow its final
        # stdout record to reach journald; no query races an active scan.
        time.sleep(0.05)
        current = observations()
    assert current[:-1] == previous and len(current) == len(previous) + 1, current
    receipt = current[-1]
    entry_failure = receipt.pop('_fixture_entry_failure', None)
    if entry_failure is not None:
        print(json.dumps(dict(fixture_installed_entry_refusal=entry_failure, action_id=action_id)))
    metadata_proof = receipt.pop('_fixture_metadata_proof', None)
    if (entry.parent / 'metadata-boundary-probe').exists() and not any(
            '_fixture_metadata_proof' in row for row in history):
        assert metadata_proof == dict(records=2, bytes_per_record=1048576,
            full_comparisons=6, no_replace_conflicts=2, actual_landlock=True), metadata_proof
    else:
        assert metadata_proof is None
    remote = receipt.pop('_fixture_remote', None)
    if remote is not None:
        assert len(_encoded(remote)) <= 16384
        _write(entry.parent / 'cloud-fixture.json', _encoded(remote))
    return receipt


def _installed_entry(root, entry):
    """Stage production entry/closure; repin only disposable compiled namespaces.

    The fixture startup barrier settles actual controller descriptors before
    real scans. It supplies no reader/clock/owner/unit proof. The adjacent write
    challenge runs under the actual installed service properties before the
    production worker; no native observation or mutation guard is replaced.
    """
    import importlib.util
    source = Path(__file__).parents[1]
    stage_path = source / 'deploy/operator-door/stage-historical-runtime.py'
    spec = importlib.util.spec_from_file_location('historical_runtime_stage_fixture', stage_path)
    stager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stager)
    installed = root / 'operator'
    stager.stage(source / 'src/blueprint_pipeline', installed / 'historical-python')
    package = installed / 'historical-python/blueprint_pipeline'
    bindings = {
        'control_plane_lane_owner_consents': 'INSTALLED_PACKAGE_ROOT = Path(' + repr(str(installed)) + ')',
        'control_plane_scratch_lifetime': 'LANE_ROOTS = (' + ','.join(
            'Path(' + repr(str(root / name)) + ')' for name in ('work/lanes', 'inputs/lanes')) + ')',
        'control_plane_lane_legacy_owner': '_GC_UNIT = Path(' + repr(str(root / 'gc.service')) + ')',
        'control_plane_lane_historical_unit': '_EXECUTABLE = ' + repr(str(entry)),
        'control_plane_lane_historical_dispatch': '_ACTION_EXECUTABLE = ' + repr(str(entry)),
        'control_plane_disk_ledger': 'DEFAULT_RESERVATION_ROOT = Path(' + repr(str(root / 'disk-reservations')) + ')',
    }
    for name, binding in bindings.items():
        path = package / (name + '.py')
        _write(path, path.read_bytes() + ('\n# Fixed disposable installed namespace.\n' + binding + '\n').encode(), 0o644)
    boot = (source / 'deploy/operator-door/historical-generation-entry.py').read_text()
    boot = boot.replace("_ROOT = Path('/opt/blueprint/operator-door')", '_ROOT = Path(' + repr(str(installed)) + ')')
    boot = boot.replace("_CONFIG = '/etc/blueprint-operator-door/door.json'", '_CONFIG = ' + repr(str(root / 'door.json')))
    if (root / 'cloud-fixture.json').exists():
        import boto3
        # Repin the fixed SDK distribution to CI's sealed disposable root
        # installation. The production source loader still verifies every
        # actual import's ancestry/owner/mode/single-link identity. This is
        # not a service venv fallback or proof of a live host SDK installation.
        # uv may spell its sealed SDK through a lib/lib64 alias. Compile the
        # physical installation directory; the isolated loader then opens only
        # that canonical namespace with its original no-follow protections.
        distribution = Path(boto3.__file__).parent.parent.resolve(strict=True)
        boot = boot.replace("_SYSTEM_PACKAGES = Path('/usr/lib/python3/dist-packages')",
                            '_SYSTEM_PACKAGES = Path(' + repr(str(distribution)) + ')')
    adjacent = root / 'work/adjacent-unselected.log'
    assert adjacent.read_bytes() == b'adjacent original bytes\n'
    # A fixture-only synchronization/challenge; original source loader and
    # production worker remain unchanged after these actual observations.
    barrier = """        ready = Path(READY_PARENT) / ('.fixture-unit-ready-' + arguments[0])
        deadline = time.monotonic() + 10
        while ready.read_bytes() != b'ready\\n':
            assert time.monotonic() < deadline, 'fixture_controller_startup_not_complete'
            time.sleep(0.01)
        try:
            descriptor = os.open(ADJACENT, os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        except OSError as error:
            assert error.errno == 30, 'adjacent denial must be actual EROFS'
        else:
            os.close(descriptor)
            raise AssertionError('installed action can write adjacent target')
""".replace('READY_PARENT', repr(str(root))).replace('ADJACENT', repr(str(adjacent)))
    needle = '        from blueprint_pipeline.control_plane_lane_historical_action import run_historical_action\n'
    assert boot.count(needle) == 1
    boot = boot.replace(needle, barrier + needle)
    if (root / 'cloud-fixture.json').exists():
        transport = """        # installed_fixture_archive_transport: only the object transport is replaced.
        import base64
        import boto3
        fixture_namespace = {}
        exec(compile(_read_source(Path(FAKE_CLOUD)), FAKE_CLOUD, 'exec'), fixture_namespace)
        seed = json.loads(_read_source(Path(CLOUD_STATE)))
        cloud = fixture_namespace['Cloud'](corrupt=seed['corrupt'])
        cloud.objects = {key: base64.b64decode(raw) for key, raw in seed['objects'].items()}
        cloud.metadata = seed['metadata']
        def object_client(service, **options):
            assert service == 's3'
            assert options['endpoint_url'] == 'https://development-only.invalid'
            assert options['aws_access_key_id'] == 'development-only-access'
            assert options['aws_secret_access_key'] == 'development-only-secret'
            assert options['region_name'] == 'us-east-1'
            return cloud
        boto3.client = object_client
        def remote_state():
            return dict(corrupt=cloud.corrupt,
                objects={key: base64.b64encode(raw).decode() for key, raw in cloud.objects.items()},
                metadata=cloud.metadata, calls=cloud.calls)
        globals()['_installed_fixture_remote'] = remote_state
""".replace('FAKE_CLOUD', repr(str(root / 'python/historical_generation_fake_cloud.py'))).replace(
            'CLOUD_STATE', repr(str(root / 'cloud-fixture.json')))
        boot = boot.replace(needle, transport + needle)
        emit = """        receipt = dict(receipt, _fixture_remote=dict(corrupt=cloud.corrupt,
            objects={key: base64.b64encode(raw).decode() for key, raw in cloud.objects.items()},
            metadata=cloud.metadata, calls=cloud.calls))
"""
        boot = boot.replace('        print(json.dumps(receipt, sort_keys=True, separators=',
                            emit + '        print(json.dumps(receipt, sort_keys=True, separators=')
        # Retain the actual fake-transport calls after a real worker refusal,
        # while preserving the shipped typed KEEP code and nonzero exit. This
        # never supplies preservation, removal, completion or provider receipts.
        refusal = "dict(status='kept', code=code, error_type=type(error).__name__)"
        assert boot.count(refusal) == 1
        boot = boot.replace(refusal, refusal[:-1] + ", **({'_fixture_remote': "
            "_installed_fixture_remote()} if '_installed_fixture_remote' in globals() else {}))")
    original_require = "def _require(value):\n    if not value:\n        raise ValueError(_ERROR)"
    diagnostic_require = """def _require(value):
    if not value:
        frame = sys._getframe(1)
        diagnosis = dict(function=frame.f_code.co_name, line=frame.f_lineno)
        before, child = frame.f_locals.get('before'), frame.f_locals.get('child')
        if isinstance(before, os.stat_result):
            diagnosis['before_identity'] = _identity(before)
        if type(child) is int:
            try:
                diagnosis['opened_identity'] = _identity(os.fstat(child))
            except OSError as error:
                diagnosis['opened_errno'] = error.errno
        globals()['_fixture_entry_failure'] = diagnosis
        raise ValueError(_ERROR)"""
    assert boot.count(original_require) == 1
    boot = boot.replace(original_require, diagnostic_require)
    refusal = "dict(status='kept', code=code, error_type=type(error).__name__"
    assert boot.count(refusal) == 1
    boot = boot.replace(refusal, refusal + ", _fixture_entry_failure=globals().get('_fixture_entry_failure')")
    _write(installed / 'historical-generation-entry.py', boot.encode(), 0o644)
    _write(entry, ('#!/bin/sh\nexec /usr/bin/python3 -I -S '
                   + str(installed / 'historical-generation-entry.py') + ' "$@"\n').encode(), 0o755)
    from blueprint_pipeline import control_plane_lane_historical_dispatch as dispatch
    dispatch._ACTION_EXECUTABLE = str(entry)


def _approve_unlogged_fixture(root, config, entry, restore, target, journals, *, short_expiry=False):
    """Actual tiny owner decision; original restore principal cannot DELETE."""
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_authority import (
        observe_historical_restore_reconciliation, approve_historical_restore_reconciliation)
    action_id = restore['action_id']
    def unchanged():
        return {path.relative_to(target).as_posix(): (
            path.stat().st_dev, path.stat().st_ino, path.stat().st_mode, path.stat().st_uid,
            path.stat().st_gid, path.stat().st_nlink, path.stat().st_size,
            path.stat().st_mtime_ns, path.stat().st_ctime_ns,
            path.read_bytes() if path.is_file() else None) for path in (target, *target.rglob('*'))}
    original = unchanged()
    before = {path.name: path.read_bytes() for path in (journals / action_id).iterdir()}
    refused = _launch_worker(entry, action_id, target, journals, restore=True, expected='failed')
    assert refused['code'] == 'historical_generation_restore_reconciliation_approval_missing', refused
    assert unchanged() == original
    assert {path.name: path.read_bytes() for path in (journals / action_id).iterdir()} == before
    packet = observe_historical_restore_reconciliation(installed_config_path=config, action_id=action_id, now=time.time())
    assert packet['execution_authorized'] is False
    assert packet['original_intent_bytes'] == dict(sha256='sha256:' + hashlib.sha256(before['e-00000.json']).hexdigest(),
                                                   size_bytes=len(before['e-00000.json']))
    options = dict(installed_config_path=config, action_id=action_id,
        ack_packet_digest=packet['packet_digest'], principal='operator', owner='owner',
        discard_unfinished_row=True, no_future_writers=True, no_future_readers=True,
        expires_at_epoch=min(time.time() + (12 if short_expiry else 300), restore['expires_at_epoch']), now=time.time())
    store = root / 'state/requests/historical-generation-actions'
    records = {path.name: path.read_bytes() for path in store.iterdir()}
    for changes in (dict(principal='restore-operator'), dict(owner='different-owner'),
        dict(ack_packet_digest='sha256:' + 'f' * 64), dict(no_future_writers=False),
        dict(expires_at_epoch=restore['expires_at_epoch'] + 1)):
        try:
            approve_historical_restore_reconciliation(**(options | changes))
        except ValueError:
            pass
        else:
            raise AssertionError('unfinished discard accepted without exact current owner DELETE')
        assert {path.name: path.read_bytes() for path in store.iterdir()} == records
        assert unchanged() == original
    approved = approve_historical_restore_reconciliation(**options)
    assert approved['packet'] == packet and approved['execution_authorized'] is False
    assert approved['discard_unfinished_row_approved'] is True
    assert unchanged() == original
    assert {path.name: path.read_bytes() for path in (journals / action_id).iterdir()} == before
    return approved


def _approve_fresh_discard_fixture(root, config, entry, restore, target, journals, previous, *, short_expiry=False):
    """Explicit fixture owner issues a NEW DELETE grant after real old expiry."""
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_authority import (
        observe_historical_restore_reconciliation, approve_historical_restore_reconciliation)
    store, directory = root / 'state/requests/historical-generation-actions', journals / restore['action_id']
    retained = {path.name: path.read_bytes() for path in store.iterdir()}
    journal = {path.name: path.read_bytes() for path in directory.iterdir()}
    def namespace():
        return {path.relative_to(target).as_posix(): (
            tuple(getattr(path.stat(), name) for name in ('st_dev', 'st_ino', 'st_mode', 'st_uid', 'st_gid',
                'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')),
            path.read_bytes() if path.is_file() else None) for path in (target, *target.rglob('*'))}
    original = namespace()
    deadline = time.monotonic() + 15
    while time.time() <= previous['expires_at_epoch']:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert time.time() < restore['expires_at_epoch']
    refused = _launch_worker(entry, restore['action_id'], target, journals, restore=True, expected='failed')
    assert refused['code'] in ('historical_generation_restore_reconciliation_approval_invalid',
                              'historical_generation_restore_absent_observation_approval_missing'), refused
    assert namespace() == original and {path.name: path.read_bytes() for path in directory.iterdir()} == journal
    assert {path.name: path.read_bytes() for path in store.iterdir()} == retained
    packet = observe_historical_restore_reconciliation(installed_config_path=config,
        action_id=restore['action_id'], now=time.time())
    previous_raw = retained[previous['decision_id'] + '.json']
    assert packet['attempt'] == previous['attempt'] + 1
    assert packet['prior_decisions'][-1] == dict(decision_id=previous['decision_id'],
        decision=dict(sha256='sha256:' + hashlib.sha256(previous_raw).hexdigest(), size_bytes=len(previous_raw)))
    assert packet['scope']['remove_member']['path'] in original
    assert packet['original_expires_at_epoch'] == restore['expires_at_epoch']
    head = json.loads(journal[max(name for name in journal if name.startswith('e-'))])
    pending = head['body'].get('phase') in ('reconcile_intent', 'reconcile_delete_resume')
    assert (packet['resume_from'] is not None) == pending
    if pending:
        assert packet['resume_from']['event_digest'] == head['event_digest']
    options = dict(installed_config_path=config, action_id=restore['action_id'], ack_packet_digest=packet['packet_digest'],
        principal='operator', owner='owner', discard_unfinished_row=True, no_future_writers=True,
        no_future_readers=True, expires_at_epoch=min(time.time() + (12 if short_expiry else 300),
            restore['expires_at_epoch']), now=time.time())
    for change in (dict(principal='restore-operator'), dict(owner='other'), dict(no_future_readers=False),
        dict(ack_packet_digest='sha256:' + 'f' * 64), dict(expires_at_epoch=restore['expires_at_epoch'] + 1)):
        try:
            approve_historical_restore_reconciliation(**(options | change))
        except ValueError:
            pass
        else:
            raise AssertionError('fresh discard bypassed exact current owner DELETE')
        assert namespace() == original and {path.name: path.read_bytes() for path in directory.iterdir()} == journal
        assert {path.name: path.read_bytes() for path in store.iterdir()} == retained
    approved = approve_historical_restore_reconciliation(**options)
    assert approved['packet'] == packet and approved['decision_id'] != previous['decision_id']
    assert approved['attempt'] == packet['attempt'] and approved['execution_authorized'] is False
    assert namespace() == original and {path.name: path.read_bytes() for path in directory.iterdir()} == journal
    assert all((store / name).read_bytes() == raw for name, raw in retained.items())
    return approved, retained


def _assert_reconciliation_death_boundary(root, restore, target, journals, approval, phase, *, original_approval=None):
    from blueprint_pipeline.control_plane_lane_historical_generation import inventory_historical_generation
    from blueprint_pipeline.control_plane_lane_historical_restore_reconciliation_scope import reconciled_parent
    from blueprint_pipeline import control_plane_lane_historical_authority as authority
    from blueprint_pipeline.control_plane_lane_historical_restore_authority import select_restore
    from blueprint_pipeline.control_plane_lane_historical_journal import HistoricalJournalObservation
    original_approval = original_approval or approval
    operation = authority._Operation(time.time(), time.monotonic)
    with authority._session(root / 'door.json', operation) as (files, config, store):
        approved, raw = store.read(approval['decision_id'], manifest=True)
        assert authority._selector(raw) == approval['observed_manifest']
        value, decision_raw = store.read(approval['decision_id'])
        assert value == approval
        value, original_raw = store.read(original_approval['decision_id'])
        assert value == original_approval
        selected = select_restore(files, config, store, root / 'door.json', restore['action_id'], operation.moment())
        head = HistoricalJournalObservation(files, config, selected, operation).head
    observed = inventory_historical_generation(target, allowed_roots=(root / 'work', root / 'inputs'))
    expected = ('reconcile_delete_resume' if phase == 'resumed_remove' else
                'reconciled' if phase == 'reconcile_consumed' else 'reconcile_intent')
    assert head['kind'] == 'restore_intent' and head['body']['phase'] == expected
    assert head['action_id'] == restore['action_id']
    assert head['body']['decision_id'] == original_approval['decision_id']
    assert head['body']['decision'] == authority._selector(original_raw)
    if phase == 'resumed_remove':
        assert head['body']['delete_resume'] == dict(decision_id=approval['decision_id'],
                                                     decision=authority._selector(decision_raw))
        assert approval['issued_at_epoch'] <= head['observed_at_epoch'] < approval['expires_at_epoch']
    assert head['body']['original_head_event_digest'] == approval['packet']['original_head_event_digest']
    if phase == 'reconcile_intent':
        assert observed == approved  # exact inode, parent, namespace and partial bytes
    else:
        parent = reconciled_parent(approved, observed, approval['packet']['scope'])
        if phase == 'reconcile_consumed':
            assert head['body']['parent_version'] == parent
            assert head['body']['uncertain'] is True and head['body']['credited_removed_allocated_bytes'] == 0
    print(json.dumps(dict(fixture_actual_reconciliation_boundary=phase,
        action_id=restore['action_id'], decision_id=approval['decision_id'], journal_head=head['event_digest'],
        observed_generation_digest=observed['generation_digest'])), flush=True)


def _approve_absent_fixture(root, config, entry, restore, target, journals, old, *, repeat_expiry=False):
    """Real current observation approval after actual unlink and old expiry."""
    from blueprint_pipeline.control_plane_lane_historical_generation import inventory_historical_generation
    from blueprint_pipeline.control_plane_lane_historical_restore_absent_authority import (
        observe_historical_restore_absence, approve_historical_restore_absence)
    action_id = restore['action_id']
    store = root / 'state/requests/historical-generation-actions'
    records = {path.name: path.read_bytes() for path in store.iterdir()}
    before = {path.name: path.read_bytes() for path in (journals / action_id).iterdir()}
    observed = inventory_historical_generation(target, allowed_roots=(root / 'work', root / 'inputs'))
    assert time.time() > old['expires_at_epoch'] and time.time() < restore['expires_at_epoch']
    refused = _launch_worker(entry, action_id, target, journals, restore=True, expected='failed')
    assert refused['code'] == 'historical_generation_restore_absent_observation_approval_missing', refused
    assert inventory_historical_generation(target, allowed_roots=(root / 'work', root / 'inputs')) == observed
    assert {path.name: path.read_bytes() for path in (journals / action_id).iterdir()} == before
    packet = observe_historical_restore_absence(installed_config_path=config, action_id=action_id, now=time.time())
    assert packet['permits_removal'] is packet['execution_authorized'] is False
    assert packet['observation_only'] is True
    options = dict(installed_config_path=config, action_id=action_id, ack_packet_digest=packet['packet_digest'],
        principal='restore-operator', owner='owner', observe_absence_only=True,
        no_future_writers=True, no_future_readers=True,
        expires_at_epoch=min(time.time() + 4, restore['expires_at_epoch']) if repeat_expiry else restore['expires_at_epoch'],
        now=time.time())
    for changes in (dict(principal='unknown'), dict(owner='different-owner'),
        dict(ack_packet_digest='sha256:' + 'f' * 64), dict(observe_absence_only=False),
        dict(no_future_writers=False), dict(expires_at_epoch=restore['expires_at_epoch'] + 1)):
        try:
            approve_historical_restore_absence(**(options | changes))
        except ValueError:
            pass
        else:
            raise AssertionError('absence observation accepted without exact current owner handling')
        assert {path.name: path.read_bytes() for path in store.iterdir()} == records
    approved = approve_historical_restore_absence(**options)
    if repeat_expiry:
        records = {path.name: path.read_bytes() for path in store.iterdir()}
        first, first_raw = approved, (store / (approved['decision_id'] + '.json')).read_bytes()
        deadline = time.monotonic() + 5
        while time.time() <= first['expires_at_epoch']:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        refused = _launch_worker(entry, action_id, target, journals, restore=True, expected='failed')
        assert refused['code'] == 'historical_generation_restore_approval_expired', refused
        assert inventory_historical_generation(target, allowed_roots=(root / 'work', root / 'inputs')) == observed
        assert {path.name: path.read_bytes() for path in (journals / action_id).iterdir()} == before
        packet = observe_historical_restore_absence(installed_config_path=config, action_id=action_id, now=time.time())
        assert packet['attempt'] == 1 and packet['prior_observations'] == [dict(decision_id=first['decision_id'],
            decision=dict(sha256='sha256:' + hashlib.sha256(first_raw).hexdigest(), size_bytes=len(first_raw)))]
        approved = approve_historical_restore_absence(**(options | dict(ack_packet_digest=packet['packet_digest'],
            expires_at_epoch=restore['expires_at_epoch'], now=time.time())))
        assert approved['decision_id'] != first['decision_id'] and approved['attempt'] == 1
    assert approved['packet'] == packet and approved['principal'] == 'restore-operator'
    assert approved['execution_authorized'] is False
    assert inventory_historical_generation(target, allowed_roots=(root / 'work', root / 'inputs')) == observed
    assert {path.name: path.read_bytes() for path in (journals / action_id).iterdir()} == before
    assert all((store / name).read_bytes() == raw for name, raw in records.items())
    return approved, records


def connected_delete(interruption=None, *, action='delete', corrupt=False,
                     restore_interruption='restore_final', installed=False, destination_conflict=False, metadata_probe=False,
                     reconciliation_interruption=None, absent_expiry=False, observation_expiry=False,
                     delete_expiry=False, resume_delete_expiry=False, resume_remove=False, resume_absent_expiry=False,
                     controlled_names=False, deep_tree=False):
    assert sys.platform == 'linux' and os.geteuid() == 0
    assert os.environ.get('BLUEPRINT_DISPOSABLE_LINUX_TEST') == '1'
    assert Path('/proc/1/exe').resolve() == Path('/usr/lib/systemd/systemd')
    assert not absent_expiry or reconciliation_interruption == 'reconcile_remove'
    assert not observation_expiry or absent_expiry
    assert not delete_expiry or (restore_interruption == 'unlogged_member' and not absent_expiry
        and reconciliation_interruption in (None, 'reconcile_intent'))
    assert not resume_delete_expiry or delete_expiry and reconciliation_interruption == 'reconcile_intent'
    assert not resume_remove or delete_expiry and reconciliation_interruption == 'reconcile_intent'
    assert not resume_absent_expiry or resume_remove
    assert not (controlled_names and deep_tree)
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
        if action == 'offload':
            from blueprint_pipeline.task_evaluation_configured_scene_object_store import _ARTIFACT_STORE_FILE_ENV
            directory = root / 'cloud-config'
            directory.mkdir(mode=0o700)
            values = dict(access_key='development-only-access', secret_key='development-only-secret',
                          bucket='development-only', endpoint='https://development-only.invalid', region='us-east-1')
            for role, name in _ARTIFACT_STORE_FILE_ENV.items():
                path = directory / role
                _write(path, values[role].encode() + b'\n')
                environment += '\n' + name + '=' + str(path)
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
        nested = ('/'.join('\x01' * size for size in (255, 255, 255, 254)) if controlled_names else
                  '/'.join(map(str, range(16))) if deep_tree else 'nested')
        (target / nested).mkdir(parents=True, mode=0o700)
        original = {'one.log': b'original diagnostics\n', nested + ('/f' if controlled_names else '/two.log'): b'nested owner bytes\n'}
        foreign = pwd.getpwnam('nobody')
        for relative, raw in original.items():
            _write(target / relative, raw)
        for path in (target, *target.rglob('*')):
            os.chown(path, foreign.pw_uid, foreign.pw_gid)
        if installed:
            # This sibling exists before the real parent-generation packet;
            # adding it after approval would truthfully invalidate that packet.
            _write(root / 'work/adjacent-unselected.log', b'adjacent original bytes\n')
        # Default-off refuses before an ID can confer authority or create state.
        try:
            run_historical_action(installed_config_path=config, action_id='a' * 32, now=time.time())
        except ValueError as error:
            assert str(error).endswith('action_disabled')
        else:
            raise AssertionError('default-off worker executed')
        assert not list(journals.iterdir())
        def gc_tick(now=time.time):
            from blueprint_pipeline.control_plane_storage_gc import RUN_ACK, run_storage_gc
            report = run_storage_gc(content_store_roots=(), derived_roots=(), queue_roots=(),
                pins_root=root / 'pins', apply=True, ack=RUN_ACK, _experiment_config_path=config, now=now)
            return report['historical_generations']
        if installed:
            disabled = gc_tick()
            assert disabled['units_started'] == disabled['removed_bytes'] == disabled['mutations'] == 0
            assert all((target / name).read_bytes() == raw for name, raw in original.items())
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
        _write(entry, ('#!/var/lib/blueprint-native-test-venv/bin/python\nimport sys\n'
            + 'sys.path.insert(0,' + repr(str(root / 'python')) + ')\n'
            + 'from fixture_acceptance import worker_main\nassert len(sys.argv)==2\n'
            + 'worker_main(' + repr(str(root)) + ',sys.argv[1])\n').encode(), 0o755)
        if installed:
            _installed_entry(root, entry)
        if metadata_probe:
            _write(root / 'metadata-boundary-probe', b'non-authorizing metadata byte projections\n')
        if interruption:
            _write(root / 'interrupt-once', interruption.encode())
            death = interruption.startswith('metadata_')
            if death:
                assert _launch_worker_once(entry, action_id, target, journals, process_death=True) is None
            else:
                interrupted = _launch_worker(entry, action_id, target, journals, expected='failed')
                assert interrupted['code'] == 'fixture_interrupted_after_' + interruption
            expected_paths = set(original) - ({'nested/two.log'} if interruption in ('removed', 'unlink') else set())
            assert all((target / path).read_bytes() == original[path] for path in expected_paths)
            if interruption in ('removed', 'unlink'):
                assert not (target / 'nested/two.log').exists()
            if not death:
                assert target.stat().st_uid == 0
            initial = [path.read_bytes() for path in sorted((journals / action_id).glob('e-*.json'))]
            head = json.loads(initial[-1])
            if death:
                assert head['kind'] == ('intent' if interruption == 'metadata_before_link' else 'fence_intent')
                assert all(path.name.startswith('e-') and path.stat().st_nlink == 1
                           for path in (journals / action_id).iterdir())
                assert all((target / path).read_bytes() == raw for path, raw in original.items())
            else:
                assert head['kind'] == {'fenced': 'fenced', 'chown': 'fence_intent',
                                   'removed': 'removed', 'unlink': 'removal_intent'}[interruption]
            (root / 'interrupt-once').unlink()
        if corrupt:
            refused = _launch_worker(entry, action_id, target, journals,
                expected='kept' if installed else 'failed', launch=gc_tick if installed else None)
            assert refused['code'].endswith('archive_preservation_failed'), refused
            assert all((target / path).read_bytes() == raw for path, raw in original.items())
            events = [json.loads(path.read_bytes()) for path in (journals / action_id).glob('e-*.json')]
            assert not any(event['kind'] in ('preservation', 'removal_intent', 'removed', 'final') for event in events)
            remote = json.loads((root / 'cloud-fixture.json').read_bytes())
            assert remote['calls'].count('readback') == 1 and remote['calls'][-1] == 'client_closed'
            return dict(historical_corrupt_offload_keeps_bytes=True)
        receipt = _launch_worker(entry, action_id, target, journals,
                                 launch=gc_tick if installed else None)
        assert receipt['action'] == action
        assert receipt['removed_files'] == (1 if interruption == 'unlink' else 2) and receipt['removed_directories'] == 1
        assert receipt['logical_bytes'] == sum(map(len, original.values())) - (len(original['nested/two.log']) if interruption == 'unlink' else 0)
        assert receipt['uncertain_removed_allocated_bytes'] == 0
        if interruption:
            assert receipt['uncertain_removed_members'] == int(interruption == 'unlink')
        assert receipt['root_directory_retained'] is True and not list(target.iterdir())
        assert target.stat().st_uid == 0 and target.stat().st_mode & 0o777 == 0o700
        events = [json.loads(path.read_bytes()) for path in sorted((journals / action_id).glob('e-*.json'))]
        assert events[-1]['kind'] == 'final' and events[-1]['body'] == _durable_receipt(receipt)
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
        if installed:
            # If the old path still launches a unit, the helper drains that
            # actual fixture unit before this RED assertion and cleanup.
            repeated = _launch_worker(entry, action_id, target, journals, launch=gc_tick)
            assert repeated['status'] == 'completed' and repeated.get('observation_only') is True
            assert repeated['execution_authorized'] is repeated['action_unit_started'] is False
            # Read-only clock-model check: past facts are still observable,
            # with no resumed worker or claim of fresh action permission.
            later = gc_tick(now=lambda: time.time() + 1000)
            assert later['units_started'] == later['removed_bytes'] == later['mutations'] == 0, later
            assert later['outcomes'] == [dict(repeated, observed_at_epoch=later['outcomes'][0]['observed_at_epoch'])]
        else:
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
        if action == 'offload':
            from blueprint_pipeline.control_plane_lane_historical_restore_authority import approve_historical_restore
            # A distinct currently mapped owner principal can approve restore;
            # the old policy selector is a past fact, never fresh permission.
            policy['principals'].append(dict(principal='restore-operator', owners=['owner'],
                allowed_actions=['register', 'offload'], max_consent_seconds=900))
            _write(root / 'policy.json', _encoded(policy))
            restore = approve_historical_restore(installed_config_path=config,
                offload_action_id=action_id, ack_final_event_digest=events[-1]['event_digest'],
                principal='restore-operator', owner='owner', expires_at_epoch=time.time() + 600,
                now=time.time())
            assert restore['action_id'] != action_id and restore['action'] == 'restore'
            assert restore['execution_authorized'] is False and restore['restore_approved'] is True
            assert restore['preservation_event_digest'] == receipt['preservation_event_digest']
            assert restore['archive'] == receipt['preservation']
            assert restore['tombstone_version'] == receipt['tombstone_version']
            assert not list(target.iterdir()) and target.stat().st_uid == 0
            assert {path.name: path.read_bytes() for path in (journals / action_id).iterdir()} == before
            store = root / 'state/requests/historical-generation-actions'
            after_approval = {path.name: path.read_bytes() for path in store.iterdir()}
            for changes in ({'owner': 'another-owner'}, {'ack_final_event_digest': 'sha256:' + 'f' * 64}):
                try:
                    approve_historical_restore(**(dict(installed_config_path=config,
                        offload_action_id=action_id, ack_final_event_digest=events[-1]['event_digest'],
                        principal='restore-operator', owner='owner', expires_at_epoch=time.time() + 600,
                        now=time.time()) | changes))
                except ValueError:
                    pass
                else:
                    raise AssertionError('unapproved restore decision published')
                assert {path.name: path.read_bytes() for path in store.iterdir()} == after_approval
            if destination_conflict:
                # A real post-approval destination write must not be overwritten
                # or incorporated into the approved empty-tombstone generation.
                sentinel = target / 'one.log'
                _write(sentinel, b'keep unapproved destination bytes')
                refused = _launch_worker(entry, restore['action_id'], target, journals, restore=True,
                    expected='kept' if installed else 'failed', launch=gc_tick if installed else None)
                assert refused['code'].endswith('restore_tombstone_changed'), refused
                assert sentinel.read_bytes() == b'keep unapproved destination bytes'
                assert set(path.name for path in target.iterdir()) == {'one.log'}
                assert target.stat().st_uid == 0 and stat.S_IMODE(target.stat().st_mode) == 0o700
                refused_events = [json.loads(path.read_bytes()) for path in
                                  (journals / restore['action_id']).glob('e-*.json')]
                assert [event['kind'] for event in refused_events] == ['intent']
                assert {path.name: path.read_bytes() for path in store.iterdir()} == after_approval
                assert {path.name: path.read_bytes() for path in (journals / action_id).iterdir()} == before
                return dict(historical_restore_existing_destination_kept=True)
            if restore_interruption is not None:
                _write(root / 'interrupt-once', restore_interruption.encode())
                if restore_interruption.startswith('unlogged_'):
                    _launch_death_worker(entry, restore['action_id'], target, journals)
                else:
                    interrupted = _launch_worker(entry, restore['action_id'], target, journals,
                                                 restore=True, expected='failed')
                    if interrupted['code'] == 'historical_generation_restore_reconciliation_approval_missing':
                        # A genuine scan refusal may have left a different
                        # unfinished syscall before the requested boundary.
                        # Retain it, obtain a real exact tiny-owner DELETE
                        # decision, then reach the SAME original fault/intent.
                        initial = (journals / restore['action_id'] / 'e-00000.json').read_bytes()
                        print(json.dumps(dict(fixture_unfinished_restore_refusal=interrupted,
                            action_id=restore['action_id'], original_intent_sha256=hashlib.sha256(initial).hexdigest())), flush=True)
                        _approve_unlogged_fixture(root, config, entry, restore, target, journals)
                        interrupted = _launch_worker(entry, restore['action_id'], target, journals,
                                                     restore=True, expected='failed')
                    assert interrupted['code'] == 'fixture_interrupted_after_' + restore_interruption, interrupted
                assert target.stat().st_uid == 0 and stat.S_IMODE(target.stat().st_mode) == 0o700
                if restore_interruption.startswith('unlogged_'):
                    stage = target / ('.historical-restore-' + restore['action_id'])
                    assert stage.is_dir() and stage.stat().st_uid == 0
                    if restore_interruption == 'unlogged_directory':
                        assert not list(stage.iterdir())
                    else:
                        assert (stage / 'nested/two.log').read_bytes() == original['nested/two.log'][:1]
                        retained_known_directories = {name: (path.stat().st_dev, path.stat().st_ino)
                            for name, path in (('', stage), ('nested', stage / 'nested'))}
                elif restore_interruption == 'unwritten_stage':
                    assert not list(target.iterdir())
                elif restore_interruption in ('stage_complete', 'restore_directory', 'restore_member'):
                    stage = target / ('.historical-restore-' + restore['action_id'])
                    assert set(path.name for path in target.iterdir()) == {stage.name}
                    assert stage.stat().st_uid == 0 and stat.S_IMODE(stage.stat().st_mode) == 0o700
                    if restore_interruption == 'stage_complete':
                        assert all((stage / name).read_bytes() == value for name, value in original.items())
                    elif restore_interruption == 'restore_directory':
                        assert not list(stage.iterdir())
                    else:
                        assert (stage / 'nested/two.log').read_bytes() == original['nested/two.log']
                        assert not (stage / 'one.log').exists()
                    retained_stage_root = (stage.stat().st_dev, stage.stat().st_ino)
                    retained_stage_inodes = {path.relative_to(stage).as_posix():
                        (path.stat().st_dev, path.stat().st_ino) for path in stage.rglob('*')}
                elif restore_interruption in ('publish_intent', 'publish_rename', 'publish_observed',
                                               'stage_remove_intent', 'stage_remove_effect'):
                    stage = target / ('.historical-restore-' + restore['action_id'])
                    moved = restore_interruption != 'publish_intent'
                    removal = restore_interruption.startswith('stage_remove_')
                    expected_names = {'nested', 'one.log'} if removal else ({'nested'} if moved else set())
                    if restore_interruption != 'stage_remove_effect':
                        expected_names.add(stage.name)
                    assert {path.name for path in target.iterdir()} == expected_names
                    assert all((target / name if removal or moved and name.startswith('nested/') else stage / name).read_bytes()
                               == value for name, value in original.items())
                    retained_stage_inodes = {name: ((target / name if removal or moved and name.startswith('nested/') else stage / name).stat().st_dev,
                        (target / name if removal or moved and name.startswith('nested/') else stage / name).stat().st_ino) for name in original}
                else:
                    assert all((target / name).read_bytes() == value for name, value in original.items())
                if restore_interruption == 'restore_member_chown':
                    # Descendants transition before their parent directories.
                    # Pin the actual partial kernel effect, not an assumed
                    # top-level iteration order or a synthetic permission row.
                    first = (target / 'nested/two.log').stat()
                    assert (first.st_uid, first.st_gid) == (foreign.pw_uid, foreign.pw_gid)
                    assert (target / 'one.log').stat().st_uid == 0
                    assert (target / 'nested').stat().st_uid == 0
                interrupted_prefix = {path.name: path.read_bytes() for path in
                                      (journals / restore['action_id']).iterdir()}
                interrupted_events = [json.loads(raw) for name, raw in interrupted_prefix.items()
                                      if name.startswith('e-')]
                if restore_interruption in ('restore_directory', 'restore_member'):
                    born_roots = [event['body'] for event in interrupted_events
                                  if event['kind'] == 'restore_directory' and event['body']['path'] == '']
                    assert len(born_roots) == 1 and tuple(born_roots[0]['version'][:2]) == retained_stage_root
                if restore_interruption == 'unlogged_member':
                    born = {event['body']['path']: tuple(event['body']['version'][:2])
                            for event in interrupted_events if event['kind'] == 'restore_directory'}
                    assert born == retained_known_directories
                if restore_interruption == 'restore_member_chown':
                    intent = max(interrupted_events, key=lambda event: event['sequence'])
                    assert intent['kind'] == 'restore_intent'
                    assert intent['body']['phase'] == 'owner_rights'
                    assert intent['body']['path'] == 'nested/two.log'
                    assert intent['body']['version'][3:5] == [0, 0]
                assert sum(event['kind'] == 'restore_final' for event in interrupted_events) == int(
                    restore_interruption in ('restore_final', 'access_intent'))
                assert ('restore.snapshot.json' in interrupted_prefix) == (
                    restore_interruption not in ('stage_removed', 'unwritten_stage', 'restore_member_chown', 'stage_complete', 'restore_directory', 'restore_member',
                                                 'publish_intent', 'publish_rename', 'publish_observed', 'stage_remove_intent', 'stage_remove_effect',
                                                 'unlogged_directory', 'unlogged_member'))
                assert not any(event['kind'] == 'access_reopened' for event in interrupted_events)
                (root / 'interrupt-once').unlink()
                if restore_interruption.startswith('unlogged_'):
                    reconciliation = _approve_unlogged_fixture(root, config, entry, restore, target, journals,
                        short_expiry=reconciliation_interruption == 'reconcile_consumed' or absent_expiry or delete_expiry)
                    if reconciliation_interruption:
                        _write(root / 'interrupt-once', reconciliation_interruption.encode())
                        assert _launch_worker_once(entry, restore['action_id'], target, journals,
                                                  restore=True, process_death=True) is None
                        (root / 'interrupt-once').unlink()
                        _assert_reconciliation_death_boundary(root, restore, target, journals,
                            reconciliation, reconciliation_interruption)
                        if reconciliation_interruption == 'reconcile_consumed' or absent_expiry:
                            # Actual elapsed time, no injected clock or renewed
                            # decision. Only consumed DELETE may be historical;
                            # the original restore expiry stays current.
                            deadline = time.monotonic() + 15
                            while time.time() <= reconciliation['expires_at_epoch']:
                                assert time.monotonic() < deadline
                                time.sleep(0.05)
                            assert time.time() < restore['expires_at_epoch']
                        if absent_expiry:
                            absence, retained_discard_records = _approve_absent_fixture(root, config, entry,
                                restore, target, journals, reconciliation, repeat_expiry=observation_expiry)
                if delete_expiry:
                    initial_discard = reconciliation
                    reconciliation, retained_discard_records = _approve_fresh_discard_fixture(root, config, entry,
                        restore, target, journals, reconciliation, short_expiry=resume_delete_expiry or resume_absent_expiry)
                    if resume_delete_expiry:
                        _write(root / 'interrupt-once', b'reconcile_delete_resume')
                        assert _launch_worker_once(entry, restore['action_id'], target, journals,
                            restore=True, process_death=True) is None
                        (root / 'interrupt-once').unlink()
                        resume_raw = max((journals / restore['action_id']).glob('e-*.json')).read_bytes()
                        resume_event = json.loads(resume_raw)
                        assert resume_event['body']['phase'] == 'reconcile_delete_resume'
                        raw = (root / 'state/requests/historical-generation-actions' / (reconciliation['decision_id'] + '.json')).read_bytes()
                        assert resume_event['body']['delete_resume'] == dict(decision_id=reconciliation['decision_id'],
                            decision=dict(sha256='sha256:' + hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)))
                        assert resume_event['observed_at_epoch'] < reconciliation['expires_at_epoch']
                        row = reconciliation['packet']['scope']['remove_member']
                        assert (target / row['path']).exists() and (target / row['path']).stat().st_ino == row['version'][1]
                        reconciliation, retained_discard_records = _approve_fresh_discard_fixture(root, config, entry,
                            restore, target, journals, reconciliation)
                    if resume_remove:
                        _write(root / 'interrupt-once', b'reconcile_remove')
                        assert _launch_worker_once(entry, restore['action_id'], target, journals,
                            restore=True, process_death=True) is None
                        (root / 'interrupt-once').unlink()
                        _assert_reconciliation_death_boundary(root, restore, target, journals, reconciliation,
                            'resumed_remove', original_approval=initial_discard)
                        if resume_absent_expiry:
                            deadline = time.monotonic() + 15
                            while time.time() <= reconciliation['expires_at_epoch']:
                                assert time.monotonic() < deadline
                                time.sleep(0.05)
                            absence, retained_discard_records = _approve_absent_fixture(root, config, entry,
                                restore, target, journals, reconciliation)
            restored = _launch_worker(entry, restore['action_id'], target, journals, restore=True,
                                      launch=gc_tick if installed else None)
            if restore_interruption is not None:
                recovery = {'restore_final': 'recovered_access',
                            'before_restore_final': 'recovered_before_final',
                            'stage_removed': 'recovered_publication', 'unwritten_stage': 'restarted_unwritten',
                            'access_intent': 'recovered_access', 'restore_member_chown': 'recovered_publication',
                            'stage_complete': 'recovered_stage',
                            'restore_directory': 'recovered_prefix', 'restore_member': 'recovered_prefix',
                            'publish_intent': 'recovered_split', 'publish_rename': 'recovered_split',
                            'publish_observed': 'recovered_split', 'stage_remove_intent': 'recovered_split',
                            'stage_remove_effect': 'recovered_split'}
                recovery.update(unlogged_directory='recovered_prefix', unlogged_member='recovered_prefix')
                if recovery[restore_interruption] in ('recovered_stage', 'recovered_split', 'recovered_prefix'):
                    _assert_boundary_recovery(restored, recovery[restore_interruption],
                        restored.get('_fixture_reference_refusals', []), restore['action_id'],
                        interrupted_prefix['e-00000.json'])
                else:
                    assert restored[recovery[restore_interruption]] is True
                if restore_interruption == 'unwritten_stage':
                    assert restored['restored_files'] == len(original)
                    assert restored['restored_logical_bytes'] == sum(map(len, original.values()))
                elif restore_interruption.startswith('unlogged_'):
                    if restore_interruption == 'unlogged_member':
                        assert ((target / 'nested').stat().st_dev, (target / 'nested').stat().st_ino) \
                            == retained_known_directories['nested']
                elif restore_interruption in ('restore_directory', 'restore_member'):
                    assert all((target / name).stat().st_dev == identity[0]
                               and (target / name).stat().st_ino == identity[1]
                               for name, identity in retained_stage_inodes.items())
                else:
                    assert restored['restored_files'] == restored['restored_logical_bytes'] == 0
                if restore_interruption in ('publish_intent', 'publish_rename', 'publish_observed',
                                           'stage_remove_intent', 'stage_remove_effect'):
                    assert all(((target / name).stat().st_dev, (target / name).stat().st_ino) == identity
                               for name, identity in retained_stage_inodes.items())
                assert all((journals / restore['action_id'] / name).read_bytes() == raw
                           for name, raw in interrupted_prefix.items())
            if restored.get('recovered_prefix') is True:
                assert '_fixture_prior_member_births' in restored
            _assert_restore_increment(restored, original)
            assert restored['action'] == 'restore' and restored['owner_access_reopened'] is True
            if restored.get('idempotent') is True:
                assert restored.get('_fixture_reference_refusals'), restored
                assert 'fresh_disk_reservation' not in restored
            else:
                assert restored['fresh_disk_reservation'] is True
            assert all((target / name).read_bytes() == value for name, value in original.items())
            assert target.stat().st_uid == foreign.pw_uid and target.stat().st_gid == foreign.pw_gid
            restore_events = [json.loads(path.read_bytes()) for path in
                              sorted((journals / restore['action_id']).glob('e-*.json'))]
            kinds = [event['kind'] for event in restore_events]
            if restore_interruption and restore_interruption.startswith('unlogged_'):
                reconciled = [event for event in restore_events if event['kind'] == 'restore_intent'
                              and event['body'].get('phase') == 'reconciled']
                assert len(reconciled) == 1 and reconciled[0]['body']['uncertain'] is True
                assert reconciled[0]['body']['credited_removed_allocated_bytes'] == 0
                if delete_expiry:
                    store = root / 'state/requests/historical-generation-actions'
                    assert all((store / name).read_bytes() == raw for name, raw in retained_discard_records.items())
                    assert reconciled[0]['observed_at_epoch'] > initial_discard['expires_at_epoch']
                    resumes = [event for event in restore_events if event['body'].get('phase') == 'reconcile_delete_resume']
                    assert len(resumes) == (2 if resume_delete_expiry else int(reconciliation_interruption is not None))
                    if resumes and not resume_absent_expiry:
                        raw = (store / (reconciliation['decision_id'] + '.json')).read_bytes()
                        assert reconciled[0]['body']['delete_resume'] == dict(decision_id=reconciliation['decision_id'],
                            decision=dict(sha256='sha256:' + hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)))
                        assert reconciled[0]['previous_event_digest'] == resumes[-1]['event_digest']
                    elif not resumes:
                        assert reconciled[0]['body']['decision_id'] == reconciliation['decision_id']
                if absent_expiry or resume_absent_expiry:
                    raw = (root / 'state/requests/historical-generation-actions' / (absence['decision_id'] + '.json')).read_bytes()
                    assert reconciled[0]['body']['observation_resume'] == dict(decision_id=absence['decision_id'],
                        decision=dict(sha256='sha256:' + hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)))
                    assert reconciled[0]['observed_at_epoch'] > reconciliation['expires_at_epoch']
                    assert all((root / 'state/requests/historical-generation-actions' / name).read_bytes() == raw
                               for name, raw in retained_discard_records.items())
                if restore_interruption == 'unlogged_member':
                    born = {event['body']['path']: tuple(event['body']['version'][:2])
                            for event in restore_events if event['kind'] == 'restore_directory'}
                    assert born == retained_known_directories
            assert kinds.index('restore_final') < kinds.index('access_reopened')
            assert sum(kind == 'restore_member' for kind in kinds) == len(original)
            directories = [event['body'] for event in restore_events if event['kind'] == 'restore_directory']
            assert len({row['path'] for row in directories}) == len(directories)
            if restore_interruption in ('restore_directory', 'restore_member'):
                roots = [row for row in directories if row['path'] == '']
                assert len(roots) == 1 and tuple(roots[0]['version'][:2]) == retained_stage_root
            finals = [event['body'] for event in restore_events if event['kind'] == 'restore_final']
            assert len(finals) == 1
            assert (finals[0]['restored_files'], finals[0]['restored_logical_bytes']) == (
                len(original), sum(map(len, original.values())))
            restore_before = {path.name: path.read_bytes() for path in (journals / restore['action_id']).iterdir()}
            restore_again = _launch_worker(entry, restore['action_id'], target, journals, restore=True,
                                           launch=gc_tick if installed else None)
            if installed:
                assert restore_again.get('observation_only') is True, restore_again
                assert restore_again['execution_authorized'] is restore_again['action_unit_started'] is False
                later = gc_tick(now=lambda: time.time() + 1000)
                assert later['units_started'] == later['removed_bytes'] == later['mutations'] == 0, later
                observed_restore = next(row for row in later['outcomes']
                                        if row['action_id'] == restore['action_id'])
                assert observed_restore == dict(restore_again,
                    observed_at_epoch=observed_restore['observed_at_epoch'])
            assert restore_again['idempotent'] is True and restore_again['owner_access_reopened'] is True
            assert restore_again['restored_files'] == 0 and restore_again['restored_logical_bytes'] == 0
            assert {path.name: path.read_bytes() for path in (journals / restore['action_id']).iterdir()} == restore_before
            assert all((target / name).read_bytes() == value for name, value in original.items())
        # A completed receipt never adopts a rewritten or repopulated tombstone.
        _write(target / 'changed-after-final', b'keep changed bytes')
        changed = _launch_worker(entry, action_id, target, journals,
                                 expected='kept' if installed else 'failed')
        # Offload restore approval above rotated the current policy. The old
        # offload action must refuse that earlier authority gate; all five
        # unchanged-policy delete cases independently exercise tombstone drift.
        refusal = 'owner_consent_policy_changed' if action == 'offload' else 'action_tombstone_changed'
        assert changed['code'].endswith(refusal), changed
        assert (target / 'changed-after-final').read_bytes() == b'keep changed bytes'
        assert {path.name: path.read_bytes() for path in (journals / action_id).iterdir()} == before
        if action == 'offload':
            return dict(historical_offload_full_readback=True, historical_restore_decision_bound=True,
                        historical_restored_bytes_and_access=True)
        return dict(actual_owner_approved_delete=True, original_member_journal=True,
                    historical_delete_idempotent=True)


CONNECTED_CASES = (
    ('delete', dict(interruption=None)),
    ('escaped_path_restore', dict(action='offload', controlled_names=True)),
    ('maximum_depth_restore', dict(action='offload', deep_tree=True, restore_interruption='stage_complete')),
    ('metadata_bounds', dict(metadata_probe=True)),
    ('metadata_before_link', dict(interruption='metadata_before_link')),
    ('metadata_after_link', dict(interruption='metadata_after_link')),
    ('metadata_before_parent_fsync', dict(interruption='metadata_before_parent_fsync')),
    ('fenced', dict(interruption='fenced')),
    ('chown', dict(interruption='chown')),
    ('removed', dict(interruption='removed')),
    ('unlink', dict(interruption='unlink')),
    ('offload', dict(action='offload')),
    ('before_restore_final', dict(action='offload', restore_interruption='before_restore_final')),
    ('unwritten_stage', dict(action='offload', restore_interruption='unwritten_stage')),
    ('access_intent', dict(action='offload', restore_interruption='access_intent')),
    ('stage_removed', dict(action='offload', restore_interruption='stage_removed')),
    ('stage_complete', dict(action='offload', restore_interruption='stage_complete')),
    ('restore_directory', dict(action='offload', restore_interruption='restore_directory')),
    ('restore_member', dict(action='offload', restore_interruption='restore_member')),
    ('unlogged_directory', dict(action='offload', restore_interruption='unlogged_directory')),
    ('unlogged_member', dict(action='offload', restore_interruption='unlogged_member')),
    ('reconcile_intent', dict(action='offload', restore_interruption='unlogged_member', reconciliation_interruption='reconcile_intent')),
    ('reconcile_remove', dict(action='offload', restore_interruption='unlogged_member', reconciliation_interruption='reconcile_remove')),
    ('reconcile_sync', dict(action='offload', restore_interruption='unlogged_member', reconciliation_interruption='reconcile_sync')),
    ('reconcile_consumed', dict(action='offload', restore_interruption='unlogged_member', reconciliation_interruption='reconcile_consumed')),
    ('reconcile_delete_expiry', dict(action='offload', restore_interruption='unlogged_member', reconciliation_interruption=None, delete_expiry=True, resume_delete_expiry=False)),
    ('reconcile_pending_delete_expiry', dict(action='offload', restore_interruption='unlogged_member', reconciliation_interruption='reconcile_intent', delete_expiry=True, resume_delete_expiry=False)),
    ('reconcile_resume_delete_expiry', dict(action='offload', restore_interruption='unlogged_member', reconciliation_interruption='reconcile_intent', delete_expiry=True, resume_delete_expiry=True)),
    ('reconcile_resume_remove', dict(action='offload', restore_interruption='unlogged_member', reconciliation_interruption='reconcile_intent', delete_expiry=True, resume_remove=True)),
    ('reconcile_resume_absent_expiry', dict(action='offload', restore_interruption='unlogged_member', reconciliation_interruption='reconcile_intent', delete_expiry=True, resume_remove=True, resume_absent_expiry=True)),
    ('reconcile_absent_expiry', dict(action='offload', restore_interruption='unlogged_member', reconciliation_interruption='reconcile_remove', absent_expiry=True)),
    ('reconcile_observation_expiry', dict(action='offload', restore_interruption='unlogged_member', reconciliation_interruption='reconcile_remove', absent_expiry=True, observation_expiry=True)),
    ('publish_intent', dict(action='offload', restore_interruption='publish_intent')),
    ('publish_rename', dict(action='offload', restore_interruption='publish_rename')),
    ('publish_observed', dict(action='offload', restore_interruption='publish_observed')),
    ('stage_remove_intent', dict(action='offload', restore_interruption='stage_remove_intent')),
    ('stage_remove_effect', dict(action='offload', restore_interruption='stage_remove_effect')),
    ('corrupt', dict(action='offload', corrupt=True)),
)


def run_connected_case(case_id):
    options = dict(CONNECTED_CASES)[case_id]
    started = time.monotonic()
    print(json.dumps({'fixture_case_started': options}), flush=True)
    result = connected_delete(**options)
    print(json.dumps({'fixture_case_completed': options,
                      'elapsed_seconds': time.monotonic() - started}), flush=True)
    return result


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
    print(json.dumps(connected_delete(), sort_keys=True))
