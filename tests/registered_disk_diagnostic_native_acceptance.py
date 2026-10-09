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
import math
import os
import pwd
import re
import select
import sys
import time
from contextlib import contextmanager, ExitStack
from pathlib import Path

from tests.test_registered_feature_linux import encoded, install_protected_feature, _run_shipped_gc_sandbox



_PID_OPEN_EVIDENCE = None


class PidOpenEvidence:
    """Test-only retained syscall observations; tokens never establish PID identity."""
    def __init__(self):
        self.children, self.tokens, self.observations = {}, {}, []
        self.truncated = False

    def child_started(self, pid, role):
        if type(pid) is int and pid > 0 and role in {'foreign_report_reader', 'ordinary_access_probe'}:
            if len(self.children) < 16 or pid in self.children:
                self.children[pid] = dict(role=role, reaped=False)
            else:
                self.truncated = True

    def child_reaped(self, pid):
        if pid in self.children:
            self.children[pid]['reaped'] = True

    def record(self, pid, attempt, scan):
        if len(self.observations) >= 8:
            self.truncated = True
            return
        elapsed = scan.last - scan.started
        if not (type(pid) is str and pid.isdecimal() and len(pid) <= 10
                and type(attempt) is int and 0 <= attempt < 3
                and type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed < 5):
            self.truncated = True
            return
        number = int(pid)
        token = self.tokens.setdefault(number, len(self.tokens) + 1)
        child = self.children.get(number)
        owned = child is not None and child['reaped'] is False
        self.observations.append(dict(pid_token=token, attempt=attempt + 1,
            elapsed_ms=round(elapsed * 1000, 3), ownership='fixture_owned_unreaped_child' if owned else 'UNKNOWN',
            child_role=child['role'] if owned else None,
            previous_fixture_child_reaped=child is not None and child['reaped'] is True))

    def packet(self):
        return dict(observation_phase='caught_pid_open_failure', observations=list(self.observations),
            truncated=self.truncated, pid_tokens_are_numeric_slots_not_start_identities=True)


@contextmanager
def observe_pid_opens(evidence):
    """Delegate unchanged syscalls; observe only the exact scanner's caught ENOENT."""
    from blueprint_pipeline import control_plane_lane_historical_processes as processes
    original = os.open
    scanner_code = processes.refuse_historical_process_references.__code__
    def observed(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except FileNotFoundError as error:
            try:
                frame = sys._getframe(1)
                local = frame.f_locals
                if (error.errno == 2 and frame.f_code is scanner_code and args
                        and args[0] == local.get('pid') and kwargs.get('dir_fd') == local.get('proc')):
                    evidence.record(local['pid'], local['_attempt'], local['scan'])
            except Exception:
                evidence.truncated = True
            raise
    os.open = observed
    try:
        yield
    finally:
        os.open = original


def _remember_fixture_child(pid, role=None, *, reaped=False):
    try:
        if _PID_OPEN_EVIDENCE is not None:
            if reaped:
                _PID_OPEN_EVIDENCE.child_reaped(pid)
            else:
                _PID_OPEN_EVIDENCE.child_started(pid, role)
    except Exception:
        pass  # Diagnostic bookkeeping never controls child lifecycle or acceptance.


def _census_reason_evidence(error):
    """Bounded retained fixed counts only; no host read or process selectors."""
    from blueprint_pipeline.control_plane_lane_historical_processes import HistoricalProcessError
    if type(error) is not HistoricalProcessError or error.args != ('historical_generation_process_unknown',):
        return None
    rows = getattr(error, 'census_attempt_reasons', None)
    keys = {'pid_open_disappeared', 'descriptor_membership_changed',
            'corroborated_process_exit', 'pid_census_changed'}
    if type(rows) is not tuple or len(rows) != 3 or any(
        type(row) is not dict or set(row) != keys
        or any(type(value) is not int or not 0 <= value <= 4096 for value in row.values())
        or row['pid_census_changed'] not in (0, 1)
        or not 1 <= sum(row.values()) <= 4096 for row in rows
    ):
        return None
    return [dict(row) for row in rows]


def failure_evidence(error):
    """Read retained contexts only after failure; never observe or alter the host.

    No arbitrary exception text, paths, frame locals or input bytes are emitted.
    Suppressed contexts survive ``from None``. This is post-unwind evidence,
    never a reconstruction of the earlier ownership or reference census.
    """
    value = dict(observation_phase='post_unwind', exceptions=[], truncated=False)
    pending, seen = [error], set()
    while pending and len(value['exceptions']) < 8:
        selected = pending.pop()
        if id(selected) in seen:
            value['truncated'] = True
            continue
        seen.add(id(selected))
        row = {'class': type(selected).__name__[:64],
               'module': type(selected).__module__[:96], 'frames': []}
        if selected.args and type(selected.args[0]) is str and re.fullmatch(
                r'(?:experiment_|historical_generation_|reference_|owner_target_)[a-z0-9_]{1,80}',
                selected.args[0]):
            row['code'] = selected.args[0]
        try:
            reasons = _census_reason_evidence(selected)
            if reasons is not None:
                row['census_attempt_reasons'] = reasons
        except Exception:
            pass  # Evidence projection cannot replace the original refusal.
        if isinstance(selected, OSError) and type(selected.errno) is int:
            row['errno'] = selected.errno
        traceback = selected.__traceback__
        while traceback is not None and len(row['frames']) < 16:
            code = traceback.tb_frame.f_code
            row['frames'].append(dict(source=Path(code.co_filename).name[:96],
                                      function=code.co_name[:64], line=traceback.tb_lineno))
            traceback = traceback.tb_next
        value['truncated'] |= traceback is not None
        value['exceptions'].append(row)
        # Retain both causal links, including context suppressed for display.
        if selected.__context__ is not None:
            pending.append(selected.__context__)
        if selected.__cause__ is not None and selected.__cause__ is not selected.__context__:
            pending.append(selected.__cause__)
    value['truncated'] |= bool(pending)
    while len(json.dumps(value).encode()) > 8192:
        value['truncated'] = True
        row = next((row for row in reversed(value['exceptions']) if row['frames']), None)
        if row is not None:
            row['frames'].pop()
        else:
            value['exceptions'].pop()
    return value


def run_observed(root):
    """Execute unchanged acceptance and preserve the same original failure."""
    global _PID_OPEN_EVIDENCE
    previous = _PID_OPEN_EVIDENCE
    evidence = PidOpenEvidence()
    _PID_OPEN_EVIDENCE = evidence
    try:
        with observe_pid_opens(evidence):
            return run(root)
    except Exception as error:
        try:
            packet = failure_evidence(error)
            packet['pid_open_evidence'] = evidence.packet()
            observed = packet['pid_open_evidence']
            while len(json.dumps(packet).encode()) > 8192 and observed['observations']:
                observed['observations'].pop()
                observed['truncated'] = True
            if len(json.dumps(packet).encode()) > 8192:
                # Preserve the original exception projection when even the
                # empty observer envelope cannot fit; its existing flag records
                # omitted evidence without adding another unbounded field.
                del packet['pid_open_evidence']
                packet['truncated'] = True
            print(json.dumps(packet, sort_keys=True), file=sys.stderr)
        except Exception:
            pass  # An evidence write cannot replace the original refusal.
        raise
    finally:
        _PID_OPEN_EVIDENCE = previous


def observe_reference_failures(reference_type, evidence):
    """Retain an actual caught UNKNOWN before GC consumes it; never clear it.

    Wrapping methods preserves the concrete class and original calls. Guard and
    constructor may observe the same exception at two explicitly named catch
    boundaries. No host access, clock, budget or reference predicate is added.
    """
    from blueprint_pipeline import control_plane_lane_historical_processes as processes
    from blueprint_pipeline.control_plane_lane_disk_diagnostic_references import OwnerTargetVersionError
    from tests.historical_generation_native_acceptance import _failed_process_view

    inspector_code = processes._inspect_process.__code__

    def record(error, boundary):
        if len(evidence['records']) >= 4:
            evidence['truncated'] = True
            return
        selected = failure_evidence(error)
        selected.update(boundary=boundary, observation_phase='caught_unknown_before_gc_consumption',
                        final_process_identity_verified=False, same_failed_views=[])
        pending, seen = [error], set()
        while pending and len(seen) < 8:
            current = pending.pop()
            if id(current) in seen:
                selected['truncated'] = True
                continue
            seen.add(id(current))
            trace, frames = current.__traceback__, 0
            while trace is not None and frames < 16:
                # A same-named foreign frame is not the installed inspector.
                if trace.tb_frame.f_code is inspector_code:
                    view = _failed_process_view(trace.tb_frame.f_locals)
                    if view:
                        if len(selected['same_failed_views']) < 4:
                            selected['same_failed_views'].append(view)
                        else:
                            selected['truncated'] = True
                trace, frames = trace.tb_next, frames + 1
            selected['truncated'] |= trace is not None
            if current.__context__ is not None:
                pending.append(current.__context__)
            if current.__cause__ is not None and current.__cause__ is not current.__context__:
                pending.append(current.__cause__)
        selected['truncated'] |= bool(pending)
        candidate = {'records': [*evidence['records'], selected], 'truncated': evidence['truncated']}
        if len(json.dumps(candidate).encode()) > 16384:
            evidence['truncated'] = True
            return
        evidence['records'].append(selected)

    def wrap(original, boundary):
        @functools.wraps(original)
        def observed(self, *args, **kwargs):
            try:
                return original(self, *args, **kwargs)
            except Exception as error:
                if type(error) is OwnerTargetVersionError and error.code == 'experiment_diagnostic_references_unknown':
                    try:
                        record(error, boundary)
                    except Exception:
                        try:
                            evidence['truncated'] = True
                        except Exception:
                            pass  # Evidence failure cannot replace the original refusal.
                raise
        return observed

    originals = reference_type.__init__, reference_type.guard
    reference_type.__init__ = wrap(originals[0], 'constructor')
    reference_type.guard = wrap(originals[1], 'guard')
    return originals


def _ordinary_denied(target, value, account):
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    read, write = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read)
        try:
            os.initgroups('blueprint', account.pw_gid)
            os.setgid(account.pw_gid)
            os.setuid(account.pw_uid)
            probes = [(path, os.O_RDONLY) for path in
                      (target, value['config'], value['policy'], target / 'disk-capacity-report.v1.json')]
            probes += [(target / 'disk-capacity-report.v1.json', os.O_WRONLY),
                       (target / 'late-writer.payload', os.O_WRONLY | os.O_CREAT | os.O_EXCL)]
            for path, flags in probes:
                try:
                    fd = os.open(path, flags | os.O_NOFOLLOW, 0o600)
                except PermissionError:
                    continue
                os.close(fd)
                raise AssertionError('ordinary UID accessed root-only diagnostic')
            os.write(write, b'root-only')
        finally:
            os.close(write)
            os._exit(0)
    _remember_fixture_child(child, 'ordinary_access_probe')
    os.close(write)
    try:
        assert select.select([read], [], [], 10)[0]
        assert os.read(read, 128) == b'root-only'
    finally:
        os.close(read)
        os.waitpid(child, 0)
        _remember_fixture_child(child, reaped=True)
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before


@contextmanager
def _foreign_report_fd(report, account):
    ready_read, ready_write = os.pipe()
    finish_read, finish_write = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(ready_read)
        os.close(finish_write)
        fd = os.open(report, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.initgroups('blueprint', account.pw_gid)
            os.setgid(account.pw_gid)
            os.setuid(account.pw_uid)
            info = os.fstat(fd)
            os.write(ready_write, encoded(dict(fd=fd, uid=os.geteuid(), dev=info.st_dev, ino=info.st_ino)))
            assert os.read(finish_read, 1) == b'x'
        finally:
            os.close(fd)
            os.close(ready_write)
            os.close(finish_read)
            os._exit(0)
    _remember_fixture_child(child, 'foreign_report_reader')
    os.close(ready_write)
    os.close(finish_read)
    try:
        assert select.select([ready_read], [], [], 10)[0]
        held = json.loads(os.read(ready_read, 512))
        observed = os.stat('/proc/' + str(child) + '/fd/' + str(held['fd']))
        expected = report.stat()
        assert held['uid'] == account.pw_uid != 0
        assert (observed.st_dev, observed.st_ino) == (held['dev'], held['ino']) == (expected.st_dev, expected.st_ino)
        yield held
    finally:
        os.write(finish_write, b'x')
        os.close(finish_write)
        os.close(ready_read)
        os.waitpid(child, 0)
        _remember_fixture_child(child, reaped=True)


def _held_fd_keeps(value, action, pins, report, account):
    with _foreign_report_fd(report, account):
        original = report.read_bytes()
        result = _run_shipped_gc_sandbox(value, action, time.time(), pins, realtime=True)
        row = next(row for row in result['report']['registered_experiments']['outcomes']
                   if row['action_id'] == action['action_id'])
        assert row['decision'] == 'kept' and row['removed_logical_bytes'] == row['removed_allocated_bytes'] == 0, row
        assert row['reason'] == 'experiment_diagnostic_process_reference', row
        assert report.read_bytes() == original


def _current_queue_keeps(value, action, pins, report, *, directory_alias=False):
    """The actual installed GC must retain a URI-selected current payload."""
    from urllib.parse import unquote, urlparse
    from blueprint_pipeline.control_plane_lane_owner_consents import _metadata
    alias = value['config'].parent / 'current-source'
    if directory_alias:
        alias.symlink_to(report.parent, target_is_directory=True)
        alias_metadata = _metadata(alias.lstat())
        source = alias / report.name
        assert source.read_bytes() == report.read_bytes()
        assert (source.stat().st_dev, source.stat().st_ino) == (report.stat().st_dev, report.stat().st_ino)
        assert report.stat().st_nlink == 1
        selected = source.as_uri()
        expected_reason = 'experiment_diagnostic_references_unknown'
    else:
        selected = 'file://' + ''.join('/' if byte == 47 else '%' + format(byte, '02X')
                                        for byte in os.fsencode(report))
        expected_reason = 'experiment_diagnostic_queue_reference'
    parsed = urlparse(selected)
    assert parsed.scheme == 'file' and parsed.netloc == ''
    assert Path(unquote(parsed.path)) == (source if directory_alias else report)
    assert report.parent.name not in selected and str(report.parent) not in selected
    queue = value['config'].parent / 'reference-tables/queue/current-diagnostic.json'
    original = encoded(dict(source_uri=selected))
    descriptor = os.open(queue, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(original)
        stream.flush()
        os.fsync(stream.fileno())
    before, initial = report.read_bytes(), _metadata(report.stat())
    queued = _metadata(queue.stat())
    try:
        result = _run_shipped_gc_sandbox(value, action, time.time(), pins, realtime=True, invocation=3)
        row = next(row for row in result['report']['registered_experiments']['outcomes']
                   if row['action_id'] == action['action_id'])
        assert row['decision'] == 'kept' and row['reason'] == expected_reason, row
        assert row['removed_logical_bytes'] == row['removed_allocated_bytes'] == 0 and row['receipt'] is None, row
        assert not result['objects'] and not result['object_metadata']
        assert report.read_bytes() == before and _metadata(report.stat()) == initial
        assert queue.read_bytes() == original and _metadata(queue.stat()) == queued
    finally:
        assert _metadata(queue.stat()) == queued and queue.read_bytes() == original
        queue.unlink()  # Only this exact owned temporary queued row.
        if directory_alias:
            assert _metadata(alias.lstat()) == alias_metadata and os.readlink(alias) == str(report.parent)
            alias.unlink()  # Only this exact owned temporary alias.


def _restore_conflict_before_publication(value, restore, pins, report, *, now=time.time):
    """Keep a real foreign destination byte-exact, then remove only our fixture."""
    from blueprint_pipeline import control_plane_lane_experiment_actions as actions
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    actual_event = actions._event
    conflicts = []
    def event(*args, **kwargs):
        result = actual_event(*args, **kwargs)
        if args[3] == 'restore_stage_ready' and not conflicts:
            # First admission authenticates the unchanged retired directory.
            # Introduce the collision only after the actual owned stage and its
            # exact byte manifest have a durable receipt. Resume must validate
            # that original stage/namespace union, never amend retired tokens.
            assert not report.exists()
            files = args[0]
            stage_fd = files._diagnostic_references.stage
            files.proof(stage_fd)
            files.location(stage_fd)
            assert files.bindings[stage_fd][0] == files._diagnostic_references.target_fd
            stage_report = report.parent / files.bindings[stage_fd][1] / report.name
            staged_info = stage_report.stat()
            staged_bytes = stage_report.read_bytes()
            fd = os.open(report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(b'owned-foreign-fixture')
            conflicts.append(((report.stat().st_dev, report.stat().st_ino),
                              stage_report, (staged_info.st_dev, staged_info.st_ino), staged_bytes))
        return result
    actions._event = event
    try:
        try:
            refused = issuer.restore_registered_experiment(restore['action_id'], expected_restore_intent=restore['restore_intent'],
                installed_config_path=value['config'], now=now, _pins_root=pins)
        except ValueError as error:
            assert str(error) in {'experiment_restore_destination_unproven',
                                  'experiment_diagnostic_namespace_changed'}, error
        else:
            raise AssertionError('restore accepted an unowned destination: ' + str(refused))
        assert len(conflicts) == 1
        foreign_identity, stage_report, stage_identity, staged_bytes = conflicts[0]
        info = report.stat()
        assert (info.st_dev, info.st_ino) == foreign_identity and report.read_bytes() == b'owned-foreign-fixture'
        staged_info = stage_report.stat()
        assert (staged_info.st_dev, staged_info.st_ino) == stage_identity and stage_report.read_bytes() == staged_bytes
    finally:
        actions._event = actual_event
    report.unlink()


def _restore_reference_after_publication(value, restore, pins, target, account, original):
    """Expose a real foreign FD after journaled link, before staged unlink."""
    from blueprint_pipeline import control_plane_lane_experiment_actions as actions
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    actual_event = actions._event
    exposed = []
    with ExitStack() as holders:
        def event(*args, **kwargs):
            result = actual_event(*args, **kwargs)
            if args[3] == 'restore_member' and not exposed:
                files = args[0]
                # Select the actual owned stage binding, never a guessed birth.
                references = files._diagnostic_references
                stage_fd = references.stage
                files.proof(stage_fd)
                files.location(stage_fd)
                assert files.bindings[stage_fd][0] == references.target_fd
                stage = references.target / files.bindings[stage_fd][1]
                report = stage / 'disk-capacity-report.v1.json'
                assert report.read_bytes() == original
                held = holders.enter_context(_foreign_report_fd(report, account))
                exposed.append((report, held))
            return result
        actions._event = event
        try:
            try:
                issuer.restore_registered_experiment(restore['action_id'], expected_restore_intent=restore['restore_intent'],
                    installed_config_path=value['config'], now=time.time, _pins_root=pins)
            except ValueError as error:
                assert str(error) == 'experiment_diagnostic_process_reference', error
            else:
                raise AssertionError('restore unlinked a payload held by a foreign reader')
            assert len(exposed) == 1
            report, held = exposed[0]
            info = report.stat()
            assert (info.st_dev, info.st_ino) == (held['dev'], held['ino'])
            assert report.read_bytes() == (target / report.name).read_bytes() == original
        finally:
            actions._event = actual_event


def run(root):
    from blueprint_pipeline import control_plane_lane_owner_consents as owners
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    from blueprint_pipeline import control_plane_lane_experiment_birth as birth
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as diagnostic
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer
    from blueprint_pipeline import control_plane_lane_experiment_archive as archive
    from blueprint_pipeline import control_plane_lane_experiment_restore as restoration
    from blueprint_pipeline import control_plane_lane_experiment_completion as completion
    from blueprint_pipeline import control_plane_lane_experiment_actions as actions
    from blueprint_pipeline.control_plane_lane_experiment_work import _ActionFiles
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
    tables = root / 'reference-tables'
    tables.mkdir(mode=0o700)
    settings_lines = ['BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT=' + str(pins)]
    for kind in ('queue', 'evidence', 'settlement'):
        selected = tables / kind
        selected.mkdir(mode=0o700)
        settings_lines.append('BLUEPRINT_CONTROL_PLANE_GC_' + kind.upper() + '_ROOTS=' + str(selected))
    gc_env.write_text('\n'.join(settings_lines) + '\n')
    gc_env.chmod(0o600)
    state = root / 'sandbox-control-plane'
    state.mkdir(mode=0o700)
    release = root / 'active-release'
    release.symlink_to(root / 'installed')
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
        'experiment_gc_environment_file': str(gc_env), 'control_plane_state': str(state),
        'active_release_link': str(release)}))
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
        _held_fd_keeps(value, action, pins, report, account)
        _current_queue_keeps(value, action, pins, report, directory_alias=method == 'offload')
        result = _run_shipped_gc_sandbox(value, action, time.time(), pins, realtime=True, invocation=1)
        row = next(row for row in result['report']['registered_experiments']['outcomes'] if row['action_id'] == action['action_id'])
        assert row['decision'] == 'retired' and row['receipt'] and row['removed_logical_bytes'] == len(before), (row, result.get('native_limit_failures'), result.get('native_reference_failures'))
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
            _restore_conflict_before_publication(value, restore, pins, report)
            _restore_reference_after_publication(value, restore, pins, target, account, before)
            outcome = issuer.restore_registered_experiment(restore['action_id'], expected_restore_intent=restore['restore_intent'],
                installed_config_path=value['config'], now=time.time, _pins_root=pins)
            assert outcome['decision'] == 'restored', outcome
            assert report.read_bytes() == before and target.stat().st_ino == original_inode
            files = _ActionFiles(now=time.time)
            try:
                config, selected_gid = actions._context(files, value['config'], time.time())
                _, _, entry = actions._selected(files, config, grant['intent_id'], time.time(), selected_gid)
                selected_completion = completion.selected_completion(files, config, entry)
                assert selected_completion['report_member'] != selected_completion['selected_report_member']
                assert selected_completion['selected_report_member'][2].split(':')[:2] == [str(report.stat().st_dev), str(report.stat().st_ino)]
            finally:
                files.finish()
                files.budget.close()
            repeated_restore = issuer.restore_registered_experiment(restore['action_id'], expected_restore_intent=restore['restore_intent'],
                installed_config_path=value['config'], now=time.time, _pins_root=pins)
            assert repeated_restore['decision'] == 'restored'
            assert repeated_restore['removed_logical_bytes'] == repeated_restore['removed_allocated_bytes'] == 0
            assert report.read_bytes() == before
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
        zero_repeat_credit=True, shipped_gc_sandbox=True, actual_encoded_queue_kept=True,
        actual_directory_alias_kept=True)
