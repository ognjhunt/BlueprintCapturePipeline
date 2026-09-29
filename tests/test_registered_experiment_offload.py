"""Actual expired evidence traverses native multipart/full readback before GC unlink."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_actions.py
#   src/blueprint_pipeline/control_plane_lane_experiment_archive.py
import io
import json
from pathlib import Path

import pytest

from tests.test_registered_experiment_issuer import installation, encoded  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_experiment_producer import _registered_fixture
from tests.test_registered_experiment_retirement_flow import _gc, _current_entry, _payload_snapshot


class Cloud:
    """Tiny in-process object transport; no provider/credential request is made."""
    def __init__(self, *, corrupt=False):
        self.objects, self.parts, self.metadata = {}, {}, {}
        self.corrupt = corrupt
        self.calls, self.bodies = [], []
    def head_object(self, **args):
        key = args['Key']
        self.calls.append('head')
        return dict(ContentLength=len(self.objects[key]), Metadata=self.metadata[key], ETag='"tiny-original"')
    def create_multipart_upload(self, **args):
        self.calls.append('create')
        self.parts[args['Key']] = []
        self.metadata[args['Key']] = args['Metadata']
        return dict(UploadId='tiny-owned-upload')
    def upload_part(self, **args):
        self.calls.append('part')
        self.parts[args['Key']].append(args['Body'])
        return dict(ETag='"tiny-part"')
    def complete_multipart_upload(self, **args):
        self.calls.append('complete')
        self.objects[args['Key']] = b''.join(self.parts[args['Key']])
        return {}
    def abort_multipart_upload(self, **args):
        self.calls.append('abort')
        return {}
    def get_object(self, **args):
        self.calls.append('readback')
        body = io.BytesIO(self.objects[args['Key']] + (b'foreign' if self.corrupt else b''))
        self.bodies.append(body)
        return dict(Body=body)
    def close(self):
        self.calls.append('client_closed')


@pytest.fixture
def expired_completed_evidence(installation, tmp_path, monkeypatch):  # noqa: F811
    from blueprint_pipeline import native_g1_development_pair as pair
    from blueprint_pipeline import native_g1_development_worker as worker
    from blueprint_pipeline import control_plane_scratch_lifetime as lifetime
    policy = json.loads(installation[3].read_bytes())
    policy['principals'][0]['allowed_actions'] += ['offload', 'keep']
    installation[3].write_bytes(encoded(policy))
    config, settings, _, _ = installation
    settings['experiment_retirement_enabled'] = True
    pins = config.parent / 'pins'
    pins.mkdir()
    env = config.parent / 'gc.env'
    env.write_text('BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT=' + str(pins) + '\n')
    env.chmod(0o600)
    settings['experiment_gc_environment_file'] = str(env)
    config.write_bytes(encoded(settings))
    consumer, target, born, paths = _registered_fixture(installation, tmp_path, monkeypatch)
    monkeypatch.setattr(pair, 'LANE_SCRATCH_ROOTS', consumer.LANE_ROOTS)
    monkeypatch.setattr(lifetime, 'LANE_ROOTS', consumer.LANE_ROOTS)
    def refused(_request):
        raise ValueError('development_only_prelaunch_refusal')
    monkeypatch.setattr(worker, '_preflight_inputs', refused)
    use = consumer.RegisteredExperimentUse.admit(target, now=lambda: 1200, _producer_request_paths=paths)
    pair.run_g1_development_pair(request_paths=paths, output_dir=target, _registered_use=use)
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root
    action = root.issue_experiment_action_intent(use.entry['intent_id'], principal='operator', owner='owner',
        action='offload', expires_at_epoch=3500, installed_config_path=config, now=lambda: 2900)
    return installation, target, born, action, use.entry['intent_id']


@pytest.mark.parametrize('kind,mode', [('file', 0o660), ('directory', 0o770), ('file', 0o1600), ('directory', 0o1700)])
def test_offload_refuses_original_metadata_native_restore_cannot_reproduce(installation, monkeypatch, request, kind, mode):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root
    actual = root.issue_experiment_action_intent
    selected = []

    def issue(intent_id, **kwargs):
        settings = json.loads(Path(kwargs['installed_config_path']).read_bytes())
        target = Path(settings['lane_scratch_work_root']) / 'g1' / ('registered-' + intent_id)
        member = (target / 'native_g1_development_pair.v1.json' if kind == 'file'
                  else next(path for path in target.iterdir() if path.is_dir()))
        member.chmod(mode)
        selected.append((member, _payload_snapshot(target)))
        return actual(intent_id, **kwargs)

    monkeypatch.setattr(root, 'issue_experiment_action_intent', issue)
    with pytest.raises(ValueError, match='experiment_restore_member_mode'):
        request.getfixturevalue('expired_completed_evidence')
    member, snapshot = selected[0]
    assert member.stat().st_mode & 0o7777 == mode
    assert _payload_snapshot(member.parent if kind == 'file' else member.parent) == snapshot


@pytest.mark.parametrize('corrupt', [False, True])
def test_actual_gc_preserves_verified_archive_before_removing_expired_evidence(
        expired_completed_evidence, monkeypatch, corrupt):
    from blueprint_pipeline import control_plane_lane_experiment_archive as archive
    value, target, _, action, intent_id = expired_completed_evidence
    cloud = Cloud(corrupt=corrupt)
    monkeypatch.setattr(archive, '_client', lambda *a: (cloud, 'development-only'))
    before = _payload_snapshot(target)
    report = _gc(value)
    outcome = report['registered_experiments']['outcomes'][0]
    assert outcome['action_id'] == action['action_id']
    assert cloud.calls and all(body.closed for body in cloud.bodies)
    if corrupt:
        assert outcome['decision'] == 'kept' and _payload_snapshot(target) == before
        assert outcome['removed_logical_bytes'] == 0
    else:
        assert outcome['decision'] == 'retired', outcome
        events = [json.loads(p.read_bytes()) for p in (value[2] / 'operations' / action['action_id']).glob('e-*.json')]
        ready = next(event for event in events if event['event_kind'] == 'preservation_ready')
        assert ready['body']['archive']['full_byte_service_account_readback_passed'] is True
        assert ready['body']['archive']['remote_identity_verified'] is True
        assert ready['sequence'] < min(e['sequence'] for e in events if e['event_kind'] == 'member_removed')
        assert _current_entry(value, intent_id)['state'] == 'retired'
        assert len(list(target.iterdir())) == 2
        assert len(cloud.objects) == 1


@pytest.mark.parametrize('certificate_case', ['intact', 'missing', 'changed', 'held_changed', 'replayed', 'activation_crash', 'before_stage_crash', 'stage_ready_crash', 'member_link_crash', 'member_link_before_event', 'member_unlink_crash', 'stage_ready_changed', 'stage_ready_replaced', 'stage_ready_unproven', 'lease_lane_lock', 'lease_truncate_crash', 'lease_write_crash', 'lease_truncate_foreign_image', 'lease_truncate_foreign_inode', 'lease_truncate_changed_payload', 'second_expiry', pytest.param('slow_readback', marks=pytest.mark.slow), pytest.param('slow_restored_hash', marks=pytest.mark.slow)])
def test_actual_root_restore_renews_lease_and_reopens_real_registered_reader(
        expired_completed_evidence, monkeypatch, certificate_case):
    from blueprint_pipeline import control_plane_lane_experiment_archive as archive
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root
    from blueprint_pipeline import control_plane_lane_experiment_restore as restore
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer
    value, target, born, action, intent_id = expired_completed_evidence
    actual_installation = value
    before = {str(path.relative_to(target)): path.read_bytes() for path in target.rglob('*') if path.is_file()}
    original_inode = target.stat().st_ino
    cloud = Cloud()
    monkeypatch.setattr(archive, '_client', lambda *a: (cloud, 'development-only'))
    assert _gc(value)['registered_experiments']['outcomes'][0]['decision'] == 'retired'
    grant = root.issue_experiment_restore_intent(intent_id, principal='operator', owner='owner',
        lease_ttl_seconds=600, expires_at_epoch=3400, installed_config_path=value[0], now=lambda: 2901)
    class Reservation:
        released = False
        def release(self, **kwargs):
            self.released = True
    reservation = Reservation()
    allocations = []
    def reserve(*args, **kwargs):
        allocations.append((args, kwargs))
        return reservation
    monkeypatch.setattr(restore, 'reserve_control_plane_disk', reserve)
    if certificate_case == 'lease_lane_lock':
        import fcntl
        import os
        actual_truncate = restore.os.ftruncate
        lease_inode = (target / '.lane-scratch.v1.json').stat().st_ino
        def truncate_under_lane_lock(fd, size):
            if os.fstat(fd).st_ino == lease_inode:
                lock = os.open(target.parent.parent / '.lane-scratch.lock', os.O_RDWR | os.O_NOFOLLOW)
                try:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    os.close(lock)
            return actual_truncate(fd, size)
        monkeypatch.setattr(restore.os, 'ftruncate', truncate_under_lane_lock)
    if certificate_case == 'slow_readback':
        import time
        original_get = cloud.get_object
        delayed = False
        def slow_get(**kwargs):
            nonlocal delayed
            if not delayed:
                delayed = True
                time.sleep(6)
            return original_get(**kwargs)
        monkeypatch.setattr(cloud, 'get_object', slow_get)
    if certificate_case == 'slow_restored_hash':
        import os
        import time
        from blueprint_pipeline import control_plane_lane_experiment_actions as actions
        actual_read = actions.os.read
        delayed_hash = False
        def slow_member_hash(fd, amount):
            nonlocal delayed_hash
            # Only a genuinely restored destination file, never metadata or stage IO.
            destination = target / 'native_g1_development_pair.v1.json'
            if destination.exists() and os.fstat(fd).st_ino == destination.stat().st_ino and not delayed_hash:
                delayed_hash = True
                time.sleep(6)
            return actual_read(fd, amount)
        monkeypatch.setattr(actions.os, 'read', slow_member_hash)
    arguments = dict(expected_restore_intent=grant['restore_intent'], installed_config_path=value[0],
        now=lambda: 2902, _pins_root=value[0].parent / 'pins')
    if certificate_case == 'before_stage_crash':
        actual_verify = archive.verify_preservation
        def interrupted(*a, **kw):
            raise OSError('killed_before_actual_stage_creation')
        monkeypatch.setattr(archive, 'verify_preservation', interrupted)
        with pytest.raises(ValueError, match='experiment_'):
            root.restore_registered_experiment(grant['action_id'], **arguments)
        assert _current_entry(value, intent_id)['state'] == 'restoring'
        assert len(list(target.iterdir())) == 2
        monkeypatch.setattr(archive, 'verify_preservation', actual_verify)
    if certificate_case.startswith('stage_ready_'):
        from blueprint_pipeline.control_plane_lane_experiment_work import _ActionFiles
        original_phase = _ActionFiles.phase
        def interrupted_before_link(files, name):
            if name in ('restore_directories', 'restore_batch'):
                raise OSError('killed_after_verified_stage_before_first_link')
            return original_phase(files, name)
        monkeypatch.setattr(_ActionFiles, 'phase', interrupted_before_link)
        with pytest.raises(ValueError, match='experiment_'):
            root.restore_registered_experiment(grant['action_id'], **arguments)
        stage = target / ('.restore-' + grant['action_id'])
        assert stage.is_dir() and list(stage.rglob('*'))
        transport_calls = list(cloud.calls)
        monkeypatch.setattr(_ActionFiles, 'phase', original_phase)
        if certificate_case != 'stage_ready_crash':
            payload = next(path for path in stage.rglob('*') if path.is_file())
            if certificate_case == 'stage_ready_changed':
                payload.write_bytes(b'foreign')
            elif certificate_case == 'stage_ready_replaced':
                raw = payload.read_bytes()
                payload.unlink()
                payload.write_bytes(raw)
            else:
                (value[2] / 'operations' / grant['action_id'] / 'e-00001.json').unlink()
            retained = {str(path.relative_to(stage)): path.read_bytes() for path in stage.rglob('*') if path.is_file()}
            with pytest.raises(ValueError, match='experiment_'):
                root.restore_registered_experiment(grant['action_id'], **arguments)
            assert cloud.calls == transport_calls and len(allocations) == 1
            assert {str(path.relative_to(stage)): path.read_bytes() for path in stage.rglob('*') if path.is_file()} == retained
            assert _current_entry(value, intent_id)['state'] == 'restoring'
            return
    if certificate_case == 'member_link_crash':
        from blueprint_pipeline import control_plane_lane_experiment_actions as actions
        actual_event = actions._event
        stopped = False
        def interrupted_after_link(*args, **kwargs):
            nonlocal stopped
            selected = actual_event(*args, **kwargs)
            if args[3] == 'restore_member' and not stopped:
                stopped = True
                raise OSError('killed_after_real_link_and_durable_event_before_source_unlink')
            return selected
        monkeypatch.setattr(actions, '_event', interrupted_after_link)
        with pytest.raises(ValueError, match='experiment_'):
            root.restore_registered_experiment(grant['action_id'], **arguments)
        assert stopped and _current_entry(value, intent_id)['state'] == 'restoring'
        transport_calls = list(cloud.calls)
        monkeypatch.setattr(actions, '_event', actual_event)
    if certificate_case == 'member_link_before_event':
        from blueprint_pipeline import control_plane_lane_experiment_actions as actions
        actual_event = actions._event
        stopped = False
        def interrupted_before_receipt(*args, **kwargs):
            nonlocal stopped
            if args[3] == 'restore_member' and not stopped:
                stopped = True
                raise OSError('killed_after_actual_link_before_durable_member_event')
            return actual_event(*args, **kwargs)
        monkeypatch.setattr(actions, '_event', interrupted_before_receipt)
        with pytest.raises(ValueError, match='experiment_'):
            root.restore_registered_experiment(grant['action_id'], **arguments)
        assert stopped and _current_entry(value, intent_id)['state'] == 'restoring'
        transport_calls = list(cloud.calls)
        monkeypatch.setattr(actions, '_event', actual_event)
    if certificate_case == 'member_unlink_crash':
        import os
        original_unlink = restore.os.unlink
        stopped = False
        stage = target / ('.restore-' + grant['action_id'])
        def interrupted_after_source_unlink(name, *args, **kwargs):
            nonlocal stopped
            parent = None
            if stage.exists() and 'dir_fd' in kwargs and not name.startswith('.target-version-'):
                selected = os.fstat(kwargs['dir_fd']).st_ino
                parent = next((path for path in [stage, *stage.rglob('*')] if path.is_dir() and path.stat().st_ino == selected), None)
            result = original_unlink(name, *args, **kwargs)
            if not stopped and parent is not None and (target / parent.relative_to(stage) / name).is_file():
                stopped = True
                raise OSError('killed_after_real_stage_source_unlink')
            return result
        monkeypatch.setattr(restore.os, 'unlink', interrupted_after_source_unlink)
        with pytest.raises(ValueError, match='experiment_'):
            root.restore_registered_experiment(grant['action_id'], **arguments)
        assert stopped and _current_entry(value, intent_id)['state'] == 'restoring'
        transport_calls = list(cloud.calls)
        monkeypatch.setattr(restore.os, 'unlink', original_unlink)
    if certificate_case.startswith('lease_truncate') or certificate_case == 'lease_write_crash':
        import os
        actual_truncate, actual_write = restore.os.ftruncate, restore.os.write
        lease_inode = (target / '.lane-scratch.v1.json').stat().st_ino
        stopped = False
        def killed_truncate(fd, size):
            nonlocal stopped
            result = actual_truncate(fd, size)
            if os.fstat(fd).st_ino == lease_inode and not stopped:
                stopped = True
                raise OSError('killed_after_original_lease_truncate')
            return result
        def killed_write(fd, payload):
            nonlocal stopped
            if os.fstat(fd).st_ino == lease_inode and not stopped:
                stopped = True
                actual_write(fd, payload[:len(payload)//2])
                raise OSError('killed_during_original_lease_write')
            return actual_write(fd, payload)
        monkeypatch.setattr(restore.os, 'ftruncate' if certificate_case.startswith('lease_truncate') else 'write',
                            killed_truncate if certificate_case.startswith('lease_truncate') else killed_write)
        with pytest.raises(ValueError, match='experiment_'):
            root.restore_registered_experiment(grant['action_id'], **arguments)
        assert stopped and _current_entry(value, intent_id)['state'] == 'restoring'
        with pytest.raises(ValueError, match='experiment_'):
            consumer.RegisteredExperimentUse.admit(target, now=lambda: 3000)
        transport_calls = list(cloud.calls)
        monkeypatch.setattr(restore.os, 'ftruncate', actual_truncate)
        monkeypatch.setattr(restore.os, 'write', actual_write)
        if certificate_case in ('lease_truncate_foreign_image', 'lease_truncate_foreign_inode', 'lease_truncate_changed_payload'):
            lease_path = target / '.lane-scratch.v1.json'
            if certificate_case == 'lease_truncate_foreign_image':
                lease_path.write_bytes(b'foreign transition')
            elif certificate_case == 'lease_truncate_foreign_inode':
                original = lease_path.stat().st_ino
                lease_path.rename(value[0].parent / 'held-original-lease')
                lease_path.write_bytes(before['.lane-scratch.v1.json'])
                lease_path.chmod(0o600)
                assert lease_path.stat().st_ino != original
            else:
                (target / 'native_g1_development_pair.v1.json').write_bytes(b'foreign payload')
            retained = _payload_snapshot(target)
            with pytest.raises(ValueError, match='experiment_'):
                root.restore_registered_experiment(grant['action_id'], **arguments)
            assert _payload_snapshot(target) == retained and cloud.calls == transport_calls and len(allocations) == 1
            assert _current_entry(value, intent_id)['state'] == 'restoring'
            return
    if certificate_case == 'activation_crash':
        from blueprint_pipeline import control_plane_lane_experiment_actions as actions
        actual_event = actions._event
        def killed(*args, **kwargs):
            if args[3] == 'activation_complete':
                raise OSError('killed_after_actual_active_head')
            return actual_event(*args, **kwargs)
        monkeypatch.setattr(actions, '_event', killed)
        with pytest.raises(ValueError, match='experiment_'):
            root.restore_registered_experiment(grant['action_id'], **arguments)
        monkeypatch.setattr(actions, '_event', actual_event)
    outcome = root.restore_registered_experiment(grant['action_id'], **arguments)
    if certificate_case in ('replayed', 'activation_crash'):
        calls = list(cloud.calls)
        replay = root.restore_registered_experiment(grant['action_id'], **arguments)
        assert replay['receipt'] == outcome['receipt'] and cloud.calls == calls

    if certificate_case in ('stage_ready_crash', 'member_link_crash', 'member_link_before_event', 'member_unlink_crash', 'lease_truncate_crash', 'lease_write_crash'):
        assert cloud.calls == transport_calls  # Durable stage proof avoids new transfer.
    assert outcome['decision'] == 'restored' and reservation.released
    assert len(allocations) == (2 if certificate_case == 'before_stage_crash' else 1)
    assert allocations[0][1]['minimum_bytes'] > 0
    entry = _current_entry(value, intent_id)
    assert entry['state'] == 'active' and entry['generation'] == born['generation']
    assert entry['restoration'] is not None and target.stat().st_ino == original_inode
    for relative, raw in before.items():
        if relative != '.lane-scratch.v1.json':
            assert (target / relative).read_bytes() == raw
    certificate = consumer.AUTHORITY_ROOT / ('restoration-' + entry['restoration']['sha256'][7:] + '.json')
    if certificate_case in ('missing', 'changed'):
        if certificate_case == 'missing':
            certificate.unlink()
        else:
            certificate.write_bytes(b'{}')
        with pytest.raises(ValueError, match='experiment_'):
            consumer.RegisteredExperimentUse.admit(target, now=lambda: 3000)
        return
    with consumer.RegisteredExperimentUse.admit(target, now=lambda: 3000) as use:
        if certificate_case == 'held_changed':
            certificate.write_bytes(b'{}')
            with pytest.raises(ValueError, match='experiment_'):
                use.check()
            return
        assert use.entry['restoration'] == entry['restoration']
        from blueprint_pipeline.native_g1_development_pair import _read_result
        from blueprint_pipeline import native_g1_development_pair as pair
        value = json.loads((target / (pair.SCHEMA + '.json')).read_bytes())
        candidate = value['attempts'][0]['candidate_id']
        worker = json.loads((target / candidate / 'native_g1_development_worker_result.v1.json').read_bytes())
        result = _read_result(target / candidate / 'native_g1_development_worker_result.v1.json',
            candidate_id=candidate, scene_plan_digest=value['scene_plan_digest'], request_digest=worker['request_digest'])
        assert result['status'] == 'blocked'
    if certificate_case == 'second_expiry':
        from blueprint_pipeline.control_plane_storage_gc import run_storage_gc, RUN_ACK
        config, settings, _, _ = actual_installation
        historical = entry['completion']
        again = root.issue_experiment_action_intent(intent_id, principal='operator', owner='owner',
            action='offload', expires_at_epoch=4000, installed_config_path=config, now=lambda: 3502)
        report = run_storage_gc(content_store_roots=(), derived_roots=(), queue_roots=(), pins_root=config.parent / 'pins',
            apply=True, ack=RUN_ACK, lane_scratch_roots=(settings['lane_scratch_work_root'], settings['lane_scratch_inputs_root']),
            lane_scratch_enabled=True, _experiment_config_path=config, now=lambda: 3502)
        retired = report['registered_experiments']['outcomes'][0]
        assert retired['decision'] == 'retired' and retired['action_id'] == again['action_id'], retired
        assert _current_entry(actual_installation, intent_id)['completion'] == historical
        assert len(list(target.iterdir())) == 2


def test_actual_issued_manifest_binds_exact_generation_and_compact_member_versions(expired_completed_evidence):
    value, target, born, action, intent_id = expired_completed_evidence
    payload = json.loads((value[2] / (action['action_id'] + '.action.json')).read_bytes())
    manifest = json.loads((value[2] / (action['action_id'] + '.manifest.json')).read_bytes())
    assert set(manifest) == {'schema_version', 'generation', 'birth', 'target_identity', 'lease',
                             'completion', 'members', 'logical_bytes', 'allocated_bytes', 'manifest_digest'}
    assert all(manifest[key] == payload[key] for key in ('generation', 'birth', 'target_identity', 'lease', 'completion'))
    assert manifest['generation'] == born['generation'] and manifest['members']
    for path, kind, identity, metadata, digest in manifest['members']:
        info = (target/path).stat()
        assert identity == f'{info.st_dev}:{info.st_ino}:' + ('r' if kind == 'file' else 'd')
        assert int(metadata.split(':')[0]) == info.st_mode & 0o7777
        assert digest is None if kind == 'directory' else digest.startswith('sha256:')


@pytest.mark.slow
def test_actual_gc_can_stream_slow_payload_without_resetting_metadata_budget(expired_completed_evidence, monkeypatch):
    import time
    from blueprint_pipeline import control_plane_lane_experiment_archive as archive
    value, target, _, action, intent_id = expired_completed_evidence
    class SlowCloud(Cloud):
        def create_multipart_upload(self, **args):
            time.sleep(6)  # Real external-call delay; fixture payload remains tiny.
            return super().create_multipart_upload(**args)
    cloud = SlowCloud()
    monkeypatch.setattr(archive, '_client', lambda *a: (cloud, 'development-only'))
    report = _gc(value)
    outcome = report['registered_experiments']['outcomes'][0]
    assert outcome['decision'] == 'retired', outcome
    assert _current_entry(value, intent_id)['state'] == 'retired'
    assert len(list(target.iterdir())) == 2 and all(body.closed for body in cloud.bodies)


def test_completed_evidence_uses_same_protected_installed_pins_selection(expired_completed_evidence):
    value, _, _, _, _ = expired_completed_evidence
    settings = json.loads(value[0].read_bytes())
    env = Path(settings['experiment_gc_environment_file'])
    assert env.parent == value[0].parent
    assert env.read_text().splitlines() == ['BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT=' + str(value[0].parent / 'pins')]
