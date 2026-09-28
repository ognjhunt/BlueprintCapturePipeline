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
    pins = tmp_path / 'pins'
    pins.mkdir()
    env = tmp_path / 'gc.env'
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
