# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_preparation_family_status.py
#   src/blueprint_pipeline/website_preparation_status.py
#   src/blueprint_pipeline/website_preparation_status_http.py
"""Actual local compiler/birth/ledger/signed-route replay, never live acceptance."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import itertools
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_preparation_family_status as family_status
from blueprint_pipeline import website_preparation_status as status
from blueprint_pipeline import website_task_context as context_reader
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.task_object_authoring_contracts import file_record
from blueprint_pipeline.task_evaluation_scene_retirement_generations import capture_birth_source_projection
from blueprint_pipeline.website_task_context_contracts import retains_task_family_preparation
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from tests.test_live_pipeline_intake_service import _signed_intake_headers
from tests.test_task_family_button_authoring import _fixture as button_fixture, _reseal_facts
from tests.test_website_preparation_status import prepared_source as prepared_source

_NONCES = itertools.count()


@pytest.fixture
def family_source(prepared_source, tmp_path, monkeypatch):
    root, owner, context, selected, ledger, _ = prepared_source
    fixture_root = tmp_path / 'compiler'
    fixture_root.mkdir()
    args, _ = button_fixture(fixture_root, monkeypatch)
    compiled_context = args['task_context']
    compiled_context.update({key: owner[key] for key in ('request_id', 'scene_id', 'capture_id')})
    geometry, masks = args['source_geometry'], args['task_masks']
    video = capture_birth_source_projection(root)['raw_video']['sha256']
    geometry['binding']['source_video_digest'] = video
    geometry['digest'] = canonical_digest(geometry, digest_field='digest')
    masks['source_geometry_digest'] = geometry['digest']
    layout = masks['registered_layout']
    layout.update(source_video_digest=video, source_geometry_digest=geometry['digest'])
    frame = layout['frame_registration']
    frame['source_geometry_digest'] = geometry['digest']
    frame['registration_digest'] = canonical_digest(frame, digest_field='registration_digest')
    reference = Path(layout['registration_evidence']['path'])
    write_json(reference, {'claim_ceiling': 'development_only', 'frame_registration': frame, 'source_video_digest': video})
    layout['registration_evidence'] = file_record(reference)
    layout['layout_digest'] = canonical_digest(layout, digest_field='layout_digest')
    masks['digest'] = canonical_digest(masks, digest_field='digest')
    facts = compiled_context['operator_answers']['family_task_facts']
    facts.update(capture_id=owner['capture_id'], source_video_digest=video)
    facts['operations'][0].pop('minimum_hold_seconds')
    _reseal_facts(args)
    context.clear()
    context.update(compiled_context)
    args['task_context'] = context
    selected['task_context_digest'] = context['context_digest']
    write_json(root / 'pipeline/website_task_context.json', context)
    args['output_root'] = root / 'pipeline/website_scene_preparation'
    preparation = compile_website_scene_preparation(**args)
    assert preparation['status'] == 'needs_input'
    handoff = {'schema_version': 'website_scene_handoff.v1', 'status': preparation['status'],
        **{key: owner[key] for key in ('scene_id', 'capture_id')},
        'job_disposition': 'accepted', 'visual_reconstruction_ready': True, 'simulator_ready': False,
        'source_geometry_digest': geometry['digest'], 'blockers': preparation['blockers'],
        'preparation_path': '/not-dereferenced/private/preparation.json', 'preparation_digest': preparation['digest'],
        'task_evaluation_inputs': preparation['task_evaluation_inputs'], 'provider_mutation_performed': False}
    handoff['digest'] = canonical_digest(handoff, digest_field='digest')
    assert retains_task_family_preparation(handoff, context)
    write_json(args['output_root'] / 'handoff.json', handoff)
    listener._write_output_commit(root, scene_id=owner['scene_id'], capture_id=owner['capture_id'],
        attempt_count=ledger['attempt_count'], result={'status': 'completed', 'website_scene_preparation': handoff},
        owner='synthetic-worker', token=ledger['lease_token'], ledger_update={'status': 'completed'})
    identity = {'schema_version': 'runtime_source_identity.v1', 'state': 'proven', 'source_commit': 'a' * 40,
        'source_digest': 'b' * 64, 'basis': 'synthetic_crypto_boundary_for_status_replay'}
    monkeypatch.setattr(family_status, 'running_source_identity', lambda *_: deepcopy(identity))
    monkeypatch.setenv(status.FAMILY_STATUS_ENV, 'true')
    return {'source': prepared_source, 'root': root, 'selected': selected, 'context': context,
        'handoff': handoff, 'preparation': preparation, 'identity': identity}


def _post(fixture, version='4'):
    body = json.dumps(fixture['selected'])
    headers = _signed_intake_headers('synthetic-secret', body, nonce=f'family-status-{next(_NONCES)}', client_id='blueprint-webapp')
    headers[status.VERSION_HEADER] = version
    return TestClient(service.create_app()).post('/api/live-pipeline/website-preparation-status', content=body, headers=headers)


def _receipt(fixture):
    response = _post(fixture)
    assert response.status_code == 200, response.text
    return response.json()


def _acceptance(fixture):
    value = _receipt(fixture)
    return {'schema_version': 'website_preparation_status_acceptance.v4', **fixture['selected'],
        'accepted': True, 'state': value['state'], 'status_digest': value['status_digest'],
        'native_execution_complete': False, 'correlation_id': 'bp-prep-' + value['status_digest'][7:23]}


def test_actual_compiler_signed_read_emits_bounded_source_bound_optional_dto(family_source):
    fixture = family_source
    before = {str(path): path.read_bytes() for path in fixture['root'].rglob('*') if path.is_file()}
    value = _receipt(fixture)
    projection = value['family_authoring']
    assert value['schema_version'] == status.STATUS_V4 and value['state'] == 'awaiting_inputs'
    assert value['outcome'] is None and value['terminal_reason'] is None
    assert projection['source_video_digest'] == capture_birth_source_projection(fixture['root'])['raw_video']['sha256']
    assert projection['task_context_digest'] == fixture['selected']['task_context_digest']
    assert projection['pipeline_source_commit'] == 'a' * 40
    assert projection['next_action']['quantity'] == 'minimum_hold_seconds'
    assert projection['automatic_paid_execution_authorized'] is False
    assert '/not-dereferenced/private' not in json.dumps(value) and '"path"' not in json.dumps(projection)
    assert len(json.dumps(value).encode()) <= 65536
    assert value['status_digest'] == cross_runtime_canonical_digest(value, digest_field='status_digest')
    assert before == {str(path): path.read_bytes() for path in fixture['root'].rglob('*') if path.is_file()}


def test_feature_off_preserves_exact_v3_and_never_reads_optional_evidence(family_source, monkeypatch):
    baseline = _post(family_source, '3').content
    monkeypatch.delenv(status.FAMILY_STATUS_ENV)
    monkeypatch.setattr(family_status, 'read_family_authoring_projection', lambda **_: pytest.fail('Optional source read while disabled'))
    assert _post(family_source).content == baseline
    assert status.requested_status_version('4') == status.STATUS_V3
    assert status.requested_status_version('2') == status.STATUS_V2


@pytest.mark.parametrize('fault', ['identity', 'original_video', 'preparation', 'symlink', 'payload_limit'])
def test_missing_optional_proof_retains_the_original_status_outcome(family_source, monkeypatch, fault):
    fixture = family_source
    if fault == 'identity':
        fixture['identity']['state'] = 'unproven'
    elif fault == 'original_video':
        preparation = deepcopy(fixture['preparation'])
        request = preparation['task_evaluation_inputs']['authoring']['request']
        request['binding']['source_video_digest'] = 'sha256:' + '0' * 64
        request['request_digest'] = canonical_digest(request, digest_field='request_digest')
        preparation['digest'] = canonical_digest(preparation, digest_field='digest')
        handoff = deepcopy(fixture['handoff'])
        handoff.update(preparation_digest=preparation['digest'], task_evaluation_inputs=preparation['task_evaluation_inputs'])
        handoff['digest'] = canonical_digest(handoff, digest_field='digest')
        write_json(fixture['root'] / 'pipeline/website_scene_preparation/preparation.json', preparation)
        write_json(fixture['root'] / 'pipeline/website_scene_preparation/handoff.json', handoff)
    elif fault == 'preparation':
        (fixture['root'] / 'pipeline/website_scene_preparation/preparation.json').write_text('{invalid')
    elif fault == 'symlink':
        path = fixture['root'] / 'pipeline/website_scene_preparation/preparation.json'
        path.rename(path.with_name('retained-private.json'))
        path.symlink_to(path.with_name('retained-private.json'))
    else:
        monkeypatch.setattr(family_status, 'read_family_authoring_projection', lambda **_: ({'oversized': 'x' * 65536}, None))
    diagnostic = []
    value = status.read_preparation_status(capture_root=fixture['root'], selectors=fixture['selected'],
        schema_version=status.STATUS_V4, _projection_diagnostics=diagnostic)
    assert value['family_authoring'] is None and value['state'] == 'awaiting_inputs'
    assert value['outcome'] is None and value['terminal_reason'] is None
    assert diagnostic and all(code.startswith('website_family_authoring_') for code in diagnostic)
    assert '/private' not in json.dumps(diagnostic)


@pytest.mark.parametrize('changed', ['handoff', 'preparation', 'identity'])
def test_optional_evidence_changed_during_projection_is_not_reported_as_current(family_source, monkeypatch, changed):
    fixture = family_source
    original = family_status.project_family_authoring_status
    touched = []
    def interleaved(**kwargs):
        result = original(**kwargs)
        if changed == 'identity':
            fixture['identity']['source_commit'] = 'b' * 40
        else:
            path = fixture['root'] / 'pipeline/website_scene_preparation' / (changed + '.json')
            path.write_bytes(path.read_bytes() + b'\n')
        touched.append(changed)
        return result
    monkeypatch.setattr(family_status, 'project_family_authoring_status', interleaved)
    diagnostic = []
    result = status.read_preparation_status(capture_root=fixture['root'], selectors=fixture['selected'],
        schema_version=status.STATUS_V4, _projection_diagnostics=diagnostic)
    assert result['family_authoring'] is None and result['state'] == 'awaiting_inputs'
    assert touched == [changed] and diagnostic


def test_processing_revision_does_not_publish_retained_previous_attempt_authoring(family_source, monkeypatch):
    fixture = family_source
    with listener._locked_job_ledger(fixture['root']) as ledger:
        listener._commit_job_ledger(fixture['root'], {**ledger, 'status': 'processing'}, previous_revision=ledger['revision'])
    monkeypatch.setattr(family_status, '_read', lambda *_: pytest.fail('Do not read previous attempt artifacts'))
    value = _receipt(fixture)
    assert value['state'] == 'preparing' and value['family_authoring'] is None


@pytest.mark.parametrize('changed', ['context', 'owner', 'birth', 'ledger', 'withdrawal'])
def test_source_authority_and_ledger_changes_after_projection_refuse_snapshot(family_source, monkeypatch, changed):
    fixture = family_source
    original = family_status.read_family_authoring_projection
    changed_state = []
    def interleaved(**kwargs):
        result = original(**kwargs)
        assert result[0] is not None
        if changed == 'context':
            fixture['context']['description'] += ' changed after snapshot'
            fixture['context']['context_digest'] = canonical_digest(fixture['context'], digest_field='context_digest')
        elif changed == 'owner':
            fixture['source'][1]['capture_rights']['derived_scene_generation_allowed'] = False
        elif changed == 'birth':
            value = capture_birth_source_projection(fixture['root'])
            Path(value['source_membership_raw_ref']['path']).write_bytes(b'changed original membership bytes')
        elif changed == 'ledger':
            with listener._locked_job_ledger(fixture['root']) as ledger:
                listener._commit_job_ledger(fixture['root'], ledger, previous_revision=ledger['revision'])
        else:
            from blueprint_pipeline.website_capture_withdrawal import acknowledge_withdrawal
            acknowledge_withdrawal(capture_root=fixture['root'], command={'schema_version': 'website_capture_withdrawal.v1',
                **{key: fixture['source'][1][key] for key in ('request_id', 'scene_id', 'capture_id')},
                'withdrawal_id': 'synthetic-after-projection', 'requested_at_iso': '2026-10-28T00:00:00Z'})
        changed_state.append(changed)
        return result
    monkeypatch.setattr(family_status, 'read_family_authoring_projection', interleaved)
    response = _post(fixture)
    assert response.status_code == 503 and response.json() == {'detail': 'website_preparation_status_unavailable'}
    assert changed_state == [changed]


def test_optional_projection_retry_uses_existing_outbox_without_provider_or_ledger_reentry(family_source, monkeypatch):
    fixture = family_source
    fixture['identity']['state'] = 'unproven'
    status.retain_preparation_wakeup(fixture['root'])
    before = listener._read_job_ledger(fixture['root'])
    observed = []
    def callback(**kwargs):
        assert kwargs['operation'] == 'preparation-status' and kwargs['payload'] == fixture['selected']
        result = _acceptance(fixture)
        observed.append(result)
        return result
    monkeypatch.setattr(context_reader, 'website_webapp_request', callback)
    storage = fixture['root'].parents[4]
    first = status.reconcile_preparation_wakeups(storage, limit=1)
    delivery_path = fixture['root'] / status.DELIVERY_FILE
    pending = json.loads(delivery_path.read_bytes())
    assert first['attempted'] == 1 and first['delivered'] == 0
    assert pending['state'] == 'pending' and pending['acceptance']['accepted'] is True
    assert pending['last_optional_projection_failure']['code'] == 'website_family_authoring_source_identity_unproven'
    assert pending['optional_projection_state'] == 'pending'
    fixture['identity']['state'] = 'proven'
    second = status.reconcile_preparation_wakeups(storage, limit=1, now_epoch=pending['next_attempt_at_epoch'])
    delivered = json.loads(delivery_path.read_bytes())
    assert second['attempted'] == second['delivered'] == 1
    assert delivered['state'] == 'delivered' and delivered['optional_projection_state'] == 'current'
    assert delivered['last_optional_projection_failure'] == pending['last_optional_projection_failure']
    assert len(observed) == 2 and listener._read_job_ledger(fixture['root']) == before


def test_export_actual_pipeline_receipts_for_independent_web_consumer(family_source):
    fixture = family_source
    known = _receipt(fixture)
    fixture['identity']['state'] = 'unproven'
    unknown = _receipt(fixture)
    fixture['identity']['state'] = 'proven'
    fixture['identity']['source_commit'] = 'b' * 40
    refreshed = _receipt(fixture)
    assert known['revision'] == unknown['revision'] == refreshed['revision']
    assert known['status_digest'] != unknown['status_digest'] != refreshed['status_digest']
    destination = os.getenv('BLUEPRINT_FAMILY_STATUS_FIXTURE_OUT')
    if destination:
        paths = [Path(status.__file__), Path(family_status.__file__)]
        value = {'schema_version': 'website_family_status_pipeline_replay.v1',
            'layer': 'actual local compiler/birth/ledger/signed Pipeline handlers; synthetic owner/storage/source-proof boundary; no production acceptance',
            'source_files': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            'selectors': fixture['selected'], 'known': known, 'unknown': unknown, 'refreshed': refreshed}
        Path(destination).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
