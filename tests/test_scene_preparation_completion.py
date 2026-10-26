"""Preparation completion needs publication/readback/closure, not robot controls."""
import json

import pytest

from blueprint_pipeline import robot_free_readiness_retention as retention
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_execution_authority import bind_scene_attempt
from blueprint_pipeline.task_evaluation_scene_preparation_completion import reconcile_preparation_completion
from blueprint_pipeline.task_evaluation_launch_terminal_evidence import _scene_configuration_terminal_projection
from tests.test_website_scene_preparation_scope import preparation_request
from tests.test_task_evaluation_configured_controls_progression_worker import _source, _digest, SOURCE_CONFIGURATION_COMMIT


def save(path, value, field=None):
    if field:
        value[field] = canonical_digest(value, digest_field=field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return value


def _prepare(tmp_path, monkeypatch, request):
    # The published references here are placeholders; retention of the robot-free
    # readiness receipt is pinned by tests/test_robot_free_readiness_retention.py.
    retained = []
    monkeypatch.setattr(retention, 'retain_robot_free_readiness',
                        lambda **kwargs: retained.append(kwargs) or {'status': 'probe_unavailable', 'retained': True})
    root, result_path = _source(tmp_path)
    run = root / 'scene-839873-qualifying'
    intent_root = tmp_path / 'intents'
    staged = intake.stage_scene_intent(value=request, queue_root=intent_root,
        authenticated_client='webapp', trusted_clients={'webapp'}, now=100)
    intent = intake._read(intent_root / staged['intent_id'] / 'intent.json', 'intent_digest')
    attempt = intake.reserve_scene_attempt(queue_root=intent_root, intent_id=intent['intent_id'],
        attempt_id='prepare', source_commit=SOURCE_CONFIGURATION_COMMIT, runtime_digest='sha256:' + 'a' * 64,
        input_digest='sha256:' + 'b' * 64, provider='vast', maximum_spend_usd=2, now=101)
    profile = save(run / 'launch_profile.json', {**bind_scene_attempt(attempt), 'source_commit': SOURCE_CONFIGURATION_COMMIT,
        'task_evaluation_run': {'run_mode': 'scene_configuration', 'configuration_run_id': 'scene-839873-configuration',
                               'task_id': intent['request']['task']['task_id']}}, 'profile_digest')
    reference = {'uri': 's3://test/qualified-scene', 'digest': 'sha256:' + 'c' * 64, 'size_bytes': 100}
    result = json.loads(result_path.read_text())
    result.update(status='completed', source_commit=SOURCE_CONFIGURATION_COMMIT, configuration_completed=True,
        configured_scene_published=True, full_byte_service_account_readback_passed=True,
        configured_scene_revision_reference=reference, configured_scene_bundle_reference=reference,
        configured_scene_revision_digest=reference['digest'], publication_result_digest=reference['digest'],
        task_thumbnail_reference=reference, task_thumbnail_selection_receipt_reference=reference)
    offering = {'schema_version': 'task_evaluation_configured_scene_offering.v1', 'status': 'configured_controls_pending',
        'configuration_run_id': result['run_id'], 'catalog_visibility': 'team_only',
        'presentation': {'task_thumbnail': reference, 'selection_receipt': reference},
        'evaluation_preparation_binding': {'configured_scene_revision': reference,
            'configured_scene_revision_digest': reference['digest'], 'configured_scene_bundle': reference}}
    offering['offering_digest'] = canonical_digest(offering, digest_field='offering_digest')
    result['configured_scene_offering'] = offering
    finalization = {'schema_version': 'task_evaluation_scene_construction_finalization.v1', 'status': 'completed',
        'queue_state': 'completed', 'finalization_performed': True, 'run_id': result['run_id'],
        'source_commit': SOURCE_CONFIGURATION_COMMIT}
    finalization['result_digest'] = canonical_digest(finalization, digest_field='result_digest')
    result['scene_construction_queue_finalization'] = finalization
    save(result_path, result, 'result_digest')
    receipt = json.loads((run / 'launch_receipt.json').read_text())
    receipt['launch_profile_digest'] = profile['profile_digest']
    receipt['terminal_evidence']['result']['digest'] = _digest(result_path)
    receipt['terminal_evidence']['scene_configuration'] = _scene_configuration_terminal_projection(result)[0]
    save(run / 'launch_receipt.json', receipt, 'receipt_digest')
    for name, field in [('webapp_sync_succeeded.json', 'sync_result_digest'),
                        ('post_teardown_provider_zero_receipt.json', 'provider_zero_receipt_digest')]:
        value = json.loads((run / name).read_text())
        value.update(receipt_digest=receipt['receipt_digest'])
        if 'response' in value:
            value['response'].update(receipt_digest=receipt['receipt_digest'])
            for row in (value, value['response']):
                row.update(configured_scene_offering_digest=offering['offering_digest'],
                           configured_scene_offering_status=offering['status'])
        else:
            value['launch_profile_digest'] = profile['profile_digest']
        save(run / name, value, field)
    return dict(intent=intent, config={'launch_execution_root': str(root), 'intent_root': str(intent_root)}), run, result_path


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    return _prepare(tmp_path, monkeypatch, preparation_request())


def test_scene_preparation_finishes_without_robot_or_policy_result(prepared):
    args, run, _ = prepared
    result = reconcile_preparation_completion(**args)
    assert result['status'] == 'completed' and result['phase'] == 'scene_prepared'
    assert result['state']['scene_preparation_completion']['robot_evaluation_performed'] is False
    assert not (run / 'policy_canary_result_projection.json').exists()


def test_completion_retains_the_robot_free_readiness_receipt_beside_the_launch(prepared, monkeypatch):
    args, run, result_path = prepared
    calls = []
    monkeypatch.setattr(retention, 'retain_robot_free_readiness',
                        lambda **kwargs: calls.append(kwargs) or {'status': 'passed', 'retained': True})
    result = reconcile_preparation_completion(**args)
    published = json.loads(result_path.read_text())
    assert calls == [{'run_root': run, 'revision_reference': published['configured_scene_revision_reference'],
                      'revision_digest': published['configured_scene_revision_digest']}]
    assert result['state']['scene_preparation_completion']['robot_free_readiness'] == {
        'status': 'passed', 'retained': True}


def test_a_transient_retention_failure_keeps_completion_pending_for_retry(prepared, monkeypatch):
    args, _, _ = prepared

    def unavailable(**_kwargs):
        raise retention.RobotFreeReadinessRetentionError('robot_free_readiness_retention_failed:OSError:x')

    monkeypatch.setattr(retention, 'retain_robot_free_readiness', unavailable)
    result = reconcile_preparation_completion(**args)
    assert result['status'] == 'awaiting_execution'
    assert result['blockers'] == ['robot_free_readiness_retention_failed:OSError:x']


@pytest.mark.parametrize('missing', ['webapp_sync_succeeded.json', 'post_teardown_provider_zero_receipt.json'])
def test_scene_cannot_finish_without_delivery_and_resource_closure(prepared, missing):
    args, run, _ = prepared
    (run / missing).unlink()
    assert reconcile_preparation_completion(**args)['status'] == 'awaiting_execution'


def test_asset_only_success_does_not_complete_scene(prepared):
    args, run, result_path = prepared
    result = json.loads(result_path.read_text())
    result['configured_scene_published'] = False
    save(result_path, result, 'result_digest')
    assert reconcile_preparation_completion(**args)['status'] != 'completed'


def test_another_owner_scene_is_not_adopted(prepared):
    args, run, _ = prepared
    profile = json.loads((run / 'launch_profile.json').read_text())
    profile['scene_attempt_binding']['intent_digest'] = 'sha256:' + 'f' * 64
    save(run / 'launch_profile.json', profile, 'profile_digest')
    assert reconcile_preparation_completion(**args) is None


def test_preparation_join_never_finishes_a_robot_evaluation(prepared):
    args, _, _ = prepared
    args['intent']['request']['execution'].pop('purpose')
    assert reconcile_preparation_completion(**args) is None


def _website_request():
    """What website scene preparation submits: its capture id, and a task named for the capture's task context."""
    value = preparation_request()
    value['submission_id'] = 'walkthrough-req-1'
    value['task']['task_id'] = 'website-0123456789abcdef0123'
    return value


def test_completion_enqueues_one_team_scene_request_and_launches_nothing(tmp_path, monkeypatch):
    import subprocess

    from blueprint_pipeline import team_scene_packet
    from blueprint_pipeline import team_scene_packet_worker as worker

    args, run, result_path = _prepare(tmp_path, monkeypatch, _website_request())
    queue = tmp_path / 'team-scene-packets'
    monkeypatch.setenv(worker.QUEUE_ROOT_ENV, str(queue))

    def forbidden(*_args, **_kwargs):
        pytest.fail('scene preparation completion must not compile or start anything')

    monkeypatch.setattr(team_scene_packet, 'prepare_team_scene', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    result = reconcile_preparation_completion(**args)
    assert result['status'] == 'completed' and result['phase'] == 'scene_prepared'
    queued = result['state']['scene_preparation_completion']['team_scene_request']
    assert queued['status'] == 'enqueued'
    pending = sorted((queue / 'pending').glob('*.json'))
    assert [path.stem for path in pending] == [queued['request_key']]
    # One sealed request: the capture, the launch its probe receipt sits beside, and the published revision.
    request = worker.validated_request(json.loads(pending[0].read_text()))
    published = json.loads(result_path.read_text())
    assert request['request_digest'] == queued['request_digest']
    assert (request['capture_id'], request['task_id']) == ('walkthrough-req-1', 'website-0123456789abcdef0123')
    assert request['source_launch_root'] == str(run.resolve())
    assert request['scene_revision_digest'] == published['configured_scene_revision_digest']
    assert request['robot_free_readiness_status'] == 'probe_unavailable'
    # A retried join never queues the same scene revision twice.
    again = reconcile_preparation_completion(**args)
    assert again['state']['scene_preparation_completion']['team_scene_request']['status'] == 'already_enqueued'
    assert len(list((queue / 'pending').glob('*.json'))) == 1
    assert not list((queue / 'processing').iterdir()) and not list((queue / 'processed').iterdir())


@pytest.mark.parametrize('case', ['not_a_website_scene', 'receipt_not_retained', 'probe_failed',
                                  'queue_not_configured'])
def test_completion_queues_no_team_scene_request_without_a_website_scene_and_its_retained_receipt(
        tmp_path, monkeypatch, case):
    from blueprint_pipeline import team_scene_packet_worker as worker

    args, _run, _ = _prepare(tmp_path, monkeypatch,
                             preparation_request() if case == 'not_a_website_scene' else _website_request())
    if case == 'receipt_not_retained':
        monkeypatch.setattr(retention, 'retain_robot_free_readiness',
                            lambda **_kwargs: {'status': 'refused', 'reason': 'x', 'retained': False})
    if case == 'probe_failed':
        monkeypatch.setattr(retention, 'retain_robot_free_readiness',
                            lambda **_kwargs: {'status': 'failed', 'retained': True, 'write': 'written'})
    queue = tmp_path / 'team-scene-packets'
    if case == 'queue_not_configured':
        monkeypatch.delenv(worker.QUEUE_ROOT_ENV, raising=False)
    else:
        monkeypatch.setenv(worker.QUEUE_ROOT_ENV, str(queue))
    result = reconcile_preparation_completion(**args)
    assert result['status'] == 'completed'
    assert result['state']['scene_preparation_completion']['team_scene_request'] == {
        'not_a_website_scene': {'status': 'not_a_website_scene'},
        'receipt_not_retained': {'status': 'not_enqueued', 'reason': 'robot_free_readiness_not_retained'},
        'probe_failed': {'status': 'not_enqueued', 'reason': 'robot_free_probe_failed'},
        'queue_not_configured': {'status': 'not_configured'}}[case]
    assert not list(queue.glob('pending/*.json'))
