"""Preparation completion needs publication/readback/closure, not robot controls."""
import json

import pytest

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


@pytest.fixture
def prepared(tmp_path):
    root, result_path = _source(tmp_path)
    run = root / 'scene-839873-qualifying'
    intent_root = tmp_path / 'intents'
    staged = intake.stage_scene_intent(value=preparation_request(), queue_root=intent_root,
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


def test_scene_preparation_finishes_without_robot_or_policy_result(prepared):
    args, run, _ = prepared
    result = reconcile_preparation_completion(**args)
    assert result['status'] == 'completed' and result['phase'] == 'scene_prepared'
    assert result['state']['scene_preparation_completion']['robot_evaluation_performed'] is False
    assert not (run / 'policy_canary_result_projection.json').exists()


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
