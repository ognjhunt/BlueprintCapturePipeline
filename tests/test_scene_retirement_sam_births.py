# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_sam31_preparation_execution.py
#   src/blueprint_pipeline/task_evaluation_scene_retirement_producer_births.py
"""ADP-009D/day28: native SAM first writes retain their original scene owner."""
import hashlib
import json
from pathlib import Path

from blueprint_pipeline import task_evaluation_sam31_preparation_execution as execution
from blueprint_pipeline import task_evaluation_scene_retirement_cache as cache
from tests.test_scene_retirement_normal_cache import owner_submission
from tests.test_scene_retirement_connected_acceptance import _raw, _sealed_file
from tests.test_task_evaluation_launch_preparation_worker import _rebind_recipe, fetcher, SERVICE_ACCOUNT


def owned_sam(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import stage_launch_preparation_request
    from blueprint_pipeline.task_evaluation_launch_preparation_contract import launch_preparation_request_digest
    _, policy, request, payloads, queue, inputs, proofs = owner_submission(tmp_path, monkeypatch, production=True)
    data = json.dumps({'schema_version': 'test_plan.v1', 'reviewer_kind': 'ai'}).encode()
    uri = 's3://blueprint-production-inputs/sam-plan.json'
    ref = dict(uri=uri, digest='sha256:' + hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    payloads[uri] = data
    request['runtime']['mounts'].append(dict(source=ref, container_path='/inputs/sam31-plan.json', mode='read_only'))
    recipe = json.loads(payloads[request['construction']['recipe']['uri']])
    stage_ref = recipe['stage_sequence'][0]['configuration']
    stage = dict(required_views={'mask_source': 'sam31_reviewed_calibrated_object_masks'},
                 sam31_review_kind='ai', sam31_preparation_plan=ref)
    data = json.dumps(stage).encode()
    payloads[stage_ref['uri']] = data
    stage_ref.update(digest='sha256:' + hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    _rebind_recipe(request, payloads, recipe)
    request_path = Path(proofs['submission_request_raw_ref']['path'])
    request_path.unlink()
    request_path.write_text(json.dumps(request))
    proofs['submission_request_raw_ref'] = _raw(request_path)
    factory_path = Path(proofs['factory_raw_ref']['path'])
    factory = json.loads(factory_path.read_bytes())
    factory_path.unlink()
    factory['submission_request'] = proofs['submission_request_raw_ref']
    _sealed_file(factory_path, factory, 'factory_digest')
    proofs['factory_raw_ref'] = _raw(factory_path)
    monkeypatch.setattr(cache.time, 'time', lambda: 101)
    cache.publish_preparation_storage_authority(queue_root=queue, request=request, now=101, **proofs)
    stage_launch_preparation_request(value=request, queue_root=queue, submitted_by='scene-progression')
    prepared = worker.process_launch_preparation_queue(queue_root=queue, input_root=inputs,
        allowed_uri_prefixes=['s3://blueprint-production-inputs/'], service_account=SERVICE_ACCOUNT,
        source_commit=request['expected_production_commit'], fetcher=fetcher(payloads),
        sam31_preparation_advancer=lambda context: {'status': 'waiting_for_child', 'evidence_refs': []})
    assert prepared['results'][0]['status'] == 'waiting_for_child', prepared
    source = inputs / 'original-source.json'
    source.write_text('{"source":"immutable"}')
    plan = inputs / 'content-addressed' / 'sha256' / ref['digest'][7:]
    child_queue = inputs / 'sam-queue'
    receipt = execution.enqueue_sam31_phase(queue_root=child_queue,
        parent_preparation_id=request['preparation_id'], parent_request_digest=launch_preparation_request_digest(request),
        expected_source_commit=request['expected_production_commit'], plan_ref=_raw(plan),
        phase='source_selections', inputs={'source': _raw(source)})
    monkeypatch.setattr(execution, '_verified_checkout_head', lambda: request['expected_production_commit'])
    args = dict(queue_root=child_queue, parent_queue_root=queue, preparation_input_root=inputs,
                execution_root=inputs / 'sam-outputs', approved_roots=(tmp_path,))
    return policy, proofs, request, receipt, args


def generation(policy, output):
    return json.loads((Path(policy['generation_store']) / (hashlib.sha256(str(output).encode()).hexdigest() + '.json')).read_bytes())


def test_native_sam_generation_precedes_first_executor_payload(tmp_path, monkeypatch):
    policy, proofs, _, _, args = owned_sam(tmp_path, monkeypatch)
    seen = []
    def executor(context):
        output = Path(context['output_root'])
        value = generation(policy, output)
        assert value['owner_raw_ref'] == proofs['intent_raw_ref']
        assert value['birth_request_raw_ref'] == proofs['attempt_raw_ref']
        assert value['ino'] == output.stat().st_ino
        assert list(output.iterdir()) == []
        seen.append(value)
        return {'status': 'waiting_for_external_result', 'artifacts': {}}
    result = execution.process_sam31_phase_queue(**args, phase_executor=executor)
    assert len(seen) == 1, result
    assert result['results'][0]['status'] == 'waiting_for_external_result'


def test_native_sam_recheck_preserves_exact_generation(tmp_path, monkeypatch):
    policy, _, _, _, args = owned_sam(tmp_path, monkeypatch)
    seen = []
    def executor(context):
        seen.append(generation(policy, Path(context['output_root'])))
        return {'status': 'waiting_for_external_result', 'artifacts': {}}
    first = execution.process_sam31_phase_queue(**args, phase_executor=executor)
    second = execution.process_sam31_phase_queue(**args, phase_executor=executor)
    assert len(seen) == 2, (first, second)
    assert seen[0] == seen[1]


def test_native_sam_changed_owner_refuses_before_executor(tmp_path, monkeypatch):
    policy, _, request, receipt, args = owned_sam(tmp_path, monkeypatch)
    parent = args['preparation_input_root'] / request['preparation_id']
    value = generation(policy, parent)
    value['owner_intent_id'] = 'foreign'
    path = Path(policy['generation_store']) / (hashlib.sha256(str(parent).encode()).hexdigest() + '.json')
    _sealed_file(path, value, 'state_digest', mode=0o600)
    called = []
    result = execution.process_sam31_phase_queue(**args,
        phase_executor=lambda context: called.append(context) or {'status': 'waiting_for_external_result', 'artifacts': {}})
    assert called == [], result
    assert result['results'][0]['status'] == 'failed'
    from blueprint_pipeline.task_evaluation_launch_preparation_contract import launch_preparation_request_digest
    assert not (args['execution_root'] / launch_preparation_request_digest(request)[7:] / receipt['child_id']).exists()
