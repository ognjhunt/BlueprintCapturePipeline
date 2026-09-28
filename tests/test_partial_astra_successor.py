"""A new run inherits exact completed CAD work and all original inference costs."""
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_partial_astra_successor as partial
from blueprint_pipeline import task_evaluation_scene_configuration_astra_phase_adoption as adoption
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_object_astra_authoring import AssetAuthoringError, budgeted_invoker
from blueprint_pipeline.task_object_astra_inherited_inference import inherited_balance
from tests.test_astra_automatic_resume import authoring_fixture, execute, FixtureInvoker  # noqa: F401
from tests.test_task_evaluation_scene_configuration_astra_driver import retained, component  # noqa: F401


@pytest.fixture
def successor_fixture(authoring_fixture):  # noqa: F811
    f = authoring_fixture
    with pytest.raises(RuntimeError, match='interrupted'):
        execute(f, FixtureInvoker(f.request, f.runtime / 'inference',
                fail='independent_visual_review_1', first_review_passed=False), f.runtime / 'authoring')
    request = f.request.model_dump(mode='json')
    request['run_id'] += '-successor'
    request['expected_production_commit'] = 'b' * 40
    request['request_digest'] = canonical_digest(request, digest_field='request_digest')
    old_binding = json.loads((f.runtime / 'stage_source_binding.json').read_text())
    binding = dict(old_binding, run_id=request['run_id'], authoring_input_digest=canonical_digest({
        k: v for k, v in request.items() if k not in {'request_digest', 'expected_production_commit'}}))
    binding['binding_digest'] = canonical_digest(binding, digest_field='binding_digest')
    identity = {'source_run_id': f.request.run_id, 'successor_run_id': request['run_id'],
                'owner_id': 'fixture-owner', 'stable_intent_id': 'fixture-intent',
                'stable_intent_digest': 'sha256:' + 'c' * 64}
    descriptor = {'schema_version': partial.SCHEMA_VERSION,
        'source_run_id': f.request.run_id, 'successor_run_id': request['run_id'],
        'source_request_digest': f.request.request_digest,
        'original_runtime_root': str(f.runtime),
        'semantic_request_digest': canonical_digest(partial.semantic_request(request)),
        'source_stage_binding_digest': old_binding['binding_digest'],
        'owner_intent_lineage': identity, 'retained_files': adoption._inventory(f.runtime)}
    descriptor['adoption_digest'] = canonical_digest(descriptor, digest_field='adoption_digest')
    return f, request, binding, identity, descriptor


def prepare(data, budget):
    f, request, binding, identity, descriptor = data
    return partial.prepare_partial_astra_successor(value=descriptor, request_value=request,
        source_binding=binding, verified_lineage=identity, package=f.package, budget_root=budget)


def test_successor_runs_only_missing_second_review_and_conserves_original_receipts(successor_fixture):
    from blueprint_pipeline.task_object_astra_authoring import AuthoringRequest
    f, request, _, _, _ = successor_fixture
    before = adoption._inventory(f.runtime)
    output = f.runtime.parent / 'successor'
    prepared = prepare(successor_fixture, output / 'inference')
    inherited = inherited_balance(output / 'inference', request['run_id'])
    assert inherited['call_count'] == prepared['prior_call_count'] > 0
    assert inherited['cost_usd'] == prepared['retained_inference_cost_usd'] > 0
    for path in (output / 'inference/inherited').rglob('completed/*.json'):
        assert json.loads(path.read_text())['run_id'] == f.request.run_id
    next_invoker, _ = budgeted_invoker(root=output / 'inference', run_id=request['run_id'], maximum_cost_usd=15)
    assert next_invoker._reserved_cost_usd == inherited['cost_usd']
    f.request = AuthoringRequest.model_validate(request)
    invoker = FixtureInvoker(f.request, output / 'inference')
    result = execute(f, invoker, output / 'authoring', **prepared['authoring_kwargs'])
    assert result['status'] == 'candidate_authored_pending_native_qualification'
    assert invoker.calls == ['independent_visual_review_1']
    assert f.executed == ['cad', 'blender', 'blender']
    assert adoption._inventory(f.runtime) == before
    # The driver retains the same source archives in every attempt.
    for name in adoption.SOURCE_ARCHIVES:
        (output / name).write_bytes((f.package / name).read_bytes())
    # Another retry validates original adopted phases and carries their costs again.
    descriptor = adoption.materialize_automatic_phase_adoption(prior_runtime=output)
    resumed = output.parent / 'same-successor-retry'
    repeated = adoption.prepare_phase_adoption(value=descriptor, request_value=request,
        package=f.package, budget_root=resumed / 'inference')
    assert repeated['prior_call_count'] == prepared['prior_call_count'] + 1
    assert repeated['retained_inference_cost_usd'] > inherited['cost_usd']


@pytest.mark.parametrize('change', ['owner', 'intent', 'geometry', 'source', 'rights', 'configuration', 'unknown'])
def test_successor_changed_identity_or_science_fails_before_new_inference(successor_fixture, change):
    f, request, binding, identity, descriptor = successor_fixture
    if change == 'owner':
        identity['owner_id'] = 'different-owner'
        # Separate admitted identity must not be controlled by descriptor aliasing.
        descriptor['owner_intent_lineage'] = dict(identity, owner_id='fixture-owner')
    elif change == 'intent':
        identity['stable_intent_digest'] = 'sha256:' + 'd' * 64
        descriptor['owner_intent_lineage'] = dict(identity, stable_intent_digest='sha256:' + 'c' * 64)
    elif change == 'geometry':
        request['dimensions_m'][0] += .1
    elif change == 'source':
        request['source_frames'][0]['sha256'] = 'sha256:' + 'e' * 64
    elif change == 'rights':
        binding['rights_admission'] = dict(binding['rights_admission'], sha256='sha256:' + 'e' * 64)
    elif change == 'configuration':
        binding['configuration_sha256'] = 'sha256:' + 'e' * 64
    else:
        next((f.runtime / 'inference/inference_reservations/completed').glob('*.json')).unlink()
        descriptor['retained_files'] = adoption._inventory(f.runtime)
    descriptor['adoption_digest'] = canonical_digest(descriptor, digest_field='adoption_digest')
    target = f.runtime.parent / 'refused'
    with pytest.raises(AssetAuthoringError):
        prepare(successor_fixture, target / 'inference')
    assert not (target / 'inference').exists()


@pytest.mark.parametrize('defect', [None, 'relocation', 'archive_bytes', 'traversal', 'missing_file'])
def test_original_root_restoration_is_exact_and_fail_closed(successor_fixture, tmp_path, defect):
    import shutil
    import zipfile
    from blueprint_pipeline.task_object_astra_authoring import file_record
    f, request, _, identity, descriptor = successor_fixture
    before = adoption._inventory(f.runtime)
    archive = tmp_path / 'retained.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        for row in before:
            if defect == 'missing_file' and row == before[0]:
                continue
            bundle.write(f.runtime / row['relative_path'], row['relative_path'])
        if defect == 'traversal':
            bundle.writestr('../escape', b'forbidden')
    descriptor['retained_runtime_archive'] = file_record(archive)
    descriptor['adoption_digest'] = canonical_digest(descriptor, digest_field='adoption_digest')
    if defect == 'archive_bytes':
        archive.write_bytes(b'changed')
    original = f.runtime
    preserved = tmp_path / 'original-preserved'
    shutil.move(original, preserved)
    try:
        target = original if defect != 'relocation' else original.parent / 'different-root'
        if defect:
            with pytest.raises(AssetAuthoringError):
                partial.restore_partial_astra(value=descriptor, request_value=request,
                    original_root=target, verified_lineage=identity, archive_path=archive)
            assert not target.exists()
        else:
            partial.restore_partial_astra(value=descriptor, request_value=request,
                original_root=original, verified_lineage=identity, archive_path=archive)
            assert adoption._inventory(original) == before
            for row in before:
                assert (original / row['relative_path']).read_bytes() == (preserved / row['relative_path']).read_bytes()
    finally:
        if not original.exists():
            shutil.move(preserved, original)


def test_driver_restores_bound_input_before_discovery_and_reaches_pending_review(successor_fixture, tmp_path):
    import shutil
    import zipfile
    from blueprint_pipeline import task_evaluation_scene_configuration_astra_driver as driver
    from blueprint_pipeline.task_object_astra_authoring import file_record
    f, request, _, identity, descriptor = successor_fixture
    before = adoption._inventory(f.runtime)
    archive = tmp_path / 'runtime.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        for row in before:
            bundle.write(f.runtime / row['relative_path'], row['relative_path'])
    descriptor['retained_runtime_archive'] = file_record(archive)
    descriptor['adoption_digest'] = canonical_digest(descriptor, digest_field='adoption_digest')
    descriptor_path = tmp_path / 'descriptor.json'
    descriptor_path.write_text(json.dumps(descriptor))
    def bound(path):
        row = file_record(path)
        return {'path': path.name, 'digest': row['sha256'], 'size_bytes': row['size_bytes'],
                'materialized_path': str(path)}
    input_path = Path(f.component.environment[driver._INPUT_ENV])
    stage = json.loads(input_path.read_text())
    stage['run_id'] = request['run_id']
    stage['construction_envelope']['partial_astra_successor'] = {
        'descriptor': bound(descriptor_path), 'runtime_archive': bound(archive), 'verified_lineage': identity}
    stage['construction_envelope']['run_id'] = request['run_id']
    stage['construction_envelope']['envelope_digest'] = canonical_digest(
        stage['construction_envelope'], digest_field='envelope_digest')
    input_path.write_text(json.dumps(stage))
    shutil.move(f.runtime, tmp_path / 'preserved-original')
    class PendingReviewReached(Exception):
        pass
    def reached(**kwargs):
        assert kwargs['request_value']['run_id'] == request['run_id']
        assert kwargs['blender_adoption_record']['source_round_index'] == 1
        assert kwargs['adopted_blender_execution']['blender_execution_repeated'] is False
        assert kwargs['invoker'].prior_calls == 5
        assert kwargs['output_root'].parent.name == 'attempt-0001'
        assert kwargs['adopted_cad_result']['passed'] is True
        raise PendingReviewReached()
    with pytest.raises(PendingReviewReached):
        driver.execute_astra_component(**{**f.component.kwargs, 'authoring_executor': reached})
    assert adoption._inventory(f.runtime) == before
    assert 'sdk' not in f.component.events
    assert f.component.seen['completion']['provider_call_performed'] is False
