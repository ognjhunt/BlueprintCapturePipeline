from copy import deepcopy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import native_task_arena_policy_canary_worker as worker


def setup_children():
    cells = [{'cell_id': str(i), 'seed': i, 'family': 'bounded_physics',
              'cell_spec_digest': str(i), 'resolved_scenario': {},
              'resolved_scenario_digest': canonical_digest({})} for i in range(2)]
    inputs = {'candidate_ids': ['team.policy'], 'cells': cells,
              'task_success_contract': {}, 'task_success_contract_digest': 'contract'}
    children = []
    for index, cell in enumerate(cells):
        child = {'selected_cell_index': index, 'task_success_contract': {},
                 'task_success_contract_digest': 'contract',
                 'status': 'runtime_selected_cell_completed_pending_aggregation',
                 'episodes': [{**cell, 'candidate_id': 'team.policy', 'status': 'completed'}]}
        children.append(child)
    children[1].update(status='blocked', episodes=[], policy_loads=[],
        candidate_policy_queried=False, preload_observation_gate=None,
        session_failure_type='ValueError', session_failure_source_trace=[
            {'file': 'worker.py', 'function': 'prepolicy_observation_gate', 'line': 1},
            {'file': 'native_rigid_friction_scenario.py', 'function': '_source_material', 'line': 36}],
        session_closeout={'runtime_closed': True}, telemetry={'message_count': 0,
            'channel_message_counts': {'observations': 0, 'episode_envelopes': 0}})
    for child in children:
        child['result_digest'] = canonical_digest(child, digest_field='result_digest')
    return inputs, children


def test_partial_batch_preserves_completed_rows_and_raw_failure_without_inventing_episode(tmp_path, monkeypatch):
    inputs, children = setup_children()
    original = deepcopy(children)
    monkeypatch.setattr(worker, '_write_policy_canary_telemetry', lambda *args, **kwargs: ({}, []))
    result = worker._aggregate_isolated_cell_results(authority={}, inputs=inputs,
        child_results=children, output_root=tmp_path, construction_lineage_mode='diagnostic')
    assert children == original
    assert result['status'] == 'runtime_partial_pending_closeout'
    assert result['episodes_per_policy'] == 1 and len(result['episodes']) == 1
    assert result['planned_episodes_per_policy'] == 2
    assert result['isolated_cell_failures'][0]['raw_child_result'] == children[1]
    assert result['episodes'][0]['cell_id'] == '0'
    assert result['result_digest'] == canonical_digest(result, digest_field='result_digest')


@pytest.mark.parametrize('fault', ['digest', 'not_closed', 'queried', 'loaded', 'observed'])
def test_partial_batch_cannot_silently_drop_unproven_policy_attempt(tmp_path, monkeypatch, fault):
    inputs, children = setup_children()
    child = children[1]
    if fault == 'not_closed':
        child['session_closeout']['runtime_closed'] = False
    if fault == 'queried':
        child['candidate_policy_queried'] = True
    if fault == 'loaded':
        child['policy_loads'] = [{'candidate_id': 'team.policy'}]
    if fault == 'observed':
        child['telemetry']['channel_message_counts']['observations'] = 1
    child['result_digest'] = canonical_digest(child, digest_field='result_digest')
    if fault == 'digest':
        child['session_failure_type'] = 'changed'
    monkeypatch.setattr(worker, '_write_policy_canary_telemetry', lambda *args, **kwargs: ({}, []))
    with pytest.raises(RuntimeError, match='isolated_cell_result_invalid'):
        worker._aggregate_isolated_cell_results(authority={}, inputs=inputs,
            child_results=children, output_root=tmp_path, construction_lineage_mode='diagnostic')


@pytest.mark.parametrize('fault', [None, 'archive', 'manifest', 'digest', 'runtime', 'unknown_failure'])
def test_historical_partial_adoption_requires_original_archive_manifest_and_prepolicy_failure(tmp_path, monkeypatch, fault):
    import hashlib
    import json
    import zipfile
    from scripts import stage_internal_team_evaluation_resume as staging
    from blueprint_pipeline import native_task_arena_policy_canary_session as session

    source = tmp_path / 'stage'
    attempt = source / 'allocator/attempts/attempt_001'
    evidence = attempt / 'immutable_execution'
    provider = attempt / 'vast_provider_run'
    provider.mkdir(parents=True)
    inputs, children = setup_children()
    inputs['runtime_inputs_digest'] = 'inputs'
    children[1]['session_failure_source_trace'] = [
        {'file': 'worker.py', 'function': 'prepolicy_observation_gate'},
        {'file': 'native_task_arena_runtime.py', 'function': 'build_native_task_arena_environment'},
        {'file': 'native_rigid_friction_scenario.py', 'function': 'prepare'},
        {'file': 'native_rigid_friction_scenario.py', 'function': '_source_material'}]
    if fault == 'unknown_failure':
        children[1]['session_failure_source_trace'][-1]['function'] = 'unknown'
    if fault == 'runtime':
        inputs['runtime_inputs_digest'] = 'changed'
    records = []
    archive = provider / 'vast_provider_runtime_output.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        for i, child in enumerate(children):
            child['construction_lineage_mode'] = 'diagnostic'
            child['result_digest'] = canonical_digest(child, digest_field='result_digest')
            if fault == 'digest' and i == 1:
                child['result_digest'] = 'stale'
            data = json.dumps(child).encode()
            relative = f'cell_runs/{i:02d}/{session.PROVIDER_RESULT_FILENAME}'
            path = evidence / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(data)
            bundle.writestr(relative, b'changed' if fault == 'archive' and i == 1 else data)
            records.append({'relative_path': 'immutable_execution/' + relative,
                'size_bytes': len(data), 'sha256': 'sha256:' + hashlib.sha256(data).hexdigest()})
    if fault == 'manifest':
        records[1]['sha256'] = 'changed'
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    write(provider / 'vast_provider_output_download_manifest.json', {'status': 'completed',
        'output_zip_path': str(archive), 'output_zip_size_bytes': archive.stat().st_size})
    manifest = attempt / 'artifact_manifest.json'
    write(manifest, {'binding': {'bundle_sha256': 'bundle'}, 'files': records})
    write(source / 'runtime_inputs/policy_canary_runtime_inputs.json', inputs)
    write(source / 'runtime_inputs/policy_canary_session_authority.json', {'runtime_inputs_digest': 'inputs'})
    write(evidence / session.PROVIDER_RESULT_FILENAME, {'status': 'blocked', 'episodes': []})
    adapter = {'native_control_result_path': str(evidence / session.PROVIDER_RESULT_FILENAME),
               'bundle_sha256': 'bundle', 'artifact_manifest_path': str(manifest)}
    write(source / 'allocator_result.json', adapter)
    monkeypatch.setattr(session, 'validate_runtime_input_manifest', lambda value: value)
    monkeypatch.setattr(session, 'validate_session_authority', lambda value: value)
    monkeypatch.setattr(worker, '_write_policy_canary_telemetry', lambda *args, **kwargs: ({}, []))
    before = (evidence / session.PROVIDER_RESULT_FILENAME).read_bytes()
    if fault:
        with pytest.raises(ValueError, match='diagnostic_partial'):
            staging.recover_partial({'root': str(source), 'cells': inputs['cells']}, adapter, tmp_path / 'derived')
    else:
        shard, derived_adapter, result = staging.recover_partial(
            {'root': str(source), 'cells': inputs['cells']}, adapter, tmp_path / 'derived')
        assert len(shard['cells']) == len(result['episodes']) == 1
        assert shard['cell_count'] == 1 and shard['original_planned_cell_count'] == 2
        assert result['isolated_cell_failures'][0]['raw_child_result'] == children[1]
        assert derived_adapter['native_control_result_digest'] == result['result_digest']
        assert result['partial_adoption']['provider_mutations'] == 0
        repeated = staging.recover_partial({'root': str(source), 'cells': inputs['cells']}, adapter, tmp_path / 'derived')
        assert repeated[2] == result
        changed = deepcopy(result)
        changed['episodes'][0]['status'] = 'substituted'
        changed['result_digest'] = canonical_digest(changed, digest_field='result_digest')
        write(tmp_path / 'derived' / session.PROVIDER_RESULT_FILENAME, changed)
        with pytest.raises(ValueError, match='existing_adoption_changed'):
            staging.recover_partial({'root': str(source), 'cells': inputs['cells']}, adapter, tmp_path / 'derived')
    assert (evidence / session.PROVIDER_RESULT_FILENAME).read_bytes() == before
