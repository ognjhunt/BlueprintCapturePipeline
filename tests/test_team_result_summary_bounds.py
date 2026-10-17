"""Bound team terminal memory while retaining full, digest-bound raw evidence."""
import hashlib
import json

import pytest

from blueprint_pipeline import native_task_arena_policy_canary_worker as worker
from blueprint_pipeline import team_candidate_quick10_result as result
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_team_evaluation_single_session import _campaign
from tests.test_vast_policy_canary_terminal_retention import _archive


@pytest.mark.slow
def test_200_large_raw_children_produce_a_small_terminal_and_keep_original_digests(tmp_path, monkeypatch):
    runtime, output, _inputs, _authority, seen, spawn = _campaign(tmp_path, monkeypatch)
    raw_bytes = []
    checkpoint = worker._team_checkpoint
    def bounded_checkpoint(**kwargs):
        for child in kwargs['children'][:-1]:
            assert 'raw_query_history' not in child['episodes'][0]['episode']
        return checkpoint(**kwargs)
    monkeypatch.setattr(worker, '_team_checkpoint', bounded_checkpoint)
    def large_spawn(**kwargs):
        rc = spawn(**kwargs)
        path = kwargs['child_root'] / worker.PROVIDER_RESULT_FILENAME
        child = json.loads(path.read_text())
        child['episodes'][0]['episode']['raw_query_history'] = 'x' * (1024 * 1024)
        child['result_digest'] = canonical_digest(child, digest_field='result_digest')
        worker._seal_result(result_path=path, result=child)
        raw_bytes.append(path.stat().st_size)
        return rc
    assert worker._run_isolated_cell_processes(runtime_root=runtime, output_root=output, run_cell_process=large_spawn) == 0
    assert len(seen) == 200 and sum(raw_bytes) > 200 * 1024**2
    terminal_path = output / worker.PROVIDER_RESULT_FILENAME
    assert terminal_path.stat().st_size < 4 * 1024**2
    terminal = json.loads(terminal_path.read_text())
    checkpoint = json.loads((output / 'team_evaluation_checkpoint.json').read_text())
    for row, ref in zip(terminal['episodes'], checkpoint['completed_child_results'], strict=True):
        assert 'raw_query_history' not in row['episode']
        source = row['source_child_result']
        path = output / source['relative_path']
        data = path.read_bytes()
        assert source['sha256'] == 'sha256:' + hashlib.sha256(data).hexdigest()
        original = json.loads(data)
        assert len(original['episodes'][0]['episode']['raw_query_history']) == 1024**2
        assert ref['result_digest'] == original['result_digest'] == source['result_digest']
        assert row['episode']['score'] == original['episodes'][0]['episode']['score']
    # The actual unchanged 512 MiB archiver retains all raw children and compact terminal.
    archive_root = tmp_path / 'archive'
    archive_root.mkdir()
    (archive_root / 'runtime_output').symlink_to(output, target_is_directory=True)
    archived = _archive(archive_root)
    assert archived.returncode == 0, archived.stderr
    from blueprint_pipeline.wam_provider_output import inspect_provider_runtime_output_zip
    from blueprint_pipeline.adp_isaac_lab_arena_vast import _extract_provider_output
    archive_path = archive_root / 'adp_arena_provider_runtime_output.zip'
    inspection = inspect_provider_runtime_output_zip(archive_path)
    assert inspection['runtime_result_parse_scope'] == 'hash_bound_status_metadata_only'
    assert inspection['runtime_result_json_validated'] is False
    extracted = _extract_provider_output(archive_path, tmp_path / 'extracted', result_name=worker.PROVIDER_RESULT_FILENAME)
    assert extracted['status'] == 'completed' and len(extracted['execution']['episodes']) == 200


def test_projected_cycle_time_loads_only_the_bound_per_case_trace(tmp_path):
    trace = {'task_state_samples': [{'simulation_time_s': 1.0}, {'simulation_time_s': 6.5}]}
    trace['trace_digest'] = canonical_digest(trace, digest_field='trace_digest')
    path = tmp_path / 'cell_runs/00/trace.json'
    path.parent.mkdir(parents=True)
    data = json.dumps(trace, sort_keys=True).encode()
    path.write_bytes(data)
    row = {'episode': {'state_trace': trace}, 'state_trace_digest': canonical_digest({'value': trace})}
    expected = result._cycle_seconds(row)
    compact = {**row, 'episode': {}, 'episode_summary_kind': 'team_terminal_episode_summary.v1',
               'evidence_artifacts': {'state_trace': {'relative_path': 'cell_runs/00/trace.json',
                   'size_bytes': len(data), 'sha256': 'sha256:' + hashlib.sha256(data).hexdigest()}}}
    def loader(value):
        return result.load_bound_state_trace(value, roots=[tmp_path])
    assert expected == result._cycle_seconds(compact, state_trace_loader=loader) == 5.5
    path.write_bytes(data.replace(b'6.5', b'7.5'))
    assert result._cycle_seconds(compact, state_trace_loader=loader) is None


def test_trace_reference_cannot_escape_the_extracted_result_root(tmp_path):
    row = {'evidence_artifacts': {'state_trace': {'relative_path': '../private.json'}}}
    assert result.load_bound_state_trace(row, roots=[tmp_path]) == {}


def test_one_gpu_dispatch_scores_projected_results_using_bound_trace_files(tmp_path, monkeypatch):
    from blueprint_pipeline import team_candidate_quick10_vast as transport
    from tests.test_team_candidate_quick10_launch import _world, _staged_run_reported
    world = _world(tmp_path, monkeypatch)
    provider = world['provider']
    def projected_provider(**kwargs):
        response = provider(**kwargs)
        native_path = __import__('pathlib').Path(response['native_control_result_path'])
        payload = json.loads(native_path.read_text())
        rows = []
        for index, row in enumerate(payload['episodes']):
            child_root = native_path.parent / 'cell_runs' / f'{index:02d}'
            child_root.mkdir(parents=True)
            trace = {'task_state_samples': [{'simulation_time_s': 1.0}, {'simulation_time_s': 6.5}]}
            trace['trace_digest'] = canonical_digest(trace, digest_field='trace_digest')
            data = json.dumps(trace).encode()
            (child_root / 'trace.json').write_bytes(data)
            row['episode']['state_trace'] = trace
            row['state_trace_digest'] = canonical_digest({'value': trace})
            row.setdefault('evidence_artifacts', {})['state_trace'] = {'relative_path': 'trace.json',
                'size_bytes': len(data), 'sha256': 'sha256:' + hashlib.sha256(data).hexdigest()}
            child = {'episodes': [row], 'result_digest': 'temporary'}
            child['result_digest'] = canonical_digest(child, digest_field='result_digest')
            worker._seal_result(result_path=child_root / worker.PROVIDER_RESULT_FILENAME, result=child)
            compact = worker._compact_team_child_result(child, child_root=child_root, output_root=native_path.parent)
            rows.append(worker._prefix_episode_evidence_paths(compact['episodes'][0], prefix=f'cell_runs/{index:02d}'))
        payload['episodes'] = rows
        payload['result_digest'] = canonical_digest(payload, digest_field='result_digest')
        native_path.write_text(json.dumps(payload))
        response['native_control_result_digest'] = payload['result_digest']
        return response
    monkeypatch.setattr(transport, 'run_arena_native_control_vast', projected_provider)
    staged, reported = _staged_run_reported(world)
    assert staged['staged'] == 1 and reported['completed'] == 1
    assert len(provider.calls) == 1
    terminal = json.loads((world['job_dir'] / 'team_candidate_quick10_terminal.json').read_text())
    assert terminal['median_cycle_seconds'] == 5.5
    assert terminal['episodes_run'] == 200
