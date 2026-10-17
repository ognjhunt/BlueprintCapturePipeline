"""The native status inspector retains bounded metadata, never all episode trees."""
import os
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile

import pytest

from blueprint_pipeline.wam_provider_output import inspect_provider_runtime_output_zip

NAME = 'native_task_arena_policy_canary_session_result.v1.json'


@pytest.mark.slow
@pytest.mark.skipif(os.name != 'posix', reason='Process memory bound uses POSIX resource limits')
def test_large_native_json_is_inspected_under_a_bounded_process_memory_budget(tmp_path):
    path = tmp_path / 'dense.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        with archive.open(NAME, 'w', force_zip64=True) as stream:
            stream.write(b'{"status":"blocked","blockers":["retained"],"episodes":[')
            block = b'"' + b'x' * (1024 * 1024) + b'"'
            for index in range(200):
                if index:
                    stream.write(b',')
                stream.write(block)
            stream.write(b']}')
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.native_task_arena_policy_canary_session import PROVIDER_STATUS_FILENAME, PROVIDER_STATUS_SCHEMA
    with zipfile.ZipFile(path) as archive:
        digest = hashlib.sha256()
        with archive.open(NAME) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        size = archive.getinfo(NAME).file_size
    summary = {'schema_version': PROVIDER_STATUS_SCHEMA, 'run_kind': 'team_candidate_evaluation',
        'inspection_scope': 'hash_bound_status_metadata_only', 'claim_ceiling': 'diagnostic_policy_execution',
        'status': 'blocked', 'blockers': ['retained'], 'source_result': {'file_name': NAME, 'size_bytes': size,
        'sha256': 'sha256:' + digest.hexdigest(), 'result_digest': 'unqualified_capacity_fixture'}}
    summary['result_digest'] = canonical_digest(summary, digest_field='result_digest')
    with zipfile.ZipFile(path, 'a') as archive:
        archive.writestr(PROVIDER_STATUS_FILENAME, json.dumps(summary))
    program = '''import json, resource, sys
from blueprint_pipeline.wam_provider_output import inspect_provider_runtime_output_zip
if sys.platform != 'darwin':
    resource.setrlimit(resource.RLIMIT_AS, (128 * 1024**2, 128 * 1024**2))
result = inspect_provider_runtime_output_zip(__import__('pathlib').Path(sys.argv[1]))
assert result['runtime_result_present'], result
assert result['runtime_result_status'] == 'blocked', result
assert result['runtime_result_blockers'] == ['retained'], result
assert result['runtime_result_json_validated'] is False, result
assert result['runtime_result_parse_scope'] == 'hash_bound_status_metadata_only', result
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
assert peak < 128 * 1024**2, peak
print(json.dumps({'status': result['runtime_result_status'], 'peak_rss_bytes': peak}))
'''
    done = subprocess.run([sys.executable, '-c', program, str(path)], capture_output=True, text=True, timeout=90,
                          env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')})
    assert done.returncode == 0, done.stderr + done.stdout


def test_native_summary_validates_skipped_json_and_refuses_oversized_terminal(tmp_path):
    invalid = tmp_path / 'invalid.zip'
    with zipfile.ZipFile(invalid, 'w') as archive:
        archive.writestr(NAME, '{"status":"completed","episodes":[truefalse]}')
    result = inspect_provider_runtime_output_zip(invalid)
    assert result['runtime_result_present'] is False and result['json_parse_errors']
    # The old large allowance is no longer admitted by the reader or producer.
    large = tmp_path / 'large.zip'
    with zipfile.ZipFile(large, 'w') as archive:
        info = zipfile.ZipInfo(NAME)
        info.file_size = 512 * 1024**2 + 1
        # A fake central-directory size is enough: no decompression or allocation is permitted.
        archive.writestr(NAME, '{}')
    with zipfile.ZipFile(large) as archive:
        archive.getinfo(NAME).file_size = 512 * 1024**2 + 1
        from blueprint_pipeline.wam_provider_output import inspect_provider_runtime_output_archive
        result = inspect_provider_runtime_output_archive(archive, zip_path=str(large), zip_size_bytes=large.stat().st_size)
    assert result['runtime_result_present'] is False and result['json_parse_errors']


def test_hash_bound_status_refuses_tampered_source_bytes(tmp_path):
    from blueprint_pipeline import native_task_arena_policy_canary_worker as worker
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    root = tmp_path / NAME
    payload = {'run_kind': 'team_candidate_evaluation', 'status': 'blocked', 'blockers': ['known']}
    payload['result_digest'] = canonical_digest(payload, digest_field='result_digest')
    worker._seal_result(result_path=root, result=payload)
    path = tmp_path / 'tampered.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        for source in tmp_path.glob('*.json'):
            data = source.read_bytes()
            if source.name == NAME:
                data = data.replace(b'known', b'other')
            archive.writestr(source.name, data)
    result = inspect_provider_runtime_output_zip(path)
    assert result['runtime_result_present'] is False and result['json_parse_errors']
