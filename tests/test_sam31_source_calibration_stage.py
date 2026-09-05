"""Hardware child continuation never duplicates allocation while billing posts."""
from __future__ import annotations

from pathlib import Path
import json

import pytest

from blueprint_pipeline import sam31_source_calibration_stage as stage
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.source_calibration_render_return import record
from tests.test_task_evaluation_sam31_preparation_profile import inputs  # noqa: F401


def test_profile_freezes_hardware_route_and_local_default(inputs):  # noqa: F811
    from blueprint_pipeline.task_evaluation_sam31_preparation_profile import materialize_sam31_preparation_profile
    local = materialize_sam31_preparation_profile(**inputs)
    gpu = materialize_sam31_preparation_profile(**inputs, calibrated_views_execution_site='provider_gpu')
    assert local['calibrated_views']['execution_site'] == 'control_plane'
    assert gpu['calibrated_views']['hardware_required'] is True
    assert gpu['calibrated_views']['hard_ttl_seconds'] == 1800
    assert gpu['profile_digest'] != local['profile_digest']


def test_hardware_child_waits_for_posted_billing_without_duplicate_allocator(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_sam31_preparation_cpu_stages as cpu
    from scripts import issue_retained_scene_render_paid_attempt_authority as issuer
    root = tmp_path/'output'
    root.mkdir()
    prepared_path = tmp_path/'prepared.json'
    prepared = {'preparation_digest': ''}
    prepared['preparation_digest'] = canonical_digest(prepared, digest_field='preparation_digest')
    prepared_path.write_text(json.dumps(prepared))
    task_path = tmp_path/'task.json'
    task_path.write_text(json.dumps({'human_authority': {'accepted_by': 'fixture-owner'}}))
    source = tmp_path/'source.ply'
    source.write_bytes(b'hermetic mock allocator input; actual packet tests cover source bytes')
    calls = []
    def prepare(job, **kwargs):
        calls.append('prepare')
        assert kwargs == {'prepare_hardware_render': True}
        return {'prepared_inputs': record(prepared_path), 'calibrated_view_request': record(task_path)}
    def build(**kwargs):
        calls.append('bundle')
        kwargs['job_dir'].mkdir()
        (kwargs['job_dir']/stage.RECEIPT_NAME).write_text('{}')
    def allocate(argv, **kwargs):
        calls.append('allocate')
        assert argv[argv.index('--probe-kind')+1] == 'adp-retained-scene-gpu-render'
        assert argv[argv.index('--adp-retained-scene-render-hard-ttl-seconds')+1] == '1800'
        result = {'status': 'completed', 'render_scope': 'source_calibration',
                  'source_calibration_return': {'return_path': str(prepared_path)}}
        Path(argv[argv.index('--adapter-output')+1]).write_text(json.dumps(result))
        return 0
    monkeypatch.setattr(cpu, 'execute_cpu_stage', prepare)
    monkeypatch.setattr(stage, 'build_source_calibration_gpu_render_bundle', build)
    monkeypatch.setattr(issuer, 'issue_paid_attempt_authority', lambda **_: {})
    monkeypatch.setattr(stage, 'verify_source_calibration_return', lambda *_: {})
    monkeypatch.setattr(stage, '_posted_charge', lambda *_: None)
    job = {'output_root': str(root), 'repo_root': str(tmp_path), 'runtime_root': str(tmp_path),
           'expected_source_commit': 'a'*40,
           'plan': {'host_inputs': {'task_request': record(task_path)}},
           'inputs': {'standard_splat_conversion_receipt': record(source), 'source_appearance': record(source)},
           'server_profile': {'approved_paid_input_roots': [str(tmp_path)], 'calibrated_views': {
               'execution_site': 'provider_gpu', 'hardware_required': True, 'max_spend_usd': 1.0,
               'hard_ttl_seconds': 1800, 'max_hourly_rate_usd': .5, 'retry_cap': 0,
               'maximum_resource_count': 1, 'allowed_geolocation_country_codes': ['US']}}}
    first = stage.execute_source_calibration_stage(job, allocator_runner=allocate)
    second = stage.execute_source_calibration_stage({**job, 'resume_only': True}, allocator_runner=allocate)
    assert first == second and first['status'] == 'waiting_for_external_result'
    assert first['waiting_reason'] == 'official_vast_billing_not_posted'
    assert calls == ['prepare', 'bundle', 'allocate']
    (root/'allocator_result.json').unlink()
    with pytest.raises(ValueError, match='prior_allocation_requires_reconciliation'):
        stage.execute_source_calibration_stage({**job, 'resume_only': True}, allocator_runner=allocate)
    assert calls == ['prepare', 'bundle', 'allocate']
