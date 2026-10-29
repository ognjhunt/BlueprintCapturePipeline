"""An actual archived partial scene reaches a held job outcome before task cleaning."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.observed_room_contracts import read_room_limit, retains_observed_room_limit
from blueprint_pipeline.website_observed_room_limit import retain_observed_room_limit


def fixture(tmp_path):
    root = tmp_path / 'site-job/captures/walkthrough-job'
    root.mkdir(parents=True)
    source = root / 'original.mp4'
    source.write_bytes(b'synthetic source; no upload')
    context = {'scene_id': 'site-job', 'capture_id': 'walkthrough-job', 'context_digest': 'sha256:' + 'b' * 64}
    limit = {'schema_version': 'atlas_task_evidence_limitation.v1', 'status': 'abstained',
        'reason': 'atlas_scene_planner_abstained', 'original_task_context_digest': context['context_digest'],
        'source_video_digest': _sha256_file(source), 'planner_receipt_digest': 'sha256:' + 'c' * 64,
        'summary': 'No observed empty destination supports the original task.'}
    limit['digest'] = canonical_digest(limit, digest_field='digest')
    result = {'schema_version': 'website_atlas_pipeline_result.v1', 'observed_scene_only': True,
        'task_evidence_limitation': limit, 'source_video_digest': _sha256_file(source),
        'task_context_digest': context['context_digest'], 'task_and_surrounding_scene_covered': False,
        'simulation_qualified': False, 'physical_evidence': False, 'native_preparation_eligible': False,
        'execution_authority_granted': False, 'download': {'zip_sha256': 'sha256:' + 'd' * 64}}
    result['digest'] = canonical_digest(result, digest_field='digest')
    job = {'state': 'ready', 'result_digest': result['digest'],
        'archive': {'uri': 'b2://private-test/observed.zip', 'digest': result['download']['zip_sha256'],
                    'remote_verified_at': '2026-10-29T00:00:00Z'}}
    write_json(root / 'pipeline/atlas/result.json', result)
    write_json(root / 'pipeline/atlas/preview_job.json', job)
    return root, source, context, result, job


def test_room_ready_task_limit_becomes_completed_preparation_without_reasking_gemini_or_native(tmp_path):
    from blueprint_pipeline.website_capture_preparation import run_website_capture_preparation
    from blueprint_pipeline.website_preparation_outcome import read_preparation_outcome
    root, source, context, result, _ = fixture(tmp_path)
    def forbidden(**_):
        pytest.fail('task cleaning/models/native must not block an already archived observed room')
    output = run_website_capture_preparation(descriptor=SimpleNamespace(metadata={'site_task_context': context}),
        capture_root=root, pipeline_dir=root / 'pipeline', source_video=source, derived_scene_generation_allowed=True,
        gates=[], prepare_media=forbidden, check_privacy=forbidden, make_gate=forbidden)
    assert output['status'] == 'completed' and output['native_execution_complete'] is False
    handoff = output['website_scene_preparation']
    assert handoff['visual_reconstruction_ready'] is True and handoff['simulator_ready'] is False
    assert retains_observed_room_limit(handoff, context)
    assert handoff['observed_room']['source_result_digest'] == result['digest']
    assert read_preparation_outcome(capture_root=root, task_context_digest=context['context_digest'], state='awaiting_inputs') == {
        'category': 'insufficient_evidence', 'reason': 'task_evidence_incomplete', 'basis': 'missing_evidence',
        'robot_id': None, 'evidence_digest': handoff['digest']}
    assert not (root / 'pipeline/website_scene_preparation/preparation.json').exists()
    assert not (root / 'pipeline/website_scene_preparation/native_runtime_contract.json').exists()


@pytest.mark.parametrize('change', ['context', 'source', 'missing_source', 'receipt', 'task_accepted', 'native', 'archive', 'missing_digest'])
def test_changed_or_unarchived_partial_work_never_claims_a_visual_ready_outcome(tmp_path, change):
    root, _, context, result, job = fixture(tmp_path)
    if change == 'context':
        context['context_digest'] = 'sha256:' + 'f' * 64
    elif change == 'source':
        result['source_video_digest'] = 'sha256:' + 'f' * 64
    elif change == 'missing_source':
        result.pop('source_video_digest')
    elif change == 'receipt':
        result['task_evidence_limitation']['summary'] = 'Changed task without its digest'
    elif change == 'task_accepted':
        result['task_and_surrounding_scene_covered'] = True
    elif change == 'native':
        result['native_preparation_eligible'] = True
    elif change == 'missing_digest':
        job['archive'].pop('digest')
        result['download'].pop('zip_sha256')
    else:
        job['archive'] = None
    result['digest'] = canonical_digest(result, digest_field='digest')
    job['result_digest'] = result['digest']
    write_json(root / 'pipeline/atlas/result.json', result)
    write_json(root / 'pipeline/atlas/preview_job.json', job)
    assert read_room_limit(root, context) is None


def test_rejected_old_worker_cannot_publish_the_partial_outcome(tmp_path):
    from blueprint_pipeline.core.handoff_lease import bind_handoff_lease
    root, source, context, _, _ = fixture(tmp_path)
    before = deepcopy(context)
    def lost(_):
        raise RuntimeError('lost-original-job-lease')
    with bind_handoff_lease(lost), pytest.raises(RuntimeError, match='lost-original-job-lease'):
        retain_observed_room_limit(capture_root=root, task_context=context, source_video=source)
    assert context == before
    assert not (root / 'pipeline/website_scene_preparation/handoff.json').exists()
