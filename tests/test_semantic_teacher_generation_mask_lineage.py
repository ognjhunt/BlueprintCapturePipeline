"""Replays the production old-mask edit hidden behind a SAM reuse receipt."""
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.semantic_teacher_candidate_reuse import load_retained_candidates
from blueprint_pipeline import semantic_teacher_candidate_discovery as discovery
from tests.test_semantic_teacher_image_edit_worker import _request_with_retained_candidate, _record
from tests.test_semantic_teacher_candidate_discovery import _workspace, _write, _seal


def _parent(request_path, change):
    request = json.loads(request_path.read_text())
    row = request['retained_candidates'][0]
    path = request_path.parent / row['source_runtime_result']['relative_path']
    result = json.loads(path.read_text())
    change(result)
    result['result_digest'] = canonical_digest(result, digest_field='result_digest')
    path.write_text(json.dumps(result))
    row['source_runtime_result'] = _record(path, root=request_path.parent)
    request['request_digest'] = canonical_digest(request, digest_field='request_digest')
    request_path.write_text(json.dumps(request))
    return request


@pytest.mark.parametrize('kind', ['old_mask', 'laundered_mask', 'missing_lineage', 'wrong_encoding'])
def test_reused_generation_cannot_inherit_new_mask_from_wrapper(tmp_path, kind):
    path, _, _, _ = _request_with_retained_candidate(tmp_path)
    initial = json.loads(path.read_text())
    admitted = load_retained_candidates(request=initial, request_root=path.parent)
    lineage = next(iter(admitted.values()))['lineage']
    if kind in ('old_mask', 'laundered_mask'):
        lineage = {'original_edit_mask_sha256': 'sha256:' + 'a' * 64 if kind == 'old_mask'
                   else lineage['original_edit_mask_sha256']}
    elif kind == 'wrong_encoding':
        lineage['generation_mask_binding']['mask_encoding'] = 'different_encoding'
    def change(result):
        frame = result['tasks'][0]['frames'][0]
        frame['provider_call_performed'] = False
        if kind != 'missing_lineage':
            frame['retained_candidate_lineage'] = lineage
    request = _parent(path, change)
    assert load_retained_candidates(request=request, request_root=path.parent) == {}


def test_verified_generation_binding_survives_multiple_reuse_hops(tmp_path):
    path, _, _, _ = _request_with_retained_candidate(tmp_path)
    request = json.loads(path.read_text())
    first = load_retained_candidates(request=request, request_root=path.parent)
    lineage = next(iter(first.values()))['lineage']
    origin = dict(lineage['generation_mask_binding'])
    for _ in range(3):
        def change(result):
            frame = result['tasks'][0]['frames'][0]
            frame['provider_call_performed'] = False
            frame['retained_candidate_lineage'] = lineage
        request = _parent(path, change)
        admitted = load_retained_candidates(request=request, request_root=path.parent)
        lineage = next(iter(admitted.values()))['lineage']
        assert lineage['generation_mask_binding'] == origin
        assert lineage['original_edit_mask_sha256'] == origin['edit_mask_sha256']


@pytest.mark.parametrize('repair', [False, True])
def test_discovery_skips_unverified_generation_in_base_and_repair_outputs(tmp_path, repair):
    cameras = ['source-01', 'source-02']
    inputs = {c: 'sha256:' + str(i + 1) * 64 for i, c in enumerate(cameras)}
    masks = {c: 'sha256:' + str(i + 3) * 64 for i, c in enumerate(cameras)}
    request = _workspace(tmp_path, 'old', cameras, inputs, masks,
                         review1={c: True for c in cameras},
                         repair=['source-01'] if repair else None,
                         review2={c: True for c in cameras})
    runtime = tmp_path / 'old' / discovery.RUNTIME
    result_path = runtime / (discovery.REPAIR_RESULT if repair else discovery.RESULT)
    result = json.loads(result_path.read_text())
    frame = result['tasks'][0]['frames'][0]
    frame['provider_call_performed'] = False
    frame['retained_candidate_lineage'] = {'original_edit_mask_sha256': masks['source-01']}
    _write(result_path, _seal(result, 'result_digest'))
    plan = discovery._workspace_plan(runtime, discovery._request_frames(request),
                                     discovery._backend(request), request.get('prompt'))
    assert plan['retained'] == ['source-02']
    assert {'camera_id': 'source-01', 'reason': 'generation_mask_unverified'} in plan['skipped']
