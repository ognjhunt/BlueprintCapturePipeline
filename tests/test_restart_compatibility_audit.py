"""ADP-009D/day-28: synthetic immutable reuse boundaries, never paid execution.

These exercise real parent, chain, path and receipt readers. A bookkeeping-only
prefix deliberately lacks renderer proof; reaching that missing field is NOT
scientific acceptance. No validator under test is replaced.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_sam31_prefix_adoption as adoption
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_preparation_contract import (
    launch_preparation_request_digest,
    validate_launch_preparation_request,
    validate_retained_preparation_request,
)
from blueprint_pipeline.task_evaluation_sam31_parent_evidence import (
    _parent, configured_parent_route, retained_parent,
)
from blueprint_pipeline.task_evaluation_sam31_phase_queue import enqueue_sam31_phase
from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import beneath
from tests.test_sam31_prefix_adoption import prefix as prefix, write
from tests.test_task_evaluation_launch_preparation_worker import production_request_with_fetchable_bytes


def _parent_fixture(root, mutate=lambda request: None):
    request, _ = production_request_with_fetchable_bytes()
    mutate(request)
    digest = canonical_digest(request)
    job = dict(parent_preparation_id=request['preparation_id'], parent_request_digest=digest)
    path = root / 'blocked' / (request['preparation_id'] + '-' + digest[7:] + '.json')
    write(path, {'request': request, 'request_digest': digest}, 'envelope_digest')
    return request, job, path


@pytest.mark.parametrize('change,accepted,predicate', [
    ('review_cap', True, 'external_spend_invalid'),
    ('ttl', False, 'parent_runtime_budget_invalid'),
    ('requests', True, ''),
    ('zero_requests', False, 'external_spend_invalid'),
    ('unknown_schema', False, ''),
    ('invalid_provenance', False, ''),
])
def test_historical_contract_is_explicit_not_a_general_policy_bypass(tmp_path, change, accepted, predicate):
    def mutate(request):
        spend = request['spend']
        if change == 'review_cap':
            spend['external_service_caps']['openai']['stage_max_cost_usd']['artifixer_visual_review'] = .64
        elif change == 'ttl':
            spend['hard_ttl_seconds'] -= 1
        elif change == 'requests':
            spend['external_service_caps']['openai']['maximum_requests'] = 31
        elif change == 'zero_requests':
            spend['external_service_caps']['openai']['maximum_requests'] = 0
        elif change == 'unknown_schema':
            request['schema_version'] = 'unknown.v0'
        else:
            request['expected_production_commit'] = 'unattributed'
    request, job, path = _parent_fixture(tmp_path, mutate)
    before = path.read_bytes()
    if accepted:
        assert retained_parent(job, tmp_path)[0] == request
        assert validate_retained_preparation_request(request) == request
    else:
        with pytest.raises(ValueError, match=predicate):
            retained_parent(job, tmp_path)
    if change == 'review_cap':
        for validator in (validate_launch_preparation_request, launch_preparation_request_digest):
            with pytest.raises(ValueError, match=predicate):
                validator(request)
        with pytest.raises(ValueError, match=predicate):
            _parent(job, tmp_path)
    assert path.read_bytes() == before


def _historical_chain(prefix, tmp_path):
    """Rebuild every cryptographic child join after changing the parent budget."""
    value, plan, profile, _ = prefix
    old_parent = json.loads(Path(value['original_parent_envelope']['path']).read_text())['request']
    old_parent['spend']['external_service_caps']['openai']['stage_max_cost_usd']['artifixer_visual_review'] = .64
    digest = canonical_digest(old_parent)
    parent_path = tmp_path / 'historical-parents/blocked' / (old_parent['preparation_id'] + '-' + digest[7:] + '.json')
    write(parent_path, {'request': old_parent, 'request_digest': digest}, 'envelope_digest')
    queue = tmp_path / 'historical-queue'
    inputs = {**plan['host_inputs'], **profile['artifact_references']}
    rows = []
    for original in value['phase_records']:
        intake = enqueue_sam31_phase(queue_root=queue, parent_preparation_id=old_parent['preparation_id'],
            parent_request_digest=digest, expected_source_commit=plan['source_commit'],
            plan_ref=value['source_plan'], phase=original['phase'], inputs=inputs)
        job_path = Path(intake['job_path'])
        job = json.loads(job_path.read_text())
        completed = queue / 'completed' / job_path.name
        job_path.rename(completed)
        result = json.loads(Path(original['result']['path']).read_text())
        result.update({key: job[key] for key in ('job_digest', 'child_id', 'parent_request_digest', 'plan_digest')})
        result_ref = write(Path(intake['result_path']), result, 'result_digest')
        receipt = json.loads(Path(original['execution_receipt']['path']).read_text())
        receipt['job_digest'] = job['job_digest']
        receipt_ref = write(tmp_path / 'historical-executions' / digest[7:] / job['child_id'] / 'phase_execution_receipt.v1.json', receipt, 'receipt_digest')
        rows.append(dict(phase=original['phase'], job=adoption.record(completed), result=result_ref, execution_receipt=receipt_ref))
        inputs.update(result['artifacts'])
        if original['phase'] == 'standard_splat_conversion':
            inputs['standard_splat_conversion'] = inputs['standard_splat_conversion_receipt']
    value.update(original_parent_request_digest=digest, original_parent_envelope=adoption.record(parent_path), phase_records=rows)
    return value, plan, profile


def test_old_budget_materializer_reaches_real_science_reader_without_rewriting_parent(prefix, tmp_path):
    value, plan, profile = _historical_chain(prefix, tmp_path)
    before = {Path(ref['path']): Path(ref['path']).read_bytes() for row in value['phase_records']
              for key, ref in row.items() if key != 'phase'}
    parent = Path(value['original_parent_envelope']['path'])
    before[parent] = parent.read_bytes()
    # Real chain validation precedes the independent renderer proof boundary.
    assert set(adoption._phase_chain(value, (tmp_path,))[3]) == set(adoption.PHASES[:5])
    zero = write(tmp_path / 'zero.json', dict(provider='vast', status='observed', api_confirmed=True,
        name_prefix='', live_resource_count=0, resources=[], http=200, observed_at_epoch=1000.))
    with pytest.raises(KeyError, match='source_calibration_prepared_inputs'):
        adoption.materialize_completed_prefix_adoption(source_plan_path=value['source_plan']['path'],
            source_profile_path=value['source_profile']['path'], parent_request_digest=value['original_parent_request_digest'],
            through_phase='calibrated_views', current_host_inputs=plan['host_inputs'],
            current_provider_profile_path=profile['artifact_references']['sam31_provider_profile']['path'],
            current_repo_root=tmp_path, expected_source_commit='b' * 40, provider_zero_path=zero['path'],
            output_path=None, approved_roots=(tmp_path,), queue_root=tmp_path / 'historical-queue',
            parent_queue_root=tmp_path / 'historical-parents', execution_root=tmp_path / 'historical-executions', now_epoch=1001.)
    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert not list((tmp_path / 'historical-queue/pending').glob('*.json'))


@pytest.mark.parametrize('state', ['processing', 'failed'])
def test_duplicate_completed_child_is_never_reused(prefix, tmp_path, state):
    value, *_ = prefix
    path = Path(value['phase_records'][0]['job']['path'])
    duplicate = path.parent.parent / state / path.name
    duplicate.parent.mkdir(exist_ok=True)
    duplicate.write_bytes(path.read_bytes())
    before = path.read_bytes()
    with pytest.raises(ValueError, match='sam31_adoption_job_identity_ambiguous'):
        adoption._phase_chain(value, (tmp_path,))
    assert path.read_bytes() == duplicate.read_bytes() == before


@pytest.mark.parametrize('change,predicate', [
    ('missing', 'sam31_adoption_file_invalid'),
    ('truncate', 'sam31_adoption_reference_changed'),
    ('same_length', 'sam31_adoption_reference_changed'),
    ('wrong_hash', 'sam31_adoption_reference_changed'),
    ('symlink', 'sam31_adoption_file_invalid'),
    ('outside_root', 'sam31_adoption_reference_changed'),
])
def test_artifact_reference_damage_is_refused_without_repair(tmp_path, change, predicate):
    root = tmp_path / 'admitted'
    ref = write(root / 'artifact.json', {'synthetic': 'aaaa'})
    path = Path(ref['path'])
    if change == 'missing':
        path.unlink()
    elif change == 'truncate':
        path.write_bytes(path.read_bytes()[:-1])
    elif change == 'same_length':
        path.write_bytes(path.read_bytes().replace(b'aaaa', b'bbbb'))
    elif change == 'wrong_hash':
        ref['sha256'] = 'sha256:' + '0' * 64
    elif change == 'symlink':
        target = root / 'original.json'
        path.rename(target)
        path.symlink_to(target)
    else:
        ref = write(tmp_path / 'unapproved/artifact.json', {'synthetic': 'aaaa'})
    before = path.read_bytes() if path.exists() else None
    with pytest.raises(ValueError, match=predicate):
        adoption._ref(ref, (root,))
    assert (path.read_bytes() if path.exists() else None) == before


def test_custom_retained_roots_are_explicit_and_paths_normalize(tmp_path):
    refs = [write(tmp_path / name / 'evidence.json', {'synthetic': name})
            for name in ('billing', 'renderer', 'outputs')]
    for ref in refs:
        path = Path(ref['path'])
        assert adoption.record(str(path)) == adoption.record(path) == ref
        assert adoption._ref(ref, (path.parent,)) == path
        with pytest.raises(ValueError, match='sam31_adoption_reference_changed'):
            adoption._ref(ref, (tmp_path / 'default-only',))


@pytest.mark.parametrize('relative', ['../escape', '/absolute', 'nested/../../escape', r'nested\escape'])
def test_manifest_traversal_is_not_an_admitted_root(tmp_path, relative):
    with pytest.raises(ValueError, match='relative_path_invalid'):
        beneath(tmp_path, relative)


def test_parent_copies_across_owned_and_legacy_stores_are_ambiguous(tmp_path, monkeypatch):
    legacy, owned = tmp_path / 'legacy', tmp_path / 'owned'
    _, job, path = _parent_fixture(legacy)
    config = {'schema_version': 'task_evaluation_scene_progression_config.v1',
        'preparation_queue_root': str(owned), 'preparation_worker': {'input_root': str(tmp_path / 'owned-inputs')}}
    ref = write(tmp_path / 'routing.json', config, 'config_digest')
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_PROGRESSION_CONFIG', ref['path'])
    assert configured_parent_route(job, legacy, tmp_path / 'legacy-inputs')[0] == legacy
    duplicate = owned / 'blocked' / path.name
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match='parent_identity_ambiguous'):
        configured_parent_route(job, legacy, tmp_path / 'legacy-inputs')


@pytest.mark.parametrize('status', ['failed', 'waiting_external'])
def test_incomplete_tail_does_not_erase_completed_upstream_chain(prefix, tmp_path, status):
    value, *_ = prefix
    row = value['phase_records'][4]
    result = json.loads(Path(row['result']['path']).read_text())
    result['status'] = status
    row['result'] = write(Path(row['result']['path']), result, 'result_digest')
    before = Path(row['result']['path']).read_bytes()
    with pytest.raises(ValueError, match='sam31_adoption_terminal_result_invalid'):
        adoption._phase_chain(value, (tmp_path,))
    upstream = deepcopy(value)
    upstream.update(through_phase='calibrated_views', phase_records=value['phase_records'][:3])
    assert set(adoption._phase_chain(upstream, (tmp_path,))[3]) == set(adoption.PHASES[:3])
    assert Path(row['result']['path']).read_bytes() == before


@pytest.mark.parametrize('case', ['restorable', 'missing_pointer', 'damaged_archive'])
def test_offloaded_pointer_requires_explicit_bounded_restore(tmp_path, case):
    import hashlib
    import io
    import tarfile
    from blueprint_pipeline.control_plane_evidence_offload import (
        POINTER_SCHEMA_VERSION, ControlPlaneEvidenceOffloadError, restore_offloaded_evidence,
    )
    payload = b'synthetic retained evidence'
    directory = tmp_path / 'retained'
    ref = {'path': str(directory / 'artifact.bin'), 'size_bytes': len(payload),
           'sha256': 'sha256:' + hashlib.sha256(payload).hexdigest()}
    memory = io.BytesIO()
    with tarfile.open(fileobj=memory, mode='w') as archive:
        member = tarfile.TarInfo('artifact.bin')
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    archive_bytes = memory.getvalue()
    pointer = {'schema_version': POINTER_SCHEMA_VERSION, 'status': 'offloaded',
        'directory': 'retained', 'uri': 's3://synthetic-only/evidence.tar',
        'digest': 'sha256:' + hashlib.sha256(archive_bytes).hexdigest(),
        'size_bytes': len(archive_bytes),
        'members': [{'relative_path': 'artifact.bin', 'sha256': ref['sha256'], 'size_bytes': len(payload)}]}
    pointer_path = tmp_path / 'retained.offloaded.json'
    write(pointer_path, pointer, 'pointer_digest')
    before = pointer_path.read_bytes()
    calls = []
    def materialize(*, reference, destination, maximum_size_bytes):
        calls.append(reference)
        assert maximum_size_bytes == len(archive_bytes)
        destination.write_bytes(archive_bytes if case != 'damaged_archive' else b'x' * len(archive_bytes))
        return {'status': 'materialized'}
    # A pointer alone does not masquerade as local source evidence.
    with pytest.raises(ValueError, match='sam31_adoption_file_invalid'):
        adoption._ref(ref, (tmp_path,))
    if case == 'missing_pointer':
        pointer_path.unlink()
        with pytest.raises(ControlPlaneEvidenceOffloadError, match='restore_pointer_invalid'):
            restore_offloaded_evidence(pointer_path=pointer_path, destination=directory, materializer=materialize)
        assert calls == []
    elif case == 'damaged_archive':
        with pytest.raises(ControlPlaneEvidenceOffloadError, match='restore_digest_mismatch'):
            restore_offloaded_evidence(pointer_path=pointer_path, destination=directory, materializer=materialize)
        assert len(calls) == 1 and not directory.exists()
    else:
        result = restore_offloaded_evidence(pointer_path=pointer_path, destination=directory, materializer=materialize)
        assert result['status'] == 'restored' and len(calls) == 1
        assert adoption._ref(ref, (tmp_path,)).read_bytes() == payload
    if pointer_path.exists():
        assert pointer_path.read_bytes() == before
    assert not list(tmp_path.glob('.restore-*'))


def test_current_disclosure_must_belong_to_current_owner(tmp_path, monkeypatch):
    from tests.test_sam31_contribution_disclosure import converted_job, authorize_full_source
    from blueprint_pipeline.sam31_contribution_disclosure import validate_full_source_disclosure
    job, _, source, original, receipt = converted_job(tmp_path, monkeypatch)
    authorize_full_source(job, source=source, original=original, receipt=receipt)
    task = json.loads(Path(job['plan']['host_inputs']['task_request']['path']).read_text())
    args = dict(task_authority=task['human_authority'], conversion_path=receipt,
        standard_splat_path=source, original_source_path=original,
        expected_source_commit=job['expected_source_commit'], publisher_scene_id='841757', approved_roots=(tmp_path,))
    validate_full_source_disclosure(**args)
    permit = Path(task['human_authority']['full_source_provider_disclosure_authority']['path'])
    before = permit.read_bytes()
    task['human_authority']['accepted_by'] = 'different-owner'
    with pytest.raises(ValueError, match='explicit_full_source_authority_invalid'):
        validate_full_source_disclosure(**args)
    assert permit.read_bytes() == before


def test_resealed_parent_cannot_change_namespace_under_old_child_identity(tmp_path):
    request, job, path = _parent_fixture(tmp_path)
    request['team_namespace'] = 'another-team'
    write(path, {'request': request, 'request_digest': job['parent_request_digest']}, 'envelope_digest')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='parent_envelope_invalid'):
        retained_parent(job, tmp_path)
    assert path.read_bytes() == before
