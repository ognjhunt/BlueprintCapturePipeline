from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline import sam31_contribution_disclosure as guard
from blueprint_pipeline import standard_splat_conversion as conversion
from blueprint_pipeline import task_evaluation_sam31_preparation_paid_stages as paid
from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from tests.test_standard_splat_conversion import _fixture, _request
from tests.test_task_evaluation_sam31_preparation_paid_stages import _job, _record, _write


def authorize_full_source(job: dict, *, source: Path, original: Path, receipt: Path) -> None:
    value = json.loads(receipt.read_text())
    raw, standard = value['source'], value['output']
    task_path = Path(job['plan']['host_inputs']['task_request']['path'])
    task = json.loads(task_path.read_text())
    authority = _write(source.parent/'explicit-full-source-authority.json', {
        'schema_version': guard.AUTHORITY_SCHEMA, 'status': 'authorized',
        'authority_kind': 'explicit_human_full_source_provider_processing',
        'authorized_by': task['human_authority']['accepted_by'],
        'authorized_on': '2026-09-05', 'authority_reference': 'hermetic explicit full-source authorization',
        'agent_accepted_terms': False, 'source_commit': job['expected_source_commit'],
        'provider_id': 'vast', 'purpose': 'released_code_segment_contribution_sweep',
        'source_binding': {
            'publisher_scene_id': '841757', 'dataset': raw['dataset'], 'publisher_revision': raw['revision'],
            'original_source_sha256': raw['sha256'], 'original_source_size_bytes': raw['size_bytes'],
            'standard_splat_sha256': standard['sha256'], 'standard_splat_size_bytes': standard['size_bytes'],
            'retained_gaussian_count': standard['gaussian_count'], 'source_gaussian_count': raw['source_gaussian_count'],
            'publisher_terms_digest': value['rights']['terms_digest'],
        },
        **{key: True for key in (
            'full_source_scene_content_upload_authorized', 'private_provider_processing_authorized',
            'publisher_rights_permit_private_full_source_processing', 'provider_retention_terms_accepted',
            'provider_training_terms_accepted', 'format_conversion_does_not_reduce_disclosure_scope',
        )},
        'publisher_rights_basis': {
            'kind': 'publisher_license_private_processing',
            'scope_explanation': 'Hermetic test-only permission; not publisher evidence.',
            'publisher_terms_evidence': job['inputs']['interiorgs_terms'],
            'private_processing_permission_evidence': job['inputs']['interiorgs_terms'],
        },
        'public_redistribution_authorized': False, 'provider_training_authorized': False,
    }, digest_field='authorization_digest')
    task['human_authority']['full_source_provider_disclosure_authority'] = _record(authority)
    _write(task_path, task)
    job['plan']['host_inputs']['task_request'] = _record(task_path)
    job['inputs'].update(source_appearance=_record(original),
                         standard_splat_conversion_receipt=_record(receipt))


def converted_job(tmp_path, monkeypatch, *, count=18):
    paths = _fixture(tmp_path/'conversion-fixture')
    terms = paths['data']/'publisher-terms.txt'
    terms.write_text('Hermetic test-only full source processing permission.')
    if count != 18:
        write_standard_3dgs_ply(SplatData(
            count=count, xyz=np.zeros((count, 3), dtype=np.float32),
            opacity=np.ones(count, dtype=np.float32), f_dc=np.zeros((count, 3), dtype=np.float32),
            scales=np.zeros((count, 3), dtype=np.float32),
            quats=np.tile(np.asarray([[1., 0., 0., 0.]], dtype=np.float32), (count, 1)), properties=(),
        ), paths['source'])
    request = _request(paths['source'], paths['data'])
    request['rights']['terms_digest'] = paid.sha(terms)
    request.pop('request_digest', None)
    paths['request'].write_text(json.dumps(conversion.build_standard_splat_conversion_request(request)))
    subprocess.run(['git', '-C', str(paths['repo']), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(paths['repo']), 'commit', '-qm', 'exact fixture count and terms'], check=True)
    def local_decoder(source, destination, **_kwargs):
        shutil.copy2(source, destination)
        return {'status': 'completed', 'decoder': 'hermetic-local-converter'}
    monkeypatch.setattr(conversion, 'convert_to_standard_ply', local_decoder)
    monkeypatch.setattr(conversion, 'read_compressed_ply_chunk_bounds',
                        lambda _path: SimpleNamespace(vertex_count=count))
    out = paths['data']/'converted'
    receipt = out/'standard_splat_conversion_receipt.v1.json'
    value = conversion.materialize_standard_splat_conversion(
        request_path=paths['request'], repo_root=paths['repo'], data_root=paths['data'],
        output_root=out, receipt_output=receipt,
    )
    job, output = _job(tmp_path, 'contribution_sweep')
    job['expected_source_commit'] = value['repository']['commit']
    job['server_profile']['source_commit'] = job['expected_source_commit']
    source = out/value['output']['relative_path']
    freeze = _write(tmp_path/'data/freeze.json', {
        'scene': {'publisher_scene_id': '841757', 'target_instance_id': '115'},
        'source_standard_splat': _record(source),
        'segment_contribution_sweep': {'kind': 'repair_supported_full_view_segment_contribution_sweep.v1'},
    }, digest_field='freeze_digest')
    cameras = _write(tmp_path/'data/cameras.json', [])
    job['inputs'].update(segment_sweep_freeze=_record(freeze), standard_splat=_record(source),
                         camera_contract=_record(cameras), source_appearance=_record(paths['source']),
                         standard_splat_conversion_receipt=_record(receipt), interiorgs_terms=_record(terms))
    return job, output, source, paths['source'], receipt


def test_actual_full_count_conversion_cannot_turn_frame_permission_into_upload(tmp_path, monkeypatch):
    job, output, _source, _original, receipt = converted_job(tmp_path, monkeypatch, count=700747)
    value = json.loads(receipt.read_text())
    assert value['source']['source_gaussian_count'] == value['output']['gaussian_count'] == 700747
    assert value['output']['gaussian_count_preserved'] is True
    assert value['rights']['raw_private_upload_authorized'] is False
    def forbidden(*_args, **_kwargs):
        pytest.fail('must block before authority synthesis, bundle creation, staging, or allocation')
    monkeypatch.setattr(paid, '_gaussian_execution_authority', forbidden)
    monkeypatch.setattr('blueprint_pipeline.adp_gaussian_excision_vast.build_gaussian_excision_vast_bundle', forbidden)
    monkeypatch.setattr('blueprint_pipeline.wam_provider_object_store.stage_wam_provider_bundle_object_store', forbidden)
    with pytest.raises(ValueError, match='explicit_full_source_authority_required'):
        paid.execute_paid_stage(job, allocator_runner=forbidden)
    assert not (output/'prepared').exists()
    assert not (output/'allocator').exists()


@pytest.mark.parametrize('fault, blocker', [
    ('wrong_provider', 'explicit_full_source_authority_invalid'),
    ('missing_publisher_basis', 'publisher_rights_basis_missing'),
    ('changed_publisher_evidence', 'input_bytes_mismatch'),
    ('wrong_source', 'explicit_full_source_authority_invalid'),
    ('unapproved_source_scope', 'explicit_full_source_scope_invalid'),
    ('changed_conversion_count', 'full_source_count_mismatch'),
    ('changed_original_bytes', 'input_bytes_mismatch'),
])
def test_explicit_authority_still_requires_exact_source_and_scope(tmp_path, monkeypatch, fault, blocker):
    job, _out, source, original, receipt = converted_job(tmp_path, monkeypatch)
    authorize_full_source(job, source=source, original=original, receipt=receipt)
    task_path = Path(job['plan']['host_inputs']['task_request']['path'])
    task = json.loads(task_path.read_text())
    ref = task['human_authority']['full_source_provider_disclosure_authority']
    path = Path(ref['path'])
    authority = json.loads(path.read_text())
    if fault == 'missing_publisher_basis':
        authority.pop('publisher_rights_basis')
    elif fault == 'changed_publisher_evidence':
        Path(authority['publisher_rights_basis']['publisher_terms_evidence']['path']).write_text('changed')
    elif fault == 'wrong_provider':
        authority['provider_id'] = 'another-provider'
    elif fault == 'wrong_source':
        authority['source_binding']['original_source_sha256'] = 'sha256:'+'f'*64
    elif fault == 'unapproved_source_scope':
        authority['full_source_scene_content_upload_authorized'] = False
    elif fault == 'changed_original_bytes':
        original.write_bytes(b'changed')
    else:
        value = json.loads(receipt.read_text())
        value['source']['source_gaussian_count'] += 1
        _write(receipt, value, digest_field='receipt_digest')
    _write(path, authority, digest_field='authorization_digest')
    task['human_authority']['full_source_provider_disclosure_authority'] = _record(path)
    with pytest.raises(ValueError, match=blocker):
        guard.validate_full_source_disclosure(
            task_authority=task['human_authority'], conversion_path=receipt,
            standard_splat_path=source, original_source_path=original,
            expected_source_commit=job['expected_source_commit'], publisher_scene_id='841757',
            approved_roots=(tmp_path,),
        )


def test_separate_exact_full_source_authority_preserves_local_conversion_rights(tmp_path, monkeypatch):
    job, _output, source, original, receipt = converted_job(tmp_path, monkeypatch)
    authorize_full_source(job, source=source, original=original, receipt=receipt)
    task = json.loads(Path(job['plan']['host_inputs']['task_request']['path']).read_text())
    proof = guard.validate_full_source_disclosure(
        task_authority=task['human_authority'], conversion_path=receipt,
        standard_splat_path=source, original_source_path=original,
        expected_source_commit=job['expected_source_commit'], publisher_scene_id='841757',
        approved_roots=(tmp_path,),
    )
    assert proof['payload_kind'] == 'full_source_scene_reencoded_standard_splat'
    assert proof['conversion_rights']['raw_private_upload_authorized'] is False
    assert proof['frame_permission_used_as_full_source_authority'] is False
