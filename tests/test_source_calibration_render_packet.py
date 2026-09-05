from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

from blueprint_pipeline import source_calibration_render_packet as packet
from blueprint_pipeline import standard_splat_conversion as conversion
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import sha
from tests.test_sam31_contribution_disclosure import authorize_full_source


def valid_source_bundle_inputs(tmp_path, monkeypatch):
    from tests import test_public_scene_inpainting_preparation as cpu
    original = cpu._write_v2_fixture
    project = Path(__file__).resolve().parents[1]
    def with_code(root, **kwargs):
        paths = original(root, **kwargs)
        for name in ['tools/splat_render/render_splat.mjs','tools/splat_render/src/render_entry.mjs',
                     'tools/splat_render/harness.html','tools/splat_render/package.json','tools/splat_render/package-lock.json',
                     'scripts/run_adp_retained_scene_render_provider_runtime.sh',
                     'scripts/adp_retained_scene_render_provider_runner.mjs','scripts/adp_retained_scene_render_rehearsal.py']:
            destination=paths['repo']/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(project/name,destination)
        subprocess.run(['git','-C',str(paths['repo']),'add','.'],check=True,capture_output=True)
        subprocess.run(['git','-C',str(paths['repo']),'commit','-qm','actual renderer code fixture'],check=True,capture_output=True)
        return paths
    monkeypatch.setattr(cpu,'_write_v2_fixture',with_code)
    paths,prepared=cpu._prepare(tmp_path)
    source=Path(prepared['layers']['images']['path'])
    original_source=paths['data']/'original_source.ply'
    original_source.write_bytes(source.read_bytes().replace(b'ply\n', b'ply\ncomment original fixture encoding\n', 1))
    terms=paths['data']/'source-terms.txt'
    terms.write_text('Hermetic fixture permits private full-source compute; not real publisher evidence.')
    request=conversion.build_standard_splat_conversion_request({
        'schema_version':'standard_splat_conversion_request.v1','program_id':'arm-decision-proof-v1',
        'frozen_before_conversion':True,'learned_policy_outcomes_observed':False,
        'source':{'relative_path':str(original_source.relative_to(paths['data'])),'sha256':sha(original_source),
                  'size_bytes':original_source.stat().st_size,'dataset':'synthetic-fixture','revision':'a'*40,'license':'fixture-only'},
        'rights':{'conversion_execution_location':'local_only','raw_private_upload_authorized':False,
                  'training_authorized':False,'terms_digest':sha(terms)},'output_filename':'source_standard.ply'})
    request_path=paths['data']/'exact-local-conversion-request.json'
    request_path.write_text(canonical_json(request))
    # This converter seam is deliberately hermetic; the real materializer
    # validates the actual source/output arrays, counts, rights and receipts.
    monkeypatch.setattr(conversion,'find_splat_transform_cli',lambda _:paths['repo']/'tools/splat_render/src/render_entry.mjs')
    monkeypatch.setattr(conversion,'read_compressed_ply_chunk_bounds',lambda _:SimpleNamespace(vertex_count=prepared['layers']['images']['retained_gaussian_count']))
    def convert(original,destination,**kwargs):
        shutil.copy2(source,destination)
        return {'status':'completed','decoder':'hermetic-local-identity-converter'}
    monkeypatch.setattr(conversion,'convert_to_standard_ply',convert)
    converted=paths['data']/'converted'
    receipt=converted/'standard_splat_conversion_receipt.v1.json'
    conversion.materialize_standard_splat_conversion(request_path=request_path,repo_root=paths['repo'],
        data_root=paths['data'],output_root=converted,receipt_output=receipt)
    task=paths['data']/'gpu-source-task.json'
    task.write_text(json.dumps({'publisher_scene_id':'841757','human_authority':{'accepted_by':'fixture-owner',
        'accepted_on':'2026-09-05','authority_reference':'hermetic bounded render authority',
        'source_calibration_gpu_render_authorized':True}}))
    def rec(p):return {'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size}
    job={'expected_source_commit':prepared['repository']['commit'],
         'plan':{'host_inputs':{'task_request':rec(task)}},'inputs':{'interiorgs_terms':rec(terms)}}
    authorize_full_source(job,source=source,original=original_source,receipt=receipt)
    data=json.loads(task.read_text())
    data['publisher_scene_id']=prepared['context']['source_identity']['scene_id']
    authority_path=Path(data['human_authority']['full_source_provider_disclosure_authority']['path'])
    authority=json.loads(authority_path.read_text())
    authority['purpose']=packet.PURPOSE
    authority['source_binding']['publisher_scene_id']=data['publisher_scene_id']
    authority['authorization_digest']=canonical_digest(authority,digest_field='authorization_digest')
    authority_path.write_text(canonical_json(authority))
    data['human_authority']['full_source_provider_disclosure_authority']=rec(authority_path)
    task.write_text(canonical_json(data))
    vendor=paths['data']/'vendor'
    for name in packet._VENDOR_PACKAGES:
        directory=vendor/name
        directory.mkdir(parents=True)
        (directory/'package.json').write_text(json.dumps({'name':name,'version':'fixture'}))
    return {'prepared_inputs_path':prepared['preparation_path'],'repo_root':paths['repo'],
        'renderer_vendor_root':vendor,'task_request_path':task,'conversion_receipt_path':receipt,
        'original_source_path':original_source,'job_dir':paths['data']/'bundle',
        'expected_source_commit':prepared['repository']['commit'],'approved_roots':(tmp_path,)},prepared


def test_real_source_packet_rehearsal_is_exact_and_excludes_other_dataset_files(tmp_path,monkeypatch):
    args,prepared=valid_source_bundle_inputs(tmp_path,monkeypatch)
    bundle=packet.build_source_calibration_gpu_render_bundle(**args)
    assert packet.validate_source_calibration_bundle(bundle,expected_commit=args['expected_source_commit'])==bundle
    assert bundle['expected_png_count']==48
    assert bundle['hard_total_spend_cap_usd']==1 and bundle['hard_ttl_seconds']==1800
    assert bundle['execution_authority']['authority_digest'] != bundle['source_disclosure']['disclosure_authority_digest']
    with zipfile.ZipFile(bundle['bundle_path']) as archive:
        inputs={name for name in archive.namelist() if name.startswith('provider_runtime/input/')}
        assert inputs=={'provider_runtime/input/'+name for name in ('images.ply','target_support.ply','scene_without_target.ply','cameras.v1.json')}
        assert not any(name.endswith(('.usd','.usdz')) or '/labels' in name or '/structure' in name for name in archive.namelist())
    runtime=args['job_dir']/'provider_runtime'
    result=subprocess.run(['node',str(runtime/'adp_retained_scene_render_provider_runner.mjs'),
        '--runtime',str(runtime),'--output',str(args['job_dir']/'node-rehearsal'),'--rehearsal'],
        capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    observed=json.loads((args['job_dir']/'node-rehearsal/provider_bundle_rehearsal.json').read_text())
    assert observed['verified_layers']==3 and observed['expected_png_count']==48
    assert observed['paid_inference_performed'] is False


def test_source_packet_requires_distinct_paid_authority_before_bundle(tmp_path,monkeypatch):
    args,_=valid_source_bundle_inputs(tmp_path,monkeypatch)
    task=json.loads(args['task_request_path'].read_text())
    task['human_authority']['source_calibration_gpu_render_authorized']=False
    args['task_request_path'].write_text(canonical_json(task))
    with pytest.raises(ValueError,match='paid_render_authority_required'):
        packet.build_source_calibration_gpu_render_bundle(**args)
    assert not args['job_dir'].exists()


def test_source_bundle_reaches_actual_canonical_allocator_admission(tmp_path, monkeypatch):
    from blueprint_pipeline import paid_resource_allocator as allocator
    from blueprint_pipeline.paid_resource_admission import require_paid_resource_admission_grant
    from scripts.issue_retained_scene_render_paid_attempt_authority import issue_paid_attempt_authority
    args, _prepared = valid_source_bundle_inputs(tmp_path, monkeypatch)
    bundle = packet.build_source_calibration_gpu_render_bundle(**args)
    receipt_path = args['job_dir']/packet.RECEIPT_NAME
    authority = issue_paid_attempt_authority(bundle_receipt_path=receipt_path,
        authorized_by='fixture-owner', max_hourly_rate_usd=.5, hard_ttl_seconds=1800)
    authority_path = tmp_path/'attempt.json'
    authority_path.write_text(canonical_json(authority))
    commit = args['expected_source_commit']
    monkeypatch.setattr(allocator, '_control_plane_checkout_blockers',
                        lambda: ([], {'orchestrator_source_commit': commit}))
    observed = []
    def provider(**kwargs):
        require_paid_resource_admission_grant(kwargs['paid_resource_admission_grant'],
                                             resource_class='vast_provider_adapter')
        assert kwargs['prepared_bundle']['bundle_sha256'] == bundle['bundle_sha256']
        assert kwargs['paid_attempt_authority']['purpose'] == packet.PURPOSE
        observed.append(kwargs)
        return {'status': 'completed', 'provider_mutations_performed': 0}
    monkeypatch.setattr(allocator, 'run_retained_scene_render_vast', provider)
    code = allocator.main(['gpu-canary', '--provider', 'vast', '--probe-kind', packet.PROBE_KIND,
        '--execute', '--expected-source-commit', commit, '--admission-out', str(tmp_path/'admission.json'),
        '--adapter-output', str(tmp_path/'adapter.json'), '--adp-retained-scene-render-bundle-receipt', str(receipt_path),
        '--adp-retained-scene-render-attempt-authority', str(authority_path),
        '--adp-retained-scene-render-job-dir', str(tmp_path/'provider'),
        '--adp-retained-scene-render-max-hourly-rate-usd', '.5', '--adp-retained-scene-render-hard-ttl-seconds', '1800'])
    admission = json.loads((tmp_path/'admission.json').read_text())
    assert code == 0, admission
    assert len(observed) == 1
    assert admission['private_scene_derived_input_only'] is False
    assert admission['full_source_scene_content_upload_authorized'] is True
