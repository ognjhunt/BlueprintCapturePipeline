"""Independent source-calibration admission and provider lifecycle regressions.

All provider/SSH boundaries are hermetic. No source dataset, model or paid
provider is accessed by this suite.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.adp_retained_scene_render_vast import PROVIDER_BUNDLE_KIND
from blueprint_pipeline import vast_provider_output_recovery as recovery
from tests.test_vast_provider_output_recovery import _install_identity
from blueprint_pipeline.vast_pre_mutation_reselection import pre_mutation_offer_reselection_attempts


@pytest.mark.parametrize('failure', [None, 'size', 'digest'])
def test_retained_source_variant_can_recover_its_exact_archive_over_pinned_stream(
    tmp_path, monkeypatch, failure,
):
    """The shared recovery seam must recognize this real bundle kind."""
    _install_identity(monkeypatch, tmp_path)
    payload = b'hermetic-source-calibration-archive'
    digest = hashlib.sha256(payload).hexdigest()
    calls = []

    def ssh(command, **kwargs):
        calls.append((command, kwargs))
        assert any('StrictHostKeyChecking=yes' in str(item) for item in command)
        assert '/workspace/adp_retained_scene_render_provider_runtime_output.zip' in command[-1]
        assert kwargs['timeout'] <= recovery.MAX_RECOVERY_SECONDS
        if kwargs.get('text'):
            size = len(payload) + (1 if failure == 'size' else 0)
            return SimpleNamespace(returncode=0, stdout=f'{size} {digest}\n')
        assert kwargs.get('capture_output') is None
        assert kwargs.get('stdout') is not None
        kwargs['stdout'].write(payload if failure != 'digest' else b'x'*len(payload))
        return SimpleNamespace(returncode=0, stderr=b'')

    monkeypatch.setattr(recovery.subprocess, 'run', ssh)
    output = tmp_path/'output.zip'
    result = recovery.recover_provider_output_before_teardown(
        connection={'ssh_host': 'example.invalid', 'ssh_port': 2222},
        provider_bundle_kind=PROVIDER_BUNDLE_KIND, output_path=output,
        attempt_dir=tmp_path/'attempt', expected_size_bytes=len(payload))
    assert result['status'] != 'not_supported', result
    if failure:
        assert result['status'] == 'blocked'
        assert not output.exists()
        assert not output.with_name(output.name + '.ssh-recovery.partial').exists()
        assert len(calls) == (1 if failure == 'size' else 2)
    else:
        assert result['status'] == 'completed'
        assert result['strict_host_key_checking'] is True
        assert result['streamed_to_disk'] is True
        assert result['recovered_sha256'] == 'sha256:' + digest
        assert result['recovered_size_bytes'] == len(payload)
        assert output.read_bytes() == payload
        assert len(calls) == 2


@pytest.mark.parametrize('defect', [None, 'privacy', 'watchdog', 'teardown', 'cleanup', 'multiple_instances', 'output'])
def test_source_variant_lifecycle_preserves_allocation_and_terminal_boundaries(tmp_path, monkeypatch, defect):
    """Exercise the real lane wrapper; replace only I/O/receipt fixtures here.

    Separate packet/return tests own source rights and exact rendered bytes.
    This test proves the admitted source variant reaches the existing adapter
    with its fixed limits and cannot close successfully on missing evidence.
    """
    from blueprint_pipeline import adp_retained_scene_render_vast as lane
    from blueprint_pipeline.source_calibration_render_packet import WESTERN_US_REGEX
    from blueprint_pipeline import source_calibration_private_store as private_store

    events = []
    archive = tmp_path/'source-runtime.zip'
    archive.write_bytes(b'fixture-immutable-runtime')
    avoidlist = tmp_path/'machine-avoidlist.json'
    avoidlist.write_text('{}')
    bundle = {'schema_version': 'adp009d_source_calibration_gpu_render_bundle.v1',
              'render_scope': 'source_calibration', 'status': 'ready',
              'bundle_path': str(archive), 'bundle_sha256': 'sha256:' + hashlib.sha256(archive.read_bytes()).hexdigest(),
              'blueprint_commit': 'a'*40, 'hard_total_spend_cap_usd': 1.,
              'preferred_geolocation_regex': WESTERN_US_REGEX}
    monkeypatch.setattr(lane, 'validate_retained_scene_render_bundle', lambda value, **kwargs: dict(value))
    monkeypatch.setattr(lane, 'validate_retained_scene_render_paid_attempt_authority', lambda value, **kwargs: dict(value))
    monkeypatch.setattr(lane, 'require_paid_resource_admission_grant', lambda *args, **kwargs: None)
    store_files = {key: str(tmp_path/key) for key in private_store.ENV}
    monkeypatch.setenv('BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL', 'https://legacy.example.invalid')

    def privacy(**kwargs):
        events.append('privacy')
        if defect == 'privacy':
            raise ValueError('private_store_readback_unknown')
        return {'staging_kwargs': store_files, 'readback': {'status': 'verified_private'}}

    monkeypatch.setattr(private_store, 'verify_private_source_store', privacy)

    def stage(**kwargs):
        events.append('stage')
        assert all(kwargs.get(key) == value for key, value in store_files.items())
        root = kwargs['job_dir']
        root.mkdir(parents=True)
        for name in ('provider_bundle_url.txt', 'provider_output_put_url.txt', 'provider_output_get_url.txt'):
            (root/name).write_text('https://example.invalid/private-fixture')
        return {'status': 'completed'}

    def arm(**kwargs):
        events.append('arm')
        assert kwargs['max_live_minutes'] == 30
        assert kwargs['allowed_active_instance_ids'] == ()
        return {'status': 'armed'}, SimpleNamespace(pod_name_prefix=kwargs['pod_name_prefix'],
            started_instance_id_path=tmp_path/'started-instance.txt')

    def consume(*args, **kwargs):
        events.append('consume')
        return {'status': 'consumed'}

    def adapter(**kwargs):
        events.append('adapter')
        assert events == ['privacy', 'stage', 'arm', 'consume', 'adapter']
        assert tuple(kwargs['allowed_geolocation_country_codes']) == ('US',)
        preference = kwargs['preferred_geolocation_regex']
        assert re.search(preference, 'California, US', re.I) or re.search(preference, 'US, California', re.I)
        assert not re.search(preference, 'New York, US', re.I)
        assert kwargs['hard_cap_usd'] == kwargs['target_spend_usd'] == 1.
        assert kwargs['max_hourly_rate'] == .8
        assert kwargs['max_live_minutes'] == kwargs['session_max_live_minutes'] == 30
        assert kwargs['startup_timeout_seconds'] == 1800
        mutable_avoidlist = kwargs['machine_avoidlist_path']
        assert mutable_avoidlist != avoidlist
        assert mutable_avoidlist.read_bytes() == avoidlist.read_bytes()
        mutable_avoidlist.write_text('{"machine_ids":[20166]}')
        assert avoidlist.read_text() == '{}'
        assert kwargs['allowed_active_instance_ids'] == ()
        assert os.environ['BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS'] == pre_mutation_offer_reselection_attempts()
        assert kwargs['forward_hf_token'] is False
        assert kwargs['instance_label_prefix'] == lane.RETAINED_RENDER_INSTANCE_LABEL_PREFIX
        root = kwargs['job_dir']
        root.mkdir(parents=True)
        ids = [123, 456] if defect == 'multiple_instances' else [123]
        result = {'status': 'completed', 'vast_instance_ids': ids, 'estimated_cost_usd': .1}
        (root/'vast_provider_adapter_result.json').write_text(json.dumps(result))
        (root/'vast_teardown_manifest.json').write_text(json.dumps({
            'vast_instance_ids': ids, 'continuing_spend_from_this_run': defect == 'teardown',
            'runner_gpu_teardown_completed': defect != 'teardown'}))
        return result

    def cleanup(*args, **kwargs):
        events.append('cleanup')
        assert kwargs == store_files
        return {'all_objects_absent': defect != 'cleanup'}

    def close(**kwargs):
        events.append('watchdog-close')
        assert events.count('adapter') == 1
        return {'status': 'provider_terminal', 'provider_absence_confirmed': defect != 'watchdog'}

    def extract(_archive, root, *args, **kwargs):
        root.mkdir(parents=True)
        value = {'schema_version': 'adp009d_source_calibration_gpu_render_result.v1',
                 'render_scope': 'source_calibration', 'status': 'completed' if defect != 'output' else 'blocked',
                 'released_renderer_executed': True, 'gpu_runtime_started': True,
                 'candidate_policy_queried': False, 'paid_inference_performed': False,
                 'provider_mutations_performed': 0}
        (root/'adp009d_source_calibration_gpu_render_result.v1.json').write_text(json.dumps(value))
        return_path = root/'source_calibration_render_return.v1.json'
        return_path.write_text('{}')
        return value, [], {'schema_version': 'source_calibration_return_relocation.v1',
                          'return_path': str(return_path), 'return_digest': 'sha256:'+'c'*64}

    monkeypatch.setattr(lane, 'stage_wam_provider_bundle_object_store', stage)
    monkeypatch.setattr(lane, 'arm_independent_vast_watchdog', arm)
    monkeypatch.setattr(lane, 'consume_retained_scene_render_paid_attempt_authority_once', consume)
    monkeypatch.setattr(lane, 'run_vast_provider_adapter', adapter)
    monkeypatch.setattr(lane, 'cleanup_staged_wam_provider_objects', cleanup)
    monkeypatch.setattr(lane, 'close_independent_vast_watchdog', close)
    monkeypatch.setattr(lane, '_extract_provider_output', extract)
    options = dict(job_dir=tmp_path/'job', execute=True,
        paid_resource_admission_grant=object(), prepared_bundle=bundle,
        paid_attempt_authority={'authorization_digest': 'sha256:'+'b'*64},
        max_hourly_rate_usd=.8, hard_ttl_seconds=1800, machine_avoidlist_path=avoidlist)
    if defect == 'privacy':
        with pytest.raises(ValueError, match='private_store_readback_unknown'):
            lane.run_retained_scene_render_vast(**options)
        assert events == ['privacy']
        return
    result = lane.run_retained_scene_render_vast(**options)
    assert events == ['privacy', 'stage', 'arm', 'consume', 'adapter', 'cleanup', 'watchdog-close']
    assert os.environ['BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL'] == 'https://legacy.example.invalid'
    assert result['retry_cap'] == 0
    assert result['status'] == ('completed' if defect is None else 'blocked'), result
    assert result['hard_cap_usd'] == 1. and result['hard_ttl_seconds'] == 1800
    assert Path(result['execution_result_path']).is_file()
    assert Path(result['source_calibration_return']['return_path']).is_file()
    if result.get('output_relocation_receipt'):
        assert Path(result['output_relocation_receipt']['path']).is_file()


@pytest.mark.parametrize('defect', [None, 'legacy_endpoint', 'wrong_bucket', 'native_endpoint', 'foreign_api',
                                  'public_bucket', 'missing_bucket', 'duplicate_bucket'])
def test_private_source_store_binds_native_provider_endpoint_and_bucket(tmp_path, monkeypatch, defect):
    from blueprint_pipeline import source_calibration_private_store as store

    values = {'access_key_id_file': 'TEST_B2_KEY_ID', 'secret_access_key_file': 'TEST_B2_APPLICATION_KEY',
              'endpoint_url_file': 'https://s3.us-east-005.backblazeb2.com',
              'bucket_file': store.EXPECTED_BUCKET, 'region_file': 'us-east-005'}
    if defect == 'legacy_endpoint':
        values['endpoint_url_file'] = 'https://sfo3.digitaloceanspaces.com'
    elif defect == 'wrong_bucket':
        values['bucket_file'] = 'unrelated-public-bucket'
    files = {}
    for key, value in values.items():
        path = tmp_path/key
        path.write_text(value)
        path.chmod(0o600)
        files[key] = str(path)
        monkeypatch.setenv(store.ENV[key], str(path))
    token = 'TEST_AUTHORIZATION_TOKEN_DO_NOT_PERSIST'
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url))
        if method == 'GET':
            assert url.startswith('https://api.backblazeb2.com/b2api/')
            assert headers['Authorization'].startswith('Basic ')
            return {'accountId': 'fixture-account', 'authorizationToken': token, 'apiInfo': {'storageApi': {
                'apiUrl': 'https://foreign.example.invalid' if defect == 'foreign_api' else 'https://api005.backblazeb2.com',
                's3ApiUrl': 'https://s3.us-west-004.backblazeb2.com' if defect == 'native_endpoint' else values['endpoint_url_file']}}}
        assert method == 'POST' and url.startswith('https://api005.backblazeb2.com/b2api/')
        assert headers['Authorization'] == token
        assert body == {'accountId': 'fixture-account', 'bucketName': store.EXPECTED_BUCKET}
        row = {'bucketName': store.EXPECTED_BUCKET, 'bucketId': 'fixture-bucket-id',
               'bucketType': 'allPublic' if defect == 'public_bucket' else 'allPrivate'}
        return {'buckets': [] if defect == 'missing_bucket' else [row, row] if defect == 'duplicate_bucket' else [row]}

    path = tmp_path/'private-store-readback.json'
    if defect:
        with pytest.raises(ValueError):
            store.verify_private_source_store(output_path=path, transport=transport)
        assert not path.exists()
        assert len(calls) == (0 if defect in {'legacy_endpoint', 'wrong_bucket'} else
                              1 if defect in {'native_endpoint', 'foreign_api'} else 2)
    else:
        result = store.verify_private_source_store(output_path=path, transport=transport)
        assert result['staging_kwargs'] == files
        assert result['readback']['provider'] == 'backblaze_b2'
        assert result['readback']['s3_endpoint'] == values['endpoint_url_file']
        assert result['readback']['bucket'] == store.EXPECTED_BUCKET
        assert result['readback']['bucket_type'] == 'allPrivate'
        assert len(calls) == 2
        retained = path.read_text()
        assert token not in retained
        assert values['access_key_id_file'] not in retained
        assert values['secret_access_key_file'] not in retained


def _json_record(path):
    return {'path': str(path), 'sha256': 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest(),
            'size_bytes': path.stat().st_size}


def _seal_json(path, value, field):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json, cross_runtime_canonical_digest
    node_record = value.get('schema_version') in {
        'sealed_camera_render_manifest.v1', 'adp009d_source_calibration_gpu_render_result.v1'}
    if node_record:
        value['digest_canonicalization'] = 'rfc8785'
    digest = cross_runtime_canonical_digest if node_record else canonical_digest
    value[field] = digest(value, digest_field=field)
    path.write_text(canonical_json(value) + '\n')


def _prepared_source(tmp_path, monkeypatch):
    from tests import test_public_scene_inpainting_preparation as cpu
    original = cpu._write_v2_fixture

    def with_renderer(root, **kwargs):
        paths = original(root, **kwargs)
        repo = paths['repo']
        # Commit a tiny hermetic renderer identity before CPU preparation takes
        # its immutable checkout snapshot. No production renderer is executed.
        files = {'tools/splat_render/render_splat.mjs': '// fixture harness\n',
                 'tools/splat_render/src/render_entry.mjs': '// fixture entry\n',
                 'tools/splat_render/package.json': '{"name":"blueprint-splat-render","version":"1.0.0"}\n',
                 'tools/splat_render/package-lock.json': '{"lockfileVersion":3,"packages":{}}\n'}
        for name, text in files.items():
            path = repo/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True, capture_output=True)
        subprocess.run(['git', '-C', str(repo), 'commit', '-qm', 'fixture renderer identity'], check=True, capture_output=True)
        return paths

    monkeypatch.setattr(cpu, '_write_v2_fixture', with_renderer)
    return cpu._prepare(tmp_path)


def _rendered_source_matrix(prepared, root):
    from blueprint_pipeline.sealed_camera_render import _camera_specs_from_calibration_file
    from tests.test_public_scene_inpainting_inputs import _fake_sealed_render
    from blueprint_pipeline import source_calibration_render_return as returned

    groups = []
    for role, layer in prepared['layers'].items():
        directory = root/role
        manifest = _fake_sealed_render(output_dir=directory, cameras=prepared['cameras'])
        repo = Path(prepared['context']['paths']['repo'])
        manifest.update(schema_version='sealed_camera_render_manifest.v1', status='rendered_exact_cameras',
            rendered_by='reference_spark_renderer_exact_camera', rendered_by_gpu=True,
            authorization_class='method_input', source_layer_role=role,
            source_calibration_preparation_digest=prepared['preparation_digest'],
            camera_set_label=layer['camera_set_label'], purpose=layer['purpose'],
            provider_splat_import_receipt_digest=layer['provider_splat_import_receipt_digest'],
            provider_reconstruction_alignment_digest=layer['alignment_digest'],
            alignment_digest=layer['alignment_digest'], candidate_policy_queried=False, paid_inference_performed=False,
            calibrated_cameras=_camera_specs_from_calibration_file(Path(prepared['camera_file']['path'])),
            calibrated_camera_file={'digest': prepared['camera_file']['sha256'], 'camera_count': 16,
                                    'binding': 'caller_file_exact_match'},
            source_splat={'digest': layer['sha256'], 'retained_gaussian_count': layer['retained_gaussian_count']},
            splat_digest=layer['sha256'], render_count=16,
            gpu_identity={'nvidia_smi_detected': True, 'gpu_rows': ['NVIDIA RTX 4090 fixture']},
            renderer_identity={'repository': prepared['repository'], 'graphics_backend': 'egl',
                'graphics_diagnostics': {'webgl_available': True, 'renderer': 'NVIDIA RTX 4090 fixture'},
                'harness_sha256': _json_record(repo/'tools/splat_render/render_splat.mjs')['sha256'],
                'render_entry_sha256': _json_record(repo/'tools/splat_render/src/render_entry.mjs')['sha256'],
                'package_manifest_sha256': _json_record(repo/'tools/splat_render/package.json')['sha256'],
                'package_lock_sha256': _json_record(repo/'tools/splat_render/package-lock.json')['sha256']})
        manifest['renders'] = [{'camera_id': camera['camera_id'],
            'relative_path': f"frames/{camera['camera_id']}.png",
            'digest': _json_record(directory/'frames'/f"{camera['camera_id']}.png")['sha256'],
            'size_bytes': (directory/'frames'/f"{camera['camera_id']}.png").stat().st_size,
            'width': 1280, 'height': 1280} for camera in prepared['cameras']]
        path = directory/'sealed_camera_render_manifest.v1.json'
        _seal_json(path, manifest, 'sealed_camera_render_manifest_digest')
        groups.append({'role': role, 'manifest': {**_json_record(path), 'relative_path': str(path.relative_to(root))}})
    result = {'schema_version': returned.RESULT_SCHEMA, 'status': 'completed', 'render_scope': 'source_calibration',
              'preparation_digest': prepared['preparation_digest'], 'blueprint_commit': prepared['repository']['commit'],
              'candidate_policy_queried': False, 'paid_inference_performed': False, 'provider_mutations_performed': 0,
              'released_renderer_executed': True, 'gpu_runtime_started': True, 'render_groups': groups}
    path = root/(returned.RESULT_SCHEMA + '.json')
    _seal_json(path, result, 'result_digest')
    return path, result


@pytest.mark.parametrize('untyped_full_source_permission', [False, True])
def test_frame_and_gpu_authority_never_disclose_full_source_before_bundle_creation(
    tmp_path, monkeypatch, untyped_full_source_permission,
):
    from blueprint_pipeline import source_calibration_render_packet as packet

    paths, prepared = _prepared_source(tmp_path, monkeypatch)
    source = prepared['layers']['images']
    original = paths['data']/'original-commented-source.ply'
    original.write_bytes(Path(source['path']).read_bytes().replace(b'ply\n', b'ply\ncomment fixture original encoding\n', 1))
    original_record = _json_record(original)
    # Synthetic identity conversion of the actual fixture source bytes. This
    # receipt grants local conversion only; neither it nor paid/frame consent
    # grants whole-scene provider processing.
    conversion = {'schema_version': 'standard_splat_conversion_receipt.v1',
        'status': 'standard_splat_conversion_materialized', 'repository': prepared['repository'],
        'claim_ceiling': 'local_format_conversion_only',
        'source': {'sha256': original_record['sha256'], 'size_bytes': original_record['size_bytes'],
                   'source_bytes_unchanged': True, 'source_gaussian_count': source['retained_gaussian_count'],
                   'dataset': 'synthetic-fixture', 'revision': 'a'*40},
        'output': {'sha256': source['sha256'], 'size_bytes': source['size_bytes'],
                   'gaussian_count': source['retained_gaussian_count'], 'gaussian_count_preserved': True,
                   'standard_3dgs_schema_validated': True},
        'rights': {'conversion_execution_location': 'local_only', 'raw_private_upload_authorized': False,
                   'training_authorized': False, 'terms_digest': 'sha256:'+'b'*64}}
    conversion_path = paths['data']/'local-only-conversion.json'
    _seal_json(conversion_path, conversion, 'receipt_digest')
    authority = {'accepted_by': 'fixture-owner', 'private_derived_frame_disclosure_authorized': True,
                 'source_calibration_gpu_render_authorized': True,
                 'max_source_calibration_gpu_render_cost_usd': 1.}
    if untyped_full_source_permission:
        authority['full_source_provider_disclosure_authority'] = True
    task_path = paths['data']/'frame-and-gpu-only-task.json'
    task_path.write_text(json.dumps({'publisher_scene_id': 'fixture', 'human_authority': authority}))
    output = paths['data']/'must-not-stage-source'
    monkeypatch.setattr(packet, '_copy_tree', lambda *args, **kwargs: pytest.fail('bundle copied before rights admission'))
    monkeypatch.setattr(packet, 'rehearse_provider_bundle_entrypoint',
                        lambda *args, **kwargs: pytest.fail('bundle executed before rights admission'))
    with pytest.raises(ValueError, match='explicit_full_source_authority_required'):
        packet.build_source_calibration_gpu_render_bundle(
            prepared_inputs_path=prepared['preparation_path'], repo_root=paths['repo'],
            renderer_vendor_root=tmp_path/'must-not-read-vendor', task_request_path=task_path,
            conversion_receipt_path=conversion_path, original_source_path=original,
            job_dir=output, expected_source_commit=prepared['repository']['commit'], approved_roots=(tmp_path,))
    assert not output.exists()


@pytest.mark.parametrize('defect', [None, '47_frames', 'duplicate_camera', 'missing_group', 'clipping',
    'background', 'alpha', 'software', 'code_identity', 'source_digest', 'gaussian_count',
    'policy_query', 'inference', 'frame_bytes'])
def test_exact_48_frame_return_is_required_before_publishing_or_finalizing(tmp_path, monkeypatch, defect):
    from blueprint_pipeline import source_calibration_render_return as returned
    from blueprint_pipeline.public_scene_inpainting_inputs import finalize_public_scene_inpainting_inputs

    paths, prepared = _prepared_source(tmp_path, monkeypatch)
    result_path, result = _rendered_source_matrix(prepared, paths['data']/'provider-output')
    group = result['render_groups'][0]
    manifest_path = Path(group['manifest']['path'])
    manifest = json.loads(manifest_path.read_text())
    first_frame = manifest_path.parent/manifest['renders'][0]['relative_path']
    if defect == '47_frames':
        first_frame.unlink()
    elif defect == 'duplicate_camera':
        manifest['renders'][-1] = deepcopy(manifest['renders'][0])
    elif defect == 'missing_group':
        result['render_groups'].pop()
    elif defect == 'clipping':
        manifest['calibrated_cameras'][0]['spec']['intrinsics']['near'] += .1
    elif defect == 'background':
        manifest['render_settings']['background_rgb'] = '#ffffff'
    elif defect == 'alpha':
        manifest['render_settings']['alpha_mode'] = 'premultiplied'
    elif defect == 'software':
        manifest['renderer_identity']['graphics_diagnostics']['renderer'] = 'ANGLE SwiftShader'
    elif defect == 'code_identity':
        manifest['renderer_identity']['harness_sha256'] = 'sha256:'+'f'*64
    elif defect == 'source_digest':
        manifest['source_splat']['digest'] = 'sha256:'+'f'*64
    elif defect == 'gaussian_count':
        manifest['source_splat']['retained_gaussian_count'] += 1
    elif defect == 'policy_query':
        result['candidate_policy_queried'] = True
    elif defect == 'inference':
        result['paid_inference_performed'] = True
    elif defect == 'frame_bytes':
        first_frame.write_bytes(b'corrupt-returned-png')
    _seal_json(manifest_path, manifest, 'sealed_camera_render_manifest_digest')
    group['manifest'].update(_json_record(manifest_path))
    _seal_json(result_path, result, 'result_digest')
    final_path = paths['data']/'source-calibration-return.json'
    if defect:
        with pytest.raises(ValueError):
            returned.materialize_source_calibration_return(prepared_inputs=prepared, result_path=result_path,
                                                          output_path=final_path)
        assert not final_path.exists(), 'Invalid returned evidence published a success-looking receipt'
        assert not (paths['output']/'public_scene_interiorgs_edit_input_receipt.v2.json').exists()
    else:
        returned.materialize_source_calibration_return(prepared_inputs=prepared, result_path=result_path,
                                                      output_path=final_path)
        verified = returned.verify_source_calibration_return(prepared, final_path)
        assert set(verified) == set(returned.ROLES)
        assert sum(len(value['manifest']['renders']) for value in verified.values()) == 48
        with pytest.raises(ValueError, match='execution_closure_missing'):
            finalize_public_scene_inpainting_inputs(preparation_path=prepared['preparation_path'], returned_group_path=final_path)
        assert not list(paths['output'].glob('*/frames/*.png'))


def _closed_source_fixture(tmp_path, monkeypatch):
    from tests.test_source_calibration_render_packet import valid_source_bundle_inputs
    from tests.test_vast_official_billing_extractor import _charge
    from blueprint_pipeline import source_calibration_render_packet as packet
    from blueprint_pipeline import source_calibration_render_return as returned
    from blueprint_pipeline import source_calibration_private_store as store
    from blueprint_pipeline.vast_official_billing_extractor import extract_vast_official_instance_charge
    from blueprint_pipeline.provider_billing_reconciler import BILLING_SOURCE_SCHEMA_VERSION, VAST_CHARGES_URL

    args, prepared = valid_source_bundle_inputs(tmp_path, monkeypatch)
    bundle = packet.build_source_calibration_gpu_render_bundle(**args)
    root = args['job_dir'].parent/'closed-evidence'
    root.mkdir()
    result_path, _ = _rendered_source_matrix(prepared, root/'provider-output')
    render_return = root/'render-only-return.json'
    returned.materialize_source_calibration_return(prepared_inputs=prepared, result_path=result_path,
                                                  output_path=render_return)
    disclosure_path = root/'source-disclosure.json'
    _seal_json(disclosure_path, bundle['source_disclosure'], 'proof_digest')
    configuration = {'access_key_id_file': 'fixture-access', 'secret_access_key_file': 'fixture-secret',
                     'endpoint_url_file': 'https://s3.us-east-005.backblazeb2.com',
                     'bucket_file': store.EXPECTED_BUCKET, 'region_file': 'us-east-005'}
    for key, value in configuration.items():
        path = root/key
        path.write_text(value)
        path.chmod(0o600)
        monkeypatch.setenv(store.ENV[key], str(path))

    def native_read(method, url, headers, body):
        if method == 'GET':
            return {'accountId': 'fixture-account', 'authorizationToken': 'fixture-token',
                    'apiInfo': {'storageApi': {'apiUrl': 'https://api005.backblazeb2.com',
                                               's3ApiUrl': configuration['endpoint_url_file']}}}
        return {'buckets': [{'bucketName': store.EXPECTED_BUCKET, 'bucketType': 'allPrivate',
                             'bucketId': 'fixture-bucket'}]}

    private_path = root/'private-store.json'
    private = store.verify_private_source_store(output_path=private_path, transport=native_read)['readback']
    instance = 123
    label = 'blueprint-adp-retained-render-source-fixture'
    teardown_path = root/'vast_teardown_manifest.json'
    teardown_path.write_text(json.dumps({'schema_version': 'vast_teardown_manifest.v1',
        'vast_instance_ids': [instance], 'continuing_spend_from_this_run': False, 'runner_gpu_teardown_completed': True}))
    execution_path = root/'retained_scene_render_vast_result.json'
    execution = {'schema_version': 'adp009d_retained_scene_gpu_render_vast_run.v1',
        'status': 'completed', 'render_scope': 'source_calibration', 'vast_instance_ids': [instance],
        'bundle_sha256': bundle['bundle_sha256'], 'continuing_spend_from_this_run': False,
        'all_staged_objects_absent': True, 'private_source_store_readback': private,
        'execution_result_path': str(result_path), 'teardown_manifest_path': str(teardown_path)}
    _seal_json(execution_path, execution, 'receipt_digest')
    inventory = {'status': 'observed', 'provider': 'vast', 'name_prefix': '', 'api_confirmed': True,
                 'live_resource_count': 0, 'resources': []}
    inspect = {'instance_id': instance, 'status': 'absent', 'api_confirmed': True, 'provider_absence_confirmed': True}
    zero_path = root/'provider-zero.json'
    zero_path.write_text(json.dumps({'status': 'provider_terminal', 'provider': 'vast',
        'provider_absence_confirmed': True, 'recorded_vast_instance_teardown': {
            'instance_id': instance, 'status': 'absent', 'provider_absence_confirmed': True,
            'inspect_attempts': [dict(inspect), dict(inspect)]},
        'initial_global_inventory': dict(inventory), 'final_global_inventory': dict(inventory)}))
    response_path = root/'official-vast-response.json'
    response_path.write_text(json.dumps({'success': True, 'results': [_charge(instance_id=instance,
        label=label, total=.4, gpu=.3, disk=.1)]}))
    source_path = root/'provider-billing-source.json'
    _seal_json(source_path, {'schema_version': BILLING_SOURCE_SCHEMA_VERSION, 'status': 'reconciled',
        'provider_totals_usd': {'vast': .4}, 'provider_mutation_performed': False, 'raw_secret_values_recorded': False,
        'sources': [{'provider': 'vast', 'retained_path': str(response_path), 'endpoint': VAST_CHARGES_URL,
                     'response_digest': _json_record(response_path)['sha256'],
                     'response_size_bytes': response_path.stat().st_size}]}, 'receipt_digest')
    charge = extract_vast_official_instance_charge(provider_billing_source_receipt_path=source_path,
                                                  instance_id=instance, launch_label=label)
    charge_path = root/'official-charge.json'
    _seal_json(charge_path, charge, 'charge_digest')
    closure = {name: _json_record(path) for name, path in {
        'source_disclosure': disclosure_path, 'private_store': private_path, 'provider_execution': execution_path,
        'official_billing': charge_path, 'teardown': teardown_path, 'provider_zero': zero_path}.items()}
    return prepared, render_return, closure, root


@pytest.mark.parametrize('defect', [None, 'missing_billing', 'source_scope', 'public_store', 'foreign_execution_result',
    'execution_schema', 'teardown_schema', 'foreign_teardown', 'foreign_inspect', 'one_inspect',
    'initial_scoped_zero', 'final_nonzero', 'changed_charge'])
def test_finalization_requires_real_disclosure_billing_and_exact_terminal_closure(tmp_path, monkeypatch, defect):
    from blueprint_pipeline import source_calibration_render_return as returned
    from blueprint_pipeline.public_scene_inpainting_inputs import finalize_public_scene_inpainting_inputs

    prepared, render_return, closure, root = _closed_source_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='execution_closure_missing'):
        returned.require_source_calibration_closure(prepared, render_return)
    if defect == 'missing_billing':
        closure.pop('official_billing')
    elif defect:
        role = {'source_scope': 'source_disclosure', 'public_store': 'private_store',
                'foreign_execution_result': 'provider_execution', 'execution_schema': 'provider_execution',
                'teardown_schema': 'teardown', 'foreign_teardown': 'teardown',
                'changed_charge': 'official_billing'}.get(defect, 'provider_zero')
        path = Path(closure[role]['path'])
        value = json.loads(path.read_text())
        if defect == 'source_scope':
            value['purpose'] = 'released_code_segment_contribution_sweep'
        elif defect == 'public_store':
            value['bucket_type'] = 'allPublic'
        elif defect == 'foreign_execution_result':
            foreign = root/'different-provider-result.json'
            foreign.write_bytes(Path(value['execution_result_path']).read_bytes())
            value['execution_result_path'] = str(foreign)
        elif defect in {'execution_schema', 'teardown_schema'}:
            value['schema_version'] = 'unrelated_receipt.v1'
        elif defect == 'foreign_teardown':
            value['vast_instance_ids'] = [456]
        elif defect == 'foreign_inspect':
            value['recorded_vast_instance_teardown']['inspect_attempts'][1]['instance_id'] = 456
        elif defect == 'one_inspect':
            value['recorded_vast_instance_teardown']['inspect_attempts'].pop()
        elif defect == 'initial_scoped_zero':
            value['initial_global_inventory']['name_prefix'] = 'only-this-run'
        elif defect == 'final_nonzero':
            value['final_global_inventory'].update(live_resource_count=1, resources=[{'id': '456'}])
        elif defect == 'changed_charge':
            value['official_charge_usd'] = .99
        field = {'source_disclosure': 'proof_digest', 'private_store': 'readback_digest',
                 'provider_execution': 'receipt_digest', 'official_billing': 'charge_digest'}.get(role)
        if field:
            _seal_json(path, value, field)
        else:
            path.write_text(json.dumps(value))
        closure[role] = _json_record(path)
    closed_path = root/'closed-return.json'
    if defect:
        with pytest.raises(ValueError):
            returned.materialize_source_calibration_closed_return(prepared_inputs=prepared,
                returned_group_path=render_return, execution_closure=closure, output_path=closed_path)
        assert not closed_path.exists()
        assert not list(Path(prepared['context']['paths']['output']).glob('*/frames/*.png'))
    else:
        result = returned.materialize_source_calibration_closed_return(prepared_inputs=prepared,
            returned_group_path=render_return, execution_closure=closure, output_path=closed_path)
        assert returned.require_source_calibration_closure(prepared, closed_path) == result
        final = finalize_public_scene_inpainting_inputs(preparation_path=prepared['preparation_path'],
                                                       returned_group_path=closed_path)
        assert final['status'] == 'render_derived_input_packet_materialized'
        assert len(final['derived_artifacts']['images']) == len(final['derived_artifacts']['masks']) == 16
        assert final['source_calibration_render']['execution_closure'] == closure
        assert final['source_calibration_render']['full_source_scene_content_transferred'] is True
