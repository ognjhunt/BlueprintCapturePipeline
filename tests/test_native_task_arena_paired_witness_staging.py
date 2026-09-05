"""A paired-cell transfer uses private capabilities for a distinct fresh key."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from types import SimpleNamespace

import pytest

from blueprint_pipeline import native_task_arena_paired_witness_staging as paired
from blueprint_pipeline import wam_provider_object_store as store
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


class Client:
    def __init__(self):
        self.objects, self.signed, self.deleted = {}, [], []
        self.existing_witness = False

    def head_object(self, *, Bucket, Key):
        if Key in self.objects or self.existing_witness and Key.endswith(paired.SUFFIX):
            return {'ResponseMetadata': {'HTTPStatusCode': 200}}
        error = RuntimeError('missing')
        error.response = {'ResponseMetadata': {'HTTPStatusCode': 404}, 'Error': {'Code': 'NoSuchKey'}}
        raise error

    def upload_file(self, source, bucket, key):
        self.objects[key] = b'bundle'

    def delete_object(self, *, Bucket, Key):
        self.deleted.append(Key)
        self.objects.pop(Key, None)
        return {'ResponseMetadata': {'HTTPStatusCode': 204}}

    def generate_presigned_url(self, operation, *, Params, ExpiresIn, HttpMethod):
        self.signed.append((operation, dict(Params), ExpiresIn, HttpMethod))
        return f'https://nyc3.digitaloceanspaces.com/{Params["Bucket"]}/{Params["Key"]}?X-Amz-Signature=private-capability-{operation}'


@pytest.fixture
def context(tmp_path, monkeypatch):
    client = Client()
    monkeypatch.setitem(sys.modules, 'boto3', SimpleNamespace(client=lambda *a, **kw: client))
    monkeypatch.setitem(sys.modules, 'botocore.client', SimpleNamespace(Config=lambda **kw: None))
    monkeypatch.setattr(store, '_signed_output_round_trip_preflight', lambda *a, **kw: {'status': 'passed', 'blockers': []})
    for name, value in [('ACCESS_KEY_ID', 'access'), ('SECRET_ACCESS_KEY', 'secret'),
                        ('BUCKET', 'bucket'), ('ENDPOINT_URL', 'https://nyc3.digitaloceanspaces.com'),
                        ('REGION', 'nyc3')]:
        path = tmp_path/(name+'.secret')
        path.write_text(value)
        path.chmod(0o600)
        monkeypatch.setenv('BLUEPRINT_WAM_OBJECT_STORE_'+name+'_FILE', str(path))
    bundle = tmp_path/'bundle.zip'
    plan = {'cadence': {'maximum_action_steps': 360, 'settle_window_samples': 20},
            'cameras': [{'role': role, 'intrinsics': {'width': 640 if role != 'overview' else 1280,
                        'height': 360 if role != 'overview' else 720}} for role in ('external', 'wrist', 'overview')]}
    with zipfile.ZipFile(bundle, 'w') as archive:
        archive.writestr(paired.PLAN_MEMBER, json.dumps(plan))
        for candidate, horizon in (('pi05_droid', 16), ('groot_n17_droid', 8)):
            archive.writestr(f"provider_runtime/runtime_inputs/policy_execution_spec.{candidate}.json",
                             json.dumps({'max_policy_queries': (360 + horizon - 1)//horizon, 'open_loop_horizon': horizon}))
    binding = {'run_id': 'scene841757-paired-test', 'authority_digest': 'sha256:'+'a'*64,
        'runtime_inputs_digest': 'sha256:'+'b'*64, 'implementation_commit': 'c'*40,
        'provider_bundle_sha256': 'sha256:'+hashlib.sha256(bundle.read_bytes()).hexdigest()}
    binding = paired.build_paired_witness_binding({'bundle_path': str(bundle), 'bundle_size_bytes': bundle.stat().st_size,
        'bundle_sha256': binding['provider_bundle_sha256'], 'implementation_commit': binding['implementation_commit']},
        binding, {'BLUEPRINT_ADP009D_CAMERA_RESOLUTION': '640x360'})
    return client, bundle, binding


def stage(context, tmp_path):
    _, bundle, binding = context
    return store.stage_wam_provider_bundle_object_store(job_dir=tmp_path/'staging', bundle_path=bundle,
        key_prefix='blueprint/policy-canary', generated_at='2026-09-05T03:00:00Z', expiration_seconds=18000,
        paired_witness_binding=binding)


def test_distinct_capabilities_bind_session_code_and_private_put_metadata(context, tmp_path):
    client, _, binding = context
    result = stage(context, tmp_path)
    assert result['status'] == 'completed'
    witness = result['paired_witness']
    assert witness['witness_key'] == result['output_key'] + paired.SUFFIX
    assert witness['witness_key'] not in client.objects
    authority = witness['authority']
    assert authority['binding_digest'] == canonical_digest(authority, digest_field='binding_digest')
    assert all(authority[k] == v for k, v in binding.items())
    assert authority['maximum_archive_bytes'] == binding['maximum_archive_bytes']
    assert 1_000_000_000 < authority['maximum_archive_bytes'] < paired.MAXIMUM_SLOT_CAPACITY_BYTES
    assert authority['expires_at'] == '2026-09-05T08:00:00Z'
    puts = [params for op, params, _, _ in client.signed if op == 'put_object' and params['Key'] == witness['witness_key']]
    assert puts[-1]['Metadata']['blueprint-witness-binding'] == authority['binding_digest']
    secrets = paired.paired_witness_secret_paths(tmp_path/'staging', result, binding)
    assert set(secrets) == set(paired.SECRET_FILES)
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in secrets.values())
    assert 'private-capability-' not in json.dumps(result)
    assert result['paired_witness']['raw_signed_urls_recorded'] is False
    assert (tmp_path/'staging/provider_output_put_url.txt').read_text() != secrets['BLUEPRINT_POLICY_CANARY_PAIRED_PUT_URL_FILE'].read_text()


def test_cleanup_includes_exact_witness_and_removes_secret_capabilities(context, tmp_path):
    client, _, _ = context
    result = stage(context, tmp_path)
    client.objects[result['paired_witness']['witness_key']] = b'actual-paired-evidence'
    cleanup = store.cleanup_staged_wam_provider_objects(tmp_path/'staging')
    assert cleanup['status'] == 'completed'
    assert cleanup['exact_object_count'] == 3
    assert cleanup['all_objects_absent'] is True
    assert result['paired_witness']['witness_key'] in client.deleted
    assert not any((tmp_path/'staging'/p).exists() for p in paired.SECRET_FILES.values())


def test_existing_witness_cannot_be_adopted_as_new_transfer(context, tmp_path):
    client, _, _ = context
    client.existing_witness = True
    result = stage(context, tmp_path)
    assert result['status'] == 'blocked'
    assert result['paired_witness']['status'] == 'not_required'
    assert not any((tmp_path/'staging'/p).exists() for p in paired.SECRET_FILES.values())
    store.cleanup_staged_wam_provider_objects(tmp_path/'staging')
    assert not any(key.endswith(paired.SUFFIX) for key in client.deleted)


@pytest.mark.parametrize('change', ['run', 'url', 'mode'])
def test_secret_forwarding_rejects_identity_url_or_mode_drift(context, tmp_path, change):
    _, _, binding = context
    result = stage(context, tmp_path)
    if change == 'run':
        binding = {**binding, 'run_id': 'other-run'}
    else:
        path = tmp_path/'staging/paired_witness_put_url.txt'
        if change == 'url':
            path.write_text('https://nyc3.digitaloceanspaces.com/bucket/another-object?X-Amz-Signature=secret')
        else:
            path.chmod(0o644)
    with pytest.raises(ValueError, match='paired_witness_'):
        paired.paired_witness_secret_paths(tmp_path/'staging', result, binding)


def test_witness_binding_cannot_refer_to_another_runtime_bundle(context, tmp_path):
    client, bundle, binding = context
    binding['provider_bundle_sha256'] = 'sha256:'+'f'*64
    result = stage((client, bundle, binding), tmp_path)
    assert result['status'] == 'blocked'
    assert 'paired_witness_provider_bundle_digest_mismatch' in result['blockers']
    assert not client.signed


def test_tampered_cleanup_key_cannot_delete_other_objects(context, tmp_path):
    client, _, _ = context
    result = stage(context, tmp_path)
    path = tmp_path/'staging/wam_provider_object_store_staging_manifest.json'
    result['paired_witness']['witness_key'] = 'blueprint/policy-canary/unrelated.zip'
    path.write_text(json.dumps(result))
    cleanup = store.cleanup_staged_wam_provider_objects(tmp_path/'staging')
    assert cleanup['status'] == 'blocked'
    assert 'paired_witness_cleanup_key_mismatch' in cleanup['blockers']
    assert not client.deleted


def test_runtime_secret_bootstrap_keeps_urls_out_of_public_environment(context, tmp_path):
    from blueprint_pipeline.vast_provider_adapter import _probe_env, _runtime_secret_file_values
    _, _, binding = context
    result = stage(context, tmp_path)
    files = paired.paired_witness_secret_paths(tmp_path/'staging', result, binding)
    values = _runtime_secret_file_values(files)
    env = _probe_env(job_dir=tmp_path, enable_isaac_smoke=False, forward_hf_token=False,
        provider_bundle_kind='native_task_arena_policy_canary_session', runtime_secret_file_values=values)
    assert not any(name in env for name in paired.SECRET_FILES)
    assert 'private-capability-' not in json.dumps(env)
    assert sum(name.endswith('_FILE') and 'PAIRED' in name for name in env) == 3


def test_paired_wrapper_binds_private_slot_and_budgets_roundtrip_bytes(context, tmp_path, monkeypatch):
    from blueprint_pipeline import native_task_arena_vast as native
    _, bundle_path, binding = context
    authority = {**binding, 'hard_cap_usd': 4., 'hard_ttl_seconds': 9000,
                 'resource_name': 'blueprint-native-task-policy-canary-test'}
    bundle = {'implementation_commit': binding['implementation_commit'],
              'bundle_sha256': binding['provider_bundle_sha256'], 'container_image': 'immutable-image',
              'bundle_path': str(bundle_path), 'bundle_size_bytes': bundle_path.stat().st_size}
    monkeypatch.setattr(native, 'validate_policy_canary_session_authority', lambda value: value)
    monkeypatch.setattr(native, 'validate_policy_canary_provider_bundle', lambda *a, **kw: bundle)
    monkeypatch.setattr(native, '_policy_provider_transfer_byte_budget', lambda candidate: (100, 20))
    monkeypatch.setattr(native, 'run_arena_native_control_vast', lambda **kwargs: kwargs)
    result = native.run_native_task_arena_policy_canary_session_vast(job_dir=tmp_path,
        prepared_bundle=bundle, session_authority=authority, paid_resource_admission_grant=None,
        execute=False, hard_ttl_seconds=9000,
        provider_runtime_environment={'BLUEPRINT_ADP009D_CAMERA_RESOLUTION': '640x360'})
    assert result['paired_witness_binding'] == binding
    assert result['expected_provider_upload_bytes'] == 8_000_000_000 + binding['maximum_archive_bytes']
    assert result['expected_provider_download_bytes'] == 200 + binding['maximum_archive_bytes']
    assert result['provider_runtime_environment'] == {'BLUEPRINT_ADP009D_CAMERA_RESOLUTION': '640x360'}
    assert result['stale_offer_create_retry_limit'] == 0


def test_capacity_matches_actual_query_frames_and_review_stride(context):
    import numpy as np
    from blueprint_pipeline.adp009d_policy_episode import _project_media_reserve_bytes
    _, _, binding = context
    basis = binding['capacity_basis']
    assert basis['review_frame_stride_steps'] == 8
    assert basis['policy_master_width'] == 640 and basis['policy_master_height'] == 360
    rgb = np.zeros((360, 640, 3), dtype=np.uint8)
    overview = np.zeros((720, 1280, 3), dtype=np.uint8)
    recorded = sum(_project_media_reserve_bytes(
        camera_rgb={'external': rgb, 'wrist': rgb}, evaluation_images={'external': rgb, 'wrist': rgb, 'overview': overview},
        max_policy_queries=basis['maximum_policy_queries'][candidate],
        open_loop_horizon=basis['open_loop_horizon'][candidate], settle_window_samples=20)
        for candidate in ('pi05_droid', 'groot_n17_droid'))
    assert recorded < binding['maximum_archive_bytes'] < recorded * 1.1
    assert 'provider_bundle_uncompressed_bytes' not in basis
    with pytest.raises(ValueError, match='capacity_binding_invalid'):
        paired.validate_binding({**binding, 'maximum_archive_bytes': 1_000_000_000})


def test_oversized_pair_rejected_before_authority_consumption_or_provider_call(context, tmp_path, monkeypatch):
    from blueprint_pipeline import native_task_arena_vast as native
    _, path, binding = context
    with zipfile.ZipFile(path) as archive:
        plan = json.loads(archive.read(paired.PLAN_MEMBER))
    plan['cadence']['maximum_action_steps'] = 10000
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(paired.PLAN_MEMBER, json.dumps(plan))
        for candidate in ('pi05_droid', 'groot_n17_droid'):
            archive.writestr(f"provider_runtime/runtime_inputs/policy_execution_spec.{candidate}.json",
                             json.dumps({'max_policy_queries': 10000, 'open_loop_horizon': 1}))
    bundle = {'bundle_path': str(path), 'bundle_size_bytes': path.stat().st_size,
              'bundle_sha256': 'sha256:'+hashlib.sha256(path.read_bytes()).hexdigest(),
              'implementation_commit': 'c'*40, 'container_image': 'immutable-image'}
    authority = {**binding, 'hard_cap_usd': 4., 'hard_ttl_seconds': 9000}
    monkeypatch.setattr(native, 'validate_policy_canary_session_authority', lambda value: value)
    monkeypatch.setattr(native, 'validate_policy_canary_provider_bundle', lambda *a, **kw: bundle)
    def forbidden(*args, **kwargs):
        raise AssertionError('admission must reject before consuming authority or staging')
    monkeypatch.setattr(native, 'consume_session_authority_once', forbidden)
    monkeypatch.setattr(native, 'run_arena_native_control_vast', forbidden)
    with pytest.raises(ValueError, match='capacity_exceeds_preallocation_ceiling'):
        native.run_native_task_arena_policy_canary_session_vast(job_dir=tmp_path, prepared_bundle=bundle,
            session_authority=authority, paid_resource_admission_grant=None, execute=True,
            hard_ttl_seconds=9000, provider_runtime_environment={'BLUEPRINT_ADP009D_CAMERA_RESOLUTION': '640x360'})


def test_capacity_rehashes_sealed_bundle_and_rejects_missing_overview(context, tmp_path):
    _, path, binding = context
    bundle = {'bundle_path': str(path), 'bundle_size_bytes': path.stat().st_size,
              'bundle_sha256': binding['provider_bundle_sha256'], 'implementation_commit': 'c'*40}
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='capacity_bundle_invalid'):
        paired.build_paired_witness_binding(bundle, binding)
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(paired.PLAN_MEMBER, json.dumps({'cadence': {'maximum_action_steps': 360,
            'settle_window_samples': 20}, 'cameras': []}))
        for candidate in ('pi05_droid', 'groot_n17_droid'):
            archive.writestr(f"provider_runtime/runtime_inputs/policy_execution_spec.{candidate}.json",
                             json.dumps({'max_policy_queries': 20, 'open_loop_horizon': 16}))
    bundle.update(bundle_size_bytes=path.stat().st_size,
                  bundle_sha256='sha256:'+hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='capacity_camera_inventory_missing'):
        paired.build_paired_witness_binding(bundle, binding)


def test_capacity_ignores_unused_action_ceiling_and_legacy_render_override(context):
    _, path, binding = context
    with zipfile.ZipFile(path) as archive:
        members = {row.filename: archive.read(row.filename) for row in archive.infolist()}
    plan = json.loads(members[paired.PLAN_MEMBER])
    plan['cadence']['maximum_action_steps'] = 10000
    members[paired.PLAN_MEMBER] = json.dumps(plan).encode()
    with zipfile.ZipFile(path, 'w') as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    bundle = {'bundle_path': str(path), 'bundle_size_bytes': path.stat().st_size,
              'bundle_sha256': 'sha256:'+hashlib.sha256(path.read_bytes()).hexdigest(),
              'implementation_commit': binding['implementation_commit']}
    changed = paired.build_paired_witness_binding(bundle, binding,
        {'BLUEPRINT_ADP009D_CAMERA_RESOLUTION': '1920x1080'})
    assert changed['maximum_archive_bytes'] == binding['maximum_archive_bytes']
    assert changed['capacity_basis']['policy_master_width'] == 640
