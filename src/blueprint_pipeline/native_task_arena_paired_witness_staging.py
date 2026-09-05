"""Private, session-bound intermediate result capability for paired canaries."""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest, canonical_json

SCHEMA = 'native_task_arena_paired_delivery_authority.v1'
SUFFIX = '.paired-witness.zip'
MAXIMUM_SLOT_CAPACITY_BYTES = 32 * 1024**3
PLAN_MEMBER = "provider_runtime/native_task_packet/native_task_arena_scene_plan.v1.json"
SECRET_FILES = {
    'BLUEPRINT_POLICY_CANARY_PAIRED_PUT_URL_FILE': 'paired_witness_put_url.txt',
    'BLUEPRINT_POLICY_CANARY_PAIRED_GET_URL_FILE': 'paired_witness_get_url.txt',
    'BLUEPRINT_POLICY_CANARY_PAIRED_DELIVERY_AUTHORITY_FILE': 'paired_witness_authority.json',
}
FIELDS = {'run_id', 'authority_digest', 'runtime_inputs_digest', 'implementation_commit', 'provider_bundle_sha256',
          'maximum_archive_bytes', 'capacity_basis'}


def validate_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    if (set(value) != FIELDS or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,240}', str(value.get('run_id', '')))
            or not re.fullmatch(r'[0-9a-f]{40}', str(value.get('implementation_commit', '')))
            or any(not re.fullmatch(r'sha256:[0-9a-f]{64}', str(value.get(k, '')))
                   for k in ('authority_digest', 'runtime_inputs_digest', 'provider_bundle_sha256'))):
        raise ValueError('paired_witness_binding_invalid')
    if value['maximum_archive_bytes'] != _archive_capacity(value['capacity_basis']):
        raise ValueError('paired_witness_capacity_binding_invalid')
    return dict(value)


def _archive_capacity(basis):
    fields = {'maximum_action_steps', 'settle_window_samples', 'policy_master_width', 'policy_master_height',
              'overview_width', 'overview_height', 'provider_bundle_uncompressed_bytes'}
    if (not isinstance(basis, dict) or set(basis) != fields
            or any(type(v) is not int or v < 0 for v in basis.values())
            or any(basis[k] <= 0 for k in fields - {'settle_window_samples'})
            or basis['policy_master_width'] < 320 or basis['policy_master_height'] < 180
            or basis['overview_width'] < 1280 or basis['overview_height'] < 720):
        raise ValueError('paired_witness_capacity_inputs_invalid')
    frames = basis['maximum_action_steps'] + basis['settle_window_samples'] + 2
    def image_bound(width, height):
        # RGBA worst-case plus PNG row/deflate/metadata slack; do not assume
        # favorable compression or substitute sampled policy observations.
        return width * height * 4 + 128 * 1024
    policy = image_bound(basis['policy_master_width'], basis['policy_master_height'])
    overview = image_bound(basis['overview_width'], basis['overview_height'])
    # Each policy stream: master, exact delivered lossless input, review-video
    # raw equivalent. Overview: retained render + review-video raw equivalent.
    # Include full-rate action/state/contact/manifest records and static copies.
    payload = (2 * frames * (6 * policy + 2 * overview + 1024**2)
               + 2 * basis['provider_bundle_uncompressed_bytes'] + 16 * 1024**2)
    size = payload + payload // 100 + 65536  # ZIP records and directory slack
    if size > MAXIMUM_SLOT_CAPACITY_BYTES:
        raise ValueError('paired_witness_capacity_exceeds_preallocation_ceiling')
    return size


def build_paired_witness_binding(bundle, authority, runtime_environment=None):
    source = Path(bundle['bundle_path'])
    if source.is_symlink() or not source.is_file() or source.stat().st_size != bundle['bundle_size_bytes']:
        raise ValueError('paired_witness_capacity_bundle_invalid')
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    if 'sha256:' + digest.hexdigest() != bundle['bundle_sha256']:
        raise ValueError('paired_witness_capacity_bundle_digest_mismatch')
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        if (len([row for row in infos if row.filename == PLAN_MEMBER]) != 1
                or archive.getinfo(PLAN_MEMBER).file_size > 4_000_000):
            raise ValueError('paired_witness_capacity_plan_invalid')
        plan = json.loads(archive.read(PLAN_MEMBER))
        uncompressed = sum(row.file_size for row in infos)
    cadence = plan['cadence']
    cameras = {row['role']: row for row in plan['cameras']}
    if not {'external', 'wrist', 'overview'} <= set(cameras):
        raise ValueError('paired_witness_capacity_camera_inventory_missing')
    resolution = (runtime_environment or {}).get('BLUEPRINT_ADP009D_CAMERA_RESOLUTION', '')
    if not resolution:
        width, height = 1280, 720  # exact production runtime fallback
    elif resolution == 'policy':
        width, height = 320, 180
    elif re.fullmatch(r'[1-9][0-9]*x[1-9][0-9]*', resolution):
        width, height = (int(v) for v in resolution.split('x'))
    else:
        raise ValueError('paired_witness_capacity_camera_resolution_invalid')
    def dimension(role, axis):
        v = cameras[role]['intrinsics'][axis]
        if type(v) is not int or v <= 0:
            raise ValueError('paired_witness_capacity_camera_dimensions_invalid')
        return v
    basis = {'maximum_action_steps': cadence['maximum_action_steps'],
        'settle_window_samples': cadence['settle_window_samples'],
        'policy_master_width': max(width, dimension('external', 'width'), dimension('wrist', 'width')),
        'policy_master_height': max(height, dimension('external', 'height'), dimension('wrist', 'height')),
        'overview_width': max(1280, dimension('overview', 'width')),
        'overview_height': max(720, dimension('overview', 'height')),
        'provider_bundle_uncompressed_bytes': uncompressed}
    capacity = _archive_capacity(basis)
    return validate_binding({'run_id': authority['run_id'], 'authority_digest': authority['authority_digest'],
        'runtime_inputs_digest': authority['runtime_inputs_digest'],
        'implementation_commit': bundle['implementation_commit'], 'provider_bundle_sha256': bundle['bundle_sha256'],
        'maximum_archive_bytes': capacity, 'capacity_basis': basis})


def stage_paired_witness_slot(*, client, bucket, output_key, binding, job_dir, generated_at, expiration_seconds):
    from .wam_provider_object_store import (
        _presigned_url_expiry_metadata, _s3_absence_confirmed, _write_sensitive_file,
        signed_output_object_binding_sha256,
    )
    metadata = validate_binding(binding)
    key = output_key + SUFFIX
    if _s3_absence_confirmed(client, bucket=bucket, key=key).get('status') != 'passed':
        raise ValueError('paired_witness_fresh_key_absence_unverified')
    params = {'Bucket': bucket, 'Key': key}
    # First bind the object path independently of signed query parameters;
    # then sign the exact session metadata header on the final PUT capability.
    put = client.generate_presigned_url('put_object', Params={**params, 'ContentType': 'application/zip'},
        ExpiresIn=expiration_seconds, HttpMethod='PUT')
    get = client.generate_presigned_url('get_object', Params={**params, 'ResponseCacheControl': 'no-store, max-age=0'},
        ExpiresIn=expiration_seconds, HttpMethod='GET')
    object_binding = signed_output_object_binding_sha256(put, get)
    authority = {'schema_version': SCHEMA, **metadata, 'generated_at': generated_at,
        'expires_at': _presigned_url_expiry_metadata(generated_at, expiration_seconds)['expires_at'],
        'content_type': 'application/zip', 'witness_key_sha256': 'sha256:' + hashlib.sha256(key.encode()).hexdigest(),
        'output_url_object_binding_sha256': object_binding, 'binding_digest': ''}
    authority['binding_digest'] = canonical_digest(authority, digest_field='binding_digest')
    put = client.generate_presigned_url('put_object', Params={**params, 'ContentType': 'application/zip',
        'Metadata': {'blueprint-witness-binding': authority['binding_digest']}},
        ExpiresIn=expiration_seconds, HttpMethod='PUT')
    if signed_output_object_binding_sha256(put, get) != object_binding:
        raise ValueError('paired_witness_signed_object_binding_mismatch')
    directory = Path(job_dir)
    for name, value in [('paired_witness_put_url.txt', put), ('paired_witness_get_url.txt', get),
                        ('paired_witness_authority.json', canonical_json(authority))]:
        path = directory/name
        if path.exists() or path.is_symlink():
            raise ValueError('paired_witness_secret_slot_already_initialized')
        _write_sensitive_file(path, value, label='paired_witness_private_capability')
    return {'schema_version': 'native_task_arena_paired_witness_staging.v1', 'status': 'ready',
        'witness_key': key, 'authority': authority, 'fresh_key_absence_verified': True,
        'raw_signed_urls_recorded': False, 'raw_secret_values_recorded': False}


def paired_witness_secret_paths(job_dir, staging, expected_binding):
    directory = Path(job_dir)
    witness = staging.get('paired_witness') or {}
    if witness.get('status') != 'ready' or witness.get('witness_key') != staging.get('output_key', '') + SUFFIX:
        raise ValueError('paired_witness_staging_not_ready')
    paths = {name: directory/filename for name, filename in SECRET_FILES.items()}
    for path in paths.values():
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
            raise ValueError('paired_witness_secret_file_invalid')
    authority = json.loads(paths['BLUEPRINT_POLICY_CANARY_PAIRED_DELIVERY_AUTHORITY_FILE'].read_text())
    if (authority != witness.get('authority')
            or authority.get('binding_digest') != canonical_digest(authority, digest_field='binding_digest')
            or any(authority.get(k) != v for k, v in validate_binding(expected_binding).items())):
        raise ValueError('paired_witness_authority_mismatch')
    from .wam_provider_object_store import signed_output_object_binding_sha256
    put = paths['BLUEPRINT_POLICY_CANARY_PAIRED_PUT_URL_FILE'].read_text().strip()
    get = paths['BLUEPRINT_POLICY_CANARY_PAIRED_GET_URL_FILE'].read_text().strip()
    try:
        matched = signed_output_object_binding_sha256(put, get) == authority['output_url_object_binding_sha256']
    except ValueError as exc:
        raise ValueError('paired_witness_url_binding_mismatch') from exc
    if not matched:
        raise ValueError('paired_witness_url_binding_mismatch')
    return paths
