"""Stream original historical members using bounded multipart/full readback.

Only transport is reused from registered experiments; this format contains the
authentic historical manifest, never a fabricated registered birth or entry.
The worker supplies its retained fence and fresh original-operation authority.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

from . import control_plane_lane_experiment_archive as transport
from . import control_plane_lane_historical_generation as generation
from . import task_evaluation_configured_scene_object_store as remote

MAGIC = b'BLUEPRINT-HISTORICAL-ARCHIVE\x00\x01'
QUANTUM = 1024**2
_FIELDS = {'uri', 'sha256', 'size_bytes', 'remote_identity_verified',
    'full_byte_service_account_readback_passed', 'readback_sha256', 'readback_size_bytes'}


def _require(value, code):
    generation._require(value, 'archive_' + code)


def _prefix(manifest, raw):
    _require(isinstance(raw, bytes) and 0 < len(raw) <= generation.MAX_MANIFEST_BYTES, 'manifest_invalid')
    try:
        _require(json.loads(raw) == manifest, 'manifest_invalid')
    except (ValueError, UnicodeError):
        raise generation.HistoricalGenerationError('historical_generation_archive_manifest_invalid') from None
    _require(0 < len(manifest['members']) == manifest['member_count'] <= generation.MAX_MEMBERS,
             'manifest_invalid')
    return MAGIC + len(raw).to_bytes(4, 'big') + raw


def _write_original(held, manifest, raw, sink, guard):
    guard()
    sink.write(_prefix(manifest, raw))
    for row in sorted(manifest['members'], key=lambda row: row['path']):
        if row['kind'] != 'file':
            continue
        guard()
        with held._opened(row['path']) as (fd, verify):
            os.lseek(fd, 0, os.SEEK_SET)
            consumed, digest = 0, hashlib.sha256()
            while True:
                guard()
                verify()
                block = os.read(fd, min(QUANTUM, row['size_bytes'] + 1 - consumed))
                guard()
                if not block:
                    break
                consumed += len(block)
                _require(consumed <= row['size_bytes'], 'payload_changed')
                digest.update(block)
                sink.write(block)
            _require(consumed == row['size_bytes'] and 'sha256:' + digest.hexdigest() == row['sha256'],
                     'payload_changed')
            verify()


def _key(digest):
    _require(isinstance(digest, str) and re.fullmatch(r'sha256:[0-9a-f]{64}', digest), 'receipt_invalid')
    return remote.LARGE_ARTIFACT_KEY_PREFIX + '/historical-generation/sha256/' + digest[7:] + '/archive.bin'


def _pointer(result, digest, size):
    _require(result['remote_identity_verified'] is True and result['full_byte_service_account_readback_passed'] is True
        and result['digest'] == result['readback_digest'] == digest
        and result['size_bytes'] == result['readback_size_bytes'] == size, 'readback_invalid')
    return dict(uri=result['uri'], sha256=digest, size_bytes=size, remote_identity_verified=True,
                full_byte_service_account_readback_passed=True, readback_sha256=digest, readback_size_bytes=size)


def preserve(held, manifest, raw, client, bucket, guard, *, origin=None):
    guarded = None
    try:
        prefix = _prefix(manifest, raw)
        payload = sum(row['size_bytes'] for row in manifest['members'] if row['kind'] == 'file')
        _require(payload == manifest['logical_payload_bytes'] and 0 <= payload <= generation.MAX_PAYLOAD_BYTES,
                 'limit')
        size = len(prefix) + payload
        controller = transport._Controller(guard, size, _origin=origin)
        class Digest:
            def __init__(self):
                self.digest = hashlib.sha256()
            def write(self, value):
                self.digest.update(value)
        sink = Digest()
        _write_original(held, manifest, raw, sink, lambda: controller.check('source'))
        digest = 'sha256:' + sink.digest.hexdigest()
        guarded = transport._LaneArchiveClient(client, bucket, _key(digest), controller)
        passes = 0
        def write(output):
            nonlocal passes
            passes += 1
            _require(passes == 1, 'source_reissued')
            _write_original(held, manifest, raw, output, lambda: controller.check('source'))
        result = remote.publish_configured_scene_stream(write_stream=write, digest=digest, size_bytes=size,
            filename='archive.bin', artifact_kind='historical-generation', client=guarded, bucket=bucket)
        controller.check()
        return _pointer(result, digest, size)
    except remote.TaskEvaluationConfiguredSceneObjectStoreError:
        raise generation.HistoricalGenerationError('historical_generation_archive_preservation_failed') from None
    finally:
        if guarded is not None:
            guarded.close()
        else:
            client.close()


def verify_preservation(pointer, client, bucket, guard, *, origin=None):
    guarded = None
    try:
        _require(type(pointer) is dict and set(pointer) == _FIELDS
            and pointer['sha256'] == pointer['readback_sha256']
            and pointer['size_bytes'] == pointer['readback_size_bytes']
            and type(pointer['size_bytes']) is int and 0 < pointer['size_bytes']
                <= generation.MAX_PAYLOAD_BYTES + generation.MAX_MANIFEST_BYTES + len(MAGIC) + 4
            and pointer['remote_identity_verified'] is True
            and pointer['full_byte_service_account_readback_passed'] is True, 'receipt_invalid')
        key = _key(pointer['sha256'])
        _require(pointer['uri'] == 's3://' + bucket + '/' + key, 'receipt_invalid')
        controller = transport._Controller(guard, pointer['size_bytes'], _origin=origin)
        guarded = transport._LaneArchiveClient(client, bucket, key, controller)
        def absent(_sink):
            raise remote.TaskEvaluationConfiguredSceneObjectStoreError('historical_archive_missing_on_resume')
        result = remote.publish_configured_scene_stream(write_stream=absent, digest=pointer['sha256'],
            size_bytes=pointer['size_bytes'], filename='archive.bin', artifact_kind='historical-generation',
            client=guarded, bucket=bucket)
        controller.check()
        _require(_pointer(result, pointer['sha256'], pointer['size_bytes']) == pointer, 'receipt_invalid')
    except remote.TaskEvaluationConfiguredSceneObjectStoreError:
        raise generation.HistoricalGenerationError('historical_generation_archive_preservation_failed') from None
    finally:
        if guarded is not None:
            guarded.close()
        else:
            client.close()
