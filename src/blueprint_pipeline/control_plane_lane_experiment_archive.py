"""Finite streaming preservation for one authenticated, exclusively held target.

Native multipart publication/readback is reused. No archive pathname, provider
request during tests, detached worker, or unverified pointer permits removal.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import threading
import time

from . import control_plane_lane_experiment_actions as actions
from . import control_plane_lane_owner_consents as owners
from . import task_evaluation_configured_scene_object_store as remote
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _require

MAGIC = b'BLUEPRINT-LANE-ARCHIVE\x00\x01'
QUANTUM = 1024 * 1024
MAX_PAYLOAD = 128 * 1024**3


def _client(files, config):
    """Fixed protected environment selects acquired secret files, never body data."""
    from urllib.parse import urlsplit
    raw, _ = files.read(config.experiment_gc_environment_file, cap=65536, protected=True)
    permitted = set(remote._ARTIFACT_STORE_FILE_ENV.values()) | set(remote._LEGACY_OBJECT_STORE_FILE_ENV.values())
    values = {}
    for line in raw.decode('utf-8').splitlines():
        name, separator, value = line.strip().partition('=')
        if name not in permitted:
            continue
        _require(separator and name not in values and len(value.encode()) <= 4096,
                 'experiment_archive_configuration_invalid')
        value = value.strip()
        if len(value) > 1 and value[0] in ("'", '"') and value[-1] == value[0]:
            value = value[1:-1]
        _require(value and not any(char in value for char in ('$', '\\', "'", '"')),
                 'experiment_archive_configuration_invalid')
        values[name] = value
    dedicated = any(name in values for name in remote._ARTIFACT_STORE_FILE_ENV.values())
    selected = remote._ARTIFACT_STORE_FILE_ENV if dedicated else remote._LEGACY_OBJECT_STORE_FILE_ENV
    secrets = {}
    for role, name in selected.items():
        required = dedicated or role in ('access_key', 'secret_key', 'bucket')
        if name not in values:
            _require(not required, 'experiment_archive_configuration_missing')
            secrets[role] = ''
            continue
        payload, record = files.read(values[name], cap=4096, protected=True)
        _require(stat.S_IMODE(record.info.st_mode) in (0o400, 0o440, 0o600, 0o640),
                 'experiment_archive_secret_unsafe')
        try:
            secrets[role] = payload.decode('utf-8').strip()
        except UnicodeError:
            raise OwnerTargetVersionError('experiment_archive_configuration_invalid') from None
        _require(not required or secrets[role], 'experiment_archive_configuration_missing')
    _require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]{0,62}', secrets['bucket']),
             'experiment_archive_configuration_invalid')
    if secrets['endpoint']:
        endpoint = urlsplit(secrets['endpoint'])
        _require(endpoint.scheme == 'https' and endpoint.hostname and not endpoint.username
                 and not endpoint.password and endpoint.path in ('', '/') and not endpoint.query
                 and not endpoint.fragment, 'experiment_archive_configuration_invalid')
    files.verify()
    try:
        import boto3
        from botocore.config import Config
        kwargs = dict(aws_access_key_id=secrets['access_key'], aws_secret_access_key=secrets['secret_key'],
            region_name=secrets['region'] or 'us-east-1', config=Config(signature_version='s3v4',
                connect_timeout=45, read_timeout=45, retries={'total_max_attempts': 1}))
        if secrets['endpoint']:
            kwargs['endpoint_url'] = secrets['endpoint']
        client = boto3.client('s3', **kwargs)
    except Exception:
        raise OwnerTargetVersionError('experiment_archive_client_unavailable') from None
    return client, secrets['bucket']


class _Controller:
    def __init__(self, guard, size):
        _require(type(size) is int and 0 < size <= MAX_PAYLOAD, 'experiment_archive_limit')
        self.guard, self.size = guard, size
        self.origin = time.monotonic()
        self.lock, self.failure = threading.RLock(), None
        self.calls = dict(source=0, readback=0, sdk=0)
        self.maximum = dict(source=24 * (math.ceil(size / QUANTUM) + 4096),
                            readback=24 * (math.ceil(size / QUANTUM) + 16384), sdk=4 * (9999 + 49152) + 16)
    def check(self, role=None):
        with self.lock:
            if self.failure is not None:
                raise remote.TaskEvaluationConfiguredSceneObjectStoreError(self.failure)
            try:
                _require(time.monotonic() - self.origin <= 4 * 3600, 'experiment_archive_deadline')
                if role is not None:
                    _require(role in self.calls and self.calls[role] < self.maximum[role], 'experiment_archive_work_limit')
                    self.calls[role] += 1
                self.guard()
            except Exception:
                self.failure = 'experiment_archive_authority_refused'
                raise remote.TaskEvaluationConfiguredSceneObjectStoreError(self.failure) from None


class _Body:
    def __init__(self, body, controller, size, done):
        self.body, self.controller, self.size = body, controller, size
        self.position, self.window, self.calls, self.closed = 0, -1, 0, False
        self.done = done
    def read(self, amount):
        self.controller.check('readback')
        _require(not self.closed and type(amount) is int and amount > 0, 'experiment_archive_body_invalid')
        window = self.position // QUANTUM
        if window != self.window:
            self.window, self.calls = window, 0
        if self.calls == 8:
            self.controller.failure = 'experiment_archive_fragment_limit'
            raise remote.TaskEvaluationConfiguredSceneObjectStoreError(self.controller.failure)
        self.calls += 1
        amount = min(amount, QUANTUM - self.position % QUANTUM, self.size + 1 - self.position)
        _require(amount > 0, 'experiment_archive_body_invalid')
        payload = self.body.read(amount)
        _require(isinstance(payload, bytes) and len(payload) <= amount, 'experiment_archive_body_invalid')
        self.position += len(payload)
        self.controller.check('readback')
        return payload
    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.body.close()
            finally:
                self.done(self)


class _LaneArchiveClient:
    def __init__(self, client, bucket, key, controller):
        self.client, self.bucket, self.key, self.controller = client, bucket, key, controller
        self.upload_id, self.parts, self.aborted, self.bodies = None, 0, False, set()
        self.completed = False
    def _call(self, method, args):
        _require(args.get('Bucket') == self.bucket and args.get('Key') == self.key, 'experiment_archive_target_changed')
        self.controller.check('sdk')
        value = getattr(self.client, method)(**args)
        if method == 'create_multipart_upload' and isinstance(value, dict):
            token = value.get('UploadId')
            if isinstance(token, str) and 0 < len(token) <= 1024:
                self.upload_id = token
        elif method == 'complete_multipart_upload':
            self.completed = True
        try:
            self.controller.check('sdk')
        except BaseException:
            if method == 'get_object' and isinstance(value, dict) and 'Body' in value:
                value['Body'].close()
            raise
        return value
    def head_object(self, **args):
        return self._call('head_object', args)
    def create_multipart_upload(self, **args):
        _require(self.upload_id is None, 'experiment_archive_upload_reissued')
        response = self._call('create_multipart_upload', args)
        token = response.get('UploadId')
        _require(isinstance(token, str) and 0 < len(token) <= 1024, 'experiment_archive_upload_invalid')
        self.upload_id = token
        return response
    def upload_part(self, **args):
        _require(args.get('UploadId') == self.upload_id and self.upload_id is not None
                 and type(args.get('PartNumber')) is int and args['PartNumber'] == self.parts + 1 <= 9999
                 and isinstance(args.get('Body'), bytes) and 0 < len(args['Body']) <= 16 * QUANTUM,
                 'experiment_archive_upload_invalid')
        response = self._call('upload_part', args)
        self.parts += 1
        return response
    def complete_multipart_upload(self, **args):
        _require(args.get('UploadId') == self.upload_id and self.upload_id is not None
                 and len(args.get('MultipartUpload', {}).get('Parts', [])) == self.parts,
                 'experiment_archive_upload_invalid')
        return self._call('complete_multipart_upload', args)
    def abort_multipart_upload(self, **args):
        _require(not self.aborted and self.upload_id is not None and args.get('UploadId') == self.upload_id
                 and args.get('Bucket') == self.bucket and args.get('Key') == self.key,
                 'experiment_archive_cleanup_invalid')
        self.aborted = True
        return self.client.abort_multipart_upload(**args)
    def get_object(self, **args):
        size = self.controller.size
        if 'Range' in args:
            selected = re.fullmatch(r'bytes=([0-9]+)-([0-9]+)', args['Range'])
            _require(selected and args.get('IfMatch'), 'experiment_archive_range_invalid')
            start, end = map(int, selected.groups())
            _require(0 <= start <= end < size and end - start < 8 * QUANTUM, 'experiment_archive_range_invalid')
            size = end - start + 1
        response = self._call('get_object', args)
        with self.controller.lock:
            _require(len(self.bodies) < 4, 'experiment_archive_body_limit')
            body = _Body(response['Body'], self.controller, size, self._done)
            self.bodies.add(body)
        return response | {'Body': body}
    def _done(self, body):
        with self.controller.lock:
            self.bodies.discard(body)
    def close(self):
        failure = None
        for body in tuple(self.bodies):
            try:
                body.close()
            except Exception:
                failure = 'experiment_archive_body_close_failed'
        if self.upload_id is not None and not self.completed and not self.aborted:
            try:
                self.abort_multipart_upload(Bucket=self.bucket, Key=self.key, UploadId=self.upload_id)
            except Exception:
                failure = 'experiment_archive_abort_unresolved'
        try:
            self.client.close()
        except Exception:
            failure = 'experiment_archive_client_close_failed'
        if failure:
            raise OwnerTargetVersionError(failure)


def preserve(files, config, target, rows, manifest_raw, guard):
    _require((remote._RANGE_READBACK_THRESHOLD_BYTES, remote._RANGE_READBACK_CHUNK_BYTES,
              remote._RANGE_READBACK_CONCURRENCY) == (32 * QUANTUM, 8 * QUANTUM, 4),
             'experiment_archive_native_policy_changed')
    _require(isinstance(manifest_raw, bytes) and len(manifest_raw) <= QUANTUM,
             'experiment_archive_manifest_invalid')
    prefix = MAGIC + len(manifest_raw).to_bytes(4, 'big') + manifest_raw
    size = len(prefix) + sum(int(row[3].split(':')[4]) for row in rows if row[1] == 'file')
    controller = _Controller(guard, size)
    def write(sink):
        controller.check('source')
        sink.write(prefix)
        for row in sorted(rows, key=lambda row: row[0]):
            if row[1] != 'file':
                continue
            controller.check('source')
            _, _, fd, info = actions._member(files, target, row)
            try:
                files.location(fd)
                os.lseek(fd, 0, os.SEEK_SET)
                count, window, fragments = 0, -1, 0
                while True:
                    controller.check('source')
                    if count // QUANTUM != window:
                        window, fragments = count // QUANTUM, 0
                    _require(fragments < 8, 'experiment_archive_fragment_limit')
                    fragments += 1
                    files.location(fd)
                    payload = os.read(fd, min(QUANTUM - count % QUANTUM, info.st_size + 1 - count))
                    controller.check('source')
                    if not payload:
                        break
                    count += len(payload)
                    _require(count <= info.st_size, 'experiment_archive_member_changed')
                    sink.write(payload)
                    controller.check('source')
                _require(count == info.st_size and owners._metadata(os.fstat(fd)) == owners._metadata(info),
                         'experiment_archive_member_changed')
            finally:
                files.close(fd)
    class Digest:
        def __init__(self):
            self.digest = hashlib.sha256()
        def write(self, payload):
            self.digest.update(payload)
    sink = Digest()
    write(sink)
    digest = 'sha256:' + sink.digest.hexdigest()
    key = remote.LARGE_ARTIFACT_KEY_PREFIX + '/lane-experiment/sha256/' + digest[7:] + '/archive.bin'
    client, bucket = _client(files, config)
    guarded = _LaneArchiveClient(client, bucket, key, controller)
    try:
        result = remote.publish_configured_scene_stream(write_stream=write, digest=digest, size_bytes=size,
            filename='archive.bin', artifact_kind='lane-experiment', client=guarded, bucket=bucket)
        controller.check()
        _require(result['remote_identity_verified'] is True and result['full_byte_service_account_readback_passed'] is True
                 and result['digest'] == result['readback_digest'] == digest
                 and result['size_bytes'] == result['readback_size_bytes'] == size,
                 'experiment_archive_readback_invalid')
        return dict(uri=result['uri'], sha256=digest, size_bytes=size, remote_identity_verified=True,
                    full_byte_service_account_readback_passed=True, readback_sha256=digest, readback_size_bytes=size)
    except remote.TaskEvaluationConfiguredSceneObjectStoreError:
        raise OwnerTargetVersionError('experiment_archive_preservation_failed') from None
    finally:
        guarded.close()


def verify_preservation(files, config, preservation, guard):
    """An old ready pointer must still prove complete remote bytes on recovery."""
    _require(isinstance(preservation, dict) and set(preservation) == {'uri', 'sha256', 'size_bytes',
        'remote_identity_verified', 'full_byte_service_account_readback_passed', 'readback_sha256', 'readback_size_bytes'}
        and preservation['remote_identity_verified'] is True
        and preservation['full_byte_service_account_readback_passed'] is True
        and preservation['sha256'] == preservation['readback_sha256']
        and preservation['size_bytes'] == preservation['readback_size_bytes'], 'experiment_archive_receipt_invalid')
    controller = _Controller(guard, preservation['size_bytes'])
    client, bucket = _client(files, config)
    key = remote.LARGE_ARTIFACT_KEY_PREFIX + '/lane-experiment/sha256/' + preservation['sha256'][7:] + '/archive.bin'
    guarded = _LaneArchiveClient(client, bucket, key, controller)
    try:
        _require(preservation['uri'] == 's3://' + bucket + '/' + key, 'experiment_archive_target_changed')
        def absent(_sink):
            raise remote.TaskEvaluationConfiguredSceneObjectStoreError('experiment_archive_missing_on_resume')
        result = remote.publish_configured_scene_stream(write_stream=absent, digest=preservation['sha256'],
            size_bytes=preservation['size_bytes'], filename='archive.bin', artifact_kind='lane-experiment',
            client=guarded, bucket=bucket)
        _require(result['digest'] == result['readback_digest'] == preservation['sha256']
                 and result['size_bytes'] == result['readback_size_bytes'] == preservation['size_bytes'],
                 'experiment_archive_readback_invalid')
    except remote.TaskEvaluationConfiguredSceneObjectStoreError:
        raise OwnerTargetVersionError('experiment_archive_preservation_failed') from None
    finally:
        guarded.close()
