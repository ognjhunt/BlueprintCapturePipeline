"""Bounded full archive extraction into a separately fenced private stage.

This parser never publishes target names or reopens owner access. A complete
stage result requires original member hashes and the whole archive digest.
"""
from __future__ import annotations

import hashlib

from . import control_plane_lane_experiment_archive as transport
from . import control_plane_lane_historical_archive as archive
from . import control_plane_lane_historical_generation as generation
from .control_plane_lane_historical_fence import _members
from .control_plane_lane_historical_processes import HistoricalProcessError


def _require(value, code):
    generation._require(value, 'restore_archive_' + code)


class _Stream:
    """Read once per bounded block, even when there are many tiny members."""
    def __init__(self, body, size):
        self.body, self.size = body, size
        self.buffer = b''
        self.count = 0
        self.digest = hashlib.sha256()

    def take(self, amount, sink):
        _require(type(amount) is int and 0 <= amount <= self.size - self.count, 'size_invalid')
        digest = hashlib.sha256()
        while amount:
            if not self.buffer:
                self.buffer = self.body.read(archive.QUANTUM)
                _require(self.buffer, 'truncated')
            value = self.buffer[:amount]
            self.buffer = self.buffer[len(value):]
            amount -= len(value)
            self.count += len(value)
            digest.update(value)
            self.digest.update(value)
            sink(value)
        return 'sha256:' + digest.hexdigest()

    def finish(self):
        _require(self.count == self.size and not self.buffer and not self.body.read(1), 'size_invalid')
        return 'sha256:' + self.digest.hexdigest()


def extract_preserved_members(manifest, raw, pointer, client, bucket, stage, guard, *, origin=None):
    """Exact original format, current guarded SDK, <=1MiB buffering, no replace.

    The supplied stage owns directory/member creation under its retained root
    fence and fresh authority. Only private bytes may be written before this
    function returns. It closes every object body and the owned client on error.
    """
    guarded = None
    try:
        rows = _members(manifest)
        prefix = archive._prefix(manifest, raw)
        payload = sum(row['size_bytes'] for row in rows.values() if row['kind'] == 'file')
        _require(0 <= payload == manifest['logical_payload_bytes'] <= generation.MAX_PAYLOAD_BYTES,
                 'size_invalid')
        size = len(prefix) + payload
        _require(type(pointer) is dict and set(pointer) == archive._FIELDS
            and pointer['sha256'] == pointer['readback_sha256']
            and type(pointer['size_bytes']) is int and pointer['size_bytes'] == pointer['readback_size_bytes'] == size
            and pointer['remote_identity_verified'] is True
            and pointer['full_byte_service_account_readback_passed'] is True, 'pointer_invalid')
        key = archive._key(pointer['sha256'])
        _require(pointer['uri'] == 's3://' + bucket + '/' + key, 'pointer_invalid')
        controller = transport._Controller(guard, size, _origin=origin)
        guarded = transport._LaneArchiveClient(client, bucket, key, controller)
        head = guarded.head_object(Bucket=bucket, Key=key)
        _require(type(head) is dict and head.get('ContentLength') == size
            and type(head.get('Metadata')) is dict and head['Metadata'].get('sha256') == pointer['sha256'][7:]
            and isinstance(head.get('ETag'), str) and 0 < len(head['ETag']) <= 1024, 'identity_changed')
        body = guarded.get_object(Bucket=bucket, Key=key, IfMatch=head['ETag'])['Body']
        stream = _Stream(body, size)
        actual_prefix = bytearray()
        stream.take(len(prefix), actual_prefix.extend)
        _require(bytes(actual_prefix) == prefix, 'manifest_changed')
        del actual_prefix
        for row in sorted(rows.values(), key=lambda row: (row['path'].count('/') + bool(row['path']), row['path'])):
            if row['kind'] == 'directory':
                controller.check()
                stage.directory(row)
                controller.check()
        count = 0
        for row in sorted(rows.values(), key=lambda row: row['path']):
            if row['kind'] != 'file':
                continue
            _require(type(row['size_bytes']) is int and 0 <= row['size_bytes'] <= payload, 'member_invalid')
            controller.check()
            with stage.member(row) as output:
                def write(value):
                    controller.check()
                    _require(output.write(value) == len(value), 'stage_write_failed')
                    controller.check()
                _require(stream.take(row['size_bytes'], write) == row['sha256'], 'member_changed')
            count += 1
        _require(stream.finish() == pointer['sha256'], 'digest_changed')
        body.close()
        _require(guarded.head_object(Bucket=bucket, Key=key) == head, 'identity_changed')
        controller.check()
        return dict(archive_sha256=pointer['sha256'], archive_size_bytes=size,
                    restored_files=count, restored_logical_bytes=payload)
    except (generation.HistoricalGenerationError, HistoricalProcessError):
        raise
    except Exception:
        raise generation.HistoricalGenerationError('historical_generation_restore_archive_extract_failed') from None
    finally:
        if guarded is not None:
            guarded.close()
        else:
            client.close()
