"""Bounded seekable HTTPS transport for immutable provider-output ZIPs.

Extends the existing RawIOBase range-reader pattern used by the single-episode
collector with version pinning, exact range validation, and bounded buffering.
No archive copy is written and signed URLs never enter exceptions or receipts.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import io
import re
import time
import urllib.error
import urllib.request

from .safe_outbound_http import (
    _enforce_redirect_policy, _open_with_policy, presigned_transfer_policy,
    validate_outbound_url,
)


class ProviderOutputTransportError(ValueError):
    """Secret-free, stable transport failure code."""


class ProviderOutputRangeReader(io.RawIOBase):
    def __init__(self, url, *, maximum_archive_bytes, deadline_seconds=3600,
                 block_bytes=8 * 1024**2, maximum_read_bytes=64 * 1024**2,
                 opener=None):
        super().__init__()
        self._url = url
        self._policy = presigned_transfer_policy(url, max_response_bytes=maximum_read_bytes)
        self._requested = validate_outbound_url(url, policy=self._policy)
        self._opener = opener or _open_with_policy
        self._deadline = time.monotonic() + deadline_seconds
        self._block_bytes, self._maximum_read = block_bytes, maximum_read_bytes
        self._position, self.transferred_bytes, self.request_count = 0, 0, 0
        self._cache = OrderedDict()
        self.identity = None
        if (type(maximum_archive_bytes) is not int or maximum_archive_bytes <= 0
                or type(block_bytes) is not int or not 1 <= block_bytes <= maximum_read_bytes
                or not 0 < deadline_seconds <= 86400):
            raise ProviderOutputTransportError('provider_output_transport_limits_invalid')
        with self._response({'Range': 'bytes=0-0'}, expected_status=206) as response:
            match = re.fullmatch(r'bytes 0-0/(\d+)', str(response.headers.get('Content-Range', '')))
            if not match or not 0 < int(match[1]) <= maximum_archive_bytes:
                raise ProviderOutputTransportError('provider_output_archive_size_invalid')
            self._size = int(match[1])
            etag = str(response.headers.get('ETag', ''))
            generation = str(response.headers.get('x-goog-generation', ''))
            etag = etag if re.fullmatch(r'"[A-Za-z0-9._:+/=-]{1,256}"', etag) else None
            generation = generation if re.fullmatch(r'\d{1,32}', generation) else None
            if not etag and not generation:
                raise ProviderOutputTransportError('provider_output_immutable_version_missing')
            self.identity = {'size_bytes': self._size, 'etag': etag, 'generation': generation}
            self._exact_body(response, 1)

    @contextmanager
    def _response(self, headers, *, expected_status):
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderOutputTransportError('provider_output_transfer_deadline_exceeded')
        headers = {'Accept-Encoding': 'identity', 'User-Agent': 'BlueprintProviderOutput/1.0', **headers}
        if self.identity:
            if self.identity['etag']:
                headers['If-Match'] = self.identity['etag']
            if self.identity['generation']:
                headers['x-goog-if-generation-match'] = self.identity['generation']
        request = urllib.request.Request(self._url, method='GET', headers=headers)
        try:
            with self._opener(request, min(60., remaining), self._policy) as response:
                self.request_count += 1
                _enforce_redirect_policy(self._requested, response.geturl(), policy=self._policy)
                if int(response.status) != expected_status:
                    raise ProviderOutputTransportError('provider_output_range_unsupported' if expected_status == 206
                                                       else 'provider_output_stream_status_invalid')
                if response.headers.get('Content-Encoding', 'identity') != 'identity':
                    raise ProviderOutputTransportError('provider_output_content_encoding_invalid')
                if self.identity:
                    if ((self.identity['etag'] and response.headers.get('ETag') != self.identity['etag'])
                            or (self.identity['generation'] and response.headers.get('x-goog-generation') != self.identity['generation'])):
                        raise ProviderOutputTransportError('provider_output_remote_version_changed')
                yield response
        except ProviderOutputTransportError:
            raise
        except urllib.error.HTTPError as exc:
            code = 'provider_output_not_ready' if exc.code == 404 else 'provider_output_remote_version_changed' if exc.code == 412 else 'provider_output_http_failed'
            raise ProviderOutputTransportError(code) from None
        except Exception:
            raise ProviderOutputTransportError('provider_output_transport_failed') from None

    def _exact_body(self, response, expected):
        if response.headers.get('Content-Length') != str(expected):
            raise ProviderOutputTransportError('provider_output_content_length_invalid')
        payload = bytearray()
        while len(payload) < expected:
            if time.monotonic() >= self._deadline:
                raise ProviderOutputTransportError('provider_output_transfer_deadline_exceeded')
            chunk = response.read(min(1024**2, expected - len(payload)))
            if not chunk:
                raise ProviderOutputTransportError('provider_output_range_truncated')
            payload.extend(chunk)
            self.transferred_bytes += len(chunk)
        if response.read(1):
            raise ProviderOutputTransportError('provider_output_range_overlong')
        return bytes(payload)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
            raise ProviderOutputTransportError('provider_output_seek_invalid')
        position = offset + (0 if whence == io.SEEK_SET else self._position if whence == io.SEEK_CUR else self._size)
        if position < 0:
            raise ProviderOutputTransportError('provider_output_seek_invalid')
        self._position = position
        return position

    def _block(self, start):
        if start in self._cache:
            self._cache.move_to_end(start)
            return self._cache[start]
        end = min(self._size, start + self._block_bytes) - 1
        with self._response({'Range': f'bytes={start}-{end}'}, expected_status=206) as response:
            if response.headers.get('Content-Range') != f'bytes {start}-{end}/{self._size}':
                raise ProviderOutputTransportError('provider_output_content_range_invalid')
            payload = self._exact_body(response, end - start + 1)
        self._cache[start] = payload
        while len(self._cache) > 2:
            self._cache.popitem(last=False)
        return payload

    def read(self, size=-1):
        available = max(0, self._size - self._position)
        size = available if size is None or size < 0 else min(size, available)
        if size > self._maximum_read:
            raise ProviderOutputTransportError('provider_output_read_memory_cap_exceeded')
        chunks = []
        while size:
            start = self._position // self._block_bytes * self._block_bytes
            block = self._block(start)
            offset = self._position - start
            chunk = block[offset:offset + size]
            chunks.append(chunk)
            self._position += len(chunk)
            size -= len(chunk)
        return b''.join(chunks)

    def archive_sha256(self):
        """Hash the complete pinned object once without storing the ZIP."""
        digest, received = hashlib.sha256(), 0
        with self._response({}, expected_status=200) as response:
            if response.headers.get('Content-Length') != str(self._size):
                raise ProviderOutputTransportError('provider_output_content_length_invalid')
            while received < self._size:
                if time.monotonic() >= self._deadline:
                    raise ProviderOutputTransportError('provider_output_transfer_deadline_exceeded')
                chunk = response.read(min(self._block_bytes, self._size - received))
                if not chunk:
                    raise ProviderOutputTransportError('provider_output_archive_truncated')
                digest.update(chunk)
                received += len(chunk)
                self.transferred_bytes += len(chunk)
            if response.read(1):
                raise ProviderOutputTransportError('provider_output_archive_overlong')
        return 'sha256:' + digest.hexdigest()
