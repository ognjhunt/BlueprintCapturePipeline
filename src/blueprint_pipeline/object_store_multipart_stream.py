"""Bounded multipart sink for content-addressed, digest-checked artifact streams."""

from __future__ import annotations

import hashlib


class MultipartStream:
    """Keep at most one upload part buffered and abort incomplete publications."""

    def __init__(self, *, client, bucket, key, metadata, expected_digest, expected_size):
        self.client, self.bucket, self.key = client, bucket, key
        self.expected_digest, self.expected_size = expected_digest, expected_size
        mib = 1024 * 1024
        self.part_size = max(8 * mib, ((expected_size + 9998) // 9999 + mib - 1) // mib * mib)
        if self.part_size > 64 * mib:
            raise ValueError("artifact_stream_size_exceeds_bounded_multipart_limit")
        self.buffer = bytearray()
        self.digest, self.size, self.parts = hashlib.sha256(), 0, []
        response = client.create_multipart_upload(Bucket=bucket, Key=key, **metadata)
        self.upload_id = response["UploadId"]
        self.completed = False

    def write(self, data):
        if self.size + len(data) > self.expected_size:
            raise ValueError("artifact_stream_exceeds_declared_size")
        self.digest.update(data)
        self.size += len(data)
        view = memoryview(data)
        while view:
            take = min(len(view), self.part_size - len(self.buffer))
            self.buffer.extend(view[:take])
            view = view[take:]
            if len(self.buffer) == self.part_size:
                self._flush()
        return len(data)

    def _flush(self):
        if not self.buffer:
            return
        number = len(self.parts) + 1
        response = self.client.upload_part(
            Bucket=self.bucket,
            Key=self.key,
            UploadId=self.upload_id,
            PartNumber=number,
            Body=bytes(self.buffer),
        )
        self.parts.append({"PartNumber": number, "ETag": response["ETag"]})
        self.buffer.clear()

    def finish(self):
        if (
            self.size != self.expected_size
            or "sha256:" + self.digest.hexdigest() != self.expected_digest
        ):
            raise ValueError("artifact_stream_identity_changed")
        self._flush()
        self.client.complete_multipart_upload(
            Bucket=self.bucket,
            Key=self.key,
            UploadId=self.upload_id,
            MultipartUpload={"Parts": self.parts},
        )
        self.completed = True

    def abort(self):
        if not self.completed:
            self.client.abort_multipart_upload(
                Bucket=self.bucket, Key=self.key, UploadId=self.upload_id
            )
