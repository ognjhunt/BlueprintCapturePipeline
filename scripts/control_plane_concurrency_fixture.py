"""Development-only disk object transport for Plan 13c, never a provider client.

ADP-009D/day 28. Only separately owned fixture storage is writable. Production
publication and readback validators consume this transport unchanged.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, BinaryIO

CHUNK = 1024 * 1024
PART_LIMIT = 8 * CHUNK
OBJECT_LIMIT = 16 * 1024**3
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class _ReadCounter:
    def __init__(self, stream: BinaryIO, store: FilesystemObjectStore, remaining: int | None = None):
        self.stream, self.store, self.remaining = stream, store, remaining

    def read(self, size: int = -1) -> bytes:
        if self.remaining is not None:
            size = self.remaining if size < 0 else min(size, self.remaining)
        result = self.stream.read(size)
        if self.remaining is not None:
            self.remaining -= len(result)
        self.store.read_bytes += len(result)
        return result

    def close(self) -> None:
        self.stream.close()

    def __enter__(self) -> _ReadCounter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class FilesystemObjectStore:
    def __init__(self, root: Path):
        if root.is_symlink() or not root.is_dir():
            raise ValueError("fixture_object_root_unsafe")
        self.root = root.resolve(strict=True)
        self.uploaded_bytes = self.read_bytes = 0

    def _path(self, bucket: str, key: str) -> Path:
        if (not isinstance(bucket, str) or not SAFE.fullmatch(bucket)
                or not isinstance(key, str) or key.startswith("/")):
            raise ValueError("fixture_object_path_unsafe")
        parts = key.split("/")
        if not parts or any(not SAFE.fullmatch(part) or part in {".", ".."} for part in parts):
            raise ValueError("fixture_object_path_unsafe")
        result = self.root / bucket / key
        if any(path.is_symlink() for path in (result, *result.parents)):
            raise ValueError("fixture_object_path_unsafe")
        return result

    def _metadata_path(self, bucket: str, key: str) -> Path:
        self._path(bucket, key)
        path = self.root / ".metadata" / bucket / (key + ".json")
        if any(item.is_symlink() for item in (path, *path.parents)):
            raise ValueError("fixture_object_path_unsafe")
        return path

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        path = self._path(Bucket, Key)
        if not path.is_file():
            raise KeyError(Key)
        state = path.stat()
        etag = '"' + ':'.join(str(value) for value in
            (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns)) + '"'
        metadata_path = self._metadata_path(Bucket, Key)
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        return {"ContentLength": state.st_size, "ETag": etag, "Metadata": metadata}

    def get_object(self, *, Bucket: str, Key: str, Range: str | None = None,
                   IfMatch: str | None = None) -> dict:
        path = self._path(Bucket, Key)
        head = self.head_object(Bucket=Bucket, Key=Key)
        if IfMatch is not None and IfMatch != head["ETag"]:
            raise ValueError("fixture_object_version_changed")
        start, end = 0, head["ContentLength"] - 1
        if Range is not None:
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", Range)
            if not match:
                raise ValueError("fixture_object_range_invalid")
            start, end = map(int, match.groups())
            if not 0 <= start <= end < head["ContentLength"]:
                raise ValueError("fixture_object_range_invalid")
        stream = path.open("rb")
        stream.seek(start)
        length = end - start + 1
        return {"Body": _ReadCounter(stream, self, length), "ContentLength": length,
                "ETag": head["ETag"], "ContentRange": f"bytes {start}-{end}/{head['ContentLength']}",
                "ResponseMetadata": {"HTTPStatusCode": 206 if Range else 200}}

    def put_object(self, *, Bucket: str, Key: str, Body: BinaryIO | bytes,
                   ContentLength: int | None = None, IfNoneMatch: str | None = None) -> dict:
        if isinstance(Body, bytes):
            ContentLength = len(Body) if ContentLength is None else ContentLength
            Body = io.BytesIO(Body)
        if IfNoneMatch not in (None, "*"):
            raise ValueError("fixture_object_create_condition_invalid")
        if type(ContentLength) is not int or not 0 < ContentLength <= OBJECT_LIMIT:
            raise ValueError("fixture_object_size_invalid")
        path = self._path(Bucket, Key)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        digest, written = hashlib.sha256(), 0
        try:
            with os.fdopen(descriptor, "wb") as destination:
                while data := Body.read(min(CHUNK, ContentLength - written + 1)):
                    written += len(data)
                    if written > ContentLength:
                        raise ValueError("fixture_object_size_mismatch")
                    destination.write(data)
                    digest.update(data)
                if written != ContentLength:
                    raise ValueError("fixture_object_size_mismatch")
                destination.flush()
                os.fsync(destination.fileno())
        except BaseException:
            # Only the exclusive file opened by this call is eligible.
            path.unlink()
            raise
        self.uploaded_bytes += written
        return {"ETag": digest.hexdigest()}

    def create_multipart_upload(self, *, Bucket: str, Key: str, Metadata: dict | None = None,
                                ContentType: str | None = None) -> dict:
        self._path(Bucket, Key)
        identifier = uuid.uuid4().hex
        root = self.root / ".multipart" / identifier
        if any(path.is_symlink() for path in (root, *root.parents)):
            raise ValueError("fixture_multipart_binding_invalid")
        root.mkdir(parents=True, mode=0o700)
        (root / "metadata").write_text(json.dumps(Metadata or {}))
        (root / "binding").write_text(Bucket + "\n" + Key + "\n")
        return {"UploadId": identifier}

    def _upload(self, bucket: str, key: str, identifier: str) -> Path:
        self._path(bucket, key)
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("fixture_multipart_binding_invalid")
        root = self.root / ".multipart" / identifier
        if any(path.is_symlink() for path in (root, *root.parents, root / "binding", root / "metadata")) or (root / "binding").read_text() != bucket + "\n" + key + "\n":
            raise ValueError("fixture_multipart_binding_invalid")
        return root

    def upload_part(self, *, Bucket: str, Key: str, UploadId: str, PartNumber: int, Body: bytes) -> dict:
        root = self._upload(Bucket, Key, UploadId)
        if (type(PartNumber) is not int or not 1 <= PartNumber <= 2048
                or not isinstance(Body, bytes) or not 0 < len(Body) <= PART_LIMIT):
            raise ValueError("fixture_multipart_part_invalid")
        with (root / str(PartNumber)).open("xb") as stream:
            stream.write(Body)
        return {"ETag": hashlib.sha256(Body).hexdigest()}

    def complete_multipart_upload(self, *, Bucket: str, Key: str, UploadId: str, MultipartUpload: dict) -> None:
        root = self._upload(Bucket, Key, UploadId)
        rows = MultipartUpload.get("Parts", [])
        if not rows or [row["PartNumber"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("fixture_multipart_part_mismatch")
        total = 0
        for row in rows:
            path = root / str(row["PartNumber"])
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != row["ETag"]:
                raise ValueError("fixture_multipart_part_mismatch")
            total += path.stat().st_size
        if not 0 < total <= OBJECT_LIMIT:
            raise ValueError("fixture_object_size_invalid")
        path = self._path(Bucket, Key)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("xb") as destination:
            for row in rows:
                with (root / str(row["PartNumber"])).open("rb") as source:
                    shutil.copyfileobj(source, destination, CHUNK)
            destination.flush()
            os.fsync(destination.fileno())
        metadata = self._metadata_path(Bucket, Key)
        metadata.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with metadata.open("x") as stream:
            stream.write((root / "metadata").read_text())
        self.uploaded_bytes += total
        shutil.rmtree(root)

    def abort_multipart_upload(self, *, Bucket: str, Key: str, UploadId: str) -> None:
        shutil.rmtree(self._upload(Bucket, Key, UploadId))
