"""Hermetic fakes for plan 14's remote CPU workers: B2 (S3 API), the GCS transport bucket, Cloud Run jobs.

They model what the remote CPU contract depends on and what the real services only show under
failure: presigned URLs that expire and name exactly one object and method; B2 deletes that only
hide an object until every version is removed; server-side copies guarded by an ETag; generation-
pinned transport reads with no listing; and Cloud Run job etags with paginated execution listings
whose outcomes are scripted.  Everything is driven by a shared ``FakeClock``; nothing touches a
network.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import itertools
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

from botocore.exceptions import ClientError

MAX_PRESIGN_SECONDS = 7 * 24 * 3600
BEHAVIOURS = ("succeed", "crash", "hang", "timeout", "lost_response", "duplicate")


class FakeClock:
    """A settable wall clock in epoch seconds, shared by every fake in one test."""

    def __init__(self, now: float = 2_000_000_000.0) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _client_error(status: int, code: str, operation: str, message: str | None = None) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": message or code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        operation,
    )


def _etag(data: bytes) -> str:
    return '"' + hashlib.md5(data).hexdigest() + '"'


@dataclass
class _Version:
    version_id: str
    data: bytes | None  # None is a delete marker: B2's "hide"
    etag: str
    metadata: dict[str, str]
    last_modified: float


@dataclass
class FakeHttpResponse:
    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)


class _Body(io.BytesIO):
    """A streaming body like botocore's: ``read`` and ``close``."""


class FakeArtifactStore:
    """A versioned S3-compatible bucket with B2 semantics, presigned URLs and byte accounting.

    ``bytes_received_from_client`` and ``bytes_sent_to_client`` count only data that crossed the
    client connection, so a server-side ``copy_object`` or ``upload_part_copy`` moves no host bytes.
    """

    endpoint = "https://s3.us-west-004.backblazeb2.test"

    def __init__(self, *, clock: FakeClock | None = None, bucket: str = "b2-bucket",
                 min_part_bytes: int = 5 * 1024**2, max_copy_bytes: int = 5 * 1024**3) -> None:
        self.clock = clock or FakeClock()
        self.buckets: dict[str, dict[str, list[_Version]]] = {bucket: {}}
        self.min_part_bytes, self.max_copy_bytes = min_part_bytes, max_copy_bytes
        self.uploads: dict[str, dict[str, Any]] = {}
        self.operations: list[tuple[str, str, str]] = []
        self.bytes_received_from_client = 0
        self.bytes_sent_to_client = 0
        self._secret = b"fake-b2-application-key"
        self._ids = itertools.count(1)

    def _objects(self, bucket: str, operation: str) -> dict[str, list[_Version]]:
        if bucket not in self.buckets:
            raise _client_error(404, "NoSuchBucket", operation)
        return self.buckets[bucket]

    def _version(self, bucket: str, key: str, operation: str, version_id: str | None = None) -> _Version:
        versions = self._objects(bucket, operation).get(key, [])
        chosen = next((item for item in versions if item.version_id == version_id), None) if version_id else (
            versions[-1] if versions else None)
        if chosen is None or chosen.data is None:
            raise _client_error(404, "NoSuchKey" if version_id is None else "NoSuchVersion", operation)
        return chosen

    def _store(self, bucket: str, key: str, data: bytes | None, metadata: Mapping[str, str] | None,
               operation: str, etag: str | None = None) -> _Version:
        version = _Version(f"4_z{next(self._ids):012d}", data, etag or ("" if data is None else _etag(data)),
                           dict(metadata or {}), self.clock.now)
        self._objects(bucket, operation).setdefault(key, []).append(version)
        self.operations.append((operation, bucket, key))
        return version

    @staticmethod
    def _source(copy_source: Any) -> tuple[str, str, str | None]:
        if isinstance(copy_source, str):
            bucket, _, key = copy_source.lstrip("/").partition("/")
            return bucket, unquote(key), None
        return copy_source["Bucket"], copy_source["Key"], copy_source.get("VersionId")

    @staticmethod
    def _range(spec: str, size: int, operation: str) -> tuple[int, int]:
        try:
            start_text, _, end_text = spec.removeprefix("bytes=").partition("-")
            start, end = int(start_text), min(int(end_text), size - 1)
        except ValueError:
            raise _client_error(416, "InvalidRange", operation) from None
        if not spec.startswith("bytes=") or start < 0 or start > end:
            raise _client_error(416, "InvalidRange", operation)
        return start, end

    def head_object(self, *, Bucket: str, Key: str, VersionId: str | None = None,
                    IfMatch: str | None = None) -> dict[str, Any]:
        version = self._version(Bucket, Key, "HeadObject", VersionId)
        if IfMatch is not None and IfMatch != version.etag:
            raise _client_error(412, "PreconditionFailed", "HeadObject")
        return {"ContentLength": len(version.data), "ETag": version.etag, "Metadata": dict(version.metadata),
                "VersionId": version.version_id, "LastModified": datetime.fromtimestamp(version.last_modified,
                                                                                       timezone.utc),
                "ResponseMetadata": {"HTTPStatusCode": 200}}

    def get_object(self, *, Bucket: str, Key: str, VersionId: str | None = None, Range: str | None = None,
                   IfMatch: str | None = None) -> dict[str, Any]:
        version = self._version(Bucket, Key, "GetObject", VersionId)
        if IfMatch is not None and IfMatch != version.etag:
            raise _client_error(412, "PreconditionFailed", "GetObject")
        data, extra = version.data, {}
        if Range is not None:
            start, end = self._range(Range, len(data), "GetObject")
            extra = {"ContentRange": f"bytes {start}-{end}/{len(data)}"}
            data = data[start:end + 1]
        self.bytes_sent_to_client += len(data)
        return {"Body": _Body(data), "ContentLength": len(data), "ETag": version.etag,
                "Metadata": dict(version.metadata), "VersionId": version.version_id, **extra,
                "ResponseMetadata": {"HTTPStatusCode": 206 if Range is not None else 200}}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, Metadata: Mapping[str, str] | None = None,
                   ContentLength: int | None = None, **_: Any) -> dict[str, Any]:
        data = Body.read() if hasattr(Body, "read") else bytes(Body)
        if ContentLength is not None and ContentLength != len(data):
            raise _client_error(400, "IncompleteBody", "PutObject")
        self.bytes_received_from_client += len(data)
        version = self._store(Bucket, Key, data, Metadata, "PutObject")
        return {"ETag": version.etag, "VersionId": version.version_id}

    def copy_object(self, *, Bucket: str, Key: str, CopySource: Any, MetadataDirective: str = "COPY",
                    Metadata: Mapping[str, str] | None = None, CopySourceIfMatch: str | None = None,
                    **_: Any) -> dict[str, Any]:
        source_bucket, source_key, source_version = self._source(CopySource)
        source = self._version(source_bucket, source_key, "CopyObject", source_version)
        if CopySourceIfMatch is not None and CopySourceIfMatch != source.etag:
            raise _client_error(412, "PreconditionFailed", "CopyObject")
        if len(source.data) > self.max_copy_bytes:
            raise _client_error(400, "InvalidRequest", "CopyObject", "copy source larger than the maximum")
        metadata = Metadata if MetadataDirective == "REPLACE" else source.metadata
        version = self._store(Bucket, Key, source.data, metadata, "CopyObject")
        return {"CopyObjectResult": {"ETag": version.etag}, "VersionId": version.version_id}

    def create_multipart_upload(self, *, Bucket: str, Key: str, Metadata: Mapping[str, str] | None = None,
                                **_: Any) -> dict[str, Any]:
        self._objects(Bucket, "CreateMultipartUpload")
        upload_id = f"upload-{next(self._ids):08d}"
        self.uploads[upload_id] = {"bucket": Bucket, "key": Key, "metadata": dict(Metadata or {}), "parts": {}}
        return {"UploadId": upload_id}

    def _upload(self, bucket: str, key: str, upload_id: str, operation: str) -> dict[str, Any]:
        upload = self.uploads.get(upload_id)
        if upload is None or (upload["bucket"], upload["key"]) != (bucket, key):
            raise _client_error(404, "NoSuchUpload", operation)
        return upload

    def upload_part(self, *, Bucket: str, Key: str, UploadId: str, PartNumber: int, Body: bytes,
                    **_: Any) -> dict[str, Any]:
        data = Body.read() if hasattr(Body, "read") else bytes(Body)
        self._upload(Bucket, Key, UploadId, "UploadPart")["parts"][PartNumber] = data
        self.bytes_received_from_client += len(data)
        return {"ETag": _etag(data)}

    def upload_part_copy(self, *, Bucket: str, Key: str, UploadId: str, PartNumber: int, CopySource: Any,
                         CopySourceRange: str | None = None, CopySourceIfMatch: str | None = None,
                         **_: Any) -> dict[str, Any]:
        upload = self._upload(Bucket, Key, UploadId, "UploadPartCopy")
        source_bucket, source_key, source_version = self._source(CopySource)
        source = self._version(source_bucket, source_key, "UploadPartCopy", source_version)
        if CopySourceIfMatch is not None and CopySourceIfMatch != source.etag:
            raise _client_error(412, "PreconditionFailed", "UploadPartCopy")
        data = source.data
        if CopySourceRange is not None:
            start, end = self._range(CopySourceRange, len(data), "UploadPartCopy")
            data = data[start:end + 1]
        upload["parts"][PartNumber] = data
        return {"CopyPartResult": {"ETag": _etag(data)}}

    def complete_multipart_upload(self, *, Bucket: str, Key: str, UploadId: str,
                                  MultipartUpload: Mapping[str, Any]) -> dict[str, Any]:
        upload = self._upload(Bucket, Key, UploadId, "CompleteMultipartUpload")
        listed = list(MultipartUpload.get("Parts", []))
        numbers = [part["PartNumber"] for part in listed]
        if not listed or numbers != sorted(set(numbers)) or any(
                number not in upload["parts"] or part["ETag"] != _etag(upload["parts"][number])
                for number, part in zip(numbers, listed)):
            raise _client_error(400, "InvalidPart", "CompleteMultipartUpload")
        chunks = [upload["parts"][number] for number in numbers]
        if any(len(chunk) < self.min_part_bytes for chunk in chunks[:-1]):
            raise _client_error(400, "EntityTooSmall", "CompleteMultipartUpload")
        etag = '"' + hashlib.md5(b"".join(hashlib.md5(c).digest() for c in chunks)).hexdigest() + f'-{len(chunks)}"'
        version = self._store(Bucket, Key, b"".join(chunks), upload["metadata"], "CompleteMultipartUpload", etag)
        del self.uploads[UploadId]
        return {"ETag": version.etag, "VersionId": version.version_id}

    def abort_multipart_upload(self, *, Bucket: str, Key: str, UploadId: str) -> dict[str, Any]:
        self._upload(Bucket, Key, UploadId, "AbortMultipartUpload")
        del self.uploads[UploadId]
        return {}

    def delete_object(self, *, Bucket: str, Key: str, VersionId: str | None = None) -> dict[str, Any]:
        """Without a version id B2 only hides the object (a delete marker); with one it removes that version."""

        versions = self._objects(Bucket, "DeleteObject").setdefault(Key, [])
        if VersionId is None:
            marker = self._store(Bucket, Key, None, None, "DeleteObject")
            return {"DeleteMarker": True, "VersionId": marker.version_id}
        removed = [item for item in versions if item.version_id == VersionId]
        versions[:] = [item for item in versions if item.version_id != VersionId]
        if not versions:
            del self.buckets[Bucket][Key]
        self.operations.append(("DeleteObjectVersion", Bucket, Key))
        return {"VersionId": VersionId, "DeleteMarker": bool(removed) and removed[0].data is None}

    def list_object_versions(self, *, Bucket: str, Prefix: str = "", KeyMarker: str | None = None,
                             VersionIdMarker: str | None = None, MaxKeys: int = 1000) -> dict[str, Any]:
        rows = [(key, version, index == len(versions) - 1)
                for key, versions in sorted(self._objects(Bucket, "ListObjectVersions").items())
                if key.startswith(Prefix)
                for index, version in reversed(list(enumerate(versions)))]
        start = 0
        if KeyMarker is not None:
            start = next((position + 1 for position, (key, version, _) in enumerate(rows)
                          if (key, version.version_id) == (KeyMarker, VersionIdMarker)), len(rows))
        page = rows[start:start + MaxKeys]
        truncated = start + MaxKeys < len(rows)
        result: dict[str, Any] = {"IsTruncated": truncated, "Versions": [], "DeleteMarkers": []}
        for key, version, latest in page:
            entry = {"Key": key, "VersionId": version.version_id, "IsLatest": latest,
                     "LastModified": datetime.fromtimestamp(version.last_modified, timezone.utc)}
            if version.data is None:
                result["DeleteMarkers"].append(entry)
            else:
                result["Versions"].append({**entry, "ETag": version.etag, "Size": len(version.data)})
        if truncated:
            result["NextKeyMarker"], result["NextVersionIdMarker"] = page[-1][0], page[-1][1].version_id
        return result

    def list_objects_v2(self, *, Bucket: str, Prefix: str = "", **_: Any) -> dict[str, Any]:
        visible = [{"Key": key, "Size": len(versions[-1].data), "ETag": versions[-1].etag}
                   for key, versions in sorted(self._objects(Bucket, "ListObjectsV2").items())
                   if key.startswith(Prefix) and versions and versions[-1].data is not None]
        return {"KeyCount": len(visible), "Contents": visible, "IsTruncated": False}

    def _signature(self, method: str, bucket: str, key: str, date: str, expires: str, length: str) -> str:
        message = "\n".join((method, bucket, key, date, expires, length)).encode("utf-8")
        return hmac.new(self._secret, message, hashlib.sha256).hexdigest()

    def generate_presigned_url(self, ClientMethod: str, Params: Mapping[str, Any] | None = None,
                               ExpiresIn: int = 3600, HttpMethod: str | None = None) -> str:
        """Presign one GET or PUT for one object; this lane never presigns a delete."""

        method = {"get_object": "GET", "put_object": "PUT"}.get(ClientMethod)
        if method is None or HttpMethod not in {None, method}:
            raise ValueError(f"fake presign refuses {ClientMethod}")
        if not isinstance(ExpiresIn, int) or not 1 <= ExpiresIn <= MAX_PRESIGN_SECONDS:
            raise ValueError("presigned URL expiry must be 1 second to 7 days")
        params = dict(Params or {})
        bucket, key = params["Bucket"], params["Key"]
        date, expires = str(int(self.clock.now)), str(ExpiresIn)
        length = str(params["ContentLength"]) if "ContentLength" in params else ""
        query = {
            "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
            "X-Amz-Credential": "FAKEAPPLICATIONKEYID/fake-date/us-west-004/s3/aws4_request",
            "X-Amz-Date": date, "X-Amz-Expires": expires,
            "X-Amz-SignedHeaders": "content-length;host" if length else "host",
            "X-Amz-Signature": self._signature(method, bucket, key, date, expires, length),
        }
        return f"{self.endpoint}/{bucket}/{quote(key)}?{urlencode(query)}"

    def request(self, method: str, url: str, *, body: bytes = b"") -> FakeHttpResponse:
        """What a worker holding only the URL gets back from B2."""

        parts = urlsplit(url)
        bucket, _, key = unquote(parts.path).lstrip("/").partition("/")
        query = {name: values[0] for name, values in parse_qs(parts.query).items()}
        date, expires = query.get("X-Amz-Date", ""), query.get("X-Amz-Expires", "")
        length = str(len(body)) if "content-length" in query.get("X-Amz-SignedHeaders", "") else ""
        if not hmac.compare_digest(query.get("X-Amz-Signature", ""),
                                   self._signature(method, bucket, key, date, expires, length)):
            return FakeHttpResponse(403, b"<Code>SignatureDoesNotMatch</Code>")
        if self.clock.now >= int(date) + int(expires):
            return FakeHttpResponse(403, b"<Code>AccessDenied</Code><Message>Request has expired</Message>")
        try:
            if method == "GET":
                return FakeHttpResponse(200, self.get_object(Bucket=bucket, Key=key)["Body"].read())
            stored = self.put_object(Bucket=bucket, Key=key, Body=body)
            return FakeHttpResponse(200, b"", {"ETag": stored["ETag"]})
        except ClientError as exc:
            return FakeHttpResponse(exc.response["ResponseMetadata"]["HTTPStatusCode"], exc.response["Error"]["Code"].encode())


class FakeGcsError(Exception):
    def __init__(self, code: int, reason: str) -> None:
        super().__init__(f"{code} {reason}")
        self.code, self.reason = code, reason


class _TransportReader:
    """The worker's view of the transport bucket: ``storage.objects.get`` only."""

    def __init__(self, bucket: FakeTransportBucket) -> None:
        self._bucket = bucket

    def get(self, object_name: str, *, generation: int) -> bytes:
        return self._bucket.get(object_name, generation=generation)


class FakeTransportBucket:
    """A GCS bucket for transport objects: create-if-absent, generation-pinned reads, no listing."""

    def __init__(self, name: str = "blueprint-8c1ca-remote-cpu-transport", *, clock: FakeClock | None = None) -> None:
        self.name, self.clock = name, clock or FakeClock()
        self._objects: dict[str, tuple[int, bytes, float]] = {}
        self._generations = itertools.count(int(self.clock.now * 1_000_000))

    def create(self, object_name: str, data: bytes, *, if_generation_match: int | None) -> int:
        current = self._objects.get(object_name)
        if if_generation_match is not None and (current[0] if current else 0) != if_generation_match:
            raise FakeGcsError(412, "conditionNotMet")
        generation = max(next(self._generations), (current[0] + 1) if current else 0)
        self._objects[object_name] = (generation, bytes(data), self.clock.now)
        return generation

    def get(self, object_name: str, *, generation: int) -> bytes:
        current = self._objects.get(object_name)
        if current is None or current[0] != generation:
            raise FakeGcsError(404, "notFound")
        return current[1]

    def exists(self, object_name: str, *, generation: int) -> bool:
        current = self._objects.get(object_name)
        return current is not None and current[0] == generation

    def delete(self, object_name: str, *, generation: int | None = None) -> None:
        current = self._objects.get(object_name)
        if current is None or generation not in {None, current[0]}:
            raise FakeGcsError(404, "notFound")
        del self._objects[object_name]

    def reader(self) -> _TransportReader:
        return _TransportReader(self)


class FakeCloudRunError(Exception):
    """A Cloud Run Admin API failure; ``status is None`` is an ambiguous (lost) response."""

    def __init__(self, status: int | None, code: str, message: str = "") -> None:
        super().__init__(f"{status} {code} {message}".strip())
        self.status, self.code = status, code


def env_value(execution: Mapping[str, Any], name: str) -> str | None:
    """The value of one environment variable in an execution's container template."""

    for container in execution.get("template", {}).get("containers", []):
        for variable in container.get("env", []):
            if variable.get("name") == name:
                return variable.get("value")
    return None


class FakeCloudRunJobs:
    """Cloud Run Admin API v2 jobs and executions, clock-driven, with scripted outcomes.

    ``succeed`` and ``crash`` finish after ``run_seconds``; ``timeout`` fails at the execution's
    task timeout; ``hang`` never finishes until cancelled; ``lost_response`` creates the execution
    but raises an ambiguous error; ``duplicate`` creates two executions for one call.  Listings are
    newest first and capped at ``max_page_size`` whatever page size is asked for.
    """

    def __init__(self, *, clock: FakeClock | None = None, max_page_size: int = 2, start_seconds: float = 5.0,
                 run_seconds: float = 120.0) -> None:
        self.clock = clock or FakeClock()
        self.max_page_size, self.start_seconds, self.run_seconds = max_page_size, start_seconds, run_seconds
        self.jobs: dict[str, dict[str, Any]] = {}
        self.executions: dict[str, list[dict[str, Any]]] = {}
        self.list_calls = 0
        self._script: list[str] = []
        self._ids = itertools.count(1)

    def add_job(self, name: str, *, image: str, command: list[str], args: list[str] | None = None,
                env: list[dict[str, str]] | None = None, task_count: int = 1, max_retries: int = 0,
                timeout_seconds: int = 1800) -> dict[str, Any]:
        self.jobs[name] = {
            "name": name, "etag": f'"job-{next(self._ids)}"',
            "template": {"taskCount": task_count, "parallelism": 1, "template": {
                "containers": [{"image": image, "command": list(command), "args": list(args or []),
                                "env": [dict(item) for item in env or []]}],
                "maxRetries": max_retries, "timeout": f"{timeout_seconds}s",
                "executionEnvironment": "EXECUTION_ENVIRONMENT_GEN2"}},
        }
        self.executions.setdefault(name, [])
        return self.get_job(name)

    def _job(self, name: str) -> dict[str, Any]:
        if name not in self.jobs:
            raise FakeCloudRunError(404, "NOT_FOUND", name)
        return self.jobs[name]

    def get_job(self, name: str) -> dict[str, Any]:
        return deepcopy(self._job(name))

    def update_job(self, name: str, *, image: str | None = None, timeout_seconds: int | None = None,
                   max_retries: int | None = None) -> dict[str, Any]:
        job = self._job(name)
        template = job["template"]["template"]
        if image is not None:
            template["containers"][0]["image"] = image
        if timeout_seconds is not None:
            template["timeout"] = f"{timeout_seconds}s"
        if max_retries is not None:
            template["maxRetries"] = max_retries
        job["etag"] = f'"job-{next(self._ids)}"'
        return self.get_job(name)

    def script(self, *behaviours: str) -> None:
        unknown = set(behaviours) - set(BEHAVIOURS)
        if unknown:
            raise ValueError(f"unknown behaviours: {sorted(unknown)}")
        self._script.extend(behaviours)

    def run_job(self, name: str, *, etag: str | None = None, overrides: Mapping[str, Any] | None = None,
                validate_only: bool = False) -> dict[str, Any]:
        job = self._job(name)
        if etag is not None and etag != job["etag"]:
            raise FakeCloudRunError(409, "ABORTED", "etag mismatch")
        template = deepcopy(job["template"]["template"])
        overrides = dict(overrides or {})
        for index, override in enumerate(overrides.get("containerOverrides", [])):
            container = template["containers"][index]
            names = {item["name"] for item in override.get("env", [])}
            container["env"] = [item for item in container["env"] if item["name"] not in names] + [
                dict(item) for item in override.get("env", [])]
            if override.get("clearArgs") or "args" in override:
                container["args"] = list(override.get("args", []))
        template["timeout"] = overrides.get("timeout", template["timeout"])
        operation = {"name": name.rsplit("/jobs/", 1)[0] + f"/operations/op-{next(self._ids)}", "done": False}
        if validate_only:
            return {**operation, "metadata": None, "validateOnly": True}
        behaviour = self._script.pop(0) if self._script else "succeed"
        created = [self._create(name, template, overrides.get("taskCount", 1), behaviour)
                   for _ in range(2 if behaviour == "duplicate" else 1)]
        if behaviour == "lost_response":
            raise FakeCloudRunError(None, "UNAVAILABLE", "connection reset after the request was accepted")
        return {**operation, "metadata": self._view(created[0])}

    def _create(self, job: str, template: dict[str, Any], task_count: int, behaviour: str) -> dict[str, Any]:
        suffix = base64.b32encode(hashlib.sha256(str(next(self._ids)).encode()).digest()).decode().lower()[:5]
        execution = {"name": f"{job}/executions/{job.rsplit('/', 1)[-1]}-{suffix}", "job": job.rsplit("/", 1)[-1],
                     "template": deepcopy(template), "taskCount": task_count, "behaviour": behaviour,
                     "created": self.clock.now, "cancelled": None}
        self.executions[job].append(execution)
        return execution

    def _view(self, execution: dict[str, Any]) -> dict[str, Any]:
        now, created = self.clock.now, execution["created"]
        started = created + self.start_seconds
        timeout = float(execution["template"]["timeout"].rstrip("s"))
        behaviour = execution["behaviour"]
        finished_at = None if behaviour == "hang" else started + (timeout if behaviour == "timeout" else self.run_seconds)
        state, reason, completed = "running" if now >= started else "pending", None, None
        if execution["cancelled"] is not None:
            state, reason, completed = "cancelled", "Cancelled", execution["cancelled"]
        elif finished_at is not None and now >= finished_at:
            completed = finished_at
            state, reason = ("failed", "NonZeroExitCode") if behaviour == "crash" else (
                ("failed", "DeadlineExceeded") if behaviour == "timeout" else ("succeeded", None))
        view = {
            "name": execution["name"], "job": execution["job"], "createTime": _iso(created),
            "startTime": _iso(started) if now >= started else None,
            "completionTime": _iso(completed) if completed is not None else None,
            "taskCount": execution["taskCount"], "retriedCount": 0,
            "runningCount": int(state == "running"), "succeededCount": int(state == "succeeded"),
            "failedCount": int(state == "failed"), "cancelledCount": int(state == "cancelled"),
            "template": deepcopy(execution["template"]), "reconciling": False,
            "etag": f'"{execution["name"].rsplit("-", 1)[-1]}-{state}"',
            "conditions": [{"type": "Completed", "state": {
                "succeeded": "CONDITION_SUCCEEDED", "failed": "CONDITION_FAILED", "cancelled": "CONDITION_FAILED",
            }.get(state, "CONDITION_PENDING"), "reason": reason}],
        }
        return view

    def _execution(self, name: str) -> dict[str, Any]:
        for rows in self.executions.values():
            for execution in rows:
                if execution["name"] == name:
                    return execution
        raise FakeCloudRunError(404, "NOT_FOUND", name)

    def get_execution(self, name: str) -> dict[str, Any]:
        return self._view(self._execution(name))

    def list_executions(self, job: str, *, page_size: int | None = None,
                        page_token: str | None = None) -> dict[str, Any]:
        self._job(job)
        self.list_calls += 1
        rows = [self._view(item) for item in reversed(self.executions[job])]
        offset = 0
        if page_token:
            try:
                prefix, _, number = base64.urlsafe_b64decode(page_token.encode()).decode().partition(":")
                offset = int(number)
                if prefix != job:
                    raise ValueError(prefix)
            except (ValueError, UnicodeDecodeError):
                raise FakeCloudRunError(400, "INVALID_ARGUMENT", "page token") from None
        size = min(page_size or self.max_page_size, self.max_page_size)
        page = {"executions": rows[offset:offset + size]}
        if offset + size < len(rows):
            page["nextPageToken"] = base64.urlsafe_b64encode(f"{job}:{offset + size}".encode()).decode()
        return page

    def cancel_execution(self, name: str, *, etag: str | None = None) -> dict[str, Any]:
        execution = self._execution(name)
        view = self._view(execution)
        if view["completionTime"] is not None:
            raise FakeCloudRunError(400, "FAILED_PRECONDITION", "execution already completed")
        if etag is not None and etag != view["etag"]:
            raise FakeCloudRunError(409, "ABORTED", "etag mismatch")
        execution["cancelled"] = self.clock.now
        return self._view(execution)
