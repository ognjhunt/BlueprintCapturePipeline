"""Offline native-document adapter for the existing AUT-21 collector.

Only explicitly injected read/retention capabilities perform IO. There is no
SDK client, credential factory, network fallback, endpoint or import hook here.
SDK snapshot-proto JSON is labelled as a serialization, never original HTTP
wire bytes. The original native envelope and derived document have separate
retained identities. Neither supplies the required actual WebApp GET.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

from .production_autonomy_acceptance import AcceptanceEvidenceError, _safe_path
from .production_autonomy_acquisition import AuthenticatedSourceRead, SourceSelector, _json

FORMATS = frozenset({"firestore_batch_get_json.v1", "firestore_sdk_snapshot_proto_json.v1"})
MAX_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_READS = 256
MAX_NODES = 100_000
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}\Z")
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
_STAMP = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?(Z|[+-]\d\d:\d\d)\Z")
_ROLES = frozenset({"queued_ack", "publication_ack", "customer_readback"})


def _require(ok, code):
    if not ok:
        raise AcceptanceEvidenceError("native_reader_" + code)


def _sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _encoded(value):
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
        ).encode()
    except (ValueError, UnicodeError, TypeError, RecursionError):
        raise AcceptanceEvidenceError("native_reader_json_value_invalid") from None


def _time(value):
    """Preserve server nanoseconds; datetime alone would silently truncate."""
    if isinstance(value, str):
        match = _STAMP.fullmatch(value)
        _require(match is not None, "timestamp_invalid")
        try:
            moment = datetime.strptime(match[1], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            raise AcceptanceEvidenceError("native_reader_timestamp_invalid") from None
        seconds = (moment - datetime(1970, 1, 1, tzinfo=timezone.utc)).days * 86400
        seconds += moment.hour * 3600 + moment.minute * 60 + moment.second
        if match[3] != "Z":
            hours, minutes = int(match[3][1:3]), int(match[3][4:6])
            _require(hours < 24 and minutes < 60, "timestamp_invalid")
            seconds -= (1 if match[3][0] == "+" else -1) * (hours * 3600 + minutes * 60)
        nanos = int((match[2] or "").ljust(9, "0"))
    else:
        _require(
            isinstance(value, Mapping)
            and set(value) <= {"seconds", "nanos"}
            and "seconds" in value,
            "timestamp_invalid",
        )
        seconds, nanos = value["seconds"], value.get("nanos", 0)
        if isinstance(seconds, str):
            _require(re.fullmatch(r"-?(?:0|[1-9][0-9]{0,11})", seconds), "timestamp_invalid")
            seconds = int(seconds)
        _require(
            type(seconds) is int and type(nanos) is int and 0 <= nanos < 1_000_000_000,
            "timestamp_invalid",
        )
    try:
        moment = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)
    except (ValueError, OverflowError):
        raise AcceptanceEvidenceError("native_reader_timestamp_invalid") from None
    stamp = moment.isoformat(timespec="seconds").replace("+00:00", "") + f".{nanos:09d}Z"
    return stamp, (seconds, nanos)


def _fields(fields, depth=0, budget=None):
    _require(isinstance(fields, dict), "fields_invalid")
    budget = [MAX_NODES] if budget is None else budget
    return {key: _value(value, depth + 1, budget) for key, value in fields.items()}


def _value(value, depth, budget):
    budget[0] -= 1
    _require(
        depth <= 32 and budget[0] >= 0 and isinstance(value, dict) and len(value) == 1,
        "value_invalid",
    )
    kind, child = next(iter(value.items()))
    if kind == "nullValue":
        _require(
            child is None or child == "NULL_VALUE" or (type(child) is int and child == 0),
            "value_invalid",
        )
        return None
    if kind == "booleanValue":
        _require(type(child) is bool, "value_invalid")
    elif kind == "stringValue":
        _require(isinstance(child, str), "value_invalid")
    elif kind == "integerValue":
        _require(
            type(child) is int
            or (isinstance(child, str) and re.fullmatch(r"-?(?:0|[1-9][0-9]{0,18})", child)),
            "integer_invalid",
        )
        child = int(child)
        _require(abs(child) <= 2**53 - 1, "integer_unsafe")
    elif kind == "doubleValue":
        _require(type(child) in (int, float), "value_invalid")
        try:
            child = float(child)
        except OverflowError:
            raise AcceptanceEvidenceError("native_reader_value_invalid") from None
        _require(math.isfinite(child), "value_invalid")
    elif kind == "mapValue":
        _require(isinstance(child, dict) and set(child) <= {"fields"}, "value_invalid")
        return _fields(child.get("fields", {}), depth, budget)
    elif kind == "arrayValue":
        _require(
            isinstance(child, dict)
            and set(child) <= {"values"}
            and isinstance(child.get("values", []), list),
            "value_invalid",
        )
        return [_value(item, depth + 1, budget) for item in child.get("values", [])]
    else:
        # References, bytes, GeoPoints, timestamps and server sentinels are not
        # the existing JSON-only run/result contract. Never stringify them.
        raise AcceptanceEvidenceError("native_reader_value_type_unsupported")
    return child


def decode_native_document(raw, *, serialization, project, database, resource):
    _require(
        serialization in FORMATS and type(raw) is bytes and 0 < len(raw) <= MAX_BYTES,
        "envelope_invalid",
    )
    try:
        envelope = _json(raw)
    except (AcceptanceEvidenceError, RecursionError):
        raise AcceptanceEvidenceError("native_reader_envelope_invalid") from None
    prefix = f"projects/{project}/databases/{database}/documents/"
    read_time, read = _time(envelope.get("readTime"))
    if set(envelope) == {"missing", "readTime"}:
        _require(envelope["missing"] == prefix + resource, "document_scope_mismatch")
        return None, {"read_time": read_time, "create_time": None, "update_time": None}
    _require(
        set(envelope) == {"found", "readTime"} and isinstance(envelope["found"], dict),
        "envelope_invalid",
    )
    document = envelope["found"]
    _require(
        set(document) == {"name", "fields", "createTime", "updateTime"}
        and document["name"] == prefix + resource,
        "document_scope_mismatch",
    )
    create_time, create = _time(document["createTime"])
    update_time, update = _time(document["updateTime"])
    _require(create <= update <= read, "version_order_invalid")
    return _fields(document["fields"]), {
        "read_time": read_time,
        "create_time": create_time,
        "update_time": update_time,
    }


@dataclass(frozen=True)
class NativeSnapshotRead:
    """Returned by an existing independently authorized reader, not job JSON.

    SDK timestamps do not prove the WebApp's serving commit/image. Those fields
    need a separately authenticated service observation, never expected-pin
    echoes. A test-created instance establishes only offline contract behavior.
    The injected reader normalizes a denied operation to PermissionError; that
    exact source is then latched without retry or another route.
    """

    selector: SourceSelector
    body: bytes
    serialization: str
    service_identity: str
    authentication: str
    environment: str
    origin: str
    commit: str
    image_digest: str
    observed_at: str


class NativeFirestoreReader:
    """Closed native run/result adapter; absent consumer GET stays unknown."""

    def __init__(
        self,
        *,
        project: str,
        database: str,
        campaign_id: str,
        service_identity: str,
        read: Callable,
        retain: Callable,
    ):
        _require(
            all(
                isinstance(v, str) and _ID.fullmatch(v)
                for v in (project, campaign_id, service_identity)
            )
            and (database == "(default)" or isinstance(database, str) and _ID.fullmatch(database))
            and callable(read)
            and callable(retain),
            "pins_invalid",
        )
        result = "screening-" + hashlib.sha256(campaign_id.encode()).hexdigest()[:32]
        self._project, self._database, self._service = project, database, service_identity
        self._paths = {
            "webapp_run": "taskEvaluationPolicyRuns/" + campaign_id,
            "webapp_result": "captureTaskEvaluationRuns/" + result,
        }
        self._read, self._retain = read, retain
        self._reads, self._bytes, self._events = 0, 0, []
        self._denied = set()

    def _keep(self, raw, label):
        _require(
            type(raw) is bytes
            and 0 < len(raw) <= MAX_BYTES
            and self._bytes + len(raw) <= MAX_TOTAL_BYTES,
            "byte_limit",
        )
        self._bytes += len(raw)
        record = self._retain(raw, label + "-" + _sha(raw)[7:])
        _require(
            isinstance(record, Mapping)
            and record.get("sha256") == _sha(raw)
            and type(record.get("size_bytes")) is int
            and record["size_bytes"] == len(raw),
            "retention_invalid",
        )
        return dict(record)

    def __call__(self, selector):
        _require(type(selector) is SourceSelector, "selector_invalid")
        if selector.kind not in self._paths:
            return None
        _require(
            selector.role in _ROLES
            and selector.phase in {"before", "after"}
            and selector.resource == self._paths[selector.kind],
            "selector_scope_mismatch",
        )
        _require(self._reads < MAX_READS, "read_limit")
        _require(selector.resource not in self._denied, "source_already_denied")
        self._reads += 1
        event = {"kind": selector.kind, "phase": selector.phase, "status": "unverified"}
        self._events.append(event)
        try:
            try:
                response = self._read(selector)
            except PermissionError:
                self._denied.add(selector.resource)
                event.update(status="denied", code="native_reader_permission_denied")
                raise AcceptanceEvidenceError(event["code"]) from None
            if response is None:
                event["status"] = "unavailable"
                return None
            _require(
                type(response) is NativeSnapshotRead and response.selector == selector,
                "response_invalid",
            )
            raw_record = self._keep(response.body, "native")
            event["native_record"] = raw_record
            value, times = decode_native_document(
                response.body,
                serialization=response.serialization,
                project=self._project,
                database=self._database,
                resource=selector.resource,
            )
            event.update(times)
            _require(
                _time(times["read_time"])[1] <= _time(response.observed_at)[1],
                "acquisition_time_invalid",
            )
            _require(
                response.service_identity == self._service
                and response.authentication == "firestore_server_identity",
                "response_identity_invalid",
            )
            if value is None:
                event["status"] = "missing"
                return None
            body = _encoded(value)
            record = self._keep(body, "document")
            native = {
                "record": raw_record,
                "serialization": response.serialization,
                "project": self._project,
                "database": self._database,
                **times,
            }
            event["status"] = "retained"
            return AuthenticatedSourceRead(
                selector,
                body,
                record,
                response.service_identity,
                response.authentication,
                response.environment,
                response.origin,
                response.commit,
                response.image_digest,
                response.observed_at,
                times["update_time"],
                native_evidence=native,
            )
        except AcceptanceEvidenceError as error:
            event["code"] = str(error)
            raise
        except Exception:
            event["code"] = "native_reader_capability_unavailable"
            raise AcceptanceEvidenceError(event["code"]) from None

    def diagnostics(self):
        return {
            "reads": self._reads,
            "retained_bytes": self._bytes,
            "events": list(self._events),
            "actual_customer_get_supported": False,
        }


def _retained_bytes(root, record):
    from .team_robot_model_delivery import _anchored_directory_fd, _walk_directory_fd

    _require(
        isinstance(record, Mapping)
        and type(record.get("size_bytes")) is int
        and 0 < record["size_bytes"] <= MAX_BYTES
        and _SHA.fullmatch(str(record.get("sha256"))),
        "retention_invalid",
    )
    root = Path(root).absolute()
    path = _safe_path(root, record.get("path", record.get("relative_path")))
    with ExitStack() as stack:
        directory = _anchored_directory_fd(root, create=False)
        stack.callback(os.close, directory)
        relative = path.relative_to(root)
        parent = _walk_directory_fd(directory, relative.parts[:-1], create=False)
        stack.callback(os.close, parent)
        fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            _require(
                stat.S_ISREG(info.st_mode) and info.st_size == record["size_bytes"],
                "retention_invalid",
            )
            raw = stream.read(record["size_bytes"] + 1)
    _require(
        len(raw) == record["size_bytes"] and _sha(raw) == record["sha256"], "retention_invalid"
    )
    return raw


def verify_native_evidence(response, *, root, project, database, frozen_at):
    """Minimal collector hook: re-open/decode native bytes, never trust metadata."""
    native = response.native_evidence
    _require(
        isinstance(native, Mapping)
        and set(native)
        == {
            "record",
            "serialization",
            "project",
            "database",
            "read_time",
            "create_time",
            "update_time",
        }
        and project is not None
        and database is not None
        and native["project"] == project
        and native["database"] == database,
        "native_pins_missing_or_changed",
    )
    _require(response.selector.kind in {"webapp_run", "webapp_result"}, "selector_scope_mismatch")
    raw = _retained_bytes(root, native["record"])
    data, times = decode_native_document(
        raw,
        serialization=native["serialization"],
        project=project,
        database=database,
        resource=response.selector.resource,
    )
    _require(
        data is not None
        and _encoded(data) == response.body
        and all(native[key] == value for key, value in times.items())
        and response.version == times["update_time"],
        "native_body_or_version_mismatch",
    )
    _require(
        _time(frozen_at)[1] <= _time(times["read_time"])[1] <= _time(response.observed_at)[1],
        "native_read_time_invalid",
    )
    return {
        "native_record": dict(native["record"]),
        "native_serialization": native["serialization"],
        **times,
    }


def verify_native_readback_interval(reads):
    """A cached snapshot cannot prove persistence after the actual HTTP read.

    Each native sidecar was re-opened and authenticated by the collector. Local
    acquisition completion cannot refresh its server readTime. Clock equality
    or ambiguity at the HTTP boundary therefore stays unproven. The legacy
    trusted-reader path without native sidecars does not call this helper.
    """
    _require(
        isinstance(reads, (tuple, list)) and len(reads) == 5, "native_readback_interval_invalid"
    )
    before, http, after = reads[:2], reads[2], reads[3:]
    _require(
        all(isinstance(row.native_evidence, Mapping) for row in (*before, *after)),
        "native_readback_mixed_mode",
    )
    interval = http.acquisition_interval
    _require(
        isinstance(interval, Mapping)
        and set(interval) == {"started_at", "completed_at"}
        and all(type(value) is str for value in interval.values()),
        "http_interval_unproven",
    )
    start, end = (_time(interval[key])[1] for key in ("started_at", "completed_at"))
    local = [_time(row.observed_at)[1] for row in reads]
    native_before = [_time(row.native_evidence["read_time"])[1] for row in before]
    native_after = [_time(row.native_evidence["read_time"])[1] for row in after]
    _require(
        start <= end == local[2]
        and local[0] <= local[1] <= start
        and native_before == sorted(native_before)
        and all(server <= acquired for server, acquired in zip(native_before, local[:2]))
        and native_after == sorted(native_after)
        and end < native_after[0] <= local[3] <= native_after[1] <= local[4]
        and (local[-1][0] - local[0][0]) * 1_000_000_000 + local[-1][1] - local[0][1]
        <= 120_000_000_000,
        "native_readback_interval_invalid",
    )
    return {"native_readback_interval": {
        "http_acquisition_interval": dict(interval),
        "before_read_times": [row.native_evidence["read_time"] for row in before],
        "after_read_times": [row.native_evidence["read_time"] for row in after],
    }}
