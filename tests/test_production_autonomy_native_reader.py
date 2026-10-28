"""Offline acquisition fixtures; no private source or production proof.

Optional SDK regeneration uses real installed DocumentSnapshot/Reference,
Timestamp and Serializer objects. It constructs no Firestore Client, strips the
child environment and forbids auth/network. Ordinary tests use the resulting
native protocol fixture; no SDK installation or credential fallback is allowed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from blueprint_pipeline import production_autonomy_native_reader as native
from blueprint_pipeline.production_autonomy_acquisition import (
    AuthenticatedAcceptanceCollector,
    SourceSelector,
)
from blueprint_pipeline.production_autonomy_acceptance import AcceptanceEvidenceError
from tests.test_production_autonomy_acquisition import sources  # noqa: F401
from tests.test_production_autonomy_acceptance import retained, stamp  # noqa: F401

PROJECT, DATABASE, CAMPAIGN = "blueprint-offline-fixture", "(default)", "campaign-fixture"
RESOURCE = "taskEvaluationPolicyRuns/" + CAMPAIGN
SELECTOR = SourceSelector("publication_ack", "webapp_run", RESOURCE)
FORMAT = "firestore_sdk_snapshot_proto_json.v1"

SDK_SCRIPT = r"""
const root=process.argv[1], input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const {createRequire}=require('node:module'), scoped=createRequire(root+'/package.json');
for(const name of ['node:http','node:https']) require(name).request=()=>{throw Error('offline_network_forbidden')};
require('node:net').Socket.prototype.connect=()=>{throw Error('offline_network_forbidden')};
require('node:tls').connect=()=>{throw Error('offline_network_forbidden')};
global.fetch=()=>{throw Error('offline_network_forbidden')};
const auth=scoped('google-auth-library');
for(const name of ['getClient','getApplicationDefault','getAccessToken'])
  auth.GoogleAuth.prototype[name]=()=>{throw Error('offline_auth_forbidden')};
const {DocumentSnapshot}=require(root+'/build/src/document');
const {DocumentReference}=require(root+'/build/src/reference/document-reference');
const {ResourcePath}=require(root+'/build/src/path'),{Serializer}=require(root+'/build/src/serializer');
const {Timestamp}=require(root+'/build/src/timestamp');
const fixture={projectId:input.project,databaseId:input.database,_settings:{useBigInt:true},
  doc(path){return new DocumentReference(this,new ResourcePath(...path.split('/')))}};
fixture._serializer=new Serializer(fixture);
const time=value=>new Timestamp(Number(value.seconds),value.nanos||0);
const ref=fixture.doc(input.resource);
const snapshot=new DocumentSnapshot(ref,input.data===null?undefined:fixture._serializer.encodeFields(input.data),
  time(input.read),input.data===null?undefined:time(input.create),input.data===null?undefined:time(input.update));
const envelope=input.data===null?{missing:ref.formattedName,readTime:snapshot.readTime.toProto().timestampValue}
  :{found:snapshot.toDocumentProto(),readTime:snapshot.readTime.toProto().timestampValue};
console.log(JSON.stringify({sdk_version:require(root+'/package.json').version,
  classes:[snapshot.constructor.name,ref.constructor.name,snapshot.readTime.constructor.name],
  envelope_json:JSON.stringify(envelope)}));
"""


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def wire(data=None, *, resource=RESOURCE, project=PROJECT):
    # Firestore native document protocol, including a nanosecond boundary that
    # ordinary datetime.fromisoformat would truncate.
    return {
        "found": {
            "name": f"projects/{project}/databases/{DATABASE}/documents/{resource}",
            "fields": {
                "updated_at_iso": {"stringValue": "2026-10-28T00:00:00Z"},
                "count": {"integerValue": "3"},
                "enabled": {"booleanValue": True},
                "empty": {"nullValue": "NULL_VALUE"},
            },
            "createTime": {"seconds": "1793145600", "nanos": 1},
            "updateTime": {"seconds": "1793145601", "nanos": 987654321},
        },
        "readTime": {"seconds": "1793145602", "nanos": 123456789},
    }


def sdk_wire(
    data,
    *,
    project=PROJECT,
    resource=RESOURCE,
    create="2026-10-28T00:00:00Z",
    update="2026-10-28T00:00:01Z",
    read="2026-10-28T00:00:02Z",
):
    root = os.environ.get("BLUEPRINT_NATIVE_READER_TEST_SDK")
    node = os.environ.get("BLUEPRINT_NATIVE_READER_TEST_NODE") or shutil.which("node")
    if not root or not node:
        pytest.skip(
            "Optional installed Firestore SDK fixture regeneration unavailable; no install attempted"
        )

    def proto(value):
        _, (seconds, nanos) = native._time(value)
        return {"seconds": str(seconds), "nanos": nanos}

    value = {
        "data": data,
        "project": project,
        "database": DATABASE,
        "resource": resource,
        "create": proto(create),
        "update": proto(update),
        "read": proto(read),
    }
    result = subprocess.run(
        [node, "-e", SDK_SCRIPT, root],
        input=json.dumps(value),
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
        env={"NODE_NO_WARNINGS": "1"},
    )
    output = json.loads(result.stdout)
    assert output["classes"] == ["DocumentSnapshot", "DocumentReference", "Timestamp"]
    return output["envelope_json"].encode(), output


def retainer(root):
    def keep(raw, label):
        path = root / (label + ".json")
        if path.exists():
            assert path.read_bytes() == raw
        else:
            path.write_bytes(raw)
            path.chmod(0o600)
        return {"path": str(path), "sha256": native._sha(raw), "size_bytes": len(raw)}

    return keep


def response(selector=SELECTOR, body=None, **changes):
    return native.NativeSnapshotRead(
        selector,
        encoded(wire()) if body is None else body,
        FORMAT,
        "webapp-service",
        "firestore_server_identity",
        "fixture",
        "hermetic",
        "a" * 40,
        "sha256:" + "b" * 64,
        "2026-10-28T00:00:03Z",
        **changes,
    )


def reader(tmp_path, callback):
    return native.NativeFirestoreReader(
        project=PROJECT,
        database=DATABASE,
        campaign_id=CAMPAIGN,
        service_identity="webapp-service",
        read=callback,
        retain=retainer(tmp_path),
    )


def verify(value, root, **changes):
    options = {
        "root": root,
        "project": PROJECT,
        "database": DATABASE,
        "frozen_at": "2026-10-28T00:00:00Z",
        **changes,
    }
    return native.verify_native_evidence(value, **options)


def test_native_roundtrip_keeps_raw_and_derived_identity_and_exact_nanos(tmp_path):
    adapted = reader(tmp_path, lambda _: response())(SELECTOR)
    assert json.loads(adapted.body)["count"] == 3
    assert adapted.version == "2026-10-28T00:00:01.987654321Z"
    assert adapted.native_evidence["read_time"] == "2026-10-28T00:00:02.123456789Z"
    assert adapted.retained_record["sha256"] != adapted.native_evidence["record"]["sha256"]
    assert verify(adapted, tmp_path)["native_serialization"] == FORMAT


def test_actual_installed_sdk_objects_serialize_without_client_auth_or_network(tmp_path):
    raw, output = sdk_wire(
        {
            "updated_at_iso": "2026-10-28T00:00:00Z",
            "count": 3,
            "empty": None,
            "enabled": True,
            "nested": {"label": "fixture"},
            "items": [1, "x"],
        },
        create="2026-10-28T00:00:00.000000001Z",
        update="2026-10-28T00:00:01.987654321Z",
        read="2026-10-28T00:00:02.123456789Z",
    )
    adapted = reader(tmp_path, lambda _: response(body=raw))(SELECTOR)
    assert json.loads(adapted.body)["nested"] == {"label": "fixture"}
    assert json.loads(adapted.body)["items"] == [1, "x"]
    assert verify(adapted, tmp_path)["update_time"] == adapted.version
    assert output["sdk_version"]


@pytest.mark.parametrize(
    "selector",
    [
        replace(SELECTOR, resource="inboundRequests/private"),
        replace(SELECTOR, resource=RESOURCE + "/nested/doc"),
        replace(SELECTOR, resource="../" + RESOURCE),
        replace(SELECTOR, role="authority"),
        replace(SELECTOR, phase="retry"),
        replace(SELECTOR, resource="https://example.invalid"),
    ],
)
def test_closed_scope_refuses_before_callback(tmp_path, selector):
    calls = []
    capability = reader(tmp_path, lambda value: calls.append(value))
    with pytest.raises(AcceptanceEvidenceError, match="selector_scope_mismatch"):
        capability(selector)
    assert calls == [] and capability.diagnostics()["reads"] == 0


def test_native_data_never_substitutes_for_actual_customer_get(tmp_path):
    capability = reader(
        tmp_path, lambda _: pytest.fail("native callback must not receive HTTP selector")
    )
    assert (
        capability(
            SourceSelector(
                "customer_readback", "webapp_workspace", "GET /api/job-workflow/admin/job-fixture"
            )
        )
        is None
    )
    assert capability.diagnostics()["actual_customer_get_supported"] is False


def test_missing_is_retained_as_missing_without_document_version_or_acceptance(tmp_path):
    body = encoded({"missing": wire()["found"]["name"], "readTime": wire()["readTime"]})
    capability = reader(tmp_path, lambda _: response(body=body))
    assert capability(SELECTOR) is None
    event = capability.diagnostics()["events"][0]
    assert event["status"] == "missing" and "native_record" in event
    assert event["read_time"] == "2026-10-28T00:00:02.123456789Z"
    assert event["create_time"] is None and event["update_time"] is None
    assert len(list(tmp_path.glob("*.json"))) == 1


@pytest.mark.parametrize(
    "kind,value",
    [
        ("referenceValue", "projects/foreign/databases/(default)/documents/x/y"),
        ("bytesValue", "Ynl0ZXM="),
        ("geoPointValue", {"latitude": 0, "longitude": 0}),
        ("timestampValue", "2026-10-28T00:00:00Z"),
        ("integerValue", "9007199254740992"),
        ("integerValue", True),
        ("doubleValue", "NaN"),
        ("doubleValue", float("inf")),
        ("booleanValue", "true"),
        ("nullValue", False),
    ],
)
def test_unsupported_or_unsafe_values_refuse_and_keep_native_failure(tmp_path, kind, value):
    envelope = wire()
    envelope["found"]["fields"]["unsafe"] = {kind: value}
    body = json.dumps(envelope).encode()
    capability = reader(tmp_path, lambda _: response(body=body))
    with pytest.raises(AcceptanceEvidenceError):
        capability(SELECTOR)
    assert capability.diagnostics()["events"][0]["status"] == "unverified"
    assert "native_record" in capability.diagnostics()["events"][0]
    assert len(list(tmp_path.glob("*.json"))) == 1


@pytest.mark.parametrize(
    "change",
    [
        "foreign_project",
        "foreign_database",
        "foreign_path",
        "version_order",
        "nanos",
        "boolean_seconds",
        "duplicate_key",
        "extra_envelope",
        "deep",
        "multi_type",
    ],
)
def test_malformed_native_envelopes_fail_closed(tmp_path, change):
    envelope = wire()
    if change == "foreign_project":
        envelope["found"]["name"] = envelope["found"]["name"].replace(PROJECT, "foreign")
    elif change == "foreign_database":
        envelope["found"]["name"] = envelope["found"]["name"].replace("(default)", "foreign")
    elif change == "foreign_path":
        envelope["found"]["name"] += "/child/doc"
    elif change == "version_order":
        envelope["readTime"] = {"seconds": "1793145601", "nanos": 987654320}
    elif change == "nanos":
        envelope["readTime"]["nanos"] = 1_000_000_000
    elif change == "boolean_seconds":
        envelope["readTime"]["seconds"] = True
    elif change == "extra_envelope":
        envelope["transaction"] = "unexpected"
    elif change == "multi_type":
        envelope["found"]["fields"]["count"]["stringValue"] = "3"
    elif change == "deep":
        field = {"stringValue": "leaf"}
        for _ in range(40):
            field = {"arrayValue": {"values": [field]}}
        envelope["found"]["fields"]["deep"] = field
    body = encoded(envelope)
    if change == "duplicate_key":
        body = body[:-1] + b',"readTime":"2026-10-28T00:00:03Z"}'
    with pytest.raises(AcceptanceEvidenceError):
        reader(tmp_path, lambda _: response(body=body))(SELECTOR)


@pytest.mark.parametrize(
    "change", ["body", "version", "pins", "read_time", "native_bytes", "missing_sidecar_pin"]
)
def test_collector_native_hook_reopens_and_checks_every_identity(tmp_path, change):
    adapted = reader(tmp_path, lambda _: response())(SELECTOR)
    options = {}
    if change == "body":
        adapted = replace(adapted, body=b'{"different":true}')
    elif change == "version":
        adapted = replace(adapted, version="2026-10-28T00:00:01.987654320Z")
    elif change == "pins":
        options["project"] = "foreign"
    elif change == "missing_sidecar_pin":
        options["project"] = None
    elif change == "read_time":
        sidecar = deepcopy(adapted.native_evidence)
        sidecar["read_time"] = "2026-10-28T00:00:02.123456788Z"
        adapted = replace(adapted, native_evidence=sidecar)
    else:
        Path(adapted.native_evidence["record"]["path"]).write_bytes(b"changed")
    with pytest.raises(AcceptanceEvidenceError):
        verify(adapted, tmp_path, **options)


def test_stale_and_future_acquisition_times_refuse(tmp_path):
    adapted = reader(tmp_path, lambda _: response())(SELECTOR)
    with pytest.raises(AcceptanceEvidenceError, match="native_read_time_invalid"):
        verify(adapted, tmp_path, frozen_at="2026-10-28T00:00:02.123456790Z")
    with pytest.raises(AcceptanceEvidenceError, match="acquisition_time_invalid"):
        reader(
            tmp_path, lambda _: replace(response(), observed_at="2026-10-28T00:00:02.123456788Z")
        )(SELECTOR)


def test_timestamp_offsets_normalize_without_losing_nanoseconds():
    assert native._time("2026-10-27T17:00:02.123456789-07:00") == native._time(
        "2026-10-28T00:00:02.123456789Z"
    )
    assert (
        native._time("2026-10-28T00:00:02.123456790+00:00")[1]
        > native._time("2026-10-28T00:00:02.123456789Z")[1]
    )


def test_read_and_byte_bounds_and_denial_never_retry(tmp_path, monkeypatch):
    calls = []

    def denied(selector):
        calls.append(selector)
        raise PermissionError("private diagnostic")

    capability = reader(tmp_path, denied)
    with pytest.raises(AcceptanceEvidenceError, match="^native_reader_permission_denied$"):
        capability(SELECTOR)
    assert len(calls) == 1
    with pytest.raises(AcceptanceEvidenceError, match="source_already_denied"):
        capability(replace(SELECTOR, role="queued_ack"))
    assert len(calls) == 1
    monkeypatch.setattr(native, "MAX_READS", 1)
    with pytest.raises(AcceptanceEvidenceError, match="read_limit"):
        capability(SELECTOR)
    assert len(calls) == 1
    monkeypatch.setattr(native, "MAX_TOTAL_BYTES", len(encoded(wire())))
    with pytest.raises(AcceptanceEvidenceError, match="byte_limit"):
        reader(tmp_path, lambda _: response())(SELECTOR)
    monkeypatch.setattr(native, "MAX_BYTES", 10)
    with pytest.raises(AcceptanceEvidenceError, match="byte_limit"):
        reader(tmp_path, lambda _: response())(SELECTOR)


def test_import_does_not_load_google_clients_auth_credentials_or_network():
    source = str(Path(native.__file__).parents[1])
    script = """import sys,socket,builtins,os
sys.path.insert(0,sys.argv[1])
def denied(*args,**kwargs):raise RuntimeError('offline_network_forbidden')
socket.socket.connect=denied
import blueprint_pipeline.production_autonomy_native_reader
assert not any(name.startswith(('google.auth','google.cloud.firestore','firebase_admin')) for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, source],
        env={},
        check=True,
        capture_output=True,
        timeout=15,
    )


@pytest.mark.parametrize("change_after", [False, True])
def test_actual_sdk_native_documents_join_collector_only_with_separate_http_read(
    sources, change_after  # noqa: F811 -- reuse the existing acquisition fixture
):
    root = sources.retained["root"]
    ticks = [0]

    def next_stamp():
        ticks[0] += 1
        return stamp(70 + ticks[0])

    def read(selector):
        data = sources.run if selector.kind == "webapp_run" else sources.result
        raw, _ = sdk_wire(
            data,
            project=PROJECT,
            resource=selector.resource,
            create=stamp(25),
            update=stamp(60).replace("+00:00", ".000000001Z")
            if change_after and selector.phase == "after"
            else stamp(60),
            read=stamp(65),
        )
        return native.NativeSnapshotRead(
            selector,
            raw,
            FORMAT,
            "webapp-service",
            "firestore_server_identity",
            "production",
            "natural_production",
            sources.attempt["releases"]["webapp_commit"],
            sources.attempt["releases"]["webapp_image_digest"],
            next_stamp(),
        )

    capability = native.NativeFirestoreReader(
        project=PROJECT,
        database=DATABASE,
        campaign_id=sources.plan["campaign_id"],
        service_identity="webapp-service",
        read=read,
        retain=retainer(root),
    )
    primary = capability(
        SourceSelector(
            "publication_ack",
            "webapp_result",
            "captureTaskEvaluationRuns/" + sources.run["result_record_id"],
        )
    )
    http_reader = sources.collector()._reader

    def joined(selector):
        if selector.kind in {"webapp_run", "webapp_result"}:
            return capability(selector)
        # Explicit separate synthetic HTTP capability, never inferred from
        # Firestore fields. This test still establishes no real production GET.
        return replace(http_reader(selector), observed_at=next_stamp())

    def collector(callback):
        return AuthenticatedAcceptanceCollector(
            frozen_attempt=sources.attempt,
            plan=sources.plan,
            publication=sources.publication,
            owner_user_id="owner-1",
            team_namespace="tenant-1",
            scene_intake_id="intake-1",
            webapp_service_identity="webapp-service",
            pipeline_service_identity="pipeline-service",
            evidence_root=root,
            reader=callback,
            firestore_project_id=PROJECT,
            firestore_database_id=DATABASE,
        )

    accepted = collector(joined)
    observation = accepted(
        "publication_ack", primary.retained_record["sha256"], json.loads(primary.body)
    )
    assert (observation is None) == change_after, accepted.diagnostics()
    if change_after:
        assert (
            accepted.diagnostics()["failures"][-1]["code"]
            == "acquisition_persistence_changed_during_readback"
        )
    assert any(row.get("native_record") for row in accepted.diagnostics()["sources"])
    missing_get = collector(capability)
    assert (
        missing_get("publication_ack", primary.retained_record["sha256"], json.loads(primary.body))
        is None
    )
