"""ADP-009D: bounded disk transport for the no-paid joined load proof."""

from __future__ import annotations

import hashlib
import io

import pytest

from scripts.control_plane_concurrency_fixture import FilesystemObjectStore


def test_fixture_store_streams_and_reads_back_exact_bytes(tmp_path):
    store = FilesystemObjectStore(tmp_path)
    payload = b"scene fixture" * 10000
    store.put_object(Bucket="fixtures", Key="scene/data", Body=io.BytesIO(payload), ContentLength=len(payload))
    assert store.head_object(Bucket="fixtures", Key="scene/data")["ContentLength"] == len(payload)
    with store.get_object(Bucket="fixtures", Key="scene/data")["Body"] as stream:
        assert hashlib.sha256(stream.read()).digest() == hashlib.sha256(payload).digest()
    assert store.uploaded_bytes == len(payload)
    assert store.read_bytes == len(payload)
    with pytest.raises(FileExistsError):
        store.put_object(Bucket="fixtures", Key="scene/data", Body=io.BytesIO(b"changed"), ContentLength=7)
    assert (tmp_path / "fixtures/scene/data").read_bytes() == payload


def test_fixture_transport_refuses_escape_symlinks_and_bad_lengths(tmp_path):
    objects = tmp_path / "objects"
    objects.mkdir()
    outside = tmp_path / "keep"
    outside.mkdir()
    (objects / "linked").symlink_to(outside, target_is_directory=True)
    store = FilesystemObjectStore(objects)
    for bucket, key in (("../keep", "data"), ("fixtures", "../keep/data"),
                        ("fixtures", "/data"), ("linked", "data")):
        with pytest.raises(ValueError, match="fixture_object_path_unsafe"):
            store.put_object(Bucket=bucket, Key=key, Body=io.BytesIO(b"data"), ContentLength=4)
    with pytest.raises(ValueError, match="fixture_object_size_mismatch"):
        store.put_object(Bucket="fixtures", Key="bad", Body=io.BytesIO(b"data"), ContentLength=3)
    assert not (objects / "fixtures/bad").exists()
    assert not list(outside.iterdir())


def test_fixture_multipart_retains_remote_bytes_and_binds_every_part(tmp_path):
    store = FilesystemObjectStore(tmp_path)
    upload = store.create_multipart_upload(Bucket="fixtures", Key="large")
    parts = []
    for number, payload in enumerate((b"a" * 100, b"b" * 100), 1):
        row = store.upload_part(Bucket="fixtures", Key="large", UploadId=upload["UploadId"],
                                PartNumber=number, Body=payload)
        parts.append({"PartNumber": number, "ETag": row["ETag"]})
    with pytest.raises(ValueError, match="fixture_multipart_part_mismatch"):
        store.complete_multipart_upload(Bucket="fixtures", Key="large", UploadId=upload["UploadId"],
                                        MultipartUpload={"Parts": [{**parts[0], "ETag": "wrong"}, parts[1]]})
    store.complete_multipart_upload(Bucket="fixtures", Key="large", UploadId=upload["UploadId"],
                                    MultipartUpload={"Parts": parts})
    assert (tmp_path / "fixtures/large").read_bytes() == b"a" * 100 + b"b" * 100
    assert store.uploaded_bytes == 200


def test_real_cas_publication_validates_fixture_transport_bytes(tmp_path):
    from blueprint_pipeline.task_evaluation_configured_scene_object_store import publish_configured_scene_stream
    store = FilesystemObjectStore(tmp_path)
    payload = b"actual closed fixture CAS input" * 100
    record = publish_configured_scene_stream(write_stream=lambda stream: stream.write(payload),
        digest="sha256:" + hashlib.sha256(payload).hexdigest(), size_bytes=len(payload),
        filename="fixture.zip", artifact_kind="provider-output", client=store, bucket="fixtures")
    assert record["status"] == "remote_verified"
    assert record["full_byte_service_account_readback_passed"] is True
    assert store.read_bytes == len(payload)
    key = record["uri"].split("s3://fixtures/", 1)[1]
    head = store.head_object(Bucket="fixtures", Key=key)
    with store.get_object(Bucket="fixtures", Key=key, Range="bytes=3-9", IfMatch=head["ETag"])["Body"] as body:
        assert body.read() == payload[3:10]


def test_multipart_total_limit_and_symlinked_control_root_fail_closed(tmp_path, monkeypatch):
    from scripts import control_plane_concurrency_fixture as module
    store = FilesystemObjectStore(tmp_path)
    monkeypatch.setattr(module, "OBJECT_LIMIT", 5)
    upload = store.create_multipart_upload(Bucket="fixtures", Key="too-large")
    part = store.upload_part(Bucket="fixtures", Key="too-large", UploadId=upload["UploadId"],
                              PartNumber=1, Body=b"sixxxx")
    with pytest.raises(ValueError, match="fixture_object_size_invalid"):
        store.complete_multipart_upload(Bucket="fixtures", Key="too-large", UploadId=upload["UploadId"],
            MultipartUpload={"Parts": [{"PartNumber": 1, "ETag": part["ETag"]}]})
    assert not (tmp_path / "fixtures/too-large").exists()
    outside = tmp_path / "outside"
    outside.mkdir()
    store.abort_multipart_upload(Bucket="fixtures", Key="too-large", UploadId=upload["UploadId"])
    (tmp_path / ".multipart").rmdir()
    (tmp_path / ".multipart").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="fixture_multipart_binding_invalid"):
        store.create_multipart_upload(Bucket="fixtures", Key="escape")
    assert not list(outside.iterdir())
