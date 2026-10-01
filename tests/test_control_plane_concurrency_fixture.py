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
