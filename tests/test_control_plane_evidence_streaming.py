from __future__ import annotations

import functools
import hashlib

import pytest

from blueprint_pipeline.object_store_multipart_stream import MultipartStream
from blueprint_pipeline import control_plane_evidence_offload as offload
from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from tests.test_control_plane_evidence_offload import _run, _unclassified, BUCKET
from tests.test_task_evaluation_configured_scene_object_store import _ContentAddressedClient


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(offload, "DEFAULT_RESERVATION_ROOT", tmp_path / "ledger")


class MultipartClient(_ContentAddressedClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pending = {}
        self.aborted = 0
        self.maximum_part_bytes = 0

    def create_multipart_upload(self, *, Bucket, Key, Metadata, ContentType):
        self.pending["upload"] = {"bucket": Bucket, "key": Key, "metadata": Metadata, "parts": {}}
        return {"UploadId": "upload"}

    def upload_part(self, *, Bucket, Key, UploadId, PartNumber, Body):
        assert len(Body) <= 8 * 1024**2
        self.maximum_part_bytes = max(self.maximum_part_bytes, len(Body))
        self.pending[UploadId]["parts"][PartNumber] = bytes(Body)
        return {"ETag": str(PartNumber)}

    def complete_multipart_upload(self, *, Bucket, Key, UploadId, MultipartUpload):
        row = self.pending.pop(UploadId)
        self.objects[(Bucket, Key)] = b"".join(
            row["parts"][p["PartNumber"]] for p in MultipartUpload["Parts"]
        )
        self.metadata[(Bucket, Key)] = row["metadata"]
        self.upload_count += 1

    def abort_multipart_upload(self, **kwargs):
        self.pending.pop(kwargs["UploadId"])
        self.aborted += 1


def test_stream_offload_fits_small_metadata_budget_and_round_trips_real_materializer(
    tmp_path, monkeypatch
):
    root = tmp_path / "runs"
    root.mkdir()
    directory = _run(root, "run", receipt="launch_receipt.json", age=100, now=1000)
    frame = directory / "episodes/frame.bin"
    frame.write_bytes(b"x" * (18 * 1024**2))
    manifest = offload.build_evidence_offload_manifest(
        evidence_roots=[root], hot_window_seconds=0, now=lambda: 10**10, classifier=_unclassified
    )
    from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk

    def reserve(*args, **kwargs):
        assert kwargs["expected_bytes"] < 2 * 1024**2
        return reserve_control_plane_disk(
            *args, **{**kwargs, "reservation_root": tmp_path / "ledger"}
        )

    monkeypatch.setattr(offload, "reserve_control_plane_disk", reserve)
    original_mkstemp = offload.tempfile.mkstemp

    def no_archive(**kwargs):
        assert ".offload-" not in kwargs.get("prefix", ""), "local tar must not be created"
        return original_mkstemp(**kwargs)

    monkeypatch.setattr(offload.tempfile, "mkstemp", no_archive)
    client = MultipartClient()
    result = offload.apply_evidence_offload(
        manifest,
        ack=offload.EXECUTE_ACK,
        stream_publisher=functools.partial(
            store.publish_configured_scene_stream, client=client, bucket=BUCKET
        ),
        now=lambda: 10**10,
    )
    assert result["offloaded_count"] == 1 and result["skipped"] == []
    assert not directory.exists() and client.maximum_part_bytes == 8 * 1024**2
    pointer = root / ("run" + offload.POINTER_SUFFIX)
    # Exercise the real downloader rather than a fixture that ignores its reference contract.
    monkeypatch.undo()
    restored = offload.restore_offloaded_evidence(
        pointer_path=pointer,
        destination=tmp_path / "restored",
        materializer=functools.partial(
            store.materialize_configured_scene_artifact, client=client, bucket=BUCKET
        ),
    )
    assert restored["status"] == "restored"
    assert (tmp_path / "restored/episodes/frame.bin").read_bytes() == b"x" * (18 * 1024**2)


@pytest.mark.parametrize("failure", ["changed_stream", "writer_failure", "bad_readback"])
def test_stream_failure_never_evicts_evidence_or_finishes_bad_content(tmp_path, failure):
    root = tmp_path / "runs"
    root.mkdir()
    directory = _run(root, "run", receipt="launch_receipt.json", age=100, now=1000)
    manifest = offload.build_evidence_offload_manifest(
        evidence_roots=[root], hot_window_seconds=0, now=lambda: 1000, classifier=_unclassified
    )
    client = MultipartClient(corrupt_readback=failure == "bad_readback")

    def publisher(**kwargs):
        writer = kwargs["write_stream"]
        if failure == "changed_stream":
            kwargs["write_stream"] = lambda sink: sink.write(b"changed")
        elif failure == "writer_failure":

            def broken(sink):
                writer(sink)
                raise OSError("transfer interrupted")

            kwargs["write_stream"] = broken
        return store.publish_configured_scene_stream(**kwargs, client=client, bucket=BUCKET)

    result = offload.apply_evidence_offload(
        manifest, ack=offload.EXECUTE_ACK, stream_publisher=publisher
    )
    assert result["offloaded_count"] == 0 and directory.is_dir()
    assert not (root / ("run" + offload.POINTER_SUFFIX)).exists()
    assert not client.pending
    if failure != "bad_readback":
        assert client.aborted == 1 and client.upload_count == 0


def test_stream_cache_hit_does_not_upload_or_invoke_writer():
    client = MultipartClient()
    payload = b"known exact bytes"
    kwargs = {
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "filename": "evidence.tar",
        "artifact_kind": "control-plane-evidence",
        "client": client,
        "bucket": BUCKET,
    }
    first = store.publish_configured_scene_stream(
        write_stream=lambda sink: sink.write(payload), **kwargs
    )
    second = store.publish_configured_scene_stream(
        write_stream=lambda _: pytest.fail("cache should be reused"), **kwargs
    )
    assert first["digest"] == second["digest"] and second["cache_hit"] and client.upload_count == 1


def test_declared_large_stream_keeps_bounded_parts_and_rejects_unbounded_size():
    client = MultipartClient()
    sink = MultipartStream(
        client=client,
        bucket=BUCKET,
        key="bounded",
        metadata={"Metadata": {}, "ContentType": "application/octet-stream"},
        expected_digest="sha256:" + "0" * 64,
        expected_size=64 * 1024**3,
    )
    assert 8 * 1024**2 <= sink.part_size <= 64 * 1024**2
    sink.abort()
    with pytest.raises(ValueError, match="bounded_multipart_limit"):
        MultipartStream(
            client=client,
            bucket=BUCKET,
            key="too-large",
            metadata={},
            expected_digest="sha256:" + "0" * 64,
            expected_size=10 * 1024**4,
        )
    assert client.pending == {}
