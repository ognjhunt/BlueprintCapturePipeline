from __future__ import annotations

import hashlib
import io
import json
import os
import pwd

import pytest

from blueprint_pipeline import task_evaluation_scene_configuration_submission_publication as publication
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_preparation_contract import launch_preparation_request_digest
from tests.test_task_evaluation_launch_preparation_contract import (
    test_configuration_request as configuration_request,
)

COMMIT = "a" * 40
ACCOUNT = pwd.getpwuid(os.geteuid()).pw_name
NAMESPACE = "fixture-production-inputs"
PREFIX = f"s3://blueprint/task-evaluation/production-inputs/{NAMESPACE}/"
RAW_URI = "https://huggingface.co/datasets/publisher/scene/resolve/" + "b" * 40 + "/source.ply"


class Store:
    def __init__(self):
        self.objects = {}
        self.puts = []
        self.uploads = {}
        self.parts = []
        self.aborts = []
        self.fail_part = False

    def head_object(self, *, Bucket, Key):
        return {"ContentLength": len(self.objects[Key])}

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, *, Bucket, Key, Body, ContentLength):
        assert not isinstance(Body, bytes), "Small upload must stream a file"
        if Key in self.objects:
            raise RuntimeError("precondition failed")
        data = Body.read()
        assert len(data) == ContentLength
        self.objects[Key] = data
        self.puts.append(Key)

    def create_multipart_upload(self, *, Bucket, Key):
        self.uploads[Key] = {}
        return {"UploadId": Key}

    def upload_part(self, *, Bucket, Key, UploadId, PartNumber, Body):
        assert len(Body) <= publication.PART_BYTES
        if self.fail_part:
            raise RuntimeError("part failed")
        self.uploads[Key][PartNumber] = Body
        self.parts.append((Key, PartNumber))
        return {"ETag": str(PartNumber)}

    def complete_multipart_upload(self, *, Bucket, Key, UploadId, MultipartUpload):
        assert Key not in self.objects
        self.objects[Key] = b"".join(self.uploads[Key][row["PartNumber"]]
                                     for row in MultipartUpload["Parts"])

    def abort_multipart_upload(self, *, Bucket, Key, UploadId):
        self.aborts.append(Key)


def _sha(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _seal_manifest(root, manifest):
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    (root / publication.MANIFEST).write_text(json.dumps(manifest))


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "staging"
    root.mkdir()
    rows = []
    def add(relative, data, *, uri=None, allowed=True):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        ref = {"uri": uri or PREFIX + relative, "digest": _sha(data), "size_bytes": len(data)}
        rows.append({"relative_path": relative, **ref, "publication_allowed": allowed})
        return ref
    request = configuration_request()
    request["publication"]["input_namespace"] = NAMESPACE
    def replace(node):
        if isinstance(node, dict):
            if set(node) == {"uri", "digest", "size_bytes"}:
                count = len(rows)
                node.update(add(f"references/{count}.json", f"fixture-{count}".encode()))
            else:
                for value in node.values():
                    replace(value)
        elif isinstance(node, list):
            for value in node:
                replace(value)
    replace(request)
    raw = add("source/appearance_3dgs/source.ply", b"RAW MUST NOT UPLOAD", uri=RAW_URI, allowed=False)
    request["scene"]["appearance"]["representation"] = raw
    publisher = {
        "source_uploaded_by_blueprint": False, "public_redistribution_allowed": False,
        "artifacts": [{"publisher_url": RAW_URI, "sha256": raw["digest"], "size_bytes": raw["size_bytes"]}],
    }
    add("provenance/publisher_intake.v1.json", json.dumps(publisher).encode())
    add(publication.REQUEST, json.dumps(request).encode())
    manifest = {
        "schema_version": "task_evaluation_scene_configuration_submission_manifest.v1",
        "status": "validated_pending_production_publication_and_submission",
        "source_commit": COMMIT, "input_namespace": NAMESPACE,
        "request_digest": launch_preparation_request_digest(request), "files": rows,
        "raw_source_upload_allowed": False, "provider_allocated": False,
    }
    _seal_manifest(root, manifest)
    monkeypatch.setattr(publication, "_verified_checkout_head", lambda: COMMIT)
    return root, manifest, Store(), tmp_path / "publication.json"


def _publish(bundle):
    root, _, store, receipt = bundle
    return publication.publish_scene_configuration_submission(
        manifest_path=root / publication.MANIFEST, receipt_path=receipt,
        expected_source_commit=COMMIT, service_account=ACCOUNT, client=store,
        lock_root=root.parent / "locks",
    )


def test_complete_publication_reads_back_every_admitted_file_never_raw_and_never_submits(bundle):
    result = _publish(bundle)
    root, manifest, store, _ = bundle
    assert result["status"] == "published_and_read_back"
    assert result["raw_source_uploaded"] is False and result["run_submitted"] is False
    assert result["global_atomic_create_claimed"] is False
    assert result["single_writer_scope"] == "production_host_namespace_flock"
    assert len(result["published_objects"]) == len(manifest["files"])
    assert len(result["host_only_source_objects"]) == 1
    assert not any("/source/" in key for key in store.objects)
    assert b"RAW MUST NOT UPLOAD" not in store.objects.values()
    before = list(store.puts)
    assert _publish(bundle) == result
    assert store.puts == before


def test_streaming_multipart_has_full_readback_without_conditional_headers(bundle, monkeypatch):
    monkeypatch.setattr(publication, "PART_BYTES", 256)
    _publish(bundle)
    assert bundle[2].parts
    assert not bundle[2].aborts


def test_partial_multipart_failure_aborts_without_success_receipt(bundle, monkeypatch):
    monkeypatch.setattr(publication, "PART_BYTES", 256)
    bundle[2].fail_part = True
    with pytest.raises(publication.SubmissionPublicationError, match="upload_failed"):
        _publish(bundle)
    assert bundle[2].aborts
    assert not bundle[3].exists()


@pytest.mark.parametrize("mode", ["flag", "renamed_duplicate"])
def test_raw_source_cannot_be_promoted_by_editing_publication_flags(bundle, mode):
    root, manifest, store, _ = bundle
    row = next(row for row in manifest["files"] if not row["publication_allowed"])
    if mode == "flag":
        row["publication_allowed"] = True
        row["uri"] = PREFIX + row["relative_path"]
    else:
        relative = "renamed-source.ply"
        (root / relative).write_bytes((root / row["relative_path"]).read_bytes())
        manifest["files"].append({**row, "relative_path": relative,
                                  "uri": PREFIX + relative, "publication_allowed": True})
    _seal_manifest(root, manifest)
    with pytest.raises(publication.SubmissionPublicationError, match="raw_upload"):
        _publish(bundle)
    assert not store.puts


def test_all_local_bytes_are_checked_before_first_upload(bundle):
    root, manifest, store, _ = bundle
    (root / manifest["files"][0]["relative_path"]).write_bytes(b"tampered")
    with pytest.raises(publication.SubmissionPublicationError, match="local_readback"):
        _publish(bundle)
    assert not store.puts


def test_uninventoried_file_and_symlink_fail_closed(bundle):
    root, _, store, _ = bundle
    (root / "surprise").write_bytes(b"surprise")
    with pytest.raises(publication.SubmissionPublicationError, match="directory_inventory"):
        _publish(bundle)
    assert not store.puts


def test_existing_different_bytes_are_never_overwritten(bundle):
    _, manifest, store, _ = bundle
    row = manifest["files"][0]
    key = row["uri"].split("s3://blueprint/", 1)[1]
    store.objects[key] = b"x" * row["size_bytes"]
    with pytest.raises(publication.SubmissionPublicationError, match="existing_object_mismatch"):
        _publish(bundle)
    assert store.objects[key] == b"x" * row["size_bytes"]
    assert not store.puts


def test_manifest_tamper_and_execution_commit_mismatch_fail_before_upload(bundle, monkeypatch):
    monkeypatch.setattr(publication, "_verified_checkout_head", lambda: "c" * 40)
    with pytest.raises(publication.SubmissionPublicationError, match="execution_commit_mismatch"):
        _publish(bundle)
    assert not bundle[2].puts


def test_receipt_must_live_outside_immutable_staging_inventory(bundle):
    root, _, store, _ = bundle
    with pytest.raises(publication.SubmissionPublicationError, match="receipt_path_invalid"):
        publication.publish_scene_configuration_submission(
            manifest_path=root / publication.MANIFEST, receipt_path=root / "receipt.json",
            expected_source_commit=COMMIT, service_account=ACCOUNT, client=store,
        lock_root=root.parent / "locks",
        )
    assert not store.puts

def test_uninventoried_directory_symlink_fails_before_upload(bundle, tmp_path):
    root, _, store, _ = bundle
    (root / "hidden").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(publication.SubmissionPublicationError, match="symlink_forbidden"):
        _publish(bundle)
    assert not store.puts


def test_wrong_returned_bytes_cannot_produce_success_receipt(bundle, monkeypatch):
    original = bundle[2].put_object
    def corrupted_put(**kwargs):
        original(**kwargs)
        bundle[2].objects[kwargs["Key"]] = b"x" * kwargs["ContentLength"]
    monkeypatch.setattr(bundle[2], "put_object", corrupted_put)
    with pytest.raises(publication.SubmissionPublicationError, match="existing_object_mismatch"):
        _publish(bundle)
    assert not bundle[3].exists()

def test_duplicate_namespace_writer_is_rejected_before_object_store_mutation(bundle):
    root, _, store, _ = bundle
    with publication._namespace_lock(root.parent / "locks", NAMESPACE):
        with pytest.raises(publication.SubmissionPublicationError, match="namespace_writer_active"):
            _publish(bundle)
    assert not store.puts
    assert _publish(bundle)["status"] == "published_and_read_back"

def test_actual_assembler_inventory_publishes_and_replays_without_raw_source_upload(
    tmp_path, monkeypatch,
):
    from tests.test_task_evaluation_scene_configuration_submission import (
        SHA, _materialize, production_fixture,
    )

    fixture = production_fixture(tmp_path)
    assembled = _materialize(fixture)
    root = fixture["staging_root"]
    store = Store()
    monkeypatch.setattr(publication, "_verified_checkout_head", lambda: SHA)
    arguments = {
        "manifest_path": root / publication.MANIFEST,
        "receipt_path": fixture["root"] / "publication.v1.json",
        "expected_source_commit": SHA,
        "service_account": ACCOUNT,
        "client": store,
        "lock_root": fixture["root"] / "publication-locks",
    }
    result = publication.publish_scene_configuration_submission(**arguments)
    manifest = json.loads((root / publication.MANIFEST).read_text())
    assert result["request_digest"] == assembled["request_digest"]
    assert result["manifest_digest"] == assembled["manifest_digest"]
    assert len(result["host_only_source_objects"]) == 5
    expected_count = 1 + sum(row["publication_allowed"] for row in manifest["files"])
    assert len(result["published_objects"]) == expected_count
    for row in result["host_only_source_objects"]:
        assert row["uri"].startswith("https://huggingface.co/datasets/")
        assert row["publication_allowed"] is False
        assert (root / row["relative_path"]).read_bytes() not in store.objects.values()
    before = len(store.puts)
    assert publication.publish_scene_configuration_submission(**arguments) == result
    assert len(store.puts) == before
