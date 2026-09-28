# Covers (for impacted-test selection):
#   src/blueprint_pipeline/provider_output_promotion.py
#   src/blueprint_pipeline/provider_output_promotion_records.py
#   src/blueprint_pipeline/provider_output_member_index.py
#   src/blueprint_pipeline/wam_provider_object_store.py
#   src/blueprint_pipeline/task_evaluation_configured_scene_object_store.py
#   tests/provider_output_fixtures.py
"""A staged provider output reaches B2, read back in full, before any cleanup may delete it."""

from __future__ import annotations

import errno
import fcntl
import functools
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

from blueprint_pipeline import provider_output_promotion as promotion
from blueprint_pipeline import provider_output_promotion_records as records
from blueprint_pipeline import task_evaluation_configured_scene_object_store as scene_store
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_paired_witness_staging import SUFFIX
from blueprint_pipeline.provider_output_member_index import validate_member_index
from blueprint_pipeline.wam_provider_object_store import (
    SCHEMA_VERSION,
    STAGING_MANIFEST_FILENAME,
    cleanup_staged_wam_provider_objects,
    presign_staged_object_get,
)
from tests.provider_output_fixtures import (
    SECRET,
    Entry,
    RangeStore,
    VirtualCasClient,
    Zeros,
    build_zip,
    quick10_shaped_archive,
    virtual_sha256,
)

SMALL = {"cells": 2, "frames_per_camera": 2, "png_bytes": 8 * 1024, "mp4_bytes": 64 * 1024,
         "policy_request_bytes": 4 * 1024}
MAXIMUM = 64 * 1024**2
KIND = "policy-canary-provider-output"
MANIFEST_MEMBER = "paired_witness_manifest.v1.json"
CAS_PREFIX = "s3://blueprint-artifacts/blueprint/arm-decision-proof-v1/configured-scenes/artifacts/"


class StagedSpaces:
    """The staging store: presigned GETs served by range, plus boto3 HEAD, DELETE and presign."""

    BUCKET = "staging-bucket"

    def __init__(self):
        self.stores: dict[str, RangeStore] = {}
        self.history: dict[str, list[RangeStore]] = {}
        self.deleted: list[str] = []

    def url(self, key):
        return f"https://spaces.example.invalid/{self.BUCKET}/{key}?X-Amz-Signature={SECRET}"

    def put(self, key, data, etag):
        store = RangeStore(data, etag=etag, url=self.url(key))
        self.stores[key] = store
        self.history.setdefault(key, []).append(store)
        return store

    def whole_object_gets(self, key):
        return sum(store.whole_object_gets() for store in self.history.get(key, []))

    def requests(self, key):
        return [row for store in self.history.get(key, []) for row in store.requests]

    def opener(self, request, timeout, policy):
        store = self.stores.get(urlparse(request.full_url).path.split("/", 2)[2])
        if store is None:
            raise urllib.error.HTTPError("redacted", 404, "Not Found", {}, None)
        return store.opener(request, timeout, policy)

    def generate_presigned_url(self, operation, *, Params, ExpiresIn, HttpMethod):
        assert (operation, HttpMethod, Params["Bucket"]) == ("get_object", "GET", self.BUCKET)
        return self.url(Params["Key"])

    def head_object(self, *, Bucket, Key):
        store = self.stores.get(Key)
        if store is None:
            error = RuntimeError("not found")
            error.response = {"ResponseMetadata": {"HTTPStatusCode": 404}, "Error": {"Code": "NoSuchKey"}}
            raise error
        return {"ContentLength": store.object.size, "ETag": store.etag}

    def delete_object(self, *, Bucket, Key):
        self.deleted.append(Key)
        self.stores.pop(Key, None)
        return {"ResponseMetadata": {"HTTPStatusCode": 204}}


class World:
    """One Quick-10 attempt: a gated staging dir, Spaces, B2 and the lane's layout."""

    def __init__(self, tmp_path: Path, monkeypatch, *, witness: bool = True):
        self.tmp_path = tmp_path
        tmp_path.mkdir(parents=True, exist_ok=True)
        # The dedicated B2 store is explicitly configured (review I4).
        for name in scene_store._ARTIFACT_STORE_FILE_ENV.values():
            monkeypatch.setenv(name, str(tmp_path / "b2-configured-by-file"))
        for name, value in (("ACCESS_KEY_ID", "access"), ("SECRET_ACCESS_KEY", "secret"),
                            ("BUCKET", StagedSpaces.BUCKET)):
            path = tmp_path / f"spaces-{name.lower()}"
            path.write_text(value + "\n", encoding="utf-8")
            monkeypatch.setenv(f"BLUEPRINT_WAM_OBJECT_STORE_{name}_FILE", str(path))
        monkeypatch.setenv("BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL", "https://spaces.example.invalid")
        monkeypatch.setenv("BLUEPRINT_WAM_OBJECT_STORE_REGION", "nyc3")
        self.spaces = StagedSpaces()
        monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda _service, **_kwargs: self.spaces))
        monkeypatch.setitem(sys.modules, "botocore", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "botocore.client", SimpleNamespace(Config=lambda **_kwargs: object()))
        monkeypatch.setattr(promotion, "RETRY_SLEEP_SECONDS", 0.0)
        monkeypatch.setattr(promotion, "LOCK_POLL_SECONDS", 0.01)
        self.cas = VirtualCasClient()
        self.attempt = tmp_path / "attempt_001"
        self.staging = self.attempt / "object_store_staging"
        self.run = self.attempt / "vast_provider_run"
        self.staging.mkdir(parents=True)
        self.run.mkdir()
        self.keys = {"bundle": "blueprint/task/job/bundles/sha256/" + "a" * 64 + ".zip",
                     "output": "blueprint/task/job/runpod_provider_runtime_output_" + "0" * 32 + ".zip"}
        manifest = {"schema_version": SCHEMA_VERSION, "status": "completed",
                    "object_store": {"key_prefix": "blueprint/task"}, "bundle_key": self.keys["bundle"],
                    "output_key": self.keys["output"], "output_promotion_required": True}
        if witness:
            self.keys["paired_witness"] = self.keys["output"] + SUFFIX
            manifest["paired_witness"] = {"status": "ready", "witness_key": self.keys["paired_witness"],
                                          "authority": {"maximum_archive_bytes": MAXIMUM}}
        (self.staging / STAGING_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.spaces.put(self.keys["bundle"], b"provider bundle bytes", '"bundle"')

    def stage(self, role, archive, etag='"spaces-1"'):
        self.cas.register(archive)
        return self.spaces.put(self.keys[role], archive, etag)

    def dependencies(self):
        return {"publisher": functools.partial(scene_store.publish_configured_scene_stream, client=self.cas,
                                               bucket=self.cas.bucket),
                "file_publisher": functools.partial(scene_store.publish_configured_scene_artifact,
                                                    client=self.cas, bucket=self.cas.bucket),
                "opener": self.spaces.opener}

    def cleanup(self):
        return cleanup_staged_wam_provider_objects(self.staging)

    def promote(self, *, observation=None, local_archive=None, cleanup=None, **options):
        arguments = {"artifact_kind": KIND, "maximum_archive_bytes": MAXIMUM, **self.dependencies(), **options}
        return promotion.promote_then_cleanup(
            cleanup=cleanup or self.cleanup, staging_dir=self.staging, attempt_root=self.attempt,
            observation=observation, local_archive=local_archive, **arguments)

    def resume(self, **options):
        return promotion.resume_provider_output_promotion(
            self.attempt, maximum_archive_bytes=MAXIMUM, **{**self.dependencies(), **options})

    def files(self):
        return sorted(path for path in self.tmp_path.rglob("*") if path.is_file())


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch, witness=False)


@pytest.fixture
def paired_world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch, witness=True)


def _observed(archive, etag='"spaces-1"'):
    return {"size_bytes": archive.size, "etag": etag}


def _cas_uri(kind, digest, filename):
    return f"{CAS_PREFIX}{kind}/sha256/{digest.removeprefix('sha256:')}/{filename}"


def _proof(world):
    return records.load_staged_object_absence_proof(world.staging)


def test_promotion_indexes_then_copies_the_pinned_object_to_cas_with_full_readback(world):
    archive = quick10_shaped_archive(**SMALL).archive
    world.stage("output", archive)

    receipt, cleanup = world.promote(observation=_observed(archive))

    digest = virtual_sha256(archive)
    assert (receipt["status"], receipt["blockers"], receipt["source"]) == ("promoted", [], "remote_observation")
    reference = receipt["durable_reference"]
    assert reference["uri"] == _cas_uri(KIND, digest, "vast_provider_runtime_output.zip")
    assert (reference["digest"], reference["size_bytes"]) == (digest, archive.size) == (
        receipt["archive_sha256"], receipt["size_bytes"])
    # Two whole-object Spaces GETs (the index pass, then the copy), both pinned.
    assert world.spaces.whole_object_gets(world.keys["output"]) == 2
    assert all(row["if_match"] == '"spaces-1"' for row in world.spaces.requests(world.keys["output"])[1:])
    # One B2 upload streamed in parts, then one readback of every byte.
    assert world.cas.uploads == 1 and not [call for call in world.cas.calls if call[0] == "upload_file"]
    assert world.cas.readback_bytes == archive.size
    # The index is sealed with the durable reference and kept as the pointer.
    index_path = world.attempt / promotion.INDEX_FILENAME
    index = validate_member_index(json.loads(index_path.read_text(encoding="utf-8")))
    assert index["archive"]["durable_reference"]["uri"] == reference["uri"]
    assert (index["archive"]["sha256"], index["archive"]["etag"]) == (digest, '"spaces-1"')
    assert receipt["member_index"] == {"path": promotion.INDEX_FILENAME, "index_digest": index["index_digest"],
                                       "sha256": "sha256:" + hashlib.sha256(index_path.read_bytes()).hexdigest()}
    assert receipt["staged_objects"]["output"] == {
        "key_sha256": records.key_sha256(world.keys["output"]), "state": "promoted",
        "versions": [{"size_bytes": archive.size, "etag": '"spaces-1"', "archive_sha256": digest,
                      "durable_uri": reference["uri"]}]}
    # Only then did cleanup delete the staged output, and the absence is proven.
    assert cleanup["status"] == "completed" and cleanup["all_objects_absent"] is True
    assert world.keys["output"] in world.spaces.deleted
    assert _proof(world)["promotion_receipt_digest"] == receipt["receipt_digest"]
    assert records.load_promotion_receipt(
        world.staging, staging_manifest_sha256=records.staging_manifest_sha256(world.staging)) == receipt
    # No local archive at any point.
    assert not [path for path in world.files() if path.suffix == ".zip"]
    assert all(path.stat().st_size < archive.size // 4 for path in world.files())


@pytest.mark.parametrize("change", [{"etag": '"spaces-2"'}, {"size_bytes": 1}])
def test_promotion_refuses_an_object_whose_etag_changed_after_observation(world, change):
    archive = quick10_shaped_archive(**SMALL).archive
    world.stage("output", archive, etag='"spaces-2"' if "etag" in change else '"spaces-1"')

    receipt, cleanup = world.promote(observation={**_observed(archive), **change} if "size_bytes" in change
                                     else _observed(archive))

    assert (receipt["status"], receipt["blockers"]) == ("failed", ["provider_output_remote_version_changed"])
    assert receipt["attempts"] == {"output": 1}  # an identity refusal is never retried
    assert receipt["staged_objects"]["output"]["state"] == "failed"
    assert receipt["staged_objects"]["output"]["versions"] == []
    assert world.cas.uploads == 0 and world.spaces.whole_object_gets(world.keys["output"]) == 0
    # The only copy stays staged; cleanup defers it and nothing claims absence.
    assert world.keys["output"] in world.spaces.stores and world.keys["output"] not in world.spaces.deleted
    assert cleanup["all_objects_absent"] is False
    assert cleanup["blockers"] == ["staged_output_promotion_receipt_missing"]
    assert _proof(world) is None and not (world.attempt / promotion.INDEX_FILENAME).exists()


def _local_zip(world, archive):
    path = world.run / "vast_provider_runtime_output.zip"
    path.write_bytes(archive.to_bytes())
    return path


def test_ssh_recovered_zip_is_published_indexed_then_removed_behind_its_pointer(world):
    archive = quick10_shaped_archive(**SMALL).archive
    local = _local_zip(world, archive)

    receipt, cleanup = world.promote(local_archive=local)

    digest = virtual_sha256(archive)
    assert (receipt["status"], receipt["source"], receipt["blockers"]) == ("promoted", "ssh_local_zip", [])
    assert receipt["local_copy_removed_after_verified_promotion"] is True and not local.exists()
    assert [call[0] for call in world.cas.calls].count("upload_file") == 1
    assert world.cas.readback_bytes == archive.size
    assert receipt["durable_reference"]["uri"] == _cas_uri(KIND, digest, "vast_provider_runtime_output.zip")
    index = validate_member_index(json.loads((world.attempt / promotion.INDEX_FILENAME).read_text()))
    assert index["archive"]["durable_reference"]["digest"] == digest and index["archive"]["etag"] is None
    # Nothing was staged in Spaces, so no staged version is recorded.
    assert receipt["staged_objects"]["output"] == {
        "key_sha256": records.key_sha256(world.keys["output"]), "state": "promoted", "versions": []}
    assert cleanup["all_objects_absent"] is True and _proof(world) is not None
    # A resume reuses that receipt and still says the local copy went behind its pointer.
    assert world.resume()["status"] == "completed"
    again = records.load_promotion_receipt(
        world.staging, staging_manifest_sha256=records.staging_manifest_sha256(world.staging))
    assert again["local_copy_removed_after_verified_promotion"] is True
    assert again["durable_reference"] == receipt["durable_reference"]


def test_a_local_zip_over_the_archive_bound_is_refused_and_kept(world):
    archive = quick10_shaped_archive(**SMALL).archive
    local = _local_zip(world, archive)

    receipt, _ = world.promote(local_archive=local, maximum_archive_bytes=archive.size - 1)

    assert (receipt["status"], receipt["blockers"]) == ("failed", ["provider_output_archive_size_invalid"])
    assert local.is_file() and world.cas.uploads == 0


def test_a_local_zip_whose_publication_disagrees_is_kept(world):
    archive = quick10_shaped_archive(**SMALL).archive
    local = _local_zip(world, archive)
    other = "sha256:" + "0" * 64

    def lying_publisher(*, path, artifact_kind):
        reference = world.dependencies()["file_publisher"](path=path, artifact_kind=artifact_kind)
        return {**reference, "digest": other, "uri": _cas_uri(artifact_kind, other, Path(path).name)}

    receipt, cleanup = world.promote(local_archive=local, file_publisher=lying_publisher)

    assert (receipt["status"], receipt["blockers"]) == ("failed", ["provider_output_local_archive_digest_mismatch"])
    assert local.is_file() and receipt["local_copy_removed_after_verified_promotion"] is False
    assert not (world.attempt / promotion.INDEX_FILENAME).exists()


def _paired(case: str = "redundant"):
    """An output archive and its paired witness: the witness is cell 00 plus a manifest."""
    base = quick10_shaped_archive(**SMALL)
    cell = {name.removeprefix("cell_runs/00/"): data for name, data in base.payloads.items()
            if name.startswith("cell_runs/00/")}
    manifest = {"schema_version": "policy_canary_paired_witness.v1", "run_id": "run-1",
                "source": "retained_first_quick10_cell", "files": sorted(cell), "manifest_digest": ""}
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    output = quick10_shaped_archive(**SMALL, extra_members={MANIFEST_MEMBER: json.dumps(manifest, indent=2).encode()})
    witness_manifest = {**manifest, "run_id": "run-2"} if case == "manifest_differs" else manifest
    members = {MANIFEST_MEMBER: json.dumps(witness_manifest, sort_keys=True).encode(), **cell}
    if case == "extra_member":
        members["episodes/unarchived_large_review.mp4"] = b"\1" * 4096
    if case == "member_differs":
        name = next(name for name in cell if name.endswith(".score_receipt.json"))
        members[name] = b'{"episode_id": "tampered"}'
    return output.archive, build_zip([Entry(name, data) for name, data in members.items()]), len(members)


@pytest.mark.parametrize("case", ["redundant", "extra_member", "manifest_differs", "member_differs"])
def test_witness_is_redundant_only_when_every_row_is_in_cell_00(paired_world, case):
    world = paired_world
    output, witness, witness_members = _paired(case)
    world.stage("output", output)
    world.stage("paired_witness", witness, etag='"witness-1"')

    receipt, cleanup = world.promote(observation=_observed(output))

    section = receipt["staged_objects"]["paired_witness"]
    (version,) = section["versions"]
    assert (version["size_bytes"], version["etag"]) == (witness.size, '"witness-1"')
    assert version["archive_sha256"] == virtual_sha256(witness)
    if case == "redundant":
        assert receipt["witness"]["disposition"] == section["state"] == "redundant_with_promoted_output"
        assert version["durable_uri"] is None and world.cas.uploads == 1  # only the output
        assert version["redundancy"] == {
            "status": "proven", "rule": "witness_rows_subset_of_output_cell_00_rows",
            "cell_prefix": "cell_runs/00/", "witness_member_count": witness_members,
            "witness_index_digest": version["redundancy"]["witness_index_digest"],
            "output_archive_sha256": receipt["archive_sha256"],
            "output_index_digest": receipt["member_index"]["index_digest"]}
    else:
        assert receipt["witness"]["disposition"] == section["state"] == "promoted"
        assert version["redundancy"]["status"] == "not_proven"
        assert version["durable_uri"] == _cas_uri(
            "policy-canary-paired-witness", version["archive_sha256"], "policy_canary_paired_witness.zip")
        assert world.cas.uploads == 2
    assert cleanup["all_objects_absent"] is True
    assert world.keys["paired_witness"] in world.spaces.deleted


def test_unindexable_archive_is_still_promoted_then_blocked(paired_world):
    world = paired_world
    # Larger than the 64 KiB end-record window, so the index reads its tail by range.
    bad = build_zip([Entry("native_task_arena_policy_canary_session_result.v1.json", b'{"status": "completed"}'),
                     Entry("frames/0001.png", Zeros(256 * 1024)), Entry("a\\b.json", b"{}")])
    _, witness, _ = _paired()
    world.stage("output", bad)
    world.stage("paired_witness", witness, etag='"witness-1"')

    receipt, cleanup = world.promote(observation=_observed(bad))

    assert receipt["status"] == "promoted" and receipt["member_index"] is None
    assert receipt["index_refusal"] == "provider_output_archive_path_invalid"
    assert receipt["blockers"] == ["provider_output_index_refused:provider_output_archive_path_invalid"]
    assert receipt["durable_reference"]["digest"] == virtual_sha256(bad)
    # The directory refused it, so the two whole GETs are one hash pass and one copy.
    assert world.spaces.whole_object_gets(world.keys["output"]) == 2
    assert not (world.attempt / promotion.INDEX_FILENAME).exists()
    # Never redundant with an archive that has no index: the witness is promoted itself.
    assert receipt["witness"]["disposition"] == "promoted" and world.cas.uploads == 2
    assert receipt["staged_objects"]["paired_witness"]["versions"][0]["redundancy"] == {
        "status": "not_proven", "reason": "promoted_output_has_no_member_index"}
    assert cleanup["all_objects_absent"] is True


def test_absent_output_promotes_a_present_witness_and_otherwise_confirms_absence(tmp_path, monkeypatch):
    world = World(tmp_path / "witness_only", monkeypatch)
    _, witness, _ = _paired()
    world.stage("paired_witness", witness, etag='"witness-1"')

    receipt, cleanup = world.promote()

    assert (receipt["status"], receipt["source"], receipt["durable_reference"]) == ("absent_confirmed", "none", None)
    assert receipt["staged_objects"]["output"] == {
        "key_sha256": records.key_sha256(world.keys["output"]), "state": "absent_confirmed", "versions": []}
    # The witness may be the only paid evidence: it is promoted, never dropped.
    assert receipt["witness"]["disposition"] == "promoted"
    assert receipt["witness"]["reference"] == _cas_uri(
        "policy-canary-paired-witness", virtual_sha256(witness), "policy_canary_paired_witness.zip")
    assert cleanup["all_objects_absent"] is True and world.keys["paired_witness"] in world.spaces.deleted

    empty = World(tmp_path / "nothing_staged", monkeypatch)
    receipt, cleanup = empty.promote()
    assert receipt["status"] == "absent_confirmed" and receipt["blockers"] == []
    assert receipt["witness"] == {"disposition": "absent_confirmed", "reference": None, "redundancy": None}
    assert cleanup["all_objects_absent"] is True and empty.cas.uploads == 0
    assert _proof(empty)["promotion_status"] == "absent_confirmed"


def test_promotion_without_an_observation_promotes_whatever_is_present(world):
    archive = quick10_shaped_archive(**SMALL).archive
    world.stage("output", archive, etag='"spaces-9"')

    receipt, cleanup = world.promote()

    assert (receipt["status"], receipt["source"], receipt["observation"]) == ("promoted", "remote_present", None)
    assert receipt["staged_objects"]["output"]["versions"] == [
        {"size_bytes": archive.size, "etag": '"spaces-9"', "archive_sha256": virtual_sha256(archive),
         "durable_uri": receipt["durable_reference"]["uri"]}]
    assert cleanup["all_objects_absent"] is True


def test_ssh_recovery_while_the_staged_object_still_exists_is_gated_by_identity(tmp_path, monkeypatch):
    first = quick10_shaped_archive(**SMALL).archive
    second = quick10_shaped_archive(**{**SMALL, "cells": 3}).archive

    # The collector refused, SSH recovered the same bytes, and Spaces still holds them.
    same = World(tmp_path / "same", monkeypatch, witness=False)
    same.stage("output", first)
    receipt, cleanup = same.promote(local_archive=_local_zip(same, first))
    (version,) = receipt["staged_objects"]["output"]["versions"]
    assert receipt["source"] == "ssh_local_zip"
    assert (version["archive_sha256"], version["durable_uri"]) == (virtual_sha256(first), receipt["durable_reference"]["uri"])
    assert same.cas.uploads == 1 and same.spaces.whole_object_gets(same.keys["output"]) == 1  # hashed, not copied
    assert cleanup["all_objects_absent"] is True

    # Different bytes in Spaces are made durable too, under their own digest.
    other = World(tmp_path / "other", monkeypatch, witness=False)
    other.stage("output", second)
    receipt, cleanup = other.promote(local_archive=_local_zip(other, first))
    (version,) = receipt["staged_objects"]["output"]["versions"]
    assert version["archive_sha256"] == virtual_sha256(second) != receipt["archive_sha256"]
    assert version["durable_uri"] == _cas_uri(KIND, virtual_sha256(second), "vast_provider_runtime_output.zip")
    assert other.cas.uploads == 2 and cleanup["all_objects_absent"] is True

    # An object replaced after promotion is never deleted until it is promoted too.
    late = World(tmp_path / "late", monkeypatch, witness=False)
    late.stage("output", first)
    late.cas.register(second)

    def replaced_then_cleaned():
        late.spaces.put(late.keys["output"], second, '"spaces-2"')
        return late.cleanup()

    receipt, cleanup = late.promote(local_archive=_local_zip(late, first), cleanup=replaced_then_cleaned)
    assert cleanup["blockers"] == ["staged_output_promotion_identity_mismatch"]
    assert late.keys["output"] in late.spaces.stores and _proof(late) is None
    resumed = late.resume()
    assert resumed["status"] == "completed" and late.keys["output"] not in late.spaces.stores
    versions = records.load_promotion_receipt(
        late.staging, staging_manifest_sha256=records.staging_manifest_sha256(late.staging))[
        "staged_objects"]["output"]["versions"]
    assert [row["etag"] for row in versions] == ['"spaces-1"', '"spaces-2"']


def test_late_upload_after_absent_confirmation_is_never_deleted_until_promoted(world):
    archive = quick10_shaped_archive(**SMALL).archive

    def late_upload_then_cleanup():
        world.stage("output", archive, etag='"late"')
        return world.cleanup()

    receipt, cleanup = world.promote(cleanup=late_upload_then_cleanup)

    assert receipt["status"] == "absent_confirmed"
    assert cleanup["blockers"] == ["staged_output_promotion_receipt_missing"]
    assert world.keys["output"] in world.spaces.stores and _proof(world) is None
    resumed = world.resume()
    assert resumed["status"] == "completed" and resumed["blockers"] == []
    assert resumed["promotion"]["status"] == "promoted" and resumed["promotion"]["source"] == "remote_present"
    assert world.keys["output"] not in world.spaces.stores and _proof(world) is not None


def test_promotion_resume_is_idempotent_and_records_no_url(paired_world):
    world = paired_world
    output, witness, _ = _paired()
    world.stage("output", output)
    world.stage("paired_witness", witness, etag='"witness-1"')
    observation = _observed(output)
    (world.run / "vast_provider_command_result.json").write_text(
        json.dumps({"provider_output_remote_observation": observation}), encoding="utf-8")
    lane_result = world.attempt / "adp_arena_vast_result.json"
    lane_result.write_text('{"status": "blocked", "all_staged_objects_absent": false}\n', encoding="utf-8")

    def interrupted():
        raise ConnectionError("cleanup interrupted")

    first, cleanup = world.promote(observation=observation, cleanup=interrupted)
    assert first["status"] == "promoted" and cleanup["status"] == "blocked"
    assert cleanup["blockers"] == ["staged_object_cleanup_failed:ConnectionError"] and _proof(world) is None
    reads, uploads = world.spaces.whole_object_gets(world.keys["output"]), world.cas.uploads

    resumed = world.resume()

    assert resumed["status"] == "completed" and resumed["blockers"] == []
    receipt = records.load_promotion_receipt(
        world.staging, staging_manifest_sha256=records.staging_manifest_sha256(world.staging))
    for field in ("archive_sha256", "durable_reference", "member_index", "staged_objects", "source"):
        assert receipt[field] == first[field], field
    # The prior receipt matched what is staged: nothing was read or uploaded again.
    assert world.spaces.whole_object_gets(world.keys["output"]) == reads and world.cas.uploads == uploads
    proof = _proof(world)
    assert resumed["absence_proof"] == {"path": records.ABSENCE_PROOF_FILENAME, "proof_digest": proof["proof_digest"]}
    assert resumed["lane_result_rewritten"] is False
    assert lane_result.read_text() == '{"status": "blocked", "all_staged_objects_absent": false}\n'
    written = json.loads((world.attempt / promotion.RESUME_FILENAME).read_text())
    assert written == resumed and written["receipt_digest"] == canonical_digest(written, digest_field="receipt_digest")

    # With everything already gone, another resume is the same answer.
    again = world.resume()
    assert again["status"] == "completed" and _proof(world) == proof
    assert world.spaces.whole_object_gets(world.keys["output"]) == reads and world.cas.uploads == uploads
    for path in world.files():
        data = path.read_bytes()
        assert SECRET.encode() not in data and b"X-Amz" not in data and b"https://" not in data, path


def test_promotion_refuses_without_an_explicit_artifact_store(world, monkeypatch):
    archive = quick10_shaped_archive(**SMALL).archive
    world.stage("output", archive)
    # Only the WAM Spaces credentials remain, which the artifact client would
    # otherwise borrow silently.
    for name in scene_store._ARTIFACT_STORE_FILE_ENV.values():
        monkeypatch.delenv(name)
    monkeypatch.setattr(scene_store, "_artifact_object_store_client",
                        lambda: pytest.fail("no fallback artifact client may be built"))
    defaults = {"publisher": scene_store.publish_configured_scene_stream,
                "file_publisher": scene_store.publish_configured_scene_artifact}

    receipt, cleanup = world.promote(observation=_observed(archive), **defaults)

    assert (receipt["status"], receipt["blockers"]) == (
        "failed", ["provider_output_promotion_artifact_store_not_configured"])
    assert world.spaces.requests(world.keys["output"]) == []  # refused before anything was read
    assert cleanup["blockers"] == ["staged_output_promotion_receipt_missing"]
    assert world.keys["output"] in world.spaces.stores

    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_BUCKET_FILE", str(world.tmp_path / "b2-bucket"))
    receipt, _ = world.promote(observation=_observed(archive), **defaults)
    assert receipt["blockers"] == ["provider_output_promotion_artifact_store_not_configured"]


def test_two_promoters_serialize_on_the_staging_lock(world):
    archive = quick10_shaped_archive(**SMALL).archive
    world.stage("output", archive)
    log: list[str] = []
    entered, release = threading.Event(), threading.Event()
    publish = world.dependencies()["publisher"]
    results: dict[str, tuple] = {}

    def slow_publisher(**kwargs):
        log.append("first:publish")
        entered.set()
        assert release.wait(10)
        return publish(**kwargs)

    def presign(name):
        def recorded(job_dir, **kwargs):
            log.append(f"{name}:presign")
            return presign_staged_object_get(job_dir, **kwargs)
        return recorded

    def cleanup(name):
        def recorded():
            log.append(f"{name}:cleanup")
            return world.cleanup()
        return recorded

    first = threading.Thread(target=lambda: results.update(first=world.promote(
        observation=_observed(archive), publisher=slow_publisher, presign_staged=presign("first"),
        cleanup=cleanup("first"))))
    second = threading.Thread(target=lambda: results.update(second=world.promote(
        observation=_observed(archive), presign_staged=presign("second"), cleanup=cleanup("second"))))
    first.start()
    assert entered.wait(10)
    second.start()
    time.sleep(0.3)
    assert "second:presign" not in log  # waiting on the staging lock
    release.set()
    first.join(10)
    second.join(10)

    assert log.index("first:cleanup") < log.index("second:presign")
    assert results["first"][0]["status"] == results["second"][0]["status"] == "promoted"
    # The second found the first's receipt: nothing was read or uploaded again.
    assert world.cas.uploads == 1 and world.spaces.whole_object_gets(world.keys["output"]) == 2
    assert results["second"][1]["all_objects_absent"] is True


def test_resume_cli_runs_resume_and_says_how_it_must_be_launched(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_resume(attempt_root, **kwargs):
        calls.append((attempt_root, kwargs))
        return {"status": "blocked", "blockers": ["staged_output_promotion_receipt_missing"],
                "promotion": {"status": "promoted"}, "cleanup": {"status": "blocked"}, "absence_proof": None}

    monkeypatch.setattr(promotion, "resume_provider_output_promotion", fake_resume)
    code = promotion.main(["resume", "--attempt-root", str(tmp_path), "--maximum-archive-bytes", "1024"])
    assert code == 1 and calls == [(tmp_path, {"artifact_kind": KIND, "maximum_archive_bytes": 1024,
                                               "lock_timeout_seconds": promotion.DEFAULT_LOCK_TIMEOUT_SECONDS})]
    assert json.loads(capsys.readouterr().out)["blockers"] == ["staged_output_promotion_receipt_missing"]
    # Review I5: the resume door must run it as the service user with a private umask.
    assert "``blueprint``" in promotion.main.__doc__ and "UMask=0077" in promotion.main.__doc__


def _ungated(world):
    path = world.staging / STAGING_MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("output_promotion_required")  # a download-mode staging manifest
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_promotion_and_resume_refuse_an_attempt_that_did_not_require_promotion(world):
    _ungated(world)
    archive = quick10_shaped_archive(**SMALL).archive
    world.stage("output", archive)
    local = _local_zip(world, archive)  # download mode's own ZIP, which SSH adoption still reads
    before = sorted(path.name for path in world.staging.iterdir())

    receipt = promotion.promote_staged_provider_output(
        staging_dir=world.staging, attempt_root=world.attempt, observation=None, local_archive=local,
        maximum_archive_bytes=MAXIMUM, **world.dependencies())
    resumed = world.resume()

    assert (receipt["status"], receipt["blockers"]) == ("failed", ["provider_output_promotion_not_required"])
    assert (resumed["status"], resumed["blockers"]) == ("blocked", ["provider_output_promotion_not_required"])
    assert resumed["cleanup"] is None and resumed["absence_proof"] is None
    # Nothing was read, published, unlinked or cleaned up.
    assert local.is_file() and world.cas.uploads == 0 and world.spaces.deleted == []
    assert world.spaces.requests(world.keys["output"]) == []
    assert sorted(path.name for path in world.staging.iterdir()) == before


@pytest.mark.parametrize("fault", ["symlinked_lock_file", "flock_unavailable"])
def test_an_unavailable_lock_still_runs_the_gated_cleanup(world, monkeypatch, fault):
    archive = quick10_shaped_archive(**SMALL).archive
    world.stage("output", archive)
    if fault == "symlinked_lock_file":  # refused under O_NOFOLLOW, even for root
        (world.staging / promotion.LOCK_FILENAME).symlink_to(world.tmp_path / "elsewhere.lock")
    else:
        def unavailable(descriptor, operation):
            raise OSError(errno.ENOLCK, "No locks available")

        monkeypatch.setattr(promotion.fcntl, "flock", unavailable)

    receipt, cleanup = world.promote(observation=_observed(archive))

    assert (receipt["status"], receipt["blockers"]) == ("failed", ["provider_output_promotion_lock_unavailable"])
    assert world.cas.uploads == 0 and world.spaces.requests(world.keys["output"]) == []
    # The gated cleanup still ran without the lock: the bundle went, the unpromoted output stayed.
    assert cleanup["blockers"] == ["staged_output_promotion_receipt_missing"]
    assert world.keys["bundle"] in world.spaces.deleted and world.keys["output"] in world.spaces.stores
    assert not (world.tmp_path / "elsewhere.lock").exists()


def test_a_held_lock_times_out_quickly_and_still_runs_the_gated_cleanup(world):
    archive = quick10_shaped_archive(**SMALL).archive
    world.stage("output", archive)
    assert promotion.DEFAULT_LOCK_TIMEOUT_SECONDS <= 600  # well inside the unit's 5 h timeout
    holder = os.open(world.staging / promotion.LOCK_FILENAME, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(holder, fcntl.LOCK_EX)
        receipt, cleanup = world.promote(observation=_observed(archive), lock_timeout_seconds=0.05)
        resumed = world.resume(lock_timeout_seconds=0.05)
    finally:
        os.close(holder)

    assert receipt["blockers"] == ["provider_output_promotion_lock_timeout"]
    assert cleanup["blockers"] == ["staged_output_promotion_receipt_missing"]
    assert "provider_output_promotion_lock_timeout" in resumed["blockers"]
    assert world.cas.uploads == 0 and world.keys["output"] in world.spaces.stores

