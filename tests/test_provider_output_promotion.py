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

    def __init__(self, tmp_path: Path, monkeypatch, *, witness: bool = True, torn_down: bool = True):
        self.tmp_path = tmp_path
        tmp_path.mkdir(parents=True, exist_ok=True)
        # The dedicated B2 store is explicitly configured (review I4), by readable files (minor 8).
        for key, name in scene_store._ARTIFACT_STORE_FILE_ENV.items():
            (tmp_path / f"b2-{key}").write_text("configured-by-file\n", encoding="utf-8")
            monkeypatch.setenv(name, str(tmp_path / f"b2-{key}"))
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
        if torn_down:  # the paid window is over: the provider can no longer read or upload
            self.tear_down()

    def tear_down(self, *, continuing_spend=False):
        (self.run / promotion.TEARDOWN_MANIFEST_NAME).write_text(json.dumps(
            {"schema_version": "vast_teardown_manifest.v1", "status": "completed", "vast_instance_ids": [7],
             "continuing_spend_from_this_run": continuing_spend}), encoding="utf-8")

    def stage(self, role, archive, etag='"spaces-1"'):
        self.cas.register(archive)
        return self.spaces.put(self.keys[role], archive, etag)

    def dependencies(self):
        return {"publisher": functools.partial(scene_store.publish_configured_scene_stream, client=self.cas,
                                               bucket=self.cas.bucket),
                "file_publisher": functools.partial(scene_store.publish_configured_scene_artifact,
                                                    client=self.cas, bucket=self.cas.bucket),
                "verifier": functools.partial(scene_store.verify_configured_scene_artifact, client=self.cas,
                                              bucket=self.cas.bucket),
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
            self.attempt, **{"maximum_archive_bytes": MAXIMUM, **self.dependencies(), **options})

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
                      "durable_uri": reference["uri"], "durable_reference": reference}]}
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
                "source": "retained_first_quick10_cell", "files": sorted(cell),
                "new_learned_episodes_executed": 0, "manifest_digest": ""}
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    output = quick10_shaped_archive(**SMALL, extra_members={MANIFEST_MEMBER: json.dumps(manifest, indent=2).encode()})
    witness_manifest = {"manifest_differs": {**manifest, "run_id": "run-2"},
                        # Equal under Python's ==, different JSON: 0 and 0.0.
                        "manifest_type_differs": {**manifest, "new_learned_episodes_executed": 0.0},
                        }.get(case, manifest)
    members = {MANIFEST_MEMBER: json.dumps(witness_manifest, sort_keys=True).encode(), **cell}
    if case == "extra_member":
        members["episodes/unarchived_large_review.mp4"] = b"\1" * 4096
    if case == "member_differs":
        name = next(name for name in cell if name.endswith(".score_receipt.json"))
        members[name] = b'{"episode_id": "tampered"}'
    return output.archive, build_zip([Entry(name, data) for name, data in members.items()]), len(members)


@pytest.mark.parametrize("case", ["redundant", "extra_member", "manifest_differs", "manifest_type_differs",
                                  "member_differs"])
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
         "durable_uri": receipt["durable_reference"]["uri"], "durable_reference": receipt["durable_reference"]}]
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

    # With everything already gone, another resume is the same answer, and it rewrites nothing
    # the proof and the sealed artifact manifest bind (review minor 4).
    sealed = (world.staging / records.RECEIPT_FILENAME).read_bytes()
    again = world.resume()
    assert again["status"] == "completed" and _proof(world) == proof
    assert (world.staging / records.RECEIPT_FILENAME).read_bytes() == sealed
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


@pytest.mark.parametrize("teardown", ["missing", "continuing_spend", "unreadable", "symlinked"])
def test_resume_refuses_an_attempt_whose_paid_window_may_be_open(tmp_path, monkeypatch, teardown):
    """Review critical 1: resume runs the gated cleanup, which deletes the staged bundle and seals a
    write-once absence proof. While the paid window may be open the provider can still read the
    bundle or upload its output, so until the teardown manifest records no continuing spend
    nothing is read, published, removed or proven -- ingestion included."""
    world = World(tmp_path, monkeypatch, witness=True, torn_down=False)
    teardown_path = world.run / promotion.TEARDOWN_MANIFEST_NAME
    if teardown == "continuing_spend":
        world.tear_down(continuing_spend=True)
    elif teardown == "unreadable":
        teardown_path.write_text("{not json", encoding="utf-8")
    elif teardown == "symlinked":
        elsewhere = tmp_path / "teardown-elsewhere.json"
        elsewhere.write_text('{"continuing_spend_from_this_run": false}', encoding="utf-8")
        teardown_path.symlink_to(elsewhere)
    before = world.files()

    resumed = world.resume(ingest=True)

    assert (resumed["status"], resumed["blockers"]) == ("blocked", ["provider_output_resume_attempt_not_torn_down"])
    assert resumed["cleanup"] is None and resumed["absence_proof"] is None and "ingestion" not in resumed
    assert world.keys["bundle"] in world.spaces.stores and world.spaces.deleted == []
    assert world.cas.uploads == 0 and world.spaces.requests(world.keys["output"]) == []
    assert _proof(world) is None and not (world.staging / records.RECEIPT_FILENAME).exists()
    assert sorted(set(world.files()) - set(before)) == [world.attempt / promotion.RESUME_FILENAME]
    assert promotion.main(["resume", "--attempt-root", str(world.attempt)]) == 1

    teardown_path.unlink(missing_ok=True)
    world.tear_down()  # the adapter's own teardown manifest now records no continuing spend
    assert world.resume()["status"] == "completed"
    assert world.keys["bundle"] not in world.spaces.stores and _proof(world) is not None


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


class _Killed(BaseException):
    """The process dies: nothing after this point runs, not even exception handlers."""


def _kill(*_args, **_kwargs):
    raise _Killed()


def _receipt_now(world):
    return records.load_promotion_receipt(
        world.staging, staging_manifest_sha256=records.staging_manifest_sha256(world.staging))


def test_a_kill_during_the_witness_step_resumes_from_the_durable_output(paired_world, monkeypatch):
    world = paired_world
    output, witness, _ = _paired()
    world.stage("output", output)
    world.stage("paired_witness", witness, etag='"witness-1"')
    observation = _observed(output)
    (world.run / "vast_provider_command_result.json").write_text(
        json.dumps({"provider_output_remote_observation": observation}), encoding="utf-8")

    with monkeypatch.context() as patch:
        patch.setattr(promotion._Promotion, "_witness", _kill)
        with pytest.raises(_Killed):
            world.promote(observation=observation)

    # The output was durable before the witness step began, and the receipt already says so.
    interim = _receipt_now(world)
    assert interim["status"] == "promoted" and len(interim["staged_objects"]["output"]["versions"]) == 1
    assert interim["staged_objects"]["paired_witness"] == {
        "key_sha256": records.key_sha256(world.keys["paired_witness"]), "state": "pending", "versions": []}
    reads, uploads = world.spaces.whole_object_gets(world.keys["output"]), world.cas.uploads
    # A resume with the CLI's own default maximum reuses it: the output is not read again.
    resumed = world.resume(maximum_archive_bytes=promotion.DEFAULT_MAXIMUM_ARCHIVE_BYTES)
    assert resumed["status"] == "completed", resumed["blockers"]
    assert world.spaces.whole_object_gets(world.keys["output"]) == reads
    assert world.cas.uploads == uploads + 1  # the witness, promoted: this run never read the output
    assert _receipt_now(world)["witness"]["disposition"] == "promoted"


def test_an_output_failure_after_the_index_was_written_resumes_with_any_maximum(world):
    first = quick10_shaped_archive(**SMALL).archive
    second = quick10_shaped_archive(**{**SMALL, "cells": 3}).archive
    world.stage("output", second)  # Spaces holds other bytes than SSH recovered
    local = _local_zip(world, first)

    def b2_down(**_kwargs):
        raise RuntimeError("b2 down")

    receipt, cleanup = world.promote(local_archive=local, publisher=b2_down)

    # The recovered ZIP is durable and indexed; only the staged object's copy failed.
    assert receipt["status"] == "promoted" and receipt["source"] == "ssh_local_zip"
    assert receipt["blockers"] == ["provider_output_promotion_failed:RuntimeError"]
    assert receipt["staged_objects"]["output"]["versions"] == []
    assert not local.exists() and (world.attempt / promotion.INDEX_FILENAME).is_file()
    assert cleanup["blockers"] == ["staged_output_promotion_identity_mismatch"]
    index = (world.attempt / promotion.INDEX_FILENAME).read_bytes()

    resumed = world.resume(maximum_archive_bytes=promotion.DEFAULT_MAXIMUM_ARCHIVE_BYTES)

    assert resumed["status"] == "completed", resumed["blockers"]
    assert (world.attempt / promotion.INDEX_FILENAME).read_bytes() == index
    assert world.keys["output"] not in world.spaces.stores
    assert json.loads(index)["limits"] == promotion.INDEX_LIMITS  # the caller's maximum is not recorded


def test_index_limits_do_not_depend_on_the_caller_s_maximum(tmp_path, monkeypatch):
    archive = quick10_shaped_archive(**SMALL).archive
    indexes = []
    for name, maximum in (("lane", MAXIMUM), ("cli", promotion.DEFAULT_MAXIMUM_ARCHIVE_BYTES)):
        world = World(tmp_path / name, monkeypatch, witness=False)
        world.stage("output", archive)
        receipt, _ = world.promote(observation=_observed(archive), maximum_archive_bytes=maximum)
        assert receipt["status"] == "promoted"
        indexes.append((world.attempt / promotion.INDEX_FILENAME).read_bytes())
    assert indexes[0] == indexes[1]


def _drop_from_b2(world, uri):
    key = urlparse(uri).path.lstrip("/")
    del world.cas.objects[key]


def test_a_reused_receipt_is_trusted_only_while_its_durable_copies_exist(tmp_path, monkeypatch):
    archive = quick10_shaped_archive(**SMALL).archive

    # A staged delete: the B2 copy vanished after promotion, before cleanup.
    staged = World(tmp_path / "staged", monkeypatch, witness=False)
    staged.stage("output", archive)
    observation = _observed(archive)
    (staged.run / "vast_provider_command_result.json").write_text(
        json.dumps({"provider_output_remote_observation": observation}), encoding="utf-8")
    first, cleanup = staged.promote(observation=observation, cleanup=lambda: {"status": "blocked"})
    _drop_from_b2(staged, first["durable_reference"]["uri"])
    uploads = staged.cas.uploads

    resumed = staged.resume()

    assert resumed["status"] == "completed", resumed["blockers"]
    assert staged.cas.uploads == uploads + 1  # promoted again from the still-staged object
    assert first["durable_reference"]["uri"].removeprefix("s3://blueprint-artifacts/") in staged.cas.objects
    # With the copy gone and nothing left to promote it from, the receipt no longer stands.
    _drop_from_b2(staged, first["durable_reference"]["uri"])
    gone = staged.resume()
    assert gone["status"] == "blocked" and "provider_output_durable_copy_missing" in gone["blockers"]
    assert _receipt_now(staged)["status"] == "failed"  # proven gone: the only case that downgrades it

    # A local unlink: the recovered ZIP stays until its bytes are durable again.
    local_world = World(tmp_path / "local", monkeypatch)
    _, witness, _ = _paired()
    local_world.stage("paired_witness", witness, etag='"witness-1"')
    local = _local_zip(local_world, archive)
    with monkeypatch.context() as patch:
        patch.setattr(promotion._Promotion, "_witness", _kill)
        with pytest.raises(_Killed):
            local_world.promote(local_archive=local)
    interim = _receipt_now(local_world)
    assert interim["local_copy_removed_after_verified_promotion"] is False and local.is_file()
    _drop_from_b2(local_world, interim["durable_reference"]["uri"])

    resumed = local_world.resume()

    assert resumed["status"] == "completed", resumed["blockers"]
    assert not local.exists() and _receipt_now(local_world)["local_copy_removed_after_verified_promotion"] is True
    assert interim["durable_reference"]["uri"].removeprefix("s3://blueprint-artifacts/") in local_world.cas.objects


def test_a_transient_head_failure_on_resume_keeps_the_durable_receipt(world):
    archive = quick10_shaped_archive(**SMALL).archive
    world.stage("output", archive)
    observation = _observed(archive)
    (world.run / "vast_provider_command_result.json").write_text(
        json.dumps({"provider_output_remote_observation": observation}), encoding="utf-8")
    first, _ = world.promote(observation=observation, cleanup=lambda: {"status": "blocked"})

    def head_failed(**_kwargs):
        raise scene_store.TaskEvaluationConfiguredSceneObjectStoreError("configured_scene_artifact_head_failed")

    resumed = world.resume(verifier=head_failed)

    assert resumed["status"] == "blocked" and "configured_scene_artifact_head_failed" in resumed["blockers"]
    # Nothing proved the durable copy gone, so the durable record stands as it was.
    kept = _receipt_now(world)
    for field in ("status", "source", "archive_sha256", "durable_reference", "member_index", "staged_objects"):
        assert kept[field] == first[field], field
    # Later the staged object is gone and B2 answers again: the pointer still leads somewhere.
    world.spaces.stores.pop(world.keys["output"], None)
    again = world.resume()
    assert again["status"] == "completed", again["blockers"]
    assert _receipt_now(world)["durable_reference"] == first["durable_reference"]


@pytest.mark.parametrize("late_output", [False, True])
def test_a_resume_killed_during_the_witness_step_keeps_the_promoted_witness(tmp_path, monkeypatch, late_output):
    world = World(tmp_path, monkeypatch)
    output, witness, _ = _paired()
    world.stage("paired_witness", witness, etag='"witness-1"')
    first, _ = world.promote(cleanup=lambda: {"status": "blocked"})
    assert (first["status"], first["witness"]["disposition"]) == ("absent_confirmed", "promoted")
    world.cleanup()  # the witness, the only paid evidence, leaves staging behind its receipt
    assert world.keys["paired_witness"] not in world.spaces.stores
    sealed = (world.staging / records.RECEIPT_FILENAME).read_bytes()
    if late_output:
        world.stage("output", output)  # this resume changes the record: it promotes an output

    with monkeypatch.context() as patch:
        patch.setattr(promotion._Promotion, "_witness", _kill)
        with pytest.raises(_Killed):
            world.resume()

    mid = _receipt_now(world)
    section = mid["staged_objects"]["paired_witness"]
    if late_output:
        # A checkpoint (closeout waits on ``pending``), yet it names the promoted witness.
        assert section["state"] == "pending" and mid["status"] == "promoted"
    else:
        # Nothing changed before the witness step: the final receipt stands, untouched (review minor 4).
        assert (world.staging / records.RECEIPT_FILENAME).read_bytes() == sealed
    assert section["versions"] == first["staged_objects"]["paired_witness"]["versions"]
    assert mid["witness"]["reference"] == first["witness"]["reference"]
    assert world.resume()["status"] == "completed"
    end = _receipt_now(world)
    assert (end["witness"]["disposition"], end["witness"]["reference"]) == ("promoted", first["witness"]["reference"])


def test_a_transient_head_failure_on_the_witness_copy_keeps_it(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    _, witness, _ = _paired()
    world.stage("paired_witness", witness, etag='"witness-1"')
    first, _ = world.promote(cleanup=lambda: {"status": "blocked"})
    world.cleanup()

    def head_failed(**_kwargs):
        raise scene_store.TaskEvaluationConfiguredSceneObjectStoreError("configured_scene_artifact_head_failed")

    resumed = world.resume(verifier=head_failed)

    assert "paired_witness_promotion_failed:configured_scene_artifact_head_failed" in resumed["blockers"]
    assert _receipt_now(world)["staged_objects"]["paired_witness"] == first["staged_objects"]["paired_witness"]
    assert world.resume()["status"] == "completed"
    assert _receipt_now(world)["witness"]["reference"] == first["witness"]["reference"]


def test_a_redundant_witness_stands_only_with_the_primary_it_was_proven_against(paired_world):
    world = paired_world
    output, witness, _ = _paired()
    world.stage("output", output)
    world.stage("paired_witness", witness, etag='"witness-1"')
    first, _ = world.promote(observation=_observed(output), cleanup=lambda: {"status": "blocked"})
    assert first["witness"]["disposition"] == "redundant_with_promoted_output"
    uploads = world.cas.uploads
    world.resume(cleanup=lambda: {"status": "blocked"})  # the same durable primary: the proof stands
    assert _receipt_now(world)["witness"]["disposition"] == "redundant_with_promoted_output"
    assert world.cas.uploads == uploads
    # B2 loses the output, and different (unindexable) bytes land at the staged output key.
    _drop_from_b2(world, first["durable_reference"]["uri"])
    late = build_zip([Entry("frames/0001.png", Zeros(256 * 1024)), Entry("a\\b.json", b"{}")])
    world.stage("output", late, etag='"late"')

    world.resume()

    end = _receipt_now(world)
    assert end["status"] == "promoted" and end["archive_sha256"] == virtual_sha256(late)
    # The witness was redundant with the lost bytes, not these: it is promoted itself.
    assert end["witness"]["disposition"] == "promoted"
    assert [row["redundancy"]["status"] for row in end["staged_objects"]["paired_witness"]["versions"]] == [
        "not_proven"]


# -- A staging-manifest rewrite never demotes a durable output (PR A edge case) ---------


def _rewrite_manifest(world):
    """What ``--refresh-output-get-url`` does: new URL metadata, the same staged keys."""
    path = world.staging / STAGING_MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["output_get_url_refresh"] = {"schema_version": "wam_provider_output_get_refresh.v1",
                                          "status": "completed", "output_object_mutated": False}
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_resume_after_a_manifest_rewrite_rebinds_the_durable_receipt(world):
    archive = quick10_shaped_archive(**SMALL).archive
    world.stage("output", archive)
    observation = _observed(archive)
    (world.run / "vast_provider_command_result.json").write_text(
        json.dumps({"provider_output_remote_observation": observation}), encoding="utf-8")
    first, cleaned = world.promote(observation=observation)
    assert first["status"] == "promoted" and cleaned["all_objects_absent"] is True
    old_digest = first["staging_manifest_sha256"]
    _rewrite_manifest(world)
    uploads, reads = world.cas.uploads, world.spaces.whole_object_gets(world.keys["output"])

    resumed = world.resume()

    assert resumed["status"] == "completed", resumed["blockers"]
    receipt = _receipt_now(world)  # bound to the rewritten manifest
    assert receipt is not None and receipt["status"] == "promoted"
    for field in ("source", "archive_sha256", "size_bytes", "durable_reference", "member_index", "staged_objects"):
        assert receipt[field] == first[field], field
    assert receipt["staging_manifest_sha256"] == records.staging_manifest_sha256(world.staging) != old_digest
    assert receipt["rebound_from_staging_manifest_sha256"] == old_digest
    # Carried forward, not re-promoted: nothing was read from Spaces or uploaded to B2 again.
    assert (world.cas.uploads, world.spaces.whole_object_gets(world.keys["output"])) == (uploads, reads)
    assert _proof(world)["promotion_status"] == "promoted"


def test_a_rewrite_never_rebinds_a_receipt_for_another_staged_object(world):
    archive = quick10_shaped_archive(**SMALL).archive
    world.stage("output", archive)
    first, _ = world.promote(observation=_observed(archive))
    _rewrite_manifest(world)
    # The adapter's recorded observation names another object than the receipt made durable.
    (world.run / "vast_provider_command_result.json").write_text(
        json.dumps({"provider_output_remote_observation": _observed(archive, etag='"spaces-9"')}),
        encoding="utf-8")

    resumed = world.resume()

    assert resumed["status"] == "blocked" and "provider_output_observed_object_missing" in resumed["blockers"]
    receipt = _receipt_now(world)
    assert receipt["status"] == "failed" and "rebound_from_staging_manifest_sha256" not in receipt
    # The durable record it did not rebind is set aside, never destroyed.
    [aside] = sorted(world.staging.glob(records.RECEIPT_FILENAME + ".superseded-*"))
    assert json.loads(aside.read_text())["receipt_digest"] == first["receipt_digest"]


def test_a_missing_primary_copy_keeps_the_versions_whose_copies_still_stand(world):
    """Probe A3: proving the primary's durable copy gone must not also drop the other versions
    whose copies still answer, on this run or on any later one."""
    first_archive = quick10_shaped_archive(**SMALL).archive
    second = quick10_shaped_archive(**SMALL, extra_members={"late-upload.json": b"{}"}).archive
    blocked = {"cleanup": lambda: {"status": "blocked"}}
    observation = _observed(first_archive)
    (world.run / "vast_provider_command_result.json").write_text(
        json.dumps({"provider_output_remote_observation": observation}), encoding="utf-8")
    world.stage("output", first_archive)
    first, _ = world.promote(observation=observation, **blocked)
    world.stage("output", second, etag='"spaces-2"')  # a re-upload after promotion
    world.resume(**blocked)
    before = _receipt_now(world)["staged_objects"]["output"]["versions"]
    assert [row["etag"] for row in before] == ['"spaces-1"', '"spaces-2"']
    assert before[1]["archive_sha256"] == virtual_sha256(second)
    _drop_from_b2(world, first["durable_reference"]["uri"])  # the primary's copy is proven gone

    # Only the re-upload is staged, which is not the observed output: the run fails.
    rerun = world.resume(**blocked)
    assert "provider_output_remote_version_changed" in rerun["blockers"]
    assert _receipt_now(world)["status"] == "failed"
    assert _receipt_now(world)["staged_objects"]["output"]["versions"] == [before[1]]

    # Nothing is staged any more: a later run fails too, still naming the copy that stands.
    world.spaces.stores.pop(world.keys["output"])
    gone = world.resume(**blocked)
    assert "provider_output_observed_object_missing" in gone["blockers"]
    assert _receipt_now(world)["staged_objects"]["output"]["versions"] == [before[1]]
