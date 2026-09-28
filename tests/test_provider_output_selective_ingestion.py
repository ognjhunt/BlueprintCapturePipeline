# Covers (for impacted-test selection):
#   src/blueprint_pipeline/provider_output_range_ingestion.py
#   src/blueprint_pipeline/provider_output_member_index.py
#   src/blueprint_pipeline/provider_output_range_transport.py
#   tests/provider_output_fixtures.py
"""Only a selection's members are fetched, by range, from a pinned CAS archive."""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import stat
from types import SimpleNamespace

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.provider_output_member_index import build_member_index, build_member_selection
from blueprint_pipeline import provider_output_range_ingestion as ingestion
from blueprint_pipeline.provider_output_range_ingestion import (
    CasArchiveSource,
    ProviderOutputIngestionError,
    ingest_selected_members,
)
from tests.provider_output_fixtures import (
    DEFLATED,
    SECRET,
    URL,
    Entry,
    RangeStore,
    Zeros,
    build_zip,
)

ETAG = '"version1"'
LARGE = 256 * 1024**2
CONSUMERS = "policy-canary-consumers.v1"
EPISODES = ["runtime/episodes/0001.json", "runtime/result.json", "logs/worker.log"]
IDENTITY = "runtime/identity.json"


@pytest.fixture(scope="module")
def indexed():
    archive = build_zip([
        Entry("runtime/", b""),
        Entry("runtime/result.json", json.dumps({"episodes": list(range(400))}).encode(), method=DEFLATED),
        Entry("runtime/identity.json", b'{"run_id": "run-1"}'),
        Entry("runtime/frames/0001.png", bytes(range(256)) * 64),
        Entry("checkpoints/ckpt_000.pt", Zeros(LARGE)),
        Entry("runtime/episodes/0001.json", b'{"step": 1}\n' * 500, method=DEFLATED, descriptor="signed"),
        Entry("runtime/empty.txt", b""),
        Entry("logs/worker.log", b"step ok\n" * 3000, method=DEFLATED),
    ])
    reader = RangeStore(archive).reader(block_bytes=8 * 1024**2, maximum_archive_bytes=archive.size)
    return archive, build_member_index(reader, maximum_expanded_bytes=2 * LARGE)


def _reference(index):
    digest = index["archive"]["sha256"]
    return {
        "schema_version": "task_evaluation_scene_artifact_reference.v1",
        "status": "remote_verified",
        "artifact_kind": "provider-output",
        "uri": ("s3://blueprint-artifacts/blueprint/arm-decision-proof-v1/configured-scenes/artifacts/"
                f"provider-output/sha256/{digest.removeprefix('sha256:')}/vast_provider_runtime_output.zip"),
        "digest": digest,
        "size_bytes": index["archive"]["size"],
        "content_addressed_key": True,
        "remote_identity_verified": True,
        "full_byte_service_account_readback_passed": True,
        "raw_secret_values_recorded": False,
    }


def _ingest(tmp_path, store, index, selection, *, reserve=None, reference=None, presign=None, **options):
    source = CasArchiveSource(reference=reference or _reference(index), presign=presign or (lambda: URL),
                              opener=store.opener, block_bytes=64 * 1024)
    return ingest_selected_members(
        source=source, index=index, selection=selection, members_root=tmp_path / "members",
        metadata_root=tmp_path / "ingestion", reserve=reserve or (lambda needed: None), **options)


def _rows(index):
    return {row["path"]: row for row in index["members"]}


def _data_range(row):
    return (row["data_offset"], row["data_offset"] + row["compressed_size"] - 1)


def _files(root):
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def _journal(tmp_path):
    return [json.loads(line) for path in sorted((tmp_path / "ingestion").glob("members-*.jsonl"))
            for line in path.read_text().splitlines()]


def test_only_selected_members_are_written_and_journaled(tmp_path, indexed):
    archive, index = indexed
    rows, store, reservations = _rows(index), RangeStore(archive), []
    selection = build_member_selection(index, EPISODES, selection_version=CONSUMERS)

    result = _ingest(tmp_path, store, index, selection, reserve=reservations.append)

    assert result["status"] == "materialized" and result["blockers"] == []
    ordered = [row["path"] for row in index["members"] if row["path"] in EPISODES]
    assert _files(tmp_path / "members") == sorted(EPISODES)
    for path in EPISODES:
        target = tmp_path / "members" / path
        assert "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest() == rows[path]["sha256"]
        assert stat.S_IMODE(target.stat().st_mode) == 0o440
    assert sorted((row["relative_path"], row["size_bytes"], row["sha256"], row["crc32"])
                  for row in _journal(tmp_path)) == sorted(
        (path, rows[path]["size"], rows[path]["sha256"], rows[path]["crc32"]) for path in EPISODES)
    # One range request per selected member, for exactly its record data, pinned.
    assert store.ranges()[1:] == [_data_range(rows[path]) for path in ordered]
    assert store.whole_object_gets() == 0
    assert all(row["if_match"] == ETAG for row in store.requests[1:])
    # reserve() precedes each write with the bytes still needed.
    sizes = [rows[path]["size"] for path in ordered]
    assert reservations == [sum(sizes[position:]) for position in range(len(sizes))]
    # The receipt names every file member: selected ones materialized, the rest remote.
    receipt = json.loads((tmp_path / "ingestion/receipt.json").read_text())
    assert receipt == result
    assert result["receipt_digest"] == canonical_digest(result, digest_field="receipt_digest")
    assert result["members"] == [
        {"path": row["path"], "disposition": "materialized" if row["path"] in EPISODES else "remote",
         "size": row["size"], "sha256": row["sha256"], "crc32": row["crc32"]}
        for row in index["members"] if row["kind"] == "file"]
    assert (result["materialized_member_count"], result["remote_member_count"]) == (3, 4)
    assert result["private_url_recorded"] is False
    for path in (tmp_path / "ingestion").iterdir():
        text = path.read_bytes()
        assert SECRET.encode() not in text and b"https:" not in text and b"X-Amz" not in text


def test_unselected_large_member_is_never_requested_or_written(tmp_path, indexed):
    archive, index = indexed
    rows, store, samples = _rows(index), RangeStore(archive), []
    checkpoint = rows["checkpoints/ckpt_000.pt"]
    wanted = [row["path"] for row in index["members"] if row["kind"] == "file" and row is not checkpoint]
    selection = build_member_selection(index, wanted, selection_version=CONSUMERS)

    def disk_usage(path):
        used = sum(item.stat().st_size for root in ("members", "ingestion")
                   for item in (tmp_path / root).rglob("*") if item.is_file())
        samples.append(used)
        return SimpleNamespace(free=10**12 - used)

    result = _ingest(tmp_path, store, index, selection, disk_usage_provider=disk_usage)

    assert result["status"] == "materialized"
    assert not any(first < checkpoint["record_end_offset"] and last >= checkpoint["local_header_offset"]
                   for first, last in store.ranges())
    assert store.whole_object_gets() == 0
    assert not (tmp_path / "members/checkpoints").exists()
    assert _files(tmp_path / "members") == sorted(wanted)
    selected_bytes = sum(rows[path]["size"] for path in wanted)
    assert samples and max(samples) < selected_bytes + 64 * 1024 < LARGE
    assert result["minimum_observed_free_bytes"] == 10**12 - max(samples)
    assert result["transferred_bytes"] < 1024**2
    remote = [row for row in result["members"] if row["disposition"] == "remote"]
    assert remote == [{"path": "checkpoints/ckpt_000.pt", "disposition": "remote", "size": LARGE,
                       "sha256": checkpoint["sha256"], "crc32": checkpoint["crc32"]}]
    assert result["remote_bytes"] == LARGE


def test_resume_refuses_a_different_index_or_contract(tmp_path, indexed):
    archive, index = indexed
    rows, store = _rows(index), RangeStore(archive)
    selection = build_member_selection(index, EPISODES, selection_version=CONSUMERS)
    calls = []

    def reserve_once(needed):
        calls.append(needed)
        if len(calls) == 2:
            raise ProviderOutputIngestionError("provider_output_disk_reservation_refused")

    first = _ingest(tmp_path, store, index, selection, reserve=reserve_once)
    assert first["status"] == "blocked" and first["blockers"] == ["provider_output_disk_reservation_refused"]
    assert _files(tmp_path / "members") == ["runtime/result.json"]
    assert [row["relative_path"] for row in _journal(tmp_path)] == ["runtime/result.json"]

    other_contract = build_member_selection(index, EPISODES, selection_version="policy-canary-consumers.v2")
    fewer_members = build_member_selection(index, EPISODES[1:], selection_version=CONSUMERS)
    other_index = copy.deepcopy(index)
    other_index["limits"]["maximum_members"] = 9_999
    other_index["index_digest"] = canonical_digest(other_index, digest_field="index_digest")
    rebound = build_member_selection(other_index, EPISODES, selection_version=CONSUMERS)
    for candidate_index, candidate in ((index, other_contract), (index, fewer_members),
                                       (other_index, rebound)):
        with pytest.raises(ProviderOutputIngestionError, match="^provider_output_resume_binding_mismatch$"):
            _ingest(tmp_path, store, candidate_index, candidate)
    # A selection is only ever read against the index it is bound to.
    with pytest.raises(ProviderOutputIngestionError, match="^provider_output_member_selection_index_mismatch$"):
        _ingest(tmp_path, store, index, rebound)
    assert _files(tmp_path / "members") == ["runtime/result.json"]

    before = len(store.requests)
    resumed = _ingest(tmp_path, store, index, selection)
    assert resumed["status"] == "materialized" and resumed["resumed_member_count"] == 1
    assert _data_range(rows["runtime/result.json"]) not in [row["range"] for row in store.requests[before:]]
    assert _files(tmp_path / "members") == sorted(EPISODES)




def test_runs_that_record_nothing_leave_no_journal_and_never_wedge_resume(tmp_path, indexed):
    archive, index = indexed
    selection = build_member_selection(index, EPISODES, selection_version=CONSUMERS)

    def refuse(needed):
        raise ProviderOutputIngestionError("provider_output_disk_reservation_refused")

    for _ in range(300):
        refused = _ingest(tmp_path, RangeStore(archive), index, selection, reserve=refuse)
        assert refused["blockers"] == ["provider_output_disk_reservation_refused"]
    assert not list((tmp_path / "ingestion").glob("members-*.jsonl"))

    result = _ingest(tmp_path, RangeStore(archive), index, selection)
    assert result["status"] == "materialized", result["blockers"]
    again = _ingest(tmp_path, RangeStore(archive), index, selection)
    assert again["status"] == "materialized" and again["resumed_member_count"] == len(EPISODES)
    assert len(list((tmp_path / "ingestion").glob("members-*.jsonl"))) == 1

def test_interrupted_member_is_fetched_again_and_its_partial_rechecked(tmp_path, indexed):
    archive, index = indexed
    frame = _rows(index)["runtime/frames/0001.png"]
    store, reservations = RangeStore(archive), []
    store.truncate_when = lambda request: request["range"] == _data_range(frame)
    selection = build_member_selection(index, [frame["path"]], selection_version=CONSUMERS)

    first = _ingest(tmp_path, store, index, selection)
    assert first["blockers"] == ["provider_output_range_truncated"]
    partials = list((tmp_path / "ingestion").glob("*.partial"))
    assert len(partials) == 1 and partials[0].stat().st_size == frame["size"] - 1
    assert _files(tmp_path / "members") == []

    store.truncate_when = None
    resumed = _ingest(tmp_path, store, index, selection, reserve=reservations.append)
    assert resumed["status"] == "materialized" and reservations == [1]
    target = tmp_path / "members" / frame["path"]
    assert "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest() == frame["sha256"]
    assert not list((tmp_path / "ingestion").glob("*.partial"))


def test_member_verified_before_an_interrupted_rename_is_adopted_on_resume(tmp_path, indexed, monkeypatch):
    archive, index = indexed
    frame = _rows(index)["runtime/frames/0001.png"]
    selection = build_member_selection(index, [frame["path"]], selection_version=CONSUMERS)
    real_rename = ingestion.os.rename

    def crash(source, target):
        raise OSError("interrupted before rename")

    with monkeypatch.context() as patch:
        patch.setattr(ingestion.os, "rename", crash)
        first = _ingest(tmp_path, RangeStore(archive), index, selection)
    assert ingestion.os.rename is real_rename
    assert first["blockers"] == ["provider_output_archive_or_io_failed"]
    (partial,) = (tmp_path / "ingestion").glob("*.partial")
    assert partial.stat().st_size == frame["size"]

    resumed = _ingest(tmp_path, RangeStore(archive), index, selection)
    assert resumed["status"] == "materialized", resumed["blockers"]
    target = tmp_path / "members" / frame["path"]
    assert "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest() == frame["sha256"]
    assert stat.S_IMODE(target.stat().st_mode) == 0o440
    assert not list((tmp_path / "ingestion").glob("*.partial"))


def test_roots_on_different_devices_are_refused_before_any_transfer(tmp_path, indexed, monkeypatch):
    archive, index = indexed
    store = RangeStore(archive)
    selection = build_member_selection(index, ["runtime/identity.json"], selection_version=CONSUMERS)
    real_device = ingestion._device
    monkeypatch.setattr(ingestion, "_device", lambda path: real_device(path) + (path.name == "members"))
    with pytest.raises(ProviderOutputIngestionError, match="^provider_output_roots_cross_device$"):
        _ingest(tmp_path, store, index, selection)
    assert store.requests == [] and not (tmp_path / "ingestion/binding.json").exists()


def test_a_rename_across_mounts_is_a_typed_refusal(tmp_path, indexed, monkeypatch):
    archive, index = indexed
    selection = build_member_selection(index, ["runtime/identity.json"], selection_version=CONSUMERS)

    def cross_device(source, target):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(ingestion.os, "rename", cross_device)
    result = _ingest(tmp_path, RangeStore(archive), index, selection)
    assert result["blockers"] == ["provider_output_roots_cross_device"]
    assert _files(tmp_path / "members") == []


def _write_journal_row(tmp_path, row, **changes):
    value = {"relative_path": row["path"], "size_bytes": row["size"], "sha256": row["sha256"],
             "crc32": row["crc32"], **changes}
    value["record_digest"] = canonical_digest(value, digest_field="record_digest")
    (tmp_path / "ingestion/members-forged.jsonl").write_text(json.dumps(value) + "\n")


def _drop_journals(tmp_path):
    for path in (tmp_path / "ingestion").glob("members-*.jsonl"):
        path.unlink()


@pytest.mark.parametrize("change, code", [
    ("smaller_remote", "provider_output_remote_size_mismatch"),
    ("new_remote_version", "provider_output_resume_remote_identity_mismatch"),
    ("journaled_unselected_member", "provider_output_resume_inventory_changed"),
    ("stray_file", "provider_output_resume_inventory_changed"),
    ("symlink", "provider_output_resume_inventory_changed"),
    ("tampered_unjournaled_target", "provider_output_resume_file_changed"),
    ("journal_disagrees_with_target", "provider_output_resume_file_changed"),
])
def test_resume_refuses_a_changed_remote_or_member_tree(tmp_path, indexed, change, code):
    archive, index = indexed
    rows, target = _rows(index), tmp_path / "members" / IDENTITY
    selection = build_member_selection(index, [IDENTITY], selection_version=CONSUMERS)
    assert _ingest(tmp_path, RangeStore(archive), index, selection)["status"] == "materialized"
    store = RangeStore(archive)
    if change == "smaller_remote":
        store = RangeStore(b"not the indexed archive" * 8)
    elif change == "new_remote_version":
        store = RangeStore(archive, etag='"version2"')
    elif change == "journaled_unselected_member":
        _write_journal_row(tmp_path, rows["runtime/result.json"])
    elif change == "stray_file":
        (tmp_path / "members/runtime/stray.txt").write_text("not selected")
    elif change == "symlink":
        (tmp_path / "members/runtime/link").symlink_to(target)
    elif change == "tampered_unjournaled_target":
        _drop_journals(tmp_path)
        target.chmod(0o640)
        target.write_bytes(b"X" + target.read_bytes()[1:])
    else:
        _drop_journals(tmp_path)
        _write_journal_row(tmp_path, rows[IDENTITY], sha256="sha256:" + "0" * 64)
    result = _ingest(tmp_path, store, index, selection)
    assert result["status"] == "blocked" and result["blockers"] == [code]


def test_member_renamed_before_its_journal_row_is_adopted_without_refetch(tmp_path, indexed, monkeypatch):
    archive, index = indexed
    selection = build_member_selection(index, [IDENTITY], selection_version=CONSUMERS)

    def crash(*args):
        raise OSError("interrupted before the journal append")

    with monkeypatch.context() as patch:
        patch.setattr(ingestion, "_append_journal", crash)
        first = _ingest(tmp_path, RangeStore(archive), index, selection)
    assert first["blockers"] == ["provider_output_archive_or_io_failed"]
    assert _files(tmp_path / "members") == [IDENTITY] and _journal(tmp_path) == []

    store = RangeStore(archive)
    resumed = _ingest(tmp_path, store, index, selection)
    assert resumed["status"] == "materialized" and resumed["resumed_member_count"] == 1
    assert store.ranges() == [(0, 0)]  # the reader's own probe: the member is not fetched again
    assert [row["relative_path"] for row in _journal(tmp_path)] == [IDENTITY]


def _raise(error):
    def raising(*args):
        raise error
    return raising


@pytest.mark.parametrize("presign", [
    _raise(RuntimeError("signing failed for " + URL)),
    lambda: "",
    lambda: URL.replace("https://", "http://"),  # refused by the transport policy
])
def test_a_failed_presign_is_a_typed_refusal_that_records_no_url(tmp_path, indexed, presign):
    archive, index = indexed
    selection = build_member_selection(index, [IDENTITY], selection_version=CONSUMERS)
    result = _ingest(tmp_path, RangeStore(archive), index, selection, presign=presign)
    assert result["blockers"] == ["provider_output_cas_presign_invalid"]
    for path in (tmp_path / "ingestion").iterdir():
        assert b"storage.example.invalid" not in path.read_bytes() and SECRET.encode() not in path.read_bytes()


@pytest.mark.parametrize("error, code", [
    (ProviderOutputIngestionError("provider_output_lane_quota_refused"), "provider_output_lane_quota_refused"),
    (OSError("lane_quota_exceeded"), "lane_quota_exceeded"),
    (ValueError("Disk full: 10 GiB needed"), "provider_output_disk_reservation_refused"),
])
def test_reserve_refusals_become_typed_codes(tmp_path, indexed, error, code):
    archive, index = indexed
    selection = build_member_selection(index, [IDENTITY], selection_version=CONSUMERS)
    result = _ingest(tmp_path, RangeStore(archive), index, selection, reserve=_raise(error))
    assert result["blockers"] == [code] and _files(tmp_path / "members") == []

def test_selected_member_must_match_its_index_digest(tmp_path, indexed):
    archive, index = indexed
    rows = _rows(index)
    # An index whose SHA-256 disagrees with bytes whose CRC-32 still matches.
    forged = copy.deepcopy(index)
    next(row for row in forged["members"] if row["path"] == "runtime/result.json")["sha256"] = "sha256:" + "f" * 64
    forged["index_digest"] = canonical_digest(forged, digest_field="index_digest")
    selection = build_member_selection(forged, ["runtime/result.json"], selection_version=CONSUMERS)
    result = _ingest(tmp_path / "forged", RangeStore(archive), forged, selection)
    assert result["status"] == "blocked" and result["blockers"] == ["provider_output_member_digest_mismatch"]
    assert _files(tmp_path / "forged/members") == []
    assert not list((tmp_path / "forged/ingestion").glob("*.partial"))

    # A store serving different bytes for a stored member under the same ETag.
    frame = rows["runtime/frames/0001.png"]
    tampered = archive.patched(frame["data_offset"] + 100, b"\xff\xfe\xfd\xfc")
    selection = build_member_selection(index, ["runtime/frames/0001.png"], selection_version=CONSUMERS)
    result = _ingest(tmp_path / "tampered", RangeStore(tampered), index, selection)
    assert result["blockers"] == ["provider_output_member_digest_mismatch"]
    assert _files(tmp_path / "tampered/members") == []
    assert not list((tmp_path / "tampered/ingestion").glob("*.partial"))

    # A deflated member whose stream no longer decodes.
    result_row = rows["runtime/result.json"]
    corrupt = archive.patched(result_row["data_offset"], b"\xff")
    selection = build_member_selection(index, ["runtime/result.json"], selection_version=CONSUMERS)
    result = _ingest(tmp_path / "corrupt", RangeStore(corrupt), index, selection)
    assert result["blockers"] == ["provider_output_archive_deflate_invalid"]
    assert _files(tmp_path / "corrupt/members") == []
    assert not list((tmp_path / "corrupt/ingestion").glob("*.partial"))


def test_cas_source_must_name_the_indexed_archive(tmp_path, indexed):
    archive, index = indexed
    selection = build_member_selection(index, ["runtime/identity.json"], selection_version=CONSUMERS)
    other = _reference(index) | {"digest": "sha256:" + "0" * 64}
    other["uri"] = other["uri"].replace(index["archive"]["sha256"].removeprefix("sha256:"), "0" * 64)
    with pytest.raises(ProviderOutputIngestionError, match="^provider_output_cas_reference_mismatch$"):
        _ingest(tmp_path, RangeStore(archive), index, selection, reference=other)
    unverified = _reference(index) | {"full_byte_service_account_readback_passed": False}
    with pytest.raises(ProviderOutputIngestionError, match="^provider_output_cas_reference_invalid$"):
        _ingest(tmp_path, RangeStore(archive), index, selection, reference=unverified)
    assert not (tmp_path / "members").exists() and not (tmp_path / "ingestion").exists()

