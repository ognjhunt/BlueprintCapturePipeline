# Covers (for impacted-test selection):
#   src/blueprint_pipeline/provider_output_remote_collection.py
#   src/blueprint_pipeline/provider_output_range_transport.py
#   src/blueprint_pipeline/wam_provider_output.py
#   src/blueprint_pipeline/vast_structured_policy_canary_inspection.py
#   tests/provider_output_fixtures.py
"""A staged provider output is observed by range; nothing is downloaded or written."""

from __future__ import annotations

import json
import re
import struct
import urllib.error
import zipfile

import pytest

from blueprint_pipeline.provider_output_remote_collection import RemoteProviderOutputCollector
from blueprint_pipeline.vast_structured_policy_canary_inspection import (
    inspect_structured_policy_canary_archive,
)
from blueprint_pipeline.wam_provider_output import inspect_provider_runtime_output_zip
from tests.provider_output_fixtures import (
    DEFLATED,
    QUICK10_RESULT,
    SECRET,
    URL,
    Entry,
    RangeStore,
    VirtualFile,
    Zeros,
    build_zip,
    no_disk_writes,
    quick10_shaped_archive,
)

BLOCK = 8 * 1024**2
OUTPUT_NAME = "vast_provider_runtime_output.zip"


def _collect(store, output_path, *, opener=None, **options):
    options.setdefault("maximum_archive_bytes", store.object.size)
    collector = RemoteProviderOutputCollector(expected_video_count=0, opener=opener or store.opener,
                                              **options)
    return collector, collector(url=URL, output_path=output_path, minimum_free_bytes=8 * 1024**3)


def _record_span(obj, info):
    """``[local header, end of record data)`` for one member of a virtual archive."""
    name_length, extra_length = struct.unpack(
        "<HH", bytes(obj.read(info.header_offset + 26, info.header_offset + 30)))
    data = info.header_offset + 30 + name_length + extra_length
    return info.header_offset, data + info.compress_size, data


def test_remote_collector_inspects_the_result_without_writing_the_archive(tmp_path, monkeypatch):
    q10 = quick10_shaped_archive()
    store = RangeStore(q10.archive)
    output = tmp_path.resolve() / OUTPUT_NAME
    monkeypatch.chdir(tmp_path)

    with no_disk_writes(monkeypatch) as attempts:
        collector, transfer = _collect(store, output)

    assert attempts == [] and list(tmp_path.iterdir()) == []
    assert (transfer["status"], transfer["delivery"]) == ("completed", "remote_only")
    assert transfer["download_attempted"] is False and transfer["downloaded_size_bytes"] == 0
    assert transfer["disk_capacity"]["status"] == "not_required"
    assert transfer["blockers"] == []
    inspection = transfer["inspection"]
    assert inspection["runtime_result_member"] == QUICK10_RESULT
    assert inspection["runtime_result_status"] == q10.result["status"]
    assert (inspection["zip_path"], inspection["zip_present"]) == (str(output), True)
    assert inspection["zip_size_bytes"] == q10.archive.size
    # No MP4 is copied, so ffprobe is never asked; the blocker says so.
    assert inspection["mp4_count"] == 60 and inspection["mp4_validation"]["files"] == []
    assert "mp4_ffprobe_validation_not_requested" in inspection["mp4_validation"]["blockers"]
    assert SECRET not in json.dumps(transfer) and "https:" not in json.dumps(transfer)

    # The same bytes inspected from disk give the same answer, but for the path they name.
    output.write_bytes(q10.archive.to_bytes())
    by_path = inspect_provider_runtime_output_zip(output, expected_video_count=0)
    assert {**inspection, "zip_path": None} == {**by_path, "zip_path": None}
    with zipfile.ZipFile(output) as archive:
        assert transfer["structured_policy_canary"] == inspect_structured_policy_canary_archive(archive)


@pytest.fixture(scope="module")
def past_4_gib():
    q10 = quick10_shaped_archive(mp4_bytes=72 * 1024**2)
    with zipfile.ZipFile(VirtualFile(q10.archive)) as archive:
        infos = {info.filename: info for info in archive.infolist()}
        start_dir = archive.start_dir
    return q10, infos, start_dir


def test_remote_collection_transfers_a_block_budget_not_bulk_members(tmp_path, past_4_gib):
    q10, infos, start_dir = past_4_gib
    size = q10.archive.size
    result_span = _record_span(q10.archive, infos[QUICK10_RESULT])[:2]
    # ZIP64: the result a collector must read starts past 4 GiB.
    assert infos[QUICK10_RESULT].header_offset > 0xFFFFFFFF and size > 4 * 1024**3
    store = RangeStore(q10.archive)

    collector, transfer = _collect(store, tmp_path / OUTPUT_NAME)

    assert transfer["status"] == "completed", transfer["blockers"]
    assert transfer["inspection"]["runtime_result_member"] == QUICK10_RESULT
    assert transfer["inspection"]["runtime_result_status"] == q10.result["status"]
    assert store.requests[0]["range"] == (0, 0) and store.whole_object_gets() == 0
    # The reader fetches aligned blocks, so neighbours of a needed record are
    # read too. Every block it fetched overlaps the end records, the central
    # directory or the inspected result's record; none lies wholly in bulk data.
    needed = [(size - 22 - 20 - 56, size), (start_dir, size - 22 - 20 - 56), result_span]
    for row in store.requests[1:]:
        first, last = row["range"]
        assert first % BLOCK == 0 and last - first < BLOCK
        assert any(first < end and last >= start for start, end in needed), row
        assert row["if_match"] == store.etag
    blocks = {block for start, end in needed for block in range(start // BLOCK, (end - 1) // BLOCK + 1)}
    bulk = sum(q10.payloads[name].size for name in q10.bulk_members)
    assert bulk > 4 * 1024**3
    assert transfer["transferred_bytes"] <= 1 + BLOCK * (len(blocks) + 2)
    assert transfer["transferred_bytes"] * 100 < bulk
    assert transfer["transferred_bytes"] == 1 + sum(last - first + 1 for first, last in store.ranges()[1:])


def test_remote_collector_pins_and_records_the_observed_etag_and_size(tmp_path):
    q10 = quick10_shaped_archive(cells=2)
    store = RangeStore(q10.archive, etag='"spaces-etag-1"')

    collector, transfer = _collect(store, tmp_path / OUTPUT_NAME)

    observation = {"size_bytes": q10.archive.size, "etag": '"spaces-etag-1"'}
    assert transfer["remote_object"] == observation and collector.observation == observation
    assert all(row["if_match"] == '"spaces-etag-1"' for row in store.requests[1:])
    assert transfer["http_request_count"] == len(store.requests)

    # A new version written after the probe is refused, never read as a mix.
    swapped = RangeStore(q10.archive)

    def reupload_after_probe(request, timeout, policy):
        response = swapped.opener(request, timeout, policy)
        swapped.etag = '"version2"'
        return response

    collector, transfer = _collect(swapped, tmp_path / OUTPUT_NAME, opener=reupload_after_probe)
    assert (transfer["status"], transfer["blockers"]) == ("blocked", ["provider_output_remote_version_changed"])
    assert collector.observation is None and "inspection" not in transfer


def test_absent_remote_output_is_a_typed_not_ready_transfer(tmp_path):
    store = RangeStore(quick10_shaped_archive(cells=1).archive)
    store.absent = True

    collector, transfer = _collect(store, tmp_path / OUTPUT_NAME)

    assert transfer["status"] == "blocked" and transfer["blockers"] == ["provider_output_not_ready"]
    assert transfer["http_status_code"] == 404 and transfer["download_attempted"] is False
    assert transfer["delivery"] == "remote_only" and transfer["disk_capacity"]["status"] == "not_required"
    assert collector.observation is None and list(tmp_path.iterdir()) == []
    assert SECRET not in json.dumps(transfer)


def _result_then_bulk(result_payload):
    return build_zip([
        Entry("cell_runs/00/" + QUICK10_RESULT, json.dumps({"status": "completed"}).encode(), method=DEFLATED),
        Entry(QUICK10_RESULT, result_payload, method=DEFLATED),
        Entry("zz_trailing_frames/0001.png", Zeros(4 * 1024**2)),
    ])


def test_result_member_over_the_read_cap_is_a_typed_refusal(tmp_path):
    # Stored, so the whole 3 MiB would be one read against a 1 MiB cap.
    archive = build_zip([Entry(QUICK10_RESULT, Zeros(3 * 1024**2)),
                         Entry("cell_runs/00/" + QUICK10_RESULT, b'{"status": "completed"}')])
    store, block = RangeStore(archive), 256 * 1024
    with zipfile.ZipFile(VirtualFile(archive)) as opened:
        _, _, data = _record_span(archive, opened.getinfo(QUICK10_RESULT))

    collector, transfer = _collect(store, tmp_path / OUTPUT_NAME, maximum_read_bytes=1024**2,
                                   block_bytes=block)

    assert (transfer["status"], transfer["blockers"]) == (
        "blocked", ["provider_output_inspected_member_over_read_cap"])
    assert collector.observation is None and "inspection" not in transfer
    # Refused from the directory alone: the member's data past its first block
    # was never requested.
    assert not any(first <= data + block <= last for first, last in store.ranges())

    # A small deflated member that would inflate past the member cap is refused the same way.
    bomb = _result_then_bulk(b"{}" * 1024**2)
    collector, transfer = _collect(RangeStore(bomb), tmp_path / OUTPUT_NAME,
                                   maximum_inspected_member_bytes=1024**2)
    assert transfer["blockers"] == ["provider_output_inspected_member_over_read_cap"]


def test_a_transport_failure_during_inspection_never_falls_back_to_a_cell_result(tmp_path):
    archive = _result_then_bulk(json.dumps({"status": "blocked", "blockers": ["policy_failed"],
                                            "padding": ["x" * 64] * 100}).encode())
    with zipfile.ZipFile(VirtualFile(archive)) as opened:
        _, _, data = _record_span(archive, opened.getinfo(QUICK10_RESULT))
    store = RangeStore(archive)
    options = {"block_bytes": 128 * 1024}

    collector, healthy = _collect(store, tmp_path / OUTPUT_NAME, **options)
    assert healthy["inspection"]["runtime_result_member"] == QUICK10_RESULT
    assert healthy["inspection"]["runtime_result_status"] == "blocked"

    def failing(request, timeout, policy):
        requested = dict(request.header_items()).get("Range", "")
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", requested)
        if match and int(match[1]) <= data <= int(match[2]):
            raise urllib.error.HTTPError("redacted", 500, "Internal Error", {}, None)
        return store.opener(request, timeout, policy)

    collector, transfer = _collect(RangeStore(archive), tmp_path / OUTPUT_NAME, opener=failing, **options)

    # The path inspection would fall back to the completed cell result; the
    # collector discards an inspection that saw any transport failure.
    assert (transfer["status"], transfer["blockers"]) == ("blocked", ["provider_output_http_failed"])
    assert collector.observation is None and "inspection" not in transfer


def test_collector_refuses_invalid_limits():
    for options in ({"maximum_archive_bytes": 0}, {"maximum_archive_bytes": 10, "block_bytes": 0},
                    {"maximum_archive_bytes": 10, "maximum_read_bytes": 1024, "block_bytes": 4096},
                    {"maximum_archive_bytes": True}):
        with pytest.raises(ValueError, match="^provider_output_remote_collection_limits_invalid$"):
            RemoteProviderOutputCollector(expected_video_count=0, **options)


@pytest.mark.parametrize("url", ["http://storage.example.invalid/run.zip?X-Amz-Signature=SECRET_DO_NOT_RECORD",
                                 "https://user:secret@storage.example.invalid/run.zip"])
def test_a_url_the_transfer_policy_refuses_is_a_typed_transport_failure(tmp_path, url):
    store = RangeStore(quick10_shaped_archive(cells=1).archive)
    collector = RemoteProviderOutputCollector(maximum_archive_bytes=store.object.size, expected_video_count=0,
                                              opener=store.opener)

    transfer = collector(url=url, output_path=tmp_path / OUTPUT_NAME, minimum_free_bytes=0)

    assert (transfer["status"], transfer["blockers"]) == ("blocked", ["provider_output_transport_failed"])
    assert store.requests == [] and collector.observation is None
    assert SECRET not in json.dumps(transfer) and "storage.example" not in json.dumps(transfer)
