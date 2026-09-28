# Covers (for impacted-test selection):
#   src/blueprint_pipeline/provider_output_member_index.py
#   src/blueprint_pipeline/provider_output_range_transport.py
#   src/blueprint_pipeline/provider_output_range_ingestion.py
#   tests/provider_output_fixtures.py
"""One pinned pass indexes a provider ZIP by member digest and byte range."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import struct
import tracemalloc
import zipfile
import zlib

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.provider_output_member_index import (
    BULK_EXTENSIONS,
    MemberInflater,
    ProviderOutputMemberIndexError,
    build_member_index,
    build_member_selection,
    main,
    validate_member_index,
    validate_member_selection,
)
from blueprint_pipeline.provider_output_range_transport import ProviderOutputTransportError
from tests.provider_output_fixtures import (
    DEFLATED,
    SECRET,
    Entry,
    RangeStore,
    VirtualFile,
    Zeros,
    build_zip,
    deflate,
    no_disk_writes,
    python_zip,
)

EXPANDED = 64 * 1024**2
MIN_BLOCK = 128 * 1024
ETAG = '"version1"'
# A known-answer vector: the SHA-256 of 4 GiB + 1 MiB of zero bytes.
ZERO_RUN = 4 * 1024**3 + 1024**2
ZERO_RUN_SHA256 = "sha256:829816e339ff597ec3ada4c30fc840d3f2298444169d242952a54bcf3fcd7747"


def _files():
    return {
        "runtime/": b"",
        "runtime/result.json": json.dumps({"status": "completed", "episodes": list(range(60))}).encode(),
        "runtime/identity.json": b'{"run_id": "run-1"}',
        "runtime/frames/": b"",
        "runtime/frames/0001.png": bytes(range(256)) * 40,
        "runtime/empty.txt": b"",
        "checkpoints/ckpt_000.pt": hashlib.sha256(b"seed").digest() * 2048,
        "logs/worker.log": b"step ok\n" * 400,
    }


def _mixed_archive():
    """Python's own writer, alternating stored and deflated members."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for position, (name, payload) in enumerate(_files().items()):
            method = zipfile.ZIP_DEFLATED if position % 2 else zipfile.ZIP_STORED
            archive.writestr(name, payload, compress_type=method)
    return buffer.getvalue()


def _index(data, **options):
    """Index through the smallest admitted block, with short reads splitting records."""
    store = data if isinstance(data, RangeStore) else RangeStore(data)
    store.max_read = options.pop("chunk_bytes", 4096)
    reader = store.reader(block_bytes=MIN_BLOCK)
    options.setdefault("maximum_expanded_bytes", EXPANDED)
    return build_member_index(reader, **options), store


def _refusal(data, **options):
    with pytest.raises(ProviderOutputMemberIndexError) as caught:
        _index(data, **options)
    return str(caught.value)


def _expected_from_zipfile(data):
    """What an independent reader says each record is, plus the directory offset."""
    rows = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            name_length, extra_length = struct.unpack(
                "<HH", data[info.header_offset + 26:info.header_offset + 30])
            rows.append({
                "path": info.filename.rstrip("/"),
                "kind": "directory" if info.is_dir() else "file",
                "size": info.file_size,
                "compressed_size": info.compress_size,
                "method": {zipfile.ZIP_STORED: "stored", zipfile.ZIP_DEFLATED: "deflate"}[info.compress_type],
                "crc32": info.CRC,
                "sha256": "sha256:" + hashlib.sha256(archive.read(info)).hexdigest(),
                "mode": info.external_attr >> 16,
                "local_header_offset": info.header_offset,
                "data_offset": info.header_offset + 30 + name_length + extra_length,
            })
        return rows, archive.start_dir


def _assert_matches_zipfile(index, data):
    expected, directory_offset = _expected_from_zipfile(data)
    members = index["members"]
    assert [{k: v for k, v in row.items() if k != "record_end_offset"} for row in members] == expected
    starts = [row["local_header_offset"] for row in members]
    ends = [row["record_end_offset"] for row in members]
    # Records tile the archive up to its central directory: no gap, no overlap.
    assert starts[0] == 0 and ends[:-1] == starts[1:] and ends[-1] == directory_offset
    assert index["directory"]["offset"] == directory_offset
    assert index["archive"]["sha256"] == "sha256:" + hashlib.sha256(data).hexdigest()
    assert index["index_digest"] == canonical_digest(index, digest_field="index_digest")


def test_one_pass_index_matches_zipfile_for_stored_and_deflated_members():
    data = _mixed_archive()
    index, store = _index(data, chunk_bytes=1000)

    _assert_matches_zipfile(index, data)
    files = [row for row in index["members"] if row["kind"] == "file"]
    assert {row["method"] for row in files} == {"stored", "deflate"}
    assert index["schema_version"] == "provider_output_member_index.v1"
    assert index["archive"] == {"sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
                                "size": len(data), "etag": ETAG, "generation": None,
                                "durable_reference": None}
    assert index["limits"] == {"maximum_members": 10_000, "maximum_expanded_bytes": EXPANDED,
                               "maximum_member_inflated_bytes": EXPANDED}
    bulk = sum(row["size"] for row in files if row["path"].endswith((".pt", ".png")))
    total = sum(row["size"] for row in files)
    assert index["totals"] == {
        "members": 8, "files": 6, "directories": 2, "bytes": total,
        "compressed_bytes": sum(row["compressed_size"] for row in index["members"]),
        "bytes_by_class": {"bulk": bulk, "small": total - bulk},
    }
    # Archive facts only: no URL, no signature, no selection vocabulary.
    text = json.dumps(index)
    assert index["private_url_recorded"] is False
    assert SECRET not in text and "https:" not in text
    assert "disposition" not in text and "consumers" not in text
    # The directory is read by range under the pinned ETag, then exactly one
    # whole-object GET streams every byte once.
    assert store.whole_object_gets() == 1
    assert all(row["if_match"] == ETAG for row in store.requests[1:])
    # Deterministic: a pass with a different transport block size agrees byte for byte.
    again, _ = _index(data, chunk_bytes=7)
    assert json.dumps(again, sort_keys=True) == json.dumps(index, sort_keys=True)


def test_record_ranges_reopen_each_member_with_one_range_request():
    data = _mixed_archive()
    index, store = _index(data)
    reader = store.reader(block_bytes=MIN_BLOCK)
    before = len(store.requests)
    for member in index["members"]:
        record = bytearray()
        reader.stream_to(record.extend, start=member["local_header_offset"],
                         end=member["record_end_offset"])
        assert record[:4] == b"PK\x03\x04"
        offset = member["data_offset"] - member["local_header_offset"]
        output = bytearray()
        inflater = MemberInflater(member["method"], member["size"], output.extend)
        inflater.feed(record[offset:offset + member["compressed_size"]])
        inflater.finish()
        assert zlib.crc32(output) == member["crc32"]
        assert "sha256:" + hashlib.sha256(output).hexdigest() == member["sha256"]
    reopened = store.requests[before:]
    assert [row["range"] for row in reopened] == [
        (row["local_header_offset"], row["record_end_offset"] - 1) for row in index["members"]]
    assert all(row["if_match"] == ETAG for row in reopened)



def test_index_document_validates_and_forgery_is_refused():
    index, _ = _index(_mixed_archive())
    assert validate_member_index(index) is index

    def forged(change):
        document = copy.deepcopy(index)
        change(document)
        document["index_digest"] = canonical_digest(document, digest_field="index_digest")
        return document

    stale = copy.deepcopy(index)
    stale["members"][1]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ProviderOutputMemberIndexError, match="^provider_output_member_index_digest_mismatch$"):
        validate_member_index(stale)
    for change in (
        lambda d: d["members"][1].update(data_offset=d["members"][1]["data_offset"] + 1),
        lambda d: d["members"][2].update(local_header_offset=d["members"][2]["local_header_offset"] + 1),
        lambda d: d["members"][1].update(path="../escape"),
        lambda d: d["members"][2].update(path=d["members"][1]["path"].upper()),
        lambda d: d["members"][1].update(disposition="materialized"),
        lambda d: d.update(private_url_recorded=True),
        lambda d: d["directory"].update(offset=d["directory"]["offset"] + 1),
        lambda d: d["archive"].update(etag=7),
        lambda d: d["archive"].update(durable_reference="s3://bucket/key"),
        # A regular file that is also another member's parent directory.
        lambda d: d["members"][2].update(path=d["members"][1]["path"] + "/nested.json"),
    ):
        with pytest.raises(ProviderOutputMemberIndexError, match="^provider_output_member_index_invalid$"):
            validate_member_index(forged(change))
    # A document that is not JSON is refused with a code, never a TypeError.
    for unserializable in ({**index, "members": [object()]}, {**index, "limits": {1: 2, "a": 3}}):
        with pytest.raises(ProviderOutputMemberIndexError, match="^provider_output_member_index_invalid$"):
            validate_member_index(unserializable)



def test_selection_is_a_versioned_document_bound_to_one_index():
    index, _ = _index(_mixed_archive())
    wanted = ["runtime/result.json", "logs/worker.log", "runtime/frames/0001.png"]
    selection = build_member_selection(index, reversed(wanted), selection_version="consumers.v1")
    assert selection == {
        "schema_version": "provider_output_member_selection.v1",
        "member_index_digest": index["index_digest"],
        "selection_version": "consumers.v1",
        "members": sorted(wanted),
        "selection_digest": canonical_digest(selection, digest_field="selection_digest"),
    }
    assert [row["path"] for row in validate_member_selection(selection, index)] == [
        row["path"] for row in index["members"] if row["path"] in wanted]
    with pytest.raises(ProviderOutputMemberIndexError, match="^provider_output_member_selection_index_mismatch$"):
        validate_member_selection(selection, {**index, "index_digest": "sha256:" + "1" * 64})
    for paths in (["runtime/missing.json"], ["runtime"]):  # unknown, and a directory
        with pytest.raises(ProviderOutputMemberIndexError, match="^provider_output_member_selection_member_unknown$"):
            build_member_selection(index, paths, selection_version="consumers.v1")

    def redigested(**changes):
        document = {**selection, **changes}
        document["selection_digest"] = canonical_digest(document, digest_field="selection_digest")
        return document

    for document in ({**selection, "members": sorted(wanted)[:1]},
                     redigested(members=sorted(wanted, reverse=True)),
                     redigested(selection_version="has space"),
                     redigested(disposition={"runtime/result.json": "materialized"}),
                     # Not JSON: refused with a code, never a TypeError.
                     {**selection, "members": [b"runtime/result.json"]},
                     {**selection, "members": [object()]}):
        with pytest.raises(ProviderOutputMemberIndexError, match="^provider_output_member_selection_invalid$"):
            validate_member_selection(document, index)

def _entries(*names):
    return [Entry(name, name.encode() * 3) for name in names]


def _unicode_path_extra(raw_name, override):
    body = struct.pack("<BL", 1, zlib.crc32(raw_name)) + override
    return struct.pack("<HH", 0x7075, len(body)) + body


_TEXT = b"provider diagnostic text " * 20
_REFUSALS = {
    "crc_mismatch": (lambda: build_zip([Entry("a.bin", b"payload", payload=b"pAyload")]), {},
                     "provider_output_archive_crc_mismatch"),
    "local_header_name_differs": (
        lambda: build_zip([Entry("a.bin", b"x", local_name=b"b.bin")]), {},
        "provider_output_archive_local_header_mismatch"),
    "descriptor_differs": (
        lambda: build_zip([Entry("a.bin", _TEXT, descriptor="signed", descriptor_crc=7)]), {},
        "provider_output_archive_descriptor_mismatch"),
    "gap_between_records": (
        lambda: build_zip([Entry("a.bin", b"x", gap_after=b"junk"), Entry("b.bin", b"y")]), {},
        "provider_output_archive_gap_invalid"),
    "gap_before_directory": (lambda: build_zip([Entry("a.bin", b"x", gap_after=b"junk")]), {},
                             "provider_output_archive_gap_invalid"),
    "gap_after_descriptor": (
        lambda: build_zip([Entry("a.bin", b"x", descriptor="signed", gap_after=b"\0" * 4),
                           Entry("b.bin", b"y")]), {},
        "provider_output_archive_gap_invalid"),
    "prepended_self_extractor": (
        lambda: build_zip(_entries("a.bin", "b.bin"), prepend=b"MZ stub", shift_offsets=False), {},
        "provider_output_archive_prepended_data"),
    "prepended_shifted_records": (
        lambda: build_zip(_entries("a.bin", "b.bin"), prepend=b"junk"), {},
        "provider_output_archive_prepended_data"),
    "overlapping_records": (
        lambda: build_zip([Entry("a.bin", _TEXT), Entry("b.bin", b"y")], central_offsets={1: 40}),
        {}, "provider_output_archive_records_overlap"),
    "shared_record_offset": (
        lambda: build_zip(_entries("a.bin", "b.bin"), central_offsets={1: 0}), {},
        "provider_output_archive_records_overlap"),
    "duplicate_name": (lambda: build_zip(_entries("a.txt", "a.txt")), {},
                       "provider_output_archive_duplicate_path"),
    "case_folded_duplicate": (lambda: build_zip(_entries("Runtime/A.TXT", "runtime/a.txt")), {},
                              "provider_output_archive_duplicate_path"),
    "normalization_duplicate": (lambda: build_zip(_entries("é.txt", "é.txt")), {},
                                "provider_output_archive_duplicate_path"),
    "parent_escape": (lambda: build_zip(_entries("../escape")), {}, "provider_output_archive_path_invalid"),
    "absolute": (lambda: build_zip(_entries("/escape")), {}, "provider_output_archive_path_invalid"),
    "backslash": (lambda: build_zip(_entries("a\\b")), {}, "provider_output_archive_path_invalid"),
    "drive_colon": (lambda: build_zip(_entries("C:drive")), {}, "provider_output_archive_path_invalid"),
    "empty_component": (lambda: build_zip(_entries("a//b")), {}, "provider_output_archive_path_invalid"),
    "dot_component": (lambda: build_zip(_entries("a/./b")), {}, "provider_output_archive_path_invalid"),
    "long_component": (lambda: build_zip(_entries("x" * 256)), {}, "provider_output_archive_path_invalid"),
    "control_character": (lambda: build_zip(_entries("a\tb")), {}, "provider_output_archive_path_invalid"),
    "nul_truncated_name": (lambda: build_zip([Entry(b"a\x00b.txt", b"x")]), {},
                           "provider_output_archive_path_invalid"),
    "unicode_path_override": (
        lambda: build_zip([Entry("a.txt", b"x", central_extra=_unicode_path_extra(b"a.txt", b"b.txt"))]),
        {}, "provider_output_archive_path_invalid"),
    "symlink": (lambda: build_zip([Entry("link", b"../../outside", mode=0o120777)]), {},
                "provider_output_archive_entry_type_invalid"),
    "file_directory_collision": (lambda: build_zip(_entries("a", "a/b")), {},
                                 "provider_output_archive_file_directory_collision"),
    "member_cap": (lambda: build_zip(_entries("a", "b", "c")), {"maximum_members": 2},
                   "provider_output_archive_member_cap_exceeded"),
    "expanded_cap": (lambda: build_zip([Entry("a.bin", _TEXT), Entry("b.bin", _TEXT)]),
                     {"maximum_expanded_bytes": len(_TEXT) + 1},
                     "provider_output_archive_expansion_cap_exceeded"),
    "member_inflate_bound": (
        lambda: build_zip([Entry("a.bin", _TEXT, method=DEFLATED)]),
        {"maximum_member_inflated_bytes": len(_TEXT) - 1},
        "provider_output_archive_member_inflate_bound_exceeded"),
    "encrypted": (lambda: build_zip([Entry("a.bin", b"x", flags=0x1)]), {},
                  "provider_output_archive_encrypted"),
    "bzip2_method": (lambda: build_zip([Entry("a.bin", b"x", method=12)]), {},
                     "provider_output_archive_method_unsupported"),
    "deflate_ends_before_record": (
        lambda: build_zip([Entry("a.bin", _TEXT, method=DEFLATED, payload=deflate(_TEXT) + b"tail")]),
        {}, "provider_output_archive_deflate_end_invalid"),
    "deflate_runs_past_record": (
        lambda: build_zip([Entry("a.bin", _TEXT, method=DEFLATED, payload=deflate(_TEXT)[:-2])]),
        {}, "provider_output_archive_deflate_end_invalid"),
    "inflates_past_declared_size": (
        lambda: build_zip([Entry("a.bin", _TEXT, method=DEFLATED, size=len(_TEXT) // 2)]), {},
        "provider_output_archive_member_size_mismatch"),
    "not_a_zip": (lambda: b"this is not a zip archive at all" * 3, {},
                  "provider_output_archive_end_record_invalid"),
    # zipfile takes the last signature, a fake inside the comment, and sees no
    # members; the index must not see different members than zipfile does.
    "end_record_signature_in_comment": (
        lambda: build_zip([Entry("a.bin", _TEXT, method=DEFLATED)],
                          comment=b"note PK\x05\x06" + b"\0" * 18 + b" trailing"),
        {}, "provider_output_archive_end_record_ambiguous"),
    "bytes_after_end_record": (lambda: build_zip(_entries("a.bin")).to_bytes() + b"junk", {},
                               "provider_output_archive_end_record_invalid"),
}


@pytest.mark.parametrize("case", sorted(_REFUSALS))
def test_index_refuses_crc_mismatch_gaps_duplicates_unsafe_paths_and_caps(case):
    build, options, code = _REFUSALS[case]
    assert _refusal(build(), **options) == code



def _oversize_records(field, count=40):
    """Central records whose name, extra field or comment is about 64 KiB."""
    huge, entries = 0xFFFF - 16, []
    for position in range(count):
        if field == "name":  # cp437 0xB0 decodes to U+2591, so a decoded name doubles in memory
            entries.append(Entry(str(position).encode() + b"\xb0" * (huge - len(str(position))), b"x"))
        elif field == "extra":
            filler = struct.pack("<HH", 0x9999, huge - 4) + bytes(huge - 4)
            entries.append(Entry(f"m{position}.bin", b"x", central_extra=filler))
        else:
            entries.append(Entry(f"m{position}.bin", b"x", central_comment=b"c" * huge))
    return build_zip(entries)


@pytest.mark.parametrize("field, code", [
    ("name", "provider_output_archive_path_invalid"),
    ("extra", "provider_output_archive_directory_record_oversize"),
    ("comment", "provider_output_archive_directory_record_oversize"),
])
def test_oversize_directory_records_are_refused_before_they_are_buffered(field, code):
    archive, block = _oversize_records(field), 128 * 1024
    reader = RangeStore(archive).reader(block_bytes=block, maximum_archive_bytes=archive.size)
    tracemalloc.start()
    try:
        with pytest.raises(ProviderOutputMemberIndexError, match=f"^{code}$"):
            build_member_index(reader, maximum_expanded_bytes=EXPANDED)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= 2 * block, peak


def _maximal_name(position, astral):
    """A name at the entry rules' 4096-byte limit, in 255-byte (or shorter) components."""
    unit = "\U0001F600" * 63 if astral else "a" * 255
    name = "/".join([f"{position:05d}"] + [unit] * 16)
    while len(name.encode()) > 4096:
        name = name[:-1]
    return name


@pytest.mark.parametrize("astral", [False, True])
def test_directory_metadata_stays_within_the_documented_bound(astral):
    count, block = 256, 128 * 1024
    archive = build_zip([Entry(_maximal_name(position, astral), b"x") for position in range(count)])
    reader = RangeStore(archive).reader(block_bytes=block, maximum_archive_bytes=archive.size)
    tracemalloc.start()
    try:
        index = build_member_index(reader, maximum_expanded_bytes=EXPANDED)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert index["totals"]["members"] == count
    assert peak <= 2 * block + count * 16 * 1024, peak


def test_index_refuses_transport_blocks_below_128_kib():
    store = RangeStore(_mixed_archive())
    reader = store.reader(block_bytes=MIN_BLOCK - 1)
    with pytest.raises(ProviderOutputMemberIndexError, match="^provider_output_member_index_block_too_small$"):
        build_member_index(reader, maximum_expanded_bytes=EXPANDED)
    assert len(store.requests) == 1  # the reader's own probe; the index read nothing

def test_truncated_stream_is_refused():
    store = RangeStore(_mixed_archive())
    reader = store.reader(block_bytes=MIN_BLOCK)
    store.truncate_whole_object = True
    with pytest.raises(ProviderOutputMemberIndexError, match="^provider_output_archive_truncated$"):
        build_member_index(reader, maximum_expanded_bytes=EXPANDED)


def _republished(data):
    """The same members written again: identical sizes, different directory bytes."""
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as target:
            for info in source.infolist():
                info.date_time = (2030, 1, 1, 0, 0, 0)
                target.writestr(info, source.read(info))
    return buffer.getvalue()


@pytest.mark.parametrize("store_behaviour", ["honours_if_match", "ignores_if_match", "same_etag"])
def test_version_change_between_directory_and_stream_is_refused(store_behaviour):
    data = _mixed_archive()
    changed = _republished(data)
    assert len(changed) == len(data) and changed != data
    store = RangeStore(data)
    reader = store.reader(block_bytes=MIN_BLOCK)
    new_etag = ETAG if store_behaviour == "same_etag" else '"version2"'
    store.next_version = (RangeStore(changed).object, new_etag)
    store.ignore_if_match = store_behaviour == "ignores_if_match"

    with pytest.raises(ProviderOutputMemberIndexError) as caught:
        build_member_index(reader, maximum_expanded_bytes=EXPANDED)

    expected = ("provider_output_archive_directory_changed" if store_behaviour == "same_etag"
                else "provider_output_remote_version_changed")
    assert str(caught.value) == expected
    # A refusal, never a retry against the new object.
    assert store.whole_object_gets() == 1
    assert all(row["if_match"] == ETAG for row in store.requests[1:])


@pytest.mark.parametrize("block", [MIN_BLOCK, 1024**2])
def test_index_pass_writes_nothing_and_buffers_at_most_two_blocks(tmp_path, monkeypatch, block):
    unit = 1024**2
    archive = build_zip([
        Entry("checkpoints/weights.pt", Zeros(24 * unit)),
        Entry("runtime/field.npy", bytes(6 * unit), method=DEFLATED),
        Entry("runtime/result.json", b'{"status": "completed"}', method=DEFLATED),
    ])
    store = RangeStore(archive)
    reader = store.reader(block_bytes=block, maximum_archive_bytes=archive.size)
    monkeypatch.chdir(tmp_path)

    with no_disk_writes(monkeypatch) as attempts:
        tracemalloc.start()
        try:
            index = build_member_index(reader, maximum_expanded_bytes=EXPANDED)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

    assert attempts == [] and list(tmp_path.iterdir()) == []
    assert index["archive"]["size"] > 24 * unit
    assert peak <= 2 * block, peak
    assert index["totals"]["bytes"] == 30 * unit + len(b'{"status": "completed"}')


def test_data_descriptors_with_and_without_signature_are_indexed():
    files = {"runtime/": b"", "runtime/a.json": b'{"a": 1}' * 30, "runtime/b.bin": bytes(range(256)) * 9}
    signed = python_zip(files, streamed=True)
    with zipfile.ZipFile(io.BytesIO(signed)) as archive:
        assert all(info.flag_bits & 0x08 for info in archive.infolist())
    index, _ = _index(signed)
    _assert_matches_zipfile(index, signed)

    unsigned = build_zip([
        Entry("runtime/a.json", b'{"a": 1}' * 30, method=DEFLATED, descriptor="unsigned"),
        Entry("runtime/b.bin", bytes(range(256)) * 9, descriptor="unsigned"),
        Entry("runtime/c.bin", b"tail", descriptor="signed"),
    ]).to_bytes()
    index, _ = _index(unsigned, chunk_bytes=5)
    _assert_matches_zipfile(index, unsigned)
    descriptor_ends = [row["record_end_offset"] - row["data_offset"] - row["compressed_size"]
                       for row in index["members"]]
    assert descriptor_ends == [12, 12, 16]


def test_zip64_records_are_indexed():
    files = {"runtime/a.json": b'{"a": 1}' * 30, "runtime/b.bin": bytes(range(256)) * 9}
    streamed = python_zip(files, streamed=True, force_zip64=("runtime/b.bin",))
    seekable = python_zip(files, force_zip64=tuple(files))
    written = build_zip([
        Entry("runtime/a.json", b'{"a": 1}' * 30, method=DEFLATED, zip64=True),
        Entry("runtime/b.bin", bytes(range(256)) * 9, zip64=True, descriptor="unsigned"),
        Entry("runtime/c.bin", b"tail", zip64=True, descriptor="signed"),
    ], zip64_end=True).to_bytes()
    for data in (streamed, seekable, written):
        index, _ = _index(data)
        _assert_matches_zipfile(index, data)
    index, _ = _index(written)
    assert index["directory"]["zip64"] is True
    assert [row["record_end_offset"] - row["data_offset"] - row["compressed_size"]
            for row in index["members"]] == [0, 20, 24]


def test_virtual_multi_gigabyte_archive_is_indexed_without_allocating_its_bytes(monkeypatch):
    block = 8 * 1024**2
    archive = build_zip([
        Entry("runtime/result.json", b'{"status": "completed"}', method=DEFLATED),
        Entry("checkpoints/ckpt_000.pt", Zeros(ZERO_RUN)),
        Entry("runtime/after_checkpoint.json", b'{"late": true}'),
    ])
    store = RangeStore(archive)
    reader = store.reader(block_bytes=block, maximum_archive_bytes=archive.size,
                          deadline_seconds=900)

    with no_disk_writes(monkeypatch) as attempts:
        tracemalloc.start()
        try:
            index = build_member_index(reader, maximum_expanded_bytes=8 * 1024**3)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

    assert attempts == [] and peak <= 2 * block, peak
    members = {row["path"]: row for row in index["members"]}
    checkpoint = members["checkpoints/ckpt_000.pt"]
    assert checkpoint["size"] == checkpoint["compressed_size"] == ZERO_RUN
    assert checkpoint["sha256"] == ZERO_RUN_SHA256 and checkpoint["crc32"] == Zeros(ZERO_RUN).crc32
    late = members["runtime/after_checkpoint.json"]
    assert late["local_header_offset"] > 0xFFFFFFFF and index["directory"]["zip64"] is True
    with zipfile.ZipFile(VirtualFile(archive)) as independent:
        assert [(info.header_offset, info.file_size, info.CRC) for info in independent.infolist()] == [
            (row["local_header_offset"], row["size"], row["crc32"]) for row in index["members"]]
    assert index["totals"]["bytes_by_class"]["bulk"] == ZERO_RUN
    assert store.whole_object_gets() == 1


def test_bulk_extensions_match_the_retention_payload_list():
    from blueprint_pipeline.task_evaluation_terminal_scene_payload_retention import (
        _BINARY_PAYLOAD_SUFFIXES,
    )

    assert BULK_EXTENSIONS == frozenset(_BINARY_PAYLOAD_SUFFIXES)


def test_stream_to_is_one_pinned_get_and_archive_sha256_stays_a_thin_caller():
    data = _mixed_archive()
    store = RangeStore(data)
    reader = store.reader(block_bytes=1000)
    assert reader.archive_sha256() == "sha256:" + hashlib.sha256(data).hexdigest()
    assert store.requests[-1] == {"range": None, "if_match": ETAG}
    received = bytearray()
    assert reader.stream_to(received.extend, start=10, end=5000) == 4990
    assert bytes(received) == data[10:5000] and store.requests[-1] == {"range": (10, 4999), "if_match": ETAG}

    class SinkRefusal(ValueError):
        pass

    def refuse(chunk):
        raise SinkRefusal("sink_refused")

    with pytest.raises(SinkRefusal):
        reader.stream_to(refuse)
    for start, end in ((0, 0), (-1, 5), (5, len(data) + 1)):
        with pytest.raises(ProviderOutputTransportError, match="range_invalid"):
            reader.stream_to(received.extend, start=start, end=end)


def test_cli_prints_the_summary_with_bytes_by_class(tmp_path, capsys):
    path = tmp_path / "vast_provider_runtime_output.zip"
    path.write_bytes(_mixed_archive())

    assert main(["--archive", str(path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    index, _ = _index(path.read_bytes())
    assert summary["index_digest"] != index["index_digest"]  # a local file has no ETag
    assert summary["archive"]["sha256"] == index["archive"]["sha256"]
    assert summary["totals"] == index["totals"]
    assert summary["totals"]["bytes_by_class"]["bulk"] > 0

    path.write_bytes(b"not a zip" * 10)
    assert main(["--archive", str(path)]) == 1
    assert "provider_output_archive_end_record_invalid" in capsys.readouterr().err
