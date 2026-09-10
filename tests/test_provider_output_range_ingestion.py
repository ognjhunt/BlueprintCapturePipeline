"""Real local HTTP ranges exercise bounded, resumable cloud ZIP collection."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import re
import stat
import threading
from types import SimpleNamespace
import urllib.request
import zipfile

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.provider_output_range_ingestion import FLOOR_BYTES, ingest_provider_output
from blueprint_pipeline.provider_output_range_transport import (
    ProviderOutputRangeReader,
    ProviderOutputTransportError,
)
from blueprint_pipeline.provider_signed_object_binding import signed_output_object_binding_sha256

URL = "https://storage.example.invalid/private/run.zip?signature=SECRET_DO_NOT_RECORD"


def _zip(extra=None, corrupt=False):
    data = b"generated diagnostic bytes"
    row = {
        "relative_path": "data.bin",
        "size_bytes": len(data),
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
    }
    identity = {
        "schema_version": "policy_canary_static_startup_preflight.v1",
        "run_id": "run-1",
        "runtime_inputs_digest": "sha256:" + "1" * 64,
        "status": "passed",
    }
    identity["result_digest"] = canonical_digest(identity, digest_field="result_digest")
    result = {
        "schema_version": "native_task_arena_policy_canary_session_result.v1",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "status": "blocked",
        "episodes": [],
        "artifact_inventory": [row],
        "artifact_inventory_digest": canonical_digest({"value": [row]}),
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    files = {
        "runtime/data.bin": data,
        "runtime/identity.json": json.dumps(identity).encode(),
        "runtime/result.json": json.dumps(result).encode(),
    }
    files.update(extra or {})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    value = buffer.getvalue()
    if corrupt:
        position = value.index(data)
        value = value[:position] + b"G" + value[position + 1 :]
    return value


@pytest.fixture
def store():
    state = SimpleNamespace(
        data=_zip(),
        etag='"version1"',
        ignore_ranges=False,
        bad_range=False,
        truncate=False,
        requests=[],
        changed=False,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            requested = self.headers.get("Range")
            state.requests.append({"range": requested, "if_match": self.headers.get("If-Match")})
            if self.headers.get("If-Match") not in (None, state.etag):
                self.send_response(412)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            data = state.data
            if requested and not state.ignore_ranges:
                match = re.fullmatch(r"bytes=(\d+)-(\d+)", requested)
                start, end = map(int, match.groups())
                payload = data[start : end + 1]
                self.send_response(206)
                self.send_header(
                    "Content-Range", f"bytes {start + int(state.bad_range)}-{end}/{len(data)}"
                )
            else:
                payload = data
                self.send_response(200)
            self.send_header("ETag", state.etag)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if state.truncate and len(payload) > 1:
                payload = payload[:-1]
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def opener(request, timeout, policy):
        local = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/object",
            headers=dict(request.header_items()),
            method=request.method,
        )
        response = urllib.request.urlopen(local, timeout=timeout)
        response.geturl = lambda: request.full_url
        return response

    state.opener = opener
    yield state
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def _binding(tmp_path, store, **overrides):
    url_file = tmp_path / "signed-get.txt"
    url_file.write_text(URL + "\n")
    url_file.chmod(0o600)
    manifest = {
        "schema_version": "wam_provider_object_store_staging.v1",
        "status": "completed",
        "blockers": [],
        "output_key_run_unique": True,
        "output_url_object_binding_sha256": signed_output_object_binding_sha256(URL, URL),
    }
    staging = tmp_path / "staging.json"
    staging.write_text(json.dumps(manifest))
    value = {
        "schema_version": "provider_output_ingestion_binding.v1",
        "run_id": "run-1",
        "instance_id": "50518293",
        "runtime_inputs_digest": "sha256:" + "1" * 64,
        "staging_manifest": {
            "path": str(staging),
            "sha256": "sha256:" + hashlib.sha256(staging.read_bytes()).hexdigest(),
            "size_bytes": staging.stat().st_size,
        },
        "signed_get_url_sha256": "sha256:" + hashlib.sha256(URL.encode()).hexdigest(),
        "maximum_archive_bytes": 20 * 1024**2,
        "maximum_extracted_bytes": 20 * 1024**2,
        "minimum_free_bytes": FLOOR_BYTES,
        "identity_document": "runtime/identity.json",
        "result_document": "runtime/result.json",
        "expected_archive_sha256": "sha256:" + hashlib.sha256(store.data).hexdigest(),
        **overrides,
    }
    value["binding_digest"] = canonical_digest(value, digest_field="binding_digest")
    return value, url_file


def _collect(tmp_path, store, **overrides):
    binding, url = _binding(tmp_path, store, **overrides)
    return ingest_provider_output(
        binding=binding,
        signed_get_url_file=url,
        output_root=tmp_path / "collected",
        opener=store.opener,
        disk_usage_provider=lambda p: SimpleNamespace(free=100 * 1024**3),
    )


def test_collect_and_resume_verified_members_without_local_zip_or_second_full_hash(tmp_path, store):
    binding, url = _binding(tmp_path, store)

    def run():
        return ingest_provider_output(
            binding=binding,
            signed_get_url_file=url,
            output_root=tmp_path / "collected",
            opener=store.opener,
            disk_usage_provider=lambda p: SimpleNamespace(free=100 * 1024**3),
        )

    first = run()
    assert (
        first["status"] == "collected_pending_finalization" and first["verified_member_count"] == 3
    )
    assert (
        first["scientific_qualification_performed"] is False
        and first["publication_performed"] is False
    )
    assert sum(row["range"] is None for row in store.requests) == 1
    assert not list((tmp_path / "collected").rglob("*.zip"))
    second = run()
    assert second["resumed_member_count"] == 3
    assert sum(row["range"] is None for row in store.requests) == 1
    for path in (tmp_path / "collected/.ingestion").iterdir():
        if path.is_file():
            assert b"SECRET_DO_NOT_RECORD" not in path.read_bytes()
    (tmp_path / "collected/native/runtime/data.bin").write_bytes(b"changed")
    assert run()["blockers"] == ["provider_output_resume_file_changed"]


@pytest.mark.parametrize("name", ["../escape", "/escape", "a\\b", "a/../b", "a//b", "C:drive"])
def test_zip_slip_refuses_before_extracting_any_native_file(tmp_path, store, name):
    store.data = _zip({name: b"bad"})
    result = _collect(tmp_path, store)
    assert result["status"] == "blocked"
    assert not list((tmp_path / "collected/native").rglob("*"))


def test_archive_symlink_refused(tmp_path, store):
    buffer = io.BytesIO(store.data)
    with zipfile.ZipFile(buffer, "a") as archive:
        entry = zipfile.ZipInfo("link")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(entry, "../../outside")
    store.data = buffer.getvalue()
    assert _collect(tmp_path, store)["blockers"] == ["provider_output_archive_entry_type_invalid"]


@pytest.mark.parametrize("fault", ["ignore_ranges", "bad_range", "truncate"])
def test_invalid_http_range_or_truncation_fails_closed(tmp_path, store, fault):
    setattr(store, fault, True)
    result = _collect(tmp_path, store)
    assert result["status"] == "blocked"
    assert not list((tmp_path / "collected/native").rglob("*"))


def test_crc_and_wrong_archive_digest_are_rejected(tmp_path, store):
    store.data = _zip(corrupt=True)
    result = _collect(tmp_path, store)
    assert result["blockers"] == ["provider_output_archive_crc_or_structure_invalid"]
    assert result["partial_evidence_retained"] is True
    other = tmp_path / "other"
    other.mkdir()
    store.data = _zip()
    result = _collect(other, store, expected_archive_sha256="sha256:" + "0" * 64)
    assert result["blockers"] == ["provider_output_archive_digest_mismatch"]
    assert not list((other / "collected/native").rglob("*"))


def test_version_change_between_requests_and_resume_is_rejected(tmp_path, store):
    reader = ProviderOutputRangeReader(
        URL, maximum_archive_bytes=100000, block_bytes=64, opener=store.opener
    )
    store.etag = '"version2"'
    with pytest.raises(ProviderOutputTransportError, match="remote_version_changed"):
        reader.read(20)
    store.etag = '"version1"'
    binding, url = _binding(tmp_path, store)
    kwargs = dict(
        binding=binding,
        signed_get_url_file=url,
        output_root=tmp_path / "collected",
        opener=store.opener,
        disk_usage_provider=lambda p: SimpleNamespace(free=100 * 1024**3),
    )
    assert ingest_provider_output(**kwargs)["status"] == "collected_pending_finalization"
    store.etag = '"version2"'
    assert ingest_provider_output(**kwargs)["blockers"] == [
        "provider_output_resume_remote_identity_mismatch"
    ]


def test_partial_resume_preserves_failure_and_appends_only_verified_prefix(tmp_path, store):
    store.data = _zip({"runtime/large.bin": b"x" * (3 * 1024**2)})
    binding, url = _binding(tmp_path, store)
    calls = 0

    def disk(path):
        nonlocal calls
        calls += 1
        return SimpleNamespace(free=0 if calls == 6 else 100 * 1024**3)

    kwargs = dict(
        binding=binding,
        signed_get_url_file=url,
        output_root=tmp_path / "collected",
        opener=store.opener,
    )
    first = ingest_provider_output(**kwargs, disk_usage_provider=disk)
    assert first["status"] == "blocked"
    partials = list((tmp_path / "collected/.ingestion").glob("*.partial"))
    assert len(partials) == 1 and 0 < partials[0].stat().st_size < 3 * 1024**2
    second = ingest_provider_output(
        **kwargs, disk_usage_provider=lambda p: SimpleNamespace(free=100 * 1024**3)
    )
    assert second["status"] == "collected_pending_finalization"
    assert (tmp_path / "collected/native/runtime/large.bin").read_bytes() == b"x" * (3 * 1024**2)
    assert (tmp_path / "collected/.ingestion/failures.jsonl").is_file()


def test_native_inventory_digest_mismatch_never_becomes_collection_success(tmp_path, store):
    store.data = _zip({"runtime/data.bin": b"changed authentic ZIP payload"})
    result = _collect(tmp_path, store)
    assert result["blockers"] == ["provider_output_native_artifact_digest_mismatch"]
    assert (tmp_path / "collected/native/runtime/data.bin").is_file()


def test_whole_archive_read_cannot_exceed_bounded_memory(store):
    reader = ProviderOutputRangeReader(
        URL,
        maximum_archive_bytes=100000,
        block_bytes=64,
        maximum_read_bytes=128,
        opener=store.opener,
    )
    with pytest.raises(ProviderOutputTransportError, match="memory_cap_exceeded"):
        reader.read()
