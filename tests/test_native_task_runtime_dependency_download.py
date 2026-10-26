"""Exact local HTTP bytes exercise concurrent native-runtime startup transport."""
import hashlib
import threading

import pytest

from blueprint_pipeline.native_task_runtime_dependency_download import (
    RuntimeDependencyDownloadError, download_runtime_dependency,
)
from blueprint_pipeline.provider_output_range_transport import ProviderOutputTransportError
from tests import test_provider_output_range_ingestion as output_transport_tests

URL = output_transport_tests.URL
store = output_transport_tests.store


def arguments(tmp_path, store):
    store.data = bytes(range(251)) * 4200
    return dict(url=URL, output_path=tmp_path / "runtime.zip.partial",
        expected_size=len(store.data), expected_sha256="sha256:" + hashlib.sha256(store.data).hexdigest(),
        workers=4, maximum_seconds=10, opener=store.opener)


def test_parallel_ranges_preserve_exact_bytes_and_object_version(tmp_path, store):
    kwargs = arguments(tmp_path, store)
    barrier = threading.Barrier(4)
    opener = store.opener
    thread_ids = set()
    lock = threading.Lock()

    def together(request, timeout, policy):
        if request.get_header("Range") != "bytes=0-0":
            with lock:
                thread_ids.add(threading.get_ident())
            barrier.wait(timeout=3)
        return opener(request, timeout, policy)

    result = download_runtime_dependency(**{**kwargs, "opener": together})
    assert result["status"] == "verified"
    assert result["parallel_ranges"] == len(thread_ids) == 4
    assert kwargs["output_path"].read_bytes() == store.data
    data_requests = [row for row in store.requests if row["range"] != "bytes=0-0"]
    assert len(data_requests) == 4
    assert all(row["if_match"] == store.etag for row in data_requests)
    assert not list(tmp_path.glob(".native-runtime-transfer-*"))


@pytest.mark.parametrize("fault", ["range", "truncate", "size", "digest", "version"])
def test_invalid_transfer_preserves_existing_partial_and_never_publishes(tmp_path, store, fault):
    kwargs = arguments(tmp_path, store)
    path = kwargs["output_path"]
    path.write_bytes(b"retained curl partial")
    if fault == "range":
        store.bad_range = True
    elif fault == "truncate":
        store.truncate = True
    elif fault == "size":
        kwargs["expected_size"] += 1
    elif fault == "digest":
        kwargs["expected_sha256"] = "sha256:" + "0" * 64
    else:
        opener = store.opener
        calls = 0
        lock = threading.Lock()

        def changed(request, timeout, policy):
            nonlocal calls
            with lock:
                calls += 1
                if calls == 2:
                    store.etag = '"version2"'
            return opener(request, timeout, policy)
        kwargs["opener"] = changed
    with pytest.raises((RuntimeDependencyDownloadError, ProviderOutputTransportError)) as raised:
        download_runtime_dependency(**kwargs)
    assert "SECRET_DO_NOT_RECORD" not in str(raised.value)
    assert path.read_bytes() == b"retained curl partial"
    assert not list(tmp_path.glob(".native-runtime-transfer-*"))


def test_transfer_refuses_symlink_output(tmp_path, store):
    kwargs = arguments(tmp_path, store)
    target = tmp_path / "retained.zip"
    target.write_bytes(b"keep")
    kwargs["output_path"].symlink_to(target)
    with pytest.raises(RuntimeDependencyDownloadError, match="path_invalid"):
        download_runtime_dependency(**kwargs)
    assert target.read_bytes() == b"keep"
    assert store.requests == []

