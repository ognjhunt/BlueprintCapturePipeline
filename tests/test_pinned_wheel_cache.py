"""Pinned PyPI wheels reach a provider bundle only as their exact bytes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import beta_robolab_runtime as runtime
from blueprint_pipeline import pinned_wheel_cache as cache
from blueprint_pipeline.control_plane_storage_roots import classify_path
from blueprint_pipeline.native_task_runtime_source_provision import _platform_tag_compatible

REPO = Path(__file__).resolve().parents[1]
PAYLOAD = b"exact wheel bytes"


def _row(payload: bytes = PAYLOAD) -> dict:
    name = "example_pkg-1.0-py3-none-any.whl"
    return {"wheel": name, "url": f"{cache.SOURCE_ORIGIN}/packages/aa/bb/{name}",
            "size_bytes": len(payload), "sha256": "sha256:" + hashlib.sha256(payload).hexdigest()}


class _Server:
    def __init__(self, payload: bytes = PAYLOAD):
        self.payload, self.calls = payload, []

    def __call__(self, url, *, output_path, expected_sha256, max_bytes):
        self.calls.append(url)
        Path(output_path).write_bytes(self.payload)


def test_miss_downloads_once_then_every_hit_is_rehashed(tmp_path):
    server = _Server()
    first = cache.fetch_pinned_wheel(_row(), destination_dir=tmp_path / "out", cache_root=tmp_path / "c",
                                     downloader=server)
    assert first == (tmp_path / "c" / _row()["sha256"].removeprefix("sha256:") / _row()["wheel"]).resolve()
    assert first.read_bytes() == PAYLOAD and len(server.calls) == 1
    assert cache.fetch_pinned_wheel(_row(), destination_dir=tmp_path / "out", cache_root=tmp_path / "c",
                                    downloader=server) == first
    assert len(server.calls) == 1
    first.write_bytes(b"damaged cache entry")
    assert cache.fetch_pinned_wheel(_row(), destination_dir=tmp_path / "out", cache_root=tmp_path / "c",
                                    downloader=server).read_bytes() == PAYLOAD
    assert len(server.calls) == 2


def test_other_bytes_or_hosts_are_refused(tmp_path):
    with pytest.raises(ValueError, match="identity_mismatch"):
        cache.fetch_pinned_wheel(_row(), destination_dir=tmp_path, cache_root=tmp_path / "c",
                                 downloader=_Server(b"other bytes, same len"))
    assert not list((tmp_path / "c").rglob("*.whl"))
    server = _Server()
    for changed in ({"url": "https://example.com/packages/" + _row()["wheel"]},
                    {"url": cache.SOURCE_ORIGIN + "/packages/other.whl"}, {"sha256": "abc"}, {"size_bytes": 0}):
        with pytest.raises(ValueError, match="identity_invalid"):
            cache.fetch_pinned_wheel({**_row(), **changed}, destination_dir=tmp_path, downloader=server)
    assert server.calls == []


def test_without_a_writable_cache_the_bytes_stay_in_the_build(tmp_path, monkeypatch):
    monkeypatch.delenv(cache.CACHE_ROOT_ENV)
    monkeypatch.setattr(cache, "DEFAULT_CACHE_PARENT", tmp_path / "absent")
    assert cache.resolve_cache_root() is None
    path = cache.fetch_pinned_wheel(_row(), destination_dir=tmp_path / "build", downloader=_Server())
    assert path == (tmp_path / "build" / _row()["wheel"]).resolve() and path.read_bytes() == PAYLOAD


def test_default_cache_is_a_classified_reproducible_cache():
    root = classify_path(str(cache.DEFAULT_CACHE_PARENT / cache.DEFAULT_CACHE_NAME / ("0" * 64) / "x.whl"))
    assert root is not None and root.storage_class == "cache"


def test_pinned_opencv_identity_is_the_approved_runtime_compatible_wheel(monkeypatch):
    monkeypatch.undo()  # the unpatched table, not the test stand-ins
    assert runtime.PYTHON_DEPENDENCY_WHEELS == (runtime.IMAGEIO_DEPENDENCY, runtime.OPENCV_DEPENDENCY)
    row = runtime.OPENCV_DEPENDENCY
    assert row["url"].startswith(cache.SOURCE_ORIGIN + "/") and row["url"].endswith("/" + row["wheel"])
    assert cache._identity(row)[1:] == (row["url"], 49_858_781, runtime.OPENCV_SHA256)
    # Rights: the exact name and version carry an unexpired owner approval.
    policy = json.loads((REPO / "docs/runtime_dependency_license_policy.json").read_text())
    approval = policy["components"][f"{row['distribution']}=={row['version']}"]
    assert approval["approved"] is True and approval["license_expression"] == row["license_spdx"]
    # CPython 3.12 (abi3 from cp37) on the Isaac Sim 6.0.1 image's glibc 2.35, x86_64.
    _name, _version, interpreter, abi, platforms = row["wheel"].removesuffix(".whl").split("-")
    assert (interpreter, abi) == ("cp37", "abi3")
    assert all(_platform_tag_compatible(tag, ("manylinux_2_35_x86_64",)) for tag in platforms.split("."))
    vendored = REPO / "src/blueprint_pipeline/_vendor" / runtime.IMAGEIO_WHEEL
    assert runtime.IMAGEIO_DEPENDENCY["size_bytes"] == vendored.stat().st_size
    assert runtime.IMAGEIO_DEPENDENCY["sha256"] == "sha256:" + hashlib.sha256(vendored.read_bytes()).hexdigest()
